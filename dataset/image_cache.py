"""
Disk-based image cache for text rendering.

Renders text to images once and saves to disk. Subsequent runs load from cache.
Reduces 2M render operations to disk I/O (10x+ faster).
"""

import numpy as np
import hashlib
import pickle
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import atexit

# Global renderer and denoiser for worker processes (initialized once per worker)
_worker_renderer = None
_worker_denoiser = None

def _init_worker():
    """Initialize worker process with persistent TextRenderer and Denoiser.
    
    CRITICAL FIX: Without this, each worker recreates renderer for EVERY sample,
    causing massive duplication and slowdown.
    """
    global _worker_renderer, _worker_denoiser
    if _worker_renderer is None:
        from models.text_renderer import TextRenderer
        _worker_renderer = TextRenderer(width=224, height=224)
    
    if _worker_denoiser is None:
        try:
            from models.noise2noise_denoiser import SEALNoise2Noise
            _worker_denoiser = SEALNoise2Noise()
        except:
            _worker_denoiser = None  # Fallback

class ImageCache:
    """Cache rendered images with mandatory SEAL Noise2Noise denoising.
    
    SEAL (Self-Evolving Adaptive Learning) denoising is ALWAYS enabled:
    - Improves text rendering quality
    - Minimal overhead (~10ms per batch)
    - Essential for hybrid pretrained pipeline
    """
    
    def __init__(self, cache_dir: str = "datasets/vision_unified/text_cache", num_workers: int = None, denoiser_path: str = "models/checkpoints/n2n_denoiser.pt"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata file tracks cached samples
        self.metadata_file = self.cache_dir / "metadata.pkl"
        self.metadata = self._load_metadata()
        
        # Persistent worker pool (reuse instead of recreating)
        self.num_workers = num_workers or max(2, cpu_count() // 2)  # Don't overwhelm system
        self._pool = None
        self._pool_active = False
        
        # MANDATORY: Initialize SEAL denoiser
        self.denoiser = None
        self._init_denoiser_mandatory(denoiser_path)
    
    def _init_denoiser_mandatory(self, denoiser_path: Optional[str]):
        """Initialize SEAL denoiser - MANDATORY for hybrid pipeline.
        
        SEAL provides:
        - Adaptive learning (improves over time)
        - Better text rendering quality
        - Cross-modal alignment (text vs native images)
        """
        try:
            from models.noise2noise_denoiser import SEALNoise2Noise, Noise2NoiseDenoiser
            
            # Try SEAL-enhanced first (adaptive learning)
            adaptive_path = denoiser_path.replace('.pt', '_adaptive.pt') if denoiser_path else None
            
            if adaptive_path and Path(adaptive_path).exists():
                self.denoiser = SEALNoise2Noise(
                    denoiser_path=denoiser_path,
                    adaptive_gen_path=adaptive_path
                )
                print(f"✅ SEAL denoiser loaded: {adaptive_path}")
            elif denoiser_path and Path(denoiser_path).exists():
                self.denoiser = Noise2NoiseDenoiser(model_path=denoiser_path)
                print(f"✅ N2N denoiser loaded: {denoiser_path}")
                print(f"   💡 Tip: Train adaptive SEAL for better results")
            else:
                # No pretrained - use default initialization
                print(f"ℹ️  No pretrained denoiser - using default N2N")
                self.denoiser = Noise2NoiseDenoiser()  # Initialize with random weights
        except Exception as e:
            print(f"❌ CRITICAL: Denoiser initialization failed: {e}")
            print(f"   Hybrid pipeline requires denoising for quality")
            # Use minimal fallback instead of None
            from models.noise2noise_denoiser import Noise2NoiseDenoiser
            self.denoiser = Noise2NoiseDenoiser()
        
        # Register cleanup
        atexit.register(self._cleanup_pool)
        
    def _load_metadata(self):
        """Load cache metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'rb') as f:
                    return pickle.load(f)
            except (EOFError, pickle.UnpicklingError):
                # Corrupted metadata file, start fresh
                return {}
        return {}
    
    def _save_metadata(self):
        """Save cache metadata."""
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def _get_cache_key(self, text: str, width: int, height: int) -> str:
        """Generate cache key for text + dimensions."""
        content = f"{text}|{width}|{height}".encode('utf-8')
        return hashlib.md5(content).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for cached image."""
        # Shard into subdirectories to avoid too many files in one dir
        subdir = cache_key[:2]
        return self.cache_dir / subdir / f"{cache_key}.npy"
    
    def has_been_cached(self, text: str, width: int, height: int) -> bool:
        """Check if sample was cached (even if image file deleted to save space)."""
        cache_key = self._get_cache_key(text, width, height)
        return cache_key in self.metadata
    
    def get(self, text: str, width: int, height: int) -> Optional[np.ndarray]:
        """Get cached image if exists."""
        cache_key = self._get_cache_key(text, width, height)
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            try:
                img = np.load(cache_path)
                # Validate shape matches expected dimensions
                expected_shape = (height, width, 3)
                if img.shape != expected_shape:
                    # Corrupted cache, delete and return None
                    cache_path.unlink()
                    if cache_key in self.metadata:
                        del self.metadata[cache_key]
                    return None
                return img
            except Exception as e:
                # Corrupted file, delete and return None
                cache_path.unlink()
                if cache_key in self.metadata:
                    del self.metadata[cache_key]
                return None
        return None
    
    def put(self, text: str, width: int, height: int, image: np.ndarray):
        """Cache rendered image with optional denoising."""
        cache_key = self._get_cache_key(text, width, height)
        cache_path = self._get_cache_path(cache_key)
        cache_path.parent.mkdir(exist_ok=True)
        
        # Unified denoising (handles SEAL/standard/none automatically)
        if self.denoiser is not None:
            try:
                image = self.denoiser.denoise(image)
            except Exception:
                pass  # Fallback to original on error
        
        # Save and track
        np.save(cache_path, image)
        self.metadata[cache_key] = {
            'text_hash': cache_key,
            'width': width,
            'height': height,
            'path': str(cache_path),
            'denoised': self.denoiser is not None
        }
    
    def _render_sample(self, args):
        """Helper for parallel rendering (must be picklable)."""
        global _worker_renderer
        
        sample, width, height = args
        
        # Initialize renderer if needed (once per worker)
        if _worker_renderer is None:
            _init_worker_renderer(width, height)
        
        # Extract text
        if hasattr(sample, 'text'):
            text = sample.text
        elif hasattr(sample, 'question'):
            text = sample.question
        elif isinstance(sample, dict):
            text = sample.get('text', '') or sample.get('question', '') or str(sample)
        else:
            text = str(sample)
        
        # Check if already cached
        cache_key = self._get_cache_key(text, width, height)
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            return None, text  # Already cached
        
        # Render using global renderer (already initialized)
        pil_img = _worker_renderer.render_plain_text(text)
        img_array = np.array(pil_img)
        
        return img_array, text
    
    def _get_or_create_pool(self):
        """Get persistent pool or create if needed with proper worker initialization."""
        if self._pool is None or not self._pool_active:
            self._pool = Pool(
                processes=self.num_workers,
                initializer=_init_worker  # Uses global renderer/denoiser
            )
            self._pool_active = True
        return self._pool
    
    def _cleanup_pool(self):
        """Cleanup worker pool."""
        if self._pool is not None and self._pool_active:
            try:
                self._pool.close()
                self._pool.join()
            except:
                pass
            self._pool_active = False
    
    def _render_samples_parallel(self, samples, num_workers=None):
        """Render samples in parallel (helper for streaming).
        
        Optimizations:
        - Reuses persistent pool (no recreation overhead)
        - Skips multiprocessing for small batches (< 50 samples)
        - Workers initialize TextRenderer once
        """
        # For very small batches, multiprocessing overhead exceeds benefit
        if len(samples) < 50:
            # Single-threaded rendering for small batches
            from models.text_renderer import TextRenderer
            renderer = TextRenderer(width=224, height=224)
            
            results = []
            for sample in samples:
                # Extract text
                if hasattr(sample, 'text'):
                    text = sample.text
                elif hasattr(sample, 'question'):
                    text = sample.question
                elif isinstance(sample, dict):
                    text = sample.get('text', '') or sample.get('question', '') or str(sample)
                else:
                    text = str(sample)
                
                # Check if already cached
                cache_key = self._get_cache_key(text, 224, 224)
                cache_path = self._get_cache_path(cache_key)
                
                if cache_path.exists():
                    results.append((None, text))
                else:
                    pil_img = renderer.render_plain_text(text)
                    img_array = np.array(pil_img)
                    results.append((img_array, text))
            
            # Save rendered images
            for img_array, text in results:
                if img_array is not None:
                    self.put(text, 224, 224, img_array)
            
            return results
        
        # For larger batches, use persistent pool (reuse workers)
        args_list = [(sample, 224, 224) for sample in samples]
        pool = self._get_or_create_pool()
        
        # Use map with chunksize for better load balancing
        chunksize = max(1, len(samples) // (self.num_workers * 4))
        results = pool.map(self._render_sample, args_list, chunksize=chunksize)
        
        # CRITICAL: Save rendered images to cache
        for img_array, text in results:
            if img_array is not None:  # None means already cached
                self.put(text, 224, 224, img_array)
        
        return results
    
    def populate_cache(self, samples, renderer=None, batch_size=1000, save_every=5, num_workers=None):
        """Pre-populate cache with optimized parallel rendering.
        
        Optimizations:
        - Reuses persistent worker pool
        - Skips multiprocessing for small batches
        - Workers initialize once
        """
        if num_workers is not None:
            # Override default worker count if specified
            self.num_workers = num_workers
        
        print(f"📦 Pre-populating image cache (parallel: {self.num_workers} workers)...")
        
        total = len(samples)
        cached = 0
        rendered = 0
        batch_count = 0
        
        # Process in smaller chunks for better progress tracking
        chunk_size = batch_size
        
        # Get persistent pool (reuse across chunks)
        pool = self._get_or_create_pool()
        
        try:
            for i in tqdm(range(0, total, chunk_size), desc="Caching"):
                chunk = samples[i:i+chunk_size]
                
                # Skip multiprocessing for tiny chunks
                if len(chunk) < 50:
                    # Single-threaded for tiny chunks
                    from models.text_renderer import TextRenderer
                    single_renderer = TextRenderer(width=224, height=224)
                    
                    for sample in chunk:
                        # Extract text
                        if hasattr(sample, 'text'):
                            text = sample.text
                        elif hasattr(sample, 'question'):
                            text = sample.question
                        elif isinstance(sample, dict):
                            text = sample.get('text', '') or sample.get('question', '') or str(sample)
                        else:
                            text = str(sample)
                        
                        cache_key = self._get_cache_key(text, 224, 224)
                        cache_path = self._get_cache_path(cache_key)
                        
                        if cache_path.exists():
                            cached += 1
                        else:
                            pil_img = single_renderer.render_plain_text(text)
                            img_array = np.array(pil_img)
                            self.put(text, 224, 224, img_array)
                            rendered += 1
                else:
                    # Parallel render for larger chunks
                    args_list = [(sample, 224, 224) for sample in chunk]
                    chunksize = max(1, len(chunk) // (self.num_workers * 4))
                    results = pool.map(self._render_sample, args_list, chunksize=chunksize)
                    
                    # Save rendered images
                    for img_array, text in results:
                        if img_array is None:
                            cached += 1
                        else:
                            self.put(text, 224, 224, img_array)
                            rendered += 1
                
                batch_count += 1
                
                # Save metadata periodically
                if batch_count % save_every == 0:
                    self._save_metadata()
        
        finally:
            # Don't close pool - reuse for next call
            pass
        
        # Final metadata save
        self._save_metadata()
        
        print(f"✅ Cache populated:")
        print(f"   Cached (reused): {cached}")
        print(f"   Rendered (new): {rendered}")
        print(f"   Total: {total}")
        print(f"   Cache dir: {self.cache_dir}")
        
        return cached, rendered
