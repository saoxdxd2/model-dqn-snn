"""
Disk-based image cache for text rendering.

Renders text to images once and saves to disk. Subsequent runs load from cache.
Reduces 2M render operations to disk I/O (10x+ faster).
"""

import numpy as np
import hashlib
import pickle
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


class ImageCache:
    """Cache rendered images to disk."""
    
    def __init__(self, cache_dir: str = "datasets/vision_unified/text_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata file tracks cached samples
        self.metadata_file = self.cache_dir / "metadata.pkl"
        self.metadata = self._load_metadata()
        
    def _load_metadata(self):
        """Load cache metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'rb') as f:
                return pickle.load(f)
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
    
    def get(self, text: str, width: int, height: int) -> Optional[np.ndarray]:
        """Get cached image if exists."""
        cache_key = self._get_cache_key(text, width, height)
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            return np.load(cache_path)
        return None
    
    def put(self, text: str, width: int, height: int, image: np.ndarray):
        """Cache rendered image."""
        cache_key = self._get_cache_key(text, width, height)
        cache_path = self._get_cache_path(cache_key)
        
        # Create subdir if needed
        cache_path.parent.mkdir(exist_ok=True)
        
        # Save image
        np.save(cache_path, image)
        
        # Update metadata
        self.metadata[cache_key] = {
            'text_hash': cache_key,
            'width': width,
            'height': height,
            'path': str(cache_path)
        }
    
    def _render_sample(self, args):
        """Helper for parallel rendering (must be picklable)."""
        sample, width, height = args
        
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
        
        # Render (create renderer in worker process)
        from models.text_renderer import TextRenderer
        renderer = TextRenderer(width=width, height=height)
        pil_img = renderer.render_plain_text(text)
        img_array = np.array(pil_img)
        
        return img_array, text
    
    def _render_samples_parallel(self, samples, num_workers=2):
        """Render samples in parallel (helper for streaming)."""
        args_list = [(sample, 224, 224) for sample in samples]
        
        with Pool(num_workers) as pool:
            results = pool.map(self._render_sample, args_list)
        
        return results
    
    def populate_cache(self, samples, renderer=None, batch_size=1000, save_every=5, num_workers=None):
        """Pre-populate cache with parallel rendering (2-3x faster)."""
        if num_workers is None:
            num_workers = max(1, cpu_count() - 1)  # Leave 1 core free
        
        print(f"📦 Pre-populating image cache (parallel: {num_workers} workers)...")
        
        total = len(samples)
        cached = 0
        rendered = 0
        batch_count = 0
        
        # Process in smaller chunks with multiprocessing for better progress tracking
        chunk_size = batch_size  # Process 1000 samples at a time
        
        with Pool(num_workers) as pool:
            for i in tqdm(range(0, total, chunk_size), desc="Caching"):
                chunk = samples[i:i+chunk_size]
                args_list = [(sample, 224, 224) for sample in chunk]
                
                # Parallel render this chunk (blocks until done)
                results = pool.map(self._render_sample, args_list, chunksize=50)
                
                # Save rendered images
                for img_array, text in results:
                    if img_array is None:
                        cached += 1
                    else:
                        self.put(text, 224, 224, img_array)
                        rendered += 1
                
                batch_count += 1
                
                # Save metadata periodically (every 5k samples)
                if batch_count % save_every == 0:
                    self._save_metadata()
        
        # Final metadata save
        self._save_metadata()
        
        print(f"✅ Cache populated:")
        print(f"   Cached (reused): {cached}")
        print(f"   Rendered (new): {rendered}")
        print(f"   Total: {total}")
        print(f"   Cache dir: {self.cache_dir}")
        
        return cached, rendered
