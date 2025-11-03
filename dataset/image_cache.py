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
    
    def populate_cache(self, samples, renderer, batch_size=1000, save_every=5):
        """Pre-populate cache with all samples (saves every 5k samples = ~40 sec)."""
        print(f"📦 Pre-populating image cache...")
        
        total = len(samples)
        cached = 0
        rendered = 0
        batch_count = 0
        
        for i in tqdm(range(0, total, batch_size), desc="Caching"):
            batch = samples[i:i+batch_size]
            
            for sample in batch:
                # Handle both dict and Pydantic DataSample objects
                if hasattr(sample, 'text'):
                    text = sample.text
                elif hasattr(sample, 'question'):
                    text = sample.question
                elif isinstance(sample, dict):
                    text = sample.get('text', '') or sample.get('question', '') or str(sample)
                else:
                    text = str(sample)
                
                # Check cache
                cached_img = self.get(text, 224, 224)
                if cached_img is not None:
                    cached += 1
                else:
                    # Render and cache (PIL Image -> numpy array)
                    pil_img = renderer.render_plain_text(text)
                    img_array = np.array(pil_img)
                    self.put(text, 224, 224, img_array)
                    rendered += 1
            
            batch_count += 1
            
            # Save metadata periodically (every 5 batches = 5k samples = ~40 sec)
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
