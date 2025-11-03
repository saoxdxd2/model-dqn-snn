"""
Streaming dataset builder that overlaps cache population with GPU encoding.

Key insight: Start encoding as soon as some samples are cached,
while CPU continues rendering remaining samples in parallel.
"""

import threading
import queue
import time
from pathlib import Path
from typing import List
import torch
from tqdm import tqdm


class StreamingCacheEncoder:
    """
    Overlap CPU rendering with GPU encoding for maximum throughput.
    
    Architecture:
    - Producer thread: CPU renders samples → saves to cache
    - Consumer thread: GPU encodes cached samples
    - Queue: Coordinates which batches are ready
    """
    
    def __init__(self, cache, encoder, device, batch_size=192):
        self.cache = cache
        self.encoder = encoder
        self.device = device
        self.batch_size = batch_size
        
        # Coordination
        self.ready_queue = queue.Queue(maxsize=10)  # Batches ready to encode
        self.stop_flag = threading.Event()
        self.cache_complete = threading.Event()
        
        # Results (consolidated periodically to save memory)
        self.encoded_results = {'sketches': [], 'checksums': [], 'children': []}
        self.temp_gpu_results = {'sketches': [], 'checksums': [], 'children': []}
        self.lock = threading.Lock()
        
    def producer_thread(self, samples, renderer, start_threshold=50000):
        """
        CPU thread: Render samples and add to cache.
        Signals GPU thread once threshold is reached.
        """
        print(f"📦 Producer: Rendering samples (parallel CPUs)...")
        
        from dataset.image_cache import ImageCache
        
        # Render in batches
        batch_size = 1000
        total = len(samples)
        cached_count = 0
        
        for i in range(0, total, batch_size):
            if self.stop_flag.is_set():
                break
            
            batch = samples[i:i+batch_size]
            
            # Render batch (uses multiprocessing internally)
            results = self.cache._render_samples_parallel(batch, num_workers=2)
            
            for img_array, text in results:
                if img_array is not None:
                    self.cache.put(text, 224, 224, img_array)
            
            cached_count = i + len(batch)
            
            # Signal GPU thread once threshold reached
            if cached_count >= start_threshold and not self.cache_complete.is_set():
                print(f"✅ Producer: {cached_count} samples cached, starting GPU encoding...")
                # Add initial batches to queue
                for batch_idx in range(0, cached_count, self.batch_size):
                    self.ready_queue.put(batch_idx)
            
            # Continue adding batches as we cache more
            elif cached_count > start_threshold:
                batch_idx = i
                self.ready_queue.put(batch_idx)
            
            # Periodic metadata save
            if (i // batch_size) % 5 == 0:
                self.cache._save_metadata()
        
        # Signal completion
        self.cache_complete.set()
        self.ready_queue.put(None)  # Sentinel to stop consumer
        print(f"✅ Producer: All {total} samples cached")
    
    def consumer_thread(self, samples):
        """
        GPU thread: Encode cached samples as they become available.
        """
        print(f"⏳ Consumer: Waiting for initial cache...")
        
        # Wait for first batch
        first_batch = self.ready_queue.get()
        if first_batch is None:
            return
        
        print(f"🚀 Consumer: Starting GPU encoding...")
        
        from torch.utils.data import Dataset, DataLoader
        
        class CachedDataset(Dataset):
            def __init__(self, samples, cache):
                self.samples = samples
                self.cache = cache
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                sample = self.samples[idx]
                
                # Extract text
                if hasattr(sample, 'text'):
                    text = sample.text
                elif hasattr(sample, 'question'):
                    text = sample.question
                elif isinstance(sample, dict):
                    text = sample.get('text', '') or sample.get('question', '')
                else:
                    text = str(sample)
                
                # Load from cache
                img = self.cache.get(text, 224, 224)
                return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        dataset = CachedDataset(samples, self.cache)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            prefetch_factor=1  # Reduce prefetch to save RAM
        )
        
        # Encode batches with periodic consolidation
        import gc
        batch_count = 0
        consolidate_every = 50  # Consolidate every 50 batches (192*50=9600 samples)
        
        with torch.no_grad():
            for batch_images in tqdm(dataloader, desc="GPU Encoding"):
                batch_images = batch_images.to(self.device)
                result = self.encoder(images=batch_images, return_children=True)
                
                # Accumulate on GPU first
                for key in self.temp_gpu_results:
                    if key in result and result[key] is not None:
                        self.temp_gpu_results[key].append(result[key])
                
                del batch_images, result
                batch_count += 1
                
                # Periodic consolidation (GPU -> CPU, free VRAM)
                if batch_count % consolidate_every == 0:
                    with self.lock:
                        for key in self.temp_gpu_results:
                            if self.temp_gpu_results[key]:
                                consolidated = torch.cat(self.temp_gpu_results[key], dim=0)
                                self.encoded_results[key].append(consolidated.half().cpu())
                                self.temp_gpu_results[key].clear()
                                del consolidated
                    
                    # Aggressive cleanup
                    gc.collect()
                    torch.cuda.empty_cache()
                
                # Light cleanup every 20 batches
                if batch_count % 20 == 0:
                    gc.collect()
        
        # Final consolidation
        with self.lock:
            for key in self.temp_gpu_results:
                if self.temp_gpu_results[key]:
                    consolidated = torch.cat(self.temp_gpu_results[key], dim=0)
                    self.encoded_results[key].append(consolidated.half().cpu())
                    self.temp_gpu_results[key].clear()
        
        print(f"✅ Consumer: Encoding complete")
    
    def stream_build(self, samples, renderer, start_threshold=50000):
        """
        Main entry point: Start both threads and coordinate.
        
        Args:
            samples: List of samples to process
            renderer: TextRenderer instance
            start_threshold: Start encoding after this many samples cached
        
        Returns:
            Encoded results dict
        """
        print(f"\n{'='*70}")
        print(f"🌊 STREAMING PIPELINE")
        print(f"   CPU: Rendering {len(samples)} samples")
        print(f"   GPU: Encoding starts after {start_threshold} cached")
        print(f"   Batch size: {self.batch_size} (large batches move data to GPU, reduce RAM)")
        print(f"   Memory: Consolidates every 50 batches (~9.6k samples to CPU)")
        print(f"   Strategy: Overlap for maximum throughput")
        print(f"{'='*70}\n")
        
        # Start producer thread
        producer = threading.Thread(
            target=self.producer_thread,
            args=(samples, renderer, start_threshold)
        )
        producer.start()
        
        # Start consumer thread
        consumer = threading.Thread(
            target=self.consumer_thread,
            args=(samples,)
        )
        consumer.start()
        
        # Wait for both to complete
        producer.join()
        consumer.join()
        
        # Consolidate results
        result = {}
        for key, chunks in self.encoded_results.items():
            if chunks:
                result[key] = torch.cat(chunks, dim=0)
            else:
                result[key] = torch.empty(0)
        
        return result
