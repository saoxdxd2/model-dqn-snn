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
    
    def __init__(self, cache, encoder, device, batch_size=256, checkpoint_dir=None, drive_dir=None):
        self.cache = cache
        self.encoder = encoder
        self.device = device
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.drive_dir = drive_dir  # Google Drive path for progress saves
        
        # Coordination
        self.ready_queue = queue.Queue(maxsize=10)  # Batches ready to encode
        self.stop_flag = threading.Event()
        self.cache_complete = threading.Event()
        
        # Direct-to-disk mode (no RAM accumulation)
        self.batch_files = []  # Track individual batch files
        self.drive_checkpoints = []  # Track Drive-saved chunks
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
            batch = samples[i:i+batch_size]
            self.cache._render_samples_parallel(batch, num_workers=2)
            
            cached_so_far = i + len(batch)
            
            # Progress update
            if cached_so_far % 50000 == 0:
                print(f"📦 Producer: {cached_so_far}/{len(samples)} samples cached...")
            
            # Periodic metadata save
            if (i // batch_size) % 5 == 0:
                self.cache._save_metadata()
        
        # Signal completion
        self.cache_complete.set()
        self.ready_queue.put(None)  # Sentinel to stop consumer
        print(f"✅ Producer: All {total} samples cached")
    
    def consumer_thread(self, samples):
        """
        GPU thread: Wait for ALL cache to complete, then encode batches.
        Saves each batch immediately to disk.
        """
        # Wait for producer to finish caching ALL samples
        print(f"⏳ Consumer: Waiting for cache completion...")
        self.cache_complete.wait()
        print(f"🚀 Consumer: Cache complete, starting GPU encoding...")
        
        # Custom dataset that loads from cache
        class CachedDataset(torch.utils.data.Dataset):
            def __init__(self, samples, cache):
                self.samples = samples
                self.cache = cache
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                sample = self.samples[idx]
                text = sample.get('text', '') or sample.get('question', '')
                
                # Load from cache (all samples should be cached now)
                img = self.cache.get(text, 224, 224)
                if img is None:
                    raise ValueError(f"Sample {idx} not in cache: {text[:50]}...")
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
        
        # Encode batches - save each directly to disk (no RAM accumulation)
        import gc
        import os
        batch_count = 0
        
        with torch.no_grad():
            for batch_images in tqdm(dataloader, desc="GPU Encoding"):
                batch_images = batch_images.to(self.device)
                result = self.encoder(images=batch_images, return_children=True)
                
                # Save this batch immediately to disk (GPU -> Disk)
                if self.checkpoint_dir:
                    batch_file = os.path.join(self.checkpoint_dir, f"batch_{batch_count:05d}.pt")
                    batch_data = {
                        'sketches': result['sketches'].half().cpu(),
                        'checksums': result['checksums'].half().cpu() if 'checksums' in result else None,
                        'children': result['children'].half().cpu() if 'children' in result else None
                    }
                    torch.save(batch_data, batch_file)
                    
                    with self.lock:
                        self.batch_files.append(batch_file)
                
                del batch_images, result, batch_data
                batch_count += 1
                
                # Cleanup every 10 batches
                if batch_count % 10 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                
                # Progress update every 500 batches
                if batch_count % 500 == 0:
                    print(f"💾 Saved {batch_count} batches to disk (RAM usage: minimal)")
                
                # Consolidate to Drive every 1000 batches (free disk space)
                if batch_count % 1000 == 0 and self.drive_dir:
                    self._consolidate_to_drive(batch_count)
        
        print(f"✅ Consumer: Encoding complete")
    
    def _consolidate_to_drive(self, up_to_batch):
        """Consolidate processed batches to Drive and clean up local files."""
        import os
        import gc
        
        # Find batches we haven't saved to Drive yet
        start_idx = len(self.drive_checkpoints) * 1000
        end_idx = up_to_batch
        batch_subset = self.batch_files[start_idx:end_idx]
        
        if not batch_subset:
            return
        
        print(f"📤 Consolidating batches {start_idx}-{end_idx} to Drive...")
        
        # Load and concatenate this chunk
        chunk_data = {'sketches': [], 'checksums': [], 'children': []}
        for batch_file in batch_subset:
            if os.path.exists(batch_file):
                batch = torch.load(batch_file, map_location='cpu')
                for key in chunk_data:
                    if key in batch and batch[key] is not None:
                        chunk_data[key].append(batch[key])
                del batch
        
        # Concatenate chunk
        consolidated = {}
        for key in chunk_data:
            if chunk_data[key]:
                consolidated[key] = torch.cat(chunk_data[key], dim=0)
        del chunk_data
        
        # Save to Drive (compressed)
        chunk_idx = len(self.drive_checkpoints)
        drive_file = os.path.join(self.drive_dir, f"encoding_chunk_{chunk_idx:03d}.pt")
        torch.save(consolidated, drive_file)
        
        # Delete previous Drive chunk (keep only latest)
        if self.drive_checkpoints:
            prev_chunk = self.drive_checkpoints[-1]
            if os.path.exists(prev_chunk):
                os.remove(prev_chunk)
                print(f"🧹 Deleted previous chunk from Drive (freed space)")
        
        self.drive_checkpoints.append(drive_file)
        del consolidated
        
        # Delete local batch files to free disk space
        for batch_file in batch_subset:
            if os.path.exists(batch_file):
                os.remove(batch_file)
        
        gc.collect()
        chunk_size_mb = os.path.getsize(drive_file) / 1024 / 1024
        print(f"✅ Saved chunk {chunk_idx} to Drive ({chunk_size_mb:.1f}MB), freed {len(batch_subset)} batch files")
    
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
        print(f"💾 CACHE-THEN-ENCODE PIPELINE")
        print(f"   Phase 1: Cache all {len(samples)} samples (CPU parallel)")
        print(f"   Phase 2: Encode all samples (GPU, batch={self.batch_size})")
        print(f"   Memory: Each batch saved directly to disk (minimal RAM)")
        print(f"   Batch files: {self.checkpoint_dir}")
        print(f"   Strategy: Sequential phases (no race conditions)")
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
        
        # Consolidate any remaining batches to Drive
        import os
        import gc
        
        if self.drive_dir:
            remaining_batches = len(self.batch_files) - (len(self.drive_checkpoints) * 1000)
            if remaining_batches > 0:
                print(f"\n📤 Consolidating final {remaining_batches} batches to Drive...")
                self._consolidate_to_drive(len(self.batch_files))
        
        # Load from Drive (only the final chunk remains - previous deleted)
        print(f"\n📚 Loading final chunk from Drive...")
        all_results = {'sketches': [], 'checksums': [], 'children': []}
        
        if self.drive_checkpoints:
            final_chunk = self.drive_checkpoints[-1]  # Only last chunk exists
            if os.path.exists(final_chunk):
                chunk = torch.load(final_chunk, map_location='cpu')
                for key in all_results:
                    if key in chunk:
                        all_results[key].append(chunk[key])
                del chunk
                gc.collect()
                print(f"  ✅ Loaded final chunk ({os.path.getsize(final_chunk)/1024/1024:.1f}MB)")
            else:
                print(f"  ⚠️  Warning: Final chunk not found at {final_chunk}")
        
        # Final concatenation
        print(f"⚙️  Final concatenation...")
        result = {}
        for key, chunks in all_results.items():
            if chunks:
                result[key] = torch.cat(chunks, dim=0).float()  # Convert back to FP32
            else:
                result[key] = torch.empty(0)
        
        # Cleanup any remaining batch files
        remaining_files = [f for f in self.batch_files if os.path.exists(f)]
        if remaining_files:
            print(f"🧹 Cleaning up {len(remaining_files)} remaining batch files...")
            for batch_file in remaining_files:
                os.remove(batch_file)
        
        # Clean up final Drive chunk
        if self.drive_checkpoints:
            final_chunk = self.drive_checkpoints[-1]
            if os.path.exists(final_chunk):
                os.remove(final_chunk)
                print(f"🧹 Cleaned up final Drive chunk")
        
        print(f"💾 Drive storage freed (all temp chunks deleted)")
        
        print(f"✅ Complete! Final shape: {result['sketches'].shape}")
        return result
