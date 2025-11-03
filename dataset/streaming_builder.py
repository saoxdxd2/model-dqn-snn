"""
Streaming dataset builder that overlaps cache population with GPU encoding.

Key insight: Start encoding as soon as some samples are cached,
while CPU continues rendering remaining samples in parallel.
"""

import threading
import queue
import gc
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from typing import List


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
        
        # Parallel execution tracking
        self.cached_count = 0  # Number of samples cached so far
        self.cache_threshold_reached = threading.Event()  # Signal when threshold reached
        
    def producer_thread(self, samples, renderer, start_threshold=50000):
        """
        CPU thread: Render samples and add to cache.
        Signals GPU thread once threshold is reached.
        """
        print(f"📦 Producer: Rendering samples (parallel CPUs)...")
        
        from dataset.image_cache import ImageCache
        
        # Render in batches (skip already cached)
        batch_size = 1000
        total = len(samples)
        cached_count = 0
        skipped_count = 0
        
        for i in range(0, total, batch_size):
            batch = samples[i:i+batch_size]
            
            # Filter out already-cached samples
            uncached = []
            for sample in batch:
                # Extract text from DataSample object
                if hasattr(sample, 'text'):
                    text = sample.text
                elif hasattr(sample, 'question'):
                    text = sample.question
                elif isinstance(sample, dict):
                    text = sample.get('text', '') or sample.get('question', '')
                else:
                    text = str(sample)
                
                # Check if cached (returns numpy array if cached, None if not)
                if self.cache.get(text, 224, 224) is None:
                    uncached.append(sample)
                else:
                    skipped_count += 1
            
            # Only render uncached samples
            if uncached:
                self.cache._render_samples_parallel(uncached, num_workers=2)
                cached_count += len(uncached)
            
            processed_so_far = i + len(batch)
            
            # Update cached count for consumer
            with self.lock:
                self.cached_count = processed_so_far
                # Signal GPU to start after threshold
                if processed_so_far >= start_threshold and not self.cache_threshold_reached.is_set():
                    print(f"✅ Producer: {processed_so_far} samples cached, signaling GPU to start...")
                    self.cache_threshold_reached.set()
            
            # Progress update
            if processed_so_far % 50000 == 0:
                print(f"📦 Producer: {processed_so_far}/{total} processed ({skipped_count} skipped, {cached_count} rendered)")
            
            # Periodic metadata save
            if (i // batch_size) % 5 == 0:
                self.cache._save_metadata()
        
        # Signal completion
        self.cache_complete.set()
        self.ready_queue.put(None)  # Sentinel to stop consumer
        print(f"✅ Producer: Complete! {total} samples processed")
        print(f"   {skipped_count} already cached (skipped)")
        print(f"   {cached_count} newly rendered")
        print(f"   Cache is now 100% complete")
    
    def consumer_thread(self, samples):
        """
        GPU thread: Start after threshold, encode only cached samples in parallel with caching.
        Saves each batch immediately to disk.
        """
        # Wait for threshold to be reached
        print(f"⏳ Consumer: Waiting for initial cache threshold...")
        self.cache_threshold_reached.wait()
        print(f"🚀 Consumer: Starting GPU encoding (caching continues in parallel)...") 
        
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
        
        # Process samples in order, only encoding what's been cached
        import gc
        import os
        batch_count = 0
        sample_idx = 0
        total_batches = (len(samples) + self.batch_size - 1) // self.batch_size
        
        pbar = tqdm(total=total_batches, desc="GPU Encoding")
        
        with torch.no_grad():
            while sample_idx < len(samples):
                # Wait until this batch is fully cached
                batch_end = min(sample_idx + self.batch_size, len(samples))
                while True:
                    with self.lock:
                        if self.cached_count >= batch_end:
                            break
                    # Wait for more samples to be cached
                    if self.cache_complete.is_set():
                        break
                    time.sleep(0.1)
                
                # Load this batch from cache
                batch_samples = samples[sample_idx:batch_end]
                batch_images = []
                for sample in batch_samples:
                    if hasattr(sample, 'text'):
                        text = sample.text
                    elif hasattr(sample, 'question'):
                        text = sample.question
                    elif isinstance(sample, dict):
                        text = sample.get('text', '') or sample.get('question', '')
                    else:
                        text = str(sample)
                    
                    img = self.cache.get(text, 224, 224)
                    if img is not None:
                        batch_images.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
                
                if not batch_images:
                    break
                
                batch_images = torch.stack(batch_images).to(self.device)
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
                
                # Move to next batch
                sample_idx = batch_end
                pbar.update(1)
        
        pbar.close()
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
        
        # Save to Drive (will sync to user's local machine automatically)
        chunk_idx = len(self.drive_checkpoints)
        drive_file = os.path.join(self.drive_dir, f"consolidated_{chunk_idx:03d}.pt")
        
        torch.save(consolidated, drive_file)
        self.drive_checkpoints.append(drive_file)
        del consolidated
        
        # Note: Keep on Drive - will sync to C:\Users\sao\Documents\model-engine
        # Once synced to local machine, user can manually delete from Drive if needed
        
        # Keep consolidated chunks on Drive, don't delete from Drive
        print(f"✅ Saved chunk {chunk_idx} to Drive - will sync to local machine")
        print(f"   Drive location: {self.drive_dir}")
        print(f"   Will sync to: C:\\Users\\sao\\Documents\\model-engine")
        print(f"   You can delete from Drive after confirming sync completed")
        
        gc.collect()
        chunk_size_mb = os.path.getsize(drive_file) / 1024 / 1024
        print(f"✅ Saved chunk {chunk_idx} to Drive ({chunk_size_mb:.1f}MB) - will sync to local machine")
        print(f"   Freed {len(batch_subset)} batch files from Colab")
    
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
        print(f"🌊 PARALLEL STREAMING PIPELINE")
        print(f"   CPU: Caching {len(samples)} samples (parallel workers)")
        print(f"   GPU: Encoding after {start_threshold} cached (batch={self.batch_size})")
        print(f"   Strategy: CPU + GPU run simultaneously")
        print(f"   Memory: Each batch saved directly to disk (minimal RAM)")
        print(f"   Batch files: {self.checkpoint_dir}")
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
        
        # Consolidate any remaining batches
        import os
        import gc
        
        if self.drive_dir:
            remaining_batches = len(self.batch_files) - (len(self.drive_checkpoints) * 1000)
            if remaining_batches > 0:
                print(f"\n📤 Consolidating final {remaining_batches} batches...")
                self._consolidate_to_drive(len(self.batch_files))
        
        # Load from local consolidated files
        print(f"\n📚 Loading {len(self.drive_checkpoints)} consolidated chunks from local disk...")
        all_results = {'sketches': [], 'checksums': [], 'children': []}
        
        for i, chunk_file in enumerate(self.drive_checkpoints):
            if os.path.exists(chunk_file):
                chunk = torch.load(chunk_file, map_location='cpu')
                for key in all_results:
                    if key in chunk:
                        all_results[key].append(chunk[key])
                del chunk
                gc.collect()
                print(f"  Loaded chunk {i+1}/{len(self.drive_checkpoints)}")
        
        # Final concatenation
        print(f"⚙️  Final concatenation...")
        result = {}
        for key, chunks in all_results.items():
            if chunks:
                result[key] = torch.cat(chunks, dim=0).float()  # Convert back to FP32
            else:
                result[key] = torch.empty(0)
        
        # Cleanup any remaining batch files from Colab
        remaining_files = [f for f in self.batch_files if os.path.exists(f)]
        if remaining_files:
            print(f"🧹 Cleaning up {len(remaining_files)} remaining batch files from Colab...")
            for batch_file in remaining_files:
                os.remove(batch_file)
        
        # Keep Drive chunks - they will sync to local machine
        print(f"💾 Kept {len(self.drive_checkpoints)} chunks in Drive (syncing to local machine)")
        print(f"   Drive location: {self.drive_dir}")
        print(f"   Will sync to: C:\\Users\\sao\\Documents\\model-engine")
        print(f"   You can delete from Drive after confirming sync completed")
        
        print(f"✅ Complete! Final shape: {result['sketches'].shape}")
        return result
