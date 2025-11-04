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
        self.new_batches_start_idx = 0  # Index where NEW batches start (after resume)
        self.consolidation_pause = threading.Event()  # Pause threads during consolidation
        self.consolidation_pause.set()  # Start unpaused
        self.threads_paused_count = 0  # Track how many threads are paused
        self.threads_paused_lock = threading.Lock()  # Lock for paused counter
        
    def producer_thread(self, samples, renderer, start_threshold=50000):
        """
        CPU thread: Render samples and add to cache.
        Signals GPU thread once threshold is reached.
        """
        print(f"📦 Producer: Rendering samples (parallel CPUs)...")
        
        from dataset.image_cache import ImageCache
        
        # Render in batches (skip already cached)
        # Small batches = frequent pause checks (responds to consolidation faster)
        batch_size = 100
        total = len(samples)
        cached_count = 0
        skipped_count = 0
        
        # Progress bar for caching
        pbar = tqdm(total=total, desc="📦 Caching", unit="samples")
        
        for i in range(0, total, batch_size):
            # Always check pause state (returns immediately if not paused)
            # Prevents race condition where pause happens between check and processing
            if not self.consolidation_pause.is_set():
                with self.threads_paused_lock:
                    self.threads_paused_count += 1
                try:
                    self.consolidation_pause.wait()
                finally:
                    with self.threads_paused_lock:
                        self.threads_paused_count -= 1
            
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
                
                # Check if cache file actually exists (not just metadata)
                # After consolidation, images are deleted but needed for remaining batches
                cached_img = self.cache.get(text, 224, 224)
                if cached_img is None:
                    uncached.append(sample)
                else:
                    skipped_count += 1
            
            # Only render uncached samples
            if uncached:
                self.cache._render_samples_parallel(uncached, num_workers=10)
                cached_count += len(uncached)
                # Parallel rendering handles disk writes - no verification needed
            
            # Update progress bar
            pbar.update(len(batch))
            pbar.set_postfix({
                'cached': cached_count,
                'skipped': skipped_count
            })
            
            # Update cached count for consumer
            processed_so_far = i + len(batch)
            with self.lock:
                self.cached_count = processed_so_far
                # Signal GPU to start after threshold
                if processed_so_far >= start_threshold and not self.cache_threshold_reached.is_set():
                    pbar.write(f"✅ GPU starting after {processed_so_far} samples cached")
                    self.cache_threshold_reached.set()
            
            # Periodic metadata save
            if (i // batch_size) % 5 == 0:
                self.cache._save_metadata()
        
        # Signal completion
        pbar.close()
        self.cache_complete.set()
        self.ready_queue.put(None)  # Sentinel to stop consumer
        print(f"✅ Caching complete! {total} samples processed")
        print(f"   {skipped_count} already cached (skipped)")
        print(f"   {cached_count} newly rendered")
    
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
        
        # Fast-forward: find first missing batch to skip checking existing ones
        if self.checkpoint_dir:
            print(f"🔍 Fast-forwarding to first unencoded batch...")
            for check_batch in range(total_batches):
                batch_file = os.path.join(self.checkpoint_dir, f"batch_{check_batch:05d}.pt")
                if not os.path.exists(batch_file):
                    # Found first missing batch
                    batch_count = check_batch
                    # CRITICAL: sample_idx stays at 0 because samples array is already sliced
                    # The slicing happened in stream_build() before threading
                    sample_idx = 0
                    
                    # Register all existing batches before this point
                    for i in range(check_batch):
                        existing_file = os.path.join(self.checkpoint_dir, f"batch_{i:05d}.pt")
                        if os.path.exists(existing_file):
                            with self.lock:
                                self.batch_files.append(existing_file)
                    
                    # Mark where new batches start (for consolidation)
                    with self.lock:
                        self.new_batches_start_idx = check_batch
                    
                    if check_batch > 0:
                        print(f"⏩ Skipped {check_batch} existing batches (already in checkpoint files)")
                        print(f"▶️  Starting encoding from beginning of remaining {len(samples):,} samples")
                    break
        
        pbar = tqdm(total=total_batches, initial=batch_count, desc="GPU Encoding")
        
        with torch.no_grad():
            while sample_idx < len(samples):
                # Always check pause state (returns immediately if not paused)
                # Prevents race condition where pause happens between check and processing
                if not self.consolidation_pause.is_set():
                    with self.threads_paused_lock:
                        self.threads_paused_count += 1
                    try:
                        self.consolidation_pause.wait()
                    finally:
                        with self.threads_paused_lock:
                            self.threads_paused_count -= 1
                
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
                missing_count = 0
                
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
                    else:
                        missing_count += 1
                
                # If batch is incomplete, wait a bit longer
                if missing_count > 0:
                    if not self.cache_complete.is_set():
                        pbar.write(f"⏳ Batch {batch_count}: {missing_count}/{len(batch_samples)} samples not cached yet, waiting...")
                        time.sleep(1)
                        continue  # Retry this batch
                    else:
                        pbar.write(f"⚠️  Batch {batch_count}: {missing_count} samples missing (cache complete)")
                
                if not batch_images:
                    # Skip this batch if no images available
                    sample_idx = batch_end
                    pbar.update(1)
                    continue
                
                # Setup batch file path and check if already exists
                batch_file = None
                if self.checkpoint_dir:
                    os.makedirs(self.checkpoint_dir, exist_ok=True)  # Ensure directory exists
                    batch_file = os.path.join(self.checkpoint_dir, f"batch_{batch_count:05d}.pt")
                    
                    if os.path.exists(batch_file):
                        # Batch already encoded, skip and register
                        with self.lock:
                            self.batch_files.append(batch_file)
                        sample_idx = batch_end
                        batch_count += 1
                        pbar.update(1)
                        pbar.write(f"⏭️  Batch {batch_count-1} already exists, skipping")
                        continue
                
                # Encode batch
                batch_images = torch.stack(batch_images).to(self.device)
                result = self.encoder(images=batch_images, return_children=True)
                
                # Save this batch immediately to disk (GPU -> Disk)
                if batch_file is not None:
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
                    pbar.write(f"💾 Saved {batch_count} batches to disk")
                
                # Consolidate every 100 batches to prevent disk overflow (critical for Colab)
                if batch_count % 100 == 0 and self.checkpoint_dir:
                    pbar.write(f"📤 Consolidating batches to free disk space...")
                    self._consolidate_to_drive(batch_count)
                
                # Move to next batch
                sample_idx = batch_end
                pbar.update(1)
        
        pbar.close()
        print(f"✅ Consumer: Encoding complete")
    
    def _consolidate_to_drive(self, up_to_batch):
        """Consolidate processed batches and clean up local files to free disk space."""
        import os
        import gc
        
        # Find batches we haven't consolidated yet (100 batches per chunk)
        chunk_size = 100
        start_idx = len(self.drive_checkpoints) * chunk_size
        end_idx = up_to_batch
        batch_subset = self.batch_files[start_idx:end_idx]
        
        if not batch_subset:
            return
        
        # Pause producer and consumer threads during consolidation
        self.consolidation_pause.clear()
        print(f"⏸️  Pausing encoding and caching...")
        
        # Wait for both threads to actually pause (no timeout - wait until done)
        import time
        while True:
            with self.threads_paused_lock:
                if self.threads_paused_count >= 2:
                    break
            time.sleep(0.1)
        
        print(f"📤 Consolidating batches {start_idx}-{end_idx}...")
        
        # Load and concatenate in smaller chunks to avoid OOM
        # Process 20 batches at a time instead of all 100+
        mini_chunk_size = 20
        all_consolidated = {'sketches': [], 'checksums': [], 'children': []}
        corrupted_files = []
        
        for mini_start in range(0, len(batch_subset), mini_chunk_size):
            mini_end = min(mini_start + mini_chunk_size, len(batch_subset))
            mini_batch_files = batch_subset[mini_start:mini_end]
            
            chunk_data = {'sketches': [], 'checksums': [], 'children': []}
            
            for batch_file in mini_batch_files:
                if os.path.exists(batch_file):
                    try:
                        batch = torch.load(batch_file, map_location='cpu')
                        for key in chunk_data:
                            if key in batch and batch[key] is not None:
                                chunk_data[key].append(batch[key])
                        del batch
                    except Exception as e:
                        corrupted_files.append(batch_file)
                        continue
            
            # Concatenate this mini chunk
            for key in chunk_data:
                if chunk_data[key]:
                    mini_concat = torch.cat(chunk_data[key], dim=0)
                    all_consolidated[key].append(mini_concat)
                    del mini_concat
            del chunk_data
            gc.collect()
        
        # Final concatenation
        consolidated = {}
        for key in all_consolidated:
            if all_consolidated[key]:
                consolidated[key] = torch.cat(all_consolidated[key], dim=0)
        del all_consolidated
        
        # Save to drive_dir if available, otherwise checkpoint_dir
        chunk_idx = len(self.drive_checkpoints)
        save_dir = self.drive_dir if self.drive_dir else self.checkpoint_dir
        consolidated_file = os.path.join(save_dir, f"consolidated_{chunk_idx:03d}.pt")
        
        torch.save(consolidated, consolidated_file)
        self.drive_checkpoints.append(consolidated_file)
        del consolidated
        gc.collect()
        
        chunk_size_mb = os.path.getsize(consolidated_file) / 1024 / 1024
        print(f"✅ Saved consolidated chunk {chunk_idx} ({chunk_size_mb:.1f}MB)")
        
        # Report corrupted files that were skipped
        if corrupted_files:
            print(f"⚠️  Skipped {len(corrupted_files)} corrupted batch files from previous session")
        
        # Delete individual batch files to free Colab disk space
        deleted_count = 0
        freed_mb = 0
        for batch_file in batch_subset:
            if os.path.exists(batch_file) and batch_file not in corrupted_files:
                batch_size_mb = os.path.getsize(batch_file) / 1024 / 1024
                os.remove(batch_file)
                deleted_count += 1
                freed_mb += batch_size_mb
        
        print(f"🧹 Deleted {deleted_count} individual batch files ({freed_mb:.1f}MB freed from Colab)")
        print(f"   Drive location: {self.drive_dir}")
        print(f"   Will sync to: C:\\Users\\sao\\Documents\\model-engine")
        
        # Resume producer and consumer threads
        self.consolidation_pause.set()
        print(f"▶️  Resuming encoding and caching...\n")
        
        # Note: Cache clearing is disabled to prevent deadlocks
        # Manually clear cache if disk space is critical: find datasets/vision_unified/text_cache -name "*.npy" -delete
    
    def _clear_cache_for_consolidated_samples(self, start_idx, end_idx):
        """Clear text cache images for consolidated samples but keep metadata.
        
        This deletes the .npy image files to save disk space but preserves metadata
        so the system knows which samples were already processed and won't re-render them.
        
        Args:
            start_idx: Starting sample index
            end_idx: Ending sample index
        """
        print(f"\n🧹 Clearing cached images for consolidated samples {start_idx:,}-{end_idx:,}...")
        
        cache_dir = self.cache.cache_dir
        
        if cache_dir.exists():
            # Delete only .npy image files, keep metadata
            cache_files = list(cache_dir.rglob('*.npy'))
            cache_size_gb = sum(f.stat().st_size for f in cache_files) / (1024**3)
            
            deleted_count = 0
            for cache_file in cache_files:
                try:
                    cache_file.unlink()  # Delete the image file
                    deleted_count += 1
                except Exception:
                    pass
            
            # Keep metadata intact - system knows what was cached
            # Won't try to re-render, saves time on resume
            
            print(f"✅ Deleted {deleted_count:,} cached images ({cache_size_gb:.2f}GB freed)")
            print(f"   Metadata preserved - won't re-render on resume")
    
    def _check_resume_state(self):
        """Check for existing batch/consolidated files and determine resume point.
        
        Returns:
            (samples_already_encoded, existing_batch_files, existing_consolidated_files)
        """
        import os
        from pathlib import Path
        
        checkpoint_dir = Path(self.checkpoint_dir)
        if not checkpoint_dir.exists():
            return 0, [], []
        
        # Check for consolidated files
        consolidated_files = sorted(checkpoint_dir.glob("consolidated_*.pt"))
        
        # Check for batch files
        batch_files = sorted(checkpoint_dir.glob("batch_*.pt"))
        
        # Calculate samples encoded
        samples_in_consolidated = len(consolidated_files) * 1000 * self.batch_size  # 1000 batches per consolidated
        samples_in_batches = len(batch_files) * self.batch_size
        total_encoded = samples_in_consolidated + samples_in_batches
        
        return total_encoded, batch_files, consolidated_files
    
    def stream_build(self, samples, renderer, start_threshold=50000, initial_cache_percent=0):
        """
        Main entry point: Start both threads and coordinate.
        Supports resume from existing batch/consolidated files.
        
        Args:
            samples: List of samples to process
            renderer: TextRenderer instance
            start_threshold: Start encoding after this many samples cached
        
        Returns:
            Encoded results dict
        """
        # Check for existing progress
        samples_encoded, existing_batches, existing_consolidated = self._check_resume_state()
        
        if samples_encoded > 0:
            print(f"\n🔄 RESUME MODE: Found existing progress")
            print(f"   Consolidated files: {len(existing_consolidated)} ({len(existing_consolidated) * 1000} batches)")
            print(f"   Batch files: {len(existing_batches)}")
            print(f"   Samples already encoded: {samples_encoded:,}/{len(samples):,} ({samples_encoded/len(samples)*100:.1f}%)")
            print(f"   Resuming from sample {samples_encoded}...\n")
            
            # Register existing files
            self.batch_files.extend(existing_batches)
            self.drive_checkpoints.extend(existing_consolidated)
            
            # Skip already processed samples
            samples = samples[samples_encoded:]
            
            if len(samples) == 0:
                print("✅ All samples already encoded!")
                # Load and return existing results
                return self._load_existing_results()
        
        # Adjust threshold based on initial cache
        if initial_cache_percent < 5:
            # If cache is empty/minimal, start GPU after just 1 batch
            actual_threshold = self.batch_size
            if samples_encoded == 0:
                print(f"\n⚡ No cache found - using fast start mode")
        else:
            actual_threshold = start_threshold
        
        print(f"\n{'='*70}")
        print(f"🌊 PARALLEL STREAMING PIPELINE")
        print(f"   CPU: Caching {len(samples)} samples (parallel workers)")
        print(f"   GPU: Starts after {actual_threshold} cached (batch={self.batch_size})")
        print(f"   Strategy: CPU + GPU run simultaneously")
        print(f"   Memory: Each batch saved directly to disk (minimal RAM)")
        print(f"   Batch files: {self.checkpoint_dir}")
        print(f"{'='*70}\n")
        
        # Start producer thread
        producer = threading.Thread(
            target=self.producer_thread,
            args=(samples, renderer, actual_threshold)
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
        
        if self.checkpoint_dir:
            remaining_batches = len(self.batch_files) - (len(self.drive_checkpoints) * 100)
            if remaining_batches > 0:
                print(f"\n📤 Consolidating final {remaining_batches} batches...")
                self._consolidate_to_drive(len(self.batch_files))
        
        return self._load_existing_results()
    
    def _load_existing_results(self):
        """Load results from existing consolidated files and remaining batches."""
        import os
        import gc
        
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
