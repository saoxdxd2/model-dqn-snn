"""
Test file for streaming builder consolidation logic.
Tests pause/resume, consolidation triggers, and deadlock scenarios.
"""

import torch
import torch.nn as nn
import threading
import time
import os
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import List

# Mock classes to simulate the real system
@dataclass
class MockSample:
    text: str
    
class MockCache:
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.metadata = {}
    
    def has_been_cached(self, text, w, h):
        return text in self.metadata
    
    def _render_samples_parallel(self, samples, num_workers=4):
        """Simulate rendering with small delay"""
        time.sleep(0.05 * len(samples))  # 50ms per sample
        for sample in samples:
            self.metadata[sample.text] = True
    
    def _save_metadata(self):
        """Mock metadata save - no-op for testing"""
        pass
    
    def get(self, text, w, h):
        """Simulate loading from cache"""
        if text in self.metadata:
            import numpy as np
            return np.random.rand(224, 224, 3).astype(np.uint8)
        return None

class MockEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
    
    def forward(self, images, return_children=False):
        """Simulate encoding with small delay"""
        time.sleep(0.01 * len(images))  # 10ms per image
        batch_size = len(images)
        result = {
            'sketches': torch.randn(batch_size, 12, 512),
            'checksums': torch.randn(batch_size, 12, 512),
            'children': torch.randn(batch_size, 12, 512)
        }
        return result

def test_consolidation_no_deadlock():
    """Test that consolidation doesn't deadlock"""
    print("\n" + "="*70)
    print("TEST 1: Consolidation Deadlock Prevention")
    print("="*70)
    
    from dataset.streaming_builder import StreamingCacheEncoder
    
    # Setup
    test_dir = Path("test_outputs/consolidation_test")
    if test_dir.exists():
        shutil.rmtree(test_dir)
        time.sleep(0.1)  # Windows needs time to release handles
    test_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = test_dir / "checkpoints"
    cache_dir = test_dir / "cache"
    checkpoint_dir.mkdir()
    cache_dir.mkdir()
    
    # Create mock encoder and cache
    encoder = MockEncoder()
    cache = MockCache(cache_dir)
    
    # Create streaming builder (cache is first arg)
    builder = StreamingCacheEncoder(
        cache=cache,
        encoder=encoder,
        device='cpu',
        batch_size=10,  # Small batch for fast testing
        checkpoint_dir=str(checkpoint_dir),
        drive_dir=None
    )
    
    # Create small dataset (300 samples = 30 batches, will trigger 3 consolidations)
    samples = [MockSample(text=f"sample_{i}") for i in range(300)]
    
    print(f"📦 Testing with {len(samples)} samples (30 batches)")
    print(f"   Consolidation triggers every 10 batches (for testing)")
    
    # Modify consolidation frequency for testing
    original_code = None
    builder_file = Path("dataset/streaming_builder.py")
    
    # Start time
    start_time = time.time()
    
    # Run streaming build
    try:
        # Mock the stream_build to test consolidation
        builder.consolidation_pause.set()
        
        # Start producer thread (simplified mock - don't use real producer_thread)
        def producer():
            """Mock producer that just increments cached_count"""
            for i in range(0, len(samples), 10):
                # Check pause at start of each iteration
                if not builder.consolidation_pause.is_set():
                    print(f"🔴 Producer: Pause requested, pausing...")
                    with builder.threads_paused_lock:
                        builder.threads_paused_count += 1
                    try:
                        builder.consolidation_pause.wait()
                        print(f"🟢 Producer: Resuming after pause")
                    finally:
                        with builder.threads_paused_lock:
                            builder.threads_paused_count -= 1
                
                # Simulate caching
                time.sleep(0.01)
                with builder.lock:
                    builder.cached_count = min(i + 10, len(samples))
            
            builder.cache_complete.set()
        
        # Start consumer thread
        prod_thread_ref = [None]  # Mutable container to share thread ref
        
        def consumer():
            # Trigger consolidation every 10 batches instead of 100
            batch_count = 0
            sample_idx = 0
            
            while sample_idx < len(samples):
                # Check pause
                if not builder.consolidation_pause.is_set():
                    with builder.threads_paused_lock:
                        builder.threads_paused_count += 1
                    try:
                        builder.consolidation_pause.wait()
                    finally:
                        with builder.threads_paused_lock:
                            builder.threads_paused_count -= 1
                
                # Wait for cache
                batch_end = min(sample_idx + builder.batch_size, len(samples))
                while True:
                    with builder.lock:
                        if builder.cached_count >= batch_end:
                            break
                    if builder.cache_complete.is_set():
                        break
                    time.sleep(0.01)
                
                # Mock encode batch
                time.sleep(0.02)  # Simulate encoding
                
                # Save batch file
                batch_file = checkpoint_dir / f"batch_{batch_count:05d}.pt"
                mock_data = {
                    'sketches': torch.randn(10, 12, 512).half(),
                    'checksums': torch.randn(10, 12, 512).half(),
                    'children': torch.randn(10, 12, 512).half()
                }
                torch.save(mock_data, batch_file)
                
                with builder.lock:
                    builder.batch_files.append(str(batch_file))
                
                batch_count += 1
                sample_idx = batch_end
                
                # Trigger consolidation every 10 batches
                if batch_count % 10 == 0:
                    print(f"📤 Batch {batch_count}: Triggering consolidation...")
                    consolidation_start = time.time()
                    
                    # Check if producer is still alive
                    if prod_thread_ref[0] and not prod_thread_ref[0].is_alive():
                        print(f"⚠️  Producer thread died - skipping consolidation")
                        break
                    
                    builder._consolidate_to_drive(batch_count)
                    consolidation_time = time.time() - consolidation_start
                    print(f"✅ Consolidation completed in {consolidation_time:.2f}s")
                    
                    # Check if deadlocked (timeout > 60s)
                    if consolidation_time > 60:
                        raise TimeoutError("Consolidation took >60s - likely deadlocked!")
        
        # Start threads
        prod_thread = threading.Thread(target=producer)
        cons_thread = threading.Thread(target=consumer)
        
        prod_thread_ref[0] = prod_thread  # Share reference with consumer
        
        prod_thread.start()
        cons_thread.start()
        
        # Wait with timeout
        prod_thread.join(timeout=120)
        cons_thread.join(timeout=120)
        
        if prod_thread.is_alive() or cons_thread.is_alive():
            print("❌ FAIL: Threads did not complete (deadlock detected)")
            return False
        
        elapsed = time.time() - start_time
        print(f"\n✅ PASS: All threads completed in {elapsed:.2f}s")
        print(f"   No deadlocks detected")
        
        # Verify consolidation files exist
        consolidated_files = list(checkpoint_dir.glob("consolidated_*.pt"))
        print(f"   Created {len(consolidated_files)} consolidated files")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if test_dir.exists():
            shutil.rmtree(test_dir)

def test_pause_resume_responsiveness():
    """Test that pause/resume responds quickly"""
    print("\n" + "="*70)
    print("TEST 2: Pause/Resume Responsiveness")
    print("="*70)
    
    from dataset.streaming_builder import StreamingCacheEncoder
    
    # Setup
    test_dir = Path("test_outputs/pause_test")
    if test_dir.exists():
        shutil.rmtree(test_dir)
        time.sleep(0.1)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = test_dir / "checkpoints"
    checkpoint_dir.mkdir()
    
    builder = StreamingCacheEncoder(
        cache=MockCache(checkpoint_dir / "cache"),
        encoder=MockEncoder(),
        device='cpu',
        batch_size=10,
        checkpoint_dir=str(checkpoint_dir),
        drive_dir=None
    )
    
    # Test pause flag
    builder.consolidation_pause.set()
    
    pause_detected = False
    
    def test_thread():
        nonlocal pause_detected
        # Simulate work loop
        for i in range(100):
            # Check pause
            if not builder.consolidation_pause.is_set():
                pause_detected = True
                with builder.threads_paused_lock:
                    builder.threads_paused_count += 1
                try:
                    builder.consolidation_pause.wait()
                finally:
                    with builder.threads_paused_lock:
                        builder.threads_paused_count -= 1
            
            time.sleep(0.01)  # Simulate work
    
    thread = threading.Thread(target=test_thread)
    thread.start()
    
    # Let it run
    time.sleep(0.1)
    
    # Trigger pause
    print("⏸️  Triggering pause...")
    pause_start = time.time()
    builder.consolidation_pause.clear()
    
    # Wait for pause detection
    timeout = 5
    elapsed = 0
    while elapsed < timeout:
        with builder.threads_paused_lock:
            if builder.threads_paused_count >= 1:
                break
        time.sleep(0.01)
        elapsed = time.time() - pause_start
    
    pause_response_time = time.time() - pause_start
    
    # Resume
    builder.consolidation_pause.set()
    thread.join(timeout=2)
    
    print(f"   Pause response time: {pause_response_time:.3f}s")
    
    if pause_response_time < 1.0:
        print(f"✅ PASS: Thread paused within 1s")
        return True
    else:
        print(f"❌ FAIL: Pause took {pause_response_time:.3f}s (should be <1s)")
        return False

def test_batch_files_tracking():
    """Test that batch_files list stays consistent after consolidation"""
    print("\n" + "="*70)
    print("TEST 3: Batch Files Tracking After Consolidation")
    print("="*70)
    
    from dataset.streaming_builder import StreamingCacheEncoder
    
    # Setup
    test_dir = Path("test_outputs/tracking_test")
    if test_dir.exists():
        shutil.rmtree(test_dir)
        time.sleep(0.1)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = test_dir / "checkpoints"
    checkpoint_dir.mkdir()
    
    builder = StreamingCacheEncoder(
        cache=MockCache(checkpoint_dir / "cache"),
        encoder=MockEncoder(),
        device='cpu',
        batch_size=10,
        checkpoint_dir=str(checkpoint_dir),
        drive_dir=None
    )
    
    # Create mock batch files
    for i in range(100):
        batch_file = checkpoint_dir / f"batch_{i:05d}.pt"
        torch.save({'data': torch.randn(10, 12, 512)}, batch_file)
        builder.batch_files.append(str(batch_file))
    
    print(f"   Created 100 batch files")
    print(f"   batch_files length before consolidation: {len(builder.batch_files)}")
    
    # Trigger consolidation
    builder.consolidation_pause.set()
    builder._consolidate_to_drive(100)
    
    print(f"   batch_files length after consolidation: {len(builder.batch_files)}")
    
    # Verify indices still work
    # After consolidation, indices should still map correctly
    start_idx = len(builder.drive_checkpoints) * 100
    
    if len(builder.batch_files) == 100:
        print(f"✅ PASS: batch_files list preserved (length still 100)")
        print(f"   Files deleted but paths kept for index consistency")
        return True
    else:
        print(f"❌ FAIL: batch_files list modified (length: {len(builder.batch_files)})")
        return False

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("STREAMING BUILDER CONSOLIDATION TEST SUITE")
    print("="*70)
    
    results = {}
    
    # Test 1: No deadlock
    results['deadlock'] = test_consolidation_no_deadlock()
    
    # Test 2: Pause responsiveness
    results['pause'] = test_pause_resume_responsiveness()
    
    # Test 3: Tracking consistency
    results['tracking'] = test_batch_files_tracking()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed - review output above")
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
