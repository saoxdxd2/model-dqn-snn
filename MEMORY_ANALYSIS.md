# Streaming Pipeline Memory Analysis

## Overview
Total samples: 1,907,747 train samples
Batch size: 192
Total batches: 9,937

---

## 1. GPU VRAM Consumption (Target: 15GB T4)

### Model Parameters (Static)
```
TRMVisionEncoder:
- Patch embedding: 768 * 3 * 16 * 16 = 590K params
- Position embeddings: 196 * 768 = 150K params
- Transformer layers: 2 layers
  - Attention: 768 * 768 * 4 (QKV + proj) * 2 = 9.4M params
  - FFN: 768 * 3072 * 2 * 2 = 9.4M params
- Capsule pooling: 768 * 12 = 9K params
Total: ~14.9M params × 4 bytes (FP32) = 59.6MB

CapsuleEncoder overhead: ~20MB
Total model: ~80MB ✅ (much smaller than expected)
```

### Input Batch (Per Forward Pass)
```
Images: [192, 3, 224, 224] × 4 bytes = 55MB
Patches: [192, 196, 768] × 4 bytes = 113MB
```

### Intermediate Activations (Peak During Forward)
```
Attention:
- Q, K, V: 3 × [192, 8, 196, 96] × 4 = 172MB
- Attention scores: [192, 8, 196, 196] × 4 = 471MB
- Context: [192, 8, 196, 96] × 4 = 57MB

FFN:
- Hidden: [192, 196, 3072] × 4 = 450MB
- Output: [192, 196, 768] × 4 = 113MB

Recursive cycles: H=2, L=3
- Each L-cycle reuses buffers
- Peak per layer: ~1.2GB
- 2 layers sequential: ~1.2GB peak

Total activations: ~1.5GB
```

### Output Tensors (Accumulated on GPU before consolidation)
```
Per batch output:
- Sketches: [192, 12, 768] × 4 = 7MB
- Checksums: [192, 12, 32] × 4 = 0.3MB
- Children: [192, 12, 4, 768] × 4 = 28MB
Total per batch: ~35MB

Accumulated for 50 batches (before consolidation):
50 × 35MB = 1.75GB
```

### GPU VRAM Total
```
Model: 80MB
Input batch: 168MB
Activations (peak): 1,500MB
Accumulated outputs (50 batches): 1,750MB
CUDA overhead: 500MB
----------------------------------
Total: ~4.0GB ✅ (Well under 15GB limit!)
```

**Why GPU shows 0.2GB usage**: Model not loaded yet during cache population phase.
**Expected after encoding starts**: 4-6GB sustained, with spikes to 8GB during consolidation.

---

## 2. CPU RAM Consumption (Limit: 12GB Colab)

### DataLoader Workers (2 workers)
```
Each worker:
- Prefetch: 1 batch × [192, 3, 224, 224] × 1 byte (uint8) = 28MB
- Python overhead: ~50MB
- 2 workers: 2 × 78MB = 156MB
```

### Producer Thread (Text Rendering)
```
Multiprocessing Pool (2 workers):
- 1000 samples in-flight
- PIL images: 1000 × 224 × 224 × 3 = 150MB
- Process overhead: 2 × 100MB = 200MB
Total: ~350MB
```

### Accumulated Results (Before Checkpoint)
```
Consolidated from GPU every 50 batches, saved to disk every 200 batches.

In-memory for 200 batches:
- Sketches: [192 × 200, 12, 768] × 2 bytes (FP16) = 1,770MB
- Checksums: [192 × 200, 12, 32] × 2 = 59MB
- Children: [192 × 200, 12, 4, 768] × 2 = 7,077MB
Total: ~8.9GB ❌ (This was the OOM cause!)

After checkpoint every 200 batches:
Max in RAM: 200 batches worth = ~8.9GB
With checkpoint: Cleared to disk, max ~2GB ✅
```

### Python Interpreter & Libraries
```
Python: 300MB
PyTorch: 500MB
Other libraries: 200MB
Total: 1,000MB
```

### CPU RAM Total
```
WITHOUT checkpointing:
DataLoader: 156MB
Producer: 350MB
Accumulated results: 8,900MB (grows continuously)
Python/libs: 1,000MB
----------------------------------
Total: 10.4GB → grows to 12GB → OOM at ~200 batches ❌

WITH checkpointing (current):
DataLoader: 156MB
Producer: 350MB
Accumulated results: 2,000MB (capped at 200 batches)
Python/libs: 1,000MB
----------------------------------
Total: ~3.5GB peak ✅ (Safe margin under 12GB)
```

---

## 3. Disk Usage

### Text Image Cache (Persistent)
```
1,907,747 images × 224 × 224 × 3 × 1 byte = 286GB
Saved as .npy compressed: ~40GB
Location: datasets/vision_unified/text_cache/
```

### Streaming Checkpoints (Temporary)
```
Per checkpoint (200 batches = 38,400 samples):
- Sketches: [38400, 12, 768] × 2 = 710MB
- Checksums: [38400, 12, 32] × 2 = 29MB
- Children: [38400, 12, 4, 768] × 2 = 2,842MB
Total per checkpoint: ~3.6GB

Total checkpoints: 9937 / 200 = ~50 checkpoints
Peak disk usage: 50 × 3.6GB = 180GB
Location: datasets/vision_unified/stream_checkpoints/

Note: Checkpoints deleted after final consolidation
```

---

## 4. Performance Calculations

### Encoding Speed
```
Observed: 2.04s per batch (192 samples)
Throughput: 192 / 2.04 = 94 samples/sec

Why slower than expected?
- TRM recursive reasoning: H=2 × L=3 = 6 cycles per layer
- 2 layers = 12 forward passes per sample
- Memory-efficient SDPA on T4 (no FlashAttention)
- Batch size 192 ≈ 85% GPU utilization

Theoretical minimum with FlashAttention (Ampere GPU):
~1.0s per batch = 192 samples/sec (2x faster)
```

### Total Time Estimate
```
Cache population (one-time):
- CPU rendering: 1,907,747 / (1000 samples/batch × 2 workers) / 3s = ~50 min
- Saves every 5 batches: Negligible overhead

GPU encoding (streaming):
- Total batches: 9,937
- Time per batch: 2.04s
- Total: 9,937 × 2.04s = 20,272s = 5.6 hours

Consolidation overhead:
- Every 50 batches: ~2s × (9937/50) = 398s = 7 min
- Every 200 batches checkpoint: ~5s × (9937/200) = 248s = 4 min

Total first run: ~6 hours (cache + encode overlap)
Subsequent runs (cache complete): ~45 min (encode only with full cache)
```

---

## 5. Bottleneck Analysis

### Current Bottleneck: GPU Encoding
```
CPU rendering: 50 min (parallel 2 workers)
GPU encoding: 5.6 hours (TRM recursive passes)

Bottleneck: GPU encoding (6x slower than CPU rendering)
```

### Optimization Potential
```
1. FlashAttention (requires Ampere GPU):
   - 2.04s → 1.0s per batch
   - Total time: 2.8 hours (2x speedup)

2. Reduce recursive cycles:
   - H=2, L=3 → H=1, L=2
   - Forward passes: 12 → 4 (3x reduction)
   - Time: ~2 hours
   - Trade-off: Lower reasoning capability

3. Mixed precision (FP16 training):
   - Already using FP16 for storage
   - Could use FP16 for forward pass
   - ~1.5x speedup on Ampere, ~1.1x on Turing (T4)

4. Larger batch size:
   - Current: 192 (4GB VRAM used)
   - Max possible: ~384 (8GB VRAM)
   - Throughput: +30% (better GPU utilization)
   - Time: ~4.3 hours
```

---

## 6. Memory-Safe Configuration Summary

### Current Configuration (Stable)
```
GPU:
- Batch size: 192
- Peak VRAM: 4-6GB
- Utilization: 85%

CPU:
- Workers: 2
- Prefetch: 1
- Peak RAM: 3.5GB
- Checkpoint every: 200 batches

Disk:
- Peak: 180GB (temp checkpoints)
- Persistent: 40GB (text cache)
```

### Risk Assessment
```
✅ GPU OOM: Very low (4-6GB used / 15GB available)
✅ CPU OOM: Low (3.5GB used / 12GB available, 8.5GB safety margin)
⚠️  Disk space: Monitor (need 220GB free)
✅ Colab timeout: Medium risk (6 hour runtime, 12h limit)
```

---

## 7. Recommendations

### Immediate (Current Config)
1. ✅ Keep batch_size=192 (stable)
2. ✅ Checkpoint every 200 batches (RAM safety)
3. ✅ Consolidate every 50 batches (VRAM safety)
4. ✅ Monitor first checkpoint save to confirm disk space

### Optional Optimizations
1. **If no disk space issues**: Increase batch to 256-384
   - Faster encoding (better GPU utilization)
   - Needs checkpoint every 150 batches (not 200)

2. **If timeout risk**: Resume from checkpoints
   - Save checkpoints to Drive
   - Load partial progress on restart

3. **For future runs**: Reduce H_cycles to 1
   - 2x faster encoding
   - Acceptable for non-reasoning tasks
