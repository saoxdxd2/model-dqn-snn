# Training Pipeline Documentation

## 🎯 Quick Overview

```
train.py → pretrain.py → Training Loop
   ↓           ↓              ↓
Dataset    Hydra Config   Model Forward/Backward
Builder    Loading        Checkpointing
```

---

## 📊 Training Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    ENTRY POINT: train.py                     │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
         ┌─────────────────────────────┐
         │  1. Check for Checkpoint    │
         │     (auto-resume enabled)   │
         └─────────────┬───────────────┘
                       ↓
         ┌─────────────────────────────┐
         │  2. Dataset Preparation     │
         │    • Streaming builder      │
         │    • Wait for consolidated  │
         └─────────────┬───────────────┘
                       ↓
         ┌─────────────────────────────┐
         │  3. Call pretrain.py        │
         │     with Hydra config       │
         └─────────────┬───────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                  CORE TRAINING: pretrain.py                  │
│                                                               │
│  ┌────────────────────────────────────────────────────┐    │
│  │ INITIALIZATION                                      │    │
│  │  • Load config (cfg_pretrain.yaml)                 │    │
│  │  • Setup distributed (if multi-GPU)                │    │
│  │  • Register graceful shutdown (Ctrl+C handler)     │    │
│  │  • Load/build datasets                             │    │
│  │  • Initialize model, optimizer, loss               │    │
│  │  • Setup EMA, gradient monitor, W&B logging        │    │
│  └────────────────────────────────────────────────────┘    │
│                       ↓                                      │
│  ┌────────────────────────────────────────────────────┐    │
│  │ TRAINING LOOP (per epoch)                          │    │
│  │                                                     │    │
│  │  FOR each batch in train_loader:                   │    │
│  │    ├─ Forward pass                                 │    │
│  │    ├─ Compute loss (+ DQN, MTP, VQ losses)         │    │
│  │    ├─ Backward pass                                │    │
│  │    ├─ Optimizer step                               │    │
│  │    ├─ Update EMA                                   │    │
│  │    ├─ Log metrics to W&B                           │    │
│  │    └─ Check shutdown signal                        │    │
│  │                                                     │    │
│  │  EVERY eval_interval epochs:                       │    │
│  │    ├─ Evaluation on test set                       │    │
│  │    ├─ Run custom evaluators (ARC, Code, etc.)      │    │
│  │    └─ Save checkpoint                              │    │
│  │                                                     │    │
│  └────────────────────────────────────────────────────┘    │
│                       ↓                                      │
│  ┌────────────────────────────────────────────────────┐    │
│  │ CHECKPOINTING                                      │    │
│  │  • Saves to: checkpoints/multimodal-hesc/latest.pt│    │
│  │  • Includes: model, optimizer, step, epoch         │    │
│  │  • EMA model saved separately                      │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 How to Run Training

### Basic Training
```bash
python train.py
```

**What happens:**
1. Checks for existing checkpoint → auto-resumes if found
2. Waits for `consolidated_000.pt` (streaming dataset)
3. Starts training with COCONUT latent planning enabled
4. Logs to Weights & Biases
5. Saves checkpoints periodically

### Training Options
```bash
# Force rebuild dataset
python train.py --rebuild-dataset

# Only build dataset, don't train
python train.py --dataset-only

# Disable incremental training (wait for full dataset)
python train.py --no-incremental
```

---

## 📂 File Locations

### Checkpoints
```
checkpoints/
├── multimodal-hesc/
│   ├── latest.pt              # Main checkpoint (auto-resumes from this)
│   └── ema_model.pt           # EMA weights
```

### Datasets
```
datasets/
└── vision_unified/
    ├── consolidated_000.pt    # First 100 encoded batches
    ├── consolidated_001.pt    # Next 100 batches
    └── ...
```

### Configs
```
config/
├── cfg_pretrain.yaml          # Main training config
└── arch/
    ├── multimodal_hesc.yaml   # Model architecture (COCONUT enabled)
    ├── code_optimized.yaml
    ├── text_optimized.yaml
    └── ...
```

---

## 🔄 Training Loop Breakdown

### 1. **Initialization Phase**
```python
# In pretrain.py launch()
- Load Hydra config (cfg_pretrain.yaml)
- Setup distributed training (if multi-GPU)
- Register Ctrl+C handler for graceful shutdown
- Load datasets (train + test splits)
- Initialize model from config/arch/*.yaml
- Create optimizer (AdamAtan2)
- Setup EMA helper
- Initialize W&B logging
```

### 2. **Training Iteration**
```python
for epoch in range(config.epochs):
    for batch in train_loader:
        # 1. Forward pass
        carry = model.initial_carry(batch)
        new_carry, outputs = model(carry, batch)
        
        # 2. Compute losses
        loss = criterion(outputs['logits'], targets)
        + dqn_loss          # Reinforcement learning
        + vq_loss           # Vector quantization
        + mtp_loss          # Multi-token prediction
        
        # 3. Backward + optimize
        loss.backward()
        optimizer.step()
        
        # 4. Update EMA
        ema_helper.update(model)
        
        # 5. Log metrics
        wandb.log({
            'loss': loss.item(),
            'lr': current_lr,
            'step': global_step
        })
```

### 3. **Evaluation Phase** (every N epochs)
```python
model.eval()
with torch.no_grad():
    for batch in eval_loader:
        outputs = model(batch)
        metrics = compute_metrics(outputs, targets)
        
# Run custom evaluators
for evaluator in evaluators:
    evaluator.run(model, eval_loader)
```

### 4. **Checkpointing**
```python
# Save every eval_interval or on graceful shutdown
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'step': global_step,
    'epoch': current_epoch,
    'config': config
}
torch.save(checkpoint, 'checkpoints/multimodal-hesc/latest.pt')
```

---

## 🎛️ Key Configuration Parameters

### Training Hyperparameters (cfg_pretrain.yaml)
```yaml
# Dataset
data_paths: ["datasets/vision_unified"]

# Training
global_batch_size: 32
epochs: 100
lr: 3e-4
lr_min_ratio: 0.1          # Cosine schedule min LR

# Checkpointing
checkpoint_path: "checkpoints/multimodal-hesc"
checkpoint_every_eval: true
eval_interval: 10           # Evaluate every 10 epochs

# Stability
ema: true                   # Exponential moving average
ema_decay: 0.999
gradient_clip: 1.0

# Logging
wandb_project: "trm-training"
```

### Model Architecture (config/arch/multimodal_hesc.yaml)
```yaml
# TRM Encoder
hidden_size: 1024
H_cycles: 2
L_cycles: 3

# COCONUT Latent Planning
enable_latent_planning: true
latent_num_paths: 4
latent_planning_depth: 2

# Features
enable_memory: true
enable_dqn: true
enable_mtp: true
use_vq_codebook: true
```

---

## 📊 Dataset Pipeline

### Streaming Builder Flow
```
Raw Data Sources (ARC, TinyStories, Code)
    ↓
text_renderer.py (convert text → images)
    ↓
TRM Vision Encoder (encode images → capsules)
    ↓
Save as batches: batch_00000.pt, batch_00001.pt, ...
    ↓
Consolidate: 100 batches → consolidated_000.pt
    ↓
Training loads consolidated_*.pt files
```

### Key Files
- `dataset/streaming_builder.py` - Streaming dataset builder
- `dataset/build_multimodal_dataset.py` - Main builder class
- `dataset/base_builder.py` - Abstract base class

---

## 🛡️ Graceful Shutdown

**Press Ctrl+C during training:**
```
1. Signal handler catches SIGINT
2. Sets shutdown_requested flag
3. Current batch finishes safely
4. Checkpoint saved
5. Training exits cleanly
```

**Resume training:**
```bash
python train.py  # Auto-resumes from latest.pt
```

---

## 📈 Monitoring Training

### Weights & Biases Dashboard
- Loss curves (total, DQN, VQ, MTP)
- Learning rate schedule
- Gradient flow statistics
- Model metrics (accuracy, perplexity)
- Hardware utilization

### Local Logs
```bash
# View recent logs
tail -f train.log

# Check checkpoint status
ls -lh checkpoints/multimodal-hesc/
```

---

## 🔧 Advanced Features

### 1. **Multi-GPU Training**
```bash
torchrun --nproc_per_node=4 pretrain.py
```

### 2. **EMA (Exponential Moving Average)**
- Maintains shadow copy of model weights
- Smoother convergence
- Used during evaluation
- Config: `ema: true`, `ema_decay: 0.999`

### 3. **Gradient Monitoring**
- Tracks gradient flow through layers
- Detects vanishing/exploding gradients
- Periodic cleanup to prevent memory leaks

### 4. **DQN Buffer Management**
- Stores experiences for RL training
- Configurable capacity: `dqn_buffer_capacity: 500000`
- Warmup period: `dqn_warmstart_steps: 10000`

---

## 🐛 Common Issues

### "RuntimeError: CUDA out of memory"
**Solutions:**
- Reduce batch size in `cfg_pretrain.yaml`
- Disable memory bank: `enable_memory: false`
- Disable COCONUT: `enable_latent_planning: false`
- Enable gradient checkpointing: `enable_gradient_checkpointing: true`

### "No consolidated files found"
**Solution:**
```bash
# Build dataset first
python train.py --dataset-only

# Wait for consolidated_000.pt to appear
ls datasets/vision_unified/
```

### Training stops without saving
**Solution:**
- Use graceful shutdown (Ctrl+C once, wait for checkpoint)
- Check disk space
- Verify checkpoint directory permissions

---

## 📝 Training Pipeline Summary

**Entry Point:** `train.py`
- Auto-resume detection
- Dataset preparation
- Calls pretrain.py

**Core Training:** `pretrain.py`
- Hydra config loading
- Distributed setup
- Training loop
- Evaluation
- Checkpointing

**Dataset:** Streaming builder
- On-the-fly encoding
- Consolidated batches
- Resume support

**Model:** TRM with COCONUT
- Vision-unified architecture
- 12 capsules → Recursive reasoning → Latent planning → Output
- 163M parameters (35M for COCONUT)

---

## 🚦 Next Steps

1. **Test Setup:** `python scripts/test.py`
2. **Build Dataset:** `python train.py --dataset-only`
3. **Start Training:** `python train.py`
4. **Monitor:** Check W&B dashboard
5. **Resume:** Same command auto-resumes

For detailed architecture info, see `PIPELINE.md`.
