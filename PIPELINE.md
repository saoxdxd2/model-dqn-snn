# TRM Model Engine - Pipeline Documentation

## 🚀 Quick Start

```bash
# Test model configuration
python scripts/test.py --config multimodal_hesc

# Test all production configs
python scripts/test.py --all

# Start training
python train.py
```

---

## 📁 Project Structure

```
model-engine/
├── config/arch/          # Model configurations
├── dataset/              # Dataset builders
├── models/               # Model architectures
├── scripts/              # Utility scripts
├── train.py             # Main training entry point
└── pretrain.py          # Core training loop
```

---

## 🎯 Production Configurations

These configs are **tested and production-ready**:

### 1. `multimodal_hesc.yaml` ⭐ **RECOMMENDED**
- **Use case:** Main training config with COCONUT latent planning
- **Features:** 
  - 12-capsule hierarchical encoding
  - COCONUT multi-path reasoning (4 paths)
  - Memory bank, DQN, MTP
  - 163M params (35M for COCONUT)
- **Hardware:** T4 GPU (15GB VRAM)
- **Test:** `python scripts/test.py --config multimodal_hesc`

### 2. `code_optimized.yaml`
- **Use case:** Programming language modeling
- **Seq length:** 512 tokens
- **Features:** Optimized for Python, JavaScript, code completion
- **Test:** `python scripts/test.py --config code_optimized`

### 3. `text_optimized.yaml`
- **Use case:** Natural language modeling
- **Seq length:** 256 tokens  
- **Features:** WikiText, books, NLP tasks
- **Test:** `python scripts/test.py --config text_optimized`

### 4. `trm_unified.yaml`
- **Use case:** Same TRM for vision encoding AND reasoning
- **Features:** Unified architecture experiment
- **Test:** `python scripts/test.py --config trm_unified`

---

## 🧪 Experimental Configurations

These use **different architecture variants** (not currently testable via main pipeline):

- `hrm.yaml` - HierarchicalReasoningModel (separate architecture)
- `trm_hier6.yaml` - 6-level hierarchical variant
- `trm_singlez.yaml` - Single-z state variant
- `arc_optimized.yaml` - ARC-AGI specific (needs investigation)
- `vision_optimized.yaml` - Vision-only (needs investigation)

---

## 🔧 Testing Pipeline

### Test Single Config
```bash
python scripts/test.py --config multimodal_hesc
```

### Test All Production Configs
```bash
python scripts/test.py --all
```

### Test with Memory Bank
```bash
python scripts/test.py --config multimodal_hesc --enable-memory
```

### Quiet Mode
```bash
python scripts/test.py --all --quiet
```

---

## 📊 Training Pipeline

### 1. Dataset Building (Streaming)
```python
# In train.py or pretrain.py
dataset_builder = StreamingMultimodalBuilder(
    checkpoint_dir="checkpoints/dataset",
    batch_size=32
)
# Encodes data on-the-fly, saves to consolidated_*.pt files
```

### 2. Training Loop
```bash
python train.py  # Uses config/arch/multimodal_hesc.yaml by default
```

### 3. Checkpointing
- Model checkpoints: `checkpoints/model_step_*.pt`
- Dataset batches: `checkpoints/dataset/consolidated_*.pt`
- Batch encoding: `checkpoints/dataset/batch_*.pt`

---

## 🗂️ Dataset Builders

### `StreamingMultimodalBuilder`
- **Location:** `dataset/streaming_builder.py`
- **Purpose:** On-the-fly encoding with checkpointing
- **Features:**
  - Consolidates batches (100 batches → 1 file)
  - Resume from interruption
  - Multi-source data mixing

### `MultimodalDatasetBuilder`
- **Location:** `dataset/build_multimodal_dataset.py`
- **Purpose:** Pre-build entire dataset
- **Use:** For reproducible experiments

### `BaseDatasetBuilder`
- **Location:** `dataset/base_builder.py`
- **Purpose:** Abstract base class for builders

---

## 🧠 Key Features

### COCONUT Latent Planning
- **File:** `models/latent_planning.py`
- **Config:** `enable_latent_planning: true`
- **Params:** 35M additional parameters
- **Benefit:** Multi-path reasoning for complex tasks

### DQN-Based Halting
- **Purpose:** Learn when to stop reasoning
- **Config:** `enable_dqn: true`
- **Params:** Q-head for action selection

### Memory Bank
- **Purpose:** Episodic memory for long-context tasks
- **Config:** `enable_memory: true`
- **Capacity:** 16K entries (configurable)

### Multi-Token Prediction (MTP)
- **Purpose:** Predict multiple future tokens (DeepSeek-V3 style)
- **Config:** `enable_mtp: true`
- **Depths:** 3 prediction layers

---

## 📝 Scripts

### Core Scripts
- `train.py` - Main training entry point
- `pretrain.py` - Training loop implementation
- `generate_text.py` - Text generation
- `consolidate_batches.py` - Merge dataset batches

### Testing
- `scripts/test.py` - **NEW** Unified model testing
- `scripts/benchmark.py` - Performance benchmarking
- `scripts/chat.py` - Interactive chat interface
- `scripts/verify_checkpoint.py` - Checkpoint validation

### Deprecated (Remove Later)
- `quick_test.py` - Replaced by `scripts/test.py`
- `test_latent_planning.py` - Replaced by `scripts/test.py`

---

## 🔄 Maintenance Status

**Last Updated:** 2025-11-07

**Recent Changes:**
- ✅ Added COCONUT latent planning (Meta AI)
- ✅ Created unified test pipeline (`scripts/test.py`)
- ✅ Fixed float16 dtype issues in VQ codebook
- ✅ Fixed `trm_unified.yaml` halt parameters

**TODO:**
- [ ] Investigate `arc_optimized` and `vision_optimized` failures
- [ ] Remove duplicate test files
- [ ] Update experimental configs or mark as deprecated
- [ ] Add visualization tools for COCONUT path selection

---

## 💡 Usage Recommendations

### For ARC-AGI Training
Use `multimodal_hesc.yaml` - it has COCONUT planning which is ideal for spatial reasoning puzzles.

### For Code/Text Tasks
Use `code_optimized.yaml` or `text_optimized.yaml` depending on domain.

### For Experiments
Copy `multimodal_hesc.yaml` and modify parameters. Test with `scripts/test.py` before training.

---

## 🐛 Troubleshooting

### "Missing halt_max_steps parameter"
Add to config:
```yaml
halt_max_steps: 64
halt_exploration_prob: 0.1
```

### "Dtype mismatch (float vs Half)"
Fixed in `models/concept_vocab.py`. Ensure using latest version.

### "Out of memory"
- Reduce batch size
- Disable memory bank: `enable_memory: false`
- Disable COCONUT: `enable_latent_planning: false`

### Test Failures
Run verbose test:
```bash
python scripts/test.py --config <name>
```

---

## 📚 Architecture Details

### Vision-Unified Pipeline
```
Text/Images → text_renderer.py → Images (224×224)
  ↓
TRMVisionEncoder (2D spatial)
  ↓
12 Capsules (512D each)
  ↓
Recursive Reasoning (H_cycles × L_cycles)
  ↓
COCONUT Latent Planning (4 paths)
  ↓
Output Generation (concepts/tokens)
```

### Key Insight
- **Input:** 2D vision processing (learned positional embeddings)
- **Output:** 1D autoregressive generation (RoPE)
- **RoPE is ONLY used for 1D output, NOT for 2D vision encoder**

---

For more details, see individual config files in `config/arch/`.
