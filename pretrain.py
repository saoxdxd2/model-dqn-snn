from typing import Optional, Any, Sequence, List
from dataclasses import dataclass
from pathlib import Path
import os
import sys

# Memory optimization: Reduce CUDA memory fragmentation and enable expandable segments
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:128'

# Explicitly disable CUDA graphs to save ~1.5GB memory
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
os.environ['TORCHINDUCTOR_CUDAGRAPHS'] = '0'

print("\n" + "="*70)
print("🔍 DEBUG: Environment Variables Set")
print(f"  PYTORCH_CUDA_ALLOC_CONF = {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'NOT SET')}")
print(f"  TORCHINDUCTOR_CUDAGRAPHS = {os.environ.get('TORCHINDUCTOR_CUDAGRAPHS', 'NOT SET')}")
print(f"  TORCHINDUCTOR_COMPILE_THREADS = {os.environ.get('TORCHINDUCTOR_COMPILE_THREADS', 'NOT SET')}")
print("="*70 + "\n")

import math
import yaml
import shutil
import copy
import hashlib
import json
import signal

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

# Enable memory-efficient attention backend (40-60% memory reduction for attention)
try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    # Force memory-efficient backend (works on T4, doesn't require Ampere)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_flash_sdp(False)  # T4 doesn't support Flash
    torch.backends.cuda.enable_math_sdp(False)   # Math is slower and uses more memory
except (ImportError, AttributeError):
    pass  # Older PyTorch versions

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig
from adam_atan2_pytorch import AdamAtan2
try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False
    print("⚠️  bitsandbytes not available - using standard optimizers (4x more memory)")

# Global flag for graceful shutdown
shutdown_requested = False

def shutdown_signal_handler(sig, frame):
    """Handle Ctrl+C (SIGINT) gracefully.
    
    Sets a flag instead of immediately exiting to allow safe checkpoint saving.
    This prevents model corruption during backpropagation.
    """
    global shutdown_requested
    shutdown_requested = True
    print("\n" + "="*70)
    print("  🛑 SHUTDOWN REQUESTED (Ctrl+C detected)")
    print("  ⏳ Will save checkpoint at next safe point...")
    print("  ⚠️  Press Ctrl+C again to force quit (may lose progress!)")
    print("="*70 + "\n")

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from models.ema import EMAHelper
from utils.gradient_monitor import GradientFlowMonitor
from utils.annealing import compute_expansion_penalty, compute_q_temperature, compute_warmup_progress
from dataset.training_progress import TrainingProgressTracker, get_training_manifest


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_paths: List[str]
    data_paths_test: List[str] = []
    # Evaluators
    evaluators: List[EvaluatorConfig] = []

    # Hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    min_eval_interval: Optional[int] = 0 # when to start eval
    eval_save_outputs: List[str] = []

    ema: bool = False # use Exponential-Moving-Average
    ema_rate: float = 0.999 # EMA-rate
    freeze_weights: bool = False # If True, freeze weights and only learn the embeddings
    
    # DQN Stability Features
    dqn_warmup_ratio: float = 0.1  # Ratio of training for DQN warmup (0.1 = first 10%)
    dqn_warmup_steps: int = 5000  # Deprecated: use dqn_warmup_ratio
    freeze_representation_during_warmup: bool = True  # Freeze h_blocks/l_blocks during warmup
    
    # Optimizer Stability
    optimizer_type: str = "adamatan2"  # "adamatan2" or "adamw"
    enable_optimizer_fallback: bool = True  # Auto-switch to AdamW on NaN
    nan_threshold_for_fallback: int = 3  # Switch after N NaN occurrences
    
    # Expansion Penalty Annealing
    expansion_penalty_schedule: str = "cosine"  # "cosine", "linear", or "fixed"
    expansion_penalty_start: float = 0.1  # Initial penalty
    expansion_penalty_end: float = 0.001  # Final penalty
    expansion_anneal_ratio: float = 0.5  # Ratio of total training (0.5 = first 50%)
    expansion_anneal_steps: int = 50000  # Deprecated: use expansion_anneal_ratio
    
    # Replay Buffer Sampling
    replay_recent_fraction: float = 0.25  # Fraction from recent transitions
    replay_max_age: int = 100000  # Discard transitions older than this
    
    # Q-Head Temperature Annealing (Exploration -> Exploitation)
    enable_q_temperature_annealing: bool = True
    q_temperature_start: float = 1.0  # High temp = exploration
    q_temperature_end: float = 0.1  # Low temp = exploitation
    q_temperature_anneal_ratio: float = 1.0  # Ratio of total training (1.0 = 100%)
    q_temperature_schedule: str = "exponential"  # "linear", "exponential", "cosine"

def per_layer_gradient_normalization(model: nn.Module, config) -> dict:
    """
    Apply per-layer gradient normalization instead of global clipping.
    
    Benefits:
    - Prevents strong gradients in one module from being clipped
    - Allows different learning dynamics per component
    - More stable training for hierarchical models like TRM
    
    Args:
        model: TRM model
        config: Training config
    
    Returns:
        Dictionary of gradient norms per component
    """
    grad_norms = {}
    
    # Define module groups and their keywords
    module_groups = {
        'encoder': ['embed', 'input_proj', 'puzzle_emb'],
        'H_level': ['H_level', 'H_init', 'h_blocks'],
        'L_level': ['L_level', 'L_init', 'l_blocks'],
        'q_head': ['q_head'],
        'memory': ['memory'],
        'lm_head': ['lm_head', 'output']
    }
    
    # Get max norms per group (from config with defaults)
    max_norms = {
        'encoder': getattr(config.arch, 'encoder_max_grad_norm', 1.0),
        'H_level': getattr(config.arch, 'h_level_max_grad_norm', 1.0),
        'L_level': getattr(config.arch, 'l_level_max_grad_norm', 1.0),
        'q_head': getattr(config.arch, 'q_head_max_grad_norm', 2.0),  # Allow larger for RL
        'memory': getattr(config.arch, 'memory_max_grad_norm', 1.0),
        'lm_head': getattr(config.arch, 'lm_head_max_grad_norm', 1.0),
    }
    
    # Clip per group
    for group_name, keywords in module_groups.items():
        # Collect parameters for this group
        group_params = []
        for name, param in model.named_parameters():
            if param.grad is not None and any(kw in name for kw in keywords):
                group_params.append(param)
        
        if group_params:
            # Clip this group
            grad_norm = torch.nn.utils.clip_grad_norm_(
                group_params,
                max_norm=max_norms[group_name]
            )
            grad_norms[f'grad_{group_name}_norm'] = grad_norm.item()
    
    return grad_norms


@dataclass
class TrainState:
    model: nn.Module  # May be torch.compile wrapped
    original_model: nn.Module  # Unwrapped model for .parameters() and EMA
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int
    last_checkpoint_step: int = 0  # Track last saved checkpoint
    representation_frozen: bool = False  # Track if representation layers are frozen
    nan_count: int = 0  # Track NaN occurrences for optimizer fallback


def detect_dataset_features(data_path: str) -> dict:
    """Auto-detect dataset type and features from data."""
    import torch
    
    features = {
        'is_capsule': False,
        'is_vision': False,
        'is_text': False,
        'has_checksums': False,
        'has_children': False,
        'enable_dqn': False,
        'enable_expansion': False
    }
    
    # Try loading as capsule dataset
    capsule_paths = [
        data_path.replace('semantic_embeddings', 'capsule_dataset'),
        os.path.join(data_path, 'capsule_dataset.pt')
    ]
    
    for path in capsule_paths:
        if os.path.exists(path):
            try:
                data = torch.load(path, map_location='cpu')
                if 'sketches' in data:
                    features['is_capsule'] = True
                    features['has_checksums'] = 'checksums' in data
                    features['has_children'] = 'children' in data
                    features['enable_expansion'] = features['has_children']
                    features['enable_dqn'] = features['enable_expansion']
                    return features
            except Exception:
                pass
    
    # Detect from path patterns
    path_lower = data_path.lower()
    if 'arc' in path_lower or 'vision' in path_lower or 'cifar' in path_lower or 'image' in path_lower:
        features['is_vision'] = True
    elif 'text' in path_lower or 'wikitext' in path_lower or 'stories' in path_lower:
        features['is_text'] = True
    
    return features


def load_datasets(config: PretrainConfig, rank: int, world_size: int, split: str = 'train'):
    """Unified dataset loading with auto-detection."""
    
    # Check if semantic_mode is explicitly set in config
    semantic_mode = getattr(config, 'semantic_mode', None)
    
    # Auto-detect dataset features only if semantic_mode not explicitly set
    if semantic_mode is None:
        dataset_path = config.data_paths[0] if config.data_paths else ''
        features = detect_dataset_features(dataset_path)
        semantic_mode = features['is_capsule']
    else:
        # Still detect features for logging purposes
        dataset_path = config.data_paths[0] if config.data_paths else ''
        features = detect_dataset_features(dataset_path) if os.path.exists(dataset_path) else {'is_capsule': semantic_mode, 'has_checksums': False, 'has_children': False, 'enable_dqn': False}
    
    if rank == 0 and features['is_capsule']:
        print(f"\n🔍 Auto-detected HESC capsule dataset")
        print(f"   Checksums: {features['has_checksums']}")
        print(f"   Children: {features['has_children']}")
        print(f"   DQN recommended: {features['enable_dqn']}")
    
    if semantic_mode:
        # Load unified multimodal capsule dataset
        import torch
        
        # Construct capsule path - handle both explicit semantic_dataset and data_paths override
        if hasattr(config, 'semantic_dataset') and config.semantic_dataset:
            capsule_path = config.semantic_dataset.replace('semantic_embeddings', 'capsule_dataset') if split == 'train' else config.semantic_eval_dataset.replace('semantic_embeddings', 'capsule_dataset')
        else:
            # Fallback: construct from data_paths
            base_path = config.data_paths[0] if split == 'train' else (config.data_paths_test[0] if config.data_paths_test else config.data_paths[0])
            capsule_path = os.path.join(base_path, 'capsule_dataset.pt' if split == 'train' else 'capsule_dataset_test.pt')
        
        if not os.path.exists(capsule_path):
            raise FileNotFoundError(
                f"Capsule dataset not found: {capsule_path}\n"
                f"Build with: python dataset/build_multimodal_dataset.py build_text --input_file wikitext2 --output_dir {os.path.dirname(capsule_path)}"
            )
        
        print(f"\n📦 Loading multimodal capsule dataset: {capsule_path}")
        data = torch.load(capsule_path)
        
        # Import CapsuleState for expansion tracking
        from models.capsule_state import CapsuleState
        
        # Extract components (unified format)
        sketches = data['sketches']  # [N, k, hidden_size]
        checksums = data.get('checksums', None)
        children = data.get('children', None)
        
        print(f"   Samples: {sketches.shape[0]}, Capsules: {sketches.shape[1]}, Dim: {sketches.shape[2]}")
        if children is not None:
            print(f"   Expandable: {children.shape[2]} children per capsule")
        
        # Metadata with concept vocabulary
        num_concepts = getattr(config.arch, 'num_concepts', 2048)
        vocab_size = num_concepts + 4  # Concept vocab + control symbols
        
        metadata = PuzzleDatasetMetadata(
            seq_len=sketches.shape[1],
            vocab_size=vocab_size,
            pad_id=0,
            ignore_label_id=-100,
            blank_identifier_id=0,
            num_puzzle_identifiers=0,
            total_groups=sketches.shape[0],
            mean_puzzle_examples=1.0,
            total_puzzles=sketches.shape[0],
            sets=["all"]
        )
        
        # Store children and checksums for batch-time CapsuleState creation
        # These will be accessed by index during training
        dataset_children = children  # [N, k, m, D] or None
        dataset_checksums = checksums  # [N, k, R] or None
        
        # Create dataset with available components
        from torch.utils.data import TensorDataset
        if children is not None and checksums is not None:
            dataset = TensorDataset(sketches, checksums, children)
        elif checksums is not None:
            dataset = TensorDataset(sketches, checksums)
        else:
            dataset = TensorDataset(sketches)
        
        local_batch_size = config.global_batch_size // world_size
        raw_dataloader = DataLoader(
            dataset,
            batch_size=local_batch_size,
            shuffle=(split == 'train'),
            num_workers=0,
            pin_memory=True
        )
        
        # Wrap dataloader to match expected format: (set_name, batch, global_batch_size)
        class CapsuleDataLoaderWrapper:
            def __init__(self, raw_loader, global_batch_size):
                self.raw_loader = raw_loader
                self.global_batch_size = global_batch_size
            
            def __iter__(self):
                for batch_data in self.raw_loader:
                    # batch_data is tuple of (sketches,) or (sketches, checksums) or (sketches, checksums, children)
                    # Sketches are already [B, 12, 512] from encoder.sketch_projection
                    if len(batch_data) == 3:
                        sketches = batch_data[0]  # [B, 12, 512]
                        batch = {
                            'inputs': sketches,
                            'checksums': batch_data[1],
                            'children': batch_data[2],
                            'puzzle_identifiers': torch.zeros(sketches.shape[0], dtype=torch.long)
                        }
                    elif len(batch_data) == 2:
                        sketches = batch_data[0]  # [B, 12, 512]
                        batch = {
                            'inputs': sketches,
                            'checksums': batch_data[1],
                            'children': None,
                            'puzzle_identifiers': torch.zeros(sketches.shape[0], dtype=torch.long)
                        }
                    else:
                        sketches = batch_data[0]  # [B, 12, 512]
                        batch = {
                            'inputs': sketches,
                            'checksums': None,
                            'children': None,
                            'puzzle_identifiers': torch.zeros(sketches.shape[0], dtype=torch.long)
                        }
                    
                    # Yield in expected format: (set_name, batch, global_batch_size)
                    yield 'capsule', batch, self.global_batch_size
            
            def __len__(self):
                return len(self.raw_loader)
        
        wrapped_dataloader = CapsuleDataLoaderWrapper(raw_dataloader, config.global_batch_size)
        return wrapped_dataloader, metadata
    
    # Token-based mode (original logic)
    dataset_paths = config.data_paths if split == 'train' else config.data_paths_test
    
    kwargs = {}
    if hasattr(config, 'group_batch_size') and config.group_batch_size is not None:
        kwargs['group_batch_size'] = config.group_batch_size
    
    if hasattr(config, 'groups_per_batch'):
        kwargs['groups_per_batch'] = config.groups_per_batch
    
    # Calculate epochs_per_iter based on eval_interval (same logic as in launch function)
    epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    
    try:
        dataset = PuzzleDataset(PuzzleDatasetConfig(
            seed=config.seed,
            dataset_paths=dataset_paths,
            global_batch_size=config.global_batch_size,
            test_set_mode=(split != 'train'),  # True for test/eval, False for train
            epochs_per_iter=epochs_per_iter,
            rank=rank,
            num_replicas=world_size,
            **kwargs
        ), split=split)
    except FileNotFoundError as e:
        # Auto-fallback: Build dataset if missing
        if rank == 0:  # Only build on rank 0 in distributed training
            print(f"\n{'='*60}")
            print(f"Dataset not found: {e}")
            print(f"Auto-building dataset with default configuration...")
            print(f"{'='*60}\n")
            
            import subprocess
            # Use NEW multimodal builder with text rendering support
            # Build to the FIRST dataset path that training expects
            output_dir = dataset_paths[0] if dataset_paths else "datasets/vision_unified"
            build_cmd = [
                "python", "dataset/build_multimodal_dataset.py",
                "--config.source-paths", "kaggle/combined/arc-agi",
                "--config.output-dir", output_dir,
                "--config.render-text-to-image",
                "--config.use-capsules",
                "--config.seed", "42"
            ]
            
            print(f"Running: {' '.join(build_cmd)}\n")
            result = subprocess.run(build_cmd, check=True)
            
            if result.returncode == 0:
                print(f"\n{'='*60}")
                print(f"Dataset built successfully! Retrying data load...")
                print(f"{'='*60}\n")
            else:
                raise RuntimeError(f"Dataset build failed with code {result.returncode}")
        
        # Wait for rank 0 to finish building in distributed mode
        if world_size > 1:
            dist.barrier()
        
        # Retry loading dataset with ALL original kwargs
        dataset = PuzzleDataset(PuzzleDatasetConfig(
            seed=config.seed,
            dataset_paths=dataset_paths,
            global_batch_size=config.global_batch_size,
            test_set_mode=(split != 'train'),
            epochs_per_iter=epochs_per_iter,
            rank=rank,
            num_replicas=world_size,
            **kwargs
        ), split=split)
    
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True
    )
    return dataloader, dataset.metadata


def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int, total_steps: int = 100000):
    """Create model with loss head.
    
    Args:
        total_steps: Total training steps for annealing schedules
    """
    # Build model config: merge YAML arch config with dataset metadata
    # __pydantic_extra__ contains all extra YAML fields (causal, input_vocab_size, etc.)
    # Dataset metadata overrides seq_len/vocab_size/num_puzzle_identifiers
    
    # Auto-detect dataset type and enable appropriate training techniques
    semantic_mode = getattr(config, 'semantic_mode', False)
    enable_capsule_expansion = getattr(config.arch, 'enable_capsule_expansion', False)
    
    if rank == 0:
        print(f"\n🔍 Dataset Analysis:")
        print(f"   Semantic/Capsule mode: {semantic_mode}")
        print(f"   Capsule expansion: {enable_capsule_expansion}")
    
    # Filter out keys that will be explicitly set to avoid duplicate keyword argument error
    # IMPORTANT: Keep input_vocab_size from YAML (for vision with separate input/output vocabs)
    extra_config = {k: v for k, v in config.arch.__pydantic_extra__.items() 
                   if k not in ['batch_size', 'vocab_size', 'seq_len', 'num_puzzle_identifiers']}
    
    # Auto-enable DQN for capsule datasets with expansion
    if semantic_mode and enable_capsule_expansion:
        if 'enable_dqn' not in extra_config or not extra_config['enable_dqn']:
            if rank == 0:
                print(f"   ⚙️  Auto-enabling DQN for capsule expansion control")
            extra_config['enable_dqn'] = True
            # Set expansion-friendly DQN params
            if 'dqn_loss_weight' not in extra_config:
                extra_config['dqn_loss_weight'] = 0.5
            if 'enable_per' not in extra_config:
                extra_config['enable_per'] = True  # Prioritized experience replay
    
    model_cfg = dict(
        **extra_config,
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
    )
    
    model_cfg['semantic_mode'] = semantic_mode
    model_cfg['enable_capsule_expansion'] = enable_capsule_expansion
    model_cfg['total_steps'] = total_steps  # For annealing schedules
    
    if rank == 0:
        print(f"\n📐 Model Config:")
        print(f"   arch: {config.arch.name}")
        print(f"   input_vocab_size: {model_cfg.get('input_vocab_size', 'NOT SET')}")
        print(f"   output_vocab_size: {model_cfg.get('output_vocab_size', 'NOT SET')}")
        print(f"   hidden_size: {model_cfg.get('hidden_size', 'NOT SET')}")
        print(f"   seq_len: {model_cfg.get('seq_len')}")
        print(f"   total_steps: {total_steps:,}")

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        print(model)
        # Pass total_steps to loss head for annealing schedules
        loss_kwargs = dict(config.arch.loss.__pydantic_extra__)
        loss_kwargs['total_steps'] = total_steps
        model = loss_head_cls(model, **loss_kwargs)  # type: ignore
        
        if "DISABLE_COMPILE" not in os.environ:
            # Configure Triton for GPU compute capability 7.5 (Tesla T4)
            # Max block size is 4096, not 8192
            try:
                import torch._inductor.config as inductor_config
                
                # CRITICAL: Explicitly disable CUDA graphs to save 1.5GB memory
                print("\n" + "="*70)
                print("🔍 DEBUG: Inductor Config BEFORE changes")
                print(f"  triton.cudagraphs = {getattr(inductor_config.triton, 'cudagraphs', 'NOT SET')}")
                print(f"  cudagraphs = {getattr(inductor_config, 'cudagraphs', 'NOT SET')}")
                print(f"  max_autotune = {inductor_config.max_autotune}")
                print("="*70 + "\n")
                
                inductor_config.triton.cudagraphs = False
                if hasattr(inductor_config, 'cudagraphs'):
                    inductor_config.cudagraphs = False
                
                # Disable aggressive optimizations that require more SMs
                inductor_config.max_autotune = False
                inductor_config.max_autotune_gemm = False
                
                print("\n" + "="*70)
                print("🔍 DEBUG: Inductor Config AFTER changes")
                print(f"  triton.cudagraphs = {inductor_config.triton.cudagraphs}")
                print(f"  cudagraphs = {getattr(inductor_config, 'cudagraphs', 'NOT SET')}")
                print(f"  max_autotune = {inductor_config.max_autotune}")
                print(f"  max_autotune_gemm = {inductor_config.max_autotune_gemm}")
                print("="*70 + "\n")
                
                # Try to set Triton configs if available (API changed in newer PyTorch)
                try:
                    if hasattr(inductor_config, 'triton') and hasattr(inductor_config.triton, 'max_block'):
                        inductor_config.triton.max_block = {"X": 4096, "Y": 4096, "Z": 4096, "R": 4096}
                        inductor_config.triton.max_tiles = 4
                        print(f"🔧 Triton config: max_block={inductor_config.triton.max_block}")
                    else:
                        print("ℹ️  Triton max_block config not available (newer PyTorch API)")
                except AttributeError:
                    pass  # Triton config API changed, skip
                
                if rank == 0:
                    print(f"✅ CUDA graphs disabled: triton.cudagraphs={inductor_config.triton.cudagraphs}")
            except Exception as e:
                print(f"⚠️  Could not configure inductor: {e}")
            
            # Compile with EAGER backend - prevents deferred CUDA graphs at runtime
            if rank == 0:
                print("\n🚀 Compiling model with torch.compile (backend='eager')...")
                print("   This prevents deferred CUDA graph allocation that bypasses config settings.")
                torch.cuda.empty_cache()
                print(f"\n📊 GPU Memory BEFORE compile:")
                print(f"  Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
                print(f"  Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
                print(f"  Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated())/1e9:.2f} GB\n")
            model = torch.compile(model, backend="eager")  # type: ignore
            if rank == 0:
                print(f"\n📊 GPU Memory AFTER compile:")
                print(f"  Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
                print(f"  Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
                print(f"  Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated())/1e9:.2f} GB\n")

        # Load checkpoint
        if rank == 0:
            load_checkpoint(model, config)

        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    # Helper function to create optimizer with fallback
    def create_optimizer_with_fallback(params, config, optimizer_type="main"):
        """Create optimizer with automatic fallback. Uses bitsandbytes 8-bit to save 75% memory."""
        lr = 1e-8  # Will be set by scheduler
        wd = config.weight_decay
        betas = (config.beta1, config.beta2)
        
        # Priority 1: Use bitsandbytes 8-bit optimizer (saves 75% memory)
        if BNB_AVAILABLE:
            try:
                optimizer = bnb.optim.AdamW8bit(
                    params, 
                    lr=lr, 
                    weight_decay=wd, 
                    betas=betas,
                    min_8bit_size=4096  # Only quantize large tensors
                )
                if rank == 0:
                    print(f"  Using bitsandbytes AdamW8bit optimizer for {optimizer_type} (saves 75% memory)")
                return optimizer
            except Exception as e:
                if rank == 0:
                    print(f"  ⚠️  bitsandbytes failed: {e}, falling back to standard optimizer")
        
        # Priority 2: Force AdamW if configured
        if config.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas)
            if rank == 0:
                print(f"  Using AdamW optimizer for {optimizer_type}")
        else:
            # Priority 3: Try AdamAtan2
            try:
                optimizer = AdamAtan2(params, lr=lr, weight_decay=wd, betas=betas)
                if rank == 0:
                    print(f"  Using AdamAtan2 optimizer for {optimizer_type}")
            except Exception as e:
                if config.enable_optimizer_fallback:
                    if rank == 0:
                        print(f"  ⚠️  AdamAtan2 failed: {e}")
                        print(f"  Falling back to AdamW optimizer for {optimizer_type}")
                    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas)
                else:
                    raise
        
        return optimizer
    
    # Optimizers and lr
    # Handle puzzle_emb_ndim=0 case: no puzzle embeddings, only main optimizer
    if config.arch.puzzle_emb_ndim == 0 or not hasattr(model.model, 'puzzle_emb'):
        # Text/Code/Vision models without puzzle-specific embeddings
        optimizers = [
            create_optimizer_with_fallback(model.parameters(), config, "main")
        ]
        optimizer_lrs = [
            config.lr
        ]
    elif config.freeze_weights:
        # Only train puzzle embeddings (freeze main model)
        if not hasattr(model.model, 'puzzle_emb'):
            raise ValueError("freeze_weights=True requires puzzle_emb_ndim > 0")
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            )
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr
        ]
    else:
        # ARC models: train both puzzle embeddings and main model
        if not hasattr(model.model, 'puzzle_emb'):
            raise ValueError("Dual optimizer mode requires puzzle_emb_ndim > 0")
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            ),
            create_optimizer_with_fallback(model.parameters(), config, "main")
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr,
            config.lr
        ]

    return model, optimizers, optimizer_lrs

def mix_weights_direct(device, alpha, net, nets):
    sd = []
    for i in range(len(nets)):
        sd += [nets[i].state_dict()]
    sd_alpha = {}
    for k in sd[0].keys():
        comb_net = alpha[0]*sd[0][k].to(device)
        for i in range(1,len(nets)):
            comb_net += alpha[i]*sd[i][k].to(device)
        sd_alpha[k] =  comb_net
    net.load_state_dict(sd_alpha)
    return net

def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    # Auto-adjust hyperparameters based on dataset characteristics
    semantic_mode = getattr(config, 'semantic_mode', False)
    dataset_size = train_metadata.total_groups
    
    if rank == 0:
        print(f"\n⚙️  Auto-tuning hyperparameters:")
        print(f"   Dataset size: {dataset_size:,} samples")
    
    # Adjust learning rate based on dataset size and mode
    adjusted_lr = config.lr
    if semantic_mode:
        # Capsule mode: can use higher LR due to compressed inputs
        adjusted_lr = config.lr * 1.5
        if rank == 0:
            print(f"   LR boost for capsule mode: {config.lr:.2e} → {adjusted_lr:.2e}")
    elif dataset_size < 1000:
        # Small dataset: reduce LR to prevent overfitting
        adjusted_lr = config.lr * 0.5
        if rank == 0:
            print(f"   LR reduction for small dataset: {config.lr:.2e} → {adjusted_lr:.2e}")
    
    # Store adjusted hyperparameters
    config.lr = adjusted_lr
    
    # Calculate total training steps correctly
    # Steps per epoch = (total samples) / batch_size
    # Total steps = steps_per_epoch * epochs
    samples_per_epoch = train_metadata.total_groups * train_metadata.mean_puzzle_examples
    steps_per_epoch = max(1, int(samples_per_epoch / config.global_batch_size))
    total_steps = steps_per_epoch * config.epochs
    
    if rank == 0:
        print(f"   Samples per epoch: {samples_per_epoch:,}")
        print(f"   Steps per epoch: {steps_per_epoch:,}")
        print(f"   Total training steps: {total_steps:,} ({config.epochs} epochs)")

    # Model (pass total_steps for annealing)
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, rank=rank, world_size=world_size, total_steps=total_steps)
    
    # Store original model reference for parameter access and EMA
    original_model = model
    
    # torch.compile for 25% speedup (T4 compatible)
    # NOTE: Compiled model is used for forward pass, original for .parameters() and EMA
    if getattr(config, 'use_torch_compile', True) and hasattr(torch, 'compile'):
        if rank == 0:
            print(f"   🚀 Compiling model with torch.compile (25% speedup)...")
        model = torch.compile(model, mode='max-autotune')

    return TrainState(
        step=0,
        total_steps=total_steps,

        model=model,
        original_model=original_model,  # Store for EMA and parameter counting
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None
    )


def get_checkpoint_name(dataset_path: str) -> str:
    """Get checkpoint filename from dataset path.
    
    Examples:
        data/text-wikitext2 -> text.pt
        data/arc-aug-1000 -> arc.pt
        data/text-tinystories -> text-tiny.pt
    """
    # Extract dataset name from path
    dataset_name = os.path.basename(dataset_path)
    
    # Map common patterns to simple names
    if 'text-wikitext' in dataset_name:
        return 'text.pt'
    elif 'text-tinystories' in dataset_name or 'tinystories' in dataset_name:
        return 'text-tiny.pt'
    elif 'arc' in dataset_name:
        return 'arc.pt'
    elif 'code' in dataset_name:
        return 'code.pt'
    elif 'vision' in dataset_name or 'cifar' in dataset_name or 'image' in dataset_name:
        return 'vision.pt'
    elif 'alpaca' in dataset_name:
        return 'alpaca.pt'
    elif 'sharegpt' in dataset_name:
        return 'sharegpt.pt'
    elif 'sudoku' in dataset_name:
        return 'sudoku.pt'
    elif 'maze' in dataset_name:
        return 'maze.pt'
    else:
        # Fallback: use sanitized dataset name
        return f"{dataset_name.replace('/', '_').replace('-', '_')}.pt"


def compute_file_checksum(filepath: str) -> str:
    """Compute SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read in 64KB chunks for memory efficiency
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def save_checkpoint_atomic(checkpoint: dict, filepath: str):
    """Save checkpoint atomically with checksum verification.
    
    Uses atomic write (temp file + rename) to prevent corruption during save.
    Generates SHA256 checksum for compression safety.
    Creates backup of previous checkpoint for rollback.
    """
    # Define all temp file paths upfront to avoid scope issues
    temp_checkpoint = filepath + ".tmp"
    backup_file = filepath + ".backup"
    checksum_file = filepath + ".sha256"
    checksum_temp = checksum_file + ".tmp"
    
    try:
        # Backup existing checkpoint before overwriting
        if os.path.exists(filepath):
            try:
                shutil.copy2(filepath, backup_file)
            except Exception as e:
                print(f"⚠️  Failed to create backup: {e}")
        
        # Save checkpoint file atomically
        try:
            torch.save(checkpoint, temp_checkpoint)
            
            # Verify integrity before committing
            try:
                test_load = torch.load(temp_checkpoint, map_location='cpu')
                del test_load  # Free memory
            except Exception as e:
                print(f"❌ Checkpoint verification failed: {e}")
                if os.path.exists(backup_file):
                    print("   Restoring from backup...")
                    shutil.copy2(backup_file, filepath)
                raise e
            
            # Move atomically (crash-safe)
            shutil.move(temp_checkpoint, filepath)
            
        except Exception as e:
            # Clean up failed temp file
            if os.path.exists(temp_checkpoint):
                os.remove(temp_checkpoint)
            raise e
        
        # Compute checksum
        checksum = compute_file_checksum(filepath)
        
        # Save checksum metadata
        metadata = {
            "checksum": checksum,
            "step": checkpoint.get('step', 0),
            "size_bytes": os.path.getsize(filepath),
            "timestamp": os.path.getmtime(filepath)
        }
        with open(checksum_temp, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Atomic rename (this is the actual "commit" point)
        # If interrupted here, temp files exist but original is intact
        os.replace(checksum_temp, checksum_file)
        
        return checksum
        
    except Exception as e:
        # Clean up temp files on error (now all variables are in scope)
        if os.path.exists(temp_checkpoint):
            os.remove(temp_checkpoint)
        if os.path.exists(checksum_temp):
            os.remove(checksum_temp)
        raise e


def save_train_state(config: PretrainConfig, train_state: TrainState, progress_tracker: Optional[TrainingProgressTracker] = None):
    """Save complete training state with corruption protection and backup.
    
    Features:
    - Atomic writes (temp file + rename)
    - SHA256 checksum verification
    - Safe for compression/transfer
    - Creates backup of previous checkpoint for rollback.
    - Saves training progress through consolidated chunks
    """
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    
    # Update last checkpoint step
    train_state.last_checkpoint_step = train_state.step
    
    checkpoint = {
        'step': train_state.step,
        'model_state_dict': train_state.original_model.state_dict(),  # Use unwrapped model
        'optimizer_states': [opt.state_dict() for opt in train_state.optimizers],
        'carry': train_state.carry,  # Save RNN/memory states
        
        # Training metadata for phase completion tracking
        'config_epochs': config.epochs,
        'total_steps': train_state.total_steps,
        'dataset': config.data_paths[0] if config.data_paths else 'unknown',
    }
    
    # Save training progress through consolidated chunks
    if progress_tracker is not None:
        progress_tracker.save()
        checkpoint['has_training_progress'] = True
    
    # Save DQN-specific state if enabled
    if hasattr(train_state.model, 'replay_buffer'):
        checkpoint['replay_buffer_size'] = len(train_state.model.replay_buffer)
        checkpoint['dqn_step_counter'] = getattr(train_state.model, 'dqn_step_counter', 0)
    
    # Get dataset-specific checkpoint name (e.g., text.pt, arc.pt, vision.pt)
    checkpoint_name = get_checkpoint_name(config.data_paths[0] if config.data_paths else 'model')
    checkpoint_file = os.path.join(config.checkpoint_path, checkpoint_name)
    backup_file = checkpoint_file + ".backup"
    
    # Save single checkpoint file with atomic write + checksum
    checksum = save_checkpoint_atomic(checkpoint, checkpoint_file)
    
    # Log save info
    print(f"   💾 Saved: {checkpoint_name}")
    print(f"   SHA256: {checksum[:16]}...")


def verify_checkpoint_integrity(checkpoint_path: str) -> bool:
    """Verify checkpoint integrity using SHA256 checksum.
    
    Returns True if checksum matches or no checksum file exists (legacy).
    Returns False if checksum verification fails.
    """
    checksum_file = checkpoint_path + ".sha256"
    
    # If no checksum file, assume legacy checkpoint (skip verification)
    if not os.path.exists(checksum_file):
        print("   ⚠️  No checksum file found (legacy checkpoint)")
        return True
    
    try:
        # Load expected checksum
        with open(checksum_file, 'r') as f:
            metadata = json.load(f)
        expected_checksum = metadata['checksum']
        
        # Compute actual checksum
        print("   🔍 Verifying checkpoint integrity...")
        actual_checksum = compute_file_checksum(checkpoint_path)
        
        if actual_checksum == expected_checksum:
            print(f"   ✅ Checksum verified: {actual_checksum[:16]}...")
            return True
        else:
            print(f"   ❌ CHECKSUM MISMATCH!")
            print(f"      Expected: {expected_checksum[:16]}...")
            print(f"      Got:      {actual_checksum[:16]}...")
            print(f"      ⚠️  Checkpoint may be corrupted!")
            return False
            
    except Exception as e:
        print(f"   ⚠️  Checksum verification failed: {e}")
        return True  # Allow loading if verification fails


def load_checkpoint(model: nn.Module, config: PretrainConfig, optimizers=None, train_state=None):
    """Load checkpoint with integrity verification.
    
    Supports cross-dataset loading (e.g., load text.pt into vision training).
    """
    if config.load_checkpoint is not None:
        print(f"Loading checkpoint {config.load_checkpoint}")
        
        # Verify integrity before loading
        if not verify_checkpoint_integrity(config.load_checkpoint):
            print("\n⚠️  WARNING: Checkpoint failed integrity check!")
            response = input("   Continue loading anyway? [y/N]: ")
            if response.lower() != 'y':
                raise RuntimeError("Checkpoint verification failed. Aborting.")

        # Load checkpoint
        checkpoint = torch.load(config.load_checkpoint, map_location="cuda")
        
        # Show checkpoint info (for cross-dataset loading)
        if 'dataset' in checkpoint:
            print(f"   Source dataset: {checkpoint['dataset']}")
        if 'step' in checkpoint:
            print(f"   Checkpoint step: {checkpoint.get('step', 0):,}")
        
        # Handle both old format (dict with state_dict) and new format (dict with model_state_dict)
        if 'model_state_dict' in checkpoint:
            # New format with full training state
            state_dict = checkpoint['model_state_dict']
            
            # Load optimizer states if provided
            if optimizers is not None and 'optimizer_states' in checkpoint:
                print(f"Restoring optimizer states from checkpoint")
                for opt, opt_state in zip(optimizers, checkpoint['optimizer_states']):
                    opt.load_state_dict(opt_state)
            
            # Restore training step
            if train_state is not None and 'step' in checkpoint:
                train_state.step = checkpoint['step']
                print(f"Resuming from step {train_state.step}")
        else:
            # Old format: checkpoint is the state_dict directly
            state_dict = checkpoint

        # Resize and reset puzzle emb if needed (only if model has puzzle embeddings)
        puzzle_emb_name = "_orig_mod.model.inner.puzzle_emb.weights"
        if hasattr(model.model, 'puzzle_emb') and model.model.puzzle_emb is not None:
            expected_shape: torch.Size = model.model.puzzle_emb.weights.shape  # type: ignore
            if puzzle_emb_name in state_dict:
                puzzle_emb = state_dict[puzzle_emb_name]
                if puzzle_emb.shape != expected_shape:
                    print(f"Resetting puzzle embedding as shape is different. Found {puzzle_emb.shape}, Expected {expected_shape}")
                    # Re-initialize using mean
                    state_dict[puzzle_emb_name] = (
                        torch.mean(puzzle_emb, dim=0, keepdim=True).expand(expected_shape).contiguous()
                    )
        model.load_state_dict(state_dict, assign=True)


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )



def create_evaluators(config: PretrainConfig, eval_metadata: PuzzleDatasetMetadata) -> List[Any]:
    data_paths =config.data_paths_test if len(config.data_paths_test)>0 else config.data_paths
    # Initialize evaluators
    evaluators = []
    for cfg in config.evaluators:
        for data_path in data_paths:
            cls = load_model_class(cfg.name, "evaluators.")(
                data_path=data_path, eval_metadata=eval_metadata, **cfg.__pydantic_extra__
            )  # type: ignore
            evaluators.append(cls)

    return evaluators

def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int, gradient_monitor=None):
    train_state.step += 1
    if train_state.step > train_state.total_steps:  # At most train_total_steps
        return
    
    # DQN Warmup: Freeze representation layers during early training
    if config.freeze_representation_during_warmup:
        # Calculate warmup steps from ratio
        if hasattr(config, 'dqn_warmup_ratio'):
            dqn_warmup_steps = int(train_state.total_steps * config.dqn_warmup_ratio)
        else:
            # Fallback to deprecated steps
            dqn_warmup_steps = config.dqn_warmup_steps
        
        if train_state.step < dqn_warmup_steps and not train_state.representation_frozen:
            # Freeze representation layers (h_blocks, l_blocks)
            if rank == 0:
                print(f"\n{'='*70}")
                print(f"  🔒 DQN WARMUP: Freezing representation layers")
                print(f"  Steps 0-{dqn_warmup_steps}: Supervised learning only")
                print(f"  Step {dqn_warmup_steps}+: Full DQN training")
                print(f"  Warmup ratio: {config.dqn_warmup_ratio*100:.1f}% of training")
                print(f"{'='*70}\n")
            
            for name, param in train_state.original_model.named_parameters():
                if 'h_blocks' in name or 'l_blocks' in name or 'h_level' in name or 'l_level' in name:
                    param.requires_grad = False
            
            train_state.representation_frozen = True
        
        elif train_state.step == dqn_warmup_steps and train_state.representation_frozen:
            # Unfreeze after warmup
            if rank == 0:
                print(f"\n{'='*70}")
                print(f"  🔓 DQN WARMUP COMPLETE: Unfreezing representation layers")
                print(f"  Full model training enabled")
                print(f"{'='*70}\n")
            
            for param in train_state.original_model.parameters():
                param.requires_grad = True
            
            train_state.representation_frozen = False

    # Handle raw_samples format (vision-unified pipeline)
    if isinstance(batch, dict) and 'raw_samples' in batch:
        import torchvision.transforms as transforms
        from PIL import Image
        import numpy as np
        
        # Extract images from raw samples
        raw_samples = batch['raw_samples']
        batch_size = len(raw_samples)
        
        # Stack input and output images
        # Each sample: {'input_image': PIL.Image, 'output_image': PIL.Image, 'task_id': str}
        input_images = []
        output_images = []
        
        # Convert PIL images to tensors (normalized to [0, 1])
        to_tensor = transforms.ToTensor()
        
        for sample in raw_samples:
            # DataSample has different fields based on modality: text, image, grid
            img = None
            
            # 1. If image field is populated (direct image)
            if sample.image is not None:
                if isinstance(sample.image, np.ndarray):
                    img = Image.fromarray(sample.image.astype(np.uint8))
                else:
                    img = sample.image  # Already PIL Image
            
            # 2. If grid field (ARC/maze) - visualize as image
            elif sample.grid is not None:
                # Convert grid to RGB image (simple colormap)
                grid = sample.grid
                if grid.dtype == np.int32 or grid.dtype == np.int64:
                    # Normalize to 0-255 range
                    grid_norm = ((grid - grid.min()) / (grid.max() - grid.min() + 1e-8) * 255).astype(np.uint8)
                    # Create RGB by repeating grayscale
                    img = Image.fromarray(grid_norm).convert('RGB')
                else:
                    img = Image.fromarray((grid * 255).astype(np.uint8)).convert('RGB')
                # Resize to standard 224x224 (ARC grids have variable sizes)
                img = img.resize((224, 224), Image.NEAREST)
            
            # 3. If text field - skip for now (would need text_renderer.py)
            elif sample.text is not None:
                # Create blank placeholder for text samples
                img = Image.new('RGB', (224, 224), color=(128, 128, 128))
            
            else:
                # Empty sample - create blank
                img = Image.new('RGB', (224, 224), color=(0, 0, 0))
            
            # Resize to standard 224x224 for consistency
            if img.size != (224, 224):
                img = img.resize((224, 224), Image.BILINEAR)
            
            # Convert to tensor
            input_img = to_tensor(img)
            
            # For output: check label field
            if sample.label is not None:
                if isinstance(sample.label, np.ndarray):
                    if sample.label.ndim == 2:  # Grid
                        label_grid = ((sample.label - sample.label.min()) / (sample.label.max() - sample.label.min() + 1e-8) * 255).astype(np.uint8)
                        output_img_pil = Image.fromarray(label_grid).convert('RGB')
                        output_img_pil = output_img_pil.resize((224, 224), Image.NEAREST)
                        output_img = to_tensor(output_img_pil)
                    else:
                        output_img_pil = Image.fromarray(sample.label.astype(np.uint8))
                        output_img_pil = output_img_pil.resize((224, 224), Image.BILINEAR)
                        output_img = to_tensor(output_img_pil)
                elif hasattr(sample.label, 'convert'):  # PIL Image
                    output_img_pil = sample.label.resize((224, 224), Image.BILINEAR)
                    output_img = to_tensor(output_img_pil)
                else:
                    output_img = input_img.clone()  # Fallback
            else:
                output_img = input_img.clone()  # Same as input
            
            input_images.append(input_img)
            output_images.append(output_img)
        
        # Stack into batches [B, 3, H, W]
        input_batch = torch.stack(input_images).cuda()
        output_batch = torch.stack(output_images).cuda()
        
        # Create batch dict in expected format
        batch = {
            'inputs': input_batch,  # Model expects 'inputs' key
            'targets': output_batch,
            'puzzle_identifiers': batch['puzzle_identifiers'].cuda(),
            'labels': torch.zeros(batch_size, dtype=torch.long).cuda()  # Placeholder
        }
    # Handle HESC capsules (tuple) vs dict batches
    elif isinstance(batch, tuple):
        # HESC mode: batch from TensorDataset
        if len(batch) == 3:
            # Capsules with children: (sketches, checksums, children)
            sketches, checksums, children = batch
            batch = {
                'inputs': sketches.cuda(),
                'capsule_sketches': sketches.cuda(),
                'capsule_checksums': checksums.cuda(),
                'capsule_children': children.cuda(),
                'labels': torch.zeros(sketches.shape[0], dtype=torch.long).cuda(),
                'puzzle_identifiers': torch.zeros(sketches.shape[0], dtype=torch.long).cuda(),
                'num_expansions': torch.zeros(sketches.shape[0], dtype=torch.long).cuda()
            }
        elif len(batch) == 2:
            # Capsules without children: (sketches, checksums)
            sketches, checksums = batch
            batch = {
                'inputs': sketches.cuda(),
                'capsule_sketches': sketches.cuda(),
                'capsule_checksums': checksums.cuda(),
                'labels': torch.zeros(sketches.shape[0], dtype=torch.long).cuda(),
                'puzzle_identifiers': torch.zeros(sketches.shape[0], dtype=torch.long).cuda(),
                'num_expansions': torch.zeros(sketches.shape[0], dtype=torch.long).cuda()
            }
        else:
            # Legacy semantic: (embeddings,)
            embeddings = batch[0].cuda()
            batch = {
                'inputs': embeddings,  # Changed from 'input' for consistency
                'labels': torch.zeros(embeddings.shape[0], dtype=torch.long).cuda(),
                'puzzle_identifiers': torch.zeros(embeddings.shape[0], dtype=torch.long).cuda()
            }
    else:
        # Standard token mode: batch is dict
        # Handle None values (e.g., missing children/checksums in capsule mode)
        batch = {k: v.cuda() if v is not None else None for k, v in batch.items()}
        
    # Add CapsuleState for expansion tracking (if enabled)
    # Works for both tuple and dict batch modes
    if hasattr(config.arch, 'enable_capsule_expansion') and config.arch.enable_capsule_expansion:
        if 'inputs' in batch and batch['inputs'].dim() == 3:
            from models.capsule_state import CapsuleState
            
            children = batch.get('capsule_children', None)
            checksums = batch.get('capsule_checksums', None)
            
            capsule_state = CapsuleState(
                sketches=batch['inputs'].clone(),
                children=children,
                checksums=checksums
            )
            batch['capsule_state'] = capsule_state

    # Init carry if it is None
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.original_model.initial_carry(batch)  # type: ignore

    # Task-Based Training (ALWAYS ENABLED - Single Unified Pipeline)
    # Model thinks for variable steps until task completion (ACT/PonderNet approach)
    max_thinking_steps = getattr(config, 'max_thinking_steps', 10)
    task_completion_threshold = getattr(config, 'task_completion_threshold', 0.95)
    use_amp = getattr(config, 'use_mixed_precision', True)
    
    # Think until completion or max steps
    total_loss = 0.0
    all_metrics = {}
    thinking_steps_used = 0
    task_completed = False
    
    for think_step in range(max_thinking_steps):
        if think_step == 0 and rank == 0:
            print(f"\n📊 GPU Memory at FIRST forward pass (step {train_state.step}):")
            print(f"  Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            print(f"  Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
            print(f"  Max Allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
            print(f"  Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated())/1e9:.2f} GB\n")
        
        # Use PyTorch native CPU offloading for activations (saves 3-4GB GPU memory)
        # This moves intermediate activations to pinned CPU memory during forward pass
        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            if use_amp:
                # Use mixed precision training
                if not hasattr(train_state, 'scaler'):
                    from torch.amp import GradScaler
                    train_state.scaler = GradScaler('cuda')
                
                from torch.amp import autocast
                with autocast('cuda'):
                    train_state.carry, step_loss, step_metrics, _, _ = train_state.model(
                        carry=train_state.carry, 
                        batch=batch, 
                        return_keys=[],
                        global_step=train_state.step
                    )
            else:
                train_state.carry, step_loss, step_metrics, _, _ = train_state.model(
                    carry=train_state.carry, 
                    batch=batch, 
                    return_keys=[],
                    global_step=train_state.step
                )
        
        for k, v in step_metrics.items():
            if k not in all_metrics:
                all_metrics[k] = v
            else:
                all_metrics[k] += v
        
        total_loss += step_loss
        thinking_steps_used += 1
        
        # Check task completion: Use halt probability from ACT
        # High halt = model is confident it finished reasoning
        if 'halt_prob' in step_metrics:
            halt_prob = step_metrics['halt_prob']
            if halt_prob > task_completion_threshold:
                task_completed = True
                break
        
        # Alternative: Check Q-value confidence
        if 'q_max' in step_metrics and 'q_std' in step_metrics:
            # High Q-value + low std = confident decision
            if step_metrics['q_max'] > 0.8 and step_metrics['q_std'] < 0.1:
                task_completed = True
                break
    
    # Average loss over thinking steps
    loss = total_loss / thinking_steps_used if thinking_steps_used > 0 else torch.tensor(0.0)
    
    # Average metrics
    for k in all_metrics:
        all_metrics[k] = all_metrics[k] / thinking_steps_used
    
    # Add task-based metrics
    all_metrics['thinking_steps'] = thinking_steps_used
    all_metrics['task_completed'] = float(task_completed)
    metrics = all_metrics
    
    # NaN Detection and Optimizer Fallback
    if torch.isnan(loss) or torch.isinf(loss):
        train_state.nan_count += 1
        
        if rank == 0:
            print(f"\n{'='*70}")
            print(f"  ⚠️  NaN/Inf DETECTED in loss at step {train_state.step}")
            print(f"  NaN count: {train_state.nan_count}/{config.nan_threshold_for_fallback}")
            print(f"  Loss value: {loss.item()}")
        
        if config.enable_optimizer_fallback and train_state.nan_count >= config.nan_threshold_for_fallback:
            if rank == 0:
                print(f"  🔄 Switching to AdamW optimizer (fallback)")
                print(f"{'='*70}\n")
            
            # Recreate optimizers with AdamW
            old_config_type = config.optimizer_type
            config.optimizer_type = "adamw"
            
            # Preserve optimizer state if possible
            for i, opt in enumerate(train_state.optimizers):
                if isinstance(opt, torch.optim.AdamW):
                    continue  # Already AdamW
                
                # Create new AdamW optimizer
                new_opt = torch.optim.AdamW(
                    opt.param_groups[0]['params'],
                    lr=train_state.optimizer_lrs[i],
                    weight_decay=config.weight_decay,
                    betas=(config.beta1, config.beta2)
                )
                train_state.optimizers[i] = new_opt
            
            train_state.nan_count = 0  # Reset counter
        
        # Skip this batch
        if rank == 0:
            print(f"  ⏭️  Skipping batch due to NaN/Inf")
            print(f"{'='*70}\n")
        return

    # Gradient Accumulation (Winner Strategy - Large Effective Batch)
    accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
    accumulation_idx = train_state.step % accumulation_steps
    
    # Scale loss by accumulation steps (accumulate gradients over multiple batches)
    scaled_loss = loss / accumulation_steps
    
    # Backward pass with mixed precision scaling
    if use_amp and hasattr(train_state, 'scaler'):
        # Scale loss and backward
        train_state.scaler.scale((1 / global_batch_size) * scaled_loss).backward()
    else:
        # Standard backward
        ((1 / global_batch_size) * scaled_loss).backward()

    # Allreduce
    if world_size > 1:
        for param in train_state.original_model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
    
    # Unscale gradients before clipping (for mixed precision)
    if use_amp and hasattr(train_state, 'scaler'):
        train_state.scaler.unscale_(train_state.optimizers[0])
    
    # Track gradients before clipping (for monitoring)
    grad_stats = None
    if gradient_monitor is not None and rank == 0:
        grad_stats = gradient_monitor.update()
    
    # Only step optimizer after accumulating N gradients (Winner Strategy)
    should_step = (accumulation_idx == accumulation_steps - 1)
    
    grad_norm = 0.0
    lr_this_step = None
    
    if should_step:
        # Gradient normalization: per-layer or global
        if getattr(config, 'enable_per_layer_grad_norm', False):
            # Per-layer gradient normalization (more fine-grained control)
            grad_norms_dict = per_layer_gradient_normalization(train_state.original_model, config)
            grad_norm = sum(grad_norms_dict.values()) / len(grad_norms_dict) if grad_norms_dict else 0.0
        else:
            # Global gradient clipping (default)
            grad_norm = torch.nn.utils.clip_grad_norm_(train_state.original_model.parameters(), max_norm=1.0)
                
        # Apply optimizer with mixed precision support
        for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
            lr_this_step = compute_lr(base_lr, config, train_state)

            for param_group in optim.param_groups:
                param_group['lr'] = lr_this_step
            
            # Use scaler.step() for mixed precision, or regular step()
            if use_amp and hasattr(train_state, 'scaler'):
                train_state.scaler.step(optim)
            else:
                optim.step()
            
            optim.zero_grad()
        
        # Update scaler for next iteration (mixed precision)
        if use_amp and hasattr(train_state, 'scaler'):
            train_state.scaler.update()

    # Reduce metrics
    if len(metrics):
        assert not any(getattr(v, 'requires_grad', False) for v in metrics.values())

        metric_keys = list(sorted(metrics.keys()))  # Sort keys to guarantee all processes use the same order.
        # Reduce and reconstruct
        metric_values = torch.stack([torch.as_tensor(metrics[k], device='cuda') if not isinstance(metrics[k], torch.Tensor) else metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}
            
            # Postprocess
            count = max(reduced_metrics["count"], 1)  # Avoid NaNs
            reduced_metrics = {f"train/{k}": v / (global_batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}

            reduced_metrics["train/lr"] = lr_this_step
            
            # Log gradient norms
            if getattr(config, 'enable_per_layer_grad_norm', False) and isinstance(grad_norm, dict):
                # Log per-component norms
                for key, value in grad_norms_dict.items():
                    reduced_metrics[f"train/{key}"] = value
                # Average for compatibility
                reduced_metrics["train/grad_norm"] = grad_norm
            else:
                # Global norm
                reduced_metrics["train/grad_norm"] = grad_norm if isinstance(grad_norm, float) else grad_norm.item()
            
            # Add gradient flow statistics if available
            if grad_stats is not None:
                for key, value in grad_stats.items():
                    reduced_metrics[f"train/grad/{key}"] = value
            
            # Periodic codebook maintenance (prevent collapse)
            if hasattr(train_state.model, 'model') and hasattr(train_state.model.model, 'inner'):
                inner = train_state.model.model.inner
                if hasattr(inner, 'lm_head') and hasattr(inner.lm_head, 'codebook'):
                    codebook = inner.lm_head.codebook
                    
                    # Reset dead codes every 5000 steps
                    if train_state.step % 5000 == 0 and train_state.step > 0:
                        codebook.reset_dead_codes(min_usage=0.01)
                    
                    # Log usage stats every 1000 steps
                    if train_state.step % 1000 == 0:
                        stats = codebook.get_usage_stats()
                        for key, value in stats.items():
                            reduced_metrics[f"train/codebook/{key}"] = value
            
            return reduced_metrics

def evaluate(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
    rank: int,
    world_size: int,
    cpu_group: Optional[dist.ProcessGroup],
):
    reduced_metrics = None

    # Skip evaluation if no eval data
    if eval_metadata is None:
        if rank == 0:
            print("Skipping evaluation: no eval data available")
        return {}

    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)

        # Run evaluation
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}

        save_preds = {}

        metric_keys = []
        metric_values = None

        carry = None
        processed_batches = 0
        
        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            if rank == 0:
                print(f"Processing batch {processed_batches}: {set_name}")
            
            # Handle capsule mode (same as train_batch)
            if isinstance(batch, tuple):
                if len(batch) == 3:
                    sketches, checksums, children = batch
                    batch = {
                        'inputs': sketches.cuda(),
                        'capsule_sketches': sketches.cuda(),
                        'capsule_checksums': checksums.cuda(),
                        'capsule_children': children.cuda(),
                        'labels': torch.zeros(sketches.shape[0], dtype=torch.long).cuda(),
                        'puzzle_identifiers': torch.zeros(sketches.shape[0], dtype=torch.long).cuda()
                    }
                elif len(batch) == 2:
                    sketches, checksums = batch
                    batch = {
                        'inputs': sketches.cuda(),
                        'capsule_sketches': sketches.cuda(),
                        'capsule_checksums': checksums.cuda(),
                        'labels': torch.zeros(sketches.shape[0], dtype=torch.long).cuda(),
                        'puzzle_identifiers': torch.zeros(sketches.shape[0], dtype=torch.long).cuda()
                    }
                else:
                    embeddings = batch[0].cuda()
                    batch = {
                        'inputs': embeddings,
                        'labels': torch.zeros(embeddings.shape[0], dtype=torch.long).cuda(),
                        'puzzle_identifiers': torch.zeros(embeddings.shape[0], dtype=torch.long).cuda()
                    }
            else:
                # Standard dict mode
                batch = {k: v.cuda() for k, v in batch.items()}
                
                # Add CapsuleState for expansion tracking (if enabled)
                if hasattr(config.arch, 'enable_capsule_expansion') and config.arch.enable_capsule_expansion:
                    if 'inputs' in batch and batch['inputs'].dim() == 3:  # [B, k, D]
                        from models.capsule_state import CapsuleState
                        
                        # Create CapsuleState with available components
                        children = batch.get('capsule_children', None)  # [B, k, m, D] if available
                        checksums = batch.get('capsule_checksums', None)  # [B, k, R] if available
                        
                        capsule_state = CapsuleState(
                            sketches=batch['inputs'].clone(),  # [B, k, D]
                            children=children,  # Full expansion support if available
                            checksums=checksums,  # Reconstructability tracking
                        )
                        batch['capsule_state'] = capsule_state
            with torch.device("cuda"):
                carry = train_state.original_model.initial_carry(batch)  # type: ignore

            # Forward
            inference_steps = 0
            while True:
                carry, loss, metrics, preds, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                inference_steps += 1

                if all_finish:
                    break

            if rank == 0:
                print(f"  Completed inference in {inference_steps} steps")

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        save_preds.setdefault(k, [])
                        save_preds[k].append(v.cpu())  # Move to CPU for saving GPU memory

            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)

            del carry, loss, preds, batch, all_finish

            # Aggregate metrics
            set_id = set_ids[set_name]

            if metric_values is None:
                metric_keys = list(
                    sorted(metrics.keys())
                )  # Sort keys to guarantee all processes use the same order.
                metric_values = torch.zeros(
                    (len(set_ids), len(metrics.values())), dtype=torch.float32, device="cuda"
                )

            metric_values[set_id] += torch.stack([torch.as_tensor(metrics[k], device='cuda') if not isinstance(metrics[k], torch.Tensor) else metrics[k] for k in metric_keys])

            del metrics

        # concatenate save preds
        save_preds = {k: torch.cat(v, dim=0) for k, v in save_preds.items()}

        # Save preds
        if config.checkpoint_path is not None and len(save_preds):
            # Each rank save predictions independently
            os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
            torch.save(
                save_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}")
            )

        del save_preds

        # Reduce to rank 0
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {
                        metric_name: reduced_metrics[set_id, metric_id]
                        for metric_id, metric_name in enumerate(metric_keys)
                    }
                    for set_id, set_name in enumerate(set_ids)
                }

                # Postprocess
                for set_name, m in reduced_metrics.items():
                    count = m.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in m.items()}

        # Run evaluators
        if rank == 0:
            print(f"\nRunning {len(evaluators)} evaluator(s)...")
            
        for i, evaluator in enumerate(evaluators):
            if rank == 0:
                print(f"Running evaluator {i+1}/{len(evaluators)}: {evaluator.__class__.__name__}")
                
            # Path for saving
            evaluator_save_path = None
            if config.checkpoint_path is not None:
                evaluator_save_path = os.path.join(
                    config.checkpoint_path,
                    f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}",
                )
                os.makedirs(evaluator_save_path, exist_ok=True)

            # Run and log
            metrics = evaluator.result(evaluator_save_path, rank=rank, world_size=world_size, group=cpu_group)
            if rank == 0 and metrics is not None:
                if reduced_metrics is None:
                    reduced_metrics = {}

                reduced_metrics.update(metrics)
                print(f"  Completed {evaluator.__class__.__name__}")
                
        if rank == 0:
            print("All evaluators completed!")

    return reduced_metrics

def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None or wandb.run is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)

            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    # Log code
    wandb.run.log_code(config.checkpoint_path)


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore

        # Naming
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_paths[0]).capitalize()}-ACT-torch"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1
    CPU_PROCESS_GROUP = None

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        
        # CPU GLOO process group
        CPU_PROCESS_GROUP = dist.new_group(backend="gloo")
        assert (
            dist.get_rank(CPU_PROCESS_GROUP) == RANK and dist.get_world_size(CPU_PROCESS_GROUP) == WORLD_SIZE
        )

    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)
    
    # Validate critical config parameters
    if RANK == 0:
        assert 0 < config.epochs <= 1_000_000, f"Invalid epochs: {config.epochs}"
        assert 1 <= config.global_batch_size <= 8192, f"Invalid batch_size: {config.global_batch_size}"
        assert 0 < config.lr <= 1.0, f"Invalid learning rate: {config.lr}"
        assert config.lr_min_ratio >= 0, f"Invalid lr_min_ratio: {config.lr_min_ratio}"
        
        if config.eval_interval is not None:
            assert config.epochs % config.eval_interval == 0, \
                f"eval_interval ({config.eval_interval}) must divide epochs ({config.epochs})"

    # Register signal handler for graceful shutdown (only on rank 0 to avoid duplicate messages)
    if RANK == 0:
        signal.signal(signal.SIGINT, shutdown_signal_handler)
        print("\n" + "="*70)
        print("  📌 Graceful shutdown enabled")
        print("  💡 Press Ctrl+C to stop training and save checkpoint")
        print("="*70 + "\n")

    # Auto-resume: Check if checkpoint exists and auto-load if not specified
    if config.load_checkpoint is None and config.checkpoint_path is not None:
        # Look for dataset-specific checkpoint (e.g., text.pt, arc.pt)
        checkpoint_name = get_checkpoint_name(config.data_paths[0] if config.data_paths else 'model')
        checkpoint_file = os.path.join(config.checkpoint_path, checkpoint_name)
        
        # Fallback to latest.pt for backward compatibility
        if not os.path.exists(checkpoint_file):
            checkpoint_file = os.path.join(config.checkpoint_path, "latest.pt")
        
        if os.path.exists(checkpoint_file):
            if RANK == 0:
                print(f"\n{'='*70}")
                print(f"  🔄 AUTO-RESUME: Found existing checkpoint")
                print(f"  📁 Loading from: {checkpoint_file}")
                print(f"{'='*70}\n")
            config.load_checkpoint = checkpoint_file
    
    # Initialize training progress tracker for consolidated dataset chunks
    progress_tracker = None
    if config.checkpoint_path is not None and RANK == 0:
        dataset_name = get_checkpoint_name(config.data_paths[0] if config.data_paths else 'model').replace('.pt', '')
        progress_tracker = TrainingProgressTracker(
            checkpoint_dir=Path(config.checkpoint_path),
            dataset_name=dataset_name
        )
        
        # Load existing progress if resuming
        if progress_tracker.has_progress():
            progress_tracker.load()
            
            # Show dataset manifest
            if config.data_paths:
                # Extract checkpoint directory from data path
                data_path = Path(config.data_paths[0])
                checkpoint_dir = data_path.parent / 'stream_checkpoints'
                if checkpoint_dir.exists():
                    manifest = get_training_manifest(checkpoint_dir)
                    print(f"\n📦 Dataset Manifest:")
                    print(f"   Available chunks: {len(manifest['available_chunks'])}")
                    print(f"   Chunks IDs: {manifest['available_chunks'][:10]}..." if len(manifest['available_chunks']) > 10 else f"   Chunk IDs: {manifest['available_chunks']}")
    
    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed + RANK)

    # Dataset
    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    train_loader, train_metadata = load_datasets(config, rank=RANK, world_size=WORLD_SIZE, split="train")
    try:
        eval_loader,  eval_metadata  = load_datasets(config, rank=RANK, world_size=WORLD_SIZE, split="test")
    except Exception as e:
        if RANK == 0:
            print(f"⚠️  No eval data found: {e}")
            print("   Training will continue without evaluation")
        eval_loader = eval_metadata = None

    try:
        evaluators = create_evaluators(config, eval_metadata)
    except Exception as e:
        if RANK == 0:
            print(f"⚠️  Failed to create evaluators: {e}")
            print("   Training will continue without custom evaluators")
        evaluators = []

    # Train state
    train_state = init_train_state(config, train_metadata, rank=RANK, world_size=WORLD_SIZE)

    # Progress bar and logger
    progress_bar = None
    ema_helper = None
    gradient_monitor = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)
        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump(), settings=wandb.Settings(_disable_stats=True))  # type: ignore
        
        # Count parameters from original model (before torch.compile)
        num_params = sum(x.numel() for x in train_state.original_model.parameters())
        wandb.log({"num_params": num_params}, step=0)
        print(f"   Total parameters: {num_params:,}")
        save_code_and_config(config)
        
        # Initialize gradient flow monitor
        gradient_monitor = GradientFlowMonitor(train_state.model, track_every_n_steps=100)
        print('Initialized gradient flow monitor (tracking every 100 steps)')
        
    if config.ema:
        print('Setup EMA')
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.original_model)  # Use unwrapped model for EMA

    # Training Loop
    # Calculate starting iteration from restored step (for auto-resume)
    steps_per_iter = train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size
    start_iter = int(train_state.step / steps_per_iter) if train_state.step > 0 else 0
    
    if RANK == 0 and start_iter > 0:
        print(f"\n{'='*70}")
        print(f"  📍 Resuming from iteration {start_iter}/{total_iters}")
        print(f"  📍 Epoch: {start_iter * train_epochs_per_iter}/{config.epochs}")
        print(f"  📍 Step: {train_state.step}/{train_state.total_steps}")
        print(f"{'='*70}\n")
    
    # Graceful shutdown handler
    try:
        for _iter_id in range(start_iter, total_iters):
            # Check for shutdown request at start of each iteration
            if shutdown_requested:
                if RANK == 0:
                    print("\n" + "="*70)
                    print("  💾 GRACEFUL SHUTDOWN IN PROGRESS")
                    print("="*70)
                    print(f"\n  Saving checkpoint at step {train_state.step}...")
                    save_train_state(config, train_state, progress_tracker)
                    print("\n✅ Training stopped successfully!")
                    print(f"   Final step: {train_state.step}/{train_state.total_steps}")
                    print(f"   Progress: {(train_state.step/train_state.total_steps)*100:.1f}%")
                    print(f"   Checkpoint saved to: {config.checkpoint_path}")
                    print("\n💡 Resume training with the same command.\n")
                
                # Clean exit
                if dist.is_initialized():
                    dist.destroy_process_group()
                if RANK == 0:
                    wandb.finish()
                return  # Exit gracefully
            
            print (f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}")

            ############ Train Iter
            if RANK == 0:
                print("TRAIN")
            train_state.original_model.train()  # Use unwrapped model for .train()
            for set_name, batch, global_batch_size in train_loader:
                metrics = train_batch(config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE, gradient_monitor=gradient_monitor)
                
                # Periodic memory cleanup to prevent leaks
                if RANK == 0 and train_state.step % 1000 == 0 and gradient_monitor is not None:
                    gradient_monitor.clear_old_data(keep_last_n=100)
                    torch.cuda.empty_cache()

                if RANK == 0 and metrics is not None:
                    wandb.log(metrics, step=train_state.step)
                    progress_bar.update(train_state.step - progress_bar.n)  # type: ignore
                if config.ema:
                    ema_helper.update(train_state.model)

            if _iter_id >= config.min_eval_interval:
                ############ Evaluation
                if RANK == 0:
                    print("EVALUATE")
                if config.ema:
                    print("SWITCH TO EMA")
                    train_state_eval = copy.deepcopy(train_state)
                    train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
                else:
                    train_state_eval = train_state
                train_state_eval.original_model.eval()  # Use unwrapped model for .eval()
                metrics = evaluate(config, 
                    train_state_eval, 
                    eval_loader, 
                    eval_metadata, 
                    evaluators,
                    rank=RANK, 
                    world_size=WORLD_SIZE,
                    cpu_group=CPU_PROCESS_GROUP)

                if RANK == 0 and metrics is not None:
                    wandb.log(metrics, step=train_state.step)
                    
                ############ Checkpointing
                if RANK == 0:
                    print("SAVE CHECKPOINT")
                if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
                    save_train_state(config, train_state_eval, progress_tracker)

                if config.ema:
                    del train_state_eval
    
    except KeyboardInterrupt:
        if RANK == 0:
            print("\n" + "="*70)
            print("  🛑 TRAINING INTERRUPTED (Ctrl+C)")
            print("="*70)
            
            # Calculate lost progress
            lost_steps = train_state.step - train_state.last_checkpoint_step
            progress_pct = (train_state.last_checkpoint_step / train_state.total_steps) * 100 if train_state.total_steps > 0 else 0
            
            print("\n⚠️  NOT saving current state (may be unsafe during training)")
            print("\n📍 Checkpoint Status:")
            print(f"   Last safe checkpoint: Step {train_state.last_checkpoint_step:,}")
            print(f"   Current training step: Step {train_state.step:,}")
            print(f"   Lost progress: {lost_steps:,} steps")
            print(f"   Saved progress: {progress_pct:.1f}% of total training")
            
            if config.checkpoint_path and config.data_paths:
                checkpoint_name = get_checkpoint_name(config.data_paths[0])
                print(f"\n💾 Last checkpoint: {config.checkpoint_path}/{checkpoint_name}")
            
            print("\n💡 Resume training with the same command to continue from last checkpoint.")
            print("\n" + "="*70)
            print("  👋 Training stopped by user")
            print("="*70 + "\n")
        
        # Clean up
        if dist.is_initialized():
            dist.destroy_process_group()
        if RANK == 0:
            wandb.finish()
        
        sys.exit(0)

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()
