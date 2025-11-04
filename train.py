"""
Unified training script for TRM models.
Supports: Text, Visual (ARC), Coding, and Multi-Modal.

Features:
  - Auto-resume: Automatically continues from latest checkpoint
  - Auto-dataset: Builds datasets if missing
  - Multi-modal: Text, visual, code generation

Usage:
    python train.py                  # Interactive mode + auto-resume
    python train.py --model text     # Train text model (auto-resumes if checkpoint exists)
    python train.py --model arc      # Train ARC model
    python train.py --model code     # Train code model
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse
import torch


# Vision-Unified: Single checkpoint path
CHECKPOINT_DIR = Path("checkpoints/multimodal-hesc")
CHECKPOINT_FILE = CHECKPOINT_DIR / "latest.pt"

# Vision-Unified: Single model for all modalities
# Text → rendered to images → TRM encoder → capsules
# Images → TRM encoder → capsules
# Grids → rendered to images → TRM encoder → capsules
MODELS = {
    "vision-unified": {
        "name": "TRM Vision-Unified (All Modalities)",
        "config": "cfg_multimodal",
        "dataset_builder": "dataset/build_multimodal_dataset.py",
        "dataset_args": [
            "--sources", "kaggle/combined", "tinystories",
            "--output-dir", "datasets/vision_unified",
            "--augment"
        ],
        "description": (
            "TRM Vision-Unified Pipeline (5M encoder + 5M reasoner):\n"
            "  • ARC puzzles: Spatial reasoning (kaggle/combined)\n"
            "  • TinyStories: Simple fluent text generation\n"
            "  • All → rendered to images → TRM encoder → 12 capsules\n"
            "  • Architecture: Encoder(2L, H=2, L=3), Reasoner(H=3, L=2)\n"
            "  • Features: DQN, Memory Bank, MTP\n"
            "  • 30× smaller than CLIP, 17× faster"
        )
    }
}


def check_checkpoint_exists() -> bool:
    """Check if vision-unified checkpoint exists."""
    return CHECKPOINT_FILE.exists()


def get_checkpoint_info() -> dict:
    """Get info about vision-unified checkpoint if it exists."""
    if not CHECKPOINT_FILE.exists():
        return None
    
    try:
        checkpoint = torch.load(CHECKPOINT_FILE, map_location='cpu')
        return {
            'epoch': checkpoint.get('epoch', 0),
            'step': checkpoint.get('step', 0),
            'path': str(CHECKPOINT_FILE)
        }
    except Exception as e:
        print(f"   ⚠️  Warning: Could not read checkpoint: {e}")
        return None


def dataset_path_to_model_key(dataset_path: str) -> str:
    """Infer model key from dataset path.
    
    Examples:
        datasets/wikitext2 -> text
        data/vision-cifar10 -> vision
        data/arc-capsules -> arc
        datasets/multimodal_unified -> multimodal
    """
    dataset_name = dataset_path.split('/')[-1] if '/' in dataset_path else dataset_path
    
    # Direct mapping (only 4 models)
    if 'wikitext' in dataset_name or 'text' in dataset_name:
        return 'text'
    elif 'arc' in dataset_name:
        return 'arc'
    elif 'cifar' in dataset_name or 'vision' in dataset_name:
        return 'vision'
    elif 'multimodal' in dataset_name or 'unified' in dataset_name:
        return 'multimodal'
    else:
        # Fallback: return first matching key
        for key in MODELS.keys():
            if key in dataset_name:
                return key
        return 'text'  # Default fallback


def recommend_next_phase(completed_datasets):
    """Recommend next training phase based on roadmap."""
    for phase in COMMUNICATION_ROADMAP:
        dataset = phase["dataset"]
        # Check if this phase's prerequisite is completed
        prereq = phase["continue_from"]
        
        # Phase 1 has no prereq
        if prereq is None:
            if dataset not in completed_datasets:
                return phase
        # Check if prereq is done but this phase isn't
        elif prereq in completed_datasets and dataset not in completed_datasets:
            return phase
    
    return None  # All phases completed!


# Removed: Old continual learning system
# Vision-unified uses simple auto-resume from checkpoints/multimodal-hesc/latest.pt

def check_dataset_exists(output_dir: str) -> bool:
    """Check if dataset already exists."""
    train_path = Path(output_dir) / "train" / "dataset.json"
    return train_path.exists()


def wait_for_consolidated_chunks(output_dir: str, poll_interval: int = 60):
    """Wait for consolidated chunks and return list of available chunks.
    
    Args:
        output_dir: Dataset output directory
        poll_interval: Seconds to wait between checks for new chunks
    
    Returns:
        List of available consolidated_*.pt files
    """
    import time
    from pathlib import Path
    
    checkpoint_dir = Path(output_dir) / "stream_checkpoints"
    
    # Wait for first chunk
    first_chunk = checkpoint_dir / "consolidated_000.pt"
    print("\n⏳ Waiting for first consolidated chunk...")
    print(f"   Watching: {checkpoint_dir}")
    print(f"   Encoding is running in parallel (check other output)\n")
    
    while not first_chunk.exists():
        time.sleep(30)  # Check every 30 seconds
    
    print(f"✅ First chunk ready: {first_chunk.name}")
    
    # Return all available chunks
    chunks = sorted(checkpoint_dir.glob("consolidated_*.pt"))
    size_gb = sum(c.stat().st_size for c in chunks) / (1024**3)
    print(f"📦 Found {len(chunks)} chunk(s) ({size_gb:.2f}GB)\n")
    
    return chunks

def download_and_build_dataset(model_config: dict, force_rebuild: bool = False, incremental: bool = True):
    """Download dataset and build preprocessed files.
    
    Args:
        incremental: If True, wait for consolidated chunks and start training early
    """
    dataset_args = model_config['dataset_args']
    
    # Handle both list and dict formats
    if isinstance(dataset_args, list):
        # Convert list format to dict
        args_dict = {}
        i = 0
        while i < len(dataset_args):
            if dataset_args[i].startswith('--'):
                key = dataset_args[i][2:].replace('-', '_')  # Remove -- and normalize
                if i + 1 < len(dataset_args) and not dataset_args[i + 1].startswith('--'):
                    args_dict[key] = dataset_args[i + 1]
                    i += 2
                else:
                    args_dict[key] = True
                    i += 1
            else:
                i += 1
        dataset_args = args_dict
    
    output_dir = dataset_args.get('output_dir', 'datasets/default')
    
    # Check if already exists (fully built)
    if check_dataset_exists(output_dir) and not incremental:
        if not force_rebuild:
            print(f"\nDataset found at: {output_dir}")
            print("   Skipping dataset building (use --rebuild-dataset to force rebuild)")
            return True
        else:
            print(f"\n🔄 Rebuilding dataset at: {output_dir}")
    
    # Incremental mode: Check for consolidated chunks
    if incremental:
        checkpoint_dir = Path(output_dir) / "stream_checkpoints"
        if checkpoint_dir.exists():
            chunks = list(checkpoint_dir.glob("consolidated_*.pt"))
            if chunks:
                print(f"\n⚡ INCREMENTAL MODE: Found {len(chunks)} consolidated chunks")
                print(f"   Training will start on available data while encoding continues")
                return True  # Dataset partially ready
        
        print(f"\n⚡ INCREMENTAL MODE: Dataset encoding in progress")
        print(f"   Training will start as soon as first chunk is ready")
        return True  # Will wait for chunks in training phase
    
    print(f"\n📦 Building dataset: {output_dir}")
    print("-" * 70)
    
    # Build command
    builder_script = model_config['dataset_builder']
    args = model_config['dataset_args']
    
    cmd = ["python", builder_script]
    
    # Add subcommand if using unified builder
    if 'dataset_command' in model_config:
        cmd.append(model_config['dataset_command'])
    
    # Convert list to dict, collecting multiple values for same key
    if isinstance(args, list):
        args_dict = {}
        i = 0
        while i < len(args):
            if args[i].startswith('--'):
                key = args[i][2:].replace('-', '_')
                values = []
                i += 1
                # Collect all non-flag values
                while i < len(args) and not args[i].startswith('--'):
                    values.append(args[i])
                    i += 1
                
                if values:
                    # Multiple values = list, single value = string
                    args_dict[key] = values if len(values) > 1 else values[0]
                else:
                    args_dict[key] = True
            else:
                i += 1
        args = args_dict
    
    # Build arguments
    for key, value in args.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        elif isinstance(value, list):
            # Multiple values for same flag
            cmd.append(flag)
            cmd.extend(value)
        else:
            cmd.extend([flag, str(value)])
    
    print(f"Running: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        if result.returncode == 0:
            print("\n✅ Dataset built successfully!")
            return True
        else:
            print(f"\n❌ Dataset building failed with code {result.returncode}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Dataset building failed: {e}")
        return False
    except FileNotFoundError:
        print(f"\n❌ Dataset builder not found: {builder_script}")
        return False


def _extract_output_dir_from_args(args_list: list) -> str:
    """Extract output directory from dataset_args command-line arguments list.
    
    Args:
        args_list: List of command-line arguments (e.g., ['--output-dir', 'datasets/vision_unified', ...])
    
    Returns:
        output_dir: The output directory path, or None if not found
    """
    try:
        # Find --output-dir flag and get the next element
        for i, arg in enumerate(args_list):
            if arg == '--output-dir' and i + 1 < len(args_list):
                return args_list[i + 1]
    except (IndexError, AttributeError):
        pass
    return None


def train_model(model_config: dict, extra_args: list, checkpoint_path: str = None, epochs_override: int = None):
    """Start training with specified config."""
    config_name = model_config['config']
    
    # Extract data path from dataset_args (which is a list of CLI arguments)
    dataset_args = model_config.get('dataset_args', [])
    if isinstance(dataset_args, dict):
        # Legacy dict format support
        data_path = dataset_args.get('output_dir')
    elif isinstance(dataset_args, list):
        # New list format (command-line args)
        data_path = _extract_output_dir_from_args(dataset_args)
    else:
        data_path = None
    
    if not data_path:
        print("\n❌ Error: Could not determine dataset output directory")
        print(f"   dataset_args: {dataset_args}")
        sys.exit(1)
    
    print(f"\n🚀 Starting training: {model_config['name']}")
    print("-" * 70)
    print(f"Config: {config_name}")
    print(f"Data: {data_path}")
    if checkpoint_path:
        print(f"Continue from: {checkpoint_path}")
    if epochs_override:
        print(f"Epochs: {epochs_override}")
    print("-" * 70)
    print()
    
    # Build training command with Hydra overrides
    cmd = [
        "python", "pretrain.py",
        f"--config-name={config_name}",
        f"data_paths=[{data_path}]"  # Override data path
    ]
    
    # Add continual learning parameters (use + prefix for Hydra)
    if checkpoint_path:
        cmd.append(f"+load_checkpoint={checkpoint_path}")
    if epochs_override:
        cmd.append(f"epochs={epochs_override}")
    
    cmd += extra_args
    
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        # Run training (will stream output)
        # KeyboardInterrupt will propagate to pretrain.py for graceful shutdown
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)


def main():
    """Simplified training entry point - just calls pretrain.py with config."""
    parser = argparse.ArgumentParser(
        description="TRM Training - Unified Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py                  Usage:
  python train.py                   # Train vision-unified model (default)
  python train.py --rebuild-dataset # Force rebuild dataset
  python train.py --dataset-only    # Only build dataset, don't train

Auto-Resume:
  Training automatically resumes from checkpoints/vision-unified/latest.pt if it exists.
  No need to manually specify --load-checkpoint!
        """
    )
    
    parser.add_argument(
        "--rebuild-dataset",
        action="store_true",
        help="Force rebuild dataset even if it exists"
    )
    parser.add_argument(
        "--dataset-only",
        action="store_true",
        help="Only build dataset, don't train"
    )
    parser.add_argument(
        "--no-incremental",
        action="store_true",
        help="Disable incremental training (wait for full dataset)"
    )
    
    args, unknown_args = parser.parse_known_args()
    
    print("="*70)
    print("  TRM Training Pipeline - Auto-Starting")
    print("  Vision-Unified Model: Text + Images + Puzzles")
    print("="*70)
    print()
    
    # Auto-select vision-unified model (no user prompts)
    selected_model = "vision-unified"
    model_config = MODELS[selected_model]
    
    # Dataset preparation
    incremental_mode = not args.no_incremental
    success = download_and_build_dataset(model_config, force_rebuild=args.rebuild_dataset, incremental=incremental_mode)
    if not success:
        print("\n❌ Cannot proceed without dataset. Exiting.")
        sys.exit(1)
    
    if args.dataset_only:
        print("\n✅ Dataset preparation complete. Exiting (--dataset-only).")
        sys.exit(0)
    
    # Incremental mode: Wait for consolidated chunks
    if incremental_mode:
        dataset_args = model_config.get('dataset_args', [])
        if isinstance(dataset_args, list):
            output_dir = _extract_output_dir_from_args(dataset_args)
        else:
            output_dir = dataset_args.get('output_dir')
        
        if output_dir:
            chunks = wait_for_consolidated_chunks(output_dir)
            print(f"🚀 Starting training with {len(chunks)} chunk(s)")
            print(f"   Encoding continues in background")
            print(f"   Training will use progressively more data as chunks complete\n")
    
    # Auto-resume from checkpoint if it exists
    checkpoint_info = get_checkpoint_info()
    checkpoint_path = None
    
    if checkpoint_info:
        print(f"\n📂 Found checkpoint: {checkpoint_info['path']}")
        print(f"   Epoch: {checkpoint_info['epoch']}, Step: {checkpoint_info['step']}")
        print("   Training will auto-resume from this checkpoint\n")
        checkpoint_path = checkpoint_info['path']
    else:
        print("\n🆕 No checkpoint found - starting fresh training\n")
    
    # Start training
    train_model(model_config, unknown_args, checkpoint_path=checkpoint_path, epochs_override=None)
    
    print("\n" + "=" * 70)
    print("  ✅ Training completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
