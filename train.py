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


# Communication Training Roadmap (optimized for chat quality)
COMMUNICATION_ROADMAP = [
    {
        "phase": 1,
        "name": "Foundation (Language Structure)",
        "dataset": "text",
        "epochs": 5000,
        "time_hours": 9,
        "goal": "Learn formal English, grammar, vocabulary",
        "continue_from": None
    },
    {
        "phase": 2,
        "name": "Fluency (Conversational Flow)",
        "dataset": "text-tiny",
        "epochs": 2000,
        "time_hours": 12,
        "goal": "Simple, fluent generation (children's stories)",
        "continue_from": "text"
    },
    {
        "phase": 3,
        "name": "Instruction Following (Q&A)",
        "dataset": "alpaca",
        "epochs": 1000,
        "time_hours": 6,
        "goal": "Learn instruction→response patterns",
        "continue_from": "text-tiny"
    },
    {
        "phase": 4,
        "name": "Conversational Polish",
        "dataset": "sharegpt",
        "epochs": 500,
        "time_hours": 8,
        "goal": "Multi-turn dialogue",
        "continue_from": "alpaca"
    }
]

MODELS = {
    "text": {
        "name": "Text Generation (WikiText-2)",
        "config": "cfg_text",
        "dataset_builder": "dataset/build_multimodal_dataset.py",
        "dataset_command": "build_text",
        "dataset_args": {
            "input_file": "wikitext2",
            "output_dir": "datasets/wikitext2",
            "num_concepts": 2048,
            "target_capsules": 12
        },
        "description": "Language modeling with HESC capsules (H=4, L=3, 16 recursions)"
    },
    "vision": {
        "name": "Vision Classification (CIFAR-10)",
        "config": "cfg_vision",
        "dataset_builder": "dataset/build_multimodal_dataset.py",
        "dataset_command": "build_image",
        "dataset_args": {
            "dataset_name": "cifar10",
            "output_dir": "data/vision-cifar10",
            "image_size": 224
        },
        "description": "Image classification with spatial reasoning (H=3, L=3, 12 recursions)"
    },
    "arc": {
        "name": "ARC-AGI Reasoning (Geometric Puzzles)",
        "config": "cfg_pretrain",
        "dataset_builder": "dataset/build_multimodal_dataset.py",
        "dataset_command": "build_arc",
        "dataset_args": {
            "input_file_prefix": "kaggle/combined/arc-agi",
            "output_dir": "data/arc-capsules",
            "subsets": ["training", "training2"],
            "test_set_name": "evaluation",
            "num_aug": 1000,
            "seed": 42
        },
        "description": "Visual reasoning on 30×30 grids (H=3, L=6, 21 recursions - TRM paper optimal)"
    },
    "multimodal": {
        "name": "Multimodal Unified (Text + Vision + ARC)",
        "config": "cfg_multimodal",
        "dataset_builder": "dataset/build_multimodal_dataset.py",
        "dataset_command": "build_composite",
        "dataset_args": {
            "sources": [
                "kaggle/combined/arc-agi_training.json",
                "wikitext2",
                "cifar10"
            ],
            "output_dir": "datasets/multimodal_unified",
            "augment": True,
            "num_concepts": 2048,
            "target_capsules": 12,
            "enable_quality_scoring": True
        },
        "description": "Cross-modal transfer learning (H=3, L=2, 9 recursions for 12-capsule compression)"
    }
}


def check_phase_completion(checkpoint_path: str) -> dict:
    """Check if training phase is complete."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        current_step = checkpoint.get('step', 0)
        target_steps = checkpoint.get('total_steps', 0)
        config_epochs = checkpoint.get('config_epochs', 0)
        dataset = checkpoint.get('dataset', 'unknown')
        
        if target_steps > 0:
            progress = (current_step / target_steps) * 100
            is_complete = current_step >= target_steps
        else:
            # Legacy checkpoint without metadata - assume incomplete
            progress = 0
            is_complete = False
        
        return {
            'current_step': current_step,
            'target_steps': target_steps,
            'config_epochs': config_epochs,
            'dataset': dataset,
            'progress_pct': progress,
            'is_complete': is_complete,
            'corrupted': False
        }
    except Exception as e:
        error_msg = str(e)
        is_corrupted = 'PytorchStreamReader' in error_msg or 'zip archive' in error_msg
        
        if is_corrupted:
            print(f"   ❌ CORRUPTED CHECKPOINT: {e}")
        else:
            print(f"   ⚠️  Could not read checkpoint metadata: {e}")
        
        return {
            'is_complete': False, 
            'progress_pct': 0,
            'corrupted': is_corrupted,
            'error': error_msg
        }


def detect_checkpoints():
    """Detect all available checkpoints in checkpoints/ directory.
    
    Looks for dataset-specific .pt files (text.pt, arc.pt, etc) or legacy latest.pt.
    """
    checkpoint_dir = Path("checkpoints")
    if not checkpoint_dir.exists():
        return []
    
    checkpoints = []
    for subdir in checkpoint_dir.iterdir():
        if subdir.is_dir():
            # Look for any .pt files in the directory
            pt_files = list(subdir.glob("*.pt"))
            
            if pt_files:
                # Prefer dataset-specific names over latest.pt
                checkpoint_file = None
                for pt_file in pt_files:
                    if pt_file.name != "latest.pt":
                        checkpoint_file = pt_file
                        break
                
                # Fallback to latest.pt if no dataset-specific file found
                if checkpoint_file is None:
                    checkpoint_file = subdir / "latest.pt"
                    if not checkpoint_file.exists():
                        continue
                
                # Check completion status
                status = check_phase_completion(str(checkpoint_file))
                
                # Extract dataset name from checkpoint filename
                dataset_name = checkpoint_file.stem  # e.g., "text" from "text.pt"
                
                checkpoints.append({
                    "name": f"{subdir.name} ({dataset_name})",
                    "path": str(checkpoint_file),
                    "dir": str(subdir),
                    "status": status
                })
    return checkpoints


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


def print_roadmap():
    """Print the communication training roadmap."""
    print("\n" + "="*70)
    print("  🗺️  COMMUNICATION TRAINING ROADMAP")
    print("="*70)
    print("\nOptimal path for chat model (total ~35 hours):\n")
    
    total_hours = 0
    for phase in COMMUNICATION_ROADMAP:
        print(f"Phase {phase['phase']}: {phase['name']}")
        print(f"  Dataset: {phase['dataset']}")
        print(f"  Epochs: {phase['epochs']}")
        print(f"  Time: ~{phase['time_hours']} hours")
        print(f"  Goal: {phase['goal']}")
        if phase['continue_from']:
            print(f"  Continue from: Phase {phase['phase']-1} ({phase['continue_from']})")
        print()
        total_hours += phase['time_hours']
    
    print(f"Total training time: ~{total_hours} hours")
    print("="*70 + "\n")


def print_banner():
    """Print welcome banner."""
    print("=" * 70)
    print("  🧠 TRM Training Pipeline - Unified Entry Point")
    print("  Recursive Reasoning Transformer with Adaptive Computation")
    print("=" * 70)
    print()


def select_continual_learning_mode():
    """Interactive continual learning: select checkpoint and next dataset."""
    
    # Detect checkpoints
    checkpoints = detect_checkpoints()
    if not checkpoints:
        print("\n❌ No checkpoints found in checkpoints/ directory.")
        print("   Train a base model first (e.g., Phase 1: WikiText-2)\n")
        return None, None, None
    
    print("\n" + "="*70)
    print("  🔄 CONTINUAL LEARNING MODE")
    print("="*70)
    print("\nAvailable checkpoints:\n")
    
    for idx, ckpt in enumerate(checkpoints, 1):
        status = ckpt['status']
        
        # Check if corrupted
        if status.get('corrupted'):
            print(f"  [{idx}] {ckpt['name']} ❌ CORRUPTED")
            print(f"      Path: {ckpt['path']}")
            print(f"      Status: ❌ Cannot load - checkpoint is corrupted")
            print(f"      ⚠️  Delete this checkpoint or re-download from Colab")
        else:
            completion_icon = "✅" if status.get('is_complete') else "⚠️"
            progress = status.get('progress_pct', 0)
            
            print(f"  [{idx}] {ckpt['name']} {completion_icon}")
            print(f"      Path: {ckpt['path']}")
            print(f"      Progress: {progress:.1f}% ({status.get('current_step', 0):,}/{status.get('target_steps', 0):,} steps)")
            
            if status.get('is_complete'):
                print(f"      Status: ✅ COMPLETE - Ready for next phase")
            else:
                remaining = status.get('target_steps', 0) - status.get('current_step', 0)
                print(f"      Status: ⚠️  INCOMPLETE - {remaining:,} steps remaining")
        print()
    
    # Select checkpoint
    while True:
        try:
            choice = input(f"Select checkpoint to continue from [1-{len(checkpoints)}]: ")
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(checkpoints):
                selected_ckpt = checkpoints[choice_idx]
                
                # Block corrupted checkpoints
                if selected_ckpt['status'].get('corrupted'):
                    print("\n❌ Cannot use corrupted checkpoint!")
                    print("\nTo fix:")
                    print("  1. Delete the corrupted checkpoint:")
                    print(f"     rm -rf {selected_ckpt['dir']}")
                    print("  2. Re-download from Colab (use tar.gz, NOT zip!)")
                    print("  3. Or start fresh training\n")
                    continue
                
                break
            else:
                print(f"Invalid choice. Enter 1-{len(checkpoints)}")
        except (ValueError, KeyboardInterrupt):
            print("\nExiting...")
            sys.exit(0)
    
    print(f"\n✅ Selected: {selected_ckpt['name']}")
    
    # Check if selected phase is complete
    selected_status = selected_ckpt['status']
    if not selected_status.get('is_complete', False):
        print("\n" + "="*70)
        print("  ⚠️  WARNING: Selected phase is INCOMPLETE!")
        print("="*70)
        print(f"\nCurrent progress: {selected_status.get('progress_pct', 0):.1f}%")
        print(f"Steps: {selected_status.get('current_step', 0):,} / {selected_status.get('target_steps', 0):,}")
        print(f"Remaining: {selected_status.get('target_steps', 0) - selected_status.get('current_step', 0):,} steps\n")
        
        print("Options:")
        print("  [1] Continue training this phase (recommended)")
        print("  [2] Move to next phase anyway (not recommended)")
        print("  [3] Cancel\n")
        
        choice = input("Select option [1-3]: ").strip()
        
        if choice == "1":
            # Continue current phase
            remaining_epochs = selected_status.get('config_epochs', 0)
            dataset_path = selected_status.get('dataset', 'data/text-wikitext2')
            model_key = dataset_to_model_key(dataset_path)
            return selected_ckpt['path'], model_key, remaining_epochs
        elif choice == "2":
            print("\n⚠️  Proceeding to next phase with incomplete base...")
        else:
            print("\nCancelled.")
            return None, None, None
    
    # Extract completed datasets from checkpoint names
    completed_datasets = []
    for ckpt in checkpoints:
        name = ckpt['name'].lower()
        status = ckpt['status']
        
        # Only count as completed if phase is done
        if status.get('is_complete', False):
            for dataset_key in MODELS.keys():
                if dataset_key in name:
                    completed_datasets.append(dataset_key)
    
    # Recommend next phase
    next_phase = recommend_next_phase(completed_datasets)
    
    if next_phase:
        print("\n" + "="*70)
        print("  💡 RECOMMENDED NEXT PHASE")
        print("="*70)
        print(f"\nPhase {next_phase['phase']}: {next_phase['name']}")
        print(f"  Dataset: {next_phase['dataset']}")
        print(f"  Epochs: {next_phase['epochs']}")
        print(f"  Time: ~{next_phase['time_hours']} hours")
        print(f"  Goal: {next_phase['goal']}")
        print("\n" + "="*70)
        
        use_recommended = input("\nUse recommended phase? [Y/n]: ").strip().lower()
        if use_recommended in ['', 'y', 'yes']:
            return selected_ckpt['path'], next_phase['dataset'], next_phase['epochs']
    
    # Manual dataset selection
    print("\nAvailable datasets:\n")
    dataset_keys = list(MODELS.keys())
    for idx, key in enumerate(dataset_keys, 1):
        info = MODELS[key]
        print(f"  [{idx}] {key} - {info['name']}")
    
    while True:
        try:
            choice = input(f"\nSelect dataset [1-{len(dataset_keys)}]: ")
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(dataset_keys):
                selected_dataset = dataset_keys[choice_idx]
                break
            else:
                print(f"Invalid choice. Enter 1-{len(dataset_keys)}")
        except (ValueError, KeyboardInterrupt):
            print("\nExiting...")
            sys.exit(0)
    
    # Get epochs
    while True:
        try:
            epochs = int(input("Enter number of epochs [default: 2000]: ") or "2000")
            break
        except ValueError:
            print("Please enter a valid number")
    
    return selected_ckpt['path'], selected_dataset, epochs


def select_model_interactive():
    """Interactive model selection."""
    
    # Check if we should show continual learning option
    checkpoints = detect_checkpoints()
    
    print("Training Modes:\n")
    print("  [1] Fresh Training (Start new model)")
    if checkpoints:
        print("  [2] Continual Learning (Continue from checkpoint)")
    print("  [3] View Communication Roadmap")
    print()
    
    mode_choice = input(f"Select mode [1-{3 if checkpoints else 1}]: ").strip()
    
    if mode_choice == "2" and checkpoints:
        ckpt_path, dataset, epochs = select_continual_learning_mode()
        if ckpt_path:
            return {"mode": "continual", "checkpoint": ckpt_path, "dataset": dataset, "epochs": epochs}
        else:
            sys.exit(0)
    elif mode_choice == "3":
        print_roadmap()
        input("Press Enter to continue...")
        return select_model_interactive()  # Recurse
    
    # Fresh training mode
    print("\nAvailable Models:\n")
    for idx, (key, info) in enumerate(MODELS.items(), 1):
        print(f"  [{idx}] {info['name']}")
        print(f"      {info['description']}")
        print()
    
    while True:
        try:
            choice = input("Select model [1-{}]: ".format(len(MODELS)))
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(MODELS):
                selected_key = list(MODELS.keys())[choice_idx]
                return {"mode": "fresh", "dataset": selected_key}
            else:
                print(f"Invalid choice. Enter 1-{len(MODELS)}")
        except (ValueError, KeyboardInterrupt):
            print("\nExiting...")
            sys.exit(0)


def check_dataset_exists(output_dir: str) -> bool:
    """Check if dataset already exists."""
    train_path = Path(output_dir) / "train" / "dataset.json"
    return train_path.exists()


def download_and_build_dataset(model_config: dict, force_rebuild: bool = False):
    """Download dataset and build preprocessed files."""
    output_dir = model_config['dataset_args']['output_dir']
    
    # Check if already exists
    if check_dataset_exists(output_dir):
        if not force_rebuild:
            print(f"\n✅ Dataset found at: {output_dir}")
            print("   Skipping dataset building (use --rebuild-dataset to force rebuild)")
            return True
        else:
            print(f"\n🔄 Rebuilding dataset at: {output_dir}")
    
    print(f"\n📦 Building dataset: {output_dir}")
    print("-" * 70)
    
    # Build command
    builder_script = model_config['dataset_builder']
    args = model_config['dataset_args']
    
    cmd = ["python", builder_script]
    
    # Add subcommand if using unified builder
    if 'dataset_command' in model_config:
        cmd.append(model_config['dataset_command'])
    
    # Add arguments
    for key, value in args.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, list):
            cmd.extend([flag] + value)
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


def train_model(model_config: dict, extra_args: list, checkpoint_path: str = None, epochs_override: int = None):
    """Start training with specified config."""
    config_name = model_config['config']
    data_path = model_config['dataset_args']['output_dir']
    
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
  python train.py                    # Interactive mode (auto-resumes if checkpoint exists)
  python train.py --model text       # Train text model (auto-resumes from epoch 100 -> 500)
  python train.py --model arc        # Train ARC visual model
  python train.py --model code       # Train code model
  python train.py --rebuild-dataset  # Force rebuild dataset

Auto-Resume:
  Training automatically resumes from checkpoints/MODEL/latest.pt if it exists.
  No need to manually specify --load-checkpoint!
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODELS.keys()),
        help="Model type to train"
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
    
    args, unknown_args = parser.parse_known_args()
    
    print_banner()
    
    # Model selection
    checkpoint_path = None
    epochs_override = None
    
    if args.model:
        selected_model = args.model
        selection = {"mode": "fresh", "dataset": selected_model}
    else:
        selection = select_model_interactive()
    
    # Handle continual learning mode
    if isinstance(selection, dict) and selection["mode"] == "continual":
        checkpoint_path = selection["checkpoint"]
        selected_model = selection["dataset"]
        epochs_override = selection["epochs"]
        print(f"\n🔄 Continual Learning Mode")
        print(f"   Checkpoint: {checkpoint_path}")
        print(f"   Dataset: {selected_model}")
        print(f"   Epochs: {epochs_override}\n")
    else:
        selected_model = selection["dataset"]
    
    if selected_model not in MODELS:
        print(f"❌ Unknown model: {selected_model}")
        print(f"Available models: {', '.join(MODELS.keys())}")
        sys.exit(1)
    
    model_config = MODELS[selected_model]
    
    # Dataset preparation (auto-skips if exists unless --rebuild-dataset)
    success = download_and_build_dataset(model_config, force_rebuild=args.rebuild_dataset)
    if not success:
        print("\n❌ Cannot proceed without dataset. Exiting.")
        sys.exit(1)
    
    if args.dataset_only:
        print("\n✅ Dataset preparation complete. Exiting (--dataset-only).")
        sys.exit(0)
    
    # Start training
    train_model(model_config, unknown_args, checkpoint_path=checkpoint_path, epochs_override=epochs_override)
    
    print("\n" + "=" * 70)
    print("  ✅ Training completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
