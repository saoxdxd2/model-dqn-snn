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
        "name": "Text Generation (Language Model)",
        "config": "cfg_text",
        "dataset_builder": "dataset/build_text_dataset.py",
        "dataset_args": {
            "input_file": "wikitext-2",
            "output_dir": "data/text-wikitext2",
            "tokenizer_name": "gpt2",
            "max_seq_len": 512,
            "stride": 256
        },
        "description": "Train on WikiText-2 for general text generation"
    },
    "text-tiny": {
        "name": "Text Generation (TinyStories - Small)",
        "config": "cfg_text",
        "dataset_builder": "dataset/build_text_dataset.py",
        "dataset_args": {
            "input_file": "tinystories",
            "output_dir": "data/text-tinystories",
            "tokenizer_name": "gpt2",
            "max_seq_len": 256,
            "stride": 128
        },
        "description": "Train on TinyStories (smaller, faster convergence)"
    },
    "arc": {
        "name": "Visual Reasoning (ARC Puzzles)",
        "config": "cfg_pretrain",
        "dataset_builder": "dataset/build_arc_dataset.py",
        "dataset_args": {
            "input_file_prefix": "kaggle/combined/arc-agi",
            "output_dir": "data/arc-aug-1000",
            "subsets": ["training", "training2"],
            "test_set_name": "evaluation",
            "num_aug": 1000,
            "seed": 42
        },
        "description": "Train on ARC visual puzzles (original)"
    },
    "code": {
        "name": "Code Generation (Python)",
        "config": "cfg_code",
        "dataset_builder": "dataset/build_text_dataset.py",
        "dataset_args": {
            "input_file": "code-python",  # Auto-downloads The Stack Python
            "output_dir": "data/code-python",
            "tokenizer_name": "Salesforce/codegen-350M-mono",
            "max_seq_len": 1024,
            "stride": 512
        },
        "description": "Train on Python code from The Stack (deduplicated)"
    },
    "vision": {
        "name": "Image Understanding (Patch Tokens + TRM + DQN)",
        "config": "cfg_vision",
        "dataset_builder": "dataset/build_image_dataset.py",
        "dataset_args": {
            "dataset_name": "cifar10",
            "output_dir": "data/vision-cifar10",
            "patch_size": 8,  # 8×8 patches (4×4 grid = 16 tokens)
            "vocab_size": 2048,  # Patch vocabulary (learned codebook)
            "image_size": 32,  # CIFAR-10 native size
            "seed": 42
        },
        "description": "Train on CIFAR-10 with patch tokenization (like BPE for images) + DQN"
    },
    "alpaca": {
        "name": "Instruction Following (Alpaca)",
        "config": "cfg_text",
        "dataset_builder": "dataset/build_text_dataset.py",
        "dataset_args": {
            "input_file": "alpaca",
            "output_dir": "data/text-alpaca",
            "tokenizer_name": "gpt2",
            "max_seq_len": 512,
            "stride": 256
        },
        "description": "Train on Alpaca instruction-response pairs"
    },
    "sharegpt": {
        "name": "Conversational (ShareGPT)",
        "config": "cfg_text",
        "dataset_builder": "dataset/build_text_dataset.py",
        "dataset_args": {
            "input_file": "sharegpt",
            "output_dir": "data/text-sharegpt",
            "tokenizer_name": "gpt2",
            "max_seq_len": 512,
            "stride": 256
        },
        "description": "Train on ShareGPT multi-turn dialogues"
    }
}


def detect_checkpoints():
    """Detect all available checkpoints in checkpoints/ directory."""
    checkpoint_dir = Path("checkpoints")
    if not checkpoint_dir.exists():
        return []
    
    checkpoints = []
    for subdir in checkpoint_dir.iterdir():
        if subdir.is_dir():
            latest_pt = subdir / "latest.pt"
            if latest_pt.exists():
                checkpoints.append({
                    "name": subdir.name,
                    "path": str(latest_pt),
                    "dir": str(subdir)
                })
    return checkpoints


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
        print(f"  [{idx}] {ckpt['name']}")
        print(f"      Path: {ckpt['path']}")
        print()
    
    # Select checkpoint
    while True:
        try:
            choice = input(f"Select checkpoint to continue from [1-{len(checkpoints)}]: ")
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(checkpoints):
                selected_ckpt = checkpoints[choice_idx]
                break
            else:
                print(f"Invalid choice. Enter 1-{len(checkpoints)}")
        except (ValueError, KeyboardInterrupt):
            print("\nExiting...")
            sys.exit(0)
    
    print(f"\n✅ Selected: {selected_ckpt['name']}")
    
    # Extract completed datasets from checkpoint names
    completed_datasets = []
    for ckpt in checkpoints:
        name = ckpt['name'].lower()
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
    
    # Add continual learning parameters
    if checkpoint_path:
        cmd.append(f"load_checkpoint={checkpoint_path}")
    if epochs_override:
        cmd.append(f"epochs={epochs_override}")
    
    cmd += extra_args
    
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        # Run training (will stream output)
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Unified TRM Training Pipeline",
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
