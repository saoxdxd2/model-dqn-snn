"""
Unified training script for TRM models.
Supports: Text, Visual (ARC), Coding, and Multi-Modal.

Usage:
    python train.py --model text
    python train.py --model arc
    python train.py --model code
    python train.py  # Interactive mode
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse


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
        "name": "Image Understanding (ViT + TRM + DQN)",
        "config": "cfg_vision",
        "dataset_builder": "dataset/build_image_dataset.py",
        "dataset_args": {
            "dataset_name": "cifar10",
            "output_dir": "data/vision-cifar10",
            "patch_size": 16,
            "image_size": 224,
            "seed": 42
        },
        "description": "Train on CIFAR-10 images with ViT patches + recursive reasoning + DQN"
    }
}


def print_banner():
    """Print welcome banner."""
    print("=" * 70)
    print("  🧠 TRM Training Pipeline - Unified Entry Point")
    print("  Recursive Reasoning Transformer with Adaptive Computation")
    print("=" * 70)
    print()


def select_model_interactive():
    """Interactive model selection."""
    print("Available Models:\n")
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
                return selected_key
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


def train_model(model_config: dict, extra_args: list):
    """Start training with specified config."""
    config_name = model_config['config']
    data_path = model_config['dataset_args']['output_dir']
    
    print(f"\n🚀 Starting training: {model_config['name']}")
    print("-" * 70)
    print(f"Config: {config_name}")
    print(f"Data: {data_path}")
    print("-" * 70)
    print()
    
    # Build training command with Hydra overrides
    cmd = [
        "python", "pretrain.py",
        f"--config-name={config_name}",
        f"data_paths=[{data_path}]"  # Override data path
    ] + extra_args
    
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
  python train.py                    # Interactive mode
  python train.py --model text       # Train text model
  python train.py --model arc        # Train ARC visual model
  python train.py --model code       # Train code model
  python train.py --skip-dataset     # Skip dataset building
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
    if args.model:
        model_key = args.model
        print(f"Selected model: {MODELS[model_key]['name']}\n")
    else:
        model_key = select_model_interactive()
    
    model_config = MODELS[model_key]
    
    # Dataset preparation (auto-skips if exists unless --rebuild-dataset)
    success = download_and_build_dataset(model_config, force_rebuild=args.rebuild_dataset)
    if not success:
        print("\n❌ Cannot proceed without dataset. Exiting.")
        sys.exit(1)
    
    if args.dataset_only:
        print("\n✅ Dataset preparation complete. Exiting (--dataset-only).")
        sys.exit(0)
    
    # Training
    train_model(model_config, unknown_args)
    
    print("\n" + "=" * 70)
    print("  ✅ Training completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
