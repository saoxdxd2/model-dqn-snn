"""
Continue training on new dataset (Transfer Learning)

Usage:
    python continue_training.py --from-checkpoint checkpoints/text-trm-10m/latest.pt --new-dataset text-tiny --epochs 2000
"""

import argparse
import subprocess
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Continue training on new dataset")
    parser.add_argument("--from-checkpoint", type=str, required=True,
                       help="Path to checkpoint to continue from (e.g., checkpoints/text-trm-10m/latest.pt)")
    parser.add_argument("--new-dataset", type=str, required=True,
                       choices=["text", "text-tiny", "arc", "code", "cifar10"],
                       help="New dataset to train on")
    parser.add_argument("--epochs", type=int, default=2000,
                       help="Number of epochs to train (default: 2000)")
    parser.add_argument("--config", type=str, default=None,
                       help="Config name (auto-detected if not specified)")
    
    args = parser.parse_args()
    
    # Verify checkpoint exists
    if not os.path.exists(args.from_checkpoint):
        print(f"❌ Checkpoint not found: {args.from_checkpoint}")
        return
    
    # Map dataset to config
    dataset_configs = {
        "text": ("cfg_text", "data/text-wikitext2"),
        "text-tiny": ("cfg_text", "data/text-tinystories"),
        "arc": ("cfg_pretrain", "data/arc-aug-1000"),
        "code": ("cfg_code", "data/code-python"),
        "cifar10": ("cfg_text", "data/cifar10-patches"),
    }
    
    config_name, data_path = dataset_configs[args.new_dataset]
    if args.config:
        config_name = args.config
    
    print(f"\n{'='*70}")
    print(f"  🔄 CONTINUAL LEARNING - Transfer Training")
    print(f"{'='*70}")
    print(f"  📚 Base checkpoint: {args.from_checkpoint}")
    print(f"  🎯 New dataset: {args.new_dataset}")
    print(f"  📁 Data path: {data_path}")
    print(f"  ⚙️  Config: {config_name}")
    print(f"  🔢 Epochs: {args.epochs}")
    print(f"{'='*70}\n")
    
    # Build training command
    cmd = [
        "python", "pretrain.py",
        f"--config-name={config_name}",
        f"data_paths=[{data_path}]",
        f"load_checkpoint={args.from_checkpoint}",
        f"epochs={args.epochs}",
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    
    # Confirm with user
    response = input("Start continual learning? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    print()
    
    # Run training
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed: {e}")
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")


if __name__ == "__main__":
    main()
