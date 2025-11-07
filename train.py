"""Simplified training wrapper - calls pretrain.py with hybrid config.

Usage:
    python train.py                              # Train with hybrid_pretrained.yaml
    python train.py --config other_config.yaml   # Train with custom config
    python train.py [any pretrain.py args]       # Pass through to pretrain.py
"""

import sys
import subprocess


# Default config for hybrid pretrained pipeline
DEFAULT_CONFIG = "config/arch/hybrid_pretrained.yaml"




def main():
    """Simple wrapper: calls pretrain.py with hybrid_pretrained config."""
    
    # Check if user specified custom config
    config = DEFAULT_CONFIG
    args = sys.argv[1:]  # All arguments after script name
    
    # If --config specified, use it
    for i, arg in enumerate(args):
        if arg == "--config" and i + 1 < len(args):
            config = args[i + 1]
            # Remove --config and its value from args
            args = args[:i] + args[i+2:]
            break
    
    print("="*70)
    print("  🚀 TRM Training - Hybrid Pretrained Pipeline")
    print(f"  Config: {config}")
    print("  Components: CLIP + ViT + N2N + TRM + COCONUT")
    print("  Optimizations: fp16 + torch.compile + batch=192")
    print("="*70)
    print()
    
    # Build command: python pretrain.py --config <config> [other args]
    cmd = ["python", "pretrain.py", "--config", config] + args
    
    print(f"Running: {' '.join(cmd)}\n")
    
    try:
        # Run pretrain.py with all arguments passed through
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
