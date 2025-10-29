"""
Check training phase completion status.

Usage:
    python scripts/check_phase_status.py checkpoints/text-trm-10m/latest.pt
"""

import sys
import torch
import json
from pathlib import Path


def check_phase_completion(checkpoint_path: str):
    """Check if training phase is complete."""
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    try:
        # Load checkpoint metadata (on CPU to avoid CUDA requirement)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        current_step = checkpoint.get('step', 0)
        target_steps = checkpoint.get('total_steps', 0)
        config_epochs = checkpoint.get('config_epochs', 0)
        dataset = checkpoint.get('dataset', 'unknown')
        
        if target_steps > 0:
            progress = (current_step / target_steps) * 100
            is_complete = current_step >= target_steps
        else:
            progress = 0
            is_complete = False
        
        return {
            'current_step': current_step,
            'target_steps': target_steps,
            'config_epochs': config_epochs,
            'dataset': dataset,
            'progress_pct': progress,
            'is_complete': is_complete
        }
        
    except Exception as e:
        print(f"❌ Error reading checkpoint: {e}")
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/check_phase_status.py <checkpoint_path>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    
    print("="*70)
    print("  📊 TRAINING PHASE STATUS")
    print("="*70)
    print()
    
    status = check_phase_completion(checkpoint_path)
    
    if status is None:
        sys.exit(1)
    
    print(f"Dataset: {status['dataset']}")
    print(f"Target epochs: {status['config_epochs']}")
    print()
    print(f"Progress: {status['current_step']:,} / {status['target_steps']:,} steps ({status['progress_pct']:.1f}%)")
    print()
    
    if status['is_complete']:
        print("✅ Phase COMPLETE - Ready for next phase!")
    else:
        remaining = status['target_steps'] - status['current_step']
        print(f"⚠️  Phase INCOMPLETE - {remaining:,} steps remaining")
        print("   Continue training this phase before moving to next.")
    
    print("="*70)
    
    sys.exit(0 if status['is_complete'] else 1)


if __name__ == "__main__":
    main()
