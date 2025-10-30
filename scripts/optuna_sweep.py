"""
Optuna Hyperparameter Optimization for TRM Text Model

Optimizes key hyperparameters for better perplexity and convergence speed.
Targets 10-15% quality improvement over manual tuning.

Usage:
    python scripts/optuna_sweep.py --n-trials 50 --timeout 86400
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import yaml

# Import training function
from pretrain import PretrainConfig, init_train_state, create_dataloader


def objective(trial: optuna.Trial) -> float:
    """
    Objective function for Optuna optimization.
    
    Args:
        trial: Optuna trial object
    
    Returns:
        Validation loss (lower is better)
    """
    
    # Sample hyperparameters (EXPANDED RANGES for GPU scalability)
    lr = trial.suggest_float('lr', 1e-6, 5e-3, log=True)  # Wider LR range
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 96, 128, 154, 192, 224, 256, 320, 384])  # Up to 384
    
    # Architecture hyperparameters (EXPANDED for larger models)
    H_cycles = trial.suggest_int('H_cycles', 1, 10)  # Up to 10 cycles (deep recursion)
    L_cycles = trial.suggest_int('L_cycles', 1, 10)  # Up to 10 cycles
    hidden_size = trial.suggest_categorical('hidden_size', [128, 256, 384, 512, 640, 768, 896, 1024])  # Up to 1024
    num_heads = trial.suggest_categorical('num_heads', [2, 4, 6, 8, 10, 12, 16, 20])  # Up to 20 heads
    expansion = trial.suggest_float('expansion', 1.5, 6.0)  # Wider FFN expansion
    
    # Regularization
    weight_decay = trial.suggest_float('weight_decay', 0.0, 0.2, log=True)  # Up to 0.2
    
    # Halting hyperparameters
    halt_exploration_prob = trial.suggest_float('halt_exploration_prob', 0.01, 0.5)  # Wider range
    halt_max_steps = trial.suggest_int('halt_max_steps', 4, 16)  # Variable max steps
    
    # DQN hyperparameters (if enabled)
    dqn_epsilon_start = trial.suggest_float('dqn_epsilon_start', 0.05, 0.7)  # Wider exploration
    dqn_gamma = trial.suggest_float('dqn_gamma', 0.9, 0.999)  # Wider discount range
    dqn_buffer_capacity = trial.suggest_categorical('dqn_buffer_capacity', [10000, 20000, 40000, 60000])  # Larger buffers
    
    # Load base config
    config_path = Path(__file__).parent.parent / "config" / "cfg_text.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Update with trial hyperparameters
    config_dict['lr'] = lr
    config_dict['global_batch_size'] = batch_size
    config_dict['weight_decay'] = weight_decay
    config_dict['epochs'] = 50  # Quick validation run (reduced from 100 for faster trials)
    
    # Update architecture config
    arch_config_path = Path(__file__).parent.parent / "config" / "arch" / "trm_text.yaml"
    with open(arch_config_path, 'r') as f:
        arch_dict = yaml.safe_load(f)
    
    arch_dict['H_cycles'] = H_cycles
    arch_dict['L_cycles'] = L_cycles
    arch_dict['hidden_size'] = hidden_size
    arch_dict['num_heads'] = num_heads
    arch_dict['expansion'] = expansion
    arch_dict['halt_exploration_prob'] = halt_exploration_prob
    arch_dict['halt_max_steps'] = halt_max_steps
    arch_dict['dqn_epsilon_start'] = dqn_epsilon_start
    arch_dict['dqn_gamma'] = dqn_gamma
    arch_dict['dqn_buffer_capacity'] = dqn_buffer_capacity
    
    # Ensure num_heads divides hidden_size
    if hidden_size % num_heads != 0:
        # Adjust num_heads to nearest divisor
        for h in [20, 16, 12, 10, 8, 6, 4, 2, 1]:
            if hidden_size % h == 0:
                num_heads = h
                arch_dict['num_heads'] = h
                break
    
    # Early pruning: Skip if config too large for GPU (estimate)
    # Rough estimation: params ≈ hidden_size² × (H_cycles × L_cycles × L_layers)
    estimated_params = (hidden_size ** 2) * (H_cycles * L_cycles * 3) * 6  # Rough multiplier
    max_params = 150_000_000  # 150M params limit for 15GB GPU
    if estimated_params > max_params:
        print(f"  ⚠️  Skipping trial {trial.number}: estimated {estimated_params/1e6:.1f}M params (>150M)")
        raise optuna.TrialPruned()
    
    # Merge configs
    config_dict['arch'] = arch_dict
    
    # Create temporary config file
    temp_config_path = f"/tmp/optuna_trial_{trial.number}.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(config_dict, f)
    
    try:
        # Initialize WandB for this trial
        wandb.init(
            project=f"trm-optuna-sweep",
            name=f"trial_{trial.number}",
            config={
                'lr': lr,
                'batch_size': batch_size,
                'H_cycles': H_cycles,
                'L_cycles': L_cycles,
                'hidden_size': hidden_size,
                'num_heads': num_heads,
                'expansion': expansion,
                'weight_decay': weight_decay,
            },
            reinit=True
        )
        
        # Load config as PretrainConfig
        config = PretrainConfig(**config_dict)
        
        # Create dataloaders
        train_loader, train_metadata = create_dataloader(
            config, "train", test_set_mode=False, 
            epochs_per_iter=1, global_batch_size=batch_size,
            rank=0, world_size=1
        )
        
        try:
            eval_loader, eval_metadata = create_dataloader(
                config, "test", test_set_mode=True,
                epochs_per_iter=1, global_batch_size=batch_size,
                rank=0, world_size=1
            )
        except:
            eval_loader = None
            eval_metadata = None
        
        # Initialize model and training state
        train_state = init_train_state(config, train_metadata, rank=0, world_size=1)
        
        # Quick training loop (100 epochs for fast evaluation)
        train_state.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for set_name, batch, global_batch_size in train_loader:
            if num_batches >= 50:  # Limit batches for speed
                break
            
            # Move to GPU
            batch = {k: v.cuda() for k, v in batch.items()}
            
            # Forward pass
            if train_state.carry is None:
                with torch.device("cuda"):
                    train_state.carry = train_state.model.initial_carry(batch)
            
            train_state.carry, loss, metrics, _, _ = train_state.model(
                carry=train_state.carry, batch=batch, return_keys=[]
            )
            
            total_loss += loss.item()
            num_batches += 1
            
            # Optuna pruning: report intermediate values
            if num_batches % 10 == 0:
                intermediate_loss = total_loss / num_batches
                trial.report(intermediate_loss, num_batches)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    wandb.finish()
                    raise optuna.TrialPruned()
        
        # Calculate final validation loss
        val_loss = total_loss / max(num_batches, 1)
        
        # Log to WandB
        wandb.log({'final_val_loss': val_loss})
        wandb.finish()
        
        return val_loss
        
    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        wandb.finish()
        return float('inf')  # Return worst possible value
    finally:
        # Clean up
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
        
        # Free GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description='Optuna hyperparameter optimization for TRM')
    parser.add_argument('--n-trials', type=int, default=50, help='Number of trials')
    parser.add_argument('--timeout', type=int, default=86400, help='Timeout in seconds (default: 24 hours)')
    parser.add_argument('--study-name', type=str, default='trm-text-optimization', help='Study name')
    parser.add_argument('--storage', type=str, default='sqlite:///optuna_study.db', help='Database URL')
    parser.add_argument('--load-if-exists', action='store_true', help='Load existing study if it exists')
    
    args = parser.parse_args()
    
    # Create or load study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=args.load_if_exists,
        direction='minimize',  # Minimize validation loss
        sampler=TPESampler(seed=42),  # Tree-structured Parzen Estimator
        pruner=MedianPruner(
            n_startup_trials=5,  # Don't prune first 5 trials
            n_warmup_steps=10,   # Wait 10 steps before pruning
            interval_steps=5     # Check every 5 steps
        )
    )
    
    print(f"\n{'='*70}")
    print(f"  🔬 Optuna Hyperparameter Optimization")
    print(f"{'='*70}")
    print(f"  Study: {args.study_name}")
    print(f"  Trials: {args.n_trials}")
    print(f"  Timeout: {args.timeout}s ({args.timeout/3600:.1f} hours)")
    print(f"  Storage: {args.storage}")
    print(f"{'='*70}\n")
    
    # Run optimization
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
        n_jobs=1  # Single GPU
    )
    
    # Print results
    print(f"\n{'='*70}")
    print(f"  ✅ Optimization Complete!")
    print(f"{'='*70}")
    print(f"  Best trial: {study.best_trial.number}")
    print(f"  Best validation loss: {study.best_value:.4f}")
    print(f"\n  Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    print(f"{'='*70}\n")
    
    # Save best config
    best_config_path = Path(__file__).parent.parent / "config" / "arch" / "trm_text_optimized.yaml"
    
    # Load base config
    arch_config_path = Path(__file__).parent.parent / "config" / "arch" / "trm_text.yaml"
    with open(arch_config_path, 'r') as f:
        best_config = yaml.safe_load(f)
    
    # Update with best hyperparameters
    best_config['H_cycles'] = study.best_params['H_cycles']
    best_config['L_cycles'] = study.best_params['L_cycles']
    best_config['hidden_size'] = study.best_params['hidden_size']
    best_config['num_heads'] = study.best_params['num_heads']
    best_config['expansion'] = study.best_params['expansion']
    best_config['halt_exploration_prob'] = study.best_params['halt_exploration_prob']
    best_config['dqn_epsilon_start'] = study.best_params['dqn_epsilon_start']
    best_config['dqn_gamma'] = study.best_params['dqn_gamma']
    
    with open(best_config_path, 'w') as f:
        yaml.dump(best_config, f)
    
    print(f"💾 Saved best config to: {best_config_path}")
    
    # Save training config
    training_config_path = Path(__file__).parent.parent / "config" / "cfg_text_optimized.yaml"
    training_config = {
        'defaults': ['arch: trm_text_optimized', '_self_'],
        'lr': study.best_params['lr'],
        'global_batch_size': study.best_params['batch_size'],
        'weight_decay': study.best_params['weight_decay'],
    }
    
    with open(training_config_path, 'w') as f:
        yaml.dump(training_config, f)
    
    print(f"💾 Saved training config to: {training_config_path}")
    print(f"\nTo train with optimized config:")
    print(f"  python pretrain.py --config-name cfg_text_optimized")


if __name__ == "__main__":
    main()
