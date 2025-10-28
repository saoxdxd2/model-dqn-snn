"""
DQN Experience Replay Buffer with BF16 compression for memory efficiency.
Stores transitions: (state, action, reward, next_state, done, step)
"""

from typing import Dict, Optional
import numpy as np
import torch
from torch import Tensor


class DQNReplayBuffer:
    """
    Experience replay buffer for DQN halting mechanism.
    
    Features:
    - BF16 compression for state storage (50% memory reduction)
    - CPU storage, GPU sampling
    - Circular buffer with fixed capacity
    - Support for distributed training (local buffers per rank)
    """
    
    def __init__(
        self,
        capacity: int = 20000,
        rank: int = 0,
        world_size: int = 1,
        compress: bool = True,
        storage_dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            rank: Current process rank (for distributed training)
            world_size: Total number of processes
            compress: Whether to use BF16 compression
            storage_dtype: Data type for compressed storage
        """
        self.capacity = capacity
        self.rank = rank
        self.world_size = world_size
        self.compress = compress
        self.storage_dtype = storage_dtype
        
        # Storage on CPU (BF16 for states, native types for others)
        self.states = []        # List of tensors [hidden_size]
        self.actions = []       # List of int (0=continue, 1=halt)
        self.rewards = []       # List of float
        self.next_states = []   # List of tensors [hidden_size]
        self.dones = []         # List of bool
        self.steps = []         # List of int
        
        self.position = 0
        self.size = 0
    
    def push(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        next_state: Tensor,
        done: Tensor,
        step: Tensor,
    ):
        """
        Store a single transition.
        
        Args:
            state: Current state [hidden_size]
            action: Action taken (0=continue, 1=halt)
            reward: Reward received (float)
            next_state: Next state [hidden_size]
            done: Whether episode ended (bool)
            step: Current step number (int)
        """
        # Move to CPU and compress
        if self.compress:
            state = state.detach().cpu().to(self.storage_dtype)
            next_state = next_state.detach().cpu().to(self.storage_dtype)
        else:
            state = state.detach().cpu()
            next_state = next_state.detach().cpu()
        
        # Extract scalar values
        action_val = action.item() if isinstance(action, Tensor) else action
        reward_val = reward.item() if isinstance(reward, Tensor) else reward
        done_val = done.item() if isinstance(done, Tensor) else done
        step_val = step.item() if isinstance(step, Tensor) else step
        
        # Store
        if self.size < self.capacity:
            # Append
            self.states.append(state)
            self.actions.append(action_val)
            self.rewards.append(reward_val)
            self.next_states.append(next_state)
            self.dones.append(done_val)
            self.steps.append(step_val)
            self.size += 1
        else:
            # Circular replacement
            idx = self.position % self.capacity
            self.states[idx] = state
            self.actions[idx] = action_val
            self.rewards[idx] = reward_val
            self.next_states[idx] = next_state
            self.dones[idx] = done_val
            self.steps[idx] = step_val
        
        self.position += 1
    
    def sample(self, batch_size: int, device: str = 'cuda') -> Dict[str, Tensor]:
        """
        Sample random batch from buffer.
        
        Args:
            batch_size: Number of transitions to sample
            device: Device to move samples to
        
        Returns:
            Dictionary with keys: state, action, reward, next_state, done, step
        """
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        # Random indices
        indices = np.random.choice(self.size, size=min(batch_size, self.size), replace=False)
        
        # Gather samples
        batch = {
            'state': torch.stack([self.states[i] for i in indices]).to(device=device, dtype=torch.float32),
            'action': torch.tensor([self.actions[i] for i in indices], device=device, dtype=torch.long),
            'reward': torch.tensor([self.rewards[i] for i in indices], device=device, dtype=torch.float32),
            'next_state': torch.stack([self.next_states[i] for i in indices]).to(device=device, dtype=torch.float32),
            'done': torch.tensor([self.dones[i] for i in indices], device=device, dtype=torch.bool),
            'step': torch.tensor([self.steps[i] for i in indices], device=device, dtype=torch.int32),
        }
        
        return batch
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size
    
    def clear(self):
        """Clear all stored transitions."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.steps.clear()
        self.position = 0
        self.size = 0
    
    def get_stats(self) -> Dict[str, float]:
        """Get buffer statistics for monitoring."""
        if self.size == 0:
            return {
                'size': 0,
                'utilization': 0.0,
                'mean_reward': 0.0,
                'std_reward': 0.0,
            }
        
        return {
            'size': self.size,
            'utilization': self.size / self.capacity,
            'mean_reward': float(np.mean(self.rewards[:self.size])),
            'std_reward': float(np.std(self.rewards[:self.size])),
            'mean_episode_length': float(np.mean(self.steps[:self.size])),
        }
