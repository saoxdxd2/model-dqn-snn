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
    - Prioritized experience replay (optional, for better learning)
    """
    
    def __init__(
        self,
        capacity: int = 20000,
        rank: int = 0,
        world_size: int = 1,
        compress: bool = True,
        storage_dtype: torch.dtype = torch.bfloat16,
        prioritized: bool = False,
        alpha: float = 0.6,  # Priority exponent
        beta: float = 0.4,   # Importance sampling exponent
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
        
        # Prioritized replay
        self.prioritized = prioritized
        self.alpha = alpha  # Priority exponent (0=uniform, 1=full prioritization)
        self.beta = beta    # Importance sampling correction (annealed to 1.0)
        self.beta_increment = 0.001  # Anneal beta over time
        
        # Pre-allocated storage on CPU (eliminates memory fragmentation)
        self.states = None           # Will be [capacity, hidden_size]
        self.actions = np.zeros(capacity, dtype=np.int8)  # 0=continue, 1=halt
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = None      # Will be [capacity, hidden_size]
        self.dones = np.zeros(capacity, dtype=bool)
        self.steps = np.zeros(capacity, dtype=np.int32)
        
        # Priority tree for efficient sampling (if prioritized)
        if self.prioritized:
            self.priorities = np.ones(capacity, dtype=np.float32)
            self.max_priority = 1.0
        
        self.position = 0
        self.size = 0
        self.hidden_size = None  # Will be initialized on first push
    
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
        # Initialize storage on first push (lazy initialization)
        if self.states is None:
            self.hidden_size = state.shape[0]
            dtype = self.storage_dtype if self.compress else torch.float32
            self.states = torch.zeros(self.capacity, self.hidden_size, dtype=dtype)
            self.next_states = torch.zeros(self.capacity, self.hidden_size, dtype=dtype)
        
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
        
        # Circular buffer index
        idx = self.position % self.capacity
        
        # Store (in-place, no append - eliminates fragmentation)
        self.states[idx] = state
        self.actions[idx] = action_val
        self.rewards[idx] = reward_val
        self.next_states[idx] = next_state
        self.dones[idx] = done_val
        self.steps[idx] = step_val
        
        # Set priority for new transition (max priority for new experiences)
        if self.prioritized:
            self.priorities[idx] = self.max_priority
        
        # Update counters
        self.size = min(self.size + 1, self.capacity)
        self.position += 1
    
    def sample(self, batch_size: int, device: str = 'cuda') -> Dict[str, Tensor]:
        """
        Sample batch from buffer (uniform or prioritized).
        
        Args:
            batch_size: Number of transitions to sample
            device: Device to move samples to
        
        Returns:
            Dictionary with keys: state, action, reward, next_state, done, step, indices, weights
        """
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        batch_size = min(batch_size, self.size)
        
        if self.prioritized:
            # Prioritized sampling
            priorities = self.priorities[:self.size] ** self.alpha
            probabilities = priorities / priorities.sum()
            
            indices = np.random.choice(self.size, size=batch_size, replace=False, p=probabilities)
            
            # Importance sampling weights (for bias correction)
            # Anneal beta from initial value to 1.0
            self.beta = min(1.0, self.beta + self.beta_increment)
            weights = (self.size * probabilities[indices]) ** (-self.beta)
            weights /= weights.max()  # Normalize for stability
            weights = torch.from_numpy(weights).to(device=device, dtype=torch.float32)
        else:
            # Uniform sampling
            indices = np.random.choice(self.size, size=batch_size, replace=False)
            weights = torch.ones(batch_size, device=device, dtype=torch.float32)
        
        # Gather samples (vectorized operations on pre-allocated arrays)
        batch = {
            'state': self.states[indices].to(device=device, dtype=torch.float32),
            'action': torch.from_numpy(self.actions[indices]).to(device=device, dtype=torch.long),
            'reward': torch.from_numpy(self.rewards[indices]).to(device=device, dtype=torch.float32),
            'next_state': self.next_states[indices].to(device=device, dtype=torch.float32),
            'done': torch.from_numpy(self.dones[indices]).to(device=device, dtype=torch.bool),
            'step': torch.from_numpy(self.steps[indices]).to(device=device, dtype=torch.int32),
            'indices': indices,  # For priority updates
            'weights': weights,  # Importance sampling weights
        }
        
        return batch
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size
    
    def update_priorities(self, indices: np.ndarray, td_errors: Tensor):
        """Update priorities for sampled transitions based on TD-errors.
        
        Args:
            indices: Indices of transitions to update
            td_errors: TD-errors for priority calculation (higher error = higher priority)
        """
        if not self.prioritized:
            return
        
        # Convert TD-errors to priorities (add small epsilon for stability)
        priorities = np.abs(td_errors.detach().cpu().numpy()) + 1e-6
        
        # Update priorities
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())
    
    def clear(self):
        """Clear all stored transitions."""
        # Reset arrays to zero (no need to reallocate)
        if self.states is not None:
            self.states.zero_()
            self.next_states.zero_()
        self.actions.fill(0)
        self.rewards.fill(0.0)
        self.dones.fill(False)
        self.steps.fill(0)
        
        if self.prioritized:
            self.priorities.fill(1.0)
            self.max_priority = 1.0
        
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
