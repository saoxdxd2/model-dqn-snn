"""
Utilities for DQN training: epsilon scheduling, running statistics, reward normalization.
"""

import torch
from torch import Tensor
from typing import Optional


class RunningStats:
    """
    Track running mean and standard deviation for normalization.
    Uses Welford's online algorithm for numerical stability.
    """
    
    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize running statistics tracker.
        
        Args:
            epsilon: Small constant for numerical stability
        """
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.epsilon = epsilon
    
    def update(self, x: Tensor):
        """
        Update running statistics with new batch.
        
        Args:
            x: New values to incorporate [batch_size] or scalar
        """
        if isinstance(x, Tensor):
            batch_mean = x.mean().item()
            batch_var = x.var().item()
            batch_count = x.numel()
        else:
            batch_mean = float(x)
            batch_var = 0.0
            batch_count = 1
        
        # Welford's online algorithm
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / (self.count + batch_count)
        
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / (self.count + batch_count)
        self.var = M2 / (self.count + batch_count) if (self.count + batch_count) > 0 else 1.0
        self.count += batch_count
    
    def normalize(self, x: Tensor) -> Tensor:
        """
        Normalize tensor using running statistics.
        
        Args:
            x: Tensor to normalize
        
        Returns:
            Normalized tensor
        """
        return (x - self.mean) / (torch.sqrt(torch.tensor(self.var)) + self.epsilon)
    
    def denormalize(self, x: Tensor) -> Tensor:
        """
        Reverse normalization.
        
        Args:
            x: Normalized tensor
        
        Returns:
            Original scale tensor
        """
        return x * (torch.sqrt(torch.tensor(self.var)) + self.epsilon) + self.mean
    
    def reset(self):
        """Reset statistics."""
        self.mean = 0.0
        self.var = 1.0
        self.count = 0


def compute_epsilon(
    step: int,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_steps: int = 100000,
) -> float:
    """
    Compute epsilon for epsilon-greedy exploration with linear decay.
    
    Args:
        step: Current training step
        epsilon_start: Initial exploration rate
        epsilon_end: Final exploration rate
        epsilon_decay_steps: Number of steps to decay over
    
    Returns:
        Current epsilon value
    """
    if step >= epsilon_decay_steps:
        return epsilon_end
    
    progress = step / epsilon_decay_steps
    return epsilon_start + (epsilon_end - epsilon_start) * progress


def select_action_epsilon_greedy(
    q_halt_logits: Tensor,
    epsilon: float,
    training: bool = True,
) -> Tensor:
    """
    Select action using epsilon-greedy strategy.
    
    Args:
        q_halt_logits: Q-values for halt action [batch_size]
        epsilon: Exploration rate (0 = greedy, 1 = random)
        training: Whether in training mode
    
    Returns:
        Boolean tensor [batch_size] indicating halt decision
    """
    if not training:
        # Evaluation: always greedy
        return q_halt_logits > 0
    
    batch_size = q_halt_logits.shape[0]
    device = q_halt_logits.device
    
    # Random mask: True = explore, False = exploit
    random_mask = torch.rand(batch_size, device=device) < epsilon
    
    # Random action: 50/50 halt/continue
    random_action = torch.rand(batch_size, device=device) < 0.5
    
    # Greedy action: halt if Q(halt) > 0
    greedy_action = q_halt_logits > 0
    
    # Combine: use random action where mask is True, greedy otherwise
    action = torch.where(random_mask, random_action, greedy_action)
    
    return action


def compute_shaped_reward(
    prev_accuracy: Tensor,
    curr_accuracy: Tensor,
    step: Tensor,
    max_steps: int,
    is_terminal: Tensor,
    seq_is_correct: Tensor,
    no_improvement_counter: Optional[Tensor] = None,
    reward_accuracy_scale: float = 10.0,
    reward_step_penalty: float = 0.02,
    reward_terminal_correct: float = 10.0,
    reward_terminal_incorrect: float = -5.0,
    reward_stagnation_penalty: float = 0.1,
    reward_stagnation_threshold: int = 3,
) -> Tensor:
    """
    Compute dense shaped reward for reasoning steps.
    
    Args:
        prev_accuracy: Accuracy at previous step [batch_size]
        curr_accuracy: Accuracy at current step [batch_size]
        step: Current step number [batch_size]
        max_steps: Maximum allowed steps
        is_terminal: Whether this is the final step [batch_size]
        seq_is_correct: Whether final answer is correct [batch_size]
        no_improvement_counter: Steps since last improvement [batch_size]
        reward_accuracy_scale: Scale for accuracy improvement reward
        reward_step_penalty: Penalty per step (encourage efficiency)
        reward_terminal_correct: Bonus for correct final answer
        reward_terminal_incorrect: Penalty for incorrect early halt
        reward_stagnation_penalty: Penalty for no progress
        reward_stagnation_threshold: Steps before stagnation penalty applies
    
    Returns:
        Shaped reward [batch_size]
    """
    batch_size = prev_accuracy.shape[0]
    device = prev_accuracy.device
    reward = torch.zeros(batch_size, device=device)
    
    # Component 1: Accuracy improvement
    acc_delta = curr_accuracy - prev_accuracy
    reward += reward_accuracy_scale * acc_delta
    
    # Component 2: Step penalty (encourage efficiency)
    reward -= reward_step_penalty
    
    # Component 3: Stagnation penalty (penalize wasted computation)
    if no_improvement_counter is not None:
        stagnation_mask = no_improvement_counter >= reward_stagnation_threshold
        stagnation_amount = (no_improvement_counter - reward_stagnation_threshold + 1).float()
        reward -= torch.where(
            stagnation_mask,
            reward_stagnation_penalty * stagnation_amount,
            torch.zeros_like(reward)
        )
    
    # Component 4: Terminal reward (large signal at halt)
    terminal_reward = torch.where(
        is_terminal,
        torch.where(
            seq_is_correct,
            # Correct: large positive + efficiency bonus
            reward_terminal_correct + (max_steps - step).float() * 0.5,
            # Incorrect: penalty
            torch.full_like(reward, reward_terminal_incorrect)
        ),
        torch.zeros_like(reward)
    )
    reward += terminal_reward
    
    return reward


def soft_update_target_network(
    online_params,
    target_params,
    tau: float = 0.005,
):
    """
    Soft update target network parameters (Polyak averaging).
    
    θ_target = τ * θ_online + (1 - τ) * θ_target
    
    Args:
        online_params: Parameters of online network
        target_params: Parameters of target network
        tau: Interpolation parameter (0 = no update, 1 = hard update)
    """
    with torch.no_grad():
        for target_param, online_param in zip(target_params, online_params):
            target_param.data.copy_(
                tau * online_param.data + (1.0 - tau) * target_param.data
            )


def hard_update_target_network(online_params, target_params):
    """
    Hard update target network (copy online network exactly).
    
    Args:
        online_params: Parameters of online network
        target_params: Parameters of target network
    """
    with torch.no_grad():
        for target_param, online_param in zip(target_params, online_params):
            target_param.data.copy_(online_param.data)
