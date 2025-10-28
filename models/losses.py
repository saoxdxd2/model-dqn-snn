from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
import math

from models.replay_buffer import DQNReplayBuffer
from models.dqn_utils import (
    RunningStats,
    compute_shaped_reward,
    soft_update_target_network,
)

IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str, enable_dqn: bool = False):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        self.enable_dqn = enable_dqn
        
        # DQN components
        if enable_dqn:
            config = model.config
            self.replay_buffer = DQNReplayBuffer(
                capacity=config.dqn_buffer_capacity,
                rank=0,  # Will be set properly in distributed training
                world_size=1,
                compress=True,
            )
            self.reward_stats = RunningStats()
            self.dqn_step_counter = 0
            self.dqn_loss_weight = 0.1  # Conservative start
            
            # Blank identifier for filtering padding
            self.blank_identifier_id = None
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            # Preds
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            # Correctness
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                
                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses

        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")
        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })
        # Q continue (bootstrapping target loss); Alexia: This fits Q-learning, but seems totally unecessary
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")

            metrics["q_continue_loss"] = q_continue_loss.detach()
        
        # DQN loss computation
        dqn_loss = 0
        if self.enable_dqn and self.training:
            # Compute current accuracy for reward shaping
            curr_accuracy = torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), torch.zeros_like(seq_is_correct.float()))
            
            # Get previous accuracy from carry (or initialize to 0)
            prev_accuracy = new_carry.prev_accuracy if new_carry.prev_accuracy is not None else torch.zeros_like(curr_accuracy)
            
            # Update no improvement counter
            improvement = (curr_accuracy > prev_accuracy).long()
            if new_carry.no_improvement_counter is not None:
                no_improvement_counter = torch.where(improvement > 0, torch.zeros_like(new_carry.no_improvement_counter), new_carry.no_improvement_counter + 1)
            else:
                no_improvement_counter = torch.zeros_like(new_carry.steps)
            
            # Compute shaped rewards
            rewards = compute_shaped_reward(
                prev_accuracy=prev_accuracy,
                curr_accuracy=curr_accuracy,
                step=new_carry.steps,
                max_steps=self.model.config.halt_max_steps,
                is_terminal=new_carry.halted,
                seq_is_correct=seq_is_correct,
                no_improvement_counter=no_improvement_counter,
                reward_accuracy_scale=self.model.config.reward_accuracy_scale,
                reward_step_penalty=self.model.config.reward_step_penalty,
                reward_terminal_correct=self.model.config.reward_terminal_correct,
                reward_terminal_incorrect=self.model.config.reward_terminal_incorrect,
                reward_stagnation_penalty=self.model.config.reward_stagnation_penalty,
                reward_stagnation_threshold=self.model.config.reward_stagnation_threshold,
            )
            
            # Normalize rewards
            self.reward_stats.update(rewards.detach())
            normalized_rewards = rewards  # Keep unnormalized for now
            
            # Store transitions in replay buffer
            self._store_transitions(model_kwargs.get('carry'), new_carry, outputs, normalized_rewards)
            
            # Train DQN if buffer is ready
            if len(self.replay_buffer) >= self.model.config.dqn_buffer_min_size:
                dqn_loss = self._compute_dqn_loss()
                metrics["dqn_loss"] = dqn_loss.detach()
                metrics["dqn_buffer_size"] = len(self.replay_buffer)
                
                # Update target network (soft update every step)
                soft_update_target_network(
                    self.model.inner.q_head.parameters(),
                    self.model.inner.q_head_target.parameters(),
                    tau=self.model.config.dqn_target_tau
                )
            
            # Update carry with new tracking values
            new_carry.prev_accuracy = curr_accuracy
            new_carry.no_improvement_counter = no_improvement_counter
            new_carry.training_step = new_carry.training_step + 1
            
            self.dqn_step_counter += 1
            
            # Additional DQN metrics
            metrics["dqn_reward_mean"] = rewards.mean().detach()
            metrics["dqn_reward_std"] = rewards.std().detach()
            metrics["dqn_epsilon"] = compute_shaped_reward.__code__.co_consts[0] if hasattr(self, '_last_epsilon') else 0.0
        
        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}
        
        # Total loss
        total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)
        if self.enable_dqn and isinstance(dqn_loss, torch.Tensor):
            total_loss = total_loss + self.dqn_loss_weight * dqn_loss

        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()
    
    def _store_transitions(self, prev_carry, new_carry, outputs, rewards):
        """Store transitions in replay buffer."""
        if prev_carry is None or prev_carry.inner_carry is None:
            return
        
        # Extract states
        prev_state = prev_carry.inner_carry.z_H[:, 0]  # [batch, hidden_size]
        curr_state = new_carry.inner_carry.z_H[:, 0]   # [batch, hidden_size]
        
        # Extract actions (halt = 1, continue = 0)
        actions = new_carry.halted.long()
        
        # Extract other info
        steps = new_carry.steps
        dones = new_carry.halted
        
        # Unbatch and store individual transitions
        batch_size = prev_state.shape[0]
        for i in range(batch_size):
            # Skip padding sequences if blank_identifier_id is set
            if self.blank_identifier_id is not None:
                if 'puzzle_identifiers' in new_carry.current_data:
                    if new_carry.current_data['puzzle_identifiers'][i] == self.blank_identifier_id:
                        continue
            
            self.replay_buffer.push(
                state=prev_state[i],
                action=actions[i],
                reward=rewards[i],
                next_state=curr_state[i],
                done=dones[i],
                step=steps[i],
            )
    
    def _compute_dqn_loss(self):
        """Sample from buffer and compute DQN TD-error loss."""
        # Sample batch
        batch = self.replay_buffer.sample(
            self.model.config.dqn_batch_size,
            device='cuda'
        )
        
        # Q-values from online network
        q_values = self.model.inner.q_head(batch['state'])  # [batch, 2]
        # Select Q-value for taken action
        q_selected = q_values.gather(1, batch['action'].unsqueeze(1)).squeeze(1)  # [batch]
        
        # Target Q-values from target network
        with torch.no_grad():
            next_q_values = self.model.inner.q_head_target(batch['next_state'])  # [batch, 2]
            # Max Q-value for next state
            next_q_max = next_q_values.max(dim=1)[0]  # [batch]
            # Compute target: r + gamma * max_a' Q_target(s', a') * (1 - done)
            target_q = batch['reward'] + self.model.config.dqn_gamma * next_q_max * (~batch['done']).float()
        
        # Compute loss (Huber loss for stability)
        dqn_loss = F.smooth_l1_loss(q_selected, target_q)
        
        return dqn_loss

