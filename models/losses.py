from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
import math

from models.replay_buffer import DQNReplayBuffer
from models.dqn_utils import (
    RunningStats,
    soft_update_target_network,
)
from models.intrinsic_reward import IntrinsicRewardModule

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
    def __init__(self, model: nn.Module, loss_type: str, enable_dqn: bool = False, deep_supervision_steps: int = 1):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        self.enable_dqn = enable_dqn
        self.deep_supervision_steps = deep_supervision_steps  # Multi-step supervision (TRM paper: +20%)
        
        # DQN components
        if enable_dqn:
            config = model.config
            self.replay_buffer = DQNReplayBuffer(
                capacity=config.dqn_buffer_capacity,
                rank=0,  # Will be set properly in distributed training
                world_size=1,
                compress=True,
                prioritized=getattr(config, 'enable_prioritized_replay', False),
                alpha=getattr(config, 'per_alpha', 0.6),
                beta=getattr(config, 'per_beta', 0.4),
            )
            self.reward_stats = RunningStats()
            self.dqn_step_counter = 0
            # Adaptive DQN loss weight: curriculum learning schedule
            # Start low (0.01) when predictions are random, increase to 0.5 as model improves
            self.dqn_loss_weight_min = 0.01
            self.dqn_loss_weight_max = 0.5
            self.dqn_loss_weight = self.dqn_loss_weight_min
            self.running_accuracy = 0.0  # Track accuracy for weight adaptation
            
            # Intrinsic motivation for improved exploration
            self.intrinsic_reward = IntrinsicRewardModule(
                hidden_size=config.hidden_size,
                enable_count=getattr(config, 'enable_count_curiosity', True),
                enable_rnd=getattr(config, 'enable_rnd_curiosity', True),
                enable_dynamics=False,  # Disabled by default (more complex)
                count_beta=getattr(config, 'curiosity_count_beta', 0.05),
                rnd_weight=getattr(config, 'curiosity_rnd_weight', 0.1)
            )
            
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
        
        # Handle deep supervision and MTP outputs
        if isinstance(outputs, dict) and ('intermediate' in outputs or 'mtp_logits' in outputs):
            logits = outputs.get('final', outputs.get('logits', outputs))
            intermediate_logits = outputs.get('intermediate')
            mtp_logits = outputs.get('mtp_logits')  # Multi-token prediction outputs
            # Preserve critical keys: q_halt_logits, q_continue_logits, z_H
            q_halt_logits = outputs.get('q_halt_logits')
            q_continue_logits = outputs.get('q_continue_logits')
            z_H = outputs.get('z_H')
            # Rebuild outputs dict with proper structure
            outputs = {'logits': logits}
            if q_halt_logits is not None:
                outputs['q_halt_logits'] = q_halt_logits
            if q_continue_logits is not None:
                outputs['q_continue_logits'] = q_continue_logits
            if z_H is not None:
                outputs['z_H'] = z_H
        else:
            logits = outputs if not isinstance(outputs, dict) else outputs.get('logits', outputs)
            intermediate_logits = None
            mtp_logits = None
            # Replace outputs with proper structure if needed
            if not isinstance(outputs, dict) or 'logits' not in outputs:
                outputs = {'logits': logits}

        with torch.no_grad():
            # Preds
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            # Correctness
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Deep supervision: compute correctness for intermediate outputs
            if intermediate_logits is not None:
                intermediate_correct = []
                for inter_logits in intermediate_logits:
                    inter_is_correct = mask & (torch.argmax(inter_logits, dim=-1) == labels)
                    inter_seq_correct = inter_is_correct.sum(-1) == loss_counts
                    intermediate_correct.append(inter_seq_correct)
            else:
                intermediate_correct = None
            
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
        
        # Deep supervision: add intermediate losses (TRM paper: +20% accuracy)
        if intermediate_logits is not None and self.training:
            deep_supervision_losses = []
            for inter_logits in intermediate_logits:
                inter_loss = (self.loss_fn(inter_logits, labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()
                deep_supervision_losses.append(inter_loss)
            
            # Average intermediate losses with final loss
            total_supervision_loss = sum(deep_supervision_losses) / len(deep_supervision_losses)
            lm_loss = (lm_loss + total_supervision_loss) / 2.0  # Balance final and intermediate
            metrics["deep_supervision_loss"] = total_supervision_loss.detach()
        
        # Multi-Token Prediction loss (DeepSeek-V3: +10-15% data efficiency)
        mtp_loss = 0
        if mtp_logits is not None and self.training:
            mtp_losses = []
            for depth, depth_logits in enumerate(mtp_logits):
                # Target is shifted labels (predict future tokens)
                depth_targets = torch.cat([labels[:, depth+1:], torch.full((labels.size(0), depth+1), IGNORE_LABEL_ID, dtype=labels.dtype, device=labels.device)], dim=1)
                depth_loss = (self.loss_fn(depth_logits, depth_targets, ignore_index=IGNORE_LABEL_ID, valid_mask=(depth_targets != IGNORE_LABEL_ID)) / loss_divisor).sum()
                mtp_losses.append(depth_loss)
            
            # Average across depths (as per DeepSeek paper)
            mtp_loss = sum(mtp_losses) / len(mtp_losses)
            metrics["mtp_loss"] = mtp_loss.detach()
            metrics["mtp_num_depths"] = len(mtp_logits)
        
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")
        
        # Entropy regularization for exploration (encourage diverse Q-value distributions)
        entropy_bonus = 0
        if self.enable_dqn and self.model.config.enable_entropy_regularization:
            # Compute entropy of Q-value distribution: H(π) = -Σ π(a) log π(a)
            # where π(a) = softmax(Q(s,a) / temperature)
            q_probs = torch.softmax(torch.stack([outputs["q_continue_logits"], outputs["q_halt_logits"]], dim=-1), dim=-1)
            entropy = -(q_probs * torch.log(q_probs + 1e-8)).sum(-1)  # [batch]
            entropy_bonus = -self.model.config.entropy_regularization_weight * entropy.sum()  # Negative because we want to maximize entropy
            metrics["q_entropy"] = entropy.mean().detach()
        
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
            
            # Simplified reward: r = Δacc - 0.01*step + terminal_bonus
            # (aligned with training strategy from memory)
            accuracy_improvement = curr_accuracy - prev_accuracy
            step_penalty = self.model.config.reward_step_penalty
            
            # Terminal bonus: fixed rewards for correct/incorrect
            terminal_bonus = torch.where(
                new_carry.halted,
                torch.where(
                    seq_is_correct,
                    torch.full_like(curr_accuracy, self.model.config.reward_terminal_correct),
                    torch.full_like(curr_accuracy, self.model.config.reward_terminal_incorrect)
                ),
                torch.zeros_like(curr_accuracy)
            )
            
            # Memory bonus (if enabled) - write to memory AFTER rewards computed
            memory_bonus = torch.zeros_like(curr_accuracy)
            if self.model.config.enable_memory and self.model.inner.memory is not None:
                # Reward bonus for memory-assisted improvements
                memory_bonus = torch.where(
                    accuracy_improvement > 0,
                    torch.full_like(curr_accuracy, self.model.config.memory_reward_bonus),
                    torch.zeros_like(curr_accuracy)
                )
                
                # Write high-reward states to memory (TEMPORAL COHERENCE FIX)
                # This happens AFTER accuracy is computed, so memory is ready for NEXT step
                if 'z_H' in outputs:
                    state_for_memory = outputs['z_H'][:, 0].detach()  # [batch, hidden_size]
                    # Normalized reward for storage decision
                    reward_for_memory = accuracy_improvement + terminal_bonus
                    self.model.inner.memory.write(
                        state_for_memory,
                        reward_for_memory,
                        threshold=self.model.config.memory_reward_threshold
                    )
            
            # Compute intrinsic curiosity bonus for exploration
            intrinsic_bonus = torch.zeros_like(curr_accuracy)
            if hasattr(self, 'intrinsic_reward') and 'z_H' in outputs:
                current_state = outputs['z_H'][:, 0].detach()  # [batch, hidden_size]
                intrinsic_bonus = self.intrinsic_reward.compute_intrinsic_reward(current_state)
            
            # Combined reward: extrinsic + intrinsic
            # r_total = r_extrinsic + β * r_intrinsic
            extrinsic_reward = accuracy_improvement - step_penalty + terminal_bonus + memory_bonus
            rewards = extrinsic_reward + intrinsic_bonus
            
            # Normalize rewards for stability
            self.reward_stats.update(rewards.detach())
            normalized_rewards = self.reward_stats.normalize(rewards) if self.reward_stats.count > 100 else rewards
            
            # Store transitions in replay buffer
            self._store_transitions(model_kwargs.get('carry'), new_carry, outputs, normalized_rewards)
            
            # Compute epsilon for current step
            from models.dqn_utils import compute_epsilon
            config = self.model.config
            current_epsilon = compute_epsilon(
                self.dqn_step_counter,
                epsilon_start=config.dqn_epsilon_start,
                epsilon_end=config.dqn_epsilon_end,
                epsilon_decay_steps=config.dqn_epsilon_decay_steps
            )
            
            # DQN Loss (if buffer has enough samples)
            if len(self.replay_buffer) >= config.dqn_buffer_min_size:
                device = outputs['logits'].device if 'logits' in outputs else 'cuda'
                batch = self.replay_buffer.sample(config.dqn_batch_size, device=device)
                
                with torch.no_grad():
                    # Target Q-values
                    target_q_values = self.model.inner.q_head(batch['next_state'])
                    target_q_max = target_q_values.max(dim=1)[0]
                    
                    # Bellman target
                    targets = batch['reward'] + config.dqn_gamma * target_q_max * (~batch['done'])
                
                # Current Q-values
                current_q_values = self.model.inner.q_head(batch['state'])
                current_q = current_q_values.gather(1, batch['action'].unsqueeze(1)).squeeze(1)
                
                # TD error (for prioritized replay)
                td_errors = current_q - targets
                
                # Importance sampling weighted loss (for prioritized replay)
                weights = batch.get('weights', torch.ones_like(td_errors))
                dqn_loss = (weights * td_errors.pow(2)).mean()
                
                # Update priorities in replay buffer
                if hasattr(self.replay_buffer, 'update_priorities'):
                    self.replay_buffer.update_priorities(batch['indices'], td_errors)
                
                metrics["dqn_loss"] = dqn_loss.detach()
                metrics["dqn_q_mean"] = current_q.mean().detach()
                metrics["dqn_target_mean"] = targets.mean().detach()
                metrics["dqn_td_error_mean"] = td_errors.abs().mean().detach()
            
            self.dqn_step_counter += 1
            
            # Adaptive DQN loss weight: curriculum learning
            # Update running accuracy (exponential moving average)
            current_batch_accuracy = curr_accuracy.mean().item()
            self.running_accuracy = 0.99 * self.running_accuracy + 0.01 * current_batch_accuracy
            
            # Increase DQN weight as model improves (sigmoid schedule)
            # At 0% acc → weight = 0.01, at 50% acc → weight = 0.5
            accuracy_ratio = min(self.running_accuracy / 0.5, 1.0)  # Normalize to [0, 1]
            self.dqn_loss_weight = self.dqn_loss_weight_min + (self.dqn_loss_weight_max - self.dqn_loss_weight_min) * accuracy_ratio
            
            # Additional DQN metrics
            metrics["dqn_reward_mean"] = rewards.mean().detach()
            metrics["dqn_reward_std"] = rewards.std().detach()
            metrics["dqn_reward_extrinsic"] = extrinsic_reward.mean().detach()
            metrics["dqn_reward_intrinsic"] = intrinsic_bonus.mean().detach()
            metrics["dqn_epsilon"] = current_epsilon
            metrics["dqn_accuracy_improvement"] = accuracy_improvement.mean().detach()
            metrics["dqn_loss_weight_adaptive"] = self.dqn_loss_weight
            metrics["dqn_running_accuracy"] = self.running_accuracy
            
            # Memory bank metrics (if enabled)
            if self.model.config.enable_memory and self.model.inner.memory is not None:
                memory_stats = self.model.inner.memory.get_memory_stats()
                metrics["memory_utilization"] = memory_stats['utilization']
                metrics["memory_active_slots"] = memory_stats['active_slots']
                metrics["memory_total_accesses"] = memory_stats['total_accesses']
                metrics["memory_bonus_mean"] = memory_bonus.mean().detach()
                
                # L1 Cache statistics (valuable for inference optimization)
                cache_stats = self.model.inner.memory.get_cache_stats()
                metrics["memory_cache_hit_rate"] = cache_stats['hit_rate']
                metrics["memory_cache_entries"] = cache_stats['cache_entries']
                metrics["memory_cache_hits"] = cache_stats['cache_hits']
                metrics["memory_cache_misses"] = cache_stats['cache_misses']
        
        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}
        
        # Total loss
        total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss) + entropy_bonus
        if self.enable_dqn and isinstance(dqn_loss, torch.Tensor):
            total_loss = total_loss + self.dqn_loss_weight * dqn_loss
        
        # Add MTP loss with configurable weight (lambda parameter)
        if isinstance(mtp_loss, torch.Tensor) and mtp_loss != 0:
            mtp_weight = getattr(self.model.config, 'mtp_loss_weight', 0.5)
            total_loss = total_loss + mtp_weight * mtp_loss
            metrics["mtp_loss_weighted"] = (mtp_weight * mtp_loss).detach()

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

