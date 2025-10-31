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
                td_threshold=getattr(config, 'dqn_td_threshold', 0.0),  # Selective storage
            )
            self.reward_stats = RunningStats()
            self.dqn_step_counter = 0
            # Adaptive DQN loss weight: curriculum learning schedule
            # Start low (0.005) when predictions are random, increase to 0.5 as model improves
            self.dqn_loss_weight_min = 0.005  # Lowered from 0.01 to prevent early instability
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
            
            # For text generation: also compute confidence-based halt target
            # This allows halting based on model confidence, not just correctness
            probs = torch.softmax(outputs["logits"], dim=-1)
            max_probs = probs.max(dim=-1).values  # [batch, seq]
            # Average confidence across sequence (only on valid tokens)
            avg_confidence = torch.where(mask, max_probs, 0.0).sum(-1) / loss_counts.clamp_min(1)
            # High confidence → should halt (model is certain)
            confidence_halt_target = avg_confidence > 0.8
            
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
        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID) / loss_divisor).sum()
        
        # Deep supervision: add intermediate losses (TRM paper: +20% accuracy)
        if intermediate_logits is not None and self.training:
            deep_supervision_losses = []
            for inter_logits in intermediate_logits:
                inter_loss = (self.loss_fn(inter_logits, labels, ignore_index=IGNORE_LABEL_ID) / loss_divisor).sum()
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
                depth_loss = (self.loss_fn(depth_logits, depth_targets, ignore_index=IGNORE_LABEL_ID) / loss_divisor).sum()
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
        
        # HESC reconstruction loss (capsule-aware)
        reconstruction_loss = 0
        if (hasattr(self.model.inner, 'capsule_encoder') and 
            self.model.inner.capsule_encoder is not None and
            'capsule_sketches' in new_carry.current_data and
            'z_H' in outputs):
            try:
                # Reconstruction: TRM output → should reconstruct original sketches
                original_sketches = new_carry.current_data['capsule_sketches']
                
                # Pool TRM hidden states
                k = min(original_sketches.size(1), outputs['z_H'].size(1))
                trm_output = outputs['z_H'][:, :k]
                orig_sketch = original_sketches[:, :k]
                
                # Cosine similarity loss (preserve semantic content)
                if trm_output.size(-1) == orig_sketch.size(-1):
                    reconstruction_loss = 1 - F.cosine_similarity(
                        trm_output.reshape(-1, trm_output.size(-1)),
                        orig_sketch.reshape(-1, orig_sketch.size(-1)),
                        dim=-1
                    ).mean()
                
                # Cycle-consistency: sketch → children → reconstructed sketch
                if 'capsule_children' in new_carry.current_data:
                    children = new_carry.current_data['capsule_children']
                    # Pool children back to sketch representation
                    reconstructed_sketch = children.mean(dim=2)  # [B, k, hidden]
                    cycle_loss = F.mse_loss(reconstructed_sketch[:, :k], orig_sketch)
                    reconstruction_loss = reconstruction_loss + 0.2 * cycle_loss
                
                # Add checksum consistency if available
                if 'capsule_checksums' in new_carry.current_data:
                    checksums = new_carry.current_data['capsule_checksums']
                    # Checksum should signal high reconstructability
                    checksum_loss = -checksums.norm(dim=-1).mean() * 0.01
                    reconstruction_loss += checksum_loss
                    
            except Exception:
                reconstruction_loss = 0
        
                
        # Expansion cost tracking (HESC)
        expansion_cost = 0
        if 'num_expansions' in new_carry.current_data:
            num_expansions = new_carry.current_data['num_expansions']
            children_per_capsule = getattr(self.model.config, 'children_per_capsule', 4)
            expansion_cost = 0.01 * num_expansions.float().mean() * children_per_capsule
        
        # VQ codebook loss (for concept vocabulary)
        vq_loss = 0
        if 'vq_loss' in outputs:
            vq_loss = outputs['vq_loss']
        
        # Per-capsule reconstruction metrics for detailed monitoring
        if (isinstance(reconstruction_loss, torch.Tensor) and 
            'capsule_sketches' in new_carry.current_data and 
            'z_H' in outputs):
            try:
                original_sketches = new_carry.current_data['capsule_sketches']
                k = min(original_sketches.size(1), outputs['z_H'].size(1))
                trm_output = outputs['z_H'][:, :k]
                orig_sketch = original_sketches[:, :k]
                
                if trm_output.size(-1) == orig_sketch.size(-1):
                    # Per-capsule cosine similarity [B, k]
                    per_capsule_cos = F.cosine_similarity(
                        trm_output.reshape(-1, k, trm_output.size(-1)),
                        orig_sketch.reshape(-1, k, orig_sketch.size(-1)),
                        dim=-1
                    )
                    metrics["cos_sim_mean"] = per_capsule_cos.mean().detach()
                    metrics["cos_sim_min"] = per_capsule_cos.min().detach()
                    metrics["cos_sim_std"] = per_capsule_cos.std().detach()
            except Exception:
                pass
        
        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
            "reconstruction_loss": reconstruction_loss.detach() if isinstance(reconstruction_loss, torch.Tensor) else 0,
            "expansion_cost": expansion_cost.detach() if isinstance(expansion_cost, torch.Tensor) else 0,
            "vq_loss": vq_loss.detach() if isinstance(vq_loss, torch.Tensor) else 0,
        })
        # Q continue (bootstrapping target loss); Alexia: This fits Q-learning, but seems totally unecessary
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")

            metrics["q_continue_loss"] = q_continue_loss.detach()
        
        # DQN loss computation
        dqn_loss = 0
        if self.enable_dqn and self.training:
            # Oracle warm-start: use heuristic policy for first N steps
            warmstart_steps = getattr(self.model.config, 'dqn_warmstart_steps', 10000)
            use_oracle = self.dqn_step_counter < warmstart_steps
            
            if use_oracle and 'capsule_children' in new_carry.current_data:
                # Heuristic oracle: expand capsules with high child-reconstruction error
                with torch.no_grad():
                    children = new_carry.current_data['capsule_children']
                    sketches = new_carry.current_data['capsule_sketches']
                    
                    # Compute reconstruction error per capsule
                    k = min(children.size(1), sketches.size(1))
                    child_avg = children[:, :k].mean(dim=2)  # [B, k, hidden]
                    recon_error = 1 - F.cosine_similarity(
                        child_avg.reshape(-1, child_avg.size(-1)),
                        sketches[:, :k].reshape(-1, sketches.size(-1)),
                        dim=-1
                    ).reshape(child_avg.size(0), -1)  # [B, k]
                    
                    # Oracle decision: expand if any capsule has high error
                    should_expand_oracle = (recon_error > 0.1).any(dim=-1)  # [B]
                    metrics["oracle_expansion_rate"] = should_expand_oracle.float().mean().detach()
                    metrics["oracle_recon_error"] = recon_error.mean().detach()
            
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
            
            # Replay buffer statistics (monitor selective storage)
            if len(self.replay_buffer) > 0:
                buffer_stats = self.replay_buffer.get_stats()
                metrics["replay_buffer_size"] = buffer_stats['size']
                metrics["replay_buffer_utilization"] = buffer_stats['utilization']
                metrics["replay_buffer_rejection_rate"] = buffer_stats['rejection_rate']
                metrics["replay_buffer_rejected_count"] = buffer_stats['rejected_count']
        
        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}
        
        # Total loss with HESC components
        total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss) + entropy_bonus
        
        # Add reconstruction loss (weight: 0.5) - Increased to prevent reconstructability collapse
        if isinstance(reconstruction_loss, torch.Tensor) and reconstruction_loss != 0:
            total_loss = total_loss + 0.5 * reconstruction_loss
        
        # Add expansion penalty (weight: 0.1)
        if isinstance(expansion_cost, torch.Tensor) and expansion_cost != 0:
            total_loss = total_loss + 0.1 * expansion_cost
        
        # Add VQ codebook loss (weight: 0.25)
        if isinstance(vq_loss, torch.Tensor) and vq_loss != 0:
            total_loss = total_loss + 0.25 * vq_loss
        
        # Add DQN loss
        if self.enable_dqn and isinstance(dqn_loss, torch.Tensor):
            total_loss = total_loss + self.dqn_loss_weight * dqn_loss
        
        # Add MTP loss with configurable weight (lambda parameter)
        if isinstance(mtp_loss, torch.Tensor) and mtp_loss != 0:
            mtp_weight = getattr(self.model.config, 'mtp_loss_weight', 0.5)
            total_loss = total_loss + mtp_weight * mtp_loss
            metrics["mtp_loss_weighted"] = (mtp_weight * mtp_loss).detach()

        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()
    
    def _store_transitions(self, prev_carry, new_carry, outputs, rewards):
        """Store transitions in replay buffer with TD-error for selective storage."""
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
        
        # Compute TD-error for selective storage (Schaul et al., 2015)
        # TD-error = |Q(s,a) - (r + γ * max_a' Q(s',a'))|
        td_errors = None
        if hasattr(self.model.inner, 'q_head'):
            with torch.no_grad():
                # Current Q-values
                q_values = self.model.inner.q_head(prev_state)  # [batch, 2]
                current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # [batch]
                
                # Target Q-values (use target network if available)
                q_head_target = getattr(self.model.inner, 'q_head_target', self.model.inner.q_head)
                next_q_values = q_head_target(curr_state)  # [batch, 2]
                next_q_max = next_q_values.max(dim=1)[0]  # [batch]
                
                # TD target
                gamma = self.model.config.dqn_gamma if hasattr(self.model.config, 'dqn_gamma') else 0.99
                targets = rewards + gamma * next_q_max * (~dones).float()
                
                # TD-error (absolute)
                td_errors = (current_q - targets).abs().cpu().numpy()
        
        # Unbatch and store individual transitions
        batch_size = prev_state.shape[0]
        for i in range(batch_size):
            # Skip padding sequences if blank_identifier_id is set
            if self.blank_identifier_id is not None:
                if 'puzzle_identifiers' in new_carry.current_data:
                    if new_carry.current_data['puzzle_identifiers'][i] == self.blank_identifier_id:
                        continue
            
            # Extract TD-error for this sample (if available)
            td_error_val = float(td_errors[i]) if td_errors is not None else None
            
            self.replay_buffer.push(
                state=prev_state[i],
                action=actions[i],
                reward=rewards[i],
                next_state=curr_state[i],
                done=dones[i],
                step=steps[i],
                td_error=td_error_val,  # Pass TD-error for selective storage
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

