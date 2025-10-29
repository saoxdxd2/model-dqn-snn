from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import copy
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from functools import partial
from pydantic import BaseModel
import random
from models.common import trunc_normal_init_
from models.layers import rms_norm, LinearSwish, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding
from models.common import *
from models.dqn_utils import compute_epsilon, select_action_epsilon_greedy
from models.q_heads import create_q_head
from models.memory_bank import AssociativeMemoryBank
from models.cnn_tokenizer import CNNTokenizer

# Selective checkpointing policy: save matmuls, recompute cheap ops
try:
    from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts
    
    def selective_checkpoint_policy(ctx, op, *args, **kwargs):
        """Save compute-intensive ops (matmuls, attention), recompute cheap ops (pointwise)."""
        compute_intensive_ops = [
            torch.ops.aten.mm,
            torch.ops.aten.bmm,
            torch.ops.aten.addmm,
            torch.ops.aten._scaled_dot_product_flash_attention,
            torch.ops.aten._scaled_dot_product_efficient_attention,
        ]
        if op in compute_intensive_ops:
            return CheckpointPolicy.MUST_SAVE
        else:
            return CheckpointPolicy.PREFER_RECOMPUTE
    
    SELECTIVE_CHECKPOINT_AVAILABLE = True
except ImportError:
    SELECTIVE_CHECKPOINT_AVAILABLE = False
    selective_checkpoint_policy = None

IGNORE_LABEL_ID = -100

@dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class TinyRecursiveReasoningModel_ACTV1Carry:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]
    
    # DQN tracking
    prev_accuracy: Optional[torch.Tensor] = None
    training_step: int = 0  # Global training step for epsilon decay
    
    # Puzzle boundary tracking (for RNN Q-head state reset)
    puzzle_ids: Optional[torch.Tensor] = None  # Track current puzzle IDs to detect changes


class TinyRecursiveReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int  # Output vocabulary (classes for vision, tokens for text)
    input_vocab_size: Optional[int] = None  # Input vocabulary (patch tokens for vision, same as vocab_size for text)

    H_cycles: int
    L_cycles: int

    H_layers: int # ignored
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    # Alexia: added
    mlp_t: bool = False # use mlp on L instead of transformer
    puzzle_emb_len: int = 16 # if non-zero, its specified to this value
    no_ACT_continue: bool =  True # No continue ACT loss, only use the sigmoid of the halt which makes much more sense
    use_learned_halting_eval: bool = True  # Enable learned halting during evaluation
    
    # Text generation mode
    causal: bool = False  # Enable causal attention for autoregressive text generation
    
    # DQN Enhancement Parameters
    enable_dqn: bool = False  # Enable DQN-based halting
    dqn_buffer_capacity: int = 20000
    dqn_buffer_min_size: int = 5000
    dqn_batch_size: int = 256
    dqn_gamma: float = 0.99
    dqn_target_tau: float = 0.005  # Soft update rate
    dqn_epsilon_start: float = 0.5
    dqn_epsilon_end: float = 0.05
    dqn_epsilon_decay_steps: int = 100000
    
    # Simplified Reward Shaping (3 parameters)
    reward_step_penalty: float = 0.01
    reward_terminal_correct: float = 1.0
    reward_terminal_incorrect: float = -0.5
    
    # Q-Head Architecture
    q_head_type: str = "mlp"  # Options: "mlp", "rnn", "mini_attention"
    q_head_hidden_size: int = 128  # Hidden size for RNN/attention Q-head
    q_head_num_layers: int = 1  # Number of RNN layers or attention heads

    # Exploration Enhancement
    entropy_regularization_weight: float = 0.01  # Weight for entropy bonus in Q-value loss
    enable_entropy_regularization: bool = True   # Enable entropy regularization for exploration
    
    # Export Configuration (for CPU inference)
    export_to_snn: bool = False     # Convert Q-head to Spiking Neural Network
    export_to_bnn: bool = False     # Convert Q-head to Binary Neural Network (1-bit weights)

    # Memory Bank Configuration
    enable_memory: bool = False
    memory_capacity: int = 4096
    memory_num_heads: int = 8
    memory_dropout: float = 0.1
    memory_reward_bonus: float = 0.5
    memory_reward_threshold: float = 0.05  # Match YAML default
    
    # Multi-Token Prediction (DeepSeek-V3 inspired)
    enable_mtp: bool = False  # Enable multi-token prediction for better data efficiency
    mtp_num_depths: int = 3  # Number of future tokens to predict (D in paper)
    mtp_loss_weight: float = 0.5  # Lambda: weight for MTP loss vs main loss
    mtp_share_embeddings: bool = True  # Share embedding across depths (parameter savings)
    mtp_share_output_head: bool = True  # Share output head across depths
    
    # CPU Optimization Flags
    cpu_optimized: bool = False  # Enable CPU-friendly optimizations
    use_gelu: bool = True  # GELU (GPU) vs ReLU (CPU faster)
    ffn_expansion_ratio: float = 4.0  # 4.0 for GPU, 2.0 for CPU


class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()

        self.config = config
        if self.config.mlp_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len, # L
                expansion=config.expansion,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=config.causal  # Use config setting for causal/non-causal
            )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # B, L, D = hidden_states.shape
        # Post Norm
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1,2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1,2)
        else:
            # Self Attention
            hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Fully Connected
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states

class TinyRecursiveReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[TinyRecursiveReasoningModel_ACTV1Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class TinyRecursiveReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        # Support separate input/output vocabularies (for vision classification)
        # input_vocab_size: patch tokens (e.g., 2048 for VQ-VAE)
        # output_vocab_size: classes (e.g., 10 for CIFAR-10)
        # For text: input_vocab_size not set, so default to vocab_size
        input_vocab_size = getattr(self.config, 'input_vocab_size', None)
        if input_vocab_size is None:
            input_vocab_size = self.config.vocab_size
        output_vocab_size = self.config.vocab_size  # Always use vocab_size for output
        
        print(f"\n📊 Vocabulary Config:")
        print(f"   Input vocab: {input_vocab_size}")
        print(f"   Output vocab: {output_vocab_size}")
        print(f"   Hidden size: {self.config.hidden_size}")
        
        # Vision: CNN tokenizer or embedding table
        use_cnn_tokenizer = getattr(self.config, 'use_cnn_tokenizer', False)
        if use_cnn_tokenizer:
            # CNN stem for vision (CCT-style)
            cnn_in_channels = getattr(self.config, 'cnn_in_channels', 3)
            cnn_conv_channels = getattr(self.config, 'cnn_conv_channels', [64, 128, self.config.hidden_size])
            self.cnn_tokenizer = CNNTokenizer(
                in_channels=cnn_in_channels,
                hidden_size=self.config.hidden_size,
                num_conv_layers=len(cnn_conv_channels),
                conv_channels=cnn_conv_channels,
                use_batch_norm=True,
                activation='relu'
            )
            self.embed_tokens = None  # No embedding table needed
        else:
            # Standard embedding table
            self.embed_tokens = CastedEmbedding(input_vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
            self.cnn_tokenizer = None
        
        self.lm_head = CastedLinear(self.config.hidden_size, output_vocab_size, bias=False)
        
        # Q-head: Configurable architecture (MLP/RNN/Mini-Attention)
        self.q_head = create_q_head(self.config)
        
        # DQN: Target network for stable Q-learning
        if self.config.enable_dqn:
            self.q_head_target = create_q_head(self.config)
            self.q_head_target.load_state_dict(self.q_head.state_dict())
            # Freeze target network
            for param in self.q_head_target.parameters():
                param.requires_grad = False

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            pass

        # Reasoning Layers
        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])

        # Memory Bank (optional)
        self.memory = None
        if self.config.enable_memory:
            self.memory = AssociativeMemoryBank(
                capacity=self.config.memory_capacity,
                hidden_size=self.config.hidden_size,
                num_heads=self.config.memory_num_heads,
                dropout=self.config.memory_dropout
            )
        
        # Multi-Token Prediction (optional, DeepSeek-V3 inspired)
        self.mtp_modules = None
        if self.config.enable_mtp:
            # Create lightweight MTP modules that reuse embeddings/output head
            self.mtp_modules = nn.ModuleList([
                self._create_mtp_module() for _ in range(self.config.mtp_num_depths)
            ])

        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

    def _create_mtp_module(self):
        """Create single MTP module that shares weights with main model."""
        class MTPDepthModule(nn.Module):
            def __init__(self, config, embedding, output_head):
                super().__init__()
                self.embedding = embedding  # Shared
                self.output_head = output_head  # Shared
                # Projection: [h_prev; token_emb] -> h_combined
                self.projection = nn.Linear(config.hidden_size * 2, config.hidden_size)
                # Lightweight transformer (1 layer)
                self.transformer = TinyRecursiveReasoningModel_ACTV1Block(config)
            
            def forward(self, h_prev, next_token_ids, **seq_info):
                # Embed next tokens
                token_emb = self.embedding(next_token_ids.to(torch.int32))
                # Combine with previous state
                combined = torch.cat([h_prev, token_emb], dim=-1)
                h_combined = self.projection(combined)
                # Transform
                h_current = self.transformer(h_combined, **seq_info)
                # Predict
                logits = self.output_head(h_current)
                return h_current, logits
        
        return MTPDepthModule(
            self.config,
            self.embed_tokens if self.embed_tokens is not None else None,
            self.lm_head
        )
    
    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding: CNN tokenizer or embedding table
        if self.cnn_tokenizer is not None:
            # CNN tokenizer: input is raw images
            # Transpose from [B, H, W, C] to [B, C, H, W] (PyTorch convention)
            if input.dim() == 4 and input.shape[-1] in [1, 3]:  # Detect [B, H, W, C]
                input = input.permute(0, 3, 1, 2)  # -> [B, C, H, W]
            embedding = self.cnn_tokenizer(input)  # -> [B, seq_len, hidden_size]
        else:
            # Standard embedding: input is token IDs [B, seq_len]
            embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int, seq_len: int = None):
        # Use provided seq_len or fall back to config
        if seq_len is None:
            seq_len = self.config.seq_len
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry):
        # Expand init states to match carry sequence length
        batch_size, seq_len, hidden_size = carry.z_H.shape
        H_init_expanded = self.H_init.view(1, 1, -1).expand(batch_size, seq_len, hidden_size)
        L_init_expanded = self.L_init.view(1, 1, -1).expand(batch_size, seq_len, hidden_size)
        
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), H_init_expanded, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), L_init_expanded, carry.z_L),
        )

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor], 
                enable_deep_supervision: bool = False, supervision_steps: int = 4) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Forward iterations
        z_H, z_L = carry.z_H, carry.z_L
        
        # Deep supervision: collect intermediate outputs
        intermediate_outputs = [] if enable_deep_supervision else None
        supervision_interval = max(1, self.config.H_cycles // supervision_steps) if enable_deep_supervision else self.config.H_cycles
        
        # Process H-cycles with gradient checkpointing for memory efficiency
        for h_step in range(self.config.H_cycles):
            # Determine if this step should have gradients
            is_supervision_step = enable_deep_supervision and (h_step % supervision_interval == 0 or h_step == self.config.H_cycles - 1)
            is_final_step = (h_step == self.config.H_cycles - 1)
            
            # Dynamic memory query at each H-cycle
            memory_content = None
            if self.memory is not None:
                with torch.no_grad():  # Memory read is always no_grad for stability
                    memory_content = self.memory.read(z_H[:, 0])  # [batch, hidden_size]
                    memory_content = memory_content.unsqueeze(1)  # [batch, 1, hidden_size]
            
            # L-cycles with memory-enhanced input
            for _L_step in range(self.config.L_cycles):
                h_input = z_H + input_embeddings
                if memory_content is not None:
                    h_input = h_input + memory_content
                z_L = self.L_level(z_L, h_input, **seq_info)
            z_H = self.L_level(z_H, z_L, **seq_info)
            
            # Collect intermediate output for supervision (keep in graph)
            if is_supervision_step and intermediate_outputs is not None:
                intermediate_output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
                intermediate_outputs.append(intermediate_output)
            
            # Detach for next iteration ONLY if not supervision step
            # This ensures gradients flow through all supervision checkpoints
            if not is_final_step and not is_supervision_step:
                z_H = z_H.detach()
                z_L = z_L.detach()

        # Final outputs
        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        
        # Multi-Token Prediction (if enabled)
        mtp_logits = None
        if self.mtp_modules is not None and self.training:
            seq_info_mtp = dict(cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None)
            mtp_logits = []
            h_prev = z_H[:, self.puzzle_emb_len:]  # Remove puzzle embeddings
            
            for depth, mtp_module in enumerate(self.mtp_modules):
                # Get next token IDs (shifted by depth+1)
                if depth + 1 < batch["inputs"].size(1):
                    next_tokens = batch["inputs"][:, depth+1:]
                    # Pad to match sequence length
                    if next_tokens.size(1) < batch["inputs"].size(1):
                        next_tokens = F.pad(next_tokens, (0, batch["inputs"].size(1) - next_tokens.size(1)))
                else:
                    next_tokens = torch.zeros_like(batch["inputs"])
                
                h_current, logits = mtp_module(h_prev, next_tokens, **seq_info_mtp)
                mtp_logits.append(logits)
                h_prev = h_current
        
        # Store intermediate outputs and MTP in dict
        if intermediate_outputs or mtp_logits:
            output_dict = {'final': output if not isinstance(output, dict) else output}
            if intermediate_outputs:
                output_dict['intermediate'] = intermediate_outputs
            if mtp_logits:
                output_dict['mtp_logits'] = mtp_logits
            output = output_dict
        
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModel_ACTV1Config(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        seq_len = batch["inputs"].shape[1]  # Get actual sequence length from input
        device = batch["inputs"].device

        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size, seq_len=seq_len),  # Use dynamic seq_len
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size, ), dtype=torch.bool, device=device),  # Default to halted
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
            
            # DQN tracking
            prev_accuracy=torch.zeros((batch_size, ), dtype=torch.float32, device=device) if self.config.enable_dqn else None,
            training_step=0,
            
            # Puzzle boundary tracking
            puzzle_ids=torch.zeros((batch_size, ), dtype=torch.int64, device=device) if hasattr(self.inner, 'q_head') and hasattr(self.inner.q_head, 'reset_hidden') else None
        )
        
    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:

        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model (enable deep supervision in training)
        enable_deep_supervision = self.training and hasattr(self.config, 'deep_supervision_steps')
        supervision_steps = getattr(self.config, 'deep_supervision_steps', 1)
        new_inner_carry, logits_or_dict, (q_halt_logits, q_continue_logits) = self.inner(
            new_inner_carry, new_current_data, 
            enable_deep_supervision=enable_deep_supervision,
            supervision_steps=supervision_steps
        )

        # Unwrap deep supervision outputs (critical for workflow)
        if isinstance(logits_or_dict, dict) and 'final' in logits_or_dict:
            # Deep supervision: pass through as-is for loss computation
            outputs = logits_or_dict  # Contains 'final' and 'intermediate'
            outputs['q_halt_logits'] = q_halt_logits
            outputs['q_continue_logits'] = q_continue_logits
            outputs['z_H'] = new_inner_carry.z_H
        else:
            # Normal: just tensor
            outputs = {
                "logits": logits_or_dict,
                "q_halt_logits": q_halt_logits,
                "q_continue_logits": q_continue_logits,
                "z_H": new_inner_carry.z_H
            }

        with torch.no_grad():
            # Initialize halted flag
            halted = carry.halted
            
            # Detect puzzle boundary changes (new puzzle started)
            puzzle_changed = torch.zeros_like(halted)
            new_puzzle_ids = carry.puzzle_ids  # Default to current
            
            if carry.puzzle_ids is not None and 'puzzle_identifiers' in new_current_data:
                current_puzzle_ids = new_current_data['puzzle_identifiers'].view(-1)
                puzzle_changed = (carry.puzzle_ids != current_puzzle_ids) & (carry.puzzle_ids != 0)
                # CRITICAL: Update tracked puzzle IDs BEFORE reset to ensure proper state tracking
                new_puzzle_ids = torch.where(carry.halted, current_puzzle_ids, carry.puzzle_ids)
            
            # Reset RNN Q-head state for puzzle boundaries (CRITICAL FIX)
            # This prevents state contamination when a new puzzle starts
            # Must happen AFTER puzzle_ids are updated but BEFORE using the Q-head
            if hasattr(self.inner.q_head, 'reset_hidden') and puzzle_changed.any():
                batch_size = batch["inputs"].shape[0]
                self.inner.q_head.reset_hidden(batch_size, mask=puzzle_changed)
            
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            # if training or (eval with learned halting enabled), and ACT is enabled
            use_halting = (self.training or self.config.use_learned_halting_eval) and (self.config.halt_max_steps > 1)
            if use_halting:

                # Halt signal
                # NOTE: In eval mode with use_learned_halting_eval=False, always use max steps for consistent batching
                
                if self.config.enable_dqn:
                    # DQN: Epsilon-greedy exploration
                    epsilon = compute_epsilon(
                        carry.training_step,
                        self.config.dqn_epsilon_start,
                        self.config.dqn_epsilon_end,
                        self.config.dqn_epsilon_decay_steps
                    )
                    halted = halted | select_action_epsilon_greedy(q_halt_logits, epsilon, self.training)
                else:
                    # Original: random exploration
                    if self.config.no_ACT_continue:
                        halted = halted | (q_halt_logits > 0)
                    else:
                        halted = halted | (q_halt_logits > q_continue_logits)

                    # Exploration
                    min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                    halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    # Compute target Q
                    # NOTE: No replay buffer and target networks for computing target Q-value.
                    # As batch_size is large, there're many parallel envs.
                    # Similar concept as PQN https://arxiv.org/abs/2407.04811
                    _, _, (next_q_halt_logits, next_q_continue_logits), _, _ = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))
            
            # Reset RNN Q-head state for sequences that halted (prevents contamination across episodes within same puzzle)
            if hasattr(self.inner.q_head, 'reset_hidden') and halted.any():
                batch_size = batch["inputs"].shape[0]
                self.inner.q_head.reset_hidden(batch_size, mask=halted)

        # Update DQN tracking fields if enabled
        new_prev_accuracy = carry.prev_accuracy
        # Increment training_step for epsilon decay (done here, not in losses)
        new_training_step = carry.training_step + 1 if self.config.enable_dqn else carry.training_step
        
        # Update puzzle_ids if tracked
        final_puzzle_ids = new_puzzle_ids if carry.puzzle_ids is not None and 'puzzle_identifiers' in new_current_data else None
        
        return TinyRecursiveReasoningModel_ACTV1Carry(
            new_inner_carry, 
            new_steps, 
            halted, 
            new_current_data,
            prev_accuracy=new_prev_accuracy,
            training_step=new_training_step,
            puzzle_ids=final_puzzle_ids
        ), outputs
