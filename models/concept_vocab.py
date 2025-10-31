"""
Semantic Concept Vocabulary (VQ-VAE style codebook)

Replaces BPE vocab_size (50k tokens) with small semantic concept vocabulary (~2-5k).
Each concept represents a phrase/idea, with optional expansion to children tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ConceptCodebook(nn.Module):
    """
    Vector-Quantized codebook of semantic concepts.
    
    Maps continuous capsule embeddings → discrete concept IDs.
    Supports expansion to BPE tokens when needed.
    """
    
    def __init__(
        self,
        num_concepts: int = 2048,  # Much smaller than 50k BPE vocab
        concept_dim: int = 768,
        commitment_cost: float = 0.25,
        use_ema: bool = True,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        
        self.num_concepts = num_concepts
        self.concept_dim = concept_dim
        self.commitment_cost = commitment_cost
        self.use_ema = use_ema
        
        # Codebook: learnable concept embeddings [num_concepts, concept_dim]
        self.embeddings = nn.Embedding(num_concepts, concept_dim)
        self.embeddings.weight.data.uniform_(-1 / num_concepts, 1 / num_concepts)
        
        # EMA statistics (for online codebook updates)
        if use_ema:
            self.register_buffer('ema_cluster_size', torch.zeros(num_concepts))
            self.register_buffer('ema_embeddings', self.embeddings.weight.data.clone())
            self.ema_decay = ema_decay
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize continuous embeddings to discrete concepts.
        
        Args:
            z: [B, seq_len, concept_dim] continuous capsule embeddings
        
        Returns:
            z_q: Quantized embeddings (same shape as z)
            concept_ids: [B, seq_len] discrete concept indices
            vq_loss: Vector quantization loss
        """
        # Flatten batch and sequence
        z_flat = z.reshape(-1, self.concept_dim)  # [B*seq_len, D]
        
        # Compute distances to all codebook vectors
        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flat, self.embeddings.weight.t())
        )  # [B*seq_len, num_concepts]
        
        # Nearest neighbor lookup
        concept_ids_flat = torch.argmin(distances, dim=1)  # [B*seq_len]
        
        # Get quantized embeddings
        z_q_flat = self.embeddings(concept_ids_flat)  # [B*seq_len, D]
        
        # Reshape back
        z_q = z_q_flat.view(z.shape)
        concept_ids = concept_ids_flat.view(z.shape[0], z.shape[1])
        
        # VQ loss: encourage encoder to produce embeddings close to codebook
        if self.training:
            # Commitment loss: encoder commits to codebook
            commitment_loss = F.mse_loss(z.detach(), z_q)
            
            # Codebook loss: codebook follows encoder
            if self.use_ema:
                # EMA update (no gradient)
                self._ema_update(z_flat, concept_ids_flat)
                codebook_loss = 0
            else:
                codebook_loss = F.mse_loss(z_q, z.detach())
            
            vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        else:
            vq_loss = torch.tensor(0.0, device=z.device)
        
        # Straight-through estimator: copy gradients from z_q to z
        z_q = z + (z_q - z).detach()
        
        return z_q, concept_ids, vq_loss
    
    def _ema_update(self, z_flat: torch.Tensor, concept_ids_flat: torch.Tensor):
        """EMA update of codebook (VQ-VAE2 style)."""
        # Count usage per concept
        encodings = F.one_hot(concept_ids_flat, self.num_concepts).float()  # [N, K]
        
        # EMA cluster size
        self.ema_cluster_size.data.mul_(self.ema_decay).add_(
            encodings.sum(0), alpha=1 - self.ema_decay
        )
        
        # Laplace smoothing
        n = self.ema_cluster_size.sum()
        self.ema_cluster_size.data.add_(1e-5).div_(n + self.num_concepts * 1e-5).mul_(n)
        
        # EMA embeddings
        dw = torch.matmul(encodings.t(), z_flat)  # [K, D]
        self.ema_embeddings.data.mul_(self.ema_decay).add_(dw, alpha=1 - self.ema_decay)
        
        # Update codebook
        self.embeddings.weight.data.copy_(
            self.ema_embeddings / self.ema_cluster_size.unsqueeze(1)
        )
    
    def lookup(self, concept_ids: torch.Tensor) -> torch.Tensor:
        """Look up concept embeddings by ID."""
        return self.embeddings(concept_ids)
    
    def get_codebook_usage(self) -> torch.Tensor:
        """Get usage statistics for each concept."""
        if self.use_ema:
            return self.ema_cluster_size / self.ema_cluster_size.sum()
        else:
            return torch.ones(self.num_concepts) / self.num_concepts


class HybridOutputHead(nn.Module):
    """
    Hybrid output head: concept vocabulary + control symbols.
    
    Vocabulary structure:
    - [0, num_concepts): Semantic concept IDs
    - num_concepts: <EXPAND> - expand concept to children
    - num_concepts+1: <STOP> - end generation
    - num_concepts+2: <MERGE> - merge with previous
    - num_concepts+3: <PAD> - padding
    """
    
    EXPAND_TOKEN = 0  # Relative to num_concepts
    STOP_TOKEN = 1
    MERGE_TOKEN = 2
    PAD_TOKEN = 3
    NUM_CONTROL = 4
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_concepts: int = 2048,
        concept_dim: int = 768,
        use_vq: bool = True,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_concepts = num_concepts
        self.total_vocab = num_concepts + self.NUM_CONTROL
        
        # Concept codebook (optional VQ)
        self.use_vq = use_vq
        if use_vq:
            self.codebook = ConceptCodebook(num_concepts, concept_dim)
        
        # Output projection: hidden → concept logits
        self.output_proj = nn.Linear(hidden_size, self.total_vocab)
        
        # Control symbol embeddings (learnable)
        self.control_embeddings = nn.Embedding(self.NUM_CONTROL, concept_dim)
    
    def forward(self, hidden: torch.Tensor, use_vq: bool = False) -> torch.Tensor:
        """
        Project hidden states to concept vocabulary.
        
        Args:
            hidden: [B, seq_len, hidden_size]
            use_vq: If True, quantize to discrete concepts
        
        Returns:
            logits: [B, seq_len, total_vocab] or concept_ids if use_vq
        """
        logits = self.output_proj(hidden)  # [B, seq_len, total_vocab]
        
        if use_vq and self.use_vq:
            # Hard quantization (for generation)
            concept_probs = F.softmax(logits[:, :, :self.num_concepts], dim=-1)
            concept_ids = torch.argmax(concept_probs, dim=-1)
            return concept_ids
        
        return logits
    
    def quantize_hidden(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize hidden states to concept IDs (VQ-VAE style)."""
        if not self.use_vq:
            raise RuntimeError("VQ not enabled")
        
        z_q, concept_ids, vq_loss = self.codebook(hidden)
        return concept_ids, vq_loss
    
    def get_control_id(self, control_type: str) -> int:
        """Get token ID for control symbol."""
        control_map = {
            'expand': self.num_concepts + self.EXPAND_TOKEN,
            'stop': self.num_concepts + self.STOP_TOKEN,
            'merge': self.num_concepts + self.MERGE_TOKEN,
            'pad': self.num_concepts + self.PAD_TOKEN,
        }
        return control_map.get(control_type.lower(), self.num_concepts + self.PAD_TOKEN)
    
    def is_control_token(self, token_id: int) -> bool:
        """Check if token is a control symbol."""
        return token_id >= self.num_concepts
    
    def decode_token(self, token_id: int) -> str:
        """Decode token ID to string (for debugging)."""
        if token_id < self.num_concepts:
            return f"<CONCEPT_{token_id}>"
        elif token_id == self.get_control_id('expand'):
            return "<EXPAND>"
        elif token_id == self.get_control_id('stop'):
            return "<STOP>"
        elif token_id == self.get_control_id('merge'):
            return "<MERGE>"
        else:
            return "<PAD>"
