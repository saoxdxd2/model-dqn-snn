"""
Associative Memory Bank for pattern storage and retrieval.
Enables recursive reasoning to reuse learned transformations.
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional


class AssociativeMemoryBank(nn.Module):
    """
    Key-value memory bank with attention-based retrieval.
    Stores high-reward reasoning patterns for reuse.
    """
    
    def __init__(self, capacity: int = 4096, hidden_size: int = 512, 
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.capacity = capacity
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Memory storage (learnable)
        self.memory_keys = nn.Parameter(torch.randn(capacity, hidden_size) / (hidden_size ** 0.5))
        self.memory_values = nn.Parameter(torch.randn(capacity, hidden_size) / (hidden_size ** 0.5))
        
        # Gating mechanism (learn when to use memory)
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # Multi-head attention for retrieval
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = dropout
        
        # Usage tracking (for LRU replacement)
        self.register_buffer('usage_count', torch.zeros(capacity))
        self.register_buffer('last_used', torch.zeros(capacity))
        self.step_counter = 0
        
    def read(self, query: torch.Tensor, use_gating: bool = True) -> torch.Tensor:
        """
        Retrieve relevant memories for current query.
        
        Args:
            query: [batch, hidden_size] - current reasoning state
            use_gating: whether to apply learned gating
        
        Returns:
            memory_output: [batch, hidden_size] - retrieved memory
        """
        batch_size = query.shape[0]
        
        # Expand keys/values for batch
        keys = self.memory_keys.unsqueeze(0).expand(batch_size, -1, -1)
        values = self.memory_values.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Attention-based retrieval (dropout respects training mode)
        query_expanded = query.unsqueeze(1)  # [batch, 1, hidden]
        attn_output, attn_weights = self.attention(
            query_expanded, keys, values,
            need_weights=True
        )  # attn_output: [batch, 1, hidden]
        # Note: nn.MultiheadAttention automatically respects self.training for dropout
        
        # Update usage statistics
        with torch.no_grad():
            # Average attention weights across batch and heads
            avg_attn = attn_weights.mean(0)  # [capacity]
            self.usage_count += avg_attn
            self.last_used = torch.where(
                avg_attn > 1e-3,
                torch.full_like(self.last_used, float(self.step_counter)),
                self.last_used
            )
            self.step_counter += 1
        
        memory_output = attn_output.squeeze(1)  # [batch, hidden]
        
        # Apply gating (learn when memory is useful)
        if use_gating:
            gate_value = self.gate(query)  # [batch, 1]
            memory_output = gate_value * memory_output
        
        return memory_output
    
    def write(self, state: torch.Tensor, reward: torch.Tensor, 
              threshold: float = 0.5) -> None:
        """
        Store high-reward states in memory (LRU replacement).
        
        Args:
            state: [batch, hidden_size] - states to potentially store
            reward: [batch] - associated rewards
            threshold: minimum reward to store
        """
        # Only store high-reward states
        high_reward_mask = reward > threshold
        if not high_reward_mask.any():
            return
        
        high_reward_states = state[high_reward_mask].detach()
        
        with torch.no_grad():
            for new_state in high_reward_states:
                # Find least recently used slot
                lru_idx = self.last_used.argmin()
                
                # Replace with new state
                self.memory_keys.data[lru_idx] = new_state
                self.memory_values.data[lru_idx] = new_state
                self.usage_count[lru_idx] = 0
                self.last_used[lru_idx] = float(self.step_counter)
    
    def get_memory_stats(self) -> dict:
        """Get statistics about memory usage."""
        return {
            'total_accesses': self.usage_count.sum().item(),
            'active_slots': (self.usage_count > 0).sum().item(),
            'utilization': (self.usage_count > 0).float().mean().item()
        }


class SparseMemoryBank(nn.Module):
    """
    Sparse memory bank using LSH (Locality Sensitive Hashing) for fast retrieval.
    Suitable for very large memory capacities (>10K slots).
    """
    
    def __init__(self, capacity: int = 16384, hidden_size: int = 512,
                 num_hashes: int = 8, hash_size: int = 16):
        super().__init__()
        self.capacity = capacity
        self.hidden_size = hidden_size
        self.num_hashes = num_hashes
        self.hash_size = hash_size
        
        # Memory storage
        self.memory_keys = nn.Parameter(torch.randn(capacity, hidden_size) / (hidden_size ** 0.5))
        self.memory_values = nn.Parameter(torch.randn(capacity, hidden_size) / (hidden_size ** 0.5))
        
        # LSH projection matrices
        self.hash_projections = nn.Parameter(
            torch.randn(num_hashes, hidden_size, hash_size) / (hidden_size ** 0.5),
            requires_grad=False  # Fixed random projections
        )
        
        # Simple output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
    def _compute_hash(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute LSH hash codes.
        
        Args:
            x: [batch, hidden_size]
        Returns:
            hash_codes: [batch, num_hashes]
        """
        # Project and binarize
        projections = torch.einsum('bh,nhd->bnd', x, self.hash_projections)
        hash_codes = (projections.sum(-1) > 0).long()
        return hash_codes
    
    def read(self, query: torch.Tensor, top_k: int = 32) -> torch.Tensor:
        """
        Fast approximate retrieval using LSH.
        
        Args:
            query: [batch, hidden_size]
            top_k: number of nearest neighbors to retrieve
        
        Returns:
            memory_output: [batch, hidden_size]
        """
        batch_size = query.shape[0]
        
        # Compute hash for query
        query_hash = self._compute_hash(query)  # [batch, num_hashes]
        
        # Compute hash for all keys (cached)
        if not hasattr(self, '_key_hashes'):
            with torch.no_grad():
                self._key_hashes = self._compute_hash(self.memory_keys)
        
        # Find candidates (keys with matching hash buckets)
        # Hamming distance between hashes
        hash_dist = (query_hash.unsqueeze(1) != self._key_hashes.unsqueeze(0)).sum(-1)
        
        # Select top-k closest by hash
        _, top_indices = torch.topk(hash_dist, k=min(top_k, self.capacity), 
                                     largest=False, dim=1)
        
        # Gather top-k keys and values
        # top_indices: [batch, top_k]
        batch_indices = torch.arange(batch_size, device=query.device).unsqueeze(1)
        selected_keys = self.memory_keys[top_indices]  # [batch, top_k, hidden]
        selected_values = self.memory_values[top_indices]
        
        # Compute attention over selected memories
        scores = torch.einsum('bh,bkh->bk', query, selected_keys) / (self.hidden_size ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)  # [batch, top_k]
        
        # Weighted sum
        memory_output = torch.einsum('bk,bkh->bh', attn_weights, selected_values)
        
        return self.output_proj(memory_output)
    
    def write(self, state: torch.Tensor, reward: torch.Tensor,
              threshold: float = 0.5) -> None:
        """Store high-reward states (random replacement for simplicity)."""
        high_reward_mask = reward > threshold
        if not high_reward_mask.any():
            return
        
        high_reward_states = state[high_reward_mask].detach()
        
        with torch.no_grad():
            for new_state in high_reward_states:
                # Random replacement
                idx = torch.randint(0, self.capacity, (1,)).item()
                self.memory_keys.data[idx] = new_state
                self.memory_values.data[idx] = new_state
            
            # Invalidate hash cache
            if hasattr(self, '_key_hashes'):
                delattr(self, '_key_hashes')
