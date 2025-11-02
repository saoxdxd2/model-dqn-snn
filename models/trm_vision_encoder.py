"""
TRM Vision Encoder: Use recursive reasoning for vision encoding.

Reuses existing TinyRecursiveReasoningModel_ACTV1 blocks for unified architecture.
Same TRM used for encoding AND reasoning!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1Block,
    TinyRecursiveReasoningModel_ACTV1ReasoningModule,
    TinyRecursiveReasoningModel_ACTV1Config,
)
from models.layers import RotaryEmbedding


class TRMVisionEncoder(nn.Module):
    """
    Vision encoder using TRM recursive reasoning architecture.
    
    Pipeline:
    1. Patch embedding: Image → tokens
    2. TRM blocks: Recursive spatial reasoning (H_cycles × L_cycles)
    3. Spatial pooling: Tokens → capsules (3×4 grid)
    
    Benefits:
    - Unified architecture (same TRM for encoding + reasoning)
    - Trainable on ARC spatial patterns
    - Small size: ~3-5M params (vs 150M CLIP)
    - Native 512D output (no projection loss)
    """
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        hidden_size: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,  # L_layers for encoder
        H_cycles: int = 2,  # High-level cycles
        L_cycles: int = 3,  # Low-level cycles per H-cycle
        target_capsules: int = 12,
        capsule_grid_shape: tuple = (3, 4),
        expansion: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_patches = (image_size // patch_size) ** 2
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.target_capsules = target_capsules
        self.capsule_grid_shape = capsule_grid_shape
        
        # Patch embedding: [B, 3, H, W] -> [B, num_patches, hidden_size]
        self.patch_embed = nn.Conv2d(
            3, hidden_size,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Positional embeddings (RoPE)
        self.rotary_emb = RotaryEmbedding(hidden_size // num_heads)
        
        # Create TRM config for encoder
        encoder_config = TinyRecursiveReasoningModel_ACTV1Config(
            batch_size=1,  # Dynamic
            seq_len=self.num_patches,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_key_value_heads=num_heads // 4,  # GQA
            expansion=expansion,
            H_layers=0,  # Not used (manual cycles)
            L_layers=num_layers,
            vocab_size=2052,  # Dummy
            num_puzzle_identifiers=0,
            puzzle_emb_ndim=0,
            causal=False,  # Non-causal for vision
            rms_norm_eps=1e-6,
            mlp_t=False,
        )
        
        # TRM L-level blocks (reuse existing architecture!)
        self.L_blocks = nn.ModuleList([
            TinyRecursiveReasoningModel_ACTV1Block(encoder_config)
            for _ in range(num_layers)
        ])
        
        # Spatial pooling to capsule grid
        self.spatial_pool = nn.AdaptiveAvgPool2d(capsule_grid_shape)
        
        print(f"\n🔧 TRM Vision Encoder Initialized:")
        print(f"   Patches: {self.num_patches} ({image_size//patch_size}×{image_size//patch_size})")
        print(f"   Hidden: {hidden_size}")
        print(f"   Layers: {num_layers}")
        print(f"   Cycles: H={H_cycles} × L={L_cycles}")
        print(f"   Output: {target_capsules} capsules (512D)")
        print(f"   Params: ~{self.count_parameters()/1e6:.1f}M")
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, images):
        """
        Encode images to capsules using TRM recursive reasoning.
        
        Args:
            images: [B, 3, 224, 224] or [B, 3, H, W]
        
        Returns:
            capsules: [B, 12, 512]
        """
        B = images.shape[0]
        device = images.device
        
        # Patch embedding: [B, 3, 224, 224] -> [B, 512, 14, 14]
        patch_embeddings = self.patch_embed(images)
        
        # Flatten to sequence: [B, 512, 14, 14] -> [B, 196, 512]
        H_patches = W_patches = self.image_size // self.patch_size
        tokens = patch_embeddings.flatten(2).transpose(1, 2)
        
        # Compute RoPE embeddings
        cos_sin = self.rotary_emb(tokens)
        
        # TRM Recursive Reasoning (H_cycles × L_cycles)
        for h_cycle in range(self.H_cycles):
            for l_cycle in range(self.L_cycles):
                # Apply L-level blocks
                for block in self.L_blocks:
                    tokens = block(
                        cos_sin=cos_sin,
                        hidden_states=tokens,
                        spatial_bias=None
                    )
        
        # Reshape to spatial grid: [B, 196, 512] -> [B, 512, 14, 14]
        tokens_spatial = tokens.transpose(1, 2).view(
            B, self.hidden_size, H_patches, W_patches
        )
        
        # Spatial pooling to capsule grid: [B, 512, 14, 14] -> [B, 512, 3, 4]
        capsules_spatial = self.spatial_pool(tokens_spatial)
        
        # Flatten to capsules: [B, 512, 3, 4] -> [B, 12, 512]
        capsules = capsules_spatial.flatten(2).transpose(1, 2)
        
        return capsules


class TRMVisionEncoderWithChecksums(nn.Module):
    """
    TRM Vision Encoder with HESC checksums and children support.
    
    Wrapper around TRMVisionEncoder to match CapsuleEncoder interface.
    """
    
    def __init__(
        self,
        hidden_size: int = 512,
        target_capsules: int = 12,
        children_per_capsule: int = 4,
        checksum_dim: int = 32,
        **trm_kwargs
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.target_capsules = target_capsules
        self.children_per_capsule = children_per_capsule
        self.checksum_dim = checksum_dim
        
        # TRM encoder
        self.encoder = TRMVisionEncoder(
            hidden_size=hidden_size,
            target_capsules=target_capsules,
            **trm_kwargs
        )
        
        # Checksum head (integrity signature)
        self.checksum_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, checksum_dim)
        )
        
        # Children projection (for hierarchical expansion)
        self.children_projection = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, images, return_children: bool = True):
        """
        Encode images to HESC capsules.
        
        Args:
            images: [B, 3, 224, 224]
            return_children: Whether to compute children embeddings
        
        Returns:
            dict with 'sketches', 'checksums', 'children'
        """
        # Encode to capsules
        sketches = self.encoder(images)  # [B, 12, 512]
        
        # Compute checksums
        checksums = self.checksum_head(sketches)  # [B, 12, 32]
        
        result = {
            'sketches': sketches,
            'checksums': checksums
        }
        
        # Children: duplicate sketches for now
        # TODO: Use spatial patches as real children
        if return_children:
            children = sketches.unsqueeze(2).repeat(
                1, 1, self.children_per_capsule, 1
            )  # [B, 12, 4, 512]
            children = self.children_projection(children)
            result['children'] = children
        
        return result


if __name__ == "__main__":
    # Test encoder
    print("Testing TRM Vision Encoder...")
    
    encoder = TRMVisionEncoderWithChecksums(
        hidden_size=512,
        target_capsules=12,
        num_layers=2,
        H_cycles=2,
        L_cycles=3
    )
    
    # Dummy image batch
    images = torch.randn(4, 3, 224, 224)
    
    # Encode
    output = encoder(images, return_children=True)
    
    print(f"\nOutput shapes:")
    print(f"  Sketches: {output['sketches'].shape}")  # [4, 12, 512]
    print(f"  Checksums: {output['checksums'].shape}")  # [4, 12, 32]
    print(f"  Children: {output['children'].shape}")  # [4, 12, 4, 512]
    
    print("\n✅ TRM Vision Encoder working!")
