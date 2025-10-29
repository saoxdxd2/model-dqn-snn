"""
CNN Tokenizer for Vision Models (CCT-style)

Replaces K-Means patch embedding with convolutional stem.
Provides better inductive bias and hierarchical features.

Reference: Compact Convolutional Transformers (Hassani et al., 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNTokenizer(nn.Module):
    """
    Convolutional tokenizer that converts images to token sequences.
    
    Uses progressive downsampling with convolutions to:
    1. Preserve spatial locality
    2. Learn hierarchical features (edges -> textures -> patterns)
    3. Provide translation equivariance
    4. Eliminate quantization error from K-Means
    
    Architecture (for CIFAR-10: 32x32x3):
        Conv1: 32x32x3  -> 32x32x64  (3x3 conv, ReLU, BN)
        Pool1: 32x32x64 -> 16x16x64  (3x3 maxpool, stride=2)
        Conv2: 16x16x64 -> 16x16x128 (3x3 conv, ReLU, BN)
        Pool2: 16x16x128 -> 8x8x128  (3x3 maxpool, stride=2)
        Conv3: 8x8x128  -> 8x8x256   (3x3 conv, ReLU, BN)
        Output: [B, 64, 256] (64 tokens of 256-dim)
    
    Parameters saved vs K-Means embedding:
        K-Means: 2048 x 256 = 524,288 params
        CNN Stem: ~371,264 params
        Savings: 153,024 params (-29%)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        hidden_size: int = 256,
        num_conv_layers: int = 3,
        conv_channels: list = None,
        kernel_size: int = 3,
        pooling_kernel: int = 3,
        pooling_stride: int = 2,
        use_batch_norm: bool = True,
        activation: str = 'relu',
    ):
        super().__init__()
        
        if conv_channels is None:
            conv_channels = [64, 128, hidden_size]
        
        assert len(conv_channels) == num_conv_layers, \
            f"conv_channels length ({len(conv_channels)}) must match num_conv_layers ({num_conv_layers})"
        assert conv_channels[-1] == hidden_size, \
            f"Final conv channel ({conv_channels[-1]}) must match hidden_size ({hidden_size})"
        
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_conv_layers = num_conv_layers
        self.conv_channels = conv_channels
        
        # Build convolutional stem
        self.layers = nn.ModuleList()
        
        in_ch = in_channels
        for i, out_ch in enumerate(conv_channels):
            # Convolutional layer
            conv = nn.Conv2d(
                in_ch, out_ch,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,  # Same padding
                bias=not use_batch_norm  # No bias if using BN
            )
            
            # Batch normalization
            bn = nn.BatchNorm2d(out_ch) if use_batch_norm else nn.Identity()
            
            # Activation
            if activation == 'relu':
                act = nn.ReLU(inplace=True)
            elif activation == 'gelu':
                act = nn.GELU()
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            # Pooling (only for first two layers)
            if i < num_conv_layers - 1:
                pool = nn.MaxPool2d(
                    kernel_size=pooling_kernel,
                    stride=pooling_stride,
                    padding=pooling_kernel // 2
                )
            else:
                pool = nn.Identity()
            
            # Add as sequential block
            self.layers.append(nn.Sequential(conv, bn, act, pool))
            
            in_ch = out_ch
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize convolutional weights with He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Convert images to token sequences.
        
        Args:
            images: [B, C, H, W] input images (e.g., [B, 3, 32, 32])
        
        Returns:
            tokens: [B, seq_len, hidden_size] token sequence
                    (e.g., [B, 64, 256] for CIFAR-10)
        """
        x = images
        
        # Apply convolutional layers
        for layer in self.layers:
            x = layer(x)
        
        # x: [B, hidden_size, H', W'] (e.g., [B, 256, 8, 8])
        B, C, H, W = x.shape
        
        # Flatten spatial dimensions: [B, C, H, W] -> [B, H*W, C]
        x = x.reshape(B, C, H * W).transpose(1, 2)
        
        # x: [B, seq_len, hidden_size] (e.g., [B, 64, 256])
        return x
    
    def get_sequence_length(self, input_size: tuple) -> int:
        """
        Calculate output sequence length given input image size.
        
        Args:
            input_size: (H, W) input image size
        
        Returns:
            seq_len: Number of output tokens
        """
        H, W = input_size
        
        # Apply pooling downsampling
        for i in range(self.num_conv_layers - 1):
            H = (H + 1) // 2  # MaxPool with stride=2
            W = (W + 1) // 2
        
        return H * W


class AdaptiveCNNTokenizer(nn.Module):
    """
    Adaptive CNN tokenizer that handles variable input sizes.
    Uses adaptive pooling to ensure fixed output sequence length.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        hidden_size: int = 256,
        target_seq_len: int = 64,  # Fixed output sequence length
        conv_channels: list = None,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        
        if conv_channels is None:
            conv_channels = [64, 128, hidden_size]
        
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.target_seq_len = target_seq_len
        
        # Calculate target spatial size: seq_len = H * W
        # Use square grid for simplicity
        self.target_h = int(target_seq_len ** 0.5)
        self.target_w = target_seq_len // self.target_h
        
        # Build convolutional layers
        self.convs = nn.ModuleList()
        in_ch = in_channels
        for out_ch in conv_channels:
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=not use_batch_norm),
                nn.BatchNorm2d(out_ch) if use_batch_norm else nn.Identity(),
                nn.ReLU(inplace=True)
            ))
            in_ch = out_ch
        
        # Adaptive pooling to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.target_h, self.target_w))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, C, H, W] any size
        Returns:
            tokens: [B, target_seq_len, hidden_size]
        """
        x = images
        
        # Apply convolutions
        for conv in self.convs:
            x = conv(x)
        
        # Adaptive pooling to fixed size
        x = self.adaptive_pool(x)  # [B, hidden_size, target_h, target_w]
        
        # Flatten
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).transpose(1, 2)
        
        return x
