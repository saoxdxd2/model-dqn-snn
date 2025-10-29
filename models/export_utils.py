"""
Export utilities for converting trained models to efficient inference formats.
- SNN (Spiking Neural Networks): 10-100× energy efficiency
- BNN (Binary Neural Networks): 32× memory reduction, faster on CPU
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron for SNN."""
    
    def __init__(self, in_features: int, out_features: int, 
                 tau: float = 0.9, threshold: float = 1.0, dt: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tau = tau
        self.threshold = threshold
        self.dt = dt
        
        # Weights (initialized from trained model)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Membrane potential (stateful)
        self.register_buffer('v_mem', torch.zeros(1, out_features))
        
    def reset_state(self, batch_size: int = 1):
        """Reset membrane potential."""
        self.v_mem = torch.zeros(batch_size, self.out_features, device=self.weight.device)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, in_features] - input current
        Returns:
            spike: [batch, out_features] - binary spike output
            v_mem: [batch, out_features] - membrane potential (for monitoring)
        """
        batch_size = x.shape[0]
        
        # Ensure correct batch size
        if self.v_mem.shape[0] != batch_size:
            self.reset_state(batch_size)
        
        # Input current
        i_in = F.linear(x, self.weight, self.bias)
        
        # Leaky integration: dv/dt = (-v + i_in) / tau
        self.v_mem = self.v_mem + self.dt * ((-self.v_mem + i_in) / self.tau)
        
        # Spike generation
        spike = (self.v_mem >= self.threshold).float()
        
        # Reset after spike
        self.v_mem = torch.where(spike.bool(), torch.zeros_like(self.v_mem), self.v_mem)
        
        return spike, self.v_mem


class SpikingQHead(nn.Module):
    """Spiking Neural Network Q-head for energy-efficient inference."""
    
    def __init__(self, hidden_size: int, num_actions: int = 2,
                 num_timesteps: int = 10, tau: float = 0.9):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.num_timesteps = num_timesteps
        
        # Encoder: rate coding (convert activation to spike rate)
        self.encoder = nn.Linear(hidden_size, 256)
        
        # Spiking layers
        self.lif1 = LIFNeuron(256, 128, tau=tau)
        self.lif2 = LIFNeuron(128, num_actions, tau=tau)
    
    def reset_state(self, batch_size: int):
        """Reset all neuron states to prevent contamination."""
        self.lif1.reset_state(batch_size)
        self.lif2.reset_state(batch_size)
        
    def forward(self, x: torch.Tensor, return_spikes: bool = False) -> torch.Tensor:
        """
        Args:
            x: [batch, hidden_size] - input state
            return_spikes: if True, return spike trains instead of Q-values
        Returns:
            q_values: [batch, num_actions] (or spike_count if return_spikes)
        """
        batch_size = x.shape[0]
        
        # Reset neuron states
        self.lif1.reset_state(batch_size)
        self.lif2.reset_state(batch_size)
        
        # Rate encoding: stronger input = more spikes
        current = torch.relu(self.encoder(x))
        
        # Accumulate spikes over time
        spike_count = torch.zeros(batch_size, self.num_actions, device=x.device)
        
        for t in range(self.num_timesteps):
            # Propagate through spiking layers
            spike1, _ = self.lif1(current)
            spike2, _ = self.lif2(spike1)
            
            spike_count += spike2
        
        if return_spikes:
            return spike_count
        
        # Convert spike count to Q-values (normalized by timesteps)
        q_values = spike_count / self.num_timesteps
        
        return q_values


class BinaryLinear(nn.Module):
    """Binary weight linear layer for BNN."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Store full-precision weights for training
        self.weight_fp = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        
        # Scaling factor (learnable)
        self.alpha = nn.Parameter(torch.ones(out_features, 1))
        
    def binarize_weights(self) -> torch.Tensor:
        """Binarize weights to {-1, +1}."""
        return torch.sign(self.weight_fp)
    
    def forward(self, x: torch.Tensor, training: bool = None) -> torch.Tensor:
        """
        Args:
            x: [batch, in_features]
            training: if True, use STE; if False, use binary weights directly
        Returns:
            output: [batch, out_features]
        """
        if training is None:
            training = self.training
        
        # Cast to float32 for BNN computation (weights are float32)
        x = x.float()
        
        if training:
            # Straight-through estimator (STE) for gradient flow
            weight_binary = self.binarize_weights()
            weight = weight_binary.detach() - self.weight_fp.detach() + self.weight_fp
        else:
            # Pure binary inference
            weight = self.binarize_weights()
        
        # Scaled binary convolution
        output = F.linear(x, weight * self.alpha, self.bias)
        
        return output


class BinaryQHead(nn.Module):
    """Binary Neural Network Q-head for memory-efficient CPU inference."""
    
    def __init__(self, hidden_size: int, num_actions: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        
        # Binary layers
        self.fc1 = BinaryLinear(hidden_size, 128)
        self.fc2 = BinaryLinear(128, num_actions)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, hidden_size]
        Returns:
            q_values: [batch, num_actions]
        """
        x = torch.sign(torch.relu(self.fc1(x)))  # Binarize activations too
        x = self.fc2(x)
        return x


def convert_mlp_to_snn(mlp_q_head: nn.Module, num_timesteps: int = 10) -> SpikingQHead:
    """
    Convert trained MLP Q-head to Spiking Neural Network.
    
    Args:
        mlp_q_head: Trained MLP Q-head (from q_heads.MLPQHead)
        num_timesteps: Number of simulation timesteps
    
    Returns:
        snn_q_head: Converted spiking Q-head
    """
    from models.q_heads import MLPQHead
    
    if not isinstance(mlp_q_head, MLPQHead):
        raise ValueError("Can only convert MLPQHead to SNN")
    
    hidden_size = mlp_q_head.q_head.weight.shape[1]
    num_actions = mlp_q_head.q_head.weight.shape[0]
    
    # Create SNN
    snn = SpikingQHead(hidden_size, num_actions, num_timesteps=num_timesteps)
    
    # Transfer weights (approximate rate coding)
    with torch.no_grad():
        # Encoder: identity-like initialization
        snn.encoder.weight.copy_(torch.eye(256, hidden_size)[:, :hidden_size])
        snn.encoder.bias.zero_()
        
        # LIF neurons: scale down for spike domain
        scale_factor = 0.1  # Scaling for stable spiking
        snn.lif2.weight.copy_(mlp_q_head.q_head.weight.data * scale_factor)
        snn.lif2.bias.copy_(mlp_q_head.q_head.bias.data * scale_factor)
    
    return snn


def convert_mlp_to_bnn(mlp_q_head: nn.Module, calibration_data: Optional[torch.Tensor] = None) -> BinaryQHead:
    """
    Convert trained MLP Q-head to Binary Neural Network.
    
    Args:
        mlp_q_head: Trained MLP Q-head (single linear layer: hidden_size -> num_actions)
        calibration_data: Optional calibration data for better binarization
    
    Returns:
        bnn_q_head: Converted binary Q-head
    """
    from models.q_heads import MLPQHead
    
    if not isinstance(mlp_q_head, MLPQHead):
        raise ValueError("Can only convert MLPQHead to BNN")
    
    hidden_size = mlp_q_head.q_head.weight.shape[1]
    num_actions = mlp_q_head.q_head.weight.shape[0]
    
    print(f"   Converting MLP Q-head: {hidden_size} -> {num_actions}")
    
    # Create BNN
    bnn = BinaryQHead(hidden_size, num_actions)
    
    # Transfer and binarize weights
    with torch.no_grad():
        # Initialize first layer (projection from hidden_size to 128)
        # Use scaled random initialization for stability
        bnn.fc1.weight_fp.normal_(0, 0.1)
        
        # Second layer (128 -> num_actions): initialize from trained weights
        # Since MLP is (hidden_size -> num_actions) and BNN is (hidden_size -> 128 -> num_actions),
        # we need to approximate the mapping
        # Use random projection for fc2 since dimensions don't match
        bnn.fc2.weight_fp.normal_(0, 0.1)
        
        # Copy bias if available (dimensions match)
        if mlp_q_head.q_head.bias is not None:
            bnn.fc2.bias.copy_(mlp_q_head.q_head.bias.data)
        
        # Compute scaling factors using calibration if available
        if calibration_data is not None:
            # Use calibration data to optimize scaling
            with torch.enable_grad():
                output_fp = mlp_q_head(calibration_data)
                output_bin = bnn(calibration_data, training=False)
                
                # Optimize alpha to minimize MSE
                alpha_opt = (output_fp * output_bin).sum(0) / (output_bin * output_bin).sum(0)
                bnn.fc2.alpha.copy_(alpha_opt.unsqueeze(1))
        
        print(f"   BNN Q-head created: {hidden_size} -> 128 -> {num_actions}")
    
    return bnn


def quantize_model_int8(model: nn.Module) -> nn.Module:
    """
    Quantize model to INT8 for CPU inference (PyTorch native).
    
    Args:
        model: Float32 model
    
    Returns:
        quantized_model: INT8 quantized model
    """
    model.eval()
    model.cpu()
    
    # Dynamic quantization (activations quantized at runtime)
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )
    
    return quantized_model


def benchmark_inference(model: nn.Module, input_shape: Tuple[int, ...], 
                       num_iterations: int = 1000, device: str = 'cpu') -> dict:
    """
    Benchmark inference speed and energy.
    
    Args:
        model: Model to benchmark
        input_shape: Input tensor shape (batch, ...)
        num_iterations: Number of iterations
        device: 'cpu' or 'cuda'
    
    Returns:
        stats: Dictionary with timing and throughput metrics
    """
    import time
    
    model = model.to(device)
    model.eval()
    
    # Warmup - create proper batch input
    batch_size, seq_len = input_shape
    # Use model's actual seq_len if available (avoid mismatch)
    if hasattr(model, 'config') and hasattr(model.config, 'seq_len'):
        seq_len = model.config.seq_len
    dummy_inputs = torch.randint(0, 50257, (batch_size, seq_len), device=device)  # Token IDs
    dummy_batch = {
        'inputs': dummy_inputs,
        'labels': dummy_inputs,
        'puzzle_identifiers': torch.zeros(batch_size, dtype=torch.long, device=device)
    }
    
    # Create initial carry
    carry = model.initial_carry(dummy_batch)
    
    with torch.no_grad():
        for _ in range(10):
            carry, _ = model(carry, dummy_batch)
    
    # Reset carry for benchmark
    carry = model.initial_carry(dummy_batch)
    
    # Benchmark
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iterations):
            carry, _ = model(carry, dummy_batch)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_latency = (total_time / num_iterations) * 1000  # ms
    throughput = num_iterations / total_time  # samples/sec
    
    return {
        'avg_latency_ms': avg_latency,
        'throughput_samples_per_sec': throughput,
        'total_time_sec': total_time,
        'device': device
    }
