"""CPU Export Script - Uses existing export_utils.py infrastructure

Exports trained TRM model for i5-1035G1 CPU inference:
- Target: <4GB model size, 10 tokens/sec
- Uses existing INT8/BNN/SNN quantization from export_utils.py
- Integrates with MTP-enhanced model
- ONNX/TensorRT export for GPU deployment
"""

import torch
import sys
import os
from pathlib import Path
import onnx
import onnxruntime as ort

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.export_utils import quantize_model_int8, convert_mlp_to_bnn, benchmark_inference

# Import ModelLoader from same directory
try:
    from load_model import ModelLoader
except ImportError:
    # If running as module
    from scripts.load_model import ModelLoader


def export_cpu_optimized(
    model_path: str,
    output_path: str = 'model_cpu_optimized.pt',
    quantization: str = 'int8'  # 'int8', 'bnn', or 'dynamic'
):
    """
    Export model for CPU using existing infrastructure.
    
    Args:
        model_path: Path to trained model checkpoint
        output_path: Where to save optimized model
        quantization: Type of quantization to apply
    """
    print(f"Loading model from {model_path}...")
    
    # Load model using ModelLoader
    loader = ModelLoader(model_path, device='cpu')
    model = loader.load_model(quantized=False)
    
    print(f"\nApplying {quantization} quantization...")
    
    if quantization == 'int8':
        # Use existing INT8 quantization
        optimized_model = quantize_model_int8(model)
        compression_ratio = 4.0
        
    elif quantization == 'bnn':
        # Convert Q-head to BNN (32× memory reduction)
        q_head_bnn = convert_mlp_to_bnn(model.inner.q_head)
        model.inner.q_head = q_head_bnn
        # Apply INT8 to rest of model
        optimized_model = quantize_model_int8(model)
        compression_ratio = 6.0  # Combined INT8 + BNN
        
    elif quantization == 'dynamic':
        # PyTorch dynamic quantization
        optimized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU},
            dtype=torch.qint8
        )
        compression_ratio = 4.0
    
    else:
        raise ValueError(f"Unknown quantization: {quantization}")
    
    # Calculate model size
    total_params = sum(p.numel() for p in optimized_model.parameters())
    model_size_mb = (total_params * 4 / compression_ratio) / (1024 ** 2)
    model_size_gb = model_size_mb / 1024
    
    print(f"\nModel Statistics:")
    print(f"  Parameters: {total_params:,}")
    print(f"  Size: {model_size_gb:.2f} GB")
    print(f"  Compression: {compression_ratio}×")
    print(f"  Target met: {'✅' if model_size_gb < 4.0 else '❌'} (<4GB)")
    
    # Benchmark on CPU
    print(f"\nBenchmarking on CPU...")
    dummy_input = torch.randn(1, 512)  # Adjust to your model's input
    stats = benchmark_inference(optimized_model, (1, 512), device='cpu', num_iterations=100)
    
    throughput_tps = 32 / (stats['avg_latency_ms'] / 1000)  # Assuming 32 token sequence
    print(f"  Latency: {stats['avg_latency_ms']:.1f} ms")
    print(f"  Throughput: {throughput_tps:.1f} tokens/sec")
    print(f"  Target met: {'✅' if throughput_tps >= 10 else '❌'} (≥10 t/s)")
    
    # Save optimized model with config
    checkpoint = {
        'model_state_dict': optimized_model.state_dict(),
        'config': loader.config  # Include config for loading
    }
    torch.save(checkpoint, output_path)
    print(f"\nSaved to: {output_path}")
    
    # Save deployment info
    deployment_info = {
        'model_size_gb': model_size_gb,
        'compression_ratio': compression_ratio,
        'quantization': quantization,
        'throughput_tokens_per_sec': throughput_tps,
        'latency_ms': stats['avg_latency_ms'],
        'cpu_settings': {
            'threads': 4,  # i5-1035G1 has 4 cores
            'mkldnn': True,
            'batch_size': 1,  # Optimal for latency
        }
    }
    
    import json
    with open(output_path.replace('.pt', '_info.json'), 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    return optimized_model, deployment_info


def export_onnx_tensorrt(
    model_path: str,
    output_onnx: str = 'model.onnx',
    output_trt: str = 'model.trt',
    precision: str = 'fp16',  # 'fp16', 'int8', 'fp32'
    max_batch_size: int = 1,
    max_seq_len: int = 1024,
):
    """
    Export TRM model to ONNX and optionally TensorRT for GPU inference.
    
    Args:
        model_path: Path to trained model checkpoint
        output_onnx: Output ONNX file path
        output_trt: Output TensorRT engine path (optional)
        precision: Inference precision (fp16 recommended for T4)
        max_batch_size: Maximum batch size for TensorRT optimization
        max_seq_len: Maximum sequence length
    
    Returns:
        output_onnx: Path to exported ONNX model
    """
    print(f"\n{'='*70}")
    print(f"  📦 ONNX Export Pipeline")
    print(f"{'='*70}")
    print(f"  Model: {model_path}")
    print(f"  Output: {output_onnx}")
    print(f"  Precision: {precision}")
    print(f"{'='*70}\n")
    
    # Load model
    print("Loading model...")
    loader = ModelLoader(model_path, device='cuda')
    model = loader.load_model(quantized=False)
    model.eval()
    
    # Prepare dummy inputs
    print("Preparing dummy inputs...")
    batch_size = 1
    seq_len = max_seq_len
    hidden_size = loader.config['hidden_size']
    
    # Create dummy batch
    dummy_batch = {
        'input_ids': torch.zeros(batch_size, seq_len, dtype=torch.long).cuda(),
        'labels': torch.zeros(batch_size, seq_len, dtype=torch.long).cuda(),
    }
    
    # Initialize carry
    with torch.device('cuda'):
        dummy_carry = model.initial_carry(dummy_batch)
    
    # Export to ONNX
    print(f"\nExporting to ONNX...")
    try:
        torch.onnx.export(
            model,
            (dummy_carry, dummy_batch),
            output_onnx,
            input_names=['carry_z_H', 'carry_z_L', 'input_ids', 'labels'],
            output_names=['logits', 'new_carry_z_H', 'new_carry_z_L'],
            dynamic_axes={
                'input_ids': {0: 'batch', 1: 'seq_len'},
                'labels': {0: 'batch', 1: 'seq_len'},
                'logits': {0: 'batch', 1: 'seq_len'},
            },
            opset_version=17,  # Use latest stable opset
            do_constant_folding=True,
            export_params=True,
            verbose=False,
        )
        
        print(f"✅ ONNX export successful: {output_onnx}")
        
        # Verify ONNX model
        print("\nVerifying ONNX model...")
        onnx_model = onnx.load(output_onnx)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model verification passed")
        
        # Get model info
        model_size_mb = os.path.getsize(output_onnx) / (1024 ** 2)
        print(f"\nONNX Model Info:")
        print(f"  Size: {model_size_mb:.2f} MB")
        print(f"  Opset: {onnx_model.opset_import[0].version}")
        
        # Test ONNX Runtime inference
        print("\nTesting ONNX Runtime inference...")
        ort_session = ort.InferenceSession(
            output_onnx,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        # Prepare inputs for ONNX Runtime
        ort_inputs = {
            'input_ids': dummy_batch['input_ids'].cpu().numpy(),
            'labels': dummy_batch['labels'].cpu().numpy(),
            'carry_z_H': dummy_carry.inner_carry.z_H.cpu().numpy(),
            'carry_z_L': dummy_carry.inner_carry.z_L.cpu().numpy(),
        }
        
        ort_outputs = ort_session.run(None, ort_inputs)
        print(f"✅ ONNX Runtime inference successful")
        print(f"   Output shape: {ort_outputs[0].shape}")
        
        # Benchmark ONNX inference
        print("\nBenchmarking ONNX Runtime...")
        import time
        num_runs = 100
        start_time = time.time()
        for _ in range(num_runs):
            ort_session.run(None, ort_inputs)
        elapsed = time.time() - start_time
        
        avg_latency_ms = (elapsed / num_runs) * 1000
        throughput = num_runs / elapsed
        
        print(f"  Latency: {avg_latency_ms:.1f} ms/sample")
        print(f"  Throughput: {throughput:.1f} samples/sec")
        
        # TensorRT conversion (if requested)
        if output_trt:
            print(f"\n{'='*70}")
            print(f"  ⚡ TensorRT Conversion")
            print(f"{'='*70}\n")
            
            try:
                # Check if trtexec is available
                import subprocess
                result = subprocess.run(['trtexec', '--help'], 
                                      capture_output=True, timeout=5)
                
                if result.returncode == 0:
                    # Build TensorRT engine
                    precision_flag = {
                        'fp32': '--fp32',
                        'fp16': '--fp16',
                        'int8': '--int8'
                    }[precision]
                    
                    cmd = [
                        'trtexec',
                        f'--onnx={output_onnx}',
                        precision_flag,
                        f'--saveEngine={output_trt}',
                        f'--minShapes=input_ids:1x1,labels:1x1',
                        f'--optShapes=input_ids:{max_batch_size}x{max_seq_len//2},labels:{max_batch_size}x{max_seq_len//2}',
                        f'--maxShapes=input_ids:{max_batch_size}x{max_seq_len},labels:{max_batch_size}x{max_seq_len}',
                        '--verbose'
                    ]
                    
                    print(f"Running: {' '.join(cmd)}\n")
                    subprocess.run(cmd, check=True)
                    
                    print(f"\n✅ TensorRT engine created: {output_trt}")
                    
                    trt_size_mb = os.path.getsize(output_trt) / (1024 ** 2)
                    print(f"   Size: {trt_size_mb:.2f} MB")
                    print(f"   Precision: {precision}")
                    
                else:
                    print("⚠️  trtexec not found. Install TensorRT to enable engine conversion.")
                    print("   ONNX model is still usable with ONNX Runtime.")
                    
            except FileNotFoundError:
                print("⚠️  trtexec not found in PATH. Skipping TensorRT conversion.")
                print("   Install TensorRT: https://developer.nvidia.com/tensorrt")
            except subprocess.TimeoutExpired:
                print("⚠️  trtexec check timed out")
            except Exception as e:
                print(f"⚠️  TensorRT conversion failed: {e}")
                print("   ONNX model is still usable with ONNX Runtime.")
        
        # Save deployment info
        deployment_info = {
            'onnx_path': output_onnx,
            'onnx_size_mb': model_size_mb,
            'tensorrt_path': output_trt if output_trt and os.path.exists(output_trt) else None,
            'precision': precision,
            'max_batch_size': max_batch_size,
            'max_seq_len': max_seq_len,
            'onnx_runtime_latency_ms': avg_latency_ms,
            'onnx_runtime_throughput': throughput,
        }
        
        import json
        info_path = output_onnx.replace('.onnx', '_info.json')
        with open(info_path, 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        print(f"\n💾 Deployment info saved: {info_path}")
        
        return output_onnx
        
    except Exception as e:
        print(f"\n❌ ONNX export failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Export TRM model for CPU/GPU inference')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--output', type=str, default='model_cpu.pt', help='Output path for CPU model')
    parser.add_argument('--quant', type=str, default='int8', choices=['int8', 'bnn', 'dynamic'],
                       help='Quantization type for CPU')
    
    # ONNX export arguments
    parser.add_argument('--export-onnx', action='store_true', help='Export to ONNX format')
    parser.add_argument('--onnx-path', type=str, default='model.onnx', help='ONNX output path')
    parser.add_argument('--tensorrt-path', type=str, default=None, help='TensorRT engine output path')
    parser.add_argument('--precision', type=str, default='fp16', choices=['fp32', 'fp16', 'int8'],
                       help='Inference precision for ONNX/TensorRT')
    parser.add_argument('--max-batch-size', type=int, default=1, help='Maximum batch size')
    parser.add_argument('--max-seq-len', type=int, default=1024, help='Maximum sequence length')
    
    args = parser.parse_args()
    
    # CPU optimizations for i5-1035G1
    torch.set_num_threads(4)
    torch.backends.mkldnn.enabled = True
    
    # Export to ONNX if requested
    if args.export_onnx:
        print("\n" + "="*70)
        print("  ONNX Export Mode")
        print("="*70 + "\n")
        export_onnx_tensorrt(
            model_path=args.model,
            output_onnx=args.onnx_path,
            output_trt=args.tensorrt_path,
            precision=args.precision,
            max_batch_size=args.max_batch_size,
            max_seq_len=args.max_seq_len,
        )
    else:
        # CPU quantization export
        print("\n" + "="*70)
        print("  CPU Quantization Mode")
        print("="*70 + "\n")
        export_cpu_optimized(args.model, args.output, args.quant)
