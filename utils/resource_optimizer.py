"""
Dynamic Resource Optimizer
Automatically adjusts batch size and num_workers based on real-time system resources.
"""
import torch
import psutil
import time
from typing import Dict, Tuple
import os


class DynamicResourceOptimizer:
    """Monitors and optimizes GPU/CPU/RAM usage during training."""
    
    def __init__(
        self,
        initial_batch_size: int = 32,
        min_batch_size: int = 16,
        max_batch_size: int = 1024,
        target_gpu_utilization: float = 0.95,  # Target 95% GPU utilization
        initial_num_workers: int = 0,
        check_interval: int = 5  # Check every 5 steps for faster adaptation
    ):
        self.batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_gpu_utilization = target_gpu_utilization
        self.num_workers = initial_num_workers
        self.check_interval = check_interval
        
        # Performance tracking
        self.step_times = []
        self.gpu_memory_history = []
        self.last_adjustment_step = 0
        self.stable_steps = 0  # Steps since last OOM
        
        # Get GPU info
        if torch.cuda.is_available():
            self.gpu_total_memory = torch.cuda.get_device_properties(0).total_memory
        else:
            self.gpu_total_memory = 0
            
        # Get CPU info
        self.cpu_count = os.cpu_count() or 1
        
    def get_gpu_memory_usage(self) -> Tuple[float, float]:
        """Returns (used_memory_gb, total_memory_gb)."""
        if not torch.cuda.is_available():
            return 0.0, 0.0
        
        used = torch.cuda.memory_allocated(0)
        total = self.gpu_total_memory
        return used / 1e9, total / 1e9
    
    def get_cpu_usage(self) -> Dict[str, float]:
        """Returns CPU and RAM usage statistics."""
        return {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'ram_percent': psutil.virtual_memory().percent,
            'ram_available_gb': psutil.virtual_memory().available / 1e9
        }
    
    def suggest_batch_size(self, current_step: int) -> int:
        """Dynamically adjust batch size based on GPU memory."""
        # Don't adjust too frequently
        if current_step - self.last_adjustment_step < self.check_interval:
            return self.batch_size
        
        used_mem, total_mem = self.get_gpu_memory_usage()
        
        if total_mem == 0:
            return self.batch_size
        
        current_utilization = used_mem / total_mem
        self.gpu_memory_history.append(current_utilization)
        
        # After torch.compile (step 10+), scale aggressively to target
        warmup_complete = self.stable_steps >= 10
        
        if not warmup_complete:
            self.stable_steps += 1
            # Conservative during warmup (torch.compile phase)
            if current_utilization < 0.5 and self.stable_steps >= 5:
                # Very low usage - safe to increase modestly
                new_batch_size = min(int(self.batch_size * 1.5), self.max_batch_size)
                if new_batch_size != self.batch_size:
                    print(f"\n📈 Warmup increase: {self.batch_size} → {new_batch_size}")
                    print(f"   GPU: {used_mem:.1f}GB / {total_mem:.1f}GB ({current_utilization*100:.1f}%)")
                    self.batch_size = new_batch_size
                    self.last_adjustment_step = current_step
            return self.batch_size
        
        # Post-warmup: Maintain 95% target (only scale if significantly off)
        tolerance = 0.03  # ±3% tolerance
        
        # Only scale up if significantly underutilized (for non-optimized configs)
        if current_utilization < 0.7:
            # Very low usage - likely unoptimized config, scale up
            if current_utilization < 0.5:
                scale = 1.5  # Lots of room - big jump
            else:
                scale = 1.3  # Moderate room
            
            new_batch_size = min(int(self.batch_size * scale), self.max_batch_size)
            if new_batch_size != self.batch_size:
                print(f"\n📈 Scaling up: {self.batch_size} → {new_batch_size}")
                print(f"   GPU: {used_mem:.1f}GB / {total_mem:.1f}GB ({current_utilization*100:.1f}%)")
                self.batch_size = new_batch_size
                self.last_adjustment_step = current_step
        
        # Always scale down if over target (safety)
        elif current_utilization > self.target_gpu_utilization + tolerance:
            # Over target - reduce to stay safe
            new_batch_size = max(int(self.batch_size * 0.9), self.min_batch_size)
            if new_batch_size != self.batch_size:
                print(f"\n📉 Scaling down: {self.batch_size} → {new_batch_size}")
                print(f"   GPU: {used_mem:.1f}GB / {total_mem:.1f}GB ({current_utilization*100:.1f}%)")
                self.batch_size = new_batch_size
                self.last_adjustment_step = current_step
        else:
            # Within tolerance - perfect!
            if current_step % 50 == 0:
                print(f"\n✅ Optimal batch size: {self.batch_size} (GPU: {current_utilization*100:.1f}%)")
        
        return self.batch_size
    
    def suggest_num_workers(self) -> int:
        """Dynamically adjust num_workers based on CPU availability."""
        cpu_stats = self.get_cpu_usage()
        cpu_usage = cpu_stats['cpu_percent']
        
        # Conservative: use up to 50% of available CPU cores
        # Leave headroom for main training process
        available_cores = max(1, self.cpu_count // 2)
        
        if cpu_usage < 50:
            # CPU underutilized - can add workers
            suggested = min(available_cores, 4)  # Cap at 4 workers
        elif cpu_usage > 80:
            # CPU overloaded - reduce workers
            suggested = max(0, self.num_workers - 1)
        else:
            # Stable - keep current
            suggested = self.num_workers
        
        if suggested != self.num_workers:
            print(f"\n🔧 Adjusting workers: {self.num_workers} → {suggested} (CPU: {cpu_usage:.1f}%)")
            self.num_workers = suggested
        
        return self.num_workers
    
    def on_oom_error(self):
        """Called when OOM occurs - aggressively reduce batch size."""
        new_batch_size = max(int(self.batch_size * 0.5), self.min_batch_size)
        print(f"\n❌ OOM detected! Reducing batch size: {self.batch_size} → {new_batch_size}")
        self.batch_size = new_batch_size
        self.stable_steps = 0  # Reset stability counter
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def print_status(self):
        """Print current resource usage."""
        used_mem, total_mem = self.get_gpu_memory_usage()
        cpu_stats = self.get_cpu_usage()
        
        print(f"\n📊 Resource Status:")
        print(f"   GPU: {used_mem:.1f}GB / {total_mem:.1f}GB ({used_mem/total_mem*100:.1f}%)")
        print(f"   CPU: {cpu_stats['cpu_percent']:.1f}%")
        print(f"   RAM: {cpu_stats['ram_percent']:.1f}% ({cpu_stats['ram_available_gb']:.1f}GB available)")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Workers: {self.num_workers}")
