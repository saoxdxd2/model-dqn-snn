"""
Simple batch size scaler for post-compilation optimization.
Increases batch size after torch.compile completes and memory stabilizes.
"""
import torch


class PostCompilationBatchScaler:
    """Increases batch size once torch.compile completes."""
    
    def __init__(
        self,
        initial_batch_size: int,
        target_batch_size: int,
        stabilization_steps: int = 20
    ):
        self.initial_batch_size = initial_batch_size
        self.target_batch_size = target_batch_size
        self.stabilization_steps = stabilization_steps
        
        self.current_batch_size = initial_batch_size
        self.compilation_complete = False
        self.stable_steps = 0
        self.memory_history = []
        
    def check_and_scale(self, current_step: int) -> int:
        """Check if we should scale up batch size."""
        
        if self.compilation_complete:
            return self.current_batch_size
        
        if not torch.cuda.is_available():
            return self.current_batch_size
        
        # Track GPU memory
        current_mem = torch.cuda.memory_allocated(0) / 1e9  # GB
        self.memory_history.append(current_mem)
        
        # Need at least 20 steps to detect stability
        if len(self.memory_history) < self.stabilization_steps:
            return self.current_batch_size
        
        # Check if memory is stable (variance < 0.5GB over last 20 steps)
        recent_memory = self.memory_history[-self.stabilization_steps:]
        memory_variance = max(recent_memory) - min(recent_memory)
        
        if memory_variance < 0.5:  # Stable within 0.5GB
            # Compilation complete - scale up batch size
            self.compilation_complete = True
            self.current_batch_size = self.target_batch_size
            
            avg_mem = sum(recent_memory) / len(recent_memory)
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            print(f"\n{'='*70}")
            print(f"  🚀 TORCH.COMPILE COMPLETED - SCALING BATCH SIZE")
            print(f"{'='*70}")
            print(f"  Memory stabilized at: {avg_mem:.1f}GB / {total_mem:.1f}GB")
            print(f"  Increasing batch: {self.initial_batch_size} → {self.target_batch_size}")
            print(f"  Expected new memory: ~{avg_mem * (self.target_batch_size / self.initial_batch_size):.1f}GB")
            print(f"  Speed increase: +{((self.target_batch_size / self.initial_batch_size) - 1) * 100:.0f}%")
            print(f"{'='*70}\n")
            
        return self.current_batch_size
