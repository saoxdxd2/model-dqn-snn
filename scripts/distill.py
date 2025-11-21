import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig
import logging
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from scripts.load_model import ModelLoader

logger = logging.getLogger(__name__)

class DistillationTrainer:
    def __init__(self, teacher_path: str, student_config: DictConfig, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.teacher = self._load_teacher(teacher_path)
        self.student = TinyRecursiveReasoningModel_ACTV1(student_config).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=1e-4)
        
        # Loss functions
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Hyperparameters
        self.temperature = 2.0
        self.alpha = 0.5 # Weight for distillation loss (vs task loss)

    def _load_teacher(self, path: str):
        logger.info(f"Loading teacher model from {path}...")
        loader = ModelLoader(path, device=self.device)
        model = loader.load_model(quantized=False)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    def train_step(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        target_ids = batch['target_ids'].to(self.device) # If available, else use input_ids for self-supervised
        
        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_logits = self.teacher(input_ids)
        
        # Student forward
        student_logits = self.student(input_ids)
        
        # 1. Distillation Loss (KL Divergence)
        # Soft targets from teacher
        soft_targets = F.log_softmax(teacher_logits / self.temperature, dim=-1)
        soft_prob = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        distill_loss = self.kl_div_loss(soft_prob, soft_targets) * (self.temperature ** 2)
        
        # 2. Task Loss (Cross Entropy with hard targets)
        # Shift for next-token prediction if standard LM training
        # Assuming input_ids shape [B, T] and logits [B, T, V]
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        task_loss = self.ce_loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Combined Loss
        loss = (self.alpha * distill_loss) + ((1 - self.alpha) * task_loss)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), distill_loss.item(), task_loss.item()

    def train_loop(self, dataloader: DataLoader, num_epochs: int = 1):
        self.student.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for i, batch in enumerate(dataloader):
                loss, d_loss, t_loss = self.train_step(batch)
                total_loss += loss
                
                if i % 10 == 0:
                    print(f"Epoch {epoch} | Step {i} | Loss: {loss:.4f} (Distill: {d_loss:.4f}, Task: {t_loss:.4f})")
            
            print(f"Epoch {epoch} Average Loss: {total_loss / len(dataloader):.4f}")
            
            # Save checkpoint
            torch.save(self.student.state_dict(), f"student_checkpoint_epoch_{epoch}.pt")

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    print("Starting Distillation...")
    
    # Dummy setup for demonstration - in real usage, would load dataset from cfg
    # teacher_path = cfg.distillation.teacher_path
    # student_config = cfg.model
    
    # For now, just printing placeholder info as we don't have a full config for distillation yet
    print("Distillation script structure ready.")
    print("To use, configure 'teacher_path' and 'student_config' in main()")

if __name__ == "__main__":
    main()
