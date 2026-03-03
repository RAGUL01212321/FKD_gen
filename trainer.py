import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from typing import Dict, Union, Any, Optional, List

class DistillationTrainer(Trainer):
    def __init__(
        self,
        teacher_models: List[nn.Module], # List of teachers
        distillation_models: List[nn.Module], # List of wrappers (one per teacher)
        distill_weight: float = 1.0,
        task_weight: float = 1.0,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.teacher_models = teacher_models
        self.distillation_wrappers = distillation_models
        self.distill_weight = distill_weight
        self.task_weight = task_weight
        
        # Move teachers to device and set to eval
        for t_model in self.teacher_models:
            t_model.to(self.args.device).eval()
            for param in t_model.parameters():
                param.requires_grad = False
                
        # Move wrappers to device
        for d_model in self.distillation_wrappers:
            d_model.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Cumulative Loss: Task Loss + Sum(Distillation Loss from each Teacher)
        """
        student_outputs = model(**inputs, output_hidden_states=True)
        task_loss = student_outputs.loss
        
        cumulative_distill_loss = 0.0
        
        for teacher, wrapper in zip(self.teacher_models, self.distillation_wrappers):
            # Get teacher outputs (no grad)
            with torch.no_grad():
                teacher_outputs = teacher(**inputs, output_hidden_states=True)
            
            # Feature Distillation Loss for this specific teacher
            distill_pairs = wrapper(
                student_outputs.hidden_states,
                teacher_outputs.hidden_states
            )
            
            step_distill_loss = 0.0
            for s_feat, t_feat in distill_pairs:
                mse_loss = F.mse_loss(s_feat, t_feat)
                
                s_flat = s_feat.view(-1, s_feat.size(-1))
                t_flat = t_feat.view(-1, t_feat.size(-1))
                cos_loss = 1.0 - F.cosine_similarity(s_flat, t_flat).mean()
                
                step_distill_loss += (mse_loss + cos_loss)
                
            cumulative_distill_loss += (step_distill_loss / len(distill_pairs))
            
        # Total Loss
        total_loss = (self.task_weight * task_loss) + (self.distill_weight * cumulative_distill_loss)
        
        return (total_loss, student_outputs) if return_outputs else total_loss
