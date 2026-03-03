import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple

class FeatureDistillationWrapper(nn.Module):
    """
    Wrapper to align student and teacher hidden states using projection layers.
    """
    def __init__(
        self,
        student_config,
        teacher_config,
        layer_mapping: Dict[int, int]
    ):
        super().__init__()
        self.layer_mapping = layer_mapping
        
        # Projection layers to align student hidden_size to teacher hidden_size
        self.projections = nn.ModuleDict({
            str(s_idx): nn.Linear(student_config.hidden_size, teacher_config.hidden_size)
            for s_idx in layer_mapping.keys()
        })
        
        # Initialize projections with Identity-like or small weights
        for proj in self.projections.values():
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(
        self,
        student_hidden_states: Tuple[torch.Tensor, ...],
        teacher_hidden_states: Tuple[torch.Tensor, ...]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns pairs of (projected_student_features, teacher_features)
        """
        pairs = []
        for s_idx, t_idx in self.layer_mapping.items():
            s_feat = student_hidden_states[s_idx]
            t_feat = teacher_hidden_states[t_idx]
            
            # Project student feature to teacher's dimension
            s_feat_proj = self.projections[str(s_idx)](s_feat)
            
            pairs.append((s_feat_proj, t_feat))
            
        return pairs

def get_layer_mapping(num_student_layers: int, num_teacher_layers: int) -> Dict[int, int]:
    """
    Stupidly simple uniform mapping: map student layers to evenly spaced teacher layers.
    Example: 6 student layers, 12 teacher layers -> {0:0, 1:2, 2:4, 3:6, 4:8, 5:10}
    """
    mapping = {}
    stride = num_teacher_layers // num_student_layers
    for i in range(num_student_layers):
        mapping[i] = i * stride
    return mapping
