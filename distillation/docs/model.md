# Feature Distillation Models (`model.py`)

This file contains the architectural components required to align student and teacher features for distillation.

## Components

### 1. `FeatureDistillationWrapper`
This is a standard PyTorch module that acts as a bridge between the student's hidden states and the teacher's hidden states.

- **Projection Layers**: Since students often have smaller hidden dimensions than teachers, the wrapper contains a `nn.ModuleDict` of linear layers (`nn.Linear`). These project the student's feature vectors into the teacher's vector space.
- **Forward Pass**: It takes the hidden states of both models, applies the applicable projections, and returns pairs of (projected_student_features, teacher_features) for loss calculation.

### 2. `get_layer_mapping`
A utility function that maps student layers to teacher layers.

- **Logic**: It uses a uniform stride mapping. If a teacher is deeper, it selects evenly spaced layers to ensure the student captures the full depth of the teacher's knowledge.
- **Example**: Mapping 6 student layers to 12 teacher layers results in mapping student layer `i` to teacher layer `2*i`.

## Why Projections?
Projections are necessary because we cannot directly compare vectors of different sizes (e.g., student $768$ vs teacher $1024$). The projection layers are learnable, allowing the model to find the best way to represent student knowledge in the teacher's space.

## Initialization
The projection layers are initialized using `xavier_uniform_` to ensure stable gradients at the start of training.
