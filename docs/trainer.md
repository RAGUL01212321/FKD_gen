# Distillation Trainer (`trainer.py`)

This file implements the custom training logic required for multi-teacher feature-based distillation.

## `DistillationTrainer` Class
Inherits from the Hugging Face `Trainer` class to leverage standard training utilities while overriding the loss calculation.

### Initialization
- **`teacher_models`**: A list of pre-trained models to act as teachers.
- **`distillation_models`**: A list of `FeatureDistillationWrapper` instances, one for each teacher.
- **Freezing Teachers**: All teacher parameters are automatically frozen (`requires_grad = False`) and set to evaluation mode (`eval()`) to save memory and computation.

### Custom Loss: `compute_loss`
The core of the distillation process happens here. The total loss is calculated as:

1. **Task Loss**: The standard Language Modeling loss from the student model.
2. **Cumulative Distillation Loss**: For each teacher in the list:
   - Perform a forward pass with the teacher (no gradients).
   - Use the corresponding wrapper to get feature pairs.
   - Calculate **MSE Loss** to align absolute values.
   - Calculate **Cosine Similarity Loss** to align vector directions.
   - The combined feature loss is added to the total.

### Formula
$$ Total\_Loss = (w_{task} \times Loss_{Task}) + (w_{distill} \times \sum_{i} Loss_{Distill, i}) $$

## Optimization
The trainer is designed to handle multiple models in memory. It uses standard `Trainer` arguments like `fp16` and `gradient_accumulation_steps` to remain efficient.
