# Cumulative Multi-Teacher LLM Distillation

A structured and optimized framework for distilling a chain of Large Language Models (LLMs) using feature-based distillation with cumulative teacher feedback.

## 🚀 Overview

This project implements a recursive distillation pipeline where a "Root Teacher" is distilled into a sequence of smaller student models. What makes this implementation unique is its **Cumulative Multi-Teacher** approach: each new student in the chain learns not only from the original teacher but also from **all preceding intermediate students**.
 
### The Theory: Feature-Based Distillation

Unlike traditional KD (Knowledge Distillation) which matches output probabilities (Logits), **Feature-Based Distillation** forces the student to mimic the *internal representations* (hidden states) of the teacher. This often results in more robust student models as they capture the "reasoning process" of the larger model.

---

### 🧩 Deep Dive: Feature Alignment Wrapper

The `FeatureDistillationWrapper` (found in `distillation/model.py`) is the architectural bridge between the student and the teacher. It solves two critical problems:

#### 1. Dimension Matching (The Projection)
Most distillation pairs have mismatched hidden dimensions (e.g., GPT-2 Medium $1024$ vs GPT-2 Base $768$). 
- **How it works**: For every aligned layer, the wrapper maintains a learnable `nn.Linear` layer.
- **Transformation**: $h'_{student} = Linear(h_{student})$. This projects the student's representation into the teacher's vector space ($d_{student} \to d_{teacher}$) so they can be compared directly via MSE.

#### 2. Depth Matching (Layer Mapping)
Teachers are usually deeper than students. 
- **Strategy**: We use a **Stride-based Mapping**. If a teacher has 24 layers and a student has 12, the student's 1st layer mimics the teacher's 1st, the student's 2nd mimics the teacher's 3rd, and so on.
- **Implementation**: The `get_layer_mapping` function calculates this stride automatically to ensure the student captures the "evolution" of features through the teacher's entire depth.

#### 3. Gradient Flow
Crucially, both the **Student Model** and the **Projection Layers** are updated during training. This allows the projections to learn *how* to best translate student knowledge into the teacher's format, while the student learns to actually *produce* that knowledge.

---

### Core Architecture

### 1. Cumulative Teacher Chain
For a chain defined as `[Teacher, S1, S2, ..., Sn]`:
- **S1** learns from `{Teacher}`.
- **S2** learns from `{Teacher, S1}`.
- **Sn** learns from `{Teacher, S1, S2, ..., Sn-1}`.

### 2. Feature Alignment Wrapper
Since students often have different hidden dimensions than their teachers (e.g., mismatch between 1024 and 768), we use a `FeatureDistillationWrapper`. 
- **Projections**: Learnable linear layers that project student features into the teacher's dimension.
- **Layer Mapping**: Automatically aligns student layers to evenly spaced teacher layers.

### 3. Loss Function
For every teacher-student pair, the loss is a combination of:
- **MSE (Mean Squared Error)**: Aligns the absolute values of the feature vectors.
- **Cosine Similarity**: Aligns the direction/orientation of the representations in vector space.
- **Task Loss**: Standard Language Modeling (Cross-Entropy) loss on the target dataset.

$$ Loss_{Total} = Loss_{LM} + \sum (Loss_{MSE} + Loss_{Cosine}) $$

---

### Project Structure

- [distillation/dataset.py](file:///home/ragul/Desktop/FKD1/distillation/dataset.md): Dataset loading and preprocessing logic.
- [distillation/model.py](file:///home/ragul/Desktop/FKD1/distillation/model.md): Alignment wrappers and projection layers.
- [distillation/trainer.py](file:///home/ragul/Desktop/FKD1/distillation/trainer.md): Custom distillation loss and training loop.
- [train.py](file:///home/ragul/Desktop/FKD1/train_info.md): Main orchestration script for the distillation chain.
- [execution_flow.md](file:///home/ragul/Desktop/FKD1/execution_flow.md): Step-by-step sequence of function calls.

---

## 🚀 Getting Started

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Configuration
Open `train.py` and define your `model_chain`:
```python
model_chain = [
    "mradermacher/Apollo-1-8B-GGUF", 
    "mradermacher/apollo-astralis-4b-GGUF",
    "FreedomIntelligence/Apollo-2B"
]
```

### 3. Run Training
```bash
python train.py
```

---

## 🔍 Inference

After training, you can use the `infer.py` script to test your distilled student model:

```bash
python infer.py --model_path ./distillation_step_1 --prompt "Artificial intelligence is"
```

### Options:
- `--model_path`: Path to the directory containing the saved student model.
- `--prompt`: The text to start generation from.

---

## 💡 Optimization & VRAM
Hold multiple teachers in memory is expensive. By default, the script:
- Sets `per_device_train_batch_size = 1`.
- Uses `gradient_accumulation_steps = 8`.
- Disables gradients for all teacher models.
- Uses FP16 precision if a CUDA device is available.

> [!TIP]
> If you run out of VRAM, consider using **Gradient Checkpointing** or distilling fewer models in a single chain.


killgfat/Apollo-7B-GGUF
FreedomIntelligence/Apollo-6B
FreedomIntelligence/Apollo-1.8B