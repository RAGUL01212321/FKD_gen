import torch
from transformers import AutoModelForCausalLM, AutoConfig, TrainingArguments
from distillation.model import FeatureDistillationWrapper, get_layer_mapping
from distillation.trainer import DistillationTrainer
from distillation.dataset import get_distillation_dataset
import os
from typing import List

def run_distillation_step(teacher_paths: List[str], student_id: str, output_dir: str, dataset):
    print(f"\n--- Distilling {teacher_paths} -> {student_id} ---")
    
    # 1. Load All Teachers
    teachers = []
    wrappers = []
    
    # We load the current student to get its config
    student = AutoModelForCausalLM.from_pretrained(student_id)
    s_config = student.config
    num_s_layers = s_config.n_layer + 1
    
    for t_path in teacher_paths:
        print(f"Loading Teacher from: {t_path}")
        teacher = AutoModelForCausalLM.from_pretrained(t_path)
        t_config = teacher.config
        num_t_layers = t_config.n_layer + 1
        
        mapping = get_layer_mapping(num_s_layers, num_t_layers)
        wrapper = FeatureDistillationWrapper(s_config, t_config, mapping)
        
        teachers.append(teacher)
        wrappers.append(wrapper)
    
    # 2. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1, # Very small as multiple models are in VRAM
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=5e-5,
        logging_steps=10,
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False
    )
    
    # 3. Initialize Multi-Teacher Trainer
    trainer = DistillationTrainer(
        model=student,
        teacher_models=teachers,
        distillation_models=wrappers,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if "validation" in dataset else None
    )
    
    # 4. Start Training
    trainer.train()
    
    # 5. Save specialized student
    print(f"Saving distilled student to {output_dir}")
    student.save_pretrained(output_dir)
    return output_dir
    
def main():
    # Define the distillation chain: Teacher -> S1 -> S2 -> S3
    model_chain = [
        "mradermacher/Apollo-1-8B-GGUF",  # Root Teacher (Might need conversion from GGUF depending on env)
        "mradermacher/apollo-astralis-4b-GGUF",  # S1
        "FreedomIntelligence/Apollo-2B",  # S2
        # Add more model IDs or local paths here
    ]
    
    print(f"Starting Multi-Teacher Distillation Chain: {' -> '.join(model_chain)}")
    
    # Prepare Dataset once
    print("Preparing Dataset...")
    dataset, _ = get_distillation_dataset(tokenizer_name=model_chain[0])
    
    all_preceding_paths = [model_chain[0]]
    
    for i in range(1, len(model_chain)):
        current_student = model_chain[i]
        step_output_dir = f"./distillation_step_{i}"
        
        # Run step with ALL preceding models as teachers
        saved_path = run_distillation_step(
            teacher_paths=all_preceding_paths,
            student_id=current_student,
            output_dir=step_output_dir,
            dataset=dataset
        )
        
        # Add the newly trained student to the teacher list for the NEXT student
        all_preceding_paths.append(saved_path)
        print(f"Step {i} complete. Model saved at {saved_path} added to teacher list.")

    print("\nFull Cumulative Distillation Chain Complete!")

if __name__ == "__main__":
    main()