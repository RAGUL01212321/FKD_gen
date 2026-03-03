from datasets import load_dataset
from transformers import AutoTokenizer

def get_distillation_dataset(
    dataset_name: str = "",
    dataset_config: str = "",
    tokenizer_name: str = "",
    max_length: int = 512
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    raw_datasets = load_dataset(dataset_name, dataset_config)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )

    # Filter out empty texts
    raw_datasets = raw_datasets.filter(lambda x: len(x["text"]) > 10)
    
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names
    )
    
    tokenized_datasets.set_format("torch")
    
    # Add labels for LM task (same as input_ids)
    def add_labels(examples):
        examples["labels"] = examples["input_ids"].clone()
        return examples
        
    tokenized_datasets = tokenized_datasets.map(add_labels, batched=False)
    
    return tokenized_datasets, tokenizer