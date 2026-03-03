# Dataset Preparation (`dataset.py`)

This file handles the loading and preprocessing of datasets for the knowledge distillation process.

## Key Function: `get_distillation_dataset`

The primary utility in this file is `get_distillation_dataset`, which performs the following steps:

### 1. Tokenization
- Uses `AutoTokenizer` from the `transformers` library.
- Automatically handles padding by setting `pad_token` to `eos_token` if not already defined.
- Truncates and pads sequences to a specified `max_length` (default: 512).

### 2. Data Cleaning
- Filters out empty or very short text entries (less than 10 characters) to ensure high-quality training data.

### 3. Feature Engineering (Labels)
- Since this is a Causal Language Modeling (CLM) task, it clones the `input_ids` to create a `labels` column.
- This allows the trainer to calculate the standard Cross-Entropy loss for the language modeling task.

### 4. Format Conversion
- Converts the dataset to `torch` format for compatibility with PyTorch training loops.

## Usage Example

```python
from distillation.dataset import get_distillation_dataset

dataset, tokenizer = get_distillation_dataset(
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    tokenizer_name="gpt2"
)
```
