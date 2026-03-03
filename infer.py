import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

def infer(model_path, prompt, max_length=100):
    print(f"Loading model from: {model_path}")
    
    # 1. Load the tokenizer and model
    # Note: We use the base model id for the tokenizer if the distilled model doesn't have one saved
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except:
        # Fallback to gpt2 if specific tokenizer not found
        print("Tokenizer not found in model path, falling back to gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # 2. Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # 3. Generate
    print(f"Generating for prompt: '{prompt}'...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id
        )

    # 4. Decode and print
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n--- Generated Output ---")
    print(result)
    print("------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer using a distilled model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the distilled model directory")
    parser.add_argument("--prompt", type=str, default="The future of artificial intelligence is", help="The prompt to generate from")
    
    args = parser.parse_args()
    infer(args.model_path, args.prompt)
