
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import pandas as pd

def evaluate_model(model_path, dataset_name, output_file):
    """
    Evaluates a fine-tuned model on a given dataset and logs the results.
    - **model_path**: The path to the fine-tuned model.
    - **dataset_name**: The name of the dataset to use for evaluation.
    - **output_file**: The file to save the evaluation results.
    """
    # Load dataset
    dataset = load_dataset(dataset_name, split="test")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # If using PEFT, load the PeftModel
    # model = PeftModel.from_pretrained(model, model_path)

    results = []
    for entry in dataset:
        # This is a simplified example. A real evaluation would be more complex.
        inputs = tokenizer(entry['text'], return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        results.append({
            "input_text": entry['text'],
            "generated_text": generated_text,
            "label": entry['label']
        })

    # Save results to a CSV file
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Evaluation results saved to {output_file}")

if __name__ == '__main__':
    # Example of how to test this module directly
    # This is a placeholder and will not run without a proper environment and data
    print("Testing model evaluation...")
    # evaluate_model(
    #     model_path="./fine-tuned-model",
    -    # dataset_name="imdb",
    #     output_file="./evaluation_results.csv"
    # )
    print("Evaluation script is set up.")
