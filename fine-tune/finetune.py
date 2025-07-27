
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from unsloth import FastLanguageModel

def finetune_model(model_name, dataset_name, output_dir):
    """
    Fine-tunes a model on a given dataset using PEFT and Unsloth.
    - **model_name**: The name of the model to fine-tune.
    - **dataset_name**: The name of the dataset to use for fine-tuning.
    - **output_dir**: The directory to save the fine-tuned model.
    """
    # Load dataset
    dataset = load_dataset(dataset_name, split="train")

    # Load tokenizer and model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    # Configure PEFT with LoRA
    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=100,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    print(f"Model fine-tuned and saved to {output_dir}")

if __name__ == '__main__':
    # Example of how to test this module directly
    # This is a placeholder and will not run without a proper environment and data
    print("Testing model fine-tuning...")
    # finetune_model(
    #     model_name="unsloth/llama-2-7b-bnb-4bit",
    #     dataset_name="imdb",
    #     output_dir="./fine-tuned-model"
    # )
    print("Fine-tuning script is set up.")
