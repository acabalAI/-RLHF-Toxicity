# main.py

import time
import numpy as np
from transformers import TrainingArguments, Trainer, AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
from trl import PPOTrainer, create_reference_model
from trl.core import LengthSampler

from data import load_and_preprocess_dataset
from models import load_peft_model, load_ppo_model, load_toxicity_model
from training import train_peft_model, train_ppo_model
from evaluation import evaluate_toxicity, generate_text

def main():
    # Configuration parameters
    dataset_name = 'knkarthick/dialogsum'
    input_min_text_length = 200
    input_max_text_length = 1000
    max_ppo_steps = 10
    num_samples = 20
    output_dir = './peft-dialogue-summary-training-' + str(int(time.time()))
    model_name = 'google/flan-t5-base'  # Define your model name here
    toxicity_model_name = 'facebook/roberta-hate-speech-dynabench-r4-target'
    device = torch.device("cuda:0")

    # Load and preprocess the dataset
    dataset = load_and_preprocess_dataset(dataset_name, input_min_text_length, input_max_text_length)

    # Load and configure the models
    peft_model, peft_tokenizer = load_peft_model(model_name, device)
    ppo_model, ref_model, ppo_config = load_ppo_model(peft_model, device)
    toxicity_model, toxicity_tokenizer = load_toxicity_model(toxicity_model_name, device)

    # Train the PEFT model
    train_peft_model(peft_model, tokenized_datasets, output_dir)

    # Train the RL PPO model
    train_ppo_model(ppo_model, ref_model, peft_tokenizer, dataset, output_length_sampler, max_ppo_steps)

    # Evaluate toxicity of generated text
    mean_toxicity, std_toxicity = evaluate_toxicity(ppo_model, toxicity_model, peft_tokenizer, dataset['test'], num_samples)

    # Generate text using the model
    input_text = "Enter your input text here."
    generated_text = generate_text(ppo_model, input_text, peft_tokenizer)

if __name__ == '__main__':
    main()
