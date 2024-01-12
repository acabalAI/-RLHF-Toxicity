
from transformers import TrainingArguments, Trainer

def train_peft_model(peft_model, tokenized_datasets, output_dir):
    # Define the training arguments and initialize the trainer for PEFT model
    output_dir = f'./peft-dialogue-summary-training-{str(int(time.time()))}'
    peft_training_args = TrainingArguments(
        output_dir=output_dir,
        auto_find_batch_size=True,
        learning_rate=1e-3,
        num_train_epochs=10,
        logging_steps=10,
        max_steps=10
    )
    peft_trainer = Trainer(model=peft_model, args=peft_training_args, train_dataset=tokenized_datasets["train"])
    
    # Start the PEFT model training
    peft_trainer.train()

def train_ppo_model(ppo_model, ref_model, tokenizer, dataset, output_length_sampler, max_ppo_steps):
    # Initialize and train the RL PPO model
    ppo_trainer = PPOTrainer(ppo_model, ref_model, tokenizer, dataset, output_length_sampler)
    for step in range(max_ppo_steps):
        ppo_trainer.train_step()
