# Module 2: models.py

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForSequenceClassification
from trl import PPOConfig, create_reference_model, PPOTrainer

def load_peft_model(model_name, device):
    # Load and configure the PEFT model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer

def load_ppo_model(peft_model, device):
    # Initialize the RL PPO model based on the PEFT model
    config = PPOConfig(
        model_name=peft_model.config.model_name,
        learning_rate=1.41e-5,
        ppo_epochs=1,
        mini_batch_size=4,
        batch_size=16
    )
    ppo_model = AutoModelForSeq2SeqLM.from_pretrained(peft_model, torch_dtype=torch.bfloat16, is_trainable=True)
    ppo_model = ppo_model.to(device)
    ref_model = create_reference_model(peft_model)
    ref_model = ref_model.to(device)

    return ppo_model, ref_model, config

def load_toxicity_model(model_name, device):
    # Load the toxicity evaluation model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model.to(device)

    return model, tokenizer
