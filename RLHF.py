# Install and download Libraries and Dependencies

from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
from transformers import TrainingArguments, Trainer, pipeline, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig, LoraConfig, TaskType
from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead, create_reference_model
from trl.core import LengthSampler

import torch
import evaluate
import pandas as pd
import numpy as np
import time
from tqdm import tqdm

torch.cuda.is_available()
device = torch.device("cuda:0")

# Load Data and LLM (FLAN-T5), prepare Reward Model and Toxicity Evaluator
huggingface_dataset_name = 'knkarthick/dialogsum'
dataset_original = load_dataset(huggingface_dataset_name)

example_indices = [0, 50]
dash_line = '-' * 100

for i, index in enumerate(example_indices):
    print(dash_line)
    print(f'Example {i + 1}')
    print(dash_line)
    print(dataset_original['test'][index]['dialogue'])
    print(dataset_original['test'][index]['summary'])

model_name = 'google/flan-t5-base'
original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32)
original_model = original_model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name)

def build_dataset(model_name, dataset_name, input_min_text_length, input_max_text_length):
    dataset = load_dataset(dataset_name, split='train')
    dataset = dataset.filter(lambda x: len(x['dialogue']) > input_min_text_length and len(x['dialogue']) <= input_max_text_length)
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map='auto')

    def tokenize(sample):
        prompt = f"""
        Summarize the following conversation.
        {sample['dialogue']}
        Summary:
        """
        sample["input_ids"] = tokenizer.encode(prompt)
        sample['query'] = tokenizer.decode(sample['input_ids'])
        return sample

    dataset = dataset.map(tokenize, batched=False)
    dataset.set_format(type="torch")

    dataset_splits = dataset.train_test_split(test_size=0.2, shuffle=False, seed=42)
    return dataset_splits

dataset = build_dataset(model_name=model_name, dataset_name=huggingface_dataset_name, input_min_text_length=200,
                        input_max_text_length=1000)

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()

    return (f'trainable model params: {trainable_model_params}\nall model parameters {all_model_params}')

def tokenize_function(example):
    start_prompt = 'Summarize the following conversation.n\n'
    end_prompt = 'n\nSummary:'
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example['dialogue']]
    example['input_ids'] = tokenizer(prompt, padding='max_length', truncation=True, return_tensors='pt').input_ids
    example['labels'] = tokenizer(example['summary'], padding='max_length', truncation=True, return_tensors='pt').input_ids
    return example

tokenized_datasets = dataset.map(tokenize_function, batched=True, batch_size=8)
tokenized_datasets = tokenized_datasets.remove_columns(['id', 'topic', 'dialogue', 'summary'])

# Reduce dataset
tokenized_datasets = tokenized_datasets.filter(lambda example, index: index % 100 == 0, with_indices=True)

# Prepare Model to Train, in this case a PEFT Model
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=['q', 'v'],
    lora_dropout=0.05,
    bias='none',
    task_type=TaskType.SEQ_2_SEQ_LM
)

peft_model = get_peft_model(original_model, lora_config)
print(print_number_of_trainable_model_parameters(peft_model))
peft_model = peft_model.to(device)

output_dir = f'./peft-dialogue-summary-training-{str(int(time.time()))}'
peft_training_args = TrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    learning_rate=1e-3,
    num_train_epochs=10,
    logging_steps=10,
    max_steps=10
)

peft_trainer = Trainer(
    model=peft_model,
    args=peft_training_args,
    train_dataset=tokenized_datasets["train"],
)

peft_trainer.train()

peft_model_path = "./peft-dialogue-summary-checkpoint-local"
peft_trainer.model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)

from peft import PeftConfig, PeftModel

peft_model_base = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
peft_model = PeftModel.from_pretrained(peft_model_base, "./peft-dialogue-summary-checkpoint-local",
                                      torch_dtype=torch.bfloat16, device_map='auto', is_trainable=False)
peft_model = peft_model.to(device)

# In this section, we will use RL PPO (Proximal Optimization Policy) passing the fine-tuned PEFT Model,
# PPO will be used to optimize the RL policy and reward Model

# RLHF

ppo_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(peft_model, torch_dtype=torch.bfloat16, is_trainable=True)
print(f'PPO model parameters to be updated (ValueHead + 769 params):\n{print_number_of_trainable_model_parameters(ppo_model)}\n')
print(ppo_model.v_head)
ppo_model = ppo_model.to(device)

ref_model = create_reference_model(peft_model)
print(f'Reference model parameters to be updated:\n{print_number_of_trainable_model_parameters(ref_model)}\n')
ref_model = ref_model.to(device)

# Prepare Reward Model

toxicity_model_name = 'facebook/roberta-hate-speech-dynabench-r4-target'
toxicity_tokenizer = AutoTokenizer.from_pretrained(toxicity_model_name, device_map='auto')
toxicity_model = AutoModelForSequenceClassification.from_pretrained(toxicity_model_name, device_map='auto')
print(toxicity_model.config.id2label)
toxicity_model.to(device)

non_toxic_text = 'I want to kiss you'
device = torch.device("cuda:0")
toxicity_input_ids = toxicity_tokenizer(non_toxic_text, return_tensors='pt').input_ids.to(device)
logits = toxicity_model(input_ids=toxicity_input_ids).logits
print(f'logits [non hate, hate]:{logits.tolist()[0]}')
probabilities = logits.softmax(dim=-1).tolist()[0]
print(f'probabilities [non hate, hate]:{probabilities}')

not_hate_index = 0
nothate_reward = (logits[:, not_hate_index]).tolist()
print(f'reward (high): {nothate_reward}')

toxic_text = 'you are disgusting and terrible and I damn hate you'
toxicity_input_ids = toxicity_tokenizer(toxic_text, return_tensors='pt').input_ids.to(device)
logits = toxicity_model(input_ids=toxicity_input_ids).logits
print(f'logits [non hate, hate]:{logits.tolist()[0]}')
probabilities = logits.softmax(dim=-1).tolist()[0]
print(f'probabilities [non hate, hate]:{probabilities}')

not_hate_index = 0
nothate_reward = (logits[:, not_hate_index]).tolist()
print(f'reward (high): {nothate_reward}')

# device = 0 if torch.cuda.is_available() else 'cpu'

sentiment_pipe = pipeline('sentiment-analysis', model=toxicity_model_name, device=device)
reward_logits_kwargs = {
    'top_k': None,
    'function_to_apply': 'none',
    'batch_size': 16,
}

reward_probabilities_kwargs = {
    'top_k': None,
    'function_to_apply': 'softmax',
    'batch_size': 16,
}

print('Reward model output for non-toxic text:')
print(sentiment_pipe(non_toxic_text, **reward_logits_kwargs))
print(sentiment_pipe(non_toxic_text, **reward_probabilities_kwargs))
print('\nReward model output for toxic text')
print(sentiment_pipe(toxic_text, **reward_logits_kwargs))
print(sentiment_pipe(toxic_text, **reward_probabilities_kwargs))

toxicity_evaluator = evaluate.load('toxicity',
                                 toxicity_model_name,
                                 module_type='measurement',
                                 toxic_label='hate')

toxicity_score = toxicity_evaluator.compute(predictions=[non_toxic_text])
print('Toxicity score for non-toxic text')
print(toxicity_score['toxicity'])
toxicity_score = toxicity_evaluator.compute(predictions=[toxic_text])

print('Toxicity score for toxic text')
print(toxicity_score['toxicity'])
toxicity_score = toxicity_evaluator.compute(predictions=[non_toxic_text])

def evaluate_toxicity(model, toxcity_evaluator, tokenizer, dataset, num_samples):
    max_new_tokens = 100
    toxicities = []
    input_texts = []

    for i, sample in tqdm(enumerate(dataset)):
        input_text = sample['query']

        if i > num_samples:
            break

        input_ids = tokenizer(input_text, return_tensors='pt', padding=True).input_ids.to(device)
        generation_config = GenerationConfig(max_new_tokens=max_new_tokens, top_k=0.0, top_p=1.0, do_sample=True)
        response_token_ids = model.generate(input_ids=input_ids, generation_config=generation_config)
        generated_text = tokenizer.decode(response_token_ids[0], skip_special_tokens=True)
        toxicity_score = toxicity_evaluator.compute(predictions=[(input_text + " " + generated_text)])
        toxicities.extend(toxicity_score['toxicity'])

    mean = np.mean(toxicities)
    std = np.std(toxicities)
    return mean, std

tokenizer = AutoTokenizer.from_pretrained(model_name, device_map='auto')
output = evaluate_toxicity(model=ref_model,
                           toxcity_evaluator=toxicity_evaluator,
                           tokenizer=tokenizer,
                           dataset=dataset['test'],
                           num_samples=20)
print(output)

# Initialize PPO Trainer

learning_rate = 1.41e-5
max_ppo_epochs = 1
mini_batch_size = 4
batch_size = 16

config = PPOConfig(
    model_name=model_name,
    learning_rate=learning_rate,
    ppo_epochs=max_ppo_epochs,
    mini_batch_size=mini_batch_size,
    batch_size=batch_size
)

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

ppo_trainer = PPOTrainer(config=config,
                        model=ppo_model,
                        ref_model=ref_model,
                        tokenizer=tokenizer,
                        dataset=dataset['train'],
                        data_collator=collator)

# Fine-Tune The Model

output_min_length = 100
output_max_length = 400
output_length_sampler = LengthSampler(output_min_length, output_max_length)

generation_kwargs = {
    "min_length": 5,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True
}

reward_kwargs = {
    "top_k": None,
    "function_to_apply": "none",
    "batch_size": 16
}

max_ppo_steps = 10

for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    if step >= max_ppo_steps:
        break

    prompt_tensors = batch["input_ids"]
    summary_tensors = []

    for prompt_tensor in prompt_tensors:
        max_new_tokens = output_length_sampler()
        generation_kwargs['max_new_tokens'] = max_new_tokens
        summary = ppo_trainer.generate(prompt_tensor, **generation_kwargs)
        summary_tensors.append(summary.squeeze()[-max_new_tokens:])

    batch['response'] = [tokenizer.decode(r.squeeze()) for r in summary_tensors]
    query_response_pairs = [q + r for q, r in zip(batch['query'], batch['response'])]
    rewards = sentiment_pipe(query_response_pairs, **reward_kwargs)

    reward_tensors = [torch.tensor(reward[not_hate_index]['score']) for reward in rewards]

    stats = ppo_trainer.step(prompt_tensors, summary_tensors, reward_tensors)
    ppo_trainer.log_stats(stats, batch, reward_tensors)

# Evaluation

# Quantitative Evaluation

# Reference Model
output_ref = evaluate_toxicity(model=ref_model,
                               toxcity_evaluator=toxicity_evaluator,
                               tokenizer=tokenizer,
                               dataset=dataset['test'],
                               num_samples=20)
print(output_ref)

output_ppo = evaluate_toxicity(model=ppo_model,
                               toxcity_evaluator=toxicity_evaluator,
                               tokenizer=tokenizer,
                               dataset=dataset['test'],
                               num_samples=20)
print(output_ppo)

batch_size = 20
compare_results = {}
df_batch = dataset['test'][0:batch_size]

compare_results['query'] = df_batch['query']
prompt_tensors = df_batch['input_ids']

summary_tensors_ref = []
summary_tensors = []
for i in tqdm(range(batch_size)):
    gen_len = output_length_sampler()
    generation_kwargs['max_new_tokens'] = gen_len

    summary = ppo_model.generate(
        input_ids=torch.as_tensor(prompt_tensors[i]).unsqueeze(dim=0).to(device),
        **generation_kwargs
    ).squeeze()[-gen_len:]

    summary_tensors_ref.append(summary)

    summary = ref_model.generate(
        input_ids=torch.as_tensor(prompt_tensors[i]).unsqueeze(dim=0).to(device),
        **generation_kwargs
    ).squeeze()[-gen_len:]

    summary_tensors.append(summary)

compare_results['response_before'] = [tokenizer.decode(summary_tensors_ref[i]) for i in range(batch_size)]
compare_results['response_after'] = [tokenizer.decode(summary_tensors[i]) for i in range(batch_size)]

texts_before = [d + s for d, s in zip(compare_results['query'], compare_results['response_before'])]
rewards_before = sentiment_pipe(texts_before, **reward_kwargs)
compare_results['reward_before'] = [reward[not_hate_index]['score'] for reward in rewards_before]

texts_after = [d + s for d, s in zip(compare_results['query'], compare_results['response_after'])]
rewards_after = sentiment_pipe(texts_after, **reward_kwargs)
compare_results['reward_after'] = [reward[not_hate_index]['score'] for reward in rewards_after]

pd.set_option('display.max_colwidth', 500)
df_compare_results = pd.DataFrame(compare_results)
df_compare_results['reward_diff'] = df_compare_results['reward_after'] - df_compare_results['reward_before']
df_compare_results_sorted = df_compare_results.sort_values(by=['reward_diff'], ascending=False).reset_index(drop=True)

# Generate an overall evaluation of the two models (before and after to compare how toxicity is changed through the RLHF process)
mean_before = df_compare_results_sorted['reward_before'].mean()
std_before = df_compare_results_sorted['reward_before'].std()
mean_after = df_compare_results_sorted['reward_after'].mean()
std_after = df_compare_results_sorted['reward_after'].std()

# Display the results
print("Mean of 'reward_before':", mean_before)
print("Standard Deviation of 'reward_before':", std_before)
print("Mean of 'reward_after':", mean_after)
print("Standard Deviation of 'reward_after':", std_after)
