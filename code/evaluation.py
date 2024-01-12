# Module 4: evaluation.py

import evaluate

def evaluate_toxicity(model, toxicity_evaluator, tokenizer, dataset, num_samples):
    import numpy as np
    from tqdm import tqdm
    from transformers import GenerationConfig

    max_new_tokens = 100
    toxicities = []

    for i, sample in tqdm(enumerate(dataset)):
        input_text = sample['query']

        if i > num_samples:
            break

        input_ids = tokenizer(input_text, return_tensors='pt', padding=True).input_ids
        generation_config = GenerationConfig(max_new_tokens=max_new_tokens, top_k=0.0, top_p=1.0, do_sample=True)
        response_token_ids = model.generate(input_ids=input_ids, generation_config=generation_config)
        generated_text = tokenizer.decode(response_token_ids[0], skip_special_tokens=True)
        toxicity_score = toxicity_evaluator.compute(predictions=[input_text + " " + generated_text])
        toxicities.extend(toxicity_score['toxicity'])

    mean_toxicity = np.mean(toxicities)
    std_toxicity = np.std(toxicities)
    return mean_toxicity, std_toxicity

def generate_text(model, input_text, tokenizer, max_new_tokens=100):
    from transformers import GenerationConfig

    input_ids = tokenizer(input_text, return_tensors='pt', padding=True).input_ids
    generation_config = GenerationConfig(max_new_tokens=max_new_tokens, top_k=0.0, top_p=1.0, do_sample=True)
    response_token_ids = model.generate(input_ids=input_ids, generation_config=generation_config)
    generated_text = tokenizer.decode(response_token_ids[0], skip_special_tokens=True)

    return generated_text
