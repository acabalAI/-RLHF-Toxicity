
# Module1: data_loader.py

from datasets import load_dataset

def load_and_preprocess_dataset(dataset_name, input_min_text_length, input_max_text_length):
    # Load the dataset using Hugging Face datasets library
    dataset = load_dataset(dataset_name)

    # Filter and preprocess the dataset based on text length
    def preprocess_example(example):
        # Define your custom preprocessing logic here
        if input_min_text_length <= len(example['dialogue']) <= input_max_text_length:
            return example

    # Apply the preprocessing function to filter the dataset
    preprocessed_dataset = dataset['train'].filter(preprocess_example)

    return preprocessed_dataset
