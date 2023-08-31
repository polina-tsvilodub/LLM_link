from transformers import T5Tokenizer, T5ForConditionalGeneration
import sentencepiece
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device = {DEVICE}")

import os

# Some helper functions
def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def load_model():
    # Load the tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")

    return tokenizer, model

def get_completion(prompt, model, tokenizer, **kwargs):
    # TODO: add docstring

    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # Generate output from the model
    outputs = model.generate(
        **input_ids,
        max_length=50,
        output_scores=True,
        num_return_sequences=1,
        return_dict_in_generate=True
    )

    # Retrieve logits from the output
    if isinstance(outputs.scores, tuple):
        logits = outputs.scores[0][0]
    else:
        logits = outputs.scores

     # Convert the generated token IDs back to text
    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

    # Calculate the sum of logits for the generated sequence
    sum_logits = logits.sum().item()

    return generated_text, sum_logits


def main():
    # Load data set
    

    # Load model and tokenizer
    tokenizer, model = load_model()

    # Define answer types
    anwsers = ["Both", "Content", "Number"]

    # Define seeds
    seeds = range(5)

    for seed in seeds:
        # Reseed the singleton RandomState instance.
        np.random.seed(seed)
        for anwser in anwsers:
            scenarios = pd.read_csv(f"prompt/prompt_fc/Maxims_prompts_FC_{anwser}.csv").dropna()
            print(scenarios.head())

            # Iterate over rows in prompt csv 
            for i, row in tqdm(scenarios.iterrows()):
                # Get prompt and generate answer
                prompt = row.prompt

                # Evaluate generated text
                generated_text, sum_logits = get_completion(prompt, model, tokenizer)

                # Record the generated text and its sum of logits
                scenarios.loc[i, "generation"] = generated_text.strip()
                scenarios.loc[i, "sum_logits"] = sum_logits

        scenarios.to_csv(f"results/fc/Maxims_results_FC_{anwser}_seed{seed}.csv", index=False)

if __name__ == "__main__":
    main()