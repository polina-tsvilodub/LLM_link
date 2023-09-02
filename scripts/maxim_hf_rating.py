from transformers import T5Tokenizer, T5ForConditionalGeneration
import sentencepiece
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device = {DEVICE}")

import os

# Some helper functions
def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def load_model():

    '''
    Load the tokenizer and model
    Model are accessed via the HuggingFace model hub
    For replication: You need to download and deploy the model locally.
    '''
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")

    return tokenizer, model

def get_completion(prompt, model, tokenizer, answer_choices=["very implausible", "implausible", "at chance", "plausible", "very plausible"], **kwargs):
    '''
    prompt: str
    model: T5ForConditionalGeneration
    tokenizer: T5Tokenizer
    answer_choices: list of str
    kwargs: additional arguments to pass to model.generate
    return: generated_text: str, probs_dict: dict
    '''

    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # Tokenize the answer choices
    answer_token_ids = [tokenizer(answer, add_special_tokens=False).input_ids for answer in answer_choices]

    # Generate output from the model
    outputs = model.generate(
        **input_ids,
        max_new_tokens=5,
        output_scores=True,
        num_return_sequences=1,
        return_dict_in_generate=True,
        temperature=1.0,
    )

    # Retrieve logits from the output
    if isinstance(outputs.scores, tuple):
        logits = outputs.scores[0][0]
    else:
        logits = outputs.scores

    # Retrieve logits for each answer choice and aggregate them (e.g., by summing)
    answer_logits = [sum(logits[id].item() for id in token_ids) for token_ids in answer_token_ids]

    # Convert the generated token IDs back to text
    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

    # Retrieve logits for each answer choice and aggregate them (e.g., by summing)
    answer_logits = [sum(logits[id].item() for id in token_ids) for token_ids in answer_token_ids]

    # Convert logits to probabilities
    probs = softmax(answer_logits)
    probs_dict = {answer: prob for answer, prob in zip(answer_choices, probs)}

    return generated_text, probs_dict


def main():
    # Load model and tokenizer
    tokenizer, model = load_model()

    # Define scales
    scales = ["plausible", "appropriate", "possible", "likely"]

    # Generate five random seeds for repeated sampling
    seeds = range(5)

    # Iterate over seeds
    for seed in seeds:
        # Reseed the singleton RandomState instance.
        np.random.seed(seed)
        # Iterate over scales
        for scale in scales:
            scenarios = pd.read_csv(f"prompt/prompt_rating/Maxims_prompts_Rating_{scale}.csv").dropna()
            print(scenarios.head())

            # Define answer choices given scales
            if scale == "plausible":
                answer_choices=["very implausible", "implausible", "neural", "plausible", "very plausible"]
            elif scale == "appropriate":
                answer_choices=["very inappropriate", "inappropriate", "neural", "appropriate", "very appropriate"]
            elif scale == "possible":
                answer_choices=["very impossible", "impossible", "neural", "possible", "very possible"]
            elif scale == "likely":
                answer_choices=["very unlikely", "unlikely", "neural", "likely", "very likely"]

            # Iterate over rows in prompt csv 
            for i, row in tqdm(scenarios.iterrows()):

                # Get prompt and generate answer
                prompt = row.prompt
                generated_answer, probs = get_completion(
                    prompt, model, tokenizer, answer_choices=answer_choices
                )

                # Evaluate generated text
                scenarios.loc[i, "generation"] = generated_answer.strip()
                scenarios.loc[i, "generation_isvalid"] = (generated_answer.strip() in answer_choices)

                # Record probability distribution over valid answers.
                scenarios.loc[i, "distribution"] = str(probs)

            # Save results
            scenarios.to_csv(f"results/rating/Maxims_results_Rating_{scale}_seed{seed}.csv", index=False)

if __name__ == "__main__":
    main()