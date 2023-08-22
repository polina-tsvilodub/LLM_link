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

    #input_text = "translate English to German: How old are you?"
    #input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    #outputs = model.generate(input_ids)

    return tokenizer, model

def get_completion(prompt, model, tokenizer, answer_choices=["very unlikely", "unlikely", "at chance", "likely", "very likely"], **kwargs):
    
    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # Tokenize the answer choices
    answer_token_ids = [tokenizer(answer, add_special_tokens=False).input_ids for answer in answer_choices]

    # Generate output from the model
    outputs = model.generate(
        **input_ids,
        max_new_tokens=2,
        output_scores=True,
        num_return_sequences=1,
        return_dict_in_generate=True
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

    # Calculate the sum of logits for the generated sequence
    #sum_logits = logits.sum().item()

    #return generated_text, sum_logits

    # Retrieve logits for each answer choice and aggregate them (e.g., by summing)
    answer_logits = [sum(logits[id].item() for id in token_ids) for token_ids in answer_token_ids]

    # Find the answer choice with the highest logit
    #generated_answer = answer_choices[np.argmax(answer_logits)]

    # Convert logits to probabilities
    probs = softmax(answer_logits)
    probs_dict = {answer: prob for answer, prob in zip(answer_choices, probs)}

    return generated_text, probs_dict


def main():
    # load data set
    scenarios = pd.read_csv("prompts_modified/Maxims_prompts_Rating.csv").dropna()
    print(scenarios.head())

    # load model and tokenizer
    tokenizer, model = load_model()

    # define answer choices
    answer_choices=["very unlikely","unlikely","at chance","likely","very likely"]

    for i, row in tqdm(scenarios.iterrows()):
        prompt = row.prompt
        generated_answer, probs = get_completion(
            prompt, model, tokenizer, answer_choices=answer_choices
        )
        # Evaluate generated text.
        scenarios.loc[i, "generation"] = generated_answer.strip()
        scenarios.loc[i, "generation_isvalid"] = (generated_answer.strip() in answer_choices)
        # Record probability distribution over valid answers.
        scenarios.loc[i, "distribution"] = str(probs)
        #scenarios.loc[i, "prob_true_answer"] = probs[str(row.randomized_true_answer)]
        # Take model "answer" to be argmax of the distribution.
        #sorted_probs = [probs[answer] for answer in answer_choices]
       # chosen_answer = str(np.argmax(sorted_probs) + 1)
        #scenarios.loc[i, "answer"] = chosen_answer
        #scenarios.loc[i, "correct"] = (chosen_answer == str(row.randomized_true_answer))
        #scenarios.loc[i, "answer_label_complex"] = eval(row.randomized_labels_complex)[int(chosen_answer)-1]

    scenarios.to_csv(f"prompts_modified/Maxims_results_Rating.csv", index=False)

if __name__ == "__main__":
    main()