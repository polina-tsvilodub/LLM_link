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

def get_completion(prompt, model, tokenizer, **kwargs):
    
    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # Generate output from the model with a maximum length of 20 tokens
    outputs = model.generate(
        **input_ids,
        max_new_tokens =100, # Initial length + 20 more tokens
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
    
    # Convert the generated token IDs back to text
    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

    # Calculate the sum of logits for the generated sequence
    sum_logits = logits.sum().item()

    return generated_text, sum_logits



def main():
    # load data set
    scenarios = pd.read_csv("prompts_modified/Maxims_prompts_Free.csv").dropna()
    print(scenarios.head())

    # load model and tokenizer
    tokenizer, model = load_model()

    for i, row in tqdm(scenarios.iterrows()):
        prompt = row.prompt
        generated_text, sum_logits = get_completion(prompt, model, tokenizer)

        # Record the generated text and its sum of logits
        scenarios.loc[i, "generation"] = generated_text.strip()
        scenarios.loc[i, "sum_logits"] = sum_logits

        # The following lines are related to predefined answer choices and may no longer apply.
        # You can keep, modify, or remove them based on your needs.

        # scenarios.loc[i, "generation_isvalid"] = (generated_text.strip() in answer_choices)
        # scenarios.loc[i, "distribution"] = str(probs)
        # sorted_probs = [probs[answer] for answer in answer_choices]
        # chosen_answer = str(np.argmax(sorted_probs) + 1)
        # scenarios.loc[i, "answer"] = chosen_answer
        # scenarios.loc[i, "correct"] = (chosen_answer == str(row.randomized_true_answer))
        # scenarios.loc[i, "answer_label_complex"] = eval(row.randomized_labels_complex)[int(chosen_answer)-1]

    scenarios.to_csv(f"prompts_modified/Maxims_results_Free.csv", index=False)


if __name__ == "__main__":
    main()