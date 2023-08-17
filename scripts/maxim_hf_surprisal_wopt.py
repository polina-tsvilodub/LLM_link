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

    input_text = "translate English to German: How old are you?"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    outputs = model.generate(input_ids)
    print(tokenizer.decode(outputs[0]))

def get_completion(prompt, model, tokenizer, answer_choices=[1,2,3,4,5], **kwargs):

    # Encode prompts and answers to tokens
    input_ids = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    answer_token_ids = tokenizer(
        [str(answer) for answer in answer_choices], 
        return_tensors="pt", add_special_tokens=False
    ).input_ids.squeeze().tolist()

    outputs = model.generate(
        **input_ids, 
        max_new_tokens=1,
        output_scores=True,
        num_return_sequences=1,
        return_dict_in_generate=True
    )
    if isinstance(outputs.scores, tuple):
        logits = outputs.scores[0][0]
    else:
        logits = outputs.scores

    answer_logits = [logits[answer_id].item() for answer_id in answer_token_ids]
    generated_answer = str(answer_choices[np.argmax(answer_logits)])
    probs = softmax(answer_logits)
    probs = {answer: probs[int(answer)-1] for answer in answer_choices}
    return generated_answer, probs

def main():
    # load data set
    phenomen = "Maxims"
    suffix = None
    seed = 0
    num_examples = 0
    df = pd.read_csv(f"prompts/{phenomen}_prompts_seed{seed}_examples{num_examples}.csv").dropna()
    print(df.head())

if __name__ == "__main__":
    #get_completion("Between 1 and 5, which number is even?", model, tokenizer)
    main()