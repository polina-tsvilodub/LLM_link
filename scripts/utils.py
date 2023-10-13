# Helper functions for loading models etc
import numpy as np
import torch
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration, 
    AutoModelForCausalLM, 
    AutoTokenizer
)

def compute_mi(cond_p, p):
    return cond_p / p

# Some helper functions
def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def load_model(model_name):

    '''
    Load the tokenizer and model
    Model are accessed via the HuggingFace model hub
    For replication: You need to download and deploy the model locally.
    '''
    # return None if the model is an OpenAI model
    if ("gpt" in model_name) or ("embedding" in model_name) or ("davinci" in model_name):
        return None, None
    
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if "t5" in model_name:
            model = T5ForConditionalGeneration.from_pretrained(model_name)
        else:
            # for LLaMA 2
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.float16)

    return tokenizer, model

def get_mlm_pl(
    prompt, 
    option,
    mask_token=["<extra_id_0>", "<extra_id_1>",],
):
    """
    Helper for creating prompts and completions with masks
    for retrieving pseudo-likelihoods of encoder-decoder T5.

    Parameters
    ----------
    prompt: str
        Instructions which don't need to be masked.
    option: str
        Option which needs to be masked. If multi-word,
        pairs of I/O for iterative scoring will be created.
    Returns
    -------
    output: [tuple(str, str)]
        List of tuples of prompts and completions.
        Length is 1 if option is single word, else
        length is number of words in option.
    """
    num_option_words = len(option.split())
    words = option.split()
    # if the option is a single word, just append mask
    if num_option_words == 1:
        return [(prompt + mask_token[0], f"<extra_id_0>{option}")]
    else:
        out_pairs = []
        for i, w in enumerate(words):
            inpt_str = " ".join(words[:i]) + " <extra_id_0> " + " ".join(words[i+1:])
            inpt_str = inpt_str.strip()
            if i == 0:
                output_str = f"{w} <extra_id_0>"
            elif i == len(words) - 1:
                output_str = f"<extra_id_0> {w}"
            else:
                output_str = f"<extra_id_0> {w} <extra_id_1>"
            out_pairs.append(
                (prompt + inpt_str, output_str)
            )

    return out_pairs

def mask_and_sum(
    labels,
    sequence_plls,
    mask_token=[1, 32098, 32099],    
):
    """
    Helper for masking irrelevant tokens and
    summing pseudo-likelihoods of relevant ones.
    Returns pseudo-loglikelihood of a single masked word.

    Parameters
    ----------
    labels: torch.Tensor
        Tensor of token IDs (first dimension is batch dimension).
    sequence_plls: torch.Tensor
        Tensor of pseudo-likelihoods (first dimension is token dimension).
    mask_token: list(str)
        List of tokens to mask.
    """ 
    # Subset the labels and logprobs we care about,
    # i.e. the non-"special" tokens (e.g., "<extra_id_0>").
    mask = torch.BoolTensor([tok_id not in mask_token for tok_id in labels[0]])
    relevant_labels = labels[0][mask]
    relevant_logprobs = sequence_plls[mask]
    # Index into logprob tensor using the relevant token IDs.
    logprobs_to_sum = [
        relevant_logprobs[i][tok_id] 
        for i, tok_id in enumerate(relevant_labels)
    ]
          
    total_logprob = sum(logprobs_to_sum).item()
    return total_logprob

def get_mlm_token_pl(
        prompt,
        option,
        tokenizer,
        DEVICE,
        mask_token=["<extra_id_0><extra_id_1>"],
):
    """
    Same as helper above, but masking is done by-token.
    """
    # tokenize mask tokens and remove EOS token
    mask_token_ids = tokenizer(mask_token, return_tensors="pt").input_ids[0, :-1].tolist()
    eos_token_id = tokenizer.eos_token_id
    # prompt stays the same, remove eos token
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0, :-1]
    # tokenize option ans remove EOS token
    option_ids = tokenizer(option, return_tensors="pt").input_ids[0, :-1]
    # iteratively mask option tokens and append rest to prompt
    prompt_option_pairs = []
    for i, o in enumerate(option_ids):
        if i == 0:
            prompt_option_pairs.append(
                (
                    torch.tensor(
                        prompt_ids.tolist() +[ mask_token_ids[0]] + option_ids[i+1:].tolist() + [eos_token_id]
                    ).unsqueeze(0).to(DEVICE),
                    torch.tensor(
                        [o.item()] + [mask_token_ids[0]] + [eos_token_id]
                    ).unsqueeze(0).to(DEVICE)
                )
            )
        elif i == len(option_ids) - 1:
            prompt_option_pairs.append(
                (
                    torch.tensor(
                        prompt_ids.tolist() + option_ids[:i].tolist() + [mask_token_ids[0]]
                    ).unsqueeze(0).to(DEVICE),
                    torch.tensor(
                        [mask_token_ids[0]] + [o.item()] + [eos_token_id]
                    ).unsqueeze(0).to(DEVICE)
                )
            )
        else:
            prompt_option_pairs.append(
                (
                    torch.tensor(
                        prompt_ids.tolist() + option_ids.tolist()[:i] + [mask_token_ids[0]] + option_ids.tolist()[i+1:] + [eos_token_id]
                    ).unsqueeze(0).to(DEVICE),
                    torch.tensor(
                        [mask_token_ids[0]] + [o.item() ]+ [mask_token_ids[1]] + [eos_token_id]
                    ).unsqueeze(0).to(DEVICE)
                )
            )
    return prompt_option_pairs