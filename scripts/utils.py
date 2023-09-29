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