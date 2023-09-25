from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
import sentencepiece
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaModel, AutoTokenizer
import os
import openai
from dotenv import load_dotenv
from utils import load_model, compute_mi, softmax

if torch.cuda.is_available():
    DEVICE = "cuda:0" 
elif torch.device.is_available("mps"): 
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"Device = {DEVICE}")


def get_completion(
        model_name,
        prompt, 
        model,  
        phenomenon,
        **kwargs
    ):
    '''
    Entrypoint for getting the free production prediction 
    of the response, given the context.

    Parameters
    -----------
    model_name: str
    prompt: str
    model: T5ForConditionalGeneration
    tokenizer: T5Tokenizer
    kwargs: additional arguments to pass to model.generate

    Return
    ------ 
    generated_text: str, sum_logits: float
    '''
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    if "google/flan-t5" or "meta/" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if "t5" in model_name:
            model = T5ForConditionalGeneration.from_pretrained(model_name)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.float16)
    elif "gpt-3.5" or "embedding" in model_name:
        pass
    else:
        raise NotImplementedError(f"Model {model_name} is not supported as a backbone.")
    
    # TODO open q: do we do the diff random seeds / orders of options, too?
    
    # iterate over the items
    df = pd.read_csv("prompt/prompt_free/{phenomenon}_prompt_Free.csv")

    for r in df.iterrows():
        prompt = 'instructions' + r['context'] + r['question'] + r['options']
        # Tokenize the prompt
        if "google/flan-t5" or "meta/" in model_name:
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

            # Convert the generated token IDs back to text
            answerTokens = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

        elif "gpt-3.5" or "embedding" in model_name:
            # TODO fix the prompt and decide on the model
            # outputs = openai.ChatCompletion.create(
            #     model      = model_name, 
            #     messages   = {
            #         "role": "user",
            #         "content": prompt
            #         },
            #     max_tokens  = 20, # TODO parametrize
            #     temperature = 0.1, 
            # )
            # outputs = outputs.choices[0]['message']['content']

            # alternative with the completions endpoint
            # TODO check if i need to update the openai package version
            outputs = openai.Completion.create(
                model      = model_name, 
                prompt   = prompt,
                max_tokens  = 20, # TODO parametrize
                temperature = 0.1, 
                echo = True,
            )
            outputs = outputs.choices[0]['text']
        else:
            raise ValueError(f"Model {model_name} is not supported as a backbone.")
        
        # Retrieve logits from the output
        if isinstance(outputs.scores, tuple):
            # retrieve HF log probs
            answerTokenLogProbs = outputs.scores[0]
        else:
            # retrieve OpenAI log probs
            text_offsets = outputs.choices[0]['logprobs']['text_offset']
            cutIndex = text_offsets.index(max(i for i in text_offsets if i < len('instructions' + r['context'] + r['question']))) + 1
            endIndex = outputs.usage.total_tokens
            answerTokens = outputs.choices[0]["logprobs"]["tokens"][cutIndex:endIndex]
            answerTokenLogProbs = outputs.choices[0]["logprobs"]["token_logprobs"][cutIndex:endIndex] 
            
    
    return answerTokens, answerTokenLogProbs



def main():
    # Load data set
    # TODO pull the remaining phenomena
    scenarios = pd.read_csv("prompt/prompt_free/Maxims_prompts_Free.csv").dropna()
    print(scenarios.head())

    # TODO Define seeds
    seeds = range(5)

    # iterate over seeds
    for seed in seeds:
        # Reseed the singleton RandomState instance.
        # TODO: Figure out why random seed not working for free production
        np.random.seed(seed)

        # Iterate over scenarios
        for i, row in tqdm(scenarios.iterrows()):
            # Get the prompt and generate anwsers
            prompt = row.prompt
            generated_text, log_probs = get_completion(prompt, model, tokenizer)

            # TODO compute the various metrics on the log probs
            

            # TODO parametize the type of metrics to compute
            # Record the generated text and its sum of logits
            scenarios.loc[i, "generation"] = generated_text.strip()
            scenarios.loc[i, "log_probs"] = log_probs

        # Save the results
        scenarios.to_csv(f"results/free/Maxims_results_Free_seed{seed}.csv", index=False)


if __name__ == "__main__":
    main()