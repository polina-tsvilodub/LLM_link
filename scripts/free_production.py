from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import os
import openai
from dotenv import load_dotenv
from utils import load_model
import argparse
from datetime import datetime
import random
import time

if torch.cuda.is_available():
    DEVICE = "cuda:0" 
elif torch.backends.mps.is_available(): 
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"Device = {DEVICE}")


def get_completion(
        prompt, 
        max_new_tokens,
        model_name=None,
        model=None,
        tokenizer=None,  
        phenomenon=None,
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
    # check if decoding scheme was specified, othrwise default to argmax / greedy
    decoding = kwargs["decoding_scheme"] if kwargs["decoding_scheme"] else "greedy"
    # specific decding schemes are only supported for HF models
    if ("gpt" in model_name) and (decoding != "greedy"):
        raise ValueError(f"Decoding scheme {decoding} is not supported for GPT models.")
    
    # Tokenize the prompt
    if ("google/flan-t5" in model_name) or ("meta" in model_name):
        model = model #.to(DEVICE)
        input_ids = tokenizer(prompt, return_tensors="pt") #.to(DEVICE)
        # Generate output from the model with a maximum length of 20 tokens
        # with the respective decoding scheme
        if decoding == "greedy":
            outputs = model.generate(
                **input_ids,
                max_new_tokens = max_new_tokens, # Initial length + 20 more tokens
                output_scores=True,
                num_return_sequences=1,
                return_dict_in_generate=True,
                temperature=kwargs["temperature"],
            )
        elif decoding == "beam_search":
            outputs = model.generate(
                **input_ids,
                max_new_tokens = max_new_tokens, # Initial length + 20 more tokens
                output_scores=True,
                num_return_sequences=1,
                return_dict_in_generate=True,
                temperature=kwargs["temperature"],
                num_beams=5,
            )
        elif decoding == "softmax":
            model.generate(
                **input_ids,
                max_new_tokens = max_new_tokens, # Initial length + 20 more tokens
                output_scores=True,
                num_return_sequences=1,
                return_dict_in_generate=True,
                temperature=kwargs["temperature"],
                do_sample=True,
            )
        else:
            raise ValueError(f"Decoding scheme {decoding} is not supported.")

        # Convert the generated token IDs back to text
        answerTokens = tokenizer.decode(
            outputs.sequences[0], 
            skip_special_tokens=True
        )

    elif ("gpt-3.5" in model_name) or ("davinci" in model_name):
        # alternative with the completions endpoint
        # TODO check if i need to update the openai package version
        outputs = openai.Completion.create(
            model      = model_name, 
            prompt   = prompt,
            max_tokens  = max_new_tokens, # TODO parametrize
            temperature = kwargs["temperature"], 
            logprobs    = 0,
            echo = True,
        )

        # get only the new tokens
        text_offsets = outputs.choices[0]['logprobs']['text_offset']
        cutIndex = text_offsets.index(max(i for i in text_offsets if i <= len(prompt)))
        endIndex = outputs.usage.total_tokens
        answerTokens = outputs.choices[0]["logprobs"]["tokens"][cutIndex:endIndex]
        print("Answer tokens:", answerTokens)
    else:
        raise ValueError(f"Model {model_name} is not supported as a backbone.")
    
    answer_sentence = "".join(answerTokens)

    return answer_sentence



def main(
    phenomenon,
    temperatures="0.1",
    model_name="gpt-3.5-turbo-instruct",
    max_new_tokens=20,
    question="",
    decoding_scheme="greedy", # parametrize decoding schemes if necessary 
    n_seeds=1,
):
    # construct path to vignettes and instructions
    file_path = "../data/data_hu_" + phenomenon + ".csv"
    instructions_path = "../prompt/prompts/" + phenomenon + "_instructions_FC.txt"
    # initialize path for dumping output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_name = file_path.split("/")[-1].replace(".csv", "")
    
    # Load model and tokenizer
    tokenizer, model = load_model(model_name)

    # Define temperatures to iterate over
    temperatures_list = [
        float(t) for t 
        in temperatures.split(",")
    ]

    # TODO
    # Define seeds
    seeds = range(n_seeds)

    # Iterate over seeds
    for seed in seeds:
        # Reseed the singleton RandomState instance.
        np.random.seed(seed)
        random.seed(seed)
        # Iterate over anwsers
        for temperature in temperatures_list:
            # final results file
            out_file = f"../results/free/{out_name}_free_temp{temperature}_seed{seed}_{timestamp}.csv"

            # Load data set
            scenarios = pd.read_csv(file_path).dropna()
            print(scenarios.head())
            

            # Iterate over rows in prompt csv 
            for i, row in tqdm(scenarios.iterrows()):
                # load instructions
                with open(instructions_path, "r") as f:
                    instructions = f.read()
                # Get prompt and generate answer
                prompt = instructions + "\n\n" + row.prompt
                # construct task question
                try:
                    question = question.format(row.speaker) #f"Why has {row.speaker} responded like this? \nYour answer:\n"
                except:
                    pass
                prompt = prompt + "\n\n" + question + "\nYour answer:\n"
                
                # retrieve response
                response = get_completion(
                    prompt, 
                    max_new_tokens,
                    model_name,
                    model,
                    tokenizer, 
                    temperature=temperature,
                    decoding_scheme=decoding_scheme,
                )

                print("generate response: ", response)
                # results df
                # TODO add other configs for saving 
                results_df = pd.DataFrame({
                    "model_name": model_name,
                    "temperature": temperature,
                    "phenomenon": phenomenon,
                    "decoding_scheme": decoding_scheme,
                    "question": question,
                    "seed": seed,
                    "item_id": row.item_number,
                    "prompt": prompt,
                    "response": response,
                }, index=[0])
                print(results_df)

                # Save results to csv continuously
                
                # write header depending on whether the file already exists
                if os.path.exists(out_file):
                    results_df.to_csv(out_file, 
                                    index=False,
                                    mode="a",
                                    header=False,
                                    )
                else:
                    results_df.to_csv(out_file, 
                                    index=False,
                                    mode="a",
                                    header=True,
                                    )
                
                time.sleep(5)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--temperatures",
        type=str,
        default="0.1",
        help="Temperature for pragmatic speaker. Can be a list of temperatures to iterate over separated by commas.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-3.5-turbo-instruct",
        help="Name of backbone model (OpenAI or HuggingFace)",
    )

    parser.add_argument(
        "--file_path",
        type=str,
        help="Path to the csv file containing the vignettes",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Number of new tokens to predict for the answer explanation.",
    )

    parser.add_argument(
        "--question",
        type=str,
        default="",
        help="Task question to append to the context",
    )

    parser.add_argument(
        "--decoding_scheme",
        type=str,
        choices=["greedy", "beam_search", "softmax"],
        default="greedy",
        help="Task question to append to the context",
    )
    
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=1,
        help="Number of seeds to run the experiment for",
    )
    parser.add_argument(
        "--phenomenon",
        type=str,
        choices=["coherence", "deceits", "humour", "indirect_speech", "irony", "maxims", "metaphor"],
        help="Phenomenon for which the computations should be run",
    )

    args = parser.parse_args()

    main(
        phenomenon=args.phenomenon,
        temperatures=args.temperatures,
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
        question=args.question,
        decoding_scheme=args.decoding_scheme,
        n_seeds=args.n_seeds,
    )