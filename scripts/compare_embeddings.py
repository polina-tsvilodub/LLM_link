from tqdm import tqdm
import openai
import torch
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from utils import load_model, compute_mi
import argparse
import random
import string
from datetime import datetime
from openai.embeddings_utils import cosine_similarity
import time

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device = {DEVICE}")

import os

# Some helper functions
def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def compute_embedding_similarity(
        prompt_randomized, 
        options,
        model_name,
        model=None, 
        tokenizer=None,
        temperature=None,
        use_labels_only=None,
        option_numbering=None,
        **kwargs,
):
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    interpretation_embedding = openai.Embedding.create(
        input=prompt_randomized, 
        model=model_name
    )["data"][0]["embedding"]

    options_embeddings = [
        openai.Embedding.create(
            input=o, 
            model=model_name
        )["data"][0]["embedding"] for o
        in options
    ]
    
    cosine_sims = [
        cosine_similarity(interpretation_embedding, oe) for oe 
        in options_embeddings
    ]
    choice_probs = softmax(cosine_sims)
    best_choice = choice_probs.tolist().index(max(choice_probs))
    chosen_option = kwargs["option_names"][best_choice]

    return cosine_sims, choice_probs, chosen_option

def main(
        phenomenon,
        temperature=0.1,
        model_name="text-embedding-ada-002",
        option_numbering=None,
        use_labels_only=False,
        question="",
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

    # Define seeds
    seeds = range(n_seeds)

    # set option numbering
    if option_numbering is None:
        option_numbering = list(string.ascii_uppercase[:len(options)])
    else:
        option_numbering = option_numbering.split(",")
    # string about options to append to the instructions
    option_instructions = ", ".join(option_numbering)
     # Load data set
    scenarios = pd.read_csv(file_path).dropna()
    print(scenarios.head())
    # retrieve option names for shuffling and then mapping results back to the type of response
    option_names = list(scenarios.loc[:, 'target':].columns)
    # Iterate over seeds
    for seed in seeds:
        # Reseed the singleton RandomState instance.
        np.random.seed(seed)
        random.seed(seed)

        # final results output file
        if use_labels_only:
            out_file = f"../results/embedding_similarity/{out_name}_embedding_labels_seed{seed}_{timestamp}.csv"
        else:
            out_file = f"../results/embedding_similarity/{out_name}_embedding_optionString_seed{seed}_{timestamp}.csv"
        
        # Iterate over rows in prompt csv 
        for i, row in tqdm(scenarios.iterrows()):
            # load instructions
            with open(instructions_path, "r") as f:
                instructions = f.read()
            # Get prompt and generate answer
            prompt = instructions + "\n\n" + row.prompt
            # construct task question
            try:
                question = question.format(row.speaker)
            except:
                pass
            # the df should have the target option and then other options as last columns
            options = list(row.loc['target':])
            # shuffle options
            shuffled_options = list(zip(
                    option_names,
                    options,
                ))
            random.shuffle(shuffled_options)
            # unpack
            shuffled_option_names, shuffled_options = zip(*shuffled_options)
            shuffled_option_names = list(shuffled_option_names)
            shuffled_options = list(shuffled_options)
            # add the list of options in a randomized seed dependent order
            if use_labels_only:
                prompt_randomized = prompt + question + "\n Choose one of the following options and return the label of that option.\n".join([". ".join(o) for o in zip(option_numbering, shuffled_options)]) + "\nYour answer:\n"
                options = option_numbering
            else:
                prompt_randomized = prompt + question + "\nYour answer:\n"

            print("---- formatted prompt ---- ", prompt_randomized)
            
            cosine_sims, choice_probs_round, chosen_option = compute_embedding_similarity(
                prompt_randomized, 
                shuffled_options,
                model_name,
                model, 
                tokenizer,
                temperature=temperature,
                use_labels_only=use_labels_only,
                option_numbering=option_numbering,
                option_names=shuffled_option_names,
            )
        
            #####################################

            # Record the retrieved log probs
            # initialize results df, so that we can write result in long format
            results_df = pd.DataFrame({
                "model_name": [model_name] * len(options),
                "seed": [seed] * len(options),
                "item_id": [row.item_number]  * len(options),
                "phenomenon": [phenomenon] * len(options),
                "prompt": [prompt_randomized] * len(options),
                "question": [question] * len(options),
                "options": options,
                "option_names": option_names,
                "shuffled_options": shuffled_options, # record randomization
                "shuffled_option_names": shuffled_option_names,
                "option_numbering": option_numbering,
                "cosine_sims": cosine_sims,
                "choice_probs_rounded": choice_probs_round,
                "chosen_option": [chosen_option] * len(options),
            })
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
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for pragmatic speaker",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="text-embedding-ada-002",
        help="Name of backbone model (OpenAI or HuggingFace)",
    )

    parser.add_argument(
        "--file_path",
        type=str,
        help="Path to the csv file containing the vignettes",
    )

    parser.add_argument(
        "--option_numbering",
        type=str,
        default="A,B,C,D",
        help="Option labels to prepend to option sentences",
    )
    parser.add_argument(
        "--use_labels_only",
        type=bool,
        default=False,
        help="Whether to use only labels for options as options to be scored, with options in context",
    )

    parser.add_argument(
        "--question",
        type=str,
        default="",
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
        temperature=args.temperature,
        model_name=args.model_name,
        option_numbering=args.option_numbering,
        use_labels_only=args.use_labels_only,
        question=args.question,
        n_seeds=args.n_seeds,
    )