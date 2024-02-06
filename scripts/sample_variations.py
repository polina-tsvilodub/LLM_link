from tqdm import tqdm
import openai
import torch
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from utils import (
    load_model, 
    compute_mi, 
    get_mlm_pl,
    mask_and_sum,
    get_mlm_token_pl,
)
import argparse
import random
import string
from datetime import datetime
import time

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device = {DEVICE}")

import os

def sample_continuation(
        model, 
        tokenizer, 
        prompt, 
        temperature=0.2,
        max_new_tokens=64,    
    ):
    """
    Helper for creating model-internal (i.e., sampled by model itself) variations of the materials.
    """
    input_ids = tokenizer(
        prompt,
        return_tensors="pt"
    )
    outputs = model.generate(
        **input_ids,
        temperature=temperature,
        do_sample=True,
        max_new_tokens=max_new_tokens,
    )
    # decode completion only
    prediction = tokenizer.decode(
        outputs[0, input_ids.input_ids.shape[-1]:],
        skip_special_tokens=True
    )

    return prediction

def main(
        phenomenon,
        temperature=0.1,
        model_name="gpt-3.5-turbo-instruct",
        option_numbering=None,
        use_labels_only=False,
        question="",
        n_seeds=1,
):
    # construct path to vignettes and instructions
    file_path = "../data/data_hu_" + phenomenon + ".csv"
    instructions_path = "../prompt/prompts_consistency/" + phenomenon + ".txt"
    # initialize path for dumping output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_name = file_path.split("/")[-1].replace(".csv", "")
    model_name_out = model_name.split("/")[-1]
    # Load model and tokenizer
    tokenizer, model = load_model(model_name)

    # Define seeds
    seeds = range(n_seeds)

    
    # Load data set
    scenarios = pd.read_csv(file_path).dropna()
    print(scenarios.head())
    # retrieve option names for shuffling and then mapping results back to the type of response
    option_names = list(scenarios.loc[:, 'target':].columns)
    # Iterate over seeds
    for seed in seeds:

        # final results output file

        out_file = f"../results/paraphrases/{model_name_out}_{out_name}_FC_labels_seed{seed}_{timestamp}.csv"
        if not os.path.exists(f"../results/paraphrases"):
            os.makedirs(f"../results/paraphrases")

        # Iterate over rows in prompt csv 
        for i, row in tqdm(scenarios.iterrows()):
            paraphrases = []
            prompts = []
            # load instructions
            with open(instructions_path, "r") as f:
                instructions = f.read()
            
            if phenomenon == "coherence":
                prompt = row.prompt.split(".")[-1] + "."
                prompts.append(prompt)

            elif phenomenon == "deceits":
                prompts = [
                    row[r] for r in ["target", "incorrect_literal", "incorrect_lexical_overlap", "incorrect_social_convention"]
                ]
            elif phenomenon == "metaphor":
                prompts = [
                    row[r] for r in ["target", "competitor", "distractor_plausibleliteral", "distractor_literal", "distractor_nonsequitut"]
                ]
            elif phenomenon == "maxims":
                prompts = [
                    row[r] for r in ["target", "incorrect_literal", "incorrect_nonliteral", "incorrect_associate"]
                ]
            elif phenomenon == "humour":
                prompts = [
                    row[r] for r in ["target", "incorrect_literal", "incorrect_nonliteral", "incorrect_associate"]
                ]
            elif phenomenon == "irony":
                prompts = [
                    row[r] for r in ["target", "competitor", "distractor_associate", "distractor_nonsequitur"]
                ]
            else:
                prompts = [
                    row[r] for r in ["target", "competitor", "distractor_lexical_overlap", "distractor_associative"]
                ]
            
            for p in prompts:
                prompt = instructions.format(
                    trigger_sentence = p
                )
                
                print("---- formatted prompt ---- ", prompt)
                
                continuation = sample_continuation(
                    model, 
                    tokenizer,
                    prompt,
                )
               
                #####################################

                # Record the retrieved log probs
                # TODO record other configs
                # initialize results df, so that we can write result in long format
                results_df = pd.DataFrame({
                    "model_name": [model_name] ,
                    "temperature": [temperature] ,
                    "seed": [seed] ,
                    "item_id": [row.item_number]  ,
                    "phenomenon": [phenomenon] ,
                    "prompt": [prompt] ,
                    "paraphrase": [continuation],
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
        default="gpt-3.5-turbo-instruct",
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