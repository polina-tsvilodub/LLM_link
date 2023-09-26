from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from utils import load_model
from score_completions import retrieve_log_probs
import openai
import random
from dotenv import load_dotenv
import argparse
from datetime import datetime
import time

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device = {DEVICE}")

import os

# Some helper functions
def softmax(x):
    return np.exp(x)/sum(np.exp(x))


def aggregare_ratings(
    option_conditional_log_probs,
    option_names,
):
    """
    Helper for aggregating the conditional log probs of the options
    """
    # for each rating, collect the "word level" log probs
    # (inner lists)
    rating_log_probs = []
    for option in option_conditional_log_probs:
        rating_log_probs.append(
            [sum(rating) for rating in option]
        )
    
    options_probs = []
    # renormalize the conditional log probs for each option
    # (top level lists)
    for option in rating_log_probs:
        options_probs.append(
            softmax(option)
        )
    # assign numeric weights to each option
    # e.g., assume they go from 1 to 5
    options_weights = list(range(len(rating_log_probs[0])))
    # multiply the weights with the probs
    weighted_options = []
    for p in options_probs:
        weighted_options.append(
            np.sum([weight * prob for weight, prob in zip(options_weights, p)])
        )
        
    # choose the option with the highest weighted prob
    # with random tie breaking
    if weighted_options.count(max(weighted_options)) > 1:
        # choose randomly amongst the max prob options
        max_options = [option_names[i] for i, j in enumerate(weighted_options) if j == max(weighted_options)]
        chosen_option = random.choice(max_options)
    else:
        chosen_option = option_names[
            weighted_options.index(max(weighted_options))
        ]
    return weighted_options, chosen_option


def main(
    file_path,
    temperature=0.1,
    model_name="gpt-3.5-turbo-instruct",
    option_numbering=None,
    use_option_numbering_only=False,
    instructions_path=None,
    question="",
    n_seeds=5,
):
    # initialize path for dumping output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_name = file_path.split("/")[-1].replace(".csv", "")
    # Load model and tokenizer
    tokenizer, model = load_model(model_name)

    # TODO: Define scales
    scales = ["plausible", "appropriate", "possible", "likely"]

    # Generate five random seeds for repeated sampling
    seeds = range(n_seeds)
    # load data
    scenarios = pd.read_csv(file_path).dropna()
    print(scenarios.head())

    # retrieve option names for shuffling and then mapping results back to the type of response
    option_names = list(scenarios.loc[:, 'target':].columns)

    # Iterate over seeds
    for seed in seeds:
        # Reseed the singleton RandomState instance.
        np.random.seed(seed)
        random.seed(seed)

        # Iterate over scales
        for scale in scales:
            
            # Define answer choices given scales
            if scale == "plausible":
                answer_choices=["very implausible", "implausible", "neutral", "plausible", "very plausible"]
            elif scale == "appropriate":
                answer_choices=["very inappropriate", "inappropriate", "neutral", "appropriate", "very appropriate"]
            elif scale == "possible":
                answer_choices=["very impossible", "impossible", "neutral", "possible", "very possible"]
            elif scale == "likely":
                answer_choices=["very unlikely", "unlikely", "neutral", "likely", "very likely"]

            # final results output file
            out_file = f"../results/rating/{out_name}_rating_scale-{scale}_seed{seed}_{timestamp}.csv"
        
            # Iterate over rows in prompt csv 
            for i, row in tqdm(scenarios.iterrows()):
                # load instructions
                with open(instructions_path, "r") as f:
                    instructions = f.read()
                # construct task question
                question = question.format(row.speaker)
                # Get prompt and generate answer
                prompt = instructions + row.prompt + question + "\nHow would you rate the following answer: "
                # retrieve the actual options
                options = list(row.loc['target':])
                # iterate over the options
                option_conditional_log_probs = []
                options_log_probs = []
                for o in options:
                    prompt_with_option = prompt + o + "\nChoose one of the following response options: " + ", ".join(answer_choices) + "\n" 
                    cond_rating_probs, rating_probs = retrieve_log_probs(
                        prompt_with_option, 
                        options=answer_choices,
                        model_name=model_name,
                        model=model, 
                        tokenizer=tokenizer, 
                        score_prior=False,
                        use_option_numbering_only=False,
                        temperature=0.1, 
                    )
                    # lists of nested lists (each top level list corresponds to one option)
                    option_conditional_log_probs.append(cond_rating_probs)
                    options_log_probs.append(rating_probs)

                    # sleep
                    time.sleep(10)
                
                # postproess the probs with aggregate_ratings
                weighted_options, chosen_option = aggregare_ratings(
                    option_conditional_log_probs,
                    option_names,
                )

                results_df = pd.DataFrame({
                    "model_name": [model_name] * len(options),
                    "temperature": [temperature] * len(options),
                    "seed": [seed] * len(options),
                    "item_id": [row.item_number]  * len(options),
                    "prompt": [prompt] * len(options),
                    "question": [question] * len(options),
                    "options": options,
                    "option_names": option_names,
                    "scale": [scale] * len(options), 
                    "weighted_options": weighted_options,
                    "chosen_option": [chosen_option] * len(options),
                    "token_cond_log_probs": option_conditional_log_probs,
                    "prior_token_log_probs": options_log_probs,
                })
                print(results_df)

                # Save results
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
        "--instructions_path",
        type=str,
        help="Path to the text file containing instructions for the task",
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
        default=5,
        help="Number of seeds to run the experiment for",
    )

    args = parser.parse_args()

    main(
        file_path=args.file_path,
        temperature=args.temperature,
        model_name=args.model_name,
        instructions_path=args.instructions_path,
        question=args.question,
        n_seeds=args.n_seeds,
    )