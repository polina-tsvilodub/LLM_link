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
    options_weights = list(range(1, len(rating_log_probs[0]) + 2))
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
    instructions_path = "../prompt/prompt_rating/" + phenomenon + "_instructions_Rating.txt"
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
                try:
                    question = question.format(row.speaker)
                except:
                    pass
                
                instructions_formatted = instructions.format(scale)
                # Get prompt and generate answer
                prompt = instructions_formatted + row.prompt + question + "\nHow would you rate the following answer: "
                prior_prompt = instructions_formatted + "\nHow would you rate the following answer: "
                # construct numbered answer choices
                answer_choices_formatted = "\n".join([
                    str(i+1) + ". " + option
                    for i, option
                    in enumerate(answer_choices)
                ])
                # retrieve the actual options
                options = list(row.loc['target':])
                # iterate over the options
                option_conditional_log_probs = []
                options_log_probs = []
                null_options_log_probs = []
                for o in options:
                    prompt_with_option = prompt + o + "\nChoose one of the following options and return the number of that option: \n" + answer_choices_formatted + "\nYour answer: " 
                    prior_prompt_with_option = prior_prompt + o + "\nChoose one of the following options and return the number of that option: \n" + answer_choices_formatted + "\nYour answer: " 
                    cond_rating_probs, rating_probs, null_rating_probs = retrieve_log_probs(
                        prompt_with_option, 
                        prior_prompt_with_option,
                        options=[str(i+1) for i in range(len(answer_choices))],
                        model_name=model_name,
                        model=model, 
                        tokenizer=tokenizer, 
                        use_labels_only=False,
                        temperature=0.1, 
                    )
                    # lists of nested lists (each top level list corresponds to one option)
                    option_conditional_log_probs.append(cond_rating_probs)
                    options_log_probs.append(rating_probs)
                    null_options_log_probs.append(null_rating_probs)
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
                    "phenomenon": [phenomenon] * len(options),
                    "prompt": [prompt] * len(options),
                    "prior_prompt": [prior_prompt] * len(options),
                    "question": [question] * len(options),
                    "options": options,
                    "option_names": option_names,
                    "scale": [scale] * len(options), 
                    "rating_options": [", ".join(answer_choices)] * len(options),
                    "weighted_options": weighted_options,
                    "chosen_option": [chosen_option] * len(options),
                    "token_cond_log_probs": option_conditional_log_probs,
                    "prior_token_log_probs": options_log_probs,
                    "null_prior_token_log_probs": null_options_log_probs,
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
        default="gpt-3.5-turbo-instruct",
        help="Name of backbone model (OpenAI or HuggingFace)",
    )

    parser.add_argument(
        "--file_path",
        type=str,
        help="Path to the csv file containing the vignettes",
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
        question=args.question,
        n_seeds=args.n_seeds,
    )