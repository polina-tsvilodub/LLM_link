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
from score_completions import retrieve_log_probs
import argparse
import random
import string
from datetime import datetime
import time

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device = {DEVICE}")

import os

# Some helper functions
def softmax(x):
    return np.exp(x)/sum(np.exp(x))

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
    file_path = "../data/experiment1_text_items_short.csv"
    instructions_path = "../prompt/prompts/scope_instructions.txt"
    # initialize path for dumping output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_name = file_path.split("/")[-1].replace(".csv", "")
    model_name_out = model_name.split("/")[-1]
    # Load model and tokenizer
    tokenizer, model = load_model(model_name)

    with open(instructions_path, "r") as f:
        instructions = f.read()

    # Define seeds
    seeds = range(n_seeds)
    scenarios = pd.read_csv(file_path).dropna()
    # define FC answer options
    quantifier_scope_options = {
        "Question": ['Yes', 'No'],
        "Question2": ['None', 'Some', 'All']
    }
    si_options = {
        "Question": ['Yes', 'No'],
        "Question2": ['Yes', 'No'],
        "Question3": ['Less than 50 percent', 'More than 50 percent']
    }
    
    label_subdirectory = "label_alpha"

    for seed in seeds:
        out_file = f"../results/log_probs/seed{str(seed)}/label_scores/{label_subdirectory}/{model_name_out}_FloyedEtal_{out_name}_FC_labels_seed{seed}_{timestamp}.csv"
        if not os.path.exists(f"../results/log_probs/seed{str(seed)}/label_scores/{label_subdirectory}"):
            os.makedirs(f"../results/log_probs/seed{str(seed)}/label_scores/{label_subdirectory}")

       


        for i, row in tqdm(scenarios.iterrows()):
            
            prompt = instructions + "\n\n" + row.Story
            if row.Task == "QuantifierScope":
                for q in ["Question", "Question2"]:
                    options = quantifier_scope_options[q]

                     # set option numbering
                    
                    option_numbering = list(string.ascii_uppercase[:len(options)])
                    option_names = options.copy()
            
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

            
                    prompt_randomized = prompt + row[q] + "\nChoose one of the following options and return the label of that option.\n" + "\n".join([". ".join(o) for o in zip(option_numbering, shuffled_options)]) + "\nYour answer:\n"
                    options = option_numbering
                    prior_prompt = prompt + "\n\n" + "Choose one of the following options and return the label of that option.\n" + "\n".join([". ".join(o) for o in zip(option_numbering, shuffled_options)]) + "\nYour answer:\n"

                    print("---- formatted prompt ---- ", prompt_randomized)
            
                    option_conditional_log_probs, log_probs, null_log_probs = retrieve_log_probs(
                        prompt_randomized, 
                        prior_prompt,
                        options,
                        model_name,
                        model, 
                        tokenizer,
                        temperature=temperature,
                        use_labels_only=True,
                        option_numbering=option_numbering,
                    )
                    ###### compute derived metrics ######
                    # 0. token probabilities
                    # nested list of conditional token probs
                    token_cond_probs = []
                    for o in option_conditional_log_probs:
                        token_cond_probs.append(
                            [np.exp(p) for p in o]
                        )
                    # and prior unconditional token probs
                    # nested list of prior token probs
                    # we use the NULL CONTEXT priors by default
                    token_probs = []
                    print("unconditional log probs ", log_probs)
                    for o in null_log_probs:
                        token_probs.append(
                            [np.exp(p) for p in o]
                        )

                    # list of prior sentence probs
                    prior_probs = [
                        np.prod(
                            np.array(t)
                        ) for t 
                        in token_probs
                    ]
                    # 1. sentence probability
                    sentence_cond_probs = [
                        np.prod(
                            np.array(t)
                        ) for t 
                        in token_cond_probs
                    ]
                    # 2. length-normalized sentence probability
                    mean_sentence_cond_probs = [
                        (1/len(t))* np.sum(
                            np.array(t)
                        ) for t 
                        in token_cond_probs
                    ]
                    # 3. prior correction (= empirical MI)
                    # NOTE: MI computed without length normalization
                    sentence_mi = [
                        compute_mi(s, p)
                        for s, p
                        in zip(sentence_cond_probs, prior_probs)
                    ]

                    # 4. sentence surprisal
                    sentence_surprisal = [
                        np.sum(
                            np.array(p)
                        ) for p 
                        in option_conditional_log_probs
                    ]
                    # 5. length-normalized sentence surprisal
                    mean_sentence_surprisal = [
                        (1/len(p)) * np.sum(
                            np.array(p)
                        ) for p 
                        in option_conditional_log_probs
                    ]
                    # 6. prior corrected (= empirical MI) sentence surprisal
                    sentence_mi_surprisal = [
                        compute_mi(s, p)
                        for s, p
                        in zip(sentence_surprisal, 
                                [np.sum(np.array(t)) for t in null_log_probs])
                    ]
                    
                    print(option_conditional_log_probs, token_cond_probs, sentence_cond_probs, sentence_mi, mean_sentence_surprisal)

                    # TODO deal with re-normalization somewhere

                    #####################################

                    # Record the retrieved log probs
                    # TODO record other configs
                    # initialize results df, so that we can write result in long format
                    results_df = pd.DataFrame({
                        "model_name": [model_name] * len(options),
                        "temperature": [temperature] * len(options),
                        "seed": [seed] * len(options),
                        "item_id": [row.ItemNum]  * len(options),
                        "task": [row.Task] * len(options),
                        "prompt": [prompt] * len(options),
                        "prior_prompt": [prior_prompt] * len(options),
                        "question_number": [q] * len(options),
                        "question": [row[q]] * len(options),
                        "item": [row.Item] * len(options),
                        "correct1": [row.Correct1] * len(options),
                        "correct2": [row.Correct2] * len(options),
                        "options": options,
                        "option_names": option_names,
                        "shuffled_options": shuffled_options, # record randomization
                        "shuffled_option_names": shuffled_option_names,
                        "option_numbering": option_numbering,
                        "token_cond_log_probs": option_conditional_log_probs,
                        "token_cond_probs": token_cond_probs,
                        "prior_token_log_probs": log_probs,
                        "null_prior_token_log_probs": null_log_probs,
                        "token_probs": token_probs,
                        "sentence_cond_probs": sentence_cond_probs,
                        "mean_sentence_cond_probs": mean_sentence_cond_probs,
                        "prior_sentence_probs": prior_probs,
                        "sentence_mi": sentence_mi,
                        "sentence_surprisal": sentence_surprisal,
                        "mean_sentence_surprisal": mean_sentence_surprisal,
                        "sentence_mi_surprisal": sentence_mi_surprisal,
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
                        
            else:
                for q in ["Question", "Question2", "Question3"]:
                    options = quantifier_scope_options[q]

                     # set option numbering
                    option_numbering = list(string.ascii_uppercase[:len(options)])
                    option_names = options.copy()
            
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

            
                    prompt_randomized = prompt + row[q] + "\nChoose one of the following options and return the label of that option.\n" + "\n".join([". ".join(o) for o in zip(option_numbering, shuffled_options)]) + "\nYour answer:\n"
                    options = option_numbering
                    prior_prompt = prompt + "\n\n" + "Choose one of the following options and return the label of that option.\n" + "\n".join([". ".join(o) for o in zip(option_numbering, shuffled_options)]) + "\nYour answer:\n"

                    print("---- formatted prompt ---- ", prompt_randomized)
            
                    option_conditional_log_probs, log_probs, null_log_probs = retrieve_log_probs(
                        prompt_randomized, 
                        prior_prompt,
                        options,
                        model_name,
                        model, 
                        tokenizer,
                        temperature=temperature,
                        use_labels_only=True,
                        option_numbering=option_numbering,
                    )
                    ###### compute derived metrics ######
                    # 0. token probabilities
                    # nested list of conditional token probs
                    token_cond_probs = []
                    for o in option_conditional_log_probs:
                        token_cond_probs.append(
                            [np.exp(p) for p in o]
                        )
                    # and prior unconditional token probs
                    # nested list of prior token probs
                    # we use the NULL CONTEXT priors by default
                    token_probs = []
                    print("unconditional log probs ", log_probs)
                    for o in null_log_probs:
                        token_probs.append(
                            [np.exp(p) for p in o]
                        )

                    # list of prior sentence probs
                    prior_probs = [
                        np.prod(
                            np.array(t)
                        ) for t 
                        in token_probs
                    ]
                    # 1. sentence probability
                    sentence_cond_probs = [
                        np.prod(
                            np.array(t)
                        ) for t 
                        in token_cond_probs
                    ]
                    # 2. length-normalized sentence probability
                    mean_sentence_cond_probs = [
                        (1/len(t))* np.sum(
                            np.array(t)
                        ) for t 
                        in token_cond_probs
                    ]
                    # 3. prior correction (= empirical MI)
                    # NOTE: MI computed without length normalization
                    sentence_mi = [
                        compute_mi(s, p)
                        for s, p
                        in zip(sentence_cond_probs, prior_probs)
                    ]

                    # 4. sentence surprisal
                    sentence_surprisal = [
                        np.sum(
                            np.array(p)
                        ) for p 
                        in option_conditional_log_probs
                    ]
                    # 5. length-normalized sentence surprisal
                    mean_sentence_surprisal = [
                        (1/len(p)) * np.sum(
                            np.array(p)
                        ) for p 
                        in option_conditional_log_probs
                    ]
                    # 6. prior corrected (= empirical MI) sentence surprisal
                    sentence_mi_surprisal = [
                        compute_mi(s, p)
                        for s, p
                        in zip(sentence_surprisal, 
                                [np.sum(np.array(t)) for t in null_log_probs])
                    ]
                    
                    print(option_conditional_log_probs, token_cond_probs, sentence_cond_probs, sentence_mi, mean_sentence_surprisal)

                    # TODO deal with re-normalization somewhere

                    #####################################

                    # Record the retrieved log probs
                    # TODO record other configs
                    # initialize results df, so that we can write result in long format
                    results_df = pd.DataFrame({
                        "model_name": [model_name] * len(options),
                        "temperature": [temperature] * len(options),
                        "seed": [seed] * len(options),
                        "item_id": [row.ItemNum]  * len(options),
                        "task": [row.Task] * len(options),
                        "prompt": [prompt] * len(options),
                        "prior_prompt": [prior_prompt] * len(options),
                        "question_number": [q] * len(options),
                        "question": [row[q]] * len(options),
                        "item": [row.Item] * len(options),
                        "item_type": [row.Type] * len(options),
                        "correct1": [row.Correct1] * len(options),
                        "correct2": [row.Correct2] * len(options),
                        "correct3": [row.Correct3] * len(options),
                        "options": options,
                        "option_names": option_names,
                        "shuffled_options": shuffled_options, # record randomization
                        "shuffled_option_names": shuffled_option_names,
                        "option_numbering": option_numbering,
                        "token_cond_log_probs": option_conditional_log_probs,
                        "token_cond_probs": token_cond_probs,
                        "prior_token_log_probs": log_probs,
                        "null_prior_token_log_probs": null_log_probs,
                        "token_probs": token_probs,
                        "sentence_cond_probs": sentence_cond_probs,
                        "mean_sentence_cond_probs": mean_sentence_cond_probs,
                        "prior_sentence_probs": prior_probs,
                        "sentence_mi": sentence_mi,
                        "sentence_surprisal": sentence_surprisal,
                        "mean_sentence_surprisal": mean_sentence_surprisal,
                        "sentence_mi_surprisal": sentence_mi_surprisal,
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
        default=None,
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
