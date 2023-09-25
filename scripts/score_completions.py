from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import openai
import torch
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from utils import load_model, compute_mi, softmax
import argparse
import string

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device = {DEVICE}")

import os

# Some helper functions
def softmax(x):
    return np.exp(x)/sum(np.exp(x))


def retrieve_log_probs(
        prompt, # or item / row in the df
        options, # options to be scored (as a list so that there can be varying numbers by phenomenon)
        model_name=None, # only to be used for openAI models
        model=None, # only to be used with HF models
        tokenizer=None, # only to be used with HF models
        phenomenon=None,
        **kwargs
    ):
    '''
    Entrypoint for getting the retrieval of log ptobabilities 
    of the response options, given the context.

    Parameters
    -----------
    model_name: str
    prompt: str
    model: T5ForConditionalGeneration
    tokenizer: T5Tokenizer
    kwargs: additional arguments to pass to model.generate

    Return
    ------ 
    log_probs: list[list[float]]
        List of token log probs.
    '''

    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # TODO open q: do we do the diff random seeds / orders of options, too?
    
    conditional_log_probs = []
    log_probs = []

    # iterate over items here so that the model won't have to be reloaded for each item & option
    for o in options:
        input_prompt = prompt + o
        print("\n---- INPUT PROMPT ----- \n ", input_prompt, "\n----------------------")

        # Tokenize the prompt
        if ("google/flan-t5" in model_name) or ("meta/" in model_name):
            print("Using HF code")
            ##### retreiver conditional log prob #####
            input_ids = tokenizer(
                input_prompt, 
                return_tensors="pt",
            ).to(DEVICE)
            model = model.to(DEVICE)
            # Generate output from the model with a maximum length of 20 tokens
            outputs = model.generate(
                **input_ids,
                max_new_tokens = 0, # we only want to score, not produce new tokens
                output_scores=True,
                num_return_sequences=1,
                return_dict_in_generate=True,
                temperature=kwargs['temperature'],
            )

            # Convert the generated token IDs back to text
            optionTokens = tokenizer.decode(
                outputs.sequences[0], 
                skip_special_tokens=True
            ) # TODO check if these need to be ported back to cpu

            #### retrieve unconditional log prob of the option ####
            option_input_ids = tokenizer(
                o, 
                return_tensors="pt",
            ).to(DEVICE)

            option_outputs = model.generate(
                **option_input_ids,
                max_new_tokens = 0, # we only want to score, not produce new tokens
                output_scores=True,
                num_return_sequences=1,
                return_dict_in_generate=True,
                temperature=kwargs['temperature'],
            )
            ####

        elif "gpt-3.5" in model_name:
            ##### retreiver conditional log prob #####
            outputs = openai.Completion.create(
                model    = model_name, 
                prompt   = input_prompt,
                max_tokens  = 0, # we only want to score, so no new tokens
                temperature = kwargs['temperature'], 
                logprobs = 0,
                echo = True,
            )

            #### retrieve unconditional log prob of the option ####
            option_outputs = openai.Completion.create(
                model    = model_name, 
                prompt   = o,
                max_tokens  = 0, # we only want to score, so no new tokens
                temperature = kwargs['temperature'], 
                logprobs = 0,
                echo = True,
            )

        else:
            raise ValueError(f"Model {model_name} is not supported as a backbone.")

        
        
        # Retrieve logits from the output
        try: 
            # retrieve HF log probs
            # [0] retrieves first element in batch (assume we always use batch size = 1)
            optionTokenConditionalLogProbs = outputs.scores[0].tolist()
            print("Log probs of HF answer tokens:", optionTokenConditionalLogProbs)

            optionTokenLogProbs = option_outputs.scores[0].tolist()

        except:
            # retrieve OpenAI log probs
            text_offsets = outputs.choices[0]['logprobs']['text_offset']
            cutIndex = text_offsets.index(max(i for i in text_offsets if i <= len(prompt)))
            endIndex = outputs.usage.total_tokens
            optionTokens = outputs.choices[0]["logprobs"]["tokens"][cutIndex:endIndex]
            optionTokenConditionalLogProbs = outputs.choices[0]["logprobs"]["token_logprobs"][cutIndex:endIndex] 
            print("Answer tokens:", optionTokens)
            print("Log probs of answer tokens:", optionTokenConditionalLogProbs)

            # unconditional log probs (see class code)
            endIndex = len(option_outputs.choices[0]["logprobs"]["tokens"])
            optionTokenLogProbs = option_outputs.choices[0]["logprobs"]["token_logprobs"][:endIndex] 
            print("Uncut prior log Ps ", optionTokenLogProbs)
            optionTokenLogProbs = optionTokenLogProbs[1:endIndex]

        conditional_log_probs.append(optionTokenConditionalLogProbs)
        log_probs.append(optionTokenLogProbs)

    return conditional_log_probs, log_probs


# TODO construct the context (i.e., A / B / 1 / 2) 
# and the option order shuffling here

def main(
        file_path,
        temperature=0.1,
        model_name="gpt-3.5-turbo-instruct",
        option_numbering=None,
        use_option_numbering_only=False,
):
    # Load model and tokenizer
    tokenizer, model = load_model(model_name)

    # Define answer types
    # TODO
    anwsers = ["Both", "Content", "Number"]

    # TODO
    # Define seeds
    seeds = range(5)

    # set option numbering
    if option_numbering is None:
        option_numbering = list(string.ascii_uppercase[:len(options)])
    else:
        option_numbering = option_numbering.split(",")

    # Iterate over seeds
    for seed in seeds:
        # Reseed the singleton RandomState instance.
        np.random.seed(seed)

        # Iterate over anwsers
        for anwser in anwsers:
            # Load data set

            scenarios = pd.read_csv(file_path).dropna()
            print(scenarios.head())
            

            # Iterate over rows in prompt csv 
            for i, row in tqdm(scenarios.iterrows()):
                # load instructions
                with open("../prompt/prompt_fc/MaximsInstructions_FC_Both.txt", "r") as f:
                    instructions = f.read()
                # Get prompt and generate answer
                prompt = instructions + "\n\n" + row.prompt
                # construct task question
                question = f"Why has {row.speaker} responded like this? \n"
                # the df should have the target option and then other options as last columns
                options = list(row.loc['target':])

                # TODO add the list of options in a randomized seed dependent order
                prompt_randomized = prompt + row.trigger + question + "\n ".join([". ".join(o) for o in zip(option_numbering, options)]) + "\nYour answer:\n"
                print("---- formatted prompt ---- ", prompt_randomized)
                
                option_conditional_log_probs, log_probs = retrieve_log_probs(
                    prompt_randomized, 
                    options,
                    model_name,
                    model, 
                    tokenizer,
                    temperature=temperature,
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
                token_probs = []
                print("unconditional log probs ", log_probs)
                for o in log_probs:
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
                    (1/len(t))* np.prod(
                        np.array(t)
                    ) for t 
                    in token_cond_probs
                ]
                # 3. prior correction (= empirical MI)
                # NOTE: MI computed without length normalization
                mean_sentence_mi = [
                    compute_mi(s, p)
                    for s, p
                    in zip(sentence_cond_probs, prior_probs)
                ]

                # 4. sentence surprisal
                sentence_surprisal = [
                    - np.sum(
                        np.array(p)
                    ) for p 
                    in option_conditional_log_probs
                ]
                # 5. length-normalized sentence surprisal
                mean_sentence_surprisal = [
                    - (1/len(p)) * np.sum(
                        np.array(p)
                    ) for p 
                    in option_conditional_log_probs
                ]
                # 6. prior corrected (= empirical MI) sentence surprisal
                # TODO: what is this metric conceptually?
                mean_sentence_mi_surprisal = [
                    compute_mi(s, p)
                    for s, p
                    in zip(sentence_surprisal, 
                           [- np.sum(np.array(t)) for t in log_probs])
                ]
                # 7. perplexity TODO double check
                ppl = [
                    np.exp(s) for s
                    in sentence_surprisal
                ]
                print(option_conditional_log_probs, token_cond_probs, sentence_cond_probs, mean_sentence_mi, mean_sentence_surprisal)

                # TODO deal with re-normalization somewhere

                # TODO utils for label probability and ranking probability

                #####################################

                # Record the retrieved log probs
                # initialize results df, so that we can write result in long format
                results_df = pd.DataFrame({
                    "answer": [anwser] * len(options),
                    "item_id": [row.item_number]  * len(options),
                    "prompt": [prompt] * len(options),
                    "options": options, # TODO record randomization, if needed
                    "option_numbering": option_numbering,
                    "token_cond_log_probs": option_conditional_log_probs,
                    "token_cond_probs": token_cond_probs,
                    "prior_token_log_probs": log_probs,
                    "token_probs": token_probs,
                    "sentence_cond_probs": sentence_cond_probs,
                    "mean_sentence_cond_probs": mean_sentence_cond_probs,
                    "prior_sentence_probs": prior_probs,
                    "mean_sentence_mi": mean_sentence_mi,
                    "sentence_surprisal": sentence_surprisal,
                    "mean_sentence_surprisal": mean_sentence_surprisal,
                    "mean_sentence_mi_surprisal": mean_sentence_mi_surprisal,
                    "ppl": ppl,
                })
                print(results_df)

                # results_df_long = results_df.explode([
                #     "options",
                #     "option_numbering",
                #     "token_cond_log_probs",
                #     "token_cond_probs",
                #     "prior_token_log_probs",
                #     "token_probs",
                #     "sentence_cond_probs",
                #     "mean_sentence_cond_probs",
                #     "prior_sentence_probs",
                #     "mean_sentence_mi",
                #     "sentence_surprisal",
                #     "mean_sentence_surprisal",
                #     "mean_sentence_mi_surprisal",
                #     "ppl",
                # ])
            

                # Save results to csv continuously
                out_name = file_path.split("/")[-1].replace(".csv", "")
                # write header depending on whether the file already exists
                if os.path.exists(f"../results/log_probs/{out_name}_{anwser}_seed{seed}.csv"):
                    results_df.to_csv(f"../results/log_probs/{out_name}_{anwser}_seed{seed}.csv", 
                                    index=False,
                                    mode="a",
                                    header=False,
                                    )
                else:
                    results_df.to_csv(f"../results/log_probs/{out_name}_{anwser}_seed{seed}.csv", 
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
        "--use_option_numbering_only",
        type=bool,
        default=False,
        help="Whether to use only labels for options as options to be scored",
    )

    args = parser.parse_args()

    main(
        file_path=args.file_path,
        temperature=args.temperature,
        model_name=args.model_name,
        option_numbering=args.option_numbering,
        use_option_numbering_only=args.use_option_numbering_only,
    )