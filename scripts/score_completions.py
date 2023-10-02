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
import time

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device = {DEVICE}")

import os

# Some helper functions
def softmax(x):
    return np.exp(x)/sum(np.exp(x))


def retrieve_log_probs(
        prompt, # or item / row in the df
        prior_prompt, # prior prompt for unconditional log probs
        options, # options to be scored (as a list so that there can be varying numbers by phenomenon)
        model_name=None, # only to be used for openAI models
        model=None, # only to be used with HF models
        tokenizer=None, # only to be used with HF models
        phenomenon=None,
        score_prior=True,
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
    null_log_probs = []

    # transformation for LLaMA 2 outputs
    logsoftmax = torch.nn.LogSoftmax(dim=-1)

    # iterate over items here so that the model won't have to be reloaded for each item & option
    for o in options:
        optionTokenLogProbs = []

        input_prompt = prompt + o
        null_input_prompt = prior_prompt + o
        print("\n---- INPUT PROMPT ----- \n ", input_prompt, "\n----------------------")

        # Tokenize the prompt
        if ("google/flan-t5" in model_name) or ("meta" in model_name):
            
            print("Using HF code")
            ##### retreiver conditional log prob #####
            input_ids_prompt = tokenizer(
                prompt, 
                return_tensors="pt",
            ).input_ids
            
            input_ids_options = tokenizer(
                o, 
                return_tensors="pt",
            ).input_ids
            # input option is sliced so that the SOS token isn't included again
            input_ids = torch.cat((input_ids_prompt, input_ids_options[:, 1:]), -1).to(DEVICE)
            print("input ids shape: ", input_ids.shape)
            #### retrieve unconditional log prob of the option ####
            option_input_ids = tokenizer(
                o, 
                return_tensors="pt",
            ).to(DEVICE)
            # also, compute null prompt prior
            null_option_input_ids_prompt = tokenizer(
                prior_prompt, 
                return_tensors="pt",
            ).input_ids
            
            null_option_input_ids = torch.cat((null_option_input_ids_prompt, input_ids_options[:, 1:]), -1).to(DEVICE)
            
            # Generate output from the model with a maximum length of 20 tokens
            if "llama" in model_name:
                outputs = model(
                    input_ids,
                )

                option_outputs = model(
                    **option_input_ids,
                )
                
                null_option_outputs = model(
                    null_option_input_ids,
                )

            elif "t5" in model_name:
                outputs = model(
                    input_ids,
                    decoder_input_ids=input_ids,
                )

                option_outputs = model(
                    **option_input_ids,
                    decoder_input_ids=option_input_ids['input_ids'],
                )

                print("raw options outputs shape ", option_outputs.logits.shape)                
                null_option_outputs = model(
                    null_option_input_ids,
                    decoder_input_ids=null_option_input_ids,
                )

            else:
                raise ValueError("Model {model_name} is not supported as a backbone for log probability retrieval.")

                

        elif ("gpt-3.5" in model_name) or ("davinci" in model_name):
            ##### retreiver conditional log prob #####
            outputs = openai.Completion.create(
                model    = model_name, 
                prompt   = input_prompt,
                max_tokens  = 0, # we only want to score, so no new tokens
                temperature = kwargs['temperature'], 
                logprobs = 0,
                echo = True,
            )

            #### retrieve unconditional log prob of the option in the null context ####
            # just option scoring isn't done with OpenAI since it doesn't return log probs of the first token
            if score_prior:
                null_option_outputs = openai.Completion.create(
                    model    = model_name, 
                    prompt   = null_input_prompt,
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
            if "llama" in model_name:
                # for llama, we manually retrieve the log probs from the output
                # and transform them into log probs
                try:
                    llama_output_scores = logsoftmax(
                        outputs.loss['logits'][0]
                    ) # access first element in batch; result has shape [n_tokens, 32000]
                    llama_option_output_scores = logsoftmax(
                        option_outputs.loss['logits'][0]
                    )
                    llama_null_option_output_scores = logsoftmax(
                        null_option_outputs.loss['logits'][0]
                    )
                except:
                    llama_output_scores = logsoftmax(
                        outputs.logits[0]
                    ) # access first element in batch; result has shape [n_tokens, 32000]
                    llama_option_output_scores = logsoftmax(
                        option_outputs.logits[0]
                    )
                    llama_null_option_output_scores = logsoftmax(
                        null_option_outputs.logits[0]
                    )
                # retreive log probs at token ids
                # transform input_ids to a tensor of shape [n_tokens, 1] for this
                input_ids_probs = input_ids.squeeze().unsqueeze(-1)
                option_ids_probs = option_input_ids['input_ids'].squeeze().unsqueeze(-1)
                null_option_ids_probs = null_option_input_ids.squeeze().unsqueeze(-1)
                # retreive
                optionTokenConditionalLogProbs = torch.gather(
                    llama_output_scores, 
                    dim=-1, 
                    index=input_ids_probs
                ).flatten().tolist()
                optionTokenLogProbs = torch.gather(
                    llama_option_output_scores, 
                    dim=-1, 
                    index=option_ids_probs
                ).flatten().tolist()[1:] # exclude SOS token
                nullOptionTokenLogProbs = torch.gather(
                    llama_null_option_output_scores, 
                    dim=-1, 
                    index=null_option_ids_probs
                ).flatten().tolist()

            elif "t5" in model_name:
                # for T5, we manually retrieve the log probs from the output
                # and transform them into log probs
                llama_output_scores = logsoftmax(
                    outputs.logits[0]
                ) # access first element in batch; result has shape [n_tokens, 32000]
                llama_option_output_scores = logsoftmax(
                    option_outputs.logits[0]
                )
                llama_null_option_output_scores = logsoftmax(
                    null_option_outputs.logits[0]
                )
                # retreive log probs at token ids
                # transform input_ids to a tensor of shape [n_tokens, 1] for this
                input_ids_probs = input_ids.squeeze().unsqueeze(-1)
                option_ids_probs = option_input_ids['input_ids'].squeeze().unsqueeze(-1)
                null_option_ids_probs = null_option_input_ids.squeeze().unsqueeze(-1)
                # retreive
                optionTokenConditionalLogProbs = torch.gather(
                    llama_output_scores, 
                    dim=-1, 
                    index=input_ids_probs
                ).flatten().tolist()
                optionTokenLogProbs = torch.gather(
                    llama_option_output_scores, 
                    dim=-1, 
                    index=option_ids_probs
                ).flatten().tolist()[1:] # exclude SOS token
                nullOptionTokenLogProbs = torch.gather(
                    llama_null_option_output_scores, 
                    dim=-1, 
                    index=null_option_ids_probs
                ).flatten().tolist()
            else:
                optionTokenConditionalLogProbs = outputs.scores[0].tolist()
                
                optionTokenLogProbs = option_outputs.scores[0].tolist()[1:] # exclude SOS token
                nullOptionTokenLogProbs = null_option_outputs.scores[0].tolist()

            
            # slice output to only get scores of the continuation
            optionTokenConditionalLogProbs = optionTokenConditionalLogProbs[input_ids_prompt.shape[-1]:]
            nullOptionTokenLogProbs = nullOptionTokenLogProbs[null_option_input_ids_prompt.shape[-1]:]

        except:
            # retrieve OpenAI log probs
            text_offsets = outputs.choices[0]['logprobs']['text_offset']
            cutIndex = text_offsets.index(max(i for i in text_offsets if i <= len(prompt)))
            endIndex = outputs.usage.total_tokens
            optionTokens = outputs.choices[0]["logprobs"]["tokens"][cutIndex:endIndex]
            optionTokenConditionalLogProbs = outputs.choices[0]["logprobs"]["token_logprobs"][cutIndex:endIndex] 
            print("Answer tokens:", optionTokens)
            print("Log probs of answer tokens:", optionTokenConditionalLogProbs)

            
            optionTokenLogProbs = []
            # unconditional log probs (see class code)
            text_offsets = null_option_outputs.choices[0]['logprobs']['text_offset']
            cutIndex = text_offsets.index(max(i for i in text_offsets if i <= len(prior_prompt)))
            endIndex = null_option_outputs.usage.total_tokens
            nullOptionTokens = null_option_outputs.choices[0]["logprobs"]["tokens"][cutIndex:endIndex]
            nullOptionTokenLogProbs = null_option_outputs.choices[0]["logprobs"]["token_logprobs"][cutIndex:endIndex]  
            print("Null prior log Ps ", nullOptionTokenLogProbs)

        conditional_log_probs.append(optionTokenConditionalLogProbs)
        log_probs.append(optionTokenLogProbs)
        null_log_probs.append(nullOptionTokenLogProbs)
    
    return conditional_log_probs, log_probs, null_log_probs


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
            out_file = f"../results/log_probs/{out_name}_FC_labels_seed{seed}_{timestamp}.csv"
        else:
            out_file = f"../results/log_probs/{out_name}_FC_optionString_seed{seed}_{timestamp}.csv"
        
        # Iterate over rows in prompt csv 
        for i, row in tqdm(scenarios.iterrows()):
            # load instructions
            with open(instructions_path, "r") as f:
                instructions = f.read()
            
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
                prompt_randomized = prompt + question + "\nChoose one of the following options and return the label of that option.\n" + "\n".join([". ".join(o) for o in zip(option_numbering, shuffled_options)]) + "\nYour answer:\n"
                options = option_numbering
                prior_prompt = instructions + "\n\n" + "Choose one of the following options and return the label of that option.\n" + "\n".join([". ".join(o) for o in zip(option_numbering, shuffled_options)]) + "\nYour answer:\n"
            else:
                prompt_randomized = prompt + question + "\nYour answer:\n"
                prior_prompt = instructions + "\nYour answer:\n"

            print("---- formatted prompt ---- ", prompt_randomized)
            
            option_conditional_log_probs, log_probs, null_log_probs = retrieve_log_probs(
                prompt_randomized, 
                prior_prompt,
                options,
                model_name,
                model, 
                tokenizer,
                temperature=temperature,
                use_labels_only=use_labels_only,
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
                "item_id": [row.item_number]  * len(options),
                "phenomenon": [phenomenon] * len(options),
                "prompt": [prompt] * len(options),
                "prior_prompt": [prior_prompt] * len(options),
                "question": [question] * len(options),
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