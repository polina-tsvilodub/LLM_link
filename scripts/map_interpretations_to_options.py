"""
Utility script for mapping freely produced interpretations onto forced choice options.
"""

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
import os
import re

MAPPER_PROMPT = '''Consider the following claim:

{interpretation}

Which of the following options would you choose as a paraphrase of the claim above?

{options}

Output the number of the most likely interpretation. Reason step by step.

Here is the structure of the answer:

"""
[step-by-step explanation,
possibly over multiple lines]
[empty line]
[number]
"""

Your answer:
'''

def shuffle_interpretation_options(
            options_keys,
            options,
        ):
        """
        Helper for shuffling interpretation options
        and keeping track of inverse mapping.

        Parameters
        ----------
        option_keys: list[str]
            List of keys for the options 
            (target, competitor etc).
        options: list[str]
            List of interpretations corresponding to the keys
            to be selected from in the FC task.

        Returns
        -------
        shuffled_keys: list[str]
            List of shuffled keys.
        shuffled_options_formatted: str
            Formatted string of shuffled options.
            Formatted as a numbered list, starting at 1.
        """
        shuffled_options_indices = np.random.choice(
            np.arange(len(options_keys)),
            size=len(options_keys),
            replace=False,
        )

        shuffled_keys = options_keys[
            shuffled_options_indices
        ]

        shuffled_options = np.array(options)[
            shuffled_options_indices
        ]

        shuffled_options_formatted = "\n".join([
            str(i+1) + ". " + option
            for i, option
            in enumerate(shuffled_options)
        ])

        return shuffled_keys, shuffled_options_formatted

def map_interpretations_to_options(
        results_file, 
        stimuli_file,
        model_name="gpt-3.5-turbo-instruct",
        temperature=0.1,
        max_tokens=32,
        **kwargs,
    ):
    # read results
    results = pd.read_csv(results_file)
    stimuli = pd.read_csv(stimuli_file)
    # construct output file name
    time = datetime.now().strftime("%Y%m%d_%H%M")
    out_name = results_file.split("/")[-1].replace(".csv", "")
    out_file = f"../results/free_production/{out_name}_postprocessed_{time}.csv"
    
    # gather options for each item
    results['options'] = results['item_id'].apply(
        lambda x: stimuli[stimuli['item_number'] == x]['target':].tolist()
    )
    # gather option names for each item
    option_names = list(stimuli.loc[:, 'target':].columns)
    results['option_names'] = option_names
    # shuffle options and names and format prompt
    
    for i, row in results.iterrows(): 
        shuffled_option_names, shuffled_options = shuffle_interpretation_options(
            option_names,
            results['options'].tolist(),
        )
        # add list of shuffled options to results df
        results.loc[i, 'shuffled_options'] = shuffled_options
        results.loc[i, 'shuffled_option_names'] = shuffled_option_names
        # format prompt
        prompt = MAPPER_PROMPT.format(
            interpretation=row['response'],
            options=shuffled_options,
        )
        # run the mapper 
        # init model
        outputs = openai.Completion.create(
            model    = model_name, 
            prompt   = prompt,
            max_tokens  = max_tokens, # we only want to score, so no new tokens
            temperature = temperature, 
            logprobs = 0,
            echo = True,
        )
        # postprocess outputs
        text_offsets = outputs.choices[0]['logprobs']['text_offset']
        cutIndex = text_offsets.index(max(i for i in text_offsets if i <= len(prompt)))
        endIndex = outputs.usage.total_tokens
        answerTokens = outputs.choices[0]["logprobs"]["tokens"][cutIndex:endIndex]
        print("Answer tokens:", answerTokens)

        # pull out option number and name
        pattern = r"\b\d+(?:\.\d+)?\b"
        
        matches = re.findall(
            pattern,
            # only consider last nonempty line
            [
                x
                for x in answerTokens.split("\n")
                if not any([x == "", x == "'''", x == '"""'])
            ][-1],
        )
        if matches:
            chosen_option_index = int(matches[-1])
            # NOTE: the following is specific to the interpretation mapping
            # of the interpretation case study
            # if the number found is 0, assume the mapping was refused
            # and output respective string
            if chosen_option_index == 0:
                chosen_option_index = "error: refused"
        else:
            chosen_option_index = "error: no number found in the text."

        if isinstance(chosen_option_index, int):
            chosen_interpr = shuffled_option_names[
                chosen_option_index -1
            ]
        else:
            chosen_interpr = 'error in interpretation mapping'

        # record
        results.loc[i, 'chosen_option_index'] = chosen_option_index
        results.loc[i, 'chosen_option_name'] = chosen_interpr

        if os.path.exists(out_file):
            results.to_csv(out_file, 
                            index=False,
                            mode="a",
                            header=False,
                            )
        else:
            results.to_csv(out_file, 
                            index=False,
                            mode="a",
                            header=True,
                            )