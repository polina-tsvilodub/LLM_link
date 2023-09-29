# Scripts
The directory contains scripts to retrieve LLM data for various metrics for mapping onto human psycholinguistic data.

In order to run the scripts, install the necessary requirements, navigate to the `scripts` directory, and run: `python <script_name>.py --file_path=<path_to_phenomenon_file> --instructions_path=<path_to_instructions_file> --option_numbering=<option label strings separated by commas>` . There are additional configurations which can be set for the experiments (also inspectable with `--help`).
These configs are (availability might differ by script):
* `--temperature`: sampling temperature
* `--model_name`: string name of the model (OpenAO or HuggingFace LLama 2 and FLAN-T5 should be supported)
* `--option_numbering`: string representing the option prefixes to be used (e.g., A, B, C etc). Defaults to A,B,C,D. **Important**: for phenomena with a different number of interpretation options than 2, the option numering must be set appropriately. Relevant for FC and embedding metrics.
* `--use_labels_only`: boolean indicating whether the options should have the shape, e.g., "A" only, or e.g., both "A. Coherent". Relevant for FC metrics (string probability and surprisal vs label probability).
* `--question`: Task question string. Defaults to "". **IMPORTANT**: for coherent prompts, the following phenomena require passing the respective question:
  * `coherence`: Is this story coherent or not?
  * `maxims`: Why has {} responded like this?
* `--n_seeds`: The number of seeds for which the experimental configuration is run. Defaults to 1.

* `compare_embeddings.py`: script for eliciting cosine similarities of embeddings. For example, we compute the embedding of the instructions, the context and the answer options: "You will read pairs of sentences. Reach each pair and decide whether they form a coherent story. The answer options are A, B. \n Mary's exam was about to start. Her palms were sweaty.  Which of the following options would you choose? A. Coherent B. Incoherent \n \n Your answer: " and the embedding of "A. Coherent". We then compute the cosine similarity of the embeddings. 
  * for this metric, we use the same instructions as for the forced choice task. 
* `score_completions.py`: all token probability metrics are computed here.


If dataset has more than 2 options, then the numbering should be appended for generating correct instructions.
Construct subdirectories in results/