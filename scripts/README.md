# Scripts
The directory contains scripts to retrieve LLM data for various metrics for mapping onto human psycholinguistic data.

In order to run the scripts, install the necessary requirements, navigate to the `scripts` directory, and run: `python <script_name>.py --phenomenon=<name of phenomenon> --option_numbering=<option label strings separated by commas>` . There are additional configurations which might need to be set for the experiments (also inspectable with `--help`).
These configs are (availability might differ by script):
* `--temperature`: sampling temperature
* `--model_name`: string name of the model (OpenAI or HuggingFace LLama 2 and FLAN-T5 should be supported)
* `--phenomenon`: name of the phenomenon for which the metrics are run. Must be one of: "coherence", "deceits", "humour", "indirect_speech", "irony", "maxims", "metaphor".
* `--option_numbering`: string representing the option prefixes to be used (e.g., A, B, C etc). Defaults to A,B,C,D. **Important**: for phenomena with a different number of interpretation options than 2, the option numering must be set appropriately (e.g., pass only 'A,B' for the coherence phenomenon). Relevant for FC and embedding metrics.
* `--use_labels_only`: boolean indicating whether to run the label probability computations (with answer options listed in the prompt), or the option string probability & surprisal computations (no answer options list in the context). Relevant for FC metrics and embeddings (string probability and surprisal vs label probability).
* `--question`: Task question string. Defaults to "". **IMPORTANT**: for coherent prompts, the following phenomena require passing the respective question:
  * `coherence`: Is this story coherent or not?
  * `maxims`: Why has {} responded like this?
* `--n_seeds`: The number of seeds for which the experimental configuration is run. Defaults to 1.
* `--max_new_tokens`: number of new tokens to sample. Defaults to 50. Relevant for free production metric.
* `--decoding_scheme`: defaults to greedy (or whatever OpenAI does). Can be set to 'softmax' or `beam_search` for HuggingFace models. Relevant for free production metric.

* `compare_embeddings.py`: script for eliciting cosine similarities of embeddings. For example, we compute the embedding of the prompt containing: instructions, the context; and the embedding of an answer option. We then compute the cosine similarity of the embeddings. 
  * for this metric, we use the same instructions as for the forced choice task, i.e., set `--instructions_path=../prompt/prompt_fc/<phenomenon prompt>.txt`. 
  * we can set whether to `--use_labels_only`.
* `free_production.py`: here, explanation of the speaker's behavior is produced freely. The responses need to be categorized into target, competitor etc categories manually. 
* `rate_completions.py`: here, ratings of the single options are scored. The ratings are on 5-point scales, described by the words appropriate, likely, possible and plausible. All the rating words are used iteratively when the script is run. 
* `score_completions.py`: all token probability metrics are computed here. If `--use_labels_only` is set, the label probability metrics are computed; otherwise, the string option metrics are computed.
  
Please create the following subdirectories within the `results` directory, if you haven't done so yet: `embedding_similarity`, `free`, `log_probs`, `rating`.

### Check list for running an experiment
* correct phenomenon passed?
* correct number and type of labels provided via `--option_numbering`?
* correct task question passed for the given phenomenon?
* correct model name set?
* if needed, is the `--use_labels_only` option set?
* results subdirectory for the metric created?


The directory also contains `utils.py` which contains helpers for loading models and computing metrics, and `map_interpretation_to_options.py`. The latter is WIP and might be used for automatic mapping of free production results onto the forced choice interpretation options (ask Polina for more details). 