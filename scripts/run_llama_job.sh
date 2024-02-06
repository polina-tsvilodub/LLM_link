#!/bin/bash
#SBATCH --partition=single
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --mem=20gb
#SBATCH --gres=gpu:A40:1

echo 'Running simulation'

# The directory name is the 
# first command line argument
dir_name=$1

# Now, change to that directory 
# (assuming it's a subdirectory of 
# the parent directory of the current script)

# print the current working directory
# (should be the directory of the script)
echo "Current working directory:"
echo $(pwd)

# Load conda
module load devel/miniconda/3
source $MINICONDA_HOME/etc/profile.d/conda.sh

conda deactivate
# Activate the conda environment
conda activate llmlink

echo "Conda environment activated:"
echo $(conda env list)
echo " "
echo "Python version:"
echo $(which python)
echo " "

# activate CUDA
module load devel/cuda/11.6

# iterate over phenomena
array=("coherence" "deceits" "humour" "indirect_speech" "irony" "maxims" "metaphor")
# and respective option numbering
array2=("A,B" "A,B,C,D" "A,B,C,D,E" "A,B,C,D" "A,B,C,D" "A,B,C,D" "A,B,C,D,E")
# and questions 
array3=("Is this story coherent or not?" "" "" "" "" "Why has {} responded like this?" "")
# models
hf_models=("EleutherAI/pythia-12b" "microsoft/phi-2" "mistralai/Mistral-7B-Instruct-v0.2") #  "mistralai/Mixtral-8x7B-v0.1")

for i in ${!array[*]}; do
    echo "phenomenon: ${array[$i]}"
    echo "options: ${array2[$i]}"
    python3 -u score_completions.py \
        --phenomenon=${array[$i]} \
        --question="${array3[$i]}" \
        --model_name="${hf_models[$i]}" \
        --option_numbering="${array2[$i]}" \
        --use_labels_only=True \
        --n_seeds=1
done
