#!/bin/bash
#SBATCH --job-name=my_model
#SBATCH --partition=ws-ia
#SBATCH --time=24:00:00
#SBATCH --nodes=1                # Same as -N1
#SBATCH --ntasks=12              # Same as -n12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=model_%j.log

source /home/loic.martins/miniforge/etc/profile.d/conda.sh
conda activate ./env
python3 -m eval.runner
