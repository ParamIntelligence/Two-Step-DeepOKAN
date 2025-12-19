#!/bin/bash
#SBATCH -N 1                                    # 
#SBATCH -n 16                                   # 
#SBATCH --mem=64g                               # 
#SBATCH -J "TrunkNet_Spline_FullBatch"          #
#SBATCH -o trunk_train_3ls_150_150_k2G40_lr1em4_%j_allfreq_fullbat.out       # name of the output file
#SBATCH -e trunk_train_3ls_150_150_k2G40_lr1em4_%j_allfreq_fullbat.err       # name of the error file
#SBATCH -p short                    # 
#SBATCH -t 10:00:00                 #
#SBATCH --gres=gpu:1                # 
#SBATCH -C "V100"                   # 

module load python/3.13.3           #
source ../.venv/bin/activate        #

python main_trunk.py         #