#!/bin/bash

#SBATCH --account=def-jiayuan
#SBATCH --array=0-4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:10:00

dirname=$1

module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index -r requirements.txt

python fraction_closer_search.py $dirname