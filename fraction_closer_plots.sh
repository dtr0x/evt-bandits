#!/bin/bash

#SBATCH --account=def-jiayuan
#SBATCH --time=00:10:00

dirname=$1

module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index -r requirements.txt

python fraction_closer_plots.py $dirname
