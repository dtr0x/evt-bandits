#!/bin/bash
#SBATCH --account=def-jiayuan
#SBATCH --array=0-9
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --time=00:80:00
#SBATCH --mail-user=dylantroop@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

#!/bin/bash

dirname=$1

module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index -r requirements.txt

python n_arm_testbed.py $dirname
