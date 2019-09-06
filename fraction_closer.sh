#!/bin/bash

dirname=$1

mkdir -p fraction_closer/$dirname/sa fraction_closer/$dirname/ev fraction_closer/plots

job_ID=$(sbatch --output=/dev/null --parsable fraction_closer_search.sh $dirname)
sbatch --output=/dev/null --depend=afterany:${job_ID} fraction_closer_plots.sh $dirname