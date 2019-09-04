#!/bin/bash

dirname=$1

mkdir -p fraction_closer/$dirname/sa fraction_closer/$dirname/ev fraction_closer/plots

job_ID=$(sbatch --parsable fraction_closer_search.sh $dirname)
sbatch --depend=afterany:${job_ID} fraction_closer_plots.sh $dirname