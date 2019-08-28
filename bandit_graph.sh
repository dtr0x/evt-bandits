#!/bin/bash

for arg in "$@"
do
    k=$(echo $arg | cut -f1 -d=)
    v=$(echo $arg | cut -f2 -d=)   

    case $k in
        dist) dist=$v ;;
        p_min) p_min=$v ;;
        p_max) p_max=$v ;;
        narms) narms=$v ;;
        alph) alph=$v ;;
        *)   
    esac    
done

dirname="${dist}_${p_min}_${p_max}_${narms}_${alph}"

mkdir -p data/bandits/$dirname/sa data/bandits/$dirname/ev plots/bandits

job_ID=$(sbatch --output=/dev/null --parsable n_arm_testbed.sh $dirname)
sbatch --output=/dev/null --depend=afterany:${job_ID} bandit_plots.sh $dirname