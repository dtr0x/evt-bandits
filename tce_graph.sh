#!/bin/bash

for arg in "$@"
do
    k=$(echo $arg | cut -f1 -d=)
    v=$(echo $arg | cut -f2 -d=)   

    case $k in
        dist) dist=$v ;;
        p1) p1=$v ;;
        p2) p2=$v ;;
        alph) alph=$v ;;
        tp_select) tp_select=$v ;;
        tp) tp=$v ;;
        tp_init) tp_init=$v ;;
        tp_num) tp_num=$v ;;
        signif) signif=$v ;;
        *)   
    esac    
done

if [ "$tp_select" == "fixed" ]; then
    dirname="${dist}_${p1}_${p2}_${alph}_${tp_select}_${tp}"
elif [ "$tp_select" == "search" ]; then
    dirname="${dist}_${p1}_${p2}_${alph}_${tp_select}_${tp_init}_${tp_num}_${signif}"
fi

mkdir -p data/$dirname/sa data/$dirname/ev plots

job_ID=$(sbatch --output=/dev/null --parsable run_sim.sh $dirname)
sbatch --output=/dev/null --depend=afterany:${job_ID} make_plots.sh $dirname