#!/bin/bash

for d in data/*; do
	dirname=$(echo $d | cut -d '/' -f2)
    python fraction_closer.py $dirname
done

