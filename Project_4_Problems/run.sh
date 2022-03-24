#!/bin/bash

TRANSACTIONS="../test_data/Problem 1\ and\ 2/in_20k.csv"
N_TRANSACTIONS="20000"

P1_RESULTS="./results/p1_results/"

mkdir -p P1_RESULTS

# Problem 1
make -C ./Problem_1/
make -C ./Problem_1/serial

variants=("./serial_mining" "./gpu_mining_starter" "gpu_mining_problem1")
trials=(5000000 10000000)

for variant in ${variants[@]}; do
    for t in ${trials[@]}; do
        out="$P1_RESULTS/out_${variant}_${t}.csv"
        time="$P1_RESULTS/time_${variant}_${t}.csv"
        
        $variant $TRANSACTIONS $N_TRANSACTIONS $t $out $time

        more $time
    done
done