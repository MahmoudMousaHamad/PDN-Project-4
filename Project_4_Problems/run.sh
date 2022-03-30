#!/bin/bash

TESTS="../test_data"
TRANSACTIONS="../test_data/Problem 1 and 2/debug_1k.csv"
N_TRANSACTIONS="1000"

P1="./Problem_1"
P2="./Problem_2"
P3="./Problem_3"
P4="./Problem_4"

P1_RESULTS="./results/p1_results"
P2_RESULTS="./results/p2_results"
P3_RESULTS="./results/p3_results"
P4_RESULTS="./results/p4_results"

mkdir -p $P1_RESULTS $P2_RESULTS $P3_RESULTS $P4_RESULTS

# Problem 1
problem_1() {
    echo "################ Problem - 1 ################"
    make -C ./Problem_1/
    make -C ./Problem_1/serial

    variants=("serial_mining" "gpu_mining_starter" "gpu_mining_problem1")
    trials=(5000000 10000000)

    for variant in ${variants[@]}; do
        for t in ${trials[@]}; do
            echo "################ $variant - $t ################"
            out="$P1_RESULTS/out_${variant}_${t}.csv"
            time="$P1_RESULTS/time_${variant}_${t}.csv"

            $P1/$variant "$TRANSACTIONS" $N_TRANSACTIONS $t $out $time

            more $time
        done
    done
}

# Problem 2
problem_2() {
    echo "################ Problem - 2 ################"

    make -C ./Problem_2/

    variants=("gpu_mining_problem2")
    trials=(5000000 10000000)

    for variant in ${variants[@]}; do
        for t in ${trials[@]}; do
            echo "################ $variant - $t ################"
            out="$P2_RESULTS/out_${variant}_${t}.csv"
            time="$P2_RESULTS/time_${variant}_${t}.csv"

            $P2/$variant "$TRANSACTIONS" $N_TRANSACTIONS $t $out $time

            more $time
        done
    done
}

problem_3() {
    echo "################ Problem - 3 ################"

    make -C ./Problem_3/
    make -C ./Problem_3/serial

    variants=("serial" "CUDA")
    test_dir="$TESTS/Problem 3"
    dimension=2048

    for v in ${variants[@]}; do
        out="$P3_RESULTS/output_convolution_${v}.csv"
        time="$P3_RESULTS/time_${v}.csv"

        $P3/convolution_$v $dimension $dimension "$test_dir/mat_input.csv" $out $time

        echo "Time for $v: $(more $time)"
    done

    # echo Total time: $(more $time)\n
    # echo Number of lines: $(wc -l $out)\n
    # echo Difference: $(diff $out "$test_dir/mat_p3_output.csv")
}

problem_4() {
    echo "################ Problem - 4 ################"

    make -C ./Problem_4/
    make -C ./Problem_4/serial

    variants=("serial" "CUDA")
    test_dir="$TESTS/Problem 4"
    dimension=2048

    for v in ${variants[@]}; do
        out="$P4_RESULTS/output_convolution_maxpooling_${v}.csv"
        time="$P4_RESULTS/time_${v}.csv"

        $P4/convolution_maxpooling_$v $dimension $dimension "$test_dir/mat_input.csv" $out $time

        echo "Time for $v: $(more $time)s"
    done

    # echo Total time: $(more $time)
    # echo Number of lines: $(wc -l $out)
    # # tail $out
    # echo Difference: $(diff $out "$test_dir/mat_p4_output.csv")
}

git pull

# problem_1
# problem_2
problem_3
problem_4
