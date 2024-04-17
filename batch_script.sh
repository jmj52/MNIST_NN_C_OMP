#!/bin/bash

# script
# variant="test"
variant="mnist"

# Path to tbd script
script_path="./$variant"

# Output directory
output_dir="results_$variant"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"
   
# Set nodes, tasks, threads
nodes=1
tasks=1
min_arg=1
max_arg=16 # max threads

# Loop from 1 to max_arg in powers of 2
for ((i=$min_arg; i<=max_arg; i*=2))
do
    # nodes=1
    # tasks=$i
    thr=$i
    suffix="n$nodes""_t$tasks""_thr$thr"

    export OMP_NUM_THREADS=$i
    output_file="$output_dir/time_$suffix.txt"
    call="srun -A c00698 -p general --nodes=$nodes --ntasks-per-node=$tasks ./$variant"
    
    echo "Running $script_path with OMP_NUM_THREADS=$OMP_NUM_THREADS, saving time info to $output_file" 
    /usr/bin/time -o "$output_file" -f "Time: %E\nCPU: %P\nMem: %M KB\nSts: %x" $call > "$output_dir/output_$suffix.txt"
done

echo "Script execution completed."

