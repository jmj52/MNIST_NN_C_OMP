#!/bin/bash
working_directory=$(pwd)

# Script
variant="main"

# Path to tbd script
script_path="./$variant"

# Output directory
output_dir="results/$variant"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"
   
# Set nodes, tasks, threads
nodes=1
tasks=1
min_arg=1
max_arg=1 # max threads

# Run parameters
mem="16G"

# Loop from 1 to max_arg in powers of 2
for ((thr=$min_arg; thr<=max_arg; thr*=2))
do
    # tasks=$thr
    suffix="n$nodes""_t$tasks""_thr$thr"

    # environment setup
    export OMP_NUM_THREADS=$thr

    # create intermediate directories
    mkdir -p "testing_net"

    # Define output files and running args
    time_file="$working_directory/$output_dir/time_$suffix.txt"
    log_file="$working_directory/$output_dir/output_$suffix.txt"
    call="$working_directory/$variant $thr"
    # call="srun -A c00698 -p general --mem=$mem --nodes=$nodes --ntasks-per-node=$tasks $working_directory/$variant $thr"
    
    # Run and save output to log file, no time file used
    echo "Running $variant with OMP_NUM_THREADS=$OMP_NUM_THREADS, saving output info to $log_file" 
    $call > $log_file

    # Run and save output to log file, time info saved to time_file. 
    # This has issues as the time_info varies depending on when resources get allocated
    # TODO: Log time accurately to output file
    # echo "Running $variant with OMP_NUM_THREADS=$OMP_NUM_THREADS, saving time info to $time_file" 
    # /usr/bin/time -o "$time_file" -f "Time: %E\nCPU: %P\nMem: %M KB\nSts: %x" $call > "$output_dir/output_$suffix.txt"
done

echo "Script execution completed."

