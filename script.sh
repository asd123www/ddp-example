#!/bin/bash

# Script to run distributed training
# Usage: ./script.sh

# Run the script for 5 times and log the output to different .txt files
for i in {1..5}; do
    torchrun --nproc_per_node=4 main.py > output_$i.txt 2>&1
done

echo "Done running the script for 5 times"
sleep 5

# compare the outputs check if they are the same
for i in {1..5}; do
diff output_1.txt output_$i.txt
done