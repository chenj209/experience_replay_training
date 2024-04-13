#!/bin/bash

#SBATCH -A m4359
#SBATCH -C gpu
#SBATCH --qos=shared
#SBATCH -t 0:10:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none

# Function to monitor CPU and memory usage of the main process
# Start memory usage monitoring in the background
memory_usage_file="memory_usage_$SLURM_JOB_ID.txt"

echo "Timestamp, Total, Used, Free, Shared, Buff/Cache, Available" > "$memory_usage_file"

# Loop by running the `free` command every second and append the output to the file
while true; do
  echo "$(date '+%Y-%m-%d %H:%M:%S'), $(free -h | awk 'NR==2{printf "%s, %s, %s, %s, %s, %s\n", $2, $3, $4, $5, $6, $7}')" >> "$memory_usage_file"
  sleep 1
done &
monitor_pid=$!

# Start your main job in the background and get its PID
srun --constraint=gpu --ntasks 1 -G 1 python ../models/autoencoder.py

kill $monitor_pid
