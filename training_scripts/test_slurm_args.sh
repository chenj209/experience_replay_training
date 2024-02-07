#!/bin/bash
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH -t 0:00:05
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=cpu
#SBATCH -o test_args.out
#SBATCH -J test_args


# Accessing arguments
arg1=$1
arg2=$2

# Rest of your script
echo "Argument 1 is $arg1"
echo "Argument 2 is $arg2"

