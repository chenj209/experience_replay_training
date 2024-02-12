#!/bin/bash
##SBATCH -A m4402
#SBATCH -A m4359
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH -t 2:00:00
#SBATCH -n 1
#SBATCH -c 32
##SBATCH --gpus-per-task=1
##SBATCH --gpu-bind=none
#SBATCH -J check_data
#SBATCH -o %x_%j.out

module load cudatoolkit/11.7
module load cudnn/8.9.1_cuda11
module load python
export SLURM_CPU_BIND="cores"
#source /global/homes/c/chenjd21/setup_conda.csh
conda activate mpi4py2
#conda activate pytorch1
#python -c "import torch; print(torch.zeros(1).cuda())"
#python -c "import torch; print(torch.cuda.is_available())"
#echo "region" $1 "ex_input" $2
srun --constraint=cpu --ntasks 1 python check_data.py
