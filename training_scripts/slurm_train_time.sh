#!/bin/bash
##SBATCH -A m4402
#SBATCH -A m4359
#SBATCH -C gpu
#SBATCH --qos=shared
#SBATCH -t 8:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH -J baseline_time
#SBATCH -o %x_%j.out

module load cudatoolkit/11.7
module load cudnn/8.9.1_cuda11
module load python
export SLURM_CPU_BIND="cores"
#source /global/homes/c/chenjd21/setup_conda.csh
conda activate mpi4py2
#conda activate pytorch1
python -c "import torch; print(torch.zeros(1).cuda())"
python -c "import torch; print(torch.cuda.is_available())"
echo "ex_input_prev" $1 "ex_input" $2
srun --constraint=gpu --ntasks 1 -G 1 python scripts_train_time_model029_ex.py  --region_mask ../consts/pacific_region_mask.npy --ex_input $2 --ex_input_prev $1
