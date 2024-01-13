#!/bin/bash
#SBATCH -A m4402
#SBATCH -C gpu
#SBATCH --qos=regular
#SBATCH -t 8:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none

module load cudatoolkit/11.7
module load cudnn/8.9.1_cuda11
module load python
export SLURM_CPU_BIND="cores"
#source /global/homes/c/chenjd21/setup_conda.csh
#conda activate /global/homes/c/chenjd21/.conda/envs/mpi4py2
conda activate pytorch1
python -c "import torch; print(torch.zeros(1).cuda())"
python -c "import torch; print(torch.cuda.is_available())"
srun --constraint=gpu --ntasks 1 -G 1 python scripts_train_time_model029_ex.py --multistep 1 --ex_input CLOUD
