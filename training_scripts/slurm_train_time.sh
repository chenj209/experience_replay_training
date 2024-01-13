#!/bin/bash
#SBATCH -A m4359
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
conda activate mpi4py2
python -c "import torch; print(torch.zeros(1).cuda())"
python -c "import torch; print(torch.cuda.is_available())"
srun --constraint=gpu --ntasks 1 -G 1 python scripts_train_time_model029.py --multistep 1
