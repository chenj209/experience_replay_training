#!/bin/bash
#SBATCH -A m4359
#SBATCH -C gpu
#SBATCH --qos=regular
#SBATCH -t 08:00:00
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
srun --constraint=gpu --ntasks 1 -G 1 python ../offline_test_region_gen_preds.py modelwithps.json test_metric.json --sample 1 --region_mask all --save_path /pscratch/sd/c/chenjd21/resmlp_pred/
