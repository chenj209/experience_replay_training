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
#SBATCH -J autoencoder_baseline
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
srun --constraint=gpu --ntasks 1 -G 1 python scripts_train_ae_model_ex.py --ae_config $1 --region_mask ../consts/pacific_region_mask.npy --rec_weight 0.2 --pred_weight 0.8 --ex_input_prev $2 --ex_input $3
