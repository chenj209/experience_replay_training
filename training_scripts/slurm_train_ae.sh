#!/bin/bash
##SBATCH -A m4402
#SBATCH -A m4359
##SBATCH -C gpu
#SBATCH -C gpu&hbm80g
#SBATCH --qos=regular
##SBATCH --qos=shared
#SBATCH -t 10:00:00
#SBATCH -n 1
#SBATCH -c 128
##SBATCH -c 32
#SBATCH --gpus-per-task=4
##SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH -J vautoencoder_all_input
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
# Start memory usage monitoring in the background
memory_usage_file="memory_usage_$SLURM_JOB_ID.txt"
gpu_memory_usage_file="gpu_memory_usage_$SLURM_JOB_ID.txt"

# Log CPU Memory Usage
echo "Timestamp, Total, Used, Free, Shared, Buff/Cache, Available" > "$memory_usage_file"

# Log GPU Memory Usage
echo "Timestamp, GPU, Total, Used, Free" > "$gpu_memory_usage_file"

# Monitor CPU Memory Usage
while true; do
  echo "$(date '+%Y-%m-%d %H:%M:%S'), $(free -h | awk 'NR==2{printf "%s, %s, %s, %s, %s, %s\n", $2, $3, $4, $5, $6, $7}')" >> "$memory_usage_file"
  sleep 1
done &
cpu_monitor_pid=$!
# Monitor GPU Memory Usage
while true; do
  nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits | awk -v OFS=', ' '{$1=$1; print}' >> "$gpu_memory_usage_file"
  sleep 1
done &
gpu_monitor_pid=$!

srun --constraint=gpu --ntasks 1 -G 4 python scripts_train_ae_model_ex.py --ae_config $1 --region_mask ../consts/pacific_region_mask.npy --rec_weight 1.0 --pred_weight 0.0 --ex_input_prev $2 --ex_input $3
#srun --constraint=gpu --ntasks 1 -G 1 python scripts_train_ae_model_ex.py --ae_config $1 --region_mask ../consts/pacific_region_mask.npy --rec_weight 1.0 --pred_weight 0.0 --ex_input_prev $2 --ex_input $3
#srun --constraint=gpu --ntasks 1 -G 4 python scripts_train_ae_model_ex.py --ae_config $1 --region_mask ../consts/pacific_region_mask.npy --rec_weight 0.2 --pred_weight 0.8 --ex_input_prev $2 --ex_input $3 --resume /pscratch/sd/c/chenjd21/ckpts_time/ae_model029_sampled1_0301_multistep1_weight0.8_0.2_vae_configs/channel_512_256_128_64_latent4_sample1_lr0.0002_EX__EXPREV_/checkpoint_epoch40.pth.tar


kill $cpu_monitor_pid
kill $gpu_monitor_pid
