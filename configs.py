import os
#CKPT_DIR = "/pscratch/sd/c/chenjd21/ckpts_time/"
#CKPT_DIR = "ckpts_time/"
CKPT_DIR = "/share3/chenj209/ckpts_time/"
DATA_DIR = "/zz/share3/chenj209/spcam_new_data_32/"
if not os.path.exists(DATA_DIR):
    DATA_DIR = "/Users/jiandachen/Projects/NNCAM_packages/nncam_training/analysis/test_data/"
SAMPLE_RATE = 7
BASELINE_CKPT_PREFIX = "baseline_model_coslr"
VAE_CKPT_PREFIX = "vae_model"
