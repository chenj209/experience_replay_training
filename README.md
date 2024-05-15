# NNCAM_Training

## Training (training_scripts)

### Setup

* Edit `consts.py`:
    * `CKPT_DIR`: directory to save model checkpoints (need to be created manually)
    * `DATA_DIR`: directory containing `*.npy` files and `col_names.txt`
    * `SAMPLE_RATE`: 12 (6 hours)/1 (30 minutes)
    * `BASELINE_CKPT_PREFIX`: prefix of output checkpoint directory for baseline training
    * `VAE_CKPT_PREFIX`: prefix of output checkpoint directory for vae training

### Baseline training

* Usage
```
usage: scripts_train_time_model029_ex.py [-h] [--multistep MULTISTEP]
                                         [--lr LR] [--region_mask REGION_MASK]
                                         [--data_means DATA_MEANS]
                                         [--data_stds DATA_STDS]
                                         [--ex_input_prev [EX_INPUT_PREV]]
                                         [--ex_input [EX_INPUT]]
                                         [--resume RESUME]

optional arguments:
  -h, --help            show this help message and exit
  --multistep MULTISTEP
                        number of previous timesteps to be used as inputs
  --lr LR               learning rate (default to 0.001)
  --region_mask REGION_MASK
                        default to all 96x144 grid, otherwirse path to a
                        region mask (in ../consts folder)
  --data_means DATA_MEANS
                        path to npz files that stores means for all variables
                        (in ../consts folder), default to
                        ../consts/all_means.npz
  --data_stds DATA_STDS
                        path to npz files that stores stds for all variables
                        (in ../consts folder), default to
                        ../consts/all_stds.npz
  --ex_input_prev [EX_INPUT_PREV]
                        input variables to be added for previous timesteps (in
                        addition to Q,T,dTls,dqls,solin,ps,qtend,stend,SOLL,SO
                        LLD,SOLS,SOLSD,FSDS
  --ex_input [EX_INPUT]
                        input variables to be added for previous timesteps (in
                        addition to Q,T,dTls,dqls,solin,ps
  --resume RESUME       resume from saved checkpoints training
```

* Examples
    * Default training setting for 96x144 grids
        | | Input variables|
        |:-------------------------------------:|:-------------------------------------:|
        | Current timestep | Q,T,dqls,dTls,solin,ps                |
        |Previous timestep | Q,T,dqls,dTls,solin,ps,qtend,stend,SOLL,SOLLD,SOLS,SOLSD,FSDS|
    ```
    python scripts_train_time_model029_ex.py
    ```
    * Training with full variables in sea region only 
        | | Input variables|
        |:-------------------------------------:|:-------------------------------------:|
        | Current timestep | Q,T,dqls,dTls,solin,ps,LWUP,CAPE,U,V|
        |Previous timestep | Q,T,dqls,dTls,solin,ps,LWUP,CAPE,U,V,qtend,stend,SOLL,SOLLD,SOLS,SOLSD,FSDS,CLOUD,SPPRECC,FLNS,FLNT,SPQRL,SPQRS|
    ```
    python scripts_train_time_model029_ex.py 
        --region_mask ../consts/seamask.npy 
        --data_means ../consts/seamask_means.npz 
        --data_stds ../consts/seamask_stds.npz
        --ex_input LWUP+CAPE+U+V
        --ex_input_prev CLOUD+SPPRECC+FLNS+FLNT+SPQRL+SPQRS
    ```



