#: <<'sample_train_test'
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_time_model_replay_buffer.py  \
    --data_dir /data/nncam_data/image_set/ \
    --ex_data_dir /data/chenj209/ex_dataset/ \
    --checkpoint /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1_full/ \
    --data_means ../consts/all_means.npz \
    --data_stds ../consts/all_stds.npz \
    --multistep 1 \
    --sample_stride 1 \
    --input_vars QL T_nn_in dqvls_nn_in dTls_nn_in SOLIN SPPS \
    --output_vars qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --input_vars_prev qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --train_batch 48 \
    --epoch 50 \
    --lr_strategy coslr \
    --manualSeed 1 \
    --workers 24 \
    --resume /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1_full/ \
    --start_epoch 28
#sample_train_test
: <<'sample_train_test_small'
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_time_model_replay_buffer.py  \
    --data_dir /data/nncam_data/image_set/ \
    --ex_data_dir /data/chenj209/ex_dataset/ \
    --checkpoint /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1_test/ \
    --data_means ../consts/all_means.npz \
    --data_stds ../consts/all_stds.npz \
    --multistep 1 \
    --sample_stride 144 \
    --input_vars QL T_nn_in dqvls_nn_in dTls_nn_in SOLIN SPPS \
    --output_vars qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --input_vars_prev qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --train_batch 48 \
    --epoch 50 \
    --lr_strategy coslr \
    --manualSeed 1 \
    --workers 24 \
    --resume /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1_full/ \
    --start_epoch 28
sample_train_test

: <<'sample_train_test'
python train_time_model_replay_buffer.py  \
    --data_dir ../dataloader/data \
    --ex_data_dir ../dataloader/ex_data \
    --checkpoint ckpts_sampled12/conv_mem/test_all_models \
    --data_means ../consts/all_means.npz \
    --data_stds ../consts/all_stds.npz \
    --multistep 1 \
    --sample_stride 12 \
    --input_vars QL T_nn_in dqvls_nn_in dTls_nn_in SOLIN SPPS \
    --output_vars qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --input_vars_prev qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --train_batch 2 \
    --lr_strategy coslr 
sample_train_test

