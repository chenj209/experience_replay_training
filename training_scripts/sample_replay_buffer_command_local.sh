#: <<'sample_train_test'
python train_time_model_replay_buffer_noprevQT_nersc.py  \
    --data_dir ../dataloader/spcam_new_data_local/ \
    --checkpoint local_ckpts_test/ \
    --norm_type std \
    --data_means ../consts/all_means.npz \
    --data_stds ../consts/all_stds.npz \
    --multistep 1 \
    --sample_stride 2 \
    --input_vars QL T_nn_in dqvls_nn_in dTls_nn_in SOLIN SPPS \
    --output_vars qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --input_vars_prev qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --train_batch 1 \
    --epoch 2 \
    --lr_strategy coslr \
    --manualSeed 1117 \
    --buffer_size 144 \
    --workers 24 \
    --mixing_ratio 1.0
#sample_train_test