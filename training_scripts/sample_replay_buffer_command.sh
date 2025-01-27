#: <<'sample_train_test'
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_time_model_replay_buffer_noprevQT.py  \
    --data_dir /data/nncam_data/image_set/ \
    --ex_data_dir /data/chenj209/ex_dataset/ \
    --checkpoint /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v3_seed1117_sample2_noprevQT_std_mix1.0_rbs144/ \
    --norm_type std \
    --data_means ../consts/all_means.npz \
    --data_stds ../consts/all_stds.npz \
    --multistep 1 \
    --sample_stride 2 \
    --input_vars QL T_nn_in dqvls_nn_in dTls_nn_in SOLIN SPPS \
    --output_vars qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --input_vars_prev qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --train_batch 24 \
    --epoch 50 \
    --lr_strategy coslr \
    --manualSeed 1117 \
    --buffer_size 144 \
    --workers 24 \
    --mixing_ratio 1.0
#sample_train_test
: <<'sample_train_test'
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_time_model_replay_buffer_noprevQT.py  \
    --data_dir /data/nncam_data/image_set/ \
    --ex_data_dir /data/chenj209/ex_dataset/ \
    --checkpoint /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v3_seed1117_sample2_noprevQT_std_mix1.0_rbs24/ \
    --norm_type std \
    --data_means ../consts/all_means.npz \
    --data_stds ../consts/all_stds.npz \
    --multistep 1 \
    --sample_stride 2 \
    --input_vars QL T_nn_in dqvls_nn_in dTls_nn_in SOLIN SPPS \
    --output_vars qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --input_vars_prev qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --train_batch 24 \
    --epoch 50 \
    --lr_strategy coslr \
    --manualSeed 1117 \
    --buffer_size 24 \
    --workers 24 \
    --mixing_ratio 1.0
sample_train_test
: <<'sample_train_test'
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_time_model_replay_buffer_noprevQT.py  \
    --data_dir /data/nncam_data/image_set/ \
    --ex_data_dir /data/chenj209/ex_dataset/ \
    --checkpoint /share3/chenj209/ckpts_sampled12/conv_mem/noreplay_buffer309_v3_seed1117_sample2_noprevQT_std/ \
    --norm_type std \
    --data_means ../consts/all_means.npz \
    --data_stds ../consts/all_stds.npz \
    --multistep 1 \
    --sample_stride 2 \
    --input_vars QL T_nn_in dqvls_nn_in dTls_nn_in SOLIN SPPS \
    --output_vars qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --input_vars_prev qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --train_batch 24 \
    --epoch 50 \
    --lr_strategy coslr \
    --manualSeed 1117 \
    --buffer_size 0 \
    --workers 24 \
    --mixing_ratio 1.0
sample_train_test
: <<'sample_train_test'
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_time_model_replay_buffer_noprevQT.py  \
    --data_dir /data/nncam_data/image_set/ \
    --ex_data_dir /data/chenj209/ex_dataset/ \
    --checkpoint /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v3_seed1117_sample2_noprevQT_std_mix1.0_rbs24/ \
    --norm_type std \
    --multistep 1 \
    --sample_stride 2 \
    --input_vars QL T_nn_in dqvls_nn_in dTls_nn_in SOLIN SPPS \
    --output_vars qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --input_vars_prev qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --train_batch 24 \
    --epoch 50 \
    --lr_strategy coslr \
    --manualSeed 1117 \
    --buffer_size 24 \
    --workers 24 \
    --mixing_ratio 1.0
sample_train_test
: <<'sample_train_test'
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_time_model_replay_buffer_noprevQT.py  \
    --data_dir /data/nncam_data/image_set/ \
    --ex_data_dir /data/chenj209/ex_dataset/ \
    --checkpoint /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v3_seed1117_sample2_noprevQT_std_mix1.0_rbs144/ \
    --norm_type std \
    --multistep 1 \
    --sample_stride 2 \
    --input_vars QL T_nn_in dqvls_nn_in dTls_nn_in SOLIN SPPS \
    --output_vars qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --input_vars_prev qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --train_batch 24 \
    --epoch 50 \
    --lr_strategy coslr \
    --manualSeed 1117 \
    --buffer_size 144 \
    --workers 24 \
    --mixing_ratio 1.0
sample_train_test
: <<'sample_train_test'
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_time_model_replay_buffer_noprevQT.py  \
    --data_dir /data/nncam_data/image_set/ \
    --ex_data_dir /data/chenj209/ex_dataset/ \
    --checkpoint /share3/chenj209/ckpts_sampled12/conv_mem/noreplay_buffer309_v3_seed1117_sample2_noprevQT_minmax_mix1.0/ \
    --norm_type minmax_legacy \
    --multistep 1 \
    --sample_stride 2 \
    --input_vars QL T_nn_in dqvls_nn_in dTls_nn_in SOLIN SPPS \
    --output_vars qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --input_vars_prev qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --train_batch 24 \
    --epoch 50 \
    --lr_strategy coslr \
    --manualSeed 1117 \
    --buffer_size 0 \
    --workers 24 \
    --mixing_ratio 1.0
sample_train_test
: <<'sample_train_test'
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_time_model_replay_buffer_noprevQT.py  \
    --data_dir /data/nncam_data/image_set/ \
    --ex_data_dir /data/chenj209/ex_dataset/ \
    --checkpoint /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v3_seed1117_sample2_noprevQT_std_mix1.0/ \
    --norm_type std \
    --data_means ../consts/all_means.npz \
    --data_stds ../consts/all_stds.npz \
    --multistep 1 \
    --sample_stride 2 \
    --input_vars QL T_nn_in dqvls_nn_in dTls_nn_in SOLIN SPPS \
    --output_vars qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --input_vars_prev qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --train_batch 24 \
    --epoch 50 \
    --lr_strategy coslr \
    --manualSeed 1117 \
    --buffer_size 288 \
    --workers 24 \
    --mixing_ratio 1.0
sample_train_test
: <<'sample_train_test'
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_time_model_replay_buffer_noprevQT.py  \
    --data_dir /data/nncam_data/image_set/ \
    --ex_data_dir /data/chenj209/ex_dataset/ \
    --checkpoint /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v3_seed1117_sample2_noprevQT_minmax_mix1.0_rbs24/ \
    --norm_type minmax_legacy \
    --multistep 1 \
    --sample_stride 2 \
    --input_vars QL T_nn_in dqvls_nn_in dTls_nn_in SOLIN SPPS \
    --output_vars qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --input_vars_prev qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --train_batch 24 \
    --epoch 50 \
    --lr_strategy coslr \
    --manualSeed 1117 \
    --buffer_size 24 \
    --workers 24 \
    --mixing_ratio 1.0
sample_train_test
: <<'sample_train_test'
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_time_model_replay_buffer_noprevQT.py  \
    --data_dir /data/nncam_data/image_set/ \
    --ex_data_dir /data/chenj209/ex_dataset/ \
    --checkpoint /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v3_seed1117_sample2_noprevQT_minmax_mix1.0/ \
    --norm_type minmax_legacy \
    --multistep 1 \
    --sample_stride 2 \
    --input_vars QL T_nn_in dqvls_nn_in dTls_nn_in SOLIN SPPS \
    --output_vars qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --input_vars_prev qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --train_batch 24 \
    --epoch 50 \
    --lr_strategy coslr \
    --manualSeed 1117 \
    --buffer_size 288 \
    --workers 24 \
    --mixing_ratio 1.0
sample_train_test
: <<'sample_train_test'
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_time_model_replay_buffer_noprevQT.py  \
    --data_dir /data/nncam_data/image_set/ \
    --ex_data_dir /data/chenj209/ex_dataset/ \
    --checkpoint /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v3_seed1117_1year_noprevQT_minmax_mix1.0/ \
    --norm_type minmax_legacy \
    --multistep 1 \
    --sample_stride 1 \
    --input_vars QL T_nn_in dqvls_nn_in dTls_nn_in SOLIN SPPS \
    --output_vars qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --input_vars_prev qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --train_batch 24 \
    --epoch 50 \
    --lr_strategy coslr \
    --manualSeed 1117 \
    --buffer_size 288 \
    --end_ts 19500 \
    --workers 24 \
    --mixing_ratio 1.0
sample_train_test
: <<'sample_train_test'
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_time_model_replay_buffer_noprevQT.py  \
    --data_dir /data/nncam_data/image_set/ \
    --ex_data_dir /data/chenj209/ex_dataset/ \
    --checkpoint /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v3_seed1117_full_noprevQT_minmax_mix1.0_bs24/ \
    --norm_type minmax_legacy \
    --multistep 1 \
    --sample_stride 1 \
    --input_vars QL T_nn_in dqvls_nn_in dTls_nn_in SOLIN SPPS \
    --output_vars qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --input_vars_prev qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --train_batch 24 \
    --epoch 50 \
    --lr_strategy coslr \
    --manualSeed 1117 \
    --buffer_size 24 \
    --workers 24 \
    --mixing_ratio 1.0
sample_train_test
: <<'sample_train_test'
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_time_model_replay_buffer_noprevQT.py  \
    --data_dir /data/nncam_data/image_set/ \
    --ex_data_dir /data/chenj209/ex_dataset/ \
    --checkpoint /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v3_seed1117_full_noprevQT_minmax_mix1.0/ \
    --norm_type minmax_legacy \
    --multistep 1 \
    --sample_stride 1 \
    --input_vars QL T_nn_in dqvls_nn_in dTls_nn_in SOLIN SPPS \
    --output_vars qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --input_vars_prev qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --train_batch 24 \
    --epoch 50 \
    --lr_strategy coslr \
    --manualSeed 1117 \
    --buffer_size 288 \
    --workers 24 \
    --mixing_ratio 1.0
sample_train_test
: <<'sample_train_test'
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_time_model_replay_buffer_noprevQT.py  \
    --data_dir /data/nncam_data/image_set/ \
    --ex_data_dir /data/chenj209/ex_dataset/ \
    --checkpoint /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v3_seed1117_full_noprevQT_minmax_mix0.5/ \
    --norm_type minmax_legacy \
    --multistep 1 \
    --sample_stride 1 \
    --input_vars QL T_nn_in dqvls_nn_in dTls_nn_in SOLIN SPPS \
    --output_vars qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --input_vars_prev qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --train_batch 24 \
    --epoch 50 \
    --lr_strategy coslr \
    --manualSeed 1117 \
    --buffer_size 288 \
    --workers 24 \
    --mixing_ratio 0.5
sample_train_test
: <<'sample_train_test'
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_time_model_replay_buffer_noprevQTls_minmax.py \
    --data_dir /data/nncam_data/image_set/ \
    --ex_data_dir /data/chenj209/ex_dataset/ \
    --checkpoint /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1117_full_noprevQTls_perturbls_minmax/ \
    --norm_type minmax_legacy \
    --multistep 1 \
    --sample_stride 1 \
    --input_vars QL T_nn_in dqvls_nn_in dTls_nn_in SOLIN SPPS \
    --output_vars qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --input_vars_prev qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --train_batch 24 \
    --epoch 50 \
    --lr_strategy coslr \
    --manualSeed 1117 \
    --buffer_size 288 \
    --workers 24 \
    --mixing_ratio 1.0
sample_train_test
: <<'sample_train_test'
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_time_model_replay_buffer_noprevQT.py  \
    --data_dir /data/nncam_data/image_set/ \
    --ex_data_dir /data/chenj209/ex_dataset/ \
    --checkpoint /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v3_seed1117_full_noprevQT_minmax/ \
    --norm_type minmax_legacy \
    --multistep 1 \
    --sample_stride 1 \
    --input_vars QL T_nn_in dqvls_nn_in dTls_nn_in SOLIN SPPS \
    --output_vars qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --input_vars_prev qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --train_batch 24 \
    --epoch 50 \
    --lr_strategy coslr \
    --manualSeed 1117 \
    --buffer_size 288 \
    --workers 24 
sample_train_test
: <<'sample_train_test'
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_time_model_replay_buffer_noprevQT.py  \
    --data_dir /data/nncam_data/image_set/ \
    --ex_data_dir /data/chenj209/ex_dataset/ \
    --checkpoint /share3/chenj209/ckpts_sampled12/conv_mem/noreplay_buffer309_v2_seed1117_full_noprevQT_minmax_noFSDS/ \
    --norm_type minmax_legacy \
    --multistep 1 \
    --sample_stride 1 \
    --input_vars QL T_nn_in dqvls_nn_in dTls_nn_in SOLIN SPPS \
    --output_vars qtend_check stend_check SOLL SOLS SOLSD SOLLD \
    --input_vars_prev qtend_check stend_check SOLL SOLS SOLSD SOLLD \
    --train_batch 24 \
    --epoch 50 \
    --lr_strategy coslr \
    --manualSeed 1117 \
    --buffer_size 0 \
    --noFSDS \
    --workers 24 
sample_train_test
: <<'sample_train_test'
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_time_model_replay_buffer_noprevQT.py  \
    --data_dir /data/nncam_data/image_set/ \
    --ex_data_dir /data/chenj209/ex_dataset/ \
    --checkpoint /share3/chenj209/ckpts_sampled12/conv_mem/noreplay_buffer309_v2_seed1117_full_noprevQT_minmax/ \
    --norm_type minmax_legacy \
    --multistep 1 \
    --sample_stride 1 \
    --input_vars QL T_nn_in dqvls_nn_in dTls_nn_in SOLIN SPPS \
    --output_vars qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --input_vars_prev qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --train_batch 24 \
    --epoch 50 \
    --lr_strategy coslr \
    --manualSeed 1117 \
    --buffer_size 0 \
    --workers 24 
sample_train_test
: <<'sample_train_test'
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_time_model_replay_buffer_noprevQTls.py \
    --data_dir /data/nncam_data/image_set/ \
    --ex_data_dir /data/chenj209/ex_dataset/ \
    --checkpoint /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1117_full_noprevQTls_perturbls_mix0.5/ \
    --data_means ../consts/all_means.npz \
    --data_stds ../consts/all_stds.npz \
    --multistep 1 \
    --sample_stride 1 \
    --input_vars QL T_nn_in dqvls_nn_in dTls_nn_in SOLIN SPPS \
    --output_vars qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --input_vars_prev qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --train_batch 24 \
    --epoch 50 \
    --lr_strategy coslr \
    --manualSeed 1117 \
    --buffer_size 288 \
    --workers 24 \
    --mixing_ratio 0.5
sample_train_test
: <<'sample_train_test'
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_time_model_replay_buffer_noprevQT.py  \
    --data_dir /data/nncam_data/image_set/ \
    --ex_data_dir /data/chenj209/ex_dataset/ \
    --checkpoint /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1117_full_noprevQT_mix0.5/ \
    --data_means ../consts/all_means.npz \
    --data_stds ../consts/all_stds.npz \
    --multistep 1 \
    --sample_stride 1 \
    --input_vars QL T_nn_in dqvls_nn_in dTls_nn_in SOLIN SPPS \
    --output_vars qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --input_vars_prev qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --train_batch 24 \
    --epoch 50 \
    --lr_strategy coslr \
    --manualSeed 1117 \
    --buffer_size 288 \
    --mixing_ratio 0.5 \
    --workers 24 
sample_train_test
: <<'sample_train_test'
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_time_model_replay_buffer_noprevQT.py  \
    --data_dir /data/nncam_data/image_set/ \
    --ex_data_dir /data/chenj209/ex_dataset/ \
    --checkpoint /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1117_full_noprevQT/ \
    --data_means ../consts/all_means.npz \
    --data_stds ../consts/all_stds.npz \
    --multistep 1 \
    --sample_stride 1 \
    --input_vars QL T_nn_in dqvls_nn_in dTls_nn_in SOLIN SPPS \
    --output_vars qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --input_vars_prev qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --train_batch 24 \
    --epoch 50 \
    --lr_strategy coslr \
    --manualSeed 1117 \
    --buffer_size 288 \
    --workers 24 
sample_train_test
: <<'sample_train_test'
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_time_model_replay_buffer_noprevQT.py  \
    --data_dir /data/nncam_data/image_set/ \
    --ex_data_dir /data/chenj209/ex_dataset/ \
    --checkpoint /share3/chenj209/ckpts_sampled12/conv_mem/noreplay_buffer309_v2_seed1117_full_noprevQT/ \
    --data_means ../consts/all_means.npz \
    --data_stds ../consts/all_stds.npz \
    --multistep 1 \
    --sample_stride 1 \
    --input_vars QL T_nn_in dqvls_nn_in dTls_nn_in SOLIN SPPS \
    --output_vars qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --input_vars_prev qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --train_batch 24 \
    --epoch 50 \
    --lr_strategy coslr \
    --manualSeed 1117 \
    --buffer_size 0 \
    --workers 24 
sample_train_test
: <<'sample_train_test'
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_time_model_replay_buffer.py  \
    --data_dir /data/nncam_data/image_set/ \
    --ex_data_dir /data/chenj209/ex_dataset/ \
    --checkpoint /share3/chenj209/ckpts_sampled12/conv_mem/noreplay_buffer309_v2_seed1117_full/ \
    --data_means ../consts/all_means.npz \
    --data_stds ../consts/all_stds.npz \
    --multistep 1 \
    --sample_stride 1 \
    --input_vars QL T_nn_in dqvls_nn_in dTls_nn_in SOLIN SPPS \
    --output_vars qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --input_vars_prev qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --train_batch 24 \
    --epoch 50 \
    --lr_strategy coslr \
    --manualSeed 1117 \
    --buffer_size 0 \
    --workers 24 
sample_train_test
: <<'sample_train_test'
: <<'sample_train_test'
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_time_model_replay_buffer.py  \
    --data_dir /data/nncam_data/image_set/ \
    --ex_data_dir /data/chenj209/ex_dataset/ \
    --checkpoint /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1117_full/ \
    --data_means ../consts/all_means.npz \
    --data_stds ../consts/all_stds.npz \
    --multistep 1 \
    --sample_stride 1 \
    --input_vars QL T_nn_in dqvls_nn_in dTls_nn_in SOLIN SPPS \
    --output_vars qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --input_vars_prev qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --train_batch 24 \
    --epoch 50 \
    --lr_strategy coslr \
    --manualSeed 1117 \
    --workers 24 
sample_train_test
: <<'sample_train_test'
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
    --train_batch 24 \
    --epoch 50 \
    --lr_strategy coslr \
    --manualSeed 1 \
    --workers 24 \
    --resume /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1_full/ \
    --start_epoch 28
sample_train_test
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

