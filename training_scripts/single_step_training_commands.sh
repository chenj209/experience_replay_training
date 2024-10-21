: <<'sample_train_test'
python train_time_model_legacy_format.py  \
    --data_dir ../dataloader/data \
    --ex_data_dir ../dataloader/ex_data \
    --checkpoint ckpts_sampled12/conv_mem/test_legacy_training \
    --data_means ../consts/all_means.npz \
    --data_stds ../consts/all_stds.npz \
    --multistep 1 \
    --sample_rate 12 \
    --input_vars QL T_nn_in dqvls_nn_in dTls_nn_in SOLIN SPPS UL VL LWUP \
    --output_vars qtend_check \
    --input_vars_prev FLNS FLNT qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --lr_strategy coslr \
    --network resnet_output30
sample_train_test

: <<'sampled12_qtend_legacy_training_with_new_vars'
python train_time_model_legacy_format.py  \
    --data_dir /data/nncam_data/image_set/ \
    --ex_data_dir /data/chenj209/ex_dataset/ \
    --checkpoint /share3/chenj209/ckpts_sampled12/conv_mem/baseline_model029_sampled12_multistep1_new_vars_region_all/ \
    --data_means ../consts/all_means.npz \
    --data_stds ../consts/all_stds.npz \
    --multistep 1 \
    --sample_rate 12 \
    --input_vars QL T_nn_in dqvls_nn_in dTls_nn_in SOLIN SPPS UL VL LWUP \
    --output_vars qtend_check \
    --input_vars_prev FLNS FLNT qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --lr_strategy coslr \
    --network resnet_output30 \
    --train_batch 32 \
    --worker 4
sampled12_qtend_legacy_training_with_new_vars

: <<'sampled12_qtend_legacy_training_with_new_vars_CAPE_CLOUD_SPPRECC'
python train_time_model_legacy_format.py  \
    --data_dir /data/nncam_data/image_set/ \
    --ex_data_dir /data/chenj209/ex_dataset/ \
    --checkpoint /share3/chenj209/ckpts_sampled12/conv_mem/baseline_model3059_sampled12_multistep1_new_vars_CCS_region_all/ \
    --data_means ../consts/all_means.npz \
    --data_stds ../consts/all_stds.npz \
    --multistep 1 \
    --sample_rate 12 \
    --input_vars QL T_nn_in dqvls_nn_in dTls_nn_in SOLIN SPPS UL VL LWUP CAPE \
    --output_vars stend_check \
    --input_vars_prev FLNS FLNT CLOUD SPPRECC qtend_check stend_check SOLL SOLS SOLSD SOLLD FSDS \
    --lr_strategy coslr \
    --network resnet_output30 \
    --train_batch 32 \
    --worker 4
sampled12_qtend_legacy_training_with_new_vars_CAPE_CLOUD_SPPRECC

: <<'sampled12_single'
CUDA_VISIBLE_DEVICES=2 python train_time_model_legacy_format.py  \
    --data_dir /data/nncam_data/image_set/ \
    --ex_data_dir /data/chenj209/ex_dataset/ \
    --checkpoint /share3/chenj209/ckpts_sampled12/no_conv_mem/baseline_model029_sampled12_multistep0_seed1215_region_all/ \
    --data_means ../consts/all_means.npz \
    --data_stds ../consts/all_stds.npz \
    --multistep 0 \
    --sample_rate 12 \
    --input_vars QL T_nn_in dqvls_nn_in dTls_nn_in SOLIN SPPS \
    --output_vars qtend_check \
    --lr_strategy coslr \
    --network resnet_output30 \
    --train_batch 32 \
    --worker 4 \
    --manualSeed 1215

CUDA_VISIBLE_DEVICES=2 python train_time_model_legacy_format.py  \
    --data_dir /data/nncam_data/image_set/ \
    --ex_data_dir /data/chenj209/ex_dataset/ \
    --checkpoint /share3/chenj209/ckpts_sampled12/no_conv_mem/baseline_model3059_sampled12_multistep0_seed1215_region_all/ \
    --data_means ../consts/all_means.npz \
    --data_stds ../consts/all_stds.npz \
    --multistep 0 \
    --sample_rate 12 \
    --input_vars QL T_nn_in dqvls_nn_in dTls_nn_in SOLIN SPPS \
    --output_vars stend_check \
    --lr_strategy coslr \
    --network resnet_output30 \
    --worker 4 \
    --manualSeed 1215

CUDA_VISIBLE_DEVICES=2 python train_time_model_legacy_format.py  \
    --data_dir /data/nncam_data/image_set/ \
    --ex_data_dir /data/chenj209/ex_dataset/ \
    --checkpoint /share3/chenj209/ckpts_sampled12/no_conv_mem/baseline_model6165_sampled12_multistep0_seed1117_region_all/ \
    --data_means ../consts/all_means.npz \
    --data_stds ../consts/all_stds.npz \
    --multistep 0 \
    --sample_rate 12 \
    --input_vars QL T_nn_in dqvls_nn_in dTls_nn_in SOLIN SPPS \
    --output_vars SOLL SOLS SOLSD SOLLD FSDS \
    --lr_strategy coslr \
    --network resnet_output5 \
    --train_batch 32 \
    --worker 4 \
    --manualSeed 1117
sampled12_single
#: <<'sampled12_single_minmax'
CUDA_VISIBLE_DEVICES=2 python train_time_model_legacy_format.py  \
    --data_dir /data/nncam_data/image_set/ \
    --ex_data_dir /data/chenj209/ex_dataset/ \
    --checkpoint /share3/chenj209/ckpts_sampled12/no_conv_mem/baseline_model029_sampled12_multistep0_seed1117_minmax_region_all/ \
    --multistep 0 \
    --sample_rate 12 \
    --input_vars QL T_nn_in dqvls_nn_in dTls_nn_in SOLIN SPPS \
    --output_vars qtend_check \
    --lr_strategy coslr \
    --network resnet_output30 \
    --train_batch 32 \
    --worker 4 \
    --manualSeed 1117 \
    --minmax_norm

CUDA_VISIBLE_DEVICES=2 python train_time_model_legacy_format.py  \
    --data_dir /data/nncam_data/image_set/ \
    --ex_data_dir /data/chenj209/ex_dataset/ \
    --checkpoint /share3/chenj209/ckpts_sampled12/no_conv_mem/baseline_model3059_sampled12_multistep0_seed1117_minmax_region_all/ \
    --multistep 0 \
    --sample_rate 12 \
    --input_vars QL T_nn_in dqvls_nn_in dTls_nn_in SOLIN SPPS \
    --output_vars stend_check \
    --lr_strategy coslr \
    --network resnet_output30 \
    --worker 4 \
    --manualSeed 1117 \
    --minmax_norm

CUDA_VISIBLE_DEVICES=2 python train_time_model_legacy_format.py  \
    --data_dir /data/nncam_data/image_set/ \
    --ex_data_dir /data/chenj209/ex_dataset/ \
    --checkpoint /share3/chenj209/ckpts_sampled12/no_conv_mem/baseline_model6165_sampled12_multistep0_seed1117_minmax_region_all/ \
    --multistep 0 \
    --sample_rate 12 \
    --input_vars QL T_nn_in dqvls_nn_in dTls_nn_in SOLIN SPPS \
    --output_vars SOLL SOLS SOLSD SOLLD FSDS \
    --lr_strategy coslr \
    --network resnet_output5 \
    --train_batch 32 \
    --worker 4 \
    --manualSeed 1117 \
    --minmax_norm
#sampled12_single_minmax
