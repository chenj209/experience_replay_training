:<<'ai002_test_Wang2022_precip_analysis'
CUDA_VISIABLE_DEVICES=2 python offline_test_newformat.py \
    --output_type 0-29 \
    --norm_type minmax_legacy \
    --start_ts 35040 \
    --sample_rate 12 \
    --multistep 0 \
    /cust_users/x-w19/nncam.ckpts/resmlp.2years.50epochs/0_29_nodesize512_num_blocks7_actrelu_bs1024_scheduler_coslr_lr0.001_ep50_noise0.0_wd0_dropout0/checkpoint.pth.tar \
    Wang2022_precip_analysis.json
ai002_test_Wang2022_precip_analysis
:<<'ai002_test_EP1117_precip_analysis'
CUDA_VISIABLE_DEVICES=2 python offline_test_newformat.py \
    --output_type 0-29 \
    --norm_type std \
    --start_ts 35040 \
    --sample_rate 12 \
    --multistep 1 \
    --inverse_output \
    --data_means ../consts/all_means.npz \
    --data_stds ../consts/all_stds.npz \
    /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1117_full/0_29_checkpoint_best_loss.pth.tar \
    EP1117_precip_analysis.json
ai002_test_EP1117_precip_analysis

#:<<'ai002_test_EP1117_precip_analysis'
CUDA_VISIABLE_DEVICES=2 python offline_test_newformat.py \
    --output_type 0-29 \
    --norm_type std \
    --start_ts 35040 \
    --sample_rate 12 \
    --multistep 1 \
    --inverse_output \
    --data_means ../consts/all_means.npz \
    --data_stds ../consts/all_stds.npz \
    /share3/chenj209/ckpts_sampled12/conv_mem/noreplay_buffer309_v2_seed1117_full_noprevQT/0_29_checkpoint_best_loss.pth.tar \
    NoEP1117_precip_analysis.json
#ai002_test_EP1117_precip_analysis
