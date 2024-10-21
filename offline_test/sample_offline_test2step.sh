:<<'local_test'
python offline_test_newformat_2step.py \
    --model029 ../training_scripts/ckpts_sampled12/conv_mem/baseline_model029_sampled12_multistep1_EX__EXPREV__region_all/checkpoint_epoch35.pth.tar \
    --model3059 ../training_scripts/ckpts_sampled12/conv_mem/baseline_model3059_sampled12_multistep1_EX__EXPREV__region_all/checkpoint_epoch35.pth.tar \
    --model6165 ../training_scripts/ckpts_sampled12/conv_mem/baseline_model6165_sampled12_multistep1_EX__EXPREV__region_all/checkpoint_epoch35.pth.tar \
    --train_configs ../training_scripts/ckpts_sampled12/conv_mem/baseline_model029_sampled12_multistep1_EX__EXPREV__region_all/configs.txt \
    --start_ts 0 \
    test_out.json
local_test

:<<'ai002_test'
python offline_test_newformat_2step.py \
    --model029 /share3/chenj209/ckpts_sampled12/conv_mem/baseline_model029_sampled12_multistep1_EX__EXPREV__region_all/checkpoint_epoch35.pth.tar \
    --model3059 /share3/chenj209/ckpts_sampled12/conv_mem/baseline_model3059_sampled12_multistep1_EX__EXPREV__region_all/checkpoint_epoch35.pth.tar \
    --model6165 /share3/chenj209/ckpts_sampled12/conv_mem/baseline_model6165_sampled12_multistep1_EX__EXPREV__region_all/checkpoint_epoch35.pth.tar \
    --train_configs /share3/chenj209/ckpts_sampled12/conv_mem/baseline_model029_sampled12_multistep1_EX__EXPREV__region_all/configs.txt \
    --start_ts 35040 \
    --sample_rate 12 \
    --multistep 2 \
    conv_offline_2step.json
ai002_test

:<<'ai002_test_replay_buffer'
python offline_test_newformat_2step.py \
    --model029 /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2/0_29_checkpoint_best_loss.pth.tar \
    --model3059 /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2/30_59_checkpoint_best_loss.pth.tar \
    --model6165 /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2/61_65_checkpoint_best_loss.pth.tar \
    --train_configs /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2/configs.txt \
    --start_ts 35040 \
    --sample_rate 12 \
    --multistep 2 \
    --legacy_order \
    replay_buffer_2step_v2.json
ai002_test_replay_buffer

:<<'ai002_test_replay_buffer'
CUDA_VISIABLE_DEVICES=2 python offline_test_newformat_2step.py \
    --model029 /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1117/0_29_checkpoint_best_loss.pth.tar \
    --model3059 /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1117/30_59_checkpoint_best_loss.pth.tar \
    --model6165 /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1117/61_65_checkpoint_best_loss.pth.tar \
    --train_configs /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1117/configs.txt \
    --start_ts 35040 \
    --sample_rate 12 \
    --multistep 2 \
    --legacy_order \
    replay_buffer_2step_v2_seed1117.json
ai002_test_replay_buffer
:<<'ai002_test_replay_buffer'
CUDA_VISIABLE_DEVICES=2 python offline_test_newformat_2step.py \
    --model029 /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1215_full/0_29_checkpoint_best_loss.pth.tar \
    --model3059 /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1215_full//30_59_checkpoint_best_loss.pth.tar \
    --model6165 /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1215_full//61_65_checkpoint_best_loss.pth.tar \
    --train_configs /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1215_full/configs.txt \
    --start_ts 35040 \
    --sample_rate 12 \
    --multistep 2 \
    --legacy_order \
    replay_buffer_2step_v2_seed1215_full.json
ai002_test_replay_buffer
:<<'ai002_test_noreplay_buffer_noprevQT'
CUDA_VISIABLE_DEVICES=2 python offline_test_newformat_2step.py \
    --model029 /share3/chenj209/ckpts_sampled12/conv_mem/noreplay_buffer309_v2_seed1117_full_noprevQT/0_29_checkpoint_best_loss.pth.tar \
    --model3059 /share3/chenj209/ckpts_sampled12/conv_mem/noreplay_buffer309_v2_seed1117_full_noprevQT/30_59_checkpoint_best_loss.pth.tar \
    --model6165 /share3/chenj209/ckpts_sampled12/conv_mem/noreplay_buffer309_v2_seed1117_full_noprevQT/61_65_checkpoint_best_loss.pth.tar \
    --train_configs /share3/chenj209/ckpts_sampled12/conv_mem/noreplay_buffer309_v2_seed1117_full_noprevQT/configs.txt \
    --start_ts 35040 \
    --sample_rate 12 \
    --multistep 2 \
    --legacy_order \
    --no_prevQT \
    noreplay_buffer_2step_v2_seed1117_full_noprevQT.json
ai002_test_noreplay_buffer_noprevQT
:<<'ai002_test_replay_buffer_noprevQT_full'
CUDA_VISIABLE_DEVICES=2 python offline_test_newformat_2step.py \
    --model029 /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1117_full_noprevQT/0_29_checkpoint_best_loss.pth.tar \
    --model3059 /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1117_full_noprevQT/30_59_checkpoint_best_loss.pth.tar \
    --model6165 /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1117_full_noprevQT/61_65_checkpoint_best_loss.pth.tar \
    --train_configs /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1117_full_noprevQT/configs.txt \
    --start_ts 35040 \
    --sample_rate 12 \
    --multistep 2 \
    --legacy_order \
    --no_prevQT \
    replay_buffer_2step_v2_seed1117_full_noprevQT.json
ai002_test_replay_buffer_noprevQT_full
#:<<'ai002_test_replay_buffer'
CUDA_VISIABLE_DEVICES=2 python offline_test_newformat_2step_baseline.py \
    --model029 /cust_users/x-w19/nncam.ckpts/resmlp.2years.50epochs/0_29_nodesize512_num_blocks7_actrelu_bs1024_scheduler_coslr_lr0.001_ep50_noise0.0_wd0_dropout0/checkpoint.pth.tar \
    --model3059 /cust_users/x-w19/nncam.ckpts/resmlp.25GB.noise0.0/30_59_nodesize512_num_blocks7_actrelu_bs1024_scheduler_coslr_lr0.001_ep50_noise0.0_wd0_dropout0/checkpoint.pth.tar \
    --model6165 /cust_users/x-w19/nncam.ckpts/resmlp.2years.50epochs/61_64_nodesize512_num_blocks7_actrelu_bs1024_scheduler_coslr_lr0.001_ep50_noise0.0_wd0_dropout0/checkpoint.pth.tar \
    --train_configs /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1215_full/configs.txt \
    --start_ts 35040 \
    --sample_rate 12 \
    --multistep 2 \
    --legacy_order \
    Wang2022.json
#ai002_test_replay_buffer
