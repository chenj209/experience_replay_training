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
#:<<'ai002_test_replay_buffer'
CUDA_VISIABLE_DEVICES=2 python offline_test_newformat_2step.py \
    --model029 /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1_full/0_29_checkpoint_best_loss.pth.tar \
    --model3059 /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1_full//30_59_checkpoint_best_loss.pth.tar \
    --model6165 /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1_full//61_65_checkpoint_best_loss.pth.tar \
    --train_configs /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1_full/configs.txt \
    --start_ts 35040 \
    --sample_rate 12 \
    --multistep 2 \
    --legacy_order \
    replay_buffer_2step_v2_seed1_full.json
#ai002_test_replay_buffer
