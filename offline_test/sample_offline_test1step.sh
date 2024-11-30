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
python offline_test_newformat_1step.py \
    --model029 /share3/chenj209/ckpts_sampled12/conv_mem/baseline_model029_sampled12_multistep1_EX__EXPREV__region_all/checkpoint_epoch35.pth.tar \
    --model3059 /share3/chenj209/ckpts_sampled12/conv_mem/baseline_model3059_sampled12_multistep1_EX__EXPREV__region_all/checkpoint_epoch35.pth.tar \
    --model6165 /share3/chenj209/ckpts_sampled12/conv_mem/baseline_model6165_sampled12_multistep1_EX__EXPREV__region_all/checkpoint_epoch35.pth.tar \
    --train_configs /share3/chenj209/ckpts_sampled12/conv_mem/baseline_model029_sampled12_multistep1_EX__EXPREV__region_all/configs.txt \
    --start_ts 35040 \
    --sample_rate 144 \
    --multistep 1 \
    test_out.json
ai002_test



#:<<'ai002_test'
python offline_test_newformat_1step.py \
    --model029 /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1117_full_noprevQT_mix0.5/0_29_checkpoint_best_loss.pth.tar \
    --model3059 /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1117_full_noprevQT_mix0.5/30_59_checkpoint_best_loss.pth.tar \
    --model6165 /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1117_full_noprevQT_mix0.5/61_65_checkpoint_best_loss.pth.tar \
    --train_configs /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1117_full_noprevQT_mix0.5/configs.txt \
    --start_ts 35040 \
    --sample_rate 144 \
    --multistep 1 \
    --no_prevQT \
    mix0.5.json
#ai002_test
