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



:<<'ai002_test'
python offline_test_newformat_1step.py \
    --model029 /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1117_full_noprevQT_mix0.5/0_29_checkpoint_best_loss.pth.tar \
    --model3059 /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1117_full_noprevQT_mix0.5/30_59_checkpoint_best_loss.pth.tar \
    --model6165 /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1117_full_noprevQT_mix0.5/61_65_checkpoint_best_loss.pth.tar \
    --train_configs /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1117_full_noprevQT_mix0.5/configs.txt \
    --start_ts 35040 \
    --sample_rate 12 \
    --multistep 1 \
    --no_prevQT \
    --legacy_order \
    mix0.5_sample12.json
ai002_test
:<<'ai002_test'
python offline_test_newformat_1step.py \
    --model029 /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1117_full_noprevQTls_perturbls/0_29_checkpoint_best_loss.pth.tar \
    --model3059 /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1117_full_noprevQTls_perturbls/30_59_checkpoint_best_loss.pth.tar \
    --model6165 /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1117_full_noprevQTls_perturbls/61_65_checkpoint_best_loss.pth.tar \
    --train_configs /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1117_full_noprevQTls_perturbls/configs.txt \
    --start_ts 35040 \
    --sample_rate 12 \
    --multistep 1 \
    --no_prevQTLS \
    --legacy_order \
    perturbLS_noLS_sample12.json
ai002_test
:<<'ai002_test'
python offline_test_newformat_1step.py \
    --model029 /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1117_full_noprevQT/0_29_checkpoint_best_loss.pth.tar \
    --model3059 /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1117_full_noprevQT/30_59_checkpoint_best_loss.pth.tar \
    --model6165 /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1117_full_noprevQT/61_65_checkpoint_best_loss.pth.tar \
    --train_configs /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1117_full_noprevQT/configs.txt \
    --start_ts 35040 \
    --sample_rate 12 \
    --multistep 1 \
    --no_prevQT \
    --legacy_order \
    seed1117_sample12.json
ai002_test
:<<'ai002_test'
python offline_test_newformat_1step.py \
    --model029 /cust_users/x-w19/nncam.ckpts/resmlp.0.4sampled.noise0.0/0_29_nodesize512_num_blocks7_actrelu_bs1024_scheduler_coslr_lr0.001_ep50_noise0.0_wd0_dropout0/checkpoint_epoch50.pth.tar \
    --model3059 /cust_users/x-w19/nncam.ckpts/resmlp.0.4sampled.noise0.0/30_59_nodesize512_num_blocks7_actrelu_bs1024_scheduler_coslr_lr0.001_ep50_noise0.0_wd0_dropout0/checkpoint_epoch50.pth.tar \
    --model6165 /temp_share/nncam-cases/smart_test/ckpts/ckpts_extend/61-65_resnet_ep50_noise0.01_wd0_dropout0/checkpoint_epoch50.pth.tar \
    --start_ts 35040 \
    --sample_rate 12 \
    --multistep 0 \
    --legacy_order \
    --norm_type minmax_legacy \
    nncam_sample12.json
ai002_test
:<<'ai002_test'
python offline_test_newformat_1step.py \
    --model029 /share3/chenj209/ckpts_sampled12/conv_mem/noreplay_buffer309_v2_seed1117_full_noprevQT_minmax/0_29_checkpoint_best_loss.pth.tar \
    --model3059 /share3/chenj209/ckpts_sampled12/conv_mem/noreplay_buffer309_v2_seed1117_full_noprevQT_minmax/30_59_checkpoint_best_loss.pth.tar \
    --model6165 /share3/chenj209/ckpts_sampled12/conv_mem/noreplay_buffer309_v2_seed1117_full_noprevQT_minmax/61_65_checkpoint_best_loss.pth.tar \
    --train_configs /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1117_full_noprevQT/configs.txt \
    --start_ts 35040 \
    --sample_rate 12 \
    --multistep 1 \
    --no_prevQT \
    --legacy_order \
    --norm_type minmax_legacy \
    noep_minmax_sample12.json
ai002_test
:<<'ai002_test'
python offline_test_newformat_1step.py \
    --model029 /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v3_seed1117_sample2_noprevQT_minmax_mix1.0/0_29_checkpoint_best_loss.pth.tar \
    --model3059 /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v3_seed1117_sample2_noprevQT_minmax_mix1.0/30_59_checkpoint_best_loss.pth.tar \
    --model6165 /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v3_seed1117_sample2_noprevQT_minmax_mix1.0/61_65_checkpoint_best_loss.pth.tar \
    --train_configs /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1117_full_noprevQT/configs.txt \
    --start_ts 35040 \
    --sample_rate 12 \
    --multistep 1 \
    --no_prevQT \
    --legacy_order \
    --norm_type minmax_legacy \
    er_minmax_sample2_rbs288.json
ai002_test
#:<<'ai002_test'
python offline_test_newformat_1step.py \
    --model029 /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v3_seed1117_sample2_noprevQT_std_mix1.0/0_29_checkpoint_best_loss.pth.tar \
    --model3059 /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v3_seed1117_sample2_noprevQT_std_mix1.0/30_59_checkpoint_best_loss.pth.tar \
    --model6165 /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v3_seed1117_sample2_noprevQT_std_mix1.0/61_65_checkpoint_best_loss.pth.tar \
    --train_configs /share3/chenj209/ckpts_sampled12/conv_mem/replay_buffer309_v2_seed1117_full_noprevQT/configs.txt \
    --start_ts 35040 \
    --sample_rate 12 \
    --multistep 1 \
    --no_prevQT \
    --legacy_order \
    --norm_type std \
    er_std_sample2_rbs288.json
#ai002_test
