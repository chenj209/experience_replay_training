CUDA_VISIABLE_DEVICES=2 python offline_test_newformat.py \
    --output_type 0-29 \
    --norm_type minmax_legacy \
    --start_ts 35040 \
    --sample_rate 12 \
    --multistep 0 \
    /cust_users/x-w19/nncam.ckpts/resmlp.2years.50epochs/0_29_nodesize512_num_blocks7_actrelu_bs1024_scheduler_coslr_lr0.001_ep50_noise0.0_wd0_dropout0/checkpoint.pth.tar \
    Wang2022_precip_analysis.json