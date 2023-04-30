import os

#DATA_DIR = "./data/baseline_online_label/"
#DATA_DIR = "/home/users/chenj209/nncam_online_data/spcam_perturb_m32/"
DATA_DIR = "/share1/x-w19/online-data/perturb-spcam/"
#baseline_ckpt_paths = {
#        "029": "/cust_users/x-w19/nncam.ckpts/resmlp.2years.50epochs/0_29_nodesize512_num_blocks7_actrelu_bs1024_scheduler_coslr_lr0.001_ep50_noise0.0_wd0_dropout0/checkpoint.pth.tar",
#        "3059": "/cust_users/x-w19/nncam.ckpts/resmlp.25GB.noise0.0/30_59_nodesize512_num_blocks7_actrelu_bs1024_scheduler_coslr_lr0.001_ep50_noise0.0_wd0_dropout0/checkpoint.pth.tar",
#        "6165": "/cust_users/x-w19/nncam.ckpts/resmlp.newData.noise0.0/61_65_nodesize512_num_blocks7_actrelu_bs1024_scheduler_coslr_lr0.001_ep50_noise0.0_wd0_dropout0/checkpoint_epoch50.pth.tar"
#        }
baseline_ckpt_paths = {
        "029": "/cust_users/x-w19/nncam.ckpts/resmlp.newData.noise0.0/0_29_nodesize512_num_blocks7_actrelu_bs1024_scheduler_coslr_lr0.001_ep50_noise0.0_wd0_dropout0/checkpoint_epoch45.pth.tar",
        "3059": "/cust_users/x-w19/nncam.ckpts/resmlp.newData.noise0.0/30_59_nodesize512_num_blocks7_actrelu_bs1024_scheduler_coslr_lr0.001_ep50_noise0.0_wd0_dropout0/checkpoint_epoch45.pth.tar",
        "6165": "/cust_users/x-w19/nncam.ckpts/resmlp.newData.noise0.0/61_65_nodesize512_num_blocks7_actrelu_bs1024_scheduler_coslr_lr0.001_ep50_noise0.0_wd0_dropout0/checkpoint_epoch45.pth.tar"
        }
name = "crash2_finetune0430"

for epoch in ['100']:
    for num_blocks in ['7']:
        for node_size in ['512']:
            #for noise_std in [0.002,0.003,0.004,0.006,0.007,0.008,0.009]:
            for noise_std in [0]:
                for lr in ['0.001']:

                    dropout = '0'
                    weight_decay = '0'
                    ################# noise #############################
                    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!注意######################
                    
                    activation = 'relu'
                    batch_size = '1024'
                    lr_strategy = 'coslr'

#               network = 'resnet_output30'
#               name = 'rmbaddata_wxnorm_subset_0-29_{}_nodesize{}_num_blocks{}_act{}_bs{}_scheduler_{}_lr{}_ep{}_noise{}_wd{}_dropout{}'.format(
#                   network, node_size, num_blocks, activation,
#                   batch_size, lr_strategy, lr, epoch, noise_std, weight_decay, dropout)
#               print(name)
#           commands = f"CUDA_VISIBLE_DEVICES=1 python online_training.py --data_dir {DATA_DIR}" + " --output_type 0-29 --noise_std {} " \
#                      '--network {} --node_size {} --num_blocks {} --activation {} --dropout {} ' \
#                      '--train-batch {} --lr_strategy {} --lr {} --epoch {} --wd {} ' \
#                      '--checkpoint online_ckpt/{}'.format(str(noise_std),
#                                                     network, node_size, num_blocks, activation,
#                                                     dropout,
#                                                     batch_size, lr_strategy, lr, epoch,
#                                                     weight_decay,
#                                                     name)
#               commands = f"CUDA_VISIBLE_DEVICES=1 python online_training.py --data_dir {DATA_DIR}" + " --output_type 0-29 --noise_std {} " \
#                          '--network {} --node_size {} --num_blocks {} --activation {} --dropout {} ' \
#                          '--train-batch {} --lr_strategy {} --lr {} --epoch {} --wd {} ' \
#                          '--checkpoint online_ckpt/{} --resume {}'.format(str(noise_std),
#                                                         network, node_size, num_blocks, activation,
#                                                         dropout,
#                                                         batch_size, lr_strategy, lr, epoch,
#                                                         weight_decay,
#                                                             name, baseline_ckpt_paths["029"])
#             
#           print(commands)
#           os.system(commands)
#
                    network = 'resnet_output30'
                    #name = 'rmbaddata_wxnorm_subset_30-59_{}_nodesize{}_num_blocks{}_act{}_bs{}_scheduler_{}_lr{}_ep{}_noise{}_wd{}_dropout{}'.format(
                    #    network, node_size, num_blocks, activation,
                    #    batch_size, lr_strategy, lr, epoch, noise_std, weight_decay, dropout)
                    print(name)
                    commands = f"CUDA_VISIBLE_DEVICES=2 python online_training.py --data_dir {DATA_DIR}" + ' --output_type 30-59 --noise_std {} ' \
                               '--network {} --node_size {} --num_blocks {} --activation {} --dropout {} ' \
                               '--train-batch {} --lr_strategy {} --lr {} --epoch {} --wd {} ' \
                               '--checkpoint online_ckpt/{} --resume {}'.format(str(noise_std),
                                                              network, node_size, num_blocks, activation,
                                                              dropout,
                                                              batch_size, lr_strategy, lr, epoch,
                                                              weight_decay,
                                                              name, baseline_ckpt_paths["3059"])
                  
                    print(commands)
                    os.system(commands)
#
#                network = 'resnet_output1'
#                name = 'rmbaddata_wxnorm_subset_60_{}_nodesize{}_num_blocks{}_act{}_bs{}_scheduler_{}_lr{}_ep{}_noise{}_wd{}_dropout{}'.format(
#                    network, node_size, num_blocks, activation,
#                    batch_size, lr_strategy, lr, epoch, noise_std, weight_decay, dropout)
#                print(name)
#                commands = 'CUDA_VISIBLE_DEVICES=2 screen python online_training.py --data_dir /data/training_data/image_set/ --output_type 60 --noise_std {} ' \
#                           '--network {} --node_size {} --num_blocks {} --activation {} --dropout {} ' \
#                           '--train-batch {} --lr_strategy {} --lr {} --epoch {} --wd {} ' \
#                           '--checkpoint online_ckpt/{}'.format(str(noise_std),
#                                                          network, node_size, num_blocks, activation,
#                                                          dropout,
#                                                          batch_size, lr_strategy, lr, epoch,
#                                                          weight_decay,
#                                                          name)
#              
#                print(commands)
#                os.system(commands)
#
#           network = 'resnet_output5'
#           name = 'rmbaddata_wxnorm_subset_nopenalty_61-65_{}_nodesize{}_num_blocks{}_act{}_bs{}_scheduler_{}_lr{}_ep{}_noise{}_wd{}_dropout{}'.format(
#               network, node_size, num_blocks, activation,
#               batch_size, lr_strategy, lr, epoch, noise_std, weight_decay, dropout)
#           print(name)
#           commands = f"CUDA_VISIBLE_DEVICES=3 screen python online_training.py --data_dir {DATA_DIR}" + " --output_type 61-65 --noise_std {} " \
#                      '--network {} --node_size {} --num_blocks {} --activation {} --dropout {} ' \
#                      '--train-batch {} --lr_strategy {} --lr {} --epoch {} --wd {} ' \
#                      '--checkpoint online_ckpt/{} --resume {}'.format(str(noise_std),
#                                                     network, node_size, num_blocks, activation,
#                                                     dropout,
#                                                     batch_size, lr_strategy, lr, epoch,
#                                                     weight_decay,
#                                                     name, baseline_ckpt_paths["6165"])
#         
#           print(commands)
#           os.system(commands)
