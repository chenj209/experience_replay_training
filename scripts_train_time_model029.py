import os

#DATA_DIR = "~/data/tkde_set/"
DATA_DIR = "/home/users/data/nncam_data/image_set/"
name = "time_model029_0412"

for epoch in ['100']:
    for num_blocks in ['7']:
        for node_size in ['512']:
            for noise_std in [0.0]:
                for lr in ['0.001']:

                    dropout = '0'
                    weight_decay = '0'
                    ################# noise #############################
                    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!注意######################
                    
                    activation = 'relu'
                    batch_size = '1024'
                    lr_strategy = 'coslr'

                    network = 'resnet_output30'
                    #name = 'rmbaddata_wxnorm_subset_0-29_{}_nodesize{}_num_blocks{}_act{}_bs{}_scheduler_{}_lr{}_ep{}_noise{}_wd{}_dropout{}'.format(
                    #    network, node_size, num_blocks, activation,
                    #    batch_size, lr_strategy, lr, epoch, noise_std, weight_decay, dropout)
                    print(name)
                    commands = f"CUDA_VISIBLE_DEVICES=1 python train_time_model.py --data_dir {DATA_DIR}" + " --output_type 0-29 --noise_std {} " \
                               '--network {} --node_size {} --num_blocks {} --activation {} --dropout {} ' \
                               '--train-batch {} --lr_strategy {} --lr {} --epoch {} --wd {} ' \
                               '--checkpoint ckpts_time/{}'.format(str(noise_std),
                                                              network, node_size, num_blocks, activation,
                                                              dropout,
                                                              batch_size, lr_strategy, lr, epoch,
                                                              weight_decay,
                                                              name)
                  
                    print(commands)
                    os.system(commands)

#                    network = 'resnet_output30'
#                    name = 'rmbaddata_wxnorm_subset_30-59_{}_nodesize{}_num_blocks{}_act{}_bs{}_scheduler_{}_lr{}_ep{}_noise{}_wd{}_dropout{}'.format(
#                        network, node_size, num_blocks, activation,
#                        batch_size, lr_strategy, lr, epoch, noise_std, weight_decay, dropout)
#                    print(name)
#                    commands = f"CUDA_VISIBLE_DEVICES=2 screen python run_de_subset_files.py --data_dir {DATA_DIR}" + ' --output_type 30-59 --noise_std {} ' \
#                               '--network {} --node_size {} --num_blocks {} --activation {} --dropout {} ' \
#                               '--train-batch {} --lr_strategy {} --lr {} --epoch {} --wd {} ' \
#                               '--checkpoint ckpts_longepoch/{}'.format(str(noise_std),
#                                                              network, node_size, num_blocks, activation,
#                                                              dropout,
#                                                              batch_size, lr_strategy, lr, epoch,
#                                                              weight_decay,
#                                                              name)
#                  
#                    print(commands)
#                    os.system(commands)
# 
#   #                 network = 'resnet_output1'
#   #                 name = 'rmbaddata_wxnorm_subset_60_{}_nodesize{}_num_blocks{}_act{}_bs{}_scheduler_{}_lr{}_ep{}_noise{}_wd{}_dropout{}'.format(
#   #                     network, node_size, num_blocks, activation,
#   #                     batch_size, lr_strategy, lr, epoch, noise_std, weight_decay, dropout)
#   #                 print(name)
#   #                 commands = 'CUDA_VISIBLE_DEVICES=2 screen python run_de_subset_files.py --data_dir /data/training_data/image_set/ --output_type 60 --noise_std {} ' \
#   #                            '--network {} --node_size {} --num_blocks {} --activation {} --dropout {} ' \
#   #                            '--train-batch {} --lr_strategy {} --lr {} --epoch {} --wd {} ' \
#   #                            '--checkpoint ckpts_longepoch/{}'.format(str(noise_std),
#   #                                                           network, node_size, num_blocks, activation,
#   #                                                           dropout,
#   #                                                           batch_size, lr_strategy, lr, epoch,
#   #                                                           weight_decay,
#   #                                                           name)
#                   
#   #                 print(commands)
#   #                 os.system(commands)
# 
#                    network = 'resnet_output5'
#                    name = 'rmbaddata_wxnorm_subset_nopenalty_61-65_{}_nodesize{}_num_blocks{}_act{}_bs{}_scheduler_{}_lr{}_ep{}_noise{}_wd{}_dropout{}'.format(
#                        network, node_size, num_blocks, activation,
#                        batch_size, lr_strategy, lr, epoch, noise_std, weight_decay, dropout)
#                    print(name)
#                    commands = f"CUDA_VISIBLE_DEVICES=3 screen python run_de_subset_files.py --data_dir {DATA_DIR}" + " --output_type 61-65 --noise_std {} " \
#                               '--network {} --node_size {} --num_blocks {} --activation {} --dropout {} ' \
#                               '--train-batch {} --lr_strategy {} --lr {} --epoch {} --wd {} ' \
#                               '--checkpoint ckpts_longepoch/{}'.format(str(noise_std),
#                                                              network, node_size, num_blocks, activation,
#                                                              dropout,
#                                                              batch_size, lr_strategy, lr, epoch,
#                                                              weight_decay,
#                                                              name)
#                  
#                    print(commands)
#                    os.system(commands)
