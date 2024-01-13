import os
import sys
for epoch in ['50']:
        for lr in ['0.0001']:
            for batch_size in ['32']:
                for noise_std in ['0']:
                    dropout = '0'
                    weight_decay = '0'
                    activation = 'relu'
                    lr_strategy = 'coslr'

                    network = sys.argv[1]
                    print("Training network: ", network)
                    name = 'modellandmask_0-29_{}_act{}_bs{}_scheduler_{}_lr{}_ep{}_noise{}_wd{}_dropout{}_wxnorm'.format(
                        network, activation, batch_size, lr_strategy, lr, epoch, noise_std, weight_decay, dropout)
                    print(name)
                    commands = 'CUDA_VISIBLE_DEVICES={} python train_nn_withlandmask.py --data_dir /home/users/data/nncam_data/image_set/ --output_type 0-29 --noise_std {} ' \
                               '--network {} --activation {} --dropout {} ' \
                               '--train-batch {} --lr_strategy {} --lr {} --epoch {} --wd {} --workers 8 ' \
                               '--checkpoint ckpts/{}'.format(sys.argv[2], str(noise_std),
                                                                                network, activation,
                                                                                dropout,
                                                                                batch_size, lr_strategy, lr, epoch,
                                                                                weight_decay,
                                                                                name)
                    
                    print(commands)
                    os.system(commands)

#                     network = 'cnn2_pad'
#                     name = 'unet_tkdedata_wx_alldata_rmBadData_30-59_{}_act{}_bs{}_scheduler_{}_lr{}_ep{}_noise{}_wd{}_dropout{}_wxnorm'.format(
#                         network, activation, batch_size, lr_strategy, lr, epoch, noise_std, weight_decay, dropout)
#                     print(name)
#                     commands = 'CUDA_VISIBLE_DEVICES=1 screen python run_de.py --data_dir /cust_users/alg/TKDE.lab/TKDE_dataset/trainset --output_type 30-59 --noise_std {} ' \
#                                '--network {} --activation {} --dropout {} ' \
#                                '--train-batch {} --lr_strategy {} --lr {} --epoch {} --wd {} --workers 8 ' \
#                                '--checkpoint ckpts/{}'.format(str(noise_std),
#                                                               network, activation,
#                                                               dropout,
#                                                               batch_size, lr_strategy, lr, epoch,
#                                                               weight_decay,
#                                                               name)

#                     print(commands)
#                     os.system(commands)

#                     network = 'cnn2_pad'
#                     name = 'unet_tkdedata_alldata_rmBadData_60_{}_act{}_bs{}_scheduler_{}_lr{}_ep{}_noise{}_wd{}_dropout{}_wxnorm'.format(
#                         network, activation, batch_size, lr_strategy, lr, epoch, noise_std, weight_decay, dropout)
#                     print(name)
#                     commands = 'CUDA_VISIBLE_DEVICES=2 screen python run_de.py --data_dir /cust_users/alg/TKDE.lab/TKDE_dataset/trainset --output_type 60 --noise_std {} ' \
#                                '--network {} --activation {} --dropout {} ' \
#                                '--train-batch {} --lr_strategy {} --lr {} --epoch {} --wd {} --workers 8 ' \
#                                '--checkpoint ckpts/{}'.format(str(noise_std),
#                                                               network, activation,
#                                                               dropout,
#                                                               batch_size, lr_strategy, lr, epoch,
#                                                               weight_decay,
#                                                               name)
                    
#                     print(commands)
#                     os.system(commands)
                    
#                     network = 'cnn2_pad'
#                     name = 'unet_tkdedata_alldata_rmBadData_61-65_{}_act{}_bs{}_scheduler_{}_lr{}_ep{}_noise{}_wd{}_dropout{}_wxnorm'.format(
#                         network, activation, batch_size, lr_strategy, lr, epoch, noise_std, weight_decay, dropout)
#                     print(name)
#                     commands = 'CUDA_VISIBLE_DEVICES=3 screen python run_de.py --data_dir /cust_users/alg/TKDE.lab/TKDE_dataset/trainset --output_type 61-65 --noise_std {} ' \
#                                '--network {} --activation {} --dropout {} ' \
#                                '--train-batch {} --lr_strategy {} --lr {} --epoch {} --wd {} --workers 8 ' \
#                                '--checkpoint ckpts/{}'.format(str(noise_std),
#                                                               network, activation,
#                                                               dropout,
#                                                               batch_size, lr_strategy, lr, epoch,
#                                                               weight_decay,
#                                                               name)
                    
#                     print(commands)
#                     os.system(commands)
