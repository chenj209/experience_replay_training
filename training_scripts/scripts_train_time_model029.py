import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--multistep", type=int, help="multistep", default=1)
    parser.add_argument("--gpu", type=int, help="gpu index", default=0)
    parser.add_argument("--lr", type=int, help="learning rate", default=0.001)
    args = parser.parse_args()

    DATA_DIR = "/home/users/data/nncam_data/image_set/"
    if not os.path.exists(DATA_DIR):
        DATA_DIR = "/data/nncam_data/image_set/"

    name = f"time_model029_sampled12_1204_multistep{args.multistep}"

    for epoch in ['50']:
        for num_blocks in ['7']:
            for node_size in ['512']:
                for noise_std in [0.0]:
                    for lr in [str(args.lr)]:

                        dropout = '0'
                        weight_decay = '0'
                        ################# noise #############################
                        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!注意######################
                        
                        activation = 'relu'
                        batch_size = '32'
                        lr_strategy = 'coslr'

                        network = 'resnet_output30'
                        print(name)
                        commands = f"CUDA_VISIBLE_DEVICES={args.gpu} python train_time_model.py --data_dir {DATA_DIR}" + " --output_type 0-29 --noise_std {} " \
                                '--network {} --node_size {} --num_blocks {} --activation {} --dropout {} ' \
                                '--train_batch {} --lr_strategy {} --lr {} --epoch {} --wd {} ' \
                                '--checkpoint ckpts_time/{} --multistep {}'.format(str(noise_std),
                                                                network, node_size, num_blocks, activation,
                                                                dropout,
                                                                batch_size, lr_strategy, lr, epoch,
                                                                weight_decay,
                                                                name, args.multistep)
                    
                        print(commands)
                        os.system(commands)
