import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--multistep", type=int, help="multistep", default=1)
    #parser.add_argument("--gpu", type=int, help="gpu index", default=0)
    parser.add_argument("--lr", type=int, help="learning rate", default=0.001)
    parser.add_argument("--ex_input", type=str, nargs="*", default=[])
    parser.add_argument("--region_mask", type=str, default="all")
    args = parser.parse_args()

    DATA_DIR = "/home/users/data/nncam_data/image_set/"
    if not os.path.exists(DATA_DIR):
        DATA_DIR = "/data/nncam_data/image_set/"
    if not os.path.exists(DATA_DIR):
        #DATA_DIR = "/global/cfs/cdirs/m4359/zhangtao/nncam/image_set/"
        DATA_DIR = "/pscratch/sd/c/chenjd21/spcam_new_data/"

    name = f"ae_model029_sampled12_0108_multistep{args.multistep}+{'+'.join(args.ex_input)}"

    for epoch in ['200']:
        for num_blocks in ['7']:
            for node_size in ['512']:
                for noise_std in [0.0]:
                    for lr in [str(args.lr)]:

                        dropout = '0'
                        weight_decay = '0'
                        ################# noise #############################
                        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!注意######################

                        activation = 'relu'
                        batch_size = '8'
                        lr_strategy = 'coslr'

                        network = 'resnet_output30'
                        print(name)
                        commands = f"python train_ae_model_ex.py --data_dir {DATA_DIR}" + " --output_type 0-29 --noise_std {} " \
                                '--network {} --node_size {} --num_blocks {} --activation {} --dropout {} ' \
                                '--train_batch {} --lr_strategy {} --lr {} --epoch {} --wd {} ' \
                                '--checkpoint ckpts_time/{} --multistep {} --sample_rate 96 --region_mask {} --ex_input {}'.format(str(noise_std),
                                                                network, node_size, num_blocks, activation,
                                                                dropout,
                                                                batch_size, lr_strategy, lr, epoch,
                                                                weight_decay,
                                                                name, args.multistep, args.region_mask,  " ".join(args.ex_input))

                        print(commands)
                        os.system(commands)
