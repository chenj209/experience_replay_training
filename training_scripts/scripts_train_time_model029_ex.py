import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--multistep", type=int, help="multistep", default=1)
    #parser.add_argument("--gpu", type=int, help="gpu index", default=0)
    parser.add_argument("--lr", type=int, help="learning rate", default=0.001)
    parser.add_argument("--region_mask", type=str, default="all")
    parser.add_argument("--ex_input_prev", type=str, nargs="?")
    parser.add_argument("--ex_input", type=str, nargs="?")
    args = parser.parse_args()

    if args.ex_input is not None:
        ex_input = args.ex_input.split("+")
    else:
        ex_input = []

    if args.ex_input_prev is not None:
        ex_input_prev = args.ex_input_prev.split("+")
    else:
        ex_input_prev = []


    DATA_DIR = "/home/users/data/nncam_data/image_set/"
    if not os.path.exists(DATA_DIR):
        DATA_DIR = "/data/nncam_data/image_set/"
    if not os.path.exists(DATA_DIR):
        #DATA_DIR = "/global/cfs/cdirs/m4359/zhangtao/nncam/image_set/"
        DATA_DIR = "/pscratch/sd/c/chenjd21/spcam_new_data/"

    name = f"time_model029_sampled1_0226_multistep{args.multistep}_EX_{'+'.join(ex_input)}_EXPREV_{'+'.join(ex_input_prev)}_region_{args.region_mask.split('/')[-1].rstrip('.npy')}_test"

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
                        batch_size = '2048'
                        lr_strategy = 'constant'

                        network = 'resnet_output30'
                        print(name)
                        commands = f"python train_time_model_ex.py --data_dir {DATA_DIR}" + " --output_type 0-29 --noise_std {} " \
                                '--network {} --node_size {} --num_blocks {} --activation {} --dropout {} ' \
                                '--train_batch {} --lr_strategy {} --lr {} --epoch {} --wd {} ' \
                                '--checkpoint /pscratch/sd/c/chenjd21/ckpts_time/{} --multistep {} --sample_rate 1 --region_mask {} --ex_input {} --ex_input_prev {}'.format(str(noise_std),
                                                                network, node_size, num_blocks, activation,
                                                                dropout,
                                                                batch_size, lr_strategy, lr, epoch,
                                                                weight_decay,
                                                                name, args.multistep, args.region_mask, " ".join(ex_input)," ".join(ex_input_prev))

                        print(commands)
                        os.system(commands)
