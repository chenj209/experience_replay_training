import os
import argparse
from consts import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--multistep", type=int, help="multistep", default=1)
    #parser.add_argument("--gpu", type=int, help="gpu index", default=0)
    parser.add_argument("--lr", type=int, help="learning rate", default=0.001)
    parser.add_argument("--region_mask", type=str, default="all")
    parser.add_argument("--data_means", type=str, default="../consts/all_means.npz")
    parser.add_argument("--data_stds", type=str, default="../consts/all_stds.npz")
    parser.add_argument("--ex_input_prev", type=str, nargs="?")
    parser.add_argument("--ex_input", type=str, nargs="?")
    parser.add_argument("--resume", type=bool, default=False)
    args = parser.parse_args()

    if args.ex_input is not None:
        ex_input = args.ex_input.split("+")
    else:
        ex_input = []

    if args.ex_input_prev is not None:
        ex_input_prev = args.ex_input_prev.split("+")
    else:
        ex_input_prev = []


    #DATA_DIR = "/share3/chenj209/spcam_new_data_32/"
    #if not os.path.exists(DATA_DIR):
        #DATA_DIR = "/data/nncam_data/image_set/"
    #if not os.path.exists(DATA_DIR):
        #DATA_DIR = "/global/cfs/cdirs/m4359/zhangtao/nncam/image_set/"
        #DATA_DIR = "/pscratch/sd/c/chenjd21/spcam_new_data_32/"

    name = f"{BASELINE_CKPT_PREFIX}_sampled{SAMPLE_RATE}_multistep{args.multistep}_EX_{'+'.join(ex_input)}_EXPREV_{'+'.join(ex_input_prev)}_region_{args.region_mask.split('/')[-1].rstrip('.npy')}"

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
                        batch_size = '8'
                        lr_strategy = 'constant'

                        network = 'resnet_output30'
                        print(name)
                        commands = f"python train_time_model_ex.py --data_dir {DATA_DIR}" + " --output_type 0-29 --noise_std {} " \
                                '--network {} --node_size {} --num_blocks {} --activation {} --dropout {} ' \
                                '--train_batch {} --lr_strategy {} --lr {} --epoch {} --wd {} --worker 8 ' \
                                '--checkpoint {} --multistep {} --sample_rate {} --region_mask {} --data_means {} --data_stds {} --ex_input {} --ex_input_prev {}'.format(str(noise_std),
                                                                network, node_size, num_blocks, activation,
                                                                dropout,
                                                                batch_size, lr_strategy, lr, epoch,
                                                                weight_decay,
                                                                CKPT_DIR+name, args.multistep, SAMPLE_RATE, args.region_mask, args.data_means, args.data_stds, " ".join(ex_input)," ".join(ex_input_prev))

                        if args.resume:
                            commands += f" --resume {CKPT_DIR}/{name}/checkpoint.pth.tar"
                        print(commands)
                        os.system(commands)
