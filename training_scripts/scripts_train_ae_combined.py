import os
import argparse
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--multistep", type=int, help="number of previous timesteps to be used as inputs", default=1)
    #parser.add_argument("--gpu", type=int, help="gpu index", default=0)
    parser.add_argument("--lr", type=float, help="learning rate (default to 0.001)", default=0.001)
    parser.add_argument("--region_mask", type=str, default="all", 
        help="default to all 96x144 grid, otherwirse path to a region mask (in ../consts folder)")
    parser.add_argument("--data_means", type=str, default="../consts/all_means.npz",
        help="path to npz files that stores means for all variables (in ../consts folder),\
                default to ../consts/all_means.npz")
    parser.add_argument("--data_stds", type=str, default="../consts/all_stds.npz",
        help="path to npz files that stores stds for all variables (in ../consts folder),\
                default to ../consts/all_stds.npz")
    # parser.add_argument("--ex_input_prev", type=str, nargs="?",
    #     help="input variables to be added for previous timesteps \
    #             (in addition to Q,T,dTls,dqls,solin,ps,qtend,stend,SOLL,SOLLD,SOLS,SOLSD,FSDS")
    # parser.add_argument("--ex_input", type=str, nargs="?",
    #     help="input variables to be added for previous timesteps \
    #             (in addition to Q,T,dTls,dqls,solin,ps")
    parser.add_argument("--rec_weight", type=float, default=0.4, help="weight for reconstruction loss")
    parser.add_argument("--pred_weight", type=float, default=0.6, help="weight for prediction loss")
    parser.add_argument("--ae_config", type=str, default=str, help="path to json file that stores the autoencoder configuration")
    parser.add_argument("--resume", type=str, nargs="?", help="path to checkpoint to resume training")
    args = parser.parse_args()

    if args.ex_input is not None:
        ex_input = args.ex_input.split("+")
    else:
        ex_input = []

    if args.ex_input_prev is not None:
        ex_input_prev = args.ex_input_prev.split("+")
    else:
        ex_input_prev = []

    # DATA_DIR = "/home/users/data/nncam_data/image_set/"
    # if not os.path.exists(DATA_DIR):
    #     DATA_DIR = "/data/nncam_data/image_set/"
    # if not os.path.exists(DATA_DIR):
    #     #DATA_DIR = "/global/cfs/cdirs/m4359/zhangtao/nncam/image_set/"
    #     DATA_DIR = "/pscratch/sd/c/chenjd21/spcam_new_data_32/"

    name = f"{VAE_CKPT_PREFIX}_sampled{SAMPLE_RATE}_multistep{args.multistep}_weight{args.pred_weight}_{args.rec_weight}_{args.ae_config.rstrip('.json')}_EX_{'+'.join(ex_input)}_EXPREV_{'+'.join(ex_input_prev)}"

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
                        batch_size = '16'
                        lr_strategy = 'constant'

                        network = 'resnet_output30'
                        print(name)
                        commands = f"python train_ae_model_ex_combined.py --data_dir {DATA_DIR}" + " --output_type 0-29 --noise_std {} " \
                                '--network {} --node_size {} --num_blocks {} --activation {} --dropout {} ' \
                                '--train_batch {} --lr_strategy {} --lr {} --epoch {} --wd {} ' \
                                f"--checkpoint {CKPT_DIR}"+'/{} --multistep {} '+f"--sample_rate {SAMPLE_RATE} --workers 16"+' --region_mask {} --data_means {} --data_stds {} ' \
                                '--rec_weight {} --pred_weight {} --ae_config {} ' \
                                '--ex_input {} --ex_input_prev {}'.format(str(noise_std),
                                                                network, node_size, num_blocks, activation,
                                                                dropout,
                                                                batch_size, lr_strategy, lr, epoch,
                                                                weight_decay,
                                                                name, args.multistep, args.region_mask,
                                                                args.data_means, args.data_stds,
                                                                args.rec_weight, args.pred_weight, args.ae_config,
                                                                " ".join(ex_input), " ".join(ex_input_prev))
                        if args.resume:
                            commands += f" --resume {args.resume}"


                        print(commands)
                        os.system(commands)
