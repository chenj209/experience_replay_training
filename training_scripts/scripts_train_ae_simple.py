import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--multistep", type=int, help="multistep", default=1)
    #parser.add_argument("--gpu", type=int, help="gpu index", default=0)
    parser.add_argument("--lr", type=int, help="learning rate", default=0.0001)
    parser.add_argument("--ex_input", type=str, nargs="?")
    parser.add_argument("--ex_input_prev", type=str, nargs="?")
    parser.add_argument("--region_mask", type=str, default="all")
    parser.add_argument("--rec_weight", type=float, default=1.0)
    parser.add_argument("--pred_weight", type=float, default=0.0)
    parser.add_argument("--ae_config", type=str, default=str)
    parser.add_argument("--resume", type=str, default="")
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

    name = f"simpleae_model029_sampled1_0314_multistep{args.multistep}_weight{args.pred_weight}_{args.rec_weight}_{args.ae_config.rstrip('.json')}_EX_{'+'.join(ex_input)}_EXPREV_{'+'.join(ex_input_prev)}"

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
                        batch_size = '128'
                        lr_strategy = 'constant'

                        network = 'resnet_output30'
                        print(name)
                        commands = f"python train_ae_model_simple.py --data_dir {DATA_DIR}" + " --output_type 0-29 --noise_std {} " \
                                '--network {} --node_size {} --num_blocks {} --activation {} --dropout {} ' \
                                '--train_batch {} --lr_strategy {} --lr {} --epoch {} --wd {} ' \
                                '--checkpoint /pscratch/sd/c/chenjd21/ckpts_time/{} --multistep {} --sample_rate 1 --workers 112 --region_mask {} ' \
                                '--rec_weight {} --pred_weight {} --ae_config {} ' \
                                '--ex_input {} --ex_input_prev {}'.format(str(noise_std),
                                                                network, node_size, num_blocks, activation,
                                                                dropout,
                                                                batch_size, lr_strategy, lr, epoch,
                                                                weight_decay,
                                                                name, args.multistep, args.region_mask,
                                                                args.rec_weight, args.pred_weight, args.ae_config,
                                                                " ".join(ex_input), " ".join(ex_input_prev))
                        if args.resume:
                            commands += f" --resume {args.resume}"


                        print(commands)
                        os.system(commands)
