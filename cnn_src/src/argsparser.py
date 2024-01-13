import argparse

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datasets
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--noise_std', default=0.05, type=float, metavar='M',
                        help='white noise added to the data')
    parser.add_argument('--data_dir', type=str, metavar='PATH',
                        help='path to dataset')
    parser.add_argument('--test_dir', type=str, metavar='PATH',
                        help='path to test dataset')
    parser.add_argument('--output_type', default='0-29', type=str,
                        help='0-29,30-59,60,61,62,63,64')
    parser.add_argument('--train62', default=False, action='store_true',
                        help='use 62 dim input instead of 122')
    
    # Optimization options
    parser.add_argument('--epoch', default=100, type=int, metavar='N',
                        help='number of epochs to run')
    parser.add_argument('--train-batch', default=10, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--lr', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay')
    parser.add_argument('--optim', default='adam', type=str,
                        help='function to approximate')
    parser.add_argument('--lr_strategy', default='', type=str,
                        help='lr strategy: coslr')
    # model potions
    parser.add_argument('--network', default='resnet', type=str,
                        help='function for regression')
    parser.add_argument('--node_size',  default=150,type=int, help='node size')
    parser.add_argument('--layers', default=7, type=int, help='num of layers')
    parser.add_argument('--num_blocks', default=2, type=int, help='num of blocks of resnet')
    parser.add_argument('--activation', default='relu', type=str, help='activation function')
    parser.add_argument('--dropout', default=0, type=float, help='probability of setting element to be zero')
    parser.add_argument('--norm', default='none', type=str)

    # Checkpoints
    parser.add_argument('--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--start_epoch', default=0, type=int)
    # Miscs
    parser.add_argument('--manualSeed', type=int, help='manual seed')

    return parser
