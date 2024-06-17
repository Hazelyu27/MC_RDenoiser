import argparse
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default="3-dog-4spp",
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--testname', default="3-dog-4spp",
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=40, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('-t', '--time_consider', default=3, type=int,
                        metavar='N', help='time_consider')
    # # model
    # parser.add_argument('--input_w', default=256, type=int,
    #                     help='image width')
    # parser.add_argument('--input_h', default=256, type=int,
    #                     help='image height')

    # dataset
    # train
    parser.add_argument('--dataset', default='3-dog//',
                        help='dataset name')
    parser.add_argument('--datapath', default='D://DataSets//volumndatasets//0614//',
                        help='dataset path')
    parser.add_argument('--noisetype', default='4spp',
                        help='noisetype')
    parser.add_argument('--reftype', default='256spp',
                        help='reftype')
    # test
    parser.add_argument('--testdataset', default='3-dog//',
                        help='dataset name')


    # optimizer
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')

    parser.add_argument('--num_workers', default=0, type=int)

    config = parser.parse_args()

    return config