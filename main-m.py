#!/usr/bin/env python3

from tools.parse_args import parse_arguments
from dataloader import DataLoader

if __name__ == "__main__":

    # Load parameters
    args_dict = parse_arguments()

    args_dict.name = 'retrieval-{}-{}'.format(args_dict.model, args_dict.answer_type)

    opts = vars(args_dict)
    dataloader = DataLoader(args_dict)
    print('------------ Options -------------')
    for k, v in sorted(opts.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-----------------------------------')

    dataset = dataloader.get_dataset()
    for a in dataset.take(1):
        print(a)
