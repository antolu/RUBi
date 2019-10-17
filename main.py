#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.optim as optim

from models.rubi.baseline_net import BaselineNet
from models.rubi.rubi import RUBi
from models.rubi.loss import RUBiLoss, BaselineLoss
from tools.parse_args import parse_arguments
from utilities.earlystopping import EarlyStopping
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


args = parse_arguments()

# Check if GPU can be used, else use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device {device}.")

model = None
if args.baseline == "baseline":
    model = BaselineNet().to(device)
    raise NotImplementedError()
elif args.baseline == "san":
    raise NotImplementedError()
elif args.baseline == "updn":
    raise NotImplementedError()

if args.rubi:
    model = RUBi(model).to(device)
    loss = RUBiLoss(args["loss-weights"][0], args["loss-weights"][1])
else:
    loss = BaselineLoss()

# Load pretrained model if exists
if args.pretrained_model:
    pretrained_model = torch.load(args.pretrained_model, map_device=device)
    model.load_state_dict(pretrained_model["model"])

# TODO: read in datasets
dataloader = None

if args.train:
    # tensorboard Writer will output to /runs directory
    tensorboard_writer = SummaryWriter(filename_suffix='train')
    losses = []
    accs = []

    model.train()

    # Initialize parameters in network

    optimizer = optim.Adamax(model.parameters(), lr=args.lr)
    # TODO: implement a LR scheduler

    # use FP16
    if args.fp16:
        import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    if args.pretrained_model:
        optimizer.load_state_dict(args.pretrained_model["optimizer"])
        if args.fp16:
            amp.load_state_dict(args.pretrained_model["amp"])

    es = EarlyStopping(min_delta=args.eps, patience=args.patience)

    try:
        epoch = 0
        while True:
            epoch += 1

            # assume inputs is a dict
            for i_batch, inputs in enumerate(dataloader):
                for key, value in inputs:
                    value.to(device)

                model.zero_grad()
                predictions = model(inputs)
                current_loss = loss(inputs["answers"], predictions)
                losses.append(current_loss)

                if args.fp16:
                    with amp.scale_loss(current_loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    current_loss.backward()

                optimizer.step()

                # early stopping if loss hasn't improved
                if es.step(current_loss):
                    break

        print("Training complete after {} epochs.".format(epoch))
    except KeyboardInterrupt:
        print("Training canceled. ")
        pass
    finally:
        # Save checkpoint
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        if args.fp16:
            checkpoint['amp'] = amp.state_dict()

        filename = args.model + "_epoch_{}_dataset_{}_{}.pt".format(epoch, args.dataset, datetime.now().strftime("%Y%m%d%H%M%S"))
        torch.save(checkpoint, filename)

    # Visualize train and test loss and accuracy graphs in tensorboard
    for n_iter in range(len(losses)):
        writer.add_scalar('Loss/train', losses[n_iter], n_iter)
    #    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    tensorboard_writer.close()

elif args.test:
    raise NotImplementedError()
    # tensorboard_writer = SummaryWriter(filename_suffix='test')
    # # Visualize train and test loss and accuracy graphs
    # # tensorboard will group them together according to their name
    # for n_iter in range(len(losses)):
    #     writer.add_scalar('Loss/train', losses[n_iter], n_iter)
    #     writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    # tensorboard_writer.close()

