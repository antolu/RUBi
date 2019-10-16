#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import sys
import amp

from models.rubi.baseline_net import BaselineNet
from models.rubi.rubi import RUBi
from models.rubi.loss import RUBiLoss, BaselineLoss
from tools.parse_args import parse_arguments


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
    model.load(args.pretrained_model, map_location=device)

# TODO: read in datasets
dataloader = None

if args.train:
    raise NotImplementedError()

    old_loss = 1e8
    epoch = 0
    model.train()

    # Initialize parameters in network

    optimizer = optim.Adamax(model.parameters(), lr=args.lr)
    # TODO: implement a LR scheduler

    # use FP16
    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    try:
        while True:
            if abs(epoch - args["no-epochs"]) < args.eps:
                break

            # assume inputs is a dict
            for i_batch, inputs in enumerate(dataloader):
                for key, value in inputs:
                    value.to(device)

                model.zero_grad()
                predictions = model(inputs)
                current_loss = loss(labels, predictions)

                if args.fp16:
                    with amp.scale_loss(current_loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    current_loss.backward()

                optimizer.step()

                # TODO: early stopping

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
        # TODO: give proper naming to checkpoints
        torch.save(checkpoint, 'amp_checkpoint.pt')

    # implement tensorboard
elif args.test:
    raise NotImplementedError()
