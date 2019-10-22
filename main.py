#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.optim as optim
import torch.utils.data as data
import os

from models.rubi.baseline_net import BaselineNet
from models.rubi.rubi import RUBi
from models.rubi.loss import RUBiLoss, BaselineLoss
from tools.parse_args import parse_arguments
from utilities.earlystopping import EarlyStopping
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from utilities.schedule_lr import LrScheduler

from dataloader import DataLoaderVQA


args = parse_arguments()

# Check if GPU can be used, else use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device {}.".format(device))

dataset = DataLoaderVQA(args)
dataloader = data.DataLoader(dataset, batch_size=args.batchsize, num_workers=args.workers, shuffle=True)
#dataloader = DataLoaderVQA(args)

model = None
if args.baseline == "rubi":
    model = BaselineNet(dir_st=args.dir_st, vocab=dataset.get_vocab()).to(device)
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

if args.train:
    # tensorboard Writer will output to /runs directory
    tensorboard_writer = SummaryWriter(filename_suffix='train')
    losses = []
    accs = []

    model.train()

    # Initialize parameters in network

    optimizer = optim.Adamax(model.parameters(), lr=args.lr)

    if args.fp16:
        scheduler = LrScheduler(optimizer.optimizer)
    else:
        scheduler = LrScheduler(optimizer)

    # use FP16
    if args.fp16:
        import apex.amp as amp
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
                for key, value in inputs.items():
                    value.to(device)

                model.zero_grad()
                predictions = model(inputs)
                current_loss = loss(inputs["idx_answer"].squeeze(1), predictions)
                losses.append(current_loss.item())
                print(current_loss.item())

                if args.fp16:
                    with amp.scale_loss(current_loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    current_loss.backward()

                optimizer.step()
            scheduler.step()

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

        filename = args.baseline + "_epoch_{}_dataset_{}_{}_{}.pt".format(epoch, args.dataset, 
                                                                          args.answer_type, 
                                                                          datetime.now().strftime("%Y%m%d%H%M%S"))
        torch.save(checkpoint, os.path.join(args.dir_model, filename))

    # Visualize train and test loss and accuracy graphs in tensorboard
    for n_iter in range(len(losses)):
        tensorboard_writer.add_scalar('Loss/train', losses[n_iter], n_iter)
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

