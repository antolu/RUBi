#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.optim as optim
import torch.utils.data as data

from models.rubi.baseline_net import BaselineNet
from models.rubi.rubi import RUBi
from models.rubi.loss import RUBiLoss, BaselineLoss
from tools.parse_args import parse_arguments
from utilities.earlystopping import EarlyStopping
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from utilities.schedule_lr import LrScheduler
from dataloader import DataLoaderVQA


def main():
    """
    main training method
    """
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

    dataloader = data.DataLoader(DataLoaderVQA(args))

    if args.train:
        # train on the train set and plot loss and acc graphs in tensorboard
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
                    for key, value in inputs:
                        value.to(device)

                    model.zero_grad()
                    predictions = model(inputs)
                    current_loss = loss(inputs["answers"], predictions)
                    current_acc = compute_acc(inputs["answers"], predictions)
                    losses.append(current_loss)
                    accs.append(current_acc)

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

            filename = args.model + "_epoch_{}_dataset_{}_{}.pt".format(epoch, args.dataset, datetime.now().strftime("%Y%m%d%H%M%S"))
            torch.save(checkpoint, filename)

        # Visualize train loss and accuracy graphs in tensorboard
        writer.add_graph(model, predictions)
        for n_iter in range(len(losses)):
            writer.add_scalar('Loss/train', losses[n_iter], n_iter)
           writer.add_scalar('Accuracy/train', accs[n_iter], n_iter)
        tensorboard_writer.close()

    elif args.test:
        # test on the test set and return loss and accuracy
        inputs = dataloader
        for key, value in inputs:
            value.to(device)

        model.zero_grad() # TODO: is this needed?
        predictions = model(inputs)
        test_loss = loss(inputs["answers"], predictions)
        test_acc = compute_acc(inputs['answers'], predictions)

        tensorboard_writer = SummaryWriter(filename_suffix='test')
        # Visualize computational graph, test loss and acc in tensorboard
        writer.add_graph(model, predictions)
        writer.add_scalar('Loss/test', test_loss, n_iter)
        writer.add_scalar('Accuracy/test', test_acc, n_iter)
        tensorboard_writer.close()

def compute_acc(labels, predictions):
    """
    predictions: output of a forward pass through the model
    computes the accuracy by comparing them with the correct labels
    """
    output = (predictions > 0.5).float()
    correct = (predictions == labels).float().sum()
    acc = correct/output.shape[0]
    return acc

if __name__ == __main__:
    main()
