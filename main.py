#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.optim as optim
import torch.utils.data as data
import os
from tqdm import trange
from math import ceil

from models.rubi.baseline_net import BaselineNet
from models.rubi.rubi import RUBi
from models.rubi.loss import RUBiLoss, BaselineLoss
from tools.parse_args import parse_arguments
from utilities.earlystopping import EarlyStopping
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from utilities.schedule_lr import LrScheduler

from dataloader import DataLoaderVQA
from utilities.vocabulary_mapping import load_vocab


timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
args = parse_arguments()

# Check if GPU can be used, else use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device {}.".format(device))

print("Loading vocabulary")
vocab, vocab2id, answer2idx, idx2answer = load_vocab(args)

print("Loading datasets")
dataset = DataLoaderVQA(args, vocab, vocab2id, answer2idx)
dataloader = data.DataLoader(dataset, batch_size=args.batchsize, num_workers=args.workers, shuffle=True)
#dataloader = DataLoaderVQA(args)

print("Initialising model")
model = None
if args.baseline == "rubi":
    model = BaselineNet(dir_st=args.dir_st, vocab=vocab).to(device)
elif args.baseline == "san":
    raise NotImplementedError()
elif args.baseline == "updn":
    raise NotImplementedError()

if args.rubi:
    model = RUBi(model).to(device)
    loss = RUBiLoss(args["loss-weights"][0], args["loss-weights"][1])
else:
    loss = BaselineLoss()
smooth_loss = None
# Load pretrained model if exists
if args.pretrained_model:
    pretrained_model = torch.load(args.pretrained_model, map_device=device)
    model.load_state_dict(pretrained_model["model"])

    
if args.train:
    print("=> Entering training mode")
    
    losses_writer = open(f"losses_{timestamp}.csv", "w")
    losses_writer.write("epoch, loss, smooth_loss")
    
    # tensorboard Writer will output to /runs directory
    tensorboard_writer = SummaryWriter(filename_suffix='train')
    losses = []
    accs = []

    model.train()

    # Initialize parameters in network

    optimizer = optim.Adamax(model.parameters(), lr=args.lr)

    # use FP16
    if args.fp16:
        import apex.amp as amp
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    if args.pretrained_model:
        optimizer.load_state_dict(args.pretrained_model["optimizer"])
        if args.fp16:
            amp.load_state_dict(args.pretrained_model["amp"])

    if args.fp16:
        scheduler = LrScheduler(optimizer.optimizer)
    else:
        scheduler = LrScheduler(optimizer)

    es = EarlyStopping(min_delta=args.eps, patience=args.patience)

    try:
        with trange(args.no_epochs) as t:
            for epoch in t:

                # assume inputs is a dict
                for i_batch, inputs in enumerate(dataloader):
                    for key, value in inputs.items():
                        inputs[key] = value.to(device)

                    model.zero_grad()
                    predictions = model(inputs)
                    current_loss = loss(inputs["idx_answer"].squeeze(1), predictions)
                    losses.append(current_loss.item())

                    if args.fp16:
                        with amp.scale_loss(current_loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        current_loss.backward()

                    optimizer.step()
                    
                     # smooth lossnless sensitive to outliers
                    if smooth_loss:
                        smooth_loss = 0.99*smooth_loss + 0.01*current_loss.item()
                    else:
                        smooth_loss = current_loss.item()
                    
                    losses_writer.write("{}, {}, {}\n".format(epoch, current_loss.item(), smooth_loss))
                    
                    t.set_description(
                        f"E:{epoch} | "
                        f"Loss:{current_loss.item():.3} | "
                        f"SmoothLoss:{smooth_loss:.3} | "
                        f"Batch {i_batch}/{ceil(len(dataset)/args.batchsize)}"
                    )
                scheduler.step()

                # early stopping if loss hasn't improved

                if epoch + 1 % 5 == 0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }
                    if args.fp16:
                        checkpoint['amp'] = amp.state_dict()

                    filename = args.baseline + "_epoch_{}_dataset_{}_{}_{}.pt".format(epoch, args.dataset,
                                                                                      args.answer_type,
                                                                                      timestamp)
                    torch.save(checkpoint, os.path.join(args.dir_model, filename))

                if es.step(smooth_loss):
                    print("Early stop activated")
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
                                                                          timestamp)
        torch.save(checkpoint, os.path.join(args.dir_model, filename))
        losses_writer.close()

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

