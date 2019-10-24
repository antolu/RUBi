#!/usr/bin/env python3

from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import csv
import os


def plot_loss(args):

    losses = []
    smooth_losses = []
    with open(args.lossfile, "r") as f:
        csv_reader = csv.reader(f, delimiter=",")
        headers = next(csv_reader)
        for row in csv_reader:
            losses.append(float(row[1]))
            smooth_losses.append((float(row[2])))

    losses = np.asarray(losses)
    smooth_losses = np.asarray(smooth_losses)

    plt.semilogx(losses)
    plt.semilogx(smooth_losses)
    plt.legend(['Loss', 'Smooth loss'])
    plt.ylabel('NLL loss')
    plt.xlabel('Iteration')

    filename = os.path.splitext(os.path.basename(args.lossfile))[0] + ".eps"
    plt.savefig(os.path.join('plots', filename), format='eps')
    plt.show()
    plt.clf()


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("lossfile", type=str, help="The csv file containing loss values")

    args = parser.parse_args()

    plot_loss(args)
