#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow.keras.backend as K
from models.rubi.rubi import RUBi
from models.rubi.loss import RUBiLoss, BaselineLoss
from tools.argparse import parse_arguments
import sys

args = parse_arguments()

# set precision to 16 if required for faster training on Volta architecture
if args.fp16:
    K.set_floatx("float16")

    # default is 1e-7 which is too small for float16.  Without adjusting the epsilon, we will get NaN predictions because of divide by zero problems
    K.set_epsilon(1e-4)

model = None
if args.baseline == "baseline":
    raise NotImplementedError()
elif args.baseline == "san":
    raise NotImplementedError()
elif args.baseline == "updn":
    raise NotImplementedError()

if args.rubi:
    model = RUBi(model)
    loss = RUBiLoss(args["loss-weights"][0], args["loss-weights"][1])
else:
    loss = BaselineLoss()

optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        current_loss = loss(labels, predictions)
        gradients = tape.gradient(current_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        train_accuracy(labels, predictions)

        return current_loss

    
@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss(labels, predictions)
    
    test_accuracy(labels, predictions)

    return t_loss


if args.train:
    raise NotImplementedError()
    old_loss = 1e8
    epoch = 0
    
    while True:
        if abs(epoch - args["no-epochs"]) < args.eps:
            break
        
        images, labels = None
        new_loss = train_step(images, labels)
        epoch += 1

    print("Training complete after {} epochs.".format(epoch))
    
    # save model
    # implement a progress bar
elif args.test:
    raise NotImplementedError()
