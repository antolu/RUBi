"""
File: schedule_lr.py

Implements the LrScheduler class
We use a learning rate of 1.5 * 10^-4 and a batch size of 256. During the first 7 epochs,
we linearly increase the learning rate to 6 * 10^-4.
After epoch 14, we apply a learning rate decay strategy which multiplies the learning rate by 0.25 every two epochs.
"""

class LrScheduler:

    def __init__(self, init_lr=1.5e-4, switch=[7, 14]):
        """
        initializes the learning rate scheduler.
        This should be done before training has started (at epoch 0)
        """
        self.epoch = 0
        self.lr = init_lr
        self.switch = switch
        self.gamma1 = 4/self.switch[0]
        self.gamma2 = 0.25

    def step(self, epoch=None):
        """
        updates the learning rate
        param epoch: manually set the epoch number
        """
        if epoch:
            self.epoch = epoch
        if self.epoch <= self.switch[0]:
            self.lr = self.gamma1*self.lr
        elif self.epoch > self.switch[1]:
            self.lr = self.gamma2*self.lr
        self.epoch += 1
        return self.lr

    def get_lr(self):
        """
        returns the current learning rate
        """
        return self.lr
