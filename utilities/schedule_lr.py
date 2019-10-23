"""
File: schedule_lr.py

Implements the LrScheduler class
We use a learning rate of 1.5 * 10^-4 and a batch size of 256. During the first 7 epochs,
we linearly increase the learning rate to 6 * 10^-4.
After epoch 14, we apply a learning rate decay strategy which multiplies the learning rate by 0.25 every two epochs.
"""


class LrScheduler:

    def __init__(self, optimizer, init_lr=1.5e-4, peak_lr=6e-4, switch=[7, 14], gamma=0.25):
        """
        initializes the learning rate scheduler.
        This should be done before training has started (at epoch 0)
        """
        self.optimizer = optimizer
        self.last_epoch = 0
        self.lr = init_lr
        self.switch = switch
        self.gamma1 = (peak_lr - init_lr)/self.switch[0]
        self.gamma2 = gamma

    def step(self, epoch=None):
        """
        updates the learning rate
        Parameters:
        ------------
        epoch: manually set the epoch number
        """
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            if self.last_epoch <= self.switch[0]:
                param_group['lr'] += self.gamma1
            elif self.last_epoch > self.switch[1]:
                param_group['lr'] *= self.gamma2
        self.lr = lr

    def get_lr(self):
        """
        Returns the learning rate at the current epoch

        Returns:
        --------
        The new learning rate
        """
        if self.last_epoch <= self.switch[0]:
            return self.gamma1 + self.lr
        elif self.last_epoch > self.switch[1]:
            return self.gamma2 * self.lr
