import math

from torch.optim.lr_scheduler import _LRScheduler
import collections


class WarmupCosLR(_LRScheduler):
    def __init__(
        self, optimizer, min_lr, lr, starting_epoch, warmup_epochs, epochs, last_epoch=-1, verbose=False
    ) -> None:
        self.min_lr = min_lr
        assert type(lr) == list
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.starting_epoch = starting_epoch
        self.need_reset_flag = True
        super(WarmupCosLR, self).__init__(optimizer, last_epoch, verbose)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_init_lr(self):
        lr = self.lr[0] / self.warmup_epochs
        return lr

    def get_lr(self):
        """
            starting_epoch is a list containing the starting epoch of each stage. 
        """

        if self.last_epoch in self.starting_epoch:
            if self.need_reset_flag:
                self.optimizer.state = collections.defaultdict(dict) # reset optimizer internal state
                self.need_reset_flag = False
            else: 
                self.need_reset_flag = True
            

        if self.last_epoch < self.starting_epoch[0]:
            lr = 0.0
        else: 
            i = 0
            while self.last_epoch >= self.starting_epoch[i]:
                
                i += 1
                if i == len(self.starting_epoch):
                    break
            i = i-1
            if self.last_epoch < self.warmup_epochs + self.starting_epoch[i]:
                lr = self.lr[i] * (self.last_epoch - self.starting_epoch[i] + 1) / self.warmup_epochs
            else:
                lr = self.min_lr + 0.5 * (self.lr[i] - self.min_lr) * (
                    1
                    + math.cos(
                        math.pi
                        * (self.last_epoch - self.warmup_epochs - self.starting_epoch[i])
                        / ((self.starting_epoch[i+1] if i+1<len(self.starting_epoch) else self.epochs) - self.warmup_epochs - self.starting_epoch[i])
                    )
                )

        if "lr_scale" in self.optimizer.param_groups[0]:
            return [lr * group["lr_scale"] for group in self.optimizer.param_groups]

        return [lr for _ in self.optimizer.param_groups]
