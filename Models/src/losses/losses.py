# Src code of loss functions
import torch
from torchmetrics import Metric
import torch.nn as nn

eps = 1e-7

class CustomCrossEntropyLoss:
    def __init__(self, lambd_pres=1, lambd_abs=1):
        super().__init__()
        # print('in my custom')
        self.lambd_abs = lambd_abs
        self.lambd_pres = lambd_pres

    def __call__(self, pred, target, reduction='mean'):
        """
        target: ground truth
        pred: prediction
        reduction: mean, sum, none
        """
        loss = (-self.lambd_pres * target * torch.log(pred + eps) - self.lambd_abs * (1 - target) * torch.log(
            1 - pred + eps))
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        else:  # reduction = None
            loss = loss

        return loss

class CustomCrossEntropy(Metric):
    def __init__(self, lambd_pres=1, lambd_abs=1, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.lambd_abs = lambd_abs
        self.lambd_pres = lambd_pres
        self.add_state("correct", default=torch.FloatTensor([0]), dist_reduce_fx="sum")
        self.add_state("total", default=torch.FloatTensor([0]), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        target: target distribution
        pred: predicted distribution
        """
        self.correct += (-self.lambd_pres * target * torch.log(pred) - self.lambd_abs * (1 - target) * torch.log(
            1 - pred)).sum()
        self.total += target.numel()

    def compute(self):
        return (self.correct / self.total)
