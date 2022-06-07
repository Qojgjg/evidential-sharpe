from torch.nn import Module
import torch
from src.modeling.solver.metrics.classification_metrics import edl_loss, cross_entropy_loss
from src.modeling.solver.metrics.trading_metrics import cal_sharpe_ratio
import torch.nn.functional as F


class EDLLoss(Module):
    def __init__(self,
                 trans_rate,
                 annealing_step,
                 device,
                 **kwargs):
        super(EDLLoss, self).__init__()
        self.trans_rate = trans_rate
        self.annealing_step = annealing_step
        self.device = device

    def forward(self,
                evidence: torch.Tensor,
                target: torch.Tensor,
                epoch_idx: int,
                **kwargs):
        loss = edl_loss(evidence, target, epoch_idx, self.annealing_step)

        return loss


class CrossEntropyLoss(Module):
    def __init__(self,
                 trans_rate,
                 annealing_step,
                 device,
                 **kwargs):
        super(CrossEntropyLoss, self).__init__()
        self.trans_rate = trans_rate
        self.annealing_step = annealing_step
        self.device = device

    def forward(self,
                target: torch.Tensor,
                logit: torch.Tensor,
                **kwargs):
        loss = cross_entropy_loss(target, logit)

        return loss


class NegativeSharpeRatioLoss(Module):
    def __init__(self,
                 trans_rate,
                 annealing_step,
                 device,
                 **kwargs):
        super(NegativeSharpeRatioLoss, self).__init__()
        self.trans_rate = trans_rate
        self.device = device
        self.annealing_step = annealing_step

    def forward(self,
                logit,
                future_return,
                **kwargs):
        weight = F.softmax(logit, dim=1)
        sharpe = cal_sharpe_ratio(weight,
                                  future_return,
                                  trans_rate=self.trans_rate,
                                  device=self.device)

        return -sharpe
