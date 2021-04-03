import torch
from torch import nn
from torch.nn import functional as F


class LabelSmoothingLoss(nn.Module):
    def __init__(self, ignore_index, eps=0.1, reduction="mean"):
        super().__init__()

        self.ignore_index = ignore_index
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        n_class = output.shape[-1]
        output = F.log_softmax(output, -1)

        if self.ignore_index > -1:
            n_class -= 1

        true_dist = torch.full_like(output, self.eps / n_class)
        true_dist.scatter_(
            1, target.data.unsqueeze(1), 1 - self.eps + self.eps / n_class
        )

        if self.ignore_index > -1:
            true_dist[:, self.ignore_index] = 0
            padding_mat = target.data == self.ignore_index
            mask = torch.nonzero(padding_mat, as_tuple=False)

            if mask.dim() > 0:
                true_dist.index_fill_(0, mask.squeeze(), 0.0)

        loss = F.kl_div(
            output,
            true_dist.detach(),
            reduction="sum" if self.reduction != "none" else "none",
        )

        if self.reduction == "none":
            loss = loss.sum(1)

        elif self.reduction == "mean":
            if self.ignore_index > -1:
                loss = loss / (target.shape[0] - padding_mat.sum().item())

            else:
                loss = loss / target.shape[0]

        return loss


class MixLoss(nn.Module):
    def __init__(self, eps=0, reduction="mean"):
        super().__init__()

        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target1, target2, interpolation):
        n_class = output.shape[-1]
        output = F.log_softmax(output, -1)

        true_dist = torch.full_like(output, self.eps / n_class)
        true1 = true_dist.scatter(
            1, target1.data.unsqueeze(1), 1 - self.eps + self.eps / n_class
        )
        true2 = true_dist.scatter(
            1, target2.data.unsqueeze(1), 1 - self.eps + self.eps / n_class
        )
        inter = torch.as_tensor(interpolation).unsqueeze(-1)
        true_dist = inter * true1 + (1 - inter) * true2

        loss = F.kl_div(
            output,
            true_dist.detach(),
            reduction="sum" if self.reduction != "none" else "none",
        )

        if self.reduction == "none":
            loss = loss.sum(1)

        elif self.reduction == "mean":
            loss = loss / target1.shape[0]

        return loss
