import torch.nn.functional as F


def compute_losses(output, target):
    # loss_func = F.smooth_l1_loss
    # loss_func = F.l1_loss
    loss_func = F.mse_loss
    losses = loss_func(output, target)
    return {"total": losses}
