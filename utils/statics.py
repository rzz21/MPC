import torch

__all__ = ['AverageMeter', 'RMSE_evaluator', 'NMSE_evaluator', 'check_channel']


class AverageMeter(object):
    r"""Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self, name):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.name = name

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"==> For {self.name}: sum={self.sum}; avg={self.avg}"


def RMSE_evaluator(pos_pred, pos_gt, ll):
    r""" Evaluation of position prediction implemented in PyTorch Tensor
         Computes root mean square error (RMSE).
    """

    with torch.no_grad():
        # Calculate the RMSE
        rmse = torch.zeros([pos_gt.shape[0],1])
        for u in range(pos_gt.shape[0]):
            mse = (pos_pred[u,:,:] - pos_gt[u,:,:]) ** 2
            mse = mse[:,:int(ll[u])]
            rmse[u] = torch.sqrt((mse.sum()/ll[u]))
        return rmse.mean()

def NMSE_evaluator(sparse_pred, sparse_gt):
    r""" Evaluation of decoding implemented in PyTorch Tensor
         Computes normalized mean square error (NMSE) and rho.
    """

    with torch.no_grad():
        # Calculate the NMSE
        power_gt = sparse_gt ** 2
        difference = sparse_gt - sparse_pred
        mse = difference ** 2
        nmse = 10 * torch.log10((mse.sum(dim=[1, 2]) / power_gt.sum(dim=[1, 2])).mean())
        return nmse


def check_channel(h, figname):
    import numpy as np
    import matplotlib.pyplot as plt
    h = h[1,:,:,:].cpu()
    h = h - 0.5
    hh = abs(np.array(torch.complex(h[0,:,:],h[1,:,:])))
    plt.imshow(hh)
    plt.savefig(figname)