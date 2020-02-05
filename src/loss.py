#!/usr/bin/env python
# https://github.com/tugstugi/pytorch-speech-commands/blob/master/mixup.py

"""
Simple implementation for mixup. The loss and onehot functions origin from: https://github.com/moskomule/mixup.pytorch
Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz: mixup: Beyond Empirical Risk Minimization
https://arxiv.org/abs/1710.09412
"""

__author__ = 'Yuan Xu, Erdene-Ochir Tuguldur'

__all__ = [ 'mixup_cross_entropy_loss', 'mixup', 'mixup_multi_targets' ]

import numpy as np
import torch
from torch.autograd import Variable

def mixup_cross_entropy_loss(input, target, size_average=True):
    """Origin: https://github.com/moskomule/mixup.pytorch
    in PyTorch's cross entropy, targets are expected to be labels
    so to predict probabilities this loss is needed
    suppose q is the target and p is the input
    loss(p, q) = -\sum_i q_i \log p_i
    """
    assert input.size() == target.size()
    assert isinstance(input, Variable) and isinstance(target, Variable)
    input = torch.log(torch.nn.functional.softmax(input, dim=1).clamp(1e-5, 1))
    # input = input - torch.log(torch.sum(torch.exp(input), dim=1)).view(-1, 1)
    loss = - torch.sum(input * target)
    return loss / input.size()[0] if size_average else loss

def onehot(targets, num_classes):
    """Origin: https://github.com/moskomule/mixup.pytorch
    convert index tensor into onehot tensor
    :param targets: index tensor
    :param num_classes: number of classes
    """
    ###TEST### assert isinstance(targets, torch.LongTensor)
    return torch.zeros(targets.size()[0], num_classes).scatter_(1, targets.view(-1, 1), 1)

ALPHA = 0.4  # fastai kernel
def mixup_multi_targets(inputs, targets1, targets2, targets3, alpha=ALPHA):
    """Mixup on 1x32x32 mel-spectrograms.
    """
    targets1, targets2, targets3 = onehot(targets1, 168), onehot(targets2, 11), onehot(targets3, 7)

    s = inputs.size()[0]
    weight = torch.Tensor(np.random.beta(alpha, alpha, s))
    index = np.random.permutation(s)

    x1, x2 = inputs, inputs[index, :, :, :]
    weight = weight.view(s, 1, 1, 1)
    inputs = weight*x1 + (1-weight)*x2

    weight = weight.view(s, 1)
    targets1 = weight*targets1 + (1-weight)*targets1[index,]
    targets2 = weight*targets2 + (1-weight)*targets2[index,]
    targets3 = weight*targets3 + (1-weight)*targets3[index,]

    return inputs, targets1, targets2, targets3

def mixup(inputs, targets, alpha=ALPHA):
    """Mixup on 1x32x32 mel-spectrograms.
    """
    s = inputs.size()[0]
    weight = torch.Tensor(np.random.beta(alpha, alpha, s))
    index = np.random.permutation(s)
    x1, x2 = inputs, inputs[index, :, :, :]
    y1, y2 = targets, targets[index,]
    weight = weight.view(s, 1, 1, 1)
    inputs = weight*x1 + (1-weight)*x2
    weight = weight.view(s, 1)
    targets = weight*y1 + (1-weight)*y2
    return inputs, targets

def _mixup_not_onehot(inputs, targets, num_classes, alpha=ALPHA):
    """Mixup on 1x32x32 mel-spectrograms.
    """
    s = inputs.size()[0]
    weight = torch.Tensor(np.random.beta(alpha, alpha, s))
    index = np.random.permutation(s)
    x1, x2 = inputs, inputs[index, :, :, :]
    y1, y2 = onehot(targets, num_classes), onehot(targets[index,], num_classes)
    weight = weight.view(s, 1, 1, 1)
    inputs = weight*x1 + (1-weight)*x2
    weight = weight.view(s, 1)
    targets = weight*y1 + (1-weight)*y2
    return inputs, targets