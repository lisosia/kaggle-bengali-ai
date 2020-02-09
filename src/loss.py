#!/usr/bin/env python
# https://github.com/tugstugi/pytorch-speech-commands/blob/master/mixup.py

"""
Simple implementation for mixup. The loss and onehot functions origin from: https://github.com/moskomule/mixup.pytorch
Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz: mixup: Beyond Empirical Risk Minimization
https://arxiv.org/abs/1710.09412
"""

__author__ = 'Yuan Xu, Erdene-Ochir Tuguldur'

__all__ = [ 'mixup_cross_entropy_loss', 'mixup', 'mixup_multi_targets', 'cutmix_multi_targets', 'onehot']

import numpy as np
import torch
from torch.autograd import Variable

import config
C = config.get_config("./config/001_seresnext_mixup.yml")

# CLASS WEIGHT
COUNT_GRAPHEME = [147, 145, 337, 318, 331, 175, 308, 153, 157, 444, 152, 151, 146, 5420, 796, 1083, 940, 761, 1633, 278, 336, 942, 2961, 5149, 336, 1127, 171, 305, 759, 2780, 437, 768, 1127, 136, 276, 476, 1024, 285, 3354, 617, 757, 305, 1957, 3630, 1057, 144, 580, 452, 1376, 321, 738, 326, 935, 3690, 592, 1680, 2688, 633, 1285, 2339, 426, 575, 868, 149, 5596, 1365, 786, 475, 631, 757, 957, 2936, 5736, 130, 1518, 1127, 1936, 957, 293, 3458, 456, 3438, 292, 1418, 460, 1363, 2094, 168, 760, 2313, 627, 1539, 1116, 622, 973, 727, 4926, 481, 627, 458, 448, 1083, 141, 3461, 160, 151, 751, 5321, 158, 908, 342, 787, 886, 4395, 150, 4015, 436, 1531, 1139, 1537, 1207, 462, 2313, 2073, 2188, 813, 159, 927, 952, 978, 144, 443, 1039, 4374, 623, 637, 1051, 562, 934, 2312, 883, 1746, 1067, 609, 612, 317, 302, 4392, 1723, 2402, 2311, 1248, 607, 1553, 732, 928, 790, 324, 143, 3281, 480, 311, 465, 165, 164, 1142, 307, 1585]
COUNT_VOWEL = [41508, 36886, 25967, 16152, 18848, 5297, 4336, 28723, 3528, 16032, 3563]
COUNT_CONSONANT = [125278, 7424, 23465, 619, 21270, 21397, 1387]
def inverse_and_norm(arr):
    inv = 1. / np.array(arr)
    return inv / np.sum(inv) * len(arr)  # equal scale to [1,...,1]
WEIGHT_GRAPHEME  = torch.Tensor(inverse_and_norm(COUNT_GRAPHEME).reshape(1, 168)).to(C.device)
WEIGHT_VOWEL     = torch.Tensor(inverse_and_norm(COUNT_VOWEL).reshape(1, 11)).to(C.device)
WEIGHT_CONSONANT = torch.Tensor(inverse_and_norm(COUNT_CONSONANT).reshape(1, 7)).to(C.device)
CLASS_WEIGHTS = [WEIGHT_GRAPHEME, WEIGHT_VOWEL, WEIGHT_CONSONANT]

def mixup_cross_entropy_loss(input, target, class_weight_dx, size_average=True):
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
    loss = - torch.sum(input * target * CLASS_WEIGHTS[class_weight_dx])
    return loss / input.size()[0] if size_average else loss

def onehot(targets, num_classes):
    """Origin: https://github.com/moskomule/mixup.pytorch
    convert index tensor into onehot tensor
    :param targets: index tensor
    :param num_classes: number of classes
    """
    ###TEST### assert isinstance(targets, torch.LongTensor)
    return torch.zeros(targets.size()[0], num_classes).to(C.device).scatter_(1, targets.view(-1, 1), 1)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# https://www.kaggle.com/c/bengaliai-cv19/discussion/126504
def cutmix_multi_targets(data, targets1, targets2, targets3, alpha):
    targets1, targets2, targets3 = onehot(targets1, 168), onehot(targets2, 11), onehot(targets3, 7)

    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets1 = targets1[indices]
    shuffled_targets2 = targets2[indices]
    shuffled_targets3 = targets3[indices]

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)

    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    t1 = lam*targets1 + (1-lam)*shuffled_targets1
    t2 = lam*targets2 + (1-lam)*shuffled_targets2
    t3 = lam*targets3 + (1-lam)*shuffled_targets3
    return data, t1, t2, t3
    ## original code
    #targets = [targets1, shuffled_targets1, targets2, shuffled_targets2, targets3, shuffled_targets3, lam]
    #return data, targets

ALPHA = 0.4  # fastai kernel
def mixup_multi_targets(inputs, targets1, targets2, targets3, alpha=ALPHA):
    """Mixup on 1x32x32 mel-spectrograms.
    """
    targets1, targets2, targets3 = onehot(targets1, 168), onehot(targets2, 11), onehot(targets3, 7)

    s = inputs.size()[0]
    weight = torch.Tensor(np.random.beta(alpha, alpha, s)).to(C.device)
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