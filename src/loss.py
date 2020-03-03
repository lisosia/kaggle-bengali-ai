#!/usr/bin/env python
# https://github.com/tugstugi/pytorch-speech-commands/blob/master/mixup.py

"""
Simple implementation for mixup. The loss and onehot functions origin from: https://github.com/moskomule/mixup.pytorch
Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz: mixup: Beyond Empirical Risk Minimization
https://arxiv.org/abs/1710.09412
"""

__author__ = 'Yuan Xu, Erdene-Ochir Tuguldur'

__all__ = [ 'mixup_cross_entropy_loss', 'mixup', 'mixup_multi_targets', 'cutmix_multi_targets', 'onehot',
            'COUNT_GRAPHEME', 'COUNT_VOWEL', 'COUNT_CONSONANT', 'mixup_binary_cross_entropy_loss',
            'mk_gridmask', 'gridmix_multi_targets', 'cut4mix_multi_targets']

import numpy as np
import torch
from torch.autograd import Variable

import config
C = config.get_config("./config/001_seresnext_mixup.yml")

# CLASS WEIGHT
COUNT_GRAPHEME = [147, 145, 337, 318, 331, 175, 308, 153, 157, 444, 152, 151, 146, 5420, 796, 1083, 940, 761, 1633, 278, 336, 942, 2961, 5149, 336, 1127, 171, 305, 759, 2780, 437, 768, 1127, 136, 276, 476, 1024, 285, 3354, 617, 757, 305, 1957, 3630, 1057, 144, 580, 452, 1376, 321, 738, 326, 935, 3690, 592, 1680, 2688, 633, 1285, 2339, 426, 575, 868, 149, 5596, 1365, 786, 475, 631, 757, 957, 2936, 5736, 130, 1518, 1127, 1936, 957, 293, 3458, 456, 3438, 292, 1418, 460, 1363, 2094, 168, 760, 2313, 627, 1539, 1116, 622, 973, 727, 4926, 481, 627, 458, 448, 1083, 141, 3461, 160, 151, 751, 5321, 158, 908, 342, 787, 886, 4395, 150, 4015, 436, 1531, 1139, 1537, 1207, 462, 2313, 2073, 2188, 813, 159, 927, 952, 978, 144, 443, 1039, 4374, 623, 637, 1051, 562, 934, 2312, 883, 1746, 1067, 609, 612, 317, 302, 4392, 1723, 2402, 2311, 1248, 607, 1553, 732, 928, 790, 324, 143, 3281, 480, 311, 465, 165, 164, 1142, 307, 1585]
COUNT_VOWEL = [41508, 36886, 25967, 16152, 18848, 5297, 4336, 28723, 3528, 16032, 3563]
COUNT_CONSONANT = [125278, 7424, 23465, 619, 21270, 21397, 1387]
def inverse_sqrt_and_norm(arr):
    inv = 1. / np.sqrt(np.array(arr))
    return inv / np.median(inv)  # scale by median. scale so that median weight is 1 
def eff_weight(arr, beta):
    inv = np.array([(1. - beta) / (1. - beta**n) for n in arr])
    ret = inv / np.median(inv)
    print("weight", ret)
    return ret

if False:
    WEIGHT_GRAPHEME  = torch.Tensor(inverse_sqrt_and_norm(COUNT_GRAPHEME).reshape(1, 168)).to(C.device)
    WEIGHT_VOWEL     = torch.Tensor(inverse_sqrt_and_norm(COUNT_VOWEL).reshape(1, 11)).to(C.device)
    WEIGHT_CONSONANT = torch.Tensor(inverse_sqrt_and_norm(COUNT_CONSONANT).reshape(1, 7)).to(C.device)
    CLASS_WEIGHTS = [WEIGHT_GRAPHEME, WEIGHT_VOWEL, WEIGHT_CONSONANT]
else:
    WEIGHT_GRAPHEME  = torch.Tensor(eff_weight(COUNT_GRAPHEME, 1-1./2000).reshape(1, 168)).to(C.device)
    WEIGHT_VOWEL     = torch.Tensor(np.ones_like(COUNT_VOWEL).reshape(1, 11)).to(C.device)
    WEIGHT_CONSONANT = torch.Tensor(np.ones_like(COUNT_CONSONANT).reshape(1, 7)).to(C.device)
    CLASS_WEIGHTS = [WEIGHT_GRAPHEME, WEIGHT_VOWEL, WEIGHT_CONSONANT]

def mixup_cross_entropy_loss(input, target, class_dx, size_average=True):
    """Origin: https://github.com/moskomule/mixup.pytorch
    in PyTorch's cross entropy, targets are expected to be labels
    so to predict probabilities this loss is needed
    suppose q is the target and p is the input
    loss(p, q) = -\sum_i q_i \log p_i
    """
    assert input.size() == target.size()
    assert isinstance(input, Variable) and isinstance(target, Variable)
    input = torch.log(torch.nn.functional.softmax(input, dim=1).clamp(1e-5, 1))
    if C.use_class_weight:
        loss = - torch.sum(input * target * CLASS_WEIGHTS[class_dx])
    else:
        # input = input - torch.log(torch.sum(torch.exp(input), dim=1)).view(-1, 1)
        loss = - torch.sum(input * target)

    return loss / input.size()[0] if size_average else loss

def mixup_binary_cross_entropy_loss(input, target, size_average=True):
    assert input.size() == target.size()
    assert isinstance(input, Variable) and isinstance(target, Variable)
    assert input.size(1) == 61
    input1 = torch.log(     torch.nn.functional.sigmoid(input).clamp(1e-5, 1))
    input2 = torch.log((1 - torch.nn.functional.sigmoid(input)).clamp(1e-5, 1))
    loss = - (torch.sum(input1 * target) + torch.sum(input2 * (1.-target)))

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
def cutmix_multi_targets(data, targets1, targets2, targets3, targets4, alpha):
    targets1, targets2, targets3 = onehot(targets1, 168), onehot(targets2, 11), onehot(targets3, 7)

    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets1 = targets1[indices]
    shuffled_targets2 = targets2[indices]
    shuffled_targets3 = targets3[indices]
    shuffled_targets4 = targets4[indices]

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)

    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    t1 = lam*targets1 + (1-lam)*shuffled_targets1
    t2 = lam*targets2 + (1-lam)*shuffled_targets2
    t3 = lam*targets3 + (1-lam)*shuffled_targets3
    t4 = lam*targets4 + (1-lam)*shuffled_targets4
    return data, t1, t2, t3,  t4
    ## original code
    #targets = [targets1, shuffled_targets1, targets2, shuffled_targets2, targets3, shuffled_targets3, lam]
    #return data, targets

def _bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    hmin, hmax = np.where(rows)[0][[0, -1]]
    wmin, wmax = np.where(cols)[0][[0, -1]]
    return hmin, hmax, wmin, wmax

def pick_center(img):  # img: [1,H,W]
    img = img.cpu().numpy()
    if False:
        plt.subplot(1,2,1)
        plt.imshow(x[0][0].cpu().numpy())

    # threth = np.percentile(img, 80)
    threth = img.min() + 0.4 * (img.max() - img.min())

    img_bin = img >= threth
    hmin, hmax, wmin, wmax = _bbox(img_bin)
    hcenter = (hmin + hmax) // 2
    wcenter = (wmin + wmax) // 2

    img_bin = img_bin.astype(float)
    img_bin[hcenter-5:hcenter+5, wcenter-5:wcenter+5] = 0.2
    if False:
        plt.subplot(1,2,2)
        plt.imshow(img_bin)
        plt.show()
    return hcenter, wcenter
    

def shift_img(_i, dh, dw):
    _i = np.roll(_i, dh, axis=0)
    _i = np.roll(_i, dw, axis=1)
    if dh >= 0:
        _i[: dh] = 0.
    else:
        _i[ dh: ] = 0.
    if dw >= 0:
        _i[:, : dw] = 0.
    else:
        _i[:, dw: ] = 0.
    return _i

def cut4mix_multi_targets(data, targets1, targets2, targets3, targets4):
    batch, c, h, w = data.size()
    targets1, targets2, targets3 = onehot(targets1, 168), onehot(targets2, 11), onehot(targets3, 7)

    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    
    out = np.zeros((batch,h,w))
    lam = np.zeros(batch)
    for b in range(batch):
        JIT = h // 8
        ch = np.random.randint(h//2-JIT, h//2+JIT) 
        cw = np.random.randint(w//2-JIT, w//2+JIT)
        
        hh, ww = pick_center(data[b, 0]) + np.random.randint(-2, 3, size=(2))
        _i  = shift_img(data[b,0].cpu().numpy(), ch - hh, cw - ww)

        hh, ww = pick_center(data[indices][b,0]) + np.random.randint(-2, 3, size=(2))
        _i2 = shift_img(data[indices][b,0].cpu().numpy(), ch - hh, cw - ww)

        cnt = 0
        out[b] = _i
        if np.random.rand() > 0.5:
            out[b][:ch, :cw] = _i2[:ch, :cw]
            cnt += 1
        if np.random.rand() > 0.5:
            out[b][:ch, cw:] = _i2[:ch, cw:]
            cnt += 1
        if np.random.rand() > 0.5:
            out[b][ch:, :cw] = _i2[ch:, :cw]
            cnt += 1
        if np.random.rand() > 0.5:
            out[b][ch:, cw:] = _i2[ch:, cw:]
            cnt += 1
        lam[b] = 1 - cnt/4.

    # lam = np.random.beta(alpha, alpha)
    # bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    lam = torch.tensor(lam).view(batch, 1).to(C.device)
    # data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    # lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    shuffled_targets1 = targets1[indices]
    shuffled_targets2 = targets2[indices]
    shuffled_targets3 = targets3[indices]
    shuffled_targets4 = targets4[indices]
    t1 = lam*targets1 + (1-lam)*shuffled_targets1
    t2 = lam*targets2 + (1-lam)*shuffled_targets2
    t3 = lam*targets3 + (1-lam)*shuffled_targets3
    t4 = lam*targets4 + (1-lam)*shuffled_targets4
    return torch.tensor(out, dtype=torch.float).view(batch, 1, h, w).to(C.device), t1, t2, t3,  t4

from PIL import Image
def mk_gridmask(h, w, d1, d2, ratio, rotate=90):
    """ratio: keep ratio (length)"""
    hh = int(1.5*h)
    ww = int(1.5*w)
    d = np.random.randint(d1, d2)
    l = int(d*ratio+0.5)
    mask = np.ones((hh, ww), np.float32)
    st_h = np.random.randint(d)
    st_w = np.random.randint(d)
    for i in range(-1, hh//d+1):
            s = d*i + st_h
            t = s+l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[s:t,:] = 0.
    for i in range(-1, ww//d+1):
            s = d*i + st_w
            t = s+l
            s = max(min(s, ww), 0)
            t = max(min(t, ww), 0)
            mask[:,s:t] = 0.
    r = np.random.randint(rotate)
    mask = Image.fromarray(np.uint8(mask))
    mask = mask.rotate(r)
    mask = np.asarray(mask)
    mask = mask[(hh-h)//2:(hh-h)//2+h, (ww-w)//2:(ww-w)//2+w]

    return mask
    ### return mask, 1-(1-ratio)*(1-ratio)
    ### return torch.from_numpy(mask).float().cuda()

def gridmix_multi_targets(data, targets1, targets2, targets3, targets4, alpha):
    device = data.device
    batch, c, h, w = data.size(0), data.size(1), data.size(2), data.size(3)

    targets1, targets2, targets3 = onehot(targets1, 168), onehot(targets2, 11), onehot(targets3, 7)

    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets1 = targets1[indices]
    shuffled_targets2 = targets2[indices]
    shuffled_targets3 = targets3[indices]
    shuffled_targets4 = targets4[indices]

    ### lam = np.random.beta(alpha, alpha)
    ### bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)

    raw_ratios = np.random.uniform(0.3, 0.4, batch)
    lam = 1 - (1 - raw_ratios) ** 2  # 0.36 ~ 0.5
    masks = torch.Tensor(np.array([mk_gridmask(
        C.image_size[0], C.image_size[1], C.image_size[0]*0.4, C.image_size[0]*0.5, _ratio) for _ratio in lam])
    ).to(device).view(batch, c, h, w)

    data[masks == 0] = data[indices][masks == 0]
    ### adjust lambda to exactly match pixel ratio
    ### lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    lam = torch.Tensor(lam).view(batch, 1).to(device)
    t1 = lam*targets1 + (1-lam)*shuffled_targets1
    t2 = lam*targets2 + (1-lam)*shuffled_targets2
    t3 = lam*targets3 + (1-lam)*shuffled_targets3
    t4 = lam*targets4 + (1-lam)*shuffled_targets4
    return data, t1, t2, t3,  t4

ALPHA = 0.4  # fastai kernel
def mixup_multi_targets(inputs, targets1, targets2, targets3, targets4, alpha):
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
    targets4 = weight*targets4 + (1-weight)*targets4[index,]

    return inputs, targets1, targets2, targets3, targets4

def mixup(inputs, targets, alpha):
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

def _mixup_not_onehot(inputs, targets, num_classes, alpha):
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
