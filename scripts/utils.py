import math
import numpy as np

class AverageMeter(object):
    def __init__(self):
        self.val = None
        self.sum = None
        self.cnt = None
        self.avg = None
        self.ema = None
        self.initialized = False

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def initialize(self, val, n):
        self.val = val
        self.sum = val * n
        self.cnt = n
        self.avg = val
        self.ema = val
        self.initialized = True

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        self.ema = self.ema * 0.99 + self.val * 0.01

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


modality_dir = {"CT": 0, "MR": 1, "US": 2}
region_dir = {"abd": 0, "lung": 1, "af": 2, "brain": 3, "hip": 4, "knee": 5, "shoulder": 6, "spine": 7, "thyroid": 8}

modality_dir_inv = {v: k for k, v in modality_dir.items()}
region_dir_inv = {v: k for k, v in region_dir.items()}

def one_hot2label(target):
    # 12次元のベクトルを3次元と9次元に分割し、それぞれの最大値のインデックスを取得
    #print(target)
    modality = target[0][:3].argmax().item()
    #print(modality)
    region = target[0][3:].argmax().item()
    #print(region)
    # それぞれのインデックスをmodality_dirとregion_dirに変換
    # それぞれの文字列ラベルを返す
    return modality_dir_inv[modality] + "_" + region_dir_inv[region]