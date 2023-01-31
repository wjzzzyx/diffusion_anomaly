# revised on https://github.com/pytorch/examples/blob/master/imagenet/main.py#L139
import collections
import numpy as np


class DefaultDictOfCustomClass(collections.defaultdict):
    def __missing__(self, key):
        self[key] = new = self.default_factory()
        return new

class AverageMeter(object):
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

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


class AverageMeterDict(dict):
    def __init__(self):
        # self.meters = collections.defaultdict(AverageMeter)
        pass
    
    def __missing__(self, key):
        self[key] = AverageMeter()
        return self[key]
    
    def reset(self):
        for key in self:
            self[key].reset()
    
    def update(self, val_dict, n=1):
        for key in val_dict:
            self[key].update(val_dict[key], n)
