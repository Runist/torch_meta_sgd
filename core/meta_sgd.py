# -*- coding: utf-8 -*-
# @File : meta_sgd.py
# @Author: Runist
# @Time : 2022/9/2 15:00
# @Software: PyCharm
# @Brief:

import torch
import torch.nn as nn
import numpy as np
import collections


class MetaSGD:
    def __init__(self, model):
        self.model = model
        self.alpha = [v for _, v in model.named_parameters()]

    def update_weights(self, loss, weights):
        self.alpha = [v for _, v in self.model.named_parameters()]

        grads = torch.autograd.grad(loss, weights.values(), create_graph=True)

        meta_weights = []
        for i, ((name, param), alpha, grad) in enumerate(zip(weights.items(), self.alpha, grads)):
            meta_weights.append((name, param - torch.mul(alpha, grad)))

        meta_weights = collections.OrderedDict(meta_weights)

        return meta_weights

