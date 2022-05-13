# Copyright (c) 2020, Zhouxing shi <zhouxingshichn@gmail.com>
# Licenced under the BSD 2-Clause License.

import torch
import torch.nn as nn
import math, time

epsilon = 1e-12

class Bounds:
    # W: actually transposed versions are stored
    def __init__(self, args, p, eps, w=None, b=None, lw=None, lb=None, uw=None, ub=None, clone=True):
        self.args = args
        self.ibp = args.method == "ibp"
        self.device = lw.device if lw is not None else w.device
        self.p = p
        self.q = 1. / (1. - 1. / p) if p != 1 else float("inf") # dual norm
        self.eps = eps 
        self.perturbed_words = args.perturbed_words        
        self.lw = lw if lw is not None else (w.clone() if clone else w)
        self.uw = uw if uw is not None else (w.clone() if clone else w)
        self.lb = lb if lb is not None else (b.clone() if clone else b)
        self.ub = ub if ub is not None else (b.clone() if clone else b)
        if self.ibp:
            self.lw, self.uw = \
                self.lw[:, :, :self.perturbed_words, :],\
                self.uw[:, :, :self.perturbed_words, :]        
        self.update_shape()

    def update_shape(self):
        self.batch_size = self.lw.shape[0]
        self.length = self.lw.shape[1]
        self.dim_in = self.lw.shape[2]
        self.dim_out = self.lw.shape[3]   

    def matmul(self, W):
        if type(W) == Bounds:
            raise NotImplementedError
        elif len(W.shape) == 2:
            W = W.t()

            pos_mask = torch.gt(W, 0).to(torch.float32)
            W_pos = W * pos_mask
            W_neg = W - W_pos

            return Bounds(
                self.args, self.p, self.eps,
                lw = self.lw.matmul(W_pos) + self.uw.matmul(W_neg),
                lb = self.lb.matmul(W_pos) + self.ub.matmul(W_neg),
                uw = self.lw.matmul(W_neg) + self.uw.matmul(W_pos),
                ub = self.lb.matmul(W_neg) + self.ub.matmul(W_pos)
            )
        else:
            W = W.transpose(1, 2)

            pos_mask = torch.gt(W, 0).to(torch.float32)
            W_pos = W * pos_mask
            W_neg = W - W_pos

            return Bounds(
                self.args, self.p, self.eps,
                lw = (self.lw.squeeze(0).bmm(W_pos) + self.uw.squeeze(0).bmm(W_neg)).unsqueeze(0),
                lb = (self.lb.transpose(0, 1).bmm(W_pos) + self.ub.transpose(0, 1).bmm(W_neg)).transpose(0, 1),
                uw = (self.lw.squeeze(0).bmm(W_neg) + self.uw.squeeze(0).bmm(W_pos)).unsqueeze(0),
                ub = (self.lb.transpose(0, 1).bmm(W_neg) + self.ub.transpose(0, 1).bmm(W_pos)).transpose(0, 1)
            )            
