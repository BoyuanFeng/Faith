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


    def concretize_l(self, lw=None):
        if lw is None: lw = self.lw
        return -self.eps * torch.norm(lw, p=self.q, dim=-2)

    def concretize_u(self, uw=None):
        if uw is None: uw = self.uw        
        return self.eps * torch.norm(uw, p=self.q, dim=-2)

    def concretize(self):
        dim = self.lw.shape[-2] // self.perturbed_words
        res_l = self.lb.clone()
        res_u = self.ub.clone()
        for i in range(self.perturbed_words):
            res_l += self.concretize_l(self.lw[:, :, (dim * i) : (dim * (i + 1)), :])
            res_u += self.concretize_u(self.uw[:, :, (dim * i) : (dim * (i + 1)), :])
        return res_l, res_u

    def new(self):
        l, u = self.concretize()

        mask_pos = torch.gt(l, 0).to(torch.float)
        mask_neg = torch.lt(u, 0).to(torch.float)
        mask_both = 1 - mask_pos - mask_neg 

        lw = torch.zeros(self.lw.shape).to(self.device)
        lb = torch.zeros(self.lb.shape).to(self.device)
        uw = torch.zeros(self.uw.shape).to(self.device)
        ub = torch.zeros(self.ub.shape).to(self.device)

        return l, u, mask_pos, mask_neg, mask_both, lw, lb, uw, ub

    def add_linear(self, mask, w_out, b_out, type, k, x0, y0, src=None):
        if mask is None: 
            mask_w = mask_b = 1
        else:
            mask_w = mask.unsqueeze(2)
            mask_b = mask
        if src is None:
            src = self
        if type == "lower":
            w_pos, b_pos = src.lw, src.lb
            w_neg, b_neg = src.uw, src.ub
        else:
            w_pos, b_pos = src.uw, src.ub
            w_neg, b_neg = src.lw, src.lb
        mask_pos = torch.gt(k, 0).to(torch.float)
        w_out += mask_w * mask_pos.unsqueeze(2) * w_pos * k.unsqueeze(2)
        b_out += mask_b * mask_pos * ((b_pos - x0) * k + y0)
        mask_neg = 1 - mask_pos
        w_out += mask_w * mask_neg.unsqueeze(2) * w_neg * k.unsqueeze(2)
        b_out += mask_b * mask_neg * ((b_neg - x0) * k + y0)

    def relu(self):
        l, u, mask_pos, mask_neg, mask_both, lw, lb, uw, ub = self.new()

        if self.ibp:
            lb = torch.max(l, torch.tensor(0.).cuda())
            ub = torch.max(u, torch.tensor(0.).cuda())
        else:
            self.add_linear(
                mask=mask_neg, w_out=lw, b_out=lb, type="lower",
                k=torch.zeros(l.shape).to(self.device), x0=0, y0=0
            )
            self.add_linear(
                mask=mask_neg, w_out=uw, b_out=ub, type="upper",
                k=torch.zeros(l.shape).to(self.device), x0=0, y0=0
            )        

            self.add_linear(
                mask=mask_pos, w_out=lw, b_out=lb, type="lower",
                k=torch.ones(l.shape).to(self.device), x0=0, y0=0
            )
            self.add_linear(
                mask=mask_pos, w_out=uw, b_out=ub, type="upper",
                k=torch.ones(l.shape).to(self.device), x0=0, y0=0
            )        

            k = u / (u - l + epsilon)
            self.add_linear(
                mask=mask_both, w_out=uw, b_out=ub, type="upper",
                k=k, x0=l, y0=0
            )

            k = torch.gt(torch.abs(u), torch.abs(l)).to(torch.float)

            self.add_linear(
                mask=mask_both, w_out=lw, b_out=lb, type="lower",
                k=k, x0=0, y0=0
            )

        return Bounds(
            self.args, self.p, self.eps,
            lw = lw, lb = lb,
            uw = uw, ub = ub
        )     
