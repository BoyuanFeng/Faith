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

    def print(self, message):
        print(message)
        l, u = self.concretize()
        print("mean abs %.5f %.5f" % (torch.mean(torch.abs(l)), torch.mean(torch.abs(u))))
        print("diff %.5f %.5f %.5f" % (torch.min(u - l), torch.max(u - l), torch.mean(u - l)))
        print("lw norm", torch.mean(torch.norm(self.lw, dim=-2)))
        print("uw norm", torch.mean(torch.norm(self.uw, dim=-2)))
        print("uw - lw norm", torch.mean(torch.norm(self.uw - self.lw, dim=-2)))
        print("min", torch.min(l))
        print("max", torch.max(u))
        print()

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

        lw = torch.zeros(self.lw.shape).to(self.device)
        lb = torch.zeros(self.lb.shape).to(self.device)
        uw = torch.zeros(self.uw.shape).to(self.device)
        ub = torch.zeros(self.ub.shape).to(self.device)

        l, u = lb, ub
        mask_pos = torch.gt(l, 0).to(torch.float)
        mask_neg = torch.lt(u, 0).to(torch.float)
        mask_both = 1 - mask_pos - mask_neg 

        return l, u, mask_pos, mask_neg, mask_both, lw, lb, uw, ub

    def get_bounds_xy(self, l_x, u_x, l_y, u_y, debug=False):
        if self.ibp:
            prod1 = l_x * l_y
            prod2 = l_x * u_y
            prod3 = u_x * l_y
            prod4 = u_x * u_y

            l = torch.min(prod1, torch.min(prod2, torch.min(prod3, prod4)))
            u = torch.max(prod1, torch.max(prod2, torch.max(prod3, prod4)))

            zeros = torch.zeros(l_x.shape).cuda()

            return zeros, zeros, l, zeros, zeros, u

        alpha_l = l_y
        beta_l = l_x
        gamma_l = -alpha_l * beta_l        

        alpha_u = u_y
        beta_u = l_x
        gamma_u = -alpha_u * beta_u 

        return alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u

    """
    Dot product for multi-head self-attention (also used for obtaining context)

    A, B [b * h, l, in, out]

    For each one in the batch:

    d[i][j] \approx \sum_k a[i][k] * b[k][j]
            \approx \sum_k (\sum_m A[i][m][k] * x^r[m])(\sum_m B[j][m][k] * x^r[m])
        
    With a relaxation on b[k][j], so that b[k][j] \in [l[k][j], r[k][j]]:
        d[i][j] \approx \sum_k (\sum_m A[i][m][k] * x^r[m]) * b[j][k]
                = \sum_m (\sum_k A[i][m][k] * b[j][k]) * x^r[m]
        
        Consider the signs of A^L[i][m][k], A^U[i][m][k], b^L[j][k], b^U[j][k]
        Most coarse/loose first:
            D^u[i][j][m] = sum_k max(abs(A^L[i][m][k]), abs(A^U[i][m][k])) * \
                max(abs(b^L[j][k]), abs(b^U[j][k]))
            D^l[i][j][m] = -d^u[i][j]
    """
    def dot_product(self, other, debug=False, verbose=False, lower=True, upper=True):
        if self.dim_in == 1:
            l1, u1 = self.lb.unsqueeze(-2), self.ub.unsqueeze(-2)
            l2, u2 = other.lb.unsqueeze(1), other.ub.unsqueeze(1)
            prod1, prod2, prod3, prod4 = l1 * l2, l1 * u2, u1 * l2, u1 * u2
            l = torch.min(torch.min(prod1, prod2), torch.min(prod3, prod4)).sum(-1)
            u = torch.max(torch.max(prod1, prod2), torch.max(prod3, prod4)).sum(-1)
            w = l.unsqueeze(-2) * 0
            return Bounds(
                self.args,  self.p, self.eps,
                lw = w, lb = l, uw = w, ub = u
            )

        l_a, u_a = self.concretize()
        l_b, u_b = other.concretize()

        lw = torch.zeros(self.lw.shape[0], self.lw.shape[1], self.dim_in, other.lw.shape[1]).to(self.device)
        uw = torch.zeros(self.lw.shape[0], self.lw.shape[1], self.dim_in, other.lw.shape[1]).to(self.device)
        lb = torch.zeros(self.lw.shape[0], self.lw.shape[1], other.lw.shape[1]).to(self.device)
        ub = torch.zeros(self.lw.shape[0], self.lw.shape[1], other.lw.shape[1]).to(self.device)

        for t in range(self.batch_size):
            alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u = \
                self.get_bounds_xy(
                    l_a[t].repeat(1, other.length).reshape(-1),
                    u_a[t].repeat(1, other.length).reshape(-1),
                    l_b[t].reshape(-1).repeat(self.length),
                    u_b[t].reshape(-1).repeat(self.length),
                    debug=debug
                )
                        
            alpha_l = alpha_l.reshape(self.length, other.length, self.dim_out)
            beta_l = beta_l.reshape(self.length, other.length, self.dim_out)
            gamma_l = gamma_l.reshape(self.length, other.length, self.dim_out)
            alpha_u = alpha_u.reshape(self.length, other.length, self.dim_out)
            beta_u = beta_u.reshape(self.length, other.length, self.dim_out)
            gamma_u = gamma_u.reshape(self.length, other.length, self.dim_out)

            lb[t] += torch.sum(gamma_l, dim=-1)
            ub[t] += torch.sum(gamma_u, dim=-1)

            matmul_batch_size = 128#64

            def add_w_alpha(new, old, weight, cmp):
                a = old[t].reshape(self.length, self.dim_in, 1, self.dim_out)
                b = (weight * cmp(weight, 0).to(torch.float))\
                    .reshape(self.length, 1, other.length, self.dim_out)\
                    .transpose(2, 3) 
                new[t, :, :, :] += a[:, :, :, :].matmul(b[:, :, :, :])\
                    .reshape(self.length, self.dim_in, other.length) 

            def add_b_alpha(new, old, weight, cmp):       
                new[t, :, :] += \
                    ((old[t].reshape(self.length, 1, self.dim_out))\
                    .bmm((weight * cmp(weight, 0).to(torch.float))\
                        .reshape(self.length, other.length, self.dim_out)\
                        .transpose(1, 2))\
                    .reshape(self.length, other.length))                    

            def add_w_beta(new, old, weight, cmp): 
                a = old[t].reshape(other.length, self.dim_in, 1, self.dim_out)
                b = (weight * cmp(weight, 0).to(torch.float))\
                    .transpose(0, 1)\
                    .reshape(other.length, 1, self.length, self.dim_out)\
                    .transpose(2, 3)
                new[t, :, :, :] += a[:, :, :, :].matmul(b[:, :, :, :])\
                    .reshape(other.length, self.dim_in, self.length).transpose(0, 2) 

            def add_b_beta(new, old, weight, cmp):
                new[t, :, :] += \
                    ((old[t].reshape(other.length, 1, self.dim_out))\
                    .bmm((weight * cmp(weight, 0).to(torch.float))\
                        .transpose(0, 1)\
                        .reshape(other.length, self.length, self.dim_out)\
                        .transpose(1, 2))\
                    .reshape(other.length, self.length)).transpose(0, 1)                        

            if lower:
                add_w_alpha(lw, self.lw, alpha_l, torch.gt)
                add_w_alpha(lw, self.uw, alpha_l, torch.lt)
                add_w_beta(lw, other.lw, beta_l, torch.gt)
                add_w_beta(lw, other.uw, beta_l, torch.lt)

                add_b_alpha(lb, self.lb, alpha_l, torch.gt)
                add_b_alpha(lb, self.ub, alpha_l, torch.lt)  
                add_b_beta(lb, other.lb, beta_l, torch.gt)
                add_b_beta(lb, other.ub, beta_l, torch.lt)         

            if upper:                 
                add_w_alpha(uw, self.uw, alpha_u, torch.gt)
                add_w_alpha(uw, self.lw, alpha_u, torch.lt)
                add_w_beta(uw, other.uw, beta_u, torch.gt)
                add_w_beta(uw, other.lw, beta_u, torch.lt)

                add_b_alpha(ub, self.ub, alpha_u, torch.gt)
                add_b_alpha(ub, self.lb, alpha_u, torch.lt)
                add_b_beta(ub, other.ub, beta_u, torch.gt)
                add_b_beta(ub, other.lb, beta_u, torch.lt)            

        return Bounds(
            self.args,  self.p, self.eps,
            lw = lw, lb = lb, uw = uw, ub = ub
        )
