"""
To test the computation correctness of relu_verification, please use:
    python forward_test_bound.py
We note that:
    1) Use FLAG to control whether we are checking correctness of "mat_mul", "dot_product", or "relu"
"""

import os
if not "CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import math, time
from torch.utils.cpp_extension import load

torch.set_printoptions(precision=6)

FLAG = "mat_mul" # ["mat_mul", "relu", "dot_product_QK", "dot_product_V", "tanh"]
epsilon = 1e-12

# base_dirc = "../Kernels/"
base_dirc = "/home/boyuan/verification/Faith-NNVerificationCompiler/" # Change this to the path on your machine.
base_dirc += "HandTunedKernels/"
# base_dirc = "/home/boyuan/Faith-NNVerificationCompiler/HandTunedKernels/"
c_bound = load(name="c_relu_verification", sources=[
    base_dirc+"kernel.cpp", 
    base_dirc+"relu_verification.cu", 
    base_dirc+"dot_product_verification.cu",
    # base_dirc+"matmul_verification.cu",
    base_dirc+"tanh_verification.cu",
    base_dirc+"convolution_verification.cu",
    ], verbose=False)

class Bounds:
    # W: actually transposed versions are stored
    def __init__(self, p, eps, lw, lb, uw, ub):
        self.device = lw.device
        self.p = p
        self.q = 1. / (1. - 1. / p) if p != 1 else float("inf") # dual norm
        self.eps = eps 
        self.perturbed_words = 1     
        self.lw = lw 
        self.uw = uw
        self.lb = lb 
        self.ub = ub 
        self.batch_size = self.lw.shape[0]
        self.length = self.lw.shape[1]
        self.dim_in = self.lw.shape[2]
        self.dim_out = self.lw.shape[3]

    def concretize_l(self, lw=None):
        if lw is None: lw = self.lw
        # print("concretiez_l, lw: ", lw[0,0,:,0])
        # print("self.q: ", self.q)
        # print("torch.norm(lw, p=self.q, dim=-2): ", torch.norm(lw, p=self.q, dim=-2))

        return -self.eps * torch.norm(lw, p=self.q, dim=-2)

    def concretize_u(self, uw=None):
        if uw is None: uw = self.uw
        
        return self.eps * torch.norm(uw, p=self.q, dim=-2)

    def concretize(self):
        dim = self.lw.shape[-2] // self.perturbed_words
        res_l = self.lb.clone()
        res_u = self.ub.clone()
        # print("self.lb: ", self.lb)
        # print("self.eps: ", self.eps)
        for i in range(self.perturbed_words):
            res_l += self.concretize_l(self.lw[:, :, (dim * i) : (dim * (i + 1)), :])
            res_u += self.concretize_u(self.uw[:, :, (dim * i) : (dim * (i + 1)), :])
            # print("i: ", i, ", l norm: ", self.concretize_l(self.lw[:, :, (dim * i) : (dim * (i + 1)), :])/self.eps)
            # print("i: ", i, ", u norm: ", self.concretize_u(self.uw[:, :, (dim * i) : (dim * (i + 1)), :])/self.eps)
            # print("uw input to norm: ", self.uw[0, 0, (dim * i) : (dim * (i + 1)), 1])
            #                                  batch_size x length x dim_in x dim_out
        # print("res_l: ", res_l)
        # print("res_u: ", res_u)

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
        # print("uk.shape: ", k.shape, ", uk: ", k)
        # print("l: ", l)
        # print("u: ", u)
        # print("l: ", l)
        # print("epsilon: ", epsilon)
        self.add_linear(
            mask=mask_both, w_out=uw, b_out=ub, type="upper",
            k=k, x0=l, y0=0
        )

        k = torch.gt(torch.abs(u), torch.abs(l)).to(torch.float)

        self.add_linear(
            mask=mask_both, w_out=lw, b_out=lb, type="lower",
            k=k, x0=0, y0=0
        )
    
        return Bounds(self.p, self.eps,lw = lw, lb = lb,uw = uw, ub = ub)

    def get_bounds_xy(self, l_x, u_x, l_y, u_y, debug=False):
        alpha_l = l_y
        beta_l = l_x
        gamma_l = -alpha_l * beta_l        

        alpha_u = u_y
        beta_u = l_x
        gamma_u = -alpha_u * beta_u 

        # print("alpha_l: ", alpha_l)
        # print("beta_l: ", beta_l)
        # print("alpha_u: ", alpha_u)
        # print("gamma_l: ", gamma_l)
        # print("gamma_u: ", gamma_u)

        return alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u

    def dot_product(self, other, debug=False, verbose=False, lower=True, upper=True):
        # This if branch is included in original PyTorch implementation for Transformer Verification.
        # if self.dim_in == 1:
        #     l1, u1 = self.lb.unsqueeze(-2), self.ub.unsqueeze(-2)
        #     l2, u2 = other.lb.unsqueeze(1), other.ub.unsqueeze(1)
        #     prod1, prod2, prod3, prod4 = l1 * l2, l1 * u2, u1 * l2, u1 * u2
        #     l = torch.min(torch.min(prod1, prod2), torch.min(prod3, prod4)).sum(-1)
        #     u = torch.max(torch.max(prod1, prod2), torch.max(prod3, prod4)).sum(-1)
        #     w = l.unsqueeze(-2) * 0
        #     return Bounds(
        #         self.args,  self.p, self.eps,
        #         lw = w, lb = l, uw = w, ub = u
        #     )

        l_a, u_a = self.concretize()
        l_b, u_b = other.concretize()

        # print("l_a: ", l_a)
        # print("u_a: ", u_a)
        # print("l_b: ", l_b)
        # print("u_b: ", u_b)

        # print("l_a:",l_a.shape,"u_a",u_a.shape)
        # print("l_b:",l_b.shape,"u_b",u_b.shape)
        # print("lw",self.lw.shape,other.lw.shape)
        # print("lb",self.lb.shape,other.lb.shape)

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
                # print("self.lb: ", self.lb)
                # print("alpha_l: ", alpha_l)

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
            self.p, self.eps,
            lw = lw, lb = lb, uw = uw, ub = ub
        )

    def matmul(self, W):
        if type(W) == Bounds:
            raise NotImplementedError
        elif len(W.shape) == 2: 
            W = W.t()

            pos_mask = torch.gt(W, 0).to(torch.float32)
            W_pos = W * pos_mask
            W_neg = W - W_pos

            # print("pytorch. self.lb.shape: ", self.lb.shape)
            # print("pytorch. w.shape: ", W.shape)
            # print("pytorch. W_pos: ", W_pos)
            # print("pytorch. W_neg: ", W_neg)
            # print("pytorch. y_lb1: ", self.lb.matmul(W_pos))
            # print("pytorch. y_lb2: ", self.ub.matmul(W_neg))

            # print("lw",self.lw.shape)
            # print("uw",self.uw.shape)

            # print("W_pos",W_pos.shape)
            # print("W_neg",W_neg.shape)

            # a = self.lw.matmul(W_pos)
            # b = self.uw.matmul(W_neg)
            # print("a:",a.shape)
            # print("b:",b.shape)
            # print("self.lb.shape: ", self.lb.shape)
            # print("self.lw.shape: ", self.lw.shape)
            # print("W_pos.shape: ", W_pos.shape)
            # print("W_neg.shape: ", W_neg.shape)
            
            # print("self.lb: ", self.lb)
            # print("W[:,0]: ", W[:,0])

            # print("self.lw.shape: ", self.lw.shape, ", self.lw[0,0,1,:]: ", self.lw[0,0,1,:])
            # print("W_pos.shape: ", W_pos.shape, ", W_pos[:,0]: ", W_pos[:,0])

            return Bounds(
                self.p, self.eps,
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

    def cuda_relu(self):
        _, _, _, _, _, _lw, _lb, _uw, _ub = self.new()
        _lw = _lw.transpose(2,3).contiguous()
        _uw = _uw.transpose(2,3).contiguous()

        _, length, dim_in, dim_out = self.lw.shape

        # print("batch_size: %d, length: %d, dim_in: %d, dim_out: %d\n" % (_, length, dim_in, dim_out))

        transposed_lw = self.lw.transpose(2,3).contiguous()
        transposed_uw = self.uw.transpose(2,3).contiguous()

        c_bound.c_relu_verification(
            self.lb, 
            self.ub, 
            transposed_lw, 
            transposed_uw,
            _lb,
            _ub,
            _lw,
            _uw,
            length,
            dim_in, 
            dim_out,
            self.eps
        )

        _lw = _lw.transpose(2,3)
        _uw = _uw.transpose(2,3)

        return Bounds(self.p, self.eps,lw = _lw, lb = _lb,uw = _uw, ub = _ub)  


    def tanh(self):
        l, u, mask_pos, mask_neg, mask_both, lw, lb, uw, ub = self.new()

        def dtanh(x):
            return 1. / torch.cosh(x).pow(2)
            
        # lower bound for negative
        m = (l + u) / 2
        k = dtanh(m)
        self.add_linear(
            mask=mask_neg, w_out=lw, b_out=lb, type="lower",
            k=k, x0=m, y0=torch.tanh(m)
        )
        # upper bound for positive
        self.add_linear(
            mask=mask_pos, w_out=uw, b_out=ub, type="upper",
            k=k, x0=m, y0=torch.tanh(m)
        )

        # upper bound for negative
        k = (torch.tanh(u) - torch.tanh(l)) / (u - l + epsilon)
        self.add_linear(
            mask=mask_neg, w_out=uw, b_out=ub, type="upper",
            k=k, x0=l, y0=torch.tanh(l)
        )
        # lower bound for positive
        self.add_linear(
            mask=mask_pos, w_out=lw, b_out=lb, type="lower",
            k=k, x0=l, y0=torch.tanh(l)
        )

        # bounds for both
        max_iter = 10

        # # lower bound for both
        # diff = lambda d: (torch.tanh(u) - torch.tanh(d)) / (u - d + epsilon) - dtanh(d)
        # d = l / 2
        # _l = l
        # _u = torch.zeros(l.shape).to(self.device)
        # for t in range(max_iter):
        #     v = diff(d)
        #     mask_p = torch.gt(v, 0).to(torch.float)
        #     _l = d * mask_p + _l * (1 - mask_p)
        #     _u = d * (1 - mask_p) + _u * mask_p
        #     d = (d + _u) / 2 * mask_p + (d + _l) / 2 * (1 - mask_p)
        # k = (torch.tanh(d) - torch.tanh(u)) / (d - u + epsilon)
        # self.add_linear(
        #     mask=mask_both, w_out=lw, b_out=lb, type="lower",
        #     k=k, x0=d, y0=torch.tanh(d)
        # )

        # upper bound for both
        diff = lambda d: (torch.tanh(d) - torch.tanh(l))/ (d - l + epsilon) - dtanh(d)
        d = u / 2
        _l = torch.zeros(l.shape).to(self.device)
        _u = u
        for t in range(max_iter):
            v = diff(d)
            mask_p = torch.gt(v, 0).to(torch.float)
            _l = d * (1 - mask_p) + _l * mask_p
            _u = d * mask_p + _u * (1 - mask_p)
            d = (d + _u) / 2 * (1 - mask_p) + (d + _l) / 2 * mask_p
        k = (torch.tanh(d) - torch.tanh(l)) / (d - l + epsilon)

        # print("pytorch. d[0][0][0]: ", d[0][0][0])
        # print("pytorch. k[0][0][0]: ", k[0][0][0], ", torch.tanh(d)[0][0][0]: ", torch.tanh(d)[0][0][0], ", torch.tanh(l)[0][0][0]: ", torch.tanh(l)[0][0][0], ", l[0][0][0]: ", l[0][0][0], ", epsilon: %f", epsilon)
        # print("pytorch. (torch.tanh(d) - torch.tanh(l))[0][0][0]: ", (torch.tanh(d) - torch.tanh(l))[0][0][0], ", (d - l + epsilon)[0][0][0]: ", (d - l + epsilon)[0][0][0], ", k[0][0][0]: ", k[0][0][0])

        self.add_linear(
            mask=mask_both, w_out=uw, b_out=ub, type="upper",
            k=k, x0=d, y0=torch.tanh(d)
        )        

        return Bounds(
            self.p, self.eps,
            lw = lw, lb = lb,
            uw = uw, ub = ub
        )

    def cuda_tanh(self):
        _, _, _, _, _, _lw, _lb, _uw, _ub = self.new()
        _lw = _lw.transpose(2,3).contiguous()
        _uw = _uw.transpose(2,3).contiguous()

        _, length, dim_in, dim_out = self.lw.shape

        # print("batch_size: %d, length: %d, dim_in: %d, dim_out: %d\n" % (_, length, dim_in, dim_out))

        transposed_lw = self.lw.transpose(2,3).contiguous()
        transposed_uw = self.uw.transpose(2,3).contiguous()

        c_bound.c_tanh_verification(
            self.lb, 
            self.ub, 
            transposed_lw, 
            transposed_uw,
            _lb,
            _ub,
            _lw,
            _uw,
            length,
            dim_in, 
            dim_out,
            self.eps
        )

        _lw = _lw.transpose(2,3)
        _uw = _uw.transpose(2,3)

        return Bounds(self.p, self.eps,lw = _lw, lb = _lb,uw = _uw, ub = _ub)  

    def cuda_dot_product_QK(self, other):
        batch_size, length, dim_in, dim_out = self.lw.shape

        lw = torch.zeros(self.lw.shape[0], self.lw.shape[1], other.lw.shape[1], self.dim_in).to(self.device)
        uw = torch.zeros(self.lw.shape[0], self.lw.shape[1], other.lw.shape[1], self.dim_in).to(self.device)
        lb = torch.zeros(self.lw.shape[0], self.lw.shape[1], other.lw.shape[1]).to(self.device)
        ub = torch.zeros(self.lw.shape[0], self.lw.shape[1], other.lw.shape[1]).to(self.device)
        x_l = torch.zeros(self.lb.shape).to(self.device)
        x_u = torch.zeros(self.lb.shape).to(self.device)
        y_l = torch.zeros(self.lb.shape).to(self.device)
        y_u = torch.zeros(self.lb.shape).to(self.device)

        transposed_lw = self.lw.transpose(2,3).contiguous()
        transposed_uw = self.uw.transpose(2,3).contiguous()
        transposed_other_lw = other.lw.transpose(2,3).contiguous()
        transposed_other_uw = other.uw.transpose(2,3).contiguous()
        
        c_bound.c_dot_product_verification_QK(
            x_l, y_l, x_u, y_u,
            self.lb, self.ub, other.lb, other.ub,  
            transposed_lw, transposed_uw, transposed_other_lw, transposed_other_uw,
            lb, ub, lw, uw,
            batch_size, length, dim_out, dim_in, self.eps
        )

        lw = lw.transpose(2,3)
        uw = uw.transpose(2,3)

        return Bounds(
            self.p, self.eps,
            lw = lw, lb = lb, uw = uw, ub = ub
        )

    def cuda_dot_product_V(self, other):
        batch_size, dim_out, dim_in, length = other.lw.shape

        lw = torch.zeros(batch_size, length, dim_out, dim_in).to(self.device)
        uw = torch.zeros(batch_size, length, dim_out, dim_in).to(self.device)
        lb = torch.zeros(batch_size, length, dim_out).to(self.device)
        ub = torch.zeros(batch_size, length, dim_out).to(self.device)
        x_l = torch.zeros(self.lb.shape).to(self.device)
        x_u = torch.zeros(self.lb.shape).to(self.device)
        y_l = torch.zeros(other.lb.shape).to(self.device)
        y_u = torch.zeros(other.lb.shape).to(self.device)

        transposed_lw = self.lw.transpose(2,3).contiguous()
        transposed_uw = self.uw.transpose(2,3).contiguous()
        transposed_other_lw = other.lw.transpose(2,3).contiguous()
        transposed_other_uw = other.uw.transpose(2,3).contiguous()
        
        c_bound.c_dot_product_verification_V(
            x_l, y_l, x_u, y_u,
            self.lb, self.ub, other.lb, other.ub,  
            transposed_lw, transposed_uw, transposed_other_lw, transposed_other_uw,
            lb, ub, lw, uw,
            batch_size, length, dim_out, dim_in, self.eps
        )

        lw = lw.transpose(2,3)
        uw = uw.transpose(2,3)

        return Bounds(
            self.p, self.eps,
            lw = lw, lb = lb, uw = uw, ub = ub
        )
        
    # def cuda_matmul(self, W):
    #     batch_size, length, dim_in, dim_out = self.lw.shape
    #     # W = W.t().contiguous()
    #     # print("cuda_matmul. W.shape: ", W.shape)
    #     dim_y_out, dim_out = W.shape

    #     lw = torch.zeros(batch_size, length, dim_in, dim_y_out).to(self.device)
    #     lb = torch.zeros(batch_size, length, dim_y_out).to(self.device)
    #     uw = torch.zeros(batch_size, length, dim_in, dim_y_out).to(self.device)
    #     ub = torch.zeros(batch_size, length, dim_y_out).to(self.device)

    #     # transposed_lw = self.lw.transpose(2,3).contiguous()
    #     # transposed_uw = self.uw.transpose(2,3).contiguous()
    #     transposed_lw = self.lw
    #     transposed_uw = self.uw
        
    #     c_bound.c_matmul_verification(
    #         self.lb, self.ub, transposed_lw, transposed_uw,
    #         W, lb, ub, lw, uw,
    #         batch_size, length, dim_in, dim_out, dim_y_out
    #     )
        
    #     return Bounds(
    #         self.p, self.eps,
    #         lw = lw, lb = lb, uw = uw, ub = ub
    #     )

def relative_error_threshold(tensor1, tensor2, epsilon):
    # return torch.all(torch.abs(((tensor1 - tensor2)/tensor2))<epsilon)
    return torch.all(torch.logical_or(((tensor1 - tensor2)/tensor2)<epsilon, (tensor1 - tensor2)<epsilon))

def relu_test(bound):
    pytorch_result = bound.relu()
    cuda_result = bound.cuda_relu()
    assert relative_error_threshold(pytorch_result.lb, cuda_result.lb, 0.000001)
    assert relative_error_threshold(pytorch_result.ub, cuda_result.ub, 0.000001)
    assert relative_error_threshold(pytorch_result.lw, cuda_result.lw, 0.000001)
    assert relative_error_threshold(pytorch_result.uw, cuda_result.uw, 0.000001)

def tanh_test(bound):
    pytorch_result = bound.tanh()
    cuda_result = bound.cuda_tanh()

    FLAG_INSPECT_DIFFERENCE = False
    # Note that the difference come from 
    if FLAG_INSPECT_DIFFERENCE:
        # print("pytorch_result.lw-cuda_result.lw: ", pytorch_result.lw-cuda_result.lw)
        # print((pytorch_result.lw-cuda_result.lw)[(pytorch_result.lw-cuda_result.lw)>0.0001])
        print("pytorch_result.ub.shape: ", pytorch_result.ub.shape, "pytorch_result.ub: ", pytorch_result.ub)
        print("cuda_result.ub.shape: ", cuda_result.ub.shape, "cuda_result.ub: ", cuda_result.ub)

        # print("(pytorch_result.lw-cuda_result.lw)/pytorch_result.lw: ", (pytorch_result.lw-cuda_result.lw)/pytorch_result.lw)
        # print("((pytorch_result.lw-cuda_result.lw)/pytorch_result.lw) < 0.0001: ", ((pytorch_result.lw-cuda_result.lw)/pytorch_result.lw)<0.0001)
        # print(((pytorch_result.lw-cuda_result.lw)/pytorch_result.lw)[((pytorch_result.lw-cuda_result.lw)/pytorch_result.lw)>0.0001])
        # print((pytorch_result.lw)[torch.abs(((pytorch_result.lw-cuda_result.lw)/cuda_result.lw))>0.0001])
        # print((cuda_result.lw)[torch.abs(((pytorch_result.lw-cuda_result.lw)/cuda_result.lw))>0.0001])


    assert relative_error_threshold(pytorch_result.lb, cuda_result.lb, 0.000001)
    assert relative_error_threshold(pytorch_result.ub, cuda_result.ub, 0.000001)
    assert relative_error_threshold(pytorch_result.lw, cuda_result.lw, 0.000001)
    assert relative_error_threshold(pytorch_result.uw, cuda_result.uw, 0.000001)

def dot_product_QK_test(bound,other_bound):
    pytorch_result = bound.dot_product(other_bound)
    cuda_result = bound.cuda_dot_product_QK(other_bound)

    FLAG_INSPECT_DIFFERENCE = False
    # Note that the difference come from 
    if FLAG_INSPECT_DIFFERENCE:
        print("pytorch_result.lw-cuda_result.lw: ", pytorch_result.lw-cuda_result.lw)
        print((pytorch_result.lw-cuda_result.lw)[(pytorch_result.lw-cuda_result.lw)>0.0001])
        print("pytorch_result.lw.shape: ", pytorch_result.lw.shape, "pytorch_result.lw: ", pytorch_result.lw)
        print("cuda_result.lw.shape: ", cuda_result.lw.shape, "cuda_result.lw: ", cuda_result.lw)

        print("(pytorch_result.lw-cuda_result.lw)/pytorch_result.lw: ", (pytorch_result.lw-cuda_result.lw)/pytorch_result.lw)
        print("((pytorch_result.lw-cuda_result.lw)/pytorch_result.lw) < 0.0001: ", ((pytorch_result.lw-cuda_result.lw)/pytorch_result.lw)<0.0001)
        print(((pytorch_result.lw-cuda_result.lw)/pytorch_result.lw)[((pytorch_result.lw-cuda_result.lw)/pytorch_result.lw)>0.0001])
        print((pytorch_result.lw)[torch.abs(((pytorch_result.lw-cuda_result.lw)/cuda_result.lw))>0.0001])
        print((cuda_result.lw)[torch.abs(((pytorch_result.lw-cuda_result.lw)/cuda_result.lw))>0.0001])

    # print("(pytorch_result.lw-cuda_result.lw)/cuda_result.lw: ", ((pytorch_result.lw-cuda_result.lw)/cuda_result.lw)<0.000001)

    # print("pytorch_result.uw: ", pytorch_result.uw)
    # print("cuda_result.uw: ", cuda_result.uw)

    assert relative_error_threshold(pytorch_result.lb, cuda_result.lb, 0.001)
    assert relative_error_threshold(pytorch_result.ub, cuda_result.ub, 0.001)
    assert relative_error_threshold(pytorch_result.lw, cuda_result.lw, 0.001)
    assert relative_error_threshold(pytorch_result.uw, cuda_result.uw, 0.001)

def dot_product_V_test(bound,other_bound):
    pytorch_result = bound.dot_product(other_bound)
    cuda_result = bound.cuda_dot_product_V(other_bound)

    FLAG_INSPECT_DIFFERENCE = False
    # Note that the difference come from 
    if FLAG_INSPECT_DIFFERENCE:
        print("pytorch_result.lw-cuda_result.lw: ", pytorch_result.lw-cuda_result.lw)
        print((pytorch_result.lw-cuda_result.lw)[(pytorch_result.lw-cuda_result.lw)>0.0001])
        print("pytorch_result.lw.shape: ", pytorch_result.lw.shape, "pytorch_result.lw: ", pytorch_result.lw)
        print("cuda_result.lw.shape: ", cuda_result.lw.shape, "cuda_result.lw: ", cuda_result.lw)

        print("(pytorch_result.lw-cuda_result.lw)/pytorch_result.lw: ", (pytorch_result.lw-cuda_result.lw)/pytorch_result.lw)
        print("((pytorch_result.lw-cuda_result.lw)/pytorch_result.lw) < 0.0001: ", ((pytorch_result.lw-cuda_result.lw)/pytorch_result.lw)<0.0001)
        print(((pytorch_result.lw-cuda_result.lw)/pytorch_result.lw)[((pytorch_result.lw-cuda_result.lw)/pytorch_result.lw)>0.0001])
        print((pytorch_result.lw)[torch.abs(((pytorch_result.lw-cuda_result.lw)/cuda_result.lw))>0.0001])
        print((cuda_result.lw)[torch.abs(((pytorch_result.lw-cuda_result.lw)/cuda_result.lw))>0.0001])

    # print("(pytorch_result.lw-cuda_result.lw)/cuda_result.lw: ", ((pytorch_result.lw-cuda_result.lw)/cuda_result.lw)<0.000001)

    # print("pytorch_result.uw: ", pytorch_result.uw)
    # print("cuda_result.uw: ", cuda_result.uw)

    assert relative_error_threshold(pytorch_result.lb, cuda_result.lb, 0.00001)
    assert relative_error_threshold(pytorch_result.ub, cuda_result.ub, 0.00001)
    assert relative_error_threshold(pytorch_result.lw, cuda_result.lw, 0.001)
    assert relative_error_threshold(pytorch_result.uw, cuda_result.uw, 0.001)

def matmul_test(W, bound):
    pytorch_res = bound.matmul(W)
    cuda_res = bound.cuda_matmul(W)

    FLAG_INSPECT_DIFFERENCE = False
    # Note that the difference come from 
    if FLAG_INSPECT_DIFFERENCE:
        print("pytorch_res.lw]: ", pytorch_res.lw, ", pytorch_res.lw[0][0][10][0]: ", pytorch_res.lw[0][0][10][0])
        print("cuda_res.lw: ", cuda_res.lw, ", cuda_res.lw[0][0][10][0]: ", cuda_res.lw[0][0][10][0]) 
        print("pytorch_res.lw-cuda_res.lw: ", pytorch_res.lw-cuda_res.lw)
        diff = pytorch_res.lw-cuda_res.lw
        print((diff>0.001).nonzero(as_tuple=False))
        print(diff[diff>0.001])
        # print((pytorch_res.lw-cuda_res.lw)[(pytorch_res.lw-cuda_res.lw)>0.0001])
        # print("pytorch_res.lw.shape: ", pytorch_res.lw.shape, "pytorch_res.lw: ", pytorch_res.lw)
        # print("cuda_res.lw.shape: ", cuda_res.lw.shape, "cuda_res.lw: ", cuda_res.lw)


    assert relative_error_threshold(pytorch_res.lb, cuda_res.lb, 0.0001)
    assert relative_error_threshold(pytorch_res.ub, cuda_res.ub, 0.0001)
    assert relative_error_threshold(pytorch_res.lw, cuda_res.lw, 0.0001)
    assert relative_error_threshold(pytorch_res.uw, cuda_res.uw, 0.0001)

def print_init_value(lb, ub, lw, uw, length, dim_in, dim_out):
    for i in range(length):
        for j in range(dim_out):
            print("lb[%d*dim_out+%d] = %ff;" % (i, j, lb[0][i][j]))
            print("ub[%d*dim_out+%d] = %ff;" % (i, j, ub[0][i][j]))

    for i in range(length):
        for j in range(dim_in):
            for k in range(dim_out):
                print("lw[%d*dim_in*dim_out+%d*dim_in+%d] = %ff;" % (i, k, j, lw[0][i][j][k]))
                print("uw[%d*dim_in*dim_out+%d*dim_in+%d] = %ff;" % (i, k, j, uw[0][i][j][k]))

if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # length = 2
    # dim_in = 64
    # dim_out = 64
    # dim_y_out = 64 # dim_y_out must equal to dim_out

    print("Checking correctness of " + FLAG + "...")
    if FLAG == "mat_mul" or FLAG == "relu" or FLAG == "dot_product_QK" or FLAG == "tanh":
        if FLAG == "dot_product_QK":
            batch_target = [1,2,4]
        else:
            batch_target = [1]
        for batch_size in batch_target: # For dot product, support any batch_size. For relu/matmul/tanh, support only batch_size=1
            for length in [2,4,8,16,32,64,128]:
                for dim_in in [64,128,256,512]:
            # for length in [2]:
            #     for dim_in in [64]:
                    if FLAG == "dot_product_QK":
                        target = [16,64,128,256,512]
                    else:
                        target = [64,128,256,512]
                    for dim_out in target:
                    # for dim_out in [64]:
                        dim_y_out = dim_in
                        # dim_y_out = 32

                        print("batch_size, %d, length:%d, dim_in: %d, dim_out: %d" % (batch_size, length, dim_in, dim_out))

                        RAND_INIT_FLAG = True
                        if RAND_INIT_FLAG:
                            lb = torch.rand(batch_size,length,dim_out).to(device)
                            ub = lb + torch.rand(batch_size,length,dim_out).to(device)
                            lw = torch.rand(batch_size,length,dim_in,dim_out).to(device) - 0.5
                            uw = torch.rand(batch_size,length,dim_in,dim_out).to(device) - 0.5

                            lb1 = torch.rand(batch_size,length,dim_out).to(device)
                            ub1 = lb1 + torch.rand(batch_size,length,dim_out).to(device)
                            lw1 = torch.rand(batch_size,length,dim_in,dim_out).to(device) - 0.5
                            uw1 = torch.rand(batch_size,length,dim_in,dim_out).to(device) - 0.5
                        else:
                            lb = torch.ones(batch_size,length,dim_out).to(device)
                            ub = lb + torch.ones(batch_size,length,dim_out).to(device)
                            lw = torch.ones(batch_size,length,dim_in,dim_out).to(device) - 0.5
                            uw = torch.ones(batch_size,length,dim_in,dim_out).to(device) - 0.5

                            lb1 = torch.ones(batch_size,length,dim_out).to(device)
                            ub1 = lb1 + torch.ones(batch_size,length,dim_out).to(device)
                            lw1 = torch.ones(batch_size,length,dim_in,dim_out).to(device) - 0.5
                            uw1 = torch.ones(batch_size,length,dim_in,dim_out).to(device) - 0.5

                        # print_init_value(lb, ub, lw, uw, length, dim_in, dim_out)
                        
                        # print("lb: ", lb)
                        # print('ub: ', ub)
                        # print('input_lw[0,0,:,1]: ', lw[0,0,:,1])
                        # print('input_uw[0,0,:,1]: ', uw[0,0,:,1])
                        # print('input_uw[0,0,1,:]: ', uw[0,0,1,:])

                        bound = Bounds(p=2,eps=0.5,lw=lw,lb=lb,uw=uw,ub=ub)

                        if FLAG == "mat_mul":
                            if RAND_INIT_FLAG:
                                W = torch.rand(dim_y_out, dim_out).to(device) - 0.5
                            else:
                                W = torch.ones(dim_y_out, dim_out).to(device) - 0.5
                            matmul_test(W, bound)
                        elif FLAG == "dot_product_QK":
                            # Note that the difference between pytorch and hand tuned kernel may increase a bit as the tensor size increase.
                            # This is expected behavior due to floating point computation.
                            bound1 = Bounds(p=2,eps=0.5,lw=lw1,lb=lb1,uw=uw1,ub=ub1)
                            dot_product_QK_test(bound, bound1)
                        elif FLAG == "relu":
                            relu_test(bound)
                        elif FLAG == "tanh":
                            tanh_test(bound);
                        else:
                            raise NotImplementedError
    elif FLAG == "dot_product_V":
        # Note that the difference between pytorch and hand tuned kernel may increase a bit as the tensor size increase.
        # This is expected behavior due to floating point computation.
        for batch_size in [1,2,4]: # For dot product, support any batch_size. For relu/matmul, support only batch_size=1
            for length in [2,4,8,16,32,64,128]:
                for dim_in in [64,128,256,512]:
                    for dim_out in [10,20,64,128,256,512]:

                        print("batch_size, %d, length:%d, dim_in: %d, dim_out: %d" % (batch_size, length, dim_in, dim_out))

                        RAND_INIT_FLAG = True
                        if RAND_INIT_FLAG:
                            lb = torch.rand(batch_size,length,length).to(device)
                            ub = lb + torch.rand(batch_size,length,length).to(device)
                            lw = torch.rand(batch_size,length,dim_in,length).to(device) - 0.5
                            uw = torch.rand(batch_size,length,dim_in,length).to(device) - 0.5

                            lb1 = torch.rand(batch_size,dim_out,length).to(device)
                            ub1 = lb1 + torch.rand(batch_size,dim_out,length).to(device)
                            lw1 = torch.rand(batch_size,dim_out, dim_in, length).to(device) - 0.5
                            uw1 = torch.rand(batch_size,dim_out, dim_in, length).to(device) - 0.5
                        else:
                            lb = torch.ones(batch_size,length,length).to(device)
                            ub = lb + torch.ones(batch_size,length,length).to(device)
                            lw = torch.ones(batch_size,length,dim_in,length).to(device) - 0.5
                            uw = torch.ones(batch_size,length,dim_in,length).to(device) - 0.5

                            lb1 = torch.ones(batch_size,dim_out,length).to(device)
                            ub1 = lb1 + torch.ones(batch_size,dim_out,length).to(device)
                            lw1 = torch.ones(batch_size,dim_out, dim_in, length).to(device) - 0.5
                            uw1 = torch.ones(batch_size,dim_out, dim_in, length).to(device) - 0.5

                        # print_init_value(lb, ub, lw, uw, length, dim_in, dim_out)
                        
                        # print("lb: ", lb)
                        # print('ub: ', ub)
                        # print('input_lw[0,0,:,1]: ', lw[0,0,:,1])
                        # print('input_uw[0,0,:,1]: ', uw[0,0,:,1])
                        # print('input_uw[0,0,1,:]: ', uw[0,0,1,:])

                        bound = Bounds(p=2,eps=0.5,lw=lw,lb=lb,uw=uw,ub=ub)

                        bound1 = Bounds(p=2,eps=0.5,lw=lw1,lb=lb1,uw=uw1,ub=ub1)
                        dot_product_V_test(bound, bound1)


