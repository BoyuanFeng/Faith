# Copyright (c) 2020, Zhouxing shi <zhouxingshichn@gmail.com>
# Licenced under the BSD 2-Clause License.

import math, os, time
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import tvm
from tvm import relay
from tvm.contrib import graph_executor

base_dirc = "/home/boyuan/verification_tianqi/HandTunedKernels/"
# base_dirc = "../"
c_bound = load(name="c_relu_verification", sources=[
    base_dirc+"kernel.cpp", 
    base_dirc+"relu_verification.cu", 
    base_dirc+"dot_product_verification.cu",
    # base_dirc+"matmul_verification.cu",
    base_dirc+"tanh_verification.cu"
    ], 
    # extra_include_paths = [
    #     "../cutlass/include", 
    #     "../cutlass/tools/util/include",
    #     "../cutlass/examples/common",
    #     ],
    verbose=False)
epsilon = 1e-12

def relative_error_threshold(tensor1, tensor2, epsilon):
    return torch.all((tensor1 - tensor2)<epsilon)

# FLAG = 1 # 0 for original version, 1 for cuda version, 2 for tvm version, 3 for ansor version

def Bounds2Tuple(x):
    return tuple((
            torch.Tensor([[x.p]]),
            torch.Tensor([[x.eps]]), 
            x.lw, x.lb, x.uw, x.ub
        ))

def Bounds2TupleWithWeight(x, W):
    return tuple((
            torch.Tensor([[x.p]]),
            torch.Tensor([[x.eps]]), 
            x.lw, x.lb, x.uw, x.ub,
            W
        ))
# def Tuple2Bounds(x):
#     return Bounds(x[0], x[1], x[2], x[3], x[4], x[5])
# def Elements2BoundsDotProduct(x1, x2, x3, x4):
#     return BoundsDotProduct(args, p=2, eps=0.1, w=None, b=None, lw=x1, lb=x2, uw=x3, ub=x4)

class BoundsReLUWrapper(nn.Module):
    def __init__(self): #, ):
        super(BoundsReLUWrapper, self).__init__()
        
    def forward(self, args, p, eps, lw, lb, uw, ub):
        # def __init__(self, args, p, eps, w=None, b=None, lw=None, lb=None, uw=None, ub=None, clone=True):
        x = BoundsWithoutArgs(float(p), float(eps), lw=lw, lb=lb, uw=uw, ub=ub).ori_relu()
        return Bounds2Tuple(x)

class BoundsMatMulWrapper(nn.Module):
    def __init__(self): 
        super(BoundsMatMulWrapper, self).__init__()
        # self.W = W

    def forward(self, p, eps, lw, lb, uw, ub, W):
        x = BoundsWithoutArgs(float(p), float(eps), lw=lw, lb=lb, uw=uw, ub=ub).ori_matmul(W)
        return Bounds2Tuple(x) 

# class BoundsDotProductWrapper(nn.Module):
#     def __init__(self):
#         super(BoundsDotProductWrapper, self).__init__()

#     def forward(self, args0, p0, eps0, w0, b0, lw0, lb0, uw0, ub0, args1, p1, eps1, w1, b1, lw1, lb1, uw1, ub1):
#         x = Bounds(args0, float(p0), float(eps0), w0, b0, lw0, lb0, uw0, ub0)
#         y = Bounds(args1, float(p1), float(eps1), w1, b1, lw1, lb1, uw1, ub1)
#         return Bounds2Tuple(x.ori_dot_product(y))


class BoundsWithoutArgs:
    def __init__(self, p, eps, w=None, b=None, lw=None, lb=None, uw=None, ub=None, clone=True, ibp=False, perturbed_words=-1):
        self.ibp = ibp
        self.device = lw.device if lw is not None else w.device
        self.p = p
        self.q = 1. / (1. - 1. / p) if p != 1 else float("inf") # dual norm
        self.eps = eps 
        self.perturbed_words = perturbed_words        
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

    def clone(self):
        return BoundsWithoutArgs(
            self.p, self.eps,
            lw = self.lw.clone(), lb = self.lb.clone(),
            uw = self.uw.clone(), ub = self.ub.clone(),
            ibp= self.ibp, perturbed_words = self.perturbed_words 
        )

    def t(self):
        return BoundsWithoutArgs(
            self.p, self.eps,
            lw = self.lw.transpose(1, 3),
            uw = self.uw.transpose(1, 3),
            lb = self.lb.transpose(1, 2),
            ub = self.ub.transpose(1, 2),
            ibp= self.ibp, perturbed_words = self.perturbed_words 
        )   

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

    def add(self, delta):
        if type(delta) == BoundsWithoutArgs:
            return BoundsWithoutArgs(
                self.p, self.eps,
                lw = self.lw + delta.lw, lb = self.lb + delta.lb,
                uw = self.uw + delta.uw, ub = self.ub + delta.ub,
                ibp = self.ibp, perturbed_words = self.perturbed_words
            )
        else:
            return BoundsWithoutArgs(
                self.p, self.eps,
                lw = self.lw, lb = self.lb + delta, 
                uw = self.uw, ub = self.ub + delta,
                ibp = self.ibp, perturbed_words = self.perturbed_words
            )

    def ori_matmul(self, W):
        if type(W) == BoundsWithoutArgs:
            raise NotImplementedError
        elif len(W.shape) == 2:
            W = W.t()

            pos_mask = torch.gt(W, 0).to(torch.float32)
            W_pos = W * pos_mask
            W_neg = W - W_pos

            return BoundsWithoutArgs(
                self.p, self.eps,
                lw = self.lw.matmul(W_pos) + self.uw.matmul(W_neg),
                lb = self.lb.matmul(W_pos) + self.ub.matmul(W_neg),
                uw = self.lw.matmul(W_neg) + self.uw.matmul(W_pos),
                ub = self.lb.matmul(W_neg) + self.ub.matmul(W_pos),
                ibp= self.ibp, perturbed_words = self.perturbed_words 
            )
        else:
            print("this branch!!!\n\n")
            W = W.transpose(1, 2)

            pos_mask = torch.gt(W, 0).to(torch.float32)
            W_pos = W * pos_mask
            W_neg = W - W_pos

            return BoundsWithoutArgs(
                self.p, self.eps,
                lw = (self.lw.squeeze(0).bmm(W_pos) + self.uw.squeeze(0).bmm(W_neg)).unsqueeze(0),
                lb = (self.lb.transpose(0, 1).bmm(W_pos) + self.ub.transpose(0, 1).bmm(W_neg)).transpose(0, 1),
                uw = (self.lw.squeeze(0).bmm(W_neg) + self.uw.squeeze(0).bmm(W_pos)).unsqueeze(0),
                ub = (self.lb.transpose(0, 1).bmm(W_neg) + self.ub.transpose(0, 1).bmm(W_pos)).transpose(0, 1),
                ibp= self.ibp, perturbed_words = self.perturbed_words 
            )
    
    def ori_relu(self):
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
        self.FLAG = args.current_flag
        if self.ibp:
            self.lw, self.uw = \
                self.lw[:, :, :self.perturbed_words, :],\
                self.uw[:, :, :self.perturbed_words, :]        
        self.update_shape()
        # self.tvm_matmul_model = None
        FLAG = args.kernel_type

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

    def clone(self):
        return Bounds(
            self.args, self.p, self.eps,
            lw = self.lw.clone(), lb = self.lb.clone(),
            uw = self.uw.clone(), ub = self.ub.clone()
        )

    def t(self):
        return Bounds(
            self.args, self.p, self.eps,
            # lw = self.lw.transpose(1, 3).contiguous(),
            # uw = self.uw.transpose(1, 3).contiguous(),
            # lb = self.lb.transpose(1, 2).contiguous(),
            # ub = self.ub.transpose(1, 2).contiguous()
            lw = self.lw.transpose(1, 3),
            uw = self.uw.transpose(1, 3),
            lb = self.lb.transpose(1, 2),
            ub = self.ub.transpose(1, 2)
        )   

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

    def add(self, delta):
        if type(delta) == Bounds:
            return Bounds(
                self.args, self.p, self.eps,
                lw = self.lw + delta.lw, lb = self.lb + delta.lb,
                uw = self.uw + delta.uw, ub = self.ub + delta.ub
            )
        else:
            return Bounds(
                self.args, self.p, self.eps,
                lw = self.lw, lb = self.lb + delta, 
                uw = self.uw, ub = self.ub + delta
            )

    def ori_matmul(self, W):
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
            print("this branch!!!\n\n")
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

    def tvm_relu(self, tvm_model_lut=None, tvm_model_path=None):
        bound_wrapper = BoundsReLUWrapper()
        example_inputs = Bounds2Tuple(self.clone())
        scripted_model = torch.jit.trace(bound_wrapper.eval(), example_inputs).eval()
        input_name = "input%d"
        shape_list = []
        if type(example_inputs) == tuple:
            for i in range(len(example_inputs)):
                shape_list.append((input_name%(i), example_inputs[i].shape))
        else:
            shape_list.append((input_name%(0), example_inputs.shape))
        tvm_model_name = 'relu_%d_%d_%d_%d'%(self.uw.shape[0], self.uw.shape[1], self.uw.shape[2], self.uw.shape[3])
        file_name = "deploy_%s.so"%(tvm_model_name)
       
        dtype = "float32"
        target = tvm.target.Target("cuda")
        dev = tvm.cuda(0)
        if not file_name in tvm_model_lut:
            mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(mod, target=target, params=params)
                tvm_relu_model = graph_executor.GraphModule(lib["default"](dev))
                tvm_model_lut[tvm_model_name] = tvm_matmul_model
                if tvm_model_path == None:
                    lib.export_library(os.getcwd()+'/tvm_model/'+file_name)
                else:
                    lib.export_library(tvm_model_path+'/tvm_model/'+file_name)
        else:
            tvm_relu_model = tvm_model_lut[file_name]
                           
        # Set inputs
        if type(example_inputs) == tuple:
            for i in range(2, len(example_inputs)):
                tvm_relu_model.set_input(input_name%(i), tvm.nd.array(example_inputs[i].cpu().numpy().astype(dtype)))
        else:
            tvm_relu_model.set_input(input_name%(0), tvm.nd.array(example_inputs.numpy().cpu().astype(dtype)))

        # tvm_relu_model.run()
        torch.cuda.synchronize()
        output = self.clone()
        output.lw = torch.tensor(tvm_relu_model.get_output(2).numpy()).to(self.device)
        output.lb = torch.tensor(tvm_relu_model.get_output(3).numpy()).to(self.device)
        output.uw = torch.tensor(tvm_relu_model.get_output(4).numpy()).to(self.device)
        output.ub = torch.tensor(tvm_relu_model.get_output(5).numpy()).to(self.device)
        return output


    def tvm_matmul(self, W, tvm_model_lut=None, tvm_model_path=None):
        bound_wrapper = BoundsMatMulWrapper()
        example_inputs = Bounds2TupleWithWeight(self.clone(), W)
        scripted_model = torch.jit.trace(bound_wrapper.eval(), example_inputs).eval()
        input_name = "input%d"
        shape_list = []
        if type(example_inputs) == tuple:
            for i in range(len(example_inputs)):
                shape_list.append((input_name%(i), example_inputs[i].shape))
        else:
            shape_list.append((input_name%(0), example_inputs.shape))
        
        # print("W.shape=", W.shape)
        # print("lw.shape=", self.lw.shape)
        tvm_model_name = 'matmul_%d_%d_%d_%d_%d_%d'%(W.shape[0], W.shape[1], self.uw.shape[0], self.uw.shape[1], self.uw.shape[2], self.uw.shape[3])
        file_name = "deploy_%s.so"%(tvm_model_name)
        # print("is matmul in tvm_model_lut:", tvm_model_name in tvm_model_lut)
       
        dtype = "float32"
        target = tvm.target.Target("cuda")
        dev = tvm.cuda(0)
        if not file_name in tvm_model_lut:
            mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
            # print("mod", mod)
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(mod, target=target, params=params)
                tvm_matmul_model = graph_executor.GraphModule(lib["default"](dev))
                tvm_model_lut[tvm_model_name] = tvm_matmul_model
                if tvm_model_path == None:
                    lib.export_library(os.getcwd()+'/tvm_model/'+file_name)
                else:
                    lib.export_library(tvm_model_path+'/tvm_model/'+file_name)
        else:
            tvm_matmul_model = tvm_model_lut[file_name]
                           
        # Set inputs
        if type(example_inputs) == tuple:
            for i in range(2, len(example_inputs)):
                tvm_matmul_model.set_input(input_name%(i), tvm.nd.array(example_inputs[i].cpu().numpy().astype(dtype)))
        else:
            tvm_matmul_model.set_input(input_name%(0), tvm.nd.array(example_inputs.numpy().cpu().astype(dtype)))

        # tvm_matmul_model.run()
        torch.cuda.synchronize()
        output = self.clone()
        output.lw = torch.tensor(tvm_matmul_model.get_output(2).numpy()).to(self.device)
        output.lb = torch.tensor(tvm_matmul_model.get_output(3).numpy()).to(self.device)
        output.uw = torch.tensor(tvm_matmul_model.get_output(4).numpy()).to(self.device)
        output.ub = torch.tensor(tvm_matmul_model.get_output(5).numpy()).to(self.device)
        return output

    def ansor_matmul(self, W, ansor_model_lut=None, ansor_model_path=None):
        bound_wrapper = BoundsMatMulWrapper()
        example_inputs = Bounds2TupleWithWeight(self.clone(), W)
        input_name = "input%d"
        shape_list = []
        if type(example_inputs) == tuple:
            for i in range(len(example_inputs)):
                shape_list.append((input_name%(i), example_inputs[i].shape))
        else:
            shape_list.append((input_name%(0), example_inputs.shape))
        
        ansor_model_name = 'matmul_%d_%d_%d_%d_%d_%d'%(W.shape[0], W.shape[1], self.uw.shape[0], self.uw.shape[1], self.uw.shape[2], self.uw.shape[3])
        # print(ansor_model_name)
        # return self
        file_name = "deploy_%s.so"%(ansor_model_name)
       
        dtype = "float32"
        target = tvm.target.Target("cuda")
        dev = tvm.cuda(self.args.gpuid)

        if not file_name in ansor_model_lut:
            scripted_model = torch.jit.trace(bound_wrapper.eval(), example_inputs).eval()
            mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
            log_file = "ansor_autotuning_json/ansor_" + ansor_model_name +".json"
            tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target, include_simple_tasks=False)
            measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10)
            
            tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
            tune_option = auto_scheduler.TuningOptions(
                num_measure_trials=100 * len(tasks),  # change this to 800 & #task to achieve the best performance
                runner=measure_ctx.runner,
                measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
            )
            tuner.tune(tune_option)

            # Compile with the history best
            print("Compile ...")
            with auto_scheduler.ApplyHistoryBest(log_file):
                with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
                    lib_ansor = relay.build(mod, target=target, params=params)

            # Create graph executer
            dtype = "float32"
            module_ansor = graph_executor.GraphModule(lib_ansor["default"](dev))
            ansor_model_lut[ansor_model_name] = module_ansor
            if ansor_model_path == None:
                lib_ansor.export_library(os.getcwd()+'/ansor_model/'+file_name)
            else:
                lib_ansor.export_library(ansor_model_path+'/ansor_model/'+file_name)
        else:
            module_ansor = ansor_model_lut[file_name]
            
        if type(example_inputs) == tuple:
            for i in range(2, len(example_inputs)):
                module_ansor.set_input(input_name%(i), tvm.nd.array(example_inputs[i].cpu().numpy().astype(dtype)))
        else:
            module_ansor.set_input(input_name%(0), tvm.nd.array(example_inputs.numpy().cpu().astype(dtype)))
        # module_ansor.run()
        torch.cuda.synchronize()
        output = self.clone()
        output.lw = torch.tensor(module_ansor.get_output(2).numpy()).to(self.device)
        output.lb = torch.tensor(module_ansor.get_output(3).numpy()).to(self.device)
        output.uw = torch.tensor(module_ansor.get_output(4).numpy()).to(self.device)
        output.ub = torch.tensor(module_ansor.get_output(5).numpy()).to(self.device)
        return output

    def multiply(self, W):
        if type(W) == float:
            if W > 0:
                return Bounds(
                    self.args, self.p, self.eps,
                    lw = self.lw * W, lb = self.lb * W, 
                    uw = self.uw * W, ub = self.ub * W
                )
            else:
                return Bounds(
                    self.args, self.p, self.eps,
                    lw = self.uw * W, lb = self.ub * W, 
                    uw = self.lw * W, ub = self.lb * W
                )        
        elif type(W) == Bounds:
            assert(self.lw.shape == W.lw.shape)

            l_a, u_a = self.concretize()
            l_b, u_b = W.concretize()

            l1, u1, mask_pos_only, mask_neg_only, mask_both, lw, lb, uw, ub = self.new()

            alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u = \
                self.get_bounds_xy(
                    l_a.reshape(-1),
                    u_a.reshape(-1),
                    l_b.reshape(-1),
                    u_b.reshape(-1)
                )

            alpha_l = alpha_l.reshape(l_a.shape)
            beta_l = beta_l.reshape(l_a.shape)
            gamma_l = gamma_l.reshape(l_a.shape)
            alpha_u = alpha_u.reshape(l_a.shape)
            beta_u = beta_u.reshape(l_a.shape)
            gamma_u = gamma_u.reshape(l_a.shape)

            self.add_linear(
                mask=None, w_out=lw, b_out=lb, type="lower",
                k=alpha_l, x0=0, y0=gamma_l
            )
            self.add_linear(
                mask=None, w_out=lw, b_out=lb, type="lower",
                k=beta_l, x0=0, y0=0, src=W
            )
            self.add_linear(
                mask=None, w_out=uw, b_out=ub, type="upper",
                k=alpha_u, x0=0, y0=gamma_u
            )
            self.add_linear(
                mask=None, w_out=uw, b_out=ub, type="upper",
                k=beta_u, x0=0, y0=0, src=W
            )      

            return Bounds(
                self.args,  self.p, self.eps,
                lw = lw, lb = lb, uw = uw, ub = ub
            )

        else:
            pos_mask = torch.gt(W, 0).to(torch.float32)
            W_pos = W * pos_mask
            W_neg = W - W_pos

            return Bounds(
                self.args, self.p, self.eps,
                lw = self.lw * W_pos + self.uw * W_neg,
                lb = self.lb * W_pos + self.ub * W_neg,
                uw = self.lw * W_neg + self.uw * W_pos,
                ub = self.lb * W_neg + self.ub * W_pos
            )

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
    def ori_dot_product(self, other, debug=False, verbose=False, lower=True, upper=True):
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

        start_time = time.time()

        l_a, u_a = self.concretize()
        l_b, u_b = other.concretize()

        # print("self.lb.shape: ", self.lb.shape, ", self.lb: ", self.lb)
        # print("l_a.shape: ", l_a.shape, ", l_a: ", l_a)

        lw = torch.zeros(self.lw.shape[0], self.lw.shape[1], self.dim_in, other.lw.shape[1]).to(self.device)
        uw = torch.zeros(self.lw.shape[0], self.lw.shape[1], self.dim_in, other.lw.shape[1]).to(self.device)
        lb = torch.zeros(self.lw.shape[0], self.lw.shape[1], other.lw.shape[1]).to(self.device)
        ub = torch.zeros(self.lw.shape[0], self.lw.shape[1], other.lw.shape[1]).to(self.device)


        # print("pytorch dot product. self.lb: ", self.lb)
        # print("pytorch dot product. self.ub: ", self.ub)
        # print("pytorch dot product. self.lw: ", self.lw)
        # print("pytorch dot product. self.uw: ", self.uw)



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

    def divide(self, W):
        if type(W) == Bounds:
            W = W.reciprocal()
            return self.multiply(W)
        else:
            raise NotImplementedError


    def dot_product(self, other, debug=False, verbose=False, lower=True, upper=True):
        if self.FLAG == 1:
            _, len1, len2 = self.lb.shape
            if len1 == len2: # self.lb.shape: [batch_size, length, length]
                return self.cuda_dot_product_V(other)
            else: # self.lb.shape: [batch_size, length, dim_out]
                return self.cuda_dot_product_QK(other)
        else:
            return self.ori_dot_product(other, debug, verbose, lower, upper)

    def context(self, value):
        value = value.t()

        # Shape after value transpose:
        # self.lb.shape: [batch_size, length, length], self.lw.shape: [batch_size, length, dim_in, length]
        # value.lb.shape: [batch_size, dim_out, length], value.lw.shape: [batch_size, dim_out, dim_in, length]
        # context.lb.shape: [batch_size, length, dim_out], context.lw.shape: [batch_size, length, dim_in, dim_out]
        return self.dot_product(value)

    """
    U: (u+l) * (x-l) + l^2 = (u + l) x - u * l

    L: 2m (x - m) + m^2
    To make the lower bound never goes to negative:
        2m (l - m) + l^2 >= 0 => m (2l - m) >= 0
        2m (u - m) + u^2 >= 0 => m (2u - m) >= 0
    """
    def sqr(self):
        l, u, mask_pos, mask_neg, mask_both, lw, lb, uw, ub = self.new()

        if self.ibp:
            lb = torch.min(l * l, u * u)
            lb -= mask_both * lb # lower bound is zero for this case
            ub = torch.max(l * l, u * u)
        else:
            # upper bound
            k = u + l
            self.add_linear(
                mask=None, w_out=uw, b_out=ub, type="upper",
                k=k, x0=l, y0=l.pow(2)
            )

            # lower bound
            m = torch.max((l + u) / 2, 2 * u)
            self.add_linear(
                mask=mask_neg, w_out=lw, b_out=lb, type="lower",
                k=2*m, x0=m, y0=m.pow(2)
            )
            m = torch.min((l + u) / 2, 2 * l)
            self.add_linear(
                mask=mask_pos, w_out=lw, b_out=lb, type="lower",
                k=2*m, x0=m, y0=m.pow(2)
            )

        return Bounds(
            self.args, self.p, self.eps,
            lw = lw, lb = lb, uw = uw, ub = ub
        )

    def sqrt(self):
        l, u, mask_pos, mask_neg, mask_both, lw, lb, uw, ub = self.new()
        assert(torch.min(l) > 0)

        if self.ibp:
            lb = torch.sqrt(l)
            ub = torch.sqrt(u)
        else:
            k = (torch.sqrt(u) - torch.sqrt(l)) / (u - l + epsilon)

            self.add_linear(
                mask=None, w_out=lw, b_out=lb, type="lower",
                k=k, x0=l, y0=torch.sqrt(l)
            )

            m = (l + u) / 2
            k = 0.5 / torch.sqrt(m)

            self.add_linear(
                mask=None, w_out=uw, b_out=ub, type="upper",
                k=k, x0=m, y0=torch.sqrt(m)
            )

        return Bounds(
            self.args, self.p, self.eps,
            lw = lw, lb = lb,
            uw = uw, ub = ub
        )     

    def ori_relu(self):
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

    """
    Relaxation for exp(x):
        L: y = e^((l + u) / 2) * (x - (l + u) / 2) + e ^ ((l + u) / 2)
        U: y = (e^u - e^l) / (u - l) * (x - l) + e^l
    """
    def exp(self):
        l, u, mask_pos, mask_neg, mask_both, lw, lb, uw, ub = self.new()        

        if self.ibp:
            lb = torch.exp(l)
            ub = torch.exp(u)
        else:
            """
            To ensure that the global lower bound is always positive:
                e^alpha (l - alpha) + e^alpha > 0
                => alpha < l + 1
            """
            m = torch.min((l + u) / 2, l + 0.99)

            thres = torch.tensor(12.).to(self.device)

            def exp_with_trick(x):
                mask = torch.lt(x, thres).to(torch.float)
                return mask * torch.exp(torch.min(x, thres)) + \
                    (1 - mask) * (torch.exp(thres) * (x - thres + 1))

            kl = torch.exp(torch.min(m, thres))
            lw = self.lw * kl.unsqueeze(2) 
            lb = kl * (self.lb - m + 1)
    
            ku = (exp_with_trick(u) - exp_with_trick(l)) / (u - l + epsilon)
            uw = self.uw * ku.unsqueeze(2)
            ub = self.ub * ku - ku * l + exp_with_trick(l)

        return Bounds(
            self.args, self.p, self.eps,
            lw = lw, lb = lb,
            uw = uw, ub = ub
        )

    def softmax(self, verbose=False):
        bounds_exp = self.exp()
        bounds_sum = Bounds(
            self.args, self.p, self.eps,
            lw = torch.sum(bounds_exp.lw, dim=-1, keepdim=True).repeat(1, 1, 1, self.dim_out),
            uw = torch.sum(bounds_exp.uw, dim=-1, keepdim=True).repeat(1, 1, 1, self.dim_out),
            lb = torch.sum(bounds_exp.lb, dim=-1, keepdim=True).repeat(1, 1, self.dim_out),
            ub = torch.sum(bounds_exp.ub, dim=-1, keepdim=True).repeat(1, 1, self.dim_out),
        )
        return bounds_exp.divide(bounds_sum)

    def dense(self, dense, tvm_model_lut=None, FLAG=0):
        # print("tvm_model_lut = ", tvm_model_lut)
        # print("bound_in shape = ", self.uw.shape, ", ", self.lw.shape)
        # print("W shape = ", dense.weight.shape)
        # res1 = self.matmul(dense.weight, tvm_model_lut, FLAG)
        # print("bound_out w/o bias shape = ", res1.uw.shape, ", ", res1.lw.shape)
        # res2 = res1.add(dense.bias)
        # print("bound_out w/ bias shape = ", res2.uw.shape, ", ", res2.lw.shape)
        # return res2
        return self.matmul(dense.weight, tvm_model_lut, ansor_model_lut, self.FLAG).add(dense.bias)

    def ori_tanh(self):
        l, u, mask_pos, mask_neg, mask_both, lw, lb, uw, ub = self.new()

        if self.ibp:
            lb = torch.tanh(l)
            ub = torch.tanh(u)
        else:
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

            # lower bound for both
            diff = lambda d: (torch.tanh(u) - torch.tanh(d)) / (u - d + epsilon) - dtanh(d)
            d = l / 2
            _l = l
            _u = torch.zeros(l.shape).to(self.device)
            for t in range(max_iter):
                v = diff(d)
                mask_p = torch.gt(v, 0).to(torch.float)
                _l = d * mask_p + _l * (1 - mask_p)
                _u = d * (1 - mask_p) + _u * mask_p
                d = (d + _u) / 2 * mask_p + (d + _l) / 2 * (1 - mask_p)
            k = (torch.tanh(d) - torch.tanh(u)) / (d - u + epsilon)
            self.add_linear(
                mask=mask_both, w_out=lw, b_out=lb, type="lower",
                k=k, x0=d, y0=torch.tanh(d)
            )

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
            self.add_linear(
                mask=mask_both, w_out=uw, b_out=ub, type="upper",
                k=k, x0=d, y0=torch.tanh(d)
            )        

        return Bounds(
            self.args, self.p, self.eps,
            lw = lw, lb = lb,
            uw = uw, ub = ub
        )

    def act(self, act_name):
        if act_name == "relu":
            return self.relu()        
        else:
            raise NotImplementedError

    def layer_norm(self, normalizer, layer_norm):
        if layer_norm == "no":
            return self

        l_in, u_in = self.concretize()
        w_avg = torch.ones((self.dim_out, self.dim_out)).to(self.device) * (1. / self.dim_out)
        minus_mu = self.add(self.matmul(w_avg).multiply(-1.))

        l_minus_mu, u_minus_mu = minus_mu.concretize()
        dim = self.dim_out        

        if layer_norm == "standard":
            variance = minus_mu.sqr().matmul(w_avg)
            normalized = minus_mu.divide(variance.add(epsilon).sqrt())
        else:
            assert(layer_norm == "no_var")
            normalized = minus_mu

        normalized = normalized.multiply(normalizer.weight).add(normalizer.bias)

        return normalized

    # """
    # Requirement: x should be guaranteed to be positive
    # """
    def reciprocal(self):
        l, u = self.concretize()

        if self.ibp:
            lw = self.lw * 0.
            uw = self.uw * 0.
            lb = 1. / u
            ub = 1. / l
        else:
            m = (l + u) / 2

            assert(torch.min(l) >= epsilon)

            kl = -1 / m.pow(2)
            lw = self.uw * kl.unsqueeze(2)
            lb = self.ub * kl + 2 / m 

            ku = -1. / (l * u)
            uw = self.lw * ku.unsqueeze(2)
            ub = self.lb * ku - ku * l + 1 / l

        return Bounds(
            self.args, self.p, self.eps,
            lw = lw, lb = lb,
            uw = uw, ub = ub
        )    

    def relu(self):
        if self.FLAG == 1:
            return self.cuda_relu()
        # elif self.FLAG == 2:
        #     return self.tvm_relu()
        else:
            return self.ori_relu()

    def tanh(self):
        if self.FLAG == 1:
            return self.cuda_tanh()
        else:
            return self.ori_tanh()

    def matmul(self, W, tvm_model_lut=None, ansor_model_lut=None, tvm_model_path=None, ansor_model_path=None,FLAG=0):
        # if FLAG == 1:
        #     # print("cuda matmul. W.shape: ", W.shape, ", self.lw.shape: ", self.lw.shape)
        #     dim_Y_out, dim_out = W.shape
        #     if dim_Y_out < 64:
        #         return self.ori_matmul(W)
        #     elif dim_Y_out > 128:
        #         return self.ori_matmul(W)
        #     else:
        #         return self.cuda_matmul(W)
        # print("W_shape = ", W.shape)
        if self.FLAG == 2:
            if tvm_model_lut == None:
                return self.ori_matmul(W)
            else:
                return self.tvm_matmul(W, tvm_model_lut, tvm_model_path)
        elif self.FLAG == 3:
            if ansor_model_lut == None:
                return self.ori_matmul(W)
            else:
                return self.ansor_matmul(W, ansor_model_lut, ansor_model_path)
        else:
            return self.ori_matmul(W)

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

        return Bounds(self.args, self.p, self.eps,lw = _lw, lb = _lb,uw = _uw, ub = _ub)  


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

        return Bounds(self.args, self.p, self.eps,lw = _lw, lb = _lb,uw = _uw, ub = _ub)  


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
        
        # lb/ub have been transposed somewhere without contiguous(). This leads to memory layout issue in cuda implementation
        self.lb = self.lb.contiguous()
        self.ub = self.lb.contiguous()
        other.lb = other.lb.contiguous()
        other.ub = other.ub.contiguous()

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
            self.args, self.p, self.eps,
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
        
        # # lb/ub have been transposed somewhere without contiguous(). This leads to memory layout issue in cuda implementation
        self_lb = self.lb.contiguous()
        self_ub = self.lb.contiguous()
        other_lb = other.lb.contiguous()
        other_ub = other.ub.contiguous()

        c_bound.c_dot_product_verification_V(
            x_l, y_l, x_u, y_u,
            self_lb, self_ub, other_lb, other_ub,  
            transposed_lw, transposed_uw, transposed_other_lw, transposed_other_uw,
            lb, ub, lw, uw,
            batch_size, length, dim_out, dim_in, self.eps
        )

        lw = lw.transpose(2,3)
        uw = uw.transpose(2,3)

        return Bounds(
            self.args, self.p, self.eps,
            lw = lw, lb = lb, uw = uw, ub = ub
        )

    # def cuda_matmul(self, W):
    #     batch_size, length, dim_in, dim_out = self.lw.shape
    #     # print(self.lw.shape)
    #     # print(W.shape)
    #     # W = W.t().contiguous()
    #     # print("cuda_matmul. W.shape: ", W.shape)
    #     dim_y_out, dim_out = W.shape

    #     lw = torch.zeros(batch_size, length, dim_in, dim_y_out).to(self.device)
    #     lb = torch.zeros(batch_size, length, dim_y_out).to(self.device)
    #     uw = torch.zeros(batch_size, length, dim_in, dim_y_out).to(self.device)
    #     ub = torch.zeros(batch_size, length, dim_y_out).to(self.device)

    #     # transposed_lw = self.lw.transpose(2,3).contiguous()
    #     # transposed_uw = self.uw.transpose(2,3).contiguous()


    #     c_bound.c_matmul_verification(
    #         self.lb, self.ub, self.lw, self.uw,
    #         W, lb, ub, lw, uw,
    #         batch_size, length, dim_in, dim_out, dim_y_out
    #     )
        
    #     return Bounds(
    #         self.args, self.p, self.eps,
    #         lw = lw, lb = lb, uw = uw, ub = ub
    #     )
