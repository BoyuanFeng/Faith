import torch
import torch.nn as nn
import math, time
import sys
import tvm
from tvm import relay
from tvm.contrib import graph_executor
from tvm import auto_scheduler
import numpy as np
from tvm.contrib import graph_executor

sys.path.append('/home/boyuan/verification_tianqi/')

epsilon = 1e-12


import os
# from Verifiers.Bounds import Bounds
from HandTunedKernels.kernel_test.cnn_forward_test_bound import Bounds


os.environ["CUDA_VISIBLE_DEVICES"]='1' # 0 for A6000 on winnie, 1 for P6000 on winnie.

def Bounds2Tuple(x):
    return tuple((torch.Tensor([[x.p]]), torch.Tensor([[x.eps]]), x.lw, x.lb, x.uw, x.ub))
def Tuple2Bounds(x):
    return Bounds(x[0], x[1], x[2], x[3], x[4], x[5])
# def Elements2BoundsDotProduct(x1, x2, x3, x4):
#     return BoundsDotProduct(args, p=2, eps=0.1, w=None, b=None, lw=x1, lb=x2, uw=x3, ub=x4)

class BoundsReLUWrapper(nn.Module):
    def __init__(self): #, ):
        super(BoundsReLUWrapper, self).__init__()
        
    def forward(self, p, eps, lw, lb, uw, ub):
        x = Bounds(float(p), float(eps), lw, lb, uw, ub).relu()
        return Bounds2Tuple(x)

class BoundsMatMulWrapper(nn.Module):
    def __init__(self, W): 
        super(BoundsMatMulWrapper, self).__init__()
        self.W = W

    def forward(self, p, eps, lw, lb, uw, ub):
        x = Bounds(float(p), float(eps), lw, lb, uw, ub).matmul(self.W)
        return Bounds2Tuple(x) 

class BoundsDotProductWrapper(nn.Module):
    def __init__(self):
        super(BoundsDotProductWrapper, self).__init__()

    def forward(self, p0, eps0, lw0, lb0, uw0, ub0, p1, eps1, lw1, lb1, uw1, ub1):
        x = Bounds(float(p0), float(eps0), lw0, lb0, uw0, ub0)
        y = Bounds(float(p1), float(eps1), lw1, lb1, uw1, ub1)
        return Bounds2Tuple(x.dot_product(y))


torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
p = 2
eps = 0.5
batch_size, length, dim_in, dim_out, dim_y_out = 8, 4, 64, 32, 32
lb = torch.rand(batch_size,length,dim_out).to(device)
ub = lb + torch.rand(batch_size,length,dim_out).to(device)
lw = torch.rand(batch_size,length,dim_in,dim_out).to(device) - 0.5
uw = torch.rand(batch_size,length,dim_in,dim_out).to(device) - 0.5
W = torch.rand(dim_y_out, dim_out).to(device) - 0.5
bound = Bounds(p=2,eps=0.5,lw=lw,lb=lb,uw=uw,ub=ub)
bound1 = Bounds(p=2,eps=0.5,lw=lw,lb=lb,uw=uw,ub=ub)

bound_relu_wrapper = BoundsReLUWrapper()
bound_matmul_wrapper = BoundsMatMulWrapper(W)
bound_dot_product_wrapper = BoundsDotProductWrapper()

example_relu_inputs = Bounds2Tuple(bound)
example_matmul_inputs = Bounds2Tuple(bound)#(p, eps, lw, lb, uw, ub)
example_dot_product_inputs = (*Bounds2Tuple(bound), *Bounds2Tuple(bound1))


# test1.forward(p, eps, lw, lb, uw, ub)
# test2.forward(p, eps, lw, lb, uw, ub)
# test3.forward(p, eps, lw, lb, uw, ub, *Bounds2Tuple(bound1))

scripted_relu_model = torch.jit.trace(bound_relu_wrapper.eval(), example_relu_inputs).eval()
# scripted_relu_model = torch.jit.script(bound_relu_wrapper)
scripted_matmul_model = torch.jit.trace(bound_matmul_wrapper.eval(), example_matmul_inputs).eval()
scripted_dot_product_model = torch.jit.trace(bound_dot_product_wrapper.eval(), example_dot_product_inputs).eval()
# scripted_dot_product_model = torch.jit.script(bound_dot_product_wrapper)