import torch
import torch.nn as nn
import math, time
import sys

sys.path.append('/home/tianqi_tang/Faith-NNVerificationCompiler/')

epsilon = 1e-12

from HandTunedKernels.kernel_test.forward_test_bound import Bounds
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size=1;length=2;dim_in=1024;dim_out=1024;dim_y_out=1024

import tvm
import tvm.testing
from tvm import te
import numpy
import timeit

target_gpu = tvm.target.Target(target="cuda", host="llvm")
dev = tvm.device(target_gpu.kind.name, 0)
dtype="float32"

dim_out = 1024 #te.var("n")
dim_y_out = 1024 #te.var("m")
w = te.placeholder((dim_out, dim_y_out), name='w')
w_pos = te.compute(w.shape, lambda i,j: te.if_then_else(w[i,j]>0, w[i,j], 0.))
w_neg = te.compute(w.shape, lambda i,j: te.if_then_else(w[i,j]<0, w[i,j], 0.))
w_con = te.compute((2*dim_out,dim_y_out), lambda i,j: te.if_then_else(i<dim_out, w_pos[i,j], w_neg[i-dim_out,j]))

s = te.create_schedule(w_con.op)
num_thread = 64

bx, tx = s[w_con].split(w_con.op.axis[0], factor=num_thread)
by, ty = s[w_con].split(w_con.op.axis[1], factor=num_thread)
ty, ty_vec = s[w_con].split(ty, factor=8)
s[w_con].bind(bx, te.thread_axis("blockIdx.x"))
s[w_con].bind(tx, te.thread_axis("threadIdx.x"))
s[w_con].vectorize(ty_vec)

s[w_neg].compute_inline()
s[w_pos].compute_inline()

# s[w_con].bind(bx, te.thread_axis("blockIdx.x"))
# s[w_con].bind(by, te.thread_axis("blockIdx.y"))
# s[w_con].bind(tx, te.thread_axis("threadIdx.x"))
# s[w_con].bind(ty, te.thread_axis("threadIdx.y"))

# bx, tx = s[w_neg].split(w_neg.op.axis[0], factor=num_thread)
# by, ty = s[w_neg].split(w_neg.op.axis[1], factor=num_thread)
# s[w_neg].bind(bx, te.thread_axis("blockIdx.x"))
# s[w_neg].bind(by, te.thread_axis("blockIdx.y"))
# s[w_neg].bind(tx, te.thread_axis("threadIdx.x"))
# s[w_neg].bind(ty, te.thread_axis("threadIdx.y"))

# bx, tx = s[w_pos].split(w_pos.op.axis[0], factor=num_thread)
# by, ty = s[w_pos].split(w_pos.op.axis[1], factor=num_thread)
# s[w_pos].bind(bx, te.thread_axis("blockIdx.x"))
# s[w_pos].bind(by, te.thread_axis("blockIdx.y"))
# s[w_pos].bind(tx, te.thread_axis("threadIdx.x"))
# s[w_pos].bind(ty, te.thread_axis("threadIdx.y"))

func = tvm.build(s, [w, w_con], "cuda", name="myexp")
print(func.imported_modules[0].get_source())
a_raw = numpy.random.rand(dim_out, dim_y_out).astype(dtype)
a_tvm = tvm.nd.array(a_raw, dev)
d_raw = numpy.zeros((2*dim_out, dim_y_out)).astype(dtype)
d_tvm = tvm.nd.array(d_raw, dev)

func(a_tvm, d_tvm)
print(w)
print(w_con, s[w_con])
b_raw = numpy.zeros((dim_out, dim_y_out)).astype(dtype)
b_raw = a_raw
b_raw[b_raw<0] = 0
c_raw = numpy.zeros((dim_out, dim_y_out)).astype(dtype)
c_raw[c_raw>0] = 0
d_raw = numpy.concatenate([b_raw, c_raw])
tvm.testing.assert_allclose(d_tvm.numpy(), d_raw)