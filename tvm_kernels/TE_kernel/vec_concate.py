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


n = 1024 #te.var("n")
m = 1024 #te.var("m")
A = te.placeholder((n,), name="A")
B = te.compute(A.shape, lambda i: te.if_then_else(A[i]>0, A[i], 0.))
C = te.compute(B.shape, lambda i: te.if_then_else(A[i]<0, A[i], 0.))
D = te.compute((2*n,), lambda i: te.if_then_else(i<n, B[i], C[i-n]))

s = te.create_schedule(D.op)
num_thread = 64

bx, tx = s[D].split(D.op.axis[0], factor=num_thread)
s[D].bind(bx, te.thread_axis("blockIdx.x"))
s[D].bind(tx, te.thread_axis("threadIdx.x"))

bx, tx = s[C].split(C.op.axis[0], factor=num_thread)
s[C].bind(bx, te.thread_axis("blockIdx.x"))
s[C].bind(tx, te.thread_axis("threadIdx.x"))

bx, tx = s[B].split(B.op.axis[0], factor=num_thread)
s[B].bind(bx, te.thread_axis("blockIdx.x"))
s[B].bind(tx, te.thread_axis("threadIdx.x"))

func = tvm.build(s, [A, D], "cuda", name="myexp")
print(func.imported_modules[0].get_source())
a_raw = numpy.random.rand(n).astype(dtype)
a_tvm = tvm.nd.array(a_raw, dev)
d_raw = numpy.zeros((2*n)).astype(dtype)
d_tvm = tvm.nd.array(d_raw, dev)

func(a_tvm, d_tvm)
print(A)
print(B, s[B])
b_raw = numpy.zeros((n)).astype(dtype)
b_raw = a_raw
b_raw[b_raw<0] = 0
c_raw = numpy.zeros((n)).astype(dtype)
c_raw[c_raw>0] = 0
d_raw = numpy.concatenate([b_raw, c_raw])
tvm.testing.assert_allclose(d_tvm.numpy(), d_raw)