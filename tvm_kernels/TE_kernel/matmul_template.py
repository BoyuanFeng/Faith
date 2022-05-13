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

m = 1024 #te.var("m")
n = 1024 #te.var("n")
k = 512 #te.var("k")

A = te.placeholder((m,k), name="A")
W = te.placeholder((k,n), name="W")
rk = te.reduce_axis((0, k), "k")
B = te.compute((m,n), lambda i, j:te.sum(A[i,rk]*W[rk,j], axis=rk))

s = te.create_schedule(B.op)
AA = s.cache_read(A, "shared", [B])
WW = s.cache_read(W, "shared", [B])
AL = s.cache_read(AA, "local", [B])
WL = s.cache_read(WW, "local", [B])
BL = s.cache_write(B, "local")

tile = 8
num_thread = 8
block_factor = tile * num_thread
step = 8
vthread = 2

# Get the GPU thread indices
block_x = te.thread_axis("blockIdx.x")
block_y = te.thread_axis("blockIdx.y")
thread_x = te.thread_axis((0, num_thread), "threadIdx.x")
thread_y = te.thread_axis((0, num_thread), "threadIdx.y")

# Split the workloads
m, n = s[B].op.axis
by, ty = s[B].split(m, factor=block_factor)
bx, tx = s[B].split(n, factor=block_factor)

# Bind the iteration variables to GPU thread indices
s[B].bind(by, block_y)
s[B].bind(bx, block_x)
s[B].bind(ty, thread_y)
s[B].bind(tx, thread_x)

s[BL].compute_at(s[B], tx)

func = tvm.build(s, [A, W, B], "cuda")
dev = tvm.cuda(0)






