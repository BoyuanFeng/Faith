import tvm.tir
import tvm.ir
import tvm.testing
from tvm import te
import numpy as np

length = 64
dim_in = 512
dim_out = dim_in
dim_Y_out = dim_out
dtype = "float32"

W = tvm.tir.decl_buffer(shape=(dim_out, dim_Y_out), dtype=dtype, name="W")
x_lb = tvm.tir.decl_buffer(shape=(length, dim_out), dtype=dtype, name="x_lb")
x_ub = tvm.tir.decl_buffer(shape=(length, dim_out), dtype=dtype, name="x_ub")

y_lb = tvm.tir.decl_buffer(shape=(length, dim_Y_out), dtype=dtype, name="y_lb")
y_ub = tvm.tir.decl_buffer(shape=(length, dim_Y_out), dtype=dtype, name="y_ub")

ib = tvm.tir.ir_builder.create()
Wb = ib.buffer_ptr(W)
x_lbb = ib.buffer_ptr(x_lb)
x_ubb = ib.buffer_ptr(x_ub)
y_lbb = ib.buffer_ptr(y_lb)
y_ubb = ib.buffer_ptr(y_ub)

dim_y_out_idx = te.thread_axis("blockIdx.x")
ib.scope_attr(dim_y_out_idx, "thread_extent", dim_Y_out)
length_idx = te.thread_axis("blockIdx.y")
ib.scope_attr(length_idx, "thread_extent", length)
tx = te.thread_axis("threadIdx.x")
ib.scope_attr(tx, "thread_extent", 32)

y_lb_val = ib.allocate(dtype, shape=(1,), name="y_lb_val", scope="local")
y_ub_val = ib.allocate(dtype, shape=(1,), name="y_ub_val", scope="local")
x_lb_val = ib.allocate(dtype, shape=(1,), name="x_lb_val", scope="local")
x_ub_val = ib.allocate(dtype, shape=(1,), name="x_ub_val", scope="local")
w = ib.allocate(dtype, shape=(1,), name="w", scope="local")

y_lb_val[0] = 0.
y_ub_val[0] = 0.

with ib.for_range(begin=0, end=dim_out/32, name="i", dtype="int32") as i:
    w[0] = Wb[dim_y_out_idx * dim_out + i * 32 + tx]
    x_lb_val[0] = x_lbb[length_idx*dim_out + i * 32 + tx]
    x_ub_val[0] = x_ubb[length_idx*dim_out + i * 32 + tx]
    with ib.if_scope(w[0] > 0):
        y_lb_val[0] += w[0] * x_lb_val[0]
        y_ub_val[0] += w[0] * x_ub_val[0]
    with ib.else_scope():
        y_lb_val[0] += w[0] * x_ub_val[0]
        y_ub_val[0] += w[0] * x_lb_val[0]

with ib.for_range(begin=0, end=5, name="offset_step", dtype="int32") as offset_step:    
    # warp shuffle intrinsic
    y_lb_val[0] += tvm.tir.call_intrin(
        "float32", "tir.cuda.__shfl_down_sync", 
        0xffffffff, y_lb_val[0], tvm.tir.Cast("int32", 16/tvm.tir.power(2, tvm.tir.Cast("float32", offset_step))))
    y_ub_val[0] += tvm.tir.call_intrin(
        "float32", "tir.cuda.__shfl_down_sync", 
        0xffffffff, y_ub_val[0], tvm.tir.Cast("int32", 16/tvm.tir.power(2, tvm.tir.Cast("float32", offset_step))))

with ib.if_scope(tx.equal(0)):
    y_lbb[length_idx*dim_Y_out + dim_y_out_idx] = y_lb_val[0]
    y_ubb[length_idx*dim_Y_out + dim_y_out_idx] = y_ub_val[0]
        


prime_func = tvm.tir.PrimFunc(params=[y_lb, y_ub, W, x_lb, x_ub], body=ib.get())
print(prime_func)

dev = tvm.cuda(0)
func = tvm.build(prime_func, target="cuda")
print(func.imported_modules[0].get_source())

# Implement a test case
W_np = np.random.uniform(size=(dim_out, dim_Y_out), low=-1, high=1).astype(W.dtype)
x_lb_np = np.random.uniform(size=(length, dim_out)).astype(x_lb.dtype)
x_ub_np = np.random.uniform(size=(length, dim_out)).astype(x_ub.dtype)

W_tvm = tvm.nd.array(W_np, dev)
x_lb_tvm = tvm.nd.array(x_lb_np, dev)
x_ub_tvm = tvm.nd.array(x_ub_np, dev)

y_lb_tvm = tvm.nd.array(np.zeros((length, dim_Y_out), dtype=y_lb.dtype), dev)
y_ub_tvm = tvm.nd.array(np.zeros((length, dim_Y_out), dtype=y_lb.dtype), dev)

func(y_lb_tvm, y_ub_tvm, W_tvm, x_lb_tvm, x_ub_tvm)

# emulate the computation with numpy
W_pos_np = np.transpose(W_np).copy()
W_neg_np = np.transpose(W_np).copy()
W_pos_np[W_pos_np < 0] = 0
W_neg_np[W_neg_np > 0] = 0

y_lb_1 = np.matmul(x_lb_np, W_pos_np)
y_lb_2 = np.matmul(x_ub_np, W_neg_np)
y_lb_np = y_lb_1 + y_lb_2

y_ub_1 = np.matmul(x_ub_np, W_pos_np)
y_ub_2 = np.matmul(x_lb_np, W_neg_np)
y_ub_np = y_ub_1 + y_ub_2

tvm.testing.assert_allclose(y_lb_tvm.numpy(), y_lb_np, atol = 0.001)
tvm.testing.assert_allclose(y_ub_tvm.numpy(), y_ub_np, atol = 0.001)
