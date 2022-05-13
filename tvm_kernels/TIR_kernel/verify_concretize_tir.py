from os import devnull
import tvm.tir
import tvm.ir
import tvm.testing
from tvm import te
import numpy as np

length = 128
dim_in = 1024
dim_out = dim_in
dim_Y_out = dim_out
dtype = "float32"

epsilon = 0.3

x_lw = tvm.tir.decl_buffer(shape=(length, dim_out, dim_in), dtype=dtype, name="x_lw")
x_uw = tvm.tir.decl_buffer(shape=(length, dim_out, dim_in), dtype=dtype, name="x_uw")
x_lb = tvm.tir.decl_buffer(shape=(length, dim_out), dtype=dtype, name="x_lb")
x_ub = tvm.tir.decl_buffer(shape=(length, dim_out), dtype=dtype, name="x_ub")

x_l = tvm.tir.decl_buffer(shape=(length, dim_out), dtype=dtype, name="x_l")
x_u = tvm.tir.decl_buffer(shape=(length, dim_out), dtype=dtype, name="x_u")

ib = tvm.tir.ir_builder.create()
x_lw_b = ib.buffer_ptr(x_lw)
x_uw_b = ib.buffer_ptr(x_uw)
x_lb_b = ib.buffer_ptr(x_lb)
x_ub_b = ib.buffer_ptr(x_ub)
x_l_b = ib.buffer_ptr(x_l)
x_u_b = ib.buffer_ptr(x_u)

length_idx = te.thread_axis("blockIdx.y")
ib.scope_attr(length_idx, "thread_extent", length)
dim_out_idx = te.thread_axis("blockIdx.x")
ib.scope_attr(dim_out_idx, "thread_extent", dim_out)
tx = te.thread_axis("threadIdx.x")
ib.scope_attr(tx, "thread_extent", 32)

idx = length_idx * dim_out + dim_out_idx
base_idx = length_idx * dim_out * dim_in + dim_out_idx * dim_in

# Compute norm
square_lw = ib.allocate(dtype, shape=(1,), name="square_lw", scope="local")
square_uw = ib.allocate(dtype, shape=(1,), name="square_uw", scope="local")

square_lw[0] = 0.
square_uw[0] = 0.

val_lw = ib.allocate(dtype, shape=(1,), name="val_lw", scope="local")
val_uw = ib.allocate(dtype, shape=(1,), name="val_uw", scope="local")

with ib.for_range(begin=0, end=dim_in/32, name="i", dtype="int32") as i:
    val_lw[0] = x_lw_b[base_idx + i * 32 + tx]
    square_lw[0] += val_lw[0] * val_lw[0]

    val_uw[0] = x_uw_b[base_idx + i * 32 + tx]
    square_uw[0] += val_uw[0] * val_uw[0]

with ib.for_range(begin=0, end=5, name="offset_step", dtype="int32") as offset_step:    
    # warp shuffle intrinsic
    square_lw[0] += tvm.tir.call_intrin(
        "float32", "tir.cuda.__shfl_down_sync", 
        0xffffffff, square_lw[0], tvm.tir.Cast("int32", 16/tvm.tir.power(2, tvm.tir.Cast("float32", offset_step))))
    square_uw[0] += tvm.tir.call_intrin(
        "float32", "tir.cuda.__shfl_down_sync", 
        0xffffffff, square_uw[0], tvm.tir.Cast("int32", 16/tvm.tir.power(2, tvm.tir.Cast("float32", offset_step))))

with ib.if_scope(tx.equal(0)):
    x_l_b[idx] = -epsilon * tvm.tir.sqrt(square_lw[0]) + x_lb_b[idx]
    x_u_b[idx] = epsilon * tvm.tir.sqrt(square_uw[0]) + x_ub_b[idx]



prime_func = tvm.tir.PrimFunc(params=[x_l, x_u, x_lb, x_ub, x_lw, x_uw], body=ib.get())
print(prime_func)

dev = tvm.cuda(0)
func = tvm.build(prime_func, target="cuda")
print(func.imported_modules[0].get_source())

x_lw_np = np.random.uniform(size=(length, dim_out, dim_in)).astype(x_lw.dtype)
x_uw_np = np.random.uniform(size=(length, dim_out, dim_in)).astype(x_uw.dtype)

x_lb_np = np.random.uniform(size=(length, dim_out)).astype(x_lb.dtype)
x_ub_np = np.random.uniform(size=(length, dim_out)).astype(x_ub.dtype)


x_lb_tvm = tvm.nd.array(x_lb_np, dev)
x_ub_tvm = tvm.nd.array(x_ub_np, dev)

x_lw_tvm = tvm.nd.array(x_lw_np, dev)
x_uw_tvm = tvm.nd.array(x_uw_np, dev)

x_l_tvm = tvm.nd.array(np.zeros((length, dim_out), dtype=x_l.dtype), dev)
x_u_tvm = tvm.nd.array(np.zeros((length, dim_out), dtype=x_u.dtype), dev)

func(x_l_tvm, x_u_tvm, x_lb_tvm, x_ub_tvm, x_lw_tvm, x_uw_tvm)

# emulate the computation with numpy
x_l_np = x_lb_np.copy() - epsilon * np.linalg.norm(x_lw_np, ord=2, axis=-1)
x_u_np = x_ub_np.copy() + epsilon * np.linalg.norm(x_uw_np, ord=2, axis=-1)

tvm.testing.assert_allclose(x_l_tvm.numpy(), x_l_np, atol = 0.001)
tvm.testing.assert_allclose(x_u_tvm.numpy(), x_u_np, atol = 0.001)