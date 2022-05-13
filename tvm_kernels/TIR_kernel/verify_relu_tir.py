import tvm.tir
import tvm.ir
import tvm.testing
from tvm import te
import numpy as np

length = 128
dim_in = 1024
dim_out = dim_in
dtype = "float32"

epsilon = 1e-6

src_lb = tvm.tir.decl_buffer(shape=(length, dim_out), dtype=dtype, name="src_lb")
src_ub = tvm.tir.decl_buffer(shape=(length, dim_out), dtype=dtype, name="src_ub")
src_lw = tvm.tir.decl_buffer(shape=(length, dim_out, dim_in), dtype=dtype, name="src_lw")
src_uw = tvm.tir.decl_buffer(shape=(length, dim_out, dim_in), dtype=dtype, name="src_uw")

out_lb = tvm.tir.decl_buffer(shape=(length, dim_out), dtype=dtype, name="out_lb")
out_ub = tvm.tir.decl_buffer(shape=(length, dim_out), dtype=dtype, name="out_ub")
out_lw = tvm.tir.decl_buffer(shape=(length, dim_out, dim_in), dtype=dtype, name="out_lw")
out_uw = tvm.tir.decl_buffer(shape=(length, dim_out, dim_in), dtype=dtype, name="out_uw")


ib = tvm.tir.ir_builder.create()

src_lb_b = ib.buffer_ptr(src_lb)
src_ub_b = ib.buffer_ptr(src_ub)
src_lw_b = ib.buffer_ptr(src_lw)
src_uw_b = ib.buffer_ptr(src_uw)

out_lb_b = ib.buffer_ptr(out_lb)
out_ub_b = ib.buffer_ptr(out_ub)
out_lw_b = ib.buffer_ptr(out_lw)
out_uw_b = ib.buffer_ptr(out_uw)

length_idx = te.thread_axis("blockIdx.y")
ib.scope_attr(length_idx, "thread_extent", length)
dim_out_idx = te.thread_axis("blockIdx.x")
ib.scope_attr(dim_out_idx, "thread_extent", dim_out)
tx = te.thread_axis("threadIdx.x")
ib.scope_attr(tx, "thread_extent", 32)

square_lw = ib.allocate(dtype, shape=(1,), name="square_lw", scope="local")
square_uw = ib.allocate(dtype, shape=(1,), name="square_uw", scope="local")

src_lw_val = ib.allocate(dtype, shape=(int(dim_in/32), 1), name="src_lw_val", scope="local")
src_uw_val = ib.allocate(dtype, shape=(int(dim_in/32), 1), name="src_uw_val", scope="local")

src_lb_val = ib.allocate(dtype, shape=(1,), name="src_lb_val", scope="local")
src_ub_val = ib.allocate(dtype, shape=(1,), name="src_ub_val", scope="local")

l_val = ib.allocate(dtype, shape=(1,), name="l_val", scope="local")
u_val = ib.allocate(dtype, shape=(1,), name="u_val", scope="local")

lk = ib.allocate(dtype, shape=(1,), name="lk", scope="local")
uk = ib.allocate(dtype, shape=(1,), name="uk", scope="local")
l_x0 = ib.allocate(dtype, shape=(1,), name="l_x0", scope="local")

square_lw[0] = 0.
square_uw[0] = 0.

base_idx = length_idx * dim_out * dim_in + dim_out_idx * dim_in
idx = length_idx * dim_out + dim_out_idx

with ib.for_range(begin=0, end=dim_in/32, name="i", dtype="int32") as i:
    src_lw_val[i] = src_lw_b[base_idx + 32 * i + tx]
    src_uw_val[i] = src_uw_b[base_idx + 32 * i + tx]
    square_lw[0] += src_lw_val[i] * src_lw_val[i]
    square_uw[0] += src_uw_val[i] * src_uw_val[i]

with ib.for_range(begin=0, end=5, name="offset_step", dtype="int32") as offset_step:    
    # warp shuffle intrinsic
    square_lw[0] += tvm.tir.call_intrin(
        "float32", "tir.cuda.__shfl_down_sync", 
        0xffffffff, square_lw[0], tvm.tir.Cast("int32", 16/tvm.tir.power(2, tvm.tir.Cast("float32", offset_step))))
    square_uw[0] += tvm.tir.call_intrin(
        "float32", "tir.cuda.__shfl_down_sync", 
        0xffffffff, square_uw[0], tvm.tir.Cast("int32", 16/tvm.tir.power(2, tvm.tir.Cast("float32", offset_step))))

with ib.if_scope(tx.equal(0)):
    src_lb_val[0] = src_lb_b[idx]
    src_ub_val[0] = src_ub_b[idx]
    l_val[0] = -epsilon * tvm.tir.sqrt(square_lw[0]) + src_lb_val[0]
    u_val[0] = epsilon * tvm.tir.sqrt(square_uw[0]) + src_ub_val[0]

    lk[0] = 0.
    uk[0] = 0.
    l_x0[0] = 0.
    with ib.if_scope(l_val[0] >= 0):
        lk[0] = 1.
        uk[0] = 1.
    with ib.if_scope(tvm.tir.And(l_val[0] < 0, u_val[0] > 0)):
        uk[0] = u_val[0] / (u_val[0] - l_val[0] + epsilon)
        l_x0[0] = l_val[0]
        with ib.if_scope(u_val[0] > (-1. * l_val[0])):
            lk[0] = 1.
    out_lb_b[idx] = src_lb_val[0] * lk[0]
    out_ub_b[idx] = (src_ub_val[0] - l_x0[0]) * uk[0]

lk[0] = tvm.tir.call_intrin("float32", "tir.cuda.__shfl_sync",
    0xffffffff, lk[0], 0)

uk[0] = tvm.tir.call_intrin("float32", "tir.cuda.__shfl_sync",
    0xffffffff, uk[0], 0)

with ib.for_range(begin=0, end=dim_in/32, name="j", dtype="int32") as j:
    idx_w = length_idx * dim_out * dim_in + dim_out_idx * dim_in + j * 32 + tx
    out_lw_b[idx_w] = src_lw_val[j] * lk[0]
    out_uw_b[idx_w] = src_uw_val[j] * uk[0]

prime_func = tvm.tir.PrimFunc(params=[out_lb, out_ub, out_lw, out_uw, src_lb, src_ub, src_lw, src_uw], body=ib.get())
print(prime_func)


dev = tvm.cuda(0)
func = tvm.build(prime_func, target="cuda")
print(func.imported_modules[0].get_source())

# Implement a test case
src_lb_np = np.random.uniform(size=(length, dim_out), low=-1, high=1).astype(src_lb.dtype)
src_ub_np = np.random.uniform(size=(length, dim_out), low=0.5, high=1).astype(src_ub.dtype) + src_lb_np
src_lw_np = np.random.uniform(size=(length, dim_out, dim_in), low=-1, high=1).astype(src_lw.dtype)
src_uw_np = np.random.uniform(size=(length, dim_out, dim_in), low=0.5, high=1).astype(src_uw.dtype) + src_lw_np

src_lb_tvm = tvm.nd.array(src_lb_np, dev)
src_ub_tvm = tvm.nd.array(src_ub_np, dev)
src_lw_tvm = tvm.nd.array(src_lw_np, dev)
src_uw_tvm = tvm.nd.array(src_uw_np, dev)

out_lb_tvm = tvm.nd.array(np.zeros((length, dim_out), dtype=out_lb.dtype), dev)
out_ub_tvm = tvm.nd.array(np.zeros((length, dim_out), dtype=out_ub.dtype), dev)
out_lw_tvm = tvm.nd.array(np.zeros((length, dim_out, dim_in), dtype=out_lw.dtype), dev)
out_uw_tvm = tvm.nd.array(np.zeros((length, dim_out, dim_in), dtype=out_uw.dtype), dev)

func(out_lb_tvm, out_ub_tvm, out_lw_tvm, out_uw_tvm, src_lb_tvm, src_ub_tvm, src_lw_tvm, src_uw_tvm)

# emulate the computation with numpy

# concretize
src_l_np = src_lb_np.copy() - epsilon * np.linalg.norm(src_lw_np, ord=2, axis=-1)
src_u_np = src_ub_np.copy() + epsilon * np.linalg.norm(src_uw_np, ord=2, axis=-1)

mask_pos = np.greater(src_l_np, 0)
mask_neg = np.less(src_u_np, 0)
mask_both = 1. - mask_pos - mask_neg

out_lb_np = np.zeros((length, dim_out), dtype=out_lb.dtype)
out_ub_np = np.zeros((length, dim_out), dtype=out_ub.dtype)
out_lw_np = np.zeros((length, dim_out, dim_in), dtype=out_lw.dtype)
out_uw_np = np.zeros((length, dim_out, dim_in), dtype=out_uw.dtype)

def add_linear(mask, w_out, b_out, type, k, x0, y0):
    mask_w = np.expand_dims(mask, axis=2)
    mask_b = mask
    if type == "lower":
        w_pos, b_pos = src_lw_np, src_lb_np
        w_neg, b_neg = src_uw_np, src_ub_np
    else:
        w_pos, b_pos = src_uw_np, src_ub_np
        w_neg, b_neg = src_lw_np, src_lb_np
    mask_pos_ = np.greater(k, 0)
    w_out += mask_w * np.expand_dims(mask_pos_, axis=2) * w_pos * np.expand_dims(k, axis=2)
    b_out += mask_b * mask_pos_ * ((b_pos - x0) * k + y0)
    mask_neg_ = 1 - mask_pos_
    w_out += mask_w * np.expand_dims(mask_neg_, axis=2) * w_neg * np.expand_dims(k, axis=2)
    b_out += mask_b * mask_neg_ * ((b_neg - x0) * k + y0)

add_linear(
    mask=mask_neg, w_out=out_lw_np, b_out=out_lb_np, 
    type="lower", k=np.zeros(src_l_np.shape), x0=0, y0=0
)     

add_linear(
    mask=mask_neg, w_out=out_uw_np, b_out=out_ub_np, 
    type="upper", k=np.zeros(src_l_np.shape), x0=0, y0=0
)

add_linear(
    mask=mask_pos, w_out=out_lw_np, b_out=out_lb_np, 
    type="lower", k=np.ones(src_l_np.shape), x0=0, y0=0
)

add_linear(
    mask=mask_pos, w_out=out_uw_np, b_out=out_ub_np, 
    type="upper", k=np.ones(src_l_np.shape), x0=0, y0=0
)   

k = src_u_np / (src_u_np - src_l_np + epsilon)

add_linear(
    mask=mask_both, w_out=out_uw_np, b_out=out_ub_np, 
    type="upper", k=k, x0=src_l_np, y0=0
)

k = np.greater(np.abs(src_u_np), np.abs(src_l_np))

add_linear(
    mask=mask_both, w_out=out_lw_np, b_out=out_lb_np, 
    type="lower", k=k, x0=0, y0=0
)

tvm.testing.assert_allclose(out_lb_tvm.numpy(), out_lb_np, atol = 0.001)
tvm.testing.assert_allclose(out_ub_tvm.numpy(), out_ub_np, atol = 0.001)
tvm.testing.assert_allclose(out_lw_tvm.numpy(), out_lw_np, atol = 0.001)
tvm.testing.assert_allclose(out_uw_tvm.numpy(), out_uw_np, atol = 0.001)