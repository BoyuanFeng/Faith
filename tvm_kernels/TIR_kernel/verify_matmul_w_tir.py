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
num_warp = 8
step_k = 32

length_dim_in = length * dim_in

tile_i = 16
tile_j = 32

thread_tile_i = 1
thread_tile_j = 2

W = tvm.tir.decl_buffer(shape=(dim_out, dim_Y_out), dtype=dtype, name="W")
x_lw = tvm.tir.decl_buffer(shape=(length, dim_in, dim_out), dtype=dtype, name="x_lw")
x_uw = tvm.tir.decl_buffer(shape=(length, dim_in, dim_out), dtype=dtype, name="x_uw")

y_lw = tvm.tir.decl_buffer(shape=(length, dim_in, dim_Y_out), dtype=dtype, name="y_lw")
y_uw = tvm.tir.decl_buffer(shape=(length, dim_in, dim_Y_out), dtype=dtype, name="y_uw")

ib = tvm.tir.ir_builder.create()
Wb = ib.buffer_ptr(W)
x_lwb = ib.buffer_ptr(x_lw)
x_uwb = ib.buffer_ptr(x_uw)
y_lwb = ib.buffer_ptr(y_lw)
y_uwb = ib.buffer_ptr(y_uw)

# Threads
block_i = te.thread_axis("blockIdx.x")
ib.scope_attr(block_i, "thread_extent", int(length_dim_in/tile_i))
block_j = te.thread_axis("blockIdx.y")
ib.scope_attr(block_j, "thread_extent", int(dim_Y_out/tile_j))
laneId = te.thread_axis("threadIdx.x")
ib.scope_attr(laneId, "thread_extent", 32)
warpId = te.thread_axis("threadIdx.y")
ib.scope_attr(warpId, "thread_extent", num_warp)

# Shared memory
x_lws = ib.allocate(dtype, shape=(step_k * tile_i,), name="shmem_x_lw", scope="shared")
x_uws = ib.allocate(dtype, shape=(step_k * tile_i,), name="shmem_x_uw", scope="shared")
ws = ib.allocate(dtype, shape=(step_k * tile_j,), name="shmem_w", scope="shared")

# Registers
y_lw_vals = ib.allocate(dtype, shape=(int(tile_i*tile_j/32/num_warp),), name="y_lw_vals", scope="local")

y_uw_vals = ib.allocate(dtype, shape=(int(tile_i*tile_j/32/num_warp),), name="y_uw_vals", scope="local")

with ib.for_range(begin=0, end=int(tile_i*tile_j/32/num_warp), name="y_init", dtype="int32") as y_init:
    y_lw_vals[y_init] = 0.
    y_uw_vals[y_init] = 0.

threads_i = tvm.tir.floordiv(warpId * 32 + laneId, int(tile_j / thread_tile_j))
threads_j = tvm.tir.floormod(warpId * 32 + laneId, int(tile_j / thread_tile_j))

with ib.for_range(begin=0, end=dim_out/32, name="k", dtype="int32") as k:
    with ib.for_range(begin=0, end=int(tile_i/num_warp), name="step", dtype="int32") as step:
        x_lws[warpId * step_k + laneId + step * num_warp * step_k] = x_lwb[block_i*tile_i * dim_out + warpId * dim_out + laneId + k * step_k + num_warp * step * dim_out]

    with ib.for_range(begin=0, end=int(tile_i/num_warp), name="step", dtype="int32") as step:
        x_uws[warpId * step_k + laneId + step * num_warp * step_k] = x_uwb[block_i*tile_i*dim_out + warpId * dim_out + laneId + k * step_k + num_warp * step * dim_out]
    
    with ib.for_range(begin=0, end=int(tile_j/num_warp), name="step", dtype="int32") as step:
        ws[warpId * step_k + laneId + step * num_warp * step_k] = Wb[block_j * tile_j * dim_out + warpId * dim_out + laneId + step_k * k + step * dim_out * num_warp]

    # Stage II: Compute y_lw, y_uw
    w_val = ib.allocate(dtype, shape=(1,), name="w_val", scope="local")
    lw_val = ib.allocate(dtype, shape=(1,), name="lw_val", scope="local")
    uw_val = ib.allocate(dtype, shape=(1,), name="uw_val", scope="local")

    with ib.for_range(begin=0, end=step_k, name="i", dtype="int32") as i:
        with ib.for_range(begin=0, end=thread_tile_i, name="t_tile_i", dtype="int32") as t_tile_i:
            lw_val[0] = x_lws[(threads_i * thread_tile_i + t_tile_i) * step_k + i]
            uw_val[0] = x_uws[(threads_i * thread_tile_i + t_tile_i) * step_k + i]
            with ib.for_range(begin=0, end=thread_tile_j, name="t_tile_j", dtype="int32") as t_tile_j:
                out_idx = t_tile_i * thread_tile_j + t_tile_j
                w_val[0] = ws[(threads_j * thread_tile_j + t_tile_j) * step_k + i]
                with ib.if_scope(w_val[0] > 0):
                    y_lw_vals[out_idx] += w_val[0] * lw_val[0]
                    y_uw_vals[out_idx] += w_val[0] * uw_val[0]
                with ib.else_scope():
                    y_lw_vals[out_idx] += w_val[0] * uw_val[0]
                    y_uw_vals[out_idx] += w_val[0] * lw_val[0]

# Write the results back
with ib.for_range(begin=0, end=thread_tile_i, name="t_tile_i", dtype="int32") as t_tile_i:
    with ib.for_range(begin=0, end=thread_tile_j, name="t_tile_j", dtype="int32") as t_tile_j:
        out_idx = t_tile_i * thread_tile_j + t_tile_j
        y_lwb[(block_i * tile_i + threads_i * thread_tile_i + t_tile_i) * dim_Y_out + block_j * tile_j + thread_tile_j * threads_j + t_tile_j] = y_lw_vals[out_idx]
        y_uwb[(block_i * tile_i + threads_i * thread_tile_i + t_tile_i) * dim_Y_out + block_j * tile_j + thread_tile_j * threads_j + t_tile_j] = y_uw_vals[out_idx]



prime_func = tvm.tir.PrimFunc(params=[y_lw, y_uw, W, x_lw, x_uw], body=ib.get())
print(prime_func)

dev = tvm.cuda(0)
func = tvm.build(prime_func, target="cuda")
print(func.imported_modules[0].get_source())

# Implement a test case
W_np = np.random.uniform(size=(dim_out, dim_Y_out), low=-1, high=1).astype(W.dtype)
x_lw_np = np.random.uniform(size=(length, dim_in, dim_out)).astype(x_lw.dtype)
x_uw_np = np.random.uniform(size=(length, dim_in, dim_out)).astype(x_uw.dtype)

W_tvm = tvm.nd.array(W_np, dev)
x_lw_tvm = tvm.nd.array(x_lw_np, dev)
x_uw_tvm = tvm.nd.array(x_uw_np, dev)

y_lw_tvm = tvm.nd.array(np.zeros((length, dim_in, dim_Y_out), dtype=y_lw.dtype), dev)
y_uw_tvm = tvm.nd.array(np.zeros((length, dim_in, dim_Y_out), dtype=y_lw.dtype), dev)

func(y_lw_tvm, y_uw_tvm, W_tvm, x_lw_tvm, x_uw_tvm)

# emulate the computation with numpy
W_pos_np = np.transpose(W_np).copy()
W_neg_np = np.transpose(W_np).copy()
W_pos_np[W_pos_np < 0] = 0
W_neg_np[W_neg_np > 0] = 0

y_lw_1 = np.matmul(x_lw_np, W_pos_np)
y_lw_2 = np.matmul(x_uw_np, W_neg_np)
y_lw_np = y_lw_1 + y_lw_2

y_uw_1 = np.matmul(x_uw_np, W_pos_np)
y_uw_2 = np.matmul(x_lw_np, W_neg_np)
y_uw_np = y_uw_1 + y_uw_2

tvm.testing.assert_allclose(y_lw_tvm.numpy(), y_lw_np, atol = 0.001)
tvm.testing.assert_allclose(y_uw_tvm.numpy(), y_uw_np, atol = 0.001)