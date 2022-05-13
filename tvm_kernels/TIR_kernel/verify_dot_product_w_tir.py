import tvm.tir
import tvm.ir
import tvm.testing
from tvm import te
import numpy as np

length = 128
dim_in = 256
dim_out = dim_in
dim_Y_out = dim_out
dtype = "float32"

x_l = tvm.tir.decl_buffer(shape=(length, dim_out), dtype=dtype, name="x_l")
y_l = tvm.tir.decl_buffer(shape=(length, dim_out), dtype=dtype, name="y_l")
y_u = tvm.tir.decl_buffer(shape=(length, dim_out), dtype=dtype, name="y_u")

x_lw = tvm.tir.decl_buffer(shape=(length, dim_out, dim_in), dtype=dtype, name="x_lw")
x_uw = tvm.tir.decl_buffer(shape=(length, dim_out, dim_in), dtype=dtype, name="x_uw")
y_lw = tvm.tir.decl_buffer(shape=(length, dim_out, dim_in), dtype=dtype, name="y_lw")
y_uw = tvm.tir.decl_buffer(shape=(length, dim_out, dim_in), dtype=dtype, name="y_uw")

z_lw = tvm.tir.decl_buffer(shape=(length, length, dim_in), dtype=dtype, name="z_lw")
z_uw = tvm.tir.decl_buffer(shape=(length, length, dim_in), dtype=dtype, name="z_uw")

ib = tvm.tir.ir_builder.create()
x_l_b = ib.buffer_ptr(x_l)
y_l_b = ib.buffer_ptr(y_l)
y_u_b = ib.buffer_ptr(y_u)

x_lw_b = ib.buffer_ptr(x_lw)
x_uw_b = ib.buffer_ptr(x_uw)
y_lw_b = ib.buffer_ptr(y_lw)
y_uw_b = ib.buffer_ptr(y_uw)

z_lw_b = ib.buffer_ptr(z_lw)
z_uw_b = ib.buffer_ptr(z_uw)

row_idx = te.thread_axis("blockIdx.z")
ib.scope_attr(row_idx, "thread_extent", length)
col_idx = te.thread_axis("blockIdx.y")
ib.scope_attr(col_idx, "thread_extent", length)
m = te.thread_axis("blockIdx.x")
ib.scope_attr(m, "thread_extent", int(dim_in/32))
tx = te.thread_axis("threadIdx.x")
ib.scope_attr(tx, "thread_extent", 32)

s_lw_val = ib.allocate(dtype, shape=(1,), name="s_lw_val", scope="local")
s_uw_val = ib.allocate(dtype, shape=(1,), name="s_uw_val", scope="local")

s_lw_val[0] = 0.
s_uw_val[0] = 0.

x_l_val = ib.allocate(dtype, shape=(1,), name="x_l_val", scope="local")
y_l_val = ib.allocate(dtype, shape=(1,), name="y_l_val", scope="local")
y_u_val = ib.allocate(dtype, shape=(1,), name="y_u_val", scope="local")
x_lw_val = ib.allocate(dtype, shape=(1,), name="x_lw_val", scope="local")
x_uw_val = ib.allocate(dtype, shape=(1,), name="x_uw_val", scope="local")
y_lw_val = ib.allocate(dtype, shape=(1,), name="y_lw_val", scope="local")
y_uw_val = ib.allocate(dtype, shape=(1,), name="y_uw_val", scope="local")

alpha_l = ib.allocate(dtype, shape=(1,), name="alpha_l", scope="local")
alpha_u = ib.allocate(dtype, shape=(1,), name="alpha_u", scope="local")
beta_l = ib.allocate(dtype, shape=(1,), name="beta_l", scope="local")
beta_u = ib.allocate(dtype, shape=(1,), name="beta_u", scope="local")

with ib.for_range(begin=0, end=dim_out, name="k", dtype="int32") as k:
    x_l_val[0] = x_l_b[row_idx * dim_out + k]
    y_l_val[0] = y_l_b[col_idx * dim_out + k]
    y_u_val[0] = y_u_b[col_idx * dim_out + k]

    alpha_l[0] = y_l_val[0]
    alpha_u[0] = y_u_val[0]
    beta_l[0]  = x_l_val[0]
    beta_u[0] = x_l_val[0]

    x_lw_val[0] = x_lw_b[row_idx * dim_in * dim_out + k * dim_in + 32 * m + tx]
    x_uw_val[0] = x_uw_b[row_idx * dim_in * dim_out + k * dim_in + 32 * m + tx]

    y_lw_val[0] = y_lw_b[col_idx * dim_in * dim_out + k * dim_in + 32 * m + tx]
    y_uw_val[0] = y_uw_b[col_idx * dim_in * dim_out + k * dim_in + 32 * m + tx]

    with ib.if_scope(alpha_l[0] > 0):
        s_lw_val[0] += alpha_l[0] * x_lw_val[0]
    with ib.else_scope():
        s_lw_val[0] += alpha_l[0] * x_uw_val[0]
    
    with ib.if_scope(alpha_u[0] > 0):
        s_uw_val[0] += alpha_u[0] * x_uw_val[0]
    with ib.else_scope():
        s_uw_val[0] += alpha_u[0] * x_lw_val[0]
    
    with ib.if_scope(beta_l[0] > 0):
        s_lw_val[0] += beta_l[0] * y_lw_val[0]
    with ib.else_scope():
        s_lw_val[0] += beta_l[0] * y_uw_val[0]

    with ib.if_scope(beta_u[0] > 0):
        s_uw_val[0] += beta_u[0] * y_uw_val[0]
    with ib.else_scope():
        s_uw_val[0] += beta_u[0] * y_lw_val[0]

z_lw_b[row_idx * length * dim_in + col_idx * dim_in + 32 * m + tx] = s_lw_val[0]
z_uw_b[row_idx * length * dim_in + col_idx * dim_in + 32 * m + tx] = s_uw_val[0]



prime_func = tvm.tir.PrimFunc(params=[z_lw, z_uw, x_l, y_l, y_u, x_lw, x_uw, y_lw, y_uw], body=ib.get())
print(prime_func)

dev = tvm.cuda(0)
func = tvm.build(prime_func, target="cuda")
print(func.imported_modules[0].get_source())

x_l_np = np.random.uniform(size=(length, dim_out), low=-1, high=1).astype(x_l.dtype)
y_l_np = np.random.uniform(size=(length, dim_out), low=-1, high=1).astype(y_l.dtype)
y_u_np = np.random.uniform(size=(length, dim_out), low=-1, high=1).astype(y_u.dtype)

x_lw_np = np.random.uniform(size=(length, dim_out, dim_in), low=-1, high=1).astype(x_lw.dtype)
x_uw_np = np.random.uniform(size=(length, dim_out, dim_in), low=-1, high=1).astype(x_uw.dtype)
y_lw_np = np.random.uniform(size=(length, dim_out, dim_in), low=-1, high=1).astype(y_lw.dtype)
y_uw_np = np.random.uniform(size=(length, dim_out, dim_in), low=-1, high=1).astype(y_uw.dtype)

x_l_tvm = tvm.nd.array(x_l_np, dev)
y_l_tvm = tvm.nd.array(y_l_np, dev)
y_u_tvm = tvm.nd.array(y_u_np, dev)

x_lw_tvm = tvm.nd.array(x_lw_np, dev)
x_uw_tvm = tvm.nd.array(x_uw_np, dev)
y_lw_tvm = tvm.nd.array(y_lw_np, dev)
y_uw_tvm = tvm.nd.array(y_uw_np, dev)

z_lw_tvm = tvm.nd.array(np.zeros((length, length, dim_in), dtype=z_lw.dtype), dev)
z_uw_tvm = tvm.nd.array(np.zeros((length, length, dim_in), dtype=z_uw.dtype), dev)

func(z_lw_tvm, z_uw_tvm, x_l_tvm, y_l_tvm, y_u_tvm, x_lw_tvm, x_uw_tvm, y_lw_tvm, y_uw_tvm)

# emulate the computation with numpy

z_lw_np = np.zeros((length, length, dim_in), dtype=z_lw.dtype)
z_uw_np = np.zeros((length, length, dim_in), dtype=z_uw.dtype)

alpha_l = np.tile(y_l_np.reshape(-1), (length,))
alpha_u = np.tile(y_u_np.reshape(-1), (length))
beta_l = np.tile(x_l_np, (1, length)).reshape(-1)
beta_u = np.tile(x_l_np, (1, length)).reshape(-1)

alpha_l = alpha_l.reshape(length, length, dim_out)
alpha_u = alpha_u.reshape(length, length, dim_out)
beta_l = beta_l.reshape(length, length, dim_out)
beta_u = beta_u.reshape(length, length, dim_out)

def add_w_alpha(new, old, weight, cmp):
    new += np.einsum("loi,lto->lti", old, weight * cmp(weight, 0))
    # a = old[t].reshape(self.length, self.dim_in, 1, self.dim_out)
    # b = (weight * cmp(weight, 0).to(torch.float))\
    #     .reshape(self.length, 1, other.length, self.dim_out)\
    #     .transpose(2, 3) 
    # new[t, :, :, :] += a[:, :, :, :].matmul(b[:, :, :, :])\
    #     .reshape(self.length, self.dim_in, other.length) 

def add_w_beta(new, old, weight, cmp):
    new += np.einsum("toi,lto->lti", old, weight * cmp(weight, 0))
    # a = old[t].reshape(other.length, self.dim_in, 1, self.dim_out)
    # b = (weight * cmp(weight, 0).to(torch.float))\
    #     .transpose(0, 1)\
    #     .reshape(other.length, 1, self.length, self.dim_out)\
    #     .transpose(2, 3)
    # new[t, :, :, :] += a[:, :, :, :].matmul(b[:, :, :, :])\
    #     .reshape(other.length, self.dim_in, self.length).transpose(0, 2) 

add_w_alpha(z_lw_np, x_lw_np, alpha_l, np.greater)
add_w_alpha(z_lw_np, x_uw_np, alpha_l, np.less)
add_w_beta(z_lw_np, y_lw_np, beta_l, np.greater)
add_w_beta(z_lw_np, y_uw_np, beta_l, np.less)

add_w_alpha(z_uw_np, x_uw_np, alpha_u, np.greater)
add_w_alpha(z_uw_np, x_lw_np, alpha_u, np.less)
add_w_beta(z_uw_np, y_uw_np, beta_u, np.greater)
add_w_beta(z_uw_np, y_lw_np, beta_u, np.less)

tvm.testing.assert_allclose(z_lw_tvm.numpy(), z_lw_np, atol = 0.001)
tvm.testing.assert_allclose(z_uw_tvm.numpy(), z_uw_np, atol = 0.001)