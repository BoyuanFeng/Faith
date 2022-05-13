import os

import numpy as np
import tvm
from tvm import te, auto_scheduler

import os
os.environ["CUDA_VISIBLE_DEVICES"]='1' # 0 for A6000 on winnie, 1 for P6000 on winnie.
from tvm.autotvm.measure.measure_methods import set_cuda_target_arch

# Note that fusing all computation into one graph is not supported yet on Ansor.
# Check: https://discuss.tvm.apache.org/t/assertion-triggered-when-auto-scheduling/9613/4
@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def verify_matmul_not_supported_yet(length, dim_in, dim_out, dim_Y_out, dtype="float32"):
    W = te.placeholder((dim_out, dim_Y_out), name="W", dtype=dtype)
    x_lw = te.placeholder((length, dim_in, dim_out), name="x_lw", dtype=dtype)
    x_uw = te.placeholder((length, dim_in, dim_out), name="x_uw", dtype=dtype)
    x_lb = te.placeholder((length, dim_out), name="x_lb", dtype=dtype)
    x_ub = te.placeholder((length, dim_out), name="x_ub", dtype=dtype)
    
    y_lw = te.placeholder((length, dim_in, dim_Y_out), name="y_lw", dtype=dtype)
    y_uw = te.placeholder((length, dim_in, dim_Y_out), name="y_uw", dtype=dtype)
    y_lb = te.placeholder((length, dim_Y_out), name="y_lb", dtype=dtype)
    y_ub = te.placeholder((length, dim_Y_out), name="y_ub", dtype=dtype)
    
    W_pos = te.compute(
        W.shape, 
        lambda i,j: te.if_then_else(W[i,j]>0, W[i,j], 0.),
        name='w_pos'
    )
    W_neg = te.compute(W.shape, lambda i,j: te.if_then_else(W[i,j]<=0, W[i,j], 0.), name='w_neg')

    dout = te.reduce_axis((0, dim_out), "dout")
    y_lb_1 = te.compute(
        y_lb.shape,
        lambda l, i: 
            te.sum(x_lb[l,dout] * W_pos[dout,i], axis=dout),
        name='y_lb_1'
    )
    y_lb_2 = te.compute(
        y_lb.shape,
        lambda l, i: 
            te.sum(x_ub[l,dout] * W_neg[dout,i], axis=dout),
        name='y_lb_2'
    )
    y_lb = te.compute(
        y_lb.shape,
        lambda l, i: y_lb_1[l,i]+y_lb_2[l,i],
        name="y_lb"
    )

    y_ub_1 = te.compute(
        y_ub.shape,
        lambda l, i: 
            te.sum(x_ub[l,dout] * W_pos[dout,i], axis=dout),
        name='y_ub_1'
    )
    y_ub_2 = te.compute(
        y_ub.shape,
        lambda l, i: 
            te.sum(x_lb[l,dout] * W_pos[dout,i], axis=dout),
        name='y_ub_2'
    )
    y_ub = te.compute(
        y_ub.shape,
        lambda l, i: y_ub_1[l,i]+y_ub_2[l,i],
        name="y_ub"
    )

    y_lw_1 = te.compute(
        y_lw.shape,
        lambda l, j, i: 
            te.sum(x_lw[l,j,dout] * W_pos[dout,i], axis=dout),
        name='y_lw_1'
    )
    y_lw_2 = te.compute(
        y_lw.shape,
        lambda l, j, i: 
            te.sum(x_uw[l,j,dout] * W_neg[dout,i], axis=dout),
        name='y_lw_2'
    )
    y_lw = te.compute(
        y_lw.shape,
        lambda l, j, i: y_lw_1[l,j,i] + y_lw_2[l,j,i],
        name="y_lw"
    )

    y_uw_1 = te.compute(
        y_uw.shape,
        lambda l, j, i: 
            te.sum(x_uw[l,j,dout] * W_pos[dout,i], axis=dout),
        name='y_uw_1'
    )
    y_uw_2 = te.compute(
        y_uw.shape,
        lambda l, j, i: 
            te.sum(x_lw[l,j,dout] * W_neg[dout,i], axis=dout),
        name='y_uw_2'
    )
    y_uw = te.compute(
        y_uw.shape,
        lambda l, j, i: y_uw_1[l,j,i] + y_uw_2[l,j,i],
        name="y_uw"
    )

    return [W, x_lw, x_uw, x_lb, x_ub, y_lw, y_uw, y_lb, y_ub]

# Ansor does not support this type of kernel yet.
# In particular, the y_lb = y_lb_1+y_lb_2 is not supported in the computation graph.
@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def verify_matmul_not_supported(length, dim_in, dim_out, dim_Y_out, dtype="float32"):
    W = te.placeholder((dim_out, dim_Y_out), name="W", dtype=dtype)
    x_lb = te.placeholder((length, dim_out), name="x_lb", dtype=dtype)
    x_ub = te.placeholder((length, dim_out), name="x_ub", dtype=dtype)
    y_lb = te.placeholder((length, dim_Y_out), name="y_lb", dtype=dtype)
    
    W_pos = te.compute(W.shape, lambda i,j: te.if_then_else(W[i,j]>0, W[i,j], 0.), name='w_pos')
    W_neg = te.compute(W.shape, lambda i,j: te.if_then_else(W[i,j]<=0, W[i,j], 0.), name='w_neg')


    dout = te.reduce_axis((0, dim_out), "dout")
    y_lb_1 = te.compute(
        y_lb.shape,
        lambda l, i: 
            te.sum(x_lb[l,dout] * W_pos[dout,i], axis=dout),
        name='y_lb_1'
    )
    y_lb_2 = te.compute(
        y_lb.shape,
        lambda l, i: 
            te.sum(x_ub[l,dout] * W_neg[dout,i], axis=dout),
        name='y_lb_2'
    )
    y_lb = te.compute(
        y_lb.shape,
        lambda l, i: y_lb_1[l,i]+y_lb_2[l,i],
        name="y_lb"
    )

    return [W, x_lb, y_lb]

# Ansor does not report error for this version.
# However, it takes 25 minutes but cannot generate 1 schedule.
@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def verify_matmul_stuck(length, dim_in, dim_out, dim_Y_out, dtype="float32"):
    W = te.placeholder((dim_out, dim_Y_out), name="W", dtype=dtype)
    x_lb = te.placeholder((length, dim_out), name="x_lb", dtype=dtype)
    x_ub = te.placeholder((length, dim_out), name="x_ub", dtype=dtype)
    y_lb_1 = te.placeholder((length, dim_Y_out), name="y_lb_1", dtype=dtype)
    y_lb_2 = te.placeholder((length, dim_Y_out), name="y_lb_2", dtype=dtype)
    
    W_pos = te.compute(W.shape, lambda i,j: te.if_then_else(W[i,j]>0, W[i,j], 0.), name='w_pos')
    W_neg = te.compute(W.shape, lambda i,j: te.if_then_else(W[i,j]<=0, W[i,j], 0.), name='w_neg')


    dout = te.reduce_axis((0, dim_out), "dout")
    y_lb_1 = te.compute(
        y_lb_1.shape,
        lambda l, i: 
            te.sum(x_lb[l,dout] * W_pos[dout,i], axis=dout),
        name='y_lb_1'
    )
    y_lb_2 = te.compute(
        y_lb_1.shape,
        lambda l, i: 
            te.sum(x_ub[l,dout] * W_neg[dout,i], axis=dout),
        name='y_lb_2'
    )
    # y_lb = te.compute(
    #     y_lb.shape,
    #     lambda l, i: y_lb_1[l,i]+y_lb_2[l,i],
    #     name="y_lb"
    # )

    return [W, x_lb, y_lb_1, y_lb_2]

# Ansor keeps reporting errors when compiling this kernel:
#    Target has been reduced to 1 due to too many failures or duplications
#    See: https://discuss.tvm.apache.org/t/autoscheduler-prints-target-has-been-reduced-to-1-due-to-too-many-failures-or-duplications-and-fails-to-tune/10774/4
#    Also tried renaming operators. But it still stucks.
@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def verify_matmul_stuck2(length, dim_in, dim_out, dim_Y_out, dtype="float32"):
    W = te.placeholder((dim_out, dim_Y_out), name="I_1", dtype=dtype)
    x_lb = te.placeholder((length, dim_out), name="I_2", dtype=dtype)
    x_ub = te.placeholder((length, dim_out), name="I_3", dtype=dtype)
    
    W_pos = te.compute(W.shape, lambda i,j: te.if_then_else(W[i,j]>0, W[i,j], 0.), name='I_5')
    W_neg = te.compute(W.shape, lambda i,j: te.if_then_else(W[i,j]<=0, W[i,j], 0.), name='I_6')


    dout = te.reduce_axis((0, dim_out), "dout")
    y_lb = te.compute(
        (length, dim_Y_out),
        lambda l, i: 
            te.sum(x_lb[l,dout] * W_pos[dout,i] + x_ub[l,dout]*W_neg[dout,i], axis=dout),
        name='Y_7'
    )

    return [W, x_lb, y_lb]

# This is the largest subgraph that is supported by Ansor.
@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def verify_matmul_lb_1(length, dim_in, dim_out, dim_Y_out, dtype="float32"):
    W = te.placeholder((dim_out, dim_Y_out), name="W", dtype=dtype)
    x_lb = te.placeholder((length, dim_out), name="x_lb", dtype=dtype)

    W_pos = te.compute(W.shape, lambda i,j: te.if_then_else(W[i,j]>0, W[i,j], 0.), name='W_pos')

    dout = te.reduce_axis((0, dim_out), "dout")
    y_lb1 = te.compute(
        (length, dim_Y_out),
        lambda l, i: 
            te.sum(x_lb[l,dout] * W_pos[dout, i], axis=dout),
        name='y_lb1'
    )

    return [W, x_lb, y_lb1]

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def verify_matmul_lb_2(length, dim_in, dim_out, dim_Y_out, dtype="float32"):
    W = te.placeholder((dim_out, dim_Y_out), name="W", dtype=dtype)
    x_ub = te.placeholder((length, dim_out), name="x_lb", dtype=dtype)
    y_lb_1 = te.placeholder((length, dim_Y_out), name="y_lb_1", dtype=dtype)
    
    W_neg = te.compute(W.shape, lambda i,j: te.if_then_else(W[i,j]<=0, W[i,j], 0.), name='W_neg')

    dout = te.reduce_axis((0, dim_out), "dout")
    y_lb_2 = te.compute(
        y_lb_1.shape,
        lambda l, i: 
            te.sum(x_ub[l,dout] * W_neg[dout,i], axis=dout),
        name='y_lb_2'
    )
    y_lb = te.compute(y_lb_1.shape, lambda l, i: y_lb_1[l,i]+y_lb_2[l,i], name="y_lb")

    return [W, x_ub, y_lb_1, y_lb]

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def verify_matmul_ub_1(length, dim_in, dim_out, dim_Y_out, dtype="float32"):
    W = te.placeholder((dim_out, dim_Y_out), name="W", dtype=dtype)
    x_ub = te.placeholder((length, dim_out), name="x_ub", dtype=dtype)
    
    W_pos = te.compute(W.shape, lambda i,j: te.if_then_else(W[i,j]>0, W[i,j], 0.), name='W_pos')

    dout = te.reduce_axis((0, dim_out), "dout")
    y_ub_1 = te.compute(
        (length, dim_Y_out),
        lambda l, i: te.sum(x_ub[l,dout] * W_pos[dout, i], axis=dout),
        name='y_ub_1'
    )

    return [W, x_ub, y_ub_1]

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def verify_matmul_ub_2(length, dim_in, dim_out, dim_Y_out, dtype="float32"):
    W = te.placeholder((dim_out, dim_Y_out), name="W", dtype=dtype)
    x_lb = te.placeholder((length, dim_out), name="x_ub", dtype=dtype)
    y_ub_1 = te.placeholder((length, dim_Y_out), name="y_ub_1", dtype=dtype)
    
    W_neg = te.compute(W.shape, lambda i,j: te.if_then_else(W[i,j]<=0, W[i,j], 0.), name='W_neg')

    dout = te.reduce_axis((0, dim_out), "dout")
    y_ub_2 = te.compute(
        y_ub_1.shape,
        lambda l, i: 
            te.sum(x_lb[l,dout] * W_neg[dout, i], axis=dout),
        name='y_ub_2'
    )
    y_ub = te.compute(y_ub_1.shape, lambda l, i: y_ub_1[l,i]+y_ub_2[l,i], name="y_ub")

    return [W, x_lb, y_ub_1, y_ub]

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def verify_matmul_lw_1(length, dim_in, dim_out, dim_Y_out, dtype="float32"):
    W = te.placeholder((dim_out, dim_Y_out), name="W", dtype=dtype)
    x_lw = te.placeholder((length, dim_in, dim_out), name="x_lw", dtype=dtype)
    
    W_pos = te.compute(
        W.shape, 
        lambda i,j: te.if_then_else(W[i,j]>0, W[i,j], 0.),
        name='w_pos'
    )

    dout = te.reduce_axis((0, dim_out), "dout")
    y_lw_1 = te.compute(
        (length, dim_in, dim_Y_out),
        lambda l, j, i: 
            te.sum(x_lw[l,j,dout] * W_pos[dout, i], axis=dout),
        name='y_lw_1'
    )

    return [W, x_lw, y_lw_1]

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def verify_matmul_lw_2(length, dim_in, dim_out, dim_Y_out, dtype="float32"):
    W = te.placeholder((dim_out, dim_Y_out), name="W", dtype=dtype)
    x_uw = te.placeholder((length, dim_in, dim_out), name="x_uw", dtype=dtype)
    y_lw_1 = te.placeholder((length, dim_in, dim_Y_out), name="y_lw_1", dtype=dtype)
    
    W_neg = te.compute(
        W.shape, 
        lambda i,j: te.if_then_else(W[i,j]<=0, W[i,j], 0.),
        name='w_neg'
    )

    dout = te.reduce_axis((0, dim_out), "dout")
    y_lw_2 = te.compute(
        y_lw_1.shape,
        lambda l, j, i: 
            te.sum(x_uw[l,j,dout] * W_neg[dout, i], axis=dout),
        name='y_lw_2'
    )
    y_lw = te.compute(
        y_lw_1.shape,
        lambda l, j, i: y_lw_1[l,j,i] + y_lw_2[l,j,i],
        name="y_lw"
    )

    return [W, x_uw, y_lw_1, y_lw]


@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def verify_matmul_uw_1(length, dim_in, dim_out, dim_Y_out, dtype="float32"):
    W = te.placeholder((dim_out, dim_Y_out), name="W", dtype=dtype)
    x_uw = te.placeholder((length, dim_in, dim_out), name="x_uw", dtype=dtype)
    
    W_pos = te.compute(
        W.shape, 
        lambda i,j: te.if_then_else(W[i,j]>0, W[i,j], 0.),
        name='w_pos'
    )

    dout = te.reduce_axis((0, dim_out), "dout")
    y_uw_1 = te.compute(
        (length, dim_in, dim_Y_out),
        lambda l, j, i: 
            te.sum(x_uw[l,j,dout] * W_pos[dout, i], axis=dout),
        name='y_uw_1'
    )

    return [W, x_uw, y_uw_1]

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def verify_matmul_uw_2(length, dim_in, dim_out, dim_Y_out, dtype="float32"):
    W = te.placeholder((dim_out, dim_Y_out), name="W", dtype=dtype)
    x_lw = te.placeholder((length, dim_in, dim_out), name="x_lw", dtype=dtype)
    y_uw_1 = te.placeholder((length, dim_in, dim_Y_out), name="y_uw_1", dtype=dtype)
    
    W_neg = te.compute(
        W.shape, 
        lambda i,j: te.if_then_else(W[i,j]<=0, W[i,j], 0.),
        name='w_neg'
    )

    dout = te.reduce_axis((0, dim_out), "dout")
    y_uw_2 = te.compute(
        y_uw_1.shape,
        lambda l, j, i: 
            te.sum(x_lw[l,j,dout] * W_neg[dout, i], axis=dout),
        name='y_lw_2'
    )
    y_uw = te.compute(
        y_uw_1.shape,
        lambda l, j, i: y_uw_1[l,j,i] + y_uw_2[l,j,i],
        name="y_uw"
    )

    return [W, x_lw, y_uw_1, y_uw]

def ansor_tuner(func_pointer, func_args, log_file="ansor_autotuning.json", target=tvm.target.Target("llvm")):# (length, dim_in, dim_out, dim_Y_out)
    # length = 2
    # dim_in = dim_out = dim_Y_out = 64
    task = tvm.auto_scheduler.SearchTask(func=func_pointer, args=func_args, target=target)

    # Inspect the computational graph
    # print("Computational DAG:")
    # print(task.compute_dag)

    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=1,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )
    # Run auto-tuning (search)
    task.tune(tune_option)
    # Apply the best schedule
    sch, args = task.apply_best(log_file)
    return sch, args

length=2
# dim_in = 512
# dim_out = 512
# dim_Y_out = 512
dim_in = dim_out = dim_Y_out = 1024

# target = tvm.target.Target("llvm")
target = tvm.target.Target("cuda")

W_np = np.random.uniform(size=(dim_out, dim_Y_out)).astype(np.float32)
x_lb_np = np.random.uniform(size=(length, dim_out)).astype(np.float32)
x_ub_np = np.random.uniform(size=(length, dim_out)).astype(np.float32)
x_lw_np = np.random.uniform(size=(length, dim_in, dim_out)).astype(np.float32)
x_uw_np = np.random.uniform(size=(length, dim_in, dim_out)).astype(np.float32)

# dev = tvm.cpu()
dev = tvm.cuda()
W_tvm = tvm.nd.array(W_np, device=dev)
x_lb_tvm = tvm.nd.array(x_lb_np, device=dev)
x_ub_tvm = tvm.nd.array(x_ub_np, device=dev)
x_lw_tvm = tvm.nd.array(x_lw_np, device=dev)
x_uw_tvm = tvm.nd.array(x_uw_np, device=dev)
y_lb_1_tvm = tvm.nd.empty((length, dim_Y_out), device=dev)
y_lb_2_tvm = tvm.nd.empty((length, dim_Y_out), device=dev)
y_lb_tvm = tvm.nd.empty((length, dim_Y_out), device=dev)
y_ub_1_tvm = tvm.nd.empty((length, dim_Y_out), device=dev)
y_ub_2_tvm = tvm.nd.empty((length, dim_Y_out), device=dev)
y_ub_tvm = tvm.nd.empty((length, dim_Y_out), device=dev)
y_lw_1_tvm = tvm.nd.empty((length, dim_in, dim_Y_out), device=dev)
y_lw_2_tvm = tvm.nd.empty((length, dim_in, dim_Y_out), device=dev)
y_lw_tvm = tvm.nd.empty((length, dim_in, dim_Y_out), device=dev)
y_uw_1_tvm = tvm.nd.empty((length, dim_in, dim_Y_out), device=dev)
y_uw_2_tvm = tvm.nd.empty((length, dim_in, dim_Y_out), device=dev)
y_uw_tvm = tvm.nd.empty((length, dim_in, dim_Y_out), device=dev)

# Evaluate execution time.
def profile(func, func_args):
    evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
    # "Execution time of this operator in ms"
    return np.mean(evaluator(*func_args).results) * 1000