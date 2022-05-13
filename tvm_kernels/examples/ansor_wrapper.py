# This is a warpper from pytorch model to ansor code.

from model.modules import Layer
import torch
from tvm import relay
import tvm
from tvm import auto_scheduler
import numpy as np
from tvm.contrib import graph_executor
import timeit

model = Layer(
    dim = 256,
    inner_dim=256,
    proj_dim=266,
    head=4,
    mult=4
).eval().cuda()


x = torch.randn(2, 4096, 256).cuda()
o = model(x)

##########################################################
# Obtrain the JIT scripted model via tracing
scripted_model = torch.jit.trace(model, x).eval().cuda()

##########################################################
# Port the model to TVM with pytorch frontend
shape_list = [("x", (2, 4096, 256))]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

target = tvm.target.Target("cuda", host="llvm")
log_file = "./performer_gpu.json"

##########################################################
# Extract search task for Ansor
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target, include_simple_tasks=False)
# Enumerate the tasks
for idx, task in enumerate(tasks):
    print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
    print(task.compute_dag)

##########################################################
# Begin Tuning

def run_tuning():
    # measure_ctx launches a different process for measurement to provide isolation
    # It protect the master process from GPU crashes
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10)

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=800 * len(tasks),  # change this to 800 & #task to achieve the best performance
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )

    tuner.tune(tune_option)

run_tuning()

dev = tvm.device(str(target), 0)
data_x = x.cpu().numpy()
output_shape = o.size()

##########################################################
# Compile and Evaluate

# Compile with the history best
print("Compile ...")
with auto_scheduler.ApplyHistoryBest(log_file):
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        lib_ansor = relay.build(mod, target=target, params=params)

# Create graph executer

module_ansor = graph_executor.GraphModule(lib_ansor["default"](dev))
module_ansor.set_input("x", data_x)

print(module_ansor)

# Evaluate
ftimer = module_ansor.module.time_evaluator("run", dev, repeat=20, min_repeat_ms=300)
prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
print("[TVM] Mean inference time (std dev): %.4f ms (%.4f ms)" % (np.mean(prof_res[-10:]), np.std(prof_res[-10:])))

# verify the output
module_ansor.run()
tvm_output = module_ansor.get_output(0, tvm.nd.empty(output_shape)).numpy()
print("max error: %.4f" % np.max(tvm_output - o.detach().cpu().numpy()))

##########################################################
# Evaluate (pytorch)

torch_res = (
    np.array(timeit.Timer(lambda: model(x)).repeat(repeat=20, number=300)) * 1e3 / 300
)
print("[Pytorch Naive] Mean inference time (std dev): %.4f ms (%.4f ms)" % (np.mean(torch_res[-10:]), np.std(torch_res[-10:])))

##########################################################
# Evaluate (pytorch JIT)

torch_res = (
    np.array(timeit.Timer(lambda: scripted_model(x)).repeat(repeat=20, number=300)) * 1e3 / 300
)
print("[Pytorch JIT] Mean inference time (std dev): %.4f ms (%.4f ms)" % (np.mean(torch_res[-10:]), np.std(torch_res[-10:])))

