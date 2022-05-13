# Command to run:
#     python Benchmark/pytorch_benchmark.py --dir model --data yelp
# Here, the --dir and --data are dummy variables. Just for reusing parser code from Transformer_Verification_Code

import torch
import math, time, os
import tvm
from tvm.contrib import graph_executor

epsilon = 1e-12

import sys

home_path = '/home/boyuan/verification_tianqi/'
sys.path.append(home_path+'Transformer_Verification_Code')
from Parser import Parser, update_arguments
from data_utils import load_data, get_batches, set_seeds

sys.path.append(home_path+'Transformer_Verification_Code/Verifiers')
from Bounds import Bounds

argv = sys.argv[1:]
parser = Parser().getParser()
args, _ = parser.parse_known_args(argv)
args = update_arguments(args)
set_seeds(args.seed)

tvm_model_lut = {}
ansor_model_lut = {}
tvm_model_path = os.getcwd()+'/../Transformer_Verification_Code'
ansor_model_path = os.getcwd()+'/../Transformer_Verification_Code'

def load_tvm_model_figure10():
    dev = tvm.cuda(0)
    # file_names = ['deploy_matmul_2_256_1_1_256_256.so', 'deploy_matmul_256_256_1_1_256_256.so', 'deploy_matmul_256_256_1_15_256_256.so', 'deploy_matmul_256_256_1_16_256_256.so', 'deploy_matmul_256_256_1_18_256_256.so', 'deploy_matmul_256_256_1_20_256_256.so', 'deploy_matmul_256_256_1_29_256_256.so', 'deploy_matmul_256_256_1_7_256_256.so', 'deploy_matmul_256_512_1_15_256_512.so', 'deploy_matmul_256_512_1_16_256_512.so', 'deploy_matmul_256_512_1_18_256_512.so', 'deploy_matmul_256_512_1_20_256_512.so', 'deploy_matmul_256_512_1_29_256_512.so', 'deploy_matmul_256_512_1_7_256_512.so', 'deploy_matmul_512_256_1_15_256_256.so', 'deploy_matmul_512_256_1_16_256_256.so', 'deploy_matmul_512_256_1_18_256_256.so', 'deploy_matmul_512_256_1_20_256_256.so', 'deploy_matmul_512_256_1_29_256_256.so', 'deploy_matmul_512_256_1_7_256_256.so']
    file_names = ['deploy_matmul_128_128_1_2_128_128.so',\
        'deploy_matmul_128_128_1_4_128_128.so',\
        'deploy_matmul_128_128_1_8_128_128.so',\
        'deploy_matmul_128_128_1_16_128_128.so',\
        'deploy_matmul_128_128_1_32_128_128.so',\
        'deploy_matmul_128_128_1_64_128_128.so',\
        'deploy_matmul_128_128_1_128_128_128.so']
    for file_name in file_names:
        if os.path.exists(os.getcwd()+'/../Transformer_Verification_Code/tvm_model/'+file_name):
            lib = tvm.runtime.load_module(os.getcwd()+'/../Transformer_Verification_Code/tvm_model/'+file_name)
            m = graph_executor.GraphModule(lib["default"](dev))
            tvm_model_lut[file_name] = m
            print("tvm model %s loaded"%(file_name))
    return tvm_model_lut

def load_tvm_model_figure11():
    dev = tvm.cuda(0)
    file_names = []
    for i in [64, 128, 256, 384, 512, 640]:
        file_names.append('deploy_matmul_%s_%s_1_16_%s_%s.so'%(i,i,i,i))
    for file_name in file_names:
        if os.path.exists(os.getcwd()+'/../Transformer_Verification_Code/tvm_model/'+file_name):
            lib = tvm.runtime.load_module(os.getcwd()+'/../Transformer_Verification_Code/tvm_model/'+file_name)
            m = graph_executor.GraphModule(lib["default"](dev))
            tvm_model_lut[file_name] = m
            print("tvm model %s loaded"%(file_name))
    return tvm_model_lut

def load_ansor_model_figure10():
    dev = tvm.cuda(0)
    file_names = ['deploy_matmul_128_128_1_2_128_128.so',\
        'deploy_matmul_128_128_1_4_128_128.so',\
        'deploy_matmul_128_128_1_8_128_128.so',\
        'deploy_matmul_128_128_1_16_128_128.so',\
        'deploy_matmul_128_128_1_32_128_128.so',\
        'deploy_matmul_128_128_1_64_128_128.so',\
        'deploy_matmul_128_128_1_128_128_128.so']
    for file_name in file_names:
        if os.path.exists(os.getcwd()+'/../Transformer_Verification_Code/ansor_model/'+file_name):
            lib = tvm.runtime.load_module(os.getcwd()+'/../Transformer_Verification_Code/ansor_model/'+file_name)
            m = graph_executor.GraphModule(lib["default"](dev))
            ansor_model_lut[file_name] = m
            print("ansor model %s loaded"%(file_name))
    return ansor_model_lut

def load_ansor_model_figure11():
    dev = tvm.cuda(0)
    file_names = []
    for i in [64, 128, 256, 384, 512, 640]:
        file_names.append('deploy_matmul_%s_%s_1_16_%s_%s.so'%(i,i,i,i))
    for file_name in file_names:
        if os.path.exists(os.getcwd()+'/../Transformer_Verification_Code/ansor_model/'+file_name):
            lib = tvm.runtime.load_module(os.getcwd()+'/../Transformer_Verification_Code/ansor_model/'+file_name)
            m = graph_executor.GraphModule(lib["default"](dev))
            ansor_model_lut[file_name] = m
            print("ansor model %s loaded"%(file_name))
    return ansor_model_lut


# Please manually set FLAG in ../Transformer_Verification_Code/Verifiers/Bounds.py 
#   for selecting pytorch or hand tuned kernels.

def profile_relu(args, batch_size, length, dim_in, dim_out, num_profile=100):
    w = torch.randn(batch_size, length, dim_in, dim_out).cuda()
    b = torch.randn(batch_size, length, dim_out).cuda()

    bound = Bounds(args, p=2, eps=0.1, w=w, b=b)
    # print(bound.lw)
    # print(bound.lb)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for i in range(num_profile):
        bound = bound.relu()
    end.record()
    # Waits for everything to finish running
    torch.cuda.synchronize()

    average_time = start.elapsed_time(end)/num_profile # Unit: Millisecond
    print("batch_size: {}, length: {}, dim_in: {}, dim_out: {}, average_time (ms): {}".format(batch_size, length, dim_in, dim_out, average_time))

def profile_tanh(args, batch_size, length, dim_in, dim_out, num_profile=100):
    w = torch.randn(batch_size, length, dim_in, dim_out).cuda()
    b = torch.randn(batch_size, length, dim_out).cuda()

    bound = Bounds(args, p=2, eps=0.1, w=w, b=b)
    # print(bound.lw)
    # print(bound.lb)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for i in range(num_profile):
        bound = bound.tanh()
    end.record()
    # Waits for everything to finish running
    torch.cuda.synchronize()

    average_time = start.elapsed_time(end)/num_profile # Unit: Millisecond
    print("batch_size: {}, length: {}, dim_in: {}, dim_out: {}, average_time (ms): {}".format(batch_size, length, dim_in, dim_out, average_time))

def profile_dot_product(args, batch_size, length, dim_in, dim_out, num_profile=1):
    w = torch.randn(batch_size, length, dim_in, dim_out).cuda()
    b = torch.randn(batch_size, length, dim_out).cuda()

    q = Bounds(args, p=2, eps=0.1, w=w, b=b)
    k = Bounds(args, p=2, eps=0.1, w=w, b=b)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for i in range(num_profile):
        s = q.dot_product(k)

    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()

    average_time = start.elapsed_time(end)/num_profile # Unit: Millisecond
    print("batch_size: {}, length: {}, dim_in: {}, dim_out: {}, average_time (ms): {}".format(batch_size, length, dim_in, dim_out, average_time))


def profile_matmul(args, batch_size, length, dim_in, dim_out, dim_y_out, num_profile=100):
    x_w = torch.randn(batch_size, length, dim_in, dim_out).cuda()
    x_b = torch.randn(batch_size, length, dim_out).cuda()
    W = torch.randn(dim_out, dim_y_out).cuda()

    bound = Bounds(args, p=2, eps=0.1, w=x_w, b=x_b)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for i in range(num_profile):
        bound = bound.matmul(W, tvm_model_lut, ansor_model_lut, tvm_model_path, ansor_model_path)
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()

    average_time = start.elapsed_time(end)/num_profile # Unit: Millisecond
    print("batch_size: {}, length: {}, dim_in: {}, dim_out: {}, dim_y_out: {}, average_time (ms): {}".format(batch_size, length, dim_in, dim_out, dim_y_out, average_time))

# Profile the performance for:
#    batch_size = 1
#    length in [2, 4, 8, 16, 32, 64, 128]
#    dim_in in [64, 128, 256, 512, 1024]
#    dim_out in [64, 128, 256, 512, 1024]

def run_figure10():
    tvm_model_lut = load_tvm_model_figure10()
    ansor_model_lut = load_ansor_model_figure10()
    KERNEL_FLAG = 'matmul' # KERNEL_FLAG in ["relu", "matmul", "dot_product"]
    for length in [2,4,8,16,32,64,128]:
        for dim_in in [128]:#[64, 128, 256, 512, 1024]:
            dim_out=dim_in # Just an assumption for profiling
            dim_y_out = dim_out
            profile_matmul(args, batch_size=1, length=length, dim_in=dim_in, dim_out=dim_out, dim_y_out=dim_y_out)

def run_figure11():
    tvm_model_lut = load_tvm_model_figure11()
    ansor_model_lut = load_ansor_model_figure11()
    KERNEL_FLAG = 'matmul' # KERNEL_FLAG in ["relu", "matmul", "dot_product"]
    for length in [16]:#[2,4,8,16,32,64,128]:
        for dim_in in [64, 128, 256, 384, 512, 640]:
            dim_out=dim_in # Just an assumption for profiling
            dim_y_out = dim_out
            profile_matmul(args, batch_size=1, length=length, dim_in=dim_in, dim_out=dim_out, dim_y_out=dim_y_out)

def run_figure12():
    KERNEL_FLAG = 'relu'
    for length in [2,4,8,16,32,64,128]:
        for dim_in in [128]:#[64, 128, 256, 512, 1024]:
            dim_out=dim_in # Just an assumption for profiling
            dim_y_out = dim_out
            profile_relu(args, batch_size=1, length=length, dim_in=dim_in, dim_out=dim_out)

def run_figure13_part1():
    KERNEL_FLAG = 'tanh' # KERNEL_FLAG in ["relu", "matmul", "dot_product"]
    for length in [2,4,8,16,32,64,128]:
        for dim_in in [128]:#[64, 128, 256, 512, 1024]:
            dim_out=dim_in # Just an assumption for profiling
            dim_y_out = dim_out
            profile_tanh(args, batch_size=1, length=length, dim_in=dim_in, dim_out=dim_out)

def run_figure13_part2():
    KERNEL_FLAG = 'dot_product' # KERNEL_FLAG in ["relu", "matmul", "dot_product"]
    for length in [2,4,8,16,32,64,128]:
        for dim_in in [128]:#[64, 128, 256, 512, 1024]:
            dim_out=dim_in # Just an assumption for profiling
            dim_y_out = dim_out
            profile_dot_product(args, batch_size=1, length=length, dim_in=dim_in, dim_out=dim_out)

if args.kernel_idx == 1:
    run_figure10()
elif args.kernel_idx == 2:
    run_figure11()
elif args.kernel_idx == 3:
    run_figure12()
elif args.kernel_idx == 4:
    run_figure13_part1()
elif args.kernel_idx == 5:
    run_figure13_part2()











