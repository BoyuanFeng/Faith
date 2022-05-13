import os
import numpy as np

home_path = os.getcwd()
log_matmul_pytorch = home_path + '/log_figure11/matmul_pytorch.txt'
log_matmul_cuda = home_path + '/log_figure11/matmul_cuda.txt'
log_matmul_tvm = home_path + '/log_figure11/matmul_tvm.txt'
log_matmul_ansor = home_path + '/log_figure11/matmul_ansor.txt'

os.chdir(home_path + '/../HandTunedKernels/')
os.system('make matmul_artifact2 &> %s'%(log_matmul_cuda))

os.chdir(home_path + '/../Benchmark/')
os.system('python pytorch_benchmark.py --dir model --data yelp --kernel_idx 2 --current_flag 0 &> %s'%(log_matmul_pytorch))
os.system('python pytorch_benchmark.py --dir model --data yelp --kernel_idx 2 --current_flag 2 &> %s'%(log_matmul_tvm))
os.system('python pytorch_benchmark.py --dir model --data yelp --kernel_idx 2 --current_flag 3 &> %s'%(log_matmul_ansor))

os.chdir(home_path)
file_matmul_pytorch = open(log_matmul_pytorch)
file_matmul_cuda = open(log_matmul_cuda)
file_matmul_tvm = open(log_matmul_tvm)
file_matmul_ansor = open(log_matmul_ansor)
res_matmul_pytorch = []
res_matmul_cuda = []
res_matmul_tvm = []
res_matmul_ansor = []

for line in file_matmul_pytorch.readlines():
    line_split = line.rstrip('\n').split(':')
    if line_split[0] == 'batch_size':
        res_matmul_pytorch.append(float(line.rstrip('\n').split(':')[-1]))

for line in file_matmul_cuda.readlines():
    line_split = line.rstrip('\n').split(' ')
    if line_split[0] == 'Time:':
        res_matmul_cuda.append(float(line_split[1]))

for line in file_matmul_tvm.readlines():
    line_split = line.rstrip('\n').split(':')
    if line_split[0] == 'batch_size':
        res_matmul_tvm.append(float(line_split[-1]))

for line in file_matmul_ansor.readlines():
    line_split = line.rstrip('\n').split(':')
    if line_split[0] == 'batch_size':
        res_matmul_ansor.append(float(line_split[-1]))

np.set_printoptions(precision=2)
print("figure 11 results: ")

print('tvm speedup:')
print(np.array(res_matmul_pytorch)/np.array(res_matmul_tvm))

print('faith speedup:')
print(np.array(res_matmul_pytorch)/np.array(res_matmul_cuda))

print('ansor speedup:')
print(np.array(res_matmul_pytorch)/np.array(res_matmul_ansor))


