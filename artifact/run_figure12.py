import os
import numpy as np

home_path = os.getcwd()
log_relu_pytorch = home_path + '/log_figure12/relu_pytorch.txt'
log_relu_cuda = home_path + '/log_figure12/relu_cuda.txt'
log_relu_tvm = home_path + '/log_figure12/relu_tvm.txt'
log_relu_ansor = home_path + '/log_figure12/relu_ansor.txt'

os.chdir(home_path + '/../HandTunedKernels/')
os.system('make relu_artifact &> %s'%(log_relu_cuda))

os.chdir(home_path + '/../Benchmark/')
os.system('python pytorch_benchmark.py --dir model --data yelp --kernel_idx 3 --current_flag 0 &> %s'%(log_relu_pytorch))
os.system('python pytorch_benchmark.py --dir model --data yelp --kernel_idx 3 --current_flag 2 &> %s'%(log_relu_tvm))
# os.system('python pytorch_benchmark.py --dir model --data yelp --kernel_idx 3 --current_flag 3 &> %s'%(log_relu_ansor))

os.chdir(home_path)
file_relu_pytorch = open(log_relu_pytorch)
file_relu_cuda = open(log_relu_cuda)
file_relu_tvm = open(log_relu_tvm)
# file_relu_ansor = open(log_relu_ansor)
res_relu_pytorch = []
res_relu_cuda = []
res_relu_tvm = []
res_relu_ansor = []

for line in file_relu_pytorch.readlines():
    res_relu_pytorch.append(float(line.rstrip('\n').split(':')[-1]))

for line in file_relu_cuda.readlines():
    line_split = line.rstrip('\n').split(' ')
    if line_split[0] == 'Time:':
        res_relu_cuda.append(float(line_split[1]))

# for line in file_relu_tvm.readlines():
#     line_split = line.rstrip('\n').split(':')
#     if line_split[0] == 'batch_size':
#         res_relu_tvm.append(float(line_split[-1]))

# print('pytorch raw:')
# print(np.array(res_relu_pytorch))

# print('tvm raw:')
# print(np.array(res_relu_tvm))
np.set_printoptions(precision=2)

# print("figure 12 results: ")
# print('tvm speedup:')
# print(np.array(res_relu_pytorch)/np.array(res_relu_tvm))

# print('faith raw:')
# print(np.array(res_relu_cuda))

print('faith speedup:')
print(np.array(res_relu_pytorch)/np.array(res_relu_cuda))


