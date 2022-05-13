import os
import numpy as np

home_path = os.getcwd()
log_tanh_pytorch = home_path + '/log_figure13/tanh_pytorch.txt'
log_dot_product_pytorch = home_path+'/log_figure13/dot_product_pytorch.txt'
log_tanh_cuda = home_path+'/log_figure13/tanh_cuda.txt'
log_dot_product_cuda = home_path+'/log_figure13/dot_product_cuda.txt'

os.chdir(home_path + '/../HandTunedKernels/')
os.system('make tanh_artifact &> %s'%(log_tanh_cuda))
os.system('make dot_product_artifact &> %s'%(log_dot_product_cuda))

os.chdir(home_path + '/../Benchmark/')
os.system('python pytorch_benchmark.py --dir model --data yelp --kernel_idx 4 &> %s'%(log_tanh_pytorch))
os.system('python pytorch_benchmark.py --dir model --data yelp --kernel_idx 5 &> %s'%(log_dot_product_pytorch))

os.chdir(home_path)
file_tanh_pytorch = open(log_tanh_pytorch)
file_dot_product_pytorch = open(log_dot_product_pytorch)
file_tanh_cuda = open(log_tanh_cuda)
file_dot_product_cuda = open(log_dot_product_cuda)
res_tanh_pytorch = []
res_dot_product_pytorch = []
res_tanh_cuda = []
res_dot_product_cuda = []

for line in file_tanh_pytorch.readlines():
    res_tanh_pytorch.append(float(line.rstrip('\n').split(':')[-1]))

for line in file_dot_product_pytorch.readlines():
    res_dot_product_pytorch.append(float(line.rstrip('\n').split(':')[-1]))

for line in file_tanh_cuda.readlines():
    line_split = line.rstrip('\n').split(' ')
    if line_split[0] == 'Time:':
        res_tanh_cuda.append(float(line_split[1]))

for line in file_dot_product_cuda.readlines():
    line_split = line.rstrip('\n').split(' ')
    if line_split[0] == 'Time:':
        res_dot_product_cuda.append(float(line_split[1]))

np.set_printoptions(precision=2)
print("figure 13 results: ")
print('dot_product speedup:')
print(np.array(res_dot_product_pytorch)/np.array(res_dot_product_cuda))

print('tanh speedup:')
print(np.array(res_tanh_pytorch)/np.array(res_tanh_cuda))


