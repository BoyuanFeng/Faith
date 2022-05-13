# """ Convolution, pooling and padding operators"""
# import sys
# sys.path.append('/home/boyuan/verification/Faith-NNVerificationCompiler/auto_LiRPA')
# from auto_LiRPA_modified import BoundConv

# import torch

# attr = {'dilations': [1, 1], 'group': 1, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [2, 2]}
# options = {'ibp_relative': False, 'conv_mode': 'patches', 'sparse_intermediate_bounds': True, 'sparse_conv_intermediate_bounds': True, 'final_shape': torch.Size([1, 10])}
# conv_bound = BoundConv(input_name = "", name = "", ori_name = "", attr = attr, inputs = [1,2,3], output_idx = 1, options = options, device='cuda:0')

# dim_in = 128
# CIN = 16
# COUT = CIN
# Width = 16
# Height = Width
# Stride = 2
# padding_size = 1
# K = 3
# dim_in = 64
# batch_size = 1

# image = torch.ones((batch_size, CIN, Height, Width))


# x = ()
# conv_bound.bound_forward()

