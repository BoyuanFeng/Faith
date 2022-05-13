# Faith-NNVerificationCompiler

In this work, we propose and develop the first compiler for NN verification.

## Environment Setup

To install the required python libraries, please use conda:

`conda env create -f environment.yml`


The following are required for installing TVM and pytorch:

To install TVM, please use the following steps:

`git clone --recursive https://github.com/apache/tvm tvm`

In the directory of tvm, please use the following line. It will create a conda environment with the dependencies specified by the yaml

`conda env create --file conda/build-environment.yaml`

Then, activate the created environment and build necessary packages

`conda activate tvm-build`

`conda install libiconv`

Remember to update `cuda_version` in build_cuda.sh. Please use CUDA-10.2. Remember to add cudnn and cublas into /usr/local/cuda-10.2.
For adding cudnn and cublas, please check the following pages:
http://icl.cs.utk.edu/magma/forum/viewtopic.php?f=2&t=2573
https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html

`sh conda/build_cuda.sh`

`conda install tvm-cu102 -c ./conda/pkg`

Finally, please install pytorch with:
`conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch`
`conda install ipykernel`


## Update Paths
In HandTunedKernels/kernel_test/forward_test_bound.py, please update the following line:

`base_dirc = "/home/boyuan/Faith-NNVerificationCompiler/" # Change this to the path on your machine.`


## Code Structure

- **Benchmark**: A set of profiling code for open-srouce PyTorch implementation for NN verification kernels.

- **Kernels**: Hand-optimized CUDA kernels for NN verification.

- **Transformer_Verification_Code**: Cloned open-source PyTorch implementation from: https://github.com/shizhouxing/Robustness-Verification-for-Transformers


