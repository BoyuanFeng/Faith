NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -std=c++11 -arch=sm_80 -lineinfo -I ../../cutlass/include -I ../../cutlass/tools/util/include -I ../../cutlass/examples/common


##################################################################

## Project file structure ##

# Object file directory
OBJ_DIR = ./bin

## Compile ##
concretize.o : concretize.cu
	@$(NVCC) $(NVCC_FLAGS) -x cu $< -o $@

matmul.o : matmul.cu
	@$(NVCC) $(NVCC_FLAGS) -x cu $< -o $@

matmul_cutlass.o : matmul_cutlass.cu
	@$(NVCC) $(NVCC_FLAGS) -x cu $< -o $@

gemm.o : gemm.cu
	@$(NVCC) $(NVCC_FLAGS) -x cu $< -o $@

matmul_ablation: matmul_ablation.cu
	$(NVCC) $(NVCC_FLAGS) -x cu matmul_ablation.cu -o matmul_ablation