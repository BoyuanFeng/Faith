dot_product:
	nvcc -arch=sm_60 -o dot_product_verification dot_product_verification.cu
	CUDA_VISIBLE_DEVICES=1 ./dot_product_verification

dot_product_artifact:
	nvcc -arch=sm_60 -o dot_product_verification_artifact dot_product_verification_artifact.cu
	CUDA_VISIBLE_DEVICES=1 ./dot_product_verification_artifact

matmul:
	nvcc -arch=sm_80 -I ../cutlass/include -I ../cutlass/tools/util/include -I ../cutlass/examples/common -o matmul_verification matmul_verification.cu
	CUDA_VISIBLE_DEVICES=1 ./matmul_verification

matmul_artifact1:
	nvcc -arch=sm_80 -I ../cutlass/include -I ../cutlass/tools/util/include -I ../cutlass/examples/common -o matmul_verification_artifact1 matmul_verification_artifact1.cu
	CUDA_VISIBLE_DEVICES=1 ./matmul_verification_artifact1

	# nvcc -arch=sm_60 -o matmul_verification_artifact1 matmul_verification_artifact1.cu
	# CUDA_VISIBLE_DEVICES=1 ./matmul_verification_artifact1

matmul_artifact2:
	nvcc -arch=sm_80 -I ../cutlass/include -I ../cutlass/tools/util/include -I ../cutlass/examples/common -o matmul_verification_artifact2 matmul_verification_artifact2.cu
	CUDA_VISIBLE_DEVICES=1 ./matmul_verification_artifact2

	# nvcc -arch=sm_60 -o matmul_verification_artifact2 matmul_verification_artifact2.cu
	# CUDA_VISIBLE_DEVICES=1 ./matmul_verification_artifact2

relu:
	nvcc -arch=sm_60 -o relu_verification relu_verification.cu
	CUDA_VISIBLE_DEVICES=1 ./relu_verification

relu_artifact:
	nvcc -arch=sm_60 -o relu_verification_artifact relu_verification_artifact.cu
	CUDA_VISIBLE_DEVICES=1 ./relu_verification_artifact

tanh:
	nvcc -arch=sm_60 -o tanh_verification tanh_verification.cu
	CUDA_VISIBLE_DEVICES=1 ./tanh_verification

tanh_artifact:
	nvcc -arch=sm_60 -o tanh_verification_artifact tanh_verification_artifact.cu
	CUDA_VISIBLE_DEVICES=1 ./tanh_verification

conv:
	nvcc -arch=sm_60 -o convolution_verification convolution_verification.cu
	CUDA_VISIBLE_DEVICES=1 ./convolution_verification

clean:
	rm -f dot_product_verification matmul_verification relu_verification tanh_verification convolution_verification
