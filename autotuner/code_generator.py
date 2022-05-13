# Generate the code from parameter.
import os

# A configuration template for generating CUDA code.
# "%s" are the parameters to be filled for a concrete CUDA implementation.
# Note: Need to adjust this template when profiling different kernels
config_template = """// Auto-generated file. DO NOT MODIFY!
using ThreadblockShape = cutlass::gemm::GemmShape<%s, %s, %s>;
using WarpShape = cutlass::gemm::GemmShape<%s, %s, %s>;
static const int NumStages = %s;"""

# A template for generating CUDA profiling code.
# "%s" are the parameters to be filled for a concrete CUDA implementation.
# Will become more generic in the future.
profiling_template = """
#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>
#include "../../tvm_kernels/cuda_kernel/mma/mma_kernel.h"
#include "matmul_config.h"
// GPU configuration.

void profile_cuda_performance() {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  int batch_size = %s;
  int length = %s;
  int dim_out = %s;
  int dim_y_out = %s;
  int dim_in = %s;

  float *x_lb, *x_ub, *x_lw, *x_uw, *W, *y_lb, *y_ub, *y_lw, *y_uw;

  cudaMalloc(reinterpret_cast<void **>(&x_lb), sizeof(float) * batch_size * length * dim_out);
  cudaMalloc(reinterpret_cast<void **>(&x_ub), sizeof(float) * batch_size * length * dim_out);
  cudaMalloc(reinterpret_cast<void **>(&y_lb), sizeof(float) * batch_size * length * dim_y_out);
  cudaMalloc(reinterpret_cast<void **>(&y_ub), sizeof(float) * batch_size * length * dim_y_out);

  cudaMalloc(reinterpret_cast<void **>(&x_lw), sizeof(float) * batch_size * length * dim_in * dim_out);
  cudaMalloc(reinterpret_cast<void **>(&x_uw), sizeof(float) * batch_size * length * dim_in * dim_out);
  cudaMalloc(reinterpret_cast<void **>(&y_lw), sizeof(float) * batch_size * length * dim_in * dim_y_out);
  cudaMalloc(reinterpret_cast<void **>(&y_uw), sizeof(float) * batch_size * length * dim_in * dim_y_out);

  cudaMalloc(reinterpret_cast<void **>(&W), sizeof(float) * dim_y_out * dim_out);

  // Run ours NUM_PROFILES times and record time.
  float bmma_ms_avg = 0.0f;
  int NUM_PROFILES = 100;
  for(int iter=0; iter<NUM_PROFILES; ++iter){
          float bmma_ms = 0.0f;
          cudaEvent_t bmma_start;
          cudaEvent_t bmma_end;
          cudaEventCreate(&bmma_start);
          cudaEventCreate(&bmma_end);
          cudaEventRecord(bmma_start);
          verify_matmul_fuse_fn<ThreadblockShape, WarpShape, NumStages>(
            batch_size*length, dim_in, dim_y_out, dim_out,
            x_lb, x_ub, W, y_lb, y_ub,
            x_lw, x_uw, y_lw, y_uw
          );
          cudaEventRecord(bmma_end);
          cudaEventSynchronize(bmma_end);
          cudaEventElapsedTime(&bmma_ms, bmma_start, bmma_end);
          cudaEventDestroy(bmma_start);
          cudaEventDestroy(bmma_end);
          bmma_ms_avg += bmma_ms;
  }

  bmma_ms_avg = bmma_ms_avg/(float)NUM_PROFILES;

  printf("%%f", bmma_ms_avg);

  cudaFree(reinterpret_cast<void *>(x_lb));
  cudaFree(reinterpret_cast<void *>(x_ub));
  cudaFree(reinterpret_cast<void *>(y_lb));
  cudaFree(reinterpret_cast<void *>(y_ub));
  cudaFree(reinterpret_cast<void *>(W));
}

int main(int argc, char **argv) {
  profile_cuda_performance();
  return EXIT_SUCCESS;
}
"""

# Generate the CUDA code following a configuration template
# Input:
#   config_template: a string with a few "%s" to be filled.
#   parameter: a vector of positive integers. This instantiates the CUDA configuration.
#           The length of this vector must be exactly the same as the number of "%s" in config_template
#  
def generate_code(template, parameter, file_name = "config.h"):
    if not os.path.isdir("scratch_space"):
        os.mkdir("scratch_space")
    config_file_path = os.path.join("scratch_space", file_name)
    # Clean previous configurations
    try:
        os.remove(config_file_path)
    except OSError:
        pass
    file = open(config_file_path,'w+')
    int_parameter = []
    for i in parameter:
        int_parameter.append(int(i))
    parameter = tuple(int_parameter)
    file.write(template % parameter)

if __name__ == "__main__":
    # Generate configuration code
    generate_code(
        config_template,
        parameter=(128,128,32,64,32,32,3),
        file_name="matmul_config.h",
        )
    
    # Generate profiling code
    generate_code(
        profiling_template,
        parameter=(1, 128, 64, 64, 64),
        file_name="cuda_profiler.cu",
    )