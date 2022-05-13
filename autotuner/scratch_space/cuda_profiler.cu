
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

  int batch_size = 1;
  int length = 128;
  int dim_out = 64;
  int dim_y_out = 64;
  int dim_in = 64;

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

  printf("%f", bmma_ms_avg);

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
