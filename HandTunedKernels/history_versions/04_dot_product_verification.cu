/*
Command to compile on Winnie, A6000:
  nvcc -arch=sm_75 -o dot_product_verification dot_product_verification.cu
Command to run on Winnie, A6000:
  ./dot_product_verification
*/

#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>

// GPU configuration.

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

#define SKEW 0

#define checkKernelErrors(expr)                             \
  do {                                                      \
    expr;                                                   \
                                                            \
    cudaError_t __err = cudaGetLastError();                 \
    if (__err != cudaSuccess) {                             \
      printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, \
             cudaGetErrorString(__err));                    \
      abort();                                              \
    }                                                       \
  } while (0)

// using namespace nvcuda;


__global__ void concretize(
    const float *lw, const float *uw, const float *lb, const float *ub, 
    float *l, float *u,
    int length, int dim_out, int dim_in, float p, float epsilon
){
  // Warp and lane identification.
  for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
    const unsigned int length_idx = block_pos / dim_out;
    const unsigned int dim_out_idx = block_pos % dim_out;

    // Stop when there are no more D matrix tiles to compute in this CTA.
    if (length_idx >= length) {
      break;
    }

    int idx = length_idx*dim_out + dim_out_idx;
    float src_lb_val = *(lb + idx);
    float src_ub_val = *(ub + idx);
    int base_idx = length_idx*dim_out*dim_in + dim_out_idx*dim_in;

    // Compute norm
    float square_lw = 0.0f;
    float square_uw = 0.0f;
    float val_lw, val_uw, square_val_lw, square_val_uw;
    for (int i=0; i<dim_in/32; i++) {
      val_lw = *(lw + base_idx + i*32 + threadIdx.x);
      square_val_lw = val_lw*val_lw;
      val_uw = *(uw + base_idx + i*32 + threadIdx.x);
      square_val_uw = val_uw*val_uw;
      for (int offset = 16; offset > 0; offset /= 2) {
        square_val_lw += __shfl_down_sync(FULL_MASK, square_val_lw, offset);
        square_val_uw += __shfl_down_sync(FULL_MASK, square_val_uw, offset);
      }
      if (threadIdx.x == 0) {
        square_lw += square_val_lw;
        square_uw += square_val_uw;
      }
    }

    if (threadIdx.x == 0) {
      *(l+idx) = -epsilon * sqrt(square_lw) + src_lb_val;
      *(u+idx) = epsilon * sqrt(square_uw) + src_ub_val;
      // printf("length_idx: %d, dim_out_idx: %d, square_lw: %f, square_uw: %f, epsilon: %f, src_lb_val: %f, src_ub_val: %f, l_val: %f, u_val: %f\n", length_idx, dim_out_idx, square_lw, square_uw, epsilon, src_lb_val, src_ub_val, *(l+idx), *(u+idx));
    }
    __syncthreads();
  }
  // if (threadIdx.x == 0 and blockIdx.x == 0) {
  //   for (int i=0; i<length; i++) {
  //     for (int j=0; j<dim_out; j++) {
  //       // float lnorm = sqrt(square_lw);
  //       // float unorm = sqrt(square_uw);
  //       // printf("length_idx: %d, dim_out_idx: %f, lnorm: %f, unorm: %f, epsilon: %f, src_lb_val: %f, src_ub_val: %f, l_val: %f, u_val: %f \n", i, j, lnorm, unorm, epsilon, src_lb_val, src_ub_val, l_val, u_val);            
  //       float l_val = *(l+i*dim_out+j);
  //       float u_val = *(u+i*dim_out+j);
  //       printf("length_idx: %d, dim_out_idx: %f, lval: %f, uval: %f\n", i, j, l_val, u_val);            
  //     }
  //   }
  // }
}

// This version of dot_product_b supports only dim_out as a multiplier of 32
__global__ void verify_dot_product_b(
    const float *x_l, const float *y_l, const float *y_u, 
    const float *x_lb, const float *x_ub, const float *y_lb, const float *y_ub,
    float *z_lb, float *z_ub,
    int length, int dim_out
){
  // Warp and lane identification.
  for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
    const unsigned int row_idx = block_pos / length;
    const unsigned int col_idx = block_pos % length;

    // Stop when there are no more D matrix tiles to compute in this CTA.
    if (row_idx >= length) {
      break;
    }

    float z_lb_val = 0.0f;
    float z_ub_val = 0.0f;
    float x_l_val, y_l_val, y_u_val, x_lb_val, x_ub_val, y_lb_val, y_ub_val;

    printf("dim_out: %d\n", dim_out);

    for (int i=0; i<dim_out/32; i++) {
        x_l_val = *(x_l + row_idx*dim_out + i*32 + threadIdx.x);
        y_l_val = *(y_l + row_idx*dim_out + i*32 + threadIdx.x);
        y_u_val = *(y_u + row_idx*dim_out + i*32 + threadIdx.x);
        x_lb_val = *(x_lb + col_idx*dim_out + i*32 + threadIdx.x);
        x_ub_val = *(x_ub + col_idx*dim_out + i*32 + threadIdx.x);
        y_lb_val = *(y_lb + col_idx*dim_out + i*32 + threadIdx.x);
        y_ub_val = *(x_ub + col_idx*dim_out + i*32 + threadIdx.x);

        printf("block_pos: %d, row_idx: %d, col_idx: %d, x_l_val: %f, y_l_val: %f, y_u_val: %f, x_lb_val: %f, x_ub_val: %f, y_lb_val: %f, y_ub_val: %f\n",
                block_pos, row_idx, col_idx, x_l_val, y_l_val, y_u_val, x_lb_val, x_ub_val, y_lb_val, y_ub_val);

        if (y_l_val>0) {
          z_lb_val += y_l_val*x_lb_val;
        } else {
          z_lb_val += y_l_val*x_ub_val;
        }

        if (y_u_val>0) {
          z_ub_val += y_u_val*x_ub_val;
        } else {
          z_ub_val += y_u_val*x_lb_val;
        }

        if (x_l_val>0) {
          z_ub_val += x_l_val*y_ub_val;
          z_lb_val += x_l_val*y_lb_val;
        } else {
          z_ub_val += x_l_val*y_lb_val;
          z_lb_val += x_l_val*y_ub_val;
        }

        z_ub_val -= x_l_val*y_u_val;
        z_lb_val -= x_l_val*y_l_val;
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        z_lb_val += __shfl_down_sync(FULL_MASK, z_lb_val, offset);
        z_ub_val += __shfl_down_sync(FULL_MASK, z_ub_val, offset);
    }

    if (threadIdx.x == 0) {
      printf("At output. block_pos: %d, row_idx: %d, col_idx: %d, z_lb_val: %d, z_ub_val: %d\n", block_pos, row_idx, col_idx, z_lb_val, z_ub_val);
      *(z_lb+row_idx*length+col_idx) = z_lb_val;
      *(z_ub+row_idx*length+col_idx) = z_ub_val;
    }
  }
}

__global__ void verify_dot_product_w(
    const float *x_l, const float *y_l, const float *y_u,
    const float *x_lw, const float *x_uw, const float *y_lw, const float *y_uw,
    float *s_lw, float *s_uw,
    int length, int dim_out, int dim_in
){

  const unsigned int laneId = threadIdx.x;
  // Warp and lane identification.
  for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
    const unsigned int row_idx = block_pos / (length*dim_in/32);
    const unsigned int col_idx = (block_pos / (dim_in/32)) % length;
    const unsigned int m = block_pos % (dim_in / 32);

    // Stop when there are no more D matrix tiles to compute in this CTA.
    if (row_idx >= length) {
      break;
    }

    float s_lw_val = 0.0f;
    float s_uw_val = 0.0f;
    float x_l_val, y_l_val, y_u_val, x_lw_val, x_uw_val, y_lw_val, y_uw_val;
    float alpha_l, alpha_u, beta_l, beta_u;

    for (int k=0; k<dim_out; k++) {
      x_l_val = *(x_l+row_idx*dim_out+k);
      y_l_val = *(y_l+col_idx*dim_out+k);
      y_u_val = *(y_u+col_idx*dim_out+k);

      alpha_l = y_l_val;
      alpha_u = y_u_val;
      beta_l = x_l_val;
      beta_u = x_l_val;

      x_lw_val = *(x_lw+row_idx*dim_in*dim_out+k*dim_in+32*m+laneId);
      x_uw_val = *(x_uw+row_idx*dim_in*dim_out+k*dim_in+32*m+laneId);

      y_lw_val = *(y_lw+col_idx*dim_in*dim_out+k*dim_in+32*m+laneId);
      y_uw_val = *(y_lw+col_idx*dim_in*dim_out+k*dim_in+32*m+laneId);

      if (alpha_l>0) {
        s_lw_val += alpha_l*x_lw_val;
      } else {
        s_lw_val += alpha_l*x_uw_val;
      }

      if (beta_l>0) {
        s_lw_val += beta_l*y_lw_val;
      } else {
        s_lw_val += beta_l*y_uw_val;
      }

      if (alpha_u>0) {
        s_uw_val += alpha_u*x_uw_val;
      } else {
        s_uw_val += alpha_u*x_lw_val;
      }

      if (beta_u>0) {
        s_uw_val += beta_u*y_uw_val;
      } else {
        s_uw_val += beta_u*y_lw_val;
      }
    }

    *(s_lw+row_idx*length*dim_in+col_idx*dim_in+32*m+laneId) = s_lw_val;
    *(s_uw+row_idx*length*dim_in+col_idx*dim_in+32*m+laneId) = s_uw_val;
  }
}


void call_dot_product_verification(
  float *x_l, float *y_l, float *x_u, float *y_u, 
  const float *x_lb, const float *x_ub, const float *y_lb, const float *y_ub,
  const float *x_lw, const float *x_uw, const float *y_lw, const float *y_uw,
  float *z_lb, float *z_ub,float *z_lw, float *z_uw,
  int length, int dim_out, int dim_in, float epsilon
)
{
  cudaDeviceProp deviceProp;
  checkKernelErrors(cudaGetDeviceProperties(&deviceProp, 0));

  concretize<<<16*deviceProp.multiProcessorCount, 32>>>(
    x_lw, x_uw, x_lb, x_ub,
    x_l, x_u,
    length, dim_out, dim_in, 2.0f, epsilon
  );

  concretize<<<16*deviceProp.multiProcessorCount, 32>>>(
    y_lw, y_uw, y_lb, y_ub,
    y_l, y_u,
    length, dim_out, dim_in, 2.0f, epsilon
  );

  verify_dot_product_b<<<16*deviceProp.multiProcessorCount, 32>>>(
    x_l, y_l, y_u,
    x_lb, x_ub, y_lb, y_ub,
    z_lb, z_ub,
    length, dim_out
  );

  verify_dot_product_w<<<16*deviceProp.multiProcessorCount, 32>>>(
    x_l, y_l, y_u,
    x_lw, x_uw, y_lw, y_uw,
    z_lw, z_uw,
    length, dim_out, dim_in
  );
}

int main(int argc, char **argv) {

  cudaDeviceProp deviceProp;
  checkKernelErrors(cudaGetDeviceProperties(&deviceProp, 0));

  for (int length = 2; length <= 128; length*=2) {
   for (int dim_out=64; dim_out <= 1024; dim_out*=2) {
      int dim_in = dim_out;

      float *x_l, *y_l, *y_u, *x_u,
            *x_lb, *x_ub, *y_lb, *y_ub, *z_lb, *z_ub,
            *x_lw, *x_uw, *y_lw, *y_uw, *z_lw, *z_uw;

      checkKernelErrors(
        cudaMalloc(reinterpret_cast<void **>(&x_l), sizeof(float) * length * dim_out));
      checkKernelErrors(
        cudaMalloc(reinterpret_cast<void **>(&x_u), sizeof(float) * length * dim_out));
      checkKernelErrors(
        cudaMalloc(reinterpret_cast<void **>(&y_l), sizeof(float) * length * dim_out));
      checkKernelErrors(
        cudaMalloc(reinterpret_cast<void **>(&y_u), sizeof(float) * length * dim_out));
      checkKernelErrors(
        cudaMalloc(reinterpret_cast<void **>(&x_lb), sizeof(float) * length * dim_out));
      checkKernelErrors(
        cudaMalloc(reinterpret_cast<void **>(&x_ub), sizeof(float) * length * dim_out));
      checkKernelErrors(
        cudaMalloc(reinterpret_cast<void **>(&y_lb), sizeof(float) * length * dim_out));
      checkKernelErrors(
        cudaMalloc(reinterpret_cast<void **>(&y_ub), sizeof(float) * length * dim_out));
      checkKernelErrors(
        cudaMalloc(reinterpret_cast<void **>(&z_lb), sizeof(float) * length * dim_out));
      checkKernelErrors(
        cudaMalloc(reinterpret_cast<void **>(&z_ub), sizeof(float) * length * dim_out));

      checkKernelErrors(
        cudaMalloc(reinterpret_cast<void **>(&x_lw), sizeof(float) * length * dim_in * dim_out));
      checkKernelErrors(
        cudaMalloc(reinterpret_cast<void **>(&x_uw), sizeof(float) * length * dim_in * dim_out));
      checkKernelErrors(
        cudaMalloc(reinterpret_cast<void **>(&y_lw), sizeof(float) * length * dim_in * dim_out));
      checkKernelErrors(
        cudaMalloc(reinterpret_cast<void **>(&y_uw), sizeof(float) * length * dim_in * dim_out));
      checkKernelErrors(
        cudaMalloc(reinterpret_cast<void **>(&z_lw), sizeof(float) * length * dim_in * length));
      checkKernelErrors(
        cudaMalloc(reinterpret_cast<void **>(&z_uw), sizeof(float) * length * dim_in * length));


      // Run ours NUM_PROFILES times and record time.
      float bmma_ms_avg = 0.0f;
      int NUM_PROFILES = 1000;
      for(int iter=0; iter<NUM_PROFILES; ++iter){
              float bmma_ms = 0.0f;
              cudaEvent_t bmma_start;
              cudaEvent_t bmma_end;
              cudaEventCreate(&bmma_start);
              cudaEventCreate(&bmma_end);
              cudaEventRecord(bmma_start);
              checkKernelErrors(
                (concretize<<<16*deviceProp.multiProcessorCount, 32>>>(
                  x_lw, x_uw, x_lb, x_ub,
                  x_l, x_u,
                  length, dim_out, dim_in, 2.0f, 0.05
                )));
              checkKernelErrors(
                (concretize<<<16*deviceProp.multiProcessorCount, 32>>>(
                  y_lw, y_uw, y_lb, y_ub,
                  y_l, y_u,
                  length, dim_out, dim_in, 2.0f, 0.05
                )));
              checkKernelErrors(
                (verify_dot_product_b<<<16*deviceProp.multiProcessorCount, 32>>>(
                  x_l, y_l, y_u,
                  x_lb, x_ub, y_lb, y_ub,
                  z_lb, z_ub,
                  length, dim_out
                )));
              checkKernelErrors(
                (verify_dot_product_w<<<16*deviceProp.multiProcessorCount, 32>>>(
                  x_l, y_l, y_u,
                  x_lw, x_uw, y_lw, y_uw,
                  z_lw, z_uw,
                  length, dim_out, dim_in
                )));
              cudaEventRecord(bmma_end);
              cudaEventSynchronize(bmma_end);
              cudaEventElapsedTime(&bmma_ms, bmma_start, bmma_end);
              cudaEventDestroy(bmma_start);
              cudaEventDestroy(bmma_end);
              bmma_ms_avg += bmma_ms;
      }
    
      bmma_ms_avg = bmma_ms_avg/(float)NUM_PROFILES;
  
      printf("length: %d, dim_out: %d\n", length, dim_out);
      printf("Time: %f ms\n", bmma_ms_avg);  
    
      checkKernelErrors(cudaFree(reinterpret_cast<void *>(x_l)));
      checkKernelErrors(cudaFree(reinterpret_cast<void *>(y_l)));
      checkKernelErrors(cudaFree(reinterpret_cast<void *>(y_u)));
      checkKernelErrors(cudaFree(reinterpret_cast<void *>(x_lb)));
      checkKernelErrors(cudaFree(reinterpret_cast<void *>(x_ub)));
      checkKernelErrors(cudaFree(reinterpret_cast<void *>(y_lb)));
      checkKernelErrors(cudaFree(reinterpret_cast<void *>(y_ub)));
      checkKernelErrors(cudaFree(reinterpret_cast<void *>(z_lb)));
      checkKernelErrors(cudaFree(reinterpret_cast<void *>(z_ub)));
      checkKernelErrors(cudaFree(reinterpret_cast<void *>(x_lw)));
      checkKernelErrors(cudaFree(reinterpret_cast<void *>(x_uw)));
      checkKernelErrors(cudaFree(reinterpret_cast<void *>(y_lw)));
      checkKernelErrors(cudaFree(reinterpret_cast<void *>(y_uw)));
      checkKernelErrors(cudaFree(reinterpret_cast<void *>(z_lw)));
      checkKernelErrors(cudaFree(reinterpret_cast<void *>(z_uw)));
    }
  }
  return EXIT_SUCCESS;
}
