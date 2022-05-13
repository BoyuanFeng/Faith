/*
Command to compile on Winnie, A6000:
  nvcc -arch=sm_75 -o dot_product_verification dot_product_verification.cu
Command to run on Winnie, A6000:
  ./dot_product_verification
Note: Current implementation assumes:
  1) dim_in can be divided by 32.
  2) dim_in is smaller tha 16K. Otherwise a float vector of size dim_in cannot be cached in shared memory.
        This is a reasonable assumption since dim_in is usually less than 2K.
  3) dim_out is a multiplier of 32
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


/*
From the implementation perspective, we can combine batch_size and length into 1 index.

In particular, suppose we have a function 
$$concretize(length, n*dim_in, dim_out)$$
If we want to concretize for a input [batch_size, length, n*dim_in, dim_out], we can simply treat it as [batch_size * length, n*dim_in, dim_out].
*/
__global__ void concretize(
    const float *lw, const float *uw, const float *lb, const float *ub, 
    float *l, float *u,
    int length, int dim_out, int dim_in, float p, float epsilon
){
  if (dim_in % 32 != 0) {
    printf("Error: Require dim_in as multiplier of 32. But the actual size is length: %d, dim_out: %d, dim_in: %d\n", length, dim_out, dim_in);
  }
  // Currently support only $p=2$.
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
    float val_lw, val_uw;
    for (int i=0; i<dim_in/32; i++) {
      val_lw = *(lw + base_idx + i*32 + threadIdx.x);
      square_lw += val_lw*val_lw;
      val_uw = *(uw + base_idx + i*32 + threadIdx.x);
      square_uw += val_uw*val_uw;
    }
    for (int offset = 16; offset > 0; offset /= 2) {
      square_lw += __shfl_down_sync(FULL_MASK, square_lw, offset);
      square_uw += __shfl_down_sync(FULL_MASK, square_uw, offset);
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
  //       printf("length_idx: %d, dim_out_idx: %d, lval: %f, uval: %f\n", i, j, l_val, u_val);            
  //     }
  //   }  
  // }
}

// This version of dot_product_b supports only dim_out as a multiplier of 32
__global__ void verify_dot_product_QK_b(
    const float *x_l, const float *y_l, const float *y_u, 
    const float *x_lb, const float *x_ub, const float *y_lb, const float *y_ub,
    float *z_lb, float *z_ub,
    int batch_size, int length, int dim_out
){
  // Warp and lane identification.
  for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
    const unsigned int batch_idx = block_pos / (length*length);
    const unsigned int row_idx_i = (block_pos%(length*length)) / length;
    const unsigned int col_idx_j = (block_pos%(length*length)) % length;

    // Stop when there are no more D matrix tiles to compute in this CTA.
    if (batch_idx >= batch_size) {
      break;
    }

    float z_lb_val = 0.0f;
    float z_ub_val = 0.0f;
    float x_l_val_row, y_l_val_col, y_u_val_col, x_lb_val_row, x_ub_val_row, y_lb_val_col, y_ub_val_col;

    for (int i=0; i<dim_out/32; i++) {
        x_l_val_row = *(x_l + batch_idx*length*dim_out + row_idx_i*dim_out + i*32 + threadIdx.x);
        y_l_val_col = *(y_l + batch_idx*length*dim_out + col_idx_j*dim_out + i*32 + threadIdx.x);
        y_u_val_col = *(y_u + batch_idx*length*dim_out + col_idx_j*dim_out + i*32 + threadIdx.x);

        x_lb_val_row = *(x_lb + batch_idx*length*dim_out + row_idx_i*dim_out + i*32 + threadIdx.x);
        x_ub_val_row = *(x_ub + batch_idx*length*dim_out + row_idx_i*dim_out + i*32 + threadIdx.x);
        y_lb_val_col = *(y_lb + batch_idx*length*dim_out + col_idx_j*dim_out + i*32 + threadIdx.x);
        y_ub_val_col = *(y_ub + batch_idx*length*dim_out + col_idx_j*dim_out + i*32 + threadIdx.x);

        z_ub_val -= (x_l_val_row*y_u_val_col);
        z_lb_val -= (x_l_val_row*y_l_val_col);

        if (y_l_val_col>0) {
          z_lb_val += y_l_val_col*x_lb_val_row;
        } else {
          z_lb_val += y_l_val_col*x_ub_val_row;
        }

        if (y_u_val_col>0) {
          z_ub_val += y_u_val_col*x_ub_val_row;
        } else {
          z_ub_val += y_u_val_col*x_lb_val_row;
        }

        if (x_l_val_row>0) {
          z_ub_val += x_l_val_row*y_ub_val_col;
          z_lb_val += x_l_val_row*y_lb_val_col;
        } else {
          z_ub_val += x_l_val_row*y_lb_val_col;
          z_lb_val += x_l_val_row*y_ub_val_col;
        }
    }

    int i = dim_out/32;
    if (i*32 + threadIdx.x < dim_out) {
      x_l_val_row = *(x_l + batch_idx*length*dim_out + row_idx_i*dim_out + i*32 + threadIdx.x);
      y_l_val_col = *(y_l + batch_idx*length*dim_out + col_idx_j*dim_out + i*32 + threadIdx.x);
      y_u_val_col = *(y_u + batch_idx*length*dim_out + col_idx_j*dim_out + i*32 + threadIdx.x);

      x_lb_val_row = *(x_lb + batch_idx*length*dim_out + row_idx_i*dim_out + i*32 + threadIdx.x);
      x_ub_val_row = *(x_ub + batch_idx*length*dim_out + row_idx_i*dim_out + i*32 + threadIdx.x);
      y_lb_val_col = *(y_lb + batch_idx*length*dim_out + col_idx_j*dim_out + i*32 + threadIdx.x);
      y_ub_val_col = *(y_ub + batch_idx*length*dim_out + col_idx_j*dim_out + i*32 + threadIdx.x);

      z_ub_val -= (x_l_val_row*y_u_val_col);
      z_lb_val -= (x_l_val_row*y_l_val_col);

      if (y_l_val_col>0) {
        z_lb_val += y_l_val_col*x_lb_val_row;
      } else {
        z_lb_val += y_l_val_col*x_ub_val_row;
      }

      if (y_u_val_col>0) {
        z_ub_val += y_u_val_col*x_ub_val_row;
      } else {
        z_ub_val += y_u_val_col*x_lb_val_row;
      }

      if (x_l_val_row>0) {
        z_ub_val += x_l_val_row*y_ub_val_col;
        z_lb_val += x_l_val_row*y_lb_val_col;
      } else {
        z_ub_val += x_l_val_row*y_lb_val_col;
        z_lb_val += x_l_val_row*y_ub_val_col;
      }
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        z_lb_val += __shfl_down_sync(FULL_MASK, z_lb_val, offset);
        z_ub_val += __shfl_down_sync(FULL_MASK, z_ub_val, offset);
    }

    if (threadIdx.x == 0) {
      *(z_lb+batch_idx*length*length + row_idx_i*length + col_idx_j) = z_lb_val;
      *(z_ub+batch_idx*length*length + row_idx_i*length + col_idx_j) = z_ub_val;
    }
    __syncthreads();
  }
}

__global__ void verify_dot_product_QK_w(
    const float *x_l, const float *y_l, const float *y_u,
    const float *x_lw, const float *x_uw, const float *y_lw, const float *y_uw,
    float *s_lw, float *s_uw,
    int batch_size, int length, int dim_out, int dim_in
){
  // Warp and lane identification.
  for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
    const unsigned int batch_idx = block_pos / (length*length*dim_in/32);
    const unsigned int row_idx_i = (block_pos % (length*length*dim_in/32)) / (length*dim_in/32);
    const unsigned int col_idx_j = (block_pos % (length*dim_in/32)) / (dim_in/32);
    const unsigned int m = block_pos % (dim_in / 32);

    // Stop when there are no more D matrix tiles to compute in this CTA.
    if (batch_idx >= batch_size) {
      break;
    }

    float s_lw_val = 0.0f;
    float s_uw_val = 0.0f;
    float x_l_val, y_l_val, y_u_val, x_lw_val, x_uw_val, y_lw_val, y_uw_val;
    float alpha_l, alpha_u, beta_l, beta_u;

    for (int k=0; k<dim_out; k++) {
      y_u_val = *(y_u+batch_idx*length*dim_out + col_idx_j*dim_out+k);
      y_l_val = *(y_l+batch_idx*length*dim_out + col_idx_j*dim_out+k);
      x_l_val = *(x_l+batch_idx*length*dim_out + row_idx_i*dim_out+k);

      alpha_l = y_l_val;
      alpha_u = y_u_val;
      beta_l = x_l_val;
      beta_u = x_l_val;

      x_lw_val = *(x_lw+batch_idx*length*dim_in*dim_out+row_idx_i*dim_in*dim_out+k*dim_in+32*m+threadIdx.x);
      x_uw_val = *(x_uw+batch_idx*length*dim_in*dim_out+row_idx_i*dim_in*dim_out+k*dim_in+32*m+threadIdx.x);

      y_lw_val = *(y_lw+batch_idx*length*dim_in*dim_out+col_idx_j*dim_in*dim_out+k*dim_in+32*m+threadIdx.x);
      y_uw_val = *(y_uw+batch_idx*length*dim_in*dim_out+col_idx_j*dim_in*dim_out+k*dim_in+32*m+threadIdx.x);

      if (alpha_l>0) {
        s_lw_val += alpha_l*x_lw_val;
      } else {
        s_lw_val += alpha_l*x_uw_val;
      }

      if (alpha_u>0) {
        s_uw_val += alpha_u*x_uw_val;
      } else {
        s_uw_val += alpha_u*x_lw_val;
      }

      if (beta_l>0) {
        s_lw_val += beta_l*y_lw_val;
      } else {
        s_lw_val += beta_l*y_uw_val;
      }

      if (beta_u>0) {
        s_uw_val += beta_u*y_uw_val;
      } else {
        s_uw_val += beta_u*y_lw_val;
      }
    }

    *(s_lw+batch_idx*(length*length*dim_in)+row_idx_i*length*dim_in+col_idx_j*dim_in+32*m+threadIdx.x) = s_lw_val;
    *(s_uw+batch_idx*(length*length*dim_in)+row_idx_i*length*dim_in+col_idx_j*dim_in+32*m+threadIdx.x) = s_uw_val;
    __syncthreads();
  }
}

__global__ void verify_dot_product_V_b(
  const float *x_l, const float *y_l, const float *y_u, 
  const float *x_lb, const float *x_ub, const float *y_lb, const float *y_ub,
  float *z_lb, float *z_ub,
  int batch_size, int length, int dim_out
){
  // Warp and lane identification.
  for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
    const unsigned int batch_idx = block_pos / (length*dim_out);
    const unsigned int row_idx_i = (block_pos%(length*dim_out)) / dim_out;
    const unsigned int col_idx_j = (block_pos%(length*dim_out)) % dim_out;

    // Stop when there are no more D matrix tiles to compute in this CTA.
    if (batch_idx >= batch_size) {
      break;
    }

    float z_lb_val = 0.0f;
    float z_ub_val = 0.0f;
    float x_l_val_row, y_l_val_col, y_u_val_col, x_lb_val_row, x_ub_val_row, y_lb_val_col, y_ub_val_col;

    for (int i=0; i<length/32; i++) {
        x_l_val_row = *(x_l + batch_idx*length*length + row_idx_i*length + i*32 + threadIdx.x);
        y_l_val_col = *(y_l + batch_idx*dim_out*length + col_idx_j*length + i*32 + threadIdx.x);
        y_u_val_col = *(y_u + batch_idx*dim_out*length + col_idx_j*length + i*32 + threadIdx.x);

        x_lb_val_row = *(x_lb + batch_idx*length*length + row_idx_i*length + i*32 + threadIdx.x);
        x_ub_val_row = *(x_ub + batch_idx*length*length + row_idx_i*length + i*32 + threadIdx.x);
        y_lb_val_col = *(y_lb + batch_idx*dim_out*length + col_idx_j*length + i*32 + threadIdx.x);
        y_ub_val_col = *(y_ub + batch_idx*dim_out*length + col_idx_j*length + i*32 + threadIdx.x);

        z_ub_val -= (x_l_val_row*y_u_val_col);
        z_lb_val -= (x_l_val_row*y_l_val_col);

        if (y_l_val_col>0) {
          z_lb_val += y_l_val_col*x_lb_val_row;
        } else {
          z_lb_val += y_l_val_col*x_ub_val_row;
        }

        if (y_u_val_col>0) {
          z_ub_val += y_u_val_col*x_ub_val_row;
        } else {
          z_ub_val += y_u_val_col*x_lb_val_row;
        }

        if (x_l_val_row>0) {
          z_ub_val += x_l_val_row*y_ub_val_col;
          z_lb_val += x_l_val_row*y_lb_val_col;
        } else {
          z_ub_val += x_l_val_row*y_lb_val_col;
          z_lb_val += x_l_val_row*y_ub_val_col;
        }
    }

    // Handle i between [t/32*32, t]
    int i = length/32;
    if (i*32+threadIdx.x < length) {
      x_l_val_row = *(x_l + batch_idx*length*length + row_idx_i*length + i*32 + threadIdx.x);
      y_l_val_col = *(y_l + batch_idx*dim_out*length + col_idx_j*length + i*32 + threadIdx.x);
      y_u_val_col = *(y_u + batch_idx*dim_out*length + col_idx_j*length + i*32 + threadIdx.x);

      x_lb_val_row = *(x_lb + batch_idx*length*length + row_idx_i*length + i*32 + threadIdx.x);
      x_ub_val_row = *(x_ub + batch_idx*length*length + row_idx_i*length + i*32 + threadIdx.x);
      y_lb_val_col = *(y_lb + batch_idx*dim_out*length + col_idx_j*length + i*32 + threadIdx.x);
      y_ub_val_col = *(y_ub + batch_idx*dim_out*length + col_idx_j*length + i*32 + threadIdx.x);

      z_ub_val -= (x_l_val_row*y_u_val_col);
      z_lb_val -= (x_l_val_row*y_l_val_col);

      if (y_l_val_col>0) {
        z_lb_val += y_l_val_col*x_lb_val_row;
      } else {
        z_lb_val += y_l_val_col*x_ub_val_row;
      }

      if (y_u_val_col>0) {
        z_ub_val += y_u_val_col*x_ub_val_row;
      } else {
        z_ub_val += y_u_val_col*x_lb_val_row;
      }

      if (x_l_val_row>0) {
        z_ub_val += x_l_val_row*y_ub_val_col;
        z_lb_val += x_l_val_row*y_lb_val_col;
      } else {
        z_ub_val += x_l_val_row*y_lb_val_col;
        z_lb_val += x_l_val_row*y_ub_val_col;
      }
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        z_lb_val += __shfl_down_sync(FULL_MASK, z_lb_val, offset);
        z_ub_val += __shfl_down_sync(FULL_MASK, z_ub_val, offset);
    }

    if (threadIdx.x == 0) {
      *(z_lb+batch_idx*length*dim_out + row_idx_i*dim_out + col_idx_j) = z_lb_val;
      *(z_ub+batch_idx*length*dim_out + row_idx_i*dim_out + col_idx_j) = z_ub_val;
    }
    __syncthreads();
  }
}

__global__ void verify_dot_product_V_w(
  const float *x_l, const float *y_l, const float *y_u,
  const float *x_lw, const float *x_uw, const float *y_lw, const float *y_uw,
  float *s_lw, float *s_uw,
  int batch_size, int length, int dim_out, int dim_in
){
  if (dim_in % 32 != 0) {
    printf("batch_size: %d, length: %d, dim_out: %d, dim_in: %d\n", batch_size, length, dim_out, dim_in);
  }
  // Warp and lane identification.
  for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
    const unsigned int batch_idx = block_pos / (length*dim_out*dim_in/32);
    const unsigned int row_idx_i = (block_pos % (length*dim_out*dim_in/32)) / (dim_out*dim_in/32);
    const unsigned int col_idx_j = (block_pos % (dim_out*dim_in/32)) / (dim_in/32);
    const unsigned int m = block_pos % (dim_in / 32);

    // Stop when there are no more D matrix tiles to compute in this CTA.
    if (batch_idx >= batch_size) {
      break;
    }

    float s_lw_val = 0.0f;
    float s_uw_val = 0.0f;
    float x_l_val, y_l_val, y_u_val, x_lw_val, x_uw_val, y_lw_val, y_uw_val;
    float alpha_l, alpha_u, beta_l, beta_u;

    for (int k=0; k<length; k++) {
      x_l_val = *(x_l+batch_idx*length*length + row_idx_i*length+k);
      y_u_val = *(y_u+batch_idx*dim_out*length + col_idx_j*length+k);
      y_l_val = *(y_l+batch_idx*dim_out*length + col_idx_j*length+k);

      alpha_l = y_l_val;
      alpha_u = y_u_val;
      beta_l = x_l_val;
      beta_u = x_l_val;

      x_lw_val = *(x_lw+batch_idx*length*dim_in*length+row_idx_i*dim_in*length+k*dim_in+32*m+threadIdx.x);
      x_uw_val = *(x_uw+batch_idx*length*dim_in*length+row_idx_i*dim_in*length+k*dim_in+32*m+threadIdx.x);

      y_lw_val = *(y_lw+batch_idx*dim_out*length*dim_in+col_idx_j*length*dim_in+k*dim_in+32*m+threadIdx.x);
      y_uw_val = *(y_uw+batch_idx*dim_out*length*dim_in+col_idx_j*length*dim_in+k*dim_in+32*m+threadIdx.x);

      if (alpha_l>0) {
        s_lw_val += alpha_l*x_lw_val;
      } else {
        s_lw_val += alpha_l*x_uw_val;
      }

      if (alpha_u>0) {
        s_uw_val += alpha_u*x_uw_val;
      } else {
        s_uw_val += alpha_u*x_lw_val;
      }

      if (beta_l>0) {
        s_lw_val += beta_l*y_lw_val;
      } else {
        s_lw_val += beta_l*y_uw_val;
      }

      if (beta_u>0) {
        s_uw_val += beta_u*y_uw_val;
      } else {
        s_uw_val += beta_u*y_lw_val;
      }
    }

    *(s_lw+batch_idx*(length*dim_out*dim_in)+row_idx_i*dim_out*dim_in+col_idx_j*dim_in+32*m+threadIdx.x) = s_lw_val;
    *(s_uw+batch_idx*(length*dim_out*dim_in)+row_idx_i*dim_out*dim_in+col_idx_j*dim_in+32*m+threadIdx.x) = s_uw_val;
    __syncthreads();
  }
}

void call_dot_product_verification_QK(
  float *x_l, float *y_l, float *x_u, float *y_u, 
  const float *x_lb, const float *x_ub, const float *y_lb, const float *y_ub,
  const float *x_lw, const float *x_uw, const float *y_lw, const float *y_uw,
  float *z_lb, float *z_ub,float *z_lw, float *z_uw,
  int batch_size, int length, int dim_out, int dim_in, 
  float epsilon
)
{
  cudaDeviceProp deviceProp;
  checkKernelErrors(cudaGetDeviceProperties(&deviceProp, 0));

  concretize<<<16*deviceProp.multiProcessorCount, 32>>>(
    x_lw, x_uw, x_lb, x_ub,
    x_l, x_u,
    batch_size*length, dim_out, dim_in, 2.0f, epsilon
  );

  concretize<<<16*deviceProp.multiProcessorCount, 32>>>(
    y_lw, y_uw, y_lb, y_ub,
    y_l, y_u,
    batch_size*length, dim_out, dim_in, 2.0f, epsilon
  );

  verify_dot_product_QK_b<<<16*deviceProp.multiProcessorCount, 32>>>(
    x_l, y_l, y_u,
    x_lb, x_ub, y_lb, y_ub,
    z_lb, z_ub,
    batch_size, length, dim_out
  );

  verify_dot_product_QK_w<<<16*deviceProp.multiProcessorCount, 32>>>(
    x_l, y_l, y_u,
    x_lw, x_uw, y_lw, y_uw,
    z_lw, z_uw,
    batch_size, length, dim_out, dim_in
  );
}


void call_dot_product_verification_V(
  float *x_l, float *y_l, float *x_u, float *y_u, 
  const float *x_lb, const float *x_ub, const float *y_lb, const float *y_ub,
  const float *x_lw, const float *x_uw, const float *y_lw, const float *y_uw,
  float *z_lb, float *z_ub,float *z_lw, float *z_uw,
  int batch_size, int length, int dim_out, int dim_in, 
  float epsilon
)
{
  cudaDeviceProp deviceProp;
  checkKernelErrors(cudaGetDeviceProperties(&deviceProp, 0));

  concretize<<<16*deviceProp.multiProcessorCount, 32>>>(
    x_lw, x_uw, x_lb, x_ub,
    x_l, x_u,
    batch_size*length, length, dim_in, 2.0f, epsilon
  );

  concretize<<<16*deviceProp.multiProcessorCount, 32>>>(
    y_lw, y_uw, y_lb, y_ub,
    y_l, y_u,
    batch_size*dim_out, length, dim_in, 2.0f, epsilon
  );

  verify_dot_product_V_b<<<16*deviceProp.multiProcessorCount, 32>>>(
    x_l, y_l, y_u,
    x_lb, x_ub, y_lb, y_ub,
    z_lb, z_ub,
    batch_size, length, dim_out
  );

  verify_dot_product_V_w<<<16*deviceProp.multiProcessorCount, 32>>>(
    x_l, y_l, y_u,
    x_lw, x_uw, y_lw, y_uw,
    z_lw, z_uw,
    batch_size, length, dim_out, dim_in
  );
}

void test_dot_product_QK() {

  cudaDeviceProp deviceProp;
  checkKernelErrors(cudaGetDeviceProperties(&deviceProp, 0));

  for (int batch_size = 1; batch_size <= 4; batch_size*=2) {
    for (int length = 2; length <= 128; length*=2) {
      for (int dim_out=64; dim_out <= 1024; dim_out*=2) {
  // for (int batch_size = 1; batch_size <= 1; batch_size*=2) {
  //   for (int length = 8; length <= 8; length*=2) {
  //     for (int dim_out=1024; dim_out <= 1024; dim_out*=2) {
        int dim_in = dim_out;
   
        float *x_l, *y_l, *y_u, *x_u,
              *x_lb, *x_ub, *y_lb, *y_ub, *z_lb, *z_ub,
              *x_lw, *x_uw, *y_lw, *y_uw, *z_lw, *z_uw;
    
        checkKernelErrors(
          cudaMalloc(reinterpret_cast<void **>(&x_l), sizeof(float) * batch_size* length * dim_out));
        checkKernelErrors(
          cudaMalloc(reinterpret_cast<void **>(&x_u), sizeof(float) * batch_size* length * dim_out));
        checkKernelErrors(
          cudaMalloc(reinterpret_cast<void **>(&y_l), sizeof(float) * batch_size* length * dim_out));
        checkKernelErrors(
          cudaMalloc(reinterpret_cast<void **>(&y_u), sizeof(float) * batch_size* length * dim_out));
        checkKernelErrors(
          cudaMalloc(reinterpret_cast<void **>(&x_lb), sizeof(float) * batch_size* length * dim_out));
        checkKernelErrors(
          cudaMalloc(reinterpret_cast<void **>(&x_ub), sizeof(float) * batch_size* length * dim_out));
        checkKernelErrors(
          cudaMalloc(reinterpret_cast<void **>(&y_lb), sizeof(float) * batch_size* length * dim_out));
        checkKernelErrors(
          cudaMalloc(reinterpret_cast<void **>(&y_ub), sizeof(float) * batch_size* length * dim_out));
        checkKernelErrors(
          cudaMalloc(reinterpret_cast<void **>(&z_lb), sizeof(float) * batch_size* length * length));
        checkKernelErrors(
          cudaMalloc(reinterpret_cast<void **>(&z_ub), sizeof(float) * batch_size* length * length));
  
        checkKernelErrors(
          cudaMalloc(reinterpret_cast<void **>(&x_lw), sizeof(float) * batch_size* length * dim_in * dim_out));
        checkKernelErrors(
          cudaMalloc(reinterpret_cast<void **>(&x_uw), sizeof(float) * batch_size* length * dim_in * dim_out));
        checkKernelErrors(
          cudaMalloc(reinterpret_cast<void **>(&y_lw), sizeof(float) * batch_size* length * dim_in * dim_out));
        checkKernelErrors(
          cudaMalloc(reinterpret_cast<void **>(&y_uw), sizeof(float) * batch_size* length * dim_in * dim_out));
        checkKernelErrors(
          cudaMalloc(reinterpret_cast<void **>(&z_lw), sizeof(float) * batch_size* length * dim_in * length));
        checkKernelErrors(
          cudaMalloc(reinterpret_cast<void **>(&z_uw), sizeof(float) * batch_size* length * dim_in * length));
  
    
        // Run ours NUM_PROFILES times and record time.
        float bmma_ms_avg = 0.0f;
        int NUM_PROFILES = 50;
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
             batch_size*length, dim_out, dim_in, 2.0f, 0.05
           )));
         checkKernelErrors(
           (concretize<<<16*deviceProp.multiProcessorCount, 32>>>(
             y_lw, y_uw, y_lb, y_ub,
             y_l, y_u,
             batch_size*length, dim_out, dim_in, 2.0f, 0.05
           )));
          checkKernelErrors(
            (verify_dot_product_QK_b<<<16*deviceProp.multiProcessorCount, 32>>>(
              x_l, y_l, y_u,
              x_lb, x_ub, y_lb, y_ub,
              z_lb, z_ub,
              batch_size, length, dim_out
            )));
          checkKernelErrors(
            (verify_dot_product_QK_w<<<16*deviceProp.multiProcessorCount, 32>>>(
              x_l, y_l, y_u,
              x_lw, x_uw, y_lw, y_uw,
              z_lw, z_uw,
              batch_size, length, dim_out, dim_in
          )));
          cudaEventRecord(bmma_end);
          cudaEventSynchronize(bmma_end);
          cudaEventElapsedTime(&bmma_ms, bmma_start, bmma_end);
          cudaEventDestroy(bmma_start);
          cudaEventDestroy(bmma_end);
          bmma_ms_avg += bmma_ms;
        }
      
        bmma_ms_avg = bmma_ms_avg/(float)NUM_PROFILES;
      
        printf("test_dot_product_QK. batch_size: %d, length: %d, dim_out: %d\n", batch_size, length, dim_out);
        printf("Time: %f ms\n", bmma_ms_avg);  
      
        checkKernelErrors(cudaFree(reinterpret_cast<void *>(x_l)));
        checkKernelErrors(cudaFree(reinterpret_cast<void *>(x_u)));
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
  }
}

void test_dot_product_V() {
  cudaDeviceProp deviceProp;
  checkKernelErrors(cudaGetDeviceProperties(&deviceProp, 0));

  for (int batch_size = 1; batch_size <= 4; batch_size*=2) {
    for (int length = 2; length <= 128; length*=2) {
      for (int dim_out=64; dim_out <= 1024; dim_out*=2) {
  // for (int batch_size = 1; batch_size <= 1; batch_size*=2) {
  //   for (int length = 8; length <= 8; length*=2) {
  //     for (int dim_out=1024; dim_out <= 1024; dim_out*=2) {
        int dim_in = dim_out;
   
        float *x_l, *y_l, *y_u, *x_u,
              *x_lb, *x_ub, *y_lb, *y_ub, *z_lb, *z_ub,
              *x_lw, *x_uw, *y_lw, *y_uw, *z_lw, *z_uw;
    
        checkKernelErrors(
          cudaMalloc(reinterpret_cast<void **>(&x_l), sizeof(float) * batch_size* length * length));
        checkKernelErrors(
          cudaMalloc(reinterpret_cast<void **>(&x_u), sizeof(float) * batch_size* length * length));
        checkKernelErrors(
          cudaMalloc(reinterpret_cast<void **>(&y_l), sizeof(float) * batch_size* length * dim_out));
        checkKernelErrors(
          cudaMalloc(reinterpret_cast<void **>(&y_u), sizeof(float) * batch_size* length * dim_out));
        checkKernelErrors(
          cudaMalloc(reinterpret_cast<void **>(&x_lb), sizeof(float) * batch_size* length * length));
        checkKernelErrors(
          cudaMalloc(reinterpret_cast<void **>(&x_ub), sizeof(float) * batch_size* length * length));
        checkKernelErrors(
          cudaMalloc(reinterpret_cast<void **>(&y_lb), sizeof(float) * batch_size* length * dim_out));
        checkKernelErrors(
          cudaMalloc(reinterpret_cast<void **>(&y_ub), sizeof(float) * batch_size* length * dim_out));
        checkKernelErrors(
          cudaMalloc(reinterpret_cast<void **>(&z_lb), sizeof(float) * batch_size* length * dim_out));
        checkKernelErrors(
          cudaMalloc(reinterpret_cast<void **>(&z_ub), sizeof(float) * batch_size* length * dim_out));
  
        checkKernelErrors(
          cudaMalloc(reinterpret_cast<void **>(&x_lw), sizeof(float) * batch_size* length * dim_in * length));
        checkKernelErrors(
          cudaMalloc(reinterpret_cast<void **>(&x_uw), sizeof(float) * batch_size* length * dim_in * length));
        checkKernelErrors(
          cudaMalloc(reinterpret_cast<void **>(&y_lw), sizeof(float) * batch_size* length * dim_in * dim_out));
        checkKernelErrors(
          cudaMalloc(reinterpret_cast<void **>(&y_uw), sizeof(float) * batch_size* length * dim_in * dim_out));
        checkKernelErrors(
          cudaMalloc(reinterpret_cast<void **>(&z_lw), sizeof(float) * batch_size* length * dim_in * dim_out));
        checkKernelErrors(
          cudaMalloc(reinterpret_cast<void **>(&z_uw), sizeof(float) * batch_size* length * dim_in * dim_out));
  
    
        // Run ours NUM_PROFILES times and record time.
        float bmma_ms_avg = 0.0f;
        int NUM_PROFILES = 50;
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
             batch_size*length, length, dim_in, 2.0f, 0.05
           )));
         checkKernelErrors(
           (concretize<<<16*deviceProp.multiProcessorCount, 32>>>(
             y_lw, y_uw, y_lb, y_ub,
             y_l, y_u,
             batch_size*dim_out, length, dim_in, 2.0f, 0.05
           )));
          checkKernelErrors(
            (verify_dot_product_V_b<<<16*deviceProp.multiProcessorCount, 32>>>(
              x_l, y_l, y_u,
              x_lb, x_ub, y_lb, y_ub,
              z_lb, z_ub,
              batch_size, length, dim_out
            )));
          checkKernelErrors(
            (verify_dot_product_V_w<<<16*deviceProp.multiProcessorCount, 32>>>(
              x_l, y_l, y_u,
              x_lw, x_uw, y_lw, y_uw,
              z_lw, z_uw,
              batch_size, length, dim_out, dim_in
          )));
          cudaEventRecord(bmma_end);
          cudaEventSynchronize(bmma_end);
          cudaEventElapsedTime(&bmma_ms, bmma_start, bmma_end);
          cudaEventDestroy(bmma_start);
          cudaEventDestroy(bmma_end);
          bmma_ms_avg += bmma_ms;
        }
      
        bmma_ms_avg = bmma_ms_avg/(float)NUM_PROFILES;
      
        printf("test_dot_product_V. batch_size: %d, length: %d, dim_out: %d\n", batch_size, length, dim_out);
        printf("Time: %f ms\n", bmma_ms_avg);  
      
        checkKernelErrors(cudaFree(reinterpret_cast<void *>(x_l)));
        checkKernelErrors(cudaFree(reinterpret_cast<void *>(x_u)));
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
  }
}


// int main(int argc, char **argv) {
//   test_dot_product_QK();
//   // test_dot_product_V();
//   return EXIT_SUCCESS;
// }
