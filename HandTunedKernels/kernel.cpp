// Defines Cuda fuctions
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


void call_relu_verification(
    const float *src_lb, 
    const float *src_ub, 
    const float *src_lw, 
    const float *src_uw,
    float *out_lb, 
    float *out_ub, 
    float *out_lw, 
    float *out_uw,
    int length, 
    int dim_in, 
    int dim_out, 
    float epsilon
);

void call_tanh_verification(
    const float *src_lb, 
    const float *src_ub, 
    const float *src_lw, 
    const float *src_uw,
    float *out_lb, 
    float *out_ub, 
    float *out_lw, 
    float *out_uw,
    int length, 
    int dim_in, 
    int dim_out, 
    float epsilon
);

// void call_matmul_verification(  
//   const float *x_lb, const float *x_ub, 
//   const float *x_lw, const float *x_uw,
//   const float *W,
//   float *y_lb, float *y_ub, float *y_lw, float *y_uw,
//   int batch_size, int length, int dim_in, int dim_out, int dim_y_out
// );

void call_dot_product_verification_QK(
  float *x_l, float *y_l, float *x_u, float *y_u, 
  const float *x_lb, const float *x_ub, const float *y_lb, const float *y_ub,
  const float *x_lw, const float *x_uw, const float *y_lw, const float *y_uw,
  float *z_lb, float *z_ub,float *z_lw, float *z_uw,
  int batch_size, int length, int dim_out, int dim_in, float epsilon
);

void call_dot_product_verification_V(
  float *x_l, float *y_l, float *x_u, float *y_u, 
  const float *x_lb, const float *x_ub, const float *y_lb, const float *y_ub,
  const float *x_lw, const float *x_uw, const float *y_lw, const float *y_uw,
  float *z_lb, float *z_ub,float *z_lw, float *z_uw,
  int batch_size, int length, int dim_out, int dim_in, 
  float epsilon
);

// void call_convolution_verification(  
//   const float *x_lb, const float *x_ub, 
//   const float *x_lw, const float *x_uw,
//   const float *W,
//   float *y_lb, float *y_ub, float *y_lw, float *y_uw,
//   int batch_size, int Height, int Width, int CIN, int COUT, int K, int Stride, int padding_size, int dim_in);

void c_relu_verification(
    const torch::Tensor &src_lb,
    const torch::Tensor &src_ub,
    const torch::Tensor &src_lw,
    const torch::Tensor &src_uw,
    torch::Tensor &out_lb,
    torch::Tensor &out_ub,
    torch::Tensor &out_lw,
    torch::Tensor &out_uw,

    int length,
    int dim_in,
    int dim_out,
    float epsilon
    ) 
{
    call_relu_verification(
    (const float *)src_lb.data_ptr(),
    (const float *)src_ub.data_ptr(),
    (const float *)src_lw.data_ptr(),
    (const float *)src_uw.data_ptr(),
    (float *)out_lb.data_ptr(),
    (float *)out_ub.data_ptr(),
    (float *)out_lw.data_ptr(),
    (float *)out_uw.data_ptr(),
    length,
    dim_in,
    dim_out,
    epsilon
  );
}

void c_tanh_verification(
    const torch::Tensor &src_lb,
    const torch::Tensor &src_ub,
    const torch::Tensor &src_lw,
    const torch::Tensor &src_uw,
    torch::Tensor &out_lb,
    torch::Tensor &out_ub,
    torch::Tensor &out_lw,
    torch::Tensor &out_uw,

    int length,
    int dim_in,
    int dim_out,
    float epsilon
    ) 
{
    call_tanh_verification(
    (const float *)src_lb.data_ptr(),
    (const float *)src_ub.data_ptr(),
    (const float *)src_lw.data_ptr(),
    (const float *)src_uw.data_ptr(),
    (float *)out_lb.data_ptr(),
    (float *)out_ub.data_ptr(),
    (float *)out_lw.data_ptr(),
    (float *)out_uw.data_ptr(),
    length,
    dim_in,
    dim_out,
    epsilon
  );
}

void c_dot_product_verification_QK(
    torch::Tensor &x_l, 
    torch::Tensor &y_l, 
    torch::Tensor &x_u, 
    torch::Tensor &y_u,

    const torch::Tensor &x_lb, 
    const torch::Tensor &x_ub, 
    const torch::Tensor &y_lb, 
    const torch::Tensor &y_ub,

    const torch::Tensor &x_lw, 
    const torch::Tensor &x_uw, 
    const torch::Tensor &y_lw, 
    const torch::Tensor &y_uw,
    torch::Tensor &z_lb, 
    torch::Tensor &z_ub,
    torch::Tensor &z_lw, 
    torch::Tensor &z_uw,
    int batch_size,
    int length, 
    int dim_out, 
    int dim_in,
    float epsilon
)
{
  call_dot_product_verification_QK(
    (float *)x_l.data_ptr(),
    (float *)y_l.data_ptr(),
    (float *)x_u.data_ptr(),
    (float *)y_u.data_ptr(),

    (const float *)x_lb.data_ptr(),
    (const float *)x_ub.data_ptr(),
    (const float *)y_lb.data_ptr(),
    (const float *)y_ub.data_ptr(),

    (const float *)x_lw.data_ptr(),
    (const float *)x_uw.data_ptr(),
    (const float *)y_lw.data_ptr(),
    (const float *)y_uw.data_ptr(),

    (float *)z_lb.data_ptr(),
    (float *)z_ub.data_ptr(),
    (float *)z_lw.data_ptr(),
    (float *)z_uw.data_ptr(),

    batch_size,
    length,
    dim_out,
    dim_in,
    epsilon
  );
}

void c_dot_product_verification_V(
    torch::Tensor &x_l, 
    torch::Tensor &y_l, 
    torch::Tensor &x_u, 
    torch::Tensor &y_u,

    const torch::Tensor &x_lb, 
    const torch::Tensor &x_ub, 
    const torch::Tensor &y_lb, 
    const torch::Tensor &y_ub,

    const torch::Tensor &x_lw, 
    const torch::Tensor &x_uw, 
    const torch::Tensor &y_lw, 
    const torch::Tensor &y_uw,
    torch::Tensor &z_lb, 
    torch::Tensor &z_ub,
    torch::Tensor &z_lw, 
    torch::Tensor &z_uw,
    int batch_size,
    int length, 
    int dim_out, 
    int dim_in,
    float epsilon
)
{
  call_dot_product_verification_V(
    (float *)x_l.data_ptr(),
    (float *)y_l.data_ptr(),
    (float *)x_u.data_ptr(),
    (float *)y_u.data_ptr(),

    (const float *)x_lb.data_ptr(),
    (const float *)x_ub.data_ptr(),
    (const float *)y_lb.data_ptr(),
    (const float *)y_ub.data_ptr(),

    (const float *)x_lw.data_ptr(),
    (const float *)x_uw.data_ptr(),
    (const float *)y_lw.data_ptr(),
    (const float *)y_uw.data_ptr(),

    (float *)z_lb.data_ptr(),
    (float *)z_ub.data_ptr(),
    (float *)z_lw.data_ptr(),
    (float *)z_uw.data_ptr(),

    batch_size,
    length,
    dim_out,
    dim_in,
    epsilon
  );
}

// void c_matmul_verification(  
//   const torch::Tensor &x_lb, const torch::Tensor &x_ub, 
//   const torch::Tensor &x_lw, const torch::Tensor &x_uw,
//   const torch::Tensor &W,
//   torch::Tensor &y_lb, torch::Tensor &y_ub, torch::Tensor &y_lw, torch::Tensor &y_uw,
//   int batch_size, int length, int dim_in, int dim_out, int dim_y_out
// ){
//   call_matmul_verification(
//   (const float *)x_lb.data_ptr(), 
//   (const float *)x_ub.data_ptr(), 
//   (const float *)x_lw.data_ptr(), 
//   (const float *)x_uw.data_ptr(),
//   (const float *)W.data_ptr(),
//   (float *)y_lb.data_ptr(), 
//   (float *)y_ub.data_ptr(), 
//   (float *)y_lw.data_ptr(), 
//   (float *)y_uw.data_ptr(),
//   batch_size,
//   length, 
//   dim_in,
//   dim_out,
//   dim_y_out
//   );
// }

// void c_convolution_verification(  
//   const torch::Tensor &x_lb, const torch::Tensor &x_ub, 
//   const torch::Tensor &x_lw, const torch::Tensor &x_uw,
//   const torch::Tensor &W,
//   torch::Tensor &y_lb, torch::Tensor &y_ub, torch::Tensor &y_lw, torch::Tensor &y_uw,
//   int batch_size, int Height, int Width, int CIN, int COUT, int K, int Stride, int padding_size, int dim_in
// ){
//   call_convolution_verification(
//     (const float *)x_lb.data_ptr(), 
//     (const float *)x_ub.data_ptr(), 
//     (const float *)x_lw.data_ptr(), 
//     (const float *)x_uw.data_ptr(),
//     (const float *)W.data_ptr(),
//     (float *)y_lb.data_ptr(), 
//     (float *)y_ub.data_ptr(), 
//     (float *)y_lw.data_ptr(), 
//     (float *)y_uw.data_ptr(),
//     batch_size,
//     Height,
//     Width,
//     CIN,
//     COUT,
//     K,
//     Stride,
//     padding_size,
//     dim_in
//   );
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("c_relu_verification", &c_relu_verification, "relu_verification (CUDA)");
  m.def("c_tanh_verification", &c_tanh_verification, "tanh_verification (CUDA)");
  m.def("c_dot_product_verification_QK", &c_dot_product_verification_QK, "c_dot_product_verification_QK (CUDA)");
  m.def("c_dot_product_verification_V", &c_dot_product_verification_V, "c_dot_product_verification_V (CUDA)");
  // m.def("c_matmul_verification", &c_matmul_verification, "matmul_verification (CUDA)");
  // m.def("c_convolution_verification", &c_convolution_verification, "convolution_verification (CUDA)");
}