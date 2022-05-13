#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "default_mma.h"



/////////////////////////////////////////////////////////////////////////////////
// Implement the cutlass version of the code
/////////////////////////////////////////////////////////////////////////////////

using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;


template<typename ThreadblockShape_, typename WarpShape_, int NumStages_>
struct MatmulVerifyConfig{
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  static const int NumStages = NumStages_;

  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

  using Mma = typename cutlass::gemm::threadblock::DefaultMmaVerify<
    float, cutlass::layout::RowMajor, 128 / cutlass::sizeof_bits<float>::value,
    float, cutlass::layout::ColumnMajor, 128 / cutlass::sizeof_bits<float>::value,
    float, cutlass::layout::RowMajor, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    ThreadblockShape, WarpShape, InstructionShape, NumStages, cutlass::arch::OpMultiplyAdd>::ThreadblockMma;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    float, 128 / cutlass::sizeof_bits<float>::value, float, float,
    cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
      ThreadblockShape, typename Mma::Operator, ThreadblockShape::kK / WarpShape::kK, EpilogueOp,
      EpilogueOp::kCount>::Epilogue;

  union SharedStorage {
      typename Mma::SharedStorage main_loop;
      typename Epilogue::SharedStorage epilogue;
  };
};



template <typename Element, typename _Mma, typename _SharedStorage, typename _Epilogue>
__device__ void cutlassMatmulVerifyBase(
    cutlass::gemm::GemmCoord problem_size,
    cutlass::gemm::GemmCoord grid_tiled_shape,
    typename _Mma::IteratorA::Params params_A_l,
    Element* __restrict__ ptr_A_l, 
    typename _Mma::IteratorA::Params params_A_u,
    Element* __restrict__ ptr_A_u,
    typename _Mma::IteratorB::Params params_B,
    Element* __restrict__ ptr_B,
    typename _Epilogue::OutputTileIterator::Params params_D_l,
    Element* __restrict__ ptr_D_l,
    typename _Epilogue::OutputTileIterator::Params params_D_u,
    Element* __restrict__ ptr_D_u,
    typename _Epilogue::OutputOp::Params output_op_,
    int gemm_k_size, int blockIdx_x_offset=0)
{
    extern __shared__ int SharedStorageBase[];

    _SharedStorage& shared_storage = *reinterpret_cast<_SharedStorage *>(SharedStorageBase);

    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord threadblock_tile_offset = threadblock_swizzle.get_tile_offset(grid_tiled_shape);

    // Early exit if CTA is out of range
    if (grid_tiled_shape.m() <= threadblock_tile_offset.m() - blockIdx_x_offset ||
        grid_tiled_shape.n() <= threadblock_tile_offset.n())
    {
        return;
    }

    // Compute initial location in logical coordinates
    cutlass::MatrixCoord tb_offset_A{
        (threadblock_tile_offset.m() - blockIdx_x_offset) * _Mma::Shape::kM,
        0// threadblock_tile_offset.k() * gemm_k_size
    };

    cutlass::MatrixCoord tb_offset_B{
        0,// threadblock_tile_offset.k() * gemm_k_size,
        threadblock_tile_offset.n() * _Mma::Shape::kN
    };

    // Problem size
    // int problem_size_k = gemm_k_size;
    int problem_size_k = problem_size.k();

    int gemm_k_iterations = (problem_size_k + _Mma::Shape::kK - 1) / _Mma::Shape::kK;

    // Compute position within threadblock
    int thread_idx = threadIdx.x;

    int batch_idx = threadblock_swizzle.get_batch_idx();

    // Construct iterators to A, B, and E operands
    typename _Mma::IteratorA iterator_A_l(
        params_A_l,
        //ref_A.data(),
        ptr_A_l,
        {problem_size.m(), problem_size_k},
        thread_idx,
        tb_offset_A
    );

    typename _Mma::IteratorA iterator_A_u(
      params_A_u,
      ptr_A_u,
      {problem_size.m(), problem_size_k},
      thread_idx,
      tb_offset_A
    );

    typename _Mma::IteratorB iterator_B(
        params_B,
        //ref_B.data(),
        ptr_B,
        {problem_size_k, problem_size.n()},
        thread_idx,
        tb_offset_B
    );

    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compuled as warp-uniform
    int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    int lane_idx = threadIdx.x % 32;

    //
    //  Main loop
    //

    // Construct thread-scoped matrix multiply
    _Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

    typename _Mma::FragmentC accumulators_l;
    typename _Mma::FragmentC accumulators_u;

    accumulators_l.clear();
    accumulators_u.clear();

    if (gemm_k_iterations > 0){
        mma(gemm_k_iterations, accumulators_l, accumulators_u, iterator_A_l, iterator_A_u, iterator_B, accumulators_l, accumulators_u);
    }

    //
    //  Epilogue
    //

    typename _Epilogue::OutputOp output_op(output_op_);

    // (blockIdx.x * TileM, blockIdx.y * TileN)
    cutlass::MatrixCoord threadblock_offset(
        (threadblock_tile_offset.m() - blockIdx_x_offset) * _Mma::Shape::kM,
        threadblock_tile_offset.n() * _Mma::Shape::kN
    );

    int block_idx = (threadblock_tile_offset.m() - blockIdx_x_offset) + threadblock_tile_offset.n() * grid_tiled_shape.m();
    
    typename _Epilogue::OutputTileIterator iterator_D_l(
        params_D_l,
        ptr_D_l,
        problem_size.mn(),
        thread_idx,
        threadblock_offset
    );

    typename _Epilogue::OutputTileIterator iterator_D_u(
      params_D_u,
      ptr_D_u,
      problem_size.mn(),
      thread_idx,
      threadblock_offset
  );
    
    _Epilogue epilogue_l(
        shared_storage.epilogue,
        thread_idx,
        warp_idx,
        lane_idx
    );

    _Epilogue epilogue_u(
      shared_storage.epilogue,
      thread_idx,
      warp_idx,
      lane_idx
  );

    epilogue_l(output_op, iterator_D_l, accumulators_l, iterator_D_l);

    __syncthreads();

    epilogue_u(output_op, iterator_D_u, accumulators_u, iterator_D_u);
}


template <typename Element, typename _Mma, typename _SharedStorage, typename _Epilogue>
__global__ void cutlassMatmulVerify(
    cutlass::gemm::GemmCoord problem_size,
    cutlass::gemm::GemmCoord grid_tiled_shape,
    typename _Mma::IteratorA::Params params_A_l,
    Element* __restrict__ ptr_A_l, 
    typename _Mma::IteratorA::Params params_A_u,
    Element* __restrict__ ptr_A_u,
    typename _Mma::IteratorB::Params params_B,
    Element* __restrict__ ptr_B,
    typename _Epilogue::OutputTileIterator::Params params_D_l,
    Element* __restrict__ ptr_D_l,
    typename _Epilogue::OutputTileIterator::Params params_D_u,
    Element* __restrict__ ptr_D_u,
    typename _Epilogue::OutputOp::Params output_op_,
    int gemm_k_size)
{
    cutlassMatmulVerifyBase<Element, _Mma, _SharedStorage, _Epilogue>(
      problem_size, grid_tiled_shape, params_A_l, ptr_A_l, params_A_u, 
      ptr_A_u, params_B, ptr_B, params_D_l, ptr_D_l, params_D_u, ptr_D_u, output_op_, gemm_k_size, 0);
}


template <typename Element, typename _Mma, typename _SharedStorage, typename _Epilogue>
__global__ void cutlassMatmulVerifyFuse(
    cutlass::gemm::GemmCoord problem_size_w,
    cutlass::gemm::GemmCoord problem_size_b,
    cutlass::gemm::GemmCoord grid_tiled_shape_w,
    cutlass::gemm::GemmCoord grid_tiled_shape_b,
    typename _Mma::IteratorA::Params params_A_l_w,
    Element* __restrict__ ptr_A_l_w, 
    typename _Mma::IteratorA::Params params_A_u_w,
    Element* __restrict__ ptr_A_u_w,
    typename _Mma::IteratorB::Params params_B_w,
    Element* __restrict__ ptr_B_w,
    typename _Epilogue::OutputTileIterator::Params params_D_l_w,
    Element* __restrict__ ptr_D_l_w,
    typename _Epilogue::OutputTileIterator::Params params_D_u_w,
    Element* __restrict__ ptr_D_u_w,
    typename _Mma::IteratorA::Params params_A_l_b,
    Element* __restrict__ ptr_A_l_b, 
    typename _Mma::IteratorA::Params params_A_u_b,
    Element* __restrict__ ptr_A_u_b,
    typename _Mma::IteratorB::Params params_B_b,
    Element* __restrict__ ptr_B_b,
    typename _Epilogue::OutputTileIterator::Params params_D_l_b,
    Element* __restrict__ ptr_D_l_b,
    typename _Epilogue::OutputTileIterator::Params params_D_u_b,
    Element* __restrict__ ptr_D_u_b,
    typename _Epilogue::OutputOp::Params output_op_,
    int gemm_k_size)
{
  if (blockIdx.x < grid_tiled_shape_b.m()){
    cutlassMatmulVerifyBase<Element, _Mma, _SharedStorage, _Epilogue>(
      problem_size_b, grid_tiled_shape_b, params_A_l_b, ptr_A_l_b, params_A_u_b, 
      ptr_A_u_b, params_B_b, ptr_B_b, params_D_l_b, ptr_D_l_b, params_D_u_b, ptr_D_u_b, output_op_, gemm_k_size, 0);
  } else {
    cutlassMatmulVerifyBase<Element, _Mma, _SharedStorage, _Epilogue>(
      problem_size_w, grid_tiled_shape_w, params_A_l_w, ptr_A_l_w, params_A_u_w, 
      ptr_A_u_w, params_B_w, ptr_B_w, params_D_l_w, ptr_D_l_w, params_D_u_w, ptr_D_u_w, output_op_, gemm_k_size, grid_tiled_shape_b.m());
  } 
}

template <typename ThreadblockShape, typename WarpShape, int NumStages>
void verify_matmul_fn(
  int length_dim_in, int dim_Y_out, int dim_out, 
  float* x_l_d, float* x_u_d, float* w_d, float* y_l_d, float* y_u_d)
{
  using Config = MatmulVerifyConfig<ThreadblockShape, WarpShape, NumStages>;

  cutlass::gemm::GemmCoord problem_size(length_dim_in, dim_Y_out, dim_out);

  auto layout_a = cutlass::layout::RowMajor::packed(problem_size.mk());
  auto layout_b = cutlass::layout::ColumnMajor::packed(problem_size.kn());
  auto layout_d = cutlass::layout::RowMajor::packed(problem_size.mn());

  ThreadblockSwizzle threadblock_swizzle;

  cutlass::gemm::GemmCoord grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
    problem_size,
    {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
    1
  );

  dim3 grid = threadblock_swizzle.get_grid_shape(grid_tiled_shape);
  dim3 block(Config::Mma::WarpCount::kCount * 32, 1, 1);

  int smem_size = int(sizeof(typename Config::SharedStorage));

  cudaFuncSetAttribute(cutlassMatmulVerify<float, typename Config::Mma, typename Config::SharedStorage, typename Config::Epilogue>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  cudaFuncSetAttribute(cutlassMatmulVerify<float, typename Config::Mma, typename Config::SharedStorage, typename Config::Epilogue>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

  int gemm_k_size = ((problem_size.k() + Config::Mma::Shape::kK - 1) / Config::Mma::Shape::kK) * Config::Mma::Shape::kK;

  cutlassMatmulVerify<float, typename Config::Mma, typename Config::SharedStorage, typename Config::Epilogue><<<grid, block, smem_size>>>(
    problem_size, grid_tiled_shape,
    layout_a, x_l_d,
    layout_a, x_u_d,
    layout_b, w_d,
    layout_d, y_l_d,
    layout_d, y_u_d,
    {1.0f, 0.0f}, gemm_k_size
  );
}


template <typename ThreadblockShape, typename WarpShape, int NumStages>
void verify_matmul_fuse_fn(
  int length, int dim_in, int dim_Y_out, int dim_out,
  float* x_lb_d, float* x_ub_d, float* w_d, float* y_lb_d, float* y_ub_d,
  float* x_lw_d, float* x_uw_d, float* y_lw_d, float* y_uw_d
){

  using Config = MatmulVerifyConfig<ThreadblockShape, WarpShape, NumStages>;

  cutlass::gemm::GemmCoord problem_size_b(length, dim_Y_out, dim_out);

  auto layout_a_b = cutlass::layout::RowMajor::packed(problem_size_b.mk());
  auto layout_b_b = cutlass::layout::ColumnMajor::packed(problem_size_b.kn());
  auto layout_d_b = cutlass::layout::RowMajor::packed(problem_size_b.mn());

  ThreadblockSwizzle threadblock_swizzle;

  cutlass::gemm::GemmCoord grid_tiled_shape_b = threadblock_swizzle.get_tiled_shape(
    problem_size_b,
    {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
    1
  );

  int smem_size = int(sizeof(typename Config::SharedStorage));

  int gemm_k_size = ((problem_size_b.k() + Config::Mma::Shape::kK - 1) / Config::Mma::Shape::kK) * Config::Mma::Shape::kK;

  cutlass::gemm::GemmCoord problem_size_w(length * dim_in, dim_Y_out, dim_out);

  auto layout_a_w = cutlass::layout::RowMajor::packed(problem_size_w.mk());
  auto layout_b_w = cutlass::layout::ColumnMajor::packed(problem_size_w.kn());
  auto layout_d_w = cutlass::layout::RowMajor::packed(problem_size_w.mn());

  cutlass::gemm::GemmCoord grid_tiled_shape_w = threadblock_swizzle.get_tiled_shape(
    problem_size_w,
    {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
    1
  );

  cutlass::gemm::GemmCoord problem_size_fuse(length * (dim_in + 1), dim_Y_out, dim_out);
  cutlass::gemm::GemmCoord grid_tiled_shape_fuse = threadblock_swizzle.get_tiled_shape(
    problem_size_fuse,
    {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
    1
  );

  dim3 grid_f = threadblock_swizzle.get_grid_shape(grid_tiled_shape_fuse);
  dim3 block_f(Config::Mma::WarpCount::kCount * 32, 1, 1);

  cudaFuncSetAttribute(cutlassMatmulVerifyFuse<float, typename Config::Mma, typename Config::SharedStorage, typename Config::Epilogue>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  cudaFuncSetAttribute(cutlassMatmulVerifyFuse<float, typename Config::Mma, typename Config::SharedStorage, typename Config::Epilogue>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cutlassMatmulVerifyFuse<float, typename Config::Mma, typename Config::SharedStorage, typename Config::Epilogue><<<grid_f, block_f, smem_size>>>(
    problem_size_w, problem_size_b,
    grid_tiled_shape_w, grid_tiled_shape_b,
    layout_a_w, x_lw_d,
    layout_a_w, x_uw_d,
    layout_b_w, w_d,
    layout_d_w, y_lw_d,
    layout_d_w, y_uw_d,
    layout_a_b, x_lb_d,
    layout_a_b, x_ub_d,
    layout_b_b, w_d,
    layout_d_b, y_lb_d,
    layout_d_b, y_ub_d,
    {1.0f, 0.0f}, gemm_k_size
  );
}