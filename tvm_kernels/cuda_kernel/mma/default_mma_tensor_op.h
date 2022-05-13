#include "mma_tensor_op.h"
namespace cutlass {
namespace gemm {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename WarpShape_,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape_,
    /// Data type of A elements
    typename ElementA_,
    /// Layout of A matrix (concept: MatrixLayout)
    typename LayoutA_,
    /// Data type of B elements
    typename ElementB_,
    /// Layout of B matrix (concept: MatrixLayout)
    typename LayoutB_,
    /// Element type of C matrix
    typename ElementC_,
    /// Layout of C matrix (concept: MatrixLayout)
    typename LayoutC_,
    /// Operator describing the tensor operation
    typename Operator_ = arch::OpMultiplyAdd,
    /// Number of partitions along K dimension
    int PartitionsK = 1,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor = false>
struct DefaultMmaTensorOpVerify;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial Specialization - inputs and output types are float - uses TF32 internally
template <
    /// Shape of one matrix production operation (concept: GemmShape)
    typename WarpShape_,
    /// Shape of target matrix multiply instruction (concept: GemmShape)
    typename InstructionShape_,
    /// Layout of A matrix (concept: MatrixLayout)
    typename LayoutA,
    /// Layout of B matrix (concept: MatrixLayout)
    typename LayoutB,
    /// Layout of C matrix (concept: MatrixLayout)
    typename LayoutC,
    /// Number of partitions along K dimension
    int PartitionsK,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor>
struct DefaultMmaTensorOpVerify<
  WarpShape_, 
  InstructionShape_, 
  float, LayoutA, 
  float, LayoutB, 
  float, LayoutC, 
  arch::OpMultiplyAdd, PartitionsK, AccumulatorsInRowMajor> {

  // Uses TF32 internally
  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
      cutlass::arch::Mma<
        InstructionShape_, 
        32, 
        tfloat32_t, cutlass::layout::RowMajor, 
        tfloat32_t, cutlass::layout::ColumnMajor,
        float, cutlass::layout::RowMajor, 
        arch::OpMultiplyAdd
      >,
      cutlass::MatrixShape<1, 1> >;

  // Define the warp-level tensor op
  using Type = cutlass::gemm::warp::MmaTensorOpVerify<
      WarpShape_, float, LayoutA, float, LayoutB, float, LayoutC,
      Policy, PartitionsK, AccumulatorsInRowMajor>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////