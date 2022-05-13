#include "numeric_conversion.h"
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math instructions.
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape_,
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
  /// Policy describing warp-level MmaTensorOp (concept: MmaTensorOp policy)
  typename Policy_,
  /// Number of partitions along K dimension
  int PartitionsK_ = 1,
  /// Store the accumulators in row major or column major.  Row major is used
  /// when output layout is interleaved.
  bool AccumulatorsInRowMajor = false,
  /// Used for partial specialization
  typename Enable = bool
>
class MmaTensorOpVerify {
public:
  /// Shape of warp-level matrix operation (concept: GemmShape)
  using Shape = Shape_;

  /// Data type of multiplicand A
  using ElementA = ElementA_;

  /// Layout of multiplicand A
  using LayoutA = LayoutA_;

  /// Data type of multiplicand B
  using ElementB = ElementB_;

  /// Layout of multiplicand B
  using LayoutB = LayoutB_;

  /// Data type of accumulator matrix C
  using ElementC = ElementC_;

  /// Layout of accumulator matrix C
  using LayoutC = LayoutC_;

  /// Shape of the warp in units of thread (concept: MmaLanePolicySimt)
  using Policy = Policy_;

  /// Underlying matrix multiply operator (concept: arch::Mma)
  using ArchMmaOperator = typename Policy::Operator;

  /// Indicates math operator 
  using MathOperator = typename ArchMmaOperator::Operator;

  /// Architecture tag from underlying instruction
  using ArchTag = typename ArchMmaOperator::ArchTag;

  /// Indicates class of matrix operator
  using OperatorClass = arch::OpClassTensorOp;

  /// Shape of underlying instruction
  using InstructionShape = typename ArchMmaOperator::Shape;

  /// Complex transform on A operand
  static ComplexTransform const kTransformA = ComplexTransform::kNone;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB = ComplexTransform::kNone;

  /// Number of threads participating in warp-level matrix product
  static int const kThreadCount = 32;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK_;

public:

  /// Iterates over the A operand in memory
  using IteratorA = MmaTensorOpMultiplicandTileIterator<
     MatrixShape<Shape::kM, Shape::kK>, Operand::kA, ElementA, LayoutA,
     MatrixShape<ArchMmaOperator::Shape::kM, ArchMmaOperator::Shape::kK>,
     Policy::OpDelta::kRow, kThreadCount, kPartitionsK>;

  /// Storage for A tile
  using FragmentA = typename IteratorA::Fragment;

  /// Storage for transformed A tile
  using TransformedFragmentA =
      Array<typename ArchMmaOperator::ElementA, FragmentA::kElements>;

  /// Iterates over the B operand in memory
  using IteratorB = MmaTensorOpMultiplicandTileIterator<
      MatrixShape<Shape::kK, Shape::kN>, Operand::kB, ElementB, LayoutB,
      MatrixShape<ArchMmaOperator::Shape::kK, ArchMmaOperator::Shape::kN>,
      Policy::OpDelta::kRow, kThreadCount, kPartitionsK>;

  /// Storage for B tile
  using FragmentB = typename IteratorB::Fragment;

  /// Storage for transformed B tile
  using TransformedFragmentB =
      Array<typename ArchMmaOperator::ElementB, FragmentB::kElements>;

  /// Iterates over the C operand in memory
  using IteratorC = MmaTensorOpAccumulatorTileIterator<
     MatrixShape<Shape::kM, Shape::kN>, ElementC, LayoutC,
     typename ArchMmaOperator::Shape, typename Policy::OpDelta>;

  /// Storage for C tile
  using FragmentC = typename IteratorC::Fragment;

  /// Number of mma operations performed
  using MmaIterations = MatrixShape<
    (Shape::kM + ArchMmaOperator::Shape::kM - 1) / ArchMmaOperator::Shape::kM,
    (Shape::kN + ArchMmaOperator::Shape::kN - 1) / ArchMmaOperator::Shape::kN
  >;

public:

  /// Underlying matrix multiply operator (concept: arch::Mma)
  ArchMmaOperator mma;

public:

  //
  // Methods
  //

  /// Ctor
  CUTLASS_DEVICE
  MmaTensorOpVerify() {}

  /// Performs a warp-level matrix multiply-accumulate operation
  CUTLASS_DEVICE
  void operator()(
    FragmentC &D_l,
    FragmentC &D_u, 
    TransformedFragmentA const &A_l,
    TransformedFragmentA const &A_u, 
    TransformedFragmentB const &B_p,
    TransformedFragmentB const &B_n, 
    FragmentC const &C_l,
    FragmentC const &C_u
  ) const {

    using MmaOperandA = typename ArchMmaOperator::FragmentA;
    using MmaOperandB = typename ArchMmaOperator::FragmentB;
    using MmaOperandC = typename ArchMmaOperator::FragmentC;

    D_l = C_l;
    D_u = C_u;

    MmaOperandA const *ptr_A_l = reinterpret_cast<MmaOperandA const *>(&A_l);
    MmaOperandA const *ptr_A_u = reinterpret_cast<MmaOperandA const *>(&A_u);
    MmaOperandB const *ptr_B_p = reinterpret_cast<MmaOperandB const *>(&B_p);
    MmaOperandB const *ptr_B_n = reinterpret_cast<MmaOperandB const *>(&B_n);
    MmaOperandC *ptr_D_l = reinterpret_cast<MmaOperandC *>(&D_l);
    MmaOperandC *ptr_D_u = reinterpret_cast<MmaOperandC *>(&D_u);

    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
      // Serpentine visitation order maximizing reuse of Rb
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < MmaIterations::kColumn; ++n) {

        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < MmaIterations::kRow; ++m) {

          int m_serpentine = ((n % 2) ? (MmaIterations::kRow - 1 - m) : m);

          if (AccumulatorsInRowMajor) {  // matrix B is reordered
            mma(
              ptr_D_l[n + m_serpentine * MmaIterations::kColumn],
              ptr_A_l[m_serpentine],
              ptr_B_p[n],
              ptr_D_l[n + m_serpentine * MmaIterations::kColumn]);
            mma(
              ptr_D_l[n + m_serpentine * MmaIterations::kColumn],
              ptr_A_u[m_serpentine],
              ptr_B_n[n],
              ptr_D_l[n + m_serpentine * MmaIterations::kColumn]);
            mma(
              ptr_D_u[n + m_serpentine * MmaIterations::kColumn],
              ptr_A_u[m_serpentine],
              ptr_B_p[n],
              ptr_D_u[n + m_serpentine * MmaIterations::kColumn]);
            mma(
              ptr_D_u[n + m_serpentine * MmaIterations::kColumn],
              ptr_A_l[m_serpentine],
              ptr_B_n[n],
              ptr_D_u[n + m_serpentine * MmaIterations::kColumn]);
          } else {
            mma(
              ptr_D_l[m_serpentine + n * MmaIterations::kRow],
              ptr_A_l[m_serpentine],
              ptr_B_p[n],
              ptr_D_l[m_serpentine + n * MmaIterations::kRow]);
            mma(
              ptr_D_l[m_serpentine + n * MmaIterations::kRow],
              ptr_A_u[m_serpentine],
              ptr_B_n[n],
              ptr_D_l[m_serpentine + n * MmaIterations::kRow]);
            mma(
              ptr_D_u[m_serpentine + n * MmaIterations::kRow],
              ptr_A_u[m_serpentine],
              ptr_B_p[n],
              ptr_D_u[m_serpentine + n * MmaIterations::kRow]);
            mma(
              ptr_D_u[m_serpentine + n * MmaIterations::kRow],
              ptr_A_l[m_serpentine],
              ptr_B_n[n],
              ptr_D_u[m_serpentine + n * MmaIterations::kRow]);
          }
        }
      }
    #elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
      // Serpentine visitation order maximizing reuse of Ra
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < MmaIterations::kRow; ++m) {

        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < MmaIterations::kColumn; ++n) {

          int n_serpentine = ((m % 2) ? (MmaIterations::kColumn - 1 - n) : n);

          if (AccumulatorsInRowMajor) {  // matrix B is reordered
            mma(
              ptr_D_l[n_serpentine + m * MmaIterations::kColumn],
              ptr_A_l[m],
              ptr_B_p[n_serpentine],
              ptr_D_l[n_serpentine + m * MmaIterations::kColumn]);
            mma(
              ptr_D_l[n_serpentine + m * MmaIterations::kColumn],
              ptr_A_u[m],
              ptr_B_n[n_serpentine],
              ptr_D_l[n_serpentine + m * MmaIterations::kColumn]);
            mma(
              ptr_D_u[n_serpentine + m * MmaIterations::kColumn],
              ptr_A_u[m],
              ptr_B_p[n_serpentine],
              ptr_D_u[n_serpentine + m * MmaIterations::kColumn]);
            mma(
              ptr_D_u[n_serpentine + m * MmaIterations::kColumn],
              ptr_A_l[m],
              ptr_B_n[n_serpentine],
              ptr_D_u[n_serpentine + m * MmaIterations::kColumn]);
          } else {
            mma(ptr_D_l[m + n_serpentine * MmaIterations::kRow],
                ptr_A_l[m],
                ptr_B_p[n_serpentine],
                ptr_D_l[m + n_serpentine * MmaIterations::kRow]);
            mma(ptr_D_l[m + n_serpentine * MmaIterations::kRow],
                ptr_A_u[m],
                ptr_B_n[n_serpentine],
                ptr_D_l[m + n_serpentine * MmaIterations::kRow]);
            mma(ptr_D_u[m + n_serpentine * MmaIterations::kRow],
                ptr_A_u[m],
                ptr_B_p[n_serpentine],
                ptr_D_u[m + n_serpentine * MmaIterations::kRow]);
            mma(ptr_D_u[m + n_serpentine * MmaIterations::kRow],
                ptr_A_l[m],
                ptr_B_n[n_serpentine],
                ptr_D_u[m + n_serpentine * MmaIterations::kRow]);
          }
        }
      }
    #else
      assert(0);
    #endif
  }

  /// Transform the mma operands to the required types
  CUTLASS_DEVICE
  void transform(TransformedFragmentA &dst_A_l, TransformedFragmentA &dst_A_u,
                 TransformedFragmentB &dst_B_p,
                 TransformedFragmentB &dst_B_n,
                 FragmentA const &A_l, FragmentA const &A_u,
                 FragmentB const &B) const {

    //
    // Define conversions from source type to instruction type
    //
    FloatRoundStyle const kRoundA =
        PreferredRoundingMode<typename ArchMmaOperator::ElementA,
                              ElementA>::kRound;
    FloatRoundStyle const kRoundB =
        PreferredRoundingMode<typename ArchMmaOperator::ElementB,
                              ElementB>::kRound;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
      detail::ConvertAndPack<typename ArchMmaOperator::ElementA, ElementA,
                            FragmentA::kElements, kRoundA>
          convert_A;
      NumericArrayConverter<typename ArchMmaOperator::ElementB, ElementB,
                            FragmentB::kElements / 2, kRoundB>
          convert_B;
      Array<ElementB, FragmentB::kElements / 2> const *ptr_B =
          reinterpret_cast<Array<ElementB, FragmentB::kElements / 2> const *>(&B);
      Array<typename ArchMmaOperator::ElementB, FragmentB::kElements / 2> *
          ptr_dst_B_l = reinterpret_cast<Array<typename ArchMmaOperator::ElementB,
                                             FragmentB::kElements / 2> *>(&dst_B_l);
      Array<typename ArchMmaOperator::ElementB, FragmentB::kElements / 2> *
          ptr_dst_B_u = reinterpret_cast<Array<typename ArchMmaOperator::ElementB,
                                             FragmentB::kElements / 2> *>(&dst_B_u);
  
      dst_A = convert_A(A);
  
      ptr_dst_B[0] = convert_B(ptr_B[0]);
      ptr_dst_B[1] = convert_B(ptr_B[1]);

    #elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
      detail::ConvertAndPack<typename ArchMmaOperator::ElementA, ElementA,
                            FragmentA::kElements / 2, kRoundA>
          convert_A;
      NumericBoundArrayConverter<typename ArchMmaOperator::ElementB, ElementB,
                            FragmentB::kElements, true, kRoundB>
          convert_B_pos;
      NumericBoundArrayConverter<typename ArchMmaOperator::ElementB, ElementB,
                            FragmentB::kElements, false, kRoundB>
          convert_B_neg;
      Array<ElementA, FragmentA::kElements / 2> const *ptr_A_l =
          reinterpret_cast<Array<ElementA, FragmentA::kElements / 2> const *>(&A_l);
      Array<typename ArchMmaOperator::ElementA, FragmentA::kElements / 2> *
          ptr_dst_A_l = reinterpret_cast<Array<typename ArchMmaOperator::ElementA,
                                             FragmentA::kElements / 2> *>(&dst_A_l);
      Array<ElementA, FragmentA::kElements / 2> const *ptr_A_u =
          reinterpret_cast<Array<ElementA, FragmentA::kElements / 2> const *>(&A_u);
      Array<typename ArchMmaOperator::ElementA, FragmentA::kElements / 2> *
          ptr_dst_A_u = reinterpret_cast<Array<typename ArchMmaOperator::ElementA,
                                             FragmentA::kElements / 2> *>(&dst_A_u);
  
      dst_B_p = convert_B_pos(B);
      dst_B_n = convert_B_neg(B);
  
      ptr_dst_A_l[0] = convert_A(ptr_A_l[0]);
      ptr_dst_A_l[1] = convert_A(ptr_A_l[1]);
      ptr_dst_A_u[0] = convert_A(ptr_A_u[0]);
      ptr_dst_A_u[1] = convert_A(ptr_A_u[1]);
    #else
      assert(0);
    #endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////