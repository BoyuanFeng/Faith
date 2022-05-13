namespace cutlass {

    /// Conversion operator for Array
template <
  typename T,
  typename S,
  int N,
  bool is_positive,
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest,
  typename Transform = cutlass::transform::thread::UnaryTransform::Identity
>
struct NumericBoundArrayConverter {

  using result_type = Array<T, N>;
  using source_type = Array<S, N>;
  static FloatRoundStyle const round_style = Round;

  static_assert(platform::is_same<Transform, cutlass::transform::thread::UnaryTransform::Identity>::value ||
                platform::is_same<Transform, cutlass::transform::thread::UnaryTransform::Conjugate>::value,
                  "Unary Operator not supported.");

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    result_type result;
    NumericConverter<T, S, Round> convert_;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      if( platform::is_same<Transform, cutlass::transform::thread::UnaryTransform::Identity>::value )
      {
        if (is_positive){
            if (s[i] >= 0){
                result[i] = convert_(s[i]);
            } else {
                result[i] = convert_(0.0f);
            }
        } else {
            if (s[i] <= 0){
                result[i] = convert_(s[i]);
            } else {
                result[i] = convert_(0.0f);
            }
        }
      } else { // conjugate
        result[i] = conj(convert_(s[i]));
      }
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
