#include <core/Tensor.hpp>
#include <functional>

namespace MyTensors::Math::TensorMath{
    using MyTensors::Core::Tensor;
    using MyTensors::Core::Shape;
    using MyTensors::Core::TENSOR_MAX_DIM;

    void add(const Tensor& a, const Tensor&b, Tensor& out);
    void sub(const Tensor& a, const Tensor&b, Tensor& out);
    void scale( const Tensor& a, float n, Tensor& out);
    void multiply(const Tensor& a, const Tensor&b, Tensor& out);
    void divide(const Tensor& a, const Tensor&b, Tensor& out);
    void sqrt(const Tensor& a, Tensor& out);
    void pow(const Tensor& a, const int i, Tensor& out);
    void log(const Tensor& a, Tensor& out);
    

    void matmul(const Tensor& a, const Tensor&b, Tensor& out);
    
    void im2col2D(
        const Tensor& in, 
        std::array<std::size_t,2> window_shape,
        std::array<std::size_t,2> strides,
        std::array<std::size_t,2> padding,
        Tensor& out
    );

    void col2im2D(
        const Tensor& in, 
        std::array<std::size_t,2> window_shape,
        std::array<std::size_t,2> strides,
        std::array<std::size_t,2> padding,
        Tensor& out
    );

    void reLU(const Tensor& a, Tensor& out);
    void sigmoid(const Tensor& a, Tensor& out);
    void tanh(const Tensor& a, Tensor& out);
    void softmax(const Tensor& a, Tensor& out);

    void sum(const Tensor& a, size_t axis, Tensor& out);
    
    void binary_positive_mask(
        const Tensor& in,
        Tensor& out
    );

    //Old, deprecated
    Tensor reshape(const Tensor& a, std::array<size_t, TENSOR_MAX_DIM> new_shape);
    
};