#pragma once
#include <core/Shape.hpp>
#include <functional>
namespace MyTensors::Math::Base{
    
    using MyTensors::Core::Shape;

    class Math{
    public:
        virtual ~Math() = default;
        Math() = default;

        virtual void add(const float* a,  Shape a_shape, const float* b, Shape b_shape, float* out, Shape out_shape) = 0;
        virtual void sub(const float* a,  Shape a_shape, const float* b, Shape b_shape, float* out, Shape out_shape) = 0;
        virtual void scale(const float* a, float* out, float _n, Shape shape) = 0;
        virtual void multiply(const float* a,  Shape a_shape, const float* b, Shape b_shape, float* out, Shape out_shape) = 0;
        virtual void divide(const float* a,  Shape a_shape, const float* b, Shape b_shape, float* out, Shape out_shape) = 0;
        virtual void sqrt(const float* a,  Shape a_shape, float* out, Shape out_shape) = 0;
        virtual void pow(const float* a,  Shape a_shape, int _n, float* out, Shape out_shape) = 0;
        virtual void log(const float* a,  Shape a_shape, float* out, Shape out_shape) = 0;

        //Linear Algebra
        virtual void matmul(
            const float* a , Shape a_shape,  
            const float* b , Shape b_shape,
            float* out, Shape out_shape
        ) = 0;
        
        virtual void im2col2d(
            const float* in, Shape in_shape,
            float* out, Shape out_shape,
            std::array<std::size_t,2> window_shape,
            std::array<std::size_t,2> strides,
            std::array<std::size_t,2> padding
        ) = 0;

        virtual void col2im2d(
            const float* in, Shape in_shape,
            float* out, Shape out_shape,
            std::array<std::size_t,2> window_shape,
            std::array<std::size_t,2> strides,
            std::array<std::size_t,2> padding
        ) = 0;

        //Activation functions
        virtual void reLU(
            const float* a, float* out, size_t n_elements
        ) = 0;
        virtual void sigmoid(const float* a, float* out, size_t n_elements) = 0;
        virtual void tanh(const float* a, float* out, size_t n_elements) = 0;
        
        virtual void binary_positive_mask(
            const float* in,
            const Shape& in_shape,
            float* out,
            const Shape& out_shape
        ) = 0;

        virtual void softmax(
            const float* a,
            Shape a_shape, 
            float* out,
            Shape out_shape
        ) = 0;
        
        //Structural ops and reshaping
        virtual void sum(
            const float* a, Shape a_shape, 
            float* out, Shape out_shape ,
            size_t axis
        ) = 0;

    };
}