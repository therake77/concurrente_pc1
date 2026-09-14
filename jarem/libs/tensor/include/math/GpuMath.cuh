#pragma once
#include <math/Math.hpp>


namespace MyTensors::Math::Base{
    constexpr std::size_t THREADS_PER_BLOCK = 256;
    constexpr std::size_t TILE_SIZE = 16;


    class GpuMath : public Math{
    public:

        ~GpuMath() override = default;
        GpuMath() = default;

        void add(const float* a,  Shape a_shape, const float* b, Shape b_shape, float* out, Shape out_shape) override;
        void sub(const float* a,  Shape a_shape, const float* b, Shape b_shape, float* out, Shape out_shape) override;
        void scale(const float* a, float* out, float _n, Shape shape) override;
        void multiply(const float* a,  Shape a_shape, const float* b, Shape b_shape, float* out, Shape out_shape) override;
        void divide(const float* a,  Shape a_shape, const float* b, Shape b_shape, float* out, Shape out_shape) override;
        void sqrt(const float* a,  Shape a_shape, float* out, Shape out_shape) override;
        void pow(const float* a,  Shape a_shape, int _n, float* out, Shape out_shape) override;
        void log(const float* a,  Shape a_shape, float* out, Shape out_shape) override;

        void matmul(
            const float* a , Shape a_shape,  
            const float* b , Shape b_shape,
            float* out, Shape out_shape
        ) override;
        
        void im2col2d(
            const float* in, Shape in_shape,
            float* out, Shape out_shape,
            std::array<std::size_t,2> window_shape,
            std::array<std::size_t,2> strides,
            std::array<std::size_t,2> padding
        ) override;

        void col2im2d(
            const float* in, Shape in_shape,
            float* out, Shape out_shape,
            std::array<std::size_t,2> window_shape,
            std::array<std::size_t,2> strides,
            std::array<std::size_t,2> padding
        ) override;
        
        void reLU(const float* a, float* out, size_t n_elements) override;
        void sigmoid(const float* a, float* out, size_t n_elements) override;
        void tanh(const float* a, float* out, size_t n_elements) override;

        void softmax(
            const float* a,
            Shape a_shape, 
            float* out,
            Shape out_shape
        ) override;
        
        void sum(
            const float* a, Shape a_shape, 
            float* out, Shape out_shape ,
            size_t axis
        ) override;
    };
}