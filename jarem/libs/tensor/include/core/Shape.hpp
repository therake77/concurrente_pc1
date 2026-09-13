#pragma once
#include <array>
#include <cstddef>

namespace MyTensors::Core{
    constexpr std::size_t TENSOR_MAX_DIM = 4;

    /* Define las dimensiones del tensor */
    class Shape{
    private:

        void _trigger_shape_recomposition();

    public:
        /* Número de dimensiones del tensor (desde 1 hasta `TENSOR_MAX_DIM`) */
        std::size_t n_dim;
        /* Total de números (floats) que almacena el tensor */
        std::size_t n_elements;
        std::array<std::size_t,TENSOR_MAX_DIM> _shapes;
        std::array<std::size_t, TENSOR_MAX_DIM> _strides;
        Shape()= delete;
        Shape(std::array<std::size_t, TENSOR_MAX_DIM>);
        const std::size_t& operator[](std::size_t index) const;
        std::size_t& operator[](std::size_t index);
        bool operator==(const Shape&) const;
        bool operator!=(const Shape&) const;

        Shape unsqueeze(std::size_t);
        void transpose();
        void transpose(std::size_t,std::size_t);
        void inner_pad(std::size_t);
        void outer_pad(std::size_t);
        bool is_contiguous();
        static void broadcast_shapes(Shape& a, Shape& b,std::size_t ignore = 0);
    };

};
