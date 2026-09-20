#include <stdexcept>
#include <cmath>
#include <math/CpuMath.hpp>
#include <vector>
#include <cassert>
#include <algorithm>
#include <string>
#include <limits>
#include <cstring>

namespace{
    using namespace MyTensors::Core;
    void _standard_2d_matrix_mul(
        const float* a, const float* b, float* out, 
        size_t M, size_t N, size_t K,
        size_t stride_am, size_t stride_ak,
        size_t stride_bk, size_t stride_bn,
        size_t stride_outm, size_t stride_outn
    ){
        for(size_t m = 0; m < M; ++m){
            for(size_t n = 0; n < N; ++n){
                float sum = 0.0f;
                for(size_t k = 0; k < K; ++k){
                    size_t a_idx = (m * stride_am) + (k * stride_ak);
                    size_t b_idx = (k * stride_bk) + (n * stride_bn);
                    sum += a[a_idx]*b[b_idx];
                }
                size_t out_idx = (m * stride_outm) + (n * stride_outn);
                out[out_idx] = sum;
            }
        }
    }

    bool _is_contiguous_2d(const Shape& shape){
        return shape.n_dim == 2 &&
            shape._strides[0] == shape[1] &&
            shape._strides[1] == 1;
    }

    bool _is_contiguous(const Shape& shape){
        size_t expected_stride = 1;
        for(size_t i = shape.n_dim; i-- > 0; ){
            if(shape._strides[i] != expected_stride){
                return false;
            }
            expected_stride *= shape[i];
        }
        return true;
    }

    bool _is_channel_broadcast_4d(const Shape& shape){
        return shape.n_dim == 4 &&
            shape._strides[0] == 0 &&
            shape._strides[1] == 1 &&
            shape._strides[2] == 0 &&
            shape._strides[3] == 0;
    }

    template<typename Op>
    bool _try_binary_fast_path(
        const float* a, Shape a_shape,
        const float* b, Shape b_shape,
        float* out, Shape out_shape,
        Op op
    ){
        if(
            a_shape._shapes == out_shape._shapes &&
            b_shape._shapes == out_shape._shapes &&
            _is_contiguous(a_shape) &&
            _is_contiguous(b_shape) &&
            _is_contiguous(out_shape)
        ){
            for(size_t i = 0; i < out_shape.n_elements; ++i){
                out[i] = op(a[i], b[i]);
            }
            return true;
        }

        if(out_shape.n_dim == 4 && _is_contiguous(out_shape)){
            size_t N = out_shape[0];
            size_t C = out_shape[1];
            size_t HW = out_shape[2] * out_shape[3];

            if(
                a_shape._shapes == out_shape._shapes &&
                b_shape._shapes == out_shape._shapes &&
                _is_contiguous(a_shape) &&
                _is_channel_broadcast_4d(b_shape)
            ){
                for(size_t n = 0; n < N; ++n){
                    for(size_t c = 0; c < C; ++c){
                        float b_value = b[c];
                        size_t base = (n * C + c) * HW;
                        for(size_t hw = 0; hw < HW; ++hw){
                            out[base + hw] = op(a[base + hw], b_value);
                        }
                    }
                }
                return true;
            }

            if(
                a_shape._shapes == out_shape._shapes &&
                b_shape._shapes == out_shape._shapes &&
                _is_channel_broadcast_4d(a_shape) &&
                _is_contiguous(b_shape)
            ){
                for(size_t n = 0; n < N; ++n){
                    for(size_t c = 0; c < C; ++c){
                        float a_value = a[c];
                        size_t base = (n * C + c) * HW;
                        for(size_t hw = 0; hw < HW; ++hw){
                            out[base + hw] = op(a_value, b[base + hw]);
                        }
                    }
                }
                return true;
            }
        }

        return false;
    }

    void _contiguous_2d_matrix_mul(
        const float* a, const float* b, float* out,
        size_t M, size_t N, size_t K
    ){
        std::fill(out, out + M * N, 0.0f);
        for(size_t m = 0; m < M; ++m){
            float* out_row = out + m * N;
            const float* a_row = a + m * K;
            for(size_t k = 0; k < K; ++k){
                const float a_value = a_row[k];
                const float* b_row = b + k * N;
                for(size_t n = 0; n < N; ++n){
                    out_row[n] += a_value * b_row[n];
                }
            }
        }
    }

    bool _increment_odometer(std::vector<size_t>& coords, std::array<size_t,TENSOR_MAX_DIM> shape){
        for(size_t i = coords.size(); i --> 0 ;){
            //guard against overflow
            if(coords[i] == std::numeric_limits<std::size_t>::max()){ throw std::runtime_error("Error: Overflow detected"); }
            //try incrementing it
            coords[i]++;
            //If the increment don't exceed the current dimension, then just break (returning true)
            if(coords[i] < shape[i]){
                return true;
            }else{ //Else, go back the coordinate to zero
                coords[i] = 0;
            }
            //And in the next loop, the next dimension would be incremented
        }
        //If the loop doesn't break and finishes, that means all coords are zero and the odometer has "overflowed". 
        // There is no element next.
        return false;
    }

    std::size_t _compute_idx_from_coords(std::vector<size_t>& coords, std::array<size_t,TENSOR_MAX_DIM> strides ){
        std::size_t offset = 0;
        for(size_t i = 0; i < coords.size();i ++){
            offset += (coords[i] * strides[i]);
        }
        return offset;
    }

};

using namespace MyTensors::Math::Base;

/*
    @brief Adds two tensors
    @param a First operand
    @param a_shape Shape of first operand
    @param b Second operand
    @param b_shape Shape of second operand
    @param out Output tensor
    @param out_shape Shape of output tensor
    @note The method is specially designed to support broadcasted tensors, using the provided shape strides
*/
void CpuMath::add(
    const float* a,  Shape a_shape, 
    const float* b, Shape b_shape, 
    float* out, Shape out_shape
){
    if(_try_binary_fast_path(a, a_shape, b, b_shape, out, out_shape, [](float x, float y){ return x + y; })){
        return;
    }

    // All shapes are equal, and have the same dimensionality
    auto n_dim = a_shape.n_dim;
    auto shapes = a_shape._shapes;
    std::vector<std::size_t> coords(n_dim,0);
    do{
        std::size_t a_idx = _compute_idx_from_coords(coords,a_shape._strides);
        std::size_t b_idx = _compute_idx_from_coords(coords,b_shape._strides);
        std::size_t out_idx = _compute_idx_from_coords(coords,out_shape._strides);

        out[out_idx] = a[a_idx] + b[b_idx]; 
    }while(_increment_odometer(coords,shapes));
}

/*
    @brief Subtracts two tensors
    @param a Pointer to first tensor operand data
    @param a_shape Shape of first operand
    @param b Pointer to second tensor operand data
    @param b_shape Shape of second operand
    @param out Pointer to output tensor data
    @param out_shape Shape of output tensor
    @note The method is specially designed to support broadcasted tensors, using the provided shape strides
*/
void CpuMath::sub(
    const float* a,  Shape a_shape, 
    const float* b, Shape b_shape, 
    float* out, Shape out_shape
) {
    assert(a_shape == b_shape && b_shape == out_shape);
    if(_try_binary_fast_path(a, a_shape, b, b_shape, out, out_shape, [](float x, float y){ return x - y; })){
        return;
    }

    // All shapes are equal, and have the same dimensionality
    auto n_dim = a_shape.n_dim;
    auto shapes = a_shape._shapes;
    std::vector<std::size_t> coords(n_dim,0);
    do{
        std::size_t a_idx = _compute_idx_from_coords(coords,a_shape._strides);
        std::size_t b_idx = _compute_idx_from_coords(coords,b_shape._strides);
        std::size_t out_idx = _compute_idx_from_coords(coords,out_shape._strides);

        out[out_idx] = a[a_idx] - b[b_idx]; 
    }while(_increment_odometer(coords,shapes));
}


/*
    @brief Scales a tensor up to a fixed float constant
    @param a Pointer to input data container 
    @param out Pointer to output data container
    @param _n Scalar
    @param shape Shared shape (out and a must have the same shape)
    
*/
void CpuMath::scale(const float* a, float* out, float _n, Shape shape) {
    for(size_t i = 0; i < shape.n_elements; i++){
        out[i] = _n*a[i];
    }
}

/*
    @brief Performs division elementwise between `a` and `b`
    @param a Pointer of first input data container
    @param a_shape Shape of first input data container
    @param b Pointer of second input data container
    @param b_shape Shape of second input data container
    @param out Pointer of output data container
    @param out_shape Shape of output data container
    @note The method is specially designed to support broadcasted tensors, using the provided shape strides
*/
void CpuMath::divide(const float* a,  Shape a_shape, const float* b, Shape b_shape, float* out, Shape out_shape){
    if(_try_binary_fast_path(a, a_shape, b, b_shape, out, out_shape, [](float x, float y){ return x / y; })){
        return;
    }

    // All shapes are equal, and have the same dimensionality
    auto n_dim = a_shape.n_dim;
    auto shapes = a_shape._shapes;
    std::vector<std::size_t> coords(n_dim,0);
    do{
        std::size_t a_idx = _compute_idx_from_coords(coords,a_shape._strides);
        std::size_t b_idx = _compute_idx_from_coords(coords,b_shape._strides);
        std::size_t out_idx = _compute_idx_from_coords(coords,out_shape._strides);

        out[out_idx] = a[a_idx] / b[b_idx]; 
    }while(_increment_odometer(coords,shapes));
}

/*
    @brief Performs the square root operation elementwise over `a`
    @param a Pointer of first input data container
    @param a_shape Shape of first input data container
    @param out Pointer of output data container
    @param out_shape Shape of output data container
    @note The method is specially designed to support broadcasted tensors, using the provided shape strides
*/
void CpuMath::sqrt(const float* a,  Shape a_shape, float* out, Shape out_shape){
    // All shapes are equal, and have the same dimensionality
    auto n_dim = a_shape.n_dim;
    auto shapes = a_shape._shapes;
    std::vector<std::size_t> coords(n_dim,0);
    do{
        std::size_t a_idx = _compute_idx_from_coords(coords,a_shape._strides);
        std::size_t out_idx = _compute_idx_from_coords(coords,out_shape._strides);

        out[out_idx] = std::sqrt(a[a_idx]); 
    }while(_increment_odometer(coords,shapes));
}

/*
    @brief Performs a `pow` operation with power `_n` elementwise over `a`
    @param a Pointer of first input data container
    @param a_shape Shape of first input data container
    @param out Pointer of output data container
    @param out_shape Shape of output data container
    @note The method is specially designed to support broadcasted tensors, provide using the correct shape strides
*/
void CpuMath::pow(const float* a,  Shape a_shape, int _n, float* out, Shape out_shape){
    // All shapes are equal, and have the same dimensionality
    auto n_dim = a_shape.n_dim;
    auto shapes = a_shape._shapes;
    std::vector<std::size_t> coords(n_dim,0);
    do{
        std::size_t a_idx = _compute_idx_from_coords(coords,a_shape._strides);
        std::size_t out_idx = _compute_idx_from_coords(coords,out_shape._strides);

        out[out_idx] = std::pow(a[a_idx],_n); 
    }while(_increment_odometer(coords,shapes));
}


void CpuMath::log(const float* a,  Shape a_shape, float* out, Shape out_shape){
    assert(a_shape == out_shape);

    std::vector<std::size_t> coords(a_shape.n_dim,0);
    do{
        std::size_t a_idx = _compute_idx_from_coords(
            coords,
            a_shape._strides
        );
        std::size_t out_idx = _compute_idx_from_coords(
            coords,
            out_shape._strides
        );
        //Clamp away from zero: log(0) is -inf, and -inf * 0 (e.g. one-hot labels) is NaN
        out[out_idx] = std::log(std::max(a[a_idx], std::numeric_limits<float>::min()));
    }while(_increment_odometer(coords,a_shape._shapes));
}


/*
    @brief Performs multiplication elementwise between `a` and `b`
    @param a Pointer of first input data container
    @param a_shape Shape of first input data container
    @param b Pointer of second input data container
    @param b_shape Shape of second input data container
    @param out Pointer of output data container
    @param out_shape Shape of output data container
    @note The method is specially designed to support broadcasted tensors, using the correct shape strides
*/
void CpuMath::multiply(
    const float* a,  Shape a_shape, 
    const float* b, Shape b_shape, 
    float* out, Shape out_shape
) {
    assert(a_shape == b_shape);
    if(_try_binary_fast_path(a, a_shape, b, b_shape, out, out_shape, [](float x, float y){ return x * y; })){
        return;
    }

    // All shapes are equal, and have the same dimensionality
    auto n_dim = a_shape.n_dim;
    auto shapes = a_shape._shapes;
    std::vector<std::size_t> coords(n_dim,0);
    do{
        std::size_t a_idx = _compute_idx_from_coords(coords,a_shape._strides);
        std::size_t b_idx = _compute_idx_from_coords(coords,b_shape._strides);
        std::size_t out_idx = _compute_idx_from_coords(coords,out_shape._strides);

        out[out_idx] = a[a_idx] * b[b_idx]; 
    }while(_increment_odometer(coords,shapes));
}

/*
    @brief Performs a tensor contraction on the two innermost dimensions between `a` and `b`
    @param a Pointer of first input data container
    @param a_shape Shape of first input data container
    @param b Pointer of second input data container
    @param b_shape Shape of second input data container
    @param out Pointer of output data container
    @param out_shape Shape of output data container
    @note The method is specially designed to support broadcasted tensors, using the correct shape strides
*/
void CpuMath::matmul(
    const float* a , Shape a_shape,
    const float* b , Shape b_shape,
    float* out, Shape out_shape
) {
    
    assert(a_shape.n_dim == b_shape.n_dim && b_shape.n_dim == out_shape.n_dim);
    std::size_t max_dim = a_shape.n_dim;
    //Simple check: ¿ do they even have the max dimension ?

    /*Basic validation*/
    if(a_shape[max_dim-1] != b_shape[max_dim-2]){ throw std::runtime_error("Error: Expected shape (...,M,K); (...,K,N)  "); }
    if(a_shape[max_dim-2] != out_shape[max_dim-2] || b_shape[max_dim-1] != out_shape[max_dim-1]){ throw std::runtime_error("Error: Expected shape (...,M,N)"); }
    
    for(size_t i = max_dim - 2 ; i-->0; ){
        if(a_shape[i] != b_shape[i] || b_shape[i] != out_shape[i] ){ throw std::runtime_error("Error: Bad broadcasting detected"); }
    }

    size_t M = out_shape[max_dim - 2];
    size_t N = out_shape[max_dim - 1];
    size_t K = a_shape[max_dim - 1];
    size_t stride_am = a_shape._strides[max_dim - 2];
    size_t stride_ak = a_shape._strides[max_dim - 1];
    size_t stride_bk = b_shape._strides[max_dim - 2];
    size_t stride_bn = b_shape._strides[max_dim - 1];
    size_t stride_outm = out_shape._strides[max_dim - 2];
    size_t stride_outn = out_shape._strides[max_dim - 1];

    if(
        max_dim == 2 &&
        _is_contiguous_2d(a_shape) &&
        _is_contiguous_2d(b_shape) &&
        _is_contiguous_2d(out_shape)
    ){
        _contiguous_2d_matrix_mul(a, b, out, M, N, K);
        return;
    }
    
    /*Now comes the main algorithm*/
    std::vector<size_t> coords(max_dim-2,0);

    do{
        /*Calculate slices*/
        const float* a_slice = a;
        const float* b_slice = b;
        float* out_slice = out;
        
        for(size_t i = 0; i < max_dim -2; ++i){
            a_slice += coords[i] * a_shape._strides[i];
            b_slice += coords[i] * b_shape._strides[i];
            out_slice += coords[i] * out_shape._strides[i];
        }

        _standard_2d_matrix_mul(
            a_slice,b_slice,out_slice,
            M,N,K,
            stride_am,stride_ak,
            stride_bk,stride_bn,
            stride_outm, stride_outn
        );
    }while(_increment_odometer(coords,out_shape._shapes));

}



/*
    @brief Given a input (image), and a series of parameters describing a 'window-sliding' operation, 
    extracts each possible window as a row of a matrix.
    @note The routine assumes `ìn` is 4-dimensional (N,C,H,W)   
*/
void CpuMath::im2col2d(
    const float* in, Shape in_shape,
    float* out, Shape out_shape,
    std::array<std::size_t,2> window_shape,
    std::array<std::size_t,2> strides,
    std::array<std::size_t,2> padding
){
    using dim_t = std::size_t;
    assert(out_shape.n_dim == 2);
    dim_t N = in_shape[0], C = in_shape[1], H = in_shape[2], W = in_shape[3];
    dim_t H_window = window_shape[0], W_window = window_shape[1];
    dim_t H_out = (H + 2 * padding[0] - H_window) / strides[0] + 1;
    dim_t W_out = (W + 2 * padding[1] - W_window) / strides[1] + 1;

    for(dim_t c = 0; c < C; ++c){
        for(dim_t kh = 0; kh < H_window; ++kh){
            for(dim_t kw = 0; kw < W_window; ++kw){
                dim_t row = (c * H_window + kh) * W_window + kw;
                for(dim_t n = 0; n < N; ++n){
                    for(dim_t oh = 0; oh < H_out; ++oh){
                        ptrdiff_t h_in = static_cast<ptrdiff_t>(oh * strides[0]) -
                            static_cast<ptrdiff_t>(padding[0]) +
                            static_cast<ptrdiff_t>(kh);
                        for(dim_t ow = 0; ow < W_out; ++ow){
                            ptrdiff_t w_in = static_cast<ptrdiff_t>(ow * strides[1]) -
                                static_cast<ptrdiff_t>(padding[1]) +
                                static_cast<ptrdiff_t>(kw);
                            dim_t col = n * H_out * W_out + oh * W_out + ow;
                            dim_t out_idx = row * out_shape._strides[0] + col * out_shape._strides[1];
                            if(
                                h_in >= 0 && h_in < static_cast<ptrdiff_t>(H) &&
                                w_in >= 0 && w_in < static_cast<ptrdiff_t>(W)
                            ){
                                dim_t in_idx =
                                    n * in_shape._strides[0] +
                                    c * in_shape._strides[1] +
                                    static_cast<dim_t>(h_in) * in_shape._strides[2] +
                                    static_cast<dim_t>(w_in) * in_shape._strides[3];
                                out[out_idx] = in[in_idx];
                            }else{
                                out[out_idx] = 0.0f;
                            }
                        }
                    }
                }
            }
        }
    }
}

/*
    @brief Given a matrix (col), and a series of parameters describing a 'window-sliding' operation that was perfomed, 
    reconstructs the shapes of the original tensor (image), using accumulation over the overlapping indices.
    @note The routine assumes `ìn` is 2-dimensional (N,C,H,W), `out` is 4-dimensional, and out is zeroed.   
*/
void CpuMath::col2im2d(
    const float* in, Shape in_shape,
    float* out, Shape out_shape,
    std::array<std::size_t,2> window_shape,
    std::array<std::size_t,2> strides,
    std::array<std::size_t,2> padding
){
    assert(in_shape.n_dim == 2 && out_shape.n_dim == 4);
    
    using dim_t = std::size_t;

    std::fill(out, out + out_shape.n_elements, 0.0f);

    dim_t N = out_shape[0], C = out_shape[1], H = out_shape[2], W = out_shape[3];
    dim_t H_window = window_shape[0], W_window = window_shape[1];
    dim_t H_in = (H + 2 * padding[0] - H_window) / strides[0] + 1;
    dim_t W_in = (W + 2 * padding[1] - W_window) / strides[1] + 1;

    for(dim_t c = 0; c < C; ++c){
        for(dim_t kh = 0; kh < H_window; ++kh){
            for(dim_t kw = 0; kw < W_window; ++kw){
                dim_t row = (c * H_window + kh) * W_window + kw;
                for(dim_t n = 0; n < N; ++n){
                    for(dim_t ih = 0; ih < H_in; ++ih){
                        ptrdiff_t h_out = static_cast<ptrdiff_t>(ih * strides[0]) -
                            static_cast<ptrdiff_t>(padding[0]) +
                            static_cast<ptrdiff_t>(kh);
                        for(dim_t iw = 0; iw < W_in; ++iw){
                            ptrdiff_t w_out = static_cast<ptrdiff_t>(iw * strides[1]) -
                                static_cast<ptrdiff_t>(padding[1]) +
                                static_cast<ptrdiff_t>(kw);
                            if(
                                h_out >= 0 && h_out < static_cast<ptrdiff_t>(H) &&
                                w_out >= 0 && w_out < static_cast<ptrdiff_t>(W)
                            ){
                                dim_t col = n * H_in * W_in + ih * W_in + iw;
                                dim_t in_idx = row * in_shape._strides[0] + col * in_shape._strides[1];
                                dim_t out_idx =
                                    n * out_shape._strides[0] +
                                    c * out_shape._strides[1] +
                                    static_cast<dim_t>(h_out) * out_shape._strides[2] +
                                    static_cast<dim_t>(w_out) * out_shape._strides[3];
                                out[out_idx] += in[in_idx];
                            }
                        }
                    }
                }
            }
        }
    }

}








/*
    @brief Performs reLU operation elementwise over `a`
    @param a Pointer of input data container
    @param a_shape Shape of input data container
    @param out Pointer of output data container
    @param out_shape Shape of output data container
    @note This method does not support broadcasting between `a` and `out`
    @note This method can safely accept that `a` points to the same container as `out`
*/
void CpuMath::reLU(const float* a, float* out, size_t n_elements) {
    for(size_t i = 0; i < n_elements; i++){
        out[i] = std::max(a[i],0.0f);
    }
}

/*
    @brief Performs sigmoid function operation elementwise over `a`
    @param a Pointer of input data container
    @param a_shape Shape of input data container
    @param out Pointer of output data container
    @param out_shape Shape of output data container
    @note This method does not support broadcasting between `a` and `out`
    @note This method can safely accept that `a` points to the same container as `out`
*/
void CpuMath::sigmoid(const float* a, float* out, size_t n_elements) {
    for(size_t i = 0; i < n_elements; i++){
        if(a[i] >= 0.0f){
            out[i] = 1/(1+std::exp(-a[i]));
        }else{
            float temp = std::exp(a[i]);
            out[i] = temp/(1+temp);
        }
    }
}

/*
    @brief Performs tanh function operation elementwise over `a`
    @param a Pointer of input data container
    @param a_shape Shape of input data container
    @param out Pointer of output data container
    @param out_shape Shape of output data container
    @note This method does not support broadcasting between `a` and `out`
    @note This method can safely accept that `a` points to the same container as `out`
*/
void CpuMath::tanh(const float* a, float* out, size_t n_elements) {
    for(size_t i = 0; i < n_elements; i++){
        out[i] = std::tanh(a[i]);
    }
}

/*
    @brief Performs softmax operation elementwise over `a`
    @param a Pointer of input data container
    @param a_shape Shape of input data container
    @param out Pointer of output data container
    @param out_shape Shape of output data container
    @note This method does not support broadcasting between `a` and `out`
    @note This method can safely accept that `a` points to the same container as `out`
    @note `a` MUST BE a 1D tensor
*/
void CpuMath::softmax(
    const float* a,
    Shape a_shape, 
    float* out,
    Shape out_shape
) {
    assert(out_shape == a_shape);
    if(a_shape.n_dim == 2 && _is_contiguous(a_shape) && _is_contiguous(out_shape)){
        std::size_t rows = a_shape[0];
        std::size_t cols = a_shape[1];

        for(std::size_t row = 0; row < rows; ++row){
            const float* a_row = a + row * cols;
            float* out_row = out + row * cols;
            float max = std::numeric_limits<float>::lowest();

            for(std::size_t col = 0; col < cols; ++col){
                if(a_row[col] > max){
                    max = a_row[col];
                }
            }

            float sum = 0.0f;
            for(std::size_t col = 0; col < cols; ++col){
                float value = std::exp(a_row[col] - max);
                out_row[col] = value;
                sum += value;
            }

            for(std::size_t col = 0; col < cols; ++col){
                out_row[col] /= sum;
            }
        }
        return;
    }

    std::vector<std::size_t> out_coords(out_shape.n_dim,0);
    std::size_t last_dim = out_shape._shapes[out_shape.n_dim-1];
    std::size_t a_last_stride = a_shape._strides[out_shape.n_dim-1];
    std::size_t out_last_stride = out_shape._strides[out_shape.n_dim-1];

    do{
        if(out_coords[out_shape.n_dim-1] == 0){
            std::size_t a_base_idx = _compute_idx_from_coords(
                out_coords,
                a_shape._strides
            );

            std::size_t out_base_idx = _compute_idx_from_coords(
                out_coords,
                out_shape._strides
            );

            float sum = 0;
            float max = std::numeric_limits<float>::lowest();
            //First, find the maximum element
            for(std::size_t i = 0; i < last_dim; i++){
                std::size_t a_global_idx = a_base_idx + i * a_last_stride;
                if(a[a_global_idx] > max) max = a[a_global_idx]; 
            }
            //Now, compute e^{x - max}
            for(std::size_t i = 0; i < last_dim; i++){
                std::size_t a_global_idx = a_base_idx + i * a_last_stride;
                std::size_t out_global_idx = out_base_idx + i * out_last_stride;
                float num = std::exp(a[a_global_idx] - max);
                out[out_global_idx] = num; 
                sum +=  num;
            }
            //Divide by the accumulated sum
            for(std::size_t i = 0; i < last_dim; i++){
                std::size_t out_global_idx = out_base_idx + i * out_last_stride;
                out[out_global_idx] /= sum; 
            }
        }
    }while(_increment_odometer(out_coords,out_shape._shapes));
    return;
}

/*
    @brief Sum two tensors along a specified axis
    @note Out should be zeroes before invoking this routine
*/
void CpuMath::sum(
    const float* a, Shape a_shape, 
    float* out, Shape out_shape ,
    size_t axis
) {
    assert(a_shape == out_shape);
    std::size_t axis_dim = a_shape[axis];
    std::vector<std::size_t> coords(a_shape.n_dim,0);
    do{
        std::size_t a_idx = _compute_idx_from_coords(
            coords,
            a_shape._strides
        );
        std::size_t out_idx = _compute_idx_from_coords(
            coords,
            out_shape._strides
        );
        out[out_idx] += a[a_idx];
    }while(_increment_odometer(coords,a_shape._shapes));
}

/*
    @brief Filters positive values from `in`, zeroing negative values. Shapes must be equal
*/
void CpuMath::binary_positive_mask(
    const float* in,
    const Shape& in_shape,
    float* out,
    const Shape& out_shape
){
    //Iterate both tensors using a unique odometer
    std::vector<std::size_t> coords(out_shape.n_dim,0);
    do{
        std::size_t in_idx = _compute_idx_from_coords(coords,in_shape._strides);
        std::size_t out_idx = _compute_idx_from_coords(coords,out_shape._strides);
        out[out_idx] = in[in_idx] > 0 ? 1.0 : 0.0;
    }while(_increment_odometer(coords,out_shape._shapes));
}

