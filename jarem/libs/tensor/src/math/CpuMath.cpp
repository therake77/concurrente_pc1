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

    bool _increment_odometer(std::array<size_t,TENSOR_MAX_DIM>& coords, std::array<size_t,TENSOR_MAX_DIM> shape){
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

    std::size_t _compute_idx_from_coords(std::array<std::size_t,MyTensors::Core::TENSOR_MAX_DIM>& coords, std::array<size_t,TENSOR_MAX_DIM> strides ){
        std::size_t offset = 0;
        for(size_t i = 0; i < coords.size();i ++){
            offset += (coords[i] * strides[i]);
        }
        return offset;
    }

    void _compute_coords_from_idx(std::size_t idx, MyTensors::Core::Shape shape, std::array<std::size_t, MyTensors::Core::TENSOR_MAX_DIM>& out){
        std::size_t remaining = idx;
        for(std::size_t i = shape.n_dim; i -->0; ){
            out[i] = remaining % shape._shapes[i];
            remaining/= shape._shapes[i];
        }
        return;
    }

    void _compute_coords_from_idx(std::size_t idx, std::array<std::size_t, MyTensors::Core::TENSOR_MAX_DIM>& shape, std::array<std::size_t, MyTensors::Core::TENSOR_MAX_DIM>& out){
        std::size_t remaining = idx;
        for(std::size_t i = TENSOR_MAX_DIM; i -->0; ){
            if(shape[i] == 0){ continue; }
            out[i] = remaining % shape[i];
            remaining/= shape[i];
        }
        return;
    }

};

namespace MyTensors::Math::Base::Routines{
    using MyTensors::Core::TENSOR_MAX_DIM;
    void elementwise_add(
        std::size_t thread_idx,
        std::size_t num_elements,
        std::size_t scope,
        const float* a,  Shape a_shape, 
        const float* b, Shape b_shape, 
        float* out, Shape out_shape
    ){
        
        //Finding out where in the out tensor we are. Out tensor is not broadcasted
        //Compute real idx
        std::size_t global_idx = thread_idx * scope;
        std::array<std::size_t, TENSOR_MAX_DIM> coords = {0};
        _compute_coords_from_idx(global_idx,out_shape,coords);

        //Now we have the coords. Compute coords from 
        for(std::size_t i = 0; i < scope; i++){
            std::size_t out_offset = _compute_idx_from_coords(coords,out_shape._strides);
            if(out_offset >= num_elements){break;}

            std::size_t a_offset = _compute_idx_from_coords(coords,a_shape._strides);
            std::size_t b_offset = _compute_idx_from_coords(coords,b_shape._strides);

            out[out_offset] = a[a_offset] + b[b_offset];
            _increment_odometer(coords,out_shape._shapes);
        }
    }

    void elementwise_sub(
        std::size_t thread_idx,
        std::size_t num_elements,
        std::size_t scope,
        const float* a,  Shape a_shape, 
        const float* b, Shape b_shape, 
        float* out, Shape out_shape
    ){
        
        //Finding out where in the out tensor we are. Out tensor is not broadcasted
        //Compute real idx
        std::size_t global_idx = thread_idx * scope;
        std::array<std::size_t, TENSOR_MAX_DIM> coords = {0};
        _compute_coords_from_idx(global_idx,out_shape,coords);

        //Now we have the coords. Compute coords from 
        for(std::size_t i = 0; i < scope; i++){
            std::size_t out_offset = _compute_idx_from_coords(coords,out_shape._strides);
            if(out_offset >= num_elements){break;}

            std::size_t a_offset = _compute_idx_from_coords(coords,a_shape._strides);
            std::size_t b_offset = _compute_idx_from_coords(coords,b_shape._strides);

            out[out_offset] = a[a_offset] - b[b_offset];
            _increment_odometer(coords,out_shape._shapes);
        }
    }

    void elementwise_scale(
        std::size_t thread_idx,
        std::size_t num_elements,
        std::size_t scope,
        const float* a, float* out, float _n
    ){
        
        //Finding out where in the out tensor we are. Out tensor is not broadcasted
        //Compute real idx
        std::size_t global_idx = thread_idx * scope;

        //Now we have the coords. Compute coords from 
        for(std::size_t i = 0; i < scope; i++){
            std::size_t offset = global_idx + i;
            if(offset >= num_elements){break;}

            out[offset] = _n*a[offset];
        }
    }
    
    void elementwise_divide(
        std::size_t thread_idx,
        std::size_t num_elements,
        std::size_t scope,
        const float* a,  Shape a_shape, 
        const float* b, Shape b_shape, 
        float* out, Shape out_shape
    ){
        
        //Finding out where in the out tensor we are. Out tensor is not broadcasted
        //Compute real idx
        std::size_t global_idx = thread_idx * scope;
        std::array<std::size_t, TENSOR_MAX_DIM> coords = {0};
        _compute_coords_from_idx(global_idx,out_shape,coords);

        //Now we have the coords. Compute coords from 
        for(std::size_t i = 0; i < scope; i++){
            std::size_t out_offset = _compute_idx_from_coords(coords,out_shape._strides);
            if(out_offset >= num_elements){break;}

            std::size_t a_offset = _compute_idx_from_coords(coords,a_shape._strides);
            std::size_t b_offset = _compute_idx_from_coords(coords,b_shape._strides);

            out[out_offset] = a[a_offset] / b[b_offset];
            _increment_odometer(coords,out_shape._shapes);
        }
    }

    void elementwise_sqrt(
        std::size_t thread_idx,
        std::size_t num_elements,
        std::size_t scope,
        const float* a,  Shape a_shape, float* out, Shape out_shape
    ){
        //Finding out where in the out tensor we are. Out tensor is not broadcasted
        //Compute real idx
        std::size_t global_idx = thread_idx * scope;
        std::array<std::size_t, TENSOR_MAX_DIM> coords = {0};
        _compute_coords_from_idx(global_idx,out_shape,coords);

        //Now we have the coords. Compute coords from 
        for(std::size_t i = 0; i < scope; i++){
            std::size_t out_offset = _compute_idx_from_coords(coords,out_shape._strides);
            if(out_offset >= num_elements){break;}

            std::size_t a_offset = _compute_idx_from_coords(coords,a_shape._strides);

            out[out_offset] = std::sqrt(a[a_offset]);
            _increment_odometer(coords,out_shape._shapes);
        }
    }

    void elementwise_pow(
        std::size_t thread_idx,
        std::size_t num_elements,
        std::size_t scope,
        const float* a,  Shape a_shape, int _n, float* out, Shape out_shape
    ){
        //Finding out where in the out tensor we are. Out tensor is not broadcasted
        //Compute real idx
        std::size_t global_idx = thread_idx * scope;
        std::array<std::size_t, TENSOR_MAX_DIM> coords = {0};
        _compute_coords_from_idx(global_idx,out_shape,coords);

        //Now we have the coords. Compute coords from 
        for(std::size_t i = 0; i < scope; i++){
            std::size_t out_offset = _compute_idx_from_coords(coords,out_shape._strides);
            if(out_offset >= num_elements){break;}

            std::size_t a_offset = _compute_idx_from_coords(coords,a_shape._strides);

            out[out_offset] = std::pow(a[a_offset],_n);
            _increment_odometer(coords,out_shape._shapes);
        }
    }
    
    void elementwise_log(
        std::size_t thread_idx,
        std::size_t num_elements,
        std::size_t scope,
        const float* a,  Shape a_shape, float* out, Shape out_shape
    ){
        //Finding out where in the out tensor we are. Out tensor is not broadcasted
        //Compute real idx
        std::size_t global_idx = thread_idx * scope;
        std::array<std::size_t, TENSOR_MAX_DIM> coords = {0};
        _compute_coords_from_idx(global_idx,out_shape,coords);

        //Now we have the coords. Compute coords from 
        for(std::size_t i = 0; i < scope; i++){
            std::size_t out_offset = _compute_idx_from_coords(coords,out_shape._strides);
            if(out_offset >= num_elements){break;}

            std::size_t a_offset = _compute_idx_from_coords(coords,a_shape._strides);

            out[out_offset] = std::log(std::max(a[a_offset], std::numeric_limits<float>::min()));
            _increment_odometer(coords,out_shape._shapes);
        }
        
    }

    void elementwise_multiply(
        std::size_t thread_idx,
        std::size_t num_elements,
        std::size_t scope,
        const float* a,  Shape a_shape, 
        const float* b, Shape b_shape, 
        float* out, Shape out_shape
    ){
        //Finding out where in the out tensor we are. Out tensor is not broadcasted
        //Compute real idx
        std::size_t global_idx = thread_idx * scope;
        std::array<std::size_t, TENSOR_MAX_DIM> coords = {0};
        _compute_coords_from_idx(global_idx,out_shape,coords);

        //Now we have the coords. Compute coords from 
        for(std::size_t i = 0; i < scope; i++){
            std::size_t out_offset = _compute_idx_from_coords(coords,out_shape._strides);
            if(out_offset >= num_elements){break;}

            std::size_t a_offset = _compute_idx_from_coords(coords,a_shape._strides);
            std::size_t b_offset = _compute_idx_from_coords(coords,b_shape._strides);

            out[out_offset] = a[a_offset] * b[b_offset];
            _increment_odometer(coords,out_shape._shapes);
        }
        return;
    }

    void matmul(
        std::size_t thread_idx,
        std::size_t num_elements,
        std::size_t scope,
        const float* a,  Shape a_shape, 
        const float* b, Shape b_shape, 
        float* out, Shape out_shape
    ){
        assert(out_shape.n_dim >= 2);
        //First figure out where we are
        std::size_t global_idx = thread_idx*scope;
        
        //Prepare odometers
        std::array<std::size_t,TENSOR_MAX_DIM> out_coords = {0} ;
        std::array<std::size_t,TENSOR_MAX_DIM> a_coords = {0};
        std::array<std::size_t,TENSOR_MAX_DIM> b_coords = {0};

        //Compute out coordinates of the first element
        _compute_coords_from_idx(global_idx,out_shape,out_coords);

        for(std::size_t i = 0 ; i < scope; i++ ){
            //Compute the out offset
            std::size_t out_offset = _compute_idx_from_coords(out_coords,out_shape._strides);
            if(out_offset >= out_shape.n_elements){ break; }
            //Copy the coordinates to a and b odometers
            std::memcpy(a_coords.data(),out_coords.data(),TENSOR_MAX_DIM*sizeof(std::size_t));
            std::memcpy(b_coords.data(),out_coords.data(),TENSOR_MAX_DIM*sizeof(std::size_t));

            //Now, a_coords needs to be (..batch_dims,M,0) and b_coords (..batch_dims,0,N)
            a_coords[a_shape.n_dim-1] = 0;
            b_coords[b_shape.n_dim-2] = 0;

            //Prepare to iterate over k dimension
            std::size_t k_dim = a_shape._shapes[a_shape.n_dim-1]; //The same as b_shape._shapes[b_shape.n_dim-2]
            float acc = 0;
            for(std::size_t k = 0; k < k_dim; k++){
                a_coords[a_shape.n_dim-1] = k;
                b_coords[b_shape.n_dim-2] = k;
                std::size_t a_offset = _compute_idx_from_coords(a_coords,a_shape._strides);
                std::size_t b_offset = _compute_idx_from_coords(b_coords,b_shape._strides);
                acc+=a[a_offset]*b[b_offset];
            }
            //Save the result on out
            out[out_offset] = acc;
            _increment_odometer(out_coords,out_shape._shapes);
        }
        return;
    }       

    void im2col2d(
        std::size_t thread_idx, std::size_t n_elements,std::size_t scope,
        const float* in, Shape in_shape,
        float* out, Shape out_shape,
        std::array<std::size_t,2> window_shape,
        std::array<std::size_t,2> strides,
        std::array<std::size_t,2> padding
    ){
        assert(out_shape.n_dim == 2);
        assert(in_shape.n_dim == 4);

        std::size_t global_offset = thread_idx * scope;
        std::array<std::size_t,TENSOR_MAX_DIM> out_coords { 0 };
        //out_coords will always have two dimensions
        _compute_coords_from_idx(global_offset,out_shape,out_coords);
        //We are going to need these shapes
        std::size_t conv_h_out = (in_shape._shapes[2] + 2* padding[0] - window_shape[0])/ strides[0] + 1;
        std::size_t conv_w_out = (in_shape._shapes[3] + 2* padding[1] - window_shape[1])/ strides[1] + 1;
        
        std::array<std::size_t, TENSOR_MAX_DIM> row_dims { in_shape[1], window_shape[0], window_shape[1], 0};
        std::array<std::size_t, TENSOR_MAX_DIM> col_dims { in_shape[0], conv_h_out, conv_w_out, 0};
        
        std::array<std::size_t, TENSOR_MAX_DIM> row_coords { 0 };
        std::array<std::size_t, TENSOR_MAX_DIM> col_coords { 0 };

        std::array<std::size_t, TENSOR_MAX_DIM> in_coords { 0 };

        for(std::size_t i = 0; i < scope; i++){
            std::size_t out_offset = _compute_idx_from_coords(out_coords,out_shape._strides);
            if(out_offset >= out_shape.n_elements){ break; }
            std::size_t row = out_coords[0];
            std::size_t col = out_coords[1];
            // Remember:
            // Ker shape : out_channels x ( channels x ker_height x ker_width )
            // Im shape : ( channels x ker_height x ker_width ) x (N x out_h_dim x out_w_dim )
            // Position is hidden behind those dimensions 
            _compute_coords_from_idx(row,row_dims,row_coords);
            _compute_coords_from_idx(col,col_dims,col_coords);
            // Now we have c, h_out, w_out. Those last ones have a direct mapping with the image
            // Recall the formulas
            // h/w_out = (h/w_in - h/w_ker + padding )//stride + 1
            //If we invert the formula: h/w_in = (h/w_out) * stride + h/w_ker - 2*padding

            ptrdiff_t h_in = (ptrdiff_t)(col_coords[1]*strides[0]) - (ptrdiff_t)(padding[0]) + (ptrdiff_t) row_coords[1];
            ptrdiff_t w_in = (ptrdiff_t)(col_coords[2]*strides[1]) - (ptrdiff_t)(padding[1]) + (ptrdiff_t) row_coords[2];
            
            
            if(h_in < 0 || h_in >= in_shape._shapes[2] || w_in < 0 || w_in >= in_shape._shapes[3]){
                out[out_offset] = 0;    
            }else{
                in_coords[0] = col_coords[0];
                in_coords[1] = row_coords[0];
                in_coords[2] = h_in;
                in_coords[3] = w_in;
                std::size_t in_idx = _compute_idx_from_coords(in_coords,in_shape._strides);
                out[out_offset] = in[in_idx];
            }
            
            _increment_odometer(out_coords,out_shape._shapes);
        }
    }

    void col2im2d(
        std::size_t thread_idx, std::size_t n_elements, std::size_t scope,
        const float* in, Shape in_shape,
        float* out, Shape out_shape,
        std::array<std::size_t,2> window_shape,
        std::array<std::size_t,2> strides,
        std::array<std::size_t,2> padding
    ){
        //Remember, out is the 4d tensor (N,C,H,W), in is the 2d matrix
        assert(in_shape.n_dim == 2);
        assert(out_shape.n_dim == 4);

        std::size_t global_idx = thread_idx * scope;
        std::array<std::size_t ,TENSOR_MAX_DIM> out_coords {0};
        //out_coords will always have four dimensions: (n, c, h, w)
        _compute_coords_from_idx(global_idx,out_shape,out_coords);

        std::size_t H_window = window_shape[0];
        std::size_t W_window = window_shape[1];
        //Number of sliding-window positions along each axis (same value im2col2d calls H_out/W_out)
        std::size_t H_in = (out_shape._shapes[2] + 2*padding[0] - H_window)/strides[0] + 1;
        std::size_t W_in = (out_shape._shapes[3] + 2*padding[1] - W_window)/strides[1] + 1;

        //Same row/col encoding im2col2d decodes; here we compose it instead, so we need
        //the strides of these "virtual" shapes, hence building real Shape objects
        Shape row_dims(std::array<std::size_t,TENSOR_MAX_DIM>{ out_shape[1], H_window, W_window, 0 });
        Shape col_dims(std::array<std::size_t,TENSOR_MAX_DIM>{ out_shape[0], H_in, W_in, 0 });

        std::array<std::size_t, TENSOR_MAX_DIM> row_coords { 0 };
        std::array<std::size_t, TENSOR_MAX_DIM> col_coords { 0 };

        for(std::size_t i = 0; i < scope; i++){
            std::size_t out_offset = _compute_idx_from_coords(out_coords,out_shape._strides);
            if(out_offset >= out_shape.n_elements){ break; }

            std::size_t n = out_coords[0];
            std::size_t c = out_coords[1];
            std::size_t h = out_coords[2];
            std::size_t w = out_coords[3];

            float acc = 0.0f;

            for(std::size_t kh = 0; kh < H_window; kh++){
                //Guard against unsigned underflow before subtracting kh
                if(kh > h + padding[0]){ continue; }
                std::size_t numerator_h = h + padding[0] - kh;
                if(numerator_h % strides[0] != 0){ continue; }
                std::size_t ih = numerator_h / strides[0];
                if(ih >= H_in){ continue; }

                for(std::size_t kw = 0; kw < W_window; kw++){
                    if(kw > w + padding[1]){ continue; }
                    std::size_t numerator_w = w + padding[1] - kw;
                    if(numerator_w % strides[1] != 0){ continue; }
                    std::size_t iw = numerator_w / strides[1];
                    if(iw >= W_in){ continue; }

                    //Compose row = (c*H_window+kh)*W_window+kw
                    row_coords[0] = c;
                    row_coords[1] = kh;
                    row_coords[2] = kw;
                    std::size_t row = _compute_idx_from_coords(row_coords,row_dims._strides);

                    //Compose col = n*H_in*W_in + ih*W_in + iw
                    col_coords[0] = n;
                    col_coords[1] = ih;
                    col_coords[2] = iw;
                    std::size_t col = _compute_idx_from_coords(col_coords,col_dims._strides);

                    std::size_t in_idx = row*in_shape._strides[0] + col*in_shape._strides[1];
                    acc += in[in_idx];
                }
            }

            out[out_offset] = acc;
            _increment_odometer(out_coords,out_shape._shapes);
        }
    }

    void elementwise_relu(
        std::size_t thread_idx, std::size_t scope,
        const float* a, float* out, size_t n_elements
    ){
        std::size_t global_idx = thread_idx * scope;

        for(std::size_t i = 0; i < scope; i++){
            std::size_t idx = global_idx + i;
            if(idx >= n_elements){ break; }
            out[idx] = std::max(a[idx],0.0f);
        }
    }

    void elementwise_sigmoid(
        std::size_t thread_idx, std::size_t scope,
        const float* a, float* out, size_t n_elements
    ){
        std::size_t global_idx = thread_idx * scope;
        for(std::size_t i = 0; i < scope; i++){
            std::size_t idx = global_idx + i;
            if(idx >= n_elements){ break; }

            if(a[idx] >= 0.0f){
                out[idx] = 1/(1+std::exp(-a[idx]));
            }else{
                float temp = std::exp(a[idx]);
                out[i] = temp/(1+temp);
            }
        }
    }

    void elementwise_tanh(
        std::size_t thread_idx, std::size_t scope,
        const float* a, float* out, size_t n_elements
    ){
        std::size_t global_idx = thread_idx * scope;
        for(std::size_t i = 0; i < scope; i++){
            std::size_t idx = global_idx + i;
            if(idx >= n_elements){ break; }

            out[idx] = std::tanh(a[idx]);
        }
    }

    void softmax2D(
        std::size_t thread_idx, std::size_t num_rows,
        const float* a,
        Shape a_shape, 
        float* out,
        Shape out_shape
    ){
        std::size_t row_dim = a_shape.n_dim == 2 ? a_shape._shapes[0] : 1;
        std::size_t col_dim = a_shape.n_dim == 2 ? a_shape._shapes[1] : a_shape._shapes[0];
        std::size_t base_row_idx = thread_idx * num_rows;
        for(std::size_t i = 0; i < num_rows; i++){
            std::size_t row_idx = base_row_idx + i;
            if(row_idx >= row_dim){ break; }
            //If is a valid row
            const float* a_row = a + row_idx*col_dim;
            float* out_row = out + row_idx*col_dim;
            //First, find max
            float max = std::numeric_limits<float>::min();
            for(std::size_t col = 0; col < col_dim; col++){
                if(a_row[col] > max  ){ max = a_row[col]; }
            }
            //Perform trick exponentation
            float sum = 0.0f;
            for(std::size_t col = 0; col < col_dim; col++){
                float value = std::exp(a_row[col] - max);
                out_row[col] = value;
                sum+=value;
            }
            //Normalize
            for(std::size_t col = 0; col < col_dim; col++){
                out_row[col] /= sum;
            }

        }
        return;
    }

    void sum_along_axis(
        std::size_t thread_idx, std::size_t scope,
        const float* a, Shape a_shape, 
        float* out, Shape out_shape,
        std::size_t axis
    ){
        std::size_t base_idx = thread_idx * scope;
        std::size_t axis_dim = a_shape[axis];
        
        //Compute the suffix
        std::size_t suffix = 1;
        for(std::size_t k = a_shape.n_dim; k --> axis+1; ){
            suffix*=a_shape[k];
        }

        
        for(std::size_t i = 0; i < scope; i++){
            std::size_t idx = base_idx + i;
            if(idx >= out_shape.n_elements){ break; }
            //Compute a_idx. remember idx = high * suffix + low, and suffix is the product of any shape tuple containing the innermost dimensions
            std::size_t low = idx % suffix;
            std::size_t high = idx / suffix;
            std::size_t a_idx = high * axis_dim * suffix + low; 
            
            float acc = 0.0f;
            for(std::size_t i = 0; i < axis_dim; i++){
                std::size_t a_axis_idx = a_idx + i * a_shape._strides[axis];
                acc+=a[a_axis_idx];
            }
            out[idx] = acc;
        }
        
    }

    void binary_positive_mask(
        std::size_t thread_idx, std::size_t scope, std::size_t n_elements,
        const float* in,
        const Shape& in_shape,
        float* out,
        const Shape& out_shape
    ){
        std::size_t base_idx = thread_idx * scope;
        std::array<std::size_t, TENSOR_MAX_DIM> out_coords { 0 };

        for(std::size_t i = 0; i < scope; i++){
            std::size_t idx = base_idx + i;
            if( idx >=  n_elements){ break; }
            out[idx] = in[idx] > 0? 1.0f : 0.0f;
        }

        return;
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
    this->pool->reset();
    std::size_t num_threads = this->pool->size();
    std::size_t scope = (out_shape.n_elements + num_threads - 1) / num_threads;

    for(std::size_t i = 0; i < num_threads; i++){
        this->pool->assign_to(
            i,
            &Routines::elementwise_add,
            i,out_shape.n_elements, scope,
            a,a_shape,
            b,b_shape,
            out,out_shape
        );
    }
    this->pool->synchronize();
    return;
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
    this->pool->reset();
    std::size_t num_threads = this->pool->size();
    std::size_t scope = (out_shape.n_elements + num_threads - 1) / num_threads;

    for(std::size_t i = 0; i < num_threads; i++){
        this->pool->assign_to(
            i,
            &Routines::elementwise_sub,
            i,out_shape.n_elements, scope,
            a,a_shape,
            b,b_shape,
            out,out_shape
        );
    }
    this->pool->synchronize();
    return;
}


/*
    @brief Scales a tensor up to a fixed float constant
    @param a Pointer to input data container 
    @param out Pointer to output data container
    @param _n Scalar
    @param shape Shared shape (out and a must have the same shape)
    
*/
void CpuMath::scale(const float* a, float* out, float _n, Shape shape) {
    this->pool->reset();
    std::size_t num_threads = this->pool->size();
    std::size_t scope = (shape.n_elements + num_threads - 1) / num_threads;

    for(std::size_t i = 0; i < num_threads; i++){
        this->pool->assign_to(
            i,
            &Routines::elementwise_scale,
            i,shape.n_elements, scope,
            a,out,_n
        );
    }
    this->pool->synchronize();
    return;
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
    this->pool->reset();
    std::size_t num_threads = this->pool->size();
    std::size_t scope = (out_shape.n_elements + num_threads - 1) / num_threads;

    for(std::size_t i = 0; i < num_threads; i++){
        this->pool->assign_to(
            i,
            &Routines::elementwise_divide,
            i,out_shape.n_elements, scope,
            a,a_shape,
            b,b_shape,
            out,out_shape
        );
    }
    this->pool->synchronize();
    return;
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
    this->pool->reset();
    std::size_t num_threads = this->pool->size();
    std::size_t scope = (out_shape.n_elements + num_threads - 1) / num_threads;

    for(std::size_t i = 0; i < num_threads; i++){
        this->pool->assign_to(
            i,
            &Routines::elementwise_sqrt,
            i,out_shape.n_elements, scope,
            a,a_shape,
            out,out_shape
        );
    }
    this->pool->synchronize();
    return;
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
    this->pool->reset();
    std::size_t num_threads = this->pool->size();
    std::size_t scope = (out_shape.n_elements + num_threads - 1) / num_threads;

    for(std::size_t i = 0; i < num_threads; i++){
        this->pool->assign_to(
            i,
            &Routines::elementwise_pow,
            i,out_shape.n_elements, scope,
            a,a_shape, _n,
            out,out_shape
        );
    }
    this->pool->synchronize();
    return;
}

void CpuMath::log(const float* a,  Shape a_shape, float* out, Shape out_shape){
    assert(a_shape == out_shape);
    this->pool->reset();
    std::size_t num_threads = this->pool->size();
    std::size_t scope = (out_shape.n_elements + num_threads - 1) / num_threads;

    for(std::size_t i = 0; i < num_threads; i++){
        this->pool->assign_to(
            i,
            &Routines::elementwise_log,
            i,out_shape.n_elements, scope,
            a,a_shape,
            out,out_shape
        );
    }
    this->pool->synchronize();
    return;
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
    this->pool->reset();
    std::size_t num_threads = this->pool->size();
    std::size_t scope = (out_shape.n_elements + num_threads - 1) / num_threads;

    for(std::size_t i = 0; i < num_threads; i++){
        this->pool->assign_to(
            i,
            &Routines::elementwise_multiply,
            i,out_shape.n_elements, scope,
            a,a_shape,
            b,b_shape,
            out,out_shape
        );
    }
    this->pool->synchronize();
    return;
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

    std::size_t n_threads = this->pool->size();
    std::size_t scope = (out_shape.n_elements + n_threads - 1)/n_threads;
    this->pool->reset();
    for(std::size_t i = 0; i < n_threads; i++){
        this->pool->assign_to(
            i,
            &Routines::matmul,
            i,out_shape.n_elements,scope,
            a,a_shape,
            b,b_shape,
            out,out_shape
        );
    }
    this->pool->synchronize();
    return;
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
    std::size_t n_threads = this->pool->size();
    this->pool->reset();
    std::size_t scope = (out_shape.n_elements + n_threads - 1)/n_threads;
    for(std::size_t i = 0; i < n_threads; i++){
        this->pool->assign_to(
            i,
            &Routines::im2col2d,
            i, out_shape.n_elements,scope,
            in, in_shape,
            out, out_shape,
            window_shape,
            strides,
            padding
        );
    }

    this->pool->synchronize();

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

    //Unlike the old scatter-based version, this gather-based routine computes each
    //output element's full sum in one shot, so no pre-zeroing of `out` is needed.
    std::size_t n_threads = this->pool->size();
    this->pool->reset();
    std::size_t scope = (out_shape.n_elements + n_threads - 1)/n_threads;
    for(std::size_t i = 0; i < n_threads; i++){
        this->pool->assign_to(
            i,
            &Routines::col2im2d,
            i, out_shape.n_elements, scope,
            in, in_shape,
            out, out_shape,
            window_shape,
            strides,
            padding
        );
    }

    this->pool->synchronize();
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
    this->pool->reset();
    std::size_t n_threads = this->pool->size();
    std::size_t scope = (n_elements + n_threads - 1) / n_threads;
    for(std::size_t i = 0; i < n_threads; i++){
        this->pool->assign_to(
            i, &Routines::elementwise_relu,
            i,scope,a,out,n_elements
        );
    };
    this->pool->synchronize();
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
    this->pool->reset();
    std::size_t n_threads = this->pool->size();
    std::size_t scope = (n_elements + n_threads - 1) / n_threads;
    for(std::size_t i = 0; i < n_threads; i++){
        this->pool->assign_to(
            i, &Routines::elementwise_sigmoid,
            i,scope,a,out,n_elements
        );
    };
    this->pool->synchronize();
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
    this->pool->reset();
    std::size_t n_threads = this->pool->size();
    std::size_t scope = (n_elements + n_threads - 1) / n_threads;
    for(std::size_t i = 0; i < n_threads; i++){
        this->pool->assign_to(
            i, &Routines::elementwise_tanh,
            i,scope,a,out,n_elements
        );
    };
    this->pool->synchronize();
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
    this->pool->reset();
    std::size_t n_rows = a_shape.n_dim == 2 ? a_shape._shapes[0] : 1;
    std::size_t n_threads = this->pool->size();
    std::size_t scope = (n_rows + n_threads - 1) / n_threads;

    for(std::size_t i = 0; i < n_threads; i++){
        this->pool->assign_to(
            i,&Routines::softmax2D,
            i,scope,
            a, a_shape,
            out, out_shape
        );
    }

    this->pool->synchronize();
    return;
}

/*
    @brief Sum a tensor along a specific axis
*/
void CpuMath::sum(
    const float* a, Shape a_shape, 
    float* out, Shape out_shape ,
    size_t axis
) {
    this->pool->reset();

    std::size_t n_threads = this->pool->size();
    std::size_t scope = (out_shape.n_elements + n_threads - 1) / n_threads;

    for(std::size_t i = 0; i < n_threads; i++){
        this->pool->assign_to( 
            i, &Routines::sum_along_axis,
            i, scope, a, a_shape,
            out, out_shape,
            axis
        );
    }

    this->pool->synchronize();
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
    assert(in_shape == out_shape);

    this->pool->reset();
    std::size_t n_threads = this->pool->size();
    std::size_t scope = (in_shape.n_elements + n_threads - 1) / n_threads;
    
    for(std::size_t i = 0; i < n_threads; i++){
        this->pool->assign_to(
            i, &Routines::binary_positive_mask,
            i, scope, in_shape.n_elements,
            in, in_shape,
            out, out_shape
        );
    }
    
    this->pool->synchronize();
    return;
}

