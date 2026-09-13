#include <math/GpuMath.cuh>
#include <stdexcept>
#include <cmath>
#include <cassert>
#include <iostream>
#include <cfloat>
#ifdef __USE_CUDA__


    namespace MyTensors::Math::Base::Kernels{
        using MyTensors::Core::Shape;
        using MyTensors::Core::TENSOR_MAX_DIM;
        struct GpuShape{
            size_t shapes[TENSOR_MAX_DIM];
            size_t strides[TENSOR_MAX_DIM];
            size_t n_dim;
            size_t n_elements;

            static GpuShape from_shape(const Shape& s){
                GpuShape g;
                g.n_dim = s.n_dim;
                g.n_elements = s.n_elements;
                for(size_t i = 0; i < TENSOR_MAX_DIM; i++){
                    g.shapes[i]  = s._shapes[i];
                    g.strides[i] = s._strides[i];
                }
                return g;
            }

            __host__ __device__ size_t operator[](size_t i) const { return shapes[i]; }
            __host__ __device__ size_t stride(size_t i)     const { return strides[i]; }
        };

        struct Gpu1DKernelDimension{
            size_t num_blocks_per_grid;
            size_t num_threads_per_block;
        };

        void debug_check_ptr(const void* ptr, const char* name){
            cudaPointerAttributes attrs;
            cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
            if(err != cudaSuccess || attrs.type != cudaMemoryTypeDevice){
                printf("WRONG MEMORY TYPE for %s: type=%d\n", name, attrs.type);
            }
        }

        inline Gpu1DKernelDimension _compute_1D_dispatch_dimensions(Shape a){
            size_t total_elements = a.n_elements;
            size_t blocks_per_grid = (total_elements + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK; 
            return {
                blocks_per_grid,
                THREADS_PER_BLOCK
            };
        };

        __device__ void _compute_coords_from_idx(
            const size_t idx,
            size_t* coords,
            const GpuShape shape ,const size_t n_dims
        ){
            size_t remaining = idx;
            for(size_t i = TENSOR_MAX_DIM; i-->0; ){
                if(i < n_dims){
                    coords[i] = remaining % shape[i];
                    remaining/=shape[i];
                }else{ 
                    coords[i] = 0;
                }
            }
        }

        __device__ void _compute_coords_from_idx(
            const size_t idx,
            size_t* coords,
            const GpuShape shape
        ){
            size_t remaining = idx;
            for(size_t i = TENSOR_MAX_DIM; i-->0; ){
                if(i < shape.n_dim){
                    coords[i] = remaining % shape[i];
                    remaining/=shape[i];
                }else{ 
                    coords[i] = 0;
                }
            }
        }

        __device__ size_t _compute_idx_from_coords(
            size_t* coords,
            const GpuShape shape,
            const size_t n_dims
        ){
            size_t idx = 0;
            for(size_t i = n_dims; i --> 0;) {
                idx += coords[i] * shape.stride(i);
            }
            return idx;
        }

        __device__ size_t _compute_idx_from_coords(
            size_t* coords,
            const GpuShape shape
        ){
            size_t idx = 0;
            for(size_t i = shape.n_dim; i --> 0;) {
                idx += coords[i] * shape.stride(i);
            }
            return idx;
        }


        __global__ void add_kernel(
            const float* in_1,
            const GpuShape in_1_shape,
            const float* in_2,
            const GpuShape in_2_shape,
            float* out,
            const GpuShape out_shape,
            const size_t n_dims,
            const size_t n_elements
        ){
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx >= n_elements) return;
            size_t coords[TENSOR_MAX_DIM];  //Shapes are equal across inputs and output
            _compute_coords_from_idx(idx,coords,out_shape,n_dims);
            size_t in_1_idx = _compute_idx_from_coords(coords,in_1_shape,n_dims);
            size_t in_2_idx = _compute_idx_from_coords(coords,in_2_shape,n_dims);
            size_t out_idx = _compute_idx_from_coords(coords,out_shape,n_dims);
            out[out_idx] = in_1[in_1_idx] + in_2[in_2_idx];
        };

        __global__ void sub_kernel(
            const float* in_1,
            const GpuShape in_1_shape,
            const float* in_2,
            const GpuShape in_2_shape,
            float* out,
            const GpuShape out_shape,
            const size_t n_dims,
            const size_t n_elements
        ){
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx >= n_elements) return;
            size_t coords[TENSOR_MAX_DIM];  //Shapes are equal across inputs and output
            _compute_coords_from_idx(idx,coords,out_shape,n_dims);
            size_t in_1_idx = _compute_idx_from_coords(coords,in_1_shape,n_dims);
            size_t in_2_idx = _compute_idx_from_coords(coords,in_2_shape,n_dims);
            size_t out_idx = _compute_idx_from_coords(coords,out_shape,n_dims);
            out[out_idx] = in_1[in_1_idx] - in_2[in_2_idx];
        };

        __global__ void multiply_kernel(
            const float* in_1,
            const GpuShape in_1_shape,
            const float* in_2,
            const GpuShape in_2_shape,
            float* out,
            const GpuShape out_shape,
            const size_t n_dims,
            const size_t n_elements
        ){
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx >= n_elements) return;
            size_t coords[TENSOR_MAX_DIM];  //Shapes are equal across inputs and output
            _compute_coords_from_idx(idx,coords,out_shape,n_dims);
            size_t in_1_idx = _compute_idx_from_coords(coords,in_1_shape,n_dims);
            size_t in_2_idx = _compute_idx_from_coords(coords,in_2_shape,n_dims);
            size_t out_idx = _compute_idx_from_coords(coords,out_shape,n_dims);
            out[out_idx] = in_1[in_1_idx] * in_2[in_2_idx];
        };

        __global__ void scale_kernel(
            const float* in,
            const GpuShape in_shape,
            const float factor,
            float* out,
            const GpuShape out_shape,
            const size_t n_dims,
            const size_t n_elements
        ){
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx >= n_elements) return;
            size_t coords[TENSOR_MAX_DIM];
            _compute_coords_from_idx(idx,coords,out_shape,n_dims);
            size_t in_idx = _compute_idx_from_coords(coords,in_shape,n_dims);
            size_t out_idx = _compute_idx_from_coords(coords,out_shape,n_dims);
            out[out_idx] = factor * in[in_idx] ;
        };

        __global__ void divide_kernel(
            const float* in_1,
            const GpuShape in_1_shape,
            const float* in_2,
            const GpuShape in_2_shape,
            float* out,
            const GpuShape out_shape,
            const size_t n_dims,
            const size_t n_elements
        ){
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx >= n_elements) return;
            size_t coords[TENSOR_MAX_DIM];  //Shapes are equal across inputs and output
            _compute_coords_from_idx(idx,coords,out_shape,n_dims);
            size_t in_1_idx = _compute_idx_from_coords(coords,in_1_shape,n_dims);
            size_t in_2_idx = _compute_idx_from_coords(coords,in_2_shape,n_dims);
            size_t out_idx = _compute_idx_from_coords(coords,out_shape,n_dims);
            out[out_idx] = in_1[in_1_idx] / in_2[in_2_idx];
        };

        __global__ void sqrt_kernel(
            const float* in,
            const GpuShape in_shape,
            float* out,
            const GpuShape out_shape,
            const size_t n_dims,
            const size_t n_elements
        ){

            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx >= n_elements) return;
            size_t coords[TENSOR_MAX_DIM];
            _compute_coords_from_idx(idx,coords,out_shape,n_dims);
            size_t in_idx = _compute_idx_from_coords(coords,in_shape,n_dims);
            size_t out_idx = _compute_idx_from_coords(coords,out_shape,n_dims);
            out[out_idx] = sqrt(in[in_idx]);
        };

        __global__ void pow_kernel(
            const float* in,
            const GpuShape in_shape,
            int _n,
            float* out,
            const GpuShape out_shape,
            const size_t n_dims,
            const size_t n_elements
        ){

            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx >= n_elements) return;
            size_t coords[TENSOR_MAX_DIM];
            _compute_coords_from_idx(idx,coords,out_shape,n_dims);
            size_t in_idx = _compute_idx_from_coords(coords,in_shape,n_dims);
            size_t out_idx = _compute_idx_from_coords(coords,out_shape,n_dims);
            out[out_idx] = pow(in[in_idx],_n);
        };

        __global__ void log_kernel(
            const float* in,
            const GpuShape in_shape,
            float* out,
            const GpuShape out_shape,
            const size_t n_dims,
            const size_t n_elements
        ){

            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx >= n_elements) return;
            size_t coords[TENSOR_MAX_DIM];
            _compute_coords_from_idx(idx,coords,out_shape,n_dims);
            size_t in_idx = _compute_idx_from_coords(coords,in_shape,n_dims);
            size_t out_idx = _compute_idx_from_coords(coords,out_shape,n_dims);
            //Clamp away from zero: log(0) is -inf, and -inf * 0 (e.g. one-hot labels) is NaN
            out[out_idx] = log(fmaxf(in[in_idx], FLT_MIN));
        };

        __global__ void softmax_2d_kernel(
            const float* in,
            const GpuShape in_shape,
            float* out,
            const GpuShape out_shape,
            size_t rows,
            size_t cols
        ){
            size_t row = blockIdx.x * blockDim.x + threadIdx.x;
            if(row >= rows) return;

            float max_value = -FLT_MAX;
            for(size_t col = 0; col < cols; ++col){
                size_t in_idx = row * in_shape.stride(0) + col * in_shape.stride(1);
                max_value = fmaxf(max_value, in[in_idx]);
            }

            float sum = 0.0f;
            for(size_t col = 0; col < cols; ++col){
                size_t in_idx = row * in_shape.stride(0) + col * in_shape.stride(1);
                size_t out_idx = row * out_shape.stride(0) + col * out_shape.stride(1);
                float value = expf(in[in_idx] - max_value);
                out[out_idx] = value;
                sum += value;
            }

            for(size_t col = 0; col < cols; ++col){
                size_t out_idx = row * out_shape.stride(0) + col * out_shape.stride(1);
                out[out_idx] /= sum;
            }
        };

        __global__ void reLU_kernel(
            const float* in,
            float* out,
            const size_t n_elements
        ){
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx >= n_elements) return;
            float value = in[idx];
            out[idx] = value > 0.0f? value : 0.0f;
        };

        __global__ void matmul_kernel(
            const float* a,
            const GpuShape a_shape,
            const float* b,
            const GpuShape b_shape,
            float* out,
            const GpuShape out_shape,
            size_t M,size_t N,size_t K,
            size_t n_dim
        ){
            size_t batch_idx = blockIdx.z;
            size_t batch_coords[TENSOR_MAX_DIM];
            
            _compute_coords_from_idx(batch_idx,batch_coords,out_shape,n_dim);
            
            size_t a_batch_idx = _compute_idx_from_coords(batch_coords,a_shape,n_dim);  
            size_t b_batch_idx = _compute_idx_from_coords(batch_coords,b_shape,n_dim);  
            size_t out_batch_idx = _compute_idx_from_coords(batch_coords,out_shape,n_dim);  

            const float* a_slice = a + a_batch_idx;
            const float* b_slice = b + b_batch_idx;
            float* out_slice = out + out_batch_idx;

            size_t row = blockIdx.y * TILE_SIZE + threadIdx.y;
            size_t col = blockIdx.x * TILE_SIZE + threadIdx.x;

            __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
            __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
            float acc = 0.0f;

            size_t n_tiles = (K + TILE_SIZE - 1 )/TILE_SIZE;
            for(size_t i = 0; i < n_tiles; i++){
                //Load tile_A
                size_t k_row = i*TILE_SIZE + threadIdx.y;
                size_t k_col = i*TILE_SIZE + threadIdx.x;
                if(row < M && k_col < K){
                    tile_A[threadIdx.y][threadIdx.x] = a_slice[
                        row * a_shape.stride(n_dim - 2) + k_col * a_shape.stride(n_dim - 1)
                    ];
                }else{
                    tile_A[threadIdx.y][threadIdx.x] = 0.0f;
                }

                //Load tile_B
                if(k_row < K && col < N){
                    tile_B[threadIdx.y][threadIdx.x] = b_slice[
                        k_row * b_shape.stride(n_dim - 2) + col * b_shape.stride(n_dim - 1)
                    ];
                }else{
                    tile_B[threadIdx.y][threadIdx.x] = 0.0f;
                }
                __syncthreads();
                for(size_t k = 0; k < TILE_SIZE; k++){
                    acc += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
                }
                __syncthreads();
            }
            if(row < M && col < N){
                out_slice[row*out_shape.stride(n_dim-2) + col * out_shape.stride(n_dim-1)] = acc;
            }
            return;
        }

        void __global__ im2col2d_kernel(
            const float* in,
            float* out,
            size_t N, size_t C, size_t H, size_t W,
            size_t in_s0, size_t in_s1, size_t in_s2, size_t in_s3,
            size_t out_rows, size_t out_cols,
            size_t out_s0, size_t out_s1,
            size_t H_window, size_t W_window,
            size_t H_out, size_t W_out,
            size_t stride_h, size_t stride_w,
            size_t pad_h, size_t pad_w,
            size_t total_elements
        ){
            size_t flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(flat_idx >= total_elements) return;

            size_t rows = flat_idx / out_cols;
            size_t cols = flat_idx % out_cols;

            size_t delta_w = rows % W_window;
            size_t delta_h = (rows / W_window) % H_window;
            size_t c       = rows / (H_window * W_window);

            size_t w_out = cols % W_out;
            size_t h_out = (cols / W_out) % H_out;
            size_t n     = cols / (H_out * W_out);

            ptrdiff_t h_in = (ptrdiff_t)(h_out * stride_h) - (ptrdiff_t)pad_h + (ptrdiff_t)delta_h;
            ptrdiff_t w_in = (ptrdiff_t)(w_out * stride_w) - (ptrdiff_t)pad_w + (ptrdiff_t)delta_w;

            size_t out_idx = rows * out_s0 + cols * out_s1;

            if(h_in >= 0 && h_in < (ptrdiff_t)H && w_in >= 0 && w_in < (ptrdiff_t)W){
                size_t in_idx = n   * in_s0
                            + c   * in_s1
                            + (size_t)h_in * in_s2
                            + (size_t)w_in * in_s3;
                out[out_idx] = in[in_idx];
            } else {
                out[out_idx] = 0.0f;
            }
        }

        __global__ void col2im2d_kernel(
            const float* in,
            float* out,
            // out shape (the original image shape N,C,H,W)
            size_t N, size_t C, size_t H, size_t W,
            size_t out_s0, size_t out_s1, size_t out_s2, size_t out_s3,
            // in shape (the col matrix)
            size_t in_rows, size_t in_cols,
            size_t in_s0, size_t in_s1,
            // window params
            size_t H_window, size_t W_window,
            size_t H_in, size_t W_in,     // spatial dims of the col matrix
            size_t stride_h, size_t stride_w,
            size_t pad_h, size_t pad_w,
            size_t total_elements
        ){
            size_t flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(flat_idx >= total_elements) return;

            // Decompose flat_idx into (rows, cols) of the input col matrix
            size_t rows = flat_idx / in_cols;
            size_t cols = flat_idx % in_cols;

            // Same decomposition as CPU col2im
            size_t delta_w = rows % W_window;
            size_t delta_h = (rows / W_window) % H_window;
            size_t c       = rows / (H_window * W_window);

            size_t w_in = cols % W_in;
            size_t h_in = (cols / W_in) % H_in;
            size_t n    = cols / (H_in * W_in);

            ptrdiff_t h_out = (ptrdiff_t)(h_in * stride_h) - (ptrdiff_t)pad_h + (ptrdiff_t)delta_h;
            ptrdiff_t w_out = (ptrdiff_t)(w_in * stride_w) - (ptrdiff_t)pad_w + (ptrdiff_t)delta_w;

            if(h_out >= 0 && h_out < (ptrdiff_t)H && w_out >= 0 && w_out < (ptrdiff_t)W){
                size_t in_idx  = rows * in_s0 + cols * in_s1;
                size_t out_idx = n            * out_s0
                            + c            * out_s1
                            + (size_t)h_out * out_s2
                            + (size_t)w_out * out_s3;

                // Atomic because multiple threads may write to the same out_idx
                atomicAdd(&out[out_idx], in[in_idx]);
            }
        }

        
        __global__ void sum_kernel(
            const float* a,
            float* out,
            GpuShape a_shape,
            GpuShape out_shape,
            size_t total_elements
        ){
            size_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(thread_idx >= total_elements) return;

            size_t coords[TENSOR_MAX_DIM];
            _compute_coords_from_idx(thread_idx, coords, a_shape);

            size_t a_idx   = _compute_idx_from_coords(coords, a_shape);
            size_t out_idx = _compute_idx_from_coords(coords, out_shape);

            atomicAdd(&out[out_idx], a[a_idx]);
        }


        __global__ void divide_by_scalar_kernel(
            float* data,
            float scalar,
            size_t n
        ){
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx >= n) return;
            data[idx] /= scalar;
        }

        //One thread per row (i.e. per element of every axis except the innermost one).
        //Mirrors CpuMath::softmax's max-subtraction trick for numerical stability.
        __global__ void softmax_kernel(
            const float* in,
            float* out,
            GpuShape in_shape,
            GpuShape out_shape,
            size_t n_dims,
            size_t last_dim,
            size_t total_rows
        ){
            size_t row = blockIdx.x * blockDim.x + threadIdx.x;
            if(row >= total_rows) return;

            //Decode `row` into coords for dims [0, n_dims-2]; the innermost dim is iterated below
            size_t coords[TENSOR_MAX_DIM];
            size_t remaining = row;
            for(size_t i = n_dims - 1; i --> 0; ){
                coords[i] = remaining % out_shape[i];
                remaining /= out_shape[i];
            }
            coords[n_dims-1] = 0;

            size_t in_base_idx  = _compute_idx_from_coords(coords, in_shape, n_dims);
            size_t out_base_idx = _compute_idx_from_coords(coords, out_shape, n_dims);
            size_t in_last_stride  = in_shape.stride(n_dims-1);
            size_t out_last_stride = out_shape.stride(n_dims-1);

            float max_val = -3.402823466e+38f; // -FLT_MAX
            for(size_t i = 0; i < last_dim; i++){
                float v = in[in_base_idx + i*in_last_stride];
                if(v > max_val) max_val = v;
            }

            float sum = 0.0f;
            for(size_t i = 0; i < last_dim; i++){
                float v = expf(in[in_base_idx + i*in_last_stride] - max_val);
                out[out_base_idx + i*out_last_stride] = v;
                sum += v;
            }

            for(size_t i = 0; i < last_dim; i++){
                out[out_base_idx + i*out_last_stride] /= sum;
            }
        }


    };

    using namespace MyTensors::Math::Base;
    //Math backend API
    void GpuMath::add(const float* a,  Shape a_shape, const float* b, Shape b_shape, float* out, Shape out_shape) {

        assert(a_shape == b_shape);
        assert(a_shape == out_shape);
        
        debug_check_ptr(a,"a");
        debug_check_ptr(b,"a");
        debug_check_ptr(out,"a");
        
        
        size_t n_dims = out_shape.n_dim;
        size_t n_elements = out_shape.n_elements;
        
        GpuShape gpu_a_shape = GpuShape::from_shape(a_shape);
        GpuShape gpu_b_shape = GpuShape::from_shape(b_shape);
        GpuShape gpu_out_shape = GpuShape::from_shape(out_shape);
        
        Gpu1DKernelDimension dims = _compute_1D_dispatch_dimensions(a_shape);
        add_kernel<<<dims.num_blocks_per_grid,dims.num_threads_per_block>>>(
            a,
            gpu_a_shape,
            b,
            gpu_b_shape,
            out,
            gpu_out_shape,
            n_dims,
            n_elements
        );

        auto status_code = cudaDeviceSynchronize();
        if(status_code != cudaSuccess){
            std::cerr<<cudaGetErrorString(status_code);
            throw std::runtime_error("Error: kernel execution failed");
        }
        return;
    }

    void GpuMath::sub(const float* a,  Shape a_shape, const float* b, Shape b_shape, float* out, Shape out_shape) {

        assert(a_shape == b_shape);
        assert(a_shape == out_shape);
        size_t n_dims = out_shape.n_dim;
        size_t n_elements = out_shape.n_elements;
        
        GpuShape gpu_a_shape = GpuShape::from_shape(a_shape);
        GpuShape gpu_b_shape = GpuShape::from_shape(b_shape);
        GpuShape gpu_out_shape = GpuShape::from_shape(out_shape);
        
        Gpu1DKernelDimension dims = _compute_1D_dispatch_dimensions(a_shape);
        sub_kernel<<<dims.num_blocks_per_grid,dims.num_threads_per_block>>>(
            a,
            gpu_a_shape,
            b,
            gpu_b_shape,
            out,
            gpu_out_shape,
            n_dims,
            n_elements
        );

        auto status_code = cudaDeviceSynchronize();
        if(status_code != cudaSuccess){
            throw std::runtime_error("Error: kernel execution failed");
        }
        return;
    }

    void GpuMath::scale(const float* a, float* out, float _n, Shape shape) {

        size_t n_dims = shape.n_dim;
        size_t n_elements = shape.n_elements;
        Gpu1DKernelDimension dims = _compute_1D_dispatch_dimensions(shape);
        
        GpuShape gpu_shape = GpuShape::from_shape(shape);

        scale_kernel<<<dims.num_blocks_per_grid,dims.num_threads_per_block>>>(
            a,
            gpu_shape,
            _n,
            out,
            gpu_shape,
            n_dims,
            n_elements
        );

        auto status_code = cudaDeviceSynchronize();
        if(status_code != cudaSuccess){
            throw std::runtime_error("Error: kernel execution failed");
        }
        return;
    }

    void GpuMath::multiply(const float* a,  Shape a_shape, const float* b, Shape b_shape, float* out, Shape out_shape) {

        assert(a_shape == b_shape);
        assert(a_shape == out_shape);
        size_t n_dims = out_shape.n_dim;
        size_t n_elements = out_shape.n_elements;
        
        GpuShape gpu_a_shape = GpuShape::from_shape(a_shape);
        GpuShape gpu_b_shape = GpuShape::from_shape(b_shape);
        GpuShape gpu_out_shape = GpuShape::from_shape(out_shape);
        
        Gpu1DKernelDimension dims = _compute_1D_dispatch_dimensions(a_shape);
        multiply_kernel<<<dims.num_blocks_per_grid,dims.num_threads_per_block>>>(
            a,
            gpu_a_shape,
            b,
            gpu_b_shape,
            out,
            gpu_out_shape,
            n_dims,
            n_elements
        );

        auto status_code = cudaDeviceSynchronize();
        if(status_code != cudaSuccess){
            throw std::runtime_error("Error: kernel execution failed");
        }
        return;
    }

    void GpuMath::divide(const float* a,  Shape a_shape, const float* b, Shape b_shape, float* out, Shape out_shape) {
        assert(a_shape == b_shape);
        assert(a_shape == out_shape);
        size_t n_dims = out_shape.n_dim;
        size_t n_elements = out_shape.n_elements;
        
        GpuShape gpu_a_shape = GpuShape::from_shape(a_shape);
        GpuShape gpu_b_shape = GpuShape::from_shape(b_shape);
        GpuShape gpu_out_shape = GpuShape::from_shape(out_shape);
        
        Gpu1DKernelDimension dims = _compute_1D_dispatch_dimensions(a_shape);
        divide_kernel<<<dims.num_blocks_per_grid,dims.num_threads_per_block>>>(
            a,
            gpu_a_shape,
            b,
            gpu_b_shape,
            out,
            gpu_out_shape,
            n_dims,
            n_elements
        );

        auto status_code = cudaDeviceSynchronize();
        if(status_code != cudaSuccess){
            throw std::runtime_error("Error: kernel execution failed");
        }
        return;
    }

    void GpuMath::sqrt(const float* a,  Shape a_shape, float* out, Shape out_shape) {

        assert(a_shape == out_shape);
        GpuShape gpu_a_shape = GpuShape::from_shape(a_shape);
        GpuShape gpu_out_shape = GpuShape::from_shape(out_shape);
        
        size_t n_dims = out_shape.n_dim;
        size_t n_elements = out_shape.n_elements;
        Gpu1DKernelDimension dims = _compute_1D_dispatch_dimensions(out_shape);
        sqrt_kernel<<<dims.num_blocks_per_grid,dims.num_threads_per_block>>>(
            a,
            gpu_a_shape,
            out,
            gpu_out_shape,
            n_dims,
            n_elements
        );
        auto status_code = cudaDeviceSynchronize();
        if(status_code != cudaSuccess){
            throw std::runtime_error("Error: kernel execution failed");
        }
        return;
    }

    void GpuMath::pow(const float* a,  Shape a_shape, int _n, float* out, Shape out_shape) {

        assert(a_shape == out_shape);
        GpuShape gpu_a_shape = GpuShape::from_shape(a_shape);
        GpuShape gpu_out_shape = GpuShape::from_shape(out_shape);
        
        size_t n_dims = out_shape.n_dim;
        size_t n_elements = out_shape.n_elements;
        Gpu1DKernelDimension dims = _compute_1D_dispatch_dimensions(out_shape);
        LLCN_MATH_KERNELS::pow_kernel<<<dims.num_blocks_per_grid,dims.num_threads_per_block>>>(
            a,
            gpu_a_shape,
            _n,
            out,
            gpu_out_shape,
            n_dims,
            n_elements
        );

        auto status_code = cudaDeviceSynchronize();
        if(status_code != cudaSuccess){
            throw std::runtime_error("Error: kernel execution failed");
        }
        return;
    }


    void GpuMath::log(const float* a,  Shape a_shape, float* out, Shape out_shape){


        assert(a_shape == out_shape);
        GpuShape gpu_a_shape = GpuShape::from_shape(a_shape);
        GpuShape gpu_out_shape = GpuShape::from_shape(out_shape);
        
        size_t n_dims = out_shape.n_dim;
        size_t n_elements = out_shape.n_elements;
        Gpu1DKernelDimension dims = _compute_1D_dispatch_dimensions(out_shape);
        LLCN_MATH_KERNELS::log_kernel<<<dims.num_blocks_per_grid,dims.num_threads_per_block>>>(
            a,
            gpu_a_shape,
            out,
            gpu_out_shape,
            n_dims,
            n_elements
        );
        auto status_code = cudaDeviceSynchronize();
        if(status_code != cudaSuccess){
            throw std::runtime_error("Error: kernel execution failed");
        }
        return;
    }

    //Linear Algebra
    void GpuMath::matmul(
        const float* a , Shape a_shape,  
        const float* b , Shape b_shape,
        float* out, Shape out_shape
    ) {
        ;
        size_t n_dim = out_shape.n_dim;
        size_t M = out_shape[n_dim - 2];
        size_t N = out_shape[n_dim - 1];
        size_t K = a_shape[n_dim - 1];

        // Total batch count = product of all dims except last two
        size_t total_batches = 1;
        for(size_t i = 0; i < n_dim - 2; i++)
            total_batches *= out_shape[i];

        // 2D tile grid over (N, M), one layer per batch
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid(
            (N + TILE_SIZE - 1) / TILE_SIZE,   // blocks along N
            (M + TILE_SIZE - 1) / TILE_SIZE,   // blocks along M
            total_batches             // one z-slice per batch
        );

        GpuShape gpu_a_shape = GpuShape::from_shape(a_shape);
        GpuShape gpu_b_shape = GpuShape::from_shape(b_shape);
        GpuShape gpu_out_shape = GpuShape::from_shape(out_shape);

        LLCN_MATH_KERNELS::matmul_kernel<<<grid, block>>>(
            a, gpu_a_shape,
            b, gpu_b_shape,
            out, gpu_out_shape,
            M, N, K, n_dim
        );

        auto result = cudaDeviceSynchronize();
        if(result != cudaSuccess){
            throw std::runtime_error("Error: kernel execution failed");
        }
        return;
    }

    void GpuMath::im2col2d(
        const float* in, Shape in_shape,
        float* out, Shape out_shape,
        std::array<size_t,2> window_shape,
        std::array<size_t,2> strides,
        std::array<size_t,2> padding
    ){
        size_t N = in_shape[0], C = in_shape[1], H = in_shape[2], W = in_shape[3];
        size_t H_window = window_shape[0], W_window = window_shape[1];
        size_t H_out = (H + 2 * padding[0] - H_window) / strides[0] + 1;
        size_t W_out = (W + 2 * padding[1] - W_window) / strides[1] + 1;

        size_t out_rows = out_shape[0];
        size_t out_cols = out_shape[1];
        size_t total    = out_rows * out_cols;

        size_t block_size = 256;
        size_t grid_size  = (total + block_size - 1) / block_size;

        LLCN_MATH_KERNELS::im2col2d_kernel<<<grid_size, block_size>>>(
            in, out,
            N, C, H, W,
            in_shape._strides[0], in_shape._strides[1],
            in_shape._strides[2], in_shape._strides[3],
            out_rows, out_cols,
            out_shape._strides[0], out_shape._strides[1],
            H_window, W_window,
            H_out, W_out,
            strides[0], strides[1],
            padding[0], padding[1],
            total
        );
        auto result = cudaDeviceSynchronize();
        if(result != cudaSuccess){
            throw std::runtime_error("Error: kernel execution failed");
        }
        return;
    }

    void GpuMath::col2im2d(
        const float* in, Shape in_shape,
        float* out, Shape out_shape,
        std::array<size_t,2> window_shape,
        std::array<size_t,2> strides,
        std::array<size_t,2> padding
    ){
        size_t N = out_shape[0], C = out_shape[1], H = out_shape[2], W = out_shape[3];
        size_t H_window = window_shape[0], W_window = window_shape[1];
        size_t H_in = (H + 2 * padding[0] - H_window) / strides[0] + 1;
        size_t W_in = (W + 2 * padding[1] - W_window) / strides[1] + 1;

        size_t in_rows = in_shape[0];
        size_t in_cols = in_shape[1];
        size_t total   = in_rows * in_cols;

        size_t block_size = 256;
        size_t grid_size  = (total + block_size - 1) / block_size;

        //col2im2d_kernel accumulates into `out` via atomicAdd, so the output buffer must start
        //at zero (the CPU path does std::fill(...,0) for the same reason). Without this, the
        //conv dInput gradient is added onto stale/random buffer contents, which silently poisons
        //upstream gradients whenever a Conv2D feeds another layer's backward pass.
        cudaMemset(out, 0, out_shape.n_elements * sizeof(float));

        LLCN_MATH_KERNELS::col2im2d_kernel<<<grid_size, block_size>>>(
            in, out,
            N, C, H, W,
            out_shape._strides[0], out_shape._strides[1],
            out_shape._strides[2], out_shape._strides[3],
            in_rows, in_cols,
            in_shape._strides[0], in_shape._strides[1],
            window_shape[0],window_shape[1],
            H_in, W_in,
            strides[0], strides[1],
            padding[0], padding[1],
            total
        );

        auto result = cudaDeviceSynchronize();
        if(result != cudaSuccess){
            throw std::runtime_error("Error: kernel execution failed");
        }
        return;

    }

    //Activation functions
    void GpuMath::reLU(
        const float* a, float* out, size_t n_elements
    ) {
        size_t blocks_per_grid = (n_elements + THREADS_PER_BLOCK - 1 )/THREADS_PER_BLOCK;
        reLU_kernel<<<blocks_per_grid,THREADS_PER_BLOCK>>>(
            a,
            out,
            n_elements
        );

        auto status_code = cudaDeviceSynchronize();
        if(status_code != cudaSuccess){
            throw std::runtime_error("Error: Add kernel execution failed");
        }
        return;
    }

    void GpuMath::sigmoid(const float* a, float* out, size_t n_elements) {

    }

    void GpuMath::tanh(const float* a, float* out, size_t n_elements) {

    }

    void GpuMath::softmax(
        const float* a,
        Shape a_shape,
        float* out,
        Shape out_shape
    ) {
        ;
        GpuShape g_a   = GpuShape::from_shape(a_shape);
        GpuShape g_out = GpuShape::from_shape(out_shape);

        size_t n_dims    = out_shape.n_dim;
        size_t last_dim  = g_out[n_dims-1];
        size_t total_rows = out_shape.n_elements / last_dim;

        size_t block_size = 256;
        size_t grid_size  = (total_rows + block_size - 1) / block_size;

        softmax_kernel<<<grid_size, block_size>>>(
            a, out,
            g_a, g_out,
            n_dims, last_dim, total_rows
        );

        auto status_code = cudaDeviceSynchronize();
        if(status_code != cudaSuccess){
            throw std::runtime_error("Error: softmax kernel execution failed");
        }
        return;
    }
        

    void GpuMath::sum(
        const float* a, Shape a_shape, 
        float* out, Shape out_shape ,
        size_t axis
    ) {
        ;
        GpuShape g_a   = GpuShape::from_shape(a_shape);
        GpuShape g_out = GpuShape::from_shape(out_shape);

        size_t total = g_a.n_elements;

        size_t block_size = 256;
        size_t grid_size  = (total + block_size - 1) / block_size;

        sum_kernel<<<grid_size, block_size>>>(
            a, out,
            g_a, g_out,
            total
        );

        auto status_code = cudaDeviceSynchronize();
        if(status_code != cudaSuccess){
            throw std::runtime_error("Error: Add kernel execution failed");
        }
        return;
    }


#endif //__USE_CUDA__
