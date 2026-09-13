#include <hardware/GpuDevice.cuh>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <cassert>
#include <core/Shape.hpp>

#ifdef __USE_CUDA__
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

using namespace MyTensors::Hardware;

void* GpuDevice::allocate(size_t _n_bytes){
    void* ptr;
    cudaError_t err = cudaMalloc(&ptr, _n_bytes);
    if(err != cudaSuccess){
        std::string s = cudaGetErrorString(err);
        throw std::runtime_error("Failed to allocate GPU memory");
    }
    return ptr;
}

void GpuDevice::free(void* ptr){
    cudaFree(ptr);
}

void GpuDevice::copy_to_host(void* host_dst, const void* src, size_t size){
    cudaError_t err = cudaMemcpy(host_dst, src, size, cudaMemcpyDeviceToHost);
    if(err != cudaSuccess){
        throw std::runtime_error("Failed to copy data from GPU to host");
    }
}

void GpuDevice::copy_to_device(void* dst, const void* host_src, size_t size){
    cudaError_t err = cudaMemcpy(dst, host_src, size, cudaMemcpyHostToDevice);
    if(err != cudaSuccess){
        throw std::runtime_error("Failed to copy data from host to GPU");
    }
}

void GpuDevice::copy_device_to_device(void* dst, const void* src, size_t size){
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    if(err != cudaSuccess){
        throw std::runtime_error("Failed to copy data within GPU");
    }
}
/*
    Sets the memory with n_floats floats
*/
void GpuDevice::memset(void* dst, float value, size_t n_floats){
    uint32_t asRaw = 0; 
    std::memcpy(&asRaw, &value, sizeof(float));
    
    CUresult err = cuMemsetD32(reinterpret_cast<CUdeviceptr>(dst), asRaw, n_floats);
    if(err != CUDA_SUCCESS){
        throw std::runtime_error("Failed to set GPU memory");
    }
}

namespace MyTensors::Hardware::Kernels{
    __global__ void rand_fill_kernel(
        float* dst,
        std::size_t n,
        unsigned long long seed
    ){
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= n) return;
        curandState state;
        curand_init(seed,idx,0,&state);
        dst[idx] = curand_uniform(&state);
    }

    __global__ void normal_rand_fill_kernel(
        float* dst,
        std::size_t n,
        float mean,
        float std_dev,
        unsigned long long seed
    ){
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= n) return;
        curandState state;
        curand_init(seed,idx,0,&state);
        dst[idx] = (curand_normal(&state) * std_dev) + mean;
    }

};


/*Set `dst` memory region holding `_n` floats with a random float value from [0,1]. 
`dst` is interpreted always as a float region of memory*/
void GpuDevice::memset_rand(void* dst, size_t _n){
    float* f_dst = static_cast<float*>(dst);
    constexpr std::size_t block_size = 256;
    std::size_t grid_size = (_n + block_size - 1) / block_size;
    unsigned long long seed = (unsigned long long)std::chrono::system_clock::now()
                                .time_since_epoch().count();
    LLCN_KERNELS::rand_fill_kernel<<<grid_size,block_size>>>(
        f_dst,
        _n,
        seed
    );

    auto result = cudaDeviceSynchronize();
    if(result != cudaSuccess){
        throw std::runtime_error("Error: Initializing memory with random values failed");
    }
    return;
}


namespace MyTensors::Hardware::Kernels{

    struct MakeContiguousParams{
        size_t shapes[TENSOR_MAX_DIM];
        size_t faulty_strides[TENSOR_MAX_DIM];

        MakeContiguousParams(
            const size_t* shapes,
            const size_t* faulty_strides
        ){
            for(size_t i = TENSOR_MAX_DIM; i --> 0;){
                this->shapes[i] = shapes[i];
                this->faulty_strides[i] = faulty_strides[i];
            }
        }

    };

    __global__ void make_contiguous_kernel(
        float* dst,
        const float* src,
        const std::size_t n_dim,
        const std::size_t n_elements,
        const MakeContiguousParams params
    ){
        std::size_t flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(flat_idx >= n_elements) return;
        std::size_t remaining = flat_idx;
        std::size_t coords[TENSOR_MAX_DIM] = {0};
        
        for( std::size_t i = n_dim; i-->0;){
            coords[i] = remaining % params.shapes[i];
            remaining/= params.shapes[i];
        }
        std::size_t src_idx = 0;
        for(std::size_t i = 0; i < n_dim; i++){
            src_idx += coords[i] * params.faulty_strides[i];
        }
        dst[flat_idx] = src[src_idx];
    }
}

void GpuDevice::memset_rand_normal_dist(void* dst, float mean, float std_dev, size_t _n){
    float* f_dst = static_cast<float*>(dst);
    constexpr std::size_t block_size = 256;
    std::size_t grid_size = (_n + block_size - 1) / block_size;
    unsigned long long seed = (unsigned long long)std::chrono::system_clock::now()
                                .time_since_epoch().count();
    LLCN_KERNELS::normal_rand_fill_kernel<<<grid_size,block_size>>>(
        f_dst,
        _n,
        mean,
        std_dev,
        seed
    );

    auto result = cudaDeviceSynchronize();
    if(result != cudaSuccess){
        throw std::runtime_error("Error: Initializing memory with random values failed");
    }
    return;
}

void GpuDevice::make_contiguous(
    void* dst, 
    const void* src, 
    const std::vector<std::size_t> shapes,
    const std::vector<std::size_t> faulty_strides
){
    using namespace LLCN_KERNELS;
    assert(shapes.size() == faulty_strides.size());
    float* f_dst = static_cast<float*>(dst);
    const float* f_src = static_cast<const float*>(src);

    std::size_t n_dim = shapes.size();
    std::size_t n_elements = 1;
    assert(n_dim <= TENSOR_MAX_DIM);
    //Compute total elements
    for(std::size_t i = 0; i < n_dim ; i++){
        n_elements *= shapes[i];
    }

    std::size_t block_size = 256;
    std::size_t grid_size = (n_elements + block_size - 1)/ block_size;

    MakeContiguousParams param = MakeContiguousParams(shapes.data(),faulty_strides.data()); 

    make_contiguous_kernel<<<grid_size,block_size>>>(
        f_dst,
        f_src,
        n_dim,
        n_elements,
        param
    );

    return;
}

DeviceType GpuDevice::get_type() const {
    return DeviceType::GPU;
}

std::string GpuDevice::get_name() const {
    return "GPU";
}

#endif //__USE_CUDA__