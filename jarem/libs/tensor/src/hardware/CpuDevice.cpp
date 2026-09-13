#include <hardware/CpuDevice.hpp>
#include <stdexcept>
#include <cstring>
#include <random>
#include <cassert>

namespace{
    bool _increment_odometer(std::vector<size_t>& coords, std::vector<size_t>& shape){
        for(size_t i = coords.size(); i --> 0 ;){
            //guard against overflow
            if(coords[i] == SIZE_MAX){ throw std::runtime_error("Error: Overflow detected"); }
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

    std::size_t _compute_idx_from_coords(std::vector<size_t>& coords, std::vector<size_t>& strides ){
        std::size_t offset = 0;
        for(std::size_t i = 0; i < coords.size(); i++){
            offset += (coords[i] * strides[i]);
        }
        return offset;
    }
};

using namespace MyTensors::Hardware;

void* CpuDevice::allocate(size_t _n_bytes){
    return std::malloc(_n_bytes);
}

void CpuDevice::free(void* ptr){
    std::free(ptr);
}

void CpuDevice::copy_to_host(void* host_dst, const void* src, size_t size){
    std::memcpy(host_dst, src, size);
}

void CpuDevice::copy_to_device(void* dst, const void* host_src, size_t size){
    std::memcpy(dst, host_src, size);
}

void CpuDevice::copy_device_to_device(void* dst, const void* src, size_t size){
    std::memcpy(dst, src, size);
}

/*Set `dst` memory region holding `_n` floats with a fixed float value. 
`dst` is interpreted always as a float region of memory*/
void CpuDevice::memset(void* dst, float value, size_t _n){
    std::fill_n(static_cast<float*>(dst), _n,value);
}

/*Set `dst` memory region holding `_n` floats with a random float value from [0,1]. 
`dst` is interpreted always as a float region of memory*/
void CpuDevice::memset_rand(void* dst, size_t _n){
    float* f_dst = static_cast<float*>(dst);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f,1.0f);

    for(size_t i = 0 ; i < _n; i++){
        f_dst[i] = dist(gen); 
    }
}

void CpuDevice::memset_rand_normal_dist(void* dst, float mean, float std_dev, size_t _n){
    float* f_dst = static_cast<float*>(dst);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(mean, std_dev);

    for(size_t i = 0 ; i < _n; i++){
        f_dst[i] = dist(gen); 
    }
}

void CpuDevice::make_contiguous(
    void* dst, 
    const void* src, 
    std::vector<std::size_t> shapes,
    std::vector<std::size_t> faulty_strides
){
    assert(shapes.size() == faulty_strides.size());
    std::size_t n_dim = shapes.size();
    if(n_dim == 0){ return; }
    std::vector<std::size_t> coords(n_dim,0);
    //Compute correct strides
    std::vector<std::size_t> dst_strides(n_dim,0);

    float* dst_float = static_cast<float*>(dst);
    const float* src_float = static_cast<const float*>(src);
    
    dst_strides[n_dim-1] = 1;    
    for(size_t i = n_dim - 1; i --> 0; ){
        dst_strides[i] =  shapes[i+1] * dst_strides[i+1];
    }
    //Copy data
    do{
        std::size_t dst_idx = _compute_idx_from_coords(
            coords,
            dst_strides
        );
        std::size_t src_idx = _compute_idx_from_coords(
            coords,
            faulty_strides
        );

        dst_float[dst_idx] = src_float[src_idx];
    }while(_increment_odometer(coords,shapes));
}

DeviceType CpuDevice::get_type() const {
    return DeviceType::CPU;
}

std::string CpuDevice::get_name() const {
    return "CPU";
}