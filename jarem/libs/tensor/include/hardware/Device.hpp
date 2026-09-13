#pragma once
#include <memory>
#include <string>
#include <math/Math.hpp>
#include <math/CpuMath.hpp>

#ifdef __USE_CUDA__
    #include <math/GpuMath.cuh>
#endif

namespace MyTensors::Hardware{
    using MyTensors::Math::Base::Math;
    enum class DeviceType {
        CPU,
        GPU
    };

    class Device {
    public:
        std::unique_ptr<Math> math;

        virtual ~Device() = default;
        Device() = default;

        /*
            @brief Allocates `_bytes` bytes in memory
        */
        virtual void* allocate(size_t _bytes) = 0;
        virtual void free(void* ptr) = 0;
        
        virtual void copy_to_host(void* host_dst, const void* src, size_t size) = 0;
        virtual void copy_to_device(void* dst, const void* host_src, size_t size) = 0;
        virtual void copy_device_to_device(void* dst, const void* src, size_t size) = 0;

        virtual void make_contiguous(
            void* dst, 
            const void* src, 
            std::vector<std::size_t> shapes,
            std::vector<std::size_t> faulty_strides
        ) = 0;

        /*
            @brief Sets `dst` with `n` floats with value `value`
            @note `n` is the number of floats, not the number of bytes
        */
        virtual void memset(void* dst, float value, size_t n_bytes) = 0;
        /*
            @brief Sets `dst` with `n` floats randomly generated from a normal distribution of 0,1
            @note `n` is the number of floats, not the number of bytes
        */
        virtual void memset_rand(void*dst, size_t size) = 0;

        virtual void memset_rand_normal_dist( void* dst, float mean, float std_dev, size_t _n ) = 0;

        virtual DeviceType get_type() const = 0;
        virtual std::string get_name() const = 0;
    };
}

