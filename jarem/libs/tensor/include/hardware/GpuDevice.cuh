#pragma once
#include <hardware/Device.hpp>
#include <math/GpuMath.cuh>
#ifdef __USE_CUDA__
namespace MyTensors::Hardware{
    class GpuDevice : public Device {
    public:

        ~GpuDevice() override = default;

        GpuDevice(){ this->math = std::make_unique<MyTensors::Math::Base::GpuMath>(); }

        void* allocate(size_t size) override;
        void free(void* ptr) override;
        
        void copy_to_host(void* host_dst, const void* src, size_t size) override;
        void copy_to_device(void* dst, const void* host_src, size_t size) override;
        void copy_device_to_device(void* dst, const void* src, size_t size) override;

        void memset(void* dst, float value, size_t size) override;
        void memset_rand(void*dst, size_t size) override;

        void memset_rand_normal_dist( void* dst, float mean, float std_dev, size_t n_floats ) override ;

        void make_contiguous(
            void* dst, 
            const void* src, 
            const std::vector<std::size_t> shapes,
            const std::vector<std::size_t> faulty_strides
        );

        DeviceType get_type() const override;
        std::string get_name() const override;
    };
}

#endif // __USE_CUDA__