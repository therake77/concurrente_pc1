#pragma once
#include <hardware/Device.hpp>
#include <math/CpuMath.hpp>


namespace MyTensors::Hardware{
    class CpuDevice : public Device {
    public:
        ~CpuDevice() override = default;

        CpuDevice(){ this->math = std::make_unique<MyTensors::Math::Base::CpuMath>(); };

        /*
            @brief Allocates `_bytes` bytes in memory
        */
        void* allocate(size_t _bytes) override;
        
        /*
            @brief Deallocates a pointer
        */
        void free(void* ptr) override;
        
        void copy_to_host(void* host_dst, const void* src, size_t size) override;
        void copy_to_device(void* dst, const void* host_src, size_t size) override;
        void copy_device_to_device(void* dst, const void* src, size_t size) override;


        /*
            @brief Sets `dst` with `n` floats with value `value`
            @note `n` is the number of floats, not the number of bytes
        */
        void memset(void* dst, float value, size_t n) override;

        
        /*
            @brief Sets `dst` with `n` floats randomly generated
            @note `n` is the number of floats, not the number of bytes
        */
        void memset_rand(void*dst, size_t n) override;

        void memset_rand_normal_dist( void* dst, float mean, float std_dev, size_t n_bytes ) override ;

        /*
            @brief Copy the contents of src, making it contiguous according to `shapes`
        */
        void make_contiguous(
            void* dst, 
            const void* src, 
            std::vector<std::size_t> shapes,
            std::vector<std::size_t> faulty_strides
        ) override;

        DeviceType get_type() const override;
        std::string get_name() const override;
    };
}