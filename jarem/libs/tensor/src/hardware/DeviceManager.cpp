#include <hardware/DeviceManager.hpp>
#include <hardware/CpuDevice.hpp>
#include <hardware/ThreadPool.hpp>
#include <stdexcept>

#ifdef __USE_CUDA__
    #include <hardware/GpuDevice.cuh>
#endif  //__USE_CUDA__

using MyTensors::Hardware::DeviceManager;
using MyTensors::Hardware::Device;
using MyTensors::Hardware::CpuDevice;
using MyTensors::Hardware::GpuDevice;
using MyTensors::Hardware::ThreadPool;

std::shared_ptr<Device> DeviceManager::get_cpu_device(
    std::shared_ptr<ThreadPool> pool
) {
    auto ptr = std::make_shared<CpuDevice>(pool);
    return ptr;
}   

std::shared_ptr<Device> DeviceManager::get_gpu_device() {
#ifdef __USE_CUDA__
    auto ptr = std::make_shared<GpuDevice>();
    return ptr;
#else
    throw std::runtime_error("GPU support is not available. CUDA compiler not found.");
#endif  //__USE_CUDA__
}