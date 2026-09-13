#include <hardware/DeviceManager.hpp>
#include <hardware/CpuDevice.hpp>
#include <stdexcept>

#ifdef __USE_CUDA__
    #include <hardware/GpuDevice.cuh>
#endif  //__USE_CUDA__

using namespace MyTensors::Hardware;

std::shared_ptr<Device> DeviceManager::get_cpu_device() {
    auto ptr = std::make_shared<CpuDevice>();
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