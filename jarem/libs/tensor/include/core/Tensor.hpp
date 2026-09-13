

#include <memory>
#include <vector>
#include <initializer_list>
#include <hardware/Device.hpp>
#include <hardware/DeviceManager.hpp>
#include <functional>
#include <array>
#include <core/Shape.hpp>
    

namespace MyTensors::Core{
    using MyTensors::Hardware::Device;
    using MyTensors::Hardware::DeviceManager;
    /*
        Clase contenedora de datos. Representa un tensor de como máximo TENSOR_MAX_DIM dimensiones (x,y,z,w)
    */
    
    class Tensor{
    private:
        std::shared_ptr<Device> device;
        std::unique_ptr<void, std::function<void(void*)>> cpu_data;
        std::unique_ptr<void, std::function<void(void*)>> gpu_data;

    public:
        Shape shape;
        Tensor(std::array<size_t,TENSOR_MAX_DIM>, std::shared_ptr<Device> = DeviceManager::get_cpu_device(), float = 0.0f, bool = false);    
        Tensor(std::array<size_t,TENSOR_MAX_DIM>, const float*, std::shared_ptr<Device> = DeviceManager::get_cpu_device());
        Tensor(const Tensor&, const std::shared_ptr<Device> = DeviceManager::get_cpu_device());
        Tensor(Tensor&&) noexcept;
        Tensor& operator=(Tensor&&) noexcept;
        float* data();
        const float* data() const;
        void to(std::shared_ptr<Device>);
        const inline std::shared_ptr<Device>& getDevice() const { return this->device; }
        void resize(Shape s);
        void reshape(std::array<std::size_t, TENSOR_MAX_DIM> new_shape);
        void copy_from(const Tensor& t);
        void copy_from(const float* src, std::size_t count);
        void zeros();
        void normal_rand(float mean, float std_dev);
        Tensor& unsqueeze(std::size_t axis);
        void inner_expand(std::size_t dims);
        void outer_expand(std::size_t dims);
        void contiguous();
    };
}
    

