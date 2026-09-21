#include <MyTensors.hpp>


int main(){

    using MyTensors::Core::Tensor;
    using MyTensors::Hardware::ThreadPool;
    using MyTensors::Hardware::DeviceManager;
    
    std::shared_ptr<ThreadPool> pool = std::make_unique<ThreadPool>(
        std::thread::hardware_concurrency()
    );

    auto device = DeviceManager::get_cpu_device(pool);

    Tensor a = Tensor(
        {1,2,1,2},
        device,
        1.0f,
        true
    );

    Tensor b = Tensor(
        {2,2,2},
        device,
        1.0f,
        true
    );

    Tensor c = Tensor(
        {1,2,2,2},
        device,
        1.0f,
        true
    );

    MyTensors::Math::TensorMath::add(
        a,b,c
    );

    MyTensors::Util::tensor_print(c);
    return 0;
}