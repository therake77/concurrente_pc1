#include <SimpleNetwork.hpp>


int main(){
    using MyTensors::Hardware::Device;
    using MyTensors::Hardware::DeviceManager;
    using MyTensors::Hardware::CpuDevice;
    auto device = DeviceManager::get_cpu_device();
    
    
    SimpleNetwork net = SimpleNetworkBuilder(3)
        .with_layer<
            FullyConnected,
            std::array<std::size_t, MyTensors::Core::TENSOR_MAX_DIM>
        >(
            {2,2},
            device
        )
        .with_layer<
            FullyConnected,
            std::array<std::size_t, MyTensors::Core::TENSOR_MAX_DIM>
        >(
            {2,1},
            device
        )
        .build();
    

    

    
}