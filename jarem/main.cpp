#include <SimpleNetwork.hpp>
#include <fstream>

int main(){
    using MyTensors::Hardware::Device;
    using MyTensors::Hardware::DeviceManager;
    using MyTensors::Hardware::CpuDevice;
    auto device = DeviceManager::get_gpu_device();
    
    //Build the layer
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
    
    //Read the data
    const int num_images = 5;
    const int pixels_per_image = 28 * 28;
    Tensor data = Tensor(
        {1,5,28,28},
        device
    );
    Tensor labels = Tensor(
        {5},
        device
    );
    std::ifstream img_file(".\\temp\\mnist_images_f32.bin",std::ios::binary);
    std::ifstream lbl_file(".\\temp\\mnist_labels_i32.bin",std::ios::binary);
    img_file.read(
        reinterpret_cast<char*>(data.data()),data.shape.n_elements * sizeof(float)
    );
    lbl_file.read(
        reinterpret_cast<char*>(labels.data()),labels.shape.n_elements * sizeof(float)
    );
    img_file.close();
    lbl_file.close();


}