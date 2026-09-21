#include <SimpleNetwork.hpp>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <thread>

using MyTensors::Core::Tensor;
using MyTensors::Core::TENSOR_MAX_DIM;

//Digits used: see scripts/mnist_script.py (DIGIT_A -> class 0, DIGIT_B -> class 1)
constexpr std::size_t IMG_PIXELS = 28 * 28;
constexpr std::size_t N_CLASSES = 2;

std::vector<float> read_all_floats(const std::string& path){
    std::ifstream f(path, std::ios::binary);
    if(!f){ throw std::runtime_error("Cannot open file: " + path); }

    f.seekg(0, std::ios::end);
    std::size_t byte_size = static_cast<std::size_t>(f.tellg());
    f.seekg(0, std::ios::beg);

    std::vector<float> buffer(byte_size / sizeof(float));
    f.read(reinterpret_cast<char*>(buffer.data()), byte_size);
    return buffer;
}

int main(){
    using MyTensors::Hardware::Device;
    using MyTensors::Hardware::DeviceManager;
    using MyTensors::Hardware::ThreadPool;
    auto pool = std::make_shared<ThreadPool>(std::thread::hardware_concurrency());
    auto device = DeviceManager::get_cpu_device(pool);

    std::vector<float> train_images_flat = read_all_floats(".\\temp\\mnist_train_images_f32.bin");
    std::vector<float> train_labels_flat = read_all_floats(".\\temp\\mnist_train_labels_onehot_f32.bin");
    std::vector<float> test_images_flat  = read_all_floats(".\\temp\\mnist_test_images_f32.bin");
    std::vector<float> test_labels_flat  = read_all_floats(".\\temp\\mnist_test_labels_onehot_f32.bin");

    if(train_images_flat.size() % IMG_PIXELS != 0 || train_labels_flat.size() % N_CLASSES != 0){
        throw std::runtime_error("Error: train data size is not a multiple of the expected sample size");
    }

    const std::size_t train_n = train_images_flat.size() / IMG_PIXELS;
    const std::size_t test_n  = test_images_flat.size() / IMG_PIXELS;

    //Flatten -> FC(784,32) -> ReLU -> FC(32,2) -> OutputLayer(2)
    SimpleNetwork net = SimpleNetworkBuilder(5)
        .with_layer<
            Flatten
        >(
            device
        )
        .with_layer<
            FullyConnected,
            std::array<std::size_t, TENSOR_MAX_DIM>
        >(
            {IMG_PIXELS,32},
            device
        )
        .with_layer<
            ActivationLayer
        >(
            device,
            ActivationTypes::ReLU
        )
        .with_layer<
            FullyConnected,
            std::array<std::size_t, TENSOR_MAX_DIM>
        >(
            {32,N_CLASSES},
            device
        )
        .with_layer<
            OutputLayer
        >(
            device,
            N_CLASSES
        )
        .build();

    net.he_initialize_all();

    constexpr std::size_t BATCH_SIZE = 50;
    constexpr std::size_t EPOCHS = 15;
    constexpr float LEARNING_RATE = 0.1f;
    const std::size_t n_batches = train_n / BATCH_SIZE;
    Tensor batch_images({BATCH_SIZE,28,28},device);
    Tensor batch_labels({BATCH_SIZE,N_CLASSES},device);

    net.set_mode(LayerMode::Training);

    for(std::size_t epoch = 0; epoch < EPOCHS; epoch++){
        float epoch_loss = 0.0f;

        for(std::size_t b = 0; b < n_batches; b++){
            const float* img_ptr = train_images_flat.data() + b * BATCH_SIZE * IMG_PIXELS;
            const float* lbl_ptr = train_labels_flat.data() + b * BATCH_SIZE * N_CLASSES;

            batch_images.copy_from(img_ptr, BATCH_SIZE * IMG_PIXELS);
            batch_labels.copy_from(lbl_ptr, BATCH_SIZE * N_CLASSES);

            const Tensor& losses = *(net.forward(batch_images,batch_labels));

            const float* loss_data = losses.data();
            float batch_loss = 0.0f;
            for(std::size_t i = 0; i < BATCH_SIZE; i++){ batch_loss += loss_data[i]; }
            epoch_loss += batch_loss / (float)BATCH_SIZE;

            net.backwards(batch_labels,batch_images);
            net.update_weights(-LEARNING_RATE);
        }

        epoch_loss /= (float)n_batches;
        std::cout << "Epoch " << (epoch + 1) << "/" << EPOCHS << " - avg loss: " << epoch_loss << std::endl;
    }

    //Inference test on the held-out test set
    net.set_mode(LayerMode::Inference);

    Tensor test_batch_images({test_n,28,28},device);
    test_batch_images.copy_from(test_images_flat.data(), test_n * IMG_PIXELS);

    const Tensor& probs = *(net.forward(test_batch_images));
    const float* probs_data = probs.data();

    MyTensors::Util::tensor_print(probs);

    std::size_t correct = 0;
    for(std::size_t i = 0; i < test_n; i++){
        float p0 = probs_data[i * N_CLASSES + 0];
        float p1 = probs_data[i * N_CLASSES + 1];
        std::size_t predicted = (p1 > p0) ? 1 : 0;
        std::size_t truth = (test_labels_flat[i * N_CLASSES + 1] > 0.5f) ? 1 : 0;
        if(predicted == truth){ correct++; }
    }

    float accuracy = 100.0f * (float)correct / (float)test_n;
    std::cout << "Test accuracy: " << accuracy << "% (" << correct << "/" << test_n << ")" << std::endl;

    return 0;
}
