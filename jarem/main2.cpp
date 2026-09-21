// CNN demo

#include <SimpleNetwork.hpp>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <thread>

using MyTensors::Core::Tensor;
using MyTensors::Core::TENSOR_MAX_DIM;

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

int main( int argc, char* argv[] ){

    unsigned int n_threads = std::thread::hardware_concurrency();
    bool use_cpu = true;

    for(int i = 1; i < argc; i++){
        std::string arg = argv[i];
        if(arg == "--use-gpu"){
            use_cpu = false;
        }else if(arg == "--thread"){
            if(i + 1 >= argc){ throw std::runtime_error("Error: --thread requires a number"); }
            n_threads = static_cast<unsigned int>(std::stoul(argv[++i]));
        }
    }

    using MyTensors::Hardware::Device;
    using MyTensors::Hardware::DeviceManager;
    using MyTensors::Hardware::ThreadPool;

    std::shared_ptr<Device> device;
    if(use_cpu){
        auto pool = std::make_shared<ThreadPool>(n_threads);
        device = DeviceManager::get_cpu_device(pool);
    }else{
        device = DeviceManager::get_gpu_device();
    }
    

    std::vector<float> train_images_flat = read_all_floats(".\\temp\\mnist_train_images_f32.bin");
    std::vector<float> train_labels_flat = read_all_floats(".\\temp\\mnist_train_labels_onehot_f32.bin");
    std::vector<float> test_images_flat  = read_all_floats(".\\temp\\mnist_test_images_f32.bin");
    std::vector<float> test_labels_flat  = read_all_floats(".\\temp\\mnist_test_labels_onehot_f32.bin");

    if(train_images_flat.size() % IMG_PIXELS != 0 || train_labels_flat.size() % N_CLASSES != 0){
        throw std::runtime_error("Error: train data size is not a multiple of the expected sample size");
    }

    const std::size_t train_n = train_images_flat.size() / IMG_PIXELS;
    const std::size_t test_n  = test_images_flat.size() / IMG_PIXELS;

    //Conv2D -> ReLU -> Flatten -> FC -> Softmax+CrossEntropy
    SimpleNetwork net = SimpleNetworkBuilder(5)
        .with_layer<Conv2D>(
            std::array<std::size_t,4> {8,1,3,3},    //Outputr channel: 8, One channel, Kernel 3x3
            device,
            std::array<std::size_t,2> {0,0},    //No padding
            std::array<std::size_t,2> {1,1}     //Stride of 1
        )
        .with_layer<ActivationLayer>(
            device,
            ActivationTypes::ReLU
        )
        .with_layer<Flatten>(device)
        .with_layer<FullyConnected>(
            std::array<std::size_t,MyTensors::Core::TENSOR_MAX_DIM>{8*26*26,2},
            device
        )
        .with_layer<OutputLayer>(device,N_CLASSES)
        .build();
        
    net.he_initialize_all();

    constexpr std::size_t BATCH_SIZE = 50;
    constexpr std::size_t EPOCHS = 15;
    constexpr float LEARNING_RATE = 0.1f;
    const std::size_t n_batches = train_n / BATCH_SIZE;

    //Set up the tensor of losses, which always uses a cpu device
    std::shared_ptr<Device> dev_2;
    if(use_cpu){
        dev_2 = device;
    }else{
        auto pool = std::make_shared<ThreadPool>(2);
        dev_2 = DeviceManager::get_cpu_device(pool);
    }
    Tensor losses = Tensor({BATCH_SIZE},dev_2);

    Tensor batch_images({BATCH_SIZE,1,28,28},device);
    Tensor batch_labels({BATCH_SIZE,N_CLASSES},device);
    
    //Training with stochastic gradient descent
    net.set_mode(LayerMode::Training);
    //Measuring time
    std::cout<<"Starting training ..."<<std::endl;
    auto training_start = std::chrono::steady_clock::now();

    for(std::size_t epoch = 0; epoch < EPOCHS; epoch++){
        float epoch_loss = 0.0f;

        for(std::size_t b = 0; b < n_batches; b++){
            const float* img_ptr = train_images_flat.data() + b * BATCH_SIZE * IMG_PIXELS;
            const float* lbl_ptr = train_labels_flat.data() + b * BATCH_SIZE * N_CLASSES;

            batch_images.copy_from(img_ptr, BATCH_SIZE * IMG_PIXELS);
            batch_labels.copy_from(lbl_ptr, BATCH_SIZE * N_CLASSES);

            const Tensor& net_output = *(net.forward(batch_images,batch_labels));
            losses.copy_from(net_output);
            
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

    auto training_end = std::chrono::steady_clock::now();
    auto training_elapsed = training_end - training_start;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(training_elapsed);
    std::cout<<" Training finished in "<<ms.count()<<" miliseconds"<<std::endl;
    
    //---------------------------------------------------------------------------------------
    

    //Inference test on the held-out test set
    net.set_mode(LayerMode::Inference);

    Tensor test_batch_images({test_n,1,28,28},device);
    test_batch_images.copy_from(test_images_flat.data(), test_n * IMG_PIXELS);

    //Starting inference
    std::cout<<"Starting inference..."<<std::endl;
    auto inference_start = std::chrono::steady_clock::now();
    
    const Tensor& inf_out = *(net.forward(test_batch_images));
    
    auto inference_end = std::chrono::steady_clock::now();
    auto inference_elapsed = inference_end - inference_start;
    auto inference_ms = std::chrono::duration_cast<std::chrono::milliseconds>(inference_elapsed);
    std::cout<<" Inference of "<<test_n<<" images took "<<inference_ms.count()<<" miliseconds"<<std::endl;

    //Copy the results to a new tensor that is always on cpu
    Tensor probs = Tensor(inf_out,dev_2);
    const float* probs_data = probs.data();

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
