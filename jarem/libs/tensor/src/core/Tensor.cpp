#include <core/Tensor.hpp>
#include <stdexcept>
#include <cstring>
#include <iostream>

using namespace MyTensors::Core;
using namespace MyTensors::Hardware;
/*Creates a tensor of shape (m,n,l,..,z,y,x) where `m` is the outermost dimension and `x` the innermost dimension. 
An optional value to fill the tensor can be specified, defaults to 0.0f */
Tensor::Tensor(
    std::array<size_t,TENSOR_MAX_DIM> _shape, 
    std::shared_ptr<Device> device, 
    float _fill,
    bool _fixed_value
)
: shape(_shape), device(device), cpu_data(nullptr, [](void* ptr){}), gpu_data(nullptr, nullptr){
    
    auto currDev = this->device.get();
    void* data_ptr = currDev->allocate(shape.n_elements * sizeof(float));
    if( data_ptr == nullptr){
        throw std::runtime_error("Failed memory allocation");
    }
    
    if(_fixed_value) currDev->memset(data_ptr,_fill,shape.n_elements);
    else currDev->memset_rand(data_ptr,shape.n_elements);
    
    auto deleter =  [currDev](void* p){
        currDev->free(p);
    };

    DeviceType type = this->device.get()->get_type();
    if(type == DeviceType::CPU){
        this->cpu_data = std::unique_ptr<void, std::function<void(void*)>>(data_ptr,deleter);
    }else{
        this->gpu_data = std::unique_ptr<void, std::function<void(void*)>>(data_ptr,deleter);
    }

}

Tensor::Tensor(
    std::array<size_t,TENSOR_MAX_DIM> _shape,
    const float* src,
    std::shared_ptr<Device> device
)
: shape(_shape), device(device), cpu_data(nullptr, [](void* ptr){}), gpu_data(nullptr, nullptr) {
    if(src == nullptr) {
        throw std::runtime_error("Error: Tensor source pointer is null");
    }

    auto currDev = this->device.get();
    void* data_ptr = currDev->allocate(shape.n_elements * sizeof(float));
    if(data_ptr == nullptr){
        throw std::runtime_error("Failed memory allocation");
    }

    //If device is CPU, then is memcpy, else is cudaMemcpy. device handles that
    this->device->copy_to_device(data_ptr, src, shape.n_elements * sizeof(float));

    auto deleter = [currDev](void* p){
        currDev->free(p);
    };

    if(this->device->get_type() == DeviceType::CPU){
        this->cpu_data = std::unique_ptr<void, std::function<void(void*)>>(data_ptr, deleter);
    }else{
        this->gpu_data = std::unique_ptr<void, std::function<void(void*)>>(data_ptr, deleter);
    }
}


Tensor::Tensor(const Tensor& t, const std::shared_ptr<Device> device) : device(device), shape(t.shape._shapes), cpu_data(nullptr, [](void*){}), gpu_data(nullptr, [](void*){}) {

    void* data_ptr = this->device->allocate(shape.n_elements * sizeof(float));
    if(data_ptr == nullptr){
        throw std::runtime_error("Failed memory allocation");
    }


    auto src_type = t.getDevice().get()->get_type();
    auto dst_type = this->device->get_type();

    size_t _size = shape.n_elements;

    if(src_type == dst_type){
        this->device->copy_device_to_device(data_ptr,t.data(), _size * sizeof(float));
    }else if(src_type == DeviceType::GPU){
        // This device is CPU and the source device is GPU
        // Then, copy from GPU to CPU
        t.getDevice()->copy_to_host(data_ptr,t.data(),_size * sizeof(float));
    }else{
        // This device is GPU and the source device is CPU
        // Then, copy from CPU to GPU using the GPU device (this)
        this->device->copy_to_device(data_ptr,t.data(),_size * sizeof(float));
    }

    auto curr_dev = this->device;
    auto deleter = [curr_dev](void* ptr){
        curr_dev->free(ptr);
    };

    if(this->device->get_type() == DeviceType::CPU){
        this->cpu_data = std::unique_ptr<void,std::function<void(void*)>>(data_ptr, deleter);
        this->gpu_data.reset();
    }else{
        this->gpu_data = std::unique_ptr<void,std::function<void(void*)>>(data_ptr, deleter);
        this->cpu_data.reset();
    }
}

Tensor::Tensor(Tensor&& other) noexcept
    : device(std::move(other.device)),
      cpu_data(std::move(other.cpu_data)),
      gpu_data(std::move(other.gpu_data)),
      shape(other.shape) {}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if(this != &other){
        device   = std::move(other.device);
        cpu_data = std::move(other.cpu_data);
        gpu_data = std::move(other.gpu_data);
        shape    = other.shape;
    }
    return *this;
}

float* Tensor::data(){
    if( this->device.get()->get_type() == DeviceType::CPU){
        return static_cast<float*>(this->cpu_data.get());
    }else{
        return static_cast<float*>(this->gpu_data.get());
    }
}

const float* Tensor::data() const{
    if( this->device.get()->get_type() == DeviceType::CPU){
        return static_cast<float*>(this->cpu_data.get());
    }else{
        return static_cast<float*>(this->gpu_data.get());
    }
}

//TODO(Still tests are needed)
void Tensor::to(std::shared_ptr<Device> device){
    if(this->device->get_type() == device->get_type()) return;
    
    size_t _size = shape.n_elements;

    void* data_ptr = device->allocate(_size * sizeof(float));
    if(data_ptr == nullptr){
        throw std::runtime_error("Failed memory allocation");
    }
    
    auto curr_dev = device;
    auto deleter = [curr_dev](void* ptr){
        curr_dev->free(ptr);
    };

    if(device->get_type() == DeviceType::CPU){
        //Data is in GPU, copy to CPU
        this->device->copy_to_host(data_ptr, this->gpu_data.get(), _size * sizeof(float));
        this->cpu_data = std::unique_ptr<void,std::function<void(void*)>>(data_ptr, deleter);
        this->gpu_data.reset();
    }else{
        //Data is in CPU, copy to GPU
        device->copy_to_device(data_ptr, this->cpu_data.get(), _size * sizeof(float));
        this->gpu_data = std::unique_ptr<void,std::function<void(void*)>>(data_ptr, deleter);
        this->cpu_data.reset();
    }
    this->device = device;
}

//TODO(Check correct management of memory)
void Tensor::resize(Shape s){
    //Checking if the number of elements remains the same (just a reshape)
    bool same_n_elements = s.n_elements == this->shape.n_elements;
    this->shape = s;
    if(same_n_elements){
        return;
    }

    //Not a simple reshape: Allocate fresh memory
    void* new_ptr = this->device->allocate(s.n_elements * sizeof(float));
    if(new_ptr == nullptr){
        throw std::runtime_error("Failed memory allocation");
    }
    
    auto currDev = this->device;
    auto deleter =  [currDev](void* p){
        currDev->free(p);
    };
    currDev->memset(new_ptr,0.0f,s.n_elements);
    //Drop the old data, assign new unique_ptr to the corresponding data
    switch(device->get_type()){
        case(DeviceType::CPU) : {
            this->cpu_data = std::unique_ptr<void, std::function<void(void*)>>(
                new_ptr,
                deleter
            );
            break;
        }
        case(DeviceType::GPU) : {
            this->gpu_data = std::unique_ptr<void, std::function<void(void*)>>(
                new_ptr,
                deleter
            );
            break;
        }
    }

    this->shape = s;
}

void Tensor::reshape(std::array<std::size_t, TENSOR_MAX_DIM> new_shape) {
    Shape s = Shape(new_shape);
    if(s.n_elements != this->shape.n_elements){
        throw std::runtime_error("Error: Impossible to reshape");
    }else{
        this->shape = s;
    }
    return;
}

void Tensor::copy_from(const Tensor& t){
    
    if( this->shape.n_elements != t.shape.n_elements){
        throw std::runtime_error("Error: Dangerous copy");
    }
    std::size_t total_bytes = this->shape.n_elements * sizeof(float);
    auto data_ptr = this->data();

    auto src_type = t.getDevice().get()->get_type();
    auto dst_type = this->device->get_type();

    if(src_type == dst_type){
        this->device->copy_device_to_device(data_ptr,t.data(), total_bytes);
    }else if(src_type == DeviceType::GPU){
        // This device is CPU and the source device is GPU
        // Then, copy from GPU to CPU
        t.getDevice()->copy_to_host(data_ptr,t.data(),total_bytes);
    }else{
        // This device is GPU and the source device is CPU
        // Then, copy from CPU to GPU using the GPU device (this)
        this->device->copy_to_device(data_ptr,t.data(),total_bytes);
    }

}

void Tensor::copy_from(const float* src, std::size_t count){
    if(src == nullptr){
        throw std::runtime_error("Error: Tensor source pointer is null");
    }

    if(count != this->shape.n_elements){
        throw std::runtime_error("Error: Tensor pointer copy size mismatch");
    }

    std::size_t total_bytes = count * sizeof(float);
    if(this->device->get_type() == DeviceType::CPU){
        std::memcpy(this->cpu_data.get(), src, total_bytes);
    }else{
        this->device->copy_to_device(this->gpu_data.get(), src, total_bytes);
    }
}

void Tensor::zeros(){
    auto device = this->device;
    switch(this->device->get_type()){
        case DeviceType::CPU:
            device->memset(
                this->cpu_data.get(),
                0.0f,
                this->shape.n_elements
            );
            break;
        case DeviceType::GPU:
            device->memset(
                this->gpu_data.get(),
                0.0f,
                this->shape.n_elements
            );
            break;
        default:
            throw std::runtime_error("Error: Unexpected device type ");
            break;
    }
}

void Tensor::normal_rand(float mean, float std_dev){
    /*
        Straightforward: Call device->normal_rand_... over the current pointer
    */
   auto device = this->device;
   auto deviceType = device->get_type();
   switch(deviceType){
        case DeviceType::CPU:
            device->memset_rand_normal_dist(this->cpu_data.get(),mean,std_dev,this->shape.n_elements);
            break;
        case DeviceType::GPU:
            device->memset_rand_normal_dist(this->gpu_data.get(),mean,std_dev,this->shape.n_elements);
            break;
        default:
            std::runtime_error("Unexpected device");
            break;
   }
}

Tensor& Tensor::unsqueeze(std::size_t axis){
    this->shape = this->shape.unsqueeze(axis);
    return *this;
}

void Tensor::inner_expand(std::size_t dims){
    this->shape.inner_pad(dims);
    return;
}

void Tensor::outer_expand(std::size_t dims){
    this->shape.outer_pad(dims);
    return;
}

void Tensor::contiguous(){
    auto curr_dev = this->device;
    if(this->shape.is_contiguous()){
        return;
    }else{
        std::vector<std::size_t> shapes(this->shape._shapes.begin(),this->shape._shapes.begin()+this->shape.n_dim);
        std::vector<std::size_t> strides(this->shape._strides.begin(),this->shape._strides.begin()+this->shape.n_dim);
        void* new_data_ptr = this->device->allocate(this->shape.n_elements * sizeof(float));
        auto deleter = [curr_dev](void* ptr){
            curr_dev->free(ptr);
        };
        
        switch(this->device->get_type()){
            case DeviceType::CPU:
                this->device->make_contiguous(
                    new_data_ptr,
                    this->cpu_data.get(),
                    shapes,
                    strides
                );
                this->cpu_data = std::unique_ptr<void,std::function<void(void*)>>(
                    new_data_ptr,deleter
                );
                break;
            case DeviceType::GPU:
                this->device->make_contiguous(
                    new_data_ptr,
                    this->gpu_data.get(),
                    shapes,
                    strides
                );
                this->gpu_data = std::unique_ptr<void,std::function<void(void*)>>(
                    new_data_ptr,deleter
                );
                break;
            default:
                break;
        }
    }
}
