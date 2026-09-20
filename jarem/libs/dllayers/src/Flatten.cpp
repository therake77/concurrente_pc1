#include <Flatten.hpp>

using MyTensors::Core::Tensor;
using MyTensors::Core::Shape;
using MyTensors::Hardware::Device;

Flatten::Flatten(
    std::shared_ptr<Device> dev
) : Layer(dev){}

/* @brief Flattens a {N,...dims} tensor into a tensor of shape {N,M}, preserving the total number of elements */
std::shared_ptr<Tensor> Flatten::forward( const Tensor& in ){
    if(in.shape.n_dim > 2){ 
        std::size_t M = 1;
        //Compute last dimension
        for(std::size_t i = 1; i < in.shape.n_dim; i++){
            M*=in.shape[i];
        }

        //Copy in to the forward_out buffer
        __check_initialization(
            this->_forward_out,
            Shape({in.shape[0],M})
        );
    }else{
        __check_initialization(
            this->_forward_out,
            in.shape
        );
    }
    
    this->_forward_out->copy_from(in);

    return this->_forward_out;
}

std::shared_ptr<Tensor> Flatten::backward( const Tensor& backward_grad, const Tensor& forward_in ){
    __check_initialization(this->_backward_out,forward_in.shape);
    this->_backward_out->copy_from(backward_grad);
    return this->_backward_out;
}

void Flatten::update_weight(float){
    return;
}
