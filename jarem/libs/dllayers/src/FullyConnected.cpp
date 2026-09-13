#include <FullyConnected.hpp>
#include <MyTensors.hpp>
#include <stdexcept>
#include <cassert>
#include <cmath>

/*
    @brief Creates a dense layer.
    @param shape Array of shapes {`in`,`out`}
    @param dev Layer device
*/
using MyTensors::Core::Tensor;
using MyTensors::Core::Shape;
using MyTensors::Hardware::Device;
using MyTensors::Core::TENSOR_MAX_DIM;

FullyConnected::FullyConnected(
    std::array<std::size_t, TENSOR_MAX_DIM> shape,
    std::shared_ptr<Device> dev
) : _weights(shape,dev),_bias({1,shape[1]},dev,0.0f,true), Layer(dev) {
    if(Shape(shape).n_dim != 2){ throw std::runtime_error("Error: Only valid 2-dimensional shapes"); }
    this->input_dim = shape[0];
    this->output_dim = shape[1];
    this->_bias_grad = std::make_unique<Tensor>(std::array<std::size_t,TENSOR_MAX_DIM>{1,this->output_dim},dev,0.0f,true);
}

std::shared_ptr<Tensor> FullyConnected::forward(const Tensor& in){
    //IMPORTANT: in is 2-dimensional (N,F) (batch, flatten)
    //Well, everyone knows the forward pass Y= XW + b
    __check_initialization(this->_forward_out,Shape({in.shape[0],this->output_dim}));
    MyTensors::Math::TensorMath::matmul(
        in,
        this->_weights,
        *(this->_forward_out)
    );
    MyTensors::Math::TensorMath::add(
        this->_bias,
        *(this->_forward_out),
        *(this->_forward_out)
    );
    return this->_forward_out;
}

std::shared_ptr<Tensor> FullyConnected::backward(const Tensor& backward_grad, const Tensor& forward_input){
    assert(backward_grad.shape.n_dim == 2);
    __check_initialization(this->_bias_grad,Shape({this->output_dim}));
    __check_initialization(this->_grad,this->_weights.shape);
    __check_initialization(this->_backward_out,Shape({forward_input.shape[0],this->input_dim}));
    //Copy the original input
    Tensor forward_input_copy = Tensor(forward_input,this->_dev);
    forward_input_copy.shape.transpose();
    //Compute weight gradient
    MyTensors::Math::TensorMath::matmul(
        forward_input_copy,
        backward_grad,
        *(this->_grad)
    );
    this->_bias_grad->reshape({this->output_dim});
    //Compute beta gradient. TensorMath::sum accumulates (+=), so zero first, otherwise this
    //picks up leftover noise/stale gradient from construction or the previous batch.
    this->_bias_grad->zeros();
    MyTensors::Math::TensorMath::sum(
        backward_grad,
        0,
        *(this->_bias_grad)
    );

    //Compute output gradient
    this->_weights.shape.transpose();
    MyTensors::Math::TensorMath::matmul(
        backward_grad,
        this->_weights,
        *(this->_backward_out)
    );
    this->_weights.shape.transpose();
    return this->_backward_out;
}


void FullyConnected::update_weight(float lr){
    assert(this->_bias_grad.get() != nullptr);
    assert(this->_grad.get() != nullptr);
    using namespace MyTensors::Math;
    TensorMath::scale(
        *(this->_grad),
        lr,
        *(this->_grad)
    );
    TensorMath::add(
        this->_weights,
        *(this->_grad),
        this->_weights
    );
    TensorMath::scale(
        *(this->_bias_grad),
        lr,
        *(this->_bias_grad)
    );
    TensorMath::add(
        this->_bias,
        *(this->_bias_grad),
        this->_bias
    );
    return;
}

void FullyConnected::he_initialization(){
    std::size_t fan_in = this->input_dim;
    float std_dev = std::sqrt(2.0f/(float)fan_in);
    this->_weights.normal_rand(0.0f,std_dev);
    this->_bias.zeros();
}
