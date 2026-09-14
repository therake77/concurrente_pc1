#include <ActivationLayer.hpp>
#include <functional>
#include <stdexcept>
#include <math/TensorMath.hpp>
#include <cassert>

namespace{
    constexpr float LEAKY_RELU_ALPHA = 0.01f;
};

ActivationLayer::ActivationLayer(
    ActivationTypes _type,
    std::shared_ptr<Device> dev 
) : type(_type), Layer(dev) {}

using MyTensors::Core::Tensor;
using MyTensors::Math::TensorMath;
std::shared_ptr<Tensor> ActivationLayer::forward( const Tensor& in ){
    /*Because all of them share the shape, we can do this*/
    __check_initialization(this->_forward_out,in.shape);
    __check_initialization(this->_backward_out,in.shape);
    __check_initialization(this->_grad,in.shape);

    /*Now, apply the operation*/
    Tensor& forward_out = *(this->_forward_out);
    switch(this->type){
        case(ActivationTypes::ReLU) : {
            TensorMath::reLU(
                in,
                forward_out
            );
            break;
        };
        case(ActivationTypes::Sigmoid) : {
            TensorMath::sigmoid(
                in,
                forward_out
            );
            break;
        };
        case(ActivationTypes::LeakyReLU) : {
            //LeakyReLU(x) = ReLU(x) + alpha*(x - ReLU(x))
            //(x - ReLU(x)) is x for x<=0 and 0 for x>0, so this is x for x>0 and alpha*x for x<=0
            Tensor relu_x = Tensor(in.shape._shapes, this->_dev);
            TensorMath::reLU(in, relu_x);
            TensorMath::sub(in, relu_x, forward_out);
            TensorMath::scale(forward_out, LEAKY_RELU_ALPHA, forward_out);
            TensorMath::add(forward_out, relu_x, forward_out);
            break;
        };
        default :
            throw std::runtime_error("A unexpected type recevied");
            break;
    }

    return this->_forward_out;
}

std::shared_ptr<Tensor> ActivationLayer::backward( const Tensor& backward_grad, const Tensor& forward_input ){
    assert(this->_backward_out.get() != nullptr && this->_grad.get() != nullptr && this->_forward_out.get() != nullptr );
    //In is coming from the next layer
    //This layer does not need to store a _grad. It has no weights
    Tensor& backward_out = *(this->_backward_out);
    switch(this->type){
        case ActivationTypes::ReLU:
            //Compute a binary mask based on the original forward input of this layer
            TensorMath::binary_positive_mask(
                forward_input,
                backward_out
            );

            //Perform element_wise multiplication on the mask with the incoming gradient
            TensorMath::multiply(
                backward_grad,
                backward_out,
                backward_out
            );
            break;
        case ActivationTypes::Sigmoid:{
            //Compute 1 - \sigma(x) = 1 - forward_out. store it in backward_out
            Tensor temp = Tensor({1},this->_dev,1.0,true);
            Tensor& forward_out = *(this->_forward_out);
            TensorMath::sub(
                temp,
                forward_out,
                backward_out
            );
            //Compute \sigma(x)(1-\sigma(x)) = forward_out * backward_out
            TensorMath::multiply(
                forward_out,
                backward_out,
                backward_out
            );
            //Compute dL/dy * \sigma(x)*(1-\sigma(x)) = backward_grad * backward_out
            //And store it in backward_out
            TensorMath::multiply(
                backward_out,
                backward_grad,
                backward_out
            );
            break;
        }
        case ActivationTypes::LeakyReLU:{
            //Derivative of LeakyReLU(x) is 1 for x>0 and alpha for x<=0.
            //binary_positive_mask gives 1/0, so blend it into 1/alpha: mask*(1-alpha) + alpha
            TensorMath::binary_positive_mask(
                forward_input,
                backward_out
            );
            TensorMath::scale(
                backward_out,
                1.0f - LEAKY_RELU_ALPHA,
                backward_out
            );
            Tensor alpha_tensor = Tensor({1},this->_dev,LEAKY_RELU_ALPHA,true);
            TensorMath::add(
                backward_out,
                alpha_tensor,
                backward_out
            );
            //Perform element_wise multiplication on the mask with the incoming gradient
            TensorMath::multiply(
                backward_grad,
                backward_out,
                backward_out
            );
            break;
        }
        default:
            throw std::runtime_error("A unexpected type recevied");
            break;
    }

    return this->_backward_out;
}

void ActivationLayer::update_weight(float){
    //Does nothing: No weigths in this layer
    return;
}