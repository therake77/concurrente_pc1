#include <OutputLayer.hpp>
#include <MyTensors.hpp>
#include <stdexcept>
#include <cassert>

using namespace MyTensors::Math;

OutputLayer::OutputLayer(
    std::shared_ptr<Device> dev,
    std::size_t logits
) : _logits(logits) , Layer(dev), _probabilities({1,logits},dev) { }

/*
Inference forward pass
*/
std::shared_ptr<Tensor> OutputLayer::forward( const Tensor& in ){

    if( mode == LayerMode::Inference){
        __check_initialization(this->_forward_out,in.shape);

        TensorMath::softmax(
            in,
            *(this->_forward_out)
        );

        return this->_forward_out;
    }else{
        throw std::runtime_error("Error: Not in inference mode");
    }
}

std::shared_ptr<Tensor> OutputLayer::forward(const Tensor& in, const Tensor& labels){
    if(this->mode == LayerMode::Training){
        std::size_t batchs = in.shape[0];
        _probabilities.resize(in.shape);
        TensorMath::softmax(
            in,
            this->_probabilities
        );

        __check_initialization(this->_backward_out, in.shape);
        Tensor& log_probs = *(this->_backward_out);

        TensorMath::log(
            this->_probabilities, //And then, use _forward_out to compute the rest
            log_probs
        );
        TensorMath::multiply(
            log_probs,
            labels,
            log_probs
        );
        __check_initialization(this->_forward_out,Shape({batchs}));
        this->_forward_out->zeros();
        TensorMath::sum(
            log_probs,
            1,
            *(this->_forward_out)
        );
        //_forward_out now holds the per-sample loss vector, shape (N,)
        TensorMath::scale(
            *(this->_forward_out),
            -1.0f,
            *(this->_forward_out)
        );
        return this->_forward_out;
    }else{
        throw std::runtime_error("Error: Not in training mode");
    }
}

//Important: backward_grad in this layer are the real labels
std::shared_ptr<Tensor> OutputLayer::backward( const Tensor& labels, const Tensor& forward_in ) {

    assert(this->_forward_out.get() != nullptr);

    std::size_t batch = forward_in.shape[0];
    __check_initialization(this->_backward_out,forward_in.shape);

    TensorMath::sub(
        this->_probabilities,
        labels,
        *(this->_backward_out)
    );
    float to_divide = 1.0f/(float)batch;
    TensorMath::scale(
        *(this->_backward_out),
        to_divide,
        *(this->_backward_out)
    );
    return this->_backward_out;
}

void OutputLayer::update_weight(float){
    //No weights
    return;
}
