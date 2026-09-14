#include <SimpleNetwork.hpp>

const std::shared_ptr<const Tensor> SimpleNetwork::forward(const Tensor& input){
    if(this->mode == LayerMode::Inference){
        if(this->layers.size() > 0){
            this->layers[0]->forward(input);
            for(std::size_t i = 1 ; i < this->layers.size(); i++){
                this->layers[i]->forward(
                    *(this->layers[i-1]->_forward_out)
                );
            }
            return this->layers[this->layers.size()-1]->_forward_out;
        }
        return nullptr;
    }
    throw std::runtime_error("Network not in inference mode");
}

const std::shared_ptr<const Tensor> SimpleNetwork::forward(const Tensor& input, const Tensor& label){
    if(this->mode == LayerMode::Training){
        //It is assumed the output layer is the last layer
        if(this->layers.size() > 0){
            //Feed the first layer
            this->layers[0]->forward(input);
            for(std::size_t i = 1 ; i < this->layers.size()-1; i++){
                this->layers[0]->forward(
                    *(this->layers[i-1]->_forward_out)
                );
            }
            if(this->layers.size() > 1){
                auto& last_layer = dynamic_cast<OutputLayer&>(*(this->layers[this->layers.size()-1]));
            
                last_layer.forward(
                    *(this->layers[this->layers.size() - 2 ]->_forward_out),
                    label
                );
            }
            return this->layers[this->layers.size()-1]->_forward_out;
        }
        return nullptr;
    }
    throw std::runtime_error("Network not in training mode");
}

const std::shared_ptr<const Tensor> SimpleNetwork::backwards(const Tensor& label, const Tensor& input){
    
    std::size_t n_layers = this->layers.size();
    if(n_layers > 0){
        //Feed the last layer
        this->layers[n_layers - 1]->backward(
            label,
            *(this->layers[n_layers-2]->_forward_out)
        );
        //Feed intermediate layers
        for(std::size_t i = n_layers - 1; i -->1;){
            this->layers[i]->backward(
                *(this->layers[i+1]->_backward_out),
                *(this->layers[i-1]->_forward_out)
            );
        }
        //Feed the first layer
        auto& first_layer_backwards = n_layers > 1 ? *(this->layers[1]->_backward_out) :  label;
        this->layers[0]->backward(
            first_layer_backwards,
            input
        );
        return this->layers[0]->_backward_out;
    }
    return nullptr;
}

void SimpleNetwork::update_weights(float lr){
    for(std::size_t i = 0; i < this->layers.size(); i++){
        this->layers[i]->update_weight(lr);
    }
}

void SimpleNetwork::set_mode(LayerMode mode){
    for(std::size_t i = 0; i < this->layers.size(); i++){
        this->layers[i]->set_mode(mode);
    }
}