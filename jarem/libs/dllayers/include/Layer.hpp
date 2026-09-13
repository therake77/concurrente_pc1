#pragma once
#include <MyTensors.hpp>

enum class LayerMode{
    Inference,
    Training
};

using MyTensors::Core::Tensor;
using MyTensors::Core::Shape;
using MyTensors::Hardware::Device;

class Layer{
protected:
    LayerMode mode = LayerMode::Inference;
    std::shared_ptr<Device> _dev;
public:
    std::shared_ptr<Tensor> _forward_out;
    std::shared_ptr<Tensor> _backward_out;
    std::shared_ptr<Tensor> _grad;

    Layer(std::shared_ptr<Device> dev) : _dev(dev) {}
    
    virtual ~Layer() = default;
    virtual std::shared_ptr<Tensor> forward( const Tensor& ) = 0;
    virtual std::shared_ptr<Tensor> backward( const Tensor&, const Tensor& ) = 0;
    virtual void update_weight(float learning_rate) = 0;

    virtual void he_initialization(){
        return;
    }

    inline void set_mode(LayerMode mode){ this->mode = mode; }

    void __check_initialization(std::shared_ptr<Tensor>& t, Shape s){
        if(t.get() == nullptr){
            t = std::make_shared<Tensor>(s._shapes,this->_dev);
        }else if(t->shape != s){
            t->resize(s);
        }
        return;
    }

    void __check_initialization(std::unique_ptr<Tensor>& t, Shape s){
        if(t.get() == nullptr){
            t = std::make_unique<Tensor>(s._shapes,this->_dev);
        }else if(t->shape != s){
            t->resize(s);
        }
        return;
    }

};