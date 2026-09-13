#pragma once
#include <Layer.hpp>

class OutputLayer : public Layer{
private:

    Tensor _probabilities;
    std::size_t _logits;
public:
    OutputLayer(
        std::shared_ptr<Device> dev,
        std::size_t logits
    );
    ~OutputLayer() override = default;

    std::shared_ptr<Tensor> forward( const Tensor& ) override;
    std::shared_ptr<Tensor> backward( const Tensor&, const Tensor& ) override;
    std::shared_ptr<Tensor> forward( const Tensor&, const Tensor& );
    void update_weight(float) override;
};