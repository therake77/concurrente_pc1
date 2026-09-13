#include <Layer.hpp>

class FullyConnected : public Layer{
private:
    std::size_t input_dim;
    std::size_t output_dim;
    Tensor _weights;
    Tensor _bias;
    std::unique_ptr<Tensor> _bias_grad;

public:

    FullyConnected(
        std::array<std::size_t, MyTensors::Core::TENSOR_MAX_DIM> weight_shape,
        std::shared_ptr<Device> dev
    );

    std::shared_ptr<Tensor> forward( const Tensor& ) override;
    std::shared_ptr<Tensor> backward( const Tensor&, const Tensor&  ) override;
    void he_initialization() override;
    void update_weight(float) override;

};
