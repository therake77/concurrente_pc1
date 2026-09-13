#include <Layer.hpp>
#include <functional>

enum class ActivationTypes{
    ReLU,
    Sigmoid,
    LeakyReLU
};

class ActivationLayer : public Layer{
private:
    ActivationTypes type;
public:

    ActivationLayer(
        ActivationTypes = ActivationTypes::ReLU,
        std::shared_ptr<Device> = DeviceManager::get_cpu_device()
    );

    std::shared_ptr<Tensor> forward( const Tensor& ) override;
    std::shared_ptr<Tensor> backward( const Tensor&, const Tensor&  ) override;
    void update_weight(float) override;

};
