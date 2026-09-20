#include <Layer.hpp>

class Flatten : public Layer{

public:
    Flatten(
        std::shared_ptr<Device> dev
    );

    ~Flatten() override = default;
    std::shared_ptr<Tensor> forward( const Tensor& ) override;
    std::shared_ptr<Tensor> backward( const Tensor&, const Tensor& ) override;
    void update_weight(float) override;
};