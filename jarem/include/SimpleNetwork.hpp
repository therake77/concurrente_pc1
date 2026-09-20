#include <DLLib.hpp>
#include <stdexcept>

class SimpleNetwork{
private:
    std::vector<std::unique_ptr<Layer>> layers;
    LayerMode mode = LayerMode::Inference;
public:

    SimpleNetwork(
        std::vector<std::unique_ptr<Layer>>&& _layers
    ) : layers(std::move(_layers)) {}

    const std::shared_ptr<const Tensor> forward(const Tensor& input);
    const std::shared_ptr<const Tensor> forward(const Tensor& input, const Tensor& label);
    const std::shared_ptr<const Tensor> backwards(const Tensor& label, const Tensor& input);

    template <typename LayerType, typename... Args>
    SimpleNetwork& with_layer(
        Args&&... args
    ){
        this->layers.push_back(
            std::make_unique<LayerType>(
                std::forward<Args>(args)...
            )
        );
        return *this;
    }

    void update_weights(float lr);
    void set_mode(LayerMode mode);
    void he_initialize_all();

};

class SimpleNetworkBuilder{
private:
    std::vector<std::unique_ptr<Layer>> layers;
public:
    SimpleNetworkBuilder(
        const std::size_t with_layers
    ){
        this->layers.reserve(with_layers);
    }


    template <typename LayerType, typename... Args>
    SimpleNetworkBuilder& with_layer(
        Args&&... args
    ){
        this->layers.push_back(
            std::make_unique<LayerType>(
                std::forward<Args>(args)...
            )
        );
        return *this;
    }

    SimpleNetwork build(){
        if(this->layers.size() == 0){ throw std::runtime_error("Nothing to build"); }
        return SimpleNetwork(
            std::move(this->layers)
        );
    }

};