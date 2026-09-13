#include <Layer.hpp>

class Conv2D : public Layer{
private:
    Tensor _kernels;    //(C_out, C_in, K_h, K_w)
    std::unique_ptr<Tensor> _forward_col_input;
    std::array<size_t,2> padding;
    std::array<size_t,2> strides;
public:
    
    ~Conv2D() override = default;

    Conv2D(
        std::array<size_t, 4> shape,
        std::shared_ptr<Device> dev, 
        std::array<size_t,2> _padding, 
        std::array<size_t,2> _strides
    );

    std::shared_ptr<Tensor> forward( const Tensor& ) override;
    std::shared_ptr<Tensor> backward( const Tensor&, const Tensor&  ) override;
    void he_initialization() override;
    void update_weight(float) override;
};