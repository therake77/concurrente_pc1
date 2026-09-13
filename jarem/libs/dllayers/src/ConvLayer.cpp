#include <ConvLayer.hpp>
#include <MyTensors.hpp>
#include <cassert>

Conv2D::Conv2D(
    std::array<size_t, 4> kernel_shape,
    std::shared_ptr<Device> dev,
    std::array<size_t,2> _padding,
    std::array<size_t,2> _strides)
  : _kernels(kernel_shape,dev),
    padding(_padding),
    strides(_strides),
    Layer(dev)
{}

/*
@brief Computes the forward pass of a convolution layer
@param[in] input Input tensor - coming from upper layers
@param[out] output Preallocated output tensor
@result A new, freshly allocated tensor if no output was given, else `void`
@note It is absolutely not recommended to pass the input tensor as the output tensor, as the method will overwrite data during the operation
*/
 std::shared_ptr<Tensor> Conv2D::forward( const Tensor& in){
    //Compute convolution
    using dim_t = std::size_t;
    dim_t features = _kernels.shape[0];
    dim_t channels = _kernels.shape[1];
    dim_t kernel_height = _kernels.shape[2];
    dim_t kernel_width = _kernels.shape[3];

    dim_t N = in.shape[0];
    dim_t H = in.shape[2];
    dim_t W = in.shape[3];
    dim_t H_out = (H + 2 * padding[0] - kernel_height) / strides[0] + 1;
    dim_t W_out = (W + 2 * padding[1] - kernel_width) / strides[1] + 1;

    __check_initialization(
        this->_forward_col_input,
        Shape({channels*kernel_height*kernel_width,N*H_out*W_out})
    );
    //First, transform the input
    TensorMath::im2col2D(
        in,
        {kernel_height,kernel_width},
        this->strides,
        this->padding,
        *(this->_forward_col_input)
    );

    //Reshape the kernels
    this->_kernels.reshape({features,channels*kernel_height*kernel_width});
    //Multiply
    __check_initialization(this->_forward_out,Shape({features,N*H_out*W_out}));
    TensorMath::matmul(
        this->_kernels,
        *(this->_forward_col_input),
        *(this->_forward_out)
    );
    //Return the original shape of the kernels
    this->_kernels.reshape({features,channels,kernel_height,kernel_width});
    //Reshape the output
    this->_forward_out->reshape({features,N,H_out,W_out});
    this->_forward_out->shape.transpose(0,1);
    this->_forward_out->contiguous();
    return _forward_out;
}

std::shared_ptr<Tensor> Conv2D::backward( const Tensor& backward_grad, const Tensor& forward_in){
    assert(this->_forward_col_input.get() != nullptr);
    using dim_t = std::size_t;
    dim_t features = this->_kernels.shape[0];
    dim_t channels = this->_kernels.shape[1];
    dim_t k_height = this->_kernels.shape[2];
    dim_t k_width = this->_kernels.shape[3];    

    __check_initialization(this->_grad,Shape({features,channels*k_height*k_width}));
    __check_initialization(this->_backward_out,forward_in.shape);
    //Create a temporal copy of the backward_grad
    Tensor _backward_grad_copy = Tensor(backward_grad,backward_grad.getDevice());
    _backward_grad_copy.shape.transpose(0,1);
    _backward_grad_copy.reshape({
        backward_grad.shape[1],
        backward_grad.shape[0] * backward_grad.shape[2] * backward_grad.shape[3]
    });
    //Compute kernel grad
    
    this->_forward_col_input->shape.transpose();
    TensorMath::matmul(
        _backward_grad_copy,
        *(this->_forward_col_input),
        *(this->_grad)
    );
    this->_grad->reshape({features,channels,k_height,k_width});
    //Compute kernel to propagate backwards. First, flatten the kernel
    this->_kernels.reshape({features,channels*k_height*k_width});
    this->_kernels.shape.transpose();
    this->_forward_col_input->shape.transpose();
    TensorMath::matmul(
        this->_kernels,
        _backward_grad_copy,
        *(this->_forward_col_input)     //Warning : use of _forward_col_input as a buffer
    );

    //Put back again the kernels to their original shapes
    this->_kernels.shape.transpose();
    this->_kernels.reshape({features,channels,k_height,k_width});

    TensorMath::col2im2D(
        *(this->_forward_col_input),
        {k_height,k_width},
        this->strides,
        this->padding,
        *(this->_backward_out)
    );
    return this->_backward_out;
}


void Conv2D::update_weight(float lr){
    assert(this->_grad.get() != nullptr);
    TensorMath::scale(
        *(this->_grad),
        lr,
        *(this->_grad)
    );
    TensorMath::add(
        this->_kernels,
        *(this->_grad),
        this->_kernels
    );
}

void Conv2D::he_initialization(){
    float fan_in = (float)this->_kernels.shape[1] * this->_kernels.shape[2] * this->_kernels.shape[3];
    float std_dev = std::sqrt(2.0f/fan_in);
    this->_kernels.normal_rand(0.0f,std_dev);
}