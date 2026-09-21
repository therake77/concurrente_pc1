#include <math/TensorMath.hpp>
#include <stdexcept>

using MyTensors::Core::Tensor;
using MyTensors::Core::Shape;
using MyTensors::Hardware::DeviceType;
using MyTensors::Core::TENSOR_MAX_DIM;

namespace{
    
    /*Ensures `a` and `b` have the same exact shape*/
    void _ensure_same_shape(const Tensor& a, const Tensor&b){
        if(a.shape != b.shape){ throw std::runtime_error("Error: Tensor shape mismatch"); }
    }

    /*Ensures `a` and `b` have the same exact shape*/
    void _ensure_same_shape(const Shape& a, const Shape& b){
        if(a != b){ throw std::runtime_error("Error: Tensor shape mismatch"); }
    }

    void _ensure_same_device(const Tensor& a, const Tensor&b, DeviceType type){
        auto a_device_type = a.getDevice()->get_type();
        auto b_device_type = b.getDevice()->get_type();
        if(a_device_type != type || a_device_type != b_device_type){ throw std::runtime_error("Error: Unexpected device"); }
    }

    void _ensure_same_device(const Tensor& a, const Tensor&b){
        auto a_device_type = a.getDevice()->get_type();
        auto b_device_type = b.getDevice()->get_type();
        if(a_device_type != b_device_type){ throw std::runtime_error("Error: Unexpected device"); }
    }

    void _check_dimensionality(const Shape& s, std::size_t expected){
        if(s.n_dim != expected){ throw std::runtime_error("Error: Unexpected number of dimensions"); }
    }

};

using namespace MyTensors::Math;
/*Function that adds two tensors component-wise. It is safe to put the input argument `a` as `out`*/
void TensorMath::add(const Tensor& a, const Tensor&b, Tensor& out){
    _ensure_same_device(a,b);
    _ensure_same_device(b,out);
    
    //Copy the shapes
    Shape a_shape = a.shape;
    Shape b_shape = b.shape;

    //Perform broadcasting
    Shape::broadcast_shapes(a_shape,b_shape);

    //Shape guards
    _ensure_same_shape(a_shape,b_shape);
    _ensure_same_shape(b_shape,out.shape);

    auto device = a.getDevice();
    device->math->add(
        a.data(),
        a_shape,
        b.data(),
        b_shape,
        out.data(),
        out.shape);

}

void TensorMath::sub(const Tensor& a, const Tensor&b, Tensor& out){
    _ensure_same_device(a,b);
    _ensure_same_device(b,out);
    
    //Copy the shapes
    Shape a_shape = a.shape;
    Shape b_shape = b.shape;

    //Perform broadcasting
    Shape::broadcast_shapes(a_shape,b_shape);

    //Shape guards
    _ensure_same_shape(a_shape,b_shape);
    _ensure_same_shape(b_shape,out.shape);

    auto device = a.getDevice();
    device->math->sub(
        a.data(),
        a_shape,
        b.data(),
        b_shape,
        out.data(),
        out.shape
    );
}

void TensorMath::scale(const Tensor& a, float n, Tensor& out){
    _ensure_same_shape(a,out);
    _ensure_same_device(a,out);

    auto device = a.getDevice();
    device->math->scale(
        a.data(),
        out.data(),
        n,
        a.shape
    );
}

void TensorMath::multiply(const Tensor& a, const Tensor&b, Tensor& out){
    
    _ensure_same_device(a,b);
    _ensure_same_device(b,out);

    //Copy the shapes
    Shape a_shape = a.shape;
    Shape b_shape = b.shape;

    //Perform broadcasting
    Shape::broadcast_shapes(a_shape,b_shape);
    _ensure_same_shape(a_shape,b_shape);
    _ensure_same_shape(b_shape,out.shape);

    auto device = a.getDevice();
    device->math->multiply(
        a.data(),
        a_shape,
        b.data(),
        b_shape,
        out.data(),
        out.shape
    );

}

void TensorMath::divide(const Tensor& a, const Tensor&b, Tensor& out){
    _ensure_same_device(a,b);
    _ensure_same_device(b,out);
    
    //Copy the shapes
    Shape a_shape = a.shape;
    Shape b_shape = b.shape;

    //Perform broadcasting
    Shape::broadcast_shapes(a_shape,b_shape);

    //Shape guards
    _ensure_same_shape(a_shape,b_shape);
    _ensure_same_shape(b_shape,out.shape);


    auto device = a.getDevice();
    device->math->divide(
        a.data(),
        a_shape,
        b.data(),
        b_shape,
        out.data(),
        out.shape
    );

}

void TensorMath::sqrt(const Tensor& a, Tensor& out){
    _ensure_same_shape(a,out);
    _ensure_same_device(a,out);

    auto device = a.getDevice();
    device->math->sqrt(
        a.data(),
        a.shape,
        out.data(),
        out.shape
    );
}

void TensorMath::pow(const Tensor& a, int i, Tensor& out){
    _ensure_same_shape(a,out);
    _ensure_same_device(a,out);

    auto device = a.getDevice();
    device->math->pow(
        a.data(),
        a.shape,
        i,
        out.data(),
        out.shape
    );
}

void TensorMath::log(const Tensor& a, Tensor& out){
    _ensure_same_device(a,out);
    _ensure_same_shape(a,out);
    auto dev = a.getDevice();
    dev->math->log(
        a.data(),
        a.shape,
        out.data(),
        out.shape
    );
    return;
}

/*Matmul routine*/
void TensorMath::matmul(const Tensor& a, const Tensor&b , Tensor& out){
    
    _ensure_same_device(a,b);
    _ensure_same_device(b,out);

    auto device = a.getDevice();

    Shape _expanded_a = a.shape;
    Shape _expanded_b = b.shape;
    Shape::broadcast_shapes(_expanded_a,_expanded_b,2); //Ignore last two (innermost) dimension
    Shape _virtual_expanded_out = out.shape;
    _virtual_expanded_out.outer_pad(std::max(a.shape.n_dim,b.shape.n_dim));

    device->math->matmul(
        a.data(),_expanded_a,
        b.data(),_expanded_b,
        out.data(), _virtual_expanded_out
    );

    //Finish
}

void TensorMath::im2col2D(
    const Tensor& in, 
    std::array<std::size_t,2> window_shape,
    std::array<std::size_t,2> strides,
    std::array<std::size_t,2> padding,
    Tensor& out
){
    _ensure_same_device(in,out);
    if( in.shape.n_dim != 4 || out.shape.n_dim != 2){
        throw std::runtime_error("Error: Input should be 4-dimensional and output 2-dimensional");
    }
    {
        using dim_t = std::size_t;
        dim_t N = in.shape[0];
        dim_t C = in.shape[1];
        dim_t H = in.shape[2];
        dim_t W = in.shape[3];
        dim_t H_out = (H + 2*padding[0] - window_shape[0])/strides[0] + 1;
        dim_t W_out = (W + 2*padding[1] - window_shape[1])/strides[1] + 1;
        if(
            out.shape[0] != C*window_shape[0] * window_shape[1] ||
            out.shape[1] != N*H_out*W_out
        ){
            throw std::runtime_error("Error: Unexpected output shape");
        }
    }

    auto device = in.getDevice();
    device->math->im2col2d(
        in.data(),
        in.shape,
        out.data(),
        out.shape,
        window_shape,
        strides,
        padding
    );
    return;
}

void TensorMath::col2im2D(
    const Tensor& in, 
    std::array<std::size_t,2> window_shape,
    std::array<std::size_t,2> strides,
    std::array<std::size_t,2> padding,
    Tensor& out
){
    _ensure_same_device(in,out);
    if( in.shape.n_dim != 2 || out.shape.n_dim != 4 ){
        throw std::runtime_error("Error: Input should be 4-dimensional and output 2-dimensional");
    }
    {
        using dim_t = std::size_t;
        dim_t N = out.shape[0];
        dim_t C = out.shape[1];
        dim_t H = out.shape[2];
        dim_t W = out.shape[3];
        dim_t H_in = (H + 2*padding[0] - window_shape[0])/strides[0] + 1;
        dim_t W_in = (W + 2*padding[1] - window_shape[1])/strides[1] + 1;
        if(
            in.shape[0] != C*window_shape[0] * window_shape[1] ||
            in.shape[1] != N*H_in*W_in
        ){
            throw std::runtime_error("Error: Unexpected output shape");
        }
    }

    auto device = in.getDevice();
    device->math->col2im2d(
        in.data(),
        in.shape,
        out.data(),
        out.shape,
        window_shape,
        strides,
        padding
    );
    return;
}

void TensorMath::reLU(const Tensor& a, Tensor& out){
    _ensure_same_shape(a,out);
    _ensure_same_device(a,out);

    auto device = a.getDevice();
    device->math->reLU(
        a.data(),
        out.data(),
        a.shape.n_elements
    );
    return;
}

void TensorMath::sigmoid(const Tensor& a, Tensor& out){
    _ensure_same_shape(a,out);
    _ensure_same_device(a,out);

    auto device = a.getDevice();
    device->math->sigmoid(
        a.data(),
        out.data(),
        a.shape.n_elements
    );
    return;
}

void TensorMath::tanh(const Tensor& a, Tensor& out){
    _ensure_same_shape(a,out);
    _ensure_same_device(a,out);

    auto device = a.getDevice();
    device->math->tanh(
        a.data(),
        out.data(),
        a.shape.n_elements
    );
    return;
}

void TensorMath::softmax(const Tensor& a, Tensor& out){
    _ensure_same_shape(a,out);
    _ensure_same_device(a,out);

    auto device = a.getDevice();
    device->math->softmax(
        a.data(),
        a.shape,
        out.data(),
        out.shape
    );
    return;
}

void TensorMath::sum(const Tensor& a, std::size_t axis, Tensor& out){
    _ensure_same_device(a,out);
    Shape a_shape_copy = a.shape;
    Shape out_shape_copy = out.shape;
    if(a.shape.n_dim - 1== out.shape.n_dim){
        out_shape_copy = out_shape_copy.unsqueeze(axis);
    }
    Shape::broadcast_shapes(a_shape_copy,out_shape_copy);
    _ensure_same_shape(a_shape_copy,out_shape_copy);
    //Because of traumas, explicitly zero the axis stride on out
    out_shape_copy._strides[axis] = 0;
    out.zeros();
    auto dev = a.getDevice();
    dev->math->sum(
        a.data(),
        a_shape_copy,
        out.data(),
        out_shape_copy,
        axis
    );
    return;
}

Tensor TensorMath::reshape(const Tensor& a, std::array<size_t, TENSOR_MAX_DIM> new_shape){
    Shape target_shape(new_shape);
    if(target_shape.n_elements != a.shape.n_elements){
        throw std::runtime_error("Error: reshape requires the same number of elements");
    }
    Tensor out(new_shape, a.getDevice());
    for(size_t i = 0; i < a.shape.n_elements; i++){
        out.data()[i] = a.data()[i];
    }
    return out;
}

void TensorMath::binary_positive_mask(
    const Tensor& in,
    Tensor& out
){
    _ensure_same_device(in,out);
    _ensure_same_shape(in,out);

    auto device = in.getDevice();

    device->math->binary_positive_mask(
        in.data(),
        in.shape,
        out.data(),
        out.shape
    );
    
    return;
}