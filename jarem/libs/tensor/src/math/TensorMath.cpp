#include <math/TensorMath.hpp>
#include <stdexcept>

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

    /*Expands the shape adding outer dimensions of length one with strides of zeros */
    Shape _expand_shape_no_strides(const Shape& a){
        //Create the new shape consisting of all dimensions of length 1
        std::array<size_t,TENSOR_MAX_DIM> _temp;
        _temp.fill(1);
        Shape new_shape = Shape(_temp);
        //Manually make all the strides zero
        for(size_t i = 0; i < TENSOR_MAX_DIM; i++){
            new_shape._strides[i] = 0;
        }
        //Now append the `a` shape which is something like (A,B,C...,M,0,0,0,0) to the new shape as ->(1,1,1,...,1,A,B,C,...,M)
        for(size_t i = 0; i < a.n_dim; i++){
            new_shape._shapes[TENSOR_MAX_DIM - a.n_dim + i] = a._shapes[i];
            new_shape._strides[TENSOR_MAX_DIM - a.n_dim + i] = a._strides[i]; 
        }
        //And return the expanded shape
        return new_shape;
    }
    /*Expands the shape adding outer dimensions of lenght 1, but strides are kept correct*/
    Shape _expand_shape_with_strides(const Shape& a){
        //FIrst make a copy of a._shape in a freshly new array
        std::array<size_t, TENSOR_MAX_DIM> _new_shape;
        _new_shape.fill(1);
        for(size_t i = 0; i < a.n_dim; i++){
            _new_shape[TENSOR_MAX_DIM - a.n_dim + i] = a._shapes[i];
        }
        //Shape constructor also initializes strides, so is okay
        return Shape(_new_shape);
    }

    /*Matmul subroutine: Broadcast shapes `a` and `b`, ignoring the two innermost dimensions*/
    void _input_broadcast(Shape& a, Shape& b){
        //A small validation: Are they expanded?
        if(a.n_dim != TENSOR_MAX_DIM || a.n_dim != b.n_dim){ throw std::runtime_error("Error: Shapes are not expanded"); }
        
        for(size_t i = TENSOR_MAX_DIM - 2; i-->0;){
            //If one of them is one, then broadcast the max shape between both
            if(a._shapes[i] == 1 || b._shapes[i] == 1){
                if(a._shapes[i] == b._shapes[i]){ continue; } //Both are 1, respect stride
                size_t max = std::max(a._shapes[i], b._shapes[i]);
                //Before broadcasting, we need sure to put a zero stride on the dimension affected
                if( a._shapes[i] == max) b._strides[i] = 0; 
                else a._strides[i] = 0;
                a._shapes[i] = max;
                b._shapes[i] = max;
            }else{
                //If neither one of them is one, then THEY SHOULD BE EQUAL, or the broadcast is impossible
                if( a._shapes[i] != b._shapes[i] ){ throw std::runtime_error("Error: Bad broadcasting"); }
            }
        }
    }

    /*Check the already broadcasted and expanded shapes of the inputs and output tensors*/
    void _check_matmul_shapes(Shape& a, Shape& b, Shape& out){
        //Clasic matrix multiplication check on two last dimensions
        if( 
            a._shapes[TENSOR_MAX_DIM - 1] != b._shapes[TENSOR_MAX_DIM - 2] || 
            a._shapes[TENSOR_MAX_DIM - 2] != out._shapes[TENSOR_MAX_DIM - 2] ||
            b._shapes[TENSOR_MAX_DIM - 1] != out._shapes[TENSOR_MAX_DIM - 1]
        ){
            throw std::exception("Error: Unexpected shapes. Expected inputs (...,M,K),(...,K,N) and output (...,M,N)");
        }
        //Now check if the rest of the shapes are equal
        for(size_t i = TENSOR_MAX_DIM - 2; i --> 0;){
            if(a._shapes[i] != b._shapes[i]){ throw std::exception("Error: Outer dimensions are supposed to be equal. Bad broadcasting"); }
            if(a._shapes[i] != out._shapes[i]){ throw std::exception("Error: Outer dimensions are supposed to be equal. Bad output expansion"); }
        }
    }

    void _check_conv2D_shapes(Shape& a, Shape& b, Shape& out){

    }

};
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

/*
Convolution operation.
Expects 4D tensors. No exceptions allowed.
*/
void TensorMath::conv2D(const Tensor& in, const Tensor& ker, Tensor& out, size_t stride, size_t padding){
    //Device guards
    _ensure_same_device(in,ker);
    _ensure_same_device(ker,out);
    //Ensure all tensors are EXACTLY 4D
    if(in.shape.n_dim != 4 || ker.shape.n_dim != 4 || out.shape.n_dim != 4){ throw std::runtime_error("Error: Tensors must be 4-dimensional"); }
    
    //Should we implement all guards here? YES TODO(IMPLEMENT THE GUARDS)

    auto device = in.getDevice();
    
    //just call the underlying math function
    device->math->conv2D(
        in.data(),in.shape,
        ker.data(),ker.shape,
        out.data(),out.shape,
        {stride,stride},
        {padding,padding}
    );
}

void TensorMath::conv2D(const Tensor& in, const Tensor& ker, Tensor& out, std::array<size_t,2> stride, std::array<size_t,2> padding){
    _ensure_same_device(in,ker);
    _ensure_same_device(ker,out);
    //Ensure all tensors are EXACTLY 4D
    if(in.shape.n_dim != 4 || ker.shape.n_dim != 4 || out.shape.n_dim != 4){ throw std::runtime_error("Error: Tensors must be 4-dimensional"); }
    
    //Should we implement all guards here?

    auto device = in.getDevice();
    
    //just call the underlying math function
    device->math->conv2D(
        in.data(),in.shape,
        ker.data(),ker.shape,
        out.data(),out.shape,
        stride,
        padding
    );
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

Tensor TensorMath::transpose(const Tensor& a){
    std::array<size_t, TENSOR_MAX_DIM> out_shape = a.shape._shapes;

    if(a.shape.n_dim >= 2){
        size_t last = a.shape.n_dim - 1;
        size_t prev = a.shape.n_dim - 2;

        size_t tmp = out_shape[prev];
        out_shape[prev] = out_shape[last];
        out_shape[last] = tmp;
    }

    auto device = a.getDevice();
    Tensor out(out_shape, device);
    device->math->transpose(a.data(), a.shape, out.data(), out.shape);
    return out;
}

void TensorMath::max_pooling2d(
    const Tensor& in,  Tensor& out, Tensor& mask, std::array<std::size_t,2> strides,  
    std::array<std::size_t,2> window_shapes, 
    std::array<std::size_t,2> padding
){
    _ensure_same_device(in,out);
    _ensure_same_device(out,mask);
    _ensure_same_shape(in,mask);
    //Ensure in,out,mask are 4D
    if( in.shape.n_dim != 4 || out.shape.n_dim != 4){ throw std::runtime_error("Error: Unexpected shape"); }
    //Ensure in and out shapes follows the rules of max_pooling
    if( in.shape[0] != out.shape[0] || in.shape[1] != out.shape[1] ) { throw std::runtime_error("Error: Unexpected shape"); }
    if( 
        out.shape[2] != (in.shape[2] + 2*padding[0] - window_shapes[0])/strides[0] + 1 ||
        out.shape[3] != (in.shape[3] + 2*padding[1] - window_shapes[1])/strides[1] + 1 
    ){
        throw std::runtime_error("Error: Unexpected shape");
    }
    //All requirements met, call the function
    auto dev = in.getDevice();
    dev->math->max_pooling2D(
        in.data(),
        in.shape,
        strides,
        window_shapes,
        mask.data(),
        mask.shape,
        out.data(),
        out.shape,
        padding
    );

    return;
}

void TensorMath::inv_max_pooling2D(
    const Tensor& max_pooled,
    std::array<std::size_t,2> strides,  
    std::array<std::size_t,2> window_shapes,    //(Height , Width)
    const Tensor& mask,
    Tensor& out,
    std::array<std::size_t, 2> padding
){
    _ensure_same_device(max_pooled,mask);
    _ensure_same_device(mask,out);
    _ensure_same_shape(mask,out);
    if( max_pooled.shape.n_dim != 4 || out.shape.n_dim != 4){ throw std::runtime_error("Error: Unexpected shape"); }
    if( max_pooled.shape[0] != out.shape[0] || max_pooled.shape[1] != out.shape[1] ) { throw std::runtime_error("Error: Unexpected shape"); }
    if( 
        max_pooled.shape[2] != (out.shape[2] + 2*padding[0] - window_shapes[0])/strides[0] + 1 ||
        max_pooled.shape[3] != (out.shape[3] + 2*padding[1] - window_shapes[1])/strides[1] + 1 
    ){
        throw std::runtime_error("Error: Unexpected shape");
    }

    auto device = max_pooled.getDevice();
    device->math->inv_max_pooling2D(
        max_pooled.data(),
        max_pooled.shape,
        strides,
        window_shapes,
        mask.data(),
        mask.shape,
        out.data(),
        out.shape,
        padding
    );


}

void TensorMath::avg_pooling2d(
    const Tensor& in, Tensor& out, std::array<std::size_t,2> strides,  
    std::array<std::size_t,2> window_shapes, 
    std::array<std::size_t, 2> padding
){
    _ensure_same_device(in,out);
    if( in.shape.n_dim != 4 || out.shape.n_dim != 4){ throw std::runtime_error("Error: Unexpected shape"); }
    //Ensure in and out shapes follows the rules of max_pooling
    if( in.shape[0] != out.shape[0] || in.shape[1] != out.shape[1] ) { throw std::runtime_error("Error: Unexpected shape"); }
    if( 
        out.shape[2] != (in.shape[2] + 2*padding[0] - window_shapes[0])/strides[0] + 1 ||
        out.shape[3] != (in.shape[3] + 2*padding[1] - window_shapes[1])/strides[1] + 1 
    ){
        throw std::runtime_error("Error: Unexpected shape");
    }
    //All requirements met, call the function
    auto dev = in.getDevice();
    dev->math->avg_pooling2D(
        in.data(),
        in.shape,
        strides,
        window_shapes,
        out.data(),
        out.shape,
        padding
    );

    return;
}

void TensorMath::inv_avg_pooling2D(
    const Tensor& avg_pooled,
    std::array<std::size_t,2> strides,  
    std::array<std::size_t,2> window_shapes,    //(Height , Width)
    Tensor& out,
    std::array<std::size_t, 2> padding
){
    _ensure_same_device(avg_pooled,out);

    if( avg_pooled.shape.n_dim != 4 || out.shape.n_dim != 4){ throw std::runtime_error("Error: Unexpected shape"); }
    if( avg_pooled.shape[0] != out.shape[0] || avg_pooled.shape[1] != out.shape[1] ) { throw std::runtime_error("Error: Unexpected shape"); }
    if( 
        avg_pooled.shape[2] != (out.shape[2] + 2*padding[0] - window_shapes[0])/strides[0] + 1 ||
        avg_pooled.shape[3] != (out.shape[3] + 2*padding[1] - window_shapes[1])/strides[1] + 1 
    ){
        throw std::runtime_error("Error: Unexpected shape");
    }

    auto device = avg_pooled.getDevice();
    device->math->inv_avg_pooling2D(
        avg_pooled.data(),
        avg_pooled.shape,
        strides,
        window_shapes,
        out.data(),
        out.shape,
        padding
    );
    
    return;
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

void TensorMath::mean(
    const Tensor& in,
    std::size_t axis,
    Tensor& out
){
    _ensure_same_device(in,out);
    _check_dimensionality(out.shape, 1);
    if(in.shape[axis] != out.shape[0]){ throw std::runtime_error("Error: Unexpected out shape"); }

    auto device = in.getDevice();
    device->math->mean(
        in.data(),
        in.shape,
        out.data(),
        out.shape,
        axis
    );

}

void TensorMath::variance(
    const Tensor& in,
    const Tensor& means,
    std::size_t axis,
    Tensor& out
){
    _ensure_same_device(in,out);
    _ensure_same_device(means,out);
    _check_dimensionality(out.shape, 1);
    _check_dimensionality(means.shape,1);

    if(in.shape[axis] != out.shape[0] || in.shape[axis] != means.shape[0]){ 
        throw std::runtime_error("Error: Unexpected out shape"); 
    }
    
    auto device = in.getDevice();
    device->math->variance(
        in.data(),
        in.shape,
        means.data(),
        means.shape,
        out.data(),
        out.shape,
        axis
    );
}

void TensorMath::reduce_all(
    const Tensor& in,
    Tensor& out,
    std::size_t axis
){
    _ensure_same_device(in,out);
    _check_dimensionality(out.shape,1);
    if(in.shape[axis] != out.shape[0]){ 
        throw std::runtime_error("Error: Unexpected out shape"); 
    }

    auto dev = in.getDevice();
    dev->math->reduce_all(
        in.data(),
        in.shape,
        out.data(),
        out.shape,
        axis
    );
    return;
}
