#include <core/Shape.hpp>
#include <stdexcept>

using namespace MyTensors::Core;

/* Please, don't put something like {5,...,0,...,0,...,5}. it will break the logic  */
Shape::Shape(std::array<size_t,TENSOR_MAX_DIM> arr) : _shapes(arr),_strides({0}) {
    
    size_t index = TENSOR_MAX_DIM - 1;
    while(index < TENSOR_MAX_DIM && arr[index] == 0){ index--; }
    if(index >= TENSOR_MAX_DIM ) { throw std::runtime_error("Error: Tensor with zero dimensions not allowed"); }
    
    size_t _n_dims = index + 1;
    this->_strides[index] = 1;
    size_t _n_elems = arr[index];
    
    for(size_t i = index; i --> 0; ){
        _n_elems*=arr[i];
        this->_strides[i] =  arr[i+1] * this->_strides[i+1];
    }
    
    this->n_elements = _n_elems;
    this->n_dim = _n_dims;

}

void Shape::_trigger_shape_recomposition(){
    //Start from the rightmost side
    std::size_t index = TENSOR_MAX_DIM - 1;
    //Ignore zero dimensions
    while(this->_shapes[index] == 0){ index--; }
    //Compute the current dimensionality
    std::size_t _n_dims = index + 1;
    //Prepare to compute the strides
    this->_strides[index] = 1;
    std::size_t _n_elems = this->_shapes[index];
    //Regenerating strides
    for(std::size_t i = index; i --> 0; ){
        _n_elems*=this->_shapes[i];
        this->_strides[i] =  this->_shapes[i+1] * this->_strides[i+1];
    }
    //Regenerate dimensionality metadata
    this->n_elements = _n_elems;
    this->n_dim = _n_dims;
}

bool Shape::operator==(const Shape& other) const{
    return this->_shapes == other._shapes;
}

bool Shape::operator!=(const Shape& other) const{
    return this->_shapes != other._shapes;
}

const size_t& Shape::operator[](std::size_t index) const{ 
    if( index >= TENSOR_MAX_DIM){ throw std::runtime_error("Index out of bounds"); }
    return this->_shapes[index]; 
}

size_t& Shape::operator[](std::size_t index){
    if(index >= TENSOR_MAX_DIM){ throw std::runtime_error("Index out of bounds"); }
    return this->_shapes[index]; 
}

Shape Shape::unsqueeze(std::size_t axis){
    if( this->n_dim == TENSOR_MAX_DIM) throw std::runtime_error("Impossible to add more dimensions");
    if( axis > this->n_dim){ throw std::runtime_error("Invalid axis"); }
    std::array<std::size_t, TENSOR_MAX_DIM> new_shape {};
    std::array<std::size_t, TENSOR_MAX_DIM> new_strides {};

    for(std::size_t i = 0 ; i < axis; i++){
        new_shape[i] = this->_shapes[i];
        new_strides[i] = this->_strides[i];
    }
    new_shape[axis] = 1;
    new_strides[axis] = 0;
    for(std::size_t i = axis + 1; i < TENSOR_MAX_DIM ; i++ ){
        new_shape[i] = this->_shapes[i-1];
        new_strides[i] = this->_strides[i-1];
    }

    Shape out(new_shape);
    out._strides = new_strides;
    out.n_elements = this->n_elements;
    out.n_dim = this->n_dim + 1;
    return out;
}

void Shape::transpose(){
    if(this->n_dim <= 1) { return; }
    std::size_t index1 = this->n_dim-1;
    std::size_t index2 = this->n_dim-2;
    std::swap(this->_shapes[index1],this->_shapes[index2]);
    std::swap(this->_strides[index1],this->_strides[index2]);
}

void Shape::transpose(std::size_t index1,std::size_t index2){
    if(index1 < this->n_dim && index2 < this->n_dim ){ 
        if(index1 == index2) { return; }
        std::swap(this->_shapes[index1],this->_shapes[index2]);
        std::swap(this->_strides[index1],this->_strides[index2]);
        
    }else{
        throw std::runtime_error("Error: index out of bounds");
    }
    return;
}

/*
    @brief Expands (or contracts) the shape, adding scalar dimensions (or removing dimensions) on the right side (innermost side)
*/
void Shape::inner_pad(std::size_t dim){
    if(dim > TENSOR_MAX_DIM){ throw std::runtime_error("Error: Impossible to expand"); }
    if( dim == n_dim){ return; }
    if(dim > n_dim){
        for(std::size_t i = n_dim; i < dim; ++i){
            this->_shapes[i] = 1;
            this->_strides[i] = 0;
        }
        this->n_dim = dim;
        return;
    }

    std::array<std::size_t, TENSOR_MAX_DIM> new_shapes {};
    std::fill(new_shapes.begin(),new_shapes.begin()+dim,1);
    std::size_t i = 0;  
    std::size_t j = 0;
    while( i < dim && j < this->n_dim){
        new_shapes[i++] = this->_shapes[j++];
    }
    this->_shapes = new_shapes;
    _trigger_shape_recomposition();
}

/*
    @brief Expands (or contracts) the shape, adding scalar dimensions (or removing dimensions) on the left side (outermost side)
*/
void Shape::outer_pad(std::size_t dim){
    if(dim > TENSOR_MAX_DIM){ throw std::runtime_error("Error: Impossible to expand"); }
    if( dim == n_dim){ return; }
    if(dim > n_dim){
        std::array<std::size_t, TENSOR_MAX_DIM> new_shapes {};
        std::array<std::size_t, TENSOR_MAX_DIM> new_strides {};
        std::fill(new_shapes.begin(), new_shapes.begin() + dim, 1);

        std::size_t offset = dim - n_dim;
        for(std::size_t i = 0; i < n_dim; ++i){
            new_shapes[offset + i] = this->_shapes[i];
            new_strides[offset + i] = this->_strides[i];
        }

        this->_shapes = new_shapes;
        this->_strides = new_strides;
        this->n_dim = dim;
        return;
    }

    std::array<std::size_t, TENSOR_MAX_DIM> new_shapes {};
    std::fill(new_shapes.begin(),new_shapes.begin()+dim,1);
    std::size_t i = dim;  
    std::size_t j = this->n_dim;
    while( i --> 0 && j --> 0){
        new_shapes[i] = this->_shapes[j];
    }
    this->_shapes = new_shapes;
    _trigger_shape_recomposition();
}

bool Shape::is_contiguous(){
    //I'm bored, i'm going to take advantage of the Shape constructor and its ability
    //of computing strides
    Shape as_it_should_be = Shape(this->_shapes);
    if(as_it_should_be._strides == this->_strides){
        return true;
    }else{
        return false;
    }
}

/*
    @brief Broadcast two shapes. If their dimensionality don't match, expands them outerly
*/
void Shape::broadcast_shapes(Shape& a, Shape& b, std::size_t ignore){
    auto n_dim = std::max(a.n_dim,b.n_dim);
    if( n_dim < ignore ) throw std::runtime_error("Error: Impossible to broadcast");
    if( n_dim == ignore ){ return; } //Nothing to broadcast
    a.outer_pad(n_dim);
    b.outer_pad(n_dim);

    for(size_t  i = n_dim - ignore; i --> 0;){
        if( a[i] == 1 || b[i] == 1 ){
            if( a[i] == b[i] ){ continue; } //Do no touch strides, shapes are already equal
            std::size_t max_shape = std::max(a[i],b[i]);
            if( a[i] == max_shape) b._strides[i] = 0; 
            else a._strides[i] = 0;
            a[i] = max_shape;
            b[i] = max_shape;
        }else{
            if(a[i] != b[i] ){ throw std::runtime_error("Error: Bad broadcast"); }
        }
    }
    return;
}
