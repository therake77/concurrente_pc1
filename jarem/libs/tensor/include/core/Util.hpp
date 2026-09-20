#pragma once
#include <core/Tensor.hpp>
#include <iostream>

namespace MyTensors::Util{
    using MyTensors::Core::Tensor;

    inline void tensor_print(Tensor& a){
        const float* data = a.data();
        std::vector<std::size_t> coords(a.shape.n_dim,0);
        for(std::size_t i = 0; i < a.shape.n_dim; ++i){
            std::cout<<"[";
        }
        for(std::size_t el = 0; el < a.shape.n_elements; ++el){
            std::size_t flat_idx = 0;
            for (std::size_t i = 0; i < a.shape.n_dim; ++i){
                flat_idx+=coords[i] * a.shape._strides[i];
            }
            std::cout << data[flat_idx];
            bool finished = false;
            std::size_t reset = 0;
            for(std::size_t i = a.shape.n_dim; i --> 0; ){
                coords[i]++;
                if(coords[i] < a.shape[i]){ break; }
                else{ if( i ==0 ){ finished = true; }else{ coords[i] = 0; reset++; } }
            }

            if(finished || el == a.shape.n_elements -1 ){
                for(std::size_t i = 0; i < a.shape.n_dim; ++i) std::cout<<"]";
                std::cout<< std::endl;
                break;
            }

            if(reset == 0){std::cout<<", ";}
            else{
                for(std::size_t i = 0; i < reset; ++i){std::cout<<"]";}
                std::cout<<",\n";
                for(std::size_t i = 0; i < a.shape.n_dim - reset; ++i){ std::cout<< " "; }
                for(std::size_t i = 0; i < reset; ++i){ std::cout<<"["; } 
            }
        }
    }
    
};