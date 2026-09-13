#include <core/Tensor.hpp>
#include <hardware/CpuDevice.hpp>
#ifdef __USE_CUDA__
    #include <hardware/GpuDevice.cuh>
#endif
#include <math/TensorMath.hpp>
