#ifndef HIPCOMMONH
#define HIPCOMMONH

#if !defined(__HIP__)
#define __HIP__ 1
#endif

#if defined(__HIP_PLATFORM_NVIDIA__)
#undef __HIP_PLATFORM_NVIDIA__
#endif

#if !defined(__HIP_PLATFORM_AMD__)
#define __HIP_PLATFORM_AMD__ 1
#endif

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include "hip/device_functions.h"
#include "hip/math_functions.h"
#include "hiprand/hiprand_kernel.h"


/**
 * Usage:
 * >  const dim3 blockDims(32,32);
 * >  const dim3 gridDims(
 * >      ceiling_div(w, blockDims.x),
 * >      ceiling_div(h, blockDims.y));
 *
 * https://github.com/ROCm/rocm-examples/blob/b1b2122a2afa4a735e68cf4045256135d60b40a6/Common/example_utils.hpp#L179
 */
template<typename T,
         typename U,
         std::enable_if_t<std::is_integral<T>::value && std::is_unsigned<U>::value, int> = 0>
constexpr __host__ __device__ auto ceiling_div(
    const T& dividend, const U& divisor)
{
    return (dividend + divisor - 1) / divisor;
}


#endif
