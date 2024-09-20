#include "pattern.hpp"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "hip-commons.hpp"

#include "color.hpp"


void Gradient::render_cpu(
    uint32_t* buf,
    uint32_t w, uint32_t h,
    uint8_t b, uint8_t a)
{
    for (uint32_t y = 0; y < h; y++)
        for (uint32_t x = 0; x < h; x++)
        {
            const uint64_t i = (uint64_t)y*w + x;
            const Color color = pixelcolor(x, y, w, h, b, a);
            buf[i] = color.col();
        }
}

__global__ void Gradient::render_gpu(
    uint32_t* d_buf,
    uint32_t w, uint32_t h,
    uint8_t b, uint8_t a)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint64_t i = (uint64_t)y*w + x;

    if (x >= w || y >= h) return; // off-screen

    const Color color = pixelcolor(x, y, w, h, b, a);
    d_buf[i] = color.col();
}
