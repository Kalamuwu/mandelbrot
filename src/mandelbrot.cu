#include "pattern.hpp"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "hip-commons.hpp"

#include "complex.hpp"
#include "color.hpp"
#include "logging.hpp"


#define MAX_DEPTH 1024lu
#define N_BANDS 16u


__host__ __device__ uint64_t Mandelbrot::depth(
    complex c,
    PRECISION_FP exponent, PRECISION_FP threshhold)
{
    if (fabs(exponent) < PRECISION_FP_EPSILON) return 0;
    const PRECISION_FP sqthresh = threshhold * threshhold;

    complex z = c;
    uint64_t i = 0;

    while (z.squared_magnitude() < sqthresh && ++i < MAX_DEPTH)
        z = z.pow(exponent) + c;

    return i;
}

/*
constexpr Color pallete[N_BANDS]
{
    Color(255,255,255,255),
    Color(255,255,255,255),
    Color(255,255,255,255),
    Color(255,255,255,255),
    Color(255,255,255,255),
    Color(255,255,255,255),
    Color(255,255,255,255),
    Color(255,255,255,255),
    Color(255,255,255,255),
    Color(255,255,255,255)
};
*/

constexpr __host__ __device__ Color colorForDepth(const uint64_t n)
{
    // quick return
    if (n >= MAX_DEPTH)
        return BLACK;

    const double val = (n%N_BANDS) / (double)N_BANDS;
    const uint8_t b = (uint8_t)(255.99 * val);
    //return gammaCorrect(Color(0, 0, b, 255));
    return Color(0, 0, b, 255);

    // if      (n == 0) return Color(255,  0,  0,255);  // red
    // else if (n == 1) return Color(255,120,  0,255);  // orange
    // else if (n == 2) return Color(255,255,  0,255);  // yellow
    // else if (n == 3) return Color(  0,255,  0,255);  // green
    // else if (n == 4) return Color(  0,  0,255,255);  // blue
    // else if (n == 5) return Color(255,  0,255,255);  // purple

    // fallback
    return WHITE;
}


void Mandelbrot::render_cpu(
    uint32_t* buf, viewsettings view,
    const PRECISION_FP exponent, const PRECISION_FP threshhold)
{
    for (uint32_t y = 0; y < view.height; y++)
        for (uint32_t x = 0; x < view.width; x++)
        {
            const uint64_t i = (uint64_t)y*view.width + x;

            const complex c = pixToComplex(x, y, view);
            const uint64_t n = Mandelbrot::depth(c, exponent, threshhold);

            const Color color = colorForDepth(n);
            buf[i] = color.col();
        }
}

__global__ void Mandelbrot::render_gpu(
    uint32_t* d_buf, viewsettings view,
    const PRECISION_FP exponent, const PRECISION_FP threshhold)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint64_t i = (uint64_t)y*view.width + x;

    if (x >= view.width || y >= view.height) return; // off-screen

    const complex c = pixToComplex(x, y, view);
    const uint64_t n = Mandelbrot::depth(c, exponent, threshhold);

    const Color color = colorForDepth(n);
    d_buf[i] = color.col();
}
