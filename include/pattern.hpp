#ifndef PATTERNH
#define PATTERNH

#include <stdint.h>

#include "hip-commons.hpp"

#include "complex.hpp"
#include "color.hpp"


struct viewsettings
{
    uint32_t width, height;
    complex center;
    PRECISION_FP zoom;
};

/**
 * Helper method to convert screenspace coordinates to the complex plane
 */
constexpr __host__ __device__ complex pixToComplex(
    uint32_t pixx, uint32_t pixy, viewsettings view)
{
    // avoid div-by-zero
    if (fabs(view.zoom) < PRECISION_FP_EPSILON)
        return complex{ 0.0, 0.0 };

    const PRECISION_FP halfx = view.width  / 2.0;
    const PRECISION_FP halfy = view.height / 2.0;

    // shift to center zoom
    const PRECISION_FP zoomedx = (pixx - halfx) / view.zoom;
    const PRECISION_FP zoomedy = (pixy - halfy) / view.zoom;

    // invert y because -screeny == +imaginary
    const PRECISION_FP real =  zoomedx + view.center.real;
    const PRECISION_FP imag = -zoomedy - view.center.imag;

    return complex{ real, imag };
}


/**
 * Gradient pattern
 *
 * A very simple pattern that lerps red across the x and green across the y,
 * with the given blue.
 *
 * Used for testing video outputs and debugging.
 */
namespace Gradient
{

    constexpr __host__ __device__ Color pixelcolor(
        uint32_t x, uint32_t y,
        uint32_t w, uint32_t h,
        uint8_t b, uint8_t a)
    {
        const PRECISION_FP xp = (PRECISION_FP)x / w;
        const PRECISION_FP yp = (PRECISION_FP)y / h;
        return Color((uint8_t)(xp * 255.99), (uint8_t)(yp * 255.99), b, a);
    }

    void render_cpu(
        uint32_t* pixels,
        uint32_t w, uint32_t h,
        uint8_t b, uint8_t a);

    __global__ void render_gpu(
        uint32_t* d_pixels,
        uint32_t w, uint32_t h,
        uint8_t b, uint8_t a);

};


/**
 * Mandelbrot pattern
 *
 * Uses the function that the Mandelbrot Set is built from;
 * >    f[0](z) = c
 * >    f[N+1](z) = f[N](z)^exp + c
 * >    ...while magnitude(z) < threshhold
 *
 * Coloring is based on how many iterations N before the magnitude of f[N](z)
 * crosses the given threshhold.
 */
namespace Mandelbrot
{

    __host__ __device__ uint64_t depth(
        complex c,
        PRECISION_FP exponent, PRECISION_FP threshhold);

    void render_cpu(
        uint32_t* pixels, viewsettings view,
        PRECISION_FP exponent, PRECISION_FP threshhold);

    __global__ void render_gpu(
        uint32_t* d_pixels, viewsettings view,
        PRECISION_FP exponent, PRECISION_FP threshhold);

};

#endif // PATTERNH
