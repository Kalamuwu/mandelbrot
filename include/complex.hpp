#ifndef COMPLEXH
#define COMPLEXH

#include <math.h>
#include <cfloat>

#include "hip-commons.hpp"

#define PRECISION_FP32 1


// allows for selecting different levels of precision
#if defined(PRECISION_FP16)
    #define PRECISION_FP float
    #define PRIpFP "f"
    #define PRECISION_FP_EPSILON FLT_EPSILON
#elif defined(PRECISION_FP32)
    #define PRECISION_FP double
    #define PRIpFP "f"
    #define PRECISION_FP_EPSILON DBL_EPSILON
#elif defined(PRECISION_FP64)
    #define PRECISION_FP long double
    #define PRIpFP "Lf"
    #define PRECISION_FP_EPSILON LDBL_EPSILON
#else
    #error "One of PRECISION_FP16, PRECISION_FP32, PRECISION_FP64 must be defined"
#endif

// because its helpful to have with polars
constexpr PRECISION_FP TWOPI = (PRECISION_FP)2 * (PRECISION_FP)M_PI;


// since operators on std::complex aren't constexpr...

#define COMPLEXH_COMPLEX constexpr __host__ __device__ complex
#define COMPLEXH_POLAR   constexpr __host__ __device__ polar
#define COMPLEXH_REAL    constexpr __host__ __device__ PRECISION_FP

struct polar
{
    PRECISION_FP r, theta;
};

struct complex
{
    PRECISION_FP real, imag;

    COMPLEXH_COMPLEX conj(void) const;

    COMPLEXH_REAL squared_magnitude(void) const;
    COMPLEXH_REAL magnitude(void) const;

    COMPLEXH_COMPLEX squared(void) const;
    COMPLEXH_COMPLEX pow(const PRECISION_FP exponent) const;
};

COMPLEXH_COMPLEX ascomplex(const polar p)
{
    // with z = x + yi, then z = r(cos(theta) + i sin(theta))
    //   r = magnitude of z
    //   x = r cos(theta)
    //   y = r sin(theta)
    return complex{ p.r*cos(p.theta), p.r*sin(p.theta) };
};
COMPLEXH_POLAR aspolar(const complex c)
{
    // find r (must by definition be >= 0):
    const PRECISION_FP r = c.magnitude();
    if (r <= PRECISION_FP_EPSILON) return polar{ 0.0, 0.0 };

    // find theta (both ways are identical):
    PRECISION_FP theta = acos(c.real / r);
    if (c.imag < 0) theta = -theta;

    // // alternate way to find theta
    // PRECISION_FP theta = asin(c.imag / r);
    // if (c.real < 0) theta = -theta;

    return polar{ r, theta };
};

COMPLEXH_COMPLEX complex::conj(void) const
{
    // conj(a+bi) = a-bi
    return complex{ real, -imag };
};

COMPLEXH_REAL complex::squared_magnitude(void) const
{
    return real*real + imag*imag;
}
COMPLEXH_REAL complex::magnitude(void) const
{
    // note: sqrt isn't constexpr
    return sqrt(squared_magnitude());
}

COMPLEXH_COMPLEX complex::squared(void) const
{
    // given:
    //   (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    // therefore:
    //   (a+bi)^2 = (aa-bb) + (ab+ab)i
    //   (a+bi)^2 = (aa-bb) + (2ab)i
    //   (a+bi)^2 = (aa-bb) + ((a+a)b)i
    return complex{ real*real - imag*imag, (real+real) * imag };
}

COMPLEXH_COMPLEX complex::pow(const PRECISION_FP n) const
{
    if (abs(n) < PRECISION_FP_EPSILON) return complex{ 1.0, 0.0 };

    // z^n = (r^n)(cos(n*theta) + i sin(n*theta))
    const polar p = aspolar(*this);
    const polar powed { powf(p.r, n), n*p.theta };
    return ascomplex(powed);
}


// generic operators

COMPLEXH_COMPLEX operator+(const complex a, const complex b)
{
    // (a+bi) + (c+di) = (a+c) + (b+d)i
    return complex{ a.real+b.real, a.imag+b.imag };
}
COMPLEXH_COMPLEX operator-(const complex a, const complex b)
{
    // (a+bi) - (c+di) = (a-c) + (b-d)i
    return complex{ a.real-b.real, a.imag-b.imag };
}

COMPLEXH_COMPLEX operator-(const complex c)
{
    // -(a+bi) = -a + -bi
    return complex{ -c.real, -c.imag };
}

COMPLEXH_COMPLEX operator*(const complex a, const complex b)
{
    // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    return complex{ a.real*b.real - a.imag*b.imag,
                    a.real*b.imag + a.imag*b.real };
}
COMPLEXH_COMPLEX operator*(const complex c, const PRECISION_FP s)
{
    return complex{ c.real * s, c.imag * s };
}
COMPLEXH_COMPLEX operator*(const PRECISION_FP s, const complex c)
{
    return complex{ c.real * s, c.imag * s };
}

COMPLEXH_COMPLEX operator/(const complex a, const complex b)
{
    // (a+bi)/(c+di) = (ac+bd)/(cc+dd) + (bc-ad)/(cc+dd) i
    const PRECISION_FP denom = b.real*b.real + b.imag*b.imag;
    if (denom < PRECISION_FP_EPSILON) return complex{ 0.0, 0.0 };
    return complex{ (a.real*b.real + a.imag*b.imag) / denom,
                    (a.imag*b.real - a.real*b.imag) / denom };
}
COMPLEXH_COMPLEX operator/(const complex c, const PRECISION_FP s)
{
    return c / complex{ s, 0.0 };
}
COMPLEXH_COMPLEX operator/(const PRECISION_FP s, const complex c)
{
    return complex{ s, 0.0 } / c;
}


#undef COMPLEXH_COMPLEX
#undef COMPLEX_POLAR
#undef COMPLEXH_REAL

#endif // COMPLEXH
