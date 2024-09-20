#ifndef COLORH
#define COLORH

#include <stdint.h>
#include <bits/endian.h>

/**
 * `Color`
 * Represents a color of a pixel.
 */
struct Color
{
    constexpr Color(
        const uint8_t r,
        const uint8_t g,
        const uint8_t b,
        const uint8_t a);
    constexpr Color(const uint32_t col);

    constexpr uint8_t r(void) const;
    constexpr uint8_t g(void) const;
    constexpr uint8_t b(void) const;
    constexpr uint8_t a(void) const;

    constexpr uint32_t col(void) const;

private:
    uint8_t m_col[4];
};

// Color functions: defined here for constexpr support

#if __BYTE_ORDER == __LITTLE_ENDIAN
constexpr Color::Color(
    const uint8_t r,
    const uint8_t g,
    const uint8_t b,
    const uint8_t a)
:
    m_col{ a, b, g, r }
{}
constexpr Color::Color(const uint32_t col)
:
    m_col{
        (uint8_t)(col >>  0),
        (uint8_t)(col >>  8),
        (uint8_t)(col >> 16),
        (uint8_t)(col >> 24)}
{}
constexpr uint8_t Color::r(void) const { return m_col[3]; }
constexpr uint8_t Color::g(void) const { return m_col[2]; }
constexpr uint8_t Color::b(void) const { return m_col[1]; }
constexpr uint8_t Color::a(void) const { return m_col[0]; }
constexpr uint32_t Color::col(void) const
{ return ((r()<<24) | (g()<<16) | (b()<<8) | a()); }
#elif __BYTE_ORDER == __BIG_ENDIAN
constexpr Color::Color(
    const uint8_t r,
    const uint8_t g,
    const uint8_t b,
    const uint8_t a)
:
    m_col{ r, g, b, a }
{}
constexpr Color::Color(const uint32_t col)
:
    m_col{
        (uint8_t)(col >> 24),
        (uint8_t)(col >> 16),
        (uint8_t)(col >>  8),
        (uint8_t)(col >>  0)}
{}
constexpr uint8_t Color::r(void) const { return m_col[0]; }
constexpr uint8_t Color::g(void) const { return m_col[1]; }
constexpr uint8_t Color::b(void) const { return m_col[2]; }
constexpr uint8_t Color::a(void) const { return m_col[3]; }
constexpr uint32_t Color::col(void) const
{ return ((a()<<24) | (b()<<16) | (g()<<8) | r()); }
#else
# error "Please fix <bits/endian.h>"
#endif


constexpr Color BLACK(  0,  0,  0,255);
constexpr Color WHITE(255,255,255,255);


#endif // COLORH
