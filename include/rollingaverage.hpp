#ifndef AVERAGEH
#define AVERAGEH

#include <stdlib.h>
#include <memory>
#include <string.h>


template <typename T>
class RollingAverage
{
public:
    RollingAverage(std::size_t size) : m_size(size)
    {
        m_buf = (T*)malloc(sizeof(T) * m_size);
        memset(m_buf, 0x00, sizeof(T) * m_size);
    }

    ~RollingAverage(void)
    {
        free(m_buf);
        m_buf = nullptr;
    }

    void push(T el)
    {
        m_buf[m_ptr] = el;
        if (++m_ptr >= m_size) m_ptr = 0;
        if (m_saturated < m_ptr) m_saturated = m_ptr;
    }

    void clear(void)
    {
        for (std::size_t i = 0; i < m_size; i++)
            m_buf[i] = T {0};
        m_saturated = 0;
        m_ptr = 0;
    }

    void clear(T el)
    {
        for (std::size_t i = 0; i < m_size; i++)
            m_buf[i] = el;
        m_saturated = 0;
        m_ptr = 0;
    }

    T sum(void) const
    {
        T ret {0};
        for (std::size_t i = 0; i < m_size; i++)
            ret += m_buf[i];
        return ret;
    }

    T avg(void) const
    {
        if (m_saturated == 0) return 0;
        return sum() / m_saturated;
    }

private:
    std::size_t m_size, m_saturated = 0;
    std::size_t m_ptr = 0;
    T* m_buf;
};

#endif // AVERAGEH
