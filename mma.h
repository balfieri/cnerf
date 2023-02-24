#ifndef _MMA_H
#define _MMA_H

namespace nvcuda
{
namespace wmma
{

class row_major
{
};

class col_major
{
};

class accumulator
{
};

class matrix_a
{
};

class matrix_b
{
};

using layout_t = uint32_t;
const layout_t mem_row_major = 0;
const layout_t mem_col_major = 1;

template<typename Use, int m, int n, int k, typename T, typename Layout=void> 
class fragment
{
public:
    fragment(void)
    {
        num_elements = m*n; // not sure if this is right
        x = new T[num_elements];
    }

    size_t num_elements;
    T *    x;
};

template<typename Use, int m, int n, int k, typename T, typename Layout=void>
void load_matrix_sync(fragment<Use, m, n, k, T, Layout> &a, const T* mptr, unsigned ldm)
{
     (void)a;
     (void)mptr;
     (void)ldm;    
     throw std::runtime_error{"load_matrix_sync not yet implemented"};
}

template<typename Use, int m, int n, int k, typename T, typename Layout=void>
void load_matrix_sync(fragment<Use, m, n, k, T, Layout> &a, const T* mptr, unsigned ldm, layout_t layout)
{
     (void)a;
     (void)mptr;
     (void)ldm;    
     (void)layout;
     throw std::runtime_error{"load_matrix_sync yet implemented"};
}

template<typename Use, int m, int n, int k, typename T, typename Layout=void>
void store_matrix_sync(T* mptr, const fragment<Use, m, n, k, T, Layout> &a, unsigned ldm, layout_t layout)
{
     (void)mptr;
     (void)a;
     (void)ldm;    
     (void)layout;
     throw std::runtime_error{"store_matrix_sync not yet implemented"};
}

template<typename Use, int m, int n, int k, typename Layout=void>
void fill_fragment(fragment<Use, m, n, k, float, Layout> &a, const float& v)
{
     (void)a;
     (void)v;
     throw std::runtime_error{"fill_fragment not yet implemented"};
}

template<typename Use, int m, int n, int k, typename Layout=void>
void fill_fragment(fragment<Use, m, n, k, __half, Layout> &a, const __half& v)
{
     (void)a;
     (void)v;
     throw std::runtime_error{"fill_fragment not yet implemented"};
}

template<int m, int n, int k, typename Layout_d=void, typename Layout_a=void, typename Layout_b=void, typename Layout_c=void>
void mma_sync(      fragment<accumulator, m, n, k, float, Layout_d> &d, 
              const fragment<matrix_a,    m, n, k, float, Layout_a> &a,
              const fragment<matrix_b,    m, n, k, float, Layout_b> &b,
              const fragment<accumulator, m, n, k, float, Layout_c> &c,
              bool  satf=false)
{
     (void)d;
     (void)a;
     (void)b;
     (void)c;
     (void)satf;
     throw std::runtime_error{"mma_sync not yet implemented"};
}

template<int m, int n, int k, typename Layout_d=void, typename Layout_a=void, typename Layout_b=void, typename Layout_c=void>
void mma_sync(      fragment<accumulator, m, n, k, __half, Layout_d> &d, 
              const fragment<matrix_a,    m, n, k, __half, Layout_a> &a,
              const fragment<matrix_b,    m, n, k, __half, Layout_b> &b,
              const fragment<accumulator, m, n, k, __half, Layout_c> &c,
              bool  satf=false)
{
     (void)d;
     (void)a;
     (void)b;
     (void)c;
     (void)satf;
     throw std::runtime_error{"mma_sync not yet implemented"};
}

}; // wmma
}; // nvcuda

#endif
