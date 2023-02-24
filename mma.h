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

template<typename Use, int m, int n, int k, typename T, typename Layout=void>
void fill_fragment(fragment<Use, m, n, k, T, Layout> &a, const T& v)
{
     (void)a;
     (void)v;
     throw std::runtime_error{"fill_fragment not yet implemented"};
}

template<typename Use, int m, int n, int k, typename T, typename Layout=void>
void mma_sync(      fragment<Use, m, n, k, T, Layout> &d, 
              const fragment<Use, m, n, k, T, Layout> &a,
              const fragment<Use, m, n, k, T, Layout> &b,
              const fragment<Use, m, n, k, T, Layout> &c,
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
