#ifndef _MMA_H
#define _MMA_H

namespace nvcuda
{
namespace wmma
{

#define mdassert(expr, msg) if ( !(expr) ) { std::cout << "ERROR: " << msg << "\n"; exit(1); }

class accumulator
{
};

class matrix_a
{
};

class matrix_b
{
};

// these are used for Layout below
// note: col_major is standard for GEMM
class row_major
{
};

class col_major         
{
};

using layout_t = uint32_t;
const layout_t mem_row_major = 0;
const layout_t mem_col_major = 1;

template<typename Use, int m, int n, int k, typename T, typename Layout=void> 
class fragment
{
public:
    fragment( void )
    {
        mdassert( m == n && n == k, "fragment; m==n==k is currently required" );
        num_elements = m*m; 
        x = new T[num_elements];
    }

    size_t num_elements;
    T *    x;                   // note: for my sanity, always stored here in row-major regardless of Layout
};

template<typename Use, int m, int n, int k, typename Layout=void>
void fill_fragment( fragment<Use, m, n, k, float, Layout> &a, const float& v )
{
    for( size_t i = 0; i < a.num_elements; i++ )
    {
        a.x[i] = v;
    }
}

template<typename Use, int m, int n, int k, typename Layout=void>
void fill_fragment( fragment<Use, m, n, k, __half, Layout> &a, const __half& v )
{
    for( size_t i = 0; i < a.num_elements; i++ )
    {
        a.x[i] = v;
    }
}

// mptr points to start of fragment in matrix.
// ldm is col/row stride for col/row major fragment. 
//
template<typename Use, int m, int n, int k, typename T, typename Layout=void>
void load_matrix_sync( fragment<Use, m, n, k, T, Layout> &a, const T* mptr, unsigned ldm )
{
    mdassert( ldm >= n, "load_matrix_sync: ldm (stride) too small" );
    bool is_row_major = std::is_same<Layout, row_major>::value;
    T * fptr = a.x;
    if ( is_row_major ) {
        for( uint32_t i = 0; i < m; i++ )
        {
            for( uint32_t j = 0; j < n; j++, fptr++, mptr++ )
            {
                *fptr = *mptr;
            }

            mptr += ldm - n;  // skip to beginning of next row
        }
    } else {
        for( uint32_t i = 0; i < m; i++ )
        {
            for( uint32_t j = 0; j < n; j++, fptr++ )
            {
                *fptr = mptr[j*ldm + i];
            }
        }
    }
}

// opposite of load_matrix_sync()
//
template<typename Use, int m, int n, int k, typename T, typename Layout=void>
void store_matrix_sync(T* mptr, const fragment<Use, m, n, k, T, Layout> &a, unsigned ldm, layout_t layout)
{
    mdassert( ldm >= n, "load_matrix_sync: ldm (stride) too small" );
    bool is_row_major = layout == mem_row_major;
    const T * fptr = a.x;
    if ( is_row_major ) {
        for( uint32_t i = 0; i < m; i++ )
        {
            for( uint32_t j = 0; j < n; j++, fptr++, mptr++ )
            {
                *mptr = *fptr;
            }

            mptr += ldm - n;  // skip to beginning of next row
        }
    } else {
        for( uint32_t i = 0; i < m; i++ )
        {
            for( uint32_t j = 0; j < n; j++, fptr++ )
            {
                mptr[j*ldm + i] = *fptr;
            }
        }
    }
}

// perform d = a*b + c (typically d == c)
//
template<int m, int n, int k, typename T, typename Layout_d=void, typename Layout_a=void, typename Layout_b=void, typename Layout_c=void>
void mma_sync(      fragment<accumulator, m, n, k, T, Layout_d> &d, 
              const fragment<matrix_a,    m, n, k, T, Layout_a> &a,
              const fragment<matrix_b,    m, n, k, T, Layout_b> &b,
              const fragment<accumulator, m, n, k, T, Layout_c> &c,
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
