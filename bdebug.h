// Bob's debug prints
//
#ifndef __BDEBUG__
#define __BDEBUG__

#include <stdlib.h>
#include <cmath>
#include <fstream>
#include <iostream>

static bool __bdebug = false;
static bool __bdebug_any = false;

#define bdebug_enable()                 __bdebug = true; __bdebug_any = true
#define bdebug_enable_if_first()        if ( !__bdebug_any ) bdebug_enable()
#define bdebug_disable()                __bdebug = false;
#define bdebug_enabled()                __bdebug
#define bdout if ( __bdebug ) std::cout

template<typename T>
inline void matrix_print( std::ostream& os, const T * data, int m, int n, bool is_row_major, bool add_newline=false ) 
{
    if ( is_row_major ) {
        os << "[ ";
        for( int mi = 0; mi < m; mi++ )
        {
            if ( mi != 0 ) os << ", ";
            os << "[";
            for( int ni = 0; ni < n; ni++, data++ )
            {
                if ( ni != 0 ) os << ",";
                os << float(*data);
            }
            os << "]";
        }
        os << " ]"; 
        if ( add_newline ) os << "\n";
    } else {
        std::cout << "ERROR: can't print col-major at the moment\n";
        exit( 1 );
    }
}

template<typename T>
inline void bdmatrix_print( std::ostream& os, const T * data, int m, int n, bool is_row_major, bool add_newline=true ) 
{
    if ( bdebug_enabled() ) {
        matrix_print( os, data, m, n, is_row_major, add_newline );
    }
}

#endif
