// cnerf.cpp - C++ single-threaded reference model based on instant NeRF sources.
//
// Initially, this will do only rendering (forward pass through neural net).
//

#include "../tinyexr/deps/miniz/miniz.c"

#include "../tiny-cuda-nn/src/common.cu"
#include "../tiny-cuda-nn/src/common_device.cu"
#include "../tiny-cuda-nn/src/encoding.cu"
#include "../tiny-cuda-nn/src/fully_fused_mlp.cu"
#include "../tiny-cuda-nn/src/loss.cu"
#include "../tiny-cuda-nn/src/network.cu"
#include "../tiny-cuda-nn/src/object.cu"
#include "../tiny-cuda-nn/src/optimizer.cu"
#include "../tiny-cuda-nn/src/reduce_sum.cu"

#include "../instant-ngp/src/main.cu"           // but main() below
#include "../instant-ngp/src/testbed.cu"
#include "../instant-ngp/src/testbed_image.cu"
#include "../instant-ngp/src/testbed_nerf.cu"   
#include "../instant-ngp/src/testbed_sdf.cu"
#include "../instant-ngp/src/testbed_volume.cu"
#include "../instant-ngp/src/camera_path.cu"
#include "../instant-ngp/src/common.cu"
#include "../instant-ngp/src/common_device.cu"
//#include "../instant-ngp/src/dlss.cu"         // requires OpenGL which I want to avoid here
#include "../instant-ngp/src/marching_cubes.cu"
#include "../instant-ngp/src/nerf_loader.cu"
//#include "../instant-ngp/src/openxr_hmd.cu"   // requires OpenGL which I want to avoid here
#include "../instant-ngp/src/render_buffer.cu"
#include "../instant-ngp/src/thread_pool.cpp"
#include "../instant-ngp/src/tinyexr_wrapper.cu"
#include "../instant-ngp/src/tinyobj_loader_wrapper.cpp"
#include "../instant-ngp/src/triangle_bvh.cu"

namespace tcnn
{
float  sdata[32] = {0};                   // not sure why we get a link error with this
__half shmem[1024*1024] = {0};            // ditto
};

void die( std::string msg )
{
    std::cout << "ERROR: " << msg << "\n";
    exit( 1 );
}
#define dassert( expr, msg ) if ( !(expr) ) die( msg )

int main( int argc, const char * argv[] )
{
    std::string scene = "";

    for( int i = 1; i < argc; i++ )
    {
        std::string arg = argv[i];
        if        ( arg == "-scene" ) {                         scene = argv[++i];
        } else {                                                die( "unknown option: " + arg );
        }
    }

    dassert( scene != "", "no -scene specified" );
    std::string scene_file = scene + ".ingp";

    ngp::Testbed testbed;
    testbed.load_file( scene_file );
    testbed.m_train = false;
    testbed.frame();

    return 0;
}
