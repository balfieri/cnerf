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
    int      width  = 1920;
    int      height = 1080;
    bool     linear = true;

    for( int i = 1; i < argc; i++ )
    {
        std::string arg = argv[i];
        if        ( arg == "-scene" ) {                         scene = argv[++i];
        } else if ( arg == "-w" ) {                             width = std::atoi( argv[++i] );
        } else if ( arg == "-h" ) {                             height = std::atoi( argv[++i] );
        } else if ( arg == "-linear" ) {                        linear = std::atoi( argv[++i] );
        } else {                                                die( "unknown option: " + arg );
        }
    }

    dassert( scene != "", "no -scene specified" );
    std::string scene_file = scene + ".ingp";

    ngp::Testbed testbed;
    testbed.load_file( scene_file );
    testbed.m_train = false;
    testbed.m_windowless_render_surface.resize( {width, height} );
    testbed.m_windowless_render_surface.reset_accumulation();
    testbed.set_camera_from_time( 0.0f );
    testbed.m_smoothed_camera = testbed.m_camera;
    testbed.render_frame( testbed.m_stream.get(),
                          testbed.m_smoothed_camera, // sample_start_cam_matrix,
                          testbed.m_smoothed_camera, // sample_end_cam_matrix,
                          testbed.m_smoothed_camera, // prev_camera_matrix,
                          testbed.m_screen_center,
                          testbed.m_relative_focal_length,
                          {0.0f, 0.0f, 0.0f, 1.0f},
                          {},
                          {},
                          testbed.m_visualized_dimension,
                          testbed.m_windowless_render_surface,
                          !linear );

    float * buf_ptr = new float[height*width*4];
    CUDA_CHECK_THROW( cudaMemcpy2DFromArray( buf_ptr, 
                                             width * sizeof(float) * 4, 
                                             testbed.m_windowless_render_surface.surface_provider().array(), 
                                             0, 0, 
                                             width * sizeof(float) * 4, 
                                             height, 
                                             cudaMemcpyDeviceToHost ) );

    return 0;
}
