// cnerf.cpp - C++ single-threaded reference model based on instant NeRF sources.
//
// Initially, this will do only rendering (forward pass through neural net).
//

#include "../instant-ngp/src/main.cu"
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
#include "../instant-ngp/src/openxr_hmd.cu"
#include "../instant-ngp/src/render_buffer.cu"
#include "../instant-ngp/src/tinyexr_wrapper.cu"
#include "../instant-ngp/src/tinyobj_loader_wrapper.cu"
#include "../instant-ngp/src/triangle_bvh.cu"
