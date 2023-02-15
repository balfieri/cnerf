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
