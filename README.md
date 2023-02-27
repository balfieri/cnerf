This is a temporary repo that is being used to build instant-ngp/src/main.cu to run on a single CPU. 

./cnerf.cpp basically #includes in all .cu files.

./cuda.h contains stubs for all functions. Anything not implemented (many things) will cause aborts. As we run this for 
a basic rendering operation, we'll see which functions really need to get emulated properly and fill them in at that time.

I am using my own GNU-style Makefile for now, one that is familiar to me.
It works across the 3 main platforms, but I am currently building only on macOS using clang++.
We can worry later about trying to put this back into instant-ngp cmake infrastructure.

See the IDIRS at the top of Makefile for parallel repos that must exist to get this to build.

Summary of changes that I had to make to tiny-cuda-nn and instant-ngp source files (not yet checked in as these will need to be reviewed):
* any kernel launch goes through a LAUNCH_KERNEL[13] macro to hide the <<<...>>> business and to allow it to be emulated on a CPU.
* I currently have a #define NO_CUTLASS that ifdefs out all Cutlass calls and replaces them with aborts. I'm hoping not to have to emulate Cutlass.
* a few other misc changes to keep clang happy.
