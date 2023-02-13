Bob Alfieri

This is a temporary repo that is being used to attempt to build
instant-ngp/src/main.cu to run on a single CPU.

I am using my own GNU-style Makefile for now, one that is familiar to me.
It works across different platforms, but I am currently building only on macOS using clang++.

We can worry later about trying to put this back into instant-ngp.

See the IDIRS at the top of Makefile for parallel repos that must exist to get this to build.
