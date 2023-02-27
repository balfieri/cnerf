IDIRS=. \
      ../ \
      ../instant-ngp/include \
      ../instant-ngp/dependencies \
      ../instant-ngp/dependencies/filesystem \
      ../instant-ngp/dependencies/nanovdb \
      ../instant-ngp/dependencies/NaturalSort \
      ../tiny-cuda-nn/include \
      ../tiny-cuda-nn/dependencies \
      ../tinylogger \
      ../eigen \
      ../zstr/src \
      ../tinyexr/deps/miniz \

INCLUDES=$(patsubst %,-I%, $(IDIRS))
LIBS=-lz -lpthread -lfmt
FLAGS=-std=c++17 -O3 -Werror -Wextra -Wstrict-aliasing -pedantic -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wmissing-include-dirs  -Woverloaded-virtual -Wredundant-decls -Wsign-promo -Wstrict-overflow=5 -Wswitch-default -Wundef -g -DTCNN_MIN_GPU_ARCH=90 -D__CUDA_EMULATION__ -DNGP_VERSION=\"1.0\" -D__CUDACC_VER_MAJOR__=11 -D__CUDACC_VER_MINOR__=0 -DMINIZ_NO_ZLIB_COMPATIBLE_NAMES -DNO_CUTLASS $(INCLUDES) $(LIBS)

OS=$(shell uname)
ifeq ($(OS), Darwin)
CC=clang
GPP=g++
FLAGS+=-Wno-shift-count-overflow -D__half=_Float16 -Wno-unused-parameter -Wno-cast-qual -Wno-return-type -Wno-sign-compare -Wno-deprecated-copy -Wno-switch -Wno-deprecated-declarations -Wno-delete-abstract-non-virtual-dtor -Wno-missing-field-initializers
else
ifeq ($(OS), Linux)
FLAGS+=-Wno-empty-body -Wno-strict-overflow -Wno-switch-default -Wno-shift-negative-value -Wno-maybe-uninitialized -Wno-sign-promo -Wno-overloaded-virtual -Wshift-count-overflow -DNO_FMT_LL 
GPP=export PATH=/home/utils/gcc-8.1.0/bin:/usr/local/bin:/bin:/usr/bin:/sbin:/usr/sbin:/usr/lib:/etc:/home/nv/bin:/home/utils/p4-2018.1/bin:/usr/local/lsf/bin; g++
else
ifeq ($(findstring CYGWIN,$(OS)),CYGWIN)
FLAGS+=-Wno-missing-include-dirs -Wno-empty-body -Wno-strict-overflow -Wno-switch-default -Wno-shift-negative-value -Wno-maybe-uninitialized -Wno-sign-promo -Wno-overloaded-virtual -Wno-unused-result -DNO_FMT_LL -U__STRICT_ANSI__
GPP=g++
else
GPP=UNKNOWN_OPERATING_SYSTEM
endif
endif
endif

#DEPS=Makefile $(patsubst %, %/*.h, $IDIRS)
DEPS=Makefile

cnerf: cnerf.cpp ${DEPS}
	$(GPP) $(FLAGS) $(EXTRA_CFLAGS) -o cnerf cnerf.cpp $(LIBS)

clean:
	rm -fr render *.o *.dSYM
