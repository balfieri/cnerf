// cuda.h - CPU-based emulation of CUDA
//
#ifndef _CUDA_H
#define _CUDA_H

#include <stdlib.h>

#include <cmath>
#include <fstream>
#include <iostream>

#define __host__
#define __device__
#define __global__
#define __shared__

#define cdassert(expr, msg) if ( !(expr) ) { std::cout << "ERROR: " << msg << "\n"; exit(1); }

class dim3
{
public:
    uint32_t	x;    
    uint32_t	y;    
    uint32_t	z;    

    inline dim3() {}
    inline dim3(uint32_t x, uint32_t y, uint32_t z) : x(x), y(y), z(z) {}
};

static dim3 threadIdx;
static dim3 blockIdx;
static dim3 blockDim;
static dim3 gridDim;
#define warpSize 32

#define LAUNCH_KERNEL3(kernel, gridDim3, blockDim3, shmem_size, stream, args)\
{\
    gridDim  = gridDim3;\
    blockDim = blockDim3;\
    for(uint32_t z = 0; z < gridDim.z; z++)\
    {\
        blockIdx.z = z;\
        for(uint32_t y = 0; y < gridDim.y; y++)\
        {\
            blockIdx.y = y;\
            for(uint32_t x = 0; x < gridDim.x; x++)\
            {\
                blockIdx.x = x;\
                for(uint32_t z = 0; z < blockDim.z; z++)\
                {\
                    threadIdx.z = z;\
                    for(uint32_t y = 0; y < blockDim.y; y++)\
                    {\
                        threadIdx.y = y;\
                        for(uint32_t x = 0; x < blockDim.x; x++)\
                        {\
                            threadIdx.x = x;\
                            kernel args;\
                        }\
                    }\
                }\
            }\
        }\
    }\
}\

static inline void __syncthreads(void) 
{
}

class float2
{
public:
    float x;
    float y;
};

class float3
{
public:
    float x;
    float y;
    float z;
};

class float4
{
public:
    float x;
    float y;
    float z;
    float w;
};

static inline float2 make_float2(float x, float y)
{
    float2 v;
    v.x = x;
    v.y = y;
}

static inline float3 make_float3(float x, float y, float z)
{
    float3 v;
    v.x = x;
    v.y = y;
    v.z = z;
}

static inline float4 make_float4(float x, float y, float z, float w)
{
    float4 v;
    v.x = x;
    v.y = y;
    v.z = z;
    v.w = w;
}

class double2
{
public:
    double x;
    double y;
};

class double3
{
public:
    double x;
    double y;
    double z;
};

class double4
{
public:
    double x;
    double y;
    double z;
    double w;
};

using half = __half;

class __half2
{
public:
    __half x;
    __half y;
};

class int4
{
public:
    int x;
    int y;
    int z;
    int w;
};

using cudaError_t = uint32_t;
const cudaError_t cudaSuccess = 0;
const cudaError_t cudaErrorNotYetImplemented = 31;

struct __cudaStream
{
    bool     active;
};
using cudaStream_t = __cudaStream *;
const cudaStream_t cudaStreamLegacy = nullptr;

using cudaStreamCaptureMode = uint32_t;
const cudaStreamCaptureMode cudaStreamCaptureModeGlobal = 0;
const cudaStreamCaptureMode cudaStreamCaptureModeThreadLocal = 1;
const cudaStreamCaptureMode cudaStreamCaptureModeRelaxed = 2;

using cudaStreamCaptureStatus = uint32_t;
const cudaStreamCaptureStatus cudaStreamCaptureStatusNone = 0;
const cudaStreamCaptureStatus cudaStreamCaptureStatusActive = 1;
const cudaStreamCaptureStatus cudaStreamCaptureStatusInvalidated = 2;
const cudaStreamCaptureStatus cudaErrorStreamCaptureImplicit = 3;

struct __cudaGraph;
using cudaGraph_t = __cudaGraph *;
struct __cudaGraphExec;
using cudaGraphExec_t = __cudaGraphExec *;
struct __cudaGraphNode;
using cudaGraphNode_t = __cudaGraphNode *;

using cudaGraphExecUpdateResult = uint32_t;
const cudaGraphExecUpdateResult cudaGraphExecUpdateSuccess = 0;

class cudaGraphExecUpdateResultInfo
{
public:
};

const unsigned int cudaMemAttachGlobal = 0x1;

using cudaMemcpyKind = uint32_t;
const cudaMemcpyKind cudaMemcpyHostToHost     = 0;
const cudaMemcpyKind cudaMemcpyHostToDevice   = 1;
const cudaMemcpyKind cudaMemcpyDeviceToHost   = 2;
const cudaMemcpyKind cudaMemcpyDeviceToDevice = 3;
const cudaMemcpyKind cudaMemcpyDefault        = 4;

using CUdevice = uint32_t;
using CUdeviceptr = uint8_t *;
using CUdevice_attribute = uint32_t;
const CUdevice_attribute CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102;
class cudaDeviceProp 
{
public:
    char                        name[256];
    int                         major;
    int                         minor;
    // many others not used
};
using CUresult = uint32_t;
const CUresult CUDA_SUCCESS = 0;
const CUresult CUDA_ERROR_NOT_SUPPORTED = 801;
using CUmemGenericAllocationHandle = void *;

using CUmemLocationType = uint32_t;
const CUmemLocationType CU_MEM_LOCATION_TYPE_INVALID = 0;
const CUmemLocationType CU_MEM_LOCATION_TYPE_DEVICE = 1;

struct CUmemLocation
{
    int                         id;
    CUmemLocationType           type;
};

using CUmemAllocationHandleType = uint32_t;
const CUmemAllocationHandleType CU_MEM_HANDLE_TYPE_NONE = 0x0;
const CUmemAllocationHandleType CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 0x1;
const CUmemAllocationHandleType CU_MEM_HANDLE_TYPE_WIN32 = 0x2;
const CUmemAllocationHandleType CU_MEM_HANDLE_TYPE_WIN32_KMT = 0x4;

using CUmemAllocationType = uint32_t;
const CUmemAllocationType CU_MEM_ALLOCATION_TYPE_INVALID = 0x0;
const CUmemAllocationType CU_MEM_ALLOCATION_TYPE_PINNED = 0x1;

class CUmemAllocationProp
{
public:
    unsigned char               compressionType;
    struct CUmemLocation        location;
    CUmemAllocationHandleType 	requestedHandleTypes;
    CUmemAllocationType         type;
    unsigned short              usage;
    void *                      win32HandleMetaData;
};

using CUmemAccess_flags = uint32_t;
const CUmemAccess_flags CU_MEM_ACCESS_FLAGS_PROT_NONE = 0x0;
const CUmemAccess_flags CU_MEM_ACCESS_FLAGS_PROT_READ = 0x1;
const CUmemAccess_flags CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 0x3;

class CUmemAccessDesc
{
public:
    CUmemAccess_flags           flags;
    struct CUmemLocation        location;
};

using CUmemAllocationGranularity_flags = uint32_t;
const CUmemAllocationGranularity_flags CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0x0;
const CUmemAllocationGranularity_flags CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 0x1;

typedef unsigned long long cudaSurfaceObject_t;
struct __cudaArray;
using cudaArray_t = __cudaArray *;
struct __cudaEvent
{
    bool active;
};
using cudaEvent_t = __cudaEvent *;

using cudaArrayKind = uint32_t;
const cudaArrayKind cudaArraySurfaceLoadStore = 0x02;

using enumcudaChannelFormatKind = uint32_t;
const enumcudaChannelFormatKind cudaChannelFormatKindSigned = 0;
const enumcudaChannelFormatKind cudaChannelFormatKindUnsigned = 1;
const enumcudaChannelFormatKind cudaChannelFormatKindFloat = 2;
const enumcudaChannelFormatKind cudaChannelFormatKindNone = 3;

typedef struct
{
    enumcudaChannelFormatKind 	f;
    int                         w;
    int                         x;
    int                         y;
    int                         z;
} cudaChannelFormatDesc;


using enumcudaResourceType = uint32_t;
const enumcudaResourceType cudaResourceTypeArray = 0x00;

struct cudaResourceDesc
{
    cudaArray_t                 array;
    cudaChannelFormatDesc 	desc;
    void *                      devPtr;
    size_t                      height;
//  cudaMipmappedArray_t        mipmap;
    size_t                      pitchInBytes;
    enumcudaResourceType        resType;
    size_t                      sizeInBytes;
    size_t                      width;
    struct {
        struct {
            cudaArray_t         array;
        }                       array;
    }                           res;
};

using cudaFuncAttribute = uint32_t;
const cudaFuncAttribute cudaFuncAttributeMaxDynamicSharedMemorySize = 8;

static const char * cudaGetErrorString(cudaError_t error)
{
    (void)error;
    switch( error )
    {
        case cudaSuccess:                       return "success";
        case cudaErrorNotYetImplemented:        return "not yet implemented";
        default:                                return "unknown error";
    }
}

static int __curr_device = 0;

static cudaError_t cudaGetDeviceCount(int* pCount)
{
    *pCount = 1;
    return cudaSuccess;
}

static cudaError_t cudaSetDevice(int device)
{
    __curr_device = device;
    return cudaSuccess;
}

static cudaError_t cudaGetDevice(int* pDevice)
{
    *pDevice = __curr_device;
    return cudaSuccess;
}

static cudaError_t cudaDeviceSynchronize(void)
{
    return cudaSuccess;
}

static cudaError_t cudaGetDeviceProperties(cudaDeviceProp* pProp, int device)
{
    sprintf(pProp->name, "emulated_cuda%d", device);
    pProp->major = 9;
    pProp->minor = 0;
    return cudaSuccess;
}

static cudaError_t cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev)
{
    (void)pi;
    (void)attrib;
    (void)dev;
    return cudaErrorNotYetImplemented;
}

static cudaError_t cudaStreamCreate(cudaStream_t* pStream)
{
    cudaStream_t stream = new __cudaStream;
    stream->active = true;
    *pStream = stream;
    return cudaSuccess;
}

static cudaError_t cudaStreamDestroy(cudaStream_t stream)
{
    cdassert(stream->active, "destroying inactive stream");
    stream->active = false;
    delete stream;
    return cudaSuccess;
}

static cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
    (void)stream;
    return cudaSuccess;
}

static cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus* pCaptureStatus)
{
    (void)stream;
    *pCaptureStatus = cudaStreamCaptureStatusNone; // pretend already capturing
    return cudaErrorNotYetImplemented;
}

static cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode)
{
    (void)stream;
    (void)mode;
    return cudaErrorNotYetImplemented;
}

static cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph)
{
    (void)stream;
    (void)pGraph;
    return cudaErrorNotYetImplemented;
}

static cudaError_t cudaGraphDestroy(cudaGraph_t graph)
{
    (void)graph;
    return cudaErrorNotYetImplemented;
}

static cudaError_t cudaGraphInstantiate (cudaGraphExec_t* pGraphExec, cudaGraph_t graph, cudaGraphNode_t* pErrorNode, char* pLogBuffer, size_t bufferSize)
{
    (void)pGraphExec;
    (void)graph;
    (void)pErrorNode;
    (void)pLogBuffer;
    (void)bufferSize;
    return cudaErrorNotYetImplemented;
}

static cudaError_t cudaGraphExecUpdate(cudaGraphExec_t graphExec, cudaGraph_t graph, cudaGraphNode_t* pErrorNode, cudaGraphExecUpdateResult* result)
{
    (void)graphExec;
    (void)graph;
    (void)pErrorNode;
    (void)result;
    return cudaErrorNotYetImplemented;
}

static cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream)
{
    (void)graphExec;
    (void)stream;
    return cudaErrorNotYetImplemented;
}

static cudaError_t cudaGraphExecDestroy(cudaGraphExec_t exec)
{
    (void)exec;
    return cudaErrorNotYetImplemented;
}

static cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t* pSurface, const cudaResourceDesc* pResDesc)
{
    (void)pSurface;
    (void)pResDesc;
    return cudaErrorNotYetImplemented;
}

static cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surface)
{
    (void)surface;
    return cudaErrorNotYetImplemented;
}

static cudaError_t cudaFreeArray(cudaArray_t array)
{
    (void)array;
    return cudaErrorNotYetImplemented;
}

template<typename T>
static cudaChannelFormatDesc cudaCreateChannelDesc(void)
{
    cudaChannelFormatDesc desc;
    desc.f = cudaChannelFormatKindFloat;
    desc.x = 0;
    desc.y = 0;
    desc.z = 0;
    desc.w = 0;
    if ( std::is_same<T, float>::value ) {
        desc.x = sizeof(float)*8;
    } else if ( std::is_same<T, float2>::value ) {
        desc.x = sizeof(float)*8;
        desc.y = sizeof(float)*8;
    } else if ( std::is_same<T, float3>::value ) {
        desc.x = sizeof(float)*8;
        desc.y = sizeof(float)*8;
        desc.z = sizeof(float)*8;
    } else if ( std::is_same<T, float4>::value ) {
        desc.x = sizeof(float)*8;
        desc.y = sizeof(float)*8;
        desc.z = sizeof(float)*8;
        desc.w = sizeof(float)*8;
    } else {
        cdassert( false, "unknown T in cudaCreateChannelDesc()" );
    }
    return desc;
}

static cudaError_t cudaMemGetInfo(size_t* pFree, size_t* pTotal)
{
    (void)pFree;
    (void)pTotal;
    return cudaErrorNotYetImplemented;
}

static cudaError_t cudaMalloc(uint8_t** devPtr, size_t size)
{
    *devPtr = (uint8_t*)malloc(size);
    return cudaSuccess;
}

static cudaError_t cudaMallocManaged(uint8_t** devPtr, size_t size, unsigned int flags=cudaMemAttachGlobal)
{
    (void)flags;
    return cudaMalloc(devPtr, size);
}

static cudaError_t cudaMallocArray(cudaArray_t* pArray, const cudaChannelFormatDesc* pDesc, size_t x, size_t y, cudaArrayKind kind)
{
    (void)pArray;
    (void)pDesc;
    (void)x;
    (void)y;
    (void)kind;
    return cudaErrorNotYetImplemented;
}

static cudaError_t cudaFree(void* devPtr)
{
    free(devPtr);
    return cudaSuccess;
}

static cudaError_t cudaMemset(void* dst, int c, size_t len, cudaStream_t stream=0)
{
    (void)stream;
    memset(dst, c, len);
    return cudaSuccess;
}

static cudaError_t cudaMemsetAsync(void* dst, int c, size_t len, cudaStream_t stream=0)
{
    (void)stream;
    memset(dst, c, len);
    return cudaSuccess;
}

static cudaError_t cudaMemCpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)
{
    (void)kind;
    memcpy(dst, src, count);
    return cudaSuccess;
}

static cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)
{
    (void)kind;
    memcpy(dst, src, count);
    return cudaSuccess;
}

static cudaError_t cudaMemcpyAsync(void* dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream=0)
{
    (void)stream;
    return cudaMemcpy(dst, src, count, kind);
}

static cudaError_t cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src, int srcDevice, size_t count, cudaStream_t stream=0)
{
    (void)dstDevice;
    (void)srcDevice;
    return cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream);
}

static cudaError_t cudaMemcpy2DFromArray(void * dst, size_t dpitch, const cudaArray_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind )
{
    (void)dst;
    (void)dpitch;
    (void)src;
    (void)wOffset;
    (void)hOffset;
    (void)width;
    (void)height;
    (void)kind;
    return CUDA_ERROR_NOT_SUPPORTED;
}

static CUresult cuGetErrorName(CUresult result, const char ** pStr)
{
    (void)result;
    *pStr = "unknown";
    return CUDA_SUCCESS;
}

static CUresult cuMemGetAllocationGranularity (size_t* granularity, const CUmemAllocationProp* prop, CUmemAllocationGranularity_flags option)
{
    (void)granularity;
    (void)prop;
    (void)option;
    return CUDA_ERROR_NOT_SUPPORTED;            
}

static CUresult cuMemAddressReserve(CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags)
{
    (void)ptr;
    (void)size;
    (void)alignment;
    (void)addr;
    (void)flags;
    return CUDA_ERROR_NOT_SUPPORTED;            // use mmap() if needed
}

static CUresult cuMemCreate(CUmemGenericAllocationHandle* handle, size_t size, const CUmemAllocationProp* prop, unsigned long long flags)
{
    (void)handle;
    (void)size;
    (void)prop;
    (void)flags;
    return CUDA_ERROR_NOT_SUPPORTED;            // use mmap() if needed
}

static CUresult cuMemMap (CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags)
{
    (void)ptr;
    (void)size;
    (void)offset;
    (void)handle;
    (void)flags;
    return CUDA_ERROR_NOT_SUPPORTED;            // use mmap() if needed
}

static CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size)
{
    (void)ptr;
    (void)size;
    return CUDA_ERROR_NOT_SUPPORTED;            // use munmap() if needed
}

static CUresult cuMemSetAccess( CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count)
{
    (void)ptr;
    (void)size;
    (void)desc;
    (void)count;
    return CUDA_ERROR_NOT_SUPPORTED;           
}

static CUresult cuMemUnmap(CUdeviceptr ptr, size_t size)
{
    (void)ptr;
    (void)size;
    return CUDA_ERROR_NOT_SUPPORTED;            // use munmap() if needed
}

static CUresult cuMemRelease(CUmemGenericAllocationHandle handle)
{
    free(handle);
    return CUDA_SUCCESS;
}

static cudaError_t cudaEventCreate(cudaEvent_t* pEvent)
{
    cudaEvent_t event = new __cudaEvent;
    event->active = true;
    *pEvent = event;
    return cudaSuccess;
}

static cudaError_t cudaEventDestroy(cudaEvent_t event)
{
    cdassert(event->active, "destroying inactive event");
    event->active = false;
    delete event;
    return cudaSuccess;
}

static cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags)
{
    (void)stream;
    (void)event;
    (void)flags;
    return cudaErrorNotYetImplemented;
}

static cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    (void)event;
    (void)stream;
    return cudaErrorNotYetImplemented;
}

static cudaError_t surf2Dread(float4 * pPixel, cudaSurfaceObject_t surface, uint32_t x, uint32_t y)
{
    (void)pPixel;
    (void)surface;
    (void)x;
    (void)y;
    return cudaErrorNotYetImplemented;
}

static cudaError_t surf2Dwrite(float4 pixel, cudaSurfaceObject_t surface, uint32_t x, uint32_t y)
{
    (void)pixel;
    (void)surface;
    (void)x;
    (void)y;
    return cudaErrorNotYetImplemented;
}

static cudaError_t surf2Dwrite(float2 pixel, cudaSurfaceObject_t surface, uint32_t x, uint32_t y)
{
    return surf2Dwrite(make_float4(pixel.x, pixel.y, pixel.x, pixel.y), surface, x, y);
}

static cudaError_t surf2Dwrite(float pixel, cudaSurfaceObject_t surface, uint32_t x, uint32_t y)
{
    return surf2Dwrite(make_float4(pixel, pixel, pixel, pixel), surface, x, y);
}

template<class T>
static cudaError_t cudaFuncSetAttribute( T* entry, cudaFuncAttribute attr, int value)
{
    (void)entry;
    (void)attr;
    (void)value;
    return cudaErrorNotYetImplemented;
}

static float normcdff(float a)
{
    (void)a;
    std::cout << "ERROR: normcdff() not yet implemented\n";
    exit(1);
    return 0.0;
}

static float rsqrtf(float a)
{
    (void)a;
    return powf(a, -0.5);	// CUDA manual hints that this is how it could be implemented
}

static void sincosf(float a, float* s, float* c)
{
    *s = sinf(a);
    *c = cosf(a);
}

static float __expf(float a)
{
    return expf(a);
}

inline float __uint_as_float( unsigned int u ) 
{ 
    float f; 
    memcpy( &f, &u, sizeof(f) ); 
    return f; 
}

inline unsigned int __float_as_uint( float f )
{ 
    unsigned int u; 
    memcpy( &u, &f, sizeof(u) );
    return u; 
}

static float __frcp_rn(float a)
{
    // TODO: needs to be round-to-even
    return 1.0f/a;
}

static float __saturatef(float a)
{
    if (a < 0.0f) return 0.0f;
    if (a > 1.0f) return 1.0f;
    return a;
}

static float __sinf(float a)
{
    return sinf(a);
}

static float __cosf(float a)
{
    return sinf(a);
}

static int min(int a, int b)
{
    return (a <= b) ? a : b;
}

static int max(int a, int b)
{
    return (a >= b) ? a : b;
}

static uint32_t umin(uint32_t a, uint32_t b)
{
    return (a <= b) ? a : b;
}

static uint32_t umax(uint32_t a, uint32_t b)
{
    return (a >= b) ? a : b;
}

static int atomicAdd(int *ptr, int val)
{
    int r = *ptr;
    *ptr += val;
    return r;
}

static uint32_t atomicAdd(uint32_t *ptr, uint32_t val)
{
    uint32_t r = *ptr;
    *ptr += val;
    return r;
}

static float atomicAdd(float *ptr, float val)
{
    float r = *ptr;
    *ptr += val;
    return r;
}

static __half2 atomicAdd(__half2 *ptr, __half2 val)
{
    __half2 r = *ptr;
    ptr->x += val.x;
    ptr->y += val.y;
    return r;
}

static int atomicMax(int *ptr, int val)
{
    int r = *ptr;
    *ptr = (r >= val) ? r : val;
    return r;
}

static uint32_t atomicMax(uint32_t *ptr, uint32_t val)
{
    uint32_t r = *ptr;
    *ptr = (r >= val) ? r : val;
    return r;
}

static float atomicMax(float *ptr, float val)
{
    float r = *ptr;
    *ptr = (r >= val) ? r : val;
    return r;
}

static float __shfl_sync(unsigned mask, const float var, int srcLane, const int width=warpSize)
{
    (void)mask;
    (void)var;
    (void)srcLane;
    (void)width;
    cdassert(false, "__shfl_sync not yet implemented");
    return 0.0f;
}

static float __shfl_xor_sync(unsigned mask, const float var, const float delta, const int width=warpSize)
{
    (void)mask;
    (void)var;
    (void)delta;
    (void)width;
    cdassert(false, "__shfl_xor_sync not yet implemented");
    return 0.0f;
}

#endif
