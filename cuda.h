// cuda.h - CPU-based emulation of CUDA
//
#ifndef _CUDA_H
#define _CUDA_H

#include <stdlib.h>

#include <cmath>
#include <fstream>

#define __host__
#define __device__
#define __global__

class dim3
{
public:
    int	x;    
    int	y;    
    int	z;    
};

static dim3 threadIdx;
static dim3 blockIdx;
static dim3 blockDim;
static dim3 gridDim;

#define LAUNCH_KERNEL3(kernel, gridDim3, blockDim3, shmem_size, stream, args)\
{\
    gridDim  = gridDim3;\
    blockDim = blockDim3;\
    for(int z = 0; z < gridDim.z; z++)\
    {\
        blockIdx.z = z;\
        for(int y = 0; y < gridDim.y; y++)\
        {\
            blockIdx.y = y;\
            for(int x = 0; x < gridDim.x; x++)\
            {\
                blockIdx.x = x;\
                for(int z = 0; z < blockDim.z; z++)\
                {\
                    threadIdx.z = z;\
                    for(int y = 0; y < blockDim.y; y++)\
                    {\
                        threadIdx.y = y;\
                        for(int x = 0; x < blockDim.x; x++)\
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

class __half2
{
public:
    __half x;
    __half y;
};

using cudaError_t = uint32_t;
const cudaError_t cudaSuccess = 0;
const cudaError_t cudaErrorNotYetImplemented = 31;

using cudaStream_t = void *;
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

using cudaGraph_t = void *;
using cudaGraphExec_t = void *;
using cudaGraphNode_t = void *;

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

using CUdeviceptr = uint8_t *;
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

static cudaError_t cudaDeviceSynchronize(void)
{
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

static cudaError_t cudaFree(void* devPtr)
{
    free(devPtr);
    return cudaSuccess;
}

static cudaError_t cudaMemCpy(void* dst, void* src, size_t count, cudaMemcpyKind kind)
{
    (void)kind;
    memcpy(dst, src, count);
    return cudaSuccess;
}


static CUresult cuGetErrorName(CUresult result, const char ** pStr)
{
    (void)result;
    *pStr = "unknown";
    return CUDA_SUCCESS;
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

#endif
