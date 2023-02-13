// cuda.h - CPU-based emulation of CUDA
//
#ifndef _CUDA_H
#define _CUDA_H

#include <stdlib.h>

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

using CUdeviceptr = void *;
using CUresult = uint32_t;
const CUresult CUDA_SUCCESS = 0;

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

static cudaError_t cudaMalloc(void** devPtr, size_t size)
{
    *devPtr = malloc(size);
    return cudaSuccess;
}

static cudaError_t cudaMalloc(uint8_t** devPtr, size_t size)
{
    return cudaMalloc((void **)devPtr, size);
}

static cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags=cudaMemAttachGlobal)
{
    return cudaMalloc(devPtr, size);
}

static cudaError_t cudaMallocManaged(uint8_t** devPtr, size_t size, unsigned int flags=cudaMemAttachGlobal)
{
    return cudaMallocManaged((void **)devPtr, size, flags);
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

#endif
