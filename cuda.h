// cuda.h - CPU-based emulation of CUDA
//
#ifndef _CUDA_H
#define _CUDA_H

using cudaError_t = uint32_t;
const cudaError_t cudaSuccess = 0;

using cudaStream_t = void *;
const cudaStream_t cudaStreamLegacy = nullptr;

using cudaStreamCaptureStatus = uint32_t;
const cudaStreamCaptureStatus cudaStreamCaptureStatusNone = 0;
const cudaStreamCaptureStatus cudaErrorStreamCaptureImplicit = 1;

using cudaGraph_t = void *;
using cudaGraphExec_t = void *;

static const char * cudaGetErrorString(cudaError_t error)
{
    (void)error;
    return "<unknown cuda error>";
}

static cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus* capture_status)
{
    (void)stream;
    *capture_status = cudaStreamCaptureStatusNone; // pretend already capturing
    return cudaSuccess;
}

#endif
