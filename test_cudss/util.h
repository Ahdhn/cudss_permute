#pragma once

#include <cudss.h>
#include <stdio.h>
#include <stdlib.h>
//********************** CUDA HandelError
#ifndef _CUDA_ERROR_
#define _CUDA_ERROR_
inline void HandleError(cudaError_t err, const char* file, int line)
{
    // Error handling micro, wrap it around function whenever possible
    if (err != cudaSuccess) {
        printf("\n%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define CUDA_ERROR(err) (HandleError(err, __FILE__, __LINE__))
#endif
//******************************************************************************


//********************** DIVIDE_UP
// used for integer rounding
#define DIVIDE_UP(num, divisor) (num + divisor - 1) / (divisor)
//******************************************************************************

#ifndef _CUDSS_ERROR_
#define _CUDSS_ERROR_
inline void cudssHandleError(cudssStatus_t status,
                             const char*   file,
                             const int     line)
{

    if (status != CUDSS_STATUS_SUCCESS) {
        auto cudssGetErrorString = [](cudssStatus_t status) {
            switch (status) {
                case CUDSS_STATUS_SUCCESS:
                    return "CUDSS_STATUS_SUCCESS";
                case CUDSS_STATUS_NOT_INITIALIZED:
                    return "CUDSS_STATUS_NOT_INITIALIZED";
                case CUDSS_STATUS_ALLOC_FAILED:
                    return "CUDSS_STATUS_ALLOC_FAILED";
                case CUDSS_STATUS_INVALID_VALUE:
                    return "CUDSS_STATUS_INVALID_VALUE";
                case CUDSS_STATUS_NOT_SUPPORTED:
                    return "CUDSS_STATUS_NOT_SUPPORTED";
                case CUDSS_STATUS_EXECUTION_FAILED:
                    return "CUDSS_STATUS_EXECUTION_FAILED";
                case CUDSS_STATUS_INTERNAL_ERROR:
                    return "CUDSS_STATUS_INTERNAL_ERROR";
                default:
                    return "UNKNOWN_ERROR";
            }
        };

        printf(
            "\n%s in %s at line %d\n", cudssGetErrorString(status), file, line);        

        exit(EXIT_FAILURE);
    }
}
#define CUDSS_ERROR(err) (cudssHandleError(err, __FILE__, __LINE__))
#endif

//********************** CUDATimer
class CUDATimer
{
   public:
    CUDATimer()
    {
        CUDA_ERROR(cudaEventCreate(&m_start));
        CUDA_ERROR(cudaEventCreate(&m_stop));
    }
    ~CUDATimer()
    {
        CUDA_ERROR(cudaEventDestroy(m_start));
        CUDA_ERROR(cudaEventDestroy(m_stop));
    }
    void start(cudaStream_t stream = 0)
    {
        m_stream = stream;
        CUDA_ERROR(cudaEventRecord(m_start, m_stream));
    }
    void stop()
    {
        CUDA_ERROR(cudaEventRecord(m_stop, m_stream));
        CUDA_ERROR(cudaEventSynchronize(m_stop));
    }
    float elapsed_millis()
    {
        float elapsed = 0;
        CUDA_ERROR(cudaEventElapsedTime(&elapsed, m_start, m_stop));
        return elapsed;
    }

   private:
    cudaEvent_t  m_start, m_stop;
    cudaStream_t m_stream;
};
//******************************************************************************