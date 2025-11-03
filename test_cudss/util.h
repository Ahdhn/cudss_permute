#pragma once

#include <cudss.h>
#include <stdio.h>
#include <stdlib.h>
#include <random>
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

template <typename T>
inline void fill_random(int n, T* h_in, T minn = -1.0, T maxx = 1.0)
{
    static_assert(std::is_floating_point_v<T>,
                  "fill_random() T should be floating point!");

    std::random_device rd;

    std::mt19937 gen(rd());

    std::uniform_real_distribution<T> dis(static_cast<T>(minn),
                                          static_cast<T>(maxx));

    for (int i = 0; i < n; ++i) {
        h_in[i] = dis(gen);
    }
}


template <typename T>
inline cudaDataType_t cuda_type()
{
    if (std::is_same_v<T, float>) {
        return CUDA_R_32F;
    } else if (std::is_same_v<T, double>) {
        return CUDA_R_64F;
    //} else if (std::is_same_v<T, cuComplex>) {
    //    return CUDA_C_32F;
    //} else if (std::is_same_v<T, cuDoubleComplex>) {
    //    return CUDA_C_64F;
    } else if (std::is_same_v<T, int8_t>) {
        return CUDA_R_8I;
    } else if (std::is_same_v<T, uint8_t>) {
        return CUDA_R_8U;
    } else if (std::is_same_v<T, int16_t>) {
        return CUDA_R_16I;
    } else if (std::is_same_v<T, uint16_t>) {
        return CUDA_R_16U;
    } else if (std::is_same_v<T, int32_t> || std::is_same_v<T, int>) {
        return CUDA_R_32I;
    } else if (std::is_same_v<T, uint32_t>) {
        return CUDA_R_32U;
    } else if (std::is_same_v<T, int64_t>) {
        return CUDA_R_64I;
    } else if (std::is_same_v<T, uint64_t>) {
        return CUDA_R_64U;
    } else {
        fprintf(stderr,
                "Unsupported type. Sparse/Dense Matrix in RXMesh can support "
                "different data type but for the solver, only float, double, "
                "cuComplex, and cuDoubleComplex are supported");
        // to silence compiler warning
    }
}