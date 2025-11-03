#pragma once

#include <cudss.h>

template <typename T>
T compute_residual_abs_norm(int                   n,
                            const int*            csr_offsets_h,
                            const int*            csr_columns_h,
                            const T*              csr_values_h,
                            const T*              x_values_h,
                            const T*              b_values_h,
                            cudssMatrixViewType_t mview)
{
    std::vector<T> Ax(n, 0.0);

    for (int row = 0; row < n; ++row) {
        for (int idx = csr_offsets_h[row]; idx < csr_offsets_h[row + 1];
             ++idx) {
            int col = csr_columns_h[idx];
            T   val = csr_values_h[idx];

            switch (mview) {
                case CUDSS_MVIEW_FULL:
                    Ax[row] += val * x_values_h[col];
                    break;
                case CUDSS_MVIEW_UPPER:
                    if (col >= row) {
                        Ax[row] += val * x_values_h[col];
                        if (col != row)
                            Ax[col] += val * x_values_h[row];
                    }
                    break;
                case CUDSS_MVIEW_LOWER:
                    if (col <= row) {
                        Ax[row] += val * x_values_h[col];
                        if (col != row)
                            Ax[col] += val * x_values_h[row];
                    }
                    break;
            }
        }
    }

    // Compute L2 norm of residual
    T error = 0.0;
    for (int i = 0; i < n; ++i) {
        T diff = Ax[i] - b_values_h[i];
        error += diff * diff;
    }

    return std::sqrt(error);
}