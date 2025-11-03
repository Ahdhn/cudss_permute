#pragma once

#include <numeric>

#include "metis.h"

idx_t* metis_permute(int n, const int* csr_offsets_h, const int* csr_columns_h)
{

    std::vector<idx_t> xadj(n + 1, 0);  // csr_offsets_h

    std::vector<idx_t> adjncy;

    // filer out the diagonal entires
    for (int i = 0; i < n; ++i) {
        xadj[i] = csr_offsets_h[i + 1] - csr_offsets_h[i];
        xadj[i] -= 1;
    }

    // prefix sum
    int prev = 0;
    for (int i = 0; i <= n; ++i) {
        int temp = xadj[i];
        xadj[i]  = prev;
        prev += temp;
    }


    adjncy.resize(xadj[n]);
    std::vector<int> offset = xadj;
    for (int i = 0; i < n; ++i) {
        int row = i;
        for (int j = csr_offsets_h[i]; j < csr_offsets_h[i + 1]; ++j) {
            int col = csr_columns_h[j];
            if (row != col) {
                int dest     = offset[row]++;
                adjncy[dest] = col;
            }
        }
    }

    idx_t* h_permute = (idx_t*)malloc(n * sizeof(idx_t));

    idx_t* h_iperm = (idx_t*)malloc(n * sizeof(idx_t));


    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);

    options[METIS_OPTION_NUMBERING] = 0;

    int metis_ret = METIS_NodeND(
        &n, xadj.data(), adjncy.data(), NULL, options, h_permute, h_iperm);

    if (metis_ret != 1) {
        fprintf(stderr, "METIS Failed");
    }

    return h_permute;
}