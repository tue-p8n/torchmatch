// AVX2 two-minimum scan for jonker_compact_kernel.
// Included only when compiled with -mavx2.
#pragma once
#ifdef __AVX2__

#include <cstdint>
#include <immintrin.h>
#include <limits>
#include <tuple>

namespace match_jonker::detail {

// AVX2 find_umins_regular: subtract v from cost row i, track two minima.
// Equivalent to the scalar loop in jonker_compact_core.h but processes 4 doubles/iter.
template <typename idx>
inline std::tuple<double, double, idx, idx>
find_umins_avx2(idx dim, idx i, const double *assign_cost, const double *v) {
    const double *row = assign_cost + static_cast<size_t>(i) * dim;

    // Scalar initialization from j=0 (no dependency on previous iteration).
    double umin = row[0] - v[0];
    double usubmin = std::numeric_limits<double>::max();
    idx j1 = 0, j2 = -1;

    // SIMD sweep over columns [1, dim) in strides of 4.
    // Condition: j + 4 <= dim ensures we never read past the end of row/v.
    idx j = 1;
    for (; j + 4 <= dim; j += 4) {
        __m256d c4 = _mm256_loadu_pd(row + j);
        __m256d v4 = _mm256_loadu_pd(v + j);
        __m256d h4 = _mm256_sub_pd(c4, v4);

        double h[4];
        _mm256_storeu_pd(h, h4);
        for (int k = 0; k < 4; ++k) {
            double hk = h[k];
            if (hk < usubmin) {
                if (hk >= umin) {
                    usubmin = hk;
                    j2 = static_cast<idx>(j + k);
                } else {
                    usubmin = umin;
                    umin = hk;
                    j2 = j1;
                    j1 = static_cast<idx>(j + k);
                }
            }
        }
    }

    // Scalar tail for remainder columns (fewer than 4 left).
    for (; j < dim; ++j) {
        double hk = row[j] - v[j];
        if (hk < usubmin) {
            if (hk >= umin) {
                usubmin = hk;
                j2 = j;
            } else {
                usubmin = umin;
                umin = hk;
                j2 = j1;
                j1 = j;
            }
        }
    }

    return {umin, usubmin, j1, j2};
}

} // namespace match_jonker::detail
#endif // __AVX2__
