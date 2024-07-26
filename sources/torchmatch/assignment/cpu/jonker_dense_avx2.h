// AVX2 implementations of Jonker dense hot loops.
// Included only when compiled with -mavx2.
#pragma once
#ifdef __AVX2__

#include <cstdint>
#include <immintrin.h>

namespace match_jonker::detail::flat {

using int_t = int32_t;
using uint_t = uint32_t;

// Vectorized: d[i] = row[i] - v[i] and pred[i] = start_i for i in [0, n).
// The scalar remainder handles n % 4 tail elements.
inline void init_d_avx2(uint_t n, const double *row, const double *v, double *d,
                        int_t start_i, int_t *pred) {
    uint_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d r4 = _mm256_loadu_pd(row + i);
        __m256d v4 = _mm256_loadu_pd(v + i);
        _mm256_storeu_pd(d + i, _mm256_sub_pd(r4, v4));
        pred[i] = start_i;
        pred[i + 1] = start_i;
        pred[i + 2] = start_i;
        pred[i + 3] = start_i;
    }
    for (; i < n; ++i) {
        d[i] = row[i] - v[i];
        pred[i] = start_i;
    }
}

} // namespace match_jonker::detail::flat
#endif // __AVX2__
