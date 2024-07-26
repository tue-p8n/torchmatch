// Flat-pointer variants of jonker_internal_flat.
// Replaces the cost_t** indirection with a single flat pointer + stride,
// enabling direct AVX2 loads and eliminating the row-pointer thread-local.
// Templated on scalar_t to support both float32 and float64 inputs.
#pragma once

#ifdef __AVX2__
#include "jonker_dense_avx2.h"
#endif

#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace match_jonker::detail::flat {

using int_t = int32_t;
using uint_t = uint32_t;
using boolean = char;

template <typename scalar_t>
constexpr scalar_t LARGE_V = static_cast<scalar_t>(1e18);

// Returns the index of the minimum-d column among cols[lo..n), moving ties
// to the front of the scan range [lo, hi).
template <typename scalar_t>
inline uint_t find_dense(uint_t n, uint_t lo, const scalar_t *d, int_t *cols) {
    uint_t hi = lo + 1;
    scalar_t mind = d[cols[lo]];
    for (uint_t k = hi; k < n; ++k) {
        int_t j = cols[k];
        if (d[j] <= mind) {
            if (d[j] < mind) {
                hi = lo;
                mind = d[j];
            }
            cols[k] = cols[hi];
            cols[hi++] = j;
        }
    }
    return hi;
}

template <typename scalar_t>
inline int_t scan_dense(uint_t n, const scalar_t *cost_flat, uint_t stride, uint_t *plo,
                        uint_t *phi, scalar_t *d, int_t *cols, int_t *pred,
                        const int_t *y, const scalar_t *v) {
    uint_t lo = *plo, hi = *phi;
    while (lo != hi) {
        int_t j = cols[lo++];
        int_t i = y[j];
        scalar_t mind = d[j];
        scalar_t h = cost_flat[static_cast<size_t>(i) * stride + j] - v[j] - mind;
        for (uint_t k = hi; k < n; ++k) {
            j = cols[k];
            scalar_t cred = cost_flat[static_cast<size_t>(i) * stride + j] - v[j] - h;
            if (cred < d[j]) {
                d[j] = cred;
                pred[j] = i;
                if (cred == mind) {
                    if (y[j] < 0)
                        return j;
                    cols[k] = cols[hi];
                    cols[hi++] = j;
                }
            }
        }
    }
    *plo = lo;
    *phi = hi;
    return -1;
}

template <typename scalar_t>
inline int_t find_path_dense(uint_t n, const scalar_t *cost_flat, uint_t stride,
                             int_t start_i, const int_t *y, scalar_t *v, int_t *pred,
                             int_t *cols, scalar_t *d) {
    uint_t lo = 0, hi = 0;
    int_t final_j = -1;
    uint_t n_ready = 0;
    const scalar_t *row = cost_flat + static_cast<size_t>(start_i) * stride;

    for (uint_t i = 0; i < n; ++i)
        cols[i] = static_cast<int_t>(i);

#ifdef __AVX2__
    if constexpr (std::is_same_v<scalar_t, double>) {
        if (__builtin_cpu_supports("avx2")) {
            init_d_avx2(n, row, v, d, start_i, pred);
            goto done_init;
        }
    }
#endif
    for (uint_t i = 0; i < n; ++i) {
        d[i] = row[i] - v[i];
        pred[i] = start_i;
    }
#ifdef __AVX2__
done_init:;
#endif

    while (final_j == -1) {
        if (lo == hi) {
            n_ready = lo;
            hi = find_dense<scalar_t>(n, lo, d, cols);
            for (uint_t k = lo; k < hi; ++k)
                if (y[cols[k]] < 0) {
                    final_j = cols[k];
                }
        }
        if (final_j == -1)
            final_j = scan_dense<scalar_t>(n, cost_flat, stride, &lo, &hi, d, cols,
                                           pred, y, v);
    }

    const scalar_t mind = d[cols[lo]];
    for (uint_t k = 0; k < n_ready; ++k)
        v[cols[k]] += d[cols[k]] - mind;

    return final_j;
}

template <typename scalar_t>
inline int_t ccrrt_dense(uint_t n, const scalar_t *cost_flat, uint_t stride,
                         int_t *free_rows, int_t *x, int_t *y, scalar_t *v,
                         boolean *unique) {
    const scalar_t LARGE = LARGE_V<scalar_t>;
    std::fill_n(unique, n, boolean(1));
    for (uint_t i = 0; i < n; ++i) {
        x[i] = -1;
        v[i] = LARGE;
        y[i] = 0;
    }

    for (uint_t i = 0; i < n; ++i) {
        const scalar_t *row = cost_flat + static_cast<size_t>(i) * stride;
        for (uint_t j = 0; j < n; ++j) {
            if (row[j] < v[j]) {
                v[j] = row[j];
                y[j] = static_cast<int_t>(i);
            }
        }
    }
    int_t j = static_cast<int_t>(n);
    do {
        --j;
        int_t i = y[j];
        if (x[i] < 0) {
            x[i] = j;
        } else {
            unique[i] = 0;
            y[j] = -1;
        }
    } while (j > 0);

    int_t n_free_rows = 0;
    for (uint_t i = 0; i < n; ++i) {
        if (x[i] < 0) {
            free_rows[n_free_rows++] = static_cast<int_t>(i);
        } else if (unique[i]) {
            int_t ji = x[i];
            scalar_t mn = LARGE;
            const scalar_t *row = cost_flat + static_cast<size_t>(i) * stride;
            for (uint_t j2 = 0; j2 < n; ++j2) {
                if (j2 == static_cast<uint_t>(ji))
                    continue;
                scalar_t c = row[j2] - v[j2];
                if (c < mn)
                    mn = c;
            }
            v[ji] -= mn;
        }
    }
    return n_free_rows;
}

template <typename scalar_t>
inline int_t carr_dense(uint_t n, const scalar_t *cost_flat, uint_t stride,
                        uint_t n_free, int_t *free_rows, int_t *x, int_t *y,
                        scalar_t *v) {
    const scalar_t LARGE = LARGE_V<scalar_t>;
    uint_t current = 0;
    int_t new_free = 0;
    uint_t rr_cnt = 0;
    while (current < n_free) {
        int_t i0, j1, j2;
        scalar_t v1, v2, v1_new;
        boolean v1_lowers;
        ++rr_cnt;
        int_t free_i = free_rows[current++];
        const scalar_t *row = cost_flat + static_cast<size_t>(free_i) * stride;
        j1 = 0;
        v1 = row[0] - v[0];
        j2 = -1;
        v2 = LARGE;
        for (uint_t j = 1; j < n; ++j) {
            scalar_t c = row[j] - v[j];
            if (c < v2) {
                if (c >= v1) {
                    v2 = c;
                    j2 = static_cast<int_t>(j);
                } else {
                    v2 = v1;
                    v1 = c;
                    j2 = j1;
                    j1 = static_cast<int_t>(j);
                }
            }
        }
        i0 = y[j1];
        v1_new = v[j1] - (v2 - v1);
        v1_lowers = v1_new < v[j1];
        if (rr_cnt < current * n) {
            if (v1_lowers) {
                v[j1] = v1_new;
            } else if (i0 >= 0 && j2 >= 0) {
                j1 = j2;
                i0 = y[j2];
            }
            if (i0 >= 0) {
                if (v1_lowers)
                    free_rows[--current] = i0;
                else
                    free_rows[new_free++] = i0;
            }
        } else {
            if (i0 >= 0)
                free_rows[new_free++] = i0;
        }
        x[free_i] = j1;
        y[j1] = free_i;
    }
    return new_free;
}

template <typename scalar_t>
inline int_t ca_dense(uint_t n, const scalar_t *cost_flat, uint_t stride, uint_t n_free,
                      const int_t *free_rows, int_t *x, int_t *y, scalar_t *v,
                      int_t *pred, int_t *cols, scalar_t *d) {
    for (uint_t f = 0; f < n_free; ++f) {
        int_t j = find_path_dense<scalar_t>(n, cost_flat, stride, free_rows[f], y, v,
                                            pred, cols, d);
        int_t i = -1;
        uint_t k = 0;
        do {
            i = pred[j];
            y[j] = i;
            int_t tmp = x[i];
            x[i] = j;
            j = tmp;
            ++k;
            if (k >= n)
                break;
        } while (i != free_rows[f]);
    }
    return 0;
}

template <typename scalar_t>
inline int jonker_internal_flat(uint_t n, const scalar_t *cost_flat, uint_t stride,
                                int_t *x, int_t *y, boolean *unique, int_t *free_rows,
                                scalar_t *v, int_t *pred, int_t *cols, scalar_t *d) {
    int_t ret = ccrrt_dense<scalar_t>(n, cost_flat, stride, free_rows, x, y, v, unique);
    int i = 0;
    while (ret > 0 && i < 2) {
        ret = carr_dense<scalar_t>(n, cost_flat, stride, ret, free_rows, x, y, v);
        ++i;
    }
    if (ret > 0)
        ca_dense<scalar_t>(n, cost_flat, stride, ret, free_rows, x, y, v, pred, cols,
                           d);
    return 0;
}

} // namespace match_jonker::detail::flat
