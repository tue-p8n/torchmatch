// JV (square) solver: header-only template implementation.
//
// Includes an AVX2 specialization hook in jonker_compact_avx2.h; the scalar path
// is the default fallback.

#pragma once

#include <cstdint>
#include <cstdio>
#include <limits>
#include <tuple>
#include <vector>

#ifdef __AVX2__
#include "jonker_compact_avx2.h"
#endif

namespace match_jonker::detail {

template <typename idx, typename cost>
inline std::tuple<cost, cost, idx, idx>
find_umins_regular(idx dim, idx i, const cost *assign_cost, const cost *v) {
    const cost *local_cost = &assign_cost[static_cast<size_t>(i) * dim];
    cost umin = local_cost[0] - v[0];
    idx j1 = 0;
    idx j2 = -1;
    cost usubmin = std::numeric_limits<cost>::max();
    for (idx j = 1; j < dim; j++) {
        cost h = local_cost[j] - v[j];
        if (h < usubmin) {
            if (h >= umin) {
                usubmin = h;
                j2 = j;
            } else {
                usubmin = umin;
                umin = h;
                j2 = j1;
                j1 = j;
            }
        }
    }
    return std::make_tuple(umin, usubmin, j1, j2);
}

// Exact Jonker-Volgenant algorithm (scalar-only).
//   dim        in  problem size
//   assign_cost in cost matrix (row-major flat)
//   rowsol     out column assigned to row in solution / size dim
//   colsol     out row assigned to column in solution / size dim
//   v          inout dual variables / size dim (caller-allocated scratch)
template <typename idx, typename cost>
void jonker_compact_kernel(int dim, const cost *assign_cost, idx *rowsol, idx *colsol,
                           cost *v) {
    static thread_local std::vector<idx> collist_vec;
    static thread_local std::vector<idx> matches_vec;
    static thread_local std::vector<idx> pred_vec;
    static thread_local std::vector<cost> d_vec;

    if (static_cast<int>(collist_vec.size()) < dim)
        collist_vec.resize(dim);
    if (static_cast<int>(matches_vec.size()) < dim)
        matches_vec.resize(dim);
    if (static_cast<int>(pred_vec.size()) < dim)
        pred_vec.resize(dim);
    if (static_cast<int>(d_vec.size()) < dim)
        d_vec.resize(dim);

    idx *collist = collist_vec.data();
    idx *matches = matches_vec.data();
    cost *d = d_vec.data();
    idx *pred = pred_vec.data();

    for (idx i = 0; i < dim; i++) {
        matches[i] = 0;
    }

    // COLUMN REDUCTION
    for (idx j = dim - 1; j >= 0; j--) {
        cost min = assign_cost[j];
        idx imin = 0;
        for (idx i = 1; i < dim; i++) {
            const cost *local_cost = &assign_cost[static_cast<size_t>(i) * dim];
            if (local_cost[j] < min) {
                min = local_cost[j];
                imin = i;
            }
        }
        v[j] = min;

        if (++matches[imin] == 1) {
            rowsol[imin] = j;
            colsol[j] = imin;
        } else {
            colsol[j] = -1;
        }
    }

    // REDUCTION TRANSFER
    idx *free_rows = matches; // reuse storage
    idx numfree = 0;
    for (idx i = 0; i < dim; i++) {
        const cost *local_cost = &assign_cost[static_cast<size_t>(i) * dim];
        if (matches[i] == 0) {
            free_rows[numfree++] = i;
        } else if (matches[i] == 1) {
            idx j1 = rowsol[i];
            cost min = std::numeric_limits<cost>::max();
            for (idx j = 0; j < dim; j++) {
                if (j != j1) {
                    cost cand = local_cost[j] - v[j];
                    if (cand < min)
                        min = cand;
                }
            }
            v[j1] = v[j1] - min;
        }
    }

    // AUGMENTING ROW REDUCTION
    for (int loopcnt = 0; loopcnt < 2; loopcnt++) {
        idx k = 0;
        idx prevnumfree = numfree;
        numfree = 0;
        while (k < prevnumfree) {
            idx i = free_rows[k++];

            cost umin, usubmin;
            idx j1, j2;
#ifdef __AVX2__
            if constexpr (std::is_same_v<cost, double>) {
                if (__builtin_cpu_supports("avx2")) {
                    std::tie(umin, usubmin, j1, j2) =
                        match_jonker::detail::find_umins_avx2<idx>(dim, i, assign_cost,
                                                                   v);
                    goto done_umins;
                }
            }
#endif
            std::tie(umin, usubmin, j1, j2) =
                find_umins_regular<idx, cost>(dim, i, assign_cost, v);
#ifdef __AVX2__
        done_umins:;
#endif

            idx i0 = colsol[j1];
            cost vj1_new = v[j1] - (usubmin - umin);
            bool vj1_lowers = vj1_new < v[j1];
            if (vj1_lowers) {
                v[j1] = vj1_new;
            } else if (i0 >= 0) {
                j1 = j2;
                i0 = colsol[j2];
            }

            rowsol[i] = j1;
            colsol[j1] = i;

            if (i0 >= 0) {
                if (vj1_lowers) {
                    free_rows[--k] = i0;
                } else {
                    free_rows[numfree++] = i0;
                }
            }
        }
    }

    // AUGMENT SOLUTION for each free row.
    for (idx f = 0; f < numfree; f++) {
        idx endofpath = -1;
        idx freerow = free_rows[f];

        for (idx j = 0; j < dim; j++) {
            d[j] = assign_cost[static_cast<size_t>(freerow) * dim + j] - v[j];
            pred[j] = freerow;
            collist[j] = j;
        }

        idx low = 0;
        idx up = 0;
        bool unassigned_found = false;
        idx last = 0;
        cost min = 0;
        do {
            if (up == low) {
                last = low - 1;
                min = d[collist[up++]];
                for (idx k = up; k < dim; k++) {
                    idx j = collist[k];
                    cost h = d[j];
                    if (h <= min) {
                        if (h < min) {
                            up = low;
                            min = h;
                        }
                        collist[k] = collist[up];
                        collist[up++] = j;
                    }
                }
                for (idx k = low; k < up; k++) {
                    if (colsol[collist[k]] < 0) {
                        endofpath = collist[k];
                        unassigned_found = true;
                        break;
                    }
                }
            }

            if (!unassigned_found) {
                idx j1 = collist[low];
                low++;
                idx i = colsol[j1];
                const cost *local_cost = &assign_cost[static_cast<size_t>(i) * dim];
                cost h = local_cost[j1] - v[j1] - min;
                for (idx k = up; k < dim; k++) {
                    idx j = collist[k];
                    cost v2 = local_cost[j] - v[j] - h;
                    if (v2 < d[j]) {
                        pred[j] = i;
                        if (v2 == min) {
                            if (colsol[j] < 0) {
                                endofpath = j;
                                unassigned_found = true;
                                break;
                            } else {
                                collist[k] = collist[up];
                                collist[up++] = j;
                            }
                        }
                        d[j] = v2;
                    }
                }
            }
        } while (!unassigned_found);

        for (idx k = 0; k <= last; k++) {
            idx j1 = collist[k];
            v[j1] = v[j1] + d[j1] - min;
        }

        {
            idx i;
            do {
                i = pred[endofpath];
                colsol[endofpath] = i;
                idx j1 = endofpath;
                endofpath = rowsol[i];
                rowsol[i] = j1;
            } while (i != freerow);
        }
    }
}

// rowsol, colsol, and v are thread-local to avoid per-call heap allocation.
inline int jonker_compact_solve_square(const double *cost_flat, int n,
                                       int64_t *row_to_col, int64_t *col_to_row) {
    if (n <= 0)
        return 0;

    static thread_local std::vector<int32_t> tl_rowsol, tl_colsol;
    static thread_local std::vector<double> tl_v;
    if (static_cast<int>(tl_rowsol.size()) < n)
        tl_rowsol.resize(n);
    if (static_cast<int>(tl_colsol.size()) < n)
        tl_colsol.resize(n);
    if (static_cast<int>(tl_v.size()) < n)
        tl_v.resize(n);

    jonker_compact_kernel<int32_t, double>(n, cost_flat, tl_rowsol.data(),
                                           tl_colsol.data(), tl_v.data());

    for (int i = 0; i < n; i++)
        row_to_col[i] = static_cast<int64_t>(tl_rowsol[i]);
    for (int j = 0; j < n; j++)
        col_to_row[j] = static_cast<int64_t>(tl_colsol[j]);
    return 0;
}

inline int jonker_compact_solve_square_f32(const float *cost_flat, int n,
                                           int64_t *row_to_col, int64_t *col_to_row) {
    if (n <= 0)
        return 0;

    static thread_local std::vector<int32_t> tl_rowsol, tl_colsol;
    static thread_local std::vector<float> tl_v;
    if (static_cast<int>(tl_rowsol.size()) < n)
        tl_rowsol.resize(n);
    if (static_cast<int>(tl_colsol.size()) < n)
        tl_colsol.resize(n);
    if (static_cast<int>(tl_v.size()) < n)
        tl_v.resize(n);

    jonker_compact_kernel<int32_t, float>(n, cost_flat, tl_rowsol.data(),
                                          tl_colsol.data(), tl_v.data());

    for (int i = 0; i < n; i++)
        row_to_col[i] = static_cast<int64_t>(tl_rowsol[i]);
    for (int j = 0; j < n; j++)
        col_to_row[j] = static_cast<int64_t>(tl_colsol[j]);
    return 0;
}

} // namespace match_jonker::detail
