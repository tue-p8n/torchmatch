// JV (rectangular) solver: a flat pointer + stride layout enables
// direct AVX2 loads and avoids per-call thread-local row-pointer
// scratch.

#include "jonker_dense_core.h"
#include "jonker_dense_flat.h"

#include <cstdint>
#include <vector>

namespace match_jonker::detail {

using namespace match_jonker::detail::flat;

template <typename scalar_t>
static int solve_square_impl(const scalar_t *cost_flat, int n, int64_t *row_to_col,
                             int64_t *col_to_row) {
    if (n <= 0)
        return 0;

    static thread_local std::vector<int_t> tl_x, tl_y, tl_free, tl_pred, tl_cols;
    static thread_local std::vector<scalar_t> tl_v, tl_d;
    static thread_local std::vector<boolean> tl_unique;

    auto grow = [](auto &v, int sz) {
        if (static_cast<int>(v.size()) < sz)
            v.resize(sz);
    };
    grow(tl_x, n);
    grow(tl_y, n);
    grow(tl_free, n);
    grow(tl_pred, n);
    grow(tl_cols, n);
    grow(tl_d, n);
    grow(tl_v, n);
    grow(tl_unique, n);

    int ret = jonker_internal_flat<scalar_t>(
        static_cast<uint_t>(n), cost_flat, static_cast<uint_t>(n), tl_x.data(),
        tl_y.data(), tl_unique.data(), tl_free.data(), tl_v.data(), tl_pred.data(),
        tl_cols.data(), tl_d.data());

    if (ret != 0)
        return -1;
    for (int i = 0; i < n; ++i)
        row_to_col[i] = static_cast<int64_t>(tl_x[i]);
    for (int j = 0; j < n; ++j)
        col_to_row[j] = static_cast<int64_t>(tl_y[j]);
    return 0;
}

int jonker_dense_solve_square(const double *cost_flat, int n, int64_t *row_to_col,
                              int64_t *col_to_row) {
    return solve_square_impl<double>(cost_flat, n, row_to_col, col_to_row);
}

int jonker_dense_solve_square_f32(const float *cost_flat, int n, int64_t *row_to_col,
                                  int64_t *col_to_row) {
    return solve_square_impl<float>(cost_flat, n, row_to_col, col_to_row);
}

} // namespace match_jonker::detail
