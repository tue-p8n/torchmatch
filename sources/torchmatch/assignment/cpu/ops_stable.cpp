// Stable-ABI variant of ops.cpp — compiled with Py_LIMITED_API + TORCH_TARGET_VERSION.
// The logic is identical to ops.cpp; only the PyTorch API surface differs:
//   at::Tensor  → torch::stable::Tensor   (torch/csrc/stable/tensor.h)
//   at::full / parallel_for → torch::stable::*  (torch/csrc/stable/ops.h)
//   TORCH_CHECK → STD_TORCH_CHECK          (torch/headeronly/util/Exception.h)
//   TORCH_LIBRARY_FRAGMENT / _IMPL → STABLE_TORCH_LIBRARY_FRAGMENT / _IMPL
//   m.impl(&fn) → m.impl(TORCH_BOX(&fn))

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>
#include <vector>

#include "jonker_compact_core.h"
#include "jonker_dense_core.h"
#include "jonker_scalar.h"

namespace ts = torch::stable;
using ScT = torch::headeronly::ScalarType;
using Tensor = ts::Tensor;

namespace {

using SolverFn64 = int (*)(const double *, int, int64_t *, int64_t *);
using SolverFn32 = int (*)(const float *, int, int64_t *, int64_t *);

inline int jonker_dense_dispatch64(const double *c, int n, int64_t *r, int64_t *col) {
    return match_jonker::detail::jonker_dense_solve_square(c, n, r, col);
}
inline int jonker_compact_dispatch64(const double *c, int n, int64_t *r, int64_t *col) {
    return match_jonker::detail::jonker_compact_solve_square(c, n, r, col);
}
inline int jonker_dense_dispatch32(const float *c, int n, int64_t *r, int64_t *col) {
    return match_jonker::detail::jonker_dense_solve_square_f32(c, n, r, col);
}
inline int jonker_compact_dispatch32(const float *c, int n, int64_t *r, int64_t *col) {
    return match_jonker::detail::jonker_compact_solve_square_f32(c, n, r, col);
}

template <typename scalar_t>
static inline scalar_t compute_sentinel(const scalar_t *data, int64_t count, int K) {
    scalar_t max_finite = std::numeric_limits<scalar_t>::lowest();
    bool any = false;
    bool has_nan = false;
    bool has_neg_inf = false;
    for (int64_t i = 0; i < count; i++) {
        const scalar_t v = data[i];
        if (std::isnan(v)) {
            has_nan = true;
        } else if (v == -std::numeric_limits<scalar_t>::infinity()) {
            has_neg_inf = true;
        } else if (std::isfinite(v)) {
            if (v > max_finite)
                max_finite = v;
            any = true;
        }
    }
    STD_TORCH_CHECK(!has_nan,
                "jonker: cost matrix contains NaN. Use +inf for forbidden pairs; "
                "NaN signals an upstream bug.");
    STD_TORCH_CHECK(!has_neg_inf,
                "jonker: cost matrix contains -inf. Use +inf for forbidden pairs; "
                "-inf has no forbidden-edge meaning.");
    return any ? (max_finite + static_cast<scalar_t>(1)) * static_cast<scalar_t>(K + 1)
               : static_cast<scalar_t>(1);
}

Tensor solve_single(const Tensor &cost, SolverFn64 solver64, SolverFn32 solver32) {
    STD_TORCH_CHECK(cost.is_cpu(), "cost must be a CPU tensor");
    STD_TORCH_CHECK(cost.dim() == 2, "cost must be 2D, got ", cost.dim());
    STD_TORCH_CHECK(cost.scalar_type() == ScT::Double || cost.scalar_type() == ScT::Float,
                "cost must be float32 or float64");

    const int n_rows = static_cast<int>(cost.size(0));
    const int n_cols = static_cast<int>(cost.size(1));

    if (n_rows == 0 || n_cols == 0)
        return ts::full({(int64_t)n_rows}, -1.0, ScT::Long);

    const int K = std::max(n_rows, n_cols);
    const bool is_f64 = cost.scalar_type() == ScT::Double;

    static thread_local std::vector<int64_t> tl_rtc, tl_ctr;
    tl_rtc.resize(K);
    tl_ctr.resize(K);

    int rc = -1;
    auto out = ts::full({(int64_t)n_rows}, -1.0, ScT::Long);
    int64_t *out_ptr = out.mutable_data_ptr<int64_t>();

    if (is_f64) {
        static thread_local std::vector<double> tl_padded;
        tl_padded.resize(static_cast<size_t>(K) * K);
        const double *src = cost.const_data_ptr<double>();
        const double sentinel = compute_sentinel(src, (int64_t)n_rows * n_cols, K);
        if (n_rows == K && n_cols == K) {
            for (int64_t k = 0; k < (int64_t)K * K; k++) {
                const double v = src[k];
                tl_padded[k] = std::isfinite(v) ? v : sentinel;
            }
        } else {
            for (int i = 0; i < K; i++)
                for (int j = 0; j < K; j++) {
                    if (i < n_rows && j < n_cols) {
                        const double v = src[i * n_cols + j];
                        tl_padded[i * K + j] = std::isfinite(v) ? v : sentinel;
                    } else {
                        tl_padded[i * K + j] = 0.0;
                    }
                }
        }
        rc = solver64(tl_padded.data(), K, tl_rtc.data(), tl_ctr.data());
    } else {
        static thread_local std::vector<float> tl_padded_f32;
        tl_padded_f32.resize(static_cast<size_t>(K) * K);
        const float *src = cost.const_data_ptr<float>();
        const float sentinel = compute_sentinel(src, (int64_t)n_rows * n_cols, K);
        if (n_rows == K && n_cols == K) {
            for (int64_t k = 0; k < (int64_t)K * K; k++) {
                const float v = src[k];
                tl_padded_f32[k] = std::isfinite(v) ? v : sentinel;
            }
        } else {
            for (int i = 0; i < K; i++)
                for (int j = 0; j < K; j++) {
                    if (i < n_rows && j < n_cols) {
                        const float v = src[i * n_cols + j];
                        tl_padded_f32[i * K + j] = std::isfinite(v) ? v : sentinel;
                    } else {
                        tl_padded_f32[i * K + j] = 0.0f;
                    }
                }
        }
        rc = solver32(tl_padded_f32.data(), K, tl_rtc.data(), tl_ctr.data());
    }

    STD_TORCH_CHECK(rc == 0, "jonker solver failed with code ", rc);
    for (int i = 0; i < n_rows; i++) {
        const int64_t j = tl_rtc[i];
        out_ptr[i] = (j >= 0 && j < n_cols) ? j : -1;
    }
    return out;
}

Tensor solve_batch(const Tensor &costs, SolverFn64 solver64, SolverFn32 solver32) {
    STD_TORCH_CHECK(costs.is_cpu(), "costs must be a CPU tensor");
    STD_TORCH_CHECK(costs.dim() == 3, "costs must be 3D (B, N, M), got ", costs.dim());
    STD_TORCH_CHECK(costs.scalar_type() == ScT::Double || costs.scalar_type() == ScT::Float,
                "costs must be float32 or float64");

    const int64_t B = costs.size(0);
    const int n_rows = static_cast<int>(costs.size(1));
    const int n_cols = static_cast<int>(costs.size(2));

    if (B == 0 || n_rows == 0 || n_cols == 0)
        return ts::full({B, (int64_t)n_rows}, -1.0, ScT::Long);

    const int K = std::max(n_rows, n_cols);
    const bool is_f64 = costs.scalar_type() == ScT::Double;
    const auto costs_c = ts::contiguous(costs);

    auto out = ts::full({B, (int64_t)n_rows}, -1.0, ScT::Long);
    int64_t *out_ptr = out.mutable_data_ptr<int64_t>();

    if (is_f64) {
        const double *src = costs_c.const_data_ptr<double>();
        const double sentinel = compute_sentinel(src, B * (int64_t)n_rows * n_cols, K);

        ts::parallel_for(0, B, /*grain_size=*/1, [&](int64_t b0, int64_t b1) {
            static thread_local std::vector<double> tl_padded;
            static thread_local std::vector<int64_t> tl_rtc, tl_ctr;
            tl_padded.resize(static_cast<size_t>(K) * K);
            tl_rtc.resize(K);
            tl_ctr.resize(K);

            const int64_t stride_b = (int64_t)n_rows * n_cols;
            for (int64_t b = b0; b < b1; b++) {
                const double *problem = src + b * stride_b;
                if (n_rows == K && n_cols == K) {
                    for (int64_t k = 0; k < (int64_t)K * K; k++) {
                        const double v = problem[k];
                        tl_padded[k] = std::isfinite(v) ? v : sentinel;
                    }
                } else {
                    for (int i = 0; i < K; i++)
                        for (int j = 0; j < K; j++) {
                            if (i < n_rows && j < n_cols) {
                                const double v = problem[i * n_cols + j];
                                tl_padded[i * K + j] = std::isfinite(v) ? v : sentinel;
                            } else {
                                tl_padded[i * K + j] = 0.0;
                            }
                        }
                }
                const int rc =
                    solver64(tl_padded.data(), K, tl_rtc.data(), tl_ctr.data());
                int64_t *out_b = out_ptr + b * n_rows;
                if (rc != 0) {
                    for (int i = 0; i < n_rows; i++)
                        out_b[i] = -1;
                    continue;
                }
                for (int i = 0; i < n_rows; i++) {
                    const int64_t j = tl_rtc[i];
                    out_b[i] = (j >= 0 && j < n_cols) ? j : -1;
                }
            }
        });
    } else {
        const float *src = costs_c.const_data_ptr<float>();
        const float sentinel = compute_sentinel(src, B * (int64_t)n_rows * n_cols, K);

        ts::parallel_for(0, B, /*grain_size=*/1, [&](int64_t b0, int64_t b1) {
            static thread_local std::vector<float> tl_padded;
            static thread_local std::vector<int64_t> tl_rtc, tl_ctr;
            tl_padded.resize(static_cast<size_t>(K) * K);
            tl_rtc.resize(K);
            tl_ctr.resize(K);

            const int64_t stride_b = (int64_t)n_rows * n_cols;
            for (int64_t b = b0; b < b1; b++) {
                const float *problem = src + b * stride_b;
                if (n_rows == K && n_cols == K) {
                    for (int64_t k = 0; k < (int64_t)K * K; k++) {
                        const float v = problem[k];
                        tl_padded[k] = std::isfinite(v) ? v : sentinel;
                    }
                } else {
                    for (int i = 0; i < K; i++)
                        for (int j = 0; j < K; j++) {
                            if (i < n_rows && j < n_cols) {
                                const float v = problem[i * n_cols + j];
                                tl_padded[i * K + j] = std::isfinite(v) ? v : sentinel;
                            } else {
                                tl_padded[i * K + j] = 0.0f;
                            }
                        }
                }
                const int rc =
                    solver32(tl_padded.data(), K, tl_rtc.data(), tl_ctr.data());
                int64_t *out_b = out_ptr + b * n_rows;
                if (rc != 0) {
                    for (int i = 0; i < n_rows; i++)
                        out_b[i] = -1;
                    continue;
                }
                for (int i = 0; i < n_rows; i++) {
                    const int64_t j = tl_rtc[i];
                    out_b[i] = (j >= 0 && j < n_cols) ? j : -1;
                }
            }
        });
    }

    return out;
}

std::tuple<Tensor, Tensor, Tensor, Tensor>
solve_batch_unpacked(const Tensor &costs, SolverFn64 solver64, SolverFn32 solver32) {
    STD_TORCH_CHECK(costs.is_cpu(), "costs must be a CPU tensor");
    STD_TORCH_CHECK(costs.dim() == 3, "costs must be 3D (B, N, M), got ", costs.dim());
    STD_TORCH_CHECK(costs.scalar_type() == ScT::Double || costs.scalar_type() == ScT::Float,
                "costs must be float32 or float64");

    const int64_t B = costs.size(0);
    const int n_rows = static_cast<int>(costs.size(1));
    const int n_cols = static_cast<int>(costs.size(2));

    if (B == 0 || n_rows == 0 || n_cols == 0) {
        auto empty_matches = ts::empty({B, (int64_t)n_rows, (int64_t)2}, ScT::Long);
        auto empty_ur     = ts::empty({B, (int64_t)n_rows}, ScT::Long);
        auto empty_uc     = ts::empty({B, (int64_t)n_cols}, ScT::Long);
        auto zero_nm      = ts::full({B}, 0.0, ScT::Long);
        return {empty_matches, empty_ur, empty_uc, zero_nm};
    }

    const int K = std::max(n_rows, n_cols);
    const bool is_f64 = costs.scalar_type() == ScT::Double;
    const auto costs_c = ts::contiguous(costs);

    auto raw = ts::full({B, (int64_t)K}, -1.0, ScT::Long);
    int64_t *raw_ptr = raw.mutable_data_ptr<int64_t>();

    auto fill_raw = [&](auto solver, auto *src, [[maybe_unused]] auto sentinel_val) {
        // Strip const: const_data_ptr returns `const T*`; we need the mutable
        // T for std::vector<T> and for computing the sentinel.
        using scalar_t = std::remove_const_t<std::remove_pointer_t<decltype(src)>>;
        const scalar_t sentinel =
            compute_sentinel(src, B * (int64_t)n_rows * n_cols, K);

        ts::parallel_for(0, B, /*grain_size=*/1, [&](int64_t b0, int64_t b1) {
            static thread_local std::vector<scalar_t> tl_padded;
            static thread_local std::vector<int64_t> tl_rtc, tl_ctr;
            tl_padded.resize(static_cast<size_t>(K) * K);
            tl_rtc.resize(K);
            tl_ctr.resize(K);

            const int64_t stride_b = (int64_t)n_rows * n_cols;
            for (int64_t b = b0; b < b1; b++) {
                const scalar_t *problem = src + b * stride_b;
                if (n_rows == K && n_cols == K) {
                    for (int64_t k = 0; k < (int64_t)K * K; k++) {
                        const scalar_t v = problem[k];
                        tl_padded[k] = std::isfinite(v) ? v : sentinel;
                    }
                } else {
                    for (int i = 0; i < K; i++)
                        for (int j = 0; j < K; j++) {
                            if (i < n_rows && j < n_cols) {
                                const scalar_t v = problem[i * n_cols + j];
                                tl_padded[i * K + j] = std::isfinite(v) ? v : sentinel;
                            } else {
                                tl_padded[i * K + j] = static_cast<scalar_t>(0);
                            }
                        }
                }
                const int rc =
                    solver(tl_padded.data(), K, tl_rtc.data(), tl_ctr.data());
                int64_t *raw_b = raw_ptr + b * K;
                if (rc != 0) {
                    for (int i = 0; i < n_rows; i++)
                        raw_b[i] = -1;
                    continue;
                }
                for (int i = 0; i < n_rows; i++) {
                    const int64_t j = tl_rtc[i];
                    raw_b[i] = (j >= 0 && j < n_cols) ? j : -1;
                }
            }
        });
    };

    if (is_f64)
        fill_raw(solver64, costs_c.const_data_ptr<double>(), 0.0);
    else
        fill_raw(solver32, costs_c.const_data_ptr<float>(), 0.0f);

    auto matches_t = ts::full({B, (int64_t)n_rows, (int64_t)2}, -1.0, ScT::Long);
    auto ur_t      = ts::full({B, (int64_t)n_rows}, -1.0, ScT::Long);
    auto uc_t      = ts::full({B, (int64_t)n_cols}, -1.0, ScT::Long);
    auto nm_t      = ts::full({B}, 0.0, ScT::Long);

    int64_t *m_ptr   = matches_t.mutable_data_ptr<int64_t>();
    int64_t *ur_ptr2 = ur_t.mutable_data_ptr<int64_t>();
    int64_t *uc_ptr  = uc_t.mutable_data_ptr<int64_t>();
    int64_t *nm_ptr  = nm_t.mutable_data_ptr<int64_t>();

    for (int64_t b = 0; b < B; ++b) {
        const int64_t *rtc = raw_ptr + b * K;
        int64_t *mb  = m_ptr  + b * n_rows * 2;
        int64_t *urb = ur_ptr2 + b * n_rows;
        int64_t *ucb = uc_ptr  + b * n_cols;

        std::vector<bool> col_used(n_cols, false);

        int64_t nm = 0, nur = 0;
        for (int i = 0; i < n_rows; ++i) {
            int64_t j = rtc[i];
            if (j >= 0 && j < n_cols) {
                mb[nm * 2]     = i;
                mb[nm * 2 + 1] = j;
                ++nm;
                col_used[j] = true;
            } else {
                urb[nur++] = i;
            }
        }
        nm_ptr[b] = nm;
        int64_t nuc = 0;
        for (int j = 0; j < n_cols; ++j)
            if (!col_used[j])
                ucb[nuc++] = j;
    }

    return {matches_t, ur_t, uc_t, nm_t};
}

Tensor jonker_dense_cpu(const Tensor &cost) {
    return solve_single(cost, &jonker_dense_dispatch64, &jonker_dense_dispatch32);
}

Tensor jonker_compact_cpu(const Tensor &cost) {
    return solve_single(cost, &jonker_compact_dispatch64, &jonker_compact_dispatch32);
}

Tensor jonker_dense_batch_cpu(const Tensor &costs) {
    return solve_batch(costs, &jonker_dense_dispatch64, &jonker_dense_dispatch32);
}

Tensor jonker_compact_batch_cpu(const Tensor &costs) {
    return solve_batch(costs, &jonker_compact_dispatch64, &jonker_compact_dispatch32);
}

std::tuple<Tensor, Tensor, Tensor, Tensor>
jonker_dense_batch_unpacked_cpu(const Tensor &costs) {
    return solve_batch_unpacked(costs, &jonker_dense_dispatch64,
                                &jonker_dense_dispatch32);
}

std::tuple<Tensor, Tensor, Tensor, Tensor>
jonker_compact_batch_unpacked_cpu(const Tensor &costs) {
    return solve_batch_unpacked(costs, &jonker_compact_dispatch64,
                                &jonker_compact_dispatch32);
}

Tensor jonker_scalar_cpu(const Tensor &cost) {
    STD_TORCH_CHECK(cost.is_cpu(), "cost must be a CPU tensor");
    STD_TORCH_CHECK(cost.dim() == 2, "cost must be 2D, got ", cost.dim());
    STD_TORCH_CHECK(cost.scalar_type() == ScT::Double || cost.scalar_type() == ScT::Float,
                "cost must be float32 or float64");

    const intptr_t nr = cost.size(0);
    const intptr_t nc = cost.size(1);
    auto out = ts::full({(int64_t)nr}, -1.0, ScT::Long);
    if (nr == 0 || nc == 0) {
        return out;
    }

    auto cost_f64 = ts::contiguous(ts::to(cost, ScT::Double));
    const intptr_t k = std::min(nr, nc);
    std::vector<int64_t> a(k), b(k);

    const int rc = solve_rectangular_linear_sum_assignment(
        nr, nc, cost_f64.mutable_data_ptr<double>(), /*maximize=*/false, a.data(), b.data());

    STD_TORCH_CHECK(rc != RECTANGULAR_LSAP_INVALID,
                "jonker_scalar: cost contains NaN or -inf");
    STD_TORCH_CHECK(rc != RECTANGULAR_LSAP_INFEASIBLE,
                "jonker_scalar: no feasible assignment (all rows blocked)");
    STD_TORCH_CHECK(rc == 0, "jonker_scalar: solver returned ", rc);

    int64_t *out_ptr = out.mutable_data_ptr<int64_t>();
    for (intptr_t i = 0; i < k; i++) {
        out_ptr[a[i]] = b[i];
    }
    return out;
}

} // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(assignment, m) {
    m.def("jonker_scalar(Tensor cost) -> Tensor");
    m.def("jonker_dense(Tensor cost) -> Tensor");
    m.def("jonker_compact(Tensor cost) -> Tensor");
    m.def("jonker_dense_batch(Tensor costs) -> Tensor");
    m.def("jonker_compact_batch(Tensor costs) -> Tensor");
    m.def("jonker_dense_batch_unpacked(Tensor costs) -> "
          "(Tensor, Tensor, Tensor, Tensor)");
    m.def("jonker_compact_batch_unpacked(Tensor costs) -> "
          "(Tensor, Tensor, Tensor, Tensor)");
}

STABLE_TORCH_LIBRARY_IMPL(assignment, CPU, m) {
    m.impl("jonker_scalar",                 TORCH_BOX(&jonker_scalar_cpu));
    m.impl("jonker_dense",                  TORCH_BOX(&jonker_dense_cpu));
    m.impl("jonker_compact",                TORCH_BOX(&jonker_compact_cpu));
    m.impl("jonker_dense_batch",            TORCH_BOX(&jonker_dense_batch_cpu));
    m.impl("jonker_compact_batch",          TORCH_BOX(&jonker_compact_batch_cpu));
    m.impl("jonker_dense_batch_unpacked",   TORCH_BOX(&jonker_dense_batch_unpacked_cpu));
    m.impl("jonker_compact_batch_unpacked", TORCH_BOX(&jonker_compact_batch_unpacked_cpu));
}
