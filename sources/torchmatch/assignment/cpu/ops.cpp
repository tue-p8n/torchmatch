// PyTorch op wrappers around the CPU Jonker-Volgenant solvers.
//
// Ops registered under the `assignment` library (shared with the CUDA
// primed-zeros Hungarian ops munkres, hybrid, lawler):
//   jonker_dense(cost) -> Tensor                 (rectangular allowed)
//   jonker_compact(cost) -> Tensor               (rectangular allowed)
//   jonker_dense_batch(costs) -> Tensor          (B, N, M) -> (B, N)  [CPU backend]
//   jonker_compact_batch(costs) -> Tensor
//   jonker_dense_batch_unpacked(costs) -> (matches, ur, uc, n_matched)
//   jonker_compact_batch_unpacked(costs)
//
// `dense` is the rectangular-capable Jonker-Volgenant (Kazmar-style
// flat). `compact` is the square-only Jonker-Volgenant
// (Markovtsev-style AVX2-tight inner loop). Both pad rectangular
// inputs internally; they differ in the inner-iteration strategy and
// in the SIMD pattern each one unlocks.
//
// All ops accept float32 or float64 inputs and return int64 row-to-col
// mappings. Entries pointing into padded columns normalize to -1.
//
// Inf values rewrite to a per-call sentinel (max_finite + 1) * (K + 1)
// inside C++, so the caller does not need to sanitize the tensor first.
// NaN inputs are rejected at the entry point: NaN is never a valid
// forbidden-pair sentinel (use +inf for that), so a NaN cell signals
// an upstream bug.

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "jonker_compact_core.h"
#include "jonker_dense_core.h"
#include "jonker_scalar.h"

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

// One pass over the input rejects NaN, rejects -inf, and computes the
// +inf sentinel (max_finite + 1) * (K + 1), avoiding a second sweep
// before the solve. Returns 1.0 in the all-inf case. NaN signals an
// upstream bug (zero-norm cosine, singular Kalman covariance, log of
// zero); -inf creates an unboundedly preferable edge and would break
// the solver's dual-variable invariants. Only +inf carries the
// forbidden-edge meaning.
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
    TORCH_CHECK(!has_nan,
                "jonker: cost matrix contains NaN. Use +inf for forbidden pairs; "
                "NaN signals an upstream bug.");
    TORCH_CHECK(!has_neg_inf,
                "jonker: cost matrix contains -inf. Use +inf for forbidden pairs; "
                "-inf has no forbidden-edge meaning.");
    return any ? (max_finite + static_cast<scalar_t>(1)) * static_cast<scalar_t>(K + 1)
               : static_cast<scalar_t>(1);
}

// `cost` must be CPU, float32 or float64, contiguous.
at::Tensor solve_single(const at::Tensor &cost, SolverFn64 solver64,
                        SolverFn32 solver32) {
    TORCH_CHECK(cost.is_cpu(), "cost must be a CPU tensor");
    TORCH_CHECK(cost.dim() == 2, "cost must be 2D, got ", cost.dim());
    TORCH_CHECK(cost.scalar_type() == at::kDouble || cost.scalar_type() == at::kFloat,
                "cost must be float32 or float64, got ", cost.scalar_type());

    const int n_rows = static_cast<int>(cost.size(0));
    const int n_cols = static_cast<int>(cost.size(1));
    const auto out_opts = at::TensorOptions().dtype(at::kLong);

    if (n_rows == 0 || n_cols == 0)
        return at::full({n_rows}, -1, out_opts);

    const int K = std::max(n_rows, n_cols);
    const bool is_f64 = cost.scalar_type() == at::kDouble;

    static thread_local std::vector<int64_t> tl_rtc, tl_ctr;
    tl_rtc.resize(K);
    tl_ctr.resize(K);

    int rc = -1;
    auto out = at::full({n_rows}, -1, out_opts);
    int64_t *out_ptr = out.data_ptr<int64_t>();

    if (is_f64) {
        static thread_local std::vector<double> tl_padded;
        tl_padded.resize(static_cast<size_t>(K) * K);
        const double *src = cost.data_ptr<double>();
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
        const float *src = cost.data_ptr<float>();
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

    TORCH_CHECK(rc == 0, "jonker solver failed with code ", rc);
    for (int i = 0; i < n_rows; i++) {
        const int64_t j = tl_rtc[i];
        out_ptr[i] = (j >= 0 && j < n_cols) ? j : -1;
    }
    return out;
}

// `costs` must be (B, N, M) CPU float32 or float64 contiguous.
at::Tensor solve_batch(const at::Tensor &costs, SolverFn64 solver64,
                       SolverFn32 solver32) {
    TORCH_CHECK(costs.is_cpu(), "costs must be a CPU tensor");
    TORCH_CHECK(costs.dim() == 3, "costs must be 3D (B, N, M), got ", costs.dim());
    TORCH_CHECK(costs.scalar_type() == at::kDouble || costs.scalar_type() == at::kFloat,
                "costs must be float32 or float64, got ", costs.scalar_type());

    const int64_t B = costs.size(0);
    const int n_rows = static_cast<int>(costs.size(1));
    const int n_cols = static_cast<int>(costs.size(2));
    const auto out_opts = at::TensorOptions().dtype(at::kLong);

    if (B == 0 || n_rows == 0 || n_cols == 0)
        return at::full({B, n_rows}, -1, out_opts);

    const int K = std::max(n_rows, n_cols);
    const bool is_f64 = costs.scalar_type() == at::kDouble;
    const auto costs_c = costs.contiguous();

    auto out = at::full({B, n_rows}, -1, out_opts);
    int64_t *out_ptr = out.data_ptr<int64_t>();

    if (is_f64) {
        const double *src = costs_c.data_ptr<double>();
        const double sentinel = compute_sentinel(src, B * (int64_t)n_rows * n_cols, K);

        at::parallel_for(0, B, /*grain_size=*/1, [&](int64_t b0, int64_t b1) {
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
        const float *src = costs_c.data_ptr<float>();
        const float sentinel = compute_sentinel(src, B * (int64_t)n_rows * n_cols, K);

        at::parallel_for(0, B, /*grain_size=*/1, [&](int64_t b0, int64_t b1) {
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

// Returning pre-unpacked tensors avoids a per-problem Python loop in
// the caller. Tensor layout:
//   matches:         (B, max_n, 2) int64; valid rows are [0, n_matches[b])
//   unmatched_rows:  (B, max_n)    int64; valid rows are [0, n_rows-n_matches[b])
//   unmatched_cols:  (B, max_m)    int64; valid rows are [0, n_cols-n_matches[b])
//   n_matches:       (B,)          int64
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
solve_batch_unpacked(const at::Tensor &costs, SolverFn64 solver64,
                     SolverFn32 solver32) {
    TORCH_CHECK(costs.is_cpu(), "costs must be a CPU tensor");
    TORCH_CHECK(costs.dim() == 3, "costs must be 3D (B, N, M), got ", costs.dim());
    TORCH_CHECK(costs.scalar_type() == at::kDouble || costs.scalar_type() == at::kFloat,
                "costs must be float32 or float64, got ", costs.scalar_type());

    const int64_t B = costs.size(0);
    const int n_rows = static_cast<int>(costs.size(1));
    const int n_cols = static_cast<int>(costs.size(2));
    const auto long_opts = at::TensorOptions().dtype(at::kLong);

    if (B == 0 || n_rows == 0 || n_cols == 0) {
        auto empty_matches = at::empty({B, n_rows, 2}, long_opts);
        auto empty_ur = at::empty({B, n_rows}, long_opts);
        auto empty_uc = at::empty({B, n_cols}, long_opts);
        auto zero_nm = at::zeros({B}, long_opts);
        return {empty_matches, empty_ur, empty_uc, zero_nm};
    }

    const int K = std::max(n_rows, n_cols);
    const bool is_f64 = costs.scalar_type() == at::kDouble;
    const auto costs_c = costs.contiguous();

    // Same path as solve_batch, but write raw rtc into a (B, K) tensor.
    auto raw = at::full({B, K}, -1L, long_opts);
    int64_t *raw_ptr = raw.data_ptr<int64_t>();

    auto fill_raw = [&](auto solver, auto *src, auto sentinel_val) {
        using scalar_t = std::remove_pointer_t<decltype(src)>;
        const scalar_t sentinel =
            compute_sentinel(src, B * (int64_t)n_rows * n_cols, K);

        at::parallel_for(0, B, /*grain_size=*/1, [&](int64_t b0, int64_t b1) {
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
        fill_raw(solver64, costs_c.data_ptr<double>(), 0.0);
    else
        fill_raw(solver32, costs_c.data_ptr<float>(), 0.0f);

    // Unpack raw (B, K) into (matches, unmatched_rows, unmatched_cols, n_matches).
    auto matches_t = at::full({B, n_rows, 2}, -1L, long_opts);
    auto ur_t = at::full({B, n_rows}, -1L, long_opts);
    auto uc_t = at::full({B, n_cols}, -1L, long_opts);
    auto nm_t = at::zeros({B}, long_opts);

    int64_t *m_ptr = matches_t.data_ptr<int64_t>();
    int64_t *ur_ptr2 = ur_t.data_ptr<int64_t>();
    int64_t *uc_ptr = uc_t.data_ptr<int64_t>();
    int64_t *nm_ptr = nm_t.data_ptr<int64_t>();

    for (int64_t b = 0; b < B; ++b) {
        const int64_t *rtc = raw_ptr + b * K;
        int64_t *mb = m_ptr + b * n_rows * 2;
        int64_t *urb = ur_ptr2 + b * n_rows;
        int64_t *ucb = uc_ptr + b * n_cols;

        std::vector<bool> col_used(n_cols, false);

        int64_t nm = 0, nur = 0;
        for (int i = 0; i < n_rows; ++i) {
            int64_t j = rtc[i];
            if (j >= 0 && j < n_cols) {
                mb[nm * 2] = i;
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

at::Tensor jonker_dense_cpu(const at::Tensor &cost) {
    return solve_single(cost, &jonker_dense_dispatch64, &jonker_dense_dispatch32);
}

at::Tensor jonker_compact_cpu(const at::Tensor &cost) {
    return solve_single(cost, &jonker_compact_dispatch64, &jonker_compact_dispatch32);
}

at::Tensor jonker_dense_batch_cpu(const at::Tensor &costs) {
    return solve_batch(costs, &jonker_dense_dispatch64, &jonker_dense_dispatch32);
}

at::Tensor jonker_compact_batch_cpu(const at::Tensor &costs) {
    return solve_batch(costs, &jonker_compact_dispatch64, &jonker_compact_dispatch32);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
jonker_dense_batch_unpacked_cpu(const at::Tensor &costs) {
    return solve_batch_unpacked(costs, &jonker_dense_dispatch64,
                                &jonker_dense_dispatch32);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
jonker_compact_batch_unpacked_cpu(const at::Tensor &costs) {
    return solve_batch_unpacked(costs, &jonker_compact_dispatch64,
                                &jonker_compact_dispatch32);
}

at::Tensor jonker_scalar_cpu(const at::Tensor &cost) {
    TORCH_CHECK(cost.is_cpu(), "cost must be a CPU tensor");
    TORCH_CHECK(cost.dim() == 2, "cost must be 2D, got ", cost.dim());
    TORCH_CHECK(cost.scalar_type() == at::kDouble || cost.scalar_type() == at::kFloat,
                "cost must be float32 or float64, got ", cost.scalar_type());

    const intptr_t nr = cost.size(0);
    const intptr_t nc = cost.size(1);
    auto out = at::full({nr}, -1, at::TensorOptions().dtype(at::kLong));
    if (nr == 0 || nc == 0) {
        return out;
    }

    auto cost_f64 = cost.to(at::kDouble).contiguous();
    const intptr_t k = std::min(nr, nc);
    std::vector<int64_t> a(k), b(k);

    const int rc = solve_rectangular_linear_sum_assignment(
        nr, nc, cost_f64.data_ptr<double>(), /*maximize=*/false, a.data(), b.data());

    TORCH_CHECK(rc != RECTANGULAR_LSAP_INVALID,
                "jonker_scalar: cost contains NaN or -inf");
    TORCH_CHECK(rc != RECTANGULAR_LSAP_INFEASIBLE,
                "jonker_scalar: no feasible assignment (all rows blocked)");
    TORCH_CHECK(rc == 0, "jonker_scalar: solver returned ", rc);

    int64_t *out_ptr = out.data_ptr<int64_t>();
    for (intptr_t i = 0; i < k; i++) {
        out_ptr[a[i]] = b[i];
    }
    return out;
}

} // namespace

// The `jonker_dense_batch` schema lives here (in the CPU extension) so
// that loading only the CPU extension still gives a fully usable op.
// The CUDA extension contributes an additional CUDA impl in dispatch.cu.

TORCH_LIBRARY_FRAGMENT(assignment, m) {
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

TORCH_LIBRARY_IMPL(assignment, CPU, m) {
    m.impl("jonker_scalar", &jonker_scalar_cpu);
    m.impl("jonker_dense", &jonker_dense_cpu);
    m.impl("jonker_compact", &jonker_compact_cpu);
    m.impl("jonker_dense_batch", &jonker_dense_batch_cpu);
    m.impl("jonker_compact_batch", &jonker_compact_batch_cpu);
    m.impl("jonker_dense_batch_unpacked", &jonker_dense_batch_unpacked_cpu);
    m.impl("jonker_compact_batch_unpacked", &jonker_compact_batch_unpacked_cpu);
}
