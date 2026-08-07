// Munkres (1957) classical Hungarian for the LAP on GPU. Single-path
// augmentation per outer iteration over primed/starred zeros and
// row/column covers. The six-step state machine runs as a host loop
// that launches one kernel per step.
//
// Cost layout: column-major float, padded to square (K, K) where
// K = max(rows, cols). The +inf rewrite sentinel keeps any match into
// padded cells provably suboptimal, so the repack step normalizes them
// to -1.
//
// State is held by a `MunkresWorkspace` per (K, device), supplied by
// `WorkspaceCache`. The coalesced `ControlBlock` lives on one managed
// page so per-iteration host reads (n_matches, goto_5, repeat_kernel,
// min_in_mat) avoid both a per-call cudaMallocManaged and a per-step
// cudaMemcpy round-trip.

#include <cuda_runtime.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/Exception.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>

#include <cub/cub.cuh>

#include "control.cuh"
#include "cuda_common.cuh"
#include "workspace.cuh"

using uint = unsigned int;

namespace {

constexpr int kMaxThreadsPerBlock = 1024;
constexpr int kColsPerBlockStep4 = 512;
constexpr int kThreadsReduction = 256;

struct ControlBlock {
    int error_code;    // assign_lap::LapErrorCode
    int n_matches;     // step 3 atomic accumulator
    int repeat_kernel; // step 2 / step 4 retry flag (bool-as-int)
    int goto_5;        // step 4 -> step 5 transition flag (bool-as-int)
    float min_in_mat;  // step 6 dual-update minimum (CUB reduce target)
    unsigned long long zeros_size;
};

// All per-solve device state, packed into a flat handle so kernels
// can take it by value. Each kernel reads a few fields; passing
// pointers individually would balloon the launch arg list.
template <typename data = float>
struct MunkresState {
    size_t SIZE, nrows, ncols;
    uint NB4, NBR;
    uint n_rows_per_block, n_cols_per_block;
    uint log2_n, log2_data_block_size, data_block_size;
    uint n_blocks_step_4;
    int row_mask;
    uint nb4;

    data *slack;
    data *min_in_rows;
    data *min_in_cols;
    size_t *zeros;
    size_t *zeros_size_b;
    int *row_of_star_at_column;
    // column_of_star_at_row is managed memory: host reads it after the
    // solve completes, and the prefetch policy in control.cuh keeps it
    // device-resident during the solve itself.
    int *column_of_star_at_row;
    int *cover_row;
    int *cover_column;
    int *column_of_prime_at_row;
    int *row_of_green_at_column;
    data *d_min_in_mat_vect;
    ControlBlock *cb;
    void *cub_temp;
    size_t cub_temp_bytes;
};

template <typename data>
struct MunkresWorkspace {
    MunkresState<data> state;
    assign_lap::WorkspacePool pool;
    int device = -1;
    bool initialized = false;
    uint num_blocks_reduction = 0;

    void allocate(std::size_t size, int dev) {
        device = dev;
        pool.reset(dev);

        const uint num_blocks_4 =
            std::max((uint)std::ceil((size * 1.0) / kColsPerBlockStep4), 1u);
        num_blocks_reduction = std::min((uint)size, 256u);
        const uint log2_size = (uint)std::ceil(std::log2((double)size));

        state.SIZE = state.nrows = state.ncols = size;
        state.row_mask = (1 << log2_size) - 1;
        state.nb4 = state.NB4 = num_blocks_4;
        state.NBR = num_blocks_reduction;
        state.n_rows_per_block = (uint)std::ceil(size / (double)num_blocks_reduction);
        state.n_cols_per_block = state.n_rows_per_block;
        state.log2_n = log2_size;
        state.n_blocks_step_4 = num_blocks_4;
        state.data_block_size =
            kColsPerBlockStep4 * (uint)std::pow(2, std::ceil(std::log2((double)size)));
        state.log2_data_block_size =
            log2_size + (uint)std::ceil(std::log2((double)kColsPerBlockStep4));

        auto alloc = [this](std::size_t bytes) { return pool.alloc(bytes); };
        state.slack = static_cast<data *>(alloc(size * size * sizeof(data)));
        state.min_in_rows = static_cast<data *>(alloc(size * sizeof(data)));
        state.min_in_cols = static_cast<data *>(alloc(size * sizeof(data)));
        state.zeros = static_cast<size_t *>(alloc(size * size * sizeof(size_t)));
        state.zeros_size_b =
            static_cast<size_t *>(alloc(num_blocks_4 * sizeof(size_t)));
        state.row_of_star_at_column = static_cast<int *>(alloc(size * sizeof(int)));
        CUDA_RUNTIME(
            cudaMallocManaged(&state.column_of_star_at_row, size * sizeof(int)));
        state.cover_row = static_cast<int *>(alloc(size * sizeof(int)));
        state.cover_column = static_cast<int *>(alloc(size * sizeof(int)));
        state.column_of_prime_at_row = static_cast<int *>(alloc(size * sizeof(int)));
        state.row_of_green_at_column = static_cast<int *>(alloc(size * sizeof(int)));
        state.d_min_in_mat_vect =
            static_cast<data *>(alloc(num_blocks_reduction * sizeof(data)));

        state.cb = assign_lap::alloc_cb<ControlBlock>(dev);
        if (!state.cb) {
            throw std::runtime_error("lap_munkres: failed to allocate ControlBlock");
        }

        // CUB scratch sized for the larger of the two reductions we
        // issue: a min reduce of `d_min_in_mat_vect` (step 6) and a sum
        // reduce of `zeros_size_b` (post-compress, post-step_6).
        size_t bytes_min = 0, bytes_sum = 0;
        cub::DeviceReduce::Reduce(nullptr, bytes_min, state.d_min_in_mat_vect,
                                  &state.cb->min_in_mat, (int)num_blocks_reduction,
                                  assign_lap::Min(), assign_lap::cost_inf<data>());
        cub::DeviceReduce::Sum(nullptr, bytes_sum, state.zeros_size_b,
                               &state.cb->zeros_size, (int)num_blocks_4);
        state.cub_temp_bytes = std::max(bytes_min, bytes_sum);
        state.cub_temp = alloc(state.cub_temp_bytes);

        initialized = true;
    }

    void deallocate() {
        if (!initialized)
            return;
        // column_of_star_at_row is managed memory, which cannot come from
        // the caching allocator, so it is freed on its own.
        CUDA_RUNTIME(cudaFree(state.column_of_star_at_row));
        assign_lap::free_cb(state.cb);
        // Everything else is pooled: one release frees all of it.
        pool.release();
        initialized = false;
    }
};

static assign_lap::WorkspaceCache<MunkresWorkspace<float>> g_munkres_cache;

template <typename data = float>
__global__ void k_init(MunkresState<data> s) {
    size_t i = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i < s.SIZE) {
        s.cover_row[i] = 0;
        s.column_of_star_at_row[i] = -1;
        s.cover_column[i] = 0;
        s.row_of_star_at_column[i] = -1;
    }
}

template <typename data = float>
__global__ void k_calc_col_min(MunkresState<data> s) {
    size_t i = (size_t)threadIdx.x * s.SIZE + (size_t)blockIdx.x;
    data thread_min = assign_lap::cost_inf<data>();
    while (i < s.SIZE * s.SIZE) {
        thread_min = min(thread_min, s.slack[i]);
        i += (size_t)blockDim.x * s.SIZE;
    }
    __syncthreads();
    using BR = cub::BlockReduce<data, kThreadsReduction>;
    __shared__ typename BR::TempStorage temp_storage;
    thread_min = BR(temp_storage).Reduce(thread_min, assign_lap::Min());
    if (threadIdx.x == 0)
        s.min_in_rows[blockIdx.x] = thread_min;
}

template <typename data = float>
__global__ void k_col_sub(MunkresState<data> s) {
    size_t i = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i < s.SIZE * s.SIZE)
        s.slack[i] -= s.min_in_rows[i % s.SIZE];
}

template <typename data = float>
__global__ void k_calc_row_min(MunkresState<data> s) {
    data thread_min = assign_lap::cost_inf<data>();
    size_t i = (size_t)blockIdx.x * s.SIZE + threadIdx.x;
    while (i < s.SIZE * ((size_t)blockIdx.x + 1)) {
        thread_min = min(thread_min, s.slack[i]);
        i += blockDim.x;
    }
    using BR = cub::BlockReduce<data, kThreadsReduction>;
    __shared__ typename BR::TempStorage temp_storage;
    thread_min = BR(temp_storage).Reduce(thread_min, assign_lap::Min());
    if (threadIdx.x == 0)
        s.min_in_cols[blockIdx.x] = thread_min;
}

template <typename data = float>
__global__ void k_row_sub(MunkresState<data> s) {
    size_t i = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i < s.SIZE * s.SIZE)
        s.slack[i] -= s.min_in_cols[i / s.SIZE];
    // Fused init of `zeros_size_b` saves a memset launch before the
    // compress pass; the same threads zero one bucket each.
    if (i < s.n_blocks_step_4)
        s.zeros_size_b[i] = 0;
}

template <typename data = float>
__global__ void k_compress_matrix(MunkresState<data> s) {
    size_t i = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i < s.SIZE * s.SIZE && assign_lap::near_zero(s.slack[i])) {
        size_t b = i >> s.log2_data_block_size;
        size_t i0 = i & ~((size_t)s.data_block_size - 1);
        size_t j = (size_t)atomicAdd((unsigned long long *)&s.zeros_size_b[b], 1ULL);
        s.zeros[i0 + j] = i;
    }
}

template <typename data = float>
__global__ void k_step_2(MunkresState<data> s) {
    uint i = threadIdx.x;
    uint b = blockIdx.x;
    __shared__ bool repeat, s_repeat_kernel;
    if (i == 0)
        s_repeat_kernel = false;
    // Inner do-while reclaims rows whose column-star failed mid-flight;
    // outer caller re-launches when *any* block flipped s_repeat_kernel.
    do {
        __syncthreads();
        if (i == 0)
            repeat = false;
        __syncthreads();
        for (uint j = i; j < s.zeros_size_b[b]; j += blockDim.x) {
            size_t z = s.zeros[((size_t)b << s.log2_data_block_size) + j];
            uint l = z % s.nrows;
            uint c = z / s.nrows;
            if (s.cover_row[l] == 0 && s.cover_column[c] == 0) {
                if (!atomicExch((int *)&s.cover_row[l], 1)) {
                    if (!atomicExch((int *)&s.cover_column[c], 1)) {
                        s.row_of_star_at_column[c] = l;
                        s.column_of_star_at_row[l] = c;
                    } else {
                        s.cover_row[l] = 0;
                        repeat = true;
                        s_repeat_kernel = true;
                    }
                }
            }
        }
        __syncthreads();
    } while (repeat);
    if (i == 0 && s_repeat_kernel)
        s.cb->repeat_kernel = 1;
}

template <typename data = float>
__global__ void k_step_3(MunkresState<data> s) {
    size_t i = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ int matches;
    if (threadIdx.x == 0)
        matches = 0;
    __syncthreads();
    if (i < s.nrows) {
        s.cover_row[i] = 0;
        s.cover_column[i] = 0;
        if (s.row_of_star_at_column[i] >= 0) {
            s.cover_column[i] = 1;
            atomicAdd(&matches, 1);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0)
        atomicAdd(&s.cb->n_matches, matches);
}

template <typename data = float>
__global__ void k_step_4_init(MunkresState<data> s) {
    size_t i = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i < s.SIZE) {
        s.column_of_prime_at_row[i] = -1;
        s.row_of_green_at_column[i] = -1;
    }
}

template <typename data = float>
__global__ void k_step_4(MunkresState<data> s) {
    __shared__ bool s_found, s_goto_5, s_repeat_kernel;
    // `volatile` plus the __threadfence below: cover writes from one
    // block must be visible to readers in other blocks within the same
    // launch. Reads through plain pointers would let the compiler
    // assume cover_* are stable across the loop.
    volatile int *v_cover_row = s.cover_row;
    volatile int *v_cover_column = s.cover_column;
    const size_t i = threadIdx.x;
    const size_t b = blockIdx.x;
    if (i == 0) {
        s_repeat_kernel = false;
        s_goto_5 = false;
    }
    do {
        __syncthreads();
        if (i == 0)
            s_found = false;
        __syncthreads();
        for (size_t j = i; j < s.zeros_size_b[b]; j += blockDim.x) {
            size_t z = s.zeros[(size_t)(b << (size_t)s.log2_data_block_size) + j];
            int l = z % s.nrows;
            int c = z / s.nrows;
            int c1 = s.column_of_star_at_row[l];
            if (!v_cover_column[c] && !v_cover_row[l]) {
                s_found = true;
                s_repeat_kernel = true;
                s.column_of_prime_at_row[l] = c;
                if (c1 >= 0) {
                    v_cover_row[l] = 1;
                    __threadfence();
                    v_cover_column[c1] = 0;
                } else {
                    s_goto_5 = true;
                }
            }
        }
        __syncthreads();
    } while (s_found && !s_goto_5);
    if (i == 0 && s_repeat_kernel)
        s.cb->repeat_kernel = 1;
    if (i == 0 && s_goto_5)
        s.cb->goto_5 = 1;
}

template <typename data = float>
__global__ void k_step_5a(MunkresState<data> s) {
    size_t i = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i < s.SIZE) {
        int c_Z0 = s.column_of_prime_at_row[i];
        if (c_Z0 >= 0 && s.column_of_star_at_row[i] < 0) {
            s.row_of_green_at_column[c_Z0] = i;
            int r_Z0;
            while ((r_Z0 = s.row_of_star_at_column[c_Z0]) >= 0) {
                c_Z0 = s.column_of_prime_at_row[r_Z0];
                s.row_of_green_at_column[c_Z0] = r_Z0;
            }
        }
    }
}

template <typename data = float>
__global__ void k_step_5b(MunkresState<data> s) {
    size_t j = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (j < s.SIZE) {
        int r_Z0 = s.row_of_green_at_column[j];
        if (r_Z0 >= 0 && s.row_of_star_at_column[j] < 0) {
            int c_Z2 = s.column_of_star_at_row[r_Z0];
            s.column_of_star_at_row[r_Z0] = j;
            s.row_of_star_at_column[j] = r_Z0;
            while (c_Z2 >= 0) {
                r_Z0 = s.row_of_green_at_column[c_Z2];
                int c_Z0 = c_Z2;
                c_Z2 = s.column_of_star_at_row[r_Z0];
                s.column_of_star_at_row[r_Z0] = c_Z0;
                s.row_of_star_at_column[c_Z0] = r_Z0;
            }
        }
    }
}

// Two-pass reduce: each block reduces a strided slice of the slack
// matrix (skipping cells whose row/column is covered) into one element
// of `g_odata`; a follow-up CUB reduction in the host driver collapses
// those partials into `cb->min_in_mat`.
template <typename data = float, uint blockSize = kThreadsReduction>
__global__ void k_min_reduce_uncovered(volatile data *g_idata, volatile data *g_odata,
                                       const size_t n, MunkresState<data> s) {
    __shared__ data sdata[blockSize];
    const uint tid = threadIdx.x;
    size_t i = (size_t)blockIdx.x * ((size_t)blockSize * 2) + tid;
    size_t gridSize = (size_t)blockSize * 2 * (size_t)gridDim.x;
    sdata[tid] = assign_lap::cost_inf<data>();
    while (i < n) {
        size_t l1 = i % s.nrows, c1 = i / s.nrows;
        data g1 = (s.cover_row[l1] || s.cover_column[c1]) ? assign_lap::cost_inf<data>()
                                                          : g_idata[i];
        size_t i2 = i + blockSize;
        data g2 = assign_lap::cost_inf<data>();
        if (i2 < s.nrows * s.nrows) {
            size_t l2 = i2 % s.nrows, c2 = i2 / s.nrows;
            if (!s.cover_row[l2] && !s.cover_column[c2])
                g2 = g_idata[i2];
        }
        sdata[tid] = min(sdata[tid], min(g1, g2));
        i += gridSize;
    }
    __syncthreads();
    using BR = cub::BlockReduce<data, blockSize>;
    __shared__ typename BR::TempStorage temp_storage;
    data minimum = BR(temp_storage).Reduce(sdata[tid], assign_lap::Min());
    if (tid == 0)
        g_odata[blockIdx.x] = minimum;
}

// Step 6 fuses the dual update with the next iteration's sparse-zero
// recompression: the same pass that adjusts slack also bucketizes every
// resulting near-zero entry into `zeros`/`zeros_size_b`. Saves a full
// re-scan over the K*K cost matrix between iterations.
template <typename data = float>
__global__ void k_step_6_update_and_compress(MunkresState<data> s) {
    const size_t i = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i < s.n_blocks_step_4)
        s.zeros_size_b[i] = 0;
    __syncthreads();
    if (i < s.SIZE * s.SIZE) {
        const size_t l = i % s.nrows;
        const size_t c = i / s.nrows;
        auto reg = s.slack[i];
        const data m = s.cb->min_in_mat;
        switch (s.cover_row[l] + s.cover_column[c]) {
        case 2:
            reg += m;
            s.slack[i] = reg;
            break;
        case 0:
            reg -= m;
            s.slack[i] = reg;
            break;
        default:
            break;
        }
        if (assign_lap::near_zero(reg)) {
            size_t b = i >> s.log2_data_block_size;
            size_t i0 = i & ~((size_t)s.data_block_size - 1);
            size_t j =
                (size_t)atomicAdd((unsigned long long *)s.zeros_size_b + b, 1ULL);
            s.zeros[i0 + j] = i;
        }
    }
}

template <typename data>
static void solve_with_workspace(MunkresWorkspace<data> &ws, cudaStream_t stream) {
    auto &s = ws.state;
    ControlBlock *cb = s.cb;
    const std::size_t N = s.SIZE;
    const uint num_blocks_reduction = ws.num_blocks_reduction;

    const uint n_thr = (uint)std::min(N, (std::size_t)64);
    const uint n_thr_full = (uint)std::min(N, (std::size_t)512);
    const std::size_t nb = (std::size_t)std::ceil(N / (double)n_thr);
    const std::size_t nb_full = (std::size_t)std::ceil((N * N) / (double)n_thr_full);

    // Bulk memset is one page fault and primes the managed page in the
    // working set for the kernels about to land on it.
    CUDA_RUNTIME(cudaMemsetAsync(cb, 0, sizeof(ControlBlock), stream));

    auto sum_zeros = [&]() -> unsigned long long {
        size_t tb = s.cub_temp_bytes;
        CUDA_RUNTIME(cub::DeviceReduce::Sum(s.cub_temp, tb, s.zeros_size_b,
                                            &cb->zeros_size, (int)s.NB4, stream));
        CUDA_RUNTIME(cudaStreamSynchronize(stream));
        return cb->zeros_size;
    };

    // Dispatching 1024 threads at a near-empty `zeros` array sends every
    // thread fighting over the same handful of atomic locations; the
    // contention explodes the do-while retry count. Cap blockDim to the
    // actual workload when we know it's small.
    auto step_block_dim = [&](unsigned long long zeros_size) -> uint {
        return (s.nb4 > 1 || zeros_size > (unsigned long long)kMaxThreadsPerBlock)
                   ? (uint)kMaxThreadsPerBlock
                   : (uint)zeros_size;
    };

    LAUNCH(k_init, nb, n_thr, stream, s);
    LAUNCH(k_calc_row_min, (uint)N, kThreadsReduction, stream, s);
    LAUNCH(k_row_sub, nb_full, n_thr_full, stream, s);
    LAUNCH(k_calc_col_min, (uint)N, kThreadsReduction, stream, s);
    LAUNCH(k_col_sub, nb_full, n_thr_full, stream, s);
    LAUNCH(k_compress_matrix, nb_full, n_thr_full, stream, s);

    unsigned long long zeros_size = sum_zeros();

    do {
        cb->repeat_kernel = 0;
        LAUNCH(k_step_2, s.nb4, step_block_dim(zeros_size), stream, s);
        CUDA_RUNTIME(cudaStreamSynchronize(stream));
    } while (cb->repeat_kernel);

    while (true) {
        cb->n_matches = 0;
        LAUNCH(k_step_3, nb, n_thr, stream, s);
        CUDA_RUNTIME(cudaStreamSynchronize(stream));
        if (cb->n_matches >= (int)s.ncols)
            break;

        LAUNCH(k_step_4_init, nb, n_thr, stream, s);

        while (true) {
            do {
                cb->goto_5 = 0;
                cb->repeat_kernel = 0;
                LAUNCH(k_step_4, s.nb4, step_block_dim(zeros_size), stream, s);
                CUDA_RUNTIME(cudaStreamSynchronize(stream));
            } while (cb->repeat_kernel && !cb->goto_5);

            if (cb->goto_5)
                break;

            // Step 6: dual update. The CUB reduce writes the min
            // directly into `cb->min_in_mat` (managed), so the host
            // check below is a managed read with no cudaMemcpy.
            LAUNCH((k_min_reduce_uncovered<data, kThreadsReduction>),
                   num_blocks_reduction, kThreadsReduction, stream, s.slack,
                   s.d_min_in_mat_vect, s.nrows * s.ncols, s);

            size_t tb = s.cub_temp_bytes;
            CUDA_RUNTIME(cub::DeviceReduce::Reduce(
                s.cub_temp, tb, s.d_min_in_mat_vect, &cb->min_in_mat,
                (int)num_blocks_reduction, assign_lap::Min(),
                assign_lap::cost_inf<data>(), stream));
            CUDA_RUNTIME(cudaStreamSynchronize(stream));

            if (cb->min_in_mat <= 0)
                throw std::runtime_error(
                    "lap_munkres: non-positive minimum in cost matrix");

            LAUNCH(k_step_6_update_and_compress, nb_full, n_thr_full, stream, s);

            zeros_size = sum_zeros();
        }

        LAUNCH(k_step_5a, nb, n_thr, stream, s);
        LAUNCH(k_step_5b, nb, n_thr, stream, s);
    }
}

} // anonymous namespace

namespace match_munkres {

using Tensor = torch::stable::Tensor;
namespace ts = torch::stable;
using ScT = torch::headeronly::ScalarType;

Tensor solve(const Tensor &cost, cudaStream_t stream) {
    const auto rows = cost.size(0);
    const auto cols = cost.size(1);

    auto cost_f = ts::to(cost, ScT::Float);
    const float sentinel = assign_lap::compute_inf_sentinel<float>(cost_f, stream);
    cost_f = assign_lap::rewrite_inf_to_sentinel(cost_f, sentinel, stream);

    // Transpose + pad to (K, K) column-major. After the transpose, what
    // was row-major (rows, cols) becomes the column-major layout of the
    // original; the narrow() call places it in the top-left (cols, rows)
    // sub-block of the K-square pad.
    auto cost_cm = ts::contiguous(ts::transpose(cost_f, 0, 1));
    const auto K = std::max(rows, cols);
    Tensor cost_sq;
    if (rows == cols) {
        cost_sq = cost_cm;
    } else {
        cost_sq = ts::full({K, K}, static_cast<double>(sentinel), ScT::Float,
                           std::nullopt, cost.device());
        Tensor view = ts::narrow(cost_sq, 0, 0, cols);
        view = ts::narrow(view, 1, 0, rows);
        ts::copy_(view, cost_cm);
    }

    const int dev = cost.get_device_index();
    auto ws = g_munkres_cache.acquire(static_cast<std::size_t>(K), dev);

    CUDA_RUNTIME(cudaMemcpyAsync(ws->state.slack, cost_sq.const_data_ptr<float>(),
                                 (std::size_t)K * (std::size_t)K * sizeof(float),
                                 cudaMemcpyDefault, stream));

    solve_with_workspace<float>(*ws, stream);
    CUDA_RUNTIME(cudaStreamSynchronize(stream));

    return assign_lap::repack_row_to_col(ws->state.column_of_star_at_row, rows, cols,
                                         cost.device());
}

} // namespace match_munkres
