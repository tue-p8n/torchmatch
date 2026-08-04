// Hybrid Hungarian for the LAP on GPU. Per outer iteration, the
// `decision()` host helper picks between classical single-path
// augmentation (S456_classical, as in munkres.cu) and Lawler tree
// augmentation (S456_tree, as in lawler.cu) based on observed
// match progress. The resulting solver inherits traits from both
// parents but, in the current benchmark sweep, has no demonstrated
// niche where it leads either; kept compiled but undocumented.
//
// Cost layout: row-major float, padded to square (K, K). Uses
// file-scope __constant__ and __managed__ device globals (SIZE, NB4,
// repeat_kernel, ...), which is why this TU stays separate from
// munkres.cu and lawler.cu: those symbols would collide. A
// process-wide mutex serializes every hybrid solve, since the same
// globals leave concurrent solves at different N undefined.

#include <cuda_runtime.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/Exception.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <mutex>
#include <optional>
#include <type_traits>
#include <vector>

#include <cub/cub.cuh>

#include "control.cuh"
#include "cuda_common.cuh"
#include "workspace.cuh"

using uint = unsigned int;

constexpr int BLOCK_DIMX = 256;

#define SYNC() CUDA_RUNTIME(cudaDeviceSynchronize())

using assign_lap::ACTIVE;
using assign_lap::AUGMENT;
using assign_lap::DORMANT;
using assign_lap::MODIFIED;
using assign_lap::REVERSE;
using assign_lap::VISITED;

__constant__ size_t SIZE;
__constant__ size_t SIZE2;
__constant__ uint NB4;
__constant__ uint NBR;
__constant__ uint DBS;
__constant__ uint L2DBS;

__managed__ __device__ int zeros_size;
__managed__ __device__ bool repeat_kernel, goto_5;
__managed__ __device__ long csr2_size, col_id_size, row_id_size;
__managed__ __device__ int nmatch_cur, nmatch_old;

template <typename T = uint>
struct VertexData {
    int *parents;
    int *children;
    int *is_visited;
    T *slack;
};

struct Predicates {
    long size;
    bool *predicates;
    long *addresses;
    long *out_addresses;
};

template <typename data>
struct HybridWorkspace {
    assign_lap::WorkspacePool pool;
    std::size_t N = 0;
    std::size_t N2 = 0;
    int device = -1;
    bool initialized = false;

    // Sizing scalars (mirrored into __constant__ on each solve)
    uint nb4 = 0;
    uint nbr = 0;
    uint dbs = 0;
    uint l2dbs = 0;

    // Step 1/2/3/6 (classical core)
    double *row_duals = nullptr;
    double *col_duals = nullptr;
    data *slack = nullptr;
    data *min_vect = nullptr;
    data *min_mat = nullptr; // managed
    int *row_ass = nullptr;
    int *col_ass = nullptr;
    int *row_cover = nullptr;
    int *col_cover = nullptr;
    size_t *zeros = nullptr;
    size_t *zeros_size_b = nullptr;
    int *row_visited = nullptr;
    int *col_visited = nullptr;

    // Tree path (CtoT): always pre-allocated; original lazily allocated.
    int *row_data_is_visited = nullptr;
    int *row_data_parents = nullptr;
    int *row_data_children = nullptr;
    int *col_data_is_visited = nullptr;
    int *col_data_parents = nullptr;
    int *col_data_children = nullptr;
    data *col_data_slack = nullptr;
    bool *vertex_predicates_p = nullptr;
    long *vertex_predicates_a = nullptr;
    int *vertices_csr1 = nullptr;
    int *vertices_csr2 = nullptr;
    double *row_duals_tree = nullptr;
    double *col_duals_tree = nullptr;

    // S456_tree per-iteration buffers (sized N, the maximum compaction size)
    bool *col_predicates_p = nullptr;
    long *col_predicates_a = nullptr;
    int *col_id_csr_elements = nullptr;
    bool *row_predicates_p = nullptr;
    long *row_predicates_a = nullptr;
    int *row_id_csr_elements = nullptr;

    // Host theta scratch (avoids per-iteration new[] / delete[])
    data *h_col_slack = nullptr;
    int *h_col_cover = nullptr;
    int *h_col_ass = nullptr;

    // CUB scratch (sized for the largest of Reduce::Min, Reduce::Sum,
    // Reduce::Sum on long, ExclusiveSum on long; computed in allocate).
    void *cub_storage = nullptr;
    size_t cub_storage_bytes = 0;
    size_t b1 = 0, b2 = 0, b3 = 0, b4 = 0;

    void allocate(std::size_t n, int dev) {
        N = n;
        N2 = n * n;
        device = dev;
        pool.reset(dev);

        const uint cpbs4 = 512;
        nb4 = std::max((uint)std::ceil((N * 1.0) / cpbs4), 1u);
        nbr = std::min((uint)N, 256u);
        dbs = cpbs4 * (uint)std::pow(2, std::ceil(std::log2((double)N)));
        l2dbs = (uint)std::log2((double)dbs);

        row_duals = static_cast<decltype(row_duals)>(pool.alloc(N * sizeof(double)));
        col_duals = static_cast<decltype(col_duals)>(pool.alloc(N * sizeof(double)));
        slack = static_cast<decltype(slack)>(pool.alloc(N2 * sizeof(data)));
        zeros = static_cast<decltype(zeros)>(pool.alloc(N2 * sizeof(size_t)));
        zeros_size_b =
            static_cast<decltype(zeros_size_b)>(pool.alloc(nb4 * sizeof(size_t)));
        row_ass = static_cast<decltype(row_ass)>(pool.alloc(N * sizeof(int)));
        col_ass = static_cast<decltype(col_ass)>(pool.alloc(N * sizeof(int)));
        row_cover = static_cast<decltype(row_cover)>(pool.alloc(N * sizeof(int)));
        col_cover = static_cast<decltype(col_cover)>(pool.alloc(N * sizeof(int)));
        min_vect = static_cast<decltype(min_vect)>(pool.alloc(nbr * sizeof(data)));
        CUDA_RUNTIME(cudaMallocManaged(&min_mat, sizeof(data)));
        row_visited = static_cast<decltype(row_visited)>(pool.alloc(N * sizeof(int)));
        col_visited = static_cast<decltype(col_visited)>(pool.alloc(N * sizeof(int)));

        row_data_is_visited =
            static_cast<decltype(row_data_is_visited)>(pool.alloc(N * sizeof(int)));
        row_data_parents =
            static_cast<decltype(row_data_parents)>(pool.alloc(N * sizeof(int)));
        row_data_children =
            static_cast<decltype(row_data_children)>(pool.alloc(N * sizeof(int)));
        col_data_is_visited =
            static_cast<decltype(col_data_is_visited)>(pool.alloc(N * sizeof(int)));
        col_data_parents =
            static_cast<decltype(col_data_parents)>(pool.alloc(N * sizeof(int)));
        col_data_children =
            static_cast<decltype(col_data_children)>(pool.alloc(N * sizeof(int)));
        col_data_slack =
            static_cast<decltype(col_data_slack)>(pool.alloc(N * sizeof(data)));
        vertex_predicates_p =
            static_cast<decltype(vertex_predicates_p)>(pool.alloc(N * sizeof(bool)));
        vertex_predicates_a =
            static_cast<decltype(vertex_predicates_a)>(pool.alloc(N * sizeof(long)));
        vertices_csr1 =
            static_cast<decltype(vertices_csr1)>(pool.alloc(N * sizeof(int)));
        vertices_csr2 =
            static_cast<decltype(vertices_csr2)>(pool.alloc(N * sizeof(int)));
        row_duals_tree =
            static_cast<decltype(row_duals_tree)>(pool.alloc(N * sizeof(double)));
        col_duals_tree =
            static_cast<decltype(col_duals_tree)>(pool.alloc(N * sizeof(double)));

        col_predicates_p =
            static_cast<decltype(col_predicates_p)>(pool.alloc(N * sizeof(bool)));
        col_predicates_a =
            static_cast<decltype(col_predicates_a)>(pool.alloc(N * sizeof(long)));
        col_id_csr_elements =
            static_cast<decltype(col_id_csr_elements)>(pool.alloc(N * sizeof(int)));
        row_predicates_p =
            static_cast<decltype(row_predicates_p)>(pool.alloc(N * sizeof(bool)));
        row_predicates_a =
            static_cast<decltype(row_predicates_a)>(pool.alloc(N * sizeof(long)));
        row_id_csr_elements =
            static_cast<decltype(row_id_csr_elements)>(pool.alloc(N * sizeof(int)));

        h_col_slack = new data[N];
        h_col_cover = new int[N];
        h_col_ass = new int[N];

        // Size CUB storage for the largest of the four reductions/scans.
        cub::DeviceReduce::Reduce(nullptr, b1, min_vect, min_mat, (int)nbr, cub::Min(),
                                  (data)1.0);
        cub::DeviceReduce::Sum(nullptr, b2, zeros_size_b, (size_t *)zeros_size_b,
                               (int)nb4);
        cub::DeviceReduce::Sum(nullptr, b3, vertex_predicates_a, (long *)nullptr,
                               (int)N);
        cub::DeviceScan::ExclusiveSum(nullptr, b4, vertex_predicates_a,
                                      (long *)vertex_predicates_a, (int)N);
        cub_storage_bytes = std::max(b1, std::max(b2, std::max(b3, b4)));
        cub_storage = static_cast<decltype(cub_storage)>(pool.alloc(cub_storage_bytes));

        initialized = true;
    }

    void deallocate() {
        if (!initialized)
            return;
        // min_mat is managed memory, which cannot come from the caching
        // allocator, so it is freed on its own.
        CUDA_RUNTIME(cudaFree(min_mat));

        delete[] h_col_slack;
        delete[] h_col_cover;
        delete[] h_col_ass;

        // All device buffers are pooled: one release frees all of them.
        pool.release();
        initialized = false;
    }
};

static assign_lap::WorkspaceCache<HybridWorkspace<float>> g_hybrid_cache;
static std::mutex g_hybrid_mtx;

template <typename data>
__global__ void row_reduce(double *row_min, data *slack) {
    const size_t tid = threadIdx.x;
    const size_t rowID = (size_t)blockIdx.x * SIZE;
    data thread_min = assign_lap::cost_inf<data>();
    for (size_t i = tid; i < SIZE; i += blockDim.x)
        thread_min = min(thread_min, slack[i + rowID]);
    __syncthreads();
    using BR = cub::BlockReduce<data, BLOCK_DIMX>;
    __shared__ typename BR::TempStorage temp_storage;
    thread_min = BR(temp_storage).Reduce(thread_min, cub::Min());
    if (threadIdx.x == 0)
        row_min[blockIdx.x] = (double)thread_min;
    __syncthreads();
    for (size_t i = tid; i < SIZE; i += blockDim.x)
        slack[i + rowID] = slack[i + rowID] - (data)row_min[blockIdx.x];
}

template <typename data>
__global__ void col_min(const data *slack, double *col_min) {
    size_t tid = (size_t)threadIdx.x;
    const size_t colID = blockIdx.x;
    data thread_min = assign_lap::cost_inf<data>();
    for (size_t i = tid; i < SIZE; i += blockDim.x)
        thread_min = min(thread_min, slack[i * SIZE + colID]);
    __syncthreads();
    using BR = cub::BlockReduce<data, BLOCK_DIMX>;
    __shared__ typename BR::TempStorage temp_storage;
    thread_min = BR(temp_storage).Reduce(thread_min, cub::Min());
    if (threadIdx.x == 0)
        col_min[blockIdx.x] = (double)thread_min;
}

template <typename data>
__global__ void col_sub(data *slack, double *col_min) {
    size_t tid = threadIdx.x;
    const size_t rowID = (size_t)blockIdx.x * SIZE;
    for (size_t i = tid; i < SIZE; i += blockDim.x)
        slack[i + rowID] = slack[i + rowID] - (data)col_min[i];
}
__global__ void init(int *row_ass, int *col_ass, int *row_cover, int *col_cover) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < SIZE) {
        col_ass[i] = -1;
        row_ass[i] = -1;
        row_cover[i] = 0;
        col_cover[i] = 0;
    }
}

template <typename data>
__global__ void compress_matrix(size_t *zeros, size_t *zeros_size_b, data *slack) {
    size_t i = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
    if (i < SIZE2) {
        if (assign_lap::near_zero(slack[i])) {
            size_t b = i >> L2DBS;
            size_t i0 = i & ~((size_t)DBS - 1);
            size_t j = (size_t)atomicAdd((unsigned long long *)&zeros_size_b[b], 1ULL);
            zeros[i0 + j] = i;
        }
    }
}

__global__ void step2(const size_t *zeros, const size_t *zeros_size_b, int *row_cover,
                      int *col_cover, int *row_ass, int *col_ass) {
    uint i = threadIdx.x;
    uint b = blockIdx.x;
    __shared__ bool repeat, s_repeat_kernel;
    if (i == 0)
        s_repeat_kernel = false;
    do {
        __syncthreads();
        if (i == 0)
            repeat = false;
        __syncthreads();
        for (uint j = i; j < zeros_size_b[b]; j += blockDim.x) {
            size_t z = zeros[(b << L2DBS) + j];
            uint l = z % SIZE;
            uint c = z / SIZE;
            if (row_cover[l] == 0 && col_cover[c] == 0) {
                if (!atomicExch((int *)&(row_cover[l]), 1)) {
                    if (!atomicExch((int *)&(col_cover[c]), 1)) {
                        row_ass[c] = l;
                        col_ass[l] = c;
                    } else {
                        row_cover[l] = 0;
                        repeat = true;
                        s_repeat_kernel = true;
                    }
                }
            }
        }
        __syncthreads();
    } while (repeat);
    if (s_repeat_kernel)
        repeat_kernel = true;
}

__global__ void step3(const int *row_ass, int *col_cover) {
    size_t tid = threadIdx.x;
    size_t i = tid + (size_t)blockIdx.x * blockDim.x;
    __shared__ int matches;
    if (tid == 0)
        matches = 0;
    __syncthreads();
    if (i < SIZE) {
        if (row_ass[i] >= 0) {
            col_cover[i] = 1;
            atomicAdd((int *)&matches, 1);
        }
    }
    __syncthreads();
    if (tid == 0)
        atomicAdd((int *)&nmatch_cur, matches);
}

namespace classical {
__global__ void S4_init(int *col_visited, int *row_visited) {
    size_t tid = threadIdx.x;
    size_t i = tid + (size_t)blockIdx.x * blockDim.x;
    if (i < SIZE) {
        col_visited[i] = -1;
        row_visited[i] = -1;
    }
}
} // namespace classical

__global__ void S4(int *row_cover, int *col_cover, int *col_visited,
                   const size_t *zeros, const size_t *zeros_size_b,
                   const int *col_ass) {
    __shared__ bool s_found, s_goto_5, s_repeat_kernel;
    volatile int *v_row_cover = row_cover;
    volatile int *v_col_cover = col_cover;
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
        for (size_t j = threadIdx.x; j < zeros_size_b[b]; j += blockDim.x) {
            size_t z = zeros[(size_t)(b << (size_t)L2DBS) + j];
            int l = z % SIZE;
            int c = z / SIZE;
            int c1 = col_ass[l];
            if (!v_col_cover[c] && !v_row_cover[l]) {
                s_found = true;
                s_repeat_kernel = true;
                col_visited[l] = c;
                if (c1 >= 0) {
                    v_row_cover[l] = 1;
                    __threadfence();
                    v_col_cover[c1] = 0;
                } else {
                    s_goto_5 = true;
                }
            }
        }
        __syncthreads();
    } while (s_found && !s_goto_5);
    if (i == 0 && s_repeat_kernel)
        repeat_kernel = true;
    if (i == 0 && s_goto_5)
        goto_5 = true;
}

__global__ void S5a(int *col_visited, int *row_visited, const int *row_ass,
                    const int *col_ass) {
    size_t i = (size_t)blockDim.x * blockIdx.x + (size_t)threadIdx.x;
    if (i < SIZE) {
        int r_Z0, c_Z0;
        c_Z0 = col_visited[i];
        if (c_Z0 >= 0 && col_ass[i] < 0) {
            row_visited[c_Z0] = i;
            while ((r_Z0 = row_ass[c_Z0]) >= 0) {
                c_Z0 = col_visited[r_Z0];
                row_visited[c_Z0] = r_Z0;
            }
        }
    }
}

__global__ void S5b(int *row_visited, int *row_ass, int *col_ass) {
    size_t j = (size_t)blockDim.x * blockIdx.x + (size_t)threadIdx.x;
    if (j < SIZE) {
        int r_Z0, c_Z0, c_Z2;
        r_Z0 = row_visited[j];
        if (r_Z0 >= 0 && row_ass[j] < 0) {
            c_Z2 = col_ass[r_Z0];
            col_ass[r_Z0] = j;
            row_ass[j] = r_Z0;
            while (c_Z2 >= 0) {
                r_Z0 = row_visited[c_Z2];
                c_Z0 = c_Z2;
                c_Z2 = col_ass[r_Z0];
                col_ass[r_Z0] = c_Z0;
                row_ass[c_Z0] = r_Z0;
            }
        }
    }
}

template <typename data = int, uint blockSize = BLOCK_DIMX>
__global__ void min_reduce_kernel1(volatile data *g_idata, volatile data *g_odata,
                                   const int *row_cover, const int *col_cover) {
    __shared__ data sdata[blockSize];
    const uint tid = threadIdx.x;
    size_t i = (size_t)blockIdx.x * ((size_t)blockSize * 2) + (size_t)tid;
    size_t gridSize = (size_t)blockSize * 2 * (size_t)gridDim.x;
    sdata[tid] = assign_lap::cost_inf<data>();
    while (i < SIZE2) {
        size_t i1 = i, i2 = i + blockSize;
        size_t l1 = i1 % SIZE, c1 = i1 / SIZE;
        data g1 = assign_lap::cost_inf<data>(), g2 = assign_lap::cost_inf<data>();
        if (!(row_cover[l1] == 1 || col_cover[c1] == 1))
            g1 = g_idata[i1];
        if (i2 < SIZE2) {
            size_t l2 = i2 % SIZE, c2 = i2 / SIZE;
            if (!(row_cover[l2] == 1 || col_cover[c2] == 1))
                g2 = g_idata[i2];
        }
        sdata[tid] = min(sdata[tid], min(g1, g2));
        i += gridSize;
    }
    __syncthreads();
    using BlockReduce = cub::BlockReduce<data, blockSize>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    data minimum = BlockReduce(temp_storage).Reduce(sdata[tid], cub::Min());
    if (tid == 0)
        g_odata[blockIdx.x] = minimum;
}

template <typename data>
__global__ void S6_DualUpdate(const int *cover_row, const int *cover_column,
                              const data *min_mat, double *min_in_rows,
                              double *min_in_cols) {
    const size_t i = (size_t)blockDim.x * blockIdx.x + (size_t)threadIdx.x;
    if (i < SIZE) {
        if (cover_row[i] == 0)
            min_in_rows[i] += ((double)1.0 * min_mat[0]) / 2;
        else
            min_in_rows[i] -= ((double)1.0 * min_mat[0]) / 2;
        if (cover_column[i] == 0)
            min_in_cols[i] += ((double)1.0 * min_mat[0]) / 2;
        else
            min_in_cols[i] -= ((double)1.0 * min_mat[0]) / 2;
    }
}

template <typename data>
__global__ void S6_update(data *slack, const int *row_cover, const int *col_cover,
                          const data *min_mat, size_t *zeros, size_t *zeros_size_b) {
    const size_t i = (size_t)blockDim.x * blockIdx.x + (size_t)threadIdx.x;
    if (i < SIZE2) {
        const size_t l = i % SIZE;
        const size_t c = i / SIZE;
        data reg = slack[i];
        switch (row_cover[l] + col_cover[c]) {
        case 2:
            reg += min_mat[0];
            slack[i] = reg;
            break;
        case 0:
            reg -= min_mat[0];
            slack[i] = reg;
            break;
        default:
            break;
        }
        if (assign_lap::near_zero(reg)) {
            size_t b = i >> L2DBS;
            size_t i0 = i & ~((size_t)DBS - 1);
            size_t j = (size_t)atomicAdd((unsigned long long *)zeros_size_b + b, 1ULL);
            zeros[i0 + j] = i;
        }
    }
}

namespace tree {

template <typename data = uint>
__global__ void Initialization(int *d_row_assignments, int *row_cover, int *col_cover,
                               VertexData<data> row_data, VertexData<data> col_data) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < SIZE) {
        int assignment = d_row_assignments[id];
        row_data.is_visited[id] = (assignment == -1) ? ACTIVE : DORMANT;
        row_cover[id] = (assignment == -1) ? 0 : 1;

        col_cover[id] = 0;
        col_data.slack[id] = INFINITY;
        col_data.is_visited[id] = DORMANT;

        row_data.parents[id] = -1;
        col_data.parents[id] = -1;
        row_data.children[id] = -1;
        col_data.children[id] = -1;
    }
}

__global__ void S4_init(int *vertices_csr1) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < SIZE)
        vertices_csr1[id] = id;
}

} // namespace tree

__global__ void vertexPredicateConstructionCSR(Predicates vertex_predicates,
                                               int *vertices_csr1, int *visited) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < SIZE) {
        int vertexid = vertices_csr1[id];
        int visit = (vertexid != -1) ? visited[vertexid] : DORMANT;
        bool predicate = (visit == ACTIVE);
        long addr = predicate ? 1 : 0;
        vertex_predicates.predicates[id] = predicate;
        vertex_predicates.addresses[id] = addr;
    }
}

__global__ void vertexScatterCSR(int *d_vertex_ids_csr, int *d_vertex_ids,
                                 int *d_visited, const Predicates d_vertex_predicates) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < SIZE) {
        int vertexid = d_vertex_ids[id];
        bool predicate = d_vertex_predicates.predicates[id];
        if (predicate) {
            long compid = d_vertex_predicates.addresses[id];
            d_vertex_ids_csr[compid] = vertexid;
            d_visited[id] = VISITED;
        }
    }
}

template <typename data = uint>
__device__ void __traverse(data *d_costs, double *row_duals, double *col_duals,
                           int *row_ass, int *col_ass, int *row_cover, int *col_cover,
                           int *d_row_parents, int *d_col_parents, int *d_row_visited,
                           int *d_col_visited, data *d_slacks, int *d_start_ptr,
                           int *d_end_ptr, const size_t colid) {
    int *ptr1 = d_start_ptr;
    while (ptr1 != d_end_ptr) {
        int rowid = *ptr1;
        data slack;
        if (std::is_same_v<data, uint> || std::is_same_v<data, int>)
            slack = (data)(d_costs[colid * SIZE + rowid] -
                           (int)(row_duals[rowid] + col_duals[colid]));
        else
            slack = d_costs[colid * SIZE + rowid] -
                    (data)(row_duals[rowid] + col_duals[colid]);
        int nxt_rowid = col_ass[colid];
        if (rowid != nxt_rowid && col_cover[colid] == 0) {
            if (slack < d_slacks[colid]) {
                d_slacks[colid] = slack;
                d_col_parents[colid] = rowid;
            }
            if (assign_lap::near_zero(d_slacks[colid])) {
                if (nxt_rowid != -1) {
                    d_row_parents[nxt_rowid] = colid;
                    row_cover[nxt_rowid] = 0;
                    col_cover[colid] = 1;
                    if (d_row_visited[nxt_rowid] != VISITED)
                        d_row_visited[nxt_rowid] = ACTIVE;
                } else {
                    d_col_visited[colid] = REVERSE;
                    goto_5 = true;
                }
            }
        }
        d_row_visited[rowid] = VISITED;
        ptr1++;
    }
}

template <typename data = uint>
__global__ void coverAndExpand(int *vertices_csr2, const size_t csr2_size,
                               data *d_costs, double *row_duals, double *col_duals,
                               int *row_ass, int *col_ass, int *row_cover,
                               int *col_cover, VertexData<data> row_data,
                               VertexData<data> col_data) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t in_size = csr2_size;
    int *st_ptr = vertices_csr2;
    int *end_ptr = vertices_csr2 + in_size;
    if (id < SIZE) {
        __traverse(d_costs, row_duals, col_duals, row_ass, col_ass, row_cover,
                   col_cover, row_data.parents, col_data.parents, row_data.is_visited,
                   col_data.is_visited, col_data.slack, st_ptr, end_ptr, id);
    }
}

__device__ inline void __reverse_traversal(int *d_row_visited, int *d_row_children,
                                           int *d_col_children, int *d_row_parents,
                                           int *d_col_parents, int init_colid) {
    int cur_colid = init_colid;
    int cur_rowid = -1;
    while (cur_colid != -1) {
        d_col_children[cur_colid] = cur_rowid;
        cur_rowid = d_col_parents[cur_colid];
        d_row_children[cur_rowid] = cur_colid;
        cur_colid = d_row_parents[cur_rowid];
    }
    d_row_visited[cur_rowid] = AUGMENT;
}

__global__ void augmentPredicateConstruction(Predicates d_predicates, int *d_visited) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    int visited = (id < SIZE) ? d_visited[id] : DORMANT;
    bool predicate = (visited == REVERSE || visited == AUGMENT);
    long addr = predicate ? 1 : 0;
    if (id < SIZE) {
        d_predicates.predicates[id] = predicate;
        d_predicates.addresses[id] = addr;
    }
}

__global__ void augmentScatter(int *vertex_ids, Predicates predicates) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    bool predicate = (id < SIZE) ? predicates.predicates[id] : false;
    long compid = predicate ? predicates.addresses[id] : -1;
    if (id < SIZE && predicate)
        vertex_ids[compid] = id;
}

template <typename data = uint>
__global__ void reverseTraversal(int *col_vertices, VertexData<data> row_data,
                                 VertexData<data> col_data, size_t size) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        int colid = col_vertices[id];
        __reverse_traversal(row_data.is_visited, row_data.children, col_data.children,
                            row_data.parents, col_data.parents, colid);
    }
}

__device__ inline void __augment(int *d_row_assignments, int *d_col_assignments,
                                 int *d_row_children, int *d_col_children,
                                 int init_rowid) {
    int cur_colid = -1;
    int cur_rowid = init_rowid;
    while (cur_rowid != -1) {
        cur_colid = d_row_children[cur_rowid];
        d_row_assignments[cur_rowid] = cur_colid;
        d_col_assignments[cur_colid] = cur_rowid;
        cur_rowid = d_col_children[cur_colid];
    }
}

template <typename data = uint>
__global__ void augment(int *row_ass, int *col_ass, int *row_vertices,
                        VertexData<data> row_data, VertexData<data> col_data,
                        size_t size) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        int rowid = row_vertices[id];
        __augment(row_ass, col_ass, row_data.children, col_data.children, rowid);
    }
}

namespace tree {

template <typename data = uint>
__global__ void dualUpdate(double min_val, double *row_duals, double *col_duals,
                           data *col_slacks, int *row_covers, int *col_covers,
                           int *col_parents, int *row_visited) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < SIZE) {
        int row_cover = row_covers[id];
        int col_cover = col_covers[id];
        if (row_cover == 0)
            row_duals[id] += min_val;
        else
            row_duals[id] -= min_val;
        if (col_cover == 1) {
            col_duals[id] -= min_val;
        } else {
            col_duals[id] += min_val;
            col_slacks[id] -= (data)(2 * min_val);
            if (assign_lap::near_zero(col_slacks[id])) {
                int par_rowid = col_parents[id];
                row_visited[par_rowid] = ACTIVE;
            }
        }
    }
}

} // namespace tree

template <typename data>
class HLAP {
  private:
    HybridWorkspace<data> *ws;
    size_t psize, psize2;
    data *h_costs;
    data *d_costs;

    const uint cpbs4 = 512;
    uint nb4, nbr, dbs, l2dbs;

    const uint n_threads = (uint)min(psize, 64UL);
    const uint n_threads_full = (uint)min(psize, 512UL);
    const uint n_threads_reduction = 256;
    size_t n_blocks, n_blocks_full;

    // Aliases to workspace buffers (set in ctor; not owned by HLAP).
    double *row_duals, *col_duals;
    data *slack;
    data *min_mat, *min_vect;
    int *row_cover, *col_cover;
    size_t *zeros, *zeros_size_b;
    int *row_visited, *col_visited;
    void *cub_storage = nullptr;

    int *vertices_csr1;
    int *vertices_csr2;
    VertexData<data> row_data, col_data;
    Predicates vertex_predicates;
    double *row_duals_tree, *col_duals_tree;

    size_t b1 = 0, b2 = 0, b3 = 0, b4 = 0;

  public:
    int *row_ass;
    int *col_ass;

    HLAP(data *cost, HybridWorkspace<data> *workspace, cudaStream_t stream)
        : ws(workspace), h_costs(cost), psize(workspace->N) {
        psize2 = psize * psize;
        CUDA_RUNTIME(cudaMemcpyToSymbol(SIZE, &psize, sizeof(SIZE)));
        CUDA_RUNTIME(cudaMemcpyToSymbol(SIZE2, &psize2, sizeof(SIZE2)));
        n_blocks = (size_t)ceil((psize * 1.0) / n_threads);
        n_blocks_full = (size_t)ceil((psize2 * 1.0) / n_threads_full);

        nb4 = ws->nb4;
        nbr = ws->nbr;
        dbs = ws->dbs;
        l2dbs = ws->l2dbs;

        CUDA_RUNTIME(cudaMemcpyToSymbol(NB4, &nb4, sizeof(NB4)));
        CUDA_RUNTIME(cudaMemcpyToSymbol(NBR, &nbr, sizeof(NBR)));
        CUDA_RUNTIME(cudaMemcpyToSymbol(DBS, &dbs, sizeof(DBS)));
        CUDA_RUNTIME(cudaMemcpyToSymbol(L2DBS, &l2dbs, sizeof(L2DBS)));

        Allocate();
        CUDA_RUNTIME(cudaMemcpyAsync(slack, cost, psize2 * sizeof(data),
                                     cudaMemcpyDefault, stream));
        // Workspace is reused across calls, so duals must be cleared here.
        CUDA_RUNTIME(cudaMemsetAsync(row_duals, 0, psize * sizeof(double), stream));
        CUDA_RUNTIME(cudaMemsetAsync(col_duals, 0, psize * sizeof(double), stream));
        d_costs = nullptr;
    }

    ~HLAP() { DeAllocate(); }

    void solve() { solve_on_stream(nullptr); }

    void solve_on_stream(cudaStream_t stream) {
        // CUB temp is sized once at workspace creation; reuse here.
        b1 = ws->b1;
        b2 = ws->b2;
        b3 = ws->b3;
        b4 = ws->b4;

        S1(stream);
        S2(stream);

        std::vector<int> match_trend;
        nmatch_cur = 0;
        nmatch_old = 0;
        SYNC();
        S3(stream);
        match_trend.push_back(nmatch_cur - nmatch_old);
        bool first = true;

        while (nmatch_cur < (int)psize) {
            if (decision(match_trend)) {
                S456_classical(stream);
            } else {
                if (first) {
                    CtoT(stream);
                    first = false;
                }
                S456_tree(stream);
            }
            S3(stream);
            match_trend.push_back(nmatch_cur - nmatch_old);
        }
        // cub_storage is owned by the workspace; do not free here.
    }

  private:
    void Allocate() {
        // Workspace owns the buffers; we just alias to its pointers.
        row_duals = ws->row_duals;
        col_duals = ws->col_duals;
        slack = ws->slack;
        zeros = ws->zeros;
        zeros_size_b = ws->zeros_size_b;
        row_ass = ws->row_ass;
        col_ass = ws->col_ass;
        row_cover = ws->row_cover;
        col_cover = ws->col_cover;
        min_vect = ws->min_vect;
        min_mat = ws->min_mat;
        row_visited = ws->row_visited;
        col_visited = ws->col_visited;
        cub_storage = ws->cub_storage;
    }

    void DeAllocate() { /* workspace owns; no-op */ }

    void S1(cudaStream_t stream) {
        LAUNCH(row_reduce, psize, BLOCK_DIMX, stream, row_duals, slack);
        SYNC();
        LAUNCH(col_min, psize, BLOCK_DIMX, stream, slack, col_duals);
        SYNC();
        LAUNCH(col_sub, psize, BLOCK_DIMX, stream, slack, col_duals);
        SYNC();
    }

    void S2(cudaStream_t stream) {
        uint gridDim = (uint)ceil(psize * 1.0 / BLOCK_DIMX);
        LAUNCH(init, gridDim, BLOCK_DIMX, stream, row_ass, col_ass, row_cover,
               col_cover);
        SYNC();
        CUDA_RUNTIME(cudaMemset(zeros_size_b, 0, nb4 * sizeof(size_t)));
        LAUNCH((compress_matrix<data>), n_blocks_full, n_threads_full, stream, zeros,
               zeros_size_b, slack);
        SYNC();
        CUDA_RUNTIME(cub::DeviceReduce::Sum(cub_storage, b2, zeros_size_b, &zeros_size,
                                            (int)nb4));
        SYNC();
        do {
            repeat_kernel = false;
            SYNC();
            uint temp_blockDim =
                (nb4 > 1 || zeros_size > 1024) ? 1024 : (uint)zeros_size;
            LAUNCH(step2, nb4, temp_blockDim, stream, zeros, zeros_size_b, row_cover,
                   col_cover, row_ass, col_ass);
            SYNC();
        } while (repeat_kernel);
    }

    void S3(cudaStream_t stream) {
        CUDA_RUNTIME(cudaMemset(row_cover, 0, psize * sizeof(int)));
        CUDA_RUNTIME(cudaMemset(col_cover, 0, psize * sizeof(int)));
        nmatch_old = nmatch_cur;
        nmatch_cur = 0;
        SYNC();
        LAUNCH(step3, n_blocks, n_threads, stream, row_ass, col_cover);
        SYNC();
    }

    void S6(cudaStream_t stream) {
        LAUNCH((min_reduce_kernel1<data, 256>), nbr, n_threads_reduction, stream, slack,
               min_vect, row_cover, col_cover);
        SYNC();
        CUDA_RUNTIME(cub::DeviceReduce::Reduce(cub_storage, b1, min_vect, min_mat, nbr,
                                               cub::Min(),
                                               assign_lap::cost_inf<data>()));
        SYNC();
        data temp;
        CUDA_RUNTIME(cudaMemcpy(&temp, min_mat, sizeof(data), cudaMemcpyDeviceToHost));
        if (temp <= 0)
            throw std::runtime_error("lap_hybrid: non-positive minimum in cost matrix");

        zeros_size = 0;
        CUDA_RUNTIME(cudaMemset(zeros_size_b, 0, nb4 * sizeof(size_t)));

        uint gridDim = (uint)ceil(psize * 1.0 / BLOCK_DIMX);
        LAUNCH(S6_DualUpdate, gridDim, BLOCK_DIMX, stream, row_cover, col_cover,
               min_mat, row_duals, col_duals);
        SYNC();
        LAUNCH(S6_update, n_blocks_full, n_threads_full, stream, slack, row_cover,
               col_cover, min_mat, zeros, zeros_size_b);
        SYNC();
        CUDA_RUNTIME(
            cub::DeviceReduce::Sum(cub_storage, b2, zeros_size_b, &zeros_size, nb4));
        SYNC();
    }

    void S456_classical(cudaStream_t stream) {
        LAUNCH(classical::S4_init, n_blocks, n_threads, stream, col_visited,
               row_visited);
        SYNC();
        while (1) {
            do {
                goto_5 = false;
                repeat_kernel = false;
                SYNC();
                uint temp_blockDim =
                    (nb4 > 1 || zeros_size > 1024) ? 1024 : (uint)zeros_size;
                LAUNCH(S4, nb4, temp_blockDim, stream, row_cover, col_cover,
                       col_visited, zeros, zeros_size_b, col_ass);
                SYNC();
            } while (repeat_kernel && !goto_5);
            if (goto_5)
                break;
            S6(stream);
        }
        LAUNCH(S5a, n_blocks, n_threads, stream, col_visited, row_visited, row_ass,
               col_ass);
        SYNC();
        LAUNCH(S5b, n_blocks, n_threads, stream, row_visited, row_ass, col_ass);
        SYNC();
    }

    void CtoT(cudaStream_t stream) {
        const size_t N = psize;
        d_costs = slack;

        // Pointers come from the workspace (allocated once at workspace
        // creation, sized to N).
        row_data.is_visited = ws->row_data_is_visited;
        row_data.parents = ws->row_data_parents;
        row_data.children = ws->row_data_children;
        col_data.is_visited = ws->col_data_is_visited;
        col_data.parents = ws->col_data_parents;
        col_data.children = ws->col_data_children;
        col_data.slack = ws->col_data_slack;
        vertex_predicates.size = (long)N;
        vertex_predicates.predicates = ws->vertex_predicates_p;
        vertex_predicates.addresses = ws->vertex_predicates_a;
        vertices_csr1 = ws->vertices_csr1;
        vertices_csr2 = ws->vertices_csr2;
        row_duals_tree = ws->row_duals_tree;
        col_duals_tree = ws->col_duals_tree;

        CUDA_RUNTIME(cudaMemsetAsync(row_duals_tree, 0, N * sizeof(double), stream));
        CUDA_RUNTIME(cudaMemsetAsync(col_duals_tree, 0, N * sizeof(double), stream));
    }

    void S456_tree(cudaStream_t stream) {
        goto_5 = false;
        SYNC();
        uint gridDim = (uint)ceil(psize * 1.0 / BLOCK_DIMX);

        LAUNCH((tree::Initialization<data>), gridDim, BLOCK_DIMX, stream, col_ass,
               row_cover, col_cover, row_data, col_data);
        SYNC();

        while (true) {
            LAUNCH(tree::S4_init, gridDim, BLOCK_DIMX, stream, vertices_csr1);
            SYNC();

            while (true) {
                CUDA_RUNTIME(cudaMemset(vertex_predicates.predicates, false,
                                        psize * sizeof(bool)));
                CUDA_RUNTIME(
                    cudaMemset(vertex_predicates.addresses, 0, psize * sizeof(long)));

                LAUNCH(vertexPredicateConstructionCSR, gridDim, BLOCK_DIMX, stream,
                       vertex_predicates, vertices_csr1, row_data.is_visited);
                SYNC();

                CUDA_RUNTIME(cub::DeviceReduce::Sum(cub_storage, b3,
                                                    vertex_predicates.addresses,
                                                    &csr2_size, (int)psize));
                CUDA_RUNTIME(cub::DeviceScan::ExclusiveSum(
                    cub_storage, b4, vertex_predicates.addresses,
                    vertex_predicates.addresses, (int)psize));
                SYNC();

                if (csr2_size > 0) {
                    LAUNCH(vertexScatterCSR, gridDim, BLOCK_DIMX, stream, vertices_csr2,
                           vertices_csr1, row_data.is_visited, vertex_predicates);
                    SYNC();
                    LAUNCH((coverAndExpand<data>), gridDim, BLOCK_DIMX, stream,
                           vertices_csr2, csr2_size, slack, row_duals_tree,
                           col_duals_tree, col_ass, row_ass, row_cover, col_cover,
                           row_data, col_data);
                    SYNC();
                } else {
                    break;
                }
            }
            if (goto_5)
                break;

            CUDA_RUNTIME(cudaMemcpyAsync(ws->h_col_slack, col_data.slack,
                                         psize * sizeof(data), cudaMemcpyDeviceToHost,
                                         stream));
            CUDA_RUNTIME(cudaMemcpyAsync(ws->h_col_cover, col_cover,
                                         psize * sizeof(int), cudaMemcpyDeviceToHost,
                                         stream));
            CUDA_RUNTIME(cudaStreamSynchronize(stream));
            double theta = UINT32_MAX;
            for (size_t j = 0; j < psize; j++) {
                if (ws->h_col_cover[j] == 0) {
                    double s = ws->h_col_slack[j];
                    if (s < theta)
                        theta = s;
                }
            }
            theta /= 2;
            LAUNCH((tree::dualUpdate<data>), gridDim, BLOCK_DIMX, stream, theta,
                   row_duals_tree, col_duals_tree, col_data.slack, row_cover, col_cover,
                   col_data.parents, row_data.is_visited);
            SYNC();
        }

        // Reverse traversal: compact REVERSE-marked columns
        Predicates col_predicates;
        col_predicates.size = psize;
        col_predicates.predicates = ws->col_predicates_p;
        col_predicates.addresses = ws->col_predicates_a;
        CUDA_RUNTIME(cudaMemsetAsync(col_predicates.predicates, false,
                                     psize * sizeof(bool), stream));
        CUDA_RUNTIME(
            cudaMemsetAsync(col_predicates.addresses, 0, psize * sizeof(long), stream));

        LAUNCH(augmentPredicateConstruction, gridDim, BLOCK_DIMX, stream,
               col_predicates, col_data.is_visited);
        SYNC();

        CUDA_RUNTIME(cub::DeviceReduce::Sum(cub_storage, b3, col_predicates.addresses,
                                            &col_id_size, (int)psize));
        CUDA_RUNTIME(
            cub::DeviceScan::ExclusiveSum(cub_storage, b4, col_predicates.addresses,
                                          col_predicates.addresses, (int)psize));
        SYNC();
        if (col_id_size > 0) {
            uint local_gridDim = (uint)ceil((col_id_size * 1.0) / BLOCK_DIMX);
            int *col_id_csr = ws->col_id_csr_elements;
            LAUNCH(augmentScatter, gridDim, BLOCK_DIMX, stream, col_id_csr,
                   col_predicates);
            SYNC();
            LAUNCH((reverseTraversal<data>), local_gridDim, BLOCK_DIMX, stream,
                   col_id_csr, row_data, col_data, col_id_size);
            SYNC();
        }

        // Augmentation pass: compact AUGMENT-marked rows
        Predicates row_predicates;
        row_predicates.size = psize;
        row_predicates.predicates = ws->row_predicates_p;
        row_predicates.addresses = ws->row_predicates_a;
        CUDA_RUNTIME(cudaMemsetAsync(row_predicates.predicates, false,
                                     psize * sizeof(bool), stream));
        CUDA_RUNTIME(
            cudaMemsetAsync(row_predicates.addresses, 0, psize * sizeof(long), stream));

        LAUNCH(augmentPredicateConstruction, gridDim, BLOCK_DIMX, stream,
               row_predicates, row_data.is_visited);
        SYNC();
        CUDA_RUNTIME(cub::DeviceReduce::Sum(cub_storage, b3, row_predicates.addresses,
                                            &row_id_size, (int)psize));
        CUDA_RUNTIME(
            cub::DeviceScan::ExclusiveSum(cub_storage, b4, row_predicates.addresses,
                                          row_predicates.addresses, (int)psize));
        SYNC();
        if (row_id_size > 0) {
            uint local_gridDim = (uint)ceil((row_id_size * 1.0) / BLOCK_DIMX);
            int *row_id_csr = ws->row_id_csr_elements;
            LAUNCH(augmentScatter, gridDim, BLOCK_DIMX, stream, row_id_csr,
                   row_predicates);
            SYNC();
            LAUNCH((augment<data>), local_gridDim, BLOCK_DIMX, stream, col_ass, row_ass,
                   row_id_csr, row_data, col_data, row_id_size);
            SYNC();
        }
    }

    bool decision(const std::vector<int> &match_trend) {
        return (match_trend.back() > 1);
    }
};

namespace match_hybrid {

using Tensor = torch::stable::Tensor;
namespace ts = torch::stable;
using ScT = torch::headeronly::ScalarType;

Tensor solve(const Tensor &cost, cudaStream_t stream) {
    const auto rows = cost.size(0);
    const auto cols = cost.size(1);

    auto cost_f = ts::to(cost, ScT::Float);
    const float sentinel = assign_lap::compute_inf_sentinel<float>(cost_f, stream);
    cost_f = assign_lap::rewrite_inf_to_sentinel(cost_f, sentinel, stream);

    auto cost_cm = ts::contiguous(ts::transpose(cost_f, 0, 1));
    const auto K = std::max(rows, cols);
    Tensor cost_square;
    if (rows == cols) {
        cost_square = cost_cm;
    } else {
        cost_square = ts::full({K, K}, static_cast<double>(sentinel), ScT::Float,
                               std::nullopt, cost.device());
        Tensor view = ts::narrow(cost_square, 0, 0, cols);
        view = ts::narrow(view, 1, 0, rows);
        ts::copy_(view, cost_cm);
    }

    const int dev = cost.get_device_index();

    // The hybrid backend uses file-scope __constant__ and __managed__
    // globals (SIZE, NB4, repeat_kernel, goto_5, ...) that prevent
    // concurrent solves from interleaving. Serialize every hybrid
    // solve in this process. Per-N workspace caching still removes
    // the cudaMalloc storm even with this serialization.
    std::lock_guard<std::mutex> hybrid_lock(g_hybrid_mtx);
    auto ws = g_hybrid_cache.acquire(static_cast<std::size_t>(K), dev);

    {
        HLAP<float> solver(cost_square.mutable_data_ptr<float>(), &(*ws), stream);
        solver.solve_on_stream(stream);
    }
    CUDA_RUNTIME(cudaStreamSynchronize(stream));

    CUDA_RUNTIME(cudaMemcpy(ws->h_col_ass, ws->col_ass, K * sizeof(int),
                            cudaMemcpyDeviceToHost));

    return assign_lap::repack_row_to_col(ws->h_col_ass, rows, cols, cost.device());
}

} // namespace match_hybrid
