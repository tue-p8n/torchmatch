// Lawler (1976) tree-augmentation Hungarian for the LAP on GPU. Per
// outer iteration, a parallel BFS expands the equality-subgraph tree
// and finds every vertex-disjoint augmenting path at once; reverse
// traversal then augments along them. Trades more per-step work for
// fewer outer iterations and much more exposed parallelism, so it
// wins over Munkres' single-path classical on dense LAPs at large N.
//
// Cost layout: row-major double, padded to square (K, K) where
// K = max(rows, cols). Step machine: 0 reduction, 1 initial assign,
// 2 cover and tree init, 3 cover-and-expand (cooperative kernel), 4
// reverse-traverse and augment, 5 dual update.
//
// State is held by a `LawlerWorkspace` per (K, device), supplied by
// `WorkspaceCache`. The coalesced `ControlBlock` holds host-readable
// flags (currently only `goto_4`); pure device-side iteration scratch
// (`d_theta`, `d_csr_counter`) is pooled through the workspace cache
// so managed-memory pages do not flap host/device on every iteration.

#include <cuda_runtime.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/Exception.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>

#include <cooperative_groups.h>

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

#include "control.cuh"
#include "cuda_common.cuh"
#include "workspace.cuh"

namespace {

// cudaMemsetAsync(col_slack, kSlackInfPattern, ...) takes a byte
// pattern; the value 1000001 expands to a finite double large enough
// for slack initialization while keeping the byte fill defined.
constexpr int kSlackInfPattern = 1000001;
constexpr int kBlockDimX = 16;
constexpr int kBlockDimY = 8;

// Tight epsilon so the double-precision slack reduction does not
// accept almost-zero values as exact zeros. At this precision a false
// positive triggers a wasted outer iteration.
inline __host__ __device__ bool slack_near_zero(double cost) {
    return cost > -assign_lap::EPS_TIGHT && cost < assign_lap::EPS_TIGHT;
}

using assign_lap::ACTIVE;
using assign_lap::AUGMENT;
using assign_lap::DORMANT;
using assign_lap::MODIFIED;
using assign_lap::REVERSE;
using assign_lap::VISITED;

inline void calculateLinearDims(dim3 &blocks_per_grid, dim3 &threads_per_block,
                                int &total_blocks, size_t size) {
    threads_per_block.x = kBlockDimX * kBlockDimY;
    int value = (int)ceil((double)(size) / threads_per_block.x);
    total_blocks = value;
    blocks_per_grid.x = value;
}

struct Array {
    long size;
    int *elements;
};

struct Matrix {
    int rowsize;
    int colsize;
    double *elements;
    double *row_duals;
    double *col_duals;
};

struct Vertices {
    int *row_assignments;
    int *col_assignments;
    int *row_covers;
    int *col_covers;
};

struct CompactEdges {
    int *neighbors;
    long *ptrs;
};

struct Predicates {
    long size;
    bool *predicates;
    long *addresses;
};

struct VertexData {
    int *parents;
    int *children;
    int *is_visited;
    double *slack;
};

// Coalesced managed control block, replacing the standalone managed
// `goto_4` flag. Device-only state (e.g. cb->theta) is intentionally
// NOT placed here: managed-memory pages flap host/device whenever the
// host touches even one field per iteration, which destroys
// throughput. Pure device-side state is pooled through the workspace
// cache instead (see TreeWorkspace::d_theta).
struct ControlBlock {
    int error_code; // assign_lap::LapErrorCode (reserved)
    int goto_4;     // step_3 -> step_4 transition flag (bool-as-int)
};

// Holds every buffer the original solver allocated per call
// (initializeDevice and the per-iteration mallocs in reversePass,
// augmentationPass, compactRowVertices).

struct TreeWorkspace {
    assign_lap::WorkspacePool pool;
    std::size_t N = 0;
    int device = -1;
    bool initialized = false;

    int *row_assignments = nullptr;
    int *col_assignments = nullptr;
    int *row_covers = nullptr;
    int *col_covers = nullptr;

    int *row_is_visited = nullptr;
    int *row_parents = nullptr;
    int *row_children = nullptr;

    int *col_is_visited = nullptr;
    int *col_parents = nullptr;
    int *col_children = nullptr;
    double *col_slack = nullptr;

    double *cost_elements = nullptr;
    double *row_duals = nullptr;
    double *col_duals = nullptr;

    // Compaction predicates (used by step 3 vertex compaction)
    bool *vertex_predicates_p = nullptr;
    long *vertex_predicates_a = nullptr;

    // Step-3 CSR vertex lists (csr1 holds initial row IDs, csr2 holds
    // active subset after compaction; both N-sized)
    int *vertices_csr1_elements = nullptr;
    int *vertices_csr2_elements = nullptr;

    // Step 4 reverse-pass / augmentation-pass per-iteration buffers
    bool *col_predicates_p = nullptr;
    long *col_predicates_a = nullptr;
    int *col_ids_csr_elements = nullptr;

    bool *row_predicates_p = nullptr;
    long *row_predicates_a = nullptr;
    int *row_ids_csr_elements = nullptr;

    // Step-1 initial-assignment locks (column-vs-row)
    int *row_lock = nullptr;
    int *col_lock = nullptr;

    // Host scratch (size N; lives in workspace to avoid per-call new[]).
    // Only the final result is read on host now; the Stage-2 device
    // theta kernel made the per-iteration slack/cover copies obsolete.
    int *h_row_assignments = nullptr;

    // Coalesced managed control block (host-readable flags only)
    ControlBlock *cb = nullptr;

    // Pure device-side scratch for the dual-update step (Stage-2 device
    // theta). Kept off the managed cb page to avoid host↔device flapping.
    double *d_theta = nullptr;

    // Stage-3 device-side scratch: atomic counter used by the cooperative
    // tree_zero_cover_persistent kernel to drive its compaction. Reset
    // inside the kernel each iteration; never read by host.
    int *d_csr_counter = nullptr;

    void allocate(std::size_t n, int dev) {
        N = n;
        device = dev;
        pool.reset(dev);

        row_assignments =
            static_cast<decltype(row_assignments)>(pool.alloc(N * sizeof(int)));
        col_assignments =
            static_cast<decltype(col_assignments)>(pool.alloc(N * sizeof(int)));
        row_covers = static_cast<decltype(row_covers)>(pool.alloc(N * sizeof(int)));
        col_covers = static_cast<decltype(col_covers)>(pool.alloc(N * sizeof(int)));

        row_is_visited =
            static_cast<decltype(row_is_visited)>(pool.alloc(N * sizeof(int)));
        row_parents = static_cast<decltype(row_parents)>(pool.alloc(N * sizeof(int)));
        row_children = static_cast<decltype(row_children)>(pool.alloc(N * sizeof(int)));

        col_is_visited =
            static_cast<decltype(col_is_visited)>(pool.alloc(N * sizeof(int)));
        col_parents = static_cast<decltype(col_parents)>(pool.alloc(N * sizeof(int)));
        col_children = static_cast<decltype(col_children)>(pool.alloc(N * sizeof(int)));
        col_slack = static_cast<decltype(col_slack)>(pool.alloc(N * sizeof(double)));

        cost_elements =
            static_cast<decltype(cost_elements)>(pool.alloc(N * N * sizeof(double)));
        row_duals = static_cast<decltype(row_duals)>(pool.alloc(N * sizeof(double)));
        col_duals = static_cast<decltype(col_duals)>(pool.alloc(N * sizeof(double)));

        vertex_predicates_p =
            static_cast<decltype(vertex_predicates_p)>(pool.alloc(N * sizeof(bool)));
        vertex_predicates_a =
            static_cast<decltype(vertex_predicates_a)>(pool.alloc(N * sizeof(long)));
        vertices_csr1_elements =
            static_cast<decltype(vertices_csr1_elements)>(pool.alloc(N * sizeof(int)));
        vertices_csr2_elements =
            static_cast<decltype(vertices_csr2_elements)>(pool.alloc(N * sizeof(int)));

        col_predicates_p =
            static_cast<decltype(col_predicates_p)>(pool.alloc(N * sizeof(bool)));
        col_predicates_a =
            static_cast<decltype(col_predicates_a)>(pool.alloc(N * sizeof(long)));
        col_ids_csr_elements =
            static_cast<decltype(col_ids_csr_elements)>(pool.alloc(N * sizeof(int)));

        row_predicates_p =
            static_cast<decltype(row_predicates_p)>(pool.alloc(N * sizeof(bool)));
        row_predicates_a =
            static_cast<decltype(row_predicates_a)>(pool.alloc(N * sizeof(long)));
        row_ids_csr_elements =
            static_cast<decltype(row_ids_csr_elements)>(pool.alloc(N * sizeof(int)));

        row_lock = static_cast<decltype(row_lock)>(pool.alloc(N * sizeof(int)));
        col_lock = static_cast<decltype(col_lock)>(pool.alloc(N * sizeof(int)));

        h_row_assignments = new int[N];

        cb = assign_lap::alloc_cb<ControlBlock>(dev);
        if (!cb) {
            throw std::runtime_error("lap_tree: failed to allocate ControlBlock");
        }

        d_theta = static_cast<decltype(d_theta)>(pool.alloc(sizeof(double)));
        d_csr_counter = static_cast<decltype(d_csr_counter)>(pool.alloc(sizeof(int)));

        initialized = true;
    }

    void deallocate() {
        if (!initialized)
            return;

        delete[] h_row_assignments;

        assign_lap::free_cb(cb);
        // All device buffers are pooled: one release frees all of them.
        pool.release();
        initialized = false;
    }
};

static assign_lap::WorkspaceCache<TreeWorkspace> g_lawler_cache;

__global__ void kernel_rowReduction(double *d_costs, double *d_row_duals, size_t N) {
    int rowid = blockIdx.x * blockDim.x + threadIdx.x;
    double min = kSlackInfPattern;
    if (rowid < N) {
        for (int colid = 0; colid < N; colid++) {
            double val = d_costs[rowid * N + colid];
            if (val < min)
                min = val;
        }
        d_row_duals[rowid] = min;
    }
}

__global__ void kernel_columnReduction(double *d_costs, double *d_row_duals,
                                       double *d_col_duals, size_t N) {
    int colid = blockIdx.x * blockDim.x + threadIdx.x;
    double min = kSlackInfPattern;
    if (colid < N) {
        for (int rowid = 0; rowid < N; rowid++) {
            double val = d_costs[rowid * N + colid] - d_row_duals[rowid];
            if (val < min)
                min = val;
        }
        d_col_duals[colid] = min;
    }
}

static void initialReduction(TreeWorkspace &ws, cudaStream_t stream) {
    dim3 blocks_per_grid, threads_per_block;
    int total_blocks = 0;
    calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, ws.N);
    LAUNCH(kernel_rowReduction, blocks_per_grid, threads_per_block, stream,
           ws.cost_elements, ws.row_duals, ws.N);
    LAUNCH(kernel_columnReduction, blocks_per_grid, threads_per_block, stream,
           ws.cost_elements, ws.row_duals, ws.col_duals, ws.N);
}

__global__ void kernel_computeInitialAssignments(
    double *d_costs, double *d_row_duals, double *d_col_duals, int *d_row_assignments,
    int *d_col_assignments, int *d_row_lock, int *d_col_lock, size_t N) {
    int colid = blockIdx.x * blockDim.x + threadIdx.x;
    if (colid < N) {
        for (int rowid = 0; rowid < N; rowid++) {
            if (d_col_lock[colid] == 1)
                break;
            double cost =
                d_costs[rowid * N + colid] - d_row_duals[rowid] - d_col_duals[colid];
            if (slack_near_zero(cost)) {
                if (atomicCAS(&d_row_lock[rowid], 0, 1) == 0) {
                    d_row_assignments[rowid] = colid;
                    d_col_assignments[colid] = rowid;
                    d_col_lock[colid] = 1;
                }
            }
        }
    }
}

static void computeInitialAssignments(TreeWorkspace &ws, cudaStream_t stream) {
    dim3 blocks_per_grid, threads_per_block;
    int total_blocks = 0;
    calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, ws.N);

    CUDA_RUNTIME(cudaMemsetAsync(ws.row_assignments, -1, ws.N * sizeof(int), stream));
    CUDA_RUNTIME(cudaMemsetAsync(ws.col_assignments, -1, ws.N * sizeof(int), stream));
    CUDA_RUNTIME(cudaMemsetAsync(ws.row_lock, 0, ws.N * sizeof(int), stream));
    CUDA_RUNTIME(cudaMemsetAsync(ws.col_lock, 0, ws.N * sizeof(int), stream));

    LAUNCH(kernel_computeInitialAssignments, blocks_per_grid, threads_per_block, stream,
           ws.cost_elements, ws.row_duals, ws.col_duals, ws.row_assignments,
           ws.col_assignments, ws.row_lock, ws.col_lock, ws.N);
}

__global__ void kernel_computeRowCovers(int *d_row_assignments, int *d_row_covers,
                                        int row_count) {
    int rowid = blockIdx.x * blockDim.x + threadIdx.x;
    if (rowid < row_count) {
        if (d_row_assignments[rowid] != -1)
            d_row_covers[rowid] = 1;
    }
}

__global__ void kernel_rowInitialization(int *d_visited, int *d_row_assignments,
                                         int row_start, int row_count) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < row_count) {
        int assignment = d_row_assignments[id + row_start];
        d_visited[id] = (assignment == -1) ? ACTIVE : DORMANT;
    }
}

static void initializeStep2(TreeWorkspace &ws, cudaStream_t stream) {
    int total_blocks = 0;
    dim3 blocks_per_grid, threads_per_block;

    CUDA_RUNTIME(cudaMemsetAsync(ws.row_covers, 0, ws.N * sizeof(int), stream));
    CUDA_RUNTIME(cudaMemsetAsync(ws.col_covers, 0, ws.N * sizeof(int), stream));
    CUDA_RUNTIME(
        cudaMemsetAsync(ws.row_is_visited, DORMANT, ws.N * sizeof(int), stream));
    CUDA_RUNTIME(
        cudaMemsetAsync(ws.col_is_visited, DORMANT, ws.N * sizeof(int), stream));
    CUDA_RUNTIME(
        cudaMemsetAsync(ws.col_slack, kSlackInfPattern, ws.N * sizeof(double), stream));
    CUDA_RUNTIME(cudaMemsetAsync(ws.row_parents, -1, ws.N * sizeof(int), stream));
    CUDA_RUNTIME(cudaMemsetAsync(ws.row_children, -1, ws.N * sizeof(int), stream));
    CUDA_RUNTIME(cudaMemsetAsync(ws.col_parents, -1, ws.N * sizeof(int), stream));
    CUDA_RUNTIME(cudaMemsetAsync(ws.col_children, -1, ws.N * sizeof(int), stream));

    calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, ws.N);
    LAUNCH(kernel_rowInitialization, blocks_per_grid, threads_per_block, stream,
           ws.row_is_visited, ws.row_assignments, 0, (int)ws.N);
}

static int computeRowCovers(TreeWorkspace &ws, cudaStream_t stream) {
    dim3 blocks_per_grid, threads_per_block;
    int total_blocks = 0;
    calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, ws.N);
    LAUNCH(kernel_computeRowCovers, blocks_per_grid, threads_per_block, stream,
           ws.row_assignments, ws.row_covers, (int)ws.N);

    // thrust::reduce over device_ptr is implicitly synchronous and returns
    // the host scalar; same shape as before, just no allocator pressure.
    thrust::device_ptr<int> ptr(ws.row_covers);
    return thrust::reduce(ptr, ptr + ws.N);
}

__global__ void kernel_step3_init(int *d_vertex_ids, int row_count) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < row_count)
        d_vertex_ids[id] = id;
}

__global__ void kernel_vertexPredicateConstructionCSR(Predicates d_vertex_predicates,
                                                      Array d_vertices_csr_in,
                                                      int *d_visited) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    size_t size = d_vertices_csr_in.size;
    int vertexid = (id < size) ? d_vertices_csr_in.elements[id] : -1;
    int visited = (id < size && vertexid != -1) ? d_visited[vertexid] : DORMANT;
    bool predicate = (visited == ACTIVE);
    long addr = predicate ? 1 : 0;
    if (id < size) {
        d_vertex_predicates.predicates[id] = predicate;
        d_vertex_predicates.addresses[id] = addr;
    }
}

__global__ void kernel_vertexScatterCSR(int *d_vertex_ids_csr, int *d_vertex_ids,
                                        int *d_visited,
                                        Predicates d_vertex_predicates) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    size_t size = d_vertex_predicates.size;
    int vertexid = (id < size) ? d_vertex_ids[id] : -1;
    bool predicate = (id < size) ? d_vertex_predicates.predicates[id] : false;
    long compid = predicate ? d_vertex_predicates.addresses[id] : -1;
    if (id < size && predicate) {
        d_vertex_ids_csr[compid] = vertexid;
        d_visited[id] = VISITED;
    }
}

__device__ void __traverse(Matrix d_costs, Vertices d_vertices, int *d_goto_4,
                           int *d_row_parents, int *d_col_parents, int *d_row_visited,
                           int *d_col_visited, double *d_slacks, int *d_start_ptr,
                           int *d_end_ptr, int colid, size_t N) {
    int *ptr1 = d_start_ptr;
    while (ptr1 != d_end_ptr) {
        int rowid = *ptr1;
        double slack = d_costs.elements[rowid * N + colid] - d_costs.row_duals[rowid] -
                       d_costs.col_duals[colid];
        int nxt_rowid = d_vertices.col_assignments[colid];
        if (rowid != nxt_rowid && d_vertices.col_covers[colid] == 0) {
            if (slack < d_slacks[colid]) {
                d_slacks[colid] = slack;
                d_col_parents[colid] = rowid;
            }
            if (slack_near_zero(d_slacks[colid])) {
                if (nxt_rowid != -1) {
                    d_row_parents[nxt_rowid] = colid;
                    d_vertices.row_covers[nxt_rowid] = 0;
                    d_vertices.col_covers[colid] = 1;
                    if (d_row_visited[nxt_rowid] != VISITED)
                        d_row_visited[nxt_rowid] = ACTIVE;
                } else {
                    d_col_visited[colid] = REVERSE;
                    *d_goto_4 = 1;
                }
            }
        }
        d_row_visited[rowid] = VISITED;
        ptr1++;
    }
}

__global__ void kernel_coverAndExpand(int *d_goto_4, Array d_vertices_csr_in,
                                      Matrix d_costs, Vertices d_vertices,
                                      VertexData d_row_data, VertexData d_col_data,
                                      int N) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    int in_size = d_vertices_csr_in.size;
    int *st_ptr = d_vertices_csr_in.elements;
    int *end_ptr = d_vertices_csr_in.elements + in_size;
    if (id < N) {
        __traverse(d_costs, d_vertices, d_goto_4, d_row_data.parents,
                   d_col_data.parents, d_row_data.is_visited, d_col_data.is_visited,
                   d_col_data.slack, st_ptr, end_ptr, id, N);
    }
}

// Matrix/Vertices/VertexData are the original code's kernel-interface
// structs; they remain so kernel signatures stay unchanged.
static Matrix mk_matrix(TreeWorkspace &ws) {
    Matrix m{};
    m.elements = ws.cost_elements;
    m.row_duals = ws.row_duals;
    m.col_duals = ws.col_duals;
    m.rowsize = (int)ws.N;
    m.colsize = (int)ws.N;
    return m;
}

static Vertices mk_vertices(TreeWorkspace &ws) {
    Vertices v{};
    v.row_assignments = ws.row_assignments;
    v.col_assignments = ws.col_assignments;
    v.row_covers = ws.row_covers;
    v.col_covers = ws.col_covers;
    return v;
}

static VertexData mk_row_data(TreeWorkspace &ws) {
    VertexData d{};
    d.is_visited = ws.row_is_visited;
    d.parents = ws.row_parents;
    d.children = ws.row_children;
    d.slack = nullptr;
    return d;
}

static VertexData mk_col_data(TreeWorkspace &ws) {
    VertexData d{};
    d.is_visited = ws.col_is_visited;
    d.parents = ws.col_parents;
    d.children = ws.col_children;
    d.slack = ws.col_slack;
    return d;
}

static Predicates mk_vertex_predicates(TreeWorkspace &ws) {
    Predicates p{};
    p.size = (long)ws.N;
    p.predicates = ws.vertex_predicates_p;
    p.addresses = ws.vertex_predicates_a;
    return p;
}

static void compactRowVertices(TreeWorkspace &ws, Predicates &d_vertex_predicates,
                               Array &d_vertices_csr_out, Array &d_vertices_csr_in,
                               cudaStream_t stream) {
    int total_blocks = 0;
    dim3 blocks_per_grid, threads_per_block;

    CUDA_RUNTIME(cudaMemsetAsync(d_vertex_predicates.predicates, false,
                                 d_vertex_predicates.size * sizeof(bool), stream));
    CUDA_RUNTIME(cudaMemsetAsync(d_vertex_predicates.addresses, 0,
                                 d_vertex_predicates.size * sizeof(long), stream));

    calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks,
                        d_vertices_csr_in.size);
    LAUNCH(kernel_vertexPredicateConstructionCSR, blocks_per_grid, threads_per_block,
           stream, d_vertex_predicates, d_vertices_csr_in, ws.row_is_visited);

    thrust::device_ptr<long> ptr(d_vertex_predicates.addresses);
    d_vertices_csr_out.size = thrust::reduce(ptr, ptr + d_vertex_predicates.size);
    thrust::exclusive_scan(ptr, ptr + d_vertex_predicates.size, ptr);

    if (d_vertices_csr_out.size > 0) {
        // csr2_elements is pre-allocated to N in the workspace.
        d_vertices_csr_out.elements = ws.vertices_csr2_elements;
        LAUNCH(kernel_vertexScatterCSR, blocks_per_grid, threads_per_block, stream,
               d_vertices_csr_out.elements, d_vertices_csr_in.elements,
               ws.row_is_visited, d_vertex_predicates);
    }
}

static void coverZeroAndExpand(TreeWorkspace &ws, Array &d_vertices_csr_in,
                               cudaStream_t stream) {
    int total_blocks = 0;
    dim3 blocks_per_grid, threads_per_block;
    calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, ws.N);
    Matrix m = mk_matrix(ws);
    Vertices v = mk_vertices(ws);
    VertexData rd = mk_row_data(ws);
    VertexData cd = mk_col_data(ws);
    LAUNCH(kernel_coverAndExpand, blocks_per_grid, threads_per_block, stream,
           &ws.cb->goto_4, d_vertices_csr_in, m, v, rd, cd, (int)ws.N);
}

// Stage-3 cooperative kernel: fuses the entire `executeZeroCover` loop
// into a single launch. Uses cooperative_groups grid-wide sync between
// phases (predicate-and-scatter, cover-and-expand) and an atomic
// counter for compaction (replacing cub::DeviceReduce::Sum,
// cub::DeviceScan::ExclusiveSum, and scatter; order does not matter
// because cover-and-expand iterates over every element).

namespace cg = cooperative_groups;

__global__ void tree_zero_cover_persistent(
    const double *__restrict__ cost_elements, const double *__restrict__ row_duals_in,
    const double *__restrict__ col_duals_in, const int *__restrict__ col_assignments_in,
    int *row_covers, int *col_covers, int *row_is_visited, int *col_is_visited,
    int *row_parents, int *col_parents, double *col_slack, int *vertices_csr1_elements,
    int *vertices_csr2_elements, int *csr_counter, int *goto_4_ptr, int N) {
    auto grid = cg::this_grid();
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * gridDim.x;

    // Initialize csr1 := identity. (Replaces kernel_step3_init.)
    for (int i = tid; i < N; i += total_threads) {
        vertices_csr1_elements[i] = i;
    }
    grid.sync();

    while (true) {
        // Phase 1: reset the compaction counter.
        if (tid == 0)
            *csr_counter = 0;
        grid.sync();

        // Phase 2: predicate-and-scatter. Read csr1, compact ACTIVE rows
        // into csr2 via atomicAdd. Equivalent to the original
        // kernel_vertexPredicateConstructionCSR + cub::Sum + cub::ExclusiveSum
        // + kernel_vertexScatterCSR pipeline (csr1 is identity, so vertexid==i).
        for (int i = tid; i < N; i += total_threads) {
            const int vertexid = vertices_csr1_elements[i];
            const int visited = (vertexid != -1) ? row_is_visited[vertexid] : DORMANT;
            if (visited == ACTIVE) {
                const int pos = atomicAdd(csr_counter, 1);
                vertices_csr2_elements[pos] = vertexid;
                // Original kernel_vertexScatterCSR writes is_visited[id] (csr
                // index) which equals vertexid because csr1 is identity. Match
                // that semantic exactly.
                row_is_visited[i] = VISITED;
            }
        }
        grid.sync();

        const int csr2_n = *csr_counter;
        if (csr2_n == 0)
            break;

        // Phase 3: cover-and-expand, per-thread one column. Each thread
        // walks all rows in csr2 and updates cover/visited state. Same
        // race-tolerant write pattern as the original kernel_coverAndExpand.
        for (int colid = tid; colid < N; colid += total_threads) {
            for (int j = 0; j < csr2_n; j++) {
                const int rowid = vertices_csr2_elements[j];
                const double slack_val = cost_elements[rowid * N + colid] -
                                         row_duals_in[rowid] - col_duals_in[colid];
                const int nxt_rowid = col_assignments_in[colid];
                if (rowid != nxt_rowid && col_covers[colid] == 0) {
                    if (slack_val < col_slack[colid]) {
                        col_slack[colid] = slack_val;
                        col_parents[colid] = rowid;
                    }
                    if (slack_near_zero(col_slack[colid])) {
                        if (nxt_rowid != -1) {
                            row_parents[nxt_rowid] = colid;
                            row_covers[nxt_rowid] = 0;
                            col_covers[colid] = 1;
                            if (row_is_visited[nxt_rowid] != VISITED)
                                row_is_visited[nxt_rowid] = ACTIVE;
                        } else {
                            col_is_visited[colid] = REVERSE;
                            *goto_4_ptr = 1;
                        }
                    }
                }
                row_is_visited[rowid] = VISITED;
            }
        }
        grid.sync();
    }
}

static bool launch_zero_cover_persistent(TreeWorkspace &ws, cudaStream_t stream) {
    constexpr int blockDim_chosen = 128;
    int max_blocks_per_sm = 0;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, (const void *)tree_zero_cover_persistent, blockDim_chosen,
        0);
    if (err != cudaSuccess || max_blocks_per_sm <= 0)
        return false;

    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, ws.device);
    const int max_blocks = max_blocks_per_sm * sm_count;

    const int needed_blocks =
        (int)((ws.N + (size_t)blockDim_chosen - 1) / (size_t)blockDim_chosen);
    const int gridDim_chosen = std::min(needed_blocks, max_blocks);
    if (gridDim_chosen < 1 || gridDim_chosen > max_blocks)
        return false;

    // Stage args. Each entry is the address of a local that holds the
    // actual argument value. cudaLaunchCooperativeKernel reads them.
    const double *cost_elements_arg = ws.cost_elements;
    const double *row_duals_arg = ws.row_duals;
    const double *col_duals_arg = ws.col_duals;
    const int *col_ass_arg = ws.col_assignments;
    int *row_covers_arg = ws.row_covers;
    int *col_covers_arg = ws.col_covers;
    int *row_is_visited_arg = ws.row_is_visited;
    int *col_is_visited_arg = ws.col_is_visited;
    int *row_parents_arg = ws.row_parents;
    int *col_parents_arg = ws.col_parents;
    double *col_slack_arg = ws.col_slack;
    int *csr1_arg = ws.vertices_csr1_elements;
    int *csr2_arg = ws.vertices_csr2_elements;
    int *counter_arg = ws.d_csr_counter;
    int *goto_4_arg = &ws.cb->goto_4;
    int N_arg = (int)ws.N;

    void *args[] = {(void *)&cost_elements_arg,  (void *)&row_duals_arg,
                    (void *)&col_duals_arg,      (void *)&col_ass_arg,
                    (void *)&row_covers_arg,     (void *)&col_covers_arg,
                    (void *)&row_is_visited_arg, (void *)&col_is_visited_arg,
                    (void *)&row_parents_arg,    (void *)&col_parents_arg,
                    (void *)&col_slack_arg,      (void *)&csr1_arg,
                    (void *)&csr2_arg,           (void *)&counter_arg,
                    (void *)&goto_4_arg,         (void *)&N_arg};

    err = cudaLaunchCooperativeKernel((const void *)tree_zero_cover_persistent,
                                      dim3(gridDim_chosen), dim3(blockDim_chosen), args,
                                      0, stream);
    if (err != cudaSuccess) {
        // Clear the error so subsequent launches don't see it.
        (void)cudaGetLastError();
        return false;
    }
    return true;
}

static void executeZeroCover(TreeWorkspace &ws, cudaStream_t stream) {
    // Stage-3 fast path: a single cooperative launch fuses the entire
    // `while (csr2_size > 0)` loop, removing roughly three host syncs
    // per iteration. Falls back to the multi-kernel host-driven path
    // when the cooperative launch cannot fit (e.g. unusually large N)
    // or is unsupported.
    if (launch_zero_cover_persistent(ws, stream))
        return;

    // Fallback: original Stage-1 multi-kernel path.
    Array d_vertices_csr1{};
    Array d_vertices_csr2{};
    d_vertices_csr1.size = (int)ws.N;
    d_vertices_csr1.elements = ws.vertices_csr1_elements;
    Predicates vp = mk_vertex_predicates(ws);

    int total_blocks = 0;
    dim3 blocks_per_grid, threads_per_block;
    calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, ws.N);
    LAUNCH(kernel_step3_init, blocks_per_grid, threads_per_block, stream,
           d_vertices_csr1.elements, d_vertices_csr1.size);

    while (true) {
        compactRowVertices(ws, vp, d_vertices_csr2, d_vertices_csr1, stream);
        if (d_vertices_csr2.size == 0)
            break;
        coverZeroAndExpand(ws, d_vertices_csr2, stream);
    }
}

__device__ inline void __augment(int *d_row_assignments, int *d_col_assignments,
                                 int *d_row_children, int *d_col_children,
                                 int init_rowid) {
    int cur_colid = -1, cur_rowid = init_rowid;
    while (cur_rowid != -1) {
        cur_colid = d_row_children[cur_rowid];
        d_row_assignments[cur_rowid] = cur_colid;
        d_col_assignments[cur_colid] = cur_rowid;
        cur_rowid = d_col_children[cur_colid];
    }
}

__device__ inline void __reverse_traversal(int *d_row_visited, int *d_row_children,
                                           int *d_col_children, int *d_row_parents,
                                           int *d_col_parents, int init_colid) {
    int cur_colid = init_colid, cur_rowid = -1;
    while (cur_colid != -1) {
        d_col_children[cur_colid] = cur_rowid;
        cur_rowid = d_col_parents[cur_colid];
        d_row_children[cur_rowid] = cur_colid;
        cur_colid = d_row_parents[cur_rowid];
    }
    d_row_visited[cur_rowid] = AUGMENT;
}

__global__ void kernel_augmentPredicateConstruction(Predicates d_predicates,
                                                    int *d_visited, int offset,
                                                    size_t size) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    int visited = (id < size) ? d_visited[id + offset] : DORMANT;
    bool predicate = (visited == REVERSE || visited == AUGMENT);
    long addr = predicate ? 1 : 0;
    if (id < size) {
        d_predicates.predicates[id] = predicate;
        d_predicates.addresses[id] = addr;
    }
}

__global__ void kernel_augmentScatter(Array d_vertex_ids, Predicates d_predicates,
                                      int offset, size_t size) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    bool predicate = (id < size) ? d_predicates.predicates[id] : false;
    long compid = predicate ? d_predicates.addresses[id] : -1;
    if (id < size && predicate)
        d_vertex_ids.elements[compid] = id + offset;
}

__global__ void kernel_reverseTraversal(Array d_col_vertices, VertexData d_row_data,
                                        VertexData d_col_data) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    int size = d_col_vertices.size;
    int colid = (id < size) ? d_col_vertices.elements[id] : -1;
    if (id < size) {
        __reverse_traversal(d_row_data.is_visited, d_row_data.children,
                            d_col_data.children, d_row_data.parents, d_col_data.parents,
                            colid);
    }
}

__global__ void kernel_augmentation(int *d_row_assignments, int *d_col_assignments,
                                    Array d_row_vertices, VertexData d_row_data,
                                    VertexData d_col_data) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    size_t size = d_row_vertices.size;
    int rowid = (id < size) ? d_row_vertices.elements[id] : -1;
    if (id < size) {
        __augment(d_row_assignments, d_col_assignments, d_row_data.children,
                  d_col_data.children, rowid);
    }
}

static void reversePass(TreeWorkspace &ws, cudaStream_t stream) {
    int total_blocks = 0;
    dim3 blocks_per_grid, threads_per_block;
    calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, ws.N);

    Predicates d_col_predicates{};
    d_col_predicates.size = (long)ws.N;
    d_col_predicates.predicates = ws.col_predicates_p;
    d_col_predicates.addresses = ws.col_predicates_a;

    CUDA_RUNTIME(cudaMemsetAsync(d_col_predicates.predicates, false,
                                 ws.N * sizeof(bool), stream));
    CUDA_RUNTIME(
        cudaMemsetAsync(d_col_predicates.addresses, 0, ws.N * sizeof(long), stream));

    LAUNCH(kernel_augmentPredicateConstruction, blocks_per_grid, threads_per_block,
           stream, d_col_predicates, ws.col_is_visited, 0, ws.N);

    thrust::device_ptr<long> ptr(d_col_predicates.addresses);
    long out_size = thrust::reduce(ptr, ptr + d_col_predicates.size);
    thrust::exclusive_scan(ptr, ptr + d_col_predicates.size, ptr);

    if (out_size > 0) {
        Array d_col_ids_csr{};
        d_col_ids_csr.size = out_size;
        d_col_ids_csr.elements = ws.col_ids_csr_elements;

        int total_blocks_1 = 0;
        dim3 blocks_per_grid_1, threads_per_block_1;
        calculateLinearDims(blocks_per_grid_1, threads_per_block_1, total_blocks_1,
                            d_col_ids_csr.size);
        LAUNCH(kernel_augmentScatter, blocks_per_grid, threads_per_block, stream,
               d_col_ids_csr, d_col_predicates, 0, ws.N);
        VertexData rd = mk_row_data(ws);
        VertexData cd = mk_col_data(ws);
        LAUNCH(kernel_reverseTraversal, blocks_per_grid_1, threads_per_block_1, stream,
               d_col_ids_csr, rd, cd);
    }
}

static void augmentationPass(TreeWorkspace &ws, cudaStream_t stream) {
    int total_blocks = 0;
    dim3 blocks_per_grid, threads_per_block;
    calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, ws.N);

    Predicates d_row_predicates{};
    d_row_predicates.size = (long)ws.N;
    d_row_predicates.predicates = ws.row_predicates_p;
    d_row_predicates.addresses = ws.row_predicates_a;

    CUDA_RUNTIME(cudaMemsetAsync(d_row_predicates.predicates, false,
                                 ws.N * sizeof(bool), stream));
    CUDA_RUNTIME(
        cudaMemsetAsync(d_row_predicates.addresses, 0, ws.N * sizeof(long), stream));

    LAUNCH(kernel_augmentPredicateConstruction, blocks_per_grid, threads_per_block,
           stream, d_row_predicates, ws.row_is_visited, 0, ws.N);

    thrust::device_ptr<long> ptr(d_row_predicates.addresses);
    long out_size = thrust::reduce(ptr, ptr + d_row_predicates.size);
    thrust::exclusive_scan(ptr, ptr + d_row_predicates.size, ptr);

    if (out_size > 0) {
        Array d_row_ids_csr{};
        d_row_ids_csr.size = out_size;
        d_row_ids_csr.elements = ws.row_ids_csr_elements;

        int total_blocks_1 = 0;
        dim3 blocks_per_grid_1, threads_per_block_1;
        calculateLinearDims(blocks_per_grid_1, threads_per_block_1, total_blocks_1,
                            d_row_ids_csr.size);
        LAUNCH(kernel_augmentScatter, blocks_per_grid, threads_per_block, stream,
               d_row_ids_csr, d_row_predicates, 0, ws.N);
        VertexData rd = mk_row_data(ws);
        VertexData cd = mk_col_data(ws);
        LAUNCH(kernel_augmentation, blocks_per_grid_1, threads_per_block_1, stream,
               ws.row_assignments, ws.col_assignments, d_row_ids_csr, rd, cd);
    }
}

// Result lands in *d_theta (device-only scratch); the dual-update
// kernel reads it directly, so no host readback or memcpy hits the
// inner loop.
//
// A single block (256 threads) covers any N via grid-stride: for
// typical problem sizes (N <= 256 in this benchmark suite) the
// reduction is one warp's work; larger N iterates the grid-stride
// loop more times.
__global__ void kernel_compute_theta(const double *d_col_slacks,
                                     const int *d_col_covers, double *d_theta,
                                     size_t N) {
    constexpr int BLOCK = 256;
    using BR = cub::BlockReduce<double, BLOCK>;
    __shared__ typename BR::TempStorage temp_storage;

    double thread_min = 1e30;
    for (size_t i = threadIdx.x; i < N; i += BLOCK) {
        if (d_col_covers[i] == 0) {
            double s = d_col_slacks[i];
            if (s < thread_min)
                thread_min = s;
        }
    }
    double block_min = BR(temp_storage).Reduce(thread_min, cub::Min());
    if (threadIdx.x == 0) {
        *d_theta = block_min / 2.0;
    }
}

__global__ void kernel_dualUpdate_2(const double *d_theta, double *d_row_duals,
                                    double *d_col_duals, double *d_col_slacks,
                                    int *d_row_cover, int *d_col_cover,
                                    int *d_col_parents, int *d_row_visited,
                                    int row_start, int row_count, size_t N) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    const double d_min_val = *d_theta;
    int row_cover = (id < row_count) ? d_row_cover[id + row_start] : -1;
    int col_cover = (id < N) ? d_col_cover[id] : -1;
    if (id < N) {
        if (row_cover == 0)
            d_row_duals[id] += d_min_val;
        else
            d_row_duals[id] -= d_min_val;
        if (col_cover == 1) {
            d_col_duals[id] -= d_min_val;
        } else {
            d_col_duals[id] += d_min_val;
            d_col_slacks[id] -= (2 * d_min_val);
            if (slack_near_zero(d_col_slacks[id])) {
                int par_rowid = d_col_parents[id];
                d_row_visited[par_rowid] = ACTIVE;
            }
        }
    }
}

static void computeTheta(TreeWorkspace &ws, cudaStream_t stream) {
    dim3 blocks_per_grid, threads_per_block;
    int total_blocks;
    calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, ws.N);

    // Stage-2 device-side theta: single-block min reduce + dualUpdate
    // launches in sequence on the user's stream. No host readback, no
    // memcpy: cb->theta flows from one kernel to the next.
    LAUNCH(kernel_compute_theta, 1, 256, stream, ws.col_slack, ws.col_covers,
           ws.d_theta, ws.N);
    LAUNCH(kernel_dualUpdate_2, blocks_per_grid, threads_per_block, stream, ws.d_theta,
           ws.row_duals, ws.col_duals, ws.col_slack, ws.row_covers, ws.col_covers,
           ws.col_parents, ws.row_is_visited, 0, (int)ws.N, ws.N);
}

static void solve_with_workspace(TreeWorkspace &ws, const double *cost_host_or_dev,
                                 cudaStream_t stream) {
    CUDA_RUNTIME(cudaMemcpyAsync(ws.cost_elements, cost_host_or_dev,
                                 ws.N * ws.N * sizeof(double), cudaMemcpyDefault,
                                 stream));
    CUDA_RUNTIME(cudaMemsetAsync(ws.row_duals, 0, ws.N * sizeof(double), stream));
    CUDA_RUNTIME(cudaMemsetAsync(ws.col_duals, 0, ws.N * sizeof(double), stream));

    CUDA_RUNTIME(cudaMemsetAsync(ws.cb, 0, sizeof(ControlBlock), stream));

    int step = 0;
    bool done = false;
    while (!done) {
        switch (step) {
        case 0:
            initialReduction(ws, stream);
            step = 1;
            break;
        case 1:
            computeInitialAssignments(ws, stream);
            step = 2;
            break;
        case 2: {
            initializeStep2(ws, stream);
            int cover_count = computeRowCovers(ws, stream); // implicit sync (thrust)
            int next = (cover_count == (int)ws.N) ? 6 : 3;
            step = next;
            break;
        }
        case 3:
            ws.cb->goto_4 = 0;
            executeZeroCover(ws, stream);
            CUDA_RUNTIME(cudaStreamSynchronize(stream));
            step = (ws.cb->goto_4 != 0) ? 4 : 5;
            break;
        case 4:
            reversePass(ws, stream);
            augmentationPass(ws, stream);
            step = 2;
            break;
        case 5:
            computeTheta(ws, stream);
            step = 3;
            break;
        case 6:
            done = true;
            break;
        }
    }

    CUDA_RUNTIME(cudaMemcpyAsync(ws.h_row_assignments, ws.row_assignments,
                                 ws.N * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_RUNTIME(cudaStreamSynchronize(stream));
}

} // anonymous namespace

namespace match_lawler {

using Tensor = torch::stable::Tensor;
namespace ts = torch::stable;
using ScT = torch::headeronly::ScalarType;

Tensor solve(const Tensor &cost, cudaStream_t stream) {
    const auto rows = cost.size(0);
    const auto cols = cost.size(1);

    auto cost_d = ts::to(cost, ScT::Double);
    const double sentinel = assign_lap::compute_inf_sentinel<double>(cost_d, stream);
    cost_d = assign_lap::rewrite_inf_to_sentinel(cost_d, sentinel, stream);

    const auto K = std::max(rows, cols);
    Tensor cost_square;
    if (rows == cols) {
        cost_square = ts::contiguous(cost_d);
    } else {
        cost_square =
            ts::full({K, K}, sentinel, ScT::Double, std::nullopt, cost.device());
        Tensor view = ts::narrow(cost_square, 0, 0, rows);
        view = ts::narrow(view, 1, 0, cols);
        ts::copy_(view, cost_d);
    }

    const int dev = cost.get_device_index();
    auto ws = g_lawler_cache.acquire(static_cast<std::size_t>(K), dev);

    solve_with_workspace(*ws, cost_square.const_data_ptr<double>(), stream);

    return assign_lap::repack_row_to_col(ws->h_row_assignments, rows, cols,
                                         cost.device());
}

} // namespace match_lawler
