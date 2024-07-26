// Batched LAP solver: one CUDA block per problem, entirely in shared memory.
//
// Runs the successive-shortest-path (Jonker-Volgenant) algorithm. Each
// block loads one NxN cost matrix into shared memory and runs the full
// JV solve with no global-memory traffic and no inter-kernel syncs.
//
// Template parameter TILE_N sets the maximum problem size. The actual
// size N (≤ TILE_N) arrives at runtime. Columns and rows beyond N are
// ignored.

#pragma once
#include <cassert>
#include <cstdint>
#include <cuda_runtime.h>
#include <limits>

namespace match_batch {

// Pad shared-memory rows by 1 to avoid bank conflicts (stride = TILE_N+1
// ensures consecutive threads hit different banks).
template <int TILE_N>
struct SharedMem {
    static constexpr int STRIDE = TILE_N + 1;

    float cost[TILE_N * STRIDE];
    float u[TILE_N];
    float v[TILE_N];
    float shortest[TILE_N];
    int path[TILE_N];
    int col4row[TILE_N];
    int row4col[TILE_N];
    uint8_t SR[TILE_N];
    uint8_t SC[TILE_N];

    // Scalars shared across the block.
    float minVal;
    int sink;
    int curI;
};

template <int TILE_N, int BLOCK_DIM>
__global__ void lap_batch_kernel(const float *__restrict__ costs, // (B, N, N) row-major
                                 int64_t *__restrict__ col4row_out, // (B, N)
                                 int B,
                                 int N) // actual square size (≤ TILE_N)
{
    const int bid = blockIdx.x;
    if (bid >= B)
        return;
    const int tid = threadIdx.x;

    constexpr int S = SharedMem<TILE_N>::STRIDE;
    constexpr float INF = 1e30f;

    __shared__ SharedMem<TILE_N> sm;

    // ---- Load cost matrix (row-major) into padded shared memory ----
    // The on-device tensor is (B, N, N); the kernel reads it at its natural
    // stride and copies row r into shared-memory offset r*S so the inner JV
    // loop can use the bank-conflict-padded stride S. Cells with r>=N or
    // c>=N are never accessed (the JV body guards with `tid < N` and the
    // min-search runs `j < N`), so leaving them uninitialized is safe and
    // avoids reading past the input buffer.
    {
        const float *src = costs + (size_t)bid * N * N;
        for (int i = tid; i < N * N; i += BLOCK_DIM) {
            int r = i / N;
            int c = i % N;
            sm.cost[r * S + c] = src[i];
        }
    }

    // ---- Initialize assignments and duals ----
    for (int i = tid; i < TILE_N; i += BLOCK_DIM) {
        sm.u[i] = 0.0f;
        sm.v[i] = 0.0f;
        sm.col4row[i] = -1;
        sm.row4col[i] = -1;
    }
    __syncthreads();

    // ---- JV: one augmenting path per row ----
    for (int curRow = 0; curRow < N; curRow++) {

        // Init shortest-path search.
        for (int j = tid; j < TILE_N; j += BLOCK_DIM) {
            sm.shortest[j] = INF;
            sm.SR[j] = 0;
            sm.SC[j] = 0;
            sm.path[j] = -1;
        }
        if (tid == 0) {
            sm.minVal = 0.0f;
            sm.sink = -1;
            sm.curI = curRow;
        }
        __syncthreads();

        // Dijkstra-like search for shortest augmenting path.
        while (sm.sink == -1) {
            const int i = sm.curI;

            if (tid == 0)
                sm.SR[i] = 1;
            __syncthreads();

            // Each thread updates one column's shortest-path cost.
            if (tid < N && !sm.SC[tid]) {
                float r = sm.minVal - sm.u[i] + sm.cost[i * S + tid] - sm.v[tid];
                if (r < sm.shortest[tid]) {
                    sm.shortest[tid] = r;
                    sm.path[tid] = i;
                }
            }
            __syncthreads();

            // Thread 0: find min unscanned column, advance search.
            if (tid == 0) {
                float lowest = INF;
                int idx = -1;
                for (int j = 0; j < N; j++) {
                    if (!sm.SC[j]) {
                        if (sm.shortest[j] < lowest ||
                            (sm.shortest[j] == lowest && sm.row4col[j] == -1)) {
                            lowest = sm.shortest[j];
                            idx = j;
                        }
                    }
                }

                if (idx < 0 || lowest >= INF) {
                    sm.sink = -2; // infeasible
                } else {
                    sm.minVal = lowest;
                    sm.SC[idx] = 1;
                    if (sm.row4col[idx] == -1) {
                        sm.sink = idx;
                    } else {
                        sm.curI = sm.row4col[idx];
                    }
                }
            }
            __syncthreads();
        }

        if (sm.sink < 0)
            continue; // infeasible row

        // ---- Update dual variables ----
        const float mv = sm.minVal;
        if (tid == 0)
            sm.u[curRow] += mv;
        if (tid < N && sm.SR[tid] && tid != curRow) {
            sm.u[tid] += mv - sm.shortest[sm.col4row[tid]];
        }
        if (tid < N && sm.SC[tid]) {
            sm.v[tid] -= mv - sm.shortest[tid];
        }
        __syncthreads();

        // ---- Augment matching along the path (thread 0) ----
        if (tid == 0) {
            int j = sm.sink;
            int ii;
            do {
                ii = sm.path[j];
                sm.row4col[j] = ii;
                int prev_j = sm.col4row[ii];
                sm.col4row[ii] = j;
                j = prev_j;
            } while (ii != curRow);
        }
        __syncthreads();
    }

    // ---- Write output ----
    // Output tensor is (B, N); write the per-batch slice at its natural
    // stride. Tile padding (TILE_N - N entries) is internal to the kernel
    // and never reaches the caller.
    for (int i = tid; i < N; i += BLOCK_DIM) {
        col4row_out[(size_t)bid * N + i] = static_cast<int64_t>(sm.col4row[i]);
    }
}

// Maximum tile size supported (limited by shared memory).
constexpr int MAX_TILE = 64;

inline int select_tile(int N) {
    if (N <= 32)
        return 32;
    if (N <= 64)
        return 64;
    return -1; // too large for tiled solver
}

// Launch the batched kernel for a specific tile size.
//
// Contract: d_costs is a contiguous (B, N, N) row-major float buffer and
// d_out is a contiguous (B, N) int64 buffer. The kernel addresses both at
// stride (N*N, N, 1) and (N, 1); a non-contiguous tensor will silently
// produce wrong indices. Callers should TORCH_CHECK contiguity upstream;
// the asserts below guard the kernel-internal invariants in debug builds.
inline void launch(int tile, int B, int N, const float *d_costs, int64_t *d_out,
                   cudaStream_t stream) {
    assert(N > 0 && N <= MAX_TILE);
    assert(N <= tile);
    assert(B >= 0);
    switch (tile) {
    case 32:
        lap_batch_kernel<32, 32><<<B, 32, 0, stream>>>(d_costs, d_out, B, N);
        break;
    case 64:
        lap_batch_kernel<64, 64><<<B, 64, 0, stream>>>(d_costs, d_out, B, N);
        break;
    default:
        break; // caller handles fallback
    }
}

} // namespace match_batch
