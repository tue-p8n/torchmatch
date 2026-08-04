// Process-level workspace cache for the torch-assign LAP backends.
//
// Replaces the per-call `cudaMalloc` / `cudaFree` storm in each
// backend with a static `(N, device) -> Workspace` map. The first
// call at a given (N, device) lazily constructs the workspace;
// later calls reuse it.
//
// Each backend supplies a `Layout` struct with:
//   - void allocate(size_t N, int device)
//   - void deallocate()
// The cache constructs Layouts in-place and never copies them.
//
// Concurrency: the cache itself is mutex-guarded for lookup and
// insert. Each workspace slot owns its own mutex; `acquire(N, device)`
// returns a `Lease` RAII object that holds the slot lock for its
// lifetime, so a concurrent caller at the same (N, device) blocks
// until the in-flight solve releases. The cache lock is released
// before the slot lock is taken, so concurrent solves at *different*
// (N, device) keys proceed without contention.

#pragma once

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/shim_utils.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace assign_lap {

// Device-memory pool backing one workspace layout.
//
// Every block is a 1-D uint8 tensor created through the AOTI C shim,
// whose storage comes from the same allocator ATen hands every other
// CUDA tensor: PyTorch's CUDA caching allocator. Staying inside that
// pool matters when the rest of the model lives there too — a raw
// cudaMalloc would fragment the address space against PyTorch's own
// blocks, would be invisible to torch.cuda.memory_allocated() /
// torch.cuda.empty_cache(), and would sit outside the cudagraph
// private-pool machinery (moot while the three solvers reject graph
// capture outright, but it keeps the option open). The shim is stable
// ABI, so this
// keeps the whole extension compilable with -DTORCH_TARGET_VERSION,
// which a direct c10::cuda::CUDACachingAllocator call would not (its
// header hard #errors under that macro).
//
// Blocks are owned for the lifetime of the pool and released together:
// the workspace layouts allocate everything up front in allocate() and
// free everything in deallocate(), so per-pointer frees are not needed.
// Allocations are at least 512-byte aligned (the caching allocator's
// granularity), so any element type the solvers cast to is aligned.
class WorkspacePool {
  public:
    WorkspacePool() = default;
    WorkspacePool(const WorkspacePool &) = delete;
    WorkspacePool &operator=(const WorkspacePool &) = delete;

    // Drop any previously held blocks and target `device` for
    // subsequent allocations.
    void reset(int device) {
        blocks_.clear();
        device_ = device;
    }

    // Allocate `bytes` of device memory. Throws on failure; the pool
    // owns the result, so callers must not free the returned pointer.
    void *alloc(std::size_t bytes) {
        const int64_t size = static_cast<int64_t>(bytes == 0 ? 1 : bytes);
        const int64_t stride = 1;
        AtenTensorHandle handle = nullptr;
        TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
            /*ndim=*/1, &size, &stride, aoti_torch_dtype_uint8(),
            aoti_torch_device_type_cuda(), device_, &handle));
        blocks_.emplace_back(handle);
        return blocks_.back().data_ptr();
    }

    void release() { blocks_.clear(); }

  private:
    std::vector<torch::stable::Tensor> blocks_;
    int device_ = 0;
};

template <typename Layout>
class WorkspaceCache {
    using Key = std::pair<std::size_t, int>;

    struct KeyHash {
        std::size_t operator()(const Key &k) const noexcept {
            return std::hash<std::size_t>{}(k.first) ^
                   (std::hash<int>{}(k.second) << 1);
        }
    };

    struct Slot {
        Layout layout;
        std::mutex mtx;
    };

    std::unordered_map<Key, std::unique_ptr<Slot>, KeyHash> cache_;
    std::mutex cache_mtx_;

  public:
    class Lease {
        Slot *slot_ = nullptr;
        std::unique_lock<std::mutex> lock_;

      public:
        Lease() = default;
        Lease(Slot *s) : slot_(s), lock_(s->mtx) {}
        Lease(Lease &&) = default;
        Lease &operator=(Lease &&) = default;
        Layout *operator->() noexcept { return &slot_->layout; }
        Layout &operator*() noexcept { return slot_->layout; }
        const Layout *operator->() const noexcept { return &slot_->layout; }
        const Layout &operator*() const noexcept { return slot_->layout; }
    };

    Lease acquire(std::size_t N, int device) {
        Slot *slot;
        {
            std::lock_guard<std::mutex> lk(cache_mtx_);
            Key k{N, device};
            auto it = cache_.find(k);
            if (it == cache_.end()) {
                auto pslot = std::make_unique<Slot>();
                pslot->layout.allocate(N, device);
                slot = pslot.get();
                cache_.emplace(k, std::move(pslot));
            } else {
                slot = it->second.get();
            }
        }
        return Lease(slot);
    }

    ~WorkspaceCache() {
        std::lock_guard<std::mutex> lk(cache_mtx_);
        for (auto &kv : cache_) {
            kv.second->layout.deallocate();
        }
    }
};

} // namespace assign_lap
