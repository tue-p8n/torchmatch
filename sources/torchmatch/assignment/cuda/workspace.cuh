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

#include <c10/cuda/CUDACachingAllocator.h>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace assign_lap {

// Allocate device memory through PyTorch's caching allocator. Throws
// c10::OutOfMemoryError on failure. Routing through the CCA matters
// when the rest of the model lives in the caching allocator's pool:
// raw cudaMalloc would fragment the address space against PyTorch's
// own blocks, and the result is invisible to torch.cuda.empty_cache()
// and to the cudagraph private-pool machinery.
inline void *cca_alloc(std::size_t bytes) {
    return c10::cuda::CUDACachingAllocator::raw_alloc(bytes);
}

inline void cca_free(void *ptr) {
    if (ptr)
        c10::cuda::CUDACachingAllocator::raw_delete(ptr);
}

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
