/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Copyright 2024 The MeepoEmbedding Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#ifndef MEEPO_EMBEDDING_COMMON_BFC_ALLOCATOR_H_
#define MEEPO_EMBEDDING_COMMON_BFC_ALLOCATOR_H_

#include <algorithm>
#include <bit>
#include <deque>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_set>

#include "meepo_embedding/include/common/logger.h"
#include "meepo_embedding/include/common/macros.h"
#include "meepo_embedding/include/common/mutex.h"
#include "meepo_embedding/include/common/strings.h"
#include "meepo_embedding/include/framework/memory_resource.hpp"

namespace meepo_embedding {
namespace memory {

// Return floor(log2(n)) for positive integer n.  Returns -1 iff n == 0.
inline int Log2Floor64(std::uint64_t n) { return std::bit_width(n) - 1; }

// A retrying wrapper for a memory allocator.
class AllocatorRetry {
 public:
  AllocatorRetry(){};

  // Call 'alloc_func' to obtain memory.  On first call,
  // 'verbose_failure' will be false.  If return value is nullptr,
  // then wait up to 'max_millis_to_wait' milliseconds, retrying each
  // time a call to DeallocateRaw() is detected, until either a good
  // pointer is returned or the deadline is exhausted.  If the
  // deadline is exhausted, try one more time with 'verbose_failure'
  // set to true.  The value returned is either the first good pointer
  // obtained from 'alloc_func' or nullptr.
  void* AllocateRaw(std::function<void*(size_t alignment, size_t num_bytes,
                                        bool verbose_failure)>
                      alloc_func,
                    int max_millis_to_wait, size_t alignment, size_t bytes);

  // Called to notify clients that some memory was returned.
  void NotifyDealloc();

 private:
  std::mutex mu_;
  std::condition_variable memory_returned_;
};

// Implementation details below
inline void AllocatorRetry::NotifyDealloc() {
  std::lock_guard<std::mutex> lock(mu_);
  memory_returned_.notify_all();
}

class BFCAllocator : public std::pmr::memory_resource {
 public:
  struct Options {
    bool allow_growth = true;

    // If true, the allocator may sleep for a period of time when it can't
    // fulfill an allocation request, in the hopes that another thread will free
    // up memory in the meantime.
    //
    // If false, the allocator will never sleep, even if
    // AllocationAttributes::attr_retry_on_failure is true.
    bool allow_retry_on_failure = true;

    // Whether the allocator will deallocate free regions to avoid OOM due to
    // memory fragmentation.
    bool garbage_collection = false;

    // Controls when a chunk should be split, if its size exceeds the requested
    // allocation size.
    double fragmentation_fraction = 0;
  };
  explicit BFCAllocator(pro::proxy<MemoryResource> sub_allocator,
                        size_t total_memory, const std::string& name,
                        const Options& opts, quill::Logger* logger = nullptr);
  void* do_allocate(const std::size_t bytes,
                    const std::size_t alignment) override {
    AllocateRaw(alignment, bytes);
  }
  void do_deallocate(void* ptr, const std::size_t /* bytes */,
                     const std::size_t /* alignment */) override {
    DeallocateRaw(ptr);
  }
  bool do_is_equal(
    const std::pmr::memory_resource& other) const noexcept override {
    bool eq = (this
               == dynamic_cast<BFCAllocator*>(
                 const_cast<std::pmr::memory_resource*>(&other)));
    if (eq) {
      return true;
    }
    return false;
  }

  std::string Name() const { return name_; }

  void* AllocateRaw(size_t alignment, size_t num_bytes);

  void DeallocateRaw(void* ptr);

  bool TracksAllocationSizes() const;

  size_t RequestedSize(const void* ptr) const;

  size_t AllocatedSize(const void* ptr) const;

  int64_t AllocationId(const void* ptr) const;

  std::optional<AllocatorStats> GetStats();

  bool ClearStats();

  void SetTimingCounter(mutex::shared_counter* sc) { timing_counter_ = sc; }

  void SetSafeFrontier(std::uint64_t count);

  MemoryResourceType GetMemoryType() const;

  bool ShouldRecordStackTrace() const {
#ifdef MeepoEmbedding_MEM_DEBUG
    return true;
#else
    return false;
#endif
  }

 private:
  // Additional private members and methods for BFCAllocator implementation
  struct Bin;

  void* AllocateRawInternal(size_t alignment, size_t num_bytes,
                            bool dump_log_on_failure,
                            std::uint64_t freed_before_count);

  void* AllocateRawInternalWithRetry(size_t alignment, size_t num_bytes);

  void DeallocateRawInternal(void* ptr);

  // Chunks whose freed_at_count is later than the safe frontier value are kept
  // on a special list and not subject to merging immediately upon being freed.
  //
  // This function sweeps that list looking for Chunks whose timestamp is now
  // safe. When found their freed_at_count is set to 0 and we attempt to merge
  // them with their neighbors.
  //
  // If required_bytes > 0 then this function is being called in the context of
  // a need for this many bytes that could not be satisfied without merging
  // unsafe chunks, so we go ahead and merge the unsafe chunks too, just up to
  // the point that a free chunk of required_bytes is produced.  Note that
  // unsafe merged chunks adopt the most conservative timestamp from their
  // constituents so they're only useful for allocations not requiring a
  // particular timestamp.
  bool MergeTimestampedChunks(size_t required_bytes)
    ME_EXCLUSIVE_LOCKS_REQUIRED(lock_);

  // Return the largest free chunk bytes from the largest bin in constant time.
  // The free chunks are sorted by size (and then address) in a bin.
  int64_t LargestFreeChunk() ME_EXCLUSIVE_LOCKS_REQUIRED(lock_);

  // A ChunkHandle is an index into the chunks_ vector in BFCAllocator
  // kInvalidChunkHandle means an invalid chunk
  typedef size_t ChunkHandle;
  static constexpr ChunkHandle kInvalidChunkHandle = SIZE_MAX;

  typedef int BinNum;
  static constexpr int kInvalidBinNum = -1;
  // The following means that the largest bin'd chunk size is 256 << 21 = 512MB.
  static constexpr int kNumBins = 21;

  // A Chunk points to a piece of memory that's either entirely free or entirely
  // in use by one user memory allocation.
  //
  // An AllocationRegion's memory is split up into one or more disjoint Chunks,
  // which together cover the whole region without gaps.  Chunks participate in
  // a doubly-linked list, and the prev/next pointers point to the physically
  // adjacent chunks.
  //
  // Since a chunk cannot be partially in use, we may need to split a free chunk
  // in order to service a user allocation.  We always merge adjacent free
  // chunks.
  //
  // Chunks contain information about whether they are in use or whether they
  // are free, and contain a pointer to the bin they are in.
  struct Chunk {
    size_t size = 0;  // Full size of buffer.

    // We sometimes give chunks that are larger than needed to reduce
    // fragmentation.  requested_size keeps track of what the client
    // actually wanted so we can understand whether our splitting
    // strategy is efficient.
    size_t requested_size = 0;

    // allocation_id is set to -1 when the chunk is not in use. It is assigned a
    // value greater than zero before the chunk is returned from
    // AllocateRaw, and this value is unique among values assigned by
    // the parent allocator.
    int64_t allocation_id = -1;
    void* ptr = nullptr;  // pointer to granted subbuffer.

    // If not kInvalidChunkHandle, the memory referred to by 'prev' is directly
    // preceding the memory used by this chunk.  E.g., It should start
    // at 'ptr - prev->size'
    ChunkHandle prev = kInvalidChunkHandle;

    // If not kInvalidChunkHandle, the memory referred to by 'next' is directly
    // following the memory used by this chunk.  E.g., It should be at
    // 'ptr + size'
    ChunkHandle next = kInvalidChunkHandle;

    // What bin are we in?
    BinNum bin_num = kInvalidBinNum;

    // Optional count when this chunk was most recently made free.
    std::uint64_t freed_at_count = 0;

    bool in_use() const { return allocation_id != -1; }

#ifdef ME_MEM_DEBUG
    // optional debugging info
    const char* op_name = nullptr;
    std::uint64_t step_id = 0;
    int64 action_count = 0;
#endif

    std::string DebugString(BFCAllocator* a,
                            bool recurse) ME_NO_THREAD_SAFETY_ANALYSIS {
      std::string dbg;
      dbg.reserve(64);
      auto a_info = strings::StrFormat(
        "  Size: {}  | Requested Size: {} | in_use: {}  | bin_num: {}",
        strings::HumanReadableNumBytes(size),
        strings::HumanReadableNumBytes(requested_size), in_use(), bin_num);
      dbg.append(a_info);
      if (recurse && prev != BFCAllocator::kInvalidChunkHandle) {
        Chunk* p = a->ChunkFromHandle(prev);
        auto ia_info =
          strings::StrFormat(", prev: {}", p->DebugString(a, false));
        dbg.append(ia_info);
      }
      if (recurse && next != BFCAllocator::kInvalidChunkHandle) {
        Chunk* n = a->ChunkFromHandle(next);
        auto ia_info =
          strings::StrFormat(", next: {}", n->DebugString(a, false));
        dbg.append(ia_info);
      }
#ifdef ME_MEM_DEBUG
      auto d_info = strings::StrFormat(", for: {}, stepid: {}, last_action: {}",
                                       op_name ? op_name : "UNKNOWN", step_id,
                                       action_count);
      dbg.append(d_info);
#endif
      return dbg;
    }
  };

  // A Bin is a collection of similar-sized free chunks.
  // Allocated chunks are never in a Bin.
  struct Bin {
    // All chunks in this bin have >= bin_size memory.
    size_t bin_size = 0;

    class ChunkComparator {
     public:
      explicit ChunkComparator(BFCAllocator* allocator)
        : allocator_(allocator) {}
      // Sort first by size and then use pointer address as a tie breaker.
      bool operator()(const ChunkHandle ha,
                      const ChunkHandle hb) const ME_NO_THREAD_SAFETY_ANALYSIS {
        const Chunk* a = allocator_->ChunkFromHandle(ha);
        const Chunk* b = allocator_->ChunkFromHandle(hb);
        if (a->size != b->size) {
          return a->size < b->size;
        }
        return a->ptr < b->ptr;
      }

     private:
      BFCAllocator* allocator_;  // The parent allocator
    };

    typedef std::set<ChunkHandle, ChunkComparator> FreeChunkSet;
    // List of free chunks within the bin, sorted by chunk size.
    // Chunk * not owned.
    FreeChunkSet free_chunks;
    Bin(BFCAllocator* allocator, size_t bs)
      : bin_size(bs), free_chunks(ChunkComparator(allocator)) {}
  };

  static constexpr size_t kMinAllocationBits = 8;
  static constexpr size_t kMinAllocationSize = 1 << kMinAllocationBits;

  // BFCAllocator allocates memory into a collection of disjoint
  // AllocationRegions.  Each AllocationRegion corresponds to one call to
  // SubAllocator::Alloc().  (Actually, if a subsequent call to
  // SubAllocator::Alloc() returns another region immediately adjacent to the
  // last, it will be used to extend the first AllocationRegion, not create a
  // separate one.)
  //
  // An AllocationRegion contains one or more Chunks, covering all of its
  // memory.  Its primary job is to map pointers to ChunkHandles.
  //
  // This class is thread-compatible.
  class AllocationRegion {
   public:
    AllocationRegion(void* ptr, size_t memory_size,
                     size_t alignment = default_alignment)
      : ptr_(ptr),
        memory_size_(memory_size),
        alignment_(alignment),
        end_ptr_(static_cast<void*>(static_cast<char*>(ptr_) + memory_size_)) {
      assert(0 == memory_size % kMinAllocationSize);
      const size_t n_handles =
        (memory_size + kMinAllocationSize - 1) / kMinAllocationSize;
      handles_.resize(n_handles, kInvalidChunkHandle);
    }

    AllocationRegion() = default;
    AllocationRegion(AllocationRegion&& other) { Swap(&other); }
    AllocationRegion& operator=(AllocationRegion&& other) {
      Swap(&other);
      return *this;
    }

    void* ptr() const { return ptr_; }
    void* end_ptr() const { return end_ptr_; }
    size_t memory_size() const { return memory_size_; }
    size_t alignment() const { return alignment_; }
    void extend(size_t size) {
      memory_size_ += size;
      assert(0UL == (memory_size_ % kMinAllocationSize));

      end_ptr_ = static_cast<void*>(static_cast<char*>(end_ptr_) + size);
      const size_t n_handles =
        (memory_size_ + kMinAllocationSize - 1) / kMinAllocationSize;
      handles_.resize(n_handles, kInvalidChunkHandle);
    }
    ChunkHandle get_handle(const void* p) const {
      return handles_[IndexFor(p)];
    }
    void set_handle(const void* p, ChunkHandle h) { handles_[IndexFor(p)] = h; }
    void erase(const void* p) { set_handle(p, kInvalidChunkHandle); }

   private:
    void Swap(AllocationRegion* other) {
      std::swap(ptr_, other->ptr_);
      std::swap(memory_size_, other->memory_size_);
      std::swap(alignment_, other->alignment_);
      std::swap(end_ptr_, other->end_ptr_);
      std::swap(handles_, other->handles_);
    }

    size_t IndexFor(const void* p) const {
      std::uintptr_t p_int = reinterpret_cast<std::uintptr_t>(p);
      std::uintptr_t base_int = reinterpret_cast<std::uintptr_t>(ptr_);
      assert(p_int >= base_int);
      assert(p_int < base_int + memory_size_);
      return static_cast<size_t>(((p_int - base_int) >> kMinAllocationBits));
    }

    // Metadata about the allocation region.
    void* ptr_ = nullptr;
    size_t memory_size_ = 0;
    size_t alignment_ = default_alignment;
    void* end_ptr_ = nullptr;

    // Array of size "memory_size / kMinAllocationSize".  It is
    // indexed by (p-base) / kMinAllocationSize, contains ChunkHandle
    // for the memory allocation represented by "p"
    std::vector<ChunkHandle> handles_;

    AllocationRegion(const AllocationRegion&) = delete;
    void operator=(const AllocationRegion&) = delete;
  };

  // RegionManager aggregates one or more "AllocationRegions" and provides
  // a layer of indirection from pointers to the underlying ChunkHandle,
  // allowing allocation across multiple discontiguous memory regions.
  //
  // This class is thread-compatible.
  class RegionManager {
   public:
    RegionManager() {}
    ~RegionManager() {}

    void AddAllocationRegion(void* ptr, size_t memory_size) {
      // Insert sorted by end_ptr.
      auto entry =
        std::upper_bound(regions_.begin(), regions_.end(), ptr, &Comparator);
      regions_.insert(entry, AllocationRegion(ptr, memory_size));
    }

    // Adds an alloation region for the given ptr and size, potentially
    // extending a region if ptr matches the end_ptr of an existing region.
    // If a region is extended, returns a pointer to the extended region so
    // that the BFC allocator can reason about chunkification.
    AllocationRegion* AddOrExtendAllocationRegion(void* ptr,
                                                  size_t memory_size) {
      // Insert sorted by end_ptr.
      auto entry =
        std::upper_bound(regions_.begin(), regions_.end(), ptr, &Comparator);
      // Check if can be coalesced with preceding region.
      const auto logger = meepo_embedding::GetDefaultLogger();
      if (entry != regions_.begin()) {
        auto preceding_region = entry - 1;
        if (preceding_region->end_ptr() == ptr) {
          LOG_TRACE_L1(
            logger, "Extending region {} of {} by {} bytes",
            reinterpret_cast<uint64_t>(preceding_region->ptr()),
            strings::HumanReadableNumBytes(preceding_region->memory_size()),
            strings::HumanReadableNumBytes(memory_size));
          preceding_region->extend(memory_size);
          return &*preceding_region;
        }
      }
      LOG_TRACE_L1(logger, "Inserting new region {} of {}",
                   reinterpret_cast<uint64_t>(ptr),
                   strings::HumanReadableNumBytes(memory_size));
      regions_.insert(entry, AllocationRegion(ptr, memory_size));
      return nullptr;
    }

    std::vector<AllocationRegion>::iterator RemoveAllocationRegion(
      std::vector<AllocationRegion>::iterator it) {
      return regions_.erase(it);
    }

    ChunkHandle get_handle(const void* p) const {
      return RegionFor(p)->get_handle(p);
    }

    void set_handle(const void* p, ChunkHandle h) {
      return MutableRegionFor(p)->set_handle(p, h);
    }
    void erase(const void* p) { return MutableRegionFor(p)->erase(p); }

    const std::vector<AllocationRegion>& regions() const { return regions_; }

   private:
    static bool Comparator(const void* ptr, const AllocationRegion& other) {
      return ptr < other.end_ptr();
    }

    AllocationRegion* MutableRegionFor(const void* p) {
      return const_cast<AllocationRegion*>(RegionFor(p));
    }

    const AllocationRegion* RegionFor(const void* p) const {
      auto entry =
        std::upper_bound(regions_.begin(), regions_.end(), p, &Comparator);

      if (entry != regions_.end()) {
        return &(*entry);
      }

      const auto logger = meepo_embedding::GetDefaultLogger();
      LOG_CRITICAL(logger, "Could not find Region for {}", p);
      return nullptr;
    }

   private:
    std::vector<AllocationRegion> regions_;
  };

  // Returns 'bytes' rounded up to the next highest kMinAllocationSize.
  static size_t RoundedBytes(size_t bytes);

  // Try to add a new memory region that can satisfy an allocation of
  // 'rounded_bytes' bytes.  Returns true on success and false on
  // failure.
  bool Extend(size_t alignment, size_t rounded_bytes)
    ME_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Deallocate free regions to give back the memory to suballocator, so that
  // we can re-allocate a larger region.  The main use scenario of this
  // function is when OOM happens but we have free regions and the sum of
  // sizes of free regions and unallocated bytes is larger than the requested
  // size, implying (external) memory fragmentation.  Returns true if any free
  // regions are found and freed; false otherwise.
  bool DeallocateFreeRegions(size_t rounded_bytes);

  // Helper function to deallocate regions.
  void DeallocateRegions(const std::unordered_set<void*>& region_ptrs)
    ME_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Returns a pointer to an underlying allocated chunk of size
  // 'rounded_bytes'.
  void* FindChunkPtr(BinNum bin_num, size_t rounded_bytes, size_t num_bytes,
                     std::uint64_t freed_before)
    ME_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Splits the chunk specified by 'h' into two chunks, one at least
  // of size 'num_bytes'.
  void SplitChunk(ChunkHandle h, size_t num_bytes)
    ME_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Merges the two chunk handles.  Requires that the chunks are
  // contiguous in their allocation.
  void Merge(ChunkHandle h, ChunkHandle h2) ME_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Adds the chunk 'h' to the proper free bin.
  void InsertFreeChunkIntoBin(ChunkHandle h) ME_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Removes the free chunk pointed to by 'c' from the set free_chunks.
  void RemoveFreeChunkIterFromBin(Bin::FreeChunkSet* free_chunks,
                                  const Bin::FreeChunkSet::iterator& c)
    ME_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Removes a free chunk from the bin.
  void RemoveFreeChunkFromBin(ChunkHandle h) ME_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void MaybeRemoveFreeChunkFromBin(ChunkHandle h)
    ME_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Removes the chunk metadata represented by 'h'.
  void DeleteChunk(ChunkHandle h) ME_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  std::string RenderOccupancy() ME_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void DumpMemoryLog(size_t num_bytes) ME_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void MaybeWriteMemoryMap() ME_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  ChunkHandle AllocateChunk() ME_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void DeallocateChunk(ChunkHandle h) ME_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  Chunk* ChunkFromHandle(ChunkHandle h) ME_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  const Chunk* ChunkFromHandle(ChunkHandle h) const
    ME_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  void MarkFree(ChunkHandle h) ME_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  ChunkHandle TryToCoalesce(ChunkHandle h, bool ignore_freed_at)
    ME_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Fragmentation is calculated as the reverse ratio of the largest free
  // chunk size over total free memory, and returns a value within [0, 1].
  double GetFragmentation() ME_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Information about a Bin that is useful for debugging.
  struct BinDebugInfo {
    size_t total_bytes_in_use = 0;
    size_t total_bytes_in_bin = 0;
    size_t total_requested_bytes_in_use = 0;
    size_t total_chunks_in_use = 0;
    size_t total_chunks_in_bin = 0;
  };

  // Computes and returns a BinDebugInfo for each Bin.
  std::array<BinDebugInfo, kNumBins> get_bin_debug_info()
    ME_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  AllocatorRetry retry_helper_;

  // Structures immutable after construction
  size_t memory_limit_ = 0;

  // Map from bin size to Bin
  Bin* BinFromIndex(BinNum index) {
    return reinterpret_cast<Bin*>(&(bins_space_[index * sizeof(Bin)]));
  }
  size_t BinNumToSize(BinNum index) {
    return static_cast<size_t>(256) << index;
  }
  BinNum BinNumForSize(size_t bytes) {
    std::uint64_t v = std::max<size_t>(bytes, 256) >> kMinAllocationBits;
    int b = std::min(kNumBins - 1, Log2Floor64(v));
    return b;
  }
  Bin* BinForSize(size_t bytes) { return BinFromIndex(BinNumForSize(bytes)); }

  char bins_space_[sizeof(Bin) * kNumBins];

  const Options opts_;

  // The size of the current region allocation.
  size_t curr_region_allocation_bytes_;

  // Whether the allocator will coalesce adjacent sub allocator provided
  // AllocationRegions. This may be disabled if discrete sub allocator
  // regions can't be treated as contiguous (e.g. if the allocation refers to
  // device visible memory which is not adjacent to the other region in the
  // device's address space).
  const bool coalesce_regions_;

  pro::proxy<MemoryResource> sub_allocator_;
  std::string name_;
  mutex::shared_counter* timing_counter_ = nullptr;
  std::deque<ChunkHandle> timestamped_chunks_;

  std::atomic<std::uint64_t> safe_frontier_ = {0};

  // Structures mutable after construction
  mutable std::shared_mutex mu_;
  RegionManager region_manager_ ME_GUARDED_BY(mu_);

  std::vector<Chunk> chunks_ ME_GUARDED_BY(mu_);

  // Pointer to head of linked list of free Chunks
  ChunkHandle free_chunks_list_ ME_GUARDED_BY(mu_);

  // Counter containing the next unique identifier to assign to a
  // newly-created chunk.
  int64_t next_allocation_id_ ME_GUARDED_BY(mu_);

  // Stats.
  AllocatorStats stats_ ME_GUARDED_BY(mu_);
#ifdef ME_MEM_DEBUG
  int64 action_counter_ ME_GUARDED_BY(mu_) = 0;
#define MEM_DEBUG_SIZE_HISTORY_SIZE 4096
  int64 size_history_[MEM_DEBUG_SIZE_HISTORY_SIZE];
#endif

  // Quill logger.
  ::quill::Logger* logger_;

  BFCAllocator(const BFCAllocator&) = delete;
  void operator=(const BFCAllocator&) = delete;
};

}  // namespace memory
}  // namespace meepo_embedding

#endif  // MEEPO_EMBEDDING_COMMON_BFC_ALLOCATOR_H_
