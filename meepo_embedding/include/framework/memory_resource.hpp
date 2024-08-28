/* Copyright 2024 The MeepoEmbedding Authors. All Rights Reserved.

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

#ifndef MEEPO_EMBEDDING_COMMON_MEMORY_RESOURCE_H_
#define MEEPO_EMBEDDING_COMMON_MEMORY_RESOURCE_H_

#include <cstdlib>
#include <functional>
#include <memory>
#include <memory_resource>
#include <nvexec/detail/memory.cuh>

#include "meepo_embedding/include/common/mutex.h"
#include "meepo_embedding/include/third_party/magic_enum.hpp"
#include "meepo_embedding/include/third_party/proxy.h"

namespace meepo_embedding {
namespace memory {

// Align to 64 byte boundary.
static constexpr size_t default_alignment = 64;

enum class MemoryResourceType {
  UNKNOWN = 0,
  HOST = 1,
  GPU_DEVICE = 2,
  GPU_MANAGED = 3,
  GPU_PINNED = 4
};

using namespace ::magic_enum::bitwise_operators;

// CUDA
typedef ::nvexec::_strm::pinned_resource stdexec_pinned_resource;
typedef ::nvexec::_strm::gpu_resource stdexec_gpu_resource;
typedef ::nvexec::_strm::managed_resource stdexec_managed_resource;
typedef ::nvexec::_strm::monotonic_buffer_resource stdexec_mono_resource;
typedef ::nvexec::_strm::synchronized_pool_resource stdexec_sync_resource;

// define defualt function begin
struct NotImplemented {
  explicit NotImplemented(auto&&...) {
    throw std::runtime_error{
      "Not implemented function in memory resource instance!"};
  }

  template <class T>
  operator T() const noexcept {
    std::unreachable();
  }
};

std::function<bool()> default_support_coalescing = []() { return false; };

std::function<MemoryResourceType()> default_get_mem_type = []() {
  return MemoryResourceType::UNKNOWN;
};

// If the first attempt to allocate the memory fails, the allocation should
// wait and retry (with a timeout).
//
// This is usually set to true, but we may set it to false in cases where a
// failure has only performance impact (e.g. optional scratch space
// allocation).
std::function<bool()> default_allow_retry_on_failure = []() { return false; };

// define defualt function end

//
// Specifications of abstraction
// For more details, please check:
// https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p3086r2.pdf
//
PRO_DEF_MEM_DISPATCH(MemAlloc, do_allocate);
PRO_DEF_MEM_DISPATCH(MemDealloc, do_deallocate);
PRO_DEF_MEM_DISPATCH(MemIsEqual, do_is_equal);
PRO_DEF_MEM_DISPATCH(MemCoalescing, SupportsCoalescing);
PRO_DEF_MEM_DISPATCH(MemMemoryType, GetMemoryType);
PRO_DEF_MEM_DISPATCH(MemRetryFail, AllowRetryOnFailure);

PRO_DEF_WEAK_DISPATCH(WeekMemAlloc, MemAlloc, NotImplemented);
PRO_DEF_WEAK_DISPATCH(WeekMemDealloc, MemDealloc, NotImplemented);
PRO_DEF_WEAK_DISPATCH(WeekMemIsEqual, MemIsEqual, NotImplemented);
PRO_DEF_WEAK_DISPATCH(WeekMemCoalescing, MemCoalescing,
                      default_support_coalescing);
PRO_DEF_WEAK_DISPATCH(WeekMemMemoryType, MemMemoryType, default_get_mem_type);
PRO_DEF_WEAK_DISPATCH(WeekMemRetryFail, MemRetryFail,
                      default_allow_retry_on_failure);

// clang-format off
struct MemoryResource : pro::facade_builder 
    ::add_convention<WeekMemAlloc,
                     void*(const std::size_t bytes,
                           const std::size_t alignment,
                           const std::size_t* bytes_received)>
    ::add_convention<WeekMemDealloc, 
                     void(void* ptr, const std::size_t bytes,
                          const std::size_t alignment)>
    ::add_convention<WeekMemIsEqual,
                     bool(const std::pmr::memory_resource& other) const>
    ::add_convention<WeekMemCoalescing, bool() const>
    ::add_convention<WeekMemMemoryType, MemoryResourceType() const>
    ::add_convention<WeekMemRetryFail, bool() const>
    ::build {};  // clang-format on

template <typename R>
concept pmr_rs_base =
  std::is_base_of_v<std::remove_pointer_t<R>, std::pmr::memory_resource>;

/*
Template specialization classes for the implementation of various memory
resources. It's easy to call their instance with pro::proxy.

Usage Example: pro::proxy<MemoryResource> mem_resource =
  pro::make_proxy<MemoryResourceImpl<stdexec_gpu_resource>>();
  MemoryResourceType mem_type = mem_resource->GetMemoryType();
*/
template <typename R>
  requires pmr_rs_base<R>
struct MemoryResourceImpl {
 public:
  gpu_resource() { memory_resourec_instance_ = std::make_unique<R>(); };

  void* do_allocate(const std::size_t bytes, const std::size_t alignment,
                    const std::size_t* bytes_received = nullptr) {
    if (ME_PREDICT_TRUE(bytes_received != nullptr)) {
      *bytes_received = bytes;
    }
    dynamic_cast<R*>(memory_resourec_instance_.get())
      ->do_allocate(bytes, alignment);
  }
  void do_deallocate(void* ptr, const std::size_t bytes,
                     const std::size_t alignment) {
    dynamic_cast<R*>(memory_resourec_instance_.get())
      ->do_deallocate(ptr, bytes, alignment);
  }
  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept {
    dynamic_cast<R*>(memory_resourec_instance_.get())->do_is_equal(other);
  };

  bool SupportsCoalescing() const { return false; }

  MemoryResourceType GetMemoryType() const {
    return MemoryResourceType::UNKNOWN;
  }

 private:
  std::unique_ptr<std::pmr::memory_resource> memory_resourec_instance_;
};

// CUDA
template <typename R>
  requires std::same_as<R, stdexec_gpu_resource>
struct MemoryResourceImpl<R> {
  MemoryResource() {
    // TODO(MoFHeka): Support cudaMallocAsync
    const char* gpu_allocator_ = std::getenv("ME_GPU_ALLOCATOR");
    if (std::strcmp(gpu_allocator_, "cuda_malloc_async")) {
      cuda_malloc_async_ = true;
    }
  }

 public:
  void* do_allocate(const std::size_t bytes, const std::size_t alignment,
                    const std::size_t* bytes_received = nullptr) {
    if (ME_PREDICT_TRUE(bytes_received != nullptr)) {
      *bytes_received = bytes;
    }
    dynamic_cast<R*>(memory_resourec_instance_.get())
      ->do_allocate(bytes, alignment);
  }
  void do_deallocate(void* ptr, const std::size_t bytes,
                     const std::size_t alignment) {
    dynamic_cast<R*>(memory_resourec_instance_.get())
      ->do_deallocate(ptr, bytes, alignment);
  }
  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept {
    dynamic_cast<R*>(memory_resourec_instance_.get())->do_is_equal(other);
  };

  bool SupportsCoalescing() const { return false; }

  MemoryResourceType GetMemoryType() const {
    return MemoryResourceType::GPU_DEVICE;
  }

 private:
  bool cuda_malloc_async_ = false;
};

template <typename R>
  requires std::same_as<R, stdexec_managed_resource>
struct MemoryResourceImpl<R> {
 public:
  bool SupportsCoalescing() const { return false; }

  MemoryResourceType GetMemoryType() const {
    return MemoryResourceType::GPU_MANAGED;
  }
};

template <typename R>
  requires std::same_as<R, stdexec_pinned_resource>
struct MemoryResourceImpl<R> {
 public:
  bool SupportsCoalescing() const { return true; }

  MemoryResourceType GetMemoryType() const {
    return MemoryResourceType::GPU_PINNED;
  }

  bool AllowRetryOnFailure() const {return true};
};

// CPU STD
template <typename R>
  requires std::same_as<R, std::pmr::monotonic_buffer_resource>
struct MemoryResourceImpl<R> {
  MemoryResource(std::size_t initial_size) {
    memory_resourec_instance_ = std::make_unique<R>(initial_size);
  }

 public:
  bool SupportsCoalescing() const { return false; }

  MemoryResourceType GetMemoryType() const { return MemoryResourceType::HOST; }
};

template <typename R>
  requires std::same_as<R, std::pmr::synchronized_pool_resource>
struct MemoryResourceImpl<R> {
 public:
  bool SupportsCoalescing() const { return true; }

  MemoryResourceType GetMemoryType() const { return MemoryResourceType::HOST; }

  bool AllowRetryOnFailure() const {return true};
};

template <typename R>
  requires std::same_as<R, std::pmr::unsynchronized_pool_resource>
struct MemoryResourceImpl<R> {
 public:
  bool SupportsCoalescing() const { return true; }

  MemoryResourceType GetMemoryType() const { return MemoryResourceType::HOST; }

  bool AllowRetryOnFailure() const {return true};
};

// Runtime statistics collected by an allocator. Exactly the same as
// stream_executor::AllocatorStats, but independently defined to preserve the
// mutual independence of StreamExecutor and TensorFlow.
struct AllocatorStats {
  int64_t num_allocs;          // Number of allocations.
  int64_t bytes_in_use;        // Number of bytes in use.
  int64_t peak_bytes_in_use;   // The peak bytes in use.
  int64_t largest_alloc_size;  // The largest single allocation seen.

  // The upper limit of bytes of user allocatable device memory, if such a
  // limit is known.
  std::optional<int64_t> bytes_limit;

  // Stats for reserved memory usage.
  int64_t bytes_reserved;       // Number of bytes reserved.
  int64_t peak_bytes_reserved;  // The peak number of bytes reserved.
  // The upper limit on the number bytes of reservable memory,
  // if such a limit is known.
  std::optional<int64_t> bytes_reservable_limit;

  int64_t largest_free_block_bytes;  // Largest free block's size in heap.

  // Number of bytes of memory held by the allocator.  This may be higher than
  // bytes_in_use if the allocator holds a pool of memory (e.g. BFCAllocator).
  std::optional<int64_t> pool_bytes;
  std::optional<int64_t> peak_pool_bytes;

  AllocatorStats()
    : num_allocs(0),
      bytes_in_use(0),
      peak_bytes_in_use(0),
      largest_alloc_size(0),
      bytes_reserved(0),
      peak_bytes_reserved(0),
      largest_free_block_bytes(0) {}

  std::string DebugString() const {
    return strings::StrFormat(
      "Limit:            {:20lld}\n"
      "InUse:            {:20lld}\n"
      "MaxInUse:         {:20lld}\n"
      "NumAllocs:        {:20lld}\n"
      "MaxAllocSize:     {:20lld}\n"
      "Reserved:         {:20lld}\n"
      "PeakReserved:     {:20lld}\n"
      "LargestFreeBlock: {:20lld}\n",
      this->bytes_limit ? *this->bytes_limit : 0, this->bytes_in_use,
      this->peak_bytes_in_use, this->num_allocs, this->largest_alloc_size,
      this->bytes_reserved, this->peak_bytes_reserved,
      this->largest_free_block_bytes);
  };
};

}  // namespace memory
}  // namespace meepo_embedding

#endif  // MEEPO_EMBEDDING_COMMON_MEMORY_RESOURCE_H_
