/*Copyright 2024 The MeepoEmbedding Authors. All Rights Reserved.

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

#ifndef MEEPO_EMBEDDING_FRAMEWORK_CONTEXT_H_
#define MEEPO_EMBEDDING_FRAMEWORK_CONTEXT_H_

#include <yaml-cpp/yaml.h>  // from @yaml-cpp

#include <cinttypes>
#include <ctime>
#include <exec/linux/io_uring_context.hpp>  // from @stdexec
#include <exec/static_thread_pool.hpp>      // from @stdexec
#include <functional>
#include <memory>
#include <nvexec/stream_context.cuh>  // from @stdexec
#include <optional>
#include <span>
#include <unordered_set>
#include <utility>
#include <vector>

#include "meepo_embedding/include/common/status.h"
#include "meepo_embedding/include/framework/resource_mgr.hpp"
#include "meepo_embedding/include/storage/register/storage_interface.hpp"

namespace meepo_embedding {

class MeepoEmbeddingContext {
 public:
  // TODO(MoFHeka): Do some cleanup of Params.
  // The Params struct is passed in to initialize an MeepoEmbeddingContext,
  // and must outlive the MeepoEmbeddingContext.
  struct Params {
    Params() : stream_ctx_(new nvexec::stream_context()) {};
    ~Params() { delete stream_ctx_; };

    // The step being executed.
    int64_t step_id = 0;

    // Timestamp for the start of graph execution. Used for latency metrics.
    int64_t start_time_usecs = 0;

    // The deadline for the context to complete by. Empty if unspecified.
    std::optional<std::time_t> deadline;

    // Linux io_uring context from Nvidia standard excutor
    ::exec::io_uring_context* io_uring_ctx_ = nullptr;

    // Static thread pool from Nvidia standard excutor
    ::exec::static_thread_pool* thread_pool_ = nullptr;

    // Nvidia GPU stream context from Nvidia standard excutor
    ::nvexec::stream_context* stream_ctx_ = nullptr;

    bool track_allocations = false;
    bool log_memory = false;

    // Unified KV lookuptable interface, cannot be empty
    std::vector<pro::proxy<storage::StorageInterface>> storage_container = {};

    // Shared resources accessible by this meepo context invocation.
    ResourceMgr* resource_manager = nullptr;

    // Per-step resources accessible by this meepo context invocation should be
    // stored in this container..
    StepContainer* step_container = nullptr;

    // Context configuration parameters. Can be nullptr.
    const YAML::Node* context_config = nullptr;

    // Unique context identifier. Can be empty.
    std::string context_handle;

    // For access to distributed coordination service.
    RendezvousAgent* rendezvous_agent = nullptr;

    // TODO(MoFHeka): A RPC Feature if needed
    // // Mechanism used by this meepo context invocation to communicate with
    // // computations running on other devices.
    // RendezvousFunctionInterface* rendezvous_func_interface = nullptr;

    // Mechanism for executing a collective op that needs to coordinate
    // with parallel instances running on other devices.
    CollectiveExecutor* collective_executor = nullptr;
  };

  // params must outlive the MeepoEmbeddingContext.
  explicit MeepoEmbeddingContext(Params* params);
  ~MeepoEmbeddingContext();

  int64_t step_id() const { return params_->step_id; }

  int64_t start_time_usecs() const { return params_->start_time_usecs; }

  const YAML::Node* context_config() const { return params_->context_config; }

  // The deadline for the session to complete by. Empty if unspecified in
  // RunOptions.
  std::optional<std::time_t> deadline() const { return params_->deadline; }

  // Communication.
  //
  // TODO(MoFHeka): A RPC Feature if needed
  // // An meepo context communicates with outside environment through
  // // Rendezvous Send() and Recv().
  // RendezvousFunctionInterface* rendezvous_func_interface() const { return
  // params_->rendezvous_func_interface; }

  CollectiveExecutor* collective_executor() const {
    return params_->collective_executor;
  }

  // Shared resources accessible to this kernel.
  ResourceMgr* resource_manager() const { return params_->resource_manager; }

  // Execution.
  //

  // Error handling.

  // An OpKernel should call Setstatus() if Compute() encounters an
  // error.
  void Setstatus(const Status& status);
  const Status& status() const { return status_; }

  // Other accessors.

  // Per-step container for use by white-listed internal ops.
  StepContainer* step_container() const { return params_->step_container; }

  // Access to distributed coordination service.
  RendezvousAgent* rendezvous_agent() const {
    return params_->rendezvous_agent;
  }

  Params* params() const { return params_; }
  void set_params(Params* params) { params_ = params; }

 private:
  // TODO(MoFHeka): Add memory usage record.
  // bool record_memory_consumption_ = false;

  // Internal common method used when allocating tensor memory
  template <typename T>
  std::errc allocate_tensor(const meepo_embedding::Tensor<T>* Tensor);

  Status status_;
  friend class CollectiveExecutor;  // for access to params_
  Params* params_;                  // not owned

  // TODO(MoFHeka): Add memory usage record.
  // // The following data members are only used when allocation tracking is
  // // enabled, memory consumption is being recorded, or tensor access is being
  // // recorded.
  // struct TrackingState {
  //   mutable std::mutex mu;
  //   std::vector<WrappedAllocator> wrapped_allocators
  //       ME_GUARDED_BY(mu);

  //   mutable std::mutex stats_mu;
  //   int64_t temp_memory_allocated ME_GUARDED_BY(stats_mu) = 0;

  //   int64_t persistent_memory_allocated ME_GUARDED_BY(stats_mu) = 0;
  //   std::vector<std::pair<const void*, int64_t>>
  //       temp_tensor_buffer_and_size ME_GUARDED_BY(stats_mu);
  //   std::vector<int64_t> persistent_alloc_ids
  //       ME_GUARDED_BY(stats_mu);
  // };
  // std::unique_ptr<TrackingState> tracking_state_;

  MeepoEmbeddingContext(const MeepoEmbeddingContext&) = delete;
  void operator=(const MeepoEmbeddingContext&) = delete;
};

}  // namespace meepo_embedding

#endif  // MEEPO_EMBEDDING_FRAMEWORK_CONTEXT_H_
