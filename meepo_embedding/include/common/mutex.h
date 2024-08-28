/*Copyright 2019 The TensorFlow Authors.. All Rights Reserved.
Copyright 2024 The MeepoEmbedding Authors

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

#ifndef MEEPO_EMBEDDING_COMMON_MUTEX_H_
#define MEEPO_EMBEDDING_COMMON_MUTEX_H_

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <shared_mutex>

#include "meepo_embedding/include/common/macros.h"

namespace meepo_embedding {
namespace mutex {

#if defined(__clang__) && (!defined(SWIG))
#define ME_INTERNAL_THREAD_ANNOTATION_ATTRIBUTE(x) __attribute__((x))
#else
#define ME_INTERNAL_THREAD_ANNOTATION_ATTRIBUTE(x)  // no-op
#endif

// Document if a class does RAII locking (such as the MutexLock class).
// The constructor should use LOCK_FUNCTION to specify the mutex that is
// acquired, and the destructor should use ME_UNLOCK_FUNCTION with no arguments;
// the analysis will assume that the destructor unlocks whatever the
// constructor locked.
#define ME_SCOPED_LOCKABLE \
  ME_INTERNAL_THREAD_ANNOTATION_ATTRIBUTE(scoped_lockable)

#define ME_SHARED_LOCK_FUNCTION(...) \
  ME_INTERNAL_THREAD_ANNOTATION_ATTRIBUTE(shared_lock_function(__VA_ARGS__))

// Document functions that expect a lock to be held on entry to the function,
// and release it in the body of the function.
#define ME_UNLOCK_FUNCTION(...) \
  ME_INTERNAL_THREAD_ANNOTATION_ATTRIBUTE(unlock_function(__VA_ARGS__))

class ME_SCOPED_LOCKABLE shared_lock_guard {
 public:
  typedef std::shared_mutex mutex_type;

  explicit shared_lock_guard(mutex_type& mu) ME_SHARED_LOCK_FUNCTION(mu)
    : mu_(&mu) {
    mu_->lock_shared();
  }

  shared_lock_guard(mutex_type& mu, std::try_to_lock_t /*tag*/)
    ME_SHARED_LOCK_FUNCTION(mu)
    : mu_(&mu) {
    if (!mu.try_lock_shared()) {
      mu_ = nullptr;
    }
  }

  // Manually nulls out the source to prevent double-free.
  // (std::move does not null the source pointer by default.)
  shared_lock_guard(shared_lock_guard&& ml) noexcept
    ME_SHARED_LOCK_FUNCTION(ml.mu_)
    : mu_(ml.mu_) {
    ml.mu_ = nullptr;
  }
  ~shared_lock_guard() ME_UNLOCK_FUNCTION() {
    if (mu_ != nullptr) {
      mu_->unlock_shared();
    }
  }
  mutex_type* mutex() { return mu_; }

  explicit operator bool() const { return mu_ != nullptr; }

 private:
  mutex_type* mu_;
};

class shared_counter {
 public:
  shared_counter() : value_(0){};
  ~shared_counter(){};
  int64_t get() { return value_; }
  int64_t next() { return ++value_; }

 private:
  std::atomic<int64_t> value_;
};

}  // namespace mutex

}  // namespace meepo_embedding

#endif  // MEEPO_EMBEDDING_COMMON_MUTEX_H_