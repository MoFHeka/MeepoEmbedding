/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Copyright 2024 The MeepoEmbedding Authors. All Rights Reserved.

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

#ifndef MEEPO_EMBEDDING_FRAMEWORK_RESOURCE_MGR_H_
#define MEEPO_EMBEDDING_FRAMEWORK_RESOURCE_MGR_H_

#include <atomic>
#include <cstring>
#include <expected>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <variant>

#include "meepo_embedding/include/common/macros.h"
#include "meepo_embedding/include/common/mutex.h"
#include "meepo_embedding/include/common/status.h"
#include "meepo_embedding/include/common/strings.h"
#include "meepo_embedding/include/common/type_index.h"
#include "meepo_embedding/include/framework/memory_resource.hpp"
#include "meepo_embedding/include/framework/resource_base.hpp"

namespace meepo_embedding {

// A ResourceMgr instance keeps track of named and typed resources
// grouped into containers.
//
// Each named resource is
// registered with ResourceMgr under a named "container" name. At any
// time, there is at most one instance of a resource given the container
// name, the resource type and the resource name.
//
// All resources for a given container can be dropped by one call of
// Cleanup().

template <typename T>
concept no_nullptr = !std::is_same_v<std::remove_cvref_t<T>, std::nullptr_t>;

template <typename T>
concept rs_base = std::is_base_of_v<ResourceBase, std::remove_pointer_t<T>>;

class ResourceMgr {
 public:
  ResourceMgr();
  explicit ResourceMgr(const std::string& default_container);
  ~ResourceMgr();

  // Returns the default container name for *this.
  const std::string& default_container() const { return default_container_; }

  // Creates a resource "name" in the "container".  The caller transfers
  // the ownership of one ref on "resource" to *this, regardless of whether this
  // operation succeeds or fails.
  //
  // REQUIRES: std::is_base_of<ResourceBase, T>
  // REQUIRES: resource != nullptr.
  template <typename T>
  std::error_code Create(const std::string& container, const std::string& name,
                         std::shared_ptr<T> resource)
    requires rs_base<T> && no_nullptr<T>
  ME_MUST_USE_RESULT;

  // Creates a unowned resource "name" in the "container".  The caller does NOT
  // transfer the ownership of any ref on "resource" to *this, regardless of
  // whether this operation succeeds or fails.
  //
  // After the resource is destroyed, lookups from the manager fail.
  // The caller must call this->Delete() on the name to free up the memory
  // entry of the name.
  //
  // REQUIRES: std::is_base_of<ResourceBase, T>
  // REQUIRES: resource != nullptr.
  template <typename T>
  std::error_code CreateUnowned(const std::string& container,
                                const std::string& name,
                                std::shared_ptr<T>
                                  resource)
    requires rs_base<T> && no_nullptr<T>
  ME_MUST_USE_RESULT;

  // If "container" has a resource "name", returns it in "*resource" and
  // the caller takes the ownership of one ref on "*resource".
  //
  // REQUIRES: std::is_base_of<ResourceBase, T>
  // REQUIRES: resource != nullptr
  template <typename T, bool use_dynamic_cast = false>
  std::error_code Lookup(const std::string& container, const std::string& name,
                         std::shared_ptr<T>* resource) const
    requires rs_base<T> && no_nullptr<T>
  ME_MUST_USE_RESULT;

  // Similar to Lookup, but looks up multiple resources at once, with only a
  // single lock acquisition.  If containers_and_names[i] is uninitialized
  // then this function does not modify resources[i].
  template <typename T, bool use_dynamic_cast = false>
  std::error_code LookupMany(
    std::span<std::pair<const std::string*, const std::string*> const>
      containers_and_names,
    std::vector<std::shared_ptr<T>>* resources) const
    requires rs_base<T> && no_nullptr<T>
  ME_MUST_USE_RESULT;

  // If "container" has a resource "name", returns it in
  // "*resource". Otherwise, invokes creator() to create the resource.
  // The caller takes the ownership of one ref on "*resource".
  //
  // WARNING: creator() must not call any methods on ResourceMgr during its
  // execution, because a non-reentrant lock is held during the creator() call
  // in order to guarantee atomicity of LookupOrCreate().
  //
  // REQUIRES: std::is_base_of<ResourceBase, T>
  // REQUIRES: resource != nullptr
  template <typename T, bool use_dynamic_cast = false>
  std::error_code LookupOrCreate(
    const std::string& container, const std::string& name,
    std::shared_ptr<T>* resource,
    std::function<std::error_code(std::shared_ptr<T>*)> creator)
    requires rs_base<T> && no_nullptr<T>
  ME_MUST_USE_RESULT;

  // Deletes the resource "name" from the "container".
  //
  // REQUIRES: std::is_base_of<ResourceBase, T>
  template <typename T>
  std::error_code Delete(const std::string& container, const std::string& name)
    requires rs_base<T>
  ME_MUST_USE_RESULT;

  // Deletes all resources from the "container" and removes the container.
  std::error_code Cleanup(const std::string& container) ME_MUST_USE_RESULT;

  // Deletes all resources in all containers.
  void Clear();

  // Returns a text description for all resources.
  std::string DebugString() const;

 private:
  typedef std::pair<uint64_t, std::string_view> Key;
  struct KeyHash {
    std::size_t operator()(const Key& k) const {
      std::size_t h1 = std::hash<uint64_t>{}(k.first);
      std::size_t h2 = std::hash<std::string_view>{}(k.second);
      return h1 ^ (h2 << 1);  // or use boost::hash_combine
    }
  };
  struct KeyEqual {
    bool operator()(const Key& x, const Key& y) const {
      return (x.second == y.second) && (x.first == y.first);
    }
  };
  struct ResourceAndName {
    std::variant<std::shared_ptr<ResourceBase>, std::weak_ptr<ResourceBase>>
      resource;
    std::unique_ptr<std::string> name;

    ResourceAndName();
    explicit ResourceAndName(const std::string& name);
    ResourceAndName(ResourceAndName&& other) noexcept;
    ~ResourceAndName();

    ResourceAndName& operator=(ResourceAndName&&) noexcept;

    // Returns a strong reference to resource, or nullptr if the resource is
    // no longer valid.
    std::shared_ptr<ResourceBase> GetResource() const;

   private:
    ResourceAndName(const ResourceAndName&) = delete;
    void operator=(const ResourceAndName&) = delete;
  };
  typedef std::pmr::unordered_map<Key, ResourceAndName, KeyHash, KeyEqual>
    Container;

  std::pmr::unsynchronized_pool_resource mem_resource_;

  const std::string default_container_;
  mutable std::shared_mutex mu_;
  std::pmr::unordered_map<std::string, Container*> containers_
    ME_GUARDED_BY(mu_){&mem_resource_};

  template <typename T, bool use_dynamic_cast = false>
  std::error_code LookupInternal(const std::string& container,
                                 const std::string& name,
                                 std::shared_ptr<T>* resource) const
    ME_SHARED_LOCKS_REQUIRED(mu_) ME_MUST_USE_RESULT;
  std::error_code LookupInternal(const std::string& container,
                                 uint64_t type_hash_code,
                                 const std::string& name,
                                 std::shared_ptr<ResourceBase>* resource) const
    ME_SHARED_LOCKS_REQUIRED(mu_) ME_MUST_USE_RESULT;

  std::error_code DoCreate(const std::string& container, TypeIndex type,
                           const std::string& name,
                           std::shared_ptr<ResourceBase> resource,
                           bool owns_resource)
    ME_EXCLUSIVE_LOCKS_REQUIRED(mu_) ME_MUST_USE_RESULT;

  std::error_code DoLookup(const std::string& container, TypeIndex type,
                           const std::string& name,
                           std::shared_ptr<ResourceBase>* resource) const
    ME_SHARED_LOCKS_REQUIRED(mu_) ME_MUST_USE_RESULT;
  std::error_code DoLookup(const std::string& container,
                           uint64_t type_hash_code,
                           const std::string& type_name,
                           const std::string& resource_name,
                           std::shared_ptr<ResourceBase>* resource) const
    ME_SHARED_LOCKS_REQUIRED(mu_) ME_MUST_USE_RESULT;

  std::error_code DoDelete(const std::string& container,
                           uint64_t type_hash_code,
                           const std::string& resource_name,
                           const std::string& type_name) ME_MUST_USE_RESULT;
  std::error_code DoDelete(const std::string& container, TypeIndex type,
                           const std::string& resource_name) ME_MUST_USE_RESULT;

  // Pops the ResourceAndName entry. The entry is moved from the list to
  // the output argument `resource_and_name`.
  std::error_code PopResourceAndName(
    const std::string& container, uint64_t type_hash_code,
    const std::string& resource_name, const std::string& type_name,
    ResourceAndName& resource_and_name) ME_MUST_USE_RESULT;
  // Inserts the type name for 'hash_code' into the hash_code to type name map.
  std::error_code InsertDebugTypeName(uint64_t hash_code,
                                      const std::string& type_name)
    ME_EXCLUSIVE_LOCKS_REQUIRED(mu_) ME_MUST_USE_RESULT;

  // Returns the type name for the 'hash_code'.
  // Returns "<unknown>" if a resource with such a type was never inserted into
  // the container.
  const char* DebugTypeName(uint64_t hash_code) const
    ME_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Map from type hash_code to type name.
  std::unordered_map<uint64_t, std::string> debug_type_names_
    ME_GUARDED_BY(mu_);

  ResourceMgr(const ResourceMgr&) = delete;
  void operator=(const ResourceMgr&) = delete;
};

// Container used for per-step resources.
class StepContainer {
 public:
  // step_id: the unique ID of this step. Doesn't have to be sequential, just
  // has to be unique.
  // cleanup: callback to delete a container of this name.
  // prefix: optional string prefix to disambiguate step containers.
  StepContainer(
    const int64_t step_id, std::function<void(const std::string&)> cleanup)
    : step_id_(step_id),
      container_(strings::StrCat("__per_step_", step_id)),
      cleanup_(cleanup),
      dirty_(false) {}

  StepContainer(const int64_t step_id,
                std::function<void(const std::string&)>
                  cleanup,
                const std::string& prefix)
    : step_id_(step_id),
      container_(strings::StrCat("__", prefix, "_per_step_", step_id)),
      cleanup_(cleanup),
      dirty_(false) {}

  ~StepContainer() { CleanUp(); }

  void CleanUp() ME_NO_THREAD_SAFETY_ANALYSIS {
    // NOTE(mrry): Avoid acquiring the mutex in the case that the container is
    // clean.
    if (dirty_) {
      const std::lock_guard<std::shared_mutex> lock(mu_);
      cleanup_(container_);
      dirty_ = false;
    }
  }

  // Pass through functions for resource lookup and creation. We do this to
  // ensure that we can appropriately set the dirty_ bit in the
  // StepContainer if the name of the container is used to create
  // resources.

  // Pass through to ResourceMgr::Create with the container name
  template <typename T>
  std::error_code Create(ResourceMgr* rm, const std::string& name,
                         std::shared_ptr<T> resource) ME_MUST_USE_RESULT;
  // Pass through to ResourceMgr::Delete with the container name
  template <typename T>
  std::error_code Delete(ResourceMgr* rm,
                         const std::string& name) ME_MUST_USE_RESULT;
  // Pass through to ResourceMgr::Lookup with the container name
  template <typename T>
  std::error_code Lookup(ResourceMgr* rm, const std::string& name,
                         std::shared_ptr<T>* resource) const ME_MUST_USE_RESULT;
  // Pass through to ResourceMgr::LookupOrCreate with the container name
  template <typename T>
  std::error_code LookupOrCreate(
    ResourceMgr* rm, const std::string& name, std::shared_ptr<T>* resource,
    std::function<std::error_code(std::shared_ptr<T>*)> creator)
    ME_MUST_USE_RESULT;
  int64_t StepId() const { return step_id_; }

 private:
  const int64_t step_id_;
  const std::string container_;
  const std::function<void(const std::string&)> cleanup_;
  std::shared_mutex mu_;
  mutable std::atomic<bool> dirty_ ME_GUARDED_BY(mu_);
};

// Implementation details below.

template <typename T>
std::error_code ResourceMgr::Create(const std::string& container,
                                    const std::string& name,
                                    std::shared_ptr<T>
                                      resource)
  requires rs_base<T> && no_nullptr<T>
{
  const std::lock_guard<std::shared_mutex> lock(mu_);
  return DoCreate(container, TypeIndex::Make<T>(), name, resource,
                  /* owns_resource */ true);
}

template <typename T>
std::error_code ResourceMgr::CreateUnowned(const std::string& container,
                                           const std::string& name,
                                           std::shared_ptr<T>
                                             resource)
  requires rs_base<T> && no_nullptr<T>
{
  const std::lock_guard<std::shared_mutex> lock(mu_);
  return DoCreate(container, TypeIndex::Make<T>(), name, resource,
                  /* owns_resource */ false);
}

template <typename T, bool use_dynamic_cast>
std::error_code ResourceMgr::Lookup(const std::string& container,
                                    const std::string& name,
                                    std::shared_ptr<T>* resource) const
  requires rs_base<T> && no_nullptr<T>
{
  const mutex::shared_lock_guard lock(mu_);
  return LookupInternal<T, use_dynamic_cast>(container, name, resource);
}

template <typename T, bool use_dynamic_cast>
std::error_code ResourceMgr::LookupMany(
  std::span<std::pair<const std::string*, const std::string*> const>
    containers_and_names,
  std::vector<std::shared_ptr<T>>* resources) const
  requires rs_base<T> && no_nullptr<T>
{
  const mutex::shared_lock_guard lock(mu_);
  resources->resize(containers_and_names.size());
  for (size_t i = 0; i < containers_and_names.size(); ++i) {
    std::shared_ptr<T> resource;
    std::error_code s =
      LookupInternal<T, use_dynamic_cast>(*containers_and_names[i].first,
                                          *containers_and_names[i].second,
                                          &resource);
    if (ME_PREDICT_TRUE(static_cast<Status>(s.value()) == Status::OK)) {
      (*resources)[i].reset(resource);
    }
  }
  return make_error_code(Status::OK);
}

// Simple wrapper to allow conditional dynamic / static casts.
template <typename T, bool use_dynamic_cast>
struct TypeCastFunctor {
  static T* Cast(ResourceBase* r) { return static_cast<T*>(r); }
};

template <typename T>
struct TypeCastFunctor<T, true> {
  static T* Cast(ResourceBase* r) { return dynamic_cast<T*>(r); }
};

template <typename T, bool use_dynamic_cast>
std::error_code ResourceMgr::LookupInternal(
  const std::string& container, const std::string& name,
  std::shared_ptr<T>* resource) const {
  ResourceBase* found = nullptr;
  std::error_code s = DoLookup(container, TypeIndex::Make<T>(), name, &found);
  if (ME_PREDICT_TRUE(static_cast<Status>(s.value()) == Status::OK)) {
    // It's safe to down cast 'found' to T* since
    // typeid(T).hash_code() is part of the map key.
    *resource = TypeCastFunctor<T, use_dynamic_cast>::Cast(found);
  }
  return s;
}

template <typename T, bool use_dynamic_cast>
std::error_code ResourceMgr::LookupOrCreate(
  const std::string& container, const std::string& name,
  std::shared_ptr<T>* resource,
  std::function<std::error_code(std::shared_ptr<T>*)> creator)
  requires rs_base<T> && no_nullptr<T>
{
  *resource = nullptr;
  std::error_code s;
  {
    mutex::shared_lock_guard lock(mu_);
    s = LookupInternal<T, use_dynamic_cast>(container, name, resource);
    if (ME_PREDICT_TRUE(static_cast<Status>(s.value()) == Status::OK)) return s;
  }
  std::lock_guard<std::shared_mutex> lock(mu_);
  s = LookupInternal<T, use_dynamic_cast>(container, name, resource);
  if (ME_PREDICT_TRUE(static_cast<Status>(s.value()) == Status::OK)) return s;
  ME_RETURN_IF_ERROR(creator(resource));
  s = DoCreate(container, TypeIndex::Make<T>(), name, *resource,
               /* owns_resource */ true);
  ME_RETURN_IF_ERROR(s);
  return s;
}

template <typename T>
std::error_code ResourceMgr::Delete(const std::string& container,
                                    const std::string& name)
  requires rs_base<T>
{
  return DoDelete(container, TypeIndex::Make<T>(), name);
}

template <typename T>
std::error_code StepContainer::Lookup(ResourceMgr* rm, const std::string& name,
                                      std::shared_ptr<T>* resource) const {
  return rm->Lookup<T>(container_, name, resource);
}

template <typename T>
std::error_code StepContainer::LookupOrCreate(
  ResourceMgr* rm, const std::string& name, std::shared_ptr<T>* resource,
  std::function<std::error_code(std::shared_ptr<T>*)> creator) {
  const std::lock_guard<std::shared_mutex> lock(mu_);
  dirty_ = true;
  return rm->LookupOrCreate<T>(container_, name, resource, creator);
}

template <typename T>
std::error_code StepContainer::Create(ResourceMgr* rm, const std::string& name,
                                      std::shared_ptr<T> resource) {
  const std::lock_guard<std::shared_mutex> lock(mu_);
  dirty_ = true;
  return rm->Create<T>(container_, name, resource);
}

template <typename T>
std::error_code StepContainer::Delete(ResourceMgr* rm,
                                      const std::string& name) {
  return rm->Delete<T>(container_, name);
}

}  // namespace meepo_embedding

#endif  // MEEPO_EMBEDDING_FRAMEWORK_RESOURCE_MGR_H_
