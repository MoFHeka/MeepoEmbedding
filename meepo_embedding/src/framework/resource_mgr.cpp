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

#include "meepo_embedding/include/framework/resource_mgr.hpp"

#include "meepo_embedding/include/common/logger.h"

namespace meepo_embedding {
std::error_code ResourceMgr::InsertDebugTypeName(uint64_t hash_code,
                                                 const std::string& type_name) {
  auto iter = debug_type_names_.emplace(hash_code, type_name);
  if (iter.first->second != type_name) {
    const auto logger = GetDefaultLogger();
    LOG_ERROR(logger, "Duplicate hash code found for type {}.", type_name);
    return make_error_code(Status::ALREADY_EXISTS);
  }
  return make_error_code(Status::OK);
}

const char* ResourceMgr::DebugTypeName(uint64_t hash_code) const {
  auto type_name_iter = debug_type_names_.find(hash_code);
  if (type_name_iter == debug_type_names_.end()) {
    return "<unknown>";
  } else {
    return type_name_iter->second.c_str();
  }
}

ResourceMgr::ResourceAndName::ResourceAndName() : name(nullptr) {}

ResourceMgr::ResourceAndName::ResourceAndName(const std::string& name)
  : name(std::make_unique<std::string>(name)) {}

std::shared_ptr<ResourceBase> ResourceMgr::ResourceAndName::GetResource()
  const {
  if (std::holds_alternative<std::shared_ptr<ResourceBase>>(resource)) {
    ResourceBase* ptr = std::get<std::shared_ptr<ResourceBase>>(resource).get();
    return std::shared_ptr<ResourceBase>(ptr);
  } else if (std::holds_alternative<std::weak_ptr<ResourceBase>>(resource)) {
    if (std::shared_ptr<ResourceBase> res =
          std::get<std::weak_ptr<ResourceBase>>(resource).lock()) {
      return res;
    } else {
      return nullptr;
    }
  } else {
    return nullptr;
  }
}

ResourceMgr::ResourceAndName::ResourceAndName(
  ResourceAndName&& other) noexcept {
  name = std::move(other.name);
  resource = std::move(other.resource);
}

ResourceMgr::ResourceAndName::~ResourceAndName() {}

ResourceMgr::ResourceAndName& ResourceMgr::ResourceAndName::operator=(
  ResourceAndName&& other) noexcept {
  name = std::move(other.name);
  resource = std::move(other.resource);
  return *this;
}

ResourceMgr::ResourceMgr()
  : default_container_("localhost") {}

ResourceMgr::ResourceMgr(const std::string& default_container)
  : default_container_(default_container) {}

ResourceMgr::~ResourceMgr() { Clear(); }

void ResourceMgr::Clear() {
  // We do the deallocation outside of the lock to avoid a potential deadlock
  // in case any of the destructors access the resource manager.
  std::pmr::unordered_map<std::string, Container*> tmp_containers;
  {
    const std::lock_guard<std::shared_mutex> lock(mu_);
    tmp_containers = std::move(containers_);
    containers_.clear();  // reinitialize after move.
  }
  for (const auto& p : tmp_containers) {
    delete p.second;
  }
}

std::string ResourceMgr::DebugString() const {
  const std::lock_guard<std::shared_mutex> lock(mu_);
  struct Line {
    const std::string* container;
    const std::string type;
    const std::string* resource;
    const std::string detail;
  };
  std::vector<Line> lines;
  for (const auto& p : containers_) {
    const std::string& container = p.first;
    for (const auto& q : *p.second) {
      const Key& key = q.first;
      const char* type = DebugTypeName(key.first);
      const std::shared_ptr<ResourceBase> resource = q.second.GetResource();
      Line l{&container, std::string(type, std::strlen(type)),
             q.second.name.get(),
             resource ? resource->DebugString() : "<nullptr>"};
      lines.emplace_back(l);
    }
  }
  std::vector<std::string> text;
  std::string_view pattern("{:20s} | {:40s} | {:40s} | {:s}");
  text.reserve(lines.size());
  for (const Line& line : lines) {
    text.emplace_back(strings::StrFormat(pattern, line.container, line.type,
                                         line.resource, line.detail));
  }
  std::sort(text.begin(), text.end());
  std::string result;
  for (auto& line : text) {
    result.append(line);
  }
  result.append("\n");
  return std::move(result);
}

std::error_code ResourceMgr::DoCreate(const std::string& container_name,
                                      TypeIndex type, const std::string& name,
                                      std::shared_ptr<ResourceBase> resource,
                                      bool owns_resource) {
  Container* container = [&]() ME_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    Container** ptr = &containers_[container_name];
    if (*ptr == nullptr) {
      *ptr = new Container{&mem_resource_};
    }
    return *ptr;
  }();

  // NOTE: Separating out the construction of the map key and value so that the
  // key can contain a std::string_view that borrows from the string in the
  // value.
  ResourceAndName resource_and_name(name);

  std::string_view borrowed_name(*resource_and_name.name);

  if (owns_resource) {
    resource_and_name.resource = std::shared_ptr<ResourceBase>(resource);
  } else {
    auto cleanup_fn = [this, container, type, borrowed_name]() {
      const std::lock_guard<std::shared_mutex> lock(mu_);
      auto iter = container->find({type.hash_code(), borrowed_name});
      if (iter != container->end()) {
        container->erase(iter);
      }
    };
    resource_and_name.resource = std::weak_ptr<ResourceBase>(resource);
  }

  Container::value_type key_and_value(Key(type.hash_code(), borrowed_name),
                                      std::move(resource_and_name));

  auto st = container->insert(std::move(key_and_value));
  if (st.second) {
    ME_RETURN_IF_ERROR(InsertDebugTypeName(type.hash_code(), type.name()));
    return make_error_code(Status::OK);
  }
  const auto logger = GetDefaultLogger();
  LOG_ERROR(logger, "Resource {}/{}/{} Already exists.", container_name, name,
            type.name());
  return make_error_code(Status::ALREADY_EXISTS);
}

std::error_code ResourceMgr::DoLookup(
  const std::string& container, TypeIndex type, const std::string& name,
  std::shared_ptr<ResourceBase>* resource) const {
  return DoLookup(container, type.hash_code(), type.name(), name, resource);
}

std::error_code ResourceMgr::DoLookup(
  const std::string& container, uint64_t type_hash_code,
  const std::string& type_name, const std::string& resource_name,
  std::shared_ptr<ResourceBase>* resource) const {
  Container* ctn_ptr = nullptr;
  if (auto search = containers_.find(container); search != containers_.end())
    ctn_ptr = search->second;
  if (ctn_ptr == nullptr) {
    const auto logger = GetDefaultLogger();
    LOG_ERROR(logger,
              "Container {} does not exist. (Could not find resource: {}/{})",
              container, container, resource_name);
    return make_error_code(Status::NOT_FOUND);
  }
  auto iter = ctn_ptr->find({type_hash_code, resource_name});
  if (iter == ctn_ptr->end()) {
    const auto logger = GetDefaultLogger();
    LOG_ERROR(logger, "Resource {}/{}/{} does not exist.", container,
              resource_name, type_name);
    return make_error_code(Status::NOT_FOUND);
  }
  std::shared_ptr<ResourceBase> ptr = std::move(iter->second.GetResource());
  if (ptr == nullptr) {
    const auto logger = GetDefaultLogger();
    LOG_ERROR(logger, "Resource {}/{}/{} has been destroyed.", container,
              resource_name, type_name);
    return make_error_code(Status::NOT_FOUND);
  }
  *resource = ptr;
  return make_error_code(Status::OK);
}

std::error_code ResourceMgr::PopResourceAndName(
  const std::string& container, uint64_t type_hash_code,
  const std::string& resource_name, const std::string& type_name,
  ResourceAndName& resource_and_name) {
  const std::lock_guard<std::shared_mutex> lock(mu_);
  Container* ctn_ptr = nullptr;
  if (auto search = containers_.find(container); search != containers_.end())
    ctn_ptr = search->second;
  if (ctn_ptr == nullptr) {
    const auto logger = GetDefaultLogger();
    LOG_ERROR(logger, "Container {} does not exist.", container, container,
              resource_name);
    return make_error_code(Status::NOT_FOUND);
  }
  auto iter = ctn_ptr->find({type_hash_code, resource_name});
  if (iter == ctn_ptr->end()) {
    const auto logger = GetDefaultLogger();
    LOG_ERROR(logger, "Resource {}/{}/{} does not exist.", container,
              resource_name, type_name);
    return make_error_code(Status::NOT_FOUND);
  }
  std::swap(resource_and_name, iter->second);
  ctn_ptr->erase(iter);
  return make_error_code(Status::OK);
}

std::error_code ResourceMgr::DoDelete(const std::string& container,
                                      uint64_t type_hash_code,
                                      const std::string& resource_name,
                                      const std::string& type_name) {
  ResourceAndName resource_and_name;
  ME_RETURN_IF_ERROR(PopResourceAndName(
    container, type_hash_code, resource_name, type_name, resource_and_name));

  if (std::holds_alternative<std::weak_ptr<ResourceBase>>(
        resource_and_name.resource)) {
    const auto logger = GetDefaultLogger();
    LOG_ERROR(logger,
              "Cannot delete an unowned Resource {}/{}/{} from ResourceMgr. "
              "This indicates ref-counting ResourceHandle is exposed to weak "
              "ResourceHandle code paths.",
              container, resource_name, type_name);
    return make_error_code(Status::INTERNAL);
  }
  return make_error_code(Status::OK);
}

std::error_code ResourceMgr::DoDelete(const std::string& container,
                                      TypeIndex type,
                                      const std::string& resource_name) {
  return DoDelete(container, type.hash_code(), resource_name, type.name());
}

std::error_code ResourceMgr::Cleanup(const std::string& container) {
  {
    mutex::shared_lock_guard lock(mu_);
    if (!containers_.contains(container)) {
      // Nothing to cleanup.
      return make_error_code(Status::OK);
    }
  }
  Container* ctn_ptr = nullptr;
  {
    std::lock_guard<std::shared_mutex> lock(mu_);
    auto iter = containers_.find(container);
    if (iter == containers_.end()) {
      // Nothing to cleanup, it's OK (concurrent cleanup).
      return make_error_code(Status::OK);
    }
    ctn_ptr = iter->second;
    containers_.erase(iter);
  }
  DEFAULT_LOGGER_CHECK(ctn_ptr != nullptr);
  delete ctn_ptr;
  return make_error_code(Status::OK);
}
}  // namespace meepo_embedding