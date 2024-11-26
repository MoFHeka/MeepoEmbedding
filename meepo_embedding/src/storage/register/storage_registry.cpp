/*Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

#include "meepo_embedding/include/storage/register/storage_registry.hpp"

#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "meepo_embedding/include/common/strings.h"

namespace meepo_embedding {
namespace storage {
namespace registry {

void StorageRegistry::DeferRegister(const DeviceType device_type,
                                    const std::string_view cls_name,
                                    pro::proxy<StorageInterface>(create_fn)()) {
  auto constructor = std::make_unique<StorageFactoryImpl>(create_fn);
  auto key =
    meepo_embedding::strings::CreateStorageFactoryKey(device_type, cls_name);
  DeferRegistrationData_.insert(std::make_pair(key, std::move(constructor)));
}

void StorageRegistry::DeferRegister(const std::string_view device,
                                    const std::string_view cls_name,
                                    pro::proxy<StorageInterface>(create_fn)()) {
  auto constructor = std::make_unique<StorageFactoryImpl>(create_fn);
  auto key =
    meepo_embedding::strings::CreateStorageFactoryKey(device, cls_name);
  DeferRegistrationData_.insert(std::make_pair(key, std::move(constructor)));
}

void StorageRegistry::DeferRegister(const DeviceType device_type,
                                    const DataType key_dtype,
                                    const DataType value_dtype,
                                    const DataType score_dtype,
                                    const std::string_view cls_name,
                                    pro::proxy<StorageInterface>(create_fn)()) {
  auto constructor = std::make_unique<StorageFactoryImpl>(create_fn);
  auto key = meepo_embedding::strings::CreateStorageFactoryKey(
    device_type, key_dtype, value_dtype, score_dtype, cls_name);
  DeferRegistrationData_.insert(std::make_pair(key, std::move(constructor)));
}

pro::proxy<StorageInterface> StorageRegistry::LookUp(
  const std::string& factory_key) {
  auto pair_found = DeferRegistrationData_.find(factory_key);
  if (pair_found == DeferRegistrationData_.end()) {
    return pro::proxy<StorageInterface>();
  } else {
    return pair_found->second->Create();
  }
}

pro::proxy<StorageInterface> StorageRegistry::LookUp(
  const DeviceType device_type, const std::string_view cls_name) {
  auto factory_key =
    meepo_embedding::strings::CreateStorageFactoryKey(device_type, cls_name);
  auto pair_found = DeferRegistrationData_.find(factory_key);
  if (pair_found == DeferRegistrationData_.end()) {
    return pro::proxy<StorageInterface>();
  } else {
    return pair_found->second->Create();
  }
}

pro::proxy<StorageInterface> StorageRegistry::LookUp(
  const std::string_view device, const std::string_view cls_name) {
  auto factory_key =
    meepo_embedding::strings::CreateStorageFactoryKey(device, cls_name);
  auto pair_found = DeferRegistrationData_.find(factory_key);
  if (pair_found == DeferRegistrationData_.end()) {
    return pro::proxy<StorageInterface>();
  } else {
    return pair_found->second->Create();
  }
}

pro::proxy<StorageInterface> StorageRegistry::LookUp(
  const DeviceType device_type,
  const DataType key_dtype,
  const DataType value_dtype,
  const DataType score_dtype,
  const std::string_view cls_name) {
  auto factory_key = meepo_embedding::strings::CreateStorageFactoryKey(
    device_type, key_dtype, value_dtype, score_dtype, cls_name);
  auto pair_found = DeferRegistrationData_.find(factory_key);
  if (pair_found == DeferRegistrationData_.end()) {
    return pro::proxy<StorageInterface>();
  } else {
    return pair_found->second->Create();
  }
}

std::size_t StorageRegistry::Size() { return DeferRegistrationData_.size(); }

pro::proxy<StorageInterface> StorageRegistry::StorageFactoryImpl::Create() {
  return (*create_func_)();
}

}  // namespace registry
}  // namespace storage
}  // namespace meepo_embedding
