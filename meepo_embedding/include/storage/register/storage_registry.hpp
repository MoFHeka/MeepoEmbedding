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

#pragma once

#ifndef MEEPO_EMBEDDING_STORAGE_REGISTRY_H_
#define MEEPO_EMBEDDING_STORAGE_REGISTRY_H_

#include <algorithm>
#include <map>
#include <memory>
#include <string>

#include "meepo_embedding/include/common/data_type.h"
#include "meepo_embedding/include/common/device_base.h"
#include "meepo_embedding/include/storage/register/storage_interface.hpp"
#include "meepo_embedding/include/storage/register/storage_registration.hpp"

namespace meepo_embedding {
namespace storage {
namespace registry {

class StorageFactory {
 public:
  virtual pro::proxy<StorageInterface> Create() = 0;
  virtual ~StorageFactory() = default;
};  // class StorageFactory

class StorageRegistry {
 private:
  std::map<std::string, std::unique_ptr<StorageFactory>> DeferRegistrationData_;

 private:
  struct StorageFactoryImpl : public StorageFactory {
    explicit StorageFactoryImpl(pro::proxy<StorageInterface> (*create_func)())
      : create_func_(create_func) {}

    pro::proxy<StorageInterface> Create() override;

    pro::proxy<StorageInterface> (*create_func_)();

  };  // struct StorageFactoryImpl

  StorageRegistry() {}

 public:
  ~StorageRegistry() {}

  void DeferRegister(const DeviceType&& device_type,
                     const std::string&& cls_name,
                     pro::proxy<StorageInterface>(create_fn)());

  pro::proxy<StorageInterface> LookUp(const std::string&& factory_key);

  static StorageRegistry* Global() {
    static registry::StorageRegistry me_global_registry;
    return &me_global_registry;
  }
};

// REGISTER_STORAGE_IMPL_2, with a unique 'ctr' as the first argument.
#define REGISTER_STORAGE_IMPL_3(                                             \
  ctr, device_type, ktype, vtype, stype, cls_name, cls_type)                 \
  static meepo_embedding::storage::InitOnStartupMarker const storage_##ctr   \
    ME_ATTRIBUTE_UNUSED =                                                    \
      ME_INIT_ON_STARTUP_IF(ME_SHOULD_REGISTER_STORAGE(cls_name)) << ([]() { \
        ::meepo_embedding::storage::registry::StorageRegistry::Global()      \
          ->DeferRegister(                                                   \
            device_type,                                                     \
            ktype,                                                           \
            vtype,                                                           \
            stype,                                                           \
            cls_name,                                                        \
            []() -> pro::proxy<meepo_embedding::storage::StorageInterface> { \
              return pro::make_proxy<                                        \
                meepo_embedding::storage::StorageInterface,                  \
                cls_type>();                                                 \
            });                                                              \
        return meepo_embedding::storage::InitOnStartupMarker{};              \
      })();

#define REGISTER_STORAGE_IMPL_2(...) \
  ME_NEW_ID_FOR_INIT(REGISTER_STORAGE_IMPL_3, __VA_ARGS__)

#define REGISTER_STORAGE_IMPL(device_type, ktype, vtype, stype, ...)           \
  static_assert(std::is_default_constructible<__VA_ARGS__>::value,             \
                "Meepo Embedding storage backend must has a default "          \
                "constructor with empty parameters!");                         \
  static_assert(std::is_same_v(std::decay_t<decltype(device_type)>,            \
                               meepo_embedding::DeviceType),                   \
                "The first parameter of REGISTER_STORAGE macro should be the " \
                "type meepo_embedding::DeviceType.");                          \
  static_assert(                                                               \
    std::is_same_v(std::decay_t<decltype(ktype)>, meepo_embedding::DataType)   \
      && std::is_same_v(std::decay_t<decltype(vtype)>,                         \
                        meepo_embedding::DataType)                             \
      && std::is_same_v(std::decay_t<decltype(stype)>,                         \
                        meepo_embedding::DataType),                            \
    "The second, third and fourth parameters of REGISTER_STORAGE macro "       \
    "should be the type meepo_embedding::DataType.");                          \
  REGISTER_STORAGE_IMPL_2(                                                     \
    device_type, ktype, vtype, stype, ME_TYPE_NAME(__VA_ARGS__), __VA_ARGS__)

#define REGISTER_STORAGE(...) REGISTER_STORAGE_IMPL(__VA_ARGS__)

}  // namespace registry
}  // namespace storage
}  // namespace meepo_embedding

#endif  // MEEPO_EMBEDDING_STORAGE_REGISTRY_H_