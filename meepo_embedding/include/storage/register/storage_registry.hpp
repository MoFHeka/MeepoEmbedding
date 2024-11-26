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

  void DeferRegister(const DeviceType device_type,
                     const std::string_view cls_name,
                     pro::proxy<StorageInterface>(create_fn)());

  void DeferRegister(const std::string_view device,
                     const std::string_view cls_name,
                     pro::proxy<StorageInterface>(create_fn)());

  void DeferRegister(const DeviceType device_type,
                     const DataType key_dtype,
                     const DataType value_dtype,
                     const DataType score_dtype,
                     const std::string_view cls_name,
                     pro::proxy<StorageInterface>(create_fn)());

  pro::proxy<StorageInterface> LookUp(const std::string& factory_key);

  pro::proxy<StorageInterface> LookUp(const DeviceType device_type,
                                      const std::string_view cls_name);

  pro::proxy<StorageInterface> LookUp(const std::string_view device,
                                      const std::string_view cls_name);

  pro::proxy<StorageInterface> LookUp(const DeviceType device_type,
                                      const DataType key_dtype,
                                      const DataType value_dtype,
                                      const DataType score_dtype,
                                      const std::string_view cls_name);

  std::size_t Size();

  static StorageRegistry* Global() {
    static registry::StorageRegistry me_global_registry;
    return &me_global_registry;
  }
};

// REGISTER_STORAGE_IMPL_1, with a unique 'ctr' as the first argument.
#define REGISTER_STORAGE_IMPL_1(                                             \
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

#define REGISTER_STORAGE_IMPL_0(...) \
  ME_NEW_ID_FOR_INIT(REGISTER_STORAGE_IMPL_1, __VA_ARGS__)

#define REGISTER_STORAGE_IMPL(device_type, ktype, vtype, stype, cls)           \
  static_assert(std::is_default_constructible<cls>::value,                     \
                "Meepo Embedding storage backend must has a default "          \
                "constructor with empty parameters!");                         \
  static_assert(std::is_same_v<std::decay_t<decltype(device_type)>,            \
                               meepo_embedding::DeviceType>,                   \
                "The first parameter of REGISTER_STORAGE macro should be the " \
                "type meepo_embedding::DeviceType.");                          \
  static_assert(                                                               \
    std::is_same_v<std::decay_t<decltype(ktype)>, meepo_embedding::DataType>   \
      && std::is_same_v<std::decay_t<decltype(vtype)>,                         \
                        meepo_embedding::DataType>                             \
      && std::is_same_v<std::decay_t<decltype(stype)>,                         \
                        meepo_embedding::DataType>,                            \
    "The second, third and fourth parameters of REGISTER_STORAGE macro "       \
    "should be the type meepo_embedding::DataType.");                          \
  REGISTER_STORAGE_IMPL_0(                                                     \
    device_type, ktype, vtype, stype, ME_TYPE_NAME(cls), cls)

// REGISTER_STORAGE_MINI_IMPL_1, with a unique 'ctr' as the first argument.
#define REGISTER_STORAGE_MINI_IMPL_1(ctr, device_type, cls_name, cls_type)   \
  static meepo_embedding::storage::InitOnStartupMarker const storage_##ctr   \
    ME_ATTRIBUTE_UNUSED =                                                    \
      ME_INIT_ON_STARTUP_IF(ME_SHOULD_REGISTER_STORAGE(cls_name)) << ([]() { \
        ::meepo_embedding::storage::registry::StorageRegistry::Global()      \
          ->DeferRegister(                                                   \
            device_type,                                                     \
            cls_name,                                                        \
            []() -> pro::proxy<meepo_embedding::storage::StorageInterface> { \
              return pro::make_proxy<                                        \
                meepo_embedding::storage::StorageInterface,                  \
                cls_type>();                                                 \
            });                                                              \
        return meepo_embedding::storage::InitOnStartupMarker{};              \
      })();

#define REGISTER_STORAGE_MINI_IMPL_0(...) \
  ME_NEW_ID_FOR_INIT(REGISTER_STORAGE_MINI_IMPL_1, __VA_ARGS__)

#define REGISTER_STORAGE_MINI_IMPL(device_type, cls)                           \
  static_assert(std::is_default_constructible<cls>::value,                     \
                "Meepo Embedding storage backend must has a default "          \
                "constructor with empty parameters!");                         \
  static_assert(std::is_same_v<std::decay_t<decltype(device_type)>,            \
                               meepo_embedding::DeviceType>,                   \
                "The first parameter of REGISTER_STORAGE macro should be the " \
                "type meepo_embedding::DeviceType.");                          \
  REGISTER_STORAGE_MINI_IMPL_0(device_type, ME_TYPE_NAME(cls), cls)

#define GET_ARG_COUNT(_0, _1, _2, _3, _4, _5, count, ...) count
// When there is no content after ##, the space before is deleted
#define SELECT_MACRO(_0, _1, _2, _3, _4, _5, ...) \
  GET_ARG_COUNT(, ##__VA_ARGS__, _5, _4, _3, _2, _1, _0)
#define REGISTER_STORAGE(...)       \
  SELECT_MACRO(_REGISTER_STORAGE_0, \
               _REGISTER_STORAGE_1, \
               _REGISTER_STORAGE_2, \
               _REGISTER_STORAGE_3, \
               _REGISTER_STORAGE_4, \
               _REGISTER_STORAGE_5, \
               ##__VA_ARGS__)       \
  (__VA_ARGS__)

#define _REGISTER_STORAGE_0(cls) \
  static_assert(                 \
    false, "Macro REGISTER_STORAGE must pass at least one class declaration.")

#define _REGISTER_STORAGE_1(cls) \
  REGISTER_STORAGE_MINI_IMPL(meepo_embedding::DeviceType::CPU, cls)

#define _REGISTER_STORAGE_2(device_type, cls) \
  REGISTER_STORAGE_MINI_IMPL(device_type, cls)

#define _REGISTER_STORAGE_3(cls) \
  static_assert(false,           \
                "Macro REGISTER_STORAGE does not support three arguments.")

#define _REGISTER_STORAGE_4(cls) \
  static_assert(false,           \
                "Macro REGISTER_STORAGE does not support four arguments.")

#define _REGISTER_STORAGE_5(device_type, ktype, vtype, stype, cls) \
  REGISTER_STORAGE_IMPL(device_type, ktype, vtype, stype, cls)

}  // namespace registry
}  // namespace storage
}  // namespace meepo_embedding

#endif  // MEEPO_EMBEDDING_STORAGE_REGISTRY_H_
