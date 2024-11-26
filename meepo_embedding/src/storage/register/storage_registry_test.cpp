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

#include "meepo_embedding/include/storage/register/storage_registry.hpp"

#include <gtest/gtest.h>    // from @googletest
#include <yaml-cpp/yaml.h>  // from @yaml-cpp

#include <algorithm>
#include <map>
#include <system_error>

#include "meepo_embedding/include/common/tensor.h"
#include "meepo_embedding/include/storage/register/storage_interface.hpp"

namespace mp = meepo_embedding;
namespace mps = meepo_embedding::storage;

TEST(StorageRegistryTest, Basic) {
  // Define a simple KV storage class following storage_interface.hpp
  class SimpleKVStorage {
   private:
    std::map<int64_t, float> kv_storage;

   public:
    SimpleKVStorage() = default;

    ~SimpleKVStorage() = default;

    // Initialization function
    std::error_code init(const YAML::Node &config) {
      kv_storage.clear();
      return std::error_code{};
    }

    // Insert function, following the definition of assign
    std::error_code assign(const std::size_t n_keys,
                           const mp::Tensor<int64_t> &keys,
                           const mp::Tensor<float> &values,
                           const mp::Tensor<uint64_t> &scores) {
      auto k_ptr = static_cast<int64_t *>(keys.data);
      auto v_ptr = static_cast<float *>(values.data);
      for (size_t i = 0; i < n_keys; ++i) {
        kv_storage[*(k_ptr + i)] = *(v_ptr + i);
      }
      return std::error_code{};
    }

    // Find function, following the definition of find
    std::error_code find(const std::size_t n_keys,
                         const mp::Tensor<int64_t> &keys,
                         mp::Tensor<float> &values) const {
      auto k_ptr = static_cast<int64_t *>(keys.data);
      auto v_ptr = static_cast<float *>(values.data);
      for (size_t i = 0; i < n_keys; ++i) {
        auto it = kv_storage.find(*(k_ptr + i));
        if (it != kv_storage.end()) {
          *(v_ptr + i) = it->second;
        } else {
          return std::make_error_code(std::errc::no_such_file_or_directory);
        }
      }
      return std::error_code{};
    }
  };

  // Register SimpleKVStorage
  ASSERT_EQ(mps::registry::StorageRegistry::Global()->Size(), 0);
  REGISTER_STORAGE(mp::DeviceType::CPU, SimpleKVStorage);
  ASSERT_EQ(mps::registry::StorageRegistry::Global()->Size(), 1);

  // Test lookup of KV storage from registry and perform simple insert and
  // lookup
  auto kv_storage =
    mps::registry::StorageRegistry::Global()->LookUp("CPU", "SimpleKVStorage");
  ASSERT_TRUE(kv_storage.has_value());

  mp::Tensor<int64_t> keys;
  mp::Tensor<float> values;
  mp::Tensor<uint64_t> scores;

  // Initialize tensors
  keys.data = new int64_t[2]{1, 2};
  values.data = new float[2]{1.0f, 2.0f};
  scores.data = new uint64_t[2]{1, 1};

  ASSERT_EQ(kv_storage->init(YAML::Node()), std::error_code{});
  ASSERT_EQ(kv_storage->assign(2, keys, values, scores), std::error_code{});

  mp::Tensor<float> lookup_values;
  lookup_values.data = new float[2];

  ASSERT_EQ(kv_storage->find(2, keys, lookup_values), std::error_code{});
  ASSERT_EQ(
    static_cast<float *>(lookup_values.data)[0],
    static_cast<float *>(values.data)[0]);  // Ensure the value is the same

  // Clean up
  delete[] static_cast<int64_t *>(keys.data);
  delete[] static_cast<float *>(values.data);
  delete[] static_cast<uint64_t *>(scores.data);
  delete[] static_cast<float *>(lookup_values.data);
}  // namespace mp::storage
