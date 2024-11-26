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

#ifndef MEEPO_EMBEDDING_COMMON_STRINGS_H_
#define MEEPO_EMBEDDING_COMMON_STRINGS_H_

#include <format>
#include <string>
#include <string_view>

#include "meepo_embedding/include/common/data_type.h"
#include "meepo_embedding/include/common/device_base.h"

namespace meepo_embedding {
namespace strings {

template <typename... Args>
std::string StrCat(Args&&... args);

template <typename... Args>
std::string StrFormat(std::string_view rt_fmt_str, Args&&... args);

template <typename... Args>
void StrAppend(std::string& buffer, Args&&... args);

template <typename T>
  requires std::integral<T>
std::string HexString(T value);

std::string HumanReadableNumBytes(std::int64_t num_bytes);

const std::string CreateStorageFactoryKey(const DeviceType device_type,
                                          const DataType key_dtype,
                                          const DataType value_dtype,
                                          const DataType score_dtype,
                                          const std::string_view cls_name);

const std::string CreateStorageFactoryKey(const std::string_view device,
                                          const std::string_view key_dtype,
                                          const std::string_view value_dtype,
                                          const std::string_view score_dtype,
                                          const std::string_view cls_name);

const std::string CreateStorageFactoryKey(const std::string_view device,
                                          const std::string_view cls_name);

const std::string CreateStorageFactoryKey(const DeviceType device_type,
                                          const std::string_view cls_name);
}  // namespace strings
}  // namespace meepo_embedding

#endif  // MEEPO_EMBEDDING_COMMON_STRINGS_H_
