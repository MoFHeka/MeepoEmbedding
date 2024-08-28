/*Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

#ifndef MEEPO_EMBEDDING_COMMON_DEVICES_H_
#define MEEPO_EMBEDDING_COMMON_DEVICES_H_

#include "meepo_embedding/include/third_party/magic_enum.hpp"

namespace meepo_embedding {
enum class DeviceType { CPU = 1, GPU = 2, TPU = 3 };

using namespace magic_enum::bitwise_operators;

}  // namespace meepo_embedding

#endif  // MEEPO_EMBEDDING_COMMON_DEVICES_H_