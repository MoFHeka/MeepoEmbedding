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

#ifndef MEEPO_EMBEDDING_COMMON_DTYPE_H_
#define MEEPO_EMBEDDING_COMMON_DTYPE_H_

#include <magic_enum.hpp>

namespace meepo_embedding {
enum class DataType {
  ME_INT4 = 1,
  ME_UINT4 = 2,
  ME_INT8 = 3,
  ME_UINT8 = 4,
  ME_INT16 = 5,
  ME_UINT16 = 6,
  ME_INT32 = 7,  // Int32 tensors are always in 'host' memory.
  ME_UINT32 = 8,
  ME_INT64 = 9,
  ME_UINT64 = 10,
  ME_BOOL = 11,
  ME_HALF = 12,
  ME_BFLOAT16 = 13,
  ME_FLOAT = 14,
  ME_DOUBLE = 15,
  ME_FLOAT8_E5M2 = 16,    // 5 exponent bits, 2 mantissa bits.
  ME_FLOAT8_E4M3FN = 17,  // 4 exponent bits, 3 mantissa bits, finite-only, with
                          // 2 NaNs (0bS1111111).
  // TODO(MoFHeka): Leaving room for remaining float8 types.
  // ME_FLOAT8_E4M3FNUZ = 18,
  // ME_FLOAT8_E4M3B11FNUZ = 19,
  // ME_FLOAT8_E5M2FNUZ = 20,
};

}  // namespace meepo_embedding

#endif  // MEEPO_EMBEDDING_COMMON_DTYPE_H_
