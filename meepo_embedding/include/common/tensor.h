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

#ifndef MEEPO_EMBEDDING_COMMON_TENSOR_H_
#define MEEPO_EMBEDDING_COMMON_TENSOR_H_

#include <dlpack/dlpack.h>  // from @dlpack

#include <concepts>
#include <cstdint>
#include <type_traits>

#if __STDCPP_FLOAT16_T__ == 1 or __STDCPP_BFLOAT16_T__ == 1
#include <stdfloat>
#endif

#if __STDCPP_FLOAT16_T__ != 1
#warning "16-bit standard float type required, use cuda_fp16.h instead."
#include <cuda_fp16.h>
#endif

#if __STDCPP_BFLOAT16_T__ != 1
#warning "16-bit standard bfloat type required, use cuda_bf16.h instead."
#include <cuda_bf16.h>
#endif

namespace meepo_embedding {
template <typename DType = std::nullptr_t>
struct Tensor : DLTensor {};

template <typename DType>
  requires std::integral<DType> && std::is_signed_v<DType>
struct Tensor<DType> : DLTensor {
  static constexpr DLDataType dtype{DLDataTypeCode::kDLInt, sizeof(DType) * 8,
                                    1};
  DType std_dtype;
};

template <typename DType>
concept ValidUintType = std::integral<DType> && std::is_unsigned_v<DType>
                        && !std::same_as<DType, bool>;

template <typename DType>
  requires ValidUintType<DType>
struct Tensor<DType> : DLTensor {
  static constexpr DLDataType dtype{DLDataTypeCode::kDLUInt, sizeof(DType) * 8,
                                    1};
  DType std_dtype;
};

template <typename DType>
  requires std::floating_point<DType>
struct Tensor<DType> : DLTensor {
  static constexpr DLDataType dtype{DLDataTypeCode::kDLFloat, sizeof(DType) * 8,
                                    1};
  DType std_dtype;
};

template <typename DType>
concept ValidHalfType =
  !std::floating_point<DType> && !std::integral<DType>
  && !std::same_as<DType, bool> && !std::same_as<DType, std::nullptr_t>
  && sizeof(DType) == 2;

#if __STDCPP_FLOAT16_T__ == 1
template <typename DType>
  requires std::floating_point<DType> && std::same_as<DType, std::float16_t>
struct Tensor<DType> : DLTensor {
  static constexpr DLDataType dtype{DLDataTypeCode::kDLFloat, sizeof(DType) * 8,
                                    1};
  DType std_dtype;
};
typedef std::float16_t me_float16_t;
#else
template <typename DType>
  requires ValidHalfType<DType>
struct Tensor<DType> : DLTensor {
  static constexpr DLDataType dtype{DLDataTypeCode::kDLFloat, sizeof(DType) * 8,
                                    1};
  DType std_dtype;
};
typedef half me_float16_t;
#endif

#if __STDCPP_BFLOAT16_T__ == 1
template <typename DType>
  requires std::floating_point<DType> && std::same_as<DType, std::bfloat16_t>
struct Tensor<DType> : DLTensor {
  static constexpr DLDataType dtype{DLDataTypeCode::kDLBfloat,
                                    sizeof(DType) * 8, 1};
  DType std_dtype;
};
typedef std::bfloat16_t me_bfloat16_t;
#else
template <typename DType>
  requires ValidHalfType<DType>
struct Tensor<DType> : DLTensor {
  static constexpr DLDataType dtype{DLDataTypeCode::kDLBfloat,
                                    sizeof(DType) * 8, 1};
  DType std_dtype;
};
typedef __nv_bfloat16 me_bfloat16_t;
#endif

template <typename DType>
  requires std::same_as<DType, bool>
struct Tensor<DType> : DLTensor {
  static constexpr DLDataType dtype{DLDataTypeCode::kDLBool, sizeof(DType) * 8,
                                    1};
  DType std_dtype;
};

}  // namespace meepo_embedding

#endif  // MEEPO_EMBEDDING_COMMON_TENSOR_H_
