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

#ifndef MEEPO_EMBEDDING_STORAGE_REGISTRATION_H_
#define MEEPO_EMBEDDING_STORAGE_REGISTRATION_H_

#include <type_traits>
#include <utility>

#include "meepo_embedding/include/common/macros.h"

// Compiler attributes
#if (defined(__GNUC__) || defined(__APPLE__)) && !defined(SWIG)
// Compiler supports GCC-style attributes
#define MEEPO_EMBEDDING_ATTRIBUTE_UNUSED __attribute__((unused))
#elif defined(_MSC_VER)
// Non-GCC equivalents
#define MEEPO_EMBEDDING_ATTRIBUTE_UNUSED
#else
// Non-GCC equivalents
#define MEEPO_EMBEDDING_ATTRIBUTE_UNUSED
#endif

// SELECTIVE_REGISTRATION (not supported now)
#define ME_SHOULD_REGISTER_STORAGE(cls) true

namespace meepo_embedding {
namespace storage {

// An InitOnStartupMarker is 'initialized' on program startup, purely for the
// side-effects of that initialization - the struct itself is empty. (The type
// is expected to be used to define globals.)
//
// The '<<' operator should be used in initializer expressions to specify what
// to run on startup. The following values are accepted:
//   - An InitOnStartupMarker. Example:
//      InitOnStartupMarker F();
//      InitOnStartupMarker const kInitF =
//        InitOnStartupMarker{} << F();
//   - Something to call, which returns an InitOnStartupMarker. Example:
//      InitOnStartupMarker const kInit =
//        InitOnStartupMarker{} << []() { G(); return
//
// See also: ME_INIT_ON_STARTUP_IF
struct InitOnStartupMarker {
  constexpr InitOnStartupMarker operator<<(InitOnStartupMarker) const {
    return *this;
  }

  template <typename T>
  constexpr InitOnStartupMarker operator<<(T&& v) const {
    return std::forward<T>(v)();
  }
};

// Conditional initializer expressions for InitOnStartupMarker:
//   ME_INIT_ON_STARTUP_IF(cond) << f
// If 'cond' is true, 'f' is evaluated (and called, if applicable) on startup.
// Otherwise, 'f' is *not evaluated*. Note that 'cond' is required to be a
// constant-expression, and so this approximates #ifdef.
//
// The implementation uses the ?: operator (!cond prevents evaluation of 'f').
// The relative precedence of ?: and << is significant; this effectively expands
// to (see extra parens):
//   !cond ? InitOnStartupMarker{} : (InitOnStartupMarker{} << f)
//
// Note that although forcing 'cond' to be a constant-expression should not
// affect binary size (i.e. the same optimizations should apply if it 'happens'
// to be one), it was found to be necessary (for a recent version of clang;
// perhaps an optimizer bug).
//
// The parens are necessary to hide the ',' from the preprocessor; it could
// otherwise act as a macro argument separator.
#define ME_INIT_ON_STARTUP_IF(cond)                     \
  (::std::integral_constant<bool, !(cond)>::value)      \
    ? ::meepo_embedding::storage::InitOnStartupMarker{} \
    : ::meepo_embedding::storage::InitOnStartupMarker {}

// Wrapper for generating unique IDs (for 'anonymous' InitOnStartup definitions)
// using __COUNTER__. The new ID (__COUNTER__ already expanded) is provided as a
// macro argument.
//
// Usage:
//   #define M_IMPL(id, a, b) ...
//   #define M(a, b) ME_NEW_ID_FOR_INIT(M_IMPL, a, b)
#define ME_NEW_ID_FOR_INIT_2(m, c, ...) m(c, __VA_ARGS__)
#define ME_NEW_ID_FOR_INIT_1(m, c, ...) ME_NEW_ID_FOR_INIT_2(m, c, __VA_ARGS__)
#define ME_NEW_ID_FOR_INIT(m, ...) \
  ME_NEW_ID_FOR_INIT_1(m, __COUNTER__, __VA_ARGS__)

}  // namespace storage
}  // namespace meepo_embedding

#endif  // MEEPO_EMBEDDING_STORAGE_REGISTRATION_H_