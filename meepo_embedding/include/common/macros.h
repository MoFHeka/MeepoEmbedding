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

#ifndef MEEPO_EMBEDDING_COMMON_MACROS_H_
#define MEEPO_EMBEDDING_COMMON_MACROS_H_

//==============================================================================
// Thread safety attribute
#if defined(__clang__) && (!defined(SWIG))
#define ME_INTERNAL_THREAD_ANNOTATION_ATTRIBUTE(x) __attribute__((x))
#else
#define ME_INTERNAL_THREAD_ANNOTATION_ATTRIBUTE(x)  // no-op
#endif

// Turns off thread safety checking within the body of a particular function.
// This is used as an escape hatch for cases where either (a) the function
// is correct, but the locking is more complicated than the analyzer can handle,
// or (b) the function contains race conditions that are known to be benign.
#define ME_NO_THREAD_SAFETY_ANALYSIS \
  ME_INTERNAL_THREAD_ANNOTATION_ATTRIBUTE(no_thread_safety_analysis)

// Document if a shared variable/field needs to be protected by a mutex.
// ME_GUARDED_BY allows the user to specify a particular mutex that should be
// held when accessing the annotated variable.  GUARDED_VAR indicates that
// a shared variable is guarded by some unspecified mutex, for use in rare
// cases where a valid mutex expression cannot be specified.
#define ME_GUARDED_BY(x) ME_INTERNAL_THREAD_ANNOTATION_ATTRIBUTE(guarded_by(x))

// Document a function that expects a mutex to be held prior to entry.
// The mutex is expected to be held both on entry to and exit from the
// function.
#define ME_EXCLUSIVE_LOCKS_REQUIRED(...) \
  ME_INTERNAL_THREAD_ANNOTATION_ATTRIBUTE(exclusive_locks_required(__VA_ARGS__))

#define ME_SHARED_LOCKS_REQUIRED(...) \
  ME_INTERNAL_THREAD_ANNOTATION_ATTRIBUTE(shared_locks_required(__VA_ARGS__))

//==============================================================================

// ERROR check macro
#define ME_RETURN_IF_ERROR(...)                                              \
  do {                                                                       \
    auto stat = (__VA_ARGS__);                                               \
    if (ME_PREDICT_FALSE(static_cast<Status>(stat.value()) != Status::OK)) { \
      return stat;                                                           \
    }                                                                        \
  } while (0)

// CHECK dies with a fatal error if condition is not true.  It is *not*
// controlled by NDEBUG, so the check will be executed regardless of
// compilation mode.  Therefore, it is safe to do things like:
//    CHECK(fp->Write(x) == 4)
#define LOGGER_CHECK(logger, condition, ...)                           \
  if (ME_PREDICT_FALSE(!(condition))) {                                \
    LOG_CRITICAL(logger, "Check failed: " #condition " " __VA_ARGS__); \
  }
#define DEFAULT_LOGGER_CHECK(condition, ...)                           \
  if (ME_PREDICT_FALSE(!(condition))) {                                \
    const auto logger = meepo_embedding::GetDefaultLogger();           \
    LOG_CRITICAL(logger, "Check failed: " #condition " " __VA_ARGS__); \
  }

// Compuler simple type name
#define ME_TYPE_NAME(type) #type

// Compiler attributes
#if (defined(__GNUC__) || defined(__APPLE__)) && !defined(SWIG)
// Compiler supports GCC-style attributes
#define ME_ATTRIBUTE_NORETURN __attribute__((noreturn))
#define ME_ATTRIBUTE_ALWAYS_INLINE __attribute__((always_inline))
#define ME_ATTRIBUTE_NOINLINE __attribute__((noinline))
#define ME_ATTRIBUTE_UNUSED __attribute__((unused))
#define ME_ATTRIBUTE_COLD __attribute__((cold))
#define ME_ATTRIBUTE_WEAK __attribute__((weak))
#define ME_PACKED __attribute__((packed))
#define ME_MUST_USE_RESULT __attribute__((warn_unused_result))
#define ME_PRINME_ATTRIBUTE(string_index, first_to_check) \
  __attribute__((__format__(__printf__, string_index, first_to_check)))
#define ME_SCANF_ATTRIBUTE(string_index, first_to_check) \
  __attribute__((__format__(__scanf__, string_index, first_to_check)))
#elif defined(_MSC_VER)
// Non-GCC equivalents
#define ME_ATTRIBUTE_NORETURN __declspec(noreturn)
#define ME_ATTRIBUTE_ALWAYS_INLINE __forceinline
#define ME_ATTRIBUTE_NOINLINE
#define ME_ATTRIBUTE_UNUSED
#define ME_ATTRIBUTE_COLD
#define ME_ATTRIBUTE_WEAK
#define ME_MUST_USE_RESULT
#define ME_PACKED
#define ME_PRINME_ATTRIBUTE(string_index, first_to_check)
#define ME_SCANF_ATTRIBUTE(string_index, first_to_check)
#else
// Non-GCC equivalents
#define ME_ATTRIBUTE_NORETURN
#define ME_ATTRIBUTE_ALWAYS_INLINE
#define ME_ATTRIBUTE_NOINLINE
#define ME_ATTRIBUTE_UNUSED
#define ME_ATTRIBUTE_COLD
#define ME_ATTRIBUTE_WEAK
#define ME_MUST_USE_RESULT
#define ME_PACKED
#define ME_PRINME_ATTRIBUTE(string_index, first_to_check)
#define ME_SCANF_ATTRIBUTE(string_index, first_to_check)
#endif

// Control visibility outside .so
#if defined(_WIN32)
#ifdef ME_COMPILE_LIBRARY
#define ME_EXPORT __declspec(dllexport)
#else
#define ME_EXPORT __declspec(dllimport)
#endif  // ME_COMPILE_LIBRARY
#else
#define ME_EXPORT __attribute__((visibility("default")))
#endif  // _WIN32

#ifdef __has_builtin
#define ME_HAS_BUILTIN(x) __has_builtin(x)
#else
#define ME_HAS_BUILTIN(x) 0
#endif

// C++11-style attributes (N2761)
#if defined(__has_cpp_attribute)
// Safely checks if an attribute is supported. Equivalent to
// ABSL_HAVE_CPP_ATTRIBUTE.
#define ME_HAS_CPP_ATTRIBUTE(n) __has_cpp_attribute(n)
#else
#define ME_HAS_CPP_ATTRIBUTE(n) 0
#endif

// [[clang::annotate("x")]] allows attaching custom strings (e.g. "x") to
// declarations (variables, functions, fields, etc.) for use by tools. They are
// represented in the Clang AST (as AnnotateAttr nodes) and in LLVM IR, but not
// in final output.
#if ME_HAS_CPP_ATTRIBUTE(clang::annotate)
#define ME_ATTRIBUTE_ANNOTATE(str) [[clang::annotate(str)]]
#else
#define ME_ATTRIBUTE_ANNOTATE(str)
#endif

// A variable declaration annotated with the `ME_CONST_INIT` attribute will
// not compile (on supported platforms) unless the variable has a constant
// initializer.
#if ME_HAS_CPP_ATTRIBUTE(clang::require_constant_initialization)
#define ME_CONST_INIT [[clang::require_constant_initialization]]
#else
#define ME_CONST_INIT
#endif

// Compilers can be told that a certain branch is not likely to be taken
// (for instance, a CHECK failure), and use that information in static
// analysis. Giving it this information can help it optimize for the
// common case in the absence of better information (ie.
// -fprofile-arcs).
#if ME_HAS_BUILTIN(__builtin_expect) || (defined(__GNUC__) && __GNUC__ >= 3)
#define ME_PREDICT_FALSE(x) (__builtin_expect(x, 0))
#define ME_PREDICT_TRUE(x) (__builtin_expect(!!(x), 1))
#else
#define ME_PREDICT_FALSE(x) (x)
#define ME_PREDICT_TRUE(x) (x)
#endif

// DEPRECATED: directly use the macro implementation instead.
// A macro to disallow the copy constructor and operator= functions
// This is usually placed in the private: declarations for a class.
#define ME_DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete;         \
  void operator=(const TypeName&) = delete

// The ME_ARRAYSIZE(arr) macro returns the # of elements in an array arr.
//
// The expression ME_ARRAYSIZE(a) is a compile-time constant of type
// size_t.
#define ME_ARRAYSIZE(a)       \
  ((sizeof(a) / sizeof(*(a))) \
   / static_cast<size_t>(!(sizeof(a) % sizeof(*(a)))))

#if defined(__GXX_EXPERIMENTAL_CXX0X__) || __cplusplus >= 201103L \
  || (defined(_MSC_VER) && _MSC_VER >= 1900)
// Define this to 1 if the code is compiled in C++11 mode; leave it
// undefined otherwise.  Do NOT define it to 0 -- that causes
// '#ifdef LANG_CXX11' to behave differently from '#if LANG_CXX11'.
#define LANG_CXX11 1
#endif

#if defined(__clang__) && defined(LANG_CXX11) && defined(__has_warning)
#if __has_feature(cxx_attributes) && __has_warning("-Wimplicit-fallthrough")
#define ME_FALLTHROUGH_INTENDED [[clang::fallthrough]]  // NOLINT
#endif
#endif

#ifndef ME_FALLTHROUGH_INTENDED
#define ME_FALLTHROUGH_INTENDED \
  do {                          \
  } while (0)
#endif

#endif  // MEEPO_EMBEDDING_COMMON_MACROS_H_
