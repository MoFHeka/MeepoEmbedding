"""
Copyright 2024 The MeepoEmbedding Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================
"""

load("@rules_cuda//cuda:defs.bzl")

# Macros for building CUDA code.
def if_cuda(if_true, if_false = []):
    """Shorthand for select()'ing on whether we're building with CUDA.

    Returns a select statement which evaluates to if_true if we're building
    with CUDA enabled.  Otherwise, the select statement evaluates to if_false.
    """
    return select({
        "@rules_cuda//cuda:is_enabled": if_true,
        "//conditions:default": if_false,
    })

def if_cuda_nvcc(if_true, if_false = []):
    """Shorthand for select()'ing on wheteher we're building with cuda-nvcc.

    Returns a select statement which evaluates to if_true if we're building
    with cuda-nvcc.  Otherwise, the select statement evaluates to if_false.
    """
    return select({
        "@rules_cuda//cuda:compiler_is_nvcc": if_true,
        "//conditions:default": if_false,
    })

def if_cuda_clang(if_true, if_false = []):
    """Shorthand for select()'ing on wheteher we're building with cuda-clang.

      Returns a select statement which evaluates to if_true if we're building
      with cuda-clang.  Otherwise, the select statement evaluates to if_false.
    """
    return select({
        "@rules_cuda//cuda:compiler_is_clang": if_true,
        "//conditions:default": if_false,
    })
