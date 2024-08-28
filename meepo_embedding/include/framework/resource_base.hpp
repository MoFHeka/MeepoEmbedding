/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
==============================================================================*/

#pragma once

#ifndef MEEPO_EMBEDDING_FRAMEWORK_RESOURCE_BASE_H_
#define MEEPO_EMBEDDING_FRAMEWORK_RESOURCE_BASE_H_

#include "meepo_embedding/include/common/strings.h"

#include <string>

namespace meepo_embedding {

class ResourceBase {
 public:
  // Returns a debug string for *this.
  virtual std::string DebugString() const = 0;

  // Returns a name for ref-counting handles.
  virtual std::string MakeRefCountingHandleName(int64_t resource_id) const {
    return strings::StrFormat("Resource-{}-at-{}", resource_id, this);
  }

  // Returns memory used by this resource.
  virtual int64_t MemoryUsed() const { return 0; }
};

}  // namespace meepo_embedding

#endif  // MEEPO_EMBEDDING_FRAMEWORK_RESOURCE_BASE_H_