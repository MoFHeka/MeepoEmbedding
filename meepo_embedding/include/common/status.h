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

#ifndef MEEPO_EMBEDDING_COMMON_STATUS_H_
#define MEEPO_EMBEDDING_COMMON_STATUS_H_

#include <map>
#include <string>
#include <system_error>

#include <magic_enum.hpp>

namespace meepo_embedding {
enum class Status {
  OK,
  ABORTED,
  ALREADY_EXISTS,
  CANCELLED,
  CONN_ERROR,
  DATA_LOSS,
  DEADLINE_EXCEEDED,
  FAILED_PRECONDITION,
  INTERNAL,
  INVALID_ARGUMENT,
  NOT_FOUND,
  OUT_OF_RANGE,
  PERMISSION_DENIED,
  RESOURCE_EXHAUSTED,
  UNAUTHENTICATED,
  UNAVAILABLE,
  UNIMPLEMENTED,
  UNKNOWN
};

class MEErrorCategory : public std::error_category {
 public:
  const char* name() const noexcept override { return "MEErrorCategory"; }

  std::string message(int errc) const override {
    static const std::map<int, std::string> messages = {
      {static_cast<int>(Status::OK), "OK"},
      {static_cast<int>(Status::ABORTED), "Aborted"},
      {static_cast<int>(Status::ALREADY_EXISTS), "Already Exists"},
      {static_cast<int>(Status::CANCELLED), "Cancelled"},
      {static_cast<int>(Status::CONN_ERROR), "Connection Error"},
      {static_cast<int>(Status::DATA_LOSS), "Data Loss"},
      {static_cast<int>(Status::DEADLINE_EXCEEDED), "Deadline Exceeded"},
      {static_cast<int>(Status::FAILED_PRECONDITION), "Failed Precondition"},
      {static_cast<int>(Status::INTERNAL), "Internal Error"},
      {static_cast<int>(Status::INVALID_ARGUMENT), "Invalid Argument"},
      {static_cast<int>(Status::NOT_FOUND), "Not Found"},
      {static_cast<int>(Status::OUT_OF_RANGE), "Out of Range"},
      {static_cast<int>(Status::PERMISSION_DENIED), "Permission Denied"},
      {static_cast<int>(Status::RESOURCE_EXHAUSTED), "Resource Exhausted"},
      {static_cast<int>(Status::UNAUTHENTICATED), "Unauthenticated"},
      {static_cast<int>(Status::UNAVAILABLE), "Unavailable"},
      {static_cast<int>(Status::UNIMPLEMENTED), "Unimplemented"},
      {static_cast<int>(Status::UNKNOWN), "Unknown Error"}};

    auto it = messages.find(errc);
    if (it != messages.end()) {
      return it->second;
    }
    return "Unknown Status";
  }

  static const MEErrorCategory& GetInstance() {
    static MEErrorCategory instance;
    return instance;
  }

 private:
  MEErrorCategory() = default;
};

inline std::error_code make_error_code(Status e) {
  return {static_cast<int>(e), MEErrorCategory::GetInstance()};
}

}  // namespace meepo_embedding

#endif  // MEEPO_EMBEDDING_COMMON_STATUS_H_
