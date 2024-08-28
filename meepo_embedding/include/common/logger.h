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

#ifndef MEEPO_EMBEDDING_COMMON_LOGGER_H_
#define MEEPO_EMBEDDING_COMMON_LOGGER_H_

#include <quill/Backend.h>
#include <quill/Frontend.h>
#include <quill/LogMacros.h>
#include <quill/Logger.h>
#include <quill/sinks/ConsoleSink.h>

#include <string>

namespace meepo_embedding {
void InitDefaultLogger(const bool verbose = false);

quill::Logger* GetDefaultLogger();

std::string CurrentStackTrace();

void FlushDefaultLogger();
}  // namespace meepo_embedding

#endif  // MEEPO_EMBEDDING_COMMON_LOGGER_H_