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

#include "meepo_embedding/include/common/logger.h"

#include <execinfo.h>

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

namespace meepo_embedding {

static quill::Logger* default_logger = nullptr;

void InitDefaultLogger(const bool verbose) {
  if (default_logger) {
    return;
  }

  quill::ConsoleColours consoleColors;
  consoleColors.set_default_colours();
  consoleColors.set_colour(quill::LogLevel::Info, quill::ConsoleColours::white);
  auto sink = quill::Frontend::create_or_get_sink<quill::ConsoleSink>(
    "console", consoleColors);
  const char* log_pattern =
    "<%(log_level_short_code)>%(time) %(short_source_location): %(message)";
  const char* timestamp_format = "%H:%M:%S.%Qms";
  default_logger = quill::Frontend::create_or_get_logger(
    "default", std::move(sink), log_pattern, timestamp_format);
  if (verbose) {
    default_logger->set_log_level(quill::LogLevel::Debug);
  }
  quill::BackendOptions options;
  options.thread_name = "MeepoEmbeddingDefaultLog";
  quill::Backend::start(options);
}

quill::Logger* GetDefaultLogger() { return default_logger; }

void FlushDefaultLogger() {
  if (default_logger) {
    default_logger->flush_log();
  }
}

std::string CurrentStackTrace() {
  const int maxFrames = 64;
  void* frames[maxFrames];
  int numFrames = backtrace(frames, maxFrames);
  char** symbols = backtrace_symbols(frames, numFrames);

  std::ostringstream oss;
  oss << "Stack trace:\n";
  for (int i = 0; i < numFrames; ++i) {
    oss << symbols[i] << "\n";
  }

  free(symbols);
  return oss.str();
}

}  // namespace meepo_embedding
