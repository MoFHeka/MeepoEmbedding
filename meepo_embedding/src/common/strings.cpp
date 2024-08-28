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

#include "meepo_embedding/include/common/strings.h"

#include <iomanip>
#include <limits>
#include <sstream>
#include <type_traits>

#include "meepo_embedding/include/common/logger.h"
#include "meepo_embedding/include/common/macros.h"

namespace meepo_embedding {
namespace strings {

template <typename... Args>
std::string StrCat(Args&&... args) {
  constexpr const size_t count = sizeof...(Args);
  if constexpr (count == 1) {
    constexpr std::string_view str("{}");
    return std::vformat(str, std::make_format_args(args...));
  } else if constexpr (count == 2) {
    constexpr std::string_view str("{}{}");
    return std::vformat(str, std::make_format_args(args...));
  } else if constexpr (count == 3) {
    constexpr std::string_view str("{}{}{}");
    return std::vformat(str, std::make_format_args(args...));
  } else if constexpr (count == 4) {
    constexpr std::string_view str("{}{}{}{}");
    return std::vformat(str, std::make_format_args(args...));
  } else {
    std::string str;
    for (size_t i = 0; i < count; ++i) {
      str += "{}";
    }
    return std::vformat(str, std::make_format_args(args...));
  }
}

template <typename... Args>
std::string StrFormat(std::string_view rt_fmt_str, Args&&... args) {
  return std::vformat(rt_fmt_str, std::make_format_args(args...));
}

void ReserveOSS(std::ostringstream& oss, std::size_t size) {
  std::stringbuf* buf = oss.rdbuf();
  std::string* str = const_cast<std::string*>(&buf->str());
  str->reserve(size);
}

template <typename T>
void AppendToOSS(std::ostringstream& oss, const T& value) {
  oss << value;
}

template <typename... Args>
void StrAppend(std::string& buffer, Args&&... args) {
  std::ostringstream oss;
  constexpr const size_t count = sizeof...(Args);
  ReserveOSS(oss, count * 8);
  oss.rdbuf append_to_stream(oss, args...);
  buffer += oss.str();
}

template <typename T>
std::string HexString(T value) {
  std::ostringstream oss;
  oss << std::hex << std::uppercase << std::setw(sizeof(T) * 2)
      << std::setfill('0') << static_cast<uint64_t>(value);
  return oss.str();
}

std::string HumanReadableNumBytes(std::int64_t num_bytes) {
  if (num_bytes == std::numeric_limits<std::int64_t>::min()) {
    // Special case for number with not representable negation.
    return "-8E";
  }

  const char* neg_str = (num_bytes < 0) ? "-" : "";
  if (num_bytes < 0) {
    num_bytes = -num_bytes;
  }

  // Special case for bytes.
  if (num_bytes < 1024) {
    // No fractions for bytes.
    char buf[8];  // Longest possible string is '-XXXXB'
    snprintf(buf, sizeof(buf), "%s%lldB", neg_str,
             static_cast<long long>(num_bytes));
    return std::string(buf);
  }

  static const char units[] = "KMGTPE";  // int64 only goes up to E.
  const char* unit = units;
  const auto logger = meepo_embedding::GetDefaultLogger();
  while (num_bytes >= static_cast<std::int64_t>(1024) * 1024) {
    num_bytes /= 1024;
    ++unit;
    LOGGER_CHECK(logger, unit < units + ME_ARRAYSIZE(units));
  }

  // We use SI prefixes.
  char buf[16];
  snprintf(buf, sizeof(buf), ((*unit == 'K') ? "%s%.1f%ciB" : "%s%.2f%ciB"),
           neg_str, num_bytes / 1024.0, *unit);
  return std::string(buf);
}

const std::string CreateStorageFactoryKey(const std::string& device,
                                          const std::string& key_dtype,
                                          const std::string& value_dtype,
                                          const std::string& score_dtype,
                                          const std::string& cls_name) {
  std::string key = device + "_" + key_dtype + "_" + value_dtype + "_"
                    + score_dtype + "_" + cls_name;
  std::transform(key.begin(), key.end(), key.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  return std::move(key);
}

const std::string CreateStorageFactoryKey(const DeviceType& device_type,
                                          const DataType&& key_dtype,
                                          const DataType&& value_dtype,
                                          const DataType&& score_dtype,
                                          const std::string& cls_name) {
  auto cls_name_str = std::string(cls_name);
  std::transform(cls_name_str.begin(), cls_name_str.end(), cls_name_str.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  auto device_str = magic_enum::enum_name(device_type);
  auto ktype_str = magic_enum::enum_name(key_dtype);
  auto vtype_str = magic_enum::enum_name(value_dtype);
  auto stype_str = magic_enum::enum_name(score_dtype);
  std::string key(cls_name);
  key.reserve(key.size() + 32);
  key.insert(0, "_");
  key.insert(0, stype_str);
  key.insert(0, "_");
  key.insert(0, vtype_str);
  key.insert(0, "_");
  key.insert(0, ktype_str);
  key.insert(0, "_");
  key.insert(0, device_str);
  return std::move(key);
}
}  // namespace strings
}  // namespace meepo_embedding