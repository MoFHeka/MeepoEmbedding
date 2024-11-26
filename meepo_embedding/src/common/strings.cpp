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

#include <inttypes.h>

#include <algorithm>
#include <cstdio>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include <magic_enum.hpp>

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
  std::string&& str = buf->str();
  str.reserve(size);
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
  AppendToOSS(oss, args...);
  buffer += oss.str();
}

void StrAppend(
  std::string& buffer, std::initializer_list<std::string_view> views) {
  std::ostringstream oss;
  for (const auto& view : views) {
    oss << view;
  }
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
    snprintf(buf, sizeof(buf), "%s%" PRId64 "B", neg_str,
             static_cast<int64_t>(num_bytes));
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

template <typename T>
std::size_t TotalStringLength(std::initializer_list<T> views) {
  std::size_t len = 0;
  for (const auto& view : views) {
    len += view.size();
  }
  return len;
}

const std::string CreateStorageFactoryKey(const std::string_view device,
                                          const std::string_view key_dtype,
                                          const std::string_view value_dtype,
                                          const std::string_view score_dtype,
                                          const std::string_view cls_name) {
  std::string key;
  const auto len = TotalStringLength<const std::string_view>(
                     {device, key_dtype, value_dtype, score_dtype, cls_name})
                   + 4;
  key.reserve(len);
  key.append(device);
  key.push_back('_');
  key.append(key_dtype);
  key.push_back('_');
  key.append(value_dtype);
  key.push_back('_');
  key.append(score_dtype);
  key.push_back('_');
  key.append(cls_name);

  std::transform(key.begin(), key.end(), key.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  return key;
}

const std::string CreateStorageFactoryKey(const DeviceType device_type,
                                          const DataType key_dtype,
                                          const DataType value_dtype,
                                          const DataType score_dtype,
                                          const std::string_view cls_name) {
  auto device_str = ::magic_enum::enum_name(device_type);
  auto ktype_str = ::magic_enum::enum_name(key_dtype);
  auto vtype_str = ::magic_enum::enum_name(value_dtype);
  auto stype_str = ::magic_enum::enum_name(score_dtype);
  std::string key;
  const auto len = TotalStringLength<std::string_view>(
                     {device_str, ktype_str, vtype_str, stype_str, cls_name})
                   + 4;
  key.reserve(len);
  key.append(device_str);
  key.push_back('_');
  key.append(ktype_str);
  key.push_back('_');
  key.append(vtype_str);
  key.push_back('_');
  key.append(stype_str);
  key.push_back('_');
  key.append(cls_name);
  std::transform(key.begin(), key.end(), key.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  return key;
}

const std::string CreateStorageFactoryKey(const std::string_view device,
                                          const std::string_view cls_name) {
  std::string key;
  const auto len = TotalStringLength<std::string_view>({device, cls_name}) + 1;
  key.reserve(len);
  key.append(device);
  key.push_back('_');
  key.append(cls_name);
  std::transform(key.begin(), key.end(), key.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  return key;
}

const std::string CreateStorageFactoryKey(const DeviceType device_type,
                                          const std::string_view cls_name) {
  auto device_str = magic_enum::enum_name(device_type);
  std::string key;
  const auto len =
    TotalStringLength<std::string_view>({device_str, cls_name}) + 1;
  key.reserve(len);
  key.append(device_str);
  key.push_back('_');
  key.append(cls_name);
  std::transform(key.begin(), key.end(), key.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  return key;
}

}  // namespace strings
}  // namespace meepo_embedding
