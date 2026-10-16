/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * ExecuTorch platform abstraction layer (PAL) for ESP-IDF.
 *
 * Derived from examples/espressif/executor_runner/esp_pal.cpp with two
 * changes for component use: log output goes through ESP_LOG under the
 * "executorch" tag (so IDF log filtering applies, on top of the
 * component's own CONFIG_ET_LOG_LEVEL_* compiled-in floor), and
 * et_pal_allocate/et_pal_free use the heap instead of returning
 * nullptr. Timing uses the CPU cycle counter, like the example.
 *
 * ET_ESP_LOG_LEVEL is an esp_log_level_t value set by the component
 * CMakeLists from Kconfig; messages above it are dropped here even if
 * the IDF runtime level would allow them. Setting
 * CONFIG_ET_ENABLE_LOGGING=n instead compiles ET_LOG out entirely in
 * the runtime (ET_LOG_ENABLED=0) and this file is never called.
 */

#include <executorch/runtime/platform/compiler.h>
#include <executorch/runtime/platform/platform.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>

#include <esp_clk_tree.h>
#include <esp_cpu.h>
#include <esp_log.h>
#include <esp_system.h>

#ifndef ET_ESP_LOG_LEVEL
#define ET_ESP_LOG_LEVEL ESP_LOG_INFO
#endif

namespace {
const char* kTag = "executorch";

esp_log_level_t to_esp_level(et_pal_log_level_t level) {
  switch (level) {
    case et_pal_log_level_t::kDebug:
      return ESP_LOG_DEBUG;
    case et_pal_log_level_t::kInfo:
      return ESP_LOG_INFO;
    case et_pal_log_level_t::kError:
      return ESP_LOG_ERROR;
    case et_pal_log_level_t::kFatal:
      return ESP_LOG_ERROR;
    default:
      return ESP_LOG_INFO;
  }
}
} // namespace

extern "C" {

void et_pal_init(void) {}

ET_NORETURN void et_pal_abort(void) {
  // Matches ESP-IDF's own panic policy (restart) and the example PAL;
  // abort() on IDF also ends in a panic, but a clean restart keeps the
  // coredump/backtrace behavior of the rest of the firmware.
  esp_restart();
}

et_timestamp_t et_pal_current_ticks(void) {
  return static_cast<et_timestamp_t>(esp_cpu_get_cycle_count());
}

et_tick_ratio_t et_pal_ticks_to_ns_multiplier(void) {
  uint32_t cpu_freq_hz = 0;
  if (esp_clk_tree_src_get_freq_hz(
          SOC_MOD_CLK_CPU,
          ESP_CLK_TREE_SRC_FREQ_PRECISION_CACHED,
          &cpu_freq_hz) == ESP_OK &&
      cpu_freq_hz != 0) {
    return {1000000000u, cpu_freq_hz};
  }
  // Fall back to the ESP32 family's common 240 MHz configuration.
  return {1000000000u, 240000000u};
}

void et_pal_emit_log_message(
    ET_UNUSED et_timestamp_t timestamp,
    et_pal_log_level_t level,
    const char* filename,
    const char* function,
    size_t line,
    const char* message,
    ET_UNUSED size_t length) {
  const esp_log_level_t esp_level = to_esp_level(level);
  if (esp_level > ET_ESP_LOG_LEVEL) {
    return;
  }
  // esp_log_write applies the runtime per-tag level as well. Keep the
  // runtime's file/function context: it is usually the only clue for
  // allocation and operator-mismatch failures.
  esp_log_write(
      esp_level, kTag, "%s:%zu %s: %s\n", filename, line, function, message);
}

void* et_pal_allocate(size_t size) {
  return malloc(size);
}

void et_pal_free(void* ptr) {
  free(ptr);
}

} // extern "C"
