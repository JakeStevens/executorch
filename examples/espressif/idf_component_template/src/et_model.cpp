/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Implementation of et_model.h over executorch::runtime::Program/Method.
//
// Program directly (rather than extension::Module) because placement
// must be controllable: Module allocates its memory-planned buffers
// with std::vector (global heap, no caps) and hard-references the
// mmap data loader, which ESP-IDF cannot provide. Module is still
// vendored in this component for advanced users; see README.md.

#include "et_model.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <utility>

#include <esp_heap_caps.h>
#include <esp_log.h>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/core/hierarchical_allocator.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/method_meta.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h>

namespace {

const char* kTag = "et_model";
constexpr size_t kDefaultArenaSize = 256 * 1024;
constexpr size_t kArenaAlignment = 16;
constexpr size_t kMaxInputs = 8;
constexpr size_t kMaxOutputs = 8;
constexpr size_t kMaxPlannedBuffers = 16;

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::extension::BufferDataLoader;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::TensorInfo;
using SpanU8 = executorch::runtime::Span<uint8_t>;

uint32_t caps_for(et_mem_location_t loc) {
  switch (loc) {
    case ET_MEM_INTERNAL:
      return MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;
    case ET_MEM_PSRAM:
      return MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT;
    case ET_MEM_DEFAULT:
    default:
      return MALLOC_CAP_DEFAULT;
  }
}

uint8_t* caps_alloc(size_t size, et_mem_location_t loc) {
  if (size == 0) {
    size =
        1; // heap_caps_aligned_alloc(0) is not useful; Method needs ptr+0 span
  }
  return static_cast<uint8_t*>(
      heap_caps_aligned_alloc(kArenaAlignment, size, caps_for(loc)));
}

} // namespace

struct et_model {
  explicit et_model(const et_model_config_t& c)
      : cfg(c), loader(c.pte, c.pte_size) {}

  bool fail(const char* what, Error err) {
    snprintf(
        error,
        sizeof(error),
        "%s failed (error 0x%x)",
        what,
        static_cast<unsigned>(err));
    ESP_LOGE(kTag, "%s", error);
    return false;
  }
  bool fail(const char* what) {
    snprintf(error, sizeof(error), "%s failed", what);
    ESP_LOGE(kTag, "%s", error);
    return false;
  }

  void free_method() {
    method.reset();
    manager.reset();
    planned.reset();
    for (size_t i = 0; i < num_planned; i++) {
      heap_caps_free(planned_ptrs[i]);
      planned_ptrs[i] = nullptr;
    }
    num_planned = 0;
    heap_caps_free(arena);
    arena = nullptr;
    allocator.reset();
    method_name = nullptr;
    for (size_t i = 0; i < kMaxInputs; i++) {
      impls[i].reset();
      tensors[i].reset();
    }
  }

  et_model_config_t cfg;
  BufferDataLoader loader;
  std::optional<Program> program;

  // Loaded-method state (all zero/null until et_model_load_method).
  const char* method_name = nullptr;
  uint8_t* arena = nullptr;
  std::optional<MemoryAllocator> allocator;
  uint8_t* planned_ptrs[kMaxPlannedBuffers] = {};
  SpanU8 planned_spans[kMaxPlannedBuffers];
  size_t num_planned = 0;
  std::optional<HierarchicalAllocator> planned;
  std::optional<MemoryManager> manager;
  std::optional<Method> method;
  std::optional<MethodMeta> meta;

  std::optional<TensorImpl> impls[kMaxInputs];
  std::optional<Tensor> tensors[kMaxInputs];
  int32_t sizes[kMaxInputs][ET_MODEL_MAX_DIMS] = {};
  uint8_t dim_order[kMaxInputs][ET_MODEL_MAX_DIMS] = {};
  EValue outputs[kMaxOutputs];

  char error[128] = "no error";
};

namespace {

bool fill_info(const TensorInfo& info, et_tensor_info_t* out) {
  if (out == nullptr) {
    return false;
  }
  memset(out, 0, sizeof(*out));
  out->scalar_type = static_cast<int32_t>(info.scalar_type());
  out->ndim = info.sizes().size();
  if (out->ndim > ET_MODEL_MAX_DIMS) {
    return false;
  }
  for (size_t i = 0; i < out->ndim; i++) {
    out->dims[i] = info.sizes()[i];
  }
  out->nbytes = info.nbytes();
  return true;
}

bool method_meta_for(
    et_model* model,
    const char* name,
    std::optional<MethodMeta>* out_meta,
    const char** out_name) {
  const char* resolved = name;
  if (resolved == nullptr) {
    if (model->program->num_methods() == 0) {
      return model->fail("program has no methods");
    }
    auto name_res = model->program->get_method_name(0);
    if (!name_res.ok()) {
      return model->fail("get_method_name", name_res.error());
    }
    resolved = *name_res;
  }
  auto meta_res = model->program->method_meta(resolved);
  if (!meta_res.ok()) {
    return model->fail("method_meta", meta_res.error());
  }
  out_meta->emplace(std::move(meta_res.get()));
  *out_name = resolved;
  return true;
}

} // namespace

extern "C" {

et_model_t* et_model_create(const et_model_config_t* cfg) {
  if (cfg == nullptr || cfg->pte == nullptr || cfg->pte_size == 0) {
    ESP_LOGE(kTag, "et_model_create: invalid config");
    return nullptr;
  }
  executorch::runtime::runtime_init();

  auto model = std::make_unique<et_model>(*cfg);
  auto program_res = Program::load(&model->loader);
  if (!program_res.ok()) {
    model->fail("Program::load", program_res.error());
    return nullptr;
  }
  model->program.emplace(std::move(program_res.get()));
  ESP_LOGI(
      kTag,
      "loaded program: %zu bytes, %zu method(s)",
      cfg->pte_size,
      static_cast<size_t>(model->program->num_methods()));
  return model.release();
}

void et_model_destroy(et_model_t* model) {
  if (model == nullptr) {
    return;
  }
  model->free_method();
  delete model;
}

size_t et_model_num_methods(et_model_t* model) {
  if (model == nullptr || !model->program) {
    return 0;
  }
  return static_cast<size_t>(model->program->num_methods());
}

const char* et_model_method_name(et_model_t* model, size_t index) {
  if (model == nullptr || !model->program) {
    return nullptr;
  }
  auto res = model->program->get_method_name(index);
  return res.ok() ? *res : nullptr;
}

bool et_model_load_method(et_model_t* model, const char* name) {
  if (model == nullptr || !model->program) {
    return false;
  }
  model->free_method();

  std::optional<MethodMeta> meta;
  const char* resolved = nullptr;
  if (!method_meta_for(model, name, &meta, &resolved)) {
    return false;
  }
  model->method_name = resolved;

  const size_t arena_size =
      model->cfg.arena_size ? model->cfg.arena_size : kDefaultArenaSize;
  model->arena = caps_alloc(arena_size, model->cfg.arena_location);
  if (model->arena == nullptr) {
    return model->fail("allocating method arena");
  }
  model->allocator.emplace(static_cast<uint32_t>(arena_size), model->arena);

  model->num_planned = meta->num_memory_planned_buffers();
  if (model->num_planned > kMaxPlannedBuffers) {
    model->free_method();
    return model->fail("too many planned buffers");
  }
  for (size_t i = 0; i < model->num_planned; i++) {
    auto size_res = meta->memory_planned_buffer_size(i);
    if (!size_res.ok()) {
      model->free_method();
      return model->fail("memory_planned_buffer_size", size_res.error());
    }
    const size_t bytes = static_cast<size_t>(size_res.get());
    uint8_t* buf = caps_alloc(bytes, model->cfg.planned_location);
    if (buf == nullptr && bytes != 0) {
      model->free_method();
      return model->fail("allocating planned buffer");
    }
    model->planned_ptrs[i] = buf;
    model->planned_spans[i] = SpanU8(buf, bytes);
  }
  model->planned.emplace(executorch::runtime::Span<SpanU8>(
      model->planned_spans, model->num_planned));
  model->manager.emplace(&*model->allocator, &*model->planned);

  auto method_res = model->program->load_method(resolved, &*model->manager);
  if (!method_res.ok()) {
    model->free_method();
    return model->fail("load_method", method_res.error());
  }
  model->method.emplace(std::move(method_res.get()));
  model->meta.emplace(std::move(*meta));
  ESP_LOGI(
      kTag,
      "loaded method '%s': arena %zu B, %zu planned buffer(s)",
      resolved,
      arena_size,
      model->num_planned);
  return true;
}

bool et_model_planned_info(
    et_model_t* model,
    const char* name,
    size_t* out_num_buffers,
    size_t* out_total_bytes) {
  if (model == nullptr || !model->program) {
    return false;
  }
  std::optional<MethodMeta> meta;
  const char* resolved = nullptr;
  const char* query = name != nullptr ? name : model->method_name;
  if (!method_meta_for(model, query, &meta, &resolved)) {
    return false;
  }
  size_t total = 0;
  for (size_t i = 0; i < meta->num_memory_planned_buffers(); i++) {
    auto size_res = meta->memory_planned_buffer_size(i);
    if (!size_res.ok()) {
      return model->fail("memory_planned_buffer_size", size_res.error());
    }
    total += static_cast<size_t>(size_res.get());
  }
  if (out_num_buffers != nullptr) {
    *out_num_buffers = meta->num_memory_planned_buffers();
  }
  if (out_total_bytes != nullptr) {
    *out_total_bytes = total;
  }
  return true;
}

size_t et_model_num_inputs(et_model_t* model) {
  return (model != nullptr && model->meta) ? model->meta->num_inputs() : 0;
}

size_t et_model_num_outputs(et_model_t* model) {
  return (model != nullptr && model->meta) ? model->meta->num_outputs() : 0;
}

bool et_model_input_info(
    et_model_t* model,
    size_t index,
    et_tensor_info_t* out) {
  if (model == nullptr || !model->meta) {
    return false;
  }
  auto info = model->meta->input_tensor_meta(index);
  return info.ok() && fill_info(info.get(), out);
}

bool et_model_output_info(
    et_model_t* model,
    size_t index,
    et_tensor_info_t* out) {
  if (model == nullptr || !model->meta) {
    return false;
  }
  auto info = model->meta->output_tensor_meta(index);
  return info.ok() && fill_info(info.get(), out);
}

bool et_model_set_input(
    et_model_t* model,
    size_t index,
    const void* data,
    size_t nbytes) {
  if (model == nullptr || !model->method || !model->meta) {
    return false;
  }
  if (index >= kMaxInputs || index >= model->meta->num_inputs()) {
    return model->fail("input index out of range");
  }
  auto info = model->meta->input_tensor_meta(index);
  if (!info.ok()) {
    return model->fail("input_tensor_meta", info.error());
  }
  if (info->nbytes() != nbytes) {
    return model->fail("input size does not match model");
  }
  if (info->sizes().size() > ET_MODEL_MAX_DIMS) {
    return model->fail("input has too many dims");
  }
  for (size_t d = 0; d < info->sizes().size(); d++) {
    model->sizes[index][d] = info->sizes()[d];
    model->dim_order[index][d] = info->dim_order()[d];
  }
  model->impls[index].emplace(
      info->scalar_type(),
      static_cast<ssize_t>(info->sizes().size()),
      model->sizes[index],
      const_cast<void*>(data),
      model->dim_order[index]);
  model->tensors[index].emplace(&*model->impls[index]);
  Error err = model->method->set_input(EValue(*model->tensors[index]), index);
  if (err != Error::Ok) {
    return model->fail("set_input", err);
  }
  return true;
}

bool et_model_execute(et_model_t* model) {
  if (model == nullptr || !model->method) {
    return false;
  }
  Error err = model->method->execute();
  if (err != Error::Ok) {
    return model->fail("execute", err);
  }
  return true;
}

const void*
et_model_output_data(et_model_t* model, size_t index, size_t* out_nbytes) {
  if (out_nbytes != nullptr) {
    *out_nbytes = 0;
  }
  if (model == nullptr || !model->method || !model->meta) {
    return nullptr;
  }
  if (index >= kMaxOutputs || index >= model->meta->num_outputs()) {
    model->fail("output index out of range");
    return nullptr;
  }
  Error err =
      model->method->get_outputs(model->outputs, model->meta->num_outputs());
  if (err != Error::Ok) {
    model->fail("get_outputs", err);
    return nullptr;
  }
  if (!model->outputs[index].isTensor()) {
    model->fail("output is not a tensor");
    return nullptr;
  }
  Tensor tensor = model->outputs[index].toTensor();
  if (out_nbytes != nullptr) {
    *out_nbytes = tensor.nbytes();
  }
  return tensor.const_data_ptr();
}

const char* et_model_last_error(et_model_t* model) {
  return model != nullptr ? model->error : "null model";
}

} // extern "C"
