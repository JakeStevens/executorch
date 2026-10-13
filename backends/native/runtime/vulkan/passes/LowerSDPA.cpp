// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/vulkan/passes/LowerSDPA.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <executorch/backends/native/runtime/graph/Argument.h>
#include <executorch/backends/native/runtime/graph/Node.h>
#include <executorch/backends/native/runtime/graph/ScalarType.h>
#include <executorch/backends/native/runtime/graph/Value.h>
#include <executorch/backends/native/runtime/graph/utils/GraphUtils.h>

namespace ptn::vulkan {
namespace {

constexpr std::string_view kAtenSDPA =
    "torch.ops.aten.scaled_dot_product_attention.default";
constexpr const char* kVulkanSDPA = "torch.ops.et_vk.sdpa.default";

const Argument* get_arg(const Node& node, std::string_view name, size_t pos) {
  const auto found = std::ranges::find_if(
      node.inputs,
      [name](const NamedArgument& input) { return input.name == name; });
  if (found != node.inputs.end()) {
    return &found->arg;
  }
  return pos < node.inputs.size() ? &node.inputs[pos].arg : nullptr;
}

std::optional<ValueId> tensor_id(const Argument* arg) {
  if (arg == nullptr || arg->kind() != ArgKind::Tensor ||
      !valid(arg->as_tensor().id)) {
    return std::nullopt;
  }
  return arg->as_tensor().id;
}

// Absent optional arguments take their schema default.
bool is_bool_or_absent(const Argument* arg, bool value) {
  return arg == nullptr ||
      (arg->kind() == ArgKind::Bool && !valid(arg->as_bool().id) &&
       arg->as_bool().value == value);
}

bool is_float_or_absent(const Argument* arg, double value) {
  return arg == nullptr ||
      (arg->kind() == ArgKind::Float && !valid(arg->as_float().id) &&
       arg->as_float().value == value);
}

bool is_float_dtype(ScalarType dtype) {
  return dtype == ScalarType::Float || dtype == ScalarType::Half;
}

ValueId add_tensor(Graph& graph, std::string name, TensorMeta meta) {
  meta.dim_order_hint.clear();
  meta.quant.reset();
  return graph.append_value(Value(std::move(name), std::move(meta)));
}

// ET-VK's fused SDPA adds attn_mask as a float bias, but Hugging Face emits
// bool keep-masks (True = attend). Convert with
// mul.Scalar(_to_copy(logical_not(mask), q_dtype), -C): kept positions get
// exactly 0 and masked positions -C, where C is the largest finite value of
// the query dtype, mirroring the masked_fill eager SDPA applies. The
// _to_copy node carries the full schema argument list the serializer emits:
// the kernel reads only args[0] (input) and the appended output, but the
// output must land at its position.
ValueId bool_mask_bias(
    Graph& graph,
    NodeId before,
    const std::string& name,
    ValueId mask_id,
    ScalarType dtype) {
  const TensorMeta bool_meta = graph.value(mask_id).tensor_meta();
  TensorMeta float_meta = bool_meta;
  float_meta.dtype = dtype;

  const ValueId inverted = add_tensor(graph, name + "_not", bool_meta);
  graph.insert_node_before(
      before,
      Node{
          .name = name + "_not",
          .target = "torch.ops.aten.logical_not.default",
          .inputs = {{.name = "self", .arg = TensorArg{mask_id}}},
          .outputs = {{.value_id = inverted}},
      });
  const ValueId as_float = add_tensor(graph, name + "_f", float_meta);
  graph.insert_node_before(
      before,
      Node{
          .name = name + "_f",
          .target = "torch.ops.aten._to_copy.default",
          .inputs =
              {{.name = "self", .arg = TensorArg{inverted}},
               {.name = "dtype", .arg = ScalarTypeArg{dtype}},
               {.name = "layout", .arg = NoneArg{}},
               {.name = "device", .arg = NoneArg{}},
               {.name = "pin_memory", .arg = NoneArg{}},
               {.name = "non_blocking", .arg = BoolArg{false}},
               {.name = "memory_format", .arg = NoneArg{}}},
          .outputs = {{.value_id = as_float}},
      });
  const double magnitude = dtype == ScalarType::Half
      ? 65504.0
      : static_cast<double>(std::numeric_limits<float>::max());
  const ValueId bias = add_tensor(graph, name + "_bias", float_meta);
  graph.insert_node_before(
      before,
      Node{
          .name = name + "_bias",
          .target = "torch.ops.aten.mul.Scalar",
          .inputs =
              {{.name = "self", .arg = TensorArg{as_float}},
               {.name = "other", .arg = FloatArg{-magnitude}}},
          .outputs = {{.value_id = bias}},
      });
  return bias;
}

} // namespace

size_t lower_sdpa(Graph& graph) {
  size_t count = 0;
  // Bool-mask conversion appends nodes and values, so iterate a copy of the
  // schedule and re-resolve the node after inserting.
  for (const NodeId node_id : std::vector<NodeId>(graph.schedule)) {
    const Node& node = graph.node(node_id);
    if (!node.is_call() || node.target != kAtenSDPA ||
        node.outputs.size() != 1 || !valid(node.outputs[0].value_id)) {
      continue;
    }

    const Argument* query = get_arg(node, "query", 0);
    const Argument* key = get_arg(node, "key", 1);
    const Argument* value = get_arg(node, "value", 2);
    const Argument* attn_mask = get_arg(node, "attn_mask", 3);
    const Argument* dropout_p = get_arg(node, "dropout_p", 4);
    const Argument* is_causal = get_arg(node, "is_causal", 5);
    const Argument* scale = get_arg(node, "scale", 6);
    const Argument* enable_gqa = get_arg(node, "enable_gqa", 7);

    const std::optional<ValueId> q_id = tensor_id(query);
    const std::optional<ValueId> k_id = tensor_id(key);
    const std::optional<ValueId> v_id = tensor_id(value);
    if (!q_id || !k_id || !v_id || !is_float_or_absent(dropout_p, 0.0) ||
        !is_bool_or_absent(is_causal, false) ||
        !is_bool_or_absent(enable_gqa, false)) {
      continue;
    }

    Argument scale_arg = NoneArg{};
    if (scale != nullptr && scale->kind() == ArgKind::Float &&
        !valid(scale->as_float().id)) {
      scale_arg = *scale;
    } else if (scale != nullptr && scale->kind() != ArgKind::None) {
      continue;
    }

    const Value& q_value = graph.value(*q_id);
    const Value& k_value = graph.value(*k_id);
    const Value& v_value = graph.value(*v_id);
    const Value& out_value = graph.value(node.outputs[0].value_id);
    if (!q_value.is_tensor() || !k_value.is_tensor() || !v_value.is_tensor() ||
        !out_value.is_tensor()) {
      continue;
    }
    // Copies: inserting the mask conversion reallocates the graph's lists.
    const TensorMeta q_meta = q_value.tensor_meta();
    const TensorMeta k_meta = k_value.tensor_meta();
    const TensorMeta v_meta = v_value.tensor_meta();
    const TensorMeta out_meta = out_value.tensor_meta();
    // [B, H, S, D] x [B, H, L, D] x [B, H, L, Dv] -> [B, H, S, Dv]. The fused
    // kernel has no GQA head broadcast, so head counts must match.
    if (q_meta.sizes.size() != 4 || k_meta.sizes.size() != 4 ||
        v_meta.sizes.size() != 4 || out_meta.sizes.size() != 4 ||
        !is_float_dtype(q_meta.dtype) || k_meta.dtype != q_meta.dtype ||
        v_meta.dtype != q_meta.dtype || out_meta.dtype != q_meta.dtype ||
        k_meta.sizes[0] != q_meta.sizes[0] ||
        v_meta.sizes[0] != q_meta.sizes[0] ||
        k_meta.sizes[1] != q_meta.sizes[1] ||
        v_meta.sizes[1] != q_meta.sizes[1] ||
        k_meta.sizes[3] != q_meta.sizes[3] ||
        v_meta.sizes[2] != k_meta.sizes[2] ||
        out_meta.sizes !=
            std::vector<int64_t>{
                q_meta.sizes[0],
                q_meta.sizes[1],
                q_meta.sizes[2],
                v_meta.sizes[3]}) {
      continue;
    }

    // The fused kernel reads the mask as an additive float bias. A float
    // mask passes through; a bool keep-mask is converted; anything else
    // would change semantics, so leave it to fail dispatch by name.
    Argument mask_arg = NoneArg{};
    if (attn_mask != nullptr && attn_mask->kind() == ArgKind::Tensor) {
      const std::optional<ValueId> mask_id = tensor_id(attn_mask);
      if (!mask_id || !graph.value(*mask_id).is_tensor()) {
        continue;
      }
      const ScalarType mask_dtype = graph.value(*mask_id).tensor_meta().dtype;
      if (is_float_dtype(mask_dtype)) {
        mask_arg = *attn_mask;
      } else if (mask_dtype == ScalarType::Bool) {
        mask_arg = TensorArg{bool_mask_bias(
            graph, node_id, node.name + "_mask", *mask_id, q_meta.dtype)};
      } else {
        continue;
      }
    } else if (attn_mask != nullptr && attn_mask->kind() != ArgKind::None) {
      continue;
    }

    Node& lowered = graph.node(node_id);
    const Argument query_arg = *get_arg(lowered, "query", 0);
    const Argument key_arg = *get_arg(lowered, "key", 1);
    const Argument value_arg = *get_arg(lowered, "value", 2);
    lowered.target = kVulkanSDPA;
    lowered.inputs = {
        {.name = "query", .arg = query_arg},
        {.name = "key", .arg = key_arg},
        {.name = "value", .arg = value_arg},
        {.name = "attn_mask", .arg = mask_arg},
        {.name = "scale", .arg = scale_arg},
    };
    ++count;
  }
  if (count > 0) {
    graph.rebuild_def_use();
    validate_graph(graph);
  }
  return count;
}

} // namespace ptn::vulkan
