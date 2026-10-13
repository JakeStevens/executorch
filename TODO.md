# TODO: test Whisper-tiny on Native Vulkan (device validation)

Status: export + Vulkan load-time lowerings are implemented and tested
*without* a Vulkan device and *without* BUCK (standalone checkout; the
buck prelude/third-party submodules were not initialized on the dev
machine). This file is the handoff for an agent on a machine **with a
Vulkan device but still no BUCK**: build the runner manually, run the
Whisper package end to end, and fix what the device exposes.

## What is in this commit

- `backends/native/examples/hf/export_hf_whisper.py` — direct
  Transformers export (no optimum) of Whisper to one `.ptn` with three
  methods: `encoder` (mel → hidden), `cross_kv` (hidden → per-layer
  cross K/V, keys then values), `decoder` (`(ids[1,T], cache_position[T],
  *cross_keys, *cross_values)` → logits, static self-attention cache).
  `examples/hf_llm` moved to `examples/hf` with shared `hf_utils.py`.
- `backends/native/runtime/vulkan/passes/LowerSDPA.{h,cpp}` — NEW:
  cache-free `aten.scaled_dot_product_attention` → `et_vk.sdpa`,
  including bool keep-mask → additive float bias conversion.
- `passes/LowerHFAttention.{h,cpp}` — generalized: optional RoPE
  (Whisper learned positions), `.contiguous()` clone transparency,
  Whisper's broadcast causal-mask form, non-default SDPA scale folded
  into q (`mul.Scalar`), since `llama.custom_sdpa` requires scale=None.
- `VulkanEngineExecutable.cpp` — `et_vk.sdpa` width-packed; `gelu_` /
  `sigmoid_` dispatch to functional kernels like `relu_`.
- `backends/native/examples/audio/run_whisper.cpp` — runner: encoder →
  cross_kv once → prefill prompt → greedy decode; `--expected` does an
  exact id-sequence comparison (exit 3 on mismatch).
- `exir/native/native.py` + `backends/native/serialization/graph_serialize.py`
  — `to_native` namespaces colliding `_lifted_tensor_constant{i}` data
  keys per method (encoder/decoder scalar collisions); real fqn
  collisions still raise.
- Tests: `backends/native/test/test_hf_whisper.py`,
  `backends/native/test/runtime/vulkan/test_lower_sdpa.cpp`, new cases in
  `test_lower_hf_attention.cpp` and `exir/native/test/test_native.py`.

## Environment notes (learned the hard way)

- The checkout root *is* the `executorch` Python package: set
  `PYTHONPATH=<parent-of-checkout>`, not the checkout itself, or imports
  resolve to an installed namespace package.
- You need a venv with `torch` + `transformers` (+ `safetensors`). The
  dev machine used torch 2.12 / transformers 5.0 nightlies; pin problems
  there, not in the exporter, before suspecting the code.
- `git submodule update --init` at least: `third-party/flatbuffers`,
  `third-party/googletest`, `third-party/gflags` (they were empty here).
- Real `openai/whisper-tiny` from the hub is fine if you have network.
  Offline, fabricate a config-only dir and use `--random_weights`:
  save a `WhisperConfig` (d_model 384, 4+4 layers, 6 heads, mel 80,
  vocab 51865, decoder_start 50258) plus a `GenerationConfig` with
  `forced_decoder_ids=[[1,50259],[2,50359],[3,50363]]`, `eos 50257`.

## Step 1 — Python export + tests (no device needed; do first)

```sh
export PYTHONPATH=<parent-of-checkout>
python -m unittest executorch.backends.native.test.test_hf_whisper \
  executorch.exir.native.test.test_native.LiftedConstantCollisionTest \
  executorch.backends.native.test.test_serialize
python -m executorch.backends.native.examples.hf.export_hf_whisper \
  --model openai/whisper-tiny --output /tmp/whisper_tiny.ptn \
  --dump_dir /tmp/whisper_vec
```

Expect: each method reports `1 native delegate(s)` and residual
`(none)`. Known pre-existing failure in this environment class:
`ToNativeEquivalenceTest` errors with missing
`exir/_serialize/program.fbs` (resource packaging artifact; reproduces
on unmodified HEAD — ignore unless it changes shape).

## Step 2 — C++ pass tests without BUCK

The pass tests are plain gtest over the graph IR; compile them directly
(any gtest: system package or the `third-party/googletest` submodule
built once). Sources that worked here:

```sh
R=executorch; V=$R/backends/native/runtime/vulkan/passes
T=$R/backends/native/test/runtime/vulkan
clang++ -std=c++20 -I. -I<googletest>/googletest/include \
  $T/test_lower_hf_attention.cpp $T/test_lower_sdpa.cpp \
  $T/test_lower_rms_norm.cpp \
  $V/LowerHFAttention.cpp $V/LowerSDPA.cpp $V/LowerRMSNorm.cpp \
  $R/backends/native/runtime/graph/{Graph,Node,Value,Argument,Scalar,TensorMeta,MemoryPlanning}.cpp \
  $R/backends/native/runtime/graph/utils/{GraphUtils,Print}.cpp \
  -lgtest -lgtest_main -o /tmp/run_pass_tests && /tmp/run_pass_tests
```

All 21 cases passed on the dev machine. Keep them green.

## Step 3 — build the runner without BUCK

This is the real work. Two parts:

1. **ET-VK libraries + shaders via CMake** (do not hand-roll shader
   generation): configure the OSS build with Vulkan on and build the
   vulkan backend targets, which runs `gen_vulkan_spv.py`/glslc for you:

   ```sh
   cmake -B build-vk -DEXECUTORCH_BUILD_VULKAN=ON \
     -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON .
   cmake --build build-vk -j --target vulkan_backend  # adjust target name
   ```

   See `backends/vulkan/README.md` and `backends/vulkan/CMakeLists.txt`
   for exact target names/options on this revision; you need the
   compute API + graph runtime (`vulkan_compute_api`,
   `vulkan_graph_runtime` in BUCK terms) and VMA
   (`include/vk_mem_alloc.h` comes from the Vulkan SDK / third-party).

2. **Native runtime + runner** with one clang++ (or a small local
   CMakeLists you write under /tmp — do not commit build glue without
   discussion). Sources:

   - `backends/native/runtime/{Deserialize,Program,MethodMeta,TensorInfo,Validation}.cpp`
   - `backends/native/runtime/graph/*.cpp`, `graph/utils/{GraphUtils,Print,ToDot}.cpp` (ToDot only if referenced)
   - `backends/native/runtime/deserialize/*.cpp`
   - `backends/native/runtime/engine/Engine.cpp`
   - `backends/native/runtime/vulkan/{VulkanEngine,VulkanEngineExecutable,VulkanConstantMaterializationTracker}.cpp`
   - `backends/native/runtime/vulkan/passes/*.cpp`
   - `backends/native/examples/runner_utils/runner_utils.cpp`
   - `backends/native/examples/audio/run_whisper.cpp`

   Generated header: `flatc --cpp --cpp-std c++11 --gen-mutable
   --scoped-enums -o <gen-dir> backends/native/serialization/native_graph.fbs`
   then include it as `executorch/backends/native/runtime/native_graph_generated.h`
   (mirror the path under <gen-dir>/executorch/backends/native/runtime/).
   Include roots: checkout parent (so `executorch/...` resolves),
   <gen-dir>, flatbuffers include, Vulkan headers, gflags.
   Link: ET-VK libs from step 1, gflags, flatbuffers (header-only ok),
   Vulkan loader. Expect a few missing-symbol iterations; resolve from
   the BUCK `targets.bzl` deps (they are the source of truth).

If BUCK turns out to be available on the test machine after all, skip
all of this: `buck2 build //executorch/backends/native/examples/audio:run_whisper`
and `buck2 test //executorch/backends/native/test/runtime/vulkan:vulkan_passes_test
//executorch/backends/native/test:test_hf_whisper //executorch/exir/native/test:test_native`.

## Step 4 — run on device

```sh
./run_whisper --package /tmp/whisper_tiny.ptn \
  --input /tmp/whisper_vec/input_features.bin \
  --prompt /tmp/whisper_vec/prompt_ids.txt \
  --expected /tmp/whisper_vec/expected_ids.txt --eos_ids 50257
```

Acceptance, in order:

1. Runner prints `result: PASS` (exact greedy ids vs eager) for the
   random-weights/config-only export AND for real `openai/whisper-tiny`
   weights with `--dump_dir` vectors regenerated from the real model.
2. Real-audio sanity: dump mel for a known clip with
   `AutoFeatureExtractor` (30 s window), run, decode ids with the
   Whisper tokenizer in Python, compare against
   `model.generate(...)` greedy output for the same clip.
3. Re-run the vision runner (`examples/vision/run_classification`) and
   an `export_hf_llm` smoke package if handy — `LowerHFAttention` and
   dispatch changed for everyone.

## Watch list (ranked by suspicion)

1. **Cross-input staging persistence.** The runner sets cross K/V inputs
   once and relies on `set_input` staging surviving later `execute()`
   calls (verified by code reading only). If transcripts degrade after
   step 1 but prefill looks right, suspect this; workaround is re-setting
   cross inputs every step (correct, slower) — then file the engine
   behavior as a bug.
2. **Mutable-state claims across three executables.** Encoder/cross_kv
   have no mutable buffers; decoder owns `key_cache_*`/`value_cache_*`.
   Any "mutable state already held" throw means a binding key collision
   crept in (check serialized `mutable_buffers` per method).
3. **Zero-cache check.** The engine requires cache initial bytes to be
   zero (`kWrittenBeforeReadAttr` path). The exported decoder must not
   have run eagerly; the exporter already uses a fresh instance for
   `--dump_dir`. Failures here look like a compile-time throw naming the
   constant.
4. **Encoder SDPA weights texture limits.** Fused SDPA materializes
   `[1, 6, 1500, 1500]` fp32 attention weights following output storage;
   on small `max_texture3d_dim` devices this may need buffer storage
   (see `backends/native/runtime/vulkan/TODO.md`).
5. **`index.Tensor` positional embedding + `repeat`** in the decoder
   (buffer-storage forcing exists for index inputs; verify no layout
   throws), and `aten.convolution` 1D path in the encoder (ET-VK has a
   channels-packed conv1d fallback; check numerics via encoder-only
   comparison if transcripts are garbage: compare encoder output
   against eager hidden states first, then cross_kv outputs, then one
   decoder step — bisect in that order).
6. **Bool-mask conversion** (`LowerSDPA`): encoder/cross masks are
   None or bool; if attention outputs look uniform/garbage while shapes
   run, dump one SDPA output vs eager.

Useful knobs while debugging: `VK_LOADER_DEBUG=all` for loader issues;
ET-VK logging via `ETVK_LOG_LEVEL` if supported on this revision; run
under SwiftShader (`VK_ICD_FILENAMES=<swiftshader icd json>`) to
separate driver issues from graph issues.

## Out of scope here (already tracked in runtime/vulkan/TODO.md)

Zero-copy tensor sharing between executables (EngineContext mutable
storage contract), plan-conversion temporaries in the memory plan, and
splitting this squashed commit into reviewable diffs (history rewrite
happens later, on purpose).
