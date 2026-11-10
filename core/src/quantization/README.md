# SOFIE quantization

The quantization library (`SOFIE_quantization`) turns quantized ONNX graphs (ONNX
`QuantizeLinear`/`DequantizeLinear`, QONNX `Quant`, and native `FLOAT8E4M3FN` tensors) into
code that runs on integer and float8 codes where it can. It is one workflow: frontends bring
every format to one canonical graph, per-operator handlers decide and lower, and a small pass
manager runs them in order. Every BLAS call in the generated code goes through sofieBLAS.

The library is optional at build time (`-DSOFIE_WITH_QUANTIZATION=OFF` leaves it, the Q/DQ and
QONNX parsers and the quantized tests out); `SOFIE_core` never depends on it.

`SOFIE_QUANT_TRACE=1` prints every workflow event (including why a handler declined) to stderr.

## Layout

Headers are under `core/inc/SOFIE/quantization/<module>/`, sources under
`core/src/quantization/<module>/`.

| Module | Holds | Key types |
|---|---|---|
| `spec/` | The one vocabulary: the grid a tensor lives on and the form it takes; the portable codec | `QuantSpec`, `EQuantForm`, `TensorQuantState`, `QuantCodec::*` |
| `frontend/` | Format adapters and the canonical boundary operators they emit | `IQuantFrontend`, `QdqFrontend`, `QonnxFrontend`, `NativeLowPrecisionFrontend`, `ROperator_QuantEncode`, `ROperator_QuantDecode` |
| `graph/` | The only access to `RModel` internals; read-only views; transactional edits | `QuantModelAccess`, `QuantGraphView`, `QuantOpView`, `QuantRewriter` |
| `handlers/` | One handler per operator type, the registry mapping types to handlers | `IQuantOpHandler`, `QuantOpHandlerRegistry`, the family handlers |
| `lowered/` | The operators handlers build: generated Alpaka kernels plus sofieBLAS calls | `ROperator_QuantizedDenseLinear`, `ROperator_QuantizedConv`, `ROperator_QuantizedElementwise`, `ROperator_QuantizedGather` |
| `passes/` | The workflow state and the passes | `QuantPlan`, `IQuantPass`, `QuantPassManager` |
| `diagnostics/` | Environment options and the report written into generated code | `QuantizationOptions`, `QuantWorkflowReport` |
| `integration/` | The `RModelCodegenPass` that code generation calls; the installer | `QuantizationWorkflowPass`, `InstallQuantizationCodegenPass` |

Dependencies point one way: `spec` ← `frontend`, `graph` ← `handlers`, `lowered` ←
`passes` ← `integration`. `SOFIE_core` does not depend on any of it; the parse-only holders
`ROperator_ONNXQuantizeLinear`, `ROperator_ONNXDequantizeLinear` and `ROperator_QONNXQuant`
live in core so the parser can build them, and throw if they ever reach code generation.

## Workflow

```
RModelParser_ONNX::Parse
  builds parse-only Q/DQ / QONNX holders, installs QuantizationWorkflowPass

RModel::Initialize
  Normalize ── frontends ────────────── Q/DQ, QONNX, native FP8  →  QuantEncode / QuantDecode
  (operators initialize: specs resolve, constant boundaries fold to codes / floats)
  Analyze ──── Annotate ─────────────── every tensor gets a TensorQuantState

Generate / GenerateGPU_ALPAKA
  BuildLoweredView
     Canonicalize ── duplicate decodes and same-grid decode→encode become views (Alpaka)
     Decide ──────── absorption, then form negotiation between producers and readers
     Lower ───────── handlers build or rewire operators; decliners are pinned to float
     Finalize ────── dead code, unused folded constants, summary line
  DiagnosticComments ── the report, written at the top of the generated header
```

### Canonical form

After `Normalize` the graph holds only float operators plus `QuantEncode` (float → codes)
and `QuantDecode` (codes → float):

- ONNX Q/DQ map one to one.
- A QONNX `Quant` becomes `Decode(Encode(x))` through an explicit `<output>_codes` tensor.
- A `Cast` to `FLOAT8E4M3FN` becomes an encode and a decode on the E4M3 grid with scale 1, and
  an FP8 initializer read by an ordinary operator moves behind a decode (`<name>_codes`). FP8
  initializers read by a Q/DQ or QONNX boundary are left alone: they are that boundary's codes.

Both canonical operators generate Alpaka code, so any boundary no handler absorbs still runs
on every Alpaka backend. Graph inputs and outputs keep the type the ONNX graph
declares (float for QONNX models, int8 for Q/DQ graphs that start or end on codes).

### Grid vs. form

A tensor's grid (`QuantSpec`: encoding, bit width, signedness, scales, zero points, axis, code
range, rounding) is fixed by the model. Its form (`EQuantForm`: `Float`, `Snapped`, `Codes`) is
what the workflow chooses. Every optimization is a choice of form:

| Form decision | Effect |
|---|---|
| a decode's readers all take `Codes` | the decode is not emitted; readers read the codes |
| an encode's producer offers `Codes` on its grid | the producer writes the codes; the encode is not emitted |
| a movement op reads and writes `Codes` | Reshape/Transpose run on int8/fp8 bytes |
| a producer offers `Snapped` | the fake-quant pair collapses into the producer's store |

### Handlers

`handlers/RQuantization_Handler.hxx` defines one method per question a pass asks:

| Pass | Method | Question |
|---|---|---|
| Annotate | `InferOutputStates` | which grid/form does each output carry? |
| Decide | `AbsorbConsumers` | which following operators fold into me? |
| Decide | `AcceptedInputForms` | in which forms can I read input *i*? |
| Decide | `OfferedOutputForms` | in which forms can I write output *j* on this grid? |
| Lower | `Lower` | build/rewire the operator for the decided forms |

A handler receives a `QuantContext`: the operator in focus (`QuantOpView`), the whole graph
(`QuantGraphView`, including every tensor's state), a negotiation
oracle (`IQuantNegotiation`: "do this tensor's readers accept codes?", "how is this tensor
carried?") and the report. Handlers never edit the model; `Lower` proposes edits through a
`QuantRewriter`, and the driver commits all of them or none. A form offered in Decide is a
promise: if `Lower` returns false (use `Decline(ctx, reason)` so the reason is traced) the
operator is pinned to float and Decide runs again, so a declining handler never leaves the
graph inconsistent.

Registered handlers (`handlers/RQuantization_HandlerRegistry.cxx`):

| Handler | Operators | Lowers to |
|---|---|---|
| `FloatOnlyHandler` | everything unregistered | float code |
| `ValuePreservingHandler` (codes) | Reshape/Flatten/Squeeze/Unsqueeze, Transpose | the same operator on codes |
| `ValuePreservingHandler` | Identity | grid annotation only |
| `GridPreservingHandler` | Relu, Clip | float; absorbable into an arithmetic epilogue |
| `FusibleFloatHandler` | Softmax, LayerNormalization | the float operator decoding its input / encoding its output in place |
| `DenseLinearHandler` | Gemm (and MatMul, parsed as Gemm) | `ROperator_QuantizedDenseLinear` (sofieBLAS int8 / fp8 matmul) |
| `ConvHandler` | Conv (1D/2D, NCHW, groups) | `ROperator_QuantizedConv` (im2col + sofieBLAS, or a direct kernel) |
| `ElementwiseHandler` | Add, Mul | `ROperator_QuantizedElementwise` |
| `GatherTableHandler` | Gather on a constant table | `ROperator_QuantizedGather` |

What each arithmetic family lowers:

- **Dense linear**: signed int8 per-tensor input (any zero point; a non-zero one needs a
  constant weight for the column sums), symmetric int8 weight per tensor or per output
  channel; or E4M3 input and weight. Any K and N lower: the vendor kernels need K and N
  aligned to 4 (K to 16 for float8), and sofieBLAS runs other shapes on its native kernels,
  which are far slower (the SOFIE trace notes such a layer).
  Batched MatMul with a runtime right operand needs it transposed to [N, K] first (the
  canonicalisation in `RModel` does that). Plain symmetric int8 outputs are written by one
  fused sofieBLAS int8-output call (per-channel weights use `int8MatmulVectorAlpha`);
  everything else runs the int32 / float accumulator through a SOFIE epilogue kernel.
- **Conv**: signed int8 or E4M3 input and weight. im2col into code columns plus a
  strided-batched sofieBLAS matmul when the weight is symmetric, the conv is not depthwise and
  the column buffer stays under 256 MiB; a direct SOFIE kernel otherwise. The choice is made at
  code generation. The window is padded to the vendor alignment so the fast kernels stay usable.
- **Elementwise / Gather**: generated kernels decoding and encoding per element.

Every arithmetic handler absorbs a following `[Encode → Decode on one grid]`, then `Relu`
and/or `Clip` with constant bounds, into its store (`AbsorbedEpilogue`, `QuantStorePlan`).

A boundary the workflow keeps as a kernel still joins SOFIE's elementwise fusion when it is
per-tensor: a single-reader chain such as `Clip → Encode → Decode` becomes one kernel instead of
three. The fused kernel then reads and writes the chain's own input and output types and rounds
each step to the type of the tensor it replaces, so the results are bit-identical to the separate
kernels. Per-channel boundaries keep their own kernel (the fused kernel has no channel index).

## Extending

**A new operator.** Write a handler (or reuse a generic one) and register it in
`RQuantization_HandlerRegistry.cxx`. If it lowers to a new operator, put that under
`lowered/` deriving from `ROperator_QuantLowered`, generate Alpaka code only, and call BLAS
through `blas.` (sofieBLAS). sofieBLAS has no "is this supported?" query: every quantized
call works for any shape and layout, falling back from the vendor kernel to a vendor multiply
plus native epilogue to fully native kernels. Give the operator `OperatorKind::QUANTIZED_GEMM` /
`QUANTIZED_CONV` if it calls BLAS: the session then declares `blas` with
`BlasOptions{.quantization = true}`, without which the quantized calls throw. A float8 BLAS call
should emit `RequireOcpFloat8()` first (a compile-time refusal on FNUZ-only AMD targets).

**A new grid** (another bit width or float format). Extend `QuantSpec`/`EQuantEncoding` and
`QuantCodec` (keep it `constexpr` and `__host__ __device__`), then teach `IsKernelGrid` and the
family planners which grids they accept. Frontends need no change unless the format has a new
ONNX spelling.

**A new format.** Add an `IQuantFrontend` that rewrites its nodes into `QuantEncode` /
`QuantDecode` with `QuantBoundaryParameters` and register it in `QuantFrontends()`.

## sofieBLAS

The generated code needs a sofieBLAS with the quantized entry points (int8, fp8, fp16), the
`BlasOptions{.quantization = true}` opt-in and the native fallback backend. On the GPU each
quantized call runs the vendor kernel, a vendor multiply plus a native epilogue, or fully native
kernels, chosen when the backend is constructed (`blas.capabilities()` reports the choice). What
sofieBLAS still lacks, and what SOFIE does meanwhile (this table is the record):

| Missing in sofieBLAS | SOFIE meanwhile |
|---|---|
| int8 matmul with leading dimensions (padded / strided views) | dense operands only |
| int8 output with a zero point, uint8, narrow range or clip | int32 accumulator + SOFIE epilogue kernel |
| int8 matmul with rescaled float output | int32 accumulator + SOFIE epilogue kernel |
| column sums for zero-point correction | folded at code generation (constant weights only) |
| fp8 matmul with a float bias (it takes BF16) | SOFIE epilogue kernel adds the float bias |
| int8 / fp8 convolution (implicit GEMM) | SOFIE im2col + strided-batched matmul, or a direct kernel |
| per-call workspace query | the fixed 32 MiB sofieBLAS workspace is not accounted for |

The vendor kernels need aligned shapes and (for float8) the TN layout; sofieBLAS now moves
anything else to its native path itself, so SOFIE no longer keeps misaligned layers in float.
That path is one to two orders of magnitude slower than the vendor kernels, so a layer that
should be fast and is not is worth checking with `SOFIE_QUANT_TRACE=1`. Float8 on an AMD target
that only has the FNUZ formats is refused at compile time, because SOFIE's codec writes OCP
E4M3FN.

## Known limitations

- A model can be generated once per parse: lowering mutates parsed operators (views, movement
  rewires) and registers aliases, so a second `Generate` call sees a rewritten graph.
- Per-channel axis default: an absent ONNX `axis` is treated as "last dimension", not the ONNX
  default of 1.
- Only FLOAT8E4M3FN has a codec; FLOAT8E5M2 boundaries are rejected. FP8 graph inputs are
  rejected (feed float and `Cast`).
- Per-channel boundaries left as kernels support up to 256 channels (kernel parameter size).
- Quantization is supported by the Alpaka generator only (`RModel::GenerateGPU_ALPAKA`). The plain
  C++ generator (`RModel::Generate`) refuses a quantized model with an error instead of emitting
  float code that ignores the quantization. Float models are unaffected.
