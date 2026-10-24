# Fast paths: the accuracy ledger

**An accuracy note is logged, never fatal.** The work keeps running; only what Metal itself refuses
ends it.

Each row of the first table is one place where the default trades exactness against Metal for
speed: pipeline-creation ("making") time or GPU time, mostly off Apple's GPUs. The fast form is the
default, the accurate form stays behind its switch, and both are kept. This file is a ledger, not a
journal: rows are edited in place, with no history. A change that adds, removes or changes a fast
default, its bound or its switch updates its row in the same commit. The test `tests::fast_paths`
fails when a `DevicePolicy` accuracy choice, one of its values, or a precision note a fast default
records is missing here, or when this file names a policy value that no longer exists.

Every switch is a field of `DevicePolicy` (the runtime's `DeviceOptions::policy`). On MoltenVK on
an Apple GPU, whose sampler and `float` functions are Metal's own, most rows do not apply. Each
difference is recorded in the pipeline's artifact as a `PrecisionNote` and announced as
`metal2vulkan-degrade-<code>` when the runtime makes the pipeline state; each note's rustdoc and
docs/API.md section 9 state its bound.

## Fast defaults

Costs are measured on the RTX 5080 Laptop GPU (NVIDIA 615.71.09) with `scripts/making-bench`:
cold, the command queue made first, the driver's disk cache off; the making of the state's
pipeline, then GPU time per draw, the default against the accurate form alone. The workloads are
three consumer pipelines: A2X (`a2x-s4`, `a2x-s96`), Xgc14 (`xgc14-bgra8`) and Xgc15
(`xgc15-rgba8`). A cell that names another source took its numbers there.

| What it is | Default (fast) | Accurate switch | Recorded bound (note, code) | Cost of the accurate form (5080) |
|---|---|---|---|---|
| Samples, gathers and depth comparisons go to the device's own sampler: its filter, footprint, level, anisotropy, sRGB decode and comparison. An 8-bit sRGB format the device cannot sample (`R8G8_SRGB` on NVIDIA) is held as its UNORM twin: a sample the raw codes do not serve is the device's filter of the codes, decoded through Metal's table. | `sampling = SamplingPolicy::NativeDevice` | `sampling = SamplingPolicy::Exact`: Metal's sampler emulated by fetches, a pipeline per bound sampler state | Linear filter within one code (1/255) on the measured devices, sRGB within 1.83; levels, anisotropy, cubes, 3D, gathers and comparisons unmeasured. A twin's filtered sample: exact for one texel and for a blend of two adjacent codes, up to 0.286 off for a blend of codes 0 and 255 (`SrgbTexelDecode`). `DeviceSampling` (`device-sampling`) | A2X under the four exact switches of the consumer's exact boot (sampling, texel conversion, half blend, NaN and min/max) on the RTX 5080 Laptop GPU (615.71.09): its function 572,061 words, made in 11.5-12.8 s; its anisotropic sampler's variant 659,590 words, 20.9-26.1 s on a loaded host where the function took 16.7-22.3 s; at the defaults 260,153 words, 1.1 s. On ANV both pass its compiler word bound, so it drops exact sampling for them (the row of that bound). At the switch (maths and texel conversions still exact): Xgc15 making 24.2-24.8 s exact against 12.5-13.9 s native |
| A two-plane YCbCr texture's linear sample (`YCBCR8_420_2P` and its kin, over a surface's planes) is the device's filter of its texels, which hold Metal's reads exactly, as any other texture's is under native sampling: a draw binding one takes the pipeline a draw binding any other texture takes. Metal sums the texels' steps on the format's grid (`1/2048`, `1/8192`), each term truncated. Under native sampling, MoltenVK's included (Metal's own sampler filters the texels as an `RGBA16Float` or `RGBA32Float` texture's); with `sampling = SamplingPolicy::Exact` every module answers Metal's filter, MoltenVK's through the slot's variant. | `ycbcr_filter = YCbCrFilterPolicy::Device` | `ycbcr_filter = YCbCrFilterPolicy::Exact`: Metal's filter emulated in the slot's YCbCr variant, a pipeline the first draw binding such a texture makes | Linear filter within 1.183 codes (1/255) of Metal's filter of the format on NVIDIA (RTX 3090), 0.877 on RADV, 1.058 on ANV, 0.872 on lavapipe; within 0.498 at coordinates on the 1/256 grid. `DeviceSampling` (`device-sampling`). On MoltenVK within one step of the grid (0.125 codes on `1/2048`, 0.016 on `1/8192`), a `half` result within one `half` ulp; a `half` sample of one texel on the grid `8192` through a runtime sampler rounds to nearest even where Metal rounds ties away. `ycbcr-filter-not-exact`, `HalfTexelConversion` (`half-texel-conversion`), announced where a draw binds such a texture | One pipeline more, made at the first draw binding such a texture. A2X (`a2x-s96` with a `YCBCR8_420_2P` texture at fragment slot 3, on ANV, Iris Xe): 2 pipelines, 13.9 s at the state's creation and 37.8 s at the draw (52.3 s wall), against the creation's one drawn as made (14.3 s) |
| A texel's `float` value is the device's conversion of its codes; a `half` result is the device's `float` texel rounded to nearest even. | `texel_conversion = TexelConversionPolicy::Device` | `texel_conversion = TexelConversionPolicy::Exact`: Metal's conversion by the bound format's class of every read; a sample's or gather's also with `sampling = SamplingPolicy::Exact` (under native sampling the device's sampler decodes it) | One `f32` ulp for 8- and 16-bit codes, 0.18 of a code for 5- and 6-bit packed channels; `half` within one ulp; NaN bits the device's. `DeviceTexelConversion` (`device-texel-conversion`), `HalfTexelConversion` (`half-texel-conversion`, also recorded for heap textures and unmeasured formats) | Exact reads (the same reads): Xgc14 0.88-0.89 → 0.90-0.93 s, GPU 0.039 → 0.046 ms; Xgc15 0.65-0.66 → 0.68 s, 0.090 → 0.094 ms; A2X unchanged (no affected read). Samples: with the sampling row |
| The maths library (`sin`, `exp`, `log`, `pow`, `atan2`...), divisions and Metal's fast reciprocal, and a fused multiply-add without `OpFmaKHR` are the device's instructions, not routines of Metal's answers. Off MoltenVK a `float` multiply, and a fused multiply-add by `OpFmaKHR`, keep the device's rounding before Metal's flush: Metal flushes a result whose exact value lies below `2^-126` before rounding it. | `float_maths = FloatMathsPolicy::Native` | `float_maths = FloatMathsPolicy::Exact`; there off MoltenVK a `float` product of `±2^-126` is Metal's zero where the exact product lies below `2^-126`, decided inline without a branch by one `OpFmaKHR` on the operands' magnitudes, or by their bits without `OpFmaKHR` (`arith.float.tiny`, `dot`'s first product too), and `OpFmaKHR` is checked: a result at most `2^-126` in magnitude, zero included, is answered by a routine called only then, with `double` one that decides only the exact sum's sign and whether it lies below `2^-126`, else the integer `fma`; a module with either check, or with a fused multiply-add the translation computes itself (the `double` and integer routines without `OpFmaKHR`, and every `half` one), declares `SignedZeroInfNanPreserve` at 32 bits, and at 64 for a `double` routine, where the device supports it, so the device keeps the zero signs they answer (without it Vulkan lets the device answer a zero of either sign; else `DeviceSignedZeroInfNan`) | Vulkan's precision table (`Sin` 2^-11 absolute in [-π, π], `Exp` 3 + 2·\|x\| ulp, `Atan` 4096 ulp) against Metal's 4-6 ulp; division 2.5 ulp; `fma` one `half` ulp (a `float` one may round twice); an `f32` denormal in a maths function, `fast::fmod` among them, read as the device reads it (`DeviceDenormals`); a product or fused multiply-add whose exact value lies within half a unit below `2^-126` answers `2^-126` of its sign where Metal answers the zero of its sign, announced once per device as it is made. `NativeTranscendentals` (`native-transcendentals`), `NativeDivision` (`native-division`), `NativeFma` (`native-fma`), `FloatTininessAfterRounding` (`float-tininess-after-rounding`) | A2X 0.99-1.02 → 9.66-9.83 s, GPU 0.097-0.098 → 0.222-0.225 ms; Xgc14 0.87-0.90 → 12.6-13.6 s, 0.038 → 0.124-0.135 ms; Xgc15 0.63-0.65 → 9.40-9.54 s, 0.089 → 0.208 ms (measured before these modules declared `SignedZeroInfNanPreserve`, whose cost is not measured). Off MoltenVK a product's check is 12 instructions inline (23 without `OpFmaKHR`), and a checked `OpFmaKHR` adds its test, a branch and a call of a routine of about 40 instructions (with `double`), which the device's compiler copies into each use |
| A NaN result's bits are the device's (no select of Metal's NaN), and `fmin`, `fmax`, `min3`, `max3`, `clamp` and `saturate` are the device's `NMin`, `NMax` and `NClamp`. | `nan_and_min_max = NanMinMaxPolicy::Native` | `nan_and_min_max = NanMinMaxPolicy::Exact` | A NaN where Metal answers one, its bits the device's; the operand Metal picks but at two NaNs, `+0` against `-0`, and `f32` denormals. `NativeNanEncoding` (`native-nan-encoding`), `NativeMinMax` (`native-min-max`) | A2X 1.00-1.02 → 1.85-1.88 s, GPU 0.098 → 0.116 ms; Xgc14 0.88-0.89 → 1.59-1.63 s, 0.039 → 0.048 ms; Xgc15 0.65-0.66 → 1.25-1.28 s, 0.090 → 0.102 ms |
| `dot`, `mix` and `fast::sqrt` are the device's `OpDot`, `FMix` and `Sqrt`, not Metal's fused chain, its two fused multiply-adds and `x · rsqrt(x)`; a `dot` of other than two to four lanes sums the `OpDot` of each chunk of up to four lanes. | `rounding_order = RoundingOrderPolicy::Native` | `rounding_order = RoundingOrderPolicy::Exact` (MoltenVK takes Metal's order under every policy): Metal's chain over every lane of a `dot` of any width (test `wide_dot_is_metals_fused_chain_in_metals_order`) | `float` `dot` of n lanes within (3n − 1)·2^-24·Σ\|aᵢbᵢ\|, `mix` within 2^-21 of its two terms, `fast::sqrt` 4.5 ulp; `FMix` and `Sqrt` read an `f32` denormal as the device does (`DeviceDenormals`). `NativeDot` (`native-dot`), `NativeMix` (`native-mix`), `NativeTranscendentals` (`native-transcendentals`); a `dot` of other than two to four lanes within Vulkan's precision of its chunks' `OpDot` and sum, an `f32` denormal read as the device reads it, `DeviceArithmetic` (`device-arithmetic`) and `DeviceDenormals` | A2X 1.00-1.02 → 1.03-1.06 s, GPU 0.098 → 0.101 ms; Xgc14 0.88-0.89 → 0.97-1.02 s, 0.039 → 0.041 ms; Xgc15 0.65-0.66 → 0.73 s, 0.090 → 0.088 ms |
| A `half` result computed in `float` rounds through the device's conversion to 16 bits (three instructions), or its `OpQuantizeToF16` without `shaderFloat16`, not Metal's rounding in integer arithmetic (about twenty), on a device whose conversion is neither measured to be Metal's nor made Metal's by its float controls. ANV and RADV round through the conversion and its widening, each `NoContraction`, under `RoundingModeRTE`, `DenormPreserve` and `SignedZeroInfNanPreserve` for 16 bits: exact by the Vulkan spec under either policy, no note. Under the same controls a `half` `+ - *` is the device's 16-bit instruction, kept apart, and a `half` fused multiply-add is `OpFmaKHR` on 16 bits where `shaderFmaFloat16` is enabled, each correctly rounded by Vulkan's precision table, so neither rounds through the pair. NVIDIA's driver folds that pair (`DriverWorkaround::HalfConversionPairFolded`), so NVIDIA takes this row. | `half_rounding = HalfRoundingPolicy::Native` | `half_rounding = HalfRoundingPolicy::Exact` | One `half` ulp; 65504 where Metal answers infinity on a device rounding toward zero; a half denormal may flush; NaN bits and a zero's sign the device's. Or no rounding at all, measured: NVIDIA (RTX 3090, 615.71.09) drops the conversion pair, and lavapipe does in arithmetic on Mesa 26.2.4 with LLVM 23.1.1 and for a loaded texel on Mesa 26.0.8 with LLVM 21.1.8, whose arithmetic measured rounded, so the result keeps `float` precision, its differences compounding across operations, and a value past `half`'s range stays finite until a `half` store saturates it to infinity; the RTX 5080 Laptop GPU (615.71.09) rounds, measured: 8192 of 8192 narrowed sums are Metal's. `NativeHalfConversion` (`native-half-conversion`), `NativeHalfQuantize` (`native-half-quantize`) | A2X 1.39-1.42 → 10.09-10.14 s, GPU 0.137-0.153 → 1.27-1.36 ms; Xgc14 1.15-1.16 → 1.16-1.19 s, Xgc15 0.81-0.87 → 0.82-0.86 s, GPU unchanged (RTX 3090). Rounding natively by `OpQuantizeToF16` instead, which the RTX 3090 and lavapipe honour (Metal's answer on every operand tested), was measured and declined: A2X 1.3 s → 18-22 s, GPU 0.14 → 1.5 ms on an RTX 3090 |
| A `half` read from its bits (a load, a reinterpretation, a texel's bits) outside every loop is widened by the device (`UnpackHalf2x16`), not decoded exactly. | `half_rounding = HalfRoundingPolicy::Native` | `half_rounding = HalfRoundingPolicy::Exact` (with the row above) | A half denormal may read as zero where the device flushes them; NaN payload the device's. `NativeHalfConversion` (`native-half-conversion`) | Not measured alone. Against the exact decode, on an older tree: A2X 1.04-1.07 s with it against 1.27-1.28 s; Xgc unchanged |
| The `half` conversion converts the value itself (no `NoContraction` barrier), so the device's compiler may fold it into the instruction computing the value, rounding once. | `half_rounding = HalfRoundingPolicy::Native` | `half_rounding = HalfRoundingPolicy::Exact` (with the rows above) | The row above's: on NVIDIA (RTX 3090), and on lavapipe for a loaded texel (Mesa 26.0.8), the pair is dropped whether or not the barrier stands, so the barrier does not restore the rounding there. `NativeHalfConversion` (`native-half-conversion`) | At its merge: A2X 0.99-1.02 s without the barrier against 1.00-1.05 s with it; Xgc within a round's spread |
| A `half` texel's NaN is the rounding's own NaN, not the texel's bits (the keep's four instructions a texel are left out). | `half_rounding = HalfRoundingPolicy::Native` and `nan_and_min_max = NanMinMaxPolicy::Native` | `nan_and_min_max = NanMinMaxPolicy::Exact` (the keep under native half rounding); `half_rounding = HalfRoundingPolicy::Exact` also keeps it, at that row's cost. The same switch also gives Metal's full rule: `0x7e00` for every NaN but a one-texel read, nearest sample or gather of a 16-bit float texture, by the slot's format class read as data, a heap handle's class index and a runtime sampler's filter. A gather with a `half` result fetches its four texels where the class is a half-float one, since NVIDIA's gather hands back every NaN as `0x7fffffff`. **Not yet exact:** a cube gather, bicubic and emulated linear, fetched or compared samples; and on ANV (Iris Xe, Mesa 26.2.4) every 16-bit float texel's NaN, which its image decode answers as `0xFFC00000` (`0xFE00` as a `half`) before any keep, and on RADV (the Radeon 780M and the RX 7900 XTX, Mesa 26.2.4) a 16-bit float texel's NaN through a **sampler**, which answers `0xFFC00000` the same way where its image loads keep the texel's bits (`half-texel-nans-r16f`'s `nearest-sample` site differs, its `read` sites are exact): reading the texel's bits through an unsigned integer view would answer Metal's. Possible, not built: the runtime sampler's filter read as push data rather than specialized, which would spare the remake below. Measured on the RTX 5080: under the switch no `half` lane of the runtime lane's texel reads, samples and gathers differs from Metal's (1,469 differ by default). Under the switch `DeviceNanEncoding` is still announced; the consumer logs notes, so no device fact is built to drop it | NaN bits the device's (`0x7FFF` on the RTX 5080), where Metal hands back a 16-bit float texel's own bits on a read, nearest sample or gather, and `0x7e00` for a 32-bit or small float texel's NaN and a linear sample's. `NativeHalfConversion` (`native-half-conversion`); the read also records `DeviceNanEncoding` (`device-nan-encoding`) | Measured on top of the NaN switch's own row (a probe with the keep against one without): A2X 341,299 → 347,519 words, 1.82-1.86 → 1.81-1.87 s, GPU 0.115-0.117 → 0.115-0.116 ms, within noise; Xgc14 and Xgc15 read no `half` texel (words unchanged). The full rule (a probe with it against the same tree without it): A2X 347,519 → 372,429 words (+7.2%), creation 1.85-1.88 → 1.99-2.04 s; where a linear runtime sampler is bound at the draw the sampler word's filter is now specialized, so the first draw remakes (1.94-1.96 s) and its GPU time rises 0.115-0.117 → 0.124-0.125 ms; with a nearest sampler GPU 0.115-0.116 → 0.115-0.117 ms; Xgc14 and Xgc15 words unchanged. On the RTX 3090 the same pair: A2X creation 2.91-3.06 → 3.17-3.23 s, the first draw's remake 3.0 s and its GPU time 0.17-0.18 → 0.79-1.15 ms, with a nearest sampler 0.17-0.18 ms both |
| On ANV and RADV, which round a `half` through the device's conversion under the float controls, a `half` read from its bits (a load, a reinterpretation, a texel's bits) is the device's widening of them as a 16-bit float, loops included, and a `half` texel takes that conversion's NaN (no keep): every value but a NaN is exact by the Vulkan spec. | `nan_and_min_max = NanMinMaxPolicy::Native` | `nan_and_min_max = NanMinMaxPolicy::Exact`: the integer decode and the texel's keep; a `half` widened to `float` then answers Metal's `0x7fc00000` for every NaN, read from memory, a texel or a select alike | A `half` NaN widened to `float` has the device's bits (ANV quiets a signalling NaN, RADV keeps it), where Metal answers `0x7fc00000`; a `half` stored back unchanged keeps its bits. `NativeNanEncoding` (`native-nan-encoding`); a texel's, `DeviceNanEncoding` (`device-nan-encoding`) | ANV (Iris Xe, Mesa 26.2.4), A2X: making 4.06-4.36 → 5.76-5.87 s, 269,828 → 280,070 words, peak RSS 0.35 → 0.51 GB, GPU 4.48 → 4.62 ms (warm); RADV (Radeon 780M, Mesa 26.2.4): 379-467 → 535-665 ms, GPU within its clock states' spread |
| A fragment function without fast math returns a `half` color to a blended attachment: the device blends in its own arithmetic and rounds once, where Metal blends a 16-bit float attachment in `half` arithmetic (each factor, product and sum rounded to `half`). | `half_blend = HalfBlendPolicy::Device` | `half_blend = HalfBlendPolicy::Exact`: the function computes Metal's blend in the interlocked storage form, for a `half` output of one to four lanes into `R16Float`, `RG16Float` or `RGBA16Float` where that form is exact for the pipeline (RADV, ANV, NVIDIA, MoltenVK); elsewhere, and for lavapipe, 32-bit float, 8-bit, a module naming no math mode or a blend of the second source, the device's blend and its note stay | 2.5 `half` ulps at the largest operand on NVIDIA (the blend color and one minus it), two on RADV, ANV and lavapipe (16-bit float); two `float` ulps (32-bit float, one on NVIDIA); one or two codes (8-bit, extended range); MoltenVK stores Metal's bytes. Measured with and without the factors' magnitudes. `HalfOutputPrecision` (`half-output-precision`) | A probe (a `half4` over a 1920x1080 `RGBA16Float` target, One / OneMinusSourceAlpha): making 11.1 → 24.0 ms; GPU 25 → 82 us one draw, 506 → 2291 us 32 overlapping draws (the interlock orders them). RTX 3090 (the same draw): making 12.9 → 27.4 ms; GPU 45 → 115 us one draw, 648 → 3268 us 32 overlapping draws |
| A `half` or `float` read of a 16-bit float texture (`R16Float`, `RG16Float`, `RGBA16Float`, `RGBX16Float`) answers the device's decode of the texel, where Metal's `texture2d<half>` never widens and hands back the texel's own code: a NaN's sign, payload and signalling bit survive on Metal and are gone once a device has decoded the texel into `f32`. | `half_texel_bits = HalfTexelBitsPolicy::Device` | `half_texel_bits = HalfTexelBitsPolicy::Exact`: a colour texture slot the function declares `access::read` takes a second sampled-image binding of the texture's unsigned-integer alias (`R16_UINT`, `R16G16_UINT`, `R16G16B16A16_UINT`; `metal::format::half_bits_read_view`, `RuntimeFact::HalfBitsTexture`), and each plain fetch branches on the bound format's class: the alias's 16 bits widened in integer arithmetic where it names 16-bit float channels, the device's texel for every other class. A slot the function may sample, gather, compare, write or update atomically keeps the device's decode, the slot having one declaration; so does a fetch with a shader-side out-of-bounds check, a sparse arm, clamp-to-edge, level views or a snapshot | Every NaN texel collapses to `0x7e00`, so a signalling NaN, a payload NaN and a negative NaN all read as the one quiet positive NaN; no finite, zero, denormal or infinite texel differs. Measured on MoltenVK (Apple M3 Max, API 1.2, this chip's `f32` decode canonicalizing): 515 of 4,096 words at each of the three read sites of `half-texel-nans-r16f`, 1,026 for `-rg16f`, 2,045 for `-rgba16f`, largest difference 33,279 as unsigned words. On RADV (Radeon 780M and RX 7900 XTX, Mesa 26.2.4) the `read` and `read-through-sample` sites are exact and the sampler's `nearest-sample` answers its own negative quiet NaN `0xfe00`, which this switch does not reach (it covers the read, not the sample). ANV unaudited, expected to be MoltenVK's shape, through its image decode. Announced `device-nan-encoding` under either value: the note's clause is the bound class's, not the arm's | Unmeasured. One sampled-image descriptor and one view at each such slot, the texture's image mutable with the alias in its format list, and per plain fetch a branch on the class word, the alias fetch and about twenty integer instructions a channel |
| A computed `half` or `float` color blended into an extended-range attachment (`BGR10_XR`, `BGRA10_XR`, `RGB10A8_2P_XR10`) rounds the blend's source as a loaded one; Metal's compiler rounds it by how it computed it. Every device. | The in-shader blend's loaded rounding (no switch) | **Not built yet:** Metal's rounding per shape is probed, not modelled | One code. `ExtendedRangeBlendSource` (`extended-range-blend-source`), `ExtendedRangeFloatBlendSource` (`extended-range-float-blend-source`) | Not measured |
| A blend into a packed 16-bit attachment the device renders through its word view (`A1BGR5Unorm`, `ABGR4Unorm` on the RTX 5080; `WordAttachment`), which the function computes in the interlocked storage form, reads the attachment's codes, where Metal's tile holds the pass's clear and each earlier blend's result as a `half`, finer; under fast math it also sums a few states otherwise than Metal. Every device rendering such an attachment. | The in-shader blend over the codes (no switch) | **Not built yet:** the extended-range attachments' `half` shadow (`render/extended_shadow.rs`, `VariantKey::extended_shadows`) taken by word attachments: the runtime's shadow start and end, its load of the codes and the clear's halves, and the word's shadowed variant in `lower::fragment::fetch`, for a single-sampled pass (a multisampled one keeps the codes, as an extended-range one does). Fast math's regrouped sums have no model | One code: without fast math exact over a loaded attachment (the 2,891 states of the M3 sweep, every channel), under fast math 3,527 of 94,732,288 channels a code apart; after a clear or an earlier blend in the pass, a code apart where the two readings store different codes. `DeviceColorConversion` (`device-color-conversion`) | Not measured. Expected from the shadow's form: an `R16G16B16A16_SFLOAT` image of the pass's extent per such color (8 bytes a pixel, 16.6 MB at 1920x1080), set at the draw it starts by a fill or a load dispatch, and an 8-byte storage read and write per blended fragment inside the interlock beside the word's |
| A bicubic sample of an 8- or 16-bit normalized texture sums its taps exactly and rounds half up; Metal truncates its row terms, row sums and column terms first. Every device. | The exact sum (no switch) | **Not built yet:** Metal's datapath is pinned (17.75M lanes measured, no miss); its emitter is not written | One grid step (1/4080 at 8 bits) within about 2^-21 of a tie. `BicubicFilterArithmetic` (`bicubic-filter-arithmetic`) | Not measured (Metal's form is expected no larger) |
| A bicubic sample of an extended-range texture (`BGR10_XR`, `BGRA10_XR`, `RGB10A8_2P_XR10` and their sRGB twins) sums its taps' steps exactly, rounds half up and clamps into the format's decoded endpoints; within 0.014 of a grid step of a rounding tie Metal may round the other way. Every device. | The exact sum (no switch) | **None known:** no model of Metal's datapath tried fits; 3,317,760 lanes measured on the M3 Max (macOS 26.7) | One grid step (1/8160 a color lane, 1/8192 an sRGB color lane, 1/4080 alpha), 4,050 of the lanes measured, each within 0.014 of a step of a tie. `BicubicFilterArithmetic` (`bicubic-filter-arithmetic`) | No accurate form |
| A multisampled `Depth32Float` texture the program only reads (shader-read usage, no render target) is held in a `D32_SFLOAT` image on a device whose depth images may not keep its texels, where a single-sampled one of it is already held in `R32_SFLOAT`: `plan_depth_twin` left the multisampled case out on the reading that it has no writer, and `BlitCommandEncoder::fill_texture` is one. Metal's fill stores the pattern's bytes as given, a denormal included. The runtime's choice: no pipeline differs. | `multisample_depth_twin = MultisampleDepthTwinPolicy::Device` | `multisample_depth_twin = MultisampleDepthTwinPolicy::Exact`: the texels in a multisampled `R32_SFLOAT` image, which keeps every `float` the fills and the sample copies write, where the device holds one it samples and copies; elsewhere the depth image, announced as under the default. No depth twin beside it: Metal reads a `depth2d_ms` and compares nothing | A denormal a GPU write puts into a `D32_SFLOAT` image comes back as a zero on RADV, which keeps only a CPU store's and so carries no `DriverFact::DepthImageKeepsDenormals`, and a multisampled image takes no CPU store: a fill's sample holds `0x00000000` where Metal's stored `0x00000001`. Measured on both RADV chips over render-target textures, which this switch does not cover (`sample-fills-d32-2`, `-d32-4`, `-d32s8-4` and `runtime_lane::sample_fills::multisample_fills_write_metal_s_samples` all make the texture `RenderTarget | ShaderRead | PixelFormatView`; exact on MoltenVK, whose depth images keep a denormal). A read-only multisampled texture's fill is unmeasured: the same driver fact covers it, and a RADV run of a fill of a `ShaderRead`-only multisampled `Depth32Float` would confirm it. Announced `depth-twin` where the colour image is taken and `depth-denormals` at the write where it is not; the render-target shape stays open (known gaps) | Unmeasured. The image's depth attachment usage, which such a texture does not ask for; same extent, samples and layers, and no second image (a single-sampled one's `D32_SFLOAT` twin doubles its memory, this does not) |
| A texture of an ASTC LDR format on a device without that ASTC format reads a decoded twin of 8-bit codes, each channel the code nearest Metal's 12-bit value (`C >> 4` over 4095). (An sRGB form's twin holds Metal's own 8-bit codes under either value, which its slots decode through Metal's table: exact.) A texture of an EAC 11-bit format (`R11`, `RG11`, unsigned and signed) on a device without ETC2 reads one of 16-bit normalized codes, each nearest Metal's value (`c / 2047`, signed `c / 1023`). The runtime's choice: no pipeline differs. | `decoded_texels = DecodedTexelsPolicy::Compact` | `decoded_texels = DecodedTexelsPolicy::Exact`: Metal's values in an `R32G32B32A32_SFLOAT` twin (an EAC 11-bit one's in `R32_SFLOAT` or `R32G32_SFLOAT`) | Within 1/510 of Metal's value. An EAC 11-bit channel within half a 16-bit step (1/131070, signed 1/65534). `decoded-texel-precision`, announced where such a texture is made; `runtime_lane::astc_textures`, `runtime_lane::etc_textures` | The twin's memory, 16 bytes a texel against 4 (a 2048 x 2048 texture's chain: 85 MiB against 21 MiB), and as much again for a second twin once the texture is viewed in its other colour space; an EAC 11-bit twin's 4 bytes a channel against 2. The CPU decode of that chain of 8 x 8 blocks on one core of the Iris Xe host: 79 ms against 68 ms. Sampling not measured |
| A heap's colour texture that Metal keeps losslessly compressed (`allowGPUOptimizedContents`, no `ShaderWrite`, `PixelFormatView` or `ShaderAtomic` usage, 16 texels or more a side), on a driver whose compressed images may keep their texels apart from the memory they are bound to (`DriverFact::CompressedImagesHideAliasedWrites`: ANV), is the driver's compressed optimal image, its compression and fast clears kept. A texture Metal keeps uncompressed takes the uncompressed image under either value. The runtime's choice: no pipeline differs. | `heap_compression = HeapCompressionPolicy::Compressed` | `heap_compression = HeapCompressionPolicy::Uncompressed`: an image the driver keeps uncompressed in the heap's bytes, as for a texture Metal keeps uncompressed: host-transfer usage where the device enabled `hostImageCopy` (`heap-texture-host-transfer`), else a linear image (`heap-texture-linear`), else the optimal one (`heap-texture-hides-aliased-writes`) | A write through a buffer or another texture over the texture's bytes may leave its texels as they were or change them otherwise than as written, where Metal's texture reads opaque magenta (`0xffff00ff`) in every texel of drawn content the write covers and the written bytes where it held only a clear; the texture's own writes may show in those bytes compressed. Within Metal's contract: Metal documents such a texture's layout as its own (`allowGPUOptimizedContents` lets it "adjust the private layout"), so no write through another view over its bytes has a defined result there either. `HeapTextureCompressed` (`heap-texture-compressed`), announced once per device where such a texture is made; `runtime_lane::heap_textures` | ANV (Iris Xe of RPL-P; `heap_textures::heap_compression_costs`), a 2048x2048 `RGBA8Unorm` render target in a `Private` placement heap: a pass that clears it 26.9 us against 676.2-676.5 us; a copy into another such texture 647.7-649.1 us against 673.9-693.2 us |
| A signed normalized color attachment (`R8Snorm`, `RG8Snorm`, `RGBA8Snorm`, `R16Snorm`, `RG16Snorm`, `RGBA16Snorm`) or an `RGB9E5Float` one, single-sampled and loaded by a pass that draws, keeps the texels its draws do not write as they were, where Metal's tile store writes each 32x32 tile a draw reaches back through its values: a channel's minimum code (-128, or -32768) as the code above it (-127, -32767), and a shared-exponent word whose largest mantissa is below 256 under an exponent above 0 normalized (its mantissas shifted up as the exponent steps down). The runtime's: no pipeline differs. Not on a driver whose pass stores them as Metal's (`DriverFact::TileStoreIsMetals`: MoltenVK on Apple's GPUs), under either value, with no load pass and no notice. | `tile_store = TileStorePolicy::Device` | `tile_store = TileStorePolicy::Exact`: the extended-range attachments' load pass (`render/unused_bits_load.rs`) takes such an attachment too, raising each minimum code and normalizing each word in the tiles the pass's draws reach (`bind::unused_bits_module` of the format); one it cannot take (a texture held as a shadow, a sparse one, or one whose image the device does not copy both ways) keeps the device's, announced | One code, only at a minimum code, which no Metal write leaves (uploaded or copied bytes); none in value for a shared-exponent word (another encoding of the same value). Within Metal's contract: `MTLLoadAction.load` is documented as "The GPU preserves the existing contents", which the kept texels are; the rewrite is Apple's tile store's own. `SnormMinimumKept` (`snorm-minimum-kept`) and `SharedExponentKept` (`shared-exponent-kept`), announced at each such draw; `runtime_lane::snorm_tile_store` | The load pass before each such pass: the render area copied into a scratch buffer and back, with a dispatch. Timed for the extended-range attachments, whose load pass runs under both values: at 1920x1080 on the Radeon 780M (RADV), a pass that loads one and draws a full-screen triangle takes 1,959 us of GPU time for `BGRA10_XR` against 135 us for `RGBA16Float`, and 1,341 us for `BGR10_XR` and 1,378 us for `RG16Bfloat` against 69 us for `RGBA8Unorm`: about 1.3 ms a loaded attachment for a 4-byte texel and 1.8 ms for an 8-byte one. Under `Exact` a snorm or `RGB9E5Float` attachment takes the same pass (not timed) |
| A multisampled color attachment whose format has Metal's resolve rule at its sample count (unorm, snorm, sRGB, 32-bit float and packed 16-bit unorm formats at every count; 16-bit float, `RG11B10Float` and `RGB9E5Float` ones at 2 and 4) resolves by the device's `AVERAGE` in the pass, which leaves the average's rounding, an sRGB average's space and what a NaN or infinite sample makes of it to the device. The runtime's: no pipeline differs. Not on a driver whose average is Metal's (`DriverFact::ColorResolveAverageIsMetals`: MoltenVK on Apple's GPUs, at 2 and 4 samples), under either value, with no notice; nor for an attachment a fetching pass's sample planes hold as it ends, which resolves from them by Metal's rule under either value. | `resolve = ResolvePolicy::Device` | `resolve = ResolvePolicy::Exact`: after the pass, one render pass instance per resolved attachment, between barriers, reads its samples (as an input attachment, or from a copy of them into a scratch image where its image takes no input attachment usage) and writes Metal's rule (`color_resolve.rs`), announced `shader-resolve` | At 2 and 4 samples, every unorm, snorm and packed 16-bit unorm channel within one code of Metal's and an sRGB channel within two (NVIDIA's at 4 samples; one elsewhere), on RADV (Mesa 26.0.8, the Radeon 780M and the RX 7900 XTX), ANV (Mesa 26.2.4, the Iris Xe), NVIDIA (615.71, the RTX 5080 Laptop GPU) and lavapipe (Mesa 26.2.4, at 4 samples). Each rounds an 8-bit unorm tie down (RADV and ANV: 32512 of `R8Unorm`'s 65536 pairs of samples one code below at 2 samples; NVIDIA 31872; lavapipe 8205 of 131072 texels at 4). A float channel differs in its last bits (counted, not bounded), and wholly at a NaN or infinite sample. `DeviceRoundedResolve` (`device-rounded-resolve`), announced at the first such pass; `runtime_lane::resolve_rounding::the_device_resolve_rounds_as_announced` | The extra pass per resolved attachment. A 1920x1080 frame of 97 command buffers of one pass of 4 indexed draws, every 4th pass resolving an `RGBA8Unorm` x4 attachment (`runtime_lane::frame_costs::frame_costs_scout_gpu`), GPU busy a frame, exact against the device's: the Iris Xe (ANV) 2.52 and 2.41 ms against 1.92 ms (a resolved pass's command buffer 77 against 51 us; the guest's queues and imports 3.04 and 2.77 against 2.43 ms); the Radeon 780M (RADV) 2.33 to 2.78 against 2.00 to 2.13 ms (63 to 75 against 45 to 48 us); the RX 7900 XTX 1.38 against 1.36 ms. Untimed, the frames were bound by the encoding thread's CPU, which neither changed |
| Batches of command buffers committed on one queue that the scheduler finds ready together (a backlog) go to the driver in one `vkQueueSubmit`, at most four, only the last signalling the queue's timeline, on a driver whose queue merges a submission that signals nothing into the next (Mesa's: ANV, RADV, lavapipe, NVK, Turnip and kin): one kernel submission, where ANV on i915 lists every `VkDeviceMemory` in each (the consumer's guest drag on the Iris Xe: about 300 entries per execbuffer, 260 of them its host imports, at 112 to 128 execbuffers a frame). A batch's event signals ride to the group's end, which signals each event's semaphore to the highest index the batches held signal it to (an event's indices rise in commit order). A batch that signals a binary semaphore, takes a fence, binds a sparse update, does first work, names the caller's wrapped image, or waits for anything not seen complete (an event, another queue's work) goes alone, after the batches held: a batch with a wait leads a group, never joins one. So that checked command buffers (the consumer checks each) are found ready together, the submission checks of up to three batches right behind a line's head that goes run with the head's, in commit order, none committed after another queue's batch: such a check runs before the previous batch's `vkQueueSubmit` rather than after it, in the scheduler pass that submits its own batch. A pass that finds no batch ready sends the batches held, unless two or more earlier submissions of their `VkQueue` are not seen complete: the GPU runs those first in any case, so the batches stay held, and those committed meanwhile join them, until one submission is left in flight (a completion turn wakes the scheduler at its value), the group fills, a later batch sends it, a head waits for the GPU, or a thread waits for one of their command buffers' completion, which sends them at once; a batch that ends a command buffer with a switch of storage forms riding on it, which settles by its submission's outcome, is never kept so. A held batch that ends its command buffer is `Scheduled` as it is held, its submission handlers run first, then its scheduled handlers, and its kernel end time set, as an M3 Max schedules a command buffer committed behind running work 0.04 to 0.18 ms after its commit. The runtime's: no pipeline differs. | `completion = CompletionPolicy::Grouped` | `completion = CompletionPolicy::Exact`: one submission per batch, each signalling its own values; each line's head checked alone | Each command buffer of a submission but the last is seen complete (status, completed handlers, waits), and the event values it signals are reached (by the GPU waits on other queues, the host's waits, value reads and listeners), once the last completed: late by the GPU time of at most three later batches (about 0.1 to 0.2 ms each in the consumer's compositor frames on the Iris Xe; about 20 ms behind a 20 ms batch), never early, in commit order. Metal's contract states no bound on when completion is seen (an M3 Max reports a command buffer complete, its handlers, status, waits and event listeners alike, 0.04 to 0.10 ms after its GPU end), and no member but the first waits, so no wait holds a member back: a valid program sees only later completion, as of slower GPU work. A batch kept while earlier work runs is seen complete within the same bound, and, where the one submission left in flight ends before its group reaches the GPU, late besides by that gap: at most a completion turn's reaction and the group's `vkQueueSubmit`. Commands, barriers and GPU times are unchanged. `GroupedCompletion` (`grouped-completion`), announced once per device at its first such submission; `runtime_lane::grouped_submissions` | ANV (Iris Xe of RPL-P; `frame_costs::frame_costs_over_imports`, the checked device, 67 command buffers a frame, GPU at 467-783 MHz), with 520 host imports live, where the Exact form's submissions fall behind the commits as the guest's do: 32.0-33.2 kernel submissions a frame against 67.0, 3.4-3.5 ms of execbuffer ioctl a frame against 6.9-7.9 ms, the scheduler thread's CPU 61-63 us a command buffer against 121-136 us, a frame through completion 4.7-5.9 ms against 8.6-9.7 ms; in the guest's shape with 520 (a second queue taking every 20th command buffer, each signalling a shared event, every 10th waiting for one): 26.2-26.7 kernel submissions a frame against 67.0, 2.8 ms of ioctl a frame against 6.9-7.0 ms, the scheduler 52-53 us a command buffer against 122-123 us, a frame through completion 6.0 ms against 8.7 ms; with 260, where it keeps up, 64.9-66.3 kernel submissions a frame against 67.0, the CPU and wall times alike. The cost (`grouped_submissions::grouped_completion_delays`, 8 command buffers of 0.3-0.5 ms of GPU work ready together): the host saw each but a group's last complete later by the later ones' GPU time, the last 0.1-0.3 ms later than alone (the completion turn's per-command-buffer work for the group, done at its end). Keeping a group while earlier work runs (UHD 770 of RPL-S, ANV, the guest's shape with 520 host imports, a submission handler on every command buffer and a 16 MiB fill ahead of each, about 1.3 ms of GPU each; the kept form against the batches sent as found): committed every 1 ms, 18.1-18.2 groups kept a frame, 24.9-25.0 kernel submissions a frame against 57.9, 2.18-2.90 ms of execbuffer ioctl a frame against 4.64-5.43 ms, the scheduler thread's CPU 67-82 us a command buffer against 113-118 us, the GPU idle within a frame's span 0.61 ms against 0.68-0.69 ms, a frame through completion 89.5-89.6 ms alike, completion seen later (GPU end to completed handler: mean 1.82-1.86 ms against 0.43-0.46 ms, p50 1.57-1.60 against 0.11-0.13, p99 4.35-4.37 against 4.09-4.11, max 4.52-4.55 against 4.18-4.23); committed at once, 18.0 kernel submissions a frame against 23.4; with no fill, 18.0 against 20.6, the GPU idle 0.59 ms against 0.58-0.66 ms. In the consumer's guest drag on the Iris Xe (i9-13900H; three passes each way): 24.4-27.5 kernel submissions a frame against 42.4-43.8, the scheduler thread 0.4-1.7 ms of CPU a frame less, 15.7-17.3 frames a second against 16.4-17.6: the frame rate alike within noise, as the scheduler thread does not bound that frame |
| Batches of command buffers on one `VkQueue`, on a driver measured to start no kernel submission of a queue before the one before it completed (ANV, RADV and NVIDIA's; `batch_order.rs`), are ordered by their ends: each ends with a barrier ordering its work before every later command of its `VkQueue` (the exact form's end orders it before the host's reads and writes alone), so a batch that depends on an earlier one there begins with no barrier of its own, often no command buffer of barriers, and its first encoder's GPU start is written at the bottom of the pipe, once everything before it completed. Metal starts an independent command buffer committed behind running work while that work runs (an M3 Max under macOS 27.0.1, probe `gpustart`); those drivers ran two batches of independent work one after the other under every shape measured (no end barrier, the host's or the widened one; in one `vkQueueSubmit` or two, merged into one kernel submission or not), where two dispatches of one command buffer ran side by side. The runtime's: no pipeline differs. | `batch_order = BatchOrderPolicy::Ends` | `batch_order = BatchOrderPolicy::Exact`: a barrier for each dependency of a batch's work on an earlier batch's of its `VkQueue`, the end's barrier to the host's reads and writes alone, the GPU start written at the top of the pipe; every other driver takes it under either value | None: on those drivers each batch starts once the one before it completed under either value, so the work's order, its results and its GPU times are the same, and a batch's GPU start is the end of the work before it, when such a driver starts it. Not announced; `runtime_lane::commit_work` and `runtime_lane::gpu_times` test both values | UHD 770 (ANV, RPL-S), the consumer's frame shape over 520 host imports, 67 command buffers a frame, Grouped completion, the widened end barrier against the exact form (a probe build of each): with no fill ahead of each command buffer, 67.0 command buffers submitted a frame against 129.3-129.7, the GPU's span a frame 1.84-1.90 ms against 1.99-2.06 ms, a frame through completion 2.13-2.22 ms against 2.27-2.38 ms; with a 1 MiB fill ahead of each, 134.0 against 262.9, 8.03-8.22 ms against 8.32-8.42 ms, 9.00-9.15 ms against 9.25-9.36 ms. Radeon 780M (RADV, Mesa 26.0.8), this switch: with no fill, 67.0 against 128.9-129.0, a frame through completion 48.3-48.6 ms against 42.6-47.0 ms; with a 16 MiB fill committed at once, 134.0 against 251.0-251.1, 50.6-50.8 ms against 50.8-57.3 ms; committed every 1 ms, 134.0 against 249.7-250.2, 71.3-83.6 ms against 106.3-113.6 ms (each kernel submission there leaves about 2.5 ms of the GPU idle over 520 host imports, which sets these frames) |

All at once: `DevicePolicy::exact_answers()` sets every switch above but sampling's (validation
adds `sampling = SamplingPolicy::Exact`). Its measured cost: A2X 0.99-1.04 → 17.5 s,
GPU 0.098 → 0.267 ms; Xgc14 0.86-0.87 → 4.95-5.01 s, 0.039 → 0.105-0.110 ms; Xgc15 0.63-0.66 →
3.06-3.15 s, 0.090 → 0.194-0.197 ms.

Speed choices that change no answer (the roll pass's floor; a specialization measured free) are not
rows.

## Not speed choices: device gaps

No switch gives Metal's answer here: the device lacks what an exact form needs, or no exact form is
known. The rest of the differences are docs/API.md section 9's precision notes.

| What it is | Why there is no switch | Note, code |
|---|---|---|
| `f32` denormals in the operations left to the device under every policy: a conversion to or from `double`. Metal reads them as zero. | Vulkan lets each device keep or flush them; Metal runs no `double`, so its reading of a conversion is not measured. Native device maths and the device's order record the note too (their rows above). Every other `float` operation reads them as Metal on every device (test `the_accurate_forms_read_denormals_as_metal`). | `DeviceDenormals` (`device-denormals`) |
| A denormal 32-bit float texel the device's sample or gather reads as zero (RADV's gathers and nearest samples, NVIDIA's gathers, ANV's nearest samples), where the module cannot fetch the texels itself. | The device's filter decides; no fetch form reaches filtered, cube, heap or runtime-sampler samples. | `DeviceTexelDenormals` (`device-texel-denormals`) |
| Depth values outside [0, 1] stored clamped (RADV, ANV) on a device without `VK_EXT_depth_range_unrestricted`. | Vulkan leaves them undefined without the extension: enable it. | `RestrictedDepthRange` (`restricted-depth-range`) |
| A multisampled 32-bit float depth written by a blit copy from a buffer or a colour texture, or a separate depth texture's copies to and from its pass's scratch image, on a device whose depth/colour copies are missing or wrong (lavapipe, MoltenVK, NVIDIA's GA10x, a device without `VK_KHR_maintenance8`): the values its driver's fragment depth changes (MoltenVK's, NVIDIA's), or on a driver not measured may change (outside [0, 1], denormals, -0), land changed. | Vulkan writes a multisampled depth image's samples by no other means there but a clear; a driver measured to store every fragment depth unchanged (lavapipe's measured releases) is exact. | `DrawnDepthCopy` (`drawn-depth-copy`) |
| Signed zeros, infinities and NaNs where the device lacks `SignedZeroInfNanPreserve` at the operation's width. | A device capability. | `DeviceSignedZeroInfNan` (`device-signed-zero-inf-nan`) |
| Every device but MoltenVK interpolates varyings by its own rounding, up to about `1.5e-4` from Metal's (`3e-4` on lavapipe). MoltenVK hands the varyings to Metal. | The exact path needs Metal's interpolation formula, which is not known. | `DeviceInterpolation` (`device-interpolation`) |
| A fast-math function's `half` color into a blend: within three `half` ulps (16-bit float) or two `float` ulps (32-bit float). A fetched 16-bit float attachment's blend, computed in the pixel interlock, where Metal folds the output's own arithmetic into it or regroups it: within 2 `half` ulps at the operands' magnitude, 16 at the stored value's under cancellation; exact elsewhere. | Metal's compiler regroups the blend per pipeline, following its passes over the whole function; no form the function and the blend state decide is Metal's. | `HalfOutputFastMathBlend` (`half-output-fast-math-blend`) |
| A fetched blended attachment of a format whose Metal blend arithmetic is not built (32-bit float, 16-bit normalized, 8-bit signed normalized, 10-bit, packed float, `A8Unorm`; an 8-bit unorm one through fewer than four output lanes; a 16-bit float or 8-bit unorm one under a blend of the second source), on a device that orders fetch only through the pixel interlock (MoltenVK, RADV, ANV): the shader blends in `float`, each step rounded, the source and blend color clamped to a normalized format's range, the device's conversion storing the result. An 8-bit sRGB one under a blend of the second source, a one- or two-channel one under a fast-math lerp, and one written by an output of fewer than four lanes: the 8-bit unorm blend's rounded forms over the decoded texel, encoded by Metal's table (Metal's without fast math, at four lanes). Bound not measured. | No Vulkan blend applies to the interlock's storage write, and Metal's blend arithmetic for these formats is not measured, so no exact form is known. | `UnmeasuredFetchBlend` (`unmeasured-fetch-blend`) |
| Float simdgroup reductions sum in the device's association order. | Apple's order is not modelled. | `ReductionOrder` (`reduction-order`) |
| A vertex or fragment function keeps more than 16 KiB a thread of private memory, up to the 192 KiB Metal holds, on a device whose Function storage was not measured: it runs on the driver's scratch, which past what the driver holds answers wrongly or hangs the GPU with no error. | Vulkan states no amount of Function storage; only a measurement of the device does. No other form is exact: a fragment function's helper invocations may not write the memory it would move to, and a vertex function's, a compute pre-pass, is not built. | `FunctionStorageAssumed` (`function-storage-assumed`) |
| A render pipeline of more color attachments than the device draws at once (the lesser of `maxColorAttachments` and `maxFragmentOutputAttachments`, four at Vulkan's floor; Metal takes eight) draws each draw once per group of that many, each group in a render pass of its own over a twin of the pass's depth and stencil. Every variant's vertex stage is one module, `Position` invariant, and every fragment variant's `FragDepth`, `SampleMask` and `FragStencilRefEXT` are `Invariant`; groups past the first may differ at discard or depth edges if the driver compiles the variants differently. | The device holds no more attachments in one subpass. Vulkan's invariance rules make only identical pipelines repeat exactly, and the groups' fragment modules and attachments differ. | `ColorAttachmentGroups` (`color-attachment-groups`) |
| Command buffers that share no resource run one after another, where Metal overlaps them while the GPU has room. | The runtime orders nothing between them and runs at each driver's floor for one completion signal per command buffer; a signal shared by several would hold an earlier one's completion for the later ones' work. | None (timing only) |
| On MoltenVK a light queue's command buffer waits for the latest one another queue committed, where Metal waits for none: MoltenVK fences every encoder of every queue device-wide when argument buffers and a residency set are on. | The runtime needs argument buffers, and MoltenVK has no setting for its residency set; `runtime_lane::queue_storms` judges the runtime against Metal under those fences. | None (timing only) |
| A `Counting` visibility result of a draw writing more than one color attachment, on ANV (Mesa 26.2.4): Metal's count multiplied by the colors written, not divided back. `Boolean` results stay exact. | The driver counts each sample once per color output (`RuntimeDriverDefect::OcclusionCountPerColorOutput`), no documented Vulkan form counts it once, and a refusal would withhold the Mac2 and Metal3 families. | `VisibilityCountPerColor` (`visibility-count-per-color`) |
| On MoltenVK a no-copy buffer or heap released with no keep-alive while batches are in flight runs its deallocator once the batches submitted before its release completed, and the program's own submissions through `CommandQueue::with_vk_queue` before it, where Metal runs it at the release. | MoltenVK 1.4.1 fails pending work that reads device addresses where memory a device-address buffer lay in is freed (`RuntimeDriverDefect::FreedAddressMemoryFailsPendingWork`), so freed memory waits for the batches in flight (`WorkCounts::freed_memory_parked`). | `HostDeallocatorDelayed` (`host-deallocator-delayed`) |
| On MoltenVK a `Shared` or `Managed` texture whose format packs more than one texel into a block (every block-compressed family) keeps its image out of host-transfer memory, so each CPU copy of it takes a submission of its own instead of the copy the driver makes as the memory binds. The stored texels are Metal's either way; what changes is the work, in `WorkCounts::command_buffers_begun_batch_work` and the `command_buffers_begun` and `command_buffers_submitted` that carry it (`bc1-rgba`, four 16 x 16 `BC1_RGBA` levels: `command_buffers_begun` 8 on the bind-time path and 15 off it; `fresh-texture-bc1-rgba-managed`: 3 and 5). | MoltenVK 1.4.2 binds such an image through a `flushToDevice` that `replaceRegion:`s every subresource at one block's bytes a row instead of the row of blocks, tripping AGX's `bytes_per_row >= used_bytes_per_row` assertion (`RuntimeDriverDefect::HostTransferImagesCopiedAtBind`). The copy is the driver's own and no call of the program's asks for it, so the only way not to provoke it is not to ask for the image; every other driver keeps the host-transfer path, as does a device without `VK_EXT_host_image_copy`. | None (cost only) |
| On ANV (Mesa 26.2.4) a pipeline whose largest stage module, under the consumer's exact forms, passes 400,000 words is retranslated with those forms dropped until it fits: native sampling first, then every module-growing exact answer native. Every kind is fitted: render (vertex and fragment), mesh render (object, mesh and fragment, or the stages that emulate them), compute and tile render; the notice names the state's kind. On a large enough module the driver's register allocator reads past its interference bitset and the process faults at pipeline creation. | ANV's `brw` register allocator faults on such modules (`DriverWorkaround::RegisterAllocatorOverflowsLargeModules`), no documented Vulkan form bounds a module's register pressure, and a refusal would withhold the consumer's large shaders. | `ExactFormsOverCompilerLimit` (`exact-forms-over-compiler-limit`) |

## Accuracy switches whose default is Metal's

- `unorm8_linear_filter = Unorm8LinearFilterPolicy::Exact` (the default) keeps Metal's filter for
  an 8-bit unsigned normalized texture's linear sample where Metal's sampler is emulated
  (`sampling = SamplingPolicy::Exact`). `unorm8_linear_filter =
  Unorm8LinearFilterPolicy::DeviceWithinOneCode` hands it to a device measured within one code
  (`DeviceUnorm8LinearFilter`, `device-unorm8-linear-filter`): it bought no GPU time on the
  consumer's fragments and cost module size and making time.
- `render_lowering = RenderLoweringPolicy::AtCall` (the default) lowers each render encoder call
  inside the call, which returns every refusal. `render_lowering = RenderLoweringPolicy::Worker`
  logs the call once its checks at the call pass, and lowers it later: in batches of 32 calls on
  the device's lowering thread while the program goes on (a program 16 batches or 4 ends ahead
  waits), the encoder's end there too, and what that thread has not taken on the program's thread
  at the next call that follows the end in its command buffer. `commit` returns after its checks
  at the call, its own work run after the end on the thread that lowers it, and later commits of
  the device behind it. A refusal only the lowering finds then fails the command buffer, naming
  the call (`CommandBufferError::CallRefused`), rather than returning from it (docs/API.md section
  1). The runtime's choice: no pipeline differs.
- `pointer_domain`, `handle_domain`, `containment` and `isolation` choose how resources resolve and
  how guests are contained, not an answer of Metal's arithmetic or sampling.

## Costs of matching Metal

Where matching what Metal observably does costs time, the runtime matches it and the cost is
recorded here.

- **`commit` returns at once.** Metal's `commit` returns before its command buffer is scheduled.
  The runtime's commit hands the batches to the device's scheduler thread, which submits them
  (`scheduler/inbox.rs`), so no driver call holds `commit`. Where that thread sleeps, a commit's
  batches wait for its wake: three command buffers committed back to back and waited for take a
  median 90 to 160 µs longer on ANV (Iris Xe), 5 µs on the RTX 5080 Laptop GPU and none on the M3
  Max. On the Iris Xe, frames of 24 command buffers committed back to back
  (`runtime_lane::commit_costs`, `frames`) take a median 5.7 to 6.0 ms against 3.2 to 4.5 ms with
  the submissions on the committing thread, the GT held at its efficient 300 MHz by its governor
  (GuC SLPC). A submission check or submission handler, which run on that thread, holds back every
  other command buffer's submission while it runs.
- **Idle dma-buf imports are released.** RADV lists every imported memory in every kernel
  submission, so a no-copy buffer's dma-buf import that only blits have read is released after its
  idle window (2 s, doubled at each import again up to 64 s). The encoding thread pays its import
  again at the first blit after the window (HOWTO.md section 6).
- **A pass in color groups runs once per group** (the `ColorAttachmentGroups` row above), the later
  groups' draws again over a twin of the depth and stencil, made for the pass and loaded by a copy
  or a clear (`render/color_groups.rs`). On lavapipe restricted to four colors
  (`runtime_lane::color_groups::color_group_costs`), 16 fullscreen draws into five 512²
  `RGBA16Float` colors took 1.04 times the same draws into four from commit to completion, and 1.51
  times with a loaded `Depth32Float_Stencil8` the draws write.
- **A draw that reads its own pass's attachment reads a copy, a snapshot or a split.** A sampled
  slot binding of a texture the render pass instance renders reads a copy taken before the
  instance (`texture/attached_reads.rs`): the texels around the draw's scissors where it reads at
  its own pixels alone, else every texel or what the instances since drew. After a barrier inside
  the instance, an own-pixel read takes a snapshot drawn in its subpass (`render/snapshot.rs`,
  docs/API.md "Snapshot reads") or splits the instance; both are exact. Over a texture whose image
  lies over its buffer's bytes (a linear image over an ordinary buffer's packed rows) it takes the
  snapshot at any area, where
  the device makes that linear image with feedback loop usage
  (`WorkCounts::feedback_loop_alias_images`; NVIDIA's drivers make it without, and a surface's
  readers split there): on the Iris Xe (`runtime_lane::attachment_read_costs::snapshot_crossover`,
  a 1920x1080 surface) 0.56 to 0.99 times the split's GPU time through scissors of 960x540 and up,
  at most 1.08 times through the whole target and 1.2 times through a 320x180 scissor it covers
  half of or more.
  Elsewhere the rule weighs the snapshot draw at 20 bytes of the split's copy for each pixel its
  scissors bound, against a copy of every texel of the color (an upper bound of the split's copy,
  which is cut to the scissors or to what changed for a one-level, one-slice texture), which fits
  the RX 7900 XTX and the RTX 5080 Laptop GPU; on the Radeon 780M the split is faster from about
  14.5%, 17% and 42% of the target for 4-, 8- and 16-byte texels, so the rule costs up to 36% there
  (`RGBA16Float` at 30% of a 1920x1080 target), and on the Iris Xe from an eighth of the reader's
  scissor covered, whatever its area, so the rule costs up to 2.1 times there (a 320x180 scissor
  covered whole over a private 1920x1080 `BGRA8Unorm` texture).
- **A `Depth16Unorm` depth whose code the fragment stage writes is tested after the stage.** Where
  the device's conversion is not measured to store Metal's code
  (`DriverDeviation::MetalDepth16Stores`, or `MetalDepth16BiasedStores` for a biased draw), a
  single-sampled draw writes each depth as Metal's code from its fragment stage
  (`ShaderUnormDepth`), so a fragment the depth hides still runs it. A bias of a known side
  declares it (`DepthGreater` or `DepthLess`) and the device may test first; an unbiased draw and
  a bias whose side a triangle's slope decides keep the cost. On the Iris Xe 16 full-screen layers
  drawn front to back under a constant bias took 9.42 ms of GPU time with no side declared and
  1.85 ms with one, against 1.46 ms unbiased through the device's conversion
  (`runtime_lane::unorm_depth_bias`).
- **A stored negative or NaN depth is tested by its magnitude.** A switched-on single-sample
  `Depth32Float` texture's pass that loads its depth takes two copies through a scratch buffer and
  a dispatch over the render area before it, and the same after it where it writes the depth; a
  sign strip (two copies and a dispatch, or a draw per layer when multisampled) follows a pass with
  depth writes where no stand-in runs (`texture/signed_depth.rs`). A texture never switched on pays
  a flag read per pass.
- **A texture over a no-copy buffer's host pages, over a host-visible buffer longer than 32 MiB, or
  over a buffer whose rows lie with bytes between them,** is a shadow on every device, whatever its
  usage (`shadowed-texture`): its texels copied in before each use and merged back after each write
  (only those inside the draws' scissors for a pass that takes it as a loaded attachment alone). An
  image bound to host pages starts `UNDEFINED` (`VUID-VkImageCreateInfo-pNext-01443`), and its first
  move may read and write every byte bound to it, which the host could keep only by undoing a CPU
  write made meanwhile, where Metal keeps it; an image over padded rows holds the bytes between
  them, which a render pass's load and store operations may rewrite too. A host-visible buffer
  longer than half a pooled block (32 MiB) lies in fresh anonymous pages imported by their pointer
  where the device imports them and the memory class's best type for the buffer takes such an
  import (`fresh_pages.rs`; on a discrete card the write-combined class's best type is a
  device-local host-visible one that a host-pointer import cannot bind, so that buffer is placed
  as ever and cleared on the host), which read zero as mapped, so its creation
  writes none of its bytes, and the CPU's first touch of its pages is paid where the program makes
  it, as on Metal (host_ab on the M3 Max, 64 MiB: made in 0.01-2.2 ms against 3.2-4.7 ms of
  allocation and memset, its first CPU write 2.4-6.5 ms against 0.7-1.4 ms, Metal's 3.7-7.4 ms):
  such a buffer's textures are shadows for the same reason (`WorkCounts::shadows_over_fresh_pages`).
  On the UHD 770 (ANV) a 1080p `BGRA8Unorm` surface's pass
  that loads, draws and stores (`guest_frame_costs`) took 1.9 ms from commit to completion, 1.4 ms
  of it the copies (copy in 0.21 ms, keep 0.60, merge 0.60), against 0.4 ms into a `Private`
  texture; a linear image over the pages took 2.6-3.1 ms, its draws slow in host memory. Twelve such
  passes in one command buffer took 21 ms against the linear image's 27.5 ms, and 0.4-0.5 ms more of
  the process's CPU a frame (1.2 ms against 0.7): about 20 µs a pass on the encoding thread, 10 µs
  on the queue's waiter and 10 µs on the recorder and reaper together. On ANV making a 1307x512 tile
  took about 135 µs as a shadow against 41 µs as an image; a frame sampling four `BGRA8Unorm` tiles
  over no-copy pages (`surface_texture_costs::read_only_padded_costs`) took 3.2-5.3 ms with the
  textures made once and 3.4-7.1 ms with new ones each frame, where images over the rows took
  3.0-6.1 and 4.6-8.6 ms. An ordinary buffer's packed rows avoid it.
- **On MoltenVK, a texture over a buffer that renders or that blits write keeps a shadow.** MoltenVK
  1.4.1 and 1.4.2 report a format's linear-tiling attachment and blit-destination features from
  uninitialized memory, so no MoltenVK description states them
  (`DriverFact::LinearTilingFeaturesUninitialized`) and such a texture is a shadow
  (`shadowed-texture`).
- **A view of another format without `PixelFormatView` usage** is a shadow image copied at each use
  (`shadowed-texture-view`): about 50 to 95 µs a command buffer for a 64x64 texture. Making the
  texture with `PixelFormatView` usage avoids it. A view in other channels of its image's bits (an
  `A8Unorm` view of an `R8Unorm` texture) binds the image itself through a swizzle, and only a pass
  that renders into it renders through a shadow.
- **A pass that loads a texture over a buffer and does not store it** renders through an image its
  texture keeps, the buffer's texels copied in first (`dont-care-stand-in`), so the buffer keeps its
  bytes as Metal's does: about 10 µs a pass on the Iris Xe (ANV). A pass that clears and stores
  `DontCare` costs no more, and one that stores renders in place.
- **The interlocked pass on the M4.** Every device splits a pass the pixel interlock would split,
  about a quarter faster than kept whole on the M3 Max (`DriverFact::InterlockedPassSplitNoSlower`);
  the M4 ran that pass about 8% faster kept whole.
- **A fetching draw in the pixel interlock waits for the draw before it.** Between two draws of a
  render pass instance where the later one reaches a color in the fetch's interlocked storage form
  (or an extended-range color's `half` shadow) that the earlier one wrote, the runtime records a
  barrier from the fragment shader's writes (`render_target::between_storage_draws`). Vulkan orders
  such draws by the interlock alone, but RADV on the Radeon 780M does not. On the Iris Xe the
  barrier costs about 15,000 GPU cycles a draw, 9 to 20 µs at the clocks measured: a 1920x1080
  pass of 27 such draws took 1.46 against 1.08 million cycles without it
  (`runtime_lane::tile_path_costs`, `wide blended in the function`). Not on a driver measured to
  order them (`DriverFact::InterlockOrdersDraws`: ANV on the Iris Xe;
  `runtime_lane::interlocked_draw_order`). The imageblock's storage and a depth stand-in keep
  theirs on every device.
- **A shadow over bytes another writer may write is merged back, not copied back.** Metal writes
  only the texels a command writes, and another writer writing bytes no command touches while a
  command buffer runs is valid under its contract. On an M3 Max under macOS 27.0.1, a CPU write to
  a texel no command touched, made 6 to 23 ms into a kernel's 58 to 68 ms run that wrote another
  texel of a texture over a `Shared` or `Managed` buffer, stayed every time; so did another
  queue's fill of such a texel of an untracked `Private` buffer, run inside a kernel's or a render
  pass's 61 ms run (5 times of 6; the sixth ran after it). Another queue's fill of a tracked
  buffer ran after the kernel or the pass ended, every time, as the runtime's tracker orders it.
  So a shadow over the bytes of a `Shared` or `Managed` buffer, host pages, the caller's
  `VkBuffer` or an untracked buffer is kept after its copy in, and the bytes the commands changed
  are merged back in place of the copy back (`shadow.rs`, `shadow_merge.rs`): two copies of the
  texture's image into a buffer and a merge, in place of one copy back, each cut to the rectangle
  the commands write in where they bound it. A tracked `Private` buffer's, which no other writer
  writes while the command buffer runs, is copied back. Where the copy back is a dispatch
  (`DriverFact::ShadowCopyBackDispatchFaster`: ANV, whose `vkCmdCopyImageToBuffer` is slow), so is
  the first copy, and the merge reads the texels the commands left from the image through an
  unsigned-integer view in place of the second. Where the copy in and the first copy are also one
  dispatch (`DriverFact::ShadowCopyInDispatchFaster`, measured faster on the UHD 770 of RPL-S
  alone), that dispatch reads the texels from the buffer once and stores them in the image and in
  the scratch: on the UHD 770 a 1920x1080 pass that loads took 1.43 ms of GPU time against
  1.64-1.67 ms, twelve 17.2 ms against 20.0 ms, a clearing pass 0.89 against 1.04 ms; on the Iris
  Xe of RPL-P twelve took a median 5% longer, so the Iris keeps the two. Where
  the merge reads the image and the copy in takes no first copy (the Iris), the first copy is a
  copy of the image's texels into a second image at the same coordinates (`vkCmdCopyImage`), which
  the merge reads beside it through the same view: the same bits, as a copy between formats of
  one compatibility class moves them. On the Iris Xe a 1920x1080 `BGRA8Unorm` image copied into
  another in 67-73 µs of GPU time, against 236-253 µs into a buffer and 270 µs for the dispatch
  it replaces; over a 1920x1080 surface of anonymous pages (`guest_frame_costs`, three rounds),
  twelve passes in as many command buffers took 7.2-8.2 ms of GPU time a frame against 9.2 ms,
  twelve in one 10.4-12.9 against 14.5-16.2 ms, and one 0.76-0.83 against 0.92-1.42 ms. That
  image takes the scratch buffer's place, 8.3 MB at 1920x1080, and is its queue's: a later keep of
  the queue of the same view and extent, whatever its texture, takes it once the encoder that
  took it ended, without waiting for the command buffers that used it, which the merge's closing
  barrier and the image's tracked uses order (`texture/kept_image.rs`); at most two that no open
  encoder holds stay for each view and extent, and none untaken over 256 takes. On the Iris Xe
  (`surface_texture_costs::surface_texture_cost_parts`, four shadowed textures a frame, 1,582,016
  texels), over anonymous pages, merged back, passes that draw took 5.0-5.4 ms a frame from commit
  to completion, and under scissors (790,448 texels) 3.6-3.9 ms; over tracked `Private` buffers,
  copied back, 3.1 ms and 2.2-2.4 ms.
- **A no-copy buffer's deallocator runs in its release.** On an M3 Max under macOS 27.0.1 an unused
  no-copy buffer's deallocator ran inside its final release, on the releasing thread, every time.
  So a release that frees the buffer's import while a deallocator waits for the free frees it
  there and then runs the deallocator, before returning. The free waits on the driver's locks a
  submission holds (ANV takes its device's lock in each execbuffer and in `vkFreeMemory`). On the
  Iris Xe (`submit_per_import::submit_per_import_release_cost`, 520 host imports live, four
  releases a frame) a free in the release took 115-128 µs (p50) on the releasing thread, 472-691
  µs a frame. A release whose keep-alive (a runtime extension) alone waits for the free goes to a
  handler thread instead, the release returning in 2.7-4.0 µs, 64-145 µs a frame: the consumer's
  guest gives its transfer buffers keep-alives.
- **Emulated tessellation's scan.** On a device without `tessellationShader` a patch draw first
  runs a one-workgroup scan that places each patch's triangles in Metal's order, patch after
  patch: about 6-60 ms of GPU time per 10,000-patch draw at factors 4 to 64 on the Radeon 780M and
  the RTX 3090, 0.6-5 ms on lavapipe (`emulated-tessellation-draws`). A parallel form is not built.

## Known gaps

Open differences from Metal that no note announces yet, and runtime costs still above the
driver's. Each is open work.

- **A sparse tile mapped after a write covered it while unmapped** reads zero for us and Metal's
  unwritten pattern (`0xFFFFFFFF`) for Metal; an unmap-and-map round trip loses the texels the same
  way. The runtime zeroes a sparse texture's unmapped texels and every newly bound block, and
  `BlitEncoder::scrub` has already marked the whole destination of the blit written, so nothing
  holds the tile unwritten; keeping the dropped part of a write unwritten in `FreshUnwritten` would
  reach it. No case compares it, because Metal means macOS means MoltenVK, which binds no sparse
  memory: confirming it needs a venue with `sparseBinding`, a sparse-binding queue family and
  `sparseResidencyImage2D` (the three `SparseNeed` checks), and the Metal half stays
  `probes/sparse_unmapped_blit.m`'s.

- **An extended-range format's sample through a constexpr clamp-to-zero sampler at a subnormal
  coordinate**: Metal's texture unit reads a subnormal normalized coordinate as the zero of its
  sign before the address mode decides, so a border axis reads texel 0 just outside the level.
  A sample of a 32-bit float texture flushes it as Metal does
  (`lower::sampler::position::subnormal_flush`, measured on an Apple M3 Max); the extended-range
  strategy (`lower::sampler::extended`), which fetches and addresses its taps itself, keeps the
  coordinate as given and reads the border there. Expected and unmeasured: the measurement is on
  `R32Float`, and Metal's flush on an extended-range format is not probed.

- **A two-plane YCbCr texture over plane 1 of a three-plane surface** (`y420`, `v0a8`): Metal
  reads plane 2's pairs as its chroma; the surface's facts (`IoSurfaceYCbCr`) name plane 1 alone,
  so the runtime reads chroma codes 0 there, as over plane 1 of a two-plane surface.

- **A mipmapped two-plane 3D texture's reads** (`RGB10A8_2P_XR10` and its sRGB twin): Metal's
  read and sample take bytes within the texture's first 4 KiB that vary from run to run, where a
  shader here reads its first texel, as of a one-level one (`lower::texture::emit::planes_volume`).

- **A draw under Apple's last provoking vertex on a device without it** (MoltenVK), whose pipeline
  state reads flat inputs or selects a slice or viewport (or a strip's whose fragment function reads
  barycentrics): it draws the list of its primitives led by their last vertices
  (`render/provoking.rs`), where Metal's mode costs nothing. A direct draw without indices draws the
  list the CPU writes into the upload arena (4 bytes an index); every other draw the list a run
  writes, one dispatch for a run of draws over one index buffer (`index_rewrite.rs`): the index
  gather pass's for a direct indexed draw of lines or triangles and for an indirect draw but an
  indexed strip's (an indirect run's after one plan dispatch), the index rewrite pass's for a direct
  indexed strip of up to a few thousand positions. On MoltenVK on the M3 Max that is 0.16, 0.24,
  0.65 and 0.71 us more frame time per direct, indexed, indexed strip and indirect draw of two
  triangles than at mode 0 (1.58, 1.72, 1.73 and 1.71 us; host_ab's `provoking`, 2000 draws, 3
  rounds), and 2.5 us per indexed indirect draw against mode 0's 4.1 (a run of the index bounds
  pass, which a gathered list does not need). A draw no run takes (a longer strip, an indexed indirect strip,
  indices an earlier draw of the encoder wrote) takes the index rewrite pass's three dispatches
  between barriers before its render pass: there 77 us per draw.

- **An indirect render command's draw of primitive type 9 that its execution's call cannot see**:
  one the CPU wrote after the call, or among a range the execution reads from a buffer. It runs
  nothing, counted as a record fault in the execution report (`RECORD`), where Metal's GPU hangs
  or faults the command buffer at it (the M3 Max, macOS 27.0.1: one written after the call, after
  the commit, or among a range read from a buffer failed as one the call sees, a direct one hung
  27.6 to 28.7 ms after `Scheduled`, an indexed one faulted 8.7 to 10.6 ms after it). One among a
  range given at the call, as the records stand then, fails the command buffer as the encoder's
  draw does (`render/executions.rs`); failing it from inside the host step that meets the record,
  its earlier encoders run, is not built.

- **A command buffer committed behind a running predicated loop on the loop's own queue waits for
  the loop to end**, as every batch of a queue waits for the ones before it, where Metal's GPU
  runs it meanwhile: a fault or hang in it ends the loop there on Metal (`recovery.rs`), and in
  the runtime it waits, for good behind a loop without bound. A fault or hang on another queue
  ends the loop as Metal's does.

- **A no-copy buffer imported by its pointer follows a remap of its pages** on a driver whose
  pointer imports follow their address (`DriverFact::PointerImportsFollowTheirAddress`: ANV on
  i915, RADV, lavapipe): pages mapped over the buffer's addresses after it was made (`mmap` with
  `MAP_FIXED`) are what its later command buffers read and write, where Metal's buffer keeps the
  pages it was made over. `HostMemory::new`'s contract forbids such a remap under a live buffer,
  and no notice tells of it. On RADV (Mesa 26.2.4, the Radeon 780M and the RX 7900 XTX) a file's
  pages mapped so over a live pointer import of anonymous pages lose the device: its next
  submission is rejected whatever it uses ("The CS has been rejected"): amdgpu's userptr takes
  only pages with no file behind them, and each submission lists every memory the device
  imported. No exact form exists once the pages are gone, and `Device::host_range_remapped` does
  not release such an import. An import of dma-bufs (`HostMemory::with_dma_buf`) keeps its pages
  on every driver, but is taken only where the device refuses the pointer.
- **A no-copy buffer's pointer import that stays live costs RADV's every kernel submission per
  page**: amdgpu gets every page of each live pointer import again in each command submission
  (the RX 7900 XTX under Linux 7.0.0: the scheduler thread's CPU per command buffer of a
  guest-shaped frame 18 us with none, 328 us with 275 imports of 512 KiB, 1144 us with 275 of
  2 MiB). An idle one is released where `DriverFact::PointerImportsFollowTheirAddress` holds (the
  780M and the RX 7900 XTX under Mesa 26.2.4), but one whose Metal address was handed out, or any
  on a device not measured with the fact, stays live. 275 kept dma-buf imports of 512 KiB each
  cost 54-56 us per command buffer there (no other size measured).
- **A detach or rebind between a command buffer's submissions.** A command buffer's later
  submission, behind an event wait, reaches a no-copy buffer detached or rebound after its first
  submission through an address (made resident, or unnamed) through the pages its first
  submission found, where Metal's reads the new pages or zeros; those pages stay imported, and
  their deallocator waits (announced), until the command buffer completes. Its commands that
  named the buffer are encoded again over the new pages, as Metal runs them, and fail with code 8
  where it is detached by then, as Metal's; those of an encoder that cannot be encoded again (a
  compute encoder whose unit of command predication held other encoders, a loop or work left out;
  work on the GPU alone) fail `EncodedUseRebound` (announced).
- **A runtime sampler's nearest sample of an 8-bit sRGB texture under native sampling.** On a
  device that samples the format, the sample keeps the device's sRGB decode (announced as
  `SrgbTexelDecode`), where the table's exact decode of the one texel it reads would cost little:
  the function does not know the bound sampler's filter, and a variant keyed on a nearest
  sampler's word is not built. Under `SamplingPolicy::Exact` a sample the linear emulation takes
  (not a cube's) reads the code through the slot's raw view there and decodes it exactly.
- **A fetched or resolved packed 16-bit attachment on a device that does not render the format**
  (`A1BGR5Unorm`, `ABGR4Unorm` on the RTX 5080). Its word view (`WordAttachment`) takes writes
  (blended and partly masked ones through the pixel interlock, the row above); a color input or
  resolve is `NotImplemented(Format)`, a blend of an output of fewer than four lanes
  `NotImplemented`, and a blend of the second source, or a blend or partial mask on a device
  without the interlock, is refused as the device lacking the format.
- **A blended, fetched or resolved one- or two-channel 8-bit sRGB attachment on a device that
  does not render the format** (`R8Unorm_sRGB`, `RG8Unorm_sRGB` on the RTX 5080 and ANV). Its
  word view takes writes (partly masked ones through the pixel interlock, or over the attachment's
  input attachment on a device that orders fetch through them, lavapipe); a blend other than one
  and zero is refused as the device lacking the format, and a color input or resolve is
  `NotImplemented(Format)`. A device that orders fetch only through the pixel interlock and renders
  any color format through its word view (these, the packed 16-bit ones, `RGB9E5Float`) answers
  the `ProgrammableBlending` family row `NotImplemented` and `isFramebufferReadSupported` no.
- **A half output, blend, partial mask, color input or resolve of an `RGB9E5Float` attachment on
  a device that does not render the format** (the RTX 5080). Its word view takes a `float`
  output's writes and clears; a `half` output, a color input and a resolve are
  `NotImplemented(Format)`, and a blend or a mask of some of its channels is refused as the
  device lacking the format.
- **ASTC textures on a device without ASTC.** Making a heap's or a sparse heap's such texture
  (`RuntimeGap::DecodedHeapTexture`) and a view of another decode (its HDR form) are refused. The
  heap size query answers one: its footprint is Metal's blocks, which no image of the device
  decides.
- **PVRTC textures on a device without PVRTC.** One without a decoded twin (a sparse heap's, or a
  heap's over the caller's host pages or laid out apart from its bytes) holds its blocks for copies
  alone: a command that reads it is refused.
- **ETC2 and EAC textures on a device without them.** A sparse heap's such texture
  (`RuntimeGap::DecodedHeapTexture`) and a view of another decode (an EAC 11-bit format's other
  signedness) are refused; one without a decoded twin (a heap's over the caller's host pages or
  laid out apart from its bytes) holds its blocks for copies alone, and a command that reads it is
  refused.
- **A level the device's arithmetic chooses.** Off MoltenVK under exact sampling, a fragment's
  implicit-level sample (its bias Metal's `B` within the device's limit), a sample with explicit
  gradients and a fragment's LOD query take the device's level of detail, where Metal's is not
  measured, and an explicit level Metal's rule is not handed on (an integer texture's, a constexpr
  sampler's with its own bias or an unmeasured clamp) takes the device's rounding. Where the level
  chooses what a sample reads and no sample-filter note records it, and for every LOD query, it is
  announced as `DeviceSampleLevel`. An extended-range texture's such sample takes Vulkan's level of
  detail computed in the shader on every device, with the same note.
- **An anisotropic cube sample on a `y` face through MoltenVK's vertex and fragment stages under
  macOS 27.** SPIRV-Cross's `spvGradientCube`, which MoltenVK 1.4.1 and 1.4.2 apply, writes a
  `+Y` or `-Y` face's gradients as `(d_x, d_z, d_x)`, so Metal loses the `z` gradient. The module hands such a
  face Metal's length of the two tangents on both (`CubeGradientRoute::AppleProjected`): isotropic
  samples, and every sample on the `x` and `z` faces, are Metal's, but an anisotropic one on a `y`
  face filters along the face's diagonal, announced as `CubeGradientFootprint`
  (`cube-gradient-footprint`). A kernel's samples are Metal's own. A form exact inside the face,
  sampling a `y` face through a 2D-array view of the cube, needs a second binding of each such cube
  from the binder and the runtime; it is not built.
- **An anisotropic cube sample out of the shader's reach through MoltenVK where its cube-gradient
  route is not measured** (`CubeGradientRoute::Unmeasured`). The shader takes Metal's level and taps
  wherever its emulation reaches (a cube or cube array of `float` or `half` texels, without
  comparison, through a constexpr sampler or a runtime sampler slot under its variant). A depth or
  comparison sample, one whose residency the function reads, an argument buffer's texture or
  sampler, or a slot of another class or swizzled storage takes Metal's isotropic footprint through
  a sampler made isotropic for that use, where Metal filters an ellipse, announced as
  `CubeGradientFootprint`. An argument buffer's sampler needs its degree in its table entry; it is
  not built.
- **A group operation in a branch arm whose speculation no probe covers.** Metal's backend runs a
  branch's short arms of plain computation in every lane of the branch's block; the module does the
  same for the arm shapes measured on Apple's GPU (`air::canon::speculate`). A branch of another
  shape whose arms hold a quad or simdgroup operation keeps the AIR's lanes, where Metal's may be
  the block's, announced as `UnmeasuredGroupSpeculation`.
- **A render pass's attachment read another way than a sampled slot binding.** Vulkan forbids
  reading an attachment in its own render pass instance (`VUID-vkCmdDraw-None-10795`); a sampled
  slot binding reads a copy or a snapshot (Costs above). Every other such read still reads the
  attachment in its instance, announced as `AttachmentReadInPlace`: a texture reached through the
  heap set (a handle, or an argument buffer's texture), a storage binding, a depth or stencil
  attachment a draw samples, a texture buffer or a buffer over the same buffer's bytes, a texture
  whose shadow is not served (`texture/usage_shadow.rs`), and an execution's command
  (`render/executions.rs`). In an instance that holds the attachment in the feedback loop layout,
  those reads name `VK_IMAGE_LAYOUT_GENERAL` in their descriptors, against the layout the instance
  holds.
- **A stored negative or NaN depth where the magnitude stand-in and the sign strip do not reach.**
  Metal's depth test compares the stored value's magnitude (a NaN above +inf), and a pass with depth
  writes stores every texel no fragment wrote without its sign; the runtime matches both through a
  magnitude stand-in or a sign strip (`texture/signed_depth.rs`, docs/API.md's
  `signed-depth-test`). Still open:
  - a multisampled or `Depth32Float_Stencil8` target's test, which compares a negative value or a
    NaN as it is, and a pass with depth writes on a device that makes no strip for it, which keeps
    the signs: announced (`signed-depth-test`); its resolve in such a pass resolves the samples
    before the strip, unmeasured on Metal;
  - a merge that keeps a stored NaN where a draw wrote exactly +inf over it, and a denormal at
    exactly 2^-126 where the device's test flushes one: announced (`signed-depth-test`,
    `depth-denormals`);
  - a pass that loads the depth through another image, and one encoded before the switching write
    in another command buffer uncommitted then: announced (`signed-depth-test`);
  - a NaN clear depth where the device writes no `0x7fc00000` into the depth before the pass:
    cleared to 1.0, announced (`signed-depth-test`);
  - a device that may clamp a depth outside `[0, 1]` (ANV): announced as `restricted-depth-upload`;
    a form that keeps the value in a 32-bit float colour image beside the stand-in is not built.
- **Compute private memory past a device's Function storage is refused.** A kernel, object or mesh
  function whose private memory (its `alloca`s, a call stack among them, and constants held as
  `Private` variables) passes what the device holds is refused (`Unsupported{PrivateMemory}`, or
  `Unsupported{Recursion}` for a call stack): past 16 KiB a thread on a device not measured, past
  its measured value on a measured one. So is a vertex or fragment function past a measured
  device's value, or past 192 KiB elsewhere (below that it runs with `FunctionStorageAssumed`, the
  device-gap row above). Vulkan states no limit on the driver's scratch, so the lane
  `runtime_lane::private_memory` measures it per device and `device::function_storage` keys it on
  the driver, vendor and device ID (a GPU that reports several read by the one measured, the Radeon
  780M of Hawk Point as Phoenix's): 96 KiB on the Radeon 780M and RX 7900 XTX, 32 KiB on the Iris
  Xe, the UHD 770 and the RTX 5080 Laptop GPU, 128 KiB on the RTX 3090, 192 KiB on the M3 Max and
  M4 under MoltenVK, 48 KiB on lavapipe; 16 KiB elsewhere. An exact form for compute, private memory
  in a runtime buffer with a slot per resident threadgroup, is not built.
- **An object grid past the device's limits where the device does not fold it.** With
  `VK_EXT_mesh_shader` but without device addresses, `shaderInt64`, `multiDrawIndirect` or
  `drawIndirectCount` (no committed profile), an indirect draw's object grid past the task stage's
  limits fails its command buffer (`CommandBufferError::MeshGridPastLimits`, drawing nothing) and
  a direct one is refused (`Unsupported{MeshShading}` naming the limit). The emulation (MoltenVK,
  ANV on the Iris Xe) refuses a direct object grid past the compute stage's limits or whose work
  buffer passes `maxStorageBufferRange` (`Unsupported{MeshShading}` naming the limit), and fails
  the command buffer of an indirect one past the compute limits with
  `CommandBufferError::MeshGridPastLimits` (ANV's y and z axes stop at 65,535; MoltenVK's limits
  pass every grid Metal runs).
- **An object threadgroup that sets a mesh grid Metal launches nothing for keeps its own writes.**
  On the M3 Max such a threadgroup's device writes (an atomic add before the set) do not land
  either, while the draw's other threadgroups run; the runtime runs the threadgroup and launches
  nothing from it.
- **A fragment function's imageblock write that its draw's tests may drop after it, where no
  form tests first.** Such a draw runs the tests before the write, early or in the shader against a
  depth stand-in. Refused as not implemented: a writing function under alpha to coverage; where
  the tests act, a draw under a visibility result mode, a depth format other than `Depth32Float`,
  a stencil attachment alone, and at more than one sample a stencil test on a device without
  multisampled storage images (MoltenVK) or a depth test where `FragCoord.z` is unmeasured (ANV);
  and on MoltenVK at more than one sample, whose early tests drop image writes
  (`DriverWorkaround::EarlyTestsDropMultisampledImageWrites`), a function declaring early tests
  that discards, returns depth, stencil or a sample mask, or writes other memory. A form testing
  at the function's entry against the stand-in is not built.
- **A vertex function that selects its slice, on a device without `shaderOutputLayer` and
  `geometryShader`.** Metal renders each primitive into the slice its first vertex selects. Vulkan
  stores `Layer` from a vertex function only with `shaderOutputLayer`, and the generated geometry
  stages that store it otherwise need `geometryShader`; a device with neither refuses the pipeline,
  `Unsupported(LayeredRendering)` needing either feature (among the committed profiles
  `vk12-floor`, `roadmap-2022` and `lunarg-desktop-baseline-2023`). So does a vertex function that
  selects its viewport there with `multiViewport`. One whose slice or viewport differs among a
  primitive's vertices needs the stages even with the outputs (refused needing `geometryShader`
  without them), except on MoltenVK, which hands the indices to Metal. An exact form, a
  layered pass drawn once per layer keeping the primitives whose first vertex selects it, is not
  built.
- **Commit to completion.** A command buffer takes longer from commit to completion through the
  runtime than a bare driver submission: on a quiet Radeon 780M (RADV) an empty command buffer took
  51.6 µs against a bare barrier's 45.1 µs. On MoltenVK most of the rest is the two GPU timestamps
  each timed command buffer writes: on an M4 a bare barrier took 14.7 µs alone and 167.8 µs with
  them. `DeviceOptions::gpu_times_from_host` (off by default, announced) answers GPU times from the
  host clock and spares them.
- **A texture made while a batch is submitted waits for that submission on a device without
  `descriptorBindingUpdateUnusedWhilePending`** (the `vk12-floor` profile), where its making
  registers a first move (a `Private` texture's): the registration takes the first-move order,
  which the batch holds from its take until its `vkQueueSubmit` returns. Metal's making waits for
  nothing. Elsewhere the batch holds the texture slots' writes instead
  (`WorkCounts::heap_slot_writes_deferred`, which stays 0 on such a device).
- **A shadow's copy back covers every texel it moves, where no merge serves it**: an image's
  shadow (a view of a layout the device's views do not reinterpret), a turned texture's and a
  two-plane one's. After a pass that draws, that is every texel inside the draws' scissors, where
  Metal's store writes back only the 32x32 tiles a draw's primitives touch: another writer's write
  during the pass to an undrawn texel in an untouched tile there can be undone where Metal keeps
  it. A shadow over a buffer's bytes another writer may write is merged back instead, only the
  words the commands changed (`shadow.rs`), and no write reaches the bytes between rows; a tracked
  `Private` buffer's, whose bytes no other writer writes during the pass, is copied back.
- **A merge through a copy rewrites the texture's bytes it did not change**: where the texture's
  first byte is not a whole word of its buffer, its bytes go through a staging buffer, or its last
  word runs past the buffer, the merge copies the bytes from the texture's first to its last into
  its scratch, merges there, and copies them back (`shadow_merge.rs`). Another writer's write to
  them between the two copies is undone.
- **A linear image over a `Shared` or `Managed` buffer's packed rows moves out of
  `PREINITIALIZED` at its first use**, a transition Vulkan lets read and write every byte bound to
  it: a CPU write to those bytes during it could be undone, where Metal keeps it (not observed).
- **A cold pipeline making on NVIDIA grows with a module's sampling sites.** NVIDIA's compiler takes
  about 1 s for macOS 26's largest CoreAnimation fragment (210,033 words) on the RTX 5080 Laptop
  GPU, once per cold cache; states that share the module then take 13 to 16 ms, and a warm cache
  2 ms. The module inlines a few sampling helpers five or six times each; keeping one copy of each
  would cut it. It is not built.
- **An intrinsic declaration from macOS 12's and 13's Metal compiler** whose type fits none of the
  forms `intrinsics::classify` derives from its name is refused with
  `InvalidInput(Intrinsic(SignatureMismatch))` (`intrinsics::bind`). AIR 2.5 (Metal compiler
  31001.723) declares the indirect command setters `air.set_kernel_buffer_compute_command` and
  `air.set_vertex_buffer_render_command` without the attribute stride, which decode takes as MSL's
  plain setter; whether the refused kernels are those setters, and so translate, is unverified.
- **A direct call whose arguments do not line up with its callee's parameters** is refused with
  `InvalidInput(Canon(CallTypeMismatch))` (`canon::post_check`): LLVM leaves such a call undefined,
  and what Metal's back end makes of it is not documented, so no exact form is known.
- **A loop that sums its counter fails to make through MoltenVK.** Apple's MSL front end rewrites a
  loop whose running sum adds its 32-bit counter (`1 + 2 + ... + n`) into a closed form in 33-bit
  integers, which the M3 Max's back end cannot legalize where it lands in the kernel's constant
  program: MoltenVK, whose SPIRV-Cross writes such a loop in that shape, fails the pipeline's making
  (`XPC_ERROR_CONNECTION_INTERRUPTED`), and the runtime returns the failure
  (`CommandBufferError::PipelineCreation`, or `Error::Vulkan` at an indirect command's commit).
  Metal runs the AIR, which gets no such rewrite (case `authored-four-iteration-loop`). No form of
  the loop is known that avoids the rewrite for every such loop.
- **An anisotropic sampler without a mip filter whose min and mag filters differ, through Metal's
  own sampler (MoltenVK).** Vulkan has no sampler without mipmapping, so the runtime hands MoltenVK
  a nearest mip filter with a level clamp: `[0, 0]` where the two filters agree, which is Metal's
  `NotMipmapped` bit for bit, and `[0, 0.25]` where they differ, which chooses Metal's filter, but
  at `maxAnisotropy` 2 to 16 no clamp reads Metal's `NotMipmapped` answer. Announced as
  `NotMipmappedAnisotropicSampler` for a sampler state and `NotMipmappedAnisotropicFilter` for a
  constexpr sampler.
- **Dynamic texel offsets past 512 arms in one function, on a device without the `Offset`
  operand.** Without `maintenance8` (a sample) or `shaderImageGatherExtended` (a gather) a dynamic
  texel offset is a switch over the values Metal reads, each arm the access with that value as
  `ConstOffset` (16 arms a dynamic lane), exactly. A function whose accesses would take more than
  512 arms together is refused (`TexelOffset` or `GatherExtended`): a driver's making grows faster
  than the arms (on ANV 512 arms make in 0.63 s, 1536 in 5.9 s). Applying the offset in the shader
  is exact only where the shader filters, so a filtered sample past the bound has no exact form yet.
- **A fetched attachment's blend through the pixel interlock (RADV, ANV, NVIDIA, MoltenVK).** No
  Vulkan blend applies to the storage form's write, so the function computes the blend: Metal's
  for an extended-range or 16-bit float attachment, and for an 8-bit unorm one (`R8Unorm`,
  `RG8Unorm`, `RGBA8Unorm`, `BGRA8Unorm`) of a four-lane output under any factor, operation,
  write mask and blend color, and for an 8-bit sRGB one the device renders (`R8Unorm_sRGB`,
  `RG8Unorm_sRGB`, `RGBA8Unorm_sRGB`, `BGRA8Unorm_sRGB`) the same blend over the decoded texel,
  encoded by Metal's table; the generic `float` blend, announced (`UnmeasuredFetchBlend`, the row
  above), for every other format whose channels read as floats and for a blend of the second
  source (an sRGB one's in the 8-bit unorm blend's rounded forms, as are a one- or two-channel
  sRGB one's fast-math lerps and an output of fewer than four lanes). A source-over survey on the
  M3 fits the 8-bit model to the 10-bit unorm and 8-bit snorm formats (with their clamps), and
  shows the 16-bit norms and `RGBA32Float` blending in `float` even for a `half4` output; their
  factor sets are not measured, so their exact forms are not built.
- **A blended signed normalized attachment on lavapipe.** lavapipe blends no snorm format
  (`R8Snorm`, `RG8Snorm`, `RGBA8Snorm`, `R16Snorm`, `RG16Snorm`, `RGBA16Snorm`: no
  `COLOR_ATTACHMENT_BLEND`), so a pipeline that blends one is refused as the device lacking the
  format. The function's own blend over the attachment's input attachment, as an extended-range
  attachment takes there, is not built: Metal's snorm blend is measured for source-over alone
  (the row above).
- **Format-less storage reads on a device without `shaderStorageImageReadWithoutFormat` (ANV).**
  There a read without a format goes by the format's own `STORAGE_READ_WITHOUT_FORMAT` bit
  (Vulkan 1.3 or `VK_KHR_format_feature_flags2`), which the 16-bit packed formats,
  `B8G8R8A8_UNORM`, `A2R10G10B10_UNORM_PACK32`, `A2B10G10R10_UNORM_PACK32` and
  `B10G11R11_UFLOAT_PACK32` lack in the committed ANV profiles: a fetch through the pixel interlock
  of a 16-bit packed color attachment (`B5G6R5Unorm`, `A1BGR5Unorm`, `ABGR4Unorm`, `BGR5A1Unorm`),
  which no SPIR-V format names, is refused (`FramebufferFetch`). A read of an argument buffer's
  writable texture is refused there whatever its format (`ArgumentBufferTextures`): the heap
  views its members for the accesses the device's features grant, for every pipeline alike, and a
  check of each member's bit against the pipeline's heap reads is not built. A fetch through a
  16-bit unsigned view that packs the channels in the shader, and heap storage classes typed per
  format, are not built either.
- **The formats from 630 to 645 under macOS 27.0.1.** The macOS 27.0.1 level answers as an Apple M3
  Max, which describes none of them (`BGRA10Unorm`, its packed, sRGB, PQ and HLG forms,
  `R10UintX6`, `RG10UintX12` and the rest): a texture of one is stopped and the linear alignment
  query refused, as on the M3 Max. An Apple M4 under 27.0.1 describes and makes them; no level the
  runtime has does.
- **Tessellation's triangle order.** On a device with `tessellationShader` but MoltenVK on an
  Apple GPU, the device's tessellator orders a patch's triangles, picks their first corners and
  joins its rings itself, announced as `TessellationTriangleOrder`
  (`tessellation-triangle-order`): every patch differs on the RTX 3090, 55-95% on RADV and
  lavapipe.
- **lavapipe aborts on the owned fixture `mesh_order`'s mesh functions.** Compiling
  `mesh_order_mesh` or `mesh_order_solo` ends the process inside lavapipe's compiler ("LLVM ERROR:
  Cannot emit physreg copy instruction"); the validation case `mesh_order_mesh-payload-mesh-draw`
  runs on a GPU only. The construct behind the abort is not isolated yet.
- **A record the GPU wrote before its ID changed holder, in a buffer the GPU wrote again since.**
  A record the GPU wrote names the state that held its ID as the GPU last could write the
  buffer's records (`released_programs.rs`). Where a command buffer wrote the record, another
  state took its ID, and a later command buffer that may write the buffer completed without
  rewriting it, the record runs and answers the later holder, where Metal's runs the state its
  write resolved. No note announces it. The exact form needs the GPU's write to carry its
  resolution: a word of the record the translated setter fills from a per-dispatch stamp of the
  device's IDs, which the host step, the getter and the GPU-alone preprocess resolve.
- **A private logic operation other than copy and no-op without Vulkan's form.** On a device
  without `logicOp` (MoltenVK), and on any device where an attachment is sRGB, a float attachment
  blends, a format's operation was not measured or the function writes an attachment itself, the
  pipeline is refused (`PrivateRenderField::LogicOperation`), where the function could apply
  Metal's operation to the codes it reads and writes in the pixel interlock.

- **A reset compute pipeline descriptor keeps its thread shape on Metal and not here.**
  `MTLComputePipelineDescriptor.reset` restores the label, the buffer mutability,
  `supportIndirectCommandBuffers`, `supportAddingBinaryFunctions` and `maxCallStackDepth`, and
  leaves `threadGroupSizeIsMultipleOfThreadExecutionWidth`, `maxTotalThreadsPerThreadgroup` and
  `requiredThreadsPerThreadgroup` as the program set them (measured on an Apple M3 Max, macOS
  27.0.1; the tile and mesh descriptors' resets restore every field). The runtime's counterpart is
  `ComputePipelineDescriptor::new`, a fresh value that resets all nine, so a consumer that resets
  loses the thread shape where Metal keeps it. Nothing announces it.
  `compute-descriptor-reset` holds it.

- **Blending into an integer-format attachment is not compared, because Metal ends the process.**
  `-[MTLRenderPipelineDescriptorInternal validateWithDevice:]` fails an assertion and the process
  ends on SIGABRT — `Blending is enabled for render target 0; however, the pixelformat
  MTLPixelFormatR32Uint for this render target is not blendable` — measured on an Apple M3 Max,
  macOS 27.0.1 for `R32Uint`, `RGBA8Uint` and `RGBA8Sint`, with blending **off** on the same
  format and blending **on** over `RGBA8Unorm` creating normally as the two controls. So Metal has
  no answer here to be conformant with and no case may ask for one. What the translator does with
  the neighbouring shape is compared: alpha-to-coverage over an integer attachment 0 is a typed
  `InvalidDescriptor` (`reads_alpha_to_coverage`), which is Metal's own pipeline error.

- **`rasterSampleCount` disagreeing with the attachment's sample count is not compared.** Plain
  Metal creates the pipeline and draws something that looks undefined — a 1-sample pipeline into a
  4-sample attachment left samples 1 to 3 at neither the clear nor the fragment — while Metal's
  API validation layer ends the process at `setRenderPipelineState:` (`the texture sample count
  (4) does not match the renderPipelineState colorSampleCount (1)`), both measured on an Apple M3
  Max, macOS 27.0.1. A plain-Metal answer that looks undefined is not one to pin as a conformance
  expectation, and a case of it would end a recording run.

- **A 16-bit float texel's NaN through RADV's sampler**: Metal answers the texel's own bits,
  `0x7e00`, where a nearest sample answers `0xfe00` here (`half-texel-nans-r16f`, `-rg16f`,
  `-rgba16f`, 515, 1025 and 2045 elements; the `read` and `read-through-sample` sites of the same
  cases are **exact** on both RADV chips, where all three differ on MoltenVK). RADV's sampler hands
  back its own negative quiet NaN, `0xFFC00000`, where Apple's hands back `0x7FC00000`, so the
  device's decode destroys the bits before any keep of ours. Announced (`device-nan-encoding`); the
  exact form, not built, is the read side of the `StorageImageClass::HalfBits` write path — read the
  image through an unsigned-integer alias so no device decode sees the texel. The fetch half of this
  difference is the first gap above, which is a different cause and needs its own fix.

- **A multisampled `Depth32Float` texture's fill of a denormal on RADV**: Metal's private fill
  stores the pattern's bytes, `0x00000001`, where this stores `0x00000000` (`sample-fills-d32-2`,
  `-d32-4`, `-d32s8-4`; both RADV chips, exact on MoltenVK;
  `runtime_lane::sample_fills::multisample_fills_write_metal_s_samples` takes the word where the
  device announces the loss and compares every other). **The device's, and announced**: RADV gives a
  denormal that a GPU write puts into a `D32_SFLOAT` depth image back as a zero and keeps only a CPU
  store's, which is why it carries no `DriverFact::DepthImageKeepsDenormals`
  (`device/driver_facts.rs`), and a multisampled image takes no CPU store. The runtime announces
  `depth-denormals` and the case declares it at its notices site while still comparing every depth
  word. Each of these textures is a render target, which is the half still open:
  `MultisampleDepthTwinPolicy::Exact` holds a multisampled `Depth32Float` texture's texels in a
  colour image, as a read-only depth texture's already are, only where no pass renders it. The exact
  form for a render target, not built: the texels of record in the multisampled `R32_SFLOAT` image
  for the texture's life, with a `D32_SFLOAT` stand-in each pass renders into
  (`attachment_stand_in::kept_backing`, as `Cause::Magnitude` keeps one for a signed depth), copied
  in where Metal loads the depth and back where it stores it, so that every fill and copy writes the
  colour image exactly and only a pass's own texels pass through the depth image.

- **A blend's result on RADV, by 1 to 64 codes** (`half-output-blends-*`, `blend-sweep-operations`,
  `blend-interaction-format-unorm8` and `-srgb8`, `argb8-targets-*`, `a2c-on-*`; 24 cases). The
  one-code rows are what a driver's own rounding of a normalized blend produces, which Vulkan does
  not pin as closely as Metal does, and those are a candidate for an announced precision note rather
  than a fix. The largest, 64 codes at `blended/sas-1msa`, is **too large for rounding** and is a
  separate open question.

- **One line texel at a diamond vertex on the RTX 5080** (`primitive-type-restart-indices`): all
  four `Line` and `LineStrip` cells differ from the M3 Max's recording by two texels on an RTX 5080
  Laptop GPU (615.71.09), and by none on llvmpipe (LLVM 23.1.1, 256 bits). The line of vertices 0
  and 1 runs exactly through (3.5, 1.0), a vertex of the diamonds of the two pixels either side of
  it: Metal draws the lower one and this driver draws the upper one. Vulkan's Bresenham diamond is
  open (`|p - x| + |q - y| < 0.5`), so a line that only touches a vertex exits no diamond and
  neither pixel is required; both choices conform and there is no Vulkan control over the tie, so
  this is the driver's and not ours. RADV and ANV unrun.

- **A heap multisample texture's sample moves on a GPU of 20 GiB or less**: a command queue keeps
  the images such moves go through up to a sixty-fourth of the device's largest device-local heap
  (`gpu_pools::device_share`, `texture/copy_images.rs`). A 4096 x 2160 `Depth32Float_Stencil8`
  texture of 4 samples -- the consumer's surface -- needs 295 MiB of them per blit, over the
  255 MiB a 16 GiB GPU keeps, so every copy makes all 16 again (measured on an RTX 5080 Laptop
  GPU, 615.71.09; `heap_sample_moves` prints each copy's images made and lent, and its 256 x 256
  shape, 2 MiB of images, makes none after the first). The images are correct and bounded; the
  cost is the making. Raising the share, or holding the bound to at least one copy's images,
  would reach it.

- **A folded object grid's cost per Metal threadgroup**: a device that runs object functions
  through `VK_EXT_mesh_shader` takes one task workgroup per Metal object threadgroup
  (`bind/mesh_arguments.rs`, the folded object form), and an RTX 5080 Laptop GPU (615.71.09) takes
  2.4 to 2.7 microseconds over each: 4,587,520 of `mesh_object_grid`'s threadgroups in 12.0 s and
  4,194,307 in 11.1 s. Metal's own GPU ran the 33.5 million-threadgroup grids of the same fixture
  inside its 5 s watchdog, about sixteen times faster per threadgroup, so those grids are past any
  bounded wait here and `mesh_draws::object_grids_run_as_metal_runs_them` now measures the device's
  cost and skips them, printing both. Nothing is wrong with the words the smaller grids leave; the
  cost is the dispatch.

- **Reflection after a compute descriptor reset on RADV**: `max_total_threads_per_threadgroup`
  answers 0 where Metal answers 96, `required_threads_per_threadgroup` `0x0x0` where Metal answers
  `8x6x2`, and `threadgroup_size_is_multiple_of_execution_width` false where Metal answers true
  (`compute-descriptor-reset`). Both RADV chips, the RTX 5080 and llvmpipe, exact on MoltenVK.

These were found by replaying the conformance suite on a driver that is not Metal: RADV first, then
the whole 1,099-case set on an RTX 5080 Laptop GPU (615.71.09) and on llvmpipe (Mesa 26.2.4), both
at a tip before the narrow-float widening above was canonicalized. **ANV is unaudited** for every one
of them, and on a card without unified memory the comparison is not yet readable at all: the runtime
announces `shared-in-system-memory` for every `Shared` buffer and no recording taken on Metal could
declare it, which counts 699 of the 5080's cases divergent with 655 of them differing in nothing
else.
