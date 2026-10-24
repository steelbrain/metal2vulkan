//! What a driver is measured to do where the specifications leave the answer to it, keyed on
//! the driver's identity and release: `VkPhysicalDeviceDriverProperties`'s `driverID`,
//! `VkPhysicalDeviceProperties`'s `vendorID` where the measurement is of one GPU vendor's, and
//! `driverVersion`. One table type serves both kinds of fact: the description's deviations, which
//! translation reads and the compile profile holds
//! ([`DriverDeviation`](crate::error::DriverDeviation)), and the runtime's driver facts, which no
//! translation reads ([`DriverFact`], in this table as `MEASURED`): a description names its
//! driver's through [`DeviceDescription::driver_facts`], which the runtime puts in its device's
//! properties. None of those is in the compile profile, a cache key or a digest.
//!
//! Each row names a driver and the first and last releases measured with the fact (the
//! coordinator's ruling of 2026-10-06). For the description's deviations and translation's
//! defects (`DEVIATIONS`, `DEFECTS`, read through `among`):
//! - **Measured**: a release from the first through the last. Its fact is taken.
//! - **Beyond**: a release newer than the last. Its fact is taken too, but recorded as a
//!   near-equivalent naming the fact and the last release measured, announced once, so it is
//!   never a silent guess. The lane that measured the fact runs at each master gate on the venues
//!   with that driver, so an upgrade of a venue's driver that changes it fails there.
//! - **Unmeasured**: a release older than the first, or another driver or vendor. The fact is not
//!   taken: the reader takes its conservative route, the exact path where one exists.
//!
//! The runtime's driver facts follow one rule instead (the owner's "correct by default"): **a
//! device whose identity was not measured is unmeasured**. An identity is the exact quadruple
//! `driverID`, `vendorID`, `deviceID` and `driverVersion`; `MEASURED_DEVICES` lists the ones the
//! lanes measure (a driver at one release on one GPU: the device class is the GPU's `deviceID`,
//! never a host's name), and [`DeviceDescription::measured_identity`] answers whether a description
//! is one. A GPU that reports more than one `deviceID` is one device class under each: every table
//! here names it by the ID it was measured under, and a description is read by its
//! [`DeviceDescription::canonical_device_id`]. A `deviceID` that names more than one GPU
//! (MoltenVK's on Apple's GPUs is the macOS release and the GPU family) is one device class for
//! every release measured on each of its GPUs; a release measured on one of them alone names that
//! GPU by its `deviceName` (`NAMED_RELEASES`), and is no measured identity of the others.
//! - **On a measured identity** a fact is taken where a row of its driver names the identity's
//!   release (from the first through the last) and, where it names them, its vendor and GPU: as
//!   measured.
//! - **On any other identity** (another release of a measured driver, newer or older, another
//!   GPU of it, or another driver) no answer fact is taken: the runtime takes the exact path that
//!   needs none. A defect fact ([`DriverFact::assumed_where_unmeasured`], the driver doing
//!   otherwise than Metal where the runtime has a workaround) is taken wherever a row of the
//!   same driver names it, at any release and on any GPU: the workaround, never the path a
//!   measurement has not cleared. The runtime says so once as such a device is made, in an
//!   informational message (`metal2vulkan-unmeasured-driver`; no degrade notice, as the paths it
//!   takes are exact).

use crate::device::description::APPLE_VENDOR_ID;
use crate::device::profile::DriverWorkaround;
use crate::device::DeviceDescription;
use crate::error::DriverDeviation;
use crate::vk::{DriverId, DynamicState, Format, FormatFeatureFlags2, PrimitiveTopology};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

/// One driver measured to have the fact `F`, over the releases measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Measured<F> {
    pub(crate) fact: F,
    pub(crate) driver: DriverId,
    /// The GPU vendor (`vendorID`) the fact is measured on, or `None` where it is the driver's
    /// on every GPU it runs.
    pub(crate) vendor: Option<u32>,
    /// The GPU (`deviceID`, as the driver reports it) the fact is measured on, or `None` where
    /// it is the driver's on every GPU of the vendor: a fact one GPU of a vendor answers and
    /// another does not (the graphics cube route).
    pub(crate) device: Option<u32>,
    /// The first release (its `driverVersion`) measured with the fact.
    pub(crate) first: u32,
    /// The last release measured with the fact.
    pub(crate) last: u32,
}

/// Where a driver's release stands to a fact measured of it (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum Standing {
    /// A release from the first measured through the last.
    Measured,
    /// A release newer than the last measured, whose `driverVersion` is `last`: the fact is
    /// taken and recorded as a near-equivalent.
    Beyond {
        /// The last release measured with the fact.
        last: u32,
    },
}

/// The facts of `table` that the driver `driver` on GPU `device` of `vendor`, named `name`
/// (`VkPhysicalDeviceProperties::deviceName`), has at release `version`, each with its standing;
/// an unmeasured fact is absent. A `device` of `None` is a GPU no row names
/// (`DeviceDescription::measured_gpu`): only the rows of every GPU hold for it. Where `version` of
/// the identity was measured on another GPU of it alone ([`NAMED_RELEASES`]), no row of its GPU
/// that covers `version` holds, so the rows of earlier releases stand beyond their last; a `name`
/// of `None` is put out by no named release.
pub(crate) fn among<F: Copy + Ord>(
    table: &[Measured<F>],
    driver: Option<DriverId>,
    (vendor, device): (u32, Option<u32>),
    version: u32,
    name: Option<&str>,
) -> BTreeMap<F, Standing> {
    let out = name.is_some_and(|name| named_out(driver, (vendor, device), version, name));
    let mut facts = BTreeMap::new();
    for row in table {
        let ours = Some(row.driver) == driver
            && row.vendor.is_none_or(|v| v == vendor)
            && row.device.is_none_or(|d| Some(d) == device);
        if !ours || version < row.first {
            continue;
        }
        if out && row.device.is_some() && version <= row.last {
            continue;
        }
        let standing = match version <= row.last {
            true => Standing::Measured,
            false => Standing::Beyond { last: row.last },
        };
        // Two rows of one fact (a driver measured on two vendors, or at two releases) keep the
        // better standing: measured, else beyond the later last release measured.
        let kept = facts.entry(row.fact).or_insert(standing);
        *kept = match (*kept, standing) {
            (Standing::Beyond { last: a }, Standing::Beyond { last: b }) => {
                Standing::Beyond { last: a.max(b) }
            }
            _ => Standing::Measured,
        };
    }
    facts
}

/// Every measured deviation translation relies on, by driver. Each but the two color-NaN stores
/// (below) is MoltenVK's on Apple's GPUs, at 1.4.1 (`driverVersion` 10401, Homebrew's build of
/// SPIRV-Cross `adec7acb`), measured on the Apple M3 Max and the Apple M4 (macOS 26.7 and 27.0.1).
/// The gate lane `runtime_lane::driver_deviations` checks lockstep, the float functions and both
/// cube routes on each, the sampler's lanes the sampler; the canonical NaN rests on the
/// measurements below:
/// - `SimdgroupLockstep`: kernels `7ab5ddfd` and `4b74b854` answer on the M3 Max what only lockstep
///   explains, and MoltenVK runs `4b74b854` without a barrier and matches; the owned fixture
///   `rtl_lockstep`'s two kernels answer Metal's lockstep order through the device's lockstep on
///   both chips.
/// - `MetalFloatFunctions`: every word of the precise math-library probe Metal's
///   through MoltenVK; on both chips, every word of a function the device's instruction answers, in
///   both math modes.
/// - `MetalCanonicalNans`: every class-A NaN Metal's canonical one through MoltenVK.
/// - `MetalSampler`: Metal's bytes on the 12 linear `half` cases through MoltenVK.
/// - `CubeGradientRoute`, a kernel's: our probe (cases `a54af728` and `315ca8b0` match
///   through MoltenVK on the M3 Max), and the same cases against the M4's own Metal rows on the M4;
///   under macOS 27.0.1 the M3 Max's MoltenVK rows of all eight committed cube-gradient cases
///   match its Metal rows (2026-10-14's re-record).
/// - `GraphicsCubeGradientRoute`, a vertex or fragment function's: measured on the M3 Max's GPU
///   under macOS 26.7 alone ([`APPLE_M3_MAX`]): SPIRV-Cross's `spvGradientCube` read in its
///   source, its transposed form measured in a Metal kernel, and the owned fixture
///   `rtl_cube_grad_fragment` matching Metal's kernel rows. On the M4 the same form differs on 96
///   and 48 of 144 samples.
/// - `GraphicsCubeGradientTangents`, a vertex or fragment function's under macOS 27, the identity
///   [`APPLE_M4`] (the M3 Max's and the M4's): the M3 Max under macOS 27.0.1 reads the face's two
///   tangents of each gradient, its normal component dropped, so SPIRV-Cross's `spvGradientCube`
///   (copied verbatim into a Metal kernel, 2026-10-15) of a gradient whose normal component is
///   zero samples as Metal's own sample of the gradient on every lane of an `x` or `z` face, under
///   every sampler (isotropic, anisotropic of degree 2 to 16, nearest and linear mip filters; 4,096
///   such lanes of 6,144 a sampler), and on a `y` face differs on 922 to 934 of 1,024 a face (the
///   fixup writes the `x` tangent on both). Handed a `y` face's `(64/91 L, 0, 0)` instead (`L`
///   Metal's length of `(g_x, g_z)`, `lower::sampler`'s `Projected` form), the fixup samples as
///   Metal on all 6,144 lanes, every face, under each of three isotropic samplers (trilinear,
///   nearest mip filter, no mips), and on a `y` face under an anisotropic one still differs on 922
///   to 932 of 1,024 a face. The gate lane checks the `x` and `z` faces' samples of
///   `rtl_cube_grad_fragment` against Metal's kernel rows. The M4's reading under macOS 27
///   (its earlier probes: both tangents counted, not the normal) is the M3 Max's.
/// - `MetalColorWrites`: `runtime_lane::color_writes` through MoltenVK stored Metal's code for
///   every input of probes (every code's exact value and every tie with two `f32` steps either
///   side, of the 8- and 16-bit unorm and snorm, 10:10:10:2, 8-bit sRGB and half-float formats),
///   plain and blended with factors one and zero, where every other venue's driver differs: on the
///   Apple M4 and the M3 Max. One row per GPU, as measured.
///
/// The color-NaN stores, measured by `runtime_lane::color_writes` (at 0c344d06f,
/// 2026-10-06): every NaN class of `float` outputs (the rounding writes `0x7fc00000`) to
/// `R16Float`, `RG16Float` and `RGBA16Float`, and every `half` NaN of a `half` output to
/// `RGBA16Float`, plain and blended with factors one and zero, against Metal's `0x7e00` and the
/// `half`'s own bits:
/// - `MetalFloatColorNans` and `MetalHalfColorNans`: NVIDIA's (`NVIDIA_PROPRIETARY`) 615.71.09
///   on the GeForce RTX 3090 (`deviceID` `0x2204`) and the RTX 5080 Laptop GPU (`0x2c19`): every
///   channel Metal's. One row per GPU, as measured.
/// - `MetalFloatColorNans` alone: lavapipe (`MESA_LLVMPIPE`) from Mesa 26.0.8 through 26.2.4,
///   every `float` NaN Metal's and a `half` NaN another: Mesa 26.0.8 on LLVM 21.1.8, x86_64 (the
///   linux-gate container; signalling `half` NaNs quieted) and arm64 (on an M4 mini; every `half`
///   NaN stored as `0x7e00`), and Mesa 26.2.4 on LLVM 23.1.1, x86_64 (Arch's, on an i9-13900, at
///   58957da0; quieted).
///
/// Measured to store another NaN for both outputs, so not listed: RADV, Mesa 26.2.4 on the
/// Radeon 780M, and ANV, Mesa 26.2.4 on the Iris Xe (RPL-P), `0xfe00`. MoltenVK 1.4.1 stores a
/// `half` NaN as `0x7e00` on the M3 Max and the M4, and its `float` NaN as Metal does
/// (`MetalColorWrites`).
///
/// `HalfColorOutputBits`, measured by `runtime_lane::color_writes` through the 16-bit output (at
/// 19411f193 and our probe at 50bc61f5e, 2026-10-07, the route then taken wherever the profile
/// declares 16-bit outputs): every `half` NaN of a `half` output to `RGBA16Float`, plain and
/// blended with factors one and zero, stored as its own bits, and every other channel Metal's,
/// through MoltenVK 1.4.1 on the M3 Max and the M4. One row per GPU, as measured. Measured to store
/// another NaN from a 16-bit output too, so not listed: RADV, Mesa 26.2.4 on the Radeon 780M, and
/// ANV, Mesa 26.2.4 on the Iris Xe (RPL-P), `0xfe00` for every NaN of a `float` and a `half`
/// output. lavapipe and NVIDIA's captures declare no 16-bit outputs.
///
/// `MetalHalfFma`: MoltenVK 1.4.1's `GLSL.std.450 Fma` on 16-bit floats, which SPIRV-Cross writes
/// as Metal's `fma(half, half, half)`, the operation a Metal `half` fused multiply-add compiles to.
/// The gate lane
/// `runtime_lane::driver_deviations::half_fused_multiply_adds_the_device_answers_are_metals` runs
/// the two committed `half` `fma` cases against each chip's Metal rows: 0 of 1,024 `half` words
/// differ in each case, on the M3 Max and the M4. One row per GPU, as measured.
///
/// `MetalFloatDenormals`: the device's `float` `+ - *`, comparisons, `OpFmaKHR`, `Dot`, `Ldexp`,
/// `Sqrt`, `InverseSqrt`, `Trunc` and `RoundEven` read an `f32` denormal operand as the zero of its
/// sign and flush a denormal result, as Metal does (`Trunc` and `RoundEven`), with no
/// float-controls execution mode declared (Vulkan's `DenormFlushToZero` would guarantee the results
/// alone). The committed rows of the denormal probes (`nan_denormals`, `nan_denormals_fast`: 256
/// lanes of operand triples rich in denormals of both signs and the smallest normals) differ from
/// Metal's on none of those operations' lanes on MoltenVK 1.4.1 on the M4, RADV (Mesa 26.2.4 on the
/// Radeon 780M), ANV (Mesa 26.2.4 on the UHD 770 of RPL-S) and NVIDIA's 615.71.09 on the RTX 5080
/// Laptop GPU; the gate lanes `float_denormals_*` measure the M3 Max, the Iris Xe of RPL-P and
/// the RTX 3090 likewise. lavapipe keeps them (Mesa 26.0.8 on LLVM 21.1.8, x86_64 and arm64:
/// 1,249 and 1,032 words of the two probes differ), so it is not listed. NVIDIA's `Floor` and
/// `Ceil` keep a denormal, which the deviation does not cover: every profile reads their operand
/// itself (`lower::arith::denormal`). One row per GPU, as measured.
///
/// `MetalHalfQuantize`: the device's `OpQuantizeToF16` rounds an `f32` below 65520 in magnitude to
/// nearest even, as Metal's conversion to `half` does, so 65504 to 65520 (`0x477FEFFF` among them)
/// is 65504, not infinity. The committed rows of the NaN model's narrow, convert and half cases
/// (`nan_narrow`, `nan_convert`, `nan_half` and their fast-math twins) match Metal's on every word
/// whose answer is the largest `half` on MoltenVK 1.4.1 on the M3 Max and the M4, RADV (Mesa 26.2.4
/// on the Radeon 780M) and ANV (Mesa 26.2.4 on the UHD 770 of RPL-S); the gate lane
/// `half_quantize_*` and the owned fixture `dfl_half_edge` measure the Iris Xe of RPL-P likewise.
/// NVIDIA's 615.71.09 rounds `0x477FEFFF` to infinity, so it is not listed; lavapipe's rows match
/// at Mesa 26.0.8 alone, below the committed `lavapipe` profile's release, so it is not listed
/// either. One row per GPU, as measured.
///
/// `MetalHalfConversion`: MoltenVK 1.4.1's `OpFConvert` from 32 to 16 bits of the result of an
/// `OpFAdd` decorated `NoContraction`, which SPIRV-Cross writes as Metal's `half(float)` of its
/// `[[clang::optnone]]` `spvFAdd` helper's result. Our probes convert `x + -0.0` so for every one
/// of the 2^32 `float` bit patterns `x`, scalar and four-lane, and for a
/// computed `x`, in each of Metal's three math modes (fast, relaxed, safe), on the M3 Max and the
/// M4: none differs from round to nearest even on the half grid with half denormals kept, the sign
/// of a zero kept and infinity from 65520 up, a NaN gives a NaN, and each NaN's half has the bits
/// of Metal's own conversion of the NaN; a control against truncation differs, as it must. Without
/// the add, Metal's compiler folds the conversion into the instruction computing `x`, which rounds
/// toward zero (measured the conversion of loaded values alone). The gate lanes `half_quantize_*`
/// and `owned_float_fixtures_*` run the `half` cases through it. One row per GPU, as measured.
/// Measured and not listed: NVIDIA's 615.71.09 on the RTX 5080 Laptop GPU converts every one of the
/// 2^32 `float` patterns through the add as Metal does when the value is loaded, four-lane,
/// `fma(x, 1, -0)` or a product (the scratch commit `e867099bb`), but with the row taken, 18 C3
/// cases that matched Metal on that GPU differed (at master `c0af7712`: a computed `half` NaN
/// stored as `0x7FFF` where Metal stores `0x7E00`, among others); on the RTX 3090 the same driver
/// converts toward zero; lavapipe (Mesa 26.0.8 on LLVM 21.1.8, x86_64) converts a loaded value
/// toward zero (its compiler drops the add).
///
/// `MetalCenterVaryings`: MoltenVK 1.4.1 hands Metal a smooth input decorated neither `Centroid`
/// nor `Sample` with no interpolation qualifier (SPIRV-Cross `adec7acb`, `spirv_msl.cpp`
/// 13664-13720), which Metal reads as `center_perspective` (`NoPerspective`:
/// `center_no_perspective`), whatever the sample rate, and Metal reads both at the pixel center in
/// every invocation, a sample-rate one's included. The six authored cases of the owned fixture
/// `engine/multisample` that read center varyings in a function Metal runs per sample
/// (`multisample_center_at_sample`, `multisample_counter` and `multisample_sampled_varying`, at 2
/// and 4 samples) match Metal's words through MoltenVK with the varyings read as they are, on the
/// M3 Max and the M4. One row per GPU, as measured.
///
/// The two color-write conversions, measured by `runtime_lane::color_writes` with the fragment
/// function's rounding left out (at the scratch commit 9bc673b64, 2026-10-07, and our probe
/// on the UHD 770 the same day): every code's exact value, every tie between neighbouring codes
/// with two `f32` steps either side, and the specials (infinities, past the largest, negatives,
/// every NaN class), of `float` outputs, plain and blended with factors one and zero, against
/// Metal's rule:
/// - `MetalSmallFloatColorWrites`: RADV, Mesa 26.2.4 on the Radeon 780M; ANV, Mesa 26.2.4 on the
///   Iris Xe (RPL-P) and on the UHD 770 (RPL-S), alike channel for channel (the one Intel venue's
///   two devices); and NVIDIA's 615.71.09 on the RTX 3090: every channel of `R16Float`,
///   `RG16Float`, `RGBA16Float` and `RG11B10Float` Metal's but NaNs (RADV and ANV store every
///   NaN of a half-float channel as `0xfe00`, as they store the rounding's; NVIDIA keeps some
///   NaNs' sign or payload, where it stores the rounding's `0x7fc00000` as Metal's NaN).
/// - `MetalNorm16ColorWrites`: RADV, Mesa 26.2.4 on the Radeon 780M: every channel of
///   `R16Unorm`, `RG16Unorm`, `RGBA16Unorm`, `R16Snorm`, `RG16Snorm` and `RGBA16Snorm` Metal's,
///   NaNs included, but a blended snorm channel's NaN (0, where Metal stores -1.0's code), from
///   `half` outputs too. ANV and NVIDIA store another code near ties.
///
/// One row per GPU, as measured. The gate lane `runtime_lane::color_writes::draws_store_metals_codes`
/// holds them: it draws through the device's own conversion wherever the profile takes them.
///
/// `MetalMinMax`: MoltenVK 1.4.1's `GLSL.std.450` `NMin`, `NMax` and `NClamp` on 32-bit floats,
/// which SPIRV-Cross writes `precise::min`, `precise::max` and `precise::clamp`, against Metal's
/// `fmin`, `fmax` and `clamp` as `math::metal_min_max` computes them, in each of Metal's three math
/// modes (fast, as MoltenVK compiles; relaxed; safe), operands through a run-time select: every
/// pair of widened `half`s (2^32, the translation holds a `half` in a `float`), scalar and
/// four-lane, 2^32 hashed `float` pairs of every class, scalar and four-lane, a 2,048-value `float`
/// set of specials, NaNs of several payloads, denormals and neighbours pair by pair, and `clamp` of
/// every widened `half` against 48 x 48 bounds and of the `float` set against 64 x 64; none
/// differs, on the M3 Max and the M4. A control over the hashed pairs whose two NaNs answer the
/// first operand differs, as it must. The `half` forms (`min`, `max`, `clamp` of `half`,
/// every pair) match too, though the translation does not emit them. One row per GPU, as measured.
///
/// `MetalHalfArithmetic`: MoltenVK 1.4.1's 16-bit `OpFAdd` and `OpFSub` decorated `NoContraction`,
/// which SPIRV-Cross writes as its `[[clang::optnone]]` `spvFAdd` and `spvFSub`, and its 16-bit
/// `Fma(a, b, -0.0)` decorated `NoContraction`, written `fma(a, b, -0.0h)`, each with operands
/// narrowed from `float` by `half()` and the result widened by `float()`, against Metal's own
/// `half` arithmetic as the translation emulates it: the exact result rounded once to nearest even
/// on the `half` grid, `half` denormals kept, an exact zero sum `+0` unless both addends are `-0`,
/// and every NaN result `0x7E00`, by integer references, over every pair of `half`s (2^32), in each
/// of Metal's three math modes, on one to four lanes, one kept result narrowed into the next, and a
/// kept result under a plain `float` sum or product (measured the native operators on `half`
/// values). None differs. The same operations undecorated differ in the fast and relaxed modes,
/// which fold the narrowings and the widening away (3,468,601,344 pairs for a sum, 4,014,926,892
/// for a product, 6,140 for a quotient); so does SPIRV-Cross's `NoContraction` `OpFMul`, `spvFMul`,
/// which answers `+0` for a `-0` product (126,974 pairs). The translation emits none of them. The
/// sums read every `half` through the device's widening (`x + -0` of every `x`), which a load of
/// `half`s whose uses read no NaN's bits takes (`lower::arith::widening`); Metal's own widening,
/// read on the M3 Max (macOS 27.0.1) over every bit pattern from threadgroup memory in fast and
/// safe math, takes every `half` but a NaN to its value, denormals included, and every NaN to
/// `0x7FC00000`. One row per GPU, as measured. `MetalFragmentSimdgroups`: MoltenVK 1.4.1's SPIRV-Cross writes a
/// fragment shader's `SubgroupLocalInvocationId` as Metal's `[[thread_index_in_simdgroup]]`, and
/// Metal runs the fragment, so each fragment subgroup is a Metal simdgroup of the same pixels.
/// Metal's own lanes were read from MSL of ours on the M3 Max (macOS 26.7) and the M4 (macOS
/// 27.0.1), equal on both and from run to run: in an aligned 8x4 block one primitive covers whole,
/// quads in row-major order; elsewhere, the rasterizer packs the covered quads of several blocks
/// and primitives into one simdgroup. The gate lane
/// `runtime_lane::driver_deviations::fragment_simdgroup_lanes_are_metal_s` compares MoltenVK's
/// subgroup lanes with Metal's lanes of the same draws on each chip. One row per GPU.
///
/// `MetalVertexSimdgroups` (the coordinator's reversal of the MoltenVK refusal, 2026-10-08):
/// MoltenVK 1.4.1 leaves the vertex stage out of `subgroupSupportedStages`, but hands the vertex
/// function to Metal, which runs it in simdgroups of 32 consecutive vertices. With the stage forced
/// into the profile it ran `rvc_simd_vertex` (`simd_sum`, `simd_shuffle`, `simd_shuffle_xor`,
/// `simd_ballot`) as Metal does on both chips (200 of 200 words); the lane
/// `runtime_lane::driver_deviations:: vertex_simdgroups_are_metal_s` runs it and the
/// explicit-width fill on each chip. One row per GPU.
///
/// `CenterFragCoordDepth`: MoltenVK 1.4.1 on the M4 gives a per-sample fragment invocation at 4
/// samples `FragCoord.z` at the pixel centre, its depth bias included (1 of 15 cases equal to the
/// stored per-sample depth, the flat one), and the vertex function's no-perspective window depth
/// read at the sample through `InterpolateAtSample`, plus `FragCoord.z` less the same varying read
/// as it is (the centre), equals the depth it stores at each sample in the flat, sloped, steep,
/// constant and negative bias, clamp, clip, clamped-range and past-the-range cases, and within 2
/// ulp in the perspective, slope-scaled bias and narrowed or reversed range cases (16,384 samples a
/// case; four arithmetic orders of the window depth alike). The M3 Max (macOS 26.7) answered every
/// 4-sample row as the M4 did (the same probe binary, the coordinator's run of 2026-10-09). One row
/// per GPU.
///
/// `SampleFragCoordDepth`: RADV, Mesa 26.2.4 on the Radeon 780M, gives a per-sample fragment
/// invocation at 4 samples `FragCoord.z` at its sample, its depth bias included: in the in-shader
/// test's form (the depth attachment kept in the subpass under `Always`), equal bit for bit to the
/// depth the device stores at each sample in all 15 cases (flat, sloped, steep and perspective
/// triangles, constant, negative, slope-scaled, steep and clamped bias, clamp and clip modes, and
/// default, narrowed, reversed and clamped viewport ranges; 13,819 to 16,384 samples a case).
/// Clamped as Metal clamps, it equals the stored depth in 14: in the steep bias case
/// the device stores 17 samples past 1 unclamped, where Metal stores 1, which the clamp gives.
/// MoltenVK (the pixel centre's, measured as `CenterFragCoordDepth`), ANV and NVIDIA (the pixel
/// centre's, 1 of 15 each) do not.
///
/// The half-float storage writes, measured by `runtime_lane::half_storage_writes` with the shader's
/// rounding left out (at fd90c81e3, 2026-10-08): a kernel's `float` write through `access::write`,
/// `access::read_write`, a texture buffer and an argument buffer's texture to `R16Float`,
/// `RG16Float` and `RGBA16Float` of every finite half and every tie between neighbouring halves
/// with one `f32` step either side, of either sign, and the specials (zeros, infinities, past the
/// largest, denormals, NaNs of four payloads), against Metal's rule (toward zero, past the largest
/// 65504, every NaN `0x7e00`; every form, plain and under Metal's debug layer):
/// - `MetalHalfStorageWrites`: every channel but NaNs Metal's, on MoltenVK 1.4.1 on the M3 Max
///   and the M4, RADV, Mesa 26.2.4 on the Radeon 780M, ANV, Mesa 26.2.4 on the Iris Xe (RPL-P) and
///   the UHD 770 (RPL-S), and NVIDIA's 615.71.09 on the RTX 3090 and the RTX 5080 Laptop GPU, and
///   lavapipe (`MESA_LLVMPIPE`) from Mesa 26.0.8 through 26.2.4: Mesa 26.0.8 on LLVM 21.1.8, x86_64
///   (the linux-gate container) and arm64 (on an M4 mini), and Mesa 26.2.4 on LLVM 23.1.1, x86_64
///   (Arch's, on an i9-13900), all at fd90c81e3.
/// - `MetalHalfStorageNans`: every NaN `0x7e00` too, on MoltenVK 1.4.1 on the M3 Max and the M4.
/// - `MetalFloatStorageNans`: the NaN `0x7fc00000` stored as `0x7e00`, on MoltenVK 1.4.1 on the
///   M3 Max and the M4, RADV on the Radeon 780M, NVIDIA's on the RTX 3090 and the RTX 5080
///   Laptop GPU, and lavapipe from Mesa 26.0.8 through 26.2.4 (the same probes). RADV and NVIDIA
///   keep another NaN's sign and top payload bits, lavapipe its sign (and on x86_64 its top
///   payload bits). ANV stores every NaN as `0xfe00`, so it is not listed.
///
/// One row per GPU, as measured.
///
/// `MetalHalfTexelWrites`, measured by `runtime_lane::half_write_nans` (at f38e6b9fc, 2026-10-09):
/// a kernel's `half4` write through `access::write` and `access::read_write` of quiet and
/// signalling NaNs of both signs and payload edges, with infinities, zeros, denormals and normals,
/// handed to the device widened: RADV, Mesa 26.2.4 on the Radeon 780M, and NVIDIA's 615.71.09 on
/// the RTX 3090 and the RTX 5080 Laptop GPU store each half's own bits in `R16Float`, `RG16Float`
/// and `RGBA16Float`, as Metal does. MoltenVK 1.4.1 on the M4 stores every NaN `0x7e00`, ANV, Mesa
/// 26.2.4 on the Iris Xe (RPL-P), `0xfe00`, and lavapipe (Mesa 26.0.8 on LLVM 21.1.8, x86_64)
/// quiets a signalling NaN, so none is listed. Every one stores a 32-bit float channel's bits as
/// written.
///
/// `MetalNormStorageWrites`, measured by `runtime_lane::norm_storage_writes` with the shader's
/// rounding left out (at a6609204 and our probe at 69fee0d4, 2026-10-10): a kernel's `float`
/// write through `access::write`, `access::read_write`, a texture buffer and an argument buffer's
/// texture of every code's value and every tie between codes with two `f32` steps either side, and
/// the specials (zeros, NaNs, infinities, past the range), to the 8- and 16-bit unorm and snorm
/// formats, `RGB10A2Unorm`, `BGR10A2Unorm`, `A8Unorm` and the packed 16-bit ones, against Metal's
/// rule (the exact product rounded to nearest with a tie away from zero, a NaN 0, a 1-bit channel 1
/// only above one half): MoltenVK 1.4.1 on the M4 stores every channel of every format it stores
/// (it binds no packed 16-bit format for storage) as Metal does, and RADV, Mesa 26.2.4 on the
/// Radeon 780M, every channel of every format but the 1-bit alpha of `A1BGR5Unorm` and
/// `BGR5A1Unorm`, which it rounds up at one half, where Metal stores 0 (such a texture's writes
/// take `StorageImageClass::OneBitAlpha` there). ANV, Mesa 26.2.4 on the Iris Xe (RPL-P), and
/// lavapipe (Mesa 26.0.8 on LLVM 21.1.8, x86_64) round the `f32` product with ties to even, and
/// NVIDIA's 615.71.09 on the RTX 3090 stores the code below many values just past a tie, so none is
/// listed. MoltenVK 1.4.1 on the M3 Max under macOS 27.0.1, of the M4's identity ([`APPLE_M4`]),
/// stores every channel as Metal does too (2026-10-14: both of `norm_storage_writes`' tests with
/// the fact taken, 0 channels off in every form and format it stores, the packed 16-bit ones
/// refused for storage as on the M4).
///
/// `Unorm8LinearFilter`, measured by the deviation probe (`probes/shb-dev` at the scratch commit
/// `aa944573d` and its `cell` variant, the translator's native coordinate; our probes compared, an
/// Apple M4 on macOS 27.0.1, 2026-10-10): 19x13 textures of `R8Unorm`, `RG8Unorm`,
/// `RGBA8Unorm` and `BGRA8Unorm` sampled at 4,096 coordinates (a third near the 1/256 grid, the
/// rest within `[-1.5, 2.5]`) through a linear filter under clamp to edge, repeat, mirrored repeat
/// and clamp to zero, at the centre of Metal's footprint step: no channel of a `float` result one
/// code (1/255) or more from Metal's. The largest: 0.849 on NVIDIA's 615.71.09 on the RTX 3090 and
/// the RTX 5080 Laptop GPU (alike lane for lane); 0.031 on RADV, Mesa 26.2.4 on the Radeon 780M,
/// and on ANV, Mesa 26.2.4 on the Iris Xe (RPL-P) and the UHD 770 (RPL-S); just under 1 on
/// lavapipe, Mesa 26.2.4 on LLVM 23.1.1, x86_64, whose filter answers whole codes. It is a bound
/// the owner accepted, not Metal's answer. One row per GPU, as measured; lavapipe's on every CPU at
/// that release. MoltenVK (whose sampler is Metal's: `MetalSampler`) and lavapipe before Mesa
/// 26.2.4 are not measured.
///
/// `MetalDepth16Stores`, measured by `runtime_lane::unorm_depth_bias`'s unbiased rows (the 16
/// depths of probe 7 and the 39 of probe 7b around the midpoints of codes 100 and 40000, each
/// drawn at a constant of 0; at d51c2aa8, 2026-10-07), every code Metal's:
/// - RADV, Mesa 26.2.4 on the Radeon 780M.
/// - ANV, Mesa 26.2.4 on the Iris Xe (RPL-P) and Intel Graphics (RPL-S) (on the Iris Xe, on the
///   RPL-S). One row per GPU, as measured.
/// - MoltenVK 1.4.1 on Apple's GPUs (an M4): Metal stores the depth on every
///   Apple GPU, by construction, MoltenVK making `D16_UNORM` Metal's own `Depth16Unorm`.
///
/// Measured not to have it, so not listed: lavapipe, Mesa 26.0.8 on LLVM 21.1.8 and Mesa 26.2.4
/// on LLVM 23.1.1, x86_64, which scales the depth by 65535 as a `float` and rounds the tie that can
/// make to even: 0.3 (`0x3e99999a`, 19660.50078 codes) stored 19660 and 0.9 (`0x3f666666`,
/// 58981.49844) stored 58982, where Metal stores 19661 and 58981; and NVIDIA's 615.71.09 on the RTX
/// 5080 Laptop GPU, which stored the code below Metal's for 26 of the 55 unbiased depths:
/// every depth whose exact scaled value lies at a code's midpoint or above it by up to 0.55 of a
/// code (0.5's 32767.5 stored 32767, 0.1's 6553.50010 stored 6553, 100.55 stored 100), Metal's
/// code from 100.5625 up, as a conversion keeping four fractional bits and rounding their
/// midpoint down would.
///
/// `MetalDepth16BiasedStores`, measured by
/// `runtime_lane::unorm_depth_bias::a_constant_bias_on_a_16_bit_unorm_depth_stores_metals_codes`
/// (every depth and constant of the probes' draws, each biased by the device: our probe at
/// d51c2aa8, 2026-10-07):
/// - RADV, Mesa 26.2.4 on the Radeon 780M, through `VK_EXT_depth_bias_control`'s float
///   representation with `depthBiasExact`: every code Metal's.
/// - MoltenVK 1.4.1 on Apple's GPUs, which takes Metal's constant in Metal's own unit (it has no
///   `VK_EXT_depth_bias_control`): every code Metal's on an M4 mini. On
///   every Apple GPU by construction: MoltenVK 1.4.1 hands the recorded constant, slope and clamp
///   unchanged to the render encoder's `setDepthBias:slopeScale:clamp:`
///   (`MVKCommandEncoderState.mm`) and makes the depth Metal's own `Depth16Unorm`
///   (`MVKPixelFormats.mm`), so Metal biases and stores the depth on whichever Apple GPU runs it.
///
/// Measured not to have it, so not listed, each storing the code next to Metal's for a sum near a
/// code's midpoint: ANV, Mesa 26.2.4 on the Iris Xe (RPL-P), which rounds the sum to nearest, not
/// toward zero; lavapipe, Mesa 26.0.8 on LLVM 21.1.8, x86_64 (the linux-gate container), which also
/// scales it by 65535 as a `float` (0 biased by 0.5 stored as 2 where Metal's is 1); NVIDIA's
/// 615.71.09 on the RTX 5080 Laptop GPU, through `depthBiasExact` in 2^-16, whose conversion of
/// every depth rounds down near a midpoint (`MetalDepth16Stores` above). ANV on the UHD 770 (RPL-S)
/// is unmeasured.
///
/// `MetalInterpolation`: MoltenVK 1.4.1 hands each fragment input to Metal with its
/// interpolation (SPIRV-Cross `adec7acb` writes it as a `[[user(...)]]` input qualified
/// `center_perspective`, `center_no_perspective`, `centroid_*` or `sample_*`, and an
/// `InterpolateAt*` read as Metal's `interpolant`), so Metal's rasterizer interpolates every
/// varying, on whichever Apple GPU runs it. Measured on the M3 Max under macOS 27.0.1 (2026-10-15):
/// seven triangle-list draws of a `float4` attribute read as a perspective and a no-perspective
/// varying (one-hot vertices of a triangle list whose `w` runs from 1 to 5, at 32x32 and 64x64;
/// texcoords and scaled texcoords over a full-screen quad at 64x64 and 100x60 and a full-screen
/// triangle at 100x60; a colour whose `w` runs from 0.6 to 3, at 64x64 and 100x60), every one of
/// the 210,082 interpolated lanes MoltenVK wrote equal to Metal's bit for bit. Every other driver
/// the same probe measured interpolates otherwise, so none is listed (bit-equal lanes of the
/// 105,041 of each form, perspective and not, and the largest difference):
/// - RADV, Mesa 26.2.4 on the Radeon 780M: 38,825 and 43,307, up to `1.2e-4` apart.
/// - ANV, Mesa 26.2.4 on the Iris Xe (RPL-P): 41,417 and 44,356, up to
///   `7.6e-5` apart.
/// - lavapipe, Mesa 26.2.4 on LLVM 23.1.1, x86_64: 32,968 and 35,422, up to
///   `3e-4` apart (it snaps vertices otherwise).
/// - NVIDIA's 615.71.09 on the RTX 5080 Laptop GPU and the RTX 3090, alike word for word: 41,433
///   and 47,662, up to `1.5e-4` apart.
///
/// Even the full-screen quad at 64x64, whose texcoords are exact at every pixel center, differs
/// from Metal on each of them in a scaled lane (`17u + 0.3`).
///
/// `MetalFirstVertexIndices`: MoltenVK 1.4.1 hands a vertex function's `Layer` and `ViewportIndex`
/// to Metal as its own `[[render_target_array_index]]` and `[[viewport_array_index]]` outputs
/// (SPIRV-Cross `adec7acb`, `spirv_msl.cpp` 17815-17820, `CompilerMSL::builtin_qualifier`), so
/// Metal's rasterizer takes a primitive's slice and viewport as it takes its own, from the
/// primitive's first vertex, on whichever Apple GPU runs it. Measured on the M3 Max under macOS
/// 27.0.1 (2026-10-16): `runtime_lane::first_vertex`'s twelve draws of the owned fixture
/// `rt6_first_vertex` (triangle lists in three orders and with indices past the counts, triangle
/// strips both ways, lines both ways, a line strip and points; each vertex its own slice and
/// viewport) left through MoltenVK, with the indices stored from the vertex function, Metal's
/// words in every slice and viewport quadrant, as a Metal probe of the same functions on the same
/// GPU left them.
///
/// MoltenVK 1.4.2 (its SPIRV-Cross `6c09849f`) on the M3 Max under macOS 27.0.1
/// ([`moltenvk_142`]; 2026-10-21, the runtime lanes in release builds under `--cfg m2v_gpu`): every
/// deviation of the 1.4.1 rows of [`APPLE_M4`]'s identity, each taken, passed the lanes that
/// measured it (`driver_deviations`' lockstep, float function, denormal, half, minimum and
/// maximum, cube gradient, fragment and vertex simdgroup checks, `color_writes`, `first_vertex`,
/// `renders::public_render_cases_match_metal` and the authored cases against the M3 Max's Metal
/// rows), and SPIRV-Cross's `spvGradientCube` is `adec7acb`'s. `MetalFullQuads` is 1.4.2's alone:
/// 1.4.2 stops advertising `shaderQuadControl` on Apple's GPUs, and its SPIRV-Cross writes no
/// mode for `RequireFullQuadsKHR` but hands the fragment function to Metal, whose quads keep their
/// helpers; the lanes that took the feature before (`discard_counts`, `discard_hoist`,
/// `helper_tangle`, `driver_deviations::fragment_simdgroup_lanes_are_metal_s` and the public
/// render cases) answer Metal's with it.
///
/// Another driver has none of them, as Vulkan states, but a listed driver at a release outside
/// its rows, or on another vendor's GPU (or another GPU than a row's), is unmeasured: the profile
/// then takes each one's conservative route (`derive.rs`).
pub(crate) const DEVIATIONS: &[Measured<DriverDeviation>] = &[
    moltenvk_141(DriverDeviation::SimdgroupLockstep),
    moltenvk_141(DriverDeviation::MetalFloatFunctions),
    moltenvk_141(DriverDeviation::MetalCanonicalNans),
    moltenvk_141(DriverDeviation::MetalSampler),
    moltenvk_141(DriverDeviation::CubeGradientRoute),
    Measured {
        device: Some(APPLE_M3_MAX),
        ..moltenvk_141(DriverDeviation::GraphicsCubeGradientRoute)
    },
    Measured {
        device: Some(APPLE_M4),
        ..moltenvk_141(DriverDeviation::GraphicsCubeGradientTangents)
    },
    Measured {
        device: Some(APPLE_M3_MAX),
        ..moltenvk_141(DriverDeviation::MetalColorWrites)
    },
    Measured {
        device: Some(APPLE_M4),
        ..moltenvk_141(DriverDeviation::MetalColorWrites)
    },
    nvidia_61571(DriverDeviation::MetalFloatColorNans, NVIDIA_RTX_3090),
    nvidia_61571(DriverDeviation::MetalFloatColorNans, NVIDIA_RTX_5080_LAPTOP),
    nvidia_61571(DriverDeviation::MetalHalfColorNans, NVIDIA_RTX_3090),
    nvidia_61571(DriverDeviation::MetalHalfColorNans, NVIDIA_RTX_5080_LAPTOP),
    Measured {
        device: Some(APPLE_M3_MAX),
        ..moltenvk_141(DriverDeviation::HalfColorOutputBits)
    },
    Measured {
        device: Some(APPLE_M4),
        ..moltenvk_141(DriverDeviation::HalfColorOutputBits)
    },
    Measured {
        device: Some(APPLE_M3_MAX),
        ..moltenvk_141(DriverDeviation::MetalHalfFma)
    },
    Measured {
        device: Some(APPLE_M4),
        ..moltenvk_141(DriverDeviation::MetalHalfFma)
    },
    Measured {
        fact: DriverDeviation::MetalFloatColorNans,
        driver: DriverId::MESA_LLVMPIPE,
        vendor: None,
        device: None,
        first: mesa(26, 0, 8),
        last: mesa(26, 2, 4),
    },
    Measured {
        device: Some(APPLE_M3_MAX),
        ..moltenvk_141(DriverDeviation::MetalFloatDenormals)
    },
    Measured {
        device: Some(APPLE_M4),
        ..moltenvk_141(DriverDeviation::MetalFloatDenormals)
    },
    mesa_2624(
        DriverDeviation::MetalFloatDenormals,
        DriverId::MESA_RADV,
        (AMD_VENDOR_ID, AMD_RADEON_780M),
    ),
    mesa_2624(
        DriverDeviation::MetalFloatDenormals,
        DriverId::INTEL_OPEN_SOURCE_MESA,
        (INTEL_VENDOR_ID, INTEL_UHD_770_RPL_S),
    ),
    mesa_2624(
        DriverDeviation::MetalFloatDenormals,
        DriverId::INTEL_OPEN_SOURCE_MESA,
        (INTEL_VENDOR_ID, INTEL_IRIS_XE_RPL_P),
    ),
    nvidia_61571(DriverDeviation::MetalFloatDenormals, NVIDIA_RTX_3090),
    nvidia_61571(DriverDeviation::MetalFloatDenormals, NVIDIA_RTX_5080_LAPTOP),
    Measured {
        device: Some(APPLE_M3_MAX),
        ..moltenvk_141(DriverDeviation::MetalHalfQuantize)
    },
    Measured {
        device: Some(APPLE_M4),
        ..moltenvk_141(DriverDeviation::MetalHalfQuantize)
    },
    mesa_2624(
        DriverDeviation::MetalHalfQuantize,
        DriverId::MESA_RADV,
        (AMD_VENDOR_ID, AMD_RADEON_780M),
    ),
    mesa_2624(
        DriverDeviation::MetalHalfQuantize,
        DriverId::INTEL_OPEN_SOURCE_MESA,
        (INTEL_VENDOR_ID, INTEL_UHD_770_RPL_S),
    ),
    mesa_2624(
        DriverDeviation::MetalHalfQuantize,
        DriverId::INTEL_OPEN_SOURCE_MESA,
        (INTEL_VENDOR_ID, INTEL_IRIS_XE_RPL_P),
    ),
    Measured {
        device: Some(APPLE_M3_MAX),
        ..moltenvk_141(DriverDeviation::MetalHalfConversion)
    },
    Measured {
        device: Some(APPLE_M4),
        ..moltenvk_141(DriverDeviation::MetalHalfConversion)
    },
    Measured {
        device: Some(APPLE_M3_MAX),
        ..moltenvk_141(DriverDeviation::MetalCenterVaryings)
    },
    Measured {
        device: Some(APPLE_M4),
        ..moltenvk_141(DriverDeviation::MetalCenterVaryings)
    },
    mesa_2624(
        DriverDeviation::MetalSmallFloatColorWrites,
        DriverId::MESA_RADV,
        (AMD_VENDOR_ID, AMD_RADEON_780M),
    ),
    mesa_2624(
        DriverDeviation::MetalSmallFloatColorWrites,
        DriverId::INTEL_OPEN_SOURCE_MESA,
        (INTEL_VENDOR_ID, INTEL_UHD_770_RPL_S),
    ),
    mesa_2624(
        DriverDeviation::MetalSmallFloatColorWrites,
        DriverId::INTEL_OPEN_SOURCE_MESA,
        (INTEL_VENDOR_ID, INTEL_IRIS_XE_RPL_P),
    ),
    nvidia_61571(DriverDeviation::MetalSmallFloatColorWrites, NVIDIA_RTX_3090),
    mesa_2624(
        DriverDeviation::MetalNorm16ColorWrites,
        DriverId::MESA_RADV,
        (AMD_VENDOR_ID, AMD_RADEON_780M),
    ),
    Measured {
        device: Some(APPLE_M3_MAX),
        ..moltenvk_141(DriverDeviation::MetalMinMax)
    },
    Measured {
        device: Some(APPLE_M4),
        ..moltenvk_141(DriverDeviation::MetalMinMax)
    },
    Measured {
        device: Some(APPLE_M3_MAX),
        ..moltenvk_141(DriverDeviation::MetalHalfArithmetic)
    },
    Measured {
        device: Some(APPLE_M4),
        ..moltenvk_141(DriverDeviation::MetalHalfArithmetic)
    },
    Measured {
        device: Some(APPLE_M3_MAX),
        ..moltenvk_141(DriverDeviation::MetalHalfStorageWrites)
    },
    Measured {
        device: Some(APPLE_M4),
        ..moltenvk_141(DriverDeviation::MetalHalfStorageWrites)
    },
    mesa_2624(
        DriverDeviation::MetalHalfStorageWrites,
        DriverId::MESA_RADV,
        (AMD_VENDOR_ID, AMD_RADEON_780M),
    ),
    mesa_2624(
        DriverDeviation::MetalHalfStorageWrites,
        DriverId::INTEL_OPEN_SOURCE_MESA,
        (INTEL_VENDOR_ID, INTEL_UHD_770_RPL_S),
    ),
    mesa_2624(
        DriverDeviation::MetalHalfStorageWrites,
        DriverId::INTEL_OPEN_SOURCE_MESA,
        (INTEL_VENDOR_ID, INTEL_IRIS_XE_RPL_P),
    ),
    nvidia_61571(DriverDeviation::MetalHalfStorageWrites, NVIDIA_RTX_3090),
    nvidia_61571(
        DriverDeviation::MetalHalfStorageWrites,
        NVIDIA_RTX_5080_LAPTOP,
    ),
    Measured {
        fact: DriverDeviation::MetalHalfStorageWrites,
        driver: DriverId::MESA_LLVMPIPE,
        vendor: None,
        device: None,
        first: mesa(26, 0, 8),
        last: mesa(26, 2, 4),
    },
    Measured {
        device: Some(APPLE_M3_MAX),
        ..moltenvk_141(DriverDeviation::MetalHalfStorageNans)
    },
    Measured {
        device: Some(APPLE_M4),
        ..moltenvk_141(DriverDeviation::MetalHalfStorageNans)
    },
    Measured {
        device: Some(APPLE_M3_MAX),
        ..moltenvk_141(DriverDeviation::MetalFloatStorageNans)
    },
    Measured {
        device: Some(APPLE_M4),
        ..moltenvk_141(DriverDeviation::MetalFloatStorageNans)
    },
    mesa_2624(
        DriverDeviation::MetalFloatStorageNans,
        DriverId::MESA_RADV,
        (AMD_VENDOR_ID, AMD_RADEON_780M),
    ),
    nvidia_61571(DriverDeviation::MetalFloatStorageNans, NVIDIA_RTX_3090),
    nvidia_61571(
        DriverDeviation::MetalFloatStorageNans,
        NVIDIA_RTX_5080_LAPTOP,
    ),
    Measured {
        fact: DriverDeviation::MetalFloatStorageNans,
        driver: DriverId::MESA_LLVMPIPE,
        vendor: None,
        device: None,
        first: mesa(26, 0, 8),
        last: mesa(26, 2, 4),
    },
    Measured {
        device: Some(APPLE_M3_MAX),
        ..moltenvk_141(DriverDeviation::MetalFragmentSimdgroups)
    },
    Measured {
        device: Some(APPLE_M4),
        ..moltenvk_141(DriverDeviation::MetalFragmentSimdgroups)
    },
    Measured {
        device: Some(APPLE_M3_MAX),
        ..moltenvk_141(DriverDeviation::MetalVertexSimdgroups)
    },
    Measured {
        device: Some(APPLE_M4),
        ..moltenvk_141(DriverDeviation::MetalVertexSimdgroups)
    },
    Measured {
        device: Some(APPLE_M3_MAX),
        ..moltenvk_141(DriverDeviation::CenterFragCoordDepth)
    },
    Measured {
        device: Some(APPLE_M4),
        ..moltenvk_141(DriverDeviation::CenterFragCoordDepth)
    },
    mesa_2624(
        DriverDeviation::SampleFragCoordDepth,
        DriverId::MESA_RADV,
        (AMD_VENDOR_ID, AMD_RADEON_780M),
    ),
    mesa_2624(
        DriverDeviation::MetalHalfTexelWrites,
        DriverId::MESA_RADV,
        (AMD_VENDOR_ID, AMD_RADEON_780M),
    ),
    nvidia_61571(DriverDeviation::MetalHalfTexelWrites, NVIDIA_RTX_3090),
    nvidia_61571(
        DriverDeviation::MetalHalfTexelWrites,
        NVIDIA_RTX_5080_LAPTOP,
    ),
    Measured {
        device: Some(APPLE_M4),
        ..moltenvk_141(DriverDeviation::MetalNormStorageWrites)
    },
    mesa_2624(
        DriverDeviation::MetalNormStorageWrites,
        DriverId::MESA_RADV,
        (AMD_VENDOR_ID, AMD_RADEON_780M),
    ),
    nvidia_61571(DriverDeviation::Unorm8LinearFilter, NVIDIA_RTX_3090),
    nvidia_61571(DriverDeviation::Unorm8LinearFilter, NVIDIA_RTX_5080_LAPTOP),
    mesa_2624(
        DriverDeviation::Unorm8LinearFilter,
        DriverId::MESA_RADV,
        (AMD_VENDOR_ID, AMD_RADEON_780M),
    ),
    mesa_2624(
        DriverDeviation::Unorm8LinearFilter,
        DriverId::INTEL_OPEN_SOURCE_MESA,
        (INTEL_VENDOR_ID, INTEL_IRIS_XE_RPL_P),
    ),
    mesa_2624(
        DriverDeviation::Unorm8LinearFilter,
        DriverId::INTEL_OPEN_SOURCE_MESA,
        (INTEL_VENDOR_ID, INTEL_UHD_770_RPL_S),
    ),
    Measured {
        fact: DriverDeviation::Unorm8LinearFilter,
        driver: DriverId::MESA_LLVMPIPE,
        vendor: None,
        device: None,
        first: mesa(26, 2, 4),
        last: mesa(26, 2, 4),
    },
    mesa_2624(
        DriverDeviation::MetalDepth16Stores,
        DriverId::MESA_RADV,
        (AMD_VENDOR_ID, AMD_RADEON_780M),
    ),
    mesa_2624(
        DriverDeviation::MetalDepth16Stores,
        DriverId::INTEL_OPEN_SOURCE_MESA,
        (INTEL_VENDOR_ID, INTEL_IRIS_XE_RPL_P),
    ),
    mesa_2624(
        DriverDeviation::MetalDepth16Stores,
        DriverId::INTEL_OPEN_SOURCE_MESA,
        (INTEL_VENDOR_ID, INTEL_UHD_770_RPL_S),
    ),
    moltenvk_141(DriverDeviation::MetalDepth16Stores),
    mesa_2624(
        DriverDeviation::MetalDepth16BiasedStores,
        DriverId::MESA_RADV,
        (AMD_VENDOR_ID, AMD_RADEON_780M),
    ),
    moltenvk_141(DriverDeviation::MetalDepth16BiasedStores),
    moltenvk_141(DriverDeviation::MetalInterpolation),
    moltenvk_141(DriverDeviation::MetalFirstVertexIndices),
    moltenvk_142(DriverDeviation::SimdgroupLockstep),
    moltenvk_142(DriverDeviation::MetalFloatFunctions),
    moltenvk_142(DriverDeviation::MetalCanonicalNans),
    moltenvk_142(DriverDeviation::MetalSampler),
    moltenvk_142(DriverDeviation::CubeGradientRoute),
    moltenvk_142(DriverDeviation::GraphicsCubeGradientTangents),
    moltenvk_142(DriverDeviation::MetalColorWrites),
    moltenvk_142(DriverDeviation::HalfColorOutputBits),
    moltenvk_142(DriverDeviation::MetalHalfFma),
    moltenvk_142(DriverDeviation::MetalFloatDenormals),
    moltenvk_142(DriverDeviation::MetalHalfQuantize),
    moltenvk_142(DriverDeviation::MetalHalfConversion),
    moltenvk_142(DriverDeviation::MetalCenterVaryings),
    moltenvk_142(DriverDeviation::MetalMinMax),
    moltenvk_142(DriverDeviation::MetalHalfArithmetic),
    moltenvk_142(DriverDeviation::MetalHalfStorageWrites),
    moltenvk_142(DriverDeviation::MetalHalfStorageNans),
    moltenvk_142(DriverDeviation::MetalFloatStorageNans),
    moltenvk_142(DriverDeviation::MetalFragmentSimdgroups),
    moltenvk_142(DriverDeviation::MetalVertexSimdgroups),
    moltenvk_142(DriverDeviation::CenterFragCoordDepth),
    moltenvk_142(DriverDeviation::MetalNormStorageWrites),
    moltenvk_142(DriverDeviation::MetalDepth16Stores),
    moltenvk_142(DriverDeviation::MetalDepth16BiasedStores),
    moltenvk_142(DriverDeviation::MetalInterpolation),
    moltenvk_142(DriverDeviation::MetalFirstVertexIndices),
    moltenvk_142(DriverDeviation::MetalFullQuads),
];

/// NVIDIA's `vendorID`.
const NVIDIA_VENDOR_ID: u32 = 0x10de;

/// The `deviceID` of the GeForce RTX 3090.
pub(crate) const NVIDIA_RTX_3090: u32 = 0x2204;

/// The `deviceID` of the GeForce RTX 5080 Laptop GPU (the committed capture
/// `nvidia-rtx5080-laptop`).
pub(crate) const NVIDIA_RTX_5080_LAPTOP: u32 = 0x2c19;

/// The row of `fact` measured on NVIDIA's 615.71.09 on the GPU `device`.
const fn nvidia_61571(fact: DriverDeviation, device: u32) -> Measured<DriverDeviation> {
    Measured {
        fact,
        driver: DriverId::NVIDIA_PROPRIETARY,
        vendor: Some(NVIDIA_VENDOR_ID),
        device: Some(device),
        first: nvidia(615, 71, 9),
        last: nvidia(615, 71, 9),
    }
}

/// AMD's `vendorID`.
const AMD_VENDOR_ID: u32 = 0x1002;

/// The `deviceID` of the Radeon 780M it was measured under, Phoenix's (the committed capture
/// `amd-780m-radv`); Hawk Point's 0x1900 is the same GPU, so it takes these rows too
/// ([`DeviceDescription::canonical_device_id`]).
pub(crate) const AMD_RADEON_780M: u32 = 0x15bf;

/// The `deviceID` of the Radeon RX 7900 XTX (Navi 31) its rows were measured under.
pub(crate) const AMD_RX_7900_XTX: u32 = 0x744c;

/// The `deviceID` of the Intel UHD Graphics 770 of Raptor Lake-S (the committed capture
/// `intel-uhd770-rpls-anv`).
pub(crate) const INTEL_UHD_770_RPL_S: u32 = 0xa780;

/// The `deviceID` of the Intel Iris Xe Graphics of Raptor Lake-P (the committed capture
/// `intel-irisxe-rplp-anv`).
pub(crate) const INTEL_IRIS_XE_RPL_P: u32 = 0xa7a0;

/// The row of `fact` measured on Mesa 26.2.4's `driver` on the GPU `(vendor, device)`.
const fn mesa_2624(
    fact: DriverDeviation,
    driver: DriverId,
    (vendor, device): (u32, u32),
) -> Measured<DriverDeviation> {
    Measured {
        fact,
        driver,
        vendor: Some(vendor),
        device: Some(device),
        first: mesa(26, 2, 4),
        last: mesa(26, 2, 4),
    }
}

/// The `deviceID` MoltenVK 1.4.1 reports for an Apple GPU of the Apple9 family under macOS 26.7:
/// the Apple M3 Max's identity where its facts were measured, before its macOS 27 capture. On an
/// Apple GPU MoltenVK's `deviceID` is not the GPU's: it is the macOS version and the GPU's highest
/// families, `(major << 24) + (minor << 16) + (Mac family << 8) + Apple family`, the minor taken
/// from `major + minor / 100` scaled and truncated, so 26.7 reads 6
/// (`MVKPhysicalDevice::getHighestGPUCapability`, MoltenVK 1.4.1). Every GPU of a family on one
/// macOS release has one identity.
pub(crate) const APPLE_M3_MAX: u32 = 0x1A06_0209;

/// The `deviceID` MoltenVK 1.4.1 reports for an Apple GPU of the Apple9 family under macOS 27.0
/// ([`APPLE_M3_MAX`]'s rule): the Apple M4's (the committed capture `moltenvk-m4`) and the Apple
/// M3 Max's under macOS 27.0.1 (the committed capture `moltenvk-m3`, 26A434), one identity. A
/// fact of this identity's rows that decides an answer is measured on both GPUs; a cost fact on
/// the M3 Max, or named as the M4's alone.
pub(crate) const APPLE_M4: u32 = 0x1B00_0209;

/// The row of `fact` measured on MoltenVK 1.4.1 on Apple's GPUs.
const fn moltenvk_141(fact: DriverDeviation) -> Measured<DriverDeviation> {
    Measured {
        fact,
        driver: DriverId::MOLTENVK,
        vendor: Some(APPLE_VENDOR_ID),
        device: None,
        first: moltenvk(1, 4, 1),
        last: moltenvk(1, 4, 1),
    }
}

/// The row of `fact` measured on MoltenVK 1.4.2 (`driverVersion` 10402, Homebrew's build, its
/// SPIRV-Cross `6c09849f`) on the Apple M3 Max under macOS 27.0.1, whose `deviceID` is
/// [`APPLE_M4`]'s: the identity [`MEASURED_DEVICES`] lists at that release, for the M3 Max alone
/// ([`NAMED_RELEASES`]). The M4, which reports the same `deviceID` under macOS 27, is not measured
/// at 1.4.2: its 1.4.1 rows stand beyond their last there.
const fn moltenvk_142<F>(fact: F) -> Measured<F> {
    Measured {
        fact,
        driver: DriverId::MOLTENVK,
        vendor: Some(APPLE_VENDOR_ID),
        device: Some(APPLE_M4),
        first: moltenvk(1, 4, 2),
        last: moltenvk(1, 4, 2),
    }
}

/// Every measured driver defect translation refuses by release (`Needs::DriverDefect`; the
/// coordinator's ruling of 2026-10-06), and every measured answer a module records as a precision
/// note: a defect is kept at a release newer than the last measured (the module docs' rule), and
/// not assumed at an older one or of another driver.
/// - `BothBarycentricFormsRejected`: MoltenVK 1.4.1 (`driverVersion` 10401, SPIRV-Cross
///   `adec7acb`) refused `tail_fragment_barycentric` (`engine/stage_io_tail`) at pipeline
///   creation on an Apple M4. SPIRV-Cross throws in its MSL backend's interface
///   pass whatever the GPU, so the row names no vendor.
/// - `MemoryWritingFragmentsRunPerSample`: MoltenVK 1.4.1 on the Apple M3 Max and the M4 ran the
///   loop over the covered samples of a memory-writing fragment shader without sample shading once
///   per covered sample, every pass repeated per covered sample. Metal runs a function that writes
///   memory once per covered sample; the row names Apple's GPUs, where that was measured.
/// - `InterpolationNeedsTheWholeQuad`: RADV on the Radeon 780M, Mesa 26.0.8 (Ubuntu's) and 26.2.4
///   (Arch's), read garbage through `InterpolateAtSample`, and through `InterpolateAtOffset` at a
///   nonzero offset, in control flow the helper invocations of the pixel's quad had left (a bare
///   Vulkan program; the RTX 3090 and ANV answered every variant right). The row names no vendor:
///   RADV runs only AMD's GPUs.
/// - `MeshIndexComponentStoresMisplaced`: lavapipe, Mesa 26.0.8 (Ubuntu's, x86, the Linux gate's),
///   drew none of `runtime_lane::mesh_draws`'s triangles, whose indices the mesh function writes
///   one at a time. The row names no vendor: the defect is the driver's
///   compiler, whatever CPU it runs on.
/// - `CompositeInsertsWrittenInPlace`: MoltenVK 1.4.1 (SPIRV-Cross `adec7acb`) on the Apple M3 Max
///   and the M4 wrote an insert into a vector defined before a loop in place in the loop, so the
///   next iteration read the inserted element. SPIRV-Cross does this in its MSL conversion whatever
///   the GPU, so the row names no vendor.
/// - `FunctionsInlinedAtEveryCall`: MoltenVK 1.4.1 (SPIRV-Cross `adec7acb`) wrote every function of
///   a module `always_inline`, its `DontInline` routines included, on the Apple M3 Max and the M4.
///   SPIRV-Cross does this in its MSL conversion whatever the GPU, so the row names no vendor.
/// - `LoadsForwardedIntoLoops`: MoltenVK 1.4.1's SPIRV-Cross (`adec7acb`, the pinned converter of
///   `scripts/moltenvk-msl-helper`) wrote a value loaded before a loop into the loop's text, after
///   which a store in the loop changed what the next iteration read (a synthetic module of our
///   own). The converter does this whatever the GPU, so the row names no vendor.
/// - `LocalizedLoadsForwardedPastAtomicsAndCalls`: MoltenVK 1.4.1 on the Apple M3 Max and the M4
///   ran authored case `8b812530` wrong, its MSL reading a threadgroup counter after the atomics
///   that overwrite it, where the module reads it before; the `spirv-cross` CLI does the same for
///   synthetic modules of our own past an atomic and past a call. The converter does this whatever
///   the GPU, so the row names no vendor.
/// - `VertexGroupingNotApples`: ANV on Mesa 26.2.4, on the Iris Xe of the Intel venue, ran the
///   vertex fixtures `rvc_simd_vertex` and `rvc_simd_vertex_fill` in subgroups
///   of 8 interleaved vertices, where the Apple M4 groups 32 consecutive ones. The row names no
///   vendor and no device: ANV runs only Intel's GPUs, and the row only makes a vertex function of
///   simdgroup operations record `PrecisionNote::VertexSimdgroupGrouping`, so on a GPU not measured
///   it announces a difference rather than hides one.
/// - `DontInlineKeepsCalls`: NVIDIA's 615.71.09 on the RTX 3090 kept a call of a `DontInline`
///   sample emulation a call, creating a fragment of 256 such calls in 0.39 to 0.40 s against 10.1
///   to 10.2 s for the same routine without the control, as it kept the linear sums declared so
///   (the RTX 3090). A cost fact: it only chooses between two exact encodings, so a release or GPU
///   not measured loses nothing by going without it and gains no reason to assume it. The same
///   release on the RTX 5080 Laptop GPU made a guest fragment's pipeline under the exact sampling
///   and arithmetic forms (`scripts/making-bench`, cold, its anisotropic emulation one sampling
///   routine) in 22.6 to 22.9 s with each sampling routine `DontInline` and in 45.8 to 54.7 s
///   without, and its linear sampler's pipeline in 8.9 to 11.9 s against 14.8 to 14.9 s, drawing in
///   0.32 against 0.43 ms. Its rows name those two GPUs, the GPUs measured; another NVIDIA GPU
///   takes it once measured.
pub(crate) const DEFECTS: &[Measured<DriverWorkaround>] = &[
    Measured {
        fact: DriverWorkaround::BothBarycentricFormsRejected,
        driver: DriverId::MOLTENVK,
        vendor: None,
        device: None,
        first: moltenvk(1, 4, 1),
        last: moltenvk(1, 4, 1),
    },
    Measured {
        fact: DriverWorkaround::MemoryWritingFragmentsRunPerSample,
        driver: DriverId::MOLTENVK,
        vendor: Some(APPLE_VENDOR_ID),
        device: None,
        first: moltenvk(1, 4, 1),
        last: moltenvk(1, 4, 1),
    },
    Measured {
        fact: DriverWorkaround::InterpolationNeedsTheWholeQuad,
        driver: DriverId::MESA_RADV,
        vendor: None,
        device: None,
        first: mesa(26, 0, 8),
        last: mesa(26, 2, 4),
    },
    Measured {
        fact: DriverWorkaround::MeshIndexComponentStoresMisplaced,
        driver: DriverId::MESA_LLVMPIPE,
        vendor: None,
        device: None,
        first: mesa(26, 0, 8),
        last: mesa(26, 0, 8),
    },
    Measured {
        fact: DriverWorkaround::CompositeInsertsWrittenInPlace,
        driver: DriverId::MOLTENVK,
        vendor: None,
        device: None,
        first: moltenvk(1, 4, 1),
        last: moltenvk(1, 4, 1),
    },
    Measured {
        fact: DriverWorkaround::FunctionsInlinedAtEveryCall,
        driver: DriverId::MOLTENVK,
        vendor: None,
        device: None,
        first: moltenvk(1, 4, 1),
        last: moltenvk(1, 4, 1),
    },
    Measured {
        fact: DriverWorkaround::LoadsForwardedIntoLoops,
        driver: DriverId::MOLTENVK,
        vendor: None,
        device: None,
        first: moltenvk(1, 4, 1),
        last: moltenvk(1, 4, 1),
    },
    Measured {
        fact: DriverWorkaround::LocalizedLoadsForwardedPastAtomicsAndCalls,
        driver: DriverId::MOLTENVK,
        vendor: None,
        device: None,
        first: moltenvk(1, 4, 1),
        last: moltenvk(1, 4, 1),
    },
    Measured {
        fact: DriverWorkaround::VertexGroupingNotApples,
        driver: DriverId::INTEL_OPEN_SOURCE_MESA,
        vendor: None,
        device: None,
        first: mesa(26, 2, 4),
        last: mesa(26, 2, 4),
    },
    Measured {
        fact: DriverWorkaround::DontInlineKeepsCalls,
        driver: DriverId::NVIDIA_PROPRIETARY,
        vendor: Some(NVIDIA_VENDOR_ID),
        device: Some(NVIDIA_RTX_3090),
        first: nvidia(615, 71, 9),
        last: nvidia(615, 71, 9),
    },
    Measured {
        fact: DriverWorkaround::DontInlineKeepsCalls,
        driver: DriverId::NVIDIA_PROPRIETARY,
        vendor: Some(NVIDIA_VENDOR_ID),
        device: Some(NVIDIA_RTX_5080_LAPTOP),
        first: nvidia(615, 71, 9),
        last: nvidia(615, 71, 9),
    },
];

/// The defects of [`DEFECTS`] the driver `description` names has at its release, measured or
/// newer.
pub(crate) fn defects(description: &DeviceDescription) -> BTreeSet<DriverWorkaround> {
    among(
        DEFECTS,
        description.driver_id,
        (description.vendor_id, description.measured_gpu()),
        description.driver_version,
        Some(&description.device_name),
    )
    .into_keys()
    .collect()
}

/// The deviations of the driver `description` names, on its GPU's vendor, at its release.
pub(crate) fn deviations(description: &DeviceDescription) -> BTreeMap<DriverDeviation, Standing> {
    among(
        DEVIATIONS,
        description.driver_id,
        (description.vendor_id, description.measured_gpu()),
        description.driver_version,
        Some(&description.device_name),
    )
}

/// The releases of an identity measured on one GPU of it alone, where its `deviceID` names more
/// than one GPU: the driver, `vendorID`, `deviceID` and `driverVersion`, and the measured GPU's
/// `deviceName`. A row of the identity's GPU that covers the release holds there for that GPU
/// alone ([`among`]'s rows, [`MEASURED`]'s); another GPU of the identity at the release is no
/// measured identity ([`DeviceDescription::measured_identity`]), and the rows of earlier releases
/// stand beyond their last for it.
///
/// MoltenVK 1.4.2 on the Apple9 family under macOS 27 ([`APPLE_M4`]'s identity): measured on the
/// Apple M3 Max (macOS 27.0.1) alone; the Apple M4, measured at 1.4.1, is not at 1.4.2.
pub(crate) const NAMED_RELEASES: &[NamedRelease] = &[NamedRelease {
    driver: DriverId::MOLTENVK,
    vendor: APPLE_VENDOR_ID,
    device: APPLE_M4,
    release: moltenvk(1, 4, 2),
    gpu: "Apple M3 Max",
}];

/// One release of an identity measured on one GPU of it alone ([`NAMED_RELEASES`]).
pub(crate) struct NamedRelease {
    pub(crate) driver: DriverId,
    pub(crate) vendor: u32,
    pub(crate) device: u32,
    pub(crate) release: u32,
    /// The measured GPU's `VkPhysicalDeviceProperties::deviceName`.
    pub(crate) gpu: &'static str,
}

/// Whether a GPU named `name` of `driver`, `vendor` and `device` (its measured `deviceID`) at
/// `version` is another GPU than the one its identity's release was measured on
/// ([`NAMED_RELEASES`]).
fn named_out(
    driver: Option<DriverId>,
    (vendor, device): (u32, Option<u32>),
    version: u32,
    name: &str,
) -> bool {
    NAMED_RELEASES.iter().any(|r| {
        Some(r.driver) == driver
            && r.vendor == vendor
            && Some(r.device) == device
            && r.release == version
            && r.gpu != name
    })
}

/// A driver's measured answer where Vulkan leaves the result undefined, or its measured cost of
/// two exact forms (a cost fact, the module docs).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DriverFact {
    /// Under `robustBufferAccess2`, an index fetched past the bound index buffer range reads
    /// zero, to which the draw adds its vertex offset: Metal's answer past the buffer (which finds
    /// no value stated). An indexed indirect draw, whose positions its arguments choose on the GPU,
    /// stays a single indirect draw on such a driver, its bound range ending where Metal's fetch
    /// stops (`draw_source.rs`); elsewhere the index bounds pass draws it (`index_bounds.rs`).
    RobustIndexFetchReadsZero,
    /// A `D32_SFLOAT` image keeps each denormal value a copy writes into it, as Metal's
    /// `Depth32Float` texel keeps any `float`. Vulkan states no such rule, and RADV's and ANV's
    /// depth images answer a denormal as zero (RADV's of its sign), to a copy out of them and, on
    /// RADV, to the shader's reads. On a driver without it `new_texture` holds a `Depth32Float`
    /// texture the program only reads or samples in an `R32_SFLOAT` image with a `D32_SFLOAT` twin
    /// for the comparisons the device makes (`image_plan::plan_depth_twin`). A clear to a denormal
    /// and a copy from a buffer keep it too on such a driver; RADV and ANV store either as a zero,
    /// and keep only a host copy's (`replace_region` on the host), so a write on the GPU that may
    /// store a denormal there announces `depth-denormals`.
    DepthImageKeepsDenormals,
    /// Without `VK_EXT_depth_range_unrestricted` enabled, a `D32_SFLOAT` image keeps each value
    /// outside `[0, 1]` a copy writes into it, where Vulkan leaves such a value undefined
    /// (`VUID-vkCmdCopyBufferToImage-pRegions-07931`) and ANV stores it clamped. With the extension
    /// Vulkan keeps it on every driver. On a driver without either, `new_texture` holds a
    /// `Depth32Float` texture the program only reads or samples with a twin, as above, and an
    /// upload into a depth image announces `restricted-depth-upload`.
    DepthImageKeepsValuesOutsideUnitRange,
    /// A fragment shader's `FragDepth` is stored in a 32-bit float depth attachment unchanged:
    /// each value in `[0, 1]` it is given, and, for the rest, the value it stored when given it
    /// again. Vulkan states the stored value only inside the depth range. A multisampled depth
    /// is copied, where maintenance8's depth/colour copies are missing or wrong, by a draw that
    /// writes each sample's fetched depth as its fragment depth on such a driver
    /// (`depth_copy.rs`); elsewhere such a copy is refused, typed.
    FragmentDepthStoresUnchanged,
    /// A depth or stencil image lies in memory of its own, not in the memory it is bound to,
    /// where Vulkan places a bound image in its memory. Metal places a heap's depth or stencil
    /// texture in the heap's bytes, so on such a driver that texture aliases none of the heap's
    /// other resources: a write through a buffer or texture over its bytes changes none of its
    /// texels, and its own writes change none of their bytes. Its placement, size and texels are
    /// Metal's otherwise. The runtime announces `heap-depth-stencil-apart` when it makes one in a
    /// heap.
    DepthStencilImagesLieApart,
    /// A render pass's `AVERAGE` resolve of a multisampled colour attachment of 2 or 4 samples
    /// writes Metal's measured resolve for every format with a crate-owned exact resolve
    /// ([`ColorResolveFormat`](crate::bind::ColorResolveFormat)): unorm's `(sum + n/2) / n`,
    /// snorm's `floor((2·sum + n) / 2n)`, sRGB's mean of 12-bit linear values encoded by Metal's
    /// thresholds, and `float`'s flushed sequential sum times `1/n`. Vulkan leaves the average's
    /// rounding to the implementation. On a driver with it the runtime resolves such an attachment
    /// in its pass under either [`ResolvePolicy`](crate::device::ResolvePolicy); elsewhere, and at
    /// 8 samples or more, by that exact shader after the pass under `ResolvePolicy::Exact`, and in
    /// its pass by the device's average, announced, under `ResolvePolicy::Device`
    /// (`color_resolve.rs`).
    /// The fact holds for every colour format the driver resolves, those without a crate-owned
    /// rule too (half floats, packed 16-bit, `RG11B10Float`, `RGB9E5Float`): MoltenVK 1.4.1
    /// resolves a subpass's colour attachment by Metal's own multisample resolve store action
    /// wherever the Metal pixel format resolves (`kMVKMTLFmtCapsResolve`, which every one of these
    /// formats has on Apple's GPUs), so its average is Metal's resolve by construction. The
    /// runtime announces no `device-rounded-resolve` for such an attachment on a driver with it.
    ColorResolveAverageIsMetals,
    /// An optimal colour image may keep its compression state apart from the memory it is bound
    /// to, so a write through a buffer or another image over its bytes can leave its texels as
    /// they were. Vulkan leaves an optimal image's texels undefined after such a write; Metal's
    /// heap texture shows it (every texel the texture reads changes). On such a driver a heap's
    /// texture takes host-transfer usage where the driver has
    /// [`DriverFact::HostTransferImagesKeepTheirBytes`], else a linear image where one serves,
    /// else its optimal image, announced (`heap-texture-hides-aliased-writes`).
    CompressedImagesHideAliasedWrites,
    /// An optimal image of `VK_IMAGE_USAGE_HOST_TRANSFER_BIT` usage keeps its texels in the
    /// memory it is bound to: a write through a buffer over its bytes changes every texel it
    /// covers. Vulkan states no such rule; on such a driver the heap textures that
    /// [`DriverFact::CompressedImagesHideAliasedWrites`] concerns take that usage.
    HostTransferImagesKeepTheirBytes,
    /// A depth or stencil image is laid out as Metal lays out its texture with
    /// `allowGPUOptimizedContents` on, compression metadata included, whatever the image's create
    /// info: no Vulkan request drops the layout, where the image's size is Metal's texture's.
    /// Vulkan leaves an image's layout and size to the driver. Metal's depth and stencil texture
    /// with the property off takes no metadata, so on such a driver a depth or stencil texture
    /// with it off answers a larger heap size than Metal's, and the runtime announces
    /// `optimized-depth-stencil-texture` when it sizes one or makes one in a heap.
    DepthStencilImagesStayGpuOptimized,
    /// A fragment shader's `FragDepth`, given the depth a sample of a `D16_UNORM` depth aspect
    /// fetches as, is stored in a `D16_UNORM` attachment as that sample's code, for every code:
    /// the fetch's conversion to `float` and the attachment's back are each other's inverse.
    /// Vulkan leaves both conversions' rounding to the device. A multisampled `Depth16Unorm`
    /// depth is copied, where maintenance8's depth/colour copies are missing or wrong, and its
    /// `Min` or `Max` resolve drawn where the device has no such mode, by draws that write each
    /// fetched depth as its fragment depth on such a driver (`depth_copy.rs`), as a 32-bit float
    /// depth's are by [`FragmentDepthStoresUnchanged`](Self::FragmentDepthStoresUnchanged);
    /// elsewhere such a copy or resolve is refused, typed.
    FragmentDepthKeepsDepth16Codes,
    /// A sampled view's fetch of a unorm or snorm format of 1 to 16 bits a channel, or of a
    /// 16-bit float format, returns for every code a value from which the mipmap filter's
    /// recovery reads that code back: a unorm code `c` of `b` bits as a `float` within half a
    /// step of `c / (2^b - 1)`, a snorm one of `c / (2^(b-1) - 1)`, a half as its own value
    /// (`bind::MipmapFilterChain::plan`'s `fetch`). Vulkan states no precision for these
    /// conversions. On a driver with it `generate_mipmaps` reads a texture's first level in place
    /// through such a view (`mipmap_filter.rs`); elsewhere it copies the level into its scratch
    /// buffer first. The integer carriers of the extended-range formats are read in place on
    /// every driver: their fetch returns the bits.
    SampledFetchRecoversCodes,
    /// A cost fact (the module docs): a render pass whose draws fetch a color through the pixel
    /// interlock after plain draws wrote it ran no slower split into two render pass instances
    /// there, as the runtime runs it on every device, than kept whole, the plain draws writing the
    /// color through the interlocked storage form (`VariantKey::interlocked_colors`). On a driver
    /// with it the runtime announces no `render-pass-split`, nor `memoryless-backed` for the
    /// split, where the interlock's attachments alone split a pass: the device's faster form.
    /// Elsewhere such a split is announced.
    InterlockedPassSplitNoSlower,
    /// A driver diagnostic (the module docs): the driver reports a warning as it creates a
    /// graphics pipeline that makes the primitive restart dynamic with its topology dynamic or a
    /// strip or fan, whatever restart the draws set later: a `VALIDATION` message named
    /// `mvk-warn` (number 0), `VK_ERROR_FEATURE_NOT_PRESENT: Metal does not support disabling
    /// primitive restart.` ([`DriverFact::pipeline_creation_diagnostics`]). MoltenVK reports it so
    /// wherever it is built without Metal's private API, Homebrew's build included
    /// (`MVKPipeline.mm` lines 881-893 at 1.4.1), because Metal restarts every strip. The runtime
    /// makes the restart dynamic only with the topology, and each draw sets the restart Metal
    /// applies (on for a strip, off for a list, which Metal never restarts), so the warning reports
    /// nothing the draws do: measured on the Apple M4 by `runtime_lane::restart` and
    /// `runtime_lane::made_at_creation` (at 41d0f054f, 25 reports, every one at
    /// such a pipeline's creation). On a driver with it the runtime withholds exactly that message
    /// from the device's message sink inside that pipeline's creation, and counts it
    /// (`WorkCounts::driver_diagnostics_withheld`); any other message, this one anywhere else, and
    /// this one from an unmeasured release relay as before. MoltenVK's own code on every GPU, so
    /// the row names no vendor; held on the release measured alone (the module docs).
    DynamicRestartWarns,
    /// A driver diagnostic (the module docs): the driver reports a warning as it creates a
    /// graphics pipeline for each color attachment whose format lacks
    /// `VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BLEND_BIT` in its optimal tiling, whether the
    /// pipeline enables blending there or not: a `VALIDATION` message named `mvk-warn` (number 0),
    /// `VK_ERROR_FEATURE_NOT_PRESENT: Blending is enabled for attachment with format <format>,
    /// which does not support it.`, the format by its registry name
    /// ([`DriverFact::pipeline_creation_diagnostics`]). MoltenVK 1.4.2 skips the blend state of
    /// such an attachment, whose Metal pixel format lacks `kMVKMTLFmtCapsBlend` (the capability it
    /// reports as that feature bit), and warns in that branch without reading the attachment's
    /// `blendEnable` (`MVKGraphicsPipeline::addFragmentOutputToPipeline`, `MVKPipeline.mm` lines
    /// 1961-1996 at 1.4.2). The runtime never enables blending there, as Vulkan forbids
    /// (`VUID-VkGraphicsPipelineCreateInfo-renderPass-06041`), so the warning reports nothing the
    /// pipeline does. On a driver with it the runtime withholds exactly those messages inside that
    /// pipeline's creation, and counts them, as [`DriverFact::DynamicRestartWarns`]'s.
    BlendlessColorAttachmentsWarn,
    /// A cost fact (the module docs): the driver runs `vkCmdFillBuffer` as a compute dispatch of
    /// a pipeline it compiles at the first fill each `VkCommandPool`'s command buffers encode,
    /// inside `vkQueueSubmit`, through the OS's shader compiler, which serves the process's
    /// compiles in order, so a submission waits behind every pipeline compile ahead of it. MoltenVK
    /// keeps the pipeline per pool (`MVKCommandEncodingPool::getCmdFillBufferMTLComputePipelineState`
    /// at 1.4.1). Measured on the Apple M4: behind a storm of pipeline creations, a commit holding
    /// a pool's first fill waited 250 to 330 ms in that submit (sampled). On
    /// a driver with it the runtime records each fill as transfers the driver compiles nothing
    /// for: `vkCmdUpdateBuffer` of the first words, then copies of the filled words onto the rest
    /// (`word_fill.rs`); elsewhere `vkCmdFillBuffer`. MoltenVK's own code on every GPU, so the row
    /// names no vendor; held on the release measured alone.
    FillCompiledPerCommandPool,
    /// A render pass's attachment stored `DONT_CARE` is stored as Metal's own `DontCare` on the
    /// same GPU, `MultisampleResolve` where it resolves in the subpass: MoltenVK hands the
    /// operation to Metal (`MVKAttachmentDescription::getMTLStoreAction` at 1.4.1) wherever it
    /// renders the attachment's whole framebuffer in its last subpass and does not restart the
    /// Metal pass inside the subpass, which it does at a barrier there whose destination reads
    /// input attachments, a tessellated draw, sample positions that change between draws, and an
    /// indirect draw of views; at each such point it stores the attachment instead. Apple's GPU
    /// keeps the attachment's earlier texels unless its tiler overflows, and a pass that clears it
    /// and draws nothing before `MultisampleResolve` leaves the clear where Metal compresses the
    /// texture. Vulkan leaves the attachment's contents undefined. On a driver with it the runtime
    /// announces no `dont-care-store` for a pass recorded as one render pass instance in which it
    /// records none of those (`Device::degrade_pass`). Held on the release measured alone (the
    /// module docs): MoltenVK's code decides it. Measured on the M4 and the M3 Max.
    DontCareStoreIsMetals,
    /// A cost fact (the module docs): the driver samples each `vkCmdWriteTimestamp` in a Metal
    /// blit pass of its own, which waits for the work before it and writes a byte of one buffer
    /// the whole device shares (`MVKCommandEncoder::encodeTimestampStageCounterSamples` at
    /// 1.4.1, on Apple's GPUs, which sample only at a pass's boundaries). Metal's hazard tracking
    /// orders every pass that writes that buffer, so a command buffer's timestamps wait for every
    /// queue's earlier timestamped work, where Vulkan orders nothing between queues: a 256-byte
    /// fill on a queue of another family completed only when a 27 ms dispatch did, in 37 to 40
    /// rounds of 40, and in about 0.25 ms without timestamps (an M4; our upstream draft). Each
    /// command buffer's two timestamps also cost it about 150 us of latency, against about 15 us
    /// for a bare submission (the runtime lane's `commit_costs`; on the consumer's M3 Max a
    /// one-fill command buffer completed 283 to 309 us after its commit without them, 356 to 399
    /// us with them). On a driver with it the runtime answers command buffers' GPU times from the
    /// host's clock and writes no timestamps (`gpu_time.rs`, announced once as
    /// `gpu-times-from-host`); elsewhere it writes them unless told otherwise. MoltenVK's own code
    /// on Apple's GPUs; held on the release measured alone.
    TimestampsWaitForEveryQueue,
    /// The driver decodes each texel of a BC1 to BC3 texture (`BC1_RGBA`, `BC2_RGBA`,
    /// `BC3_RGBA`) to Metal's value, and each texel of their sRGB forms to Metal's code before
    /// its sRGB step (which the translation takes through Metal's table,
    /// `metal::texel_format::srgb_raw`): Vulkan leaves the decode's arithmetic to the
    /// implementation. Metal's (an Apple M4, every texel of random blocks of the fourteen BC
    /// formats read): each endpoint widened by replicating its bits, the index's
    /// weight of 64 (BC7's: 21 and 43 for the thirds, 9 to 55 for BC3's alpha's sevenths, 13 to
    /// 51 for its fifths, 32 for BC1's half) summed as `(64 - w) a + w b` and rounded half up to
    /// 8 bits; BC2's alpha its 4-bit code times 17. NVIDIA's 615.71.09 (the RTX 3090 and the RTX
    /// 5080 Laptop GPU) answers channels up to 3/255 off (BC3's alpha 6/255), ANV up to 1/255,
    /// lavapipe 3/255. On a driver without it a texture of such a format announces
    /// `compressed-texel-decode` where the runtime makes it.
    ColorBlockDecodeIsMetals,
    /// The driver decodes each texel of a BC4 or BC5 texture (`BC4_RUnorm` to `BC5_RGSnorm`) to
    /// Metal's value. Metal's (an Apple M4): the index's weight of 64 as for
    /// BC3's alpha, the sum `(64 - w) a + w b` unrounded, `sum / 16320` (signed `sum / 8128`,
    /// an endpoint of -128 summed as it is and the value clamped to -1 after), the six-value
    /// constants 0 and 1 (signed -1 and 1), each correctly rounded to `float`. NVIDIA's 615.71.09
    /// answers up to 7.3/255 off (signed 0.064), ANV 1.3/255 (signed 0.013), lavapipe 2.3/255
    /// (signed 0.021), and RADV (Mesa 26.2.4, the Radeon 780M) clamps an endpoint of -128 to -127
    /// before the sum (signed up to 1.7/255 off) and rounds some unsigned values a `float` ulp
    /// off. On a driver without it a texture of such a format announces
    /// `compressed-texel-decode` where the runtime makes it.
    ChannelBlockDecodeIsMetals,
    /// A depth image a pass rendered at custom sample positions (made with
    /// `VK_IMAGE_CREATE_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT`, held in `GENERAL`) answers a
    /// shader's read of each sample with the depth at the standard positions, and a later pass
    /// at other positions that loads it compares and keeps the depth at its own positions: the
    /// driver keeps the triangle's depth as its plane and evaluates the plane where the reader's
    /// positions lie. The pass's own depth test compares at the custom positions, and a copy
    /// (`vkCmdCopyImage`) recorded after the pass keeps each sample's depth at them. Vulkan's
    /// image in `GENERAL` keeps what the pass stored (a layout transition alone takes the
    /// locations, `VkSampleLocationsInfoEXT`), and Metal keeps each sample's depth at its
    /// position. Transitions to and from `DEPTH_STENCIL_READ_ONLY_OPTIMAL`,
    /// `SHADER_READ_ONLY_OPTIMAL`, `TRANSFER_SRC_OPTIMAL` and `DEPTH_STENCIL_ATTACHMENT_OPTIMAL`
    /// right after the pass, with and without the locations, change none of it. On a driver with
    /// it a depth/stencil attachment at custom positions renders through the stand-in, made with
    /// the flag, copied in where the pass loads it and back where it stores it
    /// (`attachment_stand_in.rs`), announced as `positions-through-scratch`; elsewhere its own
    /// image renders them.
    PositionedDepthReadAtStandardPositions,
    /// The depth test compares a denormal a `D32_SFLOAT` image holds as Metal's does: a positive
    /// one above zero, a negative one not below zero (a fragment at depth 0 passes `Less`
    /// against a stored positive denormal and fails `Greater` against a stored negative one).
    /// Vulkan states no such rule: NVIDIA's, ANV's and lavapipe's depth tests compare a stored
    /// positive denormal as zero, and RADV's a stored negative one (and a stored -0) below zero.
    /// On a driver without it a write that may store a denormal in a 32-bit float depth image
    /// announces `depth-denormals`.
    DepthTestKeepsDenormals,
    /// The depth test compares a fragment's depth with the value a `D32_SFLOAT` image holds as
    /// Metal's does: by magnitude, as the integers of their low 31 bits, so the sign is dropped
    /// and a NaN orders above +inf (a fragment at 0.5 passes `Equal` against a stored -0.5).
    /// Vulkan compares the stored value as it is. On a driver whose depth test is a Metal GPU's
    /// own (MoltenVK) it holds by construction: the pass's depth test is Metal's test of a
    /// `Depth32Float` texture on that GPU, which gives that GPU's Metal answer, and its pass
    /// stores what Metal's stores (a pass with depth writes stores each texel it leaves without
    /// its sign) wherever a draw hands it a state with depth writes. On a driver with it no render
    /// target's depth renders over a magnitude stand-in, a single-sample `Depth32Float` one is
    /// stripped of its signs after a pass that sets such a state but draws under none, and a write
    /// that may store a negative value or a NaN announces nothing (`texture/signed_depth.rs`);
    /// elsewhere a single-sample `Depth32Float` target renders over the stand-in, and a
    /// multisampled or combined depth/stencil one announces `signed-depth-test`.
    DepthTestComparesMagnitudes,
    /// The device's first sparse resource's creation (`vkCreateImage` or `vkCreateBuffer` with
    /// `SPARSE_BINDING`) waits for the work submitted before it on every queue: ANV through
    /// TR-TT (Intel's `i915` kernel driver) sets its translation tables up there with a batch on
    /// each queue and waits for them (`anv_trtt_first_bind_init`, Mesa 26.2.4), and each batch
    /// runs after its queue's earlier work. Vulkan orders no creation with queue work, and Metal
    /// makes a sparse texture without waiting for the GPU. On a driver with it the runtime
    /// announces `first-sparse-resource-wait` where it makes the device's first sparse texture. A
    /// cost fact: held on the releases measured alone.
    FirstSparseResourceWaitsForQueues,
    /// The driver decodes each texel of a BC6H texture (`BC6H_RGBFloat`, `BC6H_RGBUfloat`) to
    /// Metal's value. Direct3D specifies BC6H's decode bit for bit, but Vulkan leaves a
    /// compressed format's decode to the implementation, so it is measured, not assumed. On a
    /// driver without it a texture of such a format announces `compressed-texel-decode` where
    /// the runtime makes it: the runtime has no decoded twin of BC6H.
    Bc6hBlockDecodeIsMetals,
    /// The driver decodes each texel of a BC7 texture (`BC7_RGBAUnorm`, and `BC7_RGBAUnorm_sRGB`
    /// through its unorm view and Metal's table, `metal::texel_format::srgb_raw`) to Metal's
    /// value. lavapipe (Mesa 26.0.8) answers some a `float` ulp off. On a driver without it a
    /// texture of such a format announces `compressed-texel-decode` where the runtime makes it:
    /// the runtime has no decoded twin of BC7.
    Bc7BlockDecodeIsMetals,
    /// Each timestamp query pool the driver makes is one Metal counter sample buffer
    /// (`MVKTimestampQueryPool`, `MVKQueryPool.mm` at 1.4.1), of at most 4096 timestamps, and
    /// Metal keeps at most 32 of those live per GPU (through MoltenVK 1.4.1 on an M3 Max and an
    /// M4: the 33rd pool failed, as did a pool of 8192), so every pool a device holds is one
    /// fewer for another device on the GPU. Vulkan sets no such limit. A defect fact: on a driver
    /// with it the device makes one pool of 4096 timestamps, never more, where elsewhere it adds
    /// a pool as its command buffers in flight fill the ones it has (`gpu_time.rs`).
    TimestampPoolsAreCounterSampleBuffers,
    /// A render pass's attachment stored `DONT_CARE` keeps, at each texel, either its texel from
    /// before the pass or the texel the same pass stores with `STORE`: one of the two answers
    /// Metal's `DontCare` and `MultisampleResolve` give on Apple's GPUs (the earlier texel, or
    /// the stored one where its tiler overflows), never a third. Vulkan leaves the contents
    /// undefined, and NVIDIA's 615.71.09 on the RTX 3080 left the pass's clear color in 1984 of
    /// 16384 texels of a pass that cleared, drew over them and stored `DONT_CARE`. On a driver with
    /// it, or with [`DriverFact::DontCareStoreIsMetals`], the runtime hands Metal's `DontCare` to
    /// the driver as `DONT_CARE`; on every other it stores the attachment with `STORE`, the stored
    /// answer (`render_target.rs`), and both announce `dont-care-store`.
    DontCareStoreIsOneOfMetals,
    /// A cost fact (the module docs): the driver runs `vkCmdCopyImageToBuffer` from an optimally tiled
    /// color image into a buffer's rows slower than a compute dispatch that fetches the texels through
    /// an unsigned-integer view of the image and stores their words. Measured on ANV, Mesa 26.2.4 on
    /// the Iris Xe of RPL-P (2026-10-15, a probe of one pass a frame over a
    /// 1307x512 `BGRA8Unorm` tile over guest pages with rows of 5248 bytes, 60 frames, three rounds
    /// within 3 us of each other): a pass that loads, draws and stores took 1319 us of GPU time with
    /// the copy back and 972 us with the dispatch, one half-scissored 742 and 575 us, and one that
    /// stores nothing (the mutable shadow's cost alone) 500 and 513 us. On the UHD 770 of RPL-S (a
    /// pass a frame over a 1920x1080 `BGRA8Unorm` shadow over anonymous pages, six rounds, GPU time
    /// the rounds' median), a pass that loads, draws and stores took 1.68 ms with the dispatches
    /// against 1.81 ms with the copies, one that clears 1.04 against 1.14 ms, and one drawn in a
    /// centred region of half the width and height 446 against 476 us. The driver's copy cost, so
    /// the row names no GPU: it holds on each measured identity of ANV at the release. On a driver
    /// with it a shadowed texture's copy back is such a dispatch (`shadow_copy_back.rs` in the
    /// runtime), and so is a merge's copy of the shadow into its scratch before the commands
    /// (`shadow_merge.rs`), its shadow made `MUTABLE_FORMAT` with that view, through which the
    /// merge reads the texels the commands left: the same bytes either way, as Vulkan states a
    /// compatible view reads a texel's bits unchanged. Measured no faster, so not listed: RADV,
    /// Mesa 26.2.4 on the Radeon 780M (whose passes over a mutable shadow took 84 to 150 us longer
    /// a frame) and 26.0.8 on the Radeon 780M and the Radeon RX 7900 XTX (where a merge reading a
    /// mutable shadow's texels through the view in place of a copy of them saved 6% and 2% of a
    /// 1920x1080 pass that loads, and a pass that clears took 75% and 0% more), and NVIDIA's
    /// 615.71.09 on the RTX 5080 Laptop GPU.
    ShadowCopyBackDispatchFaster,
    /// A cost fact (the module docs), on a driver with [`DriverFact::ShadowCopyBackDispatchFaster`]:
    /// the GPU runs a merged shadowed texture's copy in (`vkCmdCopyBufferToImage`) and its keep (the
    /// copy back's dispatch into the merge's scratch) slower than one dispatch that reads the
    /// texels' words from the buffer once and stores them in the shadow, through a storage view of
    /// it, and in the scratch, with the shadow's 4-byte texels viewed as `R8G8B8A8_UINT` (whose
    /// storage usage keeps the image compressed on ANV, where `R32_UINT` beside storage usage makes
    /// gfx12.0 drop its aux surface). Measured on ANV, Mesa 26.2.4 (one pass a frame, or twelve in
    /// one command buffer, over a 1920x1080 `BGRA8Unorm` shadow over anonymous pages, 1000 frames,
    /// GPU time a frame, two rounds each): on the UHD 770 of RPL-S, a pass that loads, draws and
    /// stores took 1427 us against 1643 to 1665 us with the copy back's fact alone, twelve such
    /// passes 17.2 ms against 20.0 ms, and one that clears 891 against 1042 us. On the Iris Xe of
    /// RPL-P it was measured no faster, so not listed: twelve such passes took 9.3 to 15.8 ms
    /// against 9.3 to 15.3 ms with the copy back's fact alone, slower in six of seven alternating
    /// rounds (median 5% more), one pass 3.05 against 3.48 ms, one that clears 0.40 to 1.29
    /// against 0.61 to 1.20 ms; and the `R8G8B8A8_UINT` view without the dispatch took longer
    /// still there (twelve passes 10.0 to 13.3 ms, a clearing pass 0.89 to 1.67 ms). On a device
    /// with it a merged texture's copy in and keep of the same rectangle are that dispatch
    /// (`shadow_copy_in.rs` in the runtime), and its shadow's view also stores (`shadow_copy_back.rs`
    /// in the runtime): the same bytes either way, as Vulkan states a compatible view reads and
    /// stores a texel's bits unchanged.
    ShadowCopyInDispatchFaster,
    /// A cost fact (the module docs): the driver records each region of a `vkCmdCopyBuffer` as an
    /// operation of its own, so a copy of equal rows between two buffers whose rows lie apart costs
    /// the recording CPU and the GPU more than one compute dispatch that stores the rows' words.
    /// Measured on ANV, Mesa 26.2.4 (`row_copies::row_copies_cost` in the validation crate's runtime
    /// lane, one copy a frame of 2 to 1080 rows of 7680 bytes into rows 8192 bytes apart, two rounds
    /// each, CPU time the replaying and encoding threads' together): on the Iris Xe of RPL-P, 1080
    /// rows took 3.1-3.3 ms of CPU and 5.3-7.8 ms of GPU time as regions against 0.28-0.40 ms and
    /// 1.13 ms as the dispatch, 16 rows 0.24-0.27 ms and 97-100 us against 0.12-0.17 ms and 30 us,
    /// and 2 rows the same CPU time and 21.9 us against 14.8 us; on the UHD 770 of RPL-S, 1080 rows
    /// 0.77-0.83 ms and 3.3-5.3 ms against 0.04-0.11 ms and 0.59-0.91 ms, and 2 rows the same CPU
    /// time and 8.8-14.2 us against 3.1-4.3 us. ANV's per-region cost, so its row names no GPU: it
    /// holds on each measured identity of ANV at the release. NVIDIA's 615.71.09 on the RTX 5080
    /// Laptop GPU, the GPU its row names (the same probe, two rounds each): 1080 rows took
    /// 0.19-0.21 ms of CPU and 2.03-2.05 ms of GPU time as regions against 0.12-0.13 ms and
    /// 0.42-0.43 ms as the dispatch, 16 rows 34-40 us of GPU time against 13 us, and 2 rows the
    /// same CPU time and 9.1-9.8 us against 8.7-8.8 us. On a driver with it a copy of two or more
    /// such rows is the dispatch (`row_copies.rs` in the runtime): the same bytes either way.
    /// Measured no faster, so not listed: RADV, Mesa 26.2.4 on the Radeon 780M and the Radeon RX
    /// 7900 XTX, where at 1080 rows the dispatch took 5% and 3% more GPU time.
    RowCopiesDispatchFaster,
    /// A cost fact (the module docs): the driver loads a shader's uniform read at a constant
    /// offset ahead of the shader, so a key that reads a buffer slot's uniform view at its bind
    /// offset alone (`SpecKey::RangeAtBindOffset` 1) is worth the second `VkPipeline` a state
    /// pays for it at its first draw, beside the residual form its creation made. On a driver
    /// without it every key is its residual form, which reads the same bytes at every bind offset:
    /// a state's draws at any offset take the one pipeline its creation made
    /// (`pipeline/render/residual.rs` in the runtime), and nothing an answer depends on differs,
    /// only when the driver is called.
    ///
    /// **No driver is measured with it.** The behaviour is ANV's by report -- Mesa's ANV loads a
    /// uniform read at a constant offset ahead of the shader -- with no figure taken for it, on a
    /// venue no measurement here could reach, so it carries no row and no driver takes it. A row
    /// of ANV's, with its releases and its numbers, is how it would be taken later.
    ///
    /// Measured without it, so not listed, on `blend-sweep-write-masks` (32 render pipeline
    /// states, 32 draws, `conformance cost --alone`), the second form counted and then its
    /// absence timed in alternating rounds: llvmpipe (LLVM 23.1.1, Mesa 26.2.4) made 64
    /// `VkPipeline`s against 32, at 100.5, 102.5 and 107.6 ms against 98.0, 96.3 and 103.1;
    /// RADV, Mesa 26.2.4 on the Radeon 780M, 64 against 32 at 77.5 and 77.3 ms against 76.0 and
    /// 74.5, and on the Radeon RX 7900 XTX 64 against 32; NVIDIA 615.71.09 on the RTX 5080 Laptop
    /// GPU, 32 counted with 32 more still in the device's pipeline pool against 32 with none, at
    /// 94.1, 85.7 and 84.7 ms against 92.3, 85.5 and 86.1. MoltenVK 1.4.2 on the Apple M3 Max
    /// makes the one form either way, measured alike on 1.4.1 (the host benchmark's GPU workload,
    /// two runs of each form: a heavy fragment 20.74 against 20.73 ms, a heavy kernel 19.18
    /// against 19.47, a constant-buffer kernel 8.42 against 7.69) while each first draw of a key
    /// with a 1 made a second `VkPipeline` there (Metal's specialization and pipeline compile,
    /// 5.6 to 24.5 ms).
    ConstantOffsetUniformReadsPreloaded,
    /// The driver answers a format's linear-tiling attachment features from memory it has not
    /// yet initialized, so they differ between instances of one process: MoltenVK 1.4.1 builds
    /// its format table in `MVKPhysicalDevice`'s member initializers, where
    /// `MVKPixelFormats::setFormatProperties` reads `renderLinearTextures`, before the
    /// constructor's `initMetalFeatures()` sets it, and so keeps or drops `COLOR_ATTACHMENT`,
    /// `COLOR_ATTACHMENT_BLEND`, `BLIT_DST` and `DEPTH_STENCIL_ATTACHMENT` in every format's
    /// `linearTilingFeatures` (and with them its linear images' attachment usage) by that byte.
    /// Vulkan holds a linear image's views and blits to the features its tiling reports, which
    /// may be either answer, so a description of the device leaves those features out of every
    /// format's linear tiling ([`DriverFact::unreliable_linear_tiling`],
    /// `DeviceDescription::query`), and the runtime renders and blits into a texture over a buffer
    /// as on a driver whose linear tiling lacks them. A defect fact: MoltenVK takes it at every
    /// release and on every GPU.
    LinearTilingFeaturesUninitialized,
    /// A cost fact (the module docs): about once in every thousand semaphore operations its
    /// queue submissions carry, waits and signals alike, the driver spends about 2 ms of the
    /// submitting thread's CPU inside one `vkQueueSubmit`, the same count in every run. Measured
    /// with a probe outside the runtime (frames of 67 submissions of one command buffer, each
    /// signalling a timeline that another thread waits on, 300 frames a mode): 17 such
    /// submissions in 20100 that only signal, 43 with a wait on that timeline each, 38 with a
    /// wait on another, 58 with both, 1.8 to 2.7 ms each, on the RTX 5080 Laptop GPU in each of 3
    /// runs and on the RTX 3090 alike. Mesa's ANV and RADV showed none. On a driver with it each
    /// batch with work starts with a command buffer of barriers that makes what completed work
    /// wrote visible, and waits for no timeline the host saw complete; elsewhere it waits for
    /// those timelines instead, which saves that command buffer (about 12 us of an Iris Xe's GPU
    /// a batch) for about 1 us of CPU a wait (`command_buffer/submit.rs` in the runtime). Held on
    /// the GPUs and release measured alone.
    SemaphoreOperationsStallSubmits,
    /// A cost fact (the module docs): the driver keeps a single-sample depth image's
    /// hierarchical depth, with which its depth test rejects hidden fragments a block at a time,
    /// in the `GENERAL` layout where the image has sampled usage, and drops it there, with the
    /// depth's compression, where the image has transfer-source usage without it. Every image of
    /// the runtime's rests in `GENERAL` and takes transfer usage, as Metal copies every texture,
    /// so on a driver with it a single-sample depth or depth/stencil attachment the runtime makes
    /// takes sampled usage too, where the device samples its format, whatever it costs the image
    /// (`image_plan::hiz_candidate` in the runtime): the same texels either way. A device's own
    /// texture takes that usage already wherever it costs the image nothing (the runtime's
    /// undeclared usage), as it did on the Iris Xe at each of 75 sizes from 256x256 to 4096x4096
    /// in `D16_UNORM`, `D32_SFLOAT` and `D32_SFLOAT_S8_UINT`; the fact reaches the images that
    /// take only their declared usage, a heap's textures among them. ANV on Intel's Xe-LP GPUs
    /// gives such an image write-through hierarchical depth (`ISL_AUX_USAGE_HIZ_CCS_WT`,
    /// `want_hiz_wt_for_image` in Mesa 26.2.4's `anv_image.c`), which `anv_layout_to_aux_state`
    /// keeps in `GENERAL`; without sampled usage the image takes `ISL_AUX_USAGE_HIZ_CCS`, which
    /// that function turns off in `GENERAL` on such a GPU for an image of transfer-source usage.
    /// Measured on ANV, Mesa 26.2.4 on
    /// the Iris Xe of RPL-P, with a probe outside the runtime (a 1920x1080 `D32_SFLOAT` attachment
    /// in `GENERAL` cleared to 1.0, fullscreen triangles front to back under `LESS` with writes,
    /// the GPU's time a frame, medians of 50 frames): 32 layers 2.822 ms without sampled usage
    /// against 0.475 ms with it, a clear and one layer 0.233 against 0.120 ms; `D32_SFLOAT_S8_UINT`
    /// 2.838 against 0.582 ms and `D16_UNORM` 1.623 against 0.451 ms at 32 layers; the image's
    /// size the same (9432960 bytes). Through the runtime (`runtime_lane::depth_costs` in the
    /// validation crate, the same frames in `Depth32Float`, the depth a heap's texture): 4.72 to
    /// 5.66 ms without the fact against 0.82 to 0.87 ms with it at 32 layers, 0.51 against 0.24
    /// to 0.25 ms at one; a device's own texture alike either way. Measured no faster, so not
    /// listed: NVIDIA's 615.71.09 on the RTX 3090 (0.058 ms either way at 32 layers, 0.017 at
    /// one).
    SampledDepthKeepsHiZ,
    /// A cost fact (the module docs): the driver renders into a multisampled depth image much
    /// faster in `DEPTH_STENCIL_ATTACHMENT_OPTIMAL` than in `GENERAL`, the moves from `GENERAL`
    /// before the render pass instance and back after it included. ANV on Intel's Xe-LP GPUs
    /// gives such an image hierarchical depth (`ISL_AUX_USAGE_HIZ_CCS`; the write-through form
    /// that keeps it in `GENERAL` is single-sample only, `want_hiz_wt_for_image` in Mesa 26.2.4's
    /// `anv_image.c`), unused in `GENERAL`: there its depth test rejects no fragment a block at a
    /// time. The move in costs a hierarchical depth ambiguate, the move back a resolve. On a
    /// driver with it the runtime holds a multisampled depth or depth/stencil attachment that is
    /// not transient in the attachment layout for each render pass instance, moved there just
    /// before the instance begins and back just after it ends (`render_target::depth_layout` in
    /// the runtime): the same texels, as a layout transition preserves them. A device's own
    /// `Private` such texture rests in that layout between the passes of render encoders that
    /// touch it only as their attachment, moved back to `GENERAL` only before an encoder that
    /// names it otherwise (`texture/held_depth.rs` in the runtime), so such passes spend neither
    /// move. Measured on ANV, Mesa
    /// 26.2.4 on the Iris Xe of RPL-P, with a probe outside the runtime (a 1920x1080 attachment of
    /// 4 samples cleared to 1.0, fullscreen triangles front to back under `LESS` with writes, the
    /// GPU's time a frame, medians of 50 frames, two rounds alike), in `GENERAL` against moved:
    /// `D32_SFLOAT` 12.25 against 1.44 ms at 32 layers and 1.60 against 1.10 ms at one,
    /// `D32_SFLOAT_S8_UINT` 12.38 against 1.60 and 1.73 against 1.28 ms, `D16_UNORM` 6.12 against
    /// 0.83 and 0.46 against 0.48 ms; in `GENERAL` alike with transfer-source usage or without.
    /// Through the runtime (`runtime_lane::depth_costs` in the validation crate, the same frames
    /// in `Depth32Float`, a device's own texture and a heap's alike): 12.14 against 1.44 to 1.52
    /// ms at 32 layers, 1.54 against 1.08 to 1.12 ms at one. Held between frames against moved
    /// around each instance, the GPU at 300 MHz throughout (`gt_cur_freq_mhz`; the moved form
    /// both at the commit before and, alternating with it in one session, in a heap's texture or
    /// an encoder that also binds the depth): 2.38 against 4.34 to 4.37 ms at 32 layers, 0.62
    /// against 2.56 to 2.58 ms at one. Measured no faster, so not listed: NVIDIA's 615.71.09 on
    /// the RTX 3090 (0.112 against 0.113 ms at 32 layers, 0.028 against 0.030 at one).
    MultisampledDepthFasterOutOfGeneral,
    /// A render pass that loads a single-sampled colour attachment of a signed normalized format
    /// or of `E5B9G9R9_UFLOAT_PACK32` and draws stores each 32x32 tile a draw reaches back through
    /// its texels' values, as Metal's pass does, the texels no fragment writes too: a channel's
    /// minimum code (-128, or -32768), which reads as -1.0 as the code above it does, is stored as
    /// that code (-127, -32767), and a shared-exponent word whose largest mantissa is below 256
    /// under an exponent above 0 is stored normalized. Vulkan's load and store keep the bytes no
    /// draw writes. MoltenVK records the render pass as a Metal render pass whose attachments
    /// Metal's own load and store actions load and store
    /// (`MVKAttachmentDescription::getMTLStoreAction` at 1.4.1), so the GPU's tile store is
    /// Metal's by construction. On a driver with it the runtime announces no `snorm-minimum-kept`
    /// or `shared-exponent-kept`, and its exact tile store (`TileStorePolicy::Exact`) adds no load
    /// pass for such an attachment.
    TileStoreIsMetals,
    /// A buffer bound to memory the driver imported from a host pointer
    /// (`VK_EXT_external_memory_host`) reads and writes, at each submission, the pages mapped at
    /// that pointer then: pages the process maps over the address after the import (`mmap` with
    /// `MAP_FIXED`) take the place of those it was made over, for the GPU as for the CPU. Vulkan
    /// states nothing of a remap under a live import, and Metal's no-copy buffer keeps the pages
    /// wired when it was made. ANV on i915 imports a pointer as a userptr object, whose pages the
    /// kernel gets again at the next execbuffer once a remap invalidated them (Linux 7.2.9's
    /// `i915_gem_object_userptr_submit_init`, or at its first execbuffer, before which nothing was
    /// pinned); RADV imports it as an amdgpu userptr object, whose pages the kernel gets again in
    /// each command submission once a remap invalidated them (Linux 7.2.9's
    /// `amdgpu_cs_parser_bos`, through `amdgpu_ttm_tt_get_user_pages`), and refuses a shared
    /// `memfd`'s pointer where it imports private anonymous pages'; lavapipe reads the memory
    /// through the pointer itself (`lvp_AllocateMemory`'s `LVP_DEVICE_MEMORY_TYPE_USER_PTR`). On a
    /// driver with it a no-copy buffer's own pointer import is released while idle and imported
    /// again at the same pointer at its next blit, which reads the pages the kept import would
    /// read (`host_residency.rs` in the runtime): on ANV on i915 every execbuffer lists every
    /// memory of the device, an import of a pointer costing about 0.16 us of the kernel's time in
    /// each, and on RADV amdgpu gets every page of each live pointer import again in each
    /// submission: about 0.5 us an import and 29 ns a 4 KiB page on the RX 7900 XTX under Linux
    /// 7.0.0, about a quarter of that a page on the Radeon 780M under Linux 7.2.9. Elsewhere such
    /// an import stays imported, as the driver may keep the pages it was made over.
    PointerImportsFollowTheirAddress,
    /// Consecutive draws of a subpass whose fragment shaders read and write a storage image in
    /// the ordered pixel interlock, every access coherent, are ordered by the interlock alone, as
    /// Vulkan states: each fragment's interlocked operations happen before those of a later draw's
    /// fragment of the same pixel ("Interlocked Operations", rasterization order following
    /// primitive order through the subpass), so a draw reads what the draw before it wrote with no
    /// pipeline barrier between them. On a driver with it the runtime records no barrier between
    /// two draws of a render pass instance where the later one reaches a color in the fetch's
    /// interlocked storage form (or an extended-range color's `half` shadow) that the earlier one
    /// wrote (`render.rs`'s `InterlockedStorage`); elsewhere it records one from the fragment
    /// shader's writes to its reads and writes, which RADV on the Radeon 780M needs (a later draw
    /// read texels from before an earlier one's writes without it).
    InterlockOrdersDraws,
    /// A single-sample colour image bound to memory lies in it as Metal lays out a heap texture
    /// of the image's usage there: a buffer over the memory meets each texel at the byte Metal's
    /// tiles place it. Vulkan leaves an optimal image's layout to the driver. On a driver without
    /// it the runtime announces `heap-texture-untiled-layout` for a texture of a heap over the
    /// device's memory whose image lies in the heap's bytes (`texture/heap.rs`); on one with it,
    /// only where Metal's layout is not the image's (a texture Metal may keep losslessly
    /// compressed, `heap-texture-lossless-layout`; a multisample one, a layout not measured).
    ColorImagesLieAsMetalHeapTextures,
    /// A multisample colour image bound to memory lies in it as Metal lays out a multisample heap
    /// texture it keeps uncompressed: a buffer over the memory meets sample `s` of pixel `(x, y)`
    /// at the byte Metal's tiles of `samples · B`-byte elements place the pixel, plus `s · B`
    /// (`B` the format's bytes). On a driver without it the runtime announces
    /// `heap-texture-untiled-layout` for a multisample colour texture of a heap over the device's
    /// memory whose image lies in the heap's bytes (`texture/heap.rs`); on one with it, only
    /// where Metal's layout is not the image's (a texture Metal keeps losslessly compressed,
    /// `heap-texture-lossless-layout`; a layout not measured).
    MultisampleColorImagesLieAsMetalHeapTextures,
}

impl DriverFact {
    /// Whether the fact is a defect the runtime works around (the module docs): the driver does
    /// otherwise than Metal, so a device of the driver whose identity was not measured takes the
    /// workaround, never the path a measurement has not cleared. Its depth and stencil images lie
    /// apart from their memory or keep their GPU-optimized layout (MoltenVK), its compressed
    /// images hide aliased writes (ANV), a depth rendered at custom positions reads at the
    /// standard ones (NVIDIA's), its timestamp pools are Metal's scarce counter sample buffers
    /// (MoltenVK), or its linear-tiling attachment features are read uninitialized (MoltenVK).
    /// Every other fact is an answer fact (Metal's answer, a cost or a diagnostic), taken on a
    /// measured identity alone.
    pub const fn assumed_where_unmeasured(self) -> bool {
        matches!(
            self,
            Self::DepthStencilImagesLieApart
                | Self::DepthStencilImagesStayGpuOptimized
                | Self::CompressedImagesHideAliasedWrites
                | Self::PositionedDepthReadAtStandardPositions
                | Self::TimestampPoolsAreCounterSampleBuffers
                | Self::LinearTilingFeaturesUninitialized
        )
    }

    /// The linear-tiling format features a driver with this fact may report or not
    /// ([`DriverFact::LinearTilingFeaturesUninitialized`]: the color attachment and its blending,
    /// the blit destination and the depth/stencil attachment), which a description of its device
    /// leaves out of every format's linear tiling
    /// ([`DeviceDescription::unreliable_linear_tiling`]). Empty for every other fact.
    pub const fn unreliable_linear_tiling(self) -> FormatFeatureFlags2 {
        match self {
            Self::LinearTilingFeaturesUninitialized => FormatFeatureFlags2::COLOR_ATTACHMENT
                .union(FormatFeatureFlags2::COLOR_ATTACHMENT_BLEND)
                .union(FormatFeatureFlags2::BLIT_DST)
                .union(FormatFeatureFlags2::DEPTH_STENCIL_ATTACHMENT),
            _ => FormatFeatureFlags2::empty(),
        }
    }

    /// The diagnostics a driver with this fact reports as it creates a graphics pipeline that
    /// makes `dynamic_states` dynamic, of the input assembly topology `topology`, whose color
    /// attachments of the formats `blendless_colors` lack `COLOR_ATTACHMENT_BLEND` in their
    /// optimal tiling: for [`DriverFact::DynamicRestartWarns`], one where the pipeline makes
    /// `PRIMITIVE_RESTART_ENABLE` dynamic and makes `PRIMITIVE_TOPOLOGY` dynamic or takes a
    /// strip or a fan; for [`DriverFact::BlendlessColorAttachmentsWarn`], one for each format of
    /// `blendless_colors` the registry names. None for any other pipeline, and for every other
    /// fact.
    pub fn pipeline_creation_diagnostics(
        self,
        dynamic_states: &[DynamicState],
        topology: PrimitiveTopology,
        blendless_colors: &[Format],
    ) -> Vec<DriverDiagnostic> {
        use PrimitiveTopology as T;
        match self {
            Self::DynamicRestartWarns => {
                let strip = matches!(
                    topology,
                    T::LINE_STRIP | T::TRIANGLE_STRIP | T::TRIANGLE_FAN
                );
                (dynamic_states.contains(&DynamicState::PRIMITIVE_RESTART_ENABLE)
                    && (strip || dynamic_states.contains(&DynamicState::PRIMITIVE_TOPOLOGY)))
                .then_some(DYNAMIC_RESTART_WARNING)
                .into_iter()
                .collect()
            }
            Self::BlendlessColorAttachmentsWarn => blendless_colors
                .iter()
                .filter_map(|format| format.name())
                .map(|name| DriverDiagnostic {
                    id_name: MVK_WARN,
                    id: 0,
                    text: Cow::Owned(format!(
                        "VK_ERROR_FEATURE_NOT_PRESENT: Blending is enabled for attachment with \
                         format {name}, which does not support it."
                    )),
                })
                .collect(),
            _ => Vec::new(),
        }
    }
}

/// A message a driver reports through `VK_EXT_debug_utils` where Vulkan states nothing wrong, at
/// `WARNING` severity as a `VALIDATION` message, as a driver diagnostic fact measures it (the
/// module docs; [`DriverFact::pipeline_creation_diagnostics`]). The runtime withholds it from the
/// device's message sink only inside the call the fact names, on a release measured with the
/// fact, where it matches every field exactly; anywhere else it relays.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DriverDiagnostic {
    /// Its `pMessageIdName`.
    pub id_name: &'static str,
    /// Its `messageIdNumber`.
    pub id: i32,
    /// Its `pMessage`.
    pub text: Cow<'static, str>,
}

/// The name MoltenVK gives a warning: `MVKInstance::debugReportMessage` names a warning
/// `mvk-warn` and numbers it 0, and `MVKBaseObject::reportResult` prefixes the result's name to
/// its text.
const MVK_WARN: &str = "mvk-warn";

/// MoltenVK's report of a pipeline whose primitive restart may be disabled
/// ([`DriverFact::DynamicRestartWarns`]).
const DYNAMIC_RESTART_WARNING: DriverDiagnostic = DriverDiagnostic {
    id_name: MVK_WARN,
    id: 0,
    text: Cow::Borrowed(
        "VK_ERROR_FEATURE_NOT_PRESENT: Metal does not support disabling primitive restart.",
    ),
};

/// Every measured fact, by driver.
///
/// `RobustIndexFetchReadsZero`, measured by `runtime_lane::index_fetches` (at
/// e89919c3, 2026-10-06): indexed indirect draws past their index buffer, from a buffer placed in
/// a heap and from one of its own, drew Metal's vertices under a range of a size
/// (`vkCmdBindIndexBuffer2`) and past the end of a copy's `VkBuffer` (`maintenance5` withheld):
/// - RADV (`MESA_RADV`) from Mesa 26.2.4: the AMD Radeon 780M (RADV PHOENIX), Mesa
///   26.2.4-arch3.1; and the AMD Radeon RX 7900 XTX (RADV NAVI31), Mesa 26.2.4-arch3.1, by the
///   same lane with the fact taken (2026-10-22, its identity measured).
/// - ANV (`INTEL_OPEN_SOURCE_MESA`) from Mesa 26.2.4: Intel Graphics (RPL-S), Mesa
///   26.2.4-arch3.1.
/// - NVIDIA's (`NVIDIA_PROPRIETARY`) from 615.71.09: the GeForce RTX 5080 Laptop GPU, 615.71.09,.
///
/// Not listed: lavapipe (`MESA_LLVMPIPE`), whose indirect draws read zero there but whose direct
/// draws under a range of a size drew no vertex there (x86, Mesa 26.2.3 with LLVM 23.1.1), so
/// its fetch past the range has no one answer; MoltenVK, which enables no
/// `robustBufferAccess2`; and every driver and release not measured.
///
/// `DepthImageKeepsDenormals` and `DepthImageKeepsValuesOutsideUnitRange`, measured by
/// `runtime_lane::texture_blits::depth_images_keep_what_their_drivers_claim` (at
/// d0c9878a, 2026-10-06): 11 denormals of both signs and 5 values outside `[0, 1]` through
/// `replace_region` into a render target's `Depth32Float` depth image, read back by `get_bytes`:
/// - NVIDIA's from 615.71.09 keeps every denormal, with `VK_EXT_depth_range_unrestricted`: the
///   GeForce RTX 3090 and the RTX 5080 Laptop GPU, both 615.71.09.
/// - lavapipe from Mesa 26.0.8 keeps every denormal, with the extension: arm64, Mesa 26.0.8 with
///   LLVM 21.1.8, in `scripts/linux-gate` on an M4 mini; x86, Mesa 26.2.4
///   with LLVM 23.1.1, on an Iris Xe (RPL-P) host.
/// - MoltenVK from 1.4.1 (`driverVersion` 10401) keeps both, without the extension: an Apple M4
///   mini and the M3 Max.
///
/// Measured to keep neither, so not listed: RADV (`MESA_RADV`), Mesa 26.2.4 on the Radeon 780M
/// (no denormal; every value outside `[0, 1]` under the extension), and ANV
/// (`INTEL_OPEN_SOURCE_MESA`), Mesa 26.2.4 on Intel Graphics (RPL-S) and the Iris Xe (RPL-P)
/// without the extension (no denormal, every value outside `[0, 1]` clamped). A driver with the
/// extension is not measured for `DepthImageKeepsValuesOutsideUnitRange`, which only a device
/// without it reads.
///
/// `DepthImageKeepsDenormals` again and `DepthTestKeepsDenormals`, measured by
/// `runtime_lane::depth_denormals` (2026-10-10): the smallest and largest denormal of each sign,
/// beside the smallest normal and 0.5, stored into a 4x4 render target's `Depth32Float` by a clear
/// (positive ones), `replace_region`, a blit from a buffer and a draw at that depth (positive
/// ones), each read back by `get_bytes` and depth-tested by a draw at depth 0 (`Less` against a
/// positive one, `Greater` against a negative one):
/// - MoltenVK 1.4.1 on an Apple M4 keeps each one the clear, the copy and
///   the CPU store, and its test passes against every positive one and fails against every
///   negative one: Metal's answers, as Metal's own depth test makes them. Its draw at a denormal
///   depth stores 0, as Metal's rasterizer makes it, so a draw's is no difference anywhere. On the
///   M3 Max under macOS 27.0.1, of the M4's identity ([`APPLE_M4`]), the same, 16 of 16 texels
///   each (2026-10-14, with the fact taken: no `depth-denormals` announced).
/// - NVIDIA's 615.71.09 on the RTX 3090 and lavapipe, Mesa 26.2.4 with
///   LLVM 23.1.1 on an Iris Xe (RPL-P) host, keep each one, and their tests fail against every one.
/// - RADV, Mesa 26.2.4 on the Radeon 780M, stores the clear's and the
///   copy's as a zero (the copy's of its sign) and keeps the CPU store's, and its test passes
///   against every one it holds, negative ones and -0 included.
/// - ANV, Mesa 26.2.4 on the Iris Xe (RPL-P), stores the clear's and the copy's as +0 and keeps the
///   CPU store's, and its test fails against every one.
///
/// `DepthTestComparesMagnitudes`, measured by a probe of ours (not committed; 2026-10-15) on a
/// runtime that renders every depth in place: 18 values (+-2, +-0.5, the smallest and largest
/// denormal of each sign, +-0, the smallest normal, 0.25, 1, the next `float` above 1, 1.5, +-inf
/// and a quiet NaN) each stored by `replace_region` into a 4x4 render target's `Depth32Float`,
/// read back by `get_bytes`, and tested by a draw at depth 0, 0.5 and 1 under `Less`,
/// `LessEqual`, `Equal`, `GreaterEqual` and `Greater`:
/// - MoltenVK 1.4.1 on the M3 Max under macOS 27.0.1, of the M4's identity ([`APPLE_M4`]): every
///   value kept, and 270 of 270 outcomes Metal's magnitude rule (225 of them an IEEE compare's),
///   as Metal's own probe `depthkeep` on the same GPU answers. The M4 is unmeasured; its row is
///   the identity's by construction (the fact's docs): through MoltenVK the M4's depth test is
///   the M4's Metal test, where the stand-in would impose the M3 Max's rule. Its pass with depth
///   writes over 16 such values (the runtime lane `depth_denormals`' merge, a draw at 0.5 under
///   `Less`) stores each texel as Metal's own pass on the same GPU does (probe `depthstore`): the
///   draw's depth where it passed, the others without their signs. A pass that sets such a state
///   and draws under none (no draw, a draw of no instances or no vertices, or one under a state
///   without depth writes set over it) keeps every bit there, where Metal's strips them; an
///   indirect draw whose arguments draw no instances strips them as Metal's does.
///
/// Not listed: every other driver, unmeasured for it. Vulkan's depth comparison is of the stored
/// value as it is, so such a driver's single-sample targets take the stand-in.
///
/// `FirstSparseResourceWaitsForQueues`, measured by
/// `runtime_lane::first_use::neutrals_and_sparse_textures_wait_for_no_gpu_work` and a plain
/// Vulkan program (2026-10-10): ANV, Mesa 26.2.4 on the Iris Xe (RPL-P) under `i915`, made a
/// 256x256 sparse image in 567 ms behind 0.5 s of copies, and a second in 0.02 ms behind the same
/// (an Iris Xe (RPL-P) host); gdb put the lane's wait in `vkCreateImage`, at `drmSyncobjWait`. Mesa
/// 26.2.4's `anv_trtt_first_bind_init` submits to each queue and waits, the
/// first time a sparse resource is made; under `xe`, ANV binds sparse memory through VM_BIND,
/// whose creation path makes a synchronous ioctl and waits for no queue (read, not measured).
/// Vulkan tells the two apart by nothing but `sparseAddressSpaceSize`, so the row is ANV's on
/// Intel's GPUs at the release measured.
///
/// `FragmentDepthStoresUnchanged`, measured by
/// `runtime_lane::separate_samples::fragment_depth_stores_depth_values_unchanged` (signed zero,
/// denormals, 1, values outside `[0, 1]`, infinities and NaN payloads, each stored once and
/// again, every sample read back):
/// - lavapipe (`MESA_LLVMPIPE`) from Mesa 26.0.8: Ubuntu's 26.0.8-1ubuntu0.3 on LLVM 21.1.8,
///   x86_64, the linux-gate container on a Ryzen host (2026-10-06), and Mesa 26.2.4 on LLVM 23.1.1,
///   x86_64 (two hosts, 2026-10-06); `separate_samples`' three passes then gave Metal's samples
///   through the draws on both (on 26.0.8). Again at 2026-10-07: 26.0.8 on x86_64 and on arm64 (an
///   M4 mini's container), and 26.2.4, with the passes at 2x2, 4x1, 4x2 and 1x4.
///
/// The probe's fill is a Metal fragment function, so since SEMANTICS 1137 it hands the driver
/// Metal's stored depth: a positive NaN writes nothing, and on a device with
/// `VK_EXT_depth_range_unrestricted` (lavapipe, NVIDIA's) -0 and the values outside `[0, 1]`
/// arrive clamped. The raw values above were measured before; the copy's own draw still hands
/// them raw.
///
/// Probed and not listed, since the probe changed a value: MoltenVK 1.4.1 on an M4 mini (-0 stored
/// as +0, and values outside `[0, 1]` clamped, NaN and +inf to 1, -inf and negatives to 0, each
/// kept when stored again; the denormals kept. Our probe, drawing over a depth of 0.25,
/// found a positive NaN writing nothing, the sample keeping its depth, which over a depth of 1
/// reads as stored as 1, and a negative NaN stored as 0); NVIDIA's 615.71.09 on the RTX 3090 (the
/// denormals flushed to 0, the negative one to -0, and every NaN stored as -inf, each kept when
/// stored again). At 2026-10-07: MoltenVK 1.4.1 on the M3 Max and the M4 (-0 stored as +0),
/// NVIDIA's 615.71.09 on the RTX 3090 and the RTX 5080 Laptop GPU, and RADV and ANV (Mesa
/// 26.2.4; the 780M, the Iris Xe) flush denormals, ANV also storing -0 as +0.
///
/// `DepthStencilImagesLieApart`, measured by
/// `runtime_lane::heap_textures::heap_depth_textures_keep_their_bytes_with_their_twins` (two
/// `Depth32Float` textures side by side in a placement heap, then a buffer over the first one's
/// bytes filled): MoltenVK (`MOLTENVK`) from 1.4.1 (`driverVersion` 10401), on an Apple M4 mini
/// (2026-10-06): the fill changed none of the first's texels in a `Private` heap. MoltenVK's source
/// states it for every image with a depth or stencil aspect on any GPU
/// (`MVKImagePlane::getMTLTexture` makes the texture from the device, not from the memory's
/// `MTLHeap`: "Metal support for depth/stencil from heaps is flaky"). Measured not to have it, so
/// not listed: RADV, Mesa 26.2.4 on the Radeon 780M; ANV, Mesa 26.2.4 on the Iris Xe (RPL-P);
/// NVIDIA's 615.71.09 on the RTX 3090; lavapipe, x86, Mesa 26.0.8 (the fill
/// changed every texel of the first).
///
/// `ColorResolveAverageIsMetals`, measured by
/// `runtime_lane::resolve_rounding::color_resolves_against_metals_rules` (at
/// 2b41d71d, 2026-10-06): 22 formats' every 8-bit pair and 2^17 random texels with their edges,
/// drawn to each sample, read back and resolved by a pass's `AVERAGE`, against the rules above:
/// - MoltenVK from 1.4.1 on Apple's GPUs: every channel Metal's at 2 and 4 samples on an M4
///   mini and the M3 Max; MoltenVK offers no texture of 8 samples on either.
///
/// Measured not to have it, so not listed: RADV (`MESA_RADV`), Mesa 26.2.4 on the Radeon 780M
/// (ties rounded down: 32512 of R8Unorm's 65536 pairs differ at 2 samples); NVIDIA's 615.71.09 on
/// the RTX 5080 Laptop GPU (Metal's on the 16-bit unorm and snorm formats only); ANV
/// (`INTEL_OPEN_SOURCE_MESA`), Mesa 26.2.4 on Intel Graphics (RPL-S); lavapipe (`MESA_LLVMPIPE`),
/// Mesa 26.2.4 on LLVM 23.1.1, x86_64, and Mesa 26.0.8 on LLVM 21.1.8, arm64 (no format Metal's at
/// 4 samples; no texture of 2 samples).
///
/// `CompressedImagesHideAliasedWrites`, measured by
/// `runtime_lane::heap_textures::a_buffer_write_shows_in_an_overlapping_texture` (a heap's
/// buffer and `RGBA8Unorm` texture over the same bytes; the texture written by a blit, then the
/// buffer by a copy): ANV (`INTEL_OPEN_SOURCE_MESA`) on Intel's GPUs from Mesa 26.2.4, on the
/// Iris Xe (RPL-P) and Intel Graphics (RPL-S) (2026-10-06): the texture's write changed 234 of the
/// buffer's 1024 bytes (compressed), and the buffer's write changed none of the 256 texels in a
/// `Shared` heap. Measured not to have it, so not listed: RADV, Mesa 26.2.4 on the Radeon 780M;
/// NVIDIA's 615.71.09 on the RTX 3090; lavapipe, x86, Mesa 26.0.8; MoltenVK 1.4.1 on an M4 (every
/// texel changed).
///
/// `HostTransferImagesKeepTheirBytes`, measured by our standalone probe (2026-10-06;
/// 8 colour formats, 4 usage sets, 16, 64 and 256 texels square, host-visible and device-local
/// memory, the image at the start of its own allocation and at 64 KiB of a 1 MiB one,
/// `hostImageCopy` enabled): ANV on Intel's GPUs from Mesa 26.2.4, on the same RPL-P and RPL-S: in
/// all 90 cases of each memory kind and placement an optimal image of host-transfer usage read the
/// buffer's fill at every texel, as a linear one did; the 6 more, `RGBA8Unorm_sRGB` with storage
/// usage, are no image the format takes. Neither device lists `VK_EXT_image_compression_control`,
/// the route the ruling ordered first.
///
/// `DepthStencilImagesStayGpuOptimized`, measured by
/// `runtime_lane::heap_textures::heap_textures_answer_and_place_as_metal_s_do` (a 64x64
/// `Depth32Float` render target's heap size with `allowGPUOptimizedContents` on and off; the
/// image with the property off is one of `PixelFormatView` usage): MoltenVK from 1.4.1 on Apple's
/// GPUs, an M4 mini (2026-10-07) and the M3 Max: 16512 bytes both ways, Metal's answer with the
/// property on, where Metal answers 16384 with it off (every depth and stencil format and shape
/// alike). MoltenVK's source gives a depth-only image no `PixelFormatView` usage, and sets its
/// texture's `allowGPUOptimizedContents` only from create flags (a 2D view of a 3D image, block
/// texel views) it reads after sizing the image. The row names Apple's GPUs, where Metal's sizes
/// were measured; another driver's sizes are its own, not compared with Metal's.
///
/// `FragmentDepthKeepsDepth16Codes`, measured by
/// `runtime_lane::separate_samples::fragment_depth_keeps_every_16_bit_code` (every code of a
/// 4-sample `Depth16Unorm` texture, stored by a draw of its `float` and read back, then stored
/// again as read, every sample read back):
/// - lavapipe from Mesa 26.0.8: 26.0.8 on LLVM 21.1.8, x86_64 (the linux-gate container), and Mesa
///   26.2.4 on LLVM 23.1.1, x86_64, 2026-10-07: every code kept, 512 of the first reads
///   not the code over 65535 as a correctly rounded `float` but each a different value.
/// - MoltenVK 1.4.2 (`driverVersion` 10402, Homebrew's build) on the Apple M3 Max under macOS
///   27.0.1, 2026-10-23: 65536 different reads of the 65536 codes, **0** not the code over 65535,
///   and **0** changed when stored again. The reads being distinct over the whole space makes the
///   store the identity (the conversion is monotone, so 65536 distinct results of 65536 codes can
///   only be each code itself), and none of them differing from the code over 65535 makes the
///   fetch exactly that `float`, which is what a 16-bit depth's drawn copy needs in both
///   directions (`depth_copy.rs`).
///
///   Re-measured on this stack rather than carried: the 1.4.1 reading below was taken on the M4,
///   a venue this project no longer has, and Homebrew's stock MoltenVK here is 1.4.2. A reading
///   from another GPU on another driver release cannot carry this row.
///
/// Probed and not listed, since no 16-bit depth's draw is taken there: ANV (the Iris Xe RPL-P),
/// RADV (the 780M), both Mesa 26.2.4, NVIDIA's 615.71.09 on the RTX 3090, and MoltenVK 1.4.1 on
/// the M4 kept every code too, each first read exactly the code over 65535. ANV, RADV and lavapipe
/// make no `D16_UNORM_S8_UINT` attachment, so a separate pair's scratch image is refused there
/// first -- and so does MoltenVK on the M3 Max, measured 2026-10-23
/// (`a_separate_multisampled_16_bit_depth_keeps_every_code` refuses there naming
/// `D16_UNORM_S8_UINT` as a depth/stencil attachment), so listing the fact for it does not reach
/// that path either.
///
/// `SampledFetchRecoversCodes`, measured by
/// `runtime_lane::mipmaps_exact::a_first_level_read_in_place_recovers_every_code` at a scratch
/// commit that reads every fetchable first level in place on every driver (2026-10-08): for each of
/// the 30 formats a fetch reads, every code of every component, the second level each probe's
/// filter equal to the CPU's (the filter's answers told apart every probe but those it reads alike:
/// an snorm component's two codes of -1.0, a half's NaNs). One row per GPU, as measured, since the
/// conversion is the GPU's sampler's:
/// - RADV from Mesa 26.2.4 on the Radeon 780M.
/// - ANV from Mesa 26.2.4 on the Iris Xe (RPL-P) and Intel Graphics (RPL-S) (on the Iris Xe, on the
///   RPL-S).
/// - NVIDIA's 615.71.09 on the RTX 5080 Laptop GPU.
/// - MoltenVK 1.4.1 on the M4 and the M3 Max.
/// - lavapipe from Mesa 26.0.8 to 26.2.4, a CPU's fetch with no GPU to name: 26.0.8 on LLVM
///   21.1.8, x86_64 and arm64 (the linux-gate container), and 26.2.4 on LLVM 23.1.1, x86_64.
///
/// Not listed: every GPU and release not measured, among them the RTX 5080 desktop GPU. There
/// `generate_mipmaps` copies the first level into its scratch buffer, as before.
///
/// `InterlockedPassSplitNoSlower`, a cost fact: each GPU ran the same passes in both forms, each
/// forced in turn by a scratch build whose runtime could still keep such a pass whole (2026-10-18;
/// release builds without the validation layer; the GPU's own span, medians, whole against split):
/// - `runtime_lane::tile_path_costs`, milliseconds a frame, two runs of two rounds: `wide
///   composite` (nine 1920x1080 passes a frame, each 26 blended rectangles into a memoryless
///   `RGBA16Float` color, then the fetch that resolves it into the `BGRA8Unorm` base), `base
///   fetched every fourth draw` (the rectangles into the base, every fourth fetching it) and
///   `window: tile dispatch, base fetched every fourth draw, resolve` (three 960x768 passes).
/// - Four draws, microseconds a frame, the medians of 30 frames after the first in three rounds: a
///   1920x1080 `RGBA16Float` target beside a memoryless one, three unblended plain draws then a
///   fetching draw, the fullscreen triangle each. Whole's first frame compiles each plain
///   pipeline's variant at the fetching call: 5 to 35 ms, against 0.1 ms split.
///
/// One row per GPU, as measured, which the window server's passes decide: split ran them faster on
/// every GPU, where only ANV ran the four draws faster whole, by 0.6% and 2.4% in steady frames.
/// - ANV, Mesa 26.2.4, on Intel Graphics (RPL-S), at 1650 MHz: 22.1 against 7.3, 25.5 against 8.9
///   and 12.1 against 8.8 ms; four draws 1984 to 1987 against 1997 to 1999.
/// - ANV, Mesa 26.2.4, on the Iris Xe (RPL-P), at 1200 to 1500 MHz: 6.6 to 7.6 against 3.4 to 3.7,
///   7.9 to 9.4 against 4.3 to 4.9 and 4.6 to 5.2 against 3.4 to 4.1 ms; four draws, at 300 MHz,
///   3759 against 3851.
/// - RADV, Mesa 26.2.4, on the Radeon 780M: 7.1 to 9.1 against 3.3 to 4.0, 9.4 to 9.6 against 5.5
///   to 5.9 and 4.7 against 3.9 ms; four draws 1216 to 1236 against 1134 to 1194.
/// - NVIDIA's 615.71.09 on the RTX 3090: 2.33 against 0.86, 2.62 against 1.30 and 0.75 against
///   0.44 ms; four draws 169 to 185 against 160.
/// - NVIDIA's 615.71.09 on the RTX 5080 Laptop GPU: 1.75 to 1.91 against 0.44 to 0.46, 2.06 to
///   2.25 against 0.89 to 0.90 and 0.57 to 0.61 against 0.33 to 0.35 ms; four draws 155 to 163
///   against 105.
/// - MoltenVK 1.4.1 on the M3 Max, not measured again: measured on the four draws alone by the
///   scratch lane `reroute_costs` at 85f395e1 (2026-10-08), each frame from commit to the wait's
///   return (wall) and the GPU's span, its split's plain draws blended and whole's not, which
///   weighs against the split, so the split's lead stands: under macOS 26.7 wall 968 against 871,
///   GPU 261 against 94; under macOS 27.0.1, the M4's identity (2026-10-14, the same pass at
///   8b1f8606, two runs of three rounds) wall 974 to 1045 against 721 to 769, GPU 914 to 964
///   against 671 to 719, whole's first frame 7.7 to 7.9 ms against 1.2 to 1.8 ms. The M4 itself
///   ran it faster whole (wall 1383 against 1499, though the GPU's span 82 against 12), which its
///   shared identity does not take.
///
/// Not listed: lavapipe, which fetches through input attachments and so never splits for the
/// interlock, and every GPU and release not measured, which split such a pass and announce it.
///
/// `DontCareStoreIsMetals`, read in MoltenVK 1.4.1's source
/// (`MVKAttachmentDescription::getMTLStoreAction`; the restarts of `MVKCommandEncoder`,
/// `MVKCmdPipelineBarrier` and the draws' compute stages) against Metal's answer on the GPU, and
/// measured by `runtime_lane::dont_care_stores` through MoltenVK 1.4.1 on the M4 ([`APPLE_M4`]) and
/// the M3 Max ([`APPLE_M3_MAX`]), both at 8edef9600: every kept case kept its texels. One row per
/// GPU, as measured: Apple's other GPUs are unmeasured, and announce.
///
/// `ColorBlockDecodeIsMetals` and `ChannelBlockDecodeIsMetals`: MoltenVK 1.4.1 on Apple's GPUs
/// takes both, by construction: it hands each BC1 to BC5 format, unorm, sRGB and signed, to Metal
/// as Metal's own pixel format of that name, with no substitute and no decode of its own
/// (`MVKPixelFormats.mm`: no alternate format for any of them, a device without BC reports them
/// unsupported; `MVKImage.mm` refuses a linear or multisampled compressed image; its one BC
/// adjustment, `BC1_RGB`'s alpha swizzle, is of a format the runtime never makes), so Metal's
/// sampler decodes every texel on whichever Apple GPU it runs, Metal's answer there. The M4
/// ([`APPLE_M4`]) is the measured instance: its format scout read every view, read and sample cell
/// of the fourteen BC formats' random blocks byte for byte as plain Metal did; our probe
/// repeats it on the M3 Max. RADV, Mesa 26.2.4, on the Radeon 780M takes the first, its BC1 to BC3
/// texels Metal's and its BC4 and BC5 not (an endpoint of -128 clamped before the sum, some values
/// a `float` ulp off). Each is checked by `runtime_lane::compressed_decode` against the decode
/// fitted to Metal's. NVIDIA's 615.71.09 (the RTX 3090 and the RTX 5080 Laptop GPU), ANV, Mesa
/// 26.2.4, on the Iris Xe and lavapipe, Mesa 26.0.8 decode both families otherwise. Every other
/// driver and GPU is unmeasured, and announces.
///
/// `Bc6hBlockDecodeIsMetals` and `Bc7BlockDecodeIsMetals` (before, every driver was taken to decode
/// both as Metal does, unmeasured): the format scouts' reads of random blocks of the fourteen BC
/// formats, every view, read and sample cell byte for byte against plain Metal's on the M4. BC6H's
/// reads were Metal's on NVIDIA's 615.71.09 on the RTX 3090 and the RTX 5080 Laptop GPU, RADV, Mesa
/// 26.2.4, on the Radeon 780M, ANV, Mesa 26.2.4, on the Iris Xe of Raptor Lake-P and lavapipe, Mesa
/// 26.0.8; BC7's on the same but lavapipe, whose values lie a `float` ulp off. MoltenVK 1.4.1 on
/// Apple's GPUs takes both by construction, as it takes BC1 to BC5's (it hands each format to Metal
/// as Metal's own). No lane re-measures them at the gates: `runtime_lane::compressed_decode` holds
/// no model of either decode.
///
/// `PositionedDepthReadAtStandardPositions`, measured by `runtime_lane::sample_positions` and a
/// probe of it (at f1accaf20 and our probe at 3cfc03dab, 2026-10-10): NVIDIA's
/// (`NVIDIA_PROPRIETARY`) 615.71.09 on the GeForce RTX 3090: plane X drawn at four custom positions
/// into a 4x `Depth32Float` read back each sample's depth at the standard positions, in the same
/// command buffer and in a later one, under each layout transition pair after the pass; a pass at
/// four other positions that loaded it drew no sample under `Less` (each stored depth taken at its
/// own positions); the pass's own test under `Less` drew the samples the custom positions put below
/// the clear; and a caller's image rendered through the stand-in, its depth copied back after the
/// pass, read Metal's depths. The row names no GPU: the stand-in it selects is exact on every
/// device, so NVIDIA's other GPUs at that release (the RTX 5080 Laptop GPU's lane is not yet run)
/// take it too. Measured not to have it, so not listed: RADV, Mesa 26.2.4 on the Radeon 780M (every
/// read and later pass kept the custom positions' depths); ANV, Mesa 26.2.4 on the Iris Xe (RPL-P);
/// lavapipe, x86, LLVM 21.1.8; and MoltenVK 1.4.1 on the M4, whose `runtime_lane::sample_positions`
/// passed rendering the texture's own image (at 53ae2b964). There the RTX 3090 passed through the
/// stand-in.
///
/// `TimestampPoolsAreCounterSampleBuffers` (the probe): MoltenVK 1.4.1 on the M3 Max
/// and the M4 made 32 timestamp query pools and failed the 33rd, and failed a pool of 8192
/// timestamps, with `VK_ERROR_OUT_OF_DEVICE_MEMORY`; its source makes each pool one Metal counter
/// sample buffer of the pool's count (`MVKGPUCounterQueryPool::initMTLCounterSampleBuffer`). A
/// defect fact: MoltenVK takes it at every release and on every GPU.
///
/// `LinearTilingFeaturesUninitialized`, read in MoltenVK 1.4.1's source (`MVKDevice.h` declares
/// `_pixelFormats` before `_metalFeatures`; `MVKPhysicalDevice`'s constructor initializes
/// `_pixelFormats(this)`, whose constructor builds every format's properties through
/// `MVKPixelFormats::setFormatProperties`, before its body calls `initMetalFeatures()`, which
/// clears and then sets `renderLinearTextures`) and measured on the M3 Max (macOS 27.0.1,
/// 2026-10-15): of 40 instances made one after another in one process, 38 reported
/// `R8G8B8A8_UNORM`'s `linearTilingFeatures` as `0x8000dd83` (with `COLOR_ATTACHMENT`,
/// `COLOR_ATTACHMENT_BLEND` and `BLIT_DST`) and 2 as `0x8000d403` (without them), and
/// `vkGetPhysicalDeviceImageFormatProperties` of a linear `COLOR_ATTACHMENT` image succeeded in
/// 38. MoltenVK's code on every GPU, so the row names no vendor; a defect fact, taken at every
/// release.
///
/// `TileStoreIsMetals`, measured by
/// `runtime_lane::snorm_tile_store::a_drawing_pass_keeps_a_snorm_minimum_code_and_says_so` (a
/// 4x4 target of each of the six signed normalized colour formats holding a channel's minimum code
/// in every channel, loaded and drawn over by a full-screen triangle that writes no channel):
/// MoltenVK 1.4.1 on the M3 Max under macOS 27.0.1 ([`APPLE_M4`]'s identity) stored the code
/// above it in every texel, as plain Metal's pass does on that GPU. Its `RGB9E5Float` twin
/// (`a_drawing_pass_keeps_an_unnormalized_shared_exponent_word_and_says_so`: the word `1 | 2 << 9
/// | 3 << 18 | 20 << 27` in every texel) is Metal's by the same construction: MoltenVK renders
/// `E5B9G9R9_UFLOAT_PACK32` as a Metal `RGB9E5Float` attachment, whose pass plain Metal measured
/// normalizing such a word on that GPU. One row for Apple's GPUs, by
/// construction (the fact's docs). Measured not to have it, so not listed (every texel kept its
/// code): RADV, Mesa 26.0.8, on the Radeon 780M and the RX 7900 XTX; ANV, Mesa 26.2.4, on the
/// Iris Xe (RPL-P).
///
/// `DontCareStoreIsOneOfMetals`: measured by `runtime_lane::dont_care_stores`, every texel of each
/// of its twelve kept cases the earlier one or the stored one, none other: at 66ffe6775, RADV, Mesa
/// 26.2.4, on the Radeon 780M and ANV, Mesa 26.2.4, on the UHD 770 (RPL-S); at dd718a7905 with its
/// row, ANV, Mesa 26.2.4, on the Iris Xe (RPL-P). Not listed: NVIDIA's 615.71.09, whose RTX 3080
/// left a third answer; RADV, Mesa 26.0.8, on the RX 7900 XTX, clean at dd718a7905 but an
/// unmeasured identity (`MEASURED_DEVICES`); and every GPU and release not measured, which store
/// with `STORE`.
///
/// `PointerImportsFollowTheirAddress`: ANV, Mesa 26.2.4 on the Iris Xe of RPL-P under Linux 7.2.9's
/// i915, measured with a probe outside the runtime (2026-10-17): a page of a `memfd` filled with
/// `0xaa`, mapped shared at an address and imported by its pointer as a buffer's memory, read
/// `0xaa` through a copy; another `memfd`'s page of `0xbb` mapped over the address with `MAP_FIXED`,
/// the next copy read `0xbb`; a page of `0xa2` imported and replaced by one of `0xb2` before any
/// submission, the first copy read `0xb2`. lavapipe (`MESA_LLVMPIPE`) from Mesa 26.0.8 through
/// 26.2.4 by construction, read in both releases' `lvp_AllocateMemory`, which binds an import's
/// buffers at the host pointer. RADV, Mesa 26.2.4 on the Radeon 780M under Linux 7.2.9's amdgpu,
/// measured through the runtime (2026-10-17): a no-copy buffer over two fresh anonymous private
/// pages, read by a blit; fresh anonymous pages mapped over the address with `MAP_FIXED` and
/// `Device::host_range_remapped` told, the next blit read the new pages, and the newer ones after a
/// second remap; remapped between the buffer's making and its first submission, the first blit
/// read the new pages. RADV, Mesa 26.2.4 on the RX 7900 XTX under the same kernel, measured through
/// the runtime (2026-10-22, `tests::pointer_residency` before its row): a no-copy buffer over two
/// fresh anonymous private pages, read by a blit; fresh anonymous pages mapped over the address
/// with `MAP_FIXED` and `Device::host_range_remapped` told, the next blit of the import kept all
/// along read the new pages. Not listed: RADV, Mesa 26.0.8 on the RX 7900 XTX and the Radeon 780M
/// under Linux 7.0.0, which answered the same, but are unmeasured identities (`MEASURED_DEVICES`).
/// `tests::pointer_residency` in the runtime checks it on a device with it. Every other driver is
/// unmeasured, and keeps its pointer imports.
///
/// `InterlockOrdersDraws`, measured with no barrier between such draws on any driver (a scratch
/// build of the commit that adds it), by `runtime_lane::interlocked_draw_order` (passes of 84
/// draws, fullscreen and held to bands, of two read-modify-writes of an `R32Uint` color that do
/// not commute, at one sample over 1920x1080 and 64x36 and at four over 960x540 and 64x36, every
/// texel read back) and by `runtime_lane::extended_range_blend_table`'s
/// `a_memoryless_color_2_blends_as_a_stored_color_does` (99 banded blends into a memoryless
/// extended-range color in the interlock, then a fetching copy of it): ANV, Mesa 26.2.4 on the
/// Iris Xe (RPL-P), every texel of every pass right in three runs of the first, every band right
/// in the second, and the `extended_range_blend_table` and `framebuffer_fetch` lanes passing.
/// Measured not to have it: RADV, Mesa 26.2.4 on the Radeon 780M, which passes the first but
/// leaves 19 of the second's 495 bands wrong. Every other driver is unmeasured, and takes the
/// barrier.
///
/// `ColorImagesLieAsMetalHeapTextures`, measured by
/// `runtime_lane::heap_textures::device_heap_textures_lie_in_metal_s_tiles` (textures Metal keeps
/// uncompressed in a `Private` placement heap: `RGBA8Unorm`, `R8Unorm` and `RG16Float` ones of
/// `ShaderRead | ShaderWrite` usage of 64 x 64, 37 x 5, 100 x 70 and 300 x 200 texels,
/// `RGBA8Unorm` ones of 64 x 64 of `ShaderRead` and `ShaderRead | RenderTarget` usage with
/// `allowGPUOptimizedContents` off, and one of 8 x 8 of `ShaderRead` usage; written by blits
/// naming each texel, found through a buffer over the heap): MoltenVK 1.4.2 on the M3 Max under
/// macOS 27.0.1, every texel at the byte Metal's tiles place it (the lane's restatement of the
/// tiled layout measured on Metal's own heap textures there, `heap_layouts::tiled_offset`). It
/// decides a notice, not an answer, so the M4 is not measured. Every other driver is unmeasured.
///
/// `MultisampleColorImagesLieAsMetalHeapTextures`, measured by
/// `runtime_lane::heap_device_samples::device_heap_multisample_colour_textures_lie_where_announced`
/// (multisample textures of `ShaderRead | RenderTarget` usage with `allowGPUOptimizedContents`
/// off, which Metal keeps uncompressed, in a `Private` placement heap: `R8Uint`, `R16Uint`,
/// `R32Uint`, `RGBA8Unorm`, `RG32Uint`, `RGBA16Uint` and `RGBA32Uint` ones of 37 x 5, 33 x 33,
/// 64 x 64 and 100 x 70 pixels, 2 and 4 samples; each sample written by a blit naming it and
/// found through a buffer over the heap, and the heap's bytes written there read back by a blit):
/// MoltenVK 1.4.2 on the M3 Max under macOS 27.0.1, every sample of the 56 at the byte Metal's
/// tiles place it each way (the lane's restatement of the layout measured on Metal's own heap
/// textures there, 98 shapes of those formats and of 8 x 8, 16 x 16 and 300 x 200 pixels, drawn
/// per sample or blitted, every sample there). It decides a notice, not an answer, so the M4 is
/// not measured. Every other driver is unmeasured.
///
/// MoltenVK 1.4.2 on the M3 Max under macOS 27.0.1 ([`moltenvk_142`]; 2026-10-21, the runtime
/// lanes in release builds under `--cfg m2v_gpu`, each fact taken):
/// - Measured by the lane that measured its 1.4.1 row, which passed: `DepthImageKeepsDenormals`
///   and `DepthImageKeepsValuesOutsideUnitRange`
///   (`texture_blits::depth_images_keep_what_their_drivers_claim`, `depth_denormals`),
///   `DepthTestKeepsDenormals` and `DepthTestComparesMagnitudes` (`depth_denormals`' stored
///   depths, tested and compared as Metal's), `DepthStencilImagesLieApart`
///   (`heap_textures::heap_depth_textures_keep_their_bytes_with_their_twins`),
///   `DepthStencilImagesStayGpuOptimized`
///   (`heap_textures::heap_textures_answer_and_place_as_metal_s_do`),
///   `ColorResolveAverageIsMetals` (`resolve_rounding::color_resolves_against_metals_rules`),
///   `SampledFetchRecoversCodes`
///   (`mipmaps_exact::a_first_level_read_in_place_recovers_every_code`), `DynamicRestartWarns`
///   (`restart`, `made_at_creation`: its warning withheld at each such pipeline's creation and
///   nowhere else), `DontCareStoreIsMetals` (`dont_care_stores`, `kept_buffer_bytes`),
///   `ColorBlockDecodeIsMetals` and `ChannelBlockDecodeIsMetals`
///   (`compressed_decode::bc1_to_bc5_texels_read_as_metal_s`) and `TileStoreIsMetals`
///   (`snorm_tile_store`).
/// - `BlendlessColorAttachmentsWarn`, new at 1.4.2 (the fact's docs): where the release was
///   unmeasured, the warning of each color attachment of a format the device cannot blend
///   (`R32Uint`, `RGBA32Uint` and the rest) failed the runtime's `srgb_fetch_images` and about 25
///   lanes; with the fact, none.
/// - `TimestampsWaitForEveryQueue`, by the plain Vulkan program behind the fact (our upstream
///   draft's): a 256-byte fill on a queue of another family, submitted just after a 30.7 ms
///   dispatch, completed at 30.78 ms (median) in 40 of 40 rounds where both command buffers wrote
///   timestamps after their work, and at 0.28 ms (1 of 40 held) where neither did; argument
///   buffers on, as the runtime runs. `encodeTimestampStageCounterSamples` is 1.4.1's.
/// - `TimestampPoolsAreCounterSampleBuffers` and `LinearTilingFeaturesUninitialized`, defect
///   facts, read in 1.4.2's source as at 1.4.1:
///   `MVKGPUCounterQueryPool::initMTLCounterSampleBuffer` makes each pool one counter sample
///   buffer of its query count, and `MVKPhysicalDevice` still builds `_pixelFormats` (which reads
///   `renderLinearTextures`) before `initMetalFeatures` sets it.
/// - `Bc6hBlockDecodeIsMetals` and `Bc7BlockDecodeIsMetals`, by construction as at 1.4.1: 1.4.2's
///   `MVKPixelFormats.mm` hands every BC format to Metal as Metal's own, line for line 1.4.1's.
///
/// Not extended to 1.4.2, so not taken there: `InterlockedPassSplitNoSlower`, a cost fact whose
/// 1.4.1 timing needed a scratch build that keeps such a pass whole (its passes split there,
/// announced, as on any unmeasured release); and `FillCompiledPerCommandPool`, whose wait our
/// probe did not find: a command buffer of one `vkCmdFillBuffer`, the first in a fresh command
/// pool, took 0.022 to 0.038 ms (median) in `vkQueueSubmit` beside one to eight threads making
/// compute pipelines (91 and 496 made), as a `vkCmdUpdateBuffer` did, though
/// `MVKCommandEncodingPool::getCmdFillBufferMTLComputePipelineState` is 1.4.1's (fills there take
/// `vkCmdFillBuffer`).
pub(crate) const MEASURED: &[Measured<DriverFact>] = &[
    Measured {
        fact: DriverFact::RobustIndexFetchReadsZero,
        driver: DriverId::MESA_RADV,
        vendor: None,
        device: None,
        first: mesa(26, 2, 4),
        last: mesa(26, 2, 4),
    },
    Measured {
        fact: DriverFact::RobustIndexFetchReadsZero,
        driver: DriverId::INTEL_OPEN_SOURCE_MESA,
        vendor: Some(INTEL_VENDOR_ID),
        device: Some(INTEL_UHD_770_RPL_S),
        first: mesa(26, 2, 4),
        last: mesa(26, 2, 4),
    },
    Measured {
        fact: DriverFact::RobustIndexFetchReadsZero,
        driver: DriverId::NVIDIA_PROPRIETARY,
        vendor: Some(NVIDIA_VENDOR_ID),
        device: Some(NVIDIA_RTX_5080_LAPTOP),
        first: nvidia(615, 71, 9),
        last: nvidia(615, 71, 9),
    },
    Measured {
        fact: DriverFact::DepthImageKeepsDenormals,
        driver: DriverId::NVIDIA_PROPRIETARY,
        vendor: None,
        device: None,
        first: nvidia(615, 71, 9),
        last: nvidia(615, 71, 9),
    },
    Measured {
        fact: DriverFact::DepthImageKeepsDenormals,
        driver: DriverId::MESA_LLVMPIPE,
        vendor: None,
        device: None,
        first: mesa(26, 0, 8),
        last: mesa(26, 2, 4),
    },
    Measured {
        fact: DriverFact::DepthImageKeepsDenormals,
        driver: DriverId::MOLTENVK,
        vendor: None,
        device: None,
        first: moltenvk(1, 4, 1),
        last: moltenvk(1, 4, 1),
    },
    Measured {
        fact: DriverFact::DepthImageKeepsValuesOutsideUnitRange,
        driver: DriverId::MOLTENVK,
        vendor: None,
        device: None,
        first: moltenvk(1, 4, 1),
        last: moltenvk(1, 4, 1),
    },
    Measured {
        fact: DriverFact::DepthTestKeepsDenormals,
        driver: DriverId::MOLTENVK,
        vendor: Some(APPLE_VENDOR_ID),
        device: Some(APPLE_M4),
        first: moltenvk(1, 4, 1),
        last: moltenvk(1, 4, 1),
    },
    Measured {
        fact: DriverFact::DepthTestComparesMagnitudes,
        driver: DriverId::MOLTENVK,
        vendor: Some(APPLE_VENDOR_ID),
        device: Some(APPLE_M4),
        first: moltenvk(1, 4, 1),
        last: moltenvk(1, 4, 1),
    },
    Measured {
        fact: DriverFact::FirstSparseResourceWaitsForQueues,
        driver: DriverId::INTEL_OPEN_SOURCE_MESA,
        vendor: Some(INTEL_VENDOR_ID),
        device: None,
        first: mesa(26, 2, 4),
        last: mesa(26, 2, 4),
    },
    Measured {
        fact: DriverFact::FragmentDepthStoresUnchanged,
        driver: DriverId::MESA_LLVMPIPE,
        vendor: None,
        device: None,
        first: mesa(26, 0, 8),
        last: mesa(26, 2, 4),
    },
    Measured {
        fact: DriverFact::DepthStencilImagesLieApart,
        driver: DriverId::MOLTENVK,
        vendor: None,
        device: None,
        first: moltenvk(1, 4, 1),
        last: moltenvk(1, 4, 1),
    },
    Measured {
        fact: DriverFact::ColorResolveAverageIsMetals,
        driver: DriverId::MOLTENVK,
        vendor: Some(APPLE_VENDOR_ID),
        device: None,
        first: moltenvk(1, 4, 1),
        last: moltenvk(1, 4, 1),
    },
    Measured {
        fact: DriverFact::CompressedImagesHideAliasedWrites,
        driver: DriverId::INTEL_OPEN_SOURCE_MESA,
        vendor: Some(INTEL_VENDOR_ID),
        device: None,
        first: mesa(26, 2, 4),
        last: mesa(26, 2, 4),
    },
    Measured {
        fact: DriverFact::HostTransferImagesKeepTheirBytes,
        driver: DriverId::INTEL_OPEN_SOURCE_MESA,
        vendor: Some(INTEL_VENDOR_ID),
        device: None,
        first: mesa(26, 2, 4),
        last: mesa(26, 2, 4),
    },
    Measured {
        fact: DriverFact::DepthStencilImagesStayGpuOptimized,
        driver: DriverId::MOLTENVK,
        vendor: Some(APPLE_VENDOR_ID),
        device: None,
        first: moltenvk(1, 4, 1),
        last: moltenvk(1, 4, 1),
    },
    Measured {
        fact: DriverFact::FragmentDepthKeepsDepth16Codes,
        driver: DriverId::MESA_LLVMPIPE,
        vendor: None,
        device: None,
        first: mesa(26, 0, 8),
        last: mesa(26, 2, 4),
    },
    moltenvk_142(DriverFact::FragmentDepthKeepsDepth16Codes),
    Measured {
        fact: DriverFact::SampledFetchRecoversCodes,
        driver: DriverId::MESA_RADV,
        vendor: Some(AMD_VENDOR_ID),
        device: Some(AMD_RADEON_780M),
        first: mesa(26, 2, 4),
        last: mesa(26, 2, 4),
    },
    Measured {
        fact: DriverFact::SampledFetchRecoversCodes,
        driver: DriverId::INTEL_OPEN_SOURCE_MESA,
        vendor: Some(INTEL_VENDOR_ID),
        device: Some(INTEL_IRIS_XE_RPL_P),
        first: mesa(26, 2, 4),
        last: mesa(26, 2, 4),
    },
    Measured {
        fact: DriverFact::SampledFetchRecoversCodes,
        driver: DriverId::INTEL_OPEN_SOURCE_MESA,
        vendor: Some(INTEL_VENDOR_ID),
        device: Some(INTEL_UHD_770_RPL_S),
        first: mesa(26, 2, 4),
        last: mesa(26, 2, 4),
    },
    Measured {
        fact: DriverFact::SampledFetchRecoversCodes,
        driver: DriverId::NVIDIA_PROPRIETARY,
        vendor: Some(NVIDIA_VENDOR_ID),
        device: Some(NVIDIA_RTX_5080_LAPTOP),
        first: nvidia(615, 71, 9),
        last: nvidia(615, 71, 9),
    },
    Measured {
        fact: DriverFact::SampledFetchRecoversCodes,
        driver: DriverId::MOLTENVK,
        vendor: Some(APPLE_VENDOR_ID),
        device: Some(APPLE_M4),
        first: moltenvk(1, 4, 1),
        last: moltenvk(1, 4, 1),
    },
    Measured {
        fact: DriverFact::SampledFetchRecoversCodes,
        driver: DriverId::MOLTENVK,
        vendor: Some(APPLE_VENDOR_ID),
        device: Some(APPLE_M3_MAX),
        first: moltenvk(1, 4, 1),
        last: moltenvk(1, 4, 1),
    },
    Measured {
        fact: DriverFact::SampledFetchRecoversCodes,
        driver: DriverId::MESA_LLVMPIPE,
        vendor: None,
        device: None,
        first: mesa(26, 0, 8),
        last: mesa(26, 2, 4),
    },
    Measured {
        fact: DriverFact::InterlockedPassSplitNoSlower,
        driver: DriverId::MOLTENVK,
        vendor: Some(APPLE_VENDOR_ID),
        device: Some(APPLE_M3_MAX),
        first: moltenvk(1, 4, 1),
        last: moltenvk(1, 4, 1),
    },
    Measured {
        fact: DriverFact::InterlockedPassSplitNoSlower,
        driver: DriverId::MOLTENVK,
        vendor: Some(APPLE_VENDOR_ID),
        device: Some(APPLE_M4),
        first: moltenvk(1, 4, 1),
        last: moltenvk(1, 4, 1),
    },
    Measured {
        fact: DriverFact::InterlockedPassSplitNoSlower,
        driver: DriverId::MESA_RADV,
        vendor: Some(AMD_VENDOR_ID),
        device: Some(AMD_RADEON_780M),
        first: mesa(26, 2, 4),
        last: mesa(26, 2, 4),
    },
    Measured {
        fact: DriverFact::InterlockedPassSplitNoSlower,
        driver: DriverId::NVIDIA_PROPRIETARY,
        vendor: Some(NVIDIA_VENDOR_ID),
        device: Some(NVIDIA_RTX_5080_LAPTOP),
        first: nvidia(615, 71, 9),
        last: nvidia(615, 71, 9),
    },
    Measured {
        fact: DriverFact::InterlockedPassSplitNoSlower,
        driver: DriverId::NVIDIA_PROPRIETARY,
        vendor: Some(NVIDIA_VENDOR_ID),
        device: Some(NVIDIA_RTX_3090),
        first: nvidia(615, 71, 9),
        last: nvidia(615, 71, 9),
    },
    Measured {
        fact: DriverFact::InterlockedPassSplitNoSlower,
        driver: DriverId::INTEL_OPEN_SOURCE_MESA,
        vendor: Some(INTEL_VENDOR_ID),
        device: Some(INTEL_IRIS_XE_RPL_P),
        first: mesa(26, 2, 4),
        last: mesa(26, 2, 4),
    },
    Measured {
        fact: DriverFact::InterlockedPassSplitNoSlower,
        driver: DriverId::INTEL_OPEN_SOURCE_MESA,
        vendor: Some(INTEL_VENDOR_ID),
        device: Some(INTEL_UHD_770_RPL_S),
        first: mesa(26, 2, 4),
        last: mesa(26, 2, 4),
    },
    Measured {
        fact: DriverFact::DynamicRestartWarns,
        driver: DriverId::MOLTENVK,
        vendor: None,
        device: None,
        first: moltenvk(1, 4, 1),
        last: moltenvk(1, 4, 1),
    },
    Measured {
        fact: DriverFact::FillCompiledPerCommandPool,
        driver: DriverId::MOLTENVK,
        vendor: None,
        device: None,
        first: moltenvk(1, 4, 1),
        last: moltenvk(1, 4, 1),
    },
    Measured {
        fact: DriverFact::DontCareStoreIsMetals,
        driver: DriverId::MOLTENVK,
        vendor: Some(APPLE_VENDOR_ID),
        device: Some(APPLE_M4),
        first: moltenvk(1, 4, 1),
        last: moltenvk(1, 4, 1),
    },
    Measured {
        fact: DriverFact::DontCareStoreIsMetals,
        driver: DriverId::MOLTENVK,
        vendor: Some(APPLE_VENDOR_ID),
        device: Some(APPLE_M3_MAX),
        first: moltenvk(1, 4, 1),
        last: moltenvk(1, 4, 1),
    },
    Measured {
        fact: DriverFact::SemaphoreOperationsStallSubmits,
        driver: DriverId::NVIDIA_PROPRIETARY,
        vendor: Some(NVIDIA_VENDOR_ID),
        device: Some(NVIDIA_RTX_5080_LAPTOP),
        first: nvidia(615, 71, 9),
        last: nvidia(615, 71, 9),
    },
    Measured {
        fact: DriverFact::SemaphoreOperationsStallSubmits,
        driver: DriverId::NVIDIA_PROPRIETARY,
        vendor: Some(NVIDIA_VENDOR_ID),
        device: Some(NVIDIA_RTX_3090),
        first: nvidia(615, 71, 9),
        last: nvidia(615, 71, 9),
    },
    Measured {
        fact: DriverFact::TimestampsWaitForEveryQueue,
        driver: DriverId::MOLTENVK,
        vendor: Some(APPLE_VENDOR_ID),
        device: None,
        first: moltenvk(1, 4, 1),
        last: moltenvk(1, 4, 1),
    },
    Measured {
        fact: DriverFact::TimestampPoolsAreCounterSampleBuffers,
        driver: DriverId::MOLTENVK,
        vendor: Some(APPLE_VENDOR_ID),
        device: None,
        first: moltenvk(1, 4, 1),
        last: moltenvk(1, 4, 1),
    },
    Measured {
        fact: DriverFact::LinearTilingFeaturesUninitialized,
        driver: DriverId::MOLTENVK,
        vendor: None,
        device: None,
        first: moltenvk(1, 4, 1),
        last: moltenvk(1, 4, 1),
    },
    Measured {
        fact: DriverFact::DontCareStoreIsOneOfMetals,
        driver: DriverId::MESA_RADV,
        vendor: Some(AMD_VENDOR_ID),
        device: Some(AMD_RADEON_780M),
        first: mesa(26, 2, 4),
        last: mesa(26, 2, 4),
    },
    Measured {
        fact: DriverFact::DontCareStoreIsOneOfMetals,
        driver: DriverId::INTEL_OPEN_SOURCE_MESA,
        vendor: Some(INTEL_VENDOR_ID),
        device: Some(INTEL_UHD_770_RPL_S),
        first: mesa(26, 2, 4),
        last: mesa(26, 2, 4),
    },
    Measured {
        fact: DriverFact::DontCareStoreIsOneOfMetals,
        driver: DriverId::INTEL_OPEN_SOURCE_MESA,
        vendor: Some(INTEL_VENDOR_ID),
        device: Some(INTEL_IRIS_XE_RPL_P),
        first: mesa(26, 2, 4),
        last: mesa(26, 2, 4),
    },
    Measured {
        fact: DriverFact::ColorBlockDecodeIsMetals,
        driver: DriverId::MESA_RADV,
        vendor: Some(AMD_VENDOR_ID),
        device: Some(AMD_RADEON_780M),
        first: mesa(26, 2, 4),
        last: mesa(26, 2, 4),
    },
    Measured {
        fact: DriverFact::ColorBlockDecodeIsMetals,
        driver: DriverId::MOLTENVK,
        vendor: Some(APPLE_VENDOR_ID),
        device: None,
        first: moltenvk(1, 4, 1),
        last: moltenvk(1, 4, 1),
    },
    Measured {
        fact: DriverFact::ChannelBlockDecodeIsMetals,
        driver: DriverId::MOLTENVK,
        vendor: Some(APPLE_VENDOR_ID),
        device: None,
        first: moltenvk(1, 4, 1),
        last: moltenvk(1, 4, 1),
    },
    Measured {
        fact: DriverFact::PositionedDepthReadAtStandardPositions,
        driver: DriverId::NVIDIA_PROPRIETARY,
        vendor: Some(NVIDIA_VENDOR_ID),
        device: None,
        first: nvidia(615, 71, 9),
        last: nvidia(615, 71, 9),
    },
    Measured {
        fact: DriverFact::Bc6hBlockDecodeIsMetals,
        driver: DriverId::NVIDIA_PROPRIETARY,
        vendor: Some(NVIDIA_VENDOR_ID),
        device: Some(NVIDIA_RTX_3090),
        first: nvidia(615, 71, 9),
        last: nvidia(615, 71, 9),
    },
    Measured {
        fact: DriverFact::Bc6hBlockDecodeIsMetals,
        driver: DriverId::NVIDIA_PROPRIETARY,
        vendor: Some(NVIDIA_VENDOR_ID),
        device: Some(NVIDIA_RTX_5080_LAPTOP),
        first: nvidia(615, 71, 9),
        last: nvidia(615, 71, 9),
    },
    Measured {
        fact: DriverFact::Bc6hBlockDecodeIsMetals,
        driver: DriverId::MESA_RADV,
        vendor: Some(AMD_VENDOR_ID),
        device: Some(AMD_RADEON_780M),
        first: mesa(26, 2, 4),
        last: mesa(26, 2, 4),
    },
    Measured {
        fact: DriverFact::Bc6hBlockDecodeIsMetals,
        driver: DriverId::INTEL_OPEN_SOURCE_MESA,
        vendor: Some(INTEL_VENDOR_ID),
        device: Some(INTEL_IRIS_XE_RPL_P),
        first: mesa(26, 2, 4),
        last: mesa(26, 2, 4),
    },
    Measured {
        fact: DriverFact::Bc6hBlockDecodeIsMetals,
        driver: DriverId::MESA_LLVMPIPE,
        vendor: None,
        device: None,
        first: mesa(26, 0, 8),
        last: mesa(26, 0, 8),
    },
    Measured {
        fact: DriverFact::Bc6hBlockDecodeIsMetals,
        driver: DriverId::MOLTENVK,
        vendor: Some(APPLE_VENDOR_ID),
        device: None,
        first: moltenvk(1, 4, 1),
        last: moltenvk(1, 4, 1),
    },
    Measured {
        fact: DriverFact::Bc7BlockDecodeIsMetals,
        driver: DriverId::NVIDIA_PROPRIETARY,
        vendor: Some(NVIDIA_VENDOR_ID),
        device: Some(NVIDIA_RTX_3090),
        first: nvidia(615, 71, 9),
        last: nvidia(615, 71, 9),
    },
    Measured {
        fact: DriverFact::Bc7BlockDecodeIsMetals,
        driver: DriverId::NVIDIA_PROPRIETARY,
        vendor: Some(NVIDIA_VENDOR_ID),
        device: Some(NVIDIA_RTX_5080_LAPTOP),
        first: nvidia(615, 71, 9),
        last: nvidia(615, 71, 9),
    },
    Measured {
        fact: DriverFact::Bc7BlockDecodeIsMetals,
        driver: DriverId::MESA_RADV,
        vendor: Some(AMD_VENDOR_ID),
        device: Some(AMD_RADEON_780M),
        first: mesa(26, 2, 4),
        last: mesa(26, 2, 4),
    },
    Measured {
        fact: DriverFact::Bc7BlockDecodeIsMetals,
        driver: DriverId::INTEL_OPEN_SOURCE_MESA,
        vendor: Some(INTEL_VENDOR_ID),
        device: Some(INTEL_IRIS_XE_RPL_P),
        first: mesa(26, 2, 4),
        last: mesa(26, 2, 4),
    },
    Measured {
        fact: DriverFact::Bc7BlockDecodeIsMetals,
        driver: DriverId::MOLTENVK,
        vendor: Some(APPLE_VENDOR_ID),
        device: None,
        first: moltenvk(1, 4, 1),
        last: moltenvk(1, 4, 1),
    },
    Measured {
        fact: DriverFact::ShadowCopyBackDispatchFaster,
        driver: DriverId::INTEL_OPEN_SOURCE_MESA,
        vendor: Some(INTEL_VENDOR_ID),
        device: None,
        first: mesa(26, 2, 4),
        last: mesa(26, 2, 4),
    },
    Measured {
        fact: DriverFact::ShadowCopyInDispatchFaster,
        driver: DriverId::INTEL_OPEN_SOURCE_MESA,
        vendor: Some(INTEL_VENDOR_ID),
        device: Some(INTEL_UHD_770_RPL_S),
        first: mesa(26, 2, 4),
        last: mesa(26, 2, 4),
    },
    Measured {
        fact: DriverFact::RowCopiesDispatchFaster,
        driver: DriverId::INTEL_OPEN_SOURCE_MESA,
        vendor: Some(INTEL_VENDOR_ID),
        device: None,
        first: mesa(26, 2, 4),
        last: mesa(26, 2, 4),
    },
    Measured {
        fact: DriverFact::RowCopiesDispatchFaster,
        driver: DriverId::NVIDIA_PROPRIETARY,
        vendor: Some(NVIDIA_VENDOR_ID),
        device: Some(NVIDIA_RTX_5080_LAPTOP),
        first: nvidia(615, 71, 9),
        last: nvidia(615, 71, 9),
    },
    Measured {
        fact: DriverFact::SampledDepthKeepsHiZ,
        driver: DriverId::INTEL_OPEN_SOURCE_MESA,
        vendor: Some(INTEL_VENDOR_ID),
        device: Some(INTEL_IRIS_XE_RPL_P),
        first: mesa(26, 2, 4),
        last: mesa(26, 2, 4),
    },
    Measured {
        fact: DriverFact::MultisampledDepthFasterOutOfGeneral,
        driver: DriverId::INTEL_OPEN_SOURCE_MESA,
        vendor: Some(INTEL_VENDOR_ID),
        device: Some(INTEL_IRIS_XE_RPL_P),
        first: mesa(26, 2, 4),
        last: mesa(26, 2, 4),
    },
    Measured {
        fact: DriverFact::TileStoreIsMetals,
        driver: DriverId::MOLTENVK,
        vendor: Some(APPLE_VENDOR_ID),
        device: None,
        first: moltenvk(1, 4, 1),
        last: moltenvk(1, 4, 1),
    },
    Measured {
        fact: DriverFact::PointerImportsFollowTheirAddress,
        driver: DriverId::INTEL_OPEN_SOURCE_MESA,
        vendor: Some(INTEL_VENDOR_ID),
        device: Some(INTEL_IRIS_XE_RPL_P),
        first: mesa(26, 2, 4),
        last: mesa(26, 2, 4),
    },
    Measured {
        fact: DriverFact::PointerImportsFollowTheirAddress,
        driver: DriverId::MESA_RADV,
        vendor: Some(AMD_VENDOR_ID),
        device: Some(AMD_RADEON_780M),
        first: mesa(26, 2, 4),
        last: mesa(26, 2, 4),
    },
    Measured {
        fact: DriverFact::PointerImportsFollowTheirAddress,
        driver: DriverId::MESA_RADV,
        vendor: Some(AMD_VENDOR_ID),
        device: Some(AMD_RX_7900_XTX),
        first: mesa(26, 2, 4),
        last: mesa(26, 2, 4),
    },
    Measured {
        fact: DriverFact::PointerImportsFollowTheirAddress,
        driver: DriverId::MESA_LLVMPIPE,
        vendor: None,
        device: None,
        first: mesa(26, 0, 8),
        last: mesa(26, 2, 4),
    },
    Measured {
        fact: DriverFact::InterlockOrdersDraws,
        driver: DriverId::INTEL_OPEN_SOURCE_MESA,
        vendor: Some(INTEL_VENDOR_ID),
        device: Some(INTEL_IRIS_XE_RPL_P),
        first: mesa(26, 2, 4),
        last: mesa(26, 2, 4),
    },
    moltenvk_142(DriverFact::DepthImageKeepsDenormals),
    moltenvk_142(DriverFact::DepthImageKeepsValuesOutsideUnitRange),
    moltenvk_142(DriverFact::DepthTestKeepsDenormals),
    moltenvk_142(DriverFact::DepthTestComparesMagnitudes),
    moltenvk_142(DriverFact::DepthStencilImagesLieApart),
    moltenvk_142(DriverFact::ColorResolveAverageIsMetals),
    moltenvk_142(DriverFact::DepthStencilImagesStayGpuOptimized),
    moltenvk_142(DriverFact::SampledFetchRecoversCodes),
    moltenvk_142(DriverFact::DynamicRestartWarns),
    moltenvk_142(DriverFact::BlendlessColorAttachmentsWarn),
    moltenvk_142(DriverFact::DontCareStoreIsMetals),
    moltenvk_142(DriverFact::TimestampsWaitForEveryQueue),
    moltenvk_142(DriverFact::TimestampPoolsAreCounterSampleBuffers),
    moltenvk_142(DriverFact::LinearTilingFeaturesUninitialized),
    moltenvk_142(DriverFact::ColorBlockDecodeIsMetals),
    moltenvk_142(DriverFact::ChannelBlockDecodeIsMetals),
    moltenvk_142(DriverFact::Bc6hBlockDecodeIsMetals),
    moltenvk_142(DriverFact::Bc7BlockDecodeIsMetals),
    moltenvk_142(DriverFact::TileStoreIsMetals),
    moltenvk_142(DriverFact::ColorImagesLieAsMetalHeapTextures),
    moltenvk_142(DriverFact::MultisampleColorImagesLieAsMetalHeapTextures),
];

/// Intel's `vendorID`.
const INTEL_VENDOR_ID: u32 = 0x8086;

/// Mesa's `vendorID` (`VK_VENDOR_ID_MESA`), which lavapipe reports, with `deviceID` 0.
const MESA_VENDOR_ID: u32 = 0x10005;

/// One device identity the runtime's driver facts were measured on (the module docs): a driver
/// at one release on one GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MeasuredDevice {
    /// `VkPhysicalDeviceDriverProperties::driverID`.
    pub(crate) driver: DriverId,
    /// `VkPhysicalDeviceProperties::vendorID`.
    pub(crate) vendor: u32,
    /// `VkPhysicalDeviceProperties::deviceID`: the device class a measurement covers, which a
    /// description of any `deviceID` of the same GPU is
    /// ([`DeviceDescription::canonical_device_id`]).
    pub(crate) device: u32,
    /// `VkPhysicalDeviceProperties::driverVersion`.
    pub(crate) release: u32,
}

/// Every device identity the runtime lanes measure the driver facts on: the C3 venues' devices
/// (`validation/cutover/c3-venues.json`), the M3 Max under macOS 26.7 (the MoltenVK venue's device
/// before its macOS 27 capture, which is the M4's identity), the RTX 3090 development venue, the
/// RX 7900 XTX beside the 780M venue's GPU (RADV, Mesa 26.2.4, the same host), and lavapipe at
/// Mesa 26.2.4 (the Linux hosts' own, beside the gate container's 26.0.8). A fact
/// measured nowhere else is taken nowhere else. lavapipe's identity is Mesa's release alone: its
/// LLVM and CPU are in no field of it.
pub(crate) const MEASURED_DEVICES: &[MeasuredDevice] = &[
    MeasuredDevice {
        driver: DriverId::MOLTENVK,
        vendor: APPLE_VENDOR_ID,
        device: APPLE_M3_MAX,
        release: moltenvk(1, 4, 1),
    },
    MeasuredDevice {
        driver: DriverId::MOLTENVK,
        vendor: APPLE_VENDOR_ID,
        device: APPLE_M4,
        release: moltenvk(1, 4, 1),
    },
    MeasuredDevice {
        driver: DriverId::MOLTENVK,
        vendor: APPLE_VENDOR_ID,
        device: APPLE_M4,
        release: moltenvk(1, 4, 2),
    },
    MeasuredDevice {
        driver: DriverId::MESA_LLVMPIPE,
        vendor: MESA_VENDOR_ID,
        device: 0,
        release: mesa(26, 0, 8),
    },
    MeasuredDevice {
        driver: DriverId::MESA_LLVMPIPE,
        vendor: MESA_VENDOR_ID,
        device: 0,
        release: mesa(26, 2, 4),
    },
    MeasuredDevice {
        driver: DriverId::MESA_RADV,
        vendor: AMD_VENDOR_ID,
        device: AMD_RADEON_780M,
        release: mesa(26, 2, 4),
    },
    MeasuredDevice {
        driver: DriverId::MESA_RADV,
        vendor: AMD_VENDOR_ID,
        device: AMD_RX_7900_XTX,
        release: mesa(26, 2, 4),
    },
    MeasuredDevice {
        driver: DriverId::INTEL_OPEN_SOURCE_MESA,
        vendor: INTEL_VENDOR_ID,
        device: INTEL_IRIS_XE_RPL_P,
        release: mesa(26, 2, 4),
    },
    MeasuredDevice {
        driver: DriverId::INTEL_OPEN_SOURCE_MESA,
        vendor: INTEL_VENDOR_ID,
        device: INTEL_UHD_770_RPL_S,
        release: mesa(26, 2, 4),
    },
    MeasuredDevice {
        driver: DriverId::NVIDIA_PROPRIETARY,
        vendor: NVIDIA_VENDOR_ID,
        device: NVIDIA_RTX_3090,
        release: nvidia(615, 71, 9),
    },
    MeasuredDevice {
        driver: DriverId::NVIDIA_PROPRIETARY,
        vendor: NVIDIA_VENDOR_ID,
        device: NVIDIA_RTX_5080_LAPTOP,
        release: nvidia(615, 71, 9),
    },
];

impl DeviceDescription {
    /// The runtime's driver facts ([`DriverFact`]) of the device this description names (the
    /// module docs' rule): on a measured identity ([`DeviceDescription::measured_identity`]) those
    /// its driver's rows name at its release, on its vendor and GPU; on any other identity only the
    /// defects its driver is measured with on any release or GPU
    /// ([`DriverFact::assumed_where_unmeasured`]). The runtime relies on an answer fact only where
    /// the driver has it, and elsewhere takes the exact path that needs none.
    pub fn driver_facts(&self) -> BTreeSet<DriverFact> {
        // A measured identity is no GPU a named release puts out (`NAMED_RELEASES`).
        let measured = self.measured_identity();
        let gpu = self.measured_gpu();
        MEASURED
            .iter()
            .filter(|row| Some(row.driver) == self.driver_id)
            .filter(|row| match measured {
                true => {
                    (row.first..=row.last).contains(&self.driver_version)
                        && row.vendor.is_none_or(|v| v == self.vendor_id)
                        && row.device.is_none_or(|d| Some(d) == gpu)
                }
                false => row.fact.assumed_where_unmeasured(),
            })
            .map(|row| row.fact)
            .collect()
    }

    /// Whether the device's identity (`driverID`, `vendorID`, `deviceID` and `driverVersion`) is
    /// one the runtime's driver facts were measured on (`MEASURED_DEVICES`, the module docs), its
    /// `deviceID` read as the GPU's measured one
    /// ([`canonical_device_id`](DeviceDescription::canonical_device_id)). Another release of a
    /// measured driver, another GPU of it, or another driver is not: the runtime takes no answer
    /// fact there, assumes its driver's measured defects, and says so once in the informational
    /// message `metal2vulkan-unmeasured-driver`. A description taken as unmeasured
    /// ([`DeviceDescription::assumed_unmeasured`]) is none.
    pub fn measured_identity(&self) -> bool {
        let gpu = self.measured_gpu();
        let identity = (self.vendor_id, gpu);
        !named_out(
            self.driver_id,
            identity,
            self.driver_version,
            &self.device_name,
        ) && MEASURED_DEVICES.iter().any(|m| {
            Some(m.driver) == self.driver_id
                && m.vendor == self.vendor_id
                && Some(m.device) == gpu
                && m.release == self.driver_version
        })
    }

    /// The formats the device lists as color attachments of optimal tiling without
    /// `COLOR_ATTACHMENT_BLEND`, in order: those whose pipelines a driver with
    /// [`DriverFact::BlendlessColorAttachmentsWarn`] warns of
    /// ([`DriverFact::pipeline_creation_diagnostics`]).
    pub fn blendless_color_formats(&self) -> Vec<Format> {
        let (color, blend) = (
            FormatFeatureFlags2::COLOR_ATTACHMENT,
            FormatFeatureFlags2::COLOR_ATTACHMENT_BLEND,
        );
        (self.formats.formats.iter())
            .filter(|(_, p)| p.optimal_tiling.contains(color) && !p.optimal_tiling.contains(blend))
            .map(|(&format, _)| format)
            .collect()
    }

    /// The linear-tiling format features the device's driver may report or not
    /// ([`DriverFact::unreliable_linear_tiling`] of its [`DeviceDescription::driver_facts`]),
    /// which the description's formats leave out: `DeviceDescription::query` drops them from every
    /// format's `linear_tiling`, so two descriptions of one device read alike. Empty for a driver
    /// that reports its features as they are.
    pub fn unreliable_linear_tiling(&self) -> FormatFeatureFlags2 {
        self.driver_facts()
            .into_iter()
            .fold(FormatFeatureFlags2::empty(), |features, fact| {
                features.union(fact.unreliable_linear_tiling())
            })
    }

    /// The answer facts the device's driver is measured to give on another release or GPU, which
    /// the device does not take because its identity is not measured
    /// ([`DeviceDescription::measured_identity`]): the runtime takes the exact path each needs
    /// none of instead. Empty on a measured identity, and for a driver with no answer measured.
    pub fn driver_facts_withheld(&self) -> BTreeSet<DriverFact> {
        if self.measured_identity() {
            return BTreeSet::new();
        }
        MEASURED
            .iter()
            .filter(|row| Some(row.driver) == self.driver_id)
            .filter(|row| !row.fact.assumed_where_unmeasured())
            .map(|row| row.fact)
            .collect()
    }
}

/// Mesa's `driverVersion` of a release: `VK_MAKE_API_VERSION(0, major, minor, patch)`.
pub(crate) const fn mesa(major: u32, minor: u32, patch: u32) -> u32 {
    (major << 22) | (minor << 12) | patch
}

/// NVIDIA's `driverVersion` of a release: its major, minor and patch from bits 22, 14 and 6.
pub(crate) const fn nvidia(major: u32, minor: u32, patch: u32) -> u32 {
    (major << 22) | (minor << 14) | (patch << 6)
}

/// MoltenVK's `driverVersion` of a release: `major * 10000 + minor * 100 + patch`.
pub(crate) const fn moltenvk(major: u32, minor: u32, patch: u32) -> u32 {
    major * 10000 + minor * 100 + patch
}
