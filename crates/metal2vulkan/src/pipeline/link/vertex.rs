//! Vertex fetch projection and `VertexInputState`.
//!
//! [`vertex_fetch_projection_for`] is the one function that decides, per attribute a vertex
//! function reads, whether Vulkan's fixed-function vertex input fetches it or the shader pulls it.
//! The compile API (`pipeline::compile`), link, and the tests all call it; there is no other copy.
//!
//! 1. the **numeric class**: the shader declares every attribute `float` or `uint`, and
//!    `lower::stage_io::fetch::class_match` says how the format meets it. An unnormalized 8- or
//!    16-bit integer format read into a float attribute reads each component's integer value as a
//!    float, as Metal does: fixed-function input fetches it in the format's `USCALED` or `SSCALED`
//!    form (`metal::vertex::vk_scaled_vertex_format`), which converts exactly, and the pulled fetch
//!    converts it in the shader. A 32-bit integer format read into a float attribute is
//!    `InvalidDescriptor(VertexAttribute)`, as Metal refuses the pipeline. A float-class format
//!    (float, half, normalized, packed) read into an integer attribute is refused
//!    (`Unsupported{Lowering(StageIo), NotImplemented}`): Metal's meaning for that conversion is
//!    not established, so neither fetch path guesses one. A signed-integer format feeds a `uint`
//!    variable, which fixed-function input may not (Vulkan's numeric-type match), so it is pulled
//!    and sign-extended in the shader;
//! 2. the per-attribute choice of `metal::vertex::attribute_fetch` (format, step, profile), in
//!    item 1's Vulkan format. It
//!    pulls every attribute when the profile's buffer containment is checked in the shader
//!    (`memory::contain::encoding`: containment on and no buffer robustness): Vulkan bounds
//!    fixed-function input only under robust buffer access (`VUID-vkCmdDraw-None-02721`), and a
//!    Metal draw may fetch past its buffer, so the fetch moves into the shader behind the check.
//!    Every later item then finds no fixed-function attribute.
//!    A layout of Metal's **constant step** is answered there with a divisor of 0, whose binding
//!    this module emits with a **stride of 0** ([`binding_stride`]): Vulkan's zero divisor fetches
//!    element `firstInstance`, Metal's constant step the first element whatever the base instance,
//!    and a stride of 0 is what makes those the same bytes for every draw. Items 3, 4 and 6 read
//!    the stride the binding carries and not the layout's, so such a binding meets their limits as
//!    a stride of 0 does -- including the portability subset's, which a stride of 0 fails, so the
//!    constant step stays pulled there;
//! 3. **alignment**: Metal lets a stride and an attribute offset be any byte multiple on Apple5 and
//!    later, while Vulkan's fixed-function attribute address must be a multiple of the format's
//!    component size (of the whole format for a packed one). An attribute whose offset or static
//!    binding stride is not such a multiple is pulled (the pulled fetch reads bytes), and so is one
//!    whose static stride breaks the portability subset's rule (a multiple of
//!    `minVertexInputBindingStrideAlignment` and at least as large), or that ends past its static
//!    stride where the subset withholds `vertexAttributeAccessBeyondStride`;
//! 4. an attribute whose offset or binding stride exceeds `maxVertexInputAttributeOffset` /
//!    `maxVertexInputBindingStride` is pulled;
//! 5. while the fixed-function attributes exceed `maxVertexInputAttributes` or their layouts
//!    `maxVertexInputBindings`, the highest fixed-function attribute is pulled. Locations are
//!    dense in attribute order, so an attribute's index itself is never a limit;
//! 6. when the strides are dynamic state (`strides_dynamic`: a fixed-function binding's stride
//!    is set per bind and the profile has dynamic strides), `vkCmdBindVertexBuffers2` passes
//!    every binding's stride, static ones included, and a nonzero one must cover every
//!    attribute the binding fetches (`VUID-vkCmdBindVertexBuffers2-pStrides-06209`): a
//!    fixed-function attribute of a static layout that ends past the layout's nonzero stride is
//!    pulled. Pulling it changes no per-bind layout, so the strides stay dynamic;
//! 7. a **pulled-vertex-buffer variant** (`VariantKey::pulled_vertex_buffers`) pulls every
//!    remaining fixed-function attribute of the layouts it names: the binder asks for it when a
//!    shape binds such a layout at an offset or with a per-bind stride fixed-function input
//!    cannot fetch. It runs after the items above, so the variant's fixed-function bindings are
//!    exactly the projection's others, attribute for attribute;
//! 8. the **de-indexed and private list forms** (`VariantKey::primitive_id_indices`,
//!    `VariantKey::private_list`) pull every remaining fixed-function attribute in the projection
//!    their compile takes, and link resolves their vertex input the same way
//!    (`RenderLinkInputs::pull_every`): their draw runs without indices, so fixed-function input
//!    would fetch at the stream position, where the function reads the vertex its index names. The
//!    layouts they pull are not a pulled-vertex-buffer variant's (`variant_pulled` keeps item 7's).
//!
//! A fixed-function entry keeps nothing of the format (`VertexFetch::FixedFunction`): no
//! instruction of the module names it, so every format the device fetches in fixed function
//! compiles one module, whatever its encoding, offset or stride. The format is the
//! `VertexAttribute` state this module emits, and what the device's conversion of it may answer
//! otherwise than Metal's fetch is noted where the pipeline links, on its vertex stage
//! ([`fixed_function_notes`]): a normalized format's quotient
//! (`PrecisionNote::DeviceVertexConversion`; on a profile whose conversion is measured to be
//! Metal's, `VkProperty::MetalNormalizedVertexFetch` among the pipeline's requirements in its
//! place) and a narrow float's NaN read as `float` (`PrecisionNote::DeviceNanEncoding`).
//!
//! The projection is part of the vertex function's compile key; link recomputes it from the
//! function's stage-IO contract and refuses a stage compiled under another one
//! ([`vertex_input_state`]), so a stage compiled with fixed-function input for a profile without
//! buffer robustness never links against a vertex descriptor there.

use std::collections::{BTreeMap, BTreeSet};

use crate::device::requirements::{VkFeature, VkProperty};
use crate::device::DeviceProfile;
use crate::error::{DescriptorError, DriverDeviation, Error, InternalError, InternalKind};
use crate::lower::stage_io::fetch::{class_match, encoding, ClassMatch, Encoding};
use crate::metal::vertex::{
    fetch_as, vertex_alignment, vk_scaled_vertex_format, vk_vertex_format, AttributeFetch,
};
use crate::metal::VertexFormat;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::descriptor::{
    VertexAttributeDescriptor, VertexBufferLayoutDescriptor, VertexDescriptor,
};
use crate::pipeline::interface::StageInterface;
use crate::pipeline::link::varyings::Locations;
use crate::pipeline::lower_inputs::{VertexFetch, VertexFetchProjection};
use crate::pipeline::stage_io::IoNumeric;
use crate::pipeline::sym::VaryingKey;
use crate::pipeline::vertex::{
    PulledAttribute, VertexAttribute, VertexBinding, VertexInputState, VertexStride,
};
use crate::vk;

/// How attribute `attribute` of `format` meets a variable of class `declared` (item 1 of the
/// module docs): the Vulkan format fixed-function input would deliver it in, and whether it can
/// (not an unnormalized signed-integer format into the `uint` variable every integer attribute is
/// declared as, which Vulkan's numeric-type match does not allow and which is pulled and
/// sign-extended in the shader), or the class refusals of `fetch::class_match`.
///
/// A signed **normalized** format is not that case: it is a float-class format (`Encoding`'s
/// `is_float`), so `fetch::class_match` admits it to a `float` variable alone and refuses it to
/// an integer one, and Vulkan's `SNORM` vertex formats deliver floats: the device converts it
/// where fixed-function input fetches it ([`fixed_function_notes`]).
fn class_format(
    format: VertexFormat,
    declared: IoNumeric,
    attribute: u32,
) -> Result<(vk::Format, bool), Error> {
    let e = encoding(format)?;
    match class_match(e, declared == IoNumeric::Float, attribute)? {
        ClassMatch::Same => Ok((
            vk_vertex_format(format)?,
            !matches!(
                e,
                Encoding::Int {
                    signed: true,
                    normalized: false,
                    ..
                }
            ),
        )),
        ClassMatch::IntToFloat => vk_scaled_vertex_format(format)
            .map(|f| (f, true))
            .ok_or_else(|| {
                Error::Internal(InternalError::new(
                    InternalKind::Invariant,
                    format!("{format:?} converts to float but has no scaled format"),
                ))
            }),
    }
}

/// The stride the Vulkan binding of `fetch` on `layout` is emitted with (`None`: set per bind, a
/// runtime fact the binder owns): **0** for Metal's constant step, which `metal::vertex::fetch_as`
/// answers with a divisor of 0 and no other step can, and `layout`'s own otherwise.
///
/// A zero divisor fetches element `firstInstance` ("Vertex Input Extraction"), and Metal's
/// constant step fetches the layout's first element whatever the base instance, so the binding
/// must read the same bytes at every base instance: `firstInstance * 0`. The layout's own stride
/// is not key data of a fetch that reads one element, and every limit below reads the stride the
/// binding carries rather than the layout's, so a constant-step binding meets them as a stride of
/// 0 does.
fn binding_stride(fetch: AttributeFetch, layout: &VertexBufferLayoutDescriptor) -> Option<u32> {
    match fetch {
        AttributeFetch::FixedFunction { divisor: 0, .. } => Some(0),
        _ => layout.stride,
    }
}

/// Whether Vulkan's fixed-function attribute address can be aligned for `format` at `offset` in
/// a layout of static stride `stride` (`None`: set per bind, a runtime fact the binder owns).
fn aligned(format: VertexFormat, offset: u32, stride: Option<u32>) -> Result<bool, Error> {
    let a = vertex_alignment(format)?;
    Ok(offset.is_multiple_of(a) && stride.is_none_or(|s| s.is_multiple_of(a)))
}

/// One attribute with its descriptor entries and per-attribute fetch.
struct Resolved<'a> {
    attr: &'a VertexAttributeDescriptor,
    layout: &'a VertexBufferLayoutDescriptor,
    fetch: AttributeFetch,
}

/// Every attribute of `attributes` resolved (items 1-8 of the module docs; item 8 when
/// `pull_every` is set), and the layouts of `variant` that item 7 moved (those that had a
/// fixed-function attribute before it). Any attribute of `desc` whose buffer has no layout, read
/// or not, is `InvalidDescriptor(VertexLayout)` first (`VertexDescriptor::check_layouts`).
fn resolve<'a>(
    desc: Option<&'a VertexDescriptor>,
    attributes: &BTreeMap<u32, IoNumeric>,
    p: &DeviceProfile,
    variant: &BTreeSet<u32>,
    pull_every: bool,
) -> Result<(BTreeMap<u32, Resolved<'a>>, BTreeSet<u32>), Error> {
    let l = &p.fields().limits;
    if let Some(d) = desc {
        d.check_layouts()?;
    }
    let mut out = BTreeMap::new();
    for (&a, &numeric) in attributes {
        let attr = desc
            .and_then(|d| d.attributes.get(&a))
            .ok_or(Error::InvalidDescriptor(DescriptorError::VertexAttribute {
                attribute: a,
            }))?;
        let layout = desc.and_then(|d| d.layouts.get(&attr.buffer_index)).ok_or(
            Error::InvalidDescriptor(DescriptorError::VertexLayout {
                layout: attr.buffer_index,
            }),
        )?;
        let (vk_format, fits) = class_format(attr.format, numeric, a)?;
        let mut fetch = fetch_as(vk_format, layout.step_function, layout.step_rate, p)?;
        let emitted = binding_stride(fetch, layout);
        let beyond_stride = match (fetch, emitted) {
            (AttributeFetch::FixedFunction { format, .. }, Some(stride)) => {
                let size = format.block().map_or(u32::MAX, |b| b.size);
                attr.offset.saturating_add(size) > stride
            }
            _ => false,
        };
        let pulled = !fits
            || (beyond_stride
                && p.fields()
                    .enabled
                    .portability_withholds(VkFeature::VertexAttributeAccessBeyondStride))
            || !aligned(attr.format, attr.offset, emitted)?
            || emitted.is_some_and(|s| !l.portable_vertex_stride(s))
            || attr.offset > l.max_vertex_input_attribute_offset
            || emitted.is_some_and(|s| s > l.max_vertex_input_binding_stride);
        if pulled {
            fetch = AttributeFetch::Pulled;
        }
        out.insert(
            a,
            Resolved {
                attr,
                layout,
                fetch,
            },
        );
    }
    // Pull the highest fixed-function attribute while the fixed-function attributes or their
    // layouts exceed the device's limits.
    loop {
        let fixed: Vec<&Resolved<'_>> = out
            .values()
            .filter(|r| matches!(r.fetch, AttributeFetch::FixedFunction { .. }))
            .collect();
        let layouts: BTreeSet<u32> = fixed.iter().map(|r| r.attr.buffer_index).collect();
        if fixed.len() as u64 <= u64::from(l.max_vertex_input_attributes)
            && layouts.len() as u64 <= u64::from(l.max_vertex_input_bindings)
        {
            break;
        }
        match out
            .values_mut()
            .rev()
            .find(|r| matches!(r.fetch, AttributeFetch::FixedFunction { .. }))
        {
            Some(r) => r.fetch = AttributeFetch::Pulled,
            None => break,
        }
    }
    // Static strides under dynamic stride state cover their attributes (item 6).
    let dynamic = strides_dynamic(
        p,
        out.values()
            .filter(|r| matches!(r.fetch, AttributeFetch::FixedFunction { .. }))
            .map(|r| binding_stride(r.fetch, r.layout)),
    );
    if dynamic {
        for r in out.values_mut() {
            if let (AttributeFetch::FixedFunction { format, .. }, Some(stride)) =
                (r.fetch, binding_stride(r.fetch, r.layout))
            {
                let size = format.block().map_or(u32::MAX, |b| b.size);
                if stride != 0 && r.attr.offset.saturating_add(size) > stride {
                    r.fetch = AttributeFetch::Pulled;
                }
            }
        }
    }
    // The pulled-vertex-buffer variant (item 7).
    let mut moved = BTreeSet::new();
    for r in out.values_mut() {
        if matches!(r.fetch, AttributeFetch::FixedFunction { .. })
            && variant.contains(&r.attr.buffer_index)
        {
            r.fetch = AttributeFetch::Pulled;
            moved.insert(r.attr.buffer_index);
        }
    }
    // The de-indexed and private list forms (item 8).
    if pull_every {
        for r in out.values_mut() {
            r.fetch = AttributeFetch::Pulled;
        }
    }
    Ok((out, moved))
}

/// Whether a render pipeline's vertex strides are dynamic state
/// (`VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE`) on `p` for fixed-function bindings of
/// `strides` ([`binding_stride`]; `None`: set per bind): the profile has dynamic strides (extended
/// dynamic state) and some binding's stride is set per bind. Link's graphics state declares the
/// dynamic state exactly then, and the projection checks static strides against it (item 6).
pub(crate) fn strides_dynamic(
    p: &DeviceProfile,
    mut strides: impl Iterator<Item = Option<u32>>,
) -> bool {
    p.fields().dynamic_vertex_stride && strides.any(|s| s.is_none())
}

/// How each attribute in `attributes` (the Metal attribute indices a vertex function reads, with
/// the numeric class it declares each as) is fetched on `p` for vertex descriptor `desc` (see
/// the module docs). An attribute the descriptor lacks is an invalid descriptor, and so is any
/// attribute whose layout it lacks, read or not (Metal stops at an assertion for each); the
/// attributes the descriptor has but the function does not read are not projected (they are not
/// key data). [`vertex_fetch_projection_for`] without a variant.
#[cfg(test)]
pub fn vertex_fetch_projection(
    desc: Option<&VertexDescriptor>,
    attributes: &BTreeMap<u32, IoNumeric>,
    p: &DeviceProfile,
) -> Result<VertexFetchProjection, Error> {
    vertex_fetch_projection_for(desc, attributes, p, &BTreeSet::new(), false)
}

/// `vertex_fetch_projection` under a pulled-vertex-buffer variant and a form that pulls every
/// attribute: the attributes of the layouts in `pulled_vertex_buffers`
/// (`VariantKey::pulled_vertex_buffers`) are pulled (item 7), and every attribute when
/// `pull_every` (the de-indexed and private list forms, item 8).
pub fn vertex_fetch_projection_for(
    desc: Option<&VertexDescriptor>,
    attributes: &BTreeMap<u32, IoNumeric>,
    p: &DeviceProfile,
    pulled_vertex_buffers: &BTreeSet<u32>,
    pull_every: bool,
) -> Result<VertexFetchProjection, Error> {
    let mut out = VertexFetchProjection::new();
    for (a, r) in resolve(desc, attributes, p, pulled_vertex_buffers, pull_every)?.0 {
        out.attributes.insert(a, fetch_of(&r));
    }
    Ok(out)
}

/// The projection entry of a resolved attribute: fixed function, or the pulled fetch of its
/// format, offset and step.
fn fetch_of(r: &Resolved<'_>) -> VertexFetch {
    match r.fetch {
        AttributeFetch::FixedFunction { .. } => VertexFetch::FixedFunction,
        _ => VertexFetch::pulled(r.attr, r.layout),
    }
}

/// What the device's fixed-function operations give a render pipeline: the precision notes one of
/// its stages records and the measured properties it relies on in their place. The conversions of
/// its vertex stage's fixed-function attributes ([`fixed_function_notes`]) and its alpha to
/// coverage (`graphics::coverage_notes`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct FixedFunctionNotes {
    /// The notes, sorted, each once.
    pub(crate) notes: Vec<PrecisionNote>,
    /// The measured properties an attribute that records no note relies on.
    pub(crate) properties: BTreeSet<VkProperty>,
}

/// The precision notes of vertex stage `vs`'s attributes that fixed-function vertex input fetches
/// for descriptor `desc` on `p` (its `IoRole::Attribute` inputs, which [`vertex_input_state`]
/// holds to the descriptor's projection), sorted, each once, and the measured properties the
/// pipeline requires in place of a note: the device converts each attribute's format, and the
/// module, which every such format shares, cannot say what it was.
///
/// - `PrecisionNote::DeviceVertexConversion` for a normalized format (the 8- and 16-bit
///   `Normalized` formats of either sign, `UChar4Normalized_BGRA`, `Int1010102Normalized`,
///   `UInt1010102Normalized`), whatever width the function reads it at: the device's `UNORM` or
///   `SNORM` conversion of the codes, where Metal's quotient is correctly rounded (but an
///   `Int1010102Normalized` 10-bit field's, the code times `float(1 / 511)`). Where `p` takes
///   `DriverDeviation::MetalNormalizedVertexFetch`, the device's `float` is Metal's at every code:
///   an attribute the function reads as a 32-bit `float` records no note and the pipeline requires
///   `VkProperty::MetalNormalizedVertexFetch`, and so does one read as `half` of an 8- or 10-bit
///   format, whose `float` rounds to Metal's `half` at every code. A 16-bit format read as `half`
///   ([`rounds_twice_to_half`]) records the note there too;
/// - `PrecisionNote::DeviceNanEncoding` for a float format narrower than 32 bits
///   (`Encoding::widens_narrow_float`) that the function reads as a 32-bit `float`
///   (`StageIo::f32_attributes`): Metal widens its NaN to `0x7fc00000`, the device's widening
///   need not. A `half`-declared attribute keeps the fetched code's own bits, as Metal's does.
///
/// A pulled attribute's decode is the module's and records its own notes. An attribute the
/// descriptor lacks is [`vertex_input_state`]'s refusal, which runs first.
pub(crate) fn fixed_function_notes(
    vs: &StageInterface,
    desc: Option<&VertexDescriptor>,
    p: &DeviceProfile,
) -> Result<FixedFunctionNotes, Error> {
    let f32_reads = vs.io.f32_attributes();
    let metal = p.takes_driver_deviation(DriverDeviation::MetalNormalizedVertexFetch);
    let mut notes = BTreeSet::new();
    let mut properties = BTreeSet::new();
    for (a, (_, fetch)) in vs.io.attributes() {
        if fetch != VertexFetch::FixedFunction {
            continue;
        }
        let attr = desc
            .and_then(|d| d.attributes.get(&a))
            .ok_or(Error::InvalidDescriptor(DescriptorError::VertexAttribute {
                attribute: a,
            }))?;
        let e = encoding(attr.format)?;
        let f32_read = f32_reads.contains(&a);
        if e.is_normalized() {
            if metal && (f32_read || !rounds_twice_to_half(e)) {
                properties.insert(VkProperty::MetalNormalizedVertexFetch);
            } else {
                notes.insert(PrecisionNote::DeviceVertexConversion);
            }
        }
        if e.widens_narrow_float() && f32_read {
            notes.insert(PrecisionNote::DeviceNanEncoding);
        }
    }
    Ok(FixedFunctionNotes {
        notes: notes.into_iter().collect(),
        properties,
    })
}

/// Whether normalized encoding `e`'s `float` quotient, rounded to `half` as the module rounds an
/// attribute the function reads as `half`, can differ from Metal's `half`, which rounds the
/// quotient once: a 16-bit field's can land on a `half` midpoint (codes 65455 and 65519 unsigned,
/// and 32759 and -32759 signed, each one `half` ulp from Metal's), an 8-, 10- or 2-bit field's
/// at no code (`Int1010102Normalized`'s 10-bit fields through Metal's reciprocal product too).
fn rounds_twice_to_half(e: Encoding) -> bool {
    matches!(
        e,
        Encoding::Int {
            bits: 16,
            normalized: true,
            ..
        }
    )
}

/// The vertex input state of vertex stage `vs` for descriptor `desc` under the
/// pulled-vertex-buffer variant `pulled_vertex_buffers`, every attribute pulled where `pull_every`
/// (the de-indexed and private list forms; see the module docs), with each fixed-function
/// attribute at the location link assigned its key. Fixed-function bindings are numbered densely
/// in Metal buffer-index order; `variant_pulled` records the layouts the variant moved.
pub fn vertex_input_state(
    vs: &StageInterface,
    desc: Option<&VertexDescriptor>,
    (pulled_vertex_buffers, pull_every): (&BTreeSet<u32>, bool),
    locations: &Locations,
    p: &DeviceProfile,
) -> Result<VertexInputState, Error> {
    let used = vs.io.attributes();
    let numeric: BTreeMap<u32, IoNumeric> = used.iter().map(|(a, (n, _))| (*a, *n)).collect();
    let (resolved, moved) = resolve(desc, &numeric, p, pulled_vertex_buffers, pull_every)?;
    for (a, (_, fetch)) in &used {
        // The stage was compiled under another projection (a `Compiled` stage source whose
        // `LowerInputs` do not fit this descriptor).
        if resolved.get(a).map(fetch_of) != Some(*fetch) {
            return Err(Error::InvalidDescriptor(DescriptorError::VertexAttribute {
                attribute: *a,
            }));
        }
    }
    // The varyings the stage writes alike at every vertex of a primitive hold only while the
    // attributes it took as stepped per instance are.
    for a in &vs.io.instanced_attributes {
        let per_instance = resolved.get(a).is_some_and(|r| match r.fetch {
            AttributeFetch::FixedFunction { input_rate, .. } => {
                input_rate == crate::vk::VertexInputRate::INSTANCE
            }
            AttributeFetch::Pulled => false,
        });
        if !per_instance {
            return Err(Error::InvalidDescriptor(DescriptorError::VertexAttribute {
                attribute: *a,
            }));
        }
    }
    let mut state = VertexInputState {
        variant_pulled: moved,
        ..VertexInputState::default()
    };
    let mut binding_of: BTreeMap<u32, u32> = BTreeMap::new();
    for r in resolved.values() {
        if let AttributeFetch::FixedFunction {
            input_rate,
            divisor,
            ..
        } = r.fetch
        {
            if let std::collections::btree_map::Entry::Vacant(e) =
                binding_of.entry(r.attr.buffer_index)
            {
                e.insert(0);
                state.bindings.push(VertexBinding {
                    binding: 0,
                    buffer_index: r.attr.buffer_index,
                    input_rate,
                    divisor,
                    stride: match binding_stride(r.fetch, r.layout) {
                        Some(s) => VertexStride::Static(s),
                        None => VertexStride::PerBind,
                    },
                });
            }
        }
    }
    state.bindings.sort_by_key(|b| b.buffer_index);
    for (n, b) in state.bindings.iter_mut().enumerate() {
        // At most `maxVertexInputBindings` (a u32) entries, so the index fits.
        b.binding = u32::try_from(n).unwrap_or(u32::MAX);
        binding_of.insert(b.buffer_index, b.binding);
    }
    for (&a, r) in &resolved {
        match r.fetch {
            AttributeFetch::FixedFunction { format, .. } => {
                let location = locations
                    .get(&VaryingKey::attribute(a))
                    .map(|l| l.0)
                    .ok_or_else(|| {
                        Error::Internal(InternalError::new(
                            InternalKind::Invariant,
                            format!("fixed-function attribute {a} has no location"),
                        ))
                    })?;
                state.attributes.push(VertexAttribute {
                    location,
                    binding: binding_of.get(&r.attr.buffer_index).copied().unwrap_or(0),
                    format,
                    offset: r.attr.offset,
                })
            }
            _ => state.pulled.push(PulledAttribute {
                attribute: a,
                buffer_index: r.attr.buffer_index,
                format: r.attr.format,
                offset: r.attr.offset,
                step_function: r.layout.step_function,
                step_rate: r.layout.step_rate,
                stride: match r.layout.stride {
                    Some(s) => VertexStride::Static(s),
                    None => VertexStride::PerBind,
                },
            }),
        }
    }
    Ok(state)
}
