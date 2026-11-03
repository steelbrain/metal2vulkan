//! Device tests of framebuffer fetch (its fourth part) on the device the cfg selects, under the
//! validation layer.
//!
//! Metal orders a fragment function's read of its color attachment after every earlier fragment
//! of the pixel, within a draw and across draws, whichever pipeline drew them. Each pass here
//! draws the fullscreen triangle (`rt6_resolves`'s `rt6_resolve_full`) once a step over a 4²
//! target and reads every texel back; each pass's answer follows from its steps only in that
//! order:
//!
//! - **Across pipelines**: an `R32Float` target cleared to 0.75, then `fetch_scalar`
//!   (`fragment_fetch`, half what it reads), `rt6_resolve_white` (1), `fetch_scalar` twice:
//!   0.25. A read of the clear in each fetch leaves 0.375; a fetch that missed the white write,
//!   0.09375.
//! - **Integer read-modify-writes**: an `RGBA32Uint` target cleared to (1, 2, 3, 4), then
//!   `exf_fetch_count` (three times what it reads, plus `k + 1` in lane `k`) three times:
//!   `c · 27 + (k + 1) · 13`.
//! - **A memoryless attachment fetched**: `RGBA32Uint` targets, the stored 0 cleared to
//!   (1, 2, 3, 4) and the memoryless 1 to (5, 6, 7, 8), then `exf2_fetch_unwritten` (three times
//!   what it reads of 0, plus what it reads of 1, into 0) three times: `c · 27 + t · 13`, which
//!   holds only where each fetch of the memoryless attachment reads its clear. Three command
//!   buffers of two such passes, over fresh memoryless attachments, the second loaded `DontCare`
//!   (`c · 27`), make one backing between them where the device renders the attachment into an
//!   image of its own, and passes of 12 sizes, rendered twice in turn, one backing for each size.
//!   And a memoryless attachment loaded `DontCare` that `exf2_fetch_unwritten` reads, then
//!   `mli_tag_step` (`mli_memoryless_step`: what it reads plus (1, 2, 3, 4)) steps twice, then
//!   `exf2_fetch_unwritten` folds in, in two passes in turn: `c · 11` each, which holds only where
//!   each pass's tag starts from zero, whatever an earlier pass left in its backing.
//!
//! - **Load, store, and later passes**: a 16² `R32Float` texture cleared to 0.75 and stored,
//!   then loaded and halved by `fetch_scalar` (0.375, stored), sampled by `a4_pixel_fetch` in a
//!   compute pass after it, halved again and stored `DontCare`, and halved once more through a
//!   texture view of it, stored. 16² is large enough that `STORAGE` costs the texture its
//!   compression on Apple's GPU (from 16² up; 8² and smaller have none to lose), so MoltenVK
//!   renders it through a scratch image.
//!
//! - **Clears of fetched attachments**: a fetched attachment the function only reads
//!   (`exf2_fetch_unwritten`'s tag, of each unsigned format; `fetch_blend`'s third attachment,
//!   of each float format), cleared and stored by a pass of one draw, holds the texels a pass
//!   that only clears a texture of its format stores: in the interlock form the clear runs
//!   before the pass (`vkCmdClearColorImage`), not as the attachment's load operation.
//!
//! - **Plain draws before a fetch**: beside a memoryless attachment the pipelines never write,
//!   `rt6_resolve_white` then `fetch_scalar` in one pass: in the pixel interlock it splits at the
//!   fetch, on every device.
//!
//! - **A masked fetched write**: `con_fetch_coverage` reads an `RG16Float` attachment and writes it
//!   under a mask of green alone, beside a blended attachment and one it never writes, with a depth
//!   and stencil attachment: red keeps its clear, green follows the draws in order, and the
//!   unwritten attachment keeps its clear. Then under a mask of red alone, a state of its own:
//!   green keeps its clear and red follows the draws; in the pixel interlock both masks share one
//!   module, whose draws name the channel kept.
//!
//! - **Visibility results across a split**: an `R32Float` target cleared to 0.75, with
//!   `Boolean` results at 0, 8 and 16 of a buffer of a fixed pattern, and `rt6_resolve_white`
//!   blended `One`, `OneMinusSourceAlpha` (1 over any finite texel, as the plain write stores),
//!   whose blend keeps it out of the interlock's storage form. At 0, the white then `fetch_scalar`;
//!   at 8, the white clipped (its depth past the far plane), then `fetch_scalar`; at 16,
//!   `fetch_scalar` then the white, both clipped; then, where the device has precise occlusion
//!   queries, a `Counting` result at 24 of the white then `fetch_scalar`; then `Disabled`. Metal
//!   writes 1, 1, 0 and 32 (16 samples a draw; without precise queries 24 keeps its bytes), and the
//!   target holds 0.5 (0.25). In the interlock form each change between the blended and a fetching
//!   draw splits the pass, so 0, 8 and 24 span two, three and two render pass instances, and only
//!   the last of 8's has a sample: its result holds only where the continuation queries are
//!   combined, and 24's only where they are summed.
//!
//! Each pass runs on every device and gives that answer, through input attachments in
//! rasterization order where the device has that access (lavapipe), else through storage
//! attachments in the pixel interlock (RADV, ANV, NVIDIA and MoltenVK): directly where the
//! texture holds `STORAGE`, or through a scratch image standing in for it where `STORAGE` would
//! cost the texture its compression. A pass that stores `DontCare` leaves either its rendered
//! texels or the texture's earlier ones (Metal's are undefined; with a scratch the texture keeps
//! its own); which is printed.
//!
//! - **A multisampled fetch**: the across-pipelines steps at 4x, into a memoryless target
//!   resolved into a 4² texture. Every sample takes the steps, so the resolve holds 0.25. The
//!   compile layer reads the attachment per sample in either form; what each device does with it is
//!   printed:
//!   - through input attachments (lavapipe), the runtime draws it: Metal's answer;
//!   - in the pixel interlock without multisampled storage images (ANV, MoltenVK), the compile
//!     layer reads the samples as sample planes, which the runtime keeps in an array image,
//!     splitting the memoryless attachment's backing into them and merging them back around the
//!     white draw, and resolving after the last merge in one more render pass instance
//!     (`runtime::sample_planes`): Metal's answer (`planes.rs` covers each copy format, a load and
//!     a store);
//!   - in the pixel interlock with them (RADV, NVIDIA), the compile layer reads and writes the
//!     attachment's own multisampled image: the memoryless attachment's backing is a multisampled
//!     image with `STORAGE`, which the draws that fetch leave out of their subpass, and one more
//!     render pass instance after the last resolves it: Metal's answer.
//!
//! - **A caller's multisampled image without the fetch usage**: a caller's 4x `R32_SFLOAT`
//!   image of color attachment and transfer usage alone, cleared to 0.75 and stored, then
//!   loaded, halved by `fetch_scalar`, stored and resolved (0.375), then loaded and resolved
//!   alone (0.375 again, what the fetching pass stored). It lacks the usage every form needs of
//!   a multisampled attachment (input attachment usage; `STORAGE`; the sample planes' copies'
//!   sampled usage and view), so the fetching pass renders into a 4x scratch image that has it,
//!   copied in before the pass and back after it at the same sample count, on every venue.
//!
//! - **Commands of an indirect command buffer**: an `R32Float` target
//!   cleared to 0.75, then the encoder's draws, an execution of render commands of their own
//!   pipelines (made with `supportIndirectCommandBuffers`) and the encoder's draws after it, each a
//!   `fetch_scalar` or `rt6_resolve_white`. Metal draws a command that fetches as the encoder's own
//!   draw of it. In the pixel interlock each command that fetches runs in a render pass instance
//!   that leaves the attachment out, a run of its execution (`render/executions.rs`), and the next
//!   plain draw in another of the pass's shape. Where the device fetches the texture through a
//!   scratch image, such a command's run is copied in before and back after; through input
//!   attachments too, its run's framebuffer holding the scratch in place of the texture, where it
//!   was refused `NotImplemented(IndirectRenderCommands)`.
//! - **Commands fetching a caller's image**: the same over a caller's `R32_SFLOAT` image of color
//!   attachment and transfer usage alone, which every fetch form reaches through a scratch: through
//!   input attachments (lavapipe) each fetching command's run renders the scratch a command's draw
//!   made, copied in before it and back after.
//! - **A blended 16-bit float target** (`blend16.rs`): an `RGBA16Float` target blended one over
//!   one minus the source alpha, fetched by `fbl_fetch_blend16`'s functions, which return a loaded
//!   `half4` or `float4`: Metal's blend in every channel. In the pixel interlock
//!   the function computes it, no Vulkan blend applying to the storage write.
//! - **A BGRA-order target** (`bgra.rs`): a `BGRA8Unorm` target (and `RGBA8Unorm` beside it)
//!   whose channels take every byte, halved by `fetch_ordered` one to three times in a pass,
//!   single-sampled and at 4x, under a mask of every channel and of red and alpha, beside an
//!   unwritten `RGBA16Float` color 1: Metal's bytes (the M3 probe of `bgra.rs`), but within one
//!   code where the pipeline records `device-color-conversion` ([`device_rounded`]). In the pixel
//!   interlock on a device that reads `B8G8R8A8_UNORM` without a format in neither direction
//!   (ANV), the attachment is read and written through its `R8G8B8A8_UNORM` view.
//! - **An sRGB target** (`srgb.rs`): a `BGRA8Unorm_sRGB` target whose channels take every byte,
//!   halved by `fetch_ordered` one to three times in a pass, with and without a depth and stencil
//!   attachment that passes or fails half the draws' fragments: Metal's bytes, but within one code
//!   where the pipeline records `device-color-conversion` ([`device_rounded`]).
//!   In the pixel interlock the attachment is read and written through its `R32_UINT` view, in a
//!   scratch image made mutable to it.
//! - **A memory-writing function beside depth** (`beside_depth.rs`): `fetch_coherent` counts each
//!   fragment in a buffer beside a `Depth32Float_Stencil8` attachment under six depth and stencil
//!   states. Metal counts every fragment and keeps the color of those its tests drop. Through input
//!   attachments every state is Metal's answer; in the pixel interlock the states that drop no
//!   fragment are, and the others are refused at the draw at the depth attachment, typed.
//! - **Multisampled commands** (Metal draws them as the encoder's own
//!   draws): the same at 4x, a command that fetches after the encoder's own fetch, one before it,
//!   and two about a plain command, resolved into a 4² texture (0.125, 0.25 and 0.25). Where the
//!   device reads the attachment as sample planes (ANV, MoltenVK) each command's run has the
//!   planes' split before it and merge after it.

mod beside_depth;
mod bgra;
mod blend16;
mod blend8;
mod coverage_chain;
mod generic_blend;
mod planes;
mod returned_depth;
mod returned_stencil;
mod shader_depth;
mod srgb;
mod stencil_stand_in;
mod storage_forms;
pub(super) use storage_forms::STORAGE_FORMS;

pub(super) use planes::PLANES;
pub(super) use shader_depth::SHADER_DEPTH;

use super::degrade_notices::{assert_made_once, forced, made_render_kinds};
use super::degrade_notices::{resolve_kind, shared_in_system_memory};
use super::memoryless::{memoryless, memoryless_msaa};
use super::presenting::{image_info, made_image, shared_among};
use super::resolves::{lane, quiet, Lane, Shape, FULL, ONE, WHITE};
use super::Messages;
use crate::air_library::AirLibrary;
use crate::runtime_lane::{buffer_with, pipelines_settled, read_buffer, WAIT};
use metal2vulkan::device::{DriverFact, VkFeature};
use metal2vulkan::metal::{
    BlendFactor, IndexType, IndirectCommandType, PixelFormat, PrimitiveType, TextureType,
    TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    BlitOption, Buffer, ClearColor, Degradation, Error, IndirectCommandBufferDescriptor, Library,
    LoadAction, PipelineOptions, RenderPassDescriptor, RenderPipelineState, ResourceOptions, Size,
    StoreAction, Texture, TextureDescriptor, VisibilityResultMode, VkImageLayouts,
};
use std::collections::BTreeSet;
use std::path::Path;

/// The fullscreen triangle's depth.
const Z: f32 = 0.5;

/// The texels of the 4² targets.
const TEXELS: usize = 16;

/// The engine fixture `stem`'s library on `lane`'s device.
fn library(lane: &Lane, stem: &str) -> Library {
    let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/public/engine")
        .join(format!("{stem}.ll"));
    let bitcode = llvm_assemble(&ll).unwrap();
    lane.device.new_library_with_air(&bitcode).unwrap()
}

/// A pipeline of the fullscreen triangle (of `vertices`) and `fragment` of `library`, drawing
/// `colors` (the attachment of each index) at `samples`.
fn pipeline(
    lane: &Lane,
    vertices: &Library,
    (library, fragment): (&Library, &str),
    colors: &[PixelFormat],
    samples: u32,
) -> Result<RenderPipelineState, Error> {
    let descriptor = pipeline_descriptor(vertices, (library, fragment), colors, samples);
    lane.device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
}

/// The descriptor of [`pipeline`]'s pipeline.
fn pipeline_descriptor(
    vertices: &Library,
    (library, fragment): (&Library, &str),
    colors: &[PixelFormat],
    samples: u32,
) -> RenderPipelineDescriptor {
    let function = |library: &Library, name| {
        library
            .new_function_with_name(name)
            .unwrap()
            .function_descriptor()
    };
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(function(vertices, FULL)));
    descriptor.fragment = Some(StageSource::Function(function(library, fragment)));
    for (index, &pf) in colors.iter().enumerate() {
        descriptor.color_attachments[index] = Some(ColorAttachmentDescriptor::new(pf));
    }
    descriptor.raster_sample_count = samples;
    descriptor
}

/// The kinds the makings of the pipelines of `descriptors` announce, and the `Shared` buffers
/// of the passes' draws and reads: what a test announces besides its own kinds.
fn made(lane: &Lane, descriptors: &[RenderPipelineDescriptor]) -> BTreeSet<Degradation> {
    let mut first = forced(&[(
        Degradation::SharedInSystemMemory,
        shared_in_system_memory(&lane.device),
    )]);
    for descriptor in descriptors {
        first.extend(made_render_kinds(&lane.device, descriptor));
    }
    first
}

/// Whether the notices `messages` holds, which it keeps, announce a fetched texture rendered
/// through a scratch image (`fetch-through-scratch`): where the device lays its image out
/// otherwise with the usage the fetch needs, its own answer, which the lane does not foresee. A
/// texture of the runtime's own memory then takes its storage form at its pass's commit
/// (`fetched-target-storage`, `storage_forms.rs`).
fn scratched(messages: &Messages) -> bool {
    let notices = messages.take_notices();
    let scratch = notices
        .iter()
        .filter_map(|message| message.degradation.as_ref())
        .any(|notice| notice.degradation == Degradation::FetchThroughScratch);
    messages.restore_notices(notices);
    scratch
}

/// A texel or sample that differs from Metal's: its index, its bytes and Metal's.
type Differing = (usize, [u8; 4], [u8; 4]);

/// The texels of `wrong` that differ from Metal's beyond the device's
/// conversion where the pipeline that stored them records `PrecisionNote::DeviceColorConversion`
/// (`device`): near a tie between two codes the device may store the other one, so a texel whose
/// every channel lies within one code of Metal's holds there. With the count of the channels that
/// hold so.
fn device_rounded(wrong: Vec<Differing>, device: bool) -> (Vec<Differing>, usize) {
    if !device {
        return (wrong, 0);
    }
    let mut rounded = 0;
    let beyond = wrong
        .into_iter()
        .filter(|(_, read, want)| {
            let near = read.iter().zip(want).all(|(r, w)| r.abs_diff(*w) <= 1);
            if near {
                rounded += read.iter().zip(want).filter(|(r, w)| r != w).count();
            }
            !near
        })
        .collect();
    (beyond, rounded)
}

/// `rt6_resolve_white` of `colors`, color 0 blended `One`, `OneMinusSourceAlpha`: 1 over any
/// finite texel, as the plain write stores, through a blend the interlocked storage form does
/// not build, so in the pixel interlock a pass that fetches color 0 after it splits there.
fn blended_white(lane: &Lane, vertices: &Library, colors: &[PixelFormat]) -> RenderPipelineState {
    lane.device
        .new_render_pipeline_state(
            &blended_descriptor(vertices, colors),
            &PipelineOptions::default(),
        )
        .unwrap()
}

/// The descriptor of [`blended_white`]'s pipeline.
fn blended_descriptor(vertices: &Library, colors: &[PixelFormat]) -> RenderPipelineDescriptor {
    let function = |name| {
        vertices
            .new_function_with_name(name)
            .unwrap()
            .function_descriptor()
    };
    let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(function(FULL)));
    descriptor.fragment = Some(StageSource::Function(function(WHITE)));
    for (index, &pf) in colors.iter().enumerate() {
        descriptor.color_attachments[index] = Some(ColorAttachmentDescriptor::new(pf));
    }
    if let Some(color) = descriptor.color_attachments[0].as_mut() {
        color.blending_enabled = true;
        color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
        color.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
    }
    descriptor
}

/// Whether the device's framebuffer fetch reads input attachments: it has rasterization-order
/// color attachment access, which the compile layer orders such reads with.
pub(super) fn through_inputs(lane: &Lane) -> bool {
    lane.device
        .compile_device()
        .profile()
        .fragment()
        .rasterization_order_color_access
}

/// Whether the device announces a split the interlock alone makes (`render-pass-split`, and
/// `memoryless-backed` for it): in the pixel interlock, but where its driver ran such a pass no
/// slower split than whole (`DriverFact::InterlockedPassSplitNoSlower`), its faster form.
fn announces_interlock_splits(lane: &Lane) -> bool {
    !through_inputs(lane)
        && !lane
            .device
            .properties()
            .driver_facts
            .contains(&DriverFact::InterlockedPassSplitNoSlower)
}

/// Draw `steps` in one pass of `descriptor`, each the fullscreen triangle; the first refusal of
/// a step's pipeline or draw, if any, after which the pass ends with what was drawn.
fn steps(
    lane: &Lane,
    descriptor: &RenderPassDescriptor,
    steps: &[&RenderPipelineState],
) -> Result<(), Error> {
    let mut refused = None;
    lane.run(descriptor, |lane, encoder, held: &mut Vec<Buffer>| {
        let depth = buffer_with(&lane.device, &Z.to_le_bytes()).unwrap();
        encoder.set_vertex_buffer(Some(&depth), 0, 0).unwrap();
        for step in steps {
            let drawn = encoder
                .set_render_pipeline_state(step)
                .and_then(|()| encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
            if let Err(error) = drawn {
                refused = Some(error);
                break;
            }
        }
        held.push(depth);
    })?;
    refused.map_or(Ok(()), Err)
}

/// A pass of `targets` (the attachment of each index), each cleared to its color and stored by
/// its action, the first resolved into `resolve` where it is one.
fn pass(
    targets: &[(&Texture, ClearColor, StoreAction)],
    resolve: Option<&Texture>,
) -> RenderPassDescriptor {
    let mut descriptor = RenderPassDescriptor::new();
    for (index, &(texture, clear, store)) in targets.iter().enumerate() {
        let target = &mut descriptor.color_attachments[index];
        target.texture = Some(texture.clone());
        target.load_action = LoadAction::Clear;
        target.clear_color = clear;
        target.store_action = store;
    }
    descriptor.color_attachments[0].resolve_texture = resolve.cloned();
    descriptor
}

/// A one-target pass of `texture` that loads it and stores it by `store`.
fn loading(texture: &Texture, store: StoreAction) -> RenderPassDescriptor {
    let mut descriptor = RenderPassDescriptor::new();
    let target = &mut descriptor.color_attachments[0];
    target.texture = Some(texture.clone());
    target.load_action = LoadAction::Load;
    target.store_action = store;
    descriptor
}

/// The form the device's framebuffer fetch takes, for the printed lines.
fn form(lane: &Lane) -> &'static str {
    if through_inputs(lane) {
        "input attachments"
    } else {
        "the pixel interlock"
    }
}

/// Check one pass's outcome, `ran` (its texels, or the refusal of its pipelines or draws):
/// `texels`, on every device.
fn expect(lane: &Lane, name: &str, ran: Result<Vec<u8>, Error>, texels: &[u8]) {
    let device = lane.device.name();
    match ran {
        Ok(read) => assert_eq!(read, texels, "{name} on {device}"),
        Err(error) => panic!("{name} on {device}: {error}"),
    }
    eprintln!(
        "framebuffer fetch {name} on {device} through {}: Metal's ordered answer",
        form(lane)
    );
}

/// Whether the device's framebuffer fetch reads a multisampled pass's attachments as sample
/// planes: in the pixel interlock, without multisampled storage images (`planes.rs`).
fn reads_planes(lane: &Lane) -> bool {
    let profile = lane.device.compile_device().profile();
    !through_inputs(lane) && !profile.images().storage_multisample
}

/// Check a multisampled pass's outcome, `ran`: `texels`, on every device, by the device's fetch
/// form (the module docs), which is printed.
fn multisampled(lane: &Lane, name: &str, ran: Result<Vec<u8>, Error>, texels: &[u8]) {
    let device = lane.device.name();
    match ran {
        Ok(read) => assert_eq!(read, texels, "{name} on {device}"),
        Err(error) => panic!("{name} on {device}: {error}"),
    }
    let form = if through_inputs(lane) {
        "input attachments"
    } else if reads_planes(lane) {
        "sample planes in the pixel interlock"
    } else {
        "the attachment's multisampled storage image in the pixel interlock"
    };
    eprintln!("framebuffer fetch {name} on {device} through {form}: Metal's ordered answer");
}

/// Each fetch reads what every earlier fragment of the pixel wrote, of whichever pipeline, in a
/// single-sampled pass and, where the device's form is drawn, in a multisampled one (the module
/// docs).
#[test]
fn fetches_read_every_earlier_fragment_of_the_pixel() {
    let (messages, cache, lane) = lane();
    let vertices = library(&lane, "rt6_resolves");
    let fetch = library(&lane, "fragment_fetch");
    let pf = PixelFormat::R32Float;
    let clear = ClearColor::new(0.75, 0.0, 0.0, 0.0);
    let order = |samples| -> Result<Vec<RenderPipelineState>, Error> {
        let halve = pipeline(&lane, &vertices, (&fetch, "fetch_scalar"), &[pf], samples)?;
        let white = pipeline(&lane, &vertices, (&vertices, WHITE), &[pf], samples)?;
        Ok(vec![halve.clone(), white, halve.clone(), halve])
    };
    let ran = (|| {
        let order = order(1)?;
        let target = lane.texture(pf, ONE, TextureUsage::RENDER_TARGET);
        let targets = [(&target, clear, StoreAction::Store)];
        steps(
            &lane,
            &pass(&targets, None),
            &order.iter().collect::<Vec<_>>(),
        )?;
        Ok(lane.read(&target, (0, 0), 4, BlitOption::None))
    })();
    let quarter = 0.25f32.to_le_bytes().repeat(TEXELS);
    expect(&lane, "across pipelines", ran, &quarter);
    let ran = (|| {
        let order = order(4)?;
        let msaa = memoryless_msaa(&lane, pf);
        let resolve = lane.single(pf);
        let targets = [(&msaa, clear, StoreAction::MultisampleResolve)];
        let descriptor = pass(&targets, Some(&resolve));
        steps(&lane, &descriptor, &order.iter().collect::<Vec<_>>())?;
        Ok(lane.read(&resolve, (0, 0), 4, BlitOption::None))
    })();
    multisampled(&lane, "across pipelines, 4x memoryless", ran, &quarter);
    drop(cache);
    quiet(lane, &messages);
}

/// Three integer read-modify-writes of the pixel, one a draw, in order (the module docs).
#[test]
fn integer_read_modify_writes_follow_draw_order() {
    let (messages, cache, lane) = lane();
    let vertices = library(&lane, "rt6_resolves");
    let count = library(&lane, "exf_fetch_count");
    let pf = PixelFormat::RGBA32Uint;
    let ran = (|| {
        let step = pipeline(&lane, &vertices, (&count, "exf_fetch_count"), &[pf], 1)?;
        let target = lane.texture(pf, ONE, TextureUsage::RENDER_TARGET);
        let clear = ClearColor::new(1.0, 2.0, 3.0, 4.0);
        let targets = [(&target, clear, StoreAction::Store)];
        steps(&lane, &pass(&targets, None), &[&step, &step, &step])?;
        Ok(lane.read(&target, (0, 0), 16, BlitOption::None))
    })();
    // Lane `k` starts at `k + 1`: `(k + 1) · 27 + (k + 1) · 13`.
    let texel: Vec<u8> = (1u32..=4).flat_map(|c| (c * 40).to_le_bytes()).collect();
    expect(
        &lane,
        "integer read-modify-writes",
        ran,
        &texel.repeat(TEXELS),
    );
    drop(cache);
    quiet(lane, &messages);
}

/// Each fetch of a memoryless attachment reads its clear (the module docs).
#[test]
fn memoryless_attachments_are_fetched() {
    let (messages, cache, lane) = lane();
    let vertices = library(&lane, "rt6_resolves");
    let unwritten = library(&lane, "exf2_fetch_unwritten");
    let pf = PixelFormat::RGBA32Uint;
    let ran = (|| {
        let fragment = (&unwritten, "exf2_fetch_unwritten");
        let step = pipeline(&lane, &vertices, fragment, &[pf, pf], 1)?;
        let count = lane.texture(pf, ONE, TextureUsage::RENDER_TARGET);
        let tag = memoryless(
            &lane,
            TextureType::Type2D,
            pf,
            ONE,
            TextureUsage::RENDER_TARGET,
        )?;
        let targets = [
            (
                &count,
                ClearColor::new(1.0, 2.0, 3.0, 4.0),
                StoreAction::Store,
            ),
            (
                &tag,
                ClearColor::new(5.0, 6.0, 7.0, 8.0),
                StoreAction::DontCare,
            ),
        ];
        steps(&lane, &pass(&targets, None), &[&step, &step, &step])?;
        Ok(lane.read(&count, (0, 0), 16, BlitOption::None))
    })();
    // Lane `k` counts from `k + 1` with the tag `k + 5`.
    let texel: Vec<u8> = (1u32..=4)
        .flat_map(|c| (c * 27 + (c + 4) * 13).to_le_bytes())
        .collect();
    expect(&lane, "a memoryless attachment", ran, &texel.repeat(TEXELS));
    drop(cache);
    quiet(lane, &messages);
}

/// A memoryless attachment loaded `DontCare` reads zero where a fragment fetches it, as Metal's
/// does (an Apple M4 under macOS 27.0.1: a fetch read 0 in 52 runs of 52, also
/// after a pass that cleared other bytes into the tiles). A first pass clears the memoryless
/// attachment to (5, 6, 7, 8), which a device keeping its memory as it lies would read again;
/// then `exf2_fetch_unwritten` three times over the count cleared to (1, 2, 3, 4) with the
/// memoryless attachment loaded `DontCare`: `c · 27`, the tag read as zero.
#[test]
fn a_dont_care_memoryless_attachment_reads_zero() {
    let (messages, cache, lane) = lane();
    let vertices = library(&lane, "rt6_resolves");
    let unwritten = library(&lane, "exf2_fetch_unwritten");
    let pf = PixelFormat::RGBA32Uint;
    let ran = (|| {
        let fragment = (&unwritten, "exf2_fetch_unwritten");
        let step = pipeline(&lane, &vertices, fragment, &[pf, pf], 1)?;
        let count = lane.texture(pf, ONE, TextureUsage::RENDER_TARGET);
        let tag = memoryless(
            &lane,
            TextureType::Type2D,
            pf,
            ONE,
            TextureUsage::RENDER_TARGET,
        )?;
        let targets = [
            (
                &count,
                ClearColor::new(1.0, 2.0, 3.0, 4.0),
                StoreAction::Store,
            ),
            (
                &tag,
                ClearColor::new(5.0, 6.0, 7.0, 8.0),
                StoreAction::DontCare,
            ),
        ];
        steps(&lane, &pass(&targets, None), &[&step])?;
        let mut unloaded = pass(&targets, None);
        unloaded.color_attachments[1].load_action = LoadAction::DontCare;
        steps(&lane, &unloaded, &[&step, &step, &step])?;
        Ok(lane.read(&count, (0, 0), 16, BlitOption::None))
    })();
    let texel: Vec<u8> = (1u32..=4).flat_map(|c| (c * 27).to_le_bytes()).collect();
    expect(
        &lane,
        "a memoryless attachment loaded DontCare",
        ran,
        &texel.repeat(TEXELS),
    );
    drop(cache);
    quiet(lane, &messages);
}

/// A memoryless attachment loaded `DontCare` that a pass's first draw only reads, then steps,
/// then folds into the stored count: over the count cleared to (1, 2, 3, 4),
/// `exf2_fetch_unwritten` (the tag read as zero), `mli_tag_step` twice (the owned fixture
/// `mli_memoryless_step`: the tag `2 (k + 1)` in lane `k`), `exf2_fetch_unwritten`: `11 c`, in
/// each of two passes in turn, the second of which would read `13 c` where it found what the first
/// left in a backing the queue keeps. Where the device fetches through input attachments, the
/// first draw records inline, and the first step splits the pass before it
/// (`WorkCounts::inline_memoryless_write_splits`): the backing the split brings stands in from
/// there on, cleared to the zero the first instance read, and the instance after the split
/// records inline too (`render/inline_first.rs`, `render/inline_splits.rs`). In the interlock the
/// first draw brings the backing, and the pass records as before.
#[test]
fn a_memoryless_attachment_stepped_after_an_inline_read_starts_from_zero() {
    let (messages, cache, lane) = lane();
    let vertices = library(&lane, "rt6_resolves");
    let unwritten = library(&lane, "exf2_fetch_unwritten");
    let stepping = library(&lane, "mli_memoryless_step");
    let pf = PixelFormat::RGBA32Uint;
    let counted = || {
        let counts = lane.device.work_counts();
        let splits = counts.inline_memoryless_write_splits;
        (counts.inline_passes, splits, counts.inline_split_instances)
    };
    let before = counted();
    let ran = (|| {
        let fold = pipeline(
            &lane,
            &vertices,
            (&unwritten, "exf2_fetch_unwritten"),
            &[pf, pf],
            1,
        )?;
        let step = pipeline(&lane, &vertices, (&stepping, "mli_tag_step"), &[pf, pf], 1)?;
        let count = lane.texture(pf, ONE, TextureUsage::RENDER_TARGET);
        let tag = memoryless(
            &lane,
            TextureType::Type2D,
            pf,
            ONE,
            TextureUsage::RENDER_TARGET,
        )?;
        let targets = [
            (
                &count,
                ClearColor::new(1.0, 2.0, 3.0, 4.0),
                StoreAction::Store,
            ),
            (
                &tag,
                ClearColor::new(0.0, 0.0, 0.0, 0.0),
                StoreAction::DontCare,
            ),
        ];
        let mut unloaded = pass(&targets, None);
        unloaded.color_attachments[1].load_action = LoadAction::DontCare;
        let mut read = Vec::new();
        for _ in 0..2 {
            steps(&lane, &unloaded, &[&fold, &step, &step, &fold])?;
            read.extend(lane.read(&count, (0, 0), 16, BlitOption::None));
        }
        Ok(read)
    })();
    let (inline, splits, later) = counted();
    let texel: Vec<u8> = (1u32..=4).flat_map(|c| (c * 11).to_le_bytes()).collect();
    expect(
        &lane,
        "a memoryless attachment stepped after an inline read",
        ran,
        &texel.repeat(2 * TEXELS),
    );
    let (inline, splits) = (inline - before.0, splits - before.1);
    let later = later - before.2;
    eprintln!(
        "a memoryless attachment stepped after an inline read on {}: {inline} passes inline, \
         {splits} split before a write of it, {later} instances inline after a split",
        lane.device.name()
    );
    if through_inputs(&lane) {
        assert_eq!(
            (inline, splits, later),
            (2, 2, 2),
            "inline, and split before the step into an instance inline too"
        );
    }
    drop(cache);
    quiet(lane, &messages);
}

/// The colors a pass's fetching draws read and write as storage images in the pixel interlock,
/// which their subpass leaves out: the framebuffer of each instance holds a view of a placeholder
/// the device keeps for each such color, never one of the image the draws write
/// (`storage_placeholder.rs`, `WorkCounts::storage_placeholder_attachments`): one per format and
/// sample count and per further color of them a framebuffer leaves out, however many passes take
/// it, and a larger one only for a larger framebuffer (`WorkCounts::storage_placeholders_made`).
/// Over a count cleared to (1, 2, 3, 4) and a tag cleared to (5, 6, 7, 8), both `RGBA32Uint` and
/// stored, `exf2_fetch_unwritten` once: `3 c + t`, in a pass of the lane's size, again, in one of
/// twice its side, and in one of the first size once more, each its own command buffer. In the
/// interlock the pass leaves both colors out: two placeholders, then two larger ones. Through
/// input attachments no subpass leaves a color out, and none is made. The validation layer
/// stays quiet either way.
#[test]
fn a_left_out_color_s_framebuffer_view_is_a_placeholder() {
    let (messages, cache, lane) = lane();
    let vertices = library(&lane, "rt6_resolves");
    let unwritten = library(&lane, "exf2_fetch_unwritten");
    let pf = PixelFormat::RGBA32Uint;
    let counted = || {
        let counts = lane.device.work_counts();
        let attachments = counts.storage_placeholder_attachments;
        (counts.storage_placeholders_made, attachments)
    };
    let before = counted();
    let mut after = Vec::new();
    let sides: [u64; 4] = [1, 1, 2, 1];
    let ran = (|| {
        let fold = pipeline(
            &lane,
            &vertices,
            (&unwritten, "exf2_fetch_unwritten"),
            &[pf, pf],
            1,
        )?;
        let mut read = Vec::new();
        for side in sides {
            let shape = Shape {
                size: ONE.size * side,
                ..ONE
            };
            let count = lane.texture(pf, shape, TextureUsage::RENDER_TARGET);
            let tag = lane.texture(pf, shape, TextureUsage::RENDER_TARGET);
            let targets = [
                (
                    &count,
                    ClearColor::new(1.0, 2.0, 3.0, 4.0),
                    StoreAction::Store,
                ),
                (
                    &tag,
                    ClearColor::new(5.0, 6.0, 7.0, 8.0),
                    StoreAction::Store,
                ),
            ];
            steps(&lane, &pass(&targets, None), &[&fold])?;
            read.extend(lane.read(&count, (0, 0), 16, BlitOption::None));
            let (made, attachments) = counted();
            after.push((made - before.0, attachments - before.1));
        }
        Ok(read)
    })();
    let texel: Vec<u8> = [8u32, 12, 16, 20]
        .iter()
        .flat_map(|c| c.to_le_bytes())
        .collect();
    let texels: usize = (sides.iter())
        .map(|&side| TEXELS * (side * side) as usize)
        .sum();
    expect(
        &lane,
        "a left-out color's placeholder",
        ran,
        &texel.repeat(texels),
    );
    eprintln!(
        "a left-out color's placeholder on {} through {}: placeholders made and attachments \
         taken after each pass {after:?}",
        lane.device.name(),
        form(&lane)
    );
    let expected = match through_inputs(&lane) {
        true => vec![(0, 0); sides.len()],
        false => vec![(2, 2), (2, 4), (4, 6), (4, 8)],
    };
    assert_eq!(after, expected, "placeholders made, and attachments taken");
    drop(cache);
    quiet(lane, &messages);
}

/// A memoryless attachment loaded `DontCare` that a pass's first draw leaves alone, then folded,
/// stepped twice and folded again: over the count cleared to zero, `mli_count_seed` (the count
/// (1, 2, 3, 4), fetching nothing), `exf2_fetch_unwritten`, `mli_tag_step` twice (the tag
/// `2 (k + 1)` in lane `k`), `exf2_fetch_unwritten`: `11 c`, in each of two passes in turn, the
/// second of which would read `13 c` where it found what the first left in the backing the queue
/// keeps. The first draw records inline, and the pass splits once: in the interlock before the
/// first fold, whose draws read and write the colors as storage images, the memoryless one's
/// backing arriving with it; through input attachments before the first step, a write of the
/// memoryless color (`WorkCounts::inline_memoryless_write_splits`). The instance after the split
/// records inline too, after the backing is cleared (`render/inline_splits.rs`).
#[test]
fn a_memoryless_attachment_fetched_after_a_plain_inline_draw_starts_from_zero() {
    let (messages, cache, lane) = lane();
    let vertices = library(&lane, "rt6_resolves");
    let unwritten = library(&lane, "exf2_fetch_unwritten");
    let stepping = library(&lane, "mli_memoryless_step");
    let pf = PixelFormat::RGBA32Uint;
    let counted = || {
        let counts = lane.device.work_counts();
        let splits = counts.inline_memoryless_write_splits;
        (counts.inline_passes, splits, counts.inline_split_instances)
    };
    let before = counted();
    let ran = (|| {
        let seed = pipeline(
            &lane,
            &vertices,
            (&stepping, "mli_count_seed"),
            &[pf, pf],
            1,
        )?;
        let fold = pipeline(
            &lane,
            &vertices,
            (&unwritten, "exf2_fetch_unwritten"),
            &[pf, pf],
            1,
        )?;
        let step = pipeline(&lane, &vertices, (&stepping, "mli_tag_step"), &[pf, pf], 1)?;
        let count = lane.texture(pf, ONE, TextureUsage::RENDER_TARGET);
        let tag = memoryless(
            &lane,
            TextureType::Type2D,
            pf,
            ONE,
            TextureUsage::RENDER_TARGET,
        )?;
        let zero = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        let targets = [
            (&count, zero, StoreAction::Store),
            (&tag, zero, StoreAction::DontCare),
        ];
        let mut unloaded = pass(&targets, None);
        unloaded.color_attachments[1].load_action = LoadAction::DontCare;
        let mut read = Vec::new();
        for _ in 0..2 {
            steps(&lane, &unloaded, &[&seed, &fold, &step, &step, &fold])?;
            read.extend(lane.read(&count, (0, 0), 16, BlitOption::None));
        }
        Ok(read)
    })();
    let (inline, splits, later) = counted();
    let texel: Vec<u8> = (1u32..=4).flat_map(|c| (c * 11).to_le_bytes()).collect();
    expect(
        &lane,
        "a memoryless attachment fetched after a plain inline draw",
        ran,
        &texel.repeat(2 * TEXELS),
    );
    let (inline, splits) = (inline - before.0, splits - before.1);
    let later = later - before.2;
    eprintln!(
        "a memoryless attachment fetched after a plain inline draw on {}: {inline} passes \
         inline, {splits} split before a write of it, {later} instances inline after a split",
        lane.device.name()
    );
    let written = match through_inputs(&lane) {
        true => 2,
        false => 0,
    };
    assert_eq!(
        (inline, splits, later),
        (2, written, 2),
        "inline, and split into an instance inline too"
    );
    drop(cache);
    quiet(lane, &messages);
}

/// Draw `step` three times in each pass of `descriptors`, in order, in one command buffer of
/// `lane`'s queue; the first refusal of a pass, a pipeline or a draw, if any.
fn passes(
    lane: &Lane,
    descriptors: &[RenderPassDescriptor],
    step: &RenderPipelineState,
) -> Result<(), Error> {
    let command_buffer = lane.queue.command_buffer().unwrap();
    let depth = buffer_with(&lane.device, &Z.to_le_bytes()).unwrap();
    for descriptor in descriptors {
        let mut encoder = command_buffer.render_command_encoder(descriptor)?;
        encoder.set_vertex_buffer(Some(&depth), 0, 0)?;
        for _ in 0..3 {
            encoder.set_render_pipeline_state(step)?;
            encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)?;
        }
        encoder.end_encoding()?;
    }
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    Ok(())
}

/// A queue keeps one backing for its passes' memoryless attachments of one shape, where a pass
/// renders one into an image of its own (`WorkCounts::pooled_backings_made`: one for every pass
/// and command buffer below where the device announces `memoryless-backed`, none where it does not).
/// Each of three command buffers holds two passes of `exf2_fetch_unwritten` three times, each over
/// a count cleared to (1, 2, 3, 4) and a fresh memoryless tag: the first's tag cleared to
/// (5 + r, 6 + r, 7 + r, 8 + r) in round `r`, `c · 27 + t · 13` of its own clear; the second's
/// loaded `DontCare`, `c · 27`, the tag read as zero (as `a_dont_care_memoryless_attachment_reads_zero`
/// measures Metal's). No pass reads what an earlier one left in the backing.
#[test]
fn a_queue_keeps_one_memoryless_backing_for_its_passes() {
    let (messages, cache, lane) = lane();
    let vertices = library(&lane, "rt6_resolves");
    let unwritten = library(&lane, "exf2_fetch_unwritten");
    let pf = PixelFormat::RGBA32Uint;
    let fragment = (&unwritten, "exf2_fetch_unwritten");
    let step = pipeline(&lane, &vertices, fragment, &[pf, pf], 1).unwrap();
    let before = lane.device.work_counts().pooled_backings_made;
    let usage = TextureUsage::RENDER_TARGET;
    for round in 0..3u32 {
        let counts = [lane.texture(pf, ONE, usage), lane.texture(pf, ONE, usage)];
        let ran: Result<[Vec<u8>; 2], Error> = (|| {
            let tags = [
                memoryless(&lane, TextureType::Type2D, pf, ONE, usage)?,
                memoryless(&lane, TextureType::Type2D, pf, ONE, usage)?,
            ];
            let count = ClearColor::new(1.0, 2.0, 3.0, 4.0);
            let r = f64::from(round);
            let tag = ClearColor::new(5.0 + r, 6.0 + r, 7.0 + r, 8.0 + r);
            let mut descriptors: Vec<RenderPassDescriptor> = (counts.iter().zip(&tags))
                .map(|(counted, tagged)| {
                    let targets = [
                        (counted, count, StoreAction::Store),
                        (tagged, tag, StoreAction::DontCare),
                    ];
                    pass(&targets, None)
                })
                .collect();
            descriptors[1].color_attachments[1].load_action = LoadAction::DontCare;
            passes(&lane, &descriptors, &step)?;
            Ok(counts
                .each_ref()
                .map(|c| lane.read(c, (0, 0), 16, BlitOption::None)))
        })();
        // Lane `k` counts from `k + 1` with the tag `k + 5 + round`, then with zero.
        let cleared: Vec<u8> = (1u32..=4)
            .flat_map(|c| (c * 27 + (c + 4 + round) * 13).to_le_bytes())
            .collect();
        let unloaded: Vec<u8> = (1u32..=4).flat_map(|c| (c * 27).to_le_bytes()).collect();
        let (first, second) = match ran {
            Ok([first, second]) => (Ok(first), Ok(second)),
            Err(error) => (Err(error.clone()), Err(error)),
        };
        let name = format!("a memoryless attachment of a kept backing, round {round}");
        expect(&lane, &name, first, &cleared.repeat(TEXELS));
        let name =
            format!("a memoryless attachment of a kept backing loaded DontCare, round {round}");
        expect(&lane, &name, second, &unloaded.repeat(TEXELS));
    }
    let made = lane.device.work_counts().pooled_backings_made - before;
    let backed = messages
        .take_notices()
        .into_iter()
        .filter_map(|message| message.degradation)
        .any(|notice| notice.degradation == Degradation::MemorylessBacked);
    eprintln!(
        "framebuffer fetch on {}: {made} memoryless backings made for 6 passes in 3 command buffers",
        lane.device.name()
    );
    assert_eq!(
        made,
        u64::from(backed),
        "one backing for every pass, where backed"
    );
    drop(cache);
    quiet(lane, &messages);
}

/// A queue keeps a backing for each shape of memoryless attachment its frames render, a
/// compositor's dozen (`WorkCounts::pooled_backings_made`): two rounds of the memoryless case's
/// pass at 12 sides in turn, 1 to 12 texels square, each in a command buffer of its own over a
/// fresh memoryless tag, `c · 27 + t · 13` every texel; the second round makes none where the
/// device announces `memoryless-backed` (12 in all), and none at all where it does not.
#[test]
fn a_queue_keeps_a_memoryless_backing_for_each_shape_it_renders() {
    const SIDES: u64 = 12;
    let (messages, cache, lane) = lane();
    let vertices = library(&lane, "rt6_resolves");
    let unwritten = library(&lane, "exf2_fetch_unwritten");
    let pf = PixelFormat::RGBA32Uint;
    let fragment = (&unwritten, "exf2_fetch_unwritten");
    let step = pipeline(&lane, &vertices, fragment, &[pf, pf], 1).unwrap();
    let usage = TextureUsage::RENDER_TARGET;
    let texel: Vec<u8> = (1u32..=4)
        .flat_map(|c| (c * 27 + (c + 4) * 13).to_le_bytes())
        .collect();
    let before = lane.device.work_counts().pooled_backings_made;
    for round in 0..2 {
        for size in 1..=SIDES {
            let shape = Shape { size, ..ONE };
            let count = lane.texture(pf, shape, usage);
            let ran = (|| {
                let tag = memoryless(&lane, TextureType::Type2D, pf, shape, usage)?;
                let targets = [
                    (
                        &count,
                        ClearColor::new(1.0, 2.0, 3.0, 4.0),
                        StoreAction::Store,
                    ),
                    (
                        &tag,
                        ClearColor::new(5.0, 6.0, 7.0, 8.0),
                        StoreAction::DontCare,
                    ),
                ];
                steps(&lane, &pass(&targets, None), &[&step, &step, &step])?;
                Ok(lane.read(&count, (0, 0), 16, BlitOption::None))
            })();
            let name = format!("a memoryless attachment {size} texels square, round {round}");
            let texels = usize::try_from(size * size).unwrap();
            expect(&lane, &name, ran, &texel.repeat(texels));
        }
    }
    let made = lane.device.work_counts().pooled_backings_made - before;
    let backed = messages
        .take_notices()
        .into_iter()
        .filter_map(|message| message.degradation)
        .any(|notice| notice.degradation == Degradation::MemorylessBacked);
    eprintln!(
        "framebuffer fetch on {}: {made} memoryless backings made for {SIDES} shapes rendered \
         twice",
        lane.device.name()
    );
    assert_eq!(
        made,
        SIDES * u64::from(backed),
        "one backing for each shape, where backed"
    );
    drop(cache);
    quiet(lane, &messages);
}

/// Visibility results whose modes span the render pass instances of a split pass are those of
/// one query each (the module docs).
#[test]
fn visibility_results_span_split_passes() {
    use VisibilityResultMode::{Boolean, Counting, Disabled};
    let (messages, cache, lane) = lane();
    let precise = lane
        .device
        .properties()
        .description
        .has(VkFeature::OcclusionQueryPrecise);
    let vertices = library(&lane, "rt6_resolves");
    let fetch = library(&lane, "fragment_fetch");
    let pf = PixelFormat::R32Float;
    let halve = pipeline(&lane, &vertices, (&fetch, "fetch_scalar"), &[pf], 1).unwrap();
    let white = blended_white(&lane, &vertices, &[pf]);
    let target = lane.texture(pf, ONE, TextureUsage::RENDER_TARGET);
    let clear = ClearColor::new(0.75, 0.0, 0.0, 0.0);
    let pattern: Vec<u8> = (0..32).map(|b| 0xA0 | b).collect();
    let results = buffer_with(&lane.device, &pattern).unwrap();
    let mut descriptor = pass(&[(&target, clear, StoreAction::Store)], None);
    descriptor.visibility_result_buffer = Some(results.clone());
    /// A step of the pass: a mode set at an offset, or a draw of a pipeline, seen or clipped.
    enum Command<'a> {
        Mode(VisibilityResultMode, u64),
        Draw(&'a RenderPipelineState, bool),
    }
    use Command::{Draw, Mode};
    let mut commands = vec![
        Mode(Boolean, 0),
        Draw(&white, true),
        Draw(&halve, true),
        Mode(Boolean, 8),
        Draw(&white, false),
        Draw(&halve, true),
        Mode(Boolean, 16),
        Draw(&halve, false),
        Draw(&white, false),
    ];
    if precise {
        commands.extend([Mode(Counting, 24), Draw(&white, true), Draw(&halve, true)]);
    }
    commands.push(Mode(Disabled, 0));
    lane.run(&descriptor, |lane, encoder, held: &mut Vec<Buffer>| {
        let seen = buffer_with(&lane.device, &Z.to_le_bytes()).unwrap();
        // Past the far plane: clipped, so no sample passes.
        let clipped = buffer_with(&lane.device, &2.0f32.to_le_bytes()).unwrap();
        for command in commands {
            match command {
                Mode(mode, offset) => encoder.set_visibility_result_mode(mode, offset).unwrap(),
                Draw(step, visible) => {
                    let depth = if visible { &seen } else { &clipped };
                    encoder.set_vertex_buffer(Some(depth), 0, 0).unwrap();
                    encoder.set_render_pipeline_state(step).unwrap();
                    encoder
                        .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                        .unwrap();
                }
            }
        }
        held.extend([seen, clipped]);
    })
    .unwrap();
    let device = lane.device.name();
    let mut want: Vec<u8> = [1u64, 1, 0].iter().flat_map(|w| w.to_le_bytes()).collect();
    if precise {
        want.extend(32u64.to_le_bytes());
    } else {
        want.extend(&pattern[24..]);
        eprintln!("refused: Counting across a split on {device}: no occlusionQueryPrecise");
    }
    // The three `Boolean` words by their contract (`visibility::by_contract`): nonzero where a
    // sample passed, the device's own value.
    let read = read_buffer(&results, 0, 32).unwrap();
    let words: Vec<u64> = read
        .as_chunks::<8>()
        .0
        .iter()
        .map(|w| u64::from_le_bytes(*w))
        .collect();
    let unwritten = u64::from_le_bytes(pattern[..8].try_into().unwrap());
    let mut read: Vec<u8> =
        super::visibility::by_contract(words[..3].to_vec(), &[0, 1, 2], unwritten)
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
    read.extend(words[3].to_le_bytes());
    assert_eq!(read, want, "the results on {device}");
    let ran = Ok(lane.read(&target, (0, 0), 4, BlitOption::None));
    let texel = if precise { 0.5f32 } else { 0.25 };
    expect(
        &lane,
        "with visibility results across a split",
        ran,
        &texel.to_le_bytes().repeat(TEXELS),
    );
    drop(cache);
    quiet(lane, &messages);
}

/// The kinds [`fetches_degrade_once_per_kind`] takes.
pub(super) const FETCHES: [Degradation; 3] = [
    Degradation::RenderPassSplit,
    Degradation::MemorylessBacked,
    Degradation::FetchThroughScratch,
];

/// Three passes, each encoded twice: a `SIDE`² `RGBA32Uint` target fetched by
/// `exf_fetch_count` three times; a `SIDE`² `R32Float` one drawn white under a blend
/// ([`blended_white`]), then fetched by `fetch_scalar`; and the memoryless pass above. Through the pixel interlock the second
/// splits (announced but where the device runs such a split no slower than whole), the third backs
/// its memoryless attachment, and each is announced once. A fetched texture renders into a scratch
/// image only where the device lays its image out otherwise with `STORAGE`, the device's own
/// answer, which the lane does not foresee: where it does, the scratch is announced once, naming
/// one of the two fetched textures, the first the device met so. Measured at `caea7050`: MoltenVK
/// on an M4 and ANV on an Iris Xe (RPL-P) take it for the `R32Float` texture and not the
/// `RGBA32Uint` one. Through input attachments (lavapipe) none is announced.
#[test]
fn fetches_degrade_once_per_kind() {
    let (messages, cache, lane) = lane();
    let device = lane.device.name().to_owned();
    let vertices = library(&lane, "rt6_resolves");
    let fetch = library(&lane, "fragment_fetch");
    let count = library(&lane, "exf_fetch_count");
    let unwritten = library(&lane, "exf2_fetch_unwritten");
    let shape = Shape { size: SIDE, ..ONE };
    let (uint, float) = (PixelFormat::RGBA32Uint, PixelFormat::R32Float);
    let counting = pipeline(&lane, &vertices, (&count, "exf_fetch_count"), &[uint], 1).unwrap();
    let halve = pipeline(&lane, &vertices, (&fetch, "fetch_scalar"), &[float], 1).unwrap();
    let white = blended_white(&lane, &vertices, &[float]);
    let fragment = (&unwritten, "exf2_fetch_unwritten");
    let tagged = pipeline(&lane, &vertices, fragment, &[uint, uint], 1).unwrap();
    for _ in 0..2 {
        let target = lane.texture(uint, shape, TextureUsage::RENDER_TARGET);
        let clear = ClearColor::new(1.0, 2.0, 3.0, 4.0);
        let targets = [(&target, clear, StoreAction::Store)];
        steps(&lane, &pass(&targets, None), &[&counting; 3]).unwrap();
        let target = lane.texture(float, shape, TextureUsage::RENDER_TARGET);
        let clear = ClearColor::new(0.75, 0.0, 0.0, 0.0);
        let targets = [(&target, clear, StoreAction::Store)];
        steps(&lane, &pass(&targets, None), &[&white, &halve]).unwrap();
        let stored = lane.texture(uint, ONE, TextureUsage::RENDER_TARGET);
        let tag = memoryless(
            &lane,
            TextureType::Type2D,
            uint,
            ONE,
            TextureUsage::RENDER_TARGET,
        )
        .unwrap();
        let targets = [
            (&stored, clear, StoreAction::Store),
            (&tag, clear, StoreAction::DontCare),
        ];
        steps(&lane, &pass(&targets, None), &[&tagged; 3]).unwrap();
    }
    // The pipelines' makings, and the storage form a texture fetched through a scratch takes.
    let mut first = made(
        &lane,
        &[
            pipeline_descriptor(&vertices, (&count, "exf_fetch_count"), &[uint], 1),
            pipeline_descriptor(&vertices, (&fetch, "fetch_scalar"), &[float], 1),
            blended_descriptor(&vertices, &[float]),
            pipeline_descriptor(&vertices, fragment, &[uint, uint], 1),
        ],
    );
    first.extend(forced(&[(
        Degradation::FetchedTargetStorage,
        scratched(&messages),
    )]));
    let notices = messages.take_notices();
    messages.restore_notices(notices.clone());
    let interlock = !through_inputs(&lane);
    let scratch = notices
        .iter()
        .filter_map(|message| message.degradation.as_ref())
        .find(|notice| notice.degradation == Degradation::FetchThroughScratch)
        .map(|notice| notice.object.clone());
    let expected: Vec<Degradation> = FETCHES
        .into_iter()
        .filter(|&kind| match kind {
            Degradation::FetchThroughScratch => scratch.is_some(),
            Degradation::RenderPassSplit => announces_interlock_splits(&lane),
            _ => interlock,
        })
        .collect();
    let counts = assert_made_once(&messages, &expected, &first);
    let announced: Vec<Degradation> = counts
        .keys()
        .copied()
        .filter(|kind| FETCHES.contains(kind))
        .collect();
    let mut expected = expected;
    expected.sort();
    assert_eq!(announced, expected, "on {device}: {counts:?}");
    if let Some(object) = &scratch {
        assert!(interlock, "a scratch through input attachments on {device}");
        let fetched = [uint, float].map(|pf| format!("{pf:?} {SIDE}x{SIDE} "));
        assert!(
            fetched
                .iter()
                .any(|texture| object.contains(texture.as_str())),
            "the scratch's texture on {device}: {object}"
        );
        eprintln!("framebuffer fetch's scratch on {device}: {object}");
    }
    eprintln!(
        "framebuffer fetch's notices on {device} through {}: {announced:?}",
        form(&lane)
    );
    drop(cache);
    quiet(lane, &messages);
}

/// The side of the load-and-store test's texture.
const SIDE: u64 = 16;

/// `value` in each of the `SIDE`² texels of an `R32Float` texture.
fn floats(value: f32) -> Vec<u8> {
    value.to_le_bytes().repeat((SIDE * SIDE) as usize)
}

/// What `a4_pixel_fetch` samples of each texel of the `R32Float` `texture`'s top left 4² in a
/// compute pass: `(r, 0, 0, 1)`, row by row.
fn sampled(lane: &Lane, texture: &Texture) -> Vec<u8> {
    let library = library(lane, "a4_pixel_fetch");
    let function = library.new_function_with_name("a4_pixel_fetch").unwrap();
    let state = lane
        .device
        .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
        .unwrap();
    let length = 16 * TEXELS as u64;
    let out = buffer_with(&lane.device, &vec![0; length as usize]).unwrap();
    let command_buffer = lane.queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    encoder.set_buffer(Some(&out), 0, 0).unwrap();
    encoder
        .dispatch_threads(Size::new(4, 4, 1), Size::new(4, 4, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    read_buffer(&out, 0, length).unwrap()
}

/// A fetched texture's contents reach the pass through its load and leave it through its store,
/// and later passes read them: a compute pass sampling it, and a pass rendering a texture view
/// of it (the module docs).
#[test]
fn fetched_textures_load_store_and_reach_later_passes() {
    let (messages, cache, lane) = lane();
    let device = lane.device.name();
    let vertices = library(&lane, "rt6_resolves");
    let fetch = library(&lane, "fragment_fetch");
    let pf = PixelFormat::R32Float;
    let halve = pipeline(&lane, &vertices, (&fetch, "fetch_scalar"), &[pf], 1).unwrap();
    let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    let shape = Shape { size: SIDE, ..ONE };
    let texture = lane.texture(pf, shape, usage);
    let read = |texture: &Texture| lane.read(texture, (0, 0), 4, BlitOption::None);
    let clear = ClearColor::new(0.75, 0.0, 0.0, 0.0);
    let targets = [(&texture, clear, StoreAction::Store)];
    steps(&lane, &pass(&targets, None), &[]).unwrap();
    assert_eq!(read(&texture), floats(0.75), "the clear on {device}");
    let loaded =
        steps(&lane, &loading(&texture, StoreAction::Store), &[&halve]).map(|()| read(&texture));
    expect(&lane, "of a loaded texture, stored", loaded, &floats(0.375));
    let texel: Vec<u8> = [0.375f32, 0.0, 0.0, 1.0]
        .iter()
        .flat_map(|c| c.to_le_bytes())
        .collect();
    assert_eq!(
        sampled(&lane, &texture),
        texel.repeat(TEXELS),
        "a compute pass sampling the fetched texture on {device}"
    );
    steps(&lane, &loading(&texture, StoreAction::DontCare), &[&halve]).unwrap();
    let left = read(&texture);
    let kept = if left == floats(0.375) {
        "its earlier texels"
    } else {
        assert_eq!(left, floats(0.1875), "a DontCare store on {device}");
        "the rendered texels"
    };
    eprintln!("framebuffer fetch stored DontCare on {device}: the texture holds {kept}");
    let view = texture.new_texture_view(pf).unwrap();
    let viewed =
        steps(&lane, &loading(&view, StoreAction::Store), &[&halve]).map(|()| read(&texture));
    let half: Vec<u8> = left
        .chunks(4)
        .flat_map(|c| (f32::from_le_bytes(c.try_into().unwrap()) / 2.0).to_le_bytes())
        .collect();
    expect(&lane, "through a texture view", viewed, &half);
    drop(cache);
    quiet(lane, &messages);
}

/// A caller's `R32_SFLOAT` image made with color attachment and transfer usage alone
/// (`new_texture_with_vk_image`), wrapped as a `SIDE`² `R32Float` render target: its image lacks
/// the usage either fetch form needs (input attachment usage, `STORAGE`). Passes that clear it
/// and draw it white without fetching render into it, on every venue. A pass that loads it and
/// halves it by a fetch renders into a scratch image that has the usage, copied in before the
/// pass and back after it; so does a pass that clears it, draws it white and then halves it:
/// through input attachments split where the fetch begins (the scratch's input attachment usage
/// makes it an input attachment of the later draws' subpass alone), and in the pixel interlock
/// split there too, its first instance, its one attachment cleared and stored, recorded inline
/// or not (counted in `WorkCounts::inline_passes`, `render/inline_first.rs`). Beside a memoryless
/// attachment, which no inline instance takes, the same pass records no instance inline, and in
/// the interlock splits there and backs the memoryless attachment. Each holds Metal's texels, and
/// the scratch is announced once, naming the missing usage and the caller, and the split once
/// through input attachments, and in the interlock where the device is not measured to run such a
/// split no slower than whole, with the backing.
#[test]
fn a_callers_image_without_the_fetch_usage_fetches_through_a_scratch() {
    let (messages, cache, lane) = lane();
    let device = lane.device.name().to_owned();
    let vertices = library(&lane, "rt6_resolves");
    let fetch = library(&lane, "fragment_fetch");
    let pf = PixelFormat::R32Float;
    let halve = pipeline(&lane, &vertices, (&fetch, "fetch_scalar"), &[pf], 1).unwrap();
    let white = pipeline(&lane, &vertices, (&vertices, WHITE), &[pf], 1).unwrap();
    // SAFETY: used only while the device lives; the image is destroyed after its texture and
    // every command buffer that used it.
    let raw = unsafe { lane.device.raw() };
    // SAFETY: a physical device of the instance.
    let memory = unsafe {
        raw.instance
            .get_physical_device_memory_properties(raw.physical_device)
    };
    let usage = ash::vk::ImageUsageFlags::COLOR_ATTACHMENT
        | ash::vk::ImageUsageFlags::TRANSFER_SRC
        | ash::vk::ImageUsageFlags::TRANSFER_DST;
    let info = shared_among(
        image_info(ash::vk::Format::R32_SFLOAT, usage),
        raw.queue_families,
    );
    // SAFETY: the device is live.
    let (image, image_memory) = unsafe { made_image(raw.device, &memory, &info) };
    let mut descriptor = TextureDescriptor::texture_2d(pf, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    // SAFETY: the image is the device's, made with `info`, and nothing else uses it.
    let texture = unsafe {
        lane.device.new_texture_with_vk_image(
            &descriptor,
            image,
            &info,
            VkImageLayouts::new(
                ash::vk::ImageLayout::UNDEFINED.into(),
                ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL.into(),
            ),
        )
    }
    .unwrap();
    let read = |texture: &Texture| lane.read(texture, (0, 0), 4, BlitOption::None);
    let clear = ClearColor::new(0.75, 0.0, 0.0, 0.0);
    let targets = [(&texture, clear, StoreAction::Store)];
    steps(&lane, &pass(&targets, None), &[]).unwrap();
    assert_eq!(read(&texture), floats(0.75), "the clear on {device}");
    steps(&lane, &loading(&texture, StoreAction::Store), &[&white]).unwrap();
    assert_eq!(
        read(&texture),
        floats(1.0),
        "a draw without a fetch on {device}"
    );
    let loaded =
        steps(&lane, &loading(&texture, StoreAction::Store), &[&halve]).map(|()| read(&texture));
    expect(&lane, "of a caller's image, loaded", loaded, &floats(0.5));
    let inline_passes = || lane.device.work_counts().inline_passes;
    let before = inline_passes();
    let drawn = steps(&lane, &pass(&targets, None), &[&white, &halve]).map(|()| read(&texture));
    let inline = inline_passes() > before;
    expect(
        &lane,
        "of a caller's image, drawn first",
        drawn,
        &floats(0.5),
    );
    let notices = messages.take_notices();
    messages.restore_notices(notices.clone());
    let mut kinds = vec![Degradation::FetchThroughScratch];
    let split = through_inputs(&lane) || announces_interlock_splits(&lane);
    if split {
        kinds.push(Degradation::RenderPassSplit);
    }
    // The caller's image rests in `TRANSFER_DST_OPTIMAL`, out of `GENERAL`, so each batch that
    // uses it moves it there and back.
    let mut first = made(
        &lane,
        &[
            pipeline_descriptor(&vertices, (&fetch, "fetch_scalar"), &[pf], 1),
            pipeline_descriptor(&vertices, (&vertices, WHITE), &[pf], 1),
        ],
    );
    first.insert(Degradation::WrappedImageLayoutTransitions);
    let counts = assert_made_once(&messages, &kinds, &first);
    assert_eq!(
        counts.contains_key(&Degradation::RenderPassSplit),
        split,
        "the split of the pass drawn first on {device} (inline: {inline}): {counts:?}"
    );
    let scratch = notices
        .iter()
        .filter_map(|message| message.degradation.as_ref())
        .find(|notice| notice.degradation == Degradation::FetchThroughScratch)
        .map(|notice| notice.reason.clone())
        .unwrap();
    let missing = if through_inputs(&lane) {
        "no input attachment usage"
    } else {
        "no `STORAGE` usage"
    };
    // It names the draw that fetched it: its fragment function and the color's blend state.
    assert!(
        scratch.contains(missing)
            && scratch.contains("the caller made it")
            && scratch.contains("a draw of fragment function `")
            && scratch.contains("color 0 "),
        "the scratch's notice on {device}: {scratch}"
    );
    eprintln!(
        "framebuffer fetch of a caller's image on {device} through {} (first instance inline: \
         {inline}): {counts:?}",
        form(&lane)
    );
    // Beside a memoryless attachment, which no inline instance takes.
    let colors = [pf, pf];
    let halve_beside = pipeline(&lane, &vertices, (&fetch, "fetch_scalar"), &colors, 1).unwrap();
    let white_beside = pipeline(&lane, &vertices, (&vertices, WHITE), &colors, 1).unwrap();
    let shape = Shape { size: SIDE, ..ONE };
    let tag = memoryless(
        &lane,
        TextureType::Type2D,
        pf,
        shape,
        TextureUsage::RENDER_TARGET,
    )
    .unwrap();
    let beside = [
        (&texture, clear, StoreAction::Store),
        (&tag, clear, StoreAction::DontCare),
    ];
    let before = inline_passes();
    let drawn = steps(&lane, &pass(&beside, None), &[&white_beside, &halve_beside])
        .map(|()| read(&texture));
    assert_eq!(
        inline_passes(),
        before,
        "a pass beside a memoryless attachment on {device} records no instance inline"
    );
    expect(
        &lane,
        "of a caller's image, drawn first beside a memoryless attachment",
        drawn,
        &floats(0.5),
    );
    if !through_inputs(&lane) {
        // The split was announced at the pass drawn first, where the device announces it.
        let backed = announces_interlock_splits(&lane);
        let kinds: Vec<Degradation> = [Degradation::MemorylessBacked]
            .into_iter()
            .filter(|_| backed)
            .collect();
        let mut first = made(
            &lane,
            &[
                pipeline_descriptor(&vertices, (&fetch, "fetch_scalar"), &colors, 1),
                pipeline_descriptor(&vertices, (&vertices, WHITE), &colors, 1),
            ],
        );
        first.extend(forced(&[(
            Degradation::FetchThroughScratch,
            scratched(&messages),
        )]));
        let counts = assert_made_once(&messages, &kinds, &first);
        assert!(
            !counts.contains_key(&Degradation::RenderPassSplit),
            "the pass beside a memoryless attachment on {device}: {counts:?}"
        );
        eprintln!(
            "framebuffer fetch of a caller's image beside a memoryless attachment on {device}: \
             split at the fetch"
        );
    }
    drop(tag);
    drop(texture);
    // SAFETY: nothing uses them: the texture is gone and every command buffer completed.
    unsafe {
        raw.device.destroy_image(image, None);
        raw.device.free_memory(image_memory, None);
    }
    drop(cache);
    quiet(lane, &messages);
}

/// A pass of a 4² `R32Float` target cleared to 0.75 and stored, beside a memoryless `R32Float`
/// attachment its pipelines have and never write: `rt6_resolve_white`, `fetch_scalar`, the white
/// again and `fetch_scalar` twice (0.25); the white drawn of indices, then `fetch_scalar` (0.5).
/// In the pixel interlock each splits at the fetch, which backs the memoryless attachment. The
/// white under a blend ([`blended_white`]), then `fetch_scalar`, holds 0.5 too, and in the pixel
/// interlock splits there too. A split is announced, and the backing, once each (at the plain
/// draws' pass), but on a device whose driver runs such a split no slower than whole, where
/// neither is.
#[test]
fn a_plain_draw_before_a_fetch_takes_the_devices_faster_form() {
    let (messages, cache, lane) = lane();
    let device = lane.device.name().to_owned();
    let vertices = library(&lane, "rt6_resolves");
    let fetch = library(&lane, "fragment_fetch");
    let pf = PixelFormat::R32Float;
    let colors = [pf, pf];
    let halve = pipeline(&lane, &vertices, (&fetch, "fetch_scalar"), &colors, 1).unwrap();
    let white = pipeline(&lane, &vertices, (&vertices, WHITE), &colors, 1).unwrap();
    let blended = blended_white(&lane, &vertices, &colors);
    let target = lane.texture(pf, ONE, TextureUsage::RENDER_TARGET);
    let usage = TextureUsage::RENDER_TARGET;
    let tag = memoryless(&lane, TextureType::Type2D, pf, ONE, usage).unwrap();
    let clear = ClearColor::new(0.75, 0.0, 0.0, 0.0);
    let targets = [
        (&target, clear, StoreAction::Store),
        (&tag, clear, StoreAction::DontCare),
    ];
    let descriptor = pass(&targets, None);
    let read = || lane.read(&target, (0, 0), 4, BlitOption::None);
    let texels = |value: f32| value.to_le_bytes().repeat(TEXELS);
    let split = [Degradation::RenderPassSplit, Degradation::MemorylessBacked];
    let order = [&white, &halve, &white, &halve, &halve];
    let plain = steps(&lane, &descriptor, &order).map(|()| read());
    expect(&lane, "after plain draws", plain, &texels(0.25));
    let indexed = lane
        .run(&descriptor, |lane, encoder, held: &mut Vec<Buffer>| {
            let depth = buffer_with(&lane.device, &Z.to_le_bytes()).unwrap();
            let indices: Vec<u8> = [0u16, 1, 2, 0]
                .iter()
                .flat_map(|i| i.to_le_bytes())
                .collect();
            let index_buffer = buffer_with(&lane.device, &indices).unwrap();
            encoder.set_vertex_buffer(Some(&depth), 0, 0).unwrap();
            encoder.set_render_pipeline_state(&white).unwrap();
            encoder
                .draw_indexed_primitives(
                    PrimitiveType::Triangle,
                    3,
                    IndexType::UInt16,
                    &index_buffer,
                    0,
                    1,
                    0,
                    0,
                )
                .unwrap();
            encoder.set_render_pipeline_state(&halve).unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
            held.extend([depth, index_buffer]);
        })
        .map(|()| read());
    expect(&lane, "after an indexed plain draw", indexed, &texels(0.5));
    let interlock = !through_inputs(&lane);
    let announced_split = |splits: bool| -> Vec<Degradation> {
        let announced = splits && announces_interlock_splits(&lane);
        split.into_iter().filter(|_| announced).collect()
    };
    let kinds = announced_split(interlock);
    let mut first = made(
        &lane,
        &[
            pipeline_descriptor(&vertices, (&fetch, "fetch_scalar"), &colors, 1),
            pipeline_descriptor(&vertices, (&vertices, WHITE), &colors, 1),
            blended_descriptor(&vertices, &colors),
        ],
    );
    let scratch = scratched(&messages);
    first.extend(forced(&[
        (Degradation::FetchThroughScratch, scratch),
        (Degradation::FetchedTargetStorage, scratch),
    ]));
    let counts = assert_made_once(&messages, &kinds, &first);
    let announced: Vec<Degradation> = counts
        .keys()
        .copied()
        .filter(|kind| split.contains(kind))
        .collect();
    assert_eq!(
        announced, kinds,
        "a pass of plain draws before a fetch on {device}"
    );
    let blending = steps(&lane, &descriptor, &[&blended, &halve]).map(|()| read());
    expect(&lane, "after a blended draw", blending, &texels(0.5));
    // Each kind is announced once: where the plain draws' pass split, its split and backing
    // were announced there (the RTX 3080), and this split announces neither.
    let kinds: Vec<Degradation> = Vec::new();
    let scratch = scratched(&messages);
    let first = forced(&[
        (Degradation::FetchThroughScratch, scratch),
        (Degradation::FetchedTargetStorage, scratch),
    ]);
    let counts = assert_made_once(&messages, &kinds, &first);
    let announced: Vec<Degradation> = counts
        .keys()
        .copied()
        .filter(|kind| split.contains(kind))
        .collect();
    assert_eq!(
        announced, kinds,
        "a blended draw before a fetch on {device}"
    );
    let instances = match interlock {
        true => "split at the fetch",
        false => "one render pass instance",
    };
    eprintln!(
        "framebuffer fetch after plain draws on {device} through {}: {instances}",
        form(&lane)
    );
    drop(cache);
    quiet(lane, &messages);
}

/// A caller's 4x `R32_SFLOAT` image made with color attachment and transfer usage alone
/// (`new_texture_with_vk_image`), wrapped as a 4x `SIDE`² `R32Float` render target: its image
/// lacks the usage each fetch form needs of a multisampled attachment (the module docs). A pass
/// that clears it to 0.75 and stores it renders into it. A pass that loads it, halves it by a
/// fetch, stores it and resolves it renders into a 4x scratch image that has the usage, copied in
/// before the pass and back after it: the resolve holds 0.375. A pass that loads it and only
/// resolves it then resolves what the scratch copied back, 0.375 again, by the device's average.
/// Each holds Metal's texels on every venue, and the scratch is announced once, naming the
/// missing usage and the caller. Through the attachment's multisampled storage image the fetching
/// pass splits for the device's resolve, announced as the interlock's splits are.
#[test]
fn a_callers_multisampled_image_without_the_fetch_usage_fetches_through_a_scratch() {
    let (messages, cache, lane) = lane();
    let device = lane.device.name().to_owned();
    let vertices = library(&lane, "rt6_resolves");
    let fetch = library(&lane, "fragment_fetch");
    let pf = PixelFormat::R32Float;
    let halve = pipeline(&lane, &vertices, (&fetch, "fetch_scalar"), &[pf], 4).unwrap();
    // SAFETY: used only while the device lives; the image is destroyed after its texture and
    // every command buffer that used it.
    let raw = unsafe { lane.device.raw() };
    // SAFETY: a physical device of the instance.
    let memory = unsafe {
        raw.instance
            .get_physical_device_memory_properties(raw.physical_device)
    };
    let usage = ash::vk::ImageUsageFlags::COLOR_ATTACHMENT
        | ash::vk::ImageUsageFlags::TRANSFER_SRC
        | ash::vk::ImageUsageFlags::TRANSFER_DST;
    let info =
        image_info(ash::vk::Format::R32_SFLOAT, usage).samples(ash::vk::SampleCountFlags::TYPE_4);
    let info = shared_among(info, raw.queue_families);
    // SAFETY: the device is live.
    let (image, image_memory) = unsafe { made_image(raw.device, &memory, &info) };
    let mut descriptor = TextureDescriptor::texture_2d(pf, SIDE, SIDE, false);
    descriptor.texture_type = TextureType::Type2DMultisample;
    descriptor.sample_count = 4;
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    // SAFETY: the image is the device's, made with `info`, and nothing else uses it.
    let texture = unsafe {
        lane.device.new_texture_with_vk_image(
            &descriptor,
            image,
            &info,
            VkImageLayouts::new(
                ash::vk::ImageLayout::UNDEFINED.into(),
                ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL.into(),
            ),
        )
    }
    .unwrap();
    let resolve = lane.texture(pf, Shape { size: SIDE, ..ONE }, TextureUsage::RENDER_TARGET);
    let read = || lane.read(&resolve, (0, 0), 4, BlitOption::None);
    let resolving = || {
        let mut descriptor = loading(&texture, StoreAction::StoreAndMultisampleResolve);
        descriptor.color_attachments[0].resolve_texture = Some(resolve.clone());
        descriptor
    };
    let clear = ClearColor::new(0.75, 0.0, 0.0, 0.0);
    steps(
        &lane,
        &pass(&[(&texture, clear, StoreAction::Store)], None),
        &[],
    )
    .unwrap();
    let halved = steps(&lane, &resolving(), &[&halve]).map(|()| read());
    expect(
        &lane,
        "of a caller's 4x image, resolved",
        halved,
        &floats(0.375),
    );
    let kept = steps(&lane, &resolving(), &[]).map(|()| read());
    expect(
        &lane,
        "of a caller's 4x image, stored back",
        kept,
        &floats(0.375),
    );
    let notices = messages.take_notices();
    messages.restore_notices(notices.clone());
    // The caller's image rests out of `GENERAL`. The device's average resolves it, announced
    // where it is not measured to be Metal's; where the device reads sample planes, after they are
    // merged back.
    let planes = reads_planes(&lane);
    let rounded = resolve_kind(&lane.device, 4).is_some();
    let mut first = made(
        &lane,
        &[pipeline_descriptor(
            &vertices,
            (&fetch, "fetch_scalar"),
            &[pf],
            4,
        )],
    );
    // Through the attachment's own multisampled storage image the fetching draw's instance
    // leaves the attachment out of its subpass, and the device's resolve needs it in one: the
    // pass splits there (the RTX 3080's two instances), announced where the
    // device announces the interlock's splits, and through sample planes too, which the device's
    // resolve reads merged back in one more instance.
    let split = announces_interlock_splits(&lane);
    first.extend(forced(&[
        (Degradation::WrappedImageLayoutTransitions, true),
        (Degradation::SamplePlanes, planes),
        (Degradation::DeviceRoundedResolve, rounded),
        (Degradation::RenderPassSplit, split),
    ]));
    let counts = assert_made_once(&messages, &[Degradation::FetchThroughScratch], &first);
    let scratch = notices
        .iter()
        .filter_map(|message| message.degradation.as_ref())
        .find(|notice| notice.degradation == Degradation::FetchThroughScratch)
        .map(|notice| notice.reason.clone())
        .unwrap();
    let missing = if through_inputs(&lane) {
        "no input attachment usage"
    } else if reads_planes(&lane) {
        "no sampled usage with a view"
    } else {
        "no `STORAGE` usage"
    };
    assert!(
        scratch.contains(missing) && scratch.contains("the caller made it"),
        "the scratch's notice on {device}: {scratch}"
    );
    eprintln!(
        "framebuffer fetch of a caller's 4x image on {device} through {}: {counts:?}",
        form(&lane)
    );
    drop(texture);
    // SAFETY: nothing uses them: the texture is gone and every command buffer completed.
    unsafe {
        raw.device.destroy_image(image, None);
        raw.device.free_memory(image_memory, None);
    }
    drop(cache);
    quiet(lane, &messages);
}

/// The texels a pass of `texture` that clears it to `clear` and stores it leaves, with no draw:
/// the clear as a load operation.
fn cleared(lane: &Lane, texture: &Texture, clear: ClearColor, bytes: u64) -> Vec<u8> {
    let targets = [(texture, clear, StoreAction::Store)];
    steps(lane, &pass(&targets, None), &[]).unwrap();
    lane.read(texture, (0, 0), bytes, BlitOption::None)
}

/// Bytes of one texel of the formats `fetched_attachments_clear_as_the_load_action_does` clears.
fn texel_bytes(pf: PixelFormat) -> u64 {
    match pf {
        PixelFormat::RG8Unorm => 2,
        PixelFormat::RGBA8Unorm | PixelFormat::RGBA8Uint | PixelFormat::RG16Float => 4,
        PixelFormat::RGBA16Uint | PixelFormat::RGBA16Float => 8,
        _ => 16,
    }
}

/// A fetched attachment the function only reads, cleared and stored by a pass of one draw,
/// holds the texels the clear stores as a load operation, for each format (the module docs).
/// A format the device does not store, which the interlock form needs, is refused at the
/// pipeline (`Unsupported`), printed.
#[test]
fn fetched_attachments_clear_as_the_load_action_does() {
    let (messages, cache, lane) = lane();
    let device = lane.device.name();
    let vertices = library(&lane, "rt6_resolves");
    let unwritten = library(&lane, "exf2_fetch_unwritten");
    let blend = library(&lane, "fragment_fetch");
    let shape = Shape { size: SIDE, ..ONE };
    let texture = |pf| lane.texture(pf, shape, TextureUsage::RENDER_TARGET);
    let count_pf = PixelFormat::RGBA32Uint;
    let count_clear = ClearColor::new(1.0, 2.0, 3.0, 4.0);
    let tag_clear = ClearColor::new(7.0, 200.0, 13.0, 255.0);
    for pf in [
        PixelFormat::RGBA8Uint,
        PixelFormat::RGBA16Uint,
        PixelFormat::RGBA32Uint,
    ] {
        let bytes = texel_bytes(pf);
        let reference = cleared(&lane, &texture(pf), tag_clear, bytes);
        let fragment = (&unwritten, "exf2_fetch_unwritten");
        let step = match pipeline(&lane, &vertices, fragment, &[count_pf, pf], 1) {
            Ok(step) => step,
            Err(error @ Error::Unsupported(_)) => {
                eprintln!("refused: the cleared tag of {pf:?} on {device}: {error}");
                continue;
            }
            Err(error) => panic!("the cleared tag of {pf:?} on {device}: {error}"),
        };
        let (count, tag) = (texture(count_pf), texture(pf));
        let targets = [
            (&count, count_clear, StoreAction::Store),
            (&tag, tag_clear, StoreAction::Store),
        ];
        steps(&lane, &pass(&targets, None), &[&step]).unwrap();
        assert_eq!(
            lane.read(&tag, (0, 0), bytes, BlitOption::None),
            reference,
            "the cleared tag of {pf:?} on {device}"
        );
        // Each lane of the count is `c · 3 + t`, of the tag's cleared value.
        let tags: Vec<u32> = match bytes {
            4 => reference[..4].iter().map(|&b| u32::from(b)).collect(),
            8 => reference[..8]
                .chunks(2)
                .map(|c| u32::from(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            _ => reference[..16]
                .chunks(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        };
        let counted: Vec<u8> = (1u32..=4)
            .zip(tags)
            .flat_map(|(c, t)| (c * 3 + t).to_le_bytes())
            .collect();
        assert_eq!(
            lane.read(&count, (0, 0), 16, BlitOption::None),
            counted.repeat((SIDE * SIDE) as usize),
            "the count beside the cleared tag of {pf:?} on {device}"
        );
        eprintln!(
            "framebuffer fetch of a cleared {pf:?} on {device} through {}: the load action's texels",
            form(&lane)
        );
    }
    let extra_clear = ClearColor::new(0.5, 1.0 / 3.0, 0.7, 0.25);
    let factor = buffer_with(
        &lane.device,
        &[1.0f32, 1.0, 1.0, 1.0]
            .iter()
            .flat_map(|c| c.to_le_bytes())
            .collect::<Vec<u8>>(),
    )
    .unwrap();
    for pf in [
        PixelFormat::RG8Unorm,
        PixelFormat::RGBA8Unorm,
        PixelFormat::RG16Float,
        PixelFormat::RGBA16Float,
    ] {
        let bytes = texel_bytes(pf);
        let reference = cleared(&lane, &texture(pf), extra_clear, bytes);
        let formats = [PixelFormat::RGBA16Float, PixelFormat::RGBA16Uint, pf];
        let step = match pipeline(&lane, &vertices, (&blend, "fetch_blend"), &formats, 1) {
            Ok(step) => step,
            Err(error @ Error::Unsupported(_)) => {
                eprintln!("refused: the cleared extra of {pf:?} on {device}: {error}");
                continue;
            }
            Err(error) => panic!("the cleared extra of {pf:?} on {device}: {error}"),
        };
        let (color, ids, extra) = (texture(formats[0]), texture(formats[1]), texture(pf));
        let zero = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        let targets = [
            (&color, zero, StoreAction::Store),
            (&ids, zero, StoreAction::Store),
            (&extra, extra_clear, StoreAction::Store),
        ];
        lane.run(
            &pass(&targets, None),
            |lane, encoder, held: &mut Vec<Buffer>| {
                let depth = buffer_with(&lane.device, &Z.to_le_bytes()).unwrap();
                encoder.set_vertex_buffer(Some(&depth), 0, 0).unwrap();
                encoder.set_fragment_buffer(Some(&factor), 0, 0).unwrap();
                encoder.set_render_pipeline_state(&step).unwrap();
                encoder
                    .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                    .unwrap();
                held.push(depth);
            },
        )
        .unwrap();
        assert_eq!(
            lane.read(&extra, (0, 0), bytes, BlitOption::None),
            reference,
            "the cleared extra of {pf:?} on {device}"
        );
        eprintln!(
            "framebuffer fetch of a cleared {pf:?} on {device} through {}: the load action's texels",
            form(&lane)
        );
    }
    drop(cache);
    quiet(lane, &messages);
}

/// A pipeline of the fullscreen triangle and `fragment` of `library`, drawing `pf`, made with
/// `supportIndirectCommandBuffers`.
fn indirect_pipeline(
    lane: &Lane,
    vertices: &Library,
    fragment: (&Library, &str),
    pf: PixelFormat,
) -> RenderPipelineState {
    indirect_pipeline_at(lane, vertices, fragment, pf, 1)
}

/// [`indirect_pipeline`]'s pipeline at `samples`.
fn indirect_pipeline_at(
    lane: &Lane,
    vertices: &Library,
    (library, fragment): (&Library, &str),
    pf: PixelFormat,
    samples: u32,
) -> RenderPipelineState {
    let function = |library: &Library, name| {
        library
            .new_function_with_name(name)
            .unwrap()
            .function_descriptor()
    };
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(function(vertices, FULL)));
    descriptor.fragment = Some(StageSource::Function(function(library, fragment)));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(pf));
    descriptor.raster_sample_count = samples;
    descriptor.support_indirect_command_buffers = true;
    lane.device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// Commands of an indirect command buffer that fetch the color attachment draw as the encoder's
/// own draws of their pipelines do, between the encoder's draws, in any order with commands that
/// do not (the module docs): each pass's texels are what its steps give in order. Where the
/// device fetches the texture through a scratch image standing in for it (the device's own
/// answer, announced by the encoder's own fetch first: ANV, and MoltenVK past 8²), each
/// command's run reads and writes the scratch, copied from the texture before it and back after,
/// and through input attachments the run's framebuffer holds the scratch in place of the texture,
/// where the commit was refused `NotImplemented(IndirectRenderCommands)`.
#[test]
fn indirect_render_commands_fetch_as_the_encoder_s_draws() {
    let (messages, cache, lane) = lane();
    let vertices = library(&lane, "rt6_resolves");
    let fetch = library(&lane, "fragment_fetch");
    let pf = PixelFormat::R32Float;
    let clear = ClearColor::new(0.75, 0.0, 0.0, 0.0);
    let halve = indirect_pipeline(&lane, &vertices, (&fetch, "fetch_scalar"), pf);
    let white = indirect_pipeline(&lane, &vertices, (&vertices, WHITE), pf);
    // The encoder's own fetch of such a texture: whether the device takes a scratch for it.
    let direct = lane.texture(pf, ONE, TextureUsage::RENDER_TARGET);
    let targets = [(&direct, clear, StoreAction::Store)];
    steps(&lane, &pass(&targets, None), &[&white, &halve]).unwrap();
    let scratch = messages
        .take_notices()
        .iter()
        .filter_map(|message| message.degradation.as_ref())
        .any(|notice| notice.degradation == Degradation::FetchThroughScratch);
    type Steps<'a> = &'a [&'a RenderPipelineState];
    // (name, the encoder's draws before, the commands, the encoder's draws after, the texel)
    let cases: [(&str, Steps, Steps, Steps, f32); 3] = [
        (
            "a fetch between the encoder's draws",
            &[&white],
            &[&halve],
            &[&halve],
            0.25,
        ),
        (
            "fetches about a plain command",
            &[],
            &[&halve, &white, &halve],
            &[&halve],
            0.25,
        ),
        (
            "two fetches before the encoder's plain draw",
            &[&white],
            &[&halve, &halve],
            &[&white, &halve],
            0.5,
        ),
    ];
    let device = lane.device.name();
    for (name, before, commands, after, texel) in cases {
        let target = lane.texture(pf, ONE, TextureUsage::RENDER_TARGET);
        let mut descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.command_types = IndirectCommandType::DRAW;
        descriptor.inherit_buffers = true;
        let icb = lane
            .device
            .new_indirect_command_buffer(
                &descriptor,
                commands.len() as u64,
                ResourceOptions::shared(),
            )
            .unwrap();
        for (index, state) in commands.iter().enumerate() {
            let command = icb.indirect_render_command_at(index as u64).unwrap();
            command.set_render_pipeline_state(state).unwrap();
            command
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
        }
        let command_buffer = lane.queue.command_buffer().unwrap();
        let targets = [(&target, clear, StoreAction::Store)];
        let mut encoder = command_buffer
            .render_command_encoder(&pass(&targets, None))
            .unwrap();
        let depth = buffer_with(&lane.device, &Z.to_le_bytes()).unwrap();
        encoder.set_vertex_buffer(Some(&depth), 0, 0).unwrap();
        let draw = |encoder: &mut metal2vulkan_runtime::RenderCommandEncoder,
                    state: &RenderPipelineState| {
            encoder.set_render_pipeline_state(state).unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
        };
        for state in before {
            draw(&mut encoder, state);
        }
        encoder
            .execute_commands_in_buffer(&icb, 0..commands.len() as u64)
            .unwrap();
        for state in after {
            draw(&mut encoder, state);
        }
        encoder.end_encoding().unwrap();
        if let Err(error) = command_buffer.commit() {
            panic!("{name} on {device}: the commit is refused, {error}");
        }
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None, "{name} on {device}");
        let read = lane.read(&target, (0, 0), 4, BlitOption::None);
        assert_eq!(
            read,
            texel.to_le_bytes().repeat(TEXELS),
            "{name} on {device}"
        );
        let through = if scratch { ", a scratch" } else { "" };
        eprintln!(
            "indirect render commands, {name}, on {device} through {}{through}: drawn as the \
             encoder's",
            form(&lane)
        );
        drop((icb, depth));
    }
    drop(cache);
    quiet(lane, &messages);
}

/// The pass of `descriptor` with the encoder's draws `before` (the fullscreen triangle of each
/// state), an execution of a buffer of one command per state of `commands`, which inherits the
/// encoder's buffers, and the encoder's draws `after`: whether its commit was taken.
fn execute_between(
    lane: &Lane,
    descriptor: &RenderPassDescriptor,
    (before, commands, after): (
        &[&RenderPipelineState],
        &[&RenderPipelineState],
        &[&RenderPipelineState],
    ),
) -> Result<(), Error> {
    let mut icb_descriptor = IndirectCommandBufferDescriptor::new();
    icb_descriptor.command_types = IndirectCommandType::DRAW;
    icb_descriptor.inherit_buffers = true;
    let icb = lane
        .device
        .new_indirect_command_buffer(
            &icb_descriptor,
            commands.len() as u64,
            ResourceOptions::shared(),
        )
        .unwrap();
    for (index, state) in commands.iter().enumerate() {
        let command = icb.indirect_render_command_at(index as u64).unwrap();
        command.set_render_pipeline_state(state).unwrap();
        command
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
    }
    let command_buffer = lane.queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(descriptor).unwrap();
    let depth = buffer_with(&lane.device, &Z.to_le_bytes()).unwrap();
    encoder.set_vertex_buffer(Some(&depth), 0, 0).unwrap();
    let draw = |encoder: &mut metal2vulkan_runtime::RenderCommandEncoder,
                state: &RenderPipelineState| {
        encoder.set_render_pipeline_state(state).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
    };
    for state in before {
        draw(&mut encoder, state);
    }
    encoder
        .execute_commands_in_buffer(&icb, 0..commands.len() as u64)
        .unwrap();
    for state in after {
        draw(&mut encoder, state);
    }
    encoder.end_encoding().unwrap();
    command_buffer.commit()?;
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None, "{}", lane.device.name());
    drop((icb, depth));
    Ok(())
}

/// Commands that fetch a multisampled color attachment draw as the encoder's own draws of their
/// pipelines do (Metal's answer), executed after the encoder's own fetch (0.125 in the resolve),
/// before any (0.25), and about a plain command (0.25): through input attachments (lavapipe), the
/// attachment's multisampled storage image (RADV, NVIDIA), or its sample planes (ANV, MoltenVK),
/// split from the attachment before each command's run and merged into it after, and through the
/// scratch standing in for the texture where the encoder's own fetch takes one (announced), copied
/// in before the run and back after; through input attachments a command that fetches before the
/// encoder's fetch, where that fetch takes a scratch, renders a scratch of its own in its run's
/// framebuffer, where the commit was refused `NotImplemented(IndirectRenderCommands)`. Previously a
/// command after the encoder's fetch through sample planes ran without their split and merge,
/// reading stale samples and its writes lost.
#[test]
fn indirect_render_commands_fetch_multisampled_attachments_as_the_encoder_s_draws() {
    let (messages, cache, lane) = lane();
    let vertices = library(&lane, "rt6_resolves");
    let fetch = library(&lane, "fragment_fetch");
    let pf = PixelFormat::R32Float;
    let clear = ClearColor::new(0.75, 0.0, 0.0, 0.0);
    // The 4x of `Lane::msaa`.
    let samples = 4;
    let halve = indirect_pipeline_at(&lane, &vertices, (&fetch, "fetch_scalar"), pf, samples);
    let white = indirect_pipeline_at(&lane, &vertices, (&vertices, WHITE), pf, samples);
    let planes = reads_planes(&lane);
    // The encoder's own fetch of such a texture: whether the device takes a scratch for it.
    let msaa = lane.msaa(pf);
    let resolve = lane.single(pf);
    let targets = [(&msaa, clear, StoreAction::MultisampleResolve)];
    steps(&lane, &pass(&targets, Some(&resolve)), &[&white, &halve]).unwrap();
    let scratch = messages
        .take_notices()
        .iter()
        .filter_map(|message| message.degradation.as_ref())
        .any(|notice| notice.degradation == Degradation::FetchThroughScratch);
    type Steps<'a> = &'a [&'a RenderPipelineState];
    // (name, the encoder's draws before, the commands, the encoder's draws after, the texel)
    let cases: [(&str, Steps, Steps, Steps, f32); 3] = [
        (
            "a fetch after the encoder's fetch",
            &[&white, &halve],
            &[&halve],
            &[&halve],
            0.125,
        ),
        (
            "a fetch before the encoder's",
            &[&white],
            &[&halve],
            &[&halve],
            0.25,
        ),
        (
            "fetches about a plain command",
            &[&white],
            &[&halve, &white, &halve],
            &[&halve],
            0.25,
        ),
    ];
    let device = lane.device.name();
    for (name, before, commands, after, texel) in cases {
        let msaa = lane.msaa(pf);
        let resolve = lane.single(pf);
        let targets = [(&msaa, clear, StoreAction::MultisampleResolve)];
        let descriptor = pass(&targets, Some(&resolve));
        if let Err(error) = execute_between(&lane, &descriptor, (before, commands, after)) {
            panic!("{name} on {device}: the commit is refused, {error}");
        }
        let read = lane.read(&resolve, (0, 0), 4, BlitOption::None);
        assert_eq!(
            read,
            texel.to_le_bytes().repeat(TEXELS),
            "{name} on {device}"
        );
        let planes = if planes { ", sample planes" } else { "" };
        let through = if scratch { ", a scratch" } else { "" };
        eprintln!(
            "multisampled indirect render commands, {name}, on {device} through {}{planes}\
             {through}: drawn as the encoder's",
            form(&lane)
        );
    }
    drop(cache);
    quiet(lane, &messages);
}

/// Commands of an indirect command buffer that fetch a caller's `R32_SFLOAT` image of color
/// attachment and transfer usage alone (`new_texture_with_vk_image`), which lacks the usage every
/// fetch form needs, draw as the encoder's own draws of their pipelines do (Metal's answer):
/// between the encoder's draws, about a plain command, and after the encoder's own fetch, which
/// makes the pass's scratch (texels 0.25, 0.25, 0.5 and 0.125 of a pass cleared to 0.75). Each
/// fetching command's run reads and writes a scratch a command's draw made, copied from the image
/// the pass's instances render before the run and back after it: in the pixel interlock, and
/// through input attachments (lavapipe) in the run's own framebuffer, where the commit was refused
/// `NotImplemented(IndirectRenderCommands)`. The scratch is announced.
#[test]
fn indirect_render_commands_fetch_a_callers_image_through_a_scratch() {
    let (messages, cache, lane) = lane();
    let device = lane.device.name().to_owned();
    let vertices = library(&lane, "rt6_resolves");
    let fetch = library(&lane, "fragment_fetch");
    let pf = PixelFormat::R32Float;
    let halve = indirect_pipeline(&lane, &vertices, (&fetch, "fetch_scalar"), pf);
    let white = indirect_pipeline(&lane, &vertices, (&vertices, WHITE), pf);
    // SAFETY: used only while the device lives; the image is destroyed after its texture and
    // every command buffer that used it.
    let raw = unsafe { lane.device.raw() };
    // SAFETY: a physical device of the instance.
    let memory = unsafe {
        raw.instance
            .get_physical_device_memory_properties(raw.physical_device)
    };
    let usage = ash::vk::ImageUsageFlags::COLOR_ATTACHMENT
        | ash::vk::ImageUsageFlags::TRANSFER_SRC
        | ash::vk::ImageUsageFlags::TRANSFER_DST;
    let info = shared_among(
        image_info(ash::vk::Format::R32_SFLOAT, usage),
        raw.queue_families,
    );
    // SAFETY: the device is live.
    let (image, image_memory) = unsafe { made_image(raw.device, &memory, &info) };
    let mut descriptor = TextureDescriptor::texture_2d(pf, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    // SAFETY: the image is the device's, made with `info`, and nothing else uses it.
    let texture = unsafe {
        lane.device.new_texture_with_vk_image(
            &descriptor,
            image,
            &info,
            VkImageLayouts::new(
                ash::vk::ImageLayout::UNDEFINED.into(),
                ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL.into(),
            ),
        )
    }
    .unwrap();
    let clear = ClearColor::new(0.75, 0.0, 0.0, 0.0);
    type Steps<'a> = &'a [&'a RenderPipelineState];
    // (name, the encoder's draws before, the commands, the encoder's draws after, the texel)
    let cases: [(&str, Steps, Steps, Steps, f32); 4] = [
        (
            "a fetch between the encoder's draws",
            &[&white],
            &[&halve],
            &[&halve],
            0.25,
        ),
        (
            "fetches about a plain command",
            &[],
            &[&halve, &white, &halve],
            &[&halve],
            0.25,
        ),
        (
            "two fetches before the encoder's plain draw",
            &[&white],
            &[&halve, &halve],
            &[&white, &halve],
            0.5,
        ),
        (
            "a fetch after the encoder's fetch",
            &[&white, &halve],
            &[&halve],
            &[&halve],
            0.125,
        ),
    ];
    for (name, before, commands, after, texel) in cases {
        let targets = [(&texture, clear, StoreAction::Store)];
        let descriptor = pass(&targets, None);
        if let Err(error) = execute_between(&lane, &descriptor, (before, commands, after)) {
            panic!("{name} on {device}: the commit is refused, {error}");
        }
        let read = lane.read(&texture, (0, 0), 4, BlitOption::None);
        assert_eq!(read, floats(texel), "{name} on {device}");
        eprintln!(
            "indirect render commands fetching a caller's image, {name}, on {device} through {}: \
             drawn as the encoder's",
            form(&lane)
        );
    }
    let notices = messages.take_notices();
    assert!(
        notices
            .iter()
            .filter_map(|message| message.degradation.as_ref())
            .any(|notice| notice.degradation == Degradation::FetchThroughScratch),
        "the scratch is announced on {device}"
    );
    drop(texture);
    // SAFETY: nothing uses them: the texture is gone and every command buffer completed.
    unsafe {
        raw.device.destroy_image(image, None);
        raw.device.free_memory(image_memory, None);
    }
    drop(cache);
    quiet(lane, &messages);
}

/// The bytes of `RGBA16Float` or `RG16Float` texels of `lanes` (exact halves).
fn halves(lanes: &[f32]) -> Vec<u8> {
    lanes
        .iter()
        .flat_map(|&x| super::surface_aliases::half_bits(x).to_le_bytes())
        .collect()
}

/// A guest's fetching pipeline in its own attachments: `con_fetch_coverage` reads the `RG16Float`
/// color 1 (write mask none) and blends a `half4` into the `RGB10A8_2P_XR10` color 0, source factor
/// one and destination factor one minus the source's alpha (premultiplied over), beside a second
/// `RGB10A8_2P_XR10` attachment of mask none and a `Depth32Float_Stencil8` attachment, with no
/// vertex descriptor. The pipeline and the pass of three draws are made on every device that
/// fetches through the interlock or through input attachments; a device whose fetch is neither
/// refuses the pipeline by the format, printed. Color 1 and color 2 keep their clears (each texture
/// against one a pass of the clear alone stored); color 0 holds the three draws' `tint * 0.625`
/// over its clear of zero, each channel within four of the format's codes (`extended_range_blends`
/// checks the blend's bytes against Metal's).
#[test]
fn a_guests_fetching_pipeline_blends_an_extended_range_attachment() {
    use metal2vulkan::error::{Feature, Needs};
    use metal2vulkan::metal::{BlendFactor, ColorWriteMask};
    let (messages, cache, lane) = lane();
    let vertices = library(&lane, "rt6_resolves");
    let coverage = library(&lane, "con_fetch_coverage");
    let device = lane.device.name();
    let (color_pf, coverage_pf, depth_pf) = (
        PixelFormat::RGB10A82PXR10,
        PixelFormat::RG16Float,
        PixelFormat::Depth32FloatStencil8,
    );
    let function = |library: &Library, name| {
        library
            .new_function_with_name(name)
            .unwrap()
            .function_descriptor()
    };
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(function(&vertices, FULL)));
    descriptor.fragment = Some(StageSource::Function(function(
        &coverage,
        "con_fetch_coverage",
    )));
    let mut color = ColorAttachmentDescriptor::new(color_pf);
    color.blending_enabled = true;
    color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
    color.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
    let mut covered = ColorAttachmentDescriptor::new(coverage_pf);
    covered.write_mask = ColorWriteMask::NONE;
    let mut unwritten = ColorAttachmentDescriptor::new(color_pf);
    unwritten.write_mask = ColorWriteMask::NONE;
    descriptor.color_attachments[0] = Some(color);
    descriptor.color_attachments[1] = Some(covered);
    descriptor.color_attachments[2] = Some(unwritten);
    descriptor.depth_format = Some(depth_pf);
    descriptor.stencil_format = Some(depth_pf);
    let state = match lane
        .device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
    {
        Ok(state) => state,
        Err(metal2vulkan_runtime::Error::Unsupported(u))
            if u.feature == Feature::Format && u.needs == Needs::NotImplemented =>
        {
            eprintln!("framebuffer fetch a guest's blending pipeline on {device}: refused: {u:?}");
            drop(cache);
            quiet(lane, &messages);
            return;
        }
        Err(error) => panic!("the pipeline on {device}: {error}"),
    };
    let clears = [
        ClearColor::new(0.0, 0.0, 0.0, 0.0),
        ClearColor::new(0.75, 0.25, 0.0, 0.0),
        ClearColor::new(0.125, 0.25, 0.5, 1.0),
    ];
    // Each color cleared and stored by a pass of no draw, and the depth and stencil attachment.
    let pass_of = |targets: &[Texture], depth: Option<&Texture>| {
        let mut pass = RenderPassDescriptor::new();
        for (index, (texture, clear)) in targets.iter().zip(clears).enumerate() {
            let target = &mut pass.color_attachments[index];
            target.texture = Some(texture.clone());
            target.load_action = LoadAction::Clear;
            target.clear_color = clear;
            target.store_action = StoreAction::Store;
        }
        if let Some(depth) = depth {
            pass.depth_attachment.texture = Some(depth.clone());
            pass.depth_attachment.load_action = LoadAction::Clear;
            pass.depth_attachment.clear_depth = 1.0;
            pass.depth_attachment.store_action = StoreAction::DontCare;
            pass.stencil_attachment.texture = Some(depth.clone());
            pass.stencil_attachment.load_action = LoadAction::Clear;
            pass.stencil_attachment.store_action = StoreAction::DontCare;
        }
        pass
    };
    let formats = [color_pf, coverage_pf, color_pf];
    let cleared: Vec<Texture> = formats.iter().map(|&pf| lane.single(pf)).collect();
    lane.run(&pass_of(&cleared, None), |_, _, _: &mut Vec<Buffer>| {})
        .unwrap_or_else(|error| panic!("the clears on {device}: {error}"));
    let targets: Vec<Texture> = formats.iter().map(|&pf| lane.single(pf)).collect();
    let depth = lane.single(depth_pf);
    let tint = halves(&[1.0, 0.5, 0.25, 0.0]);
    lane.run(
        &pass_of(&targets, Some(&depth)),
        |lane, encoder, held: &mut Vec<Buffer>| {
            let z = buffer_with(&lane.device, &Z.to_le_bytes()).unwrap();
            let tint = buffer_with(&lane.device, &tint).unwrap();
            encoder.set_vertex_buffer(Some(&z), 0, 0).unwrap();
            encoder.set_fragment_buffer(Some(&tint), 0, 0).unwrap();
            encoder.set_render_pipeline_state(&state).unwrap();
            for _ in 0..3 {
                encoder
                    .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                    .unwrap();
            }
            held.extend([z, tint]);
        },
    )
    .unwrap_or_else(|error| panic!("the pass on {device}: {error}"));
    let read = |texture: &Texture, bytes| lane.read(texture, (0, 0), bytes, BlitOption::None);
    // `RG16Float`'s texels and `RGB10A8_2P_XR10`'s color words alike, 4 bytes each.
    for index in [1, 2] {
        assert_eq!(
            read(&targets[index], 4),
            read(&cleared[index], 4),
            "color {index} keeps its clear on {device}"
        );
    }
    // Color 0's color words: three 10-bit codes, `(code - 384) / 510` each (red's sum past the
    // format's largest value, 1.875, at its top codes).
    let want = [1.875f64, 0.9375, 0.468_75];
    for word in read(&targets[0], 4).chunks(4) {
        let word = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
        let codes = [(word >> 20) & 0x3FF, (word >> 10) & 0x3FF, word & 0x3FF];
        for (channel, (code, want)) in codes.iter().zip(want).enumerate() {
            let got = (f64::from(*code) - 384.0) / 510.0;
            let near = if want > 1.25 {
                *code >= 1018
            } else {
                (got - want).abs() <= 4.0 / 510.0
            };
            assert!(
                near,
                "color 0's channel {channel} on {device}: code {code}, {got}, want {want}"
            );
        }
    }
    eprintln!(
        "framebuffer fetch a guest's blending pipeline on {device} through {}: made; the fetched and the \
         unwritten attachments keep their clears, the blended one holds the draws",
        form(&lane)
    );
    drop(cache);
    quiet(lane, &messages);
}

/// A guest's fetching pipeline that discards: `tsw_fetch_discard` (without early tests) in
/// `a_guests_fetching_pipeline_blends_an_extended_range_attachment`'s attachments: color 0
/// `RGB10A8_2P_XR10` premultiplied over, color 1 `RG16Float` fetched under mask none, color 2
/// `RGB10A8_2P_XR10` of mask none, and `Depth32Float_Stencil8`. Each case is a pass of one draw at
/// depth 0.5 over a depth clear of 1 and a stencil clear of 0, color 1 cleared to (0.75, 0.25) (the
/// function then writes `tint * 0.625` to color 0) or to (0, 0.25) (it discards). Metal tests each
/// fragment after the function: color 0 holds the write where the fragment survives its discard and
/// its tests, and its clear elsewhere; colors 1 and 2 keep their clears. Cases: no state, depth
/// `Always` written (no test drops a fragment: the pipeline as compiled); a stencil `Equal` of
/// reference 1 and of 0, a depth `Greater` and a depth `Less`, none written (tests drop fragments
/// and write nothing: the early-tests variant in the pixel interlock); a discard under no state and
/// under the passing stencil; a depth `Less` written (in the pixel interlock the function tests the
/// depth itself against the runtime's stand-in; where it was refused). The passing stencil under a
/// `Boolean` visibility result is refused `NotImplemented(FramebufferFetch)` at the draw in the
/// pixel interlock (a store there would precede tests that count), and draws Metal's answer through
/// input attachments. A device whose fetch is neither refuses the pipeline by the format, printed.
#[test]
fn a_discarding_fetch_stores_only_what_metals_later_tests_keep() {
    use metal2vulkan::error::{BindError, Feature, Needs};
    use metal2vulkan::metal::state::DepthStencilDescriptor;
    use metal2vulkan::metal::{BlendFactor, ColorWriteMask, CompareFunction};
    let (messages, cache, lane) = lane();
    let vertices = library(&lane, "rt6_resolves");
    let tested = library(&lane, "tsw_fetch_discard");
    let device = lane.device.name();
    let (color_pf, coverage_pf, depth_pf) = (
        PixelFormat::RGB10A82PXR10,
        PixelFormat::RG16Float,
        PixelFormat::Depth32FloatStencil8,
    );
    let function = |library: &Library, name| {
        library
            .new_function_with_name(name)
            .unwrap()
            .function_descriptor()
    };
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(function(&vertices, FULL)));
    descriptor.fragment = Some(StageSource::Function(function(
        &tested,
        "tsw_fetch_discard",
    )));
    let mut color = ColorAttachmentDescriptor::new(color_pf);
    color.blending_enabled = true;
    color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
    color.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
    let mut covered = ColorAttachmentDescriptor::new(coverage_pf);
    covered.write_mask = ColorWriteMask::NONE;
    let mut unwritten = ColorAttachmentDescriptor::new(color_pf);
    unwritten.write_mask = ColorWriteMask::NONE;
    descriptor.color_attachments[0] = Some(color);
    descriptor.color_attachments[1] = Some(covered);
    descriptor.color_attachments[2] = Some(unwritten);
    descriptor.depth_format = Some(depth_pf);
    descriptor.stencil_format = Some(depth_pf);
    let state = match lane
        .device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
    {
        Ok(state) => state,
        Err(metal2vulkan_runtime::Error::Unsupported(u))
            if u.feature == Feature::Format && u.needs == Needs::NotImplemented =>
        {
            eprintln!(
                "framebuffer fetch a guest's discarding pipeline on {device}: refused: {u:?}"
            );
            drop(cache);
            quiet(lane, &messages);
            return;
        }
        Err(error) => panic!("the pipeline on {device}: {error}"),
    };
    let interlock = !through_inputs(&lane);
    let mut equal = DepthStencilDescriptor::default();
    for face in [&mut equal.front_face_stencil, &mut equal.back_face_stencil] {
        face.stencil_compare_function = CompareFunction::Equal;
    }
    let depth = |compare: CompareFunction, write: bool| {
        let mut d = DepthStencilDescriptor::default();
        d.depth_compare_function = compare;
        d.depth_write_enabled = write;
        d
    };
    /// What Metal leaves in color 0, and whether the pixel interlock refuses the draw.
    #[derive(Clone, Copy, PartialEq, Debug)]
    enum Want {
        Stored,
        Kept,
        Refused,
    }
    let cases = [
        ("no state", None, 0, false, false, Want::Stored),
        (
            "depth always, written",
            Some(depth(CompareFunction::Always, true)),
            0,
            false,
            false,
            Want::Stored,
        ),
        (
            "stencil equal, failing",
            Some(equal),
            1,
            false,
            false,
            Want::Kept,
        ),
        (
            "stencil equal, passing",
            Some(equal),
            0,
            false,
            false,
            Want::Stored,
        ),
        (
            "depth greater, unwritten",
            Some(depth(CompareFunction::Greater, false)),
            0,
            false,
            false,
            Want::Kept,
        ),
        (
            "depth less, unwritten",
            Some(depth(CompareFunction::Less, false)),
            0,
            false,
            false,
            Want::Stored,
        ),
        ("discarded, no state", None, 0, true, false, Want::Kept),
        (
            "discarded, stencil passing",
            Some(equal),
            0,
            true,
            false,
            Want::Kept,
        ),
        (
            "depth less, written",
            Some(depth(CompareFunction::Less, true)),
            0,
            false,
            false,
            Want::Stored,
        ),
        (
            "stencil passing, counted",
            Some(equal),
            0,
            false,
            true,
            Want::Refused,
        ),
    ];
    let tint = halves(&[1.0, 0.5, 0.25, 0.0]);
    let read = |texture: &Texture, bytes| lane.read(texture, (0, 0), bytes, BlitOption::None);
    // Color 0's color words: three 10-bit codes, `(code - 384) / 510` each, within four codes.
    let codes = |texture: &Texture, want: [f64; 3], case: &str| {
        for word in read(texture, 4).chunks(4) {
            let word = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
            let codes = [(word >> 20) & 0x3FF, (word >> 10) & 0x3FF, word & 0x3FF];
            for (channel, (code, want)) in codes.iter().zip(want).enumerate() {
                let got = (f64::from(*code) - 384.0) / 510.0;
                assert!(
                    (got - want).abs() <= 4.0 / 510.0,
                    "{case}: color 0's channel {channel} on {device}: code {code}, {got}, want {want}"
                );
            }
        }
    };
    for (case, ds, reference, discards, counted, want) in cases {
        let formats = [color_pf, coverage_pf, color_pf];
        let targets: Vec<Texture> = formats.iter().map(|&pf| lane.single(pf)).collect();
        let depth_texture = lane.single(depth_pf);
        let clears = [
            ClearColor::new(0.0, 0.0, 0.0, 0.0),
            ClearColor::new(if discards { 0.0 } else { 0.75 }, 0.25, 0.0, 0.0),
            ClearColor::new(0.125, 0.25, 0.5, 1.0),
        ];
        let mut pass = RenderPassDescriptor::new();
        for (index, (texture, clear)) in targets.iter().zip(clears).enumerate() {
            let target = &mut pass.color_attachments[index];
            target.texture = Some(texture.clone());
            target.load_action = LoadAction::Clear;
            target.clear_color = clear;
            target.store_action = StoreAction::Store;
        }
        pass.depth_attachment.texture = Some(depth_texture.clone());
        pass.depth_attachment.load_action = LoadAction::Clear;
        pass.depth_attachment.clear_depth = 1.0;
        pass.depth_attachment.store_action = StoreAction::DontCare;
        pass.stencil_attachment.texture = Some(depth_texture.clone());
        pass.stencil_attachment.load_action = LoadAction::Clear;
        pass.stencil_attachment.store_action = StoreAction::DontCare;
        let results = buffer_with(&lane.device, &[0u8; 8]).unwrap();
        if counted {
            pass.visibility_result_buffer = Some(results.clone());
        }
        let ds_state = ds.map(|d| lane.device.new_depth_stencil_state(&d).unwrap());
        let mut drawn = None;
        lane.run(&pass, |lane, encoder, held: &mut Vec<Buffer>| {
            let z = buffer_with(&lane.device, &Z.to_le_bytes()).unwrap();
            let tint = buffer_with(&lane.device, &tint).unwrap();
            encoder.set_vertex_buffer(Some(&z), 0, 0).unwrap();
            encoder.set_fragment_buffer(Some(&tint), 0, 0).unwrap();
            encoder.set_render_pipeline_state(&state).unwrap();
            if let Some(ds_state) = &ds_state {
                encoder.set_depth_stencil_state(ds_state).unwrap();
            }
            encoder.set_stencil_reference_value(reference);
            if counted {
                encoder
                    .set_visibility_result_mode(VisibilityResultMode::Boolean, 0)
                    .unwrap();
            }
            drawn = Some(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
            if counted {
                encoder
                    .set_visibility_result_mode(VisibilityResultMode::Disabled, 0)
                    .unwrap();
            }
            held.extend([z, tint]);
        })
        .unwrap_or_else(|error| panic!("{case}: the pass on {device}: {error}"));
        let drawn = drawn.expect("the draw");
        let stored = match (want, interlock) {
            (Want::Refused, true) => {
                match drawn {
                    Err(
                        metal2vulkan_runtime::Error::Unsupported(u)
                        | metal2vulkan_runtime::Error::Bind(BindError::Unsupported(u)),
                    ) => assert_eq!(
                        (u.feature, u.needs),
                        (Feature::FramebufferFetch, Needs::NotImplemented),
                        "{case} on {device}"
                    ),
                    other => panic!("{case} on {device}: {other:?}"),
                }
                false
            }
            (Want::Refused, false) | (Want::Stored, _) => {
                drawn.unwrap_or_else(|error| panic!("{case} on {device}: {error}"));
                true
            }
            (Want::Kept, _) => {
                drawn.unwrap_or_else(|error| panic!("{case} on {device}: {error}"));
                false
            }
        };
        let want0 = match stored {
            true => [0.625, 0.3125, 0.156_25],
            false => [0.0; 3],
        };
        codes(&targets[0], want0, case);
        assert_eq!(
            read(&targets[1], 4),
            halves(&[if discards { 0.0 } else { 0.75 }, 0.25]).repeat(TEXELS),
            "{case}: color 1 keeps its clear on {device}"
        );
        if counted && stored {
            let words = read_buffer(&results, 0, 8).unwrap();
            assert_ne!(words, [0u8; 8], "{case}: the visibility result on {device}");
        }
        eprintln!(
            "framebuffer fetch a guest's discarding pipeline, {case}, on {device} through {}: \
             Metal's answer ({})",
            form(&lane),
            match (want, interlock) {
                (Want::Refused, true) => "refused at the draw",
                _ if stored => "stored",
                _ => "kept",
            }
        );
    }
    drop(cache);
    quiet(lane, &messages);
}

/// A state whose fragment function stores before Metal's tests makes, at its creation, the
/// pipeline a first draw whose tests drop fragments and write nothing takes: `tsw_fetch_discard` in
/// `a_discarding_fetch_stores_only_what_metals_later_tests_keep`'s attachments and pass. Its
/// creation makes one pipeline (in the pixel interlock the early-tests variant's, through input
/// attachments the pipeline as compiled); a draw under a depth `Less` test that does not write
/// makes none and takes it, and color 0 holds Metal's write. A device whose fetch is neither
/// refuses the pipeline by the format, printed.
#[test]
fn a_state_storing_before_its_tests_makes_its_first_draw_s_pipeline_at_creation() {
    use metal2vulkan::error::{Feature, Needs};
    use metal2vulkan::metal::state::DepthStencilDescriptor;
    use metal2vulkan::metal::{BlendFactor, ColorWriteMask, CompareFunction};
    let (messages, cache, lane) = lane();
    let vertices = library(&lane, "rt6_resolves");
    let tested = library(&lane, "tsw_fetch_discard");
    let device = lane.device.name();
    let (color_pf, coverage_pf, depth_pf) = (
        PixelFormat::RGB10A82PXR10,
        PixelFormat::RG16Float,
        PixelFormat::Depth32FloatStencil8,
    );
    let function = |library: &Library, name| {
        library
            .new_function_with_name(name)
            .unwrap()
            .function_descriptor()
    };
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(function(&vertices, FULL)));
    descriptor.fragment = Some(StageSource::Function(function(
        &tested,
        "tsw_fetch_discard",
    )));
    let mut color = ColorAttachmentDescriptor::new(color_pf);
    color.blending_enabled = true;
    color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
    color.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
    let mut covered = ColorAttachmentDescriptor::new(coverage_pf);
    covered.write_mask = ColorWriteMask::NONE;
    let mut unwritten = ColorAttachmentDescriptor::new(color_pf);
    unwritten.write_mask = ColorWriteMask::NONE;
    descriptor.color_attachments[0] = Some(color);
    descriptor.color_attachments[1] = Some(covered);
    descriptor.color_attachments[2] = Some(unwritten);
    descriptor.depth_format = Some(depth_pf);
    descriptor.stencil_format = Some(depth_pf);
    let before = lane.device.work_counts();
    let started = std::time::Instant::now();
    let state = match lane
        .device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
    {
        Ok(state) => state,
        Err(metal2vulkan_runtime::Error::Unsupported(u))
            if u.feature == Feature::Format && u.needs == Needs::NotImplemented =>
        {
            eprintln!("a state storing before its tests on {device}: refused: {u:?}");
            drop(cache);
            quiet(lane, &messages);
            return;
        }
        Err(error) => panic!("the pipeline on {device}: {error}"),
    };
    let creation = started.elapsed();
    let created = lane.device.work_counts();
    assert_eq!(
        created.creation_pipelines_made - before.creation_pipelines_made,
        1,
        "the first draw's pipeline, made at the creation on {device} through {}",
        form(&lane)
    );
    let mut less = DepthStencilDescriptor::default();
    less.depth_compare_function = CompareFunction::Less;
    less.depth_write_enabled = false;
    let ds_state = lane.device.new_depth_stencil_state(&less).unwrap();
    let formats = [color_pf, coverage_pf, color_pf];
    let targets: Vec<Texture> = formats.iter().map(|&pf| lane.single(pf)).collect();
    let depth_texture = lane.single(depth_pf);
    let clears = [
        ClearColor::new(0.0, 0.0, 0.0, 0.0),
        ClearColor::new(0.75, 0.25, 0.0, 0.0),
        ClearColor::new(0.125, 0.25, 0.5, 1.0),
    ];
    let mut pass = RenderPassDescriptor::new();
    for (index, (texture, clear)) in targets.iter().zip(clears).enumerate() {
        let target = &mut pass.color_attachments[index];
        target.texture = Some(texture.clone());
        target.load_action = LoadAction::Clear;
        target.clear_color = clear;
        target.store_action = StoreAction::Store;
    }
    pass.depth_attachment.texture = Some(depth_texture.clone());
    pass.depth_attachment.load_action = LoadAction::Clear;
    pass.depth_attachment.clear_depth = 1.0;
    pass.depth_attachment.store_action = StoreAction::DontCare;
    pass.stencil_attachment.texture = Some(depth_texture.clone());
    pass.stencil_attachment.load_action = LoadAction::Clear;
    pass.stencil_attachment.store_action = StoreAction::DontCare;
    let tint = halves(&[1.0, 0.5, 0.25, 0.0]);
    let mut drawn = None;
    lane.run(&pass, |lane, encoder, held: &mut Vec<Buffer>| {
        let z = buffer_with(&lane.device, &Z.to_le_bytes()).unwrap();
        let tint = buffer_with(&lane.device, &tint).unwrap();
        encoder.set_vertex_buffer(Some(&z), 0, 0).unwrap();
        encoder.set_fragment_buffer(Some(&tint), 0, 0).unwrap();
        encoder.set_render_pipeline_state(&state).unwrap();
        encoder.set_depth_stencil_state(&ds_state).unwrap();
        drawn = Some(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
        held.extend([z, tint]);
    })
    .unwrap_or_else(|error| panic!("the pass on {device}: {error}"));
    drawn
        .expect("the draw")
        .unwrap_or_else(|error| panic!("the draw on {device}: {error}"));
    // Its constant slots bound at offset 0, the draw's key reads them at the bind offset alone
    // where the creation's reads them at the residual: it draws with the creation's pipeline, its
    // own key made on the pool behind it, nothing waiting (`pipeline/render/residual.rs`), and
    // counted once the pool settles.
    let after = pipelines_settled(&lane.device);
    let behind = after.residual_form_makings - created.residual_form_makings;
    assert_eq!(
        (
            after.variant_pipelines_made - created.variant_pipelines_made,
            after.creation_pipelines_drawn - created.creation_pipelines_drawn,
        ),
        (behind, 1),
        "the draw took the pipeline made at the creation on {device} through {}, nothing made \
         but its own key behind it ({behind})",
        form(&lane)
    );
    // Color 0's color words: three 10-bit codes, `(code - 384) / 510` each, within four codes.
    for word in lane
        .read(&targets[0], (0, 0), 4, BlitOption::None)
        .chunks(4)
    {
        let word = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
        let codes = [(word >> 20) & 0x3FF, (word >> 10) & 0x3FF, word & 0x3FF];
        for (channel, (code, want)) in codes.iter().zip([0.625, 0.3125, 0.156_25]).enumerate() {
            let got = (f64::from(*code) - 384.0) / 510.0;
            assert!(
                (got - want).abs() <= 4.0 / 510.0,
                "color 0's channel {channel} on {device}: code {code}, {got}, want {want}"
            );
        }
    }
    eprintln!(
        "a state storing before its tests on {device} through {}: its first draw took the \
         pipeline made at its creation (the creation took {:.1} ms), Metal's answer",
        form(&lane),
        creation.as_secs_f64() * 1e3
    );
    drop(cache);
    quiet(lane, &messages);
}

/// A fetched attachment's write mask, in the shape of a guest's path-coverage pipeline:
/// `con_fetch_coverage` reads the `RG16Float` color 1 and writes it under a mask of green alone,
/// beside a blended `RGBA16Float` color 0 it writes and an `RGBA16Float` color 2 it never writes
/// (mask none), with a depth and stencil attachment. Color 1 cleared to (0.75, 0.25), three draws
/// in order leave (0.75, 2^-8): red, which the mask leaves out, keeps the clear, and green is a
/// quarter of what each draw read. Color 0, cleared to zero and blended `One`,
/// `OneMinusSourceAlpha` with a tint of alpha 0, holds the sum of the three draws' `tint * w`
/// (`w` = 0.625, 0.4375 and 0.390625, every value an exact half); color 2 keeps its clear. Then the
/// same pipeline under a mask of red alone, a state of its own made after the first: green keeps
/// the clear, and red follows `r / 2 + 0.25` (`w` = 0.625, 0.5625 and 0.53125). Through input
/// attachments the pipeline's mask is Vulkan's; in the pixel interlock the storage write reads the
/// texel first and writes the kept channel back as it read it, both masks through one module whose
/// draws name the channel kept.
#[test]
fn masked_fetched_writes_keep_the_channels_the_mask_leaves_out() {
    use metal2vulkan::metal::{BlendFactor, ColorWriteMask};
    let (messages, cache, lane) = lane();
    let vertices = library(&lane, "rt6_resolves");
    let coverage = library(&lane, "con_fetch_coverage");
    let device = lane.device.name();
    let (color_pf, coverage_pf, depth_pf) = (
        PixelFormat::RGBA16Float,
        PixelFormat::RG16Float,
        PixelFormat::Depth32FloatStencil8,
    );
    let function = |library: &Library, name| {
        library
            .new_function_with_name(name)
            .unwrap()
            .function_descriptor()
    };
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(function(&vertices, FULL)));
    descriptor.fragment = Some(StageSource::Function(function(
        &coverage,
        "con_fetch_coverage",
    )));
    let mut color = ColorAttachmentDescriptor::new(color_pf);
    color.blending_enabled = true;
    color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
    color.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
    let mut unwritten = ColorAttachmentDescriptor::new(color_pf);
    unwritten.write_mask = ColorWriteMask::NONE;
    descriptor.color_attachments[0] = Some(color);
    descriptor.color_attachments[2] = Some(unwritten);
    descriptor.depth_format = Some(depth_pf);
    descriptor.stencil_format = Some(depth_pf);
    // Each mask: the coverage it leaves, and the three draws' `w`.
    for (mask, coverage_left, w) in [
        (
            ColorWriteMask::GREEN,
            [0.75, 0.003_906_25],
            [0.625, 0.4375, 0.390_625],
        ),
        (
            ColorWriteMask::RED,
            [0.531_25, 0.25],
            [0.625, 0.5625, 0.531_25],
        ),
    ] {
        let mut covered = ColorAttachmentDescriptor::new(coverage_pf);
        covered.write_mask = mask;
        descriptor.color_attachments[1] = Some(covered);
        let state = lane
            .device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap_or_else(|error| panic!("the pipeline of {mask:?} on {device}: {error}"));
        let targets = [
            lane.single(color_pf),
            lane.single(coverage_pf),
            lane.single(color_pf),
        ];
        let depth = lane.single(depth_pf);
        let clears = [
            ClearColor::new(0.0, 0.0, 0.0, 0.0),
            ClearColor::new(0.75, 0.25, 0.0, 0.0),
            ClearColor::new(0.125, 0.25, 0.5, 1.0),
        ];
        let mut pass = RenderPassDescriptor::new();
        for (index, (texture, clear)) in targets.iter().zip(clears).enumerate() {
            let target = &mut pass.color_attachments[index];
            target.texture = Some(texture.clone());
            target.load_action = LoadAction::Clear;
            target.clear_color = clear;
            target.store_action = StoreAction::Store;
        }
        pass.depth_attachment.texture = Some(depth.clone());
        pass.depth_attachment.load_action = LoadAction::Clear;
        pass.depth_attachment.clear_depth = 1.0;
        pass.depth_attachment.store_action = StoreAction::DontCare;
        pass.stencil_attachment.texture = Some(depth.clone());
        pass.stencil_attachment.load_action = LoadAction::Clear;
        pass.stencil_attachment.store_action = StoreAction::DontCare;
        let tint = halves(&[1.0, 0.5, 0.25, 0.0]);
        lane.run(&pass, |lane, encoder, held: &mut Vec<Buffer>| {
            let z = buffer_with(&lane.device, &Z.to_le_bytes()).unwrap();
            let tint = buffer_with(&lane.device, &tint).unwrap();
            encoder.set_vertex_buffer(Some(&z), 0, 0).unwrap();
            encoder.set_fragment_buffer(Some(&tint), 0, 0).unwrap();
            encoder.set_render_pipeline_state(&state).unwrap();
            for _ in 0..3 {
                encoder
                    .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                    .unwrap();
            }
            held.extend([z, tint]);
        })
        .unwrap_or_else(|error| panic!("the pass on {device}: {error}"));
        let sum: f32 = w.iter().sum();
        let read = |texture: &Texture, bytes| lane.read(texture, (0, 0), bytes, BlitOption::None);
        assert_eq!(
            read(&targets[1], 4),
            halves(&coverage_left).repeat(TEXELS),
            "the fetched attachment under {mask:?} on {device}"
        );
        assert_eq!(
            read(&targets[0], 8),
            halves(&[sum, sum * 0.5, sum * 0.25, 0.0]).repeat(TEXELS),
            "the blended attachment under {mask:?} on {device}"
        );
        assert_eq!(
            read(&targets[2], 8),
            halves(&[0.125, 0.25, 0.5, 1.0]).repeat(TEXELS),
            "the unwritten attachment under {mask:?} on {device}"
        );
        eprintln!(
            "framebuffer fetch a masked fetched write ({mask:?}) on {device} through {}: Metal's \
         answer, the unmasked channel and the unwritten attachment kept",
            form(&lane)
        );
    }
    drop(cache);
    quiet(lane, &messages);
}
