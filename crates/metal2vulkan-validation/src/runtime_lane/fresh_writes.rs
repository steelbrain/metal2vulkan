//! A new texture written by blits, a resolve or separate attachments, with no clearing pass,
//! then read (the consumer's move32: a buffer copied into a new 4-sample `Depth32Float` and
//! copied back out died at `fresh-pattern-zeroed`), on the device the cfg selects, under the
//! validation layer.
//!
//! A new compressed `Private` texture whose pattern the device cannot write reads zero where Metal
//! reads its pattern, announced (`fresh-pattern-zeroed`) only where a command first reads a texel
//! still unwritten (`runtime::fresh_unwritten`). A blit's destination box, a pass's clear over its
//! render area and a resolve over it write their texels, so the read after them sees what they
//! wrote and announces nothing.
//!
//! The first test writes each new texture whole by one blit; the second writes them in parts
//! (four quarters in an order of no rows or columns), reads back only the written half of one
//! (a read that reaches no unwritten texel), resolves a pass into a new depth, renders separate
//! depth and stencil textures, and writes a level of a mipmapped depth through a view. Each read
//! compares the bytes and checks no `fresh-pattern-zeroed` was announced. The controls, a new
//! 4-sample `Depth32Float` copied out unwritten and the half-written one read whole, announce it
//! once (every device notes one: no clear writes its pattern, a NaN), so the tracker is live
//! where the cases ran. A single-sample `Depth32Float` is noted only where the device writes no
//! all-ones depth, so its cases are checked there and pass trivially elsewhere. On a device without
//! Metal's compression (`DeviceProperties::lossless_compression` false) no new texture holds a
//! pattern, so the controls announce nothing (`fresh_contents::patterns_of`).

use super::degrade_notices::shared_in_system_memory;
use super::degrade_notices::{assert_made_once, assert_once, clamps_depth_uploads};
use super::degrade_notices::{
    draws_depth_copies, forced, loses_depth_denormals, signs_depth_tests,
};
use super::fresh_contents::{patterns_of, unfilled};
use super::{
    buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device, Messages,
    ScratchCache, WAIT,
};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan_runtime::{
    BlitOption, Buffer, CommandBufferStatus, CommandQueue, Degradation, Device, LoadAction,
    MultisampleDepthResolveFilter, Origin, RenderPassDescriptor, ResourceOptions, Size,
    StoreAction, Texture, TextureDescriptor,
};
use std::collections::BTreeSet;

/// A new `Private` render target of `pf`, `side` square at `samples`, mipmapped where
/// `mipmapped`.
fn fresh(device: &Device, pf: PixelFormat, side: u64, samples: u64, mipmapped: bool) -> Texture {
    let mut d = TextureDescriptor::texture_2d(pf, side, side, mipmapped);
    if samples > 1 {
        d.texture_type = TextureType::Type2DMultisample;
        d.sample_count = samples;
    }
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::private();
    device.new_texture(&d).unwrap()
}

/// A box of a texture's `level`: its origin `at` and `size`, `[x, y]`, of `samples` 4-byte
/// units a pixel.
#[derive(Clone, Copy, Debug)]
struct Texels {
    level: u64,
    at: [u64; 2],
    size: [u64; 2],
    samples: u64,
}

impl Texels {
    /// The whole of level 0, `side` square.
    fn whole(side: u64, samples: u64) -> Texels {
        Texels {
            level: 0,
            at: [0, 0],
            size: [side, side],
            samples,
        }
    }

    /// Its rows' bytes, tight.
    fn row(&self) -> u64 {
        self.size[0] * self.samples * 4
    }

    /// Its bytes, tight.
    fn bytes(&self) -> u64 {
        self.row() * self.size[1]
    }
}

/// The bytes a `side` square of `samples` at 4 bytes a unit takes, tight.
fn layout(side: u64, samples: u64) -> (u64, u64) {
    let whole = Texels::whole(side, samples);
    (whole.row(), whole.bytes())
}

/// Distinct depths in `(0, 1)`, one a unit.
fn depths(units: u64) -> Vec<u8> {
    (0..units)
        .flat_map(|i| ((i + 1) as f32 / (units + 1) as f32).to_le_bytes())
        .collect()
}

/// Copy `buffer`'s bytes from `offset`, its rows `row` bytes apart, into `texels` of `texture`,
/// in one command buffer.
fn copy_in(
    queue: &CommandQueue,
    (buffer, offset, row): (&Buffer, u64, u64),
    texture: &Texture,
    texels: Texels,
) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    let [x, y] = texels.at;
    blit.copy_from_buffer_to_texture(
        buffer,
        offset,
        row,
        row * texels.size[1],
        Size::new(texels.size[0], texels.size[1], 1),
        texture,
        0,
        texels.level,
        Origin::new(x, y, 0),
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// `texels` of `texture` copied out, tight, in their own command buffer.
fn copy_out(device: &Device, queue: &CommandQueue, texture: &Texture, texels: Texels) -> Vec<u8> {
    copy_out_slice(device, queue, texture, 0, texels)
}

/// `texels` of `slice` of `texture` copied out, tight, in their own command buffer.
fn copy_out_slice(
    device: &Device,
    queue: &CommandQueue,
    texture: &Texture,
    slice: u64,
    texels: Texels,
) -> Vec<u8> {
    let (row, bytes) = (texels.row(), texels.bytes());
    let out = buffer_with(device, &vec![0xee; bytes as usize]).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    let [x, y] = texels.at;
    blit.copy_from_texture_to_buffer(
        texture,
        slice,
        texels.level,
        Origin::new(x, y, 0),
        Size::new(texels.size[0], texels.size[1], 1),
        &out,
        0,
        row,
        bytes,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    read_buffer(&out, 0, bytes).unwrap()
}

/// Run `pass` with no draw call, in its own command buffer; `false` where the device refuses
/// the pass (printed).
fn run_pass(name: &str, queue: &CommandQueue, pass: &RenderPassDescriptor, case: &str) -> bool {
    let command_buffer = queue.command_buffer().unwrap();
    let encoder = match command_buffer.render_command_encoder(pass) {
        Ok(encoder) => encoder,
        Err(error) => {
            println!("FRESH-WRITE {name} {case}: the pass is refused: {error:?}");
            return false;
        }
    };
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    true
}

/// The kinds copies of a buffer's bytes into `Depth32Float` textures announce on `device`, where
/// a device announces them first: what a float depth's write may store
/// (`degrade_notices::loses_depth_denormals`, `clamps_depth_uploads`, `signs_depth_tests`: the
/// textures render), the draws that copy into
/// a multisampled one where `multisampled` (`draws_depth_copies`), and the `Shared` buffers.
fn copied_in(device: &Device, multisampled: bool) -> BTreeSet<Degradation> {
    forced(&[
        (
            Degradation::DepthDenormals,
            loses_depth_denormals(device, false),
        ),
        (
            Degradation::RestrictedDepthUpload,
            clamps_depth_uploads(device),
        ),
        (Degradation::SignedDepthTest, signs_depth_tests(device)),
        (
            Degradation::DrawnDepthCopy,
            multisampled && draws_depth_copies(device),
        ),
        (
            Degradation::SharedInSystemMemory,
            shared_in_system_memory(device),
        ),
    ])
}

/// `count` words of `bits`.
fn words(bits: u32, count: u64) -> Vec<u8> {
    (0..count).flat_map(|_| bits.to_le_bytes()).collect()
}

/// Each blit writing a new texture whole, then a read, announces nothing; the control does.
#[test]
fn a_new_texture_a_blit_wrote_whole_reads_its_texels_unannounced() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let queue = device.new_command_queue().unwrap();
        messages.take_notices();
        // A buffer into a new multisampled depth (the consumer's screensaver and login shapes,
        // per sample) and into a new single-sample one.
        for (side, samples) in [(54u64, 4u64), (80, 4), (64, 1)] {
            let (_, image) = layout(side, samples);
            let bytes = depths(image / 4);
            let source = buffer_with(&device, &bytes).unwrap();
            let texture = fresh(&device, PixelFormat::Depth32Float, side, samples, false);
            let whole = Texels::whole(side, samples);
            copy_in(&queue, (&source, 0, whole.row()), &texture, whole);
            let back = copy_out(&device, &queue, &texture, whole);
            assert_eq!(
                back, bytes,
                "{name} {side}x{side} at {samples}: the copy back"
            );
            println!("FRESH-WRITE {name} buffer into Depth32Float {side}x{side} at {samples}: read back whole");
        }
        // A texture into a new one: written whole by the copy.
        let (side, samples) = (54u64, 4u64);
        let (_, image) = layout(side, samples);
        let bytes = depths(image / 4);
        let source = buffer_with(&device, &bytes).unwrap();
        let whole = Texels::whole(side, samples);
        let from = fresh(&device, PixelFormat::Depth32Float, side, samples, false);
        copy_in(&queue, (&source, 0, whole.row()), &from, whole);
        let to = fresh(&device, PixelFormat::Depth32Float, side, samples, false);
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_texture(&from, &to).unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let back = copy_out(&device, &queue, &to, whole);
        assert_eq!(back, bytes, "{name}: the texture copy's copy back");
        println!("FRESH-WRITE {name} texture into Depth32Float {side}x{side} at {samples}: read back whole");
        let quiet = assert_made_once(&messages, &[], &copied_in(&device, true));
        assert!(
            !quiet.contains_key(&Degradation::FreshPatternZeroed),
            "{name}: a texture a blit wrote whole announced its pattern: {quiet:?}"
        );
        // The control: a new 4-sample depth read unwritten announces once, where the device
        // notes it.
        let unwritten = fresh(&device, PixelFormat::Depth32Float, side, samples, false);
        copy_out(&device, &queue, &unwritten, whole);
        let control = assert_once(&messages.take_notices(), patterns_of(&device));
        println!("FRESH-WRITE {name} control, read unwritten: announced {control:?}");
    }
    layer_reported_nothing_at_all(&messages);
}

/// Writes of parts, a resolve, separate attachments and a view's write, each of a new texture
/// with no clearing pass of it, then reads of what they wrote: nothing announced; the controls
/// (the half-written texture read whole) announce once (the module docs).
#[test]
fn a_new_texture_written_in_parts_or_by_a_pass_reads_its_texels_unannounced() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let queue = device.new_command_queue().unwrap();
        messages.take_notices();
        let depth = PixelFormat::Depth32Float;
        let (side, samples) = (32u64, 4u64);
        let whole = Texels::whole(side, samples);
        let quarters_bytes = depths(whole.bytes() / 4);
        let quarters_source = buffer_with(&device, &quarters_bytes).unwrap();
        let (bytes, source) = (&quarters_bytes, &quarters_source);
        let half = side / 2;
        // Four quarters, each its own copy from the whole's bytes, in an order of no rows or
        // columns: together they write the level.
        let quarters = fresh(&device, depth, side, samples, false);
        for [x, y] in [[half, 0], [0, half], [half, half], [0, 0]] {
            let offset = y * whole.row() + x * samples * 4;
            let quarter = Texels {
                at: [x, y],
                size: [half, half],
                ..whole
            };
            copy_in(&queue, (source, offset, whole.row()), &quarters, quarter);
        }
        let back = copy_out(&device, &queue, &quarters, whole);
        assert_eq!(&back, bytes, "{name}: the quarters' copy back");
        println!("FRESH-WRITE {name} four quarters into Depth32Float {side}x{side} at {samples}: read back whole");
        // The top half written, then read: the read reaches no unwritten texel.
        let halves = fresh(&device, depth, side, samples, false);
        let top = Texels {
            size: [side, half],
            ..whole
        };
        copy_in(&queue, (source, 0, whole.row()), &halves, top);
        let back = copy_out(&device, &queue, &halves, top);
        assert_eq!(
            back,
            bytes[..top.bytes() as usize],
            "{name}: the top half's copy back"
        );
        println!("FRESH-WRITE {name} top half into Depth32Float {side}x{side} at {samples}: read back the half");
        // A pass's clear of a new 4-sample depth resolved into a new single-sample one: the
        // resolve writes its render area.
        let attachment = fresh(&device, depth, side, samples, false);
        let resolved = fresh(&device, depth, side, 1, false);
        let mut pass = RenderPassDescriptor::new();
        let target = &mut pass.depth_attachment;
        target.texture = Some(attachment.clone());
        target.load_action = LoadAction::Clear;
        target.clear_depth = 0.25;
        target.store_action = StoreAction::MultisampleResolve;
        target.resolve_texture = Some(resolved.clone());
        target.depth_resolve_filter = MultisampleDepthResolveFilter::Sample0;
        let resolved_ran = run_pass(&name, &queue, &pass, "resolve");
        if resolved_ran {
            let back = copy_out(&device, &queue, &resolved, Texels::whole(side, 1));
            assert_eq!(back, words(0x3e80_0000, side * side), "{name}: the resolve");
            println!("FRESH-WRITE {name} 4x clear resolved into Depth32Float {side}x{side}: read back whole");
        }
        // Separate depth and stencil textures, each cleared and stored.
        let (separate, stencil) = (
            fresh(&device, depth, side, 1, false),
            fresh(&device, PixelFormat::Stencil8, side, 1, false),
        );
        let mut pass = RenderPassDescriptor::new();
        pass.depth_attachment.texture = Some(separate.clone());
        pass.depth_attachment.load_action = LoadAction::Clear;
        pass.depth_attachment.clear_depth = 0.25;
        pass.depth_attachment.store_action = StoreAction::Store;
        pass.stencil_attachment.texture = Some(stencil.clone());
        pass.stencil_attachment.load_action = LoadAction::Clear;
        pass.stencil_attachment.clear_stencil = 0x5a;
        pass.stencil_attachment.store_action = StoreAction::Store;
        let separated = run_pass(&name, &queue, &pass, "separate depth and stencil");
        if separated {
            let back = copy_out(&device, &queue, &separate, Texels::whole(side, 1));
            assert_eq!(
                back,
                words(0x3e80_0000, side * side),
                "{name}: the separate depth"
            );
            println!("FRESH-WRITE {name} separate Depth32Float and Stencil8 {side}x{side} cleared: read back the depth whole");
        }
        // A view of level 1 of a new mipmapped depth, written whole, then the texture's level 1
        // read.
        let chain = fresh(&device, depth, 2 * side, 1, true);
        let view = chain
            .new_texture_view_with_levels_slices(depth, TextureType::Type2D, 1..2, 0..1)
            .unwrap();
        let level = Texels::whole(side, 1);
        let bytes = depths(level.bytes() / 4);
        let source = buffer_with(&device, &bytes).unwrap();
        copy_in(&queue, (&source, 0, level.row()), &view, level);
        let back = copy_out(&device, &queue, &chain, Texels { level: 1, ..level });
        assert_eq!(
            back, bytes,
            "{name}: the view's level read through the texture"
        );
        println!("FRESH-WRITE {name} a view's level 1 of a {}-wide Depth32Float chain: read back through the texture", 2 * side);
        // A view of slice 1 of a new 4-sample array, written whole, then the array's slice 1
        // read.
        let mut d = TextureDescriptor::texture_2d(depth, side, side, false);
        d.texture_type = TextureType::Type2DMultisampleArray;
        d.sample_count = samples;
        d.array_length = 2;
        d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        d.resource_options = ResourceOptions::private();
        let array = device.new_texture(&d).unwrap();
        let slice = array
            .new_texture_view_with_levels_slices(depth, TextureType::Type2DMultisample, 0..1, 1..2)
            .unwrap();
        copy_in(&queue, (&quarters_source, 0, whole.row()), &slice, whole);
        let back = copy_out_slice(&device, &queue, &array, 1, whole);
        assert_eq!(
            back, quarters_bytes,
            "{name}: the view's slice read through the array"
        );
        println!("FRESH-WRITE {name} a view's slice 1 of a 4x Depth32Float array: read back through the array");
        // The separate textures render through a combined scratch (`separate-depth-stencil`),
        // and the resolve stores the cleared attachment `MultisampleResolve` with no draw, which
        // no driver is measured to store as Metal's (`dont-care-store`): each where its pass ran.
        let mut first = copied_in(&device, true);
        first.extend(forced(&[
            (Degradation::SeparateDepthStencil, separated),
            (Degradation::DontCareStore, resolved_ran),
        ]));
        let quiet = assert_made_once(&messages, &[], &first);
        assert!(
            !quiet.contains_key(&Degradation::FreshPatternZeroed),
            "{name}: a texture read where it was written announced its pattern: {quiet:?}"
        );
        // The control: the half-written texture read whole reaches its unwritten half.
        copy_out(&device, &queue, &halves, whole);
        let control = assert_once(&messages.take_notices(), patterns_of(&device));
        println!(
            "FRESH-WRITE {name} control, the top half's texture read whole: announced {control:?}"
        );
    }
    // Whether the device notes a new single-sample `Depth32Float`, so whether the resolve, the
    // separate depth and the chain's view were checked (a device of its own: the notice is
    // announced once a device).
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let queue = device.new_command_queue().unwrap();
        messages.take_notices();
        let unwritten = fresh(&device, PixelFormat::Depth32Float, 32, 1, false);
        copy_out(&device, &queue, &unwritten, Texels::whole(32, 1));
        // Noted where a copy of all-ones bytes, Metal's pattern, may store a clamped depth
        // (`degrade_notices::clamps_depth_uploads`); the read's buffer is `Shared`.
        let first = forced(&[
            (
                Degradation::FreshPatternZeroed,
                clamps_depth_uploads(&device) && !unfilled(&device),
            ),
            (
                Degradation::SharedInSystemMemory,
                shared_in_system_memory(&device),
            ),
        ]);
        let counts = assert_made_once(&messages, &[], &first);
        let noted = counts.contains_key(&Degradation::FreshPatternZeroed);
        println!("FRESH-WRITE {name} a new single-sample Depth32Float is noted: {noted}");
    }
    layer_reported_nothing_at_all(&messages);
}

/// A clear the pass renders through a scratch and does not store writes nothing of the texture:
/// separate 4-sample `Depth32Float` and `Stencil8` textures (noted on every device), their pass
/// rendered through a combined scratch, the depth cleared and stored `DontCare` (the stencil
/// cleared and stored). Metal keeps the depth texture's texels, its pattern, so the depth read
/// after it announces `fresh-pattern-zeroed`, where the tracker counted the clear as written and
/// announced nothing. The pass itself announces nothing of it.
#[test]
fn a_clear_through_a_scratch_the_pass_does_not_store_writes_nothing() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let queue = device.new_command_queue().unwrap();
        messages.take_notices();
        let (side, samples) = (32u64, 4u64);
        let whole = Texels::whole(side, samples);
        let (depth, stencil) = (
            fresh(&device, PixelFormat::Depth32Float, side, samples, false),
            fresh(&device, PixelFormat::Stencil8, side, samples, false),
        );
        let mut pass = RenderPassDescriptor::new();
        pass.depth_attachment.texture = Some(depth.clone());
        pass.depth_attachment.load_action = LoadAction::Clear;
        pass.depth_attachment.clear_depth = 0.25;
        pass.depth_attachment.store_action = StoreAction::DontCare;
        pass.stencil_attachment.texture = Some(stencil.clone());
        pass.stencil_attachment.load_action = LoadAction::Clear;
        pass.stencil_attachment.clear_stencil = 0x5a;
        pass.stencil_attachment.store_action = StoreAction::Store;
        if run_pass(&name, &queue, &pass, "separate 4x depth stored DontCare") {
            // The separate textures render through a combined scratch.
            let quiet = assert_once(
                &messages.take_notices(),
                &[Degradation::SeparateDepthStencil],
            );
            assert!(
                !quiet.contains_key(&Degradation::FreshPatternZeroed),
                "{name}: the pass alone announced the pattern: {quiet:?}"
            );
            copy_out(&device, &queue, &depth, whole);
            let first = forced(&[(
                Degradation::SharedInSystemMemory,
                shared_in_system_memory(&device),
            )]);
            let counts = assert_made_once(&messages, patterns_of(&device), &first);
            println!(
                "FRESH-WRITE {name} separate 4x Depth32Float cleared, stored DontCare, read whole: announced {counts:?}"
            );
        }
    }
    layer_reported_nothing_at_all(&messages);
}
