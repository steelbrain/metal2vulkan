//! Device tests of the ordering of textures over buffers against the other uses of their bytes
//! on the device the cfg selects, under the validation layer. A command writes the bytes through a
//! texture over a buffer, through the buffer, through another texture over them, or with a blit,
//! and the next encoder reads them another way; each pair runs in one command buffer and across
//! two. Texture buffers are the buffer's bytes; a 2D texture over a buffer is a shadow image the
//! encoders copy in and back (`runtime/shadow.rs`).
//!
//! Each chain is checked against the runtime's own run of the reading case over the bytes the
//! writer leaves, written by the CPU (`run_case`), and the test checks that the bytes a read
//! before the write would see give another output.

use super::degrade_notices::counts;
use super::{
    at_plan, buffer_with, encode_case, layer_reported_nothing_at_all, pipeline_state, public_case,
    read_buffer, run_case, texture_descriptor, validated_device, validated_device_with,
    CaseResources, Messages, PublicCase, ScratchCache, WAIT,
};
use crate::candidate::engine::ComputeOutput;
use crate::case::TextureType;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan_runtime::DeviceOptions;
use metal2vulkan_runtime::{
    BlitOption, Buffer, CommandBuffer, CommandBufferStatus, Degradation, Device, Origin,
    ResourceOptions, Size, Texture, TextureDescriptor,
};
use std::collections::BTreeMap;

/// Writes eight `rgba8_uint` texels through a texture buffer (texture 1, its output).
const TEXEL_WRITER: &str = "write-four-bytes-into-a-storage-texel-buffer";

/// Reads a four-texel `r32_uint` texture buffer (texture 0) into its 32-byte output buffer (1).
const TEXEL_READER: &str = "reads-and-writes-past-a-texture-buffer";

/// Reads four texture buffers of 256 texels each into its output buffer.
const FOUR_TEXEL_READS: &str = "half-conversion-by-texture-buffer-reads";

/// Writes one of two 4x1 `rgba32_float` 2D storage textures per thread (texture 0, its output,
/// and texture 1).
const TEXTURE_WRITER: &str = "write-through-a-storage-texture-selected-per-thread";

/// Samples three 2x2 `rgba32_float` 2D textures (texture 0 among them) into its output buffer.
const SAMPLER: &str = "sample-each-of-three-textures-through-one-dynamic-array-index";

/// Reads a 64-byte input buffer (0) into a 64-byte output buffer (1).
const BUFFER_COPIER: &str = "a-union-member-store-lands-on-the-slot-it-names";

/// Run `test` on a validated device of the lane, then fail on any validation message.
///
/// `adjust` is applied to the options of each device it makes (`at_plan` gives them their
/// queue plan).
fn on_device_with(adjust: fn(&mut DeviceOptions), test: impl FnOnce(&Device)) {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), adjust).unwrap();
        test(&device);
    }
    layer_reported_nothing_at_all(&messages);
}

/// A binding of a case: its texture or buffer at a slot.
#[derive(Clone, Copy, Debug)]
enum Slot {
    Texture(u32),
    Buffer(u32),
}

/// The literal bytes of `case` at `slot`.
fn literal(case: &PublicCase, slot: Slot) -> &[u8] {
    match slot {
        Slot::Texture(index) => &case.compute.textures[&index].bytes,
        Slot::Buffer(index) => &case.compute.buffers[&index],
    }
}

/// `case` with `bytes` (as many as it holds) as its literal bytes at `slot`.
fn with_literal(mut case: PublicCase, slot: Slot, bytes: &[u8]) -> PublicCase {
    let held = match slot {
        Slot::Texture(index) => &mut case.compute.textures.get_mut(&index).unwrap().bytes,
        Slot::Buffer(index) => case.compute.buffers.get_mut(&index).unwrap(),
    };
    let n = held.len();
    held.copy_from_slice(&bytes[..n]);
    case
}

/// One case's run in a chain, its binding at `slot` over `under`: its buffers and resources.
struct Bound {
    buffers: BTreeMap<u32, Buffer>,
    resources: CaseResources,
}

/// `case`'s buffers and resources on `device`, each binding `over` names over its buffer: the
/// buffer itself, or a texture over it from its first byte, rows `pad` bytes longer than the
/// packed row (a texture buffer's row the literal's length).
fn bound_over(device: &Device, case: &PublicCase, over: &[(Slot, &Buffer)], pad: u64) -> Bound {
    let mut buffers: BTreeMap<u32, Buffer> = case
        .compute
        .buffers
        .iter()
        .map(|(&index, bytes)| (index, buffer_with(device, bytes).unwrap()))
        .collect();
    let mut resources = CaseResources::new(device, case).unwrap();
    for &(slot, under) in over {
        match slot {
            Slot::Buffer(index) => {
                buffers.insert(index, under.clone());
            }
            Slot::Texture(index) => {
                let literal = &case.compute.textures[&index];
                let descriptor = texture_descriptor(literal, case.texture_usage[&index]).unwrap();
                let bytes_per_row = match literal.texture_type {
                    TextureType::Buffer => literal.bytes.len() as u64,
                    TextureType::D2 => {
                        descriptor.width * literal.format.bytes_per_pixel() as u64 + pad
                    }
                    other => panic!("{}: a {other:?} texture over a buffer", case.case.name),
                };
                let texture = under.new_texture(&descriptor, 0, bytes_per_row).unwrap();
                resources.textures.insert(index, texture);
            }
        }
    }
    Bound { buffers, resources }
}

/// Encode `case`'s dispatch over `bound` as one compute encoder of `command_buffer`.
fn encode_dispatch(command_buffer: &CommandBuffer, case: &PublicCase, bound: &Bound) {
    let pipeline = pipeline_state(command_buffer.device(), case).unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encode_case(
        &mut encoder,
        &pipeline,
        &case.compute,
        &bound.buffers,
        &bound.resources,
    )
    .unwrap();
    encoder.end_encoding().unwrap();
}

/// A blit of `size` bytes from `source` to `destination` as one encoder of `command_buffer`.
fn encode_copy(command_buffer: &CommandBuffer, source: &Buffer, destination: &Buffer, size: u64) {
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_buffer(source, 0, destination, 0, size)
        .unwrap();
    blit.end_encoding().unwrap();
}

/// Encode `first` and then `second`, each one encoder, in one command buffer or (`split`) in two
/// committed in order; wait for them, and check each completed.
fn two_encoders(
    device: &Device,
    split: bool,
    first: impl FnOnce(&CommandBuffer),
    second: impl FnOnce(&CommandBuffer),
) {
    let queue = device.new_command_queue().unwrap();
    let mut command_buffers = vec![queue.command_buffer().unwrap()];
    first(&command_buffers[0]);
    if split {
        command_buffers[0].commit().unwrap();
        command_buffers.push(queue.command_buffer().unwrap());
    }
    second(command_buffers.last().unwrap());
    command_buffers.last().unwrap().commit().unwrap();
    for command_buffer in &command_buffers {
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    }
}

/// The output `case` left in `bound` (a buffer output).
fn buffer_output(case: &PublicCase, bound: &Bound) -> Vec<u8> {
    match case.compute.output {
        ComputeOutput::Buffer {
            binding,
            offset,
            length,
        } => read_buffer(&bound.buffers[&binding], offset, length).unwrap(),
        other => panic!("{}: output {other:?}", case.case.name),
    }
}

/// The bytes the writer case `writer` leaves at `slot` (its output, the whole binding): its run.
fn written(device: &Device, writer: &PublicCase, slot: Slot) -> Vec<u8> {
    let output = run_case(device, writer).unwrap().output;
    assert_eq!(
        output.len(),
        literal(writer, slot).len(),
        "{}: its output is not its whole binding {slot:?}",
        writer.case.name
    );
    output
}

/// `reader` reading `slot` over `before` and over `after`, as its literal bytes (written by the
/// CPU): its output over `after`, which differs from its output over `before`.
fn read_over(
    device: &Device,
    reader: &PublicCase,
    slot: Slot,
    before: &[u8],
    after: &[u8],
) -> (PublicCase, Vec<u8>) {
    let name = &reader.case.name;
    let stale = with_literal(public_case(name).unwrap(), slot, before);
    let stale = run_case(device, &stale).unwrap().output;
    let fresh = with_literal(public_case(name).unwrap(), slot, after);
    let expected = run_case(device, &fresh).unwrap().output;
    assert_ne!(expected, stale, "{name}: the write leaves its output alone");
    (fresh, expected)
}

/// The kernel `writer` writes the bytes of a buffer at its `wrote` and the next encoder's kernel
/// `reader` reads them at its `read`: it reads what the writer wrote, in one command buffer and
/// across two.
fn kernel_after_kernel(
    device: &Device,
    writer: &str,
    wrote: Slot,
    reader: &str,
    read: Slot,
    pad: u64,
) {
    let writer = public_case(writer).unwrap();
    let before = literal(&writer, wrote).to_vec();
    let after = written(device, &writer, wrote);
    let (reader, expected) =
        read_over(device, &public_case(reader).unwrap(), read, &before, &after);
    for split in [false, true] {
        let under = buffer_with(device, &before).unwrap();
        let first = bound_over(device, &writer, &[(wrote, &under)], pad);
        let second = bound_over(device, &reader, &[(read, &under)], pad);
        two_encoders(
            device,
            split,
            |command_buffer| encode_dispatch(command_buffer, &writer, &first),
            |command_buffer| encode_dispatch(command_buffer, &reader, &second),
        );
        assert_eq!(
            buffer_output(&reader, &second),
            expected,
            "{} {wrote:?} then {} {read:?}, split {split}",
            writer.case.name,
            reader.case.name
        );
    }
}

/// The kernel `writer` writes the bytes of a buffer at its `wrote` and the next encoder, a blit,
/// copies the buffer: the copy holds what the kernel wrote, in one command buffer and across two.
fn blit_after_kernel(device: &Device, writer: &str, wrote: Slot, pad: u64) {
    let writer = public_case(writer).unwrap();
    let before = literal(&writer, wrote).to_vec();
    let after = written(device, &writer, wrote);
    assert_ne!(
        after, before,
        "{}: the write leaves its bytes",
        writer.case.name
    );
    for split in [false, true] {
        let under = buffer_with(device, &before).unwrap();
        let bound = bound_over(device, &writer, &[(wrote, &under)], pad);
        let copy = device
            .new_buffer(after.len() as u64, ResourceOptions::shared())
            .unwrap();
        two_encoders(
            device,
            split,
            |command_buffer| encode_dispatch(command_buffer, &writer, &bound),
            |command_buffer| encode_copy(command_buffer, &under, &copy, after.len() as u64),
        );
        assert_eq!(
            read_buffer(&copy, 0, after.len() as u64).unwrap(),
            after,
            "{} {wrote:?} then a blit, split {split}",
            writer.case.name
        );
    }
}

/// A blit writes the buffers under `reader`'s `reads` and the next encoder's kernel reads them:
/// it reads what the blit wrote, in one command buffer and across two.
fn kernel_after_blit(device: &Device, reader: &str, reads: &[Slot]) {
    let reader = public_case(reader).unwrap();
    let expected = run_case(device, &reader).unwrap().output;
    // What a read before the blit sees: the new buffers' zeros.
    let mut zeroed = public_case(&reader.case.name).unwrap();
    for &slot in reads {
        let zeros = vec![0; literal(&reader, slot).len()];
        zeroed = with_literal(zeroed, slot, &zeros);
    }
    let stale = run_case(device, &zeroed).unwrap().output;
    assert_ne!(
        stale, expected,
        "{}: zeros give its output",
        reader.case.name
    );
    for split in [false, true] {
        // Each binding over a new buffer, and a buffer of its bytes to copy there.
        let copies: Vec<(Slot, Buffer, Buffer)> = reads
            .iter()
            .map(|&slot| {
                let bytes = literal(&reader, slot);
                let under = device
                    .new_buffer(bytes.len() as u64, ResourceOptions::shared())
                    .unwrap();
                (slot, buffer_with(device, bytes).unwrap(), under)
            })
            .collect();
        let over: Vec<(Slot, &Buffer)> = copies
            .iter()
            .map(|(slot, _, under)| (*slot, under))
            .collect();
        let bound = bound_over(device, &reader, &over, 0);
        two_encoders(
            device,
            split,
            |command_buffer| {
                for (_, source, under) in &copies {
                    encode_copy(command_buffer, source, under, source.length());
                }
            },
            |command_buffer| encode_dispatch(command_buffer, &reader, &bound),
        );
        assert_eq!(
            buffer_output(&reader, &bound),
            expected,
            "a blit then {}, split {split}",
            reader.case.name
        );
    }
}

/// Texture buffers: a kernel's writes through one are in its buffer for a blit and for another
/// texture buffer over the bytes, a kernel's writes through the buffer are in the texture buffer,
/// and a blit's writes are in four texture buffers a kernel reads.
#[test]
fn texture_buffers_are_ordered_with_the_other_uses_of_their_bytes() {
    on_device_with(at_plan, |device| {
        blit_after_kernel(device, TEXEL_WRITER, Slot::Texture(1), 0);
        kernel_after_kernel(
            device,
            TEXEL_WRITER,
            Slot::Texture(1),
            TEXEL_READER,
            Slot::Texture(0),
            0,
        );
        kernel_after_kernel(
            device,
            TEXEL_READER,
            Slot::Buffer(1),
            TEXEL_READER,
            Slot::Texture(0),
            0,
        );
        kernel_after_blit(device, FOUR_TEXEL_READS, &[0, 1, 2, 3].map(Slot::Texture));
    });
}

/// Rows of [`TEXTURE_WRITER`]'s one-row textures longer than any device's linear row of them:
/// such a texture is a shadow image on every device, its bytes the packed row's.
const SHADOWED: u64 = 256;

/// 2D textures over buffers: a kernel's writes through one are in its buffer for a blit and for
/// a kernel reading the buffer, and a kernel's or a blit's writes of the buffer are in the
/// texture a kernel samples. Packed rows alias the buffer where the device lays its linear image
/// so; one-row textures with longer rows are shadow images everywhere.
#[test]
fn textures_over_buffers_are_ordered_with_the_other_uses_of_their_bytes() {
    on_device_with(at_plan, |device| {
        for pad in [0, SHADOWED] {
            blit_after_kernel(device, TEXTURE_WRITER, Slot::Texture(0), pad);
            kernel_after_kernel(
                device,
                TEXTURE_WRITER,
                Slot::Texture(0),
                BUFFER_COPIER,
                Slot::Buffer(0),
                pad,
            );
        }
        kernel_after_kernel(
            device,
            BUFFER_COPIER,
            Slot::Buffer(1),
            SAMPLER,
            Slot::Texture(0),
            0,
        );
        kernel_after_blit(device, SAMPLER, &[Slot::Texture(0)]);
    });
}

/// [`TEXTURE_WRITER`]'s dispatch with `textures` bound, in a command buffer of its own that
/// completes: the dispatch's refusal, if any.
fn write_through(
    device: &Device,
    case: &PublicCase,
    textures: BTreeMap<u32, Texture>,
) -> Option<String> {
    let pipeline = pipeline_state(device, case).unwrap();
    let resources = CaseResources {
        textures,
        samplers: BTreeMap::new(),
        ..CaseResources::default()
    };
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    let result = encode_case(
        &mut encoder,
        &pipeline,
        &case.compute,
        &BTreeMap::new(),
        &resources,
    );
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    result.err().map(String::from)
}

/// A 2D texture over a buffer whose rows the device's linear image lays as Metal does aliases
/// the buffer, and any other is a shadow image, which the device announces
/// (`shadowed-texture`): either way its writes land in the buffer's bytes as Metal's do. Which
/// layouts alias is the device's; the forms each layout took are printed, each layout on a device
/// of its own so that the announcement is its own. Longer rows than any device's linear row are
/// shadow images everywhere.
#[test]
fn textures_over_buffers_alias_where_the_device_lays_them_as_metal_does() {
    let case = public_case(TEXTURE_WRITER).unwrap();
    let usage = case.texture_usage[&0] | case.texture_usage[&1];
    let literal = &case.compute.textures[&0];
    let mut forms = Vec::new();
    let mut name = String::new();
    // 16 texels make 256-byte rows: a multiple of every measured device's linear row pitch.
    for (width, pad) in [(16, 0), (4, 0), (16, SHADOWED)] {
        let row = width * 16 + pad;
        let messages = Messages::default();
        let cache = ScratchCache::new().unwrap();
        {
            let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
            name = device.name().to_owned();
            // What the device announced as it was made.
            messages.take_notices();
            let mut d = texture_descriptor(literal, usage).unwrap();
            d.width = width;
            // Texture 0's writes land in the buffer's bytes as Metal's land in its texture.
            let under = buffer_with(&device, &vec![0xAB; row as usize]).unwrap();
            let mut textures = CaseResources::new(&device, &case).unwrap().textures;
            textures.insert(0, under.new_texture(&d, 0, row).unwrap());
            assert_eq!(write_through(&device, &case, textures), None);
            assert!(
                read_buffer(&under, 0, case.metal_output.len() as u64).unwrap()
                    == case.metal_output,
                "{width} texels, rows of {row} bytes: the buffer's bytes are not Metal's"
            );
        }
        let shadowed = counts(&messages.take_notices()).contains_key(&Degradation::ShadowedTexture);
        forms.push((width, row, if shadowed { "shadowed" } else { "aliased" }));
        layer_reported_nothing_at_all(&messages);
    }
    eprintln!("textures over buffers on {name}: {forms:?}");
    assert_eq!(
        forms[2].2, "shadowed",
        "rows longer than any linear row alias"
    );
}

/// `name`'s texture 0 made with `usage` alone over a buffer, `width` texels wide in rows `pad`
/// bytes longer than packed (its literal's texels at the start of each row), and the case run on
/// a validated device of its own: whether the texture or its binding was shadowed
/// (`shadowed-texture` or `undeclared-usage-shadow` announced). The run gives the case's Metal
/// output: the case's buffer output, or for a case whose output is texture 0, the buffer's bytes
/// under it. Fails on a refusal (none is `UndeclaredTextureUsage`), on another output, and on any
/// validation message.
fn with_undeclared_usage(name: &str, usage: TextureUsage, width: u64, pad: u64) -> bool {
    let case = public_case(name).unwrap();
    let literal = &case.compute.textures[&0];
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        messages.take_notices();
        let mut d = texture_descriptor(literal, usage).unwrap();
        let packed = d.width * literal.format.bytes_per_pixel() as u64;
        d.width = width;
        let row = width * literal.format.bytes_per_pixel() as u64 + pad;
        let mut bytes = vec![0xAB; (row * d.height) as usize];
        for (y, texels) in literal.bytes.chunks(packed as usize).enumerate() {
            let at = y * row as usize;
            bytes[at..at + texels.len()].copy_from_slice(texels);
        }
        let under = buffer_with(&device, &bytes).unwrap();
        let texture = under.new_texture(&d, 0, row).unwrap();
        assert_eq!(texture.usage(), usage, "{name}");
        let mut resources = CaseResources::new(&device, &case).unwrap();
        resources.textures.insert(0, texture);
        let bound = Bound {
            buffers: case
                .compute
                .buffers
                .iter()
                .map(|(&index, bytes)| (index, buffer_with(&device, bytes).unwrap()))
                .collect(),
            resources,
        };
        let pipeline = pipeline_state(&device, &case).unwrap();
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        let result = encode_case(
            &mut encoder,
            &pipeline,
            &case.compute,
            &bound.buffers,
            &bound.resources,
        );
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        match result.err().map(String::from) {
            None => {
                let output = match case.compute.output {
                    ComputeOutput::Texture { binding: 0, .. } => {
                        read_buffer(&under, 0, case.metal_output.len() as u64).unwrap()
                    }
                    _ => buffer_output(&case, &bound),
                };
                assert!(
                    output == case.metal_output,
                    "{name}, rows of {row} bytes: the output is not Metal's"
                );
            }
            Some(refusal) => panic!("{name}, rows of {row} bytes: {refusal}"),
        }
    }
    let notices = counts(&messages.take_notices());
    let shadowed = notices.contains_key(&Degradation::ShadowedTexture)
        || notices.contains_key(&Degradation::UndeclaredUsageShadow);
    layer_reported_nothing_at_all(&messages);
    shadowed
}

/// A 2D texture over a buffer used beyond its declared usage runs as plain Metal runs it (an M4
/// under macOS 27.0.1: a `ShaderRead` texture over a buffer written through `access::write` and
/// `access::read_write`, and a `ShaderWrite` one read through `access::read`, `access::read_write`
/// and `sample()`, each as the texture with both usages; Metal's validation stops each; the
/// reims-vgpu consumer's plane views of its two-plane XR10 textures, made `ShaderRead` and written
/// by its paging kernel): [`TEXTURE_WRITER`]'s texture made `ShaderRead` alone, and [`SAMPLER`]'s
/// texture 0 made `ShaderWrite` alone, at a layout the device may alias and at one it shadows. An
/// aliasing texture takes the usage with its linear image (a linear image is never compressed and
/// its memory is the buffer's); a shadow takes it where that costs the shadow nothing, and
/// elsewhere the binding takes a second shadow of the bytes with the usage
/// (`undeclared-usage-shadow`). Each runs as Metal's; the form each layout
/// took is printed.
#[test]
fn textures_over_buffers_run_beyond_their_declared_usage() {
    let mut forms = Vec::new();
    for (name, usage, width) in [
        (TEXTURE_WRITER, TextureUsage::SHADER_READ, 16),
        (SAMPLER, TextureUsage::SHADER_WRITE, 2),
    ] {
        for pad in [0, SHADOWED] {
            let shadowed = with_undeclared_usage(name, usage, width, pad);
            forms.push((name, pad, if shadowed { "shadowed" } else { "aliased" }));
        }
    }
    eprintln!("textures over buffers beyond their declared usage: {forms:?}");
}

/// The kind a slot binding beyond a texture's declared usage takes where its image lacks the usage
/// ([`textures_run_beyond_their_declared_usage`], and for a texture over a buffer
/// [`textures_over_buffers_run_beyond_their_declared_usage`]).
pub(super) const USAGE_SHADOWS: [Degradation; 1] = [Degradation::UndeclaredUsageShadow];

/// The side of [`beyond_usage`]'s textures: large enough that a device compresses an
/// `RGBA8Unorm` image of it, so the storage or sampled usage its descriptor lacks costs the image
/// something and the binding takes a shadow (on a device that compresses nothing it is free).
const BEYOND_SIDE: u64 = 256;

/// `name`'s case on a validated device of its own with texture 0 replaced by a
/// [`BEYOND_SIDE`]-square `RGBA8Unorm` `Private` texture of `usage`, filled by a blit with a
/// pattern: the texture's texels after the run, read by a blit, and the case's buffer output
/// (none for a case whose output is texture 0), and whether a binding took a shadow
/// (`undeclared-usage-shadow` announced). Fails on a refusal and on any validation message.
fn beyond_usage(name: &str, usage: TextureUsage) -> (Vec<u8>, Vec<u8>, bool) {
    let case = public_case(name).unwrap();
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let side = BEYOND_SIDE;
    let bytes = side * side * 4;
    let ran = {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        messages.take_notices();
        let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, side, side, false);
        d.usage = usage;
        d.resource_options = ResourceOptions::private();
        let texture = device.new_texture(&d).unwrap();
        let pattern: Vec<u8> = (0..bytes).map(|i| (i % 251) as u8).collect();
        let source = buffer_with(&device, &pattern).unwrap();
        let out = buffer_with(&device, &vec![0; bytes as usize]).unwrap();
        let queue = device.new_command_queue().unwrap();
        let fill = queue.command_buffer().unwrap();
        let mut blit = fill.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_texture(
            &source,
            0,
            side * 4,
            bytes,
            Size::new(side, side, 1),
            &texture,
            0,
            0,
            Origin::new(0, 0, 0),
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        fill.commit().unwrap();
        let mut resources = CaseResources::new(&device, &case).unwrap();
        resources.textures.insert(0, texture.clone());
        let bound = Bound {
            buffers: case
                .compute
                .buffers
                .iter()
                .map(|(&index, bytes)| (index, buffer_with(&device, bytes).unwrap()))
                .collect(),
            resources,
        };
        let pipeline = pipeline_state(&device, &case).unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        let result = encode_case(
            &mut encoder,
            &pipeline,
            &case.compute,
            &bound.buffers,
            &bound.resources,
        );
        encoder.end_encoding().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_texture_to_buffer(
            &texture,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(side, side, 1),
            &out,
            0,
            side * 4,
            bytes,
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        if let Err(refusal) = result {
            panic!("{name}, usage {usage:?}: {}", String::from(refusal));
        }
        let output = match case.compute.output {
            ComputeOutput::Texture { binding: 0, .. } => Vec::new(),
            _ => buffer_output(&case, &bound),
        };
        (read_buffer(&out, 0, bytes).unwrap(), output)
    };
    let shadowed =
        counts(&messages.take_notices()).contains_key(&Degradation::UndeclaredUsageShadow);
    layer_reported_nothing_at_all(&messages);
    (ran.0, ran.1, shadowed)
}

/// A texture of its own used beyond its declared usage runs as plain Metal runs it (an M4 under
/// macOS 27.0.1: a write to a `ShaderRead` texture stores and a read of a `ShaderWrite` one reads,
/// each as the texture with both usages; Metal's validation stops each): [`TEXTURE_WRITER`]'s
/// texture 0 made `ShaderRead` alone, and [`SAMPLER`]'s made `ShaderWrite` alone, and each made of
/// Apple's private 0x8 alone (written and sampled on the M3 Max under macOS 27.0.1), each a
/// [`BEYOND_SIDE`]-square `RGBA8Unorm` texture ([`beyond_usage`]), give the texels and the output
/// the same texture with both usages gives. Where the image takes the usage free it has it;
/// elsewhere the binding takes a shadow, announced once (`undeclared-usage-shadow`). The form each
/// case took is printed.
#[test]
fn textures_run_beyond_their_declared_usage() {
    let both = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    let mut forms = Vec::new();
    // Apple's private 0x8 alone, whose Metal texture answers no usage: plain Metal writes and
    // samples it too (the M3 Max under macOS 27.0.1).
    let dropped = TextureUsage::try_from(0x8u32).unwrap();
    for (name, usage) in [
        (TEXTURE_WRITER, TextureUsage::SHADER_READ),
        (SAMPLER, TextureUsage::SHADER_WRITE),
        (TEXTURE_WRITER, dropped),
        (SAMPLER, dropped),
    ] {
        let (texels, output, shadowed) = beyond_usage(name, usage);
        let (control_texels, control_output, control_shadowed) = beyond_usage(name, both);
        assert!(
            !control_shadowed,
            "{name}: a texture with both usages took a shadow"
        );
        assert!(
            texels == control_texels,
            "{name}: the texels differ from the texture's with both usages"
        );
        assert_eq!(
            output, control_output,
            "{name}: the output differs from the texture's with both usages"
        );
        forms.push((name, if shadowed { "shadowed" } else { "free" }));
    }
    eprintln!("textures beyond their declared usage: {forms:?}");
}

/// `bound` with its texture at `index` replaced by a view of it as `pixel_format`.
fn through_view(mut bound: Bound, index: u32, pixel_format: PixelFormat) -> Bound {
    let view = bound.resources.textures[&index]
        .new_texture_view(pixel_format)
        .unwrap();
    bound.resources.textures.insert(index, view);
    bound
}

/// A view of a texture over a buffer shares its texture's storage: a kernel's writes through a view
/// of a 2D texture over a buffer (aliasing or shadowed) and of a texture buffer land in the
/// buffer's bytes, and a kernel reads a texture buffer's bytes through a view of another format of
/// the same size, as Metal's views do (a Metal probe on the M3 Max and an M4).
#[test]
fn views_of_textures_over_buffers_share_their_storage() {
    on_device_with(at_plan, |device| {
        for (name, slot, pad) in [
            (TEXTURE_WRITER, 0, 0),
            (TEXTURE_WRITER, 0, SHADOWED),
            (TEXEL_WRITER, 1, 0),
        ] {
            let writer = public_case(name).unwrap();
            let before = literal(&writer, Slot::Texture(slot)).to_vec();
            let after = written(device, &writer, Slot::Texture(slot));
            let under = buffer_with(device, &before).unwrap();
            let bound = bound_over(device, &writer, &[(Slot::Texture(slot), &under)], pad);
            let pixel_format = bound.resources.textures[&slot].pixel_format();
            let bound = through_view(bound, slot, pixel_format);
            two_encoders(
                device,
                false,
                |command_buffer| encode_dispatch(command_buffer, &writer, &bound),
                |_| {},
            );
            assert_eq!(
                read_buffer(&under, 0, after.len() as u64).unwrap(),
                after,
                "{name} through a view, rows padded by {pad}"
            );
        }
        // `TEXEL_READER`'s `R32Uint` texture buffer as a view of an `RGBA8Uint` one over the
        // same bytes.
        let reader = public_case(TEXEL_READER).unwrap();
        let expected = run_case(device, &reader).unwrap().output;
        let bytes = literal(&reader, Slot::Texture(0));
        let under = buffer_with(device, bytes).unwrap();
        let mut bound = bound_over(device, &reader, &[], 0);
        let source = &reader.compute.textures[&0];
        let usage = reader.texture_usage[&0] | TextureUsage::PIXEL_FORMAT_VIEW;
        let mut descriptor = texture_descriptor(source, usage).unwrap();
        descriptor.pixel_format = PixelFormat::RGBA8Uint;
        let texture = under
            .new_texture(&descriptor, 0, bytes.len() as u64)
            .unwrap();
        bound.resources.textures.insert(0, texture);
        let bound = through_view(bound, 0, PixelFormat::R32Uint);
        two_encoders(
            device,
            false,
            |command_buffer| encode_dispatch(command_buffer, &reader, &bound),
            |_| {},
        );
        assert_eq!(buffer_output(&reader, &bound), expected, "a format view");
    });
}
