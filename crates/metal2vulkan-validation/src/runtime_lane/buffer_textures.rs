//! Device tests of textures over buffers on the device the cfg selects, under the validation layer:
//! texture buffers, 2D textures and arrays made over a buffer (texture buffers by `new_texture`
//! too), or refused typed as Metal refuses them; CPU access through them as access to their
//! buffer's bytes; and unbound texture-buffer slots reading the device's neutral texel buffers;
//! shareable texture buffers, whose handles open texture buffers over the same texels. Their
//! ordering against other uses of the bytes is `buffer_texture_order.rs`'s.

use super::pool_slots::holding_the_space;
use super::{
    at_plan, bound_buffers, buffer_with, encode_case, layer_reported_nothing_at_all,
    pipeline_state, public_case, read_buffer, run_case, texture_descriptor, validated_device_with,
    CaseResources, Messages, ScratchCache, WAIT,
};
use crate::candidate::engine::ComputeOutput;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::StatusReport;
use metal2vulkan::error::Feature;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan_runtime::DeviceOptions;
use metal2vulkan_runtime::{
    BlitOption, BufferTextureProblem, CommandBufferStatus, CopyProblem, CpuCacheMode, Device,
    Error, HazardTrackingMode, IdPlacement, Invalid, Origin, Problem, Region, ResourceOptions,
    Size, StorageMode, TextureDescriptor, ViewProblem,
};

/// A case that writes through a texture buffer, and a case that reads four.
const TEXEL_CASES: [&str; 2] = [
    "write-four-bytes-into-a-storage-texel-buffer",
    "half-conversion-by-texture-buffer-reads",
];

/// Run `test` on a validated device of the lane, then fail on any validation message.
fn on_device(test: impl FnOnce(&Device)) {
    on_device_with(|_| {}, test)
}

/// [`on_device`], `adjust` applied to the options of each device it makes (`at_plan` gives them
/// their queue plan).
fn on_device_with(adjust: fn(&mut DeviceOptions), test: impl FnOnce(&Device)) {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), adjust).unwrap();
        test(&device);
    }
    layer_reported_nothing_at_all(&messages);
}

fn invalid(call: &'static str, problem: BufferTextureProblem) -> Error {
    Error::Invalid(Invalid::new(call, Problem::BufferTexture(problem)))
}

/// `n` bytes of a pattern seeded by `seed`.
fn pattern(n: usize, seed: u8) -> Vec<u8> {
    (0..n)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
}

/// A texture buffer of `pf` and `width` texels in `Shared` storage, read and written by shaders.
fn texture_buffer(pf: PixelFormat, width: u64) -> TextureDescriptor {
    TextureDescriptor::texture_buffer(
        pf,
        width,
        ResourceOptions::shared(),
        TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
    )
}

/// Texture buffers, 2D textures and arrays are made over a buffer with the buffer's options,
/// whatever the descriptor names, and texture buffers by `new_texture` over a new buffer; what Metal stops is refused `Invalid`
/// with its rule; a texture buffer or an array whose texels run past its buffer's length is
/// made.
#[test]
fn textures_over_buffers_are_made_or_refused_as_metal_makes_them() {
    on_device(|device| {
        const CALL: &str = "new_texture";
        let buffer = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
        let d = texture_buffer(PixelFormat::R32Uint, 64);
        let texture = buffer.new_texture(&d, 256, 256).unwrap();
        assert_eq!(texture.texture_type(), TextureType::TextureBuffer);
        assert_eq!((texture.width(), texture.height()), (64, 1));
        assert_eq!(texture.storage_mode(), StorageMode::Shared);
        assert_eq!(texture.resource_options(), buffer.resource_options());
        // A texture buffer's row is any length that covers its texels.
        buffer.new_texture(&d, 256, 300).unwrap();
        let refused = |d: &TextureDescriptor, offset, bytes_per_row| {
            buffer.new_texture(d, offset, bytes_per_row).unwrap_err()
        };
        // A texture buffer at an offset off 16 bytes, which Metal's validation alone stops, is
        // made (`a_texture_buffer_off_16_bytes_reads_and_writes_from_below`).
        buffer.new_texture(&d, 8, 256).unwrap();
        assert_eq!(
            refused(&d, 0, 255),
            invalid(
                CALL,
                BufferTextureProblem::RowPitch {
                    bytes_per_row: 255,
                    row_bytes: 256
                }
            )
        );
        // A texture buffer whose row ends past the buffer's length, which Metal's validation
        // counts and stops: plain Metal makes it over bytes past the length (`past.m`, an M3 Max
        // under macOS 27.0.1), and so does the runtime, those texels in the texture's tail
        // (`tests::buffer_tail_device`).
        let past = buffer.new_texture(&d, 3968, 256).unwrap();
        assert_eq!(
            (past.texture_type(), past.width()),
            (TextureType::TextureBuffer, 64)
        );
        // A plain run takes the buffer's options whatever the descriptor names.
        let mut private = d;
        private.resource_options = ResourceOptions::private();
        let taken = buffer.new_texture(&private, 0, 256).unwrap();
        assert_eq!(taken.resource_options(), buffer.resource_options());
        let mut compressed = TextureDescriptor::texture_2d(PixelFormat::BC1RGBA, 8, 8, false);
        compressed.resource_options = ResourceOptions::shared();
        assert_eq!(
            refused(&compressed, 0, 256),
            invalid(
                CALL,
                BufferTextureProblem::NotLinear {
                    pixel_format: PixelFormat::BC1RGBA
                }
            )
        );
        // Plain Metal makes a 4:2:2 texture over a buffer; the runtime holds none over a buffer.
        let mut chroma = TextureDescriptor::texture_2d(PixelFormat::GBGR422, 4, 4, false);
        chroma.resource_options = ResourceOptions::shared();
        assert!(
            matches!(
                refused(&chroma, 0, 16),
                Error::Unsupported(u) if u.feature == Feature::Format
            ),
            "a 4:2:2 texture over a buffer"
        );
        let mut volume = TextureDescriptor::texture_2d(PixelFormat::R32Uint, 4, 4, false);
        volume.texture_type = TextureType::Type3D;
        volume.depth = 2;
        volume.resource_options = ResourceOptions::shared();
        assert_eq!(
            refused(&volume, 0, 16),
            invalid(
                CALL,
                BufferTextureProblem::TextureType {
                    texture_type: TextureType::Type3D
                }
            )
        );
        // A 2D texture, and through the SPI an array; an array only through the SPI.
        let mut flat = TextureDescriptor::texture_2d(PixelFormat::R32Uint, 16, 4, false);
        flat.resource_options = ResourceOptions::shared();
        let made = buffer.new_texture(&flat, 0, 64).unwrap();
        assert_eq!(
            (made.texture_type(), made.width(), made.height()),
            (TextureType::Type2D, 16, 4)
        );
        assert_eq!(
            refused(&flat, 8, 64),
            invalid(
                CALL,
                BufferTextureProblem::OffsetAlignment {
                    offset: 8,
                    alignment: 16
                }
            )
        );
        // A memoryless 2D descriptor too: its texels are the buffer's bytes, which the CPU
        // reads and writes through it.
        let mut memoryless = flat;
        memoryless.resource_options = ResourceOptions::new(
            StorageMode::Memoryless,
            CpuCacheMode::DefaultCache,
            HazardTrackingMode::Default,
        );
        let taken = buffer.new_texture(&memoryless, 1024, 64).unwrap();
        assert_eq!(taken.storage_mode(), StorageMode::Shared);
        assert_eq!(taken.resource_options(), buffer.resource_options());
        let written = pattern(256, 7);
        taken
            .replace_region(Region::new_2d(0, 0, 16, 4), 0, 0, &written, 64, 0)
            .unwrap();
        let mut got = vec![0; 256];
        taken
            .get_bytes(&mut got, 64, 0, Region::new_2d(0, 0, 16, 4), 0, 0)
            .unwrap();
        assert_eq!(got, written);
        assert_eq!(read_buffer(&buffer, 1024, 256).unwrap(), written);
        let mut array = flat;
        array.texture_type = TextureType::Type2DArray;
        array.array_length = 2;
        let made = buffer.new_linear_texture(&array, 0, 64, 256).unwrap();
        assert_eq!(
            (made.texture_type(), made.array_length()),
            (TextureType::Type2DArray, 2)
        );
        // The SPI ignores the bytes-per-image, a multiple of 128 or not: an array made at 80 lays
        // its second slice where the one made at 256 does.
        let unaligned = buffer.new_linear_texture(&array, 0, 64, 80).unwrap();
        let second = pattern(256, 41);
        made.replace_region(Region::new_2d(0, 0, 16, 4), 0, 1, &second, 64, 0)
            .unwrap();
        let mut got = vec![0; 256];
        unaligned
            .get_bytes(&mut got, 64, 0, Region::new_2d(0, 0, 16, 4), 0, 1)
            .unwrap();
        assert_eq!(got, second);
        assert_eq!(read_buffer(&buffer, 256, 256).unwrap(), second);
        assert_eq!(
            refused(&array, 0, 64),
            invalid(
                CALL,
                BufferTextureProblem::TextureType {
                    texture_type: TextureType::Type2DArray
                }
            )
        );
        // A view of one slice of an array over a buffer shares its storage; Metal stops a view
        // of more than one.
        assert_eq!(
            made.new_texture_view(PixelFormat::R32Uint).unwrap_err(),
            Error::Invalid(Invalid::new(
                "new_texture_view",
                Problem::TextureView(ViewProblem::BufferSlices)
            ))
        );
        let slice = made
            .new_texture_view_with_levels_slices(
                PixelFormat::R32Uint,
                TextureType::Type2D,
                0..1,
                1..2,
            )
            .unwrap();
        assert_eq!(
            (slice.texture_type(), slice.parent_relative_slice()),
            (TextureType::Type2D, 1)
        );
        // An array whose second slice starts past the buffer's length, inside Metal's
        // allocation: Metal makes it, and so does the runtime, that slice in the texture's tail
        // (`tests::buffer_tail_device`).
        let short = device.new_buffer(128, ResourceOptions::shared()).unwrap();
        let mut rows = TextureDescriptor::texture_2d(PixelFormat::R32Uint, 4, 3, false);
        rows.texture_type = TextureType::Type2DArray;
        rows.array_length = 2;
        rows.resource_options = ResourceOptions::shared();
        let past = short.new_linear_texture(&rows, 0, 16, 0).unwrap();
        assert_eq!(
            (past.texture_type(), past.array_length()),
            (TextureType::Type2DArray, 2)
        );
        // `new_texture` of a texture buffer: over a new buffer of its options.
        let made = device.new_texture(&d).unwrap();
        assert_eq!(made.texture_type(), TextureType::TextureBuffer);
        assert_eq!(made.width(), 64);
        assert_eq!(made.storage_mode(), StorageMode::Shared);
    });
}

/// A texture buffer's CPU access is access to its buffer's bytes: `get_bytes` reads them where
/// the texture lies, `replace_region` writes them there and leaves the rest; a new texture
/// buffer reads zero; a private one is refused as any private texture is.
#[test]
fn a_texture_buffers_cpu_access_is_its_buffers_bytes() {
    on_device(|device| {
        let bytes = pattern(4096, 3);
        let buffer = buffer_with(device, &bytes).unwrap();
        let d = texture_buffer(PixelFormat::RGBA8Uint, 16);
        let texture = buffer.new_texture(&d, 512, 64).unwrap();
        let mut got = vec![0u8; 64];
        texture
            .get_bytes(&mut got, 64, 0, Region::new_1d(0, 16), 0, 0)
            .unwrap();
        assert_eq!(got, bytes[512..576]);
        let mut got = vec![0u8; 16];
        texture
            .get_bytes(&mut got, 16, 0, Region::new_1d(4, 4), 0, 0)
            .unwrap();
        assert_eq!(got, bytes[528..544]);
        let written = pattern(16, 99);
        texture
            .replace_region(Region::new_1d(2, 4), 0, 0, &written, 16, 0)
            .unwrap();
        let mut expected = bytes.clone();
        expected[520..536].copy_from_slice(&written);
        assert_eq!(read_buffer(&buffer, 0, 4096).unwrap(), expected);
        assert!(matches!(
            texture.get_bytes(&mut got, 16, 0, Region::new_1d(14, 4), 0, 0),
            Err(Error::Invalid(_))
        ));
        let made = device.new_texture(&d).unwrap();
        let mut got = vec![0xFFu8; 64];
        made.get_bytes(&mut got, 64, 0, Region::new_1d(0, 16), 0, 0)
            .unwrap();
        assert_eq!(got, [0u8; 64]);
        let written = pattern(64, 7);
        made.replace_region(Region::new_1d(0, 16), 0, 0, &written, 64, 0)
            .unwrap();
        made.get_bytes(&mut got, 64, 0, Region::new_1d(0, 16), 0, 0)
            .unwrap();
        assert_eq!(got, written);
        let mut private = d;
        private.resource_options = ResourceOptions::private();
        let hidden = device.new_texture(&private).unwrap();
        assert_eq!(
            hidden
                .get_bytes(&mut got, 64, 0, Region::new_1d(0, 16), 0, 0)
                .unwrap_err(),
            Error::Invalid(Invalid::new(
                "get_bytes",
                Problem::Copy(CopyProblem::CpuAccessPrivate)
            ))
        );
    });
}

/// Kernels whose texture-buffer slots nothing is bound to read and write the device's neutral
/// texel buffers: their dispatches complete with a clean execution report and the validation
/// layer reports nothing (what Metal reads there is undefined, so the bytes are not compared).
#[test]
fn unbound_texture_buffer_slots_use_neutrals() {
    on_device_with(at_plan, |device| {
        for name in TEXEL_CASES {
            let case = public_case(name).unwrap();
            let pipeline = pipeline_state(device, &case).unwrap();
            let buffers = case
                .compute
                .buffers
                .iter()
                .map(|(&index, bytes)| (index, buffer_with(device, bytes).unwrap()))
                .collect();
            let queue = device.new_command_queue().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encode_case(
                &mut encoder,
                &pipeline,
                &case.compute,
                &buffers,
                &CaseResources::default(),
            )
            .unwrap();
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(
                command_buffer.status(),
                CommandBufferStatus::Completed,
                "{name}"
            );
            assert_eq!(
                command_buffer.execution_report(),
                Some(StatusReport::default()),
                "{name}"
            );
        }
    });
}

/// A 2D texture and an array over a buffer read and write the buffer's bytes where Metal lays
/// their texels: rows `bytes_per_row` apart, the bytes between them left alone, and an array's
/// slices `bytes_per_row` × height apart, rounded up to 128 bytes.
#[test]
fn textures_over_buffers_lie_where_metal_lays_them() {
    on_device(|device| {
        let bytes = pattern(4096, 5);
        let buffer = buffer_with(device, &bytes).unwrap();
        let mut d = TextureDescriptor::texture_2d(PixelFormat::R32Uint, 3, 2, false);
        d.resource_options = ResourceOptions::shared();
        let flat = buffer.new_texture(&d, 256, 32).unwrap();
        let mut got = vec![0u8; 24];
        flat.get_bytes(&mut got, 12, 0, Region::new_2d(0, 0, 3, 2), 0, 0)
            .unwrap();
        assert_eq!(got, [&bytes[256..268], &bytes[288..300]].concat());
        let written = pattern(24, 77);
        flat.replace_region(Region::new_2d(0, 0, 3, 2), 0, 0, &written, 12, 0)
            .unwrap();
        let mut expected = bytes.clone();
        expected[256..268].copy_from_slice(&written[..12]);
        expected[288..300].copy_from_slice(&written[12..]);
        assert_eq!(read_buffer(&buffer, 0, 4096).unwrap(), expected);
        // Two 4x3 slices of 16-byte rows from 1024: the second starts 128 bytes after the first.
        let mut d = TextureDescriptor::texture_2d(PixelFormat::R32Uint, 4, 3, false);
        d.texture_type = TextureType::Type2DArray;
        d.array_length = 2;
        d.resource_options = ResourceOptions::shared();
        let array = buffer.new_linear_texture(&d, 1024, 16, 0).unwrap();
        let mut got = vec![0u8; 48];
        array
            .get_bytes(&mut got, 16, 48, Region::new_2d(0, 0, 4, 3), 0, 1)
            .unwrap();
        assert_eq!(got, expected[1152..1200]);
    });
}

/// A `Private` texture buffer is made shareable, as Metal makes one (a Metal probe on the M3 Max
/// and an M4), and a texture opened from its handle is a texture buffer over the same texels, with
/// a resource ID of its own: the case's texels uploaded by a blit into the first, the case run with
/// the second bound, and the first copied out give the case's own output (`TEXEL_CASES`' writer,
/// which writes four bytes of a texture buffer). A view of it is not shareable.
#[test]
fn a_shared_texture_buffer_opens_over_its_texels() {
    on_device_with(at_plan, |device| {
        let case = public_case(TEXEL_CASES[0]).unwrap();
        let written = run_case(device, &case).unwrap().output;
        // The case writes a 1D texture and a texture buffer; its output is the texture buffer.
        let ComputeOutput::Texture { binding: index, .. } = case.compute.output else {
            panic!("{}: the case's output is no texture", TEXEL_CASES[0]);
        };
        let literal = &case.compute.textures[&index];
        let mut descriptor = texture_descriptor(literal, case.texture_usage[&index]).unwrap();
        descriptor.resource_options = ResourceOptions::private();
        let original = device.new_shared_texture(&descriptor).unwrap();
        assert!(original.is_shareable());
        assert_eq!(original.texture_type(), TextureType::TextureBuffer);
        let handle = original.new_shared_texture_handle().unwrap();
        let opened = device
            .new_shared_texture_with_handle(&handle, IdPlacement::Any)
            .unwrap();
        assert!(opened.is_shareable());
        assert_eq!(opened.texture_type(), TextureType::TextureBuffer);
        assert_eq!(
            (opened.pixel_format(), opened.width()),
            (original.pixel_format(), original.width())
        );
        assert_ne!(opened.gpu_resource_id(), original.gpu_resource_id());
        let view = original.new_texture_view(original.pixel_format()).unwrap();
        assert!(!view.is_shareable());

        let length = literal.bytes.len() as u64;
        let size = Size::new(original.width(), 1, 1);
        let source = buffer_with(device, &literal.bytes).unwrap();
        let readback = buffer_with(device, &vec![0xEE; literal.bytes.len()]).unwrap();
        let queue = device.new_command_queue().unwrap();
        let upload = queue.command_buffer().unwrap();
        let mut blit = upload.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_texture(
            &source,
            0,
            length,
            length,
            size,
            &original,
            0,
            0,
            Origin::new(0, 0, 0),
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        upload.commit().unwrap();

        let mut resources = CaseResources::new(device, &case).unwrap();
        resources.textures.insert(index, opened.clone());
        let buffers = bound_buffers(device, &case.compute, &resources).unwrap();
        let pipeline = pipeline_state(device, &case).unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encode_case(&mut encoder, &pipeline, &case.compute, &buffers, &resources).unwrap();
        encoder.end_encoding().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_texture_to_buffer(
            &original,
            0,
            0,
            Origin::new(0, 0, 0),
            size,
            &readback,
            0,
            length,
            length,
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        assert_eq!(read_buffer(&readback, 0, length).unwrap(), written);
    });
}

/// A texture buffer the device makes, plain or shareable, takes no range of the Metal address
/// space (the runtime's `address.rs`): Metal shows no address of a texture's texels. With the
/// whole space held, each is made, and the CPU reads back what it wrote through the plain one; a
/// texture buffer whose texels took a range of the space would find none (`OutOfMemory`).
#[test]
fn texture_buffers_the_device_makes_lie_outside_the_metal_address_space() {
    on_device(|device| {
        let space = holding_the_space(device);
        let d = texture_buffer(PixelFormat::RGBA8Uint, 16);
        let made = device.new_texture(&d).unwrap_or_else(|error| {
            panic!("a texture buffer made while the space is held: {error}")
        });
        let written = pattern(64, 11);
        made.replace_region(Region::new_1d(0, 16), 0, 0, &written, 64, 0)
            .unwrap();
        let mut got = vec![0u8; 64];
        made.get_bytes(&mut got, 64, 0, Region::new_1d(0, 16), 0, 0)
            .unwrap();
        assert_eq!(got, written);
        let mut shareable = d;
        shareable.resource_options = ResourceOptions::private();
        let shared = device
            .new_shared_texture(&shareable)
            .unwrap_or_else(|error| {
                panic!("a shareable texture buffer made while the space is held: {error}")
            });
        assert_eq!(shared.texture_type(), TextureType::TextureBuffer);
        drop((made, shared, space));
    });
}

/// A texture buffer over a buffer at an offset off 16 bytes, which Metal's validation alone
/// stops: a plain run makes it, its functions read and write it from the offset rounded down to
/// 16 bytes, and its blits and the CPU at the offset itself (an M4 under macOS 27.0.1). Each of
/// [`TEXEL_CASES`] runs with its texture buffer at offsets 4, 20 and 33 of a buffer whose bytes
/// from the offset rounded down are the texture's: the reader gives its own output and leaves the
/// buffer as it was, the writer's bytes land from the offset rounded down and nowhere else, and
/// `get_bytes` and a blit read the bytes at the offset itself.
#[test]
fn a_texture_buffer_off_16_bytes_reads_and_writes_from_below() {
    on_device_with(at_plan, |device| {
        let queue = device.new_command_queue().unwrap();
        let mut runs = 0;
        for name in TEXEL_CASES {
            let case = public_case(name).unwrap();
            let own = run_case(device, &case).unwrap().output;
            let pipeline = pipeline_state(device, &case).unwrap();
            for offset in [4u64, 20, 33] {
                let below = (offset - offset % 16) as usize;
                let mut resources = CaseResources::new(device, &case).unwrap();
                let mut placed = None;
                for (&slot, literal) in &case.compute.textures {
                    let descriptor =
                        texture_descriptor(literal, case.texture_usage[&slot]).unwrap();
                    if descriptor.texture_type != TextureType::TextureBuffer {
                        continue;
                    }
                    let len = literal.bytes.len();
                    let mut bytes = pattern(below + len + 64, 41);
                    bytes[below..below + len].copy_from_slice(&literal.bytes);
                    let buffer = buffer_with(device, &bytes).unwrap();
                    let texture = buffer.new_texture(&descriptor, offset, len as u64).unwrap();
                    let at = offset as usize;
                    let mut got = vec![0u8; len];
                    let row = Region::new_1d(0, descriptor.width);
                    texture
                        .get_bytes(&mut got, len as u64, 0, row, 0, 0)
                        .unwrap();
                    assert_eq!(got, bytes[at..at + len], "{name} at {offset}: get_bytes");
                    resources.textures.insert(slot, texture.clone());
                    placed = Some((slot, buffer, bytes, texture, descriptor.width));
                }
                let (slot, buffer, before, texture, width) = placed.expect("a texture buffer");
                let len = before.len() - below - 64;
                let buffers = bound_buffers(device, &case.compute, &resources).unwrap();
                let readback = buffer_with(device, &vec![0xEE; len]).unwrap();
                let command_buffer = queue.command_buffer().unwrap();
                let mut encoder = command_buffer.compute_command_encoder().unwrap();
                encode_case(&mut encoder, &pipeline, &case.compute, &buffers, &resources).unwrap();
                encoder.end_encoding().unwrap();
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.copy_from_texture_to_buffer(
                    &texture,
                    0,
                    0,
                    Origin::new(0, 0, 0),
                    Size::new(width, 1, 1),
                    &readback,
                    0,
                    len as u64,
                    len as u64,
                    BlitOption::None,
                )
                .unwrap();
                blit.end_encoding().unwrap();
                command_buffer.commit().unwrap();
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(command_buffer.error(), None, "{name} at {offset}");
                let after = read_buffer(&buffer, 0, before.len() as u64).unwrap();
                let at = offset as usize;
                assert_eq!(
                    read_buffer(&readback, 0, len as u64).unwrap(),
                    after[at..at + len],
                    "{name} at {offset}: the blit reads at the offset"
                );
                match case.compute.output {
                    ComputeOutput::Texture { binding, .. } if binding == slot => {
                        assert_eq!(after[below..below + len], own, "{name} at {offset}");
                        assert_eq!(after[..below], before[..below], "{name} at {offset}");
                        assert_eq!(
                            after[below + len..],
                            before[below + len..],
                            "{name} at {offset}"
                        );
                    }
                    ComputeOutput::Buffer {
                        binding,
                        offset: start,
                        length,
                    } => {
                        let output = read_buffer(&buffers[&binding], start, length).unwrap();
                        assert_eq!(output, own, "{name} at {offset}");
                        assert_eq!(after, before, "{name} at {offset}: the reader wrote");
                    }
                    ref other => panic!("{name}: an output of another kind: {other:?}"),
                }
                runs += 1;
            }
        }
        assert_eq!(runs, 6);
    });
}
