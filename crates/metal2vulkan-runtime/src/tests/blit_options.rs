//! A copy between a buffer and a texture of a `BlitOption`.
//!
//! - A plane option takes the aspect it names where the pixel format has it: the depth (4 bytes a
//!   texel) or the stencil (1) of `Depth32Float_Stencil8`, the depth of `Depth16Unorm` and the
//!   stencil of `Stencil8`, as Metal takes the option of a format of that aspect alone (the
//!   conformance crate's `buffer-depth-samples` rows). `None` of a format of both, a plane option of
//!   a format without its aspect and one of a multisample texture are `Gap::TextureBytes`; a copy
//!   into the depth of `Depth32Float_Stencil8` is `Gap::FloatDepthUpload`, as one into
//!   `Depth32Float` is.
//! - Under a device cfg, on every device of the cfg's kind: a pass clears a `Depth32Float_Stencil8`
//!   texture's depth to 0.25 and its stencil to 0x5a; a copy of `StencilFromDepthStencil` writes a
//!   region's stencil from a buffer; copies of each plane option read the whole level back, the
//!   region's bytes with 0x5a around them and 0.25 at every depth texel. A `Depth16Unorm` texture's
//!   depth and a `Stencil8` texture's stencil go in and back out of their options bit for bit.

use crate::blit::{buffer_copy, BufferCopy, Lies, Side};
use crate::error::{Error, Gap};
use crate::{BlitOption, BufferLayout, PixelFormat, TextureDescriptor, TexturePlace, TextureType};
use ash::vk;

/// What a copy of `option` between an 8 x 8 texture of `pf`, held in `format` with `samples`, and
/// a tight buffer answers, into the texture where `into_image`.
fn copy(
    pf: PixelFormat,
    format: vk::Format,
    samples: u32,
    option: BlitOption,
    into_image: bool,
) -> Result<Option<BufferCopy>, Error> {
    let mut d = TextureDescriptor::new(TextureType::Type2D, pf, 8, 8);
    d.sample_count = samples;
    let side = Side {
        descriptor: &d,
        format,
        unit: None,
        base: [0, 0],
        rows: None,
        bound: None,
    };
    let lies = Lies {
        buffer: vk::Buffer::null(),
        base: 0,
        len: 4096,
        layout: BufferLayout {
            offset: 0,
            bytes_per_row: 32,
            bytes_per_image: 256,
        },
    };
    let at = TexturePlace {
        slice: 0,
        level: 0,
        origin: [0; 3],
    };
    buffer_copy(&side.with_option(option)?, at, [8, 8, 1], lies, into_image)
}

/// The aspect (its bits) and the unit's bytes of the copy `result`, which copies something.
fn plane(result: Result<Option<BufferCopy>, Error>) -> (u32, u32) {
    match result {
        Ok(Some(copy)) => {
            let aspect = copy.unit.aspect;
            assert!(copy
                .slabs()
                .flatten()
                .all(|r| r.image_subresource.aspect_mask == aspect));
            (aspect.as_raw(), copy.unit.bytes)
        }
        Ok(None) => panic!("it copies nothing"),
        Err(error) => panic!("{error}"),
    }
}

fn gap(result: Result<Option<BufferCopy>, Error>) -> Gap {
    match result {
        Err(Error::Unsupported(gap)) => gap,
        Err(other) => panic!("not a gap: {other}"),
        Ok(_) => panic!("not a gap: it was checked"),
    }
}

#[test]
fn a_plane_option_copies_the_aspect_it_names_where_the_format_has_it() {
    use vk::ImageAspectFlags as A;
    use BlitOption::{DepthFromDepthStencil as Depth, StencilFromDepthStencil as Stencil};
    let both = PixelFormat::Depth32FloatStencil8;
    let d32s8 = vk::Format::D32_SFLOAT_S8_UINT;
    for into_image in [false, true] {
        assert_eq!(
            plane(copy(both, d32s8, 1, Stencil, into_image)),
            (A::STENCIL.as_raw(), 1)
        );
        let d16 = (PixelFormat::Depth16Unorm, vk::Format::D16_UNORM);
        assert_eq!(
            plane(copy(d16.0, d16.1, 1, Depth, into_image)),
            (A::DEPTH.as_raw(), 2)
        );
        let s8 = (PixelFormat::Stencil8, vk::Format::S8_UINT);
        assert_eq!(
            plane(copy(s8.0, s8.1, 1, Stencil, into_image)),
            (A::STENCIL.as_raw(), 1)
        );
        let refused = [
            (both, d32s8, 1, BlitOption::None),
            (s8.0, s8.1, 1, Depth),
            (d16.0, d16.1, 1, Stencil),
            (
                PixelFormat::RGBA8Unorm,
                vk::Format::R8G8B8A8_UNORM,
                1,
                Stencil,
            ),
            (s8.0, s8.1, 4, Stencil),
            (both, d32s8, 4, Depth),
        ];
        for (pf, format, samples, option) in refused {
            assert_eq!(
                gap(copy(pf, format, samples, option, into_image)),
                Gap::TextureBytes { pixel_format: pf },
                "{pf:?} at {samples} samples, {option:?}"
            );
        }
    }
    assert_eq!(
        plane(copy(both, d32s8, 1, Depth, false)),
        (A::DEPTH.as_raw(), 4)
    );
    assert_eq!(
        gap(copy(both, d32s8, 1, Depth, true)),
        Gap::FloatDepthUpload { pixel_format: both }
    );
}

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::each_device;
    use crate::tests::draw::on_device::{run, target, Buffers, SIDE};
    use crate::{
        BlitOption, Buffer, BufferLayout, LoadAction, PixelFormat, RenderPassDescriptor,
        StoreAction, TexturePlace, TextureUsage,
    };

    /// A Shared buffer's bytes, as the host reads them.
    fn read(buffer: &Buffer) -> Vec<u8> {
        let contents = buffer.contents().expect("a Shared buffer");
        // SAFETY: the buffer is alive, `length` bytes long, and no GPU work uses it.
        unsafe { std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize) }.to_vec()
    }

    /// A place at `origin` in level 0 and slice 0.
    fn at(origin: [u32; 3]) -> TexturePlace {
        TexturePlace {
            slice: 0,
            level: 0,
            origin,
        }
    }

    /// A tight layout of rows of `row` bytes.
    fn tight(row: u64) -> BufferLayout {
        BufferLayout {
            offset: 0,
            bytes_per_row: row,
            bytes_per_image: row * u64::from(SIDE),
        }
    }

    #[test]
    fn every_device_copies_each_plane_of_a_depth_and_stencil_texture() {
        each_device("blit_options", |name, _, device| {
            use BlitOption::{DepthFromDepthStencil as Depth, StencilFromDepthStencil as Stencil};
            let render = TextureUsage::RENDER_TARGET;
            let both = target(&device, PixelFormat::Depth32FloatStencil8, render);
            let d16 = target(&device, PixelFormat::Depth16Unorm, render);
            let s8 = target(&device, PixelFormat::Stencil8, render);
            let side = SIDE as usize;
            let mut buffers = Buffers::new(&device);
            // A 3 x 2 region at (2, 1) of the stencil, its bytes 3 apart.
            let region: Vec<u8> = (0..6).map(|i| 0x10 + i * 0x21).collect();
            let stencil_in = buffers.of(&device, &region);
            let unwritten = |len: usize| vec![0xee; len];
            let stencil_out = buffers.of(&device, &unwritten(side * side));
            let depth_out = buffers.of(&device, &unwritten(4 * side * side));
            let d16_bytes: Vec<u8> = (0..side * side)
                .flat_map(|i| ((i * 7919 + 13) as u16).to_le_bytes())
                .collect();
            let s8_bytes: Vec<u8> = (0..side * side).map(|i| (i * 37 + 11) as u8).collect();
            let (d16_in, s8_in) = (
                buffers.of(&device, &d16_bytes),
                buffers.of(&device, &s8_bytes),
            );
            let d16_out = buffers.of(&device, &unwritten(2 * side * side));
            let s8_out = buffers.of(&device, &unwritten(side * side));

            let mut pass = RenderPassDescriptor::default();
            let z = &mut pass.depth_attachment;
            z.texture = Some(both.clone());
            z.load_action = LoadAction::Clear;
            z.store_action = StoreAction::Store;
            z.clear_depth = 0.25;
            let s = &mut pass.stencil_attachment;
            s.texture = Some(both.clone());
            s.load_action = LoadAction::Clear;
            s.store_action = StoreAction::Store;
            s.clear_stencil = 0x5a;
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            (command_buffer.render_command_encoder(&pass))
                .unwrap_or_else(|error| panic!("{name}: {error}"))
                .end_encoding();
            let mut blit = (command_buffer.blit_command_encoder())
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let ok = |result: Result<(), crate::Error>| {
                result.unwrap_or_else(|error| panic!("{name}: {error}"))
            };
            let whole = [SIDE, SIDE, 1];
            ok(blit.copy_from_buffer_with_options(
                &stencil_in,
                tight(3),
                [3, 2, 1],
                &both,
                at([2, 1, 0]),
                Stencil,
            ));
            let tight_s8 = tight(u64::from(SIDE));
            ok(blit.copy_from_texture_to_buffer_with_options(
                &both,
                at([0; 3]),
                whole,
                &stencil_out,
                tight_s8,
                Stencil,
            ));
            ok(blit.copy_from_texture_to_buffer_with_options(
                &both,
                at([0; 3]),
                whole,
                &depth_out,
                tight(4 * u64::from(SIDE)),
                Depth,
            ));
            let tight_d16 = tight(2 * u64::from(SIDE));
            ok(blit.copy_from_buffer_with_options(
                &d16_in,
                tight_d16,
                whole,
                &d16,
                at([0; 3]),
                Depth,
            ));
            ok(blit.copy_from_texture_to_buffer_with_options(
                &d16,
                at([0; 3]),
                whole,
                &d16_out,
                tight_d16,
                Depth,
            ));
            ok(blit.copy_from_buffer_with_options(
                &s8_in,
                tight_s8,
                whole,
                &s8,
                at([0; 3]),
                Stencil,
            ));
            ok(blit.copy_from_texture_to_buffer_with_options(
                &s8,
                at([0; 3]),
                whole,
                &s8_out,
                tight_s8,
                Stencil,
            ));
            blit.end_encoding();
            run(name, &command_buffer);

            let mut stencil = vec![0x5a; side * side];
            for (i, byte) in region.iter().enumerate() {
                stencil[(1 + i / 3) * side + 2 + i % 3] = *byte;
            }
            assert_eq!(read(&stencil_out), stencil, "{name}: the stencil plane");
            let depth: Vec<u8> = (0..side * side)
                .flat_map(|_| 0.25f32.to_le_bytes())
                .collect();
            assert_eq!(read(&depth_out), depth, "{name}: the depth plane");
            assert_eq!(read(&d16_out), d16_bytes, "{name}: Depth16Unorm's depth");
            assert_eq!(read(&s8_out), s8_bytes, "{name}: Stencil8's stencil");
            println!(
                "{name}: the stencil and the depth plane of Depth32Float_Stencil8, the depth of \
                 Depth16Unorm and the stencil of Stencil8 copied through their options"
            );
        });
    }
}
