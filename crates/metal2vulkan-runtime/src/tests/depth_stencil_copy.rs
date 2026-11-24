//! A blit's copy between textures of a format of depth and stencil both, under a device cfg, on
//! every device of the cfg's kind, as a guest copies a depth and stencil render target it also
//! samples (one level, Private, render target and shader read usage).
//!
//! - `Depth32Float_Stencil8`, 300 x 150: a pass clears a source's depth to 0.25 and its stencil to
//!   0x5a, and a destination's to 0.75 and 0xa5; blits write a region of the source's depth
//!   (`i / 65536`, each exact in `[0, 1)`) and an overlapping region of its stencil from buffers
//!   through the plane options. Then one blit encoder copies a region from the source into the
//!   destination, a region of the destination into another apart from it, and a region of the
//!   destination onto itself a texel right and two down, overlapping. Both planes of both
//!   textures read back through the plane options, every texel the host's model of the copies,
//!   each reading its source whole before it writes (`tests::blit`, the M3's answer).
//! - `Depth24Unorm_Stencil8` where the device holds it in `D24_UNORM_S8_UINT` (no other form of it
//!   is made; else the test says SKIPPED with the device's format query): the same at 4 x 150,
//!   the depths the clears', read through the owned `depth_fill` fixture's `depth_fill_read`
//!   kernel, four texels a row, before and after the copies, as no copy to a buffer takes its
//!   depth.
//! - Recorded (`tests::layouts`): each `vkCmdCopyImage` of a copy names both aspects in each
//!   region, the textures in `TRANSFER_SRC_OPTIMAL` and `TRANSFER_DST_OPTIMAL` between two, in
//!   `GENERAL` within one; the overlapping copy is more than one command, in slabs. Each run
//!   prints the formats, aspects and layouts recorded.
//!
//! `llvm-as` assembles the fixture; without it the 24-bit run skips visibly.

use crate::tests::command::on_device::{each_device, finished};
use crate::tests::draw::on_device::Buffers;
use crate::tests::layouts::{image_copies, Recording};
use crate::texture::TextureObject;
use crate::{
    BlitOption, Buffer, BufferLayout, CommandBuffer, ComputePipelineDescriptor,
    ComputePipelineState, Device, DispatchType, FunctionDescriptor, IndexClass, Library,
    LoadAction, PixelFormat, RenderPassDescriptor, StageSource, StorageMode, StoreAction, Texture,
    TextureDescriptor, TexturePlace, TextureType, TextureUsage,
};
use ash::vk::{self, Handle};
use metal2vulkan::device::FormatUsage;
use metal2vulkan::vk::{Format, ImageAspectFlags, ImageLayout};
use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};

/// A region's first texel and its size.
type Area = ([u32; 2], [u32; 2]);

/// One run: the textures' format and size, the regions of the source the buffers write, and the
/// three copies, each from, size and to: into the destination from the source, then within the
/// destination apart and overlapping.
struct Run {
    format: PixelFormat,
    size: [u32; 2],
    depth_in: Option<Area>,
    stencil_in: Area,
    copies: [([u32; 2], [u32; 2], [u32; 2]); 3],
}

const D32S8: Run = Run {
    format: PixelFormat::Depth32FloatStencil8,
    size: [300, 150],
    depth_in: Some(([5, 3], [250, 140])),
    stencil_in: ([20, 10], [260, 120]),
    copies: [
        ([10, 7], [200, 100], [60, 30]),
        ([60, 30], [50, 20], [240, 0]),
        ([60, 30], [120, 60], [61, 32]),
    ],
};

const D24S8: Run = Run {
    format: PixelFormat::Depth24UnormStencil8,
    size: [4, 150],
    depth_in: None,
    stencil_in: ([0, 10], [4, 120]),
    copies: [
        ([0, 7], [4, 100], [0, 30]),
        ([0, 30], [4, 20], [0, 0]),
        ([0, 30], [3, 60], [1, 32]),
    ],
};

/// The clears: the source's depth and stencil, the destination's.
const CLEARS: [(f64, u32); 2] = [(0.25, 0x5a), (0.75, 0xa5)];

/// A texture's two planes as the host models them: each texel's depth (its read's bits) and
/// stencil, rows `width` texels long.
#[derive(Clone, Debug, PartialEq)]
struct Planes {
    width: usize,
    depth: Vec<u32>,
    stencil: Vec<u8>,
}

impl Planes {
    /// The texels of `area`, row by row.
    fn texels(&self, (at, size): Area) -> impl Iterator<Item = usize> + '_ {
        let [x, y, w, h] = [at[0], at[1], size[0], size[1]].map(|v| v as usize);
        (y..y + h).flat_map(move |row| (x..x + w).map(move |column| row * self.width + column))
    }

    /// Copies `size` texels at `from` in `source` to `to` here, reading the source whole first.
    fn copy(&mut self, source: &Planes, from: [u32; 2], size: [u32; 2], to: [u32; 2]) {
        let read: Vec<(u32, u8)> = (source.texels((from, size)))
            .map(|i| (source.depth[i], source.stencil[i]))
            .collect();
        let texels: Vec<usize> = self.texels((to, size)).collect();
        for (i, (depth, stencil)) in texels.into_iter().zip(read) {
            self.depth[i] = depth;
            self.stencil[i] = stencil;
        }
    }
}

/// The depths a buffer writes: `i / 65536` at its `i`-th texel, exact in a `float`.
fn depths(count: usize) -> Vec<f32> {
    (0..count).map(|i| i as f32 / 65536.0).collect()
}

/// The stencil bytes a buffer writes.
fn stencils(count: usize) -> Vec<u8> {
    (0..count).map(|i| (i * 37 + 11) as u8).collect()
}

/// A Shared buffer's bytes, as the host reads them.
fn read(buffer: &Buffer) -> Vec<u8> {
    let contents = buffer.contents().expect("a Shared buffer");
    // SAFETY: the buffer is alive, `length` bytes long, and no GPU work uses it.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize) }.to_vec()
}

/// The 32-bit words of `bytes`.
fn words(bytes: &[u8]) -> Vec<u32> {
    let (words, _) = bytes.as_chunks::<4>();
    words.iter().map(|word| u32::from_le_bytes(*word)).collect()
}

/// Panics, naming the first texel and how many differ, where `got` is not `want`.
fn exact<T: PartialEq + std::fmt::Debug>(
    name: &str,
    what: &str,
    width: usize,
    got: &[T],
    want: &[T],
) {
    assert_eq!(got.len(), want.len(), "{name}: {what}: the texels read");
    let differ: Vec<usize> = (0..got.len()).filter(|&i| got[i] != want[i]).collect();
    if let Some(&first) = differ.first() {
        panic!(
            "{name}: {what}: {} of {} texels differ, the first at ({}, {}): {:?}, not {:?}",
            differ.len(),
            got.len(),
            first % width,
            first / width,
            got[first],
            want[first]
        );
    }
}

fn at(origin: [u32; 2]) -> TexturePlace {
    TexturePlace {
        slice: 0,
        level: 0,
        origin: [origin[0], origin[1], 0],
    }
}

/// A tight layout of rows of `row` bytes.
fn tight(row: u64, rows: u32) -> BufferLayout {
    BufferLayout {
        offset: 0,
        bytes_per_row: row,
        bytes_per_image: row * u64::from(rows),
    }
}

/// Reads the depth of each of `textures`, four texels a row, into each of `outs` through `kernel`.
fn kernel_reads(
    name: &str,
    command_buffer: &CommandBuffer,
    kernel: &ComputePipelineState,
    textures: [&Texture; 2],
    outs: &[Buffer],
    count: u32,
) {
    let mut encoder = (command_buffer.compute_command_encoder(DispatchType::Serial))
        .unwrap_or_else(|error| panic!("{name}: {error}"));
    (encoder.set_compute_pipeline_state(kernel)).unwrap_or_else(|error| panic!("{name}: {error}"));
    for (texture, out) in textures.into_iter().zip(outs) {
        (encoder.set_texture(Some(texture), 0)).unwrap_or_else(|error| panic!("{name}: {error}"));
        (encoder.set_buffer(Some(out), 0, 0)).unwrap_or_else(|error| panic!("{name}: {error}"));
        (encoder.dispatch_threads([count, 1, 1], [64, 1, 1]))
            .unwrap_or_else(|error| panic!("{name}: the read: {error}"));
    }
    encoder.end_encoding();
}

/// Runs `run` on `device`, its depths read through `kernel` where it has one and through the
/// depth plane option where not, its two textures at indices `index` and the next.
fn copied(
    name: &str,
    device: &Device,
    run: &Run,
    kernel: Option<&ComputePipelineState>,
    index: u32,
) {
    use BlitOption::{DepthFromDepthStencil as Depth, StencilFromDepthStencil as Stencil};
    let handle = device.core().device.handle();
    let [width, height] = run.size;
    let texels = (width * height) as usize;
    let mut d = TextureDescriptor::new(TextureType::Type2D, run.format, width, height);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.storage_mode = StorageMode::Private;
    let made = |index| device.new_texture(&d, Some(index));
    let source = match made(index) {
        Ok(texture) => texture,
        Err(error) => {
            let profile = device.core().translator.profile();
            let held = |usage| profile.supports_format(Format::D24_UNORM_S8_UINT, usage);
            println!(
                "SKIPPED: {name}: {:?}: {error}; the format query: D24_UNORM_S8_UINT depth-stencil \
                 attachment {}, sampled image {}",
                run.format,
                held(FormatUsage::DepthStencilAttachment),
                held(FormatUsage::SampledImage)
            );
            return;
        }
    };
    let destination = made(index + 1).unwrap_or_else(|error| panic!("{name}: {error}"));
    let both = [&source, &destination];
    let mut buffers = Buffers::sized(device, 1 << 21);
    let depth_in = (run.depth_in).map(|(_, [w, h])| depths((w * h) as usize));
    let depth_buffer = (depth_in.as_ref()).map(|values| {
        buffers.of(
            device,
            &values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
    });
    let stencil_in = stencils((run.stencil_in.1[0] * run.stencil_in.1[1]) as usize);
    let stencil_buffer = buffers.of(device, &stencil_in);
    let unwritten = |len: usize| vec![0xee; len];
    let stencil_outs = both.map(|_| buffers.of(device, &unwritten(texels)));
    let depth_outs = [0, 1, 2, 3].map(|_| buffers.of(device, &unwritten(4 * texels)));

    let queue = device.new_command_queue(4).expect("a queue");
    let command_buffer = queue.command_buffer().expect("a command buffer");
    for (texture, (depth, stencil)) in both.iter().zip(CLEARS) {
        let mut pass = RenderPassDescriptor::default();
        let z = &mut pass.depth_attachment;
        z.texture = Some((*texture).clone());
        z.load_action = LoadAction::Clear;
        z.store_action = StoreAction::Store;
        z.clear_depth = depth;
        let s = &mut pass.stencil_attachment;
        s.texture = Some((*texture).clone());
        s.load_action = LoadAction::Clear;
        s.store_action = StoreAction::Store;
        s.clear_stencil = stencil;
        (command_buffer.render_command_encoder(&pass))
            .unwrap_or_else(|error| panic!("{name}: {error}"))
            .end_encoding();
    }
    if let Some(kernel) = kernel {
        kernel_reads(
            name,
            &command_buffer,
            kernel,
            both,
            &depth_outs[..2],
            texels as u32,
        );
    }
    let mut blit =
        (command_buffer.blit_command_encoder()).unwrap_or_else(|error| panic!("{name}: {error}"));
    let ok = |result: Result<(), crate::Error>, what: &str| {
        result.unwrap_or_else(|error| panic!("{name}: {what}: {error}"))
    };
    if let (Some((origin, [w, h])), Some(buffer)) = (run.depth_in, &depth_buffer) {
        let layout = tight(4 * u64::from(w), h);
        let into = blit.copy_from_buffer_with_options(
            buffer,
            layout,
            [w, h, 1],
            &source,
            at(origin),
            Depth,
        );
        ok(into, "the depths in");
    }
    let (origin, [w, h]) = run.stencil_in;
    ok(
        blit.copy_from_buffer_with_options(
            &stencil_buffer,
            tight(u64::from(w), h),
            [w, h, 1],
            &source,
            at(origin),
            Stencil,
        ),
        "the stencils in",
    );
    let before = image_copies(handle).len();
    for (n, (from, [w, h], to)) in run.copies.into_iter().enumerate() {
        let from_texture = if n == 0 { &source } else { &destination };
        let copy = blit.copy_from_texture(from_texture, at(from), [w, h, 1], &destination, at(to));
        ok(copy, "a copy");
    }
    let whole = [width, height, 1];
    for (texture, out) in both.iter().zip(&stencil_outs) {
        let layout = tight(u64::from(width), height);
        let back = blit.copy_from_texture_to_buffer_with_options(
            texture,
            at([0, 0]),
            whole,
            out,
            layout,
            Stencil,
        );
        ok(back, "the stencils out");
    }
    if kernel.is_none() {
        for (texture, out) in both.iter().zip(&depth_outs[2..]) {
            let layout = tight(4 * u64::from(width), height);
            let back = blit.copy_from_texture_to_buffer_with_options(
                texture,
                at([0, 0]),
                whole,
                out,
                layout,
                Depth,
            );
            ok(back, "the depths out");
        }
    }
    blit.end_encoding();
    if let Some(kernel) = kernel {
        kernel_reads(
            name,
            &command_buffer,
            kernel,
            both,
            &depth_outs[2..],
            texels as u32,
        );
    }
    command_buffer.commit().expect("a commit");
    finished(name, &command_buffer);
    assert_eq!(command_buffer.error(), None, "{name}");

    // The model: the clears, as the kernel read them where it reads, then the buffers' writes.
    let w = width as usize;
    let cleared = |n: usize| {
        let (depth, stencil) = CLEARS[n];
        let depth = match kernel {
            Some(_) => words(&read(&depth_outs[n])),
            None => vec![(depth as f32).to_bits(); texels],
        };
        Planes {
            width: w,
            depth,
            stencil: vec![stencil as u8; texels],
        }
    };
    let mut model = [cleared(0), cleared(1)];
    if let (Some(area), Some(values)) = (run.depth_in, &depth_in) {
        let texels: Vec<usize> = model[0].texels(area).collect();
        for (i, value) in texels.into_iter().zip(values) {
            model[0].depth[i] = value.to_bits();
        }
    }
    let texels_in: Vec<usize> = model[0].texels(run.stencil_in).collect();
    for (i, value) in texels_in.into_iter().zip(&stencil_in) {
        model[0].stencil[i] = *value;
    }
    for (n, (from, size, to)) in run.copies.into_iter().enumerate() {
        let source_planes = match n {
            0 => model[0].clone(),
            _ => model[1].clone(),
        };
        model[1].copy(&source_planes, from, size, to);
    }
    for (n, what) in ["the source", "the destination"].into_iter().enumerate() {
        let got = Planes {
            width: w,
            depth: words(&read(&depth_outs[2 + n])),
            stencil: read(&stencil_outs[n]),
        };
        exact(
            name,
            &format!("{what}'s depth"),
            w,
            &got.depth,
            &model[n].depth,
        );
        exact(
            name,
            &format!("{what}'s stencil"),
            w,
            &got.stencil,
            &model[n].stencil,
        );
    }

    // What ran: the copies' commands between the two images, as recorded, by their numbers;
    // printed by the registry's names.
    let images = [source.image(), destination.image()].map(|image| image.as_raw());
    let recorded: Vec<_> = (image_copies(handle).into_iter().skip(before))
        .filter(|copy| {
            images.contains(&copy.source.as_raw()) && images.contains(&copy.destination.as_raw())
        })
        .collect();
    let aspects = vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL;
    let between = [
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    ];
    let within = [vk::ImageLayout::GENERAL; 2];
    assert!(
        recorded.len() > 3,
        "{name}: {} copy commands between the textures",
        recorded.len()
    );
    for (n, copy) in recorded.iter().enumerate() {
        let (pair, layouts) = match n {
            0 => (images, between),
            _ => ([images[1]; 2], within),
        };
        let raw = [copy.source, copy.destination].map(|image| image.as_raw());
        assert_eq!(raw, pair, "{name}: command {n}'s images");
        assert_eq!(
            copy.layouts.map(vk::ImageLayout::as_raw),
            layouts.map(vk::ImageLayout::as_raw),
            "{name}: command {n}'s layouts"
        );
        let masks: Vec<[u32; 2]> = (copy.aspects.iter())
            .map(|masks| masks.map(vk::ImageAspectFlags::as_raw))
            .collect();
        assert!(
            !masks.is_empty() && masks.iter().all(|m| *m == [aspects.as_raw(); 2]),
            "{name}: command {n}'s regions' aspects: {masks:?}"
        );
    }
    let format = |texture: &Texture| Format::from_raw(texture.format().as_raw());
    let layout = |layout: vk::ImageLayout| ImageLayout::from_raw(layout.as_raw());
    println!(
        "{name}: {:?} in {:?} to {:?}, {width} x {height}: {} vkCmdCopyImage, every region \
         {:?} from and to; between the textures {:?} to {:?}, within the destination {:?} \
         (apart 1 command, overlapping {} in slabs); both planes of both textures exact, \
         {texels} texels each, depth read by {}",
        run.format,
        format(&source),
        format(&destination),
        recorded.len(),
        ImageAspectFlags::from_raw(aspects.as_raw()),
        layout(between[0]),
        layout(between[1]),
        layout(within[0]),
        recorded.len() - 2,
        match kernel {
            Some(_) => "the depth_fill_read kernel",
            None => "the depth plane option",
        }
    );
}

#[test]
fn every_device_copies_both_aspects_of_a_depth_and_stencil_texture() {
    let library = match llvm_assemble(&engine_dir().join("depth_fill.ll")) {
        Ok(bitcode) => Some(Library::from_air(&bitcode).expect("the library")),
        Err(AssembleError::Missing(why)) => {
            println!("SKIPPED: the 24-bit run: {why}");
            None
        }
        Err(error) => panic!("depth_fill.ll: {error:?}"),
    };
    let _recording = Recording::start();
    each_device("depth_stencil_copy", |name, _, device| {
        let first = IndexClass::Texture.range().start;
        let _grant = (device.grant_indices(IndexClass::Texture, first, 4))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        copied(name, &device, &D32S8, None, first);
        let Some(library) = &library else {
            return;
        };
        let function = library.function("depth_fill_read").expect("the kernel");
        let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(function),
        ));
        let kernel = (device
            .new_compute_pipeline_state(&descriptor, crate::tests::no_constexpr_samplers))
        .unwrap_or_else(|error| panic!("{name}: the kernel: {error}"));
        copied(name, &device, &D24S8, Some(&kernel), first + 2);
    });
}
