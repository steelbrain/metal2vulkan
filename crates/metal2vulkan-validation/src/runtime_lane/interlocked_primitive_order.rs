//! The order of interlocked read-modify-writes across the primitives of one draw, on the device
//! the cfg selects, under the validation layer. Metal orders a fragment function's read of its
//! color attachment, and its accesses in a raster order group, after every earlier fragment of the
//! pixel, those of its own draw's earlier primitives included. Both run in the pixel interlock
//! where a device has no rasterization-order color access (`metal2vulkan_runtime`'s
//! `framebuffer_fetch`, the translator's `lower/fragment/rog.rs`), and the runtime records nothing
//! between the primitives of a draw: Vulkan orders them by the interlock alone, its ordered
//! interlock's operations happening "before interlocked operations in fragment shader invocations
//! that execute later in rasterization order and cover at least one sample in the same pixel"
//! (Fragment Operations, "Interlocked Operations"), rasterization order following the order of a
//! draw's primitives ("Primitive Order"). This test is that measurement, the within-draw
//! counterpart of `interlocked_draw_order`'s.
//!
//! Each pass makes one draw of [`PRIMITIVES`] overlapping primitives (`ipo_primitive_order`) whose
//! fragments apply `3c + 1` or `5c + 3`, by the parity of the bits set in their primitive's number,
//! to the pixel's `c` in one of three places ([`Access`]), first set to [`CLEAR`]: an `R32Uint`
//! color attachment by framebuffer fetch (`ipo_mix`), an `R32Uint` texture's texel in raster order
//! group 0 (`ipo_rog_texture`), or a buffer's word in it (`ipo_rog_buffer`). The draw takes one of
//! four [`Shape`]s: `ipo_fan`'s fullscreen triangles or `ipo_steps`' rectangles (each from a corner
//! that steps with its number, [`corner`], to the target's right and bottom edges, two triangles
//! sharing a diagonal), each as one instance of every primitive or as one instance per primitive.
//! The maps do not commute, so a pixel holds [`expected`] only where each fragment read what the
//! fragment of the primitive before it wrote there. [`RUNS`] passes of each access and shape over
//! 1920x1080, 64x36 and 61x37, each read back whole. Every pixel of every pass must hold its
//! composition; the wrong ones are counted and printed per pass, with the 8x4 groups (8 columns by
//! 4 rows from the top left, the last ones cut by the edges) they fall in, how many of those are
//! wrong in every pixel, and the first of them; and per target, the framebuffer attachments that
//! took a storage placeholder (`WorkCounts::storage_placeholder_attachments`), which a device whose
//! fetch takes the storage form (the pixel interlock without rasterization-order color access) must
//! count: its fetching passes left the target out of their subpass and read and wrote it in the
//! interlock. A device without the pixel interlock must refuse the raster order groups, typed
//! (`Unsupported` of `Feature::RasterOrderGroups`), printed as a `ROG-SKIP` line.
//!
//! A buffer in a raster order group is ordered only where set 0 is pushed on a driver that packs a
//! bound set 0 into one Metal argument buffer
//! (`RuntimeDriverDefect::InterlockedBoundBuffersUnordered`: MoltenVK, whose pushed sets also lose
//! their buffers' sizes, so it pushes no other draw's set 0). So each pass counts its draws that
//! pushed set 0 (`WorkCounts::draws_pushing_set_zero`): a
//! [`Access::RogBuffer`] draw must push where the device has the push command and either pushes
//! every set 0 or has that defect, and on a driver whose pushed sets lose their sizes no other
//! draw may push. [`a_bound_set_of_interlocked_buffers_is_announced_once`] takes the bound form:
//! the same draw on a device without the push command.

use super::degrade_notices::{assert_once, made_render_kinds, on_device_with};
use super::framebuffer_fetch::fetches;
use super::{
    at_plan, buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device_with,
    Messages, ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::VkExtension;
use metal2vulkan::error::{Feature, RuntimeDriverDefect};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    BlitOption, Buffer, ClearColor, CommandBuffer, Degradation, Device, Error, Library, LoadAction,
    Origin, PipelineOptions, RenderPassDescriptor, RenderPipelineState, ResourceOptions, Size,
    StoreAction, Texture, TextureDescriptor,
};
use std::collections::BTreeMap;

/// The primitives of a draw: triangles for `ipo_fan`, rectangles for `ipo_steps`.
const PRIMITIVES: u32 = 64;

/// The passes of each shape and size.
const RUNS: usize = 8;

/// Every pixel's value before a pass's draw.
const CLEAR: u32 = 7;

/// The groups a wrong pass names, at most.
const NAMED: usize = 6;

/// A draw's shape: its vertex function, and whether each primitive is an instance of its own.
#[derive(Clone, Copy, Debug)]
struct Shape {
    /// `ipo_steps`' rectangles, else `ipo_fan`'s triangles.
    steps: bool,
    /// One instance per primitive, else one instance of all of them.
    instanced: bool,
}

impl Shape {
    /// The vertex function's name.
    fn vertex(self) -> &'static str {
        match self.steps {
            true => "ipo_steps",
            false => "ipo_fan",
        }
    }

    /// The vertices of a primitive.
    fn vertices(self) -> u32 {
        match self.steps {
            true => 6,
            false => 3,
        }
    }
}

/// The four shapes, in the order the passes take them.
const SHAPES: [Shape; 4] = [
    Shape {
        steps: false,
        instanced: false,
    },
    Shape {
        steps: false,
        instanced: true,
    },
    Shape {
        steps: true,
        instanced: false,
    },
    Shape {
        steps: true,
        instanced: true,
    },
];

/// Where a draw's fragments read, modify and write their pixel's value.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Access {
    /// Color attachment 0, by framebuffer fetch (`ipo_mix`).
    Fetch,
    /// A texture's texel, in raster order group 0 (`ipo_rog_texture`).
    RogTexture,
    /// A buffer's word, in raster order group 0 (`ipo_rog_buffer`).
    RogBuffer,
}

impl Access {
    /// The fragment function's name.
    fn fragment(self) -> &'static str {
        match self {
            Access::Fetch => "ipo_mix",
            Access::RogTexture => "ipo_rog_texture",
            Access::RogBuffer => "ipo_rog_buffer",
        }
    }
}

/// `ipo_corner`: the top left corner of rectangle `k`, in pixels from the target's top left.
fn corner(k: u32) -> (u64, u64) {
    (u64::from((k * 37 + 1) % 61), u64::from((k * 11 + 1) % 23))
}

/// What texel `(x, y)` holds after a draw of `shape`, from [`CLEAR`]: every primitive that covers
/// it, in order, by the parity of its number's bits.
fn expected(shape: Shape, (x, y): (u64, u64)) -> u32 {
    (0..PRIMITIVES)
        .filter(|&k| {
            let (left, top) = corner(k);
            !shape.steps || (left <= x && top <= y)
        })
        .fold(CLEAR, |c, k| match k.count_ones() % 2 {
            1 => c.wrapping_mul(3).wrapping_add(1),
            _ => c.wrapping_mul(5).wrapping_add(3),
        })
}

/// The descriptor of `shape`'s draws through `access`, into an `R32Uint` attachment.
fn descriptor(library: &Library, shape: Shape, access: Access) -> RenderPipelineDescriptor {
    let function = |name: &str| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut descriptor = RenderPipelineDescriptor::new(function(shape.vertex()));
    descriptor.fragment = Some(function(access.fragment()));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::R32Uint));
    descriptor
}

/// Whether `device` draws through the raster order groups: where it has the pixel interlock it
/// must, and elsewhere it must refuse them, typed, printed as a `ROG-SKIP` line.
fn orders_groups(device: &Device, library: &Library) -> bool {
    let fragment = device.compile_device().profile().fragment();
    if fragment.pixel_interlock {
        return true;
    }
    let descriptor = descriptor(library, SHAPES[0], Access::RogTexture);
    let name = device.name();
    match device.new_render_pipeline_state(&descriptor, &PipelineOptions::default()) {
        Err(Error::Unsupported(refusal)) if refusal.feature == Feature::RasterOrderGroups => {
            println!("ROG-SKIP on {name}: no pixel interlock, refused: {refusal}");
            false
        }
        Ok(_) => panic!("{name}: a raster order group's pipeline was made with no interlock"),
        Err(error) => {
            panic!("{name}: a raster order group's pipeline was refused otherwise: {error}")
        }
    }
}

/// Whether `device` pushes set 0 for an [`Access::RogBuffer`] draw, and whether it pushes no
/// other draw's: it has the push command (Vulkan 1.4 or `VK_KHR_push_descriptor`, whose
/// `maxPushDescriptors` these draws' set 0 fits), and its pushed sets hold their buffers' sizes or
/// its driver orders such a buffer only pushed; no other draw pushes where they do not
/// (`RuntimeDriverDefect::PushedBufferSizesUnset`).
fn pushes(device: &Device) -> (bool, bool) {
    let defects = &device.properties().driver_defects;
    let unset = defects.contains(&RuntimeDriverDefect::PushedBufferSizesUnset);
    let ordered = defects.contains(&RuntimeDriverDefect::InterlockedBoundBuffersUnordered);
    let push = device
        .compile_device()
        .profile()
        .descriptors()
        .push_descriptor;
    (push && (!unset || ordered), unset)
}

/// A private `R32Uint` texture of `sides` and `usage`.
fn texture(device: &Device, (width, height): (u64, u64), usage: TextureUsage) -> Texture {
    let mut d = TextureDescriptor::texture_2d(PixelFormat::R32Uint, width, height, false);
    d.usage = usage;
    d.resource_options = ResourceOptions::private();
    device.new_texture(&d).unwrap()
}

/// A render pass over `texture` that clears it to [`CLEAR`] and stores it.
fn cleared(texture: &Texture) -> RenderPassDescriptor {
    let mut descriptor = RenderPassDescriptor::new();
    let color = &mut descriptor.color_attachments[0];
    color.texture = Some(texture.clone());
    color.load_action = LoadAction::Clear;
    color.clear_color = ClearColor::new(f64::from(CLEAR), 0.0, 0.0, 0.0);
    color.store_action = StoreAction::Store;
    descriptor
}

/// The textures of a size the passes reach.
struct Targets {
    /// Each pass's color attachment, which `ipo_mix` fetches.
    target: Texture,
    /// The texture `ipo_rog_texture` reads and writes.
    groups: Texture,
}

/// One pass of `shape`'s draw through `access` over `targets` of `sides`, the place it reaches set
/// to [`CLEAR`] first; its GPU time in milliseconds, and every value that place holds after it.
fn pass(
    device: &Device,
    targets: &Targets,
    pipeline: &RenderPipelineState,
    (shape, access): (Shape, Access),
    sides: (u64, u64),
) -> (f64, Vec<u32>) {
    let words = (access == Access::RogBuffer).then(|| {
        let clear = CLEAR.to_le_bytes().repeat((sides.0 * sides.1) as usize);
        buffer_with(device, &clear).unwrap()
    });
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    if access == Access::RogTexture {
        let clear = command_buffer.render_command_encoder(&cleared(&targets.groups));
        clear.unwrap().end_encoding().unwrap();
    }
    let size = [sides.0 as f32, sides.1 as f32];
    let mut encoder = command_buffer
        .render_command_encoder(&cleared(&targets.target))
        .unwrap();
    encoder.set_render_pipeline_state(pipeline).unwrap();
    encoder
        .set_vertex_bytes(&size.map(f32::to_le_bytes).concat(), 0)
        .unwrap();
    match (access, &words) {
        (Access::RogTexture, _) => encoder
            .set_fragment_texture(Some(&targets.groups), 0)
            .unwrap(),
        (Access::RogBuffer, Some(words)) => {
            encoder.set_fragment_buffer(Some(words), 0, 0).unwrap();
            let width = (sides.0 as u32).to_le_bytes();
            encoder.set_fragment_bytes(&width, 1).unwrap();
        }
        _ => {}
    }
    let (vertices, instances) = match shape.instanced {
        true => (shape.vertices(), PRIMITIVES),
        false => (shape.vertices() * PRIMITIVES, 1),
    };
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, vertices, instances, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    let gpu = completed(&command_buffer);
    let values = match (access, words) {
        (Access::RogBuffer, Some(words)) => words_of(&words, sides),
        (Access::RogTexture, _) => texels(device, &targets.groups, sides),
        _ => texels(device, &targets.target, sides),
    };
    (gpu, values)
}

/// The GPU time of `command_buffer` in milliseconds, once it completed without error.
fn completed(command_buffer: &CommandBuffer) -> f64 {
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    (command_buffer.gpu_end_time().unwrap() - command_buffer.gpu_start_time().unwrap()) * 1e3
}

/// Every word of `buffer`, a pixel's of `sides` each.
fn words_of(buffer: &Buffer, (width, height): (u64, u64)) -> Vec<u32> {
    words(&read_buffer(buffer, 0, width * height * 4).unwrap())
}

/// `bytes` as little-endian words.
fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect()
}

/// Every texel of `target` of `sides`, read back through a buffer.
fn texels(device: &Device, target: &Texture, (width, height): (u64, u64)) -> Vec<u32> {
    let bytes = width * height * 4;
    let buffer = device.new_buffer(bytes, ResourceOptions::shared()).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        target,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(width, height, 1),
        &buffer,
        0,
        width * 4,
        bytes,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    completed(&command_buffer);
    words_of(&buffer, (width, height))
}

/// The 8x4 groups of a target of `sides` that hold a wrong texel, by their top left texel, with
/// whether every texel of the group is wrong.
fn wrong_groups(wrong: &[bool], (width, height): (u64, u64)) -> BTreeMap<(u64, u64), bool> {
    let mut groups = BTreeMap::new();
    for (i, _) in wrong.iter().enumerate().filter(|(_, &w)| w) {
        let (x, y) = (i as u64 % width, i as u64 / width);
        groups.entry((x / 8 * 8, y / 4 * 4)).or_insert(true);
    }
    for (&(left, top), whole) in groups.iter_mut() {
        *whole = (top..(top + 4).min(height))
            .flat_map(|y| (left..(left + 8).min(width)).map(move |x| (x, y)))
            .all(|(x, y)| wrong[(y * width + x) as usize]);
    }
    groups
}

/// Every pass's pixels hold the composition of its draw's primitives in order (the module docs).
#[test]
fn overlapping_primitives_of_a_fetching_draw_read_each_others_writes() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
    if !fetches(&device) {
        return;
    }
    let ll = engine_dir().join("ipo_primitive_order.ll");
    let library = device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap();
    let fragment = device.compile_device().profile().fragment();
    let interlocked = fragment.pixel_interlock && !fragment.rasterization_order_color_access;
    eprintln!(
        "PRIMITIVE-ORDER on {}: rasterization-order color access {}, pixel interlock {}",
        device.name(),
        fragment.rasterization_order_color_access,
        fragment.pixel_interlock,
    );
    let accesses = match orders_groups(&device, &library) {
        true => vec![Access::Fetch, Access::RogTexture, Access::RogBuffer],
        false => vec![Access::Fetch],
    };
    let (buffer_pushes, others_bound) = pushes(&device);
    let mut wrong_passes = Vec::new();
    for sides in [(1920u64, 1080u64), (64, 36), (61, 37)] {
        let targets = Targets {
            target: texture(
                &device,
                sides,
                TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
            ),
            groups: texture(
                &device,
                sides,
                TextureUsage::RENDER_TARGET
                    | TextureUsage::SHADER_READ
                    | TextureUsage::SHADER_WRITE,
            ),
        };
        let mut placeholders = 0;
        for &access in &accesses {
            for shape in SHAPES {
                // Every corner lies within the first 61 columns and 23 rows, so a pixel past them
                // holds what the last of them holds.
                let reach: Vec<u32> = (0..23)
                    .flat_map(|y| (0..61).map(move |x| expected(shape, (x, y))))
                    .collect();
                let want: Vec<u32> = (0..sides.1)
                    .flat_map(|y| (0..sides.0).map(move |x| (x.min(60), y.min(22))))
                    .map(|(x, y)| reach[(y * 61 + x) as usize])
                    .collect();
                let descriptor = descriptor(&library, shape, access);
                let pipeline = device
                    .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
                    .unwrap();
                for run in 0..RUNS {
                    let counted = device.work_counts();
                    let (gpu, values) = pass(&device, &targets, &pipeline, (shape, access), sides);
                    let counts = device.work_counts();
                    placeholders += counts.storage_placeholder_attachments
                        - counted.storage_placeholder_attachments;
                    let pushed = counts.draws_pushing_set_zero - counted.draws_pushing_set_zero;
                    let buffer = access == Access::RogBuffer;
                    assert!(
                        !(buffer && buffer_pushes) || pushed > 0,
                        "{}: a raster order group buffer's draw bound set 0",
                        device.name()
                    );
                    assert!(
                        !(!buffer && others_bound) || pushed == 0,
                        "{}: a {access:?} draw pushed set 0 where pushed sets lose their \
                         buffers' sizes",
                        device.name()
                    );
                    assert_eq!(
                        counts.instance_splits_interlocked_writes,
                        counted.instance_splits_interlocked_writes,
                        "{}: a pass of one draw split",
                        device.name()
                    );
                    let wrong: Vec<bool> = values.iter().zip(&want).map(|(v, w)| v != w).collect();
                    let count = wrong.iter().filter(|&&w| w).count();
                    let groups = wrong_groups(&wrong, sides);
                    let whole = groups.values().filter(|&&whole| whole).count();
                    let named: Vec<_> = groups.keys().take(NAMED).collect();
                    eprintln!(
                        "PRIMITIVE-ORDER on {}: {}x{} {access:?} {shape:?}, pass {run}: {count} \
                         of {} texels wrong, in {} 8x4 groups ({whole} wrong whole), first \
                         {named:?}; set 0 pushed by {pushed} draw(s); GPU {gpu:.2} ms",
                        device.name(),
                        sides.0,
                        sides.1,
                        values.len(),
                        groups.len(),
                    );
                    if count > 0 {
                        wrong_passes.push((sides, access, shape, run, count, groups.len(), whole));
                    }
                }
            }
        }
        eprintln!(
            "PRIMITIVE-ORDER on {}: {}x{}: {placeholders} framebuffer attachment(s) took a \
             storage placeholder",
            device.name(),
            sides.0,
            sides.1,
        );
        assert!(
            !interlocked || placeholders > 0,
            "{}: {}x{}: no pass left the target out for the interlock",
            device.name(),
            sides.0,
            sides.1,
        );
    }
    assert!(
        wrong_passes.is_empty(),
        "{}: passes whose primitives did not read each other's writes (sides, access, shape, \
         pass, texels, groups, whole groups): {wrong_passes:?}",
        device.name()
    );
    drop(device);
    layer_reported_nothing_at_all(&messages);
}

/// The kind [`a_bound_set_of_interlocked_buffers_is_announced_once`] takes where the device's
/// driver orders a buffer in a raster order group only in a pushed set 0
/// (`RuntimeDriverDefect::InterlockedBoundBuffersUnordered`), and expects absent elsewhere.
pub(super) const BOUND_INTERLOCKS: [Degradation; 1] = [Degradation::InterlockedBufferOrder];

/// On a device without the push command (`VK_KHR_push_descriptor` withheld, on a device of the
/// instance's Vulkan 1.2), a render pipeline state whose fragment function writes a buffer in
/// raster order group 0 (`ipo_rog_buffer`) binds its set 0. Where its driver orders such a buffer
/// only pushed, its making announces that once ([`BOUND_INTERLOCKS`]) and a second state of the
/// same descriptor announces nothing again; elsewhere neither announces it. Its draw runs either
/// way: one pass of the fan, its words read back (in order or not, the device's answer).
#[test]
fn a_bound_set_of_interlocked_buffers_is_announced_once() {
    let withheld = |options: &mut metal2vulkan_runtime::DeviceOptions| {
        options
            .withheld_extensions
            .insert(VkExtension::KhrPushDescriptor);
    };
    on_device_with(withheld, |device, messages| {
        let ll = engine_dir().join("ipo_primitive_order.ll");
        let library = device
            .new_library_with_air(&llvm_assemble(&ll).unwrap())
            .unwrap();
        if !orders_groups(device, &library) {
            return;
        }
        assert!(!pushes(device).0, "withheld, the device pushes");
        let defects = &device.properties().driver_defects;
        let unordered = defects.contains(&RuntimeDriverDefect::InterlockedBoundBuffersUnordered);
        let shape = SHAPES[0];
        let descriptor = descriptor(&library, shape, Access::RogBuffer);
        let kinds = made_render_kinds(device, &descriptor);
        assert_eq!(
            kinds.contains(&Degradation::InterlockedBufferOrder),
            unordered,
            "{}: the predicted kinds {kinds:?}",
            device.name()
        );
        messages.take_notices();
        let pipeline = device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        let kinds: Vec<Degradation> = kinds.into_iter().collect();
        assert_once(&messages.take_notices(), &kinds);
        drop(
            device
                .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
                .unwrap(),
        );
        assert_once(&messages.take_notices(), &[]);
        let groups = texture(
            device,
            (64, 36),
            TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
        );
        let targets = Targets {
            target: texture(
                device,
                (64, 36),
                TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
            ),
            groups,
        };
        let counted = device.work_counts();
        let (_, values) = pass(
            device,
            &targets,
            &pipeline,
            (shape, Access::RogBuffer),
            (64, 36),
        );
        let counts = device.work_counts();
        assert_eq!(
            counts.draws_pushing_set_zero,
            counted.draws_pushing_set_zero
        );
        let want = expected(shape, (0, 0));
        let wrong = values.iter().filter(|&&v| v != want).count();
        println!(
            "BOUND-INTERLOCKS on {}: announced {unordered}; {wrong} of {} words wrong",
            device.name(),
            values.len()
        );
    });
}
