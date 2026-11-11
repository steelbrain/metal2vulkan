//! The order of framebuffer fetches across the primitives of one draw, on the device the cfg
//! selects, under the validation layer. Metal orders a fragment function's read of its color
//! attachment after every earlier fragment of the pixel, those of its own draw's earlier
//! primitives included. In the pixel interlock (`metal2vulkan_runtime`'s `framebuffer_fetch`) the
//! runtime records nothing between the primitives of a draw: Vulkan orders them by the interlock
//! alone, its ordered interlock's operations happening "before interlocked operations in fragment
//! shader invocations that execute later in rasterization order and cover at least one sample in
//! the same pixel" (Fragment Operations, "Interlocked Operations"), rasterization order following
//! the order of a draw's primitives ("Primitive Order"). This test is that measurement, the
//! within-draw counterpart of `interlocked_draw_order`'s.
//!
//! Each pass clears an `R32Uint` target to [`CLEAR`] and makes one draw of [`PRIMITIVES`]
//! overlapping primitives (`ipo_primitive_order`), each fragment `ipo_mix`'s: `3c + 1` or `5c + 3`
//! by the parity of the bits set in its primitive's number. The draw takes one of four [`Shape`]s:
//! `ipo_fan`'s fullscreen triangles or `ipo_steps`' rectangles (each from a corner that steps
//! with its number, [`corner`], to the target's right and bottom edges, two triangles sharing a
//! diagonal), each as one instance of every primitive or as one instance per primitive. The maps
//! do not commute, so a texel holds [`expected`] only where each fragment read what the fragment
//! of the primitive before it wrote there. [`RUNS`] passes of each shape over 1920x1080, 64x36
//! and 61x37, each read back whole. Every texel of every pass must hold its composition; the
//! wrong ones are counted and printed per pass, with the 8x4 groups (8 columns by 4 rows from the
//! target's top left, the last ones cut by its edges) they fall in, how many of those are wrong
//! in every texel, and the first of them; and per target, the storage forms it took
//! (`WorkCounts::storage_forms_taken`), 1 where the fetch ran in the interlock's storage form.

use super::framebuffer_fetch::fetches;
use super::{
    at_plan, layer_reported_nothing_at_all, read_buffer, validated_device_with, Messages,
    ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    BlitOption, ClearColor, Device, Library, LoadAction, Origin, PipelineOptions,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, Size, StoreAction, Texture,
    TextureDescriptor,
};
use std::collections::BTreeMap;

/// The primitives of a draw: triangles for `ipo_fan`, rectangles for `ipo_steps`.
const PRIMITIVES: u32 = 64;

/// The passes of each shape and size.
const RUNS: usize = 8;

/// The target's clear value.
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

/// `shape`'s pipeline, `ipo_mix` into an `R32Uint` attachment.
fn pipeline(device: &Device, library: &Library, shape: Shape) -> RenderPipelineState {
    let function = |name: &str| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut descriptor = RenderPipelineDescriptor::new(function(shape.vertex()));
    descriptor.fragment = Some(function("ipo_mix"));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::R32Uint));
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A private `R32Uint` render target of `sides`.
fn target(device: &Device, (width, height): (u64, u64)) -> Texture {
    let mut d = TextureDescriptor::texture_2d(PixelFormat::R32Uint, width, height, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::private();
    device.new_texture(&d).unwrap()
}

/// One pass over `target`: its clear, then one draw of `shape`; and its GPU time in milliseconds.
fn draw(device: &Device, target: &Texture, pipeline: &RenderPipelineState, shape: Shape) -> f64 {
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut descriptor = RenderPassDescriptor::new();
    let color = &mut descriptor.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Clear;
    color.clear_color = ClearColor::new(f64::from(CLEAR), 0.0, 0.0, 0.0);
    color.store_action = StoreAction::Store;
    let size = [target.width() as f32, target.height() as f32];
    let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
    encoder.set_render_pipeline_state(pipeline).unwrap();
    encoder
        .set_vertex_bytes(&size.map(f32::to_le_bytes).concat(), 0)
        .unwrap();
    let (vertices, instances) = match shape.instanced {
        true => (shape.vertices(), PRIMITIVES),
        false => (shape.vertices() * PRIMITIVES, 1),
    };
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, vertices, instances, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    (command_buffer.gpu_end_time().unwrap() - command_buffer.gpu_start_time().unwrap()) * 1e3
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
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    read_buffer(&buffer, 0, bytes)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect()
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

/// Every pass's texels hold the composition of its draw's primitives in order (the module docs).
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
    eprintln!(
        "PRIMITIVE-ORDER on {}: rasterization-order color access {}, pixel interlock {}",
        device.name(),
        fragment.rasterization_order_color_access,
        fragment.pixel_interlock,
    );
    let mut wrong_passes = Vec::new();
    for sides in [(1920u64, 1080u64), (64, 36), (61, 37)] {
        let target = target(&device, sides);
        let mut forms = 0;
        for shape in SHAPES {
            // Every corner lies within the first 61 columns and 23 rows, so a texel past them
            // holds what the last of them holds.
            let reach: Vec<u32> = (0..23)
                .flat_map(|y| (0..61).map(move |x| expected(shape, (x, y))))
                .collect();
            let want: Vec<u32> = (0..sides.1)
                .flat_map(|y| (0..sides.0).map(move |x| (x.min(60), y.min(22))))
                .map(|(x, y)| reach[(y * 61 + x) as usize])
                .collect();
            let pipeline = pipeline(&device, &library, shape);
            for run in 0..RUNS {
                let counted = device.work_counts();
                let gpu = draw(&device, &target, &pipeline, shape);
                let texels = texels(&device, &target, sides);
                let counts = device.work_counts();
                forms += counts.storage_forms_taken - counted.storage_forms_taken;
                assert_eq!(
                    counts.instance_splits_interlocked_writes,
                    counted.instance_splits_interlocked_writes,
                    "{}: a pass of one draw split",
                    device.name()
                );
                let wrong: Vec<bool> = texels.iter().zip(&want).map(|(t, w)| t != w).collect();
                let count = wrong.iter().filter(|&&w| w).count();
                let groups = wrong_groups(&wrong, sides);
                let whole = groups.values().filter(|&&whole| whole).count();
                let named: Vec<_> = groups.keys().take(NAMED).collect();
                eprintln!(
                    "PRIMITIVE-ORDER on {}: {}x{} {shape:?}, pass {run}: {count} of {} texels \
                     wrong, in {} 8x4 groups ({whole} wrong whole), first {named:?}; GPU {gpu:.2} \
                     ms",
                    device.name(),
                    sides.0,
                    sides.1,
                    texels.len(),
                    groups.len(),
                );
                if count > 0 {
                    wrong_passes.push((sides, shape, run, count, groups.len(), whole));
                }
            }
        }
        eprintln!(
            "PRIMITIVE-ORDER on {}: {}x{} took {forms} storage form(s)",
            device.name(),
            sides.0,
            sides.1,
        );
    }
    assert!(
        wrong_passes.is_empty(),
        "{}: passes whose primitives did not read each other's writes (sides, shape, pass, \
         texels, groups, whole groups): {wrong_passes:?}",
        device.name()
    );
    drop(device);
    layer_reported_nothing_at_all(&messages);
}
