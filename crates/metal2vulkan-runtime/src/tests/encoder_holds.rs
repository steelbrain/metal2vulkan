//! A render, compute or resource state encoder's buffers, textures, heaps and states held until
//! its command buffer completed: `blit_holds.rs`'s drop-after-commit legs for the other encoder
//! kinds. Each leg encodes a command that reads or writes a no-copy buffer over host pages, by one
//! way a command reaches a buffer (a binding, an index or indirect buffer, an attachment or a
//! texture over the buffer, an address held in memory that `use_resource` or `use_heap` makes
//! resident, an indirect command buffer's command, a sparse texture's indirect mapping update),
//! before busy work is committed, commits it after that work, and releases the no-copy buffer
//! and every other object the command uses: once at the commit and once more where the command
//! buffer was scheduled. While the command buffer has not completed, the buffer's deallocator
//! must not have run and every object the command buffer holds must live; once it completed, the
//! command's bytes must be where they belong. Memory freed with its command in flight gives its
//! device address to the next allocation, which the command's late writes corrupt silently.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are the owned fixtures
//! `validation/fixtures/public/examples/runtime_examples.ll`, `examples/icb_direct_arguments.ll`,
//! `engine/tob_renders.ll` and `engine/ob2_loaded_pointer.ll` (`llvm-as`).

use super::blit_holds::{
    busy, device, over_host, until, Host, Scratch, BYTES, FILLED, SOURCE, WAIT,
};
use crate::error::RuntimeGap;
use crate::{
    BlitOption, Buffer, ClearColor, CommandBuffer, CommandBufferStatus, CommandQueue,
    ComputePipelineState, Device, Error, HeapDescriptor, HeapType, IndirectCommandBufferDescriptor,
    Library, LoadAction, Origin, PipelineOptions, Region, RenderPassDescriptor,
    RenderPipelineState, ResourceOptions, ResourceRef, Size, StoreAction, Texture,
    TextureDescriptor,
};
use metal2vulkan::bind::TileMapping;
use metal2vulkan::error::Feature;
use metal2vulkan::metal::{
    IndexType, IndirectCommandType, PixelFormat, PrimitiveType, ResourceUsage, TextureType,
    TextureUsage,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource,
};
use std::any::Any;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// The words of the no-copy buffer: a `float` or a `uint` each.
const WORDS: u64 = BYTES as u64 / 4;

/// The side of a draw's target, and the color it draws: `(0.2, 0.4, 0.6, 1)`, whose `RGBA8Unorm`
/// bytes are exact.
const SIDE: u64 = 64;
const COLOR: [f32; 4] = [0.2, 0.4, 0.6, 1.0];
const PACKED: u32 = u32::from_le_bytes([51, 102, 153, 255]);

/// A texture over the no-copy buffer: `RGBA8Unorm`, rows of [`ROW`] bytes, its whole bytes.
const ROW: u64 = 2048;
const OVER_WIDTH: u64 = ROW / 4;
const OVER_HEIGHT: u64 = BYTES as u64 / ROW;

/// The byte offset of the colors a vertex leg reads after its positions.
const COLORS_AT: u64 = 256;

/// The threads of a threadgroup of a dispatch over the buffer's words.
const GROUP: u64 = 64;

/// What a leg's command does with the no-copy buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Leg {
    /// A draw whose vertex function reads its positions and colors from it, bound
    /// (`set_vertex_buffer`).
    VertexBuffers,
    /// The same, bound after a draw of the encoder that read them from another buffer bound at
    /// those indices: the second draw's retain of the vertex table takes the no-copy buffer, which
    /// the encoder did not hold as it was bound, where a retain of the table after calls that
    /// bound only objects the encoder held looks at none (`argument_table::retain_into`).
    VertexBuffersRebound,
    /// A draw whose fragment function reads its color from it, bound (`set_fragment_buffer`).
    FragmentBuffer,
    /// An indexed draw whose indices it holds.
    IndexBuffer,
    /// An indirect draw whose arguments it holds.
    IndirectDraw,
    /// A draw into a texture over it, the pass's color attachment.
    Attachment,
    /// A dispatch that reads it, bound.
    KernelReads,
    /// A dispatch that writes it, bound.
    KernelWrites,
    /// An indirect dispatch whose arguments it holds.
    IndirectDispatch,
    /// A dispatch that writes a texture over it.
    TextureWrite,
    /// A dispatch that reads and writes it through an address held in memory, made resident with
    /// `use_resource`.
    UsedResource,
    /// A dispatch that writes it, bound, from a heap's buffer that it reads through an address
    /// held in memory, the heap made resident with `use_heap`: the heap is held, not the buffer
    /// placed in it, as Metal's command buffer holds what its encoders were given. A placed buffer
    /// the program releases gives its range back to the heap, which a buffer placed later may take
    /// while the command runs, on Metal as here.
    UsedHeap,
    /// An indirect command buffer's dispatch that writes it, made resident with `use_resource`.
    IndirectCommands,
    /// A sparse texture's indirect mapping update whose arguments it holds, which map the texture.
    SparseMapping,
    /// The same, its arguments mapping nothing: the texture and the buffer are held still.
    SparseMappingNothing,
}

/// Where the program releases what the command uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Release {
    /// As `commit` returns.
    Committed,
    /// Once `wait_until_scheduled` returned.
    Scheduled,
}

/// A check of the bytes once the command buffer completed.
type Check = Box<dyn FnOnce(&Host)>;

/// Whether an object a leg released still lives.
type Lives = Box<dyn Fn() -> bool>;

/// What a leg encoded: the objects the program releases with the no-copy buffer, those of them
/// the command buffer must hold, by name, and the check once it completed.
struct Encoded {
    released: Vec<Box<dyn Any>>,
    watched: Vec<(&'static str, Lives)>,
    check: Check,
}

/// Watch the object `held` is a reference to (a buffer's, texture's or render state's `held`).
fn watch(name: &'static str, held: Arc<dyn Any + Send + Sync>) -> (&'static str, Lives) {
    let weak = Arc::downgrade(&held);
    (name, Box::new(move || weak.strong_count() > 0))
}

/// Watch a compute state.
fn watch_state(state: &ComputePipelineState) -> (&'static str, Lives) {
    let weak = state.downgrade();
    (
        "the compute state",
        Box::new(move || weak.upgrade().is_some()),
    )
}

/// Watch a render state.
fn watch_render_state(state: &RenderPipelineState) -> (&'static str, Lives) {
    let weak = state.downgrade().expect("a drawing state");
    (
        "the render state",
        Box::new(move || weak.upgrade().is_some()),
    )
}

/// The little-endian bytes of `values`.
pub(super) fn bytes_of<T: Copy, const N: usize>(
    values: &[T],
    to_bytes: impl Fn(T) -> [u8; N],
) -> Vec<u8> {
    values.iter().flat_map(|&value| to_bytes(value)).collect()
}

/// The bits of `values`.
pub(super) fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// The words of `buffer`, a host-visible buffer no command writes now.
pub(super) fn words_of(buffer: &Buffer) -> Vec<u32> {
    let contents = buffer.contents().unwrap();
    // SAFETY: the buffer's `length` bytes, which no command buffer writes now.
    let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize) };
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect()
}

/// The library of the fixture at `path` under `validation/fixtures/public/`.
pub(super) fn library(device: &Device, path: &str) -> Library {
    let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public")
        .join(path);
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let library = metal2vulkan::library::Library::from_air(&bitcode).unwrap();
    device.new_library_with_compile_library(library)
}

/// A state of `kernel` of the fixture at `path`, made for indirect command buffers where `icb`.
fn compute_state(device: &Device, path: &str, kernel: &str, icb: bool) -> ComputePipelineState {
    let function = library(device, path)
        .new_function_with_name(kernel)
        .unwrap();
    let mut descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.support_indirect_command_buffers = icb;
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A state that draws into `RGBA8Unorm` with the functions `vertex` and `fragment` of the
/// fixture at `path`.
pub(super) fn render_state(
    device: &Device,
    path: &str,
    vertex: &str,
    fragment: &str,
) -> RenderPipelineState {
    let library = library(device, path);
    let vertex = library.new_function_with_name(vertex).unwrap();
    let fragment = library.new_function_with_name(fragment).unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
    descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A pass that clears `target` to zero and stores what it draws.
pub(super) fn pass(target: &Texture) -> RenderPassDescriptor {
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Clear;
    color.store_action = StoreAction::Store;
    color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
    pass
}

/// An `RGBA8Unorm` texture over the no-copy buffer's bytes of `usage`; `None`, printed, where the
/// device makes none.
fn over_buffer(no_copy: &Buffer, usage: TextureUsage) -> Option<Texture> {
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, OVER_WIDTH, OVER_HEIGHT, false);
    descriptor.usage = usage;
    descriptor.resource_options = ResourceOptions::default();
    match no_copy.new_texture(&descriptor, 0, ROW) {
        Ok(texture) => Some(texture),
        Err(error) => {
            eprintln!("ENCODER-HOLDS: no texture over the no-copy buffer: {error}");
            None
        }
    }
}

/// A `Shared` buffer of the device's holding `words`.
fn shared_words(device: &Device, words: &[u32]) -> Buffer {
    device
        .new_buffer_with_bytes(
            &bytes_of(words, u32::to_le_bytes),
            ResourceOptions::shared(),
        )
        .unwrap()
}

/// Encode `leg`'s command over `no_copy`, the host pages' buffer, in `command_buffer`, the pages
/// holding its inputs first; `None`, printed, where the device has not what the leg needs.
fn encode(
    leg: Leg,
    device: &Device,
    queue: &CommandQueue,
    command_buffer: &CommandBuffer,
    host: &Host,
    no_copy: &Buffer,
) -> Option<Encoded> {
    match leg {
        Leg::VertexBuffers
        | Leg::VertexBuffersRebound
        | Leg::FragmentBuffer
        | Leg::IndexBuffer
        | Leg::IndirectDraw => Some(draw(leg, device, command_buffer, host, no_copy)),
        Leg::Attachment => attachment(device, command_buffer, host, no_copy),
        Leg::KernelReads | Leg::KernelWrites | Leg::IndirectDispatch => {
            Some(dispatch(leg, device, command_buffer, host, no_copy))
        }
        Leg::TextureWrite => texture_write(device, command_buffer, host, no_copy),
        Leg::UsedResource => Some(used_resource(device, command_buffer, host, no_copy)),
        Leg::UsedHeap => Some(used_heap(device, queue, command_buffer, host, no_copy)),
        Leg::IndirectCommands => Some(indirect_commands(device, command_buffer, host, no_copy)),
        Leg::SparseMapping | Leg::SparseMappingNothing => {
            sparse_mapping(leg, device, command_buffer, host, no_copy)
        }
    }
}

/// A draw leg: a rectangle (or a triangle) over a `Private` target in the color, read from the
/// no-copy buffer as the leg says, the target copied into a `Shared` buffer after the pass.
fn draw(
    leg: Leg,
    device: &Device,
    command_buffer: &CommandBuffer,
    host: &Host,
    no_copy: &Buffer,
) -> Encoded {
    let state = match leg {
        Leg::VertexBuffers | Leg::VertexBuffersRebound => render_state(
            device,
            "examples/runtime_examples.ll",
            "colored_vertex",
            "colored_fragment",
        ),
        _ => render_state(device, "engine/tob_renders.ll", "tob_rect", "tob_solid"),
    };
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    let target = device.new_texture(&descriptor).unwrap();
    let back = device
        .new_buffer(SIDE * SIDE * 4, ResourceOptions::shared())
        .unwrap();
    host.fill(0);
    let mut encoder = command_buffer
        .render_command_encoder(&pass(&target))
        .unwrap();
    encoder.set_render_pipeline_state(&state).unwrap();
    let corners = bytes_of(&[-1.0f32, -1.0, 1.0, 1.0], f32::to_le_bytes);
    let color = bytes_of(&COLOR, f32::to_le_bytes);
    let mut watched = Vec::new();
    let mut released: Vec<Box<dyn Any>> = Vec::new();
    let mut counts_before_draw = None;
    match leg {
        Leg::VertexBuffers | Leg::VertexBuffersRebound => {
            // A triangle over the target: its positions, then each vertex's color.
            let triangle = bits(&[-1.0, -1.0, 3.0, -1.0, -1.0, 3.0]);
            host.set(0, &triangle);
            host.set(COLORS_AT as usize / 4, &bits(&COLOR.repeat(3)));
            if leg == Leg::VertexBuffersRebound {
                // The same triangle from another buffer first, then the no-copy buffer's at the
                // same indices, then again: the second draw's retain takes the no-copy buffer,
                // and the third's looks at nothing, the encoder holding it as it was bound.
                let mut words = vec![0; (COLORS_AT / 4) as usize];
                words[..triangle.len()].copy_from_slice(&triangle);
                words.extend(bits(&COLOR.repeat(3)));
                let other = shared_words(device, &words);
                encoder.set_vertex_buffer(Some(&other), 0, 0).unwrap();
                encoder
                    .set_vertex_buffer(Some(&other), COLORS_AT, 1)
                    .unwrap();
                encoder
                    .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                    .unwrap();
                watched.push(watch("the first draw's buffer", other.held()));
                released.push(Box::new(other));
            }
            let before = device.work_counts();
            encoder.set_vertex_buffer(Some(no_copy), 0, 0).unwrap();
            encoder
                .set_vertex_buffer(Some(no_copy), COLORS_AT, 1)
                .unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
            if leg == Leg::VertexBuffersRebound {
                encoder.set_vertex_buffer(Some(no_copy), 0, 0).unwrap();
                encoder
                    .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                    .unwrap();
            }
            counts_before_draw = Some(before);
        }
        Leg::FragmentBuffer => {
            host.set(0, &bits(&COLOR));
            encoder.set_vertex_bytes(&corners, 0).unwrap();
            encoder.set_fragment_buffer(Some(no_copy), 0, 0).unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                .unwrap();
        }
        Leg::IndexBuffer => {
            host.set(0, &[1 << 16, 2 | 3 << 16, 4 | 5 << 16]);
            encoder.set_vertex_bytes(&corners, 0).unwrap();
            encoder.set_fragment_bytes(&color, 0).unwrap();
            encoder
                .draw_indexed_primitives(
                    PrimitiveType::Triangle,
                    6,
                    IndexType::UInt16,
                    no_copy,
                    0,
                    1,
                    0,
                    0,
                )
                .unwrap();
        }
        _ => {
            // `MTLDrawPrimitivesIndirectArguments`: six vertices, one instance.
            host.set(0, &[6, 1, 0, 0]);
            encoder.set_vertex_bytes(&corners, 0).unwrap();
            encoder.set_fragment_bytes(&color, 0).unwrap();
            encoder
                .draw_primitives_indirect(PrimitiveType::Triangle, no_copy, 0)
                .unwrap();
        }
    }
    encoder.end_encoding().unwrap();
    // The draw's retains took the no-copy buffer, once for its two slots, and nothing else (the
    // fragment table binds nothing); the draw after it, which bound it again, looked at nothing.
    if let Some(before) = counts_before_draw {
        let after = device.work_counts();
        let grew = |count: fn(&crate::WorkCounts) -> u64| count(&after) - count(&before);
        let held = grew(|c| c.argument_tables_held_at_bind);
        assert_eq!(
            held,
            u64::from(leg == Leg::VertexBuffersRebound),
            "{leg:?}: retains of tables whose objects the encoder held at bind"
        );
        let retained = (
            grew(|c| c.argument_tables_retained),
            grew(|c| c.argument_objects_retained),
        );
        assert_eq!(retained, (1, 1), "{leg:?}: tables and objects retained");
    }
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        &target,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(SIDE, SIDE, 1),
        &back,
        0,
        SIDE * 4,
        SIDE * SIDE * 4,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    watched.extend([
        watch("the target", target.held()),
        watch_render_state(&state),
    ]);
    released.extend([Box::new(target) as Box<dyn Any>, Box::new(state)]);
    Encoded {
        watched,
        released,
        check: Box::new(move |_| {
            assert!(
                words_of(&back).iter().all(|&word| word == PACKED),
                "{leg:?}: the draw's color is not in the target"
            );
        }),
    }
}

/// The attachment leg: a rectangle in the color over a texture over the no-copy buffer.
fn attachment(
    device: &Device,
    command_buffer: &CommandBuffer,
    host: &Host,
    no_copy: &Buffer,
) -> Option<Encoded> {
    let target = over_buffer(no_copy, TextureUsage::RENDER_TARGET)?;
    let state = render_state(device, "engine/tob_renders.ll", "tob_rect", "tob_solid");
    host.fill(SOURCE);
    let mut encoder = command_buffer
        .render_command_encoder(&pass(&target))
        .unwrap();
    encoder.set_render_pipeline_state(&state).unwrap();
    let corners = bytes_of(&[-1.0f32, -1.0, 1.0, 1.0], f32::to_le_bytes);
    encoder.set_vertex_bytes(&corners, 0).unwrap();
    encoder
        .set_fragment_bytes(&bytes_of(&COLOR, f32::to_le_bytes), 0)
        .unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    Some(Encoded {
        watched: vec![
            watch("the texture over the buffer", target.held()),
            watch_render_state(&state),
        ],
        released: vec![Box::new(target), Box::new(state)],
        check: Box::new(|host| {
            assert!(
                host.holds(PACKED),
                "the draw's color is not in the pages under the attachment"
            );
        }),
    })
}

/// A dispatch leg of `scale_and_offset` (`output[i] = input[i] * scale + i`) over the buffer's
/// words, scale 1: reading the no-copy buffer (its words 1), writing it (from zeros), or reading
/// its threadgroup counts from it.
fn dispatch(
    leg: Leg,
    device: &Device,
    command_buffer: &CommandBuffer,
    host: &Host,
    no_copy: &Buffer,
) -> Encoded {
    let state = compute_state(
        device,
        "examples/runtime_examples.ll",
        "scale_and_offset",
        false,
    );
    // The input where it is not the no-copy buffer, and the output where that is not.
    let zeros = (leg != Leg::KernelReads).then(|| shared_words(device, &vec![0; WORDS as usize]));
    let output = (leg != Leg::KernelWrites).then(|| {
        device
            .new_buffer(BYTES as u64, ResourceOptions::shared())
            .unwrap()
    });
    match leg {
        Leg::KernelReads => host.fill(1.0f32.to_bits()),
        Leg::KernelWrites => host.fill(SOURCE),
        _ => {
            host.fill(0);
            host.set(0, &[(WORDS / GROUP) as u32, 1, 1]);
        }
    }
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    encoder
        .set_buffer(Some(zeros.as_ref().unwrap_or(no_copy)), 0, 0)
        .unwrap();
    encoder
        .set_buffer(Some(output.as_ref().unwrap_or(no_copy)), 0, 1)
        .unwrap();
    encoder.set_bytes(&1.0f32.to_le_bytes(), 2).unwrap();
    let group = Size::new(GROUP, 1, 1);
    match leg {
        Leg::IndirectDispatch => encoder.dispatch_threadgroups_indirect(no_copy, 0, group),
        _ => encoder.dispatch_threads(Size::new(WORDS, 1, 1), group),
    }
    .unwrap();
    encoder.end_encoding().unwrap();
    let base = match leg {
        Leg::KernelReads => 1.0,
        _ => 0.0,
    };
    let expected = move |words: &[u32]| {
        words
            .iter()
            .enumerate()
            .all(|(i, &word)| f32::from_bits(word) == base + i as f32)
    };
    let mut watched = vec![watch_state(&state)];
    let mut released: Vec<Box<dyn Any>> = vec![Box::new(state)];
    if let Some(zeros) = zeros {
        watched.push(watch("the input buffer", zeros.held()));
        released.push(Box::new(zeros));
    }
    Encoded {
        watched,
        released,
        check: Box::new(move |host| match output {
            None => assert!(
                expected(host.words()),
                "the dispatch's words are not in the pages"
            ),
            Some(output) => assert!(
                expected(&words_of(&output)),
                "{leg:?}: the dispatch did not read the pages"
            ),
        }),
    }
}

/// The texture write leg: `invert_texels` from a texture of zeros into a texture over the no-copy
/// buffer, which takes `(1, 1, 1, 0)`.
fn texture_write(
    device: &Device,
    command_buffer: &CommandBuffer,
    host: &Host,
    no_copy: &Buffer,
) -> Option<Encoded> {
    let destination = over_buffer(no_copy, TextureUsage::SHADER_WRITE)?;
    let library = library(device, "examples/runtime_examples.ll");
    let function = library.new_function_with_name("invert_texels").unwrap();
    let state = device
        .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
        .unwrap();
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, OVER_WIDTH, OVER_HEIGHT, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    let source = device.new_texture(&descriptor).unwrap();
    source
        .replace_region(
            Region::new_2d(0, 0, OVER_WIDTH, OVER_HEIGHT),
            0,
            0,
            &vec![0; BYTES],
            ROW,
            0,
        )
        .unwrap();
    host.fill(SOURCE);
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    encoder.set_texture(Some(&source), 0).unwrap();
    encoder.set_texture(Some(&destination), 1).unwrap();
    encoder
        .dispatch_threads(Size::new(OVER_WIDTH, OVER_HEIGHT, 1), Size::new(8, 8, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    Some(Encoded {
        watched: vec![
            watch_state(&state),
            watch("the source texture", source.held()),
            watch("the texture over the buffer", destination.held()),
        ],
        released: vec![Box::new(state), Box::new(source), Box::new(destination)],
        check: Box::new(|host| {
            assert!(
                host.holds(u32::from_le_bytes([255, 255, 255, 0])),
                "the dispatch's texels are not in the pages under the texture"
            );
        }),
    })
}

/// The indices `ob2_loaded_pointer`'s eight threads load from (`0` to `7`) and store to (`8` to
/// `15`), as bytes.
fn loaded_pointer_indices() -> Vec<u8> {
    bytes_of(&(0..16u32).collect::<Vec<_>>(), u32::to_le_bytes)
}

/// The resident resource leg: `ob2_loaded_pointer` through the no-copy buffer's address, which
/// its threads load words 0 to 7 from and store `0xB0000000 | t` to words 8 to 15 of.
fn used_resource(
    device: &Device,
    command_buffer: &CommandBuffer,
    host: &Host,
    no_copy: &Buffer,
) -> Encoded {
    let state = compute_state(
        device,
        "engine/ob2_loaded_pointer.ll",
        "ob2_loaded_pointer",
        false,
    );
    let out = device.new_buffer(32, ResourceOptions::shared()).unwrap();
    host.fill(SOURCE);
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    encoder
        .use_resource(
            ResourceRef::Buffer(no_copy),
            ResourceUsage::READ | ResourceUsage::WRITE,
        )
        .unwrap();
    encoder.set_bytes(&loaded_pointer_indices(), 0).unwrap();
    encoder
        .set_bytes(&no_copy.gpu_address().0.to_le_bytes(), 1)
        .unwrap();
    encoder.set_buffer(Some(&out), 0, 2).unwrap();
    encoder
        .dispatch_threads(Size::new(8, 1, 1), Size::new(8, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    Encoded {
        watched: vec![watch_state(&state)],
        released: vec![Box::new(state)],
        check: Box::new(move |host| {
            let stored: Vec<u32> = (0..8).map(|t| 0xB000_0000 | t).collect();
            assert_eq!(
                &host.words()[..16],
                [[SOURCE; 8].as_slice(), &stored].concat(),
                "the stores through the address are not in the pages"
            );
            assert_eq!(
                words_of(&out),
                [SOURCE; 8],
                "the loads through the address did not read the pages"
            );
        }),
    }
}

/// The resident heap leg: `ob2_loaded_pointer` through the address of a heap's buffer that holds
/// [`SOURCE`], the heap made resident with `use_heap`; its threads load the buffer's words 0 to
/// 7 into the no-copy buffer's.
fn used_heap(
    device: &Device,
    queue: &CommandQueue,
    command_buffer: &CommandBuffer,
    host: &Host,
    no_copy: &Buffer,
) -> Encoded {
    let state = compute_state(
        device,
        "engine/ob2_loaded_pointer.ll",
        "ob2_loaded_pointer",
        false,
    );
    let heap = device
        .new_heap(&HeapDescriptor::new(
            HeapType::Automatic,
            1 << 20,
            ResourceOptions::private(),
        ))
        .unwrap();
    let placed = heap
        .new_buffer(64 << 10, ResourceOptions::private())
        .unwrap();
    let filling = queue.command_buffer().unwrap();
    let mut blit = filling.blit_command_encoder().unwrap();
    blit.fill_buffer(&placed, 0..placed.length(), SOURCE as u8)
        .unwrap();
    blit.end_encoding().unwrap();
    filling.commit().unwrap();
    filling.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(filling.error(), None);
    host.fill(0);
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    encoder.use_heap(&heap).unwrap();
    encoder.set_bytes(&loaded_pointer_indices(), 0).unwrap();
    encoder
        .set_bytes(&placed.gpu_address().0.to_le_bytes(), 1)
        .unwrap();
    encoder.set_buffer(Some(no_copy), 0, 2).unwrap();
    encoder
        .dispatch_threads(Size::new(8, 1, 1), Size::new(8, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    let weak = heap.downgrade();
    Encoded {
        watched: vec![
            watch_state(&state),
            ("the heap", Box::new(move || weak.upgrade().is_some())),
        ],
        released: vec![Box::new(state), Box::new(placed), Box::new(heap)],
        check: Box::new(|host| {
            assert_eq!(
                &host.words()[..9],
                [[SOURCE; 8].as_slice(), &[0]].concat(),
                "the heap's words are not in the pages"
            );
        }),
    }
}

/// The indirect command leg: an indirect command buffer's `scale_and_offset` over the buffer's
/// words from zeros, scale 1, the buffers made resident with `use_resource`. The program keeps
/// the state, which Metal's command does not hold.
fn indirect_commands(
    device: &Device,
    command_buffer: &CommandBuffer,
    host: &Host,
    no_copy: &Buffer,
) -> Encoded {
    let state = compute_state(
        device,
        "examples/runtime_examples.ll",
        "scale_and_offset",
        true,
    );
    let zeros = shared_words(device, &vec![0; WORDS as usize]);
    let scale = shared_words(device, &[1.0f32.to_bits()]);
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH_THREADS;
    descriptor.max_kernel_buffer_bind_count = 3;
    let icb = device
        .new_indirect_command_buffer(&descriptor, 1, ResourceOptions::shared())
        .unwrap();
    let command = icb.indirect_compute_command_at(0).unwrap();
    command.set_compute_pipeline_state(&state).unwrap();
    command.set_kernel_buffer(Some(&zeros), 0, 0).unwrap();
    command.set_kernel_buffer(Some(no_copy), 0, 1).unwrap();
    command.set_kernel_buffer(Some(&scale), 0, 2).unwrap();
    command
        .concurrent_dispatch_threads(Size::new(WORDS, 1, 1), Size::new(GROUP, 1, 1))
        .unwrap();
    host.fill(SOURCE);
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder
        .use_resource(ResourceRef::Buffer(&zeros), ResourceUsage::READ)
        .unwrap();
    encoder
        .use_resource(ResourceRef::Buffer(no_copy), ResourceUsage::WRITE)
        .unwrap();
    encoder
        .use_resource(ResourceRef::Buffer(&scale), ResourceUsage::READ)
        .unwrap();
    encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
    encoder.end_encoding().unwrap();
    Encoded {
        watched: vec![
            watch("the input buffer", zeros.held()),
            watch("the scale buffer", scale.held()),
        ],
        released: vec![Box::new(icb), Box::new(zeros), Box::new(scale)],
        check: Box::new(move |host| {
            // The program's state, released once the command buffer completed.
            drop(state);
            assert!(
                host.words()
                    .iter()
                    .enumerate()
                    .all(|(i, &word)| f32::from_bits(word) == i as f32),
                "the indirect command's words are not in the pages"
            );
        }),
    }
}

/// A sparse mapping leg: an indirect update that maps every tile of a sparse texture's level 0, or
/// nothing, its `MTLMapIndirectBuffer` in the no-copy buffer; `None`, printed, where the device has
/// no sparse textures.
fn sparse_mapping(
    leg: Leg,
    device: &Device,
    command_buffer: &CommandBuffer,
    host: &Host,
    no_copy: &Buffer,
) -> Option<Encoded> {
    let lacks = |error: &Error| {
        matches!(error, Error::Unsupported(u) if matches!(u.feature, Feature::SparseTextures { .. }))
            || *error == Error::NotImplemented(RuntimeGap::SparseTileMaps)
    };
    let descriptor = HeapDescriptor::new(HeapType::Sparse, 1 << 20, ResourceOptions::private());
    let heap = match device.new_heap(&descriptor) {
        Ok(heap) => heap,
        Err(error) if lacks(&error) => {
            eprintln!("ENCODER-HOLDS: no sparse heap: {error:?}");
            return None;
        }
        Err(error) => panic!("a sparse heap: {error:?}"),
    };
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::R32Float, 256, 256, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    let texture = match heap.new_texture(&descriptor) {
        Ok(texture) => texture,
        Err(error) if lacks(&error) => {
            eprintln!("ENCODER-HOLDS: no sparse texture: {error:?}");
            return None;
        }
        Err(error) => panic!("a sparse texture: {error:?}"),
    };
    // The count of mappings, then each one's region origin and size in tiles, level and slice.
    let tile = device.sparse_tile_size(TextureType::Type2D, PixelFormat::R32Float, 1);
    let (across, down) = ((256 / tile.width) as u32, (256 / tile.height) as u32);
    host.fill(0);
    if leg == Leg::SparseMapping {
        host.set(0, &[1, 0, 0, 0, across, down, 1, 0, 0]);
    }
    let mut encoder = command_buffer.resource_state_command_encoder().unwrap();
    encoder
        .update_texture_mapping_indirect(&texture, TileMapping::Map, no_copy, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    let weak = heap.downgrade();
    Some(Encoded {
        watched: vec![
            watch("the sparse texture", texture.held()),
            (
                "the sparse heap",
                Box::new(move || weak.upgrade().is_some()),
            ),
        ],
        released: vec![Box::new(texture), Box::new(heap)],
        check: Box::new(|_| ()),
    })
}

/// Run `leg` at each release: its command encoded, then the busy work committed, then its
/// command buffer committed, and the no-copy buffer and the leg's objects released; the
/// deallocator must not run, nor any watched object go, before the command buffer completed, and
/// the leg's bytes must land.
fn run(leg: Leg) {
    let scratch = Scratch::new(&format!("encoder-{leg:?}"));
    let host = Host::new();
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let target = device
        .new_buffer(FILLED, ResourceOptions::private())
        .unwrap();
    for release in [Release::Committed, Release::Scheduled] {
        let runs = Arc::new(AtomicUsize::new(0));
        let Some(no_copy) = over_host(&device, &host, &runs) else {
            return;
        };
        let command_buffer = queue.command_buffer().unwrap();
        let Some(encoded) = encode(leg, &device, &queue, &command_buffer, &host, &no_copy) else {
            return;
        };
        let busy = busy(&queue, &target);
        command_buffer.commit().unwrap();
        if release == Release::Scheduled {
            command_buffer.wait_until_scheduled().unwrap();
        }
        let pending = command_buffer.status() != CommandBufferStatus::Completed;
        let Encoded {
            released,
            watched,
            check,
        } = encoded;
        drop(no_copy);
        drop(released);
        let at_release = runs.load(Ordering::SeqCst);
        let gone: Vec<&str> = watched
            .iter()
            .filter(|(_, lives)| !lives())
            .map(|&(name, _)| name)
            .collect();
        let still = command_buffer.status() != CommandBufferStatus::Completed;
        eprintln!(
            "ENCODER-HOLDS {leg:?} released {release:?} on {}: pending at the release {pending}, \
             still after it {still}, deallocator runs at the release {at_release}, gone {gone:?}",
            device.name()
        );
        if pending && still {
            assert_eq!(
                at_release, 0,
                "{leg:?}: the no-copy buffer's import went while its command buffer ran"
            );
            assert!(
                gone.is_empty(),
                "{leg:?}: released while its command buffer ran: {gone:?}"
            );
        } else {
            eprintln!("ENCODER-HOLDS {leg:?}: completed before the release: nothing to hold");
        }
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        busy.wait_until_completed_for(WAIT).unwrap();
        until("the deallocator never ran", || {
            runs.load(Ordering::SeqCst) == 1
        });
        check(&host);
    }
}

#[test]
fn a_no_copy_buffer_a_committed_draw_reads_as_vertex_buffers_keeps_its_import_until_completion() {
    run(Leg::VertexBuffers);
}

#[test]
fn a_no_copy_buffer_a_committed_draw_binds_over_another_keeps_its_import_until_completion() {
    run(Leg::VertexBuffersRebound);
}

#[test]
fn a_no_copy_buffer_a_committed_draw_reads_as_a_fragment_buffer_keeps_its_import_until_completion()
{
    run(Leg::FragmentBuffer);
}

#[test]
fn a_no_copy_buffer_a_committed_draw_reads_indices_from_keeps_its_import_until_completion() {
    run(Leg::IndexBuffer);
}

#[test]
fn a_no_copy_buffer_a_committed_draw_reads_arguments_from_keeps_its_import_until_completion() {
    run(Leg::IndirectDraw);
}

#[test]
fn a_no_copy_buffer_a_committed_draw_renders_into_keeps_its_import_until_completion() {
    run(Leg::Attachment);
}

#[test]
fn a_no_copy_buffer_a_committed_dispatch_reads_keeps_its_import_until_completion() {
    run(Leg::KernelReads);
}

#[test]
fn a_no_copy_buffer_a_committed_dispatch_writes_keeps_its_import_until_completion() {
    run(Leg::KernelWrites);
}

#[test]
fn a_no_copy_buffer_a_committed_dispatch_reads_arguments_from_keeps_its_import_until_completion() {
    run(Leg::IndirectDispatch);
}

#[test]
fn a_no_copy_buffer_a_committed_dispatch_writes_a_texture_over_keeps_its_import_until_completion() {
    run(Leg::TextureWrite);
}

#[test]
fn a_no_copy_buffer_a_committed_dispatch_reaches_by_address_keeps_its_import_until_completion() {
    run(Leg::UsedResource);
}

#[test]
fn a_heap_a_committed_dispatch_reaches_by_address_lives_until_completion() {
    run(Leg::UsedHeap);
}

#[test]
fn a_no_copy_buffer_a_committed_indirect_command_writes_keeps_its_import_until_completion() {
    run(Leg::IndirectCommands);
}

#[test]
fn a_no_copy_buffer_a_committed_sparse_update_reads_keeps_its_import_until_completion() {
    run(Leg::SparseMapping);
}

#[test]
fn a_sparse_update_that_maps_nothing_keeps_its_texture_and_buffer_until_completion() {
    run(Leg::SparseMappingNothing);
}
