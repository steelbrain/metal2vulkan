//! A new heap set takes each image array's elements from a live set's array of the same class
//! (`heap_sets.rs`, **Heap sets**): copied, with what that array notes of them, and no texture
//! taken (`WorkCounts::heap_arrays_copied`).
//!
//! - **The elements.** Two textures made, a kernel of one layout (`handles.ll`'s
//!   `argument_buffer_read`, a sampled `texture2d<float>`) reads the first through its resource ID,
//!   making the first set; then a kernel of a second layout sharing that class
//!   (`a4_heap_textures`: a sampled `texture2d<float>` and a written one) reads the first through
//!   its ID and writes the second through its own: the second set's sampled array took its elements
//!   from the first set's, and the kernel reads the first texture's texels through it.
//! - **The notes.** A shadowed texture (over host pages, `HostPageTexturePolicy::Shadow`) read
//!   through its ID by the first kernel, which names it (`use_resource`), and then by the second,
//!   which does not: the second set's copied array notes the shadow as the first's did, so its
//!   command announces `UnnamedShadowedTexture`, where the first's announced nothing.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{over_host, Host, Scratch, WAIT};
use crate::{
    Buffer, CommandBufferStatus, ComputePipelineState, Degradation, DegradeNotice, Device,
    DeviceKind, DeviceOptions, MessageSink, PipelineOptions, Region, ResourceOptions,
    ShaderCacheOptions, Size, Texture, TextureDescriptor,
};
use metal2vulkan::device::HostPageTexturePolicy;
use metal2vulkan::metal::{PixelFormat, ResourceUsage, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use std::path::Path;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};

/// The textures' width, in texels, and the threads of each dispatch.
const SIDE: u64 = 8;

/// The notices a device announced.
type Heard = Arc<Mutex<Vec<DegradeNotice>>>;

/// A device of the kind the cfg selects, every texture over host pages a shadow, its notices
/// into `heard`, each kind once.
fn device(cache: &Path, heard: &Heard) -> Device {
    let heard = Arc::clone(heard);
    let mut options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        messages: Some(MessageSink::new(move |message| {
            if let Some(notice) = &message.degradation {
                heard.lock().unwrap().push(notice.clone());
            }
        })),
        ..DeviceOptions::default()
    };
    options.policy.host_page_textures = HostPageTexturePolicy::Shadow;
    Device::system_default_with(&options).unwrap()
}

/// How many notices of `kind` `heard` holds.
fn heard_of(heard: &Heard, kind: Degradation) -> usize {
    let heard = heard.lock().unwrap();
    heard.iter().filter(|n| n.degradation == kind).count()
}

/// The compute state of `entry` of the owned engine fixture `stem` on `device`.
fn state(device: &Device, stem: &str, entry: &str) -> ComputePipelineState {
    let ll = metal2vulkan_devtools::engine_dir().join(format!("{stem}.ll"));
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let library = metal2vulkan::library::Library::from_air(&bitcode).unwrap();
    let function = device
        .new_library_with_compile_library(library)
        .new_function_with_name(entry)
        .unwrap();
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A shared buffer of `bytes`.
fn buffer(device: &Device, bytes: &[u8]) -> Buffer {
    let buffer = device
        .new_buffer(bytes.len() as u64, ResourceOptions::shared())
        .unwrap();
    let contents = buffer.contents().unwrap();
    // SAFETY: the new buffer's bytes, which no command uses yet.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
    buffer
}

/// A shared `RGBA32Float` texture of [`SIDE`] by 1 texels of `usage`.
fn texture(device: &Device, usage: TextureUsage) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA32Float, SIDE, 1, false);
    descriptor.usage = usage;
    descriptor.resource_options = ResourceOptions::shared();
    device.new_texture(&descriptor).unwrap()
}

/// Texel `x` of the source.
fn texel(x: u64) -> [f32; 4] {
    [x as f32 + 1.0, 2.0 * x as f32 + 1.0, 0.25, 1.0]
}

/// The little-endian bytes of `words`.
fn bytes_of(words: impl IntoIterator<Item = u64>) -> Vec<u8> {
    words.into_iter().flat_map(u64::to_le_bytes).collect()
}

/// Dispatch `state` over [`SIDE`] threads with `argument` at buffer 0 and `out` at buffer 1
/// where given, `named` made resident, in a command buffer of its own, and wait for it.
fn dispatch(
    device: &Device,
    state: &ComputePipelineState,
    argument: &Buffer,
    out: Option<&Buffer>,
    named: Option<&Texture>,
) {
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    if let Some(named) = named {
        encoder
            .use_resource(named.as_resource(), ResourceUsage::READ)
            .unwrap();
    }
    encoder.set_buffer(Some(argument), 0, 0).unwrap();
    if let Some(out) = out {
        encoder.set_buffer(Some(out), 0, 1).unwrap();
    }
    encoder
        .dispatch_threads(Size::new(SIDE, 1, 1), Size::new(SIDE, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
}

#[test]
fn a_second_layout_s_set_takes_the_class_s_elements_from_the_first_s() {
    let scratch = Scratch::new("heap-set-copies");
    let heard = Heard::default();
    let device = device(&scratch.0, &heard);
    let source = texture(&device, TextureUsage::SHADER_READ);
    let texels: Vec<u8> = (0..SIDE)
        .flat_map(texel)
        .flat_map(f32::to_le_bytes)
        .collect();
    let whole = Region::new_2d(0, 0, SIDE, 1);
    source
        .replace_region(whole, 0, 0, &texels, 16 * SIDE, 0)
        .unwrap();
    let target = texture(&device, TextureUsage::SHADER_WRITE);
    // The first layout's set, made at its first command, walks the live textures.
    let first = state(&device, "handles", "argument_buffer_read");
    let read = buffer(&device, &[0; 16 * SIDE as usize]);
    let named = buffer(&device, &bytes_of([source.gpu_resource_id().0]));
    dispatch(&device, &first, &named, Some(&read), None);
    let contents = read.contents().unwrap();
    // SAFETY: the buffer's bytes, which the completed command buffer wrote.
    let read = unsafe { std::slice::from_raw_parts(contents.as_ptr(), texels.len()) };
    assert_eq!(
        read, texels,
        "the first kernel reads the source through its ID"
    );
    // The second layout's set takes its sampled array's elements from the first set's.
    let before = device.work_counts().heap_arrays_copied;
    let second = state(&device, "a4_heap_textures", "a4_heap_textures");
    let ids = [source.gpu_resource_id().0, target.gpu_resource_id().0];
    dispatch(
        &device,
        &second,
        &buffer(&device, &bytes_of(ids)),
        None,
        None,
    );
    let copied = device.work_counts().heap_arrays_copied - before;
    assert!(
        copied >= 1,
        "the second set copies the sampled class's array from the first set's"
    );
    let mut written = vec![0; texels.len()];
    target
        .get_bytes(&mut written, 16 * SIDE, 0, whole, 0, 0)
        .unwrap();
    let want: Vec<u8> = (0..SIDE)
        .flat_map(|x| {
            let [r, g, b, a] = texel(x);
            [r + SIDE as f32, g, b, a]
        })
        .flat_map(f32::to_le_bytes)
        .collect();
    assert_eq!(
        written, want,
        "the second kernel reads the source through its ID in the copied array (plus its width)"
    );
}

#[test]
fn a_copied_array_announces_the_shadowed_texture_its_source_noted() {
    // The pages, declared before the device that uses them, so dropped after it.
    let host = Host::new();
    let scratch = Scratch::new("heap-set-copies-shadow");
    let heard = Heard::default();
    let device = device(&scratch.0, &heard);
    let runs = Arc::new(AtomicUsize::new(0));
    let Some(pages) = over_host(&device, &host, &runs) else {
        return;
    };
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA32Float, SIDE, 1, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = pages.resource_options();
    let row = device
        .minimum_linear_texture_alignment_for_pixel_format(PixelFormat::RGBA32Float)
        .unwrap()
        .max(16 * SIDE);
    let shadowed = pages.new_texture(&descriptor, 0, row).unwrap();
    assert!(
        shadowed.shadow().is_some(),
        "a shadow holds the host pages' texels"
    );
    let target = texture(&device, TextureUsage::SHADER_WRITE);
    // The first set, made walking the live textures, notes the shadow; its command names it.
    let first = state(&device, "handles", "argument_buffer_read");
    let read = buffer(&device, &[0; 16 * SIDE as usize]);
    let named = buffer(&device, &bytes_of([shadowed.gpu_resource_id().0]));
    dispatch(&device, &first, &named, Some(&read), Some(&shadowed));
    assert_eq!(
        heard_of(&heard, Degradation::UnnamedShadowedTexture),
        0,
        "the first command names the shadowed texture"
    );
    // The second set's copied array notes it too; its command does not name it.
    let before = device.work_counts().heap_arrays_copied;
    let second = state(&device, "a4_heap_textures", "a4_heap_textures");
    let ids = [shadowed.gpu_resource_id().0, target.gpu_resource_id().0];
    dispatch(
        &device,
        &second,
        &buffer(&device, &bytes_of(ids)),
        None,
        None,
    );
    assert!(device.work_counts().heap_arrays_copied > before);
    assert_eq!(
        heard_of(&heard, Degradation::UnnamedShadowedTexture),
        1,
        "the second command, which does not name it, reads the shadow its copied array holds"
    );
    drop((shadowed, pages));
}
