//! A write through a shadowed texture over bytes the CPU may write while a command buffer runs
//! (a `Shared` buffer's, a no-copy buffer's host pages, `shadow.rs`): the shadow is kept after its
//! copy in and the bytes the commands changed are merged back in place of the copy back, so no
//! texel the commands did not write goes back over the CPU's writes, as Metal leaves them. The
//! tests order the CPU's writes against the GPU by the commit and the completion alone, which
//! Metal and Vulkan both define (`tests::shadow` checks the copies planned for a CPU write made
//! while the command buffer runs, and `runtime_lane::host_contract` makes one).
//!
//! Each leg makes a 16x16 `RGBA8Unorm` texture over a buffer, rows of 256 bytes (a shadow on
//! every device but one that runs on Metal, whose texture is Metal's over the buffer): over a
//! `Shared` buffer, and over a no-copy buffer's host pages. A compute encoder's dispatch writes
//! texel (0, 0) through it (`invert_texels`). Once the encoder ended and before the commit, the CPU
//! writes texel (5, 5)'s bytes. Once the command buffer completed, texel
//! (0, 0) holds the dispatch's value, texel (5, 5) the CPU's, and every other byte what it held:
//! what Metal leaves. No texel was copied back (`WorkCounts::shadow_texels_copied_back`).
//!
//! A shadow's bytes over host pages lie in no `VkBuffer` while they are detached
//! (`ShadowBuffer::raw`): the device's stand-in, which holds none of them, is never what a copy
//! or a merge's window takes, and a rebind gives the new pages' `VkBuffer`.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The function is the owned fixture
//! `validation/fixtures/public/examples/runtime_examples.ll` (`llvm-as`).

use super::blit_holds::{over_host, Host, Scratch, BYTES, WAIT};
use crate::shadow::ShadowBuffer;
use crate::{
    Buffer, CommandBufferStatus, ComputePipelineState, Device, DeviceKind, DeviceOptions, Library,
    PipelineOptions, Region, ResourceOptions, ShaderCacheOptions, Size, Texture, TextureDescriptor,
};
use ash::vk;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::path::Path;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

/// The device the cfg selects, its shader cache in `cache`.
fn device(cache: &Path) -> Device {
    let options = DeviceOptions {
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
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// The texture: `SIDE` texels square, rows of `ROW` bytes (its texels' 64 and 192 between).
const SIDE: u64 = 16;
const ROW: u64 = 256;
/// What every byte of the texture's rows holds before the leg.
const BEFORE: u8 = 0x11;
/// The dispatch's texel: `invert_texels` of a zero texel.
const WRITTEN: [u8; 4] = [255, 255, 255, 0];
/// The texel the CPU writes, and its bytes.
const CPU_TEXEL: (u64, u64) = (5, 5);
const CPU_BYTES: [u8; 4] = [1, 2, 3, 4];

/// The library of the fixture at `path` under `validation/fixtures/public/`.
fn library(device: &Device, path: &str) -> Library {
    let ll = metal2vulkan_devtools::fixtures_dir().join(path);
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let library = metal2vulkan::library::Library::from_air(&bitcode).unwrap();
    device.new_library_with_compile_library(library)
}

fn state(device: &Device, path: &str, kernel: &str) -> ComputePipelineState {
    let function = library(device, path)
        .new_function_with_name(kernel)
        .unwrap();
    device
        .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
        .unwrap()
}

/// The leg over `buffer`'s first bytes (a texture of rows of [`ROW`] bytes), which the CPU reaches
/// at `bytes`, on `device`.
fn leg(what: &str, device: &Device, buffer: &Buffer, bytes: *mut u8) {
    let queue = device.new_command_queue().unwrap();
    let span = (ROW * SIDE) as usize;
    // SAFETY: the leg's bytes, which no command uses now.
    unsafe { std::ptr::write_bytes(bytes, BEFORE, span) };
    let before = device.work_counts().shadowed_textures;
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    descriptor.resource_options = buffer.resource_options();
    let target: Texture = buffer.new_texture(&descriptor, 0, ROW).unwrap();
    let shadowed = device.work_counts().shadowed_textures - before;
    eprintln!("SHADOW-HOST-WRITES: {what}: {shadowed} shadow made");
    let mut source = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 1, 1, false);
    source.usage = TextureUsage::SHADER_READ;
    source.resource_options = ResourceOptions::shared();
    let source = device.new_texture(&source).unwrap();
    source
        .replace_region(Region::new_2d(0, 0, 1, 1), 0, 0, &[0; 4], 4, 0)
        .unwrap();
    let invert = state(device, "examples/runtime_examples.ll", "invert_texels");

    // Counted as the encoder plans its copies.
    let copied_back = device.work_counts().shadow_texels_copied_back;
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&invert).unwrap();
    encoder.set_texture(Some(&source), 0).unwrap();
    encoder.set_texture(Some(&target), 1).unwrap();
    encoder
        .dispatch_threads(Size::new(1, 1, 1), Size::new(1, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    let (x, y) = CPU_TEXEL;
    let at = (y * ROW + x * 4) as usize;
    // SAFETY: a texel's bytes inside the buffer, which no command uses until the commit.
    unsafe { std::ptr::copy_nonoverlapping(CPU_BYTES.as_ptr(), bytes.add(at), 4) };
    command_buffer.commit().unwrap();
    assert!(
        command_buffer.wait_until_completed_for(WAIT).is_ok(),
        "{what}: the command buffer never completed"
    );
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let copied_back = device.work_counts().shadow_texels_copied_back - copied_back;

    // SAFETY: the command buffer completed; nothing writes the bytes now.
    let now = unsafe { std::slice::from_raw_parts(bytes, span) };
    let mut expected = vec![BEFORE; span];
    expected[..4].copy_from_slice(&WRITTEN);
    expected[at..at + 4].copy_from_slice(&CPU_BYTES);
    let differing: Vec<usize> = (0..span).filter(|&i| now[i] != expected[i]).collect();
    assert!(
        differing.is_empty(),
        "{what}: bytes {:?} differ (texel (5, 5) holds {:?}, the CPU wrote \
         {CPU_BYTES:?}; texel (0, 0) holds {:?}, the dispatch wrote {WRITTEN:?})",
        &differing[..differing.len().min(16)],
        &now[at..at + 4],
        &now[..4],
    );
    // Merged back: no texel goes back as a copy.
    assert_eq!(copied_back, 0, "{what}: texels copied back");
    eprintln!(
        "SHADOW-HOST-WRITES: {what}: {copied_back} texels copied back; texel (5, 5) holds {:?}",
        &now[at..at + 4]
    );
}

#[test]
fn a_shadow_over_a_shared_buffer_is_merged_back() {
    let scratch = Scratch::new("shadow-host-writes-shared");
    let device = device(&scratch.0);
    let buffer = device
        .new_buffer(ROW * SIDE, ResourceOptions::shared())
        .unwrap();
    let bytes = buffer.contents().unwrap().as_ptr();
    leg("a Shared buffer", &device, &buffer, bytes);
}

#[test]
fn a_shadow_over_host_pages_is_merged_back() {
    let scratch = Scratch::new("shadow-host-writes-pages");
    let host = Host::new();
    let device = device(&scratch.0);
    let runs = Arc::new(AtomicUsize::new(0));
    let Some(buffer) = over_host(&device, &host, &runs) else {
        return;
    };
    assert!(ROW * SIDE <= BYTES as u64);
    let bytes = buffer.contents().unwrap().as_ptr();
    leg("host pages", &device, &buffer, bytes);
    drop(buffer);
    drop(device);
}

/// A shadow's bytes over a no-copy buffer's host pages, detached, lie in no `VkBuffer` (a merge
/// planned then bound a window of the device's 256-byte stand-in past its size,
/// `VUID-VkDescriptorBufferInfo-range-00342`), and in the new pages' once a rebind gives them.
#[test]
fn a_shadow_over_detached_host_pages_takes_no_buffer_until_a_rebind() {
    let scratch = Scratch::new("shadow-host-writes-detached");
    let (host, other) = (Host::new(), Host::new());
    let device = device(&scratch.0);
    let runs = Arc::new(AtomicUsize::new(0));
    let Some(buffer) = over_host(&device, &host, &runs) else {
        return;
    };
    let shadow = ShadowBuffer::of(&buffer, buffer.at(0).0);
    assert_eq!(shadow.raw(), buffer.at(0).0, "in pages");
    assert_eq!(buffer.detach_backing(), Ok(true), "the detach");
    assert_eq!(shadow.raw(), vk::Buffer::null(), "detached");
    // SAFETY: `other` outlives the buffer (dropped after the device), which alone uses it.
    unsafe { buffer.rebind_bytes_no_copy(other.memory(), None, None) }.unwrap();
    assert_ne!(shadow.raw(), vk::Buffer::null(), "given pages again");
    assert_eq!(shadow.raw(), buffer.at(0).0, "given pages again");
    drop(buffer);
    drop(device);
}
