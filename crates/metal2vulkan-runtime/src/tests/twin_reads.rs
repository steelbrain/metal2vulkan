//! A decoded twin whose texels are not Metal's decode exactly announces `decoded-texel-precision`
//! where a command may first read it (`texture/decoded_twin.rs`, **Reads**), once per device, and
//! never where its texture is made, written, filled, copied or read back: those move the
//! carrier's blocks.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. An `ASTC_4x2_LDR` texture lies in a
//! carrier with a twin of 8-bit codes on every device (no Vulkan format holds its block), so every
//! path is walked on every device; a 4:2:2 texture's twin holds the device's own conversion where
//! the device converts its format (`YCbCrSampler::pairs`: MoltenVK), and its paths are walked
//! there. The notice is announced once per device, so each path runs on a device of its own,
//! whose first read is that path's:
//!
//! - **Set-0 slots**: a kernel's `read`, `sample` and `gather` of the texture (`Texture::bound_view`
//!   through `slot_view`), and a `read` of a view of it in its own format.
//! - **Heap sets**: a kernel reading the texture through its resource ID in an argument buffer
//!   (`handles.ll`'s `argument_buffer_read`), with the texture named by `use_resource` and without
//!   it (`heap_sets::bind`).
//! - **A placement while a reader is pending**: a command buffer that bound a heap set and has not
//!   committed, then the texture made: its view is placed in a set the command buffer may read, so
//!   it announces as it is made (`heap_sets::SetEntry::twin`).
//! - **A second twin**: an `ASTC_4x2_sRGB` texture, whose twin holds Metal's codes, read (nothing),
//!   then viewed as `ASTC_4x2_LDR` (a second twin of 8-bit codes nearest Metal's values; nothing)
//!   and read through the view.

use super::blit_holds::{Scratch, WAIT};
use crate::ycbcr_device::YCbCrSampler;
use crate::{
    BlitOption, Buffer, CommandBufferStatus, ComputePipelineState, Degradation, DegradeNotice,
    Device, DeviceKind, DeviceOptions, MessageSink, Origin, PipelineOptions, Region,
    ResourceOptions, ShaderCacheOptions, Size, Texture, TextureDescriptor,
};
use metal2vulkan::bind::{ChromaDecode, SamplerDescriptor};
use metal2vulkan::metal::{PixelFormat, ResourceUsage, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// The textures' side, in texels: two `ASTC_4x2` blocks across and four down, four 4:2:2 pairs.
const SIDE: u64 = 8;

/// The notices a device announced.
type Heard = Arc<Mutex<Vec<DegradeNotice>>>;

/// A device of the kind the cfg selects, its notices into `heard`, each kind once.
fn device(cache: &Path, heard: &Heard) -> Device {
    let heard = Arc::clone(heard);
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
        messages: Some(MessageSink::new(move |message| {
            if let Some(notice) = &message.degradation {
                heard.lock().unwrap().push(notice.clone());
            }
        })),
        ..DeviceOptions::default()
    };
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

/// A `Shared` buffer holding `bytes`.
fn buffer(device: &Device, bytes: &[u8]) -> Buffer {
    let buffer = device
        .new_buffer(bytes.len() as u64, ResourceOptions::shared())
        .unwrap();
    let contents = buffer.contents().unwrap();
    // SAFETY: the new buffer's bytes, which no command uses yet.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
    buffer
}

/// A row's bytes of a [`SIDE`]-wide texture of `pf`: two 16-byte `ASTC_4x2` blocks, or 2 bytes a
/// 4:2:2 texel. Its whole image is [`SIDE`] rows (an ASTC one's blocks two rows of texels each).
fn row_bytes(pf: PixelFormat) -> u64 {
    match ChromaDecode::of(pf) {
        Some(_) => 2 * SIDE,
        None => 32,
    }
}

fn image_bytes(pf: PixelFormat) -> u64 {
    match ChromaDecode::of(pf) {
        Some(_) => row_bytes(pf) * SIDE,
        None => row_bytes(pf) * SIDE / 2,
    }
}

/// A [`SIDE`]-square 2D texture of `pf`, shader-read, `Shared` or `Private`.
fn texture(device: &Device, pf: PixelFormat, shared: bool) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(pf, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW;
    descriptor.resource_options = match shared {
        true => ResourceOptions::shared(),
        false => ResourceOptions::private(),
    };
    device
        .new_texture(&descriptor)
        .unwrap_or_else(|e| panic!("{pf:?}: {e:?}"))
}

/// Run `record` in one blit encoder of a new command buffer, committed and waited for.
fn blit(device: &Device, record: impl FnOnce(&mut crate::BlitCommandEncoder)) {
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    record(&mut encoder);
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// Every write and read-back of `pf`'s texels that moves its carrier's blocks and reads no twin:
/// a `Shared` texture's `replace_region` and `get_bytes`; a `Private` one's blit from a buffer, a
/// pattern fill, a copy into another texture and a blit into a buffer. The texture written last,
/// the `Private` one.
fn written(device: &Device, pf: PixelFormat) -> Texture {
    let (row, image) = (row_bytes(pf), image_bytes(pf));
    let bytes: Vec<u8> = (0..image).map(|i| (i * 37 + 11) as u8).collect();
    let whole = Region::new_2d(0, 0, SIDE, SIDE);
    let shared = texture(device, pf, true);
    shared.replace_region(whole, 0, 0, &bytes, row, 0).unwrap();
    let mut back = vec![0; image as usize];
    shared.get_bytes(&mut back, row, 0, whole, 0, 0).unwrap();
    assert_eq!(back, bytes, "{pf:?}: its bytes");
    let private = texture(device, pf, false);
    let copy = texture(device, pf, false);
    let source = buffer(device, &bytes);
    let out = buffer(device, &vec![0; image as usize]);
    let side = Size::new(SIDE, SIDE, 1);
    let at = Origin::new(0, 0, 0);
    blit(device, |e| {
        e.copy_from_buffer_to_texture(
            &source,
            0,
            row,
            image,
            side,
            &private,
            0,
            0,
            at,
            BlitOption::None,
        )
        .unwrap();
        e.fill_texture(&private, 0, 0, Region::new_2d(0, 0, SIDE, 2), &[0x5a; 16])
            .unwrap();
        e.copy_from_texture_to_texture(&private, 0, 0, at, side, &copy, 0, 0, at)
            .unwrap();
        e.copy_from_texture_to_buffer(&copy, 0, 0, at, side, &out, 0, row, image, BlitOption::None)
            .unwrap();
    });
    private
}

/// How a command first reads the twin: a route of the module docs.
#[derive(Clone, Copy, Debug)]
enum Route {
    SlotRead,
    SlotSample,
    SlotGather,
    ViewRead,
    HeapNamed,
    HeapUnnamed,
}

const ROUTES: [Route; 6] = [
    Route::SlotRead,
    Route::SlotSample,
    Route::SlotGather,
    Route::ViewRead,
    Route::HeapNamed,
    Route::HeapUnnamed,
];

/// The kernels the routes dispatch on one device.
struct Kernels {
    read: ComputePipelineState,
    sample: ComputePipelineState,
    gather: ComputePipelineState,
    heap: ComputePipelineState,
}

impl Kernels {
    fn new(device: &Device) -> Kernels {
        let fixture = "fpv_swizzled_reads";
        Kernels {
            read: state(device, fixture, "fpv_swz_read"),
            sample: state(device, fixture, "fpv_swz_sample"),
            gather: state(device, fixture, "fpv_swz_gather_x"),
            heap: state(device, "handles", "argument_buffer_read"),
        }
    }
}

/// `route`'s read of `t` on `device`, one dispatch of [`SIDE`] threads in a command buffer of its
/// own; with `commit` false, the command buffer is handed back encoded and uncommitted.
fn read(
    device: &Device,
    kernels: &Kernels,
    route: Route,
    t: &Texture,
    commit: bool,
) -> Option<crate::CommandBuffer> {
    let out = device
        .new_buffer(16 * SIDE * SIDE, ResourceOptions::shared())
        .unwrap();
    let texels: Vec<u8> = (0..SIDE)
        .flat_map(|i| [(i as u32), (i as u32) / 2])
        .flat_map(u32::to_le_bytes)
        .collect();
    let uvs: Vec<u8> = (0..SIDE)
        .flat_map(|i| [(i as f32 + 0.5) / SIDE as f32, 0.5])
        .flat_map(f32::to_le_bytes)
        .collect();
    let sampler = device.new_sampler_state(&SamplerDescriptor::new()).unwrap();
    let argument = buffer(device, &t.gpu_resource_id().0.to_le_bytes());
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    let (texels, uvs) = (buffer(device, &texels), buffer(device, &uvs));
    match route {
        Route::SlotRead | Route::ViewRead => {
            encoder.set_compute_pipeline_state(&kernels.read).unwrap();
            encoder.set_buffer(Some(&texels), 0, 1).unwrap();
        }
        Route::SlotSample | Route::SlotGather => {
            let state = match route {
                Route::SlotSample => &kernels.sample,
                _ => &kernels.gather,
            };
            encoder.set_compute_pipeline_state(state).unwrap();
            encoder.set_sampler_state(Some(&sampler), 0).unwrap();
            encoder.set_buffer(Some(&uvs), 0, 1).unwrap();
        }
        Route::HeapNamed | Route::HeapUnnamed => {
            encoder.set_compute_pipeline_state(&kernels.heap).unwrap();
            if matches!(route, Route::HeapNamed) {
                encoder
                    .use_resource(t.as_resource(), ResourceUsage::READ)
                    .unwrap();
            }
            encoder.set_buffer(Some(&argument), 0, 0).unwrap();
        }
    }
    match route {
        Route::SlotRead | Route::SlotSample | Route::SlotGather => {
            encoder.set_texture(Some(t), 0).unwrap();
            encoder.set_buffer(Some(&out), 0, 0).unwrap();
        }
        Route::ViewRead => {
            let view = t.new_texture_view(t.pixel_format()).unwrap();
            encoder.set_texture(Some(&view), 0).unwrap();
            encoder.set_buffer(Some(&out), 0, 0).unwrap();
        }
        Route::HeapNamed | Route::HeapUnnamed => {
            encoder.set_buffer(Some(&out), 0, 1).unwrap();
        }
    }
    encoder
        .dispatch_threads(Size::new(SIDE, 1, 1), Size::new(SIDE, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    if !commit {
        return Some(command_buffer);
    }
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
    None
}

/// The formats whose twin's texels are not Metal's decode exactly on `device`: `ASTC_4x2_LDR` on
/// every device, and the 4:2:2 formats where the device converts them.
fn imprecise(device: &Device) -> Vec<PixelFormat> {
    let mut formats = vec![PixelFormat::ASTC4x2LDR];
    for pf in [PixelFormat::BGRG422, PixelFormat::GBGR422] {
        let decode = ChromaDecode::of(pf).unwrap();
        if YCbCrSampler::pairs(device, decode, [SIDE as u32; 2]).is_some() {
            formats.push(pf);
        }
    }
    formats
}

/// Each imprecise format, on a device of its own for each route: made, written, filled, copied and
/// read back, it announces `carried-decoded-twin` and no `decoded-texel-precision`; the route's
/// first read announces `decoded-texel-precision` once, and every later read of every route
/// nothing more.
#[test]
fn a_decoded_twin_announces_its_precision_at_its_first_read_alone() {
    let scratch = Scratch::new("twin-reads");
    let probe = device(&scratch.0, &Heard::default());
    let formats = imprecise(&probe);
    drop(probe);
    println!("twin reads walked for {formats:?}");
    for &pf in &formats {
        for route in ROUTES {
            let heard = Heard::default();
            let device = device(&scratch.0, &heard);
            let kernels = Kernels::new(&device);
            let t = written(&device, pf);
            let what = format!("{pf:?} {route:?}");
            assert_eq!(
                heard_of(&heard, Degradation::CarriedDecodedTwin),
                1,
                "{what}: the twin's cost, as it is made"
            );
            assert_eq!(
                heard_of(&heard, Degradation::DecodedTexelPrecision),
                0,
                "{what}: no read yet"
            );
            read(&device, &kernels, route, &t, true);
            assert_eq!(
                heard_of(&heard, Degradation::DecodedTexelPrecision),
                1,
                "{what}: the first read"
            );
            for again in ROUTES {
                read(&device, &kernels, again, &t, true);
            }
            assert_eq!(
                heard_of(&heard, Degradation::DecodedTexelPrecision),
                1,
                "{what}: once per device"
            );
        }
    }
}

/// A texture made while a command buffer that bound a heap set has not committed is placed in a
/// set the command buffer may read through its ID: it announces as it is made. A texture made on
/// a device where no command bound a heap set announces nothing until a command reads it.
#[test]
fn a_twin_placed_in_a_heap_set_a_pending_command_buffer_bound_announces_as_it_is_made() {
    let scratch = Scratch::new("twin-placed");
    let heard = Heard::default();
    let device = device(&scratch.0, &heard);
    let kernels = Kernels::new(&device);
    let plain = texture(&device, PixelFormat::RGBA8Unorm, false);
    let pending = read(&device, &kernels, Route::HeapUnnamed, &plain, false).unwrap();
    assert_eq!(heard_of(&heard, Degradation::DecodedTexelPrecision), 0);
    let carried = texture(&device, PixelFormat::ASTC4x2LDR, false);
    assert_eq!(
        heard_of(&heard, Degradation::DecodedTexelPrecision),
        1,
        "placed in a set the pending command buffer bound"
    );
    pending.commit().unwrap();
    pending.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(pending.status(), CommandBufferStatus::Completed);
    read(&device, &kernels, Route::HeapUnnamed, &carried, true);
    assert_eq!(heard_of(&heard, Degradation::DecodedTexelPrecision), 1);
}

/// An `ASTC_4x2_sRGB` texture's twin holds Metal's own codes: read, it announces no precision. Its
/// view as `ASTC_4x2_LDR` gives it a second twin of 8-bit codes nearest Metal's values, which
/// announces nothing as it is made, and its precision at the view's first read.
#[test]
fn a_second_twin_announces_its_precision_at_its_first_read() {
    let scratch = Scratch::new("twin-second");
    let heard = Heard::default();
    let device = device(&scratch.0, &heard);
    let kernels = Kernels::new(&device);
    let t = written(&device, PixelFormat::ASTC4x2SRGB);
    read(&device, &kernels, Route::SlotRead, &t, true);
    read(&device, &kernels, Route::HeapUnnamed, &t, true);
    assert_eq!(
        heard_of(&heard, Degradation::DecodedTexelPrecision),
        0,
        "an sRGB twin holds Metal's codes"
    );
    let ldr = t.new_texture_view(PixelFormat::ASTC4x2LDR).unwrap();
    assert_eq!(
        heard_of(&heard, Degradation::DecodedTexelPrecision),
        0,
        "the second twin made"
    );
    read(&device, &kernels, Route::SlotRead, &ldr, true);
    assert_eq!(
        heard_of(&heard, Degradation::DecodedTexelPrecision),
        1,
        "the second twin's first read"
    );
}
