//! Device tests of textures over buffers that a render pass does not store, on the device the cfg
//! selects, under the validation layer: the DontCare-store probe's cases (an Apple M3 Max under
//! macOS 27.0.1, each case 20 times alike), where Metal leaves the buffer's bytes in every texel
//! after a pass that stores its color attachment over them `DontCare`, and no byte between the
//! rows changes.
//!
//! Each case fills a buffer with the bytes `(7i + 3) mod 256`, makes a 2D texture over it at
//! offset 0, and runs one pass that loads the texture by `Load`, `Clear` (to 0.5 in every channel)
//! or `DontCare`, draws a rectangle of 64/255 in every channel over the whole texture, its left
//! half, its top left corner (at most 32x32) or nowhere, and stores it `DontCare`. Every byte of
//! the buffer must be the one it held before the pass. The probe's sweep, cut to the runtime's
//! questions:
//!
//! - **Every format** ([`FORMATS`]: `BGRA8Unorm`, `RGBA8Unorm`, `BGRA8Unorm_sRGB`, `RGBA16Float`,
//!   `R8Unorm`), over an ordinary `Shared` buffer and a no-copy buffer of host pages: 16x16 at
//!   tight rows and 64x32 at rows a multiple of 256 bytes (at which RADV renders a linear image
//!   over the buffer), each load action and full, half and no coverage; 1307x512 at padded rows
//!   (5248 bytes for `BGRA8Unorm`, as a window server's tile, the probe's rows for the others),
//!   each load action at half coverage and a corner; and, for `BGRA8Unorm` and `RGBA16Float`,
//!   333x77 at odd rows, each load action at a corner.
//! - **After a stored render** (`BGRA8Unorm`, `RGBA16Float`, `R8Unorm`): a pass loads the texture,
//!   draws 32/255 over it and stores it, then the case's pass: the buffer keeps the first pass's
//!   bytes.
//! - **Beside a second attachment** stored `Store` (the same three formats): an ordinary texture
//!   of the format, cleared to zero, which the draw writes 96/255 into: it holds them, and the
//!   buffer keeps its bytes.
//! - **Over a surface's plane** (`Buffer::new_texture_with_iosurface_plane` over a managed
//!   no-copy buffer, as a consumer stands one in for an IOSurface): `BGRA8Unorm` at 16x16 (rows of
//!   64 bytes), 64x32 (rows of 256) and 1307x512 (rows of 5248), each load action at full, half and
//!   no coverage. The case reads the buffer's bytes after a blit's `synchronize_resource` of it, as
//!   Metal asks of a `Managed` resource the GPU wrote: on a device without unified memory the
//!   buffer is two copies, and its pages take the GPU's writes there alone.
//! - **Store actions**: `Unknown` left at the pass's end, and `Unknown` set to `DontCare` as the
//!   pass ends (`set_color_store_action`), keep the bytes; `Unknown` set to `Store` stores the
//!   draw. `MultisampleResolve` of the single-sampled attachment into an ordinary texture keeps
//!   the buffer's bytes and resolves the draw (plain Metal's answer, its validation off);
//!   `StoreAndMultisampleResolve` stores and resolves it; each at 16x16 and at 64x32. One
//!   2048x2048 `BGRA8Unorm` pass over tight rows keeps the bytes too.
//! - **Through a barrier**: the draw, a barrier, and the draw again, where Metal keeps the
//!   buffer's bytes as well (the barrier probe, the same M3 Max, 20 times alike, over buffers,
//!   render targets, textures and the buffer alone): a barrier over buffers, which splits the pass
//!   into two render pass instances, and one over the render targets, where MoltenVK restarts its
//!   Metal pass; `BGRA8Unorm` and `RGBA16Float` at 16x16 and 64x32, each load action at full and
//!   half coverage, over ordinary, no-copy and (`BGRA8Unorm`) surface buffers.
//! - **Over dma-buf pages** (`HostMemory::with_dma_buf`, on Linux where `/dev/udmabuf` opens):
//!   a `BGRA8Unorm` texture over a no-copy buffer of memfd pages, which RADV makes an image over
//!   a dma-buf of its own pages, and a device that imports the pointer an image over the
//!   pointer's.
//!
//! The controls store the same passes with `Store`: the draw's texels are the rendered value (the
//! bytes the same draw stores into an ordinary texture of the format), and outside the draw the
//! buffer's (`Load`) or the clear value's (`Clear`, the bytes a clear stores there), every byte
//! between the rows the buffer's.
//!
//! The runtime renders such a pass through an image the pass copies nothing back from: the
//! texture's shadow where it has one (announced `shadowed-texture` as the texture is made), else,
//! as the texture's image lies over the buffer (linear, or on MoltenVK a Metal texture over its
//! `MTLBuffer`), through an image its texture keeps, which it
//! announces as `dont-care-stand-in` ([`KEPT_BYTES`]) and counts
//! (`WorkCounts::kept_bytes_passes`); but where the device renders the image only beside other
//! linear ones (NVIDIA's `VK_NV_linear_color_attachment`) and the pass has a second attachment,
//! through the image the pass renders it apart in (`linear-attachment-copies`), which copies back
//! only what the pass stores; and where the driver stores `DONT_CARE` as Metal's own `DontCare`
//! (`DriverFact::DontCareStoreIsMetals`, MoltenVK on the M3 Max and the M4) and the pass is one
//! render pass instance the driver runs unbroken, in place, handed to Metal, which keeps the bytes
//! (a pass through a barrier, or whose resolve copies the attachment's texels, renders through the
//! kept image there too). The device announces these kinds at each occurrence here
//! (`DeviceOptions::announce_each`): each pass announces `dont-care-stand-in` and counts one pass
//! exactly where its texture is neither shadowed, rendered apart nor handed to Metal, a `Store`
//! control neither; no
//! pass announces `dont-care-store`, Metal's answer being exact. A pass without a draw call that
//! loads by `Load` or `DontCare` records nothing at all (`render/untouched.rs`,
//! `WorkCounts::untouched_passes_skipped`), so it neither announces nor counts. Each case prints
//! its form (`KEPT-BYTES`).
//!
//! Metal's one exception, the first pass in a process to overflow its tiler (some 2 to 4 million
//! triangle-tile entries), which leaves the rendered texels, is not modelled: no case comes near.

use super::degrade_notices::{counts, on_device_with};
#[cfg(target_os = "linux")]
use super::dma_buf_imports::{udmabuf_openable, Calls, Memfd};
use super::host_memory::{host_page, no_copy, not_importable, Mapping};
use super::{Messages, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::device::DriverFact;
use metal2vulkan::metal::{BarrierScope, PixelFormat, PrimitiveType, RenderStages, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandBufferStatus, CommandQueue, Deallocator, Degradation, Device,
    IoSurfacePlane, Library, LoadAction, PipelineOptions, Region, RenderCommandEncoder,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, StorageMode, StoreAction, Texture,
    TextureDescriptor,
};
use std::collections::BTreeMap;

/// The kind a pass that renders a texture over a buffer through the image its texture keeps
/// announces.
pub(super) const KEPT_BYTES: [Degradation; 1] = [Degradation::DontCareStandIn];

/// The probe's formats and the bytes of a texel of each.
const FORMATS: [(PixelFormat, u64); 5] = [
    (PixelFormat::BGRA8Unorm, 4),
    (PixelFormat::RGBA8Unorm, 4),
    (PixelFormat::BGRA8UnormSRGB, 4),
    (PixelFormat::RGBA16Float, 8),
    (PixelFormat::R8Unorm, 1),
];

/// The probe's values: the draw's, a stored render's before it, the clear's, and the second
/// attachment's.
const RENDER: f32 = 64.0 / 255.0;
const PRIOR: f32 = 32.0 / 255.0;
const CLEAR: f64 = 0.5;
const SECOND: f32 = 96.0 / 255.0;

/// `MTLResourceStorageModeManaged`.
const MANAGED: u64 = 1 << 4;

/// The probe's functions, written here: a rectangle from corner `r.xy` to corner `r.zw` (two
/// triangles of six vertices), a fragment function returning `c[0]`, and one returning `c[0]` to
/// color 0 and `c[1]` to color 1.
const AIR: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define <{ <4 x float> }> @kb_rect(i32 %vid, ptr addrspace(2) %r) {
  %c = load <4 x float>, ptr addrspace(2) %r, align 16
  %odd = and i32 %vid, -3
  %one_or_three = icmp eq i32 %odd, 1
  %four = icmp eq i32 %vid, 4
  %right = or i1 %one_or_three, %four
  %two = icmp eq i32 %vid, 2
  %pair = and i32 %vid, -2
  %four_or_five = icmp eq i32 %pair, 4
  %up = or i1 %two, %four_or_five
  %x0 = extractelement <4 x float> %c, i64 0
  %y0 = extractelement <4 x float> %c, i64 1
  %x1 = extractelement <4 x float> %c, i64 2
  %y1 = extractelement <4 x float> %c, i64 3
  %x = select i1 %right, float %x1, float %x0
  %y = select i1 %up, float %y1, float %y0
  %px = insertelement <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 1.000000e+00>, float %x, i64 0
  %p = insertelement <4 x float> %px, float %y, i64 1
  %out = insertvalue <{ <4 x float> }> undef, <4 x float> %p, 0
  ret <{ <4 x float> }> %out
}

define <4 x float> @kb_one(<4 x float> %position, ptr addrspace(2) %c) {
  %v = load <4 x float>, ptr addrspace(2) %c, align 16
  ret <4 x float> %v
}

define <{ <4 x float>, <4 x float> }> @kb_two(<4 x float> %position, ptr addrspace(2) %c) {
  %v0 = load <4 x float>, ptr addrspace(2) %c, align 16
  %at1 = getelementptr inbounds <4 x float>, ptr addrspace(2) %c, i64 1
  %v1 = load <4 x float>, ptr addrspace(2) %at1, align 16
  %out0 = insertvalue <{ <4 x float>, <4 x float> }> undef, <4 x float> %v0, 0
  %out = insertvalue <{ <4 x float>, <4 x float> }> %out0, <4 x float> %v1, 1
  ret <{ <4 x float>, <4 x float> }> %out
}

!air.vertex = !{!0}
!air.fragment = !{!5, !10}
!air.version = !{!20}
!air.language_version = !{!21}
!0 = !{ptr @kb_rect, !1, !3}
!1 = !{!2}
!2 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!3 = !{!4, !9}
!4 = !{i32 0, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"vid"}
!9 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"r"}
!5 = !{ptr @kb_one, !6, !8}
!6 = !{!7}
!7 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!8 = !{!13, !14}
!10 = !{ptr @kb_two, !11, !8}
!11 = !{!7, !12}
!12 = !{!"air.render_target", i32 1, i32 0, !"air.arg_type_name", !"float4"}
!13 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"position", !"air.arg_unused"}
!14 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"c"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// What a case's buffer is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// An ordinary `Shared` buffer.
    Shared,
    /// A no-copy buffer over fresh anonymous host pages.
    NoCopy,
    /// A managed no-copy buffer standing for an IOSurface, the texture over its plane 0.
    Surface,
}

/// Where the case's draw lies: the probe's pixel rectangles from the texture's top left corner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Coverage {
    Full,
    Half,
    /// The top left quarter, at most 32x32 texels.
    Corner,
    /// No draw.
    None,
}

impl Coverage {
    /// The far corner of the draw's pixel rectangle over a `w` x `h` texture; `None` for no draw.
    fn corner(self, (w, h): (u64, u64)) -> Option<(u64, u64)> {
        match self {
            Coverage::Full => Some((w, h)),
            Coverage::Half => Some((w / 2, h)),
            Coverage::Corner => Some(((w / 2).min(32), (h / 2).min(32))),
            Coverage::None => None,
        }
    }

    fn covers(self, extent: (u64, u64), x: u64, y: u64) -> bool {
        self.corner(extent).is_some_and(|(x1, y1)| x < x1 && y < y1)
    }
}

/// What comes before the case's pass, or beside it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Variant {
    FirstUse,
    /// A pass loads the texture, draws [`PRIOR`] over all of it and stores it first.
    RenderedOnce,
    /// An ordinary texture of the format is color attachment 1, cleared to zero and stored, the
    /// draw writing [`SECOND`] into it.
    SecondAttachment,
    /// The draw again after a barrier over buffers, from the fragment stage to the vertex stage,
    /// which splits the pass into two render pass instances, the first storing the attachment.
    Split,
    /// The draw again after a barrier over the render targets, from the fragment stage to the
    /// fragment stage, where MoltenVK restarts its Metal pass, storing the attachment.
    Restarted,
}

impl Variant {
    /// The barrier the variant's pass draws again after, and its stages.
    fn barrier(self) -> Option<(BarrierScope, RenderStages)> {
        match self {
            Variant::Split => Some((BarrierScope::BUFFERS, RenderStages::VERTEX)),
            Variant::Restarted => Some((BarrierScope::RENDER_TARGETS, RenderStages::FRAGMENT)),
            _ => None,
        }
    }
}

/// A case's texture: its extent and the bytes between its rows' starts.
#[derive(Clone, Copy, Debug)]
struct Shape {
    extent: (u64, u64),
    rows: u64,
}

/// What a pass counted (`WorkCounts::kept_bytes_passes`, `WorkCounts::untouched_passes_skipped`)
/// and announced.
struct Seen {
    counted: u64,
    skipped: u64,
    notices: BTreeMap<Degradation, usize>,
}

/// A buffer of the pattern and the texture over it.
struct Over {
    buffer: Buffer,
    texture: Texture,
    /// The buffer's bytes.
    len: usize,
    /// Whether the device announced the texture shadowed.
    shadowed: bool,
}

/// The lane's device, its message sink, a queue, the functions, and each format's reference
/// texels and pipelines, made at first use.
struct Lane<'a> {
    device: &'a Device,
    /// The driver stores `DONT_CARE` as Metal's own `DontCare`
    /// (`DriverFact::DontCareStoreIsMetals`).
    metals: bool,
    messages: &'a Messages,
    queue: CommandQueue,
    library: Library,
    formats: BTreeMap<PixelFormat, Format>,
}

/// A format's pipelines (one color attachment, two) and the bytes a texel of it stores for the
/// draw ([`RENDER`]), a clear ([`CLEAR`]), the stored render ([`PRIOR`]) and the second
/// attachment's draw ([`SECOND`]), on the device.
struct Format {
    bpp: u64,
    one: RenderPipelineState,
    two: RenderPipelineState,
    rendered: Vec<u8>,
    cleared: Vec<u8>,
    prior: Vec<u8>,
    second: Vec<u8>,
}

/// A pass's color attachment 0: its texture, load and store actions.
struct Attachment<'t> {
    texture: &'t Texture,
    load: LoadAction,
    store: StoreAction,
}

/// The byte `i` of a case's buffer of texels of `pf`: `(7i + 3) mod 256`, but that an
/// `RGBA16Float` texel's halves are no NaN or infinity (bit 6 of each high byte clear), as a pass
/// that stores the texture may rewrite a NaN's bits on some devices (lavapipe quiets a signalling
/// one, announced `device-nan-encoding`), which is not this lane's question.
fn pattern(pf: PixelFormat, i: usize) -> u8 {
    let byte = (7 * i + 3) as u8;
    match pf == PixelFormat::RGBA16Float && i % 2 == 1 {
        true => byte & 0xbf,
        false => byte,
    }
}

fn floats(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

impl<'a> Lane<'a> {
    fn new(device: &'a Device, messages: &'a Messages) -> Lane<'a> {
        let library = device
            .new_library_with_air(&llvm_assemble_text(AIR).unwrap())
            .unwrap();
        Lane {
            device,
            metals: (device.properties().driver_facts).contains(&DriverFact::DontCareStoreIsMetals),
            messages,
            queue: device.new_command_queue().unwrap(),
            library,
            formats: BTreeMap::new(),
        }
    }

    /// A pipeline drawing [`AIR`]'s rectangle into `colors` attachments of `pf`.
    fn pipeline(&self, pf: PixelFormat, colors: usize) -> RenderPipelineState {
        let function = |name: &str| {
            StageSource::Function(
                self.library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let mut descriptor = RenderPipelineDescriptor::new(function("kb_rect"));
        descriptor.fragment = Some(function(match colors {
            1 => "kb_one",
            _ => "kb_two",
        }));
        for slot in &mut descriptor.color_attachments[..colors] {
            *slot = Some(ColorAttachmentDescriptor::new(pf));
        }
        self.device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap()
    }

    /// `pf`'s pipelines and reference texels, made at first use.
    fn format(&mut self, pf: PixelFormat) -> &Format {
        if !self.formats.contains_key(&pf) {
            let bpp = FORMATS.iter().find(|f| f.0 == pf).unwrap().1;
            let one = self.pipeline(pf, 1);
            let two = self.pipeline(pf, 2);
            let mut format = Format {
                bpp,
                one,
                two,
                rendered: Vec::new(),
                cleared: Vec::new(),
                prior: Vec::new(),
                second: Vec::new(),
            };
            format.rendered = self.reference(&format, pf, Some(RENDER));
            format.cleared = self.reference(&format, pf, None);
            format.prior = self.reference(&format, pf, Some(PRIOR));
            format.second = self.reference(&format, pf, Some(SECOND));
            self.formats.insert(pf, format);
        }
        &self.formats[&pf]
    }

    /// The bytes texel (1, 1) of an ordinary `Shared` 4x4 texture of `pf` holds after a pass that
    /// draws `value` over all of it (a `DontCare` load), or clears it ([`CLEAR`]) where `None`,
    /// and stores it.
    fn reference(&self, format: &Format, pf: PixelFormat, value: Option<f32>) -> Vec<u8> {
        let mut d = TextureDescriptor::texture_2d(pf, 4, 4, false);
        d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        d.resource_options = ResourceOptions::shared();
        let texture = self.device.new_texture(&d).unwrap();
        let (load, coverage) = match value {
            Some(_) => (LoadAction::DontCare, Coverage::Full),
            None => (LoadAction::Clear, Coverage::None),
        };
        let color = Attachment {
            texture: &texture,
            load,
            store: StoreAction::Store,
        };
        self.pass(
            &format.one,
            &color,
            None,
            coverage,
            (value.unwrap_or(0.0), 0.0),
            |_| {},
        );
        let bpp = format.bpp;
        let mut bytes = vec![0; (16 * bpp) as usize];
        texture
            .get_bytes(&mut bytes, 4 * bpp, 0, Region::new_2d(0, 0, 4, 4), 0, 0)
            .unwrap();
        bytes[(5 * bpp) as usize..(6 * bpp) as usize].to_vec()
    }

    /// Encode one pass of `color` (cleared to [`CLEAR`]) and `second` (cleared to zero and
    /// stored), drawing `pipeline`'s rectangle over `coverage` of `color`'s texture in `values`
    /// (colors 0 and 1), `end` encoding the pass's last commands; commit it and wait.
    fn pass(
        &self,
        pipeline: &RenderPipelineState,
        color: &Attachment<'_>,
        second: Option<&Texture>,
        coverage: Coverage,
        (c0, c1): (f32, f32),
        end: impl FnOnce(&mut RenderCommandEncoder),
    ) {
        let mut descriptor = RenderPassDescriptor::new();
        let target = &mut descriptor.color_attachments[0];
        target.texture = Some(color.texture.clone());
        target.load_action = color.load;
        target.store_action = color.store;
        target.clear_color = ClearColor::new(CLEAR, CLEAR, CLEAR, CLEAR);
        if let Some(second) = second {
            let target = &mut descriptor.color_attachments[1];
            target.texture = Some(second.clone());
            target.load_action = LoadAction::Clear;
            target.store_action = StoreAction::Store;
            target.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        }
        self.encode(&descriptor, |e| {
            let (w, h) = (color.texture.width(), color.texture.height());
            if let Some((x1, y1)) = coverage.corner((w, h)) {
                let rect = [
                    -1.0,
                    1.0,
                    2.0 * x1 as f32 / w as f32 - 1.0,
                    1.0 - 2.0 * y1 as f32 / h as f32,
                ];
                e.set_render_pipeline_state(pipeline).unwrap();
                e.set_vertex_bytes(&floats(&rect), 0).unwrap();
                e.set_fragment_bytes(&floats(&[c0, c0, c0, c0, c1, c1, c1, c1]), 0)
                    .unwrap();
                e.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                    .unwrap();
            }
            end(e);
        });
    }

    /// Encode `descriptor`'s pass by `body` in a command buffer of its own, commit it and wait.
    fn encode(
        &self,
        descriptor: &RenderPassDescriptor,
        body: impl FnOnce(&mut RenderCommandEncoder),
    ) {
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(descriptor).unwrap();
        body(&mut encoder);
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        assert_eq!(command_buffer.error(), None);
    }

    /// A buffer of `kind` of the pattern and a texture of `pf` and `shape` over it at offset 0;
    /// `None` where the device imports no host pages.
    fn over(&self, kind: Kind, pf: PixelFormat, shape: Shape) -> Option<Over> {
        let (w, h) = shape.extent;
        let len = (shape.rows * h) as usize;
        let (buffer, len) = match kind {
            Kind::Shared => {
                let options = ResourceOptions::shared();
                (self.device.new_buffer(len as u64, options).unwrap(), len)
            }
            Kind::NoCopy | Kind::Surface => {
                let len = len.next_multiple_of(host_page());
                let mapping = Mapping::anonymous(len);
                let pointer = mapping.pointer();
                let release: Deallocator = Box::new(move |_, _| drop(mapping));
                let options = match kind {
                    Kind::Surface => ResourceOptions::from_raw(MANAGED).unwrap(),
                    _ => ResourceOptions::default(),
                };
                // SAFETY: the mapping stays mapped until the deallocator, which owns it, runs.
                match unsafe { no_copy(self.device, pointer, len, options, Some(release)) } {
                    Ok(buffer) => (buffer, len),
                    Err(error) => {
                        assert!(not_importable(&error), "{error:?}");
                        return None;
                    }
                }
            }
        };
        let bytes: Vec<u8> = (0..len).map(|i| pattern(pf, i)).collect();
        let contents = buffer.contents().unwrap();
        // SAFETY: the buffer's own bytes, which no command uses yet.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), len) };
        buffer.did_modify_range(0..len as u64).unwrap();
        // What making the buffer announced.
        counts(&self.messages.take_notices());
        let mut d = TextureDescriptor::texture_2d(pf, w, h, false);
        d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        d.resource_options = buffer.resource_options();
        let texture = match kind {
            Kind::Surface => buffer
                .new_texture_with_iosurface_plane(&d, IoSurfacePlane::new(0, shape.rows, false)),
            _ => buffer.new_texture(&d, 0, shape.rows),
        }
        .unwrap();
        let shadowed =
            counts(&self.messages.take_notices()).contains_key(&Degradation::ShadowedTexture);
        Some(Over {
            buffer,
            texture,
            len,
            shadowed,
        })
    }

    /// The buffer's bytes: a `Managed` one's after a blit's `synchronize_resource` of it, which
    /// copies the GPU's copy to its pages where the device has no unified memory (and records
    /// nothing where it has).
    fn bytes(&self, over: &Over) -> Vec<u8> {
        if over.buffer.storage_mode() == StorageMode::Managed {
            let command_buffer = self.queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.synchronize_resource((&over.buffer).into()).unwrap();
            blit.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None);
        }
        let contents = over.buffer.contents().unwrap();
        // SAFETY: the buffer's own bytes; no command that writes them is pending.
        unsafe { std::slice::from_raw_parts(contents.as_ptr(), over.len).to_vec() }
    }

    /// Run `pass`: what it counted and announced, which [`Lane::announced`] checks once the
    /// buffer's bytes are.
    fn run(&self, pass: impl FnOnce()) -> Seen {
        let before = self.device.work_counts();
        pass();
        let after = self.device.work_counts();
        Seen {
            counted: after.kept_bytes_passes - before.kept_bytes_passes,
            skipped: after.untouched_passes_skipped - before.untouched_passes_skipped,
            notices: counts(&self.messages.take_notices()),
        }
    }

    /// Check what a pass that rendered the texture of `over` counted and announced (`seen`): one
    /// whose store keeps the buffer's bytes (`keeps`: `DontCare`, `MultisampleResolve`, `Unknown`)
    /// rendered through the image the texture keeps exactly where the texture is not shadowed
    /// (announced and counted once), and nothing else; one that stores it, nothing. But where the
    /// device renders the texture's image only beside other linear ones
    /// (`VK_NV_linear_color_attachment`'s rule) and the pass has a second attachment, the pass
    /// renders it apart (`linear-attachment-copies`), through an image that copies back only what
    /// the pass stores: neither announced nor counted. And where the driver stores `DONT_CARE` as
    /// Metal's own `DontCare` and the pass, one render pass instance the driver runs unbroken,
    /// hands it the attachment so (`whole`: no barrier, no resolve copying its texels), the
    /// pass renders it in place, Metal keeping the bytes: neither announced nor counted. And a
    /// pass that changes no texel records nothing (`render/untouched.rs`): neither announced nor
    /// counted.
    fn announced(&self, case: &str, over: &Over, (keeps, whole): (bool, bool), seen: Seen) {
        let Seen {
            counted,
            skipped,
            notices,
        } = seen;
        let skipped = skipped > 0;
        let apart = notices.contains_key(&Degradation::LinearAttachmentCopies);
        let handed = keeps && whole && self.metals && !over.shadowed;
        let kept = keeps && !over.shadowed && !apart && !handed && !skipped;
        let form = match (keeps, over.shadowed, apart, handed) {
            _ if skipped => "not recorded, as it changes no texel",
            (_, true, _, _) => "shadowed",
            (_, false, true, _) => "apart, beside a second attachment",
            (_, false, false, true) => "in place, handed to Metal's own DontCare",
            (true, false, false, false) => "through the kept image",
            (false, false, false, false) => "in place",
        };
        eprintln!("KEPT-BYTES {case} on {}: {form}", self.device.name());
        assert_eq!(counted, u64::from(kept), "{case}: kept-bytes passes");
        assert_eq!(
            notices.get(&Degradation::DontCareStandIn).copied(),
            kept.then_some(1),
            "{case}: {notices:?}"
        );
        assert!(
            !notices.contains_key(&Degradation::DontCareStore),
            "{case}: {notices:?}"
        );
    }
}

/// Check `got`, a buffer's bytes after a pass over a texture of `shape` and `bpp` at offset 0,
/// against `before`, its bytes before it: texel (x, y) is `want(x, y)` (`None` for `before`'s),
/// every other byte `before`'s.
fn check(
    case: &str,
    got: &[u8],
    before: &[u8],
    (shape, bpp): (Shape, u64),
    want: impl Fn(u64, u64) -> Option<Vec<u8>>,
) {
    let mut expected = before.to_vec();
    let (w, h) = shape.extent;
    for y in 0..h {
        for x in 0..w {
            if let Some(texel) = want(x, y) {
                let at = (y * shape.rows + x * bpp) as usize;
                expected[at..at + bpp as usize].copy_from_slice(&texel);
            }
        }
    }
    if let Some(i) = (0..got.len()).find(|&i| got[i] != expected[i]) {
        let (y, column) = (i as u64 / shape.rows, i as u64 % shape.rows);
        let place = match (y < h, column / bpp) {
            (true, x) if x < w => format!("texel ({x}, {y})"),
            _ => "between the rows".to_owned(),
        };
        let changed = (0..got.len()).filter(|&i| got[i] != expected[i]).count();
        panic!(
            "{case}: byte {i} ({place}) is {:#04x}, Metal's {:#04x} (the buffer's {:#04x}); \
             {changed} bytes differ",
            got[i], expected[i], before[i]
        );
    }
}

/// One case: the pass over a texture of `pf` and `shape` over a buffer of `kind`, `variant`
/// before or beside it, stored `DontCare`, keeps the buffer's bytes; and where `control`, the same
/// pass stored `Store` leaves Metal's bytes (the module docs).
fn case(
    lane: &mut Lane<'_>,
    (pf, kind, shape): (PixelFormat, Kind, Shape),
    (load, coverage, variant): (LoadAction, Coverage, Variant),
    control: bool,
) {
    let (w, h) = shape.extent;
    let name = format!(
        "{pf:?} {w}x{h} (rows of {}) over a {kind:?} buffer, {load:?} load, {coverage:?} draw, \
         {variant:?}",
        shape.rows
    );
    let stores = match control {
        true => &[StoreAction::DontCare, StoreAction::Store][..],
        false => &[StoreAction::DontCare][..],
    };
    for &store in stores {
        let case = format!("{name}, {store:?} store");
        let Some(over) = lane.over(kind, pf, shape) else {
            eprintln!("{case}: no {kind:?} buffer on {}", lane.device.name());
            return;
        };
        let format = lane.format(pf);
        let (bpp, rendered, cleared) =
            (format.bpp, format.rendered.clone(), format.cleared.clone());
        let (one, two) = (format.one.clone(), format.two.clone());
        let (prior, second_texel) = (format.prior.clone(), format.second.clone());
        if variant == Variant::RenderedOnce {
            let before = lane.bytes(&over);
            let color = Attachment {
                texture: &over.texture,
                load: LoadAction::Load,
                store: StoreAction::Store,
            };
            let first = format!("{case}: the stored render before it");
            let seen =
                lane.run(|| lane.pass(&one, &color, None, Coverage::Full, (PRIOR, 0.0), |_| {}));
            check(&first, &lane.bytes(&over), &before, (shape, bpp), |_, _| {
                Some(prior.clone())
            });
            lane.announced(&first, &over, (false, true), seen);
        }
        let second = (variant == Variant::SecondAttachment).then(|| {
            let mut d = TextureDescriptor::texture_2d(pf, w, h, false);
            d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
            d.resource_options = ResourceOptions::shared();
            lane.device.new_texture(&d).unwrap()
        });
        let before = lane.bytes(&over);
        let color = Attachment {
            texture: &over.texture,
            load,
            store,
        };
        let pipeline = match second {
            Some(_) => &two,
            None => &one,
        };
        let keeps = store == StoreAction::DontCare;
        let seen = lane.run(|| {
            lane.pass(
                pipeline,
                &color,
                second.as_ref(),
                coverage,
                (RENDER, SECOND),
                |e| {
                    if let Some((scope, before)) = variant.barrier() {
                        e.memory_barrier_with_scope(scope, RenderStages::FRAGMENT, before);
                        e.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                            .unwrap();
                    }
                },
            )
        });
        // A pass without a draw call that clears nothing records nothing (`render/untouched.rs`).
        let untouched = coverage == Coverage::None
            && load != LoadAction::Clear
            && second.is_none()
            && variant.barrier().is_none();
        assert_eq!(
            seen.skipped,
            u64::from(untouched),
            "{case}: passes not recorded"
        );
        let got = lane.bytes(&over);
        match keeps {
            true => check(&case, &got, &before, (shape, bpp), |_, _| None),
            false => check(&case, &got, &before, (shape, bpp), |x, y| {
                match (coverage.covers(shape.extent, x, y), load) {
                    (true, _) => Some(rendered.clone()),
                    (false, LoadAction::Clear) => Some(cleared.clone()),
                    (false, _) => None,
                }
            }),
        }
        let whole = variant.barrier().is_none();
        lane.announced(&case, &over, (keeps, whole), seen);
        if let (Some(second), Some(_)) = (&second, coverage.corner(shape.extent)) {
            let mut texel = vec![0; bpp as usize];
            second
                .get_bytes(&mut texel, w * bpp, 0, Region::new_2d(0, 0, 1, 1), 0, 0)
                .unwrap();
            assert_eq!(
                texel, second_texel,
                "{case}: the second attachment's texel (0, 0)"
            );
        }
    }
}

/// Run `test` on a validated device announcing [`KEPT_BYTES`], `dont-care-store`,
/// `shadowed-texture` and `linear-attachment-copies` at each occurrence.
fn on_lane(test: impl FnOnce(&mut Lane<'_>)) {
    let each = [
        Degradation::DontCareStandIn,
        Degradation::DontCareStore,
        Degradation::ShadowedTexture,
        Degradation::LinearAttachmentCopies,
    ];
    on_device_with(
        |options| options.announce_each = each.into_iter().collect(),
        |device, messages| {
            // What the device announced as it was made.
            messages.take_notices();
            test(&mut Lane::new(device, messages));
        },
    );
}

const LOADS: [LoadAction; 3] = [LoadAction::Load, LoadAction::Clear, LoadAction::DontCare];

/// The probe's 16x16 texture at tight rows (16-byte aligned); a 64x32 one at rows a multiple of
/// 256 bytes, tight but for `R8Unorm` (the rows at which RADV renders a linear image over the
/// buffer, where it shadows the others); and the probe's 1307x512 at padded rows: 5248 bytes for
/// `BGRA8Unorm`, else the texels' bytes and one more, rounded up to 64.
fn sizes(pf: PixelFormat, bpp: u64) -> [Shape; 3] {
    let padded = match pf {
        PixelFormat::BGRA8Unorm => 5248,
        _ => (1307 * bpp + 1).next_multiple_of(64),
    };
    [
        Shape {
            extent: (16, 16),
            rows: (16 * bpp).next_multiple_of(16),
        },
        Shape {
            extent: (64, 32),
            rows: (64 * bpp).next_multiple_of(256),
        },
        Shape {
            extent: (1307, 512),
            rows: padded,
        },
    ]
}

/// The sweep of one format over ordinary and no-copy buffers (the module docs), each `Load` and
/// `Clear` case controlled by `Store`.
fn sweep(pf: PixelFormat) {
    on_lane(|lane| {
        let bpp = FORMATS.iter().find(|f| f.0 == pf).unwrap().1;
        let [tight, aligned, padded] = sizes(pf, bpp);
        for kind in [Kind::Shared, Kind::NoCopy] {
            for load in LOADS {
                let control = load != LoadAction::DontCare;
                for shape in [tight, aligned] {
                    for coverage in [Coverage::Full, Coverage::Half, Coverage::None] {
                        let at = (load, coverage, Variant::FirstUse);
                        case(lane, (pf, kind, shape), at, control);
                    }
                }
                for coverage in [Coverage::Half, Coverage::Corner] {
                    let at = (load, coverage, Variant::FirstUse);
                    case(lane, (pf, kind, padded), at, control);
                }
                if matches!(pf, PixelFormat::BGRA8Unorm | PixelFormat::RGBA16Float) {
                    let odd = Shape {
                        extent: (333, 77),
                        rows: (333 * bpp).next_multiple_of(16),
                    };
                    let at = (load, Coverage::Corner, Variant::FirstUse);
                    case(lane, (pf, kind, odd), at, false);
                }
            }
        }
    });
}

/// The probe's sweep in `BGRA8Unorm` keeps the buffer's bytes (the module docs).
#[test]
fn dont_care_stores_keep_the_bytes_of_bgra8_unorm_textures_over_buffers() {
    sweep(PixelFormat::BGRA8Unorm);
}

/// The probe's sweep in `RGBA8Unorm` keeps the buffer's bytes (the module docs).
#[test]
fn dont_care_stores_keep_the_bytes_of_rgba8_unorm_textures_over_buffers() {
    sweep(PixelFormat::RGBA8Unorm);
}

/// The probe's sweep in `BGRA8Unorm_sRGB` keeps the buffer's bytes (the module docs).
#[test]
fn dont_care_stores_keep_the_bytes_of_bgra8_unorm_srgb_textures_over_buffers() {
    sweep(PixelFormat::BGRA8UnormSRGB);
}

/// The probe's sweep in `RGBA16Float` keeps the buffer's bytes (the module docs).
#[test]
fn dont_care_stores_keep_the_bytes_of_rgba16_float_textures_over_buffers() {
    sweep(PixelFormat::RGBA16Float);
}

/// The probe's sweep in `R8Unorm` keeps the buffer's bytes (the module docs).
#[test]
fn dont_care_stores_keep_the_bytes_of_r8_unorm_textures_over_buffers() {
    sweep(PixelFormat::R8Unorm);
}

/// After a pass that loads the texture, draws over it and stores it, a pass storing it
/// `DontCare` keeps the stored bytes; beside an ordinary attachment stored `Store`, which holds
/// the draw, it keeps the buffer's (the probe's `rendered-once` and `second-attachment`, in
/// `BGRA8Unorm`, `RGBA16Float` and `R8Unorm`, at 16x16, 64x32 and 1307x512).
#[test]
fn dont_care_stores_keep_the_bytes_after_a_stored_render_and_beside_a_stored_attachment() {
    on_lane(|lane| {
        for (pf, bpp) in [FORMATS[0], FORMATS[3], FORMATS[4]] {
            let [tight, aligned, padded] = sizes(pf, bpp);
            for kind in [Kind::Shared, Kind::NoCopy] {
                for variant in [Variant::RenderedOnce, Variant::SecondAttachment] {
                    for load in LOADS {
                        for shape in [tight, aligned] {
                            let at = (load, Coverage::Full, variant);
                            case(lane, (pf, kind, shape), at, false);
                        }
                    }
                    let at = (LoadAction::Load, Coverage::Half, variant);
                    case(
                        lane,
                        (pf, kind, padded),
                        at,
                        variant == Variant::SecondAttachment,
                    );
                }
            }
        }
    });
}

/// A `BGRA8Unorm` texture over a surface's plane keeps the surface's bytes, at 16x16 over rows of
/// 64 bytes, at 64x32 over rows of 256 and at 1307x512 over rows of 5248, each load action at full,
/// half and no coverage; the `Load` and `Clear` cases controlled by `Store` (the probe's IOSurface
/// cases, and the rows at which RADV renders a linear image over the surface).
#[test]
fn dont_care_stores_keep_the_bytes_of_a_surface() {
    on_lane(|lane| {
        let pf = PixelFormat::BGRA8Unorm;
        for shape in sizes(pf, 4) {
            for load in LOADS {
                for coverage in [Coverage::Full, Coverage::Half, Coverage::None] {
                    let at = (load, coverage, Variant::FirstUse);
                    case(
                        lane,
                        (pf, Kind::Surface, shape),
                        at,
                        load != LoadAction::DontCare,
                    );
                }
            }
        }
    });
}

/// A pass that draws, then draws again after a barrier, storing a texture over a buffer
/// `DontCare`, keeps the buffer's bytes as the unbroken pass does (Metal's answer, measured with
/// barriers over buffers, render targets, textures and the buffer alone: the barrier probe, the
/// M3 Max under macOS 27.0.1, 20 times alike): a barrier over buffers, which splits the pass into
/// two render pass instances, and one over the render targets, where MoltenVK restarts its Metal
/// pass. In `BGRA8Unorm` and `RGBA16Float` over ordinary and no-copy buffers, and `BGRA8Unorm`
/// over a surface's plane, at 16x16 and 64x32, each load action at full and half coverage; the
/// `Load` and `Clear` cases controlled by `Store`.
#[test]
fn dont_care_stores_keep_the_bytes_through_a_barrier() {
    on_lane(|lane| {
        for (pf, bpp) in [FORMATS[0], FORMATS[3]] {
            let [tight, aligned, _] = sizes(pf, bpp);
            let kinds: &[Kind] = match pf {
                PixelFormat::BGRA8Unorm => &[Kind::Shared, Kind::NoCopy, Kind::Surface],
                _ => &[Kind::Shared, Kind::NoCopy],
            };
            for &kind in kinds {
                for variant in [Variant::Split, Variant::Restarted] {
                    for load in LOADS {
                        for shape in [tight, aligned] {
                            for coverage in [Coverage::Full, Coverage::Half] {
                                let at = (load, coverage, variant);
                                let control = load != LoadAction::DontCare;
                                case(lane, (pf, kind, shape), at, control);
                            }
                        }
                    }
                }
            }
        }
    });
}

/// The store actions that store nothing of a texture over a buffer keep its bytes, and those
/// that store it store the draw (the module docs): `Unknown` at the pass's end, set to `DontCare`
/// or `Store` as it ends; a single-sampled `MultisampleResolve` into an ordinary texture, which
/// holds the draw, and `StoreAndMultisampleResolve`. And one 2048x2048 pass keeps the bytes.
#[test]
fn store_actions_that_store_nothing_keep_the_bytes() {
    on_lane(|lane| {
        let (pf, bpp) = FORMATS[0];
        let format = lane.format(pf);
        let (one, rendered) = (format.one.clone(), format.rendered.clone());
        let [tight, aligned, _] = sizes(pf, bpp);
        let cases = [Kind::Shared, Kind::NoCopy].map(|kind| [(kind, tight), (kind, aligned)]);
        for (kind, shape) in cases.into_iter().flatten() {
            let (w, h) = shape.extent;
            for (what, store, late) in [
                ("Unknown", StoreAction::Unknown, None),
                (
                    "Unknown, then DontCare",
                    StoreAction::Unknown,
                    Some(StoreAction::DontCare),
                ),
                (
                    "Unknown, then Store",
                    StoreAction::Unknown,
                    Some(StoreAction::Store),
                ),
            ] {
                let case = format!("{pf:?} over a {kind:?} buffer, {what}");
                let Some(over) = lane.over(kind, pf, shape) else {
                    eprintln!("{case}: no {kind:?} buffer on {}", lane.device.name());
                    continue;
                };
                let before = lane.bytes(&over);
                let color = Attachment {
                    texture: &over.texture,
                    load: LoadAction::Clear,
                    store,
                };
                let stored = late == Some(StoreAction::Store);
                let seen = lane.run(|| {
                    lane.pass(&one, &color, None, Coverage::Half, (RENDER, 0.0), |e| {
                        if let Some(late) = late {
                            e.set_color_store_action(late, 0).unwrap();
                        }
                    })
                });
                let cleared = lane.format(pf).cleared.clone();
                check(
                    &case,
                    &lane.bytes(&over),
                    &before,
                    (shape, bpp),
                    |x, _| match (stored, x < w / 2) {
                        (false, _) => None,
                        (true, true) => Some(rendered.clone()),
                        (true, false) => Some(cleared.clone()),
                    },
                );
                lane.announced(&case, &over, (!stored, true), seen);
            }
            for store in [
                StoreAction::MultisampleResolve,
                StoreAction::StoreAndMultisampleResolve,
            ] {
                let case = format!("{pf:?} over a {kind:?} buffer, {store:?}");
                let Some(over) = lane.over(kind, pf, shape) else {
                    continue;
                };
                let mut d = TextureDescriptor::texture_2d(pf, w, h, false);
                d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
                d.resource_options = ResourceOptions::shared();
                let resolve = lane.device.new_texture(&d).unwrap();
                let before = lane.bytes(&over);
                let mut descriptor = RenderPassDescriptor::new();
                let target = &mut descriptor.color_attachments[0];
                target.texture = Some(over.texture.clone());
                target.load_action = LoadAction::Clear;
                target.store_action = store;
                target.clear_color = ClearColor::new(CLEAR, CLEAR, CLEAR, CLEAR);
                target.resolve_texture = Some(resolve.clone());
                let resolves_alone = store == StoreAction::MultisampleResolve;
                let seen = lane.run(|| {
                    lane.encode(&descriptor, |e| {
                        e.set_render_pipeline_state(&one).unwrap();
                        e.set_vertex_bytes(&floats(&[-1.0, 1.0, 1.0, -1.0]), 0)
                            .unwrap();
                        e.set_fragment_bytes(&floats(&[RENDER; 8]), 0).unwrap();
                        e.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                            .unwrap();
                    })
                });
                check(&case, &lane.bytes(&over), &before, (shape, bpp), |_, _| {
                    (!resolves_alone).then(|| rendered.clone())
                });
                // The resolve copies the attachment's texels, so the pass stores them.
                lane.announced(&case, &over, (resolves_alone, false), seen);
                let mut texels = vec![0; (w * h * bpp) as usize];
                resolve
                    .get_bytes(&mut texels, w * bpp, 0, Region::new_2d(0, 0, w, h), 0, 0)
                    .unwrap();
                assert!(
                    texels.chunks(bpp as usize).all(|texel| texel == rendered),
                    "{case}: the resolve texture holds the draw"
                );
            }
        }
        let big = Shape {
            extent: (2048, 2048),
            rows: 2048 * bpp,
        };
        let at = (LoadAction::Load, Coverage::Half, Variant::FirstUse);
        case(lane, (pf, Kind::Shared, big), at, false);
    });
}

/// A `BGRA8Unorm` 64x32 texture over a no-copy buffer of memfd pages, rows 256 bytes apart (an
/// image over a dma-buf of its own pages on RADV, which imports them so; the pointer elsewhere),
/// keeps the pages' bytes through a pass that clears it, draws over half of it and stores it
/// `DontCare`, and a `Store` control stores the draw and the clear. Skipped where
/// `/dev/udmabuf` cannot be opened.
#[cfg(target_os = "linux")]
#[test]
fn dont_care_stores_keep_the_bytes_of_dma_buf_pages() {
    const TEST: &str = "dma-buf pages";
    if !udmabuf_openable(TEST) {
        return;
    }
    on_lane(|lane| {
        let (pf, bpp) = FORMATS[0];
        let shape = Shape {
            extent: (64, 32),
            rows: 256,
        };
        let (w, h) = shape.extent;
        let len = (shape.rows * h) as usize;
        let format = lane.format(pf);
        let (one, rendered, cleared) = (
            format.one.clone(),
            format.rendered.clone(),
            format.cleared.clone(),
        );
        for store in [StoreAction::DontCare, StoreAction::Store] {
            let case = format!("{TEST}, {store:?} store");
            let ram = Memfd::new(len);
            let pages = ram.view(&[(0, len)]);
            let before: Vec<u8> = (0..len).map(|i| pattern(pf, i)).collect();
            ram.whole.write(0, &before);
            let calls = Calls::default();
            let source = ram.source(&pages, &calls);
            let buffer =
                super::dma_buf_imports::no_copy(lane.device, &pages, 0, len, source).unwrap();
            counts(&lane.messages.take_notices());
            let mut d = TextureDescriptor::texture_2d(pf, w, h, false);
            d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
            d.resource_options = buffer.resource_options();
            let texture = buffer.new_texture(&d, 0, shape.rows).unwrap();
            let shadowed =
                counts(&lane.messages.take_notices()).contains_key(&Degradation::ShadowedTexture);
            let over = Over {
                buffer,
                texture,
                len,
                shadowed,
            };
            let color = Attachment {
                texture: &over.texture,
                load: LoadAction::Clear,
                store,
            };
            let keeps = store == StoreAction::DontCare;
            let seen =
                lane.run(|| lane.pass(&one, &color, None, Coverage::Half, (RENDER, 0.0), |_| {}));
            let got = ram.whole.read(0..len);
            check(&case, &got, &before, (shape, bpp), |x, _| {
                match (keeps, x < w / 2) {
                    (true, _) => None,
                    (false, true) => Some(rendered.clone()),
                    (false, false) => Some(cleared.clone()),
                }
            });
            lane.announced(&case, &over, (keeps, true), seen);
            drop(over);
            eprintln!("{case}: the source's calls {:?}", calls.lock().unwrap());
        }
    });
}
