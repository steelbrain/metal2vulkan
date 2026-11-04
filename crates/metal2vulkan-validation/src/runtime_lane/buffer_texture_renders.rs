//! Device tests of textures over buffers in render passes on the device the cfg selects, under the
//! validation layer. The render cases (`r.*`) of a Metal probe on the M3 Max and an M4 run on the
//! runtime with the probe's functions (the owned fixture `tob_renders`), over its bytes: a buffer
//! whose byte `i` is `7i + 3`, holding 4x4 `RGBA8Unorm` textures at offset 256 with rows 32 bytes
//! apart (and 256 and 16 for the clears), two-slice linear arrays and 16-texel texture buffers,
//! filled with texels that name themselves. The textures are color attachments, resolve targets,
//! and textures a fragment function reads, samples and writes, and their bytes are ordered with
//! blits in one command buffer. Each case runs over an ordinary buffer and a no-copy one, so each
//! texture takes the aliasing or the shadow form the device gives it.
//!
//! Every byte of the buffer, and of the ordinary texture a pass renders into, is checked against
//! Metal's answer, the bytes the probe recorded on the M4 and the M3 Max rebuilt by a model at the
//! probe's layout: a texel a conversion produced (a clear, a draw's color, a resolve, a sample)
//! within one unit of Metal's, as Vulkan leaves the rounding of a unorm tie to the device; every
//! other byte, the bytes between rows among them, exactly.
//!
//! One command that writes a buffer's bytes through a texture over it and also another way keeps
//! both writes, as Metal does (probe cases `k.two_ways_*`): a dispatch that writes a texel and the
//! next texel's bytes or those between rows, and a pass that stores a texture while its draw
//! writes, through another texture over the buffer, the texels between the first's rows. Where a
//! texture is shadowed, the runtime merges its writes into the buffer and announces it once
//! ([`MERGES`]).
//!
//! Where the runtime does not give Metal's answer, the test says so: the texels a draw leaves
//! after a `DontCare` load are the old ones where Metal's tiler leaves zeros (printed). A texture
//! buffer is an attachment and a resolve target through a 2D texture of one row over its bytes, and
//! a view of another format renders in its encoding (`view_attachments.rs` has the sRGB view of a
//! texture over a buffer).

use super::buffer_texture_blits::{hex_rows, texel, texels, Held, Kind, OFF};
use super::degrade_notices::counts;
use super::indirect_host_steps::encoding_state_set;
use super::{
    buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device, Messages,
    ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{
    BarrierScope, IndirectCommandType, PixelFormat, PrimitiveType, RenderStages, ResourceUsage,
    SamplerMinMagFilter, TextureType, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandBuffer, CommandBufferStatus, CommandQueue, Degradation, Device,
    Error, IndirectCommandBufferDescriptor, Library, LoadAction, Origin, PipelineOptions, Region,
    RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState, ResourceOptions, RuntimeGap,
    SamplerState, Size, StoreAction, Texture, TextureDescriptor,
};
use std::path::Path;

/// The probe's usage of the textures it renders.
const RT: TextureUsage = TextureUsage::SHADER_READ
    .union(TextureUsage::SHADER_WRITE)
    .union(TextureUsage::RENDER_TARGET)
    .union(TextureUsage::PIXEL_FORMAT_VIEW);
/// The probe's rows.
const BPR: u64 = 32;
/// A linear array's slice stride at those rows: four rows, rounded up to 128 bytes.
const SLICE: u64 = 128;
const ALL: [u64; 2] = [4, 4];
const LEFT_HALF: [u64; 2] = [2, 4];
/// The probe's clear color and draw color.
const CLEAR: [f32; 4] = [0.25, 0.5, 0.75, 1.0];
const COLOR: [f32; 4] = [0.875, 0.625, 0.375, 0.125];
/// The probe's rectangles: the whole target, and its left half.
const FULL: [f32; 4] = [-1.0, -1.0, 1.0, 1.0];
const LEFT: [f32; 4] = [-1.0, -1.0, 0.0, 1.0];

/// The degrade a command writing a shadowed texture's bytes two ways announces.
pub(super) const MERGES: [Degradation; 1] = [Degradation::ShadowMerge];
/// The rows the probe recorded on the M4 and the M3 Max after `tob_two_ways` wrote texel (0, 0) and
/// the bytes of texel (1, 0) (`k.two_ways_disjoint`), or the four bytes after row 0's texels
/// (`k.two_ways_gap`).
const PROBE_TWO_WAYS_DISJOINT: &str =
    "ff0000ff5a5b5c5d122042c2132043c3737a81888f969da4abb2b9c0c7ced5dc|\
102144c4112145c5122146c6132147c7535a61686f767d848b9299a0a7aeb5bc|\
102248c8112249c912224aca13224bcb333a41484f565d646b727980878e959c|\
10234ccc11234dcd12234ece13234fcf131a21282f363d444b525960676e757c";
const PROBE_TWO_WAYS_GAP: &str =
    "ff0000ff112041c1122042c2132043c35a5b5c5d8f969da4abb2b9c0c7ced5dc|\
102144c4112145c5122146c6132147c7535a61686f767d848b9299a0a7aeb5bc|\
102248c8112249c912224aca13224bcb333a41484f565d646b727980878e959c|\
10234ccc11234dcd12234ece13234fcf131a21282f363d444b525960676e757c";

/// `c` as `RGBA8Unorm` bytes, a tie rounded up as Metal rounds it.
fn unorm(c: [f32; 4]) -> [u8; 4] {
    c.map(|v| (v * 255.0).round() as u8)
}

/// What `tob_write` writes at (x, y).
fn written(x: u64, y: u64) -> [u8; 4] {
    unorm([0.125 * x as f32, 0.125 * y as f32, 0.75, 1.0])
}

/// The probe's resolve of its triangle, (-1, -1), (1, -1), (-1, 0.9), drawn in `COLOR` into a 4x
/// target cleared to opaque black: covered below the diagonal, half covered on it (two samples of
/// four), uncovered above (probe case `r.resolve`).
fn resolved(x: u64, y: u64) -> [u8; 4] {
    match x.cmp(&y) {
        std::cmp::Ordering::Less => unorm(COLOR),
        std::cmp::Ordering::Equal => [0x70, 0x50, 0x30, 0x90],
        std::cmp::Ordering::Greater => [0, 0, 0, 0xff],
    }
}

/// Bytes as Metal leaves them: each exact, or (`loose`) within one unit.
#[derive(Clone)]
struct Expect {
    bytes: Vec<u8>,
    loose: Vec<bool>,
}

impl Expect {
    fn of(bytes: &[u8]) -> Expect {
        Expect {
            bytes: bytes.to_vec(),
            loose: vec![false; bytes.len()],
        }
    }

    /// Texel (x, y) of the region `at`, `extent` of a texture at `offset`, rows `bpr` apart,
    /// becomes `value(x, y)`, within one unit where `loose`.
    fn set(
        &mut self,
        (offset, bpr): (u64, u64),
        ([x0, y0], [w, h]): ([u64; 2], [u64; 2]),
        value: impl Fn(u64, u64) -> [u8; 4],
        loose: bool,
    ) {
        for y in y0..y0 + h {
            for x in x0..x0 + w {
                let at = (offset + y * bpr + 4 * x) as usize;
                self.bytes[at..at + 4].copy_from_slice(&value(x, y));
                self.loose[at..at + 4].fill(loose);
            }
        }
    }

    /// The first byte of `got` that is not Metal's, if any.
    fn differs(&self, got: &[u8]) -> Option<usize> {
        assert_eq!(got.len(), self.bytes.len());
        (0..got.len()).find(|&i| {
            let (g, w) = (got[i], self.bytes[i]);
            if self.loose[i] {
                g.abs_diff(w) > 1
            } else {
                g != w
            }
        })
    }

    fn check(&self, got: &[u8], what: &str) {
        if let Some(i) = self.differs(got) {
            panic!(
                "{what}: byte {i} is {:#04x}, Metal's {:#04x}{}",
                got[i],
                self.bytes[i],
                if self.loose[i] { " (within one)" } else { "" }
            );
        }
    }
}

/// The lane's device, queue and the fixture's library.
struct Lane {
    device: Device,
    queue: CommandQueue,
    library: Library,
}

impl Lane {
    fn new(device: &Device) -> Lane {
        let ll =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/public/engine/tob_renders.ll");
        let bitcode = llvm_assemble(&ll).unwrap();
        Lane {
            queue: device.new_command_queue().unwrap(),
            library: device.new_library_with_air(&bitcode).unwrap(),
            device: device.clone(),
        }
    }

    /// A pipeline of `vertex` and `fragment` drawing into one attachment of `pf`, `samples`.
    fn pipeline(
        &self,
        (vertex, fragment): (&str, &str),
        pf: PixelFormat,
        samples: u32,
    ) -> RenderPipelineState {
        let function = |name| self.library.new_function_with_name(name).unwrap();
        let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(
            function(vertex).function_descriptor(),
        ));
        descriptor.fragment = Some(StageSource::Function(
            function(fragment).function_descriptor(),
        ));
        descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(pf));
        descriptor.raster_sample_count = samples;
        self.device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap()
    }

    fn floats(&self, values: &[f32]) -> Buffer {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        buffer_with(&self.device, &bytes).unwrap()
    }

    /// One draw of `pipeline`'s rectangle `rect` (`tob_rect`), its fragment's color `color`
    /// where it takes one (`tob_solid`).
    fn rect(
        &self,
        e: &mut RenderCommandEncoder,
        held: &mut Vec<Buffer>,
        pipeline: &RenderPipelineState,
        rect: [f32; 4],
        color: Option<[f32; 4]>,
    ) {
        e.set_render_pipeline_state(pipeline).unwrap();
        let corners = self.floats(&rect);
        e.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
        held.push(corners);
        if let Some(color) = color {
            let color = self.floats(&color);
            e.set_fragment_buffer(Some(&color), 0, 0).unwrap();
            held.push(color);
        }
        e.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
            .unwrap();
    }

    /// The probe's resolve triangle in `COLOR` (`tob_triangle`, `tob_solid`).
    fn triangle(
        &self,
        e: &mut RenderCommandEncoder,
        held: &mut Vec<Buffer>,
        pipeline: &RenderPipelineState,
    ) {
        e.set_render_pipeline_state(pipeline).unwrap();
        let points = self.floats(&[-1.0, -1.0, 1.0, -1.0, -1.0, 0.9]);
        let color = self.floats(&COLOR);
        e.set_vertex_buffer(Some(&points), 0, 0).unwrap();
        e.set_fragment_buffer(Some(&color), 0, 0).unwrap();
        e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
        held.extend([points, color]);
    }

    /// Encode encoders into one command buffer by `body`, commit it and wait.
    fn commit(&self, body: impl FnOnce(&CommandBuffer, &mut Vec<Buffer>)) {
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut held = Vec::new();
        body(&command_buffer, &mut held);
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        assert_eq!(command_buffer.error(), None);
    }

    /// Encode `descriptor`'s pass by `body` in a command buffer of its own; the encoder's refusal
    /// of the pass, if any.
    fn run(
        &self,
        descriptor: &RenderPassDescriptor,
        body: impl FnOnce(&Lane, &mut RenderCommandEncoder, &mut Vec<Buffer>),
    ) -> Result<(), Error> {
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(descriptor)?;
        let mut held = Vec::new();
        body(self, &mut encoder, &mut held);
        encoder.end_encoding()?;
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        Ok(())
    }

    /// An ordinary `Shared` texture of `pf` and the probe's usage, holding set `s`'s texels.
    fn ordinary(&self, pf: PixelFormat, [w, h]: [u64; 2], set: Option<u64>) -> Texture {
        let mut d = TextureDescriptor::texture_2d(pf, w, h, false);
        d.usage = RT;
        d.resource_options = ResourceOptions::shared();
        let texture = self.device.new_texture(&d).unwrap();
        if let Some(s) = set {
            texture
                .replace_region(Region::new_2d(0, 0, w, h), 0, 0, &texels(w, h, s), 4 * w, 0)
                .unwrap();
        }
        texture
    }

    /// A private 4x multisampled 4x4 `RGBA8Unorm` render target.
    fn msaa(&self) -> Texture {
        let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 4, 4, false);
        d.texture_type = TextureType::Type2DMultisample;
        d.sample_count = 4;
        d.usage = TextureUsage::RENDER_TARGET;
        d.resource_options = ResourceOptions::private();
        self.device.new_texture(&d).unwrap()
    }

    fn sampler(&self, filter: SamplerMinMagFilter) -> SamplerState {
        let mut d = SamplerDescriptor::new();
        d.min_filter = filter;
        d.mag_filter = filter;
        self.device.new_sampler_state(&d).unwrap()
    }
}

/// The texels of an ordinary texture.
fn texture_bytes(texture: &Texture, [w, h]: [u64; 2]) -> Vec<u8> {
    let mut bytes = vec![0; (4 * w * h) as usize];
    texture
        .get_bytes(&mut bytes, 4 * w, 0, Region::new_2d(0, 0, w, h), 0, 0)
        .unwrap();
    bytes
}

/// A pass of one color attachment, `texture`, loaded by `load` (cleared to `clear`) and stored
/// by `store`.
fn pass(
    texture: &Texture,
    load: LoadAction,
    store: StoreAction,
    clear: [f32; 4],
) -> RenderPassDescriptor {
    let mut descriptor = RenderPassDescriptor::new();
    let target = &mut descriptor.color_attachments[0];
    target.texture = Some(texture.clone());
    target.load_action = load;
    target.store_action = store;
    let [r, g, b, a] = clear.map(f64::from);
    target.clear_color = ClearColor::new(r, g, b, a);
    descriptor
}

/// Run `test` on a validated device of the lane, then fail on any validation message.
fn on_device(test: impl FnOnce(&Device)) {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        test(&device);
    }
    layer_reported_nothing_at_all(&messages);
}

/// Run `cases` over each kind of buffer the device makes.
fn each_kind(device: &Device, cases: impl Fn(&Lane, Kind)) {
    let lane = Lane::new(device);
    for kind in [Kind::Ordinary, Kind::NoCopy] {
        if Held::new(device, kind).is_none() {
            eprintln!(
                "{}: no {kind:?} buffer (the device imports none)",
                device.name()
            );
            continue;
        }
        cases(&lane, kind);
    }
}

/// A texture over a buffer as a color attachment stores Metal's bytes: clears at rows of 32, 256
/// and 16 bytes write the texel rows alone; a `DontCare` store leaves the bytes, after a clear with
/// no draw, a clear and a draw over the whole texture, a `DontCare` load and a draw over its left
/// half, and a load and a draw over the whole (the DontCare-store probe, an M3 Max under macOS
/// 27.0.1, 20 times each alike); a load keeps the texels a draw leaves; a view of the texture's own
/// format renders into it; slice 1 of a linear array renders there (probe cases `r.clear*`,
/// `r.load_draw`, `r.clear_draw`, `r.dontcare_full`, `r.view_same`, `r.slice1`). After a `DontCare`
/// load, the texels outside the draw are the buffer's or zero (Metal's), which is printed
/// (`r.dontcare_half`).
#[test]
fn attachments_over_buffers_store_metals_bytes() {
    on_device(|device| each_kind(device, attachments));
}

fn attachments(lane: &Lane, kind: Kind) {
    let device = &lane.device;
    let solid = lane.pipeline(("tob_rect", "tob_solid"), PixelFormat::RGBA8Unorm, 1);
    let name = |case: &str| format!("r.{case} over a {kind:?} buffer");
    let over = |b: &Held, bpr: u64| b.texture_of(PixelFormat::RGBA8Unorm, RT, (OFF, bpr), ALL);
    for (case, bpr) in [("clear", BPR), ("clear_bpr256", 256), ("clear_packed", 16)] {
        let b = Held::new(device, kind).unwrap();
        let t = over(&b, bpr);
        lane.run(
            &pass(&t, LoadAction::Clear, StoreAction::Store, CLEAR),
            |_, _, _| {},
        )
        .unwrap();
        let mut want = Expect::of(&b.model);
        want.set((OFF, bpr), ([0, 0], ALL), |_, _| unorm(CLEAR), true);
        want.check(&b.bytes(), &name(case));
    }
    {
        let b = Held::new(device, kind).unwrap();
        let t = over(&b, BPR);
        let descriptor = pass(&t, LoadAction::Clear, StoreAction::DontCare, CLEAR);
        lane.run(&descriptor, |_, _, _| {}).unwrap();
        Expect::of(&b.model).check(&b.bytes(), &name("clear_dontcare"));
    }
    for (case, load, rect) in [
        ("clear_draw_dontcare", LoadAction::Clear, FULL),
        ("dontcare_half_dontcare", LoadAction::DontCare, LEFT),
        ("load_draw_dontcare", LoadAction::Load, FULL),
    ] {
        let b = Held::new(device, kind).unwrap();
        let t = over(&b, BPR);
        let descriptor = pass(&t, load, StoreAction::DontCare, CLEAR);
        lane.run(&descriptor, |l, e, h| {
            l.rect(e, h, &solid, rect, Some(COLOR))
        })
        .unwrap();
        Expect::of(&b.model).check(&b.bytes(), &name(case));
    }
    for case in ["load_draw", "clear_draw", "dontcare_full", "view_same"] {
        let mut b = Held::new(device, kind).unwrap();
        if case != "clear_draw" {
            b.fill(OFF, BPR, ALL, 0);
        }
        let t = over(&b, BPR);
        let target = match case {
            "view_same" => t.new_texture_view(PixelFormat::RGBA8Unorm).unwrap(),
            _ => t.clone(),
        };
        let (load, rect, drawn) = match case {
            "clear_draw" => (LoadAction::Clear, LEFT, LEFT_HALF),
            "dontcare_full" => (LoadAction::DontCare, FULL, ALL),
            _ => (LoadAction::Load, LEFT, LEFT_HALF),
        };
        lane.run(
            &pass(&target, load, StoreAction::Store, CLEAR),
            |l, e, h| l.rect(e, h, &solid, rect, Some(COLOR)),
        )
        .unwrap();
        let mut want = Expect::of(&b.model);
        if load == LoadAction::Clear {
            want.set((OFF, BPR), ([0, 0], ALL), |_, _| unorm(CLEAR), true);
        }
        want.set((OFF, BPR), ([0, 0], drawn), |_, _| unorm(COLOR), true);
        want.check(&b.bytes(), &name(case));
    }
    {
        let mut b = Held::new(device, kind).unwrap();
        b.fill(OFF, BPR, ALL, 0);
        let t = over(&b, BPR);
        let descriptor = pass(&t, LoadAction::DontCare, StoreAction::Store, CLEAR);
        lane.run(&descriptor, |l, e, h| {
            l.rect(e, h, &solid, LEFT, Some(COLOR))
        })
        .unwrap();
        let mut kept = Expect::of(&b.model);
        kept.set((OFF, BPR), ([0, 0], LEFT_HALF), |_, _| unorm(COLOR), true);
        let mut zeroed = kept.clone();
        zeroed.set((OFF, BPR), ([2, 0], LEFT_HALF), |_, _| [0; 4], false);
        let bytes = b.bytes();
        let (is_kept, is_zeroed) = (
            kept.differs(&bytes).is_none(),
            zeroed.differs(&bytes).is_none(),
        );
        assert!(
            is_kept || is_zeroed,
            "{}: the draw, or the texels outside it, are neither the buffer's nor zero",
            name("dontcare_half")
        );
        eprintln!(
            "{} on {}: the texels outside the draw {}",
            name("dontcare_half"),
            device.name(),
            if is_zeroed {
                "are zero, as on Metal"
            } else {
                "keep the buffer's bytes (Metal leaves zeros)"
            }
        );
    }
    {
        let mut b = Held::new(device, kind).unwrap();
        b.fill(OFF, BPR, ALL, 0);
        b.fill(OFF + SLICE, BPR, ALL, 1);
        let array = b.array_of(RT, OFF, BPR);
        let mut descriptor = pass(&array, LoadAction::Load, StoreAction::Store, CLEAR);
        descriptor.color_attachments[0].slice = 1;
        lane.run(&descriptor, |l, e, h| {
            l.rect(e, h, &solid, LEFT, Some(COLOR))
        })
        .unwrap();
        let mut want = Expect::of(&b.model);
        want.set(
            (OFF + SLICE, BPR),
            ([0, 0], LEFT_HALF),
            |_, _| unorm(COLOR),
            true,
        );
        want.check(&b.bytes(), &name("slice1"));
    }
}

/// A multisample resolve into a texture over a buffer, and into slice 1 of a linear array over one,
/// writes Metal's resolved texels (probe cases `r.resolve`, `r.resolve_store` and
/// `r.resolve_slice`).
#[test]
fn resolves_into_textures_over_buffers_write_metals_bytes() {
    on_device(|device| each_kind(device, resolves));
}

fn resolves(lane: &Lane, kind: Kind) {
    let device = &lane.device;
    let pipeline = lane.pipeline(("tob_triangle", "tob_solid"), PixelFormat::RGBA8Unorm, 4);
    let name = |case: &str| format!("r.{case} over a {kind:?} buffer");
    let black = [0.0, 0.0, 0.0, 1.0];
    for (case, store) in [
        ("resolve", StoreAction::MultisampleResolve),
        ("resolve_store", StoreAction::StoreAndMultisampleResolve),
    ] {
        let b = Held::new(device, kind).unwrap();
        let t = b.texture_of(PixelFormat::RGBA8Unorm, RT, (OFF, BPR), ALL);
        let msaa = lane.msaa();
        let mut descriptor = pass(&msaa, LoadAction::Clear, store, black);
        descriptor.color_attachments[0].resolve_texture = Some(t.clone());
        lane.run(&descriptor, |l, e, h| l.triangle(e, h, &pipeline))
            .unwrap();
        let mut want = Expect::of(&b.model);
        want.set((OFF, BPR), ([0, 0], ALL), resolved, true);
        want.check(&b.bytes(), &name(case));
    }
    let mut b = Held::new(device, kind).unwrap();
    b.fill(OFF, BPR, ALL, 0);
    b.fill(OFF + SLICE, BPR, ALL, 1);
    let array = b.array_of(RT, OFF, BPR);
    let msaa = lane.msaa();
    let mut descriptor = pass(
        &msaa,
        LoadAction::Clear,
        StoreAction::MultisampleResolve,
        black,
    );
    descriptor.color_attachments[0].resolve_texture = Some(array.clone());
    descriptor.color_attachments[0].resolve_slice = 1;
    lane.run(&descriptor, |l, e, h| l.triangle(e, h, &pipeline))
        .unwrap();
    let mut want = Expect::of(&b.model);
    want.set((OFF + SLICE, BPR), ([0, 0], ALL), resolved, true);
    want.check(&b.bytes(), &name("resolve_slice"));
}

/// Fragment functions read, sample (nearest and linear) and write textures over buffers and texture
/// buffers as Metal does (probe cases `r.read`, `r.sample`, `r.sample_linear`, `r.read_tb`,
/// `r.frag_write`, `r.frag_write_tb`, and `r.efi`, the consumer's EFI frame: a 20x4 `BGRA8Unorm`
/// texture at offset 0 with packed rows, read alone).
#[test]
fn fragment_functions_read_and_write_textures_over_buffers() {
    on_device(|device| each_kind(device, fragments));
}

fn fragments(lane: &Lane, kind: Kind) {
    let device = &lane.device;
    let name = |case: &str| format!("r.{case} over a {kind:?} buffer");
    let rgba = PixelFormat::RGBA8Unorm;
    let read = lane.pipeline(("tob_rect", "tob_read"), rgba, 1);
    let sample = lane.pipeline(("tob_rect", "tob_sample"), rgba, 1);
    let read_buffer = lane.pipeline(("tob_rect", "tob_read_buffer"), rgba, 1);
    let write = lane.pipeline(("tob_rect", "tob_write"), rgba, 1);
    let write_buffer = lane.pipeline(("tob_rect", "tob_write_buffer"), rgba, 1);
    // Each state states whether its fragment function writes a texture: a write into a 2D
    // texture or a texture buffer counts; a read and a sample do not, nor do the vertex
    // function's outputs, nor the color the fragment function returns.
    for (state, writes) in [
        (&read, false),
        (&sample, false),
        (&read_buffer, false),
        (&write, true),
        (&write_buffer, true),
    ] {
        assert_eq!(state.fragment_writes_textures(), writes);
        assert!(!state.vertex_writes_textures() && !state.tile_writes_textures());
        assert!(!state.object_writes_textures() && !state.mesh_writes_textures());
    }
    let nearest = lane.sampler(SamplerMinMagFilter::Nearest);
    let linear = lane.sampler(SamplerMinMagFilter::Linear);
    // Into an ordinary texture, cleared to zero, through `pipeline` reading `texture`.
    let into_ordinary = |pipeline: &RenderPipelineState,
                         texture: &Texture,
                         sampler: Option<&SamplerState>,
                         (pf, extent): (PixelFormat, [u64; 2])| {
        let o = lane.ordinary(pf, extent, None);
        lane.run(
            &pass(&o, LoadAction::Clear, StoreAction::Store, [0.0; 4]),
            |l, e, h| {
                e.set_fragment_texture(Some(texture), 0).unwrap();
                e.set_fragment_sampler_state(sampler, 0).unwrap();
                l.rect(e, h, pipeline, FULL, None);
            },
        )
        .unwrap();
        texture_bytes(&o, extent)
    };
    for (case, pipeline, sampler) in [
        ("read", &read, None),
        ("sample", &sample, Some(&nearest)),
        ("sample_linear", &sample, Some(&linear)),
    ] {
        let mut b = Held::new(device, kind).unwrap();
        b.fill(OFF, BPR, ALL, 0);
        let t = b.texture_of(rgba, RT, (OFF, BPR), ALL);
        let o = into_ordinary(pipeline, &t, sampler, (rgba, ALL));
        let mut want = Expect::of(&texels(4, 4, 0));
        want.set(
            (0, 16),
            ([0, 0], ALL),
            |x, y| texel(x, y, 0),
            sampler.is_some(),
        );
        want.check(&o, &name(case));
        Expect::of(&b.model).check(&b.bytes(), &name(case));
    }
    {
        let mut b = Held::new(device, kind).unwrap();
        b.fill(OFF, 64, [16, 1], 0);
        let tb = b.texture_buffer_of(RT, OFF, 16);
        let o = into_ordinary(&read_buffer, &tb, None, (rgba, ALL));
        Expect::of(&texels(16, 1, 0)).check(&o, &name("read_tb"));
        Expect::of(&b.model).check(&b.bytes(), &name("read_tb"));
    }
    for (case, buffer) in [("frag_write", false), ("frag_write_tb", true)] {
        let b = Held::new(device, kind).unwrap();
        let (t, pipeline, bpr) = match buffer {
            false => (b.texture_of(rgba, RT, (OFF, BPR), ALL), &write, BPR),
            true => (b.texture_buffer_of(RT, OFF, 16), &write_buffer, 16),
        };
        let o = into_ordinary(pipeline, &t, None, (rgba, ALL));
        let mut black = Expect::of(&o);
        black.set((0, 16), ([0, 0], ALL), |_, _| [0, 0, 0, 0xff], false);
        black.check(&o, &name(case));
        let mut want = Expect::of(&b.model);
        want.set((OFF, bpr), ([0, 0], ALL), written, true);
        want.check(&b.bytes(), &name(case));
    }
    {
        let mut b = Held::new(device, kind).unwrap();
        let efi = [20, 4];
        b.fill(0, 80, efi, 0);
        let bgra = PixelFormat::BGRA8Unorm;
        let read_bgra = lane.pipeline(("tob_rect", "tob_read"), bgra, 1);
        let t = b.texture_of(bgra, TextureUsage::SHADER_READ, (0, 80), efi);
        let o = into_ordinary(&read_bgra, &t, None, (bgra, efi));
        Expect::of(&texels(20, 4, 0)).check(&o, &name("efi"));
        Expect::of(&b.model).check(&b.bytes(), &name("efi"));
    }
    // A draw writes the texture, a memory barrier splits the pass, and a draw reads its bytes
    // through a texture buffer over them: what the first wrote (Metal's barrier makes the writes
    // visible), through copies between the instances where the texture is shadowed.
    {
        let b = Held::new(device, kind).unwrap();
        let t = b.texture_of(rgba, RT, (OFF, 16), ALL);
        let tb = b.texture_buffer_of(RT, OFF, 16);
        let o = lane.ordinary(rgba, ALL, None);
        lane.run(
            &pass(&o, LoadAction::Clear, StoreAction::Store, [0.0; 4]),
            |l, e, h| {
                e.set_fragment_texture(Some(&t), 0).unwrap();
                l.rect(e, h, &write, FULL, None);
                e.memory_barrier_with_scope(
                    BarrierScope::TEXTURES,
                    RenderStages::FRAGMENT,
                    RenderStages::FRAGMENT,
                );
                e.set_fragment_texture(Some(&tb), 0).unwrap();
                l.rect(e, h, &read_buffer, FULL, None);
            },
        )
        .unwrap();
        let what = format!("a write, a barrier and a read over a {kind:?} buffer");
        let mut want = Expect::of(&b.model);
        want.set((OFF, 16), ([0, 0], ALL), written, true);
        want.check(&b.bytes(), &what);
        let mut seen = Expect::of(&texture_bytes(&o, ALL));
        seen.set((0, 16), ([0, 0], ALL), written, true);
        seen.check(&texture_bytes(&o, ALL), &what);
    }
}

/// A pass's bytes are ordered with the other encoders' uses of the buffer in one command buffer: a
/// pass, then a pass that reads the texture; a pass, then a blit of the buffer's bytes; a blit into
/// the bytes, then a pass that reads them; a blit into the texture, then a pass that loads it; a
/// pass, then a blit out of the texture (probe cases `r.chain`, `r.then_kernel` and
/// `r.kernel_then_read` with blits for the kernels, `r.blit_then_attach`, `r.attach_then_blit`).
#[test]
fn passes_over_buffers_order_with_the_other_uses_of_the_bytes() {
    on_device(|device| each_kind(device, ordered));
}

fn ordered(lane: &Lane, kind: Kind) {
    let device = &lane.device;
    let name = |case: &str| format!("r.{case} over a {kind:?} buffer");
    let rgba = PixelFormat::RGBA8Unorm;
    let solid = lane.pipeline(("tob_rect", "tob_solid"), rgba, 1);
    let read = lane.pipeline(("tob_rect", "tob_read"), rgba, 1);
    let cleared_left = |want: &mut Expect, (offset, bpr): (u64, u64)| {
        want.set((offset, bpr), ([0, 0], ALL), |_, _| unorm(CLEAR), true);
        want.set(
            (offset, bpr),
            ([0, 0], LEFT_HALF),
            |_, _| unorm(COLOR),
            true,
        );
    };
    {
        let b = Held::new(device, kind).unwrap();
        let t = b.texture_of(rgba, RT, (OFF, BPR), ALL);
        let o = lane.ordinary(rgba, ALL, None);
        lane.commit(|command_buffer, held| {
            let first = pass(&t, LoadAction::Clear, StoreAction::Store, CLEAR);
            let mut e = command_buffer.render_command_encoder(&first).unwrap();
            lane.rect(&mut e, held, &solid, LEFT, Some(COLOR));
            e.end_encoding().unwrap();
            let second = pass(&o, LoadAction::Clear, StoreAction::Store, [0.0; 4]);
            let mut f = command_buffer.render_command_encoder(&second).unwrap();
            f.set_fragment_texture(Some(&t), 0).unwrap();
            lane.rect(&mut f, held, &read, FULL, None);
            f.end_encoding().unwrap();
        });
        let mut want = Expect::of(&b.model);
        cleared_left(&mut want, (OFF, BPR));
        want.check(&b.bytes(), &name("chain"));
        let mut seen = Expect::of(&texture_bytes(&o, ALL));
        cleared_left(&mut seen, (0, 16));
        seen.check(&texture_bytes(&o, ALL), &name("chain"));
    }
    {
        let b = Held::new(device, kind).unwrap();
        let t = b.texture_of(rgba, RT, (OFF, BPR), ALL);
        let out = buffer_with(device, &[0xEE; 128]).unwrap();
        lane.commit(|command_buffer, _| {
            let first = pass(&t, LoadAction::Clear, StoreAction::Store, CLEAR);
            command_buffer
                .render_command_encoder(&first)
                .unwrap()
                .end_encoding()
                .unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.copy_from_buffer_to_buffer(&b.buffer, OFF, &out, 0, 128)
                .unwrap();
            blit.end_encoding().unwrap();
        });
        let mut want = Expect::of(&b.model[OFF as usize..OFF as usize + 128]);
        want.set((0, BPR), ([0, 0], ALL), |_, _| unorm(CLEAR), true);
        want.check(&read_buffer(&out, 0, 128).unwrap(), &name("then_kernel"));
    }
    {
        let b = Held::new(device, kind).unwrap();
        let t = b.texture_of(rgba, RT, (OFF, BPR), ALL);
        let set: Vec<u8> = (0..128u8).map(|g| 0x60 + (g & 0x1f)).collect();
        let source = buffer_with(device, &set).unwrap();
        let o = lane.ordinary(rgba, ALL, None);
        lane.commit(|command_buffer, held| {
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.copy_from_buffer_to_buffer(&source, 0, &b.buffer, OFF, 128)
                .unwrap();
            blit.end_encoding().unwrap();
            let second = pass(&o, LoadAction::Clear, StoreAction::Store, [0.0; 4]);
            let mut f = command_buffer.render_command_encoder(&second).unwrap();
            f.set_fragment_texture(Some(&t), 0).unwrap();
            lane.rect(&mut f, held, &read, FULL, None);
            f.end_encoding().unwrap();
        });
        let row: Vec<u8> = (0x60..0x70).collect();
        Expect::of(&row.repeat(4)).check(&texture_bytes(&o, ALL), &name("kernel_then_read"));
    }
    {
        let b = Held::new(device, kind).unwrap();
        let t = b.texture_of(rgba, RT, (OFF, BPR), ALL);
        let o = lane.ordinary(rgba, ALL, Some(1));
        lane.commit(|command_buffer, held| {
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.copy_from_texture_to_texture(
                &o,
                0,
                0,
                Origin::new(0, 0, 0),
                Size::new(4, 4, 1),
                &t,
                0,
                0,
                Origin::new(0, 0, 0),
            )
            .unwrap();
            blit.end_encoding().unwrap();
            let second = pass(&t, LoadAction::Load, StoreAction::Store, CLEAR);
            let mut f = command_buffer.render_command_encoder(&second).unwrap();
            lane.rect(&mut f, held, &solid, LEFT, Some(COLOR));
            f.end_encoding().unwrap();
        });
        let mut want = Expect::of(&b.model);
        want.set((OFF, BPR), ([0, 0], ALL), |x, y| texel(x, y, 1), false);
        want.set((OFF, BPR), ([0, 0], LEFT_HALF), |_, _| unorm(COLOR), true);
        want.check(&b.bytes(), &name("blit_then_attach"));
    }
    {
        let b = Held::new(device, kind).unwrap();
        let t = b.texture_of(rgba, RT, (OFF, BPR), ALL);
        let o = lane.ordinary(rgba, ALL, None);
        lane.commit(|command_buffer, _| {
            let first = pass(&t, LoadAction::Clear, StoreAction::Store, CLEAR);
            command_buffer
                .render_command_encoder(&first)
                .unwrap()
                .end_encoding()
                .unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.copy_from_texture_to_texture(
                &t,
                0,
                0,
                Origin::new(0, 0, 0),
                Size::new(4, 4, 1),
                &o,
                0,
                0,
                Origin::new(0, 0, 0),
            )
            .unwrap();
            blit.end_encoding().unwrap();
        });
        let mut want = Expect::of(&texture_bytes(&o, ALL));
        want.set((0, 16), ([0, 0], ALL), |_, _| unorm(CLEAR), true);
        want.check(&texture_bytes(&o, ALL), &name("attach_then_blit"));
    }
}

/// A texture buffer is an attachment and a resolve target, rendered through a 2D texture of one
/// row over its bytes (its attachment twin), as Metal renders into one plain (a probe, an M3 Max
/// and an M4: `tb_over_buffer`, `tb_over_buffer_load`, `tb_resolve`, `tb_device`; its API
/// validation stops a texture buffer made with render target usage at creation). Over each kind
/// of buffer, a 16-texel texture buffer at offset 256 cleared and its left half drawn holds the
/// draw's 8 texels and the clear's 8, every other byte of the buffer kept; loaded, the texels
/// the draw leaves are the buffer's; resolved into from a 16x1 multisampled attachment, it
/// holds the resolve. A texture buffer the device made renders the same.
#[test]
fn texture_buffers_are_attachments() {
    on_device(|device| each_kind(device, texture_buffers));
    on_device(|device| {
        let lane = Lane::new(device);
        let rgba = PixelFormat::RGBA8Unorm;
        let solid = lane.pipeline(("tob_rect", "tob_solid"), rgba, 1);
        let d = TextureDescriptor::texture_buffer(rgba, 16, ResourceOptions::shared(), RT);
        let tb = device.new_texture(&d).unwrap();
        lane.run(
            &pass(&tb, LoadAction::Clear, StoreAction::Store, CLEAR),
            |l, e, h| l.rect(e, h, &solid, LEFT, Some(COLOR)),
        )
        .unwrap();
        let mut texels = vec![0; 64];
        tb.get_bytes(&mut texels, 64, 0, Region::new_1d(0, 16), 0, 0)
            .unwrap();
        let mut want = Expect::of(&[0; 64]);
        want.set((0, 64), ([0, 0], [8, 1]), |_, _| unorm(COLOR), true);
        want.set((0, 64), ([8, 0], [8, 1]), |_, _| unorm(CLEAR), true);
        want.check(&texels, "tb_device");
    });
}

fn texture_buffers(lane: &Lane, kind: Kind) {
    let device = &lane.device;
    let rgba = PixelFormat::RGBA8Unorm;
    let solid = lane.pipeline(("tob_rect", "tob_solid"), rgba, 1);
    let name = |case: &str| format!("{case} over a {kind:?} buffer");
    let row = (OFF, 64);
    for (case, load) in [
        ("tb_over_buffer", LoadAction::Clear),
        ("tb_over_buffer_load", LoadAction::Load),
    ] {
        let b = Held::new(device, kind).unwrap();
        let tb = b.texture_buffer_of(RT, OFF, 16);
        lane.run(&pass(&tb, load, StoreAction::Store, CLEAR), |l, e, h| {
            l.rect(e, h, &solid, LEFT, Some(COLOR))
        })
        .unwrap();
        let mut want = Expect::of(&b.model);
        want.set(row, ([0, 0], [8, 1]), |_, _| unorm(COLOR), true);
        if load == LoadAction::Clear {
            want.set(row, ([8, 0], [8, 1]), |_, _| unorm(CLEAR), true);
        }
        want.check(&b.bytes(), &name(case));
    }
    let b = Held::new(device, kind).unwrap();
    let tb = b.texture_buffer_of(RT, OFF, 16);
    let mut d = TextureDescriptor::texture_2d(rgba, 16, 1, false);
    d.texture_type = TextureType::Type2DMultisample;
    d.sample_count = 4;
    d.usage = TextureUsage::RENDER_TARGET;
    d.resource_options = ResourceOptions::private();
    let msaa = device.new_texture(&d).unwrap();
    let pipeline = lane.pipeline(("tob_rect", "tob_solid"), rgba, 4);
    let mut descriptor = pass(
        &msaa,
        LoadAction::Clear,
        StoreAction::MultisampleResolve,
        CLEAR,
    );
    descriptor.color_attachments[0].resolve_texture = Some(tb.clone());
    lane.run(&descriptor, |l, e, h| {
        l.rect(e, h, &pipeline, LEFT, Some(COLOR))
    })
    .unwrap();
    let mut want = Expect::of(&b.model);
    want.set(row, ([0, 0], [8, 1]), |_, _| unorm(COLOR), true);
    want.set(row, ([8, 0], [8, 1]), |_, _| unorm(CLEAR), true);
    want.check(&b.bytes(), &name("tb_resolve"));
}

/// One command writes a buffer's bytes through a texture over it and another way, and Metal keeps
/// both writes: `tob_two_ways` writes texel (0, 0) of a texture over the buffer and four bytes of
/// the buffer, those of texel (1, 0) or those after row 0's texels, in one dispatch, leaving the
/// rows the probe recorded on the M4 and the M3 Max (`k.two_ways_disjoint`, `k.two_ways_gap`) and
/// every other byte the buffer's; and a pass stores a texture over the buffer while its draw
/// writes, through a second texture over the same buffer 16 bytes on, the texels between the
/// first's rows, leaving each texture's texels as its writer left them, and the same pass split by
/// a memory barrier before a draw that reads the second texture's texels into the first sees them
/// (the merge between the instances). Where a texture is shadowed (the device announces
/// `shadowed-texture`), the runtime merges its writes into the buffer and announces that once
/// ([`MERGES`]); where none is, nothing is merged.
#[test]
fn a_command_writing_a_buffers_bytes_two_ways_keeps_both_writes() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        // What the device announced as it was made.
        messages.take_notices();
        each_kind(&device, two_ways);
        let counted = counts(&messages.take_notices());
        let shadowed = counted.contains_key(&Degradation::ShadowedTexture);
        assert_eq!(
            counted.get(&Degradation::ShadowMerge).copied(),
            shadowed.then_some(1),
            "{} merges where a texture is shadowed, once: {counted:?}",
            device.name()
        );
    }
    layer_reported_nothing_at_all(&messages);
}

fn two_ways(lane: &Lane, kind: Kind) {
    let device = &lane.device;
    let rgba = PixelFormat::RGBA8Unorm;
    let function = lane.library.new_function_with_name("tob_two_ways").unwrap();
    let kernel = device
        .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
        .unwrap();
    // The kernel writes a texel (and a buffer's bytes).
    assert!(kernel.writes_textures());
    for (case, off, rows) in [
        ("k.two_ways_disjoint", OFF + 4, PROBE_TWO_WAYS_DISJOINT),
        ("k.two_ways_gap", OFF + 16, PROBE_TWO_WAYS_GAP),
    ] {
        let what = format!("{case} over a {kind:?} buffer");
        let mut b = Held::new(device, kind).unwrap();
        b.fill(OFF, BPR, ALL, 0);
        let t = b.texture_of(rgba, RT, (OFF, BPR), ALL);
        let at = buffer_with(device, &(off as u32).to_le_bytes()).unwrap();
        lane.commit(|command_buffer, _| {
            let mut e = command_buffer.compute_command_encoder().unwrap();
            e.set_compute_pipeline_state(&kernel).unwrap();
            e.set_texture(Some(&t), 0).unwrap();
            e.set_buffer(Some(&b.buffer), 0, 0).unwrap();
            e.set_buffer(Some(&at), 0, 1).unwrap();
            e.dispatch_threads(Size::new(1, 1, 1), Size::new(1, 1, 1))
                .unwrap();
            e.end_encoding().unwrap();
        });
        let bytes = b.bytes();
        assert_eq!(
            hex_rows(&bytes, OFF, BPR, 4),
            rows,
            "{what}: the rows Metal's probe recorded"
        );
        let mut want = Expect::of(&b.model);
        want.set(
            (OFF, BPR),
            ([0, 0], [1, 1]),
            |_, _| [0xff, 0, 0, 0xff],
            false,
        );
        let off = off as usize;
        want.bytes[off..off + 4].copy_from_slice(&[0x5a, 0x5b, 0x5c, 0x5d]);
        want.check(&bytes, &what);
    }
    let b = Held::new(device, kind).unwrap();
    let (stored, between) = (
        b.texture_of(rgba, RT, (OFF, BPR), ALL),
        b.texture_of(rgba, RT, (OFF + 16, BPR), ALL),
    );
    let write = lane.pipeline(("tob_rect", "tob_write"), rgba, 1);
    lane.run(
        &pass(&stored, LoadAction::Clear, StoreAction::Store, CLEAR),
        |l, e, h| {
            e.set_fragment_texture(Some(&between), 0).unwrap();
            l.rect(e, h, &write, FULL, None);
        },
    )
    .unwrap();
    // The draw's color, opaque black, in the stored texture; its writes in the other.
    let mut want = Expect::of(&b.model);
    want.set((OFF, BPR), ([0, 0], ALL), |_, _| [0, 0, 0, 0xff], false);
    want.set((OFF + 16, BPR), ([0, 0], ALL), written, true);
    want.check(
        &b.bytes(),
        &format!("a pass storing a texture over a {kind:?} buffer and writing another"),
    );
    // The same, then a memory barrier, which splits the pass, and a draw that reads the second
    // texture's texels into the first: what the first instance wrote both ways, merged between
    // the instances.
    let b = Held::new(device, kind).unwrap();
    let (stored, between) = (
        b.texture_of(rgba, RT, (OFF, BPR), ALL),
        b.texture_of(rgba, RT, (OFF + 16, BPR), ALL),
    );
    let read = lane.pipeline(("tob_rect", "tob_read"), rgba, 1);
    lane.run(
        &pass(&stored, LoadAction::Clear, StoreAction::Store, CLEAR),
        |l, e, h| {
            e.set_fragment_texture(Some(&between), 0).unwrap();
            l.rect(e, h, &write, FULL, None);
            e.memory_barrier_with_scope(
                BarrierScope::TEXTURES,
                RenderStages::FRAGMENT,
                RenderStages::FRAGMENT,
            );
            l.rect(e, h, &read, FULL, None);
        },
    )
    .unwrap();
    let mut want = Expect::of(&b.model);
    want.set((OFF, BPR), ([0, 0], ALL), written, true);
    want.set((OFF + 16, BPR), ([0, 0], ALL), written, true);
    want.check(
        &b.bytes(),
        &format!("a pass split after writing a {kind:?} buffer's bytes two ways"),
    );
}

/// A render pass that executes render commands copies a shadowed texture between its instances,
/// as one that only draws does: over each kind of buffer, a draw reads a texture over it, an
/// execution's command writes the first row's bytes through the buffer (the owned
/// `icb_render_replay_template`, its vertex `i` writing `params[0] * (i + 1) + params[1]`), a
/// memory barrier from the vertex stage to the fragment stage splits the pass, and a draw reads the
/// texture again: it reads the command's words, which Metal's barrier makes visible, through a copy
/// into the shadow between the instances where the texture is shadowed, and the texels the command
/// left. Before RUNTIME 148 such a pass was refused `NotImplemented(IndirectRenderCommands)`. The
/// first draw's colors are drawn over, so its read of the row the command writes, which Metal's
/// pass races, is not checked.
#[test]
fn a_pass_executing_render_commands_copies_shadowed_textures_between_its_instances() {
    on_device(|device| {
        each_kind(device, |lane, kind| {
            executed(lane, kind, Before::Nothing, 16)
        })
    });
}

/// [`a_pass_executing_render_commands_copies_shadowed_textures_between_its_instances`] over a
/// texture of rows 32 bytes apart (16 bytes of texels: shadowed, as the shadow copies it counts
/// show), where a compute encoder before the pass made the indirect command buffer resident for
/// writing, so the GPU may write its records and a host step records the pass from them, over the
/// command buffer's resident resources: the buffer's records, which no shadowed texture's bytes
/// share, change no copy, and the pass draws as before. Where a compute encoder after the pass
/// made the shadowed texture resident, past what the call gathered (an earlier encoder's shadowed
/// textures join the call's own), the pass is refused (`NotImplemented(IndirectRenderCommands)`):
/// its copies follow what the call gathered.
#[test]
fn a_shadowed_pass_executes_commands_the_gpu_may_write() {
    on_device(|device| {
        each_kind(device, |lane, kind| {
            let copies = lane.device.work_counts().shadow_copies;
            executed(lane, kind, Before::Writer, BPR);
            let copied = lane.device.work_counts().shadow_copies - copies;
            assert!(copied > 0, "{kind:?}: the texture is shadowed");
            executed(lane, kind, Before::WriterAndTexture, BPR);
        })
    });
}

/// What the command buffer of [`executed`] encodes before its pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Before {
    Nothing,
    /// A compute encoder that makes the indirect command buffer resident for writing, after one
    /// that sets a state that may encode (`indirect_host_steps::encoding_state_set`).
    Writer,
    /// That, and a compute encoder after the pass that makes the shadowed texture resident.
    WriterAndTexture,
}

/// The test over texture rows `bpr` bytes apart.
fn executed(lane: &Lane, kind: Kind, before: Before, bpr: u64) {
    let device = &lane.device;
    let rgba = PixelFormat::RGBA8Unorm;
    let read = lane.pipeline(("tob_rect", "tob_read"), rgba, 1);
    let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/public/engine/icb_render_replay_template.ll");
    let library = device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap();
    let function = library
        .new_function_with_name("icb_render_replay_template")
        .unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.rasterization_enabled = false;
    descriptor.support_indirect_command_buffers = true;
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(rgba));
    let writer = device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap();
    let mut b = Held::new(device, kind).unwrap();
    let t = b.texture_of(rgba, RT, (OFF, bpr), ALL);
    let params: Vec<u8> = [0x0101_0101u32, 0x4030_2010]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    let params = buffer_with(device, &params).unwrap();
    let mut commands = IndirectCommandBufferDescriptor::new();
    commands.command_types = IndirectCommandType::DRAW;
    commands.max_vertex_buffer_bind_count = 2;
    let icb = device
        .new_indirect_command_buffer(&commands, 1, ResourceOptions::shared())
        .unwrap();
    let command = icb.indirect_render_command_at(0).unwrap();
    command.set_render_pipeline_state(&writer).unwrap();
    command.set_vertex_buffer(Some(&params), 0, 0).unwrap();
    command.set_vertex_buffer(Some(&b.buffer), OFF, 1).unwrap();
    command
        .draw_primitives(PrimitiveType::Point, 0, 4, 1, 0)
        .unwrap();
    let o = lane.ordinary(rgba, ALL, None);
    let command_buffer = lane.queue.command_buffer().unwrap();
    if before != Before::Nothing {
        encoding_state_set(&command_buffer);
        let mut writer = command_buffer.compute_command_encoder().unwrap();
        writer
            .use_resource(icb.as_resource(), ResourceUsage::WRITE)
            .unwrap();
        writer.end_encoding().unwrap();
    }
    let descriptor = pass(&o, LoadAction::Clear, StoreAction::Store, [0.0; 4]);
    let mut e = command_buffer.render_command_encoder(&descriptor).unwrap();
    let h = &mut Vec::new();
    e.use_resource(
        params.as_resource(),
        ResourceUsage::READ,
        RenderStages::VERTEX,
    )
    .unwrap();
    e.use_resource(
        b.buffer.as_resource(),
        ResourceUsage::WRITE,
        RenderStages::VERTEX,
    )
    .unwrap();
    e.set_fragment_texture(Some(&t), 0).unwrap();
    lane.rect(&mut e, h, &read, LEFT, None);
    e.execute_commands_in_buffer(&icb, 0..1).unwrap();
    e.memory_barrier_with_scope(
        BarrierScope::BUFFERS | BarrierScope::TEXTURES,
        RenderStages::VERTEX,
        RenderStages::FRAGMENT,
    );
    lane.rect(&mut e, h, &read, FULL, None);
    e.end_encoding().unwrap();
    if before == Before::WriterAndTexture {
        let mut after = command_buffer.compute_command_encoder().unwrap();
        after
            .use_resource(t.as_resource(), ResourceUsage::READ)
            .unwrap();
        after.end_encoding().unwrap();
    }
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(icb.gpu_written(), before != Before::Nothing, "{before:?}");
    if before == Before::WriterAndTexture {
        assert_eq!(command_buffer.status(), CommandBufferStatus::Error);
        assert!(
            matches!(
                command_buffer.refusal(),
                Some(Error::NotImplemented(RuntimeGap::IndirectRenderCommands))
            ),
            "{kind:?}: {:?}",
            command_buffer.refusal()
        );
        return;
    }
    assert_eq!(
        command_buffer.status(),
        CommandBufferStatus::Completed,
        "{kind:?} {before:?}: {:?}",
        command_buffer.refusal()
    );
    let words: Vec<u8> = (1..=4u32)
        .flat_map(|i| (0x0101_0101 * i + 0x4030_2010).to_le_bytes())
        .collect();
    b.model[OFF as usize..OFF as usize + 16].copy_from_slice(&words);
    let what = format!(
        "a read, an execution's write, a barrier and a read over a {kind:?} buffer, {before:?}"
    );
    Expect::of(&b.model).check(&b.bytes(), &what);
    let texels: Vec<u8> = (0..4)
        .flat_map(|row| {
            let at = (OFF + row * bpr) as usize;
            b.model[at..at + 16].to_vec()
        })
        .collect();
    Expect::of(&texels).check(&texture_bytes(&o, ALL), &what);
}
