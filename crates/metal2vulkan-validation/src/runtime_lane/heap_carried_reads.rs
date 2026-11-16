//! A heap's carried textures read and sample as the same texture outside a heap, on the device
//! the cfg selects, under the validation layer. Metal's PVRTC, ETC2 and 4:2:2 textures in a heap
//! sample and read as they do outside one; on a device without the format (a 4:2:2 one's on every
//! device) the runtime holds their blocks in a carrier whose sampled views read a decoded twin
//! (`runtime/src/texture/decoded_twin.rs`, `chroma_pairs.rs`), in a heap as outside one: bound in
//! the heap's memory, or, where the carrier asks more of the heap than the answer it is held to
//! (`HeapAnswerPolicy::Metal` on RADV, whose carrier takes 4096 bytes at 4096 where Metal answers
//! 128 at 128; a 4:2:2 texture under both policies, whose answer is Metal's rows), a shadow of the
//! heap's bytes whose whole twin is decoded after each copy in that moved texels
//! (`runtime/src/texture/heap.rs`, `shadow.rs`).
//!
//! Under both heap answer policies, for a `PVRTC_RGBA_4BPP`, an `ETC2_RGB8`, a `GBGR422` and a
//! `BGRG422` texture, 16 and 64 texels square, in a `Private` and a `Shared` automatic heap, beside
//! a texture of the same descriptor outside the heap, each step done to both:
//!
//! - **Filled** by a blit of the same seeded blocks; a kernel's reads (twice, in two encoders of
//!   one command buffer) and a sampler's samples of the first 16 x 16 texels equal the other's.
//! - **Written** by a blit of other blocks; the samples equal the other's and differ from before.
//! - **Written by the CPU** (a `Shared` heap's): `replaceRegion` of a third set of blocks; the
//!   samples after the next copy in equal the other's and differ from before.
//! - **Read back**: a blit of its blocks into a buffer, and a `Shared` one's `getBytes`, equal the
//!   other's.
//!
//! Each case prints a `HEAPDECODE` line: the heap's answer, whether its image shadows the heap's
//! bytes (`shadowed-texture` at its making, each kind announced at each texture,
//! `DeviceOptions::announce_each`), whether it took a twin (`carried-decoded-twin` at its making),
//! and for each use the shadow's copies its encoders planned
//! (`WorkCounts::shadow_copies`) and the decodes recorded after copies in
//! (`WorkCounts::shadow_twin_decodes`), and a 4:2:2 texture's twin's route: the device's own
//! conversion of the pairs where it converts their format (`WorkCounts::chroma_device_twins`), else
//! Metal's decode. The two reads in one command buffer take one copy in, the second encoder
//! carrying the image the first left: a shadowed texture with a twin decodes it once, one whose
//! format the device samples itself (ANV's ETC2: a shadow with no twin) never, and a texture bound
//! in the heap's memory never. A `Shared` shadow's samples after `replaceRegion` take a copy in,
//! decoded where it has a twin.

use super::astc_textures::{reader, texels_of};
use super::degrade_notices::on_device_with;
use super::{at_plan, buffer_with, read_buffer, Messages, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::device::HeapAnswerPolicy;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    BlitCommandEncoder, BlitOption, Buffer, CommandBufferStatus, ComputePipelineState, Degradation,
    Device, DeviceOptions, Error, HeapDescriptor, HeapType, Origin, PipelineOptions, Region,
    ResourceOptions, SamplerState, Size, StorageMode, Texture, TextureDescriptor, WorkCounts,
};

/// The carried formats a heap's texture holds in a carrier with a decoded twin where the device
/// lacks them (every device but PowerVR's lacks PVRTC; RADV and lavapipe lack ETC2, which ANV
/// samples itself: an Iris Xe's heap `ETC2_RGB8` texture takes no twin), and the 4:2:2 formats,
/// held so on every device.
const FORMATS: [PixelFormat; 4] = [
    PixelFormat::PVRTCRGBA4BPP,
    PixelFormat::ETC2RGB8,
    PixelFormat::GBGR422,
    PixelFormat::BGRG422,
];

/// The textures' sides: the readers' own, and the side the first-use lane's heap makings take.
const SIDES: [u64; 2] = [16, 64];

/// The side of the region the readers read (`format_scout.ll`'s kernels, rows of 16 texels).
const READ: u64 = 16;

/// The route kinds a heap texture's making announces, each at each texture.
const ROUTES: [Degradation; 5] = [
    Degradation::ShadowedTexture,
    Degradation::HeapTextureHostTransfer,
    Degradation::HeapTextureLinear,
    Degradation::HeapTextureLargerThanMetal,
    Degradation::HeapTextureUntiledLayout,
];

/// The kind a carried texture's twin announces at its making, each at each texture: whether the
/// heap's texture took one.
const TWIN: Degradation = Degradation::CarriedDecodedTwin;

/// `n` bytes of a seeded xorshift: random blocks of every mode.
fn seeded(seed: u64, n: usize) -> Vec<u8> {
    let mut x = seed | 1;
    let mut out = Vec::with_capacity(n + 8);
    while out.len() < n {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        out.extend_from_slice(&x.wrapping_mul(0x2545_f491_4f6c_dd1d).to_le_bytes());
    }
    out.truncate(n);
    out
}

/// A square 2D texture of `pf`, `side` texels, read by shaders, of `options`.
fn descriptor(pf: PixelFormat, side: u64, options: ResourceOptions) -> TextureDescriptor {
    let mut d = TextureDescriptor::texture_2d(pf, side, side, false);
    d.usage = TextureUsage::SHADER_READ;
    d.resource_options = options;
    d
}

/// Whether `pf` is a 4:2:2 format, whose texels are 2 bytes each.
fn four_two_two(pf: PixelFormat) -> bool {
    matches!(pf, PixelFormat::GBGR422 | PixelFormat::BGRG422)
}

/// The bytes of a row of `pf`'s blocks (a 4:2:2 format's texels) across a `side`-texel level, and
/// of the whole level.
fn level_bytes(pf: PixelFormat, side: u64) -> (u64, u64) {
    match four_two_two(pf) {
        true => (side * 2, side * 2 * side),
        false => (side / 4 * 8, side / 4 * 8 * (side / 4)),
    }
}

/// The option a blit of `pf`'s blocks row by row takes: PVRTC's row-linear option.
fn rows(pf: PixelFormat) -> BlitOption {
    match pf {
        PixelFormat::PVRTCRGBA4BPP => BlitOption::RowLinearPVRTC,
        _ => BlitOption::None,
    }
}

/// The format scout's `scout_sample_f` on `device`, which samples each texel of a 16 x 16 region
/// a quarter and three quarters of the way into it.
fn sampling(device: &Device) -> ComputePipelineState {
    let bitcode = llvm_assemble(&engine_dir().join("format_scout.ll")).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library.new_function_with_name("scout_sample_f").unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap()
}

/// `error` as a line.
fn said(error: Error) -> String {
    format!("{error:?}")
}

/// The device's two readers and a nearest sampler.
struct Readers {
    read: ComputePipelineState,
    sample: ComputePipelineState,
    sampler: SamplerState,
}

/// One use's counts: the shadow copies its encoders planned and the decodes after copies in.
#[derive(Clone, Copy, Debug)]
struct Counted {
    copies: u64,
    decodes: u64,
}

impl Counted {
    /// What happened between `before` and `after`.
    fn between(before: &WorkCounts, after: &WorkCounts) -> Counted {
        Counted {
            copies: after.shadow_copies - before.shadow_copies,
            decodes: after.shadow_twin_decodes - before.shadow_twin_decodes,
        }
    }
}

/// One command buffer of the blit `encode` records, committed and waited for: its error, or the
/// command buffer's where it did not complete.
fn blit(
    device: &Device,
    encode: impl FnOnce(&mut BlitCommandEncoder) -> Result<(), Error>,
) -> Result<(), String> {
    let queue = device.new_command_queue().map_err(said)?;
    let command_buffer = queue.command_buffer().map_err(said)?;
    let mut encoder = command_buffer.blit_command_encoder().map_err(said)?;
    let recorded = encode(&mut encoder);
    encoder.end_encoding().map_err(said)?;
    command_buffer.commit().map_err(said)?;
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    recorded.map_err(said)?;
    match command_buffer.status() {
        CommandBufferStatus::Completed => Ok(()),
        status => Err(format!("{status:?}: {:?}", command_buffer.error())),
    }
}

/// Copy `blocks` (`row` bytes a row of blocks) into the whole `side`-texel level of `texture`.
fn put(
    device: &Device,
    texture: &Texture,
    blocks: &[u8],
    side: u64,
    row: u64,
) -> Result<(), String> {
    let pf = texture.pixel_format();
    let source = buffer_with(device, blocks).unwrap();
    let size = Size::new(side, side, 1);
    let image = blocks.len() as u64;
    blit(device, |encoder| {
        let at = Origin::new(0, 0, 0);
        encoder.copy_from_buffer_to_texture(
            &source,
            0,
            row,
            image,
            size,
            texture,
            0,
            0,
            at,
            rows(pf),
        )
    })
}

/// The blocks of the whole `side`-texel level of `texture`, blitted into a buffer `row` bytes a
/// row of blocks: `length` bytes.
fn get(
    device: &Device,
    texture: &Texture,
    side: u64,
    row: u64,
    length: u64,
) -> Result<Vec<u8>, String> {
    let pf = texture.pixel_format();
    let out: Buffer = buffer_with(device, &vec![0; length as usize]).unwrap();
    let size = Size::new(side, side, 1);
    blit(device, |encoder| {
        let at = Origin::new(0, 0, 0);
        encoder.copy_from_texture_to_buffer(texture, 0, 0, at, size, &out, 0, row, length, rows(pf))
    })?;
    Ok(read_buffer(&out, 0, length).unwrap())
}

/// The `float4` texels of the first [`READ`] x [`READ`] texels of `texture` as `state` reads them
/// (through `sampler` where given), each of `encoders` encoders of one command buffer reading
/// them once into its own buffer: its error where a call refused the read.
fn texels(
    device: &Device,
    state: &ComputePipelineState,
    sampler: Option<&SamplerState>,
    texture: &Texture,
    encoders: usize,
) -> Result<Vec<Vec<[f32; 4]>>, String> {
    let length = 16 * READ * READ;
    let outs: Vec<Buffer> = (0..encoders)
        .map(|_| buffer_with(device, &vec![0; length as usize]).unwrap())
        .collect();
    let queue = device.new_command_queue().map_err(said)?;
    let command_buffer = queue.command_buffer().map_err(said)?;
    let mut recorded = Ok(());
    for out in &outs {
        let mut encoder = command_buffer.compute_command_encoder().map_err(said)?;
        let read = (|| {
            encoder.set_compute_pipeline_state(state)?;
            encoder.set_texture(Some(texture), 0)?;
            if let Some(sampler) = sampler {
                encoder.set_sampler_state(Some(sampler), 0)?;
            }
            encoder.set_buffer(Some(out), 0, 0)?;
            encoder.dispatch_threads(Size::new(READ, READ, 1), Size::new(READ, READ, 1))
        })();
        encoder.end_encoding().map_err(said)?;
        if recorded.is_ok() {
            recorded = read;
        }
    }
    command_buffer.commit().map_err(said)?;
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    recorded.map_err(|error| format!("refused: {error:?}"))?;
    match command_buffer.status() {
        CommandBufferStatus::Completed => {}
        status => return Err(format!("{status:?}: {:?}", command_buffer.error())),
    }
    Ok((outs.iter())
        .map(|out| texels_of(&read_buffer(out, 0, length).unwrap()))
        .collect())
}

/// What a case found: a line of evidence, and its failures.
struct Case {
    line: String,
    wrong: Vec<String>,
}

/// A case's failures so far, and the per-use counts for its line.
struct Checks<'a> {
    what: &'a str,
    device: &'a Device,
    wrong: Vec<String>,
    uses: Vec<String>,
}

impl Checks<'_> {
    /// Run `step` (a use of the heap's texture), counting what its shadow did, and answer what it
    /// answered with the counts; `name` labels them on the line.
    fn counted<T>(&mut self, name: &str, step: impl FnOnce() -> T) -> (T, Counted) {
        let before = self.device.work_counts();
        let answer = step();
        let counted = Counted::between(&before, &self.device.work_counts());
        self.uses.push(format!(
            "{name} copies {} decodes {}",
            counted.copies, counted.decodes
        ));
        (answer, counted)
    }

    /// The heap texture's texels `got` (each encoder's) against the other texture's `want`, both
    /// read as `how` says: the other's texels, where both reads ran.
    fn alike(
        &mut self,
        how: &str,
        got: Result<Vec<Vec<[f32; 4]>>, String>,
        want: Result<Vec<Vec<[f32; 4]>>, String>,
    ) -> Option<Vec<[f32; 4]>> {
        let what = self.what;
        let (got, want) = match (got, want) {
            (Ok(got), Ok(want)) => (got, want),
            (got, want) => {
                if let Err(error) = got {
                    self.wrong.push(format!("{what}: {how}: {error}"));
                }
                if let Err(error) = want {
                    self.wrong
                        .push(format!("{what}: {how} outside a heap: {error}"));
                }
                return None;
            }
        };
        let want = want.into_iter().next()?;
        for (encoder, got) in got.iter().enumerate() {
            let differ = (got.iter().zip(&want)).filter(|(a, b)| a != b).count();
            if differ > 0 {
                self.wrong.push(format!(
                    "{what}: {how} (encoder {encoder}): {differ} of {} texels otherwise than \
                     outside a heap",
                    READ * READ
                ));
            }
        }
        Some(want)
    }

    /// Texels `now` differ from `before` somewhere: a write's new texels show.
    fn changed(&mut self, how: &str, before: Option<&[[f32; 4]]>, now: Option<&[[f32; 4]]>) {
        if let (Some(before), Some(now)) = (before, now) {
            if before == now {
                let what = self.what;
                self.wrong
                    .push(format!("{what}: {how}: the samples did not change"));
            }
        }
    }
}

/// One case on `device`: a texture of `pf`, `side` texels, in an automatic heap of `options`,
/// against the same texture outside a heap (the module docs), its blocks seeded by `seed`.
fn case(
    device: &Device,
    messages: &Messages,
    readers: &Readers,
    (pf, side, options): (PixelFormat, u64, ResourceOptions),
    seed: u64,
) -> Case {
    let name = device.name();
    let storage = options.storage_mode().unwrap();
    let what = format!("{pf:?} {side}x{side} {storage:?}");
    let d = descriptor(pf, side, options);
    let answer = device.heap_texture_size_and_align(&d);
    let heap = device
        .new_heap(&HeapDescriptor::new(HeapType::Automatic, 1 << 20, options))
        .unwrap();
    messages.take_notices();
    let twins = device.work_counts().chroma_device_twins;
    let made = heap.new_texture(&d);
    let kinds: Vec<Degradation> = (messages.take_notices().iter())
        .filter_map(|m| m.degradation.as_ref().map(|n| n.degradation))
        .collect();
    let routes: Vec<Degradation> = (kinds.iter().copied())
        .filter(|kind| ROUTES.contains(kind))
        .collect();
    let pairs = device.work_counts().chroma_device_twins - twins;
    let twinned = kinds.contains(&TWIN);
    let twin = match (twinned, four_two_two(pf), pairs) {
        (false, ..) => "none",
        (true, false, _) => "the blocks' decode",
        (true, true, 0) => "Metal's decode of the pairs",
        (true, true, _) => "the device's conversion of the pairs",
    };
    let shadowed = routes.contains(&Degradation::ShadowedTexture);
    let in_heap = match made {
        Ok(texture) => texture,
        Err(error) => {
            let line = format!("HEAPDECODE {name} {what}: answer {answer:?}; making refused");
            let wrong = vec![format!("{what}: refused its making in the heap: {error:?}")];
            return Case { line, wrong };
        }
    };
    let own = device.new_texture(&d).unwrap();
    let mut checks = Checks {
        what: &what,
        device,
        wrong: Vec::new(),
        uses: Vec::new(),
    };
    let (row, image) = level_bytes(pf, side);
    let both = [&in_heap, &own];
    let (read, sample, sampler) = (&readers.read, &readers.sample, Some(&readers.sampler));

    // Filled by a blit; read twice in one command buffer, then sampled.
    let filled = seeded(seed, image as usize);
    for texture in both {
        if let Err(error) = put(device, texture, &filled, side, row) {
            checks
                .wrong
                .push(format!("{what}: a blit into it: {error}"));
        }
    }
    let (got, reads) = checks.counted("read x2", || texels(device, read, None, &in_heap, 2));
    let want = texels(device, read, None, &own, 1);
    checks.alike("its kernel's reads", got, want);
    // A shadow's one copy in for both encoders (the reads write nothing, so no copy back),
    // decoded once where it has a twin, and not where the device samples its format itself (ANV's
    // ETC2); no decode for an image in the heap's memory.
    let carried = match (shadowed, twinned) {
        (true, true) => reads.copies == 1 && reads.decodes == 1,
        (true, false) => reads.copies == 1 && reads.decodes == 0,
        (false, _) => reads.decodes == 0,
    };
    if !carried {
        checks.wrong.push(format!(
            "{what}: two reads in one command buffer, shadowed {shadowed}, twinned {twinned}: \
             {reads:?}"
        ));
    }
    let (got, _) = checks.counted("sample", || texels(device, sample, sampler, &in_heap, 1));
    let want = texels(device, sample, sampler, &own, 1);
    let first = checks.alike("its samples", got, want);

    // Written by a blit of other blocks: the samples show them.
    let blitted = seeded(seed ^ 0xb11e, image as usize);
    for texture in both {
        if let Err(error) = put(device, texture, &blitted, side, row) {
            checks
                .wrong
                .push(format!("{what}: a second blit into it: {error}"));
        }
    }
    let (got, _) = checks.counted("blit+sample", || {
        texels(device, sample, sampler, &in_heap, 1)
    });
    let want = texels(device, sample, sampler, &own, 1);
    let second = checks.alike("its samples after a blit", got, want);
    checks.changed("after a blit", first.as_deref(), second.as_deref());

    // A `Shared` one written by the CPU (PVRTC's whole level in Metal's CPU order, at a pitch of
    // 0): the samples after the next copy in show it.
    let pitch = match pf {
        PixelFormat::PVRTCRGBA4BPP => 0,
        _ => row,
    };
    let region = Region::new_2d(0, 0, side, side);
    if storage == StorageMode::Shared {
        let written = seeded(seed ^ 0xc0de, image as usize);
        for texture in both {
            if let Err(error) = texture.replace_region(region, 0, 0, &written, pitch, 0) {
                checks
                    .wrong
                    .push(format!("{what}: replaceRegion: {error:?}"));
            }
        }
        let (got, cpu) = checks.counted("replace+sample", || {
            texels(device, sample, sampler, &in_heap, 1)
        });
        let want = texels(device, sample, sampler, &own, 1);
        let third = checks.alike("its samples after replaceRegion", got, want);
        checks.changed("after replaceRegion", second.as_deref(), third.as_deref());
        // The CPU's blocks reach a shadow by a copy in, which a twin decodes.
        let reached = match twinned {
            true => cpu.decodes > 0,
            false => cpu.copies > 0,
        };
        if shadowed && !reached {
            checks.wrong.push(format!(
                "{what}: a sample after replaceRegion copied or decoded nothing (twinned \
                 {twinned}): {cpu:?}"
            ));
        }
    }

    // Read back: a blit of its blocks, and a `Shared` one's CPU copy.
    let blocks = both.map(|texture| get(device, texture, side, row, image));
    match &blocks {
        [Ok(a), Ok(b)] if a == b => {}
        [Ok(_), Ok(_)] => (checks.wrong).push(format!(
            "{what}: a blit out reads other blocks than outside a heap"
        )),
        [a, b] => (checks.wrong).push(format!(
            "{what}: a blit out: {:?} / {:?}",
            a.as_ref().err(),
            b.as_ref().err()
        )),
    }
    if storage == StorageMode::Shared {
        let mut got = [vec![0xee; image as usize], vec![0xdd; image as usize]];
        let results: Vec<Result<(), Error>> = (both.iter().zip(got.iter_mut()))
            .map(|(texture, bytes)| texture.get_bytes(bytes, pitch, 0, region, 0, 0))
            .collect();
        let alike = got[0] == got[1];
        if !(alike && results.iter().all(Result::is_ok)) {
            (checks.wrong).push(format!("{what}: getBytes {results:?}, alike {alike}"));
        }
    }
    let line = format!(
        "HEAPDECODE {name} {what}: answer {answer:?}; shadowed {shadowed}; route notices \
         {routes:?}; twin {twin}; per use {}",
        checks.uses.join(", ")
    );
    let wrong = checks.wrong;
    drop((in_heap, own, heap));
    Case { line, wrong }
}

/// A heap's carried textures read and sample as the same texture outside a heap, under both heap
/// answer policies (the module docs).
#[test]
fn a_heap_s_carried_textures_read_as_outside_a_heap() {
    let mut wrong = Vec::new();
    for answers in [HeapAnswerPolicy::Metal, HeapAnswerPolicy::Device] {
        let each = |options: &mut DeviceOptions| {
            at_plan(options);
            options.policy.heap_answers = answers;
            options.announce_each = ROUTES.into_iter().chain([TWIN]).collect();
        };
        on_device_with(each, |device, messages| {
            let readers = Readers {
                read: reader(device),
                sample: sampling(device),
                sampler: device.new_sampler_state(&SamplerDescriptor::new()).unwrap(),
            };
            let mut seed = 0x5eed_0000;
            for pf in FORMATS {
                for side in SIDES {
                    for options in [ResourceOptions::private(), ResourceOptions::shared()] {
                        seed += 1;
                        let found = case(device, messages, &readers, (pf, side, options), seed);
                        println!("{answers:?} {}", found.line);
                        let name = device.name();
                        (wrong).extend(
                            found
                                .wrong
                                .into_iter()
                                .map(|w| format!("{answers:?} {name}: {w}")),
                        );
                    }
                }
            }
        });
    }
    assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
}
