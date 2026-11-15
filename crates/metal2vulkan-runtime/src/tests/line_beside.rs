//! Two commands of one command buffer with no dependency between them, one writing the texels of
//! a plane in place over a buffer of the runtime's own `Private` memory, the other writing bytes
//! of that buffer beside the plane, each keep the other's bytes, as on a Mac: Metal orders
//! neither (an untracked buffer's commands are the program's to order, and the runtime's tracker
//! orders a tracked buffer's uses by byte range, `tracker.rs`, so the plane's texels and bytes
//! outside them take no barrier), and a command writes the bytes it names alone. On the Iris Xe
//! (ANV), whose GPU writes into host memory write back the whole 64-byte line they end in while
//! the CPU writes beside them (`docs/FAST-PATHS.md`, **Known gaps**), bytes beside a plane may
//! share such a line with its texels:
//!
//! - **The planes.** An `R8Unorm` IOSurface plane at byte `0x20`, of 1920x64 texels at rows 1920
//!   bytes apart (tight) and of 1000x36 at the pitch the device's own linear image of it takes
//!   (padded: 1024 on ANV), over a `Private` buffer that lays byte `0x20` on a multiple of 256
//!   (`BufferDescriptor::aligned_byte`: the plane's first texel starts a line) and over one made
//!   without the option (its first byte on a multiple of 256, so the plane's first texels share a
//!   line with bytes `0x0..0x20`), each tracked and untracked.
//! - **The writes.** In one command buffer, a render pass that clears the plane at its load and
//!   stores it (the plane made with `RenderTarget` usage), or a kernel (`invert_texels`) that
//!   writes each of its texels (`ShaderWrite`), then a blit's copies or a kernel's stores
//!   (`scale_and_offset`) into bytes `0x0..0x20` and into the bytes after the plane's last texel
//!   to the end of their line of memory (`Device::in_place_check_line`; the next line where the
//!   plane ends on one), three rounds each, the buffer refilled with a pattern of the round's own
//!   first. Every texel then reads the plane's write, every byte the second command wrote reads
//!   its bytes, and every other byte the pattern. Each case's route is printed (in place, or a
//!   shadow whose copies back write the same lines).
//! - **A lost write shows.** A round that leaves the second command out reads the pattern at
//!   every byte that command would have written, which the comparison counts as lost, and
//!   nothing else.
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects. Printed (`LINE-BESIDE`): each case's plane, route, regions and what
//! its rounds read.

use super::blit_holds::{device, Scratch, WAIT};
use super::dispatch_pending::state;
use super::encoder_holds::library;
use crate::{
    BlitCommandEncoder, Buffer, BufferContents, BufferDescriptor, ClearColor, CommandQueue,
    ComputePipelineState, CpuCacheMode, Device, HazardTrackingMode, IoSurfacePlane, LoadAction,
    PipelineOptions, Region, RenderPassDescriptor, ResourceOptions, Size, StorageMode, StoreAction,
    Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::ops::Range;
use std::time::Duration;

/// The byte the planes lie at: CoreVideo's first plane's.
const K: u64 = 0x20;

/// The rounds of each case.
const ROUNDS: u64 = 3;

/// The longest the device's in-place checks may take to settle.
const CHECK_WAIT: Duration = Duration::from_secs(120);

/// A plane's shape: its texels a row and its rows, and whether its rows lie at the device's own
/// linear pitch past their texels (padded) or one after another (tight).
#[derive(Clone, Copy, Debug)]
enum Shape {
    Tight,
    Padded,
}

impl Shape {
    /// Its texels a row and its rows.
    fn size(self) -> (u64, u64) {
        match self {
            Shape::Tight => (1920, 64),
            Shape::Padded => (1000, 36),
        }
    }
}

/// The command that writes the plane's texels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Plane {
    /// A render pass that clears it at its load (to zero) and stores it.
    Pass,
    /// A kernel's write of each texel (`invert_texels` of a zero texture: 1).
    Kernel,
}

impl Plane {
    /// The plane's usage for this write.
    fn usage(self) -> TextureUsage {
        match self {
            Plane::Pass => TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET,
            Plane::Kernel => TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
        }
    }

    /// The byte each texel reads after it.
    fn texel(self) -> u8 {
        match self {
            Plane::Pass => 0x00,
            Plane::Kernel => 0xff,
        }
    }
}

/// The command after it that writes the bytes beside the plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Beside {
    /// A blit's copies of them from a `Shared` buffer.
    Blit,
    /// A kernel's stores of them (`scale_and_offset`, one dispatch each run of them).
    Kernel,
    /// None: the control, whose rounds must read every such byte lost.
    Left,
}

/// Where a case's bytes lie in its buffer.
#[derive(Clone, Debug)]
struct Layout {
    width: u64,
    pitch: u64,
    /// The bytes before the plane: `0..K`.
    front: Range<u64>,
    /// The bytes after its last texel to the end of their line (the next line where it ends on
    /// one).
    back: Range<u64>,
    /// The buffer's bytes: a line past `back`, which no command writes.
    length: u64,
}

impl Layout {
    /// The layout of `width` x `height` texels at `pitch` from byte `K`, in a buffer whose lines
    /// of `line` bytes start at `first` (and every `line` bytes from there).
    fn new((width, height, pitch): (u64, u64, u64), first: u64, line: u64) -> Layout {
        let end = K + (height - 1) * pitch + width;
        let back_end = first + ((end - first) / line + 1) * line;
        Layout {
            width,
            pitch,
            front: 0..K,
            back: end..back_end,
            length: back_end + line,
        }
    }

    /// Whether byte `at` is one of the plane's texels.
    fn texel(&self, at: u64) -> bool {
        at >= K && at < self.back.start && (at - K) % self.pitch < self.width
    }

    /// The region byte `at` lies in: 0 before the plane, 1 a texel, 2 between rows, 3 after the
    /// plane in its line, 4 past that.
    fn region(&self, at: u64) -> usize {
        if self.front.contains(&at) {
            0
        } else if self.texel(at) {
            1
        } else if at < self.back.start {
            2
        } else if self.back.contains(&at) {
            3
        } else {
            4
        }
    }
}

/// The regions' names, by [`Layout::region`].
const REGIONS: [&str; 5] = ["front", "texels", "between rows", "back", "past"];

/// What a case's rounds read: per region, the bytes read as the pattern where a command wrote
/// something else (lost), and those read as neither (wrong), and the first byte of either.
#[derive(Clone, Debug, Default)]
struct Found {
    lost: [u64; 5],
    wrong: [u64; 5],
    first: Option<(u64, u64, u8, u8, u8)>,
}

/// The byte the pattern of `round` holds at `at`: never 0 and never 0xff.
fn pattern_byte(at: u64, round: u64) -> u8 {
    ((at * 13 + at / 251 + round * 31) % 251) as u8 + 1
}

/// The words the beside kernel's input holds in `round`: its output word `i` is word `i` plus
/// `i`, exact in `f32`.
fn kernel_inputs(round: u64, words: u64) -> Vec<f32> {
    (0..words)
        .map(|i| (1u32 << 20) as f32 + (round * 64 + i * 3) as f32)
        .collect()
}

/// The device's state, its queue, and the kernels.
struct Bench {
    device: Device,
    queue: CommandQueue,
    invert: ComputePipelineState,
    scale_and_offset: ComputePipelineState,
    /// `scale_and_offset`'s scale: 1.
    scale: Buffer,
    line: u64,
}

impl Bench {
    /// One command buffer of the blit commands `encode` encodes, completed.
    fn blit(&self, encode: impl FnOnce(&mut BlitCommandEncoder)) {
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        encode(&mut blit);
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
    }

    /// A `Shared` buffer holding `bytes`.
    fn shared(&self, bytes: &[u8]) -> Buffer {
        self.device
            .new_buffer_with_bytes(bytes, ResourceOptions::shared())
            .unwrap()
    }

    /// `buffer`'s first `length` bytes, through a blit into a `Shared` buffer.
    fn read_back(&self, buffer: &Buffer, length: u64) -> Vec<u8> {
        let read = self
            .device
            .new_buffer(length, ResourceOptions::shared())
            .unwrap();
        self.blit(|blit| {
            blit.copy_from_buffer_to_buffer(buffer, 0, &read, 0, length)
                .unwrap()
        });
        let at = read.contents().expect("a Shared buffer's contents");
        // SAFETY: the buffer's mapped bytes, `length` of them, which no command uses now.
        unsafe { std::slice::from_raw_parts(at.as_ptr(), length as usize) }.to_vec()
    }

    /// A zero `R8Unorm` texture of `width` x `height`, which `invert_texels` reads.
    fn zero_texture(&self, width: u64, height: u64) -> Texture {
        let mut descriptor =
            TextureDescriptor::texture_2d(PixelFormat::R8Unorm, width, height, false);
        descriptor.usage = TextureUsage::SHADER_READ;
        descriptor.resource_options = ResourceOptions::shared();
        let texture = self.device.new_texture(&descriptor).unwrap();
        let zeros = vec![0u8; (width * height) as usize];
        let region = Region::new_2d(0, 0, width, height);
        (texture.replace_region(region, 0, 0, &zeros, width, 0)).unwrap();
        texture
    }

    /// One case: `rounds` rounds of `plane`'s write then `beside`'s over a plane of `shape` at
    /// byte `K` of a `Private` buffer, laid (`aligned_byte`) where `laid`, of `tracking`: its
    /// layout and what its rounds read, printed.
    fn case(
        &self,
        (shape, laid, tracking): (Shape, bool, HazardTrackingMode),
        (plane, beside): (Plane, Beside),
        rounds: u64,
    ) -> (Layout, Found) {
        let device = &self.device;
        let (width, height) = shape.size();
        let mut descriptor =
            TextureDescriptor::texture_2d(PixelFormat::R8Unorm, width, height, false);
        descriptor.usage = plane.usage();
        descriptor.resource_options =
            ResourceOptions::new(StorageMode::Private, CpuCacheMode::DefaultCache, tracking);
        let pitch = match shape {
            Shape::Tight => width,
            Shape::Padded => (device.linear_texture_bytes_per_row(&descriptor).unwrap())
                .filter(|pitch| pitch.is_multiple_of(16) && *pitch > width)
                .unwrap_or(1024),
        };
        // The buffer's lines start at its first byte, or `K` where laid (`aligned_byte`).
        let first = if laid { K } else { 0 };
        let layout = Layout::new((width, height, pitch), first, self.line);
        let made = BufferDescriptor::new(
            layout.length,
            descriptor.resource_options,
            BufferContents::Zero,
        );
        let made = if laid {
            made.with_aligned_byte(K)
        } else {
            made
        };
        let buffer = device.new_buffer_with_descriptor(made).unwrap();
        let shadowed = device.work_counts().shadowed_textures;
        let texture = (buffer
            .new_texture_with_iosurface_plane(&descriptor, IoSurfacePlane::new(K, pitch, true)))
        .unwrap();
        let shadowed = device.work_counts().shadowed_textures - shadowed;
        let zero = self.zero_texture(width, height);
        let mut found = Found::default();
        for round in 0..rounds {
            let back_words = (layout.back.end - layout.back.start) / 4;
            let inputs = kernel_inputs(round, back_words.max(K / 4));
            // What each byte must read after the round, and the pattern, which differs from it at
            // every byte a command writes.
            let mut pattern: Vec<u8> = (0..layout.length)
                .map(|at| pattern_byte(at, round))
                .collect();
            let mut want = pattern.clone();
            for at in 0..layout.length {
                let i = at as usize;
                match layout.region(at) {
                    1 => want[i] = plane.texel(),
                    0 | 3 => {
                        let start = if at < K { 0 } else { layout.back.start };
                        want[i] = match beside {
                            Beside::Blit | Beside::Left => pattern[i] % 251 + 1,
                            Beside::Kernel => {
                                let word = (at - start) / 4;
                                (inputs[word as usize] + word as f32).to_le_bytes()
                                    [((at - start) % 4) as usize]
                            }
                        };
                        if want[i] == pattern[i] {
                            pattern[i] = pattern[i] % 251 + 1;
                        }
                    }
                    _ => {}
                }
            }
            let source = self.shared(&pattern);
            self.blit(|blit| {
                blit.copy_from_buffer_to_buffer(&source, 0, &buffer, 0, layout.length)
                    .unwrap()
            });
            let wanted = self.shared(&want);
            let input_bytes: Vec<u8> = inputs.iter().flat_map(|word| word.to_le_bytes()).collect();
            let input = self.shared(&input_bytes);
            let command_buffer = self.queue.command_buffer().unwrap();
            match plane {
                Plane::Pass => {
                    let mut pass = RenderPassDescriptor::new();
                    let color = &mut pass.color_attachments[0];
                    color.texture = Some(texture.clone());
                    color.load_action = LoadAction::Clear;
                    color.store_action = StoreAction::Store;
                    color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
                    (command_buffer.render_command_encoder(&pass).unwrap())
                        .end_encoding()
                        .unwrap();
                }
                Plane::Kernel => {
                    let mut encoder = command_buffer.compute_command_encoder().unwrap();
                    encoder.set_compute_pipeline_state(&self.invert).unwrap();
                    encoder.set_texture(Some(&zero), 0).unwrap();
                    encoder.set_texture(Some(&texture), 1).unwrap();
                    (encoder.dispatch_threads(Size::new(width, height, 1), Size::new(8, 4, 1)))
                        .unwrap();
                    encoder.end_encoding().unwrap();
                }
            }
            let runs = [layout.front.clone(), layout.back.clone()];
            match beside {
                Beside::Blit => {
                    let mut blit = command_buffer.blit_command_encoder().unwrap();
                    for run in &runs {
                        let length = run.end - run.start;
                        (blit.copy_from_buffer_to_buffer(
                            &wanted, run.start, &buffer, run.start, length,
                        ))
                        .unwrap();
                    }
                    blit.end_encoding().unwrap();
                }
                Beside::Kernel => {
                    let mut encoder = command_buffer.compute_command_encoder().unwrap();
                    encoder
                        .set_compute_pipeline_state(&self.scale_and_offset)
                        .unwrap();
                    encoder.set_buffer(Some(&input), 0, 0).unwrap();
                    encoder.set_buffer(Some(&self.scale), 0, 2).unwrap();
                    for run in &runs {
                        let words = (run.end - run.start) / 4;
                        encoder.set_buffer(Some(&buffer), run.start, 1).unwrap();
                        (encoder.dispatch_threads(Size::new(words, 1, 1), Size::new(words, 1, 1)))
                            .unwrap();
                    }
                    encoder.end_encoding().unwrap();
                }
                Beside::Left => {}
            }
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None);
            let read = self.read_back(&buffer, layout.length);
            for at in 0..layout.length {
                let i = at as usize;
                if read[i] == want[i] {
                    continue;
                }
                let region = layout.region(at);
                match read[i] == pattern[i] {
                    true => found.lost[region] += 1,
                    false => found.wrong[region] += 1,
                }
                if found.first.is_none() {
                    found.first = Some((round, at, read[i], want[i], pattern[i]));
                }
            }
        }
        println!(
            "LINE-BESIDE {}: {shape:?} {width}x{height} at {pitch} over a {} {tracking:?} Private \
             buffer ({} bytes, lines of {} from byte {first}), {plane:?} then {beside:?}: aliases \
             {} ({shadowed} shadowed), front {:?}, back {:?}; {rounds} rounds: lost {:?}, wrong \
             {:?} of {REGIONS:?}, first (round, byte, read, want, pattern) {:?}",
            device.name(),
            if laid { "laid" } else { "plain" },
            layout.length,
            self.line,
            texture.aliases_buffer(),
            layout.front,
            layout.back,
            found.lost,
            found.wrong,
            found.first,
        );
        (layout, found)
    }
}

#[test]
fn a_write_beside_a_plane_over_private_memory_keeps_its_bytes_under_the_plane_s_writes() {
    let scratch = Scratch::new("line-beside");
    let device = device(&scratch.0);
    let settled = device.settle_in_place_checks(CHECK_WAIT);
    let examples = library(&device, "examples/runtime_examples.ll");
    let kernel = examples.new_function_with_name("invert_texels").unwrap();
    let invert = (device
        .new_compute_pipeline_state_with_function(&kernel, &PipelineOptions::default()))
    .unwrap();
    let scale_and_offset = state(&device);
    let bench = Bench {
        queue: device.new_command_queue().unwrap(),
        invert,
        scale_and_offset,
        scale: (device.new_buffer_with_bytes(&1f32.to_le_bytes(), ResourceOptions::shared()))
            .unwrap(),
        line: device.in_place_check_line().max(4),
        device,
    };
    println!(
        "LINE-BESIDE {}: in-place checks settled {settled}, line {}",
        bench.device.name(),
        bench.line
    );
    let mut failures = Vec::new();
    for shape in [Shape::Tight, Shape::Padded] {
        for laid in [true, false] {
            for tracking in [HazardTrackingMode::Tracked, HazardTrackingMode::Untracked] {
                for plane in [Plane::Pass, Plane::Kernel] {
                    for beside in [Beside::Blit, Beside::Kernel] {
                        let (_, found) =
                            bench.case((shape, laid, tracking), (plane, beside), ROUNDS);
                        if found.first.is_some() {
                            failures.push(format!(
                                "{shape:?} laid {laid} {tracking:?} {plane:?} then {beside:?}: \
                                 {found:?}"
                            ));
                        }
                    }
                }
            }
            // The control: no write beside, so each byte it would have written reads lost.
            let case = (shape, laid, HazardTrackingMode::Tracked);
            let (layout, found) = bench.case(case, (Plane::Kernel, Beside::Left), 1);
            let mut lost = [0; 5];
            lost[0] = layout.front.end - layout.front.start;
            lost[3] = layout.back.end - layout.back.start;
            assert_eq!(
                (found.lost, found.wrong),
                (lost, [0; 5]),
                "{shape:?} laid {laid}: the control reads every byte beside lost, and no other"
            );
        }
    }
    assert!(
        failures.is_empty(),
        "on {}, bytes a command wrote read otherwise:\n{}",
        bench.device.name(),
        failures.join("\n")
    );
}
