//! What a GPU write into memory whose backing is read-only reaches: a read taken in the same
//! command buffer, a read taken in a later one, and the pages themselves.
//!
//! The owner's rule is that where Metal drops such a write we drop it too, on every driver, with no
//! refusal — a real session died on the gap. The read-only state is Metal's private
//! `MTLResourceOptions` bit 24 ([`HostBufferSpec::read_only`]), the only way a Metal program can
//! say "the GPU's writes to this memory go nowhere".
//!
//! # What Metal does, measured on this M3 Max (macOS 27.0.1)
//!
//! `probes/readonly_gpu_writes.m` measured eleven shapes against the backings, one shape per
//! process, each against a writable twin of the same program, eight runs a process and separate
//! cold-process runs beside them. Metal's answer splits cleanly in two, and the split is **not**
//! between one kind of write and another but between **which engine writes**:
//!
//! - **What a shader or the blit engine writes is dropped, silently.** A blit fill, a kernel's
//!   stores, a vertex-stage store, a fragment-stage store, a kernel or fragment store through an
//!   address loaded out of an argument buffer, and a pass's visibility result all complete with
//!   **no error**, and neither a later command buffer nor the CPU sees any of it. The writable twin
//!   of each lands its write in all three places, which is what makes "dropped" a measurement
//!   rather than a program that wrote nothing.
//! - **What the render pass's own machinery writes FAULTS.** A colour attachment over such memory,
//!   a resolve into it, and a depth or a stencil attachment in a heap made with the bit all end
//!   their command buffer with `MTLCommandBufferErrorDomain` /
//!   `kIOGPUCommandBufferCallbackErrorPageFault`, on the first command buffer of a fresh process,
//!   every time — and **every later command buffer in that process then errors too**. None of those
//!   is a case here, for exactly that reason: a case that faults on every run and poisons its own
//!   process's queue would take every case after it down with it. They are measured in the probe
//!   and recorded in its header; `no_row_writes_through_an_attachment_or_a_resolve` holds this
//!   family to it.
//!
//! So the two cases here are the two benign shapes a committed fixture can reach: a **pass's
//! visibility result** written into read-only memory, and a **kernel's stores** into it.
//!
//! # Bit 24 is a validated stand-in for the SPI, not an assumed one
//!
//! Bit 24 is what this family's cases state, and it is not the state a consumer reaches in
//! production: that one is Apple's SPI, `detachBacking` then
//! `replaceBackingWithRanges:readOnly:YES` over the pages. The probe measured both forms over the
//! same programs, eight runs a shape, each against the writable control of its own creation path,
//! and they **agree on all ten shapes measurable under both** — byte for byte, error for error and
//! flap for flap. So a case stated in bit 24 is a case about the production state too. The SPI pair
//! behaves the same whichever call made the buffer, and a **detached** backing is a different state
//! again: a command buffer naming a detached resource ends in error where one naming a
//! read-only-replaced resource completes.
//!
//! These cases state bit 24 rather than the SPI because bit 24 is what the runtime's own public
//! API offers (`ResourceOptions::PRIVATE_BIT`); reaching the SPI from a case would need a `Side`
//! call for it on both legs, and the forms agreeing is what makes that unnecessary.
//!
//! # Every shape is asked twice, and the writable run is not breadth
//!
//! Against read-only memory alone, "the write was dropped" and "my program wrote nothing" read
//! identically: both leave the pages holding the pattern they started with. So each shape is a
//! **pair** of cases over the same program, differing in one bit of
//! [`HostBufferSpec::read_only`]. The writable row is the control that makes the read-only row a
//! measurement.
//!
//! # Both halves of the round trip are in one command buffer, and that is the subject
//!
//! A whole [`RenderWork`] list is one command buffer. The write and the
//! [`RenderWork::CopyBuffer`] that reads the memory back are in **one** list here, on purpose:
//! same-command-buffer visibility is the surprising half of the answer, and a case that committed
//! between the write and the read could not see it at all. The later-command-buffer read is a
//! second [`Side::render`] call, into a second destination, so the two reads are two sites and
//! never one.
//!
//! # The same-command-buffer read depends on what ran before it, and the recording says so
//!
//! Measured over eight runs a shape, the **later-command-buffer** and **CPU** answers never moved:
//! the write is gone from both, in every run of every dropped shape. The **same-command-buffer**
//! answer moved, and which way depends on the engine and on what ran before in the same process:
//!
//! - A blit fill and both compute stores were visible there in every run, the **first run of a
//!   fresh process included** (three fresh processes each).
//! - The visibility result and all three draw-stage stores were **not** visible there in the first
//!   run of a fresh process (twelve fresh processes, every one), and were visible in seven runs of
//!   eight once the process had run the shape before.
//!
//! That is what the runtime's own `read_only_view.rs` already says in the word **may**: "later
//! commands of the same command buffer *may* read some back from the GPU's caches". It is not a
//! contract to reproduce — it is unstable on Metal itself.
//!
//! So [`Case::record_runs`] is `RUNS` here rather than one: a case whose answer moves records
//! every transcript Metal produced as its outcome set, and a replay matching any of them passes.
//! That is not a loosening — an outcome is a **whole** transcript, and both of Metal's hold the
//! same later-command-buffer and CPU answers, so our run still has to drop the write to match
//! either one.
//!
//! **These four recordings in fact came out deterministic**, one outcome over eight runs each, with
//! the same-command-buffer site holding the written bytes: the suite's own program differs from the
//! probe's in enough ways (it copies four kilobytes where the probe copies sixty-four, and it makes
//! its library and pipeline first) that the cold reading did not reproduce here. That is the reason
//! to leave `RUNS` at eight rather than to drop it to one: the answer is known to move, so a
//! recording of one run would hold every later run to whichever side of it that run happened to
//! see.
//!
//! # The window is wider than the write, because the bytes that did not move are the answer
//!
//! Each site reads `WINDOW` bytes from the start of the memory, not the span a shape writes. The
//! write lands at `OFFSET`, so the window holds 512 bytes before it and over three kilobytes
//! past it that no shape names. A write that ran outside what the case asked for is visible there
//! and nowhere else, and for read-only memory the untouched bytes **are** the result.
//!
//! The pages start as `PAD` rather than zero: against zeroed pages a write of zeros and a write
//! that was dropped would read alike. Each read-back destination starts as `UNREAD`, a different
//! byte again, so a copy that moved nothing at all is visible as itself rather than as "the memory
//! was untouched".
//!
//! # Two measured shapes of this subject are not cases, and want a fixture
//!
//! A **draw-stage store** and a **store through an address loaded out of an argument buffer** were
//! both measured as dropped with no error, and neither is here: no owned fixture's vertex or
//! fragment function stores into a bound buffer, so a case of either wants a fixture of its own.
//!
//! Nothing here can fault a GPU, hang one or trip a watchdog: one 4 x 4 target, three vertices, one
//! dispatch of 64 threads, no loop, two command buffers over 16 KiB of memory committed and waited
//! for. The shapes that do fault are measured in the probe and are deliberately absent.

use crate::allowance::Allowance;
use crate::case::{call, made, Abort, Case, Failure, Handle, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::host_pages::{HostBufferSpec, PAGE_MULTIPLE};
use crate::recording::ShaderRef;
use crate::render::{
    ColorState, Dispatch, PassColor, RenderPass, RenderPipelineSpec, RenderStep, RenderWork,
    TargetSpec,
};
use crate::transcript::{ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan_runtime::VisibilityResultMode;

/// How many times record mode runs each case.
///
/// Eight, because the same-command-buffer read of a dropped write is not stable: it held the
/// written bytes in seven runs of eight and the original pattern in the first run of a fresh
/// process. One run would record whichever of those it happened to see and hold every later run
/// to it.
const RUNS: u32 = 8;

/// How many bytes of host memory a case hands the device: one page of the largest page in play,
/// which is the smallest a [`HostBufferSpec`] may be, so every venue is handed the same bytes.
const BYTES: u64 = PAGE_MULTIPLE;

/// How many of those bytes each read-back site holds.
///
/// Wider than anything a shape writes, so a write that left the span the case asked for is visible.
/// Not the whole 16 KiB: a site is compared exactly, and three sites a case over four cases of
/// 16 KiB each would put bytes in the recordings to say what 4 KiB already says.
const WINDOW: usize = 4096;

/// Where every shape aims its write in the memory.
///
/// 512 is a multiple of 8, which a visibility result's offset needs, and of 4, which a word store
/// needs, and it is far enough in that a write running backwards off its span shows in bytes no
/// case named.
const OFFSET: u64 = 512;

/// The target the visibility shape draws into, in texels. Its 16 pixels are the 16 samples a
/// covering draw passes at one sample a pixel, so the result word is 16.
const SIDE: u32 = 4;

/// The format the visibility shape's target is of.
const FORMAT: PixelFormat = PixelFormat::RGBA8Unorm;

/// The byte every one of the host pages holds before anything runs.
///
/// **Never zero.** A write of zeros and a write that was dropped read alike against zeroed pages,
/// which is exactly the distinction this family exists to make.
const PAD: u8 = 0xab;

/// The byte each read-back destination holds before the copy.
///
/// A third value, so a copy that moved nothing at all reads as itself rather than as "the memory
/// kept its pattern".
const UNREAD: u8 = 0x5c;

/// The clear colour the visibility shape's pass uses, as
/// `MTLRenderPassColorAttachmentDescriptor` takes it: four doubles.
///
/// Its target is the device's own memory and nothing reads its texels, so one thing matters about
/// these: they are stated, so both legs clear the same value and the case's fingerprint covers it.
const CLEAR: [f64; 4] = [0.125, 0.375, 0.625, 0.875];

/// The owned fixture the visibility shape draws with, which the visibility family already uses:
/// `rt6_full` takes a depth and a slope as one `float2` at vertex buffer 0, and `rt6_white`
/// returns white. The fullscreen triangle covers all 16 pixels of a 4 x 4 target.
const DRAW_FIXTURE: &str = "rt6_render_state";
const VERTEX: &str = "rt6_full";
const FRAGMENT: &str = "rt6_white";

/// The depth and slope `rt6_full` takes: at the near plane, flat, so every pixel is covered and
/// every sample passes.
const DEPTH_AND_SLOPE: [f32; 2] = [0.0, 0.0];

/// The owned fixture the kernel shape stores with: `out[t] = in[3t + 1]`, the input at Metal buffer
/// 0 and the output at buffer 1.
///
/// Chosen because it is already committed and golden-blessed, and because its store is a plain
/// `device uint *` write of one word a thread — the simplest write into read-only memory a shader
/// can make, with no texture and no second effect to confuse the answer.
const KERNEL_FIXTURE: &str = "a4_index_word";
const KERNEL: &str = "a4_index_word";

/// How many words the kernel writes, one a thread.
const KERNEL_WORDS: u32 = 64;

/// How many threads a threadgroup of that dispatch has.
const KERNEL_THREADS: u32 = 32;

/// The words the kernel reads: `in[i] = 0xc0de_0000 + i`, so `out[t]` is `0xc0de_0000 + 3t + 1` and
/// each written word says which thread wrote it and where it read from.
///
/// Long enough for the highest index the dispatch reads, `3 * (KERNEL_WORDS - 1) + 1`, because a
/// short input would have the last threads reading past it — a question about out-of-bounds reads
/// and not the one this case asks.
fn kernel_input() -> Vec<u8> {
    let words = 3 * (KERNEL_WORDS - 1) + 2;
    (0..words)
        .flat_map(|i| (0xc0de_0000u32 + i).to_le_bytes())
        .collect()
}

/// What a case writes into the memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    /// A pass whose visibility-result buffer is the memory, counting the samples of one draw over
    /// a target of the device's own memory.
    Visibility,
    /// One dispatch whose kernel stores a word a thread into the memory.
    KernelStore,
}

impl Shape {
    /// The owned fixture the shape runs.
    const fn fixture(self) -> &'static str {
        match self {
            Shape::Visibility => DRAW_FIXTURE,
            Shape::KernelStore => KERNEL_FIXTURE,
        }
    }

    /// The entry point, or pair of them, the shape runs, as its recording names it.
    fn entry(self) -> String {
        match self {
            Shape::Visibility => format!("{VERTEX}+{FRAGMENT}"),
            Shape::KernelStore => KERNEL.to_owned(),
        }
    }
}

/// One case: a shape, the backing it writes into, and the name both give it.
struct Row {
    /// The case name, as the report and the recording spell it. A literal rather than something
    /// built from the fields, so a recording's file name is a string a reader can grep for here.
    case: &'static str,
    shape: Shape,
    /// Whether the memory is made with Metal's private `MTLResourceOptions` bit 24.
    read_only: bool,
}

/// The four cases: two shapes, each over read-only memory and each over writable memory.
const ROWS: [Row; 4] = [
    Row {
        case: "host-page-writes-visibility-read-only",
        shape: Shape::Visibility,
        read_only: true,
    },
    Row {
        case: "host-page-writes-visibility-writable",
        shape: Shape::Visibility,
        read_only: false,
    },
    Row {
        case: "host-page-writes-kernel-store-read-only",
        shape: Shape::KernelStore,
        read_only: true,
    },
    Row {
        case: "host-page-writes-kernel-store-writable",
        shape: Shape::KernelStore,
        read_only: false,
    },
];

/// Every case of the family.
pub fn all() -> Vec<Box<dyn Case>> {
    ROWS.iter()
        .map(|row| Box::new(HostPageWrite { row }) as _)
        .collect()
}

/// A case: one program that writes into host memory of one backing, read back three ways.
pub struct HostPageWrite {
    row: &'static Row,
}

impl Row {
    /// The host memory the case hands the device.
    const fn memory(&self) -> HostBufferSpec {
        let spec = HostBufferSpec::writable(BYTES, PAD);
        match self.read_only {
            true => spec.read_only(),
            false => spec,
        }
    }

    /// The target the visibility shape draws into: the device's own memory, so the pass's only
    /// write into the pages is the result word.
    ///
    /// `None` for the kernel shape, which attaches nothing at all. That is the property
    /// `no_row_writes_through_an_attachment_or_a_resolve` holds: no row of this family puts an
    /// attachment or a resolve target in the read-only memory, because Metal faults on those.
    fn target(&self) -> Option<TargetSpec> {
        match self.shape {
            Shape::Visibility => Some(TargetSpec::plain(FORMAT, [SIDE, SIDE])),
            Shape::KernelStore => None,
        }
    }

    /// The one colour attachment the drawing pipeline writes, held so a spec can borrow it.
    fn color_state(&self) -> [Option<ColorState>; 1] {
        [Some(ColorState::new(FORMAT))]
    }

    /// The render pipeline the shape draws with, where it draws.
    fn pipeline<'a>(&self, colors: &'a [Option<ColorState>]) -> Option<RenderPipelineSpec<'a>> {
        match self.shape {
            Shape::Visibility => Some(RenderPipelineSpec::new(VERTEX, FRAGMENT, colors)),
            Shape::KernelStore => None,
        }
    }

    /// The pass the visibility shape runs: its colour attachment cleared, the memory as its
    /// visibility-result buffer, and one covering draw counted at [`OFFSET`].
    fn pass(&self, target: Handle, memory: Handle, state: Handle) -> RenderPass {
        RenderPass::clearing(vec![PassColor::cleared(target, CLEAR)])
            .counting_into(memory)
            .drawing(vec![
                RenderStep::Pipeline(state),
                RenderStep::VertexBytes {
                    bytes: DEPTH_AND_SLOPE
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect(),
                    index: 0,
                },
                RenderStep::VisibilityResultMode {
                    mode: VisibilityResultMode::Counting,
                    offset: OFFSET,
                },
                RenderStep::triangle(),
                RenderStep::VisibilityResultMode {
                    mode: VisibilityResultMode::Disabled,
                    offset: OFFSET,
                },
            ])
    }

    /// The dispatch the kernel shape runs: the input at Metal buffer 0 and the memory at buffer 1,
    /// from [`OFFSET`], one word a thread.
    fn dispatch(&self, input: Handle, memory: Handle, state: Handle) -> Dispatch {
        Dispatch {
            pipeline: state,
            buffers: vec![(input, 0), (memory, OFFSET)],
            textures: Vec::new(),
            grid: [KERNEL_WORDS, 1, 1],
            threads: [KERNEL_THREADS, 1, 1],
        }
    }

    /// What the dispatch's own numbers fingerprint to: its grid, its threadgroup and the offset
    /// the store lands at. No handle, which is a side's own.
    fn dispatch_words(&self) -> Vec<u8> {
        let dispatch = self.dispatch(Handle(0), Handle(0), Handle(0));
        [dispatch.grid, dispatch.threads]
            .into_iter()
            .flatten()
            .flat_map(u32::to_le_bytes)
            .chain(OFFSET.to_le_bytes())
            .collect()
    }

    /// The window of memory a read-back site holds, as rows of 32 bytes.
    ///
    /// **Every byte is an element**, so the bytes no shape names compare like any other: the 512
    /// before the write and every byte past it. That is the whole reason the memory is read rather
    /// than the thing written.
    fn layout(&self) -> ByteLayout {
        let window = u32::try_from(WINDOW).expect("the window in a u32");
        let row = 32;
        ByteLayout {
            region: Region::Buffer {
                offset: 0,
                length: WINDOW as u64,
            },
            element: Element::U8,
            row_offset: 0,
            row_elements: row,
            row_pitch: row,
            rows: window / row,
            image_pitch: window,
            images: 1,
            // A row's elements are its whole pitch: this view divides the window up, it does not
            // lay anything over it, so nothing lies between these rows.
            gaps: Gaps::Ignored,
        }
    }
}

impl Case for HostPageWrite {
    fn name(&self) -> &'static str {
        self.row.case
    }

    fn metal_source(&self) -> MetalSource {
        // Every call is public but one, and that one is a number rather than a selector: the
        // private `MTLResourceOptions` bit 24 of `newBufferWithBytesNoCopy:length:options:
        // deallocator:`, whose options argument takes a raw value like any other.
        // `renderCommandEncoderWithDescriptor:`, `setVisibilityResultMode:offset:`,
        // `dispatchThreads:threadsPerThreadgroup:`,
        // `copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:` and the buffer's own
        // `contents` are all public.
        MetalSource::Run
    }

    fn record_runs(&self) -> u32 {
        // The same-command-buffer read of a dropped write is not stable (the module docs), so the
        // recording holds the set of transcripts Metal produced rather than whichever one a single
        // run happened to see.
        RUNS
    }

    fn allowances(&self) -> Vec<Allowance> {
        // **One declaration per code a venue was measured announcing for this row**, and nothing
        // else. A notice our side announces that this list does not name is a divergence at the
        // notices site; a code this list names that no run announces is an unfired allowance,
        // which is noise the tally exists to show. So this is a measured set and not a guess.
        //
        // Measured on this M3 Max through MoltenVK: the two **drawing** rows announce
        // `fast-math-rewrites` and nothing else -- the translator rewrites the arithmetic of
        // `rt6_render_state`'s vertex function, which is the fixture, not this family's subject --
        // and the two **kernel** rows announce nothing at all, so they declare nothing. The first
        // replay of this family declared neither and reported both drawing rows divergent at
        // three sites of `case/notices` with no byte of any read-back moving.
        //
        // `private-resource-option` **has** its venue now, which is what the earlier reading of
        // this comment was waiting for. On macOS a no-copy buffer with bit 24 announces nothing:
        // the runtime imports the pages through a read-only alias of them and
        // `Device::no_copy_options` returns without announcing. Off macOS the GPU's writes reach
        // the pages and the notice fires, and it fired on two drivers at once -- the RTX 5080
        // Laptop (615.71.09) and llvmpipe (Mesa 26.2.4) -- for
        // `host-page-writes-visibility-read-only` and `host-page-writes-kernel-store-read-only`
        // and for neither writable row (replay-nvidia.log, replay-llvmpipe.log). So it is declared
        // from `Row::read_only`, which is the bit it is about.
        //
        // `shader-color-conversion` is the two **drawing** rows': they store into a normalized
        // colour attachment, which our side may round in the shader. Announced by both on both
        // drivers; neither kernel row stores a colour.
        //
        // `shared-in-system-memory` is every row's and belongs to the venue, not to this family:
        // each row reads its pages back through a `Shared` buffer
        // (`crate::replay::shared_memory_notice`). Announced by all four on the 5080.
        //
        // **None of these loosens a byte**, and that matters most here: the read-only rows' pages
        // really do differ on both drivers (`case/bytes-the-cpu-reads-bytes`, 8 bytes, Metal 171
        // against ours 16), which is this family's whole subject. Those bytes stay compared
        // exactly and those rows stay divergent on them.
        let mut named = vec![crate::replay::shared_memory_notice()];
        if matches!(self.row.shape, Shape::Visibility) {
            named.push(crate::replay::notice_allowance(
                metal2vulkan_runtime::Degradation::FastMathRewrites.code(),
            ));
            named.push(crate::replay::notice_allowance(
                metal2vulkan_runtime::Degradation::ShaderColorConversion.code(),
            ));
        }
        if self.row.read_only {
            named.push(crate::replay::notice_allowance(
                metal2vulkan_runtime::Degradation::PrivateResourceOption.code(),
            ));
        }
        named
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let fixture = Fixture::new(self.row.shape.fixture());
        let entry = self.row.shape.entry();
        [RUNTIME_TEXT, METAL_TEXT]
            .into_iter()
            .map(|extension| {
                Ok(ShaderRef {
                    fixture: format!("{}{extension}", fixture.name),
                    entry: entry.clone(),
                    source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
                })
            })
            .collect()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        // What the case states and nothing a side owns: the memory's own three facts, the fixture
        // in both dialects, and the shape of the work. A handle's number is a side's own, so a
        // placeholder stands in for every one of them here.
        let placeholder = Handle(0);
        let fixture = Fixture::new(self.row.shape.fixture());
        let mut inputs = vec![
            Input::bytes("memory", &self.row.memory().words()),
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
        ];
        match self.row.shape {
            Shape::Visibility => {
                let colors = self.row.color_state();
                let state = self
                    .row
                    .pipeline(&colors)
                    .ok_or("the visibility shape states a pipeline")?;
                let target = self
                    .row
                    .target()
                    .ok_or("the visibility shape states a target")?;
                inputs.push(Input::bytes("pipeline", &state.words()));
                inputs.push(Input::bytes("target", &target.words()));
                inputs.push(Input::bytes(
                    "pass",
                    &self.row.pass(placeholder, placeholder, placeholder).words(),
                ));
            }
            Shape::KernelStore => {
                inputs.push(Input::bytes("kernel-input", &kernel_input()));
                inputs.push(Input::bytes("dispatch", &self.row.dispatch_words()));
            }
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let memory = made(sites, "host-buffer", side.host_buffer(&self.row.memory()))?;
        let loaded = Fixture::new(self.row.shape.fixture())
            .loaded()
            .map_err(|why| Abort {
                at: "library".to_owned(),
                failure: Failure::named("FixtureNotRead", why),
            })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;

        // The destinations the GPU copies the memory into. Ordinary device memory, pre-filled with
        // a third byte, so a copy that moved nothing reads as itself.
        let same = made(
            sites,
            "same-destination",
            side.buffer(&vec![UNREAD; WINDOW]),
        )?;
        let later = made(
            sites,
            "later-destination",
            side.buffer(&vec![UNREAD; WINDOW]),
        )?;

        let write = match self.row.shape {
            Shape::Visibility => {
                let colors = self.row.color_state();
                let spec = self
                    .row
                    .pipeline(&colors)
                    .expect("the visibility shape states a pipeline");
                let state = made(
                    sites,
                    "render-pipeline",
                    side.render_pipeline(library, &spec),
                )?;
                let target = self
                    .row
                    .target()
                    .expect("the visibility shape states a target");
                let target = made(sites, "target", side.target(&target))?;
                RenderWork::Pass(self.row.pass(target, memory, state))
            }
            Shape::KernelStore => {
                let state = made(
                    sites,
                    "compute-pipeline",
                    side.compute_pipeline(library, KERNEL),
                )?;
                let input = made(sites, "kernel-input", side.buffer(&kernel_input()))?;
                RenderWork::Dispatch(self.row.dispatch(input, memory, state))
            }
        };

        // ONE command buffer: the write, then the copy that reads the memory back. Metal orders a
        // command buffer's encoders, so this copy sees whatever the write left for a later command
        // of the write's own command buffer -- which is the half of the answer a case that
        // committed in between could not see at all, and the half that is not stable.
        let work = vec![
            write,
            RenderWork::CopyBuffer {
                from: memory,
                from_offset: 0,
                into: same,
                into_offset: 0,
                length: WINDOW as u64,
            },
        ];
        let wrote = call(sites, "write-and-read", side.render(&work), |_| {
            Value::Absent
        })?;
        super::extended_range_targets::program(sites, "write-program", &wrote, &work);

        // A second command buffer, which is where the measured answer says the write is gone.
        let read = vec![RenderWork::CopyBuffer {
            from: memory,
            from_offset: 0,
            into: later,
            into_offset: 0,
            length: WINDOW as u64,
        }];
        let afterwards = call(sites, "read-again", side.render(&read), |_| Value::Absent)?;
        super::extended_range_targets::program(sites, "read-program", &afterwards, &read);

        for (at, from) in [
            ("bytes-in-the-same-command-buffer", same),
            ("bytes-in-a-later-command-buffer", later),
            ("bytes-the-cpu-reads", memory),
        ] {
            // Each name is its own: three reads of the same bytes are three questions, and two
            // sites of one name at one path is a site a comparison cannot settle.
            let bytes = call(sites, at, side.read_buffer(from, 0, WINDOW), |_| {
                Value::Absent
            })?;
            sites.push(Node::leaf(
                format!("{at}-bytes"),
                Observed::bytes(bytes, self.row.layout()),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// Every row's name is its own, and every shape is asked both ways.
    ///
    /// The pairing is the point: a read-only row with no writable twin is a row whose answer
    /// cannot be told from "the program wrote nothing", which is the reading this family exists to
    /// rule out.
    #[test]
    fn every_shape_is_asked_read_only_and_writable() {
        let names: BTreeSet<&str> = ROWS.iter().map(|row| row.case).collect();
        assert_eq!(names.len(), ROWS.len(), "{names:?}");
        for shape in [Shape::Visibility, Shape::KernelStore] {
            let asked: BTreeSet<bool> = ROWS
                .iter()
                .filter(|row| row.shape == shape)
                .map(|row| row.read_only)
                .collect();
            assert_eq!(
                asked,
                BTreeSet::from([false, true]),
                "{shape:?} is not asked both ways, so its read-only answer has no control"
            );
        }
    }

    /// A row's name says which backing it is, so a reader of the report cannot mistake the control
    /// for the measurement.
    #[test]
    fn a_rows_name_says_which_backing_it_is() {
        for row in &ROWS {
            let expected = match row.read_only {
                true => "-read-only",
                false => "-writable",
            };
            assert!(
                row.case.ends_with(expected),
                "{} does not end in {expected}",
                row.case
            );
        }
    }

    /// No row of this family puts a render attachment or a resolve target in the read-only memory.
    ///
    /// Measured on this M3 Max: a colour attachment over such memory, a resolve into it, and a
    /// depth or stencil attachment in a heap made with the bit all end their command buffer with
    /// `kIOGPUCommandBufferCallbackErrorPageFault`, and every later command buffer in the same
    /// process then errors too. So a case of one would take every case after it down with it. Held
    /// by a test because the way such a shape gets in is somebody adding a row, not somebody
    /// changing the vocabulary: a row's attachment is the device's own memory
    /// ([`Row::target`]) or there is no attachment at all, and nothing here names the host buffer
    /// as one.
    #[test]
    fn no_row_writes_through_an_attachment_or_a_resolve() {
        for row in &ROWS {
            let attached = row.target();
            match row.shape {
                // Its colour attachment is a target of the device's own memory -- it states no
                // backing -- so the only write into the pages is the result word.
                Shape::Visibility => {
                    let spec = attached.unwrap_or_else(|| panic!("{}", row.case));
                    assert!(
                        spec.over_buffer.is_none(),
                        "{}'s attachment lies over a buffer, which Metal faults on",
                        row.case
                    );
                    assert!(spec.surface.is_none(), "{}", row.case);
                }
                // It attaches nothing at all.
                Shape::KernelStore => assert!(attached.is_none(), "{}", row.case),
            }
        }
    }

    /// The memory each row states is one every venue can map, and the read-back window fits inside
    /// it with room past the write that no shape names.
    #[test]
    fn the_memory_is_mappable_and_the_window_fits_inside_it() {
        for row in &ROWS {
            assert_eq!(row.memory().unmappable(), None, "{}", row.case);
            assert_eq!(row.memory().pattern, PAD, "{}", row.case);
            assert_eq!(row.memory().read_only, row.read_only, "{}", row.case);
        }
        // Read out of the layout each row actually emits, rather than off the constants, so this
        // holds the site a comparison walks and not two numbers a compiler folds together.
        let written = u64::from(KERNEL_WORDS) * 4;
        for row in &ROWS {
            let Region::Buffer { offset, length } = row.layout().region else {
                panic!("{} reads something other than a buffer", row.case)
            };
            // The window lies inside the memory, the whole of the widest write lies inside the
            // window, and bytes of the window lie BEFORE the write as well as after it: that last
            // one is what makes a write that ran backwards off its span visible.
            assert_eq!(offset, 0, "{}", row.case);
            assert!(length <= BYTES, "{} reads {length} of {BYTES}", row.case);
            assert!(
                OFFSET > 0 && OFFSET + written < length,
                "{}: the write at {OFFSET} of {written} bytes does not sit inside {length} with \
                 bytes either side of it",
                row.case
            );
        }
    }

    /// The pad and the unread byte differ, and the offset is aligned for every use a row makes of
    /// it.
    ///
    /// The equality would hide an answer: a destination byte equal to the pad would make a copy
    /// that moved nothing read as memory that kept its pattern.
    #[test]
    fn the_two_marker_bytes_differ_and_the_offset_is_aligned() {
        assert_ne!(PAD, UNREAD);
        for (unit, what) in [(8, "a visibility result's offset"), (4, "a word store")] {
            assert!(
                OFFSET.is_multiple_of(unit),
                "{OFFSET} is not aligned for {what}, which needs {unit}"
            );
        }
    }

    /// The kernel's input is long enough for the highest word the dispatch reads, and the dispatch
    /// divides evenly into threadgroups.
    ///
    /// `a4_index_word` computes `out[t] = in[3t + 1]`, so a short input would have the last threads
    /// reading past it -- which is a question about out-of-bounds reads and not the one this case
    /// asks.
    #[test]
    fn the_kernels_input_covers_every_index_it_reads() {
        let highest = usize::try_from(3 * (KERNEL_WORDS - 1) + 1).expect("an index in a usize");
        let words = kernel_input().len() / 4;
        assert!(
            words > highest,
            "{words} words do not cover index {highest}"
        );
        assert!(KERNEL_WORDS.is_multiple_of(KERNEL_THREADS));
    }
}
