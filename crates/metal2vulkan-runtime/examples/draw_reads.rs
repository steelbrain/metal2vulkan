//! Not a consumer's example: what the GPU pays a draw for reading the draw's own data, the tables
//! and bytes the runtime writes into the command buffer's upload arena (`upload.rs`), by what the
//! draw's fragment function reads.
//!
//! ```sh
//! cargo run --release -p metal2vulkan-runtime --example draw_reads -- \
//!     [--draws N] [--tile T] [--side S] [--runs R] [--only READS,...]
//! ```
//!
//! Each run is one command buffer of one render pass over an `S` by `S` `RGBA8Unorm` target
//! (1024 by default), cleared and stored, of `N` draws (2048 by default), each a 4-vertex
//! triangle strip over the next `T` by `T` tile of the target (64 by default), in rows. Each draw
//! sets its tile's corner and side as vertex bytes at index 0, which `tile_vertex` reads, then
//! what its fragment function reads at fragment index 0, by `READS`:
//!
//! - **`bytes`**: its color, as fragment bytes (`direct_fragment`, a `constant float4`);
//! - **`direct`**: the color, at the next of 256 slots of a `Private` buffer
//!   (`set_fragment_buffer`, `direct_fragment`);
//! - **`ptr`**: the address of that color, at the next of 256 slots of another `Private` buffer
//!   (`holder_fragment`: a `Holder` of one pointer, as an argument buffer holds a material's
//!   parameters): every fragment loads the pointer and reads the color through it, which the
//!   resident set resolves (`metal2vulkan::bind::ResidentSet`).
//!
//! Every color is opaque magenta, and once a kind's runs ended its target is read back: every
//! texel must hold it.
//!
//! Each line is the median over `R` runs (9 by default) after one not counted, of the command
//! buffer's GPU time (`gpu_start_time` to `gpu_end_time`) a draw, with the runtime's counts a
//! command buffer that moved over the counted runs (`WorkCounts`, by name from their `Debug`
//! form):
//!
//! ```text
//! DRAW-READS <reads> draws <N> tile <T> side <S> gpu_us_per_draw <median> least <least> <device>
//! DRAW-READS-COUNTS <reads> <name>=<per command buffer> ...
//! ```
//!
//! The functions are LLVM text here, assembled by `llvm-as` from `PATH`.

use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_runtime::{
    BlitOption, Buffer, ClearColor, CommandBuffer, CommandBufferStatus, Device, LoadAction, Origin,
    PipelineOptions, RenderPassDescriptor, RenderPipelineState, ResourceOptions, Size, StoreAction,
    Texture, TextureDescriptor, WorkCounts,
};
use std::collections::BTreeMap;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Duration;

/// The example's result: every runtime refusal is a typed `metal2vulkan_runtime::Error`.
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// The longest the example waits for a command buffer.
const WAIT: Duration = Duration::from_secs(60);

/// The slots of the colors' buffer and of the holders' buffer.
const SLOTS: u64 = 256;

/// Every draw's color: opaque magenta, exact in `RGBA8Unorm`.
const COLOR: [f32; 4] = [1.0, 0.0, 1.0, 1.0];
const TEXEL: [u8; 4] = [255, 0, 255, 255];

/// `tile_vertex`: vertex `id`'s corner of the tile `rect` (x, y, width, height in normalized
/// device coordinates) sets at index 0; `direct_fragment`: the `float4` at fragment index 0;
/// `holder_fragment`: the `float4` the pointer of the `Holder` at fragment index 0 points at.
const FUNCTIONS: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\n\
target triple = \"air64_v28-apple-macosx26.0.0\"\n\n\
%struct.Holder = type { ptr addrspace(1) }\n\n\
define <4 x float> @tile_vertex(ptr addrspace(2) noundef %rect, i32 noundef %id) {\n\
entry:\n\
  %r = load <4 x float>, ptr addrspace(2) %rect, align 16\n\
  %bx = and i32 %id, 1\n\
  %by = lshr i32 %id, 1\n\
  %fx = call float @air.convert.f.f32.u.i32(i32 %bx)\n\
  %fy = call float @air.convert.f.f32.u.i32(i32 %by)\n\
  %x0 = extractelement <4 x float> %r, i64 0\n\
  %y0 = extractelement <4 x float> %r, i64 1\n\
  %w = extractelement <4 x float> %r, i64 2\n\
  %h = extractelement <4 x float> %r, i64 3\n\
  %dx = fmul float %fx, %w\n\
  %dy = fmul float %fy, %h\n\
  %x = fadd float %x0, %dx\n\
  %y = fadd float %y0, %dy\n\
  %p0 = insertelement <4 x float> <float 0.0, float 0.0, float 0.0, float 1.0>, float %x, i64 0\n\
  %p = insertelement <4 x float> %p0, float %y, i64 1\n\
  ret <4 x float> %p\n\
}\n\n\
define <4 x float> @direct_fragment(ptr addrspace(2) noundef %c) {\n\
entry:\n\
  %v = load <4 x float>, ptr addrspace(2) %c, align 16\n\
  ret <4 x float> %v\n\
}\n\n\
define <4 x float> @holder_fragment(ptr addrspace(1) noundef readonly %h) {\n\
entry:\n\
  %hp = getelementptr inbounds %struct.Holder, ptr addrspace(1) %h, i64 0, i32 0\n\
  %b = load ptr addrspace(1), ptr addrspace(1) %hp, align 8\n\
  %v = load <4 x float>, ptr addrspace(1) %b, align 16\n\
  ret <4 x float> %v\n\
}\n\n\
declare float @air.convert.f.f32.u.i32(i32)\n\n\
!llvm.module.flags = !{!0, !1}\n!air.vertex = !{!20}\n!air.fragment = !{!30, !40}\n\
!air.version = !{!8}\n!air.language_version = !{!9}\n\
!0 = !{i32 7, !\"air.max_device_buffers\", i32 31}\n\
!1 = !{i32 7, !\"air.max_constant_buffers\", i32 31}\n\
!8 = !{i32 2, i32 8, i32 0}\n!9 = !{!\"Metal\", i32 4, i32 0, i32 0}\n\
!20 = !{ptr @tile_vertex, !21, !23}\n\
!21 = !{!22}\n\
!22 = !{!\"air.position\", !\"air.arg_type_name\", !\"float4\"}\n\
!23 = !{!24, !25}\n\
!24 = !{i32 0, !\"air.buffer\", !\"air.buffer_size\", i32 16, !\"air.location_index\", i32 0, i32 1, !\"air.read\", !\"air.address_space\", i32 2, !\"air.arg_type_size\", i32 16, !\"air.arg_type_align_size\", i32 16, !\"air.arg_type_name\", !\"float4\", !\"air.arg_name\", !\"rect\"}\n\
!25 = !{i32 1, !\"air.vertex_id\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"id\"}\n\
!26 = !{!27}\n\
!27 = !{!\"air.render_target\", i32 0, i32 0, !\"air.arg_type_name\", !\"float4\"}\n\
!30 = !{ptr @direct_fragment, !26, !31}\n\
!31 = !{!32}\n\
!32 = !{i32 0, !\"air.buffer\", !\"air.buffer_size\", i32 16, !\"air.location_index\", i32 0, i32 1, !\"air.read\", !\"air.address_space\", i32 2, !\"air.arg_type_size\", i32 16, !\"air.arg_type_align_size\", i32 16, !\"air.arg_type_name\", !\"float4\", !\"air.arg_name\", !\"c\"}\n\
!40 = !{ptr @holder_fragment, !26, !41}\n\
!41 = !{!42}\n\
!42 = !{i32 0, !\"air.indirect_buffer\", !\"air.buffer_size\", i32 8, !\"air.location_index\", i32 0, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.struct_type_info\", !43, !\"air.arg_type_size\", i32 8, !\"air.arg_type_align_size\", i32 8, !\"air.arg_type_name\", !\"Holder\", !\"air.arg_name\", !\"h\"}\n\
!43 = !{i32 0, i32 8, i32 0, !\"float4\", !\"p\", !\"air.indirect_argument\", !44}\n\
!44 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 16, !\"air.arg_type_align_size\", i32 16, !\"air.arg_type_name\", !\"float4\", !\"air.arg_name\", !\"p\"}\n";

/// What a draw's fragment function reads.
#[derive(Clone, Copy, PartialEq)]
enum Reads {
    Bytes,
    Direct,
    Pointer,
}

impl Reads {
    const ALL: [Reads; 3] = [Reads::Bytes, Reads::Direct, Reads::Pointer];

    fn name(self) -> &'static str {
        match self {
            Reads::Bytes => "bytes",
            Reads::Direct => "direct",
            Reads::Pointer => "ptr",
        }
    }
}

/// The run's shape: draws a command buffer, a tile's and the target's side in texels, timed runs,
/// and the kinds to run.
struct Shape {
    draws: u32,
    tile: u64,
    side: u64,
    runs: usize,
    only: Vec<String>,
}

impl Shape {
    fn from_args() -> Result<Shape> {
        let mut shape = Shape {
            draws: 2048,
            tile: 64,
            side: 1024,
            runs: 9,
            only: Vec::new(),
        };
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            let mut value = || args.next().ok_or_else(|| format!("{arg} takes a value"));
            match arg.as_str() {
                "--draws" => shape.draws = value()?.parse()?,
                "--tile" => shape.tile = value()?.parse()?,
                "--side" => shape.side = value()?.parse()?,
                "--runs" => shape.runs = value()?.parse()?,
                "--only" => shape.only = value()?.split(',').map(str::to_owned).collect(),
                other => return Err(format!("unknown argument {other}").into()),
            }
        }
        if shape.tile == 0 || !shape.side.is_multiple_of(shape.tile) || shape.runs == 0 {
            return Err("the side must be a multiple of the tile, and runs at least 1".into());
        }
        Ok(shape)
    }

    fn runs(&self, reads: Reads) -> bool {
        self.only.is_empty() || self.only.iter().any(|name| name == reads.name())
    }

    /// Draw `n`'s tile as vertex bytes: its corner and side in normalized device coordinates.
    fn rect(&self, n: u32) -> Vec<u8> {
        let across = self.side / self.tile;
        let tile = u64::from(n) % (across * across);
        let step = 2.0 * self.tile as f32 / self.side as f32;
        let x = -1.0 + step * (tile % across) as f32;
        let y = -1.0 + step * (tile / across) as f32;
        [x, y, step, step]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }
}

/// Run `tool` from `PATH` over `input`: its output.
fn tool(tool: &str, args: &[&str], input: &str) -> Result<Vec<u8>> {
    let mut child = Command::new(tool)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("cannot run {tool} (put it on PATH): {error}"))?;
    child
        .stdin
        .take()
        .ok_or("no stdin")?
        .write_all(input.as_bytes())?;
    let output = child.wait_with_output()?;
    if !output.status.success() {
        return Err(format!("{tool}: {}", String::from_utf8_lossy(&output.stderr)).into());
    }
    Ok(output.stdout)
}

/// The counts of `counts` by name, read from their `Debug` form.
fn counts(counts: &WorkCounts) -> BTreeMap<String, u64> {
    let text = format!("{counts:?}");
    text.split([',', '{'])
        .filter_map(|part| {
            let (name, value) = part.split_once(':')?;
            Some((name.trim().to_owned(), value.trim().parse().ok()?))
        })
        .collect()
}

/// Commit `command_buffer` and wait for it: an error where it did not complete.
fn run(command_buffer: &CommandBuffer) -> Result<()> {
    command_buffer.commit()?;
    command_buffer
        .wait_until_completed_for(WAIT)
        .map_err(|_| format!("a command buffer did not complete within {WAIT:?}"))?;
    match command_buffer.status() {
        CommandBufferStatus::Completed => Ok(()),
        status => Err(format!(
            "a command buffer ended {status:?}: {:?}",
            command_buffer.error()
        )
        .into()),
    }
}

/// What every run draws with.
struct Ours<'a> {
    device: &'a Device,
    direct: RenderPipelineState,
    holder: RenderPipelineState,
    target: Texture,
    /// [`SLOTS`] colors of 16 bytes.
    colors: Buffer,
    /// [`SLOTS`] addresses of 8 bytes, each of its color.
    holders: Buffer,
}

impl Ours<'_> {
    /// One command buffer of `shape`'s draws reading as `reads` says, committed and waited for:
    /// its GPU time a draw in microseconds.
    fn run(&self, reads: Reads, shape: &Shape) -> Result<f64> {
        let queue = self.device.new_command_queue()?;
        let command_buffer = queue.command_buffer()?;
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(self.target.clone());
        color.load_action = LoadAction::Clear;
        color.store_action = StoreAction::Store;
        color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        let mut encoder = command_buffer.render_command_encoder(&pass)?;
        encoder.set_render_pipeline_state(match reads {
            Reads::Pointer => &self.holder,
            _ => &self.direct,
        })?;
        let bytes: Vec<u8> = COLOR.iter().flat_map(|value| value.to_le_bytes()).collect();
        for n in 0..shape.draws {
            encoder.set_vertex_bytes(&shape.rect(n), 0)?;
            let slot = u64::from(n) % SLOTS;
            match reads {
                Reads::Bytes => encoder.set_fragment_bytes(&bytes, 0)?,
                Reads::Direct => encoder.set_fragment_buffer(Some(&self.colors), 16 * slot, 0)?,
                Reads::Pointer => encoder.set_fragment_buffer(Some(&self.holders), 8 * slot, 0)?,
            }
            encoder.draw_primitives(PrimitiveType::TriangleStrip, 0, 4, 1, 0)?;
        }
        encoder.end_encoding()?;
        run(&command_buffer)?;
        let gpu = command_buffer.gpu_end_time()? - command_buffer.gpu_start_time()?;
        Ok(gpu * 1e6 / f64::from(shape.draws))
    }

    /// Every texel of the target, read back, is [`TEXEL`].
    fn check(&self, reads: Reads, shape: &Shape) -> Result<()> {
        let row = shape.side * 4;
        let readback = self
            .device
            .new_buffer(row * shape.side, ResourceOptions::shared())?;
        let queue = self.device.new_command_queue()?;
        let command_buffer = queue.command_buffer()?;
        let mut blit = command_buffer.blit_command_encoder()?;
        blit.copy_from_texture_to_buffer(
            &self.target,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(shape.side, shape.side, 1),
            &readback,
            0,
            row,
            row * shape.side,
            BlitOption::None,
        )?;
        blit.end_encoding()?;
        run(&command_buffer)?;
        let contents = readback.contents().ok_or("the read-back has no contents")?;
        // SAFETY: `contents` holds the buffer's bytes, which no command buffer writes now.
        let texels =
            unsafe { std::slice::from_raw_parts(contents.as_ptr(), readback.length() as usize) };
        match texels
            .as_chunks::<4>()
            .0
            .iter()
            .position(|texel| *texel != TEXEL)
        {
            None => Ok(()),
            Some(at) => Err(format!(
                "{}: texel {at} is {:?}, not {TEXEL:?}",
                reads.name(),
                &texels[4 * at..4 * at + 4]
            )
            .into()),
        }
    }
}

/// A `Private` buffer holding `bytes`, copied in from a `Shared` one.
fn private_with(device: &Device, bytes: &[u8]) -> Result<Buffer> {
    let staging = device.new_buffer_with_bytes(bytes, ResourceOptions::shared())?;
    let buffer = device.new_buffer(bytes.len() as u64, ResourceOptions::private())?;
    let queue = device.new_command_queue()?;
    let command_buffer = queue.command_buffer()?;
    let mut blit = command_buffer.blit_command_encoder()?;
    blit.copy_from_buffer_to_buffer(&staging, 0, &buffer, 0, bytes.len() as u64)?;
    blit.end_encoding()?;
    run(&command_buffer)?;
    Ok(buffer)
}

fn main() -> Result<()> {
    let shape = Shape::from_args()?;
    let device = Device::system_default()?;
    let name = device.name().to_string();
    println!("device: {name}");
    let bitcode = tool("llvm-as", &["-o", "-", "-"], FUNCTIONS)?;
    let library = device
        .new_library_with_compile_library(metal2vulkan::library::Library::from_air(&bitcode)?);
    let vertex = library.new_function_with_name("tile_vertex")?;
    let state = |fragment: &str| -> Result<RenderPipelineState> {
        let fragment = library.new_function_with_name(fragment)?;
        let mut descriptor =
            RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
        descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        Ok(device.new_render_pipeline_state(&descriptor, &PipelineOptions::default())?)
    };
    let mut target =
        TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, shape.side, shape.side, false);
    target.usage = TextureUsage::RENDER_TARGET;
    target.resource_options = ResourceOptions::private();
    let colors: Vec<u8> = (0..SLOTS)
        .flat_map(|_| COLOR)
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let colors = private_with(&device, &colors)?;
    let holders: Vec<u8> = (0..SLOTS)
        .flat_map(|slot| (colors.gpu_address().0 + 16 * slot).to_le_bytes())
        .collect();
    let ours = Ours {
        device: &device,
        direct: state("direct_fragment")?,
        holder: state("holder_fragment")?,
        target: device.new_texture(&target)?,
        holders: private_with(&device, &holders)?,
        colors,
    };
    for reads in Reads::ALL {
        if !shape.runs(reads) {
            continue;
        }
        let mut per_draw = Vec::new();
        let mut moved: BTreeMap<String, u64> = BTreeMap::new();
        for run in 0..=shape.runs {
            let before = counts(&device.work_counts());
            let per = ours.run(reads, &shape)?;
            if run == 0 {
                continue;
            }
            per_draw.push(per);
            for (count, value) in counts(&device.work_counts()) {
                let delta = value.saturating_sub(before.get(&count).copied().unwrap_or(0));
                if delta > 0 {
                    *moved.entry(count).or_default() += delta;
                }
            }
        }
        ours.check(reads, &shape)?;
        per_draw.sort_by(f64::total_cmp);
        println!(
            "DRAW-READS {} draws {} tile {} side {} gpu_us_per_draw {:.3} least {:.3} {name}",
            reads.name(),
            shape.draws,
            shape.tile,
            shape.side,
            per_draw[per_draw.len() / 2],
            per_draw[0],
        );
        let line: Vec<String> = (moved.iter())
            .map(|(count, sum)| format!("{count}={:.2}", *sum as f64 / shape.runs as f64))
            .collect();
        println!("DRAW-READS-COUNTS {} {}", reads.name(), line.join(" "));
    }
    Ok(())
}
