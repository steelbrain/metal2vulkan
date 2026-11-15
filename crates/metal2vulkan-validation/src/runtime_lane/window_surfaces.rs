//! A compositor's passes over a window server's surface (the consumer's x86 macOS guests: their
//! 1920x1080 `BGRA8Unorm` framebuffers are textures over guest buffers), on the device the cfg
//! selects, under the validation layer. Every byte of the scene is exact on any GPU, so each
//! stage's bytes are compared with the scene's model whole (Metal's bytes on an
//! Apple M4 equal the model at every stage).
//!
//! The surface is a 1920x1080 `BGRA8Unorm` texture (render target and shader read usage) over a
//! buffer, its rows 7680 bytes apart (Metal's `bytesPerRow`, a multiple of 128): over a no-copy
//! buffer's host pages at offset 0 ([`fixed_function_blends_over_host_pages`], the window
//! server's shape: on NVIDIA a linear image the device renders by `VK_NV_linear_color_attachment`)
//! and at offset 64 ([`fixed_function_blends_over_host_pages_at_offset_64`], where a linear image
//! cannot lie on NVIDIA, its images aligned to 128: a shadow), and over a `Shared` buffer of the
//! device's ([`fixed_function_blends_over_a_shared_buffer`]). Its pages first hold a wallpaper,
//! then eight command buffers run, each read back (the texture's bytes in the buffer) once it
//! completes:
//!
//! 1. four rounded boxes blended `One`, `OneMinusSourceAlpha` (premultiplied, alpha one half), no
//!    scissor;
//! 2. four more of the same kind of pass;
//! 3. after a host write of a block of the buffer's bytes, through a scissor at odd offsets, an
//!    alert's box blended, a title bar drawn opaque (no blending), a box blended inside;
//! 4. another such pass through the same scissor;
//! 5. a blit of a 333x111 box at (5, 7) out into a buffer (read back too) and one of another
//!    pattern into the box at the surface's bottom right corner, then one blended box over it;
//! 6. a dim of the whole surface blended, and an opaque box;
//! 7. another such pass of two other boxes;
//! 8. a pass that clears the surface, then two blended boxes.
//!
//! On NVIDIA the loaded passes render the surface in place, and the clear through copies back
//! (pass 8), no command of it reading the surface; no pass reads the surface in a shader, so
//! `fetch-through-scratch` is never announced (it was, for the copies, until stand-ins announced by
//! their cause). Where a fetch's scratch is announced, it names `zws_over`.
//! [`programmable_blends_over_host_pages`] runs the scene with the blended boxes drawn by
//! programmable blending (`zws_over`), which reads the surface through the pixel interlock where
//! the device has it; a device that refuses the fetch's pipeline, typed `Unsupported`, skips that
//! leg, printed.
//!
//! The scene's values are chosen so the exact blend of every texel lands on an integer: the
//! wallpaper's and the opaque colors' codes are multiples of 32, the blended colors' premultiplied
//! codes multiples of 16 at alpha one half, and no texel is blended more than five times between
//! writes, so every code a blend halves is even and the sum `k + v / 2` is the exact answer under
//! any of the GPUs' blend arithmetic; the alpha stays 255. Each stage prints its sha256
//! (`ZWS <leg> <stage>`) and the first texels that differ from the model, the device's linear
//! passes in place and through copies, and the notices the leg announced (`ZWS-NOTICE`).

use super::degrade_notices::{counts, on_device};
use super::host_memory::{no_copy, not_importable, Mapping, GUEST_PAGE};
use super::no_copy_textures::staged;
use super::WAIT;
use crate::air_library::AirLibrary;
use metal2vulkan::metal::{BlendFactor, PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    BlitOption, Buffer, ClearColor, CommandQueue, Degradation, Device, LoadAction, Origin,
    PipelineOptions, RenderPassDescriptor, RenderPipelineState, ResourceOptions, ScissorRect, Size,
    StoreAction, Texture, TextureDescriptor,
};
use sha2::{Digest, Sha256};

/// The surface's size, its rows' bytes, and its bytes.
const W: usize = 1920;
const H: usize = 1080;
const BPR: usize = 7680;
const BYTES: usize = BPR * H;

/// The no-copy buffers' length: whole guest pages past the texture at offset 64.
const PAGES: usize = (64 + BYTES).next_multiple_of(GUEST_PAGE);

/// What the no-copy buffers' pages hold past the texture's bytes.
const OUTSIDE: u8 = 0xa5;

/// A box of the scene: blended (alpha one half, premultiplied) or opaque, its corners in pixels
/// (half-open), its corners' radius, and its color's codes (red, green, blue).
#[derive(Clone, Copy)]
struct Box2 {
    blended: bool,
    rect: [usize; 4],
    radius: usize,
    codes: [u8; 3],
}

const fn b(rect: [usize; 4], radius: usize, codes: [u8; 3]) -> Box2 {
    Box2 {
        blended: true,
        rect,
        radius,
        codes,
    }
}

const fn o(rect: [usize; 4], radius: usize, codes: [u8; 3]) -> Box2 {
    Box2 {
        blended: false,
        rect,
        radius,
        codes,
    }
}

/// One command buffer of the scene.
struct Frame {
    boxes: &'static [Box2],
    /// The pass's scissor: x, y, width, height.
    scissor: Option<[usize; 4]>,
    /// A host write of [`CPU_BLOCK`] before it.
    cpu: bool,
    /// The blits out of [`BLIT_OUT`] and into [`BLIT_IN`] before the pass, in its command buffer.
    blits: bool,
    /// The pass clears to these codes (red, green, blue; alpha 255) rather than loading.
    clear: Option<[u8; 3]>,
}

/// The pass's scissor of passes 3 and 4.
const SCISSOR: Option<[usize; 4]> = Some([37, 101, 1001, 555]);

const FRAMES: [Frame; 8] = [
    Frame {
        boxes: &[
            b([101, 57, 733, 431], 12, [16, 32, 48]),
            b([500, 300, 1301, 777], 24, [112, 16, 48]),
            b([1700, 1000, 1920, 1080], 9, [0, 0, 0]),
            b([0, 0, 40, 23], 5, [64, 64, 64]),
        ],
        scissor: None,
        cpu: false,
        blits: false,
        clear: None,
    },
    Frame {
        boxes: &[
            b([1203, 77, 1801, 478], 16, [112, 112, 112]),
            b([1243, 150, 1761, 458], 10, [16, 32, 48]),
            b([3, 700, 400, 1077], 40, [48, 96, 16]),
            b([900, 850, 1300, 1000], 30, [96, 0, 96]),
        ],
        scissor: None,
        cpu: false,
        blits: false,
        clear: None,
    },
    Frame {
        boxes: &[
            b([661, 340, 1259, 741], 16, [112, 112, 112]),
            o([700, 400, 1219, 437], 0, [32, 32, 32]),
            b([720, 450, 1200, 700], 8, [16, 96, 48]),
        ],
        scissor: SCISSOR,
        cpu: true,
        blits: false,
        clear: None,
    },
    Frame {
        boxes: &[
            b([40, 120, 600, 640], 20, [48, 48, 96]),
            o([100, 600, 500, 640], 4, [224, 192, 160]),
            b([300, 200, 900, 500], 12, [32, 0, 32]),
        ],
        scissor: SCISSOR,
        cpu: false,
        blits: false,
        clear: None,
    },
    Frame {
        boxes: &[b([1550, 950, 1919, 1079], 11, [80, 48, 112])],
        scissor: None,
        cpu: false,
        blits: true,
        clear: None,
    },
    Frame {
        boxes: &[
            b([0, 0, 1920, 1080], 0, [0, 0, 0]),
            o([1800, 10, 1900, 60], 6, [96, 160, 224]),
        ],
        scissor: None,
        cpu: false,
        blits: false,
        clear: None,
    },
    Frame {
        boxes: &[
            b([1300, 800, 1700, 1000], 25, [48, 16, 16]),
            o([20, 1040, 300, 1075], 3, [128, 128, 128]),
        ],
        scissor: None,
        cpu: false,
        blits: false,
        clear: None,
    },
    Frame {
        boxes: &[
            b([400, 300, 1500, 800], 24, [64, 32, 16]),
            b([600, 400, 1300, 700], 16, [16, 16, 16]),
        ],
        scissor: None,
        cpu: false,
        blits: false,
        clear: Some([32, 64, 96]),
    },
];

/// The box blitted out (x, y, width, height), and the corner the other pattern is blitted into.
const BLIT_OUT: [usize; 4] = [5, 7, 333, 111];
const BLIT_IN: [usize; 2] = [1587, 969];

/// The block the host writes before pass 3: x, y, width, height.
const CPU_BLOCK: [usize; 4] = [211, 333, 300, 200];

/// A texel's bytes in memory, `BGRA8Unorm`'s order.
fn texel(b: usize, g: usize, r: usize) -> [u8; 4] {
    [
        (b & 7) as u8 * 32,
        (g & 7) as u8 * 32,
        (r & 7) as u8 * 32,
        255,
    ]
}

/// The wallpaper's texel at `(x, y)`.
fn wallpaper(x: usize, y: usize) -> [u8; 4] {
    texel(x * 3 + y * 5, (x >> 2) + y, x ^ (y >> 1))
}

/// The host block's texel at `(x, y)`.
fn host_block(x: usize, y: usize) -> [u8; 4] {
    texel(x + 2 * y, 3 * x + y, x * y)
}

/// The blitted pattern's texel at `(i, j)` of its box.
fn blit_block(i: usize, j: usize) -> [u8; 4] {
    texel(i * 5 + j, i + j * 3, (i >> 1) + (j >> 1))
}

/// The bytes of `[x0, x0 + w) x [y0, y0 + h)` of `f`, rows `w * 4` bytes apart.
fn block(x0: usize, y0: usize, w: usize, h: usize, f: fn(usize, usize) -> [u8; 4]) -> Vec<u8> {
    (y0..y0 + h)
        .flat_map(|y| (x0..x0 + w).flat_map(move |x| f(x, y)))
        .collect()
}

/// Whether the pixel center of `(x, y)` lies inside `bx`'s rounded corners and `scissor`.
fn covers(bx: &Box2, scissor: Option<[usize; 4]>, x: usize, y: usize) -> bool {
    let [x0, y0, x1, y1] = bx.rect;
    let [sx, sy, sw, sh] = scissor.unwrap_or([0, 0, W, H]);
    if !(x0..x1).contains(&x) || !(y0..y1).contains(&y) {
        return false;
    }
    if !(sx..sx + sw).contains(&x) || !(sy..sy + sh).contains(&y) {
        return false;
    }
    // Half-integers squared: exact in `f64`.
    let r = bx.radius as f64;
    let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
    let qx = px - px.clamp(x0 as f64 + r, x1 as f64 - r);
    let qy = py - py.clamp(y0 as f64 + r, y1 as f64 - r);
    qx * qx + qy * qy <= r * r
}

/// The scene's model: the texture's bytes, and after each stage its name and the bytes Metal
/// leaves (the texture's, or the blit's box out).
fn model() -> Vec<(String, Vec<u8>)> {
    let mut t = block(0, 0, W, H, wallpaper);
    let mut stages = Vec::new();
    for (n, frame) in FRAMES.iter().enumerate() {
        if frame.cpu {
            let [x0, y0, w, h] = CPU_BLOCK;
            for y in y0..y0 + h {
                for x in x0..x0 + w {
                    t[y * BPR + x * 4..][..4].copy_from_slice(&host_block(x, y));
                }
            }
        }
        if frame.blits {
            let [x0, y0, w, h] = BLIT_OUT;
            let out = (y0..y0 + h)
                .flat_map(|y| t[y * BPR + x0 * 4..][..w * 4].to_vec())
                .collect();
            stages.push(("blitout".to_owned(), out));
            let [bx, by] = BLIT_IN;
            for j in 0..h {
                for i in 0..w {
                    t[(by + j) * BPR + (bx + i) * 4..][..4].copy_from_slice(&blit_block(i, j));
                }
            }
        }
        if let Some([r, g, b]) = frame.clear {
            for at in t.as_chunks_mut::<4>().0.iter_mut() {
                at.copy_from_slice(&[b, g, r, 255]);
            }
        }
        for bx in frame.boxes {
            let [r, g, b] = bx.codes;
            let k = [b, g, r];
            let [x0, y0, x1, y1] = bx.rect;
            let [sx, sy, sw, sh] = frame.scissor.unwrap_or([0, 0, W, H]);
            for y in y0.max(sy)..y1.min(sy + sh) {
                for x in x0.max(sx)..x1.min(sx + sw) {
                    if !covers(bx, frame.scissor, x, y) {
                        continue;
                    }
                    let at = &mut t[y * BPR + x * 4..][..4];
                    for c in 0..3 {
                        at[c] = match bx.blended {
                            true => {
                                assert!(at[c].is_multiple_of(2), "the scene blends an odd code");
                                k[c] + at[c] / 2
                            }
                            false => k[c],
                        };
                    }
                    assert_eq!(at[3], 255, "the scene's alpha stays 255");
                }
            }
        }
        stages.push((format!("frame{}", n + 1), t.clone()));
    }
    stages
}

/// The hex of `bytes`' sha256.
fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// `got` against the model's `want` (rows `row` bytes apart), printed: its sha256, the texels that
/// differ (the first eight, their count and the largest code difference). Whether they are equal.
fn compare(leg: &str, stage: &str, got: &[u8], want: &[u8], row: usize, outside: usize) -> bool {
    let differ: Vec<usize> = (0..got.len() / 4)
        .filter(|&i| got[i * 4..][..4] != want[i * 4..][..4])
        .collect();
    let largest = differ
        .iter()
        .flat_map(|&i| (0..4).map(move |c| got[i * 4 + c].abs_diff(want[i * 4 + c])))
        .max()
        .unwrap_or(0);
    let first: Vec<String> = differ
        .iter()
        .take(8)
        .map(|&i| {
            let (x, y) = ((i * 4 % row) / 4, i * 4 / row);
            format!(
                "({x},{y}) {:?} not {:?}",
                &got[i * 4..][..4],
                &want[i * 4..][..4]
            )
        })
        .collect();
    println!(
        "ZWS {leg} {stage}: sha256 {}; {} texels differ from the model (largest {largest}){}{}; \
         outside the texture {outside} bytes changed",
        sha256(got),
        differ.len(),
        if first.is_empty() { "" } else { ": " },
        first.join(", ")
    );
    differ.is_empty() && outside == 0
}

/// The engine fixture `zws_compositor`'s render pipeline: its box, `fragment`, blended
/// `One`, `OneMinusSourceAlpha` where `blends`; the device's refusal, typed (a fetch it cannot
/// make: ANV on an Iris Xe reads a `BGRA8Unorm` storage image only without a format, which it
/// lacks, `shaderStorageImageReadWithoutFormat`).
fn pipeline(
    device: &Device,
    fragment: &str,
    blends: bool,
) -> Result<RenderPipelineState, metal2vulkan_runtime::Error> {
    let ll = metal2vulkan_devtools::engine_dir().join("zws_compositor.ll");
    let library = device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap();
    let function = |name: &str| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut d = RenderPipelineDescriptor::new(function("zws_box"));
    d.fragment = Some(function(fragment));
    let mut color = ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm);
    if blends {
        color.blending_enabled = true;
        color.source_rgb_blend_factor = BlendFactor::One;
        color.source_alpha_blend_factor = BlendFactor::One;
        color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
        color.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
    }
    d.color_attachments[0] = Some(color);
    device.new_render_pipeline_state(&d, &PipelineOptions::default())
}

/// `zws_box`'s and the fragment functions' `ZwsBox` of `bx`: its corners, its premultiplied
/// color, what of the attachment's color stays (one half for a blended box), and its radius and
/// the target's size.
fn uniform(bx: &Box2) -> Vec<u8> {
    let [x0, y0, x1, y1] = bx.rect.map(|v| v as f32);
    let [r, g, b] = bx.codes.map(|c| f32::from(c) / 255.0);
    let (alpha, keep) = match bx.blended {
        true => (0.5, 0.5),
        false => (1.0, 0.0),
    };
    [
        x0,
        y0,
        x1,
        y1,
        r,
        g,
        b,
        alpha,
        keep,
        keep,
        keep,
        keep,
        bx.radius as f32,
        W as f32,
        H as f32,
        0.0,
    ]
    .iter()
    .flat_map(|v| v.to_le_bytes())
    .collect()
}

/// Where the surface's bytes lie.
enum Under {
    /// A no-copy buffer over host pages, the texture at this offset.
    Pages(Mapping, usize),
    /// A `Shared` buffer of the device's.
    Shared,
}

/// The bytes at `range` of a `Shared` buffer no command uses now.
fn shared_bytes(buffer: &Buffer, range: std::ops::Range<usize>) -> Vec<u8> {
    let contents = buffer.contents().unwrap();
    // SAFETY: the range lies in the buffer's bytes, which no command uses now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr().add(range.start), range.len()) }.to_vec()
}

/// Writes `bytes` at `at` of a `Shared` buffer no command uses now.
fn write_shared(buffer: &Buffer, at: usize, bytes: &[u8]) {
    let contents = buffer.contents().unwrap();
    // SAFETY: the range lies in the buffer's bytes, which no command uses now.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr().add(at), bytes.len())
    };
}

/// The scene's leg `leg` over `under` (`at`: 64 for the offset leg), the blended boxes drawn by
/// programmable blending where `fetches`: every stage equals the model, and nothing past the
/// texture changes; where the leg draws no fetch, `fetch-through-scratch` is not announced.
fn leg(leg: &str, at: usize, shared: bool, fetches: bool) {
    on_device(|device, messages| {
        messages.take_notices();
        let under = match shared {
            true => Under::Shared,
            false => {
                let mapping = Mapping::anonymous(PAGES);
                mapping.fill(OUTSIDE);
                Under::Pages(mapping, at)
            }
        };
        let wall = block(0, 0, W, H, wallpaper);
        let buffer = match &under {
            Under::Shared => {
                let buffer = device
                    .new_buffer(BYTES as u64, ResourceOptions::shared())
                    .unwrap();
                write_shared(&buffer, 0, &wall);
                buffer
            }
            Under::Pages(mapping, at) => {
                mapping.write(*at, &wall);
                // SAFETY: the mapping outlives every buffer over it (dropped at the leg's end).
                let made = unsafe {
                    no_copy(
                        device,
                        mapping.pointer(),
                        PAGES,
                        ResourceOptions::default(),
                        None,
                    )
                };
                match made {
                    Ok(buffer) => buffer,
                    Err(error) if not_importable(&error) => {
                        println!("ZWS {leg}: the device imports no host memory: {error:?}");
                        return;
                    }
                    Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
                }
            }
        };
        let at = match &under {
            Under::Pages(_, at) => *at,
            Under::Shared => 0,
        };
        let mut d =
            TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, W as u64, H as u64, false);
        d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        d.resource_options = buffer.resource_options();
        let texture = buffer.new_texture(&d, at as u64, BPR as u64).unwrap();
        // SAFETY: the handle is only compared, while the texture lives.
        let aliased = unsafe { texture.vk_image() }.is_some();
        println!(
            "ZWS {leg} on {}: the texture is {}",
            device.name(),
            match aliased {
                true => "its own linear image",
                false => "a shadow",
            }
        );
        let queue = device.new_command_queue().unwrap();
        let opaque = pipeline(device, "zws_plain", false).unwrap();
        let blended = match fetches {
            true => match pipeline(device, "zws_over", false) {
                Ok(fetch) => fetch,
                Err(error @ metal2vulkan_runtime::Error::Unsupported(_)) => {
                    println!("ZWS {leg}: the device refuses the fetch: {error:?}");
                    return;
                }
                Err(error) => panic!("the fetch's pipeline: {error:?}"),
            },
            false => pipeline(device, "zws_plain", true).unwrap(),
        };
        let model = model();
        let mut stages = model.iter();
        let mut equal = true;
        let [_, _, bw, bh] = BLIT_OUT;
        let readback = device
            .new_buffer((bw * bh * 4) as u64, ResourceOptions::shared())
            .unwrap();
        let pattern = staged(device, &block(0, 0, bw, bh, blit_block));
        let read = |range: std::ops::Range<usize>| match &under {
            Under::Pages(mapping, _) => mapping.read(range),
            Under::Shared => shared_bytes(&buffer, range),
        };
        for frame in &FRAMES {
            if frame.cpu {
                let [x0, y0, w, h] = CPU_BLOCK;
                for y in y0..y0 + h {
                    let row = block(x0, y, w, 1, host_block);
                    let offset = at + y * BPR + x0 * 4;
                    match &under {
                        Under::Pages(mapping, _) => mapping.write(offset, &row),
                        Under::Shared => write_shared(&buffer, offset, &row),
                    }
                }
            }
            let before = device.work_counts();
            run(
                &queue,
                &texture,
                frame,
                (&blended, &opaque),
                (&readback, &pattern),
            );
            let after = device.work_counts();
            let ways = (
                after.linear_passes_in_place - before.linear_passes_in_place,
                after.linear_passes_through_copies - before.linear_passes_through_copies,
            );
            if frame.blits {
                let (stage, want) = stages.next().unwrap();
                let got = shared_bytes(&readback, 0..bw * bh * 4);
                equal &= compare(leg, stage, &got, want, bw * 4, 0);
            }
            let (stage, want) = stages.next().unwrap();
            let got = read(at..at + BYTES);
            let outside = match &under {
                Under::Pages(mapping, _) => {
                    let all = mapping.read(0..PAGES);
                    (0..PAGES)
                        .filter(|&i| !(at..at + BYTES).contains(&i) && all[i] != OUTSIDE)
                        .count()
                }
                Under::Shared => 0,
            };
            println!(
                "ZWS {leg} {stage}: linear passes in place {}, through copies {}",
                ways.0, ways.1
            );
            equal &= compare(leg, stage, &got, want, BPR, outside);
        }
        let notices = messages.take_notices();
        for notice in &notices {
            println!("ZWS-NOTICE {leg}: {}", notice.text);
        }
        let counts = counts(&notices);
        drop((texture, buffer));
        drop(under);
        assert!(equal, "a stage's bytes differ from the model");
        if !fetches {
            assert_eq!(
                counts.get(&Degradation::FetchThroughScratch),
                None,
                "a leg without a fetch announced fetch-through-scratch"
            );
        }
        // A fetch's scratch names the draw that fetched it: `zws_over`, unblended.
        let scratch = notices
            .iter()
            .filter_map(|message| message.degradation.as_ref())
            .find(|notice| notice.degradation == Degradation::FetchThroughScratch);
        if let Some(notice) = scratch {
            assert!(
                notice
                    .reason
                    .starts_with("a draw of fragment function `zws_over` (color 0 unblended"),
                "the scratch's notice: {}",
                notice.reason
            );
        }
    });
}

/// One command buffer of `frame` over `texture`: its blits (into `readback` and from `pattern`),
/// then its render pass of its boxes, blended ones by `pipelines.0` and opaque ones by
/// `pipelines.1`; it completes.
fn run(
    queue: &CommandQueue,
    texture: &Texture,
    frame: &Frame,
    pipelines: (&RenderPipelineState, &RenderPipelineState),
    (readback, pattern): (&Buffer, &Buffer),
) {
    let command_buffer = queue.command_buffer().unwrap();
    if frame.blits {
        let [x0, y0, w, h] = BLIT_OUT.map(|v| v as u64);
        let [bx, by] = BLIT_IN.map(|v| v as u64);
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_texture_to_buffer(
            texture,
            0,
            0,
            Origin::new(x0, y0, 0),
            Size::new(w, h, 1),
            readback,
            0,
            w * 4,
            w * h * 4,
            BlitOption::None,
        )
        .unwrap();
        blit.copy_from_buffer_to_texture(
            pattern,
            0,
            w * 4,
            w * h * 4,
            Size::new(w, h, 1),
            texture,
            0,
            0,
            Origin::new(bx, by, 0),
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
    }
    let mut descriptor = RenderPassDescriptor::new();
    let attachment = &mut descriptor.color_attachments[0];
    attachment.texture = Some(texture.clone());
    attachment.store_action = StoreAction::Store;
    match frame.clear {
        Some([r, g, b]) => {
            attachment.load_action = LoadAction::Clear;
            let code = |c: u8| f64::from(c) / 255.0;
            attachment.clear_color = ClearColor::new(code(r), code(g), code(b), 1.0);
        }
        None => attachment.load_action = LoadAction::Load,
    }
    let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
    if let Some([x, y, w, h]) = frame.scissor {
        let [x, y, w, h] = [x, y, w, h].map(|v| v as u64);
        encoder
            .set_scissor_rect(ScissorRect::new(x, y, w, h))
            .unwrap();
    }
    for bx in frame.boxes {
        let pipeline = match bx.blended {
            true => pipelines.0,
            false => pipelines.1,
        };
        encoder.set_render_pipeline_state(pipeline).unwrap();
        let bytes = uniform(bx);
        encoder.set_vertex_bytes(&bytes, 0).unwrap();
        encoder.set_fragment_bytes(&bytes, 0).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
            .unwrap();
    }
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// The window server's shape: no-copy host pages at offset 0, blended by the fixed-function blend.
#[test]
fn fixed_function_blends_over_host_pages() {
    leg("pages0", 0, false, false);
}

/// The same over a `Shared` buffer of the device's.
#[test]
fn fixed_function_blends_over_a_shared_buffer() {
    leg("shared0", 0, true, false);
}

/// The same over host pages at offset 64: no linear image of the device's lies there where its
/// images are aligned to more, so the texture is a shadow.
#[test]
fn fixed_function_blends_over_host_pages_at_offset_64() {
    leg("pages64", 64, false, false);
}

/// The window server's shape with the blended boxes drawn by programmable blending.
#[test]
fn programmable_blends_over_host_pages() {
    leg("pages0-fetch", 0, false, true);
}
