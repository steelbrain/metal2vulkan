//! A texture buffer wider than the device's `maxTexelBufferElements`: Metal makes one of up to
//! 2^28 texels on every Mac (`texture_descriptor.rs`'s `MAX_TEXTURE_BUFFER_WIDTH`), and a kernel
//! reads and writes each of its texels, while lavapipe and ANV view at most 2^27 texels through one
//! texel buffer. The runtime binds such a texture buffer through views of its parts
//! (`metal2vulkan::bind::texel_buffer_parts`).
//!
//! The kernel is `ob1_texel_buffer` (the owned fixture
//! `validation/fixtures/public/engine/ob1_texel_buffer.ll`): thread `t` of eight reads texel
//! `idx[t]` of its first texture buffer into `out[t]` and writes `0xA0000000 | t` to texel
//! `idx[8 + t]` of its second.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, Scratch};
use super::buffer_tail_device::TexelKernel;
use crate::ResourceOptions;

/// The widest texture buffer Metal makes.
const METAL_WIDEST: u64 = 1 << 28;

/// The word texel `at` holds before the kernel writes: never zero, Metal's out-of-bounds read.
fn word(at: u64) -> u32 {
    0x5000_0000 ^ at as u32
}

/// A texture buffer of `R32Uint` texels 64 past the device's `maxTexelBufferElements` (the
/// device's limit `L`, its width `L + 64`): a kernel reads texels on both sides of `L`, and its
/// last, as the buffer holds them, and Metal's zero past its last; it writes texels on both
/// sides of `L` and its last, which the buffer and a kernel then read, and drops the writes past
/// its last. Each dispatch binds the texture buffer through its parts
/// (`WorkCounts::texel_buffer_parts_bound`). A device that views 2^28 texels through one texel
/// buffer has no texture buffer past its limit to make, and runs nothing.
#[test]
fn a_texture_buffer_past_the_device_s_texel_buffer_elements_reads_and_writes_every_texel() {
    let scratch = Scratch::new("texture-buffer-width");
    let device = device(&scratch.0);
    let limit = u64::from(
        device
            .compile_device()
            .profile()
            .limits()
            .max_texel_buffer_elements,
    );
    if limit + 64 > METAL_WIDEST {
        eprintln!("maxTexelBufferElements {limit}: no texture buffer Metal makes is wider");
        return;
    }
    let width = limit + 64;
    let kernel = TexelKernel::new(&device);
    let shared = ResourceOptions::shared();
    let buffer = device.new_buffer(width * 4, shared).unwrap();
    let texels = TexelKernel::over(&buffer, width, 0);
    let other = TexelKernel::over(&device.new_buffer(256, shared).unwrap(), 64, 0);
    let words = buffer.contents().unwrap().as_ptr().cast::<u32>();
    let held: [u64; 8] = [0, 1, 2, limit - 1, limit, limit + 1, limit + 2, width - 1];
    for &at in &held {
        // SAFETY: texel `at` lies in the buffer's mapped words, which no command uses yet.
        unsafe { words.add(at as usize).write_unaligned(word(at)) };
    }
    let index = |at: u64| u32::try_from(at).unwrap();
    let before = device.work_counts();
    // Reads on both sides of the limit and the last texel, then past the last.
    let reads = [
        0,
        limit - 1,
        limit,
        limit + 1,
        width - 1,
        1,
        width,
        u64::from(u32::MAX),
    ];
    let got = kernel.run((&texels, reads.map(index)), (&other, [0; 8]));
    let want: Vec<u32> = reads
        .iter()
        .map(|&at| if at < width { word(at) } else { 0 })
        .collect();
    assert_eq!(
        got, want,
        "reads of {reads:?} in {width} texels (limit {limit})"
    );
    // Writes on both sides of the limit and the last texel; those past the last are dropped.
    let writes = [
        limit - 1,
        limit,
        limit + 2,
        width - 1,
        2,
        width,
        u64::from(u32::MAX),
        0,
    ];
    kernel.run((&other, [0; 8]), (&texels, writes.map(index)));
    let written = |at: u64| -> u32 {
        match writes.iter().position(|&w| w == at) {
            Some(t) => 0xA000_0000 | t as u32,
            None => word(at),
        }
    };
    for &at in &held {
        // SAFETY: texel `at` lies in the buffer's mapped words, which no command uses now.
        let now = unsafe { words.add(at as usize).read_unaligned() };
        assert_eq!(
            now,
            written(at),
            "texel {at} after the writes of {writes:?}"
        );
    }
    let got = kernel.run((&texels, held.map(index)), (&other, [0; 8]));
    let want: Vec<u32> = held.iter().map(|&at| written(at)).collect();
    assert_eq!(got, want, "a kernel's reads of {held:?} after the writes");
    // Each of the three dispatches bound the texture buffer's second part.
    let after = device.work_counts();
    assert!(
        after.texel_buffer_parts_bound >= before.texel_buffer_parts_bound + 3,
        "the parts bound: {} then {}",
        before.texel_buffer_parts_bound,
        after.texel_buffer_parts_bound
    );
}
