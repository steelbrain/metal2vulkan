//! Where the build of one of the runtime's own kernels falls, counted.
//!
//! A format the runtime emulates needs kernels of its own, and a kernel is built by the Metal
//! call that first needs it and then kept for the device (`owned_kernel.rs`). So the cost of the
//! build lands inside one call and inside no other, and nothing about that call says it: Metal
//! builds no such kernel, and a program that times its calls sees one of them cost far more than
//! Metal's for reasons the call itself cannot explain.
//!
//! `WorkCounts::owned_kernels_made` and `owned_kernel_spirv_bytes` name it. Here a two-plane YCbCr
//! texture over a surface's planes, whose making needs the conversion's kernels
//! (`ycbcr.rs`, `texture/ycbcr_image.rs`), is made twice: the first making builds them and the
//! second builds none. The same counter is what an encoding increments where the encoding is the
//! call that first needs a kernel, so the two sites can be told apart in one recording.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, Scratch};
use crate::{
    Device, Error, IoSurfacePlane, IoSurfaceYCbCr, ResourceOptions, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};

/// The surface's size, as a stream of such a texture holds it.
const W: u64 = 3840;
const H: u64 = 2160;

/// The luma plane's offset and rows, and the chroma plane's: 10-bit codes, 2 bytes each, the
/// chroma plane after the luma plane and half its height.
const LUMA: (u64, u64) = (0, 2 * W);
const CHROMA: (u64, u64) = (2 * W * H, 2 * W);

/// The surface's bytes.
const LENGTH: usize = (CHROMA.0 + CHROMA.1 * H / 2) as usize;

/// The kernels built and the SPIR-V bytes they compiled while `run` ran.
fn built(device: &Device, run: impl FnOnce()) -> [u64; 2] {
    let before = device.work_counts();
    run();
    let after = device.work_counts();
    [
        after.owned_kernels_made - before.owned_kernels_made,
        after.owned_kernel_spirv_bytes - before.owned_kernel_spirv_bytes,
    ]
}

/// A two-plane YCbCr texture over the `x420` surface's planes, as the suite's cases make one.
fn plane_texture(buffer: &crate::Buffer) -> Result<Texture, Error> {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::YCBCR104202P, W, H, false);
    descriptor.resource_options = ResourceOptions::shared();
    descriptor.usage = TextureUsage::SHADER_READ;
    let facts = IoSurfaceYCbCr::new(u32::from_be_bytes(*b"x420"), 0, [W, H], CHROMA.0, CHROMA.1);
    let plane = IoSurfacePlane::new(LUMA.0, LUMA.1, true).with_ycbcr(facts);
    buffer.new_texture_with_iosurface_plane(&descriptor, plane)
}

/// The first making of such a texture builds the conversion's kernels and the second builds
/// none, so the count says which call paid for them.
#[test]
fn an_emulated_formats_kernels_are_built_by_the_call_that_first_needs_them() {
    let scratch = Scratch::new("owned-kernel-builds");
    let device = device(&scratch.0);
    let bytes = vec![0x40; LENGTH];
    let Ok(buffer) = device.new_buffer_with_bytes(&bytes, ResourceOptions::shared()) else {
        eprintln!("{}: no buffer of the surface's bytes", device.name());
        return;
    };
    let mut first = None;
    let counted = built(&device, || first = Some(plane_texture(&buffer)));
    match first {
        Some(Ok(_texture)) => {}
        made => {
            eprintln!("{}: no `x420` plane texture: {made:?}", device.name());
            return;
        }
    }
    let [kernels, words] = counted;
    let again = built(&device, || drop(plane_texture(&buffer)));
    eprintln!(
        "{}: making the first such texture built {kernels} kernels of {words} SPIR-V bytes, \
         the second {} of {}",
        device.name(),
        again[0],
        again[1]
    );
    assert!(
        kernels > 0 && words > 0,
        "the making of a texture of an emulated format built no kernel of the runtime's own, \
         so the counter no longer names where the build falls"
    );
    assert_eq!(
        again,
        [0, 0],
        "a second making built kernels again, so they are not kept for the device"
    );
}
