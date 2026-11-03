//! Two shadowed textures over the same bytes (twins), each carried across its command buffer's
//! encoders (`shadow::ShadowCarry`): a pass over one writes its changed texels back into the
//! bytes as its encoder ends (merged back, or copied back), so the other's carried image no
//! longer holds them, and the other copies them in again at its next use.
//!
//! One command buffer alternates passes over the two `RGBA8Unorm` textures (each loading and
//! storing its texture beside a cleared `Depth32Float` depth attachment, blending a rectangle that
//! overlaps the one before) with blits reading one texture's texels, over a buffer whose rows hold
//! bytes between them: a no-copy buffer's host pages, merged back, and a `Private` buffer, copied
//! back, a shadow on every device either way. The same passes and reads over one private texture
//! of the same first texels are the reference: each read, and the texels the command buffer
//! leaves, are the same byte for byte, and the bytes between the rows keep theirs.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are the owned fixture
//! `validation/fixtures/public/engine/tob_renders.ll` (`llvm-as`).

use super::blit_holds::{device, Scratch, WAIT};
use super::encoder_holds::bytes_of;
use super::merge_scratch::{
    blending, blit, bytes, drawn, first_bytes, read, staged, target, texels, Host, Over, H, TEXEL,
};
use crate::{
    BlitOption, Buffer, CommandBuffer, CommandQueue, LoadAction, Origin, RenderPassDescriptor,
    RenderPipelineState, ResourceOptions, Size, StoreAction, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};

/// The textures' width, and their rows' pitch in the buffer: 224 bytes between each two rows.
const W: u64 = 200;
const ROW: u64 = 1024;

/// One of the two textures.
#[derive(Clone, Copy, Debug)]
enum Twin {
    A,
    B,
}

/// One step of the command buffer.
#[derive(Clone, Copy, Debug)]
enum Step {
    /// A pass over the texture blending rectangle `i` ([`drawn`]), which overlaps `i - 1`'s.
    Pass(Twin, usize),
    /// A blit copying the texture's texels into a buffer, read after the command buffer.
    Read(Twin),
}

use Step::{Pass, Read};
use Twin::{A, B};

/// Each pass over one texture after a pass over the other, each read of a texture after a pass
/// over the other: the texture first used, `B`, is carried past `A`'s first pass.
const STEPS: [Step; 7] = [
    Pass(B, 0),
    Pass(A, 1),
    Read(B),
    Pass(B, 2),
    Read(A),
    Pass(A, 3),
    Read(B),
];

/// What the steps run over: the two textures (one private texture twice, for the reference), the
/// blending state, and the depth texture its passes clear.
struct Scene<'a> {
    a: &'a Texture,
    b: &'a Texture,
    state: &'a RenderPipelineState,
    depth: &'a Texture,
}

impl Scene<'_> {
    fn texture(&self, twin: Twin) -> &Texture {
        match twin {
            A => self.a,
            B => self.b,
        }
    }
}

/// Encode a pass over `texture`, loading and storing it beside the scene's depth texture, which
/// it clears, that blends rectangle `i`.
fn pass(command_buffer: &CommandBuffer, scene: &Scene<'_>, texture: &Texture, i: usize) {
    let mut descriptor = RenderPassDescriptor::new();
    let color = &mut descriptor.color_attachments[0];
    color.texture = Some(texture.clone());
    color.load_action = LoadAction::Load;
    color.store_action = StoreAction::Store;
    descriptor.depth_attachment.texture = Some(scene.depth.clone());
    descriptor.depth_attachment.load_action = LoadAction::Clear;
    descriptor.depth_attachment.store_action = StoreAction::DontCare;
    let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
    encoder.set_render_pipeline_state(scene.state).unwrap();
    let (corners, rgba) = drawn(i);
    (encoder.set_vertex_bytes(&bytes_of(&corners, f32::to_le_bytes), 0)).unwrap();
    (encoder.set_fragment_bytes(&bytes_of(&rgba, f32::to_le_bytes), 0)).unwrap();
    (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
    encoder.end_encoding().unwrap();
}

/// Encode a blit copying `texture`'s texels into `out`, packed.
fn read_texels(command_buffer: &CommandBuffer, texture: &Texture, out: &Buffer) {
    let packed = W * TEXEL;
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    (encoder.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(W, H, 1),
        out,
        0,
        packed,
        packed * H,
        BlitOption::None,
    ))
    .unwrap();
    encoder.end_encoding().unwrap();
}

/// [`STEPS`] over `scene` in one command buffer, committed and completed: what each [`Read`]
/// copied, packed.
fn steps(queue: &CommandQueue, scene: &Scene<'_>) -> Vec<Vec<u8>> {
    let device = queue.device();
    let length = W * TEXEL * H;
    let command_buffer = queue.command_buffer().unwrap();
    let mut reads = Vec::new();
    for step in STEPS {
        match step {
            Pass(twin, i) => pass(&command_buffer, scene, scene.texture(twin), i),
            Read(twin) => {
                let out = device
                    .new_buffer(length, ResourceOptions::shared())
                    .unwrap();
                read_texels(&command_buffer, scene.texture(twin), &out);
                reads.push(out);
            }
        }
    }
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    reads.iter().map(|out| read(out, length)).collect()
}

/// A `RGBA8Unorm` texture of [`W`] x [`H`] over `buffer`'s first bytes, rows [`ROW`] apart.
fn twin_over(buffer: &Buffer) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, W, H, false);
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    descriptor.resource_options = buffer.resource_options();
    buffer.new_texture(&descriptor, 0, ROW).unwrap()
}

/// A `Private` texture of `format`, [`W`] x [`H`], of `usage`.
fn private(queue: &CommandQueue, format: PixelFormat, usage: TextureUsage) -> Texture {
    let mut d = TextureDescriptor::texture_2d(format, W, H, false);
    d.usage = usage;
    d.resource_options = ResourceOptions::private();
    queue.device().new_texture(&d).unwrap()
}

/// The number of bytes at which `got` and `expected` differ.
fn differ(got: &[u8], expected: &[u8]) -> usize {
    assert_eq!(got.len(), expected.len());
    (0..got.len()).filter(|&i| got[i] != expected[i]).count()
}

/// [`STEPS`] over two textures over the same bytes of `over`'s buffer, against one private
/// texture's: every read and the texels left the same, byte for byte, and the bytes between the
/// rows kept.
fn twins(over: Over) {
    let cache = Scratch::new(&format!("shadow-twins-{over:?}"));
    let host = Host::new(ROW * H);
    let device = device(&cache.0);
    let queue = device.new_command_queue().unwrap();
    let state = blending(&device);
    let depth = private(
        &queue,
        PixelFormat::Depth32Float,
        TextureUsage::RENDER_TARGET,
    );
    let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    let reference = private(&queue, PixelFormat::RGBA8Unorm, usage);
    let packed = W * TEXEL;
    let source = staged(&device, &first_bytes(W, packed));
    blit(&queue, |b| {
        (b.copy_from_buffer_to_texture(
            &source,
            0,
            packed,
            packed * H,
            Size::new(W, H, 1),
            &reference,
            0,
            0,
            Origin::new(0, 0, 0),
            BlitOption::None,
        ))
        .unwrap()
    });
    let on_reference = Scene {
        a: &reference,
        b: &reference,
        state: &state,
        depth: &depth,
    };
    let mut expected = steps(&queue, &on_reference);
    // The texels it left, read alone.
    let left = device
        .new_buffer(packed * H, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    read_texels(&command_buffer, &reference, &left);
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    expected.push(read(&left, packed * H));

    let Some((buffer, a)) = target(&queue, over, (W, ROW), &host) else {
        return;
    };
    let b = twin_over(&buffer);
    let merged = over != Over::Private;
    for texture in [&a, &b] {
        assert!(
            texture.shadow().is_some_and(|c| c.merges_back() == merged),
            "{over:?}: each texture is a shadow, merged back {merged}"
        );
    }
    let on_twins = Scene {
        a: &a,
        b: &b,
        state: &state,
        depth: &depth,
    };
    let before = device.work_counts();
    let mut got = steps(&queue, &on_twins);
    let after = device.work_counts();
    let pages = bytes(&queue, over, &buffer, ROW * H);
    got.push(texels(&pages, W, ROW));
    println!(
        "SHADOW-TWINS {over:?} {W}x{H}, rows of {ROW}: shadow copies {}, texels copied in {}, \
         first copies in carried {} texels",
        after.shadow_copies - before.shadow_copies,
        after.shadow_texels_copied_in - before.shadow_texels_copied_in,
        after.shadow_texels_first_copy_in_carried - before.shadow_texels_first_copy_in_carried,
    );
    let reads = STEPS.iter().filter(|step| matches!(step, Read(_)));
    let names = reads
        .map(|step| format!("{step:?}"))
        .chain(["the bytes left".into()]);
    for ((name, got), expected) in names.zip(&got).zip(&expected) {
        assert_eq!(
            differ(got, expected),
            0,
            "{over:?}, {name}: the texels differ from the private texture's"
        );
    }
    let first = first_bytes(W, ROW);
    let between = (0..pages.len())
        .filter(|&i| (i as u64 % ROW) >= W * TEXEL && pages[i] != first[i])
        .count();
    assert_eq!(between, 0, "{over:?}: bytes between the rows changed");
}

/// Twins over a no-copy buffer's host pages, each pass's writes merged back.
#[test]
fn a_merge_leaves_its_twins_carried_image_to_copy_in_again() {
    twins(Over::HostPages);
}

/// Twins over a `Private` buffer, each pass's writes copied back.
#[test]
fn a_copy_back_leaves_its_twins_carried_image_to_copy_in_again() {
    twins(Over::Private);
}
