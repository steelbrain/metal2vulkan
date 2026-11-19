//! Render passes: load and store actions, clear values, and what a pass does not build.
//!
//! - Metal's load actions are Vulkan's of the same name, and both plain store actions store
//!   (a pass that ends its rendering early loads what it stored); a resolving store action is
//!   `Gap::RenderPass` of `Resolve`, `Unknown` and `CustomSampleDepthStore` of `StoreAction`.
//! - A clear color is a `float` for a float, normalized or sRGB channel, rotated for a rotated
//!   format and for `A8Unorm` held in `R8_UNORM`; an integer channel's is truncated toward zero
//!   and saturated to its width, NaN 0; a format only a shader clears (extended range, bfloat)
//!   has none.
//! - Under a device cfg, on every device of the cfg's kind: a pass clears a mipmap level of a
//!   shader-read texture (which it moves to the attachment layout and back), an integer texture
//!   and a stencil texture, and stores each, the other level kept; a pass that loads and stores
//!   keeps the texels; a pass cut by `renderTargetWidth` and `renderTargetHeight` clears only its
//!   area; a pass clears one slice of a 2D array, the other slices kept. A pass alone is one
//!   submission, three barriers (its two and the segment's end) and one pass, and copies no byte
//!   of the runtime's own. Each attachment's view is made once and kept with its texture, then
//!   freed with it. What Metal refuses is refused (a depth attachment of a colour format among
//!   it), what a pass does not build is a gap (a depth and a stencil attachment of two textures,
//!   and a clear depth past `[0, 1]` or NaN, among them), and either leaves the command buffer
//!   open for another encoder.

use crate::error::{Error, Gap, RenderPassKind};
use crate::pass::{clear_color, ops};
use crate::{ClearColor, LoadAction, PixelFormat, StoreAction};
use ash::vk;
use metal2vulkan::metal::format::FormatEmulation;

/// Each load action is Vulkan's of the same name, and the two plain store actions store; the
/// others are gaps.
#[test]
fn load_actions_are_one_to_one_and_plain_store_actions_store() {
    let pairs = [
        (LoadAction::DontCare, vk::AttachmentLoadOp::DONT_CARE),
        (LoadAction::Load, vk::AttachmentLoadOp::LOAD),
        (LoadAction::Clear, vk::AttachmentLoadOp::CLEAR),
    ];
    for (load, op) in pairs {
        for (store, store_op) in [
            (StoreAction::DontCare, vk::AttachmentStoreOp::STORE),
            (StoreAction::Store, vk::AttachmentStoreOp::STORE),
        ] {
            assert!(
                ops(load, store).ok() == Some((op, store_op)),
                "{load:?} {store:?}"
            );
        }
    }
    for (store, kind) in [
        (StoreAction::MultisampleResolve, RenderPassKind::Resolve),
        (
            StoreAction::StoreAndMultisampleResolve,
            RenderPassKind::Resolve,
        ),
        (StoreAction::Unknown, RenderPassKind::StoreAction),
        (
            StoreAction::CustomSampleDepthStore,
            RenderPassKind::StoreAction,
        ),
    ] {
        match ops(LoadAction::Load, store) {
            Err(Error::Unsupported(Gap::RenderPass { kind: k })) => assert_eq!(k, kind),
            other => panic!("{store:?}: {:?}", other.map(|_| "ops")),
        }
    }
}

/// The clear values of each kind of channel.
#[test]
fn a_clear_color_is_a_float_or_a_saturated_integer() {
    let floats = |pf, emulation, c: [f64; 4]| {
        let value = clear_color(pf, emulation, ClearColor::new(c[0], c[1], c[2], c[3]))
            .unwrap_or_else(|| panic!("{pf:?}: a clear value"));
        // SAFETY: a float format's value is its float32 member.
        unsafe { value.float32 }
    };
    let direct = FormatEmulation::Direct;
    assert_eq!(
        floats(PixelFormat::RGBA8Unorm, direct, [0.2, 0.6, 0.0, 1.0]),
        [0.2, 0.6, 0.0, 1.0]
    );
    assert_eq!(
        floats(PixelFormat::RGBA16Float, direct, [1e-8, -2.5, 70000.0, 0.5]),
        [1e-8, -2.5, 70000.0, 0.5]
    );
    // `A8Unorm` held in `R8_UNORM`: Metal's alpha in red.
    let alpha = floats(
        PixelFormat::A8Unorm,
        FormatEmulation::AlphaInRed,
        [0.1, 0.2, 0.3, 0.5],
    );
    assert_eq!(alpha[0], 0.5);
    let uints = |pf, c: [f64; 4]| {
        let value = clear_color(pf, direct, ClearColor::new(c[0], c[1], c[2], c[3]))
            .unwrap_or_else(|| panic!("{pf:?}: a clear value"));
        // SAFETY: an unsigned integer format's value is its uint32 member.
        unsafe { value.uint32 }
    };
    assert_eq!(
        uints(PixelFormat::RGBA8Uint, [300.7, 2.9, -3.0, f64::NAN]),
        [255, 2, 0, 0]
    );
    assert_eq!(
        uints(PixelFormat::RGB10A2Uint, [2000.0, 1023.9, 5.5, 7.0]),
        [1023, 1023, 5, 3]
    );
    assert_eq!(
        uints(PixelFormat::RGBA32Uint, [5e9, 16777217.0, 0.0, 1.0]),
        [u32::MAX, 16777217, 0, 1]
    );
    let value = clear_color(
        PixelFormat::RGBA8Sint,
        direct,
        ClearColor::new(200.0, -200.0, -1.9, 1.9),
    )
    .expect("a clear value");
    // SAFETY: a signed integer format's value is its int32 member.
    assert_eq!(unsafe { value.int32 }, [127, -128, -1, 1]);
    let none = |pf, emulation| clear_color(pf, emulation, ClearColor::default()).is_none();
    assert!(none(PixelFormat::BGR10XR, FormatEmulation::ExtendedRange));
    assert!(none(PixelFormat::RGBA16Float, FormatEmulation::Bfloat));
}

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::blit::on_device::settled;
    use crate::tests::command::on_device::{each_device, finished, within, Arena};
    use crate::tests::compute::on_device::step;
    use crate::{
        ClearColor, CommandBuffer, Device, DispatchType, Error, Gap, IndexClass, LoadAction,
        PixelFormat, Refusal, Region, RenderPassDescriptor, RenderPassKind, StorageMode,
        StoreAction, Texture, TextureDescriptor, TextureType, TextureUsage,
    };

    fn texture(
        device: &Device,
        texture_type: TextureType,
        format: PixelFormat,
        extent: [u32; 2],
        levels: u32,
        usage: TextureUsage,
        index: Option<u32>,
    ) -> Texture {
        let mut d = TextureDescriptor::new(texture_type, format, extent[0], extent[1]);
        d.mipmap_level_count = levels;
        d.usage = usage;
        d.storage_mode = StorageMode::Shared;
        if texture_type == TextureType::Type2DArray {
            d.array_length = 3;
        }
        device
            .new_texture(&d, index)
            .unwrap_or_else(|error| panic!("{format:?}: {error}"))
    }

    /// The texels of `level` and `slice`, `unit` bytes each.
    fn read(texture: &Texture, level: u32, slice: u32, unit: usize) -> Vec<u8> {
        let d = texture.descriptor();
        let [w, h] = [d.width, d.height].map(|e| (e >> level).max(1));
        let row = w as usize * unit;
        let mut bytes = vec![0; row * h as usize];
        let region = Region {
            origin: [0; 3],
            size: [w, h, 1],
        };
        texture
            .get_bytes(
                &mut bytes,
                row as u64,
                (row * h as usize) as u64,
                region,
                level,
                slice,
            )
            .expect("get_bytes");
        bytes
    }

    /// Writes a pattern of `seed` over `level` and `slice`, `unit` bytes a texel, and returns it.
    fn write(texture: &Texture, level: u32, slice: u32, unit: usize, seed: u8) -> Vec<u8> {
        let d = texture.descriptor();
        let [w, h] = [d.width, d.height].map(|e| (e >> level).max(1));
        let row = w as usize * unit;
        let bytes: Vec<u8> = (0..row * h as usize)
            .map(|i| (i as u8).wrapping_mul(29).wrapping_add(seed))
            .collect();
        let region = Region {
            origin: [0; 3],
            size: [w, h, 1],
        };
        texture
            .replace_region(
                region,
                level,
                slice,
                &bytes,
                row as u64,
                (row * h as usize) as u64,
            )
            .expect("replace_region");
        bytes
    }

    /// `texel` repeated over `count` texels.
    fn filled(texel: &[u8], count: usize) -> Vec<u8> {
        texel.repeat(count)
    }

    /// `command_buffer` committed and complete with no error.
    fn run(name: &str, command_buffer: &CommandBuffer) {
        command_buffer.commit().expect("a commit");
        finished(name, command_buffer);
        assert_eq!(command_buffer.error(), None, "{name}");
    }

    /// Clears and stores land, loads keep, the area cuts, a slice of an array is its own; a pass
    /// alone is one submission, three barriers and one pass where an attachment moves (a
    /// shader-read texture's), and one barrier, its command buffer's end, where every attachment
    /// renders at home; each view is made once and freed with its texture.
    #[test]
    fn every_device_clears_loads_and_stores_render_attachments() {
        each_device("render_passes", |name, _, device| {
            device
                .grant_indices(IndexClass::Texture, 4096, 4)
                .expect("a grant");
            let read_render = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
            let render = TextureUsage::RENDER_TARGET;
            let colors = texture(
                &device,
                TextureType::Type2D,
                PixelFormat::RGBA8Unorm,
                [8, 4],
                2,
                read_render,
                Some(4096),
            );
            let words = texture(
                &device,
                TextureType::Type2D,
                PixelFormat::RGBA8Uint,
                [4, 2],
                1,
                render,
                None,
            );
            let stencil = texture(
                &device,
                TextureType::Type2D,
                PixelFormat::Stencil8,
                [4, 2],
                1,
                render,
                None,
            );
            let array = texture(
                &device,
                TextureType::Type2DArray,
                PixelFormat::R8Unorm,
                [2, 2],
                1,
                render,
                None,
            );
            let level0 = write(&colors, 0, 0, 4, 1);
            write(&colors, 1, 0, 4, 2);
            write(&words, 0, 0, 4, 3);
            let slices: Vec<Vec<u8>> = (0..3)
                .map(|s| write(&array, 0, s, 1, 4 + s as u8))
                .collect();
            let queue = device.new_command_queue(4).expect("a queue");
            let first = queue.command_buffer().expect("a command buffer");
            first
                .record_for_test(&[], |_, _| {})
                .expect("a command buffer's commands");
            run(name, &first);

            // Clears into a level of a shader-read texture, an integer texture and a stencil.
            let mut d = RenderPassDescriptor::default();
            let c0 = &mut d.color_attachments[0];
            c0.texture = Some(colors.clone());
            c0.level = 1;
            c0.load_action = LoadAction::Clear;
            c0.clear_color = ClearColor::new(0.2, 0.6, 0.0, 1.0);
            let c1 = &mut d.color_attachments[1];
            c1.texture = Some(words.clone());
            c1.load_action = LoadAction::Clear;
            c1.clear_color = ClearColor::new(300.7, 2.9, -3.0, f64::NAN);
            let s = &mut d.stencil_attachment;
            s.texture = Some(stencil.clone());
            s.load_action = LoadAction::Clear;
            s.store_action = StoreAction::Store;
            s.clear_stencil = 0x1a5;
            let counters = device.counters();
            settled(&device);
            let before = counters.metrics();
            let views = counters.gauges().attachment_views;
            let command_buffer = queue.command_buffer().expect("a command buffer");
            command_buffer
                .render_command_encoder(&d)
                .unwrap_or_else(|error| panic!("{name}: {error}"))
                .end_encoding();
            run(name, &command_buffer);
            let after = counters.metrics();
            assert_eq!(
                (
                    after.submissions - before.submissions,
                    after.barriers - before.barriers,
                    after.bytes_copied - before.bytes_copied,
                    after.passes - before.passes,
                ),
                (1, 3, 0, 1),
                "{name}: submissions, barriers, bytes copied, passes"
            );
            assert_eq!(
                counters.gauges().attachment_views - views,
                3,
                "{name}: a view of each attachment"
            );
            assert_eq!(read(&colors, 0, 0, 4), level0, "{name}: level 0 kept");
            assert_eq!(
                read(&colors, 1, 0, 4),
                filled(&[51, 153, 0, 255], 8),
                "{name}: level 1 cleared"
            );
            assert_eq!(
                read(&words, 0, 0, 4),
                filled(&[255, 2, 0, 0], 8),
                "{name}: integers truncated and saturated"
            );
            if stencil.unit().is_some() {
                assert_eq!(
                    read(&stencil, 0, 0, 1),
                    filled(&[0xa5], 8),
                    "{name}: the stencil's low 8 bits"
                );
            } else {
                println!("{name}: the stencil's bytes are not copied on this device");
            }

            // A load and store keeps the texels; a clear cut to 2x1 clears only those. Two
            // passes, two barriers each, and no new view.
            let mut keep = RenderPassDescriptor::default();
            keep.color_attachments[0].texture = Some(colors.clone());
            keep.color_attachments[0].level = 1;
            keep.color_attachments[0].load_action = LoadAction::Load;
            let mut cut = keep.clone();
            cut.color_attachments[0].load_action = LoadAction::Clear;
            cut.color_attachments[0].clear_color = ClearColor::new(1.0, 0.0, 0.2, 0.6);
            cut.render_target_width = 2;
            cut.render_target_height = 1;
            settled(&device);
            let before = counters.metrics();
            let command_buffer = queue.command_buffer().expect("a command buffer");
            for pass in [&keep, &cut] {
                command_buffer
                    .render_command_encoder(pass)
                    .unwrap_or_else(|error| panic!("{name}: {error}"))
                    .end_encoding();
            }
            run(name, &command_buffer);
            let after = counters.metrics();
            assert_eq!(
                (
                    after.submissions - before.submissions,
                    after.barriers - before.barriers,
                    after.passes - before.passes,
                ),
                (1, 5, 2),
                "{name}: submissions, barriers, passes"
            );
            assert_eq!(
                counters.gauges().attachment_views - views,
                3,
                "{name}: views kept"
            );
            let mut expect = filled(&[51, 153, 0, 255], 8);
            expect[..8].copy_from_slice(&[255, 0, 51, 153, 255, 0, 51, 153]);
            assert_eq!(read(&colors, 1, 0, 4), expect, "{name}: the area alone");

            // Passes whose attachments render at home. The first opens its command buffer and
            // records no first barrier; each owes its second, which the next pass's first barrier
            // or the segment's end pays: two passes are two barriers, the second loading what the
            // first cleared. A dispatch after such a pass records the owed barrier first.
            let mut whole = RenderPassDescriptor::default();
            whole.color_attachments[0].texture = Some(words.clone());
            whole.color_attachments[0].load_action = LoadAction::Clear;
            whole.color_attachments[0].clear_color = ClearColor::new(7.0, 8.0, 9.0, 10.0);
            let mut corner = whole.clone();
            corner.color_attachments[0].clear_color = ClearColor::new(1.0, 2.0, 3.0, 4.0);
            corner.render_target_width = 2;
            corner.render_target_height = 1;
            let at_home = |passes: &[&RenderPassDescriptor], dispatch: bool| {
                settled(&device);
                let before = counters.metrics();
                let command_buffer = queue.command_buffer().expect("a command buffer");
                for pass in passes {
                    command_buffer
                        .render_command_encoder(pass)
                        .unwrap_or_else(|error| panic!("{name}: {error}"))
                        .end_encoding();
                }
                let mut arena = Arena::new(&device);
                let x = arena.buffer();
                if dispatch {
                    let state = step(&device);
                    let mut encoder = (command_buffer
                        .compute_command_encoder(DispatchType::Serial))
                    .expect("an encoder");
                    encoder.set_compute_pipeline_state(&state).expect("a state");
                    encoder.set_buffer(Some(&x), 0, 0).expect("a buffer");
                    (encoder.dispatch_threadgroups([4, 1, 1], [16, 1, 1])).expect("a dispatch");
                    encoder.end_encoding();
                }
                run(name, &command_buffer);
                let after = counters.metrics();
                (
                    after.submissions - before.submissions,
                    after.barriers - before.barriers,
                    after.passes - before.passes,
                )
            };
            assert_eq!(
                at_home(&[&whole], false),
                (1, 1, 1),
                "{name}: one pass at home: submissions, barriers, passes"
            );
            assert_eq!(
                at_home(&[&whole, &corner], false),
                (1, 2, 2),
                "{name}: two passes at home: submissions, barriers, passes"
            );
            let mut expect = filled(&[7, 8, 9, 10], 8);
            expect[..8].copy_from_slice(&[1, 2, 3, 4, 1, 2, 3, 4]);
            assert_eq!(
                read(&words, 0, 0, 4),
                expect,
                "{name}: the second loads the first"
            );
            // The compute encoder is a pass of its own in `Metrics::passes`.
            assert_eq!(
                at_home(&[&whole], true),
                (1, 2, 2),
                "{name}: a pass at home, then a dispatch: submissions, barriers, passes"
            );

            // One slice of a 2D array.
            let mut one = RenderPassDescriptor::default();
            one.color_attachments[0].texture = Some(array.clone());
            one.color_attachments[0].slice = 2;
            one.color_attachments[0].load_action = LoadAction::Clear;
            one.color_attachments[0].clear_color = ClearColor::new(0.6, 0.0, 0.0, 1.0);
            let command_buffer = queue.command_buffer().expect("a command buffer");
            command_buffer
                .render_command_encoder(&one)
                .unwrap_or_else(|error| panic!("{name}: {error}"))
                .end_encoding();
            run(name, &command_buffer);
            assert_eq!(read(&array, 0, 0, 1), slices[0], "{name}: slice 0 kept");
            assert_eq!(read(&array, 0, 1, 1), slices[1], "{name}: slice 1 kept");
            assert_eq!(read(&array, 0, 2, 1), filled(&[153], 4), "{name}: slice 2");

            // Each view goes with its texture.
            drop((
                colors, words, stencil, array, d, keep, cut, one, whole, corner,
            ));
            within("the attachment views' free", || {
                counters.gauges().attachment_views == views
            });
        });
    }

    /// What Metal refuses is refused, what a pass does not build is a gap, and either leaves the
    /// command buffer open.
    #[test]
    fn every_device_refuses_what_a_pass_does_not_build() {
        each_device("render_pass_refusals", |name, _, device| {
            device
                .grant_indices(IndexClass::Texture, 4096, 4)
                .expect("a grant");
            let colors = texture(
                &device,
                TextureType::Type2D,
                PixelFormat::RGBA8Unorm,
                [4, 4],
                1,
                TextureUsage::RENDER_TARGET,
                None,
            );
            let sampled = texture(
                &device,
                TextureType::Type2D,
                PixelFormat::RGBA8Unorm,
                [4, 4],
                1,
                TextureUsage::SHADER_READ,
                Some(4096),
            );
            let stencil = texture(
                &device,
                TextureType::Type2D,
                PixelFormat::Stencil8,
                [4, 4],
                1,
                TextureUsage::RENDER_TARGET,
                None,
            );
            let depth = texture(
                &device,
                TextureType::Type2D,
                PixelFormat::Depth32Float,
                [4, 4],
                1,
                TextureUsage::RENDER_TARGET,
                None,
            );
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let color = |texture: &Texture, level| {
                let mut d = RenderPassDescriptor::default();
                d.color_attachments[0].texture = Some(texture.clone());
                d.color_attachments[0].level = level;
                d
            };
            let mut store = color(&colors, 0);
            store.color_attachments[0].store_action = StoreAction::Unknown;
            let mut depth_of_color = color(&colors, 0);
            depth_of_color.depth_attachment.texture = Some(colors.clone());
            let mut two_views = color(&colors, 0);
            two_views.depth_attachment.texture = Some(depth.clone());
            two_views.stencil_attachment.texture = Some(stencil.clone());
            let mut far_clear = color(&colors, 0);
            far_clear.depth_attachment.texture = Some(depth.clone());
            far_clear.depth_attachment.clear_depth = 2.0;
            let mut nan_clear = far_clear.clone();
            nan_clear.depth_attachment.clear_depth = f64::NAN;
            let mut stencil_of_color = color(&colors, 0);
            stencil_of_color.stencil_attachment.texture = Some(colors.clone());
            let cases = [
                (
                    color(&sampled, 0),
                    Error::Refused(Refusal::AttachmentUsage {
                        pixel_format: PixelFormat::RGBA8Unorm,
                    }),
                ),
                (
                    color(&colors, 3),
                    Error::Refused(Refusal::AttachmentPlace { level: 3, slice: 0 }),
                ),
                (
                    color(&stencil, 0),
                    Error::Refused(Refusal::AttachmentFormat {
                        pixel_format: PixelFormat::Stencil8,
                    }),
                ),
                (
                    stencil_of_color,
                    Error::Refused(Refusal::AttachmentFormat {
                        pixel_format: PixelFormat::RGBA8Unorm,
                    }),
                ),
                (
                    depth_of_color,
                    Error::Refused(Refusal::AttachmentFormat {
                        pixel_format: PixelFormat::RGBA8Unorm,
                    }),
                ),
                (
                    two_views,
                    Error::Unsupported(Gap::RenderPass {
                        kind: RenderPassKind::DepthStencilViews,
                    }),
                ),
                (
                    far_clear,
                    Error::Unsupported(Gap::RenderPass {
                        kind: RenderPassKind::ClearDepth,
                    }),
                ),
                (
                    nan_clear,
                    Error::Unsupported(Gap::RenderPass {
                        kind: RenderPassKind::ClearDepth,
                    }),
                ),
                (
                    store,
                    Error::Unsupported(Gap::RenderPass {
                        kind: RenderPassKind::StoreAction,
                    }),
                ),
                (
                    RenderPassDescriptor::default(),
                    Error::Unsupported(Gap::RenderPass {
                        kind: RenderPassKind::NoAttachment,
                    }),
                ),
            ];
            for (d, expected) in cases {
                match command_buffer.render_command_encoder(&d) {
                    Err(error) => {
                        assert_eq!(format!("{error:?}"), format!("{expected:?}"), "{name}")
                    }
                    Ok(_) => panic!("{name}: {expected:?} was not refused"),
                }
            }
            // The command buffer is open for another encoder.
            command_buffer
                .blit_command_encoder()
                .unwrap_or_else(|error| panic!("{name}: {error}"))
                .end_encoding();
            run(name, &command_buffer);
            drop((colors, sampled, stencil, depth));
        });
    }
}
