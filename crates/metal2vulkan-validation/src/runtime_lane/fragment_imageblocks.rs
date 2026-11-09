//! Fragment imageblocks through the runtime on the device the cfg selects, under the validation
//! layer. The committed public fragment cases with an imageblock run in `renders`' census, between
//! the Metal oracle's generated initializer and resolver (`candidate::runtime_case`); this module
//! runs the pass's own rules over the owned fixtures `fragment_imageblock_views` (its `Cell`
//! functions, whose master is a `half4` and a `half` in 16 bytes aligned to 8, and
//! `imageblock_transition`, of another master) and `k10ts_tile_grid` (a tile function over an
//! `R32Float` and two `R32Uint` attachments), in passes of those three attachments over 8x4 pixels
//! whose imageblock sample length is the `Cell` pipelines' 32 bytes:
//!
//! - The pass keeps its imageblock across draws and across the split a tile dispatch makes: a
//!   cleared cell, two accumulations around the dispatch, and a resolve into the color attachment
//!   give each pixel the value the fixture's arithmetic gives (exact in `half`), the cells a
//!   discarded fragment left at the storage's zero.
//! - Its storage is announced once (`fragment-imageblock-storage`), beside the split's and the
//!   dispatch's scratch.
//! - Commands of an indirect command buffer reach the same storage as the encoder's draws, in their
//!   order, where the encoder made it and where only a command did: Metal draws such a command as
//!   the encoder's own draw of it.
//! - A pipeline of more imageblock bytes per sample than the pass gives is refused as it is set,
//!   as Metal's validation refuses it; a draw of another master's alignment after the pass's
//!   first, and a draw beside a tile dispatch whose tile function's explicit imageblock is of
//!   another size (in either order), are `NotImplemented`.
//!
//! A device without the ordered pixel interlock refuses an imageblock's state, and each test skips
//! it (`Fixture::imageblocks`).

use super::degrade_notices::{assert_once, made_render_kinds, made_tile_kinds};
use super::shared_kinds;
use super::tile_states::descriptor as tile_descriptor;
use super::{
    buffer_with, layer_reported_nothing_at_all, validated_device, Messages, ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::VkFeature;
use metal2vulkan::error::Feature;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, CommandBufferStatus, Degradation, Device, DrawProblem, Error, Invalid, Library,
    LoadAction, PipelineOptions, Problem, Region, RenderCommandEncoder, RenderPassDescriptor,
    RenderPipelineState, ResourceOptions, RuntimeGap, Size, StoreAction, Texture,
    TextureDescriptor,
};

/// The kind a pass's fragment imageblock takes (`degrade_notices`' closure).
pub(super) const FRAGMENT_IMAGEBLOCKS: [Degradation; 1] = [Degradation::FragmentImageblockStorage];

/// The attachments' formats: `k10ts_tile_grid`'s.
const FORMATS: [PixelFormat; 3] = [
    PixelFormat::R32Float,
    PixelFormat::R32Uint,
    PixelFormat::R32Uint,
];

/// The pass's extent.
const AREA: [u64; 2] = [8, 4];

/// The `Cell` pipelines' imageblock bytes per sample: the 16-byte cell, then the attachments'
/// 12 bytes rounded up to 8 and to the cell's alignment of 8.
const SAMPLE_LENGTH: u64 = 32;

/// The resolve's scale, a `half` 2.0.
const SCALE: u16 = 0x4000;

/// The fixtures' functions and states on one device.
struct Fixture {
    device: Device,
    views: Library,
}

impl Fixture {
    fn new(device: &Device) -> Fixture {
        let bitcode = llvm_assemble(&engine_dir().join("fragment_imageblock_views.ll")).unwrap();
        Fixture {
            views: device.new_library_with_air(&bitcode).unwrap(),
            device: device.clone(),
        }
    }

    /// Whether the device runs a fragment imageblock: it has the ordered pixel interlock and
    /// stores in the fragment stage, which the cells' writes need, and then this makes nothing.
    /// Where it lacks either, `imageblock_accumulate`'s state must be refused as `Imageblocks`; the
    /// refusal is printed and the caller skips the device.
    fn imageblocks(&self) -> bool {
        let profile = self.device.compile_device().profile();
        if profile.fragment().pixel_interlock
            && profile
                .enabled()
                .features
                .contains(&VkFeature::FragmentStoresAndAtomics)
        {
            return true;
        }
        let name = self.device.name();
        let descriptor = self.descriptor("imageblock_accumulate", false);
        match self
            .device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        {
            Err(Error::Unsupported(refusal)) if refusal.feature == Feature::Imageblocks => {
                println!("IMAGEBLOCKS-SKIP on {name}: no imageblock form, refused: {refusal}");
                false
            }
            Ok(_) => panic!("{name}: an imageblock's state was made with no form in the profile"),
            Err(error) => panic!("{name}: an imageblock's state was refused otherwise: {error}"),
        }
    }

    /// The state of the fullscreen triangle and the views fixture's `fragment` over the pass's
    /// attachments.
    fn state(&self, fragment: &str) -> RenderPipelineState {
        self.state_for(fragment, false)
    }

    /// [`Fixture::state`]'s state, made with `supportIndirectCommandBuffers` where `indirect`.
    fn state_for(&self, fragment: &str, indirect: bool) -> RenderPipelineState {
        self.device
            .new_render_pipeline_state(
                &self.descriptor(fragment, indirect),
                &PipelineOptions::default(),
            )
            .unwrap()
    }

    /// The descriptor of [`Fixture::state_for`]'s state.
    fn descriptor(&self, fragment: &str, indirect: bool) -> RenderPipelineDescriptor {
        let function = |name: &str| {
            let function = self.views.new_function_with_name(name).unwrap();
            StageSource::Function(function.function_descriptor())
        };
        let mut descriptor = RenderPipelineDescriptor::new(function("imageblock_vertex"));
        descriptor.fragment = Some(function(fragment));
        for (slot, format) in descriptor.color_attachments.iter_mut().zip(FORMATS) {
            *slot = Some(ColorAttachmentDescriptor::new(format));
        }
        descriptor.support_indirect_command_buffers = indirect;
        descriptor
    }

    /// The tile state of the owned tile function `stem` over the pass's attachments, or why the
    /// device makes none (`Unsupported`).
    fn tile_state(&self, stem: &str) -> Result<RenderPipelineState, Error> {
        self.device.new_tile_render_pipeline_state(
            &tile_descriptor(&self.device, stem, &FORMATS),
            &PipelineOptions::default(),
        )
    }

    /// The pass's attachments, each holding `texel(x, y)` at every pixel.
    fn targets(&self, texel: impl Fn(u64, u64) -> u32) -> [Texture; 3] {
        FORMATS.map(|format| {
            let [width, height] = AREA;
            let mut descriptor = TextureDescriptor::texture_2d(format, width, height, false);
            descriptor.usage = TextureUsage::RENDER_TARGET;
            descriptor.resource_options = ResourceOptions::shared();
            let texture = self.device.new_texture(&descriptor).unwrap();
            let bytes: Vec<u8> = pixels()
                .flat_map(|(x, y)| texel(x, y).to_le_bytes())
                .collect();
            texture
                .replace_region(
                    Region::new_2d(0, 0, width, height),
                    0,
                    0,
                    &bytes,
                    width * 4,
                    0,
                )
                .unwrap();
            texture
        })
    }

    /// Encode a pass over `targets`, loaded and stored, with a 16x16 tile and `sample_length`
    /// imageblock bytes per sample, by `encode`, and run it.
    fn run(
        &self,
        targets: &[Texture; 3],
        sample_length: u64,
        encode: impl FnOnce(&mut RenderCommandEncoder),
    ) {
        let queue = self.device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer
            .render_command_encoder(&pass(targets, sample_length))
            .unwrap();
        encode(&mut encoder);
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    }
}

/// A pass over `targets`, each loaded and stored, with a 16x16 tile and `sample_length` imageblock
/// bytes per sample.
fn pass(targets: &[Texture; 3], sample_length: u64) -> RenderPassDescriptor {
    let mut pass = RenderPassDescriptor::new();
    for (attachment, target) in pass.color_attachments.iter_mut().zip(targets) {
        attachment.texture = Some(target.clone());
        attachment.load_action = LoadAction::Load;
        attachment.store_action = StoreAction::Store;
    }
    pass.tile_width = 16;
    pass.tile_height = 16;
    pass.imageblock_sample_length = sample_length;
    pass
}

/// Draw `state`'s fullscreen triangle.
fn draw(encoder: &mut RenderCommandEncoder, state: &RenderPipelineState) -> Result<(), Error> {
    encoder.set_render_pipeline_state(state)?;
    encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
}

/// Every pixel's (x, y), in row order.
fn pixels() -> impl Iterator<Item = (u64, u64)> {
    (0..AREA[1]).flat_map(|y| (0..AREA[0]).map(move |x| (x, y)))
}

/// Every texel of `texture`, as 32-bit words in row order.
fn words(texture: &Texture) -> Vec<u32> {
    let [width, height] = AREA;
    let mut bytes = vec![0u8; (width * height * 4) as usize];
    texture
        .get_bytes(
            &mut bytes,
            width * 4,
            0,
            Region::new_2d(0, 0, width, height),
            0,
            0,
        )
        .unwrap();
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect()
}

/// A shared buffer of the resolve's scale.
fn scale(device: &Device) -> Buffer {
    buffer_with(device, &SCALE.to_le_bytes()).unwrap()
}

/// One pass on one device: `imageblock_clear_odd` (an even pixel's cell `(0, y + 0.5)`, an odd
/// one's discarded, so it stays the storage's zero), `imageblock_accumulate` (`color * 0.5 +
/// (x + 0.5, y + 0.5, 0.25, 0.5)`, `depth + 1`), a tile dispatch of `k10ts_tile_grid` (which splits
/// the pass and transforms the attachments), `imageblock_accumulate` again, and
/// `imageblock_resolve` (`color * depth * 2` into attachment 0's red). Every pixel's red is
/// `1.5 (x + 0.5)` times its depth, `y + 2.5` at an even pixel and 2 at an odd one, exact in
/// `half`; the dispatch's transforms of attachments 1 and 2 stay. Two such passes announce the
/// storage, the split and the dispatch's scratch once each, beside what the states' stages record.
#[test]
fn a_pass_keeps_its_imageblock_across_draws_and_a_split() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    // Beside the test's kinds, what its `Shared` buffers force on the device and its states'
    // makings announce.
    let mut kinds = vec![
        Degradation::FragmentImageblockStorage,
        Degradation::RenderPassSplit,
        Degradation::TileDispatchScratch,
    ];
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        kinds.extend(shared_kinds(&device));
        let fixture = Fixture::new(&device);
        if !fixture.imageblocks() {
            return;
        }
        for fragment in [
            "imageblock_clear_odd",
            "imageblock_accumulate",
            "imageblock_resolve",
        ] {
            kinds.extend(made_render_kinds(
                &device,
                &fixture.descriptor(fragment, false),
            ));
        }
        kinds.extend(made_tile_kinds(
            &device,
            &tile_descriptor(&device, "k10ts_tile_grid", &FORMATS),
        ));
        let clear = fixture.state("imageblock_clear_odd");
        let accumulate = fixture.state("imageblock_accumulate");
        let resolve = fixture.state("imageblock_resolve");
        assert_eq!(accumulate.imageblock_sample_length(), Ok(SAMPLE_LENGTH));
        let grid = fixture.tile_state("k10ts_tile_grid").unwrap();
        let scale = scale(&device);
        let out = device
            .new_buffer(512 * 12 * 4, ResourceOptions::shared())
            .unwrap();
        for _ in 0..2 {
            let targets = fixture.targets(|x, y| (7 * x + y) as u32);
            fixture.run(&targets, SAMPLE_LENGTH, |encoder| {
                draw(encoder, &clear).unwrap();
                draw(encoder, &accumulate).unwrap();
                encoder.set_render_pipeline_state(&grid).unwrap();
                encoder.set_tile_buffer(Some(&out), 0, 0).unwrap();
                encoder
                    .dispatch_threads_per_tile(Size::new(8, 8, 1))
                    .unwrap();
                draw(encoder, &accumulate).unwrap();
                encoder.set_fragment_buffer(Some(&scale), 0, 0).unwrap();
                draw(encoder, &resolve).unwrap();
            });
            let [reds, vs, tags] = [&targets[0], &targets[1], &targets[2]].map(words);
            for (i, (x, y)) in pixels().enumerate() {
                let depth = if x % 2 == 0 { y as f32 + 2.5 } else { 2.0 };
                let red = 1.5 * (x as f32 + 0.5) * depth * 2.0;
                assert_eq!(f32::from_bits(reds[i]), red, "red at ({x}, {y})");
                let k = 1 + (x % 2) as u32 + 2 * (y % 2) as u32;
                assert_eq!(vs[i], (7 * x + y) as u32 * 3 + k, "v at ({x}, {y})");
                let tag = 0xA000_0000 | ((y as u32) << 8) | x as u32;
                assert_eq!(tags[i], tag, "tag at ({x}, {y})");
            }
        }
    }
    layer_reported_nothing_at_all(&messages);
    assert_once(&messages.take_notices(), &kinds);
}

/// A `Cell` pipeline in a pass that gives the imageblock only its attachments' 12 bytes is
/// refused as it is set (Metal: "the pipelineState's per sample imageBlock usage(32) is greater
/// than the encoder's perSample imageBlock usage(12)"); a draw of `imageblock_transition`, whose
/// master is another of the `Cell`'s 16 bytes but aligned to 4 where the `Cell` aligns to 8,
/// after a `Cell` draw is `NotImplemented(ImageblockMasters)`; and a `Cell` draw and a tile
/// dispatch of `a4_imageblock_extent`, whose tile function's explicit imageblock is the pass's
/// storage but of 8 bytes, where the `Cell` is 16, refuse each other in either order
/// (`NotImplemented(ImageblockMasters)`). Each refused call records nothing, so each pass still
/// runs.
#[test]
fn imageblock_draws_are_refused_where_metal_or_the_runtime_refuses_them() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let fixture = Fixture::new(&device);
        if !fixture.imageblocks() {
            return;
        }
        let accumulate = fixture.state("imageblock_accumulate");
        let transition = fixture.state("imageblock_transition");
        let targets = fixture.targets(|_, _| 0);
        fixture.run(&targets, 0, |encoder| {
            assert_eq!(
                encoder.set_render_pipeline_state(&accumulate),
                Err(Error::Invalid(Invalid::new(
                    "set_render_pipeline_state",
                    Problem::Draw(DrawProblem::ImageblockSampleLength {
                        pipeline: SAMPLE_LENGTH,
                        pass: 12,
                    }),
                )))
            );
        });
        fixture.run(&targets, SAMPLE_LENGTH, |encoder| {
            draw(encoder, &accumulate).unwrap();
            assert_eq!(
                draw(encoder, &transition),
                Err(Error::NotImplemented(RuntimeGap::ImageblockMasters))
            );
        });
        let explicit = match fixture.tile_state("a4_imageblock_extent") {
            Ok(state) => state,
            Err(Error::Unsupported(why)) => {
                eprintln!("a4_imageblock_extent's tile state is not made on this device: {why}");
                return;
            }
            Err(error) => panic!("a4_imageblock_extent's tile state: {error}"),
        };
        let out = device
            .new_buffer(16 * 16 * 16, ResourceOptions::shared())
            .unwrap();
        let dispatch = |encoder: &mut RenderCommandEncoder| {
            encoder.set_render_pipeline_state(&explicit).unwrap();
            encoder.set_tile_buffer(Some(&out), 0, 0).unwrap();
            encoder.dispatch_threads_per_tile(Size::new(8, 8, 1))
        };
        fixture.run(&targets, SAMPLE_LENGTH, |encoder| {
            draw(encoder, &accumulate).unwrap();
            assert_eq!(
                dispatch(encoder),
                Err(Error::NotImplemented(RuntimeGap::ImageblockMasters))
            );
        });
        fixture.run(&targets, SAMPLE_LENGTH, |encoder| {
            dispatch(encoder).unwrap();
            assert_eq!(
                draw(encoder, &accumulate),
                Err(Error::NotImplemented(RuntimeGap::ImageblockMasters))
            );
        });
    }
    layer_reported_nothing_at_all(&messages);
}

/// What a step of [`commands_reach_the_pass_s_imageblock`] draws: the encoder's own draw of a
/// state, or an execution of a buffer of one command per state, which inherits the encoder's
/// buffers.
enum Step<'a> {
    Draw(&'a RenderPipelineState),
    Execute(&'a [&'a RenderPipelineState]),
}

/// Commands of an indirect command buffer whose fragment functions reach the fragment imageblock
/// read and write the pass's storage as the encoder's own draws of their pipelines do, Metal's
/// answer (an imageblock initialized by the encoder's draw, accumulated
/// by a command and resolved by the encoder's draw, as the encoder's three draws give). Each pass
/// is `a_pass_keeps_its_imageblock_across_draws_and_a_split`'s without the tile dispatch,
/// `imageblock_clear_odd`, `imageblock_accumulate` twice and `imageblock_resolve`, some of them
/// commands: every pixel's red is `1.5 (x + 0.5)` times its depth times 2, its depth `y + 2.5` at
/// an even pixel and 2 at an odd one, whose cell keeps the storage's zero; the other attachments
/// keep their texels. Where every step is a command, a command's draw makes the storage, which
/// the pass clears before its first instance as the encoder's own; it is announced once.
#[test]
fn commands_reach_the_pass_s_imageblock() {
    use metal2vulkan::metal::IndirectCommandType;
    use metal2vulkan_runtime::IndirectCommandBufferDescriptor;
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let fixture = Fixture::new(&device);
        if !fixture.imageblocks() {
            return;
        }
        let clear = fixture.state_for("imageblock_clear_odd", true);
        let accumulate = fixture.state_for("imageblock_accumulate", true);
        let resolve = fixture.state_for("imageblock_resolve", true);
        let scale = scale(&device);
        let cases: [(&str, &[Step]); 4] = [
            (
                "a command between the encoder's draws",
                &[
                    Step::Draw(&clear),
                    Step::Execute(&[&accumulate]),
                    Step::Draw(&accumulate),
                    Step::Draw(&resolve),
                ],
            ),
            (
                "two commands",
                &[
                    Step::Draw(&clear),
                    Step::Execute(&[&accumulate, &accumulate]),
                    Step::Draw(&resolve),
                ],
            ),
            (
                "commands before the encoder's first such draw",
                &[
                    Step::Execute(&[&clear, &accumulate]),
                    Step::Draw(&accumulate),
                    Step::Draw(&resolve),
                ],
            ),
            (
                "commands alone",
                &[Step::Execute(&[&clear, &accumulate, &accumulate, &resolve])],
            ),
        ];
        let name = device.name();
        let mut held = Vec::new();
        for (case, steps) in cases {
            let targets = fixture.targets(|x, y| (7 * x + y) as u32);
            fixture.run(&targets, SAMPLE_LENGTH, |encoder| {
                encoder.set_fragment_buffer(Some(&scale), 0, 0).unwrap();
                for step in steps {
                    match step {
                        Step::Draw(state) => draw(encoder, state).unwrap(),
                        Step::Execute(states) => {
                            let mut descriptor = IndirectCommandBufferDescriptor::new();
                            descriptor.command_types = IndirectCommandType::DRAW;
                            descriptor.inherit_buffers = true;
                            let icb = device
                                .new_indirect_command_buffer(
                                    &descriptor,
                                    states.len() as u64,
                                    ResourceOptions::shared(),
                                )
                                .unwrap();
                            for (index, state) in states.iter().enumerate() {
                                let command = icb.indirect_render_command_at(index as u64).unwrap();
                                command.set_render_pipeline_state(state).unwrap();
                                command
                                    .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                                    .unwrap();
                            }
                            encoder
                                .execute_commands_in_buffer(&icb, 0..states.len() as u64)
                                .unwrap();
                            held.push(icb);
                        }
                    }
                }
            });
            let [reds, vs, tags] = [&targets[0], &targets[1], &targets[2]].map(words);
            for (i, (x, y)) in pixels().enumerate() {
                let depth = if x % 2 == 0 { y as f32 + 2.5 } else { 2.0 };
                let red = 1.5 * (x as f32 + 0.5) * depth * 2.0;
                assert_eq!(
                    f32::from_bits(reds[i]),
                    red,
                    "{case}: red at ({x}, {y}) on {name}"
                );
                let texel = (7 * x + y) as u32;
                assert_eq!(vs[i], texel, "{case}: v at ({x}, {y}) on {name}");
                assert_eq!(tags[i], texel, "{case}: tag at ({x}, {y}) on {name}");
            }
            eprintln!("fragment imageblock, {case}, on {name}: drawn as the encoder's draws");
        }
        drop(held);
    }
    layer_reported_nothing_at_all(&messages);
    let notices = messages.take_notices();
    let storage = notices
        .iter()
        .filter_map(|message| message.degradation.as_ref())
        .filter(|notice| notice.degradation == Degradation::FragmentImageblockStorage)
        .count();
    assert_eq!(storage, 1, "the storage announced once");
}
