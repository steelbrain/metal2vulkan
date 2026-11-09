//! Device tests of a fragment function's `[[stencil]]` output on the device the cfg selects, under
//! the validation layer. Metal tests and writes the stencil with the output as both faces'
//! reference (its low 8 bits). A device with `VK_EXT_shader_stencil_export` hands the output to its
//! fixed-function test; on one without it (NVIDIA's) the runtime runs the shader-side stencil, a
//! shadow of the stencil the function tests and writes itself, copied in before each run of such
//! draws and back after it (`runtime/shadow_stencil.rs`). Each pass here runs on the device as
//! opened and again with the extension withheld (`DeviceOptions::withheld_extensions`), so that
//! every venue runs the shadow, and both give Metal's answer, that of a Metal probe on an M4. A
//! device with neither the extension nor the ordered pixel interlock refuses the output's states,
//! and that run is skipped (`Lane::outputs_stencil`).
//!
//! The pass, as the probe drew it: a 4x1 `Stencil8` target of 1 or 4 samples, its stencil loaded
//! by `Clear` to 0x40 or by `Load` of an earlier pass's clear to 0x40, the encoder's stencil
//! reference 0x99 (which a function's output replaces), draws the fullscreen triangle three times:
//! by the owned fixture `stx_stencil`'s `stx_stencil_by_pixel` (word `x` of a buffer) under both
//! faces' `Always` and `Replace`; by a pipeline without a fragment function under `IncrementClamp`;
//! and by the output again, under a front face of `Never` and a back face of `Equal` with read mask
//! 0x0f, write mask 0xf0, failure `Invert` and pass `Zero`. At 4 samples the stencil also resolves
//! sample 0. Every sample is read back (`runtime_samples`' kernel at 4 samples, a blit at 1). The
//! same pass also draws at 4 samples over a `Depth32Float_Stencil8` target attached as depth too
//! (cleared, not stored) and resolving nothing, whose stencil Metal's stencil rules give the same
//! samples (the probe drew `Stencil8` only); on a driver whose depth-stencil/colour copies are
//! wrong, its shadow is copied by a kernel and stencil draws (`runtime/stencil_copy.rs`). And it
//! draws over a texture made in `X32_Stencil8` at 1 and 4 samples, resolving nothing, which plain
//! Metal makes as a `Stencil8` texture: its stencil Metal's too.
//!
//! The executed pass draws the same at 1 and 4 samples over a `Stencil8` target, its first and
//! third draws as commands of an indirect command buffer executed between the encoder's draws, each
//! under the encoder's state at the call, which Metal draws as the encoder's own: Metal's stencil.
//! Through the shadow each such command draws in a run of its own, the
//! shadow filled before it and copied back after it. It then draws one execution of an output
//! command, a plain command and an output command under the first draw's state, whose stencil is
//! the first draw's outputs.
//!
//! The memoryless pass (probe 9b) draws the same over a memoryless 1-sample stencil cleared to
//! 0x40 beside an `R32Uint` colour, the first and third draws by `stx_stencil_ms`'s
//! `stx_stencil_ms_color`, which returns colour 1, then 2, beside the stencil: the third draw's
//! colour lands only where its stencil test passes. The shadow renders a memoryless stencil into
//! its stand-in, as a split pass does.
//!
//! The host step's pass executes a command whose records the GPU writes, so the execution runs in a
//! host step, drawing by `stx_stencil_ms_color` with a state made after the command buffer's
//! commit: over an `R32Uint` colour alone, the function's stencil reaches no attachment and the
//! shadow's plan binds a neutral its state's creation made; beside a stencil, the shadow's run is
//! one of another shape, whose render pass, framebuffer and shadow the host step makes.

use super::degrade_notices::shared_in_system_memory;
use super::degrade_notices::{assert_made_once, forced, made_compute_kinds, made_render_kinds};
use super::{
    buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device_with, Messages,
    ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::candidate::runtime_samples;
use crate::executor_contract::multisample::SampleRead;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::{DriverFact, VkExtension};
use metal2vulkan::error::Feature;
use metal2vulkan::metal::state::{DepthStencilDescriptor, StencilDescriptor};
use metal2vulkan::metal::{
    CompareFunction, IndirectCommandType, PixelFormat, PrimitiveType, StencilOperation,
    TextureType, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    BlitOption, Buffer, ClearColor, CommandBuffer, CommandBufferStatus, CommandQueue, CpuCacheMode,
    Degradation, DepthStencilState, Device, Error, EventRef, Function, HazardTrackingMode,
    IndirectCommandBufferDescriptor, LoadAction, MultisampleStencilResolveFilter, Origin,
    PipelineOptions, RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, Size, StorageMode, StoreAction, Texture, TextureDescriptor,
};
use std::collections::BTreeSet;

/// The kind the shader-side stencil's passes take (`degrade_notices`' closure).
pub(super) const SHADOW: [Degradation; 1] = [Degradation::ShaderSideStencil];

/// The pass's width; it is one pixel high.
const WIDTH: u32 = 4;

/// The combined depth/stencil format the pass also draws over at 4 samples, whose shadow a driver
/// with `RuntimeDriverDefect::MultisampledDepthStencilColourCopies` copies by a kernel and stencil
/// draws (`runtime/stencil_copy.rs`), not by colour copies.
const COMBINED: PixelFormat = PixelFormat::Depth32FloatStencil8;

/// The stencil the pass clears to, or loads from an earlier pass's clear.
const CLEARED: u32 = 0x40;

/// The first draw's outputs, by pixel.
const FIRST: [u32; 4] = [0x10, 0x50, 0x3f, 0x40];

/// The third draw's outputs, by pixel.
const THIRD: [u32; 4] = [0x11, 0x00, 0x50, 0x41];

/// The encoder's stencil reference, which a function's output replaces.
const REFERENCE: u32 = 0x99;

/// Metal's stencil after the three draws, by pixel, in each of its samples (a Metal probe on an M4
/// with macOS 27.0.1, with and without Metal's API validation, every case): the fullscreen triangle
/// is back-facing under Metal's default winding, so the third draw takes the back face.
const METAL: [u8; 4] = [0x01, 0xa1, 0x00, 0x01];

/// The colours the memoryless pass's first and third draws return, every pixel.
const COLORS: [u32; 2] = [1, 2];

/// Metal's colour after the memoryless pass's three draws, by pixel (a Metal probe on an M4 with
/// macOS 27.0.1, with and without Metal's API validation): the third draw's colour where its
/// stencil test passes, the first's where it fails.
const METAL_COLOR: [u32; 4] = [2, 1, 2, 2];

/// The lane's device, its queue, and the functions it draws with: `runtime_samples`' vertex
/// function (whose library also holds the readback kernels), `stx_stencil`'s
/// `stx_stencil_by_pixel`, and `stx_stencil_ms`'s `stx_stencil_ms_color`, which returns a colour
/// beside the stencil.
struct Lane {
    device: Device,
    queue: CommandQueue,
    vertex: Function,
    by_pixel: Function,
    with_color: Function,
}

/// A fragment buffer: its index and the buffer.
type Bound<'a> = (u32, &'a Buffer);

impl Lane {
    fn new(device: Device) -> Lane {
        let samples = runtime_samples::library(&device).unwrap();
        let library = |stem: &str| {
            let bitcode = llvm_assemble(&engine_dir().join(format!("{stem}.ll"))).unwrap();
            device.new_library_with_air(&bitcode).unwrap()
        };
        Lane {
            queue: device.new_command_queue().unwrap(),
            vertex: samples.new_function_with_name("rt6_fill_vertex").unwrap(),
            by_pixel: library("stx_stencil")
                .new_function_with_name("stx_stencil_by_pixel")
                .unwrap(),
            with_color: library("stx_stencil_ms")
                .new_function_with_name("stx_stencil_ms_color")
                .unwrap(),
            device,
        }
    }

    /// A pipeline of the fullscreen triangle and `fragment` (none: no fragment function) over a
    /// stencil attachment of `format` (a combined format also as the depth attachment), and a
    /// colour one of `color` where given, of `samples`.
    fn pipeline(
        &self,
        fragment: Option<&Function>,
        color: Option<PixelFormat>,
        format: PixelFormat,
        samples: u32,
    ) -> Result<RenderPipelineState, Error> {
        self.pipeline_for(fragment, color, (format, samples), false)
    }

    /// [`Lane::pipeline`]'s pipeline, made with `supportIndirectCommandBuffers` where
    /// `indirect`.
    fn pipeline_for(
        &self,
        fragment: Option<&Function>,
        color: Option<PixelFormat>,
        (format, samples): (PixelFormat, u32),
        indirect: bool,
    ) -> Result<RenderPipelineState, Error> {
        let descriptor = self.descriptor_for(fragment, color, (format, samples), indirect);
        self.device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
    }

    /// The descriptor of [`Lane::pipeline_for`]'s pipeline.
    fn descriptor_for(
        &self,
        fragment: Option<&Function>,
        color: Option<PixelFormat>,
        (format, samples): (PixelFormat, u32),
        indirect: bool,
    ) -> RenderPipelineDescriptor {
        let mut descriptor =
            RenderPipelineDescriptor::new(StageSource::Function(self.vertex.function_descriptor()));
        descriptor.fragment = fragment.map(|f| StageSource::Function(f.function_descriptor()));
        descriptor.color_attachments[0] = color.map(ColorAttachmentDescriptor::new);
        descriptor.stencil_format = Some(format);
        if format == COMBINED {
            descriptor.depth_format = Some(format);
        }
        descriptor.raster_sample_count = samples;
        descriptor.support_indirect_command_buffers = indirect;
        descriptor
    }

    /// The kinds the makings of an output pipeline of `fragment` and a plain one over `color`
    /// and a stencil of `format` at `samples` announce ([`Lane::pipeline_for`]), and where
    /// `samples` exceed one, the kernel's that reads the stencil's samples.
    fn made(
        &self,
        fragment: &Function,
        color: Option<PixelFormat>,
        target: (PixelFormat, u32),
        indirect: bool,
    ) -> BTreeSet<Degradation> {
        let mut made = BTreeSet::new();
        for fragment in [Some(fragment), None] {
            let descriptor = self.descriptor_for(fragment, color, target, indirect);
            made.extend(made_render_kinds(&self.device, &descriptor));
        }
        if target.1 > 1 {
            let read = runtime_samples::read_function(
                &self.device,
                PixelFormat::Stencil8,
                SampleRead::Stencil,
            )
            .unwrap();
            made.extend(made_compute_kinds(&self.device, read));
        }
        made
    }

    /// Whether the device runs a fragment function's `[[stencil]]` output: it has
    /// `VK_EXT_shader_stencil_export`, or the ordered pixel interlock the shadow runs in, and
    /// then this makes nothing. Where it has neither, the output's state ([`Lane::stepped`]'s,
    /// over the colour alone) must be refused as `StencilExport`; the refusal is printed and the
    /// caller skips the device.
    fn outputs_stencil(&self) -> bool {
        let fragment = self.device.compile_device().profile().fragment();
        if fragment.stencil_export || fragment.pixel_interlock {
            return true;
        }
        let name = self.device.name();
        let descriptor = self.stepped_descriptor(false);
        match self
            .device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        {
            Err(Error::Unsupported(refusal)) if refusal.feature == Feature::StencilExport => {
                println!("STENCIL-SKIP on {name}: no stencil output form, refused: {refusal}");
                false
            }
            Ok(_) => {
                panic!("{name}: a stencil output's state was made with no form in the profile")
            }
            Err(error) => panic!("{name}: a stencil output's state was refused otherwise: {error}"),
        }
    }

    /// The descriptor of the state [`Lane::stepped`] makes after the commit.
    fn stepped_descriptor(&self, attached: bool) -> RenderPipelineDescriptor {
        let mut descriptor =
            RenderPipelineDescriptor::new(StageSource::Function(self.vertex.function_descriptor()));
        descriptor.fragment = Some(StageSource::Function(self.with_color.function_descriptor()));
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::R32Uint));
        descriptor.stencil_format = attached.then_some(PixelFormat::Stencil8);
        descriptor.support_indirect_command_buffers = true;
        descriptor
    }

    /// A texture of `pf`, `samples` and `usage`, of `options`, the pass's extent.
    fn texture(
        &self,
        pf: PixelFormat,
        samples: u32,
        usage: TextureUsage,
        options: ResourceOptions,
    ) -> Texture {
        let mut descriptor = TextureDescriptor::texture_2d(pf, WIDTH.into(), 1, false);
        if samples > 1 {
            descriptor.texture_type = TextureType::Type2DMultisample;
            descriptor.sample_count = samples.into();
        }
        descriptor.usage = usage;
        descriptor.resource_options = options;
        self.device.new_texture(&descriptor).unwrap()
    }

    /// A private stencil render target of `format` and `samples`, which shaders may read (a
    /// combined format through its `X32Stencil8` view, so of `PixelFormatView` usage too).
    fn stencil(&self, format: PixelFormat, samples: u32) -> Texture {
        let mut usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        if format == COMBINED {
            usage |= TextureUsage::PIXEL_FORMAT_VIEW;
        }
        self.texture(format, samples, usage, ResourceOptions::private())
    }

    /// A shared buffer of `values`.
    fn words(&self, values: &[u32]) -> Buffer {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        buffer_with(&self.device, &bytes).unwrap()
    }

    /// The three draws' depth/stencil states, the depth test `Always` without writes: both faces
    /// `Always` and `Replace`; both `Always` and `IncrementClamp`; and a front face of `Never`
    /// beside a back face of `Equal`, read mask 0x0f, write mask 0xf0, failure `Invert` and pass
    /// `Zero`.
    fn states(&self) -> [DepthStencilState; 3] {
        use CompareFunction as C;
        use StencilOperation as O;
        let state = |front: StencilDescriptor, back: StencilDescriptor| {
            let mut descriptor = DepthStencilDescriptor::default();
            descriptor.front_face_stencil = front;
            descriptor.back_face_stencil = back;
            self.device.new_depth_stencil_state(&descriptor).unwrap()
        };
        let replace = face(C::Always, O::Keep, O::Replace);
        let increment = face(C::Always, O::Keep, O::IncrementClamp);
        let mut equal = face(C::Equal, O::Invert, O::Zero);
        equal.read_mask = 0x0f;
        equal.write_mask = 0xf0;
        [
            state(replace, replace),
            state(increment, increment),
            state(face(C::Never, O::Keep, O::Keep), equal),
        ]
    }

    /// Encode the pass's three draws into `encoder`, the stencil reference set to [`REFERENCE`]
    /// and `shared` bound first: `output`, binding `first`; the pipeline without a fragment
    /// function, `plain`; `output` again, binding `third`; each under its state.
    fn draw(
        &self,
        encoder: &mut RenderCommandEncoder,
        (output, plain): (&RenderPipelineState, &RenderPipelineState),
        shared: &[Bound<'_>],
        [first, third]: [&[Bound<'_>]; 2],
    ) -> Result<(), Error> {
        let states = self.states();
        encoder.set_stencil_reference_value(REFERENCE);
        for &(index, buffer) in shared {
            encoder.set_fragment_buffer(Some(buffer), 0, index)?;
        }
        for (pipeline, state, bound) in [
            (output, &states[0], first),
            (plain, &states[1], &[][..]),
            (output, &states[2], third),
        ] {
            encoder.set_render_pipeline_state(pipeline)?;
            encoder.set_depth_stencil_state(state)?;
            for &(index, buffer) in bound {
                encoder.set_fragment_buffer(Some(buffer), 0, index)?;
            }
            encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)?;
        }
        Ok(())
    }

    /// Encode into `command_buffer` a blit of `texture`'s texels, `bytes` each, into a shared
    /// buffer, which it returns.
    fn blit_read(
        &self,
        command_buffer: &CommandBuffer,
        texture: &Texture,
        bytes: u64,
    ) -> Result<Buffer, Error> {
        let length = bytes * u64::from(WIDTH);
        let buffer = self.device.new_buffer(length, ResourceOptions::shared())?;
        let mut blit = command_buffer.blit_command_encoder()?;
        blit.copy_from_texture_to_buffer(
            texture,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(WIDTH.into(), 1, 1),
            &buffer,
            0,
            length,
            length,
            BlitOption::None,
        )?;
        blit.end_encoding()?;
        Ok(buffer)
    }

    /// Commit `command_buffer` and wait for it to complete.
    fn run(&self, command_buffer: &CommandBuffer) {
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    }

    /// The pass over a stencil of `format` at `samples`, its stencil loaded by `load`: each
    /// pixel's samples in order, and, for `Stencil8` at more than one sample, the resolve's
    /// pixels. A combined format is also the pass's depth attachment, cleared and not stored;
    /// it and `X32Stencil8` resolve nothing. A pipeline the device does not make is its refusal.
    fn drawn(
        &self,
        format: PixelFormat,
        samples: u32,
        load: LoadAction,
    ) -> Result<(Vec<u8>, Option<Vec<u8>>), Error> {
        let output = self.pipeline(Some(&self.by_pixel), None, format, samples)?;
        let plain = self.pipeline(None, None, format, samples)?;
        let (first, third, width) = (self.words(&FIRST), self.words(&THIRD), self.words(&[WIDTH]));
        let stencil = self.stencil(format, samples);
        let resolve =
            (samples > 1 && format == PixelFormat::Stencil8).then(|| self.stencil(format, 1));
        let command_buffer = self.queue.command_buffer()?;
        if load == LoadAction::Load {
            let mut clear = RenderPassDescriptor::new();
            clear.stencil_attachment.texture = Some(stencil.clone());
            clear.stencil_attachment.load_action = LoadAction::Clear;
            clear.stencil_attachment.clear_stencil = CLEARED;
            clear.stencil_attachment.store_action = StoreAction::Store;
            command_buffer
                .render_command_encoder(&clear)?
                .end_encoding()?;
        }
        let mut pass = RenderPassDescriptor::new();
        if format == COMBINED {
            let depth = &mut pass.depth_attachment;
            depth.texture = Some(stencil.clone());
            depth.load_action = LoadAction::Clear;
            depth.store_action = StoreAction::DontCare;
        }
        let target = &mut pass.stencil_attachment;
        target.texture = Some(stencil.clone());
        target.load_action = load;
        target.clear_stencil = CLEARED;
        target.store_action = StoreAction::Store;
        if let Some(resolve) = &resolve {
            target.store_action = StoreAction::StoreAndMultisampleResolve;
            target.resolve_texture = Some(resolve.clone());
            target.stencil_resolve_filter = MultisampleStencilResolveFilter::Sample0;
        }
        let mut encoder = command_buffer.render_command_encoder(&pass)?;
        self.draw(
            &mut encoder,
            (&output, &plain),
            &[(1, &width)],
            [&[(0, &first)], &[(0, &third)]],
        )?;
        encoder.end_encoding()?;
        // A combined texture's stencil is read through its `X32Stencil8` view.
        let samples_read = (samples > 1).then(|| {
            let source = match format {
                COMBINED => stencil.new_texture_view(PixelFormat::X32Stencil8).unwrap(),
                _ => stencil.clone(),
            };
            runtime_samples::encode_through(
                &self.device,
                &command_buffer,
                &source,
                "rt6_read_stencil",
                SampleRead::Stencil,
                [WIDTH, 1],
                samples,
            )
            .unwrap()
        });
        // The single-sampled texels: the resolve's, or the stencil's at one sample.
        let single = match (&resolve, samples) {
            (Some(resolve), _) => Some(resolve),
            (None, 1) => Some(&stencil),
            (None, _) => None,
        };
        let single = single
            .map(|texture| self.blit_read(&command_buffer, texture, 1))
            .transpose()?;
        self.run(&command_buffer);
        let single = single.map(|buffer| read_buffer(&buffer, 0, WIDTH.into()).unwrap());
        Ok(match samples_read {
            Some(read) => (read.bytes().unwrap(), single),
            None => (single.expect("one sample's texels"), None),
        })
    }

    /// The executed pass over a cleared `Stencil8` at `samples` (the module docs): where
    /// `sequence`, the three draws with the first and third as one command each of an indirect
    /// command buffer executed under the encoder's state at the call, else one execution of an
    /// output command, a plain command and an output command under the first draw's state, the
    /// outputs [`FIRST`] both times; each pixel's samples in order. A pipeline the device does not
    /// make is its refusal.
    fn executed(&self, samples: u32, sequence: bool) -> Result<Vec<u8>, Error> {
        let format = PixelFormat::Stencil8;
        let output = self.pipeline_for(Some(&self.by_pixel), None, (format, samples), true)?;
        let plain = self.pipeline_for(None, None, (format, samples), true)?;
        let (first, third, width) = (self.words(&FIRST), self.words(&THIRD), self.words(&[WIDTH]));
        let stencil = self.stencil(format, samples);
        let states = self.states();
        let commands: &[&RenderPipelineState] = match sequence {
            true => &[&output],
            false => &[&output, &plain, &output],
        };
        let mut descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.command_types = IndirectCommandType::DRAW;
        descriptor.inherit_buffers = true;
        let count = commands.len() as u64;
        let icb = self.device.new_indirect_command_buffer(
            &descriptor,
            count,
            ResourceOptions::shared(),
        )?;
        for (index, state) in commands.iter().enumerate() {
            let command = icb.indirect_render_command_at(index as u64)?;
            command.set_render_pipeline_state(state)?;
            command.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)?;
        }
        let command_buffer = self.queue.command_buffer()?;
        let mut pass = RenderPassDescriptor::new();
        let target = &mut pass.stencil_attachment;
        target.texture = Some(stencil.clone());
        target.load_action = LoadAction::Clear;
        target.clear_stencil = CLEARED;
        target.store_action = StoreAction::Store;
        let mut encoder = command_buffer.render_command_encoder(&pass)?;
        encoder.set_stencil_reference_value(REFERENCE);
        encoder.set_fragment_buffer(Some(&width), 0, 1)?;
        encoder.set_depth_stencil_state(&states[0])?;
        encoder.set_fragment_buffer(Some(&first), 0, 0)?;
        encoder.execute_commands_in_buffer(&icb, 0..count)?;
        if sequence {
            encoder.set_render_pipeline_state(&plain)?;
            encoder.set_depth_stencil_state(&states[1])?;
            encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)?;
            encoder.set_depth_stencil_state(&states[2])?;
            encoder.set_fragment_buffer(Some(&third), 0, 0)?;
            encoder.execute_commands_in_buffer(&icb, 0..count)?;
        }
        encoder.end_encoding()?;
        let read = match samples {
            1 => Read::Blit(self.blit_read(&command_buffer, &stencil, 1)?),
            _ => Read::Samples(
                runtime_samples::encode_through(
                    &self.device,
                    &command_buffer,
                    &stencil,
                    "rt6_read_stencil",
                    SampleRead::Stencil,
                    [WIDTH, 1],
                    samples,
                )
                .unwrap(),
            ),
        };
        command_buffer.commit()?;
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        drop(icb);
        Ok(match read {
            Read::Blit(buffer) => read_buffer(&buffer, 0, WIDTH.into()).unwrap(),
            Read::Samples(read) => read.bytes().unwrap(),
        })
    }

    /// The memoryless pass: a memoryless 1-sample `Stencil8` cleared to [`CLEARED`] and not
    /// stored, beside an `R32Uint` colour cleared to 0 and stored, drawn as [`Lane::drawn`] draws
    /// it, the first and third draw by `stx_stencil_ms_color` returning [`COLORS`] as their
    /// colours: the colour, by pixel.
    fn memoryless(&self) -> Result<Vec<u32>, Error> {
        let uint = PixelFormat::R32Uint;
        let stencil8 = PixelFormat::Stencil8;
        let output = self.pipeline(Some(&self.with_color), Some(uint), stencil8, 1)?;
        let plain = self.pipeline(None, Some(uint), stencil8, 1)?;
        let stencil = self.texture(
            PixelFormat::Stencil8,
            1,
            TextureUsage::RENDER_TARGET,
            ResourceOptions::new(
                StorageMode::Memoryless,
                CpuCacheMode::DefaultCache,
                HazardTrackingMode::Default,
            ),
        );
        let color = self.texture(
            uint,
            1,
            TextureUsage::RENDER_TARGET,
            ResourceOptions::private(),
        );
        let mut pass = RenderPassDescriptor::new();
        let target = &mut pass.color_attachments[0];
        target.texture = Some(color.clone());
        target.load_action = LoadAction::Clear;
        target.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        target.store_action = StoreAction::Store;
        let target = &mut pass.stencil_attachment;
        target.texture = Some(stencil);
        target.load_action = LoadAction::Clear;
        target.clear_stencil = CLEARED;
        target.store_action = StoreAction::DontCare;
        let (first, third, width) = (self.words(&FIRST), self.words(&THIRD), self.words(&[WIDTH]));
        let [one, two] = COLORS.map(|c| self.words(&[c; WIDTH as usize]));
        let command_buffer = self.queue.command_buffer()?;
        let mut encoder = command_buffer.render_command_encoder(&pass)?;
        self.draw(
            &mut encoder,
            (&output, &plain),
            &[(2, &width)],
            [&[(0, &first), (1, &one)], &[(0, &third), (1, &two)]],
        )?;
        encoder.end_encoding()?;
        let read = self.blit_read(&command_buffer, &color, 4)?;
        self.run(&command_buffer);
        Ok(read_buffer(&read, 0, 4 * u64::from(WIDTH))
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect())
    }

    /// The host step's pass (the module docs): an `R32Uint` colour cleared to 0 and stored,
    /// beside, where `attached`, a 1-sample `Stencil8` cleared to [`CLEARED`] and stored under
    /// the first draw's depth/stencil state and the encoder's [`REFERENCE`], executes command 0
    /// of a buffer of two draw commands that inherit no buffers. An earlier command buffer's
    /// blit copied command 0 to command 1, so the GPU writes the records and the execution runs
    /// in a host step. The pass's command buffer waits for an event; once its commit returned,
    /// the state is made, `stx_stencil_ms_color` over the colour and the stencil where
    /// `attached`, with `supportIndirectCommandBuffers`, and command 0 draws the fullscreen
    /// triangle with it, its fragment buffers the outputs [`FIRST`], the colour
    /// [`COLORS`]`[0]` at every pixel and the width; then the event is signalled and the command
    /// buffer waited for. The command buffer, and a blit of the colour's texels encoded after
    /// the pass. A pipeline the device does not make is its refusal, the command buffer run with
    /// command 0 empty.
    fn stepped(&self, attached: bool) -> Result<(CommandBuffer, Buffer), Error> {
        let uint = PixelFormat::R32Uint;
        let mut commands = IndirectCommandBufferDescriptor::new();
        commands.command_types = IndirectCommandType::DRAW;
        commands.inherit_buffers = false;
        commands.max_fragment_buffer_bind_count = 3;
        let icb =
            self.device
                .new_indirect_command_buffer(&commands, 2, ResourceOptions::shared())?;
        let copying = self.queue.command_buffer()?;
        let mut blit = copying.blit_command_encoder()?;
        blit.copy_indirect_command_buffer(&icb, 0..1, &icb, 1)?;
        blit.end_encoding()?;
        self.run(&copying);
        let color = self.texture(
            uint,
            1,
            TextureUsage::RENDER_TARGET,
            ResourceOptions::private(),
        );
        let mut pass = RenderPassDescriptor::new();
        let target = &mut pass.color_attachments[0];
        target.texture = Some(color.clone());
        target.load_action = LoadAction::Clear;
        target.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        target.store_action = StoreAction::Store;
        if attached {
            let target = &mut pass.stencil_attachment;
            target.texture = Some(self.stencil(PixelFormat::Stencil8, 1));
            target.load_action = LoadAction::Clear;
            target.clear_stencil = CLEARED;
            target.store_action = StoreAction::Store;
        }
        let event = self.device.new_shared_event()?;
        let command_buffer = self.queue.command_buffer()?;
        command_buffer.encode_wait_for_event(EventRef::from(&event), 1)?;
        let mut encoder = command_buffer.render_command_encoder(&pass)?;
        if attached {
            encoder.set_stencil_reference_value(REFERENCE);
            encoder.set_depth_stencil_state(&self.states()[0])?;
        }
        encoder.execute_commands_in_buffer(&icb, 0..1)?;
        encoder.end_encoding()?;
        let read = self.blit_read(&command_buffer, &color, 4)?;
        command_buffer.commit()?;
        // After the commit: the state, and command 0 drawing with it.
        let descriptor = self.stepped_descriptor(attached);
        let state = self
            .device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default());
        let bound = (self.words(&FIRST), self.words(&[COLORS[0]; WIDTH as usize]));
        let width = self.words(&[WIDTH]);
        let drawn = state.and_then(|state| {
            let command = icb.indirect_render_command_at(0)?;
            command.set_render_pipeline_state(&state)?;
            for (index, buffer) in [(0, &bound.0), (1, &bound.1), (2, &width)] {
                command.set_fragment_buffer(Some(buffer), 0, index)?;
            }
            command.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)?;
            Ok(state)
        });
        event.set_signaled_value(1);
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        drop(drawn?);
        drop(icb);
        Ok((command_buffer, read))
    }
}

/// A face of `compare`, its failure and pass operations `fail` and `pass`, every mask bit set.
fn face(
    compare: CompareFunction,
    fail: StencilOperation,
    pass: StencilOperation,
) -> StencilDescriptor {
    let mut face = StencilDescriptor::default();
    face.stencil_compare_function = compare;
    face.stencil_failure_operation = fail;
    face.depth_stencil_pass_operation = pass;
    face
}

/// The pass over `Stencil8` at 1 and 4 samples, over `Depth32Float_Stencil8` at 4 and over a
/// texture made in `X32_Stencil8` at 1 and 4, cleared and loaded, and the memoryless pass, on the device as opened and with
/// `VK_EXT_shader_stencil_export` withheld: every sample holds Metal's stencil, and so does the
/// resolve; the memoryless pass's colour is Metal's. A pipeline the device does not
/// make (the shadow of more than one sample needs `shaderStorageImageMultisample`) is reported as
/// its refusal. Where the shadow ran, it is announced once with the split it makes (and the
/// memoryless stencil's backing); where it did not, never.
#[test]
fn a_stencil_output_tests_and_writes_the_stencil_as_metal() {
    for withheld in [false, true] {
        let cache = ScratchCache::new().unwrap();
        let messages = Messages::default();
        {
            let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
                if withheld {
                    options
                        .withheld_extensions
                        .insert(VkExtension::ExtShaderStencilExport);
                }
            })
            .unwrap();
            let shadow = !device.compile_device().profile().fragment().stencil_export;
            assert!(
                shadow || !withheld,
                "the extension was withheld, yet enabled"
            );
            let form = if shadow { "shadow" } else { "export" };
            let name = device.name().to_owned();
            // A pass the shadow splits runs in more than one instance, whose `DontCare` stores
            // no driver is measured to store as Metal's (`DriverFact::DontCareStoreIsMetals`).
            let dont_care_metals = !shadow
                && device
                    .properties()
                    .driver_facts
                    .contains(&DriverFact::DontCareStoreIsMetals);
            let system_memory = shared_in_system_memory(&device);
            let lane = Lane::new(device);
            if !lane.outputs_stencil() {
                continue;
            }
            let mut answers = Vec::new();
            let mut drew = false;
            // The kinds the drawn passes' state makings announce, and whether the combined
            // format's pass, which stores its depth `DontCare`, drew.
            let mut made = BTreeSet::new();
            let mut dont_care = false;
            for (format, samples) in [
                (PixelFormat::Stencil8, 1u32),
                (PixelFormat::Stencil8, 4),
                (COMBINED, 4),
                (PixelFormat::X32Stencil8, 1),
                (PixelFormat::X32Stencil8, 4),
            ] {
                for load in [LoadAction::Clear, LoadAction::Load] {
                    let at = format!("{form} {format:?} {samples}x {load:?}");
                    match lane.drawn(format, samples, load) {
                        Ok((stencil, resolved)) => {
                            let expected: Vec<u8> = METAL
                                .iter()
                                .flat_map(|&s| std::iter::repeat_n(s, samples as usize))
                                .collect();
                            assert_eq!(stencil, expected, "{at} on {name}: every sample");
                            if let Some(resolved) = resolved {
                                assert_eq!(resolved, METAL, "{at} on {name}: the resolve");
                            }
                            drew = true;
                            made.extend(lane.made(&lane.by_pixel, None, (format, samples), false));
                            dont_care |= format == COMBINED;
                            answers.push(format!("{at} Metal's"));
                        }
                        Err(Error::Unsupported(u)) => answers.push(format!("{at} refused, {u:?}")),
                        Err(error) => panic!("{at} on {name}: {error}"),
                    }
                }
            }
            let at = format!("{form} memoryless");
            let mut backed = false;
            match lane.memoryless() {
                Ok(color) => {
                    assert_eq!(color, METAL_COLOR, "{at} on {name}: the colour");
                    backed = shadow;
                    drew = true;
                    let target = (PixelFormat::Stencil8, 1);
                    let uint = Some(PixelFormat::R32Uint);
                    made.extend(lane.made(&lane.with_color, uint, target, false));
                    answers.push(format!("{at} Metal's"));
                }
                Err(Error::Unsupported(u)) => answers.push(format!("{at} refused, {u:?}")),
                Err(error) => panic!("{at} on {name}: {error}"),
            }
            // The passes draw from `Shared` buffers.
            let mut first = forced(&[
                (Degradation::DontCareStore, dont_care && !dont_care_metals),
                (Degradation::SharedInSystemMemory, system_memory),
            ]);
            first.extend(made);
            if shadow && drew {
                let mut kinds = vec![Degradation::ShaderSideStencil, Degradation::RenderPassSplit];
                if backed {
                    kinds.push(Degradation::MemorylessBacked);
                }
                assert_made_once(&messages, &kinds, &first);
            } else {
                let counts = assert_made_once(&messages, &[], &first);
                assert!(
                    !counts.contains_key(&Degradation::ShaderSideStencil),
                    "{form} on {name}: {counts:?}"
                );
            }
            eprintln!("stencil outputs on {name}: {}", answers.join("; "));
            drop(lane);
        }
        layer_reported_nothing_at_all(&messages);
    }
}

/// How [`Lane::executed`] reads the stencil back: a blit of one sample's texels, or the samples'
/// kernel.
enum Read {
    Blit(Buffer),
    Samples(runtime_samples::SampleReadback),
}

/// The executed pass (the module docs) at 1 and 4 samples, on the device as opened and with
/// `VK_EXT_shader_stencil_export` withheld: the commands' stencil is Metal's, as the encoder's
/// draws of their pipelines give it, and one execution of two output commands about a plain one
/// leaves the first draw's outputs. A pipeline the device does not make is reported as its
/// refusal. Where the shadow ran, it is announced once, as the encoder's.
#[test]
fn executed_commands_test_and_write_the_stencil_as_metal() {
    for withheld in [false, true] {
        let cache = ScratchCache::new().unwrap();
        let messages = Messages::default();
        {
            let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
                if withheld {
                    options
                        .withheld_extensions
                        .insert(VkExtension::ExtShaderStencilExport);
                }
            })
            .unwrap();
            let shadow = !device.compile_device().profile().fragment().stencil_export;
            assert!(
                shadow || !withheld,
                "the extension was withheld, yet enabled"
            );
            let form = if shadow { "shadow" } else { "export" };
            let name = device.name().to_owned();
            let system_memory = shared_in_system_memory(&device);
            let lane = Lane::new(device);
            if !lane.outputs_stencil() {
                continue;
            }
            let mut answers = Vec::new();
            let mut drew = false;
            let mut made = BTreeSet::new();
            for samples in [1u32, 4] {
                for (sequence, outputs) in [(true, METAL), (false, FIRST.map(|s| s as u8))] {
                    let what = if sequence {
                        "the three draws"
                    } else {
                        "one execution"
                    };
                    let at = format!("{form} executed, {what}, {samples}x");
                    match lane.executed(samples, sequence) {
                        Ok(stencil) => {
                            let expected: Vec<u8> = outputs
                                .iter()
                                .flat_map(|&s| std::iter::repeat_n(s, samples as usize))
                                .collect();
                            assert_eq!(stencil, expected, "{at} on {name}: every sample");
                            drew = true;
                            let target = (PixelFormat::Stencil8, samples);
                            made.extend(lane.made(&lane.by_pixel, None, target, true));
                            answers.push(format!("{at} Metal's"));
                        }
                        Err(Error::Unsupported(u)) => answers.push(format!("{at} refused, {u:?}")),
                        Err(error) => panic!("{at} on {name}: {error}"),
                    }
                }
            }
            // Through the shadow, each command draws in a run of its own (`render-pass-split`);
            // the commands and the passes' buffers are `Shared`.
            let kinds = match shadow && drew {
                true => vec![Degradation::ShaderSideStencil, Degradation::RenderPassSplit],
                false => Vec::new(),
            };
            let mut first = forced(&[(Degradation::SharedInSystemMemory, system_memory)]);
            first.extend(made);
            let counts = assert_made_once(&messages, &kinds, &first);
            assert_eq!(
                counts.contains_key(&Degradation::ShaderSideStencil),
                shadow && drew,
                "{form} on {name}: {counts:?}"
            );
            eprintln!("executed stencil outputs on {name}: {}", answers.join("; "));
            drop(lane);
        }
        layer_reported_nothing_at_all(&messages);
    }
}

/// The host step's pass over the colour alone (the module docs), on the device as opened and
/// with `VK_EXT_shader_stencil_export` withheld: a host step draws a state made after the commit
/// whose fragment function writes `[[stencil]]`, which reaches no attachment. Through the shadow
/// its plan writes the shadow's neutral, made at the state's creation on the program's thread:
/// every variant of the state shares it (a shadow's presence is the descriptor's stencil format,
/// its samples the descriptor's), so no variant a step draws asks for one nothing made. The draw
/// completes and writes Metal's colour, the first colour at every pixel (no stencil to test),
/// under the validation layer, announcing the host step once and no shadow.
#[test]
fn a_host_step_s_draw_whose_stencil_output_reaches_no_attachment_binds_the_neutral_made_before_it()
{
    for withheld in [false, true] {
        let cache = ScratchCache::new().unwrap();
        let messages = Messages::default();
        {
            let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
                if withheld {
                    options
                        .withheld_extensions
                        .insert(VkExtension::ExtShaderStencilExport);
                }
            })
            .unwrap();
            let shadow = !device.compile_device().profile().fragment().stencil_export;
            assert!(
                shadow || !withheld,
                "the extension was withheld, yet enabled"
            );
            let form = if shadow { "shadow" } else { "export" };
            let name = device.name().to_owned();
            let system_memory = shared_in_system_memory(&device);
            let lane = Lane::new(device);
            if !lane.outputs_stencil() {
                continue;
            }
            let (command_buffer, read) = lane.stepped(false).unwrap();
            assert_eq!(
                (command_buffer.status(), command_buffer.error()),
                (CommandBufferStatus::Completed, None),
                "{form} on {name}"
            );
            let color: Vec<u32> = read_buffer(&read, 0, 4 * u64::from(WIDTH))
                .unwrap()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| u32::from_le_bytes(*word))
                .collect();
            assert_eq!(color, [COLORS[0]; WIDTH as usize], "{form} on {name}");
            // The commands and the buffers are `Shared`.
            let descriptor = lane.stepped_descriptor(false);
            let mut first = forced(&[(Degradation::SharedInSystemMemory, system_memory)]);
            first.extend(made_render_kinds(&lane.device, &descriptor));
            let counts =
                assert_made_once(&messages, &super::indirect_host_steps::HOST_STEPS, &first);
            assert!(
                !counts.contains_key(&Degradation::ShaderSideStencil),
                "{form} on {name}: {counts:?}"
            );
            drop(lane);
        }
        layer_reported_nothing_at_all(&messages);
    }
}

/// The host step's pass beside a stencil (the module docs), on the device as opened and with
/// `VK_EXT_shader_stencil_export` withheld. Where the device exports the stencil, the command
/// draws in the execution's instance. Through the shadow, the command's draw needs a run of
/// another shape, whose render pass, framebuffer and shadow nothing made before the step, the
/// state having been made after the commit: the step makes them, inside the exit guard's section
/// (in a build with debug assertions, an entry point outside its table ends the process). Either
/// way the command buffer completes with the first colour at every pixel (the first state's
/// `Always` test), as Metal draws it, under the validation layer.
#[test]
fn a_host_step_s_run_of_another_shape_makes_its_objects_and_draws() {
    for withheld in [false, true] {
        let cache = ScratchCache::new().unwrap();
        let messages = Messages::default();
        {
            let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
                if withheld {
                    options
                        .withheld_extensions
                        .insert(VkExtension::ExtShaderStencilExport);
                }
            })
            .unwrap();
            let shadow = !device.compile_device().profile().fragment().stencil_export;
            assert!(
                shadow || !withheld,
                "the extension was withheld, yet enabled"
            );
            let form = if shadow { "shadow" } else { "export" };
            let name = device.name().to_owned();
            let system_memory = shared_in_system_memory(&device);
            let lane = Lane::new(device);
            if !lane.outputs_stencil() {
                continue;
            }
            let (command_buffer, read) = lane.stepped(true).unwrap();
            let (status, error) = (command_buffer.status(), command_buffer.error());
            assert_eq!(
                (status, error),
                (CommandBufferStatus::Completed, None),
                "{form} on {name}"
            );
            let color: Vec<u32> = read_buffer(&read, 0, 4 * u64::from(WIDTH))
                .unwrap()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| u32::from_le_bytes(*word))
                .collect();
            assert_eq!(color, [COLORS[0]; WIDTH as usize], "{form} on {name}");
            // The commands and the buffers are `Shared`; through the shadow, the run of another
            // shape splits the pass around the shadow's copies.
            let descriptor = lane.stepped_descriptor(true);
            let mut first = forced(&[
                (Degradation::SharedInSystemMemory, system_memory),
                (Degradation::ShaderSideStencil, shadow),
                (Degradation::RenderPassSplit, shadow),
            ]);
            first.extend(made_render_kinds(&lane.device, &descriptor));
            assert_made_once(&messages, &super::indirect_host_steps::HOST_STEPS, &first);
            drop(lane);
        }
        layer_reported_nothing_at_all(&messages);
    }
}
