//! A runtime image whose parts no copy writes reads zero there, or what a new texture of its
//! storage mode reads, never what its memory held: a released texture's texels, of another
//! process of a consumer whose processes share the device (a recycled image's, `recycle.rs`), on
//! the device the GPU cfg selects.
//!
//! - **A surface depth texture's stencil.** A `Private` 16x16 `Depth32Float_Stencil8` texture over
//!   a plane of a surface with planes holds its depth alone in the plane: no copy writes the
//!   stencil of its shadow (`Device::storage_over_buffer`) or of the twin its passes render into
//!   (`texture/surface_depth.rs`). Two such textures, each passed with its stencil cleared to
//!   0xA5 and stored, are released, and two more over other surfaces take their four parked
//!   images (`WorkCounts::object_churn.images_recycled`): a kernel's read of each one's
//!   `X32_Stencil8` view reads 0, as Metal's does, and a pass that loads its stencil `DontCare`,
//!   stores nothing and draws where the stencil is 0xA5 draws nothing (a driver's `DONT_CARE`
//!   load that leaves the image's texels, lavapipe's, reads the twin's; one the device clears
//!   reads 0 whatever the twin holds).
//! - **A tiled texture's later level.** A `Private` 16x16 `RGBX8Unorm` texture of two levels over a
//!   `Private` buffer (`Buffer::new_tiled_texture`), whose second level its copies taken at use do
//!   not make (`shadow::LaterLevels::Kept`: the runtime makes no mipmaps of an RGBX format), its
//!   second level written by a blit, is released, and one more over another buffer takes its
//!   parked image: a blit of that one's second level reads what a new texture's of the same
//!   descriptor reads.
//! - **Separate depth and stencil textures' scratch image.** A pass of a `Private`
//!   `Depth32Float` texture and a `Private` `Stencil8` one renders through a scratch image of the
//!   combined format (`depth_stencil_scratch.rs`). One pass clears the stencil to 0xA5 and stores
//!   it, its textures and scratch image released; a pass of two new textures takes the three
//!   parked images and loads its stencil `DontCare` with a clear value of 0, stores nothing and
//!   draws where the stencil is 0xA5: it draws nothing, where the device leaves that load to the
//!   driver's `DONT_CARE` (`clear_reach::UnstoredLoads`, lavapipe's, which leaves the image's
//!   texels) as where it clears it to the clear value, Metal's answer. Run under each
//!   `UnstoredDontCarePolicy`, so each device takes both forms.
//!
//! Printed (`UNWRITTEN-PARTS`): what each read found.

use super::blit_holds::{device, options, Scratch, WAIT};
use super::encoder_holds::library;
use crate::{
    BlitOption, Buffer, ClearColor, CommandBuffer, CommandBufferStatus, CommandQueue,
    ComputePipelineState, DepthStencilState, Device, IoSurfacePlane, LoadAction, Origin,
    PipelineOptions, RenderPassDescriptor, RenderPipelineState, ResourceOptions, Size, StoreAction,
    Texture, TextureDescriptor,
};
use metal2vulkan::device::UnstoredDontCarePolicy;
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{CompareFunction, PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use std::time::Instant;

/// Each texture's side.
const SIDE: u64 = 16;

/// The stencil the released textures' passes clear to.
const MARK: u32 = 0xA5;

/// Commit `command_buffer` and wait for it to complete.
fn commit(command_buffer: &CommandBuffer) {
    command_buffer.commit().unwrap();
    assert!(command_buffer.wait_until_completed_for(WAIT).is_ok());
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// A copy of `buffer`'s bytes, which no command uses now.
fn bytes(buffer: &Buffer) -> Vec<u8> {
    let length = buffer.length() as usize;
    // SAFETY: a `Shared` buffer's contents span its length, and no command uses them now.
    unsafe { std::slice::from_raw_parts(buffer.contents().unwrap().as_ptr(), length) }.to_vec()
}

/// Wait until the device parked `count` more images than `before`.
fn parked(device: &Device, before: u64, count: u64) {
    let until = Instant::now() + WAIT;
    loop {
        let now = device.work_counts().object_churn.images_parked;
        if now >= before + count {
            return;
        }
        assert!(
            Instant::now() < until,
            "{} of {count} images parked",
            now - before
        );
        std::thread::yield_now();
    }
}

/// The level-0 bytes of `level` of a texture of `SIDE`, four a texel, blitted out.
fn level_bytes(queue: &CommandQueue, texture: &Texture, level: u64) -> Vec<u8> {
    let side = SIDE >> level;
    let out = (queue.device())
        .new_buffer(side * side * 4, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        level,
        Origin::new(0, 0, 0),
        Size::new(side, side, 1),
        &out,
        0,
        side * 4,
        0,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    commit(&command_buffer);
    bytes(&out)
}

/// The queue, the kernel that reads a `uint` texture, the state that draws where the stencil is
/// [`MARK`] and the stencil state that tests it.
struct Tools {
    queue: CommandQueue,
    read: ComputePipelineState,
    draw: RenderPipelineState,
    /// [`draw`](Self::draw)'s state for separate `Depth32Float` and `Stencil8` textures.
    separate: RenderPipelineState,
    equal: DepthStencilState,
}

impl Tools {
    fn new(device: &Device) -> Tools {
        let scout = library(device, "engine/format_scout.ll");
        let function = |name: &str| scout.new_function_with_name(name).unwrap();
        let options = PipelineOptions::default();
        let read = function("scout_read_u");
        let read = device
            .new_compute_pipeline_state_with_function(&read, &options)
            .unwrap();
        let stage = |name: &str| StageSource::Function(function(name).function_descriptor());
        let mut descriptor = RenderPipelineDescriptor::new(stage("scout_vertex"));
        descriptor.fragment = Some(stage("scout_fragment_f"));
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        descriptor.depth_format = Some(PixelFormat::Depth32FloatStencil8);
        descriptor.stencil_format = Some(PixelFormat::Depth32FloatStencil8);
        let draw = device
            .new_render_pipeline_state(&descriptor, &options)
            .unwrap();
        descriptor.depth_format = Some(PixelFormat::Depth32Float);
        descriptor.stencil_format = Some(PixelFormat::Stencil8);
        let separate = device
            .new_render_pipeline_state(&descriptor, &options)
            .unwrap();
        let mut equal = DepthStencilDescriptor::default();
        for face in [&mut equal.front_face_stencil, &mut equal.back_face_stencil] {
            face.stencil_compare_function = CompareFunction::Equal;
        }
        Tools {
            queue: device.new_command_queue().unwrap(),
            read,
            draw,
            separate,
            equal: device.new_depth_stencil_state(&equal).unwrap(),
        }
    }

    /// A new `Private` `Depth32Float_Stencil8` texture over plane 0 of a new surface, rows packed.
    fn surface_depth(&self) -> Texture {
        let device = self.queue.device();
        let mut descriptor =
            TextureDescriptor::texture_2d(PixelFormat::Depth32FloatStencil8, SIDE, SIDE, false);
        descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
        descriptor.resource_options = ResourceOptions::private();
        let surface = device.new_buffer(16384, ResourceOptions::shared()).unwrap();
        let plane = IoSurfacePlane::new(0, SIDE * 4, true);
        (surface.new_texture_with_iosurface_plane(&descriptor, plane)).unwrap()
    }

    /// A pass of no draw call over `texture` that clears its depth to 0.5 and its stencil to
    /// [`MARK`] and stores both.
    fn mark(&self, texture: &Texture) {
        let mut pass = RenderPassDescriptor::new();
        let depth = &mut pass.depth_attachment;
        depth.texture = Some(texture.clone());
        depth.load_action = LoadAction::Clear;
        depth.store_action = StoreAction::Store;
        depth.clear_depth = 0.5;
        let stencil = &mut pass.stencil_attachment;
        stencil.texture = Some(texture.clone());
        stencil.load_action = LoadAction::Clear;
        stencil.store_action = StoreAction::Store;
        stencil.clear_stencil = MARK;
        let command_buffer = self.queue.command_buffer().unwrap();
        (command_buffer.render_command_encoder(&pass).unwrap())
            .end_encoding()
            .unwrap();
        commit(&command_buffer);
    }

    /// The stencil of each texel of `texture`, read by a kernel through its `X32_Stencil8` view.
    fn stencils(&self, texture: &Texture) -> Vec<u32> {
        let view = texture.new_texture_view(PixelFormat::X32Stencil8).unwrap();
        let out = (self.queue.device())
            .new_buffer(SIDE * SIDE * 16, ResourceOptions::shared())
            .unwrap();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&self.read).unwrap();
        encoder.set_texture(Some(&view), 0).unwrap();
        encoder.set_buffer(Some(&out), 0, 0).unwrap();
        (encoder.dispatch_threads(Size::new(SIDE, SIDE, 1), Size::new(8, 8, 1))).unwrap();
        encoder.end_encoding().unwrap();
        commit(&command_buffer);
        bytes(&out)
            .chunks(16)
            .map(|texel| u32::from_le_bytes(texel[..4].try_into().unwrap()))
            .collect()
    }

    /// What a pass into `color`, cleared to zero and stored, beside `texture` loaded and stored
    /// `DontCare`, draws over the whole target where the stencil is [`MARK`].
    fn drawn(&self, texture: &Texture, color: &Texture) -> Vec<u8> {
        let mut pass = RenderPassDescriptor::new();
        let target = &mut pass.color_attachments[0];
        target.texture = Some(color.clone());
        target.load_action = LoadAction::Clear;
        target.store_action = StoreAction::Store;
        target.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        let depth = &mut pass.depth_attachment;
        depth.texture = Some(texture.clone());
        depth.load_action = LoadAction::DontCare;
        depth.store_action = StoreAction::DontCare;
        let stencil = &mut pass.stencil_attachment;
        stencil.texture = Some(texture.clone());
        stencil.load_action = LoadAction::DontCare;
        stencil.store_action = StoreAction::DontCare;
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&self.draw).unwrap();
        encoder.set_depth_stencil_state(&self.equal).unwrap();
        encoder.set_stencil_reference_value(MARK);
        (encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)).unwrap();
        encoder.end_encoding().unwrap();
        commit(&command_buffer);
        level_bytes(&self.queue, color, 0)
    }
}

impl Tools {
    /// A new `Private` texture of `format` that renders, `SIDE` square.
    fn attachment(&self, format: PixelFormat) -> Texture {
        let mut descriptor = TextureDescriptor::texture_2d(format, SIDE, SIDE, false);
        descriptor.usage = TextureUsage::RENDER_TARGET;
        descriptor.resource_options = ResourceOptions::private();
        self.queue.device().new_texture(&descriptor).unwrap()
    }

    /// What a pass into `color`, cleared to zero and stored, beside `depth` and `stencil` loaded
    /// with `load` and stored with `store`, draws over the whole target where the stencil is
    /// [`MARK`]: their clear values 0.5 and [`MARK`] where `load` is `Clear`, else 0.5 and 0, so a
    /// `DontCare` load the device clears to the clear value (Metal's answer) is told apart from
    /// one that reads a released texture's [`MARK`].
    fn separate(
        &self,
        (depth, stencil): (&Texture, &Texture),
        color: &Texture,
        (load, store): (LoadAction, StoreAction),
    ) -> Vec<u8> {
        let mut pass = RenderPassDescriptor::new();
        let target = &mut pass.color_attachments[0];
        target.texture = Some(color.clone());
        target.load_action = LoadAction::Clear;
        target.store_action = StoreAction::Store;
        target.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        let attachment = &mut pass.depth_attachment;
        attachment.texture = Some(depth.clone());
        (attachment.load_action, attachment.store_action) = (load, store);
        attachment.clear_depth = 0.5;
        let attachment = &mut pass.stencil_attachment;
        attachment.texture = Some(stencil.clone());
        (attachment.load_action, attachment.store_action) = (load, store);
        attachment.clear_stencil = match load {
            LoadAction::Clear => MARK,
            _ => 0,
        };
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&self.separate).unwrap();
        encoder.set_depth_stencil_state(&self.equal).unwrap();
        encoder.set_stencil_reference_value(MARK);
        (encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)).unwrap();
        encoder.end_encoding().unwrap();
        commit(&command_buffer);
        level_bytes(&self.queue, color, 0)
    }
}

/// What `values` hold, in words.
fn found<T: PartialEq + Copy + Default>(values: &[T], mark: T) -> &'static str {
    if values.iter().all(|&v| v == T::default()) {
        "zero"
    } else if values.contains(&mark) {
        "the released texture's mark"
    } else {
        "other values"
    }
}

#[test]
fn a_surface_depth_texture_s_stencil_reads_zero_never_a_released_texture_s() {
    let scratch = Scratch::new("unwritten-stencil");
    let device = device(&scratch.0);
    let tools = Tools::new(&device);
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::shared();
    let colors: Vec<Texture> = (0..2)
        .map(|_| device.new_texture(&descriptor).unwrap())
        .collect();
    // Two textures whose shadows and twins are parked, each twin's stencil the mark.
    let before = device.work_counts().object_churn;
    for _ in 0..2 {
        let released = tools.surface_depth();
        tools.mark(&released);
    }
    parked(&device, before.images_parked, 4);
    let taken = device.work_counts().object_churn.images_recycled;
    let mut wrong = Vec::new();
    for (index, color) in colors.iter().enumerate() {
        let texture = tools.surface_depth();
        let read = tools.stencils(&texture);
        let drawn = tools.drawn(&texture, color);
        let (read, drawn) = (found(&read, MARK), found(&drawn, 0x40));
        println!(
            "UNWRITTEN-PARTS surface stencil {index} on {}: kernel read {read}, draw {drawn}",
            device.name()
        );
        if read != "zero" || drawn != "zero" {
            wrong.push(format!("{index}: kernel read {read}, draw {drawn}"));
        }
    }
    let taken = device.work_counts().object_churn.images_recycled - taken;
    assert_eq!(taken, 4, "the two textures took the four parked images");
    assert!(wrong.is_empty(), "{wrong:?}");
}

#[test]
fn a_tiled_texture_s_later_level_reads_as_a_new_texture_s_never_a_released_texture_s() {
    let scratch = Scratch::new("unwritten-levels");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBX8Unorm, SIDE, SIDE, false);
    descriptor.mipmap_level_count = 2;
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    let tiled = || {
        let bytes = device
            .new_buffer(SIDE * SIDE * 4, ResourceOptions::private())
            .unwrap();
        bytes.new_tiled_texture(&descriptor, 0, SIDE * 4).unwrap()
    };
    let fresh = level_bytes(&queue, &device.new_texture(&descriptor).unwrap(), 1);
    // A texture whose second level a blit writes, released.
    let side = SIDE / 2;
    let mark = vec![0x5Au8; (side * side * 4) as usize];
    assert_ne!(fresh, mark, "a new texture reads the mark");
    let source = device
        .new_buffer(side * side * 4, ResourceOptions::shared())
        .unwrap();
    // SAFETY: the new buffer's mapped bytes, which no command uses yet.
    unsafe {
        let at = source.contents().unwrap().as_ptr();
        std::ptr::copy_nonoverlapping(mark.as_ptr(), at, mark.len());
    }
    let before = device.work_counts().object_churn;
    {
        let released = tiled();
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_texture(
            &source,
            0,
            side * 4,
            0,
            Size::new(side, side, 1),
            &released,
            0,
            1,
            Origin::new(0, 0, 0),
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        commit(&command_buffer);
        assert_eq!(
            level_bytes(&queue, &released, 1),
            mark,
            "the blit wrote level 1"
        );
    }
    parked(&device, before.images_parked, 1);
    let taken = device.work_counts().object_churn.images_recycled;
    let texture = tiled();
    let taken = device.work_counts().object_churn.images_recycled - taken;
    let read = level_bytes(&queue, &texture, 1);
    let what = match (read == fresh, read == mark) {
        (true, _) => "a new texture's",
        (false, true) => "the released texture's",
        (false, false) => "other bytes",
    };
    println!(
        "UNWRITTEN-PARTS tiled level 1 on {}: {what} (a new texture's {})",
        device.name(),
        found(&fresh, 0x5A)
    );
    assert_eq!(taken, 1, "the texture took the parked image");
    assert_eq!(what, "a new texture's");
}

#[test]
fn separate_depth_and_stencil_textures_read_no_released_texture_s_stencil() {
    for policy in [
        UnstoredDontCarePolicy::Driver,
        UnstoredDontCarePolicy::Cleared,
    ] {
        separate_released(policy);
    }
}

/// [`separate_depth_and_stencil_textures_read_no_released_texture_s_stencil`] on a device of
/// `policy`.
fn separate_released(policy: UnstoredDontCarePolicy) {
    let scratch = Scratch::new("unwritten-separate");
    let mut options = options(&scratch.0);
    options.policy.unstored_dont_care_loads = policy;
    let device = Device::system_default_with(&options).unwrap();
    let tools = Tools::new(&device);
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::shared();
    let color = device.new_texture(&descriptor).unwrap();
    let textures = || {
        let depth = tools.attachment(PixelFormat::Depth32Float);
        (depth, tools.attachment(PixelFormat::Stencil8))
    };
    // A pass whose scratch image's stencil is the mark, it and its textures released.
    let before = device.work_counts().object_churn;
    let (depth, stencil) = textures();
    let marked = (LoadAction::Clear, StoreAction::Store);
    let drawn = tools.separate((&depth, &stencil), &color, marked);
    assert_ne!(found(&drawn, 0x40), "zero", "the marking pass drew nothing");
    drop((depth, stencil));
    parked(&device, before.images_parked, 3);
    let taken = device.work_counts().object_churn.images_recycled;
    let (depth, stencil) = textures();
    let unstored = (LoadAction::DontCare, StoreAction::DontCare);
    let drawn = tools.separate((&depth, &stencil), &color, unstored);
    let taken = device.work_counts().object_churn.images_recycled - taken;
    let drawn = found(&drawn, 0x40);
    println!(
        "UNWRITTEN-PARTS separate stencil on {} under {policy:?}: images taken {taken}, draw \
         {drawn}",
        device.name()
    );
    assert_eq!(
        taken, 3,
        "{policy:?}: the two textures and the scratch took the parked images"
    );
    assert_eq!(drawn, "zero", "{policy:?}");
}
