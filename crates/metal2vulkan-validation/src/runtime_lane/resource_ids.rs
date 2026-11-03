//! Device tests of resource IDs on the device the cfg selects, under the validation layer: textures
//! and sampler states placed at an ID (`resourceIndex` with `forceResourceIndex`, a runtime
//! extension), reservations and their limits, sampler-state uniquing (on an M3 Max and an M4), and
//! what a kernel reading a texture's ID from its argument buffer reaches: the texture holding the
//! ID as its command buffer runs, whichever texture the encoder named, or none, as Metal's did (the
//! M4 probe, residency part C), a heap's and a sparse heap's alike, and the neutral
//! texture once that holder went. The kernel samples a 2-by-2 `RGBA32Uint` texture through a
//! sampler held beside it, so its output is the texels it reached (a float twin of the case reads
//! `RGBA16Float` linearly). Each degradation site of the kinds (the device
//! heap's: [`UNVIEWABLE`], [`UNNAMED_SHADOWS`], [`PAST_CAPACITY`], [`PAST_LAYOUT_SUPPORT`],
//! [`OWN_COPIES`]) is taken on a device of its own and announced once.

use super::buffer_texture_blits::linear_rows_differ;
use super::degrade_notices::{
    assert_made_once, assert_once, forced, refused_sets, shared_in_system_memory,
};
use super::host_memory::{no_copy, not_importable, Mapping, GUEST_PAGE};
use super::sparse_textures::{lacks_sparse, SparseFacts};
use super::{
    bound_buffers, buffer_with, case_making_kinds, creation_checked, encode_case,
    layer_reported_nothing_at_all, made_lane_device, pipeline_state, public_case, read_buffer,
    texture_descriptor, validated_device_with, CaseResources, Messages, PublicCase, ScratchCache,
};
use crate::candidate::engine::resources::argument_sampler_descriptor;
use crate::candidate::engine::ComputeOutput;
use crate::candidate::unsupported_set;
use crate::gpu::TEST_DEVICE;
use crate::literal::LiteralTexture;
use ash::vk;
use metal2vulkan::bind::indirect_commands::RECORDS_ID_LIMIT;
use metal2vulkan::bind::{SamplerDescriptor, TileMapping};
use metal2vulkan::device::DriverFact;
use metal2vulkan::device::VkFeature;
use metal2vulkan::error::{Feature, RuntimeDriverDefect};
use metal2vulkan::metal::format::direct_format;
use metal2vulkan::metal::{
    CompareFunction, IndirectCommandType, PixelFormat, RenderStages, ResourceUsage,
    SamplerMinMagFilter, TextureType, TextureUsage,
};
use metal2vulkan_runtime::{
    BlitOption, Buffer, BufferContents, BufferDescriptor, CommandBuffer, CommandBufferStatus,
    CommandQueue, ComputePipelineState, Deallocator, DebugMessage, Degradation, Device,
    DeviceOptions, Error, EventRef, ExecutedPipeline, Fence, Heap, HeapDescriptor, HeapType,
    HostMemory, IdPlacement, IndirectCommandBuffer, IndirectCommandBufferDescriptor, Invalid,
    IoSurfacePlane, KeepAlive, MessageSeverity, MessageSink, Origin, Problem, Region,
    RenderPassDescriptor, ResourceId, ResourceIdKind, ResourceOptions, SamplerState,
    SamplerStateOptions, SharedEvent, Size, Texture, TextureDescriptor, TextureSwizzleChannels,
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The kind [`a_sampler_state_at_any_id_is_listed`] takes where more sampler states live than
/// the device's or the pipeline's sampler arrays hold (and expects absent elsewhere).
pub(super) const UNLISTED: [Degradation; 1] = [Degradation::UnlistedArgumentSampler];

/// What a dispatch sampling through an argument buffer announces while a sampler state of pixel
/// coordinates lives.
const PIXEL_ARGUMENT: [Degradation; 1] = [Degradation::ArgumentPixelSamplerScaling];

/// The kind [`a_texture_its_heap_array_cannot_view_is_announced_once`] takes.
pub(super) const UNVIEWABLE: [Degradation; 1] = [Degradation::UnnamedTextureUnviewable];

/// The kind [`a_shadowed_view_reached_through_its_id_is_announced_once`] takes where the view is
/// a shadow (and expects absent elsewhere).
pub(super) const UNNAMED_SHADOWS: [Degradation; 1] = [Degradation::UnnamedShadowedTexture];

/// The kind [`a_texture_at_the_highest_id_is_reached_where_the_arrays_hold_it`] takes where the
/// pipeline's heap arrays end below the highest ID Metal hands out (and expects absent
/// elsewhere).
pub(super) const PAST_CAPACITY: [Degradation; 1] = [Degradation::UnnamedTexturesPastCapacity];

/// The kind [`a_heap_set_its_driver_refuses_is_announced_once_where_it_runs`] takes where the
/// device's driver refuses the pipeline's heap set through its support query (and expects absent
/// elsewhere).
pub(super) const PAST_LAYOUT_SUPPORT: [Degradation; 1] = [Degradation::SetPastLayoutSupport];

/// The kind [`a_device_without_pending_writes_binds_a_copy_per_command_buffer`] takes.
pub(super) const OWN_COPIES: [Degradation; 1] = [Degradation::HeapSetCopied];

/// The kind [`a_no_copy_buffer_s_deallocator_runs_at_its_release_given_a_keep_alive`] takes where
/// a texture's image over the buffer's pages, which a submitted command buffer may read, holds
/// their import and the buffer has no keep-alive; and the tests of a released buffer whose pages a
/// table of the live buffers a command buffer holds lists (`rebind_reach.rs`,
/// `indirect_host_steps.rs`); and the test of a buffer released behind work that reads device
/// addresses, where the device's driver has `FreedAddressMemoryFailsPendingWork`
/// (`freed_memory.rs`).
pub(super) const DELAYED_DEALLOCATORS: [Degradation; 1] = [Degradation::HostDeallocatorDelayed];

// Each test asserts exactly the kinds of its sites it took, beside those its steps force on the
// device ([`Sampling::forced`], [`heap_texture_kinds`]); `id_kinds`, which let any other kind
// pass, went.

/// The kinds a colour texture of `descriptor` made in a heap whose memory its optimal image takes
/// forces as the device's first is made: `heap-texture-larger-than-metal` where the device
/// answers its size past Metal's modelled answer (its image needs the bytes), and, on a driver
/// whose compressed images hide writes through an alias
/// (`DriverFact::CompressedImagesHideAliasedWrites`), the route its image takes, in
/// `texture/heap.rs`'s order: host-transfer usage where the driver keeps such images' bytes
/// (`DriverFact::HostTransferImagesKeepTheirBytes`) and the format takes the usage at optimal
/// tiling (the runtime enables `hostImageCopy` wherever the device has it), else a linear image
/// where the format's linear tiling samples, else the optimal image. The size query announces the
/// first kind as the heap's texture would. And `heap-texture-untiled-layout` where its image
/// lies in the heap's memory in the driver's layout, not Metal's: on any driver but one whose
/// single-sample colour images bound to memory lie as Metal lays a heap texture out
/// (`DriverFact::ColorImagesLieAsMetalHeapTextures`), or whose route is another than the
/// optimal image bound there (`texture/heap.rs`, `Device::device_heap_layout`).
///
/// A texture the device made as a shadow of its elements in Metal's tiles in the heap's bytes
/// (`tiled`: it moved `WorkCounts::heap_textures_tiled`, the route `texture/heap.rs` takes where
/// the image it would bind there is itself a shadow or keeps a layout Metal does not, and the
/// device moves the tiles) lies as Metal lays it out at Metal's answer: it announces that shadow
/// (`shadowed-texture`) and no route or layout of an image bound there. The lavapipe of the Linux
/// gate's container takes that route for a `Private` heap's texture.
fn heap_texture_kinds(
    device: &Device,
    descriptor: &TextureDescriptor,
    tiled: bool,
) -> BTreeSet<Degradation> {
    let answer = device.heap_texture_size_and_align(descriptor).unwrap();
    let metal = device
        .metal_heap_texture_size_and_align(descriptor)
        .unwrap();
    if tiled {
        return forced(&[
            (
                Degradation::HeapTextureLargerThanMetal,
                metal.is_some_and(|metal| answer != metal),
            ),
            (Degradation::ShadowedTexture, true),
        ]);
    }
    let facts = &device.properties().driver_facts;
    let hides = facts.contains(&DriverFact::CompressedImagesHideAliasedWrites);
    let format = direct_format(descriptor.pixel_format).expect("a direct format");
    let format = vk::Format::from_raw(format.as_raw());
    let mut three = vk::FormatProperties3::default();
    let mut properties = vk::FormatProperties2::default().push_next(&mut three);
    // SAFETY: an instance-level query of the live device's physical device; the chain lives for
    // the call.
    unsafe {
        let raw = device.raw();
        (raw.instance).get_physical_device_format_properties2(
            raw.physical_device,
            format,
            &mut properties,
        );
    }
    let (optimal, linear) = (three.optimal_tiling_features, three.linear_tiling_features);
    let transfer = facts.contains(&DriverFact::HostTransferImagesKeepTheirBytes)
        && optimal.contains(vk::FormatFeatureFlags2::HOST_IMAGE_TRANSFER_EXT);
    let linear = linear.contains(vk::FormatFeatureFlags2::SAMPLED_IMAGE);
    forced(&[
        (
            Degradation::HeapTextureLargerThanMetal,
            metal.is_some_and(|metal| answer != metal),
        ),
        (Degradation::HeapTextureHostTransfer, hides && transfer),
        (Degradation::HeapTextureLinear, hides && !transfer && linear),
        (
            Degradation::HeapTextureHidesAliasedWrites,
            hides && !transfer && !linear,
        ),
        (
            Degradation::HeapTextureUntiledLayout,
            hides || !facts.contains(&DriverFact::ColorImagesLieAsMetalHeapTextures),
        ),
    ])
}

/// The kinds a texture over a buffer of [`RGBA8Uint`](PixelFormat::RGBA8Uint) rows
/// [`ROW_BYTES`] apart, [`ROW_TEXELS`] wide and 2 high, forces where no linear image of its device
/// holds its texels alone ([`linear_rows_differ`]): its shadow (`shadowed-texture`), and the heap
/// set holding it as its ID's holder (`UNNAMED_SHADOWS`).
fn row_shadow_kinds(device: &Device) -> Vec<Degradation> {
    let rows = [ROW_TEXELS as u32, 2];
    match linear_rows_differ(device, PixelFormat::RGBA8Uint, rows, 4, ROW_BYTES) {
        true => [Degradation::ShadowedTexture, UNNAMED_SHADOWS[0]].into(),
        false => Vec::new(),
    }
}

/// The longest a command buffer of these tests may take.
const WAIT: Duration = Duration::from_secs(60);

/// Samples an unsigned texture through a nearest sampler, both held in its argument buffer.
const HANDLE_SAMPLER: &str =
    "sample-an-unsigned-texture-through-a-sampler-held-beside-it-in-an-argument-buffer";

/// The first ID Metal ended the process for (probes 3 and 4).
const ID_LIMIT: u64 = 16381;

/// The diagnostic code of a reservation made too late.
const LATE_RESERVATION: &str = "metal2vulkan-late-reservation";

/// Run `test` on a validated device of the lane reporting to its messages, then fail on any
/// validation message.
fn on_device(test: impl FnOnce(&Device, &Messages)) {
    on_device_with(|_| {}, test);
}

/// [`on_device`] with the options `adjust` changes.
fn on_device_with(adjust: impl FnOnce(&mut DeviceOptions), test: impl FnOnce(&Device, &Messages)) {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), adjust).unwrap();
        test(&device, &messages);
    }
    layer_reported_nothing_at_all(&messages);
}

/// [`HANDLE_SAMPLER`] (or another case of its shape: a 2-by-2 texture and a sampler state in its
/// argument buffer, a buffer output) on a device: its pipeline state, its resources (the texture
/// holding the case's texels, and the sampler state), and its argument-buffer texture's literal
/// and usage.
pub(super) struct Sampling {
    case: PublicCase,
    pipeline: ComputePipelineState,
    queue: CommandQueue,
    base: CaseResources,
    key: (u32, u32),
    literal: LiteralTexture,
    usage: TextureUsage,
}

impl Sampling {
    pub(super) fn new(device: &Device) -> Sampling {
        Sampling::of(device, HANDLE_SAMPLER)
    }

    /// The case `name` on `device`.
    fn of(device: &Device, name: &str) -> Sampling {
        let case = public_case(name).unwrap();
        let pipeline = pipeline_state(device, &case).unwrap();
        let base = CaseResources::new(device, &case).unwrap();
        let [(&key, literal)] = case
            .compute
            .argument_buffer_textures
            .iter()
            .collect::<Vec<_>>()[..]
        else {
            panic!("{name}: one argument-buffer texture");
        };
        assert_eq!(literal.dimensions, [2, 2, 1], "{name}: a 2-by-2 texture");
        let literal = literal.clone();
        let usage = case.argument_texture_usage[&key];
        Sampling {
            queue: device.new_command_queue().unwrap(),
            case,
            pipeline,
            base,
            key,
            literal,
            usage,
        }
    }

    /// The kinds the lane's steps force on `device` beside a site's own where it has not
    /// announced them: the making of the case's pipeline state ([`case_making_kinds`]), and its
    /// `Shared` buffers' system memory.
    fn forced(&self, device: &Device) -> BTreeSet<Degradation> {
        let mut kinds = case_making_kinds(device, &self.case);
        if shared_in_system_memory(device) {
            kinds.insert(Degradation::SharedInSystemMemory);
        }
        kinds
    }

    /// The notices since the last check are exactly `kinds`, each once, beside the
    /// [`forced`](Self::forced) kinds `device` has not announced.
    fn announced(&self, device: &Device, messages: &Messages, kinds: &[Degradation]) {
        assert_made_once(messages, kinds, &self.forced(device));
    }

    /// The texture holding the case's texels, at a plain ID.
    fn texture(&self) -> &Texture {
        &self.base.argument_textures[&self.key]
    }

    /// A texture as the case's, of `pixel_format`, at `placement`, holding `bytes`.
    fn made(
        &self,
        device: &Device,
        pixel_format: PixelFormat,
        placement: IdPlacement,
        bytes: &[u8],
    ) -> Texture {
        let mut descriptor = texture_descriptor(&self.literal, self.usage).unwrap();
        descriptor.pixel_format = pixel_format;
        descriptor.placement = placement;
        let texture = device.new_texture(&descriptor).unwrap();
        let row = bytes.len() as u64 / 2;
        texture
            .replace_region(Region::new_2d(0, 0, 2, 2), 0, 0, bytes, row, 2 * row)
            .unwrap();
        texture
    }

    /// A texture as the case's at `placement`, holding the case's texels inverted.
    fn inverted(&self, device: &Device, placement: IdPlacement) -> Texture {
        let bytes: Vec<u8> = self.literal.bytes.iter().map(|byte| !byte).collect();
        self.made(device, PixelFormat::RGBA32Uint, placement, &bytes)
    }

    /// A command buffer encoding the case's dispatch, its argument buffer holding `named`'s
    /// ID and the encoder making `resident` resident (none when `None`), after waiting for
    /// `wait`'s value 1 when given; not committed. The output buffer comes with it.
    fn encode(
        &self,
        device: &Device,
        named: &Texture,
        resident: Option<&Texture>,
        wait: Option<EventRef<'_>>,
    ) -> (CommandBuffer, BTreeMap<u32, Buffer>) {
        let reaching = CaseResources {
            argument_textures: resident
                .map(|t| (self.key, t.clone()))
                .into_iter()
                .collect(),
            ..CaseResources::default()
        };
        self.encode_reaching(device, named, reaching, wait)
    }

    /// [`encode`](Self::encode), the encoder binding and making resident the textures of
    /// `reaching` (its bound and argument-buffer textures).
    fn encode_reaching(
        &self,
        device: &Device,
        named: &Texture,
        reaching: CaseResources,
        wait: Option<EventRef<'_>>,
    ) -> (CommandBuffer, BTreeMap<u32, Buffer>) {
        let naming = CaseResources {
            argument_textures: BTreeMap::from([(self.key, named.clone())]),
            argument_samplers: self.base.argument_samplers.clone(),
            ..CaseResources::default()
        };
        let buffers = bound_buffers(device, &self.case.compute, &naming).unwrap();
        let command_buffer = self.queue.command_buffer().unwrap();
        if let Some(event) = wait {
            command_buffer.encode_wait_for_event(event, 1).unwrap();
        }
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encode_case(
            &mut encoder,
            &self.pipeline,
            &self.case.compute,
            &buffers,
            &reaching,
        )
        .unwrap();
        encoder.end_encoding().unwrap();
        (command_buffer, buffers)
    }

    /// The output of `command_buffer`, committed, over `buffers`, once it completed.
    fn output(&self, command_buffer: &CommandBuffer, buffers: &BTreeMap<u32, Buffer>) -> Vec<u8> {
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(
            command_buffer.status(),
            CommandBufferStatus::Completed,
            "{:?}",
            command_buffer.error()
        );
        let ComputeOutput::Buffer {
            binding,
            offset,
            length,
        } = self.case.compute.output
        else {
            panic!("{}: a buffer output", self.case.case.name);
        };
        read_buffer(&buffers[&binding], offset, length).unwrap()
    }

    /// The output of one command buffer as [`encode`](Self::encode) makes it, committed at once.
    fn run(&self, device: &Device, named: &Texture, resident: Option<&Texture>) -> Vec<u8> {
        let (command_buffer, buffers) = self.encode(device, named, resident, None);
        command_buffer.commit().unwrap();
        self.output(&command_buffer, &buffers)
    }

    /// The output of one command buffer whose argument buffer holds `named`'s ID, its encoder
    /// making `heap` resident (`use_heap`) and nothing else, committed at once.
    fn run_using_heap(&self, device: &Device, named: &Texture, heap: &Heap) -> Vec<u8> {
        let (command_buffer, buffers) = self.encode_using_heap(device, named, heap);
        command_buffer.commit().unwrap();
        self.output(&command_buffer, &buffers)
    }

    /// [`run_using_heap`](Self::run_using_heap)'s command buffer and buffers, not committed.
    fn encode_using_heap(
        &self,
        device: &Device,
        named: &Texture,
        heap: &Heap,
    ) -> (CommandBuffer, BTreeMap<u32, Buffer>) {
        let naming = CaseResources {
            argument_textures: BTreeMap::from([(self.key, named.clone())]),
            argument_samplers: self.base.argument_samplers.clone(),
            ..CaseResources::default()
        };
        let buffers = bound_buffers(device, &self.case.compute, &naming).unwrap();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.use_heap(heap).unwrap();
        let nothing = CaseResources::default();
        encode_case(
            &mut encoder,
            &self.pipeline,
            &self.case.compute,
            &buffers,
            &nothing,
        )
        .unwrap();
        encoder.end_encoding().unwrap();
        (command_buffer, buffers)
    }

    /// The output of one command buffer whose encoder makes `heap` resident (`use_heap`) and
    /// nothing else, and only then gets `named` from `make` (which makes it, in the heap or
    /// not, and fills it) and dispatches the case with `named`'s ID in its argument buffer,
    /// committed at once.
    fn run_making_after_use_heap(
        &self,
        device: &Device,
        heap: &Heap,
        make: impl FnOnce() -> Texture,
    ) -> Vec<u8> {
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.use_heap(heap).unwrap();
        let named = make();
        let naming = CaseResources {
            argument_textures: BTreeMap::from([(self.key, named)]),
            argument_samplers: self.base.argument_samplers.clone(),
            ..CaseResources::default()
        };
        let buffers = bound_buffers(device, &self.case.compute, &naming).unwrap();
        let nothing = CaseResources::default();
        encode_case(
            &mut encoder,
            &self.pipeline,
            &self.case.compute,
            &buffers,
            &nothing,
        )
        .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        self.output(&command_buffer, &buffers)
    }

    /// A blit in `command_buffer` writing the case's texels into `texture`, not committed.
    fn upload(&self, device: &Device, command_buffer: &CommandBuffer, texture: &Texture) {
        let bytes = &self.literal.bytes;
        let source = device
            .new_buffer(bytes.len() as u64, ResourceOptions::shared())
            .unwrap();
        let contents = source.contents().unwrap();
        // SAFETY: the buffer holds `bytes.len()` bytes, and no command uses them yet.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
        let row = bytes.len() as u64 / 2;
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_texture(
            &source,
            0,
            row,
            2 * row,
            Size::new(2, 2, 1),
            texture,
            0,
            0,
            Origin::new(0, 0, 0),
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
    }

    /// [`run`](Self::run) with no texture resident and `bound` bound at texture index 0, which
    /// the kernel does not read.
    fn run_bound(&self, device: &Device, named: &Texture, bound: &Texture) -> Vec<u8> {
        let reaching = CaseResources {
            textures: BTreeMap::from([(0, bound.clone())]),
            ..CaseResources::default()
        };
        let (command_buffer, buffers) = self.encode_reaching(device, named, reaching, None);
        command_buffer.commit().unwrap();
        self.output(&command_buffer, &buffers)
    }
}

/// What a kernel reads through the case's texture's ID: its texels (Metal's output), the
/// inverted texels of a texture of their own, and the neutral texture's, read through an ID whose
/// holder went though the texture it took the ID from lives (the slot keeps no earlier holder).
struct Reads {
    case: Vec<u8>,
    inverted: Vec<u8>,
    neutral: Vec<u8>,
}

impl Reads {
    fn of(device: &Device, sampling: &Sampling) -> Reads {
        let texture = sampling.texture();
        let case = sampling.run(device, texture, Some(texture));
        assert_eq!(case, sampling.case.metal_output, "the case's own texture");
        let own = sampling.inverted(device, IdPlacement::Any);
        let inverted = sampling.run(device, &own, Some(&own));
        let taken = sampling.inverted(device, IdPlacement::Any);
        let holder = sampling.inverted(device, IdPlacement::At(taken.gpu_resource_id()));
        drop(holder);
        let neutral = sampling.run(device, &taken, Some(&taken));
        assert!(case != inverted && case != neutral && inverted != neutral);
        Reads {
            case,
            inverted,
            neutral,
        }
    }
}

/// A kernel reading texture handles from memory binds the device's texture table: its variant
/// reads its layout for the device's tables at its first dispatch alone
/// (`WorkCounts::heap_tables_worked_out`), so the dispatches after it read it no more, and read the
/// case's texels as the first did.
#[test]
fn a_variant_reads_its_layout_for_the_device_tables_once() {
    on_device(|device, messages| {
        let sampling = Sampling::new(device);
        let texture = sampling.texture();
        let worked_out = || device.work_counts().heap_tables_worked_out;
        let at_start = worked_out();
        let mut seen = Vec::new();
        for _ in 0..4 {
            let before = worked_out();
            let output = sampling.run(device, texture, Some(texture));
            assert_eq!(output, sampling.case.metal_output, "the case's own texture");
            seen.push(worked_out() - before);
        }
        assert_eq!(seen, [1, 0, 0, 0], "from {at_start}");
        sampling.announced(device, messages, &[]);
    });
}

/// A texture placed at a live texture's ID takes it over: a kernel reading the ID reaches the
/// new texture though the encoder made only the old one resident, as it does with only the new one
/// resident, and both answer the ID. With neither resident it reaches the new one too, as
/// Metal's kernel did (probe 7; residency part C). Once the new one goes the ID reads the neutral
/// texture, though the old one, resident, lives: the slot keeps no earlier holder (probes 2 and
/// 3; Metal's kernel read the gone one's texels there).
#[test]
fn an_id_reads_the_texture_holding_it() {
    on_device(|device, messages| {
        let sampling = Sampling::new(device);
        let reads = Reads::of(device, &sampling);
        let texture = sampling.texture();
        let id = texture.gpu_resource_id();
        let holder = sampling.inverted(device, IdPlacement::At(id));
        assert_eq!(holder.gpu_resource_id(), id);
        assert_eq!(texture.gpu_resource_id(), id, "each answers the ID");
        let through_old = sampling.run(device, texture, Some(texture));
        assert_eq!(through_old, reads.inverted, "the old one resident");
        let through_new = sampling.run(device, &holder, Some(&holder));
        assert_eq!(through_new, reads.inverted, "the new one resident");
        let neither = sampling.run(device, texture, None);
        assert_eq!(neither, reads.inverted, "neither resident");
        drop(holder);
        let stale = sampling.run(device, texture, Some(texture));
        assert_eq!(stale, reads.neutral, "the holder went");
        sampling.announced(device, messages, &[]);
    });
}

/// A texture made in a heap is reached through its ID once the encoder uses the heap, as Metal's
/// `useHeap:` makes a heap's textures resident: the kernel reads the case's texels from a `Private`
/// automatic heap's texture holding them; and where the encoder made nothing resident, as Metal's
/// kernel did (residency part C).
#[test]
fn a_heap_s_texture_is_reached_through_its_id_once_the_heap_is_used() {
    on_device(|device, _| {
        let sampling = Sampling::new(device);
        let heap = device
            .new_heap(&HeapDescriptor::new(
                HeapType::Automatic,
                1 << 20,
                ResourceOptions::private(),
            ))
            .unwrap();
        let mut descriptor = texture_descriptor(&sampling.literal, sampling.usage).unwrap();
        descriptor.resource_options = ResourceOptions::private();
        let texture = heap.new_texture(&descriptor).unwrap();
        let command_buffer = sampling.queue.command_buffer().unwrap();
        sampling.upload(device, &command_buffer, &texture);
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        let used = sampling.run_using_heap(device, &texture, &heap);
        assert_eq!(used, sampling.case.metal_output, "the heap used");
        let unnamed = sampling.run(device, &texture, None);
        assert_eq!(unnamed, sampling.case.metal_output, "nothing resident");
    });
}

/// A texture made in a heap after the encoder used the heap is reached through its ID by the
/// encoder's later dispatches, as Metal's `useHeap:` reaches every texture the heap holds at
/// commit (SPH heap probe 1, an M3 Max and an M4): the kernel reads the case's texels from a
/// `Private` automatic heap's texture made and filled after `use_heap`, and from one made outside
/// the heap, as Metal's kernel did (residency part C).
#[test]
fn a_texture_made_in_a_used_heap_is_reached_by_later_dispatches() {
    on_device(|device, _| {
        let sampling = Sampling::new(device);
        let heap = device
            .new_heap(&HeapDescriptor::new(
                HeapType::Automatic,
                1 << 20,
                ResourceOptions::private(),
            ))
            .unwrap();
        let mut descriptor = texture_descriptor(&sampling.literal, sampling.usage).unwrap();
        descriptor.resource_options = ResourceOptions::private();
        let filled = |texture: Texture| {
            let command_buffer = sampling.queue.command_buffer().unwrap();
            sampling.upload(device, &command_buffer, &texture);
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None);
            texture
        };
        let used = sampling.run_making_after_use_heap(device, &heap, || {
            filled(heap.new_texture(&descriptor).unwrap())
        });
        assert_eq!(used, sampling.case.metal_output, "made in the heap after");
        let apart = sampling.run_making_after_use_heap(device, &heap, || {
            filled(device.new_texture(&descriptor).unwrap())
        });
        assert_eq!(apart, sampling.case.metal_output, "made outside the heap");
    });
}

/// A texture made in a used heap after a dispatch that reads texture handles was encoded, and
/// before commit, is what the dispatch reaches through its ID, as Metal's dispatch is (SPH heap
/// probe 1): the dispatch's argument buffer names a texture of inverted texels, which goes; one
/// placed at its ID in the heap, filled with the case's texels by another command buffer, is
/// what the kernel reads (residency part C; it was refused `NotImplemented(HeapTextures)`).
#[test]
fn a_texture_made_in_a_used_heap_after_a_dispatch_is_reached() {
    on_device(|device, messages| {
        let sampling = Sampling::new(device);
        let heap = device
            .new_heap(&HeapDescriptor::new(
                HeapType::Automatic,
                1 << 20,
                ResourceOptions::private(),
            ))
            .unwrap();
        let named = sampling.inverted(device, IdPlacement::Any);
        let id = named.gpu_resource_id();
        let (command_buffer, buffers) = sampling.encode_using_heap(device, &named, &heap);
        drop(named);
        let mut descriptor = texture_descriptor(&sampling.literal, sampling.usage).unwrap();
        descriptor.resource_options = ResourceOptions::private();
        let mut first = sampling.forced(device);
        let unplaced = descriptor;
        descriptor.placement = IdPlacement::At(id);
        let made = device.work_counts().heap_textures_tiled;
        let late = heap.new_texture(&descriptor).unwrap();
        let tiled = device.work_counts().heap_textures_tiled > made;
        first.extend(heap_texture_kinds(device, &unplaced, tiled));
        let upload = sampling.queue.command_buffer().unwrap();
        sampling.upload(device, &upload, &late);
        upload.commit().unwrap();
        command_buffer.commit().unwrap();
        let output = sampling.output(&command_buffer, &buffers);
        assert_eq!(
            output, sampling.case.metal_output,
            "made after the dispatch"
        );
        drop(late);
        assert_made_once(messages, &[], &first);
    });
}

/// A sparse heap's texture is reached through its ID once the encoder uses the heap, as Metal's
/// `useHeap:` makes a sparse heap's textures resident: the kernel reads the case's texels from a
/// 2-by-2 sparse texture, wholly its mip tail, mapped and then written, and the neutral texture
/// where the encoder made nothing resident. A device without sparse textures prints its typed
/// refusal.
#[test]
fn a_sparse_heap_s_texture_is_reached_through_its_id_once_the_heap_is_used() {
    on_device(|device, _| {
        let name = device.name().to_owned();
        let sampling = Sampling::new(device);
        let heap = HeapDescriptor::new(HeapType::Sparse, 1 << 20, ResourceOptions::private());
        let mut descriptor = texture_descriptor(&sampling.literal, sampling.usage).unwrap();
        descriptor.resource_options = ResourceOptions::private();
        let made = device
            .new_heap(&heap)
            .and_then(|heap| Ok((heap.new_texture(&descriptor)?, heap)));
        let (texture, heap) = match made {
            Ok(made) => made,
            Err(error) if lacks_sparse(&error) => {
                println!("SPARSE {name}: refused {error:?}");
                return;
            }
            Err(error) => panic!("a sparse texture on {name}: {error:?}"),
        };
        assert_eq!(texture.first_mipmap_in_tail(), Some(0), "{name}");
        let command_buffer = sampling.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.resource_state_command_encoder().unwrap();
        encoder
            .update_texture_mapping(&texture, TileMapping::Map, Region::new_2d(0, 0, 1, 1), 0, 0)
            .unwrap();
        encoder.end_encoding().unwrap();
        sampling.upload(device, &command_buffer, &texture);
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        let used = sampling.run_using_heap(device, &texture, &heap);
        assert_eq!(used, sampling.case.metal_output, "the heap used on {name}");
        let unnamed = sampling.run(device, &texture, None);
        assert_eq!(
            unnamed, sampling.case.metal_output,
            "nothing resident on {name}"
        );
    });
}

/// A texture placed at a texture's ID keeps it once that texture goes: the kernel reaches it
/// through the ID with it resident (probes 2 and 7).
#[test]
fn an_id_outlives_the_texture_it_was_taken_from() {
    on_device(|device, messages| {
        let sampling = Sampling::new(device);
        let reads = Reads::of(device, &sampling);
        let bytes = sampling.literal.bytes.clone();
        let taken = sampling.made(device, PixelFormat::RGBA32Uint, IdPlacement::Any, &bytes);
        let holder = sampling.inverted(device, IdPlacement::At(taken.gpu_resource_id()));
        drop(taken);
        let output = sampling.run(device, &holder, Some(&holder));
        assert_eq!(output, reads.inverted);
        sampling.announced(device, messages, &[]);
    });
}

/// A texture bound at an index of the dispatch, though the kernel does not read the index,
/// reaches its ID as a resident one does: the kernel reading the ID reads its holder with the
/// texture holding it bound, and with the texture it was taken from bound (probe 7).
#[test]
fn a_bound_texture_reaches_its_id() {
    on_device(|device, messages| {
        let sampling = Sampling::new(device);
        let reads = Reads::of(device, &sampling);
        let texture = sampling.texture();
        let own = sampling.run_bound(device, texture, texture);
        assert_eq!(own, reads.case, "the texture bound");
        let holder = sampling.inverted(device, IdPlacement::At(texture.gpu_resource_id()));
        let through_new = sampling.run_bound(device, texture, &holder);
        assert_eq!(through_new, reads.inverted, "the new one bound");
        let through_old = sampling.run_bound(device, texture, texture);
        assert_eq!(through_old, reads.inverted, "the old one bound");
        drop(holder);
        sampling.announced(device, messages, &[]);
    });
}

/// A command buffer reads an ID's holder as it runs: a texture placed at the ID after the
/// dispatch was encoded, before commit, is what the kernel reaches, and a holder that went
/// before commit leaves the neutral texture.
#[test]
fn a_command_buffer_reads_the_holder_at_commit() {
    on_device(|device, messages| {
        let sampling = Sampling::new(device);
        let reads = Reads::of(device, &sampling);
        let texture = sampling.texture();
        let id = texture.gpu_resource_id();
        let (command_buffer, buffers) = sampling.encode(device, texture, Some(texture), None);
        let holder = sampling.inverted(device, IdPlacement::At(id));
        command_buffer.commit().unwrap();
        let taken = sampling.output(&command_buffer, &buffers);
        assert_eq!(taken, reads.inverted, "made between encoding and commit");
        let (command_buffer, buffers) = sampling.encode(device, texture, Some(texture), None);
        drop(holder);
        command_buffer.commit().unwrap();
        let gone = sampling.output(&command_buffer, &buffers);
        assert_eq!(gone, reads.neutral, "gone between encoding and commit");
        sampling.announced(device, messages, &[]);
    });
}

/// A texture placed at an ID a committed command buffer reads, before the command buffer runs
/// (it waits for an event), is what the kernel reaches, as Metal's is: the device's heap holds the
/// ID's holder as the command buffer runs (residency part C). Nothing is announced.
#[test]
fn a_takeover_of_an_id_a_committed_command_buffer_reads_is_reached() {
    on_device(|device, messages| {
        let sampling = Sampling::new(device);
        let reads = Reads::of(device, &sampling);
        let texture = sampling.texture();
        let event = device.new_shared_event().unwrap();
        let wait = EventRef::from(&event);
        let (command_buffer, buffers) = sampling.encode(device, texture, Some(texture), Some(wait));
        command_buffer.commit().unwrap();
        let holder = sampling.inverted(device, IdPlacement::At(texture.gpu_resource_id()));
        event.set_signaled_value(1);
        let output = sampling.output(&command_buffer, &buffers);
        assert_eq!(output, reads.inverted, "the holder as it runs");
        drop(holder);
        sampling.announced(device, messages, &[]);
    });
}

/// A texture of another pixel format placed at the ID of the one the encoder made resident is
/// what the kernel reads through the ID, as Metal's is: an `R32Uint` texture, read as it is read
/// where the encoder makes it resident itself (residency part C). Nothing is announced.
#[test]
fn a_holder_of_another_format_is_read_as_itself() {
    on_device(|device, messages| {
        let sampling = Sampling::new(device);
        let reads = Reads::of(device, &sampling);
        let texture = sampling.texture();
        let id = texture.gpu_resource_id();
        let other = sampling.made(device, PixelFormat::R32Uint, IdPlacement::At(id), &[7; 16]);
        let output = sampling.run(device, texture, Some(texture));
        let own = sampling.run(device, &other, Some(&other));
        assert_eq!(output, own, "the holder, through the old one's residency");
        assert_ne!(output, reads.neutral);
        drop(other);
        sampling.announced(device, messages, &[]);
    });
}

/// Metal's answers of the M4 probe, plain and under API validation alike: a kernel reads
/// a texture through its ID in its argument buffer whatever the command's encoder named.
/// `tex.none`: no encoder of the command buffer named it. `tex.later_compute`: an earlier encoder
/// dispatches, and a later one only makes it resident. And a texture made after the dispatch was
/// encoded, at an ID whose texture went, is what it reaches. Nothing is announced. (Under
/// `MTL_SHADER_VALIDATION`, Metal's `tex.none` read zero: shader validation treats the unnamed
/// texture as absent.)
#[test]
fn an_unnamed_texture_is_reached_through_its_id() {
    on_device(|device, messages| {
        let sampling = Sampling::new(device);
        let reads = Reads::of(device, &sampling);
        let texture = sampling.texture();
        let none = sampling.run(device, texture, None);
        assert_eq!(none, reads.case, "tex.none");
        // tex.later_compute: the dispatch's encoder names nothing; a later encoder of the same
        // command buffer makes the texture resident.
        let (command_buffer, buffers) = sampling.encode(device, texture, None, None);
        let mut later = command_buffer.compute_command_encoder().unwrap();
        later
            .use_resource(texture.as_resource(), ResourceUsage::READ)
            .unwrap();
        later.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        let later = sampling.output(&command_buffer, &buffers);
        assert_eq!(later, reads.case, "tex.later_compute");
        // Made after the dispatch was encoded, at the ID of the texture it named, gone.
        let named = sampling.inverted(device, IdPlacement::Any);
        let id = named.gpu_resource_id();
        let (command_buffer, buffers) = sampling.encode(device, &named, None, None);
        drop(named);
        let bytes = sampling.literal.bytes.clone();
        let made = sampling.made(device, PixelFormat::RGBA32Uint, IdPlacement::At(id), &bytes);
        command_buffer.commit().unwrap();
        let after = sampling.output(&command_buffer, &buffers);
        assert_eq!(after, reads.case, "made after encoding");
        drop(made);
        sampling.announced(device, messages, &[]);
    });
}

/// A texture read through its ID goes while the command buffer that reads it waits to run: the
/// kernel reads the neutral texture, as the slot of a gone holder reads (Metal's kernel read the
/// gone texture's texels, probe 2), and the texture's image stays until the command buffer
/// completed, so the device reports nothing (the lane fails on any validation message).
#[test]
fn a_texture_gone_while_its_reader_waits_reads_the_neutral() {
    on_device(|device, messages| {
        let sampling = Sampling::new(device);
        let reads = Reads::of(device, &sampling);
        let doomed = sampling.inverted(device, IdPlacement::Any);
        let event = device.new_shared_event().unwrap();
        let wait = EventRef::from(&event);
        let (command_buffer, buffers) = sampling.encode(device, &doomed, None, Some(wait));
        command_buffer.commit().unwrap();
        drop(doomed);
        event.set_signaled_value(1);
        let output = sampling.output(&command_buffer, &buffers);
        assert_eq!(output, reads.neutral, "gone before it ran");
        sampling.announced(device, messages, &[]);
    });
}

/// A range token that notes its drop.
struct Noted(Arc<AtomicBool>);

impl Drop for Noted {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

impl Sampling {
    /// A texture as the case's, `Private` and never written, keeping a range token
    /// (`Device::new_texture_with_range_token`), and whether the token dropped.
    fn with_token(&self, device: &Device) -> (Texture, Arc<AtomicBool>) {
        let mut descriptor = texture_descriptor(&self.literal, self.usage).unwrap();
        descriptor.resource_options = ResourceOptions::private();
        let dropped = Arc::new(AtomicBool::new(false));
        let token = Box::new(Noted(Arc::clone(&dropped)));
        let texture = device
            .new_texture_with_range_token(&descriptor, token)
            .unwrap();
        (texture, dropped)
    }
}

/// A texture's range token drops as the program releases the texture where no command buffer
/// holds the texture object, as Metal frees its range (an M4, three runs alike:
/// `table`, `unrelated` and `firstmove`), though the runtime keeps its image meanwhile for the
/// command buffers that may reach it by its ID or took its first move: one reached only through
/// its ID by a command buffer waiting to run (which then reads the neutral, its image kept, so
/// the device reports nothing); one, its first move done, released while such a command buffer,
/// reaching another texture, waits; one made just before such a command buffer was committed.
#[test]
fn a_texture_s_range_token_drops_at_its_release_whatever_keeps_its_image() {
    on_device(|device, messages| {
        let sampling = Sampling::new(device);
        let reads = Reads::of(device, &sampling);
        let texture = sampling.texture();
        for case in ["through its ID", "another's reader", "first move"] {
            let (doomed, dropped) = sampling.with_token(device);
            if case == "another's reader" {
                // Its first move done by a command buffer that completed: only the waiting
                // command buffer's reach of the device's texture table keeps its image.
                assert_eq!(sampling.run(device, texture, Some(texture)), reads.case);
            }
            let event = device.new_shared_event().unwrap();
            let wait = Some(EventRef::from(&event));
            let named = match case {
                "through its ID" => &doomed,
                _ => texture,
            };
            let resident = (case != "through its ID").then_some(texture);
            let (command_buffer, buffers) = sampling.encode(device, named, resident, wait);
            command_buffer.commit().unwrap();
            drop(doomed);
            assert!(dropped.load(Ordering::SeqCst), "{case}: at its release");
            event.set_signaled_value(1);
            let output = sampling.output(&command_buffer, &buffers);
            let expected = match case {
                "through its ID" => &reads.neutral,
                _ => &reads.case,
            };
            assert_eq!(&output, expected, "{case}");
        }
        sampling.announced(device, messages, &[]);
    });
}

/// A sparse heap's range token drops as the heap and its textures go, as Metal frees the heap,
/// though the runtime keeps a gone sparse texture's image for an earlier command buffer that
/// binds the device's texture table: the image holds its heap weakly, the texture object
/// strongly (its `heap` answers it). That command buffer then reads its own texture as before.
#[test]
fn a_sparse_heap_s_token_drops_as_it_goes_whatever_keeps_its_textures_images() {
    on_device(|device, messages| {
        let sampling = Sampling::new(device);
        let reads = Reads::of(device, &sampling);
        let texture = sampling.texture();
        let dropped = Arc::new(AtomicBool::new(false));
        let token = Box::new(Noted(Arc::clone(&dropped)));
        let descriptor = HeapDescriptor::new(HeapType::Sparse, 1 << 20, ResourceOptions::private());
        let heap = match device.new_heap_with_range_token(&descriptor, token) {
            Ok(heap) => heap,
            Err(error) if lacks_sparse(&error) => {
                println!("SPARSE {}: refused {error:?}", device.name());
                assert!(dropped.load(Ordering::SeqCst), "a refused heap's");
                return;
            }
            Err(error) => panic!("a sparse heap on {}: {error:?}", device.name()),
        };
        let mut d = TextureDescriptor::texture_2d(PixelFormat::R32Float, 64, 64, false);
        d.usage = TextureUsage::SHADER_READ;
        d.resource_options = ResourceOptions::private();
        let sparse = match heap.new_texture(&d) {
            Ok(sparse) => sparse,
            Err(error) if lacks_sparse(&error) => {
                println!("SPARSE {}: texture refused {error:?}", device.name());
                return;
            }
            Err(error) => panic!("a sparse texture on {}: {error:?}", device.name()),
        };
        assert!(sparse.heap() == Some(&heap), "the texture answers its heap");
        let event = device.new_shared_event().unwrap();
        let wait = Some(EventRef::from(&event));
        let (command_buffer, buffers) = sampling.encode(device, texture, Some(texture), wait);
        command_buffer.commit().unwrap();
        drop(heap);
        assert!(
            !dropped.load(Ordering::SeqCst),
            "its texture keeps the heap"
        );
        drop(sparse);
        assert!(dropped.load(Ordering::SeqCst), "gone with its texture");
        event.set_signaled_value(1);
        assert_eq!(sampling.output(&command_buffer, &buffers), reads.case);
        let mut first = sampling.forced(device);
        first.extend(SparseFacts::of(device, PixelFormat::R32Float).made(device));
        assert_made_once(messages, &[], &first);
    });
}

/// An automatic or placement heap's range token drops as the program releases the heap and then
/// its texture, as Metal frees the heap's range (an M4, three runs alike: `table`
/// and `firstmove` free it at the release; `texheld`: the texture keeps it), though the runtime
/// keeps the texture's image, and the heap's memory it is bound to, for a command buffer that
/// reaches it only through its ID (which then reads the neutral, its image kept, so the device
/// reports nothing) or that took its first move (reaching another texture).
#[test]
fn a_heap_s_token_drops_at_its_release_whatever_keeps_its_textures_images() {
    on_device(|device, messages| {
        let sampling = Sampling::new(device);
        let reads = Reads::of(device, &sampling);
        let texture = sampling.texture();
        let mut d = texture_descriptor(&sampling.literal, sampling.usage).unwrap();
        d.resource_options = ResourceOptions::private();
        let made = device.work_counts().heap_textures_tiled;
        for heap_type in [HeapType::Automatic, HeapType::Placement] {
            for case in ["through its ID", "first move"] {
                let dropped = Arc::new(AtomicBool::new(false));
                let token = Box::new(Noted(Arc::clone(&dropped)));
                let descriptor =
                    HeapDescriptor::new(heap_type, 1 << 20, ResourceOptions::private());
                let heap = device
                    .new_heap_with_range_token(&descriptor, token)
                    .unwrap();
                let doomed = match heap_type {
                    HeapType::Placement => heap.new_texture_at_offset(&d, 0),
                    _ => heap.new_texture(&d),
                }
                .unwrap();
                assert!(doomed.heap() == Some(&heap), "the texture answers its heap");
                drop(heap);
                let what = format!("{heap_type:?} heap, {case}");
                assert!(
                    !dropped.load(Ordering::SeqCst),
                    "{what}: its texture keeps the heap"
                );
                let event = device.new_shared_event().unwrap();
                let wait = Some(EventRef::from(&event));
                let named = match case {
                    "through its ID" => &doomed,
                    _ => texture,
                };
                let resident = (case != "through its ID").then_some(texture);
                let (command_buffer, buffers) = sampling.encode(device, named, resident, wait);
                command_buffer.commit().unwrap();
                drop(doomed);
                assert!(
                    dropped.load(Ordering::SeqCst),
                    "{what}: at its texture's release"
                );
                event.set_signaled_value(1);
                let output = sampling.output(&command_buffer, &buffers);
                let expected = match case {
                    "through its ID" => &reads.neutral,
                    _ => &reads.case,
                };
                assert_eq!(&output, expected, "{what}");
            }
        }
        let tiled = device.work_counts().heap_textures_tiled > made;
        let mut first = sampling.forced(device);
        first.extend(heap_texture_kinds(device, &d, tiled));
        // A shadow is announced as its ID's holder once: the command that reaches it through its
        // ID alone binds the heap set holding it, neither naming it nor making it resident.
        if tiled {
            first.insert(UNNAMED_SHADOWS[0]);
        }
        assert_made_once(messages, &[], &first);
    });
}

/// A texture over a buffer, the buffer released first and then the texture while a command
/// buffer waiting to run reaches the texture only through its ID in an argument buffer: the
/// buffer goes at the texture's release, as Metal frees it (an M4, three runs
/// alike: `table` frees its range at the release; `texheld`: the texture keeps it), though the
/// runtime keeps the texture's image, and the memory it is bound to, until that command buffer
/// completed (which then reads the neutral, so the device reports nothing). Two forms:
///
/// - the consumer's surface texture (its `Beside::PendingArgument`): a texture of an IOSurface
///   plane over a no-copy buffer of the surface's pages (`new_texture_with_iosurface_plane`),
///   a shadow on every device (`shadowed-texture`), whose image lies in memory of its own;
///   the buffer's deallocator, the consumer's hold on the surface (a use the host sees), runs at
///   the texture's release, as Metal lets go of the surface. The deallocator scribbles over the
///   pages, as a caller reusing them would: the waiting command buffer, which starts after it,
///   reads the neutral through the ID, never the pages, and completes cleanly;
/// - a texture over a buffer of the runtime's memory, an image bound to the buffer's memory where
///   a linear image of the device holds its texels alone, else a shadow, whose range token drops
///   then.
///
/// A shadow is announced as its ID's holder once (`UNNAMED_SHADOWS`): the command binds the heap
/// set holding it with its ID in the argument buffer, neither naming it nor making it resident,
/// so its function may read the shadow stale; nothing else is.
#[test]
fn a_buffer_under_a_texture_goes_at_its_release_whatever_keeps_the_image() {
    let mapping = Mapping::anonymous(GUEST_PAGE);
    on_device(|device, messages| {
        let sampling = Sampling::new(device);
        let reads = Reads::of(device, &sampling);
        let mut surface = false;
        for form in ["surface plane", "runtime memory"] {
            let gone = Arc::new(AtomicBool::new(false));
            let noted = Arc::clone(&gone);
            let buffer = match form {
                "surface plane" => {
                    let deallocator: Deallocator = Box::new(move |at, length| {
                        // SAFETY: the mapping's bytes, which outlive the device; the caller's
                        // again from here, as the runtime let go of them.
                        unsafe { std::ptr::write_bytes(at.as_ptr(), 0xEE, length as usize) };
                        noted.store(true, Ordering::SeqCst);
                    });
                    let options = ResourceOptions::default();
                    let pointer = mapping.pointer();
                    // SAFETY: the mapping outlives the device, and so every buffer over it.
                    let made =
                        unsafe { no_copy(device, pointer, GUEST_PAGE, options, Some(deallocator)) };
                    match made {
                        Ok(buffer) => buffer,
                        Err(error) if not_importable(&error) => {
                            println!("SURFACE-HOLD {}: refused {error:?}", device.name());
                            continue;
                        }
                        Err(error) => panic!("a no-copy buffer on {}: {error:?}", device.name()),
                    }
                }
                _ => {
                    let token = Box::new(Noted(noted));
                    let length = GUEST_PAGE as u64;
                    let options = ResourceOptions::shared();
                    let descriptor = BufferDescriptor::new(length, options, BufferContents::Zero)
                        .with_range_token(token);
                    device.new_buffer_with_descriptor(descriptor).unwrap()
                }
            };
            // A format and rows a device's linear image holds alone (an image bound to the
            // runtime's memory, not a shadow, where it can: its rows packed, as wide as a device's
            // linear rows), unsigned as the kernel's: what it reads is gone anyway.
            let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Uint, ROW_TEXELS, 2, false);
            d.usage = TextureUsage::SHADER_READ;
            d.resource_options = buffer.resource_options();
            let doomed = match form {
                "surface plane" => {
                    let plane = IoSurfacePlane::new(0, ROW_BYTES, false);
                    buffer.new_texture_with_iosurface_plane(&d, plane)
                }
                _ => buffer.new_texture(&d, 0, ROW_BYTES),
            }
            .unwrap();
            // SAFETY: the handle is only compared, while the texture lives.
            let bound = unsafe { doomed.vk_image() }.is_some();
            println!(
                "SURFACE-HOLD {} {form}: an image bound to the bytes: {bound}",
                device.name()
            );
            assert!(
                !(bound && form == "surface plane"),
                "{form}: a texture over host pages takes an image bound to them"
            );
            surface |= form == "surface plane";
            drop(buffer);
            assert!(!gone.load(Ordering::SeqCst), "{form}: its texture keeps it");
            let event = device.new_shared_event().unwrap();
            let wait = Some(EventRef::from(&event));
            let (command_buffer, buffers) = sampling.encode(device, &doomed, None, wait);
            command_buffer.commit().unwrap();
            drop(doomed);
            assert!(
                gone.load(Ordering::SeqCst),
                "{form}: at the texture's release"
            );
            event.set_signaled_value(1);
            let output = sampling.output(&command_buffer, &buffers);
            assert_eq!(output, reads.neutral, "{form}");
        }
        // The surface's shadow, and the runtime memory's where the device's linear rows lie
        // otherwise, which its driver decides.
        let shadowed = match surface {
            true => [Degradation::ShadowedTexture, UNNAMED_SHADOWS[0]].into(),
            false => row_shadow_kinds(device),
        };
        println!(
            "SURFACE-HOLD {}: a shadow: {}",
            device.name(),
            !shadowed.is_empty()
        );
        sampling.announced(device, messages, &shadowed);
    });
}

/// A no-copy buffer and a texture over it released while a submitted command buffer that may
/// read the texture through its ID runs (its dispatch submitted and done, which an event it
/// signals then shows; a wait for another event after it holds it). The texture is a shadow
/// (`shadowed-texture`), whose image lies in memory of its own and holds no import of the pages:
/// the deallocator runs at the release, as Metal's does, with a keep-alive or
/// without, and nothing waits for the command buffer; a keep-alive goes on one of the device's
/// handler threads, after the import's free (`HostImport`'s drop). A command buffer committed but
/// not yet submitted holds nothing
/// ([`a_buffer_under_a_texture_goes_at_its_release_whatever_keeps_the_image`]). The device
/// reports nothing either way.
#[test]
fn a_no_copy_buffer_s_deallocator_runs_at_its_release_given_a_keep_alive() {
    let mapping = Mapping::anonymous(GUEST_PAGE);
    on_device(|device, messages| {
        let sampling = Sampling::new(device);
        for keeps in [true, false] {
            let ran = Arc::new(AtomicBool::new(false));
            let kept_dropped = Arc::new(AtomicBool::new(false));
            let noted = Arc::clone(&ran);
            let deallocator: Deallocator =
                Box::new(move |_, _| noted.store(true, Ordering::SeqCst));
            let keep_alive = keeps.then(|| Box::new(Noted(Arc::clone(&kept_dropped))) as KeepAlive);
            // SAFETY: the mapping outlives the device, and so every buffer over it.
            let memory = unsafe { HostMemory::new(mapping.pointer()) };
            let contents = BufferContents::NoCopy {
                memory,
                deallocator: Some(deallocator),
                keep_alive,
            };
            let options = ResourceOptions::default();
            let descriptor = BufferDescriptor::new(GUEST_PAGE as u64, options, contents);
            let buffer = match device.new_buffer_with_descriptor(descriptor) {
                Ok(buffer) => buffer,
                Err(error) if not_importable(&error) => {
                    println!("DEALLOCATOR-HOLD {}: refused {error:?}", device.name());
                    return;
                }
                Err(error) => panic!("a no-copy buffer on {}: {error:?}", device.name()),
            };
            let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Uint, ROW_TEXELS, 2, false);
            d.usage = TextureUsage::SHADER_READ;
            d.resource_options = buffer.resource_options();
            let doomed = buffer.new_texture(&d, 0, ROW_BYTES).unwrap();
            let what = format!("keep-alive {keeps}");
            // SAFETY: the handle is only compared, while the texture lives.
            let bound = unsafe { doomed.vk_image() }.is_some();
            assert!(
                !bound,
                "{what}: a texture over host pages takes an image bound to them"
            );
            let reaching = sampling.reaching(device, &doomed);
            let delayed = device.work_counts().host_deallocators_delayed;
            drop(buffer);
            drop(doomed);
            println!("DEALLOCATOR-HOLD {}: {what}", device.name());
            assert!(
                ran.load(Ordering::SeqCst),
                "{what}: the deallocator at the release"
            );
            // Nothing holds the import: it is freed with only the keep-alive waiting for the free,
            // on one of the device's handler threads, and the keep-alive goes there after it.
            let deadline = Instant::now() + WAIT;
            while keeps && !kept_dropped.load(Ordering::SeqCst) && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(10));
            }
            assert_eq!(
                kept_dropped.load(Ordering::SeqCst),
                keeps,
                "{what}: the keep-alive with the import's free"
            );
            assert_eq!(
                device.work_counts().host_deallocators_delayed,
                delayed,
                "{what}: counted delayed"
            );
            reaching.complete(&sampling);
        }
        // The shadow, reached through its ID, is announced as the lane above says.
        let expected = [Degradation::ShadowedTexture, UNNAMED_SHADOWS[0]];
        sampling.announced(device, messages, &expected);
    });
}

/// A command buffer reaching a texture only through its ID in its argument buffer
/// ([`Sampling::reaching`]), submitted and running: its dispatch submitted and done, which an event
/// it signals then shows, and a wait for another event after it holding it. It holds the
/// texture's image, and the memory the image is bound to, and nothing of the texture object or a
/// buffer under it, until [`complete`](Reaching::complete).
pub(super) struct Reaching {
    command_buffer: CommandBuffer,
    buffers: BTreeMap<u32, Buffer>,
    release: SharedEvent,
}

impl Sampling {
    /// A command buffer whose dispatch reaches `texture` only through its ID, committed and
    /// submitted: returned once its dispatch ran ([`Reaching`]).
    pub(super) fn reaching(&self, device: &Device, texture: &Texture) -> Reaching {
        let (started, release) = (
            device.new_shared_event().unwrap(),
            device.new_shared_event().unwrap(),
        );
        let (command_buffer, buffers) = self.encode(device, texture, None, None);
        command_buffer
            .encode_signal_event(EventRef::from(&started), 1)
            .unwrap();
        command_buffer
            .encode_wait_for_event(EventRef::from(&release), 1)
            .unwrap();
        command_buffer.commit().unwrap();
        assert!(
            started.wait_until_signaled_value(1, WAIT),
            "the dispatch's batch ran"
        );
        Reaching {
            command_buffer,
            buffers,
            release,
        }
    }
}

impl Reaching {
    /// Let the command buffer go on, and wait until it completed.
    pub(super) fn complete(self, sampling: &Sampling) {
        self.release.set_signaled_value(1);
        // The dispatch read the texture or, once it went, the neutral: either is Metal's.
        sampling.output(&self.command_buffer, &self.buffers);
    }
}

/// The bytes between rows of the textures over a buffer the lanes above make: Metal's linear
/// alignment, and an IOSurface's, for every format.
const ROW_BYTES: u64 = 256;

/// The `RGBA8Uint` texels of such a row, packed: a device whose linear rows align to 256 bytes or
/// less binds the image to the bytes.
const ROW_TEXELS: u64 = ROW_BYTES / 4;

/// Samples a float texture through a linear sampler, both held in its argument buffer.
const LINEAR_SAMPLER: &str =
    "sample-through-a-linear-sampler-held-beside-its-texture-in-an-argument-buffer";

/// The set-1 capacity of the pipeline `command_buffer`'s one dispatch ran: its heap arrays'
/// element count.
fn heap_capacity(command_buffer: &CommandBuffer) -> u32 {
    let [execution] = &command_buffer.executions()[..] else {
        panic!("one dispatch");
    };
    let ExecutedPipeline::Compute(pipeline) = execution.pipeline() else {
        panic!("a compute pipeline");
    };
    let set = &pipeline.layout.sets[1];
    set.bindings.iter().map(|b| b.count).min().unwrap()
}

/// A texture that a heap array's class fits, whose format the array cannot view (`BGRA10_XR_sRGB`,
/// whose raw texels need its function's decode and tables, and which no heap reads raw), is the
/// neutral at its ID in the device's heap set, announced once as the set holds it (`UNVIEWABLE`): a
/// second is not announced again, and the case's own texture still reads its texels.
#[test]
fn a_texture_its_heap_array_cannot_view_is_announced_once() {
    on_device(|device, messages| {
        let sampling = Sampling::of(device, LINEAR_SAMPLER);
        let texture = sampling.texture();
        let case = sampling.run(device, texture, Some(texture));
        assert_eq!(case, sampling.case.metal_output, "the case's own texture");
        sampling.announced(device, messages, &[]);
        let first = sampling.made(
            device,
            PixelFormat::BGRA10XRSRGB,
            IdPlacement::Any,
            &[0x5A; 32],
        );
        sampling.announced(device, messages, &UNVIEWABLE);
        let second = sampling.made(
            device,
            PixelFormat::BGRA10XRSRGB,
            IdPlacement::Any,
            &[0xA5; 32],
        );
        sampling.announced(device, messages, &[]);
        assert_eq!(sampling.run(device, texture, Some(texture)), case);
        drop((first, second));
    });
}

/// A view of another format of a texture made without `PixelFormatView` usage (`BGRA8Unorm` of
/// an `RGBA8Unorm` texture) holds its texels in a shadow image the encoders that name it copy them
/// into. Through its ID it reads as a plain `BGRA8Unorm` texture of those bytes does, named or not:
/// an encoder that named it copied them in first. The view is a shadow on every device
/// (`shadowed-texture-view`), and the device's heap set holding it is announced once
/// (`UNNAMED_SHADOWS`).
#[test]
fn a_shadowed_view_reached_through_its_id_is_announced_once() {
    on_device(|device, messages| {
        let sampling = Sampling::of(device, LINEAR_SAMPLER);
        let texture = sampling.texture();
        let bytes: Vec<u8> = (0..16u8).map(|i| 17 * i).collect();
        let plain = sampling.made(device, PixelFormat::BGRA8Unorm, IdPlacement::Any, &bytes);
        let reference = sampling.run(device, &plain, Some(&plain));
        let base = sampling.made(device, PixelFormat::RGBA8Unorm, IdPlacement::Any, &bytes);
        let view = base.new_texture_view(PixelFormat::BGRA8Unorm).unwrap();
        assert_eq!(sampling.run(device, &view, Some(&view)), reference, "named");
        assert_eq!(
            sampling.run(device, &view, Some(texture)),
            reference,
            "unnamed"
        );
        // The view is a shadow on every device: Vulkan views an image of another format only
        // where it was made of mutable format listing it, and a texture made without
        // `PixelFormatView` usage lists only its sRGB twin.
        let expected = [Degradation::ShadowedTextureView, UNNAMED_SHADOWS[0]];
        sampling.announced(device, messages, &expected);
        drop((view, base, plain));
    });
}

/// The row pitch of [`Sampling::over`]'s texture: its two texels' 32 bytes and 16 more, a pitch
/// no device's linear image of two `RGBA32Uint` texels has, so the texture is a shadow
/// (`shadowed-texture`) on every venue.
const SHADOW_ROW: u64 = 48;

impl Sampling {
    /// A texture as the case's over `buffer` from its first byte, [`SHADOW_ROW`] bytes a row.
    fn over(&self, buffer: &Buffer) -> Texture {
        let descriptor = texture_descriptor(&self.literal, self.usage).unwrap();
        buffer.new_texture(&descriptor, 0, SHADOW_ROW).unwrap()
    }

    /// One compute encoder in `command_buffer` dispatching the case with `named`'s ID in its
    /// argument buffer, binding and making resident the textures of `reaching` and the heap
    /// `heap` where given, after waiting for the fence `after` where given: the dispatch's
    /// buffers, its output among them.
    fn encode_in(
        &self,
        device: &Device,
        command_buffer: &CommandBuffer,
        named: &Texture,
        (reaching, heap): (CaseResources, Option<&Heap>),
        after: Option<&Fence>,
    ) -> BTreeMap<u32, Buffer> {
        let naming = CaseResources {
            argument_textures: BTreeMap::from([(self.key, named.clone())]),
            argument_samplers: self.base.argument_samplers.clone(),
            ..CaseResources::default()
        };
        let buffers = bound_buffers(device, &self.case.compute, &naming).unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        if let Some(fence) = after {
            encoder.wait_for_fence(fence).unwrap();
        }
        if let Some(heap) = heap {
            encoder.use_heap(heap).unwrap();
        }
        encode_case(
            &mut encoder,
            &self.pipeline,
            &self.case.compute,
            &buffers,
            &reaching,
        )
        .unwrap();
        encoder.end_encoding().unwrap();
        buffers
    }

    /// The case's output in `buffers`, their command buffer completed.
    fn output_of(&self, buffers: &BTreeMap<u32, Buffer>) -> Vec<u8> {
        let ComputeOutput::Buffer {
            binding,
            offset,
            length,
        } = self.case.compute.output
        else {
            panic!("{}: a buffer output", self.case.case.name);
        };
        read_buffer(&buffers[&binding], offset, length).unwrap()
    }

    /// The case's output over a texture of its own holding `texels`: what a read of a texture
    /// holding them answers.
    fn reads(&self, device: &Device, texels: &[u8]) -> Vec<u8> {
        let own = self.made(device, PixelFormat::RGBA32Uint, IdPlacement::Any, texels);
        self.run(device, &own, Some(&own))
    }
}

/// The case's 2-by-2 texels `texels` laid out [`SHADOW_ROW`] bytes a row, 0xEE past each row's.
fn shadow_rows(texels: &[u8]) -> Vec<u8> {
    texels
        .chunks(texels.len() / 2)
        .flat_map(|row| {
            let mut padded = row.to_vec();
            padded.resize(SHADOW_ROW as usize, 0xEE);
            padded
        })
        .collect()
}

/// Write `bytes` at the start of `buffer`'s contents, which no command uses.
fn write_at_start(buffer: &Buffer, bytes: &[u8]) {
    let contents = buffer.contents().unwrap();
    // SAFETY: the buffer holds at least `bytes.len()` bytes (the caller made it so), and no
    // command uses them now.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
}

/// Which of `wants` (named) `got` is, for the lane's line.
fn which(got: &[u8], wants: &[(&str, &[u8])]) -> String {
    let name = wants.iter().find(|(_, want)| *want == got);
    name.map_or_else(|| "neither".to_owned(), |(name, _)| (*name).to_owned())
}

/// Whether `notices` hold a shadowed texture's (`shadowed-texture`).
fn shadowed(notices: &[DebugMessage]) -> bool {
    notices.iter().any(|message| {
        (message.degradation.as_ref())
            .is_some_and(|n| n.degradation == Degradation::ShadowedTexture)
    })
}

/// Metal's residency lasts the command buffer (the coordinator's ruling of 2026-10-07): a texture
/// over a buffer one encoder made resident is read through its ID, by a later encoder of the
/// command buffer that names nothing, as its bytes stand then, after a blit between the two changed
/// them. Metal's answer, the M4 probe: the later encoder reads the blit's
/// bytes (with no residency at all, `cb_none`, it read the first encoder's). The texture is a
/// shadow ([`SHADOW_ROW`]); the line `SHADOW-READ` reports what each encoder read.
#[test]
fn a_shadowed_texture_a_command_buffer_made_resident_is_read_as_its_bytes_stand() {
    on_device(|device, messages| {
        let sampling = Sampling::new(device);
        let first = sampling.literal.bytes.clone();
        let second: Vec<u8> = first.iter().map(|byte| !byte).collect();
        let (want_first, want_second) = (
            sampling.reads(device, &first),
            sampling.reads(device, &second),
        );
        assert_ne!(want_first, want_second);
        let buffer = buffer_with(device, &shadow_rows(&first)).unwrap();
        let source = buffer_with(device, &shadow_rows(&second)).unwrap();
        let texture = sampling.over(&buffer);
        let command_buffer = sampling.queue.command_buffer().unwrap();
        let resident = CaseResources {
            argument_textures: BTreeMap::from([(sampling.key, texture.clone())]),
            ..CaseResources::default()
        };
        let a = sampling.encode_in(device, &command_buffer, &texture, (resident, None), None);
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_buffer(&source, 0, &buffer, 0, buffer.length())
            .unwrap();
        blit.end_encoding().unwrap();
        let nothing = (CaseResources::default(), None);
        let c = sampling.encode_in(device, &command_buffer, &texture, nothing, None);
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let notices = messages.take_notices();
        let wants: [(&str, &[u8]); 2] = [("first", &want_first), ("second", &want_second)];
        let (a, c) = (sampling.output_of(&a), sampling.output_of(&c));
        eprintln!(
            "SHADOW-READ {} command buffer: a shadow {}; A (resident) read {}, C (after the blit, \
             naming nothing) read {}",
            device.name(),
            shadowed(&notices),
            which(&a, &wants),
            which(&c, &wants),
        );
        assert_eq!(a, want_first, "A, which made it resident");
        assert_eq!(c, want_second, "C, after the blit, naming nothing");
        drop((texture, buffer));
    });
}

/// `useHeap:` makes resident every resource of the heap, a texture over a buffer placed in it
/// among them: such a texture is read through its ID, by an encoder that uses only the heap, as
/// its bytes stand, after the CPU wrote them and after a blit of the command buffer wrote them.
/// Metal's answer, the M4 probe: each read the bytes as they stood (with no
/// `useHeap:`, `heap_none`, the first two read zeros). The heap's resources are untracked (a
/// heap's default), so the reading encoder waits for a fence the blit updates, as a Metal program
/// must: on the M3 Max under macOS 27.0.1 (probe `heap_order`) a compute encoder read a texture
/// over such a buffer before the blit just before it had written it, in every run without the
/// fence and in none with it. The texture is a shadow ([`SHADOW_ROW`]); the line `SHADOW-READ`
/// reports what each read.
#[test]
fn a_shadowed_texture_over_a_heap_s_buffer_is_read_through_use_heap_as_its_bytes_stand() {
    on_device(|device, messages| {
        let sampling = Sampling::new(device);
        let first = sampling.literal.bytes.clone();
        let second: Vec<u8> = first.iter().map(|byte| !byte).collect();
        let third: Vec<u8> = first.iter().map(|byte| byte ^ 0x55).collect();
        let wants = [&first, &second, &third].map(|texels| sampling.reads(device, texels));
        let heap = device
            .new_heap(&HeapDescriptor::new(
                HeapType::Automatic,
                1 << 16,
                ResourceOptions::shared(),
            ))
            .unwrap();
        let length = 2 * SHADOW_ROW;
        let buffer = heap.new_buffer(length, ResourceOptions::shared()).unwrap();
        write_at_start(&buffer, &shadow_rows(&first));
        let texture = sampling.over(&buffer);
        let source = buffer_with(device, &shadow_rows(&third)).unwrap();
        let fence = device.new_fence();
        let mut got = Vec::new();
        for step in 0..3 {
            if step == 1 {
                write_at_start(&buffer, &shadow_rows(&second));
            }
            let command_buffer = sampling.queue.command_buffer().unwrap();
            let after = (step == 2).then(|| {
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.copy_from_buffer_to_buffer(&source, 0, &buffer, 0, length)
                    .unwrap();
                blit.update_fence(&fence).unwrap();
                blit.end_encoding().unwrap();
                &fence
            });
            let using_heap = (CaseResources::default(), Some(&heap));
            let buffers = sampling.encode_in(device, &command_buffer, &texture, using_heap, after);
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            got.push(sampling.output_of(&buffers));
        }
        let notices = messages.take_notices();
        let named: [(&str, &[u8]); 3] = [
            ("first", &wants[0]),
            ("second", &wants[1]),
            ("third", &wants[2]),
        ];
        eprintln!(
            "SHADOW-READ {} use_heap: a shadow {}; CB1 read {}, CB2 (after a CPU write) read {}, \
             CB3 (after its blit) read {}",
            device.name(),
            shadowed(&notices),
            which(&got[0], &named),
            which(&got[1], &named),
            which(&got[2], &named),
        );
        assert_eq!(got[0], wants[0], "CB1");
        assert_eq!(got[1], wants[1], "CB2, after a CPU write");
        assert_eq!(got[2], wants[2], "CB3, after a blit in it and its fence");
        drop((texture, buffer, heap));
    });
}

/// A texture placed at the highest ID Metal hands out (16380) is reached through it by a command
/// that does not name it, where the pipeline's heap arrays hold that element (every venue
/// measured: their count is 65,536 or the device's limits shared among them). Where they end below
/// it, the kernel reads the neutral texture, announced once (`PAST_CAPACITY`). The line
/// `TEXTURE-CAPACITY` reports the pipeline's count.
#[test]
fn a_texture_at_the_highest_id_is_reached_where_the_arrays_hold_it() {
    on_device(|device, messages| {
        let sampling = Sampling::new(device);
        let reads = Reads::of(device, &sampling);
        let high = ID_LIMIT - 1;
        let placed = sampling.inverted(device, IdPlacement::At(ResourceId(high)));
        assert_eq!(placed.gpu_resource_id(), ResourceId(high));
        let texture = sampling.texture();
        let (command_buffer, buffers) = sampling.encode(device, &placed, Some(texture), None);
        command_buffer.commit().unwrap();
        let output = sampling.output(&command_buffer, &buffers);
        let capacity = heap_capacity(&command_buffer);
        eprintln!(
            "TEXTURE-CAPACITY {}: pipeline set-1 capacity {capacity}",
            device.name()
        );
        if u64::from(capacity) > high {
            assert_eq!(output, reads.inverted, "through ID {high}");
            sampling.announced(device, messages, &[]);
        } else {
            assert_eq!(output, reads.neutral, "past the arrays");
            sampling.announced(device, messages, &PAST_CAPACITY);
        }
        drop(placed);
    });
}

/// The case's pipeline state reads its texture through its ID, so its set 1 holds the heap's
/// arrays, past the 1024 descriptors every Vulkan device's set layouts hold. Where the device's
/// driver refuses that layout through `vkGetDescriptorSetLayoutSupport` (asked here apart from
/// the runtime, `degrade_notices::refused_sets`) and runs it as Metal runs the argument buffer
/// (`RuntimeDriverDefect::SetLayoutSupportUnderstated`: MoltenVK), the making announces it once
/// (`PAST_LAYOUT_SUPPORT`), and the validation layer's report of each such layout passes after
/// the notice (`candidate/unsupported_set.rs`); a second state of the case announces nothing
/// again. Where the driver supports the layout, nothing is announced. The dispatch reads the
/// case's texels through the set either way. The line `SET-SUPPORT` reports the sets the driver
/// refuses.
#[test]
fn a_heap_set_its_driver_refuses_is_announced_once_where_it_runs() {
    on_device(|device, messages| {
        let sampling = Sampling::new(device);
        let texture = sampling.texture();
        let (command_buffer, buffers) = sampling.encode(device, texture, Some(texture), None);
        command_buffer.commit().unwrap();
        let output = sampling.output(&command_buffer, &buffers);
        assert_eq!(output, sampling.case.metal_output, "through the heap set");
        let [execution] = &command_buffer.executions()[..] else {
            panic!("one dispatch");
        };
        let ExecutedPipeline::Compute(pipeline) = execution.pipeline() else {
            panic!("a compute pipeline");
        };
        let refused = refused_sets(device, &pipeline.layout);
        eprintln!(
            "SET-SUPPORT {}: the driver refuses the sets (number, descriptors) {refused:?}",
            device.name()
        );
        let understated = (device.properties().driver_defects)
            .contains(&RuntimeDriverDefect::SetLayoutSupportUnderstated);
        assert!(
            refused.is_empty() || understated,
            "made a set its driver refuses: {refused:?}"
        );
        let kinds: &[Degradation] = match refused.is_empty() {
            true => &[],
            false => &PAST_LAYOUT_SUPPORT,
        };
        sampling.announced(device, messages, kinds);
        drop(pipeline_state(device, &sampling.case).unwrap());
        sampling.announced(device, messages, &[]);
    });
}

/// On a device without `descriptorBindingUpdateUnusedWhilePending` (withheld here; every venue
/// has it), which writes no descriptor set a pending command buffer binds, each command buffer
/// binds a copy of the heap set of its own, announced once (`OWN_COPIES`). The IDs read as on any
/// device (the case's texture, a texture of its own, the neutral once its holder went); a texture
/// made after the dispatch was encoded and before the commit is reached, as its copy takes
/// writes until then. A command buffer reads each ID's texture as it committed: one made while
/// it waits to run is the neutral for it (a later command buffer reaches it). One that went after
/// the commit and before the command buffer's first submission is the neutral for it too, its image
/// gone at its release, where Metal's reads the new holder (it was read still, its image kept until
/// the command buffer completed).
#[test]
fn a_device_without_pending_writes_binds_a_copy_per_command_buffer() {
    let feature = VkFeature::DescriptorBindingUpdateUnusedWhilePending;
    on_device_with(
        |options| {
            options.withheld_features.insert(feature);
        },
        |device, messages| {
            assert!(!device.enabled().features.contains(&feature));
            let sampling = Sampling::new(device);
            let reads = Reads::of(device, &sampling);
            sampling.announced(device, messages, &OWN_COPIES);
            let bytes = sampling.literal.bytes.clone();
            // Made after the dispatch was encoded, at the ID of the texture it named, gone.
            let named = sampling.inverted(device, IdPlacement::Any);
            let id = named.gpu_resource_id();
            let (command_buffer, buffers) = sampling.encode(device, &named, None, None);
            drop(named);
            let made = sampling.made(device, PixelFormat::RGBA32Uint, IdPlacement::At(id), &bytes);
            command_buffer.commit().unwrap();
            let output = sampling.output(&command_buffer, &buffers);
            assert_eq!(output, reads.case, "made after encoding");
            drop(made);
            // Made while the committed command buffer waits to run.
            let named = sampling.inverted(device, IdPlacement::Any);
            let id = named.gpu_resource_id();
            let event = device.new_shared_event().unwrap();
            let wait = EventRef::from(&event);
            let (command_buffer, buffers) = sampling.encode(device, &named, None, Some(wait));
            drop(named);
            command_buffer.commit().unwrap();
            let late = sampling.made(device, PixelFormat::RGBA32Uint, IdPlacement::At(id), &bytes);
            event.set_signaled_value(1);
            let output = sampling.output(&command_buffer, &buffers);
            assert_eq!(output, reads.neutral, "made after the commit");
            assert_eq!(sampling.run(device, &late, None), reads.case, "a later one");
            drop(late);
            // Gone after the commit, before the first submission, its ID taken by another.
            let kept = sampling.inverted(device, IdPlacement::Any);
            let id = kept.gpu_resource_id();
            let event = device.new_shared_event().unwrap();
            let wait = EventRef::from(&event);
            let (command_buffer, buffers) = sampling.encode(device, &kept, None, Some(wait));
            command_buffer.commit().unwrap();
            drop(kept);
            let taker = sampling.made(device, PixelFormat::RGBA32Uint, IdPlacement::At(id), &bytes);
            event.set_signaled_value(1);
            let output = sampling.output(&command_buffer, &buffers);
            assert_eq!(output, reads.neutral, "gone before its first submission");
            assert_eq!(
                sampling.run(device, &taker, None),
                reads.case,
                "a later one"
            );
            drop(taker);
            sampling.announced(device, messages, &[]);
        },
    );
}

/// On a device without `descriptorBindingUpdateUnusedWhilePending` (withheld here; every venue
/// has it), whose command buffers bind copies of the heap set of their own (`OWN_COPIES`,
/// announced once), a function reaching a sampler state by its ID reads the state that held it as
/// its command buffer's first batch was submitted, the copy taking a state's elements until then:
/// in [`LINEAR_SAMPLER`], a state of the case's descriptor made at the ID the dispatch's argument
/// buffer names after the dispatch was encoded, the case's gone, reads the case's answer, and so
/// does one made there after the commit, while the command buffer waits to run, the earlier one
/// released.
#[test]
fn a_device_without_pending_writes_reads_the_sampler_states_of_its_submission() {
    let feature = VkFeature::DescriptorBindingUpdateUnusedWhilePending;
    on_device_with(
        |options| {
            options.withheld_features.insert(feature);
        },
        |device, messages| {
            assert!(!device.enabled().features.contains(&feature));
            let mut sampling = Sampling::of(device, LINEAR_SAMPLER);
            let key = sampling.sampler_key();
            let texture = sampling.texture().clone();
            let metal = sampling.case.metal_output.clone();
            // Made after encoding, before the commit.
            let (command_buffer, buffers) = sampling.encode(device, &texture, Some(&texture), None);
            let (id, descriptor) = sampling.drop_case_state();
            let before = distinct_state_at(device, id, &descriptor, 1);
            command_buffer.commit().unwrap();
            assert_eq!(
                sampling.output(&command_buffer, &buffers),
                metal,
                "made before the commit"
            );
            sampling.announced(device, messages, &OWN_COPIES);
            // Made after the commit, while the command buffer waits to run.
            sampling.base.argument_samplers.insert(key, before);
            let event = device.new_shared_event().unwrap();
            let wait = EventRef::from(&event);
            let (command_buffer, buffers) =
                sampling.encode(device, &texture, Some(&texture), Some(wait));
            command_buffer.commit().unwrap();
            drop(sampling.base.argument_samplers.remove(&key));
            let after = distinct_state_at(device, id, &descriptor, 2);
            event.set_signaled_value(1);
            assert_eq!(
                sampling.output(&command_buffer, &buffers),
                metal,
                "made after the commit"
            );
            sampling.announced(device, messages, &[]);
            drop(after);
        },
    );
}

/// A sampler state of `descriptor` but for its upper LOD clamp, `n` levels (which a texture of one
/// level never reaches), with argument-buffer support, placed at `id`: another state than any of
/// `descriptor` alone, which a command buffer may hold still.
fn distinct_state_at(
    device: &Device,
    id: ResourceId,
    descriptor: &SamplerDescriptor,
    n: u32,
) -> SamplerState {
    let mut distinct = *descriptor;
    distinct.lod_max_clamp = n as f32;
    pixel_state_at(device, id, &distinct)
}

/// A sampler state made at the ID a dispatch's argument buffer names after the dispatch was
/// encoded samples through its own Vulkan sampler, whatever the states live as the dispatch was
/// encoded: in [`LINEAR_SAMPLER`], the case's linear state gone, a nearest state at its ID
/// reads the nearest texels, and a linear one made there next the case's answer, each in a
/// command buffer of its own encoded before it was made. Nothing is announced.
#[test]
fn a_sampler_state_made_after_encoding_samples_through_its_own_sampler() {
    on_device(|device, messages| {
        let mut sampling = Sampling::of(device, LINEAR_SAMPLER);
        let key = sampling.sampler_key();
        let texture = sampling.texture().clone();
        let metal = sampling.case.metal_output.clone();
        let (command_buffer, buffers) = sampling.encode(device, &texture, Some(&texture), None);
        let (id, descriptor) = sampling.drop_case_state();
        assert_eq!(descriptor.min_filter, SamplerMinMagFilter::Linear);
        let mut nearest = descriptor;
        nearest.min_filter = SamplerMinMagFilter::Nearest;
        nearest.mag_filter = SamplerMinMagFilter::Nearest;
        let state = pixel_state_at(device, id, &nearest);
        command_buffer.commit().unwrap();
        let read = sampling.output(&command_buffer, &buffers);
        assert_ne!(
            read, metal,
            "a nearest state reads other texels than the case's linear one"
        );
        sampling.base.argument_samplers.insert(key, state);
        assert_eq!(
            sampling.run(device, &texture, Some(&texture)),
            read,
            "the nearest state, live as the dispatch was encoded"
        );
        let (command_buffer, buffers) = sampling.encode(device, &texture, Some(&texture), None);
        drop(sampling.base.argument_samplers.remove(&key));
        let linear = distinct_state_at(device, id, &descriptor, 1);
        command_buffer.commit().unwrap();
        assert_eq!(
            sampling.output(&command_buffer, &buffers),
            metal,
            "a linear state made after encoding"
        );
        sampling.announced(device, messages, &[]);
        drop(linear);
    });
}

/// A sampler state that takes the ID a dispatch's argument buffer names, made after its command
/// buffer committed and before the event it waits for is signalled, is reached through that ID, as
/// Metal's function reaches it (the M3 Max under macOS 27.0.1: a state made after the commit,
/// its ID written into the argument buffer before the signal, is the one sampled): the device's
/// sampler table maps the ID to it as it takes the ID's slot. Nothing is announced. (On a device
/// without `descriptorBindingUpdateUnusedWhilePending`, which no venue is, the command buffer's
/// copy of the heap set holds the states of its commit:
/// [`a_device_without_pending_writes_reads_the_sampler_states_of_its_submission`].)
#[test]
fn a_sampler_state_made_after_the_commit_before_its_wait_is_reached_by_its_id() {
    on_device(|device, messages| {
        let mut sampling = Sampling::new(device);
        let key = *sampling
            .case
            .compute
            .argument_buffer_samplers
            .keys()
            .next()
            .unwrap();
        let resource = &sampling.case.compute.argument_buffer_samplers[&key];
        let mut descriptor = argument_sampler_descriptor(resource);
        descriptor.normalized_coordinates = false;
        assert_ne!(corner(), sampling.case.metal_output);
        let texture = sampling.texture().clone();
        let event = device.new_shared_event().unwrap();
        let wait = EventRef::from(&event);
        let (command_buffer, buffers) =
            sampling.encode(device, &texture, Some(&texture), Some(wait));
        command_buffer.commit().unwrap();
        let own = sampling.base.argument_samplers.remove(&key).unwrap();
        let id = own.gpu_resource_id();
        drop(own);
        let mut options = SamplerStateOptions::default();
        options.support_argument_buffers = true;
        options.placement = IdPlacement::At(id);
        let state = device
            .new_sampler_state_with_options(&descriptor, options)
            .unwrap();
        assert_eq!(state.gpu_resource_id(), id);
        event.set_signaled_value(1);
        assert_eq!(
            sampling.output(&command_buffer, &buffers),
            corner(),
            "through ID {} after the commit",
            id.0
        );
        sampling.announced(device, messages, &[]);
        drop(state);
    });
}

/// Sampler states made after a dispatch's command buffer committed and before the event it waits
/// for is signalled, 64 more than the case's (past twice the capacity class of the states live as
/// the dispatch was encoded, the most a table of the command's own held), and then a state in
/// pixel coordinates at the ID the dispatch's argument buffer names: the kernel reads texel (0, 0)
/// through it, as Metal's reaches the state holding the ID as it runs. Nothing is announced or
/// counted (`WorkCounts::unlisted_argument_sampler_buffers`).
#[test]
fn sampler_states_made_after_the_commit_past_the_class_are_reached_by_their_ids() {
    on_device(|device, messages| {
        let mut sampling = Sampling::new(device);
        let texture = sampling.texture().clone();
        let unlisted = || device.work_counts().unlisted_argument_sampler_buffers;
        let before = unlisted();
        let event = device.new_shared_event().unwrap();
        let wait = EventRef::from(&event);
        let (command_buffer, buffers) =
            sampling.encode(device, &texture, Some(&texture), Some(wait));
        command_buffer.commit().unwrap();
        let (id, descriptor) = sampling.drop_case_sampler();
        let past = distinct_sampler_states(device, 64);
        let state = pixel_state_at(device, id, &descriptor);
        event.set_signaled_value(1);
        assert_eq!(
            sampling.output(&command_buffer, &buffers),
            corner(),
            "64 more states, then ID {}",
            id.0
        );
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        assert_eq!(unlisted(), before);
        sampling.announced(device, messages, &[]);
        drop((past, state));
    });
}

impl Sampling {
    /// Drop the case's argument-buffer sampler state: its ID, which the dispatches encoded
    /// before still name, and its descriptor in pixel coordinates, through which
    /// [`HANDLE_SAMPLER`]'s kernel reads texel (0, 0) at each of its coordinates ([`corner`]).
    fn drop_case_sampler(&mut self) -> (ResourceId, SamplerDescriptor) {
        let (id, mut descriptor) = self.drop_case_state();
        assert!(descriptor.normalized_coordinates, "the case's state");
        descriptor.normalized_coordinates = false;
        assert_ne!(
            corner(),
            self.case.metal_output,
            "the case's state's answer differs"
        );
        (id, descriptor)
    }

    /// Drop the case's argument-buffer sampler state: its ID and its descriptor.
    fn drop_case_state(&mut self) -> (ResourceId, SamplerDescriptor) {
        let key = self.sampler_key();
        let descriptor =
            argument_sampler_descriptor(&self.case.compute.argument_buffer_samplers[&key]);
        let own = self.base.argument_samplers.remove(&key).unwrap();
        (own.gpu_resource_id(), descriptor)
    }

    /// The key of the case's argument-buffer sampler state.
    fn sampler_key(&self) -> (u32, u32) {
        *self
            .case
            .compute
            .argument_buffer_samplers
            .keys()
            .next()
            .unwrap()
    }
}

/// A sampler state of `descriptor` with argument-buffer support, placed at `id`.
fn pixel_state_at(device: &Device, id: ResourceId, descriptor: &SamplerDescriptor) -> SamplerState {
    let mut options = SamplerStateOptions::default();
    options.support_argument_buffers = true;
    options.placement = IdPlacement::At(id);
    let state = device
        .new_sampler_state_with_options(descriptor, options)
        .unwrap();
    assert_eq!(state.gpu_resource_id(), id);
    state
}

/// Texel (0, 0) of [`HANDLE_SAMPLER`]'s texture, (x + 2y, 10 + x, 20 + y, 7), at each of its four
/// threads: its kernel's output through a state of its descriptor in pixel coordinates, each of
/// its coordinates being below 1.
fn corner() -> Vec<u8> {
    [0u32, 10, 20, 7]
        .repeat(4)
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect()
}

/// A sampler state that takes the ID a dispatch's argument buffer names, made after the dispatch
/// is encoded and before its command buffer commits, is reached through that ID, as Metal's
/// function reaches the state holding an ID as its command buffer runs (an M4: a state made after
/// encoding and written into the argument buffer before the commit is the one sampled; the state
/// named goes after encoding, one in pixel coordinates made before the commit takes its ID 2, and
/// the kernel reads texel (0, 0) through it): the case's state goes once the dispatch is encoded,
/// and one of its descriptor in pixel coordinates takes its ID, so the kernel reads texel (0, 0) at
/// each of its coordinates. The device's sampler table maps the ID to the new state as it takes
/// the ID's slot: nothing is announced.
#[test]
fn a_sampler_state_made_before_the_commit_is_reached_by_its_id() {
    on_device(|device, messages| {
        let mut sampling = Sampling::new(device);
        let key = *sampling
            .case
            .compute
            .argument_buffer_samplers
            .keys()
            .next()
            .unwrap();
        let resource = &sampling.case.compute.argument_buffer_samplers[&key];
        let mut descriptor = argument_sampler_descriptor(resource);
        assert!(descriptor.normalized_coordinates, "the case's state");
        descriptor.normalized_coordinates = false;
        assert_ne!(
            corner(),
            sampling.case.metal_output,
            "the case's state's answer differs"
        );
        let texture = sampling.texture().clone();
        let (command_buffer, buffers) = sampling.encode(device, &texture, Some(&texture), None);
        let own = sampling.base.argument_samplers.remove(&key).unwrap();
        let id = own.gpu_resource_id();
        drop(own);
        let mut options = SamplerStateOptions::default();
        options.support_argument_buffers = true;
        options.placement = IdPlacement::At(id);
        let state = device
            .new_sampler_state_with_options(&descriptor, options)
            .unwrap();
        assert_eq!(state.gpu_resource_id(), id);
        command_buffer.commit().unwrap();
        assert_eq!(
            sampling.output(&command_buffer, &buffers),
            corner(),
            "through ID {}",
            id.0
        );
        sampling.announced(device, messages, &[]);
        drop(state);
    });
}

/// A sampler state of a shape a dispatch's plan was not made for (a comparison), made after the
/// dispatch is encoded and before its command buffer commits, is in the device's heap sets and
/// sampler tables as any state is: nothing is announced, and the kernel reads the case's state as
/// Metal's does.
#[test]
fn a_sampler_state_of_a_new_shape_made_before_the_commit_is_listed() {
    on_device(|device, messages| {
        let sampling = Sampling::new(device);
        let texture = sampling.texture();
        let (command_buffer, buffers) = sampling.encode(device, texture, Some(texture), None);
        let mut options = SamplerStateOptions::default();
        options.support_argument_buffers = true;
        let mut descriptor = SamplerDescriptor::new();
        descriptor.compare_function = CompareFunction::Less;
        let state = device
            .new_sampler_state_with_options(&descriptor, options)
            .unwrap();
        command_buffer.commit().unwrap();
        let output = sampling.output(&command_buffer, &buffers);
        assert_eq!(output, sampling.case.metal_output);
        drop(state);
        sampling.announced(device, messages, &[]);
    });
}

/// `count` sampler states of descriptors of their own, made on `device` (none another's).
pub(super) fn distinct_sampler_states(device: &Device, count: u64) -> Vec<SamplerState> {
    (0..count)
        .map(|i| {
            let mut distinct = SamplerDescriptor::new();
            distinct.lod_min_clamp = (i + 1) as f32 / 65536.0;
            device.new_sampler_state(&distinct).unwrap()
        })
        .collect()
}

/// Sampler states made after a dispatch is encoded and before its command buffer commits are
/// reached by their IDs however many: with the case's states alone live at encoding (at most 11:
/// class 0, 16 entries), 20 more (past the class, within twice it) and 64 more (past twice it, the
/// most a table of the command's own held), each time followed by a state in pixel coordinates at
/// the ID the dispatch's argument buffer names, through which the kernel reads texel (0, 0).
/// Nothing is announced or counted (`WorkCounts::unlisted_argument_sampler_buffers`).
#[test]
fn late_sampler_states_past_twice_the_class_are_reached_by_their_ids() {
    on_device(|device, messages| {
        let unlisted = || device.work_counts().unlisted_argument_sampler_buffers;
        let before = unlisted();
        for more in [20, 64] {
            let mut sampling = Sampling::new(device);
            let live = sampling.base.argument_samplers.len() + sampling.base.samplers.len();
            assert!(
                live <= 11,
                "{live} states of the case: with 20 more, past class 0's 16 entries and within \
                 its 32"
            );
            let texture = sampling.texture().clone();
            let (command_buffer, buffers) = sampling.encode(device, &texture, Some(&texture), None);
            let (id, descriptor) = sampling.drop_case_sampler();
            let late = distinct_sampler_states(device, more);
            let state = pixel_state_at(device, id, &descriptor);
            command_buffer.commit().unwrap();
            assert_eq!(
                sampling.output(&command_buffer, &buffers),
                corner(),
                "{more} more states, then ID {}",
                id.0
            );
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            sampling.announced(device, messages, &[]);
            drop((late, state));
        }
        assert_eq!(unlisted(), before);
    });
}

/// Sampler states at IDs past the sampler tables' low window are reached by their IDs, as Metal's
/// are (the M3 Max under macOS 27.0.1: with sampler ID 100000 reserved, states took 100001 to
/// 100003 and sampled through an argument buffer as their address modes say). With 100000
/// reserved before anything takes a sampler ID, the case's states take IDs past it, and a state
/// in pixel coordinates standing in the argument buffer for the case's, at the next plain ID,
/// reads texel (0, 0), where the neutral sampler reads the case's own answer. The dispatch
/// samples through an argument buffer while that pixel state lives, which announces
/// `argument-pixel-sampler-scaling` and nothing else.
#[test]
fn sampler_states_past_the_low_window_are_reached_by_their_ids() {
    on_device(|device, messages| {
        device
            .reserve_resource_ids(ResourceIdKind::SamplerState, &[ResourceId(100_000)])
            .unwrap();
        let mut sampling = Sampling::new(device);
        let ids: Vec<u64> = sampling
            .base
            .argument_samplers
            .values()
            .map(|state| state.gpu_resource_id().0)
            .collect();
        assert!(
            ids.iter().all(|&id| id > 100_000),
            "the case's states past the reservation: {ids:?}"
        );
        let key = *sampling
            .case
            .compute
            .argument_buffer_samplers
            .keys()
            .next()
            .unwrap();
        let (_, descriptor) = sampling.drop_case_sampler();
        let mut options = SamplerStateOptions::default();
        options.support_argument_buffers = true;
        let state = device
            .new_sampler_state_with_options(&descriptor, options)
            .unwrap();
        let id = state.gpu_resource_id().0;
        assert!(id > 100_000, "a plain ID past the reservation: {id}");
        sampling.base.argument_samplers.insert(key, state);
        let texture = sampling.texture().clone();
        assert_eq!(
            sampling.run(device, &texture, Some(&texture)),
            corner(),
            "through ID {id}"
        );
        sampling.announced(device, messages, &PIXEL_ARGUMENT);
    });
}

/// A sampler state made without argument-buffer support has an ID too, and a kernel reaches it
/// through that ID in its argument buffer, as Metal's does (probe 7, an M3 Max and an M4: a
/// linear, repeating state sampled 6.99 and 6.99 either way).
#[test]
fn a_sampler_state_without_argument_buffer_support_is_reached_by_its_id() {
    on_device(|device, messages| {
        let mut sampling = Sampling::new(device);
        let key = *sampling
            .case
            .compute
            .argument_buffer_samplers
            .keys()
            .next()
            .unwrap();
        let resource = &sampling.case.compute.argument_buffer_samplers[&key];
        let state = device
            .new_sampler_state(&argument_sampler_descriptor(resource))
            .unwrap();
        let supporting = sampling
            .base
            .argument_samplers
            .insert(key, state.clone())
            .unwrap();
        assert!(state != supporting);
        assert_ne!(state.gpu_resource_id(), supporting.gpu_resource_id());
        let texture = sampling.texture().clone();
        let output = sampling.run(device, &texture, Some(&texture));
        assert_eq!(output, sampling.case.metal_output);
        sampling.announced(device, messages, &[]);
    });
}

/// The most sampler states [`a_sampler_state_at_any_id_is_listed`] makes to crowd a device's
/// or pipeline's set-1 sampler arrays.
const CROWD: u64 = 2048;

/// A sampler state at an ID past what a set-1 sampler array holds is listed at an element of
/// its own, and a function reaching its ID samples through it, as Metal's does: one at the highest
/// ID Metal makes (16380), in pixel coordinates and standing in the argument buffer for the case's
/// state, reads texel (0, 0) at all four of the case's coordinates (each below 1), as Metal's does
/// (the M3 Max: bound and through an argument buffer alike), where the neutral
/// sampler the state would miss to reads the case's own answer; the dispatch announces
/// `argument-pixel-sampler-scaling` alone, sampling through an argument buffer while the pixel
/// state lives. A state is
/// left out only where more states live than the arrays hold: the line `SAMPLER-CAPACITY` reports
/// the device's and the case's pipeline's, and where either holds at most [`CROWD`] elements the
/// lane makes states past it, announced once (`UNLISTED`), and the command buffer completes with
/// the state at the high ID listed, made before them.
#[test]
fn a_sampler_state_at_any_id_is_listed() {
    on_device(|device, messages| {
        let mut sampling = Sampling::new(device);
        let key = *sampling
            .case
            .compute
            .argument_buffer_samplers
            .keys()
            .next()
            .unwrap();
        let resource = &sampling.case.compute.argument_buffer_samplers[&key];
        let mut descriptor = argument_sampler_descriptor(resource);
        assert!(descriptor.normalized_coordinates, "the case's state");
        descriptor.normalized_coordinates = false;
        let high = ID_LIMIT - 1;
        let mut options = SamplerStateOptions::default();
        options.support_argument_buffers = true;
        options.placement = IdPlacement::At(ResourceId(high));
        let state = device
            .new_sampler_state_with_options(&descriptor, options)
            .unwrap();
        assert_eq!(state.gpu_resource_id(), ResourceId(high));
        let own = sampling
            .base
            .argument_samplers
            .insert(key, state.clone())
            .unwrap();
        let corner = corner();
        assert_ne!(
            corner, sampling.case.metal_output,
            "the neutral's answer differs"
        );
        let texture = sampling.texture().clone();
        let (command_buffer, buffers) = sampling.encode(device, &texture, Some(&texture), None);
        command_buffer.commit().unwrap();
        assert_eq!(
            sampling.output(&command_buffer, &buffers),
            corner,
            "through ID {high}"
        );
        sampling.announced(device, messages, &PIXEL_ARGUMENT);
        let [execution] = &command_buffer.executions()[..] else {
            panic!("one dispatch");
        };
        let ExecutedPipeline::Compute(pipeline) = execution.pipeline() else {
            panic!("a compute pipeline");
        };
        let set = &pipeline.layout.sets[1];
        let pipeline_capacity = set.bindings.iter().map(|b| b.count).min().unwrap();
        let sampler_arrays = set
            .bindings
            .iter()
            .filter(|b| b.ty == metal2vulkan::vk::DescriptorType::SAMPLER)
            .count();
        assert!(
            sampler_arrays > 0,
            "{HANDLE_SAMPLER} reads argument samplers"
        );
        let limits = device.compile_device().profile().limits();
        let device_capacity = limits
            .max_descriptor_set_update_after_bind_samplers
            .min(limits.max_per_stage_descriptor_update_after_bind_samplers);
        eprintln!(
            "SAMPLER-CAPACITY {}: pipeline set-1 capacity {pipeline_capacity} ({sampler_arrays} \
             sampler arrays), device sampler arrays {device_capacity}, crowd {CROWD}",
            device.name()
        );
        let capacity = u64::from(device_capacity.min(pipeline_capacity));
        if capacity <= CROWD {
            let crowd: Vec<SamplerState> = (0..capacity)
                .map(|i| {
                    let mut distinct = SamplerDescriptor::new();
                    distinct.lod_min_clamp = (i + 1) as f32 / 65536.0;
                    device.new_sampler_state(&distinct).unwrap()
                })
                .collect();
            let (command_buffer, buffers) = sampling.encode(device, &texture, Some(&texture), None);
            command_buffer.commit().unwrap();
            assert_eq!(
                sampling.output(&command_buffer, &buffers),
                corner,
                "with {capacity} more states"
            );
            assert_eq!(command_buffer.error(), None);
            sampling.announced(device, messages, &UNLISTED);
            drop(crowd);
        }
        drop((state, own));
    });
}

/// Sampler states are uniqued as Metal's are (probe 5): an equal request gets the live state,
/// whatever ID it asks for; a field changed, or argument-buffer support, gives another; a
/// placed state of its own has the ID it asked for; once the state goes, an equal request gets
/// a new one at the lowest free ID.
#[test]
fn sampler_states_are_uniqued_by_their_descriptors() {
    on_device(|device, _| {
        let options = |placement| {
            let mut options = SamplerStateOptions::default();
            options.support_argument_buffers = true;
            options.placement = placement;
            options
        };
        let made = |descriptor: &SamplerDescriptor, placement| {
            device
                .new_sampler_state_with_options(descriptor, options(placement))
                .unwrap()
        };
        let base = SamplerDescriptor::new();
        let first = made(&base, IdPlacement::Any);
        assert_eq!(first.gpu_resource_id(), ResourceId(1));
        assert!(made(&base, IdPlacement::Any) == first);
        assert!(
            made(&base, IdPlacement::At(ResourceId(40))) == first,
            "placed, but equal"
        );
        let mut linear = base;
        linear.min_filter = SamplerMinMagFilter::Linear;
        let other = made(&linear, IdPlacement::Any);
        assert!(other != first);
        assert_eq!(other.gpu_resource_id(), ResourceId(2));
        let without = device.new_sampler_state(&base).unwrap();
        assert!(without != first && without == device.new_sampler_state(&base).unwrap());
        let mut nearest_mip = base;
        nearest_mip.lod_max_clamp = 4.0;
        let placed = made(&nearest_mip, IdPlacement::At(ResourceId(40)));
        assert_eq!(placed.gpu_resource_id(), ResourceId(40));
        drop(first);
        let again = made(&base, IdPlacement::Any);
        assert_eq!(again.gpu_resource_id(), ResourceId(1));
        drop((other, placed, again, without));
    });
}

/// The device takes a sampler ID for itself as it makes its first command queue, indirect
/// command buffer, or texture whose memory is its own (shared or not), as Metal's does (the
/// consumer's M3 Max report at 701572bbf; the M4 probes): sampler states made after it take the IDs
/// past it, it stays taken once that object is gone, and a sampler reservation after it is ignored.
/// A heap's or a buffer's texture, a compute pipeline state, a render pipeline state without a
/// fragment function and a sampler state take none; a render state with one numbers it while it
/// lives (`numbered_samplers.rs`).
#[test]
fn the_device_s_first_queue_texture_or_icb_takes_a_sampler_id() {
    // Each state of a descriptor of its own: an equal one gets the live state.
    let clamp = std::cell::Cell::new(0.0f32);
    let state = |device: &Device| {
        let mut descriptor = SamplerDescriptor::new();
        clamp.set(clamp.get() + 1.0);
        descriptor.lod_min_clamp = clamp.get();
        let mut options = SamplerStateOptions::default();
        options.support_argument_buffers = true;
        device
            .new_sampler_state_with_options(&descriptor, options)
            .unwrap()
    };
    let id = |state: &SamplerState| state.gpu_resource_id().0;
    let plain = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 4, 4, false);
    // States made first take 1 and 2; the first texture takes 3 for the device. Each state is
    // kept: a state gone gives its ID back.
    on_device(|device, _| {
        let (one, two) = (state(device), state(device));
        assert_eq!([id(&one), id(&two)], [1, 2]);
        let texture = device.new_texture(&plain).unwrap();
        let four = state(device);
        assert_eq!(id(&four), 4);
        drop(texture);
        let five = state(device);
        assert_eq!(id(&five), 5, "the device's ID stays taken");
        drop((one, two, four, five));
    });
    // Each first object that takes one: the states start at 2.
    for first in [
        "texture, gone",
        "shared texture",
        "queue",
        "indirect command buffer",
    ] {
        on_device(|device, _| {
            let kept: Box<dyn std::any::Any> = match first {
                "texture, gone" => {
                    drop(device.new_texture(&plain).unwrap());
                    Box::new(())
                }
                "shared texture" => {
                    let mut shared = plain;
                    shared.resource_options = ResourceOptions::private();
                    Box::new(device.new_shared_texture(&shared).unwrap())
                }
                "queue" => Box::new(device.new_command_queue().unwrap()),
                _ => {
                    let mut descriptor = IndirectCommandBufferDescriptor::new();
                    descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH;
                    let options = ResourceOptions::shared();
                    Box::new(
                        device
                            .new_indirect_command_buffer(&descriptor, 4, options)
                            .unwrap(),
                    )
                }
            };
            assert_eq!(id(&state(device)), 2, "{first}");
            drop(kept);
        });
    }
    // None of these takes one: a heap's texture, a buffer's, a compute state, and a render state
    // without a fragment function.
    on_device(|device, _| {
        let private = ResourceOptions::private();
        let heap = device
            .new_heap(&HeapDescriptor::new(HeapType::Automatic, 1 << 16, private))
            .unwrap();
        let mut in_heap = plain;
        in_heap.resource_options = private;
        let heaps = heap.new_texture(&in_heap).unwrap();
        let buffer = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
        let buffers = buffer.new_texture(&plain, 0, 256).unwrap();
        let case = public_case(HANDLE_SAMPLER).unwrap();
        let compute = pipeline_state(device, &case).unwrap();
        let render = super::indirect_command_buffers::render_template_state(device, false);
        let one = state(device);
        assert_eq!(id(&one), 1);
        // A texture of its own then takes 2.
        let texture = device.new_texture(&plain).unwrap();
        assert_eq!(id(&state(device)), 3);
        drop((heaps, buffers, compute, render, texture, one));
    });
    // A sampler reservation once the device took its ID is ignored, as Metal ignores it (the
    // probe's `g1_tex_then_res2`): 2 is handed out.
    on_device(|device, _| {
        let texture = device.new_texture(&plain).unwrap();
        device
            .reserve_resource_ids(ResourceIdKind::SamplerState, &[ResourceId(2)])
            .unwrap();
        assert_eq!(id(&state(device)), 2);
        drop(texture);
    });
}

/// A command queue or an indirect command buffer made before a texture reservation leaves the
/// texture pool's first range beside the reservation's, as Metal's device does (an M4, and the
/// consumer's M3 Max): after a queue, a reservation of 64 to 66 hands out 1, 2 and 3, where it
/// hands out 67 on a fresh device. After a buffer and a reservation of 20, Metal hands out 21 to
/// 31, 1 to 15, then 1 to 15 again while those textures live, and so does the runtime (the
/// consumer's G331), announcing nothing: two live textures answer each of 1 to 15, and an ID both
/// give back comes back once.
#[test]
fn a_queue_or_buffer_made_before_a_texture_reservation_keeps_the_pool_s_first_range() {
    let plain = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 4, 4, false);
    let textures = |device: &Device, count: usize| -> Vec<Texture> {
        (0..count)
            .map(|_| device.new_texture(&plain).unwrap())
            .collect()
    };
    let ids = |textures: &[Texture]| -> Vec<u64> {
        textures.iter().map(|t| t.gpu_resource_id().0).collect()
    };
    on_device(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        drop(queue);
        let reserved = [64, 65, 66].map(ResourceId);
        device
            .reserve_resource_ids(ResourceIdKind::Texture, &reserved)
            .unwrap();
        let made = textures(device, 4);
        assert_eq!(ids(&made), [1, 2, 3, 4]);
        assert_once(&messages.take_notices(), &[]);
    });
    on_device(|device, messages| {
        let mut descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH;
        let _buffer = device
            .new_indirect_command_buffer(&descriptor, 1, ResourceOptions::shared())
            .unwrap();
        device
            .reserve_resource_ids(ResourceIdKind::Texture, &[ResourceId(20)])
            .unwrap();
        let mut made = textures(device, 45);
        let want: Vec<u64> = (21..=31)
            .chain(1..=15)
            .chain(1..=15)
            .chain(16..=19)
            .collect();
        assert_eq!(ids(&made), want);
        // Both holders of 5 go (positions 15 and 30): 5 comes back once, then 32.
        assert_eq!((ids(&made[15..16]), ids(&made[30..31])), (vec![5], vec![5]));
        made.remove(30);
        made.remove(15);
        let next = textures(device, 2);
        assert_eq!(ids(&next), [5, 32]);
        assert_once(&messages.take_notices(), &[]);
    });
    // Made after the reservation, a queue changes nothing.
    on_device(|device, messages| {
        let reserved = [64, 65, 66].map(ResourceId);
        device
            .reserve_resource_ids(ResourceIdKind::Texture, &reserved)
            .unwrap();
        let _queue = device.new_command_queue().unwrap();
        let made = textures(device, 3);
        assert_eq!(ids(&made), [67, 68, 69]);
        assert_once(&messages.take_notices(), &[]);
    });
}

/// Plain IDs after a reservation are the ones Metal hands out, for textures, sampler states,
/// pipeline states and indirect command buffers, and an ID given back joins its free neighbours
/// (each sequence measured on an M4, those of indirect command buffers past 2^23 on an M3 Max):
/// only a kind's first reservation with an ID counts, and IDs past Metal's placement limit are
/// taken.
#[test]
fn plain_ids_after_a_reservation_are_metal_s() {
    let plain = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 4, 4, false);
    let runs = |runs: &[(u64, u64)]| -> Vec<u64> {
        runs.iter()
            .flat_map(|&(first, last)| first..=last)
            .collect()
    };
    let textures = |device: &Device, count: usize| -> Vec<Texture> {
        (0..count)
            .map(|_| device.new_texture(&plain).unwrap())
            .collect()
    };
    let ids = |textures: &[Texture]| -> Vec<u64> {
        textures.iter().map(|t| t.gpu_resource_id().0).collect()
    };
    // One reserved texture ID inside the pool's second half: 11 to 15 first (`o1_0_10`); a
    // second call is ignored, even before any texture was made (`w_16_5`). A call of no ID, of
    // 0 alone, or holding an ID of 2^32 or more (dropped whole) is not the reservation
    // (`empty_first`, `y_high_and_10`, `y_high_then_10`), but the call of 0 alone leaves the pool
    // made, so its first range's IDs are handed out twice: 1 after 9, where the M4's
    // `y_zero_then_10` handed out 16 (an M3 Max, macOS 27.0.1, this very sequence).
    on_device(|device, _| {
        let reserve = |ids: &[u64]| {
            let ids: Vec<ResourceId> = ids.iter().copied().map(ResourceId).collect();
            device.reserve_resource_ids(ResourceIdKind::Texture, &ids)
        };
        reserve(&[]).unwrap();
        reserve(&[0]).unwrap();
        reserve(&[(1 << 32) + 10, 12]).unwrap();
        reserve(&[10]).unwrap();
        reserve(&[12]).unwrap();
        let made = textures(device, 17);
        assert_eq!(ids(&made), runs(&[(11, 15), (1, 9), (1, 3)]));
    });
    // IDs past Metal's placement limit (`hi_0_16380`, `multi_0`): 16383 is handed out.
    on_device(|device, _| {
        let reserved = [ResourceId(2), ResourceId(16382)];
        device
            .reserve_resource_ids(ResourceIdKind::Texture, &reserved)
            .unwrap();
        let made = textures(device, 4);
        assert_eq!(ids(&made), [1, 16383, 3, 4]);
    });
    // Given back: 2, 3 and 5 of 1 to 8 come back as 5, 2, 3, then 9 (`c_plain_235`).
    on_device(|device, _| {
        let mut made: Vec<Option<Texture>> = textures(device, 8).into_iter().map(Some).collect();
        for at in [1, 2, 4] {
            made[at] = None;
        }
        let again = textures(device, 4);
        assert_eq!(ids(&again), [5, 2, 3, 9]);
    });
    // Sampler states (`b1_a`), made before anything takes the device's sampler ID.
    on_device(|device, _| {
        let reserved = [3, 11, 25].map(ResourceId);
        device
            .reserve_resource_ids(ResourceIdKind::SamplerState, &reserved)
            .unwrap();
        let states: Vec<SamplerState> = (0..10)
            .map(|i| {
                let mut descriptor = SamplerDescriptor::new();
                descriptor.lod_min_clamp = i as f32;
                let mut options = SamplerStateOptions::default();
                options.support_argument_buffers = true;
                device
                    .new_sampler_state_with_options(&descriptor, options)
                    .unwrap()
            })
            .collect();
        let got: Vec<u64> = states.iter().map(|s| s.gpu_resource_id().0).collect();
        assert_eq!(got, runs(&[(1, 2), (26, 31), (4, 5)]));
    });
    // Indirect command buffers, from a pool of 32 (`i_17_20`), and a pipeline state made with
    // `supportIndirectCommandBuffers` (`o1_4_17`).
    on_device(|device, _| {
        device
            .reserve_resource_ids(
                ResourceIdKind::IndirectCommandBuffer,
                &[ResourceId(17), ResourceId(20)],
            )
            .unwrap();
        device
            .reserve_resource_ids(ResourceIdKind::PipelineState, &[ResourceId(17)])
            .unwrap();
        let mut descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH;
        let buffers: Vec<_> = (0..4)
            .map(|_| {
                device
                    .new_indirect_command_buffer(&descriptor, 1, ResourceOptions::shared())
                    .unwrap()
            })
            .collect();
        let got: Vec<u64> = buffers.iter().map(|b| b.gpu_resource_id().0).collect();
        assert_eq!(got, [18, 19, 21, 22]);
        let state = super::indirect_command_buffers::render_template_state(device, true);
        assert_eq!(state.gpu_resource_id(), ResourceId(18));
    });
    // Refused, reserving nothing: a texture or sampler state ID of 2^28 or more, which ended
    // Metal's process (`x_0_2p28p5`, `x_1_2p28p5`), and an indirect command buffer ID of 2^31 to
    // 2^32 - 1, at which Metal's reservation never returned (an M3 Max), as such.
    on_device(|device, _| {
        let hangs = |id| Problem::ReservationHangs {
            id,
            limit: RECORDS_ID_LIMIT,
        };
        let refused = [
            (
                ResourceIdKind::Texture,
                1 << 28,
                Problem::ResourceId {
                    id: 1 << 28,
                    limit: 1 << 28,
                },
            ),
            (
                ResourceIdKind::SamplerState,
                1 << 28,
                Problem::ResourceId {
                    id: 1 << 28,
                    limit: 1 << 28,
                },
            ),
            (
                ResourceIdKind::IndirectCommandBuffer,
                RECORDS_ID_LIMIT,
                hangs(RECORDS_ID_LIMIT),
            ),
            (
                ResourceIdKind::IndirectCommandBuffer,
                (1 << 32) - 1,
                hangs((1 << 32) - 1),
            ),
        ];
        for (kind, id, expected) in refused {
            let error = device
                .reserve_resource_ids(kind, &[ResourceId(id)])
                .unwrap_err();
            let Error::Invalid(Invalid { problem, .. }) = error else {
                panic!("{kind:?}: {error:?}");
            };
            assert_eq!(problem, expected);
        }
        // Nothing was reserved: 10 is the reservation still.
        device
            .reserve_resource_ids(ResourceIdKind::Texture, &[ResourceId(10)])
            .unwrap();
        assert_eq!(
            device.new_texture(&plain).unwrap().gpu_resource_id(),
            ResourceId(11)
        );
    });
    // Plain indirect command buffer IDs past a reservation are the ones Metal hands out, each
    // buffer made (an M3 Max, macOS 27.0.1): 2^23 + 5, 16,000,000, 2^30 + 5 and 3 * 2^29 give the
    // next IDs, 2^31 - 2 gives 2^31 - 1, then 1, and 2^23, 2^30 - 1, 2^30 and 2^31 - 1 give 1
    // (`x_5_2p23`).
    let mut icb = IndirectCommandBufferDescriptor::new();
    icb.command_types = IndirectCommandType::CONCURRENT_DISPATCH;
    let made: [(u64, &[u64]); 9] = [
        ((1 << 23) + 5, &[(1 << 23) + 6, (1 << 23) + 7]),
        (16_000_000, &[16_000_001, 16_000_002, 16_000_003]),
        (1 << 23, &[1, 2]),
        ((1 << 30) - 1, &[1]),
        (1 << 30, &[1]),
        ((1 << 30) + 5, &[(1 << 30) + 6]),
        (3 << 29, &[(3 << 29) + 1]),
        (RECORDS_ID_LIMIT - 2, &[RECORDS_ID_LIMIT - 1, 1]),
        (RECORDS_ID_LIMIT - 1, &[1, 2]),
    ];
    for (reserved, ids) in made {
        on_device(|device, _| {
            device
                .reserve_resource_ids(
                    ResourceIdKind::IndirectCommandBuffer,
                    &[ResourceId(reserved)],
                )
                .unwrap();
            let buffers: Vec<IndirectCommandBuffer> = ids
                .iter()
                .map(|_| {
                    device
                        .new_indirect_command_buffer(&icb, 1, ResourceOptions::shared())
                        .unwrap()
                })
                .collect();
            let got: Vec<u64> = buffers.iter().map(|b| b.gpu_resource_id().0).collect();
            assert_eq!(got, ids, "{reserved}");
        });
    }
    // A late call (after a buffer, or after a reservation) is ignored whatever it holds, as
    // Metal ignores and logs it (an M3 Max, macOS 27.0.1): IDs of 2^31 and up, refused in a first
    // call, 2^32 + 5, dropped in a first call, and 0 alone; textures' 2^28 + 5 too.
    let late = [
        (
            ResourceIdKind::IndirectCommandBuffer,
            None,
            RECORDS_ID_LIMIT + 5,
            [2, 3],
        ),
        (
            ResourceIdKind::IndirectCommandBuffer,
            None,
            (1 << 32) - 1,
            [2, 3],
        ),
        (
            ResourceIdKind::IndirectCommandBuffer,
            Some(5),
            RECORDS_ID_LIMIT,
            [1, 2],
        ),
        (
            ResourceIdKind::IndirectCommandBuffer,
            None,
            (1 << 32) + 5,
            [2, 3],
        ),
        (ResourceIdKind::IndirectCommandBuffer, None, 0, [2, 3]),
        (ResourceIdKind::Texture, None, (1 << 28) + 5, [2, 3]),
        (ResourceIdKind::Texture, Some(10), (1 << 28) + 5, [11, 12]),
    ];
    for (kind, first, held, next) in late {
        on_device(|device, _| {
            // Each kept alive, so its ID stays held.
            let (mut textures, mut buffers) = (Vec::new(), Vec::new());
            let mut make = || match kind {
                ResourceIdKind::Texture => {
                    let texture = device.new_texture(&plain).unwrap();
                    let id = texture.gpu_resource_id().0;
                    textures.push(texture);
                    id
                }
                _ => {
                    let buffer = device
                        .new_indirect_command_buffer(&icb, 1, ResourceOptions::shared())
                        .unwrap();
                    let id = buffer.gpu_resource_id().0;
                    buffers.push(buffer);
                    id
                }
            };
            match first {
                Some(id) => device
                    .reserve_resource_ids(kind, &[ResourceId(id)])
                    .unwrap(),
                None => assert_eq!(make(), 1),
            }
            assert_eq!(
                device.reserve_resource_ids(kind, &[ResourceId(held)]),
                Ok(()),
                "{kind:?}: {held} after {first:?}"
            );
            let got = [make(), make()];
            assert_eq!(got, next, "{kind:?}: {held} after {first:?}");
        });
    }
    // After a call of 0 alone, a reservation keeps the pool's first range beside its own, whose
    // IDs Metal hands out twice (an M3 Max, macOS 27.0.1): 0 then 10 gives 1 to 9, 11 to 31,
    // then 1 to 15; 0 then 20, 21 to 31, 1 to 19, then 1 to 10.
    for (reserved, runs) in [
        (10, &[(1, 9), (11, 31), (1, 15)]),
        (20, &[(21, 31), (1, 19), (1, 10)]),
    ] {
        on_device(|device, _| {
            for ids in [0, reserved] {
                device
                    .reserve_resource_ids(ResourceIdKind::IndirectCommandBuffer, &[ResourceId(ids)])
                    .unwrap();
            }
            let want: Vec<u64> = runs
                .iter()
                .flat_map(|&(first, last)| first..=last)
                .collect();
            let buffers: Vec<IndirectCommandBuffer> = want
                .iter()
                .map(|_| {
                    device
                        .new_indirect_command_buffer(&icb, 1, ResourceOptions::shared())
                        .unwrap()
                })
                .collect();
            let got: Vec<u64> = buffers.iter().map(|b| b.gpu_resource_id().0).collect();
            assert_eq!(got, want, "0, then {reserved}");
        });
    }
}

/// What holds an indirect command buffer's ID once the program releases the buffer, as plain
/// Metal holds it (on an M4, the `h_` cases): a compute execution and a compute
/// or render `useResource` until their command buffer completes, and nothing once it is dropped
/// uncommitted; a blit's copy from it or into it, its reset and its optimize not at all, nor a
/// command buffer that does not use it. Each command buffer waits on an event while the buffer
/// is released and another is made: that one takes the released ID where nothing holds it.
#[test]
fn an_indirect_command_buffer_s_id_is_held_as_metal_holds_it() {
    #[derive(Clone, Copy, Debug)]
    enum Use {
        Execute,
        ComputeResidency,
        RenderResidency,
        CopyFrom,
        CopyInto,
        Reset,
        Optimize,
        Nothing,
    }
    // Each use, whether its command buffer is committed, and whether it holds the ID.
    let cases = [
        (Use::Execute, true, true),
        (Use::ComputeResidency, true, true),
        (Use::RenderResidency, true, true),
        (Use::CopyFrom, true, false),
        (Use::CopyInto, true, false),
        (Use::Reset, true, false),
        (Use::Optimize, true, false),
        (Use::Nothing, true, false),
        (Use::Execute, false, false),
        (Use::ComputeResidency, false, false),
    ];
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH;
    for (usage, commit, holds) in cases {
        on_device(|device, _| {
            let made = || {
                device
                    .new_indirect_command_buffer(&descriptor, 4, ResourceOptions::shared())
                    .unwrap()
            };
            let partner = matches!(usage, Use::CopyFrom | Use::CopyInto).then(made);
            let used = made();
            let id = used.gpu_resource_id();
            let event = device.new_shared_event().unwrap();
            let queue = device.new_command_queue().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            command_buffer
                .encode_wait_for_event(EventRef::from(&event), 1)
                .unwrap();
            let blit = || command_buffer.blit_command_encoder().unwrap();
            match usage {
                Use::Execute => {
                    let mut encoder = command_buffer.compute_command_encoder().unwrap();
                    encoder.execute_commands_in_buffer(&used, 0..1).unwrap();
                    encoder.end_encoding().unwrap();
                }
                Use::ComputeResidency => {
                    let mut encoder = command_buffer.compute_command_encoder().unwrap();
                    encoder
                        .use_resource(used.as_resource(), ResourceUsage::READ)
                        .unwrap();
                    encoder.end_encoding().unwrap();
                }
                Use::RenderResidency => {
                    let mut pass = RenderPassDescriptor::new();
                    pass.render_target_width = 2;
                    pass.render_target_height = 2;
                    pass.default_raster_sample_count = 1;
                    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
                    encoder
                        .use_resource(
                            used.as_resource(),
                            ResourceUsage::READ,
                            RenderStages::VERTEX,
                        )
                        .unwrap();
                    encoder.end_encoding().unwrap();
                }
                Use::CopyFrom | Use::CopyInto => {
                    let partner = partner.as_ref().unwrap();
                    let (from, to) = match usage {
                        Use::CopyFrom => (&used, partner),
                        _ => (partner, &used),
                    };
                    let mut encoder = blit();
                    encoder
                        .copy_indirect_command_buffer(from, 0..1, to, 0)
                        .unwrap();
                    encoder.end_encoding().unwrap();
                }
                Use::Reset => {
                    let mut encoder = blit();
                    encoder.reset_commands_in_buffer(&used, 0..1).unwrap();
                    encoder.end_encoding().unwrap();
                }
                Use::Optimize => {
                    let mut encoder = blit();
                    encoder
                        .optimize_indirect_command_buffer(&used, 0..1)
                        .unwrap();
                    encoder.end_encoding().unwrap();
                }
                Use::Nothing => {}
            }
            if commit {
                command_buffer.commit().unwrap();
            }
            drop(used);
            let committed = commit.then_some(command_buffer);
            let next = ResourceId(id.0 + 1);
            let [waiting, after] = if holds { [next, id] } else { [id, next] };
            let while_waiting = made();
            assert_eq!(
                while_waiting.gpu_resource_id(),
                waiting,
                "{usage:?}, committed {commit}: while its command buffer waits"
            );
            event.set_signaled_value(1);
            if let Some(command_buffer) = &committed {
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            }
            assert_eq!(
                made().gpu_resource_id(),
                after,
                "{usage:?}, committed {commit}: once it completed"
            );
            drop((while_waiting, partner));
        });
    }
}

/// An indirect command buffer placed at an ID has it, as Metal places it (on an M4, plain and under
/// API validation): the index's low 32 bits, plain allocation where those are 0; plain allocation
/// neither skips a placed ID nor is kept from a later reservation by it, which a placement makes
/// the kind's first allocation; a placement at 16384 or more, past Metal's table, is refused typed.
#[test]
fn indirect_command_buffers_are_placed_as_metal_places_them() {
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH;
    let made = |device: &Device, placement: IdPlacement| {
        let mut placed = descriptor;
        placed.placement = placement;
        device.new_indirect_command_buffer(&placed, 4, ResourceOptions::shared())
    };
    let at = |id: u64| IdPlacement::At(ResourceId(id));
    on_device(|device, _| {
        let buffers: Vec<_> = [at((1 << 32) + 5), IdPlacement::Any, at(1 << 32), at(16383)]
            .into_iter()
            .map(|placement| made(device, placement).unwrap())
            .collect();
        let ids: Vec<u64> = buffers.iter().map(|b| b.gpu_resource_id().0).collect();
        assert_eq!(ids, [5, 1, 2, 16383]);
        let error = made(device, at(16384)).unwrap_err();
        let Error::Invalid(Invalid { problem, .. }) = error else {
            panic!("{error:?}");
        };
        let limit = 16384;
        assert_eq!(problem, Problem::ResourceId { id: limit, limit });
    });
    // Plain, plain, placed at 3, then plain: 1, 2, 3, then 3 and 4 (`p_plain_skips_nothing`).
    on_device(|device, _| {
        let buffers: Vec<_> = [IdPlacement::Any, IdPlacement::Any, at(3)]
            .into_iter()
            .chain([IdPlacement::Any; 2])
            .map(|placement| made(device, placement).unwrap())
            .collect();
        let ids: Vec<u64> = buffers.iter().map(|b| b.gpu_resource_id().0).collect();
        assert_eq!(ids, [1, 2, 3, 3, 4]);
    });
    // Placed at 9, then a reservation of 2, ignored: 1 and 2 are handed out (`p_then_reserve`).
    on_device(|device, _| {
        let placed = made(device, at(9)).unwrap();
        device
            .reserve_resource_ids(ResourceIdKind::IndirectCommandBuffer, &[ResourceId(2)])
            .unwrap();
        let plain: Vec<_> = (0..2)
            .map(|_| made(device, IdPlacement::Any).unwrap())
            .collect();
        let ids: Vec<u64> = plain.iter().map(|b| b.gpu_resource_id().0).collect();
        assert_eq!(ids, [1, 2]);
        drop(placed);
    });
}

/// Placed IDs Metal ended the process for are refused typed, for every kind and call; one the
/// device's heap arrays cannot hold is refused `Unsupported`; a placement at 0 is plain
/// allocation; a view placed at an ID has it. A reservation of such an ID is taken
/// ([`plain_ids_after_a_reservation_are_metal_s`]).
#[test]
fn placed_ids_past_the_limits_are_refused_typed() {
    on_device(|device, _| {
        let sampling = Sampling::new(device);
        let limit = ResourceId(ID_LIMIT);
        let refused = |error: Error| {
            let Error::Invalid(Invalid { problem, .. }) = error else {
                panic!("{error:?}");
            };
            let at_limit = Problem::ResourceId {
                id: ID_LIMIT,
                limit: ID_LIMIT,
            };
            assert_eq!(problem, at_limit);
        };
        let mut descriptor = texture_descriptor(&sampling.literal, sampling.usage).unwrap();
        descriptor.placement = IdPlacement::At(limit);
        refused(device.new_texture(&descriptor).unwrap_err());
        let texture = sampling.texture();
        refused(
            texture
                .new_texture_view_with_placement(PixelFormat::RGBA32Uint, IdPlacement::At(limit))
                .unwrap_err(),
        );
        let mut options = SamplerStateOptions::default();
        options.support_argument_buffers = true;
        options.placement = IdPlacement::At(limit);
        refused(
            device
                .new_sampler_state_with_options(&SamplerDescriptor::new(), options)
                .unwrap_err(),
        );
        // Below the limit: placed, or past a set-1 array the texture's usage binds it through.
        let high = ResourceId(ID_LIMIT - 1);
        descriptor.placement = IdPlacement::At(high);
        match device.new_texture(&descriptor) {
            Ok(texture) => assert_eq!(texture.gpu_resource_id(), high),
            Err(Error::Unsupported(unsupported)) => {
                assert_eq!(unsupported.feature, Feature::ArgumentBufferTextures)
            }
            Err(other) => panic!("{other:?}"),
        }
        // A texture no shader binds is held by no array.
        let mut target = descriptor;
        target.usage = TextureUsage::RENDER_TARGET;
        assert_eq!(device.new_texture(&target).unwrap().gpu_resource_id(), high);
        descriptor.placement = IdPlacement::At(ResourceId(0));
        let plain = device.new_texture(&descriptor).unwrap();
        assert_ne!(plain.gpu_resource_id(), ResourceId(0));
        let view = texture
            .new_texture_view_with_placement(
                PixelFormat::RGBA32Uint,
                IdPlacement::At(ResourceId(50)),
            )
            .unwrap();
        assert_eq!(view.gpu_resource_id(), ResourceId(50));
        for (swizzle, id) in [(None, 51), (Some(TextureSwizzleChannels::default()), 52)] {
            let subview = texture
                .new_texture_subview_with_placement(
                    PixelFormat::RGBA32Uint,
                    TextureType::Type2D,
                    0..1,
                    0..1,
                    swizzle,
                    IdPlacement::At(ResourceId(id)),
                )
                .unwrap();
            assert_eq!(subview.gpu_resource_id(), ResourceId(id));
        }
        let mut shared = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 4, 4, false);
        shared.usage = TextureUsage::SHADER_READ;
        shared.resource_options = ResourceOptions::private();
        shared.placement = IdPlacement::At(ResourceId(53));
        let original = device.new_shared_texture(&shared).unwrap();
        assert_eq!(original.gpu_resource_id(), ResourceId(53));
        let handle = original.new_shared_texture_handle().unwrap();
        let opened = device
            .new_shared_texture_with_handle(&handle, IdPlacement::At(ResourceId(54)))
            .unwrap();
        assert_eq!(opened.gpu_resource_id(), ResourceId(54));
        refused(
            device
                .new_shared_texture_with_handle(&handle, IdPlacement::At(limit))
                .unwrap_err(),
        );
    });
}

/// A texture `new_texture` makes [`IdPlacement::Outside`] takes no resource ID (the consumer's
/// helper textures), in the consumer's sequence: after a reservation of 20000 it answers 0, and a
/// plain swizzled view of it takes 20001, the ID plain `new_texture` gives on a device that made a
/// plain texture there instead (a view placed `At` 20001 is refused, as Metal ends the process for
/// a placement of 16381 or more). It takes the device's sampler ID, so the sampler states after it
/// are numbered as after the plain texture. It starts no space: a reservation after it counts.
/// `Outside` anywhere else is refused `Invalid(OutsidePlacement)`.
#[test]
fn an_outside_texture_takes_no_id() {
    let plain = TextureDescriptor::texture_2d(PixelFormat::RGBA16Float, 4, 4, false);
    let mut outside = plain;
    outside.placement = IdPlacement::Outside;
    let reserve = |device: &Device, id: u64| {
        device
            .reserve_resource_ids(ResourceIdKind::Texture, &[ResourceId(id)])
            .unwrap()
    };
    // Two sampler states of descriptors of their own, made after the textures, kept by the
    // caller: a state gone gives its ID back.
    let states = |device: &Device| -> Vec<SamplerState> {
        (0..2)
            .map(|i| {
                let mut descriptor = SamplerDescriptor::new();
                descriptor.lod_min_clamp = 1.0 + i as f32;
                let mut options = SamplerStateOptions::default();
                options.support_argument_buffers = true;
                device
                    .new_sampler_state_with_options(&descriptor, options)
                    .unwrap()
            })
            .collect()
    };
    let ids = |states: &[SamplerState]| -> Vec<u64> {
        states.iter().map(|s| s.gpu_resource_id().0).collect()
    };
    let swizzled = |texture: &Texture, placement| {
        texture.new_texture_subview_with_placement(
            PixelFormat::RGBA16Float,
            TextureType::Type2D,
            0..1,
            0..1,
            Some(TextureSwizzleChannels::default()),
            placement,
        )
    };
    let refused = |error: Error, want: Problem| {
        let Error::Invalid(Invalid { problem, .. }) = error else {
            panic!("{error:?}");
        };
        assert_eq!(problem, want);
    };
    // Plain `new_texture` after the reservation: the texture takes 20001.
    let mut plain_ids = (0, Vec::new());
    on_device(|device, _| {
        reserve(device, 20000);
        let texture = device.new_texture(&plain).unwrap();
        let kept = states(device);
        plain_ids = (texture.gpu_resource_id().0, ids(&kept));
    });
    assert_eq!(plain_ids.0, 20001);
    // `Outside` in its place: 0, and its plain view takes 20001; a view placed at 20001 is
    // refused at Metal's placement limit.
    on_device(|device, _| {
        reserve(device, 20000);
        let shadow = device.new_texture(&outside).unwrap();
        assert_eq!(shadow.gpu_resource_id(), ResourceId(0));
        refused(
            swizzled(&shadow, IdPlacement::At(ResourceId(20001))).unwrap_err(),
            Problem::ResourceId {
                id: 20001,
                limit: ID_LIMIT,
            },
        );
        let view = swizzled(&shadow, IdPlacement::Any).unwrap();
        assert_eq!(view.gpu_resource_id().0, plain_ids.0);
        let kept = states(device);
        assert_eq!(ids(&kept), plain_ids.1, "the sampler states after it");
        // Another plain texture takes the next plain ID, as after the plain texture.
        let next = device.new_texture(&plain).unwrap();
        assert_eq!(next.gpu_resource_id(), ResourceId(20002));
        // `Outside` anywhere else is refused.
        let outside_refused = |error: Error| refused(error, Problem::OutsidePlacement);
        outside_refused(swizzled(&shadow, IdPlacement::Outside).unwrap_err());
        outside_refused(
            shadow
                .new_texture_view_with_placement(PixelFormat::RGBA16Float, IdPlacement::Outside)
                .unwrap_err(),
        );
        let mut options = SamplerStateOptions::default();
        options.support_argument_buffers = true;
        options.placement = IdPlacement::Outside;
        outside_refused(
            device
                .new_sampler_state_with_options(&SamplerDescriptor::new(), options)
                .unwrap_err(),
        );
        let heap = device
            .new_heap(&HeapDescriptor::new(
                HeapType::Automatic,
                1 << 16,
                ResourceOptions::private(),
            ))
            .unwrap();
        let mut in_heap = outside;
        in_heap.resource_options = ResourceOptions::private();
        outside_refused(heap.new_texture(&in_heap).unwrap_err());
        let mut descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH;
        descriptor.placement = IdPlacement::Outside;
        outside_refused(
            device
                .new_indirect_command_buffer(&descriptor, 4, ResourceOptions::shared())
                .unwrap_err(),
        );
    });
    // It starts no space: a reservation after it counts, and the next plain texture takes 20001.
    on_device(|device, _| {
        let shadow = device.new_texture(&outside).unwrap();
        reserve(device, 20000);
        let texture = device.new_texture(&plain).unwrap();
        assert_eq!(
            [shadow.gpu_resource_id().0, texture.gpu_resource_id().0],
            [0, 20001]
        );
    });
}

/// What [`reservations_keep_ids_from_plain_allocation`]'s sink keeps: the late reservation's
/// diagnostic lines, the validation layer's failing messages, and the notices.
type Seen = (Vec<DebugMessage>, Vec<DebugMessage>, Vec<DebugMessage>);

/// Reserved IDs stay out of plain allocation (probe 6: Metal's plain IDs after a reservation are
/// an allocator detail; only that is kept), and a resource placed at one has it. A reservation
/// once the device handed out an ID of its kind is ignored with one diagnostic line, as Metal's
/// is, and neither refused nor a degradation.
#[test]
fn reservations_keep_ids_from_plain_allocation() {
    let messages = Messages::default();
    // The device's sink keeps the reservation's diagnostic lines, what the validation layer
    // reported that fails a run (the rule `Messages` applies: a set layout's report after the
    // device announced `set-past-layout-support` passes, `candidate/unsupported_set.rs`), and the
    // notices.
    let seen: Arc<Mutex<Seen>> = Arc::default();
    let cache = ScratchCache::new().unwrap();
    {
        let keep = Arc::clone(&seen);
        // The device's creation notices reach this sink, not the lane's: they are checked here.
        let device = made_lane_device(TEST_DEVICE, &messages, cache.path(), |options| {
            options.messages = Some(MessageSink::new(move |message: &DebugMessage| {
                let mut seen = keep.lock().unwrap_or_else(|p| p.into_inner());
                if message.id_name == LATE_RESERVATION {
                    seen.0.push(message.clone());
                } else if message.degradation.is_some() {
                    seen.2.push(message.clone());
                } else if message.severity == MessageSeverity::Error
                    || (message.severity == MessageSeverity::Warning && message.kind.validation)
                {
                    let announced = seen.2.iter().any(|notice| {
                        notice.degradation.as_ref().is_some_and(|notice| {
                            notice.degradation == Degradation::SetPastLayoutSupport
                        })
                    });
                    if !(announced && unsupported_set::shape(&message.id_name, &message.text)) {
                        seen.1.push(message.clone());
                    }
                }
            }));
        })
        .unwrap();
        let created = seen.lock().unwrap().2.clone();
        creation_checked(&device, &created).unwrap();
        let kind = ResourceIdKind::Texture;
        device
            .reserve_resource_ids(kind, &[ResourceId(2), ResourceId(4)])
            .unwrap();
        let sampling = Sampling::new(&device);
        assert_eq!(sampling.texture().gpu_resource_id(), ResourceId(1));
        let plain: Vec<Texture> = (0..2)
            .map(|_| sampling.inverted(&device, IdPlacement::Any))
            .collect();
        let ids: Vec<ResourceId> = plain.iter().map(Texture::gpu_resource_id).collect();
        assert_eq!(ids, [ResourceId(3), ResourceId(5)]);
        let placed = sampling.inverted(&device, IdPlacement::At(ResourceId(2)));
        assert_eq!(placed.gpu_resource_id(), ResourceId(2));
        assert!(seen.lock().unwrap().0.is_empty());
        device.reserve_resource_ids(kind, &[ResourceId(6)]).unwrap();
        let next = sampling.inverted(&device, IdPlacement::Any);
        assert_eq!(
            next.gpu_resource_id(),
            ResourceId(6),
            "the late reservation is ignored"
        );
        let lines = seen.lock().unwrap().0.clone();
        let [line] = &lines[..] else {
            panic!("one diagnostic line: {lines:#?}");
        };
        assert_eq!(line.severity, MessageSeverity::Warning);
        assert!(line.kind.general && line.degradation.is_none(), "{line:?}");
        drop((sampling, plain, placed, next));
    }
    let failing = seen.lock().unwrap().1.clone();
    assert!(
        failing.is_empty(),
        "the validation layer reported: {failing:#?}"
    );
}
