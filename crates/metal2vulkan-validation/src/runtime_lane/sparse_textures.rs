//! Sparse heaps and their textures on a device: a sparse heap counts the tiles mapped into its
//! textures; a mapping, bound before the work encoded after it in the same command buffer, holds
//! what is written to it; an unmapped tile reads zero on a device with strict residency, and a
//! device without it announces `non-strict-residency` once ([`SPARSE`]); a single 16-KiB tile maps
//! on a device whose blocks are 64 KiB, the unmapped tiles of its block reading zero on every
//! device after a blit's or a render pass's writes; an indirect update maps what its buffer holds
//! when it runs, which the host reads (`host-read-mapping-arguments`, once); a move takes mappings,
//! and their texels, to unmapped tiles, of the same heap or another, counting nothing; a
//! multisampled texture reads zero where unmapped, as Metal's; committed sparse kernel cases read
//! through the runtime what Metal read. A device that binds no sparse
//! memory emulates sparse textures over ordinary images where its functions read tile maps
//! (MoltenVK), every test here running on them; elsewhere it refuses the heap typed, naming what it
//! lacks.

use super::degrade_notices::{assert_made_once, assert_once, forced, on_device};
use super::kernel_textures::census;
use super::resolves::{color_pass, quiet, FULL, WHITE};
use super::{public_cases, read_buffer, shared_kinds, NamedCase};
use crate::case::Stage;
use ash::vk;
use metal2vulkan::bind::TileMapping;
use metal2vulkan::device::DriverFact;
use metal2vulkan::device::VkFeature;
use metal2vulkan::error::Feature;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan_runtime::{
    BlitOption, Buffer, ClearColor, CommandBuffer, CommandBufferError, CommandQueue, Degradation,
    Device, Error, Heap, HeapDescriptor, HeapType, Invalid, LoadAction, Origin, Problem, Region,
    ResourceOptions, RuntimeGap, Size, SparseProblem, StoreAction, Texture, TextureDescriptor,
};
use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

const WAIT: Duration = Duration::from_secs(60);

/// A 16-KiB page: the heap's tile.
const PAGE: u64 = 16384;

/// The kinds sparse textures take: a device without strict residency announces the first when it
/// makes one; an indirect mapping update announces the second; a device whose functions read
/// residency from tile maps announces the third when it makes one, an ordinary image.
pub(super) const SPARSE: [Degradation; 3] = [
    Degradation::NonStrictResidency,
    Degradation::HostReadMappingArguments,
    Degradation::EmulatedSparseTexture,
];

/// What `device` makes a sparse texture of `pf` (of 2D images) of, read through Vulkan apart from
/// the runtime: a sparse image where it binds sparse memory (`sparseBinding` and
/// `sparseResidencyImage2D`, `sparseResidency2Samples` or `4Samples` for a multisampled one, and
/// its queue family's sparse-binding bit) and lists the format's sparse image properties of the
/// colour aspect, else an ordinary image whose unmapped tiles the runtime keeps at zero (MoltenVK);
/// and whether a sparse image's residency is `residencyNonResidentStrict`.
#[derive(Clone, Copy)]
pub(super) struct SparseFacts {
    /// A sparse image, not an ordinary one (`emulated-sparse-texture`).
    pub(super) image: bool,
    /// An unmapped tile reads zero: an ordinary image, or a sparse image of strict residency.
    pub(super) strict: bool,
}

impl SparseFacts {
    /// The facts of `device` for a single-sampled sparse texture of `pf`.
    pub(super) fn of(device: &Device, pf: PixelFormat) -> SparseFacts {
        SparseFacts::of_samples(device, pf, 1)
    }

    /// The facts of `device` for a sparse texture of `pf` of 1, 2 or 4 `samples`, a multisampled
    /// one a render target (the runtime makes its image an attachment).
    fn of_samples(device: &Device, pf: PixelFormat, samples: u32) -> SparseFacts {
        let format = metal2vulkan::metal::format::direct_format(pf).expect("a sparse format");
        let format = vk::Format::from_raw(format.as_raw());
        let mut usage = vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST;
        if samples > 1 {
            usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
        }
        let count = vk::SampleCountFlags::from_raw(samples);
        // SAFETY: instance-level queries of the live device's physical device.
        let (features, sparse, family, formats) = unsafe {
            let raw = device.raw();
            let (instance, physical) = (raw.instance, raw.physical_device);
            let families = instance.get_physical_device_queue_family_properties(physical);
            (
                instance.get_physical_device_features(physical),
                instance
                    .get_physical_device_properties(physical)
                    .sparse_properties,
                families[raw.queue_family as usize].queue_flags,
                instance.get_physical_device_sparse_image_format_properties(
                    physical,
                    format,
                    vk::ImageType::TYPE_2D,
                    count,
                    usage,
                    vk::ImageTiling::OPTIMAL,
                ),
            )
        };
        let residency = match samples {
            1 => features.sparse_residency_image2_d,
            2 => features.sparse_residency2_samples,
            _ => features.sparse_residency4_samples,
        };
        let binds = features.sparse_binding == vk::TRUE
            && residency == vk::TRUE
            && family.contains(vk::QueueFlags::SPARSE_BINDING);
        let listed = (formats.iter()).any(|f| f.aspect_mask.contains(vk::ImageAspectFlags::COLOR));
        let image = binds && listed;
        SparseFacts {
            image,
            strict: !image || sparse.residency_non_resident_strict == vk::TRUE,
        }
    }

    /// The kinds the device's first sparse texture of these announces as it is made: an ordinary
    /// image's `emulated-sparse-texture`, a sparse image's `non-strict-residency` where its
    /// residency is not strict, and its `first-sparse-resource-wait` where the driver's first
    /// sparse resource waits for every queue (`DriverFact::FirstSparseResourceWaitsForQueues`).
    pub(super) fn made(&self, device: &Device) -> BTreeSet<Degradation> {
        let waits = (device.properties().driver_facts)
            .contains(&DriverFact::FirstSparseResourceWaitsForQueues);
        forced(&[
            (Degradation::EmulatedSparseTexture, !self.image),
            (Degradation::NonStrictResidency, !self.strict),
            (Degradation::FirstSparseResourceWait, self.image && waits),
        ])
    }
}

fn run(command_buffer: &CommandBuffer) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// Whether `error` is the typed refusal of a device without what a sparse texture needs: the
/// device's lack, or, where it binds no sparse memory, the runtime's gap of the tile map's
/// ordinary images.
pub(super) fn lacks_sparse(error: &Error) -> bool {
    matches!(error, Error::Unsupported(u) if matches!(u.feature, Feature::SparseTextures { .. }))
        || *error == Error::NotImplemented(RuntimeGap::SparseTileMaps)
}

/// The extent of `level` of a 256x256 texture.
fn side(level: u64) -> u64 {
    (256u64 >> level).max(1)
}

/// Level `level`'s texels: `1 + x + 256 * y + 65536 * level`, as floats.
fn texels(level: u64) -> Vec<f32> {
    let n = side(level);
    (0..n * n)
        .map(|i| (1 + i % n + 256 * (i / n) + 65536 * level) as f32)
        .collect()
}

/// A Shared buffer holding `values`.
fn buffer_of(device: &Device, values: &[f32]) -> Buffer {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buffer = device
        .new_buffer(bytes.len() as u64, ResourceOptions::shared())
        .unwrap();
    let contents = buffer.contents().expect("a host-visible buffer");
    // SAFETY: the buffer holds `bytes.len()` bytes at `contents`, and no command uses them now.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
    buffer
}

/// Encode copies of each of `levels`' texels ([`texels`]) into `texture`.
fn write_levels(command_buffer: &CommandBuffer, texture: &Texture, levels: &[u64]) {
    let device = command_buffer.device();
    let sources: Vec<(u64, Buffer)> = levels
        .iter()
        .map(|&level| (level, buffer_of(device, &texels(level))))
        .collect();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    for (level, source) in &sources {
        let n = side(*level);
        blit.copy_from_buffer_to_texture(
            source,
            0,
            n * 4,
            n * n * 4,
            Size::new(n, n, 1),
            texture,
            0,
            *level,
            Origin::new(0, 0, 0),
            BlitOption::None,
        )
        .unwrap();
    }
    blit.end_encoding().unwrap();
}

/// `level` of `texture`, read back by a blit.
fn read_level(queue: &CommandQueue, texture: &Texture, level: u64) -> Vec<f32> {
    let n = side(level);
    let out = queue
        .device()
        .new_buffer(n * n * 4, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        level,
        Origin::new(0, 0, 0),
        Size::new(n, n, 1),
        &out,
        0,
        n * 4,
        n * n * 4,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    run(&command_buffer);
    read_buffer(&out, 0, n * n * 4)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect()
}

/// One command buffer of a resource state encoder's `updates`, each `(level, region)` of slice 0
/// in one `update_texture_mappings`, then `then`'s encoders: the update's answer, the command
/// buffer run when it was taken.
fn update(
    queue: &CommandQueue,
    texture: &Texture,
    mode: TileMapping,
    updates: &[(u64, Region)],
    then: impl FnOnce(&CommandBuffer),
) -> Result<(), Error> {
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.resource_state_command_encoder().unwrap();
    let levels: Vec<u64> = updates.iter().map(|(level, _)| *level).collect();
    let regions: Vec<Region> = updates.iter().map(|(_, region)| *region).collect();
    let answer =
        encoder.update_texture_mappings(texture, mode, &regions, &levels, &vec![0; levels.len()]);
    encoder.end_encoding().unwrap();
    then(&command_buffer);
    run(&command_buffer);
    answer
}

/// The texels of `level` inside the tiles `[0, tiles)` (64x64 texels each) along x and y.
fn inside(level: u64, tiles: u64, i: usize) -> bool {
    let n = side(level) as usize;
    let edge = (tiles * 64) as usize;
    i % n < edge && i / n < edge
}

/// The tile of level 0's texel `i` (64x64 texels each).
fn tile_of(i: usize) -> [u64; 2] {
    [(i % 256 / 64) as u64, (i / 256 / 64) as u64]
}

/// A 256x256 `R32Float` texture of 9 levels from a 1-MiB sparse heap of 16-KiB pages: Metal's
/// 64x64 tiles, its tail from level 3 in one page; mapped tiles hold what a blit after the mapping
/// in the same command buffer writes, unmapped ones read zero where residency is strict; a single
/// tile maps on every device that makes the texture, its block's unmapped tiles reading zero on
/// every device, and an unmap of one tile of a block keeps the others; levels 1 to 8 and the tail,
/// mapped in one update, hold their writes; the tail unmaps alone, keeping level 2; an unmap
/// returns the tiles to the heap.
#[test]
fn mapped_tiles_hold_writes_and_unmapped_tiles_read_zero() {
    on_device(|device, messages| {
        let name = device.name().to_owned();
        let descriptor = HeapDescriptor::new(HeapType::Sparse, 1 << 20, ResourceOptions::private());
        let heap: Heap = match device.new_heap(&descriptor) {
            Ok(heap) => heap,
            Err(error) if lacks_sparse(&error) => {
                println!("SPARSE {name}: refused {error:?}");
                return;
            }
            Err(error) => panic!("a sparse heap on {name}: {error:?}"),
        };
        assert_eq!(
            (heap.heap_type(), heap.size(), heap.used_size()),
            (HeapType::Sparse, 1 << 20, 0)
        );
        assert_eq!(
            device.sparse_tile_size(TextureType::Type2D, PixelFormat::R32Float, 1),
            Size::new(64, 64, 1)
        );
        let mut d = TextureDescriptor::texture_2d(PixelFormat::R32Float, 256, 256, false);
        d.mipmap_level_count = 9;
        d.usage = TextureUsage::SHADER_READ;
        d.resource_options = ResourceOptions::private();
        let texture = match heap.new_texture(&d) {
            Ok(texture) => texture,
            Err(error) if lacks_sparse(&error) => {
                println!("SPARSE {name}: texture refused {error:?}");
                return;
            }
            Err(error) => panic!("a sparse texture on {name}: {error:?}"),
        };
        let facts = SparseFacts::of(device, PixelFormat::R32Float);
        assert_made_once(messages, &[], &facts.made(device));
        let strict = facts.strict;
        // An ordinary image only where the device makes no sparse image, which never reads the
        // image's own residency, announced once: a second texture announces nothing.
        let emulated = !facts.image;
        let residency = device
            .enabled()
            .features
            .contains(&VkFeature::ShaderResourceResidency);
        assert!(!(emulated && residency), "emulated and residency on {name}");
        drop(heap.new_texture(&d).unwrap());
        assert_eq!(assert_once(&messages.take_notices(), &[]), BTreeMap::new());
        assert!(texture.is_sparse());
        assert_eq!(texture.first_mipmap_in_tail(), Some(3));
        assert_eq!(texture.tail_size_in_bytes(), Some(16384));
        assert_eq!(heap.used_size(), 0);

        // The top-left 2x2 tiles of level 0, then every texel of level 0 written after the
        // mapping in the same command buffer.
        let queue = device.new_command_queue().unwrap();
        let top_left = Region::new_2d(0, 0, 2, 2);
        update(&queue, &texture, TileMapping::Map, &[(0, top_left)], |cb| {
            write_levels(cb, &texture, &[0])
        })
        .unwrap();
        assert_eq!(heap.used_size(), 4 * PAGE);
        let want = texels(0);
        let level0 = read_level(&queue, &texture, 0);
        for (i, (&got, &written)) in level0.iter().zip(&want).enumerate() {
            if inside(0, 2, i) {
                assert_eq!(got, written, "a mapped texel {i} on {name}");
            } else if strict {
                assert_eq!(got, 0.0, "an unmapped texel {i} on {name}");
            }
        }

        // A single tile maps on every device that makes the texture, and so does its neighbour,
        // tiles (2, 0) and (3, 0), of one 64-KiB block where the device's blocks are: what a blit
        // writes after them lands there, and the unmapped tiles (2, 1) and (3, 1) of their block
        // read zero on every device, Metal's answer (the runtime keeps them at zero).
        let single = |x| [(0, Region::new_2d(x, 0, 1, 1))];
        update(&queue, &texture, TileMapping::Map, &single(2), |_| {}).unwrap();
        update(&queue, &texture, TileMapping::Map, &single(3), |cb| {
            write_levels(cb, &texture, &[0])
        })
        .unwrap();
        assert_eq!(heap.used_size(), 6 * PAGE);
        let held = |tile: [u64; 2], i| inside(0, 2, i) || tile == [2, 0] || tile == [3, 0];
        let level0 = read_level(&queue, &texture, 0);
        for (i, (&got, &written)) in level0.iter().zip(&want).enumerate() {
            let tile = tile_of(i);
            if held(tile, i) {
                assert_eq!(got, written, "a mapped texel {i} on {name}");
            } else if strict || tile == [2, 1] || tile == [3, 1] {
                assert_eq!(got, 0.0, "an unmapped texel {i} of tile {tile:?} on {name}");
            }
        }
        // An unmap of one of them keeps the other's texels; the unmapped one reads zero.
        update(&queue, &texture, TileMapping::Unmap, &single(2), |_| {}).unwrap();
        assert_eq!(heap.used_size(), 5 * PAGE);
        let level0 = read_level(&queue, &texture, 0);
        for (i, (&got, &written)) in level0.iter().zip(&want).enumerate() {
            match tile_of(i) {
                [3, 0] => assert_eq!(got, written, "a kept texel {i} on {name}"),
                [2, 0] | [2, 1] | [3, 1] => {
                    assert_eq!(got, 0.0, "an unmapped texel {i} on {name}")
                }
                _ => {}
            }
        }
        let level0_pages = 5;

        // Levels 1 and 2 and the tail in one update: whole device units whatever its blocks.
        let rest = [
            (1, Region::new_2d(0, 0, 2, 2)),
            (2, Region::new_2d(0, 0, 1, 1)),
            (3, Region::new_2d(0, 0, 1, 1)),
        ];
        let levels: Vec<u64> = (1..9).collect();
        update(&queue, &texture, TileMapping::Map, &rest, |cb| {
            write_levels(cb, &texture, &levels)
        })
        .unwrap();
        assert_eq!(heap.used_size(), (level0_pages + 4 + 1 + 1) * PAGE);
        for &level in &levels {
            assert_eq!(
                read_level(&queue, &texture, level),
                texels(level),
                "level {level} on {name}"
            );
        }

        // The tail alone unmaps on every device, as one page; level 2, which a device's own tail
        // may hold with it, keeps its texels, and the tail's levels read zero where residency is
        // strict.
        update(
            &queue,
            &texture,
            TileMapping::Unmap,
            &[(3, Region::new_2d(0, 0, 0, 0))],
            |_| {},
        )
        .unwrap();
        let mapped = level0_pages + 4 + 1;
        assert_eq!(heap.used_size(), mapped * PAGE);
        assert_eq!(read_level(&queue, &texture, 2), texels(2), "{name}");
        if strict {
            for level in 3..9 {
                let zero = vec![0.0; texels(level).len()];
                assert_eq!(read_level(&queue, &texture, level), zero, "{name}");
            }
        }

        // Unmapping level 0's top-left tiles returns them to the heap; they read zero after.
        update(
            &queue,
            &texture,
            TileMapping::Unmap,
            &[(0, top_left)],
            |_| {},
        )
        .unwrap();
        assert_eq!(heap.used_size(), (mapped - 4) * PAGE);
        if strict {
            let level0 = read_level(&queue, &texture, 0);
            for (i, &got) in level0.iter().enumerate() {
                if inside(0, 2, i) {
                    assert_eq!(got, 0.0, "an unmapped texel {i} on {name}");
                }
            }
        }
        println!("SPARSE {name}: strict {strict}, emulated {emulated}");

        // A released texture's tiles stay counted in the heap, as Metal's.
        drop(texture);
        assert_eq!(heap.used_size(), (mapped - 4) * PAGE);
        // Nothing but, without unified memory, the first `Shared` buffer's.
        assert_made_once(messages, &[], &shared_kinds(device));
    });
}

/// The device's sparse heap of `size` bytes, or `None`, printed, where the device has no sparse
/// binding.
fn sparse_heap(device: &Device, size: u64) -> Option<Heap> {
    let descriptor = HeapDescriptor::new(HeapType::Sparse, size, ResourceOptions::private());
    match device.new_heap(&descriptor) {
        Ok(heap) => Some(heap),
        Err(error) if lacks_sparse(&error) => {
            println!("SPARSE {}: refused {error:?}", device.name());
            None
        }
        Err(error) => panic!("a sparse heap on {}: {error:?}", device.name()),
    }
}

/// A 256x256 `R32Float` texture of one level (4x4 tiles) from `heap`.
fn level_texture(heap: &Heap, pixel_format: PixelFormat) -> Texture {
    let mut d = TextureDescriptor::texture_2d(pixel_format, 256, 256, false);
    d.usage = TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::private();
    heap.new_texture(&d).unwrap()
}

/// A buffer of `options` holding `words`, written through a `Shared` copy.
fn words_buffer(queue: &CommandQueue, words: &[u32], options: ResourceOptions) -> Buffer {
    let device = queue.device();
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let shared = device
        .new_buffer(bytes.len() as u64, ResourceOptions::shared())
        .unwrap();
    let contents = shared.contents().expect("a host-visible buffer");
    // SAFETY: the buffer holds `bytes.len()` bytes at `contents`, and no command uses them now.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
    if options == ResourceOptions::shared() {
        return shared;
    }
    let buffer = device.new_buffer(bytes.len() as u64, options).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_buffer(&shared, 0, &buffer, 0, bytes.len() as u64)
        .unwrap();
    blit.end_encoding().unwrap();
    run(&command_buffer);
    buffer
}

/// One command buffer of an indirect update of `texture` from `buffer`, then `then`'s encoders,
/// committed: its error.
fn indirect(
    queue: &CommandQueue,
    texture: &Texture,
    mode: TileMapping,
    buffer: &Buffer,
    then: impl FnOnce(&CommandBuffer),
) -> Option<CommandBufferError> {
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.resource_state_command_encoder().unwrap();
    encoder
        .update_texture_mapping_indirect(texture, mode, buffer, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    then(&command_buffer);
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    command_buffer.error()
}

/// Level 0 of `texture` after a write of every texel: which of its 4x4 tiles hold what was
/// written (the rest read zero where residency is strict, which this checks).
fn mapped_tiles(queue: &CommandQueue, texture: &Texture, strict: bool) -> Vec<[u64; 2]> {
    let command_buffer = queue.command_buffer().unwrap();
    write_levels(&command_buffer, texture, &[0]);
    run(&command_buffer);
    let want = texels(0);
    let got = read_level(queue, texture, 0);
    let mut tiles = Vec::new();
    for ty in 0..4 {
        for tx in 0..4 {
            let at = |i: usize| (ty * 64 + i / 64) * 256 + tx * 64 + i % 64;
            let held = (0..64 * 64).all(|i| got[at(i)] == want[at(i)]);
            let zero = (0..64 * 64).all(|i| got[at(i)] == 0.0);
            assert!(held || zero || !strict, "tile ({tx}, {ty}) half written");
            if held {
                tiles.push([tx as u64, ty as u64]);
            }
        }
    }
    tiles
}

/// The 2x2 tiles from `(x, y)`.
fn quad(x: u64, y: u64) -> Vec<[u64; 2]> {
    vec![[x, y], [x + 1, y], [x, y + 1], [x + 1, y + 1]]
}

fn sorted(mut tiles: Vec<[u64; 2]>) -> Vec<[u64; 2]> {
    tiles.sort_by_key(|&[x, y]| (y, x));
    tiles
}

/// An indirect update maps or unmaps the regions its buffer holds when it runs, a `Shared` one
/// or a `Private` one a blit wrote just before in the same command buffer: a count, then that
/// many regions, as an M4 reads them; a count of 0 maps nothing; a single tile maps on every device
/// that makes the texture. Arguments Metal's validation stops map as plain Metal maps them; a
/// count word past the buffer is refused where encoded; an offset off 4 reads at the multiple of 4
/// below it, as plain Metal reads it. The device announces the host's reads once.
#[test]
fn indirect_updates_map_what_the_buffer_holds_when_they_run() {
    on_device(|device, messages| {
        let name = device.name().to_owned();
        let Some(heap) = sparse_heap(device, 1 << 20) else {
            return;
        };
        let texture = level_texture(&heap, PixelFormat::R32Float);
        let facts = SparseFacts::of(device, PixelFormat::R32Float);
        assert_made_once(messages, &[], &facts.made(device));
        let strict = facts.strict;
        let queue = device.new_command_queue().unwrap();
        let first_shared = shared_kinds(device);
        let shared = ResourceOptions::shared();
        let private = ResourceOptions::private();

        // The top-left 2x2 tiles, then a write after them in the same command buffer.
        let top_left = words_buffer(&queue, &[1, 0, 0, 0, 2, 2, 1, 0, 0], shared);
        let error = indirect(&queue, &texture, TileMapping::Map, &top_left, |cb| {
            write_levels(cb, &texture, &[0])
        });
        assert_eq!(error, None, "{name}");
        assert_eq!(heap.used_size(), 4 * PAGE);
        let level0 = read_level(&queue, &texture, 0);
        for (i, (&got, &written)) in level0.iter().zip(&texels(0)).enumerate() {
            if inside(0, 2, i) {
                assert_eq!(got, written, "a mapped texel {i} on {name}");
            } else if strict {
                assert_eq!(got, 0.0, "an unmapped texel {i} on {name}");
            }
        }
        assert_made_once(
            messages,
            &[Degradation::HostReadMappingArguments],
            &first_shared,
        );

        // A count of 0 maps nothing; a Private buffer a blit writes in the same command buffer
        // unmaps what the blit wrote.
        let none = words_buffer(&queue, &[0], shared);
        assert_eq!(
            indirect(&queue, &texture, TileMapping::Unmap, &none, |_| {}),
            None
        );
        assert_eq!(heap.used_size(), 4 * PAGE);
        let written = device.new_buffer(36, private).unwrap();
        let source = words_buffer(&queue, &[1, 0, 0, 0, 2, 2, 1, 0, 0], shared);
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_buffer(&source, 0, &written, 0, 36)
            .unwrap();
        blit.end_encoding().unwrap();
        let mut encoder = command_buffer.resource_state_command_encoder().unwrap();
        encoder
            .update_texture_mapping_indirect(&texture, TileMapping::Unmap, &written, 0)
            .unwrap();
        encoder.end_encoding().unwrap();
        run(&command_buffer);
        assert_eq!(heap.used_size(), 0, "{name}");

        // Arguments Metal's validation stops map as plain Metal maps them (the M3 Max under macOS
        // 27.0.1): a region past its level's tiles maps the tiles inside them; one whose origin is
        // past them, of no size, or of a level or slice past the texture's maps nothing; and the
        // regions a count names past the buffer map nothing. Each unmapped again after.
        let map = |mode, buffer: &Buffer| indirect(&queue, &texture, mode, buffer, |_| {});
        let past = words_buffer(&queue, &[1, 3, 0, 0, 2, 1, 1, 0, 0], shared);
        assert_eq!(map(TileMapping::Map, &past), None, "{name}");
        assert_eq!(heap.used_size(), PAGE, "{name}");
        if strict {
            assert_eq!(
                mapped_tiles(&queue, &texture, strict),
                vec![[3, 0]],
                "{name}"
            );
        }
        assert_eq!(map(TileMapping::Unmap, &past), None, "{name}");
        assert_eq!(heap.used_size(), 0, "{name}");
        let nothing = words_buffer(
            &queue,
            &[
                4, 4, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
                1, 1, 1, 0, 1,
            ],
            shared,
        );
        assert_eq!(map(TileMapping::Map, &nothing), None, "{name}");
        assert_eq!(heap.used_size(), 0, "{name}");
        let counted = words_buffer(&queue, &[2, 0, 0, 0, 1, 1, 1, 0, 0], private);
        assert_eq!(map(TileMapping::Map, &counted), None, "{name}");
        assert_eq!(heap.used_size(), PAGE, "{name}");
        assert_eq!(map(TileMapping::Unmap, &counted), None, "{name}");
        assert_eq!(heap.used_size(), 0, "{name}");
        // A single tile.
        let single = words_buffer(&queue, &[1, 2, 0, 0, 1, 1, 1, 0, 0], shared);
        let error = indirect(&queue, &texture, TileMapping::Map, &single, |cb| {
            write_levels(cb, &texture, &[0])
        });
        assert_eq!(error, None, "a single indirect tile on {name}");
        assert_eq!(heap.used_size(), PAGE);
        let level0 = read_level(&queue, &texture, 0);
        for (i, (&got, &written)) in level0.iter().zip(&texels(0)).enumerate() {
            match tile_of(i) {
                [2, 0] => assert_eq!(got, written, "a mapped texel {i} on {name}"),
                [3, 0] | [2, 1] | [3, 1] => {
                    assert_eq!(got, 0.0, "an unmapped texel {i} on {name}")
                }
                _ if strict => assert_eq!(got, 0.0, "an unmapped texel {i} on {name}"),
                _ => {}
            }
        }

        // An offset off 4 reads at the multiple of 4 below it: the top-left tiles again. A count
        // past the buffer is refused where encoded, from such an offset too.
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.resource_state_command_encoder().unwrap();
        let call = "update_texture_mapping_indirect";
        let refused = |problem| Error::Invalid(Invalid::new(call, Problem::Sparse(problem)));
        encoder
            .update_texture_mapping_indirect(&texture, TileMapping::Map, &top_left, 2)
            .unwrap();
        for offset in [36, 39] {
            assert_eq!(
                encoder.update_texture_mapping_indirect(
                    &texture,
                    TileMapping::Map,
                    &top_left,
                    offset
                ),
                Err(refused(SparseProblem::IndirectArguments {
                    offset: 36,
                    bytes: 4,
                    length: 36
                }))
            );
        }
        encoder.end_encoding().unwrap();
        run(&command_buffer);
        assert_eq!(heap.used_size(), 5 * PAGE, "{name}");
        let notices = assert_once(&messages.take_notices(), &[]);
        assert!(notices.is_empty(), "announced again: {notices:?}");
    });
}

/// Assert that level 0 `after` holds, in the 2x2 tiles from `to`, what level 0 `before` held in the
/// 2x2 tiles from `from`: a move's texels go with it, as Metal's do for a texture without
/// GPU-optimized contents.
fn assert_moved(after: &[f32], to: [u64; 2], before: &[f32], from: [u64; 2], what: &str) {
    for y in 0..128u64 {
        for x in 0..128u64 {
            let at = |[tx, ty]: [u64; 2]| ((ty * 64 + y) * 256 + tx * 64 + x) as usize;
            assert_eq!(
                after[at(to)],
                before[at(from)],
                "{what}: texel ({x}, {y}) of the tiles from {to:?}"
            );
        }
    }
}

/// A move takes each mapped source tile to its destination tile where that one is unmapped, as an
/// M4's does: onto mapped tiles nothing moves; between textures of one heap or of two, whose counts
/// do not change. What a Metal sparse probe measured on an M3 Max and an M4 moves too: within one
/// texture over overlapping regions, one tile at a time in Metal's order (only the tiles that land
/// on unmapped ones), between formats of different tiles tile for tile, and a slice's tail at
/// `firstMipmapInTail`, onto another's tail or onto a tile and back. Each moved tile's texels go
/// with it, on every device that makes the textures, whatever its blocks, as Metal's do for a
/// texture without GPU-optimized contents. Fences order the encoder like any other.
#[test]
fn moves_take_mappings_to_unmapped_tiles() {
    on_device(|device, messages| {
        let name = device.name().to_owned();
        let (Some(heap), Some(other_heap)) =
            (sparse_heap(device, 1 << 20), sparse_heap(device, 1 << 20))
        else {
            return;
        };
        let r32 = PixelFormat::R32Float;
        let (a, b) = (level_texture(&heap, r32), level_texture(&heap, r32));
        let c = level_texture(&other_heap, r32);
        let facts = SparseFacts::of(device, r32);
        assert_made_once(messages, &[], &facts.made(device));
        let strict = facts.strict;
        let queue = device.new_command_queue().unwrap();
        let map = |texture: &Texture, x, y| {
            update(
                &queue,
                texture,
                TileMapping::Map,
                &[(0, Region::new_2d(x, y, 2, 2))],
                |_| {},
            )
            .unwrap()
        };
        map(&a, 0, 0);
        map(&b, 2, 0);
        assert_eq!(heap.used_size(), 8 * PAGE);
        let command_buffer = queue.command_buffer().unwrap();
        write_levels(&command_buffer, &a, &[0]);
        write_levels(&command_buffer, &b, &[0]);
        run(&command_buffer);
        let level0 = |texture: &Texture| read_level(&queue, texture, 0);
        let moved = |from: &Texture, origin: Origin, to: &Texture, destination: Origin| {
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.resource_state_command_encoder().unwrap();
            let size = Size::new(2, 2, 1);
            let answer =
                encoder.move_texture_mappings(from, 0, 0, origin, size, to, 0, 0, destination);
            encoder.end_encoding().unwrap();
            run(&command_buffer);
            answer
        };
        let (a0, b0) = (level0(&a), level0(&b));
        // Onto mapped tiles: nothing moves.
        moved(&a, Origin::new(0, 0, 0), &b, Origin::new(2, 0, 0)).unwrap();
        // Onto unmapped tiles of the same heap, then of another heap, the texels with them.
        moved(&a, Origin::new(0, 0, 0), &b, Origin::new(0, 2, 0)).unwrap();
        moved(&b, Origin::new(2, 0, 0), &c, Origin::new(0, 0, 0)).unwrap();
        assert_eq!(heap.used_size(), 8 * PAGE, "{name}");
        assert_eq!(other_heap.used_size(), 0, "{name}");
        let b1 = level0(&b);
        assert_moved(&b1, [0, 2], &a0, [0, 0], &format!("a to b on {name}"));
        assert_moved(
            &level0(&c),
            [0, 0],
            &b0,
            [2, 0],
            &format!("b to c on {name}"),
        );

        // Shifted by one tile, splitting the device's blocks of both textures.
        moved(&b, Origin::new(0, 2, 0), &a, Origin::new(1, 1, 0)).unwrap();
        assert_moved(
            &level0(&a),
            [1, 1],
            &b1,
            [0, 2],
            &format!("shifted on {name}"),
        );
        if strict {
            assert_eq!(mapped_tiles(&queue, &a, strict), sorted(quad(1, 1)));
            assert_eq!(mapped_tiles(&queue, &b, strict), Vec::<[u64; 2]>::new());
            assert_eq!(mapped_tiles(&queue, &c, strict), sorted(quad(0, 0)));
        }

        // Within one texture over overlapping regions: (0, 0) onto the mapped (1, 0) stays, (1, 0)
        // moves onto (2, 0), and so on the next row; the tiles left unmapped in the blocks that
        // stay bound read zero on every device.
        let d = level_texture(&heap, r32);
        map(&d, 0, 0);
        let command_buffer = queue.command_buffer().unwrap();
        write_levels(&command_buffer, &d, &[0]);
        run(&command_buffer);
        let d0 = level0(&d);
        moved(&d, Origin::new(0, 0, 0), &d, Origin::new(1, 0, 0)).unwrap();
        let d1 = level0(&d);
        for (i, (&got, &was)) in d1.iter().zip(&d0).enumerate() {
            let what = format!("texel {i} of the overlapping move on {name}");
            match tile_of(i) {
                [0, 0] | [0, 1] => assert_eq!(got, was, "{what}"),
                [2, y] if y < 2 => assert_eq!(got, d0[i - 64], "{what}"),
                [1, y] | [3, y] if y < 2 => assert_eq!(got, 0.0, "{what}"),
                _ if strict => assert_eq!(got, 0.0, "{what}"),
                _ => {}
            }
        }
        // Between formats of different tiles, tile for tile: 64x64 `R32Float` tiles onto 128x64
        // `RG8Unorm` ones.
        let e = level_texture(&heap, r32);
        map(&e, 0, 0);
        let wide = level_texture(&heap, PixelFormat::RG8Unorm);
        moved(&e, Origin::new(0, 0, 0), &wide, Origin::new(0, 0, 0)).unwrap();
        if strict {
            assert_eq!(mapped_tiles(&queue, &e, strict), Vec::<[u64; 2]>::new());
        }
        assert_eq!(heap.used_size(), 16 * PAGE, "{name}");

        // A slice's tail at `firstMipmapInTail` (level 3 of 9): the destination's tail holds the
        // source's texels, the source's reads zero.
        let mut d9 = TextureDescriptor::texture_2d(r32, 256, 256, false);
        d9.mipmap_level_count = 9;
        d9.usage = TextureUsage::SHADER_READ;
        d9.resource_options = ResourceOptions::private();
        let (g, h) = (
            heap.new_texture(&d9).unwrap(),
            heap.new_texture(&d9).unwrap(),
        );
        let tail = [(3, Region::new_2d(0, 0, 1, 1))];
        update(&queue, &g, TileMapping::Map, &tail, |cb| {
            write_levels(cb, &g, &[3, 4])
        })
        .unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.resource_state_command_encoder().unwrap();
        let size = Size::new(1, 1, 1);
        let at = Origin::new(0, 0, 0);
        encoder
            .move_texture_mappings(&g, 0, 3, at, size, &h, 0, 3, at)
            .unwrap();
        encoder.end_encoding().unwrap();
        run(&command_buffer);
        for level in [3, 4] {
            assert_eq!(read_level(&queue, &h, level), texels(level), "{name}");
            if strict {
                let zero = vec![0.0; texels(level).len()];
                assert_eq!(read_level(&queue, &g, level), zero, "{name}");
            }
        }
        assert_eq!(heap.used_size(), 17 * PAGE, "{name}");

        // A tail onto a tiled level's tile and back, as Metal's validation takes them and Metal
        // moves them: the mappings move and the heap's count stays; the destination reads zero, the
        // runtime's near-equivalent of Metal's moved bytes in the other layout.
        let moved_level = |from: &Texture, from_level: u64, to: &Texture, to_level: u64| {
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.resource_state_command_encoder().unwrap();
            encoder
                .move_texture_mappings(from, 0, from_level, at, size, to, 0, to_level, at)
                .unwrap();
            encoder.end_encoding().unwrap();
            run(&command_buffer);
        };
        moved_level(&h, 3, &g, 0);
        assert_eq!(heap.used_size(), 17 * PAGE, "{name}");
        let g0 = read_level(&queue, &g, 0);
        for (i, &got) in g0.iter().enumerate() {
            if tile_of(i) == [0, 0] || strict {
                assert_eq!(
                    got, 0.0,
                    "texel {i} of the tile the tail moved onto on {name}"
                );
            }
        }
        if strict {
            for level in [3, 4] {
                let zero = vec![0.0; texels(level).len()];
                assert_eq!(read_level(&queue, &h, level), zero, "{name}");
            }
            assert_eq!(mapped_tiles(&queue, &g, strict), vec![[0, 0]], "{name}");
        }
        moved_level(&g, 0, &h, 3);
        assert_eq!(heap.used_size(), 17 * PAGE, "{name}");
        for level in [3, 4] {
            let zero = vec![0.0; texels(level).len()];
            assert_eq!(read_level(&queue, &h, level), zero, "the tail on {name}");
        }
        let command_buffer = queue.command_buffer().unwrap();
        write_levels(&command_buffer, &h, &[3, 4]);
        run(&command_buffer);
        for level in [3, 4] {
            assert_eq!(read_level(&queue, &h, level), texels(level), "{name}");
        }
        if strict {
            assert_eq!(mapped_tiles(&queue, &g, strict), Vec::<[u64; 2]>::new());
        }

        // Fences: an update after the bind, which a blit of another command buffer waits for;
        // a wait after the encoder's own update waits for the update before the encoder (none
        // here), and the bind runs, as plain Metal's does.
        let fence = device.new_fence();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.resource_state_command_encoder().unwrap();
        encoder
            .update_texture_mapping(&c, TileMapping::Unmap, Region::new_2d(0, 0, 2, 2), 0, 0)
            .unwrap();
        encoder.update_fence(&fence).unwrap();
        encoder.wait_for_fence(&fence).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        let after = queue.command_buffer().unwrap();
        let mut blit = after.blit_command_encoder().unwrap();
        blit.wait_for_fence(&fence).unwrap();
        blit.end_encoding().unwrap();
        run(&after);
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        if strict {
            assert_eq!(mapped_tiles(&queue, &c, strict), Vec::<[u64; 2]>::new());
        }
        // Nothing but, without unified memory, the first `Shared` buffer `read_level` made.
        assert_made_once(messages, &[], &shared_kinds(device));
    });
}

/// A blit into a block of the device's that holds one mapped tile writes that tile alone: the
/// block's unmapped tiles read zero after it on every device, as Metal drops a blit's writes to
/// unmapped tiles; a copy from another texture is dropped there too.
#[test]
fn a_blit_into_a_partly_mapped_block_leaves_its_unmapped_tiles_zero() {
    on_device(|device, messages| {
        let name = device.name().to_owned();
        let Some(heap) = sparse_heap(device, 1 << 20) else {
            return;
        };
        let r32 = PixelFormat::R32Float;
        let texture = level_texture(&heap, r32);
        let facts = SparseFacts::of(device, r32);
        assert_made_once(messages, &[], &facts.made(device));
        let strict = facts.strict;
        let queue = device.new_command_queue().unwrap();
        let tile = [(0, Region::new_2d(1, 1, 1, 1))];
        update(&queue, &texture, TileMapping::Map, &tile, |cb| {
            write_levels(cb, &texture, &[0])
        })
        .unwrap();
        let check = |what: &str| {
            let level0 = read_level(&queue, &texture, 0);
            for (i, (&got, &written)) in level0.iter().zip(&texels(0)).enumerate() {
                match tile_of(i) {
                    [1, 1] => assert_eq!(got, written, "{what}: a mapped texel {i} on {name}"),
                    [0, 0] | [1, 0] | [0, 1] => {
                        assert_eq!(got, 0.0, "{what}: an unmapped texel {i} on {name}")
                    }
                    _ if strict => {
                        assert_eq!(got, 0.0, "{what}: an unmapped texel {i} on {name}")
                    }
                    _ => {}
                }
            }
        };
        check("a buffer's copy");
        // A copy of every texel from an ordinary texture.
        let mut d = TextureDescriptor::texture_2d(r32, 256, 256, false);
        d.usage = TextureUsage::SHADER_READ;
        d.resource_options = ResourceOptions::private();
        let plain = device.new_texture(&d).unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        write_levels(&command_buffer, &plain, &[0]);
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        let all = Size::new(256, 256, 1);
        let at = Origin::new(0, 0, 0);
        blit.copy_from_texture_to_texture(&plain, 0, 0, at, all, &texture, 0, 0, at)
            .unwrap();
        blit.end_encoding().unwrap();
        run(&command_buffer);
        check("a texture's copy");
        assert_made_once(messages, &[], &shared_kinds(device));
    });
}

/// A mapping update of a texture that is not sparse, and a move between two such, record
/// nothing and leave the texels as they were, as plain Metal does (an M4 under macOS 27.0.1: map,
/// unmap, indirect unmap, move; its validation stops them); a move between a sparse texture and one
/// that is not stays refused (not measured).
#[test]
fn updates_of_a_texture_that_is_not_sparse_record_nothing() {
    on_device(|device, _| {
        let queue = device.new_command_queue().unwrap();
        let plain = || {
            let mut d = TextureDescriptor::texture_2d(PixelFormat::R32Float, 256, 256, false);
            d.usage = TextureUsage::SHADER_READ;
            d.resource_options = ResourceOptions::private();
            let texture = device.new_texture(&d).unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            write_levels(&command_buffer, &texture, &[0]);
            run(&command_buffer);
            texture
        };
        let (a, b) = (plain(), plain());
        let region = Region::new_2d(0, 0, 1, 1);
        for mode in [TileMapping::Map, TileMapping::Unmap] {
            update(&queue, &a, mode, &[(0, region)], |_| {}).unwrap();
        }
        let arguments: Vec<u8> = [1u32, 0, 0, 0, 1, 1, 1, 0, 0]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let arguments = device
            .new_buffer_with_bytes(&arguments, ResourceOptions::shared())
            .unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.resource_state_command_encoder().unwrap();
        encoder
            .update_texture_mapping_indirect(&a, TileMapping::Unmap, &arguments, 0)
            .unwrap();
        let at = Origin::new(0, 0, 0);
        encoder
            .move_texture_mappings(&a, 0, 0, at, Size::new(1, 1, 1), &b, 0, 0, at)
            .unwrap();
        encoder.end_encoding().unwrap();
        run(&command_buffer);
        assert_eq!(read_level(&queue, &a, 0), texels(0), "the updated texture");
        assert_eq!(
            read_level(&queue, &b, 0),
            texels(0),
            "the move's destination"
        );
        let Some(heap) = sparse_heap(device, 1 << 20) else {
            return;
        };
        let sparse = level_texture(&heap, PixelFormat::R32Float);
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.resource_state_command_encoder().unwrap();
        let refused = Error::Invalid(Invalid::new(
            "move_texture_mappings",
            Problem::Sparse(SparseProblem::NotSparse),
        ));
        assert_eq!(
            encoder.move_texture_mappings(&sparse, 0, 0, at, Size::new(1, 1, 1), &a, 0, 0, at),
            Err(refused)
        );
        encoder.end_encoding().unwrap();
        run(&command_buffer);
    });
}

/// A render pass's stores into unmapped tiles are dropped, as a Metal sparse probe measured on an
/// M3 Max and an M4 (a clear, a draw, a load's draw): a checkerboard of mapped tiles cleared and
/// drawn white, then loaded and stored again, reads white in its mapped tiles and zero in the
/// unmapped ones on every device that makes the texture, which the device's blocks hold two of
/// each; once the rest is mapped, those read zero too.
#[test]
fn a_render_pass_drops_its_stores_into_unmapped_tiles() {
    let (messages, _cache, lane) = super::resolves::lane();
    let device = &lane.device;
    let name = device.name().to_owned();
    let Some(heap) = sparse_heap(device, 1 << 20) else {
        return;
    };
    let rgba8 = PixelFormat::RGBA8Unorm;
    let mut d = TextureDescriptor::texture_2d(rgba8, 256, 256, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::private();
    let texture = match heap.new_texture(&d) {
        Ok(texture) => texture,
        Err(error) if lacks_sparse(&error) => {
            println!("SPARSE {name}: a render target refused {error:?}");
            return;
        }
        Err(error) => panic!("a sparse render target on {name}: {error:?}"),
    };
    assert_eq!(
        device.sparse_tile_size(TextureType::Type2D, rgba8, 1),
        Size::new(64, 64, 1)
    );
    let queue = &lane.queue;
    let board: Vec<(u64, Region)> = (0..4)
        .flat_map(|y| (0..4).map(move |x| (x, y)))
        .filter(|(x, y)| (x + y) % 2 == 0)
        .map(|(x, y)| (0, Region::new_2d(x, y, 1, 1)))
        .collect();
    update(queue, &texture, TileMapping::Map, &board, |_| {}).unwrap();
    assert_eq!(heap.used_size(), 8 * PAGE);
    let white = lane.pipeline((FULL, WHITE), &[rgba8], (None, None), 1);
    let mut pass = color_pass(&texture, None, StoreAction::Store, (0, 0));
    pass.color_attachments[0].clear_color = ClearColor::new(0.25, 0.5, 0.75, 1.0);
    lane.run(&pass, |l, e, h| l.draw(e, h, &white, 0.5))
        .unwrap();
    let mut load = pass.clone();
    load.color_attachments[0].load_action = LoadAction::Load;
    lane.run(&load, |_, _, _| {}).unwrap();
    let check = |what: &str, mapped: bool| {
        let texels = lane.read(&texture, (0, 0), 4, BlitOption::None);
        for (i, texel) in texels.as_chunks::<4>().0.iter().enumerate() {
            let [x, y] = tile_of(i);
            let want: &[u8] = if (x + y) % 2 == 0 {
                &[0xff; 4]
            } else {
                &[0; 4]
            };
            assert_eq!(
                texel, want,
                "{what}: texel {i} of tile ({x}, {y}) on {name}"
            );
        }
        assert_eq!(heap.used_size(), if mapped { 16 } else { 8 } * PAGE);
    };
    check("the checkerboard", false);
    let rest: Vec<(u64, Region)> = (0..4)
        .flat_map(|y| (0..4).map(move |x| (x, y)))
        .filter(|(x, y)| (x + y) % 2 == 1)
        .map(|(x, y)| (0, Region::new_2d(x, y, 1, 1)))
        .collect();
    update(queue, &texture, TileMapping::Map, &rest, |_| {}).unwrap();
    check("the rest mapped", true);
    println!("SPARSE {name}: a render pass's stores into unmapped tiles dropped");
    drop(texture);
    drop(heap);
    quiet(lane, &messages);
}

/// A multisampled sparse texture reads zero where its tiles are unmapped, as a Metal probe measured
/// on the M3 Max at 2x and 4x (`RGBA8Unorm`, tiles of 64x32 and 32x32 texels): in a texture of
/// 4x2 tiles, tiles (0, 0) and (2, 1) mapped read zero before any write; a red clear stores into
/// those two alone; (0, 0) unmapped reads zero, and so it does mapped again, as (1, 0) does mapped
/// new; a green clear, then (2, 1) unmapped, leaves the first two tiles green and the rest zero; the
/// heap counts the two tiles mapped then. Each read is a pass that loads the texture and resolves
/// it into a plain one, read per tile (`~` where a tile's texels differ): the probe's pass resolved
/// alone; this one stores too, which a device need not keep otherwise, and whose stores into the
/// unmapped tiles are dropped as a clear's are.
#[test]
fn a_multisampled_sparse_texture_reads_zero_where_unmapped() {
    let (messages, _cache, lane) = super::resolves::lane();
    let device = &lane.device;
    let name = device.name().to_owned();
    let Some(heap) = sparse_heap(device, 1 << 20) else {
        return;
    };
    let rgba8 = PixelFormat::RGBA8Unorm;
    let queue = &lane.queue;
    for (samples, side) in [(2, [64, 32]), (4, [32, 32])] {
        let tile = device.sparse_tile_size(TextureType::Type2DMultisample, rgba8, samples);
        assert_eq!([tile.width, tile.height], side, "{samples}x tile on {name}");
        let [width, height] = [side[0] * 4, side[1] * 2];
        let mut d = TextureDescriptor::texture_2d(rgba8, width, height, false);
        d.texture_type = TextureType::Type2DMultisample;
        d.sample_count = samples;
        d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        d.resource_options = ResourceOptions::private();
        // Refused only where a function may read the image's own residency and the device
        // makes no sparse image of the texture: the device's lack, named. Where it binds no
        // sparse memory the heap was refused; everywhere else the runtime makes it.
        let texture = match heap.new_texture(&d) {
            Ok(texture) => texture,
            Err(Error::Unsupported(u)) if matches!(u.feature, Feature::SparseTextures { .. }) => {
                println!("SPARSE {name}: a {samples}x texture refused {u:?}");
                return;
            }
            Err(error) => panic!("a {samples}x sparse texture on {name}: {error:?}"),
        };
        assert_eq!(heap.used_size(), 0);
        let facts = SparseFacts::of_samples(device, rgba8, samples as u32);
        let mut r = TextureDescriptor::texture_2d(rgba8, width, height, false);
        r.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        r.resource_options = ResourceOptions::private();
        let resolved = device.new_texture(&r).unwrap();
        let map = |mode, x, y| {
            update(
                queue,
                &texture,
                mode,
                &[(0, Region::new_2d(x, y, 1, 1))],
                |_| {},
            )
            .unwrap();
        };
        let clear = |red: f64, green: f64| {
            let mut pass = color_pass(&texture, None, StoreAction::Store, (0, 0));
            pass.color_attachments[0].clear_color = ClearColor::new(red, green, 0.0, 1.0);
            lane.run(&pass, |_, _, _| {}).unwrap();
        };
        let show = |what: &str, want: &str| {
            let store = StoreAction::StoreAndMultisampleResolve;
            let mut pass = color_pass(&texture, Some(&resolved), store, (0, 0));
            pass.color_attachments[0].load_action = LoadAction::Load;
            lane.run(&pass, |_, _, _| {}).unwrap();
            let texels = lane.read(&resolved, (0, 0), 4, BlitOption::None);
            let texels = texels.as_chunks::<4>().0;
            let rows: Vec<String> = (0..2)
                .map(|ty| {
                    (0..4)
                        .map(|tx| {
                            let at = |x: u64, y: u64| {
                                texels[((ty * side[1] + y) * width + tx * side[0] + x) as usize]
                            };
                            let first = at(0, 0);
                            let same =
                                (0..side[1]).all(|y| (0..side[0]).all(|x| at(x, y) == first));
                            let c = match first {
                                [0, 0, 0, 0] => '.',
                                [0xff, 0, 0, 0xff] => 'R',
                                [0, 0xff, 0, 0xff] => 'G',
                                _ => '?',
                            };
                            format!("{c}{}", if same { "" } else { "~" })
                        })
                        .collect()
                })
                .collect();
            assert_eq!(rows.join(" "), want, "{samples}x, {what}, on {name}");
        };
        map(TileMapping::Map, 0, 0);
        map(TileMapping::Map, 2, 1);
        show("mapped (0, 0) and (2, 1), never written", ".... ....");
        clear(1.0, 0.0);
        show("after a red clear", "R... ..R.");
        map(TileMapping::Unmap, 0, 0);
        show("(0, 0) unmapped", ".... ..R.");
        map(TileMapping::Map, 0, 0);
        show("(0, 0) mapped again", ".... ..R.");
        map(TileMapping::Map, 1, 0);
        show("(1, 0) mapped too", ".... ..R.");
        clear(0.0, 1.0);
        map(TileMapping::Unmap, 2, 1);
        show("a green clear, then (2, 1) unmapped", "GG.. ....");
        assert_eq!(heap.used_size(), 2 * PAGE, "{samples}x on {name}");
        map(TileMapping::Unmap, 0, 0);
        map(TileMapping::Unmap, 1, 0);
        let (strict, emulated) = (facts.strict, !facts.image);
        println!(
            "SPARSE {name}: a {samples}x texture reads zero where unmapped (strict {strict}, \
             emulated {emulated})"
        );
    }
    drop(heap);
    quiet(lane, &messages);
}

/// An update or a move through a view of a sparse texture names the texture's own tiles, as a Metal
/// sparse probe measured on an M3 Max and an M4 and our probes measured further: the level names
/// the texture's level whatever the view's levels, past them too; the slice is the view's, its
/// texture's past the view's first slice, past the view's slices too,
/// and one past the texture's is refused as on the texture; a view of another pixel format of the
/// same tile names the same tiles; an indirect update's regions and a move's sides read so too.
#[test]
fn updates_through_views_map_the_texture_s_own_tiles() {
    on_device(|device, messages| {
        let name = device.name().to_owned();
        let Some(heap) = sparse_heap(device, 1 << 20) else {
            return;
        };
        let r32 = PixelFormat::R32Float;
        let queue = device.new_command_queue().unwrap();
        let mut d = TextureDescriptor::texture_2d(r32, 256, 256, false);
        d.mipmap_level_count = 9;
        d.usage = TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW;
        d.resource_options = ResourceOptions::private();
        let texture = heap.new_texture(&d).unwrap();
        let facts = SparseFacts::of(device, r32);
        assert_made_once(messages, &[], &facts.made(device));
        let strict = facts.strict;
        let view = texture
            .new_texture_view_with_levels_slices(r32, TextureType::Type2D, 1..3, 0..1)
            .unwrap();
        // The view's level 0: the texture's level 0, not its level 1.
        let top_left = [(0, Region::new_2d(0, 0, 2, 2))];
        update(&queue, &view, TileMapping::Map, &top_left, |_| {}).unwrap();
        assert_eq!(heap.used_size(), 4 * PAGE, "{name}");
        if strict {
            assert_eq!(mapped_tiles(&queue, &texture, strict), sorted(quad(0, 0)));
        }
        // Its level 2, past the view's levels 1 and 2 of the texture: the texture's level 2, one
        // 64x64 tile, which holds a write after it; level 5, past `firstMipmapInTail`, maps
        // nothing, as on the texture.
        let level2 = [(2, Region::new_2d(0, 0, 1, 1))];
        update(&queue, &view, TileMapping::Map, &level2, |cb| {
            write_levels(cb, &texture, &[2])
        })
        .unwrap();
        assert_eq!(heap.used_size(), 5 * PAGE, "{name}");
        assert_eq!(read_level(&queue, &texture, 2), texels(2), "{name}");
        let level5 = [(5, Region::new_2d(0, 0, 1, 1))];
        update(&queue, &view, TileMapping::Map, &level5, |_| {}).unwrap();
        assert_eq!(heap.used_size(), 5 * PAGE, "{name}");
        // A view of another pixel format of the same 64x64 tile maps the same tiles.
        let uints = texture.new_texture_view(PixelFormat::R32Uint).unwrap();
        let bottom_right = [(0, Region::new_2d(2, 2, 2, 2))];
        update(&queue, &uints, TileMapping::Map, &bottom_right, |_| {}).unwrap();
        assert_eq!(heap.used_size(), 9 * PAGE, "{name}");
        if strict {
            let both = [quad(0, 0), quad(2, 2)].concat();
            assert_eq!(mapped_tiles(&queue, &texture, strict), sorted(both));
        }
        // A move from the view's level 0 names the texture's level 0: its tile (0, 0) onto the
        // texture's unmapped (3, 0); the heap's count stays.
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.resource_state_command_encoder().unwrap();
        let (one, zero) = (Size::new(1, 1, 1), Origin::new(0, 0, 0));
        encoder
            .move_texture_mappings(&view, 0, 0, zero, one, &texture, 0, 0, Origin::new(3, 0, 0))
            .unwrap();
        encoder.end_encoding().unwrap();
        run(&command_buffer);
        assert_eq!(heap.used_size(), 9 * PAGE, "{name}");
        if strict {
            let moved = [vec![[3, 0], [1, 0], [0, 1], [1, 1]], quad(2, 2)].concat();
            assert_eq!(mapped_tiles(&queue, &texture, strict), sorted(moved));
        }

        // The view's slice 0 of slices 2 and 3: the texture's slice 2.
        let mut a = TextureDescriptor::texture_2d(r32, 256, 256, false);
        a.texture_type = TextureType::Type2DArray;
        a.array_length = 4;
        a.usage = TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW;
        a.resource_options = ResourceOptions::private();
        let array = heap.new_texture(&a).unwrap();
        let slices = array
            .new_texture_view_with_levels_slices(r32, TextureType::Type2DArray, 0..1, 2..4)
            .unwrap();
        update(&queue, &slices, TileMapping::Map, &top_left, |_| {}).unwrap();
        assert_eq!(heap.used_size(), 13 * PAGE, "{name}");
        let one_update = |texture: &Texture, mode: TileMapping, slice: u64| {
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.resource_state_command_encoder().unwrap();
            let answer =
                encoder.update_texture_mapping(texture, mode, Region::new_2d(0, 0, 2, 2), 0, slice);
            encoder.end_encoding().unwrap();
            run(&command_buffer);
            answer
        };
        one_update(&array, TileMapping::Unmap, 2).unwrap();
        assert_eq!(heap.used_size(), 9 * PAGE, "slice 2 unmapped on {name}");
        // Slices 1 and 2: the view's slice 2 is the texture's slice 3, past the view's; its slice
        // 3 would be the texture's slice 4, past the texture's, which the call takes and maps
        // nothing for, as on the texture and as plain Metal takes it.
        let middle = array
            .new_texture_view_with_levels_slices(r32, TextureType::Type2DArray, 0..1, 1..3)
            .unwrap();
        one_update(&middle, TileMapping::Map, 2).unwrap();
        assert_eq!(heap.used_size(), 13 * PAGE, "{name}");
        one_update(&array, TileMapping::Unmap, 3).unwrap();
        assert_eq!(heap.used_size(), 9 * PAGE, "slice 3 unmapped on {name}");
        one_update(&middle, TileMapping::Map, 3).unwrap();
        assert_eq!(
            heap.used_size(),
            9 * PAGE,
            "a slice past the texture's maps nothing on {name}"
        );
        // An indirect update's slice 1 through the view: the texture's slice 2.
        let shared = ResourceOptions::shared();
        let arguments = words_buffer(&queue, &[1, 0, 0, 0, 2, 2, 1, 0, 1], shared);
        assert_eq!(
            indirect(&queue, &middle, TileMapping::Map, &arguments, |_| {}),
            None,
            "{name}"
        );
        assert_eq!(heap.used_size(), 13 * PAGE, "{name}");
        one_update(&array, TileMapping::Unmap, 2).unwrap();
        assert_eq!(
            heap.used_size(),
            9 * PAGE,
            "slice 2 unmapped again on {name}"
        );
        // A move from the view's slice 0, the texture's slice 1, onto the texture's slice 0.
        one_update(&array, TileMapping::Map, 1).unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.resource_state_command_encoder().unwrap();
        let quad_size = Size::new(2, 2, 1);
        encoder
            .move_texture_mappings(&middle, 0, 0, zero, quad_size, &array, 0, 0, zero)
            .unwrap();
        encoder.end_encoding().unwrap();
        run(&command_buffer);
        assert_eq!(heap.used_size(), 13 * PAGE, "{name}");
        one_update(&array, TileMapping::Unmap, 1).unwrap();
        assert_eq!(heap.used_size(), 13 * PAGE, "slice 1 left empty on {name}");
        one_update(&array, TileMapping::Unmap, 0).unwrap();
        assert_eq!(heap.used_size(), 9 * PAGE, "slice 0 unmapped on {name}");
        assert_made_once(
            messages,
            &[Degradation::HostReadMappingArguments],
            &shared_kinds(device),
        );
    });
}

/// Face `face`'s texels of a 256x256 `R32Float` cube: `1 + x + 256 * y + 100000 * face`.
fn face_texels(face: u64) -> Vec<f32> {
    (0..256 * 256u64)
        .map(|i| (1 + i + 100_000 * face) as f32)
        .collect()
}

/// A sparse cube texture's faces map their own tiles (an M3 Max, and an M4, alike): a 256x256
/// `R32Float` cube of one level from a 1-MiB heap of 16-KiB pages, its tile 64x64; face 1's tile
/// (0, 0) and face 4's tiles (1, 1) and (2, 1) mapped take three pages, and a blit of every face
/// after them lands in those tiles alone, every other tile reading zero where residency is strict;
/// an unmap of face 4's tile (1, 1) gives its page back and it reads zero, its neighbour keeping
/// its texels. On a device whose functions read residency from tile maps (MoltenVK) the cube is an
/// ordinary image with no tile map, which a slot reading residency from a map refuses as it is
/// bound; it was refused as it was made before.
#[test]
fn a_sparse_cube_maps_each_face_s_own_tiles() {
    on_device(|device, messages| {
        let name = device.name().to_owned();
        let Some(heap) = sparse_heap(device, 1 << 20) else {
            return;
        };
        assert_eq!(
            device.sparse_tile_size(TextureType::Cube, PixelFormat::R32Float, 1),
            Size::new(64, 64, 1)
        );
        let mut d = TextureDescriptor::texture_cube(PixelFormat::R32Float, 256, false);
        d.usage = TextureUsage::SHADER_READ;
        d.resource_options = ResourceOptions::private();
        let texture = match heap.new_texture(&d) {
            Ok(texture) => texture,
            Err(error) if lacks_sparse(&error) => {
                println!("SPARSE {name}: cube refused {error:?}");
                return;
            }
            Err(error) => panic!("a sparse cube on {name}: {error:?}"),
        };
        let facts = SparseFacts::of(device, PixelFormat::R32Float);
        assert_made_once(messages, &[], &facts.made(device));
        let (strict, emulated) = (facts.strict, !facts.image);
        let queue = device.new_command_queue().unwrap();
        let mapped = [(1, [0, 0]), (4, [1, 1]), (4, [2, 1])];
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.resource_state_command_encoder().unwrap();
        encoder
            .update_texture_mappings(
                &texture,
                TileMapping::Map,
                &[Region::new_2d(0, 0, 1, 1), Region::new_2d(1, 1, 2, 1)],
                &[0, 0],
                &[1, 4],
            )
            .unwrap();
        encoder.end_encoding().unwrap();
        let sources: Vec<Buffer> = (0..6)
            .map(|face| buffer_of(device, &face_texels(face)))
            .collect();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        for (face, source) in sources.iter().enumerate() {
            blit.copy_from_buffer_to_texture(
                source,
                0,
                256 * 4,
                256 * 256 * 4,
                Size::new(256, 256, 1),
                &texture,
                face as u64,
                0,
                Origin::new(0, 0, 0),
                BlitOption::None,
            )
            .unwrap();
        }
        blit.end_encoding().unwrap();
        run(&command_buffer);
        assert_eq!(heap.used_size(), 3 * PAGE, "{name}");
        // Face `face` read back: each texel its write where its tile is mapped, zero elsewhere
        // where residency is strict.
        let check = |held: &dyn Fn(u64, [u64; 2]) -> bool, what: &str| {
            for face in 0..6u64 {
                let got = read_slice(&queue, &texture, face);
                let want = face_texels(face);
                for (i, (&got, &written)) in got.iter().zip(&want).enumerate() {
                    if held(face, tile_of(i)) {
                        assert_eq!(got, written, "{what}: face {face} texel {i} on {name}");
                    } else if strict {
                        assert_eq!(got, 0.0, "{what}: face {face} texel {i} on {name}");
                    }
                }
            }
        };
        check(&|face, tile| mapped.contains(&(face, tile)), "mapped");
        update_slice(
            &queue,
            &texture,
            TileMapping::Unmap,
            Region::new_2d(1, 1, 1, 1),
            4,
        );
        assert_eq!(heap.used_size(), 2 * PAGE, "{name}");
        check(
            &|face, tile| mapped.contains(&(face, tile)) && (face, tile) != (4, [1, 1]),
            "after the unmap",
        );
        println!("SPARSE {name}: cube strict {strict}, emulated {emulated}");
        drop(texture);
        assert_made_once(messages, &[], &shared_kinds(device));
    });
}

/// Slice `slice` of level 0 of a 256x256 texture, read back by a blit.
fn read_slice(queue: &CommandQueue, texture: &Texture, slice: u64) -> Vec<f32> {
    let out = queue
        .device()
        .new_buffer(256 * 256 * 4, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        texture,
        slice,
        0,
        Origin::new(0, 0, 0),
        Size::new(256, 256, 1),
        &out,
        0,
        256 * 4,
        256 * 256 * 4,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    run(&command_buffer);
    read_buffer(&out, 0, 256 * 256 * 4)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect()
}

/// One command buffer of a resource state encoder's update of `region` of level 0 of slice
/// `slice`, run.
fn update_slice(
    queue: &CommandQueue,
    texture: &Texture,
    mode: TileMapping,
    region: Region,
    slice: u64,
) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.resource_state_command_encoder().unwrap();
    encoder
        .update_texture_mappings(texture, mode, &[region], &[0], &[slice])
        .unwrap();
    encoder.end_encoding().unwrap();
    run(&command_buffer);
}

/// Every committed public kernel case with a sparse texture runs through the runtime as Metal ran
/// it, or as the candidate executor runs it afresh, or is refused honestly (`kernel_textures`'
/// census): the residency and the values its function reads of mapped and unmapped tiles, the
/// texture made from a sparse heap and mapped by a resource state encoder
/// (`runtime_case/sparse.rs`).
#[test]
fn sparse_cases_match_metal() {
    census("runtime sparse kernels", &sparse_cases());
}

/// Every committed public kernel case that binds a sparse texture.
fn sparse_cases() -> Vec<NamedCase> {
    let cases = public_cases(|case| {
        case.stage == Stage::Kernel
            && case.public_source.is_some()
            && case.textures.iter().any(|texture| texture.sparse.is_some())
    })
    .unwrap();
    assert!(
        !cases.is_empty(),
        "no committed case binds a sparse texture"
    );
    cases
}
