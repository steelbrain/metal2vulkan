//! Where an encoder copies shadowed textures from their buffers and back (`shadow.rs`).

use super::super::error::{internal, Error};
use super::super::host_aliases::{self, Alias};
use super::super::key_hash::KeyMap;
use super::super::resource::ResourceKey;
use super::super::shadow::{
    self, Backing, CopyCause, KeptArea, Lost, ShadowBuffer, ShadowCarry, ShadowCopies, ShadowCopy,
    ShadowSource, ShadowSync, ShadowTarget, ShadowUse, Step, Within,
};
use super::super::shadow_merge::ScratchSpares;
use super::super::tracker::{Access, Use, UseRange};
use ash::vk;
use metal2vulkan::error::Feature;
use std::ops::Range;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

/// A merge's scratch, numbered, keeping every texel.
impl KeptArea for u32 {
    fn kept_area(&self) -> Option<[u32; 4]> {
        None
    }
}

/// A merge's scratch, numbered, keeping the texels of its rectangle (`None` every texel).
#[derive(Clone, Debug, PartialEq, Eq)]
struct Boxed(u32, Option<[u32; 4]>);

impl KeptArea for Boxed {
    fn kept_area(&self) -> Option<[u32; 4]> {
        self.1
    }
}

/// A shadowed texture over `bytes` of the buffer `buffer`.
fn shadow(buffer: ResourceKey, bytes: Range<u64>) -> Arc<ShadowCopies> {
    shadow_joined(buffer, None, bytes)
}

/// A shadowed texture over `bytes` of the buffer `buffer`, which joined the host memory of the
/// key `alias` where given (`host_aliases.rs`).
fn shadow_joined(
    buffer: ResourceKey,
    alias: Option<ResourceKey>,
    bytes: Range<u64>,
) -> Arc<ShadowCopies> {
    shadow_in(
        ShadowBuffer::Fixed(vk::Buffer::null(), 0),
        buffer,
        alias,
        bytes,
    )
}

/// `texture` over a `Private` buffer's bytes, tracked where `tracked`: the CPU writes none, and
/// another queue's commands write them while a command buffer runs only where it is untracked
/// (`ShadowCopies::merges_back`).
fn over_private(texture: Arc<ShadowCopies>, tracked: bool) -> Arc<ShadowCopies> {
    let mut copies = Arc::into_inner(texture).expect("one holder");
    copies.backing = Backing::PrivateBuffer;
    copies.tracked = tracked;
    Arc::new(copies)
}

/// A shadowed texture over `bytes` of the buffer `buffer`, in `raw`.
fn shadow_in(
    raw: ShadowBuffer,
    buffer: ResourceKey,
    alias: Option<ResourceKey>,
    bytes: Range<u64>,
) -> Arc<ShadowCopies> {
    Arc::new(ShadowCopies {
        source: ShadowSource::Bytes {
            buffer: raw,
            key: buffer,
            bytes,
            target: ShadowTarget::Image {
                image: vk::Image::null(),
                regions: Vec::new(),
                back: Vec::new(),
                staging: None,
                texel: Some(4),
                words: None,
                unorm: None,
            },
            alias,
        },
        shadow_key: ResourceKey::new(),
        shadow_range: UseRange::Image {
            aspects: vk::ImageAspectFlags::COLOR,
            levels: 0..1,
            layers: 0..1,
        },
        tracked: true,
        drops_writes: false,
        backing: Backing::MappedBuffer,
        merge_scratch: ScratchSpares::default(),
        touched: AtomicU64::new(0),
        stood: AtomicU64::new(0),
    })
}

fn shader() -> Access {
    Access::new(
        vk::PipelineStageFlags2::COMPUTE_SHADER,
        vk::AccessFlags2::SHADER_STORAGE_READ,
    )
}

/// A command's use of `texture`, writing through it when `writes`.
fn through(texture: &Arc<ShadowCopies>, writes: bool) -> (ShadowUse, Use<ResourceKey>) {
    let range = texture.shadow_range.clone();
    let used = if writes {
        Use::writes(texture.shadow_key, true, range, shader())
    } else {
        Use::reads(texture.shadow_key, true, range, shader())
    };
    (
        ShadowUse {
            copies: Arc::clone(texture),
            writes,
            overwrites: false,
            kept: false,
            area: None,
            loads: None,
            tile_stores: false,
        },
        used,
    )
}

/// A use of `bytes` of `buffer`, writing when `writes`.
fn bytes(buffer: ResourceKey, bytes: Range<u64>, writes: bool) -> Use<ResourceKey> {
    if writes {
        Use::writes(buffer, true, UseRange::Bytes(bytes), shader())
    } else {
        Use::reads(buffer, true, UseRange::Bytes(bytes), shader())
    }
}

/// What a copy moves: into the image (`true`) or back (`false`); a keep and a merge are the
/// tests of their own below.
fn named(copies: &[ShadowCopy<u32>]) -> Vec<(ResourceKey, bool)> {
    copies
        .iter()
        .map(|copy| match copy.step {
            Step::Into => (copy.copies.shadow_key, true),
            Step::Back => (copy.copies.shadow_key, false),
            Step::Keep(_) | Step::Merge(_) => panic!("a keep or a merge"),
        })
        .collect()
}

/// A copy's step, its scratch numbered.
#[derive(Debug, PartialEq, Eq)]
enum Moved {
    Into(ResourceKey),
    Back(ResourceKey),
    Keep(ResourceKey, u32),
    Merge(ResourceKey, u32),
}

fn moved(copies: &[ShadowCopy<u32>]) -> Vec<Moved> {
    copies
        .iter()
        .map(|copy| {
            let key = copy.copies.shadow_key;
            match copy.step {
                Step::Into => Moved::Into(key),
                Step::Back => Moved::Back(key),
                Step::Keep(n) => Moved::Keep(key, n),
                Step::Merge(n) => Moved::Merge(key, n),
            }
        })
        .collect()
}

/// The shadows and uses of a command through `textures` with `others`.
fn command_uses(
    textures: &[(&Arc<ShadowCopies>, bool)],
    others: &[Use<ResourceKey>],
) -> (Vec<ShadowUse>, Vec<Use<ResourceKey>>) {
    let (shadows, mut uses): (Vec<ShadowUse>, Vec<Use<ResourceKey>>) = textures
        .iter()
        .map(|&(texture, writes)| through(texture, writes))
        .unzip();
    uses.extend_from_slice(others);
    (shadows, uses)
}

/// One command of an encoder whose merges' scratch is numbered from `made` on: the copies
/// before it, then its record.
fn planned(
    sync: &mut ShadowSync<u32>,
    made: &mut u32,
    textures: &[(&Arc<ShadowCopies>, bool)],
    others: &[Use<ResourceKey>],
) -> Vec<ShadowCopy<u32>> {
    let (shadows, uses) = command_uses(textures, others);
    let copies = sync
        .before(&shadows, &uses, |_, _, _| {
            *made += 1;
            Ok(*made)
        })
        .unwrap();
    sync.after(&shadows, &uses, &[]);
    copies
}

/// One command of an encoder that writes no bytes two ways: the copies before it, then its
/// record.
fn command(
    sync: &mut ShadowSync<u32>,
    textures: &[(&Arc<ShadowCopies>, bool)],
    others: &[Use<ResourceKey>],
) -> Vec<(ResourceKey, bool)> {
    named(&planned(sync, &mut 0, textures, others))
}

/// An encoder copies a texture over a buffer's bytes in before its first use, keeps it before
/// the first command that writes through it, and merges what its commands wrote back as it ends.
/// A texture only read is not merged back.
#[test]
fn an_encoder_copies_a_texture_in_before_its_first_use_and_merges_its_writes_back() {
    let buffer = ResourceKey::new();
    let t = shadow(buffer, 256..1280);
    let k = t.shadow_key;
    let mut made = 0;
    let mut run = |sync: &mut ShadowSync<u32>, textures: &[(&Arc<ShadowCopies>, bool)]| {
        moved(&planned(sync, &mut made, textures, &[]))
    };
    let mut sync = ShadowSync::default();
    assert_eq!(run(&mut sync, &[(&t, false)]), [Moved::Into(k)]);
    assert_eq!(run(&mut sync, &[(&t, false)]), []);
    assert_eq!(run(&mut sync, &[(&t, true)]), [Moved::Keep(k, 1)]);
    assert_eq!(run(&mut sync, &[(&t, true)]), []);
    assert_eq!(moved(&sync.finish()), [Moved::Merge(k, 1)]);
    assert_eq!(moved(&sync.finish()), []);
    let mut sync = ShadowSync::default();
    run(&mut sync, &[(&t, false)]);
    assert_eq!(moved(&sync.finish()), []);
}

/// The areas of the copies back `sync` makes as its encoder ends.
fn back_areas(sync: &mut ShadowSync<u32>) -> Vec<Option<[u32; 4]>> {
    sync.finish().iter().map(|copy| copy.area).collect()
}

/// A copy back (an image's shadow's: a buffer's bytes' shadow is merged back) covers only where
/// the commands since the last wrote through the texture, where each bounds its writes (a pass's
/// attachment it loads, inside its draws' scissors): the rectangle bounding their areas, or every
/// texel where one does not bound them. A copy in is of every texel, and joining a use that reads
/// keeps the area of the one that writes.
#[test]
fn a_copy_back_covers_only_where_the_commands_wrote() {
    let t = image_shadow(ResourceKey::new(), 0, 1);
    let writing = |area: Option<[u32; 4]>| ShadowUse {
        copies: Arc::clone(&t),
        writes: true,
        overwrites: false,
        kept: false,
        area,
        loads: None,
        tile_stores: false,
    };
    let (_, written) = through(&t, true);
    let run = |areas: &[Option<[u32; 4]>]| {
        let mut sync = ShadowSync::<u32>::default();
        for &area in areas {
            let shadows = [writing(area)];
            let uses = [written.clone()];
            let copies = sync.before(&shadows, &uses, |_, _, _| Ok(0)).unwrap();
            assert!(copies.iter().all(|copy| copy.area.is_none()), "a copy in");
            sync.after(&shadows, &uses, &[]);
        }
        back_areas(&mut sync)
    };
    assert_eq!(run(&[Some([0, 0, 8, 4])]), [Some([0, 0, 8, 4])]);
    assert_eq!(
        run(&[Some([0, 0, 8, 4]), Some([4, 2, 16, 8])]),
        [Some([0, 0, 16, 8])]
    );
    assert_eq!(run(&[Some([0, 0, 8, 4]), None]), [None]);
    assert_eq!(run(&[None]), [None]);
    // Joined with a use that reads, the write keeps its area; with another write that does not
    // bound it, it has none.
    let mut joined = vec![writing(Some([2, 2, 6, 6]))];
    shadow::join(&mut joined, &through(&t, false).0);
    assert_eq!(joined[0].area, Some([2, 2, 6, 6]));
    let mut joined = vec![through(&t, false).0];
    shadow::join(&mut joined, &writing(Some([2, 2, 6, 6])));
    assert_eq!(joined[0].area, Some([2, 2, 6, 6]));
    shadow::join(&mut joined, &through(&t, true).0);
    assert_eq!(joined[0].area, None);
}

/// A shadowed texture over `bytes` of the buffer `buffer` whose copies are cut to a rectangle
/// (`ShadowCopies::cuts`): one region of a 64x32 image, 4 bytes a texel, rows 320 bytes apart.
fn cut_shadow(buffer: ResourceKey, bytes: Range<u64>) -> Arc<ShadowCopies> {
    cut_shadow_in(ShadowBuffer::Fixed(vk::Buffer::null(), 0), buffer, bytes)
}

/// [`cut_shadow`]'s texture, its bytes in `raw`.
fn cut_shadow_in(raw: ShadowBuffer, buffer: ResourceKey, bytes: Range<u64>) -> Arc<ShadowCopies> {
    let mut copies = Arc::into_inner(shadow_in(raw, buffer, None, bytes)).expect("one holder");
    if let ShadowSource::Bytes {
        target: ShadowTarget::Image { regions, .. },
        ..
    } = &mut copies.source
    {
        regions.push(vk::BufferImageCopy {
            buffer_offset: 256,
            buffer_row_length: 80,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D::default(),
            image_extent: vk::Extent3D {
                width: 64,
                height: 32,
                depth: 1,
            },
        });
    }
    assert!(copies.cuts());
    Arc::new(copies)
}

/// The copies `sync` plans before a command of `shadows` with `uses`, then the command noted:
/// each copy's step (`'i'` into the image, `'b'` back, `'k'` a keep, `'m'` a merge) and area.
fn planned_areas(
    sync: &mut ShadowSync<u32>,
    shadows: &[ShadowUse],
    uses: &[Use<ResourceKey>],
) -> Vec<(char, Option<[u32; 4]>)> {
    let copies = sync.before(shadows, uses, |_, _, _| Ok(0)).unwrap();
    sync.after(shadows, uses, &[]);
    copies.iter().map(|copy| (step(copy), copy.area)).collect()
}

/// A copy's step: `'i'` into the image, `'b'` back, `'k'` a keep, `'m'` a merge.
fn step<M>(copy: &ShadowCopy<M>) -> char {
    match copy.step {
        Step::Into => 'i',
        Step::Back => 'b',
        Step::Keep(_) => 'k',
        Step::Merge(_) => 'm',
    }
}

/// A use that reads and writes a texture only inside a rectangle (`ShadowUse::loads`, a pass's
/// attachment it loads and takes no other way, inside its draws' scissors) copies in only the
/// texels there, and is kept for its merge back. A later command of the encoder that reads past
/// them first takes the merge of what was written and a copy in of every texel, so no texel the
/// image did not take from the buffer is read. A shadow whose copies are not cut to a rectangle,
/// a use joined with one that reads anywhere, and a texture written two ways copy every texel in;
/// a use inside the rectangle the image holds copies nothing in.
#[test]
fn a_copy_in_covers_only_where_the_command_reads_and_writes() {
    let buffer = ResourceKey::new();
    let t = cut_shadow(buffer, 256..256 + 320 * 32);
    let (_, written) = through(&t, true);
    let (_, read) = through(&t, false);
    let inside = |area: [u32; 4], writes: bool| ShadowUse {
        copies: Arc::clone(&t),
        writes,
        overwrites: false,
        kept: false,
        area: writes.then_some(area),
        loads: Some(area),
        tile_stores: false,
    };
    let a = [8, 4, 40, 20];
    let mut sync = ShadowSync::<u32>::default();
    let uses = [written.clone()];
    assert_eq!(
        planned_areas(&mut sync, &[inside(a, true)], &uses),
        [('i', Some(a)), ('k', None)]
    );
    // Inside what the image holds: nothing more is copied in.
    assert_eq!(
        planned_areas(&mut sync, &[inside([10, 5, 20, 10], true)], &uses),
        []
    );
    assert_eq!(moved(&sync.finish()), [Moved::Merge(t.shadow_key, 0)]);
    // Past it: what was written is merged back, then every texel comes in.
    let mut sync = ShadowSync::<u32>::default();
    planned_areas(&mut sync, &[inside(a, true)], &uses);
    let (anywhere, _) = through(&t, false);
    assert_eq!(
        planned_areas(&mut sync, &[anywhere], std::slice::from_ref(&read)),
        [('m', None), ('i', None)]
    );
    assert_eq!(back_areas(&mut sync), []);
    // Past it, unwritten: only the copy in.
    let mut sync = ShadowSync::<u32>::default();
    planned_areas(&mut sync, &[inside(a, false)], std::slice::from_ref(&read));
    assert_eq!(
        planned_areas(&mut sync, &[inside([0, 0, 64, 32], false)], &[read]),
        [('i', Some([0, 0, 64, 32]))]
    );
    // Joined with a use that reads anywhere, a use reads anywhere.
    let mut joined = vec![inside(a, true)];
    shadow::join(&mut joined, &through(&t, false).0);
    assert_eq!((joined[0].loads, joined[0].area), (None, Some(a)));
    let mut sync = ShadowSync::<u32>::default();
    assert_eq!(
        planned_areas(&mut sync, &joined, &uses),
        [('i', None), ('k', None)]
    );
    // Two uses inside rectangles read inside the one bounding both.
    let mut joined = vec![inside(a, true)];
    shadow::join(&mut joined, &inside([30, 10, 50, 30], false));
    assert_eq!(joined[0].loads, Some([8, 4, 50, 30]));
    // A shadow whose copies are not cut: every texel.
    let whole = shadow(buffer, 256..1280);
    let mut sync = ShadowSync::<u32>::default();
    let uncut = ShadowUse {
        copies: Arc::clone(&whole),
        ..inside(a, true)
    };
    let uses = [through(&whole, true).1];
    assert_eq!(
        planned_areas(&mut sync, &[uncut], &uses),
        [('i', None), ('k', None)]
    );
    // Written two ways: kept, and every texel copied in for the merge to compare.
    let mut sync = ShadowSync::<u32>::default();
    let uses = [written, bytes(buffer, 256..512, true)];
    let copies = sync
        .before(&[inside(a, true)], &uses, |_, _, _| Ok(7))
        .unwrap();
    assert_eq!(
        moved(&copies),
        [Moved::Into(t.shadow_key), Moved::Keep(t.shadow_key, 7)]
    );
    assert_eq!(copies[0].area, None);
    // The texels a cut copy moves.
    let copy = ShadowCopy::<u32> {
        copies: Arc::clone(&t),
        step: Step::Into,
        kept: false,
        area: Some(a),
        cause: Some(CopyCause::First),
        moves: None,
    };
    assert_eq!((copy.texels_in(), copy.texels_back()), (32 * 16, 0));
    assert_eq!(copy.texels_copied_in(), 32 * 16);
}

/// Whether each copy is a copy in that is the encoder's first of its texture
/// (`CopyCause::First`).
fn firsts(copies: &[ShadowCopy<u32>]) -> Vec<(ResourceKey, bool)> {
    let into = copies.iter().filter(|copy| matches!(copy.step, Step::Into));
    into.map(|copy| {
        let first = copy.cause == Some(CopyCause::First);
        (copy.copies.shadow_key, first)
    })
    .collect()
}

/// An encoder's first copy in of a texture is the one before the first command that uses it, or
/// the first after work it records apart (`ShadowSync::leave`), which forgets every texture; a
/// copy in after a write of the bytes, or before a command needing texels the image does not
/// hold, is not (`WorkCounts::shadow_texels_first_copied_in`).
#[test]
fn an_encoders_first_copy_in_of_a_texture_is_counted_apart() {
    let buffer = ResourceKey::new();
    let t = cut_shadow(buffer, 256..256 + 320 * 32);
    let u = shadow(ResourceKey::new(), 0..1024);
    let mut sync = ShadowSync::default();
    let first = |sync: &mut ShadowSync<u32>, textures: &[_], others: &[_]| {
        firsts(&planned(sync, &mut 0, textures, others))
    };
    assert_eq!(
        first(&mut sync, &[(&t, false)], &[]),
        [(t.shadow_key, true)]
    );
    assert_eq!(first(&mut sync, &[(&t, false)], &[]), []);
    // A write of the bytes another way: the next copy in is a later one; a texture the encoder
    // meets with it takes its first.
    assert_eq!(first(&mut sync, &[], &[bytes(buffer, 256..512, true)]), []);
    assert_eq!(
        first(&mut sync, &[(&t, false), (&u, false)], &[]),
        [(t.shadow_key, false), (u.shadow_key, true)]
    );
    // A command reading past the rectangle the image holds.
    let inside = ShadowUse {
        loads: Some([0, 0, 8, 8]),
        ..through(&t, false).0
    };
    let mut cut = ShadowSync::<u32>::default();
    let uses = [through(&t, false).1];
    let copies = cut.before(&[inside], &uses, |_, _, _| Ok(0)).unwrap();
    assert_eq!(firsts(&copies), [(t.shadow_key, true)]);
    let copies = cut
        .before(&[through(&t, false).0], &uses, |_, _, _| Ok(0))
        .unwrap();
    assert_eq!(firsts(&copies), [(t.shadow_key, false)]);
    // Work recorded apart: the encoder forgets every texture.
    assert_eq!(named(&sync.leave()), []);
    assert_eq!(
        first(&mut sync, &[(&t, false)], &[]),
        [(t.shadow_key, true)]
    );
    // An image's shadow's first copy in moves every texel of its regions, which
    // `texels_in` (a buffer's texture's) leaves out.
    let mut copies = Arc::into_inner(view_of(ResourceKey::new(), 0..1, 0..2)).expect("one");
    if let ShadowSource::Image { regions, .. } = &mut copies.source {
        regions.push(vk::ImageCopy {
            src_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 2,
            },
            dst_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 2,
            },
            extent: vk::Extent3D {
                width: 8,
                height: 4,
                depth: 1,
            },
            ..Default::default()
        });
    }
    let v = Arc::new(copies);
    let copies = planned(&mut ShadowSync::default(), &mut 0, &[(&v, false)], &[]);
    let moved: Vec<(u64, u64)> = (copies.iter())
        .map(|copy| (copy.texels_in(), copy.texels_copied_in()))
        .collect();
    assert_eq!(moved, [(0, 8 * 4 * 2)]);
    // Host pages the texture's own image lies in too are told apart; nothing else changes.
    assert_eq!(Backing::HostPages.aliased(), Backing::AliasedHostPages);
    assert_eq!(Backing::MappedBuffer.aliased(), Backing::MappedBuffer);
}

/// Why each copy in is made (`ShadowCopy::cause`). A texture over a buffer's bytes that merges
/// back, first written through in the encoder, takes the encoder's first copy in, before its keep,
/// though the keep's scratch made the encoder meet the texture first. Every later copy in after a
/// change of the truth names the change that came first: a merge of another texture over the same
/// bytes, the texture's own merge after a command that wrote them through it and through another
/// texture, a write of them another way, and writes Metal drops. Two textures over the same bytes
/// count once as sharing their truth (`ShadowSync::sharing_truth`).
#[test]
fn each_copy_in_names_why_it_is_made() {
    let buffer = ResourceKey::new();
    let t = shadow(buffer, 256..1280);
    let u = shadow(buffer, 256..1280);
    let other = shadow(ResourceKey::new(), 0..1024);
    let (kt, ku) = (t.shadow_key, u.shadow_key);
    assert!(t.merges_back());
    let into = |copies: &[ShadowCopy<u32>]| -> Vec<(ResourceKey, Option<CopyCause>)> {
        (copies.iter())
            .filter(|copy| matches!(copy.step, Step::Into))
            .map(|copy| (copy.copies.shadow_key, copy.cause))
            .collect()
    };
    let changed = |lost| Some(CopyCause::Changed(lost));
    let mut made = 0;
    let mut sync = ShadowSync::default();
    let copies = planned(&mut sync, &mut made, &[(&t, true)], &[]);
    assert_eq!(moved(&copies), [Moved::Into(kt), Moved::Keep(kt, 1)]);
    assert_eq!(into(&copies), [(kt, Some(CopyCause::First))]);
    // Another texture over the bytes, read: its own first copy in, after the merge of the first.
    let copies = planned(&mut sync, &mut made, &[(&u, false), (&other, false)], &[]);
    assert_eq!(
        into(&copies),
        [
            (ku, Some(CopyCause::First)),
            (other.shadow_key, Some(CopyCause::First))
        ]
    );
    assert_eq!(sync.sharing_truth(), 1);
    // Written through the first again, then the second read: the first's merge changed the bytes.
    planned(&mut sync, &mut made, &[(&t, true)], &[]);
    let copies = planned(&mut sync, &mut made, &[(&u, false)], &[]);
    assert_eq!(into(&copies), [(ku, changed(Lost::OtherShadow))]);
    // Written through both in one command: two ways, each merged before the next command.
    planned(&mut sync, &mut made, &[(&t, true), (&u, true)], &[]);
    let copies = planned(&mut sync, &mut made, &[(&t, false)], &[]);
    assert_eq!(into(&copies), [(kt, changed(Lost::TwoWays))]);
    // Its bytes written another way.
    planned(&mut sync, &mut made, &[], &[bytes(buffer, 256..512, true)]);
    let copies = planned(&mut sync, &mut made, &[(&t, false)], &[]);
    assert_eq!(into(&copies), [(kt, changed(Lost::Written))]);
    // Writes Metal drops.
    let mut dropping = Arc::into_inner(shadow(ResourceKey::new(), 0..1024)).expect("one holder");
    dropping.drops_writes = true;
    let dropping = Arc::new(dropping);
    let mut sync = ShadowSync::default();
    let copies = planned(&mut sync, &mut made, &[(&dropping, true)], &[]);
    assert_eq!(
        into(&copies),
        [(dropping.shadow_key, Some(CopyCause::First))]
    );
    let copies = planned(&mut sync, &mut made, &[(&dropping, false)], &[]);
    assert_eq!(
        into(&copies),
        [(dropping.shadow_key, changed(Lost::Dropped))]
    );
    assert_eq!(sync.sharing_truth(), 0);
}

/// Whether `copy` is a copy in of what changed within a rectangle (`CopyCause::Bounded`).
fn bounded(copy: &ShadowCopy<u32>) -> bool {
    copy.cause == Some(CopyCause::Bounded)
}

/// Each copy's step (into the image, or not), area, and whether it is a copy in of what changed
/// within a rectangle.
fn steps(copies: &[ShadowCopy<u32>]) -> Vec<(bool, Option<[u32; 4]>, bool)> {
    copies
        .iter()
        .map(|copy| (matches!(copy.step, Step::Into), copy.area, bounded(copy)))
        .collect()
}

/// The copies before a command that reads `texture` anywhere, then the command noted.
fn read_anywhere(
    sync: &mut ShadowSync<u32>,
    texture: &Arc<ShadowCopies>,
) -> Vec<(bool, Option<[u32; 4]>, bool)> {
    let (shadow, used) = through(texture, false);
    let shadows = [shadow];
    let uses = [used];
    let copies = sync.before(&shadows, &uses, |_, _, _| Ok(0)).unwrap();
    sync.after(&shadows, &uses, &[]);
    steps(&copies)
}

/// The copies before a command that uses `shadows`, has `others` and writes `parts` within
/// rectangles, as a pass's plan gives them (`render/shadows.rs`: every use before it), then the
/// command noted.
fn within_command(
    sync: &mut ShadowSync<u32>,
    shadows: &[ShadowUse],
    parts: &[Within],
    others: &[Use<ResourceKey>],
) -> Vec<(bool, Option<[u32; 4]>, bool)> {
    let mut uses = others.to_vec();
    uses.extend(parts.iter().flat_map(|part| part.uses.iter().cloned()));
    let copies = sync.before(shadows, &uses, |_, _, _| Ok(0)).unwrap();
    sync.after(shadows, others, parts);
    steps(&copies)
}

/// A command whose writes of a texture's bytes change its texels inside a rectangle alone
/// (`shadow::Within`: a render pass instance's draws into a color attachment it loads, inside
/// their scissors) leaves an image of those texels current but there: the next command that
/// uses it copies in the texels inside the rectangle bounding every such write since its copy
/// in, and nothing where none came. A write of the bytes another way, one of a texture with no
/// shadow, one whose texels are laid out otherwise (`ShadowCopies::same_texels`), an image whose
/// copies are not cut, and one kept for a merge copy every texel in again.
#[test]
fn writes_within_a_rectangle_copy_in_only_the_texels_they_changed() {
    let buffer = ResourceKey::new();
    let bytes_of = 256..256 + 320 * 32;
    let t = cut_shadow(buffer, bytes_of.clone());
    let attachment = bytes(buffer, bytes_of.clone(), true);
    let within = |area: [u32; 4], texels: Option<&Arc<ShadowCopies>>| Within {
        uses: vec![attachment.clone()],
        texels: texels.cloned(),
        area,
    };
    let whole = (true, None, false);
    let mut sync = ShadowSync::<u32>::default();
    assert_eq!(read_anywhere(&mut sync, &t), [whole]);
    let part = [within([8, 4, 12, 6], Some(&t))];
    assert_eq!(within_command(&mut sync, &[], &part, &[]), []);
    assert_eq!(
        read_anywhere(&mut sync, &t),
        [(true, Some([8, 4, 12, 6]), true)]
    );
    assert_eq!(read_anywhere(&mut sync, &t), []);
    // Two commands before the next read: the rectangle bounding both.
    within_command(&mut sync, &[], &[within([8, 4, 12, 6], Some(&t))], &[]);
    within_command(&mut sync, &[], &[within([30, 1, 31, 2], Some(&t))], &[]);
    assert_eq!(
        read_anywhere(&mut sync, &t),
        [(true, Some([8, 1, 31, 6]), true)]
    );
    // A texture over the same bytes, laid out texel for texel as `t`: the same rectangle.
    let twin = cut_shadow(buffer, bytes_of.clone());
    within_command(&mut sync, &[], &[within([0, 0, 2, 2], Some(&twin))], &[]);
    assert_eq!(
        read_anywhere(&mut sync, &t),
        [(true, Some([0, 0, 2, 2]), true)]
    );
    // Then a write of the bytes another way: every texel.
    within_command(&mut sync, &[], &[within([0, 0, 2, 2], Some(&t))], &[]);
    within_command(&mut sync, &[], &[], &[bytes(buffer, 256..260, true)]);
    assert_eq!(read_anywhere(&mut sync, &t), [whole]);
    // A write within whose texture has no shadow, or whose texels lie otherwise: every texel.
    within_command(&mut sync, &[], &[within([0, 0, 2, 2], None)], &[]);
    assert_eq!(read_anywhere(&mut sync, &t), [whole]);
    let other = shadow(buffer, bytes_of.clone());
    within_command(&mut sync, &[], &[within([0, 0, 2, 2], Some(&other))], &[]);
    assert_eq!(read_anywhere(&mut sync, &t), [whole]);
    // Writes elsewhere in the buffer: nothing.
    let apart = Within {
        uses: vec![bytes(buffer, 0..256, true)],
        texels: Some(Arc::clone(&t)),
        area: [0, 0, 2, 2],
    };
    within_command(&mut sync, &[], &[apart], &[]);
    assert_eq!(read_anywhere(&mut sync, &t), []);
    // A shadow whose copies are not cut: every texel.
    let uncut = shadow(buffer, bytes_of);
    let mut sync = ShadowSync::<u32>::default();
    assert_eq!(read_anywhere(&mut sync, &uncut), [whole]);
    within_command(&mut sync, &[], &[within([0, 0, 2, 2], Some(&uncut))], &[]);
    assert_eq!(read_anywhere(&mut sync, &uncut), [whole]);
    // An image written through since its copy in goes back first, then holds the bytes: only
    // what the writes within changed comes in.
    let mut sync = ShadowSync::<u32>::default();
    read_anywhere(&mut sync, &t);
    let (writing, _) = through(&t, true);
    within_command(&mut sync, std::slice::from_ref(&writing), &[], &[]);
    let part = [within([0, 0, 2, 2], Some(&t))];
    assert_eq!(
        within_command(&mut sync, &[], &part, &[]),
        [(false, None, false)]
    );
    assert_eq!(
        read_anywhere(&mut sync, &t),
        [(true, Some([0, 0, 2, 2]), true)]
    );
    // One written through by the command that writes within too is kept and merged: every texel.
    let mut sync = ShadowSync::<u32>::default();
    read_anywhere(&mut sync, &t);
    assert_eq!(
        within_command(&mut sync, &[writing], &part, &[]),
        [(false, None, false)]
    );
    assert_eq!(read_anywhere(&mut sync, &t), [(false, None, false), whole]);
}

/// A command that reads a texture only inside a rectangle and writes none (a draw that reads it
/// at its own pixels alone, `ShadowUse::loads`) takes in the texels writes within a rectangle
/// changed there alone, and leaves the others it did not move to the next command that reads
/// them: one that reads anywhere copies in the rectangle bounding those, never every texel, as
/// the image holds the bytes everywhere else.
#[test]
fn a_read_inside_a_rectangle_leaves_the_other_changed_texels_to_the_next_read() {
    let buffer = ResourceKey::new();
    let bytes_of = 256..256 + 320 * 32;
    let t = cut_shadow(buffer, bytes_of.clone());
    let attachment = bytes(buffer, bytes_of, true);
    let (_, read) = through(&t, false);
    let mut sync = ShadowSync::<u32>::default();
    let drawn = |sync: &mut ShadowSync<u32>, area: [u32; 4]| {
        let part = Within {
            uses: vec![attachment.clone()],
            texels: Some(Arc::clone(&t)),
            area,
        };
        within_command(sync, &[], &[part], &[])
    };
    let read_inside = |sync: &mut ShadowSync<u32>, area: [u32; 4]| {
        let reader = ShadowUse {
            copies: Arc::clone(&t),
            writes: false,
            overwrites: false,
            kept: false,
            area: None,
            loads: Some(area),
            tile_stores: false,
        };
        within_command(sync, &[reader], &[], std::slice::from_ref(&read))
    };
    assert_eq!(read_anywhere(&mut sync, &t), [(true, None, false)]);
    // Read inside a part of what changed: that part, then the rest of it for a read anywhere.
    drawn(&mut sync, [8, 4, 40, 20]);
    assert_eq!(
        read_inside(&mut sync, [10, 5, 20, 10]),
        [(true, Some([10, 5, 20, 10]), true)]
    );
    assert_eq!(
        read_anywhere(&mut sync, &t),
        [(true, Some([8, 4, 40, 20]), true)]
    );
    assert_eq!(read_anywhere(&mut sync, &t), []);
    // A read across it from top to bottom leaves the columns beside it; one from side to side,
    // the rows; one past it, nothing.
    drawn(&mut sync, [8, 4, 40, 20]);
    assert_eq!(
        read_inside(&mut sync, [0, 0, 20, 32]),
        [(true, Some([8, 4, 20, 20]), true)]
    );
    assert_eq!(
        read_anywhere(&mut sync, &t),
        [(true, Some([20, 4, 40, 20]), true)]
    );
    drawn(&mut sync, [8, 4, 40, 20]);
    read_inside(&mut sync, [0, 10, 64, 32]);
    assert_eq!(
        read_anywhere(&mut sync, &t),
        [(true, Some([8, 4, 40, 10]), true)]
    );
    drawn(&mut sync, [8, 4, 40, 20]);
    read_inside(&mut sync, [0, 0, 64, 32]);
    assert_eq!(read_anywhere(&mut sync, &t), []);
    // A read apart from it moves nothing, and leaves all of it.
    drawn(&mut sync, [8, 4, 12, 6]);
    assert_eq!(read_inside(&mut sync, [30, 20, 40, 30]), []);
    assert_eq!(
        read_anywhere(&mut sync, &t),
        [(true, Some([8, 4, 12, 6]), true)]
    );
}

/// A copy back of a part of a shadowed attachment's image (a pass's attachment of a texture over a
/// tracked `Private` buffer whose own image is a shadow, written inside its draws' scissors)
/// leaves another image laid out as it, texel for texel, current but inside the part: the next
/// command that reads that image copies in those texels alone. An image laid out otherwise copies
/// every texel in.
#[test]
fn a_copy_back_of_a_part_leaves_a_twin_stale_there_alone() {
    let buffer = ResourceKey::new();
    let bytes_of = 256..256 + 320 * 32;
    let attachment = over_private(cut_shadow(buffer, bytes_of.clone()), true);
    let read = over_private(cut_shadow(buffer, bytes_of.clone()), true);
    let other = over_private(shadow(buffer, bytes_of), true);
    let (_, written) = through(&attachment, true);
    let instance = |sync: &mut ShadowSync<u32>, area: [u32; 4], also: &[&Arc<ShadowCopies>]| {
        let mut shadows = vec![ShadowUse {
            copies: Arc::clone(&attachment),
            writes: true,
            overwrites: false,
            kept: false,
            area: Some(area),
            loads: Some([0, 0, 64, 32]),
            tile_stores: false,
        }];
        let mut uses = vec![written.clone()];
        for texture in also {
            let (shadow, used) = through(texture, false);
            shadows.push(shadow);
            uses.push(used);
        }
        let copies = sync.before(&shadows, &uses, |_, _, _| Ok(0)).unwrap();
        sync.after(&shadows, &uses, &[]);
        copies
            .iter()
            .map(|copy| {
                let key = copy.copies.shadow_key;
                (
                    key,
                    matches!(copy.step, Step::Into),
                    copy.area,
                    bounded(copy),
                )
            })
            .collect::<Vec<_>>()
    };
    let mut sync = ShadowSync::<u32>::default();
    let all = Some([0, 0, 64, 32]);
    let a = attachment.shadow_key;
    let (r, o) = (read.shadow_key, other.shadow_key);
    assert_eq!(
        instance(&mut sync, [8, 4, 12, 6], &[]),
        [(a, true, all, false)]
    );
    assert_eq!(
        instance(&mut sync, [20, 4, 24, 6], &[&read, &other]),
        [
            (a, false, Some([8, 4, 12, 6]), false),
            (r, true, None, false),
            (o, true, None, false),
        ]
    );
    assert_eq!(
        instance(&mut sync, [0, 0, 1, 1], &[&read, &other]),
        [
            (a, false, Some([20, 4, 24, 6]), false),
            (r, true, Some([20, 4, 24, 6]), true),
            (o, true, None, false),
        ]
    );
}

/// A shadowed attachment over a buffer's bytes another writer may write (a pass's attachment of a
/// texture whose own image is a shadow), written inside its draws' scissors, is kept and merged
/// back: the merge changes only words the draws changed, inside the part they wrote, so it leaves
/// another image laid out as the attachment's current but inside that part, as a copy back of the
/// part does ([`a_copy_back_of_a_part_leaves_a_twin_stale_there_alone`]): the next command that
/// reads that image copies in those texels alone. An image laid out otherwise copies every texel
/// in.
#[test]
fn a_merge_of_a_part_leaves_a_twin_stale_there_alone() {
    let buffer = ResourceKey::new();
    let bytes_of = 256..256 + 320 * 32;
    let attachment = cut_shadow(buffer, bytes_of.clone());
    let read = cut_shadow(buffer, bytes_of.clone());
    let other = shadow(buffer, bytes_of);
    assert!(attachment.merges_back());
    let (_, written) = through(&attachment, true);
    let instance = |sync: &mut ShadowSync<u32>, area: [u32; 4], also: &[&Arc<ShadowCopies>]| {
        let mut shadows = vec![ShadowUse {
            copies: Arc::clone(&attachment),
            writes: true,
            overwrites: false,
            kept: false,
            area: Some(area),
            loads: Some([0, 0, 64, 32]),
            tile_stores: false,
        }];
        let mut uses = vec![written.clone()];
        for texture in also {
            let (shadow, used) = through(texture, false);
            shadows.push(shadow);
            uses.push(used);
        }
        let copies = sync.before(&shadows, &uses, |_, _, _| Ok(0)).unwrap();
        sync.after(&shadows, &uses, &[]);
        copies
            .iter()
            .map(|copy| (copy.copies.shadow_key, step(copy), copy.area, bounded(copy)))
            .collect::<Vec<_>>()
    };
    let mut sync = ShadowSync::<u32>::default();
    let all = Some([0, 0, 64, 32]);
    let a = attachment.shadow_key;
    let (r, o) = (read.shadow_key, other.shadow_key);
    assert_eq!(
        instance(&mut sync, [8, 4, 12, 6], &[]),
        [(a, 'i', all, false), (a, 'k', None, false)]
    );
    assert_eq!(
        instance(&mut sync, [20, 4, 24, 6], &[&read, &other]),
        [
            (a, 'm', None, false),
            (r, 'i', None, false),
            (o, 'i', None, false),
            (a, 'k', None, false),
        ]
    );
    assert_eq!(
        instance(&mut sync, [0, 0, 1, 1], &[&read, &other]),
        [
            (a, 'm', None, false),
            (r, 'i', Some([20, 4, 24, 6]), true),
            (o, 'i', None, false),
            (a, 'k', None, false),
        ]
    );
}

/// A copy back of a part of a one-level, one-slice texture's image is its region cut to the
/// rectangle: the buffer's offset moved to the rectangle's first texel at the region's row length,
/// clamped to the image, and nothing where no texel lies inside.
#[test]
fn a_part_of_a_region_starts_at_its_first_texel() {
    let region = vk::BufferImageCopy {
        buffer_offset: 256,
        buffer_row_length: 20,
        buffer_image_height: 0,
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        image_offset: vk::Offset3D::default(),
        image_extent: vk::Extent3D {
            width: 16,
            height: 8,
            depth: 1,
        },
    };
    let part = shadow::part_of(&region, [2, 1, 6, 3], 4).unwrap();
    assert_eq!(part.buffer_offset, 256 + (20 + 2) * 4);
    assert_eq!(part.buffer_row_length, 20);
    assert_eq!(
        (
            part.image_offset.x,
            part.image_offset.y,
            part.image_offset.z
        ),
        (2, 1, 0)
    );
    assert_eq!(
        (
            part.image_extent.width,
            part.image_extent.height,
            part.image_extent.depth
        ),
        (4, 2, 1)
    );
    let clamped = shadow::part_of(&region, [12, 6, 40, 40], 4).unwrap();
    assert_eq!(
        (clamped.image_extent.width, clamped.image_extent.height),
        (4, 2)
    );
    assert!(shadow::part_of(&region, [16, 0, 20, 8], 4).is_none());
    assert!(shadow::part_of(&region, [3, 3, 3, 5], 4).is_none());
    // A region of tight rows (row length 0) steps by its width.
    let tight = vk::BufferImageCopy {
        buffer_row_length: 0,
        ..region
    };
    let part = shadow::part_of(&tight, [0, 2, 16, 3], 4).unwrap();
    assert_eq!(part.buffer_offset, 256 + 2 * 16 * 4);
}

/// A shadow whose writes Metal drops (a rotated texture's functions' image) is copied in as any
/// shadow, never copied back, and copied in again at its next use after a command that wrote
/// through it; a command writing it and its bytes another way is no two-way write.
#[test]
fn a_shadow_whose_writes_are_dropped_is_copied_in_again_and_never_back() {
    let buffer = ResourceKey::new();
    let mut copies = Arc::into_inner(shadow(buffer, 256..1280)).expect("one holder");
    copies.drops_writes = true;
    let t = Arc::new(copies);
    let mut sync = ShadowSync::default();
    assert_eq!(
        command(&mut sync, &[(&t, false)], &[]),
        [(t.shadow_key, true)]
    );
    assert_eq!(command(&mut sync, &[(&t, false)], &[]), []);
    assert_eq!(command(&mut sync, &[(&t, true)], &[]), []);
    assert_eq!(
        command(&mut sync, &[(&t, false)], &[]),
        [(t.shadow_key, true)]
    );
    assert_eq!(
        command(&mut sync, &[(&t, true)], &[bytes(buffer, 256..512, true)]),
        []
    );
    assert_eq!(named(&sync.finish()), []);
}

/// A use that replaces every texel before it reads any (a pass's attachment that clears the
/// whole of a one-level, one-slice texture, `ShadowUse::overwrites`) takes no copy in and no
/// keep, and the image is current after it: storing it, the pass's writes are copied back as the
/// encoder ends; not storing it, nothing is, and the buffer keeps its bytes. Joined with a use
/// that reads, it is copied in and kept for its merge back (`shadow::join`), and a texture kept
/// to merge a write two ways still copies in.
#[test]
fn a_use_that_overwrites_every_texel_takes_no_copy_in() {
    let buffer = ResourceKey::new();
    let t = shadow(buffer, 256..1280);
    let overwriting = |writes: bool| ShadowUse {
        copies: Arc::clone(&t),
        writes,
        overwrites: true,
        kept: false,
        area: None,
        loads: None,
        tile_stores: false,
    };
    let (_, written) = through(&t, true);
    for stores in [true, false] {
        let mut sync = ShadowSync::<u32>::default();
        let shadows = [overwriting(stores)];
        let uses = [written.clone()];
        let copies = sync.before(&shadows, &uses, |_, _, _| Ok(0)).unwrap();
        assert_eq!(named(&copies), [], "stores {stores}");
        sync.after(&shadows, &uses, &[]);
        assert_eq!(
            command(&mut sync, &[(&t, false)], &[]),
            [],
            "stores {stores}"
        );
        let back: &[(ResourceKey, bool)] = if stores {
            &[(t.shadow_key, false)]
        } else {
            &[]
        };
        assert_eq!(named(&sync.finish()), back, "stores {stores}");
    }
    let mut joined = vec![overwriting(true)];
    shadow::join(&mut joined, &through(&t, false).0);
    assert!(!joined[0].overwrites && joined[0].writes);
    let mut sync = ShadowSync::<u32>::default();
    let copies = sync
        .before(&joined, std::slice::from_ref(&written), |_, _, _| Ok(0))
        .unwrap();
    assert_eq!(
        moved(&copies),
        [Moved::Into(t.shadow_key), Moved::Keep(t.shadow_key, 0)]
    );
    // Written two ways (the bytes too), the texture is kept and copied in.
    let mut sync = ShadowSync::<u32>::default();
    let uses = [written, bytes(buffer, 256..512, true)];
    let copies = sync
        .before(&[overwriting(true)], &uses, |_, _, _| Ok(7))
        .unwrap();
    assert_eq!(
        moved(&copies),
        [Moved::Into(t.shadow_key), Moved::Keep(t.shadow_key, 7)]
    );
}

/// A command that touches a texture's bytes another way takes the merge of what was written
/// through it first: a read of the bytes leaves the image current, a write leaves it to copy in
/// again at its next use. Bytes elsewhere, or another buffer's, take nothing.
#[test]
fn a_command_touching_the_bytes_another_way_orders_the_copies_around_it() {
    let buffer = ResourceKey::new();
    let t = shadow(buffer, 256..1280);
    let k = t.shadow_key;
    let mut made = 0;
    let mut run = |sync: &mut ShadowSync<u32>,
                   textures: &[(&Arc<ShadowCopies>, bool)],
                   others: &[Use<ResourceKey>]| {
        moved(&planned(sync, &mut made, textures, others))
    };
    let mut sync = ShadowSync::default();
    assert_eq!(
        run(&mut sync, &[(&t, true)], &[]),
        [Moved::Into(k), Moved::Keep(k, 1)]
    );
    // Bytes elsewhere in the buffer, or another buffer's: nothing to copy.
    assert_eq!(run(&mut sync, &[], &[bytes(buffer, 0..256, true)]), []);
    let other = ResourceKey::new();
    assert_eq!(run(&mut sync, &[], &[bytes(other, 0..4096, true)]), []);
    // A read of the bytes: the texture's writes first; the image stays current.
    assert_eq!(
        run(&mut sync, &[], &[bytes(buffer, 0..4096, false)]),
        [Moved::Merge(k, 1)]
    );
    assert_eq!(run(&mut sync, &[(&t, false)], &[]), []);
    // A write of the bytes: the next use copies them in again.
    assert_eq!(run(&mut sync, &[(&t, true)], &[]), [Moved::Keep(k, 1)]);
    assert_eq!(
        run(&mut sync, &[], &[bytes(buffer, 1024..1028, true)]),
        [Moved::Merge(k, 1)]
    );
    assert_eq!(run(&mut sync, &[(&t, false)], &[]), [Moved::Into(k)]);
    assert_eq!(moved(&sync.finish()), []);
}

/// Textures over the same bytes see each other's writes: one written is merged back before
/// another over its bytes is copied in, and leaves the other to copy in again at its next use. A
/// texture over other bytes is left alone.
#[test]
fn textures_over_the_same_bytes_see_each_others_writes() {
    let buffer = ResourceKey::new();
    let a = shadow(buffer, 0..1024);
    let b = shadow(buffer, 512..2048);
    let apart = shadow(buffer, 4096..8192);
    let (ka, kb, kapart) = (a.shadow_key, b.shadow_key, apart.shadow_key);
    let mut made = 0;
    let mut run = |sync: &mut ShadowSync<u32>, textures: &[(&Arc<ShadowCopies>, bool)]| {
        moved(&planned(sync, &mut made, textures, &[]))
    };
    let mut sync = ShadowSync::default();
    assert_eq!(
        run(&mut sync, &[(&a, true), (&apart, false)]),
        [Moved::Into(ka), Moved::Into(kapart), Moved::Keep(ka, 1)]
    );
    // `a` written: `b` is copied in after `a` is merged back.
    assert_eq!(
        run(&mut sync, &[(&b, false)]),
        [Moved::Merge(ka, 1), Moved::Into(kb)]
    );
    // `b` written: `a` is copied in again at its next use, after `b`'s merge.
    assert_eq!(run(&mut sync, &[(&b, true)]), [Moved::Keep(kb, 2)]);
    assert_eq!(
        run(&mut sync, &[(&a, false), (&apart, false)]),
        [Moved::Merge(kb, 2), Moved::Into(ka)]
    );
    assert_eq!(moved(&sync.finish()), []);
}

/// Two textures over two no-copy buffers made over the same host memory, the second of which
/// joined the first's (`host_aliases.rs`): each sees the other's writes as textures over one
/// buffer do, and a write of the second buffer's bytes, which carries a use of the first's key
/// (`HostAliases::augment`), makes the first copy in again; a third buffer's do not.
#[test]
fn textures_over_two_buffers_of_one_memory_see_each_others_writes() {
    let (first, second, other) = (ResourceKey::new(), ResourceKey::new(), ResourceKey::new());
    let a = shadow(first, 0..1024);
    let b = shadow_joined(second, Some(first), 0..1024);
    let c = shadow(other, 0..1024);
    let aliases: KeyMap<_, _> = [(second, Alias { key: first })].into_iter().collect();
    let augmented = |uses: &[Use<ResourceKey>]| {
        let mut all = Vec::new();
        match host_aliases::augmented(Some(&aliases), None, uses, &mut all) {
            true => all,
            false => uses.to_vec(),
        }
    };
    let (ka, kb, kc) = (a.shadow_key, b.shadow_key, c.shadow_key);
    let mut made = 0;
    let mut run = |sync: &mut ShadowSync<u32>,
                   textures: &[(&Arc<ShadowCopies>, bool)],
                   others: &[Use<ResourceKey>]| {
        moved(&planned(sync, &mut made, textures, others))
    };
    let mut sync = ShadowSync::default();
    assert_eq!(
        run(&mut sync, &[(&a, true), (&c, false)], &[]),
        [Moved::Into(ka), Moved::Into(kc), Moved::Keep(ka, 1)]
    );
    // `a` written: `b` is copied in after `a` is merged back; `c` lies elsewhere.
    assert_eq!(
        run(&mut sync, &[(&b, false), (&c, false)], &[]),
        [Moved::Merge(ka, 1), Moved::Into(kb)]
    );
    // `b` written: `a` is copied in again at its next use, after `b`'s merge.
    assert_eq!(run(&mut sync, &[(&b, true)], &[]), [Moved::Keep(kb, 2)]);
    assert_eq!(
        run(&mut sync, &[(&a, false)], &[]),
        [Moved::Merge(kb, 2), Moved::Into(ka)]
    );
    // The second buffer's bytes written: `a` copies in again, and so does `b`.
    let written = augmented(&[bytes(second, 0..64, true)]);
    assert_eq!(written.len(), 2);
    assert_eq!(run(&mut sync, &[], &written), []);
    assert_eq!(
        run(&mut sync, &[(&a, false), (&b, false)], &[]),
        [Moved::Into(ka), Moved::Into(kb)]
    );
    // The third buffer's bytes written: neither does.
    run(&mut sync, &[], &augmented(&[bytes(other, 0..64, true)]));
    assert_eq!(run(&mut sync, &[(&a, false), (&b, false)], &[]), []);
    assert_eq!(moved(&sync.finish()), []);
}

/// One command that writes through a texture and its bytes another way (the buffer, or another
/// texture over them) keeps the texture after its copies in, in a scratch of its own, and the
/// merge goes before the next command, whatever it touches, or at the end, after which the
/// texture is copied in again at its next use. Reading the bytes, writing bytes the texture does
/// not lie in, or reading the other texture is one way: kept in the encoder's scratch, and merged
/// back at the end.
#[test]
fn one_command_writing_a_texture_and_its_bytes_another_way_keeps_and_merges_it() {
    let buffer = ResourceKey::new();
    let a = shadow(buffer, 0..1024);
    let b = shadow(buffer, 512..2048);
    let elsewhere = ResourceKey::new();
    let (ka, kb) = (a.shadow_key, b.shadow_key);
    let mut sync = ShadowSync::default();
    let mut made = 0;
    let mut run = |sync: &mut ShadowSync<u32>,
                   textures: &[(&Arc<ShadowCopies>, bool)],
                   others: &[Use<ResourceKey>]| {
        moved(&planned(sync, &mut made, textures, others))
    };
    assert_eq!(
        run(&mut sync, &[(&a, true)], &[bytes(buffer, 0..4096, true)]),
        [Moved::Into(ka), Moved::Keep(ka, 1)]
    );
    // The merge before a command of other bytes, which leaves `a` to copy in again.
    assert_eq!(
        run(&mut sync, &[], &[bytes(elsewhere, 0..64, true)]),
        [Moved::Merge(ka, 1)]
    );
    assert_eq!(run(&mut sync, &[(&a, false)], &[]), [Moved::Into(ka)]);
    // Two textures over the same bytes, both written: each kept, each merged at the end.
    assert_eq!(
        run(&mut sync, &[(&a, true), (&b, true)], &[]),
        [Moved::Into(kb), Moved::Keep(ka, 2), Moved::Keep(kb, 3)]
    );
    assert_eq!(
        moved(&sync.finish()),
        [Moved::Merge(ka, 2), Moved::Merge(kb, 3)]
    );
    // Two ways twice in a row: the first merge, `a` copied in again, its second keep.
    let mut sync = ShadowSync::default();
    let written = [bytes(buffer, 1020..1028, true)];
    run(&mut sync, &[(&a, true)], &written);
    assert_eq!(
        run(&mut sync, &[(&a, true)], &written),
        [Moved::Merge(ka, 4), Moved::Into(ka), Moved::Keep(ka, 5)]
    );
    assert_eq!(moved(&sync.finish()), [Moved::Merge(ka, 5)]);
    // One way: the encoder's scratch, merged at the end.
    for (n, others) in [
        (6, vec![bytes(buffer, 0..4096, false)]),
        (7, vec![bytes(buffer, 1024..4096, true)]),
    ] {
        let mut sync = ShadowSync::default();
        assert_eq!(
            run(&mut sync, &[(&a, true)], &others),
            [Moved::Into(ka), Moved::Keep(ka, n)]
        );
        assert_eq!(moved(&sync.finish()), [Moved::Merge(ka, n)]);
    }
    let mut sync = ShadowSync::default();
    assert_eq!(
        run(&mut sync, &[(&a, true), (&b, false)], &[]),
        [Moved::Into(ka), Moved::Into(kb), Moved::Keep(ka, 8)]
    );
}

/// A texture over a buffer's bytes (`ShadowCopies::merges_back`): the image is kept after its copy
/// in, before the first command since the last copy back that writes through it, and that copy
/// back is a merge, in one scratch of the encoder's, after which the image still holds the bytes:
/// a later read takes no copy in, and a later write keeps it again in the same scratch. A command
/// that touches its bytes another way takes the merge first. A command that replaces every texel
/// keeps nothing and is copied back, before the keep of a later command that writes through it
/// without replacing them; one that writes the bytes two ways is kept in a scratch of its own and
/// merged before the next command. An image's shadow, one whose writes Metal drops, and one over a
/// tracked `Private` buffer's bytes, which no other writer writes while a command buffer runs,
/// merge nothing back: the last is copied in and back; an untracked `Private` buffer's merges.
#[test]
fn a_texture_over_a_buffer_s_bytes_is_merged_where_it_would_be_copied_back() {
    let buffer = ResourceKey::new();
    let a = shadow(buffer, 0..1024);
    let ka = a.shadow_key;
    assert!(a.merges_back());
    assert!(!image_shadow(ResourceKey::new(), 0, 1).merges_back());
    assert!(over_private(shadow(buffer, 0..1024), false).merges_back());
    let private = over_private(shadow(buffer, 0..1024), true);
    assert!(!private.merges_back());
    let mut dropped = Arc::into_inner(shadow(buffer, 0..1024)).expect("one holder");
    dropped.drops_writes = true;
    assert!(!dropped.merges_back());
    let mut made = 0;
    let mut run = |sync: &mut ShadowSync<u32>,
                   textures: &[(&Arc<ShadowCopies>, bool)],
                   others: &[Use<ResourceKey>]| {
        moved(&planned(sync, &mut made, textures, others))
    };
    let mut sync = ShadowSync::default();
    assert_eq!(
        run(&mut sync, &[(&a, true)], &[]),
        [Moved::Into(ka), Moved::Keep(ka, 1)]
    );
    assert_eq!(run(&mut sync, &[(&a, true)], &[]), [], "kept already");
    assert_eq!(run(&mut sync, &[(&a, false)], &[]), [], "current");
    assert_eq!(
        run(&mut sync, &[], &[bytes(buffer, 0..64, false)]),
        [Moved::Merge(ka, 1)]
    );
    assert_eq!(
        run(&mut sync, &[(&a, true)], &[]),
        [Moved::Keep(ka, 1)],
        "current after the merge, kept again in the encoder's scratch"
    );
    assert_eq!(moved(&sync.finish()), [Moved::Merge(ka, 1)]);
    // Another encoder: a scratch of its own.
    let mut sync = ShadowSync::default();
    assert_eq!(
        run(&mut sync, &[(&a, true)], &[]),
        [Moved::Into(ka), Moved::Keep(ka, 2)]
    );
    assert_eq!(moved(&sync.finish()), [Moved::Merge(ka, 2)]);
    // Over a tracked `Private` buffer's bytes: copied in and back, nothing kept.
    let kp = private.shadow_key;
    let mut sync = ShadowSync::default();
    assert_eq!(run(&mut sync, &[(&private, true)], &[]), [Moved::Into(kp)]);
    assert_eq!(run(&mut sync, &[(&private, true)], &[]), []);
    assert_eq!(moved(&sync.finish()), [Moved::Back(kp)]);
    // Every texel replaced: no copy in, no keep, a copy back.
    let mut sync = ShadowSync::<u32>::default();
    let replacing = ShadowUse {
        overwrites: true,
        ..through(&a, true).0
    };
    let uses = [through(&a, true).1];
    let copies = sync
        .before(std::slice::from_ref(&replacing), &uses, |_, _, _| Ok(9))
        .unwrap();
    sync.after(std::slice::from_ref(&replacing), &uses, &[]);
    assert_eq!(moved(&copies), []);
    assert_eq!(moved(&sync.finish()), [Moved::Back(ka)]);
    // Every texel replaced, then written through without replacing them: the first command's
    // writes are copied back before the keep, as the merge after it takes only the words changed
    // since the keep.
    let mut sync = ShadowSync::<u32>::default();
    sync.before(std::slice::from_ref(&replacing), &uses, |_, _, _| Ok(9))
        .unwrap();
    sync.after(std::slice::from_ref(&replacing), &uses, &[]);
    assert_eq!(
        run(&mut sync, &[(&a, true)], &[]),
        [Moved::Back(ka), Moved::Keep(ka, 3)]
    );
    assert_eq!(moved(&sync.finish()), [Moved::Merge(ka, 3)]);
    // Two ways: the command's own scratch, merged before the next command, which copies in.
    let mut sync = ShadowSync::default();
    assert_eq!(
        run(&mut sync, &[(&a, true)], &[bytes(buffer, 0..4096, true)]),
        [Moved::Into(ka), Moved::Keep(ka, 4)]
    );
    assert_eq!(
        run(&mut sync, &[(&a, true)], &[]),
        [Moved::Merge(ka, 4), Moved::Into(ka), Moved::Keep(ka, 5)]
    );
    assert_eq!(moved(&sync.finish()), [Moved::Merge(ka, 5)]);
}

/// A command whose scratch is refused changes nothing: the next command copies the texture in.
#[test]
fn a_refused_scratch_changes_nothing() {
    let buffer = ResourceKey::new();
    let a = shadow(buffer, 0..1024);
    let mut sync = ShadowSync::<u32>::default();
    let (shadows, uses) = command_uses(&[(&a, true)], &[bytes(buffer, 0..4096, true)]);
    let refused = sync.before(&shadows, &uses, |_, _, _| Err(internal("no scratch")));
    assert!(
        matches!(refused, Err(Error::Compile(_))),
        "{:?}",
        refused.err()
    );
    assert_eq!(
        command(&mut sync, &[(&a, false)], &[]),
        [(a.shadow_key, true)]
    );
}

/// A shadow of `levels` and `layers` of the image `image`: a texture view of another component
/// layout on a device whose Vulkan views reinterpret none (`texture/view_shadow.rs`).
fn view_of(image: ResourceKey, levels: Range<u32>, layers: Range<u32>) -> Arc<ShadowCopies> {
    let range = UseRange::Image {
        aspects: vk::ImageAspectFlags::COLOR,
        levels,
        layers,
    };
    Arc::new(ShadowCopies {
        source: ShadowSource::Image {
            image: vk::Image::null(),
            key: image,
            range: range.clone(),
            shadow: vk::Image::null(),
            regions: Vec::new(),
            through: None,
        },
        shadow_key: ResourceKey::new(),
        shadow_range: range,
        tracked: true,
        drops_writes: false,
        backing: Backing::Image,
        merge_scratch: ScratchSpares::default(),
        touched: AtomicU64::new(0),
        stood: AtomicU64::new(0),
    })
}

/// A use of `levels` and `layers` of the image `image`, writing when `writes`.
fn texels(
    image: ResourceKey,
    levels: Range<u32>,
    layers: Range<u32>,
    writes: bool,
) -> Use<ResourceKey> {
    let range = UseRange::Image {
        aspects: vk::ImageAspectFlags::COLOR,
        levels,
        layers,
    };
    if writes {
        Use::writes(image, true, range, shader())
    } else {
        Use::reads(image, true, range, shader())
    }
}

/// A view's shadow of an image follows the rules a texture over a buffer does, the image in the
/// buffer's place: copied in before its first use, back before a command that touches the
/// image's levels and layers it presents another way and at the end of the encoder, and in again
/// after a command writes them another way. Other levels or layers, or another image, touch
/// nothing.
#[test]
fn a_views_shadow_of_an_image_is_copied_as_a_shadow_of_bytes_is() {
    let image = ResourceKey::new();
    let v = view_of(image, 1..2, 0..2);
    let mut sync = ShadowSync::default();
    assert_eq!(
        command(&mut sync, &[(&v, false)], &[]),
        [(v.shadow_key, true)]
    );
    assert_eq!(command(&mut sync, &[(&v, true)], &[]), []);
    // Another level, another layer, another image: nothing to copy.
    let other = ResourceKey::new();
    for elsewhere in [
        texels(image, 0..1, 0..2, true),
        texels(image, 1..2, 2..3, true),
        texels(other, 0..4, 0..4, true),
    ] {
        assert_eq!(command(&mut sync, &[], &[elsewhere]), []);
    }
    // A read of the texels: the view's writes first; the shadow stays current.
    assert_eq!(
        command(&mut sync, &[], &[texels(image, 0..2, 1..2, false)]),
        [(v.shadow_key, false)]
    );
    assert_eq!(command(&mut sync, &[(&v, false)], &[]), []);
    // A write of them: the next use copies them in again.
    assert_eq!(
        command(&mut sync, &[], &[texels(image, 1..2, 0..1, true)]),
        []
    );
    assert_eq!(
        command(&mut sync, &[(&v, true)], &[]),
        [(v.shadow_key, true)]
    );
    assert_eq!(named(&sync.finish()), [(v.shadow_key, false)]);
}

/// One command that writes an image both through a view's shadow and another way (the image,
/// or another view's shadow of the same texels) is refused, typed, before any merge is made
/// (only a buffer's bytes are merged), and changes nothing. Writing through one alias and reading
/// through the other, or writing levels the shadow does not present, is one way: the shadow
/// holds the texels as they were before the command.
#[test]
fn a_command_writing_an_image_through_a_views_shadow_and_another_way_is_refused() {
    let image = ResourceKey::new();
    let v = view_of(image, 0..1, 0..1);
    let w = view_of(image, 0..2, 0..1);
    let refusals: [(&[(&Arc<ShadowCopies>, bool)], Vec<Use<ResourceKey>>); 3] = [
        (&[(&v, true)], vec![texels(image, 0..1, 0..1, true)]),
        (&[(&v, true), (&w, true)], Vec::new()),
        (
            &[(&v, false), (&w, true)],
            vec![texels(image, 0..1, 0..1, true)],
        ),
    ];
    for (textures, others) in refusals {
        let mut sync = ShadowSync::<u32>::default();
        let (shadows, uses) = command_uses(textures, &others);
        let refused = sync.before(&shadows, &uses, |_, _, _| {
            panic!("a merge of an image's shadow")
        });
        match refused {
            Err(Error::Unsupported(gap)) => {
                assert_eq!(gap.feature, Feature::ViewFormatReinterpretation)
            }
            other => panic!("{:?}", other.map(|copies| named(&copies))),
        }
        assert_eq!(
            command(&mut sync, &[(&v, false)], &[]),
            [(v.shadow_key, true)]
        );
    }
    let one_way: [(&[(&Arc<ShadowCopies>, bool)], Vec<Use<ResourceKey>>); 4] = [
        (&[(&v, true)], vec![texels(image, 0..1, 0..1, false)]),
        (&[(&v, false)], vec![texels(image, 0..1, 0..1, true)]),
        (&[(&v, true), (&w, false)], Vec::new()),
        (&[(&v, true)], vec![texels(image, 1..2, 0..1, true)]),
    ];
    for (textures, others) in one_way {
        let mut sync = ShadowSync::<u32>::default();
        let (shadows, uses) = command_uses(textures, &others);
        let copies = sync
            .before(&shadows, &uses, |_, _, _| {
                panic!("a merge of an image's shadow")
            })
            .unwrap();
        let into: Vec<(ResourceKey, bool)> = textures
            .iter()
            .map(|(texture, _)| (texture.shadow_key, true))
            .collect();
        assert_eq!(named(&copies), into);
    }
}

/// A shadow of level `level` and slice 0 of the image `image`, 64x32 there, copied image to image
/// (a 2D texture's own image's, `texture/usage_shadow.rs`), with `layers` slices in its one region.
fn image_shadow(image: ResourceKey, level: u32, layers: u32) -> Arc<ShadowCopies> {
    let mut copies =
        Arc::into_inner(view_of(image, level..level + 1, 0..layers)).expect("one holder");
    if let ShadowSource::Image { regions, .. } = &mut copies.source {
        let subresource = vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: level,
            base_array_layer: 0,
            layer_count: layers,
        };
        regions.push(vk::ImageCopy {
            src_subresource: subresource,
            dst_subresource: subresource,
            extent: vk::Extent3D {
                width: 64,
                height: 32,
                depth: 1,
            },
            ..Default::default()
        });
    }
    Arc::new(copies)
}

/// Each copy's step (into the shadow, or not), area, and why a copy in is made.
fn causes(copies: &[ShadowCopy<u32>]) -> Vec<(bool, Option<[u32; 4]>, Option<CopyCause>)> {
    copies
        .iter()
        .map(|copy| (matches!(copy.step, Step::Into), copy.area, copy.cause))
        .collect()
}

/// A shadow of one level and one slice of an image, copied image to image, is cut to rectangles
/// as a one-level, one-slice texture's over a buffer is (`ShadowCopies::cuts`): a read inside a
/// rectangle copies in the texels there, a later read anywhere every texel (beyond what the
/// shadow held), writes within a rectangle of the image the texels they changed alone, and each
/// cut copy moves the texels inside its rectangle. Its copies back move every texel, so a command
/// that writes through it copies every texel in first. A shadow of several slices is not cut.
#[test]
fn an_images_shadow_of_one_level_and_slice_is_cut_as_a_buffers_texture_is() {
    let image = ResourceKey::new();
    let s = image_shadow(image, 2, 1);
    assert!(s.cuts());
    assert_eq!(s.texels(), 64 * 32);
    assert!(!image_shadow(image, 2, 2).cuts());
    let (_, read) = through(&s, false);
    let reader = |area: [u32; 4]| ShadowUse {
        loads: Some(area),
        tile_stores: false,
        ..through(&s, false).0
    };
    let attachment = texels(image, 2..3, 0..1, true);
    let drawn = |area: [u32; 4]| Within {
        uses: vec![attachment.clone()],
        texels: Some(Arc::clone(&s)),
        area,
    };
    let command = |sync: &mut ShadowSync<u32>, shadows: &[ShadowUse], parts: &[Within]| {
        let mut uses = vec![read.clone()];
        uses.extend(parts.iter().flat_map(|part| part.uses.iter().cloned()));
        let copies = sync.before(shadows, &uses, |_, _, _| Ok(0)).unwrap();
        sync.after(shadows, std::slice::from_ref(&read), parts);
        copies
    };
    let mut sync = ShadowSync::<u32>::default();
    let a = [8, 4, 12, 6];
    let copies = command(&mut sync, &[reader(a)], &[]);
    assert_eq!(causes(&copies), [(true, Some(a), Some(CopyCause::First))]);
    assert_eq!(copies[0].texels_copied_in(), 4 * 2);
    // Inside what the shadow holds: nothing.
    assert!(command(&mut sync, &[reader([9, 4, 10, 5])], &[]).is_empty());
    // A read anywhere: every texel, beyond what the shadow held.
    let anywhere = through(&s, false).0;
    let copies = command(&mut sync, std::slice::from_ref(&anywhere), &[]);
    assert_eq!(causes(&copies), [(true, None, Some(CopyCause::BeyondHeld))]);
    assert_eq!(copies[0].texels_copied_in(), 64 * 32);
    // Draws within a rectangle of the image's level: the texels they changed alone.
    command(&mut sync, &[], &[drawn([20, 10, 30, 14])]);
    let copies = command(&mut sync, std::slice::from_ref(&anywhere), &[]);
    assert_eq!(
        causes(&copies),
        [(true, Some([20, 10, 30, 14]), Some(CopyCause::Bounded))]
    );
    assert_eq!(copies[0].texels_copied_in(), 10 * 4);
    // A write of the level another way: every texel, after the change.
    sync.after(&[], &[texels(image, 2..3, 0..1, true)], &[]);
    let copies = command(&mut sync, std::slice::from_ref(&anywhere), &[]);
    let written = Some(CopyCause::Changed(Lost::Written));
    assert_eq!(causes(&copies), [(true, None, written)]);
    // Writes within a rectangle whose texels are a shadow's laid out otherwise (another level's):
    // every texel.
    let other_level = image_shadow(image, 1, 1);
    let elsewhere = Within {
        uses: vec![texels(image, 2..3, 0..1, true)],
        texels: Some(Arc::clone(&other_level)),
        area: a,
    };
    command(&mut sync, &[], &[elsewhere]);
    let copies = command(&mut sync, std::slice::from_ref(&anywhere), &[]);
    let within = Some(CopyCause::Changed(Lost::Within));
    assert_eq!(causes(&copies), [(true, None, within)]);
    // A command writing through it reads and writes inside a rectangle, yet takes every texel in:
    // its copy back moves every texel.
    let mut sync = ShadowSync::<u32>::default();
    command(&mut sync, &[reader(a)], &[]);
    let writer = ShadowUse {
        writes: true,
        area: Some(a),
        loads: Some(a),
        tile_stores: false,
        ..through(&s, true).0
    };
    let written = through(&s, true).1;
    let copies = sync
        .before(
            std::slice::from_ref(&writer),
            std::slice::from_ref(&written),
            |_, _, _| Ok(0),
        )
        .unwrap();
    assert_eq!(causes(&copies), [(true, None, Some(CopyCause::BeyondHeld))]);
    sync.after(std::slice::from_ref(&writer), &[written], &[]);
    assert_eq!(named(&sync.finish()), [(s.shadow_key, false)]);
    // Two shadows of one level and slice of one image hold its texels alike; another level's do
    // not, nor another image's.
    assert!(s.same_texels(&image_shadow(image, 2, 1)));
    assert!(!s.same_texels(&other_level));
    assert!(!s.same_texels(&image_shadow(ResourceKey::new(), 2, 1)));
}

/// A part of an image's region is the same texels of both images, offset alike, clamped to the
/// region, and nothing where no texel lies inside.
#[test]
fn a_part_of_an_images_region_moves_both_offsets() {
    let layers = vk::ImageSubresourceLayers {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        mip_level: 1,
        base_array_layer: 3,
        layer_count: 1,
    };
    let region = vk::ImageCopy {
        src_subresource: layers,
        src_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
        dst_subresource: layers,
        dst_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
        extent: vk::Extent3D {
            width: 16,
            height: 8,
            depth: 1,
        },
    };
    let part = shadow::image_part_of(&region, [2, 1, 6, 3]).unwrap();
    let at = |o: vk::Offset3D| (o.x, o.y, o.z);
    assert_eq!(
        (at(part.src_offset), at(part.dst_offset)),
        ((2, 1, 0), (2, 1, 0))
    );
    assert_eq!(
        (part.extent.width, part.extent.height, part.extent.depth),
        (4, 2, 1)
    );
    assert_eq!(
        (
            part.src_subresource.mip_level,
            part.dst_subresource.base_array_layer
        ),
        (1, 3)
    );
    let clamped = shadow::image_part_of(&region, [12, 6, 40, 40]).unwrap();
    assert_eq!((clamped.extent.width, clamped.extent.height), (4, 2));
    assert!(shadow::image_part_of(&region, [16, 0, 20, 8]).is_none());
    assert!(shadow::image_part_of(&region, [3, 3, 3, 5]).is_none());
}

/// The encoder's scratch keeps the rectangle its commands write in, where they bound it and the
/// copies are cut to rectangles: a later command that writes inside it
/// keeps nothing more; one that writes outside it takes the merge of what was kept, and a scratch
/// of the rectangle bounding both, kept again after its copy in; one that writes anywhere, a
/// scratch of every texel, which then holds every later write.
#[test]
fn a_keep_holds_the_rectangle_its_commands_write_in() {
    let buffer = ResourceKey::new();
    let a = cut_shadow(buffer, 0..8192);
    assert!(a.merges_back());
    let written = through(&a, true).1;
    let mut made = 0;
    let mut run = |sync: &mut ShadowSync<Boxed>, area: Option<[u32; 4]>| {
        let writer = ShadowUse {
            writes: true,
            area,
            loads: area,
            tile_stores: false,
            ..through(&a, true).0
        };
        let copies = sync
            .before(
                std::slice::from_ref(&writer),
                std::slice::from_ref(&written),
                |_, _, kept| {
                    made += 1;
                    Ok(Boxed(made, kept))
                },
            )
            .unwrap();
        sync.after(
            std::slice::from_ref(&writer),
            std::slice::from_ref(&written),
            &[],
        );
        kept_steps(&copies)
    };
    let mut sync = ShadowSync::default();
    assert_eq!(
        run(&mut sync, Some([8, 4, 24, 12])),
        ["into Some([8, 4, 24, 12])", "keep 1 Some([8, 4, 24, 12])"]
    );
    assert!(run(&mut sync, Some([10, 5, 20, 10])).is_empty(), "inside");
    assert_eq!(
        run(&mut sync, Some([30, 0, 40, 8])),
        [
            "merge 1 Some([8, 4, 24, 12])",
            "into Some([30, 0, 40, 8])",
            "keep 2 Some([8, 0, 40, 12])"
        ],
        "outside"
    );
    assert_eq!(
        run(&mut sync, None),
        ["merge 2 Some([8, 0, 40, 12])", "into None", "keep 3 None"],
        "anywhere"
    );
    assert!(run(&mut sync, Some([0, 0, 4, 4])).is_empty(), "held");
    assert_eq!(kept_steps(&sync.finish()), ["merge 3 None"]);
}

/// Each copy's step, with its area, or its scratch's number and rectangle.
fn kept_steps(copies: &[ShadowCopy<Boxed>]) -> Vec<String> {
    copies
        .iter()
        .map(|copy| match &copy.step {
            Step::Into => format!("into {:?}", copy.area),
            Step::Back => format!("back {:?}", copy.area),
            Step::Keep(Boxed(n, area)) => format!("keep {n} {area:?}"),
            Step::Merge(Boxed(n, area)) => format!("merge {n} {area:?}"),
        })
        .collect()
}

/// A copy of a 2D texture's image takes the regions of its own direction
/// ([`ShadowTarget::image_regions`]): the copy-in set going in, the copy-back set going out. They
/// are one set for every shadow but one over an IOSurface plane whose rows share bytes, where
/// going in reads each shared byte for both its rows and going back writes it once, from the
/// earlier row — so a copy back that took the copy-in set would write the shared bytes twice in
/// one `vkCmdCopyImageToBuffer` (`VUID-vkCmdCopyImageToBuffer-pRegions-12485`).
#[test]
fn a_copy_of_an_image_takes_the_regions_of_its_direction() {
    // The two sets a 5x4 RGBA8Unorm plane 16 bytes a row takes
    // (`buffer_texture::plane_copy_regions`): the head of every row and the rest of every row
    // going in, and going back the rest of every row and the head of row 0 alone.
    let region = |at: u64, x: i32, width: u32, height: u32| vk::BufferImageCopy {
        buffer_offset: at,
        buffer_row_length: 4,
        buffer_image_height: 0,
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        image_offset: vk::Offset3D { x, y: 0, z: 0 },
        image_extent: vk::Extent3D {
            width,
            height,
            depth: 1,
        },
    };
    let into = vec![region(0, 0, 1, 4), region(4, 1, 4, 4)];
    let back = vec![region(0, 0, 1, 1), region(4, 1, 4, 4)];
    let shape = |regions: &[vk::BufferImageCopy]| {
        (regions.iter())
            .map(|r| {
                let e = r.image_extent;
                (r.buffer_offset, r.image_offset.x, e.width, e.height)
            })
            .collect::<Vec<_>>()
    };
    let target = ShadowTarget::Image {
        image: vk::Image::null(),
        regions: into.clone(),
        back: back.clone(),
        staging: None,
        texel: Some(4),
        words: None,
        unorm: None,
    };
    assert_eq!(
        shape(target.image_regions(true).expect("an image's regions")),
        shape(&into)
    );
    assert_eq!(
        shape(target.image_regions(false).expect("an image's regions")),
        shape(&back)
    );
    // The two sets differ, so taking the wrong one is visible: the copy-in set reads the shared
    // head of all four rows, the copy-back set writes row 0's alone.
    assert_ne!(shape(&into), shape(&back));
}

/// The texels of a 3840x2160 surface.
const SURFACE_TEXELS: u64 = 3840 * 2160;

/// A shadowed texture over a 4K IOSurface plane's bytes in `plane`: one region of 3840x2160
/// texels of 4 bytes, its rows 4096 texels apart (a padded plane, which no linear image of the
/// device aliases, so its texture is shadowed). Its copies are cut to rectangles
/// (`ShadowCopies::cuts`) and it merges back, as a surface's does.
fn surface_shadow(plane: ResourceKey) -> Arc<ShadowCopies> {
    let bytes = 0..4 * 4096 * SURFACE_TEXELS / 3840;
    let mut copies = Arc::into_inner(shadow_in(
        ShadowBuffer::Fixed(vk::Buffer::null(), 0),
        plane,
        None,
        bytes,
    ))
    .expect("one holder");
    copies.backing = Backing::Surface;
    if let ShadowSource::Bytes {
        target: ShadowTarget::Image { regions, .. },
        ..
    } = &mut copies.source
    {
        regions.push(vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 4096,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D::default(),
            image_extent: vk::Extent3D {
                width: 3840,
                height: 2160,
                depth: 1,
            },
        });
    }
    assert!(copies.cuts());
    Arc::new(copies)
}

/// The texels the copies in among `copies` move (`WorkCounts::shadow_texels_copied_in`).
fn texels_in(copies: &[ShadowCopy<u32>]) -> u64 {
    copies.iter().map(ShadowCopy::texels_copied_in).sum()
}

/// One command buffer's encoders in turn, each carrying what the one before left shadowed
/// textures' images holding (`ShadowCarry`), and the texels their copies in moved. A fresh one is
/// another command buffer, which carries nothing.
struct Frame {
    /// `None` while an encoder of it holds what it carries.
    carry: Option<ShadowCarry>,
    made: u32,
    texels: u64,
    /// How its encoders' first copies in stood against their images' last standing: the texels
    /// the carry took off, those of them of large planes, and the large planes' texels a first
    /// copy in moved behind an earlier command buffer's standing or behind none.
    tally: (u64, u64, u64, u64),
}

impl Frame {
    fn new() -> Frame {
        Frame {
            carry: Some(ShadowCarry::default()),
            made: 0,
            texels: 0,
            tally: (0, 0, 0, 0),
        }
    }

    /// Open an encoder on what the command buffer carries, which it holds until it ends.
    fn encoder(&mut self) -> ShadowSync<u32> {
        ShadowSync::carrying(self.carry.take().unwrap_or_default())
    }

    /// End `sync`, its copies back counted and what it leaves carried taken back.
    fn end(&mut self, mut sync: ShadowSync<u32>) {
        self.texels += texels_in(&sync.finish());
        self.take_tally(&mut sync);
        self.carry = Some(sync.ended());
    }

    /// Add how `sync`'s first copies in so far stood: the large planes' texels the carry took
    /// off, those a first copy in moved behind an earlier command buffer's standing, and those
    /// it moved behind none.
    fn take_tally(&mut self, sync: &mut ShadowSync<u32>) {
        let tally = sync.tally();
        self.tally.0 += tally.carried;
        self.tally.1 += tally.carried_large;
        self.tally.2 += tally.stood_earlier_large;
        self.tally.3 += tally.unstood_large;
    }

    /// One encoder that samples `texture` in each of `commands` commands. A binding bounds
    /// nothing it reads (`ShadowUse::loads` is `None` for every use but a pass's attachment), so
    /// a copy in of a sampled texture moves every texel.
    fn sampling(&mut self, texture: &Arc<ShadowCopies>, commands: usize) -> &mut Frame {
        let mut sync = self.encoder();
        for _ in 0..commands {
            let copies = planned(&mut sync, &mut self.made, &[(texture, false)], &[]);
            self.texels += texels_in(&copies);
        }
        self.end(sync);
        self
    }

    /// The texels its encoders' first copies in stood behind: carried in all, carried of large
    /// planes, and large planes' behind an earlier command buffer's standing or behind none.
    fn stood(&self) -> (u64, u64, u64, u64) {
        self.tally
    }

    /// One encoder whose render pass renders a color attachment of `texture` that it loads, its
    /// draws inside `area`: such a use bounds both what it reads and what it writes
    /// (`ShadowUse::loads`, `ShadowUse::area`), so its copy in moves only those texels.
    fn rendering(&mut self, texture: &Arc<ShadowCopies>, area: [u32; 4]) -> &mut Frame {
        let mut sync = self.encoder();
        let shadows = [ShadowUse {
            area: Some(area),
            loads: Some(area),
            tile_stores: false,
            ..through(texture, true).0
        }];
        let uses = [through(texture, true).1];
        let made = &mut self.made;
        let copies = sync
            .before(&shadows, &uses, |_, _, _| {
                *made += 1;
                Ok(*made)
            })
            .unwrap();
        sync.after(&shadows, &uses, &[]);
        self.texels += texels_in(&copies);
        self.end(sync);
        self
    }

    /// One encoder that writes `bytes` of the buffer `buffer` and meets no shadowed texture: a
    /// blit's fill of an IOSurface's plane. Its command's uses reach the encoder whatever it
    /// meets (`EncoderCore::track`).
    fn writing(&mut self, buffer: ResourceKey, written: Range<u64>) -> &mut Frame {
        let mut sync = self.encoder();
        sync.wrote(&[bytes(buffer, written, true)]);
        self.end(sync);
        self
    }

    /// The command buffer waits for an event or a binary semaphore
    /// (`CommandBuffer::encode_wait_for_event`).
    fn waits(&mut self) -> &mut Frame {
        if let Some(carry) = &mut self.carry {
            carry.clear();
        }
        self
    }
}

/// A large surface's shadowed texture is copied in whole once for a command buffer that samples
/// it, however many of its encoders do: Metal shows every encoder of a command buffer the bytes
/// as of its commit (`shadow.rs`'s copy-in rule), so an encoder takes the image its command
/// buffer's earlier encoder left holding the truth (`ShadowCarry`). A second command buffer that
/// samples the same plane pays it again, whether through the same texture or another over the
/// same bytes: Metal shows a later command buffer a CPU write made since the commit.
///
/// The counts are this state machine's, taken by this test; they are no measurement of any
/// program.
#[test]
fn a_large_surface_is_copied_in_whole_once_for_a_command_buffer_that_samples_it() {
    let plane = ResourceKey::new();
    let s = surface_shadow(plane);
    assert_eq!(s.texels(), SURFACE_TEXELS);
    assert_eq!(s.backing, Backing::Surface);
    assert!(s.merges_back(), "a surface's bytes have another writer");
    // One encoder: its first command's copy in is whole, and every later command of it takes
    // none, the image holding the plane as the encoder's commands left it.
    let mut frame = Frame::new();
    frame.sampling(&s, 1);
    assert_eq!(frame.texels, SURFACE_TEXELS);
    let mut frame = Frame::new();
    frame.sampling(&s, 4);
    assert_eq!(frame.texels, SURFACE_TEXELS);
    // Three encoders of one command buffer, each sampling the plane: one whole copy in.
    let mut frame = Frame::new();
    frame.sampling(&s, 1).sampling(&s, 1).sampling(&s, 1);
    assert_eq!(frame.texels, SURFACE_TEXELS);
    // A second command buffer that samples it, through the same texture or through another over
    // the same bytes: one whole copy in each.
    let mut next = Frame::new();
    next.sampling(&s, 1);
    assert_eq!(next.texels, SURFACE_TEXELS);
    let other = surface_shadow(plane);
    assert_ne!(other.shadow_key, s.shadow_key);
    let mut next = Frame::new();
    next.sampling(&other, 1);
    assert_eq!(next.texels, SURFACE_TEXELS);
}

/// A command buffer carries a shadowed texture's image on only while nothing lets another writer
/// of its truth in: a command of any of its encoders that writes the bytes another way, an
/// encoder of another command buffer that ended with the texture in between, a wait for an event
/// or a fence, work the encoder records apart, and an encoder that did not end holding the truth
/// each leave the next encoder to copy the plane in again.
#[test]
fn a_carried_surface_is_dropped_by_anything_that_lets_another_writer_in() {
    let plane = ResourceKey::new();
    let s = surface_shadow(plane);
    let twice = 2 * SURFACE_TEXELS;
    // Nothing in between: carried.
    let mut frame = Frame::new();
    frame.sampling(&s, 1).sampling(&s, 1);
    assert_eq!(frame.texels, SURFACE_TEXELS);
    // A command of an encoder that meets no shadowed texture writes the plane's bytes.
    let mut frame = Frame::new();
    frame
        .sampling(&s, 1)
        .writing(plane, 0..4096)
        .sampling(&s, 1);
    assert_eq!(frame.texels, twice, "the bytes written another way");
    // A write of other bytes of the same buffer that the texture does not lie in leaves it
    // carried; the plane's are 0..33 177 600.
    let mut frame = Frame::new();
    frame
        .sampling(&s, 1)
        .writing(plane, 1 << 26..1 << 27)
        .sampling(&s, 1);
    assert_eq!(frame.texels, SURFACE_TEXELS, "other bytes");
    // An encoder of another command buffer ended with the texture between this command buffer's
    // two: what this one carries is no longer what the image holds.
    let mut frame = Frame::new();
    frame.sampling(&s, 1);
    let mut other = Frame::new();
    other.sampling(&s, 1);
    frame.sampling(&s, 1);
    assert_eq!(
        frame.texels, twice,
        "another command buffer's encoder between"
    );
    // One that ended with the texture while this command buffer's encoder was still open: its
    // own end carries nothing either, as the image is no longer the one it left.
    let mut frame = Frame::new();
    let mut sync = frame.encoder();
    let mut made = 0;
    frame.texels += texels_in(&planned(&mut sync, &mut made, &[(&s, false)], &[]));
    let mut other = Frame::new();
    other.sampling(&s, 1);
    frame.end(sync);
    frame.sampling(&s, 1);
    assert_eq!(
        frame.texels, twice,
        "another command buffer's encoder while open"
    );
    // The command buffer waits for an event or a binary semaphore; an encoder's wait for a
    // fence is `ShadowSync::not_carried`, below.
    let mut frame = Frame::new();
    frame.sampling(&s, 1).waits().sampling(&s, 1);
    assert_eq!(frame.texels, twice, "an event wait");
    // An encoder's wait for a fence: its own commands after the wait copy the plane in again,
    // the truth now holding what the update's writer left; the image holds it again from there,
    // so the next encoder of the command buffer carries it.
    let mut frame = Frame::new();
    let mut sync = frame.encoder();
    let mut made = 0;
    frame.texels += texels_in(&planned(&mut sync, &mut made, &[(&s, false)], &[]));
    sync.not_carried();
    frame.texels += texels_in(&planned(&mut sync, &mut made, &[(&s, false)], &[]));
    frame.end(sync);
    frame.sampling(&s, 1);
    assert_eq!(frame.texels, twice, "a fence wait");
    // Work the encoder records apart stands between its commands and the next encoder's, and may
    // write any truth: it leaves the encoder's own textures to copy in again.
    let mut frame = Frame::new();
    let mut sync = frame.encoder();
    let mut made = 0;
    frame.texels += texels_in(&planned(&mut sync, &mut made, &[(&s, false)], &[]));
    frame.texels += texels_in(&sync.leave());
    frame.texels += texels_in(&planned(&mut sync, &mut made, &[(&s, false)], &[]));
    frame.end(sync);
    assert_eq!(
        frame.texels, twice,
        "work recorded apart, the encoder's own"
    );
    // And what the command buffer carried into an encoder that records work apart without ever
    // meeting the texture.
    let mut frame = Frame::new();
    frame.sampling(&s, 1);
    let mut sync = frame.encoder();
    frame.texels += texels_in(&sync.leave());
    frame.end(sync);
    frame.sampling(&s, 1);
    assert_eq!(frame.texels, twice, "work recorded apart, carried");
    // An encoder that did not end holding the truth: one that wrote through a texture whose
    // writes Metal drops.
    let mut dropped = Arc::into_inner(surface_shadow(plane)).expect("one holder");
    dropped.drops_writes = true;
    let dropped = Arc::new(dropped);
    let mut frame = Frame::new();
    let mut sync = frame.encoder();
    let mut made = 0;
    frame.texels += texels_in(&planned(&mut sync, &mut made, &[(&dropped, true)], &[]));
    frame.end(sync);
    frame.sampling(&dropped, 1);
    assert_eq!(frame.texels, twice, "writes Metal drops");
}

/// A render pass that renders a large surface's texture as a color attachment it loads pays only
/// the texels inside its draws' scissors, where it bounds them, and the image then holds the
/// plane inside that rectangle alone: anything that samples the plane afterwards pays every texel
/// of it, bounded by nothing, in that command buffer or a later one. So a present that draws the
/// plane into a window costs a whole plane a frame however small the damage the guest's own pass
/// repainted, and no carry covers it.
///
/// The counts are this state machine's, taken by this test; they are no measurement of any
/// program.
#[test]
fn sampling_a_large_surface_costs_every_texel_whatever_its_pass_repainted() {
    let plane = ResourceKey::new();
    let s = surface_shadow(plane);
    let damage = [0, 0, 256, 128];
    let repainted = 256 * 128;
    let mut frame = Frame::new();
    frame.rendering(&s, damage);
    assert_eq!(frame.texels, repainted);
    // The same command buffer: the image holds the damage rectangle alone, so a sample of the
    // plane takes every texel beyond it.
    let mut frame = Frame::new();
    frame.rendering(&s, damage).sampling(&s, 1);
    assert_eq!(frame.texels, repainted + SURFACE_TEXELS);
    // A command buffer of its own, as a window's present is.
    let mut frame = Frame::new();
    frame.rendering(&s, damage);
    assert_eq!(frame.texels, repainted);
    let mut present = Frame::new();
    present.sampling(&s, 1);
    assert_eq!(present.texels, SURFACE_TEXELS);
}

/// An encoder's first copy in of a large IOSurface plane falls in one of three places, which
/// divide them and say what carrying the image across a command buffer's encoders is worth: the
/// carry took it off, an earlier command buffer's standing stands behind it and Metal requires it
/// (the module's copy-in rule), or the image never held a standing and no rule avoids it.
/// A standing of the command buffer's own that something let a writer past falls in none of the
/// three: `WorkCounts::shadow_texels_first_copied_in_again` counts those.
///
/// The counts are this state machine's, taken by this test; they are no measurement of any
/// program.
#[test]
fn a_large_surfaces_first_copies_in_divide_into_carried_stood_earlier_and_unstood() {
    let plane = ResourceKey::new();
    let s = surface_shadow(plane);
    assert!(s.large_surface());
    let one = SURFACE_TEXELS;
    // The first command buffer: its first encoder finds no standing anywhere, its second takes
    // the image the first left.
    let mut first = Frame::new();
    first.sampling(&s, 1).sampling(&s, 1);
    assert_eq!(
        first.stood(),
        (one, one, 0, one),
        "the first command buffer"
    );
    assert_eq!(first.texels, one);
    // A second command buffer: the image's last standing is the first's, which Metal requires it
    // to copy in again; its own second encoder then carries.
    let mut next = Frame::new();
    next.sampling(&s, 1).sampling(&s, 1);
    assert_eq!(next.stood(), (one, one, one, 0), "a second command buffer");
    assert_eq!(next.texels, one);
    // Three command buffers of one encoder each, as a compositor's frame is: every copy in is
    // the first of its own command buffer, and the carry takes nothing off.
    let frames: Vec<(u64, u64, u64, u64)> = (0..3)
        .map(|_| {
            let mut frame = Frame::new();
            frame.sampling(&s, 1);
            frame.stood()
        })
        .collect();
    assert_eq!(
        frames,
        [(0, 0, one, 0); 3],
        "one encoder per command buffer"
    );
    // A standing of the command buffer's own that a write of the bytes another way let a writer
    // past: in none of the three.
    let mut frame = Frame::new();
    frame
        .sampling(&s, 1)
        .writing(plane, 0..4096)
        .sampling(&s, 1);
    assert_eq!(frame.stood(), (0, 0, one, 0), "the standing was let past");
    assert_eq!(frame.texels, 2 * one);
    // A texture no command buffer ever stood behind is unstood, however many stood behind others.
    let other = surface_shadow(plane);
    let mut frame = Frame::new();
    frame.sampling(&other, 1);
    assert_eq!(frame.stood(), (0, 0, 0, one), "a texture of its own");
    // A shadow of no IOSurface plane, or of one under 2^20 texels, moves none of the three: it
    // is carried as any other is, and counted among the carried texels alone.
    let small = cut_shadow(ResourceKey::new(), 0..8192);
    assert!(!small.large_surface());
    let mut frame = Frame::new();
    frame.sampling(&small, 1).sampling(&small, 1);
    assert_eq!(frame.stood(), (64 * 32, 0, 0, 0), "a small mapped buffer's");
}

/// The scratch kinds a command's merges are made for (`MergeFor::blocks`) and the rectangles they
/// keep, in a fresh encoder, for a command that uses the shadowed textures `shadows` and has
/// `uses`.
fn made_for(shadows: &[ShadowUse], uses: &[Use<ResourceKey>]) -> Vec<(bool, Option<[u32; 4]>)> {
    let mut sync = ShadowSync::<u32>::default();
    let mut made = Vec::new();
    sync.before(shadows, uses, |_, merge_for, area| {
        made.push((merge_for.blocks, area));
        Ok(0)
    })
    .unwrap();
    made
}

/// A render pass's color it loads and stores inside its draws' scissors
/// (`ShadowUse::tile_stores`) is kept for a merge that stores whole blocks of that rectangle's
/// rows, as Metal's own store writes back every texel of each tile its draws reach; a pass whose
/// writes no rectangle bounds, and one whose texture a draw also writes through, keep one that
/// stores the changed words alone.
#[test]
fn a_pass_storing_its_loaded_area_merges_in_whole_blocks() {
    let buffer = ResourceKey::new();
    let t = cut_shadow(buffer, 256..256 + 320 * 32);
    assert!(t.merges_back() && t.cuts());
    let area = [8, 4, 40, 20];
    let pass = |area: Option<[u32; 4]>, tile_stores: bool| ShadowUse {
        area,
        loads: area,
        tile_stores,
        ..through(&t, true).0
    };
    let used = [through(&t, true).1];
    assert_eq!(
        made_for(&[pass(Some(area), true)], &used),
        [(true, Some(area))]
    );
    // Not by tile stores alone, or bounded by no rectangle: the changed words alone.
    assert_eq!(
        made_for(&[pass(Some(area), false)], &used),
        [(false, Some(area))]
    );
    assert_eq!(made_for(&[pass(None, true)], &used), [(false, None)]);
    // A draw's write through the texture joins the pass's use: the changed words alone.
    let mut joined = Vec::new();
    shadow::join(&mut joined, &pass(Some(area), true));
    shadow::join(&mut joined, &through(&t, true).0);
    assert!(!joined[0].tile_stores && joined[0].area.is_none());
    assert_eq!(made_for(&joined, &used), [(false, None)]);
    // A draw's read of it leaves the pass's tile stores.
    let mut joined = Vec::new();
    shadow::join(&mut joined, &pass(Some(area), true));
    shadow::join(&mut joined, &through(&t, false).0);
    assert!(joined[0].tile_stores);
    assert_eq!(made_for(&joined, &used), [(true, Some(area))]);
}
