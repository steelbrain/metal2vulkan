//! `Buffer::texture_placement` and `texture_placement_with_iosurface_plane` answer the route a
//! texture made of the same arguments then takes (`texture/route.rs`), read from the device's
//! counts across the making (`WorkCounts::host_textures_in_place`, `device_copy_textures`,
//! `host_texture_shadows_*`) and the texture's storage:
//!
//! - **Pending, then known.** On a device whose shader cache holds no in-place record, the first
//!   question over a no-copy buffer answers `Pending` (nothing started the check), and the texture
//!   made then takes the answer of a question asked just before or just after it (the check runs
//!   on its own thread meanwhile). Once `Device::settle_in_place_checks` answers `true`, the same
//!   question answers in place exactly where the check passed for the pages' kind, the texture's
//!   access and its row ends, and a texture made then takes that answer; the texture made before
//!   moves to it by the end of the next quiet command buffer.
//! - **Each other route** the device reaches: a store width (an `R8Unorm` texture a shader
//!   writes), a render target, an IOSurface plane, a buffer of fresh pages where the device puts
//!   a long buffer in them, texels past the buffer's length, a buffer of no host pages, a texture
//!   buffer (another shape), detached bytes, `HostPageTexturePolicy::Shadow`, and a check that
//!   failed (`DeviceOptions::fail_in_place_check`). A discrete GPU's device copy and a copied
//!   buffer's bytes are not reached here.
//! - **Refusals** are the making's own: a pitch below the row, texels past the allocation, and an
//!   IOSurface plane's client direction give the same error to both.
//! - **Pitches the device lays no image at.** Once the check settled, `BGRA8Unorm` textures over a
//!   no-copy buffer at pitches past their rows (a 61-texel row of 244 bytes at 256, 272, 320, 512
//!   and 1024 bytes, `ShaderRead`; the consumer's 3664-texel surface of 14656 bytes a row at 14656,
//!   14672 and 14848, `ShaderRead | RenderTarget`) each answer what the making takes, the image's
//!   making up to its bind included: where the device lays no image at the pitch (its linear image
//!   laid out at another, and no `DRM_FORMAT_MOD_LINEAR` image at it), `Shadow(Refused(Pitch))`,
//!   counted in `host_texture_shadows_refused_pitch`. At least one is: 272 bytes, 16 past a
//!   256-byte row, is a pitch no driver measured lays rows at (llvmpipe's linear images step by
//!   64 bytes, MoltenVK lays them at their own pitch alone, RADV's modifier images step by 256).
//!   Each answer, asked before any texture is made, costs one or two `vkCreateImage` calls where
//!   it is `InPlace` (more where refused, a pair per wider plan the making tries), no
//!   `vkBindImageMemory` and no submission (`ObjectChurn`, `WorkCounts::queue_submissions`).
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. Printed (`TEXTURE-PLACEMENT`): each
//! case's answer and what the making took.

use super::blit_holds::{device, over_host, shadow_device, Host, Scratch};
use crate::in_place::{Access, PagesClass, RowEnds, Verdict};
use crate::{
    BlitOption, Buffer, Device, DeviceKind, DeviceOptions, Error, InPlaceAccess, InPlaceRefusal,
    InPlaceRowEnds, IoSurfacePlane, Origin, ResourceOptions, ShaderCacheOptions, Size, Texture,
    TextureDescriptor, TexturePlacement, TextureShadowReason, WorkCounts,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::path::Path;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::Duration;

/// The longest the device's check may take here.
const SETTLE: Duration = Duration::from_secs(120);

/// The longest a command buffer may take here.
const WAIT: Duration = Duration::from_secs(20);

/// What a texture's making took, as the device's counts across it name it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Took {
    InPlace,
    Pending,
    Policy,
    StoreWidth,
    CheckFailed,
    Unchecked,
    Refused(InPlaceRefusal),
    DeviceCopy,
    /// No count over host pages moved: texels past the buffer's length, a buffer of no host
    /// pages, or another shape.
    Uncounted,
}

/// What a making that `placement` answers for takes, as the counts name it.
fn expected(placement: TexturePlacement) -> Took {
    use TextureShadowReason as R;
    match placement {
        TexturePlacement::InPlace => Took::InPlace,
        TexturePlacement::Pending => Took::Pending,
        TexturePlacement::Shadow(R::Policy) => Took::Policy,
        TexturePlacement::Shadow(R::StoreWidth { .. }) => Took::StoreWidth,
        TexturePlacement::Shadow(R::CheckFailed { .. }) => Took::CheckFailed,
        TexturePlacement::Shadow(R::Unchecked { .. }) => Took::Unchecked,
        TexturePlacement::Shadow(R::Refused(refusal)) => Took::Refused(refusal),
        TexturePlacement::Shadow(R::DeviceCopy) => Took::DeviceCopy,
        TexturePlacement::Shadow(R::PastLength | R::Mipmapped { .. } | R::BytesBetweenRows)
        | TexturePlacement::NotHostPages
        | TexturePlacement::OtherShape => Took::Uncounted,
    }
}

/// What the making between `before` and `after` took: the one count over host pages that moved.
fn took(before: &WorkCounts, after: &WorkCounts) -> Took {
    let delta = |of: fn(&WorkCounts) -> u64| of(after) - of(before);
    let counts: [(Took, fn(&WorkCounts) -> u64); 14] = [
        (Took::InPlace, |c| c.host_textures_in_place),
        (Took::Pending, |c| c.host_texture_shadows_check_pending),
        (Took::Policy, |c| c.host_texture_shadows_exact),
        (Took::StoreWidth, |c| c.host_texture_shadows_store_width),
        (Took::CheckFailed, |c| c.host_texture_shadows_check_failed),
        (Took::Unchecked, |c| c.host_texture_shadows_unchecked),
        (Took::DeviceCopy, |c| c.device_copy_textures),
        (Took::Refused(InPlaceRefusal::Rows), |c| {
            c.host_texture_shadows_refused_rows
        }),
        (Took::Refused(InPlaceRefusal::Detached), |c| {
            c.host_texture_shadows_refused_detached
        }),
        (Took::Refused(InPlaceRefusal::Image), |c| {
            c.host_texture_shadows_refused_image
        }),
        (Took::Refused(InPlaceRefusal::Pitch), |c| {
            c.host_texture_shadows_refused_pitch
        }),
        (Took::Refused(InPlaceRefusal::Placement), |c| {
            c.host_texture_shadows_refused_placement
        }),
        (Took::Refused(InPlaceRefusal::Source), |c| {
            c.host_texture_shadows_refused_source
        }),
        (Took::Refused(InPlaceRefusal::DmaBuf), |c| {
            c.host_texture_shadows_refused_dma_buf
        }),
    ];
    let moved: Vec<Took> = counts
        .iter()
        .filter(|(_, of)| delta(*of) != 0)
        .map(|&(took, _)| took)
        .collect();
    match moved[..] {
        [] => Took::Uncounted,
        [one] => one,
        _ => panic!("one making moved several counts: {moved:?}"),
    }
}

/// A texture's path, as its storage shows it.
fn path(texture: &Texture) -> &'static str {
    match (texture.shadow().is_some(), texture.aliases_buffer()) {
        (true, _) => "shadow",
        (false, true) => "aliasing image",
        (false, false) => "no image over the buffer",
    }
}

/// Ask `ask`, make with `make`, and ask again: both answers, what the making took, and the
/// texture. Printed as `case`.
fn asked_and_made(
    device: &Device,
    case: &str,
    ask: impl Fn() -> Result<TexturePlacement, Error>,
    make: impl FnOnce() -> Result<Texture, Error>,
) -> ([TexturePlacement; 2], Took, Texture) {
    let first = ask().unwrap();
    let before = device.work_counts();
    let texture = make().unwrap();
    let after = device.work_counts();
    let then = ask().unwrap();
    let took = took(&before, &after);
    eprintln!(
        "TEXTURE-PLACEMENT on {}: {case}: asked {first:?}, then {then:?}; made {took:?}, {}",
        device.name(),
        path(&texture)
    );
    ([first, then], took, texture)
}

/// As [`asked_and_made`], where no verdict can change between the two questions: both answer
/// alike, and the making took that answer.
fn made_as_asked(
    device: &Device,
    case: &str,
    ask: impl Fn() -> Result<TexturePlacement, Error>,
    make: impl FnOnce() -> Result<Texture, Error>,
) -> (TexturePlacement, Texture) {
    let ([first, then], took, texture) = asked_and_made(device, case, ask, make);
    assert_eq!(
        first, then,
        "{case}: the answer moved with nothing to move it"
    );
    assert_eq!(
        took,
        expected(first),
        "{case}: the making took another route"
    );
    (first, texture)
}

/// A `width` by `height` 2D texture of `format` and `usage` over a buffer of `options`.
fn descriptor(
    format: PixelFormat,
    (width, height): (u64, u64),
    usage: TextureUsage,
    options: ResourceOptions,
) -> TextureDescriptor {
    let mut descriptor = TextureDescriptor::texture_2d(format, width, height, false);
    descriptor.usage = usage;
    descriptor.resource_options = options;
    descriptor
}

/// Whether the device's check gave `verdict` for `access` over pages imported by their pointer,
/// for texels laid out as a `rows`-row texture of `row` bytes a row, packed, from `buffer`'s
/// first byte (`in_place::RowEnds`); and those row ends.
fn verdict_of(
    device: &Device,
    buffer: &Buffer,
    access: Access,
    (row, rows): (u64, u64),
) -> (Verdict, RowEnds) {
    let layout = crate::linear_plan::BufferLayout {
        offset: 0,
        row_pitch: row,
        slice_pitch: row * rows,
        row_bytes: row,
        rows,
        slices: 1,
    };
    let start = buffer.contents().map_or(0, |at| at.as_ptr().addr() as u64);
    let checks = &device.shared().in_place;
    let ends = RowEnds::of(start, &layout, checks.line());
    (checks.verdict(PagesClass::Pointer, access, ends), ends)
}

/// The public name of `rows`.
fn public_rows(rows: RowEnds) -> InPlaceRowEnds {
    match rows {
        RowEnds::InLines => InPlaceRowEnds::InLines,
        RowEnds::OnLines => InPlaceRowEnds::OnLines,
    }
}

/// The options of a device the cfg selects, its shader cache in `cache`, whose in-place check
/// fails whatever its moves did (`DeviceOptions::fail_in_place_check`).
fn failing_device(cache: &Path) -> Device {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        fail_in_place_check: true,
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

#[test]
fn the_answer_is_the_route_the_texture_takes() {
    // The pages, declared before the devices that use them, so dropped after them.
    let hosts: Vec<Host> = (0..4).map(|_| Host::new()).collect();
    let scratch = Scratch::new("texture-placement");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let runs = Arc::new(AtomicUsize::new(0));
    let Some(buffer) = over_host(&device, &hosts[0], &runs) else {
        return;
    };
    let options = buffer.resource_options();
    let side = 64u64;
    let read = descriptor(
        PixelFormat::BGRA8Unorm,
        (side, side),
        TextureUsage::SHADER_READ,
        options,
    );

    // Refused alike, before anything asks a verdict.
    let narrow = buffer.texture_placement(&read, 0, 16);
    assert!(narrow.is_err(), "a pitch below the row: {narrow:?}");
    assert_eq!(narrow.err(), buffer.new_texture(&read, 0, 16).err());
    let past = buffer.texture_placement(&read, 4 << 20, side * 4);
    assert!(past.is_err(), "texels past the allocation: {past:?}");
    assert_eq!(
        past.err(),
        buffer.new_texture(&read, 4 << 20, side * 4).err()
    );
    let directed = IoSurfacePlane::new(0, side * 4, false).with_client_direction(1);
    let refused = buffer.texture_placement_with_iosurface_plane(&read, directed);
    assert!(refused.is_err(), "a client direction: {refused:?}");
    let made = buffer.new_texture_with_iosurface_plane(&read, directed);
    assert_eq!(refused.err(), made.err());

    // The first question: nothing started the check, so it is pending; it starts the check, and
    // the texture made next takes the answer of the question before it or after it.
    let ask = || buffer.texture_placement(&read, 0, side * 4);
    let make = || buffer.new_texture(&read, 0, side * 4);
    let ([first, then], early_took, early) =
        asked_and_made(&device, "before the verdict", ask, make);
    assert_eq!(first, TexturePlacement::Pending, "the first question");
    assert!(
        [expected(first), expected(then)].contains(&early_took),
        "made before the verdict: {early_took:?}, between {first:?} and {then:?}"
    );

    // Once the verdicts are known, the answer is in place exactly where the check passed.
    assert!(device.settle_in_place_checks(SETTLE), "the check settled");
    assert!(
        device.settle_in_place_checks(Duration::ZERO),
        "settled for good"
    );
    let (verdict, rows) = verdict_of(&device, &buffer, Access::Reads, (side * 4, side));
    let settled = ask().unwrap();
    let due = match verdict {
        Verdict::Passed => TexturePlacement::InPlace,
        Verdict::Failed => TexturePlacement::Shadow(TextureShadowReason::CheckFailed {
            access: InPlaceAccess::Reads,
            row_ends: public_rows(rows),
        }),
        Verdict::Unchecked => TexturePlacement::Shadow(TextureShadowReason::Unchecked {
            access: InPlaceAccess::Reads,
            row_ends: public_rows(rows),
        }),
        Verdict::Pending => panic!("a pending verdict once settled"),
    };
    assert_eq!(
        settled, due,
        "once settled, for a check that answered {verdict:?}"
    );
    let ask = || buffer.texture_placement(&read, side * side * 4, side * 4);
    let make = || buffer.new_texture(&read, side * side * 4, side * 4);
    let (late, _) = made_as_asked(&device, "once settled", ask, make);

    // The texture made before moves to the answer by the end of the next quiet command buffer.
    let out = device
        .new_buffer(side * side * 4, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    (blit.copy_from_texture_to_buffer(
        &early,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(side, side, 1),
        &out,
        0,
        side * 4,
        side * side * 4,
        BlitOption::None,
    ))
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let moved = path(&early);
    eprintln!("TEXTURE-PLACEMENT: the texture made before the verdict is now {moved}");
    let due = match late {
        TexturePlacement::InPlace => "aliasing image",
        _ => "shadow",
    };
    assert_eq!(
        moved, due,
        "the texture made before the verdict, once {late:?}"
    );

    // A texture a shader writes, of texels of a width the check's kernel stores none of.
    let narrow = descriptor(
        PixelFormat::R8Unorm,
        (side * 4, side),
        TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
        options,
    );
    let at = 2 * side * side * 4;
    let ask = || buffer.texture_placement(&narrow, at, side * 4);
    let make = || buffer.new_texture(&narrow, at, side * 4);
    let (answer, _) = made_as_asked(&device, "R8Unorm a shader writes", ask, make);
    let width = TextureShadowReason::StoreWidth { texel_bytes: 1 };
    assert_eq!(answer, TexturePlacement::Shadow(width));

    // A render target, whatever its check answered.
    let renders = descriptor(
        PixelFormat::BGRA8Unorm,
        (side, side),
        TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET,
        options,
    );
    let at = 3 * side * side * 4;
    let ask = || buffer.texture_placement(&renders, at, side * 4);
    let make = || buffer.new_texture(&renders, at, side * 4);
    made_as_asked(&device, "a render target", ask, make);

    // An IOSurface plane over the same pages.
    let plane = IoSurfacePlane::new(4 * side * side * 4, side * 4, false);
    let ask = || buffer.texture_placement_with_iosurface_plane(&renders, plane);
    let make = || buffer.new_texture_with_iosurface_plane(&renders, plane);
    made_as_asked(&device, "an IOSurface plane", ask, make);

    // A texture buffer over the pages: another shape.
    let row = TextureDescriptor::texture_buffer(
        PixelFormat::RGBA8Unorm,
        side,
        options,
        TextureUsage::SHADER_READ,
    );
    let at = 5 * side * side * 4;
    let ask = || buffer.texture_placement(&row, at, side * 4);
    let make = || buffer.new_texture(&row, at, side * 4);
    let (answer, _) = made_as_asked(&device, "a texture buffer", ask, make);
    assert_eq!(answer, TexturePlacement::OtherShape);

    // A buffer of the device's own memory: no host pages.
    let own = device
        .new_buffer(1 << 20, ResourceOptions::shared())
        .unwrap();
    let shared = descriptor(
        PixelFormat::BGRA8Unorm,
        (side, side),
        TextureUsage::SHADER_READ,
        own.resource_options(),
    );
    let ask = || own.texture_placement(&shared, 0, side * 4);
    let make = || own.new_texture(&shared, 0, side * 4);
    let (answer, _) = made_as_asked(&device, "a buffer of the device's memory", ask, make);
    assert_eq!(answer, TexturePlacement::NotHostPages);

    // Texels past a short buffer's length: the tail's shadow.
    let short = device.new_buffer(4000, ResourceOptions::shared()).unwrap();
    let tailed = descriptor(
        PixelFormat::BGRA8Unorm,
        (32, 32),
        TextureUsage::SHADER_READ,
        short.resource_options(),
    );
    let ask = || short.texture_placement(&tailed, 0, 128);
    let make = || short.new_texture(&tailed, 0, 128);
    let (answer, texture) = made_as_asked(&device, "texels past the length", ask, make);
    let past = TexturePlacement::Shadow(TextureShadowReason::PastLength);
    assert_eq!((answer, path(&texture)), (past, "shadow"));

    // A long buffer of the runtime's own, in fresh pages where the device puts it there.
    let long = device
        .new_buffer(40 << 20, ResourceOptions::shared())
        .unwrap();
    let fresh = descriptor(
        PixelFormat::BGRA8Unorm,
        (side, side),
        TextureUsage::SHADER_READ,
        long.resource_options(),
    );
    let ask = || long.texture_placement(&fresh, 0, side * 4);
    let make = || long.new_texture(&fresh, 0, side * 4);
    made_as_asked(&device, "a long buffer of the device's", ask, make);

    // Detached bytes.
    let Some(detached) = over_host(&device, &hosts[1], &runs) else {
        return;
    };
    assert_eq!(detached.detach_backing(), Ok(true), "the detach");
    let ask = || detached.texture_placement(&read, 0, side * 4);
    let make = || detached.new_texture(&read, 0, side * 4);
    let (answer, _) = made_as_asked(&device, "detached bytes", ask, make);
    let refused = TextureShadowReason::Refused(InPlaceRefusal::Detached);
    assert_eq!(answer, TexturePlacement::Shadow(refused));

    // The exact form's policy.
    let exact_scratch = Scratch::new("texture-placement-exact");
    let exact = shadow_device(&exact_scratch.0);
    let Some(exact_buffer) = over_host(&exact, &hosts[2], &runs) else {
        return;
    };
    let ask = || exact_buffer.texture_placement(&read, 0, side * 4);
    let make = || exact_buffer.new_texture(&read, 0, side * 4);
    let (answer, _) = made_as_asked(&exact, "the exact form's policy", ask, make);
    assert_eq!(
        answer,
        TexturePlacement::Shadow(TextureShadowReason::Policy)
    );
    assert!(
        !exact.settle_in_place_checks(Duration::ZERO),
        "no check under the policy"
    );

    // A check that failed: a texture that reads and one that renders, each named by its access.
    let failing_scratch = Scratch::new("texture-placement-failing");
    let failing = failing_device(&failing_scratch.0);
    let Some(failing_buffer) = over_host(&failing, &hosts[3], &runs) else {
        return;
    };
    assert!(
        failing.settle_in_place_checks(SETTLE),
        "the failing check settled"
    );
    for (usage, access, at) in [
        (TextureUsage::SHADER_READ, Access::Reads, 0),
        (
            TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET,
            Access::Renders,
            side * side * 4,
        ),
    ] {
        let options = failing_buffer.resource_options();
        let failed = descriptor(PixelFormat::BGRA8Unorm, (side, side), usage, options);
        let ask = || failing_buffer.texture_placement(&failed, at, side * 4);
        let make = || failing_buffer.new_texture(&failed, at, side * 4);
        let case = format!("a failed check, {access:?}");
        let (answer, _) = made_as_asked(&failing, &case, ask, make);
        let (verdict, rows) = verdict_of(&failing, &failing_buffer, access, (side * 4, side));
        let public_access = match access {
            Access::Reads => InPlaceAccess::Reads,
            Access::Renders => InPlaceAccess::Renders,
            Access::Writes
            | Access::BufferWrites
            | Access::NarrowWrites
            | Access::NarrowBufferWrites => InPlaceAccess::Writes,
        };
        let (access, row_ends) = (public_access, public_rows(rows));
        let due = match verdict {
            Verdict::Failed => TextureShadowReason::CheckFailed { access, row_ends },
            Verdict::Unchecked => TextureShadowReason::Unchecked { access, row_ends },
            other => panic!("{case}: the failing check answered {other:?}"),
        };
        assert_eq!(answer, TexturePlacement::Shadow(due), "{case}");
    }
}

#[test]
fn a_pitch_the_device_lays_no_image_at_answers_its_shadow() {
    // The pages, declared before the device that uses them, so dropped after it.
    let host = Host::new();
    let scratch = Scratch::new("texture-placement-pitch");
    let device = device(&scratch.0);
    let runs = Arc::new(AtomicUsize::new(0));
    let Some(buffer) = over_host(&device, &host, &runs) else {
        return;
    };
    assert!(device.settle_in_place_checks(SETTLE), "the check settled");
    let options = buffer.resource_options();
    let reads = TextureUsage::SHADER_READ;
    let renders = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    let cases = [
        ((61, 16), reads, 256),
        ((61, 16), reads, 272),
        ((61, 16), reads, 320),
        ((61, 16), reads, 512),
        ((61, 16), reads, 1024),
        ((3664, 16), renders, 14656),
        ((3664, 16), renders, 14672),
        ((3664, 16), renders, 14848),
    ];
    // Each answer's cost, asked before any texture is made: the images the query made (a call the
    // driver refuses is counted made and has nothing to release), none bound, nothing submitted.
    for (at, &(size, usage, bytes_per_row)) in cases.iter().enumerate() {
        let shaped = descriptor(PixelFormat::BGRA8Unorm, size, usage, options);
        let offset = at as u64 * (256 << 10);
        let case = format!("{}x{} at {bytes_per_row} bytes a row", size.0, size.1);
        let before = device.work_counts();
        let answer = buffer
            .texture_placement(&shaped, offset, bytes_per_row)
            .unwrap();
        let after = device.work_counts();
        let delta = |of: fn(&WorkCounts) -> u64| of(&after) - of(&before);
        let made = delta(|c| c.object_churn.vk_images_made);
        let released = delta(|c| c.object_churn.vk_images_destroyed);
        assert!(
            released <= made,
            "{case}: {released} released of {made} made"
        );
        let bound = delta(|c| c.object_churn.vk_image_bindings);
        assert_eq!(bound, 0, "{case}: the query bound an image");
        let submitted = delta(|c| c.queue_submissions);
        assert_eq!(submitted, 0, "{case}: the query submitted work");
        if answer == TexturePlacement::InPlace {
            assert!(
                (1..=2).contains(&made),
                "{case}: {made} images made for an image in place"
            );
        }
        eprintln!(
            "TEXTURE-PLACEMENT on {}: {case}: asked {answer:?}: {made} vkCreateImage, {released} \
             vkDestroyImage, none bound, nothing submitted",
            device.name()
        );
    }
    let mut refused = 0;
    for (at, &(size, usage, bytes_per_row)) in cases.iter().enumerate() {
        let shaped = descriptor(PixelFormat::BGRA8Unorm, size, usage, options);
        let offset = at as u64 * (256 << 10);
        let ask = || buffer.texture_placement(&shaped, offset, bytes_per_row);
        let make = || buffer.new_texture(&shaped, offset, bytes_per_row);
        let case = format!("{}x{} at {bytes_per_row} bytes a row", size.0, size.1);
        let (answer, texture) = made_as_asked(&device, &case, ask, make);
        let pitch = TextureShadowReason::Refused(InPlaceRefusal::Pitch);
        if answer == TexturePlacement::Shadow(pitch) {
            assert_eq!(path(&texture), "shadow", "{case}");
            refused += 1;
        }
    }
    // Which pitches a device refuses is the driver's answer: MoltenVK refuses six of these,
    // llvmpipe none (its modifier image takes every one). Every answer above matched the texture
    // made, refused or not; the count says whether this device reached the refusal.
    eprintln!(
        "TEXTURE-PLACEMENT on {}: {refused} of {} pitches refused",
        device.name(),
        cases.len()
    );
}
