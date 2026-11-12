//! A run of render pipeline creations over one vertex-and-fragment pair, differing only in blend
//! state, is handed **one** `VkPipelineCache`, the one the run's first creation made for their
//! shared record, so no sibling's stages are converted or compiled again
//! (`ShaderCache::caches_shared`; where the device reports what it found,
//! `ShaderCacheStatistics::driver_pipeline_hits` too).
//!
//! # What this records, and why it is worth a test
//!
//! A making's record is named by what reaches the driver's conversion of one of its stages and by
//! nothing else -- the names of its pre-rasterization and fragment shader parts, its vertex input,
//! and whether it rasterizes points (`GraphicsContent::record`) -- and the device keeps one
//! `VkPipelineCache` per record name, made by the first making that names it and handed to every
//! later one (`shader_cache/driver.rs`). So a sweep of blend states over one function pair is one
//! record and one cache, however many states it has.
//!
//! It is worth a test because the arrangement it replaced was worse than handing the driver no
//! cache at all. Every making had a `VkPipelineCache` of its own, named by the whole graphics key
//! and destroyed once the call returned, so 32 states over two modules were 32 records and 32
//! caches, no two of which the other could read. MoltenVK answers a stage's lookup out of the
//! shader module's own library cache when it is handed no pipeline cache and out of the pipeline
//! cache alone when it is handed one, so every stage of every making was converted again: 64
//! conversions over 32 blend states where one shared cache converts 2, 1.8x to 10.5x of the call
//! by the module's size (measured on an M3 Max, macOS 27.0.1, MoltenVK 1.4.2, through the driver's
//! own activity tracking,
//! `metal2vulkan-conformance/probes/pipeline_cache_across_blend_states.c`). The sibling test
//! `render_sibling_work` holds the translator's side of such a sweep -- the modules, the layout
//! and the specialized entries are shared, so nothing is translated twice; this holds the
//! driver's.
//!
//! # What each assertion says
//!
//! **Every sibling was handed the cache the first creation made** (`caches_shared`), which is a
//! count of this crate's own arrangement and holds on every device, whatever its driver reports
//! and whatever the host's load. Since the device holds its caches under the record's name, the
//! count says both that the siblings named one record and that each took that record's cache.
//!
//! **What the driver reports is the device's own.** A driver that keys a converted stage, and says
//! so per stage (MoltenVK), reports every sibling found in the cache: nothing of theirs was
//! converted or compiled again. A driver that reports a hit for the whole pipeline alone reports
//! every sibling a miss, which is right -- each sibling's pipeline is genuinely new, and
//! `pipelineCompile` is one call a state in every arrangement measured. So the run asserts only
//! that the two agree: where it reports anything it reports one answer per creation, and where it
//! reports a hit at all it reports one for every sibling.
//!
//! **Where the device links its render pipelines from graphics pipeline libraries** (lavapipe,
//! RADV, ANV, NVIDIA: `VK_EXT_graphics_pipeline_library` with fast linking,
//! `pipeline/libraries.rs`), no sibling makes a pipeline through a cache at all: its
//! pre-rasterization and fragment shader libraries are the first creation's, found by their
//! parts' content, and the only library it makes is its fragment output's, the part blend state
//! names, which holds no shader and takes no cache. So there the run asserts the same claim on
//! that route: every sibling links its own pipeline, makes exactly one library, and makes no
//! library of shaders, so nothing of its stages is converted or compiled again; and no making
//! asks for a cache, so none is shared.
//!
//! **A pass's attachment formats name no record**: creations of one pair whose passes differ only
//! in their colour format each make their libraries of shaders against their own pass, handed the
//! first's caches, so a driver that keys a stage's compile without the format (ANV) finds it
//! there; a format the translation reads changes the fragment module, which keeps a record of
//! its own (`creations_over_colour_formats_take_the_first_s_records`).
//!
//! **A library of shaders reports what its driver found**, as a whole pipeline's making does: a
//! later device over the directory an earlier one wrote counts one report per library of shaders
//! it makes, and reads no data back for one found in its record
//! (`a_library_of_shaders_reports_what_its_driver_found`). So do the runtime's own kernels, and
//! one found in its record writes no record again
//! (`the_runtime_s_own_kernels_found_in_their_records_write_nothing`).
//!
//! The device takes a shader cache directory of its own, empty, so no run reads a record an
//! earlier run stored.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects. The functions are the owned fixture
//! `validation/fixtures/public/engine/ubo_bind_offsets.ll`'s (`llvm-as`).

use super::blit_holds::{device, Scratch};
use super::descriptor_key_once::descriptor;
use crate::owned_kernel::OwnedModule;
use crate::PipelineOptions;
use metal2vulkan::metal::{BlendFactor, BlendOperation};

/// The siblings the run makes after the warm-up's one.
const RUN: u64 = 16;

/// Blend factors no two of which make one pipeline: each sibling is its own descriptor, and so
/// its own `VkPipeline`, while the record and the cache of all of them are one.
const FACTORS: [BlendFactor; 8] = [
    BlendFactor::Zero,
    BlendFactor::One,
    BlendFactor::SourceColor,
    BlendFactor::OneMinusSourceColor,
    BlendFactor::SourceAlpha,
    BlendFactor::OneMinusSourceAlpha,
    BlendFactor::DestinationColor,
    BlendFactor::DestinationAlpha,
];

/// Every sibling render creation of one function pair is handed the cache the first creation made
/// for their shared record: blend state names no record of its own, so nothing is converted again.
/// Where the device links from libraries, every sibling links from the first creation's libraries
/// of shaders instead, and takes no cache.
#[test]
fn every_sibling_render_creation_takes_the_first_s_driver_cache() {
    let scratch = Scratch::new("driver-pipeline-compile");
    let device = device(&scratch.0);
    let base = descriptor(&device);
    let options = PipelineOptions::default();
    let sibling = |i: u64| {
        let mut d = base.clone();
        let color = d.color_attachments[0]
            .as_mut()
            .expect("the fixture's target");
        color.blending_enabled = true;
        color.source_rgb_blend_factor = FACTORS[(i % 8) as usize];
        color.destination_rgb_blend_factor = FACTORS[(i / 8 % 8) as usize];
        color.rgb_blend_operation = BlendOperation::Add;
        d
    };
    // The warm-up's creation makes the record's cache, converts the stages and compiles them, so
    // the run's siblings ask the driver for nothing but a pipeline over stages its cache holds.
    let _first = device
        .new_render_pipeline_state(&sibling(0), &options)
        .unwrap();
    let shared_before = device.shared().shader_cache.caches_shared();
    let before = device.shader_cache_statistics();
    let counts_before = device.work_counts();
    let mut kept = Vec::with_capacity(RUN as usize);
    for i in 0..RUN {
        kept.push(
            device
                .new_render_pipeline_state(&sibling(1 + i), &options)
                .unwrap(),
        );
    }
    let after = device.shader_cache_statistics();
    let shared = device.shared().shader_cache.caches_shared() - shared_before;
    let hits = after.driver_pipeline_hits - before.driver_pipeline_hits;
    let misses = after.driver_pipeline_misses - before.driver_pipeline_misses;
    println!(
        "DRIVER-PIPELINE-COMPILE: {shared} of {RUN} siblings took the first's cache, \
         {hits} hits and {misses} misses reported"
    );
    if device.shared().libraries.is_some() {
        let counts = device.work_counts();
        let made = |of: fn(&crate::WorkCounts) -> u64| of(&counts) - of(&counts_before);
        let linked = made(|c| c.pipelines_linked);
        let libraries = made(|c| c.pipeline_libraries_made);
        let fragment_shaders = made(|c| c.fragment_shader_libraries_made);
        println!(
            "DRIVER-PIPELINE-COMPILE: linked from libraries: {linked} pipelines linked, \
             {libraries} libraries made, {fragment_shaders} of a fragment shader"
        );
        assert_eq!(
            (linked, libraries, fragment_shaders, shared),
            (RUN, RUN, 0, 0),
            "every one of {RUN} sibling creations must link its own pipeline from the first \
             creation's libraries of shaders, making its fragment output's library alone, which \
             takes no cache: {linked} linked, {libraries} libraries made, {fragment_shaders} of a \
             fragment shader, {shared} caches handed over"
        );
        return;
    }
    assert_eq!(
        shared, RUN,
        "every one of {RUN} sibling creations must be handed the `VkPipelineCache` the first \
         creation made for the record they share, blend state naming no record of its own: \
         {shared} were"
    );
    if hits + misses == 0 {
        // The device reports nothing it found, so there is nothing more to read here.
        return;
    }
    assert_eq!(
        hits + misses,
        RUN,
        "a device that reports what it found must report one answer per creation: {hits} hits and \
         {misses} misses over {RUN} creations"
    );
    assert!(
        hits == 0 || hits == RUN,
        "a driver that reports a sibling found in the first's cache must report every one of \
         {RUN} found there, their stages being the same: {hits} hits and {misses} misses"
    );
}

/// Where the device links its render pipelines from libraries and its makings report, the making
/// of each library of shaders reports what its driver found, as a whole pipeline's making does: a
/// later device over the directory an earlier one's creation of the same state wrote counts one
/// report for each of the state's pre-rasterization and fragment shader libraries, and each one
/// the driver reports found in the cache its record loaded reads no data back and writes nothing
/// (`driver_data_found`). Unreported, a library's making read its data back and compared the
/// bytes, and RADV's data after a library found in its record is now and then not the record's
/// byte for byte (`shader_cache/driver.rs`), so such a device wrote the record again.
#[test]
fn a_library_of_shaders_reports_what_its_driver_found() {
    let scratch = Scratch::new("driver-library-reports");
    let options = PipelineOptions::default();
    {
        let device = device(&scratch.0);
        let shared = device.shared();
        if shared.libraries.is_none() || !shared.shader_cache.driver_reports() {
            println!(
                "DRIVER-LIBRARY-REPORTS: not covered (links from libraries: {}, makings report: \
                 {})",
                shared.libraries.is_some(),
                shared.shader_cache.driver_reports()
            );
            return;
        }
        drop(
            device
                .new_render_pipeline_state(&descriptor(&device), &options)
                .unwrap(),
        );
    }
    // The first device is gone, so its cache thread wrote what its makings handed it.
    let device = device(&scratch.0);
    let base = descriptor(&device);
    let before = device.shader_cache_statistics();
    let counts_before = device.work_counts();
    let _state = device.new_render_pipeline_state(&base, &options).unwrap();
    let after = device.shader_cache_statistics();
    let counts = device.work_counts();
    let made = |of: fn(&crate::WorkCounts) -> u64| of(&counts) - of(&counts_before);
    let since = |of: fn(&crate::ShaderCacheStatistics) -> u64| of(&after) - of(&before);
    let (hits, misses) = (
        since(|s| s.driver_pipeline_hits),
        since(|s| s.driver_pipeline_misses),
    );
    let (found, saves) = (since(|s| s.driver_data_found), since(|s| s.driver_saves));
    println!(
        "DRIVER-LIBRARY-REPORTS: {hits} hits and {misses} misses reported, {found} found and read \
         nothing back, {saves} records' data written, {} loaded",
        since(|s| s.driver_loads)
    );
    // The fresh device makes each of the state's four libraries once and links them.
    assert_eq!(
        (
            made(|c| c.pipelines_linked),
            made(|c| c.pipeline_libraries_made),
            made(|c| c.fragment_shader_libraries_made)
        ),
        (1, 4, 1),
        "the state links one pipeline from four libraries made for it"
    );
    assert_eq!(
        hits + misses,
        2,
        "each of the state's two libraries of shaders reports what its driver found: {hits} hits \
         and {misses} misses"
    );
    assert_eq!(
        found, hits,
        "each library its driver reports found in the cache its record loaded reads no data back"
    );
    if hits == 2 {
        assert_eq!(saves, 0, "libraries found in their records write nothing");
    }
}

/// The runtime's own kernels made again on a later device report what their driver found, and
/// those it reports found in the records an earlier device over the same directory wrote write no
/// record again (`driver_saves`), whatever bytes the driver's data holds after them: RADV's
/// differs from the record's now and then (`shader_cache/driver.rs`), which a byte comparison
/// took for new data and wrote the record again at the device's every making of them.
#[test]
fn the_runtime_s_own_kernels_found_in_their_records_write_nothing() {
    let scratch = Scratch::new("driver-own-kernels");
    let kernels = [OwnedModule::RowsCopy, OwnedModule::VisibilityCombine];
    {
        let device = device(&scratch.0);
        let reports = device.shared().shader_cache.driver_reports();
        if !reports {
            println!("DRIVER-OWN-KERNELS: not covered (the device's makings report nothing)");
            return;
        }
        for which in kernels {
            device.owned_kernel(which).unwrap();
        }
    }
    // The first device is gone, so its cache thread wrote what its makings handed it.
    let device = device(&scratch.0);
    let before = device.shader_cache_statistics();
    let counts_before = device.work_counts();
    for which in kernels {
        device.owned_kernel(which).unwrap();
    }
    let after = device.shader_cache_statistics();
    let made = device.work_counts().owned_kernels_made - counts_before.owned_kernels_made;
    let since = |of: fn(&crate::ShaderCacheStatistics) -> u64| of(&after) - of(&before);
    let (hits, misses) = (
        since(|s| s.driver_pipeline_hits),
        since(|s| s.driver_pipeline_misses),
    );
    let (found, saves) = (since(|s| s.driver_data_found), since(|s| s.driver_saves));
    println!(
        "DRIVER-OWN-KERNELS: {made} kernels made, {hits} hits and {misses} misses reported, \
         {found} found and read nothing back, {saves} records' data written, {} loaded",
        since(|s| s.driver_loads)
    );
    assert_eq!(
        made,
        kernels.len() as u64,
        "the later device made each kernel"
    );
    assert_eq!(
        hits + misses,
        made,
        "each of the runtime's kernels reports what its driver found: {hits} hits and {misses} \
         misses"
    );
    assert_eq!(
        found, hits,
        "each kernel its driver reports found in the cache its record loaded reads no data back"
    );
    if hits == made {
        assert_eq!(saves, 0, "kernels found in their records write nothing");
    }
}

/// Render creations of one function pair whose passes differ only in their colour attachment's
/// format take the first creation's records: a pass's attachment formats reach no stage's
/// conversion, so they name no record (`GraphicsContent::record_parts`), and each making of
/// shaders is handed the cache the first made. Where the device links from libraries, each such
/// creation makes its own pre-rasterization and fragment shader libraries, against its own pass,
/// and both are handed the first's caches; a whole making is handed the first's one. A format
/// the translation reads keeps a record of its own: under `TextureWriteRoundingMode::NearestEven`
/// the fragment function rounds a half-float channel itself (`pipeline/color_writes.rs`), so its
/// module for an `RGBA16Float` attachment is another, and its making is handed a cache of its
/// own. Where the device links, that creation makes its fragment shader library alone: its
/// pre-rasterization library is the one the `RGBA16Float` creation before it made against the
/// same pass.
///
/// The counts are this crate's own arrangement and hold on every device. What the driver
/// reports it found is the driver's: ANV finds a fragment shader library of one module made
/// against another format's pass in the shared cache (Mesa 26.2.6, Iris Xe:
/// `APPLICATION_PIPELINE_CACHE_HIT`, where a cache of its own compiled it again, 3.3 to 3.7 s
/// for a module of some 200,000 words), so the run prints it and asserts only that a reporting
/// device reports one answer per making of shaders.
#[test]
fn creations_over_colour_formats_take_the_first_s_records() {
    use metal2vulkan::metal::{PixelFormat, TextureWriteRoundingMode};
    use metal2vulkan::pipeline::ColorAttachmentDescriptor;
    let scratch = Scratch::new("driver-pipeline-formats");
    let device = device(&scratch.0);
    let base = descriptor(&device);
    let options = PipelineOptions::default();
    let links = device.shared().libraries.is_some();
    let with = |format: PixelFormat, rounding: TextureWriteRoundingMode| {
        let mut d = base.clone();
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(format));
        d.texture_write_rounding_mode = rounding;
        d
    };
    // What a creation of `d` did: the caches it was handed that an earlier making made, the
    // libraries it made of a fragment shader, and what its driver reported.
    let create = |d: &metal2vulkan::pipeline::RenderPipelineDescriptor| {
        let shared_before = device.shared().shader_cache.caches_shared();
        let before = device.shader_cache_statistics();
        let counts_before = device.work_counts();
        let state = device.new_render_pipeline_state(d, &options).unwrap();
        let after = device.shader_cache_statistics();
        let shared = device.shared().shader_cache.caches_shared() - shared_before;
        let fragment_shaders = device.work_counts().fragment_shader_libraries_made
            - counts_before.fragment_shader_libraries_made;
        let hits = after.driver_pipeline_hits - before.driver_pipeline_hits;
        let misses = after.driver_pipeline_misses - before.driver_pipeline_misses;
        (state, shared, fragment_shaders, hits, misses)
    };
    let _first = create(&with(
        PixelFormat::RGBA8Unorm,
        TextureWriteRoundingMode::TowardZero,
    ));
    // Each making of shaders a creation over another format makes: where the device links, its
    // two libraries of shaders, each against its own pass; else its one whole pipeline.
    let makings = if links { 2 } else { 1 };
    let mut kept = Vec::new();
    for format in [PixelFormat::BGRA8Unorm, PixelFormat::RGBA16Float] {
        let (state, shared, fragment_shaders, hits, misses) =
            create(&with(format, TextureWriteRoundingMode::TowardZero));
        println!(
            "DRIVER-PIPELINE-FORMATS: {format:?}: {shared} of {makings} makings took an earlier \
             making's cache, {fragment_shaders} fragment shader libraries made, {hits} hits and \
             {misses} misses reported"
        );
        assert_eq!(
            shared, makings,
            "a creation whose pass differs from the first's only in its colour format ({format:?}) \
             must be handed the first's cache for each of its {makings} makings of shaders, the \
             format naming no record: {shared} were"
        );
        if links {
            assert_eq!(
                fragment_shaders, 1,
                "a pass of another format takes a fragment shader library of its own"
            );
        }
        if hits + misses > 0 {
            assert_eq!(
                hits + misses,
                makings,
                "a device that reports what it found reports one answer per making of shaders"
            );
        }
        kept.push(state);
    }
    // A format the translation reads: the rounding module is another, so its making names a
    // record of its own and is handed no earlier making's cache. Where the device links, its
    // fragment shader library is the one library it makes.
    let (_rounded, shared, fragment_shaders, hits, misses) = create(&with(
        PixelFormat::RGBA16Float,
        TextureWriteRoundingMode::NearestEven,
    ));
    println!(
        "DRIVER-PIPELINE-FORMATS: RGBA16Float rounded to nearest even: {shared} makings took an \
         earlier making's cache, {fragment_shaders} fragment shader libraries made, {hits} hits \
         and {misses} misses reported"
    );
    assert_eq!(
        shared, 0,
        "a fragment module the format changes keeps a record of its own: {shared} makings took an \
         earlier making's cache"
    );
    if links {
        assert_eq!(
            fragment_shaders, 1,
            "the rounding module takes a fragment shader library of its own"
        );
    }
    if hits + misses > 0 {
        assert_eq!(
            hits + misses,
            1,
            "a device that reports what it found reports one answer per making of shaders"
        );
    }
}
