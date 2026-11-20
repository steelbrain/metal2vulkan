//! Render pipeline states, under a device cfg on every device of the cfg's kind, of the owned
//! fixtures `examples/runtime_examples.ll` and `engine/rt6_samples.ll`.
//!
//! - The first state of a content lowers its two functions and makes one `VkPipeline` (the
//!   triangle class's); a second state of that descriptor lowers nothing, makes nothing and
//!   shares the first's compiled content, which the device's map of contents holds beside its
//!   compute kernels. Another colour format, a blended attachment, or a stencil attachment with a
//!   second colour attachment is another content.
//! - Filled past a cap of 2, the map holds 2 render contents and forgets the least recently used,
//!   which lives while a state holds it and goes once that state does; a forgotten content
//!   compiles again, and one the map holds does not.
//! - More than one sample is refused by kind, making no `VkPipeline`.
//! - A fragment function that reads its `[[sample_id]]` makes its state, on the floor's
//!   `sampleRateShading`.
//! - A device on a cache directory a device before it used lowers nothing at creation: both
//!   lowerings are read from the directory, and the triangle class's `VkPipeline` is made from
//!   its file; the first device writes one pipeline file for each `VkPipeline` it made, and the
//!   second writes nothing.
//! - A device on a cache directory a device before it made a render content in takes the content
//!   by its record, reading the two lowerings it names and specializing neither function; with
//!   the record removed it specializes both, and a record whose lowering was removed is stale,
//!   counted. Each kind of input that changes a lowering (the library's content, the function, its
//!   constant values, a field its specialization reads, and the device profile where the host has
//!   two devices of the cfg's kind), changed once, misses the record, and a record of another
//!   content under a content's name is no record of it.
//!
//! `llvm-as` assembles the fixtures; without it a test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::cache::{hex, read_record};
    use crate::tests::command::on_device::{each_device, within};
    use crate::tests::draw::on_device::ABSENT_READ;
    use crate::tests::{devices_of_the_cfgs_kind, keeps_a_blob};
    use crate::{
        ColorAttachmentDescriptor, Device, DeviceOptions, Error, FunctionDescriptor, Gap, Gauges,
        Library, Metrics, PhysicalDevice, PixelFormat, RenderPipelineDescriptor,
        RenderPipelineKind, RenderPipelineState, StageSource,
    };
    use metal2vulkan::cache::{Hash256, VariantKey};
    use metal2vulkan::library::{
        FunctionCompileKey, FunctionConstantValue, FunctionConstantValues,
    };
    use metal2vulkan::metal::{BlendFactor, DataType};
    use metal2vulkan::pipeline::StageEntries;
    use metal2vulkan::Budget;
    use metal2vulkan_devtools::{fixtures_dir, llvm_assemble, llvm_assemble_text, AssembleError};
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::time::SystemTime;

    /// The fixture `examples/runtime_examples.ll`'s library, or None where `llvm-as` is missing.
    fn library() -> Option<Library> {
        library_of("examples/runtime_examples.ll")
    }

    /// The library of the fixture at `fixture` under the fixtures directory, or None where
    /// `llvm-as` is missing.
    fn library_of(fixture: &str) -> Option<Library> {
        let ll = fixtures_dir().join(fixture);
        match llvm_assemble(&ll) {
            Ok(bitcode) => Some(Library::from_air(&bitcode).expect("the library")),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                None
            }
            Err(error) => panic!("{}: {error:?}", ll.display()),
        }
    }

    /// `colored_vertex` with the fragment function `fragment`, drawing into one colour attachment
    /// of `color`.
    fn descriptor(
        library: &Library,
        fragment: &str,
        color: PixelFormat,
    ) -> RenderPipelineDescriptor {
        let function = |name| {
            StageSource::Function(FunctionDescriptor::new(
                library.function(name).expect("the function"),
            ))
        };
        let mut d = RenderPipelineDescriptor::new(function("colored_vertex"));
        d.fragment = Some(function(fragment));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(color));
        d
    }

    fn state(name: &str, device: &Device, d: &RenderPipelineDescriptor) -> RenderPipelineState {
        (device.new_render_pipeline_state(d, crate::tests::no_constexpr_samplers))
            .unwrap_or_else(|error| panic!("{name}: {error}"))
    }

    /// Whether two states share their compiled content.
    fn shared(a: &RenderPipelineState, b: &RenderPipelineState) -> bool {
        Arc::ptr_eq(&a.inner.compiled, &b.inner.compiled)
    }

    /// The lowerings (made and read from the cache directory) and the `VkPipeline`s from `before`
    /// to `after`.
    fn made(before: &Metrics, after: &Metrics) -> (u64, u64) {
        (
            (after.translations + after.translations_loaded)
                - (before.translations + before.translations_loaded),
            after.pipelines_created - before.pipelines_created,
        )
    }

    /// Every device makes the state of `rt6_samples`' full-screen fill, whose fragment function
    /// reads its `[[sample_id]]`: SPIR-V's `SampleRateShading`, which the floor's
    /// `sampleRateShading` enables on every device. Without it in the floor, the translator
    /// refuses the function on every device.
    #[test]
    fn every_device_makes_a_state_whose_fragment_reads_its_sample() {
        let Some(library) = library_of("engine/rt6_samples.ll") else {
            return;
        };
        let function = |name| {
            StageSource::Function(FunctionDescriptor::new(
                library.function(name).expect("the function"),
            ))
        };
        let mut d = RenderPipelineDescriptor::new(function("rt6_fill_vertex"));
        d.fragment = Some(function("rt6_fill_rgba16float"));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float));
        each_device("render_sample_id", |name, _, device| {
            let counters = device.counters();
            let before = counters.metrics();
            let made_state = state(name, &device, &d);
            assert_eq!(
                made(&before, &counters.metrics()).1,
                1,
                "{name}: the fill's pipeline"
            );
            drop(made_state);
        });
    }

    #[test]
    fn every_device_shares_a_render_content_among_its_states() {
        let Some(library) = library() else {
            return;
        };
        let first = descriptor(&library, "colored_fragment", PixelFormat::BGRA8Unorm);
        let mut blended = first.clone();
        if let Some(color) = &mut blended.color_attachments[0] {
            color.blending_enabled = true;
            color.source_rgb_blend_factor = BlendFactor::SourceAlpha;
            color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
        }
        let mut stencil = descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm);
        stencil.color_attachments[1] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        stencil.stencil_format = Some(PixelFormat::Stencil8);
        let others = [
            (
                "another format",
                descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm),
            ),
            ("blended", blended),
            ("a stencil attachment and two colours", stencil),
        ];
        each_device("render_shared", |name, _, device| {
            let counters = device.counters();
            let gauges = counters.gauges();
            let before = counters.metrics();
            let a = state(name, &device, &first);
            assert_eq!(
                made(&before, &counters.metrics()),
                (2, 1),
                "{name}: the first state's lowerings and pipelines"
            );
            let held = counters.gauges();
            assert_eq!(
                (held.render_contents, held.kernel_entries),
                (gauges.render_contents + 1, gauges.kernel_entries + 1),
                "{name}: one content, held by the map"
            );
            let before = counters.metrics();
            let b = state(name, &device, &first.clone());
            assert_eq!(
                made(&before, &counters.metrics()),
                (0, 0),
                "{name}: a second state of the content"
            );
            assert!(shared(&a, &b), "{name}: one compiled content");
            for (other, d) in &others {
                let before = counters.metrics();
                let c = state(name, &device, d);
                let (lowered, pipelines) = made(&before, &counters.metrics());
                assert_eq!(pipelines, 1, "{name}: {other}: its own pipeline");
                assert!(lowered <= 2, "{name}: {other}: at most two lowerings");
                assert!(!shared(&a, &c), "{name}: {other}: another content");
            }
            assert_eq!(
                counters.gauges().render_contents,
                gauges.render_contents + 1 + others.len() as u64,
                "{name}: a content each"
            );
            let compiled = Arc::clone(&a.inner.compiled);
            drop((a, b));
            // The map holds it still, beside this test's handle.
            assert_eq!(Arc::strong_count(&compiled), 2, "{name}: held by the map");
            println!(
                "{name}: two states of a content share it; {} other contents made apart",
                others.len()
            );
        });
    }

    /// Filled past a cap of 2, the map holds 2 render contents and forgets the least recently
    /// used, which lives while a state holds it and goes once that state does; a content
    /// forgotten compiles again, and one the map holds does not.
    #[test]
    fn every_device_bounds_its_render_contents_and_frees_a_forgotten_one_with_its_last_state() {
        let Some(library) = library() else {
            return;
        };
        let [one, two, three] = [
            PixelFormat::BGRA8Unorm,
            PixelFormat::RGBA8Unorm,
            PixelFormat::RGBA16Float,
        ]
        .map(|color| descriptor(&library, "colored_fragment", color));
        each_device("render_past_cap", |name, _, device| {
            device.core().kernels.set_cap(2);
            let counters = device.counters();
            let gauges = || {
                let gauges = counters.gauges();
                (gauges.render_contents, gauges.kernel_entries)
            };
            let lowered = || counters.metrics().translations;
            let start = lowered();
            let a = state(name, &device, &one);
            let each = lowered() - start;
            assert!(each > 0, "{name}: a lowering");
            let b = state(name, &device, &two);
            assert_eq!(gauges(), (2, 2), "{name}");
            // `one` is the least recently used, and goes from the map; its state keeps it.
            let c = state(name, &device, &three);
            assert_eq!(
                gauges(),
                (3, 2),
                "{name}: a forgotten content its state holds"
            );
            let b_again = state(name, &device, &two);
            assert!(shared(&b, &b_again), "{name}: `two` is held");
            assert_eq!(lowered() - start, 3 * each, "{name}: `two` compiled once");
            drop(a);
            assert_eq!(
                gauges(),
                (2, 2),
                "{name}: the forgotten content went with its last state"
            );
            // `one` compiles again, and `three`, now the least recently used, goes from the map.
            let a = state(name, &device, &one);
            assert_eq!(lowered() - start, 4 * each, "{name}: `one` compiled again");
            assert_eq!(gauges(), (3, 2), "{name}");
            drop(c);
            assert_eq!(gauges(), (2, 2), "{name}");
            drop((a, b, b_again));
            assert_eq!(gauges(), (2, 2), "{name}: the map holds 2");
            println!(
                "{name}: the map held 2 of 3 render contents, and freed a forgotten one with its state"
            );
        });
    }

    #[test]
    fn every_device_refuses_what_it_does_not_build_by_kind() {
        let Some(library) = library() else {
            return;
        };
        let plain = descriptor(&library, "colored_fragment", PixelFormat::BGRA8Unorm);
        let mut samples = plain.clone();
        samples.raster_sample_count = 3;
        let mut separate = plain.clone();
        separate.depth_format = Some(PixelFormat::Depth32Float);
        separate.stencil_format = Some(PixelFormat::Stencil8);
        let cases = [
            ("three samples", samples, RenderPipelineKind::SampleCount),
            (
                "separate depth and stencil formats",
                separate,
                RenderPipelineKind::DepthStencilFormats,
            ),
        ];
        each_device("render_refused", |name, _, device| {
            let counters = device.counters();
            for (case, d, kind) in &cases {
                let before = counters.metrics().pipelines_created;
                match device.new_render_pipeline_state(d, crate::tests::no_constexpr_samplers) {
                    Err(Error::Unsupported(Gap::RenderPipeline { kind: refused })) => {
                        assert_eq!(refused, *kind, "{name}: {case}")
                    }
                    Err(error) => panic!("{name}: {case}: {error}"),
                    Ok(_) => panic!("{name}: {case}: made"),
                }
                assert_eq!(
                    counters.metrics().pipelines_created,
                    before,
                    "{name}: {case}: no pipeline"
                );
            }
            assert_eq!(counters.gauges().render_contents, 0, "{name}: no content");
            println!("{name}: {} kinds refused, no pipeline made", cases.len());
        });
    }

    /// The pipeline cache blobs in `dir`, by name, with their lengths and times.
    fn blobs(dir: &Path) -> Vec<(String, u64, SystemTime)> {
        let mut all = Vec::new();
        for file in std::fs::read_dir(dir).into_iter().flatten().flatten() {
            let name = file.file_name().to_string_lossy().into_owned();
            if name.starts_with("pipelines-") {
                let meta = file.metadata().expect("its metadata");
                all.push((name, meta.len(), meta.modified().expect("its time")));
            }
        }
        all.sort();
        all
    }

    #[test]
    fn every_device_takes_a_render_content_a_device_before_it_made_from_the_cache_directory() {
        let Some(library) = library() else {
            return;
        };
        let d = descriptor(&library, "colored_fragment", PixelFormat::BGRA8Unorm);
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let dir = std::env::temp_dir().join(format!("m2v-rt-render-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            let mut first = Vec::new();
            for round in 0..2 {
                let options = DeviceOptions {
                    pipeline_cache_dir: Some(dir.clone()),
                    ..DeviceOptions::default()
                };
                let device = Device::new(physical, options)
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                let kept = keeps_a_blob(name, &device);
                let counters = device.counters();
                let made = state(name, &device, &d);
                let metrics = counters.metrics();
                let counted = (metrics.translations, metrics.translations_loaded);
                assert_eq!(
                    counted,
                    [(2, 0), (0, 2)][round],
                    "{name}: device {round}'s lowerings"
                );
                assert_eq!(metrics.pipelines_created, 1, "{name}: device {round}");
                drop(made);
                drop(device);
                // The teardown ends once the reaper wrote what it was given.
                within("the teardown", || counters.gauges() == Gauges::default());
                let files = blobs(&dir);
                if round == 0 {
                    let want = usize::from(kept);
                    assert_eq!(
                        files.len(),
                        want,
                        "{name}: the device's pipeline cache blob"
                    );
                    first = files;
                } else {
                    let written = counters.metrics().cache_bytes_written;
                    assert_eq!(written, 0, "{name}: the second device writes nothing");
                    assert_eq!(files, first, "{name}: the blob as it was");
                }
            }
            let _ = std::fs::remove_dir_all(&dir);
            println!("{name}: a second device lowers nothing and writes nothing");
        }
    }

    /// The files of kind `kind` (`renders`, the records, or `translations`, the lowerings) under
    /// the generations of the cache directory `dir`, by path.
    fn kept_files(dir: &Path, kind: &str) -> Vec<PathBuf> {
        let mut files = Vec::new();
        for generation in std::fs::read_dir(dir).into_iter().flatten().flatten() {
            if generation.file_name().to_string_lossy().starts_with('g') {
                let kept = std::fs::read_dir(generation.path().join(kind));
                files.extend(kept.into_iter().flatten().flatten().map(|file| file.path()));
            }
        }
        files.sort();
        files
    }

    /// A device of `physical` on the cache directory `dir`.
    fn on_directory(name: &str, physical: &PhysicalDevice, dir: &Path) -> Device {
        let options = DeviceOptions {
            pipeline_cache_dir: Some(dir.to_owned()),
            ..DeviceOptions::default()
        };
        Device::new(physical, options).unwrap_or_else(|error| panic!("{name}: {error}"))
    }

    /// The translator's key of `d`'s descriptor on `device`, which names its record.
    fn content(device: &Device, d: &RenderPipelineDescriptor) -> Hash256 {
        *device
            .core()
            .translator
            .queried_render_descriptor_key(d)
            .digest()
    }

    /// Drops `device` and waits for its teardown, which ends once the reaper wrote what it was
    /// given.
    fn torn_down(device: Device) {
        let counters = device.counters();
        drop(device);
        within("the teardown", || counters.gauges() == Gauges::default());
    }

    /// A device on a cache directory a device before it made a render content in takes the
    /// content by its record: it reads the two lowerings the record names, and reads neither
    /// function's AIR (the translator's stage entries count no read of a stage, to specialize it
    /// or for a plain answer). The negative control: with the record removed, a device reads the
    /// same two lowerings but reads both functions' AIR again, and keeps the record again. A record
    /// whose fragment lowering was removed is stale, counted, and the device lowers that function
    /// again and reads the vertex function's lowering.
    #[test]
    fn every_device_takes_a_render_content_by_its_record_without_specializing() {
        let Some(library) = library() else {
            return;
        };
        let d = descriptor(&library, "colored_fragment", PixelFormat::BGRA8Unorm);
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let dir = std::env::temp_dir().join(format!("m2v-rt-record-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            let mut key = None;
            for round in 0..4 {
                if round == 2 {
                    for record in kept_files(&dir, "renders") {
                        std::fs::remove_file(record).expect("the record removed");
                    }
                }
                if round == 3 {
                    let records = kept_files(&dir, "renders");
                    assert_eq!(records.len(), 1, "{name}: the record kept again");
                    let bytes = std::fs::read(&records[0]).expect("the record");
                    let key = key.expect("the content's key");
                    let lowerings = read_record(&bytes, &key).expect("a record of the content");
                    let fragment = hex(&lowerings.fragment.expect("the fragment's lowering"));
                    let lowered = kept_files(&dir, "translations");
                    let file = lowered.iter().find(|file| file.ends_with(&fragment));
                    std::fs::remove_file(file.expect("the fragment's lowering"))
                        .expect("the fragment's lowering removed");
                }
                let device = on_directory(name, physical, &dir);
                key = Some(content(&device, &d));
                let counters = device.counters();
                let (read_before, _) = device.core().entries.counts();
                let made = state(name, &device, &d);
                let (read_after, _) = device.core().entries.counts();
                let m = counters.metrics();
                let counted = (
                    m.render_records_read,
                    m.render_records_stale,
                    m.translations,
                    m.translations_loaded,
                );
                let want = [(0, 0, 2, 0), (1, 0, 0, 2), (0, 0, 0, 2), (0, 1, 1, 1)][round];
                assert_eq!(
                    counted, want,
                    "{name}: device {round}'s records read and stale, lowerings made and read"
                );
                let reads = read_after - read_before;
                assert_eq!(
                    reads > 0,
                    round != 1,
                    "{name}: device {round} read a stage's AIR {reads} times"
                );
                drop(made);
                torn_down(device);
                let records = kept_files(&dir, "renders").len();
                assert_eq!(records, 1, "{name}: device {round} leaves one record");
            }
            let _ = std::fs::remove_dir_all(&dir);
            println!("{name}: a record takes the content with no read of a function's AIR");
        }
    }

    /// Each kind of input that changes a render content's lowerings, changed once from a content a
    /// device made on a cache directory, misses that content's record on a device after it, and
    /// never reads another content's record: the library's content (one constant of its fragment
    /// function's body, every name kept), the fragment function, its constant values, and a field
    /// of the descriptor its specialization reads (the colour attachment moved from index 0 to 1,
    /// so the colour the function writes has none). The translator's keys of each changed
    /// content's compile show the change reaches a lowering's key, and the base content, made
    /// first on the second device, reads its record, so the record is there to miss. The base
    /// content's record copied over the record of the constant's content is no record of it. The
    /// device profile is the fifth kind: where the host has two devices of the cfg's kind whose
    /// profiles differ, the second misses the first's record of the base descriptor; with one, that
    /// part skips visibly. The translator's semantics and format versions are in every record,
    /// which `a_record_reads_only_as_its_own_content_s` reads, and the semantics version in every
    /// key of the translator's.
    #[test]
    fn every_device_misses_a_record_for_each_kind_of_input_that_changes_a_lowering() {
        let Some(examples) = library() else {
            return;
        };
        let assemble = |text: &str| match llvm_assemble_text(text) {
            Ok(bitcode) => Some(Library::from_air(&bitcode).expect("the library")),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                None
            }
            Err(error) => panic!("the fragment: {error:?}"),
        };
        let Some(gated) = assemble(ABSENT_READ) else {
            return;
        };
        let added =
            "<float 0.000000e+00, float 1.000000e+00, float 0.000000e+00, float 0.000000e+00>";
        let moved = added.replace("float 1.000000e+00", "float 2.000000e+00");
        assert_eq!(
            ABSENT_READ.matches(added).count(),
            1,
            "the constant the fragment adds"
        );
        let Some(moved) = assemble(&ABSENT_READ.replace(added, &moved)) else {
            return;
        };
        let fragment = |library: &Library, has_t: u8| {
            let function = library.function("absent_read_fragment");
            let mut f = FunctionDescriptor::new(function.expect("the function"));
            f.constants = FunctionConstantValues::defaults()
                .with_index(0, FunctionConstantValue::new(DataType::Bool, [has_t]));
            f.specialized = true;
            StageSource::Function(f)
        };
        let with = |fragment: StageSource| {
            let vertex = examples.function("colored_vertex").expect("the function");
            let mut d = RenderPipelineDescriptor::new(StageSource::Function(
                FunctionDescriptor::new(vertex),
            ));
            d.fragment = Some(fragment);
            d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
            d
        };
        let base = with(fragment(&gated, 0));
        let mut field = base.clone();
        field.color_attachments[0] = None;
        field.color_attachments[1] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        let changes = [
            ("the library's content", with(fragment(&moved, 0))),
            (
                "the fragment function",
                descriptor(&examples, "colored_fragment", PixelFormat::RGBA8Unorm),
            ),
            ("its constant values", with(fragment(&gated, 1))),
            ("a field its specialization reads", field),
        ];
        // The keys of each stage's lowering, as the translator keys the compile.
        let keys = |device: &Device, d: &RenderPipelineDescriptor| {
            let keys = (device.core().translator).render_variant_key(
                d,
                &VariantKey::new(),
                &Budget::standard(),
                &StageEntries::new(),
            );
            let keys = keys.unwrap_or_else(|error| panic!("the keys: {error}"));
            (keys.vertex, keys.fragment)
        };
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        let dir_of = |n: usize| {
            let dir = format!("m2v-rt-records-{}-{n}", std::process::id());
            std::env::temp_dir().join(dir)
        };
        let mut contents = Vec::new();
        for (n, (name, physical)) in devices.iter().enumerate() {
            let dir = dir_of(n);
            let _ = std::fs::remove_dir_all(&dir);
            let first = on_directory(name, physical, &dir);
            contents.push(content(&first, &base));
            drop(state(name, &first, &base));
            let base_keys: (FunctionCompileKey, Option<FunctionCompileKey>) = keys(&first, &base);
            torn_down(first);
            let device = on_directory(name, physical, &dir);
            // The base content's record, copied over the constant's content's.
            let records = kept_files(&dir, "renders");
            assert_eq!(records.len(), 1, "{name}: the base content's record");
            let over = records[0].with_file_name(hex(&content(&device, &changes[2].1)));
            std::fs::copy(&records[0], &over).expect("the record copied");
            let counters = device.counters();
            drop(state(name, &device, &base));
            let m = counters.metrics();
            assert_eq!(
                m.render_records_read, 1,
                "{name}: the base content's record read"
            );
            for (what, d) in &changes {
                assert_ne!(keys(&device, d), base_keys, "{name}: {what} changes a key");
                let before = counters.metrics();
                drop(state(name, &device, d));
                let after = counters.metrics();
                let counted = |m: &Metrics| (m.render_records_read, m.render_records_stale);
                assert_eq!(
                    counted(&after),
                    counted(&before),
                    "{name}: {what}: no record"
                );
                assert!(
                    after.translations > before.translations,
                    "{name}: {what}: a lowering made"
                );
            }
            torn_down(device);
            println!("{name}: each kind of input changed once misses the record");
        }
        // The device profile: each other device of the cfg's kind on the first's directory, which
        // holds the first's record of the base content.
        let first_name = &devices[0].0;
        let mut others = 0;
        for ((name, physical), content) in devices.iter().zip(&contents).skip(1) {
            if *content == contents[0] {
                println!("{name}: the profile of {first_name}");
                continue;
            }
            let device = on_directory(name, physical, &dir_of(0));
            let counters = device.counters();
            drop(state(name, &device, &base));
            let m = counters.metrics();
            let counted = (m.render_records_read, m.render_records_stale);
            assert_eq!(counted, (0, 0), "{name}: no record of {first_name}'s");
            torn_down(device);
            others += 1;
        }
        if others == 0 {
            println!("SKIPPED: the device profile: one profile among this cfg's devices");
        }
        for n in 0..devices.len() {
            let _ = std::fs::remove_dir_all(dir_of(n));
        }
    }
}
