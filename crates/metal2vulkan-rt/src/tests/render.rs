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
//!
//! `llvm-as` assembles the fixtures; without it a test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, within};
    use crate::tests::{devices_of_the_cfgs_kind, keeps_a_blob};
    use crate::{
        ColorAttachmentDescriptor, Device, DeviceOptions, Error, FunctionDescriptor, Gap, Gauges,
        Library, Metrics, PixelFormat, RenderPipelineDescriptor, RenderPipelineKind,
        RenderPipelineState, StageSource,
    };
    use metal2vulkan::metal::BlendFactor;
    use metal2vulkan_devtools::{fixtures_dir, llvm_assemble, AssembleError};
    use std::path::Path;
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
}
