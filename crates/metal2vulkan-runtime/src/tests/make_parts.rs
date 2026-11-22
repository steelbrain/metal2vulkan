//! The timed parts of a texture's and an exported surface's make (`Metrics::image_creates` and
//! the others'), under a device cfg, on every device of the cfg's kind: a texture made at an index
//! is one image create and one bind, each of its image's bytes, one view create for its sampled
//! view, one wait for the arrays' lock and one for the work lock, and no surface part; a surface
//! of one plane made in place is, beside its plane's image, bind, view and wait for the work lock,
//! one allocation and one export of its handle (an opaque one) or three (a dma-buf, which makes its
//! plane's image twice), each of the allocation's bytes. A surface of the copy form is printed.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::each_device;
    use crate::{
        ExportHandle, IndexClass, Metrics, PixelFormat, PlaneSpec, SurfaceForm, SurfaceSpec,
        TextureDescriptor, TextureType, TextureUsage,
    };

    /// The calls and bytes each make part added from `before` to `after`, by name.
    fn added(before: &Metrics, after: &Metrics) -> [(&'static str, u64, u64); 7] {
        let parts = |m: &Metrics| {
            [
                ("image creates", m.image_creates),
                ("image binds", m.image_binds),
                ("image view creates", m.image_view_creates),
                ("texture place lock waits", m.texture_place_lock_waits),
                ("home lock waits", m.home_lock_waits),
                ("surface allocations", m.surface_allocations),
                ("surface handles", m.surface_handles),
            ]
        };
        let (was, now) = (parts(before), parts(after));
        std::array::from_fn(|at| {
            let ((name, was), (_, now)) = (was[at], now[at]);
            (name, now.count - was.count, now.bytes - was.bytes)
        })
    }

    #[test]
    fn every_device_times_a_textures_make_by_part() {
        each_device("make_parts texture", |name, _, device| {
            let _grant = (device.grant_indices(IndexClass::Texture, 1, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut d =
                TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, 64, 64);
            d.usage = TextureUsage::SHADER_READ;
            let counters = device.counters();
            let before = counters.metrics();
            let texture =
                (device.new_texture(&d, Some(1))).unwrap_or_else(|error| panic!("{name}: {error}"));
            let added = added(&before, &counters.metrics());
            let bytes = added[0].2;
            assert!(bytes >= 64 * 64 * 4, "{name}: {added:?}");
            assert_eq!(
                added,
                [
                    ("image creates", 1, bytes),
                    ("image binds", 1, bytes),
                    ("image view creates", 1, 0),
                    ("texture place lock waits", 1, 0),
                    ("home lock waits", 1, 0),
                    ("surface allocations", 0, 0),
                    ("surface handles", 0, 0),
                ],
                "{name}"
            );
            drop(texture);
        });
    }

    #[test]
    fn every_device_times_an_exported_surfaces_make_by_part() {
        each_device("make_parts surface", |name, _, device| {
            for handle in ExportHandle::ALL {
                let exports = &device.core().exports;
                if !exports.offers(handle) || (handle.foreign() && !exports.foreign) {
                    continue;
                }
                let mut descriptor =
                    TextureDescriptor::new(TextureType::Type2D, PixelFormat::R8Unorm, 256, 64);
                descriptor.usage = TextureUsage::SHADER_READ;
                let planes = [PlaneSpec {
                    descriptor,
                    index: None,
                    rows_below: 0,
                }];
                let spec = SurfaceSpec {
                    handle,
                    pitch_multiple: 1,
                    start_multiple: 1,
                    planes: &planes,
                };
                let counters = device.counters();
                let before = counters.metrics();
                let (surface, export) = (device.new_exported_surface(&spec))
                    .unwrap_or_else(|error| panic!("{name}: {handle:?}: {error}"));
                let added = added(&before, &counters.metrics());
                println!(
                    "make_parts: {name}: {handle:?}: {:?}: {added:?}",
                    export.form
                );
                if export.form != SurfaceForm::InPlace {
                    assert_eq!((added[5].1, added[6].1), (0, 0), "{name}: {handle:?}");
                    continue;
                }
                let (size, dma) = (export.export.size, handle == ExportHandle::DmaBuf);
                let (images, handles) = if dma { (2, 3) } else { (1, 1) };
                let counts = added.map(|(part, count, _)| (part, count));
                assert_eq!(
                    counts,
                    [
                        ("image creates", images),
                        ("image binds", 1),
                        ("image view creates", 1),
                        ("texture place lock waits", 0),
                        ("home lock waits", 1),
                        ("surface allocations", 1),
                        ("surface handles", handles),
                    ],
                    "{name}: {handle:?}"
                );
                assert_eq!(
                    (added[5].2, added[6].2),
                    (size, handles * size),
                    "{name}: {handle:?}"
                );
                drop((surface, export));
            }
        });
    }
}
