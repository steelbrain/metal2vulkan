//! The Direct3D 11 surface import, a Windows host's: under a device cfg, on every device of the
//! cfg's kind, refused typed for now, for no plane and for a plane.

#[cfg(all(windows, any(m2v_gpu, m2v_cpu_icd)))]
mod on_device {
    use crate::tests::command::on_device::each_device;
    use crate::{D3d11Plane, Error, Gap, PixelFormat, TextureDescriptor, TextureType};
    use std::os::windows::io::AsHandle;

    #[test]
    fn every_device_refuses_a_d3d11_surface_typed() {
        each_device("import_d3d11_surface", |name, _, device| {
            // A handle the call borrows and never reads: the test's own executable's.
            let file = std::env::current_exe()
                .and_then(std::fs::File::open)
                .expect("a file");
            let plane = D3d11Plane {
                handle: file.as_handle(),
                image_format: PixelFormat::R8Unorm,
                descriptor: TextureDescriptor::new(
                    TextureType::Type2D,
                    PixelFormat::R8Unorm,
                    64,
                    32,
                ),
                index: None,
            };
            for planes in [&[][..], &[plane][..]] {
                let error = device.import_d3d11_surface(planes).err();
                assert!(
                    matches!(error, Some(Error::Unsupported(Gap::D3d11Surface))),
                    "{name}: {} planes: {error:?}",
                    planes.len()
                );
            }
            println!("d3d11: {name}: the surface import refused typed");
        });
    }
}
