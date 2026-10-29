//! No crate-owned compute module divides the all-ones integer of its width (`2^32 - 1` or
//! `2^64 - 1`): that division is the overflow test (`b <= (2^32 - 1) / a`) NVIDIA's compiler
//! refuses in some modules, and the modules test a product's overflow by its high word instead
//! (`bind::replay::checked_mul`). The modules are every compute module function of `bind`, at
//! every value of each closed parameter (every mipmap kernel a chain may dispatch, every
//! tessellation scan of each domain, partitioning, winding, step and scaling), each built on
//! every committed profile: one a profile refuses (`Unsupported`) is skipped there, and each
//! function builds on one profile at least.

use std::collections::BTreeSet;
use std::sync::Arc;

use crate::bind::{
    self, DepthStandInForm, PlanesCopy, RewritePhase, RewriteProgram, TessellationScanKernel,
};
use crate::device::{Device, DevicePolicy};
use crate::error::{Error, Feature};
use crate::metal::format::ExtendedRange;
use crate::metal::{
    TessellationFactorStepFunction as Step, TessellationPartitionMode as Partition, Winding,
};
use crate::pipeline::interface::PatchDomain;
use crate::spirv::testing::{all_ones_divisions, committed_descriptions};

/// A module function at one parameter: what it is called, and its words on a device.
type Build = (
    String,
    Box<dyn Fn(&Device) -> Result<Arc<[u32]>, Error> + Sync>,
);

/// Every value of `ExtendedRange`.
const EXTENDED: [ExtendedRange; 13] = [
    ExtendedRange::Bgr10 { srgb: false },
    ExtendedRange::Bgr10 { srgb: true },
    ExtendedRange::Bgra10 { srgb: false },
    ExtendedRange::Bgra10 { srgb: true },
    ExtendedRange::Planes { srgb: false },
    ExtendedRange::Planes { srgb: true },
    ExtendedRange::R10X6,
    ExtendedRange::Rg10X12,
    ExtendedRange::R10X6Srgb,
    ExtendedRange::Rg10X12Srgb,
    ExtendedRange::R16Bfloat,
    ExtendedRange::Rg16Bfloat,
    ExtendedRange::Rgba16Bfloat,
];

/// Both forms of a shadow merge's stores.
const STORES: [bind::ShadowMergeStores; 2] = [
    bind::ShadowMergeStores::Words,
    bind::ShadowMergeStores::Blocks,
];

fn modules() -> Vec<Build> {
    let mut out: Vec<Build> = Vec::new();
    macro_rules! plain {
        ($($f:ident),* $(,)?) => {
            $(out.push((stringify!($f).into(), Box::new(|d| Ok(bind::$f(d)?.spirv))));)*
        };
    }
    plain!(
        flat_dispatch_module,
        index_bounds_fill_module,
        index_bounds_plan_module,
        indirect_clamp_module,
        mesh_object_arguments_module,
        mesh_scan_module,
        plane_bytes_copy_module,
        plane_copy_module,
        plane_words_module,
        predication_step_module,
        replay_preprocess_module,
        rotate_texels_module,
        rows_copy_module,
        tiled_texels_module,
        two_plane_texels_module,
        visibility_combine_module,
        visibility_resolve_module,
        ycbcr_refresh_module,
        ycbcr_write_module,
    );
    let mut push =
        |name: String, build: Box<dyn Fn(&Device) -> Result<Arc<[u32]>, Error> + Sync>| {
            out.push((name, build));
        };
    for stores in STORES {
        push(
            format!("shadow_merge_module({stores:?})"),
            Box::new(move |d| Ok(bind::shadow_merge_module(d, stores)?.spirv)),
        );
    }
    for form in EXTENDED {
        push(
            format!("extended_shadow_module({form:?})"),
            Box::new(move |d| Ok(bind::extended_shadow_module(d, form)?.spirv)),
        );
    }
    for program in RewriteProgram::ALL {
        for phase in RewritePhase::ALL {
            push(
                format!("index_rewrite_module({program:?}, {phase:?})"),
                Box::new(move |d| Ok(bind::index_rewrite_module(d, program, phase)?.spirv)),
            );
        }
    }
    for view in bind::ShadowTexelView::ALL {
        push(
            format!("shadow_copy_back_module({view:?})"),
            Box::new(move |d| Ok(bind::shadow_copy_back_module(d, view)?.spirv)),
        );
        push(
            format!("shadow_copy_in_module({view:?})"),
            Box::new(move |d| Ok(bind::shadow_copy_in_module(d, view)?.spirv)),
        );
        for stores in STORES {
            push(
                format!("shadow_merge_kept_image_module({view:?}, {stores:?})"),
                Box::new(move |d| Ok(bind::shadow_merge_kept_image_module(d, view, stores)?.spirv)),
            );
        }
    }
    for read in bind::ShadowMergeRead::ALL {
        for stores in STORES {
            push(
                format!("shadow_merge_image_module({read:?}, {stores:?})"),
                Box::new(move |d| Ok(bind::shadow_merge_image_module(d, read, stores)?.spirv)),
            );
        }
    }
    for conversion in crate::bind::ycbcr_convert::conversions() {
        push(
            format!("ycbcr_convert_module({conversion:?})"),
            Box::new(move |d| Ok(bind::ycbcr_convert_module(d, conversion)?.spirv)),
        );
        push(
            format!("ycbcr_convert_floats_module({conversion:?})"),
            Box::new(move |d| Ok(bind::ycbcr_convert_floats_module(d, conversion)?.spirv)),
        );
    }
    for copy in PlanesCopy::ALL {
        push(
            format!("planes_split_module({copy:?})"),
            Box::new(move |d| Ok(bind::planes_split_module(d, copy)?.spirv)),
        );
        for samples in [2, 4, 8] {
            push(
                format!("interleave_split_module({copy:?}, {samples})"),
                Box::new(move |d| Ok(bind::interleave_split_module(d, copy, samples)?.spirv)),
            );
        }
    }
    for form in [
        bind::MeshArgumentsForm::Mesh,
        bind::MeshArgumentsForm::Object,
    ] {
        push(
            format!("mesh_arguments_module({form:?})"),
            Box::new(move |d| Ok(bind::mesh_arguments_module(d, form)?.spirv)),
        );
    }
    push(
        "stencil_split_module".into(),
        Box::new(|d| Ok(bind::stencil_split_module(d, Feature::Format)?.spirv)),
    );
    for form in [
        DepthStandInForm::Multisampled,
        DepthStandInForm::SamplePlanes,
    ] {
        push(
            format!("depth_split_module({form:?})"),
            Box::new(move |d| Ok(bind::depth_split_module(d, form, Feature::Format)?.spirv)),
        );
    }
    for kernel in bind::mipmap_filter::tests::every_kernel() {
        push(
            format!("mipmap_filter_module({kernel:?})"),
            Box::new(move |d| Ok(bind::mipmap_filter_module(d, kernel)?.spirv)),
        );
    }
    for domain in [PatchDomain::Triangle, PatchDomain::Quad] {
        for &partition_mode in Partition::ALL {
            for &output_winding_order in Winding::ALL {
                for &factor_step_function in Step::ALL {
                    for factor_scale_enabled in [false, true] {
                        let kernel = TessellationScanKernel {
                            domain,
                            partition_mode,
                            output_winding_order,
                            max_tessellation_factor: 64,
                            factor_step_function,
                            factor_scale_enabled,
                        };
                        push(
                            format!("tessellation_scan_module({kernel:?})"),
                            Box::new(
                                move |d| Ok(bind::tessellation_scan_module(d, &kernel)?.spirv),
                            ),
                        );
                    }
                }
            }
        }
    }
    out
}

/// The function a build's name calls.
fn function(name: &str) -> &str {
    name.split('(').next().unwrap_or(name)
}

#[test]
fn no_crate_owned_compute_module_divides_the_all_ones_integer() {
    let devices: Vec<(String, Device)> = committed_descriptions()
        .expect("profiles")
        .iter()
        .map(|(stem, desc)| {
            let device = Device::new(desc.clone(), DevicePolicy::default()).expect(stem);
            (stem.clone(), device)
        })
        .collect();
    let modules = modules();
    let jobs: Vec<(usize, usize)> = (0..modules.len())
        .flat_map(|m| (0..devices.len()).map(move |d| (m, d)))
        .collect();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let results: Vec<(usize, Result<bool, String>)> = std::thread::scope(|scope| {
        let handles: Vec<_> = jobs
            .chunks(jobs.len().div_ceil(threads))
            .map(|chunk| {
                let (modules, devices) = (&modules, &devices);
                scope.spawn(move || {
                    let mut out = Vec::new();
                    for &(m, d) in chunk {
                        let (name, build) = &modules[m];
                        let (stem, device) = &devices[d];
                        let result = match build(device) {
                            Ok(words) => match all_ones_divisions(&words) {
                                0 => Ok(true),
                                n => Err(format!("{name} on {stem}: {n} all-ones divisions")),
                            },
                            Err(Error::Unsupported(_)) => Ok(false),
                            Err(e) => Err(format!("{name} on {stem}: {e}")),
                        };
                        out.push((m, result));
                    }
                    out
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("a checking thread"))
            .collect()
    });
    let failures: Vec<&String> = results
        .iter()
        .filter_map(|(_, r)| r.as_ref().err())
        .collect();
    assert!(failures.is_empty(), "{failures:#?}");
    let built: BTreeSet<&str> = results
        .iter()
        .filter(|(_, r)| matches!(r, Ok(true)))
        .map(|(m, _)| function(&modules[*m].0))
        .collect();
    let named: BTreeSet<&str> = modules.iter().map(|(name, _)| function(name)).collect();
    let unbuilt: Vec<&&str> = named.difference(&built).collect();
    assert!(
        unbuilt.is_empty(),
        "built on no committed profile: {unbuilt:?}"
    );
}
