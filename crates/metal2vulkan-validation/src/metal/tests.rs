use super::*;

// Metal execution needs the Metal device, which tests open only on macOS under `--cfg m2v_gpu`.
// The ordinary gate opens no device.
#[cfg(all(target_os = "macos", m2v_gpu))]
mod graphics;
#[cfg(all(target_os = "macos", m2v_gpu))]
mod imageblocks;
#[cfg(all(target_os = "macos", m2v_gpu))]
mod mesh_draws;
mod oracle_modules;
mod qualification_contracts;
#[cfg(all(target_os = "macos", m2v_gpu))]
mod shader_inputs;

/// `row` run as compiled, drawn with `companion`: these tests execute statically bounded shaders
/// directly, without admission.
#[cfg(all(target_os = "macos", m2v_gpu))]
fn as_compiled<'a>(
    row: &'a crate::source::SourceRow,
    companion: Option<&'a crate::companion::Companion>,
) -> OracleSource<'a> {
    OracleSource {
        row,
        entry_ll: &row.air_ll,
        air: platform::shader_air(row).unwrap().bitcode.into(),
        bound: OracleBound::LoopFree,
        companion,
    }
}

/// The generated companion the checker derives for `case` over `row`.
#[cfg(all(target_os = "macos", m2v_gpu))]
fn companion_of(
    row: &crate::source::SourceRow,
    case: &AuthoredCase,
) -> Option<crate::companion::Companion> {
    let bitcode = crate::check::loop_free::assemble(&row.air_ll).unwrap();
    let decoded = crate::check::loop_free::DecodedModule::from_bitcode("test", &bitcode).unwrap();
    let abi = decoded.abi().unwrap();
    crate::companion::derive(case, decoded.module(), abi, &interface_of(row, case)).unwrap()
}

/// The AIR-layer interface of `case`'s entry in `row`, for the case's constants.
#[cfg(all(target_os = "macos", m2v_gpu))]
fn interface_of(
    row: &crate::source::SourceRow,
    case: &AuthoredCase,
) -> crate::check::interface::AirInterface {
    let constants = crate::check::case_constants(case).unwrap();
    crate::check::interface::tests::of_text(&row.air_ll, &case.entry, &constants).unwrap()
}

#[cfg(all(target_os = "macos", m2v_gpu))]
fn compile_metal_fixture(name: &str) -> Vec<u8> {
    let scratch = crate::ScratchDir::new(name).unwrap();
    let source = metal2vulkan_devtools::fixtures_dir()
        .join(name)
        .with_extension("metal");
    let air = scratch.path().join("fixture.air");
    let output = crate::gpu::compile::xcrun(
        "metal",
        [
            std::ffi::OsStr::new("-c"),
            source.as_os_str(),
            "-o".as_ref(),
            air.as_os_str(),
        ],
    )
    .unwrap();
    assert!(
        output.success,
        "compile {}: {}{}",
        source.display(),
        output.stdout,
        output.stderr
    );
    std::fs::read(air).unwrap()
}
