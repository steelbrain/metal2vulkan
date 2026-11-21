mod candidate_batch;
mod candidate_common;

use metal2vulkan_validation::gpu::DeviceType;
use metal2vulkan_validation::observation::Backend;
use std::path::PathBuf;

fn main() {
    // Every pipeline this process or a run it starts creates compiles cold, so the ceiling times a
    // compile, not a hit in the driver's disk cache.
    metal2vulkan_validation::budgets::turn_driver_shader_caches_off();
    // The batch modes run this executable once per case, --capture-formats writes a venue
    // profile's format supplement, and --probe-runtime measures the runtime's device facts; every
    // other invocation, the workers' included, is the single-case executor.
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("--capture-formats") => {
            if let Err(error) = capture_formats(args.collect()) {
                eprintln!("corpus-vulkan --capture-formats: {error}");
                std::process::exit(1);
            }
            return;
        }
        Some("--probe-runtime") => {
            if let Err(error) = probe_runtime(args.collect()) {
                eprintln!("corpus-vulkan --probe-runtime: {error}");
                std::process::exit(1);
            }
            return;
        }
        _ => {}
    }
    if !candidate_batch::serve(Backend::Vulkan) {
        candidate_common::main(Backend::Vulkan);
    }
}

const PROBE_USAGE: &str = "usage: corpus-vulkan --probe-runtime OUT.json --device-type cpu|gpu\n\
Measures, on the device a venue run on this host selects, the facts the Metal-like runtime's\n\
design takes from each device: host memory imported without a copy\n\
(VK_EXT_external_memory_host) and used by a dispatch, and linear images over a buffer's memory\n\
(row pitches, aliasing, explicit layouts through VK_EXT_image_drm_format_modifier), and whether a\n\
fragment's FragCoord.z, clamped to the viewport's range, is the depth the device stores (bias,\n\
clamp and range cases, one sample and per sample, the depth attachment in the\n\
subpass, out of it, or in it with its test off). Writes the JSON report to OUT.json and exits 1\n\
when the validation layer reported anything.";

/// `--probe-runtime OUT --device-type cpu|gpu`.
fn probe_runtime(args: Vec<String>) -> Result<(), String> {
    let (out, device_type) = out_and_device_type(args, PROBE_USAGE)?;
    let report = metal2vulkan_validation::candidate::probe_runtime(device_type)?;
    let text = serde_json::to_string_pretty(&report)
        .map_err(|error| format!("serialize the report: {error}"))?
        + "\n";
    std::fs::write(&out, &text).map_err(|error| format!("write {}: {error}", out.display()))?;
    let messages = report["validation_messages"].as_u64().unwrap_or(u64::MAX);
    println!(
        "probed {}: device {}, {messages} validation message(s)",
        out.display(),
        report["device"]
    );
    if messages != 0 {
        return Err(format!(
            "the validation layer reported {messages} message(s): the report is not evidence"
        ));
    }
    Ok(())
}

/// The `OUT` and `--device-type` arguments of a device capture.
fn out_and_device_type(args: Vec<String>, usage: &str) -> Result<(PathBuf, DeviceType), String> {
    let mut out = None;
    let mut device_type = None;
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--device-type" => {
                let value = args.next().ok_or("--device-type requires a value")?;
                device_type = Some(DeviceType::parse(&value)?);
            }
            "-h" | "--help" => {
                println!("{usage}");
                std::process::exit(0);
            }
            _ if out.is_none() && !arg.starts_with('-') => out = Some(PathBuf::from(arg)),
            _ => return Err(format!("unknown argument {arg:?}\n{usage}")),
        }
    }
    let out = out.ok_or_else(|| format!("OUT is required\n{usage}"))?;
    let device_type = device_type.ok_or_else(|| format!("--device-type is required\n{usage}"))?;
    Ok((out, device_type))
}

const CAPTURE_USAGE: &str =
    "usage: corpus-vulkan --capture-formats OUT.json --device-type cpu|gpu\n\
Writes the VkFormatProperties3 supplement of a venue profile's `vulkaninfo --json` capture:\n\
the identity of the device a venue run on this host selects, and the three\n\
VkFormatFeatureFlags2 masks of every format the pinned registry names that it exposes, for\n\
scripts/regen-device-profiles --formats NAME=OUT.json. Physical-device queries only.";

/// `--capture-formats OUT --device-type cpu|gpu`.
fn capture_formats(args: Vec<String>) -> Result<(), String> {
    let (out, device_type) = out_and_device_type(args, CAPTURE_USAGE)?;
    let captured = metal2vulkan_validation::candidate::capture_formats(device_type)?;
    let text = serde_json::to_string_pretty(&captured)
        .map_err(|error| format!("serialize the capture: {error}"))?
        + "\n";
    std::fs::write(&out, &text).map_err(|error| format!("write {}: {error}", out.display()))?;
    let formats = captured["formats"]
        .as_object()
        .map_or(0, serde_json::Map::len);
    let supported = captured["formats"].as_object().map_or(0, |formats| {
        formats
            .values()
            .filter(|masks| {
                masks
                    .as_object()
                    .is_some_and(|masks| masks.values().any(|mask| mask.as_u64() != Some(0)))
            })
            .count()
    });
    println!(
        "captured {}: {formats} formats queried, {supported} with a feature, device {}",
        out.display(),
        captured["device"]
    );
    Ok(())
}
