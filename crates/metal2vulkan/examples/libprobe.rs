//! Probe (never landed): list a metallib's kernels and write, for each, its bitcode and a device
//! file per profile under the runtime's identity policy.
//!
//! usage: libprobe LIB.metallib OUTDIR PROFILE.json...
//! writes OUTDIR/kernels.tsv (index, name hash, name), OUTDIR/kNNN.bc, OUTDIR/<stem>.device.json

use metal2vulkan::device::{DeviceDescription, DevicePolicy, HandleDomain, PointerDomain};
use metal2vulkan::repro::ReproDevice;
use metal2vulkan_air::container::{self, FunctionStage, Library};
use std::fmt::Write as _;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let lib = std::fs::read(&args[1]).expect("read library");
    let out = std::path::Path::new(&args[2]);
    std::fs::create_dir_all(out).expect("outdir");
    let Library::Metallib(m) = container::parse(&lib).expect("parse") else {
        panic!("not a metallib");
    };
    let mut tsv = String::new();
    let mut index = 0;
    for f in &m.functions {
        if !matches!(f.stage, FunctionStage::Kernel) {
            continue;
        }
        let hash = container::sha256(f.name.as_bytes());
        let short: String = hash[..4].iter().map(|b| format!("{b:02x}")).collect();
        writeln!(tsv, "{index}\t{short}\t{}", f.name).unwrap();
        std::fs::write(out.join(format!("k{index:03}.bc")), f.bitcode).expect("write bc");
        index += 1;
    }
    std::fs::write(out.join("kernels.tsv"), tsv).expect("write tsv");
    eprintln!("kernels: {index} of {} functions", m.functions.len());
    for p in &args[3..] {
        let json = std::fs::read_to_string(p).expect("read profile");
        let description = DeviceDescription::from_khronos_json(&json, None).expect("profile");
        let mut policy = DevicePolicy::default();
        policy.pointer_domain = PointerDomain::Identity;
        policy.handle_domain = HandleDomain::Identity;
        let device = ReproDevice::new(description, policy);
        let stem = std::path::Path::new(p).file_stem().unwrap().to_string_lossy().into_owned();
        std::fs::write(out.join(format!("{stem}.device.json")), device.to_json().unwrap())
            .expect("write device");
    }
}
