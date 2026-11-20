//! Registry and accessors for the operational `METAL2VULKAN_*` variables.
//!
//! Every supported variable has a typed accessor and a registry entry used by the `metal2vulkan`
//! binary's `--help`. Paths retain their raw `OsString`. Accessors read the environment when
//! called. No variable here changes what the library compiles.

use std::ffi::OsString;

/// One row in the documented registry. `default`/`effect` are for `--help` only.
pub struct EnvVar {
    pub name: &'static str,
    pub default: &'static str,
    pub effect: &'static str,
}

/// Every `METAL2VULKAN_*` var, for `--help`. Order is stable so the help text is deterministic.
pub const REGISTRY: &[EnvVar] = &[
    EnvVar {
        name: "METAL2VULKAN_REPRO_DIR",
        default: "$TMPDIR/metal2vulkan-repros",
        effect: "base directory for FALLBACK repro bundles: the input, the device file, the \
                 request's options and descriptor files, repro.sh and failure.txt, in a directory \
                 named by its content, which replays the failure with this binary; the binary \
                 writes one for each failure after it read its inputs",
    },
    EnvVar {
        name: "METAL2VULKAN_<TOOL>",
        default: "PATH search",
        effect: "absolute per-tool override (for example METAL2VULKAN_LLVM_AS or \
                 METAL2VULKAN_SPIRV_VAL)",
    },
];

/// Multi-line `--help` block listing every registry var.
pub fn help_text() -> String {
    let mut out = String::from("environment variables:\n");
    for v in REGISTRY {
        out.push_str(&format!(
            "  {:<32} {} [default: {}]\n",
            v.name, v.effect, v.default
        ));
    }
    out
}

/// The base directory for the binary's FALLBACK repro bundles (`REGISTRY`), as set; the binary
/// writes under `$TMPDIR/metal2vulkan-repros` where unset. It writes files, and changes no
/// translation and no answer.
pub fn repro_dir() -> Option<OsString> {
    std::env::var_os("METAL2VULKAN_REPRO_DIR")
}

/// Optional path override for an external tool `cmd` (`llvm-as`, `spirv-val`, …), read from
/// `METAL2VULKAN_<CMD>` (uppercased, `-`→`_`). A dynamic family — one var per tool the binary or
/// the crate's test support runs — documented in `REGISTRY` as a single `METAL2VULKAN_<TOOL>`
/// family row rather than one entry per tool. Absent → the binary searches `PATH`, and the test
/// support searches the known LLVM install directories, then `PATH`.
pub fn tool_path_override(cmd: &str) -> Option<OsString> {
    std::env::var_os(format!(
        "METAL2VULKAN_{}",
        cmd.replace('-', "_").to_ascii_uppercase()
    ))
}
