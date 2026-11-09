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
        effect: "base directory for FALLBACK repro bundles",
    },
    EnvVar {
        name: "METAL2VULKAN_<TOOL>",
        default: "PATH search",
        effect: "absolute per-tool override (for example METAL2VULKAN_LLVM_AS or \
                 METAL2VULKAN_SPIRV_VAL)",
    },
    EnvVar {
        name: "METAL2VULKAN_LANE_VALIDATION",
        default: "unset (the validation layer's own checks)",
        effect: "validation's runtime lane only, diagnostics: the validation layer's extra \
                 checks every lane device enables, comma-separated (`sync`: synchronization \
                 validation with shader accesses; `load-store`: load operations after earlier \
                 stores)",
    },
    EnvVar {
        name: "METAL2VULKAN_LOG_PIPELINES",
        default: "unset (off)",
        effect: "metal2vulkan-runtime's diagnostics: one line per render VkPipeline a device \
                 makes, through the device's message sink or on stderr: its state, its key, the \
                 key's fields that differ from the state's closest earlier pipeline, and the \
                 creation's time, split by the making's parts (the digests, each library made \
                 with its record loaded or missed or another caller's, the link) and by \
                 MoltenVK's activities where its performance tracking is on \
                 (MVK_CONFIG_PERFORMANCE_TRACKING=1); and once per reason, why a draw \
                 that reads its own pass's color attachment after a barrier took the split rather \
                 than a snapshot, and how many such draws the snapshot's area rule weighed at each \
                 share of their colors' pixels and texel bytes; set to anything but empty or 0",
    },
    EnvVar {
        name: "METAL2VULKAN_LOG_MEMORY",
        default: "unset (off)",
        effect: "metal2vulkan-runtime's diagnostics: one line per VkDeviceMemory a device \
                 allocates, imports or frees, through the device's message sink or on stderr: \
                 its kind, handle, memory type, bytes, host address (an import's) and device \
                 address range (a block of buffers that take device addresses), to place a GPU \
                 fault's address; and one line per placement of \
                 4 MiB or more in a pooled block as it is placed and returned: image or buffer, \
                 its memory class (placed), bytes, offset and block, to show what makes and \
                 frees blocks; set to anything but empty or 0",
    },
    EnvVar {
        name: "METAL2VULKAN_PENDING_AUDIT",
        default: "unset (off)",
        effect: "metal2vulkan-runtime's diagnostics: each device checks, at each \
                 vkBeginCommandBuffer, vkResetCommandPool and command pool given back to its \
                 spares, and at each submission of a command buffer submitted before, that the \
                 GPU completed every submission carrying the command buffers concerned (a \
                 secondary through each primary that executes it), reading those submissions' \
                 timelines with vkGetSemaphoreCounterValue; it logs each one found pending, \
                 through the device's message sink or on stderr, with the thread's backtrace for \
                 the first 8, and changes no work. Costs when set: a lock of the device's ledger \
                 at each of those calls and at each vkCmdExecuteCommands, an entry per command \
                 buffer per submission until a check finds it completed, one \
                 vkGetSemaphoreCounterValue per carrying timeline at each check of a command \
                 buffer still in the ledger; set to anything but empty or 0",
    },
    EnvVar {
        name: "METAL2VULKAN_LOG_SHADOWS",
        default: "unset (off)",
        effect: "metal2vulkan-runtime's diagnostics: one line per shadow a device makes of a \
                 texture (over a buffer's bytes, an IOSurface plane's among them; a usage \
                 shadow; a rotated, two-plane, tiled depth or YCbCr texture's), through the \
                 device's message sink or on stderr: the call, the texture's format, size, \
                 levels and usage, its truth (the buffer's storage mode, length and object ID, \
                 the texels' offset and row pitch, or its own image), and what made it a \
                 shadow; set to anything but empty or 0",
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

/// The base directory for the binary's FALLBACK repro bundles.
pub fn repro_dir() -> Option<OsString> {
    std::env::var_os("METAL2VULKAN_REPRO_DIR")
}

/// The validation layer's extra checks validation's runtime lane enables on every device it makes,
/// as set: a comma-separated list the lane parses and refuses when it names an unknown check (or
/// is not Unicode). A default-off diagnostic of the lane's test devices; no product path reads it.
pub fn lane_validation() -> Option<OsString> {
    std::env::var_os("METAL2VULKAN_LANE_VALIDATION")
}

/// Whether metal2vulkan-runtime logs each render `VkPipeline` it makes (`REGISTRY`): set, and
/// neither empty nor `0`. A default-off diagnostic: it changes no pipeline, only what is printed.
pub fn log_pipelines() -> bool {
    std::env::var_os("METAL2VULKAN_LOG_PIPELINES")
        .is_some_and(|value| !value.is_empty() && value != "0")
}

/// Whether metal2vulkan-runtime logs each `VkDeviceMemory` a device allocates, imports or frees,
/// and each placement of 4 MiB or more in a pooled block (`REGISTRY`): set, and neither empty nor
/// `0`. A default-off diagnostic: it changes no
/// allocation, only what is printed.
pub fn log_memory() -> bool {
    std::env::var_os("METAL2VULKAN_LOG_MEMORY")
        .is_some_and(|value| !value.is_empty() && value != "0")
}

/// Whether metal2vulkan-runtime logs each shadow a device makes of a texture (`REGISTRY`): set,
/// and neither empty nor `0`. A default-off diagnostic: it changes no
/// texture, only what is printed.
pub fn log_shadows() -> bool {
    std::env::var_os("METAL2VULKAN_LOG_SHADOWS")
        .is_some_and(|value| !value.is_empty() && value != "0")
}

/// Whether each metal2vulkan-runtime device runs its pending audit (`REGISTRY`): set, and neither
/// empty nor `0`. A default-off diagnostic: it reads timelines and prints, and changes no work.
pub fn pending_audit() -> bool {
    std::env::var_os("METAL2VULKAN_PENDING_AUDIT")
        .is_some_and(|value| !value.is_empty() && value != "0")
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
