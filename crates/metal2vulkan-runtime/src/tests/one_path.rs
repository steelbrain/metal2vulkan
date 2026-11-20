//! One path for every device: the crate's product sources name no vendor, device or driver
//! identity, read no environment variable and declare no policy type. Differences between devices
//! come only from Vulkan's feature, property, format, limit and memory-type queries.
//!
//! The check reads every source file under `src/` except the tests, with line comments removed,
//! so prose may still name what the code must not do.

use crate::tests::product_sources;

/// Identifiers that only an identity-keyed branch needs: Vulkan's vendor, device and driver
/// identity fields and the structures that carry the driver's identity.
const IDENTITY_NAMES: &[&str] = &[
    "vendor_id",
    "vendorID",
    "device_id",
    "deviceID",
    "driver_id",
    "driverID",
    "DriverId",
    "driver_name",
    "driverName",
    "driver_info",
    "driverInfo",
    "device_name",
    "deviceName",
    "PhysicalDeviceDriverProperties",
    "PhysicalDeviceDriverPropertiesKHR",
];

/// PCI and Khronos vendor IDs of GPU makers, as hex literals (lower case, no separators).
const VENDOR_LITERALS: &[&str] = &[
    "0x1002", "0x10de", "0x8086", "0x106b", "0x13b5", "0x5143", "0x1010", "0x14e4", "0x10005",
];

/// What reading the environment looks like in source.
const ENVIRONMENT_READS: &[&str] = &["env::var", "var_os(", "option_env!", "std::env"];

/// One finding: the file, the 1-based line and what it named.
#[derive(Debug, PartialEq, Eq)]
struct Finding {
    file: String,
    line: usize,
    what: String,
}

/// The findings in one source text. `file` only labels them.
fn scan(file: &str, text: &str) -> Vec<Finding> {
    let mut findings = Vec::new();
    for (index, raw) in text.lines().enumerate() {
        let code = raw.split("//").next().unwrap_or("");
        let mut found = |what: String| {
            findings.push(Finding {
                file: file.to_string(),
                line: index + 1,
                what,
            })
        };
        for token in code.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')) {
            if token.is_empty() {
                continue;
            }
            if IDENTITY_NAMES.contains(&token) || token.starts_with("VK_VENDOR_ID") {
                found(format!("identity name `{token}`"));
            }
            let hex = token.to_ascii_lowercase().replace('_', "");
            if let Some(stem) = VENDOR_LITERALS
                .iter()
                .find(|vendor| hex == **vendor || hex.strip_prefix(**vendor).is_some_and(is_suffix))
            {
                found(format!("vendor literal `{stem}`"));
            }
        }
        for read in ENVIRONMENT_READS {
            if code.contains(read) {
                found(format!("environment read `{read}`"));
            }
        }
        let words: Vec<&str> = code.split_whitespace().collect();
        for pair in words.windows(2) {
            if (pair[0] == "struct" || pair[0] == "enum") && pair[1].contains("Policy") {
                found(format!("policy type `{}`", pair[1]));
            }
        }
    }
    findings
}

/// Whether `rest`, after a vendor literal, is an integer type suffix (`u32`, `u64`, `i32` ...).
fn is_suffix(rest: &str) -> bool {
    matches!(
        rest,
        "u8" | "u16" | "u32" | "u64" | "usize" | "i8" | "i16" | "i32" | "i64" | "isize"
    )
}

#[test]
fn the_product_sources_take_one_path_for_every_device() {
    let findings: Vec<Finding> = product_sources()
        .iter()
        .flat_map(|(label, text)| scan(label, text))
        .collect();
    assert!(
        findings.is_empty(),
        "product code must decide by Vulkan queries alone (AGENTS.md, rule 2): {findings:#?}"
    );
}

#[test]
fn the_scan_finds_each_forbidden_form() {
    let cases = [
        (
            "if properties.vendor_id == 4098 {",
            "identity name `vendor_id`",
        ),
        ("let id = props.deviceID;", "identity name `deviceID`"),
        ("match driver.driver_id {", "identity name `driver_id`"),
        ("vk::DriverId::MESA_RADV", "identity name `DriverId`"),
        ("const AMD: u32 = 0x1002;", "vendor literal `0x1002`"),
        ("if vendor == 0x10DEu32 {", "vendor literal `0x10de`"),
        ("let intel = 0x80_86;", "vendor literal `0x8086`"),
        (
            "let x = std::env::var(\"X\");",
            "environment read `env::var`",
        ),
        ("pub enum SubmitPolicy {", "policy type `SubmitPolicy`"),
        ("VK_VENDOR_ID_MESA", "identity name `VK_VENDOR_ID_MESA`"),
    ];
    for (line, expected) in cases {
        let findings = scan("case.rs", line);
        assert!(
            findings.iter().any(|finding| finding.what == expected),
            "`{line}` should report {expected}, reported {findings:?}"
        );
    }
}

#[test]
fn the_scan_passes_prose_and_unrelated_numbers() {
    let text = "\
// No vendor_id, no 0x1002 and no env::var in a line comment.
let limit = properties.limits.max_push_constants_size;
let mask = 0x10020u32;
let count = 0x8;
";
    assert_eq!(scan("case.rs", text), Vec::new());
}
