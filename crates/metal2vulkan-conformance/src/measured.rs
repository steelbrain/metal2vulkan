//! The Metal measurements this tree already holds, as a case reads them.
//!
//! A probe's values live in two places: a literal table, which a case restates, and, where a table
//! was too large to write out, a committed text beside the family's fixtures
//! (`fixtures/public/<family>/`). Those texts are our own measurements over our own fixtures,
//! written by our own probes, and reaching them is what the boundary allows: the harvested corpus,
//! the stored observations and the censuses are a different thing and no case touches them.
//!
//! A case reads such a file through [`public_fixture`], which resolves from
//! `metal2vulkan_devtools`, and pins its SHA-256 in its own input fingerprint. So the case says
//! which version of the measurement its recording stands on, and a file that changes makes the
//! case [`Unrecorded`](crate::verdict::Verdict::Unrecorded) rather than quietly comparing against
//! other bytes.

/// The committed measurement `path` under the public fixtures, as text.
pub fn public_fixture(path: &str) -> Result<String, String> {
    let file = metal2vulkan_devtools::fixtures_dir().join(path);
    std::fs::read_to_string(&file).map_err(|error| format!("{}: {error}", file.display()))
}

/// Bytes from a run of hexadecimal digits, two to a byte.
pub fn hex_bytes(text: &str) -> Result<Vec<u8>, String> {
    if !text.len().is_multiple_of(2) {
        return Err(format!("{text}: an odd number of hexadecimal digits"));
    }
    (0..text.len())
        .step_by(2)
        .map(|at| {
            u8::from_str_radix(&text[at..at + 2], 16)
                .map_err(|error| format!("{}: {error}", &text[at..at + 2]))
        })
        .collect()
}

/// `length` bytes, byte `i` of them `i * 37 mod 251`: the stripes our texture-buffer probes fill
/// over.
pub fn stripes(length: usize) -> Vec<u8> {
    (0..length).map(|i| (i * 37 % 251) as u8).collect()
}
