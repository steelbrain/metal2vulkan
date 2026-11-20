mod build_support;

fn main() {
    // The product's source fingerprint, `cache::SOURCE_DIGEST`: what a cache of translations across
    // builds keys on beside the versions, as a runtime's shader cache does, so a build
    // of other sources never reads another build's translations, even when no version moved. It
    // asks Cargo to rerun this script when one of its inputs changes
    // (`build_support::product_inputs`).
    use sha2::Digest;
    let hash = build_support::product_hash(std::path::Path::new("."));
    println!(
        "cargo:rustc-env=M2V_SOURCE_DIGEST={}",
        hex(&hash.finalize())
    );
}

/// `bytes` as lowercase hex, two digits a byte: the form of every digest string here.
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
