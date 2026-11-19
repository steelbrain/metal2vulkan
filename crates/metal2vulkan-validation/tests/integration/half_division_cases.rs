//! The half-division fixture's cases (`fixtures/public/engine/grd_half_every_quotient.metal`)
//! divide every pair of `half` values once between them. The kernel's lane at `(c, r)` takes the
//! dividend `base + 64 r + c / 16`, `base` its function constant 0, and the 4096 divisors from
//! `4096 (c mod 16)`, so a grid 1024 wide covers 64 dividends a row, each by all 65536 divisors,
//! and the cases' bases and heights must tile the 65536 dividends with no gap and no overlap. The
//! dividends are cut in eighths so that each case's command buffer stays a fraction of Metal's
//! bound on a CPU device (`.config/nextest.toml`).

use base64::Engine as _;
use metal2vulkan_validation::case::{AuthoredCase, ScalarType};
use metal2vulkan_validation::store::CorpusStore;

/// The fixture the cases run, as their `public_source` names it.
const FIXTURE: &str = "engine/grd_half_every_quotient.ll";
/// The kernel's lanes a dividend, and divisors a lane.
const LANES_A_DIVIDEND: u32 = 16;
const DIVISORS_A_LANE: u64 = 4096;

#[test]
fn the_half_division_cases_divide_every_pair_of_half_values_once() {
    let root = metal2vulkan_devtools::validation_dir().join("corpus");
    let cases: Vec<AuthoredCase> = CorpusStore::new(&root)
        .read_all_cases()
        .expect("committed cases")
        .into_iter()
        .filter(|case| {
            case.public_source
                .as_deref()
                .is_some_and(|path| path.ends_with(FIXTURE))
        })
        .collect();
    let mut ranges = Vec::new();
    for case in &cases {
        let dispatch = case.dispatch.as_ref().expect("a dispatch");
        let [width, height, depth] = dispatch.grid;
        assert_eq!(
            (width, depth),
            (64 * LANES_A_DIVIDEND, 1),
            "{}: 64 dividends a row",
            case.name
        );
        let [constant] = case.function_constants.as_slice() else {
            panic!("{}: one function constant, the base", case.name);
        };
        assert_eq!(
            (constant.index, constant.scalar_type),
            (0, ScalarType::U32),
            "{}",
            case.name
        );
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(&constant.bytes_b64)
            .expect("the base's bytes");
        let base = u64::from(u32::from_le_bytes(
            bytes.try_into().expect("the base's four bytes"),
        ));
        ranges.push((base, base + 64 * u64::from(height), case.name.clone()));
    }
    ranges.sort();
    let mut next = 0;
    for (start, end, name) in &ranges {
        assert_eq!(
            *start, next,
            "{name}: a gap or an overlap before its dividends"
        );
        next = *end;
    }
    assert_eq!(
        next,
        1 << 16,
        "the cases' dividends end short of or past every half value"
    );
    let pairs: u64 = ranges
        .iter()
        .map(|(start, end, _)| (end - start) * u64::from(LANES_A_DIVIDEND) * DIVISORS_A_LANE)
        .sum();
    assert_eq!(pairs, 1 << 32);
    assert_eq!(ranges.len(), 8, "an eighth of the dividends a case");
}
