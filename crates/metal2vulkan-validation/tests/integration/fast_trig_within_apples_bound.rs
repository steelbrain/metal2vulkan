//! Fast `sin`, `cos` and `tan` on `[-pi, pi]` keep the device's own instruction, and on a device
//! whose functions are Metal's (MoltenVK) fast `sinpi`, `cospi` and `tanpi` keep the device's `Sin`
//! and `Cos` around their exact reduction: a near-equivalent of Metal's answer, accepted by the
//! user's standing sign-off only where the device answers within Apple's fast-math bound (MSL Table
//! 8.2: `sin` and `cos` within `2^-13` absolute on `[-pi, pi]`, `sinpi` and `cospi` on `[-1, 1]`,
//! `tan` as `sin(x) * (1 / cos(x))` and `tanpi` as `tan(x * pi)`). This test holds that condition
//! to the committed evidence instead of to prose. Every committed Vulkan and MoltenVK row of the
//! fast math-library probe case (1,024 operands under Metal's default fast math; words `32t` to
//! `32t + 4` are the fast `sin`, `cos`, `tan`, and `sincos` of operand `t`, and words `32t + 26`
//! to `32t + 28` its fast `sinpi`, `cospi`, and `tanpi`) is judged by C3's bound check
//! (`c3::bound::check`) in each of those words where it differs from Metal's:
//!
//! - **Within** Apple's stated accuracy: the accepted near-equivalent.
//! - **Open**: where Apple states no accuracy. That is past `pi` (the float nearest `pi` is past
//!   it) or past 1, where the product computes Metal's own form but on MoltenVK's `sinpi`, and what
//!   differs is the model's residue; and `tan` where `cos` may be 0 within its bound, where the
//!   fast division the formula takes is a recorded near-equivalent. Neither is part of the accepted
//!   near-equivalent, and both stay open.
//! - **Outside** Apple's accuracy, or a word the check cannot judge: the test fails, since then the
//!   accepted near-equivalent does not hold on that venue.

use std::collections::{BTreeMap, BTreeSet};

use base64::Engine as _;
use metal2vulkan::device::DeviceProfile;
use metal2vulkan_validation::acceptance::derived_profile;
use metal2vulkan_validation::c3::bound::msl::{Format, Math};
use metal2vulkan_validation::c3::bound::{self, BoundedLane, Operand, Part, WordVerdict, Words};
use metal2vulkan_validation::c3::VenueProfile;
use metal2vulkan_validation::observation::CandidateObservation;
use metal2vulkan_validation::source::shard_name_for_hash;
use metal2vulkan_validation::store::{read_jsonl_if_exists, CorpusStore};

/// The fast math-library probe case, `math-library-accuracy-at-1024-operand-pairs-under-fast-math`
/// over the public fixture `trn_math_fast`.
const FAST_PROBE: &str = "f6524207c4b2f5cd7ef709315414acfd46e960e6fc38a42e23f38550bc8be9f5";

/// Thread `t` reads `x` as element `2t` of buffer 0 and writes 32 words from word `32t`.
const THREADS: usize = 1024;
const WORDS_PER_THREAD: usize = 32;

/// The eight fast trigonometric words of each thread: the word's offset among the thread's 32,
/// what it computes, its name in the report, and the edge of the domain where Table 8.2 bounds it.
const FUNCTIONS: [(usize, &str, &str, f64); 8] = [
    (0, "sin(a)", "sin", PI),
    (1, "cos(a)", "cos", PI),
    (2, "tan(a)", "tan", PI),
    (3, "sin(a)", "sincos's sine", PI),
    (4, "cos(a)", "sincos's cosine", PI),
    (26, "sinpi(a)", "sinpi", 1.0),
    (27, "cospi(a)", "cospi", 1.0),
    (28, "tanpi(a)", "tanpi", 1.0),
];

/// π, which no `float` is.
const PI: f64 = std::f64::consts::PI;

fn decode(b64: &str) -> Vec<u8> {
    base64::engine::general_purpose::STANDARD
        .decode(b64)
        .expect("a committed row's output decodes")
}

fn lanes() -> Vec<BoundedLane> {
    FUNCTIONS
        .iter()
        .map(|&(offset, computes, _, _)| BoundedLane {
            words: Words {
                first: offset,
                count: THREADS,
                stride: WORDS_PER_THREAD,
            },
            result: Part::F32,
            math: Math::Fast,
            computes: computes.into(),
            operands: vec![Operand::Buffer {
                binding: 0,
                format: Format::F32,
                first: 0,
                stride: 2,
            }],
        })
        .collect()
}

#[test]
fn fast_trig_differs_from_metal_only_within_apples_bound_or_where_it_states_none() {
    let root = metal2vulkan_devtools::validation_dir().join("corpus");
    let store = CorpusStore::new(&root);
    let case = store
        .find_case(FAST_PROBE)
        .expect("the committed cases read")
        .expect("the fast probe case is committed");
    assert_eq!(
        case.name,
        "math-library-accuracy-at-1024-operand-pairs-under-fast-math"
    );
    let lanes = lanes();
    bound::check_lanes(&lanes).expect("the lanes are well formed");

    let operands = decode(
        case.buffers
            .iter()
            .find(|buffer| buffer.binding == 0)
            .and_then(|buffer| buffer.bytes_b64.as_deref())
            .expect("buffer 0 states its bytes"),
    );
    let operand = |thread: usize| {
        let at = thread * 8;
        f32::from_le_bytes(operands[at..at + 4].try_into().unwrap())
    };

    let metals = store
        .read_metal_of_case(&case)
        .expect("the Metal rows read");
    let shard = shard_name_for_hash(&case.air_sha256).unwrap();
    let mut rows = Vec::new();
    for backend in ["vulkan", "moltenvk"] {
        let path = root.join("observations").join(backend).join(&shard);
        let shard_rows: Vec<CandidateObservation> =
            read_jsonl_if_exists(&path).expect("the committed candidate shard reads");
        rows.extend(
            shard_rows
                .into_iter()
                .filter(|row| row.case_id == FAST_PROBE),
        );
    }
    assert!(
        rows.len() >= 10,
        "only {} committed rows of the fast probe case",
        rows.len()
    );

    let mut failures = Vec::new();
    let mut report = BTreeMap::new();
    // Each row's venue profile, which the product's notes for an operation read.
    let mut profiles: BTreeMap<String, DeviceProfile> = BTreeMap::new();
    for row in &rows {
        let golden = metals
            .iter()
            .find(|metal| metal.metal_output_sha256 == row.golden_output_sha256)
            .map(|metal| decode(&metal.output_b64))
            .unwrap_or_else(|| {
                panic!(
                    "{}: no committed Metal row answers {}",
                    row.environment_id, row.golden_output_sha256
                )
            });
        let output = decode(&row.output_b64);
        let recorded: BTreeSet<_> = row
            .census
            .iter()
            .flat_map(|census| &census.modules)
            .flat_map(|module| module.precision.iter().copied())
            .collect();
        let stem = VenueProfile::recorded_stem(&row.environment)
            .unwrap_or_else(|| panic!("{}: the row records no venue profile", row.environment_id));
        let profile = profiles.entry(stem.to_owned()).or_insert_with(|| {
            let venue = VenueProfile::committed(stem).expect("the venue profile reads");
            derived_profile(venue.description).expect("the venue profile derives")
        });
        let check = bound::check(&lanes, &case.buffers, &recorded, profile, &golden, &output);
        for word in &check.words {
            let offset = word.word % WORDS_PER_THREAD;
            let Some(&(_, _, name, edge)) = FUNCTIONS.iter().find(|function| function.0 == offset)
            else {
                continue;
            };
            let x = operand(word.word / WORDS_PER_THREAD);
            let region = match (f64::from(x).abs() <= edge, edge == PI) {
                (true, true) => "on [-pi, pi]",
                (false, true) => "past pi",
                (true, false) => "on [-1, 1]",
                (false, false) => "past 1",
            };
            let class = match &word.verdict {
                WordVerdict::Within => "within",
                WordVerdict::Unstated(_) | WordVerdict::NearEquivalent(_) => "open",
                WordVerdict::Outside(_) | WordVerdict::NotCheckable(_) | WordVerdict::Uncovered => {
                    failures.push(format!(
                        "{} word {} ({name} of {x:e}, {region}): {} {}",
                        row.environment_id,
                        word.word,
                        word.verdict.label(),
                        word.verdict.detail()
                    ));
                    "failing"
                }
            };
            *report
                .entry((row.environment_id.clone(), name, class, region))
                .or_insert(0usize) += 1;
        }
    }
    for ((environment, name, class, region), count) in &report {
        eprintln!("{environment}: {name}: {count} {class} {region}");
    }
    assert!(
        failures.is_empty(),
        "{} fast trigonometric word(s) of committed rows are outside Apple's bound or not \
         checkable, so the device's instruction is not an accepted near-equivalent there:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
