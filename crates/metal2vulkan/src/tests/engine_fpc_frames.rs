//! Metal's frame per function on the owned fixture `fpc_frames`: recursive visible functions whose
//! frames Metal sizes differently, each called by a kernel `frames_*` at the depth buffer 2 holds,
//! with the mode word buffer 3 holds.
//!
//! - Creation: Metal refuses a pipeline whose `maxCallStackDepth` `k` times its linked frames
//!   passes 261728 bytes. The largest `k` it accepts is measured on an M4 under macOS 27.0.1
//!   (plain and under API validation alike); the lowering refuses past `261728 / L`
//!   with its own linked frames `L` (frames packed as item 29 says). For eight of the fourteen
//!   kernels the bounds are Metal's. For the others Metal's backend keeps less than AIR says lives
//!   across the call (one-dimensional arrays of up to 32 words, and a pair's big frame when the
//!   kernel enters at its small one), which no structure of the AIR shows; those bounds are
//!   recorded beside Metal's.
//! - Reach: at `k` 1, 2, 3, 4 and 8 every depth Metal answered correctly (its first wrong depth
//!   measured at each) is answered as the MSL computes it: a lone function's stack holds `k`
//!   activations, so the deepest correct depth is `k - 1`, Metal's, and a pair's holds at least
//!   Metal's. Past the stack a call is not made and yields zero, which the expected values model.
//!
//! `llvm-as` assembles the fixture; a missing tool skips visibly.

use super::engine_recursion::{buffer, compile_linked, depth, slot};
use crate::device::VkLimit;
use crate::error::{Capability, Error, Feature, Limit, Unsupported};
use crate::library::FunctionConstantValues;
use crate::pipeline::compile::LinkedSource;
use crate::pipeline::testing::{
    engine_dir, llvm_assemble, profile, typed_module_linked_with, AssembleError,
};
use crate::spirv::interp::{run, Config, Resources};
use crate::spirv::ir::TypedModule;

/// The fixture assembled.
fn fixture() -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join("fpc_frames.ll")) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected fpc_frames: {why}"),
    }
}

const PAIR: &[&str] = &["fr_pa", "fr_pb"];
const QUAD: &[&str] = &["fr_qa", "fr_qb"];
const S: &[&str] = &["fr_sa", "fr_sb"];
const T: &[&str] = &["fr_ta", "fr_tb"];
const U: &[&str] = &["fr_ua", "fr_ub"];

/// Each kernel of the fixture, the functions its pipeline links, the largest `maxCallStackDepth`
/// the lowering compiles it at, and the largest Metal accepts: `261728 / L`, `L`
/// the linked frames in bytes.
///
/// - Equal: 8 values live across the call (48 bytes), 32 (144), arrays `[4][4]` (80), `[8][8]`
///   (272), `[3][2]` of `uint4` (112, packed largest alignment first; a struct of the frame's
///   slots in order is 128), a 64-word array (272), and the pairs of a 16-byte and a 272-byte
///   frame entered at the big one or by a mode (288).
/// - Metal keeps a one-dimensional array of 8, 16 or 32 words out of the frame (16 bytes where
///   ours are 48, 80, 144) and the 16-word array of the pair (32 where ours are 96).
/// - Metal reserves 544 for a pair entered only at its small frame (`fr_sa`) and for the pair of
///   a 16-word and a 64-word array, where ours are 288 and 352: it refuses from 482 a pipeline
///   the lowering compiles to 908 and 743. The stack the lowering allocates is its own, so these
///   run correctly; Metal refuses them.
const BOUNDS: [(&str, &[&str], u32, u32); 14] = [
    ("frames_live8", &["fr_live8"], 5452, 5452),
    ("frames_live32", &["fr_live32"], 1817, 1817),
    ("frames_nest44", &["fr_nest44"], 3271, 3271),
    ("frames_nest88", &["fr_nest88"], 962, 962),
    ("frames_nestv", &["fr_nestv"], 2336, 2336),
    ("frames_w8", &["fr_w8"], 5452, 16358),
    ("frames_w16", &["fr_w16"], 3271, 16358),
    ("frames_w32", &["fr_w32"], 1817, 16358),
    ("frames_w64", &["fr_w64"], 962, 962),
    ("frames_pair", PAIR, 2726, 8179),
    ("frames_quad", QUAD, 743, 481),
    ("frames_s", S, 908, 481),
    ("frames_t", T, 908, 908),
    ("frames_u", U, 908, 908),
];

/// The lowering compiles each kernel to its bound and refuses one past it, as Metal refuses past
/// its own; eight of the bounds are Metal's ([`BOUNDS`]).
#[test]
fn fpc_frames_compile_to_their_bounds_and_are_refused_past_them() {
    let Some(bitcode) = fixture() else {
        return;
    };
    let p = profile("moltenvk-m3", Default::default()).expect("profile");
    let refused = Err(Error::LimitExceeded(Limit::CallStackBytes));
    for (entry, linked, ours, metal) in BOUNDS {
        let r = compile_linked(&bitcode, entry, linked, ours, &p).map(|_| ());
        assert_ne!(r, refused, "{entry} at {ours}");
        let r = compile_linked(&bitcode, entry, linked, ours + 1, &p).map(|_| ());
        assert_eq!(r, refused, "{entry} at {} (Metal: {metal})", ours + 1);
    }
}

/// A pair's stack holds its frames uniformly, each sized for either member, as many as `k L` bytes
/// hold of the smaller: `frames_t`'s two stacks (one for each function the kernel may call) at `k`
/// 4, 72 frames of 268 bytes each, pass the 16 KiB a thread every device holds and are refused
/// `Unsupported{Recursion}`, where Metal runs them; at `k` 1 they fit.
#[test]
fn a_pair_s_uniform_frames_pass_16_kib_at_depth_4() {
    let Some(bitcode) = fixture() else {
        return;
    };
    let p = profile("nvidia-rtx5080-laptop", Default::default()).expect("profile");
    let refused = Err(Error::Unsupported(Unsupported::needs_any_of(
        Feature::Recursion,
        vec![vec![Capability::Limit {
            limit: VkLimit::FunctionStorageBytesPerThread,
            required: 2 * 72 * 268,
            available: 16 * 1024,
        }]],
    )));
    let r = compile_linked(&bitcode, "frames_t", T, 4, &p).map(|_| ());
    assert_eq!(r, refused);
    let r = compile_linked(&bitcode, "frames_t", T, 1, &p).map(|_| ());
    assert_eq!(r, Ok(()));
}

/// A function of the fixture, as its MSL computes it.
#[derive(Clone, Copy, Debug)]
enum Fr {
    /// `fr_live8`, `fr_live32`: this many hashed words live across the call.
    Live(u32),
    /// `fr_nest44`, `fr_nest88`: a square array of this side.
    Nest(u32),
    /// `fr_nestv`: a `[3][2]` array of `uint4`.
    NestV,
    /// `fr_w8` to `fr_w64`: a one-dimensional array of this many words.
    W(u32),
    /// `fr_pa` in this mode: only `n` across the call, to `fr_pb` (to itself in mode 2).
    Pa(u32),
    /// `fr_pb` in this mode: a 16-word array, to `fr_pa` in mode 0 and to itself otherwise.
    Pb(u32),
    /// `fr_qa`: a 16-word array, to `fr_qb`.
    Qa,
    /// `fr_qb`: a 64-word array, to `fr_qa`.
    Qb,
    /// `fr_sa`, `fr_ta`, `fr_ub`: only `n` across the call, to [`Fr::Big`].
    Small,
    /// `fr_sb`, `fr_tb`, `fr_ua`: a 64-word array, to [`Fr::Small`].
    Big,
}

/// `inner` plus the weighted sum of the `words`-word array `w[i] = n words + i` (`WUSE`).
fn weighted(n: u32, words: u32, inner: u32) -> u32 {
    (0..words).fold(inner, |s, j| {
        let w = n
            .wrapping_mul(words)
            .wrapping_add(j.wrapping_mul(7).wrapping_add(n) & (words - 1));
        s.wrapping_add(w.wrapping_mul(j + 1))
    })
}

/// A call of `f(n)` with `frames` activations left on the stack: the first is its own, and with
/// none left the call is not made and yields zero.
fn call(f: Fr, n: u32, frames: u32) -> u32 {
    let Some(left) = frames.checked_sub(1) else {
        return 0;
    };
    let inner = |g: Fr| if n == 0 { 0 } else { call(g, n - 1, left) };
    match f {
        Fr::Live(count) => {
            if n == 0 {
                return 0;
            }
            let mut h = n.wrapping_mul(0x9E37_79B9).wrapping_add(0x7F4A_7C15);
            let a: Vec<u32> = (0..count)
                .map(|_| {
                    h = (h ^ (h >> 13)).wrapping_mul(0x85EB_CA6B);
                    h
                })
                .collect();
            (1u32..)
                .zip(a)
                .fold(inner(f), |s, (i, x)| s.wrapping_add(x.wrapping_mul(i)))
        }
        Fr::Nest(side) => (0..side * side).fold(inner(f), |s, j| {
            let i = j.wrapping_mul(7).wrapping_add(n) & (side - 1);
            let jj = j.wrapping_mul(5).wrapping_add(n) & (side - 1);
            let w = n
                .wrapping_mul(side * side)
                .wrapping_add(i * side)
                .wrapping_add(jj);
            s.wrapping_add(w.wrapping_mul(j + 1))
        }),
        Fr::NestV => (0..24u32).fold(inner(f), |s, j| {
            let i = j.wrapping_add(n) % 3;
            let jj = j.wrapping_mul(5).wrapping_add(n) & 1;
            let c = j.wrapping_mul(3).wrapping_add(n) & 3;
            let w = n.wrapping_mul(24).wrapping_add(i * 8 + jj * 4 + c);
            s.wrapping_add(w.wrapping_mul(j + 1))
        }),
        Fr::W(words) => weighted(n, words, inner(f)),
        Fr::Pa(mode) if n > 0 => {
            let next = if mode == 2 {
                Fr::Pa(mode)
            } else {
                Fr::Pb(mode)
            };
            n.wrapping_add(inner(next))
        }
        Fr::Pb(mode) => {
            let next = if mode == 0 {
                Fr::Pa(mode)
            } else {
                Fr::Pb(mode)
            };
            weighted(n, 16, inner(next))
        }
        Fr::Qa => weighted(n, 16, inner(Fr::Qb)),
        Fr::Qb => weighted(n, 64, inner(Fr::Qa)),
        Fr::Small if n > 0 => n.wrapping_add(inner(Fr::Big)),
        Fr::Big => weighted(n, 64, inner(Fr::Small)),
        Fr::Pa(_) | Fr::Small => 0,
    }
}

/// A kernel's call in one mode: the kernel, the functions its pipeline links, the mode, the
/// function it calls, Metal's deepest correct depth at `k` (at `k` 1, 2, 3, 4, 8 and
/// 16), and the linked frames `L` and smallest member frame `s` of the lowering's stack (a lone
/// function's ratio is 1).
type Reach = (
    &'static str,
    &'static [&'static str],
    u32,
    Fr,
    fn(u32) -> u32,
    (u64, u64),
);

/// The calls the reach test makes.
fn reaches() -> [Reach; 17] {
    let lone: fn(u32) -> u32 = |k| k - 1;
    let twice: fn(u32) -> u32 = |k| 2 * k - 1;
    let four: fn(u32) -> u32 = |k| 4 * k - 1;
    [
        ("frames_live8", &["fr_live8"], 0, Fr::Live(8), lone, (1, 1)),
        (
            "frames_live32",
            &["fr_live32"],
            0,
            Fr::Live(32),
            lone,
            (1, 1),
        ),
        (
            "frames_nest44",
            &["fr_nest44"],
            0,
            Fr::Nest(4),
            lone,
            (1, 1),
        ),
        (
            "frames_nest88",
            &["fr_nest88"],
            0,
            Fr::Nest(8),
            lone,
            (1, 1),
        ),
        ("frames_nestv", &["fr_nestv"], 0, Fr::NestV, lone, (1, 1)),
        ("frames_w8", &["fr_w8"], 0, Fr::W(8), lone, (1, 1)),
        ("frames_w16", &["fr_w16"], 0, Fr::W(16), lone, (1, 1)),
        ("frames_w32", &["fr_w32"], 0, Fr::W(32), lone, (1, 1)),
        ("frames_w64", &["fr_w64"], 0, Fr::W(64), lone, (1, 1)),
        ("frames_pair", PAIR, 0, Fr::Pa(0), twice, (96, 16)),
        ("frames_pair", PAIR, 1, Fr::Pb(1), twice, (96, 16)),
        ("frames_pair", PAIR, 2, Fr::Pa(2), twice, (96, 16)),
        ("frames_quad", QUAD, 0, Fr::Qa, four, (352, 80)),
        ("frames_s", S, 0, Fr::Small, four, (288, 16)),
        ("frames_t", T, 0, Fr::Small, |k| 2 * k, (288, 16)),
        ("frames_t", T, 1, Fr::Big, twice, (288, 16)),
        ("frames_u", U, 0, Fr::Big, twice, (288, 16)),
    ]
}

/// The activations the lowering's stack holds at `k`: `k L` bytes of frames of `s`, and at least
/// `k`.
fn activations(k: u32, (l, s): (u64, u64)) -> u32 {
    let a = (u64::from(k) * l).div_ceil(s).max(u64::from(k));
    u32::try_from(a).expect("activations")
}

/// The typed module of `entry` at depth `k` with `linked` linked, on `moltenvk-m3` with a thread's
/// 192 KiB of Function storage, as the M3 Max holds: past the 16 KiB every device holds, so the
/// deep stacks compile for the interpreter.
fn typed(bitcode: &[u8], entry: &str, linked: &[&str], k: u32) -> TypedModule {
    let mut p = profile("moltenvk-m3", Default::default()).expect("profile");
    p.edit(|f| f.function_storage = crate::device::profile::FunctionStorage::Assumed(192 * 1024));
    let none = FunctionConstantValues::defaults();
    let sources: Vec<LinkedSource<'_>> = linked
        .iter()
        .map(|name| LinkedSource::new(name, bitcode, &none))
        .collect();
    typed_module_linked_with(bitcode, entry, &p, &sources, &depth(k))
        .unwrap_or_else(|e| panic!("{entry} at depth {k}: {e}"))
        .1
}

/// Run `m` with one thread per entry of `depths` (buffer 2) and the mode word `mode` (buffer 3):
/// the words it writes to buffer 0.
fn dispatch(m: &TypedModule, depths: &[u32], mode: u32) -> Vec<u32> {
    let n = depths.len();
    let res = Resources {
        buffers: vec![
            buffer(slot(0), vec![0; n * 4]),
            buffer(
                slot(2),
                depths.iter().flat_map(|d| d.to_le_bytes()).collect(),
            ),
            buffer(slot(3), mode.to_le_bytes().to_vec()),
        ],
        push: vec![0; 128],
        spec: Default::default(),
    };
    let cfg = Config {
        workgroups: [n as u32, 1, 1],
        ..Config::default()
    };
    let out = run(m, &cfg, res).unwrap_or_else(|e| panic!("{e:?}"));
    out.buffers[0]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Every depth Metal answered correctly is answered as the MSL computes it, at `k` 1, 2, 3, 4 and
/// 8: a lone function reaches `k - 1`, Metal's, and the pairs at least Metal's reach. Around the
/// lowering's own stack each answer is the modelled one, a call past it yielding zero.
#[test]
fn fpc_frames_reach_every_depth_metal_answers() {
    let Some(bitcode) = fixture() else {
        return;
    };
    for (entry, linked, mode, root, metal, frames) in reaches() {
        for k in [1, 2, 3, 4, 8] {
            let held = activations(k, frames);
            let reach = metal(k);
            let mut ds: Vec<u32> = (0..4)
                .chain(reach.saturating_sub(1)..=reach + 1)
                .chain(held.saturating_sub(1)..=held + 1)
                .collect();
            ds.sort_unstable();
            ds.dedup();
            let m = typed(&bitcode, entry, linked, k);
            let got = dispatch(&m, &ds, mode);
            let want: Vec<u32> = ds.iter().map(|&d| call(root, d, held)).collect();
            let at = format!("{entry} mode {mode} at depth {k}: {ds:?}");
            assert_eq!(got, want, "{at}");
            for (&d, &g) in ds.iter().zip(&got) {
                if d <= reach {
                    assert_eq!(g, call(root, d, u32::MAX), "{at}: depth {d}");
                }
            }
        }
    }
}
