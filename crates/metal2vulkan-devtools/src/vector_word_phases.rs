//! The owned fixture `vector_word_phases`: `uint4` and `float4` accesses at each word phase past a
//! 4-aligned bound base, and through a 16-aligned `float4` buffer ([`WORDS`], [`VEC16`]; the
//! fixture's source says what each thread does). Its inputs ([`inputs`]), the bind offsets a run
//! takes ([`OFFSETS`], [`vec16_offsets`]), and the four buffers' bytes afterwards where every
//! access lands at its own address ([`answer`]), which is where Metal places it: the
//! conformance crate's `vector-word-phases-*` cases record Metal's answer on an Apple M3 Max and
//! hold it equal to [`answer`], the product's CPU-interpreter test and the new runtime's device
//! test hold theirs equal to it too.

/// The fixture.
pub const FIXTURE: &str = "vector_word_phases";
/// The kernel over two `uint` buffers bound at a multiple of 4.
pub const WORDS: &str = "vwp_words";
/// The kernel over two `float4` buffers bound at a multiple of 16.
pub const VEC16: &str = "vwp_vec16";
/// Threads of each kernel's grid, in one threadgroup.
pub const THREADS: u32 = 64;
/// The bind offset of both of [`WORDS`]'s buffers in each run: each word phase.
pub const OFFSETS: [u64; 4] = [0, 4, 8, 12];
/// Bytes of each of [`WORDS`]'s buffers, `src` and `dst`.
pub const WORDS_BYTES: usize = 8192;
/// Bytes of each of [`VEC16`]'s buffers, `a` and `b`.
pub const VEC16_BYTES: usize = 4096;

/// The words between two phases' regions of `dst` (the fixture's `PHASE`).
const PHASE: usize = 65;
/// A `float`'s sign bit, which negation flips.
const SIGN: u32 = 0x8000_0000;
/// The bits of a `float` but its exponent, the part of each `float4` update's key it keeps.
const NOT_EXPONENT: u32 = 0x807f_ffff;

/// The bind offsets of [`VEC16`]'s `a` and `b` in the run that binds [`WORDS`]'s at `offset`:
/// multiples of 16, as Metal binds a `float4` buffer, that differ from run to run.
pub fn vec16_offsets(offset: u64) -> (u64, u64) {
    (16 + 4 * offset, 64 - 4 * offset)
}

/// The four buffers' bytes: the inputs before a run, or what a run leaves.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Buffers {
    pub src: Vec<u8>,
    pub dst: Vec<u8>,
    pub a: Vec<u8>,
    pub b: Vec<u8>,
}

/// The buffers before every run: words that are each a normal `float` (an exponent field neither
/// 0 nor 255), from a fixed seed, so every `float` the kernels move is one and no device may
/// quiet, flush or canonicalize it.
pub fn inputs() -> Buffers {
    let mut state = 0x5eed_0f0a_11a1_9e57u64;
    let mut bytes = |n: usize| -> Vec<u8> {
        (0..n / 4)
            .flat_map(|_| {
                state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                z ^= z >> 31;
                let exponent = 1 + (z >> 32) % 254;
                let word = (z as u32 & NOT_EXPONENT) | (exponent as u32) << 23;
                word.to_le_bytes()
            })
            .collect()
    };
    Buffers {
        src: bytes(WORDS_BYTES),
        dst: bytes(WORDS_BYTES),
        a: bytes(VEC16_BYTES),
        b: bytes(VEC16_BYTES),
    }
}

/// The buffers after [`WORDS`] runs over [`THREADS`] threads with `src` and `dst` bound at
/// `offset`, then [`VEC16`] with `a` and `b` bound at [`vec16_offsets`] of it, from [`inputs`],
/// every access at its own address.
pub fn answer(offset: u64) -> Buffers {
    let mut out = inputs();
    let base = usize::try_from(offset / 4).expect("an offset in a usize");
    let src = words(&out.src);
    let mut dst = words(&out.dst);
    for t in 0..THREADS as usize {
        let (p, i) = (t % 4, t / 4);
        for j in 0..4 {
            dst[base + 4 * t + j] = src[base + p + 4 * i + j];
            dst[base + 256 + 4 * t + j] = src[base + 256 + p + 4 * i + j] ^ SIGN;
            dst[base + 512 + PHASE * p + 4 * i + j] = src[base + 512 + 4 * t + j];
            dst[base + 772 + PHASE * p + 4 * i + j] = src[base + 768 + 4 * t + j] ^ SIGN;
            dst[base + 1032 + PHASE * p + 4 * i + j] ^= src[base + 1024 + 4 * t + j];
            let key = src[base + 1280 + 4 * t + j] & NOT_EXPONENT;
            dst[base + 1292 + PHASE * p + 4 * i + j] ^= SIGN ^ key;
        }
    }
    out.dst = bytes(&dst);
    let (a_at, b_at) = vec16_offsets(offset);
    let (a_at, b_at) = (
        usize::try_from(a_at / 4).expect("an offset in a usize"),
        usize::try_from(b_at / 4).expect("an offset in a usize"),
    );
    let a = words(&out.a);
    let mut b = words(&out.b);
    for t in 0..THREADS as usize {
        for j in 0..4 {
            b[b_at + 4 * t + j] = a[a_at + 4 * t + j] ^ SIGN;
            b[b_at + 4 * (64 + t) + j] ^= a[a_at + 4 * (64 + t) + j] & NOT_EXPONENT;
            b[b_at + 4 * (128 + t) + j] = a[a_at + 4 * (128 + t) + j];
        }
    }
    out.b = bytes(&b);
    out
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect()
}

fn bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every input word is a normal `float`, so the kernels' negations and moves are exact on
    /// every device.
    #[test]
    fn every_input_word_is_a_normal_float() {
        let i = inputs();
        for (name, b) in [("src", &i.src), ("dst", &i.dst), ("a", &i.a), ("b", &i.b)] {
            for (n, w) in words(b).into_iter().enumerate() {
                let exponent = (w >> 23) & 0xff;
                assert!((1..=254).contains(&exponent), "{name} word {n}: {w:#010x}");
            }
        }
    }

    /// A run at each offset writes inside its buffers' regions only and differs from the run at
    /// every other offset, so an access placed at the wrong phase cannot match the answer of
    /// another.
    #[test]
    fn each_offset_answers_its_own_words() {
        let before = inputs();
        let answers: Vec<Buffers> = OFFSETS.iter().map(|&o| answer(o)).collect();
        for (k, (o, after)) in OFFSETS.iter().zip(&answers).enumerate() {
            assert_eq!(after.src, before.src, "offset {o}: src is never written");
            assert_eq!(after.a, before.a, "offset {o}: a is never written");
            let base = (*o / 4) as usize;
            let (d0, d1) = (words(&before.dst), words(&after.dst));
            for (n, (x, y)) in d0.iter().zip(&d1).enumerate() {
                if x != y {
                    assert!(
                        (base..base + 1552).contains(&n),
                        "offset {o}: dst word {n} written outside the regions"
                    );
                }
            }
            for other in &answers[k + 1..] {
                assert_ne!(after.dst, other.dst, "offset {o}: dst equal to another's");
                assert_ne!(after.b, other.b, "offset {o}: b equal to another's");
            }
        }
    }
}
