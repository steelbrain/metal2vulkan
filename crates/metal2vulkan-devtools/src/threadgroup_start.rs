//! The owned fixture `tgz_reads`: every thread reads its words of a declared threadgroup array and
//! of a `[[threadgroup(0)]]` region before any thread of its threadgroup writes them, then stores a
//! nonzero pattern into both and reads its neighbour's words back (the fixture's source says what
//! each thread does). The dispatches a run makes ([`GROUPS`] threadgroups of each of [`SIZES`]
//! threads, [`DISPATCHES`] times each, the region [`REGION_BYTES`] long), and what each dispatch
//! answers where threadgroup memory reads zero until the program writes it ([`answer`]): the
//! conformance case `compute-threadgroup-memory-reads-zero` records Metal's answer and holds it
//! equal to [`answer`], and the new runtime's device test holds its own equal to it.

/// The fixture, and its kernel.
pub const FIXTURE: &str = "tgz_reads";
/// The words of the fixture's declared threadgroup array, its `WORDS`.
pub const WORDS: u32 = 2048;
/// The bytes a run sets for the `[[threadgroup(0)]]` region: as many words as the array.
pub const REGION_BYTES: u32 = 4 * WORDS;
/// The threadgroups of each dispatch: many waves of threadgroups of 16 KiB on every GPU, so later
/// threadgroups run where earlier ones stored their patterns.
pub const GROUPS: u32 = 2048;
/// The threads of a threadgroup, one dimension: whole SIMD groups of 32, then two sizes that are
/// not (a last SIMD group of 26 threads, and one SIMD group of 30).
pub const SIZES: [u32; 3] = [256, 250, 30];
/// The dispatches of each size, one after the other in one compute pass.
pub const DISPATCHES: usize = 2;

/// What one dispatch of `groups` threadgroups of `size` threads writes to its four output words
/// where threadgroup memory reads zero until the program writes it: no word read nonzero before a
/// write (words 0 and 1), the wrapping sum over every threadgroup's words of the pattern stored in
/// the array plus twice the one stored in the region (word 2), and the sum of the squares of the
/// threads of each SIMD group, Metal's SIMD groups of 32 in thread order (word 3).
pub fn answer(groups: u32, size: u32) -> [u32; 4] {
    let mut sum = 0u32;
    for word in 0..groups * WORDS {
        let pattern = 0x8000_0000 | (word & 0x3fff_ffff);
        sum = sum
            .wrapping_add(pattern)
            .wrapping_add((pattern | 0x4000_0000).wrapping_mul(2));
    }
    let shape: u32 = (0..size)
        .step_by(32)
        .map(|first| (size - first).min(32).pow(2))
        .sum();
    [0, 0, sum, groups * shape]
}
