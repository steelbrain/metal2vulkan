//! What a compute encoder moves by value: `compute_command_encoder` returns it, and
//! `end_encoding` takes it, so each of those, and each move of it in between, copies the encoder
//! whole. Its core and its bind state lie in boxes the device recycles (`encoder::SpareCores`,
//! `compute::SpareLiveDispatches`), so a move copies pointers to them: under 1 KiB, where the two
//! held in place would make each move a copy of over 4 KiB.

/// A compute encoder is at most 1 KiB.
#[test]
fn a_compute_encoder_moves_at_most_1_kib() {
    let size = std::mem::size_of::<crate::ComputeCommandEncoder>();
    println!("ENCODER-MOVES ComputeCommandEncoder {size} bytes");
    assert!(size <= 1024, "{size} bytes");
}
