//! What a compute encoder moves by value: `compute_command_encoder` returns it, and
//! `end_encoding` takes it, so each of those, and each move of it in between, copies the encoder
//! whole. Its state lies in a box of its own, and its core and its bind state in boxes the device
//! recycles (`encoder::SpareCores`, `compute::SpareLiveDispatches`), so a move copies one pointer,
//! where its state held in place made each move a copy of 888 bytes.

/// A compute encoder is one pointer.
#[test]
fn a_compute_encoder_moves_one_pointer() {
    let size = std::mem::size_of::<crate::ComputeCommandEncoder>();
    println!("ENCODER-MOVES ComputeCommandEncoder {size} bytes");
    assert_eq!(size, std::mem::size_of::<usize>(), "{size} bytes");
}
