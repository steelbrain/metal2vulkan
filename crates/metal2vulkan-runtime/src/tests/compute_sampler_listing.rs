//! A compute encoder's dispatches take the device's live sampler states to list only where they
//! changed since the listing their encoder's bind state keeps, as draws do
//! (`sampler::ArgumentSamplers`, `WorkCounts::argument_samplers_taken`): a kernel whose variant has
//! no set-1 arrays, dispatched with a constant of its own each time (so no memo entry holds),
//! takes none while no sampler state is made or goes, beside eight a program keeps alive; a state
//! made between two dispatches, and that state gone, each make the next dispatch take them once.
//! Every dispatch writes its own words, read back against Metal's answer.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The kernel is `grid_classes`'s `flat2d`
//! (`llvm-as`).

use super::blit_holds::{device, Scratch, WAIT};
use super::grid_classes::{state, FLAT};
use super::opencl_mode::{words, words_buffer};
use crate::{CommandBufferStatus, Size};
use metal2vulkan::bind::SamplerDescriptor;

/// The dispatches of the encoder.
const DISPATCHES: usize = 9;

/// The words each dispatch has to itself.
const WORDS: usize = 64;

/// The threads of each dispatch's ragged grid of one row.
const THREADS: u32 = 37;

/// What a word holds before a dispatch writes it.
const UNWRITTEN: u32 = u32::MAX;

#[test]
fn dispatches_take_the_device_s_sampler_states_only_where_they_changed() {
    let scratch = Scratch::new("compute-sampler-listing");
    let device = device(&scratch.0);
    let state = state(&device, FLAT, "flat2d");
    // A program's sampler states, alive through the dispatches.
    let _kept: Vec<_> = (0..8)
        .map(|n| {
            let mut descriptor = SamplerDescriptor::new();
            descriptor.lod_max_clamp = 1.0 + n as f32;
            device.new_sampler_state(&descriptor).unwrap()
        })
        .collect();
    let out = words_buffer(&device, &vec![UNWRITTEN; WORDS * DISPATCHES]);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    let taken = || device.work_counts().argument_samplers_taken;
    let mut counts = Vec::new();
    let mut made = None;
    for n in 0..DISPATCHES {
        // A sampler state made before dispatch 3, gone before dispatch 6.
        if n == 3 {
            let mut descriptor = SamplerDescriptor::new();
            descriptor.lod_max_clamp = 99.0;
            made = Some(device.new_sampler_state(&descriptor).unwrap());
        }
        if n == 6 {
            made = None;
        }
        let before = taken();
        let k = n as u32 + 1;
        encoder
            .set_buffer(Some(&out), (n * WORDS * 4) as u64, 0)
            .unwrap();
        encoder.set_bytes(&k.to_le_bytes(), 1).unwrap();
        let grid = Size::new(u64::from(THREADS), 1, 1);
        encoder.dispatch_threads(grid, Size::new(64, 1, 1)).unwrap();
        counts.push(taken() - before);
    }
    assert!(made.is_none());
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    // The first dispatch settles the encoder's state from none, and may take them.
    assert_eq!(counts[1..], [0, 0, 1, 0, 0, 1, 0, 0], "{counts:?}");
    let written = words(&out);
    for n in 0..DISPATCHES {
        let k = n as u32 + 1;
        let want: Vec<u32> = (0..WORDS as u32)
            .map(|i| match i < THREADS {
                true => 3 * i + k,
                false => UNWRITTEN,
            })
            .collect();
        assert_eq!(written[n * WORDS..(n + 1) * WORDS], want, "dispatch {n}");
    }
}
