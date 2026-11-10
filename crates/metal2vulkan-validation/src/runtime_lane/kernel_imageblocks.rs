//! Kernel imageblocks through the runtime on the device the cfg selects, under the validation
//! layer:
//!
//! - Every committed public kernel case with an explicit imageblock, and no tile pass, runs through
//!   the runtime (its size set with `set_imageblock_size`) against its committed Metal observation,
//!   or the output of the engine candidate run afresh where the two differ ([`census_with`]).
//! - `set_imageblock_size` takes what plain Metal takes and refuses the rest at the call, changing
//!   nothing. A size set before the pipeline state holds for it, and an encoder that never sets one
//!   runs a 1 by 1 imageblock, as Metal's does.
//! - A pipeline state answers `imageblock_memory_length_for_dimensions`: 0 without an imageblock,
//!   and for one each side rounded up to a power of two.

use super::kernel_textures::census_with;
use super::{
    at_plan, bound_buffers, encode_case, layer_reported_nothing_at_all, pipeline_state,
    public_case, public_cases, validated_device_with, CaseResources, Messages, PublicCase,
    ScratchCache, WAIT,
};
use crate::candidate::runtime_case::Committed;
use crate::case::Stage;
use crate::gpu::TEST_DEVICE;
use metal2vulkan_runtime::DeviceOptions;
use metal2vulkan_runtime::{
    CommandBufferStatus, ComputeCommandEncoder, Device, Error, Invalid, Problem, Size,
};

/// A kernel that reads its imageblock's extent and a neighbour's cell, 4 by 2 cells.
const EXTENT: &str = "imageblock-extent-queries-and-a-neighbours-cell";

/// A kernel of a 1 by 1 imageblock.
const WORD: &str = "imageblock-word";

/// A kernel without an imageblock.
const PLAIN: &str = "threadgroup-scratch-of-eighty-byte-elements";

/// Run `test` on a validated device of the lane, then fail on any validation message.
fn on_device(test: impl FnOnce(&Device)) {
    on_device_with(|_| {}, test)
}

/// [`on_device`], `adjust` applied to the options of each device it makes (`at_plan` gives them
/// their queue plan).
fn on_device_with(adjust: fn(&mut DeviceOptions), test: impl FnOnce(&Device)) {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), adjust).unwrap();
        test(&device);
    }
    layer_reported_nothing_at_all(&messages);
}

/// Every committed public kernel case with an explicit imageblock, and no tile pass, passes the
/// [`census_with`].
#[test]
fn kernels_with_explicit_imageblocks_match_metal() {
    let mut cases = public_cases(|case| {
        case.stage == Stage::Kernel && case.public_source.is_some() && case.imageblock.is_some()
    })
    .unwrap();
    // A tile pass's imageblock is its render targets': `tile_dispatches`' census runs it.
    cases.retain(|(_, case)| case.as_ref().map_or(true, |c| c.compute.tile.is_none()));
    assert!(
        !cases.is_empty(),
        "no committed case has an explicit imageblock"
    );
    census_with(at_plan, "runtime kernel imageblocks", &cases);
}

/// `case` run once on `device` with its imageblock size left to `size`, which runs on the
/// encoder before the case's pipeline state and bindings are set: its output.
fn run_sized(
    device: &Device,
    case: &PublicCase,
    size: impl FnOnce(&mut ComputeCommandEncoder),
) -> Vec<u8> {
    let mut compute = case.compute.clone();
    compute.imageblock = None;
    let pipeline = pipeline_state(device, case).unwrap();
    let resources = CaseResources::new(device, case).unwrap();
    let buffers = bound_buffers(device, &compute, &resources).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    size(&mut encoder);
    encode_case(&mut encoder, &pipeline, &compute, &buffers, &resources).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let committed = Committed {
        command_buffer,
        buffers,
        resources,
        attachments: Default::default(),
        executions: Vec::new(),
    };
    committed.output(&compute).unwrap()
}

/// `setImageblockWidth:height:` as plain Metal takes it where its API validation asserts: a side
/// that is not a power of two runs at the power above, and a side of 0, or more than 1024 cells in
/// one row, runs 1 by 1. More than 1024 cells over more rows and columns, and a side past 65536,
/// are refused at the call, typed, and the size set before holds. A size set before the pipeline
/// state holds for it, and an encoder that sets none runs 1 by 1, as Metal's kernel reads.
#[test]
fn an_imageblock_size_is_taken_or_refused_at_the_call() {
    on_device_with(at_plan, |device| {
        let case = public_case(EXTENT).unwrap();
        let [width, height] = case.compute.imageblock.unwrap();
        let output = run_sized(device, &case, |encoder| {
            encoder
                .set_imageblock_size(width.into(), height.into())
                .unwrap();
            for (w, h) in [(48, 40), (64, 32), (2, 1024), (65537, 1)] {
                assert_eq!(
                    encoder.set_imageblock_size(w, h),
                    Err(Error::Invalid(Invalid::new(
                        "set_imageblock_size",
                        Problem::ImageblockSize {
                            width: w,
                            height: h,
                        },
                    ))),
                    "{w}x{h}"
                );
            }
        });
        assert!(
            output == case.metal_output,
            "{EXTENT}: the run at the size set before four refused ones is not Metal's"
        );
        assert!(width >= 4, "{EXTENT}: its width rounds from below");
        let below = u64::from(width / 2 + 1);
        let output = run_sized(device, &case, |encoder| {
            encoder.set_imageblock_size(below, height.into()).unwrap();
        });
        assert!(
            output == case.metal_output,
            "{EXTENT}: a width of {below} did not run as {width}"
        );
        let word = public_case(WORD).unwrap();
        assert_eq!(word.compute.imageblock, Some([1, 1]));
        let output = run_sized(device, &word, |_| {});
        assert!(
            output == word.metal_output,
            "{WORD}: an encoder that set no size is not 1 by 1"
        );
        for (w, h) in [(0, 3), (8, 0), (2048, 1), (1, 2048)] {
            let output = run_sized(device, &word, |encoder| {
                encoder.set_imageblock_size(w, h).unwrap();
            });
            assert!(output == word.metal_output, "{WORD}: {w}x{h} is not 1 by 1");
        }
    });
}

/// A pipeline state's `imageblock_memory_length_for_dimensions` is 0 without an imageblock; for an
/// explicit one it is none at a side of 0, rounds each side up to a power of two, ignores the
/// depth, and grows with the cells.
#[test]
fn imageblock_lengths_round_sides_to_powers_of_two() {
    on_device(|device| {
        let plain = pipeline_state(device, &public_case(PLAIN).unwrap()).unwrap();
        assert_eq!(
            plain.imageblock_memory_length_for_dimensions(Size::new(8, 8, 1)),
            0
        );
        let extent = pipeline_state(device, &public_case(EXTENT).unwrap()).unwrap();
        let length = |w, h, d| extent.imageblock_memory_length_for_dimensions(Size::new(w, h, d));
        assert_eq!(length(0, 8, 1), 0);
        assert_eq!(length(8, 0, 1), 0);
        assert!(length(1, 1, 1) > 0);
        assert_eq!(length(3, 5, 1), length(4, 8, 1));
        assert_eq!(length(4, 8, 2), length(4, 8, 1));
        assert_eq!(length(8, 8, 1), 2 * length(4, 8, 1));
    });
}
