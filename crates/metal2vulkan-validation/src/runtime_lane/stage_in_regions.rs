//! The stage-in region a kernel's `[[stage_in]]` fetch clamps to, on the device the cfg selects,
//! under the validation layer, through the runtime as a consumer reaches it (`set_stage_in_region`,
//! `set_stage_in_region_indirect`, `dispatch_threads`). The owned fixture `stage_in_regions` runs
//! each case our probes ran (the M3 Max, macOS 26.7, and an M4, macOS 27.0.1), with the same
//! descriptor, buffers, region, grid and threadgroup, and must write the
//! words Metal wrote on both chips, plain and under Metal's validation layer alike:
//!
//! - a step by thread position at rate `r` reads the region's origin plus `min(position / r,
//!   size - 1)` on its axis: a thread past the region reads the region's last element;
//! - an indexed step reads the index at `min(position, size - 1)` plus the origin;
//! - a region of size 0 clamps nothing; each axis of a 2D region clamps apart; an indirect region
//!   and a region set twice read as the direct one; a constant step reads element 0.
//!
//! Buffer 1 holds `0x10 + e` at element `e`, buffer 2 `0x200 + e`, and buffer 3 the index `20 +
//! 2i` at `i`, each a `UInt` of stride 4. `stage_in_region_word` writes its attribute plus 0x100
//! (Metal's probe kernel wrote it alone, so its words are compared plus 0x100), and
//! `stage_in_region_pair` writes `(a << 16) | b`, `a` stepping by x and `b` by y.

use super::{
    at_plan, buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device_with,
    Messages, ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{AttributeFormat, IndexType, StepFunction};
use metal2vulkan::pipeline::{
    ComputePipelineDescriptor, StageInputAttributeDescriptor, StageInputDescriptor,
    StageInputLayoutDescriptor, StageSource,
};
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    CommandBufferStatus, ComputePipelineState, Device, PipelineOptions, Region, Size,
};
use std::path::Path;

/// How a case sets the stage-in region.
#[derive(Clone, Copy, Debug)]
enum How {
    /// `set_stage_in_region`.
    Direct,
    /// `set_stage_in_region_indirect` from a buffer holding the region's six words.
    Indirect,
    /// `set_stage_in_region` of 8 x 8 x 1, then of the case's region.
    Twice,
}

/// One measured case: the layouts of buffers 1 (and 2) as `(step, rate)` (one layout runs
/// `stage_in_region_word`, two `stage_in_region_pair`), the region, the grid and threadgroup in
/// x and y, how the region is set, and the words Metal's probe kernel wrote.
struct Case {
    name: &'static str,
    layouts: &'static [(StepFunction, u32)],
    origin: [u32; 3],
    size: [u32; 3],
    grid: [u32; 2],
    group: [u32; 2],
    how: How,
    metal: &'static [u32],
}

/// Each case our probes measured: its kernel, steps, region, grid, threadgroup, how the region is
/// set, and the words Metal wrote, row by row.
const CASES: &[Case] = &[
    Case {
        name: "x_r8_g8",
        layouts: &[(StepFunction::ThreadPositionInGridX, 1)],
        origin: [0, 0, 0],
        size: [8, 1, 1],
        grid: [8, 1],
        group: [8, 1],
        how: How::Direct,
        metal: &[0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17],
    },
    Case {
        name: "x_r6_g8",
        layouts: &[(StepFunction::ThreadPositionInGridX, 1)],
        origin: [0, 0, 0],
        size: [6, 1, 1],
        grid: [8, 1],
        group: [8, 1],
        how: How::Direct,
        metal: &[0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x15, 0x15],
    },
    Case {
        name: "x_r6_g8_tg4",
        layouts: &[(StepFunction::ThreadPositionInGridX, 1)],
        origin: [0, 0, 0],
        size: [6, 1, 1],
        grid: [8, 1],
        group: [4, 1],
        how: How::Direct,
        metal: &[0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x15, 0x15],
    },
    Case {
        name: "x_o2_r4_g8",
        layouts: &[(StepFunction::ThreadPositionInGridX, 1)],
        origin: [2, 0, 0],
        size: [4, 1, 1],
        grid: [8, 1],
        group: [8, 1],
        how: How::Direct,
        metal: &[0x12, 0x13, 0x14, 0x15, 0x15, 0x15, 0x15, 0x15],
    },
    Case {
        name: "x_r0_g4",
        layouts: &[(StepFunction::ThreadPositionInGridX, 1)],
        origin: [0, 0, 0],
        size: [0, 1, 1],
        grid: [4, 1],
        group: [4, 1],
        how: How::Direct,
        metal: &[0x10, 0x11, 0x12, 0x13],
    },
    Case {
        name: "x_o3_r0_g4",
        layouts: &[(StepFunction::ThreadPositionInGridX, 1)],
        origin: [3, 0, 0],
        size: [0, 1, 1],
        grid: [4, 1],
        group: [4, 1],
        how: How::Direct,
        metal: &[0x13, 0x14, 0x15, 0x16],
    },
    Case {
        name: "x_rate2_r2_g8",
        layouts: &[(StepFunction::ThreadPositionInGridX, 2)],
        origin: [0, 0, 0],
        size: [2, 1, 1],
        grid: [8, 1],
        group: [8, 1],
        how: How::Direct,
        metal: &[0x10, 0x10, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11],
    },
    Case {
        name: "x_rate2_r5_g8",
        layouts: &[(StepFunction::ThreadPositionInGridX, 2)],
        origin: [0, 0, 0],
        size: [5, 1, 1],
        grid: [8, 1],
        group: [8, 1],
        how: How::Direct,
        metal: &[0x10, 0x10, 0x11, 0x11, 0x12, 0x12, 0x13, 0x13],
    },
    Case {
        name: "x_ind_r6_g8",
        layouts: &[(StepFunction::ThreadPositionInGridX, 1)],
        origin: [0, 0, 0],
        size: [6, 1, 1],
        grid: [8, 1],
        group: [8, 1],
        how: How::Indirect,
        metal: &[0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x15, 0x15],
    },
    Case {
        name: "x_twice_r6_g8",
        layouts: &[(StepFunction::ThreadPositionInGridX, 1)],
        origin: [0, 0, 0],
        size: [6, 1, 1],
        grid: [8, 1],
        group: [8, 1],
        how: How::Twice,
        metal: &[0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x15, 0x15],
    },
    Case {
        name: "y_r3_g2x6",
        layouts: &[(StepFunction::ThreadPositionInGridY, 1)],
        origin: [0, 0, 0],
        size: [1, 3, 1],
        grid: [2, 6],
        group: [2, 6],
        how: How::Direct,
        metal: &[
            0x10, 0x10, 0x11, 0x11, 0x12, 0x12, 0x12, 0x12, 0x12, 0x12, 0x12, 0x12,
        ],
    },
    Case {
        name: "y_o1_r3_g2x6",
        layouts: &[(StepFunction::ThreadPositionInGridY, 1)],
        origin: [0, 1, 0],
        size: [1, 3, 1],
        grid: [2, 6],
        group: [2, 6],
        how: How::Direct,
        metal: &[
            0x11, 0x11, 0x12, 0x12, 0x13, 0x13, 0x13, 0x13, 0x13, 0x13, 0x13, 0x13,
        ],
    },
    Case {
        name: "xy_r3x2_g4x4",
        layouts: &[
            (StepFunction::ThreadPositionInGridX, 1),
            (StepFunction::ThreadPositionInGridY, 1),
        ],
        origin: [0, 0, 0],
        size: [3, 2, 1],
        grid: [4, 4],
        group: [4, 4],
        how: How::Direct,
        metal: &[
            0x100200, 0x110200, 0x120200, 0x120200, 0x100201, 0x110201, 0x120201, 0x120201,
            0x100201, 0x110201, 0x120201, 0x120201, 0x100201, 0x110201, 0x120201, 0x120201,
        ],
    },
    Case {
        name: "xy_o1x1_r2x2_g4x4",
        layouts: &[
            (StepFunction::ThreadPositionInGridX, 1),
            (StepFunction::ThreadPositionInGridY, 1),
        ],
        origin: [1, 1, 0],
        size: [2, 2, 1],
        grid: [4, 4],
        group: [4, 4],
        how: How::Direct,
        metal: &[
            0x110201, 0x120201, 0x120201, 0x120201, 0x110202, 0x120202, 0x120202, 0x120202,
            0x110202, 0x120202, 0x120202, 0x120202, 0x110202, 0x120202, 0x120202, 0x120202,
        ],
    },
    Case {
        name: "xy_r0x2_g4x4",
        layouts: &[
            (StepFunction::ThreadPositionInGridX, 1),
            (StepFunction::ThreadPositionInGridY, 1),
        ],
        origin: [0, 0, 0],
        size: [0, 2, 1],
        grid: [4, 4],
        group: [4, 4],
        how: How::Direct,
        metal: &[
            0x100200, 0x110200, 0x120200, 0x130200, 0x100201, 0x110201, 0x120201, 0x130201,
            0x100201, 0x110201, 0x120201, 0x130201, 0x100201, 0x110201, 0x120201, 0x130201,
        ],
    },
    Case {
        name: "xi_r6_g8",
        layouts: &[(StepFunction::ThreadPositionInGridXIndexed, 1)],
        origin: [0, 0, 0],
        size: [6, 1, 1],
        grid: [8, 1],
        group: [8, 1],
        how: How::Direct,
        metal: &[0x24, 0x26, 0x28, 0x2a, 0x2c, 0x2e, 0x2e, 0x2e],
    },
    Case {
        name: "xi_o2_r4_g8",
        layouts: &[(StepFunction::ThreadPositionInGridXIndexed, 1)],
        origin: [2, 0, 0],
        size: [4, 1, 1],
        grid: [8, 1],
        group: [8, 1],
        how: How::Direct,
        metal: &[0x26, 0x28, 0x2a, 0x2c, 0x2c, 0x2c, 0x2c, 0x2c],
    },
    Case {
        name: "xi_r0_g4",
        layouts: &[(StepFunction::ThreadPositionInGridXIndexed, 1)],
        origin: [0, 0, 0],
        size: [0, 1, 1],
        grid: [4, 1],
        group: [4, 1],
        how: How::Direct,
        metal: &[0x24, 0x26, 0x28, 0x2a],
    },
    Case {
        name: "yi_r3_g2x6",
        layouts: &[(StepFunction::ThreadPositionInGridYIndexed, 1)],
        origin: [0, 0, 0],
        size: [1, 3, 1],
        grid: [2, 6],
        group: [2, 6],
        how: How::Direct,
        metal: &[
            0x24, 0x24, 0x26, 0x26, 0x28, 0x28, 0x28, 0x28, 0x28, 0x28, 0x28, 0x28,
        ],
    },
    Case {
        name: "yi_o1_r3_g2x6",
        layouts: &[(StepFunction::ThreadPositionInGridYIndexed, 1)],
        origin: [0, 1, 0],
        size: [1, 3, 1],
        grid: [2, 6],
        group: [2, 6],
        how: How::Direct,
        metal: &[
            0x25, 0x25, 0x27, 0x27, 0x29, 0x29, 0x29, 0x29, 0x29, 0x29, 0x29, 0x29,
        ],
    },
    Case {
        name: "c_r6_g8",
        layouts: &[(StepFunction::Constant, 0)],
        origin: [0, 0, 0],
        size: [6, 1, 1],
        grid: [8, 1],
        group: [8, 1],
        how: How::Direct,
        metal: &[0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10],
    },
    Case {
        name: "c_r0_g4",
        layouts: &[(StepFunction::Constant, 0)],
        origin: [0, 0, 0],
        size: [0, 1, 1],
        grid: [4, 1],
        group: [4, 1],
        how: How::Direct,
        metal: &[0x10, 0x10, 0x10, 0x10],
    },
    Case {
        name: "x_r1_g4",
        layouts: &[(StepFunction::ThreadPositionInGridX, 1)],
        origin: [0, 0, 0],
        size: [1, 1, 1],
        grid: [4, 1],
        group: [4, 1],
        how: How::Direct,
        metal: &[0x10, 0x10, 0x10, 0x10],
    },
    Case {
        name: "x_rate2_o1_r2_g8",
        layouts: &[(StepFunction::ThreadPositionInGridX, 2)],
        origin: [1, 0, 0],
        size: [2, 1, 1],
        grid: [8, 1],
        group: [8, 1],
        how: How::Direct,
        metal: &[0x11, 0x11, 0x12, 0x12, 0x12, 0x12, 0x12, 0x12],
    },
    Case {
        name: "x_rate2_o3_r0_g8",
        layouts: &[(StepFunction::ThreadPositionInGridX, 2)],
        origin: [3, 0, 0],
        size: [0, 1, 1],
        grid: [8, 1],
        group: [8, 1],
        how: How::Direct,
        metal: &[0x13, 0x13, 0x14, 0x14, 0x15, 0x15, 0x16, 0x16],
    },
    Case {
        name: "x_rate3_o2_r3_g12",
        layouts: &[(StepFunction::ThreadPositionInGridX, 3)],
        origin: [2, 0, 0],
        size: [3, 1, 1],
        grid: [12, 1],
        group: [12, 1],
        how: How::Direct,
        metal: &[
            0x12, 0x12, 0x12, 0x13, 0x13, 0x13, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14,
        ],
    },
    Case {
        name: "y_rate2_o1_r2_g2x8",
        layouts: &[(StepFunction::ThreadPositionInGridY, 2)],
        origin: [0, 1, 0],
        size: [1, 2, 1],
        grid: [2, 8],
        group: [2, 8],
        how: How::Direct,
        metal: &[
            0x11, 0x11, 0x11, 0x11, 0x12, 0x12, 0x12, 0x12, 0x12, 0x12, 0x12, 0x12, 0x12, 0x12,
            0x12, 0x12,
        ],
    },
    Case {
        name: "xi_o2_r0_g4",
        layouts: &[(StepFunction::ThreadPositionInGridXIndexed, 1)],
        origin: [2, 0, 0],
        size: [0, 1, 1],
        grid: [4, 1],
        group: [4, 1],
        how: How::Direct,
        metal: &[0x26, 0x28, 0x2a, 0x2c],
    },
];

/// The fixture's pipeline for `layouts`: attribute `i` a `UInt` at offset 0 of buffer `i + 1`,
/// whose layout has stride 4 and the step and rate of `layouts[i]`; 32-bit indices at buffer 3.
fn pipeline(device: &Device, layouts: &[(StepFunction, u32)]) -> ComputePipelineState {
    let ll =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/public/engine/stage_in_regions.ll");
    let library = device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap();
    let entry = match layouts.len() {
        1 => "stage_in_region_word",
        _ => "stage_in_region_pair",
    };
    let function = library.new_function_with_name(entry).unwrap();
    let mut input = StageInputDescriptor::new();
    for (attribute, &(step, rate)) in (0..).zip(layouts) {
        let buffer = attribute + 1;
        input.attributes.insert(
            attribute,
            StageInputAttributeDescriptor::new(AttributeFormat::UInt, 0, buffer),
        );
        let mut layout = StageInputLayoutDescriptor::new(Some(4));
        layout.step_function = step;
        layout.step_rate = rate;
        input.layouts.insert(buffer, layout);
    }
    (input.index_type, input.index_buffer_index) = (IndexType::UInt32, 3);
    let mut descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.stage_input = Some(input);
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// 256 words of `f(i)`.
fn words(f: impl Fn(u32) -> u32) -> Vec<u8> {
    (0..256).flat_map(|i| f(i).to_le_bytes()).collect()
}

/// The words case `case` writes on `device`.
fn dispatched(device: &Device, case: &Case) -> Vec<u32> {
    let pipeline = pipeline(device, case.layouts);
    let out = buffer_with(device, &[0xee; 1024]).unwrap();
    let sources = [
        words(|e| 0x10 + e),
        words(|e| 0x200 + e),
        words(|i| 20 + 2 * i),
    ]
    .map(|b| buffer_with(device, &b).unwrap());
    let [ox, oy, oz] = case.origin.map(u64::from);
    let [sx, sy, sz] = case.size.map(u64::from);
    let region_words: Vec<u8> = case
        .origin
        .iter()
        .chain(&case.size)
        .flat_map(|w| w.to_le_bytes())
        .collect();
    let indirect = buffer_with(device, &region_words).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&pipeline).unwrap();
    encoder.set_buffer(Some(&out), 0, 0).unwrap();
    for (index, buffer) in (1..).zip(&sources) {
        encoder.set_buffer(Some(buffer), 0, index).unwrap();
    }
    let region = Region::new_3d(ox, oy, oz, sx, sy, sz);
    match case.how {
        How::Direct => encoder.set_stage_in_region(region).unwrap(),
        How::Indirect => encoder.set_stage_in_region_indirect(&indirect, 0).unwrap(),
        How::Twice => {
            encoder
                .set_stage_in_region(Region::new_3d(0, 0, 0, 8, 8, 1))
                .unwrap();
            encoder.set_stage_in_region(region).unwrap();
        }
    }
    let [gx, gy] = case.grid.map(u64::from);
    let [tx, ty] = case.group.map(u64::from);
    encoder
        .dispatch_threads(Size::new(gx, gy, 1), Size::new(tx, ty, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
    read_buffer(&out, 0, 4 * gx * gy)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect()
}

/// Every measured case writes the words Metal wrote (plus 0x100 for the word kernel).
#[test]
fn a_thread_past_the_stage_in_region_reads_as_metal() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
        for case in CASES {
            let added = match case.layouts.len() {
                1 => 0x100,
                _ => 0,
            };
            let metal: Vec<u32> = case.metal.iter().map(|w| w + added).collect();
            assert_eq!(dispatched(&device, case), metal, "{}", case.name);
        }
    }
    layer_reported_nothing_at_all(&messages);
}
