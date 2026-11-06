//! Device test of the GPU-family audit on the device the cfg selects: every family's
//! `supportsFamily:` answer and the rows that block it, printed as the venue's burn-down.

use super::{layer_reported_nothing_at_all, validated_device, Messages, ScratchCache};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::TextureUsage;
use metal2vulkan_runtime::{
    FamilyRow, FormatCapability, FormatRow, GpuFamily, HeapDescriptor, HeapType, LimitRow, Refusal,
    ResourceOptions, RowAudit, RowState, RuntimeGap, TextureDescriptor,
};

/// Every family answers `supportsFamily:` true exactly where none of its rows blocks, and the
/// blocking rows are the family's rows that block. Each family's answer and its blocking rows are
/// printed (`FAMILY` and `BLOCK` lines), the venue's burn-down. The families without a column
/// have their one row; Apple1 and Mac1 answer as Apple2 and Mac2; a value that names no family is
/// false. Each format row's state agrees with the runtime's per-format capability query, and
/// Apple9's argument-buffer sampler row with the device's `maxArgumentBufferSamplerCount`
/// (`DeviceProperties::max_argument_buffer_sampler_count`) against the cell's 500000 states,
/// printed (`ARGUMENT_SAMPLERS`). Once everything is dropped, the validation layer reported
/// nothing.
#[test]
fn every_family_answers_with_its_blocking_rows() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        eprintln!("GPU families on {}", device.name());
        for family in GpuFamily::ALL {
            let rows = device.family_rows(family);
            let blockers = device.family_blockers(family);
            let supported = device.supports_family(family.raw());
            let blocking: Vec<RowAudit> =
                rows.iter().filter(|r| r.state.blocks()).cloned().collect();
            assert_eq!(blockers, blocking, "{family:?}");
            assert_eq!(supported, blockers.is_empty(), "{family:?}");
            eprintln!(
                "FAMILY {family:?} raw={} supports={supported} rows={} blocking={}",
                family.raw(),
                rows.len(),
                blockers.len()
            );
            for r in &blockers {
                eprintln!(
                    "BLOCK {family:?} | {:?} | {} | {:?}",
                    r.row,
                    r.row.label(),
                    r.state
                );
            }
            // A format row is the runtime's own per-format answer.
            for r in &rows {
                if let FamilyRow::Format(row, capability) = r.row {
                    let given = row
                        .formats()
                        .iter()
                        .all(|&pf| device.pixel_format_capabilities(pf).contains(capability));
                    assert_eq!(given, r.state == RowState::Exact, "{family:?} {:?}", r.row);
                }
            }
        }
        let answer = device.properties().max_argument_buffer_sampler_count;
        let samplers = FamilyRow::Limit(LimitRow::ArgumentBufferSamplers);
        let apple9 = (device.family_rows(GpuFamily::Apple9).into_iter())
            .find(|r| r.row == samplers)
            .map(|r| r.state);
        eprintln!("ARGUMENT_SAMPLERS answer={answer} apple9={apple9:?}");
        let want = match answer >= 500_000 {
            true => RowState::Exact,
            false => RowState::Refused(Refusal::ArgumentBufferSamplerCount {
                required: 500_000,
                available: answer,
            }),
        };
        assert_eq!(apple9, Some(want));
        assert_eq!(
            device.family_rows(GpuFamily::Metal4),
            vec![RowAudit {
                row: FamilyRow::Metal4Api,
                state: RowState::Gap(RuntimeGap::Metal4Api),
            }]
        );
        assert_eq!(
            device.family_rows(GpuFamily::Apple1),
            device.family_rows(GpuFamily::Apple2)
        );
        assert_eq!(
            device.family_rows(GpuFamily::Mac1),
            device.family_rows(GpuFamily::Mac2)
        );
        for raw in [0, 1012, 6001, -1009] {
            assert!(!device.supports_family(raw), "{raw}");
        }
        // A format's Sparse answer is whether the device makes a sparse texture of it: the answer
        // and a 16 x 16 texture of a sparse heap agree.
        let rgba8 = FormatRow::RGBA8Unorm.formats()[0];
        let sparse = device
            .pixel_format_capabilities(rgba8)
            .contains(FormatCapability::Sparse);
        let heap = HeapDescriptor::new(HeapType::Sparse, 1 << 20, ResourceOptions::private());
        let mut texture = TextureDescriptor::texture_2d(rgba8, 16, 16, false);
        texture.usage = TextureUsage::SHADER_READ;
        texture.resource_options = ResourceOptions::private();
        let made = device
            .new_heap(&heap)
            .and_then(|heap| heap.new_texture(&texture));
        assert_eq!(sparse, made.is_ok(), "{made:?}");
    }
    layer_reported_nothing_at_all(&messages);
}
