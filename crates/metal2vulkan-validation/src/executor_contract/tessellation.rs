//! How both executors draw a tessellation case: the Metal oracle's `MTLRenderPipelineDescriptor`
//! tessellation fields, `setTessellationFactorBuffer:offset: instanceStride:`, `setCullMode:`, and
//! `drawPatches:...`, and the engine candidate's `RenderPipelineDescriptor::tessellation`,
//! tessellation factor buffer, cull mode, and `DrawCall::Patches`, derived once here, so the two
//! agree.
//!
//! A case's patches are drawn from patch 0 with no patch index buffer, as many instances as the
//! case states from base instance 0, `numberOfPatchControlPoints` the function's
//! `[[patch(_, N)]]`. Each patch's factors are its half-precision edge factors then its inside
//! factors, little-endian, in one buffer bound at offset 0 with instance stride 0, read per patch
//! (`MTLTessellationFactorStepFunctionPerPatch`), never scaled. The pipeline partitions as the
//! case states (integer when it states nothing), clamps every factor to at most 64, winds its
//! output as the case states (counter-clockwise when it states nothing), and reads no
//! control-point index buffer; the encoder culls as the case states (nothing when it states
//! nothing) with Metal's default clockwise front face. Changing any of it changes the oracle's
//! draws, which is a Metal oracle version (`crate::metal::ORACLE_ABI`) and an engine executor
//! version together.

use crate::case::{
    CullMode, TessellationDraw, TessellationFactors, TessellationPartition, TessellationWinding,
};
use crate::check::interface::AirInterface;
use metal2vulkan::metal;
use metal2vulkan::pipeline::TessellationDescriptor;
use metal2vulkan_air::abi::Stage as AbiStage;

/// `maxTessellationFactor`: every factor is clamped to at most it.
pub const MAX_TESSELLATION_FACTOR: u32 = 64;
/// `tessellationFactorFormat`.
pub const FACTOR_FORMAT: metal::TessellationFactorFormat = metal::TessellationFactorFormat::Half;
/// `tessellationFactorStepFunction`: each patch reads its own factors, every instance the same.
pub const FACTOR_STEP_FUNCTION: metal::TessellationFactorStepFunction =
    metal::TessellationFactorStepFunction::PerPatch;
/// `tessellationControlPointIndexType`: no control-point index buffer.
pub const CONTROL_POINT_INDEX_TYPE: metal::TessellationControlPointIndexType =
    metal::TessellationControlPointIndexType::None;
/// The factor buffer's `offset`.
pub const FACTOR_OFFSET: u64 = 0;
/// The factor buffer's `instanceStride`.
pub const FACTOR_INSTANCE_STRIDE: u64 = 0;
/// `setTessellationFactorScale:` is never called: Metal's default.
pub const FACTOR_SCALE: f32 = 1.0;
/// The draw's `patchStart`.
pub const PATCH_START: u32 = 0;
/// The draw's `baseInstance`.
pub const BASE_INSTANCE: u32 = 0;

/// The tessellation fields of the render pipeline descriptor both executors create for `draw`.
pub fn descriptor(draw: &TessellationDraw) -> TessellationDescriptor {
    let mut descriptor = TessellationDescriptor::default();
    descriptor.max_tessellation_factor = MAX_TESSELLATION_FACTOR;
    descriptor.partition_mode = partition_mode(draw.partition_mode);
    descriptor.factor_format = FACTOR_FORMAT;
    descriptor.factor_step_function = FACTOR_STEP_FUNCTION;
    descriptor.output_winding_order = winding(draw.output_winding);
    descriptor.control_point_index_type = CONTROL_POINT_INDEX_TYPE;
    descriptor.factor_scale_enabled = false;
    descriptor
}

/// The Metal partition mode of an authored one.
pub fn partition_mode(partition: TessellationPartition) -> metal::TessellationPartitionMode {
    use metal::TessellationPartitionMode as Mode;
    match partition {
        TessellationPartition::Pow2 => Mode::Pow2,
        TessellationPartition::Integer => Mode::Integer,
        TessellationPartition::FractionalOdd => Mode::FractionalOdd,
        TessellationPartition::FractionalEven => Mode::FractionalEven,
    }
}

/// The Metal winding of an authored output winding.
pub fn winding(winding: TessellationWinding) -> metal::Winding {
    match winding {
        TessellationWinding::Clockwise => metal::Winding::Clockwise,
        TessellationWinding::CounterClockwise => metal::Winding::CounterClockwise,
    }
}

/// The encoder's cull mode for `draw`.
pub fn cull_mode(draw: &TessellationDraw) -> metal::CullMode {
    match draw.cull_mode {
        CullMode::None => metal::CullMode::None,
        CullMode::Front => metal::CullMode::Front,
        CullMode::Back => metal::CullMode::Back,
    }
}

/// The factor buffer's bytes: each patch's edge factors, then its inside factors, as
/// little-endian binary16 words (`MTLTriangleTessellationFactorsHalf` or
/// `MTLQuadTessellationFactorsHalf` per patch, in patch order).
pub fn factor_bytes(factors: &[TessellationFactors]) -> Vec<u8> {
    factors
        .iter()
        .flat_map(|patch| patch.edge_f16.iter().chain(&patch.inside_f16))
        .flat_map(|word| word.to_le_bytes())
        .collect()
}

/// The draw's `numberOfPatchControlPoints`: the `[[patch(_, N)]]` the function whose AIR-layer
/// interface is `interface` states. A function that is no post-tessellation vertex function draws
/// no patches, and one whose patch states no count (`[[patch(quad)]]`, which takes the draw's)
/// has none to draw with: a case states none.
pub fn control_points(interface: &AirInterface) -> Result<u32, String> {
    match interface.stage {
        AbiStage::Vertex { patch: Some(patch) } => patch.control_points.ok_or_else(|| {
            "a tessellation draw of a function whose patch states no control-point count".into()
        }),
        _ => Err(
            "a tessellation draw of a function that is no post-tessellation vertex \
                  function"
                .into(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each patch's factors are its edge factors, then its inside factors, low byte first.
    #[test]
    fn factor_bytes_are_each_patchs_edge_then_inside_halves() {
        let factors = [
            TessellationFactors {
                edge_f16: vec![0x3c00, 0x4000, 0x4400],
                inside_f16: vec![0x4200],
            },
            TessellationFactors {
                edge_f16: vec![0x0001, 0x0000, 0x8000, 0x7e00],
                inside_f16: vec![0x5400, 0x7c00],
            },
        ];
        assert_eq!(
            factor_bytes(&factors),
            [
                0x00, 0x3c, 0x00, 0x40, 0x00, 0x44, 0x00, 0x42, //
                0x01, 0x00, 0x00, 0x00, 0x00, 0x80, 0x00, 0x7e, 0x00, 0x54, 0x00, 0x7c,
            ]
        );
    }

    /// A draw that states nothing is drawn in the state every tessellation case was once drawn in;
    /// each authored mode is the Metal mode of its name.
    #[test]
    fn a_draw_is_drawn_in_its_authored_state_and_otherwise_the_original_one() {
        let one = TessellationFactors {
            edge_f16: vec![0x3c00; 4],
            inside_f16: vec![0x3c00; 2],
        };
        let mut draw = TessellationDraw {
            factors: vec![one],
            instance_count: 1,
            amplification_count: 1,
            control_points: vec![],
            patch_inputs: vec![],
            partition_mode: TessellationPartition::default(),
            output_winding: TessellationWinding::default(),
            cull_mode: CullMode::default(),
        };
        let original = descriptor(&draw);
        assert_eq!(original.max_tessellation_factor, 64);
        assert_eq!(
            original.partition_mode,
            metal::TessellationPartitionMode::Integer
        );
        assert_eq!(
            original.output_winding_order,
            metal::Winding::CounterClockwise
        );
        assert_eq!(
            original.factor_step_function,
            metal::TessellationFactorStepFunction::PerPatch
        );
        assert_eq!(
            original.factor_format,
            metal::TessellationFactorFormat::Half
        );
        assert_eq!(
            original.control_point_index_type,
            metal::TessellationControlPointIndexType::None
        );
        assert!(!original.factor_scale_enabled);
        assert_eq!(cull_mode(&draw), metal::CullMode::None);

        for (authored, mode) in [
            (
                TessellationPartition::Pow2,
                metal::TessellationPartitionMode::Pow2,
            ),
            (
                TessellationPartition::FractionalOdd,
                metal::TessellationPartitionMode::FractionalOdd,
            ),
            (
                TessellationPartition::FractionalEven,
                metal::TessellationPartitionMode::FractionalEven,
            ),
        ] {
            draw.partition_mode = authored;
            assert_eq!(descriptor(&draw).partition_mode, mode);
        }
        draw.partition_mode = TessellationPartition::Integer;
        draw.output_winding = TessellationWinding::Clockwise;
        assert_eq!(
            descriptor(&draw).output_winding_order,
            metal::Winding::Clockwise
        );
        draw.output_winding = TessellationWinding::CounterClockwise;
        for (authored, mode) in [
            (CullMode::Front, metal::CullMode::Front),
            (CullMode::Back, metal::CullMode::Back),
        ] {
            draw.cull_mode = authored;
            assert_eq!(cull_mode(&draw), mode);
        }
    }
}
