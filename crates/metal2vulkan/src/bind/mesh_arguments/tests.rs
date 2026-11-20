//! The mesh arguments module: valid on every committed profile with `VK_EXT_mesh_shader`, Metal's
//! rule on the CPU interpreter (copied, zeroed, and zeroed with `MESH_GRID_PAST_LIMITS` or
//! `MESH_GRID_DISCARDED`), including the source residual and a short range, the unfolded object
//! form's, and the folded object form's fold, word for word the CPU's (`fold_object_grid`).

use super::*;
use crate::bind::status::{ExecutionStatus, StatusReport};
use crate::device::VkExtension;
use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
use crate::spirv::testing::{assert_valid, committed_profiles};

fn words_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

/// Run the module of `form` and `rule` over `source` (bound whole) with push `residual` and
/// `packed`: the scratch's words and the status report.
fn run_pass(
    form: MeshArgumentsForm,
    rule: MeshGridRule,
    source: &[u32],
    (residual, packed): (u32, u32),
) -> (Vec<u32>, StatusReport) {
    let m = typed_module(form, rule).expect("module");
    let bytes = MeshArguments::scratch_bytes(form) as usize;
    let mut out = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let res = Resources {
            buffers: vec![
                Buffer {
                    slot: Some(SOURCE),
                    address: None,
                    bytes: source.iter().flat_map(|w| w.to_le_bytes()).collect(),
                },
                Buffer {
                    slot: Some(SCRATCH),
                    address: None,
                    bytes: vec![0xEE; bytes + 4],
                },
                Buffer {
                    slot: Some(STATUS),
                    address: None,
                    bytes: ExecutionStatus::initial().to_vec(),
                },
            ],
            push: [residual, packed]
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect(),
            spec: Default::default(),
        };
        let cfg = Config {
            order,
            ..Config::default()
        };
        out.push(run(&m, &cfg, res).expect("run").buffers);
    }
    assert_eq!(out[0], out[1], "order independent");
    let b = &out[0];
    let scratch = words(&b[1]);
    assert_eq!(
        scratch[bytes / 4],
        0xEEEE_EEEE,
        "the pass writes its scratch only"
    );
    let status: [u8; ExecutionStatus::BYTES] = b[2].as_slice().try_into().expect("status");
    (
        scratch[..bytes / 4].to_vec(),
        ExecutionStatus::decode(&status),
    )
}

/// The mesh form's pass: the command and the status report.
fn pass(rule: MeshGridRule, source: &[u32], residual: u32) -> ([u32; 3], StatusReport) {
    let (scratch, report) = run_pass(MeshArgumentsForm::Mesh, rule, source, (residual, 0));
    ([scratch[0], scratch[1], scratch[2]], report)
}

/// RADV's mesh and task limits on the Radeon 780M.
fn radv() -> crate::device::MeshLimits {
    crate::device::MeshLimits {
        max_task_work_group_count: [65535; 3],
        max_task_work_group_total_count: 4_194_304,
        max_mesh_work_group_count: [65535; 3],
        max_mesh_work_group_total_count: 4_194_304,
        ..Default::default()
    }
}

/// RADV's task limits on the Radeon 780M.
fn radv_task() -> MeshGridLimits {
    let l = radv();
    MeshGridLimits {
        axes: l.max_task_work_group_count,
        total: l.max_task_work_group_total_count,
    }
}

#[test]
fn the_module_is_valid_on_every_committed_profile_with_mesh_shading() {
    for (name, p) in committed_profiles().expect("profiles") {
        if !p
            .fields()
            .enabled
            .extensions
            .contains(&VkExtension::ExtMeshShader)
        {
            continue;
        }
        for form in [
            MeshArgumentsForm::Mesh,
            MeshArgumentsForm::Object,
            MeshArgumentsForm::FoldedObject,
        ] {
            let rule = MeshGridRule::of(form, &p.fields().limits.mesh);
            // Every such profile launches each mesh grid Metal draws.
            if form == MeshArgumentsForm::Mesh {
                assert!(
                    rule.device.axes.iter().all(|&a| a >= METAL_MESH_GRID_AXIS)
                        && rule.device.total >= METAL_MESH_GRID_TOTAL,
                    "{name}: {rule:?}"
                );
            }
            assert_valid(typed_module(form, rule).expect("module"), &p);
        }
    }
}

/// Metal's mesh grids (the module docs): drawn up to 65,535 an axis and 1,048,575 in all, else
/// nothing, with no fault.
#[test]
fn a_mesh_grid_draws_as_metal_s() {
    let rule = MeshGridRule::of(MeshArgumentsForm::Mesh, &radv());
    for (counts, drawn) in [
        ([3, 1, 1], true),
        ([65535, 1, 1], true),
        ([1, 65535, 1], true),
        ([1, 1, 65535], true),
        ([1023, 1025, 1], true),
        ([16, 65535, 1], true),
        ([41, 41, 39], true),
        ([0, 5, 1], true),
        ([65536, 1, 1], false),
        ([1, 1, 65536], false),
        ([1024, 1024, 1], false),
        ([128, 128, 64], false),
        ([4096, 1024, 1], false),
        ([u32::MAX, u32::MAX, u32::MAX], false),
        ([65535, 65535, 65535], false),
    ] {
        let (out, report) = pass(rule, &counts, 0);
        let want = if drawn { counts } else { [0; 3] };
        assert_eq!(out, want, "{counts:?}");
        assert!(!report.mesh_grid_past_limits, "{counts:?}");
        assert!(!report.mesh_grid_discarded, "{counts:?}");
        assert_eq!(metal_mesh_grid_draws_nothing(counts), !drawn, "{counts:?}");
    }
}

/// An object grid on a device that does not fold object grids: copied within the task limits,
/// else zeroed and reported, as past the device's limits or, past what Metal runs, as Metal's
/// discard.
#[test]
fn an_unfolded_object_grid_past_the_task_limits_is_reported() {
    let rule = MeshGridRule::of(MeshArgumentsForm::Object, &radv());
    for (counts, within, discarded) in [
        ([3, 1, 1], true, false),
        ([65535, 64, 1], true, false),
        ([2048, 2048, 1], true, false),
        ([0, 70000, 1], false, false),
        ([65536, 1, 1], false, false),
        ([2048, 2049, 1], false, false),
        ([1 << 29, 1, 1], false, false),
        ([(1 << 29) + 1, 1, 1], false, true),
        ([u32::MAX, 2, 1], false, true),
    ] {
        let (scratch, report) = run_pass(MeshArgumentsForm::Object, rule, &counts, (0, 0));
        let want = if within { counts } else { [0; 3] };
        assert_eq!(scratch, want, "{counts:?}");
        assert_eq!(report.mesh_grid_discarded, discarded, "{counts:?}");
        assert_eq!(
            report.mesh_grid_past_limits,
            !within && !discarded,
            "{counts:?}"
        );
    }
}

/// The folded object form's scratch for the fold: the record, the count, and `OBJECT_PIECES` commands of
/// the piece's dispatch (zero where nothing runs).
fn object_scratch(fold: &ObjectFold) -> Vec<u32> {
    let mut out = fold.record.to_vec();
    out.push(fold.pieces);
    let command = if fold.pieces == 0 {
        [0; 3]
    } else {
        fold.dispatch
    };
    for _ in 0..MeshArguments::OBJECT_PIECES {
        out.extend(command);
    }
    out
}

/// Every piece's threadgroups are Metal's in order, each once: piece `p`'s workgroup `w` is Metal's
/// threadgroup `p · nx · ny + w`, and the pieces cover the grid.
fn covers(fold: &ObjectFold, counts: [u32; 3]) -> bool {
    let n: u64 = counts.iter().map(|&c| u64::from(c)).product();
    let piece = u64::from(fold.dispatch[0]) * u64::from(fold.dispatch[1]);
    u64::from(fold.pieces) * piece >= n && u64::from(fold.pieces.saturating_sub(1)) * piece < n
}

/// Metal's object grids, as the M3 Max ran the owned fixture `mesh_object_grid` (macOS 27.0.1):
/// threads per grid wrap to 32 bits along each axis; an indirect draw runs every threadgroup, a
/// direct one those the wrapped threads fill, rounded up.
#[test]
fn metal_s_object_grid_wraps_its_threads_per_grid() {
    for (groups, threads, direct, runs, reported) in [
        (
            [70_000, 1, 1],
            [1, 1, 1],
            true,
            [70_000, 1, 1],
            [70_000, 1, 1],
        ),
        (
            [300, 300, 2],
            [4, 2, 1],
            true,
            [300, 300, 2],
            [1200, 600, 2],
        ),
        (
            [4_194_303, 1, 1],
            [1024, 1, 1],
            true,
            [4_194_303, 1, 1],
            [4_294_966_272, 1, 1],
        ),
        ([4_194_304, 1, 1], [1024, 1, 1], true, [0, 1, 1], [0, 1, 1]),
        (
            [4_194_304, 1, 1],
            [1024, 1, 1],
            false,
            [4_194_304, 1, 1],
            [0, 1, 1],
        ),
        (
            [4_194_305, 1, 1],
            [1024, 1, 1],
            true,
            [1, 1, 1],
            [1024, 1, 1],
        ),
        (
            [4_194_305, 1, 1],
            [1024, 1, 1],
            false,
            [4_194_305, 1, 1],
            [1024, 1, 1],
        ),
        (
            [4_294_968, 1, 1],
            [1000, 1, 1],
            true,
            [1, 1, 1],
            [704, 1, 1],
        ),
        (
            [4_294_970, 1, 1],
            [1000, 1, 1],
            true,
            [3, 1, 1],
            [2704, 1, 1],
        ),
        (
            [4_294_970, 1, 1],
            [1000, 1, 1],
            false,
            [4_294_970, 1, 1],
            [2704, 1, 1],
        ),
        (
            [2, 4_194_305, 1],
            [1, 1024, 1],
            true,
            [2, 1, 1],
            [2, 1024, 1],
        ),
        (
            [65536, 64, 1],
            [1024, 1, 1],
            true,
            [65536, 64, 1],
            [67_108_864, 64, 1],
        ),
    ] {
        assert_eq!(
            metal_object_grid(groups, threads, direct),
            Some((runs, reported)),
            "{groups:?} of {threads:?}, direct {direct}"
        );
    }
    assert_eq!(metal_object_grid([3, 1, 1], [0, 1, 1], true), None);
}

/// The folded object form folds any grid it runs within the task stage's limits, in the pieces that
/// cover it, word for word as the CPU's fold of an indirect draw (and a direct draw's upload is
/// the scratch through its pieces), and draws nothing for a zero count, a grid Metal's GPU
/// discards (setting `MESH_GRID_DISCARDED`), or one past the device's pieces (setting
/// `MESH_GRID_PAST_LIMITS`).
#[test]
fn an_object_grid_folds_within_the_task_limits_as_the_cpu_does() {
    let nvidia = MeshGridLimits {
        axes: [4_194_304, 65535, 65535],
        total: 4_194_304,
    };
    let small = MeshGridLimits {
        axes: [100, 100, 100],
        total: 5000,
    };
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Outcome {
        Runs,
        Nothing,
        Discarded,
        Past,
    }
    use Outcome::*;
    // An `x` axis just past half of Vulkan's least task total: one row a piece, the fewest
    // threadgroups a piece of such a device runs, so 2^29 takes all `OBJECT_PIECES`.
    let wide = MeshGridLimits {
        axes: [2_097_153, 65535, 65535],
        total: 4_194_304,
    };
    for (task, counts, threads, outcome) in [
        (radv_task(), [3, 1, 1], [1, 1, 1], Runs),
        (radv_task(), [65535, 64, 1], [32, 1, 1], Runs),
        (radv_task(), [70_000, 1, 1], [1, 1, 1], Runs),
        (radv_task(), [2048, 2049, 1], [2, 2, 1], Runs),
        (radv_task(), [1 << 24, 1, 1], [1, 1, 1], Runs),
        (radv_task(), [65536, 4096, 1], [1, 1, 1], Runs),
        (radv_task(), [7, 9, 11], [4, 4, 4], Runs),
        (radv_task(), [1 << 23, 1, 1], [1024, 1, 1], Runs),
        (radv_task(), [4_294_970, 1, 1], [1000, 1, 1], Runs),
        (radv_task(), [65536, 8192, 1], [1, 1, 1], Runs),
        (radv_task(), [1 << 29, 1, 1], [1, 1, 1], Runs),
        (nvidia, [70_000, 3, 1], [1, 1, 1], Runs),
        (nvidia, [1 << 24, 4, 1], [1, 1, 1], Runs),
        (nvidia, [1 << 29, 1, 1], [1, 1, 1], Runs),
        (wide, [1 << 29, 1, 1], [1, 1, 1], Runs),
        (small, [600_000, 1, 1], [1, 1, 1], Runs),
        (small, [1_280_000, 1, 1], [1, 1, 1], Runs),
        (radv_task(), [0, 70_000, 1], [1, 1, 1], Nothing),
        (radv_task(), [65536, 8193, 1], [1, 1, 1], Discarded),
        (radv_task(), [(1 << 29) + 1, 1, 1], [1, 1, 1], Discarded),
        (radv_task(), [u32::MAX, 2, 1], [1, 1, 1], Discarded),
        (small, [1_280_001, 1, 1], [1, 1, 1], Past),
    ] {
        let rule = MeshGridRule {
            metal: None,
            device: task,
        };
        let packed = crate::lower::dispatch::pack_nominal(threads).expect("packs");
        let fold = fold_object_grid(counts, threads, task, false).expect("fold");
        let what = format!("{counts:?} of {threads:?} on {task:?}: {fold:?}");
        let runs = outcome == Runs;
        assert_eq!(fold.pieces > 0, runs, "{what}");
        assert_eq!(fold.discarded, outcome == Discarded, "{what}");
        assert_eq!(fold.past, outcome == Past, "{what}");
        if runs {
            let [nx, ny, nz] = fold.dispatch;
            assert!(
                nx <= task.axes[0] && ny <= task.axes[1] && nz == 1,
                "{what}"
            );
            assert!(nx * ny <= task.total, "{what}");
            assert!(fold.pieces <= MeshArguments::OBJECT_PIECES, "{what}");
            assert!(covers(&fold, counts), "{what}");
            let per_grid: Vec<u32> = (0..3).map(|i| counts[i].wrapping_mul(threads[i])).collect();
            assert_eq!(fold.record[..3], counts[..], "{what}");
            assert_eq!(fold.record[4..7], per_grid[..], "{what}");
        }
        assert_eq!(fold.record[3], packed, "{what}");
        let (scratch, report) =
            run_pass(MeshArgumentsForm::FoldedObject, rule, &counts, (0, packed));
        assert_eq!(scratch, object_scratch(&fold), "{what}");
        let upload = 4 * (9 + 3 * fold.pieces as usize);
        assert_eq!(fold.bytes(), words_bytes(&scratch)[..upload], "{what}");
        assert_eq!(report.mesh_grid_past_limits, fold.past, "{what}");
        assert_eq!(report.mesh_grid_discarded, fold.discarded, "{what}");
    }
}

#[test]
fn the_residual_selects_the_arguments_and_a_short_range_reads_zero() {
    let rule = MeshGridRule::of(MeshArgumentsForm::Mesh, &radv());
    let (out, _) = pass(rule, &[9, 9, 3, 4, 5, 9], 8);
    assert_eq!(out, [3, 4, 5]);
    let (out, report) = pass(rule, &[500, 2], 4);
    assert_eq!(out, [2, 0, 0]);
    assert!(!report.mesh_grid_past_limits);
}

#[test]
fn the_pass_binds_an_aligned_source_range() {
    let form = MeshArgumentsForm::Mesh;
    let a = MeshArguments::new(IndirectRef::new(3, 300), form, 256, [1; 3]).expect("pass");
    assert_eq!((a.source_range.offset, a.source_range.range), (256, 56));
    assert_eq!(a.push_constants, [44, 0, 0, 0, 0, 0, 0, 0]);
    let a = MeshArguments::new(IndirectRef::new(3, 12), form, 1, [1; 3]).expect("pass");
    assert_eq!((a.source_range.offset, a.source_range.range), (12, 12));
    assert_eq!(
        MeshArguments::new(IndirectRef::new(3, 2), form, 4, [1; 3]),
        None
    );
    let object = MeshArgumentsForm::Object;
    let a = MeshArguments::new(IndirectRef::new(3, 8), object, 4, [32, 2, 1]).expect("pass");
    assert_eq!(a.push_constants, [0; 8]);
    let folded = MeshArgumentsForm::FoldedObject;
    let a = MeshArguments::new(IndirectRef::new(3, 8), folded, 4, [32, 2, 1]).expect("pass");
    assert_eq!(a.push_constants, [0, 0, 0, 0, 31, 4, 0, 0]);
    assert_eq!(
        MeshArguments::new(IndirectRef::new(3, 8), folded, 4, [0; 3]),
        None
    );
}
