//! The kernels a consumer runs for its own work beside a program's ([`OwnKernel`]), built from no
//! Metal function. Each has the interface every function the translator compiles has: the
//! pipeline layout's one push range, which holds the kernel's one field at offset 0, the device
//! address (`u64`) of its parameter block in memory the consumer writes for the dispatch
//! ([`PushField::ArgumentTableAddress`]), and no descriptor. Each is a compute stage of one
//! invocation a workgroup, entry point `main`, dispatched with as many workgroups along x as the
//! kernel names.

use crate::budget::{Budget, Meter};
use crate::device::DeviceProfile;
use crate::error::Error;
use crate::pipeline::sym::PushField;
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{TypeId, Value};
use crate::spirv::ir::{BinOp, ConvOp, ExecMode, ExecModel, IoDir, MemAccess, Type, TypedModule};

/// A kernel of a consumer's own work.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OwnKernel {
    /// The sequence counts of an execution of an indirect command buffer's commands over the range
    /// a buffer holds (Metal's `executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:`)
    /// that runs the buffer's commands as parts from command 0, each part a
    /// `vkCmdExecuteGeneratedCommandsEXT` whose `sequenceCountAddress` is its count. Workgroup `p`
    /// writes part `p`'s count: how many of its commands, from its first, it runs. That is through
    /// the range's last command, and none for a part wholly before the range's location or wholly
    /// past its end; a part that holds the location runs its commands before it too, which the
    /// vertex function's range test skips. The range is clamped to the buffer's commands. Its
    /// parameter block is [`ExecutionCounts`]'s, and it is dispatched with a workgroup a part.
    ExecutionCounts,
}

/// The parameter block of [`OwnKernel::ExecutionCounts`]: 8-byte aligned, of
/// [`COUNTS_AT`](Self::COUNTS_AT) bytes and 4 a part.
#[derive(Clone, Copy, Debug)]
pub struct ExecutionCounts;

impl ExecutionCounts {
    /// The device address (`u64`) of the range: Metal's `MTLIndirectCommandBufferExecutionRange`
    /// there, a `u32` location, then a `u32` length; a multiple of 4.
    pub const RANGE_AT: u32 = 0;
    /// The commands of the buffer (`u32`).
    pub const CAPACITY_AT: u32 = 8;
    /// The commands of each part (`u32`, not 0); the last part holds the rest.
    pub const PART_AT: u32 = 12;
    /// Part 0's count (`u32`), part `p`'s at `COUNTS_AT + 4 * p`: what the kernel writes.
    pub const COUNTS_AT: u32 = 16;
}

/// `kernel`'s SPIR-V for a device of `profile` (entry point `main`), verified as a compiled
/// function's module is.
///
/// Err: what the verifier finds the profile lacks.
pub fn own_kernel(kernel: OwnKernel, profile: &DeviceProfile) -> Result<Vec<u32>, Error> {
    let module = match kernel {
        OwnKernel::ExecutionCounts => execution_counts()?,
    };
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&module, profile, &mut meter)?;
    // The module names no resource: it has no relocation.
    Ok(crate::spirv::serialize::serialize(module)?.words)
}

/// A `u32` constant.
fn k32(fb: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    let c = fb.u32_const(v)?;
    fb.module().const_value(c)
}

/// The `u32` at `offset` bytes past device address `address`, which nothing writes while the
/// kernel runs.
fn word(fb: &mut FunctionBuilder<'_>, u32_: TypeId, address: Value, offset: u32) -> BResult<Value> {
    let at = match offset {
        0 => address,
        offset => {
            let offset = crate::memory::roots::k64(fb, u64::from(offset))?;
            fb.binary(BinOp::IAdd, address, offset)?
        }
    };
    let place = fb.device_place_read_only(at, u32_)?;
    let access = MemAccess {
        aligned: Some(4),
        ..MemAccess::default()
    };
    fb.load(place, access)
}

/// The lesser of `a` and `b`, unsigned.
fn umin(fb: &mut FunctionBuilder<'_>, a: Value, b: Value) -> BResult<Value> {
    let less = fb.binary(BinOp::ULessThan, a, b)?;
    fb.select(less, a, b)
}

/// [`OwnKernel::ExecutionCounts`].
fn execution_counts() -> BResult<TypedModule> {
    let mut mb = ModuleBuilder::new();
    let u32_ = mb.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let u64_ = mb.ty(Type::Int {
        width: 64,
        signed: false,
    })?;
    let uvec3 = mb.ty(Type::Vector { elem: u32_, len: 3 })?;
    let void = mb.ty(Type::Void)?;
    let field = mb.push_member(PushField::ArgumentTableAddress, u64_, 0)?;
    let push = mb.push_global()?;
    let group = mb.builtin(crate::spv::BuiltIn::WorkgroupId, IoDir::Input, uvec3)?;
    let mut fb = mb.function(void)?;
    let base = fb.module().global_value(push)?;
    let member = k32(&mut fb, field.index)?;
    let at = fb.access_chain(base, &[member])?;
    let block = fb.load(at, MemAccess::default())?;
    let range_at = fb.device_place_read_only(block, u64_)?;
    let eight = MemAccess {
        aligned: Some(8),
        ..MemAccess::default()
    };
    let range = fb.load(range_at, eight)?;
    let capacity = word(&mut fb, u32_, block, ExecutionCounts::CAPACITY_AT)?;
    let part = word(&mut fb, u32_, block, ExecutionCounts::PART_AT)?;
    let location = word(&mut fb, u32_, range, 0)?;
    let length = word(&mut fb, u32_, range, 4)?;
    let group = fb.module().global_value(group)?;
    let group = fb.load(group, MemAccess::default())?;
    let p = fb.composite_extract(group, &[0])?;
    // The range, clamped to the buffer's commands: `[first, end)`.
    let first = umin(&mut fb, location, capacity)?;
    let room = fb.binary(BinOp::ISub, capacity, first)?;
    let taken = umin(&mut fb, length, room)?;
    let end = fb.binary(BinOp::IAdd, first, taken)?;
    // Part `p`: `[start, stop)`.
    let start = fb.binary(BinOp::IMul, p, part)?;
    let past = fb.binary(BinOp::IAdd, start, part)?;
    let stop = umin(&mut fb, past, capacity)?;
    let before_stop = fb.binary(BinOp::ULessThan, first, stop)?;
    let after_start = fb.binary(BinOp::UGreaterThan, end, start)?;
    let holds = fb.binary(BinOp::LogicalAnd, before_stop, after_start)?;
    let through = umin(&mut fb, end, stop)?;
    let runs = fb.binary(BinOp::ISub, through, start)?;
    let none = k32(&mut fb, 0)?;
    let count = fb.select(holds, runs, none)?;
    let four = k32(&mut fb, 4)?;
    let offset = fb.binary(BinOp::IMul, p, four)?;
    let counts = k32(&mut fb, ExecutionCounts::COUNTS_AT)?;
    let offset = fb.binary(BinOp::IAdd, offset, counts)?;
    let offset = fb.convert(ConvOp::UConvert, offset, u64_)?;
    let to = fb.binary(BinOp::IAdd, block, offset)?;
    let to = fb.device_place(to, u32_)?;
    let access = MemAccess {
        aligned: Some(4),
        ..MemAccess::default()
    };
    fb.store(to, count, access)?;
    fb.ret(None)?;
    let f = fb.finish()?;
    mb.entry(
        ExecModel::GLCompute,
        f,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )?;
    mb.finalize()
}

#[cfg(test)]
mod tests {
    use super::{execution_counts, own_kernel, ExecutionCounts, OwnKernel};
    use crate::spirv::interp::{run, Buffer, Config, Resources};

    /// Where the interpreter puts the parameter block and the range.
    const BLOCK: u64 = 0x1000_0000;
    const RANGE: u64 = 0x2000_0000;

    /// Part `p`'s count by brute force: one past the part's last command in the range, less the
    /// part's first; 0 where none of its commands is in the range.
    fn want(capacity: u32, part: u32, location: u32, length: u32) -> Vec<u32> {
        let end = u64::from(location) + u64::from(length);
        let held = |c: u32| c >= location && u64::from(c) < end;
        (0..capacity.div_ceil(part))
            .map(|p| {
                let start = p * part;
                let stop = (start + part).min(capacity);
                (start..stop)
                    .rev()
                    .find(|&c| held(c))
                    .map_or(0, |c| c + 1 - start)
            })
            .collect()
    }

    /// The counts the kernel writes on the interpreter, every count's word filled with a mark
    /// first.
    fn counts(capacity: u32, part: u32, location: u32, length: u32) -> Vec<u32> {
        let module = execution_counts().expect("the module");
        let parts = capacity.div_ceil(part);
        let mut block = RANGE.to_le_bytes().to_vec();
        block.extend(capacity.to_le_bytes());
        block.extend(part.to_le_bytes());
        assert_eq!(block.len(), ExecutionCounts::COUNTS_AT as usize);
        block.resize(block.len() + 4 * parts as usize, 0xAB);
        let range: Vec<u8> = [location, length]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let resources = Resources {
            buffers: vec![
                Buffer {
                    slot: None,
                    address: Some(BLOCK),
                    bytes: block,
                },
                Buffer {
                    slot: None,
                    address: Some(RANGE),
                    bytes: range,
                },
            ],
            push: BLOCK.to_le_bytes().to_vec(),
            spec: Default::default(),
        };
        let config = Config {
            workgroups: [parts, 1, 1],
            ..Config::default()
        };
        let out = run(&module, &config, resources).expect("the run");
        (out.buffers[0][ExecutionCounts::COUNTS_AT as usize..].chunks(4))
            .map(|w| u32::from_le_bytes(w.try_into().expect("a word")))
            .collect()
    }

    /// Each part's count runs through the range's last command and no further, none for a part
    /// wholly before the location or past the end, over ranges at location 0, inside a part,
    /// across parts, empty, past the commands and of the widest `u32`s.
    #[test]
    fn each_parts_count_runs_through_the_ranges_last_command() {
        for (capacity, part) in [(10, 4), (785, 785), (1, 1), (64, 7), (8162, 2701)] {
            let ranges = [
                (0, capacity),
                (0, 0),
                (0, 3),
                (3, 2),
                (5, 3),
                (part, part),
                (part + part / 2, part),
                (capacity - 1, 100),
                (capacity + 2, 5),
                (0, u32::MAX),
                (u32::MAX, u32::MAX),
            ];
            for (location, length) in ranges {
                assert_eq!(
                    counts(capacity, part, location, length),
                    want(capacity, part, location, length),
                    "{capacity} commands in parts of {part}, the range {location} for {length}"
                );
            }
        }
    }

    /// The kernel is valid SPIR-V (the owned verifier and `spirv-val`) on every committed profile,
    /// and the public function makes it there.
    #[test]
    fn the_count_kernel_is_valid_on_every_committed_profile() {
        let profiles = crate::spirv::testing::committed_profiles().expect("the profiles");
        for (stem, profile) in &profiles {
            let module = execution_counts().expect("the module");
            if let Err(e) = crate::spirv::testing::check_valid(module, profile) {
                panic!("{stem}: {e}");
            }
            if let Err(e) = own_kernel(OwnKernel::ExecutionCounts, profile) {
                panic!("{stem}: {e}");
            }
        }
        assert!(!profiles.is_empty(), "no committed profile");
    }
}
