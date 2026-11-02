//! Stage-level families: derivatives, boolean vector reductions, function constants, fragment
//! operations, and imageblocks.

use super::sig::{Operand, Signature};
use super::ty::{Elem, NumTy};
use crate::abi::Builtin;
use crate::ir::AddrSpace;

fn num(elem: Elem, lanes: u32) -> Operand {
    Operand::Num(NumTy { elem, lanes })
}

/// A screen-space derivative.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DerivativeOp {
    /// `dfdx`: the derivative along screen x.
    Dfdx,
    /// `dfdy`: the derivative along screen y.
    Dfdy,
    /// `fwidth`: `abs(dfdx(p)) + abs(dfdy(p))`.
    Fwidth,
}

/// `air.[fast_]{dfdx,dfdy}.<T>` and `air.fwidth.<T>`: `T (T)`, `T` a half or float scalar or
/// vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Derivative {
    /// Which derivative.
    pub op: DerivativeOp,
    /// Spelled with the `fast_` prefix (`dfdx` and `dfdy` only).
    pub fast: bool,
    /// The value type: a `half` or `float` scalar or vector.
    pub ty: NumTy,
}

impl Derivative {
    pub(super) fn parse(rest: &str) -> Option<Derivative> {
        let (fast, rest) = match rest.strip_prefix("fast_") {
            Some(r) => (true, r),
            None => (false, rest),
        };
        let (stem, ty) = rest.split_once('.')?;
        let op = match stem {
            "dfdx" => DerivativeOp::Dfdx,
            "dfdy" => DerivativeOp::Dfdy,
            "fwidth" if !fast => DerivativeOp::Fwidth,
            _ => return None,
        };
        let ty = NumTy::parse(ty)?;
        matches!(ty.elem, Elem::F16 | Elem::F32).then_some(Derivative { op, fast, ty })
    }

    /// The typed signature.
    pub fn signature(&self) -> Signature {
        Signature::new(Operand::Num(self.ty), &[Operand::Num(self.ty)])
    }
}

/// `air.{all,any}.v<N>i1`: `i1 (<N x i1>)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BoolReduce {
    /// `true` for `all`, `false` for `any`.
    pub all: bool,
    /// The lane count of the `i1` vector operand.
    pub lanes: u32,
}

impl BoolReduce {
    pub(super) fn parse(rest: &str) -> Option<BoolReduce> {
        let (stem, ty) = rest.split_once('.')?;
        let all = match stem {
            "all" => true,
            "any" => false,
            _ => return None,
        };
        let ty = NumTy::parse(ty)?;
        (ty.elem == Elem::I1 && ty.is_vector()).then_some(BoolReduce {
            all,
            lanes: ty.lanes,
        })
    }

    /// The typed signature.
    pub fn signature(&self) -> Signature {
        Signature::new(num(Elem::I1, 1), &[num(Elem::I1, self.lanes)])
    }
}

/// Function-constant queries (resolved by canonicalization's specialization), and the read of
/// an argument the constants removed, which specialization makes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FunctionConstant {
    /// `air.is_function_constant_defined`: `i1 (ptr addrspace(2))`, the constant's global.
    IsDefined,
    /// `air.normalize_function_constant_predicate.<iN>`: `iN (iN)`.
    NormalizePredicate(Elem),
    /// The texture operation `t` (a read, sample, gather, query, or atomic) of a texture argument
    /// the pipeline's function constants removed, or through a sampler argument they removed,
    /// whose result something observes: `T ()`, where `T` is `t`'s result. Metal gives such a
    /// read no value, so lowering marks the command buffer's execution status where it runs and
    /// yields zero, which no result reported as good then carries. Not an AIR intrinsic: no name
    /// parses to it. Specialization makes it (`canon::interface`).
    AbsentRead(super::texture::Texture),
    /// A load or atomic of threadgroup memory the pipeline's function constants removed, whose
    /// result something observes: `T ()`, where `T` is the scalar or vector it reads. Metal gives
    /// such an access no defined value (its compiler dropped stores of a load's value, and an
    /// atomic changed the threadgroup memory of another argument under its validation layer and
    /// not without it), so lowering marks the command buffer's execution status
    /// where it runs and yields zero, as for [`FunctionConstant::AbsentRead`]. Not an AIR
    /// intrinsic: no name parses to it. Specialization makes it (`canon::interface`).
    AbsentMemory(AbsentAccess),
}

/// What an access of threadgroup memory the function constants removed reads
/// ([`FunctionConstant::AbsentMemory`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AbsentAccess {
    /// The scalar or vector it reads.
    pub ty: NumTy,
    /// Whether it was an atomic, which also writes: it stays where nothing observes its result.
    pub atomic: bool,
}

impl FunctionConstant {
    pub(super) fn parse(rest: &str) -> Option<FunctionConstant> {
        if rest == "is_function_constant_defined" {
            return Some(FunctionConstant::IsDefined);
        }
        let ty = NumTy::parse(rest.strip_prefix("normalize_function_constant_predicate.")?)?;
        (matches!(ty.elem, Elem::I8 | Elem::I16 | Elem::I32) && !ty.is_vector())
            .then_some(FunctionConstant::NormalizePredicate(ty.elem))
    }

    /// The typed signature.
    pub fn signature(&self) -> Signature {
        match *self {
            FunctionConstant::IsDefined => {
                Signature::new(num(Elem::I1, 1), &[Operand::Ptr(AddrSpace::Constant)])
            }
            FunctionConstant::NormalizePredicate(e) => Signature::new(num(e, 1), &[num(e, 1)]),
            FunctionConstant::AbsentRead(t) => Signature::new(t.signature().ret, &[]),
            FunctionConstant::AbsentMemory(a) => Signature::new(Operand::Num(a.ty), &[]),
        }
    }
}

/// Fragment and rasterization operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Fragment {
    /// `air.discard_fragment`: `void ()`.
    Discard,
    /// `air.map_screen_to_physical_coordinates.v2f32.p2i8.i32` and the inverse:
    /// `<2 x float> (<2 x float>, ptr addrspace(2) rate map, i32 layer)`; with `integer`, the
    /// `uint2` overloads `.v2i32.p2i8.i32`: `<2 x i32> (<2 x i32>, ptr addrspace(2), i32)`.
    MapCoordinates {
        /// `true` for `map_screen_to_physical_coordinates`, `false` for the inverse,
        /// `map_physical_to_screen_coordinates`.
        to_physical: bool,
        /// The `uint2` overload (`.v2i32`) rather than the `float2` one (`.v2f32`).
        integer: bool,
    },
    /// `air.get_num_samples.i32`: `i32 (i32)`, the rasterization sample count.
    NumSamples,
    /// `air.quad_is_helper_thread` and `air.simd_is_helper_thread`: `i1 ()`, whether the calling
    /// thread is a helper. Apple's compiler takes both in every stage, and Metal answers them alike
    /// for every thread (measured): a fragment function's helper, one rasterization runs only to
    /// complete a quad or one a discard made a helper, is one; no kernel or vertex thread is.
    IsHelperThread,
}

impl Fragment {
    pub(super) fn parse(rest: &str) -> Option<Fragment> {
        Some(match rest {
            "discard_fragment" => Fragment::Discard,
            "map_screen_to_physical_coordinates.v2f32.p2i8.i32" => Fragment::MapCoordinates {
                to_physical: true,
                integer: false,
            },
            "map_physical_to_screen_coordinates.v2f32.p2i8.i32" => Fragment::MapCoordinates {
                to_physical: false,
                integer: false,
            },
            "map_screen_to_physical_coordinates.v2i32.p2i8.i32" => Fragment::MapCoordinates {
                to_physical: true,
                integer: true,
            },
            "map_physical_to_screen_coordinates.v2i32.p2i8.i32" => Fragment::MapCoordinates {
                to_physical: false,
                integer: true,
            },
            "get_num_samples.i32" => Fragment::NumSamples,
            "quad_is_helper_thread" | "simd_is_helper_thread" => Fragment::IsHelperThread,
            _ => return None,
        })
    }

    /// The typed signature.
    pub fn signature(&self) -> Signature {
        match self {
            Fragment::Discard => Signature::new(Operand::Void, &[]),
            Fragment::MapCoordinates { integer, .. } => {
                let v2 = num(if *integer { Elem::I32 } else { Elem::F32 }, 2);
                Signature::new(
                    v2,
                    &[v2, Operand::Ptr(AddrSpace::Constant), num(Elem::I32, 1)],
                )
            }
            Fragment::NumSamples => Signature::new(num(Elem::I32, 1), &[num(Elem::I32, 1)]),
            Fragment::IsHelperThread => Signature::new(num(Elem::I1, 1), &[]),
        }
    }
}

/// Where an interpolation function reads its `interpolant`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum InterpolateAt {
    /// `interpolate_at_center()`: at the pixel's center.
    Center,
    /// `interpolate_at_centroid()`: at the centroid of the pixel's covered area.
    Centroid,
    /// `interpolate_at_sample(uint)`: at a sample's location, an `i32` operand.
    Sample,
    /// `interpolate_at_offset(float2)`: at an offset from the pixel's upper-left corner, a
    /// `<2 x float>` operand.
    Offset,
}

/// `air.interpolate_{center,centroid,sample,offset}_{perspective,no_perspective}.<T>`: Metal's
/// pull-model interpolation of an `interpolant` fragment input, a handle (`ptr addrspace(1)`).
/// `T` is a `half` or `float` scalar or vector: `T (ptr)` at the center or the centroid,
/// `T (ptr, i32)` at a sample, `T (ptr, <2 x float>)` at an offset.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Interpolate {
    /// Where it reads.
    pub at: InterpolateAt,
    /// `_perspective` (`true`) or `_no_perspective` (`false`): the interpolant's own qualifier.
    pub perspective: bool,
    /// The value type.
    pub ty: NumTy,
}

impl Interpolate {
    pub(super) fn parse(rest: &str) -> Option<Interpolate> {
        let (stem, ty) = rest.strip_prefix("interpolate_")?.split_once('.')?;
        let (at, qualifier) = stem.split_once('_')?;
        let at = match at {
            "center" => InterpolateAt::Center,
            "centroid" => InterpolateAt::Centroid,
            "sample" => InterpolateAt::Sample,
            "offset" => InterpolateAt::Offset,
            _ => return None,
        };
        let perspective = match qualifier {
            "perspective" => true,
            "no_perspective" => false,
            _ => return None,
        };
        let ty = NumTy::parse(ty)?;
        (matches!(ty.elem, Elem::F16 | Elem::F32) && matches!(ty.lanes, 1..=4)).then_some(
            Interpolate {
                at,
                perspective,
                ty,
            },
        )
    }

    /// The typed signature.
    pub fn signature(&self) -> Signature {
        let handle = Operand::Handle(super::sig::HandleKind::Interpolant);
        let ret = Operand::Num(self.ty);
        match self.at {
            InterpolateAt::Center | InterpolateAt::Centroid => Signature::new(ret, &[handle]),
            InterpolateAt::Sample => Signature::new(ret, &[handle, num(Elem::I32, 1)]),
            InterpolateAt::Offset => Signature::new(ret, &[handle, num(Elem::F32, 2)]),
        }
    }
}

/// Imageblock operations. The `(i32, <2 x i16>, i32, i16)` tail of the implicit accesses and the
/// `(<2 x i16>, i32, i16)` operands of `imageblock_data` are typed operands lowering interprets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Imageblock {
    /// `air.imageblock_data`: `ptr addrspace(4) (<2 x i16>, i32, i16)`.
    Data,
    /// `air.get_imageblock_{width,height}`: `i16 ()`.
    Width,
    /// `air.get_imageblock_height`: `i16 ()`.
    Height,
    /// `air.load.implicit_imageblock.<T>`: `T (i32, <2 x i16>, i32, i16)`.
    Load(NumTy),
    /// `air.store.implicit_imageblock.<T>`: `void (T, i32, <2 x i16>, i32, i16)`.
    Store(NumTy),
}

impl Imageblock {
    pub(super) fn parse(rest: &str) -> Option<Imageblock> {
        match rest {
            "imageblock_data" => return Some(Imageblock::Data),
            "get_imageblock_width" => return Some(Imageblock::Width),
            "get_imageblock_height" => return Some(Imageblock::Height),
            _ => {}
        }
        let (load, ty) = if let Some(t) = rest.strip_prefix("load.implicit_imageblock.") {
            (true, t)
        } else {
            (false, rest.strip_prefix("store.implicit_imageblock.")?)
        };
        let ty = NumTy::parse(ty)?;
        let ok = matches!(ty.elem, Elem::F16 | Elem::F32 | Elem::I16 | Elem::I32)
            && matches!(ty.lanes, 1..=4);
        ok.then_some(if load {
            Imageblock::Load(ty)
        } else {
            Imageblock::Store(ty)
        })
    }

    /// The typed signature.
    pub fn signature(&self) -> Signature {
        let (i16s, i32s, v2i16) = (num(Elem::I16, 1), num(Elem::I32, 1), num(Elem::I16, 2));
        match *self {
            Imageblock::Data => Signature::new(
                Operand::Ptr(AddrSpace::ThreadgroupImageblock),
                &[v2i16, i32s, i16s],
            ),
            Imageblock::Width | Imageblock::Height => Signature::new(i16s, &[]),
            Imageblock::Load(t) => Signature::new(Operand::Num(t), &[i32s, v2i16, i32s, i16s]),
            Imageblock::Store(t) => {
                Signature::new(Operand::Void, &[Operand::Num(t), i32s, v2i16, i32s, i16s])
            }
        }
    }
}

/// A kernel built-in a program-scope variable declares (Metal 3.1's global built-ins), by the
/// Metal attribute it reads. `threads_per_simdgroup` is not one: its variable reads
/// `air.get_simdgroup_size` ([`super::SimdgroupSize`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GlobalVar {
    /// `[[thread_position_in_grid]]`: `air.get_global_id`.
    ThreadPositionInGrid,
    /// `[[thread_position_in_threadgroup]]`: `air.get_local_id`.
    ThreadPositionInThreadgroup,
    /// `[[threadgroup_position_in_grid]]`: `air.get_group_id`.
    ThreadgroupPositionInGrid,
    /// `[[threads_per_threadgroup]]`: `air.get_local_size`.
    ThreadsPerThreadgroup,
    /// `[[threads_per_grid]]`: `air.get_global_size`.
    ThreadsPerGrid,
    /// `[[threadgroups_per_grid]]`: `air.get_num_groups`.
    ThreadgroupsPerGrid,
    /// `[[dispatch_threads_per_threadgroup]]`: `air.get_dispatch_threads_per_threadgroup`.
    DispatchThreadsPerThreadgroup,
    /// `[[thread_index_in_threadgroup]]`: `air.get_local_linear_id`.
    ThreadIndexInThreadgroup,
    /// `[[thread_index_in_simdgroup]]`: `air.get_thread_index_in_simdgroup`.
    ThreadIndexInSimdgroup,
    /// `[[simdgroup_index_in_threadgroup]]`: `air.get_simdgroup_index_in_threadgroup`.
    SimdgroupIndexInThreadgroup,
    /// `[[simdgroups_per_threadgroup]]`: `air.get_simdgroups_per_threadgroup`.
    SimdgroupsPerThreadgroup,
    /// `[[dispatch_simdgroups_per_threadgroup]]`: `air.get_dispatch_simdgroups_per_threadgroup`.
    DispatchSimdgroupsPerThreadgroup,
    /// `[[thread_index_in_quadgroup]]`: `air.get_thread_index_in_quadgroup`.
    ThreadIndexInQuadgroup,
    /// `[[quadgroup_index_in_threadgroup]]`: `air.get_quadgroup_index_in_threadgroup`.
    QuadgroupIndexInThreadgroup,
    /// `[[quadgroups_per_threadgroup]]`: `air.get_quadgroups_per_threadgroup`.
    QuadgroupsPerThreadgroup,
}

impl GlobalVar {
    /// Every global built-in with its AIR name (after `air.`, before the type) and whether it is
    /// a vector (`uint3`) read one lane at a time.
    pub const ALL: [(GlobalVar, &'static str, bool); 15] = [
        (GlobalVar::ThreadPositionInGrid, "get_global_id", true),
        (GlobalVar::ThreadPositionInThreadgroup, "get_local_id", true),
        (GlobalVar::ThreadgroupPositionInGrid, "get_group_id", true),
        (GlobalVar::ThreadsPerThreadgroup, "get_local_size", true),
        (GlobalVar::ThreadsPerGrid, "get_global_size", true),
        (GlobalVar::ThreadgroupsPerGrid, "get_num_groups", true),
        (
            GlobalVar::DispatchThreadsPerThreadgroup,
            "get_dispatch_threads_per_threadgroup",
            true,
        ),
        (
            GlobalVar::ThreadIndexInThreadgroup,
            "get_local_linear_id",
            false,
        ),
        (
            GlobalVar::ThreadIndexInSimdgroup,
            "get_thread_index_in_simdgroup",
            false,
        ),
        (
            GlobalVar::SimdgroupIndexInThreadgroup,
            "get_simdgroup_index_in_threadgroup",
            false,
        ),
        (
            GlobalVar::SimdgroupsPerThreadgroup,
            "get_simdgroups_per_threadgroup",
            false,
        ),
        (
            GlobalVar::DispatchSimdgroupsPerThreadgroup,
            "get_dispatch_simdgroups_per_threadgroup",
            false,
        ),
        (
            GlobalVar::ThreadIndexInQuadgroup,
            "get_thread_index_in_quadgroup",
            false,
        ),
        (
            GlobalVar::QuadgroupIndexInThreadgroup,
            "get_quadgroup_index_in_threadgroup",
            false,
        ),
        (
            GlobalVar::QuadgroupsPerThreadgroup,
            "get_quadgroups_per_threadgroup",
            false,
        ),
    ];

    /// The entry-parameter built-in the variable reads.
    pub fn builtin(self) -> Builtin {
        match self {
            GlobalVar::ThreadPositionInGrid => Builtin::ThreadPositionInGrid,
            GlobalVar::ThreadPositionInThreadgroup => Builtin::ThreadPositionInThreadgroup,
            GlobalVar::ThreadgroupPositionInGrid => Builtin::ThreadgroupPositionInGrid,
            GlobalVar::ThreadsPerThreadgroup => Builtin::ThreadsPerThreadgroup,
            GlobalVar::ThreadsPerGrid => Builtin::ThreadsPerGrid,
            GlobalVar::ThreadgroupsPerGrid => Builtin::ThreadgroupsPerGrid,
            GlobalVar::DispatchThreadsPerThreadgroup => Builtin::DispatchThreadsPerThreadgroup,
            GlobalVar::ThreadIndexInThreadgroup => Builtin::ThreadIndexInThreadgroup,
            GlobalVar::ThreadIndexInSimdgroup => Builtin::ThreadIndexInSimdgroup,
            GlobalVar::SimdgroupIndexInThreadgroup => Builtin::SimdgroupIndexInThreadgroup,
            GlobalVar::SimdgroupsPerThreadgroup => Builtin::SimdgroupsPerThreadgroup,
            GlobalVar::DispatchSimdgroupsPerThreadgroup => {
                Builtin::DispatchSimdgroupsPerThreadgroup
            }
            GlobalVar::ThreadIndexInQuadgroup => Builtin::ThreadIndexInQuadgroup,
            GlobalVar::QuadgroupIndexInThreadgroup => Builtin::QuadgroupIndexInThreadgroup,
            GlobalVar::QuadgroupsPerThreadgroup => Builtin::QuadgroupsPerThreadgroup,
        }
    }

    /// Whether the built-in is a vector (`uint3`), each call reading the lane its operand names.
    pub fn is_vector(self) -> bool {
        GlobalVar::ALL
            .iter()
            .any(|&(v, _, vector)| v == self && vector)
    }

    /// Whether Metal refuses a fragment function that reads this built-in as a global variable.
    /// Apple's compiler takes a global `thread_index_in_simdgroup`, `threads_per_simdgroup`
    /// (`air.get_simdgroup_size`) and `thread_index_in_quadgroup` in a fragment function, beyond
    /// the MSL specification's section 5.9 (kernel, mesh and object functions). The M4 (macOS
    /// 27.0.1) and the M3 Max (macOS 26.7) make the render pipeline of the first two and refuse the
    /// third's: "Encountered unlowered function call to air.get_thread_index_in_quadgroup.i32"
    /// (`.i16` for a `ushort` one; the M4 adds "Verify device or GPU architecture supports this
    /// feature"), while they run the same built-in as the fragment's parameter.
    pub fn metal_refuses_in_fragment(self) -> bool {
        self == GlobalVar::ThreadIndexInQuadgroup
    }
}

/// `air.get_<built-in>.<i16|i32>`: a read of a kernel built-in a program-scope variable declares
/// (Metal 3.1). A vector built-in's is `T (T)`, the operand the lane it reads; a scalar one's is
/// `T ()`. `T` is the variable's element: `i32` for `uint` and `uint3`, `i16` for `ushort` and
/// `ushort3` (`xcrun metal` 32023.921 reads `g.y` of `uint3 g [[threads_per_threadgroup]];` as
/// `air.get_local_size.i32(i32 1)`, and `ushort3`'s as `air.get_local_size.i16(i16 1)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GlobalBuiltin {
    /// The built-in read.
    pub var: GlobalVar,
    /// The element the name spells: `I16` or `I32`.
    pub ty: Elem,
}

impl GlobalBuiltin {
    pub(super) fn parse(rest: &str) -> Option<GlobalBuiltin> {
        let (var, ty) = GlobalVar::ALL
            .iter()
            .find_map(|&(var, name, _)| Some((var, rest.strip_prefix(name)?.strip_prefix('.')?)))?;
        let ty = NumTy::parse(ty)?;
        (matches!(ty.elem, Elem::I16 | Elem::I32) && !ty.is_vector())
            .then_some(GlobalBuiltin { var, ty: ty.elem })
    }

    /// The typed signature.
    pub fn signature(&self) -> Signature {
        let t = num(self.ty, 1);
        if self.var.is_vector() {
            Signature::new(t, &[t])
        } else {
            Signature::new(t, &[])
        }
    }
}
