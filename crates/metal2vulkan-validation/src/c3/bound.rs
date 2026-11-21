//! Machine-checkable bounds for C3's Apple readings.
//!
//! The user's decision of 2026-10-03 on the `apple_reading` entries: "Accept only within Apple's
//! bound". A reading is signed off only if every word in which the device differs from Metal is
//! within the accuracy Apple promises for what that word computes (MSL Tables 8.1, 8.2, and 8.3,
//! sections 8.1 and 8.5). So each reading declares its [`BoundedLane`]s: which output words compute
//! what MSL expression of which operands, under which math mode and in which format. [`check`]
//! recomputes every differing word from the case's own inputs: it reads each operand from the
//! case's buffer (or the thread-derived bit pattern the kernel builds), evaluates what Apple's rows
//! admit there ([`msl::admits`]), and judges the device's word, and Metal's as a check that the lane
//! describes the word. A word no lane covers, or one whose bound cannot be derived, is never within.
//!
//! Where Apple states no accuracy (fast `fmod` "Undefined", fast `sin` and `cos` "larger otherwise"
//! outside `[-pi, pi]`, the fast division outside its divisor domain, a NaN or INF under fast math),
//! any answer of the device conforms, by the user's rule, and [`UNSTATED_WORDS`] is the one flag
//! that would keep such words open instead. That reading covers the device's own answer, not the
//! product's approximation: where the product computes the word with an operation it records as a
//! near-equivalent of Metal's ([`near_equivalent`]: the product's own notes for the AIR operation
//! Metal compiles it to, [`AirOp`], on the row's venue profile and in the lane's math mode), the
//! user's standing rule keeps the word open until an exact path exists
//! ([`WordVerdict::NearEquivalent`]).
//!
//! Three notes record a difference that can be told value by value
//! ([`crate::observation_contract::value_difference`]): Metal's canonical NaN, its flush of
//! `float` denormals, and its rounding of a `half` to nearest even, each of which a device need
//! not share. A value within Apple's stated accuracy, or where Apple states none, that shows the
//! difference of such a note its row records is a near-equivalent as well, open with the exact
//! path that would close it; Apple's bound does not close it, because Metal's answer there is
//! reproducible. Every other value keeps its verdict.

pub mod msl;

use crate::case::BufferResource;
use crate::observation_contract::{value_difference, ValueDifference};
use base64::Engine as _;
use metal2vulkan::device::{DeviceProfile, FloatWidth};
use metal2vulkan::pipeline::PrecisionNote;
use metal2vulkan_air::intrinsics::AirIntrinsic;
use metal2vulkan_air::ir::{BinOp, FastMath};
use msl::{Admits, Expr, Format, Math, Prim};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// How a word where Apple states no accuracy is judged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnstatedWords {
    /// Within Apple's promise: Apple promises nothing there, so no answer is less precise than
    /// the promise.
    WithinPromise,
    /// Open: such a word keeps its reading from being signed off.
    Open,
}

/// The reading of the user's rule (confirmed by the orchestrator on 2026-10-03): a word where Apple
/// states no accuracy is within Apple's promise. Flip this one flag to keep such words open.
pub const UNSTATED_WORDS: UnstatedWords = UnstatedWords::WithinPromise;

fn one() -> usize {
    1
}

fn is_one(value: &usize) -> bool {
    *value == 1
}

/// Output words `first`, `first + stride`, ... (`count` of them); lane value `k` is word
/// `first + k * stride`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Words {
    pub first: usize,
    pub count: usize,
    #[serde(default = "one", skip_serializing_if = "is_one")]
    pub stride: usize,
}

impl Words {
    /// The lane value word `word` holds, if the lane covers it.
    fn index(&self, word: usize) -> Option<usize> {
        let offset = word.checked_sub(self.first)?;
        (offset % self.stride == 0 && offset / self.stride < self.count)
            .then_some(offset / self.stride)
    }
}

/// Where a lane's result sits in each of its 32-bit words.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Part {
    /// The whole word, a `float`.
    F32,
    /// Its low 16 bits, a `half`.
    F16Low,
    /// Its high 16 bits, a `half`.
    F16High,
}

impl Part {
    pub fn as_str(self) -> &'static str {
        match self {
            Part::F32 => "f32",
            Part::F16Low => "f16_low",
            Part::F16High => "f16_high",
        }
    }

    pub fn format(self) -> Format {
        match self {
            Part::F32 => Format::F32,
            Part::F16Low | Part::F16High => Format::F16,
        }
    }

    fn bits(self, word: u32) -> u32 {
        match self {
            Part::F32 => word,
            Part::F16Low => word & 0xffff,
            Part::F16High => word >> 16,
        }
    }
}

/// One operand of a lane: lane value `k` reads its element `first + k * stride`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Operand {
    /// An element of the case's buffer `binding` (its input bytes), in `format`.
    Buffer {
        binding: u32,
        format: Format,
        first: usize,
        #[serde(default = "one", skip_serializing_if = "is_one")]
        stride: usize,
    },
    /// The bit pattern `first + k * stride` in `format`: the operand a kernel builds from its
    /// thread index.
    Bits {
        format: Format,
        first: u32,
        stride: u32,
    },
}

/// Output words that each compute one MSL expression of the case's inputs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundedLane {
    pub words: Words,
    pub result: Part,
    /// `fast` (MSL Table 8.2, section 8.1's rule that a NaN or INF is undefined, and section
    /// 1.6.3's insignificant zero sign) or `precise` (Table 8.1); half-precision operations follow
    /// Table 8.3 either way.
    pub math: Math,
    /// What each word computes, over the operands `a`, `b`, ...: MSL operations by name
    /// (`atan(a)`, `powr(a, b)`, `sub(a, mul(b, c))`); the bound is their rows'.
    pub computes: String,
    pub operands: Vec<Operand>,
}

impl BoundedLane {
    /// The lane is well formed: words to cover, an expression of exactly its operands whose
    /// every operation has a row in MSL's tables for the lane's format and math, and bit-pattern
    /// operands that fit their format.
    pub fn check(&self) -> Result<(), String> {
        if self.words.count == 0 || self.words.stride == 0 {
            return Err("a lane covers no word".into());
        }
        let expr = Expr::parse(&self.computes)?;
        if expr.operands() != self.operands.len() {
            return Err(format!(
                "{:?} reads {} operands, the lane lists {}",
                self.computes,
                expr.operands(),
                self.operands.len()
            ));
        }
        let format = self.result.format();
        let mut calls = vec![];
        expr.calls(&mut calls);
        if calls.is_empty() {
            return Err(format!("{:?} calls no MSL operation", self.computes));
        }
        for prim in calls {
            if msl::row(prim, format, self.math).is_none() {
                return Err(format!(
                    "MSL's tables state no {} for {} under {} math",
                    prim.name(),
                    format.as_str(),
                    self.math.as_str()
                ));
            }
        }
        for operand in &self.operands {
            match operand {
                Operand::Buffer { stride: 0, .. } => return Err("an operand's stride is 0".into()),
                Operand::Buffer { .. } => {}
                Operand::Bits {
                    format,
                    first,
                    stride,
                } => {
                    let last =
                        u64::from(*first) + u64::from(*stride) * (self.words.count as u64 - 1);
                    let width = match format {
                        Format::F32 => u64::from(u32::MAX),
                        Format::F16 => u64::from(u16::MAX),
                    };
                    if last > width {
                        return Err(format!(
                            "a bit-pattern operand reaches {last:#x}, past {}",
                            format.as_str()
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Each operation's row in MSL's tables, as the report states the lane's bound.
    pub fn bound(&self) -> String {
        let format = self.result.format();
        let mut calls = vec![];
        if let Ok(expr) = Expr::parse(&self.computes) {
            expr.calls(&mut calls);
        }
        calls
            .into_iter()
            .map(|prim| match msl::row(prim, format, self.math) {
                Some(row) => format!("{} ({}): {}", prim.name(), format.as_str(), row.describe()),
                None => format!("{}: no row", prim.name()),
            })
            .collect::<Vec<_>>()
            .join("; ")
    }

    fn covers(&self, word: usize) -> Option<usize> {
        self.words.index(word)
    }
}

/// Every lane is well formed and no two cover the same part of a word.
pub fn check_lanes(lanes: &[BoundedLane]) -> Result<(), String> {
    let mut covered = BTreeSet::new();
    for lane in lanes {
        lane.check()
            .map_err(|error| format!("lane {:?}: {error}", lane.computes))?;
        for k in 0..lane.words.count {
            let word = lane.words.first + k * lane.words.stride;
            let parts: &[Part] = match lane.result {
                Part::F32 => &[Part::F16Low, Part::F16High],
                Part::F16Low => &[Part::F16Low],
                Part::F16High => &[Part::F16High],
            };
            for part in parts {
                if !covered.insert((word, *part)) {
                    return Err(format!("word {word} is covered by two lanes"));
                }
            }
        }
    }
    Ok(())
}

/// The AIR operation Metal compiles an MSL operation to: the ABI the product reads, so the only
/// mapping validation keeps of the product's input ([`AirOp::of`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AirOp {
    /// A float binary operator, the LLVM instruction.
    Binop(BinOp),
    /// A call of the `air` math function of this name.
    Call(String),
}

impl AirOp {
    /// The AIR operation Metal compiles `prim` to in `format` under `math` (observed with Apple's
    /// `metal` 32023.921 for every [`Prim`] in both formats, with `-ffast-math` and
    /// `-fno-fast-math`): an operator is its LLVM instruction (a reciprocal `1 / x` is a
    /// division), and a function is the call `air.<name>.<type>`, with the `fast_` prefix under
    /// fast math on `float` only (a fast-math `half` function and `sign` keep the precise
    /// spelling).
    pub fn of(prim: Prim, format: Format, math: Math) -> AirOp {
        let binop = match prim {
            Prim::Add => Some(BinOp::FAdd),
            Prim::Sub => Some(BinOp::FSub),
            Prim::Mul => Some(BinOp::FMul),
            Prim::Div | Prim::Recip => Some(BinOp::FDiv),
            _ => None,
        };
        if let Some(op) = binop {
            return AirOp::Binop(op);
        }
        let fast = math == Math::Fast && format == Format::F32 && prim != Prim::Sign;
        AirOp::Call(format!(
            "air.{}{}.{}",
            if fast { "fast_" } else { "" },
            prim.name(),
            format.as_str()
        ))
    }

    /// The precision notes the product's lowering of this operation on `format` records under
    /// `math` on profile `profile`: the product's own answer (`PrecisionNote::of_float_operator`,
    /// or `PrecisionNote::of_math_call` of the call's name as [`AirIntrinsic::parse`] decodes it).
    /// Under fast math Metal's calls and operators carry every fast-math flag (`fast`); otherwise
    /// they carry none, and a division without them is Metal's precise one. An error where the
    /// product does not lower the operation as a float operation.
    pub fn product_notes(
        &self,
        format: Format,
        math: Math,
        profile: &DeviceProfile,
    ) -> Result<BTreeSet<PrecisionNote>, String> {
        let fmf = match math {
            Math::Fast => FastMath::ALL
                .iter()
                .fold(FastMath::NONE, |all, (flag, _)| all.union(*flag)),
            Math::Precise => FastMath::NONE,
        };
        let notes = match self {
            AirOp::Binop(op) => PrecisionNote::of_float_operator(
                *op,
                match format {
                    Format::F32 => FloatWidth::F32,
                    Format::F16 => FloatWidth::F16,
                },
                fmf,
                profile,
            ),
            AirOp::Call(name) => match AirIntrinsic::parse(name) {
                Some(AirIntrinsic::Math(call)) => PrecisionNote::of_math_call(call, fmf, profile),
                _ => None,
            },
        };
        notes.ok_or_else(|| format!("the product does not lower {self} as a float operation"))
    }
}

impl std::fmt::Display for AirOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AirOp::Binop(op) => write!(f, "{}", format!("{op:?}").to_lowercase()),
            AirOp::Call(name) => f.write_str(name),
        }
    }
}

/// The note under which the product records its lowering of `prim` (in `format` under `math`)
/// as a near-equivalent of Metal's, if it does, and the AIR operation it lowers
/// ([`AirOp::of`]). `DeviceArithmetic` covers operations at Vulkan's precision rather than
/// Metal's (a near-equivalent): a float division, a square root or remainder, and a function the
/// product computes with a division of its own (fast `float` `tanh` and `atan2`). Whether the
/// product records it is the product's answer ([`AirOp::product_notes`]) on the row's venue profile
/// `profile`.
///
/// The device's transcendentals (`DeviceTranscendentals`) are the device's own answer, which the
/// user accepts within Apple's bound, so they are no near-equivalent.
pub fn near_equivalent(
    prim: Prim,
    format: Format,
    math: Math,
    profile: &DeviceProfile,
) -> Result<Option<(PrecisionNote, AirOp)>, String> {
    let (op, notes) = lowering_notes(prim, format, math, profile)?;
    Ok(notes
        .contains(&PrecisionNote::DeviceArithmetic)
        .then_some((PrecisionNote::DeviceArithmetic, op)))
}

/// The AIR operation Metal compiles `prim` to in `format` under `math` ([`AirOp::of`]), and
/// every note the product's lowering of it records on profile `profile`
/// ([`AirOp::product_notes`]).
pub fn lowering_notes(
    prim: Prim,
    format: Format,
    math: Math,
    profile: &DeviceProfile,
) -> Result<(AirOp, BTreeSet<PrecisionNote>), String> {
    let op = AirOp::of(prim, format, math);
    let notes = op.product_notes(format, math, profile)?;
    Ok((op, notes))
}

/// One differing value, as [`value_near_equivalent`] reads it.
struct Difference<'a> {
    format: Format,
    /// The infinitely precise value of every operand, constant, and operation of the lane's
    /// expression, its result last ([`Expr::exact_values`]).
    exact: &'a [f64],
    metal: f64,
    device: f64,
}

impl Difference<'_> {
    /// Whether the value differs the way `kind` lets it differ.
    fn shows(&self, kind: ValueDifference) -> bool {
        match kind {
            ValueDifference::NanEncoding => self.metal.is_nan() && self.device.is_nan(),
            ValueDifference::Denormal => self
                .exact
                .iter()
                .chain([&self.metal, &self.device])
                .any(|value| self.format.is_subnormal(*value)),
        }
    }
}

/// Why a differing value is a near-equivalent its row records, if it is: for each note the row
/// records that has a [`ValueDifference`], the value shows that difference, and the note is the
/// value's: `DeviceNanEncoding`, `DeviceDenormals`: the product records them per operation, so an
/// operation the lane computes must be one whose lowering records the note ([`lowering_notes`];
/// `float` forms only).
///
/// Each reason names the exact path that would close the value: the canonicalizing or flushing
/// select (or `DenormFlushToZero` where a profile offers it).
fn value_near_equivalent(
    recorded: &BTreeSet<PrecisionNote>,
    expr: &Expr,
    math: Math,
    profile: &DeviceProfile,
    difference: &Difference<'_>,
) -> Result<Option<String>, String> {
    let shown = recorded
        .iter()
        .filter_map(|note| {
            value_difference(*note)
                .filter(|kind| difference.shows(*kind))
                .map(|kind| (*note, kind))
        })
        .collect::<Vec<_>>();
    if shown.is_empty() {
        return Ok(None);
    }
    let mut prims = vec![];
    expr.calls(&mut prims);
    let calls = prims
        .into_iter()
        .map(|prim| {
            lowering_notes(prim, difference.format, math, profile)
                .map(|(op, notes)| (prim, op, notes))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut reasons = vec![];
    for (note, kind) in shown {
        let ops = calls
            .iter()
            .filter(|(_, _, notes)| notes.contains(&note))
            .collect::<Vec<_>>();
        if ops.is_empty() {
            continue;
        }
        let named = ops
            .iter()
            .map(|(prim, op, _)| format!("{op} (Metal's {})", prim.name()))
            .collect::<Vec<_>>()
            .join(" and ");
        reasons.push(match kind {
            ValueDifference::NanEncoding => format!(
                "both answers are NaNs, and the row records {note:?}, which the product records \
                 for its lowering of {named}: Metal answers the canonical 0x7fc00000, the device \
                 its own encoding. A near-equivalent, open until an exact path exists: a \
                 canonicalizing select after each operation"
            ),
            ValueDifference::Denormal => format!(
                "a subnormal is an operand, an operation's exact value, or an answer, and the row \
                 records {note:?}, which the product records for its lowering of {named}: Metal \
                 flushes float denormals, the device need not. A near-equivalent, open until an \
                 exact path exists: a flushing select after each operation, or \
                 DenormFlushToZero where a profile offers it"
            ),
        });
    }
    Ok((!reasons.is_empty()).then(|| reasons.join("; and ")))
}

/// The verdict on one differing value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WordVerdict {
    /// Within the accuracy Apple states.
    Within,
    /// Apple states no accuracy here (the text cites where).
    Unstated(String),
    /// The difference is a near-equivalent the row records, not the device's latitude: where
    /// Apple states no accuracy, the row's modules record the product's lowering of an operation
    /// the word computes as a near-equivalent of Metal's ([`near_equivalent`]); within Apple's
    /// stated accuracy or where it states none, the value shows the difference of a note the row
    /// records that can be told value by value ([`crate::observation_contract::value_difference`]).
    /// The user's standing decision of 2026-10-03 keeps a recorded near-equivalent open until an
    /// exact path exists, whatever [`UNSTATED_WORDS`] says.
    NearEquivalent(String),
    /// Outside Apple's accuracy, by the measure the text states.
    Outside(String),
    /// No lane covers this part of the word.
    Uncovered,
    /// No bound could be derived, or the lane does not describe the word (Metal's own value is
    /// outside what it admits).
    NotCheckable(String),
}

impl WordVerdict {
    pub fn label(&self) -> &'static str {
        match self {
            WordVerdict::Within => "within",
            WordVerdict::Unstated(_) => "unstated",
            WordVerdict::NearEquivalent(_) => "near_equivalent",
            WordVerdict::Outside(_) => "outside",
            WordVerdict::Uncovered => "uncovered",
            WordVerdict::NotCheckable(_) => "not_checkable",
        }
    }

    pub fn detail(&self) -> &str {
        match self {
            WordVerdict::Unstated(text)
            | WordVerdict::NearEquivalent(text)
            | WordVerdict::Outside(text)
            | WordVerdict::NotCheckable(text) => text,
            WordVerdict::Within | WordVerdict::Uncovered => "",
        }
    }

    /// Whether this value keeps its reading open under `unstated`.
    pub fn keeps_open(&self, unstated: UnstatedWords) -> bool {
        match self {
            WordVerdict::Within => false,
            WordVerdict::Unstated(_) => unstated == UnstatedWords::Open,
            WordVerdict::NearEquivalent(_)
            | WordVerdict::Outside(_)
            | WordVerdict::Uncovered
            | WordVerdict::NotCheckable(_) => true,
        }
    }
}

/// One differing value: a word, or one half of it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WordCheck {
    pub word: usize,
    pub part: Part,
    pub verdict: WordVerdict,
}

/// Every value in which one observation differs from Metal, judged against Apple's accuracy.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BoundCheck {
    pub words: Vec<WordCheck>,
}

impl BoundCheck {
    /// Values with the verdict `label` ([`WordVerdict::label`]).
    pub fn count(&self, label: &str) -> usize {
        self.words
            .iter()
            .filter(|word| word.verdict.label() == label)
            .count()
    }

    /// The values that keep the reading open under `unstated`.
    pub fn open(&self, unstated: UnstatedWords) -> impl Iterator<Item = &WordCheck> {
        self.words
            .iter()
            .filter(move |word| word.verdict.keeps_open(unstated))
    }

    /// Every differing value is within Apple's promise under `unstated`.
    pub fn clean(&self, unstated: UnstatedWords) -> bool {
        self.open(unstated).next().is_none()
    }
}

fn word_at(bytes: &[u8], index: usize) -> Option<u32> {
    bytes
        .get(index * 4..index * 4 + 4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
}

/// The input bytes of the case's buffer `binding`.
fn buffer_bytes(buffers: &[BufferResource], binding: u32) -> Result<Vec<u8>, String> {
    let buffer = buffers
        .iter()
        .find(|buffer| buffer.binding == binding)
        .ok_or_else(|| format!("the case has no buffer {binding}"))?;
    let b64 = buffer
        .bytes_b64
        .as_ref()
        .or(buffer.initial_bytes_b64.as_ref())
        .ok_or_else(|| format!("buffer {binding} states no input bytes"))?;
    base64::engine::general_purpose::STANDARD
        .decode(b64)
        .map_err(|error| format!("buffer {binding}: {error}"))
}

fn operand_value(buffers: &[BufferResource], operand: &Operand, k: usize) -> Result<f64, String> {
    match operand {
        Operand::Buffer {
            binding,
            format,
            first,
            stride,
        } => {
            let bytes = buffer_bytes(buffers, *binding)?;
            let index = first + k * stride;
            let bits = match format {
                Format::F32 => word_at(&bytes, index),
                Format::F16 => bytes
                    .get(index * 2..index * 2 + 2)
                    .map(|half| u32::from(u16::from_le_bytes([half[0], half[1]]))),
            };
            bits.map(|bits| format.decode(bits)).ok_or_else(|| {
                format!(
                    "element {index} is past buffer {binding}'s {} bytes",
                    bytes.len()
                )
            })
        }
        Operand::Bits {
            format,
            first,
            stride,
        } => u32::try_from(k)
            .ok()
            .and_then(|k| stride.checked_mul(k))
            .and_then(|step| first.checked_add(step))
            .map(|bits| format.decode(bits))
            .ok_or_else(|| format!("bit pattern {first} + {stride} * {k} overflows")),
    }
}

/// Judge `device` (one part of a differing word) against what `lane` admits at its value `k`; the
/// row's modules, compiled for profile `profile`, record the precision notes `recorded`.
fn judge(
    lane: &BoundedLane,
    buffers: &[BufferResource],
    recorded: &BTreeSet<PrecisionNote>,
    profile: &DeviceProfile,
    k: usize,
    metal: u32,
    device: u32,
) -> WordVerdict {
    let format = lane.result.format();
    let expr = match Expr::parse(&lane.computes) {
        Ok(expr) => expr,
        Err(error) => return WordVerdict::NotCheckable(error),
    };
    let values = match lane
        .operands
        .iter()
        .map(|operand| operand_value(buffers, operand, k))
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(values) => values,
        Err(error) => return WordVerdict::NotCheckable(error),
    };
    let bindings = values
        .iter()
        .map(|value| msl::operand(*value, format, lane.math))
        .collect::<Vec<_>>();
    let admits = match msl::admits(&expr, &bindings, format, lane.math) {
        Ok(admits) => admits,
        Err(error) => return WordVerdict::NotCheckable(error),
    };
    let bits = (lane.result.bits(metal), lane.result.bits(device));
    let (metal, device) = (format.decode(bits.0), format.decode(bits.1));
    if !admits.admits(metal) {
        return WordVerdict::NotCheckable(format!(
            "Metal's {metal:e} is outside what {} admits at {values:?}: the lane does not \
             describe this word",
            lane.computes
        ));
    }
    // A value Apple's bound would close stays open where it shows a recorded near-equivalent's
    // difference.
    let value_verdict = |closed: WordVerdict, prefix: &str| {
        let mut exact = vec![];
        expr.exact_values(&values, &mut exact);
        let difference = Difference {
            format,
            exact: &exact,
            metal,
            device,
        };
        match value_near_equivalent(recorded, &expr, lane.math, profile, &difference) {
            Ok(Some(why)) => WordVerdict::NearEquivalent(format!("{prefix}; but {why}")),
            Ok(None) => closed,
            Err(error) => WordVerdict::NotCheckable(error),
        }
    };
    match admits {
        Admits::Unstated(why) => {
            let mut calls = vec![];
            expr.calls(&mut calls);
            for prim in calls {
                match near_equivalent(prim, format, lane.math, profile) {
                    Ok(Some((note, op))) if recorded.contains(&note) => {
                        return WordVerdict::NearEquivalent(format!(
                            "{why}; but the row records {note:?}, which the product records for \
                             its lowering of {op} (Metal's {}): a near-equivalent, open until an \
                             exact path exists",
                            prim.name()
                        ))
                    }
                    Ok(_) => {}
                    Err(error) => return WordVerdict::NotCheckable(error),
                }
            }
            value_verdict(WordVerdict::Unstated(why.clone()), &why)
        }
        admits if admits.admits(device) => {
            value_verdict(WordVerdict::Within, "within Apple's stated accuracy")
        }
        Admits::Values { ranges, .. } => {
            let reference = expr.reference(&values);
            let ulps = if reference.is_finite() && device.is_finite() {
                format!(
                    "{:.1} ulp",
                    (device - reference).abs() / format.ulp(reference)
                )
            } else {
                "not finite".into()
            };
            let nearest = ranges
                .iter()
                .map(|range| format!("[{:e}, {:e}]", range.lo, range.hi))
                .collect::<Vec<_>>()
                .join(" or ");
            WordVerdict::Outside(format!(
                "{} at {values:?}: the device's {device:e} is {ulps} from {reference:e}; Apple \
                 admits {nearest} ({})",
                lane.computes,
                lane.bound()
            ))
        }
    }
}

/// Judge every value in which `output` differs from Metal's `golden` against `lanes`, reading the
/// operands from the case's `buffers`; the row's executed modules, compiled for its venue's
/// profile `profile`, record the precision notes `recorded`. A differing word that half lanes
/// cover is judged by each of its halves that differs; any other differing word is one value.
pub fn check(
    lanes: &[BoundedLane],
    buffers: &[BufferResource],
    recorded: &BTreeSet<PrecisionNote>,
    profile: &DeviceProfile,
    golden: &[u8],
    output: &[u8],
) -> BoundCheck {
    let mut words = vec![];
    for (index, (metal, device)) in golden.chunks(4).zip(output.chunks(4)).enumerate() {
        if metal == device {
            continue;
        }
        let (Some(metal), Some(device)) = (word_at(golden, index), word_at(output, index)) else {
            words.push(WordCheck {
                word: index,
                part: Part::F32,
                verdict: WordVerdict::NotCheckable("a partial word".into()),
            });
            continue;
        };
        let lane_for = |part: Part| {
            lanes
                .iter()
                .filter(|lane| lane.result == part)
                .find_map(|lane| lane.covers(index).map(|k| (lane, k)))
        };
        // A word with half lanes is judged by its halves; any other word is one value.
        let halves = lane_for(Part::F16Low).is_some() || lane_for(Part::F16High).is_some();
        let parts: &[Part] = if lane_for(Part::F32).is_none() && halves {
            &[Part::F16Low, Part::F16High]
        } else {
            &[Part::F32]
        };
        for &part in parts {
            if part != Part::F32 && part.bits(metal) == part.bits(device) {
                continue;
            }
            let verdict = match lane_for(part) {
                Some((lane, k)) => judge(lane, buffers, recorded, profile, k, metal, device),
                None => WordVerdict::Uncovered,
            };
            words.push(WordCheck {
                word: index,
                part,
                verdict,
            });
        }
    }
    BoundCheck { words }
}

#[cfg(test)]
mod tests;
