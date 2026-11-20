//! A texture the function constants leave out takes no slot, even where its slot expression
//! names a live texture's slot.
//!
//! Metal compiles `[[texture(fc_expr)]]` into a running sum over the arguments the constants
//! enable, a global the module's `air.static_init` constructor computes. The sum is an argument's
//! slot only while the argument is enabled; for one the constants leave out, the same global holds
//! wherever the sum stopped, which is a live argument's slot. The old engine read it anyway and put
//! two resources on one binding in 338 of 14579 corpus sources, 250 of them for this reason alone
//! (`tests/gated_texture_descriptor_slots.rs`, deleted with the old engine in 0.2.0).
//!
//! The engine drops a gated-off argument before its slot global resolves (`pipeline::interface`'s
//! entry). A write to a gated-off texture is dropped, as Metal drops it, and an observed read of
//! one is an absent read, which fails the command buffer where it runs
//! (`src/tests/engine_function_constant_interface.rs`), so each body here writes through each
//! texture only under its own constant, as a real shader of alternatives does:
//!
//! - **Summed slots.** A `texture2d_array` gated by constant `x` comes first, then a live
//!   `texture2d`; the constructor computes the running sum, so with `x` false both slot globals
//!   hold 0. Off, the function has one texture slot, 0, the live texture's; on, the two take
//!   slots 0 and 1. Each module passes L3. Written unconditionally with `x` false, the gated
//!   texture's write is dropped rather than landing on the live texture's slot, where the old
//!   engine's store landed, and a read of it is an absent read.
//! - **Literal slots**, the control: the two textures written down at `[[texture(0)]]` under `x`
//!   and `!x`, Metal's own spelling of mutually exclusive alternatives. Under each value the one
//!   alternative left keeps slot 0, so "drop every gated texture" cannot pass the test above.
//!
//! `llvm-as` assembles the templates; without it the tests skip visibly.

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::Error;
use crate::library::{CompiledFunction, FunctionConstantValue, FunctionConstantValues, Library};
use crate::metal::DataType;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::sym::{SlotClass, Stage};
use crate::pipeline::testing::{link_and_validate, llvm_assemble, AssembleError};
use crate::pipeline::{FunctionDescriptor, LoweringInputs};
use crate::Budget;

const HEADER: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"
"#;

const TRAILER: &str = r#"!air.version = !{!93}
!air.language_version = !{!94}
!93 = !{i32 2, i32 8, i32 0}
!94 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// The constant, the two gate globals the constructor computes from it, and the kernel's body:
/// it writes the array texture under `x` and the 2D texture under `LIVE_GATE` (`true` for the
/// summed form, `!x` for the literal one).
const COMMON: &str = r#"@x.MTL_FC_INIT_0_b = internal addrspace(2) externally_initialized constant i8 undef, section "air.fc_initializer", align 1
@gated_enabled = internal addrspace(2) global i8 0, align 1
@live_enabled = internal addrspace(2) global i8 0, align 1

declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)
declare void @air.write_texture_2d_array.v4f32(ptr addrspace(1), <2 x i32>, i32, <4 x float>, i32, i32)

define void @blit(ptr addrspace(1) %gated, ptr addrspace(1) %live) {
entry:
  %xb = load i8, ptr addrspace(2) @x.MTL_FC_INIT_0_b, align 1
  %x = icmp ne i8 %xb, 0
  %not_x = xor i1 %x, true
  br i1 %x, label %write_gated, label %maybe_live

write_gated:
  tail call void @air.write_texture_2d_array.v4f32(ptr addrspace(1) %gated, <2 x i32> zeroinitializer, i32 0, <4 x float> zeroinitializer, i32 0, i32 2)
  br label %maybe_live

maybe_live:
  br i1 LIVE_GATE, label %write_live, label %done

write_live:
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) %live, <2 x i32> zeroinitializer, <4 x float> zeroinitializer, i32 0, i32 2)
  br label %done

done:
  ret void
}

!air.kernel = !{!0}
!air.function_constants = !{!7}
!0 = !{ptr @blit, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!5 = !{ptr addrspace(2) @gated_enabled, !"bool", !"gated_enabled"}
!6 = !{ptr addrspace(2) @live_enabled, !"bool", !"live_enabled"}
!7 = !{ptr addrspace(2) @x.MTL_FC_INIT_0_b, !"bool", !"x", i32 0}
"#;

/// The summed form: the constructor computes the running sum, so the gated texture's global is 0
/// and the live one's is 0 with `x` false, 1 with `x` true.
const SUMMED: &str = r#"@gated_location = internal addrspace(2) global i32 0, align 4
@live_location = internal addrspace(2) global i32 0, align 4

define internal void @_GLOBAL__sub_I_alternatives() section "air.static_init" {
entry:
  %xb = load i8, ptr addrspace(2) @x.MTL_FC_INIT_0_b, align 1
  %x = icmp ne i8 %xb, 0
  %on = zext i1 %x to i8
  store i8 %on, ptr addrspace(2) @gated_enabled, align 1
  store i8 1, ptr addrspace(2) @live_enabled, align 1
  store i32 0, ptr addrspace(2) @gated_location, align 4
  %after = zext i1 %x to i32
  store i32 %after, ptr addrspace(2) @live_location, align 4
  ret void
}

!3 = !{i32 0, !"air.function_constant", !5, !"air.texture", !"air.location_index", ptr addrspace(2) @gated_location, i32 1, !"air.write", !"air.arg_type_name", !"texture2d_array<float, write>", !"air.arg_name", !"gated"}
!4 = !{i32 1, !"air.function_constant", !6, !"air.texture", !"air.location_index", ptr addrspace(2) @live_location, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"live"}
"#;

/// The literal form: both at `[[texture(0)]]`, gated by `x` and `!x`.
const LITERAL: &str = r#"define internal void @_GLOBAL__sub_I_alternatives() section "air.static_init" {
entry:
  %xb = load i8, ptr addrspace(2) @x.MTL_FC_INIT_0_b, align 1
  %x = icmp ne i8 %xb, 0
  %on = zext i1 %x to i8
  %off = xor i8 %on, 1
  store i8 %on, ptr addrspace(2) @gated_enabled, align 1
  store i8 %off, ptr addrspace(2) @live_enabled, align 1
  ret void
}

!3 = !{i32 0, !"air.function_constant", !5, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d_array<float, write>", !"air.arg_name", !"gated"}
!4 = !{i32 1, !"air.function_constant", !6, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"live"}
"#;

/// `llvm-as` of `text`, through a scratch file that is removed at once. `None` when `llvm-as` is
/// not installed (the reason is printed).
fn assemble(text: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "m2v-gated-texture-slots-{}-{}.ll",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&path, text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let assembled = llvm_assemble(&path);
    std::fs::remove_file(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    match assembled {
        Ok(bitcode) => Some(bitcode),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejects an input:\n{why}\n{text}"),
    }
}

fn device(stem: &str) -> Device {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let description = DeviceDescription::from_khronos_json(&text, None).expect(stem);
    Device::from_queries(description, DevicePolicy::default()).expect(stem)
}

/// Profiles that bind textures by descriptor and by heap index.
const PROFILES: [&str; 2] = ["nvidia-rtx5080-laptop", "moltenvk-m3"];

/// The kernel of `form` with constant `x` set to `x`, compiled on `stem`.
fn compile(
    common: &str,
    form: &str,
    x: bool,
    stem: &str,
) -> Option<(Device, Result<CompiledFunction, Error>)> {
    let bitcode = assemble(&format!("{HEADER}\n{common}\n{form}\n{TRAILER}"))?;
    let device = device(stem);
    let library = Library::from_air(&bitcode).expect("the kernel decodes");
    let mut descriptor = FunctionDescriptor::new(library.function("blit").expect("the kernel"));
    descriptor.constants = FunctionConstantValues::defaults().with_index(
        0,
        FunctionConstantValue::new(DataType::Bool, vec![u8::from(x)]),
    );
    let compiled = device.compile_function(&descriptor, &LoweringInputs::new(), &Budget::default());
    Some((device, compiled))
}

/// The kernel of `form` with constant `x` set to `x`, compiled and checked at L3 on `stem`: its
/// texture slots' indices.
fn texture_slots(form: &str, live_gate: &str, x: bool, stem: &str) -> Option<Vec<u32>> {
    let (device, compiled) = compile(&COMMON.replace("LIVE_GATE", live_gate), form, x, stem)?;
    let compiled = compiled.unwrap_or_else(|e| panic!("{stem}, x = {x}: {e:?}"));
    link_and_validate(compiled.artifact(), device.profile())
        .unwrap_or_else(|e| panic!("{stem}, x = {x}: L3: {e:?}"));
    Some(
        compiled
            .artifact()
            .interface
            .slots
            .keys()
            .filter(|slot| slot.class == SlotClass::Texture)
            .map(|slot| slot.index)
            .collect(),
    )
}

#[test]
fn a_summed_slot_the_constants_leave_out_takes_no_slot() {
    for stem in PROFILES {
        let Some(off) = texture_slots(SUMMED, "true", false, stem) else {
            return;
        };
        assert_eq!(
            off,
            vec![0],
            "{stem}: x off leaves only the live texture, at slot 0"
        );
        let on = texture_slots(SUMMED, "true", true, stem);
        assert_eq!(on, Some(vec![0, 1]), "{stem}: x on declares both");
    }
}

#[test]
fn a_literal_slot_keeps_its_slot_under_its_own_constant() {
    for stem in PROFILES {
        for x in [false, true] {
            let Some(slots) = texture_slots(LITERAL, "%not_x", x, stem) else {
                return;
            };
            assert_eq!(
                slots,
                vec![0],
                "{stem}, x = {x}: the remaining alternative keeps slot 0"
            );
        }
    }
}

/// The number of `OpImageWrite` instructions in SPIR-V `words`.
fn image_writes(words: &[u32]) -> usize {
    let mut n = 0;
    let mut i = 5;
    while let Some(w) = words.get(i) {
        n += usize::from(w & 0xFFFF == 99);
        i += ((w >> 16) as usize).max(1);
    }
    n
}

/// Metal drops a write to a texture its function constant removes (measured on an M3 Max and an M4,
/// with its layers off and on), so an unconditional write through the left-out texture compiles to
/// nothing: the function keeps the live texture's slot alone, and the only image write left is the
/// live one, never the gated write landing on the live texture's slot.
#[test]
fn a_write_through_the_left_out_summed_texture_is_dropped() {
    let unconditional = COMMON.replace("LIVE_GATE", "true").replace(
        "br i1 %x, label %write_gated, label %maybe_live",
        "br label %write_gated",
    );
    for stem in PROFILES {
        let Some((device, compiled)) = compile(&unconditional, SUMMED, false, stem) else {
            return;
        };
        let compiled = compiled.unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        let (_, words) = link_and_validate(compiled.artifact(), device.profile())
            .unwrap_or_else(|e| panic!("{stem}: L3: {e:?}"));
        let slots: Vec<u32> = compiled
            .artifact()
            .interface
            .slots
            .keys()
            .filter(|slot| slot.class == SlotClass::Texture)
            .map(|slot| slot.index)
            .collect();
        assert_eq!(slots, vec![0], "{stem}: only the live texture takes a slot");
        assert_eq!(
            image_writes(&words),
            1,
            "{stem}: only the live write is left"
        );
    }
}

/// A query of the left-out texture is an absent read: Metal gives it no value, so the kernel marks
/// the command buffer where it runs and takes zero, and the left-out texture takes no slot.
#[test]
fn a_read_through_the_left_out_texture_is_an_absent_read() {
    let reading = COMMON
        .replace("LIVE_GATE", "true")
        .replace(
            "br i1 %x, label %write_gated, label %maybe_live",
            "br label %write_gated",
        )
        .replace(
            "  tail call void @air.write_texture_2d_array.v4f32(ptr addrspace(1) %gated, <2 x i32> zeroinitializer, i32 0, <4 x float> zeroinitializer, i32 0, i32 2)\n",
            "  %w = tail call i32 @air.get_width_texture_2d_array(ptr addrspace(1) %gated, i32 0)\n  %wf = uitofp i32 %w to float\n  %wv = insertelement <4 x float> zeroinitializer, float %wf, i32 0\n  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) %live, <2 x i32> zeroinitializer, <4 x float> %wv, i32 0, i32 2)\n",
        )
        .replace(
            "declare void @air.write_texture_2d.v4f32(",
            "declare i32 @air.get_width_texture_2d_array(ptr addrspace(1), i32)\ndeclare void @air.write_texture_2d.v4f32(",
        );
    assert!(reading.contains("@air.get_width_texture_2d_array(ptr addrspace(1) %gated"));
    for stem in PROFILES {
        let Some((device, compiled)) = compile(&reading, SUMMED, false, stem) else {
            return;
        };
        let compiled = compiled.unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        let fact = RuntimeFact::AbsentArgumentRead {
            stage: Stage::Kernel,
        };
        assert!(
            compiled.artifact().facts.contains(&fact),
            "{stem}: {:?}",
            compiled.artifact().facts
        );
        link_and_validate(compiled.artifact(), device.profile())
            .unwrap_or_else(|e| panic!("{stem}: L3: {e:?}"));
        let slots: Vec<u32> = compiled
            .artifact()
            .interface
            .slots
            .keys()
            .filter(|slot| slot.class == SlotClass::Texture)
            .map(|slot| slot.index)
            .collect();
        assert_eq!(slots, vec![0], "{stem}: only the live texture takes a slot");
    }
}
