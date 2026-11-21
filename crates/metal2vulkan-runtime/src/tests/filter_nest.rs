//! A filter nest's texel offsets, under a device cfg on every device of the cfg's kind: a kernel
//! samples an 8 x 8 `RGBA8Unorm` texture through a nearest, clamped constexpr sampler at each of
//! its 64 texel centres, in a 3x3 nest whose counters make the texel offset, summing the red
//! channel. Where the constants decide the counters, the translator unrolls the nest and every
//! offset is a constant; where the inner loop's end is loaded (2: the same runs), the offset stays
//! dynamic: the sample's own `Offset` operand where device creation enabled `maintenance8`, and a
//! switch over its values where it did not, which a translator told of no `maintenance8` (the
//! tests' description of the device) shows on a device that has it. Every form writes the same
//! sums, the nearest filter's, and the cache directory keeps the operand's and the switch's
//! lowerings apart.
//!
//! `llvm-as` assembles the kernels; without it the test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::floor::MAINTENANCE8;
    use crate::tests::command::on_device::{
        each_device, each_device_with, finished, within, Arena,
    };
    use crate::tests::devices_of_the_cfgs_kind;
    use crate::{
        ComputePipelineDescriptor, Device, DeviceOptions, DispatchType, FunctionDescriptor, Gauges,
        IndexClass, Library, NumberedSampler, PixelFormat, Region, StageSource, TextureDescriptor,
        TextureType, TextureUsage,
    };
    use metal2vulkan::device::{DeviceDescription, VkFeature};
    use metal2vulkan::Budget;
    use metal2vulkan_devtools::{llvm_assemble_text, AssembleError};
    use std::collections::BTreeMap;

    /// SPIR-V's `OpSwitch`.
    const OP_SWITCH: u32 = 251;

    /// The texture's side.
    const SIDE: u32 = 8;
    /// A constexpr sampler's words: nearest, clamp to edge, normalized.
    const NEAREST_WORDS: u64 = 34_901_797_601_017_929;
    /// The constexpr sampler's index the test gives each state.
    const CONSTEXPR: u32 = 1;

    /// The red byte of texel `(x, y)`.
    fn red(x: u32, y: u32) -> u8 {
        (16 * x + 2 * y + 1) as u8
    }

    /// The kernel: thread `tid` samples at texel `(tid % 8, tid / 8)`'s centre at offsets
    /// `int2(x, y)` of the nest, `x` leaving at `x_end` (an `i32` operand; `%end` is the first word
    /// of buffer 0), and stores the red channels' sum at `out[tid]` (buffer 1).
    fn kernel(x_end: &str) -> String {
        format!(
            r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {NEAREST_WORDS}, i64 0], align 8

define void @filter_nest(ptr addrspace(1) %t, ptr addrspace(1) %offs, ptr addrspace(1) %out, i32 %tid) {{
entry:
  %i = zext i32 %tid to i64
  %end = load i32, ptr addrspace(1) %offs, align 8
  %tx = and i32 %tid, 7
  %ty = lshr i32 %tid, 3
  %fx = uitofp i32 %tx to float
  %fy = uitofp i32 %ty to float
  %cx = fadd float %fx, 5.000000e-01
  %cy = fadd float %fy, 5.000000e-01
  %ux = fmul float %cx, 1.250000e-01
  %uy = fmul float %cy, 1.250000e-01
  %uv0 = insertelement <2 x float> poison, float %ux, i64 0
  %uv = insertelement <2 x float> %uv0, float %uy, i64 1
  br label %ys
ys:
  %y = phi i32 [ -1, %entry ], [ %yn, %ye ]
  %ay = phi float [ 0.000000e+00, %entry ], [ %sx, %ye ]
  br label %xs
xs:
  %x = phi i32 [ -1, %ys ], [ %xn, %xs ]
  %ax = phi float [ %ay, %ys ], [ %sx, %xs ]
  %o0 = insertelement <2 x i32> poison, i32 %x, i64 0
  %o = insertelement <2 x i32> %o0, i32 %y, i64 1
  %r = call {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, <2 x float> %uv, i1 true, <2 x i32> %o, i1 true, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue {{ <4 x float>, i8 }} %r, 0
  %e = extractelement <4 x float> %v, i64 0
  %sx = fadd float %ax, %e
  %xn = add nsw i32 %x, 1
  %xd = icmp eq i32 %xn, {x_end}
  br i1 %xd, label %ye, label %xs
ye:
  %yn = add nsw i32 %y, 1
  %yd = icmp eq i32 %yn, 2
  br i1 %yd, label %done, label %ys
done:
  %q = getelementptr inbounds float, ptr addrspace(1) %out, i64 %i
  store float %sx, ptr addrspace(1) %q, align 4
  ret void
}}

declare {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)

!air.kernel = !{{!0}}
!air.sampler_states = !{{!7}}
!0 = !{{ptr @filter_nest, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5, !6}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"int2", !"air.arg_name", !"offs"}}
!5 = !{{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}}
!6 = !{{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}}
!7 = !{{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}}
"#
        )
    }

    /// The kernel's library, or None where `llvm-as` is missing.
    fn library(x_end: &str) -> Option<Library> {
        match llvm_assemble_text(&kernel(x_end)) {
            Ok(bitcode) => Some(Library::from_air(&bitcode).expect("the library")),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                None
            }
            Err(error) => panic!("filter_nest: {error:?}"),
        }
    }

    /// The 64 sums `library`'s kernel writes on `device`, sampling the test's texture at `index`.
    fn sums(name: &str, device: &Device, library: &Library, index: u32) -> Vec<u32> {
        let function = library.function("filter_nest").expect("the kernel");
        let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(function),
        ));
        let state = (device.new_compute_pipeline_state(&descriptor, |samplers| {
            assert!(
                matches!(samplers, [NumberedSampler::Constexpr(_)]),
                "{name}: {samplers:?}"
            );
            Ok(vec![CONSTEXPR])
        }))
        .unwrap_or_else(|error| panic!("{name}: {error}"));
        let mut d =
            TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, SIDE, SIDE);
        d.usage = TextureUsage::SHADER_READ;
        let texture =
            (device.new_texture(&d, Some(index))).unwrap_or_else(|error| panic!("{name}: {error}"));
        let texels: Vec<u8> = (0..SIDE)
            .flat_map(|y| (0..SIDE).flat_map(move |x| [red(x, y), 0, 0, 255]))
            .collect();
        let whole = Region {
            origin: [0; 3],
            size: [SIDE, SIDE, 1],
        };
        let (row, image) = (u64::from(SIDE * 4), u64::from(SIDE * SIDE * 4));
        (texture.replace_region(whole, 0, 0, crate::HostBytes::from(&texels[..]), row, image))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let mut arena = Arena::new(device);
        let (offs, out) = (arena.buffer(), arena.buffer());
        let end = 2i32.to_le_bytes();
        let contents = offs.contents().expect("a Shared buffer");
        // SAFETY: the buffer holds 256 bytes, more than the 4 written, and nothing reads it
        // meanwhile.
        unsafe {
            end.as_ptr()
                .copy_to_nonoverlapping(contents.as_ptr(), end.len())
        };
        let queue = device.new_command_queue(1).expect("a queue");
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut encoder =
            (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
        encoder
            .set_compute_pipeline_state(&state)
            .expect("the state");
        encoder.set_texture(Some(&texture), 0).expect("the texture");
        encoder.set_buffer(Some(&offs), 0, 0).expect("offs");
        encoder.set_buffer(Some(&out), 0, 1).expect("out");
        (encoder.dispatch_threads([SIDE * SIDE, 1, 1], [SIDE, 1, 1]))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        encoder.end_encoding();
        command_buffer.commit().expect("a commit");
        finished(name, &command_buffer);
        assert_eq!(command_buffer.error(), None, "{name}");
        let contents = out.contents().expect("a Shared buffer");
        // SAFETY: the dispatch that wrote the buffer completed, and its writes reach the host.
        let words = unsafe { std::slice::from_raw_parts(contents.as_ptr().cast::<u32>(), 64) };
        words.to_vec()
    }

    /// As a device without `maintenance8` answers, on the same Vulkan device: its translator takes
    /// no `Offset` operand on a sample, and branches over a run-time offset's values.
    fn no_offset_operand(mut d: DeviceDescription) -> DeviceDescription {
        d.features.remove(&VkFeature::Maintenance8);
        d
    }

    /// The form `device`'s translator gives `library`'s kernel: whether its module needs
    /// `maintenance8` (an `Offset` operand on a sample), and its `OpSwitch` count.
    fn form(name: &str, device: &Device, library: &Library) -> (bool, usize) {
        let function = library.function("filter_nest").expect("the kernel");
        let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(function),
        ));
        let pipeline = (device.translator())
            .compile_compute_pipeline(&descriptor, &Budget::standard())
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let words = &pipeline.stage.spirv;
        let (mut at, mut switches) = (5, 0);
        while let Some(&word) = words.get(at) {
            switches += usize::from(word & 0xffff == OP_SWITCH);
            at += (word >> 16).max(1) as usize;
        }
        let operand = (pipeline.requirements.features).contains(&VkFeature::Maintenance8);
        (operand, switches)
    }

    /// The unrolled kernel and the loaded one write the same sums, the nearest filter's, on every
    /// device: the loaded one with the run-time offset as the sample's `Offset` operand where
    /// device creation enabled `maintenance8` (no switch in its module), and as a switch over its
    /// values otherwise. On the same device with a translator told of no `maintenance8`, the loaded
    /// kernel takes the switch, and writes the same sums as the operand.
    #[test]
    fn every_device_samples_a_filter_nest_alike_unrolled_by_operand_and_switched() {
        let (Some(decided), Some(loaded)) = (library("2"), library("%end")) else {
            return;
        };
        let mut dynamic = BTreeMap::new();
        each_device("filter nest", |name, _, device| {
            let _textures = (device.grant_indices(IndexClass::Texture, 1, 2))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let _samplers = (device.grant_indices(IndexClass::Sampler, CONSTEXPR, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let operand = device.translator().profile().images().dynamic_sample_offset;
            let enabled = (device.extensions().iter()).any(|e| e.as_c_str() == MAINTENANCE8);
            assert_eq!(
                operand, enabled,
                "{name}: the translator's answer is creation's"
            );
            let (needs, switches) = form(name, &device, &loaded);
            assert_eq!(needs, operand, "{name}: the module's Offset operand");
            assert_eq!(switches == 0, operand, "{name}: {switches} switches");
            let unrolled = sums(name, &device, &decided, 1);
            let sampled = sums(name, &device, &loaded, 2);
            assert_eq!(unrolled, sampled, "{name}: the two kernels' sums");
            // Both are the nearest filter's: the clamped texels' red channels, summed in order.
            for (tid, sum) in unrolled.iter().enumerate() {
                let (tx, ty) = (tid as i32 % 8, tid as i32 / 8);
                let mut want = 0.0f32;
                for y in -1..=1 {
                    for x in -1..=1 {
                        let at = |v: i32| v.clamp(0, SIDE as i32 - 1) as u32;
                        want += f32::from(red(at(tx + x), at(ty + y))) / 255.0;
                    }
                }
                let got = f32::from_bits(*sum);
                assert!(
                    (got - want).abs() < 1e-4,
                    "{name}: thread {tid}: {got} against {want}"
                );
            }
            let taken = if operand {
                "the Offset operand (maintenance8)"
            } else {
                "a switch (no maintenance8)"
            };
            println!("filter nest: {name}: the unrolled kernel and {taken} write the same 64 sums");
            dynamic.insert(name.to_owned(), sampled);
        });
        let options = || DeviceOptions {
            described: Some(no_offset_operand),
            ..DeviceOptions::default()
        };
        each_device_with("filter nest switched", options, |name, _, device| {
            let _textures = (device.grant_indices(IndexClass::Texture, 2, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let _samplers = (device.grant_indices(IndexClass::Sampler, CONSTEXPR, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            assert!(!device.translator().profile().images().dynamic_sample_offset);
            let (needs, switches) = form(name, &device, &loaded);
            assert!(!needs && switches > 0, "{name}: {needs} {switches}");
            let switched = sums(name, &device, &loaded, 2);
            assert_eq!(
                Some(&switched),
                dynamic.get(name),
                "{name}: the switch's sums"
            );
            println!("filter nest: {name}: the switch writes the default form's 64 sums");
        });
    }

    /// One cache directory, three devices of one physical device in turn: a translator told of no
    /// `maintenance8` lowers the loaded kernel (the switch); a default one, which takes the `Offset`
    /// operand where creation enabled `maintenance8`, lowers it again and loads nothing, as the
    /// translator's profile fingerprint in the lowering's key moved; a third told of none again
    /// loads the switch's lowering and lowers nothing. On a device without `maintenance8` the
    /// default one loads the switch's.
    #[test]
    fn every_device_keeps_the_operand_s_and_the_switch_s_lowerings_apart() {
        let Some(loaded) = library("%end") else {
            return;
        };
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let dir = std::env::temp_dir().join(format!("m2v-rt-texoff-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            let mut lowered = 0;
            for (round, described) in [
                Some(no_offset_operand as fn(_) -> _),
                None,
                Some(no_offset_operand),
            ]
            .into_iter()
            .enumerate()
            {
                let options = DeviceOptions {
                    pipeline_cache_dir: Some(dir.clone()),
                    described,
                    ..DeviceOptions::default()
                };
                let device = Device::new(physical, options)
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                let operand = device.translator().profile().images().dynamic_sample_offset;
                let counters = device.counters();
                {
                    let _textures = (device.grant_indices(IndexClass::Texture, 1, 1))
                        .unwrap_or_else(|error| panic!("{name}: {error}"));
                    let _samplers = (device.grant_indices(IndexClass::Sampler, CONSTEXPR, 1))
                        .unwrap_or_else(|error| panic!("{name}: {error}"));
                    sums(name, &device, &loaded, 1);
                }
                let metrics = counters.metrics();
                let counted = (metrics.translations, metrics.translations_loaded);
                if round == 0 {
                    assert!(counted.0 > 0 && counted.1 == 0, "{name}: {counted:?}");
                    lowered = counted.0;
                }
                let want = match (round, operand) {
                    (1, true) => (lowered, 0),
                    _ if round > 0 => (0, lowered),
                    _ => counted,
                };
                assert_eq!(counted, want, "{name}: device {round} (operand {operand})");
                drop(device);
                within("the teardown", || counters.gauges() == Gauges::default());
            }
            let _ = std::fs::remove_dir_all(&dir);
            println!("filter nest: {name}: the operand's and the switch's lowerings kept apart");
        }
    }
}
