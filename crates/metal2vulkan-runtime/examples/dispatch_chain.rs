//! Not a consumer's example: what a chain of dependent compute dispatches costs the GPU, by how a
//! command buffer holds them, beside the same chain recorded straight through Vulkan on a
//! `VkDevice` of the example's own, made on the runtime device's physical device.
//!
//! ```sh
//! cargo run --release -p metal2vulkan-runtime --example dispatch_chain -- \
//!     [--dispatches N] [--groups G] [--threads T] [--runs R] [--only SHAPE,...]
//! ```
//!
//! Each run is one command buffer of `N` dispatches (200 by default) of `G` threadgroups of `T`
//! threads (4096 of 32 by default) over one `Private` buffer of `G * T` words, each thread adding
//! 1 to its own word: every dispatch reads what the one before wrote. The shapes:
//!
//! - **`encoders`**: `N` compute encoders of one dispatch each, as a program that opens an encoder
//!   for each dispatch records them;
//! - **`serial`**: one serial compute encoder of the `N` dispatches;
//! - **`concurrent`**: one concurrent encoder of them, with no memory barrier: nothing orders the
//!   dispatches, so this is the kernels' own time with no barrier between them (their words race,
//!   which is not checked);
//! - **`raw-barrier`**: one `VkCommandBuffer` of the `N` dispatches of the same work (a SPIR-V
//!   kernel written here, assembled by `spirv-as`), each after one `VkMemoryBarrier` from
//!   `COMPUTE_SHADER`'s `SHADER_WRITE` to `COMPUTE_SHADER`'s `SHADER_READ | SHADER_WRITE`, the
//!   least barrier Vulkan states for a read after a write between dispatches;
//! - **`raw-wide`**: the same, with the runtime's barrier between dispatches of an encoder
//!   (`Barrier::between_dispatches`: also `DRAW_INDIRECT` and its `INDIRECT_COMMAND_READ`, and
//!   `UNIFORM_READ`);
//! - **`raw-split`**: `raw-barrier` with each dispatch in a `VkCommandBuffer` of its own, after
//!   a `VkCommandBuffer` holding its barrier alone: `2N` command buffers in one `vkQueueSubmit`;
//! - **`raw-none`**: one `VkCommandBuffer` of the dispatches with no barrier.
//!
//! The runtime's shapes take the GPU's time from the command buffer's `gpu_start_time` to its
//! `gpu_end_time`, the raw ones from a timestamp before the first dispatch to one after the last.
//! Each line is the median over `R` runs (9 by default) after one not counted, per dispatch, with
//! the runtime's counts per command buffer that moved (`WorkCounts`, by name from their `Debug`
//! form), and those of the run not counted, the shape's first command buffer, apart: its first
//! recording of the kernel's dispatches, as a program's first command buffer of a kind is. The
//! words are read back after each chained shape: each must hold the dispatches run on it.
//!
//! ```text
//! CHAIN <shape> dispatches <N> groups <G> threads <T> gpu_us_per_dispatch <median> least <least>
//!     <device>
//! CHAIN-COUNTS <shape> <name>=<per command buffer> ...
//! CHAIN-FIRST <shape> <name>=<the first command buffer's> ...
//! ```
//!
//! The kernels are text here: LLVM IR assembled by `llvm-as`, SPIR-V by `spirv-as`, both from
//! `PATH`.

use ash::vk;
use metal2vulkan::metal::DispatchType;
use metal2vulkan_runtime::{
    Buffer, CommandBufferStatus, ComputePipelineState, Device, DeviceOptions, PipelineOptions,
    ResourceOptions, Size, WorkCounts,
};
use std::collections::BTreeMap;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Duration;

/// The example's result: every runtime refusal is a typed `metal2vulkan_runtime::Error`.
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// The longest the example waits for a command buffer.
const WAIT: Duration = Duration::from_secs(60);

/// `bump`: `a[gid] += 1`, `a` a buffer of words at index 0.
const KERNEL: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\n\
target triple = \"air64_v28-apple-macosx26.0.0\"\n\n\
define void @bump(ptr addrspace(1) noundef %a, i32 noundef %gid) {\n\
entry:\n\
  %gw = zext i32 %gid to i64\n\
  %p = getelementptr inbounds i32, ptr addrspace(1) %a, i64 %gw\n\
  %x = load i32, ptr addrspace(1) %p, align 4\n\
  %v = add i32 %x, 1\n\
  store i32 %v, ptr addrspace(1) %p, align 4\n\
  ret void\n\
}\n\n\
!llvm.module.flags = !{!0, !1}\n!air.kernel = !{!20}\n!air.version = !{!8}\n!air.language_version = !{!9}\n\
!0 = !{i32 7, !\"air.max_device_buffers\", i32 31}\n\
!1 = !{i32 7, !\"air.max_threadgroup_buffers\", i32 31}\n\
!3 = !{}\n!4 = !{!5, !7}\n\
!5 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"a\"}\n\
!7 = !{i32 1, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"gid\"}\n\
!8 = !{i32 2, i32 8, i32 0}\n!9 = !{!\"Metal\", i32 4, i32 0, i32 0}\n\
!20 = !{ptr @bump, !3, !4}\n";

/// The same work in SPIR-V, its threadgroup `{T}` threads wide: `words[gid.x] += 1` over a
/// storage buffer at set 0, binding 0.
const SPIRV: &str = "OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main \"main\" %gid %buf
OpExecutionMode %main LocalSize {T} 1 1
OpDecorate %gid BuiltIn GlobalInvocationId
OpDecorate %words ArrayStride 4
OpMemberDecorate %block 0 Offset 0
OpDecorate %block Block
OpDecorate %buf DescriptorSet 0
OpDecorate %buf Binding 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%v3uint = OpTypeVector %uint 3
%in_v3 = OpTypePointer Input %v3uint
%gid = OpVariable %in_v3 Input
%words = OpTypeRuntimeArray %uint
%block = OpTypeStruct %words
%sb_block = OpTypePointer StorageBuffer %block
%buf = OpVariable %sb_block StorageBuffer
%sb_uint = OpTypePointer StorageBuffer %uint
%zero = OpConstant %uint 0
%one = OpConstant %uint 1
%main = OpFunction %void None %fn
%entry = OpLabel
%g = OpLoad %v3uint %gid
%x = OpCompositeExtract %uint %g 0
%p = OpAccessChain %sb_uint %buf %zero %x
%v = OpLoad %uint %p
%w = OpIAdd %uint %v %one
OpStore %p %w
OpReturn
OpFunctionEnd
";

/// The run's shape: dispatches a command buffer, threadgroups a dispatch, threads a threadgroup,
/// timed runs, and the shapes to run.
struct Shape {
    dispatches: u32,
    groups: u32,
    threads: u32,
    runs: usize,
    only: Option<Vec<String>>,
}

impl Shape {
    fn from_args() -> Result<Shape> {
        let mut shape = Shape {
            dispatches: 200,
            groups: 4096,
            threads: 32,
            runs: 9,
            only: None,
        };
        let mut args = std::env::args().skip(1);
        while let Some(flag) = args.next() {
            let value = args.next().ok_or(format!("{flag} takes a value"))?;
            match flag.as_str() {
                "--dispatches" => shape.dispatches = value.parse()?,
                "--groups" => shape.groups = value.parse()?,
                "--threads" => shape.threads = value.parse()?,
                "--runs" => shape.runs = value.parse()?,
                "--only" => shape.only = Some(value.split(',').map(str::to_owned).collect()),
                _ => return Err(format!("unknown flag {flag}").into()),
            }
        }
        Ok(shape)
    }

    fn runs(&self, name: &str) -> bool {
        self.only
            .as_ref()
            .is_none_or(|only| only.iter().any(|shape| shape == name))
    }

    fn words(&self) -> u64 {
        u64::from(self.groups) * u64::from(self.threads)
    }
}

/// Run `tool` from `PATH` over `input` on its standard input: its standard output.
fn tool(tool: &str, args: &[&str], input: &str) -> Result<Vec<u8>> {
    let mut child = Command::new(tool)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("cannot run {tool} (put it on PATH): {error}"))?;
    child
        .stdin
        .take()
        .ok_or("no stdin")?
        .write_all(input.as_bytes())?;
    let output = child.wait_with_output()?;
    if !output.status.success() {
        return Err(format!("{tool}: {}", String::from_utf8_lossy(&output.stderr)).into());
    }
    Ok(output.stdout)
}

/// The state of [`KERNEL`].
fn state(device: &Device) -> Result<ComputePipelineState> {
    let bitcode = tool("llvm-as", &["-o", "-", "-"], KERNEL)?;
    let library = metal2vulkan::library::Library::from_air(&bitcode)?;
    let function = device
        .new_library_with_compile_library(library)
        .new_function_with_name("bump")?;
    Ok(device.new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())?)
}

/// The counts of `counts` by name, read from their `Debug` form.
fn counts(counts: &WorkCounts) -> BTreeMap<String, u64> {
    let text = format!("{counts:?}");
    text.split([',', '{'])
        .filter_map(|part| {
            let (name, value) = part.split_once(':')?;
            Some((name.trim().to_owned(), value.trim().parse().ok()?))
        })
        .collect()
}

/// The samples of one shape: GPU microseconds per dispatch, the counts' sums over the runs
/// counted, and the counts of the run not counted, its first.
#[derive(Default)]
struct Samples {
    per_dispatch: Vec<f64>,
    counts: BTreeMap<String, u64>,
    first: BTreeMap<String, u64>,
}

/// What moved from `before` to `after`, added to `into`.
fn add_moved(into: &mut BTreeMap<String, u64>, before: &WorkCounts, after: &WorkCounts) {
    let before = counts(before);
    for (name, value) in counts(after) {
        let moved = value.saturating_sub(before.get(&name).copied().unwrap_or(0));
        if moved > 0 {
            *into.entry(name).or_default() += moved;
        }
    }
}

impl Samples {
    fn add_counts(&mut self, before: &WorkCounts, after: &WorkCounts) {
        add_moved(&mut self.counts, before, after);
    }

    fn print(mut self, name: &str, shape: &Shape, device: &str) {
        self.per_dispatch.sort_by(f64::total_cmp);
        println!(
            "CHAIN {name} dispatches {} groups {} threads {} gpu_us_per_dispatch {:.2} least {:.2} \
             {device}",
            shape.dispatches,
            shape.groups,
            shape.threads,
            self.per_dispatch[self.per_dispatch.len() / 2],
            self.per_dispatch[0],
        );
        if !self.counts.is_empty() {
            let runs = self.per_dispatch.len() as f64;
            let line: Vec<String> = (self.counts.iter())
                .map(|(name, sum)| format!("{name}={:.2}", *sum as f64 / runs))
                .collect();
            println!("CHAIN-COUNTS {name} {}", line.join(" "));
        }
        if !self.first.is_empty() {
            let line: Vec<String> = (self.first.iter())
                .map(|(name, value)| format!("{name}={value}"))
                .collect();
            println!("CHAIN-FIRST {name} {}", line.join(" "));
        }
    }
}

/// The runtime's shapes over one buffer.
struct Ours<'a> {
    device: &'a Device,
    state: ComputePipelineState,
    words: Buffer,
    readback: Buffer,
}

impl Ours<'_> {
    /// One command buffer of `shape`'s dispatches as `name` holds them, committed and waited for:
    /// its GPU time per dispatch in microseconds.
    fn run(&self, name: &str, shape: &Shape) -> Result<f64> {
        let queue = self.device.new_command_queue()?;
        let command_buffer = queue.command_buffer()?;
        let groups = Size::new(u64::from(shape.groups), 1, 1);
        let threads = Size::new(u64::from(shape.threads), 1, 1);
        let encoders = match name {
            "encoders" => shape.dispatches,
            _ => 1,
        };
        for _ in 0..encoders {
            let mut encoder = match name {
                "concurrent" => command_buffer
                    .compute_command_encoder_with_dispatch_type(DispatchType::Concurrent)?,
                _ => command_buffer.compute_command_encoder()?,
            };
            encoder.set_compute_pipeline_state(&self.state)?;
            encoder.set_buffer(Some(&self.words), 0, 0)?;
            for _ in 0..shape.dispatches / encoders {
                encoder.dispatch_threadgroups(groups, threads)?;
            }
            encoder.end_encoding()?;
        }
        command_buffer.commit()?;
        command_buffer
            .wait_until_completed_for(WAIT)
            .map_err(|_| format!("a command buffer did not complete within {WAIT:?}"))?;
        if command_buffer.status() != CommandBufferStatus::Completed {
            return Err(format!("a command buffer ended {:?}", command_buffer.error()).into());
        }
        let gpu = command_buffer.gpu_end_time()? - command_buffer.gpu_start_time()?;
        Ok(gpu * 1e6 / f64::from(shape.dispatches))
    }

    /// Clear the words, run `shape` as `name` holds it, and check what the chained shapes left.
    fn shape(&self, name: &str, shape: &Shape, device: &str) -> Result<()> {
        self.fill(0)?;
        let mut samples = Samples::default();
        for run in 0..=shape.runs {
            let before = self.device.work_counts();
            let per = self.run(name, shape)?;
            match run {
                0 => add_moved(&mut samples.first, &before, &self.device.work_counts()),
                _ => {
                    samples.per_dispatch.push(per);
                    samples.add_counts(&before, &self.device.work_counts());
                }
            }
        }
        if name != "concurrent" {
            self.check(shape.dispatches * (shape.runs as u32 + 1), name)?;
        }
        samples.print(name, shape, device);
        Ok(())
    }

    fn fill(&self, value: u8) -> Result<()> {
        let queue = self.device.new_command_queue()?;
        let command_buffer = queue.command_buffer()?;
        let mut blit = command_buffer.blit_command_encoder()?;
        blit.fill_buffer(&self.words, 0..self.words.length(), value)?;
        blit.end_encoding()?;
        command_buffer.commit()?;
        command_buffer
            .wait_until_completed_for(WAIT)
            .map_err(|_| "the fill did not complete")?;
        Ok(())
    }

    /// Copy the words back and check each is `want`.
    fn check(&self, want: u32, name: &str) -> Result<()> {
        let queue = self.device.new_command_queue()?;
        let command_buffer = queue.command_buffer()?;
        let mut blit = command_buffer.blit_command_encoder()?;
        blit.copy_from_buffer_to_buffer(&self.words, 0, &self.readback, 0, self.words.length())?;
        blit.end_encoding()?;
        command_buffer.commit()?;
        command_buffer
            .wait_until_completed_for(WAIT)
            .map_err(|_| "the read back did not complete")?;
        let contents = self.readback.contents().ok_or("no contents")?;
        for at in 0..(self.words.length() / 4) as usize {
            // SAFETY: `readback` is `Shared`, as long as `words`, and no command buffer writes it
            // while the CPU reads it.
            let word = unsafe { contents.as_ptr().cast::<u32>().add(at).read_volatile() };
            if word != want {
                return Err(format!("{name}: word {at} is {word}, not {want}").into());
            }
        }
        Ok(())
    }
}

/// The raw shapes' Vulkan objects, on a `VkDevice` of their own over the runtime device's
/// physical device, whose instance they borrow.
struct Raw<'a> {
    _runtime: &'a Device,
    vk: ash::Device,
    queue: vk::Queue,
    pool: vk::CommandPool,
    commands: Vec<vk::CommandBuffer>,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
    descriptors: vk::DescriptorPool,
    set: vk::DescriptorSet,
    module: vk::ShaderModule,
    pipeline: vk::Pipeline,
    buffers: [(vk::Buffer, vk::DeviceMemory); 2],
    bytes: u64,
    mapped: *const u32,
    queries: vk::QueryPool,
    fence: vk::Fence,
    period: f64,
}

/// The least barrier between dispatches, a read after a write.
fn least() -> vk::MemoryBarrier<'static> {
    vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
}

impl<'a> Raw<'a> {
    /// The objects, on a `VkDevice` made with one queue of the runtime device's primary family.
    fn new(device: &'a Device, shape: &Shape) -> Result<Raw<'a>> {
        // SAFETY: the handles are used while the runtime device lives (the result borrows it),
        // and none of the runtime's objects is destroyed.
        let handles = unsafe { device.raw() };
        let family = handles.queue_family;
        let priorities = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        // SAFETY: the instance's own physical device and a valid creation; the device is
        // destroyed in `drop`, before the instance (which the runtime device holds) goes.
        let vk = unsafe {
            handles.instance.create_device(
                handles.physical_device,
                &vk::DeviceCreateInfo::default().queue_create_infos(&queues),
                None,
            )
        }?;
        // SAFETY: the family and index were created.
        let queue = unsafe { vk.get_device_queue(family, 0) };
        let text = SPIRV.replace("{T}", &shape.threads.to_string());
        let bytes = tool(
            "spirv-as",
            &["--target-env", "vulkan1.2", "-o", "-", "-"],
            &text,
        )?;
        let code: Vec<u32> = (bytes.as_chunks::<4>().0.iter())
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        let properties =
            // SAFETY: the instance's own physical device.
            unsafe { handles.instance.get_physical_device_properties(handles.physical_device) };
        let memory =
            // SAFETY: as above.
            unsafe { handles.instance.get_physical_device_memory_properties(handles.physical_device) };
        // SAFETY: every info below is valid and lives for its call; every object is of `vk`.
        // Objects made before a failure here are not destroyed: the example ends on it.
        unsafe {
            let make_buffer =
                |bytes: u64, usage, flags| -> Result<(vk::Buffer, vk::DeviceMemory)> {
                    let buffer = vk.create_buffer(
                        &vk::BufferCreateInfo::default().size(bytes).usage(usage),
                        None,
                    )?;
                    let requirements = vk.get_buffer_memory_requirements(buffer);
                    let index = (0..memory.memory_type_count)
                        .find(|&index| {
                            requirements.memory_type_bits & (1 << index) != 0
                                && memory.memory_types[index as usize]
                                    .property_flags
                                    .contains(flags)
                        })
                        .ok_or("no memory type for the buffer")?;
                    let allocation = vk.allocate_memory(
                        &vk::MemoryAllocateInfo::default()
                            .allocation_size(requirements.size)
                            .memory_type_index(index),
                        None,
                    )?;
                    vk.bind_buffer_memory(buffer, allocation, 0)?;
                    Ok((buffer, allocation))
                };
            let bytes = shape.words() * 4;
            let words = make_buffer(
                bytes,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;
            let readback = make_buffer(
                bytes,
                vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            let mapped = vk
                .map_memory(readback.1, 0, bytes, vk::MemoryMapFlags::empty())?
                .cast::<u32>();
            let binding = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)];
            let set_layout = vk.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&binding),
                None,
            )?;
            let layout = vk.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout]),
                None,
            )?;
            let module =
                vk.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main");
            let pipeline = vk
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(stage)
                        .layout(layout)],
                    None,
                )
                .map_err(|(_, error)| error)?[0];
            let sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)];
            let descriptors = vk.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&sizes),
                None,
            )?;
            let set = vk.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptors)
                    .set_layouts(&[set_layout]),
            )?[0];
            let info = [vk::DescriptorBufferInfo::default()
                .buffer(words.0)
                .range(vk::WHOLE_SIZE)];
            vk.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&info)],
                &[],
            );
            let pool = vk.create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(family),
                None,
            )?;
            let commands = vk.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(2 * shape.dispatches + 2),
            )?;
            let queries = vk.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(2),
                None,
            )?;
            let fence = vk.create_fence(&vk::FenceCreateInfo::default(), None)?;
            Ok(Raw {
                _runtime: device,
                vk,
                queue,
                pool,
                commands,
                layout,
                set_layout,
                descriptors,
                set,
                module,
                pipeline,
                buffers: [words, readback],
                bytes,
                mapped,
                queries,
                fence,
                period: f64::from(properties.limits.timestamp_period),
            })
        }
    }

    /// Submit `commands` and wait for them.
    ///
    /// # Safety
    ///
    /// Each is recorded and executable, and none is pending.
    unsafe fn submit(&self, commands: &[vk::CommandBuffer]) -> Result<()> {
        // SAFETY: the caller's contract; the queue is the caller's alone.
        unsafe {
            self.vk.queue_submit(
                self.queue,
                &[vk::SubmitInfo::default().command_buffers(commands)],
                self.fence,
            )?;
            self.vk
                .wait_for_fences(&[self.fence], true, WAIT.as_nanos() as u64)?;
            self.vk.reset_fences(&[self.fence])?;
        }
        Ok(())
    }

    /// Begin `command` for one submission.
    ///
    /// # Safety
    ///
    /// `command` is of the pool, not pending.
    unsafe fn begin(&self, command: vk::CommandBuffer) -> Result<()> {
        let info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the caller's contract.
        unsafe { self.vk.begin_command_buffer(command, &info) }?;
        Ok(())
    }

    /// Record a dispatch of the kernel into `command`.
    ///
    /// # Safety
    ///
    /// `command` is recording.
    unsafe fn dispatch(&self, command: vk::CommandBuffer, shape: &Shape) {
        // SAFETY: the caller's contract; the objects are live.
        unsafe {
            self.vk
                .cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            self.vk.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[self.set],
                &[],
            );
            self.vk.cmd_dispatch(command, shape.groups, 1, 1);
        }
    }

    /// Record `barrier` from `src` stages to `dst` stages into `command`.
    ///
    /// # Safety
    ///
    /// `command` is recording.
    unsafe fn barrier(
        &self,
        command: vk::CommandBuffer,
        (src, dst, barrier): (
            vk::PipelineStageFlags,
            vk::PipelineStageFlags,
            vk::MemoryBarrier,
        ),
    ) {
        // SAFETY: the caller's contract.
        unsafe {
            self.vk.cmd_pipeline_barrier(
                command,
                src,
                dst,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
    }

    /// One run of `name`: its GPU time per dispatch in microseconds.
    fn run(&self, name: &str, shape: &Shape) -> Result<f64> {
        let compute = vk::PipelineStageFlags::COMPUTE_SHADER;
        let barrier = match name {
            "raw-wide" => Some((
                compute,
                compute | vk::PipelineStageFlags::DRAW_INDIRECT,
                vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(
                        vk::AccessFlags::SHADER_READ
                            | vk::AccessFlags::SHADER_WRITE
                            | vk::AccessFlags::UNIFORM_READ
                            | vk::AccessFlags::INDIRECT_COMMAND_READ,
                    ),
            )),
            "raw-none" => None,
            _ => Some((compute, compute, least())),
        };
        let split = name == "raw-split";
        // SAFETY: every command buffer is of the pool, none pending (each run waits for its
        // submission), and the pool is reset before they are recorded again.
        unsafe {
            self.vk
                .reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())?;
            let first = self.commands[0];
            self.begin(first)?;
            self.vk.cmd_reset_query_pool(first, self.queries, 0, 2);
            self.vk.cmd_write_timestamp(
                first,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                self.queries,
                0,
            );
            let mut used = vec![first];
            let mut current = first;
            for n in 0..shape.dispatches as usize {
                if split {
                    self.vk.end_command_buffer(current)?;
                    current = self.commands[1 + 2 * n];
                    self.begin(current)?;
                    used.push(current);
                }
                if let (Some(barrier), true) = (barrier, n > 0) {
                    self.barrier(current, barrier);
                }
                if split {
                    self.vk.end_command_buffer(current)?;
                    current = self.commands[2 + 2 * n];
                    self.begin(current)?;
                    used.push(current);
                }
                self.dispatch(current, shape);
            }
            self.vk.cmd_write_timestamp(
                current,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.queries,
                1,
            );
            self.vk.end_command_buffer(current)?;
            self.submit(&used)?;
            let mut stamps = [0u64; 2];
            self.vk.get_query_pool_results(
                self.queries,
                0,
                &mut stamps,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )?;
            let nanos = stamps[1].wrapping_sub(stamps[0]) as f64 * self.period;
            Ok(nanos / 1e3 / f64::from(shape.dispatches))
        }
    }

    /// Fill the words with `value`, or copy them back.
    fn transfer(&self, fill: Option<u32>) -> Result<()> {
        let command = self.commands[0];
        // SAFETY: as `run`.
        unsafe {
            self.vk
                .reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())?;
            self.begin(command)?;
            match fill {
                Some(value) => {
                    self.vk
                        .cmd_fill_buffer(command, self.buffers[0].0, 0, vk::WHOLE_SIZE, value)
                }
                None => {
                    let to_copy = vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
                    self.barrier(
                        command,
                        (
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::PipelineStageFlags::TRANSFER,
                            to_copy,
                        ),
                    );
                    let region = vk::BufferCopy::default().size(self.bytes);
                    self.vk.cmd_copy_buffer(
                        command,
                        self.buffers[0].0,
                        self.buffers[1].0,
                        &[region],
                    );
                    let to_host = vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::HOST_READ);
                    self.barrier(
                        command,
                        (
                            vk::PipelineStageFlags::TRANSFER,
                            vk::PipelineStageFlags::HOST,
                            to_host,
                        ),
                    );
                }
            }
            if fill.is_some() {
                let to_compute = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
                self.barrier(
                    command,
                    (
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        to_compute,
                    ),
                );
            }
            self.vk.end_command_buffer(command)?;
            self.submit(&[command])?;
        }
        Ok(())
    }
}

impl Drop for Raw<'_> {
    fn drop(&mut self) {
        // SAFETY: nothing is pending (each submission was waited for), and every object is this
        // example's own.
        unsafe {
            self.vk.destroy_fence(self.fence, None);
            self.vk.destroy_query_pool(self.queries, None);
            self.vk.destroy_command_pool(self.pool, None);
            self.vk.destroy_descriptor_pool(self.descriptors, None);
            self.vk.destroy_pipeline(self.pipeline, None);
            self.vk.destroy_shader_module(self.module, None);
            self.vk.destroy_pipeline_layout(self.layout, None);
            self.vk.destroy_descriptor_set_layout(self.set_layout, None);
            for (buffer, memory) in self.buffers {
                self.vk.destroy_buffer(buffer, None);
                self.vk.free_memory(memory, None);
            }
            self.vk.destroy_device(None);
        }
    }
}

fn main() -> Result<()> {
    let shape = Shape::from_args()?;
    let device = Device::system_default_with(&DeviceOptions::default())?;
    let name = device.name().to_string();
    println!("device: {name}");
    let bytes = shape.words() * 4;
    let ours = Ours {
        device: &device,
        state: state(&device)?,
        words: device.new_buffer(bytes, ResourceOptions::private())?,
        readback: device.new_buffer(bytes, ResourceOptions::shared())?,
    };
    for shape_name in ["encoders", "serial", "concurrent"] {
        if shape.runs(shape_name) {
            ours.shape(shape_name, &shape, &name)?;
        }
    }
    let raw = Raw::new(&device, &shape)?;
    for shape_name in ["raw-barrier", "raw-wide", "raw-split", "raw-none"] {
        if !shape.runs(shape_name) {
            continue;
        }
        raw.transfer(Some(0))?;
        let mut samples = Samples::default();
        for run in 0..=shape.runs {
            let per = raw.run(shape_name, &shape)?;
            if run > 0 {
                samples.per_dispatch.push(per);
            }
        }
        if shape_name != "raw-none" {
            raw.transfer(None)?;
            let want = shape.dispatches * (shape.runs as u32 + 1);
            for at in 0..shape.words() as usize {
                // SAFETY: the mapped read-back, as long as the words, written by the copy the
                // transfer waited for.
                let word = unsafe { raw.mapped.add(at).read_volatile() };
                if word != want {
                    return Err(format!("{shape_name}: word {at} is {word}, not {want}").into());
                }
            }
        }
        samples.print(shape_name, &shape, &name);
    }
    Ok(())
}
