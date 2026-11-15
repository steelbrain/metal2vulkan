//! Where an indirect command buffer's records lie (`indirect_command_buffer/home.rs`): on a device
//! without unified memory that reserves a staging queue, a buffer the GPU uses first keeps them in
//! device-local memory, held by each command buffer that uses them; a `Shared` one's move into the
//! host's memory at the first host access no command buffer holds them for, with their bytes where
//! the GPU wrote them and none where it did not; a `Private` one's never move; any other host
//! access is staged, announced once per device. Elsewhere every buffer's records lie in the host's
//! memory: nothing moves and nothing is staged. On every device the commands draw what was
//! encoded, whoever encoded them, and the getters answer it; a GPU reset before an encode leaves
//! nothing of what an earlier encode set (`lower::access::encode`'s `default_record`).
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects. The kernel is the owned fixture
//! `validation/fixtures/public/kernel_icb_encode_render_commands.ll`, which encodes one draw a
//! thread through an argument buffer that holds the indirect command buffer; the render functions
//! are LLVM text here (`llvm-as`).

use super::blit_holds::{options, Scratch, WAIT};
use super::encoder_holds::library;
use crate::{
    BlitOption, Buffer, ClearColor, CommandBuffer, CommandBufferStatus, ComputePipelineState,
    Degradation, DegradeNotice, Device, IndirectCommandBuffer, IndirectCommandBufferDescriptor,
    IndirectDraw, LoadAction, MessageSink, Origin, PipelineOptions, RenderPassDescriptor,
    RenderPipelineState, ResourceOptions, Size, StoreAction, Texture, TextureDescriptor,
    WorkCounts,
};
use ash::vk;
use metal2vulkan::bind::indirect_commands::written_flag_offset;
use metal2vulkan::metal::{
    IndirectCommandType, PixelFormat, PrimitiveType, RenderStages, ResourceUsage, TextureUsage,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource,
};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

/// The commands of a buffer, and the cells of the target: a command draws the lower half of the
/// cell of its vertex start's third.
const COMMANDS: u32 = 32;
const SIDE: u64 = 64;
const CELL: u64 = 8;
const CELLS: u64 = (SIDE / CELL) * (SIDE / CELL);

/// The cell a CPU setter moves command 0's draw to, past every command's own.
const MOVED_TO: u64 = 40;

/// `icb_vertex`: the `float4` at vertex `id` of vertex buffer 0; `icb_fragment`: opaque magenta.
const FUNCTIONS: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\n\
target triple = \"air64_v28-apple-macosx26.0.0\"\n\n\
define <4 x float> @icb_vertex(ptr addrspace(1) noundef readonly %v, i32 noundef %id) {\n\
entry:\n\
  %i = zext i32 %id to i64\n\
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %v, i64 %i\n\
  %x = load <4 x float>, ptr addrspace(1) %p, align 16\n\
  ret <4 x float> %x\n\
}\n\n\
define <4 x float> @icb_fragment() {\n\
entry:\n\
  ret <4 x float> <float 1.0, float 0.0, float 1.0, float 1.0>\n\
}\n\n\
!llvm.module.flags = !{!0, !1}\n!air.vertex = !{!20}\n!air.fragment = !{!30}\n\
!air.version = !{!8}\n!air.language_version = !{!9}\n\
!0 = !{i32 7, !\"air.max_device_buffers\", i32 31}\n\
!1 = !{i32 7, !\"air.max_constant_buffers\", i32 31}\n\
!8 = !{i32 2, i32 8, i32 0}\n!9 = !{!\"Metal\", i32 4, i32 0, i32 0}\n\
!20 = !{ptr @icb_vertex, !21, !23}\n\
!21 = !{!22}\n\
!22 = !{!\"air.position\", !\"air.arg_type_name\", !\"float4\"}\n\
!23 = !{!24, !25}\n\
!24 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 16, !\"air.arg_type_align_size\", i32 16, !\"air.arg_type_name\", !\"float4\", !\"air.arg_name\", !\"v\"}\n\
!25 = !{i32 1, !\"air.vertex_id\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"id\"}\n\
!26 = !{!27}\n\
!27 = !{!\"air.render_target\", i32 0, i32 0, !\"air.arg_type_name\", !\"float4\"}\n\
!28 = !{}\n\
!30 = !{ptr @icb_fragment, !26, !28}\n";

/// The fragment function's color, exact in `RGBA8Unorm`.
const TEXEL: [u8; 4] = [255, 0, 255, 255];

/// The owned fixture's kernel, and its file's stem.
const WRITER: &str = "kernel_icb_encode_render_commands";

/// The fixture's kernel with its body replaced: each thread `t`, where `reset`, first resets
/// command `t` on the GPU (`air.reset_render_command`); then an even `t` encodes what the fixture
/// does (its vertex and fragment buffers and its draw), and an odd one nothing.
fn reset_writer_text(reset: bool) -> String {
    let path = metal2vulkan_devtools::fixtures_dir().join(format!("{WRITER}.ll"));
    let text = std::fs::read_to_string(path).unwrap();
    let define = text.find(&format!("define void @{WRITER}(")).unwrap();
    let open = define + text[define..].find("{\n").unwrap() + 2;
    let close = open + text[open..].find("\n}\n").unwrap();
    let attributes = text.find("attributes #0").unwrap();
    let reset = match reset {
        true => {
            "  tail call void @air.reset_render_command(ptr addrspace(1) captures(none) %6, i32 %3) #2\n"
        }
        false => "",
    };
    let body = format!(
        "  %5 = getelementptr inbounds %struct.Commands, ptr addrspace(1) %0, i64 0, i32 0, i32 0\n\
  %6 = load ptr addrspace(1), ptr addrspace(1) %5, align 8\n\
{reset}  %odd = and i32 %3, 1\n\
  %even = icmp eq i32 %odd, 0\n\
  br i1 %even, label %encode, label %done\n\n\
encode:\n\
  tail call void @air.set_vertex_buffer_render_command.p1i8(ptr addrspace(1) captures(none) %6, i32 %3, ptr addrspace(1) %1, i64 -1, i32 0) #2\n\
  tail call void @air.set_fragment_buffer_render_command.p1i8(ptr addrspace(1) captures(none) %6, i32 %3, ptr addrspace(1) %1, i32 1) #2\n\
  %7 = mul i32 %3, 3\n\
  tail call void @air.draw_primitives_render_command(ptr addrspace(1) captures(none) %6, i32 %3, i32 3, i32 %7, i32 3, i32 1, i32 0) #2\n\
  br label %done\n\n\
done:\n\
  ret void"
    );
    let declare = "declare void @air.reset_render_command(ptr addrspace(1) captures(none), i32) local_unnamed_addr #1\n\n";
    format!(
        "{}{body}{}{declare}{}",
        &text[..open],
        &text[close..attributes],
        &text[attributes..]
    )
}

/// What every test uses, on a device whose notices it hears.
struct Ours {
    device: Device,
    heard: Arc<Mutex<Vec<DegradeNotice>>>,
    writer: ComputePipelineState,
    state: RenderPipelineState,
    target: Texture,
    /// Three vertices a cell: its lower half.
    vertices: Buffer,
    /// The writer's third argument, a word a thread.
    out: Buffer,
    _scratch: Scratch,
}

/// The counts a test reads: moves, their bytes, and staged accesses.
fn moves(counts: &WorkCounts) -> [u64; 3] {
    [
        counts.icb_records_moved_to_host,
        counts.icb_record_bytes_moved,
        counts.icb_records_staged_accesses,
    ]
}

/// Commit `command_buffer` and wait for it to complete.
fn run(command_buffer: &CommandBuffer) {
    command_buffer.commit().unwrap();
    wait(command_buffer);
}

/// Wait for `command_buffer` to complete.
fn wait(command_buffer: &CommandBuffer) {
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(
        command_buffer.status(),
        CommandBufferStatus::Completed,
        "{:?}",
        command_buffer.error()
    );
}

/// Three vertices a cell in normalized device coordinates, the cells in rows from the bottom left.
fn vertices() -> Vec<u8> {
    let across = SIDE / CELL;
    let step = 2.0 * CELL as f32 / SIDE as f32;
    (0..CELLS)
        .flat_map(|cell| {
            let x = -1.0 + step * (cell % across) as f32;
            let y = -1.0 + step * (cell / across) as f32;
            [[x, y], [x + step, y], [x, y + step]]
        })
        .flat_map(|[x, y]| [x, y, 0.0, 1.0])
        .flat_map(f32::to_le_bytes)
        .collect()
}

impl Ours {
    fn new(leg: &str) -> Ours {
        let scratch = Scratch::new(leg);
        let heard = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&heard);
        let mut options = options(&scratch.0);
        options.messages = Some(MessageSink::new(move |message| {
            if let Some(notice) = &message.degradation {
                sink.lock().unwrap().push(notice.clone());
            }
        }));
        let device = Device::system_default_with(&options).unwrap();
        let bitcode = metal2vulkan_devtools::llvm_assemble_text(FUNCTIONS).unwrap();
        let functions = device.new_library_with_compile_library(
            metal2vulkan::library::Library::from_air(&bitcode).unwrap(),
        );
        let function = |name: &str| {
            StageSource::Function(
                functions
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let mut descriptor = RenderPipelineDescriptor::new(function("icb_vertex"));
        descriptor.fragment = Some(function("icb_fragment"));
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        descriptor.support_indirect_command_buffers = true;
        let state = device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        let writer = library(&device, &format!("{WRITER}.ll"))
            .new_function_with_name(WRITER)
            .unwrap();
        let writer =
            ComputePipelineDescriptor::new(StageSource::Function(writer.function_descriptor()));
        let writer = device
            .new_compute_pipeline_state(&writer, &PipelineOptions::default())
            .unwrap();
        let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
        target.usage = TextureUsage::RENDER_TARGET;
        target.resource_options = ResourceOptions::private();
        let target = device.new_texture(&target).unwrap();
        let vertices = device
            .new_buffer_with_bytes(&vertices(), ResourceOptions::shared())
            .unwrap();
        let out = device
            .new_buffer(4 * u64::from(COMMANDS), ResourceOptions::shared())
            .unwrap();
        Ours {
            device,
            heard,
            writer,
            state,
            target,
            vertices,
            out,
            _scratch: scratch,
        }
    }

    /// A kernel of the fixture's name and arguments, made of LLVM text `text`.
    fn writer_of(&self, text: &str) -> ComputePipelineState {
        let bitcode = metal2vulkan_devtools::llvm_assemble_text(text).unwrap();
        let library = self.device.new_library_with_compile_library(
            metal2vulkan::library::Library::from_air(&bitcode).unwrap(),
        );
        let function = library.new_function_with_name(WRITER).unwrap();
        let descriptor =
            ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
        self.device
            .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap()
    }

    /// Whether the device places records by their first use: without unified memory, with a
    /// staging queue (`home.rs`).
    fn placing(&self) -> bool {
        !self.device.properties().has_unified_memory && self.device.has_staging_queue()
    }

    /// A buffer of [`COMMANDS`] draws that inherit the pipeline state, of `options`' storage.
    fn icb(&self, options: ResourceOptions) -> IndirectCommandBuffer {
        let mut descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.command_types = IndirectCommandType::DRAW;
        descriptor.inherit_pipeline_state = true;
        descriptor.max_vertex_buffer_bind_count = 2;
        descriptor.max_fragment_buffer_bind_count = 2;
        self.device
            .new_indirect_command_buffer(&descriptor, u64::from(COMMANDS), options)
            .unwrap()
    }

    /// The kernel's writes of every command of `icb` through an argument buffer that holds it,
    /// committed and waited for.
    fn encode_on_gpu(&self, icb: &IndirectCommandBuffer) {
        self.encode_with(icb, &self.writer);
    }

    /// [`encode_on_gpu`](Self::encode_on_gpu) by `writer`, a kernel of the fixture's arguments.
    fn encode_with(&self, icb: &IndirectCommandBuffer, writer: &ComputePipelineState) {
        let arguments = self
            .device
            .new_buffer_with_bytes(
                &icb.gpu_resource_id().0.to_le_bytes(),
                ResourceOptions::shared(),
            )
            .unwrap();
        let queue = self.device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(writer).unwrap();
        encoder.set_buffer(Some(&arguments), 0, 0).unwrap();
        encoder.set_buffer(Some(&self.vertices), 0, 1).unwrap();
        encoder.set_buffer(Some(&self.out), 0, 2).unwrap();
        encoder
            .use_resource(icb.as_resource(), ResourceUsage::WRITE)
            .unwrap();
        encoder
            .dispatch_threads(Size::new(u64::from(COMMANDS), 1, 1), Size::new(32, 1, 1))
            .unwrap();
        encoder.end_encoding().unwrap();
        run(&command_buffer);
    }

    /// The CPU's writes of every command of `icb`, as the kernel writes them.
    fn encode_on_cpu(&self, icb: &IndirectCommandBuffer) {
        for index in 0..u64::from(COMMANDS) {
            let command = icb.indirect_render_command_at(index).unwrap();
            command
                .set_vertex_buffer(Some(&self.vertices), 0, 0)
                .unwrap();
            command
                .set_fragment_buffer(Some(&self.vertices), 0, 1)
                .unwrap();
            command
                .draw_primitives(PrimitiveType::Triangle, 3 * index, 3, 1, 0)
                .unwrap();
        }
    }

    /// A render pass over the cleared target that executes every command of `icb`, encoded, not
    /// committed.
    fn execute(&self, icb: &IndirectCommandBuffer) -> CommandBuffer {
        let queue = self.device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(self.target.clone());
        color.load_action = LoadAction::Clear;
        color.store_action = StoreAction::Store;
        color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&self.state).unwrap();
        encoder
            .use_resource(
                self.vertices.as_resource(),
                ResourceUsage::READ,
                RenderStages::VERTEX,
            )
            .unwrap();
        encoder
            .execute_commands_in_buffer(icb, 0..u64::from(COMMANDS))
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer
    }

    /// The cells the last pass drew: texel (1, 6) of each, inside its lower half's triangle
    /// whatever the rasterizer's rules for its edges, is [`TEXEL`].
    fn drawn(&self) -> Vec<u64> {
        let row = SIDE * 4;
        let readback = self
            .device
            .new_buffer(row * SIDE, ResourceOptions::shared())
            .unwrap();
        let queue = self.device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_texture_to_buffer(
            &self.target,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(SIDE, SIDE, 1),
            &readback,
            0,
            row,
            row * SIDE,
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        run(&command_buffer);
        let contents = readback.contents().unwrap();
        // SAFETY: `contents` holds the buffer's bytes, which no command buffer writes now.
        let texels =
            unsafe { std::slice::from_raw_parts(contents.as_ptr(), readback.length() as usize) };
        let across = SIDE / CELL;
        (0..CELLS)
            .filter(|cell| {
                let (cx, cy) = (cell % across, cell / across);
                let x = cx * CELL + 1;
                let y = SIDE - 1 - (cy * CELL + 1);
                let at = ((y * SIDE + x) * 4) as usize;
                texels[at..at + 4] == TEXEL
            })
            .collect()
    }

    /// The vertex start command `index`'s draw holds, as its getter answers it.
    fn vertex_start(&self, icb: &IndirectCommandBuffer, index: u64) -> u64 {
        let command = icb.indirect_render_command_at(index).unwrap();
        match command.draw().unwrap() {
            Some(IndirectDraw::Primitives { vertex_start, .. }) => vertex_start,
            other => panic!("command {index} holds {other:?}"),
        }
    }

    /// How many `staged-indirect-records` notices the device announced.
    fn staged_notices(&self) -> usize {
        (self.heard.lock().unwrap().iter())
            .filter(|notice| notice.degradation == Degradation::StagedIndirectRecords)
            .count()
    }

    /// The counts of [`moves`] since `before`.
    fn moved_since(&self, before: [u64; 3]) -> [u64; 3] {
        let now = moves(&self.device.work_counts());
        [0, 1, 2].map(|at| now[at] - before[at])
    }
}

/// Every command's own cell.
fn every_command() -> Vec<u64> {
    (0..u64::from(COMMANDS)).collect()
}

/// The bytes of a buffer of [`COMMANDS`]' records: its header, records and written flags.
fn storage_bytes() -> u64 {
    written_flag_offset(COMMANDS, COMMANDS)
}

/// A `Shared` buffer a kernel writes through an argument buffer, executed twice: its records
/// lie in device-local memory where the device places them, held by no command buffer once both
/// completed, and nothing moves or is staged; the passes draw every command.
#[test]
fn gpu_encoded_shared_records_stay_on_the_device_across_executions() {
    let ours = Ours::new("icb-home-gpu");
    let icb = ours.icb(ResourceOptions::shared());
    let before = moves(&ours.device.work_counts());
    ours.encode_on_gpu(&icb);
    for _ in 0..2 {
        run(&ours.execute(&icb));
        assert_eq!(ours.drawn(), every_command());
    }
    assert_eq!(icb.records_on_device(), (ours.placing(), 0));
    assert_eq!(ours.moved_since(before), [0, 0, 0]);
    assert_eq!(ours.staged_notices(), 0);
}

/// A `Shared` buffer the CPU writes first: its records lie in the host's memory from the first
/// setter, a move of no bytes where the device places records, and no access is staged; the pass
/// draws every command and the getters answer them.
#[test]
fn cpu_encoded_shared_records_move_once_with_no_bytes() {
    let ours = Ours::new("icb-home-cpu");
    let icb = ours.icb(ResourceOptions::shared());
    let before = moves(&ours.device.work_counts());
    ours.encode_on_cpu(&icb);
    run(&ours.execute(&icb));
    assert_eq!(ours.drawn(), every_command());
    assert_eq!(ours.vertex_start(&icb, 7), 21);
    assert_eq!(icb.records_on_device(), (false, 0));
    assert_eq!(ours.moved_since(before), [u64::from(ours.placing()), 0, 0]);
    assert_eq!(ours.staged_notices(), 0);
}

/// A getter of a `Shared` buffer a kernel wrote, once no command buffer holds its records: they
/// move into the host's memory with the GPU's bytes, a getter answers what the kernel wrote, and
/// a later execution draws every command from there.
#[test]
fn a_getter_after_the_gpu_wrote_moves_the_records_with_their_bytes() {
    let ours = Ours::new("icb-home-getter");
    let icb = ours.icb(ResourceOptions::shared());
    ours.encode_on_gpu(&icb);
    let before = moves(&ours.device.work_counts());
    for index in [0, 5, u64::from(COMMANDS) - 1] {
        assert_eq!(ours.vertex_start(&icb, index), 3 * index);
    }
    let bytes = storage_bytes() * u64::from(ours.placing());
    assert_eq!(
        ours.moved_since(before),
        [u64::from(ours.placing()), bytes, 0]
    );
    assert_eq!(icb.records_on_device(), (false, 0));
    run(&ours.execute(&icb));
    assert_eq!(ours.drawn(), every_command());
    assert_eq!(ours.staged_notices(), 0);
}

/// A `Private` buffer's records never move: a kernel writes them, then each host access, a getter
/// and a setter, is staged where the device places records, announced once; the getter answers
/// what the kernel wrote and the pass draws the setter's write.
#[test]
fn private_records_never_move_and_their_host_accesses_are_staged() {
    let ours = Ours::new("icb-home-private");
    let icb = ours.icb(ResourceOptions::private());
    ours.encode_on_gpu(&icb);
    let before = moves(&ours.device.work_counts());
    assert_eq!(ours.vertex_start(&icb, 9), 27);
    let command = icb.indirect_render_command_at(0).unwrap();
    command
        .draw_primitives(PrimitiveType::Triangle, 3 * MOVED_TO, 3, 1, 0)
        .unwrap();
    run(&ours.execute(&icb));
    let mut expected: Vec<u64> = (1..u64::from(COMMANDS)).collect();
    expected.push(MOVED_TO);
    assert_eq!(ours.drawn(), expected);
    let [moved, bytes, staged] = ours.moved_since(before);
    assert_eq!((moved, bytes), (0, 0));
    match ours.placing() {
        true => assert!(staged >= 2, "{staged} staged accesses"),
        false => assert_eq!(staged, 0),
    }
    assert_eq!(icb.records_on_device(), (ours.placing(), 0));
    assert_eq!(ours.staged_notices(), usize::from(ours.placing()));
}

/// A getter and a setter of a `Shared` buffer a kernel wrote while a command buffer that executes
/// it is not yet committed: each is staged where the device places records, announced once, the
/// getter answering what the kernel wrote and the command buffer drawing the setter's write once
/// committed; once it completed, the next access moves the records, and is not staged.
#[test]
fn a_held_buffer_s_host_accesses_are_staged_until_its_holder_completes() {
    let ours = Ours::new("icb-home-held");
    let icb = ours.icb(ResourceOptions::shared());
    ours.encode_on_gpu(&icb);
    let holder = ours.execute(&icb);
    assert_eq!(
        icb.records_on_device(),
        (ours.placing(), u32::from(ours.placing()))
    );
    let before = moves(&ours.device.work_counts());
    assert_eq!(ours.vertex_start(&icb, 9), 27);
    let command = icb.indirect_render_command_at(0).unwrap();
    command
        .draw_primitives(PrimitiveType::Triangle, 3 * MOVED_TO, 3, 1, 0)
        .unwrap();
    let staged = ours.moved_since(before);
    match ours.placing() {
        true => assert_eq!(staged, [0, 0, 2]),
        false => assert_eq!(staged, [0, 0, 0]),
    }
    run(&holder);
    let mut expected: Vec<u64> = (1..u64::from(COMMANDS)).collect();
    expected.push(MOVED_TO);
    assert_eq!(ours.drawn(), expected);
    assert_eq!(icb.records_on_device(), (ours.placing(), 0));
    let before = moves(&ours.device.work_counts());
    assert_eq!(ours.vertex_start(&icb, 0), 3 * MOVED_TO);
    let bytes = storage_bytes() * u64::from(ours.placing());
    assert_eq!(
        ours.moved_since(before),
        [u64::from(ours.placing()), bytes, 0]
    );
    assert_eq!(icb.records_on_device(), (false, 0));
    assert_eq!(ours.staged_notices(), usize::from(ours.placing()));
}

/// A kernel writes a `Shared` buffer through an argument buffer, a pass executes it, a blit resets
/// its commands, a pass executes it again and draws nothing (the control: the check of what is
/// drawn sees an empty pass), the kernel writes it again and a pass draws every command: the
/// blit's reset is GPU work, so the records stay in device-local memory throughout where the
/// device places them, nothing moves and nothing is staged.
#[test]
fn a_blit_reset_between_gpu_encodes_keeps_the_records_on_the_device() {
    let ours = Ours::new("icb-home-reset");
    let icb = ours.icb(ResourceOptions::shared());
    let before = moves(&ours.device.work_counts());
    ours.encode_on_gpu(&icb);
    run(&ours.execute(&icb));
    assert_eq!(ours.drawn(), every_command());
    let queue = ours.device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.reset_commands_in_buffer(&icb, 0..u64::from(COMMANDS))
        .unwrap();
    blit.end_encoding().unwrap();
    run(&command_buffer);
    assert_eq!(icb.records_on_device(), (ours.placing(), 0));
    run(&ours.execute(&icb));
    assert_eq!(ours.drawn(), Vec::<u64>::new());
    ours.encode_on_gpu(&icb);
    run(&ours.execute(&icb));
    assert_eq!(ours.drawn(), every_command());
    assert_eq!(icb.records_on_device(), (ours.placing(), 0));
    assert_eq!(ours.moved_since(before), [0, 0, 0]);
    assert_eq!(ours.staged_notices(), 0);
}

/// A GPU reset clears what a command may have set (`lower::access::encode`'s `default_record`,
/// which writes the command's words, the bindings below each stage's count and the words past the
/// bindings), so a reset, an encode and an execution are exact: after the fixture's encode of every
/// command, a kernel that resets each command and encodes the even ones alone leaves every odd
/// command empty (no draw, no vertex or fragment buffer, as the getters read them) and the even
/// ones the fixture's, and the execution draws the even commands' cells alone. The negative
/// control: the same kernel without the reset leaves the odd commands what the first encode wrote,
/// their draws and buffers, and every cell drawn: the test sees what a reset leaves. Records of
/// both storage modes, wherever they lie.
#[test]
fn a_gpu_reset_then_an_encode_then_an_execution_are_exact() {
    let ours = Ours::new("icb-gpu-reset");
    let resetting = ours.writer_of(&reset_writer_text(true));
    let unresetting = ours.writer_of(&reset_writer_text(false));
    let evens: Vec<u64> = every_command().into_iter().filter(|c| c % 2 == 0).collect();
    for options in [ResourceOptions::shared(), ResourceOptions::private()] {
        for (writer, reset) in [(&resetting, true), (&unresetting, false)] {
            let icb = ours.icb(options);
            ours.encode_on_gpu(&icb);
            ours.encode_with(&icb, writer);
            run(&ours.execute(&icb));
            let want = match reset {
                true => evens.clone(),
                false => every_command(),
            };
            assert_eq!(ours.drawn(), want, "{options:?} reset {reset}");
            for index in every_command() {
                let command = icb.indirect_render_command_at(index).unwrap();
                let empty = reset && index % 2 == 1;
                let draw = command.draw().unwrap();
                let vertex = command.vertex_buffer(0).unwrap();
                let fragment = command.fragment_buffer(1).unwrap();
                match empty {
                    true => assert!(
                        draw.is_none() && vertex.is_none() && fragment.is_none(),
                        "{options:?} {index}: {draw:?} {vertex:?} {fragment:?}"
                    ),
                    false => {
                        assert_eq!(ours.vertex_start(&icb, index), 3 * index);
                        assert!(vertex.is_some() && fragment.is_some(), "{index}");
                    }
                }
            }
        }
    }
}

/// A test timeline of `device`'s at value 0, which [`Device::hold_staging_submissions`] holds the
/// staging submissions on until the test signals it.
fn hold_timeline(device: &Device) -> vk::Semaphore {
    let mut kind = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(0);
    let info = vk::SemaphoreCreateInfo::default().push_next(&mut kind);
    // SAFETY: the device is live and enables `timelineSemaphore`; `info` is live for the call.
    unsafe { device.shared().vk().create_semaphore(&info, None) }.unwrap()
}

/// A buffer whose records a use placed in device-local memory goes wherever its last hold lets
/// go, one of the runtime's threads among them (a command buffer's completion, on its queue's
/// waiter or a handler thread): its home waits there for the staging submission that placed the
/// records, where the host has not seen it complete, inside a section of the exit guard
/// (`home.rs`), as every Vulkan call of the runtime's threads (`exit_guard::check`, which in these
/// builds ends the process at one made outside a section). The placing submission is held on the
/// GPU ([`Device::hold_staging_submissions`]) until the buffer's last hold went on a thread of the
/// runtime's ([`exit_guard::spawn`]), so the host cannot have seen it complete, whatever polls the
/// staging queue meanwhile. A device that keeps every buffer's records in the host's memory places
/// none and waits for none.
#[test]
fn a_buffer_let_go_of_last_on_a_runtime_thread_waits_for_its_placing_inside_a_section() {
    let ours = Ours::new("icb-home-runtime-drop");
    let name = ours.device.name().to_owned();
    if !ours.placing() {
        println!("{name}: keeps every buffer's records in the host's memory: nothing placed");
        return;
    }
    let hold = hold_timeline(&ours.device);
    ours.device.hold_staging_submissions(Some((hold, 1)));
    let icb = ours.icb(ResourceOptions::private());
    // Its residency in a command buffer never committed: the GPU's first use, which places the
    // records, and a hold that goes with the command buffer.
    {
        let queue = ours.device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder
            .use_resource(icb.as_resource(), ResourceUsage::READ)
            .unwrap();
        encoder.end_encoding().unwrap();
    }
    ours.device.hold_staging_submissions(None);
    assert_eq!(
        icb.records_on_device(),
        (true, 0),
        "{name}: the use placed the records, and nothing holds them"
    );
    let waits = || crate::indirect_command_buffer::RUNTIME_PLACING_WAITS.load(Ordering::Relaxed);
    let before = waits();
    let builder = std::thread::Builder::new().name("icb-home-drop".into());
    let dropped = crate::exit_guard::spawn(builder, move || drop(icb)).unwrap();
    let signal = vk::SemaphoreSignalInfo::default().semaphore(hold).value(1);
    // SAFETY: a host signal of the test's timeline, above its value.
    unsafe { ours.device.shared().vk().signal_semaphore(&signal) }.unwrap();
    dropped.join().unwrap();
    let waited = waits() - before;
    println!(
        "{name}: {waited} drops on a runtime thread waited for the placing submission the host had \
         not seen complete"
    );
    assert_eq!(
        waited, 1,
        "{name}: the drop waited for the placing submission on one of the runtime's threads"
    );
    // SAFETY: the semaphore is the test's, and no submission waits on it any more.
    unsafe { ours.device.shared().vk().destroy_semaphore(hold, None) };
}
