//! The placements of a heap whose bytes meet (`crate::alias`), under a device cfg, on every device
//! of the cfg's kind, read through `Metrics::heap_moves` and through the checks each segment of a
//! command buffer holds before its commit. P and Q are textures at offset 0 of one heap, P placed
//! first, so Q is stale from its making; each is written through its view (`rows_write`).
//!
//! - Commit order against encode order: A writes P and B writes Q, encoded A then B and committed
//!   B then A. Each holds one check; the submit thread resolves them in the order the queue runs
//!   them, so Q's move goes first and leaves P stale for A's.
//! - A read of an alias, then a write of a texture nothing earlier in the command buffer wrote:
//!   the segment ends between them, and the check sits at the head of the second.
//! - Writes of P, Q and P in one command buffer: the second and third move in place.
//! - A draw whose fragment writes P after a dispatch wrote Q: P moves through the pass's
//!   interruption.
//! - A heap whose two textures' bytes do not meet: no links, no check and no move.
//!
//! `llvm-as` assembles the kernels' libraries; without it the tests are skipped, visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished};
    use crate::tests::draw::on_device::{library, pass, state};
    use crate::tests::heap_alias::on_device::{
        input, kernels, libraries, place, place_at, read, write, Buffers, Kernels, Placed,
    };
    use crate::texture::TextureObject;
    use crate::{
        Block, Buffer, ColorAttachmentDescriptor, CommandBuffer, CommandQueue, CullMode, Device,
        FunctionDescriptor, HeapRange, IndexClass, MemoryClass, PixelFormat, PrimitiveType,
        RenderPipelineDescriptor, StageSource, StorageMode, TextureDescriptor, TextureType,
        TextureUsage,
    };
    use metal2vulkan_devtools::fixtures_dir;

    /// The first texture index the tests grant.
    const FIRST: u32 = 4600;

    fn moves(device: &Device) -> u64 {
        device.counters().metrics().heap_moves
    }

    /// A Private block and a heap over all of it.
    fn heap(name: &str, device: &Device) -> (Block, HeapRange) {
        let len = MemoryClass::Private.block_len();
        let block = (device.take_block(MemoryClass::Private, len))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let heap = HeapRange {
            block: block.id,
            offset: 0,
            len,
        };
        (block, heap)
    }

    /// P and Q, 64 by 64 `RGBA16Float` at offset 0 of `heap`, P placed first, and the inputs
    /// `rows_write` writes into each.
    struct Pair {
        p: Placed,
        q: Placed,
        inputs: [Buffer; 2],
        _buffers: Buffers,
    }

    fn pair(name: &str, device: &Device, heap: HeapRange) -> Pair {
        let format = PixelFormat::RGBA16Float;
        let p = place(name, device, heap, (0, format, 64, 64), FIRST);
        let q = place(name, device, heap, (1, format, 64, 64), FIRST + 2);
        let mut buffers = Buffers::new(name, device, 4);
        let inputs = [&p, &q].map(|t| buffers.of(name, device, &input(t)));
        Pair {
            p,
            q,
            inputs,
            _buffers: buffers,
        }
    }

    /// Runs one command buffer of `queue` that writes `order`'s textures in turn.
    fn settle(name: &str, queue: &CommandQueue, kernels: &Kernels, order: [(&Placed, &Buffer); 2]) {
        let cb = queue.command_buffer().expect("a command buffer");
        for (t, input) in order {
            write(name, &cb, kernels, t, input);
        }
        run(name, &cb);
    }

    fn run(name: &str, cb: &CommandBuffer) {
        cb.commit().expect("a commit");
        finished(name, cb);
        assert_eq!(cb.error(), None, "{name}");
    }

    /// Commit order against encode order: A writes P and B writes Q, encoded A then B, committed
    /// B then A, after a command buffer that wrote Q then P (Q's move from its making, P's in
    /// place). Q is stale and P is not as B runs; B's write of Q leaves P stale for A, so each
    /// check finds a move owed, as the queue runs them.
    #[test]
    fn every_device_resolves_heap_checks_in_the_order_the_queue_runs_them() {
        let Some(libraries) = libraries() else {
            return;
        };
        each_device("alias_order", |name, _, device| {
            let kernels = kernels(name, &device, &libraries);
            let _grant = (device.grant_indices(IndexClass::Texture, FIRST, 4))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let (block, heap) = heap(name, &device);
            let Pair { p, q, inputs, .. } = pair(name, &device, heap);
            let queue = device.new_command_queue(4).expect("a queue");
            let start = moves(&device);
            settle(name, &queue, &kernels, [(&q, &inputs[1]), (&p, &inputs[0])]);
            assert_eq!(
                moves(&device) - start,
                2,
                "{name}: Q from its making, P in place"
            );
            let before = moves(&device);
            let a = queue.command_buffer().expect("a command buffer");
            write(name, &a, &kernels, &p, &inputs[0]);
            let b = queue.command_buffer().expect("a command buffer");
            write(name, &b, &kernels, &q, &inputs[1]);
            assert_eq!(a.alias_checks_for_test(), [1], "{name}: A");
            assert_eq!(b.alias_checks_for_test(), [1], "{name}: B");
            assert_eq!(moves(&device), before, "{name}: nothing moves in place");
            run(name, &b);
            run(name, &a);
            assert_eq!(moves(&device) - before, 2, "{name}: Q's move, then P's");
            drop((p, q, inputs, queue));
            (device.release_block(block)).unwrap_or_else(|error| panic!("{name}: {error}"));
            println!("heap checks: {name}: resolved in the order the queue ran them");
        });
    }

    /// A read of Q, then a write of P, which nothing earlier in the command buffer wrote nor an
    /// alias of it: the segment ends between them, and P's check heads the second, where the
    /// move it finds owed (P stale after a command buffer that wrote P then Q) runs after the
    /// read. A command buffer with no such read holds the check in its one segment.
    #[test]
    fn every_device_ends_a_segment_before_a_heap_check_behind_an_alias_s_read() {
        let Some(libraries) = libraries() else {
            return;
        };
        each_device("alias_split", |name, _, device| {
            let kernels = kernels(name, &device, &libraries);
            let _grant = (device.grant_indices(IndexClass::Texture, FIRST, 4))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let (block, heap) = heap(name, &device);
            let Pair { p, q, inputs, .. } = pair(name, &device, heap);
            let mut buffers = Buffers::new(name, &device, 1);
            let out = buffers.of(name, &device, &vec![0.0; input(&q).len()]);
            let queue = device.new_command_queue(4).expect("a queue");
            settle(name, &queue, &kernels, [(&p, &inputs[0]), (&q, &inputs[1])]);
            let before = moves(&device);
            let cb = queue.command_buffer().expect("a command buffer");
            read(name, &cb, &kernels, &q, &out);
            write(name, &cb, &kernels, &p, &inputs[0]);
            assert_eq!(cb.alias_checks_for_test(), [0, 1], "{name}: split");
            run(name, &cb);
            assert_eq!(
                moves(&device) - before,
                1,
                "{name}: P's move after Q's read"
            );
            let cb = queue.command_buffer().expect("a command buffer");
            write(name, &cb, &kernels, &q, &inputs[1]);
            assert_eq!(cb.alias_checks_for_test(), [1], "{name}: no read, no split");
            run(name, &cb);
            drop((p, q, inputs, out, buffers, queue));
            (device.release_block(block)).unwrap_or_else(|error| panic!("{name}: {error}"));
            println!("heap checks: {name}: a check behind an alias's read heads a segment");
        });
    }

    /// Writes of P, Q and P in one command buffer, after one that wrote P then Q: P's first write
    /// is a check, which finds P stale; Q's and P's second move in place, each after its alias's
    /// write.
    #[test]
    fn every_device_moves_heap_textures_in_place_after_their_alias_s_write() {
        let Some(libraries) = libraries() else {
            return;
        };
        each_device("alias_in_place", |name, _, device| {
            let kernels = kernels(name, &device, &libraries);
            let _grant = (device.grant_indices(IndexClass::Texture, FIRST, 4))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let (block, heap) = heap(name, &device);
            let Pair { p, q, inputs, .. } = pair(name, &device, heap);
            let queue = device.new_command_queue(4).expect("a queue");
            settle(name, &queue, &kernels, [(&p, &inputs[0]), (&q, &inputs[1])]);
            let before = moves(&device);
            let cb = queue.command_buffer().expect("a command buffer");
            write(name, &cb, &kernels, &p, &inputs[0]);
            write(name, &cb, &kernels, &q, &inputs[1]);
            write(name, &cb, &kernels, &p, &inputs[0]);
            assert_eq!(cb.alias_checks_for_test(), [1], "{name}: P's first write");
            assert_eq!(moves(&device) - before, 2, "{name}: Q and P in place");
            run(name, &cb);
            assert_eq!(moves(&device) - before, 3, "{name}: and P's check");
            drop((p, q, inputs, queue));
            (device.release_block(block)).unwrap_or_else(|error| panic!("{name}: {error}"));
            println!("heap checks: {name}: moves in place after an alias's write");
        });
    }

    /// The draw's target and fragment texture's extent.
    const WIDTH: u32 = 32;
    const HEIGHT: u32 = 16;

    /// A dispatch writes Q, then a draw whose fragment (`lw_late_write`) writes P, an
    /// `RGBA32Float` texture at Q's offset: P moves in place, through the pass's interruption,
    /// and the fragment runs.
    #[test]
    fn every_device_moves_a_draw_s_written_heap_texture_through_the_pass() {
        let Some(libraries) = libraries() else {
            return;
        };
        let Some(writes) = library(&fixtures_dir().join("examples/lost_writes.ll")) else {
            return;
        };
        each_device("alias_draw", |name, _, device| {
            let kernels = kernels(name, &device, &libraries);
            let _grant = (device.grant_indices(IndexClass::Texture, FIRST, 4))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let (block, heap) = heap(name, &device);
            let format = PixelFormat::RGBA16Float;
            let q = place(name, &device, heap, (1, format, 64, 64), FIRST);
            let p = place(
                name,
                &device,
                heap,
                (0, PixelFormat::RGBA32Float, WIDTH, HEIGHT),
                FIRST + 2,
            );
            let mut buffers = Buffers::new(name, &device, 2);
            let q_in = buffers.of(name, &device, &input(&q));
            let runs = buffers.of(name, &device, &[0.0]);
            let function = |name| {
                StageSource::Function(FunctionDescriptor::new(
                    writes.function(name).expect("the function"),
                ))
            };
            let mut d = RenderPipelineDescriptor::new(function("lw_cover"));
            d.fragment = Some(function("lw_late_write"));
            d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
            let drawn = state(name, &device, &d);
            let mut target =
                TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, WIDTH, HEIGHT);
            target.usage = TextureUsage::RENDER_TARGET;
            target.storage_mode = StorageMode::Private;
            let target = (device.new_texture_for_views(&target))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let queue = device.new_command_queue(4).expect("a queue");
            let before = moves(&device);
            let cb = queue.command_buffer().expect("a command buffer");
            write(name, &cb, &kernels, &q, &q_in);
            let mut encoder = (cb.render_command_encoder(&pass(&target, None)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let done = |result: Result<(), crate::Error>| {
                result.unwrap_or_else(|error| panic!("{name}: {error}"))
            };
            done(encoder.set_render_pipeline_state(&drawn));
            encoder.set_cull_mode(CullMode::None);
            done(encoder.set_fragment_texture(Some(&p.texture), 0));
            done(encoder.set_fragment_buffer(Some(&runs), 0, 0));
            done(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
            encoder.end_encoding();
            assert_eq!(cb.alias_checks_for_test(), [1], "{name}: Q's write");
            assert_eq!(
                moves(&device) - before,
                1,
                "{name}: P in place, in the pass"
            );
            run(name, &cb);
            let contents = runs.contents().expect("a Shared buffer");
            // SAFETY: the buffer holds one word, which the finished draw wrote and whose writes
            // reach the host.
            let ran = unsafe { contents.as_ptr().cast::<u32>().read_unaligned() };
            assert!(ran > 0, "{name}: the fragment ran");
            drop((p, q, q_in, runs, buffers, target, cb, queue));
            (device.release_block(block)).unwrap_or_else(|error| panic!("{name}: {error}"));
            println!("heap checks: {name}: a draw's write moved through the pass ({ran} runs)");
        });
    }

    /// Two textures of one heap whose bytes do not meet, written in turn: no links, no check, no
    /// move and one segment, in one command buffer and across two.
    #[test]
    fn every_device_takes_no_check_or_move_on_a_heap_of_apart_placements() {
        let Some(libraries) = libraries() else {
            return;
        };
        each_device("alias_apart", |name, _, device| {
            let kernels = kernels(name, &device, &libraries);
            let _grant = (device.grant_indices(IndexClass::Texture, FIRST, 4))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let (block, heap) = heap(name, &device);
            let format = PixelFormat::RGBA16Float;
            let mut d = TextureDescriptor::new(TextureType::Type2D, format, 64, 64);
            d.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
            d.storage_mode = StorageMode::Private;
            let fit = (device.heap_texture_size_and_align(&d))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let apart = fit.size.next_multiple_of(fit.align);
            let p = place(name, &device, heap, (0, format, 64, 64), FIRST);
            let r = place_at(name, &device, (heap, apart), (1, format, 64, 64), FIRST + 2);
            for t in [&p, &r] {
                let place = (t.texture.object().place()).expect("a heap texture's placement");
                assert!(!place.linked(), "{name}: no links");
            }
            let mut buffers = Buffers::new(name, &device, 2);
            let inputs = [&p, &r].map(|t| buffers.of(name, &device, &input(t)));
            let queue = device.new_command_queue(4).expect("a queue");
            let before = moves(&device);
            let cb = queue.command_buffer().expect("a command buffer");
            for _ in 0..2 {
                write(name, &cb, &kernels, &p, &inputs[0]);
                write(name, &cb, &kernels, &r, &inputs[1]);
            }
            assert_eq!(
                cb.alias_checks_for_test(),
                [0],
                "{name}: one segment, no check"
            );
            run(name, &cb);
            settle(name, &queue, &kernels, [(&r, &inputs[1]), (&p, &inputs[0])]);
            assert_eq!(moves(&device), before, "{name}: no move");
            drop((p, r, inputs, buffers, queue));
            (device.release_block(block)).unwrap_or_else(|error| panic!("{name}: {error}"));
            println!("heap checks: {name}: apart placements take no check or move");
        });
    }
}
