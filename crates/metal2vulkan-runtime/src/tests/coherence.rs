//! Shared memory's coherence (`crate::coherence`). Under a device cfg, on every device of the cfg's
//! kind: the device holds a coherence state exactly where its Shared type, as it reads it, is not
//! host-coherent, and makes no flush or invalidate where it holds none. Where it holds one (every
//! device the floor opens through the refused device's view,
//! `crate::tests::command::on_device::cached_types_incoherent`): the zero source is flushed at its
//! make; a fill into a Shared buffer is flushed over the buffer's atoms before its submission, then
//! flushed and invalidated there before its status reads completed, and its bytes are the fill's; a
//! fill into a Private buffer makes no call; the bytes a texture's `replace_region` stages are
//! flushed once, and those its `get_bytes` reads flushed and invalidated once. Each call is counted
//! with its bytes, and none fails. A device whose one host-visible, host-cached buffer type also
//! holds the parameter ring is refused under the view, with its refusal.
//!
//! Under the view, each of these sites makes its calls too, each counted by its ranges' bytes
//! widened to the device's atom, with no call failing, and the bytes read after them are the
//! bytes written:
//!
//! - a texture over a Shared buffer and a view of it, in place and in an image of its own, read
//!   by a blit after a host write to the buffer: the submission flushes the texture's buffer, and
//!   the completion settles it too where the texture is an image of its own;
//! - a Shared buffer a kernel reaches only through an address, declared with `use_resource`:
//!   flushed before the submission, and settled at the completion where the declaration writes;
//! - a Shared heap's block, whose buffer a kernel reaches only through an address, declared with
//!   `use_heap`: flushed whole before the submission and settled whole at the completion;
//! - a blit's Shared source: flushed before the submission, nothing settled for it;
//! - a visible function table's words, which the host writes: flushed before the submission of a
//!   dispatch that calls through the table, and not settled;
//! - a shared event's value a GPU signal raised: before `signaledValue` or a wait shows it, what
//!   the command buffer in flight that signalled it may write is flushed and invalidated;
//! - a glyph or tile upload's route, a blit of a sub-range of a 4 KiB Shared source the host
//!   wrote, its rows no multiple of 64 bytes apart, into an R8 texture over a Shared buffer that
//!   is also a render attachment: the source and the texture's buffer flushed before the
//!   submission, the texture's buffer settled before the host reads it; a source in host memory's
//!   import, whose type is host-coherent, takes no call;
//! - a range a submission only writes, a new Shared buffer the host zeroed, written by the GPU
//!   alone in the first submission that reaches it, as a blit's destination and as an
//!   attachment's buffer: flushed before that submission, settled at its completion.
//!
//! `llvm-as` assembles the kernels' fixtures; without it those tests skip visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::blit::on_device::settled;
    use crate::tests::command::on_device::{
        each_device_under_the_view, finished, on_each_device, shared_event, within, PATIENCE,
    };
    use crate::tests::import::on_device::{free, host_memory, releasing, wait_until};
    use crate::texture::TextureObject;
    use crate::{
        Buffer, BufferLayout, ClearColor, CommandBuffer, CommandBufferStatus, CommandQueue,
        ComputeCommandEncoder, ComputePipelineDescriptor, ComputePipelineState, Device,
        DispatchType, Error, FunctionDescriptor, FunctionHandle, HostBytes, IndexClass, Library,
        Limit, LinkedFunction, LoadAction, MemoryClass, Metrics, PixelFormat, Region,
        RenderPassDescriptor, Resource, ResourceUsage, SharedEvent, StageSource, Texture,
        TextureDescriptor, TexturePlace, TextureType, TextureUsage, VisibleFunctionTableDescriptor,
    };
    use metal2vulkan::bind::function_table_entry_offset;
    use metal2vulkan::pipeline::LinkedFunctions;
    use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};
    use std::sync::atomic::Ordering;

    /// From `before` to `after`: the flushes, their bytes, the invalidates, their bytes, and the
    /// calls that failed.
    fn calls(before: &Metrics, after: &Metrics) -> [u64; 5] {
        [
            after.shared_flushes - before.shared_flushes,
            after.shared_flushed_bytes - before.shared_flushed_bytes,
            after.shared_invalidates - before.shared_invalidates,
            after.shared_invalidated_bytes - before.shared_invalidated_bytes,
            after.shared_coherence_failures - before.shared_coherence_failures,
        ]
    }

    /// The bytes a flush or an invalidate names of the `len` bytes at `at` in an allocation:
    /// widened to whole atoms of `atom` bytes.
    fn atoms(at: u64, len: u64, atom: u64) -> u64 {
        (at + len).next_multiple_of(atom) - (at - at % atom)
    }

    /// The calls of one command buffer that runs in one submission, reaching `reached` bytes of
    /// Shared memory and maybe writing `written` of them: the submission's flush of every byte it
    /// reaches, then, where it may write any, its completion's flush and invalidate of those.
    fn one_submission(reached: u64, written: u64) -> [u64; 5] {
        let settles = u64::from(written > 0);
        [1 + settles, reached + written, settles, written, 0]
    }

    /// The device's `nonCoherentAtomSize`, held by the coherence state the view built.
    fn atom(name: &str, device: &Device) -> u64 {
        let atom = (device.core().coherence.as_ref()).map(|coherence| coherence.atom);
        atom.unwrap_or_else(|| panic!("{name}: the view built the coherence state"))
    }

    /// `command_buffer` committed and complete with no error.
    fn run(name: &str, command_buffer: &CommandBuffer) {
        command_buffer.commit().expect("a commit");
        finished(name, command_buffer);
        assert_eq!(command_buffer.error(), None, "{name}");
    }

    /// Writes `bytes` over the first bytes of `buffer`, which no command uses meanwhile.
    fn write(buffer: &Buffer, bytes: &[u8]) {
        assert!(bytes.len() as u64 <= buffer.length(), "the bytes fit");
        let contents = buffer.contents().expect("host bytes");
        // SAFETY: the buffer is alive and holds `bytes.len()` bytes, which no command uses
        // meanwhile.
        unsafe {
            (contents.as_ptr()).copy_from_nonoverlapping(bytes.as_ptr(), bytes.len());
        }
    }

    /// The bytes of `buffer`, read once the work that wrote them completed.
    fn bytes_of(buffer: &Buffer) -> Vec<u8> {
        let contents = buffer.contents().expect("host bytes");
        // SAFETY: the buffer is alive and holds its length's bytes, which no command writes
        // meanwhile.
        unsafe { std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize) }.to_vec()
    }

    /// `len` bytes, the `i`th `seed + 7 i`.
    fn pattern(len: u64, seed: u8) -> Vec<u8> {
        (0..len)
            .map(|i| seed.wrapping_add((i as u8).wrapping_mul(7)))
            .collect()
    }

    /// The `height` rows of `row` bytes that lie `pitch` apart from the start of `bytes`, packed.
    fn rows(bytes: &[u8], (row, pitch, height): (u64, u64, u32)) -> Vec<u8> {
        (0..u64::from(height))
            .flat_map(|at| {
                let start = (at * pitch) as usize;
                bytes[start..start + row as usize].iter().copied()
            })
            .collect()
    }

    /// The fixture `fixture`'s library, or None where `llvm-as` is missing.
    fn library(fixture: &str) -> Option<Library> {
        match llvm_assemble(&engine_dir().join(format!("{fixture}.ll"))) {
            Ok(bitcode) => Some(Library::from_air(&bitcode).expect("the library")),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                None
            }
            Err(error) => panic!("{fixture}: {error:?}"),
        }
    }

    /// A command buffer of `queue` with one fill of `range` of `buffer` to `value`, committed and
    /// complete with no error.
    fn fill(
        name: &str,
        queue: &CommandQueue,
        buffer: &Buffer,
        range: std::ops::Range<u64>,
        value: u8,
    ) {
        let command_buffer: CommandBuffer = queue.command_buffer().expect("a command buffer");
        let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
        (encoder.fill_buffer(buffer, range, value))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        encoder.end_encoding();
        run(name, &command_buffer);
    }

    /// The calls of the zero source's make, then of a fill into a Shared buffer, of one into a
    /// Private buffer, and of a texture's copies through staging, with `atom` the device's
    /// `nonCoherentAtomSize` where it holds a coherence state, else `None`. Asserts the fill's
    /// bytes and the texture's.
    fn calls_of_the_work(name: &str, device: &Device, atom: Option<u64>) {
        const OFFSET: u64 = 256;
        const LEN: u64 = 64;
        let counters = device.counters();
        settled(device);
        within("the zero source", || device.core().filler.get().is_some());
        let made = counters.metrics();
        println!(
            "coherence: {name}: atom {atom:?}; at the zero source's make: flushes {}, bytes {}",
            made.shared_flushes, made.shared_flushed_bytes
        );

        let shared = (device.take_block(MemoryClass::Shared, 4096)).expect("a Shared block");
        let buffer = (device.new_buffer(shared.id, OFFSET, LEN, Some(0))).expect("a buffer");
        let private = (device.take_block(MemoryClass::Private, 4096)).expect("a Private block");
        let unseen = (device.new_buffer(private.id, 0, LEN, None)).expect("a buffer");
        let queue = device.new_command_queue(4).expect("a queue");
        let recording: CommandBuffer = queue.command_buffer().expect("a command buffer");
        assert_eq!(
            (recording.holds_coherence_for_test())
                .unwrap_or_else(|error| panic!("{name}: {error}")),
            atom.is_some(),
            "{name}: a command buffer holds the coherence state exactly where its device does"
        );
        drop(recording);
        // The first fill of each, so the second meets no first use.
        fill(name, &queue, &buffer, 0..LEN, 1);
        fill(name, &queue, &unseen, 0..LEN, 1);

        let before = counters.metrics();
        fill(name, &queue, &buffer, 0..LEN, 0x5a);
        let after = counters.metrics();
        let into_shared = calls(&before, &after);
        let bytes = bytes_of(&buffer);

        let before = counters.metrics();
        fill(name, &queue, &unseen, 0..LEN, 0x5a);
        let into_private = calls(&before, &counters.metrics());

        let d = TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, 4, 4);
        // A texture with no index: only its copies reach it.
        let texture =
            (device.new_texture_for_views(&d)).unwrap_or_else(|error| panic!("{name}: {error}"));
        let whole = Region {
            origin: [0; 3],
            size: [4, 4, 1],
        };
        let texels: Vec<u8> = (0..64).collect();
        let before = counters.metrics();
        (texture.replace_region(whole, 0, 0, HostBytes::from(&texels[..]), 16, 0))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let replaced = counters.metrics();
        let mut read = vec![0u8; 64];
        (texture.get_bytes(&mut read, 16, 0, whole, 0, 0))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let staged_in = calls(&before, &replaced);
        let staged_out = calls(&replaced, &counters.metrics());
        println!(
            "coherence: {name}: flushes, flushed bytes, invalidates, invalidated bytes, \
             failures: a fill into Shared {into_shared:?}, into Private {into_private:?}, \
             replace_region {staged_in:?}, get_bytes {staged_out:?}"
        );
        assert!(bytes.iter().all(|&byte| byte == 0x5a), "{name}: the fill");
        assert_eq!(read, texels, "{name}: the texture's bytes");
        assert_eq!(into_private, [0; 5], "{name}: a fill into a Private buffer");
        let Some(atom) = atom else {
            assert_eq!(
                (made.shared_flushes, made.shared_invalidates),
                (0, 0),
                "{name}: the zero source's make, with no coherence state"
            );
            for (work, calls) in [
                ("a fill into a Shared buffer", into_shared),
                ("replace_region", staged_in),
                ("get_bytes", staged_out),
            ] {
                assert_eq!(calls, [0; 5], "{name}: {work}, with no coherence state");
            }
            return;
        };
        assert!(
            made.shared_flushes >= 1,
            "{name}: the zero source was flushed at its make"
        );
        // The buffer's bytes in its block, widened to whole atoms; a block starts at a multiple
        // of 64 KiB of its allocation.
        let widened = atoms(OFFSET, LEN, atom);
        assert_eq!(
            into_shared,
            [2, 2 * widened, 1, widened, 0],
            "{name}: a fill into a Shared buffer: its submission's flush, then its completion's \
             flush and invalidate, each over the buffer's atoms ({widened} bytes)"
        );
        assert_eq!(
            [staged_in[0], staged_in[2], staged_in[4]],
            [1, 0, 0],
            "{name}: replace_region's flushes, invalidates and failures"
        );
        assert!(
            staged_in[1] >= 64,
            "{name}: replace_region flushed its 64 bytes"
        );
        assert_eq!(
            [staged_out[0], staged_out[2], staged_out[4]],
            [1, 1, 0],
            "{name}: get_bytes's flushes, invalidates and failures"
        );
        assert!(
            staged_out[1] >= 64 && staged_out[3] >= 64,
            "{name}: get_bytes flushed and invalidated its 64 bytes"
        );
    }

    /// Every device of the cfg's kind, made as the device tests make it: it holds a coherence
    /// state exactly where its Shared type lacks `HOST_COHERENT` as it reads it, and its work's
    /// calls follow. Under `m2v_shared_view` the device reads its types through the suite's view,
    /// which leaves no cached type coherent, so there every device builds the state.
    #[test]
    fn every_device_flushes_and_invalidates_only_where_its_shared_type_is_not_coherent() {
        use ash::vk::MemoryPropertyFlags as F;
        let runtime = crate::tests::validation::instance(&[]);
        let (_, vk_instance) = runtime.raw();
        on_each_device("coherence", &runtime, |name, _, device| {
            let physical = device.raw().physical_device;
            // SAFETY: the device's physical device is one of the instance's.
            let mut memory = unsafe { vk_instance.get_physical_device_memory_properties(physical) };
            if cfg!(m2v_shared_view) {
                memory = crate::tests::command::on_device::viewed(memory);
            }
            let index = device.memory_types().shared;
            let shared = memory.memory_types_as_slice()[index as usize].property_flags;
            let atom = (device.core().coherence.as_ref()).map(|coherence| coherence.atom);
            let state = if atom.is_some() { "built" } else { "none" };
            let flags = crate::floor::flag_names(shared);
            println!(
                "coherence: {name}: Shared type {index} with {flags}; coherence state {state}"
            );
            assert_eq!(
                atom.is_some(),
                !shared.contains(F::HOST_COHERENT),
                "{name}: the coherence state is built exactly where Shared's type is not coherent"
            );
            assert!(
                atom.is_some() || !cfg!(m2v_shared_view),
                "{name}: under m2v_shared_view every device builds the coherence state"
            );
            calls_of_the_work(name, &device, atom);
        });
    }

    /// Every device of the cfg's kind, read through the refused device's view
    /// (`crate::tests::command::on_device::cached_types_incoherent`), decided by the floor's choice
    /// over the view (`each_device_under_the_view`). A device the choice opens has a Shared type
    /// that is cached and not coherent, so it holds the coherence state and makes the calls, on
    /// memory that is coherent in fact. A device whose one host-visible, host-cached buffer type
    /// also holds the parameter ring is left with no coherent type for the ring, so the floor
    /// refuses it under the view, with that choice's refusal, which is printed.
    #[test]
    fn every_device_opened_under_the_view_flushes_and_invalidates_the_shared_memory_work_reaches() {
        each_device_under_the_view("coherence", |name, _, device, _| {
            let atom = (device.core().coherence.as_ref()).map(|coherence| coherence.atom);
            assert!(atom.is_some(), "{name}: the view built the coherence state");
            calls_of_the_work(name, &device, atom);
        });
    }

    /// Under the view: a texture over a Shared buffer, in place at the device's own pitch and in
    /// an image of its own at one alignment past it, and a view of each, read by a blit into a
    /// Shared buffer after a host write to the texture's buffer. The blit's submission flushes the
    /// texture's buffer and the destination, each over its atoms; its completion flushes and
    /// invalidates the destination, and the texture's buffer too where the texture is an image of
    /// its own, whose bytes the submissions copy back. The destination holds the host's texels.
    #[test]
    fn every_device_opened_under_the_view_flushes_a_texture_over_a_shared_buffer_and_its_view() {
        const HEIGHT: u32 = 4;
        each_device_under_the_view("coherence_over", |name, _, device, _| {
            let atom = atom(name, &device);
            let counters = device.counters();
            let format = PixelFormat::RGBA8Unorm;
            let align = (device.minimum_linear_texture_alignment(format))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            // Rows of the device's own pitch, a row of texels each: a linear image takes them.
            let row = align.max(256);
            let width = u32::try_from(row / 4).expect("a width");
            let mut d = TextureDescriptor::new(TextureType::Type2D, format, width, HEIGHT);
            d.usage = TextureUsage::SHADER_READ;
            let _grant = (device.grant_indices(IndexClass::Texture, 4096, 4))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let queue = device.new_command_queue(4).expect("a queue");
            let packed = row * u64::from(HEIGHT);
            let out_block = (device.take_block(MemoryClass::Shared, packed)).expect("a block");
            let out = (device.new_buffer(out_block.id, 0, packed, Some(0))).expect("a buffer");
            // Writes the host's bytes into `buffer`, then reads `texture` into `out` by a blit of
            // a command buffer of its own: the calls it made.
            let read = |texture: &Texture, buffer: &Buffer, bytes: &[u8]| {
                write(buffer, bytes);
                let before = counters.metrics();
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
                let layout = BufferLayout {
                    offset: 0,
                    bytes_per_row: row,
                    bytes_per_image: 0,
                };
                let size = [width, HEIGHT, 1];
                let at = TexturePlace::default();
                (encoder.copy_from_texture_to_buffer(texture, at, size, &out, layout))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                encoder.end_encoding();
                run(name, &command_buffer);
                calls(&before, &counters.metrics())
            };
            let mut indices = 4096..;
            let mut seeds = 1..;
            for (form, pitch) in [("in place", row), ("an image of its own", row + align)] {
                let len = pitch * u64::from(HEIGHT);
                let block = (device.take_block(MemoryClass::Shared, len)).expect("a block");
                let buffer = (device.new_buffer(block.id, 0, len, None)).expect("a buffer");
                let texture = (buffer.new_texture(&d, indices.next(), 0, pitch))
                    .unwrap_or_else(|error| panic!("{name}: {form}: {error}"));
                let copied = texture.copied_over().is_some();
                assert_eq!(copied, pitch != row, "{name}: {form}");
                let view = (texture.new_texture_view(
                    format,
                    TextureType::Type2D,
                    0..1,
                    0..1,
                    indices.next(),
                ))
                .unwrap_or_else(|error| panic!("{name}: {form}: a view: {error}"));
                let own = atoms(block.offset, len, atom);
                let into = atoms(out_block.offset, packed, atom);
                let written = into + if copied { own } else { 0 };
                for (what, texture) in [("the texture", &texture), ("its view", &view)] {
                    // The first read takes the image's first use; the second is counted.
                    read(
                        texture,
                        &buffer,
                        &pattern(len, seeds.next().expect("a seed")),
                    );
                    let host = pattern(len, seeds.next().expect("a seed"));
                    let made = read(texture, &buffer, &host);
                    println!(
                        "coherence_over: {name}: {form}, {what}: {made:?}; the texture's buffer \
                         {own} bytes, the destination {into}"
                    );
                    assert_eq!(
                        made,
                        one_submission(own + into, written),
                        "{name}: {form}, {what}: the submission flushes the texture's buffer and \
                         the destination; the completion settles the destination, and the \
                         texture's buffer where it is an image of its own"
                    );
                    assert!(
                        bytes_of(&out) == rows(&host, (row, pitch, HEIGHT)),
                        "{name}: {form}, {what}: the host's texels"
                    );
                }
            }
        });
    }

    /// The words `ab_address` reads and writes, one a thread.
    const THREADS: u32 = 64;

    /// The bytes of [`THREADS`] words.
    const WORDS: u64 = THREADS as u64 * 4;

    /// The state of `ab_pointers`' `ab_address`: thread `i` stores at `out[i]` (buffer 1) the
    /// word `i` past the address the constant buffer at buffer 0 holds.
    fn ab_address(name: &str, device: &Device, library: &Library) -> ComputePipelineState {
        let function = library.function("ab_address").expect("the kernel");
        let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(function),
        ));
        (device.new_compute_pipeline_state(&descriptor, crate::tests::no_constexpr_samplers))
            .unwrap_or_else(|error| panic!("{name}: ab_address: {error}"))
    }

    /// One dispatch of `ab_address` over [`THREADS`] threads, in a command buffer of its own,
    /// with `held` at buffer 0 and `out` at buffer 1, the encoder's declarations made by
    /// `declare`: the calls it made.
    fn by_address(
        name: &str,
        device: &Device,
        (queue, state): (&CommandQueue, &ComputePipelineState),
        [held, out]: [&Buffer; 2],
        declare: impl FnOnce(&mut ComputeCommandEncoder),
    ) -> [u64; 5] {
        let counters = device.counters();
        let before = counters.metrics();
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut encoder =
            (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
        encoder.set_compute_pipeline_state(state).expect("a state");
        encoder.set_buffer(Some(held), 0, 0).expect("held");
        encoder.set_buffer(Some(out), 0, 1).expect("out");
        declare(&mut encoder);
        (encoder.dispatch_threadgroups([2, 1, 1], [THREADS / 2, 1, 1]))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        encoder.end_encoding();
        run(name, &command_buffer);
        calls(&before, &counters.metrics())
    }

    /// Under the view: a Shared buffer that `ab_address` reaches only through an address the host
    /// wrote into the constant buffer at buffer 0, declared with `use_resource`, a read and then a
    /// read and a write. The dispatch's submission flushes it with the two bound buffers, each
    /// over its atoms; its completion flushes and invalidates the bound buffer the kernel writes,
    /// and the declared buffer too where its declaration writes. The kernel reads the words the
    /// host wrote there.
    #[test]
    fn every_device_opened_under_the_view_flushes_a_buffer_declared_with_use_resource() {
        let Some(library) = library("ab_pointers") else {
            return;
        };
        each_device_under_the_view("coherence_use_resource", |name, _, device, _| {
            let atom = atom(name, &device);
            let state = ab_address(name, &device, &library);
            let queue = device.new_command_queue(4).expect("a queue");
            let bound = (device.take_block(MemoryClass::Shared, 1)).expect("a block");
            let held = (device.new_buffer(bound.id, 0, 8, Some(0))).expect("a buffer");
            let out = (device.new_buffer(bound.id, 256, WORDS, Some(0))).expect("a buffer");
            let block = (device.take_block(MemoryClass::Shared, 1)).expect("a block");
            let far = (device.new_buffer(block.id, 512, WORDS, None)).expect("a buffer");
            write(&held, &far.gpu_address().to_le_bytes());
            let both = atoms(bound.offset, 8, atom) + atoms(bound.offset + 256, WORDS, atom);
            let into = atoms(bound.offset + 256, WORDS, atom);
            let declared = atoms(block.offset + 512, WORDS, atom);
            let mut seeds = 1..;
            for (what, usage, writes) in [
                ("a read", ResourceUsage::READ, false),
                (
                    "a read and a write",
                    ResourceUsage::READ | ResourceUsage::WRITE,
                    true,
                ),
            ] {
                let dispatch = |host: &[u8]| {
                    write(&far, host);
                    let declare = |encoder: &mut ComputeCommandEncoder| {
                        (encoder.use_resource(Resource::Buffer(&far), usage))
                            .unwrap_or_else(|error| panic!("{name}: {error}"));
                    };
                    by_address(name, &device, (&queue, &state), [&held, &out], declare)
                };
                // The first dispatch takes the first uses; the second is counted.
                dispatch(&pattern(WORDS, seeds.next().expect("a seed")));
                let host = pattern(WORDS, seeds.next().expect("a seed"));
                let made = dispatch(&host);
                println!(
                    "coherence_use_resource: {name}: {what}: {made:?}; the declared buffer \
                     {declared} bytes, the bound ones {both}"
                );
                let written = into + if writes { declared } else { 0 };
                assert_eq!(
                    made,
                    one_submission(both + declared, written),
                    "{name}: {what}: the submission flushes the declared buffer with the bound \
                     ones; the completion settles the written ones"
                );
                assert_eq!(bytes_of(&out), host, "{name}: {what}: the words read");
            }
        });
    }

    /// Under the view: a buffer in a Shared heap's block that `ab_address` reaches only through
    /// an address the host wrote into the constant buffer at buffer 0, the block declared with
    /// `use_heap`. The dispatch's submission flushes the whole block with the two bound buffers;
    /// its completion flushes and invalidates the whole block and the bound buffer the kernel
    /// writes. The kernel reads the words the host wrote in the heap.
    #[test]
    fn every_device_opened_under_the_view_flushes_and_settles_a_shared_heap_named_by_use_heap() {
        let Some(library) = library("ab_pointers") else {
            return;
        };
        each_device_under_the_view("coherence_use_heap", |name, _, device, _| {
            let atom = atom(name, &device);
            let state = ab_address(name, &device, &library);
            let queue = device.new_command_queue(4).expect("a queue");
            let bound = (device.take_block(MemoryClass::Shared, 1)).expect("a block");
            let held = (device.new_buffer(bound.id, 0, 8, Some(0))).expect("a buffer");
            let out = (device.new_buffer(bound.id, 256, WORDS, Some(0))).expect("a buffer");
            let heap = (device.take_block(MemoryClass::Shared, 1)).expect("a heap's block");
            let inside = (device.new_buffer(heap.id, 4096, WORDS, None)).expect("a buffer");
            write(&held, &inside.gpu_address().to_le_bytes());
            let both = atoms(bound.offset, 8, atom) + atoms(bound.offset + 256, WORDS, atom);
            let into = atoms(bound.offset + 256, WORDS, atom);
            let whole = atoms(heap.offset, heap.len, atom);
            let dispatch = |host: &[u8]| {
                write(&inside, host);
                let declare = |encoder: &mut ComputeCommandEncoder| {
                    (encoder.use_heap(heap.id)).unwrap_or_else(|error| panic!("{name}: {error}"));
                };
                by_address(name, &device, (&queue, &state), [&held, &out], declare)
            };
            // The first dispatch takes the first uses; the second is counted.
            dispatch(&pattern(WORDS, 1));
            let host = pattern(WORDS, 2);
            let made = dispatch(&host);
            println!(
                "coherence_use_heap: {name}: {made:?}; the heap's block {whole} bytes, the bound \
                 buffers {both}"
            );
            assert_eq!(
                made,
                one_submission(both + whole, into + whole),
                "{name}: the submission flushes the heap's block with the bound buffers; the \
                 completion settles the block and the buffer the kernel writes"
            );
            assert_eq!(bytes_of(&out), host, "{name}: the words read");
        });
    }

    /// Under the view: a blit's Shared source, which the host writes before the commit. A copy
    /// from it into a Private buffer is flushed over the source's atoms before its submission,
    /// and settles nothing, the command buffer writing no Shared memory; a copy of that Private
    /// buffer into a Shared one is flushed and settled over the destination's atoms, which then
    /// hold the source's bytes.
    #[test]
    fn every_device_opened_under_the_view_flushes_a_blits_shared_source_before_its_submission() {
        const LEN: u64 = 1000;
        each_device_under_the_view("coherence_blit_source", |name, _, device, _| {
            let atom = atom(name, &device);
            let counters = device.counters();
            let queue = device.new_command_queue(4).expect("a queue");
            let shared = (device.take_block(MemoryClass::Shared, 1)).expect("a block");
            let source = (device.new_buffer(shared.id, 256, LEN, None)).expect("a buffer");
            let destination = (device.new_buffer(shared.id, 2048, LEN, None)).expect("a buffer");
            let private = (device.take_block(MemoryClass::Private, 1)).expect("a block");
            let between = (device.new_buffer(private.id, 0, LEN, None)).expect("a buffer");
            let copy = |from: &Buffer, to: &Buffer| {
                let before = counters.metrics();
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
                (encoder.copy_from_buffer_to_buffer(from, 0, to, 0, LEN))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                encoder.end_encoding();
                run(name, &command_buffer);
                calls(&before, &counters.metrics())
            };
            let round = |host: &[u8]| {
                write(&source, host);
                (copy(&source, &between), copy(&between, &destination))
            };
            // The first copies take the first uses; the second are counted.
            round(&pattern(LEN, 1));
            let host = pattern(LEN, 2);
            let (out, back) = round(&host);
            let from = atoms(shared.offset + 256, LEN, atom);
            let into = atoms(shared.offset + 2048, LEN, atom);
            println!(
                "coherence_blit_source: {name}: into Private {out:?}, back into Shared {back:?}; \
                 the source {from} bytes, the destination {into}"
            );
            assert_eq!(
                out,
                one_submission(from, 0),
                "{name}: the copy from the Shared source flushes it, and settles nothing"
            );
            assert_eq!(
                back,
                one_submission(into, into),
                "{name}: the copy into the Shared destination flushes and settles it"
            );
            assert_eq!(bytes_of(&destination), host, "{name}: the source's bytes");
        });
    }

    /// Under the view: a visible function table's words, which the host writes
    /// (`setFunctions:withRange:`), read by a dispatch of `visible_table_caller`'s
    /// `visible_table_dispatch` (linking `visible_table_functions`' `table_add` and `table_mul`)
    /// that calls entry `tid % 2` with `(tid, 3)` and writes the result at buffer 0. The
    /// dispatch's submission flushes the table's words with the bound buffer, each over its
    /// atoms; its completion flushes and invalidates the bound buffer alone. Each thread's word is
    /// what the function its entry holds returns, the entries swapped between two dispatches.
    #[test]
    fn every_device_opened_under_the_view_flushes_a_function_tables_words_before_a_call() {
        const CALLS: u32 = 12;
        let (Some(caller), Some(functions)) = (
            library("visible_table_caller"),
            library("visible_table_functions"),
        ) else {
            return;
        };
        let function = |name: &str| functions.function(name).expect(name);
        let linked = LinkedFunctions::new()
            .with_function(function("table_add"))
            .with_function(function("table_mul"));
        let kernel = caller
            .function("visible_table_dispatch")
            .expect("the kernel");
        let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(kernel).with_linked(linked),
        ));
        each_device_under_the_view("coherence_function_table", |name, _, device, _| {
            let atom = atom(name, &device);
            let counters = device.counters();
            let state = (device
                .new_compute_pipeline_state(&descriptor, crate::tests::no_constexpr_samplers))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            let handle = |f: &str| state.function_handle(&LinkedFunction::new(function(f)));
            let (add, mul) = (handle("table_add"), handle("table_mul"));
            let table = (state.new_visible_function_table(&VisibleFunctionTableDescriptor::new(2)))
                .unwrap_or_else(|error| panic!("{name}: a table: {error}"));
            let queue = device.new_command_queue(4).expect("a queue");
            let block = (device.take_block(MemoryClass::Shared, 1)).expect("a block");
            let len = u64::from(CALLS) * 4;
            let out = (device.new_buffer(block.id, 0, len, Some(0))).expect("a buffer");
            let dispatch = |entries: &[Option<FunctionHandle>]| {
                (table.set_functions(entries, 0)).unwrap_or_else(|error| panic!("{name}: {error}"));
                let before = counters.metrics();
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = (command_buffer.compute_command_encoder(DispatchType::Serial))
                    .expect("an encoder");
                encoder.set_compute_pipeline_state(&state).expect("a state");
                encoder.set_buffer(Some(&out), 0, 0).expect("out");
                (encoder.set_visible_function_table(Some(&table), 1)).expect("the table");
                (encoder.dispatch_threads([CALLS, 1, 1], [CALLS, 1, 1]))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                encoder.end_encoding();
                run(name, &command_buffer);
                calls(&before, &counters.metrics())
            };
            // The first dispatch takes the first uses; the second, its entries swapped by the
            // host, is counted.
            dispatch(&[add, mul]);
            let made = dispatch(&[mul, add]);
            // The table's words: its header and two entries, at the start of a block of its own,
            // which starts at a multiple of 64 KiB of its allocation.
            let words = atoms(0, function_table_entry_offset(2), atom);
            let into = atoms(block.offset, len, atom);
            println!(
                "coherence_function_table: {name}: {made:?}; the table's words {words} bytes, the \
                 bound buffer {into}"
            );
            assert_eq!(
                made,
                one_submission(words + into, into),
                "{name}: the submission flushes the table's words with the bound buffer; the \
                 completion settles the bound buffer alone"
            );
            let expected: Vec<u8> = (0..CALLS)
                .map(|tid| if tid % 2 == 0 { tid * 3 } else { tid + 3 })
                .flat_map(u32::to_le_bytes)
                .collect();
            assert_eq!(
                bytes_of(&out),
                expected,
                "{name}: the swapped entries' calls"
            );
        });
    }

    /// How the host sees a shared event's value: its `signaledValue` reaching it, or a wait for
    /// it returning.
    type Sees = fn(&SharedEvent, u64);

    /// Under the view: a shared event's value a GPU signal raised, seen while the command buffer
    /// that signalled it is in flight, waiting for a value only the host sets after: the command
    /// buffer fills a Shared buffer, signals the value, then waits. Before `signaledValue` shows
    /// the value, and before a wait for it returns, the buffer is flushed and invalidated over
    /// its atoms, once, after the submission's flush of it, and the host reads the fill's bytes.
    /// Once the host sets the value waited for, the command buffer's completion invalidates the
    /// buffer once more.
    #[test]
    fn every_device_opened_under_the_view_settles_the_writes_in_flight_before_a_signal_is_seen() {
        const OFFSET: u64 = 256;
        const LEN: u64 = 64;
        each_device_under_the_view("coherence_event", |name, _, device, _| {
            let atom = atom(name, &device);
            let counters = device.counters();
            let queue = device.new_command_queue(4).expect("a queue");
            let block = (device.take_block(MemoryClass::Shared, 1)).expect("a block");
            let buffer = (device.new_buffer(block.id, OFFSET, LEN, Some(0))).expect("a buffer");
            // The first fill, so the counted ones meet no first use.
            fill(name, &queue, &buffer, 0..LEN, 1);
            let event = shared_event(&device);
            let ways: [(&str, Sees); 2] = [
                ("signaledValue", |event, value| {
                    within("the signal", || event.signaled_value() >= value);
                }),
                ("a wait", |event, value| {
                    let met = (event.wait_until_signaled_value(value, PATIENCE)).expect("a wait");
                    assert!(met, "the wait met its value");
                }),
            ];
            let widened = atoms(block.offset + OFFSET, LEN, atom);
            for ((way, sees), (signal, value)) in ways.into_iter().zip([(1, 0x31), (3, 0x32)]) {
                let before = counters.metrics();
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
                (encoder.fill_buffer(&buffer, 0..LEN, value))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                encoder.end_encoding();
                (command_buffer.encode_signal_event(event.as_event(), signal)).expect("a signal");
                (command_buffer.encode_wait_for_event(event.as_event(), signal + 1))
                    .expect("a wait");
                command_buffer.commit().expect("a commit");
                sees(&event, signal);
                let seen = counters.metrics();
                let status = command_buffer.status();
                let bytes = bytes_of(&buffer);
                // The command buffer goes on, so a failed assertion leaves no device in flight.
                event.set_signaled_value(signal + 1);
                finished(name, &command_buffer);
                let after = counters.metrics();
                assert_eq!(command_buffer.error(), None, "{name}: {way}");
                let at_sight = calls(&before, &seen);
                let at_completion = calls(&seen, &after);
                println!(
                    "coherence_event: {name}: {way}: to the value seen {at_sight:?}, then to the \
                     completion {at_completion:?}; the buffer {widened} bytes"
                );
                assert_ne!(
                    status,
                    CommandBufferStatus::Completed,
                    "{name}: {way}: in flight when the value was seen"
                );
                assert_eq!(
                    at_sight,
                    [2, 2 * widened, 1, widened, 0],
                    "{name}: {way}: the submission's flush, then the settle before the value is \
                     seen, each over the buffer's atoms"
                );
                assert!(
                    bytes.iter().all(|&byte| byte == value),
                    "{name}: {way}: the fill's bytes when the value is seen"
                );
                assert_eq!(
                    [at_completion[2], at_completion[3], at_completion[4]],
                    [1, widened, 0],
                    "{name}: {way}: the completion's invalidate"
                );
            }
        });
    }

    /// Under the view, the route of a glyph or tile upload: a blit of a sub-range of a 4 KiB
    /// Shared source in a pool block, at an offset of 64 there, its rows 100 bytes apart (no
    /// multiple of 64), into an R8 texture over a Shared buffer that is also a render attachment,
    /// whose buffer the host then reads. The host writes the source right before the commit. The
    /// submission's flush counts the source by its bytes widened to the atom, with the texture's
    /// buffer; the completion flushes and invalidates the texture's buffer, which then holds the
    /// source's rows in the region and the zeros of its make elsewhere. From a source of the same
    /// layout in host memory's import, whose type is host-coherent, the submission flushes the
    /// texture's buffer alone, and the completion settles it as before. A device that imports no
    /// host memory, or whose host-pointer query offers no type coherent under the view, refuses
    /// the import, with its refusal, which is printed.
    #[test]
    fn every_device_opened_under_the_view_flushes_an_uploads_source_and_settles_its_texture() {
        const SOURCE: u64 = 4096;
        const AT: u64 = 64;
        const HEIGHT: u32 = 16;
        // The region: its rows in the source, its place and size in the texture.
        const LAYOUT: BufferLayout = BufferLayout {
            offset: 5,
            bytes_per_row: 100,
            bytes_per_image: 0,
        };
        const ORIGIN: [u32; 2] = [3, 2];
        const SIZE: [u32; 2] = [90, 12];
        each_device_under_the_view("coherence_upload", |name, _, device, _| {
            let atom = atom(name, &device);
            let counters = device.counters();
            let format = PixelFormat::R8Unorm;
            let align = (device.minimum_linear_texture_alignment(format))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            // Rows of the device's own pitch, a row of texels each.
            let pitch = align.max(256);
            let width = u32::try_from(pitch).expect("a width");
            let len = pitch * u64::from(HEIGHT);
            let mut d = TextureDescriptor::new(TextureType::Type2D, format, width, HEIGHT);
            d.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
            let _grant = (device.grant_indices(IndexClass::Texture, 4096, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let block = (device.take_block(MemoryClass::Shared, len)).expect("a block");
            let buffer = (device.new_buffer(block.id, 0, len, Some(0))).expect("a buffer");
            let texture = (buffer.new_texture(&d, Some(4096), 0, pitch))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let form = match texture.copied_over() {
                Some(_) => "an image of its own",
                None => "in place",
            };
            let queue = device.new_command_queue(4).expect("a queue");
            let upload = |source: &Buffer, host: &[u8]| {
                write(source, host);
                let before = counters.metrics();
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
                let size = [SIZE[0], SIZE[1], 1];
                let to = TexturePlace {
                    origin: [ORIGIN[0], ORIGIN[1], 0],
                    ..TexturePlace::default()
                };
                (encoder.copy_from_buffer(source, LAYOUT, size, &texture, to))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                encoder.end_encoding();
                run(name, &command_buffer);
                calls(&before, &counters.metrics())
            };
            // The texture's buffer once the region took `host`'s rows: the zeros of its make
            // elsewhere.
            let expected = |host: &[u8]| {
                let mut bytes = vec![0; len as usize];
                for row in 0..u64::from(SIZE[1]) {
                    let from = (LAYOUT.offset + row * LAYOUT.bytes_per_row) as usize;
                    let to = ((u64::from(ORIGIN[1]) + row) * pitch + u64::from(ORIGIN[0])) as usize;
                    let span = SIZE[0] as usize;
                    bytes[to..to + span].copy_from_slice(&host[from..from + span]);
                }
                bytes
            };
            let into = atoms(block.offset, len, atom);
            let shared = (device.take_block(MemoryClass::Shared, 1)).expect("a pool block");
            let source = (device.new_buffer(shared.id, AT, SOURCE, None)).expect("a buffer");
            let from = atoms(shared.offset + AT, SOURCE, atom);
            // The first upload takes the first uses; the second is counted.
            upload(&source, &pattern(SOURCE, 1));
            let host = pattern(SOURCE, 2);
            let made = upload(&source, &host);
            println!(
                "coherence_upload: {name}: into an R8 attachment over a Shared buffer ({form}), \
                 from a Shared source: {made:?}; before the submission {} bytes: the source \
                 {from}, the texture's buffer {into}",
                made[1] - made[3]
            );
            assert_eq!(
                made,
                one_submission(from + into, into),
                "{name}: the submission flushes the source and the texture's buffer; the \
                 completion settles the texture's buffer"
            );
            assert_eq!(
                made[1] - made[3],
                from + into,
                "{name}: the flush before the submission counts the source's bytes"
            );
            assert!(
                bytes_of(&buffer) == expected(&host),
                "{name}: the source's rows in the region, the zeros elsewhere"
            );

            let Some((_, alignment)) = device.core().imports.host.as_ref() else {
                println!("coherence_upload: {name}: imports no host memory");
                return;
            };
            let imported = (AT + SOURCE).next_multiple_of(*alignment);
            let (pointer, mapping) = host_memory(imported, *alignment);
            let (release, released) = releasing();
            // SAFETY: the memory is the test's, allocated until it frees it after the import's
            // release, or at once where the import is refused.
            let (import, _) = match unsafe { device.import_host_memory(pointer, imported, release) }
            {
                Ok(import) => import,
                Err(error @ Error::Limit(Limit::ImportType { .. })) => {
                    free(mapping);
                    println!("coherence_upload: {name}: an import under the view: {error}");
                    return;
                }
                Err(error) => panic!("{name}: an import: {error}"),
            };
            let source = (device.new_buffer(import.id, AT, SOURCE, None)).expect("a buffer");
            upload(&source, &pattern(SOURCE, 3));
            let host = pattern(SOURCE, 4);
            let made = upload(&source, &host);
            println!(
                "coherence_upload: {name}: from an imported source {made:?}; the texture's \
                 buffer {into} bytes"
            );
            assert_eq!(
                made,
                one_submission(into, into),
                "{name}: an imported source takes no call; the texture's buffer is flushed and \
                 settled"
            );
            assert!(
                bytes_of(&buffer) == expected(&host),
                "{name}: the imported source's rows in the region, the zeros elsewhere"
            );
            drop((source, import));
            wait_until(&device, "the release", || released.load(Ordering::SeqCst));
            free(mapping);
        });
    }

    /// Under the view, a range a submission only writes, flushed before that submission: a new
    /// Shared buffer, zeroed by the host at its make, which the GPU alone writes in the first
    /// submission that reaches it, as a blit's destination (a copy from a Private buffer a fill
    /// wrote) and as a render pass's attachment (a texture over it, cleared). The submission's
    /// flush counts the buffer by its bytes widened to the atom, so the host's zeros reach the
    /// device before the GPU writes there, and nothing of the host's is left to land over the
    /// GPU's bytes at the completion's flush; the completion flushes and invalidates it. The host
    /// reads the GPU's bytes, not the zeros.
    #[test]
    fn every_device_opened_under_the_view_flushes_a_range_a_submission_only_writes_before_it() {
        const AT: u64 = 192;
        const LEN: u64 = 1000;
        const HEIGHT: u32 = 4;
        each_device_under_the_view("coherence_written", |name, _, device, _| {
            let atom = atom(name, &device);
            let counters = device.counters();
            settled(&device);
            let queue = device.new_command_queue(4).expect("a queue");
            let private = (device.take_block(MemoryClass::Private, 1)).expect("a block");
            let filled = (device.new_buffer(private.id, 0, LEN, None)).expect("a buffer");
            fill(name, &queue, &filled, 0..LEN, 0x6b);

            let shared = (device.take_block(MemoryClass::Shared, 1)).expect("a block");
            let written = (device.new_buffer(shared.id, AT, LEN, Some(0))).expect("a buffer");
            let before = counters.metrics();
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
            (encoder.copy_from_buffer_to_buffer(&filled, 0, &written, 0, LEN))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            encoder.end_encoding();
            run(name, &command_buffer);
            let copied = calls(&before, &counters.metrics());
            let into = atoms(shared.offset + AT, LEN, atom);

            let format = PixelFormat::RGBA8Unorm;
            let align = (device.minimum_linear_texture_alignment(format))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let row = align.max(256);
            let len = row * u64::from(HEIGHT);
            let width = u32::try_from(row / 4).expect("a width");
            let mut d = TextureDescriptor::new(TextureType::Type2D, format, width, HEIGHT);
            d.usage = TextureUsage::RENDER_TARGET;
            let _grant = (device.grant_indices(IndexClass::Texture, 4096, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let block = (device.take_block(MemoryClass::Shared, len)).expect("a block");
            let attached = (device.new_buffer(block.id, 0, len, Some(0))).expect("a buffer");
            let texture = (attached.new_texture(&d, Some(4096), 0, row))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let form = match texture.copied_over() {
                Some(_) => "an image of its own",
                None => "in place",
            };
            let mut pass = RenderPassDescriptor::default();
            let color = &mut pass.color_attachments[0];
            color.texture = Some(texture.clone());
            color.load_action = LoadAction::Clear;
            color.clear_color = ClearColor::new(1.0, 0.0, 1.0, 1.0);
            let before = counters.metrics();
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let encoder = (command_buffer.render_command_encoder(&pass))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            encoder.end_encoding();
            run(name, &command_buffer);
            let cleared = calls(&before, &counters.metrics());
            let over = atoms(block.offset, len, atom);
            println!(
                "coherence_written: {name}: a copy into a new Shared buffer {copied:?}, before \
                 the submission {} bytes, the buffer {into}; a clear of an attachment over one \
                 ({form}) {cleared:?}, before the submission {} bytes, the buffer {over}",
                copied[1] - copied[3],
                cleared[1] - cleared[3]
            );
            for (what, made, bytes) in [
                ("a blit's destination", copied, into),
                ("an attachment's buffer", cleared, over),
            ] {
                assert_eq!(
                    made,
                    one_submission(bytes, bytes),
                    "{name}: {what}: flushed before the submission and settled at the completion"
                );
                assert_eq!(
                    made[1] - made[3],
                    bytes,
                    "{name}: {what}: the flush before the submission counts the bytes it only \
                     writes"
                );
            }
            assert!(
                bytes_of(&written).iter().all(|&byte| byte == 0x6b),
                "{name}: the copy's bytes, not the zeros"
            );
            assert!(
                (bytes_of(&attached).chunks(4)).all(|texel| texel == [255, 0, 255, 255]),
                "{name}: the clear's texels, not the zeros"
            );
        });
    }
}
