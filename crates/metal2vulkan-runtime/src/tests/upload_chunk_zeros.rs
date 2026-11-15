//! A command buffer's upload arena clears the bytes of a region that the region's data does not
//! cover, and never a chunk of its own (`upload.rs`).
//!
//! # The contract
//!
//! One thing reads a byte of a chunk that nothing wrote: a region whose data is shorter than the
//! region. Its bytes past the data are zero — a table padded to a fixed length reads zero in the
//! slots a command left out, and a region the GPU writes reads zero where it wrote nothing. No
//! reader reaches a byte of a chunk outside a region: a descriptor is written with the region's
//! own offset and length, a device address is the region's, and the host's reads and writes go
//! through [`crate::upload::Upload`], which holds the region's length. The bytes an alignment
//! leaves between two regions are in no region at all.
//!
//! So the arena makes a chunk without a clear and zeroes each region's own tail as it hands it
//! out. That is the same answer for every reader above and strictly fewer bytes written: the
//! regions' tails sum to at most the chunk, where a clear of the chunk wrote every byte of it
//! whether a region took it or not — 4 KiB for the first region of a command buffer, whatever that
//! region held ([`FIRST_CHUNK_BYTES`]).
//!
//! # Why a chunk really does hold the last owner's bytes
//!
//! A chunk is a buffer of the runtime's own, so a parked one of the same making is handed back
//! (`recycle.rs`) with its last owner's bytes in it. `a_regions_tail_reads_zero_over_a_dirty_chunk`
//! fills a chunk with 0xab, lets it park, takes a short region out of the chunk that comes back,
//! and reads **both** sides of the region's end: the tail reads zero, and the chunk past the
//! region still reads 0xab. The second half is what makes the first half evidence. Without it a
//! freshly allocated chunk that happens to read zero would pass the test with the zeroing removed.
//!
//! # The counts
//!
//! `new_buffer_bytes_cleared` is the one counter either form moves, so the fix shows up in it as
//! a smaller number rather than as work moved somewhere the gate does not read
//! (`WorkCounts::new_buffer_cleared`). `a_command_buffers_fixed_clear_is_not_a_chunk` reads it
//! over a sweep of two dispatch counts and two command-buffer counts, which separates the two
//! terms: what a command buffer clears whatever it holds, and what each dispatch adds. The chunk
//! was the first of those, the same 4 KiB at every dispatch count, and it is what this fix
//! removes.
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`).

use super::blit_holds::{until, Scratch, WAIT};
use crate::upload::{Uploads, FIRST_CHUNK_BYTES};
use crate::{
    Device, DeviceKind, DeviceOptions, PipelineOptions, ResourceOptions, ShaderCacheOptions, Size,
};
use std::path::Path;

/// The byte a chunk's last owner left in it.
const POISON: u8 = 0xab;

/// The byte a region's data holds.
const DATA: u8 = 0x11;

/// The device the cfg selects, its shader cache in `cache`.
fn device(cache: &Path) -> Device {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// A region's bytes past its data read zero, and the chunk it came out of still holds the bytes
/// its last owner left past the region: the zeros are the region's own and not a chunk's clear.
#[test]
fn a_regions_tail_reads_zero_over_a_dirty_chunk() {
    let scratch = Scratch::new("upload-dirty-chunk");
    let device = device(&scratch.0);
    let parked = || device.work_counts().object_churn.buffers_parked;
    let before_parked = parked();
    // A region over the whole of a first chunk, every byte of it the poison, and then the chunk
    // parked as the arena goes.
    {
        let mut arena = Uploads::new();
        let poison = vec![POISON; FIRST_CHUNK_BYTES as usize];
        let region = arena
            .upload(&device, &poison, FIRST_CHUNK_BYTES)
            .expect("a region over the whole first chunk");
        assert_eq!(
            region.len(),
            FIRST_CHUNK_BYTES,
            "the region is the whole chunk"
        );
        let whole = region.read::<{ FIRST_CHUNK_BYTES as usize }>();
        assert_eq!(whole.as_slice(), poison, "the poison was written");
    }
    until("the chunk was parked", || parked() > before_parked);
    // The chunk comes back for the next arena of the same making, holding the poison.
    let mut arena = Uploads::new();
    let before = device.work_counts().new_buffer_bytes_cleared;
    let data = [DATA; 8];
    let region = arena
        .upload(&device, &data, 64)
        .expect("a short region of a recycled chunk");
    let cleared = device.work_counts().new_buffer_bytes_cleared - before;
    let bytes = region.read::<64>();
    assert_eq!(&bytes[..8], &data, "the region holds its data");
    assert_eq!(
        &bytes[8..],
        &[0; 56],
        "the region's bytes past its data must read zero"
    );
    // The other side of the region's end, which is what makes the line above evidence: the chunk
    // was handed over dirty, so a region that did not zero its own tail would have read the
    // poison there.
    let (chunk, offset) = region.buffer();
    let contents = chunk.contents().expect("a chunk is a `Shared` buffer");
    let length = chunk.length();
    assert_eq!(length, FIRST_CHUNK_BYTES, "the recycled chunk's length");
    let past = (offset + region.len()) as usize;
    // SAFETY: the chunk's bytes, its length from `contents`, which no command uses: this arena
    // handed out one region and submitted nothing.
    let chunk = unsafe { std::slice::from_raw_parts(contents.as_ptr(), length as usize) };
    assert!(
        chunk[past..].iter().all(|&byte| byte == POISON),
        "the chunk past the region must still hold its last owner's bytes: it was cleared whole, \
         so the zeros the region read are not its own"
    );
    // And the count is the region's tail, not the chunk: 56 bytes, where a chunk's clear was
    // 4,096 whatever the region held.
    assert_eq!(
        cleared, 56,
        "the region cleared {cleared} bytes, not the 56 of its own that must read zero"
    );
}

/// `scale_and_offset` of the examples fixture, its two device buffers bound and its constant set
/// inline before each dispatch (a region of its own in the arena, the scale `index`), dispatched
/// `dispatches` times in each of `command_buffers` command buffers: the
/// `new_buffer_bytes_cleared` those command buffers moved, and the arena bytes the last of them
/// took.
///
/// One command buffer of one dispatch runs first, outside the window, so that what the device sets
/// up once — its pools, its sets, its kernel — is not read as a command buffer's own cost. Every
/// figure here is a difference, and the sweep's two axes are what separate the per-command-buffer
/// term from the per-dispatch one.
fn cleared(command_buffers: usize, dispatches: usize) -> (u64, u64) {
    let scratch = Scratch::new("upload-sweep");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let ll = metal2vulkan_devtools::fixtures_dir().join("examples/runtime_examples.ll");
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let library = metal2vulkan::library::Library::from_air(&bitcode).unwrap();
    let function = device
        .new_library_with_compile_library(library)
        .new_function_with_name("scale_and_offset")
        .unwrap();
    let state = device
        .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
        .unwrap();
    let buffer = device.new_buffer(1024, ResourceOptions::shared()).unwrap();
    let run = |dispatches: usize| {
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&state).unwrap();
        for index in [0, 1] {
            encoder.set_buffer(Some(&buffer), 0, index).unwrap();
        }
        for index in 0..dispatches {
            encoder.set_bytes(&(index as f32).to_le_bytes(), 2).unwrap();
            encoder
                .dispatch_threads(Size::new(1, 1, 1), Size::new(1, 1, 1))
                .unwrap();
        }
        encoder.end_encoding().unwrap();
        let arena = command_buffer.upload_bytes();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None, "the dispatches ran");
        arena
    };
    run(1);
    let before = device.work_counts().new_buffer_bytes_cleared;
    let mut arena = 0;
    for _ in 0..command_buffers {
        arena = run(dispatches);
    }
    (
        device.work_counts().new_buffer_bytes_cleared - before,
        arena,
    )
}

/// What a command buffer clears whatever it holds is no longer a chunk.
///
/// The sweep is two dispatch counts and two command-buffer counts, because an absolute says
/// nothing here: a command buffer's cost is a fixed term plus a term for each dispatch, and the
/// defect was in the fixed one. The first region any command of a command buffer took opened the
/// arena's first chunk, which was a new buffer cleared whole, so the fixed term was
/// [`FIRST_CHUNK_BYTES`] at one dispatch and at sixteen alike, and a command buffer that uploaded
/// sixteen bytes paid 4,096 of them.
///
/// Now a region clears its own unwritten bytes and a chunk clears none, so the fixed term is what
/// the command buffer's own regions leave unwritten — nothing, for this kernel — and each
/// dispatch's inline constant clears its region's tail past its four bytes and no more.
#[test]
fn a_command_buffers_fixed_clear_is_not_a_chunk() {
    let (one_one, arena_one) = cleared(1, 1);
    let (one_sixteen, arena_sixteen) = cleared(1, 16);
    let (four_one, _) = cleared(4, 1);
    let (four_sixteen, _) = cleared(4, 16);
    // One region a dispatch, so the arena's own bytes grow with the dispatches: without that the
    // counts below would be about a command buffer that uploads nothing.
    assert!(
        arena_sixteen > arena_one,
        "sixteen dispatches took {arena_sixteen} arena bytes and one took {arena_one}, so the \
         dispatches uploaded nothing and this sweep measures nothing"
    );
    // Each command buffer pays the same, so the per-command-buffer figure is the quarter of four.
    assert_eq!(
        (four_one, four_sixteen),
        (one_one * 4, one_sixteen * 4),
        "four command buffers of one dispatch cleared {four_one} bytes and of sixteen \
         {four_sixteen}, where one cleared {one_one} and {one_sixteen}: a command buffer's clear \
         is its own"
    );
    // The two terms, separated: what a dispatch adds, and what the command buffer pays whatever
    // its dispatch count.
    let per_dispatch = (one_sixteen - one_one) / 15;
    assert_eq!(
        one_sixteen,
        one_one + per_dispatch * 15,
        "the clear is not linear in the dispatches: {one_one} at one and {one_sixteen} at sixteen"
    );
    let fixed = one_one - per_dispatch;
    println!(
        "cleared bytes: 1x1 {one_one}, 1x16 {one_sixteen}, 4x1 {four_one}, 4x16 {four_sixteen}; \
         a command buffer's fixed term {fixed}, a dispatch's {per_dispatch}; arena bytes 1 \
         dispatch {arena_one}, 16 {arena_sixteen}"
    );
    assert!(
        fixed < FIRST_CHUNK_BYTES,
        "a command buffer cleared {fixed} bytes whatever its dispatches ({one_one} at one \
         dispatch, {one_sixteen} at sixteen, so {per_dispatch} a dispatch), which is a chunk's \
         {FIRST_CHUNK_BYTES}: its chunks must be made without a clear and its regions clear their \
         own unwritten bytes alone"
    );
    // And what a dispatch adds is its constant region's own tail, which the arena's own figure
    // bounds from above: a region's bytes cleared can never exceed the bytes it took.
    let per_dispatch_arena = (arena_sixteen - arena_one) / 15;
    assert!(
        per_dispatch > 0 && per_dispatch <= per_dispatch_arena,
        "a dispatch cleared {per_dispatch} bytes of the {per_dispatch_arena} its region took"
    );
}
