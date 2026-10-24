//! What a new buffer or texture holds before anything writes it, and what memory that was written
//! and released reads as once it comes back.
//!
//! Ported from the runtime lane `runtime_lane/fresh_contents.rs`. Metal's answer, which that lane's
//! header states from our own probe and which record mode now takes row by row:
//!
//! - **Zero, every read:** buffers of every storage mode; `Shared` and `Managed` textures of every
//!   type and format tried; the `Private` textures Metal keeps uncompressed (shader-write usage,
//!   GPU optimization off, a block-compressed format); a texture over a new buffer.
//! - **A fixed pattern, every read:** the `Private` textures Metal compresses — `RGBA8Unorm` reads
//!   `ff00ffff` in every texel, depth reads all ones — in each level whose shorter side is 16 or
//!   more.
//!
//! # Where these rows get their teeth
//!
//! A fresh resource's contents are **not a Metal promise**. A row that simply made a buffer and
//! read zero out of it would be reading memory nobody had touched, and would pass on an
//! implementation that handed out whatever the allocator left, as long as the allocator happened to
//! be clean. So every row here states the write it is measured against:
//!
//! - a resource of the row's own shape is made, **filled with [`OLD`] through a blit**, and read
//!   back, so the row records that the old bytes really were there (the `written` site);
//! - it is then [`released`](crate::work::Step::Release) and
//!   [`made again`](crate::work::Step::Make), in that order, so the second allocation happens after
//!   the first was freed and may be handed the same memory;
//! - every read goes through a copy into a shared buffer the case **prefilled with [`BEFORE`]**, so
//!   a row that reads zero has also proved its copy ran: zero is otherwise indistinguishable from a
//!   read-back that never happened.
//!
//! The `fresh-pattern-*` rows are the one group with no release in them, because the lane's own
//! per-format probe made each texture first: what they pin is Metal's compressed-texture pattern
//! per format, and their prefilled destination is what makes the read honest.
//!
//! # What the rows are held to, and the one branch of the lane that is not exercised
//!
//! **Not one byte of this family is loosened.** Every row's bytes are Metal's own on MoltenVK, on
//! lavapipe and on an RTX 5080 Laptop, so the only field that may differ is the set of degrade
//! notices our side announced. Four rows name a notice of their own: lavapipe keeps a decoded twin
//! of a `BC1_RGBA` image (`decoded-block-twin`) and cannot hold Metal's denormal depths
//! (`depth-denormals`). Each is named on the row that announced it, not on the family.
//!
//! Every row names one more, and it belongs to the venue rather than to any row:
//! [`SHARED_MEMORY_NOTICE`], the notice a device without unified memory announces for the `Shared`
//! buffer each row reads its answer back through.
//!
//! The lane also carries a branch for a device that **cannot** write Metal's pattern and so reads
//! zero, announcing `fresh-pattern-zeroed`. That branch is right for the lane, which must run
//! anywhere, but **neither venue of this round takes it**: MoltenVK and lavapipe both write Metal's
//! pattern exactly, in all 58 formats and every patterned shape. So these rows hold our side to
//! Metal's pattern and name no allowance for the zeroing — an allowance that fires on no venue is
//! a claim no run has checked, which the suite counts and which this family deliberately does not
//! add. A device that does zero will read as `Divergent` here, and the allowance belongs in the
//! commit that has that device's run behind it.
//!
//! # Why one process is the right place for this, and more than the lane had
//!
//! The lane's header describes a probe across *three fresh processes* and one after 768 MiB of
//! written and released memory. That probe's job was to show the answer does **not** vary with how
//! much a process has allocated, and it has done it. Nothing in these cases depends on being first:
//! each row makes, fills, releases and remakes inside its own program.
//!
//! `conformance replay` runs every case in **one** process, where nextest gives each lane test its
//! own, so these rows run in a **dirtier** process than the lane's — hundreds of cases' allocations
//! have come and gone before them. That cuts one way only: a warm process can hand back more
//! soiled memory, never less, so it cannot manufacture a green row, only expose a dirty one. Do not
//! "fix" this family by giving its cases a process each; the shared process is the stronger test.
//!
//! # The record shape
//!
//! Every answer here is one texel repeated by construction — a level holds Metal's pattern, or the
//! old bytes, or zero — so each site records the **runs of equal texels** rather than the bytes
//! themselves (`texel_runs`). That is a lossless encoding and not a digest: the bytes are
//! rebuilt from it in the guard that keeps it honest
//! (`the_runs_rebuild_the_bytes_they_came_from`), and it sizes a row by how many times its
//! answer changes instead of by how many bytes it holds, which is what lets the 16 MiB buffer row
//! record its whole answer in one line.
//!
//! Per texel, not per byte, and that was measured rather than assumed: a texel of Metal's pattern
//! is not one byte repeated — `RGBA8Unorm` reads `ff00ffff` — so runs of equal bytes made 8,193
//! runs of one 16 KiB level and a 3.5 MB recording for a row whose answer is one sentence.
//!
//! # What is not ported, and why
//!
//! - **The 3D and cube rows.** [`crate::work::TextureKind`] describes no 3D and no cube texture and
//!   [`crate::work::TextureSpec`] carries no depth, so porting them means growing the shared
//!   vocabulary for two rows. They stay in the lane.
//! - **The multisampled rows** (the 4x colour targets, the NaN pattern copied from a scratch, and
//!   the depth target loaded, cleared and read by a shader). The Metal side of a copy-and-fill
//!   program builds no clearing render pass and no per-sample read-back, so those rows have no
//!   Metal leg to record; they stay in the lane.
//! - **The two-plane `RGB10A8_2P_XR10` row**, whose planes are read by a fixture's kernels rather
//!   than by a copy.
//! - **The host's own read of a `Shared` buffer's `contents`.** Every read here is a copy into a
//!   shared buffer instead, so the three storage modes compare in one shape and a `Managed`
//!   resource needs no host synchronization.
//!
//! Nothing here can fault a GPU: every command is a buffer fill or a copy inside its own resource,
//! there is no shader and no loop, and every wait is bounded.

use crate::allowance::Allowance;
use crate::case::{call, Abort, Case, Input, MetalSource, Side};
use crate::transcript::{Node, Observed, Sites, Value};
use crate::work::{Done, Plane, Resource, Step, Storage, TextureKind, TextureSpec, Work};
use metal2vulkan_runtime::Degradation;

/// The byte a resource is filled with before it is released, so the memory the row's second
/// allocation may be handed held something nothing else would produce. The lane's own byte.
pub const OLD: u8 = 0xa5;

/// The byte every read-back buffer holds before its copy. Neither zero nor [`OLD`], so a site that
/// reads zero has proved the copy ran and a site that reads [`OLD`] has proved the old bytes came
/// back.
pub const BEFORE: u8 = 0x5c;

/// `MTLTextureUsageShaderRead`.
const READ: u32 = 1;

/// `MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite`.
const READ_AND_WRITE: u32 = 3;

/// `MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget`.
const READ_AND_TARGET: u32 = 5;

/// The runs of equal texels `bytes` holds, `texel` bytes each: the first texel of the run, how
/// many texels it covers, and the texel's own bytes.
///
/// This is a **lossless encoding** of the bytes and not a digest of them
/// (`the_runs_rebuild_the_bytes_they_came_from`), and it is per texel rather than per byte
/// for a measured reason: every answer in this family is one texel repeated, but a texel of
/// Metal's pattern is not one byte repeated — `RGBA8Unorm` reads `ff00ffff` — so runs of equal
/// *bytes* turned a 16 KiB level into 8,193 runs and a 3.5 MB recording, while runs of equal
/// texels make it one line. A tail shorter than a texel is a run of its own, so nothing is lost
/// where a length is not a whole number of texels.
fn texel_runs(bytes: &[u8], texel: usize) -> Value {
    let width = texel.max(1);
    let mut runs: Vec<Value> = Vec::new();
    let mut held: Option<(usize, &[u8], u64)> = None;
    for (at, chunk) in bytes.chunks(width).enumerate() {
        match held {
            Some((_, last, ref mut count)) if last == chunk => *count += 1,
            _ => {
                if let Some((first, last, count)) = held {
                    runs.push(run_value(first, count, last));
                }
                held = Some((at, chunk, 1));
            }
        }
    }
    if let Some((first, last, count)) = held {
        runs.push(run_value(first, count, last));
    }
    Value::List(runs)
}

/// One run of [`texel_runs`].
fn run_value(at: usize, texels: u64, texel: &[u8]) -> Value {
    Value::fields([
        ("at", Value::Uint(at as u64)),
        ("texels", Value::Uint(texels)),
        ("texel", Value::Bytes(texel.to_vec())),
    ])
}

/// A site whose bytes are recorded as their runs of equal `texel`-byte texels (the module docs).
fn runs_site(name: impl Into<String>, bytes: &[u8], texel: usize) -> Node {
    Node::leaf(name, Observed::value(texel_runs(bytes, texel)))
}

/// The command buffers' statuses, in commit order.
fn statuses_site(done: &[String]) -> Node {
    Node::leaf("statuses", Observed::value(Value::Lines(done.to_vec())))
}

// ---------------------------------------------------------------------------------------------
// Buffers
// ---------------------------------------------------------------------------------------------

/// The buffer lengths each storage mode is swept at: the lane's own, from one 16-byte buffer to
/// 16 MiB.
const LENGTHS: [u64; 5] = [16, 4096, 65536, 1 << 20, 16 << 20];

/// The program one buffer row runs at one length: fill a buffer of this shape with [`OLD`], read it
/// back, release it, make another of the same declaration, and read every byte of that one.
fn buffer_program(length: u64, storage: Storage) -> Work {
    let mut work = Work::new();
    let subject = work.unwritten(length, storage);
    let written_out = work.buffer(vec![BEFORE; length as usize]);
    let fresh_out = work.buffer(vec![BEFORE; length as usize]);
    work.step(Step::FillBuffer {
        buffer: subject,
        offset: 0,
        length,
        value: OLD,
    });
    work.step(Step::Commit);
    work.step(Step::CopyBufferToBuffer {
        from: subject,
        from_offset: 0,
        to: written_out,
        to_offset: 0,
        length,
    });
    work.step(Step::Commit);
    work.step(Step::ReadBuffer {
        buffer: written_out,
        offset: 0,
        length: length as usize,
    });
    // The teeth: the filled buffer is freed, and the next one is allocated afterwards.
    work.step(Step::Release { resource: subject });
    work.step(Step::Make { resource: subject });
    work.step(Step::CopyBufferToBuffer {
        from: subject,
        from_offset: 0,
        to: fresh_out,
        to_offset: 0,
        length,
    });
    work.step(Step::Commit);
    work.step(Step::ReadBuffer {
        buffer: fresh_out,
        offset: 0,
        length: length as usize,
    });
    work
}

/// A buffer row: every buffer length the family sweeps, in one storage mode.
pub struct FreshBuffers {
    case: &'static str,
    storage: Storage,
}

/// The notice every row of this family announces on a device whose GPU does not share the CPU's
/// memory: each row reads its answer back through a `Shared` buffer, and without unified memory
/// such a buffer lies in host-cached system memory rather than in the device's own
/// ([`Degradation::SharedInSystemMemory`]).
///
/// **It is the venue's memory model, not ours, and no recording could have declared it.** Every
/// recording of this suite was taken on a chip whose GPU shares the CPU's memory, where the notice
/// cannot fire at all, so Metal recorded nothing to compare it against. The runtime announces it
/// once per device, and the row is right either way: a discrete card really does read that buffer
/// across the bus, which is what the notice says.
///
/// Measured on the **RTX 5080 Laptop (615.71.09)**, where it fired for all 87 rows of the family
/// and each read `divergent` on this one field and nothing else
/// (`replay-nvidia.log`, "we announced the degrade notice
/// metal2vulkan-degrade-shared-in-system-memory, which the case declares nowhere"). MoltenVK and
/// lavapipe announce it for no row.
///
/// It loosens the notices site alone, so every byte of the family still compares exactly.
fn shared_memory_notice() -> Allowance {
    crate::replay::notice_allowance(Degradation::SharedInSystemMemory.code())
}

impl Case for FreshBuffers {
    fn name(&self) -> &'static str {
        self.case
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        vec![shared_memory_notice()]
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut words = vec![OLD, BEFORE];
        words.extend(self.storage.word().as_bytes());
        for length in LENGTHS {
            words.extend(length.to_le_bytes());
        }
        let mut inputs = vec![Input::bytes("lengths-and-bytes", &words)];
        for length in LENGTHS {
            inputs.extend(program_inputs(
                &format!("bytes-{length}"),
                &buffer_program(length, self.storage),
            )?);
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        for length in LENGTHS {
            let work = buffer_program(length, self.storage);
            let at = format!("bytes-{length}");
            let done = call(sites, &format!("work/{at}"), side.work(&work), |_| {
                Value::Absent
            })?;
            sites.push(
                Node::branch(at, Vec::new())
                    .with(runs_site("written", read(&done, 0), 1))
                    .with(runs_site("fresh", read(&done, 1), 1))
                    .with(statuses_site(&done.statuses)),
            );
        }
        Ok(())
    }
}

/// The inputs one program feeds its run: one input a resource it is made with, and one for the
/// **steps it encodes, in order**.
///
/// The steps are in the fingerprint and not only the resources' bytes, and that was found by
/// breaking something rather than reasoned out: reordering one row's release on purpose — so the
/// release came while a command buffer was open and the interpreter refused it — left the
/// fingerprint unchanged, so the run compared against a recording taken from a *different* program
/// and read as our side diverging. A case must prove its inputs, and a program's order is one of
/// them.
///
/// # Keyed by the Metal call, never by `Debug`
///
/// Every byte here is **the name of the Metal call the program makes, and the values it hands that
/// call**. It was `format!("{:?}")` of the resources and of the steps, which worked and was a
/// trap: that text moves whenever anyone adds a field to a shared type in [`crate::work`], so two
/// new step fields and one texture field restaled all 87 of this family's recordings without
/// changing a single Metal call any of these cases makes.
///
/// Three properties keep that from happening again, and each is a rule for whoever widens
/// [`crate::work`] next:
///
/// - **A call's name leads its bytes**, so a new [`Resource`] or [`Step`] variant renumbers
///   nothing: there is no positional discriminant in here for one to shift.
/// - **Every variant this family uses is destructured field by field, with no `..`**, so a field
///   added to one of them is a *compile error* here rather than a silent restale, and its arm
///   decides whether the field is part of what the call is asked to do.
/// - **A value Metal's own call leaves at a default is written only where it is not that
///   default**, each behind its own tag. A field whose default every case of this family takes
///   therefore writes nothing at all, and no recording moves.
///
/// A resource's index *is* in here, deliberately: it is the program's own name for a resource it
/// declared itself ([`Work::resources`]), not a side's object identity, and two of this family's
/// steps differ by nothing else. [`buffer_program`] copies its subject into one prefilled buffer
/// before the release and into another after it; drop the indices and those two copies — and the
/// two reads that follow them — key identically, so a program that read the wrong one would match
/// a recording taken from the right one
/// (`a_program_that_reads_the_wrong_buffer_keys_differently`).
fn program_inputs(what: &str, work: &Work) -> Result<Vec<Input>, String> {
    let mut inputs = Vec::new();
    for (at, resource) in work.resources.iter().enumerate() {
        // The `buffer:`/`resource:` names are the ones the recordings already hold and the index
        // is what tells one resource of a program from the next. The call's own name is in the
        // bytes, which is where the fingerprint is.
        let named = match resource {
            Resource::Buffer { .. } => "buffer",
            _ => "resource",
        };
        inputs.push(Input::bytes(
            format!("{what}/{named}:{at}"),
            &resource_words(resource)?,
        ));
    }
    let mut steps = Vec::new();
    for step in &work.steps {
        steps.extend(step_words(step)?);
    }
    inputs.push(Input::bytes(format!("{what}/steps"), &steps));
    Ok(inputs)
}

/// One parameter of a call, little-endian.
fn word(out: &mut Vec<u8>, value: u64) {
    out.extend(value.to_le_bytes());
}

/// A parameter named by its own tag, so a value can never be read as another field's and a field
/// that writes nothing shifts nothing after it.
fn tagged(out: &mut Vec<u8>, tag: &[u8], value: u64) {
    out.extend(tag);
    word(out, value);
}

/// The call that makes `resource`, and the values it hands that call.
///
/// A resource none of this family's programs declares is an error rather than a key: inventing one
/// would be inventing a fingerprint for bytes no recording was taken against. Every program here
/// is built in this module, so no case reaches it; a case that wants such a resource adds its arm
/// (`a_resource_this_family_does_not_declare_is_refused`).
fn resource_words(resource: &Resource) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    // Every pattern below names every field it has, with no `..`, for the reason written at the
    // destructuring in `descriptor_words`: a field added to one of these variants must be a
    // compile error here rather than 87 recordings going quietly stale.
    match resource {
        Resource::Buffer { bytes } => {
            out.extend(b"newBufferWithBytes");
            word(&mut out, bytes.len() as u64);
            out.extend(bytes);
        }
        Resource::UnwrittenBuffer { length, storage } => {
            out.extend(b"newBufferWithLength");
            word(&mut out, *length);
            // `MTLResourceOptions`' own default storage mode is shared, so a shared buffer states
            // nothing and each mode that is not the default says which it is.
            if *storage != Storage::Shared {
                out.extend(b"storageMode");
                out.extend(storage.word().as_bytes());
            }
        }
        Resource::Texture { spec } => {
            out.extend(b"newTextureWithDescriptor");
            out.extend(descriptor_words(spec));
        }
        Resource::TextureOverBuffer {
            buffer,
            spec,
            offset,
            bytes_per_row,
        } => {
            out.extend(b"newTextureWithDescriptorOffsetBytesPerRow");
            word(&mut out, *buffer as u64);
            word(&mut out, *offset);
            word(&mut out, *bytes_per_row);
            out.extend(descriptor_words(spec));
        }
        // The `Debug` here is an error message and not a key: it says which resource has no call
        // name, and nothing is fingerprinted by it.
        other => return Err(format!("this family keys no resource {other:?}")),
    }
    Ok(out)
}

/// The `MTLTextureDescriptor` a texture is made from: the fields every row of this family states
/// for itself, then each field Metal gives a default, only where the row does not take it.
///
/// Destructured whole on purpose. A field added to [`TextureSpec`] stops this compiling, which is
/// the point: its arm then decides whether the field is part of what the descriptor asks for, and
/// writing it only where it is not the default leaves every recording where it is.
fn descriptor_words(spec: &TextureSpec) -> Vec<u8> {
    // **Never add `..` here, and never add it to a variant's pattern in `resource_words` or
    // `step_words`.** Naming every field is the whole protection: a field added to `TextureSpec`
    // is then a *compile error in this file*, where before it was 87 recordings quietly going
    // stale in a stream no longer running to re-record them. Whoever adds the field decides, here,
    // whether it is part of what the descriptor is asked for, and writes it only where it is not
    // the default — so the answer "it changes no call these cases make" costs no recording.
    //
    // The honest limit: a new *variant* of `Resource` or `Step` still compiles and writes nothing,
    // because those matches end in a catch-all that refuses rather than keys. That is right — this
    // family cannot construct a variant it does not name — but it means the compiler guards fields
    // and not variants.
    let TextureSpec {
        format,
        kind,
        width,
        height,
        depth,
        levels,
        slices,
        samples,
        storage,
        render_target,
        gpu_optimized,
        usage,
    } = spec;
    let mut out = Vec::new();
    // The three a row of this family always states.
    tagged(&mut out, b"pixelFormat", u64::from(*format));
    tagged(&mut out, b"width", *width);
    tagged(&mut out, b"height", *height);
    // And every other field only where it is not the descriptor's own default.
    if *kind != TextureKind::Type2D {
        out.extend(b"textureType");
        out.extend(texture_type(*kind).as_bytes());
    }
    if *depth != 1 {
        tagged(&mut out, b"depth", *depth);
    }
    if *levels != 1 {
        tagged(&mut out, b"mipmapLevelCount", *levels);
    }
    if *slices != 1 {
        tagged(&mut out, b"arrayLength", *slices);
    }
    if *samples != 1 {
        tagged(&mut out, b"sampleCount", *samples);
    }
    if *storage != Storage::Shared {
        out.extend(b"storageMode");
        out.extend(storage.word().as_bytes());
    }
    if *render_target {
        out.extend(b"renderTarget");
    }
    if !*gpu_optimized {
        out.extend(b"noGPUOptimizedContents");
    }
    if let Some(usage) = usage {
        tagged(&mut out, b"usage", u64::from(*usage));
    }
    out
}

/// The `MTLTextureType` a kind names.
///
/// Exhaustive on purpose: a kind added to [`TextureKind`] names itself here rather than falling
/// into a catch-all whose bytes nobody chose.
fn texture_type(kind: TextureKind) -> &'static str {
    match kind {
        TextureKind::Type2D => "MTLTextureType2D",
        TextureKind::Type2DArray => "MTLTextureType2DArray",
        TextureKind::Cube => "MTLTextureTypeCube",
        TextureKind::Type3D => "MTLTextureType3D",
        TextureKind::Type2DMultisample => "MTLTextureType2DMultisample",
        TextureKind::Type2DMultisampleArray => "MTLTextureType2DMultisampleArray",
        TextureKind::TextureBuffer => "MTLTextureTypeTextureBuffer",
    }
}

/// The call one step runs, and the values it takes.
///
/// A step none of this family's programs encodes is an error rather than a key, for the reason
/// [`resource_words`] gives (`a_step_this_family_does_not_encode_is_refused`).
fn step_words(step: &Step) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    // No `..` in any pattern here either; see `descriptor_words`.
    match step {
        Step::FillBuffer {
            buffer,
            offset,
            length,
            value,
        } => {
            out.extend(b"fillBuffer");
            word(&mut out, *buffer as u64);
            word(&mut out, *offset);
            word(&mut out, *length);
            out.push(*value);
        }
        Step::CopyBufferToBuffer {
            from,
            from_offset,
            to,
            to_offset,
            length,
        } => {
            out.extend(b"copyFromBufferToBuffer");
            for value in [*from as u64, *from_offset, *to as u64, *to_offset, *length] {
                word(&mut out, value);
            }
        }
        Step::CopyBufferToTexture {
            buffer,
            offset,
            bytes_per_row,
            bytes_per_image,
            size,
            texture,
            slice,
            level,
            origin,
            plane,
        } => {
            out.extend(b"copyFromBufferToTexture");
            for value in [
                *buffer as u64,
                *offset,
                *bytes_per_row,
                *bytes_per_image,
                *texture as u64,
                *slice,
                *level,
            ] {
                word(&mut out, value);
            }
            out.extend(extent_words(origin, size));
            out.extend(option_words(*plane));
        }
        Step::CopyTextureToBuffer {
            texture,
            slice,
            level,
            origin,
            size,
            buffer,
            offset,
            bytes_per_row,
            bytes_per_image,
            plane,
        } => {
            out.extend(b"copyFromTextureToBuffer");
            for value in [
                *texture as u64,
                *slice,
                *level,
                *buffer as u64,
                *offset,
                *bytes_per_row,
                *bytes_per_image,
            ] {
                word(&mut out, value);
            }
            out.extend(extent_words(origin, size));
            out.extend(option_words(*plane));
        }
        Step::ReadBuffer {
            buffer,
            offset,
            length,
        } => {
            out.extend(b"contents");
            for value in [*buffer as u64, *offset, *length as u64] {
                word(&mut out, value);
            }
        }
        // These two are the program's own steps and not Metal selectors: the side drops its
        // reference to the resource, and makes it again from its own declaration. They are named
        // for what they do, which is what a step's key is for.
        Step::Release { resource } => {
            out.extend(b"release");
            word(&mut out, *resource as u64);
        }
        Step::Make { resource } => {
            out.extend(b"makeAgain");
            word(&mut out, *resource as u64);
        }
        Step::Commit => out.extend(b"commit"),
        // As in [`resource_words`], the `Debug` is the message and never the key.
        other => return Err(format!("this family keys no step {other:?}")),
    }
    Ok(out)
}

/// A copy's origin and size, in texels.
fn extent_words(origin: &[u64; 3], size: &[u64; 3]) -> Vec<u8> {
    let mut out = Vec::new();
    for axis in origin.iter().chain(size) {
        out.extend(axis.to_le_bytes());
    }
    out
}

/// The `MTLBlitOption` a copy takes, where it takes one.
///
/// [`Plane::Whole`] is `MTLBlitOptionNone`, the default every copy of this family makes, so it
/// writes nothing; exhaustive, so a plane added to [`Plane`] names itself here.
fn option_words(plane: Plane) -> Vec<u8> {
    let named = match plane {
        Plane::Whole => return Vec::new(),
        Plane::Depth => "MTLBlitOptionDepthFromDepthStencil",
        Plane::Stencil => "MTLBlitOptionStencilFromDepthStencil",
        Plane::RowLinearPvrtc => "MTLBlitOptionRowLinearPVRTC",
    };
    let mut out = b"options".to_vec();
    out.extend(named.as_bytes());
    out
}

/// Read `index` of a program's reads, or nothing where the program answered fewer. A short answer
/// shows up as runs that are not the recording's rather than as a panic inside a case.
fn read(done: &Done, index: usize) -> &[u8] {
    done.reads.get(index).map(Vec::as_slice).unwrap_or_default()
}

/// The three buffer rows, one a storage mode.
const BUFFERS: [(&str, Storage); 3] = [
    ("fresh-buffer-shared", Storage::Shared),
    ("fresh-buffer-managed", Storage::Managed),
    ("fresh-buffer-private", Storage::Private),
];

// ---------------------------------------------------------------------------------------------
// Textures of their own memory
// ---------------------------------------------------------------------------------------------

/// A texture shape the lane sweeps: the descriptor's facts, the bytes its blocks take, what the
/// [`OLD`] write puts in each block, and what Metal reads in a `Private` one of this shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Shape {
    format: u32,
    /// Level 0's extent in texels.
    size: [u64; 2],
    levels: u64,
    slices: u64,
    /// Bytes one block takes, and the block's side in texels (1 for an uncompressed format).
    block: (u64, u64),
    /// The raw `MTLTextureUsage` bits, stated because Metal refuses some pairings and because the
    /// usage decides the layout.
    usage: u32,
    /// Whether `allowGPUOptimizedContents` is **off**, which is how the lane picks out a `Private`
    /// texture Metal keeps uncompressed.
    plain: bool,
    /// The bytes the [`OLD`] write puts in each block: one block's worth.
    old: &'static [u8],
    /// Metal's pattern in a `Private` texture of this shape, and the number of levels from 0 it
    /// reads it in. `None` where Metal reads zero whatever the storage.
    pattern: Option<(&'static [u8], u64)>,
}

impl Shape {
    /// Level `level`'s extent in texels, and its bytes a row and a slice.
    fn level(&self, level: u64) -> ([u64; 2], u64, u64) {
        let [w, h] = self.size.map(|axis| (axis >> level).max(1));
        let (bytes, side) = self.block;
        let row = w.div_ceil(side) * bytes;
        (([w, h]), row, row * h.div_ceil(side))
    }

    /// Every `(level, slice)` of the texture, in the order the rows are read.
    fn subresources(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        (0..self.levels).flat_map(move |level| (0..self.slices).map(move |slice| (level, slice)))
    }

    /// The bytes a read of every level and slice answers.
    fn total(&self) -> u64 {
        self.subresources()
            .map(|(level, _)| self.level(level).2)
            .sum()
    }

    /// The spec a texture of this shape in `storage` is made from.
    fn spec(&self, storage: Storage) -> TextureSpec {
        let mut spec = TextureSpec::texture_2d(self.format, self.size[0], self.size[1])
            .used(self.usage)
            .stored(storage);
        spec.levels = self.levels;
        if self.slices > 1 {
            spec = spec.array(self.slices).used(self.usage).stored(storage);
            spec.levels = self.levels;
        }
        if self.plain {
            spec = spec.plain();
        }
        spec
    }

    /// What the lane expects a fresh texture of this shape in `storage` to read, level by level and
    /// slice by slice: Metal's pattern in a `Private` one's patterned levels, zero everywhere else.
    ///
    /// Nothing in the suite compares against this: it is what the port's cross-check holds the
    /// recorded Metal answer to, which is why it is the tests' alone.
    #[cfg(test)]
    fn expected_fresh(&self, storage: Storage) -> Vec<u8> {
        let mut all = Vec::new();
        for (level, _) in self.subresources() {
            let bytes = self.level(level).2 as usize;
            match self.pattern {
                Some((texel, levels))
                    if storage == Storage::Private && level < levels && !texel.is_empty() =>
                {
                    all.extend(texel.iter().copied().cycle().take(bytes));
                }
                _ => all.extend(std::iter::repeat_n(0u8, bytes)),
            }
        }
        all
    }

    /// The words the shape fingerprints as.
    fn words(&self) -> Vec<u8> {
        let mut out = self.format.to_le_bytes().to_vec();
        for axis in self.size {
            out.extend(axis.to_le_bytes());
        }
        out.extend(self.levels.to_le_bytes());
        out.extend(self.slices.to_le_bytes());
        out.extend(self.block.0.to_le_bytes());
        out.extend(self.block.1.to_le_bytes());
        out.extend(self.usage.to_le_bytes());
        out.push(u8::from(self.plain));
        out.extend(self.old);
        out
    }
}

/// A 2D shape of one level and one slice.
const fn shape(format: u32, size: [u64; 2], block: (u64, u64), usage: u32) -> Shape {
    Shape {
        format,
        size,
        levels: 1,
        slices: 1,
        block,
        usage,
        plain: false,
        old: &[OLD, OLD, OLD, OLD],
        pattern: None,
    }
}

/// `MTLPixelFormatRGBA8Unorm`.
const RGBA8: u32 = 70;

/// Metal's pattern of `RGBA8Unorm`: opaque magenta.
const MAGENTA: &[u8] = &[0xff, 0x00, 0xff, 0xff];

/// A 64-square `RGBA8Unorm` render target. Metal compresses a `Private` one, which then reads
/// magenta.
const RGBA8_2D: Shape = Shape {
    pattern: Some((MAGENTA, 1)),
    ..shape(RGBA8, [64, 64], (4, 1), READ_AND_TARGET)
};

/// A 64-square `RGBA8Unorm` render target of seven levels: Metal reads its pattern in the levels
/// whose shorter side is 16 or more, which are 64, 32 and 16.
const RGBA8_MIPMAPPED: Shape = Shape {
    levels: 7,
    pattern: Some((MAGENTA, 3)),
    ..shape(RGBA8, [64, 64], (4, 1), READ_AND_TARGET)
};

/// A 32-square `RGBA8Unorm` render target of three array slices.
const RGBA8_ARRAY: Shape = Shape {
    slices: 3,
    pattern: Some((MAGENTA, 1)),
    ..shape(RGBA8, [32, 32], (4, 1), READ_AND_TARGET)
};

/// A 64-square `R32Float` a kernel writes: Metal keeps a `Private` one uncompressed, so it reads
/// zero in every storage mode.
const R32FLOAT_WRITTEN: Shape = shape(55, [64, 64], (4, 1), READ_AND_WRITE);

/// A 64-square `RGBA16Float` render target, whose `Private` pattern holds halves that are NaNs.
const RGBA16FLOAT: Shape = Shape {
    old: &[OLD; 8],
    pattern: Some((&[0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff], 1)),
    ..shape(115, [64, 64], (8, 1), READ_AND_TARGET)
};

/// A 64-square `BC1_RGBA`, whose blocks are 4 texels square and 8 bytes each: Metal keeps a
/// `Private` block-compressed texture uncompressed, so it reads zero.
const BC1_RGBA: Shape = Shape {
    old: &[OLD; 8],
    ..shape(130, [64, 64], (8, 4), READ)
};

/// A 64-square `RGBA8Unorm` render target with `allowGPUOptimizedContents` **off**: the property
/// is what makes Metal keep a `Private` texture uncompressed, so this row reads zero where
/// [`RGBA8_2D`] reads magenta. It is the one pair in the family that isolates the property.
const RGBA8_PLAIN: Shape = Shape {
    plain: true,
    ..shape(RGBA8, [64, 64], (4, 1), READ_AND_TARGET)
};

/// A 64-square `Depth32Float` render target. Metal's pattern for it is all ones, a NaN depth.
///
/// The [`OLD`] write puts a depth of 0.5 in it rather than [`OLD`] bytes: a value every depth image
/// holds as it is, which is the lane's own reason.
const DEPTH32FLOAT: Shape = Shape {
    old: &[0x00, 0x00, 0x00, 0x3f],
    pattern: Some((&[0xff; 4], 1)),
    ..shape(252, [64, 64], (4, 1), READ_AND_TARGET)
};

/// A mipmapped `Private` chain of `size`, `levels` levels, reading Metal's pattern in its first
/// `wide` levels: the levels whose shorter side is 16 or more.
const fn chain(size: [u64; 2], levels: u64, wide: u64) -> Shape {
    Shape {
        levels,
        pattern: Some((MAGENTA, wide)),
        ..shape(RGBA8, size, (4, 1), READ)
    }
}

/// One texture row: its case name, its shape, its storage mode, and the degrade notices a device
/// that cannot hold this shape exactly announces for it.
///
/// Metal announces none, so the notices site is the only field of a row that may differ at all —
/// every byte compares exactly on both venues of this round — and each kind listed is what
/// justifies that one difference. A row that announced a kind it does not list fails until the kind
/// is added, which is the model working.
struct Row {
    case: &'static str,
    shape: Shape,
    storage: Storage,
    notices: &'static [Degradation],
}

/// A row of `shape` in `storage` that a device holds exactly, announcing nothing.
const fn row(case: &'static str, shape: Shape, storage: Storage) -> Row {
    Row {
        case,
        shape,
        storage,
        notices: &[],
    }
}

/// A row of `shape` in `storage` whose shape a device may not hold exactly, announcing `notices`.
const fn degraded(
    case: &'static str,
    shape: Shape,
    storage: Storage,
    notices: &'static [Degradation],
) -> Row {
    Row {
        case,
        shape,
        storage,
        notices,
    }
}

/// What a device without block-compressed images announces for a `BC1_RGBA` texture: it keeps a
/// decoded twin of the image and writes it after each write into the texture.
///
/// Measured, not guessed: lavapipe announces exactly this for all three of the `BC1_RGBA` rows,
/// and MoltenVK announces nothing for them, which is why the kind is named per row rather than for
/// the family.
const BLOCK_TWIN: &[Degradation] = &[Degradation::DecodedBlockTwin];

/// What a device whose depth format cannot hold Metal's denormal depths announces for a
/// `Depth32Float` target: lavapipe announces it for the depth row, MoltenVK announces nothing.
const DEPTH_DENORMALS: &[Degradation] = &[Degradation::DepthDenormals];

/// Every texture row of the family.
///
/// Six shapes swept over the three storage modes, then the four rows that are `Private` by nature:
/// the texture whose GPU optimization is off, the depth target, and the chains whose levels say
/// where Metal's "shorter side of 16 or more" rule puts the switch.
const TEXTURES: &[Row] = &[
    row(
        "fresh-texture-2d-rgba8unorm-shared",
        RGBA8_2D,
        Storage::Shared,
    ),
    row(
        "fresh-texture-2d-rgba8unorm-managed",
        RGBA8_2D,
        Storage::Managed,
    ),
    row(
        "fresh-texture-2d-rgba8unorm-private",
        RGBA8_2D,
        Storage::Private,
    ),
    row(
        "fresh-texture-mipmapped-rgba8unorm-shared",
        RGBA8_MIPMAPPED,
        Storage::Shared,
    ),
    row(
        "fresh-texture-mipmapped-rgba8unorm-managed",
        RGBA8_MIPMAPPED,
        Storage::Managed,
    ),
    row(
        "fresh-texture-mipmapped-rgba8unorm-private",
        RGBA8_MIPMAPPED,
        Storage::Private,
    ),
    row(
        "fresh-texture-array-rgba8unorm-shared",
        RGBA8_ARRAY,
        Storage::Shared,
    ),
    row(
        "fresh-texture-array-rgba8unorm-managed",
        RGBA8_ARRAY,
        Storage::Managed,
    ),
    row(
        "fresh-texture-array-rgba8unorm-private",
        RGBA8_ARRAY,
        Storage::Private,
    ),
    row(
        "fresh-texture-r32float-shared",
        R32FLOAT_WRITTEN,
        Storage::Shared,
    ),
    row(
        "fresh-texture-r32float-managed",
        R32FLOAT_WRITTEN,
        Storage::Managed,
    ),
    row(
        "fresh-texture-r32float-private",
        R32FLOAT_WRITTEN,
        Storage::Private,
    ),
    row(
        "fresh-texture-rgba16float-shared",
        RGBA16FLOAT,
        Storage::Shared,
    ),
    row(
        "fresh-texture-rgba16float-managed",
        RGBA16FLOAT,
        Storage::Managed,
    ),
    row(
        "fresh-texture-rgba16float-private",
        RGBA16FLOAT,
        Storage::Private,
    ),
    degraded(
        "fresh-texture-bc1-rgba-shared",
        BC1_RGBA,
        Storage::Shared,
        BLOCK_TWIN,
    ),
    degraded(
        "fresh-texture-bc1-rgba-managed",
        BC1_RGBA,
        Storage::Managed,
        BLOCK_TWIN,
    ),
    degraded(
        "fresh-texture-bc1-rgba-private",
        BC1_RGBA,
        Storage::Private,
        BLOCK_TWIN,
    ),
    row(
        "fresh-texture-plain-rgba8unorm-private",
        RGBA8_PLAIN,
        Storage::Private,
    ),
    degraded(
        "fresh-texture-depth32float-private",
        DEPTH32FLOAT,
        Storage::Private,
        DEPTH_DENORMALS,
    ),
    row(
        "fresh-chain-32x32-private",
        chain([32, 32], 6, 2),
        Storage::Private,
    ),
    row(
        "fresh-chain-100x40-private",
        chain([100, 40], 7, 2),
        Storage::Private,
    ),
    row(
        "fresh-chain-16x256-private",
        chain([16, 256], 9, 1),
        Storage::Private,
    ),
    row(
        "fresh-chain-17x17-private",
        chain([17, 17], 5, 1),
        Storage::Private,
    ),
];

/// The program one texture row runs: write [`OLD`] into every level and slice, read them back,
/// release the texture, make another of the same descriptor, and read every level and slice of
/// that one.
fn texture_program(shape: &Shape, storage: Storage) -> Work {
    let spec = shape.spec(storage);
    let total = shape.total();
    let (_, _, first_image) = shape.level(0);
    let mut work = Work::new();
    let subject = work.add(Resource::Texture { spec });
    // One source for every level: the biggest level's image of the row's own block bytes, which
    // every smaller level's copy reads the front of.
    let source = work.buffer(
        shape
            .old
            .iter()
            .copied()
            .cycle()
            .take(first_image as usize)
            .collect(),
    );
    let written_out = work.buffer(vec![BEFORE; total as usize]);
    let fresh_out = work.buffer(vec![BEFORE; total as usize]);
    for (level, slice) in shape.subresources() {
        let ([w, h], row_bytes, image) = shape.level(level);
        work.step(Step::CopyBufferToTexture {
            buffer: source,
            offset: 0,
            bytes_per_row: row_bytes,
            bytes_per_image: image,
            size: [w, h, 1],
            texture: subject,
            slice,
            level,
            origin: [0, 0, 0],
            plane: crate::work::Plane::Whole,
        });
    }
    work.step(Step::Commit);
    read_subresources(&mut work, shape, subject, written_out);
    work.step(Step::Commit);
    work.step(Step::ReadBuffer {
        buffer: written_out,
        offset: 0,
        length: total as usize,
    });
    // The teeth: the written texture is freed, and the next one is allocated afterwards.
    work.step(Step::Release { resource: subject });
    work.step(Step::Make { resource: subject });
    read_subresources(&mut work, shape, subject, fresh_out);
    work.step(Step::Commit);
    work.step(Step::ReadBuffer {
        buffer: fresh_out,
        offset: 0,
        length: total as usize,
    });
    work
}

/// Copy every level and slice of `texture` into `out`, each at its own offset, in the order
/// [`Shape::subresources`] gives them.
fn read_subresources(work: &mut Work, shape: &Shape, texture: usize, out: usize) {
    let mut offset = 0;
    for (level, slice) in shape.subresources() {
        let ([w, h], row_bytes, image) = shape.level(level);
        work.step(Step::CopyTextureToBuffer {
            texture,
            slice,
            level,
            origin: [0, 0, 0],
            size: [w, h, 1],
            buffer: out,
            offset,
            bytes_per_row: row_bytes,
            bytes_per_image: image,
            plane: crate::work::Plane::Whole,
        });
        offset += image;
    }
}

/// A texture row of the family.
pub struct FreshTexture {
    row: &'static Row,
}

impl Case for FreshTexture {
    fn name(&self) -> &'static str {
        self.row.case
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut words = self.row.shape.words();
        words.extend(self.row.storage.word().as_bytes());
        words.extend([OLD, BEFORE]);
        let mut inputs = vec![Input::bytes("shape-and-bytes", &words)];
        inputs.extend(program_inputs(
            "work",
            &texture_program(&self.row.shape, self.row.storage),
        )?);
        Ok(inputs)
    }

    fn allowances(&self) -> Vec<Allowance> {
        let mut allowances = vec![shared_memory_notice()];
        allowances.extend(
            self.row
                .notices
                .iter()
                .map(|kind| crate::replay::notice_allowance(kind.code())),
        );
        allowances
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let work = texture_program(&self.row.shape, self.row.storage);
        let done = call(sites, "work", side.work(&work), |_| Value::Absent)?;
        let texel = self.row.shape.block.0 as usize;
        sites.push(runs_site("written", read(&done, 0), texel));
        sites.push(runs_site("fresh", read(&done, 1), texel));
        sites.push(statuses_site(&done.statuses));
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// Metal's pattern, format by format
// ---------------------------------------------------------------------------------------------

/// The side of every [`PATTERNS`] texture, in texels: the lane's own, and wide enough that Metal
/// compresses it.
const PATTERN_SIDE: u64 = 64;

/// One format whose `Private` pattern the lane's probe read.
struct Pattern {
    case: &'static str,
    format: u32,
    /// Metal's pattern: the bytes of one texel, which every texel of a new `Private` texture of
    /// this format holds.
    texel: &'static [u8],
    /// The raw `MTLTextureUsage` bits the lane makes the texture with.
    usage: u32,
}

/// A pattern row.
const fn pattern(case: &'static str, format: u32, texel: &'static [u8], usage: u32) -> Pattern {
    Pattern {
        case,
        format,
        texel,
        usage,
    }
}

/// Metal's pattern of each format its probes read: a new `Private` 64-square texture of the format
/// reads these bytes in every texel.
///
/// The table is the lane's, transcribed with each format's raw `MTLPixelFormat` so a case can name
/// one with no public name as readily as a public one. Each row records Metal's own answer, so what
/// the lane kept as a literal is now measured.
const PATTERNS: &[Pattern] = &[
    pattern("fresh-pattern-1", 1, &[0xff], READ), // MTLPixelFormatA8Unorm
    pattern("fresh-pattern-10", 10, &[0xff], READ), // MTLPixelFormatR8Unorm
    pattern("fresh-pattern-11", 11, &[0xff], READ), // MTLPixelFormatR8Unorm_sRGB
    pattern("fresh-pattern-12", 12, &[0xff], READ), // MTLPixelFormatR8Snorm
    pattern("fresh-pattern-13", 13, &[0xff], READ), // MTLPixelFormatR8Uint
    pattern("fresh-pattern-14", 14, &[0xff], READ), // MTLPixelFormatR8Sint
    pattern("fresh-pattern-30", 30, &[0xff, 0x00], READ), // MTLPixelFormatRG8Unorm
    pattern("fresh-pattern-31", 31, &[0xff, 0x00], READ), // MTLPixelFormatRG8Unorm_sRGB
    pattern("fresh-pattern-32", 32, &[0xff, 0x00], READ), // MTLPixelFormatRG8Snorm
    pattern("fresh-pattern-33", 33, &[0xff, 0x00], READ), // MTLPixelFormatRG8Uint
    pattern("fresh-pattern-34", 34, &[0xff, 0x00], READ), // MTLPixelFormatRG8Sint
    pattern("fresh-pattern-20", 20, &[0xff, 0xff], READ), // MTLPixelFormatR16Unorm
    pattern("fresh-pattern-22", 22, &[0xff, 0xff], READ), // MTLPixelFormatR16Snorm
    pattern("fresh-pattern-23", 23, &[0xff, 0xff], READ), // MTLPixelFormatR16Uint
    pattern("fresh-pattern-24", 24, &[0xff, 0xff], READ), // MTLPixelFormatR16Sint
    pattern("fresh-pattern-25", 25, &[0xff, 0xff], READ), // MTLPixelFormatR16Float
    pattern("fresh-pattern-40", 40, &[0x1f, 0xf8], READ), // MTLPixelFormatB5G6R5Unorm
    pattern("fresh-pattern-41", 41, &[0xc1, 0xff], READ), // MTLPixelFormatA1BGR5Unorm
    pattern("fresh-pattern-42", 42, &[0x0f, 0xff], READ), // MTLPixelFormatABGR4Unorm
    pattern("fresh-pattern-43", 43, &[0x1f, 0xfc], READ), // MTLPixelFormatBGR5A1Unorm
    pattern("fresh-pattern-70", 70, MAGENTA, READ), // MTLPixelFormatRGBA8Unorm
    pattern("fresh-pattern-71", 71, MAGENTA, READ), // MTLPixelFormatRGBA8Unorm_sRGB
    pattern("fresh-pattern-72", 72, MAGENTA, READ), // MTLPixelFormatRGBA8Snorm
    pattern("fresh-pattern-73", 73, MAGENTA, READ), // MTLPixelFormatRGBA8Uint
    pattern("fresh-pattern-74", 74, MAGENTA, READ), // MTLPixelFormatRGBA8Sint
    pattern("fresh-pattern-80", 80, MAGENTA, READ), // MTLPixelFormatBGRA8Unorm
    pattern("fresh-pattern-81", 81, MAGENTA, READ), // MTLPixelFormatBGRA8Unorm_sRGB
    pattern("fresh-pattern-90", 90, &[0xff, 0x03, 0xf0, 0xff], READ), // MTLPixelFormatRGB10A2Unorm
    pattern("fresh-pattern-91", 91, &[0xff, 0x03, 0xf0, 0xff], READ), // MTLPixelFormatRGB10A2Uint
    pattern("fresh-pattern-94", 94, &[0xff, 0x03, 0xf0, 0xff], READ), // MTLPixelFormatBGR10A2Unorm
    pattern("fresh-pattern-92", 92, &[0xe0, 0x07, 0x00, 0xfc], READ), // MTLPixelFormatRG11B10Float
    pattern("fresh-pattern-93", 93, &[0xff, 0x01, 0xfc, 0xff], READ), // MTLPixelFormatRGB9E5Float
    pattern("fresh-pattern-60", 60, &[0xff, 0xff, 0x00, 0x00], READ), // MTLPixelFormatRG16Unorm
    pattern("fresh-pattern-62", 62, &[0xff, 0xff, 0x00, 0x00], READ), // MTLPixelFormatRG16Snorm
    pattern("fresh-pattern-63", 63, &[0xff, 0xff, 0x00, 0x00], READ), // MTLPixelFormatRG16Uint
    pattern("fresh-pattern-64", 64, &[0xff, 0xff, 0x00, 0x00], READ), // MTLPixelFormatRG16Sint
    pattern("fresh-pattern-65", 65, &[0xff, 0xff, 0x00, 0x00], READ), // MTLPixelFormatRG16Float
    pattern("fresh-pattern-55", 55, &[0xff; 4], READ), // MTLPixelFormatR32Float
    pattern("fresh-pattern-53", 53, &[0xff; 4], READ), // MTLPixelFormatR32Uint
    pattern("fresh-pattern-54", 54, &[0xff; 4], READ), // MTLPixelFormatR32Sint
    pattern("fresh-pattern-554", 554, &[0xff, 0x03, 0xf0, 0xff], READ), // MTLPixelFormatBGR10_XR
    pattern("fresh-pattern-555", 555, &[0xff, 0x03, 0xf0, 0xff], READ), // MTLPixelFormatBGR10_XR_sRGB
    pattern(
        "fresh-pattern-115",
        115,
        &[0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff],
        READ,
    ), // MTLPixelFormatRGBA16Float
    pattern(
        "fresh-pattern-110",
        110,
        &[0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff],
        READ,
    ), // MTLPixelFormatRGBA16Unorm
    pattern(
        "fresh-pattern-112",
        112,
        &[0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff],
        READ,
    ), // MTLPixelFormatRGBA16Snorm
    pattern(
        "fresh-pattern-113",
        113,
        &[0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff],
        READ,
    ), // MTLPixelFormatRGBA16Uint
    pattern(
        "fresh-pattern-114",
        114,
        &[0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff],
        READ,
    ), // MTLPixelFormatRGBA16Sint
    pattern(
        "fresh-pattern-105",
        105,
        &[0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0],
        READ,
    ), // MTLPixelFormatRG32Float
    pattern(
        "fresh-pattern-103",
        103,
        &[0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0],
        READ,
    ), // MTLPixelFormatRG32Uint
    pattern(
        "fresh-pattern-104",
        104,
        &[0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0],
        READ,
    ), // MTLPixelFormatRG32Sint
    pattern(
        "fresh-pattern-552",
        552,
        &[0xc0, 0xff, 0, 0, 0xc0, 0xff, 0x80, 0xdf],
        READ,
    ), // MTLPixelFormatBGRA10_XR
    pattern(
        "fresh-pattern-553",
        553,
        &[0xc0, 0xff, 0, 0, 0xc0, 0xff, 0x80, 0xdf],
        READ,
    ), // MTLPixelFormatBGRA10_XR_sRGB
    pattern(
        "fresh-pattern-125",
        125,
        &[
            0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        ],
        READ,
    ), // MTLPixelFormatRGBA32Float
    pattern(
        "fresh-pattern-123",
        123,
        &[
            0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        ],
        READ,
    ), // MTLPixelFormatRGBA32Uint
    pattern(
        "fresh-pattern-124",
        124,
        &[
            0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        ],
        READ,
    ), // MTLPixelFormatRGBA32Sint
    pattern("fresh-pattern-250", 250, &[0xff, 0xff], READ_AND_TARGET), // MTLPixelFormatDepth16Unorm
    pattern("fresh-pattern-252", 252, &[0xff; 4], READ_AND_TARGET),    // MTLPixelFormatDepth32Float
    pattern("fresh-pattern-253", 253, &[0xff], READ_AND_TARGET),       // MTLPixelFormatStencil8
];

impl Pattern {
    /// The shape of the texture the row makes: a 64-square `Private` texture of the format, one
    /// level, whose blocks are its texels.
    fn shape(&self) -> Shape {
        Shape {
            pattern: Some((self.texel, 1)),
            old: self.texel,
            ..shape(
                self.format,
                [PATTERN_SIDE, PATTERN_SIDE],
                (self.texel.len() as u64, 1),
                self.usage,
            )
        }
    }

    /// The bytes a read of the whole texture answers. The cross-check against the lane's table is
    /// what reads it, so it is the tests' alone.
    #[cfg(test)]
    fn total(&self) -> u64 {
        self.shape().total()
    }

    /// The program the row runs: make the texture and read every texel of it, writing nothing.
    ///
    /// Nothing is released here, because the lane's own per-format probe made each texture first.
    /// What makes the read honest is the destination: it holds [`BEFORE`] before the copy, so the
    /// recorded answer is the texture's own bytes and not a copy that never ran.
    fn program(&self) -> Work {
        let shape = self.shape();
        let total = shape.total();
        let mut work = Work::new();
        let subject = work.add(Resource::Texture {
            spec: shape.spec(Storage::Private),
        });
        let out = work.buffer(vec![BEFORE; total as usize]);
        read_subresources(&mut work, &shape, subject, out);
        work.step(Step::Commit);
        work.step(Step::ReadBuffer {
            buffer: out,
            offset: 0,
            length: total as usize,
        });
        work
    }
}

/// A pattern row of the family: one format.
pub struct FreshPattern {
    row: &'static Pattern,
}

impl Case for FreshPattern {
    fn name(&self) -> &'static str {
        self.row.case
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        vec![shared_memory_notice()]
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut words = self.row.shape().words();
        words.push(BEFORE);
        let mut inputs = vec![Input::bytes("shape-and-bytes", &words)];
        inputs.extend(program_inputs("work", &self.row.program())?);
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let work = self.row.program();
        let done = call(sites, "work", side.work(&work), |_| Value::Absent)?;
        sites.push(runs_site("fresh", read(&done, 0), self.row.texel.len()));
        sites.push(statuses_site(&done.statuses));
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// Textures over a new buffer
// ---------------------------------------------------------------------------------------------

/// The extent and the row pitch of a texture over a buffer: the lane's own.
const OVER_BUFFER: (u64, u64) = (64, 256);

/// A row of a texture over a new buffer, in one storage mode.
pub struct FreshOverBuffer {
    case: &'static str,
    storage: Storage,
}

/// The two rows of a texture over a new buffer: the storage modes the lane sweeps there.
const OVER_BUFFERS: [(&str, Storage); 2] = [
    ("fresh-texture-over-buffer-shared", Storage::Shared),
    ("fresh-texture-over-buffer-private", Storage::Private),
];

impl FreshOverBuffer {
    /// The shape of the texture over the buffer: a 64-square `RGBA8Unorm` a shader reads.
    fn shape(&self) -> Shape {
        shape(RGBA8, [OVER_BUFFER.0, OVER_BUFFER.0], (4, 1), READ)
    }

    /// The program the row runs: fill a buffer with [`OLD`], read the texture over it, release the
    /// texture and then the buffer, make both again in that order, and read the new texture.
    fn program(&self) -> Work {
        let shape = self.shape();
        let (side, pitch) = OVER_BUFFER;
        let length = pitch * side;
        let total = shape.total();
        let mut work = Work::new();
        let buffer = work.unwritten(length, self.storage);
        let texture = work.add(Resource::TextureOverBuffer {
            buffer,
            spec: shape.spec(self.storage),
            offset: 0,
            bytes_per_row: pitch,
        });
        let written_out = work.buffer(vec![BEFORE; total as usize]);
        let fresh_out = work.buffer(vec![BEFORE; total as usize]);
        work.step(Step::FillBuffer {
            buffer,
            offset: 0,
            length,
            value: OLD,
        });
        work.step(Step::Commit);
        read_subresources(&mut work, &shape, texture, written_out);
        work.step(Step::Commit);
        work.step(Step::ReadBuffer {
            buffer: written_out,
            offset: 0,
            length: total as usize,
        });
        // The texture goes first: it holds a reference to the buffer, so releasing the buffer
        // under it would free nothing.
        work.step(Step::Release { resource: texture });
        work.step(Step::Release { resource: buffer });
        work.step(Step::Make { resource: buffer });
        work.step(Step::Make { resource: texture });
        read_subresources(&mut work, &shape, texture, fresh_out);
        work.step(Step::Commit);
        work.step(Step::ReadBuffer {
            buffer: fresh_out,
            offset: 0,
            length: total as usize,
        });
        work
    }
}

impl Case for FreshOverBuffer {
    fn name(&self) -> &'static str {
        self.case
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        vec![shared_memory_notice()]
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut words = self.shape().words();
        words.extend(self.storage.word().as_bytes());
        words.extend(OVER_BUFFER.1.to_le_bytes());
        words.extend([OLD, BEFORE]);
        let mut inputs = vec![Input::bytes("shape-and-bytes", &words)];
        inputs.extend(program_inputs("work", &self.program())?);
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let work = self.program();
        let done = call(sites, "work", side.work(&work), |_| Value::Absent)?;
        let texel = self.shape().block.0 as usize;
        sites.push(runs_site("written", read(&done, 0), texel));
        sites.push(runs_site("fresh", read(&done, 1), texel));
        sites.push(statuses_site(&done.statuses));
        Ok(())
    }
}

/// Every case of the family, in the order the report lists them: the buffers, the textures of their
/// own memory, Metal's pattern format by format, and the textures over a new buffer.
pub fn all() -> Vec<Box<dyn Case>> {
    let mut cases: Vec<Box<dyn Case>> = Vec::new();
    for (case, storage) in BUFFERS {
        cases.push(Box::new(FreshBuffers { case, storage }));
    }
    for row in TEXTURES {
        cases.push(Box::new(FreshTexture { row }));
    }
    for row in PATTERNS {
        cases.push(Box::new(FreshPattern { row }));
    }
    for (case, storage) in OVER_BUFFERS {
        cases.push(Box::new(FreshOverBuffer { case, storage }));
    }
    cases
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allowance::Justification;
    use std::collections::BTreeSet;

    /// The runs the recorded site at `path` holds, rebuilt into the bytes they encode.
    ///
    /// `path` is node names from the root's children, `/`-separated, so a buffer row's five
    /// lengths cannot be confused for one another.
    fn recorded_bytes(case: &str, path: &str) -> Vec<u8> {
        let recording = crate::recording::read(case).unwrap_or_else(|why| panic!("{case}: {why}"));
        let answer = recording
            .answers
            .first()
            .unwrap_or_else(|| panic!("{case}: the recording holds an answer"));
        let outcome = answer
            .outcomes
            .first()
            .unwrap_or_else(|| panic!("{case}: the answer holds an outcome"));
        let mut node = &outcome.transcript.root;
        for name in path.split('/') {
            node = node
                .child(name)
                .unwrap_or_else(|| panic!("{case}: the recording holds no {path}"));
        }
        rebuild(&node.observed.value, &format!("{case}/{path}"))
    }

    /// The bytes a runs value encodes.
    fn rebuild(value: &Value, where_: &str) -> Vec<u8> {
        let Value::List(runs) = value else {
            panic!("{where_}: the recording holds {}", value.kind());
        };
        let mut bytes = Vec::new();
        let mut texels = 0u64;
        for run in runs {
            let Value::Fields(fields) = run else {
                panic!("{where_}: a run is fields");
            };
            let word = |name: &str| match fields[name] {
                Value::Uint(word) => word,
                _ => panic!("{where_}: {name} is a word"),
            };
            let Some(Value::Bytes(texel)) = fields.get("texel") else {
                panic!("{where_}: a run's texel is bytes");
            };
            assert_eq!(word("at"), texels, "{where_}");
            for _ in 0..word("texels") {
                bytes.extend_from_slice(texel);
            }
            texels += word("texels");
        }
        bytes
    }

    /// The runs are a lossless encoding: bytes rebuilt from them are the bytes they were taken
    /// from, whatever the texel's width and whether the length is a whole number of texels.
    ///
    /// Without this the whole family would be recording a summary nobody had checked, and the one
    /// projection it uses would be load-bearing on trust.
    #[test]
    fn the_runs_rebuild_the_bytes_they_came_from() {
        let cases: [(Vec<u8>, usize); 6] = [
            (Vec::new(), 4),
            (vec![0u8; 64], 1),
            (vec![0xa5; 1024], 8),
            (MAGENTA.repeat(4096), 4),
            // A length that is not a whole number of texels: the tail is its own run.
            ((0..30u8).collect(), 4),
            // Content that is not one repeated texel at all, which is the shape a divergence takes.
            ((0..=255u8).map(|b| b / 7).collect(), 2),
        ];
        for (bytes, texel) in cases {
            let runs = texel_runs(&bytes, texel);
            assert_eq!(rebuild(&runs, "a run of the guard"), bytes, "{texel} bytes");
        }
        // And it really is a run encoding: no two neighbours share a texel.
        let Value::List(runs) = texel_runs(&(0..=255u8).map(|b| b / 7).collect::<Vec<u8>>(), 2)
        else {
            panic!("the runs are a list");
        };
        for pair in runs.windows(2) {
            let texel = |value: &Value| match value {
                Value::Fields(fields) => fields["texel"].clone(),
                _ => panic!("a run is fields"),
            };
            assert_ne!(texel(&pair[0]), texel(&pair[1]));
        }
    }

    /// Every program of the family, with the case that runs it.
    fn programs() -> Vec<(&'static str, Work)> {
        let mut programs: Vec<(&'static str, Work)> = Vec::new();
        for (case, storage) in BUFFERS {
            for length in LENGTHS {
                programs.push((case, buffer_program(length, storage)));
            }
        }
        for row in TEXTURES {
            programs.push((row.case, texture_program(&row.shape, row.storage)));
        }
        for row in PATTERNS {
            programs.push((row.case, row.program()));
        }
        for (case, storage) in OVER_BUFFERS {
            programs.push((case, FreshOverBuffer { case, storage }.program()));
        }
        programs
    }

    /// The call names this family's programs key by: every resource's and every step's.
    const RESOURCE_CALLS: [&[u8]; 4] = [
        b"newBufferWithBytes",
        b"newBufferWithLength",
        b"newTextureWithDescriptor",
        b"newTextureWithDescriptorOffsetBytesPerRow",
    ];
    const STEP_CALLS: [&[u8]; 8] = [
        b"fillBuffer",
        b"copyFromBufferToBuffer",
        b"copyFromBufferToTexture",
        b"copyFromTextureToBuffer",
        b"contents",
        b"release",
        b"makeAgain",
        b"commit",
    ];

    /// **Every byte of the key is a Metal call's name and the values it takes.** Each resource's
    /// bytes open with the call that makes it and each step's with the call it runs, so no variant
    /// of [`Resource`] or [`Step`] is keyed by its position, and nothing in here is `Debug` text —
    /// which is the trap this family fell into once and the reason all 87 of its recordings moved.
    #[test]
    fn the_key_names_the_metal_call_and_holds_no_debug_text() {
        let mut keyed = 0;
        for (case, work) in programs() {
            for resource in &work.resources {
                let words = resource_words(resource).unwrap_or_else(|why| panic!("{case}: {why}"));
                assert!(
                    RESOURCE_CALLS.iter().any(|call| words.starts_with(call)),
                    "{case}: a resource keyed as {:?}",
                    String::from_utf8_lossy(&words[..words.len().min(32)])
                );
                keyed += 1;
            }
            for step in &work.steps {
                let words = step_words(step).unwrap_or_else(|why| panic!("{case}: {why}"));
                assert!(
                    STEP_CALLS.iter().any(|call| words.starts_with(call)),
                    "{case}: a step keyed as {:?}",
                    String::from_utf8_lossy(&words[..words.len().min(32)])
                );
                keyed += 1;
            }
            // And the shape a derived `Debug` would leave behind is in none of it. A key written
            // from `format!("{:?}")` holds a field name, a brace and `Some(`; this one holds the
            // call's own name and numbers.
            let mut steps = Vec::new();
            for step in &work.steps {
                steps.extend(step_words(step).unwrap());
            }
            for print in [&b" { "[..], b"Some(", b"bytes_per_row"] {
                assert!(
                    !steps.windows(print.len()).any(|window| window == print),
                    "{case}: the steps key holds {:?}",
                    String::from_utf8_lossy(print)
                );
            }
        }
        assert!(keyed > 0);
    }

    /// **A descriptor states a field only where it is not Metal's default for it.** This is the
    /// property that keeps a widened [`TextureSpec`] from moving a recording: a field every row of
    /// this family leaves at its default writes nothing, so adding one writes nothing.
    ///
    /// The all-default descriptor's bytes are pinned exactly, so a field that started writing
    /// itself unconditionally would be caught here rather than by 87 stale recordings.
    #[test]
    fn a_descriptor_states_a_field_only_where_it_is_not_the_default() {
        let mut expected = Vec::new();
        expected.extend(b"pixelFormat");
        expected.extend(u64::from(RGBA8).to_le_bytes());
        expected.extend(b"width");
        expected.extend(64u64.to_le_bytes());
        expected.extend(b"height");
        expected.extend(64u64.to_le_bytes());
        let plain = TextureSpec::texture_2d(RGBA8, 64, 64).shared();
        assert_eq!(descriptor_words(&plain), expected);
        // Every field set away from its default adds its own tag, and no two of them collide.
        let mut volume = plain;
        volume.depth = 4;
        let mut sampled = plain;
        sampled.samples = 4;
        let varied: [(&str, TextureSpec); 7] = [
            ("storage", plain.stored(Storage::Private)),
            ("levels", plain.levels(7)),
            ("array", plain.array(3)),
            ("plain", plain.plain()),
            ("usage", plain.used(READ)),
            ("depth", volume),
            ("samples", sampled),
        ];
        let mut keys: Vec<(&str, Vec<u8>)> = vec![("default", expected.clone())];
        for (named, spec) in varied {
            let words = descriptor_words(&spec);
            assert!(
                words.starts_with(&expected) && words.len() > expected.len(),
                "{named}: states nothing of its own"
            );
            keys.push((named, words));
        }
        for (at, (named, words)) in keys.iter().enumerate() {
            for (other, theirs) in &keys[at + 1..] {
                assert_ne!(words, theirs, "{named} keys as {other}");
            }
        }
        // And a field put back at its default writes nothing again.
        let mut back = volume;
        back.depth = 1;
        assert_eq!(descriptor_words(&back), expected);
    }

    /// **The storage modes a row sweeps still key apart.** The default-elision writes a shared
    /// buffer's mode nowhere, so this is the check that it has not made the three buffer rows'
    /// inputs one key.
    #[test]
    fn the_three_storage_modes_key_apart() {
        let keys: Vec<Vec<u8>> = [Storage::Shared, Storage::Managed, Storage::Private]
            .into_iter()
            .map(|storage| {
                resource_words(&Resource::UnwrittenBuffer {
                    length: 16,
                    storage,
                })
                .unwrap()
            })
            .collect();
        assert_ne!(keys[0], keys[1]);
        assert_ne!(keys[0], keys[2]);
        assert_ne!(keys[1], keys[2]);
        // The same for a texture, whose rows sweep the modes too.
        let shape = RGBA8_2D;
        let specs: Vec<Vec<u8>> = [Storage::Shared, Storage::Managed, Storage::Private]
            .into_iter()
            .map(|storage| descriptor_words(&shape.spec(storage)))
            .collect();
        assert_ne!(specs[0], specs[1]);
        assert_ne!(specs[0], specs[2]);
        assert_ne!(specs[1], specs[2]);
    }

    /// **A program that read the wrong buffer keys differently.** This is why a resource's index is
    /// in the key: [`buffer_program`]'s two copies and two reads differ by nothing else, so a key
    /// without the indices would let a program that read its *written* buffer twice match a
    /// recording taken from the program that reads the fresh one.
    #[test]
    fn a_program_that_reads_the_wrong_buffer_keys_differently() {
        let right = buffer_program(16, Storage::Shared);
        let keyed = |work: &Work| {
            let mut out = Vec::new();
            for step in &work.steps {
                out.extend(step_words(step).unwrap());
            }
            out
        };
        // The last copy and the last read are redirected at the buffer the first pair used.
        let mut wrong = right.clone();
        for step in &mut wrong.steps {
            match step {
                Step::CopyBufferToBuffer { to, .. } if *to == 2 => *to = 1,
                Step::ReadBuffer { buffer, .. } if *buffer == 2 => *buffer = 1,
                _ => {}
            }
        }
        assert_ne!(right, wrong, "the mutation changed the program");
        assert_ne!(
            keyed(&right),
            keyed(&wrong),
            "two programs reading different buffers key alike"
        );
    }

    /// **The order of the steps is in the key.** The reason the steps were fingerprinted at all:
    /// a release moved to where a command buffer is still open is a different program, which must
    /// not match this one's recording.
    #[test]
    fn reordering_a_release_moves_the_key() {
        let right = buffer_program(16, Storage::Shared);
        let keyed = |work: &Work| {
            let mut out = Vec::new();
            for step in &work.steps {
                out.extend(step_words(step).unwrap());
            }
            out
        };
        let at = right
            .steps
            .iter()
            .position(|step| matches!(step, Step::Release { .. }))
            .expect("the program releases its subject");
        let mut moved = right.clone();
        moved.steps.swap(at, at - 1);
        assert_ne!(keyed(&right), keyed(&moved));
    }

    /// A resource or a step this family does not state has **no key**, rather than a `Debug` one
    /// whose bytes no recording was taken against. No case reaches it — every program here is
    /// built in this module — and a case that wants one adds its arm.
    #[test]
    fn a_resource_or_step_this_family_does_not_state_is_refused() {
        assert!(resource_words(&Resource::Heap {
            size: 4096,
            storage: Storage::Private,
        })
        .is_err());
        assert!(step_words(&Step::FillTexture {
            texture: 0,
            level: 0,
            slice: 0,
            region: crate::work::Rect::new(0, 0, 4, 4),
            pattern: vec![0xff],
        })
        .is_err());
        // And every program of the family does have one, so the refusal is reachable only from
        // outside what these cases state.
        for (case, work) in programs() {
            for resource in &work.resources {
                assert!(resource_words(resource).is_ok(), "{case}");
            }
            for step in &work.steps {
                assert!(step_words(step).is_ok(), "{case}");
            }
        }
    }

    /// Every case names itself once, and no two rows share a name.
    #[test]
    fn the_cases_are_distinct() {
        let names: Vec<String> = all().iter().map(|case| case.name().to_owned()).collect();
        let distinct: BTreeSet<&String> = names.iter().collect();
        assert_eq!(distinct.len(), names.len(), "a name is used twice");
        assert_eq!(names.len(), 3 + TEXTURES.len() + PATTERNS.len() + 2);
    }

    /// Every case states its inputs and asks a program that reads something, so no row can record
    /// an answer nothing read.
    #[test]
    fn every_case_states_its_inputs_and_reads_something() {
        for case in all() {
            let inputs = case
                .inputs()
                .unwrap_or_else(|why| panic!("{}: {why}", case.name()));
            assert!(!inputs.is_empty(), "{}", case.name());
        }
        for (_, storage) in BUFFERS {
            for length in LENGTHS {
                assert_eq!(buffer_program(length, storage).reads(), 2);
            }
        }
        for row in TEXTURES {
            assert_eq!(
                texture_program(&row.shape, row.storage).reads(),
                2,
                "{}",
                row.case
            );
        }
        for row in PATTERNS {
            assert_eq!(row.program().reads(), 1, "{}", row.case);
        }
    }

    /// **Every row that reads a fresh resource first writes one.** This is the guard the family's
    /// honesty rests on: a fresh resource's contents are not a Metal promise, so a row that read
    /// one without the old bytes having been there would be reading memory nobody touched.
    ///
    /// Each program with a `fresh` read holds, in order, a fill or a copy into the subject, a
    /// commit, a release of the subject and a make of it, before the read-back the `fresh` site
    /// comes from. The pattern rows are the stated exception and are checked the other way: they
    /// release nothing, and their destination holds [`BEFORE`].
    #[test]
    fn every_released_row_writes_before_it_reads() {
        let mut checked = 0;
        let mut programs: Vec<(&str, Work)> = Vec::new();
        for (case, storage) in BUFFERS {
            for length in LENGTHS {
                programs.push((case, buffer_program(length, storage)));
            }
        }
        for row in TEXTURES {
            programs.push((row.case, texture_program(&row.shape, row.storage)));
        }
        for (case, storage) in OVER_BUFFERS {
            programs.push((case, FreshOverBuffer { case, storage }.program()));
        }
        for (case, work) in &programs {
            let wrote = work.steps.iter().position(|step| {
                matches!(
                    step,
                    Step::FillBuffer { value, .. } if *value == OLD
                ) || matches!(step, Step::CopyBufferToTexture { .. })
            });
            let released = work
                .steps
                .iter()
                .position(|step| matches!(step, Step::Release { .. }));
            let made = work
                .steps
                .iter()
                .position(|step| matches!(step, Step::Make { .. }));
            let (wrote, released, made) = (
                wrote.unwrap_or_else(|| panic!("{case}: nothing wrote the subject")),
                released.unwrap_or_else(|| panic!("{case}: nothing was released")),
                made.unwrap_or_else(|| panic!("{case}: nothing was made again")),
            );
            assert!(wrote < released, "{case}: released before it was written");
            assert!(released < made, "{case}: made again before it was released");
            // Every release follows a commit, so the work that used the resource has completed and
            // the release frees something.
            let committed = work.steps[..released]
                .iter()
                .rposition(|step| matches!(step, Step::Commit));
            assert!(
                committed.is_some(),
                "{case}: released with work outstanding"
            );
            // The last read-back comes after the resource was made again.
            let last_read = work
                .steps
                .iter()
                .rposition(|step| step.reads())
                .unwrap_or_else(|| panic!("{case}: nothing is read"));
            assert!(
                made < last_read,
                "{case}: the fresh read is not after the make"
            );
            checked += 1;
        }
        assert_eq!(checked, programs.len());
        // The pattern rows release nothing, and say so here rather than by omission.
        for row in PATTERNS {
            let work = row.program();
            assert!(
                !work
                    .steps
                    .iter()
                    .any(|step| matches!(step, Step::Release { .. } | Step::Make { .. })),
                "{}",
                row.case
            );
        }
    }

    /// Every read-back destination holds [`BEFORE`], which is neither zero nor [`OLD`], so a site
    /// that reads zero has proved its copy ran.
    #[test]
    fn every_destination_holds_a_byte_that_is_neither_zero_nor_old() {
        assert_ne!(BEFORE, 0);
        assert_ne!(BEFORE, OLD);
        let mut programs: Vec<(&str, Work)> = vec![];
        for (case, storage) in BUFFERS {
            programs.push((case, buffer_program(LENGTHS[0], storage)));
        }
        for row in TEXTURES {
            programs.push((row.case, texture_program(&row.shape, row.storage)));
        }
        for row in PATTERNS {
            programs.push((row.case, row.program()));
        }
        for (case, storage) in OVER_BUFFERS {
            programs.push((case, FreshOverBuffer { case, storage }.program()));
        }
        for (case, work) in &programs {
            for step in &work.steps {
                if let Step::ReadBuffer { buffer, length, .. } = step {
                    match &work.resources[*buffer] {
                        Resource::Buffer { bytes } => {
                            assert_eq!(bytes.len(), *length, "{case}");
                            assert!(bytes.iter().all(|&byte| byte == BEFORE), "{case}");
                        }
                        other => panic!("{case}: a read of {other:?}"),
                    }
                }
            }
        }
    }

    /// Every copy a program encodes lies inside its resource, so no program can hand a device an
    /// out-of-bounds transfer.
    #[test]
    fn every_copy_lies_inside_its_resource() {
        let mut programs: Vec<(&str, Work)> = vec![];
        for row in TEXTURES {
            programs.push((row.case, texture_program(&row.shape, row.storage)));
        }
        for row in PATTERNS {
            programs.push((row.case, row.program()));
        }
        for (case, work) in &programs {
            let length = |at: usize| match &work.resources[at] {
                Resource::Buffer { bytes } => bytes.len() as u64,
                Resource::UnwrittenBuffer { length, .. } => *length,
                other => panic!("{case}: resource is {other:?}"),
            };
            for step in &work.steps {
                match step {
                    Step::CopyTextureToBuffer {
                        buffer,
                        offset,
                        bytes_per_image,
                        ..
                    } => assert!(
                        offset + bytes_per_image <= length(*buffer),
                        "{case}: a read past the destination"
                    ),
                    Step::CopyBufferToTexture {
                        buffer,
                        offset,
                        bytes_per_image,
                        ..
                    } => assert!(
                        offset + bytes_per_image <= length(*buffer),
                        "{case}: a write past the source"
                    ),
                    _ => {}
                }
            }
        }
    }

    /// The pattern table names each format once, and each row's name names its format, so a
    /// recording cannot be read as another format's.
    #[test]
    fn every_pattern_row_names_its_own_format() {
        let formats: BTreeSet<u32> = PATTERNS.iter().map(|row| row.format).collect();
        assert_eq!(formats.len(), PATTERNS.len(), "a format is repeated");
        for row in PATTERNS {
            assert_eq!(row.case, format!("fresh-pattern-{}", row.format));
            assert!(!row.texel.is_empty(), "{}", row.case);
            assert!(
                row.texel.len().is_power_of_two() && row.texel.len() <= 16,
                "{}: {} bytes a texel",
                row.case,
                row.texel.len()
            );
        }
    }

    /// **A row loosens the notices site and nothing else.** Not one byte of this family is ever
    /// loosened: MoltenVK, lavapipe and an RTX 5080 Laptop all read Metal's own bytes, and the only
    /// field that may differ is the set of notices our side announced.
    ///
    /// Every row names [`shared_memory_notice`], which is the venue's memory model and not the
    /// row's; **only** the four rows a venue announced something of their own for name a second
    /// kind, and each names the kind that venue announced. A row whose notices were a guess would
    /// show here as a row loosening a site nothing needed.
    #[test]
    fn a_row_loosens_the_notices_site_alone() {
        let notices = format!("case/{}/*", crate::replay::NOTICES_SITE);
        let shared = Degradation::SharedInSystemMemory.code().to_owned();
        for case in all() {
            let allowances = case.allowances();
            for allowance in &allowances {
                assert_eq!(allowance.path, notices, "{}", case.name());
            }
            // The venue's notice is the one every row declares, and it is declared exactly once.
            let declared = allowances
                .iter()
                .filter(|allowance| {
                    allowance.justified_by == Justification::Degradation(shared.clone())
                })
                .count();
            assert_eq!(declared, 1, "{}", case.name());
        }
        let named: Vec<&str> = TEXTURES
            .iter()
            .filter(|row| !row.notices.is_empty())
            .map(|row| row.case)
            .collect();
        assert_eq!(
            named,
            vec![
                "fresh-texture-bc1-rgba-shared",
                "fresh-texture-bc1-rgba-managed",
                "fresh-texture-bc1-rgba-private",
                "fresh-texture-depth32float-private",
            ]
        );
        for row in TEXTURES {
            assert_eq!(
                FreshTexture { row }.allowances().len(),
                1 + row.notices.len(),
                "{}",
                row.case
            );
        }
        // Everything else is held to Metal exactly, the venue's own notice apart.
        for (case, storage) in BUFFERS {
            assert_eq!(
                FreshBuffers { case, storage }.allowances().len(),
                1,
                "{case}"
            );
        }
        for (case, storage) in OVER_BUFFERS {
            assert_eq!(
                FreshOverBuffer { case, storage }.allowances().len(),
                1,
                "{case}"
            );
        }
        for row in PATTERNS {
            assert_eq!(FreshPattern { row }.allowances().len(), 1, "{}", row.case);
        }
    }

    /// The geometry of a shape is the lane's: a level's extent halves and stops at one, and a
    /// block-compressed level's rows are whole blocks.
    #[test]
    fn a_levels_geometry_is_the_lanes() {
        let ([w, h], row, image) = RGBA8_MIPMAPPED.level(2);
        assert_eq!([w, h], [16, 16]);
        assert_eq!((row, image), (64, 1024));
        let ([w, h], row, image) = BC1_RGBA.level(0);
        assert_eq!([w, h], [64, 64]);
        // Sixteen blocks a row, eight bytes each; sixteen rows of blocks.
        assert_eq!((row, image), (128, 2048));
        let ([w, h], ..) = chain([16, 256], 9, 1).level(8);
        assert_eq!([w, h], [1, 1]);
        // A 100 by 40 chain's second level is 50 by 20, and its third 25 by 10.
        assert_eq!(chain([100, 40], 7, 2).level(1).0, [50, 20]);
        assert_eq!(chain([100, 40], 7, 2).level(2).0, [25, 10]);
    }

    /// The lane's expectation, as this family states it: zero in every `Shared` and `Managed`
    /// texture, Metal's pattern in a `Private` one's patterned levels and zero in the rest.
    #[test]
    fn the_expectation_is_the_lanes_rule() {
        for storage in [Storage::Shared, Storage::Managed] {
            let expected = RGBA8_2D.expected_fresh(storage);
            assert!(expected.iter().all(|&byte| byte == 0), "{storage:?}");
        }
        let patterned = RGBA8_2D.expected_fresh(Storage::Private);
        assert!(patterned.chunks(4).all(|texel| texel == MAGENTA));
        // The mipmapped chain's first three levels hold the pattern and the rest hold zero.
        let chain_bytes = RGBA8_MIPMAPPED.expected_fresh(Storage::Private);
        let wide: usize = (0..3).map(|l| RGBA8_MIPMAPPED.level(l).2 as usize).sum();
        assert!(chain_bytes[..wide].chunks(4).all(|texel| texel == MAGENTA));
        assert!(chain_bytes[wide..].iter().all(|&byte| byte == 0));
        // The uncompressed `Private` rows read zero, which is what their missing pattern says.
        for shape in [RGBA8_PLAIN, R32FLOAT_WRITTEN, BC1_RGBA] {
            assert!(shape
                .expected_fresh(Storage::Private)
                .iter()
                .all(|&byte| byte == 0));
        }
    }

    /// **The cross-check the port owes the lane.** Metal's own bytes, as the committed recordings
    /// hold them, are the bytes the lane expects: the old bytes come back out of the written
    /// resource, and the fresh one reads zero or Metal's pattern by the lane's rule.
    ///
    /// Without this the port would be a set of recordings nobody had compared with the lane that
    /// justified writing them.
    #[test]
    fn metals_texels_are_the_lanes_expectation() {
        // A buffer of every storage mode and every length: the old bytes come back out of the
        // written one, and the one made after it was released reads zero.
        for (case, _) in BUFFERS {
            for length in LENGTHS {
                let at = length as usize;
                assert_eq!(
                    recorded_bytes(case, &format!("bytes-{length}/written")),
                    vec![OLD; at],
                    "{case} at {length} bytes: the written buffer"
                );
                assert_eq!(
                    recorded_bytes(case, &format!("bytes-{length}/fresh")),
                    vec![0u8; at],
                    "{case} at {length} bytes: the buffer made after it was released"
                );
            }
        }
        // A texture of every shape: the old bytes come back, and the fresh one reads the lane's
        // rule — zero, or Metal's pattern in a `Private` one's patterned levels.
        for row in TEXTURES {
            assert_eq!(
                recorded_bytes(row.case, "written"),
                written_bytes(&row.shape),
                "{}: the written texture",
                row.case
            );
            assert_eq!(
                recorded_bytes(row.case, "fresh"),
                row.shape.expected_fresh(row.storage),
                "{}: the texture made after it was released",
                row.case
            );
        }
        // Each format's pattern: every texel of a new `Private` texture holds the lane's bytes.
        for row in PATTERNS {
            let texels = row.total() as usize / row.texel.len();
            assert_eq!(
                recorded_bytes(row.case, "fresh"),
                row.texel.repeat(texels),
                "{}: Metal's pattern",
                row.case
            );
        }
        // A texture over a buffer: the buffer's old bytes read through the texture, and zero out of
        // the one made over a buffer allocated after the old one was released.
        for (case, storage) in OVER_BUFFERS {
            let total = FreshOverBuffer { case, storage }.shape().total() as usize;
            assert_eq!(recorded_bytes(case, "written"), vec![OLD; total], "{case}");
            assert_eq!(recorded_bytes(case, "fresh"), vec![0u8; total], "{case}");
        }
    }

    /// The bytes a texture of `shape` reads once the [`OLD`] write has covered every level and
    /// slice: one block's bytes repeated through each level's image.
    fn written_bytes(shape: &Shape) -> Vec<u8> {
        let mut all = Vec::new();
        for (level, _) in shape.subresources() {
            let bytes = shape.level(level).2 as usize;
            all.extend(shape.old.iter().copied().cycle().take(bytes));
        }
        all
    }
}
