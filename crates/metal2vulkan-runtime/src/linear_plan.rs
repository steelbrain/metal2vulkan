//! Textures over buffers: which pixel formats a buffer can back, where Metal lays a texture out in
//! its buffer, which of Metal's rules a texture over a buffer breaks, and when a linear Vulkan
//! image in the buffer's memory holds exactly Metal's layout.
//!
//! **Metal's rules** (probes 9 to 9f, on an Apple M3 Max, macOS 26.7, and an M4, macOS 27.0.1,
//! which agree; each case run plain and under API validation). The runtime refuses as `Invalid`
//! what Metal asserts on in a plain run, and as `Nil` what plain Metal answers nil for; what only
//! its API validation layer stops is no refusal, since plain Metal takes it:
//!
//! - **Formats.** `minimumLinearTextureAlignmentForPixelFormat:` and
//!   `minimumTextureBufferAlignmentForPixelFormat:` are both 16 bytes for every format a buffer
//!   can back: every uncompressed color format, and `Unspecialized`. Compressed formats (the 4:2:2
//!   ones among them) and depth and stencil formats cannot be backed by a buffer, and Metal's
//!   query asserts on them ([`BufferTextureProblem::NotLinear`]). A plain run of
//!   `newTextureWithDescriptor:offset:bytesPerRow:` makes a 4:2:2 texture all the same; the
//!   runtime, which holds no 4:2:2 texture over a buffer, refuses it `Unsupported` (`Format`)
//!   there, not as Metal's refusal. An IOSurface plane of either 4:2:2 format is made (below).
//! - **Types.** `newTextureWithDescriptor:offset:bytesPerRow:` makes 2D textures and texture
//!   buffers; the SPI `newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:` also
//!   makes 2D arrays. Each has one mip level. The SPI
//!   `newTiledTextureWithDescriptor:offset:bytesPerRow:` makes what the first makes, and a 2D
//!   texture of more levels too, a copy of the buffer's rows at the offset and the `bytesPerRow`
//!   taken at its first use, its later levels made from them (`Buffer::new_tiled_texture`); its
//!   other rules are the first's.
//! - **Options.** The texture's options are the buffer's, whatever the descriptor names (probe
//!   9d, every pair of six buffer and nine descriptor options): Metal's validation alone stops a
//!   storage or CPU cache mode other than the buffer's, or a hazard tracking mode other than the
//!   default and the buffer's, and a plain run takes the buffer's. An IOSurface plane takes its
//!   descriptor's instead (below).
//! - **Pitches.** `bytesPerRow` covers the packed row (`width` × the texel's bytes). A 2D
//!   texture's offset and `bytesPerRow` are multiples of 16 ([`LINEAR_ALIGNMENT`]); an array's
//!   offset is a multiple of 128 ([`ARRAY_ALIGNMENT`]) and its `bytesPerRow` of 16; a texture
//!   buffer's `bytesPerRow` is any length that covers the row. A texture buffer's offset is any:
//!   Metal's validation alone stops one off 16, and a plain run makes the texture, its functions
//!   reading and writing from the offset rounded down to 16 and its blits and the CPU at the
//!   offset (an M4, macOS 27.0.1; `buffer_texture.rs`), so the runtime does too. The SPIs' texture
//!   buffers, unmeasured there, keep the validated rule. An array's `bytesPerImage` is ignored:
//!   each slice starts `bytesPerRow` × `height`, rounded up to 128, after the one before. Metal's
//!   validation alone asserts that it is a multiple of 128; a plain run makes the array at any (16,
//!   32, 64, 80 and 192 measured, probe 9b), so the runtime does too.
//! - **Length.** Metal's validation counts `offset + bytesPerRow × (height × slices − 1) + the
//!   packed row` against the buffer's length, for an array as if its slices were not rounded, and
//!   stops a texture whose count runs past it; plain Metal reads the buffer's `allocatedSize`
//!   alone. It answers nil where the buffer is at least 16 KiB and the texture's rounded slices
//!   end past the length rounded up to 16 KiB (its `allocatedSize`), and otherwise makes the
//!   texture over bytes past the length, however far past (an M3 Max under macOS 27.0.1, the
//!   layer off: a 2D texture and a texture buffer over a 4096-byte buffer whose rows end 128 and
//!   12288 bytes past it are made, as is one over a 20000-byte buffer, `allocatedSize` 32768,
//!   from offset 20000; the same ending past 32768 is nil, and so is one over a 16384-byte
//!   buffer, `allocatedSize` 16384, ending at 16640, where one ending at 16384 is made). Every
//!   path takes those texels as the memory after the buffer's (an M3 Max under macOS 27.0.1, the
//!   layer off, one shape and one path a process, `probes/texels_past_buffer_length.m`): over a
//!   4096-byte `Shared` buffer, rows of 256 bytes ending 256 to 2048 bytes past it, a kernel's
//!   `read()` and `sample()`, a blit into a buffer and `getBytes` read the buffer allocated next
//!   (at the target's `gpuAddress` plus 4096) there, and a kernel's `write()`, a blit from a
//!   buffer and `replaceRegion` write it there, which every read then reads; over a 20000-byte
//!   buffer (`allocatedSize` 32768) the bytes past the length, inside the allocation, read zero,
//!   and what each path writes there every path reads. The texels inside the length are the
//!   buffer's bytes on every path. A texture buffer's paths answer alike
//!   (`probes/texture_buffer_past_length.m`, 64 `RGBA8Unorm` texels ending 128 to 256 bytes past
//!   a 4096-byte buffer, and 96 past a 20000-byte one), its kernels from its offset rounded down
//!   to 16 bytes and its blits and `getBytes` from the offset itself, as inside the length. The
//!   runtime makes such a 2D texture, array or texture buffer as Metal does: its texels inside
//!   the length are the buffer's bytes, and those past it lie in a tail of the texture's own
//!   (`buffer_texture.rs`), zero when made, which every path reads and writes alike and which is
//!   no other allocation's memory; the nil one it answers as Metal's nil
//!   ([`BufferTextureProblem::PastAllocation`]).
//!
//! **IOSurface planes** (`Buffer::new_texture_with_iosurface_plane`, the stand-in for
//! `newTextureWithDescriptor:iosurface:plane:` over the caller's memory). Measured on the M3 Max
//! (macOS 26.7) and an M4 (macOS 27.0.1), which agree on every byte but the memory-dependent reads
//! and the overlapping writes below (one process a case, plain twice and
//! under API validation, which flags none of them):
//!
//! - **Made.** A surface without planes takes a `bytesPerRow` that is a multiple of 16 only: Metal
//!   asserts on any other, plain too ("IOSurface texture: bytesPerRow (4) must be aligned to 16
//!   bytes"). A plane of a planar surface (the consumer's form: one plane dictionary, or three for
//!   its '420v' backing, base and offset alike) takes any offset and any `bytesPerRow` that covers
//!   its row, a multiple of the texel's size or not. IOSurface makes no plane whose pitch is below
//!   its row.
//! - **Options** (an M4, macOS 27.0.1, flat and planar surfaces). The texture
//!   takes and reports its descriptor's storage, CPU cache and hazard tracking modes (its default
//!   hazard tracking mode resolved to tracked), whatever the surface's memory or the buffer
//!   standing for it: `Shared` and `Managed`, each CPU cache mode and each hazard tracking mode are
//!   made silently, and read and write the surface's bytes. Metal's API validation layer asserts on
//!   `Private` and `Memoryless` ("IOSurface textures must use MTLStorageModeManaged or
//!   MTLStorageModeShared"); a plain run makes both, and a `Private` one's GPU and CPU read and
//!   write the surface's bytes as a `Shared` one's do, so the runtime makes it too. A plain run
//!   makes a `Memoryless` one too, whose CPU reaches the plane's bytes while its GPU uses fail the
//!   command buffer and a render pass into it stores nothing; the runtime holds no form that
//!   answers all three, so it refuses that texture as not implemented
//!   ([`BufferRefusal::MemorylessPlane`],
//!   [`RuntimeGap::MemorylessSurfacePlane`](crate::RuntimeGap::MemorylessSurfacePlane), which
//!   carries the measurement).
//! - **The CPU** (`getBytes`, `replaceRegion`) reads and writes the plane where the surface lays it
//!   out: from its offset, rows `bytesPerRow` apart.
//! - **The GPU** (a blit's reads and writes, a copy to or from another texture, a kernel's
//!   `read()`, `sample()` and `write()`, a render pass's store) reads and writes it from its offset
//!   rounded down to 16, rows `bytesPerRow` rounded down to 16 apart ([`iosurface_gpu_layout`]):
//!   `bytesPerRow` itself where it is a multiple of 16, else other bytes than the CPU's (a plane
//!   at offset 4 starts at byte 0, an NV12 chroma plane at 36 at byte 32).
//! - **Under 16 bytes a row** that pitch is 0. The GPU writes the rows 16 bytes apart, and reads
//!   the first row there, but its reads of every later row are memory-dependent: zeros in a fresh
//!   process, and after a write other bytes, neither the surface's nor the written ones (not the
//!   same on the M3 Max and the M4). The runtime's GPU reads those rows where its writes put them,
//!   a recorded near-equivalent (`Degradation::NarrowSurfaceRows`, [`Placed::narrow`]).
//! - **4:2:2 planes** (an M4 under macOS 27.0.1, plain twice and validated
//!   alike): a texture of either 4:2:2 format over a CoreVideo `2vuy` or `yuvs` surface (no planes,
//!   `bytesPerRow` 64 for 16 x 4) is made, in either format over either surface; `getBytes` reads
//!   the surface's rows, `replaceRegion` writes them leaving the bytes between untouched, and a
//!   blit into a buffer reads them, 2 bytes a texel, unchanged. The runtime holds it as the 4:2:2
//!   textures of its own are held, in the carrier (`chroma_pairs.rs`), 2 bytes a texel, under the
//!   rules above.
//! - **Rows that overlap.** A row of more than 16 bytes whose `bytesPerRow` rounded down to 16 is
//!   below it overlaps the next row on the GPU: a read sees the overlap, and where two rows'
//!   writes meet the bytes they share take whichever of the two rows Metal's command wrote last.
//!   That order is no one order (an Apple M3 Max under macOS 27.0.1, one case a process,
//!   `probes/overlapping_plane_row_writes.m`): it follows the command, the GPU pitch and the row
//!   count, and how far such rows overlap changes none of it. `copyFromBuffer:toTexture:` and
//!   `copyFromTexture:toTexture:` at a GPU pitch of 16 keep the earlier row's bytes below every
//!   even row and the later row's below every odd one, alternating over 2 to 12 rows; at a pitch
//!   of 32 they keep the earlier row's over 4 rows and the later row's over 3, 5 and 6;
//!   `copyFromTexture:toTexture:` into 33 bytes a row at a pitch of 32 over 4 rows has no answer
//!   at all, 23 fresh processes giving three whole-plane outcomes; and a kernel's winner moves
//!   with the threadgroup shape.
//!
//!   Rows wider than a plane's `bytesPerRow` at a multiple of 16 (**Beyond the plane's rows**)
//!   overlap the rows after them on the GPU and the CPU alike, by any whole number of texels, a
//!   byte shared by as many rows as reach it. Metal's GPU gives their shared bytes the answers it
//!   gives the rows of the same GPU pitch and row above (`r8.20x4.w16`, `r8.24x4.w16`,
//!   `r8.31x4.w16`, `r8.20x8.w16` and `rgba8.5x4.w4` beside their twins at a pitch off 16, three
//!   processes a case: alike in every case but one of three 32 x 1 kernel runs over 20 x 8),
//!   though how far wider rows overlap does move a blit's order (at a pitch of 32 over 4 rows the
//!   earlier row's where they share 1 byte, the later row's where they share 16); its kernel of
//!   8 x 4 threadgroups keeps the earliest row's within the first threadgroup's rows, where three
//!   rows share a byte too (`r8.40x4.w16`); and its `replaceRegion` writes the rows in order, the
//!   latest row's bytes standing wherever rows meet (all nine shapes, three processes each).
//!
//!   A plane's pitch off 16 covers its own row, so that overlap is 1 to 15 bytes, a whole number
//!   of texels, and between consecutive rows alone (`tests::linear_plan`). No image holds rows
//!   that overlap, so such a plane is a shadow whose copies reproduce the sharing: two regions a
//!   slice each way, one of every row's shared head (in pieces no wider than the pitch, where it is
//!   wider) and one of the rest of each row going in, and going back the second of them for every
//!   row beside the head of row 0 alone, which tile the plane's bytes exactly once and leave the
//!   earliest row's bytes wherever rows meet (`buffer_texture.rs`,
//!   [`plane_copy_regions`](super::buffer_texture::plane_copy_regions)). A kernel is the only GPU
//!   writer that reaches such a plane, every blit into one being refused below, and the earliest
//!   row's bytes are one of the rows a kernel of Metal's leaves there, announced
//!   (`Degradation::CollidingSurfaceRowWrites`). The CPU writes a wide plane's rows in order, as
//!   Metal's `replaceRegion` does.
//!
//!   Still refused ([`BufferRefusal::OverlappingRows`], `NotImplemented`,
//!   `RuntimeGap::OverlappingSurfaceRows`): a plane a render pass stores into, where what Metal
//!   leaves of the shared bytes is not measured; the shapes whose rows of codes or units no row
//!   rule bounds (a two-plane YCbCr texture's, which overlap by any amount across any number of
//!   rows, `ycbcr.rs`; a two-plane extended-range one's, a 4:2:2 one's pairs, a depth or stencil
//!   plane's elements), which two regions do not express; and a blit that would write such a
//!   plane's rows as bytes (`blit_texture.rs`: a copy into it from a buffer, and one between two
//!   textures over buffers, whichever side it is on), which writes every shared byte from one
//!   fixed row where Metal's order is no one order ([`super::blit_plan::RowsCopy`] holds only the
//!   one overlap that has an order, a buffer's rows whose images meet). A blit out of such a plane
//!   into a buffer is served instead of refused, its rows being read and not written: each of the
//!   buffer's rows takes the plane's bytes at its own GPU row, so a byte two rows share is read
//!   once for each of them, which is what Metal's `copyFromTexture:toBuffer:` reads (an Apple M3
//!   Max under macOS 27.0.1, one shape a process, `probes/overlapping_plane_row_reads.m`). A read
//!   is not the mirror of a write here: a write must settle which row leaves the shared bytes
//!   behind and Metal settles it no one way, while a read settles nothing. A copy between the
//!   plane and an image texture is refused either way, its one `bufferRowLength` stating no
//!   sharing. A rotated plane never reaches
//!   any of this: its `bytesPerRow` must be a multiple of 16
//!   (`buffer_texture::turned_plane`), so its GPU rows lie at the pitch itself and share no byte
//!   (`tests::buffer_texture::no_rotated_plane_s_gpu_rows_share_bytes`).
//! - **Two-plane YCbCr codes** (an M3 Max under macOS 27.0.1; `ycbcr.rs`'s **Surfaces**). Metal
//!   makes a two-plane YCbCr texture over any plane, its codes at the plane's rows however wide
//!   they are: a 10-bit format over an 8-bit plane, whose rows of codes overlap where
//!   `bytesPerRow` is under twice the width, and a texture taller than its plane, whose rows run
//!   past the surface, where its reads are zeros. The runtime refuses rows that overlap as above,
//!   and codes past the buffer `NotImplemented` (`TexelsPastBufferLength`).
//! - **Depth and stencil over a planar surface** (an M4 under macOS 27.0.1, the format scout's
//!   cells: TTG2's scout, `scout4-metal-vs.tsv`). Metal makes, samples or reads, and draws a
//!   `Depth16Unorm`, `Depth32Float`, `Depth32Float_Stencil8`, `Stencil8` or `X32_Stencil8` texture
//!   over a plane of a surface with planes; over a surface without planes it stops one, as every
//!   buffer's texture of a depth or stencil format ([`BufferTextureProblem::NotLinear`]). Over a
//!   plane the texture keeps one aspect's unit a texel (`copy_plan::surface_depth_unit`: its depth,
//!   else its stencil; the stencil of `Depth32Float_Stencil8` is not in the plane), laid out by the
//!   rules above as a colour texel's: its reads, samples and CPU access take the plane's bytes
//!   there. The texture is a shadow of those bytes (`shadow.rs`), whose copies move that aspect
//!   alone. A render pass stores it otherwise, in Metal's tiled layout (`texture/heap_tiled.rs`)
//!   from the plane's offset rounded down to 128: it renders through a tiled twin
//!   (`texture/surface_depth.rs`, which lists what stays the runtime's gap,
//!   `NotImplemented(RuntimeGap::SurfaceDepthStencil)`, with a blit of the stencil of
//!   `Depth32Float_Stencil8`).
//! - **Block-compressed surfaces** (an AMD Radeon Pro 5300M and an Intel UHD 630 under macOS
//!   26.7.1, `metal2vulkan-conformance/probes/surface_texture_envelope.m`; the consumer's
//!   `oracle_surftex` 66, 67, 125, 131 to 135, 147 and 148). The x86 Macs make a BC1 to BC7
//!   texture over a surface's rows wherever they fit, over a surface with planes or without, in
//!   `Managed` or `Private` storage; the M3 Max answers nil. Their blits (a `Private`
//!   texture's too), `getBytes`, the UHD 630's `replaceRegion` and both GPUs' samples take block
//!   row `j` at the plane's offset + `j` × `bytesPerRow`, the row its blocks' bytes (8 or 16 a
//!   block, [`surface_blocks`]), a last block of a row or column whole however few of its texels
//!   the texture covers. Over a surface without planes Metal stops a row of blocks wider than the
//!   surface's ("Descriptor rowBytes (264) > IOSurface rowBytes (256)": `RowPitch`) and texels
//!   past the surface's rows, counted in rows of texels ("Descriptor height*arrayLength >
//!   IOSurface height": `PastBuffer`). The runtime lays such a texture out as those rows of blocks
//!   (a shadow image, or an aliasing one where the device's linear image holds the format): its
//!   shadow copies them as the texels they hold (`plane_copy_regions`), and its blits and CPU calls
//!   count a region in blocks (`blit_plan::texel_rows`, `Texels::region_layout`). Over a plane of
//!   a surface with planes it holds a 2D one at an offset and `bytesPerRow` that are multiples of
//!   16, and one of 8-byte blocks (BC1, BC4) at an offset or `bytesPerRow` off 16 that its block's
//!   bytes divide, at least 16 bytes a row ([`blocks_off_sixteen`]): there both GPUs' blits and
//!   samples take block row `j` at the plane's offset + `j` × `bytesPerRow`, unrounded
//!   (`bgraat8.bc1` at offset 8, `bgrarow264.bc1` at 264 bytes a row; BC4 was not measured
//!   there), and so do the runtime's GPU commands and its CPU calls, as the 5300M's `getBytes`
//!   does, where the UHD 630's `getBytes` and `replaceRegion` take the plane at offset 8 from
//!   byte 0. Any other is the runtime's gap ([`BufferRefusal::PlaneShape`]): a BC texture of
//!   16-byte blocks off 16, and one of 8-byte blocks off them or under 16 bytes a row, which no
//!   GPU measured. Two answers of the 5300M's are not reproduced: its `replaceRegion` wrote other
//!   bytes than the texture's past the first row of blocks, and some of its blits into a BC
//!   texture over a surface, and of its `replaceRegion` calls, ended the process. Every other
//!   compressed format over a surface is no buffer's (`NotLinear`).
//! - **Beyond the plane's rows** (the same probe, on those two and the M3 Max under macOS
//!   27.0.1). Over a plane of a surface with planes Metal makes a texture of any type, extent and
//!   levels (`oracle_surftex` 17, 21, 42, 116 to 123, 127, 129), but the three GPUs and their CPU
//!   calls lay out alike only a texture buffer (its texels contiguous from the plane's offset,
//!   however wide: past the plane's row too on the M3 Max and the 5300M, where the UHD 630's
//!   kernels read other bytes), a 1D texture whose row fits and a 2D array of one slice, each
//!   where the 2D texture of its rows lies; the runtime makes those. A texture buffer over a
//!   plane at an offset or `bytesPerRow` off 16 (`bgraat8.texbuf` and `bgrarow264.texbuf`, the
//!   M3 Max) is read by its functions from the offset rounded down to 16 and by `getBytes` and
//!   `replaceRegion` from the offset, as a buffer's texture buffer is: the runtime makes it so.
//!   Plain Metal asserts on a cube and a cube array over a plane on the M3 Max
//!   ("carriesComplexTexture || numFaces == 1", the API validation layer off), which the runtime
//!   refuses as Metal's own ([`BufferTextureProblem::TextureType`]). A 2D texture of more than
//!   one level, a 3D texture and a 2D multisample one lie in no one place on the M3 Max, their
//!   first level and slice among them (`bgra1x2.mips2`, `bgra1x2.mips7`, `420v.p0.mips2`,
//!   `bgra1x2.3d.d2`, `420v.p0.3d.d2`, `bgra1x2.2dms4.private`: its blits and kernels read other
//!   bytes than its blits, `replaceRegion` and `getBytes` put there), where the x86 Macs keep the
//!   first level and slice over the plane: the runtime makes each of a colour format in memory of
//!   its own, zero when made, announced (`Degradation::SurfaceTexelsApart`,
//!   [`apart_from_plane`]). A 2D array of more than one slice keeps its first slice over the plane
//!   on every path of the M3 Max's and lays its second in no one place (`bgra1x2.2darray.a2`,
//!   `bgra1x2.2darray.a2.64x64`: its blits and `getBytes` read it at the plane's offset, over the
//!   first, its kernels one image on, and its writes land at the plane's offset): the runtime
//!   keeps the first slice in place and the others in a tail of the texture's own
//!   ([`slices_apart`], [`Placed::kept`]), announced (`Degradation::SurfaceTexelsTail`), for a
//!   colour format at an offset and `bytesPerRow` that are multiples of 16. A 2D texture whose
//!   rows are wider than the plane's `bytesPerRow`, and a 1D texture wider than one, lie in one
//!   place on every path of the M3 Max's (`bgra1x2.65x64`, `bgra1x2.128x32`, `420v.p0.65x64`,
//!   `420v.p1.33x32`, `bgra1x2.1d.128`): row `y` from the plane's offset plus `y` times
//!   `bytesPerRow`, the whole row, so a 2D texture's rows overlap. The runtime lays them out so
//!   for the GPU and the CPU alike ([`wide_rows`]), for a colour format at an offset and
//!   `bytesPerRow` that are multiples of 16, and holds the 2D texture's rows as every plane's that
//!   overlap (**Rows that overlap**). Every other shape lies differently on each, or was not
//!   measured: [`BufferRefusal::PlaneShape`]
//!   ([`RuntimeGap::SurfacePlaneShape`](crate::RuntimeGap::SurfacePlaneShape), which carries the
//!   measurements). A texture taller than its plane is held as the M3 Max's GPU and CPU lay it
//!   out, its rows past the plane at the plane's pitch, inside the surface; the x86 Macs' GPUs
//!   keep it to the plane (the UHD 630's reads other bytes there and drops the writes, the 5300M's
//!   reads zeros and its blit into it writes nothing), which the runtime does not, as it is given
//!   no plane's height. Rows past the surface's end lie in no one place (the M3 Max under macOS
//!   27.0.1, `probes/surface_texture_envelope.m`'s `bgra1.64x65`, `bgra1.64x128` and
//!   `420v.p1.32x512`: its blits read zeros there, its kernels zeros and, 32 KiB past the plane,
//!   the plane's rows again, its `getBytes` zeros and other memory, and its blits' and
//!   `replaceRegion`'s writes there change no byte of the surface). The runtime keeps those texels
//!   in a tail of the texture's own, zero when made, which every path reads and writes alike
//!   (`buffer_texture.rs`'s **Tails**), announced (`Degradation::SurfaceTexelsTail`); the rows
//!   inside the surface are its bytes, as Metal's are on every path. The shapes it holds no tail
//!   of are refused there ([`BufferRefusal::TexelsPastLength`], [`tail_held`]).
//!
//! The runtime holds both layouts: the GPU's ([`Placed::gpu`]) is the texture's texels for every
//! command, the copies of its shadow, an aliasing image and their hazards; the CPU's
//! ([`Placed::cpu`]) is `replace_region`'s and `get_bytes`'s. Both lie in the buffer, or past it
//! in the texture's tail (a plane of a surface with planes, above), else the texture is refused:
//! over a surface without planes, which plain Metal holds the rows to, `Invalid` (`PastBuffer`).
//! A surface without planes starts at a multiple of 16 in the buffer, as a surface's base does
//! (`OffsetAlignment` otherwise).
//!
//! **Aliasing**. A 2D texture or array over a buffer is a linear image bound to the buffer's memory
//! where the device's linear image has every feature its optimal image has for the texture's uses
//! (it lacks nothing, [`linear_lack`]), and the image the device made, at the pitch it chose
//! itself, sits exactly on Metal's layout and holds no byte but the texels ([`aliases`]). A
//! texture whose rows or slices lie with the buffer's bytes between them, which Metal leaves as the
//! program writes them, never is one, whatever its usage ([`holding`]): its image would hold those
//! bytes too, which Vulkan lets its layout transitions (its first move among them) and a render
//! pass's load and store operations write. Otherwise it is a shadow image the encoders copy from
//! and to the buffer (`shadow.rs`), whose copies address the buffer's texels alone, which is exact
//! for Metal's defined behaviour.

use super::chroma_pairs::is_four_two_two;
use super::copy_plan::block_extent;
use super::image_plan::{block_family, compatible, ImageFacts, ImagePlan};
use super::pixel_format_info::pixel_format_info;
use super::texture_descriptor::TextureLayout;
use super::two_planes;
use super::values::StorageMode;
use metal2vulkan::error::LinearNeed;
use metal2vulkan::metal::format::{is_depth_stencil, ycbcr};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::vk::{
    Format, FormatFeatureFlags2, ImageCreateFlags, ImageUsageFlags, SampleCountFlags,
};

/// The alignment Metal gives every format a buffer can back: of a texture's offset and rows in
/// its buffer (`minimumLinearTextureAlignmentForPixelFormat:`), and of a texture buffer's offset
/// (`minimumTextureBufferAlignmentForPixelFormat:`).
pub(super) const LINEAR_ALIGNMENT: u64 = 16;

/// The alignment of a 2D array's offset in its buffer, and the multiple each of its slices is
/// rounded up to.
pub(super) const ARRAY_ALIGNMENT: u64 = 128;

/// The granularity of a Metal buffer of at least this length: its `allocatedSize` is its length
/// rounded up to it (a shorter buffer's is its length).
pub(super) const METAL_PAGE: u64 = 16384;

/// A texture over a buffer plain Metal itself refuses: it asserts, or answers nil
/// ([`PastAllocation`](Self::PastAllocation)). What only Metal's API validation layer stops is
/// not here (the module's rules).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferTextureProblem {
    /// A pixel format no buffer can back: compressed, depth or stencil, or no format.
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: plain Metal asserts "Linear texture:
    /// cannot create compressed, depth, or stencil textures" over `Depth32Float` and `Stencil8`,
    /// the alignment query asserts "Linear textures do not support depth/stencil pixel formats",
    /// and `BC1_RGBA` answers nil).
    NotLinear {
        /// The descriptor's pixel format.
        pixel_format: PixelFormat,
    },
    /// A texture type the call does not make over a buffer, or over a plane of an IOSurface with
    /// planes: a cube or a cube array there.
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: plain Metal asserts "Linear texture:
    /// must be of type MTLTextureType2D, textureType (MTLTextureType3D) disallowed", and the same
    /// for a 2D array; over a surface's plane it asserts "carriesComplexTexture || numFaces == 1"
    /// in `-[IOGPUMetalTexture initWithDevice:descriptor:iosurface:plane:field:args:argsSize:]`
    /// on a cube and a cube array, the API validation layer off,
    /// `metal2vulkan-conformance/probes/surface_texture_envelope.m`).
    TextureType {
        /// The descriptor's texture type.
        texture_type: TextureType,
    },
    /// More than one mip level, of a texture but a tiled 2D one
    /// (`Buffer::new_tiled_texture`).
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: plain Metal asserts "Linear texture:
    /// cannot be mipmapped, mipmapLevelCount (2) disallowed").
    MipmapLevelCount {
        /// The descriptor's level count.
        count: u32,
    },
    /// A `bytesPerRow` that does not cover the packed row of `row_bytes`.
    ///
    /// Basis: unknown (plain Metal's answer for a `bytesPerRow` below the packed row is unmeasured:
    /// the runs on record vary its alignment, not its length).
    RowPitch {
        /// The `bytesPerRow` given.
        bytes_per_row: u64,
        /// A packed row's bytes.
        row_bytes: u64,
    },
    /// A `bytesPerRow` that is not a multiple of `alignment`.
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: plain Metal asserts "Linear texture:
    /// bytesPerRow (8) must be aligned to 16 bytes", and the same at 17, 24 and 36, where 32 is
    /// made).
    RowPitchAlignment {
        /// The `bytesPerRow` given.
        bytes_per_row: u64,
        /// The alignment it needs, in bytes.
        alignment: u64,
    },
    /// An offset that is not a multiple of `alignment`: a 2D texture's or array's, an IOSurface
    /// plane's, or an SPI's texture buffer's (`newTextureWithDescriptor:offset:bytesPerRow:`'s
    /// texture buffer takes any).
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: plain Metal asserts "Linear texture:
    /// offset (4330536964) must be aligned to 16 bytes" at offsets 4 and 8, where 16 is made).
    OffsetAlignment {
        /// The offset given.
        offset: u64,
        /// The alignment it needs, in bytes.
        alignment: u64,
    },
    /// A texture over a surface without planes whose rows end past the surface's bytes,
    /// `counted` of the `length` the buffer standing for it holds: a descriptor taller than the
    /// surface, which plain Metal asserts on, the API validation layer off (`-[IOGPUMetalTexture
    /// initWithDevice:descriptor:iosurface:plane:field:args:argsSize:]`, "Descriptor
    /// height*arrayLength > IOSurface height": a 16 by 8 and a 16 by 64 descriptor over a 16 by 4
    /// `BGRA` surface, on an M3 Max under macOS 27.0.1, where the 16 by 4 one is made).
    ///
    /// A *plane* of a surface with planes is not held to it: Metal makes a texture whose rows
    /// run past the surface, which the runtime keeps in a tail of the texture's own, or refuses
    /// for a shape it keeps no tail of
    /// ([`RuntimeGap::TexelsPastBufferLength`](crate::RuntimeGap::TexelsPastBufferLength)); and a
    /// texture over a buffer past its bytes Metal makes or answers nil for by the buffer's
    /// allocation alone (the module's **Length** and **Beyond the plane's rows**).
    ///
    /// Basis: metal-measured (an M3 Max, macOS 27.0.1: plain Metal asserts "Descriptor
    /// height*arrayLength > IOSurface height", the API validation layer off, where the surface-high
    /// descriptor is made).
    PastBuffer {
        /// The bytes the rows end at, from the buffer's start.
        counted: u64,
        /// The buffer's length.
        length: u64,
    },
    /// An array whose slices end past the buffer's `allocatedSize`: Metal answers nil (an
    /// [`Error::Nil`](crate::Error::Nil)).
    ///
    /// Basis: metal-measured (an M3 Max, macOS 27.0.1: plain Metal answers nil where the texture's
    /// rounded slices end past the buffer's `allocatedSize`, the layer off, and makes it where they
    /// do not).
    PastAllocation {
        /// Where the last slice ends, in bytes from the buffer's start.
        end: u64,
        /// The buffer's `allocatedSize`.
        allocation: u64,
    },
}

/// Whether a buffer can back a texture of `pf` (the module's rule), of a format Metal describes:
/// not `Invalid`, compressed, depth or stencil.
pub(super) fn is_linear(pf: PixelFormat) -> bool {
    use PixelFormat as P;
    let compressed = block_extent(pf).is_some() || matches!(pf, P::GBGR422 | P::BGRG422);
    pf != P::Invalid && !compressed && !is_depth_stencil(pf)
}

/// Whether `pf` is of the BC family (BC1 to BC7, their sRGB and signed forms), whose textures
/// Metal makes over an IOSurface plane's rows, a row of blocks to the plane's `bytesPerRow` (the
/// module's **Block-compressed surfaces**), where no other buffer backs one.
pub(super) fn surface_blocks(pf: PixelFormat) -> bool {
    use PixelFormat as P;
    matches!(
        block_family(pf),
        P::BC1RGBA
            | P::BC2RGBA
            | P::BC3RGBA
            | P::BC4RUnorm
            | P::BC5RGUnorm
            | P::BC6HRGBUfloat
            | P::BC7RGBAUnorm
    )
}

/// Whether a buffer can back a texture of `pf` ([`is_linear`]), or why not.
fn backed(pf: PixelFormat) -> Result<(), BufferTextureProblem> {
    if is_linear(pf) {
        Ok(())
    } else {
        Err(BufferTextureProblem::NotLinear { pixel_format: pf })
    }
}

/// `pf`'s alignment in a buffer ([`LINEAR_ALIGNMENT`]), or why no buffer can back it: also a
/// format Metal describes none of (`pixel_format_info::describes`), whose alignment queries Metal
/// asserts on, and one it describes as a depth or a stencil format, `AnyDepth` and `AnyStencil`
/// among them, which Vulkan has no format for ("Linear textures do not support depth/stencil pixel
/// formats").
pub(super) fn linear_alignment(pf: PixelFormat) -> Result<u64, BufferTextureProblem> {
    let info = pixel_format_info(u64::from(pf.as_raw()));
    if !info.describes_a_format() || info.depth_or_stencil() {
        return Err(BufferTextureProblem::NotLinear { pixel_format: pf });
    }
    backed(pf).map(|()| LINEAR_ALIGNMENT)
}

/// The Metal call that makes a texture over a buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BufferTextureCall {
    /// `newTextureWithDescriptor:offset:bytesPerRow:`: a 2D texture or a texture buffer.
    Texture,
    /// The SPI `newTiledTextureWithDescriptor:offset:bytesPerRow:`: as
    /// [`Texture`](Self::Texture), and a 2D texture of more than one level, its first over the
    /// buffer's rows.
    Tiled,
    /// The SPI `newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:`, which makes
    /// 2D arrays too. Metal lays an array's slices out itself, whatever `bytesPerImage` is.
    Linear,
    /// `newTextureWithDescriptor:iosurface:plane:` over the caller's memory, of a plane of a
    /// planar surface (`planar`) or of a surface without planes (the module's rules).
    IoSurfacePlane { planar: bool },
}

/// A texture over a buffer as a call asks for it.
#[derive(Clone, Copy, Debug)]
pub(super) struct BufferTexture {
    /// The descriptor's checked layout.
    pub(super) layout: TextureLayout,
    /// The bytes of one texel of the pixel format.
    pub(super) texel_bytes: u64,
    /// The descriptor's storage mode, which only an IOSurface plane is held to.
    pub(super) storage: StorageMode,
    pub(super) offset: u64,
    pub(super) bytes_per_row: u64,
    pub(super) call: BufferTextureCall,
    /// The descriptor's usage, which only an IOSurface plane whose GPU rows overlap is held to:
    /// what Metal's render pass leaves of the bytes two rows share is not measured
    /// ([`overlap_held`]).
    pub(super) usage: TextureUsage,
}

/// The buffer a texture is made over: its length.
#[derive(Clone, Copy, Debug)]
pub(super) struct BufferFacts {
    pub(super) length: u64,
}

/// Where a texture over a buffer lies in it, as Metal lays it out: `slices` of `rows` rows, each
/// row's `row_bytes` of texels starting `row_pitch` after the one above, and each slice
/// `slice_pitch` after the one before.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BufferLayout {
    /// The texture's first byte in the buffer.
    pub(super) offset: u64,
    pub(super) row_pitch: u64,
    pub(super) slice_pitch: u64,
    pub(super) row_bytes: u64,
    pub(super) rows: u64,
    pub(super) slices: u64,
}

impl BufferLayout {
    /// The bytes consecutive rows share, where the rows overlap: a row's bytes past the pitch, 0
    /// where each row lies at or past the end of the one above. Only an IOSurface plane's GPU
    /// layout overlaps, and a heap's one-plane YCbCr texture's rows (`texture/heap.rs`,
    /// `one_plane_pitch`): at a pitch rounded down to 16 ([`iosurface_gpu_layout`]) by 1 to 15
    /// bytes, a whole number of texels, between consecutive rows alone; and where a plane's rows
    /// are wider than its pitch ([`wide_rows`]), its CPU layout too, by any whole number of
    /// texels, a byte shared by as many rows as reach it (`tests::linear_plan`).
    pub(super) fn shared_bytes(&self) -> u64 {
        match self.rows > 1 {
            true => self.row_bytes.saturating_sub(self.row_pitch),
            false => 0,
        }
    }

    /// Whether its rows, or its slices, lie with bytes between them that are not texels.
    pub(super) fn padded(&self) -> bool {
        apart(self) != (false, false)
    }

    /// One past the texture's last texel byte in the buffer.
    pub(super) fn end(&self) -> u64 {
        self.offset
            .saturating_add(self.slice_pitch.saturating_mul(self.slices - 1))
            .saturating_add(self.row_pitch.saturating_mul(self.rows - 1))
            .saturating_add(self.row_bytes)
    }
}

/// Where a texture over a buffer lies in it: its texels for the GPU's commands (`gpu`: copies,
/// views, aliases and their hazards) and for the CPU's `replaceRegion` and `getBytes` (`cpu`),
/// which differ only for an IOSurface plane (the module's rules).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Placed {
    pub(super) gpu: BufferLayout,
    pub(super) cpu: BufferLayout,
    /// An IOSurface plane of more than one row under 16 bytes a row, whose rows past the first
    /// Metal's GPU reads as memory-dependent bytes: the runtime's reads them where its writes put
    /// them (the module's rules).
    pub(super) narrow: bool,
    /// Where the texels' bytes in place end, for a 2D array of more than one slice over a plane
    /// of a surface with planes ([`slices_apart`]): its second slice's first byte, or the
    /// buffer's length where that comes first. Its bytes from here on, its slices past the first,
    /// lie in a tail of the texture's own (`buffer_texture.rs`'s **Tails**). `None` for every
    /// other texture, whose texels lie in place up to the buffer's length.
    pub(super) kept: Option<u64>,
}

impl Placed {
    /// One layout for the GPU and the CPU alike: every texture over a buffer but an IOSurface
    /// plane.
    fn same(layout: BufferLayout) -> Placed {
        Placed {
            gpu: layout,
            cpu: layout,
            narrow: false,
            kept: None,
        }
    }
}

/// Whether Metal's GPU reads and writes two of a texture's rows over bytes they share, its GPU
/// layout `gpu` ([`Placed::gpu`]): an IOSurface plane's that it takes at its pitch rounded down
/// to 16 ([`iosurface_gpu_layout`]) while the plane's own rows, which the CPU takes, lie a row
/// apart or further (the module's **Rows that overlap**), and a plane's whose rows are wider than
/// its `bytesPerRow`, which every path takes that far apart ([`wide_rows`]).
///
/// Only a plane answers true. A rotated texture's two layouts are one
/// (`buffer_texture::rotated_rows`) and share no byte at all, its `bytesPerRow` covering its own
/// row (`buffer_texture::covers_its_row`); and the only plan whose CPU rows share bytes is a wide
/// plane's, whose GPU rows are the CPU's, every other call but a two-plane YCbCr texture's
/// holding `bytesPerRow` to the row ([`plan_buffer_texture`]) and that one's GPU rows, which
/// round down to 16 or below the CPU's, sharing bytes too and so being refused
/// ([`BufferRefusal::OverlappingRows`]).
pub(super) fn plane_rows_share_bytes(gpu: &BufferLayout) -> bool {
    gpu.shared_bytes() > 0
}

/// Where Metal's GPU reads and writes the texels of an IOSurface plane laid out as `cpu` for the
/// CPU (the module's rules): from its offset rounded down to 16, rows its pitch rounded down to 16
/// apart, or 16 apart where that is 0 (a plane under 16 bytes a row, [`Placed::narrow`]).
///
/// Rows that overlap lie at that pitch itself, which the texture's shadow holds in two copy
/// regions each way (the module's **Rows that overlap**), a plane's whose rows are wider than its
/// pitch among them ([`wide_rows`], where that is the CPU's pitch too); every other plane's lie
/// at least a row apart, so that one region a slice states them and its `bufferRowLength` covers
/// its width. A plane of one row lies a row apart whatever its pitch: it overlaps nothing, and
/// one region states it.
pub(super) fn iosurface_gpu_layout(cpu: &BufferLayout) -> BufferLayout {
    let down = |n: u64| n - n % LINEAR_ALIGNMENT;
    let pitch = down(cpu.row_pitch);
    // Every texel size divides a multiple of 16, as the shadow's copies need.
    let row_pitch = match cpu.rows > 1 && pitch != 0 && pitch < cpu.row_bytes {
        true => pitch,
        false => pitch.max(
            cpu.row_bytes
                .checked_next_multiple_of(LINEAR_ALIGNMENT)
                .unwrap_or(u64::MAX),
        ),
    };
    BufferLayout {
        offset: down(cpu.offset),
        row_pitch,
        slice_pitch: row_pitch.saturating_mul(cpu.rows),
        ..*cpu
    }
}

/// Whether the runtime holds an IOSurface plane of `texture` where its GPU rows overlap, its
/// shadow's two copy regions each way reproducing the sharing (the module's **Rows that
/// overlap**), or refuses it ([`BufferRefusal::OverlappingRows`]).
///
/// A plain plane is held. Refused are a plane a render pass stores into, where what Metal leaves
/// of the bytes two rows share is not measured, and the shapes whose rows of codes or units no
/// row rule bounds, which two regions do not express: a two-plane YCbCr texture's (`ycbcr.rs`), a
/// two-plane extended-range one's (`texture/surface_two_planes.rs`), a 4:2:2 one's pairs
/// (`chroma_pairs.rs`) and a depth or stencil plane's elements (`texture/surface_depth.rs`). A
/// rotated plane's `bytesPerRow` is already a multiple of 16, so its GPU rows never overlap
/// (`buffer_texture::turned_plane`).
fn overlap_held(texture: &BufferTexture) -> bool {
    let pf = texture.layout.pixel_format;
    !texture.usage.contains(TextureUsage::RENDER_TARGET)
        && !ycbcr(pf)
        && !two_planes::holds_planes(pf)
        && !is_four_two_two(pf)
        && !is_depth_stencil(pf)
}

/// Whether `texture` is a 2D or 1D texture over a plane of a surface with planes whose rows are
/// wider than the plane's `bytesPerRow` that the runtime makes, a 2D texture's rows `bytesPerRow`
/// apart and overlapping, on the GPU and the CPU alike, and a 1D texture's one row from the
/// plane's offset (the module's **Beyond the plane's rows**): of a colour format a buffer backs
/// ([`is_linear`]) but a two-plane extended-range one, over a plane whose offset and
/// `bytesPerRow` are multiples of 16, where the GPU's rows lie where the CPU's do. Metal makes
/// such a texture, and every path of the M3 Max's reads row `y` from the plane's offset plus `y`
/// times `bytesPerRow`, the whole row (an Apple M3 Max under macOS 27.0.1,
/// `probes/surface_texture_envelope.m`'s `bgra1x2.65x64`, `bgra1x2.128x32`, `420v.p0.65x64`,
/// `420v.p1.33x32` and `bgra1x2.1d.128`), while its writes of the bytes two rows share land in no
/// one order (the module's **Rows that overlap**). A plane off 16, whose GPU rows would lie at
/// another pitch or offset than the CPU's, is not measured.
fn wide_rows(texture: &BufferTexture) -> bool {
    let layout = &texture.layout;
    let planar = matches!(
        texture.call,
        BufferTextureCall::IoSurfacePlane { planar: true }
    );
    let aligned = texture.offset.is_multiple_of(LINEAR_ALIGNMENT)
        && texture.bytes_per_row.is_multiple_of(LINEAR_ALIGNMENT)
        && texture.bytes_per_row != 0;
    let pf = layout.pixel_format;
    planar
        && matches!(
            layout.texture_type,
            TextureType::Type2D | TextureType::Type1D
        )
        && aligned
        && is_linear(pf)
        && !two_planes::holds_planes(pf)
}

/// Whether `texture` is a BC texture of 8-byte blocks (BC1, BC4) over a plane of a surface with
/// planes at an offset or `bytesPerRow` off 16 that the runtime makes, at the plane's own offset
/// and `bytesPerRow` for the GPU and the CPU alike (the module's **Block-compressed surfaces**):
/// one whose block's bytes divide both, its rows at least 16 bytes apart. The x86 Macs make
/// such a texture, their blits and samples taking block row `j` at the plane's offset plus `j`
/// times `bytesPerRow`, unrounded (an AMD Radeon Pro 5300M and an Intel UHD 630 under macOS
/// 26.7.1, `probes/surface_texture_envelope.m`'s `bgraat8.bc1` and `bgrarow264.bc1`; the M3 Max
/// answers nil for every BC texture). A BC texture of 16-byte blocks off 16, and one under 16
/// bytes a row, are not measured.
fn blocks_off_sixteen(texture: &BufferTexture) -> bool {
    let planar = matches!(
        texture.call,
        BufferTextureCall::IoSurfacePlane { planar: true }
    );
    let block = texture.texel_bytes;
    planar
        && surface_blocks(texture.layout.pixel_format)
        && block == 8
        && texture.offset.is_multiple_of(block)
        && texture.bytes_per_row.is_multiple_of(block)
        && texture.bytes_per_row >= LINEAR_ALIGNMENT
}

/// Whether the runtime keeps the texels of an IOSurface plane of `texture` that lie past the
/// buffer standing for its surface in a tail of the texture's own, zero when made (the module's
/// **Beyond the plane's rows**), or refuses them ([`BufferRefusal::TexelsPastLength`]).
///
/// A colour plane's are kept, a 2D texture's, a 1D texture's, an array's and a texture buffer's:
/// the shadow's copies, its blits and its CPU calls reach the tail as they reach a
/// buffer's texture's (`buffer_texture.rs`'s **Tails**). Refused are the shapes whose texels
/// another form holds than those copies of their rows: a two-plane YCbCr texture's codes, which
/// a conversion reads (`ycbcr.rs`); a two-plane extended-range one's, its alpha in the surface's
/// other plane (`texture/surface_two_planes.rs`); a 4:2:2 one's pairs (`chroma_pairs.rs`); a
/// depth or stencil plane's elements, which a render pass stores in Metal's tiled layout
/// (`texture/surface_depth.rs`); and a BC texture's blocks, whose rows past a surface no GPU of
/// Metal's that makes them was measured at.
fn tail_held(texture: &BufferTexture) -> bool {
    let pf = texture.layout.pixel_format;
    !ycbcr(pf)
        && !two_planes::holds_planes(pf)
        && !is_four_two_two(pf)
        && !is_depth_stencil(pf)
        && !surface_blocks(pf)
}

/// Whether the runtime makes the texture of an IOSurface plane of `texture` in memory of its own,
/// zero when made, rather than over the plane (the module's **Beyond the plane's rows**): a colour
/// 2D texture of more than one level, a 3D texture or a 2D multisample one over a plane of a
/// surface with planes. Metal makes each and lays its texels out in no one place, its first
/// level's and slice's among them (an Apple M3 Max under macOS 27.0.1,
/// `probes/surface_texture_envelope.m`'s `bgra1x2.mips2`, `bgra1x2.mips7`, `420v.p0.mips2`,
/// `bgra1x2.3d.d2`, `420v.p0.3d.d2` and `bgra1x2.2dms4.private`): its GPU reads them from other
/// bytes than its writes and its CPU put them. A colour format is one a buffer backs
/// ([`is_linear`]) and that keeps a tail past a surface ([`tail_held`]); every other format's
/// such texture is planned, and refused, as before.
pub(super) fn apart_from_plane(texture: &BufferTexture) -> bool {
    use TextureType as T;
    let planar = matches!(
        texture.call,
        BufferTextureCall::IoSurfacePlane { planar: true }
    );
    let layout = &texture.layout;
    let shape = match layout.texture_type {
        T::Type2D => layout.levels > 1,
        T::Type3D | T::Type2DMultisample => true,
        _ => false,
    };
    planar && shape && is_linear(layout.pixel_format) && tail_held(texture)
}

/// Whether `texture` is a 2D array of more than one slice over a plane of a surface with planes
/// that the runtime makes with its first slice over the plane and its others in a tail of the
/// texture's own, zero when made (the module's **Beyond the plane's rows**, [`Placed::kept`]):
/// one of a colour format ([`is_linear`], [`tail_held`]) over a plane whose offset and
/// `bytesPerRow` are multiples of 16, where the GPU's and the CPU's rows lie alike. Metal makes
/// such an array and lays its first slice over the plane on every path, and its second in no one
/// place: its blits and `getBytes` read the second at the plane's offset, over the first, and its
/// kernels one image on, and its blits' and `replaceRegion`'s writes of it land at the plane's
/// offset (an Apple M3 Max under macOS 27.0.1, `probes/surface_texture_envelope.m`'s
/// `bgra1x2.2darray.a2` and `bgra1x2.2darray.a2.64x64`). An array over a plane off 16 is not
/// measured, and every other format's array is planned, and refused, as before.
pub(super) fn slices_apart(texture: &BufferTexture) -> bool {
    let layout = &texture.layout;
    let planar = matches!(
        texture.call,
        BufferTextureCall::IoSurfacePlane { planar: true }
    );
    let aligned = texture.offset.is_multiple_of(LINEAR_ALIGNMENT)
        && texture.bytes_per_row.is_multiple_of(LINEAR_ALIGNMENT);
    planar
        && layout.texture_type == TextureType::Type2DArray
        && layout.layers > 1
        && aligned
        && is_linear(layout.pixel_format)
        && tail_held(texture)
}

/// Why a texture over a buffer is not made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BufferRefusal {
    /// What Metal asserts on or stops under validation (`Invalid`), or answers nil for (`Nil`:
    /// [`BufferTextureProblem::PastAllocation`]).
    Metal(BufferTextureProblem),
    /// An IOSurface plane's GPU or CPU rows past the buffer standing for its surface, which
    /// Metal makes over bytes past the length, of a shape the runtime keeps no tail of
    /// ([`tail_held`]: `NotImplemented`). A colour plane keeps those texels in a tail of its own
    /// instead, as a 2D texture, an array or a texture buffer over a buffer does (the module's
    /// **Length**).
    TexelsPastLength,
    /// An IOSurface plane whose rows overlap on Metal's GPU that the runtime holds no form of
    /// (`NotImplemented`; the module's **Rows that overlap**): a plane a render pass stores into,
    /// or one of a format whose rows of codes or units no row rule bounds (a two-plane YCbCr
    /// texture's, which overlap by any amount across any number of rows, `ycbcr.rs`; a two-plane
    /// extended-range one's, a 4:2:2 one's pairs, a depth or stencil plane's elements). A plain
    /// plane, whose pitch covers its row and whose rows share 1 to 15 bytes, is a shadow of two
    /// copy regions each way instead.
    OverlappingRows,
    /// An IOSurface plane whose descriptor asks for `Memoryless` storage, which plain Metal makes
    /// and serves on the CPU while every GPU use fails the command buffer (`NotImplemented`; the
    /// module's rules).
    MemorylessPlane,
    /// A texture over a plane of a surface with planes that Metal makes and lays out no one way
    /// across its GPUs and its CPU calls (`NotImplemented`; the module's **Beyond the plane's
    /// rows**), and that the runtime holds neither in memory of its own ([`apart_from_plane`],
    /// which takes a colour 2D texture of several levels, a 3D and a multisample one before any
    /// plan is asked) nor with slices in a tail ([`slices_apart`]) nor at overlapping rows
    /// ([`wide_rows`]): rows wider than the plane's `bytesPerRow`, or a 1D texture wider than the
    /// plane's row, of a format but a colour one or over a plane off 16 bytes; more than one
    /// level, a type but 2D, a texture buffer, a 1D texture and a 2D array (a cube and a cube
    /// array, which plain Metal asserts on, are [`BufferTextureProblem::TextureType`]); and a
    /// block-compressed texture over a plane whose offset or `bytesPerRow` is off 16 bytes but
    /// the 8-byte blocks [`blocks_off_sixteen`] takes.
    PlaneShape,
}

impl From<BufferTextureProblem> for BufferRefusal {
    fn from(problem: BufferTextureProblem) -> BufferRefusal {
        BufferRefusal::Metal(problem)
    }
}

/// Where `texture` lies in `buffer`, by the module's rules, or why it is not made.
pub(super) fn plan_buffer_texture(
    texture: &BufferTexture,
    buffer: &BufferFacts,
) -> Result<Placed, BufferRefusal> {
    use BufferTextureProblem as P;
    use TextureType as T;
    let layout = &texture.layout;
    // A surface's 4:2:2 plane is made (the module docs).
    let surface = matches!(texture.call, BufferTextureCall::IoSurfacePlane { .. });
    // A format Metal describes none of was stopped by the descriptor's check before
    // (`TextureDescriptor::check_made`).
    // Over a plane of a surface with planes Metal makes a depth or stencil texture, laid out as
    // a colour texture's in its one aspect's units (the module docs).
    let planar = matches!(
        texture.call,
        BufferTextureCall::IoSurfacePlane { planar: true }
    );
    let planar_depth = planar && is_depth_stencil(layout.pixel_format);
    // Over a surface's rows Metal makes a BC texture, its blocks a row of blocks a row (the
    // module's **Block-compressed surfaces**).
    let blocks = surface && surface_blocks(layout.pixel_format);
    if !(surface && is_four_two_two(layout.pixel_format)) && !planar_depth && !blocks {
        backed(layout.pixel_format)?;
    }
    let texture_type = layout.texture_type;
    // Over a plane of a surface with planes Metal makes every type, of any levels (the module's
    // **Beyond the plane's rows**): the runtime holds the ones its texels lie as a 2D texture's
    // in.
    let one_slice = texture_type == T::Type2DArray && layout.layers == 1;
    // A 2D array of more than one slice over a plane keeps its first slice in place and its
    // others in its tail (the module's **Beyond the plane's rows**, [`slices_apart`]).
    let sliced = slices_apart(texture);
    let made = match texture.call {
        BufferTextureCall::Texture | BufferTextureCall::Tiled => {
            matches!(texture_type, T::Type2D | T::TextureBuffer)
        }
        BufferTextureCall::Linear => {
            matches!(texture_type, T::Type2D | T::TextureBuffer | T::Type2DArray)
        }
        BufferTextureCall::IoSurfacePlane { planar: false } => {
            matches!(texture_type, T::Type2D | T::TextureBuffer)
        }
        BufferTextureCall::IoSurfacePlane { planar: true } => {
            matches!(texture_type, T::Type2D | T::TextureBuffer | T::Type1D) || one_slice || sliced
        }
    };
    // Over a plane of a surface with planes plain Metal asserts on a texture of more than one
    // face, the API validation layer off (an Apple M3 Max under macOS 27.0.1,
    // `probes/surface_texture_envelope.m`'s `bgra1x2.cube.16` and `bgra1x2.cubearray.16`):
    // Metal's own refusal, where the x86 Macs make a cube.
    if planar && matches!(texture_type, T::Cube | T::CubeArray) {
        return Err(P::TextureType { texture_type }.into());
    }
    if !made && planar {
        return Err(BufferRefusal::PlaneShape);
    }
    if !made {
        return Err(P::TextureType { texture_type }.into());
    }
    // A BC texture over a plane is measured 2D alone.
    if blocks && texture_type != T::Type2D {
        return Err(BufferRefusal::PlaneShape);
    }
    if layout.levels != 1 && planar {
        return Err(BufferRefusal::PlaneShape);
    }
    // A tiled 2D texture takes its levels, its first over the buffer's rows (the module's
    // **Types**).
    let tiled = texture.call == BufferTextureCall::Tiled && texture_type == T::Type2D;
    if layout.levels != 1 && !tiled {
        return Err(P::MipmapLevelCount {
            count: layout.levels,
        }
        .into());
    }
    if surface && texture.storage == StorageMode::Memoryless {
        return Err(BufferRefusal::MemorylessPlane);
    }
    let [width, height, _] = layout.extent.map(u64::from);
    let (offset, bytes_per_row) = (texture.offset, texture.bytes_per_row);
    // A block-compressed texture's rows are rows of its blocks, `texel_bytes` each.
    let [across, down] = match blocks {
        true => units(layout.pixel_format, [width, height]),
        false => [width, height],
    };
    let row_bytes = across.saturating_mul(texture.texel_bytes);
    // A two-plane YCbCr texture reads its codes at the surface plane's rows whatever their width
    // (the module docs): Metal checks no row against them.
    let ycbcr = surface && ycbcr(layout.pixel_format);
    // A texture buffer over a plane of a surface with planes is its bytes from the plane's
    // offset on, however wide, and a 2D or 1D texture's rows wider than the plane's
    // `bytesPerRow` lie that far apart, a 2D texture's overlapping ([`wide_rows`]; the module's
    // **Beyond the plane's rows**); one over a surface without planes Metal holds to its row.
    let texels_run = planar && texture_type == T::TextureBuffer;
    if bytes_per_row < row_bytes && !ycbcr && !texels_run && !wide_rows(texture) {
        return Err(match planar {
            true => BufferRefusal::PlaneShape,
            false => P::RowPitch {
                bytes_per_row,
                row_bytes,
            }
            .into(),
        });
    }
    if let BufferTextureCall::IoSurfacePlane { planar } = texture.call {
        // A BC texture lies at the plane's own offset and `bytesPerRow` on the GPU and the CPU
        // alike: over a plane at multiples of 16, and one of 8-byte blocks at multiples of its
        // block off 16 ([`blocks_off_sixteen`]; the module's **Block-compressed surfaces**). A
        // texture buffer's functions read a plane from its offset rounded down to 16 and its CPU
        // calls from the offset, as a buffer's texture buffer's, at any pitch (the module's
        // **Beyond the plane's rows**).
        let aligned = offset.is_multiple_of(LINEAR_ALIGNMENT)
            && bytes_per_row.is_multiple_of(LINEAR_ALIGNMENT);
        if planar && blocks && !aligned && !blocks_off_sixteen(texture) {
            return Err(BufferRefusal::PlaneShape);
        }
        let cpu = BufferLayout {
            offset,
            row_pitch: bytes_per_row,
            slice_pitch: bytes_per_row.saturating_mul(down),
            row_bytes,
            rows: down,
            slices: match sliced {
                true => u64::from(layout.layers),
                false => 1,
            },
        };
        // Over a surface without planes Metal holds a BC texture's rows of texels, not of
        // blocks, to the surface's ("Descriptor height*arrayLength > IOSurface height").
        if blocks && !planar {
            let counted = (height - 1)
                .saturating_mul(bytes_per_row)
                .saturating_add(row_bytes)
                .saturating_add(offset);
            if counted > buffer.length {
                return Err(P::PastBuffer {
                    counted,
                    length: buffer.length,
                }
                .into());
            }
        }
        let held = (overlap_held(texture), tail_held(texture));
        let placed = plan_iosurface_plane(cpu, (planar, blocks), buffer.length, held)?;
        // A 2D array's slices past its first lie in its tail, from its second slice's first
        // byte on (one image on from the plane's offset, where Metal's kernels read it).
        let kept = sliced.then(|| (cpu.offset.saturating_add(cpu.slice_pitch)).min(buffer.length));
        return Ok(Placed { kept, ..placed });
    }
    let array = texture_type == T::Type2DArray;
    let alignment = if array {
        ARRAY_ALIGNMENT
    } else {
        LINEAR_ALIGNMENT
    };
    let any_offset = texture_type == T::TextureBuffer && texture.call == BufferTextureCall::Texture;
    if !any_offset && !offset.is_multiple_of(alignment) {
        return Err(P::OffsetAlignment { offset, alignment }.into());
    }
    if texture_type != T::TextureBuffer && !bytes_per_row.is_multiple_of(LINEAR_ALIGNMENT) {
        return Err(P::RowPitchAlignment {
            bytes_per_row,
            alignment: LINEAR_ALIGNMENT,
        }
        .into());
    }
    let slices = u64::from(layout.layers);
    let image = bytes_per_row.saturating_mul(height);
    let slice_pitch = if array {
        image
            .checked_next_multiple_of(ARRAY_ALIGNMENT)
            .unwrap_or(u64::MAX)
    } else {
        image
    };
    let placed = BufferLayout {
        offset,
        row_pitch: bytes_per_row,
        slice_pitch,
        row_bytes,
        rows: height,
        slices,
    };
    let length = buffer.length;
    // Metal's validation counts the slices' rows as if contiguous against the length and stops
    // the texture there; plain Metal reads the buffer's allocation alone (the module's
    // **Length**), so the runtime does too. A 2D texture, an array or a texture buffer whose
    // texels run past the length keeps those texels in its tail (`buffer_texture.rs`).
    if placed.end() > length {
        let end = offset.saturating_add(slice_pitch.saturating_mul(slices));
        if length >= METAL_PAGE {
            let allocation = length
                .checked_next_multiple_of(METAL_PAGE)
                .unwrap_or(length);
            if end > allocation {
                return Err(P::PastAllocation { end, allocation }.into());
            }
        }
    }
    Ok(Placed::same(placed))
}

/// The blocks of a block-compressed `pf` a texture of `extent` texels spans across and down,
/// each side rounded up to whole blocks.
fn units(pf: PixelFormat, [width, height]: [u64; 2]) -> [u64; 2] {
    let [w, h] = block_extent(pf).map_or([1, 1], |extent| extent.map(u64::from));
    [width.div_ceil(w), height.div_ceil(h)]
}

/// An IOSurface plane laid out as `cpu` in a buffer of `length` bytes, of a planar surface or
/// not (`planar`), by the module's rules; its type, levels, options and row were checked as
/// every texture over a buffer's are. `blocks` says whether it is a BC texture's rows of blocks,
/// which the GPU takes where the CPU does (the module's **Block-compressed surfaces**), rather
/// than at [`iosurface_gpu_layout`]. `held` says whether the runtime holds the plane where its
/// GPU rows overlap ([`overlap_held`]), or refuses it there; `tail` whether it keeps the plane's
/// texels past the buffer in a tail of the texture's own ([`tail_held`]), or refuses them.
fn plan_iosurface_plane(
    cpu: BufferLayout,
    (planar, blocks): (bool, bool),
    length: u64,
    (held, tail): (bool, bool),
) -> Result<Placed, BufferRefusal> {
    use BufferTextureProblem as P;
    let alignment = LINEAR_ALIGNMENT;
    if !planar && !cpu.row_pitch.is_multiple_of(alignment) {
        return Err(P::RowPitchAlignment {
            bytes_per_row: cpu.row_pitch,
            alignment,
        }
        .into());
    }
    if !planar && !cpu.offset.is_multiple_of(alignment) {
        return Err(P::OffsetAlignment {
            offset: cpu.offset,
            alignment,
        }
        .into());
    }
    // A surface without planes holds the texture's rows: plain Metal asserts on a descriptor
    // taller than it (`P::PastBuffer`).
    if cpu.end() > length && !planar {
        return Err(P::PastBuffer {
            counted: cpu.end(),
            length,
        }
        .into());
    }
    let gpu = match blocks {
        true => cpu,
        false => iosurface_gpu_layout(&cpu),
    };
    if gpu.shared_bytes() != 0 && !held {
        return Err(BufferRefusal::OverlappingRows);
    }
    // Over a plane Metal makes a texture whose rows run past the surface and lays those texels
    // in no one place (the module's **Beyond the plane's rows**): the runtime keeps them in a
    // tail of the texture's own (`buffer_texture.rs`), or refuses the shapes it holds no tail of.
    if gpu.end().max(cpu.end()) > length && !tail {
        return Err(BufferRefusal::TexelsPastLength);
    }
    Ok(Placed {
        gpu,
        cpu,
        narrow: !blocks && cpu.rows > 1 && cpu.row_pitch < alignment,
        kept: None,
    })
}

/// The features of an image's tiling each of its usages relies on: what the runtime's views,
/// samplers, attachments and copies of the texture may use.
const USAGE_FEATURES: [(ImageUsageFlags, FormatFeatureFlags2); 6] = [
    (
        ImageUsageFlags::SAMPLED,
        FormatFeatureFlags2::SAMPLED_IMAGE
            .union(FormatFeatureFlags2::SAMPLED_IMAGE_FILTER_LINEAR)
            .union(FormatFeatureFlags2::SAMPLED_IMAGE_FILTER_MINMAX),
    ),
    (
        ImageUsageFlags::STORAGE,
        FormatFeatureFlags2::STORAGE_IMAGE
            .union(FormatFeatureFlags2::STORAGE_IMAGE_ATOMIC)
            .union(FormatFeatureFlags2::STORAGE_READ_WITHOUT_FORMAT)
            .union(FormatFeatureFlags2::STORAGE_WRITE_WITHOUT_FORMAT),
    ),
    (
        ImageUsageFlags::COLOR_ATTACHMENT,
        FormatFeatureFlags2::COLOR_ATTACHMENT.union(FormatFeatureFlags2::COLOR_ATTACHMENT_BLEND),
    ),
    (
        ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        FormatFeatureFlags2::DEPTH_STENCIL_ATTACHMENT,
    ),
    (
        ImageUsageFlags::TRANSFER_SRC,
        FormatFeatureFlags2::TRANSFER_SRC,
    ),
    (
        ImageUsageFlags::TRANSFER_DST,
        FormatFeatureFlags2::TRANSFER_DST,
    ),
];

/// The features of an image's tiling a texture of several mip levels relies on beyond its
/// usages': `generate_mipmaps` blits each level from the one above, linearly filtered, where the
/// format has them (`mipmap_plan.rs`; the compute filter elsewhere).
const MIPMAP_FEATURES: FormatFeatureFlags2 = FormatFeatureFlags2::BLIT_SRC
    .union(FormatFeatureFlags2::BLIT_DST)
    .union(FormatFeatureFlags2::SAMPLED_IMAGE_FILTER_LINEAR);

/// What a linear image of a plan lacks of all its optimal image can do ([`linear_lack`]):
/// `features` of `format`'s linear tiling, which its optimal tiling has and the plan relies on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct LinearLack {
    pub(super) format: Format,
    pub(super) features: FormatFeatureFlags2,
}

/// What a linear image of `plan` lacks of all its optimal image can do for the plan's usages:
/// for the image's own format and each format it is viewed as ([`viewed_formats`]), the linear
/// tiling must have every feature of [`USAGE_FEATURES`] the optimal tiling has, and of
/// [`MIPMAP_FEATURES`] for a plan of several levels. A plan of extended usage (a usage its own
/// format lacks, which a view format has) is checked alike: in each format, the optimal tiling has
/// none of the features of a usage that format does not take, so only those it takes are asked of
/// its linear tiling. `None` where the linear image lacks nothing.
pub(super) fn linear_lack(plan: &ImagePlan, facts: &ImageFacts) -> Option<LinearLack> {
    tiling_lack(plan, facts, |format| facts.linear(format))
}

/// What an image of `plan` in another tiling than optimal, whose features in a format are
/// `tiling`'s answer, lacks of all its optimal image can do, as [`linear_lack`] asks it of linear
/// tiling: a `DRM_FORMAT_MOD_LINEAR` image's features are its modifier's (`dma_buf.rs`). `None`
/// where it lacks nothing.
pub(super) fn tiling_lack(
    plan: &ImagePlan,
    facts: &ImageFacts,
    tiling: impl Fn(Format) -> FormatFeatureFlags2,
) -> Option<LinearLack> {
    let needed = USAGE_FEATURES
        .iter()
        .filter(|(usage, _)| plan.usage.contains(*usage))
        .fold(FormatFeatureFlags2::empty(), |needed, &(_, features)| {
            needed.union(features)
        })
        .union(if plan.levels > 1 {
            MIPMAP_FEATURES
        } else {
            FormatFeatureFlags2::empty()
        });
    std::iter::once(plan.format)
        .chain(viewed_formats(plan, facts))
        .find_map(|format| {
            let (optimal, linear) = (facts.optimal(format), tiling(format));
            let features = optimal.intersection(needed).difference(linear);
            (!features.is_empty()).then_some(LinearLack { format, features })
        })
}

/// The formats a plan's image may be viewed as: its list, or, where its views may take any
/// format (a texture of `pixelFormatView` usage: `MUTABLE_FORMAT` without a list), each format
/// the runtime views a texture of a pixel format as for the plan's usages (`view_formats`) that
/// is compatible with the image's, of every pixel format: the device's own table of those
/// (`ImageFacts::any_views`), in `Format` order. Read in place, with no list made: a heap
/// texture's plan asks them at each texture whose image is linear (`texture/heap.rs`).
pub(super) fn viewed_formats<'a>(
    plan: &'a ImagePlan,
    facts: &ImageFacts<'a>,
) -> impl Iterator<Item = Format> + 'a {
    let listed =
        !plan.flags.contains(ImageCreateFlags::MUTABLE_FORMAT) || !plan.view_formats.is_empty();
    let (list, any): (&[Format], &[Format]) = match listed {
        true => (&plan.view_formats, &[]),
        false => (&[], facts.any_views.of(plan.usage, facts.profile)),
    };
    let any = any.iter().copied();
    (list.iter().copied()).chain(
        any.filter(move |&format| format != plan.format && compatible(plan.format, format, true)),
    )
}

/// Whether a linear image of `plan` fits `limits`, the device's image limits for it
/// (`vkGetPhysicalDeviceImageFormatProperties2` with linear tiling), or what it lacks: a device
/// that reports no such image at all reports every limit 0 ([`LinearNeed::Image`]).
pub(super) fn check_linear_limits(
    plan: &ImagePlan,
    limits: &ash::vk::ImageFormatProperties,
) -> Result<(), LinearNeed> {
    if limits.max_mip_levels == 0 {
        return Err(LinearNeed::Image);
    }
    let max = [
        limits.max_extent.width,
        limits.max_extent.height,
        limits.max_extent.depth,
    ];
    if let Some((&required, &available)) = plan.extent.iter().zip(max.iter()).find(|(w, h)| w > h) {
        return Err(LinearNeed::Extent {
            required,
            available,
        });
    }
    if plan.levels > limits.max_mip_levels {
        return Err(LinearNeed::MipLevels {
            required: plan.levels,
            available: limits.max_mip_levels,
        });
    }
    if plan.layers > limits.max_array_layers {
        return Err(LinearNeed::ArrayLayers {
            required: plan.layers,
            available: limits.max_array_layers,
        });
    }
    if !SampleCountFlags::from(limits.sample_counts).contains(plan.samples) {
        return Err(LinearNeed::SampleCount(plan.samples.as_raw()));
    }
    Ok(())
}

/// A linear image the device made for a texture over a buffer, as Vulkan reports it: its memory
/// requirements' alignment and size, and the layout of its first subresource
/// (`vkGetImageSubresourceLayout`), its bytes (`subresource`, `VkSubresourceLayout::size`) among
/// them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct LinearImage {
    pub(super) alignment: u64,
    pub(super) size: u64,
    pub(super) offset: u64,
    pub(super) subresource: u64,
    pub(super) row_pitch: u64,
    pub(super) array_pitch: u64,
}

/// Whether `image`, bound where `layout` starts in its buffer's memory (`memory_offset` in the
/// memory block), holds exactly Metal's texels in the `room` bytes from there to the end of the
/// buffer's memory: the binding is a multiple of the image's alignment, its texels start at the
/// binding, its row pitch is Metal's and so is its slice pitch where it has several slices, its
/// subresources hold no byte but the texture's texels ([`texels_alone`]), or, where the buffer
/// declares the bytes between rows and slices unobserved (`padding_unobserved`,
/// `BufferDescriptor::padding_unobserved`), none but its texels and those bytes
/// ([`within_padding`]), and its memory stays in the buffer's. A driver may report a pitch other
/// than the one asked of it (NVIDIA does for explicit layouts it does not support), so the image's
/// own report decides.
pub(super) fn aliases(
    layout: &BufferLayout,
    memory_offset: u64,
    room: u64,
    image: &LinearImage,
    padding_unobserved: bool,
) -> bool {
    let held = match padding_unobserved {
        true => within_padding(layout, image.subresource, image.array_pitch),
        false => texels_alone(layout, image.subresource, image.array_pitch),
    };
    image.alignment != 0
        && memory_offset.is_multiple_of(image.alignment)
        && image.offset == 0
        && image.row_pitch == layout.row_pitch
        && (layout.slices == 1 || image.array_pitch == layout.slice_pitch)
        && held
        && image.size <= room
}

/// Whether an image over `layout`'s texels, for a texture of Metal `usage`, may hold them: where
/// its rows (and slices) lie one after another, or it has one row (one slice); else why not,
/// where consecutive rows share bytes (an IOSurface plane's on Metal's GPU, which no image holds
/// a texel of twice), or where they lie with bytes between them that are not texels (the
/// buffer's, which Metal's writes of the texture never reach), whatever its usage.
///
/// An image whose rows lie `row_pitch` apart holds those bytes in its subresource
/// (`VkSubresourceLayout`: "size includes any extra memory that is required based on rowPitch"),
/// and Vulkan lets two kinds of operation write them, against which no query of the device states
/// otherwise:
///
/// - Every image's layout transitions, its first move into `GENERAL` among them (the
///   Synchronization chapter's Image Layout Transitions): "Image layout transitions may perform
///   read and write accesses on all memory bound to the image subresource range", and "If the old
///   layout is VK_IMAGE_LAYOUT_UNDEFINED, the contents of that range may be discarded". No
///   operation lets an image be read before it leaves `UNDEFINED` or `PREINITIALIZED`, the layouts
///   it is made in, so a texture that only reads takes a transition too. It takes no other: every
///   command uses a runtime image in `GENERAL`, and its barriers are memory barriers, which
///   transition no layout (`tracker.rs`).
/// - A render pass's attachments (the Render Pass chapter's Load Operations, Store Operations and
///   Multisample Resolve Operations): "any writes performed by a load operation (as defined by its
///   access masks) to a given attachment may read and write back any memory locations within the
///   image subresource bound for that attachment", and a store and a resolve operation alike.
///
/// A copy and a shader's read or write of the image touch its texels alone (the Memory
/// Model: "A memory operation must not affect memory at a memory location not within its set of
/// memory locations"). But every image takes a first move, across which the host could keep the
/// bytes only by reading them before it and writing them back after (`texture/first_move.rs`'s
/// `Keep`), undoing a CPU write made to them meanwhile, where Metal keeps every one; and lavapipe's
/// shader writes read the bytes its 4-texel blocks reach past a row's texels and write them back, a
/// signaling NaN's bits quieted. So such a texture is a shadow, whose copies address its texels
/// alone; but where `padding_unobserved` (the buffer, of the runtime's own memory, declares that
/// nothing reads those bytes: `BufferDescriptor::padding_unobserved`), an image over its rows and
/// slices may hold the bytes between them, which those operations may then change.
pub(super) fn holding(
    layout: &BufferLayout,
    usage: TextureUsage,
    padding_unobserved: bool,
) -> Result<(), String> {
    // An IOSurface plane's rows that overlap: no image holds them, whatever its usage.
    let shared = layout.shared_bytes();
    if shared != 0 {
        return Err(format!(
            "its rows lie {} bytes apart, {} bytes of texels each, so consecutive rows share \
             {shared} bytes: no image holds a texel twice",
            layout.row_pitch, layout.row_bytes
        ));
    }
    let (rows, slices) = apart(layout);
    if padding_unobserved {
        return Ok(());
    }
    let apart = if rows {
        format!(
            "its rows lie {} bytes apart, {} bytes of texels each",
            layout.row_pitch, layout.row_bytes
        )
    } else if slices {
        format!(
            "its slices lie {} bytes apart, {} rows {} bytes apart each",
            layout.slice_pitch, layout.rows, layout.row_pitch
        )
    } else {
        return Ok(());
    };
    let writers = if usage.contains(TextureUsage::RENDER_TARGET) {
        "its layout transitions read and write, and a render pass's load and store operations \
         write back"
    } else {
        "its layout transitions read and write, its first move from the layout it is made in \
         among them"
    };
    Err(format!(
        "{apart}: an image over them holds the bytes between them, which Vulkan lets {writers}"
    ))
}

/// Whether a linear image over `layout`'s texels, whose subresources take `size` bytes each from
/// the first texel of their slice, `array_pitch` apart, holds no byte but the texture's texels:
/// the texels lie with no byte between them ([`holding`]), and its last subresource ends at their
/// last byte or before. A driver may lay rows out past the texture's (lavapipe rounds the height
/// up to 4), where the buffer's later bytes lie (a surface's next plane). A byte of the image's
/// memory past every subresource is no texel's, and no write through the image reaches it:
/// Vulkan's Memory Aliasing has the bytes a write through an image affects be "those overlapped
/// by the image subresources that were written". A driver that reports one array's whole bytes
/// as its first subresource's has the array refused (the texture is a shadow).
pub(super) fn texels_alone(layout: &BufferLayout, size: u64, array_pitch: u64) -> bool {
    let texels = layout.end() - layout.offset;
    let last = array_pitch.saturating_mul(layout.slices - 1);
    apart(layout) == (false, false) && last.saturating_add(size) <= texels
}

/// Whether a linear image over `layout`'s texels, whose subresources take `size` bytes each from
/// the first texel of their slice, `array_pitch` apart, holds no byte but the texture's texels and
/// the bytes between its rows and its slices, those of a buffer that declares them unobserved
/// (`BufferDescriptor::padding_unobserved`): its rows share no byte, its slices lie its slice
/// pitch apart (where it has several), each at or past the end of the one before's last row to
/// its pitch, and each subresource ends at its slice's last row's pitch or before. Each row's
/// bytes after its texels to its pitch, the last row's included, are such bytes; a byte past the
/// last row's pitch is not, so a driver that lays rows out past the texture's (lavapipe rounds the
/// height up to 4) has the texture refused. A byte of the image's memory past every subresource
/// is no texel's and no padding's, and no operation on the image reaches it ([`texels_alone`]).
pub(super) fn within_padding(layout: &BufferLayout, size: u64, array_pitch: u64) -> bool {
    let rows = layout.row_pitch.saturating_mul(layout.rows);
    let slices = layout.slices == 1 || (array_pitch == layout.slice_pitch && array_pitch >= rows);
    layout.shared_bytes() == 0 && slices && size <= rows
}

/// Whether `layout`'s rows, and its slices, lie with bytes between them that are not texels.
fn apart(layout: &BufferLayout) -> (bool, bool) {
    let rows = layout.rows > 1 && layout.row_pitch != layout.row_bytes;
    let slices =
        layout.slices > 1 && layout.slice_pitch != layout.row_pitch.saturating_mul(layout.rows);
    (rows, slices)
}
