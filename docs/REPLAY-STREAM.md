# Replay streams (format version 2)

A replay stream records, in order, what Metal programs asked of their devices: every call with its
objects and arguments, the bytes the CPU left in GPU-visible memory before each commit, and the GPU
addresses the programs saw. `command-replay play` runs a stream on a Vulkan device through a
runtime, with no guest, and reports deterministic metrics and checksums, so two runs, two devices
or two runtimes compare.

Records use Metal's own selectors and raw enum values. Structs and descriptors are maps keyed by
the recorder's field names (section 2). This page is all a reader needs. The reader, the writer and
the player live in `crates/metal2vulkan-validation` (`src/replay/`).

A stream recorded from a consumer's guest is the consumer's data: it stays on the hosts that may
hold guest data and is never committed. Tests write their own streams.

## 1. Container

- **File:** a 16-byte header, then records. The header is ASCII `M2VRSTRM`, a u32 version (`2`)
  and a u32 of flags. Fixed-width values are little-endian.
- **Flags.** Bytes 12-15 are the flags, a little-endian u32, the same in every segment's header.
  Bit 0, `ANNOUNCED_BYTES_RECORDED` (`FLAG_ANNOUNCED_BYTES_RECORDED`), says the stream keeps the
  announce invariant of section 6; streams recorded before it carry flags 0. Bit 1,
  `COMPLETION_BYTES_RECORDED` (`FLAG_COMPLETION_BYTES_RECORDED`, value 2), says it keeps the
  completion invariant of section 6; a stream recorded after the recorder's completion fix reads
  flags 3. Other bits are 0. The version stays 2: a reader that checks only the magic and the
  version reads a stream with either bit as before. This crate's reader refuses a segment whose
  flags hold another bit or differ from the first segment's.
- **Record:** a kind (u8), the payload's length (varint), the payload. Varints are unsigned LEB128
  of at most 10 bytes; signed values are zigzag-encoded.
- **Append-only.** A recorder only appends whole records and flushes at least at every commit. A
  record cut short at the end of the last segment is a truncated tail: the reader reports it, and
  everything before it plays. A recording process that exits can lose its last records this way,
  about the last 50 ms of calls.
- **Segments.** A stream may be split at any record boundary into files `NAME.0000.m2vr`,
  `NAME.0001.m2vr`, ..., each starting with the header; the stream is the segments in order, and
  names and blobs carry across them. A record cut short in any segment but the last is an error.
- **An unknown record kind** is an error naming the kind and its offset.
- The recorder's statistics file is written apart from the stream and can trail it by up to 5 s;
  the stream is the record.
- **The recording range sidecar.** `NAME.range` beside the first segment (`NAME.0000.m2vr`, or
  `NAME.range` beside a stream of one file `NAME.m2vr`) states how far the recording is sound: one
  line, `last-frame F` (through frame `F`'s present, frames counted from 0) or `whole`. A player
  ends its play there by default (section 10).
- **The display library sidecar.** `NAME.display.metallib` beside the first segment, as
  `NAME.range`, is the metallib of the host's own display shaders, which the host makes as a
  library from source: the stream's `newLibraryWithSource:options:error:` of the device with no
  arguments. It holds the host's display shaders alone, never a program's: a program's own
  libraries reach the stream as data (`newLibraryWithData:error:`, section 9), never as source. A
  player makes that one library of the sidecar, as of data, and nothing else of it; a stream
  without one leaves it skipped and counted, with the host's draws that use it.

| kind | record | payload |
|---|---|---|
| 1 | name | UTF-8 text. The n-th name record of the stream defines name id n, from 1 |
| 2 | blob | the data's SHA-256 (32 bytes), then the data. Each hash at most once per stream, before its first use |
| 3 | call | section 3 |

## 2. Values

Each value starts with a one-byte tag.

| tag | type | encoding | used for |
|---|---|---|---|
| 0 | nil | none | nil; an optional field left unset; host memory |
| 1, 2 | false, true | none | `BOOL` |
| 3 | uint | varint | unsigned integers, raw enums and option sets |
| 4 | sint | zigzag varint | signed integers |
| 5 | f32 | 4 bytes | read as a number; the recorder writes `float` as f64 |
| 6 | f64 | 8 bytes | `float`, `double`, `CGFloat` |
| 7 | object | varint id | Metal objects; object 0, which no object is, a property set to nil (section 9) |
| 8 | address | varint | a GPU address: moved with its object (section 5) |
| 9 | string | varint length, UTF-8 | `NSString`, names |
| 10 | bytes | varint length, data | data, inline |
| 11 | blob | 32-byte SHA-256 | data, by content |
| 12 | list | varint count, values | arrays, tuples and ranges |
| 13 | map | varint count, (name id, value) pairs | structs and descriptors; a key at most once |

- **Structs and descriptors are maps** keyed by the recorder's field names in camelCase, which are
  Metal's property names except where the table in section 9 says. `MTLOrigin` is `{x, y, z}`,
  `MTLSize` `{width, height, depth}`, `MTLRegion` `{origin, size}`, `NSRange` `{location, length}`,
  `MTLScissorRect` `{x, y, width, height}`, `MTLViewport` `{originX, originY, width, height,
  znear, zfar}`, `MTLClearColor` and `MTLTextureSwizzleChannels` `{red, green, blue, alpha}`.
- **An optional field** is nil where it is unset, and a player leaves the property as it was. One
  set to nil reads the same, but for the object properties of the pass descriptor's setters
  (section 9): one of those set to nil is object 0, which a player clears.
- **Other shapes:** a range of host addresses is the list `[start, end]`; a tuple is a list; an
  enum value without data is its name as a string (`"Itu709"`); one with data is a map of one
  pair, `{Name: value}` (a resource `{Buffer: object}` or `{Texture: object}`, an option setter
  `{ResourceOptions: raw}`). A C array and its count are one list.
- **Host memory** (a pointer into the host's or the guest's memory) is nil: its bytes arrive as
  writes (section 6) or as a copy's data (section 7).

## 3. Calls

The payload is: the selector (name id), the receiver (object id, 0 for none), the result (the id
of the object the call makes, 0 for none), the argument count and the arguments, the answer count
and the (name id, value) answer pairs.

- **Selector:** Metal's Objective-C selector as Metal's headers spell it
  (`dispatchThreadgroups:threadsPerThreadgroup:`), a C function's name
  (`MTLCreateSystemDefaultDevice`), `Class.selector` for a class method
  (`MTLRenderPassDescriptor.new`), or an `m2v.` name for what Metal has no call for. Where two
  classes share a selector, the receiver's kind tells them apart.
- **Receiver 0** is the host's platform: the device, descriptor and drawable records it makes.
  A call of receiver 0 that makes nothing and whose selector names no class (no `.`) is a message
  the program sent to nil: it does nothing, and a player plays nothing for it.
- **Arguments:** one value per argument of the selector, in order; an encoder command's are its
  fields. Error out-parameters, blocks and handlers are left out: their effects reach the stream as
  later calls and writes.
- **Answers:** what the call returned that the program could see (section 5). A player ignores
  answer keys it does not use.
- Getters are not recorded, but for `gpuResourceID` (section 5).

## 4. Objects

- **Ids** are nonzero u64s, unique across the whole stream and never reused, so an object that
  several processes use (a surface's buffer, a shared event) has one id.
- An object is made by the call whose result names it and dies at `release` (receiver: the id;
  its last reference). Releasing a device releases everything it made.
- **Devices:** `MTLCreateSystemDefaultDevice`, one per device the recording's host made; every
  guest process may share one. The player serves every recorded device from one runtime device.
- **Device identity:** `m2v.deviceIdentity` (receiver: the device; no arguments) follows each
  `MTLCreateSystemDefaultDevice` record, and opens every later segment, one record per live device
  in id order. Its answers are `name` (the recording runtime's device name) and
  `hasUnifiedMemory` (what the guest's `hasUnifiedMemory` answered), and later keys may join them.
  It plays as nothing; the player takes from it how the recording kept Managed buffers
  (section 10), from the record of the device the guest makes its objects on: the one that most
  calls making an object have as their receiver (the lowest id of a tie). Another device's record
  (the host's own, such as the display's) says nothing of the guest's buffers, and the report names
  such devices. A stream without it says nothing of that.
- Everything else is made by its Metal selector: queues, command buffers, encoders of every kind,
  buffers, heaps, textures and views, samplers, depth and stencil states, events, shared events,
  fences, indirect command buffers, libraries, functions, pipeline states, argument encoders and
  visible function tables, and the descriptors Metal makes as objects (section 9).
- **The host's own plumbing** is recorded as `m2v.` records that stand for a plain Metal call; a
  player plays each as that call:

| record | arguments | what the host did | plays as |
|---|---|---|---|
| `m2v.newBufferOfLengthPaddingUnobserved` | `[length, options, alignedByte]` | `newBufferWithLength:options:`, whose texture rows may keep or change the bytes between them; `alignedByte` (or nil) is the buffer's byte the host put on a 256-byte boundary | `newBufferWithLength:options:` `[length, options]` |
| `m2v.newExportableBufferOfLength` | `[length, options, alignedByte]` | the same buffer in an allocation of its own, which another API imports; result 0 where the host made none | the same, and nothing at result 0 |
| `m2v.newDmaBufBufferOfLengthPaddingUnobserved` | `[length, options, alignedByte]` | the same, also exported as a dma-buf for video decoding; the video's writes arrive as a later commit's writes (section 6); result 0 where the host made none | the same, and nothing at result 0 |
| `m2v.newCommandQueueUnshared` | `[]` | a queue the host kept apart | `newCommandQueue` |
| `m2v.externalWriteDone` | `[]`, on an exported buffer | another API's write into the buffer landed (a decoded picture) | nothing where the textures over the buffer read it in place; the new runtime's backend fills the textures over it at their next use (section 10) |

## 5. Memory and addresses

- **Blocks.** `m2v.newBlockWithLength:storageMode:` (receiver: the device; arguments: the length
  and the `MTLStorageMode`) makes a range the guest sub-allocates in, and answers `gpuAddress` and
  `gpuLength`. A buffer in a block is the block's `m2v.newBufferInBlock:offset:length:options:`
  (arguments: the block, the offset, the length, the `MTLResourceOptions`), which answers
  `gpuAddress` and `gpuLength`.
- **Other buffers and heaps** answer `gpuAddress` and `gpuLength` where the host pinned them at an
  address: `newBufferWithDescriptor:`, `newHeapWithDescriptor:`, and a heap's
  `newBufferWithLength:options:offset:` (`[length, options, offset]`). An object the host made
  without a pinned address answers none. `newBufferWithBytesNoCopy:length:options:deallocator:`
  is `[nil, length, options, nil]`, and a no-copy descriptor's `contents` nil; their bytes arrive
  as writes.
- **A no-copy Managed buffer's bytes at its making.** Where the device's `hasUnifiedMemory` is
  false, the GPU copy of a Managed buffer is made from its bytes at its making, then changed only by
  `didModifyRange:`. A no-copy one (storage mode Managed) then carries those bytes:
  `newBufferWithBytesNoCopy:length:options:deallocator:`'s first argument holds all its bytes as
  the making read them (a blob where they are 64 bytes or longer, bytes otherwise), and a no-copy
  descriptor's `contents` holds its bytes up to `zeroFrom` (all of them where `zeroFrom` is nil;
  nil where it is 0), its GPU copy zero from there. Its bytes still arrive as writes before its
  first use, as any buffer's do.
- **Copied buffers.** A buffer `newBufferWithDescriptor:` made as a copy of host memory (`contents`
  set, `noCopy` not) carries the bytes it starts with as its descriptor's `contents`: a blob where
  they are 64 bytes or longer, bytes otherwise. The player makes the buffer, then puts them at its
  first byte as a write's bytes (section 6).
- **Textures over memory:** a buffer's `newTextureWithDescriptor:offset:bytesPerRow:`,
  `newTiledTextureWithDescriptor:offset:bytesPerRow:` and
  `newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:`, a heap's
  `newTextureWithDescriptor:offset:`, each taking a texture descriptor object (section 9). The
  linear form (Apple's SPI) makes what the first makes of a 2D descriptor, and also a 2D array,
  whose slices Metal lays out itself; the new runtime's player plays a 2D one as the first and
  refuses another.
- **The address map** is the ranges `[gpuAddress, gpuAddress + gpuLength)` of the live objects that
  answered them; `release` removes an object's range. A range that meets a live one in part (the
  program put a newer object in memory an older one still held, as a guest's own allocator in a
  block may) ends the older range, whose inner ranges stay: a word there is the newer object's.
  The player's report counts the ranges it ended (`parted`).
- **Recorded addresses on the old runtime.** The old runtime's player makes every object that
  answered a `gpuAddress` at that address (`AddressPlacement::At`): a buffer, a block or a heap
  at its own answer, the descriptor's `pinnedGpuAddress` aside; a buffer in a heap lands at the
  heap's address plus its offset. A heap with a recorded range plays as its buffers and its
  textures apart, decided by what the stream places in it (read ahead), whatever the heap's type:
  - its buffers (`newBufferWithLength:options:offset:` or `newBufferWithLength:options:` of it)
    go in a heap at its recorded address and of its recorded size, each at its recorded offset,
    never laid out again: a pinned span never grows;
  - its textures (`newTextureWithDescriptor:offset:` or `newTextureWithDescriptor:` of it) go in
    a heap outside the space, laid out for the runtime's sizes (section 10). A texture has no
    address and a heap answers none, so no address the program holds lies in it. It is one heap,
    so textures the stream placed at one offset still share memory.

  A heap the stream places only buffers in is the first heap alone; one it places no buffer in is
  the second alone, and its range keeps its recorded address in the address map; one it places
  both in is both. A texture and a buffer that shared memory in the recording no longer share it.
  The report counts such textures: those whose range in the recording (the heap's recorded
  address plus the texture's offset, for the bytes the runtime answers for its descriptor, as the
  stream holds no texture's size) met the recorded range of a buffer live in the same heap. A
  count above 0 is reported, and the play goes on. Every heap the streams recorded so far hold is
  a placement heap; the rule for an automatic one is held by the player's unit test alone. A heap
  without a recorded range goes outside the space whole, laid out as on any runtime. Every other
  object, and the player's own buffers (its staging, readback and host copies), goes outside the
  space a pin can take (`AddressPlacement::Outside`, `[2^47, 2^48)`; a pin's span ends at or
  below 2^47). A recorded address the runtime refuses (its span meets a live pin's, its first page
  starts no 32-KiB granule, or it leaves the space), a buffer the runtime will not place at its
  recorded offset in its heap's recorded span, or an object the runtime made at another address
  than it recorded, stops the play, naming the object and its recorded address
  (`PlayError::Unpinned`); no other placement is tried. Each range's address in the play is then
  its recorded one, so no address argument moves (below). The report says how many objects it
  pinned, how many heaps' textures it made outside the space and how many textures met a live
  buffer (section 10).
- **Address arguments move.** On the new runtime the player makes each object where the runtime
  has room. Every address value (tag 8) whose value lies in a live range becomes the player's
  address of that range's object plus its offset into the range; the innermost range wins, so an
  address in a buffer in a block moves with the buffer. One outside every live range stays. The
  report counts the address values (`address_arguments`) and those that moved
  (`relocated_words`): none on the old runtime, whose ranges lie at their recorded addresses.
- **Written words play as recorded.** A word's value is no reason to move it: the bytes the
  player puts into an object (`m2v.write`, a copied buffer's `contents`, the copies from the
  guest's RAM, a Private range's bytes before a dispatch) and the data of every
  `set*Bytes:length:atIndex:` play as the stream holds them, on either runtime. A word there that
  is an address, which a stream recorded before the escape note cannot tell from data, keeps its
  recorded value. The player counts, as a census that writes nothing, the words that look like
  addresses: in the bytes put into an object that answered a `gpuAddress` and in `set*Bytes`
  data, each 8-byte word at an offset that is a multiple of 8 from the target's start whose value
  lies in a live range (`address_looking_words`), and apart from them the 8-byte values at 4 past
  a multiple of 8 that do (`misaligned_address_words`). Both are the stream's, the same on every
  play on either runtime.
- **The escape note.** `m2v.bufferEscaped` (receiver: a buffer; arguments: `[kind]`) says the
  program may hold the buffer's address in memory from then on; the whole buffer escapes. Its
  kind is 0 for a `gpuAddress` read of the buffer or of a texture over it, 1 for an argument
  encoder's write of its address or of a texture-over-it's resource ID, 2 for a `gpuResourceID`
  read of a texture or view over it, and 3 for a residency set's add. The recorder writes it once
  in the buffer's life, at the first escape, in stream order: after the `m2v.write` records of
  the commits before it and before those of every commit after it. The buffer is a no-copy one,
  never Managed. Once a buffer's note is read, a written word that looks like an address (above)
  whose value lies in that buffer's recorded range stops the play, naming its target and offset,
  its value, the buffer and the note's kind (`PlayError::NotedWord`): the word may be the
  buffer's address, which the play would have to move, and that move is not built. A note of a
  kind past 3, or of an object no call made, is an invalid stream. A stream recorded before the
  recorder wrote the note holds none, and the report's `addresses` line says it predates the note
  (section 10).
- **Resource IDs.** A descriptor's resource index places its object at that ID, as Metal places it
  whether or not the index is forced: a texture descriptor's and a render pipeline's
  `resourceIndex` (`[index, forced]`), a sampler's (an index), and a view's `resourceIndex:`
  argument. Every other object takes the ID its runtime gives it; on the new runtime, whose
  consumer gives every texture and sampler index, the index of the ID the program saw of the
  object (its first `gpuResourceID` record, read ahead), else the lowest index of its class that
  no live object holds and the program did not reserve (`reserveResourceIndicesForResourceType:
  indices:indexCount:`), as Metal's pool hands out a sampler state's ID. A reserved index is taken
  only where the stream names it. A named index a live object of the play holds is taken from
  it, as Metal's later texture at an ID takes the ID: the earlier object's release leaves the
  index. A placement at an index waits first for the commits of the command buffers that named
  an object while it held the index, live or released since, to retire (their completed handlers have run, which the runtime runs once it
  has let go of what they held): Metal leaves it to the program never to take an element from
  under work that runs, and the program's own timing kept that. An object named after a later
  one took its index holds it no more: the index is the later one's, and that naming owes it
  nothing. Where the runtime refuses a named index (it lies outside the object's class), the object
  takes the lowest free one, and the report counts it. A constexpr sampler, which the stream names
  no index for, takes the highest free sampler index on the new runtime, outside the sequence the
  program's sampler states take, so none of theirs takes its index.
- **Textures outside the program's IDs.** A texture the recorder makes for itself, not the
  program's (the host's cursor), comes from a texture descriptor record whose map holds
  `outside: true` and a device's `newTextureWithDescriptor:` of that descriptor. Live, it took no
  ID in Metal's space, so the program's textures took theirs as if it were not there. The player
  makes it where it moves no ID of the program's: on the old runtime with no ID
  (`IdPlacement::Outside`), on the new runtime at the highest free index of its class, outside
  the sequence the program's textures take. A marked descriptor with a `resourceIndex`, a marked
  descriptor any other call makes a texture of, and a drawable's map with the mark are refused; a
  `gpuResourceID` record of a marked texture stops the play (`invalid stream`): the program never
  saw its ID, so the mark or the record is wrong.
- **The IDs the program saw.** Each `gpuResourceID` the platform answered of a texture, sampler,
  depth-stencil state, compute or render pipeline, indirect command buffer or visible function
  table is a `gpuResourceID` record: the object its receiver, no arguments, result 0, and the ID
  under the answer `gpuResourceID`. A getter that raised answers nothing. The player compares the
  first answer of each object with the ID its runtime gives the object, and reports each object
  where they differ (section 10): the bytes in which the program wrote that ID name no object of
  the play.

## 6. CPU writes

- `m2v.write` (receiver: a block or a buffer, a surface's pages included; arguments: one or more
  pairs of an offset and its data, bytes or blob): the bytes the CPU left at those offsets.
- **Before each `commit`**, the stream holds a write for every byte that changed since it was last
  recorded, in the memory the command buffer can reach: its bound buffers, textures over buffers
  and argument buffers; its `useResource:` and `useHeap:` resources; its attachments, blit operands
  and executed indirect command buffers. Memory reached only through pointers the command buffer
  did not declare is undefined in Metal and outside this format.
- **The comparison point** is the bytes as last recorded. When a command buffer completes, the
  4 KiB chunks wholly inside ranges only the GPU writes (attachments, blit destinations) take the
  GPU's bytes as the stream's without a write: the player's own run writes them again.
- **The completion invariant.** In a stream whose header has bit 1 set, at each command buffer's
  completion the stream holds every chunk that the command buffer's work could store to as the GPU
  left it, before any later CPU write, on each of the four late routes by which a recorder learns
  of a completion: a signalled event, a final status read, a refused hook and a released command
  buffer. Each such chunk that differs from the comparison point is an `m2v.write` after the
  command buffer's `commit` record, holding the bytes the run left. The chunks are those a blit
  wrote only part of, and everything the command buffer's compute and render encoders reach: each
  buffer, texture and heap they bind or use, and what an indirect command buffer they execute
  binds, except a resource `useResources:usage:` names for reading alone. A `Managed` buffer
  counts only on a device that keeps one copy of it; where there are two, the GPU never writes the
  CPU's copy. A player waits before each write (section 10), so it puts back what its own run
  wrote and changes nothing. The write moves the comparison point: when the CPU later restores the
  earlier bytes (a program resetting counters a kernel adds to), the restore differs from it and is
  recorded before the next commit that reaches those bytes.
- **A stream without bit 1** was recorded before the recorder's completion fix and keeps no such
  invariant: where a completion's bytes are missing, a CPU write that restores the bytes last
  recorded is missing too, and the play runs the next command buffer from its own run's bytes
  where the recording device ran it from the restored ones. Such a stream may differ from what the
  program did, and every report the player writes of it says so on its first line (section 10).
- **No half-written bytes.** Before capturing, a recorder waits for every earlier command buffer
  that can run: one not held on an event wait that a later record satisfies. The bytes may then
  include final bytes the GPU wrote; the player puts them back at the same point, after its own
  wait, so they change nothing.
- Private memory never appears in a write. A Managed buffer's changed range is a write on the
  buffer, immediately before its `didModifyRange:`.
- **The announce invariant.** In a stream whose header has bit 0 set, at each `didModifyRange:`
  record on a buffer, for every 4 KiB chunk the range touches (of the target its `m2v.write`
  records name: the buffer, or the block it lies in), the last `m2v.write` record before it that
  carries the chunk holds the bytes the program's CPU copy of the buffer held at the call, bytes the
  GPU wrote there (a blit's, a pass's, `synchronizeResource:`'s read-back) that the recorder took as
  the stream's without a write included, however late it took them. A player may push its recorded
  bytes of the range at the announce. A stream without the bit gives no such promise where the GPU
  wrote the chunk, and a player pushes its own read-back there.
- Granularity is the recorder's: whole pages or changed runs. Data of 64 bytes or more is best a
  blob, so a repeated page is stored once.

## 7. Surfaces and the guest's RAM

- **A surface's pages** are a no-copy buffer, whose CPU changes arrive as writes at each commit
  that reaches it (section 6). There are no surface records of their own.
- **A texture over a surface's plane** is `newTextureWithDescriptor:iosurface:plane:` with that
  buffer as its receiver and the arguments `[descriptor, {plane, offset, bytesPerRow, planar,
  facts}]`: the plane's index, its first byte and row pitch in the buffer, whether the surface is
  planar, and `facts`, what the texture reads of the surface besides: `{pixelFormat, planeCount,
  width, height, chromaOffset, chromaBytesPerRow, ycbcrMatrix}`, the pixel format as its four
  characters big-endian, the matrix one of `"Itu601"`, `"Itu709"`, `"Itu2020"`,
  `"Smpte240M1995"`, `"DciP3"`, `"P3D65"`, `"Other"`, or nil. Where `facts` is nil, the buffer
  stands for no surface and the texture is the buffer's `newTextureWithDescriptor:offset:bytesPerRow:`.
  A texture of a two-plane YCbCr or XR10 format over a surface of two planes reads the facts.
- **Copies between the guest's RAM and a buffer** are blit-encoder commands, ordered against the
  encoder's others as a buffer-to-buffer copy is. Each is recorded only where the host's platform
  made it, after the earlier command buffers that can run were waited for.
  - `m2v.copyFromGuestRam` and `m2v.copyFromGuestRamAt` (`[buffer, [[byte, data], ...]]`): each
    piece's data, as the copy read it (bytes, or a blob where long), lands in the buffer from its
    byte. The two differ in the host's call, one offset for all its runs or one per run, not in
    the record.
  - `m2v.copyToGuestRam` (`[buffer, [[byte, [start, end]], ...]]`): the buffer's bytes from each
    byte, as long as its run of host addresses, go back to the guest's RAM. It carries no data:
    later reads of that RAM carry their own bytes, so it needs nothing beyond its order. The
    player copies the bytes into memory of its own that nothing reads.
  - Streams recorded before the host left out the copies its platform refused may hold one in the
    call's own arguments; a player stops there.

## 8. Commands, commits and frames

- **Encoder commands** are Metal's selectors on the encoder: every render, tile, compute, blit and
  resource-state call. An indirect command buffer's contents are `indirectRenderCommandAtIndex:`
  records on it, `[index, commands]`.
- **Uploads and downloads:** `replaceRegion:...` carries its bytes; `didModifyRange:`,
  `synchronizeResource:` and `synchronizeTexture:slice:level:` sit where they were made;
  `getBytes:...` has a nil destination and may answer `sha256`.
- **Commits:** the order of `commit` records is the global commit order. A command buffer's queue is
  the receiver of the call that made it; each queue keeps its own order, and an event wait may hold
  one queue while the others go on. Record order is the order of the program's calls, not of its
  GPU work: a command buffer committed before a call may wait for an event that only a later
  record signals, so a commit record does not promise that the work ran, or could run, before the
  records after it.
- **Synchronization:** `encodeSignalEvent:value:` and `encodeWaitForEvent:value:` on a command
  buffer; fences as encoder calls; `setSignaledValue:`; and `waitUntilCompleted` and
  `waitUntilSignaledValue:timeoutMS:` where the program's CPU waited.
- **The host's own commit splits** (no Metal counterpart): `m2v.split` commits its command buffer
  as a part and makes the command buffer of the same queue that goes on from it (its result);
  `m2v.part` makes a command buffer of the same queue for a later range (its result), and
  `m2v.commitPart` commits its receiver as a part where its answer `cut` is `"Cut"`. Where it is
  `"Fails"` or `{Whole: reason}`, the host committed nothing; a record without the answer commits.
  A split or part of result 0 is one the host did not cut: it commits nothing and makes nothing.
- **The host's scheduling records** change nothing a program sees, and a player plays none of
  them: `m2v.setCompletionJoinsNext` (`[joins]`: the host groups the command buffer's completion
  with the next ones of its queue), `m2v.holdsNoPredication` (`[]`), `m2v.fold` (`[later]`: the
  command buffer takes in how the GPU fails the later one's encoders) and `m2v.rebindNoCopy`
  (`[bytes, deallocator]`, recorded `[nil, nil]`: a no-copy buffer moved to new host memory, whose
  bytes arrive as writes).
- `m2v.privateBytes` (receiver: a buffer, which a `release` may already have ended; arguments
  `[dispatch, {location, length}, before, after]`) holds the bytes of a Private buffer's range
  before and after one dispatch of a compute pipeline made with linked functions: `dispatch` is
  the dispatch's position in the stream (how many call records come before it, from 0), and
  `before` and `after` are blobs that lie in the blob directory alone. It is recorded at the
  command buffer's completion, so later in the stream than the dispatch. The player reads these
  records ahead and plays the dispatch they name between the bytes before and a comparison with
  the bytes after (section 10).
- `m2v.present` (arguments: the texture over the guest surface shown) is recorded where the
  device's display shows a guest present, with or without a window. It ends a frame: the player
  waits for what can complete and closes the frame's counters; it reads nothing.
- `m2v.check` (arguments: objects) asks the player to read those objects there and report their
  checksums. Its optional answer `sha256` is a list of the recording's own, one per object.

## 9. Descriptors and libraries

- **Descriptor objects.** The platform (receiver 0) makes the descriptors Metal makes as objects,
  and setter records on them change them; a creation call names the object, as it is then.
  - `m2v.textureDescriptor` (`[map]`): `MTLTextureDescriptor.new` with the map's properties set,
    in the order `textureType`, `pixelFormat`, `width`, `height`, `depth`, `mipmapLevelCount`,
    `sampleCount`, `arrayLength`, `usage`, `resourceOptions`, `swizzle`, `resourceIndex`
    (`[index, forced]`), `protectionOptions`, `framebufferOnly`, `isDrawable`,
    `writeSwizzleEnabled`, `allowGpuOptimizedContents`, `rotation`, and `outside` (a bool; only
    `true` is written), the mark of a texture outside the program's IDs (section 5).
    `MTLTextureDescriptor.texture2DDescriptorWithPixelFormat:width:height:mipmapped:` makes one
    too; `setUsage:` and `setResourceOptions:` on it change it.
  - `MTLRenderPassDescriptor.new` (`[]`) makes a render pass descriptor. `m2v.setPass` (`[map]`)
    sets its own properties: `visibilityResultBuffer`, `visibilityResultType`,
    `renderTargetArrayLength`, `renderTargetWidth`, `renderTargetHeight`,
    `defaultRasterSampleCount`, `defaultColorSampleCount`, `rasterizationRateMap`,
    `imageblockSampleLength`, `threadgroupMemoryLength`, `tileWidth`, `tileHeight`.
    `m2v.setColorAttachment` (`[index, map]`), `m2v.setDepthAttachment` (`[map]`) and
    `m2v.setStencilAttachment` (`[map]`) set that attachment's: `texture`, `resolveTexture`,
    `level`, `slice`, `depthPlane`, `resolveLevel`, `resolveSlice`, `resolveDepthPlane`,
    `loadAction`, `storeAction`, `storeActionOptions`, then `clearColor`, `clearDepth` and
    `depthResolveFilter`, or `clearStencil` and `stencilResolveFilter`. A nil property takes
    nothing: the descriptor keeps what it had. An object property (`visibilityResultBuffer`,
    `rasterizationRateMap`, `texture`, `resolveTexture`) set to nil is object 0, which clears it.
  - `m2v.rateMapDescriptor` (`[{screenSize, mutability, minFactor}]`) makes an
    `MTLRasterizationRateMapDescriptor`; its layers are set on it afterwards.
- **Drawables.** A `newTextureWithDescriptor:` of receiver 0 whose argument is a map
  (`{textureType, pixelFormat, width, height, usage, storageMode}`) and whose answer
  `m2v.drawable` is true is a window's swapchain image the stream never saw made. The player makes
  a texture of that descriptor to stand in for it.
- **Creation descriptors** are a map argument of their creation call. Option setters are a list of
  `{Setter: raw}` applied in order to the options of Metal's new descriptor: `ResourceOptions`,
  `ResourceOptionsOr` (bits or'ed in), `StorageMode`, `HazardTrackingMode`.
  - `newBufferWithDescriptor:`: `{length, options, pinnedGpuAddress, contents, noCopy,
    deallocator, zeroFrom}`.
  - `newHeapWithDescriptor:`: `{size, heapType, options, pinnedGpuAddress}`. The old runtime's
    player makes the heap's buffers in a heap at its pin and its textures in a heap outside the
    pinned space (section 5).
  - `newSamplerStateWithDescriptor:`: `{minFilter, magFilter, lodAverage, compareFunction,
    sAddressMode, tAddressMode, rAddressMode, mipFilter, maxAnisotropy, normalizedCoordinates,
    supportArgumentBuffers, reductionMode, lodMinClamp, lodMaxClamp, lodBias, borderColor,
    resourceIndex}`.
  - `newDepthStencilStateWithDescriptor:`: `{depthCompareFunction, depthWriteEnabled, front,
    back}`, each face `{stencilCompareFunction, stencilFailureOperation, depthFailureOperation,
    depthStencilPassOperation, readMask, writeMask}`.
  - `newRenderPipelineStateWithDescriptor:error:`: `{label, vertex, fragment, vertexDescriptor,
    colors, depthPixelFormat, stencilPixelFormat, rasterSampleCount, alphaToCoverage, alphaToOne,
    rasterizationEnabled, inputPrimitiveTopology, maxVertexAmplificationCount,
    supportIndirectCommandBuffers, vertexLinked, fragmentLinked, privateState, tessellation,
    maxVertexCallStackDepth, maxFragmentCallStackDepth}`. The last three are always written where
    a recorder carries them: `tessellation` is `{partitionMode, maxFactor, factorScaleEnabled,
    factorFormat, controlPointIndexType, factorStepFunction, outputWinding}`, Metal's raw values,
    and the depths are uints. A record without them, as every stream recorded before the recorder
    carried them, means Metal's defaults: `{0, 16, false, 0, 0, 0, 0}` and depths 1 and 1. A player
    gives a tessellation state other than the defaults to the pipeline; the translator draws a
    post-tessellation vertex function's patch with it, which it knows from the function's AIR, and
    ignores it for any other, as Metal does. The vertex
    descriptor is `{layouts: [{bufferIndex, stride, stepFunction, stepRate}], attributes:
    [{attributeIndex, format, offset, bufferIndex}]}` (a stride of 2^64 - 1 is
    `MTLBufferLayoutStrideDynamic`); `colors` is `[[index, {pixelFormat, blendingEnabled,
    sourceRgbBlendFactor, destinationRgbBlendFactor, rgbBlendOperation, sourceAlphaBlendFactor,
    destinationAlphaBlendFactor, alphaBlendOperation, writeMask}], ...]`; linked functions are
    `{functions, binaryFunctions, groups, privateFunctions}`, a group `[name, functions]`; the
    private state (Apple's) is `{openGlModeEnabled, sampleMask, sampleCoverage,
    sampleCoverageInvert, logicOperationEnabled, logicOperation, depthStencilWriteDisabled}`. The
    host's own pipelines take the short form `{label, vertex, fragment, color0PixelFormat,
    resourceIndex}`.
  - `newComputePipelineStateWithDescriptor:options:reflection:error:`: `{label, function,
    threadgroupSizeIsMultipleOfExecutionWidth, maxTotalThreadsPerThreadgroup, maxCallStackDepth,
    supportIndirectCommandBuffers, stageInput, linked, libraries: {preloaded, loaded},
    openClMode}`.
- **Keys that are not Metal's property names:**

| key | Metal's |
|---|---|
| `vertex`, `fragment`, `function` | `vertexFunction`, `fragmentFunction`, `computeFunction` |
| `colors` | `colorAttachments` |
| `depthPixelFormat`, `stencilPixelFormat` | `depthAttachmentPixelFormat`, `stencilAttachmentPixelFormat` |
| `vertexLinked`, `fragmentLinked`, `linked` | `vertexLinkedFunctions`, `fragmentLinkedFunctions`, `linkedFunctions` |
| `libraries` | `preloadedLibraries` and the libraries the functions load |
| `alphaToCoverage`, `alphaToOne` | `alphaToCoverageEnabled`, `alphaToOneEnabled` |
| `tessellation` | the tessellation properties: `partitionMode`, `maxFactor`, `factorScaleEnabled`, `factorFormat`, `controlPointIndexType`, `factorStepFunction`, `outputWinding` are `tessellationPartitionMode`, `maxTessellationFactor`, `tessellationFactorScaleEnabled`, `tessellationFactorFormat`, `tessellationControlPointIndexType`, `tessellationFactorStepFunction`, `tessellationOutputWindingOrder` |
| `front`, `back` | `frontFaceStencil`, `backFaceStencil` |
| `heapType` | `type` |
| `options` | the setters of `resourceOptions`, `storageMode` and `hazardTrackingMode` |
| `pinnedGpuAddress`, `contents`, `noCopy`, `deallocator`, `zeroFrom` | the host's buffer creation: a pin, the bytes a copied buffer starts with or a no-copy Managed one's at its making (section 5), no-copy host memory and its deallocator, and the bytes known zero from an offset |
| `openGlModeEnabled`, `openClMode` | `openGLModeEnabled`, `openCLModeEnabled` |
| `znear`, `zfar` | `MTLViewport`'s `znear` and `zfar` |
| `rotation`, `isDrawable`, `framebufferOnly`, `protectionOptions`, `writeSwizzleEnabled` | Apple's texture descriptor properties of those names |

- **Functions:** `newFunctionWithName:constantValues:error:` is `[name, constants]`, each
  constant `{index, dataType, value}`, the value 32 bytes holding the constant packed from the
  first (the rest zero); `newFunctionWithDescriptor:error:` is `[name, constants, options,
  specializedName]`.
- **Libraries:** `newLibraryWithData:error:` takes the metallib as a blob: the SHA-256 of the exact
  bytes passed (for a universal file that is not the runtime's `Library::content_hash`, which
  hashes the AIR slice). The blob record may be left out; the player then reads the data from its
  blob directory (`--blobs DIR`, one file per blob named by its lowercase hex hash) and checks it
  against the hash.

## 10. The player

`command-replay play --stream FILE --out DIR --device-type cpu|gpu --shader-cache DIR
[--blobs DIR] [--runtime old|rt] [--snapshots yes|no] [--managed one-copy|two-copy]
[--until-frame F] [--past-range yes|no] [--gpu-wait-timeout SECONDS]` plays a stream (its first
segment) on one device through the old runtime (the default) or the new one (`rt`), and writes
`DIR/report.txt`. Each wait on the device's work is bounded by `--gpu-wait-timeout` (10 seconds
by default; docs/VALIDATION.md, the hang tripwire), which a play on a CPU device raises. It
writes each frame's `frame` and `end` lines as the frame ends, flushed, so a play that never ends
leaves every frame it finished. Every line it writes to `report.txt`, the frame lines among them, it writes to
its standard error too, so a caller that loses the directory keeps the report.
With `--snapshots no` it takes no `m2v.privateBytes` snapshot on either runtime: each such dispatch
plays as it is, counted on the `unsnapshotted` line, so two runtimes' plays do the same work and
their times and memory compare. With `--until-frame F` it stops after frame `F`'s present (frames
counted from 0), completes the work committed so far, reports as at the stream's end and says
`until-frame F`. `command-replay dump --stream FILE` prints the call records as text, and
`command-replay shapes --stream FILE` each selector's argument shapes, with no data.

- **The recording range.** A play ends at the stream's recording range by default: through the
  frame its sidecar (section 1) states, else the frame the player's table `RECORDED_RANGES`
  states for it (the streams recorded before sidecars, found by their first segment's file name
  and byte length), else the whole stream. `--past-range yes` plays past it, to the stream's end
  or to `--until-frame F`; without it, `--until-frame F` stops the play at the earlier of `F` and
  the range. The
  report's `range` line says which range the play used, where it came from, and whether the play
  ends there or `--past-range yes` takes it past; a play that ends at the range says
  `until-frame F` as `--until-frame` does. A sidecar that reads as neither `last-frame F` nor
  `whole` is an error.
- **The display library.** The report's line after `range` is `display library the stream's
  sidecar, N bytes, sha256 HEX, git blob ID` where the stream has `NAME.display.metallib` (section
  1), which names the file the play ran by its SHA-256 and its git blob id in the host's tree, and
  the host's library from source is made of it on either runtime; else `display library none: the
  host's library from source is skipped`.
- **Streams recorded before the completion fix.** Where the stream's header lacks bit 1 (section
  6), the report's first line is, by itself, `recorded before the recorder's completion fix: the
  stream may differ from what the program did`, so no table taken from the report loses it.
- It moves address arguments (section 5) and plays the host's plumbing as the plain calls it
  stands for (section 4).
- **Managed buffers** (the new runtime) play in the form the recording device kept them: its
  `m2v.deviceIdentity` record's `hasUnifiedMemory`, or where the stream has none, `--managed`
  (`one-copy` for unified memory, `two-copy` for a GPU copy). With neither, a Managed buffer stops
  the play with an error that says so. One copy: every Managed buffer plays as a Shared buffer, on
  every device. Two copies: it plays as the runtime's Managed buffer, whose GPU copy starts from
  the bytes it was made with: a copied buffer's bytes, or a no-copy one's bytes at its making
  (section 5). A stream with no identity record gives a no-copy buffer none, so its GPU copy takes
  the writes recorded before its first use (a commit of a command buffer one of whose encoders
  names it), the nearest record the stream has of its bytes at its making. Where the device keeps
  a GPU copy, the player places it as it places a Private buffer of the buffer's length (below), a
  buffer made in a stream's block or heap included, and frees that range as it frees the buffer's.
  On a device with unified memory the runtime keeps one copy anyway. The report's `managed` line says which form
  played and from what, counts the Managed buffers and the no-copy ones whose contents came from
  first use, not from create, and says where a two-copy recording played as one copy.
- **No-copy buffers** (the new runtime) play as the product makes them, by creator:
  - A `newBufferWithDescriptor:` with `noCopy` is a pool buffer in the product: a guest buffer's
    storage in one of the host's blocks. Where its recorded address and length lie inside a live
    block's recorded range (`m2v.newBlockWithLength:storageMode:`) it plays as a buffer of the
    block at that offset, as `m2v.newBufferInBlock:offset:length:options:` does, its bytes at its
    making (`contents`, where the stream gives them) first put there; elsewhere, in a place of its
    own, as any other buffer. No import. The recordings do not tell the program's own no-copy
    memory, which the product imports, from these, so it plays as a pool buffer too: a known gap.
    The host's recorder makes such a buffer over the program's own memory only with a
    `pinnedGpuAddress` and no `zeroFrom`, so the report's `gap: descriptor-form buffers that may
    be the app's own no-copy memory are played as pool buffers, count unknown: at most N, ...`
    line bounds the gap by those outside every block and their bytes, wherever there are any.
  - A `newBufferWithBytesNoCopy:length:options:deallocator:`, which in the recordings is the host's
    own buffer and never the program's, is made over zeroed host pages of the player's own: its
    length rounded up to the device's `minImportedHostPointerAlignment`, imported as a block of its
    own (`Device::import_host_memory`), as the product imports such pages, and freed once the
    runtime frees the import. Where the buffer plays with two copies, the import holds the copy
    the CPU writes where the runtime keeps a GPU copy, and the copy shaders and textures reach
    where the player keeps the second copy itself. On a device that imports no host memory such a
    buffer stops the play with a refusal that says so.

  A texture over an imported buffer is the runtime's counted copy. Each frame's `over_copies` and
  `over_copy_bytes` count every copy of a texture over a buffer, and its `over_copies_imported` and
  `over_copy_bytes_imported` those over an imported one alone; the rest are over the runtime's own
  blocks, where the device's linear image does not take the texture. The report's `no-copy
  buffers` line counts the descriptor-form ones in a block, those outside every block and their
  bytes, and the bytes-form ones and their bytes imported, apart; its `textures over buffers` line
  counts the textures over buffers it made (not substituted), in place and in an image of their
  own, by whether the runtime's `textures_copied_over` gauge rose across the making (a copied
  texture freed on another thread in that moment hides one), and its `textures over buffers in an
  image of their own, by pixel format and usage` line those in an image of their own by the
  descriptor's pixel format and usage, most first.
- **Announced bytes** (both runtimes' backends). On a stream whose header has bit 0 set (section
  6), a `synchronizeResource:` of a buffer played with two copies leaves the CPU's copy as the
  stream's writes made it, so the next `didModifyRange:` puts the stream's bytes into the GPU's
  copy: on the new runtime it copies the GPU's copy into an aside buffer of the player's own, which
  a check of the buffer reads from the read-back to its next `didModifyRange:`; on the old runtime
  it moves nothing, and a check reads the GPU's copy as it always does. A stream without the bit
  plays its read-backs into the CPU's copy, as Metal does. A Managed buffer played as one copy (in
  the stream's block, or on the old runtime on a device with unified memory) holds the GPU's bytes
  either way. The report's `announced bytes` line says which rule played and counts the read-backs
  it kept out of the CPU's copies.
- **Unwritten buffers** (the new runtime): its report's `unwritten` line counts the Shared
  buffers made without bytes that a compute encoder bound or used and whose command buffer
  committed before any recorded write, upload or GPU pass reached their bytes since their making,
  and names the first few with the command buffer that read each. Such a buffer plays with
  whatever its memory held, which is where a recording that missed the program's writes shows.
  What reaches a buffer's bytes: an `m2v.write` into it, into its block over them or into another
  buffer of the block over them; a blit or render encoder's call that names it or such another
  buffer; and the commit of a dispatch that binds it, which may write it. A dispatch that writes
  such a buffer before it reads it is counted too, and a texture over a buffer is not followed.
  The line counts; it stops nothing.
- **Writes into Managed textures** (both runtimes' backends tell a texture's storage mode): the
  report's `texture_writes` line counts the CPU's writes into Managed textures
  (`replaceRegion:...` and any other call on a texture that carries bytes), their bytes and
  textures, and of them those made while a command buffer that used the texture (an encoder of it
  named the texture or a view of it, or it was made with a descriptor that names one) was committed
  and the stream records no completion of it. The play waits for each command buffer at its commit;
  the stream records a completion only where the program's CPU waited: `waitUntilCompleted` of the
  command buffer or a later one of its queue, or `waitUntilSignaledValue:timeoutMS:` reaching a
  value that it or a later one of its queue signals. So such a write may have met a user still
  running on the recording device. It counts them by the presents between the user's commit and the
  write (0, 1, 2, 3 or more), names the first eight (position, texture, user, presents) and counts
  the writes into textures whose storage the backend does not tell. It stops nothing.
- After each commit, before each write, check and present, and where the program waited, it waits
  for every committed command buffer that can complete: one whose event waits are met by values the
  CPU set or by signals of command buffers that can complete. It never waits on a command buffer
  held behind an event wait. Each wait is bounded under the validation crate's hang protocol.
- It reads the `m2v.privateBytes` records (section 8) ahead, before it plays the stream, and plays
  each dispatch they name with its ranges' bytes before put in and its bytes after compared: it
  puts the bytes before (as recorded, counted as a write's are) in a Shared buffer of its own;
  ends the dispatch's compute encoder; copies them into the ranges in a blit encoder; makes the
  compute encoder again by the call that made it, with every `set…` and `use…` call the program
  made on it; dispatches; then ends it, copies the ranges out in a blit encoder and makes it again the
  same way. A fence of its own orders each of these encoders after the one before it. Once the
  command buffer completed, it reads the bytes after back and counts the 4-byte words of each
  range that differ from the recording's bytes after. A difference is reported, and
  the play goes on. Its own objects take ids from 2^63 + 1. On a backend that takes no snapshots
  (the new runtime's, which has no blits between buffers yet), each such dispatch plays
  as it is and is counted on the report's `unsnapshotted` line.
- It lays each placement heap out again for the runtime's sizes and alignments, as a program that
  lays out its heap by `heapTextureSizeAndAlignWithDescriptor:` and
  `heapBufferSizeAndAlignWithLength:options:` does: the stream's offsets and heap sizes are the
  recording device's answers. Before it plays, it reads ahead every `newTextureWithDescriptor:offset:`
  and `newBufferWithLength:options:offset:` of each placement heap and asks the runtime the size
  and alignment of each, a texture's of its descriptor as the calls before it made and set it.
  Going up the offsets the stream placed at, each keeps at least its distance from the one below,
  moves up until the resources placed below it end, and rounds up to the largest alignment of the
  resources placed there. The heap keeps at least its distance past its last offset and holds the
  resources placed there. Resources the stream placed at one offset still share it; resources at different offsets no longer share memory, even where the
  stream's did, on every runtime. Where every resource ends before the next offset and every offset
  is aligned, the layout is the stream's. A placement whose size the runtime does not answer keeps
  its offset, and the play meets the runtime's refusal where it lies. On the old runtime a heap
  grown past its recorded span would hold pages the recording gave the objects after it, so a heap
  with a recorded range is laid out by its textures alone, which go in a heap of their own outside
  the space, while its buffers keep their recorded offsets in a heap of its recorded size at its
  recorded address (section 5).
- It reports, for each group of command buffers it waited for (normally the one just committed),
  for each frame and in total, two sets of counts:
  - **Fixed by the stream** (`stream`), the same on every play of the stream on any runtime: the
    calls played (`calls`), the command buffers committed (`commits`, a host's split included),
    the encoders made (`encoders`: the objects a command buffer or an encoder makes), the draws
    and dispatches their calls issued (`draws` and `dispatches`: an encoder's calls whose
    selectors begin `draw` and `dispatch`), the written words that look like addresses
    (`address_looking_words`) and the 8-byte values at offsets 4 past a multiple of 8 that lie in
    a live range (`misaligned_address_words`), both played as recorded (section 5), the address
    arguments (`address_arguments`) and those it moved (`relocated_words`), and the bytes it
    wrote or copied in, `m2v.write`, the copied
    buffers' `contents`, the copies from the guest's RAM and the `m2v.privateBytes` bytes before
    (`written_bytes`). A group's calls are those played before it completed:
    the call that waited for it counts in the next.
  - **As executed** (`executed`): the counters the runtime exposes. The old runtime's are the
    Vulkan commands it recorded (submissions, render passes, barriers, dispatches, draws, copies)
    and the pipelines it made. They differ between runtimes, and between plays on one runtime
    where what it records depends on timing. The old runtime's commands depend on whether the
    pipelines it makes in the background have landed: a draw whose own pipeline is still being
    made records through the pipeline's general form and keeps nothing for a later encoder to
    draw again. So a play with cold shader caches records more commands than a warm one, and warm
    plays can move a few draws and barriers from one group or frame to another. Only the stream's
    counts are compared between plays.

  Its own readbacks are not counted. The new runtime's backend reads a Private buffer, which has
  no CPU mapping, through a blit copy of it into a Shared buffer of the play's own, on a queue of
  the play's own, and waits for the copy. Its own work around a dispatch an `m2v.privateBytes` record
  names (its buffer, fence and blits and the encoders it ends and makes again) is in none of the
  stream's counts. The runtime's work for it cannot be told apart from the program's (a runtime
  may record the barriers a split needs at the program's next command or at the commit), so a
  group that held such a dispatch reports no executed counts: they are added up apart, on the
  report's `split` line, and no frame or total holds them.
- A selector, argument or descriptor key the runtime's backend does not implement is a typed
  refusal naming it; the run stops there and the report keeps what ran. The old runtime's backend
  refuses rate maps, stage inputs, dynamic libraries, OpenCL mode, indirect command buffers,
  accumulated visibility results, and protection options, rotation, `framebufferOnly` and
  `isDrawable` other than their defaults; it does not play `defaultColorSampleCount`, which changes
  nothing a pass shows. It plays a no-copy buffer as a buffer of its length, so one of the
  read-only option (bit 24) is made without it and keeps the GPU writes Metal drops. No device
  here compiles Metal source, so it skips a library from source, whatever a call makes of a
  skipped object (`skipped_objects`), every call of one (`skipped_calls`, with a render encoder's
  `setRenderPipelineState:` of one) and the draws of a render encoder whose pipeline state is one
  (`skipped_draws`).
- The new runtime's backend plays compute, render, blits, `synchronizeResource:`, fences, events
  and queues. What the new runtime does not build yet it skips and counts, and never fakes:
  - objects: libraries from source, and whatever a call makes of a skipped object
    (`skipped_objects`);
  - render pipeline states the runtime answers with a typed gap or the translator refuses
    (`skipped_render_states`), and compute pipeline states the same way
    (`skipped_compute_states`), whose dispatches are skipped (`skipped_dispatches`);
  - parallel render encoders, and render encoders of a pass the runtime does not build
    (`skipped_encoders`): a pass whose setter records leave in it what the runtime's render pass
    descriptor does not hold (a resolve texture, a depth plane other than 0, store action
    options, a visibility result buffer, a render target array length above 1, tile or
    imageblock sizes, or a skipped texture), or one the runtime answers with a typed gap; and
    every call of a skipped object (`skipped_calls`, with `useResource:usage:` and `useHeap:` of
    one);
  - dispatches whose pipeline state or bound buffer or texture is a skipped object
    (`skipped_dispatches`), and draws whose pipeline state, or a buffer or texture bound to
    either stage, or whose index buffer is one (`skipped_draws`);
  - render commands the runtime answers with a typed gap, a draw it does not build among them
    (`skipped_render_commands`, and a line for each gap with its first frame and call);
  - render encoders' sampler binds of LOD clamps the runtime answers with its typed gap (clamps
    whose samples read otherwise than the sampler's own: `skipped_clamped_samplers`), the draws
    while the slot holds one skipped with them; a bind it takes plays with its clamps;
  - blit commands that name a skipped object (`skipped_blits`), and by kind:
    `synchronizeTexture:slice:level:` (`skipped_texture_syncs`), and the pieces of a copy from
    the guest's RAM that are not whole 4-byte words at a 4-byte offset, into a buffer with no
    mapping (`skipped_guest_ram_writes`). `optimizeContentsForGPUAccess:`, a layout hint, plays
    as nothing, and `generateMipmapsForTexture:` as the runtime's `generate_mipmaps`.

  A buffer-to-buffer copy plays as the runtime's (`copy_from_buffer_to_buffer`). A copy from the
  guest's RAM, the recording's plumbing, plays as its pieces' bytes in a Shared staging place of
  the backend's own, then a buffer copy into the target in the same blit encoder, so the GPU's
  order holds; a piece's edge bytes up to whole words come from the target's mapping where it has
  one. The staging place is given back at the command buffer's commit and free once it completes
  (`staged_guest_ram_bytes`). A copy back to the guest's RAM, the recording host's own work, plays
  as buffer copies of its runs, each widened to the whole words it touches within the buffer, into
  a Shared place of the backend's own that nothing reads, in the same blit encoder, given back the
  same way (`copied_back_guest_ram_bytes`).

  Textures over a buffer play as the runtime's (`Buffer::new_texture`) and texture views as its
  views (`Texture::new_texture_view`). Where the runtime cannot make a texture the program made,
  the backend makes an ordinary texture of the same descriptor in its place, so that the IDs and
  addresses the program wrote name a live texture, and counts it: a tiled texture over a buffer
  (`substituted_tiled_textures`), and a texture over a buffer or a view the runtime answers with
  a typed gap (`substituted_gap_textures`). A substitute is the recording's artefact, never the
  runtime's, and its texels are not the program's: `substitutes_read` counts the substitutes, and
  the views of one, that a call other than the making of a view named (a binding, an attachment,
  a blit, a descriptor's field, a call on it), and the report's `substitutes read` line gives the
  first few. A play with a substitute read is equal at best where played; one with a substitute
  only made is not touched by it.

  A texture over a surface's plane (section 7) plays as the product's per-shape form makes it: one
  texture of the runtime's made for views (`Device::new_texture_for_views`) for each plane (the
  receiver buffer, the plane, its offset and row pitch) and shape (width, height and bytes a
  texel, and whether shaders may write it), and each recorded surface texture, and each view of
  one, a view of it at the texture's own index. A plain shape, made for a texture whose usage has
  neither shader write nor shader atomic (a storage image's operation) and is not Metal's unknown,
  takes shader read and render-target usage alone, so it views as its own format and that format's
  sRGB sibling; a texture or view of another format over it is refused
  (`surface_format_refusals`), and textures over it that declare pixel-format-view usage are
  counted (`surface_plain_format_views`). A shape shaders may write takes the usage Metal gives a
  texture whose use it does not know (shader atomic too where the texture declares it), so a
  texture of another format of the same bytes a texel shares its texels; it serves a later texture
  of its extent that does not declare it, and a texture that declares it over a plane whose shape
  of its extent has none is another shape of the plane. Textures over the same pages at
  another offset or pitch are another plane: nothing moves between two planes but through the
  buffer, at the recorded copies. The receiver buffer holds the pages' bytes as the stream writes
  them, and every shape is current over its whole extent, by the product's three rules:
  - A new shape is filled from the buffer, once the buffer's dirty shapes are copied into it, by
    blits in a command buffer of the backend's own committed at the making, after every earlier
    commit. The same shape made again is a view of it, with no copy.
  - A shape is dirty once a committed command buffer may have written it, until it is copied into
    the buffer. A copy back to the guest's RAM first copies every dirty shape of the buffer, and
    every shape its command buffer may have written before, into the buffer, in the copy's own blit
    encoder, where the GPU's order holds it.
  - At the end of an encoder that may have written a shape, the write goes into each other shape of
    its plane that a live texture views or that is dirty: the write's texels that the others
    overlap (every two shapes of a plane overlap from its first texel) are copied into the buffer at
    the plane's offset and pitch, then from there into each other shape, in two blit encoders of the
    same command buffer; the plane's other shapes go. A texel the write's rows reach only in part
    keeps its bytes (`surface_straddled`), and an encoder that may have written two shapes of one
    plane takes the one noted last as the plane's write (`surface_unordered_writes`).

  An encoder may have written a shape through a render pass's attachment, a blit's destination, a
  texture bound at a slot that the bound pipeline state's interface writes
  (`texture_accesses`), and, where no interface is at hand (a stage the runtime answers none for, a
  skipped state, a `useResource:usage:`), a texture whose usage lets a shader write it
  (`surface_usage_writes`). A write through an argument buffer with no use named is not seen, and
  not counted. A shape no texture views stays while it is one of its buffer's four most recently
  used such shapes, or dirty. A copy from the guest's RAM into the buffer also copies the bytes into
  every shape they reach, in the copy's own blit encoder, the guest's bytes winning over a shape's
  write not yet in the buffer. A recorded CPU write reaches the texels it covers of every shape
  through the runtime's upload (`Texture::replace_region`), until the buffer's first
  `m2v.externalWriteDone`. That record (a picture landed) makes every shape of its buffer owe a
  fill from the buffer, its own write lost, as the product's shapes owe one after a video command
  buffer: the first command buffer whose encoders use a texture of the buffer fills every owing
  shape of the buffers it uses before its commit, by blits in a command buffer of the backend's
  own committed then, and a picture that another one lands over before any use is never copied
  (`surface_pictures_filled`, `surface_pictures_unfilled`). Another call that names a receiver
  buffer as a buffer is refused, and a texture of a format that reads across a surface's two planes
  (a two-plane YCbCr or XR10 format) is skipped (`skipped_surfaces_across_planes`). The counts
  (`surface_textures`, `surface_shapes`, `surface_shapes_found`, `surface_order_shapes`,
  `surface_shapes_dropped`, `surface_written_back`, `surface_written_back_bytes`,
  `surface_overlap_copies`, `surface_overlap_bytes`, `surface_straddled`,
  `surface_unordered_writes`, `surface_usage_writes`, `surface_declared_writes`,
  `surface_cpu_writes`, `surface_external_writes`, `surface_refills`, `surface_refill_bytes`,
  `surface_refill_gpu_nanos`, `surface_pictures_filled`, `surface_pictures_unfilled`,
  `surface_copies_in`, `surface_copies_out`, `surface_bytes`,
  `surface_format_refusals`, `surface_plain_format_views`) are the player's work, among its
  executed counters; `surface_refill_gpu_nanos` is the GPU time of the refills' command buffers, counted in the frame whose end first sees each
  complete. The report's `surfaces` line gives them with the longest refill command buffer. A
  check of a texture reads its first level and slice as the old runtime's backend lays them out.

  The counts are among its executed counters; a `skipped_` one above zero makes the play no gate
  result, and `report.txt` says so on a `skipped` line. It places buffers of their own, and the GPU
  copies of two-copy Managed buffers, in the runtime's blocks of 16 MiB of either class (a longer
  one in a block of its own), each place its length rounded up to the buffer's recorded granule:
  the largest power of two that divides its recorded `gpuAddress`, from 256 bytes to 64 KiB (256
  bytes where it has none). A program that writes past a buffer's end then writes into the slack
  the buffer had in the recording, not into the next buffer; the backend's `places by recorded
  granule` line counts the places by granule and gives the most bytes of slack held at once. A
  place is free for reuse once every command buffer committed before its buffer's release
  completed. The backend zeroes a block or heap of the stream once at its making (as the live
  system zeroes a heap's memory once: a buffer placed there reads zero where nothing wrote it,
  though the runtime may take a released block's bytes again), gives the runtime back a block or
  heap of the stream at its release (`release_block`), and gives textures and sampler states
  indices from the whole of each index class (section 5): its
  `indices named N: ...` line counts the objects placed at the index the stream named, those
  whose named index the runtime refused, the indices the program reserved, the placements
  that waited for an index's holders' commits, with the time waited, and the sampler states placed
  without waiting at an index whose last sampler state had the same recorded descriptor while its
  commits were in flight (Metal hands an equal descriptor the same sampler state and ID, so work in
  flight reads nothing new there). A placement that owes its index to a command buffer no record
  played so far lets complete (its event waits only later records signal, or it is behind such a
  one on its queue) shows that the recording device gave the index to the new object before that
  command buffer completed, as its index policy does: it takes the index without waiting, as the
  recording did, and its own `indices given while a commit they owe is held N: ...` line counts
  it. The index stays owed to that command buffer. Its readback serves
  Shared and Managed buffers only; a check of anything else is a refusal. The runtime gives a
  compute or render pipeline state no `gpuResourceID`, so each one the program saw is reported on
  an `id` line as `none` and counted (`gap_pipeline_ids`); each skipped object's the program saw
  is reported so too and counted (`skipped_resource_ids`). Nothing in the play reads either, so
  neither counts among the differing IDs. A render pipeline state takes no label and no
  `resourceIndex`, and `defaultColorSampleCount` and `defaultRasterSampleCount` are not played:
  the runtime refuses a pass with no attachment, the one kind they would change. Render encoders
  play `setRenderPipelineState:`, `setDepthStencilState:`, the vertex and fragment stages'
  buffers, offsets, bytes, textures and sampler states, the viewport, scissor, cull mode, winding,
  blend color and stencil references, the draws (indexed or not, with instance counts, base
  vertex and base instance), fences and memory barriers (their stages are the runtime's to
  order). On the new runtime a bind of a range (`withRange:`) of buffers, textures or sampler
  states plays as the single-slot bind of each, from the range's location, on render and compute
  encoders alike.
  Compute encoders on the new runtime play `setStageInRegion:` (the runtime takes it and it
  changes nothing a dispatch of a kernel without a stage-in argument does, the only kernel it
  builds); the old runtime refuses it.

`report.txt` holds one line per item: the line `recorded before the recorder's completion fix: the
stream may differ from what the program did` first where the stream's header lacks bit 1 (section
6), then `range last-frame F (SOURCE); the play ends there`, `range last-frame F (SOURCE);
--past-range yes plays past it` or `range whole (SOURCE)`, where `SOURCE` is `the stream's
sidecar`, `the player's table, stream LABEL` or `no sidecar and no table row`, then
`device NAME runtime old|rt snapshots yes|no`, then
`addresses address_looking_words=N misaligned_address_words=N address_arguments=N
relocated_words=N differing_ids=N: written words play as recorded; NOTES` (the census of the
written words that look like addresses, at a multiple of 8 and 4 past one, which play as the
stream holds them; the address arguments and those moved; `NOTES` `the stream predates the escape
note` where it holds no `m2v.bufferEscaped` record, else `escape notes N`; and the objects
whose ID differs from the recording's, those of the compute pipeline states' typed gap and of
the skipped objects aside: any such object makes the play an error, as a word that names it
names something else in the play),
`completed IDS COUNTS`, `frame N COUNTS`,
`end N wall_nanos=NANOS resident_bytes=BYTES times KEY=NANOS ... live KEY=VALUE ...` for each
frame (the nanoseconds from the frame before's end, or the first call played, to this frame's,
the wait for its work included; the process's resident memory then where the system tells it,
`VmRSS` on Linux; where the frame's wall time went on the player's thread, in nanoseconds:
`stream` reading and decoding the stream and its blobs, `write` putting bytes in the runtime's
memory, `read` reading bytes back, `wait` waiting for command buffers, `resource` the calls that
make or release a buffer, texture, heap or block and `resource_max` the longest of them,
`pipeline` the calls that make a library, a function or a pipeline state and `pipeline_max` the
longest of them, `other` every other call the runtime ran, `upload` the host's copies from the
guest's RAM, which a backend stages and copies (the recording's plumbing, the player's share),
`count` reading the runtime's
counters, and `player` what of the wall time is left, the player's own work; then `gpu_busy`, the
union of the GPU intervals of the command buffers the frame waited for, where the runtime gives
them, which runs beside the rest; and the backend's gauges then, as they stand, never as growth:
the new runtime's live
blocks, buffers, textures, pipelines, kernels, command buffers, spilled bytes, Managed buffers'
spilled bytes (`managed_spilled_bytes`: the bytes of those whose GPU copy the player placed in a
spilled Private block) and idle Shared and
Private bytes the pool keeps, and the bytes
of the blocks the backend holds for the stream by class, `shared_block_bytes` and
`private_block_bytes`; the old runtime's live memory blocks, empty ones kept and dedicated ones),
`devices apart from the guest's: #ID ...` where the stream recorded devices besides the one the
guest makes its objects on (section 4), `check ID SHA256 match|differs|unrecorded|covered|receiver`, `id ID RECORDED PLAYED` for each object whose runtime
gives another `gpuResourceID` than the program saw (both in hex, `PLAYED` `none` where the
runtime's getter raises), `id ID RECORDED PLAYED word WORD` on the new runtime, which lays its IDs
out as [RESOURCE-ID.md](RESOURCE-ID.md) does: there the ID's low word, its index, is compared with
the recording's, and its high word with `WORD`, the document's own function of the descriptor the
object was made of (a texture's `TextureWord`, a sampler state's `SamplerWord`), never with the
recording's high word, which is the recording runtime's own; `private ID DISPATCH LOCATION+LENGTH match|differs N/WORDS|covered|receiver` for each
`m2v.privateBytes` range compared, `covered N: texture writes over other objects of a placement
heap; C checks and R snapshot ranges they covered not compared; B covers taken off by a binding,
which may be a read` where a texture write covered
bytes of another object the stream placed in its placement heap (`N` the times one did): Metal
leaves an aliased resource's contents undefined once another resource is written over its
memory, so from that write until the object is written again its check (`covered`) and the
snapshot ranges that meet those bytes (`covered`) are not compared. The ranges are the stream's
offsets in the heap with a buffer's length and a texture's size as the runtime answers it, and
the writes are found by structure: a draw's or a dispatch's to the texture slots the pipeline
state's interface marks written (the new runtime's; the old runtime answers no slot access, so
there only the following count), a render pass's to the textures its descriptor names, a blit's
to its destination, `generateMipmapsForTexture:`, a `useResource:usage:` with write usage and the
CPU's `replaceRegion:…`; a write of a texture over a buffer is the buffer's. An object is written
again by the player's writes into it, a blit into it, and, as the stream carries no buffer
access, any binding of it to an encoder, a `useResource:usage:` with write usage or a `useHeap:`
of its heap; a binding may be a read, so `B` counts the covers a binding or a `useHeap:` took off,
and a difference in that object's later compares may be the covering texture's bytes. `receivers:
C checks and R snapshot ranges of surfaces' receiver buffers not compared; the textures over the
surfaces hold the texels` where the stream checks a buffer a texture over a surface's plane was made
on (section 7), while that buffer lives: a runtime may keep the texels apart from the buffer, as the
product does, so such a check (`receiver`) and the snapshot ranges in such a buffer (`receiver`) are
compared on no runtime, and the textures' own checks carry the bytes. `split N executed COUNTS` where the player split compute
encoders around them (`N` the encoders split, `COUNTS` the executed counts of their groups),
`unsnapshotted N` where the backend took none of them (`N` the dispatches played as they are),
`parted N` where a recorded range met a live one in part (`N` the older ranges ended),
`pinned N of N; K heaps' textures outside; S textures whose recorded range (by the runtime's
texture sizes) met a live buffer's` on the old runtime (`N` the objects made at their recorded
addresses, every other object with a recorded range there too, so the address map moves no word;
`K` the heaps with a recorded range whose textures it made in a heap outside the space, those the
stream places a texture or no buffer in; `S` the textures whose range in the recording, by the
runtime's size for each, met the recorded range of a buffer live in the same heap, read ahead and
not a stop; section 5),
`relaid HEAPS moved OFFSETS grown BYTES` where it laid placement heaps out again (the heaps, the
offsets it moved in them and the bytes it added to their sizes), `texture_writes ...` (the
CPU's writes into Managed textures, below), `total COUNTS` (its `calls` the
calls played: where
the play stopped, the stopping call is the next), the new runtime's backend's
`pipeline I wall_nanos=N kernel=new|again` for each compute pipeline state it made, in order (the
making's wall time, and whether no state of another specialization of the kernel, a library's
function by name, was made before), its `skipped render states GAP N, the first #ID: WHY` for each
typed gap a render pipeline state met (`GAP` the runtime's gap as Rust prints it for debugging, such
as `RenderPipeline { kind: DepthAttachment }`, or `Translation` where the translator refused it, `N`
the states skipped, `ID` the first one's object and `WHY` the runtime's answer), its
`skipped compute states GAP N, the first #ID: WHY` the same for compute pipeline states, its
`skipped render commands GAP N, the first at frame F, call C` for each typed gap a render command
met, a draw the runtime does not build among them (`GAP` the runtime's gap as Rust prints it for
debugging, such as `Draw { kind: Points }`, `N` the commands skipped, `F`
and `C` the first one's frame and stream position), its `skipped render encoders KIND N, the first
at frame F, call C` for each kind of render encoder it does not build (`KIND`
`ParallelRenderCommandEncoder`, `Unheld(KEYS)` for a pass whose descriptor sets properties `KEYS`
the runtime's does not hold, or the runtime's gap for the pass as Rust prints it for debugging),
`cut SEGMENT OFFSET` where the last segment ends
in a record cut short, `skipped KEY=VALUE ...: no gate result` where the backend skipped
anything, `gaps KEY=VALUE ...` where it met a typed gap (`gap_pipeline_ids`: the compute and
render pipeline states whose `gpuResourceID` the player asked, which the new runtime does not
give; bookkeeping: no draw or dispatch is skipped for it, and only an indirect command buffer,
which the runtime does not build, would read such an ID),
then `ok`, `ids: played to the end; N objects' gpuResourceID differ from the recording's` where
the play ended but objects whose ID the program saw have other IDs in the runtime, or
`error WHAT`. Where the backend skipped anything, the `skipped` line and the last
one end `; not a verdict past frame F, call C`: `F` and `C` the frame and stream position of the
call at which it first skipped, past which nothing the play shows is a verdict. A dispatch is
skipped only where it names a skipped object. `COUNTS` is `stream KEY=VALUE ... executed
KEY=VALUE ...`. On the new runtime the player then ends its device and prints, for each of the
runtime's gauges read once the device dropped, `M2V-MATRIX gauge replay-rt/device/GAUGE@drop N`:
the matrix's own record, so the matrix's rule (`scripts/matrix/results.py`) fails a play whose
gauge is not 0 at device drop. Such a gauge, or a device that did not quiesce within 30 seconds,
also writes `error teardown: WHAT`. The old runtime's counts end with its device, so its plays
print no such line. The exit status is 0 for `ok`, 1 for a run that stopped or an `ok` followed
by `error teardown`, 2 for a harness failure (a bad command line or an unreadable stream) and 3
for `ids`, a run that played to the end.

With `--dump-command-buffers FIRST..LAST --dump-dir DIR`, the player also writes, once each
command buffer whose stream id lies in `FIRST..=LAST` completed (a split command buffer's parts by
their own ids), every buffer its encoders named, read after the wait that saw the command buffer
complete and before the next call plays: `cbC-objO.bin` its bytes, `cbC-objO.cover` the ranges of
it covered then (as above, `start..end` byte offsets joined by commas, or `-`), and
`cbC-objO.unread` in place of the bytes where the backend could not read it. `command-replay
compare-dumps --a DIR --b DIR` compares two plays' dumps word by word (4 bytes), leaving out a word
wholly inside a range either play held covered (the old runtime answers no slot access, so its
plays cover less), and prints `differs cb #C obj #O: N words outside every cover, the first word
W; M covered; lengths A and B` for each buffer that differs outside every cover, `only in a|b:
cbC-objO` for a dump one play lacks, `unread in a|b: cbC-objO: WHY`, then `compared N buffers: D
differ, U words outside every cover, M covered; O dumps in one play only, R unread: agree|DIFFER`.
Its exit status is 0 for `agree`, 1 for `DIFFER` and 2 for a bad command line or an unreadable
dump. This compare says whether two plays of a range of command buffers agree; a whole-object
checksum does not, as a buffer partly under a cover would differ under it and a difference outside
the cover would show only as the same mismatch.
