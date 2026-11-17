# Replay streams (format version 2)

A replay stream records, in order, what Metal programs asked of their devices: every call with its
objects and arguments, the bytes the CPU left in GPU-visible memory before each commit, and the GPU
addresses and resource IDs the programs saw. `command-replay play` runs a stream on a Vulkan device
through a runtime, with no guest, and reports deterministic metrics and checksums, so two runs, two
devices or two runtimes compare.

Records use Metal's own selectors, raw enum values and descriptor property names. This page is all
a reader needs. The reader, the writer and the player live in `crates/metal2vulkan-validation`
(`src/replay/`).

A stream recorded from a consumer's guest is the consumer's data: it stays on the hosts that may
hold guest data and is never committed. Tests write their own streams.

## 1. Container

- **File:** a 16-byte header, then records. The header is ASCII `M2VRSTRM`, a u32 version (`2`)
  and a u32 of flags (`0`). Fixed-width values are little-endian.
- **Record:** a kind (u8), the payload's length (varint), the payload. Varints are unsigned LEB128
  of at most 10 bytes; signed values are zigzag-encoded.
- **Append-only.** A recorder only appends whole records and flushes at least at every commit. A
  record cut short at the end of the last segment is a truncated tail: the reader reports it, and
  everything before it plays.
- **Segments.** A stream may be split at any record boundary into files `NAME.0000.m2vr`,
  `NAME.0001.m2vr`, ..., each starting with the header; the stream is the segments in order, and
  names and blobs carry across them. A record cut short in any segment but the last is an error.
- **An unknown record kind** is an error naming the kind and its offset.

| kind | record | payload |
|---|---|---|
| 1 | name | UTF-8 text. The n-th name record of the stream defines name id n, from 1 |
| 2 | blob | the data's SHA-256 (32 bytes), then the data. Each hash at most once per stream, before its first use |
| 3 | call | section 3 |

## 2. Values

Each value starts with a one-byte tag.

| tag | type | encoding | used for |
|---|---|---|---|
| 0 | nil | none | nil; an argument not recorded |
| 1, 2 | false, true | none | `BOOL` |
| 3 | uint | varint | unsigned integers, raw enums and option sets, resource IDs (never rewritten) |
| 4 | sint | zigzag varint | signed integers |
| 5 | f32 | 4 bytes | `float` |
| 6 | f64 | 8 bytes | `double`, `CGFloat` |
| 7 | object | varint id | Metal objects, surfaces |
| 8 | address | varint | a GPU address: relocated (section 5) |
| 9 | string | varint length, UTF-8 | `NSString` |
| 10 | bytes | varint length, data | data, inline |
| 11 | blob | 32-byte SHA-256 | data, by content |
| 12 | list | varint count, values | structs and C arrays |
| 13 | map | varint count, (name id, value) pairs | descriptors; a key at most once |

Structs are lists in Metal's field order: `MTLOrigin` `[x, y, z]`, `MTLSize` `[w, h, d]`,
`MTLRegion` `[x, y, z, w, h, d]`, `NSRange` `[location, length]`, `MTLScissorRect`
`[x, y, w, h]`, `MTLViewport` six f64, `MTLClearColor` four f64, `MTLSamplePosition` two f32,
`MTLTextureSwizzleChannels` four uints. A C array and its count are one list.

## 3. Calls

The payload is: the selector (name id), the receiver (object id, 0 for none), the result (the id
of the object the call makes, 0 for none), the argument count and the arguments, the answer count
and the (name id, value) answer pairs.

- **Selector:** Metal's Objective-C selector as Metal's headers spell it
  (`dispatchThreadgroups:threadsPerThreadgroup:`), a C function's name
  (`MTLCreateSystemDefaultDevice`, `IOSurfaceCreate`, `IOSurfaceLock`, `IOSurfaceUnlock`), or an
  `m2v.` name for what Metal has no call for. Where two classes share a selector, the receiver's
  kind tells them apart.
- **Arguments:** one value per argument of the selector, in order. Error out-parameters, blocks and
  handlers are left out: their effects reach the stream as later calls and writes. A pointer whose
  bytes the call does not consume (the pointer of `newBufferWithBytesNoCopy:...`, the destination
  of `getBytes:...`) is nil.
- **Answers:** what the call returned that the program could see (section 5). A call that failed
  in the recording has result 0 and the answer `error` (a string). A player ignores answer keys it
  does not use.
- Getters are not recorded.

## 4. Objects

- **Ids** are nonzero u64s, unique across the whole stream and never reused, so an object that
  several processes use (a surface's buffer, a shared event) has one id.
- An object is made by the call whose result names it and dies at `release` (receiver: the id;
  its last reference). Releasing a device releases everything it made.
- **Devices:** `MTLCreateSystemDefaultDevice`, one per device the recording's host made; every
  guest process may share one. The player serves every recorded device from one runtime device.
- Everything else is made by its Metal selector: queues, command buffers, encoders of every kind,
  buffers, heaps, textures and views, samplers, depth and stencil states, events, shared events,
  fences, indirect command buffers, libraries, functions, pipeline states, argument encoders and
  visible function tables. Descriptors are maps (section 9).

## 5. Memory, addresses and resource IDs

- **Blocks.** `m2v.newBlockWithLength:storageMode:` (receiver: the device; arguments: the length
  and the `MTLStorageMode`) makes a range the guest sub-allocates in, and answers `gpuAddress` and
  `gpuLength`. A buffer in a block is `m2v.newBufferInBlock:offset:length:options:` (arguments: the
  block, the offset, the length, the `MTLResourceOptions`), which answers `gpuAddress` and
  `gpuLength`.
- **Other buffers and heaps** answer `gpuAddress` and `gpuLength` on their making call:
  `newBufferWithLength:options:`, `newBufferWithBytes:length:options:`, `newHeapWithDescriptor:`
  and a heap's placement calls. `newBufferWithBytesNoCopy:length:options:deallocator:` is
  `[nil, length, options, nil]`; its bytes arrive as writes, and it answers `readOnly` true where
  its pages were read-only.
- **Textures** answer `gpuResourceID`, and so do sampler states, indirect command buffers,
  pipeline states made for indirect command buffers and visible function tables. A texture over a
  buffer is the buffer's `newTextureWithDescriptor:offset:bytesPerRow:`.
- **The address map** is the ranges `[gpuAddress, gpuAddress + gpuLength)` of the live objects that
  answered them; `release` removes an object's range.
- **Relocation.** The player makes each object where its runtime has room. In the bytes it puts
  into an object that answered a `gpuAddress` (`m2v.write`, `m2v.copyBytes:toBuffer:offset:`,
  `m2v.copyFromGuestRam` and `m2v.copyFromGuestRamAt`), in the data of every
  `set*Bytes:length:atIndex:` and in every address value, each 8-byte word at an offset that is a
  multiple of 8 from the target's start, whose value lies in a live range, becomes the player's
  address of that range's object plus the word's offset into the range. The innermost range wins:
  a buffer in a block moves with its own new address. Words outside every live range stay as they
  are. Bytes put into an object with no address (a surface's pages or device copy), the data of
  `replaceRegion:...`, library bytes and function constants are never rewritten. The player counts the words it rewrote; a data word that is no address but
  lies in a live range is rewritten too, and the count shows it.
- **Resource IDs** are reproduced: the player makes each object at its recorded `gpuResourceID`
  and refuses the stream where the runtime gives another.

## 6. CPU writes

- `m2v.write` (receiver: a block or a buffer, a surface's pages included; arguments: one or more
  pairs of an offset and its data, bytes or blob): the bytes the CPU left at those offsets.
- **Before each `commit`**, the stream holds a write for every byte that changed since it was last
  recorded, in the memory the command buffer can reach: its bound buffers, textures over buffers
  and argument buffers; its `useResource:` and `useHeap:` resources; its attachments, blit operands
  and executed indirect command buffers. Memory reached only through pointers the command buffer
  did not declare is undefined in Metal and outside this format.
- **The comparison point** is the bytes as last recorded. It is not refreshed when a command buffer
  completes, except for ranges only the GPU writes (attachments, blit destinations).
- **No half-written bytes.** Before capturing, a recorder waits for every earlier command buffer
  that can run: one not held on an event wait that a later record satisfies. The bytes may then
  include final bytes the GPU wrote; the player puts them back at the same point, after its own
  wait, so they change nothing.
- Private memory never appears in a write. A Managed buffer's changed range is a write on the
  buffer, immediately before its `didModifyRange:`.
- Granularity is the recorder's: whole pages or changed runs. Data of 64 bytes or more is best a
  blob, so a repeated page is stored once.

## 7. Surfaces

A surface is recorded as the host's runtime sees it:

- **Its pages** are a no-copy buffer (`newBufferWithBytesNoCopy:length:options:deallocator:`), whose
  CPU changes arrive as writes at each commit that reaches it (section 6).
- **A texture over a plane** is `newTextureWithDescriptor:iosurface:plane:` with that buffer as its
  receiver and the arguments `[descriptor, plane, offset, bytesPerRow, facts]`: the plane's first
  byte and row pitch in the buffer, and a map of what the texture reads of the surface besides, or
  nil (`PixelFormat` as its four characters big-endian, `PlaneCount`, `Width`, `Height`,
  `ChromaOffset`, `ChromaBytesPerRow`, `YCbCrMatrix` as `601`, `709` or `2020`).
- **A device copy** of a surface is an ordinary buffer. Its fill from the CPU side is
  `m2v.copyBytes:toBuffer:offset:` on a blit encoder (`[data, buffer, offset]`, the data taken at
  the call); its copy back to the CPU side is `m2v.copyToHost:` on a blit encoder (`[buffer,
  offset, length]`, no data), where the guest's lock asked for it.
- **Copies between the guest's RAM and a buffer** are blit-encoder commands, ordered against the
  encoder's others as a buffer-to-buffer copy is. A run is `[start, end]`, a range of host
  addresses in the host's mapping of the guest's RAM, which the player never reads; only its
  length counts. A copy the host did not encode is not recorded.
  - `m2v.copyFromGuestRam` (`[runs, buffer, offset, data]`): the runs' bytes, in order, land in the
    buffer from the offset. `data` is those bytes as the copy read them, bytes or blob, as long as
    the runs together.
  - `m2v.copyFromGuestRamAt` (`[placed runs, buffer, data]`): each placed run, `[at, run]`, lands
    its bytes in the buffer from byte `at`. `data` is the runs' bytes in the order given.
  - `m2v.copyToGuestRam` (`[buffer, placed runs]`): the buffer's bytes from each run's `at`, as long
    as its run, go back to the guest's RAM. It carries no data; the player copies them into memory
    of its own that nothing reads.
- `IOSurfaceCreate`, `IOSurfaceLock` and `IOSurfaceUnlock` are kept for a host whose runtime makes
  surfaces device images synced at lock and unlock: `IOSurfaceCreate` (receiver 0) takes the
  `kIOSurface` properties as a map, each key without its prefix, and the lock calls their
  options.

## 8. Commands, commits and frames

- **Encoder commands** are Metal's selectors on the encoder: every render, tile, compute, blit and
  resource-state call. An indirect command buffer's contents are calls on it named
  `MTLIndirectRenderCommand.<setter>` and `MTLIndirectComputeCommand.<setter>`, whose first
  argument is the command's index.
- **Uploads and downloads:** `replaceRegion:...` carries its bytes; `didModifyRange:`,
  `synchronizeResource:` and `synchronizeTexture:slice:level:` sit where they were made;
  `getBytes:...` has a nil destination and may answer `sha256`.
- **Commits:** the order of `commit` records is the global commit order. A command buffer's queue is
  the receiver of the call that made it; each queue keeps its own order, and an event wait may hold
  one queue while the others go on.
- **Synchronization:** `encodeSignalEvent:value:` and `encodeWaitForEvent:value:` on a command
  buffer; fences as encoder calls; `setSignaledValue:`; and `waitUntilCompleted` and
  `waitUntilSignaledValue:timeoutMS:` where the program's CPU waited.
- **The host's own commit splits** (no Metal counterpart): `m2v.split` commits its command buffer
  as a part and makes the command buffer of the same queue that goes on from it (its result);
  `m2v.part` makes a command buffer of the same queue for a later range (its result), and
  `m2v.commitPart` commits its receiver as a part. `m2v.fold`, `m2v.setCompletionJoinsNext:`,
  `m2v.holdsNoPredication` and `m2v.rebind` are recorded where the host made them; the last
  three change nothing a program sees.
- `m2v.present` (arguments: the texture shown, nil where the stream never made it) is recorded
  where the device's display shows a guest present, with or without a window. It ends a frame:
  the player waits for what can complete and closes the frame's counters; it reads nothing.
- `m2v.check` (arguments: objects) asks the player to read those objects there and report their
  checksums. Its optional answer `sha256` is a list of the recording's own, one per object.

## 9. Descriptors and libraries

- **Descriptors** are maps keyed by Metal's property names; a property at its default may be left
  out. An array property is a list by Metal's index, nil where unset. Functions and textures in a
  descriptor are objects. Function constant values are a list of `[index, MTLDataType, bytes]`.
- **Libraries:** `newLibraryWithData:error:` takes the metallib as a blob: the SHA-256 of the exact
  bytes passed (for a universal file that is not the runtime's `Library::content_hash`, which
  hashes the AIR slice). The blob record may be left out; the player then reads the data from its
  blob directory (`--blobs DIR`, one file per blob named by its lowercase hex hash) and checks it
  against the hash.

## 10. The player

`command-replay play --stream FILE --out DIR --device-type cpu|gpu --shader-cache DIR
[--blobs DIR]` plays a stream (its first segment) on one runtime device and writes
`DIR/report.txt`. `command-replay dump --stream FILE` prints the call records as text.

- It relocates and reproduces resource IDs (section 5).
- After each commit, before each write, check and present, and where the program waited, it waits
  for every committed command buffer that can complete: one whose event waits are met by values the
  CPU set or by signals of command buffers that can complete. It never waits on a command buffer
  held behind an event wait.
- It reports, for each group of command buffers it waited for (normally the one just committed),
  for each frame and in total, the counters the runtime exposes (submissions, render passes,
  barriers, dispatches, draws, copies, pipelines made), the words it relocated, the 8-byte values
  at offsets 4 past a multiple of 8 that lie in a live range (left as they are), and the bytes it
  wrote or copied in (`m2v.write`, `m2v.copyBytes:toBuffer:offset:` and the copies from guest RAM).
  Its own readbacks are not counted.
- `m2v.setCompletionJoinsNext:`, `m2v.holdsNoPredication` and `m2v.rebind` change nothing it plays.
- A selector, argument or descriptor key the runtime's backend does not implement is a typed
  refusal naming it; the run stops there and the report keeps what ran.

`report.txt` holds one line per item: `device NAME`, `completed IDS KEY=VALUE ...`,
`frame N KEY=VALUE ...`, `check ID SHA256 match|differs|unrecorded`, `total KEY=VALUE ...`,
`cut SEGMENT OFFSET` where the last segment ends in a record cut short, then `ok` or
`error WHAT`. The exit status is 0 for `ok`, 1 for a run that stopped and 2 for a harness
failure (a bad command line or an unreadable stream).
