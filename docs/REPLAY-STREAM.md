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
  and a u32 of flags (`0`). Fixed-width values are little-endian.
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
| 7 | object | varint id | Metal objects |
| 8 | address | varint | a GPU address: relocated (section 5) |
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
- **An optional field** is nil where it is unset. One set to nil reads the same: a player leaves
  the property as it was either way.
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
- **Arguments:** one value per argument of the selector, in order; an encoder command's are its
  fields. Error out-parameters, blocks and handlers are left out: their effects reach the stream as
  later calls and writes.
- **Answers:** what the call returned that the program could see (section 5). A player ignores
  answer keys it does not use.
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
  visible function tables, and the descriptors Metal makes as objects (section 9).
- **The host's own plumbing** is recorded as `m2v.` records that stand for a plain Metal call; a
  player plays each as that call:

| record | arguments | what the host did | plays as |
|---|---|---|---|
| `m2v.newBufferOfLengthPaddingUnobserved` | `[length, options, alignedByte]` | `newBufferWithLength:options:`, whose texture rows may keep or change the bytes between them; `alignedByte` (or nil) is the buffer's byte the host put on a 256-byte boundary | `newBufferWithLength:options:` `[length, options]` |
| `m2v.newExportableBufferOfLength` | `[length, options, alignedByte]` | the same buffer in an allocation of its own, which another API imports; result 0 where the host made none | the same, and nothing at result 0 |
| `m2v.newDmaBufBufferOfLengthPaddingUnobserved` | `[length, options, alignedByte]` | the same, also exported as a dma-buf for video decoding; the video's writes arrive as a later commit's writes (section 6); result 0 where the host made none | the same, and nothing at result 0 |
| `m2v.newCommandQueueUnshared` | `[]` | a queue the host kept apart | `newCommandQueue` |

## 5. Memory and addresses

- **Blocks.** `m2v.newBlockWithLength:storageMode:` (receiver: the device; arguments: the length
  and the `MTLStorageMode`) makes a range the guest sub-allocates in, and answers `gpuAddress` and
  `gpuLength`. A buffer in a block is `m2v.newBufferInBlock:offset:length:options:` (arguments: the
  block, the offset, the length, the `MTLResourceOptions`), which answers `gpuAddress` and
  `gpuLength`.
- **Other buffers and heaps** answer `gpuAddress` and `gpuLength` where the host pinned them at an
  address: `newBufferWithDescriptor:`, `newHeapWithDescriptor:`, and a heap's
  `newBufferWithLength:options:offset:` (`[length, options, offset]`). An object the host made
  without a pinned address answers none. `newBufferWithBytesNoCopy:length:options:deallocator:`
  is `[nil, length, options, nil]`, and a no-copy descriptor's `contents` nil; their bytes arrive
  as writes.
- **Textures over memory:** a buffer's `newTextureWithDescriptor:offset:bytesPerRow:` and
  `newTiledTextureWithDescriptor:offset:bytesPerRow:`, a heap's `newTextureWithDescriptor:offset:`,
  each taking a texture descriptor object (section 9).
- **The address map** is the ranges `[gpuAddress, gpuAddress + gpuLength)` of the live objects that
  answered them; `release` removes an object's range.
- **Relocation.** The player makes each object where its runtime has room. In the bytes it puts
  into an object that answered a `gpuAddress` (`m2v.write` and the copies from the guest's RAM), in
  the data of every `set*Bytes:length:atIndex:` and in every address value, each 8-byte word at an
  offset that is a multiple of 8 from the target's start, whose value lies in a live range, becomes
  the player's address of that range's object plus the word's offset into the range. The innermost
  range wins: a buffer in a block moves with its own new address. Words outside every live range
  stay as they are. Bytes put into an object with no address, the data of `replaceRegion:...`,
  library bytes and function constants are never rewritten. The player counts the words it
  rewrote; a data word that is no address but lies in a live range is rewritten too, and the count
  shows it.
- **Resource IDs** are not answered. A descriptor's resource index places its object at that ID, as
  Metal places it whether or not the index is forced: a texture descriptor's and a render
  pipeline's `resourceIndex` (`[index, forced]`), a sampler's (an index), and a view's
  `resourceIndex:` argument. Every other object takes the ID its runtime gives it.

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
  one queue while the others go on.
- **Synchronization:** `encodeSignalEvent:value:` and `encodeWaitForEvent:value:` on a command
  buffer; fences as encoder calls; `setSignaledValue:`; and `waitUntilCompleted` and
  `waitUntilSignaledValue:timeoutMS:` where the program's CPU waited.
- **The host's own commit splits** (no Metal counterpart): `m2v.split` commits its command buffer
  as a part and makes the command buffer of the same queue that goes on from it (its result);
  `m2v.part` makes a command buffer of the same queue for a later range (its result), and
  `m2v.commitPart` commits its receiver as a part. A split or part of result 0 is one the host did
  not cut: it commits nothing and makes nothing.
- **The host's scheduling records** change nothing a program sees, and a player plays none of
  them: `m2v.setCompletionJoinsNext` (`[joins]`: the host groups the command buffer's completion
  with the next ones of its queue), `m2v.holdsNoPredication` (`[]`), `m2v.fold` (`[later]`: the
  command buffer takes in how the GPU fails the later one's encoders) and `m2v.rebindNoCopy`
  (`[bytes, deallocator]`, recorded `[nil, nil]`: a no-copy buffer moved to new host memory, whose
  bytes arrive as writes).
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
    `writeSwizzleEnabled`, `allowGpuOptimizedContents`, `rotation`.
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
    nothing: the descriptor keeps what it had.
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
  - `newHeapWithDescriptor:`: `{size, heapType, options, pinnedGpuAddress}`.
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
    supportIndirectCommandBuffers, vertexLinked, fragmentLinked, privateState}`. The vertex
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
| `front`, `back` | `frontFaceStencil`, `backFaceStencil` |
| `heapType` | `type` |
| `options` | the setters of `resourceOptions`, `storageMode` and `hazardTrackingMode` |
| `pinnedGpuAddress`, `contents`, `noCopy`, `deallocator`, `zeroFrom` | the host's buffer creation: a pin, no-copy host memory and its deallocator, and the bytes known zero from an offset |
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
[--blobs DIR]` plays a stream (its first segment) on one runtime device and writes
`DIR/report.txt`. `command-replay dump --stream FILE` prints the call records as text, and
`command-replay shapes --stream FILE` each selector's argument shapes, with no data.

- It relocates (section 5) and plays the host's plumbing as the plain calls it stands for
  (section 4).
- After each commit, before each write, check and present, and where the program waited, it waits
  for every committed command buffer that can complete: one whose event waits are met by values the
  CPU set or by signals of command buffers that can complete. It never waits on a command buffer
  held behind an event wait. Each wait is bounded under the validation crate's hang protocol.
- It reports, for each group of command buffers it waited for (normally the one just committed),
  for each frame and in total, the counters the runtime exposes (submissions, render passes,
  barriers, dispatches, draws, copies, pipelines made), the words it relocated, the 8-byte values
  at offsets 4 past a multiple of 8 that lie in a live range (left as they are), and the bytes it
  wrote or copied in (`m2v.write` and the copies from the guest's RAM). Its own readbacks are not
  counted.
- A selector, argument or descriptor key the runtime's backend does not implement is a typed
  refusal naming it; the run stops there and the report keeps what ran. The old runtime's backend
  refuses rate maps, stage inputs, dynamic libraries, OpenCL mode, indirect command buffers,
  accumulated visibility results, and protection options, rotation, `framebufferOnly` and
  `isDrawable` other than their defaults; it does not play `defaultColorSampleCount`, which changes
  nothing a pass shows.

`report.txt` holds one line per item: `device NAME`, `completed IDS KEY=VALUE ...`,
`frame N KEY=VALUE ...`, `check ID SHA256 match|differs|unrecorded`, `total KEY=VALUE ...`,
`cut SEGMENT OFFSET` where the last segment ends in a record cut short, then `ok` or
`error WHAT`. The exit status is 0 for `ok`, 1 for a run that stopped and 2 for a harness
failure (a bad command line or an unreadable stream).
