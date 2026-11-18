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
| 7 | object | varint id | Metal objects; object 0, which no object is, a property set to nil (section 9) |
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
  (section 10). A stream without it says nothing of that.
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
- **Textures over memory:** a buffer's `newTextureWithDescriptor:offset:bytesPerRow:` and
  `newTiledTextureWithDescriptor:offset:bytesPerRow:`, a heap's `newTextureWithDescriptor:offset:`,
  each taking a texture descriptor object (section 9).
- **The address map** is the ranges `[gpuAddress, gpuAddress + gpuLength)` of the live objects that
  answered them; `release` removes an object's range. A range that meets a live one in part (the
  program put a newer object in memory an older one still held, as a guest's own allocator in a
  block may) ends the older range, whose inner ranges stay: a word there is the newer object's.
  The player's report counts the ranges it ended (`parted`).
- **Relocation.** The player makes each object where its runtime has room. In the bytes it puts
  into an object that answered a `gpuAddress` (`m2v.write`, a copied buffer's `contents` and the
  copies from the guest's RAM), in the data of every `set*Bytes:length:atIndex:` and in every address value, each 8-byte word at an
  offset that is a multiple of 8 from the target's start, whose value lies in a live range, becomes
  the player's address of that range's object plus the word's offset into the range. The innermost
  range wins: a buffer in a block moves with its own new address. Words outside every live range
  stay as they are. Bytes put into an object with no address, the data of `replaceRegion:...`,
  library bytes and function constants are never rewritten. The player counts the words it
  rewrote; a data word that is no address but lies in a live range is rewritten too, and the count
  shows it.
- **Resource IDs.** A descriptor's resource index places its object at that ID, as Metal places it
  whether or not the index is forced: a texture descriptor's and a render pipeline's
  `resourceIndex` (`[index, forced]`), a sampler's (an index), and a view's `resourceIndex:`
  argument. Every other object takes the ID its runtime gives it.
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
[--until-frame F]` plays a stream (its first segment) on one device through the old runtime (the
default) or the new one (`rt`), and writes `DIR/report.txt`. It writes each frame's `frame` and
`end` lines as the frame ends, flushed, so a play that never ends leaves every frame it finished.
With `--snapshots no` it takes no `m2v.privateBytes` snapshot on either runtime: each such dispatch
plays as it is, counted on the `unsnapshotted` line, so two runtimes' plays do the same work and
their times and memory compare. With `--until-frame F` it stops after frame `F`'s present (frames
counted from 0), completes the work committed so far, reports as at the stream's end and says
`until-frame F`. `command-replay dump --stream FILE` prints the call records as text, and
`command-replay shapes --stream FILE` each selector's argument shapes, with no data.

- It relocates (section 5) and plays the host's plumbing as the plain calls it stands for
  (section 4).
- **Managed buffers** (the new runtime) play in the form the recording device kept them: its
  `m2v.deviceIdentity` record's `hasUnifiedMemory`, or where the stream has none, `--managed`
  (`one-copy` for unified memory, `two-copy` for a GPU copy). With neither, a Managed buffer stops
  the play with an error that says so. One copy: every Managed buffer plays as a Shared buffer, on
  every device. Two copies: it plays as the runtime's Managed buffer, whose GPU copy starts from
  the bytes it was made with: a copied buffer's bytes, or a no-copy one's bytes at its making
  (section 5). A stream with no identity record gives a no-copy buffer none, so its GPU copy takes
  the writes recorded before its first use (a commit of a command buffer one of whose encoders
  names it), the nearest record the stream has of its bytes at its making. On a device with
  unified memory the runtime keeps one copy anyway. The report's `managed` line says which form
  played and from what, counts the Managed buffers and the no-copy ones whose contents came from
  first use, not from create, and says where a two-copy recording played as one copy.
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
- After each commit, before each write, check and present, and where the program waited, it waits
  for every committed command buffer that can complete: one whose event waits are met by values the
  CPU set or by signals of command buffers that can complete. It never waits on a command buffer
  held behind an event wait. Each wait is bounded under the validation crate's hang protocol.
- It reads the `m2v.privateBytes` records (section 8) ahead, before it plays the stream, and plays
  each dispatch they name with its ranges' bytes before put in and its bytes after compared: it
  puts the bytes before (relocated as a write's are) in a Shared buffer of its own; ends the
  dispatch's compute encoder; copies them into the ranges in a blit encoder; makes the compute
  encoder again by the call that made it, with every `set…` and `use…` call the program made on
  it; dispatches; then ends it, copies the ranges out in a blit encoder and makes it again the
  same way. A fence of its own orders each of these encoders after the one before it. Once the
  command buffer completed, it reads the bytes after back and counts the 4-byte words of each
  range that differ from the recording's, relocated the same way. A difference is reported, and
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
  its offset, and the play meets the runtime's refusal where it lies.
- It reports, for each group of command buffers it waited for (normally the one just committed),
  for each frame and in total, two sets of counts:
  - **Fixed by the stream** (`stream`), the same on every play of the stream on any runtime: the
    calls played (`calls`), the command buffers committed (`commits`, a host's split included),
    the encoders made (`encoders`: the objects a command buffer or an encoder makes), the draws
    and dispatches their calls issued (`draws` and `dispatches`: an encoder's calls whose
    selectors begin `draw` and `dispatch`), the words it relocated (`relocated_words`), the
    8-byte values at offsets 4 past a multiple of 8 that lie in a live range, left as they are
    (`misaligned_address_words`), and the bytes it wrote or copied in, `m2v.write`, the copied
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

  Its own readbacks are not counted. Its own work around a dispatch an `m2v.privateBytes` record
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
- The new runtime's backend plays compute, blits, `synchronizeResource:`, fences, events and
  queues. What the new runtime does not build yet it skips and counts, and never fakes:
  - objects: render pipelines, depth-stencil states, render pass descriptors, libraries from
    source, and whatever a call makes of a skipped object (`skipped_objects`);
  - render and parallel render encoders (`skipped_encoders`), and every call of a skipped object
    (`skipped_calls`, with `useResource:usage:` of one and a compute encoder's `useHeap:`);
  - dispatches whose pipeline state or bound buffer or texture is a skipped object
    (`skipped_dispatches`);
  - blit commands that name a skipped object (`skipped_blits`), and by kind: the host's copies
    back to the guest's RAM, which nothing in a play reads (`skipped_guest_ram_reads`),
    `synchronizeTexture:slice:level:` (`skipped_texture_syncs`), mipmaps (`skipped_mipmaps`), and
    the pieces of a copy from the guest's RAM that are not whole 4-byte words at a 4-byte offset,
    into a buffer with no mapping (`skipped_guest_ram_writes`). `optimizeContentsForGPUAccess:`,
    a layout hint, plays as nothing.

  A buffer-to-buffer copy plays as the runtime's (`copy_from_buffer_to_buffer`). A copy from the
  guest's RAM, the recording's plumbing, plays as its pieces' bytes in a Shared staging place of
  the backend's own, then a buffer copy into the target in the same blit encoder, so the GPU's
  order holds; a piece's edge bytes up to whole words come from the target's mapping where it has
  one. The staging place is given back at the command buffer's commit and free once it completes
  (`staged_guest_ram_bytes`).

  Textures over a buffer play as the runtime's (`Buffer::new_texture`) and texture views as its
  views (`Texture::new_texture_view`). Where the runtime cannot make a texture the program made,
  the backend makes an ordinary texture of the same descriptor in its place, so that the IDs and
  addresses the program wrote name a live texture, and counts it: a texture over an IOSurface,
  which is its host's import (`substituted_surface_textures`), a tiled texture over a buffer
  (`substituted_tiled_textures`), and a texture over a buffer or a view the runtime answers with
  a typed gap (`substituted_gap_textures`). A substitute is the recording's artefact, never the
  runtime's.

  The counts are among its executed counters. A play with any of them above zero is no gate
  result: `report.txt` says so on a `skipped` line. It places buffers of their own in the
  runtime's blocks of 16 MiB of either class (a longer one in a block of its own), a
  range freed for reuse once every command buffer committed before its release completed, gives
  the runtime back a block or heap of the stream at its release (`release_block`), and gives
  textures and sampler states indices from the whole of each index class. Its readback serves
  Shared and Managed buffers only; a check of anything else is a refusal. The runtime gives a
  compute pipeline state no `gpuResourceID`, so each one the program saw is reported on an `id`
  line as `none` and counted (`gap_pipeline_ids`).

`report.txt` holds one line per item: `device NAME runtime old|rt snapshots yes|no` first, then
`relocation relocated_words=N misaligned_address_words=N address_arguments=N unrelocated_writes=N
differing_ids=N` (the play's address words relocated and those left at 4 past a multiple of 8,
the address arguments beside which a data word that only looks like an address stands out, the
writes and uploads into objects with no recorded range, which nothing relocates, and the objects
whose ID differs from the recording's, the compute pipeline states' typed gap aside: any such
object makes the play an error, as a word that names it names something else in the play),
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
blocks, buffers, textures, pipelines, kernels, command buffers, spilled bytes and idle Shared and
Private bytes the pool keeps, and the bytes
of the blocks the backend holds for the stream by class, `shared_block_bytes` and
`private_block_bytes`; the old runtime's live memory blocks, empty ones kept and dedicated ones),
`check ID SHA256 match|differs|unrecorded`, `id ID RECORDED PLAYED` for each object whose runtime
gives another `gpuResourceID` than the program saw (both in hex, `PLAYED` `none` where the
runtime's getter raises), `private ID DISPATCH LOCATION+LENGTH match|differs N/WORDS` for each
`m2v.privateBytes` range compared, `split N executed COUNTS` where the player split compute
encoders around them (`N` the encoders split, `COUNTS` the executed counts of their groups),
`unsnapshotted N` where the backend took none of them (`N` the dispatches played as they are),
`parted N` where a recorded range met a live one in part (`N` the older ranges ended),
`relaid HEAPS moved OFFSETS grown BYTES` where it laid placement heaps out again (the heaps, the
offsets it moved in them and the bytes it added to their sizes), `total COUNTS` (its `calls` the
calls played: where
the play stopped, the stopping call is the next), the new runtime's backend's
`pipeline I wall_nanos=N kernel=new|again` for each compute pipeline state it made, in order (the
making's wall time, and whether no state of another specialization of the kernel, a library's
function by name, was made before), `cut SEGMENT OFFSET` where the last segment ends
in a record cut short, `skipped KEY=VALUE ...: no gate result` where the backend skipped
anything, `gaps KEY=VALUE ...` where it met a typed gap (`gap_pipeline_ids`: the compute
pipeline states whose `gpuResourceID` the player asked, which the new runtime does not give),
then `ok` or `error WHAT`. `COUNTS` is `stream KEY=VALUE ... executed
KEY=VALUE ...`. The exit status is 0 for `ok`, 1 for a run that stopped and 2 for a harness
failure (a bad command line or an unreadable stream).
