// What Metal reads of a two-plane YCbCr texture whose rows of codes are wider than its planes'
// `bytesPerRow`, so that each row of codes overlaps the next, and what its CPU calls move.
//
// `ycbcr.rs`'s **Surfaces** says Metal reads such a texture's luma codes from the asked plane at
// its offset and rows, and its chroma pairs from the plane after it at that plane's rows, however
// wide the texture's rows are. This holds that to Metal over the shapes whose rows overlap:
//
//   1. `YCBCR8_420_2P` 66 x 64 over a 64 x 64 '420v' surface whose planes are 64 bytes a row
//      (plane 0 at 512, plane 1 at 4608, 8192 bytes): each luma row and each chroma row of the
//      texture is 66 bytes, 2 more than the pitch;
//   2. `YCBCR10_420_2P` 64 x 64 over the same 8-bit planes: each row of 16-bit codes is 128
//      bytes, twice the pitch, so a byte is shared by three rows;
//   3. a control, `YCBCR8_420_2P` 64 x 64 over the same planes, whose rows overlap nothing.
//
// For each shape the texture over the surface is set beside a twin: the same format and size over
// a second '420v' surface whose rows lie a multiple of 16 bytes at or past the row apart, row `y`
// of each plane holding the bytes the first surface's row `y` of codes reaches. Metal reads the
// twin's codes at its own rows, which share no byte, so where the texture's `read()` and nearest
// sample of every texel are the twin's bit for bit, Metal read the overlapping rows' codes at the
// planes' rows. Then, over the first surface: `getBytes` of the whole texture against the luma
// plane's rows at its pitch, and `replaceRegion` of the whole texture against the rows written in
// order, the later row's bytes standing where two rows meet, every byte of the surface compared.
// Last, over the first surface's bytes again, `copyFromBuffer:toTexture:` of the whole texture:
// whether Metal performs it, and for each byte two or more rows reach, whether it holds the
// earliest row's byte, a middle one's, the latest one's or none of them.
//
// Every byte of the first surface is drawn from a fixed generator, so every byte a row reaches is
// known. One shape a process, the shape's index in argv[1].
//
// clang -fobjc-arc -framework Metal -framework Foundation -framework IOSurface \
//     -o ycbcr_wide_code_rows ycbcr_wide_code_rows.m

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>

// One shape: a texture of `format`, `w` x `h`, its codes `unit` bytes each.
struct Shape {
	const char *name;
	MTLPixelFormat format;
	size_t w;
	size_t h;
	size_t unit;
};

static const struct Shape kShapes[] = {
    {"8bit.66x64", (MTLPixelFormat)500, 66, 64, 1},
    {"10bit.64x64", (MTLPixelFormat)505, 64, 64, 2},
    {"8bit.64x64.control", (MTLPixelFormat)500, 64, 64, 1},
};
static const size_t kShapeCount = sizeof(kShapes) / sizeof(kShapes[0]);

// The first surface: 64 x 64 '420v', both planes 64 bytes a row.
enum {
	kSize = 64,
	kPitch = 64,
	kLumaAt = 512,
	kChromaAt = kLumaAt + kPitch * kSize,
	kLength = 8192,
};

static uint64_t gState;
static uint8_t nextByte(void) {
	gState = gState * 6364136223846793005ull + 1442695040888963407ull;
	return (uint8_t)(gState >> 33);
}

static size_t roundUp16(size_t n) { return (n + 15) & ~(size_t)15; }

// A '420v' surface of `w` x `h`: plane 0 of `w` x `h` one-byte elements at `luma` with rows
// `lumaRow` apart, plane 1 of `w/2` x `h/2` two-byte pairs at `chroma` with rows `chromaRow` apart,
// in an allocation of `length` bytes. NULL where IOSurface lays it out otherwise.
static IOSurfaceRef makeSurface(size_t w, size_t h, size_t luma, size_t lumaRow, size_t chroma,
                                size_t chromaRow, size_t length) {
	NSDictionary *plane0 = @{
		(__bridge NSString *)kIOSurfacePlaneWidth : @(w),
		(__bridge NSString *)kIOSurfacePlaneHeight : @(h),
		(__bridge NSString *)kIOSurfacePlaneBytesPerRow : @(lumaRow),
		(__bridge NSString *)kIOSurfacePlaneOffset : @(luma),
		(__bridge NSString *)kIOSurfacePlaneSize : @(lumaRow * h),
		(__bridge NSString *)kIOSurfacePlaneBytesPerElement : @(1),
		(__bridge NSString *)kIOSurfacePlaneElementWidth : @(1),
		(__bridge NSString *)kIOSurfacePlaneElementHeight : @(1),
	};
	NSDictionary *plane1 = @{
		(__bridge NSString *)kIOSurfacePlaneWidth : @(w / 2),
		(__bridge NSString *)kIOSurfacePlaneHeight : @(h / 2),
		(__bridge NSString *)kIOSurfacePlaneBytesPerRow : @(chromaRow),
		(__bridge NSString *)kIOSurfacePlaneOffset : @(chroma),
		(__bridge NSString *)kIOSurfacePlaneSize : @(chromaRow * (h / 2)),
		(__bridge NSString *)kIOSurfacePlaneBytesPerElement : @(2),
		(__bridge NSString *)kIOSurfacePlaneElementWidth : @(1),
		(__bridge NSString *)kIOSurfacePlaneElementHeight : @(1),
	};
	NSDictionary *properties = @{
		(__bridge NSString *)kIOSurfaceWidth : @(w),
		(__bridge NSString *)kIOSurfaceHeight : @(h),
		(__bridge NSString *)kIOSurfaceBytesPerElement : @(1),
		(__bridge NSString *)kIOSurfacePixelFormat : @((uint32_t)'420v'),
		(__bridge NSString *)kIOSurfaceAllocSize : @(length),
		(__bridge NSString *)kIOSurfacePlaneInfo : @[ plane0, plane1 ],
	};
	IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)properties);
	if (!surface) {
		printf("IOSurfaceCreate answered NULL for %zux%zu\n", w, h);
		return NULL;
	}
	uint8_t *base = IOSurfaceGetBaseAddress(surface);
	size_t at0 = (size_t)((uint8_t *)IOSurfaceGetBaseAddressOfPlane(surface, 0) - base);
	size_t at1 = (size_t)((uint8_t *)IOSurfaceGetBaseAddressOfPlane(surface, 1) - base);
	size_t row0 = IOSurfaceGetBytesPerRowOfPlane(surface, 0);
	size_t row1 = IOSurfaceGetBytesPerRowOfPlane(surface, 1);
	if (at0 != luma || at1 != chroma || row0 != lumaRow || row1 != chromaRow ||
	    IOSurfaceGetAllocSize(surface) < length) {
		printf("surface laid otherwise: planes at %zu (%zu a row) and %zu (%zu a row), %zu "
		       "bytes; asked %zu (%zu), %zu (%zu), %zu\n",
		       at0, row0, at1, row1, IOSurfaceGetAllocSize(surface), luma, lumaRow, chroma,
		       chromaRow, length);
		CFRelease(surface);
		return NULL;
	}
	return surface;
}

static void copyIn(IOSurfaceRef surface, const uint8_t *bytes, size_t length) {
	IOSurfaceLock(surface, 0, NULL);
	memcpy(IOSurfaceGetBaseAddress(surface), bytes, length);
	IOSurfaceUnlock(surface, 0, NULL);
}

static void copyOut(IOSurfaceRef surface, uint8_t *bytes, size_t length) {
	IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL);
	memcpy(bytes, IOSurfaceGetBaseAddress(surface), length);
	IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
}

static NSString *const kSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void reads(texture2d<float, access::read> t [[texture(0)]],\n"
     "                  device float4 *out [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {\n"
     "    uint w = t.get_width(), h = t.get_height();\n"
     "    if (gid.x >= w || gid.y >= h) return;\n"
     "    out[gid.y * w + gid.x] = t.read(gid);\n"
     "}\n"
     "kernel void samples(texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]],\n"
     "                    device float4 *out [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {\n"
     "    uint w = t.get_width(), h = t.get_height();\n"
     "    if (gid.x >= w || gid.y >= h) return;\n"
     "    out[gid.y * w + gid.x] = t.sample(s, (float2(gid) + 0.5) / float2(w, h));\n"
     "}\n";

// The `float4`s `state` reads of every texel of `texture` (through `sampler`, where given).
static NSData *texels(id<MTLDevice> device, id<MTLCommandQueue> queue,
                      id<MTLComputePipelineState> state, id<MTLTexture> texture,
                      id<MTLSamplerState> sampler) {
	size_t n = texture.width * texture.height;
	id<MTLBuffer> out = [device newBufferWithLength:n * 16 options:MTLResourceStorageModeShared];
	memset(out.contents, 0xee, n * 16);
	id<MTLCommandBuffer> commands = [queue commandBuffer];
	id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
	[encoder setComputePipelineState:state];
	[encoder setTexture:texture atIndex:0];
	[encoder setBuffer:out offset:0 atIndex:0];
	if (sampler) {
		[encoder setSamplerState:sampler atIndex:0];
	}
	[encoder dispatchThreads:MTLSizeMake(texture.width, texture.height, 1)
	    threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
	[encoder endEncoding];
	[commands commit];
	[commands waitUntilCompleted];
	if (commands.status != MTLCommandBufferStatusCompleted) {
		printf("command buffer status %ld: %s\n", (long)commands.status,
		       commands.error.description.UTF8String);
	}
	return [NSData dataWithBytes:out.contents length:n * 16];
}

static uint32_t fnv(const uint8_t *bytes, size_t length) {
	uint32_t h = 0x811c9dc5u;
	for (size_t i = 0; i < length; i++) {
		h = (h ^ bytes[i]) * 0x01000193u;
	}
	return h;
}

// How many of the texels `a` and `b` hold differ, the first few printed.
static size_t compareTexels(const char *what, NSData *a, NSData *b, size_t w) {
	const float *x = a.bytes, *y = b.bytes;
	size_t n = a.length / 16, off = 0;
	for (size_t i = 0; i < n; i++) {
		if (memcmp(x + 4 * i, y + 4 * i, 16) != 0) {
			if (off < 4) {
				printf("  %s texel (%zu, %zu): (%a %a %a %a), the twin's (%a %a %a %a)\n", what,
				       i % w, i / w, x[4 * i], x[4 * i + 1], x[4 * i + 2], x[4 * i + 3], y[4 * i],
				       y[4 * i + 1], y[4 * i + 2], y[4 * i + 3]);
			}
			off++;
		}
	}
	return off;
}

static id<MTLTexture> planeTexture(id<MTLDevice> device, const struct Shape *shape,
                                   IOSurfaceRef surface) {
	MTLTextureDescriptor *d = [MTLTextureDescriptor new];
	d.textureType = MTLTextureType2D;
	d.pixelFormat = shape->format;
	d.width = shape->w;
	d.height = shape->h;
	d.usage = MTLTextureUsageShaderRead;
	d.storageMode = MTLStorageModeShared;
	return [device newTextureWithDescriptor:d iosurface:surface plane:0];
}

int main(int argc, char **argv) {
	@autoreleasepool {
		if (argc != 2 || (size_t)atoi(argv[1]) >= kShapeCount) {
			fprintf(stderr, "usage: %s SHAPE (0 to %zu)\n", argv[0], kShapeCount - 1);
			return 2;
		}
		const struct Shape *shape = &kShapes[atoi(argv[1])];
		id<MTLDevice> device = MTLCreateSystemDefaultDevice();
		printf("%s on %s, %s\n", shape->name, device.name.UTF8String,
		       [NSProcessInfo processInfo].operatingSystemVersionString.UTF8String);
		NSError *error = nil;
		id<MTLLibrary> library = [device newLibraryWithSource:kSource options:nil error:&error];
		if (!library) {
			printf("library: %s\n", error.description.UTF8String);
			return 1;
		}
		id<MTLComputePipelineState> reads =
		    [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"reads"]
		                                          error:&error];
		id<MTLComputePipelineState> samples =
		    [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"samples"]
		                                          error:&error];
		MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
		sd.minFilter = MTLSamplerMinMagFilterNearest;
		sd.magFilter = MTLSamplerMinMagFilterNearest;
		id<MTLSamplerState> nearest = [device newSamplerStateWithDescriptor:sd];
		id<MTLCommandQueue> queue = [device newCommandQueue];

		uint8_t bytes[kLength];
		gState = 2026;
		for (size_t i = 0; i < kLength; i++) {
			bytes[i] = nextByte();
		}
		IOSurfaceRef surface =
		    makeSurface(kSize, kSize, kLumaAt, kPitch, kChromaAt, kPitch, kLength);
		if (!surface) {
			return 1;
		}
		copyIn(surface, bytes, kLength);

		// The twin: rows at a multiple of 16 at or past the row, each holding what the first
		// surface's row of codes reaches.
		size_t w = shape->w, h = shape->h, cw = w / 2, ch = h / 2;
		size_t lumaRow = w * shape->unit, chromaRow = cw * 2 * shape->unit;
		size_t twinLuma = roundUp16(lumaRow), twinChroma = roundUp16(chromaRow);
		size_t twinChromaAt = twinLuma * h, twinLength = twinChromaAt + twinChroma * ch;
		IOSurfaceRef twin = makeSurface(w, h, 0, twinLuma, twinChromaAt, twinChroma, twinLength);
		if (!twin) {
			return 1;
		}
		uint8_t *twinBytes = calloc(twinLength, 1);
		for (size_t y = 0; y < h; y++) {
			memcpy(twinBytes + y * twinLuma, bytes + kLumaAt + y * kPitch, lumaRow);
		}
		for (size_t y = 0; y < ch; y++) {
			memcpy(twinBytes + twinChromaAt + y * twinChroma, bytes + kChromaAt + y * kPitch,
			       chromaRow);
		}
		copyIn(twin, twinBytes, twinLength);
		printf("rows of codes: luma %zu and chroma %zu bytes at a pitch of %d; the twin's at %zu "
		       "and %zu\n",
		       lumaRow, chromaRow, kPitch, twinLuma, twinChroma);

		id<MTLTexture> texture = planeTexture(device, shape, surface);
		id<MTLTexture> twinTexture = planeTexture(device, shape, twin);
		printf("texture: %s; twin: %s\n", texture ? "made" : "nil", twinTexture ? "made" : "nil");
		if (!texture || !twinTexture) {
			return 1;
		}

		NSData *read = texels(device, queue, reads, texture, nil);
		NSData *twinRead = texels(device, queue, reads, twinTexture, nil);
		NSData *sampled = texels(device, queue, samples, texture, nearest);
		NSData *twinSampled = texels(device, queue, samples, twinTexture, nearest);
		printf("read(): %zu of %zu texels differ from the twin's (fnv %08x, the twin's %08x)\n",
		       compareTexels("read()", read, twinRead, w), w * h, fnv(read.bytes, read.length),
		       fnv(twinRead.bytes, twinRead.length));
		printf("nearest sample: %zu of %zu texels differ from the twin's; %zu from read()\n",
		       compareTexels("sample", sampled, twinSampled, w), w * h,
		       compareTexels("sample vs read()", sampled, read, w));

		// getBytes: the luma plane's rows at its pitch, each the whole row of codes.
		uint8_t *got = malloc(lumaRow * h);
		[texture getBytes:got
		      bytesPerRow:lumaRow
		       fromRegion:MTLRegionMake2D(0, 0, w, h)
		      mipmapLevel:0];
		size_t off = 0;
		for (size_t y = 0; y < h; y++) {
			for (size_t i = 0; i < lumaRow; i++) {
				off += got[y * lumaRow + i] != bytes[kLumaAt + y * kPitch + i];
			}
		}
		printf("getBytes: %zu of %zu bytes differ from the plane's rows\n", off, lumaRow * h);

		// replaceRegion: the rows written in order, the later row's bytes standing.
		uint8_t *pattern = malloc(lumaRow * h);
		gState = 77;
		for (size_t i = 0; i < lumaRow * h; i++) {
			pattern[i] = nextByte();
		}
		[texture replaceRegion:MTLRegionMake2D(0, 0, w, h)
		           mipmapLevel:0
		             withBytes:pattern
		           bytesPerRow:lumaRow];
		uint8_t want[kLength], after[kLength];
		memcpy(want, bytes, kLength);
		for (size_t y = 0; y < h; y++) {
			memcpy(want + kLumaAt + y * kPitch, pattern + y * lumaRow, lumaRow);
		}
		copyOut(surface, after, kLength);
		off = 0;
		size_t first = kLength;
		for (size_t i = 0; i < kLength; i++) {
			if (after[i] != want[i]) {
				off++;
				first = first < i ? first : i;
			}
		}
		printf("replaceRegion: %zu of %d surface bytes differ from the rows written in order",
		       off, kLength);
		if (off) {
			printf(" (first at %zu: %02x, in order %02x)", first, after[first], want[first]);
		}
		printf("\n");

		// copyFromBuffer:toTexture: of the whole texture, over the surface's first bytes again:
		// whether Metal performs it, and, for each byte two or more rows of codes reach, which
		// of them its bytes are.
		copyIn(surface, bytes, kLength);
		id<MTLBuffer> source = [device newBufferWithLength:lumaRow * h
		                                           options:MTLResourceStorageModeShared];
		uint8_t *blitted = source.contents;
		gState = 91;
		for (size_t i = 0; i < lumaRow * h; i++) {
			blitted[i] = nextByte();
		}
		id<MTLCommandBuffer> commands = [queue commandBuffer];
		id<MTLBlitCommandEncoder> blit = [commands blitCommandEncoder];
		[blit copyFromBuffer:source
		           sourceOffset:0
		      sourceBytesPerRow:lumaRow
		    sourceBytesPerImage:lumaRow * h
		             sourceSize:MTLSizeMake(w, h, 1)
		              toTexture:texture
		       destinationSlice:0
		       destinationLevel:0
		      destinationOrigin:MTLOriginMake(0, 0, 0)];
		[blit endEncoding];
		[commands commit];
		[commands waitUntilCompleted];
		printf("blit from a buffer: command buffer status %ld%s%s\n", (long)commands.status,
		       commands.error ? ", " : "",
		       commands.error ? commands.error.description.UTF8String : "");
		copyOut(surface, after, kLength);
		// Per byte: untouched where no row reaches it, else the earliest, a middle, the latest
		// or none of the rows that reach it.
		size_t lone = 0, loneOff = 0, outside = 0, earliest = 0, middle = 0, latest = 0, none = 0;
		for (size_t p = 0; p < kLength; p++) {
			uint8_t values[4];
			size_t k = 0;
			for (size_t y = 0; y < h && k < 4; y++) {
				size_t start = kLumaAt + y * kPitch;
				if (p >= start && p < start + lumaRow) {
					values[k++] = blitted[y * lumaRow + (p - start)];
				}
			}
			if (k == 0) {
				outside += after[p] != bytes[p];
			} else if (k == 1) {
				lone++;
				loneOff += after[p] != values[0];
			} else if (after[p] == values[k - 1]) {
				latest++;
			} else if (after[p] == values[0]) {
				earliest++;
			} else {
				size_t j = 1;
				while (j + 1 < k && after[p] != values[j]) {
					j++;
				}
				if (j + 1 < k) {
					middle++;
				} else {
					none++;
				}
			}
		}
		printf("blit: %zu bytes of one row, %zu of them off; %zu bytes no row reaches changed; "
		       "shared bytes: %zu the latest row's, %zu the earliest's, %zu a middle row's, %zu "
		       "none\n",
		       lone, loneOff, outside, latest, earliest, middle, none);
		free(got);
		free(pattern);
		free(twinBytes);
		CFRelease(twin);
		CFRelease(surface);
	}
	return 0;
}
