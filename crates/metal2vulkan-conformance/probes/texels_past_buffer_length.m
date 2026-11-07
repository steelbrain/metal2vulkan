// What Metal answers for the texels of a 2D texture over a plain MTLBuffer that lie past the
// buffer's length: -[MTLBuffer newTextureWithDescriptor:offset:bytesPerRow:] with rows of 256
// bytes (64 RGBA8Unorm texels) over a 4096-byte Shared buffer, its last row or rows past the
// buffer's bytes, which a plain run makes.
//
// ONE SHAPE AND ONE PATH PER PROCESS, chosen on the command line: a write past the buffer may
// reach memory that is not the buffer's, and a later path in the same process would read what an
// earlier one left there.
//
//   texels_past_buffer_length <shape> <path>
//
// Shapes (offset, height; the buffer 4096 bytes unless named):
//   in16      0, 16: every texel inside the buffer (the control)
//   past17    0, 17: the last row past it
//   off256    256, 16: the last row past it
//   off2048   2048, 16: the last 8 rows past it
//   big       0, 80 over a 20000-byte buffer (allocatedSize 32768): rows past the length, inside
//             the allocation
//
// Paths: read (a kernel's read()), sample (a kernel's sample() at each texel's centre, nearest),
// blit (copyFromTexture:toBuffer:), getbytes (getBytes:), and the writes kwrite (a kernel's
// write()), bwrite (copyFromBuffer:toTexture:) and replace (replaceRegion:), each into the rows
// past the length alone, after which the process reads the texture back by read, blit and getbytes.
//
// Every word of the target buffer is 0xA0000000 | its offset / 4, and every word of each of the
// 128 neighbour buffers made around it (4096 bytes each, Shared) 0xB0000000 | k << 16 | offset / 4,
// so a texel's word names the byte it came from: the buffer's own byte where the texture lays it
// out (`own`), another of the target's (`tgt@<offset>`), a neighbour's (`nb<k>@<offset>`, with
// that neighbour's gpuAddress against the target's), zero, or other. After a write the probe counts
// the target's and every neighbour's words that changed.
//
// Measured on an Apple M3 Max under macOS 27.0.1, plain Metal (no API validation layer), every
// shape made. The 4096-byte buffer's allocatedSize is its length, and the buffers made around it
// lie 4096 bytes apart in gpuAddress. Inside the length every path reads the buffer's own bytes,
// and no write there or past it changes one of the target's words. Past the length (past17,
// off256, off2048) read, sample, blit and getbytes all read the buffer allocated next (at the
// target's gpuAddress + 4096) from its byte 0, row for row; kwrite, bwrite and replace each
// change that neighbour's words (64 a row past) and no other buffer's, and read, blit and getbytes
// then read what was written. Over the 20000-byte buffer (big, allocatedSize 32768) row 78 holds
// 8 of the buffer's words and then zeros, row 79 zeros, on every path; a write of rows 78 and 79
// changes the target's 8 words of row 78 inside the length and no other buffer's, and every path
// then reads what was written. So Metal's paths agree: the bytes past the length are one memory,
// what the allocation holds there, which every path reads and writes alike.
//
// Build: clang -fobjc-arc -framework Metal -framework Foundation texels_past_buffer_length.m -o tpbl
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { W = 64, PITCH = 256, NEIGHBOURS = 128 };

static const char *SOURCE =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void rd(texture2d<float, access::read> t [[texture(0)]], device uint *o [[buffer(0)]],\n"
    "               uint2 g [[thread_position_in_grid]]) {\n"
    "  if (g.x < t.get_width() && g.y < t.get_height())\n"
    "    o[g.y * t.get_width() + g.x] = pack_float_to_unorm4x8(t.read(g));\n"
    "}\n"
    "kernel void sm(texture2d<float, access::sample> t [[texture(0)]], device uint *o [[buffer(0)]],\n"
    "               uint2 g [[thread_position_in_grid]]) {\n"
    "  constexpr sampler s(filter::nearest, address::clamp_to_edge, coord::normalized);\n"
    "  float2 c = (float2(g) + 0.5) / float2(t.get_width(), t.get_height());\n"
    "  if (g.x < t.get_width() && g.y < t.get_height())\n"
    "    o[g.y * t.get_width() + g.x] = pack_float_to_unorm4x8(t.sample(s, c, level(0)));\n"
    "}\n"
    "kernel void wr(texture2d<float, access::write> t [[texture(0)]], constant uint &first [[buffer(0)]],\n"
    "               uint2 g [[thread_position_in_grid]]) {\n"
    "  if (g.x < t.get_width() && g.y < t.get_height() && g.y >= first)\n"
    "    t.write(unpack_unorm4x8_to_float(0xC0000000u | (g.y << 8) | g.x), g);\n"
    "}\n";

static id<MTLDevice> dev;
static id<MTLCommandQueue> queue;
static id<MTLLibrary> lib;

static uint32_t target_word(NSUInteger o) { return 0xA0000000u | (uint32_t)(o / 4); }
static uint32_t neighbour_word(unsigned k, NSUInteger o) { return 0xB0000000u | (k << 16) | (uint32_t)(o / 4); }

static void fill(id<MTLBuffer> b, uint32_t (^word)(NSUInteger))
{
	uint32_t *w = b.contents;
	for (NSUInteger o = 0; o < b.length; o += 4)
		w[o / 4] = word(o);
}

static void wait_for(id<MTLCommandBuffer> cb, const char *what)
{
	[cb commit];
	[cb waitUntilCompleted];
	if (cb.status != MTLCommandBufferStatusCompleted)
		printf("  %s: command buffer status %ld, error %s\n", what, (long)cb.status,
		       cb.error ? cb.error.description.UTF8String : "none");
}

static void dispatch(NSString *name, id<MTLTexture> t, id<MTLBuffer> b, NSUInteger h)
{
	NSError *err = nil;
	id<MTLComputePipelineState> p =
	    [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&err];
	if (!p) {
		printf("  pipeline %s: %s\n", name.UTF8String, err.description.UTF8String);
		exit(1);
	}
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
	[e setComputePipelineState:p];
	[e setTexture:t atIndex:0];
	[e setBuffer:b offset:0 atIndex:0];
	[e dispatchThreads:MTLSizeMake(W, h, 1) threadsPerThreadgroup:MTLSizeMake(16, 1, 1)];
	[e endEncoding];
	wait_for(cb, name.UTF8String);
}

// Prints each row's texels as runs: a run is texels whose words name consecutive bytes of one
// source (or all read zero, or another word).
static void classify(const char *what, const uint32_t *v, NSUInteger h, NSUInteger offset, NSUInteger length,
                     NSArray<id<MTLBuffer>> *nb, id<MTLBuffer> target)
{
	printf("  %s:\n", what);
	for (NSUInteger r = 0; r < h; r++) {
		NSUInteger row_start = offset + r * PITCH;
		char line[2048];
		int n = snprintf(line, sizeof line, "    row %2lu (bytes %lu..%lu%s):", (unsigned long)r,
		                 (unsigned long)row_start, (unsigned long)(row_start + W * 4),
		                 row_start + W * 4 > length ? ", past the length" : "");
		NSUInteger c = 0;
		while (c < W) {
			uint32_t x = v[r * W + c];
			unsigned kind = x >> 28;
			NSUInteger start = c;
			char label[96];
			if (kind == 0xA) {
				NSUInteger o = (x & 0xFFFF) * 4;
				NSUInteger want = row_start + c * 4;
				if (o == want)
					snprintf(label, sizeof label, "own");
				else
					snprintf(label, sizeof label, "tgt@%lu", (unsigned long)o);
				c++;
				while (c < W && v[r * W + c] == x + (uint32_t)(c - start))
					c++;
			} else if (kind == 0xB) {
				unsigned k = (x >> 16) & 0xFFF;
				NSUInteger o = (x & 0xFFFF) * 4;
				long long rel = k < nb.count ? (long long)nb[k].gpuAddress - (long long)target.gpuAddress : 0;
				snprintf(label, sizeof label, "nb%u@%lu (its gpuAddress target%+lld)", k, (unsigned long)o, rel);
				c++;
				while (c < W && v[r * W + c] == x + (uint32_t)(c - start))
					c++;
			} else if (kind == 0xC) {
				snprintf(label, sizeof label, "written(r%u c%u)", (x >> 8) & 0xFF, x & 0xFF);
				c++;
				while (c < W && v[r * W + c] == x + (uint32_t)(c - start))
					c++;
			} else if (x == 0) {
				snprintf(label, sizeof label, "zero");
				c++;
				while (c < W && v[r * W + c] == 0)
					c++;
			} else {
				snprintf(label, sizeof label, "other(%08x)", x);
				c++;
			}
			n += snprintf(line + n, sizeof line - n, " %s x%lu", label, (unsigned long)(c - start));
			if (n > 1900)
				break;
		}
		printf("%s\n", line);
	}
}

static void changed(id<MTLBuffer> target, NSArray<id<MTLBuffer>> *nb)
{
	const uint32_t *w = target.contents;
	NSUInteger t = 0;
	for (NSUInteger o = 0; o < target.length; o += 4)
		t += w[o / 4] != target_word(o);
	printf("  target words changed: %lu of %lu\n", (unsigned long)t, (unsigned long)(target.length / 4));
	NSUInteger total = 0;
	for (unsigned k = 0; k < nb.count; k++) {
		const uint32_t *x = nb[k].contents;
		NSUInteger m = 0, first = 0;
		for (NSUInteger o = 0; o < nb[k].length; o += 4)
			if (x[o / 4] != neighbour_word(k, o)) {
				if (!m)
					first = o;
				m++;
			}
		if (m)
			printf("  neighbour %u (gpuAddress target%+lld) words changed: %lu, from offset %lu\n", k,
			       (long long)nb[k].gpuAddress - (long long)target.gpuAddress, (unsigned long)m,
			       (unsigned long)first);
		total += m;
	}
	printf("  neighbour words changed: %lu\n", (unsigned long)total);
}

int main(int argc, char **argv)
{
	@autoreleasepool {
		if (argc != 3) {
			fprintf(stderr, "usage: %s <in16|past17|off256|off2048|big> <read|sample|blit|getbytes|kwrite|bwrite|replace|made>\n", argv[0]);
			return 2;
		}
		const char *shape = argv[1], *path = argv[2];
		NSUInteger offset = 0, h = 16, length = 4096;
		if (!strcmp(shape, "in16"))
			;
		else if (!strcmp(shape, "past17"))
			h = 17;
		else if (!strcmp(shape, "off256"))
			offset = 256;
		else if (!strcmp(shape, "off2048"))
			offset = 2048;
		else if (!strcmp(shape, "big"))
			h = 80, length = 20000;
		else
			return 2;
		dev = MTLCreateSystemDefaultDevice();
		queue = [dev newCommandQueue];
		NSError *err = nil;
		lib = [dev newLibraryWithSource:@(SOURCE) options:nil error:&err];
		if (!lib) {
			printf("library: %s\n", err.description.UTF8String);
			return 1;
		}
		// Neighbours made before and after the target, so the allocator places some of them beside it.
		NSMutableArray<id<MTLBuffer>> *nb = [NSMutableArray array];
		for (unsigned k = 0; k < NEIGHBOURS / 2; k++)
			[nb addObject:[dev newBufferWithLength:4096 options:MTLResourceStorageModeShared]];
		id<MTLBuffer> target = [dev newBufferWithLength:length options:MTLResourceStorageModeShared];
		for (unsigned k = NEIGHBOURS / 2; k < NEIGHBOURS; k++)
			[nb addObject:[dev newBufferWithLength:4096 options:MTLResourceStorageModeShared]];
		fill(target, ^uint32_t(NSUInteger o) { return target_word(o); });
		for (unsigned k = 0; k < NEIGHBOURS; k++)
			fill(nb[k], ^uint32_t(NSUInteger o) { return neighbour_word(k, o); });
		long long after = 0, before = 0;
		for (unsigned k = 0; k < NEIGHBOURS; k++) {
			long long rel = (long long)nb[k].gpuAddress - (long long)target.gpuAddress;
			if (rel > 0 && (!after || rel < after))
				after = rel;
			if (rel < 0 && (!before || rel > before))
				before = rel;
		}
		printf("%s, macOS %s: %s %s: buffer length %lu, allocatedSize %lu; nearest neighbours at gpuAddress "
		       "target%+lld and target%+lld\n",
		       dev.name.UTF8String, NSProcessInfo.processInfo.operatingSystemVersionString.UTF8String, shape,
		       path, (unsigned long)target.length, (unsigned long)target.allocatedSize, before, after);
		BOOL writes = !strcmp(path, "kwrite") || !strcmp(path, "bwrite") || !strcmp(path, "replace");
		MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
		                                                                             width:W
		                                                                            height:h
		                                                                         mipmapped:NO];
		d.storageMode = MTLStorageModeShared;
		d.usage = writes ? (MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite) : MTLTextureUsageShaderRead;
		id<MTLTexture> t = [target newTextureWithDescriptor:d offset:offset bytesPerRow:PITCH];
		printf("  texture 64x%lu at offset %lu (ends at %lu): %s", (unsigned long)h, (unsigned long)offset,
		       (unsigned long)(offset + PITCH * (h - 1) + W * 4), t ? "made" : "nil\n");
		if (!t)
			return 0;
		printf(", allocatedSize %lu, bufferOffset %lu, bufferBytesPerRow %lu\n", (unsigned long)t.allocatedSize,
		       (unsigned long)t.bufferOffset, (unsigned long)t.bufferBytesPerRow);
		if (!strcmp(path, "made"))
			return 0;
		NSUInteger first_past = 0;
		while (first_past < h && offset + first_past * PITCH + W * 4 <= length)
			first_past++;
		NSUInteger bytes = W * 4 * h;
		id<MTLBuffer> out = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
		void (^read_back)(const char *) = ^(const char *how) {
			memset(out.contents, 0x5A, bytes);
			if (!strcmp(how, "read") || !strcmp(how, "sample")) {
				dispatch(!strcmp(how, "read") ? @"rd" : @"sm", t, out, h);
				classify(how, out.contents, h, offset, length, nb, target);
			} else if (!strcmp(how, "blit")) {
				id<MTLCommandBuffer> cb = [queue commandBuffer];
				id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
				[e copyFromTexture:t
				             sourceSlice:0
				             sourceLevel:0
				            sourceOrigin:MTLOriginMake(0, 0, 0)
				              sourceSize:MTLSizeMake(W, h, 1)
				                toBuffer:out
				       destinationOffset:0
				  destinationBytesPerRow:W * 4
				destinationBytesPerImage:bytes];
				[e endEncoding];
				wait_for(cb, "blit");
				classify(how, out.contents, h, offset, length, nb, target);
			} else {
				uint32_t *v = malloc(bytes);
				memset(v, 0x5A, bytes);
				[t getBytes:v bytesPerRow:W * 4 fromRegion:MTLRegionMake2D(0, 0, W, h) mipmapLevel:0];
				classify(how, v, h, offset, length, nb, target);
				free(v);
			}
		};
		if (!writes) {
			read_back(path);
			changed(target, nb);
			return 0;
		}
		NSUInteger past = h - first_past;
		if (past == 0) {
			// No row lies past the length (`in16`): a write of no rows is no question, and a blit
			// of a region of height 0 is not a call to make.
			printf("  no rows past the length: nothing to write\n");
			return 0;
		}
		printf("  writing rows %lu..%lu (past the length)\n", (unsigned long)first_past, (unsigned long)h);
		uint32_t *src = malloc(W * 4 * past);
		for (NSUInteger r = 0; r < past; r++)
			for (NSUInteger c = 0; c < W; c++)
				src[r * W + c] = 0xC0000000u | (uint32_t)((first_past + r) << 8) | (uint32_t)c;
		if (!strcmp(path, "kwrite")) {
			uint32_t f = (uint32_t)first_past;
			id<MTLBuffer> fb = [dev newBufferWithBytes:&f length:4 options:MTLResourceStorageModeShared];
			dispatch(@"wr", t, fb, h);
		} else if (!strcmp(path, "bwrite")) {
			id<MTLBuffer> sb = [dev newBufferWithBytes:src length:W * 4 * past options:MTLResourceStorageModeShared];
			id<MTLCommandBuffer> cb = [queue commandBuffer];
			id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
			[e copyFromBuffer:sb
			           sourceOffset:0
			      sourceBytesPerRow:W * 4
			    sourceBytesPerImage:W * 4 * past
			             sourceSize:MTLSizeMake(W, past, 1)
			              toTexture:t
			       destinationSlice:0
			       destinationLevel:0
			      destinationOrigin:MTLOriginMake(0, first_past, 0)];
			[e endEncoding];
			wait_for(cb, "bwrite");
		} else {
			[t replaceRegion:MTLRegionMake2D(0, first_past, W, past) mipmapLevel:0 withBytes:src bytesPerRow:W * 4];
		}
		free(src);
		changed(target, nb);
		read_back("read");
		read_back("blit");
		read_back("getbytes");
		changed(target, nb);
	}
	return 0;
}
