// What Metal answers for the texels of a texture buffer over a plain MTLBuffer that lie past the
// buffer's length: -[MTLBuffer newTextureWithDescriptor:offset:bytesPerRow:] of a
// MTLTextureTypeTextureBuffer descriptor, 64 RGBA8Unorm texels (256 bytes), Shared, which a plain
// run makes wherever its texels end (texels_past_buffer_length.m asks the same of 2D textures).
//
// ONE SHAPE AND ONE PATH PER PROCESS, chosen on the command line: a write past the buffer may
// reach memory that is not the buffer's, and a later path in the same process would read what an
// earlier one left there.
//
//   texture_buffer_past_length <shape> <path>
//
// Shapes (offset; the buffer 4096 bytes unless named):
//   tin      3840: every texel inside the buffer (the control)
//   tpast    3968: the last 32 texels past it
//   toff8    3976: the last 34 texels past it, at an offset off 16 bytes
//   twhole   4096: every texel past it
//   tbig     19840 over a 20000-byte buffer (allocatedSize 32768): the last 24 texels past the
//            length, inside the allocation
//
// Paths: made, read (a kernel's read()), blit (copyFromTexture:toBuffer:), getbytes (getBytes:),
// and the writes kwrite (a kernel's write()), bwrite (copyFromBuffer:toTexture:) and replace
// (replaceRegion:), each of the texels past the length alone, after which the process reads the
// texture back by read, blit and getbytes.
//
// Every word of the target buffer is 0xA0000000 | its offset / 4, and every word of each of the
// 128 neighbour buffers made around it (4096 bytes each, Shared) 0xB0000000 | k << 16 | offset / 4,
// so a texel's word names the byte it came from: the buffer's own byte where the texture lays it
// out (`own`), another of the target's (`tgt@<offset>`), a neighbour's (`nb<k>@<offset>`, with
// that neighbour's gpuAddress against the target's), zero, or other. A written texel is
// 0xC0000000 | its index. After a write the probe counts the target's and every neighbour's words
// that changed.
//
// Measured on an Apple M3 Max under macOS 27.0.1, plain Metal (no API validation layer), every
// shape made (type 9, its bufferOffset the offset given; minimumTextureBufferAlignment 16). The
// 4096-byte buffer's allocatedSize is its length, and the buffers made around it lie 4096 bytes
// apart in gpuAddress. Inside the length every path reads the buffer's own bytes. Past it (tpast,
// toff8, twhole) read, blit and getbytes all read the buffer allocated next (target + 4096) from its
// byte 0; kwrite, bwrite and replace each change that neighbour's words and no other buffer's,
// and every path then reads what was written. At toff8 the kernel reads and writes from the
// offset rounded down to 16 (3968) and the blits and getBytes from the offset itself, as inside
// the length. Over the 20000-byte buffer (tbig, allocatedSize 32768) the texels past the length
// read zero on every path, a write there changes no buffer's words, and every path then reads
// what was written. A second process of tpast, toff8 and twhole read the same.
//
// Build: clang -fobjc-arc -framework Metal -framework Foundation texture_buffer_past_length.m -o tbpl
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { W = 64, NEIGHBOURS = 128 };

static const char *SOURCE =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void rd(texture_buffer<float, access::read> t [[texture(0)]], device uint *o [[buffer(0)]],\n"
    "               uint i [[thread_position_in_grid]]) {\n"
    "  if (i < t.get_width()) o[i] = pack_float_to_unorm4x8(t.read(i));\n"
    "}\n"
    "kernel void wr(texture_buffer<float, access::write> t [[texture(0)]], constant uint &first [[buffer(0)]],\n"
    "               uint i [[thread_position_in_grid]]) {\n"
    "  if (i < t.get_width() && i >= first) t.write(unpack_unorm4x8_to_float(0xC0000000u | i), i);\n"
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

static void dispatch(NSString *name, id<MTLTexture> t, id<MTLBuffer> b)
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
	[e dispatchThreads:MTLSizeMake(W, 1, 1) threadsPerThreadgroup:MTLSizeMake(16, 1, 1)];
	[e endEncoding];
	wait_for(cb, name.UTF8String);
}

// Prints the texels as runs: a run is texels whose words name consecutive bytes of one source (or
// all read zero, or another word). Texel i's own byte is `offset + 4i`.
static void classify(const char *what, const uint32_t *v, NSUInteger offset, NSUInteger length,
                     NSArray<id<MTLBuffer>> *nb, id<MTLBuffer> target)
{
	char line[4096];
	int n = snprintf(line, sizeof line, "  %s (bytes %lu..%lu, the length %lu):", what, (unsigned long)offset,
	                 (unsigned long)(offset + W * 4), (unsigned long)length);
	NSUInteger c = 0;
	while (c < W) {
		uint32_t x = v[c];
		unsigned kind = x >> 28;
		NSUInteger start = c;
		char label[96];
		if (kind == 0xA) {
			NSUInteger o = (x & 0xFFFF) * 4;
			if (o == offset + c * 4)
				snprintf(label, sizeof label, "own");
			else
				snprintf(label, sizeof label, "tgt@%lu", (unsigned long)o);
		} else if (kind == 0xB) {
			unsigned k = (x >> 16) & 0xFFF;
			NSUInteger o = (x & 0xFFFF) * 4;
			long long rel = k < nb.count ? (long long)nb[k].gpuAddress - (long long)target.gpuAddress : 0;
			snprintf(label, sizeof label, "nb%u@%lu (its gpuAddress target%+lld)", k, (unsigned long)o, rel);
		} else if (kind == 0xC) {
			snprintf(label, sizeof label, "written(%u)", x & 0xFFFF);
		} else if (x == 0) {
			snprintf(label, sizeof label, "zero");
		} else {
			snprintf(label, sizeof label, "other(%08x)", x);
		}
		c++;
		if (x == 0)
			while (c < W && v[c] == 0)
				c++;
		else if (kind >= 0xA && kind <= 0xC)
			while (c < W && v[c] == x + (uint32_t)(c - start))
				c++;
		n += snprintf(line + n, sizeof line - n, " [%lu] %s x%lu", (unsigned long)start, label,
		              (unsigned long)(c - start));
		if (n > 3900)
			break;
	}
	printf("%s\n", line);
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
			fprintf(stderr, "usage: %s <tin|tpast|toff8|twhole|tbig> <made|read|blit|getbytes|kwrite|bwrite|replace>\n",
			        argv[0]);
			return 2;
		}
		const char *shape = argv[1], *path = argv[2];
		NSUInteger offset, length = 4096;
		if (!strcmp(shape, "tin"))
			offset = 3840;
		else if (!strcmp(shape, "tpast"))
			offset = 3968;
		else if (!strcmp(shape, "toff8"))
			offset = 3976;
		else if (!strcmp(shape, "twhole"))
			offset = 4096;
		else if (!strcmp(shape, "tbig"))
			offset = 19840, length = 20000;
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
		       "target%+lld and target%+lld; minimumTextureBufferAlignment %lu\n",
		       dev.name.UTF8String, NSProcessInfo.processInfo.operatingSystemVersionString.UTF8String, shape, path,
		       (unsigned long)target.length, (unsigned long)target.allocatedSize, before, after,
		       (unsigned long)[dev minimumTextureBufferAlignmentForPixelFormat:MTLPixelFormatRGBA8Unorm]);
		BOOL writes = !strcmp(path, "kwrite") || !strcmp(path, "bwrite") || !strcmp(path, "replace");
		MTLTextureUsage usage = writes ? (MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite) : MTLTextureUsageShaderRead;
		MTLTextureDescriptor *d = [MTLTextureDescriptor textureBufferDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
		                                                                                 width:W
		                                                                       resourceOptions:MTLResourceStorageModeShared
		                                                                                 usage:usage];
		id<MTLTexture> t = [target newTextureWithDescriptor:d offset:offset bytesPerRow:W * 4];
		printf("  texture buffer of %u texels at offset %lu (ends at %lu): %s", W, (unsigned long)offset,
		       (unsigned long)(offset + W * 4), t ? "made" : "nil\n");
		if (!t)
			return 0;
		printf(", type %lu, width %lu, allocatedSize %lu, bufferOffset %lu, bufferBytesPerRow %lu\n",
		       (unsigned long)t.textureType, (unsigned long)t.width, (unsigned long)t.allocatedSize,
		       (unsigned long)t.bufferOffset, (unsigned long)t.bufferBytesPerRow);
		if (!strcmp(path, "made"))
			return 0;
		// The first texel whose bytes are not all inside the length.
		NSUInteger first = 0;
		while (first < W && offset + first * 4 + 4 <= length)
			first++;
		const NSUInteger bytes = W * 4;
		id<MTLBuffer> out = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
		void (^read_back)(const char *) = ^(const char *how) {
			memset(out.contents, 0x5A, bytes);
			if (!strcmp(how, "read")) {
				dispatch(@"rd", t, out);
				classify(how, out.contents, offset, length, nb, target);
			} else if (!strcmp(how, "blit")) {
				id<MTLCommandBuffer> cb = [queue commandBuffer];
				id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
				[e copyFromTexture:t
				             sourceSlice:0
				             sourceLevel:0
				            sourceOrigin:MTLOriginMake(0, 0, 0)
				              sourceSize:MTLSizeMake(W, 1, 1)
				                toBuffer:out
				       destinationOffset:0
				  destinationBytesPerRow:bytes
				destinationBytesPerImage:bytes];
				[e endEncoding];
				wait_for(cb, "blit");
				classify(how, out.contents, offset, length, nb, target);
			} else {
				uint32_t *v = malloc(bytes);
				memset(v, 0x5A, bytes);
				[t getBytes:v bytesPerRow:bytes fromRegion:MTLRegionMake1D(0, W) mipmapLevel:0];
				classify(how, v, offset, length, nb, target);
				free(v);
			}
		};
		if (!writes) {
			read_back(path);
			changed(target, nb);
			return 0;
		}
		NSUInteger past = W - first;
		if (past == 0) {
			printf("  no texels past the length: nothing to write\n");
			return 0;
		}
		printf("  writing texels %lu..%lu (past the length)\n", (unsigned long)first, (unsigned long)W);
		uint32_t *src = malloc(4 * past);
		for (NSUInteger c = 0; c < past; c++)
			src[c] = 0xC0000000u | (uint32_t)(first + c);
		if (!strcmp(path, "kwrite")) {
			uint32_t f = (uint32_t)first;
			id<MTLBuffer> fb = [dev newBufferWithBytes:&f length:4 options:MTLResourceStorageModeShared];
			dispatch(@"wr", t, fb);
		} else if (!strcmp(path, "bwrite")) {
			id<MTLBuffer> sb = [dev newBufferWithBytes:src length:4 * past options:MTLResourceStorageModeShared];
			id<MTLCommandBuffer> cb = [queue commandBuffer];
			id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
			[e copyFromBuffer:sb
			           sourceOffset:0
			      sourceBytesPerRow:4 * past
			    sourceBytesPerImage:4 * past
			             sourceSize:MTLSizeMake(past, 1, 1)
			              toTexture:t
			       destinationSlice:0
			       destinationLevel:0
			      destinationOrigin:MTLOriginMake(first, 0, 0)];
			[e endEncoding];
			wait_for(cb, "bwrite");
		} else {
			[t replaceRegion:MTLRegionMake1D(first, past) mipmapLevel:0 withBytes:src bytesPerRow:4 * past];
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
