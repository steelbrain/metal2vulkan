// What Metal stores through a texture view made with a swizzle when the view is a render pass's
// colour attachment (the pass's clear and a draw) and when a plain Metal kernel writes it
// (texture2d<float, access::write>), read back from the base texture's bytes and sampled through
// the view. Our own shaders, compiled from source at run time; nothing is written to disk.
//
//   xcrun clang -fobjc-arc -framework Metal -framework Foundation \
//       -o swizzled_view_stores swizzled_view_stores.m
//   ./swizzled_view_stores              every case and path, then each case and path again in a
//                                       child process with MTL_DEBUG_LAYER=1 (its exit status; on
//                                       stderr any assertion of Metal's validation)
//   ./swizzled_view_stores CASE PATH    one case (its index) and path (render or compute)
//
// Cases: a base texture, 8 x 2, usage ShaderRead | ShaderWrite | RenderTarget | PixelFormatView,
// and a view of it made with newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:.
//   0  RG32Float over a Shared buffer (newTextureWithDescriptor:offset:bytesPerRow:), the view
//      RG32Float (Red, Green, Zero, One)
//   1  the same base, the view RG32Float (Green, Red, Zero, One)
//   2  RGBA8Unorm, Private, the view RGBA8Unorm (Blue, Green, Red, Alpha)
// Paths:
//   render   the view is colour attachment 0, its load action a clear to C1; a full-screen
//            triangle under the scissor (0, 0, 4, 2) writes C2, so texels x < 4 are drawn and
//            texels x >= 4 only cleared.
//   compute  the base's bytes are 0xee; a kernel writes C2 through the view at texels x < 4.
//   C1 = (0x1a, 0x4d, 0x99, 0xe6) / 255     C2 = (0x33, 0x66, 0xb3, 0xcc) / 255
// Then the base's bytes are read (the buffer's bytes for a texture over a buffer, a blit to a
// buffer otherwise) and a kernel samples every texel through the view (nearest, at its centre).
//
// A line per kind of texel (clear, draw, write; untouched: x >= 4 after the write) prints the
// base's bytes of texel (x0, 0), one group a channel in memory order, the channels of the view's
// format as lanes of the colour (r, g, b, a of C1 or C2, 0 or 1, ? for none), how many texels of
// the kind hold the same bytes, and the reading:
//   as-written   channel c holds lane c;
//   permuted     channel c holds lane l where the swizzle's lane l names c (the write undoes the
//                swizzle a read applies), or channel c holds lane s[c] (the swizzle applied to
//                the written colour); for a swap the two are one permutation;
//   neither      none of these;
// for an identity on the format's channels the readings coincide and the line says so. Then the
// sample's four lanes, whether they are the read rule's (lane l is the stored channel the
// swizzle's lane l names, a missing channel 0 or alpha 1), and whether they are the colour as
// written.
//
// WHAT IT MEASURED, Apple M3 Max, macOS 27.0.1 (potluck job m3-max-76829, MTL_DEBUG_LAYER unset):
//   render   every case, clear and draw: as-written, on all 8 texels of each kind. Case 0: clear
//            d1d0d03d.9b9a9a3e (0.1020, 0.3020), draw cdcc4c3e.cdcccc3e (0.2, 0.4), sampled
//            r, g, 0, 1. Case 1: the same bytes, sampled g, r, 0, 1. Case 2: clear 1a.4d.99.e6,
//            draw 33.66.b3.cc, sampled b, g, r, a. Every sample is the read rule's over the
//            stored texel.
//   compute  case 0 as-written (cdcc4c3e.cdcccc3e); case 1 permuted, channels g, r
//            (cdcccc3e.cdcc4c3e); case 2 permuted, channels b, g, r, a (b3.66.33.cc). The sample
//            through the view then answers the colour as written. Texels x >= 4 keep 0xee.
//   usage    every view's usage is 0x11 (ShaderRead | PixelFormatView) over a base of 0x17.
//   layer    with MTL_DEBUG_LAYER=1 every case's render path ends the process (signal 6) with
//            "Texture at colorAttachment[0] has usage (0x11) which doesn't specify
//            MTLTextureUsageRenderTarget (0x04)", and every compute path with "Compute
//            Function(kw): writes texture (t[0]) whose usage (0x11) doesn't specify
//            MTLTextureUsageShaderWrite (0x02)".
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <spawn.h>
#include <sys/wait.h>

extern char **environ;

enum { kW = 8, kH = 2, kN = kW * kH, kSplit = 4 };

static const float kC1[4] = {0x1a / 255.0f, 0x4d / 255.0f, 0x99 / 255.0f, 0xe6 / 255.0f};
static const float kC2[4] = {0x33 / 255.0f, 0x66 / 255.0f, 0xb3 / 255.0f, 0xcc / 255.0f};
static const char kLanes[4] = {'r', 'g', 'b', 'a'};
static const char *gLayer = "off";

typedef struct {
	MTLPixelFormat format;
	const char *name;
	int channelSize; // 1: unorm8, 4: float32
	const char *order; // the channels in memory order
} Format;

static const Format kFormats[] = {
    {MTLPixelFormatRG32Float, "RG32Float", 4, "RG"},
    {MTLPixelFormatRGBA8Unorm, "RGBA8Unorm", 1, "RGBA"},
};

typedef struct {
	const char *name;
	MTLPixelFormat base, view;
	MTLTextureSwizzleChannels swizzle;
	BOOL overBuffer;
} Case;

#define S(r, g, b, a) {MTLTextureSwizzle##r, MTLTextureSwizzle##g, MTLTextureSwizzle##b, MTLTextureSwizzle##a}
static const Case kCases[] = {
    {"rg32f-buffer/rg01", MTLPixelFormatRG32Float, MTLPixelFormatRG32Float, S(Red, Green, Zero, One), YES},
    {"rg32f-buffer/gr01", MTLPixelFormatRG32Float, MTLPixelFormatRG32Float, S(Green, Red, Zero, One), YES},
    {"rgba8-private/bgra", MTLPixelFormatRGBA8Unorm, MTLPixelFormatRGBA8Unorm, S(Blue, Green, Red, Alpha), NO},
};
enum { kCaseCount = sizeof kCases / sizeof kCases[0] };

static const MTLTextureUsage kUsage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite |
                                      MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView;

static NSString *const kMsl = @"#include <metal_stdlib>\n"
                               "using namespace metal;\n"
                               "constant float4 c2 = float4(0x33, 0x66, 0xb3, 0xcc) / 255.0f;\n"
                               "vertex float4 vs(uint v [[vertex_id]]) {\n"
                               "  float2 p = float2((v << 1) & 2, v & 2);\n"
                               "  return float4(p * 2.0f - 1.0f, 0.0f, 1.0f); }\n"
                               "fragment float4 fs() { return c2; }\n"
                               "kernel void kw(texture2d<float, access::write> t [[texture(0)]],\n"
                               "               uint2 g [[thread_position_in_grid]]) {\n"
                               "  if (g.x < 4) t.write(c2, g); }\n"
                               "kernel void ks(texture2d<float, access::sample> t [[texture(0)]],\n"
                               "               device float4 *o [[buffer(0)]], uint2 g [[thread_position_in_grid]]) {\n"
                               "  constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_edge);\n"
                               "  o[g.y * 8 + g.x] = t.sample(s, float2(g) + 0.5f); }\n";

typedef struct {
	id<MTLDevice> d;
	id<MTLCommandQueue> q;
	id<MTLLibrary> lib;
	id<MTLComputePipelineState> write, sample;
} Ctx;

static const char *oneLine(NSError *e)
{
	if (e == nil)
		return "(none)";
	return [[e.description componentsSeparatedByCharactersInSet:NSCharacterSet.newlineCharacterSet]
	           componentsJoinedByString:@" "]
	    .UTF8String;
}

static const Format *formatOf(MTLPixelFormat f)
{
	for (size_t i = 0; i < sizeof kFormats / sizeof kFormats[0]; i++)
		if (kFormats[i].format == f)
			return &kFormats[i];
	return NULL;
}

static int bpp(const Format *f) { return f->channelSize * (int)strlen(f->order); }

static int channelOf(char c) { return c == 'R' ? 0 : c == 'G' ? 1 : c == 'B' ? 2 : 3; }

static char source(uint8_t s)
{
	return s == MTLTextureSwizzleZero ? '0' : s == MTLTextureSwizzleOne ? '1' : kLanes[s - MTLTextureSwizzleRed];
}

// One texel of `f` at `p`: v[c] for each channel c (R, G, B, A) the format has.
static void decode(const Format *f, const uint8_t *p, float v[4], bool present[4])
{
	for (int c = 0; c < 4; c++)
		present[c] = false, v[c] = NAN;
	for (int i = 0; f->order[i]; i++) {
		int c = channelOf(f->order[i]);
		const uint8_t *q = p + i * f->channelSize;
		if (f->channelSize == 1)
			v[c] = q[0] / 255.0f;
		else
			memcpy(&v[c], q, 4);
		present[c] = true;
	}
}

// The lane of `ref` that `x` is, or 0 or 1, or ? for none.
static char lane(float x, const float ref[4])
{
	for (int l = 0; l < 4; l++)
		if (fabsf(x - ref[l]) < 0.02f)
			return kLanes[l];
	if (fabsf(x) < 0.02f)
		return '0';
	if (fabsf(x - 1.0f) < 0.02f)
		return '1';
	return '?';
}

static void printBytes(const Format *f, const uint8_t *p)
{
	for (int i = 0; f->order[i]; i++) {
		for (int b = 0; b < f->channelSize; b++)
			printf("%02x", p[i * f->channelSize + b]);
		printf(f->order[i + 1] ? "." : "");
	}
}

// Prints one kind of texel, x in [x0, x1), of the base's bytes `bytes` and the samples `samples`.
static void printKind(const char *head, const char *kind, const Case *k, const uint8_t *bytes,
                      const float *samples, int x0, int x1, const float ref[4])
{
	const Format *vf = formatOf(k->view);
	int b = bpp(vf);
	const uint8_t *first = bytes + x0 * b;
	int same = 0, count = 0;
	for (int y = 0; y < kH; y++)
		for (int x = x0; x < x1; x++) {
			count++;
			same += memcmp(bytes + (y * kW + x) * b, first, b) == 0;
		}
	printf("%s texel=%s x0=%d bytes(%s)=", head, kind, x0, vf->order);
	printBytes(vf, first);
	printf(" same=%d/%d", same, count);
	if (ref == NULL) {
		printf("\n");
		return;
	}
	float v[4];
	bool present[4];
	decode(vf, first, v, present);
	const uint8_t *s = (const uint8_t *)&k->swizzle;
	char stored[5] = "----", written[5] = "----", inverse[5] = "----", forward[5] = "----";
	for (int c = 0; c < 4; c++) {
		if (!present[c])
			continue;
		stored[c] = lane(v[c], ref);
		written[c] = kLanes[c];
		inverse[c] = kLanes[c];
		for (int l = 0; l < 4; l++)
			if (s[l] == MTLTextureSwizzleRed + c) {
				inverse[c] = kLanes[l];
				break;
			}
		forward[c] = source(s[c]);
	}
	printf(" values=");
	for (int c = 0; c < 4; c++)
		if (present[c])
			printf("%s%.4f", c ? "," : "", v[c]);
	printf(" stored(rgba)=%s reading=", stored);
	bool w = !strcmp(stored, written), i = !strcmp(stored, inverse), f = !strcmp(stored, forward);
	if (w && i && f)
		printf("as-written(the readings coincide)");
	else if (w)
		printf("as-written");
	else if (i && f)
		printf("permuted(%s)", inverse);
	else if (i)
		printf("permuted(write-undoes-read:%s)", inverse);
	else if (f)
		printf("permuted(swizzle-applied:%s)", forward);
	else
		printf("neither");
	const float *o = samples + 4 * x0;
	// rule: the read rule over the stored texel; plain: an identity read of the colour as written.
	char sampled[5] = "", rule[5] = "", plain[5] = "";
	for (int l = 0; l < 4; l++) {
		sampled[l] = lane(o[l], ref);
		int c = s[l] - MTLTextureSwizzleRed;
		rule[l] = s[l] < MTLTextureSwizzleRed ? source(s[l]) : present[c] ? stored[c] : c == 3 ? '1' : '0';
		plain[l] = present[l] ? kLanes[l] : l == 3 ? '1' : '0';
	}
	printf(" sampled=%.4f,%.4f,%.4f,%.4f lanes=%s read_rule(%s)=%s as_written(%s)=%s\n", o[0], o[1], o[2], o[3],
	       sampled, rule, strcmp(sampled, rule) ? "no" : "yes", plain, strcmp(sampled, plain) ? "no" : "yes");
}

// The base and its swizzled view, with the base's buffer for a texture over one.
static id<MTLTexture> makeBase(Ctx *x, const Case *k, id<MTLBuffer> *buffer, NSUInteger *rowBytes)
{
	MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:k->base
	                                                                              width:kW
	                                                                             height:kH
	                                                                          mipmapped:NO];
	td.usage = kUsage;
	*buffer = nil;
	*rowBytes = kW * bpp(formatOf(k->base));
	if (!k->overBuffer) {
		td.storageMode = MTLStorageModePrivate;
		return [x->d newTextureWithDescriptor:td];
	}
	NSUInteger align = [x->d minimumLinearTextureAlignmentForPixelFormat:k->base];
	*rowBytes = (*rowBytes + align - 1) / align * align;
	*buffer = [x->d newBufferWithLength:*rowBytes * kH options:0];
	td.storageMode = MTLStorageModeShared;
	return [*buffer newTextureWithDescriptor:td offset:0 bytesPerRow:*rowBytes];
}

static id<MTLTexture> makeView(id<MTLTexture> base, const Case *k)
{
	return [base newTextureViewWithPixelFormat:k->view
	                               textureType:MTLTextureType2D
	                                    levels:NSMakeRange(0, 1)
	                                    slices:NSMakeRange(0, 1)
	                                   swizzle:k->swizzle];
}

static bool finish(const char *head, const char *what, id<MTLCommandBuffer> cb)
{
	[cb commit];
	[cb waitUntilCompleted];
	if (cb.status == MTLCommandBufferStatusCompleted)
		return true;
	printf("%s %s status=%ld error=%s\n", head, what, (long)cb.status, oneLine(cb.error));
	return false;
}

// Reads the base's bytes (tight rows) into `bytes`, and samples every texel through the view.
static bool readBack(Ctx *x, const char *head, const Case *k, id<MTLTexture> base, id<MTLBuffer> buffer,
                     NSUInteger rowBytes, id<MTLTexture> view, uint8_t *bytes, float *samples)
{
	int b = bpp(formatOf(k->base));
	id<MTLBuffer> copy = [x->d newBufferWithLength:kN * b options:MTLResourceStorageModeShared];
	id<MTLBuffer> out = [x->d newBufferWithLength:16 * kN options:MTLResourceStorageModeShared];
	memset(out.contents, 0xcd, out.length);
	id<MTLCommandBuffer> cb = [x->q commandBuffer];
	if (buffer == nil) {
		id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
		[be copyFromTexture:base
		                 sourceSlice:0
		                 sourceLevel:0
		                sourceOrigin:MTLOriginMake(0, 0, 0)
		                  sourceSize:MTLSizeMake(kW, kH, 1)
		                    toBuffer:copy
		           destinationOffset:0
		      destinationBytesPerRow:kW * b
		    destinationBytesPerImage:kN * b];
		[be endEncoding];
	}
	id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
	[ce setComputePipelineState:x->sample];
	[ce setTexture:view atIndex:0];
	[ce setBuffer:out offset:0 atIndex:0];
	[ce dispatchThreads:MTLSizeMake(kW, kH, 1) threadsPerThreadgroup:MTLSizeMake(kW, kH, 1)];
	[ce endEncoding];
	if (!finish(head, "readback", cb))
		return false;
	for (int y = 0; y < kH; y++) {
		const uint8_t *row = buffer ? (const uint8_t *)buffer.contents + y * rowBytes
		                            : (const uint8_t *)copy.contents + y * kW * b;
		memcpy(bytes + y * kW * b, row, kW * b);
	}
	memcpy(samples, out.contents, 16 * kN);
	return true;
}

static void describe(const char *head, id<MTLTexture> base, id<MTLTexture> view, NSUInteger rowBytes)
{
	MTLTextureSwizzleChannels s = view.swizzle;
	printf("%s base_format=%lu base_usage=%#lx base_storage=%lu base_buffer=%s row_bytes=%lu view_format=%lu "
	       "view_usage=%#lx view_swizzle=%c%c%c%c\n",
	       head, (unsigned long)base.pixelFormat, (unsigned long)base.usage, (unsigned long)base.storageMode,
	       base.buffer ? "yes" : "no", (unsigned long)rowBytes, (unsigned long)view.pixelFormat,
	       (unsigned long)view.usage, source(s.red), source(s.green), source(s.blue), source(s.alpha));
}

static void runRender(Ctx *x, int ci)
{
	const Case *k = &kCases[ci];
	char head[128];
	snprintf(head, sizeof head, "layer=%s case=%d(%s) path=render", gLayer, ci, k->name);
	id<MTLBuffer> buffer;
	NSUInteger rowBytes;
	id<MTLTexture> base = makeBase(x, k, &buffer, &rowBytes), view = base ? makeView(base, k) : nil;
	if (view == nil) {
		printf("%s base=%s view=nil\n", head, base ? "yes" : "nil");
		return;
	}
	describe(head, base, view, rowBytes);
	MTLRenderPipelineDescriptor *rp = [MTLRenderPipelineDescriptor new];
	rp.vertexFunction = [x->lib newFunctionWithName:@"vs"];
	rp.fragmentFunction = [x->lib newFunctionWithName:@"fs"];
	rp.colorAttachments[0].pixelFormat = k->view;
	NSError *e = nil;
	id<MTLRenderPipelineState> ps = [x->d newRenderPipelineStateWithDescriptor:rp error:&e];
	if (ps == nil) {
		printf("%s pipeline=nil error=%s\n", head, oneLine(e));
		return;
	}
	MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
	pass.colorAttachments[0].texture = view;
	pass.colorAttachments[0].loadAction = MTLLoadActionClear;
	pass.colorAttachments[0].storeAction = MTLStoreActionStore;
	pass.colorAttachments[0].clearColor = MTLClearColorMake(kC1[0], kC1[1], kC1[2], kC1[3]);
	id<MTLCommandBuffer> cb = [x->q commandBuffer];
	id<MTLRenderCommandEncoder> re = [cb renderCommandEncoderWithDescriptor:pass];
	[re setRenderPipelineState:ps];
	[re setScissorRect:(MTLScissorRect){0, 0, kSplit, kH}];
	[re drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
	[re endEncoding];
	if (!finish(head, "draw", cb))
		return;
	uint8_t bytes[kN * 16];
	float samples[4 * kN];
	if (!readBack(x, head, k, base, buffer, rowBytes, view, bytes, samples))
		return;
	printKind(head, "clear", k, bytes, samples, kSplit, kW, kC1);
	printKind(head, "draw", k, bytes, samples, 0, kSplit, kC2);
}

static void runCompute(Ctx *x, int ci)
{
	const Case *k = &kCases[ci];
	char head[128];
	snprintf(head, sizeof head, "layer=%s case=%d(%s) path=compute", gLayer, ci, k->name);
	id<MTLBuffer> buffer;
	NSUInteger rowBytes;
	id<MTLTexture> base = makeBase(x, k, &buffer, &rowBytes), view = base ? makeView(base, k) : nil;
	if (view == nil) {
		printf("%s base=%s view=nil\n", head, base ? "yes" : "nil");
		return;
	}
	describe(head, base, view, rowBytes);
	int b = bpp(formatOf(k->base));
	id<MTLCommandBuffer> cb = [x->q commandBuffer];
	if (buffer) {
		memset(buffer.contents, 0xee, buffer.length);
	} else {
		id<MTLBuffer> fill = [x->d newBufferWithLength:kN * b options:MTLResourceStorageModeShared];
		memset(fill.contents, 0xee, fill.length);
		id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
		[be copyFromBuffer:fill
		           sourceOffset:0
		      sourceBytesPerRow:kW * b
		    sourceBytesPerImage:kN * b
		             sourceSize:MTLSizeMake(kW, kH, 1)
		              toTexture:base
		       destinationSlice:0
		       destinationLevel:0
		      destinationOrigin:MTLOriginMake(0, 0, 0)];
		[be endEncoding];
	}
	if (!finish(head, "fill", cb))
		return;
	cb = [x->q commandBuffer];
	id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
	[ce setComputePipelineState:x->write];
	[ce setTexture:view atIndex:0];
	[ce dispatchThreads:MTLSizeMake(kW, kH, 1) threadsPerThreadgroup:MTLSizeMake(kW, kH, 1)];
	[ce endEncoding];
	if (!finish(head, "write", cb))
		return;
	uint8_t bytes[kN * 16];
	float samples[4 * kN];
	if (!readBack(x, head, k, base, buffer, rowBytes, view, bytes, samples))
		return;
	printKind(head, "write", k, bytes, samples, 0, kSplit, kC2);
	printKind(head, "untouched", k, bytes, samples, kSplit, kW, NULL);
}

static bool setUp(Ctx *x)
{
	x->d = MTLCreateSystemDefaultDevice();
	if (x->d == nil) {
		printf("layer=%s device=none\n", gLayer);
		return false;
	}
	printf("layer=%s device=%s os=%s\n", gLayer, x->d.name.UTF8String,
	       NSProcessInfo.processInfo.operatingSystemVersionString.UTF8String);
	x->q = [x->d newCommandQueue];
	NSError *e = nil;
	x->lib = [x->d newLibraryWithSource:kMsl options:nil error:&e];
	if (x->lib == nil) {
		printf("layer=%s library=nil error=%s\n", gLayer, oneLine(e));
		return false;
	}
	x->write = [x->d newComputePipelineStateWithFunction:[x->lib newFunctionWithName:@"kw"] error:&e];
	x->sample = [x->d newComputePipelineStateWithFunction:[x->lib newFunctionWithName:@"ks"] error:&e];
	if (x->write == nil || x->sample == nil) {
		printf("layer=%s compute_pipeline=nil error=%s\n", gLayer, oneLine(e));
		return false;
	}
	return true;
}

// Runs case `ci`, path `path`, in a child process with MTL_DEBUG_LAYER=1.
static void validated(const char *self, int ci, const char *path)
{
	int n = 0;
	while (environ[n])
		n++;
	char **env = calloc(n + 2, sizeof *env);
	int m = 0;
	for (int i = 0; i < n; i++)
		if (strncmp(environ[i], "MTL_DEBUG_LAYER=", 16) != 0)
			env[m++] = environ[i];
	env[m++] = "MTL_DEBUG_LAYER=1";
	char index[16];
	snprintf(index, sizeof index, "%d", ci);
	char *argv[] = {(char *)self, index, (char *)path, NULL};
	pid_t pid;
	int status = posix_spawn(&pid, self, NULL, NULL, argv, env);
	free(env);
	if (status != 0) {
		printf("layer=on case=%d(%s) path=%s spawn=%d\n", ci, kCases[ci].name, path, status);
		return;
	}
	waitpid(pid, &status, 0);
	printf("layer=on case=%d(%s) path=%s exit=%d signal=%d\n", ci, kCases[ci].name, path,
	       WIFEXITED(status) ? WEXITSTATUS(status) : -1, WIFSIGNALED(status) ? WTERMSIG(status) : 0);
}

int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IOLBF, 0);
	@autoreleasepool {
		const char *layer = getenv("MTL_DEBUG_LAYER");
		gLayer = layer && *layer && strcmp(layer, "0") ? "on" : "off";
		Ctx x;
		if (!setUp(&x))
			return 1;
		if (argc == 3) {
			int ci = atoi(argv[1]);
			if (ci < 0 || ci >= kCaseCount)
				return 2;
			if (!strcmp(argv[2], "render"))
				runRender(&x, ci);
			else
				runCompute(&x, ci);
			return 0;
		}
		for (int ci = 0; ci < kCaseCount; ci++) {
			runRender(&x, ci);
			runCompute(&x, ci);
		}
		for (int ci = 0; ci < kCaseCount; ci++) {
			validated(argv[0], ci, "render");
			validated(argv[0], ci, "compute");
		}
	}
	return 0;
}
