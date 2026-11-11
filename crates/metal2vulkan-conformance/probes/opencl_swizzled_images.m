// What Metal answers for a swizzled texture view a kernel writes, reads, samples or only queries,
// in plain Metal and in OpenCL mode, and what Apple's OpenCL binds for an image of each channel
// order. Our own kernels; nothing is written to disk.
//
//   xcrun clang -fobjc-arc -framework Metal -framework Foundation -framework OpenCL \
//       -o opencl_swizzled_images opencl_swizzled_images.m
//   ./opencl_swizzled_images [DEVICE]      (DEVICE: a substring of the Metal device's name)
//
// Parts:
//   trace  Apple's OpenCL (the API) on the GPU, a 4 x 2 image of each channel order
//          it supports with CL_UNORM_INT8, through three kernels (`cq` queries a write-only image,
//          `cw` writes one, `cr` reads a read-only one): the texture each `setTexture:atIndex:` of
//          the process's compute encoders binds (format, type, size, usage, swizzle, parent), and
//          what the kernels answer through the API.
//   views  Over a 4 x 2 view of an RGBA8Unorm or R8Unorm texture, made with each swizzle below
//          (`newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:`): plain Metal's
//          `mw` (access::write), `mrww` (read_write, written), `mrwr` (read_write, read), `mr`
//          (access::read), `ms` (access::sample, nearest, at the texel's centre) and `mq`
//          (access::write, only get_width and get_height); then OpenCL mode driven directly (the
//          library of OpenCL C, the variant `newFunctionWithPluginData:bitcodeType:2`, a pipeline
//          with `openCLModeEnabled`, the three hidden buffers bound, the third's word i
//          `i * 0x10000 + 0x1234`): `cw`, `cq`, `cr` (a read with no sampler and one through a
//          constant nearest sampler) and, where the compiler takes OpenCL C 2.0, `crw` (a
//          `__read_write` image read then written).
//
// Texel i (x + 4y) of a filled texture holds bytes (0x10 + i, 0x50 + i, 0x90 + i, 0xd0 + i)
// (R8Unorm: 0x10 + i); a write stores (0x11 + i, 0x51 + i, 0x91 + i, 0xd1 + i) / 255 into a
// texture filled with 0xee. Each line prints the parent's bytes after a write, or each read lane as
// its byte (the float times 255, rounded) and the words in hex, and `rule=` says whether the
// answer is the one the rule below predicts:
//   read   lane l is the channel the swizzle's l names (a format's missing green and blue 0, its
//          missing alpha 1), Zero 0 and One 1;
//   write  channel c takes the first lane whose source is c, and lane c where no lane names it.
#define CL_SILENCE_DEPRECATION
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <OpenCL/opencl.h>
#include <objc/message.h>
#include <objc/runtime.h>
#include <math.h>

enum { kW = 4, kH = 2, kN = kW * kH };

static const char *oneLine(NSError *e)
{
	if (e == nil)
		return "(none)";
	return [[e.description componentsSeparatedByCharactersInSet:NSCharacterSet.newlineCharacterSet]
	           componentsJoinedByString:@" "]
	    .UTF8String;
}

// --- part 1: Apple's OpenCL, traced ----------------------------------------------------------------

static IMP gSetTexture;
static int gTraced;

static void describe(const char *what, id<MTLTexture> t, NSUInteger i)
{
	if (t == nil) {
		printf("  %s index=%lu texture=nil\n", what, (unsigned long)i);
		return;
	}
	MTLTextureSwizzleChannels s = t.swizzle;
	id<MTLTexture> p = t.parentTexture;
	printf("  %s index=%lu format=%lu type=%lu size=%lux%lux%lu usage=%#lx swizzle=%u,%u,%u,%u storage=%lu parent=%s",
	       what, (unsigned long)i, (unsigned long)t.pixelFormat, (unsigned long)t.textureType, (unsigned long)t.width,
	       (unsigned long)t.height, (unsigned long)t.depth, (unsigned long)t.usage, s.red, s.green, s.blue, s.alpha,
	       (unsigned long)t.storageMode, p ? "yes" : "no");
	if (p)
		printf(" parent_format=%lu parent_usage=%#lx parent_swizzle=%u,%u,%u,%u", (unsigned long)p.pixelFormat,
		       (unsigned long)p.usage, p.swizzle.red, p.swizzle.green, p.swizzle.blue, p.swizzle.alpha);
	printf(" buffer=%s\n", t.buffer ? "yes" : "no");
}

static void w_setTexture(id self, SEL cmd, id<MTLTexture> t, NSUInteger i)
{
	gTraced++;
	describe("setTexture", t, i);
	((void (*)(id, SEL, id, NSUInteger))gSetTexture)(self, cmd, t, i);
}

// Wraps setTexture:atIndex: on the class in `c`'s chain that owns it.
static void wrapSetTexture(Class c)
{
	SEL sel = @selector(setTexture:atIndex:);
	for (Class k = c; k != Nil; k = class_getSuperclass(k)) {
		unsigned m = 0;
		Method *methods = class_copyMethodList(k, &m);
		for (unsigned j = 0; j < m; j++) {
			if (sel_isEqual(method_getName(methods[j]), sel)) {
				gSetTexture = method_setImplementation(methods[j], (IMP)w_setTexture);
				printf("# wrapped -[%s setTexture:atIndex:]\n", class_getName(k));
				free(methods);
				return;
			}
		}
		free(methods);
	}
	printf("# no owner of setTexture:atIndex: in %s's chain\n", class_getName(c));
}

static const char *kClSource =
    "__constant sampler_t cs = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;\n"
    "__kernel void cw(__write_only image2d_t img) {\n"
    "  int x = get_global_id(0), y = get_global_id(1), i = y * 4 + x;\n"
    "  write_imagef(img, (int2)(x, y), (float4)(0x11 + i, 0x51 + i, 0x91 + i, 0xd1 + i) / 255.0f); }\n"
    "__kernel void cq(__write_only image2d_t img, __global int *o) {\n"
    "  if (get_global_id(0) == 0 && get_global_id(1) == 0) {\n"
    "    o[0] = get_image_width(img); o[1] = get_image_height(img);\n"
    "    o[2] = get_image_channel_order(img); o[3] = get_image_channel_data_type(img); } }\n"
    "__kernel void cr(__read_only image2d_t img, __global float4 *o) {\n"
    "  int x = get_global_id(0), y = get_global_id(1), i = y * 4 + x;\n"
    "  o[2 * i] = read_imagef(img, (int2)(x, y)); o[2 * i + 1] = read_imagef(img, cs, (int2)(x, y)); }\n";

static const char *kClRwSource =
    "__kernel void crw(__read_write image2d_t img, __global float4 *o) {\n"
    "  int x = get_global_id(0), y = get_global_id(1), i = y * 4 + x;\n"
    "  o[i] = read_imagef(img, (int2)(x, y));\n"
    "  write_imagef(img, (int2)(x, y), (float4)(0x11 + i, 0x51 + i, 0x91 + i, 0xd1 + i) / 255.0f); }\n";

static const struct {
	cl_channel_order order;
	const char *name;
} kOrders[] = {
    {CL_R, "R"},       {CL_A, "A"},         {CL_RG, "RG"},
    {CL_RA, "RA"},     {CL_RGB, "RGB"},     {CL_RGBA, "RGBA"},
    {CL_BGRA, "BGRA"}, {CL_ARGB, "ARGB"},   {CL_INTENSITY, "INTENSITY"},
    {CL_LUMINANCE, "LUMINANCE"}, {CL_Rx, "Rx"}, {CL_RGx, "RGx"},
    {CL_RGBx, "RGBx"},
};

static const char *orderName(cl_channel_order o)
{
	for (size_t i = 0; i < sizeof kOrders / sizeof kOrders[0]; i++)
		if (kOrders[i].order == o)
			return kOrders[i].name;
	return "?";
}

static int tracePart(id<MTLDevice> d, const char *want)
{
	{
		id<MTLCommandBuffer> cb = [[d newCommandQueue] commandBuffer];
		id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
		wrapSetTexture(object_getClass(e));
		[e endEncoding];
	}
	cl_platform_id platform;
	if (clGetPlatformIDs(1, &platform, NULL) != CL_SUCCESS) {
		printf("trace opencl=none\n");
		return 1;
	}
	cl_device_id devs[8], dev = NULL;
	cl_uint n = 0;
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 8, devs, &n);
	for (cl_uint i = 0; i < n; i++) {
		char name[256] = "";
		clGetDeviceInfo(devs[i], CL_DEVICE_NAME, sizeof name, name, NULL);
		printf("# OpenCL GPU %u: %s\n", i, name);
		if (dev == NULL && (want == NULL || strcasestr(name, want)))
			dev = devs[i];
	}
	if (dev == NULL) {
		printf("trace device=none\n");
		return 1;
	}
	cl_int err = 0;
	cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
	cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &err);
	cl_program prog = clCreateProgramWithSource(ctx, 1, &kClSource, NULL, &err);
	err = clBuildProgram(prog, 1, &dev, "", NULL, NULL);
	printf("trace build=%d\n", err);
	if (err != CL_SUCCESS)
		return 1;
	cl_image_format formats[256];
	cl_uint nf = 0;
	clGetSupportedImageFormats(ctx, CL_MEM_READ_WRITE, CL_MEM_OBJECT_IMAGE2D, 256, formats, &nf);
	printf("trace unorm_int8_orders=");
	for (cl_uint i = 0; i < nf; i++)
		if (formats[i].image_channel_data_type == CL_UNORM_INT8)
			printf("%s(%#x) ", orderName(formats[i].image_channel_order), formats[i].image_channel_order);
	printf("\n");
	cl_kernel cw = clCreateKernel(prog, "cw", &err), cq = clCreateKernel(prog, "cq", &err),
	          cr = clCreateKernel(prog, "cr", &err);
	cl_mem out = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 16 * 2 * kN, NULL, &err);
	for (cl_uint f = 0; f < nf; f++) {
		if (formats[f].image_channel_data_type != CL_UNORM_INT8)
			continue;
		cl_image_format fmt = formats[f];
		const char *on = orderName(fmt.image_channel_order);
		cl_image_desc desc = {0};
		desc.image_type = CL_MEM_OBJECT_IMAGE2D;
		desc.image_width = kW;
		desc.image_height = kH;
		uint8_t host[kN * 4];
		for (int i = 0; i < kN * 4; i++)
			host[i] = (uint8_t)(0x10 + 0x40 * (i % 4) + (i / 4));
		cl_mem img = clCreateImage(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, &fmt, &desc, host, &err);
		if (img == NULL) {
			printf("trace order=%s image=%d\n", on, err);
			continue;
		}
		size_t element = 0;
		clGetImageInfo(img, CL_IMAGE_ELEMENT_SIZE, sizeof element, &element, NULL);
		size_t global[2] = {kW, kH};
		struct {
			const char *name;
			cl_kernel k;
		} runs[3] = {{"cq", cq}, {"cr", cr}, {"cw", cw}};
		for (int r = 0; r < 3; r++) {
			clSetKernelArg(runs[r].k, 0, sizeof img, &img);
			if (runs[r].k != cw)
				clSetKernelArg(runs[r].k, 1, sizeof out, &out);
			printf("trace order=%s element=%zu kernel=%s\n", on, element, runs[r].name);
			int before = gTraced;
			err = clEnqueueNDRangeKernel(q, runs[r].k, 2, NULL, global, NULL, 0, NULL, NULL);
			clFinish(q);
			printf("  enqueue=%d traced=%d\n", err, gTraced - before);
			if (runs[r].k == cq) {
				int o[4] = {0};
				clEnqueueReadBuffer(q, out, CL_TRUE, 0, sizeof o, o, 0, NULL, NULL);
				printf("  answers width=%d height=%d order=%#x type=%#x\n", o[0], o[1], o[2], o[3]);
			} else if (runs[r].k == cr) {
				float o[2 * kN * 4];
				clEnqueueReadBuffer(q, out, CL_TRUE, 0, sizeof o, o, 0, NULL, NULL);
				printf("  answers texel0 nosampler=%g,%g,%g,%g sampler=%g,%g,%g,%g\n", o[0] * 255, o[1] * 255,
				       o[2] * 255, o[3] * 255, o[4] * 255, o[5] * 255, o[6] * 255, o[7] * 255);
			} else {
				uint8_t back[kN * 4] = {0};
				size_t origin[3] = {0, 0, 0}, region[3] = {kW, kH, 1};
				clEnqueueReadImage(q, img, CL_TRUE, origin, region, 0, 0, back, 0, NULL, NULL);
				printf("  answers texel0 bytes=");
				for (size_t b = 0; b < element; b++)
					printf("%02x", back[b]);
				printf("\n");
			}
		}
		clReleaseMemObject(img);
	}
	return 0;
}

// --- part 2: swizzled views -------------------------------------------------------------------------

static NSString *const kMsl = @"#include <metal_stdlib>\n"
                               "using namespace metal;\n"
                               "static float4 val(uint2 g) { uint i = g.y * 4 + g.x;\n"
                               "  return float4(0x11 + i, 0x51 + i, 0x91 + i, 0xd1 + i) / 255.0f; }\n"
                               "kernel void mw(texture2d<float, access::write> t [[texture(0)]], uint2 g [[thread_position_in_grid]]) {\n"
                               "  t.write(val(g), g); }\n"
                               "kernel void mrww(texture2d<float, access::read_write> t [[texture(0)]], uint2 g [[thread_position_in_grid]]) {\n"
                               "  t.write(val(g), g); }\n"
                               "kernel void mrwr(texture2d<float, access::read_write> t [[texture(0)]], device float4 *o [[buffer(0)]],\n"
                               "                 uint2 g [[thread_position_in_grid]]) { o[g.y * 4 + g.x] = t.read(g); }\n"
                               "kernel void mr(texture2d<float, access::read> t [[texture(0)]], device float4 *o [[buffer(0)]],\n"
                               "               uint2 g [[thread_position_in_grid]]) { o[g.y * 4 + g.x] = t.read(g); }\n"
                               "kernel void ms(texture2d<float, access::sample> t [[texture(0)]], device float4 *o [[buffer(0)]],\n"
                               "               uint2 g [[thread_position_in_grid]]) {\n"
                               "  constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_edge);\n"
                               "  o[g.y * 4 + g.x] = t.sample(s, float2(g) + 0.5f); }\n"
                               "kernel void mq(texture2d<float, access::write> t [[texture(0)]], device uint *o [[buffer(0)]],\n"
                               "               uint2 g [[thread_position_in_grid]]) {\n"
                               "  if (g.x == 0 && g.y == 0) { o[0] = t.get_width(); o[1] = t.get_height(); } }\n";

typedef struct {
	const char *name;
	MTLPixelFormat format;
	int channels;
	MTLTextureSwizzleChannels swizzle;
} View;

#define S(r, g, b, a) {MTLTextureSwizzle##r, MTLTextureSwizzle##g, MTLTextureSwizzle##b, MTLTextureSwizzle##a}
static const View kViews[] = {
    {"rgba8/identity", MTLPixelFormatRGBA8Unorm, 4, S(Red, Green, Blue, Alpha)},
    {"rgba8/BGRA", MTLPixelFormatRGBA8Unorm, 4, S(Blue, Green, Red, Alpha)},
    {"rgba8/ARGB", MTLPixelFormatRGBA8Unorm, 4, S(Green, Blue, Alpha, Red)},
    {"rgba8/A", MTLPixelFormatRGBA8Unorm, 4, S(Zero, Zero, Zero, Red)},
    {"rgba8/INTENSITY", MTLPixelFormatRGBA8Unorm, 4, S(Red, Red, Red, Red)},
    {"rgba8/LUMINANCE", MTLPixelFormatRGBA8Unorm, 4, S(Red, Red, Red, One)},
    {"r8/A", MTLPixelFormatR8Unorm, 1, S(Zero, Zero, Zero, Red)},
    {"r8/INTENSITY", MTLPixelFormatR8Unorm, 1, S(Red, Red, Red, Red)},
    {"r8/LUMINANCE", MTLPixelFormatR8Unorm, 1, S(Red, Red, Red, One)},
};

static uint8_t filled(int i, int c) { return (uint8_t)(0x10 + 0x40 * c + i); }
static uint8_t written(int i, int c) { return (uint8_t)(0x11 + 0x40 * c + i); }

// The byte a read's lane `l` answers by the rule, of texel i of a filled texture of `v`.
static int predictedRead(const View *v, int i, int l)
{
	uint8_t s = ((const uint8_t *)&v->swizzle)[l];
	switch (s) {
	case MTLTextureSwizzleZero:
		return 0;
	case MTLTextureSwizzleOne:
		return 255;
	default: {
		int c = s - MTLTextureSwizzleRed;
		if (c < v->channels)
			return filled(i, c);
		return c == 3 ? 255 : 0;
	}
	}
}

// The byte channel `c` of texel i holds by the rule after a write of `written` through `v`.
static int predictedWrite(const View *v, int i, int c)
{
	const uint8_t *s = (const uint8_t *)&v->swizzle;
	for (int l = 0; l < 4; l++)
		if (s[l] == MTLTextureSwizzleRed + c)
			return written(i, l);
	return written(i, c);
}

static id<MTLTexture> parent(id<MTLDevice> d, const View *v, BOOL fill)
{
	MTLTextureDescriptor *t = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:v->format
	                                                                             width:kW
	                                                                            height:kH
	                                                                         mipmapped:NO];
	t.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsagePixelFormatView;
	t.storageMode = MTLStorageModeManaged;
	id<MTLTexture> x = [d newTextureWithDescriptor:t];
	uint8_t bytes[kN * 4];
	for (int i = 0; i < kN; i++)
		for (int c = 0; c < v->channels; c++)
			bytes[i * v->channels + c] = fill ? filled(i, c) : 0xee;
	[x replaceRegion:MTLRegionMake2D(0, 0, kW, kH) mipmapLevel:0 withBytes:bytes bytesPerRow:kW * v->channels];
	return x;
}

static id<MTLTexture> swizzled(id<MTLTexture> p, const View *v)
{
	return [p newTextureViewWithPixelFormat:v->format
	                            textureType:MTLTextureType2D
	                                 levels:NSMakeRange(0, 1)
	                                 slices:NSMakeRange(0, 1)
	                                swizzle:v->swizzle];
}

static void commit(id<MTLCommandBuffer> cb, id<MTLTexture> sync)
{
	if (sync) {
		id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
		[b synchronizeResource:sync];
		[b endEncoding];
	}
	[cb commit];
	[cb waitUntilCompleted];
}

// Prints the parent's bytes after a write, and whether they are the rule's.
static void printWrite(const char *view, const char *kernel, const View *v, id<MTLTexture> p, id<MTLCommandBuffer> cb)
{
	uint8_t got[kN * 4];
	[p getBytes:got bytesPerRow:kW * v->channels fromRegion:MTLRegionMake2D(0, 0, kW, kH) mipmapLevel:0];
	int bad = 0;
	printf("view=%s kernel=%s status=%ld error=%s bytes=", view, kernel, (long)cb.status, oneLine(cb.error));
	for (int i = 0; i < kN; i++) {
		for (int c = 0; c < v->channels; c++) {
			printf("%02x", got[i * v->channels + c]);
			bad += got[i * v->channels + c] != predictedWrite(v, i, c);
		}
		printf(i + 1 < kN ? "." : "");
	}
	printf(" rule=%s\n", bad ? "differs" : "ok");
}

// Prints `count` float4 reads at `stride` (in float4s) from `o`, and whether they are the rule's.
static void printReads(const char *view, const char *kernel, const View *v, const float *o, int stride, int first,
                       id<MTLCommandBuffer> cb)
{
	int bad = 0;
	printf("view=%s kernel=%s status=%ld error=%s lanes=", view, kernel, (long)cb.status, oneLine(cb.error));
	for (int i = 0; i < kN; i++) {
		const float *f = o + 4 * (stride * i + first);
		for (int l = 0; l < 4; l++) {
			int b = (int)lrintf(f[l] * 255.0f);
			printf("%02x", b & 0xff);
			bad += b != predictedRead(v, i, l);
		}
		printf(i + 1 < kN ? "." : "");
	}
	uint32_t w[4];
	memcpy(w, o + 4 * first, sizeof w);
	printf(" texel0_words=%08x,%08x,%08x,%08x rule=%s\n", w[0], w[1], w[2], w[3], bad ? "differs" : "ok");
}

static id<MTLComputePipelineState> mslPipeline(id<MTLDevice> d, id<MTLLibrary> lib, NSString *name)
{
	NSError *e = nil;
	id<MTLComputePipelineState> p = [d newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&e];
	if (p == nil)
		printf("msl %s pipeline=no error=%s\n", name.UTF8String, oneLine(e));
	return p;
}

static MTLCompileOptions *clOptions(const char *std)
{
	MTLCompileOptions *o = [MTLCompileOptions new];
	SEL language = sel_registerName("setSourceLanguage:");
	SEL arguments = sel_registerName("setAdditionalCompilerArguments:");
	if (![o respondsToSelector:language] || ![o respondsToSelector:arguments])
		return nil;
	((void (*)(id, SEL, NSUInteger))objc_msgSend)(o, sel_registerName("setLanguageVersion:"), 0x20002);
	((void (*)(id, SEL, NSInteger))objc_msgSend)(o, language, 1);
	((void (*)(id, SEL, id))objc_msgSend)(
	    o, arguments,
	    [NSString stringWithFormat:@"-cl-single-precision-constant -fapplecl-lax-vector-conversions "
	                               @"-fapplecl-functional-casts -fapplecl-inline-definitions -cl-std=%s",
	                               std]);
	return o;
}

// A pipeline of OpenCL mode of kernel `name` of `lib`, with the reflection's texture and buffer
// indices of its arguments.
static id<MTLComputePipelineState> clPipeline(id<MTLDevice> d, id<MTLLibrary> lib, NSString *name, NSUInteger *texture,
                                             NSUInteger *buffer)
{
	id<MTLFunction> fn = [lib newFunctionWithName:[@"__entry_" stringByAppendingString:name]];
	uint8_t pd[32] = {2};
	SEL plugin = sel_registerName("newFunctionWithPluginData:bitcodeType:");
	id variant = fn && [(id)fn respondsToSelector:plugin]
	                 ? ((id(*)(id, SEL, NSData *, NSUInteger))objc_msgSend)(fn, plugin, [NSData dataWithBytes:pd length:32], 2)
	                 : nil;
	if (variant == nil) {
		printf("cl %s variant=nil\n", name.UTF8String);
		return nil;
	}
	MTLComputePipelineDescriptor *desc = [MTLComputePipelineDescriptor new];
	desc.computeFunction = variant;
	((void (*)(id, SEL, BOOL))objc_msgSend)(desc, sel_registerName("setOpenCLModeEnabled:"), YES);
	NSError *error = nil;
	MTLComputePipelineReflection *r = nil;
	id<MTLComputePipelineState> p = [d newComputePipelineStateWithDescriptor:desc
	                                                                 options:MTLPipelineOptionArgumentInfo
	                                                              reflection:&r
	                                                                   error:&error];
	if (p == nil) {
		printf("cl %s pipeline=no error=%s\n", name.UTF8String, oneLine(error));
		return nil;
	}
	*texture = *buffer = NSNotFound;
	for (MTLArgument *a in r.arguments) {
		if (a.type == MTLArgumentTypeTexture)
			*texture = a.index;
		else if (a.type == MTLArgumentTypeBuffer && a.index >= 3)
			*buffer = a.index;
	}
	printf("cl %s pipeline=yes texture=%ld buffer=%ld\n", name.UTF8String, (long)*texture, (long)*buffer);
	return p;
}

static void bindHidden(id<MTLComputeCommandEncoder> e, id<MTLBuffer> printfBuf)
{
	uint32_t hidden0[4] = {0, 0, 0, 2};
	uint32_t table[64];
	for (uint32_t i = 0; i < 64; i++)
		table[i] = i * 0x10000 + 0x1234;
	[e setBytes:hidden0 length:sizeof hidden0 atIndex:0];
	[e setBuffer:printfBuf offset:0 atIndex:1];
	[e setBytes:table length:sizeof table atIndex:2];
}

static int viewsPart(id<MTLDevice> d)
{
	NSError *error = nil;
	id<MTLLibrary> msl = [d newLibraryWithSource:kMsl options:nil error:&error];
	printf("msl library=%s error=%s\n", msl ? "yes" : "no", oneLine(error));
	id<MTLLibrary> cl = [d newLibraryWithSource:@(kClSource) options:clOptions("CL1.2") error:&error];
	printf("cl library=%s error=%s\n", cl ? "yes" : "no", oneLine(error));
	id<MTLLibrary> cl2 = [d newLibraryWithSource:@(kClRwSource) options:clOptions("CL2.0") error:&error];
	printf("cl2 library=%s error=%s\n", cl2 ? "yes" : "no", oneLine(error));
	if (msl == nil || cl == nil)
		return 1;
	id<MTLCommandQueue> q = [d newCommandQueue];
	id<MTLBuffer> printfBuf = [d newBufferWithLength:4096 options:MTLResourceStorageModeShared];
	id<MTLBuffer> out = [d newBufferWithLength:16 * 2 * kN options:MTLResourceStorageModeShared];
	NSDictionary<NSString *, id<MTLComputePipelineState>> *m = @{
		@"mw" : mslPipeline(d, msl, @"mw"),
		@"mrww" : mslPipeline(d, msl, @"mrww"),
		@"mrwr" : mslPipeline(d, msl, @"mrwr"),
		@"mr" : mslPipeline(d, msl, @"mr"),
		@"ms" : mslPipeline(d, msl, @"ms"),
		@"mq" : mslPipeline(d, msl, @"mq"),
	};
	NSUInteger tw, bw, tq, bq, tr, br, trw = NSNotFound, brw = NSNotFound;
	id<MTLComputePipelineState> cw = clPipeline(d, cl, @"cw", &tw, &bw);
	id<MTLComputePipelineState> cq = clPipeline(d, cl, @"cq", &tq, &bq);
	id<MTLComputePipelineState> cr = clPipeline(d, cl, @"cr", &tr, &br);
	id<MTLComputePipelineState> crw = cl2 ? clPipeline(d, cl2, @"crw", &trw, &brw) : nil;
	for (size_t vi = 0; vi < sizeof kViews / sizeof kViews[0]; vi++) {
		const View *v = &kViews[vi];
		const char *vn = v->name;
		// Writes: the parent filled with 0xee, a write through the view.
		struct {
			const char *name;
			id<MTLComputePipelineState> p;
			NSUInteger texture;
		} writes[3] = {{"mw", m[@"mw"], 0}, {"mrww", m[@"mrww"], 0}, {"cw", cw, tw}};
		for (int w = 0; w < 3; w++) {
			if (writes[w].p == nil)
				continue;
			id<MTLTexture> p = parent(d, v, NO), view = swizzled(p, v);
			id<MTLCommandBuffer> cb = [q commandBuffer];
			id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
			[e setComputePipelineState:writes[w].p];
			if (writes[w].p == cw)
				bindHidden(e, printfBuf);
			[e setTexture:view atIndex:writes[w].texture];
			[e dispatchThreads:MTLSizeMake(kW, kH, 1) threadsPerThreadgroup:MTLSizeMake(kW, kH, 1)];
			[e endEncoding];
			commit(cb, p);
			printWrite(vn, writes[w].name, v, p, cb);
			if (w == 0)
				printf("view=%s view_usage=%#lx view_swizzle=%u,%u,%u,%u\n", vn, (unsigned long)view.usage,
				       view.swizzle.red, view.swizzle.green, view.swizzle.blue, view.swizzle.alpha);
		}
		// Reads: the parent filled, a read through the view.
		struct {
			const char *name;
			id<MTLComputePipelineState> p;
			NSUInteger texture, buffer;
			int stride;
		} reads[4] = {{"mrwr", m[@"mrwr"], 0, 0, 1},
		              {"mr", m[@"mr"], 0, 0, 1},
		              {"ms", m[@"ms"], 0, 0, 1},
		              {"cr", cr, tr, br, 2}};
		for (int r = 0; r < 4; r++) {
			if (reads[r].p == nil)
				continue;
			id<MTLTexture> p = parent(d, v, YES), view = swizzled(p, v);
			memset(out.contents, 0xcd, out.length);
			id<MTLCommandBuffer> cb = [q commandBuffer];
			id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
			[e setComputePipelineState:reads[r].p];
			if (reads[r].p == cr)
				bindHidden(e, printfBuf);
			[e setTexture:view atIndex:reads[r].texture];
			[e setBuffer:out offset:0 atIndex:reads[r].buffer];
			[e dispatchThreads:MTLSizeMake(kW, kH, 1) threadsPerThreadgroup:MTLSizeMake(kW, kH, 1)];
			[e endEncoding];
			commit(cb, nil);
			if (reads[r].p == cr) {
				printReads(vn, "cr(nosampler)", v, out.contents, 2, 0, cb);
				printReads(vn, "cr(sampler)", v, out.contents, 2, 1, cb);
			} else {
				printReads(vn, reads[r].name, v, out.contents, 1, 0, cb);
			}
		}
		// Queries of a write-only binding.
		struct {
			const char *name;
			id<MTLComputePipelineState> p;
			NSUInteger texture, buffer;
		} queries[2] = {{"mq", m[@"mq"], 0, 0}, {"cq", cq, tq, bq}};
		for (int k = 0; k < 2; k++) {
			if (queries[k].p == nil)
				continue;
			id<MTLTexture> p = parent(d, v, YES), view = swizzled(p, v);
			memset(out.contents, 0xcd, out.length);
			id<MTLCommandBuffer> cb = [q commandBuffer];
			id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
			[e setComputePipelineState:queries[k].p];
			if (queries[k].p == cq)
				bindHidden(e, printfBuf);
			[e setTexture:view atIndex:queries[k].texture];
			[e setBuffer:out offset:0 atIndex:queries[k].buffer];
			[e dispatchThreads:MTLSizeMake(kW, kH, 1) threadsPerThreadgroup:MTLSizeMake(kW, kH, 1)];
			[e endEncoding];
			commit(cb, nil);
			const uint32_t *o = out.contents;
			printf("view=%s kernel=%s status=%ld error=%s words=%#x,%#x,%#x,%#x\n", vn, queries[k].name,
			       (long)cb.status, oneLine(cb.error), o[0], o[1], o[2], o[3]);
		}
		// A read_write image of OpenCL C 2.0: read, then written.
		if (crw) {
			id<MTLTexture> p = parent(d, v, YES), view = swizzled(p, v);
			memset(out.contents, 0xcd, out.length);
			id<MTLCommandBuffer> cb = [q commandBuffer];
			id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
			[e setComputePipelineState:crw];
			bindHidden(e, printfBuf);
			[e setTexture:view atIndex:trw];
			[e setBuffer:out offset:0 atIndex:brw];
			[e dispatchThreads:MTLSizeMake(kW, kH, 1) threadsPerThreadgroup:MTLSizeMake(kW, kH, 1)];
			[e endEncoding];
			commit(cb, p);
			printReads(vn, "crw(read)", v, out.contents, 1, 0, cb);
			printWrite(vn, "crw(write)", v, p, cb);
		}
	}
	return 0;
}

int main(int argc, char **argv)
{
	@autoreleasepool {
		const char *want = argc > 1 ? argv[1] : NULL;
		id<MTLDevice> d = nil;
		for (id<MTLDevice> x in MTLCopyAllDevices())
			if (d == nil && (want == NULL || strcasestr(x.name.UTF8String, want)))
				d = x;
		if (d == nil) {
			printf("device=none\n");
			return 1;
		}
		printf("device=%s os=%s\n", d.name.UTF8String, NSProcessInfo.processInfo.operatingSystemVersionString.UTF8String);
		int views = viewsPart(d);
		int trace = tracePart(d, want);
		return views || trace;
	}
}
