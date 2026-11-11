// What Metal's reflection states for the image arguments of a compute pipeline of OpenCL mode. Our
// own OpenCL C kernels, one library each: an `image1d_t`, an `image1d_array_t`, an `image2d_t` and
// an `image3d_t` read with `read_imagef`, `read_imagei` and `read_imageui` (and `read_imageh` where
// the compiler takes `cl_khr_fp16`), written with the write forms, only queried, unused, read
// through a sampler, two images read as float and int, and a `__read_write` image (OpenCL C 2.0).
// Nothing is written to disk.
//
//   xcrun clang -fobjc-arc -framework Metal -framework Foundation \
//       -o opencl_image_reflection opencl_image_reflection.m
//   ./opencl_image_reflection [DEVICE]      (DEVICE: a substring of the Metal device's name)
//
// For each kernel: the library (OpenCL C source, `setSourceLanguage:` 1), the OpenCL variant of
// `__entry_<name>` (`newFunctionWithPluginData:bitcodeType:` 2), a pipeline with
// `openCLModeEnabled`, and its reflection both ways: `arguments` (`MTLArgument`: type, index,
// access, texture type, texture data type, depth, array length, active) and `bindings`
// (`MTLTextureBinding`: the same fields).
//
// WHAT IT MEASURED, Apple M3 Max, macOS 27.0.1:
//   - Every image argument's texture data type is 0 (`MTLDataTypeNone`), both ways: read as float,
//     int, uint and half (`rh1` compiles), written as float, int and uint, only queried (`q1`),
//     unused (`n1`: active 0, used 0), read through a sampler (`s1`), and each of two images read
//     as float and int (`mix1`).
//   - Texture types: `image1d_t` 2 (`MTLTextureType2D`), `image1d_array_t` 3 (`2DArray`),
//     `image2d_t` 2, `image3d_t` 7 (`3D`). Access: `__read_only` 0 (read-only), `__write_only` 2
//     (write-only). Depth 0, array length 1, at the image's own index.
//   - The `__read_write` image (`rw1`, OpenCL C 2.0): no library, "Undefined symbol(s) for
//     architecture 'air64'" for `read_imagef` and `write_imagef` of `ocl_image1d_rw`.
#define CL_SILENCE_DEPRECATION
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <objc/message.h>
#include <objc/runtime.h>

static const char *oneLine(NSError *e)
{
	if (e == nil)
		return "(none)";
	return [[e.description componentsSeparatedByCharactersInSet:NSCharacterSet.newlineCharacterSet]
	           componentsJoinedByString:@" "]
	    .UTF8String;
}

struct Kernel {
	const char *name;
	const char *std;
	const char *source;
};

static const struct Kernel kKernels[] = {
    {"rf1", "CL1.2", "__kernel void rf1(__read_only image1d_t a, __global float4 *o) { o[0] = read_imagef(a, 0); }"},
    {"ri1", "CL1.2", "__kernel void ri1(__read_only image1d_t a, __global int4 *o) { o[0] = read_imagei(a, 0); }"},
    {"ru1", "CL1.2", "__kernel void ru1(__read_only image1d_t a, __global uint4 *o) { o[0] = read_imageui(a, 0); }"},
    {"rh1", "CL1.2",
     "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
     "__kernel void rh1(__read_only image1d_t a, __global half *o) { half4 v = read_imageh(a, 0); o[0] = v.x; }"},
    {"wf1", "CL1.2", "__kernel void wf1(__write_only image1d_t a) { write_imagef(a, 0, (float4)(1.0f)); }"},
    {"wi1", "CL1.2", "__kernel void wi1(__write_only image1d_t a) { write_imagei(a, 0, (int4)(1)); }"},
    {"wu1", "CL1.2", "__kernel void wu1(__write_only image1d_t a) { write_imageui(a, 0, (uint4)(1)); }"},
    {"q1", "CL1.2", "__kernel void q1(__read_only image1d_t a, __global int *o) { o[0] = get_image_width(a); }"},
    {"n1", "CL1.2", "__kernel void n1(__read_only image1d_t a, __global int *o) { o[0] = 1; }"},
    {"s1", "CL1.2",
     "__kernel void s1(__read_only image1d_t a, sampler_t s, __global float4 *o) { o[0] = read_imagef(a, s, 0.5f); }"},
    {"mix1", "CL1.2",
     "__kernel void mix1(__read_only image1d_t a, __read_only image1d_t b, __global float4 *o, __global int4 *p) "
     "{ o[0] = read_imagef(a, 0); p[0] = read_imagei(b, 0); }"},
    {"rw1", "CL2.0",
     "__kernel void rw1(__read_write image1d_t a, __global float4 *o) "
     "{ o[0] = read_imagef(a, 0); write_imagef(a, 0, (float4)(1.0f)); }"},
    {"rfa", "CL1.2",
     "__kernel void rfa(__read_only image1d_array_t a, __global float4 *o) { o[0] = read_imagef(a, (int2)(0, 0)); }"},
    {"ria", "CL1.2",
     "__kernel void ria(__read_only image1d_array_t a, __global int4 *o) { o[0] = read_imagei(a, (int2)(0, 0)); }"},
    {"rua", "CL1.2",
     "__kernel void rua(__read_only image1d_array_t a, __global uint4 *o) { o[0] = read_imageui(a, (int2)(0, 0)); }"},
    {"wia", "CL1.2",
     "__kernel void wia(__write_only image1d_array_t a) { write_imagei(a, (int2)(0, 0), (int4)(1)); }"},
    {"rf2", "CL1.2",
     "__kernel void rf2(__read_only image2d_t a, __global float4 *o) { o[0] = read_imagef(a, (int2)(0, 0)); }"},
    {"ri2", "CL1.2",
     "__kernel void ri2(__read_only image2d_t a, __global int4 *o) { o[0] = read_imagei(a, (int2)(0, 0)); }"},
    {"ru2", "CL1.2",
     "__kernel void ru2(__read_only image2d_t a, __global uint4 *o) { o[0] = read_imageui(a, (int2)(0, 0)); }"},
    {"wu2", "CL1.2",
     "__kernel void wu2(__write_only image2d_t a) { write_imageui(a, (int2)(0, 0), (uint4)(1)); }"},
    {"ri3", "CL1.2",
     "__kernel void ri3(__read_only image3d_t a, __global int4 *o) { o[0] = read_imagei(a, (int4)(0)); }"},
};

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

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
static void printArguments(const char *name, MTLComputePipelineReflection *r)
{
	for (MTLArgument *a in r.arguments) {
		printf("argument kernel=%s name=%s type=%lu index=%lu access=%lu active=%d", name, a.name.UTF8String,
		       (unsigned long)a.type, (unsigned long)a.index, (unsigned long)a.access, a.isActive);
		if (a.type == MTLArgumentTypeTexture)
			printf(" texture_type=%lu texture_data_type=%lu depth=%d array_length=%lu", (unsigned long)a.textureType,
			       (unsigned long)a.textureDataType, a.isDepthTexture, (unsigned long)a.arrayLength);
		printf("\n");
	}
}
#pragma clang diagnostic pop

static void printBindings(const char *name, MTLComputePipelineReflection *r)
{
	for (id<MTLBinding> b in r.bindings) {
		printf("binding kernel=%s name=%s type=%ld index=%lu access=%lu used=%d", name, b.name.UTF8String,
		       (long)b.type, (unsigned long)b.index, (unsigned long)b.access, b.used);
		if ([(id)b conformsToProtocol:@protocol(MTLTextureBinding)]) {
			id<MTLTextureBinding> t = (id<MTLTextureBinding>)b;
			printf(" texture_type=%lu texture_data_type=%lu depth=%d array_length=%lu", (unsigned long)t.textureType,
			       (unsigned long)t.textureDataType, t.depthTexture, (unsigned long)t.arrayLength);
		}
		printf("\n");
	}
}

static void probe(id<MTLDevice> d, const struct Kernel *k)
{
	NSError *error = nil;
	MTLCompileOptions *options = clOptions(k->std);
	if (options == nil) {
		printf("kernel=%s options=unavailable\n", k->name);
		return;
	}
	id<MTLLibrary> lib = [d newLibraryWithSource:@(k->source) options:options error:&error];
	if (lib == nil) {
		printf("kernel=%s library=no error=%s\n", k->name, oneLine(error));
		return;
	}
	NSString *entry = [@"__entry_" stringByAppendingString:@(k->name)];
	id<MTLFunction> fn = [lib newFunctionWithName:entry];
	uint8_t pd[32] = {2};
	SEL plugin = sel_registerName("newFunctionWithPluginData:bitcodeType:");
	id variant = fn && [(id)fn respondsToSelector:plugin]
	                 ? ((id(*)(id, SEL, NSData *, NSUInteger))objc_msgSend)(fn, plugin, [NSData dataWithBytes:pd length:32], 2)
	                 : nil;
	if (variant == nil) {
		printf("kernel=%s function=%s variant=nil\n", k->name, fn ? "yes" : "no");
		return;
	}
	MTLComputePipelineDescriptor *desc = [MTLComputePipelineDescriptor new];
	desc.computeFunction = variant;
	((void (*)(id, SEL, BOOL))objc_msgSend)(desc, sel_registerName("setOpenCLModeEnabled:"), YES);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
	MTLPipelineOption both = MTLPipelineOptionArgumentInfo | MTLPipelineOptionBindingInfo;
#pragma clang diagnostic pop
	MTLComputePipelineReflection *r = nil;
	id<MTLComputePipelineState> p = [d newComputePipelineStateWithDescriptor:desc
	                                                                 options:both
	                                                              reflection:&r
	                                                                   error:&error];
	printf("kernel=%s pipeline=%s error=%s\n", k->name, p ? "yes" : "no", p ? "(none)" : oneLine(error));
	if (p != nil) {
		printArguments(k->name, r);
		printBindings(k->name, r);
	}
}

int main(int argc, char **argv)
{
	@autoreleasepool {
		id<MTLDevice> d = nil;
		for (id<MTLDevice> each in MTLCopyAllDevices()) {
			if (argc < 2 || strstr(each.name.UTF8String, argv[1]) != NULL) {
				d = each;
				break;
			}
		}
		if (d == nil) {
			printf("no device\n");
			return 1;
		}
		NSOperatingSystemVersion v = NSProcessInfo.processInfo.operatingSystemVersion;
		printf("device=%s macos=%ld.%ld.%ld\n", d.name.UTF8String, (long)v.majorVersion, (long)v.minorVersion,
		       (long)v.patchVersion);
		for (size_t i = 0; i < sizeof kKernels / sizeof kKernels[0]; i++)
			probe(d, &kKernels[i]);
	}
	return 0;
}
