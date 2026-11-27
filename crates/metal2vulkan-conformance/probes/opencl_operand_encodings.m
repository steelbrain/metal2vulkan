// Which encoding the enum and flag operands of OpenCL C reach AIR in: our own OpenCL C kernels
// using barrier, mem_fence, read_mem_fence and write_mem_fence of each flag, the atomic functions
// on __global and __local memory, constant samplers of address modes, filters and coordinates
// whose OpenCL bits differ from Metal's sampler words, a sampler argument, a read without a
// sampler and a write. Each kernel is its own program, compiled two ways, the bytes of each
// written for opencl_operand_encodings.sh to disassemble:
//
//   - OpenCL's program-binary route: OpenCL.framework's clBuildProgram on the GPU device and
//     CL_PROGRAM_BINARIES, written to cl-<kernel>.bin;
//   - Metal's OpenCL mode: newLibraryWithSource with setSourceLanguage: 1 (the options of
//     opencl_image_reflection.m), and every getter of the library and of `__entry_<kernel>` whose
//     selector names bitcode or library data and that answers bytes, written to
//     mtl-<kernel>-<library|entry>-<selector>.bin. The getters each class has are printed.
//
//   xcrun clang -fobjc-arc -framework Metal -framework Foundation -framework OpenCL \
//       -o probe opencl_operand_encodings.m
//   ./probe            (in the directory the .bin files go to)
//
// WHAT IT MEASURED, Apple M3 Max, macOS 27.0.1 (job m3-max-76799; metalfe-32023.921.5, AIR 2.9,
// language "Metal" 1.1):
//   - OpenCL.framework's program binary is a property list of the options and the source: no AIR.
//     `-[_MTLFunctionInternal bitcodeData]` of `__entry_<kernel>` answers the AIR (a wrapped
//     module), as does the library's `libraryDataContents`.
//   - barrier(flags): each call site passes OpenCL's flags, `_Z7barrierj(i32 1)` for
//     CLK_LOCAL_MEM_FENCE, 2 for CLK_GLOBAL_MEM_FENCE, 3 for both, and the wrapper passes them on
//     unconverted: `air.wg.barrier(i32 %0, i32 5, i32 1)`.
//   - mem_fence, read_mem_fence and write_mem_fence(flags): each wrapper converts the flags to
//     Metal's mask (`trunc` to i2, `llvm.bitreverse.i2`, `zext`) and calls
//     `air.atomic.fence(flags, order, 1)`, order 4 (acq_rel), 2 (acquire) and 3 (release).
//   - The 1.2 atomics: `air.atomic.{global,local}.<op>` in the AIR 2.9 form, order 0, scope 2 on
//     __global and 1 on __local, mem_flags 0, volatile true; atomic_cmpxchg orders 0 and 0.
//   - Constant samplers: `air.sampler_states` globals of two words in Metal's encoding, never
//     OpenCL's bits: CLAMP_TO_EDGE, unnormalized, nearest 0x7bff0000088049; REPEAT, normalized,
//     linear 0x7bff0000080a92; CLAMP, nearest 0x7bff0000080000 (clamp_to_zero); MIRRORED_REPEAT,
//     linear 0x7bff0000080adb; NONE, unnormalized, linear 0x7bff0000088a00; second word 0. Each
//     has compare 8 (never) and the LOD clamp 0 to 65504.
//   - Images: a sample through a sampler has access 0, a read without one (through
//     `air.get_read_sampler`) access 1, a write access 2.
#define CL_SILENCE_DEPRECATION
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <OpenCL/opencl.h>
#include <objc/message.h>
#include <objc/runtime.h>
#include <stdio.h>
#include <string.h>

struct Kernel {
	const char *name;
	const char *source;
};

#define SAMPLERS                                                                                       \
	"__constant sampler_t s_edge = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;\n" \
	"__constant sampler_t s_repeat = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR;\n"

static const struct Kernel kKernels[] = {
    {"bar_l", "__kernel void bar_l(__global int *g, __local int *l) { int i = get_global_id(0); l[i] = i;"
              " barrier(CLK_LOCAL_MEM_FENCE); g[i] = l[i ^ 1]; }\n"},
    {"bar_g", "__kernel void bar_g(__global int *g) { int i = get_global_id(0); g[i] = i;"
              " barrier(CLK_GLOBAL_MEM_FENCE); g[i] += g[i ^ 1]; }\n"},
    {"bar_lg", "__kernel void bar_lg(__global int *g, __local int *l) { int i = get_global_id(0); l[i] = i;"
               " g[i] = i; barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); g[i] += l[i ^ 1] + g[i ^ 1]; }\n"},
    {"mf_l", "__kernel void mf_l(__global int *g, __local int *l) { int i = get_global_id(0); l[i] = i;"
             " mem_fence(CLK_LOCAL_MEM_FENCE); g[i] = l[i]; }\n"},
    {"mf_g", "__kernel void mf_g(__global int *g) { int i = get_global_id(0); g[i] = i;"
             " mem_fence(CLK_GLOBAL_MEM_FENCE); g[i + 1] = 2; }\n"},
    {"rmf_l", "__kernel void rmf_l(__global int *g, __local int *l) { int i = get_global_id(0); l[i] = i;"
              " read_mem_fence(CLK_LOCAL_MEM_FENCE); g[i] = l[i]; }\n"},
    {"rmf_g", "__kernel void rmf_g(__global int *g) { int i = get_global_id(0); int a = g[i];"
              " read_mem_fence(CLK_GLOBAL_MEM_FENCE); g[i] = a + g[i + 1]; }\n"},
    {"wmf_l", "__kernel void wmf_l(__global int *g, __local int *l) { int i = get_global_id(0); l[i] = i;"
              " write_mem_fence(CLK_LOCAL_MEM_FENCE); l[i + 1] = 3; g[i] = l[i]; }\n"},
    {"wmf_g", "__kernel void wmf_g(__global int *g) { int i = get_global_id(0); g[i] = i;"
              " write_mem_fence(CLK_GLOBAL_MEM_FENCE); g[i + 1] = 4; }\n"},
    {"atomics", "__kernel void atomics(__global int *g, __global uint *u, __global float *f, __local int *l) {"
                " int i = get_global_id(0); atomic_add(g, i); atomic_cmpxchg(g + 1, 0, i); atomic_min(g + 2, i);"
                " atom_inc(u); atomic_xchg(f, 1.0f); atomic_add(&l[0], i); atomic_inc(&l[1]);"
                " atomic_max(&l[2], i); barrier(CLK_LOCAL_MEM_FENCE); g[3] = l[0] + l[1] + l[2]; }\n"},
    {"samplers", SAMPLERS "__kernel void samplers(__read_only image2d_t img, sampler_t arg, __global float4 *o) {\n"
                          "  const sampler_t s_clamp = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n"
                          "  const sampler_t s_mirror = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_MIRRORED_REPEAT"
                          " | CLK_FILTER_LINEAR;\n"
                          "  const sampler_t s_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_LINEAR;\n"
                          "  o[0] = read_imagef(img, s_edge, (int2)(1, 0));\n"
                          "  o[1] = read_imagef(img, s_repeat, (float2)(1.25f, 0.5f));\n"
                          "  o[2] = read_imagef(img, s_clamp, (float2)(-0.25f, 0.5f));\n"
                          "  o[3] = read_imagef(img, s_mirror, (float2)(1.75f, 0.5f));\n"
                          "  o[4] = read_imagef(img, s_none, (float2)(1.5f, 0.5f));\n"
                          "  o[5] = read_imagef(img, arg, (float2)(0.5f, 0.5f));\n"
                          "  o[6] = read_imagef(img, (int2)(1, 1));\n"
                          "}\n"},
    {"one_sampler", "__constant sampler_t s_edge = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE"
                    " | CLK_FILTER_NEAREST;\n"
                    "__kernel void one_sampler(__read_only image2d_t img, __global float4 *o) {"
                    " o[0] = read_imagef(img, s_edge, (int2)(1, 0)); }\n"},
    {"writes", "__kernel void writes(__write_only image2d_t img) { write_imagef(img, (int2)(0, 0), (float4)(1.0f)); }\n"},
};

static void writeBytes(const char *path, const void *bytes, size_t n)
{
	FILE *f = fopen(path, "wb");
	if (f == NULL || fwrite(bytes, 1, n, f) != n) {
		printf("write %s failed\n", path);
		if (f)
			fclose(f);
		return;
	}
	fclose(f);
	const unsigned char *b = bytes;
	printf("wrote %s bytes=%zu head=", path, n);
	for (size_t i = 0; i < n && i < 8; i++)
		printf("%02x", b[i]);
	printf("\n");
}

static void openclRoute(cl_context context, cl_device_id device, const struct Kernel *k)
{
	cl_int err = CL_SUCCESS;
	const char *source = k->source;
	cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, &err);
	if (program == NULL) {
		printf("cl kernel=%s no program err=%d\n", k->name, err);
		return;
	}
	err = clBuildProgram(program, 1, &device, "-cl-std=CL1.2", NULL, NULL);
	char log[4096] = {0};
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof log - 1, log, NULL);
	size_t size = 0;
	clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof size, &size, NULL);
	printf("cl kernel=%s build=%d binary=%zu log=%s\n", k->name, err, size, log);
	if (err == CL_SUCCESS && size > 0) {
		unsigned char *binary = malloc(size);
		unsigned char *binaries[1] = {binary};
		if (clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof binaries, binaries, NULL) == CL_SUCCESS) {
			char path[128];
			snprintf(path, sizeof path, "cl-%s.bin", k->name);
			writeBytes(path, binary, size);
		}
		free(binary);
	}
	clReleaseProgram(program);
}

// Each getter of `obj`'s class chain that returns an object and whose selector names bitcode or
// library data: printed, and where it answers bytes, written.
static void getters(const char *kernel, const char *tag, id obj)
{
	for (Class c = object_getClass(obj); c != Nil && c != [NSObject class]; c = class_getSuperclass(c)) {
		unsigned n = 0;
		Method *methods = class_copyMethodList(c, &n);
		for (unsigned i = 0; i < n; i++) {
			SEL sel = method_getName(methods[i]);
			const char *s = sel_getName(sel);
			if (!strcasestr(s, "bitcode") && !strcasestr(s, "librarydata") && !strcasestr(s, "airdata"))
				continue;
			char ret[64] = {0};
			method_getReturnType(methods[i], ret, sizeof ret);
			printf("getter kernel=%s %s class=%s sel=%s ret=%s\n", kernel, tag, class_getName(c), s, ret);
			if (strchr(s, ':') != NULL || ret[0] != '@' || strncmp(s, "new", 3) == 0)
				continue;
			id value = ((id(*)(id, SEL))objc_msgSend)(obj, sel);
			printf("value kernel=%s %s sel=%s class=%s\n", kernel, tag, s,
			       value ? class_getName(object_getClass(value)) : "nil");
			if ([value isKindOfClass:[NSData class]]) {
				NSData *d = value;
				char path[256];
				snprintf(path, sizeof path, "mtl-%s-%s-%s.bin", kernel, tag, s);
				writeBytes(path, d.bytes, d.length);
			}
		}
		free(methods);
	}
}

static void metalRoute(id<MTLDevice> device, MTLCompileOptions *options, const struct Kernel *k)
{
	NSError *error = nil;
	id<MTLLibrary> lib = [device newLibraryWithSource:@(k->source) options:options error:&error];
	if (lib == nil) {
		printf("metal kernel=%s library=no error=%s\n", k->name,
		       [[error.description componentsSeparatedByCharactersInSet:NSCharacterSet.newlineCharacterSet]
		           componentsJoinedByString:@" "]
		           .UTF8String);
		return;
	}
	printf("metal kernel=%s library=%s functions=%s\n", k->name, class_getName(object_getClass(lib)),
	       [lib.functionNames componentsJoinedByString:@","].UTF8String);
	getters(k->name, "library", lib);
	id<MTLFunction> fn = [lib newFunctionWithName:[@"__entry_" stringByAppendingString:@(k->name)]];
	printf("metal kernel=%s entry=%s\n", k->name, fn ? class_getName(object_getClass(fn)) : "nil");
	if (fn != nil)
		getters(k->name, "entry", fn);
}

int main(void)
{
	@autoreleasepool {
		cl_platform_id platform;
		cl_device_id device;
		cl_context context = NULL;
		cl_int err = clGetPlatformIDs(1, &platform, NULL);
		if (err == CL_SUCCESS)
			err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
		if (err == CL_SUCCESS)
			context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
		char name[256] = {0};
		if (context)
			clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof name - 1, name, NULL);
		printf("cl device=%s err=%d\n", name, err);

		id<MTLDevice> mtl = MTLCreateSystemDefaultDevice();
		MTLCompileOptions *o = [MTLCompileOptions new];
		SEL language = sel_registerName("setSourceLanguage:");
		SEL arguments = sel_registerName("setAdditionalCompilerArguments:");
		BOOL metal = [o respondsToSelector:language] && [o respondsToSelector:arguments];
		if (metal) {
			((void (*)(id, SEL, NSUInteger))objc_msgSend)(o, sel_registerName("setLanguageVersion:"), 0x20002);
			((void (*)(id, SEL, NSInteger))objc_msgSend)(o, language, 1);
			((void (*)(id, SEL, id))objc_msgSend)(
			    o, arguments,
			    @"-cl-single-precision-constant -fapplecl-lax-vector-conversions -fapplecl-functional-casts "
			    @"-fapplecl-inline-definitions -cl-std=CL1.2");
		}
		printf("metal device=%s options=%s\n", mtl.name.UTF8String, metal ? "yes" : "unavailable");

		for (size_t i = 0; i < sizeof kKernels / sizeof kKernels[0]; i++) {
			if (context)
				openclRoute(context, device, &kKernels[i]);
			if (metal)
				metalRoute(mtl, o, &kKernels[i]);
		}
		if (context)
			clReleaseContext(context);
	}
	return 0;
}
