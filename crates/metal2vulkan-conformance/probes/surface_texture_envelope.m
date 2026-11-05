// Which bytes of an IOSurface a texture over one of its planes reads and writes, for the textures
// Metal makes there beyond a 2D texture of one level whose rows lie inside the plane: rows wider
// than the plane's `bytesPerRow`, more rows than the plane has, more than one mip level, a type
// other than 2D (1D, 2D array, 3D, cube, texture buffer, 2D multisample), Private storage, and
// block-compressed formats (BC1, BC7) over a surface's rows.
//
// The reims-vgpu oracle (`oracle_surftex`, its Q20) records only that Metal makes these on an AMD
// 5300M and an Intel UHD 630 under macOS 26.7.1, and all but the cube and the BC ones on an Apple
// M3 Max under macOS 27.0.1. This measures what they read and write, one shape a process:
//
//   1. the surface's every 4-byte word holds a tag naming its own offset (`tag_byte`, marker
//      0xC1), written by the CPU under `IOSurfaceLock`, before the texture is made;
//   2. a blit (`copyFromTexture:toBuffer:`) of each level and slice, a kernel's `read()` of each,
//      and `getBytes`, each say for every row which surface bytes it read: a run of rows from
//      offset B, P bytes apart; zeros; or other bytes;
//   3. a blit into the texture (`copyFromBuffer:toTexture:`) and `replaceRegion` each write a
//      second tag (marker 0xD7) naming the texture's own byte (its levels and slices one after
//      another, rows tight), after the surface was tagged again; the surface's changed bytes then
//      say which texture byte each holds, and which changed bytes no texture byte names;
//   4. a multisample render target is read by a kernel, sample by sample, and stored by a render
//      pass that clears it, the surface's changed bytes listed.
//
// Before running a shape the probe counts where its rows end if every level, slice and face lay
// at the plane's `bytesPerRow` one after another from the plane's offset, against the surface's
// `IOSurfaceGetAllocSize`. A shape that would reach past it runs only under `--hazard`: such a read
// or write can fault the GPU.
//
//   surface_texture_envelope DEVICE                every shape that stays in its surface, each in
//                                                  a child process of its own
//   surface_texture_envelope DEVICE --hazard       the same, and the shapes that reach past it
//   surface_texture_envelope DEVICE --shape N [--hazard]   shape N alone (what the children run)
//   surface_texture_envelope DEVICE --list         the shapes, one a line
//
// DEVICE is a substring of the device's name (`AMD`, `Intel`, `Apple`); the probe never calls
// `MTLCreateSystemDefaultDevice`, which switches a dual-GPU MacBook's display to its discrete GPU.
// The API validation layer stays off (`MTL_DEBUG_LAYER` unset): what a guest's process meets.
//
// clang -fobjc-arc -Wall -Wextra -framework Metal -framework Foundation -framework IOSurface \
//     -framework CoreVideo -o surface_texture_envelope surface_texture_envelope.m
// (add `-arch x86_64` for an Intel Mac).

#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

// The marker byte of a word of the surface's own tag, and of the tag a write puts in.
static const uint8_t kReadMarker = 0xC1;
static const uint8_t kWriteMarker = 0xD7;

// Byte `offset % 4` of the tag of word `offset / 4` under `marker`: the marker, then the word's
// index 6 bits a byte with the top bit set, so no tag byte is 0 and only a marker is 0xC0 or more.
static uint8_t tag_byte(uint8_t marker, uint64_t offset)
{
	uint64_t word = offset / 4;
	switch (offset % 4) {
	case 0:
		return marker;
	case 1:
		return (uint8_t)(0x80 | (word & 0x3f));
	case 2:
		return (uint8_t)(0x80 | ((word >> 6) & 0x3f));
	default:
		return (uint8_t)(0x80 | ((word >> 12) & 0x3f));
	}
}

static void fill_tag(uint8_t *bytes, uint64_t length, uint8_t marker, uint64_t first)
{
	for (uint64_t k = 0; k < length; k++)
		bytes[k] = tag_byte(marker, first + k);
}

// Where `bytes[0]` came from, if they hold a run of `marker`'s tag: the offset whose tag the first
// whole word names, less the bytes before it; -1 where no whole word of the tag is there.
static int64_t tag_origin(const uint8_t *bytes, uint64_t length, uint8_t marker)
{
	for (uint64_t j = 0; j < 4 && j + 3 < length; j++) {
		if (bytes[j] != marker)
			continue;
		uint64_t word = (uint64_t)(bytes[j + 1] & 0x3f) | (uint64_t)(bytes[j + 2] & 0x3f) << 6 |
		                (uint64_t)(bytes[j + 3] & 0x3f) << 12;
		return (int64_t)(word * 4) - (int64_t)j;
	}
	return -1;
}

// How many of `bytes` from the start hold `marker`'s tag of the offsets from `origin`.
static uint64_t tag_prefix(const uint8_t *bytes, uint64_t length, uint8_t marker, int64_t origin)
{
	if (origin < 0)
		return 0;
	uint64_t k = 0;
	while (k < length && bytes[k] == tag_byte(marker, (uint64_t)origin + k))
		k++;
	return k;
}

static bool all_zero(const uint8_t *bytes, uint64_t length)
{
	for (uint64_t k = 0; k < length; k++)
		if (bytes[k] != 0)
			return false;
	return true;
}

static void hex(const uint8_t *bytes, uint64_t length, char *out, size_t n)
{
	size_t k = 0;
	out[0] = 0;
	for (uint64_t i = 0; i < length && i < 8 && k + 3 < n; i++)
		k += (size_t)snprintf(out + k, n - k, "%02x", bytes[i]);
}

// One row's answer: its bytes from `origin` of the surface (`whole` of them, or `prefix` and then
// zeros or other bytes), zeros, or other bytes.
enum RowKind { kRowFrom, kRowZero, kRowOther };
struct Row {
	enum RowKind kind;
	int64_t origin;
	uint64_t prefix;
	bool rest_zero;
};

static struct Row row_answer(const uint8_t *bytes, uint64_t length)
{
	struct Row r = {kRowOther, -1, 0, false};
	int64_t origin = tag_origin(bytes, length, kReadMarker);
	if (origin >= 0) {
		r.prefix = tag_prefix(bytes, length, kReadMarker, origin);
		if (r.prefix > 0) {
			r.kind = kRowFrom;
			r.origin = origin;
			r.rest_zero = all_zero(bytes + r.prefix, length - r.prefix);
			return r;
		}
	}
	if (all_zero(bytes, length))
		r.kind = kRowZero;
	return r;
}

// Prints which surface bytes each of `rows` rows of `row_bytes` (one after another in `bytes`)
// came from, consecutive rows read whole at one pitch as one run.
static void report_rows(const char *what, const uint8_t *bytes, uint64_t row_bytes, uint64_t rows)
{
	uint64_t others = 0;
	uint64_t y = 0;
	while (y < rows) {
		struct Row r = row_answer(bytes + y * row_bytes, row_bytes);
		bool whole = r.kind == kRowFrom && r.prefix == row_bytes;
		uint64_t end = y + 1;
		if (whole) {
			int64_t pitch = 0;
			int64_t last = r.origin;
			while (end < rows) {
				struct Row n = row_answer(bytes + end * row_bytes, row_bytes);
				if (n.kind != kRowFrom || n.prefix != row_bytes)
					break;
				if (end == y + 1)
					pitch = n.origin - last;
				else if (n.origin - last != pitch)
					break;
				last = n.origin;
				end++;
			}
			printf("  %s: rows %llu-%llu read whole from surface byte %lld, %lld apart\n", what,
			       (unsigned long long)y, (unsigned long long)(end - 1), (long long)r.origin,
			       (long long)pitch);
		} else if (r.kind == kRowZero) {
			while (end < rows && row_answer(bytes + end * row_bytes, row_bytes).kind == kRowZero)
				end++;
			printf("  %s: rows %llu-%llu read zeros\n", what, (unsigned long long)y,
			       (unsigned long long)(end - 1));
		} else if (r.kind == kRowFrom) {
			printf("  %s: row %llu read %llu bytes from surface byte %lld, then %s\n", what,
			       (unsigned long long)y, (unsigned long long)r.prefix, (long long)r.origin,
			       r.rest_zero ? "zeros" : "other bytes");
		} else if (others++ < 6) {
			char h[40];
			hex(bytes + y * row_bytes, row_bytes, h, sizeof h);
			printf("  %s: row %llu read other bytes (%s...)\n", what, (unsigned long long)y, h);
		}
		y = end;
	}
	if (others > 6)
		printf("  %s: %llu rows of other bytes in all\n", what, (unsigned long long)others);
}

// One run of the surface's changed bytes: `length` bytes from `at` holding the write's tag of the
// texture's bytes from `from` (-1: zeros, -2: other bytes).
struct Run {
	uint64_t at;
	uint64_t length;
	int64_t from;
};

// Prints which of the surface's `length` bytes a write changed from the surface's own tag, and
// which texture byte each changed byte holds, runs of equal length at a constant stride on both
// sides as one line.
static void report_changes(const char *what, const uint8_t *surface, uint64_t length)
{
	NSMutableData *store = [NSMutableData data];
	uint64_t o = 0;
	while (o < length) {
		if (surface[o] == tag_byte(kReadMarker, o)) {
			o++;
			continue;
		}
		uint64_t end = o;
		while (end < length && surface[end] != tag_byte(kReadMarker, end))
			end++;
		// Split the changed bytes [o, end) into runs of one origin each.
		uint64_t at = o;
		while (at < end) {
			int64_t from = tag_origin(surface + at, end - at, kWriteMarker);
			uint64_t n = tag_prefix(surface + at, end - at, kWriteMarker, from);
			if (n == 0) {
				uint64_t k = at;
				bool zero = surface[k] == 0;
				while (k < end && (surface[k] == 0) == zero &&
				       tag_origin(surface + k, end - k, kWriteMarker) < 0)
					k++;
				if (k == at)
					k = at + 1;
				struct Run run = {at, k - at, zero ? -1 : -2};
				[store appendBytes:&run length:sizeof run];
				at = k;
				continue;
			}
			struct Run run = {at, n, from};
			[store appendBytes:&run length:sizeof run];
			at += n;
		}
		o = end;
	}
	const struct Run *runs = store.bytes;
	size_t count = store.length / sizeof(struct Run);
	uint64_t changed = 0;
	for (size_t i = 0; i < count; i++)
		changed += runs[i].length;
	printf("  %s: %llu surface bytes changed, in %zu runs\n", what, (unsigned long long)changed,
	       count);
	size_t i = 0;
	while (i < count) {
		const struct Run *r = &runs[i];
		size_t end = i + 1;
		if (r->from >= 0 && i + 1 < count && runs[i + 1].from >= 0 &&
		    runs[i + 1].length == r->length) {
			int64_t stride = (int64_t)runs[i + 1].at - (int64_t)r->at;
			int64_t texture_stride = runs[i + 1].from - r->from;
			while (end < count && runs[end].from >= 0 && runs[end].length == r->length &&
			       (int64_t)runs[end].at - (int64_t)runs[end - 1].at == stride &&
			       runs[end].from - runs[end - 1].from == texture_stride)
				end++;
			printf("  %s: %zu runs of %llu bytes from surface byte %llu, %lld apart, hold the "
			       "texture's bytes from %lld, %lld apart\n",
			       what, end - i, (unsigned long long)r->length, (unsigned long long)r->at,
			       (long long)stride, (long long)r->from, (long long)texture_stride);
		} else if (r->from >= 0) {
			printf("  %s: surface bytes %llu-%llu hold the texture's bytes from %lld\n", what,
			       (unsigned long long)r->at, (unsigned long long)(r->at + r->length - 1),
			       (long long)r->from);
		} else {
			char h[40];
			hex(surface + r->at, r->length, h, sizeof h);
			printf("  %s: surface bytes %llu-%llu changed to %s (%s...)\n", what,
			       (unsigned long long)r->at, (unsigned long long)(r->at + r->length - 1),
			       r->from == -1 ? "zeros" : "other bytes", h);
		}
		i = end;
	}
}

enum Surface { kBGRA, kBGRAPad, kBGRA1, kBGRA1x2, k420v };
static const char *const kSurfaceNames[] = {"bgra", "bgrapad", "bgra1", "bgra1x2", "420v"};
enum { kSize = 64, kDefault = -1 };

struct Shape {
	const char *name;
	enum Surface surface;
	int plane;
	MTLPixelFormat format;
	MTLTextureType type;
	int width, height, depth, array, mips, samples;
	int storage; // kDefault: the descriptor's own
};

#define BGRA8 MTLPixelFormatBGRA8Unorm
#define R8 MTLPixelFormatR8Unorm
#define RG8 MTLPixelFormatRG8Unorm
#define BC1 MTLPixelFormatBC1_RGBA
#define BC7 MTLPixelFormatBC7_RGBAUnorm
#define T2D MTLTextureType2D
// A 2D texture of FORMAT at W x H over SURFACE's plane PLANE.
#define S(NAME, SURFACE, PLANE, FORMAT, W, H)                                                     \
	{NAME, SURFACE, PLANE, FORMAT, T2D, W, H, 1, 1, 1, 1, kDefault}
// The same of TYPE, DEPTH, ARRAY, MIPS and SAMPLES.
#define ST(NAME, SURFACE, PLANE, FORMAT, TYPE, W, H, DEPTH, ARRAY, MIPS, SAMPLES)                 \
	{NAME, SURFACE, PLANE, FORMAT, TYPE, W, H, DEPTH, ARRAY, MIPS, SAMPLES, kDefault}
// A 2D texture of STORAGE.
#define SS(NAME, SURFACE, FORMAT, W, H, STORAGE)                                                  \
	{NAME, SURFACE, 0, FORMAT, T2D, W, H, 1, 1, 1, 1, STORAGE}

static const struct Shape kShapes[] = {
    // The 2D textures inside their plane the runtime makes today: controls.
    S("bgra1.base", kBGRA1, 0, BGRA8, 64, 64),
    S("420v.p0.base", k420v, 0, R8, 64, 64),
    S("420v.p1.base", k420v, 1, RG8, 32, 32),
    // Rows wider than the plane's bytesPerRow (oracle_surftex 17, 19, 21).
    S("bgra1x2.65x64", kBGRA1x2, 0, BGRA8, 65, 64),
    S("bgra1x2.128x32", kBGRA1x2, 0, BGRA8, 128, 32),
    S("420v.p0.65x64", k420v, 0, R8, 65, 64),
    S("420v.p1.33x32", k420v, 1, RG8, 33, 32),
    // More rows than the plane (oracle_surftex 18, 20, 22, 23).
    S("bgra1x2.64x65", kBGRA1x2, 0, BGRA8, 64, 65),
    S("bgra1x2.64x128", kBGRA1x2, 0, BGRA8, 64, 128),
    S("420v.p0.64x65", k420v, 0, R8, 64, 65),
    S("420v.p0.64x96", k420v, 0, R8, 64, 96),
    S("420v.p1.32x33", k420v, 1, RG8, 32, 33),
    S("420v.p1.64x64", k420v, 1, RG8, 64, 64),
    // More than one level (oracle_surftex 42, 123).
    ST("bgra1x2.mips2", kBGRA1x2, 0, BGRA8, T2D, 64, 64, 1, 1, 2, 1),
    ST("bgra1x2.mips7", kBGRA1x2, 0, BGRA8, T2D, 64, 64, 1, 1, 7, 1),
    ST("420v.p0.mips2", k420v, 0, R8, T2D, 64, 64, 1, 1, 2, 1),
    // Other types (oracle_surftex 116 to 122, 127 to 129, 35).
    ST("bgra1x2.1d", kBGRA1x2, 0, BGRA8, MTLTextureType1D, 64, 1, 1, 1, 1, 1),
    ST("bgra1x2.1d.128", kBGRA1x2, 0, BGRA8, MTLTextureType1D, 128, 1, 1, 1, 1, 1),
    ST("bgra1x2.2darray.a1", kBGRA1x2, 0, BGRA8, MTLTextureType2DArray, 64, 64, 1, 1, 1, 1),
    ST("bgra1x2.2darray.a2", kBGRA1x2, 0, BGRA8, MTLTextureType2DArray, 64, 32, 1, 2, 1, 1),
    ST("bgra1x2.2darray.a2.64x64", kBGRA1x2, 0, BGRA8, MTLTextureType2DArray, 64, 64, 1, 2, 1,
       1),
    ST("bgra1x2.3d.d2", kBGRA1x2, 0, BGRA8, MTLTextureType3D, 64, 32, 2, 1, 1, 1),
    ST("420v.p0.3d.d2", k420v, 0, R8, MTLTextureType3D, 64, 32, 2, 1, 1, 1),
    ST("bgra1x2.cube.16", kBGRA1x2, 0, BGRA8, MTLTextureTypeCube, 16, 16, 1, 1, 1, 1),
    ST("bgra1x2.texbuf.64", kBGRA1x2, 0, BGRA8, MTLTextureTypeTextureBuffer, 64, 1, 1, 1, 1, 1),
    ST("bgra1x2.texbuf.128", kBGRA1x2, 0, BGRA8, MTLTextureTypeTextureBuffer, 128, 1, 1, 1, 1,
       1),
    ST("420v.p0.texbuf.64", k420v, 0, R8, MTLTextureTypeTextureBuffer, 64, 1, 1, 1, 1, 1),
    ST("bgra.texbuf.64", kBGRA, 0, BGRA8, MTLTextureTypeTextureBuffer, 64, 1, 1, 1, 1, 1),
    {"bgra1x2.2dms4.private", kBGRA1x2, 0, BGRA8, MTLTextureType2DMultisample, 64, 64, 1, 1, 1,
     4, MTLStorageModePrivate},
    // Storage modes over a plane.
    SS("bgra1x2.private", kBGRA1x2, BGRA8, 64, 64, MTLStorageModePrivate),
    SS("bgra1x2.shared", kBGRA1x2, BGRA8, 64, 64, MTLStorageModeShared),
    // Block-compressed formats over a surface's rows (oracle_surftex 66, 67, 125, 131, 147, 148).
    S("bgra.bc1", kBGRA, 0, BC1, 64, 64),
    S("bgrapad.bc1", kBGRAPad, 0, BC1, 64, 64),
    S("bgra.bc7", kBGRA, 0, BC7, 64, 64),
    S("bgra.bc1.62x64", kBGRA, 0, BC1, 62, 64),
    S("bgra.bc1.128x64", kBGRA, 0, BC1, 128, 64),
    SS("bgra.bc1.private", kBGRA, BC1, 64, 64, MTLStorageModePrivate),
    S("bgra1x2.bc1", kBGRA1x2, 0, BC1, 64, 64),
    // Past the surface's end (`--hazard` only, by the count above).
    S("bgra1.64x65", kBGRA1, 0, BGRA8, 64, 65),
    S("bgra1.64x128", kBGRA1, 0, BGRA8, 64, 128),
    ST("bgra1.mips2", kBGRA1, 0, BGRA8, T2D, 64, 64, 1, 1, 2, 1),
    S("420v.p1.32x512", k420v, 1, RG8, 32, 512),
};
enum { kShapeCount = sizeof kShapes / sizeof kShapes[0] };

// The bytes of a block of `format` and its texels' width and height (1 for an uncompressed one).
static void block_of(MTLPixelFormat format, uint64_t *bytes, uint64_t *extent)
{
	switch (format) {
	case BC1:
		*bytes = 8, *extent = 4;
		return;
	case BC7:
		*bytes = 16, *extent = 4;
		return;
	case R8:
		*bytes = 1, *extent = 1;
		return;
	case RG8:
		*bytes = 2, *extent = 1;
		return;
	default:
		*bytes = 4, *extent = 1;
		return;
	}
}

static IOSurfaceRef flat_surface(int bytes_per_row)
{
	NSMutableDictionary *props = [@{
		(id)kIOSurfaceWidth : @(kSize),
		(id)kIOSurfaceHeight : @(kSize),
		(id)kIOSurfaceBytesPerElement : @4,
		(id)kIOSurfacePixelFormat : @((uint32_t)'BGRA'),
	} mutableCopy];
	if (bytes_per_row != 0)
		props[(id)kIOSurfaceBytesPerRow] = @(bytes_per_row);
	return IOSurfaceCreate((__bridge CFDictionaryRef)props);
}

// A 'BGRA' surface of one plane dictionary, 64 x 64 at 256 bytes a row, in `alloc` bytes.
static IOSurfaceRef one_plane_surface(size_t alloc)
{
	const size_t row = kSize * 4;
	NSDictionary *props = @{
		(id)kIOSurfaceWidth : @(kSize),
		(id)kIOSurfaceHeight : @(kSize),
		(id)kIOSurfacePixelFormat : @((uint32_t)'BGRA'),
		(id)kIOSurfaceAllocSize : @(alloc),
		(id)kIOSurfacePlaneInfo : @[ @{
			(id)kIOSurfacePlaneWidth : @(kSize),
			(id)kIOSurfacePlaneHeight : @(kSize),
			(id)kIOSurfacePlaneBytesPerElement : @4,
			(id)kIOSurfacePlaneBytesPerRow : @(row),
			(id)kIOSurfacePlaneOffset : @0,
			(id)kIOSurfacePlaneSize : @(row * kSize),
		} ],
	};
	return IOSurfaceCreate((__bridge CFDictionaryRef)props);
}

static IOSurfaceRef make_surface(enum Surface surface, CVPixelBufferRef *keep)
{
	switch (surface) {
	case kBGRA:
		return flat_surface(0);
	case kBGRAPad:
		return flat_surface(512);
	case kBGRA1:
		return one_plane_surface(kSize * 4 * kSize);
	case kBGRA1x2:
		return one_plane_surface(2 * kSize * 4 * kSize);
	case k420v: {
		NSDictionary *attrs = @{(id)kCVPixelBufferIOSurfacePropertiesKey : @{}};
		if (CVPixelBufferCreate(kCFAllocatorDefault, kSize, kSize, '420v',
		                        (__bridge CFDictionaryRef)attrs, keep) != kCVReturnSuccess)
			return NULL;
		IOSurfaceRef s = CVPixelBufferGetIOSurface(*keep);
		if (s != NULL)
			CFRetain(s);
		return s;
	}
	}
	return NULL;
}

// The plane's offset from the surface's base and its bytes per row.
static void plane_of(IOSurfaceRef s, int plane, uint64_t *offset, uint64_t *bytes_per_row)
{
	if (IOSurfaceGetPlaneCount(s) == 0) {
		*offset = 0;
		*bytes_per_row = IOSurfaceGetBytesPerRow(s);
		return;
	}
	IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
	*offset = (uint64_t)((uint8_t *)IOSurfaceGetBaseAddressOfPlane(s, (size_t)plane) -
	                     (uint8_t *)IOSurfaceGetBaseAddress(s));
	IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
	*bytes_per_row = IOSurfaceGetBytesPerRowOfPlane(s, (size_t)plane);
}

static void tag_surface(IOSurfaceRef s)
{
	IOSurfaceLock(s, 0, NULL);
	fill_tag(IOSurfaceGetBaseAddress(s), IOSurfaceGetAllocSize(s), kReadMarker, 0);
	IOSurfaceUnlock(s, 0, NULL);
}

static void report_surface(const char *what, IOSurfaceRef s)
{
	IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
	report_changes(what, IOSurfaceGetBaseAddress(s), IOSurfaceGetAllocSize(s));
	IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
}

// The texture's levels, and each level's slices: its array's, its cube's faces, its 3D depth.
struct Extent {
	uint64_t width, height, slices;
};

static struct Extent level_extent(const struct Shape *c, int level)
{
	struct Extent e;
	e.width = (uint64_t)c->width >> level;
	e.height = (uint64_t)c->height >> level;
	if (e.width == 0)
		e.width = 1;
	if (e.height == 0)
		e.height = 1;
	switch (c->type) {
	case MTLTextureType2DArray:
		e.slices = (uint64_t)c->array;
		break;
	case MTLTextureTypeCube:
		e.slices = 6;
		break;
	case MTLTextureType3D:
		e.slices = (uint64_t)c->depth >> level ? (uint64_t)c->depth >> level : 1;
		break;
	case MTLTextureType2DMultisample:
		e.slices = (uint64_t)c->samples;
		break;
	default:
		e.slices = 1;
	}
	return e;
}

// The bytes of a row of `e`'s level (a row of blocks for a compressed format) and its rows.
static void rows_of(const struct Shape *c, struct Extent e, uint64_t *row_bytes, uint64_t *rows)
{
	uint64_t block, extent;
	block_of(c->format, &block, &extent);
	*row_bytes = (e.width + extent - 1) / extent * block;
	*rows = (e.height + extent - 1) / extent;
}

// Where the texture's rows end if every level, slice and face lay at the plane's pitch one after
// another from its offset: the count the hazard gate takes.
static uint64_t naive_end(const struct Shape *c, uint64_t offset, uint64_t bytes_per_row)
{
	uint64_t end = offset;
	for (int level = 0; level < c->mips; level++) {
		struct Extent e = level_extent(c, level);
		uint64_t row_bytes, rows;
		rows_of(c, e, &row_bytes, &rows);
		uint64_t pitch = bytes_per_row > row_bytes ? bytes_per_row : row_bytes;
		end += pitch * rows * e.slices;
	}
	return end;
}

static NSString *const kSource =
    @"#include <metal_stdlib>\n"
    @"using namespace metal;\n"
    @"struct P { uint w; uint h; uint level; uint slice; };\n"
    @"static uchar4 q(float4 c) { return uchar4(round(saturate(c) * 255.0)); }\n"
    @"kernel void r2d(texture2d<float, access::read> t [[texture(0)]], device uchar4 *o "
    @"[[buffer(0)]], constant P &p [[buffer(1)]], uint2 g [[thread_position_in_grid]]) {\n"
    @"  if (g.x < p.w && g.y < p.h) o[g.y * p.w + g.x] = q(t.read(g, p.level)); }\n"
    @"kernel void r2da(texture2d_array<float, access::read> t [[texture(0)]], device uchar4 *o "
    @"[[buffer(0)]], constant P &p [[buffer(1)]], uint2 g [[thread_position_in_grid]]) {\n"
    @"  if (g.x < p.w && g.y < p.h) o[g.y * p.w + g.x] = q(t.read(g, p.slice, p.level)); }\n"
    @"kernel void r3d(texture3d<float, access::read> t [[texture(0)]], device uchar4 *o "
    @"[[buffer(0)]], constant P &p [[buffer(1)]], uint2 g [[thread_position_in_grid]]) {\n"
    @"  if (g.x < p.w && g.y < p.h) o[g.y * p.w + g.x] = q(t.read(uint3(g, p.slice), "
    @"p.level)); }\n"
    @"kernel void rcube(texturecube<float, access::read> t [[texture(0)]], device uchar4 *o "
    @"[[buffer(0)]], constant P &p [[buffer(1)]], uint2 g [[thread_position_in_grid]]) {\n"
    @"  if (g.x < p.w && g.y < p.h) o[g.y * p.w + g.x] = q(t.read(g, p.slice, p.level)); }\n"
    @"kernel void r1d(texture1d<float, access::read> t [[texture(0)]], device uchar4 *o "
    @"[[buffer(0)]], constant P &p [[buffer(1)]], uint2 g [[thread_position_in_grid]]) {\n"
    @"  if (g.x < p.w && g.y == 0) o[g.x] = q(t.read(g.x)); }\n"
    @"kernel void rbuf(texture_buffer<float, access::read> t [[texture(0)]], device uchar4 *o "
    @"[[buffer(0)]], constant P &p [[buffer(1)]], uint2 g [[thread_position_in_grid]]) {\n"
    @"  if (g.x < p.w && g.y == 0) o[g.x] = q(t.read(g.x)); }\n"
    @"kernel void rms(texture2d_ms<float, access::read> t [[texture(0)]], device uchar4 *o "
    @"[[buffer(0)]], constant P &p [[buffer(1)]], uint2 g [[thread_position_in_grid]]) {\n"
    @"  if (g.x < p.w && g.y < p.h) o[g.y * p.w + g.x] = q(t.read(g, p.slice)); }\n";

static const char *kernel_for(MTLTextureType type)
{
	switch (type) {
	case MTLTextureType1D:
		return "r1d";
	case MTLTextureType2DArray:
		return "r2da";
	case MTLTextureType3D:
		return "r3d";
	case MTLTextureTypeCube:
		return "rcube";
	case MTLTextureTypeTextureBuffer:
		return "rbuf";
	case MTLTextureType2DMultisample:
		return "rms";
	default:
		return "r2d";
	}
}

// A kernel's colour of a texel as the texel's bytes in memory.
static void memory_bytes(MTLPixelFormat format, const uint8_t *rgba, uint8_t *out)
{
	switch (format) {
	case R8:
		out[0] = rgba[0];
		return;
	case RG8:
		out[0] = rgba[0], out[1] = rgba[1];
		return;
	default: // BGRA8Unorm
		out[0] = rgba[2], out[1] = rgba[1], out[2] = rgba[0], out[3] = rgba[3];
		return;
	}
}

static id<MTLDevice> pick_device(const char *name)
{
	for (id<MTLDevice> d in MTLCopyAllDevices())
		if (strstr(d.name.UTF8String, name) != NULL)
			return d;
	fprintf(stderr, "no device named like %s\n", name);
	exit(2);
}

static bool blits(const struct Shape *c)
{
	return c->type != MTLTextureTypeTextureBuffer && c->type != MTLTextureType2DMultisample;
}

static bool compressed(const struct Shape *c)
{
	return c->format == BC1 || c->format == BC7;
}

static void run_shape(id<MTLDevice> d, int n, bool hazard)
{
	const struct Shape *c = &kShapes[n];
	CVPixelBufferRef keep = NULL;
	IOSurfaceRef s = make_surface(c->surface, &keep);
	if (s == NULL) {
		printf("shape %d %s => no surface\n", n, c->name);
		return;
	}
	uint64_t offset, bytes_per_row;
	plane_of(s, c->plane, &offset, &bytes_per_row);
	uint64_t alloc = IOSurfaceGetAllocSize(s);
	printf("shape %d %s | %s: %zu planes, alloc %llu", n, c->name, kSurfaceNames[c->surface],
	       IOSurfaceGetPlaneCount(s), (unsigned long long)alloc);
	for (size_t p = 0; p < IOSurfaceGetPlaneCount(s); p++) {
		uint64_t o, b;
		plane_of(s, (int)p, &o, &b);
		printf("; plane %zu at %llu, %zux%zu, %llu a row", p, (unsigned long long)o,
		       IOSurfaceGetWidthOfPlane(s, p), IOSurfaceGetHeightOfPlane(s, p),
		       (unsigned long long)b);
	}
	if (IOSurfaceGetPlaneCount(s) == 0)
		printf("; %zu a row", IOSurfaceGetBytesPerRow(s));
	printf(" | plane %d: fmt %lu type %lu %dx%dx%d a %d m %d s %d storage %d\n", c->plane,
	       (unsigned long)c->format, (unsigned long)c->type, c->width, c->height, c->depth,
	       c->array, c->mips, c->samples, c->storage);
	uint64_t end = naive_end(c, offset, bytes_per_row);
	if (end > alloc && !hazard) {
		printf("shape %d %s => skipped: its rows at the plane's pitch end at %llu, past the "
		       "surface's %llu (--hazard runs it)\n",
		       n, c->name, (unsigned long long)end, (unsigned long long)alloc);
		return;
	}
	fflush(stdout);
	tag_surface(s);

	MTLTextureDescriptor *td = [MTLTextureDescriptor new];
	td.textureType = c->type;
	td.pixelFormat = c->format;
	td.width = (NSUInteger)c->width;
	td.height = (NSUInteger)c->height;
	td.depth = (NSUInteger)c->depth;
	td.arrayLength = (NSUInteger)c->array;
	td.mipmapLevelCount = (NSUInteger)c->mips;
	td.sampleCount = (NSUInteger)c->samples;
	td.usage = c->type == MTLTextureType2DMultisample
	               ? MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead
	               : MTLTextureUsageShaderRead;
	if (c->storage != kDefault)
		td.storageMode = (MTLStorageMode)c->storage;
	id<MTLTexture> t = [d newTextureWithDescriptor:td iosurface:s plane:(NSUInteger)c->plane];
	if (t == nil) {
		printf("shape %d %s => nil\n", n, c->name);
		return;
	}
	printf("shape %d %s => made: storage %lu\n", n, c->name, (unsigned long)t.storageMode);
	fflush(stdout);

	id<MTLCommandQueue> queue = [d newCommandQueue];
	bool cpu = t.storageMode != MTLStorageModePrivate && c->type != MTLTextureType2DMultisample;
	char what[96];

	// The texture's bytes as one tight run: each level, then each slice, rows one after another.
	uint64_t total = 0;
	for (int level = 0; level < c->mips; level++) {
		struct Extent e = level_extent(c, level);
		uint64_t row_bytes, rows;
		rows_of(c, e, &row_bytes, &rows);
		printf("  texture bytes from %llu: level %d, %llu slices of %llu rows of %llu\n",
		       (unsigned long long)total, level, (unsigned long long)e.slices,
		       (unsigned long long)rows, (unsigned long long)row_bytes);
		total += row_bytes * rows * e.slices;
	}
	id<MTLBuffer> out = [d newBufferWithLength:total + 64 options:MTLResourceStorageModeShared];

	// 2. Reads: blits, kernels, getBytes.
	if (blits(c)) {
		id<MTLCommandBuffer> cb = [queue commandBuffer];
		id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
		uint64_t at = 0;
		for (int level = 0; level < c->mips; level++) {
			struct Extent e = level_extent(c, level);
			uint64_t row_bytes, rows;
			rows_of(c, e, &row_bytes, &rows);
			for (uint64_t slice = 0; slice < e.slices; slice++) {
				bool three = c->type == MTLTextureType3D;
				[blit copyFromTexture:t
				                 sourceSlice:three ? 0 : (NSUInteger)slice
				                 sourceLevel:(NSUInteger)level
				                sourceOrigin:MTLOriginMake(0, 0, three ? (NSUInteger)slice : 0)
				                  sourceSize:MTLSizeMake((NSUInteger)e.width,
				                                         (NSUInteger)e.height, 1)
				                    toBuffer:out
				           destinationOffset:(NSUInteger)at
				      destinationBytesPerRow:(NSUInteger)row_bytes
				    destinationBytesPerImage:(NSUInteger)(row_bytes * rows)];
				at += row_bytes * rows;
			}
		}
		[blit endEncoding];
		[cb commit];
		[cb waitUntilCompleted];
		at = 0;
		for (int level = 0; level < c->mips; level++) {
			struct Extent e = level_extent(c, level);
			uint64_t row_bytes, rows;
			rows_of(c, e, &row_bytes, &rows);
			for (uint64_t slice = 0; slice < e.slices; slice++) {
				snprintf(what, sizeof what, "blit read L%d S%llu", level,
				         (unsigned long long)slice);
				report_rows(what, (uint8_t *)out.contents + at, row_bytes, rows);
				at += row_bytes * rows;
			}
		}
		fflush(stdout);
	}
	if (!compressed(c)) {
		NSError *err = nil;
		id<MTLLibrary> lib = [d newLibraryWithSource:kSource options:nil error:&err];
		id<MTLFunction> f = [lib newFunctionWithName:@(kernel_for(c->type))];
		id<MTLComputePipelineState> ps = f ? [d newComputePipelineStateWithFunction:f
		                                                                      error:&err]
		                                   : nil;
		if (ps == nil) {
			printf("  kernel: no pipeline (%s)\n", err.localizedDescription.UTF8String);
		} else {
			uint64_t block, extent;
			block_of(c->format, &block, &extent);
			for (int level = 0; level < c->mips; level++) {
				struct Extent e = level_extent(c, level);
				for (uint64_t slice = 0; slice < e.slices; slice++) {
					id<MTLBuffer> rgba =
					    [d newBufferWithLength:e.width * e.height * 4
					                   options:MTLResourceStorageModeShared];
					uint32_t p[4] = {(uint32_t)e.width, (uint32_t)e.height, (uint32_t)level,
					                 (uint32_t)slice};
					id<MTLCommandBuffer> cb = [queue commandBuffer];
					id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
					[ce setComputePipelineState:ps];
					[ce setTexture:t atIndex:0];
					[ce setBuffer:rgba offset:0 atIndex:0];
					[ce setBytes:p length:sizeof p atIndex:1];
					MTLSize group = MTLSizeMake(8, 8, 1);
					[ce dispatchThreadgroups:MTLSizeMake((NSUInteger)(e.width + 7) / 8,
					                                     (NSUInteger)(e.height + 7) / 8, 1)
					    threadsPerThreadgroup:group];
					[ce endEncoding];
					[cb commit];
					[cb waitUntilCompleted];
					NSMutableData *bytes = [NSMutableData dataWithLength:e.width * e.height * block];
					const uint8_t *src = rgba.contents;
					for (uint64_t k = 0; k < e.width * e.height; k++)
						memory_bytes(c->format, src + 4 * k, (uint8_t *)bytes.mutableBytes + k * block);
					snprintf(what, sizeof what, "kernel read L%d %s%llu", level,
					         c->type == MTLTextureType2DMultisample ? "sample " : "S",
					         (unsigned long long)slice);
					report_rows(what, bytes.bytes, e.width * block, e.height);
				}
			}
		}
		fflush(stdout);
	}
	if (cpu && c->type != MTLTextureTypeTextureBuffer) {
		for (int level = 0; level < c->mips; level++) {
			struct Extent e = level_extent(c, level);
			uint64_t row_bytes, rows;
			rows_of(c, e, &row_bytes, &rows);
			for (uint64_t slice = 0; slice < e.slices; slice++) {
				NSMutableData *bytes = [NSMutableData dataWithLength:row_bytes * rows];
				bool three = c->type == MTLTextureType3D;
				MTLRegion region = MTLRegionMake3D(0, 0, three ? (NSUInteger)slice : 0,
				                                   (NSUInteger)e.width, (NSUInteger)e.height, 1);
				[t getBytes:bytes.mutableBytes
				    bytesPerRow:(NSUInteger)row_bytes
				  bytesPerImage:(NSUInteger)(row_bytes * rows)
				     fromRegion:region
				    mipmapLevel:(NSUInteger)level
				          slice:three ? 0 : (NSUInteger)slice];
				snprintf(what, sizeof what, "getBytes L%d S%llu", level, (unsigned long long)slice);
				report_rows(what, bytes.bytes, row_bytes, rows);
			}
		}
		fflush(stdout);
	}

	// 3. Writes: a blit from a buffer of the write's tag, then replaceRegion.
	if (blits(c)) {
		tag_surface(s);
		id<MTLBuffer> in = [d newBufferWithLength:total + 64 options:MTLResourceStorageModeShared];
		fill_tag(in.contents, total, kWriteMarker, 0);
		id<MTLCommandBuffer> cb = [queue commandBuffer];
		id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
		uint64_t at = 0;
		for (int level = 0; level < c->mips; level++) {
			struct Extent e = level_extent(c, level);
			uint64_t row_bytes, rows;
			rows_of(c, e, &row_bytes, &rows);
			for (uint64_t slice = 0; slice < e.slices; slice++) {
				bool three = c->type == MTLTextureType3D;
				[blit copyFromBuffer:in
				               sourceOffset:(NSUInteger)at
				          sourceBytesPerRow:(NSUInteger)row_bytes
				        sourceBytesPerImage:(NSUInteger)(row_bytes * rows)
				                 sourceSize:MTLSizeMake((NSUInteger)e.width,
				                                        (NSUInteger)e.height, 1)
				                  toTexture:t
				           destinationSlice:three ? 0 : (NSUInteger)slice
				           destinationLevel:(NSUInteger)level
				          destinationOrigin:MTLOriginMake(0, 0, three ? (NSUInteger)slice : 0)];
				at += row_bytes * rows;
			}
		}
		[blit endEncoding];
		[cb commit];
		[cb waitUntilCompleted];
		report_surface("blit write", s);
		if (t.storageMode == MTLStorageModeManaged) {
			id<MTLCommandBuffer> sync = [queue commandBuffer];
			id<MTLBlitCommandEncoder> sb = [sync blitCommandEncoder];
			[sb synchronizeResource:t];
			[sb endEncoding];
			[sync commit];
			[sync waitUntilCompleted];
			report_surface("blit write, then synchronizeResource", s);
		}
		fflush(stdout);
	}
	if (c->type == MTLTextureType2DMultisample) {
		tag_surface(s);
		MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
		rp.colorAttachments[0].texture = t;
		rp.colorAttachments[0].loadAction = MTLLoadActionClear;
		rp.colorAttachments[0].storeAction = MTLStoreActionStore;
		rp.colorAttachments[0].clearColor = MTLClearColorMake(0.25, 0.5, 0.75, 1.0);
		id<MTLCommandBuffer> cb = [queue commandBuffer];
		[[cb renderCommandEncoderWithDescriptor:rp] endEncoding];
		[cb commit];
		[cb waitUntilCompleted];
		report_surface("render pass clear and store", s);
		fflush(stdout);
	}
	if (cpu && c->type != MTLTextureTypeTextureBuffer) {
		tag_surface(s);
		NSMutableData *in = [NSMutableData dataWithLength:total];
		fill_tag(in.mutableBytes, total, kWriteMarker, 0);
		uint64_t at = 0;
		for (int level = 0; level < c->mips; level++) {
			struct Extent e = level_extent(c, level);
			uint64_t row_bytes, rows;
			rows_of(c, e, &row_bytes, &rows);
			for (uint64_t slice = 0; slice < e.slices; slice++) {
				bool three = c->type == MTLTextureType3D;
				MTLRegion region = MTLRegionMake3D(0, 0, three ? (NSUInteger)slice : 0,
				                                   (NSUInteger)e.width, (NSUInteger)e.height, 1);
				[t replaceRegion:region
				     mipmapLevel:(NSUInteger)level
				           slice:three ? 0 : (NSUInteger)slice
				       withBytes:(uint8_t *)in.mutableBytes + at
				     bytesPerRow:(NSUInteger)row_bytes
				   bytesPerImage:(NSUInteger)(row_bytes * rows)];
				at += row_bytes * rows;
			}
		}
		report_surface("replaceRegion", s);
		fflush(stdout);
	}
	// A texture buffer's CPU access last: Metal may assert on it.
	if (cpu && c->type == MTLTextureTypeTextureBuffer) {
		uint64_t row_bytes, rows;
		rows_of(c, level_extent(c, 0), &row_bytes, &rows);
		NSMutableData *bytes = [NSMutableData dataWithLength:row_bytes];
		printf("  texture buffer getBytes:\n");
		fflush(stdout);
		[t getBytes:bytes.mutableBytes
		    bytesPerRow:(NSUInteger)row_bytes
		     fromRegion:MTLRegionMake1D(0, (NSUInteger)c->width)
		    mipmapLevel:0];
		report_rows("getBytes", bytes.bytes, row_bytes, 1);
		tag_surface(s);
		fill_tag(bytes.mutableBytes, row_bytes, kWriteMarker, 0);
		printf("  texture buffer replaceRegion:\n");
		fflush(stdout);
		[t replaceRegion:MTLRegionMake1D(0, (NSUInteger)c->width)
		     mipmapLevel:0
		       withBytes:bytes.bytes
		     bytesPerRow:(NSUInteger)row_bytes];
		report_surface("replaceRegion", s);
	}
	printf("shape %d %s => done\n", n, c->name);
	fflush(stdout);
	t = nil;
	CFRelease(s);
	if (keep != NULL)
		CVPixelBufferRelease(keep);
}

int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IOLBF, 0);
	if (argc < 2) {
		fprintf(stderr, "usage: %s DEVICE [--hazard] [--shape N [--hazard] | --list]\n", argv[0]);
		return 2;
	}
	bool hazard = false;
	int shape = -1;
	for (int a = 2; a < argc; a++) {
		if (!strcmp(argv[a], "--hazard"))
			hazard = true;
		else if (!strcmp(argv[a], "--list")) {
			for (int n = 0; n < kShapeCount; n++)
				printf("%d %s\n", n, kShapes[n].name);
			return 0;
		} else if (!strcmp(argv[a], "--shape") && a + 1 < argc)
			shape = atoi(argv[++a]);
	}
	if (shape >= 0) {
		// Metal's words where it ends the process go to stderr: keep them in the shape's lines.
		dup2(STDOUT_FILENO, STDERR_FILENO);
		if (shape >= kShapeCount)
			return 2;
		@autoreleasepool {
			run_shape(pick_device(argv[1]), shape, hazard);
		}
		return 0;
	}
	id<MTLDevice> d = pick_device(argv[1]);
	printf("surface_texture_envelope on %s, %s\n", d.name.UTF8String,
	       NSProcessInfo.processInfo.operatingSystemVersionString.UTF8String);
	fflush(stdout);
	for (int n = 0; n < kShapeCount; n++) {
		char index[16];
		snprintf(index, sizeof index, "%d", n);
		char *args[] = {argv[0], argv[1], "--shape", index, hazard ? "--hazard" : NULL, NULL};
		pid_t pid;
		if (posix_spawn(&pid, argv[0], NULL, NULL, args, environ))
			return 1;
		int st;
		waitpid(pid, &st, 0);
		if (WIFSIGNALED(st))
			printf("=> shape %d died on signal %d\n", n, WTERMSIG(st));
		else if (WEXITSTATUS(st))
			printf("=> shape %d exited %d\n", n, WEXITSTATUS(st));
		fflush(stdout);
	}
	return 0;
}
