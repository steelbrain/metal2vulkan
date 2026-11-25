// Whether Metal takes a copy of a compressed texture's level smaller than one block whose size is
// the whole block (4x4 texels) where Apple's page for the copy asks for the level's own edge, and
// what lands. Apple's page for `copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:
// toTexture:destinationSlice:destinationLevel:destinationOrigin:`: "If sourceTexture uses a
// compressed pixel format, set sourceSize to a multiple of the pixel format's block size. If the
// block extends outside the bounds of the texture, clamp sourceSize to the edge of the texture."
// MTLBlitCommandEncoder.h and MTLTexture.h say nothing of blocks for these calls.
//
// ONE CASE PER PROCESS, chosen on the command line, run once under MTL_DEBUG_LAYER=1 and once
// without it (`--all` runs every case both ways, each in a process of its own, and prints how
// each process ended): the validation layer ends the process at a call it does not take, so a
// case that prints "taken" got past the call. Every texture is 2048x1024 with 12 levels, so
// level 8 is 8x4 texels (two blocks), 9 is 4x2, 10 is 2x1 and 11 is 1x1 (one block each). A case
// is FORMAT-KIND-LEVEL-SIZE:
//
//   FORMAT  bc1 (BC1_RGBA, 8-byte blocks) or bc3 (BC3_RGBA, 16-byte blocks)
//   KIND    mp       texture to texture, Managed to Private (the shape a guest met)
//           pp       texture to texture, Private to Private
//           b2t      buffer to texture, into a Private texture
//           t2b      texture to buffer, of a Private texture
//           replace  replaceRegion into a Managed texture
//           get      getBytes of a Managed texture
//   LEVEL   8 (the control: whole blocks), 9, 10 or 11
//   SIZE    clamp    the level's own extent, at (0,0) (level 8: 8x4)
//           block    4x4 at (0,0) (level 8: 4x4 at (4,0), whole blocks)
//           over     8x4 at (0,0), level 9 alone: past the level by more than the block's rounding
//
// and, for mp and pp alone, a copy between two levels, LEVEL being AtoB: 9to8, 8to9, 10to8 or
// 8to10, level 8's place being its second block, (4,0), and the other's (0,0); SIZE block is 4x4,
// and clamp the smaller level's extent (4x2, or 2x1), which is the source's edge in 9to8 and
// 10to8 (Apple's form) and the destination's in 8to9 and 8to10.
//
// Levels 8 to 11 of a source texture hold blocks whose byte k (in the level's rows of blocks) is
// 0x11 * level + 3 * k + 1; every destination texture's levels 8 to 11 hold 0xdd, a buffer
// destination 64 bytes of 0xee, and a buffer source's blocks past the level 0xaa. The fills are
// the clamped size (Apple's form). A texture destination's levels 8 to 11 are read back by a blit
// into a Shared buffer, each block named by its source: `s` the source block at its place, `m`
// the block a copy between two levels copied, `d` the destination's 0xdd, `a` the buffer's 0xaa,
// `?` anything else; then the level's bytes, and
// the FNV-1a 64 of all four levels. A buffer destination's 64 bytes are named a block at a time
// the same way (`e` its 0xee), in the copy's row order.
//
//   xcrun clang -fobjc-arc -framework Metal -framework Foundation \
//       -o compressed_copies_of_small_levels compressed_copies_of_small_levels.m
//   ./compressed_copies_of_small_levels --all
//   MTL_DEBUG_LAYER=1 ./compressed_copies_of_small_levels bc3-mp-9-block
//
// WHAT IT MEASURED, Apple M3 Max, macOS 27.0.1 (Build 26A434), job m3-max-73696, BC1 and BC3
// alike:
//
//   - with the layer off, every case is taken and completes (status 4, no error);
//   - 4x4 at levels 9, 10 and 11 lands exactly what the clamped size lands, by every kind: the
//     level's one block, everything else kept (levels FNV-1a, BC1 then BC3: level 9
//     df214fe18bfe662d and cd215888fb218a95, 10 fda65c9a502da7a5 and 716f1b3f1cb39e25, 11
//     3145d998df908f15 and 0a3e663ab6db29e5; t2b and get bytes, level 9 eb66b13e7d8437ad and
//     b5b6a640cc0c0ed5, 10 e466c8dac77e3825 and e80282a8667247e5, 11 e2d8ac956a0c98d5 and
//     7e4011a76428e2e5);
//   - with the layer on, 4x4 at levels 9 to 11 runs as with it off for mp, pp, t2b, replace and
//     get (the layer bounds the region by the level rounded up to the block: "(origin.x +
//     adjustedSize.width)(8) must be <= paddedWidth(4)"), and ends the process (SIGABRT) for
//     b2t: "(destinationOrigin.y + sourceSize.height)(4) must be <= height(2)";
//   - over (8x4 at level 9) ends the process under the layer for every kind ("must be <=
//     paddedWidth(4)"); without it mp, pp and b2t land the level's block alone, t2b writes the
//     block and a block of 0 after it, replace writes nothing and get reads nothing;
//   - between levels, 4x4 is taken with the layer and without, and 9to8, 8to9 and 10to8 land
//     the block named whole (`m`); 8to10 lands level 8's first block, (0,0), not the one at
//     (4,0) it names; the smaller level's extent lands the same, and ends the process under the
//     layer: "size.height(2) must be a multiple of MTLPixelFormatBC1_RGBA blockHeight(4)" (2x1:
//     the width too), Apple's clamped form into a larger level (9to8, 10to8) among them;
//   - level 8, whole blocks, is taken everywhere and lands its blocks.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

enum { WIDTH = 2048, HEIGHT = 1024, FIRST = 8, LAST = 11, OUT = 64 };

static const char *FORMATS[] = {"bc1", "bc3"};
static const char *KINDS[] = {"mp", "pp", "b2t", "t2b", "replace", "get"};
static const char *SHAPES[] = {"8-clamp", "8-block",  "9-clamp",  "9-block", "9-over",
                               "10-clamp", "10-block", "11-clamp", "11-block"};
static const char *MIXED[] = {"9to8-block",  "9to8-clamp",  "8to9-block",  "8to9-clamp",
                              "10to8-block", "10to8-clamp", "8to10-block", "8to10-clamp"};
#define COUNT(a) (sizeof(a) / sizeof((a)[0]))

typedef struct {
    NSUInteger x, y, w, h;
} Box;

static NSUInteger side(NSUInteger extent, NSUInteger level) {
    NSUInteger e = extent >> level;
    return e ? e : 1;
}

static NSUInteger blocks(NSUInteger texels) { return (texels + 3) / 4; }

static NSUInteger across(NSUInteger level) { return blocks(side(WIDTH, level)); }

static NSUInteger high(NSUInteger level) { return blocks(side(HEIGHT, level)); }

static uint8_t source(NSUInteger level, NSUInteger k) { return (uint8_t)(0x11 * level + 3 * k + 1); }

// Block (bx, by) of `level` in the source, or NO past the level.
static BOOL source_block(NSUInteger level, NSUInteger bx, NSUInteger by, NSUInteger unit,
                         uint8_t *out) {
    if (bx >= across(level) || by >= high(level)) {
        return NO;
    }
    NSUInteger k0 = (by * across(level) + bx) * unit;
    for (NSUInteger i = 0; i < unit; i++) {
        out[i] = source(level, k0 + i);
    }
    return YES;
}

static BOOL all(const uint8_t *bytes, NSUInteger length, uint8_t value) {
    for (NSUInteger i = 0; i < length; i++) {
        if (bytes[i] != value) {
            return NO;
        }
    }
    return YES;
}

// The class of `block`, which lies where source block (bx, by) of `level` would; `moved` the
// block a copy between two levels copied, or NULL.
static char name_block(const uint8_t *block, NSUInteger level, NSUInteger bx, NSUInteger by,
                       NSUInteger unit, uint8_t blank, const uint8_t *moved) {
    uint8_t want[16];
    if (source_block(level, bx, by, unit, want) && memcmp(block, want, unit) == 0) {
        return 's';
    }
    if (moved && memcmp(block, moved, unit) == 0) {
        return 'm';
    }
    if (all(block, unit, blank)) {
        return blank == 0xdd ? 'd' : 'e';
    }
    if (all(block, unit, 0xaa)) {
        return 'a';
    }
    return '?';
}

static uint64_t fnv(const uint8_t *bytes, NSUInteger length) {
    uint64_t hash = 0xcbf29ce484222325ull;
    for (NSUInteger i = 0; i < length; i++) {
        hash ^= bytes[i];
        hash *= 0x100000001b3ull;
    }
    return hash;
}

static void hex(const uint8_t *bytes, NSUInteger length) {
    for (NSUInteger i = 0; i < length; i++) {
        printf("%s%02x", i && i % 8 == 0 ? " " : "", bytes[i]);
    }
}

// The bytes of a region `box` of `level` in rows of blocks, packed: source blocks, 0xaa past the
// level.
static NSData *region_bytes(NSUInteger level, Box box, NSUInteger unit) {
    NSUInteger rw = blocks(box.w), rh = blocks(box.h);
    NSMutableData *data = [NSMutableData dataWithLength:rw * rh * unit];
    uint8_t *bytes = data.mutableBytes;
    for (NSUInteger j = 0; j < rh; j++) {
        for (NSUInteger i = 0; i < rw; i++) {
            uint8_t *at = bytes + (j * rw + i) * unit;
            if (!source_block(level, box.x / 4 + i, box.y / 4 + j, unit, at)) {
                memset(at, 0xaa, unit);
            }
        }
    }
    return data;
}

// A whole level's bytes: `value`, or the source's where it is below 0.
static NSData *level_bytes(NSUInteger level, NSUInteger unit, int value) {
    Box box = {0, 0, side(WIDTH, level), side(HEIGHT, level)};
    if (value < 0) {
        return region_bytes(level, box, unit);
    }
    NSMutableData *data = [NSMutableData dataWithLength:across(level) * high(level) * unit];
    memset(data.mutableBytes, value, data.length);
    return data;
}

static MTLRegion whole(NSUInteger level) {
    return MTLRegionMake2D(0, 0, side(WIDTH, level), side(HEIGHT, level));
}

static id<MTLTexture> texture(id<MTLDevice> device, MTLPixelFormat format, MTLStorageMode mode) {
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                                 width:WIDTH
                                                                                height:HEIGHT
                                                                             mipmapped:YES];
    d.storageMode = mode;
    d.usage = MTLTextureUsageShaderRead;
    return [device newTextureWithDescriptor:d];
}

static void wait_for(id<MTLCommandBuffer> cb, const char *what) {
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        printf("%s: status %ld, error %s\n", what, (long)cb.status,
               cb.error ? cb.error.description.UTF8String : "none");
    }
}

// Levels 8 to 11 of `t` filled with `value`, or the source's blocks where it is below 0: by
// replaceRegion where the texture is Managed, else by a blit from a Shared buffer.
static void fill(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLTexture> t,
                 NSUInteger unit, int value) {
    if (t.storageMode == MTLStorageModeManaged) {
        for (NSUInteger level = FIRST; level <= LAST; level++) {
            NSData *bytes = level_bytes(level, unit, value);
            [t replaceRegion:whole(level)
                 mipmapLevel:level
                   withBytes:bytes.bytes
                 bytesPerRow:across(level) * unit];
        }
        return;
    }
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    for (NSUInteger level = FIRST; level <= LAST; level++) {
        NSData *bytes = level_bytes(level, unit, value);
        id<MTLBuffer> staging = [device newBufferWithBytes:bytes.bytes
                                                    length:bytes.length
                                                   options:MTLResourceStorageModeShared];
        [blit copyFromBuffer:staging
                   sourceOffset:0
              sourceBytesPerRow:across(level) * unit
            sourceBytesPerImage:0
                     sourceSize:whole(level).size
                      toTexture:t
               destinationSlice:0
               destinationLevel:level
              destinationOrigin:MTLOriginMake(0, 0, 0)];
    }
    [blit endEncoding];
    wait_for(cb, "fill");
}

// Levels 8 to 11 of `t`, read back by a blit and named a block at a time.
static void report_texture(const char *name, id<MTLDevice> device, id<MTLCommandQueue> queue,
                           id<MTLTexture> t, NSUInteger unit, const uint8_t *moved) {
    NSUInteger at[LAST + 2] = {0};
    for (NSUInteger level = FIRST; level <= LAST; level++) {
        at[level + 1] = at[level] + across(level) * high(level) * unit;
    }
    id<MTLBuffer> out = [device newBufferWithLength:at[LAST + 1]
                                            options:MTLResourceStorageModeShared];
    memset(out.contents, 0xbb, out.length);
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    for (NSUInteger level = FIRST; level <= LAST; level++) {
        [blit copyFromTexture:t
                         sourceSlice:0
                         sourceLevel:level
                        sourceOrigin:MTLOriginMake(0, 0, 0)
                          sourceSize:whole(level).size
                            toBuffer:out
                   destinationOffset:at[level]
              destinationBytesPerRow:across(level) * unit
            destinationBytesPerImage:across(level) * high(level) * unit];
    }
    [blit endEncoding];
    wait_for(cb, "read back");
    const uint8_t *bytes = out.contents;
    for (NSUInteger level = FIRST; level <= LAST; level++) {
        printf("%s: level %lu ", name, (unsigned long)level);
        for (NSUInteger by = 0; by < high(level); by++) {
            for (NSUInteger bx = 0; bx < across(level); bx++) {
                const uint8_t *block = bytes + at[level] + (by * across(level) + bx) * unit;
                printf("%c", name_block(block, level, bx, by, unit, 0xdd, moved));
            }
        }
        printf(" ");
        hex(bytes + at[level], at[level + 1] - at[level]);
        printf("\n");
    }
    printf("%s: levels fnv %016llx\n", name, (unsigned long long)fnv(bytes, out.length));
}

// A buffer destination's bytes, named a block at a time in the copy's row order.
static void report_bytes(const char *name, const uint8_t *bytes, NSUInteger level, Box box,
                         NSUInteger unit) {
    NSUInteger rw = blocks(box.w), rh = blocks(box.h);
    printf("%s: bytes ", name);
    for (NSUInteger i = 0; i < OUT / unit; i++) {
        NSUInteger bx = i < rw * rh ? box.x / 4 + i % rw : ~(NSUInteger)0;
        NSUInteger by = i < rw * rh ? box.y / 4 + i / rw : ~(NSUInteger)0;
        printf("%c", name_block(bytes + i * unit, level, bx, by, unit, 0xee, NULL));
    }
    printf(" ");
    hex(bytes, OUT);
    printf("\n%s: bytes fnv %016llx\n", name, (unsigned long long)fnv(bytes, OUT));
}

static int run_case(const char *name) {
    char format_name[8], kind[16], levels[16], shape[16];
    unsigned from_number = 0, to_number = 0;
    if (sscanf(name, "%7[^-]-%15[^-]-%15[^-]-%15s", format_name, kind, levels, shape) != 4) {
        printf("no case %s\n", name);
        return 2;
    }
    int parsed = sscanf(levels, "%uto%u", &from_number, &to_number);
    BOOL mixed = parsed == 2;
    NSUInteger level = from_number, to_level = mixed ? to_number : from_number;
    BOOL bc3 = strcmp(format_name, "bc3") == 0;
    BOOL t2t = strcmp(kind, "mp") == 0 || strcmp(kind, "pp") == 0;
    if ((!bc3 && strcmp(format_name, "bc1") != 0) || parsed < 1 || level < FIRST || level > LAST ||
        to_level < FIRST || to_level > LAST || (mixed && (!t2t || strcmp(shape, "over") == 0))) {
        printf("no case %s\n", name);
        return 2;
    }
    MTLPixelFormat format = bc3 ? MTLPixelFormatBC3_RGBA : MTLPixelFormatBC1_RGBA;
    NSUInteger unit = bc3 ? 16 : 8;
    Box box;
    if (strcmp(shape, "clamp") == 0) {
        box = (Box){0, 0, side(WIDTH, level), side(HEIGHT, level)};
    } else if (strcmp(shape, "block") == 0) {
        box = (Box){level == FIRST ? 4 : 0, 0, 4, 4};
    } else if (strcmp(shape, "over") == 0) {
        box = (Box){0, 0, 8, 4};
    } else {
        printf("no case %s\n", name);
        return 2;
    }
    NSUInteger to_x = box.x;
    if (mixed) {
        box.x = level == FIRST ? 4 : 0;
        to_x = to_level == FIRST ? 4 : 0;
        if (strcmp(shape, "clamp") == 0) {
            NSUInteger small = level == FIRST ? to_level : level;
            box.w = side(WIDTH, small);
            box.h = side(HEIGHT, small);
        }
    }
    NSUInteger row = blocks(box.w) * unit, image = row * blocks(box.h);
    MTLOrigin origin = MTLOriginMake(box.x, box.y, 0);
    MTLOrigin to_origin = MTLOriginMake(to_x, box.y, 0);
    MTLSize size = MTLSizeMake(box.w, box.h, 1);
    MTLRegion region = MTLRegionMake2D(box.x, box.y, box.w, box.h);

    const char *layer = getenv("MTL_DEBUG_LAYER");
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    printf("%s: %s, %s, BC %s, MTL_DEBUG_LAYER %s, region (%lu,%lu) %lux%lu of %lux%lu, to "
           "(%lu,%lu) of %lux%lu\n",
           name, device.name.UTF8String,
           [NSProcessInfo processInfo].operatingSystemVersionString.UTF8String,
           device.supportsBCTextureCompression ? "yes" : "no", layer ? layer : "unset",
           (unsigned long)box.x, (unsigned long)box.y, (unsigned long)box.w,
           (unsigned long)box.h, (unsigned long)side(WIDTH, level),
           (unsigned long)side(HEIGHT, level), (unsigned long)to_x, (unsigned long)box.y,
           (unsigned long)side(WIDTH, to_level), (unsigned long)side(HEIGHT, to_level));
    id<MTLCommandQueue> queue = [device newCommandQueue];

    if (strcmp(kind, "mp") == 0 || strcmp(kind, "pp") == 0 || strcmp(kind, "b2t") == 0 ||
        strcmp(kind, "t2b") == 0) {
        BOOL managed = strcmp(kind, "mp") == 0;
        id<MTLTexture> from = nil, to = nil;
        id<MTLBuffer> bytes = nil;
        if (strcmp(kind, "b2t") == 0) {
            NSData *data = region_bytes(level, box, unit);
            bytes = [device newBufferWithBytes:data.bytes
                                        length:data.length
                                       options:MTLResourceStorageModeShared];
        } else {
            from = texture(device, format,
                           managed ? MTLStorageModeManaged : MTLStorageModePrivate);
            fill(device, queue, from, unit, -1);
        }
        if (strcmp(kind, "t2b") == 0) {
            bytes = [device newBufferWithLength:OUT options:MTLResourceStorageModeShared];
            memset(bytes.contents, 0xee, OUT);
        } else {
            to = texture(device, format, MTLStorageModePrivate);
            fill(device, queue, to, unit, 0xdd);
        }
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        printf("%s: calling\n", name);
        if (strcmp(kind, "b2t") == 0) {
            [blit copyFromBuffer:bytes
                       sourceOffset:0
                  sourceBytesPerRow:row
                sourceBytesPerImage:0
                         sourceSize:size
                          toTexture:to
                   destinationSlice:0
                   destinationLevel:level
                  destinationOrigin:origin];
        } else if (strcmp(kind, "t2b") == 0) {
            [blit copyFromTexture:from
                             sourceSlice:0
                             sourceLevel:level
                            sourceOrigin:origin
                              sourceSize:size
                                toBuffer:bytes
                       destinationOffset:0
                  destinationBytesPerRow:row
                destinationBytesPerImage:image];
        } else {
            [blit copyFromTexture:from
                      sourceSlice:0
                      sourceLevel:level
                     sourceOrigin:origin
                       sourceSize:size
                        toTexture:to
                 destinationSlice:0
                 destinationLevel:to_level
                destinationOrigin:to_origin];
        }
        printf("%s: taken\n", name);
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        printf("%s: status %ld, error %s\n", name, (long)cb.status,
               cb.error ? cb.error.description.UTF8String : "none");
        uint8_t moved[16];
        if (to) {
            BOOL any = mixed && source_block(level, box.x / 4, 0, unit, moved);
            report_texture(name, device, queue, to, unit, any ? moved : NULL);
        } else {
            report_bytes(name, bytes.contents, level, box, unit);
        }
    } else if (strcmp(kind, "replace") == 0) {
        id<MTLTexture> to = texture(device, format, MTLStorageModeManaged);
        fill(device, queue, to, unit, 0xdd);
        NSData *data = region_bytes(level, box, unit);
        printf("%s: calling\n", name);
        [to replaceRegion:region mipmapLevel:level withBytes:data.bytes bytesPerRow:row];
        printf("%s: taken\n", name);
        report_texture(name, device, queue, to, unit, NULL);
    } else if (strcmp(kind, "get") == 0) {
        id<MTLTexture> from = texture(device, format, MTLStorageModeManaged);
        fill(device, queue, from, unit, -1);
        uint8_t out[OUT];
        memset(out, 0xee, OUT);
        printf("%s: calling\n", name);
        [from getBytes:out bytesPerRow:row fromRegion:region mipmapLevel:level];
        printf("%s: taken\n", name);
        report_bytes(name, out, level, box, unit);
    } else {
        printf("no case %s\n", name);
        return 2;
    }
    return 0;
}

// Every case, each in a process of its own, under the layer and without it.
static int run_all(const char *self) {
    NSUInteger count = 0;
    while (environ[count]) {
        count++;
    }
    char **plain = calloc(count + 2, sizeof(char *));
    char **layered = calloc(count + 2, sizeof(char *));
    NSUInteger n = 0;
    for (NSUInteger i = 0; i < count; i++) {
        if (strncmp(environ[i], "MTL_DEBUG_LAYER=", 16) != 0) {
            plain[n] = environ[i];
            layered[n] = environ[i];
            n++;
        }
    }
    layered[n] = "MTL_DEBUG_LAYER=1";
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, 1, 2);
    NSUInteger shapes = COUNT(SHAPES) + COUNT(MIXED);
    for (size_t f = 0; f < COUNT(FORMATS); f++) {
        for (size_t k = 0; k < COUNT(KINDS); k++) {
            for (size_t s = 0; s < shapes; s++) {
                if (s >= COUNT(SHAPES) && k > 1) {
                    break;
                }
                const char *shape = s < COUNT(SHAPES) ? SHAPES[s] : MIXED[s - COUNT(SHAPES)];
                char name[64];
                snprintf(name, sizeof name, "%s-%s-%s", FORMATS[f], KINDS[k], shape);
                for (int on = 1; on >= 0; on--) {
                    char *argv[] = {(char *)self, name, NULL};
                    pid_t pid;
                    fflush(stdout);
                    if (posix_spawn(&pid, self, &actions, NULL, argv, on ? layered : plain) != 0) {
                        printf("END %s %s spawn failed\n", name, on ? "layer" : "plain");
                        continue;
                    }
                    int status = 0;
                    waitpid(pid, &status, 0);
                    if (WIFEXITED(status)) {
                        printf("END %s %s exit %d\n", name, on ? "layer" : "plain",
                               WEXITSTATUS(status));
                    } else {
                        printf("END %s %s signal %d\n", name, on ? "layer" : "plain",
                               WTERMSIG(status));
                    }
                }
            }
        }
    }
    return 0;
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        setvbuf(stdout, NULL, _IONBF, 0);
        if (argc != 2) {
            printf("usage: compressed_copies_of_small_levels CASE | --all\n");
            return 2;
        }
        if (strcmp(argv[1], "--all") == 0) {
            return run_all(argv[0]);
        }
        alarm(30);
        return run_case(argv[1]);
    }
}
