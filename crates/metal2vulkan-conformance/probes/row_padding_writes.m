// Metal's row padding: whether a GPU write of a linear texture's texels changes the bytes between
// a row's last texel and the next row (bytesPerRow past width * 4), over a buffer's bytes and over
// an IOSurface's plane.
//
// On every device, a BGRA8Unorm texture 3662 or 61 texels wide and 64 rows high (each row's
// texels ending inside a 64-byte line), at a bytesPerRow past its texels (a multiple of the
// device's minimumLinearTextureAlignmentForPixelFormat, or of 64 for an IOSurface), takes three
// writes, each in 100 command buffers one after another: a render pass that clears and stores it,
// a pass that loads it, draws one triangle over all of it and stores it, and a kernel that writes
// every texel.
//
// - Unified memory (a Shared buffer, an IOSurface): a CPU thread writes every padding byte over
//   and over while the command buffers run, reading each byte before it writes it, which must
//   still hold what its last write left there: a GPU write that read a padding byte and wrote it
//   back puts an older value over a CPU write made in between, and the thread counts the bytes
//   so found changed. Every padding byte must hold the thread's last write at the end.
// - Discrete memory (a Managed buffer and surface): the padding filled with a pattern
//   (didModifyRange), the writes, a blit's synchronize, and every padding byte must hold the
//   pattern.
//
// Prints `ROW-PADDING <device> <form> <width>x<rows> bpr <n> <write>: ...` per case, with the
// padding bytes of the last row (past its texels, to bytesPerRow, inside the buffer) apart.
//
// What it found, 100 rounds of each write in each case:
// - Apple M3 Max, macOS 27.0.1, arm64 and x86_64: no padding byte changed under any write, over a
//   buffer or an IOSurface.
// - Intel UHD Graphics 630, macOS 26.7.1 (unified memory, linear texture alignment 256): over a
//   buffer, a kernel's writes put older values back over the CPU's writes to the padding, 10,197
//   bytes found changed at 3662x64 (bpr 14848), 304 of them in the last row, and 28,162 at 61x64
//   (bpr 256), 307 in the last row; its clears and stores and its draws over a buffer, and every
//   write over an IOSurface, changed none.
// - AMD Radeon Pro 5300M, macOS 26.7.1 (discrete memory, the padding read once each write
//   completed): no padding byte changed under any write.
//
// clang -fobjc-arc -arch arm64 -arch x86_64 -framework Metal -framework Foundation \
//     -framework IOSurface -Wno-deprecated-declarations -o row_padding_writes row_padding_writes.m

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#include <stdatomic.h>

static NSString *const kSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "struct V { float4 p [[position]]; };\n"
     "vertex V full(uint vid [[vertex_id]]) {\n"
     "    float2 xy = float2((vid << 1) & 2, vid & 2);\n"
     "    V v; v.p = float4(xy * 2.0 - 1.0, 0.0, 1.0); return v;\n"
     "}\n"
     "fragment float4 fill(V in [[stage_in]], constant float4 &c [[buffer(0)]]) { return c; }\n"
     "kernel void store(texture2d<float, access::write> t [[texture(0)]],\n"
     "                  constant float4 &c [[buffer(0)]],\n"
     "                  uint2 gid [[thread_position_in_grid]]) {\n"
     "    if (gid.x < t.get_width() && gid.y < t.get_height()) { t.write(c, gid); }\n"
     "}\n";

static const NSUInteger kRows = 64;
static const NSUInteger kRounds = 100;

typedef enum { WriteClear, WriteDraw, WriteKernel } Write;
static const char *const kWriteNames[] = {"clear and store", "draw", "kernel"};

// The CPU thread's view of the padding: `rows` rows `bpr` apart, each past its `texels` bytes.
typedef struct {
    uint8_t *base;
    NSUInteger rows;
    NSUInteger bpr;
    NSUInteger texels;
    atomic_bool stop;
    uint64_t sweeps;
    uint64_t changed;
    uint64_t last_row;
    long first_row;
    long first_byte;
    int first_due;
    int first_read;
} Padding;

// The value sweep `s` writes at padding byte `b` of row `r`: each sweep's differs from the last's.
static uint8_t valueAt(uint64_t s, NSUInteger r, NSUInteger b) {
    return (uint8_t)(s + r * 7 + b * 13);
}

// Read each padding byte, which must hold the last sweep's value, and write this sweep's.
static void sweep(Padding *p, uint64_t s, bool write) {
    for (NSUInteger r = 0; r < p->rows; r += 1) {
        volatile uint8_t *row = p->base + r * p->bpr;
        for (NSUInteger b = p->texels; b < p->bpr; b += 1) {
            uint8_t due = valueAt(s - 1, r, b);
            uint8_t read = row[b];
            if (read != due) {
                p->last_row += r + 1 == p->rows;
                if (p->changed == 0) {
                    p->first_row = (long)r;
                    p->first_byte = (long)b;
                    p->first_due = due;
                    p->first_read = read;
                }
                p->changed += 1;
            }
            if (write) { row[b] = valueAt(s, r, b); }
        }
    }
}

static void fillPadding(uint8_t *base, NSUInteger rows, NSUInteger bpr, NSUInteger texels,
                        uint64_t s) {
    for (NSUInteger r = 0; r < rows; r += 1) {
        for (NSUInteger b = texels; b < bpr; b += 1) { base[r * bpr + b] = valueAt(s, r, b); }
    }
}

typedef struct {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLRenderPipelineState> draw;
    id<MTLComputePipelineState> store;
} Gpu;

static id<MTLCommandBuffer> encode(Gpu *g, id<MTLTexture> t, Write write, NSUInteger round) {
    id<MTLCommandBuffer> cb = [g->queue commandBuffer];
    float c[4] = {(float)(round % 7) / 7.0f, 0.25f, 0.5f, 1.0f};
    if (write == WriteKernel) {
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:g->store];
        [e setTexture:t atIndex:0];
        [e setBytes:c length:sizeof c atIndex:0];
        [e dispatchThreads:MTLSizeMake(t.width, t.height, 1)
            threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [e endEncoding];
        return cb;
    }
    MTLRenderPassDescriptor *d = [MTLRenderPassDescriptor renderPassDescriptor];
    d.colorAttachments[0].texture = t;
    d.colorAttachments[0].loadAction = write == WriteClear ? MTLLoadActionClear : MTLLoadActionLoad;
    d.colorAttachments[0].clearColor = MTLClearColorMake(c[0], c[1], c[2], c[3]);
    d.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:d];
    if (write == WriteDraw) {
        [e setRenderPipelineState:g->draw];
        [e setFragmentBytes:c length:sizeof c atIndex:0];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }
    [e endEncoding];
    return cb;
}

// How a case's CPU and GPU share the bytes: at once (unified memory), or through a synchronize
// of `sync` (a Managed buffer) or the surface's lock (`modified` after the CPU's writes,
// `refresh` before its reads, `done` after them).
typedef struct {
    bool unified;
    id<MTLResource> sync;
    void (^modified)(void);
    void (^refresh)(void);
    void (^done)(void);
} Sharing;

// One case: `t` over `base`, its rows `bpr` apart, each `t.width * 4` bytes of texels.
static void runCase(Gpu *g, const char *form, id<MTLTexture> t, uint8_t *base, NSUInteger bpr,
                    Sharing sharing) {
    NSUInteger texels = t.width * 4;
    for (int w = WriteClear; w <= WriteKernel; w += 1) {
        Write write = (Write)w;
        if (sharing.unified) {
            __block Padding p = {.base = base, .rows = kRows, .bpr = bpr, .texels = texels};
            atomic_init(&p.stop, false);
            fillPadding(base, kRows, bpr, texels, 0);
            dispatch_semaphore_t done = dispatch_semaphore_create(0);
            dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
                uint64_t s = 1;
                while (!atomic_load(&p.stop)) { sweep(&p, s, true); s += 1; }
                sweep(&p, s, false);
                p.sweeps = s - 1;
                dispatch_semaphore_signal(done);
            });
            id<MTLCommandBuffer> last = nil;
            for (NSUInteger round = 0; round < kRounds; round += 1) {
                last = encode(g, t, write, round);
                [last commit];
                [last waitUntilCompleted];
            }
            atomic_store(&p.stop, true);
            dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
            printf("ROW-PADDING %s %s %lux%lu bpr %lu %s: %lu rounds, %llu sweeps, padding bytes "
                   "found changed %llu",
                   g->device.name.UTF8String, form, (unsigned long)t.width, (unsigned long)kRows,
                   (unsigned long)bpr, kWriteNames[w], (unsigned long)kRounds,
                   (unsigned long long)p.sweeps, (unsigned long long)p.changed);
            printf(", of them in the last row %llu", (unsigned long long)p.last_row);
            if (p.changed != 0) {
                printf(" (first: row %ld byte %ld held 0x%02x where 0x%02x was due)", p.first_row,
                       p.first_byte, p.first_read, p.first_due);
            }
            printf("%s\n", last.error ? " [command buffer error]" : "");
            continue;
        }
        // Discrete memory: the pattern, the writes, the synchronize, the check.
        fillPadding(base, kRows, bpr, texels, 5);
        sharing.modified();
        for (NSUInteger round = 0; round < kRounds; round += 1) {
            id<MTLCommandBuffer> cb = encode(g, t, write, round);
            if (round + 1 == kRounds && sharing.sync != nil) {
                id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
                [b synchronizeResource:sharing.sync];
                [b endEncoding];
            }
            [cb commit];
            [cb waitUntilCompleted];
        }
        sharing.refresh();
        uint64_t changed = 0;
        uint64_t last_row = 0;
        for (NSUInteger r = 0; r < kRows; r += 1) {
            for (NSUInteger b = texels; b < bpr; b += 1) {
                bool differs = base[r * bpr + b] != valueAt(5, r, b);
                changed += differs;
                last_row += differs && r + 1 == kRows;
            }
        }
        printf("ROW-PADDING %s %s %lux%lu bpr %lu %s: %lu rounds, padding bytes changed %llu, of "
               "them in the last row %llu (synchronized)\n",
               g->device.name.UTF8String, form, (unsigned long)t.width, (unsigned long)kRows,
               (unsigned long)bpr, kWriteNames[w], (unsigned long)kRounds,
               (unsigned long long)changed, (unsigned long long)last_row);
        sharing.done();
    }
}

static MTLTextureDescriptor *descriptor(NSUInteger width, MTLStorageMode mode) {
    MTLTextureDescriptor *d =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                           width:width
                                                          height:kRows
                                                       mipmapped:NO];
    d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    d.storageMode = mode;
    return d;
}

static void onDevice(id<MTLDevice> device) {
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:kSource options:nil error:&error];
    if (library == nil) {
        printf("ROW-PADDING %s: no library: %s\n", device.name.UTF8String,
               error.localizedDescription.UTF8String);
        return;
    }
    MTLRenderPipelineDescriptor *rd = [MTLRenderPipelineDescriptor new];
    rd.vertexFunction = [library newFunctionWithName:@"full"];
    rd.fragmentFunction = [library newFunctionWithName:@"fill"];
    rd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    Gpu g = {
        .device = device,
        .queue = [device newCommandQueue],
        .draw = [device newRenderPipelineStateWithDescriptor:rd error:&error],
        .store = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"store"]
                                                        error:&error],
    };
    if (g.draw == nil || g.store == nil) {
        printf("ROW-PADDING %s: no pipeline: %s\n", device.name.UTF8String,
               error.localizedDescription.UTF8String);
        return;
    }
    bool unified = device.hasUnifiedMemory;
    MTLStorageMode mode = unified ? MTLStorageModeShared : MTLStorageModeManaged;
    NSUInteger align =
        [device minimumLinearTextureAlignmentForPixelFormat:MTLPixelFormatBGRA8Unorm];
    printf("ROW-PADDING %s: unified memory %d, linear texture alignment %lu\n",
           device.name.UTF8String, unified, (unsigned long)align);
    for (NSUInteger i = 0; i < 2; i += 1) {
        NSUInteger width = i == 0 ? 3662 : 61;
        NSUInteger texels = width * 4;
        // A buffer's texture: the smallest multiple of the alignment past the texels.
        NSUInteger bpr = (texels / align + 1) * align;
        MTLResourceOptions options = unified ? MTLResourceStorageModeShared
                                             : MTLResourceStorageModeManaged;
        id<MTLBuffer> buffer = [device newBufferWithLength:bpr * kRows options:options];
        id<MTLTexture> t = [buffer newTextureWithDescriptor:descriptor(width, mode)
                                                     offset:0
                                                bytesPerRow:bpr];
        if (t == nil) {
            printf("ROW-PADDING %s buffer %lu: no texture\n", device.name.UTF8String,
                   (unsigned long)width);
        } else {
            Sharing sharing = {
                .unified = unified,
                .sync = buffer,
                .modified = ^{ [buffer didModifyRange:NSMakeRange(0, bpr * kRows)]; },
                .refresh = ^{},
                .done = ^{},
            };
            runCase(&g, "buffer", t, buffer.contents, bpr, sharing);
        }
        // An IOSurface's plane: the smallest multiple of 64 past the texels.
        NSUInteger surfaceBpr = (texels / 64 + 1) * 64;
        NSDictionary *properties = @{
            (id)kIOSurfaceWidth : @(width),
            (id)kIOSurfaceHeight : @(kRows),
            (id)kIOSurfaceBytesPerElement : @4,
            (id)kIOSurfaceBytesPerRow : @(surfaceBpr),
            (id)kIOSurfacePixelFormat : @(0x42475241u), // 'BGRA'
        };
        IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)properties);
        if (surface == NULL) {
            printf("ROW-PADDING %s surface %lu: no IOSurface\n", device.name.UTF8String,
                   (unsigned long)width);
            continue;
        }
        NSUInteger made = IOSurfaceGetBytesPerRow(surface);
        id<MTLTexture> st = [device newTextureWithDescriptor:descriptor(width, mode)
                                                   iosurface:surface
                                                       plane:0];
        if (st == nil) {
            printf("ROW-PADDING %s surface %lu: no texture\n", device.name.UTF8String,
                   (unsigned long)width);
        } else {
            // Unified memory: locked once, the CPU thread writing while the GPU writes. Discrete:
            // locked for each CPU access, so the unlock after the pattern hands it to the GPU and
            // a read-only lock after the writes brings the GPU's bytes back.
            IOSurfaceLock(surface, 0, NULL);
            uint8_t *base = IOSurfaceGetBaseAddress(surface);
            Sharing sharing = {
                .unified = unified,
                .sync = nil,
                .modified = ^{ IOSurfaceUnlock(surface, 0, NULL); },
                .refresh = ^{ IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL); },
                .done = ^{
                    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
                    IOSurfaceLock(surface, 0, NULL);
                },
            };
            runCase(&g, "iosurface", st, base, made, sharing);
            IOSurfaceUnlock(surface, 0, NULL);
        }
        CFRelease(surface);
    }
}

int main(void) {
    @autoreleasepool {
        for (id<MTLDevice> device in MTLCopyAllDevices()) { onDevice(device); }
    }
    return 0;
}
