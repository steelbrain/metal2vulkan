// What Metal does with a `copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:` whose
// offsets or size are not multiples of 4.
//
// Apple's page for the call says that on macOS both offsets and the size are multiples of 4. This
// asks what a shipped program sees when it breaks that rule: whether the GPU copies every byte of
// the range, rounds the range, drops the copy, or does something else.
//
// ONE COPY PER PROCESS, chosen on the command line:
//
//   blit_buffer_copy_unaligned DEVICE STORAGE SOURCE-OFFSET DESTINATION-OFFSET SIZE
//
// DEVICE is a piece of the device's name ("M3", "5300M", "UHD"); every device whose name holds it
// runs the copy. STORAGE is the two buffers' storage mode: shared, managed or private. The source
// holds byte i = i % 127 (0x00 to 0x7e), the destination starts as byte j = 0x80 | (j % 127)
// (0x80 to 0xfe), so a destination byte below 0x80 is one a copy wrote, and each buffer runs 64
// bytes past its range, so a size rounded up shows. A private pair is filled from, and read back
// through, Shared staging buffers by copies whose offsets and sizes are multiples of 4; a managed
// pair is read back after `synchronizeResource:`. Each process prints one ROW line per device:
// the command buffer's status and error, the bytes in the range that hold the source byte, that
// still hold the starting byte or hold neither, the destination bytes outside the range that
// changed, whether the source changed, and a verdict:
//
//   exact     every byte of the range copied, no byte outside it touched
//   dropped   no destination byte changed
//   rounded   the changed bytes are one run, each the source byte the call maps it to, but the
//             run is not the range
//   other     anything else, with the first bytes that differ
//
// With MTL_DEBUG_LAYER=1 the validation layer's default error mode ends the process at a call
// it does not take, so a process that prints no ROW line and exits on a signal was stopped there
// (its standard error holds the layer's words).
//
//   xcrun clang -arch arm64 -arch x86_64 -mmacosx-version-min=13.0 -O1 -fobjc-arc -Wall -Werror \
//       -framework Metal -framework Foundation \
//       -o blit_buffer_copy_unaligned blit_buffer_copy_unaligned.m
//   for c in "0 0 62302" "0 0 1" "0 0 3" "1 0 8" "0 2 8" "3 5 4097" "0 0 64" "4 8 4096"; do
//       /usr/bin/env -u MTL_DEBUG_LAYER ./blit_buffer_copy_unaligned M3 shared $c; done
//
// WHAT IT MEASURED, those eight copies in each storage mode, one binary (built in job m3-max-69612):
// every one exact, its command buffer completed with no error. On an Apple M3 Max, macOS 27.0.1,
// job m3-max-69613, plainly and under MTL_DEBUG_LAYER=1, which took every copy without a word. On
// an AMD Radeon Pro 5300M and an Intel UHD Graphics 630, macOS 26.7.1, job m3-max-69614, plainly.
// The rt crate's `BlitCommandEncoder::copy_from_buffer_to_buffer` copies as they do.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/// Bytes each buffer runs past its range, so a size rounded up shows.
static const NSUInteger PAD = 64;

static uint8_t source_byte(NSUInteger i) { return (uint8_t)(i % 127); }
static uint8_t start_byte(NSUInteger j) { return (uint8_t)(0x80 | (j % 127)); }
static NSUInteger round4(NSUInteger n) { return (n + 3) & ~(NSUInteger)3; }

static const char *status_name(MTLCommandBufferStatus status) {
    switch (status) {
    case MTLCommandBufferStatusNotEnqueued: return "not-enqueued";
    case MTLCommandBufferStatusEnqueued: return "enqueued";
    case MTLCommandBufferStatusCommitted: return "committed";
    case MTLCommandBufferStatusScheduled: return "scheduled";
    case MTLCommandBufferStatusCompleted: return "completed";
    case MTLCommandBufferStatusError: return "error";
    }
    return "unknown";
}

/// Commits `cb` and waits at most ten seconds for it; exits 3 if it does not complete.
static void run(id<MTLCommandBuffer> cb, const char *what) {
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    [cb addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull b) {
        (void)b;
        dispatch_semaphore_signal(done);
    }];
    [cb commit];
    if (dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC)) != 0) {
        printf("TIMEOUT %s: status %s\n", what, status_name(cb.status));
        fflush(stdout);
        exit(3);
    }
}

static NSString *error_text(id<MTLCommandBuffer> cb) {
    if (!cb.error) return @"none";
    return [NSString stringWithFormat:@"%@/%ld \"%@\"", cb.error.domain, (long)cb.error.code,
                                      cb.error.localizedDescription];
}

/// One copy on `device`, printed as one ROW line.
static void probe(id<MTLDevice> device, const char *storage, NSUInteger src, NSUInteger dst,
                  NSUInteger size) {
    NSUInteger source_length = round4(src + size + PAD);
    NSUInteger destination_length = round4(dst + size + PAD);
    uint8_t *source = malloc(source_length);
    uint8_t *start = malloc(destination_length);
    for (NSUInteger i = 0; i < source_length; i++) source[i] = source_byte(i);
    for (NSUInteger j = 0; j < destination_length; j++) start[j] = start_byte(j);

    BOOL private_ = strcmp(storage, "private") == 0;
    BOOL managed = strcmp(storage, "managed") == 0;
    MTLResourceOptions shared_options = MTLResourceStorageModeShared;
    MTLResourceOptions options = private_ ? MTLResourceStorageModePrivate
                                 : managed ? MTLResourceStorageModeManaged
                                           : MTLResourceStorageModeShared;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLBuffer> from, to, from_staging = nil, to_staging = nil;
    if (private_) {
        from_staging = [device newBufferWithBytes:source length:source_length options:shared_options];
        to_staging = [device newBufferWithBytes:start length:destination_length options:shared_options];
        from = [device newBufferWithLength:source_length options:options];
        to = [device newBufferWithLength:destination_length options:options];
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:from_staging sourceOffset:0 toBuffer:from destinationOffset:0 size:source_length];
        [blit copyFromBuffer:to_staging sourceOffset:0 toBuffer:to destinationOffset:0 size:destination_length];
        [blit endEncoding];
        run(cb, "fill");
        if (cb.status != MTLCommandBufferStatusCompleted) {
            printf("FILL-FAILED status %s error %s\n", status_name(cb.status), error_text(cb).UTF8String);
            exit(1);
        }
    } else {
        from = [device newBufferWithBytes:source length:source_length options:options];
        to = [device newBufferWithBytes:start length:destination_length options:options];
    }

    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:from sourceOffset:src toBuffer:to destinationOffset:dst size:size];
    [blit endEncoding];
    run(cb, "copy");
    MTLCommandBufferStatus status = cb.status;
    NSString *error = error_text(cb);

    // Read both buffers back.
    const uint8_t *got_source, *got;
    if (private_) {
        id<MTLCommandBuffer> back = [queue commandBuffer];
        id<MTLBlitCommandEncoder> b = [back blitCommandEncoder];
        [b copyFromBuffer:from sourceOffset:0 toBuffer:from_staging destinationOffset:0 size:source_length];
        [b copyFromBuffer:to sourceOffset:0 toBuffer:to_staging destinationOffset:0 size:destination_length];
        [b endEncoding];
        run(back, "read-back");
        got_source = from_staging.contents;
        got = to_staging.contents;
    } else {
        if (managed) {
            id<MTLCommandBuffer> back = [queue commandBuffer];
            id<MTLBlitCommandEncoder> b = [back blitCommandEncoder];
            [b synchronizeResource:from];
            [b synchronizeResource:to];
            [b endEncoding];
            run(back, "synchronize");
        }
        got_source = from.contents;
        got = to.contents;
    }

    NSUInteger copied = 0, untouched = 0, neither = 0, outside = 0;
    NSUInteger first_changed = NSUIntegerMax, last_changed = 0;
    NSUInteger first_outside = NSUIntegerMax, last_outside = 0;
    BOOL mapped = YES;  // every changed byte holds the source byte the call maps it to
    char differ[512] = "";
    int shown = 0;
    for (NSUInteger j = 0; j < destination_length; j++) {
        BOOL inside = j >= dst && j < dst + size;
        BOOL changed = got[j] != start[j];
        long from_index = (long)j - (long)dst + (long)src;
        BOOL holds_source =
            from_index >= 0 && (NSUInteger)from_index < source_length && got[j] == source[from_index];
        if (changed) {
            if (first_changed == NSUIntegerMax) first_changed = j;
            last_changed = j;
            if (!holds_source) mapped = NO;
        }
        if (inside) {
            if (holds_source) copied++;
            else if (!changed) untouched++;
            else neither++;
        } else if (changed) {
            outside++;
            if (first_outside == NSUIntegerMax) first_outside = j;
            last_outside = j;
        }
        uint8_t expected = inside ? source[from_index] : start[j];
        if (got[j] != expected && shown < 6) {
            size_t used = strlen(differ);
            snprintf(differ + used, sizeof differ - used, "%s%lu:%02x(want %02x)", shown ? "," : "",
                     (unsigned long)j, got[j], expected);
            shown++;
        }
    }
    NSUInteger source_moved = 0;
    for (NSUInteger i = 0; i < source_length; i++) source_moved += got_source[i] != source[i];

    NSUInteger changed_count = 0;
    for (NSUInteger j = 0; j < destination_length; j++) changed_count += got[j] != start[j];
    const char *verdict;
    char detail[160] = "";
    if (copied == size && outside == 0) {
        verdict = "exact";
    } else if (changed_count == 0) {
        verdict = "dropped";
    } else if (mapped && changed_count == last_changed - first_changed + 1) {
        verdict = "rounded";
        snprintf(detail, sizeof detail, " run=[%lu,%lu)", (unsigned long)first_changed,
                 (unsigned long)last_changed + 1);
    } else {
        verdict = "other";
        snprintf(detail, sizeof detail, " changed=%lu first=%lu last=%lu mapped=%d",
                 (unsigned long)changed_count, (unsigned long)first_changed,
                 (unsigned long)last_changed, mapped);
    }
    printf("ROW device=\"%s\" class=%s storage=%s case=%lu/%lu/%lu status=%s error=%s "
           "in-range: copied=%lu untouched=%lu neither=%lu outside-changed=%lu",
           device.name.UTF8String, NSStringFromClass([device class]).UTF8String, storage,
           (unsigned long)src, (unsigned long)dst, (unsigned long)size, status_name(status),
           error.UTF8String, (unsigned long)copied, (unsigned long)untouched,
           (unsigned long)neither, (unsigned long)outside);
    if (outside) printf("[%lu..%lu]", (unsigned long)first_outside, (unsigned long)last_outside);
    printf(" source-changed=%lu verdict=%s%s", (unsigned long)source_moved, verdict, detail);
    if (shown) printf(" differ=%s", differ);
    printf("\n");
    fflush(stdout);
    free(source);
    free(start);
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc != 6) {
            printf("usage: blit_buffer_copy_unaligned DEVICE STORAGE SOURCE-OFFSET "
                   "DESTINATION-OFFSET SIZE\n");
            return 2;
        }
        const char *storage = argv[2];
        if (strcmp(storage, "shared") && strcmp(storage, "managed") && strcmp(storage, "private")) {
            printf("no storage %s\n", storage);
            return 2;
        }
        NSUInteger src = strtoul(argv[3], NULL, 10), dst = strtoul(argv[4], NULL, 10),
                   size = strtoul(argv[5], NULL, 10);
        NSString *want = [NSString stringWithUTF8String:argv[1]];
        int ran = 0;
        for (id<MTLDevice> device in MTLCopyAllDevices()) {
            if ([device.name rangeOfString:want options:NSCaseInsensitiveSearch].location == NSNotFound)
                continue;
            probe(device, storage, src, dst, size);
            ran++;
        }
        if (!ran) {
            printf("no device named like %s\n", argv[1]);
            return 2;
        }
    }
    return 0;
}
