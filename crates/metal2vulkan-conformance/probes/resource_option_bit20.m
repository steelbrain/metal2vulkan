// What Metal does with bit 20 (0x100000) of MTLResourceOptions, a bit no public option names: the
// options a guest's OpenCL runtime passes for its buffers (0x100010, `Managed` storage and bit 20,
// over its own page-aligned pages with newBufferWithBytesNoCopy:, and 0x100000 for host-pointer
// memory). For each way of making a resource it prints whether Metal made it, and what the resource
// reports of its options, one line each:
//
//   BIT20 <what> <options asked, hex> made=<0|1> options=<resourceOptions, hex> storage=<n>
//     cache=<n> hazard=<n>
//
// WHAT IT MEASURED, plain and under API validation, an Apple M3 Max under macOS 27.0.1 and an AMD
// Radeon Pro 5300M under macOS 26.7.1 alike where both ran:
//
//   - newBufferWithLength: (0x100000, 0x100010, 0x100020), newBufferWithBytes: and
//     newBufferWithBytesNoCopy: (0x100000, 0x100010) are made, and report their options WITHOUT
//     bit 20: 0x200, 0x210 and 0x220 (the hazard tracking mode resolved to tracked);
//   - a heap with it (the M3 Max: `Shared` and `Private`; the 5300M refuses a `Shared` heap
//     whatever the bit) is made and reports its options without it; a heap's newBufferWithLength:
//     with it is made plainly, reporting without it, and stopped by API validation ("unknown bits
//     0x100000");
//   - a texture descriptor keeps it (`resourceOptions` reads 0x100000), and newTextureWithDescriptor:
//     with it ends the process plainly (IOGPUMetalSuballocator: "Different MTLResourceOptions used
//     for the same heap index").
//
// So `ResourceOptions::from_raw` takes it, a buffer and a heap are made as without it, and a
// texture descriptor with it is refused (`ResourceOptions::BIT_20`).
//
// Run it twice, plain and under API validation (a refusal there ends the process, so the line
// before it says how far it got):
//
//   clang -fobjc-arc -framework Metal -framework Foundation -o resource_option_bit20 \
//     resource_option_bit20.m
//   ./resource_option_bit20
//   MTL_DEBUG_LAYER=1 ./resource_option_bit20

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>

static const MTLResourceOptions kBit20 = 0x100000;

static void report(const char *what, MTLResourceOptions asked, id<MTLResource> resource) {
  if (resource == nil) {
    printf("BIT20 %s 0x%llx made=0\n", what, (unsigned long long)asked);
  } else {
    printf("BIT20 %s 0x%llx made=1 options=0x%llx storage=%lu cache=%lu hazard=%lu\n", what,
           (unsigned long long)asked, (unsigned long long)resource.resourceOptions,
           (unsigned long)resource.storageMode, (unsigned long)resource.cpuCacheMode,
           (unsigned long)resource.hazardTrackingMode);
  }
  fflush(stdout);
}

int main(void) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    printf("BIT20 device %s unified=%d\n", device.name.UTF8String, device.hasUnifiedMemory);
    const MTLResourceOptions storages[] = {MTLResourceStorageModeShared,
                                           MTLResourceStorageModeManaged,
                                           MTLResourceStorageModePrivate};
    const size_t page = (size_t)getpagesize();
    const size_t length = 4 * page;
    for (size_t i = 0; i < sizeof storages / sizeof storages[0]; i++) {
      MTLResourceOptions options = storages[i] | kBit20;
      report("length", options, [device newBufferWithLength:length options:options]);
    }
    char bytes[64] = {0};
    for (size_t i = 0; i < 2; i++) {
      MTLResourceOptions options = storages[i] | kBit20;
      report("bytes", options, [device newBufferWithBytes:bytes length:sizeof bytes options:options]);
    }
    for (size_t i = 0; i < 2; i++) {
      MTLResourceOptions options = storages[i] | kBit20;
      void *pages = mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      report("nocopy", options,
             [device newBufferWithBytesNoCopy:pages
                                       length:length
                                      options:options
                                  deallocator:^(void *pointer, NSUInteger size) {
                                    munmap(pointer, size);
                                  }]);
    }
    for (size_t i = 0; i < 3; i += 2) {
      MTLHeapDescriptor *descriptor = [MTLHeapDescriptor new];
      descriptor.size = 1 << 20;
      descriptor.resourceOptions = storages[i] | kBit20;
      id<MTLHeap> heap = [device newHeapWithDescriptor:descriptor];
      if (heap == nil) {
        printf("BIT20 heap 0x%llx made=0\n", (unsigned long long)descriptor.resourceOptions);
        continue;
      }
      printf("BIT20 heap 0x%llx made=1 options=0x%llx storage=%lu\n",
             (unsigned long long)descriptor.resourceOptions,
             (unsigned long long)heap.resourceOptions, (unsigned long)heap.storageMode);
      report("heap-buffer", storages[i] | kBit20,
             [heap newBufferWithLength:page options:storages[i] | kBit20]);
      report("heap-buffer-without", storages[i], [heap newBufferWithLength:page options:storages[i]]);
    }
    // Last: a texture descriptor's validation may end the process.
    for (size_t i = 0; i < 3; i++) {
      MTLTextureDescriptor *descriptor =
          [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                             width:16
                                                            height:16
                                                         mipmapped:NO];
      descriptor.resourceOptions = storages[i] | kBit20;
      printf("BIT20 texture-descriptor 0x%llx reads=0x%llx\n",
             (unsigned long long)(storages[i] | kBit20),
             (unsigned long long)descriptor.resourceOptions);
      fflush(stdout);
      report("texture", storages[i] | kBit20, [device newTextureWithDescriptor:descriptor]);
    }
  }
  return 0;
}
