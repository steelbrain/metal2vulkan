// What Metal's argument-buffer vocabulary answers for the `MTLDataType` values no case reaches,
// and what `newArgumentEncoderForBufferAtIndex:` answers for a member that is and is not a nested
// argument buffer.
//
// Four questions the conformance suite cannot ask in a case, because each may end the process:
//
//   descriptor <MTLDataType ordinal>   -[MTLDevice newArgumentEncoderWithArguments:] over one
//                                      MTLArgumentDescriptor of that dataType. MTLDataTypeNone (0)
//                                      and MTLDataTypeDepthStencilState (33) are the two values a
//                                      descriptor is the only candidate route to; MTLDataTypeTensor
//                                      (140) is Metal 4's.
//   nested <buffer index> <argument index>
//                                      -[MTLArgumentEncoder newArgumentEncoderForBufferAtIndex:] of
//                                      the fixture kernel's own layout, at a member that is a
//                                      nested argument buffer and at one that is not.
//   function <MTLFunctionOptions bits>  -[MTLLibrary newFunctionWithDescriptor:error:] under option
//                                      bits no case passes, with no binary archive listed.
//
// ONE SHAPE PER PROCESS, chosen on the command line: Metal's own validation of a descriptor ends
// the process on SIGABRT, and a process that died says nothing about the shapes after the one that
// killed it. A fatal answer is a legitimate measurement here; it is why this is a probe and not a
// conformance case.
//
// The metallib path is an argument so the probe needs nothing of this tree's layout.
//
//   clang -fobjc-arc -framework Metal -framework Foundation -o /tmp/adt argument_descriptor_types.m
//   /tmp/adt descriptor 33
//   /tmp/adt nested /path/to/argument_buffers.metallib reflection_argument_buffers 0 15
//   /tmp/adt function /path/to/argument_buffers.metallib reflection_argument_buffers 1

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static id<MTLDevice> open_device(void) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        fprintf(stderr, "no Metal device\n");
        exit(2);
    }
    return device;
}

static id<MTLLibrary> load(id<MTLDevice> device, const char *path) {
    NSError *error = nil;
    NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
    id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
    if (library == nil) {
        printf("library=nil error=%s\n", error.localizedDescription.UTF8String);
        exit(3);
    }
    return library;
}

int main(int argc, const char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s descriptor <dataType> | nested <metallib> <entry> <buffer> <arg>"
                        " | function <metallib> <entry> <options>\n",
                argv[0]);
        return 2;
    }
    @autoreleasepool {
        id<MTLDevice> device = open_device();
        if (strcmp(argv[1], "descriptor") == 0 && argc == 3) {
            NSUInteger type = (NSUInteger)strtoul(argv[2], NULL, 0);
            MTLArgumentDescriptor *one = [MTLArgumentDescriptor argumentDescriptor];
            one.dataType = (MTLDataType)type;
            one.index = 0;
            one.access = MTLBindingAccessReadOnly;
            printf("dataType=%lu descriptor-made\n", (unsigned long)type);
            fflush(stdout);
            id<MTLArgumentEncoder> encoder =
                [device newArgumentEncoderWithArguments:@[ one ]];
            if (encoder == nil) {
                printf("dataType=%lu encoder=nil\n", (unsigned long)type);
            } else {
                printf("dataType=%lu encodedLength=%lu alignment=%lu\n", (unsigned long)type,
                       (unsigned long)encoder.encodedLength, (unsigned long)encoder.alignment);
            }
            return 0;
        }
        if (strcmp(argv[1], "nested") == 0 && argc == 6) {
            id<MTLLibrary> library = load(device, argv[2]);
            id<MTLFunction> function =
                [library newFunctionWithName:[NSString stringWithUTF8String:argv[3]]];
            if (function == nil) {
                printf("function=nil\n");
                return 3;
            }
            NSUInteger buffer = (NSUInteger)strtoul(argv[4], NULL, 0);
            NSUInteger index = (NSUInteger)strtoul(argv[5], NULL, 0);
            id<MTLArgumentEncoder> outer = [function newArgumentEncoderWithBufferIndex:buffer];
            if (outer == nil) {
                printf("outer=nil\n");
                return 3;
            }
            printf("outer encodedLength=%lu alignment=%lu\n",
                   (unsigned long)outer.encodedLength, (unsigned long)outer.alignment);
            fflush(stdout);
            id<MTLArgumentEncoder> nested = [outer newArgumentEncoderForBufferAtIndex:index];
            if (nested == nil) {
                printf("index=%lu nested=nil\n", (unsigned long)index);
            } else {
                printf("index=%lu nested encodedLength=%lu alignment=%lu\n",
                       (unsigned long)index, (unsigned long)nested.encodedLength,
                       (unsigned long)nested.alignment);
            }
            return 0;
        }
        if (strcmp(argv[1], "function") == 0 && argc == 5) {
            id<MTLLibrary> library = load(device, argv[2]);
            MTLFunctionDescriptor *made = [MTLFunctionDescriptor functionDescriptor];
            made.name = [NSString stringWithUTF8String:argv[3]];
            made.options = (MTLFunctionOptions)strtoul(argv[4], NULL, 0);
            printf("options=%#lx descriptor-made\n", (unsigned long)made.options);
            fflush(stdout);
            NSError *error = nil;
            id<MTLFunction> function = [library newFunctionWithDescriptor:made error:&error];
            if (function == nil) {
                printf("options=%#lx function=nil error=%s\n", (unsigned long)made.options,
                       error.localizedDescription.UTF8String);
            } else {
                printf("options=%#lx options-back=%#lx name=%s type=%lu\n",
                       (unsigned long)made.options, (unsigned long)function.options,
                       function.name.UTF8String, (unsigned long)function.functionType);
            }
            return 0;
        }
        if (strcmp(argv[1], "layout") == 0 && argc >= 3) {
            // "type[:array[:blockAlign]],..." as one argument buffer's descriptors, in order.
            NSMutableArray *descriptors = [NSMutableArray array];
            char *spec = strdup(argv[2]);
            NSUInteger index = 0;
            for (char *item = strtok(spec, ","); item != NULL; item = strtok(NULL, ",")) {
                unsigned long type = 0, array = 0, block = 0;
                sscanf(item, "%lu:%lu:%lu", &type, &array, &block);
                MTLArgumentDescriptor *one = [MTLArgumentDescriptor argumentDescriptor];
                one.dataType = (MTLDataType)type;
                one.index = index;
                one.arrayLength = array;
                one.access = MTLBindingAccessReadOnly;
                one.constantBlockAlignment = block;
                index += array > 0 ? array : 1;
                [descriptors addObject:one];
            }
            printf("spec=%s descriptors=%lu\n", argv[2], (unsigned long)descriptors.count);
            fflush(stdout);
            id<MTLArgumentEncoder> encoder = [device newArgumentEncoderWithArguments:descriptors];
            if (encoder == nil) {
                printf("encoder=nil\n");
                return 0;
            }
            printf("encodedLength=%lu alignment=%lu\n", (unsigned long)encoder.encodedLength,
                   (unsigned long)encoder.alignment);
            // Where each constant argument sits, which is the layout fact an offset stands for.
            id<MTLBuffer> buffer = [device newBufferWithLength:4096
                                                      options:MTLResourceStorageModeShared];
            NSUInteger start = argc > 3 ? (NSUInteger)strtoul(argv[3], NULL, 0) : 0;
            long element = argc > 4 ? strtol(argv[4], NULL, 0) : -1;
            if (element < 0) {
                [encoder setArgumentBuffer:buffer offset:start];
            } else {
                [encoder setArgumentBuffer:buffer startOffset:start arrayElement:(NSUInteger)element];
            }
            printf("start=%lu element=%ld set\n", (unsigned long)start, element);
            fflush(stdout);
            for (MTLArgumentDescriptor *one in descriptors) {
                if (one.dataType >= MTLDataTypeFloat && one.dataType <= MTLDataTypeBool4) {
                    void *at = [encoder constantDataAtIndex:one.index];
                    printf("constant index=%lu offset=%ld\n", (unsigned long)one.index,
                           (long)((char *)at - (char *)buffer.contents));
                }
            }
            return 0;
        }
        fprintf(stderr, "unknown shape\n");
        return 2;
    }
}
