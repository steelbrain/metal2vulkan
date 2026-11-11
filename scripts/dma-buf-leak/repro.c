/*
 * Minimal reproducer: dma-bufs that stay alive, attached to the GPU, after Vulkan freed their
 * imports.
 *
 * Plain Vulkan (VK_KHR_external_memory_fd, VK_EXT_external_memory_dma_buf) and udmabuf, nothing
 * else. Pages come from one sealed memfd; each import is a udmabuf of runs of its pages
 * (UDMABUF_CREATE_LIST), named "m2vleak-c<pid>" (DMA_BUF_SET_NAME), imported as the memory of a
 * VkBuffer, bound, and later freed (vkDestroyBuffer, vkFreeMemory). THREADS threads each keep up
 * to WINDOW imports alive and free a random one past it. After each round every import is freed
 * and every descriptor of ours closed, so each of our dma-bufs the kernel still lists in
 * /sys/kernel/debug/dma_buf/bufinfo is one it kept.
 *
 * Build: cc -O2 -pthread -o repro repro.c -ldl -lm
 * Run:   ./repro [options]      (reads bufinfo directly, or through `sudo -n cat` with -S)
 *
 *   -t THREADS  importing threads (8)        -w WINDOW  imports each keeps alive (4)
 *   -n COUNT    imports a thread makes a round (50)   -r ROUNDS (5)
 *   -m KIB      smallest import (16)          -M MIB  largest import (64)
 *   -l 0|1      1: UDMABUF_CREATE_LIST of scattered runs; 0: one UDMABUF_CREATE (1)
 *   -c 0|1      0: never used by the GPU; 1: each copied from once, waited for, before its
 *               free (0)
 *   -b 0|1      1: a thread keeps the queue busy with copies between two buffers of its own (0)
 *   -a 0|1      1: bufferDeviceAddress, imports with VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT (0)
 *   -x 0|1      1: one lock around every import and free (0)
 *   -d INDEX    physical device (0)           -f MIB  the memfd's size (512)
 *   -S          read bufinfo through `sudo -n cat`
 *
 * It prints one line a round: imports made and freed, our dma-bufs the kernel still lists, and
 * how many of them are attached to a device. A control first imports one buffer and checks that
 * the kernel lists it while imported.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <math.h>

#if __has_include(<vulkan/vulkan_core.h>) && !defined(REPRO_OWN_VULKAN)
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#else
/* The few Vulkan declarations used, for hosts without the headers (identical ABI). */
#define VK_NO_PROTOTYPES
typedef uint32_t VkFlags;
typedef uint32_t VkBool32;
typedef uint64_t VkDeviceSize;
typedef int32_t VkResult;
typedef int32_t VkStructureType;
typedef struct VkInstance_T *VkInstance;
typedef struct VkPhysicalDevice_T *VkPhysicalDevice;
typedef struct VkDevice_T *VkDevice;
typedef struct VkQueue_T *VkQueue;
typedef struct VkCommandBuffer_T *VkCommandBuffer;
typedef uint64_t VkDeviceMemory, VkBuffer, VkCommandPool, VkFence, VkSemaphore;
typedef void (*PFN_vkVoidFunction)(void);
typedef PFN_vkVoidFunction (*PFN_vkGetInstanceProcAddr)(VkInstance, const char *);
#define VK_SUCCESS 0
#define VK_TRUE 1
#define VK_MAKE_API_VERSION(v, a, b, c) ((((uint32_t)(v)) << 29) | (((uint32_t)(a)) << 22) | (((uint32_t)(b)) << 12) | ((uint32_t)(c)))
#define VK_STRUCTURE_TYPE_APPLICATION_INFO 0
#define VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO 1
#define VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO 2
#define VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO 3
#define VK_STRUCTURE_TYPE_SUBMIT_INFO 4
#define VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO 5
#define VK_STRUCTURE_TYPE_FENCE_CREATE_INFO 8
#define VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO 12
#define VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO 39
#define VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO 40
#define VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO 42
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 1000059000
#define VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO 1000060000
#define VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO 1000072001
#define VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR 1000074000
#define VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR 1000074001
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES 1000257000
#define VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT 0x00000200
#define VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT 0x00000002
#define VK_BUFFER_USAGE_TRANSFER_SRC_BIT 0x1
#define VK_BUFFER_USAGE_TRANSFER_DST_BIT 0x2
#define VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT 0x10
#define VK_BUFFER_USAGE_STORAGE_BUFFER_BIT 0x20
#define VK_BUFFER_USAGE_INDEX_BUFFER_BIT 0x40
#define VK_BUFFER_USAGE_VERTEX_BUFFER_BIT 0x80
#define VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT 0x100
#define VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT 0x20000
#define VK_SHARING_MODE_EXCLUSIVE 0
#define VK_QUEUE_GRAPHICS_BIT 0x1
#define VK_COMMAND_BUFFER_LEVEL_PRIMARY 0
#define VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT 0x2
#define VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT 0x1
typedef struct { VkStructureType sType; const void *pNext; const char *pApplicationName; uint32_t applicationVersion; const char *pEngineName; uint32_t engineVersion; uint32_t apiVersion; } VkApplicationInfo;
typedef struct { VkStructureType sType; const void *pNext; VkFlags flags; const VkApplicationInfo *pApplicationInfo; uint32_t enabledLayerCount; const char *const *ppEnabledLayerNames; uint32_t enabledExtensionCount; const char *const *ppEnabledExtensionNames; } VkInstanceCreateInfo;
typedef struct { VkStructureType sType; const void *pNext; VkFlags flags; uint32_t queueFamilyIndex; uint32_t queueCount; const float *pQueuePriorities; } VkDeviceQueueCreateInfo;
typedef struct { VkStructureType sType; const void *pNext; VkFlags flags; uint32_t queueCreateInfoCount; const VkDeviceQueueCreateInfo *pQueueCreateInfos; uint32_t enabledLayerCount; const char *const *ppEnabledLayerNames; uint32_t enabledExtensionCount; const char *const *ppEnabledExtensionNames; const void *pEnabledFeatures; } VkDeviceCreateInfo;
typedef struct { uint32_t width, height, depth; } VkExtent3D;
typedef struct { VkFlags queueFlags; uint32_t queueCount; uint32_t timestampValidBits; VkExtent3D minImageTransferGranularity; } VkQueueFamilyProperties;
typedef struct { VkBool32 values[55]; } VkPhysicalDeviceFeatures;
typedef struct { VkStructureType sType; void *pNext; VkPhysicalDeviceFeatures features; } VkPhysicalDeviceFeatures2;
typedef struct { VkStructureType sType; void *pNext; VkBool32 bufferDeviceAddress; VkBool32 bufferDeviceAddressCaptureReplay; VkBool32 bufferDeviceAddressMultiDevice; } VkPhysicalDeviceBufferDeviceAddressFeatures;
typedef struct { VkStructureType sType; const void *pNext; VkFlags flags; VkDeviceSize size; VkFlags usage; int32_t sharingMode; uint32_t queueFamilyIndexCount; const uint32_t *pQueueFamilyIndices; } VkBufferCreateInfo;
typedef struct { VkStructureType sType; const void *pNext; VkFlags handleTypes; } VkExternalMemoryBufferCreateInfo;
typedef struct { VkDeviceSize size; VkDeviceSize alignment; uint32_t memoryTypeBits; } VkMemoryRequirements;
typedef struct { VkStructureType sType; const void *pNext; VkDeviceSize allocationSize; uint32_t memoryTypeIndex; } VkMemoryAllocateInfo;
typedef struct { VkStructureType sType; const void *pNext; VkFlags handleType; int fd; } VkImportMemoryFdInfoKHR;
typedef struct { VkStructureType sType; void *pNext; uint32_t memoryTypeBits; } VkMemoryFdPropertiesKHR;
typedef struct { VkStructureType sType; const void *pNext; VkFlags flags; uint32_t deviceMask; } VkMemoryAllocateFlagsInfo;
typedef struct { VkStructureType sType; const void *pNext; VkFlags flags; uint32_t queueFamilyIndex; } VkCommandPoolCreateInfo;
typedef struct { VkStructureType sType; const void *pNext; VkCommandPool commandPool; int32_t level; uint32_t commandBufferCount; } VkCommandBufferAllocateInfo;
typedef struct { VkStructureType sType; const void *pNext; VkFlags flags; const void *pInheritanceInfo; } VkCommandBufferBeginInfo;
typedef struct { VkDeviceSize srcOffset, dstOffset, size; } VkBufferCopy;
typedef struct { VkStructureType sType; const void *pNext; uint32_t waitSemaphoreCount; const VkSemaphore *pWaitSemaphores; const VkFlags *pWaitDstStageMask; uint32_t commandBufferCount; const VkCommandBuffer *pCommandBuffers; uint32_t signalSemaphoreCount; const VkSemaphore *pSignalSemaphores; } VkSubmitInfo;
typedef struct { VkStructureType sType; const void *pNext; VkFlags flags; } VkFenceCreateInfo;
typedef VkResult (*PFN_vkCreateInstance)(const VkInstanceCreateInfo *, const void *, VkInstance *);
typedef VkResult (*PFN_vkEnumeratePhysicalDevices)(VkInstance, uint32_t *, VkPhysicalDevice *);
typedef void (*PFN_vkGetPhysicalDeviceProperties)(VkPhysicalDevice, void *);
typedef void (*PFN_vkGetPhysicalDeviceQueueFamilyProperties)(VkPhysicalDevice, uint32_t *, VkQueueFamilyProperties *);
typedef VkResult (*PFN_vkCreateDevice)(VkPhysicalDevice, const VkDeviceCreateInfo *, const void *, VkDevice *);
typedef void (*PFN_vkGetDeviceQueue)(VkDevice, uint32_t, uint32_t, VkQueue *);
typedef VkResult (*PFN_vkGetMemoryFdPropertiesKHR)(VkDevice, VkFlags, int, VkMemoryFdPropertiesKHR *);
typedef VkResult (*PFN_vkCreateBuffer)(VkDevice, const VkBufferCreateInfo *, const void *, VkBuffer *);
typedef void (*PFN_vkDestroyBuffer)(VkDevice, VkBuffer, const void *);
typedef void (*PFN_vkGetBufferMemoryRequirements)(VkDevice, VkBuffer, VkMemoryRequirements *);
typedef VkResult (*PFN_vkAllocateMemory)(VkDevice, const VkMemoryAllocateInfo *, const void *, VkDeviceMemory *);
typedef void (*PFN_vkFreeMemory)(VkDevice, VkDeviceMemory, const void *);
typedef VkResult (*PFN_vkBindBufferMemory)(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize);
typedef VkResult (*PFN_vkCreateCommandPool)(VkDevice, const VkCommandPoolCreateInfo *, const void *, VkCommandPool *);
typedef void (*PFN_vkDestroyCommandPool)(VkDevice, VkCommandPool, const void *);
typedef VkResult (*PFN_vkAllocateCommandBuffers)(VkDevice, const VkCommandBufferAllocateInfo *, VkCommandBuffer *);
typedef VkResult (*PFN_vkBeginCommandBuffer)(VkCommandBuffer, const VkCommandBufferBeginInfo *);
typedef VkResult (*PFN_vkEndCommandBuffer)(VkCommandBuffer);
typedef void (*PFN_vkCmdCopyBuffer)(VkCommandBuffer, VkBuffer, VkBuffer, uint32_t, const VkBufferCopy *);
typedef VkResult (*PFN_vkQueueSubmit)(VkQueue, uint32_t, const VkSubmitInfo *, VkFence);
typedef VkResult (*PFN_vkCreateFence)(VkDevice, const VkFenceCreateInfo *, const void *, VkFence *);
typedef void (*PFN_vkDestroyFence)(VkDevice, VkFence, const void *);
typedef VkResult (*PFN_vkWaitForFences)(VkDevice, uint32_t, const VkFence *, VkBool32, uint64_t);
typedef VkResult (*PFN_vkResetFences)(VkDevice, uint32_t, const VkFence *);
typedef VkResult (*PFN_vkDeviceWaitIdle)(VkDevice);
typedef void (*PFN_vkDestroyDevice)(VkDevice, const void *);
typedef void (*PFN_vkDestroyInstance)(VkInstance, const void *);
#endif

/* udmabuf and dma-buf ioctls (Linux uapi), declared here so no kernel headers are needed. */
struct udmabuf_create_item { uint32_t memfd, pad; uint64_t offset, size; };
struct udmabuf_create { uint32_t memfd, flags; uint64_t offset, size; };
struct udmabuf_create_list { uint32_t flags, count; /* then the items */ };
#define UDMABUF_FLAGS_CLOEXEC 0x01
#define UDMABUF_CREATE _IOW('u', 0x42, struct udmabuf_create)
#define UDMABUF_CREATE_LIST _IOW('u', 0x43, struct udmabuf_create_list)
#define DMA_BUF_SET_NAME _IOW('b', 1, const char *)
#define MOST_RUNS 1024
#define BUFINFO "/sys/kernel/debug/dma_buf/bufinfo"

#define CHECK(call) do { VkResult r_ = (call); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s:%d: %s: VkResult %d\n", __FILE__, __LINE__, #call, r_); exit(1); } } while (0)

static struct {
    int threads, window, count, rounds, list, copy, busy, address, serial, sudo, device;
    uint64_t smallest, largest, file;
} opt = { 8, 4, 50, 5, 1, 0, 0, 0, 0, 0, 0, 16 << 10, 64 << 20, 512 << 20 };

#define F(name) static PFN_##name name
F(vkGetInstanceProcAddr); F(vkCreateInstance); F(vkEnumeratePhysicalDevices);
F(vkGetPhysicalDeviceProperties); F(vkGetPhysicalDeviceQueueFamilyProperties);
F(vkCreateDevice); F(vkGetDeviceQueue); F(vkGetMemoryFdPropertiesKHR);
F(vkCreateBuffer); F(vkDestroyBuffer); F(vkGetBufferMemoryRequirements); F(vkAllocateMemory);
F(vkFreeMemory); F(vkBindBufferMemory); F(vkCreateCommandPool); F(vkDestroyCommandPool);
F(vkAllocateCommandBuffers); F(vkBeginCommandBuffer); F(vkEndCommandBuffer); F(vkCmdCopyBuffer);
F(vkQueueSubmit); F(vkCreateFence); F(vkDestroyFence); F(vkWaitForFences); F(vkResetFences);
F(vkDeviceWaitIdle); F(vkDestroyDevice); F(vkDestroyInstance);
#undef F

static VkInstance instance;
static VkDevice device;
static VkQueue queue;
static uint32_t family;
static int memfd;
static uint64_t page, pages;
static char tag[32];
static pthread_mutex_t queue_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t serial_lock = PTHREAD_MUTEX_INITIALIZER;
static volatile int busy_stop;
static uint64_t made, freed; /* updated atomically */

static uint64_t next(uint64_t *s) { *s ^= *s >> 12; *s ^= *s << 25; *s ^= *s >> 27; return *s * 0x2545f4914f6cdd1dull; }

/* One udmabuf of LENGTH bytes of the memfd: scattered runs (a list), or one run. */
static int make_dmabuf(uint64_t *rng, uint64_t length) {
    uint64_t n = length / page;
    uint64_t runs = opt.list ? 1 + next(rng) % (n < MOST_RUNS ? n : MOST_RUNS) : 1;
    int dev = open("/dev/udmabuf", O_RDWR | O_CLOEXEC);
    if (dev < 0) { perror("/dev/udmabuf"); exit(1); }
    int fd;
    if (runs == 1) {
        struct udmabuf_create create = { (uint32_t)memfd, UDMABUF_FLAGS_CLOEXEC, next(rng) % (pages - n + 1) * page, length };
        fd = ioctl(dev, UDMABUF_CREATE, &create);
    } else {
        size_t bytes = 8 + runs * sizeof(struct udmabuf_create_item);
        uint32_t *list = calloc(1, bytes);
        struct udmabuf_create_item *items = (struct udmabuf_create_item *)(list + 2);
        list[0] = UDMABUF_FLAGS_CLOEXEC;
        list[1] = (uint32_t)runs;
        for (uint64_t i = 0; i < runs; i++) {
            uint64_t run = n / runs + (i < n % runs);
            items[i] = (struct udmabuf_create_item){ (uint32_t)memfd, 0, next(rng) % (pages - run + 1) * page, run * page };
        }
        fd = ioctl(dev, UDMABUF_CREATE_LIST, list);
        free(list);
    }
    close(dev);
    if (fd < 0) { perror("udmabuf"); exit(1); }
    if (ioctl(fd, DMA_BUF_SET_NAME, tag) != 0) { perror("DMA_BUF_SET_NAME"); exit(1); }
    return fd;
}

struct import { VkBuffer buffer; VkDeviceMemory memory; VkDeviceSize size; };

/* Import a new dma-buf as the memory of a buffer of its length. */
static struct import import(uint64_t *rng) {
    double lo = log((double)opt.smallest), hi = log((double)opt.largest);
    uint64_t length = (uint64_t)exp(lo + (double)(next(rng) >> 11) / 9007199254740992.0 * (hi - lo)) / page * page;
    if (length < opt.smallest) length = opt.smallest;
    if (length > opt.largest) length = opt.largest;
    int fd = make_dmabuf(rng, length);
    if (opt.serial) pthread_mutex_lock(&serial_lock);
    VkMemoryFdPropertiesKHR props = { VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR };
    CHECK(vkGetMemoryFdPropertiesKHR(device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, fd, &props));
    VkExternalMemoryBufferCreateInfo external = { VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, NULL, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT };
    VkBufferCreateInfo info = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &external, 0, length,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | (opt.address ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT : 0),
        VK_SHARING_MODE_EXCLUSIVE, 0, NULL };
    struct import im = { 0, 0, length };
    CHECK(vkCreateBuffer(device, &info, NULL, &im.buffer));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, im.buffer, &req);
    uint32_t types = props.memoryTypeBits & req.memoryTypeBits;
    if (!types) { fprintf(stderr, "no memory type for the dma-buf\n"); exit(1); }
    /* Vulkan owns the descriptor it imports; ours is closed after, as a program keeps its own. */
    int handed = dup(fd);
    VkMemoryAllocateFlagsInfo flags = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, NULL, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, 0 };
    VkImportMemoryFdInfoKHR fdinfo = { VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR, opt.address ? &flags : NULL, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, handed };
    VkMemoryAllocateInfo alloc = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &fdinfo, length, (uint32_t)__builtin_ctz(types) };
    CHECK(vkAllocateMemory(device, &alloc, NULL, &im.memory));
    CHECK(vkBindBufferMemory(device, im.buffer, im.memory, 0));
    if (opt.serial) pthread_mutex_unlock(&serial_lock);
    close(fd);
    __atomic_add_fetch(&made, 1, __ATOMIC_RELAXED);
    return im;
}

static void release(struct import im) {
    if (opt.serial) pthread_mutex_lock(&serial_lock);
    vkDestroyBuffer(device, im.buffer, NULL);
    vkFreeMemory(device, im.memory, NULL);
    if (opt.serial) pthread_mutex_unlock(&serial_lock);
    __atomic_add_fetch(&freed, 1, __ATOMIC_RELAXED);
}

static void drop_plain(struct import im) {
    vkDestroyBuffer(device, im.buffer, NULL);
    vkFreeMemory(device, im.memory, NULL);
}

/* A device buffer and its memory, for copies. */
static struct import plain(VkDeviceSize size) {
    VkBufferCreateInfo info = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL };
    struct import im = { 0, 0, size };
    CHECK(vkCreateBuffer(device, &info, NULL, &im.buffer));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, im.buffer, &req);
    VkMemoryAllocateInfo alloc = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size, (uint32_t)__builtin_ctz(req.memoryTypeBits) };
    CHECK(vkAllocateMemory(device, &alloc, NULL, &im.memory));
    CHECK(vkBindBufferMemory(device, im.buffer, im.memory, 0));
    return im;
}

struct copier { VkCommandPool pool; VkCommandBuffer cmd[2]; VkFence fence[2]; int pending[2]; int next; };

static void copier_init(struct copier *c) {
    memset(c, 0, sizeof *c);
    VkCommandPoolCreateInfo pool = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, family };
    CHECK(vkCreateCommandPool(device, &pool, NULL, &c->pool));
    VkCommandBufferAllocateInfo alloc = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, c->pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 2 };
    CHECK(vkAllocateCommandBuffers(device, &alloc, c->cmd));
    VkFenceCreateInfo fence = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, NULL, 0 };
    CHECK(vkCreateFence(device, &fence, NULL, &c->fence[0]));
    CHECK(vkCreateFence(device, &fence, NULL, &c->fence[1]));
}

static void copier_wait(struct copier *c, int i) {
    if (!c->pending[i]) return;
    CHECK(vkWaitForFences(device, 1, &c->fence[i], VK_TRUE, 60ull * 1000000000ull));
    CHECK(vkResetFences(device, 1, &c->fence[i]));
    c->pending[i] = 0;
}

/* Submit copies from SRC to DST (REPEAT times, SIZE bytes); wait where WAIT. */
static void copy(struct copier *c, VkBuffer src, VkBuffer dst, VkDeviceSize size, int repeat, int wait) {
    int i = c->next;
    c->next ^= 1;
    copier_wait(c, i);
    VkCommandBufferBeginInfo begin = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL };
    CHECK(vkBeginCommandBuffer(c->cmd[i], &begin));
    VkBufferCopy region = { 0, 0, size };
    for (int k = 0; k < repeat; k++) vkCmdCopyBuffer(c->cmd[i], src, dst, 1, &region);
    CHECK(vkEndCommandBuffer(c->cmd[i]));
    VkSubmitInfo submit = { VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &c->cmd[i], 0, NULL };
    pthread_mutex_lock(&queue_lock);
    CHECK(vkQueueSubmit(queue, 1, &submit, c->fence[i]));
    pthread_mutex_unlock(&queue_lock);
    c->pending[i] = 1;
    if (wait) copier_wait(c, i);
}

static void copier_free(struct copier *c) {
    copier_wait(c, 0);
    copier_wait(c, 1);
    vkDestroyFence(device, c->fence[0], NULL);
    vkDestroyFence(device, c->fence[1], NULL);
    vkDestroyCommandPool(device, c->pool, NULL);
}

struct worker { int index, round; };

static void *work(void *arg) {
    struct worker *w = arg;
    uint64_t rng = 0x9e3779b97f4a7c15ull * (uint64_t)(w->round * 64 + w->index + 1);
    struct import *live = calloc((size_t)opt.window + 1, sizeof *live);
    int held = 0;
    struct copier c;
    struct import out = { 0 };
    if (opt.copy) { copier_init(&c); out = plain(4096); }
    for (int i = 0; i < opt.count; i++) {
        live[held++] = import(&rng);
        if (opt.copy) copy(&c, live[held - 1].buffer, out.buffer, 4096 < live[held - 1].size ? 4096 : live[held - 1].size, 1, 1);
        if (held > opt.window) {
            int k = (int)(next(&rng) % (uint64_t)held);
            release(live[k]);
            live[k] = live[--held];
        }
    }
    while (held) release(live[--held]);
    if (opt.copy) { copier_free(&c); drop_plain(out); }
    free(live);
    return NULL;
}

static void *keep_busy(void *arg) {
    (void)arg;
    struct copier c;
    copier_init(&c);
    struct import a = plain(16 << 20), b = plain(16 << 20);
    while (!busy_stop) copy(&c, a.buffer, b.buffer, 16 << 20, 4, 0);
    copier_free(&c);
    drop_plain(a);
    drop_plain(b);
    return NULL;
}

/* Our dma-bufs the kernel lists, and how many of them are attached to a device; -1 unread. */
static void listed(long *count, long *attached) {
    FILE *f = opt.sudo ? popen("sudo -n cat " BUFINFO, "r") : fopen(BUFINFO, "r");
    *count = *attached = -1;
    if (!f) return;
    char line[4096];
    int ours = 0;
    long n = 0, a = 0, total = -1;
    size_t taglen = strlen(tag);
    while (fgets(line, sizeof line, f)) {
        size_t len = strcspn(line, "\n");
        line[len] = 0;
        if (line[0] >= '0' && line[0] <= '9' && strchr(line, '\t')) {
            ours = len > taglen && line[len - taglen - 1] == '\t' && strcmp(line + len - taglen, tag) == 0;
            n += ours;
        } else if (strstr(line, " devices attached") && sscanf(line, "Total %ld", &total) == 1) {
            a += ours && total > 0;
            ours = 0;
        }
    }
    int failed = opt.sudo ? pclose(f) != 0 : fclose(f) != 0;
    if (!failed) { *count = n; *attached = a; }
}

/* The listed count once it stops falling (at most 2 s). */
static void settled(long *count, long *attached) {
    listed(count, attached);
    for (int i = 0; i < 40 && *count > 0; i++) {
        usleep(50000);
        listed(count, attached);
    }
}

static void load(void) {
    void *lib = dlopen("libvulkan.so.1", RTLD_NOW);
    if (!lib) { fprintf(stderr, "libvulkan.so.1: %s\n", dlerror()); exit(1); }
    vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)dlsym(lib, "vkGetInstanceProcAddr");
#define I(name) name = (PFN_##name)vkGetInstanceProcAddr(instance, #name)
    I(vkCreateInstance);
    VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "dma-buf-leak-repro", 1, NULL, 0, VK_MAKE_API_VERSION(0, 1, 2, 0) };
    VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app, 0, NULL, 0, NULL };
    CHECK(vkCreateInstance(&ici, NULL, &instance));
    I(vkEnumeratePhysicalDevices); I(vkGetPhysicalDeviceProperties); I(vkGetPhysicalDeviceQueueFamilyProperties);
    I(vkCreateDevice); I(vkDestroyInstance);
    I(vkGetDeviceQueue); I(vkGetMemoryFdPropertiesKHR); I(vkCreateBuffer); I(vkDestroyBuffer);
    I(vkGetBufferMemoryRequirements); I(vkAllocateMemory); I(vkFreeMemory); I(vkBindBufferMemory);
    I(vkCreateCommandPool); I(vkDestroyCommandPool); I(vkAllocateCommandBuffers); I(vkBeginCommandBuffer);
    I(vkEndCommandBuffer); I(vkCmdCopyBuffer); I(vkQueueSubmit); I(vkCreateFence); I(vkDestroyFence);
    I(vkWaitForFences); I(vkResetFences); I(vkDeviceWaitIdle); I(vkDestroyDevice);
#undef I
}

int main(int argc, char **argv) {
    for (int c; (c = getopt(argc, argv, "t:w:n:r:m:M:l:c:b:a:x:d:f:S")) != -1;) {
        switch (c) {
        case 't': opt.threads = atoi(optarg); break;
        case 'w': opt.window = atoi(optarg); break;
        case 'n': opt.count = atoi(optarg); break;
        case 'r': opt.rounds = atoi(optarg); break;
        case 'm': opt.smallest = strtoull(optarg, NULL, 0) << 10; break;
        case 'M': opt.largest = strtoull(optarg, NULL, 0) << 20; break;
        case 'l': opt.list = atoi(optarg); break;
        case 'c': opt.copy = atoi(optarg); break;
        case 'b': opt.busy = atoi(optarg); break;
        case 'a': opt.address = atoi(optarg); break;
        case 'x': opt.serial = atoi(optarg); break;
        case 'd': opt.device = atoi(optarg); break;
        case 'f': opt.file = strtoull(optarg, NULL, 0) << 20; break;
        case 'S': opt.sudo = 1; break;
        default: fprintf(stderr, "see the comment at the top of repro.c\n"); return 2;
        }
    }
    page = (uint64_t)sysconf(_SC_PAGESIZE);
    opt.smallest = (opt.smallest + page - 1) / page * page;
    if (opt.largest > (64ull << 20)) opt.largest = 64ull << 20;
    if (opt.smallest > opt.largest) opt.smallest = opt.largest;
    snprintf(tag, sizeof tag, "m2vleak-c%d", (int)getpid());

    memfd = memfd_create("dma-buf-leak-repro", MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (memfd < 0 || ftruncate(memfd, (off_t)opt.file) != 0 ||
        fcntl(memfd, F_ADD_SEALS, F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL) != 0) { perror("memfd"); return 1; }
    pages = opt.file / page;

    load();
    uint32_t n = 0;
    CHECK(vkEnumeratePhysicalDevices(instance, &n, NULL));
    VkPhysicalDevice phys[16];
    if (n > 16) n = 16;
    CHECK(vkEnumeratePhysicalDevices(instance, &n, phys));
    if ((uint32_t)opt.device >= n) { fprintf(stderr, "no device %d\n", opt.device); return 1; }
    VkPhysicalDevice pd = phys[opt.device];
    _Alignas(8) unsigned char properties[2048];
    vkGetPhysicalDeviceProperties(pd, (void *)properties);
    const char *name = (const char *)properties + 20; /* VkPhysicalDeviceProperties::deviceName */
    uint32_t nf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nf, NULL);
    VkQueueFamilyProperties qf[32];
    if (nf > 32) nf = 32;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nf, qf);
    for (family = 0; family < nf && !(qf[family].queueFlags & VK_QUEUE_GRAPHICS_BIT); family++) {}
    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, family, 1, &priority };
    const char *extensions[] = { "VK_KHR_external_memory_fd", "VK_EXT_external_memory_dma_buf" };
    VkPhysicalDeviceBufferDeviceAddressFeatures bda = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES, NULL, VK_TRUE, 0, 0 };
    VkPhysicalDeviceFeatures2 features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &bda };
    VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, opt.address ? &features : NULL, 0, 1, &qci, 0, NULL, 2, extensions, NULL };
    CHECK(vkCreateDevice(pd, &dci, NULL, &device));
    vkGetDeviceQueue(device, family, 0, &queue);

    long count, attached;
    listed(&count, &attached);
    printf("repro on %s: threads %d, window %d, %d imports a thread a round, %d rounds, %llu to %llu bytes, %s, copies %d, busy %d, device address %d, serial %d, %llu-byte pages; ",
           name, opt.threads, opt.window, opt.count, opt.rounds, (unsigned long long)opt.smallest,
           (unsigned long long)opt.largest, opt.list ? "UDMABUF_CREATE_LIST" : "UDMABUF_CREATE",
           opt.copy, opt.busy, opt.address, opt.serial, (unsigned long long)page);
    if (count < 0) {
        printf("bufinfo unreadable (run as root, or with -S and sudo that asks no password)\n");
    } else {
        uint64_t rng = 1;
        struct import im = import(&rng);
        long c1, a1, c2, a2;
        listed(&c1, &a1);
        release(im);
        settled(&c2, &a2);
        printf("control: one import listed %ld (%ld attached), %ld after its free\n", c1, a1, c2);
        made = freed = 0;
        if (c1 < 1) { fprintf(stderr, "the kernel lists no dma-buf of ours while one is imported\n"); return 1; }
    }
    fflush(stdout);

    long kept_before = 0;
    uint64_t total = 0;
    pthread_t busy;
    if (opt.busy) pthread_create(&busy, NULL, keep_busy, NULL);
    for (int round = 1; round <= opt.rounds; round++) {
        pthread_t threads[64];
        struct worker workers[64];
        int t = opt.threads > 64 ? 64 : opt.threads;
        uint64_t made0 = made, freed0 = freed;
        for (int i = 0; i < t; i++) {
            workers[i] = (struct worker){ i, round };
            pthread_create(&threads[i], NULL, work, &workers[i]);
        }
        for (int i = 0; i < t; i++) pthread_join(threads[i], NULL);
        settled(&count, &attached);
        uint64_t m = made - made0, f = freed - freed0;
        total += m;
        if (count < 0) {
            printf("round %d: imports %llu, freed %llu; bufinfo unread\n", round, (unsigned long long)m, (unsigned long long)f);
        } else {
            printf("round %d: imports %llu, freed %llu; ours listed %ld (%ld attached), kept this round %ld (%.1f%%)\n",
                   round, (unsigned long long)m, (unsigned long long)f, count, attached, count - kept_before,
                   100.0 * (double)(count - kept_before) / (double)(m ? m : 1));
            kept_before = count;
        }
        fflush(stdout);
    }
    if (opt.busy) { busy_stop = 1; pthread_join(busy, NULL); }
    CHECK(vkDeviceWaitIdle(device));
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
    usleep(500000);
    listed(&count, &attached);
    printf("all rounds: imports %llu, ours listed after the device went %ld (%ld attached)\n",
           (unsigned long long)total, count, attached);
    return 0;
}
