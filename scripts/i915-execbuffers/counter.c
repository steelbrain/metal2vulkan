// A counter of the kernel submissions a process makes on i915 and amdgpu, loaded with LD_PRELOAD:
// each DRM_IOCTL_I915_GEM_EXECBUFFER2 and DRM_IOCTL_AMDGPU_CS, the buffer objects it lists (by how
// each was made: created with its size, a userptr import, a dma-buf import) and the time the ioctl
// took, by thread. It reads the ioctls' arguments and results and changes nothing. ANV on i915
// lists every VkDeviceMemory of the device, and its own buffer objects, in every execbuffer (Mesa
// 26.2.4, src/intel/vulkan/i915/anv_batch_chain.c, setup_execbuf_for_cmd_buffers), RADV on amdgpu
// in every command submission's BO list (src/amd/vulkan/winsys/amdgpu), and the kernel's cost of
// each is per object.
//
//   gcc -O2 -Wall -shared -fPIC -o counter.so counter.c
//   LD_PRELOAD=$PWD/counter.so M2V_I915_SHIM_OUT=counts.txt <program>
//
// At exit it appends its totals, the objects' origins and sizes, and the execbuffers per thread to
// M2V_I915_SHIM_OUT (stderr without it); "execbuffers" counts amdgpu's command submissions too. A
// program reads the totals so far through the exported m2v_i915_shim_counts.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <drm/drm.h>
#include <drm/i915_drm.h>
#include <drm/amdgpu_drm.h>

enum origin { O_NONE = 0, O_CREATE, O_CREATE_EXT, O_USERPTR, O_PRIME, O_COUNT };
static const char *origin_names[] = {"unknown", "create", "create_ext", "userptr", "prime"};

#define MAX_HANDLES (1u << 20)
#define MAX_FDS 8
struct bo { uint64_t size; uint8_t origin; };
static struct bo *tables[MAX_FDS];
static int fds[MAX_FDS];
static int nfds;

struct m2v_shim_counts {
  uint64_t execbuffers, objects, nanos, userptr, prime, block64, created_other, unknown,
      max_objects, min_objects;
};
static struct m2v_shim_counts counts = {.min_objects = UINT64_MAX};

// (origin, size) histogram of objects listed by execbuffers.
#define HIST 4096
struct hist { uint64_t size; uint8_t origin; uint64_t seen; };
static struct hist hist[HIST];
static int nhist;
// Objects-per-execbuffer histogram.
static uint64_t per_eb[2048];
// Execbuffers per thread.
struct tidc { pid_t tid; uint64_t n; uint64_t nanos; char comm[32]; };
static struct tidc tids[64];
static int ntids;

static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static int (*real_ioctl)(int, unsigned long, ...);

// Each file descriptor's kernel driver, by DRM_IOCTL_VERSION, so one driver's ioctl numbers are
// not read as another's.
enum driver { D_UNKNOWN = 0, D_OTHER, D_I915, D_AMDGPU };
#define MAX_DRIVER_FDS 1024
static uint8_t drivers[MAX_DRIVER_FDS];

static enum driver driver_of(int fd) {
  if (fd < 0 || fd >= MAX_DRIVER_FDS) return D_OTHER;
  if (drivers[fd] != D_UNKNOWN) return (enum driver)drivers[fd];
  char name[32] = {0};
  struct drm_version v = {0};
  v.name = name;
  v.name_len = sizeof name - 1;
  enum driver d = D_OTHER;
  if (real_ioctl(fd, DRM_IOCTL_VERSION, &v) == 0) {
    if (strcmp(name, "i915") == 0) d = D_I915;
    else if (strcmp(name, "amdgpu") == 0) d = D_AMDGPU;
  }
  drivers[fd] = d;
  return d;
}

static struct bo *table_for(int fd, int make) {
  for (int i = 0; i < nfds; i++)
    if (fds[i] == fd) return tables[i];
  if (!make || nfds == MAX_FDS) return NULL;
  tables[nfds] = calloc(MAX_HANDLES, sizeof(struct bo));
  fds[nfds] = fd;
  return tables[nfds++];
}

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

static void note(int fd, uint32_t handle, uint64_t size, uint8_t origin) {
  pthread_mutex_lock(&mu);
  struct bo *t = table_for(fd, 1);
  if (t && handle < MAX_HANDLES) { t[handle].size = size; t[handle].origin = origin; }
  pthread_mutex_unlock(&mu);
}

static void hist_add(uint64_t size, uint8_t origin) {
  for (int i = 0; i < nhist; i++)
    if (hist[i].size == size && hist[i].origin == origin) { hist[i].seen++; return; }
  if (nhist < HIST) { hist[nhist].size = size; hist[nhist].origin = origin; hist[nhist].seen = 1; nhist++; }
}

void m2v_i915_shim_counts(struct m2v_shim_counts *out) {
  pthread_mutex_lock(&mu);
  *out = counts;
  pthread_mutex_unlock(&mu);
}

// One kernel submission of `n` objects, the handle of object i at `handles + i * stride`, that
// took `dt` nanoseconds; under `mu`.
static void count_submission(int fd, uint32_t n, const uint8_t *handles, size_t stride, uint64_t dt) {
    pid_t tid = (pid_t)syscall(SYS_gettid);
    struct bo *t = table_for(fd, 1);
    counts.execbuffers++;
    counts.objects += n;
    counts.nanos += dt;
    if (n > counts.max_objects) counts.max_objects = n;
    if (n < counts.min_objects) counts.min_objects = n;
    per_eb[n < 2047 ? n : 2047]++;
    for (uint32_t i = 0; i < n; i++) {
      uint32_t h;
      memcpy(&h, handles + (size_t)i * stride, sizeof h);
      struct bo b = (t && h < MAX_HANDLES) ? t[h] : (struct bo){0, O_NONE};
      switch (b.origin) {
      case O_USERPTR: counts.userptr++; break;
      case O_PRIME: counts.prime++; break;
      case O_CREATE: case O_CREATE_EXT:
        if (b.size == (64ull << 20)) counts.block64++; else counts.created_other++;
        break;
      default: counts.unknown++;
      }
      hist_add(b.size, b.origin);
    }
    int found = 0;
    for (int i = 0; i < ntids; i++)
      if (tids[i].tid == tid) { tids[i].n++; tids[i].nanos += dt; found = 1; break; }
    if (!found && ntids < 64) {
      tids[ntids].tid = tid; tids[ntids].n = 1; tids[ntids].nanos = dt;
      char path[64];
      snprintf(path, sizeof path, "/proc/self/task/%d/comm", tid);
      FILE *f = fopen(path, "r");
      if (f) { if (fgets(tids[ntids].comm, sizeof tids[ntids].comm, f)) tids[ntids].comm[strcspn(tids[ntids].comm, "\n")] = 0; fclose(f); }
      ntids++;
    }
}

// The objects of an amdgpu command submission: its BO handles chunk's entries.
static void count_cs(int fd, union drm_amdgpu_cs *cs, uint64_t dt) {
  const uint64_t *chunks = (const uint64_t *)(uintptr_t)cs->in.chunks;
  uint32_t n = 0;
  const uint8_t *entries = NULL;
  size_t stride = sizeof(struct drm_amdgpu_bo_list_entry);
  for (uint32_t i = 0; i < cs->in.num_chunks; i++) {
    const struct drm_amdgpu_cs_chunk *chunk = (const void *)(uintptr_t)chunks[i];
    if (chunk->chunk_id != AMDGPU_CHUNK_ID_BO_HANDLES) continue;
    const struct drm_amdgpu_bo_list_in *list = (const void *)(uintptr_t)chunk->chunk_data;
    n = list->bo_number;
    stride = list->bo_info_size;
    entries = (const uint8_t *)(uintptr_t)list->bo_info_ptr;
  }
  count_submission(fd, entries ? n : 0, entries, stride, dt);
}

// An amdgpu file's ioctl: its command submissions counted, its buffers' origins noted.
static int amdgpu_ioctl(int fd, unsigned long request, void *arg) {
  if (request == DRM_IOCTL_AMDGPU_CS) {
    // The result overwrites the argument's first words; the chunks stay.
    union drm_amdgpu_cs copy = *(union drm_amdgpu_cs *)arg;
    uint64_t t0 = now_ns();
    int r = real_ioctl(fd, request, arg);
    uint64_t dt = now_ns() - t0;
    pthread_mutex_lock(&mu);
    count_cs(fd, &copy, dt);
    pthread_mutex_unlock(&mu);
    return r;
  }
  // A buffer's size is in the argument the result overwrites.
  uint64_t size = 0;
  if (request == DRM_IOCTL_AMDGPU_GEM_CREATE)
    size = ((union drm_amdgpu_gem_create *)arg)->in.bo_size;
  int r = real_ioctl(fd, request, arg);
  if (r != 0) return r;
  if (request == DRM_IOCTL_AMDGPU_GEM_CREATE)
    note(fd, ((union drm_amdgpu_gem_create *)arg)->out.handle, size, O_CREATE);
  if (request == DRM_IOCTL_AMDGPU_GEM_USERPTR) {
    struct drm_amdgpu_gem_userptr *u = arg;
    note(fd, u->handle, u->size, O_USERPTR);
  }
  return r;
}

int ioctl(int fd, unsigned long request, ...) {
  va_list ap;
  va_start(ap, request);
  void *arg = va_arg(ap, void *);
  va_end(ap);
  if (!real_ioctl) real_ioctl = (int (*)(int, unsigned long, ...))dlsym(RTLD_NEXT, "ioctl");
  // Only a DRM driver's own ioctls, of a DRM device's file.
  if (_IOC_TYPE(request) != DRM_IOCTL_BASE) return real_ioctl(fd, request, arg);
  if (request == DRM_IOCTL_GEM_CLOSE) {
    int r = real_ioctl(fd, request, arg);
    if (r == 0) {
      struct drm_gem_close *c = arg;
      pthread_mutex_lock(&mu);
      struct bo *t = table_for(fd, 0);
      if (t && c->handle < MAX_HANDLES) t[c->handle] = (struct bo){0, O_NONE};
      pthread_mutex_unlock(&mu);
    }
    return r;
  }
  if (request == DRM_IOCTL_PRIME_FD_TO_HANDLE) {
    int r = real_ioctl(fd, request, arg);
    if (r == 0) {
      struct drm_prime_handle *p = arg;
      note(fd, p->handle, 0, O_PRIME);
    }
    return r;
  }
  enum driver driver = driver_of(fd);
  if (driver == D_AMDGPU) return amdgpu_ioctl(fd, request, arg);
  if (driver != D_I915) return real_ioctl(fd, request, arg);
  if (request == DRM_IOCTL_I915_GEM_EXECBUFFER2 || request == DRM_IOCTL_I915_GEM_EXECBUFFER2_WR) {
    struct drm_i915_gem_execbuffer2 *eb = arg;
    uint64_t t0 = now_ns();
    int r = real_ioctl(fd, request, arg);
    uint64_t dt = now_ns() - t0;
    pthread_mutex_lock(&mu);
    count_submission(fd, eb->buffer_count, (const uint8_t *)(uintptr_t)eb->buffers_ptr,
                     sizeof(struct drm_i915_gem_exec_object2), dt);
    pthread_mutex_unlock(&mu);
    return r;
  }
  int r = real_ioctl(fd, request, arg);
  if (r != 0) return r;
  switch (request) {
  case DRM_IOCTL_I915_GEM_CREATE: {
    struct drm_i915_gem_create *c = arg;
    note(fd, c->handle, c->size, O_CREATE);
    break;
  }
  case DRM_IOCTL_I915_GEM_CREATE_EXT: {
    struct drm_i915_gem_create_ext *c = arg;
    note(fd, c->handle, c->size, O_CREATE_EXT);
    break;
  }
  case DRM_IOCTL_I915_GEM_USERPTR: {
    struct drm_i915_gem_userptr *u = arg;
    note(fd, u->handle, u->user_size, O_USERPTR);
    break;
  }
  }
  return r;
}

static int by_seen(const void *a, const void *b) {
  const struct hist *x = a, *y = b;
  return x->seen < y->seen ? 1 : x->seen > y->seen ? -1 : 0;
}

__attribute__((destructor)) static void report(void) {
  if (!counts.execbuffers) return;
  const char *out = getenv("M2V_I915_SHIM_OUT");
  FILE *f = out ? fopen(out, "a") : stderr;
  if (!f) f = stderr;
  fprintf(f, "I915-SHIM pid %d: %llu execbuffers, %llu objects (%.1f each, min %llu max %llu), %.1f us each in the ioctl\n",
          getpid(), (unsigned long long)counts.execbuffers, (unsigned long long)counts.objects,
          (double)counts.objects / counts.execbuffers, (unsigned long long)counts.min_objects,
          (unsigned long long)counts.max_objects, counts.nanos / 1e3 / counts.execbuffers);
  qsort(hist, nhist, sizeof hist[0], by_seen);
  for (int i = 0; i < nhist && i < 60; i++)
    fprintf(f, "I915-SHIM   %-10s size %10llu: %.2f per execbuffer\n", origin_names[hist[i].origin],
            (unsigned long long)hist[i].size, (double)hist[i].seen / counts.execbuffers);
  for (int i = 0; i < ntids; i++)
    fprintf(f, "I915-SHIM   thread %d %s: %llu execbuffers, %.1f us each\n", tids[i].tid, tids[i].comm,
            (unsigned long long)tids[i].n, tids[i].nanos / 1e3 / tids[i].n);
  for (int i = 0; i < 2048; i++)
    if (per_eb[i]) fprintf(f, "I915-SHIM   %d objects: %llu execbuffers\n", i, (unsigned long long)per_eb[i]);
  if (f != stderr) fclose(f);
}
