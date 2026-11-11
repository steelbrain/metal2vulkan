//! Whether a sparse buffer can be bound to memory imported from host pointers
//! (`VK_EXT_external_memory_host`): a probe for a consumer that would lay separate chunks of its
//! RAM into one buffer, no product path. On the device the cfg selects it prints (`SPARSE-HOST`
//! lines):
//!
//! - the device's sparse features and properties, `sparseAddressSpaceSize` and
//!   `minImportedHostPointerAlignment`;
//! - for a buffer of `SPARSE_BINDING` (and of `SPARSE_BINDING` with `SPARSE_RESIDENCY`, where the
//!   device offers it), for transfer, storage, uniform and vertex use and all four: whether the
//!   device imports host allocations into such a buffer
//!   (`vkGetPhysicalDeviceExternalBufferProperties`), and the memory types and alignment it
//!   takes, without and with the host-allocation handle type;
//! - the memory types `vkGetMemoryHostPointerPropertiesEXT` answers for a page of a `memfd` mapped
//!   shared and for a page of anonymous memory, and the intersection with the buffer's.
//!
//! Where the device imports host allocations into such a buffer and the intersection is not
//! empty, it imports four separate `memfd` mappings of at least 64 KiB each, binds them in reverse
//! order into one buffer with `vkQueueBindSparse`, copies a pattern into the buffer on the GPU
//! and reads it through each mapping, then writes another through the mappings and copies it out
//! on the GPU. Every call is valid usage by the specification: the buffer names the
//! host-allocation handle type (VUID-VkSparseMemoryBind-memory-02731,
//! VUID-VkBufferCreateInfo-pNext-00920), the memory type is among those both answer
//! (VUID-VkMemoryAllocateInfo-memoryTypeIndex-01744),
//! and offsets and sizes are multiples of the buffer's alignment and of
//! `minImportedHostPointerAlignment` (VUID-VkSparseMemoryBind-resourceOffset-09491,
//! VUID-VkMemoryAllocateInfo-allocationSize-01745), on a `VkDevice` of the test's own with
//! `sparseBinding` enabled and a queue of a family that binds sparsely and copies. It asserts
//! nothing a device may answer either way: a refusal, a failed bind or a round trip that differs
//! is printed.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on Linux
//! (`memfd`), on the kind of device the cfg selects.

use super::blit_holds::{device, Scratch};
use ash::vk;
use std::ffi::c_void;
use std::fmt::Display;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::ptr::NonNull;
use std::time::Duration;

/// How many chunks the probe binds into one buffer.
const CHUNKS: usize = 4;

/// The least length of a chunk.
const LEAST_CHUNK: u64 = 64 * 1024;

/// A bound on each wait for the probe's device.
const WAIT: Duration = Duration::from_secs(20);

/// Transfer use, both ways.
const TRANSFER: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
    vk::BufferUsageFlags::TRANSFER_SRC.as_raw() | vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
);

/// The uses the probe makes a sparse buffer for, each named.
const USES: [(&str, vk::BufferUsageFlags); 5] = [
    ("transfer", TRANSFER),
    ("storage", vk::BufferUsageFlags::STORAGE_BUFFER),
    ("uniform", vk::BufferUsageFlags::UNIFORM_BUFFER),
    ("vertex", vk::BufferUsageFlags::VERTEX_BUFFER),
    (
        "all",
        vk::BufferUsageFlags::from_raw(
            TRANSFER.as_raw()
                | vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
                | vk::BufferUsageFlags::UNIFORM_BUFFER.as_raw()
                | vk::BufferUsageFlags::VERTEX_BUFFER.as_raw(),
        ),
    ),
];

/// The host-allocation handle type.
const HOST: vk::ExternalMemoryHandleTypeFlags =
    vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;

/// Print `line` for the device `name`.
fn say(name: &str, line: impl Display) {
    eprintln!("SPARSE-HOST {name}: {line}");
}

/// A byte of pattern `seed` at `index`.
fn pattern(index: usize, seed: u32) -> u8 {
    let mut x = (index as u32).wrapping_mul(2654435761) ^ seed.wrapping_mul(0x9e37_79b9);
    x ^= x >> 15;
    x = x.wrapping_mul(2246822519);
    (x ^ (x >> 13)) as u8
}

/// The greatest common divisor of `a` and `b`.
fn gcd(a: u64, b: u64) -> u64 {
    match b {
        0 => a,
        _ => gcd(b, a % b),
    }
}

/// The least common multiple of `a` and `b`, both above zero.
fn lcm(a: u64, b: u64) -> u64 {
    a / gcd(a, b) * b
}

/// The host's page size.
fn page() -> u64 {
    // SAFETY: reads a system constant.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    u64::try_from(page).expect("a page size")
}

/// A mapping of `length` bytes read and write, unmapped as it drops: of a `memfd` mapped shared,
/// or of anonymous memory.
struct Mapping {
    pointer: NonNull<u8>,
    length: usize,
    _fd: Option<OwnedFd>,
}

impl Mapping {
    /// `length` bytes of a new `memfd`, mapped shared.
    fn memfd(length: usize) -> Mapping {
        // SAFETY: a NUL-terminated name.
        let fd = unsafe { libc::memfd_create(c"m2v-sparse-host".as_ptr(), libc::MFD_CLOEXEC) };
        assert!(fd >= 0, "memfd: {}", std::io::Error::last_os_error());
        // SAFETY: a new descriptor, owned at once.
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };
        // SAFETY: sizes the memfd made above.
        let sized = unsafe { libc::ftruncate(fd.as_raw_fd(), length as libc::off_t) };
        assert_eq!(sized, 0, "ftruncate: {}", std::io::Error::last_os_error());
        let flags = libc::MAP_SHARED;
        // SAFETY: a new mapping of the memfd's `length` bytes, at an address the kernel chooses.
        let mapped = unsafe { Mapping::map(length, flags, fd.as_raw_fd()) };
        Mapping {
            pointer: mapped,
            length,
            _fd: Some(fd),
        }
    }

    /// `length` bytes of new anonymous memory, private.
    fn anonymous(length: usize) -> Mapping {
        let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS;
        // SAFETY: a new anonymous mapping, at an address the kernel chooses.
        let mapped = unsafe { Mapping::map(length, flags, -1) };
        Mapping {
            pointer: mapped,
            length,
            _fd: None,
        }
    }

    /// `mmap` of `length` bytes read and write under `flags`, of `fd`.
    ///
    /// # Safety
    ///
    /// `fd` is -1 for an anonymous mapping, else a descriptor of at least `length` bytes.
    unsafe fn map(length: usize, flags: libc::c_int, fd: libc::c_int) -> NonNull<u8> {
        let protection = libc::PROT_READ | libc::PROT_WRITE;
        // SAFETY: the caller's contract; the kernel chooses the address.
        let mapped = unsafe { libc::mmap(std::ptr::null_mut(), length, protection, flags, fd, 0) };
        assert_ne!(
            mapped,
            libc::MAP_FAILED,
            "mmap: {}",
            std::io::Error::last_os_error()
        );
        NonNull::new(mapped.cast()).expect("a mapping")
    }

    /// Whether the mapping starts at a multiple of `alignment`.
    fn aligned(&self, alignment: u64) -> bool {
        (self.pointer.as_ptr() as u64).is_multiple_of(alignment)
    }

    /// The mapping's bytes.
    fn bytes(&self) -> &[u8] {
        // SAFETY: the mapping's bytes, which no GPU command writes while they are read here.
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.length) }
    }

    /// Write `bytes` at the start of the mapping.
    fn write(&self, bytes: &[u8]) {
        assert!(bytes.len() <= self.length);
        // SAFETY: within the mapping, which no GPU command reads while it is written here.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.pointer.as_ptr(), bytes.len())
        };
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: the mapping made in `map`, which no imported memory holds any more (the probe's
        // device, which frees its imports, drops first).
        unsafe { libc::munmap(self.pointer.as_ptr().cast(), self.length) };
    }
}

/// The probe's own `VkDevice` and what it made, destroyed as it drops.
struct Probe {
    device: ash::Device,
    queue: vk::Queue,
    host: ash::ext::external_memory_host::Device,
    pool: vk::CommandPool,
    buffers: Vec<vk::Buffer>,
    memories: Vec<vk::DeviceMemory>,
    fence: vk::Fence,
}

impl Drop for Probe {
    fn drop(&mut self) {
        // SAFETY: the probe's own objects, which no command uses once the device is idle.
        unsafe {
            let _ = self.device.device_wait_idle();
            for &buffer in &self.buffers {
                self.device.destroy_buffer(buffer, None);
            }
            for &memory in &self.memories {
                self.device.free_memory(memory, None);
            }
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_device(None);
        }
    }
}

impl Probe {
    /// A buffer of `size` bytes, `flags` and `usage`, naming the host-allocation handle type where
    /// `external`, kept until the probe drops: with its memory requirements.
    fn buffer(
        &mut self,
        (flags, usage, size): (vk::BufferCreateFlags, vk::BufferUsageFlags, u64),
        external: bool,
    ) -> Result<(vk::Buffer, vk::MemoryRequirements), vk::Result> {
        let mut handles = vk::ExternalMemoryBufferCreateInfo::default().handle_types(HOST);
        let mut info = vk::BufferCreateInfo::default()
            .flags(flags)
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        if external {
            info = info.push_next(&mut handles);
        }
        // SAFETY: the probe's device enables the features `flags` needs (the caller's), and the
        // handle type is one the device imports into such a buffer where `external` (the caller's).
        let buffer = unsafe { self.device.create_buffer(&info, None) }?;
        self.buffers.push(buffer);
        // SAFETY: the buffer is the device's.
        let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        Ok((buffer, requirements))
    }

    /// The memory types `vkGetMemoryHostPointerPropertiesEXT` answers for the start of `mapping`.
    fn host_types(&self, mapping: &Mapping) -> Result<u32, vk::Result> {
        let mut properties = vk::MemoryHostPointerPropertiesEXT::default();
        // SAFETY: the pointer is host memory of this process, aligned to
        // `minImportedHostPointerAlignment` (the caller's).
        let result = unsafe {
            (self.host.fp().get_memory_host_pointer_properties_ext)(
                self.device.handle(),
                HOST,
                mapping.pointer.as_ptr().cast_const().cast::<c_void>(),
                &mut properties,
            )
        };
        result.result_with_success(properties.memory_type_bits)
    }

    /// Memory of type `index` imported from `mapping`'s bytes, kept until the probe drops.
    fn import(&mut self, mapping: &Mapping, index: u32) -> Result<vk::DeviceMemory, vk::Result> {
        let mut import = vk::ImportMemoryHostPointerInfoEXT::default()
            .handle_type(HOST)
            .host_pointer(mapping.pointer.as_ptr().cast::<c_void>());
        let info = vk::MemoryAllocateInfo::default()
            .allocation_size(mapping.length as u64)
            .memory_type_index(index)
            .push_next(&mut import);
        // SAFETY: the pointer and the length are multiples of `minImportedHostPointerAlignment`, the
        // type is one the pointer's properties answered (the caller's), and the mapping outlives
        // the memory (it drops after the probe).
        let memory = unsafe { self.device.allocate_memory(&info, None) }?;
        self.memories.push(memory);
        Ok(memory)
    }

    /// Memory of type `index` of `size` bytes of the device's own, kept until the probe drops,
    /// mapped.
    fn own(&mut self, size: u64, index: u32) -> Result<(vk::DeviceMemory, *mut u8), vk::Result> {
        let info = vk::MemoryAllocateInfo::default()
            .allocation_size(size)
            .memory_type_index(index);
        // SAFETY: a plain allocation of a host-visible type (the caller's).
        let memory = unsafe { self.device.allocate_memory(&info, None) }?;
        self.memories.push(memory);
        // SAFETY: the memory is host-visible, mapped once, whole.
        let mapped = unsafe {
            self.device
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        }?;
        Ok((memory, mapped.cast()))
    }

    /// Wait for the probe's fence, then reset it.
    fn wait(&self) -> Result<(), vk::Result> {
        let nanos = WAIT.as_nanos() as u64;
        // SAFETY: the fence is the device's.
        unsafe {
            self.device.wait_for_fences(&[self.fence], true, nanos)?;
            self.device.reset_fences(&[self.fence])
        }
    }

    /// Copy `size` bytes from `source` to `destination` on the GPU, the writes then made visible
    /// to the host, and wait for it.
    fn copy(
        &self,
        source: vk::Buffer,
        destination: vk::Buffer,
        size: u64,
    ) -> Result<(), vk::Result> {
        let allocate = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: the pool is the device's; the command buffer is recorded, submitted and waited
        // for here, and freed after.
        unsafe {
            let command_buffer = self.device.allocate_command_buffers(&allocate)?[0];
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device.begin_command_buffer(command_buffer, &begin)?;
            let region = vk::BufferCopy::default().size(size);
            self.device
                .cmd_copy_buffer(command_buffer, source, destination, &[region]);
            let visible = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ);
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[visible],
                &[],
                &[],
            );
            self.device.end_command_buffer(command_buffer)?;
            let command_buffers = [command_buffer];
            let submit = vk::SubmitInfo::default().command_buffers(&command_buffers);
            let submitted = self.device.queue_submit(self.queue, &[submit], self.fence);
            let waited = submitted.and_then(|()| self.wait());
            self.device
                .free_command_buffers(self.pool, &command_buffers);
            waited
        }
    }
}

/// The memory types of `bits` with their properties and heaps, printed.
fn types(memory: &vk::PhysicalDeviceMemoryProperties, bits: u32) -> String {
    let named: Vec<String> = (0..memory.memory_type_count)
        .filter(|i| bits & (1 << i) != 0)
        .map(|i| {
            let t = memory.memory_types[i as usize];
            let heap = memory.memory_heaps[t.heap_index as usize];
            format!(
                "{i} properties {:#x} heap {} (flags {:#x}, {} MiB)",
                t.property_flags.as_raw(),
                t.heap_index,
                heap.flags.as_raw(),
                heap.size >> 20
            )
        })
        .collect();
    format!("{bits:#x} [{}]", named.join("; "))
}

#[test]
fn a_sparse_buffer_over_imported_host_pages() {
    let scratch = Scratch::new("sparse-host");
    let runtime = device(&scratch.0);
    let name = runtime.name().to_owned();
    // SAFETY: the handles are read while the device lives; the probe's `VkDevice` is its own.
    let handles = unsafe { runtime.raw() };
    let (instance, physical) = (handles.instance, handles.physical_device);
    // SAFETY: the physical device is the instance's.
    let (features, properties, memory, families, listed) = unsafe {
        (
            instance.get_physical_device_features(physical),
            instance.get_physical_device_properties(physical),
            instance.get_physical_device_memory_properties(physical),
            instance.get_physical_device_queue_family_properties(physical),
            instance.enumerate_device_extension_properties(physical),
        )
    };
    let sparse = properties.sparse_properties;
    say(
        &name,
        format!(
            "sparseBinding {} sparseResidencyBuffer {} sparseResidencyAliased {}; \
             sparseAddressSpaceSize {:#x}; residencyStandard2DBlockShape {} \
             residencyStandard2DMultisampleBlockShape {} residencyStandard3DBlockShape {} \
             residencyAlignedMipSize {} residencyNonResidentStrict {}",
            features.sparse_binding,
            features.sparse_residency_buffer,
            features.sparse_residency_aliased,
            properties.limits.sparse_address_space_size,
            sparse.residency_standard2_d_block_shape,
            sparse.residency_standard2_d_multisample_block_shape,
            sparse.residency_standard3_d_block_shape,
            sparse.residency_aligned_mip_size,
            sparse.residency_non_resident_strict,
        ),
    );
    let host_listed = listed.unwrap().iter().any(|extension| {
        extension.extension_name_as_c_str() == Ok(ash::ext::external_memory_host::NAME)
    });
    let mut host_properties = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
    if host_listed {
        let mut all = vk::PhysicalDeviceProperties2::default().push_next(&mut host_properties);
        // SAFETY: as above; the extension is listed, so its structure is known to the device.
        unsafe { instance.get_physical_device_properties2(physical, &mut all) };
    }
    let alignment = host_properties.min_imported_host_pointer_alignment;
    say(
        &name,
        format!(
            "VK_EXT_external_memory_host {host_listed}, minImportedHostPointerAlignment \
             {alignment:#x}"
        ),
    );
    if features.sparse_binding == vk::FALSE || !host_listed || alignment == 0 {
        say(
            &name,
            "no sparse binding or no host-pointer import: nothing more to probe",
        );
        return;
    }
    let copies = vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER;
    let Some(family) = families.iter().position(|f| {
        f.queue_flags.contains(vk::QueueFlags::SPARSE_BINDING) && f.queue_flags.intersects(copies)
    }) else {
        say(
            &name,
            "no queue family both binds sparsely and copies: nothing more to probe",
        );
        return;
    };
    let residency = features.sparse_residency_buffer;
    let enabled = vk::PhysicalDeviceFeatures {
        sparse_binding: vk::TRUE,
        sparse_residency_buffer: residency,
        ..vk::PhysicalDeviceFeatures::default()
    };
    let priorities = [1.0];
    let queues = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(family as u32)
        .queue_priorities(&priorities)];
    let extensions = [ash::ext::external_memory_host::NAME.as_ptr()];
    let info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queues)
        .enabled_extension_names(&extensions)
        .enabled_features(&enabled);
    // The mappings outlive the probe's device, which frees the memory imported from them.
    let mut mappings: Vec<Mapping> = Vec::new();
    // SAFETY: the physical device is the instance's; the features and the extension are offered.
    let raw = unsafe { instance.create_device(physical, &info, None) }.unwrap();
    // SAFETY: the device is live; the pool and the fence are its own, destroyed as the probe drops.
    let (queue, pool, fence) = unsafe {
        let pool_info = vk::CommandPoolCreateInfo::default().queue_family_index(family as u32);
        (
            raw.get_device_queue(family as u32, 0),
            raw.create_command_pool(&pool_info, None).unwrap(),
            raw.create_fence(&vk::FenceCreateInfo::default(), None)
                .unwrap(),
        )
    };
    let mut probe = Probe {
        host: ash::ext::external_memory_host::Device::new(instance, &raw),
        device: raw,
        queue,
        pool,
        buffers: Vec::new(),
        memories: Vec::new(),
        fence,
    };
    say(
        &name,
        format!(
            "family {family} (queue flags {:#x})",
            families[family].queue_flags.as_raw()
        ),
    );
    // The host pointer's types: a page of a memfd mapped shared, a page of anonymous memory.
    let unit = lcm(page(), alignment) as usize;
    let mut host_bits = 0;
    for (kind, mapping) in [
        ("a memfd mapped shared", Mapping::memfd(unit)),
        ("anonymous memory", Mapping::anonymous(unit)),
    ] {
        let answer = match mapping.aligned(alignment) {
            true => probe.host_types(&mapping),
            false => Err(vk::Result::ERROR_UNKNOWN),
        };
        match answer {
            Ok(bits) => {
                say(
                    &name,
                    format!("host pointer of {kind}: types {}", types(&memory, bits)),
                );
                if kind.starts_with("a memfd") {
                    host_bits = bits;
                }
            }
            Err(error) => say(&name, format!("host pointer of {kind}: {error:?}")),
        }
    }
    let mut variants = vec![("binding", vk::BufferCreateFlags::SPARSE_BINDING)];
    if residency == vk::TRUE {
        let both = vk::BufferCreateFlags::SPARSE_BINDING | vk::BufferCreateFlags::SPARSE_RESIDENCY;
        variants.push(("binding+residency", both));
    }
    for (variant, flags) in variants {
        let mut transfer_bits = None;
        for (use_name, usage) in USES {
            let query = vk::PhysicalDeviceExternalBufferInfo::default()
                .flags(flags)
                .usage(usage)
                .handle_type(HOST);
            let mut external = vk::ExternalBufferProperties::default();
            // SAFETY: the physical device is the instance's (Vulkan 1.1).
            unsafe {
                instance.get_physical_device_external_buffer_properties(
                    physical,
                    &query,
                    &mut external,
                )
            };
            let props = external.external_memory_properties;
            let importable = props
                .external_memory_features
                .contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)
                && props.compatible_handle_types.contains(HOST);
            let size = LEAST_CHUNK * CHUNKS as u64;
            let plain = probe.buffer((flags, usage, size), false);
            let named = match importable {
                true => Some(probe.buffer((flags, usage, size), true)),
                false => None,
            };
            let shown = |made: &Result<(vk::Buffer, vk::MemoryRequirements), vk::Result>| match made
            {
                Ok((_, r)) => format!(
                    "types {} alignment {:#x} size {:#x}",
                    types(&memory, r.memory_type_bits),
                    r.alignment,
                    r.size
                ),
                Err(error) => format!("{error:?}"),
            };
            let with = match &named {
                Some(made) => {
                    let bits = made.as_ref().map(|(_, r)| r.memory_type_bits).unwrap_or(0);
                    if use_name == "transfer" {
                        transfer_bits = Some(bits);
                    }
                    format!(
                        "{}; with the host handle type {}; intersection with the memfd \
                         pointer's {}",
                        shown(&plain),
                        shown(made),
                        types(&memory, bits & host_bits)
                    )
                }
                None => shown(&plain),
            };
            say(
                &name,
                format!(
                    "{variant} {use_name}: host import features {:#x} compatible {:#x} \
                     (importable {importable}); {with}",
                    props.external_memory_features.as_raw(),
                    props.compatible_handle_types.as_raw()
                ),
            );
        }
        // The round trip, through a transfer buffer naming the host handle type.
        let Some(bits) = transfer_bits else {
            say(
                &name,
                format!("{variant}: no host import into such a buffer: no bind"),
            );
            continue;
        };
        let common = bits & host_bits;
        if common == 0 {
            say(
                &name,
                format!("{variant}: the intersection is empty: no bind"),
            );
            continue;
        }
        let outcome = round_trip(
            &mut probe,
            &mut mappings,
            (flags, common, alignment),
            &memory,
        );
        say(&name, format!("{variant}: {outcome}"));
    }
    drop(probe);
    drop(mappings);
    drop(runtime);
}

/// Import [`CHUNKS`] separate memfd mappings into memory of a type of `common`, bind them in
/// reverse order into one sparse buffer of `flags` naming the host handle type, copy a pattern in
/// on the GPU and read it through the mappings, then write another through the mappings and copy
/// it out: what happened, printed.
fn round_trip(
    probe: &mut Probe,
    mappings: &mut Vec<Mapping>,
    (flags, common, alignment): (vk::BufferCreateFlags, u32, u64),
    memory: &vk::PhysicalDeviceMemoryProperties,
) -> String {
    let coherent = vk::MemoryPropertyFlags::HOST_COHERENT;
    let has = |i: u32, f: vk::MemoryPropertyFlags| {
        memory.memory_types[i as usize].property_flags.contains(f)
    };
    let Some(index) =
        (0..memory.memory_type_count).find(|&i| common & (1 << i) != 0 && has(i, coherent))
    else {
        return format!(
            "no host-coherent type among {}: no bind",
            types(memory, common)
        );
    };
    // The buffer's alignment, from a buffer of the least chunks.
    let least = (flags, TRANSFER, LEAST_CHUNK * CHUNKS as u64);
    let buffer_alignment = match probe.buffer(least, true) {
        Ok((_, r)) => r.alignment,
        Err(error) => return format!("the sparse buffer: {error:?}"),
    };
    let chunk = lcm(lcm(lcm(LEAST_CHUNK, buffer_alignment), alignment), page());
    let size = chunk * CHUNKS as u64;
    let (buffer, requirements) = match probe.buffer((flags, TRANSFER, size), true) {
        Ok(made) => made,
        Err(error) => return format!("the sparse buffer of {size:#x} bytes: {error:?}"),
    };
    if !chunk.is_multiple_of(requirements.alignment) || requirements.size > size {
        return format!(
            "the buffer of {size:#x} bytes takes alignment {:#x} and size {:#x}: no bind",
            requirements.alignment, requirements.size
        );
    }
    let first = mappings.len();
    for _ in 0..CHUNKS {
        mappings.push(Mapping::memfd(chunk as usize));
    }
    let chunks = &mappings[first..];
    if let Some(at) = chunks.iter().position(|m| !m.aligned(alignment)) {
        return format!("memfd mapping {at} lies off {alignment:#x}: no import");
    }
    let mut imported = Vec::with_capacity(CHUNKS);
    for (at, mapping) in chunks.iter().enumerate() {
        match probe.import(mapping, index) {
            Ok(m) => imported.push(m),
            Err(error) => return format!("import of memfd chunk {at} as type {index}: {error:?}"),
        }
    }
    // Slot `s` of the buffer is chunk `CHUNKS - 1 - s`.
    let binds: Vec<vk::SparseMemoryBind> = (0..CHUNKS)
        .map(|slot| {
            vk::SparseMemoryBind::default()
                .resource_offset(slot as u64 * chunk)
                .size(chunk)
                .memory(imported[CHUNKS - 1 - slot])
                .memory_offset(0)
        })
        .collect();
    let buffer_binds = [vk::SparseBufferMemoryBindInfo::default()
        .buffer(buffer)
        .binds(&binds)];
    let bind = vk::BindSparseInfo::default().buffer_binds(&buffer_binds);
    // SAFETY: the queue binds sparsely; the buffer and the memory are the device's, every
    // offset and size a multiple of the buffer's alignment, the memory's type the buffer's, and
    // the buffer names the memory's handle type.
    let bound = unsafe {
        probe
            .device
            .queue_bind_sparse(probe.queue, &[bind], probe.fence)
    }
    .and_then(|()| probe.wait());
    if let Err(error) = bound {
        return format!(
            "{CHUNKS} memfd chunks of {chunk:#x} bytes imported as type {index}; \
             vkQueueBindSparse: {error:?}"
        );
    }
    // A staging buffer of the device's own host-visible, host-coherent memory.
    let (staging, staged) =
        match probe.buffer((vk::BufferCreateFlags::empty(), TRANSFER, size), false) {
            Ok(made) => made,
            Err(error) => return format!("bound; the staging buffer: {error:?}"),
        };
    let visible = vk::MemoryPropertyFlags::HOST_VISIBLE | coherent;
    let Some(own_index) = (0..memory.memory_type_count)
        .find(|&i| staged.memory_type_bits & (1 << i) != 0 && has(i, visible))
    else {
        return "bound; no host-visible coherent type for the staging buffer".to_owned();
    };
    let (own, mapped) = match probe.own(staged.size, own_index) {
        Ok(made) => made,
        Err(error) => return format!("bound; the staging memory: {error:?}"),
    };
    // SAFETY: the staging buffer and its memory are the device's, unbound until here.
    if let Err(error) = unsafe { probe.device.bind_buffer_memory(staging, own, 0) } {
        return format!("bound; the staging bind: {error:?}");
    }
    let length = size as usize;
    // SAFETY: the staging memory's mapping covers `size` bytes; no GPU command uses it now.
    let staged_bytes = unsafe { std::slice::from_raw_parts_mut(mapped, length) };
    // In: the GPU copies pattern 1 into the buffer; each chunk holds its slot's bytes.
    for (i, byte) in staged_bytes.iter_mut().enumerate() {
        *byte = pattern(i, 1);
    }
    if let Err(error) = probe.copy(staging, buffer, size) {
        return format!("bound; the copy into the buffer: {error:?}");
    }
    let c = chunk as usize;
    let into: Vec<usize> = (0..CHUNKS)
        .filter(|&slot| {
            *chunks[CHUNKS - 1 - slot].bytes() == staged_bytes[slot * c..(slot + 1) * c]
        })
        .collect();
    // Out: the host writes pattern 2 through each chunk's mapping; the GPU copies the buffer out.
    for slot in 0..CHUNKS {
        let bytes: Vec<u8> = (slot * c..(slot + 1) * c).map(|i| pattern(i, 2)).collect();
        chunks[CHUNKS - 1 - slot].write(&bytes);
    }
    staged_bytes.fill(0);
    if let Err(error) = probe.copy(buffer, staging, size) {
        return format!("bound; the copy out of the buffer: {error:?}");
    }
    let out: Vec<usize> = (0..CHUNKS)
        .filter(|&slot| (slot * c..(slot + 1) * c).all(|i| staged_bytes[i] == pattern(i, 2)))
        .collect();
    format!(
        "{CHUNKS} memfd chunks of {chunk:#x} bytes imported as type {index}, bound in reverse \
         order: vkQueueBindSparse succeeded; the GPU's copy in reached the chunks of slots \
         {into:?}, the host's writes copied out matched in slots {out:?} (of 0..{CHUNKS})"
    )
}
