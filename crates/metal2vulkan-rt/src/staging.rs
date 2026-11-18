//! Staging: the bytes a copy between a texture and the CPU passes through.
//!
//! The device has one staging region, a Shared block of the pool taken at the first copy and kept
//! for the device's life, so no copy allocates memory and the host writes pages already resident.
//! It is a ring: each copy takes a span, which the host writes before the copy is recorded
//! (`replace_region`) or reads once the copy has completed (`get_bytes`). A span is free again
//! once the submission that carries its copy has completed and no host reads it. The next span
//! goes after the last one taken, or back at the start where the end has no room and the start is
//! free; with no span live the ring starts again at 0. Where no room is free, the caller waits for
//! the oldest span's submission ([`Work::stage`](crate::submit::Work::stage)). A copy larger than
//! the region has a staging buffer of its own, which the upload's bound counts.

use crate::counters::Gauge;
use crate::device::Core;
use crate::error::Error;
use crate::memory::{self, Mapping, Memory, MemoryClass};
use crate::reaper::{Free, Job};
use ash::vk;
use std::collections::VecDeque;
use std::ptr::NonNull;
use std::sync::Arc;

/// The alignment of every span: a multiple of every texel block's bytes, of the 4 a depth or
/// stencil copy needs, and of every device's optimal copy offset alignment.
pub(crate) const SPAN_ALIGN: u64 = 256;

/// The ring's arithmetic: its spans, oldest first, each with the progress value at which it is
/// free.
pub(crate) struct Ring {
    len: u64,
    /// The end of the last span taken.
    head: u64,
    spans: VecDeque<Span>,
    /// The number of the front span; spans are numbered in the order they were taken.
    front: u64,
}

struct Span {
    start: u64,
    end: u64,
    at: At,
    /// A host reads its bytes ([`Texture::get_bytes`](crate::Texture::get_bytes)).
    read: bool,
}

/// Where a span's copy is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum At {
    /// Taken: its bytes are being written, or its copy is being recorded.
    Taken,
    /// Its copy is in the pending upload.
    Recorded,
    /// Its copy is in the submission of this progress value.
    Submitted(u64),
    /// Given up with no copy recorded.
    Done,
}

/// What frees the ring's oldest span, for a caller that finds no room.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Oldest {
    /// The submission of this progress value completing.
    Submitted(u64),
    /// The pending upload going, and then completing.
    Pending,
    /// Another thread finishing its call: writing the span's bytes, or reading them.
    Busy,
}

impl Ring {
    pub(crate) fn new(len: u64) -> Ring {
        Ring {
            len,
            head: 0,
            spans: VecDeque::new(),
            front: 0,
        }
    }

    pub(crate) fn len(&self) -> u64 {
        self.len
    }

    /// Frees the spans at the front that are done: given up, or carried by a submission at or
    /// before `completed` with no host reading them.
    pub(crate) fn reclaim(&mut self, completed: u64) {
        while let Some(span) = self.spans.front() {
            let free = !span.read
                && match span.at {
                    At::Done => true,
                    At::Submitted(value) => value <= completed,
                    At::Taken | At::Recorded => false,
                };
            if !free {
                break;
            }
            self.spans.pop_front();
            self.front += 1;
        }
        if self.spans.is_empty() {
            self.head = 0;
        }
    }

    /// A span of `len` bytes (at most the ring's) at a multiple of [`SPAN_ALIGN`]: its number and
    /// its offset, where there is room.
    pub(crate) fn take(&mut self, len: u64, read: bool) -> Option<(u64, u64)> {
        debug_assert!(len <= self.len);
        let after = self.head.next_multiple_of(SPAN_ALIGN);
        let start = match self.spans.front().map(|span| span.start) {
            None => 0,
            // The live spans lie in `tail..head`: after them, else from the start.
            Some(tail) if self.head > tail => match after.checked_add(len) {
                Some(end) if end <= self.len => after,
                _ if len <= tail => 0,
                _ => return None,
            },
            // The live spans wrap: the room is `head..tail`.
            Some(tail) => match after.checked_add(len) {
                Some(end) if end <= tail => after,
                _ => return None,
            },
        };
        self.spans.push_back(Span {
            start,
            end: start + len,
            at: At::Taken,
            read,
        });
        self.head = start + len;
        Some((self.front + self.spans.len() as u64 - 1, start))
    }

    fn span(&mut self, number: u64) -> Option<&mut Span> {
        let index = number.checked_sub(self.front)?;
        self.spans.get_mut(usize::try_from(index).ok()?)
    }

    /// Moves span `number` on to `at`. Nothing for a span no longer live.
    pub(crate) fn set(&mut self, number: u64, at: At) {
        if let Some(span) = self.span(number) {
            span.at = at;
        }
    }

    /// The host no longer reads span `number`.
    pub(crate) fn unread(&mut self, number: u64) {
        if let Some(span) = self.span(number) {
            span.read = false;
        }
    }

    /// What frees the oldest span; `None` with no span live.
    pub(crate) fn oldest(&self) -> Option<Oldest> {
        let span = self.spans.front()?;
        Some(match span.at {
            At::Submitted(value) if !span.read => Oldest::Submitted(value),
            At::Recorded => Oldest::Pending,
            At::Taken | At::Submitted(_) | At::Done => Oldest::Busy,
        })
    }

    /// For the tests: every live span's offsets and where its copy is.
    #[cfg(test)]
    pub(crate) fn live(&self) -> Vec<(u64, u64, At)> {
        (self.spans.iter())
            .map(|span| (span.start, span.end, span.at))
            .collect()
    }
}

/// The device's staging region: a Shared block of the pool, its mapping, and its ring.
pub(crate) struct Region {
    pub(crate) memory: Memory,
    pub(crate) host: Mapping,
    pub(crate) ring: Ring,
}

impl Region {
    /// The region over `memory`, a Shared block mapped at `host`.
    pub(crate) fn new(memory: Memory, host: Mapping) -> Region {
        Region {
            ring: Ring::new(memory.len),
            memory,
            host,
        }
    }
}

/// Where a copy's bytes are staged.
pub(crate) enum Stage {
    /// Span `number` of the region, at `offset` in its buffer.
    Region {
        number: u64,
        buffer: vk::Buffer,
        offset: u64,
        host: Mapping,
    },
    /// A staging buffer of its own, for a copy larger than the region.
    Own(Arc<Staging>),
}

impl Stage {
    /// The buffer the copy reads or writes.
    pub(crate) fn buffer(&self) -> vk::Buffer {
        match self {
            Stage::Region { buffer, .. } => *buffer,
            Stage::Own(staging) => staging.buffer,
        }
    }

    /// The copy's offset in [`buffer`](Self::buffer).
    pub(crate) fn offset(&self) -> u64 {
        match self {
            Stage::Region { offset, .. } => *offset,
            Stage::Own(_) => 0,
        }
    }

    /// The host address of the copy's first byte.
    pub(crate) fn host(&self) -> NonNull<u8> {
        match self {
            Stage::Region { host, .. } => host.get(),
            Stage::Own(staging) => staging.mapping.get(),
        }
    }
}

/// A host-visible buffer of a copy's bytes, in memory of its own of the Shared type, mapped for
/// its life, for a copy larger than the region. The reaper frees it once its last holder, the
/// upload that copies through it among them, is gone.
pub(crate) struct Staging {
    core: Arc<Core>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapping: Mapping,
    pub(crate) len: u64,
}

impl Staging {
    /// A staging buffer of `len` bytes.
    pub(crate) fn new(core: &Arc<Core>, len: u64) -> Result<Staging, Error> {
        let device = &core.device;
        let info = vk::BufferCreateInfo::default()
            .size(len)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: the device is live.
        let buffer = unsafe { device.create_buffer(&info, None) }
            .map_err(Error::vulkan("vkCreateBuffer"))?;
        // SAFETY: the buffer is the device's.
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let allocate = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(core.memory_type(MemoryClass::Shared));
        // SAFETY: the Shared type serves a buffer of every usage, so one of fewer too.
        let allocated =
            unsafe { memory::allocate(device, &core.counters, core.max_allocations, &allocate) };
        let memory = match allocated {
            Ok(memory) => memory,
            Err(error) => {
                // SAFETY: the buffer is unused.
                unsafe { device.destroy_buffer(buffer, None) };
                return Err(error);
            }
        };
        // SAFETY: the allocation is the buffer's size, of a type it allows; the memory is
        // host-visible and not mapped.
        let mapped = unsafe {
            device
                .bind_buffer_memory(buffer, memory, 0)
                .map_err(Error::vulkan("vkBindBufferMemory"))
                .and_then(|()| {
                    device
                        .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                        .map_err(Error::vulkan("vkMapMemory"))
                })
        };
        let pointer = match mapped {
            Ok(pointer) => pointer,
            Err(error) => {
                // SAFETY: nothing uses them.
                unsafe {
                    device.destroy_buffer(buffer, None);
                    device.free_memory(memory, None);
                }
                core.counters.fall(Gauge::Blocks);
                return Err(error);
            }
        };
        // A successful map gives a non-null address.
        let Some(pointer) = NonNull::new(pointer.cast::<u8>()) else {
            // SAFETY: nothing uses them.
            unsafe {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
            }
            core.counters.fall(Gauge::Blocks);
            return Err(Error::Vulkan {
                call: "vkMapMemory",
                result: vk::Result::ERROR_MEMORY_MAP_FAILED,
            });
        };
        Ok(Staging {
            core: Arc::clone(core),
            buffer,
            memory,
            mapping: Mapping::new(pointer),
            len,
        })
    }
}

impl Drop for Staging {
    fn drop(&mut self) {
        self.core.reaper.give(Job::Free(Free::Staging {
            buffer: self.buffer,
            memory: self.memory,
        }));
    }
}
