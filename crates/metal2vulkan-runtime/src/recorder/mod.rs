//! The door every command a `VkCommandBuffer` takes goes through, counted by kind as it is
//! recorded. The encoders record into the driver's command buffers at each call, on the program's
//! thread, as the scheduler's thread records its batches' own; this module only counts.
//!
//! - **One door.** The device's functions hold this module's entry point in place of the driver's
//!   for each command a `VkCommandBuffer` takes ([`Command`]), its begin and its end among them. A
//!   device the runtime makes or adopts takes one of [`SLOTS`] slots ([`route`]): its functions
//!   (`device/functions.rs`, the exit guard's table among them) and the extension commands it loads
//!   ([`Slot::route`]) hold the slot's entry points. Each is generic over its slot, so it finds its
//!   device's command with one load, without a lock or a lookup by handle; a slot is let go of
//!   with its device, after `vkDestroyDevice`. A source test fails when a file outside this module
//!   names a command that is not routed.
//! - **Counted by kind.** Each command recorded through a slot counts once for its kind, one
//!   relaxed increment into a fixed array of the slot's indexed by the command, whatever the
//!   command buffer and the thread (`WorkCounts::commands_replayed`). A device past the slots
//!   records with its driver's commands, uncounted.

mod thunks;

use ash::vk;
use std::ffi::{c_void, CStr};
use std::sync::atomic::{AtomicBool, Ordering};

pub(crate) use thunks::Command;
pub use thunks::CommandsReplayed;

/// How many devices route their commands here at once; one made past them keeps its driver's.
pub(crate) const SLOTS: usize = 8;

/// The slots taken.
static TAKEN: [AtomicBool; SLOTS] = [const { AtomicBool::new(false) }; SLOTS];

/// Run `$body` with `$s` the const slot index `$index` names.
macro_rules! per_slot {
    ($index:expr, $s:ident => $body:expr) => {
        match $index {
            0 => {
                const $s: usize = 0;
                $body
            }
            1 => {
                const $s: usize = 1;
                $body
            }
            2 => {
                const $s: usize = 2;
                $body
            }
            3 => {
                const $s: usize = 3;
                $body
            }
            4 => {
                const $s: usize = 4;
                $body
            }
            5 => {
                const $s: usize = 5;
                $body
            }
            6 => {
                const $s: usize = 6;
                $body
            }
            _ => {
                const $s: usize = 7;
                $body
            }
        }
    };
}

/// One device's slot (the module docs): its commands, until it is dropped, after the device is
/// destroyed.
pub(crate) struct Slot {
    index: usize,
}

impl Slot {
    /// `real`, the device's command `command` as its driver gave it, routed through the slot:
    /// the slot's entry point, which calls `real`.
    ///
    /// # Safety
    ///
    /// `real` is a function pointer of `command`'s signature, the slot's device's, and not one
    /// of the slot's entry points (a function's address is no identity to compare against, so
    /// the caller knows which it holds: a command of the device's routed functions is routed).
    pub(crate) unsafe fn route<F: Copy>(&self, command: Command, real: F) -> F {
        assert_eq!(
            std::mem::size_of::<F>(),
            std::mem::size_of::<*const c_void>()
        );
        // SAFETY: `F` is a function pointer (the caller's), of a pointer's size.
        let pointer = unsafe { std::mem::transmute_copy::<F, *const c_void>(&real) };
        thunks::store(self.index, command, pointer);
        let thunk = per_slot!(self.index, S => thunks::thunk::<S>(command));
        // SAFETY: the entry point has `command`'s signature, `F`'s (the caller's).
        unsafe { std::mem::transmute_copy::<*const c_void, F>(&thunk) }
    }

    /// The commands recorded through the slot so far, by kind (the module docs).
    pub(crate) fn counts(&self) -> CommandsReplayed {
        thunks::counts(self.index)
    }

    /// The slot's entry point in place of the driver's `name`, if it routes that command: the exit
    /// guard's table (`device/functions.rs`) holds it too.
    pub(crate) fn entry(&self, name: &CStr) -> Option<*const c_void> {
        let command = Command::ALL.iter().find(|command| command.name() == name)?;
        Some(per_slot!(self.index, S => thunks::thunk::<S>(*command)))
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        thunks::clear(self.index);
        TAKEN[self.index].store(false, Ordering::Release);
    }
}

/// `device`'s functions with the entry points of a slot of its own in place of its driver's
/// commands, and the slot; the functions as they are, and no slot, where every slot is taken.
///
/// # Safety
///
/// `device` is live, and its functions are its driver's (or its topmost layer's, or routed for
/// its pipeline makings, `compiles.rs`), not already routed here.
pub(crate) unsafe fn route(device: ash::Device) -> (ash::Device, Option<Slot>) {
    let Some(index) = TAKEN.iter().position(|taken| {
        taken
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
    }) else {
        return (device, None);
    };
    let slot = Slot { index };
    thunks::clear_counts(index);
    let mut v1_0 = device.fp_v1_0().clone();
    let mut v1_3 = device.fp_v1_3().clone();
    per_slot!(index, S => thunks::route_tables::<S>(&mut v1_0, &mut v1_3));
    let routed = ash::Device::from_parts_1_3(
        device.handle(),
        v1_0,
        device.fp_v1_1().clone(),
        device.fp_v1_2().clone(),
        v1_3,
    );
    (routed, Some(slot))
}

/// `real`, a device's command `command`, through `slot` where the device has one
/// ([`Slot::route`]), else as it is.
///
/// # Safety
///
/// As [`Slot::route`].
pub(crate) unsafe fn through<F: Copy>(slot: Option<&Slot>, command: Command, real: F) -> F {
    match slot {
        // SAFETY: the caller's.
        Some(slot) => unsafe { slot.route(command, real) },
        None => real,
    }
}

/// The extension commands of `VK_EXT_mesh_shader` a draw records, routed (`render.rs`).
#[derive(Clone, Copy)]
pub(crate) struct MeshDraws {
    draw: vk::PFN_vkCmdDrawMeshTasksEXT,
    indirect: vk::PFN_vkCmdDrawMeshTasksIndirectEXT,
    indirect_count: vk::PFN_vkCmdDrawMeshTasksIndirectCountEXT,
}

impl MeshDraws {
    /// The commands of `loaded`, through `slot` where the device has one.
    pub(crate) fn new(loaded: &ash::ext::mesh_shader::Device, slot: Option<&Slot>) -> MeshDraws {
        let functions = loaded.fp();
        let (draw, indirect, indirect_count) = (
            functions.cmd_draw_mesh_tasks_ext,
            functions.cmd_draw_mesh_tasks_indirect_ext,
            functions.cmd_draw_mesh_tasks_indirect_count_ext,
        );
        match slot {
            // SAFETY: each is the device's command of that name.
            Some(slot) => unsafe {
                MeshDraws {
                    draw: slot.route(Command::CmdDrawMeshTasks, draw),
                    indirect: slot.route(Command::CmdDrawMeshTasksIndirect, indirect),
                    indirect_count: slot
                        .route(Command::CmdDrawMeshTasksIndirectCount, indirect_count),
                }
            },
            None => MeshDraws {
                draw,
                indirect,
                indirect_count,
            },
        }
    }

    /// `vkCmdDrawMeshTasksEXT`.
    ///
    /// # Safety
    ///
    /// As the command's.
    pub(crate) unsafe fn cmd_draw_mesh_tasks(
        &self,
        command_buffer: vk::CommandBuffer,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    ) {
        // SAFETY: the caller's.
        unsafe { (self.draw)(command_buffer, group_count_x, group_count_y, group_count_z) }
    }

    /// `vkCmdDrawMeshTasksIndirectEXT`.
    ///
    /// # Safety
    ///
    /// As the command's.
    pub(crate) unsafe fn cmd_draw_mesh_tasks_indirect(
        &self,
        command_buffer: vk::CommandBuffer,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        draw_count: u32,
        stride: u32,
    ) {
        // SAFETY: the caller's.
        unsafe { (self.indirect)(command_buffer, buffer, offset, draw_count, stride) }
    }

    /// `vkCmdDrawMeshTasksIndirectCountEXT`, its count in `buffer` too (the device enabled
    /// `drawIndirectCount` wherever a pipeline draws it, `bind::mesh_arguments`).
    ///
    /// # Safety
    ///
    /// As the command's.
    pub(crate) unsafe fn cmd_draw_mesh_tasks_indirect_count(
        &self,
        command_buffer: vk::CommandBuffer,
        buffer: vk::Buffer,
        (offset, count_offset): (vk::DeviceSize, vk::DeviceSize),
        max_draw_count: u32,
        stride: u32,
    ) {
        // SAFETY: the caller's.
        unsafe {
            (self.indirect_count)(
                command_buffer,
                buffer,
                offset,
                buffer,
                count_offset,
                max_draw_count,
                stride,
            )
        }
    }
}
