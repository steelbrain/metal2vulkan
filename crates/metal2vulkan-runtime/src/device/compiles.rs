//! The device's pipeline makings at the user-initiated class (a concurrency change). MoltenVK
//! compiles a pipeline's shaders on a dispatch queue that takes the class of the thread asking. A
//! pipeline made on a thread at the default class or below therefore compiles at that class in
//! Metal's compiler service, behind the host's busier threads. So each `vkCreateGraphicsPipelines`
//! and `vkCreateComputePipelines` the runtime makes on a device runs at the user-initiated quality
//! of service or above, whichever thread asks:
//!
//! - **One door.** The device's functions, as made or adopted (`Device::assemble`), hold this
//!   module's two entry points in place of the driver's ([`route`]), and so does the exit guard's
//!   table made from them (`functions.rs`, [`entry`]). Every pipeline the runtime makes on the
//!   device comes here, priming's too. Each entry point finds its device's own and calls it.
//! - **On the asking thread at that class**: a thread of the pipeline pool's (`pipeline/pool.rs`),
//!   or a program's thread at the user-initiated or user-interactive class.
//! - **Else on a thread of the runtime's own, the asking thread waiting** ([`at_user_initiated`]):
//!   a thread started for the call at the user-initiated class. It makes the one Vulkan call with
//!   the asking thread's arguments, hands back its result, and is joined before the call returns
//!   (a full join). Nothing else is done there: what to make, in what order, and what the program
//!   sees are decided on the asking thread, before and after. The asking thread's class, overrides
//!   and scheduling stay as found.
//! - **The driver's measured diagnostics.** The driver reports a message on the thread that makes
//!   the call, and the diagnostics the asking thread's call is measured to report are withheld
//!   inside that call alone (`driver_facts.rs`). So the thread takes the asking thread's
//!   expectation for its one call, and the asking thread counts what it withheld.
//! - **The program's code stays on the asking thread** (the consumer's review). The only code of
//!   the program's a driver call reaches is the device's message sink, which the driver's debug
//!   messenger calls on the thread that makes the call (`create.rs`'s `deliver`). On the runtime's
//!   thread each such message is held ([`held`]) and delivered on the asking thread once the call
//!   returns, in the order raised, before the asking thread's Vulkan call returns: the sink runs
//!   where it ran before, so a lock the program holds across the creation, which the sink takes
//!   again, does not deadlock against the full join. A messenger the program registered itself on
//!   an adopted instance is not the runtime's: the driver calls it on the thread that makes the
//!   call, the runtime's here. An asking thread that itself holds its call's messages (a
//!   self-check's, [`holding`]) keeps those its runtime thread held, for the self-check to read.
//! - **The exit guard.** The thread is one of the runtime's (`exit_guard::spawn_scoped`) and makes
//!   the call inside a section of its own. Where the host starts no thread, or once the process is
//!   exiting (the guard closed), the asking thread makes the call itself, as before. Each driver
//!   call that returned, on whichever thread made it, registers the exit hook again
//!   (`exit_guard::pipelines_made`), after the destructors the driver's compiler registered in it.
//! - **No compile outlives its call.** The runtime's thread is joined before the call returns
//!   (`std::thread::scope`). Each device counts the calls in progress here, and in builds with
//!   debug assertions the runtime's last hold of the device ([`unregister`], in the teardown just
//!   before `vkDestroyDevice`) checks that none is.
//! - **The time in the driver is counted.** Each call's nanoseconds inside the driver's entry
//!   point are added to its device's record, graphics and compute apart ([`driver_nanos`],
//!   `WorkCounts::render_pipeline_create_nanos` and `compute_pipeline_create_nanos`). The clock is
//!   read on whichever thread makes the call, just before and just after the driver's one call, so
//!   a call relayed to a thread of the runtime's is counted there and none is dropped; the relay's
//!   own thread start and join, which the asking thread pays, lie outside the window and are the
//!   runtime's.
//! - Elsewhere than macOS there is no class: every call is made on the asking thread.
//!
//! The device's creation (`create.rs`) runs `vkCreateDevice` through [`at_user_initiated`] too,
//! since MoltenVK compiles its own command shaders there. The compiles MoltenVK makes inside
//! `vkQueueSubmit` stay at their caller's class.

use crate::exit_guard::{self, Call};
use crate::{DebugMessage, MessageSink};
use ash::vk;
use std::cell::RefCell;
use std::ffi::{c_void, CStr};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, PoisonError, RwLock};

/// A device whose pipeline makings come here: its driver's entry points, the runtime's holds of
/// it, its calls in progress, and the time they spent in the driver.
struct Driver {
    device: vk::Device,
    /// The runtime's devices over it: adopting one `VkDevice` twice holds it twice.
    holds: usize,
    graphics: vk::PFN_vkCreateGraphicsPipelines,
    compute: vk::PFN_vkCreateComputePipelines,
    in_progress: Arc<AtomicUsize>,
    nanos: Arc<Nanos>,
}

/// The time a routed device's pipeline makings spent inside its driver's entry points, by call
/// ([`driver_nanos`], `WorkCounts::render_pipeline_create_nanos` and
/// `compute_pipeline_create_nanos`). One record per `VkDevice`, so a `VkDevice` the runtime holds
/// twice counts each making in both its devices' figures.
#[derive(Default)]
struct Nanos {
    graphics: AtomicU64,
    compute: AtomicU64,
}

/// Every device routed here and not yet let go ([`route`], [`unregister`]).
static DRIVERS: RwLock<Vec<Driver>> = RwLock::new(Vec::new());

/// `device`'s functions with this module's pipeline makings in place of the driver's, which it
/// keeps for the device until the runtime lets go of it ([`unregister`]).
///
/// # Safety
///
/// `device` is a live device, and its functions are its driver's (or its topmost layer's), not
/// already routed.
pub(super) unsafe fn route(device: ash::Device) -> ash::Device {
    let handle = device.handle();
    let functions = device.fp_v1_0();
    let graphics: vk::PFN_vkCreateGraphicsPipelines = create_graphics_pipelines;
    debug_assert!(
        !std::ptr::fn_addr_eq(functions.create_graphics_pipelines, graphics),
        "a device routed twice"
    );
    {
        let mut drivers = DRIVERS.write().unwrap_or_else(PoisonError::into_inner);
        match drivers.iter_mut().find(|driver| driver.device == handle) {
            // The same device again: the same entry points.
            Some(driver) => {
                driver.holds += 1;
                driver.graphics = functions.create_graphics_pipelines;
                driver.compute = functions.create_compute_pipelines;
            }
            None => drivers.push(Driver {
                device: handle,
                holds: 1,
                graphics: functions.create_graphics_pipelines,
                compute: functions.create_compute_pipelines,
                in_progress: Arc::default(),
                nanos: Arc::default(),
            }),
        }
    }
    let mut routed = functions.clone();
    routed.create_graphics_pipelines = create_graphics_pipelines;
    routed.create_compute_pipelines = create_compute_pipelines;
    ash::Device::from_parts_1_3(
        handle,
        routed,
        device.fp_v1_1().clone(),
        device.fp_v1_2().clone(),
        device.fp_v1_3().clone(),
    )
}

/// The entry point this module holds in place of the driver's `name`, if any: the exit guard's
/// table (`functions.rs`) holds it too.
pub(super) fn entry(name: &CStr) -> Option<*const c_void> {
    let graphics: vk::PFN_vkCreateGraphicsPipelines = create_graphics_pipelines;
    let compute: vk::PFN_vkCreateComputePipelines = create_compute_pipelines;
    if name == Call::CreateGraphicsPipelines.name() {
        Some(graphics as *const c_void)
    } else if name == Call::CreateComputePipelines.name() {
        Some(compute as *const c_void)
    } else {
        None
    }
}

/// The runtime lets go of one hold of `device` ([`route`]), just before it destroys the device
/// or as it stops using it; the device's record goes with its last hold.
pub(super) fn unregister(device: vk::Device) {
    let mut drivers = DRIVERS.write().unwrap_or_else(PoisonError::into_inner);
    let Some(at) = drivers.iter().position(|driver| driver.device == device) else {
        return;
    };
    drivers[at].holds -= 1;
    if drivers[at].holds == 0 {
        let driver = drivers.swap_remove(at);
        debug_assert_eq!(
            driver.in_progress.load(Ordering::SeqCst),
            0,
            "a pipeline making outlived its call: its device went with it in progress"
        );
    }
}

/// Nanoseconds `device`'s pipeline makings spent inside its driver's entry points so far, graphics
/// then compute; zeroes for a device not routed (`WorkCounts::render_pipeline_create_nanos`).
pub(super) fn driver_nanos(device: vk::Device) -> (u64, u64) {
    let drivers = DRIVERS.read().unwrap_or_else(PoisonError::into_inner);
    match drivers.iter().find(|driver| driver.device == device) {
        Some(driver) => (
            driver.nanos.graphics.load(Ordering::Relaxed),
            driver.nanos.compute.load(Ordering::Relaxed),
        ),
        None => (0, 0),
    }
}

/// One call in progress on a routed device, counted until it returns.
struct InProgress(Arc<AtomicUsize>);

impl InProgress {
    /// `device`'s driver entry points and its record of their time, with this call counted in
    /// progress; none for a device not routed.
    fn of(
        device: vk::Device,
    ) -> Option<(
        vk::PFN_vkCreateGraphicsPipelines,
        vk::PFN_vkCreateComputePipelines,
        Arc<Nanos>,
        InProgress,
    )> {
        let drivers = DRIVERS.read().unwrap_or_else(PoisonError::into_inner);
        let driver = drivers.iter().find(|driver| driver.device == device)?;
        driver.in_progress.fetch_add(1, Ordering::SeqCst);
        Some((
            driver.graphics,
            driver.compute,
            Arc::clone(&driver.nanos),
            InProgress(Arc::clone(&driver.in_progress)),
        ))
    }
}

/// `call`, the driver's one pipeline-making call, with the nanoseconds it took added to `counter`:
/// the clock is read on whichever thread makes the call, just before and just after it, so the
/// whole of the driver's work is in the window and nothing else of the runtime's is. The pair
/// costs 27 ns on an M3 Max, against a driver's hundred microseconds and more.
fn counting<R>(counter: &AtomicU64, call: impl FnOnce() -> R) -> R {
    let entered = crate::gpu_time::host_nanos();
    let made = call();
    let taken = crate::gpu_time::host_nanos().saturating_sub(entered);
    counter.fetch_add(taken, Ordering::Relaxed);
    made
}

impl Drop for InProgress {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// The asking thread's arguments to a call made on the runtime's thread for it: alive, and
/// untouched by the asking thread, until the call returns, since it waits for it.
pub(super) struct Sent<T>(T);

// SAFETY: the asking thread waits for the call that takes these, and does nothing else meanwhile
// (`at_user_initiated`): they are used on one thread at a time.
unsafe impl<T> Send for Sent<T> {}

impl<T> Sent<T> {
    /// Wrap the asking thread's arguments.
    pub(super) fn new(arguments: T) -> Sent<T> {
        Sent(arguments)
    }

    /// The arguments. A method, so a closure that calls it captures the whole wrapper rather than
    /// its field alone.
    pub(super) fn into_inner(self) -> T {
        self.0
    }
}

thread_local! {
    /// The messages for the device's message sink the driver raised on this thread while it
    /// makes a call for an asking thread ([`at_user_initiated`]), in order; `None` elsewhere.
    static HELD: RefCell<Option<Vec<(MessageSink, DebugMessage)>>> = const { RefCell::new(None) };
}

/// Whether `message`, for `sink`, is held for the asking thread: on a thread of the runtime's
/// making a call for one ([`at_user_initiated`]), which then delivers it on the asking thread
/// once the call returns (the module docs).
pub(super) fn held(sink: &MessageSink, message: &DebugMessage) -> bool {
    HELD.try_with(|held| {
        let Ok(mut held) = held.try_borrow_mut() else {
            return false;
        };
        match held.as_mut() {
            Some(messages) => {
                messages.push((sink.clone(), message.clone()));
                true
            }
            None => false,
        }
    })
    .unwrap_or(false)
}

/// `call`, with this thread holding the sink messages the driver raises in it ([`held`]): its
/// result, and the messages in the order raised, whichever thread made the driver call
/// ([`at_user_initiated`] holds a relayed call's messages here). A thread of the runtime's making
/// a call for an asking thread holds them for it; a self-check holds its own driver call's, whose
/// failure is the answer it asked for (`pipeline::objects::takes_multi_variable_dynamic`). A
/// holding inside another keeps the outer one's messages as they were.
pub(crate) fn holding<R>(call: impl FnOnce() -> R) -> (R, Vec<(MessageSink, DebugMessage)>) {
    type Messages = Option<Vec<(MessageSink, DebugMessage)>>;
    /// Puts the outer holding back however `call` ends.
    struct Restore(Messages);
    impl Drop for Restore {
        fn drop(&mut self) {
            let outer = self.0.take();
            let _ = HELD.try_with(|held| *held.borrow_mut() = outer);
        }
    }
    // Past this thread's thread-locals' end (a program's exit handler, `this_thread.rs`), `call`
    // alone: its messages are delivered as they are raised.
    let Ok(outer) = HELD.try_with(|held| held.borrow_mut().replace(Vec::new())) else {
        return (call(), Vec::new());
    };
    let restore = Restore(outer);
    let made = call();
    let messages = HELD
        .try_with(|held| held.borrow_mut().take())
        .ok()
        .flatten()
        .unwrap_or_default();
    drop(restore);
    (made, messages)
}

/// Run `call`, one Vulkan call that may compile, at the user-initiated quality of service or
/// above (the module docs). It runs here where this thread is at that class, and otherwise on a
/// thread of the runtime's own at it while this thread waits. It runs here too where no thread
/// starts or the process is exiting.
pub(super) fn at_user_initiated<R: Send>(call: impl FnOnce() -> R + Send) -> R {
    if exit_guard::at_least_user_initiated() {
        return call();
    }
    // The driver diagnostics this thread's call is measured to report: the driver reports them
    // on the thread that makes the call (`driver_facts.rs`).
    let expected = crate::driver_facts::expected_here();
    let mut asked = Some(call);
    let made = std::thread::scope(|scope| {
        let taken = &mut asked;
        let thread = exit_guard::spawn_scoped(
            scope,
            std::thread::Builder::new().name("metal2vulkan-compile".into()),
            move || {
                exit_guard::raise_to_user_initiated();
                let _busy = exit_guard::enter()?;
                let call = taken.take()?;
                Some(holding(|| crate::driver_facts::expecting(expected, call)))
            },
        );
        match thread {
            Ok(thread) => thread
                .join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic)),
            Err(_) => None,
        }
    });
    match (made, asked) {
        (Some(((made, withheld), messages)), _) => {
            crate::driver_facts::withheld_for_here(withheld);
            // The program's sink, on the asking thread, in the order the driver raised them:
            // held again where this thread holds its call's own ([`holding`]).
            for (sink, message) in messages {
                if !held(&sink, &message) {
                    sink.deliver(&message);
                }
            }
            made
        }
        (None, Some(call)) => call(),
        (None, None) => unreachable!("the runtime's thread took the call and made it"),
    }
}

/// `vkCreateGraphicsPipelines` of a routed device: its driver's, at the user-initiated class.
unsafe extern "system" fn create_graphics_pipelines(
    device: vk::Device,
    pipeline_cache: vk::PipelineCache,
    create_info_count: u32,
    p_create_infos: *const vk::GraphicsPipelineCreateInfo<'_>,
    p_allocator: *const vk::AllocationCallbacks<'_>,
    p_pipelines: *mut vk::Pipeline,
) -> vk::Result {
    let Some((driver, _, nanos, _in_progress)) = InProgress::of(device) else {
        return vk::Result::ERROR_UNKNOWN;
    };
    let asked = Sent::new((
        pipeline_cache,
        create_info_count,
        p_create_infos,
        p_allocator,
        p_pipelines,
    ));
    at_user_initiated(move || {
        let (cache, count, infos, allocator, pipelines) = asked.into_inner();
        // SAFETY: the caller's arguments, alive and untouched while it waits for this call.
        let made = counting(&nanos.graphics, || unsafe {
            driver(device, cache, count, infos, allocator, pipelines)
        });
        exit_guard::called(Call::CreateGraphicsPipelines);
        exit_guard::pipelines_made();
        made
    })
}

/// `vkCreateComputePipelines` of a routed device: its driver's, at the user-initiated class.
unsafe extern "system" fn create_compute_pipelines(
    device: vk::Device,
    pipeline_cache: vk::PipelineCache,
    create_info_count: u32,
    p_create_infos: *const vk::ComputePipelineCreateInfo<'_>,
    p_allocator: *const vk::AllocationCallbacks<'_>,
    p_pipelines: *mut vk::Pipeline,
) -> vk::Result {
    let Some((_, driver, nanos, _in_progress)) = InProgress::of(device) else {
        return vk::Result::ERROR_UNKNOWN;
    };
    let asked = Sent::new((
        pipeline_cache,
        create_info_count,
        p_create_infos,
        p_allocator,
        p_pipelines,
    ));
    at_user_initiated(move || {
        let (cache, count, infos, allocator, pipelines) = asked.into_inner();
        // SAFETY: the caller's arguments, alive and untouched while it waits for this call.
        let made = counting(&nanos.compute, || unsafe {
            driver(device, cache, count, infos, allocator, pipelines)
        });
        exit_guard::called(Call::CreateComputePipelines);
        exit_guard::pipelines_made();
        made
    })
}

#[cfg(test)]
mod tests {
    use super::{route, unregister};
    use ash::vk::{self, Handle};
    use std::cell::Cell;
    use std::ffi::c_void;

    thread_local! {
        /// The test's asking thread.
        static ASKING: Cell<bool> = const { Cell::new(false) };
    }

    /// This thread's quality of service class (0 where there is none).
    fn class() -> u32 {
        #[cfg(target_os = "macos")]
        {
            use libc::qos_class_t as Q;
            let mut class = Q::QOS_CLASS_UNSPECIFIED;
            let mut relative = 0;
            // SAFETY: the calling thread's own handle, and two out-parameters that live for the
            // call.
            let read = unsafe {
                libc::pthread_get_qos_class_np(libc::pthread_self(), &mut class, &mut relative)
            };
            assert_eq!(read, 0, "pthread_get_qos_class_np");
            class as u32
        }
        #[cfg(not(target_os = "macos"))]
        0
    }

    /// A driver's `vkCreateGraphicsPipelines` that makes each pipeline's handle say where it ran:
    /// the class of its thread, and whether that is the asking thread (bit 32).
    unsafe extern "system" fn told(
        _: vk::Device,
        _: vk::PipelineCache,
        count: u32,
        _: *const vk::GraphicsPipelineCreateInfo<'_>,
        _: *const vk::AllocationCallbacks<'_>,
        pipelines: *mut vk::Pipeline,
    ) -> vk::Result {
        let asking = u64::from(ASKING.with(Cell::get));
        for at in 0..count as usize {
            // SAFETY: the caller's array of `count` pipelines.
            unsafe {
                *pipelines.add(at) = vk::Pipeline::from_raw(u64::from(class()) | asking << 32)
            };
        }
        vk::Result::SUCCESS
    }

    /// A driver's `vkCreateGraphicsPipelines` that reports MoltenVK's measured warning of a
    /// dynamic restart on the thread it runs on, as MoltenVK does, and makes each pipeline's
    /// handle say whether the runtime withheld it (1) or relayed it to the sink (0).
    unsafe extern "system" fn reports(
        _: vk::Device,
        _: vk::PipelineCache,
        count: u32,
        _: *const vk::GraphicsPipelineCreateInfo<'_>,
        _: *const vk::AllocationCallbacks<'_>,
        pipelines: *mut vk::Pipeline,
    ) -> vk::Result {
        let warning = restart_warning();
        let message = crate::DebugMessage {
            severity: crate::MessageSeverity::Warning,
            kind: crate::MessageKind {
                validation: true,
                ..Default::default()
            },
            id_name: warning.id_name.into(),
            id: warning.id,
            text: warning.text.into_owned(),
            degradation: None,
        };
        let withheld = u64::from(crate::driver_facts::withholds(&message));
        for at in 0..count as usize {
            // SAFETY: the caller's array of `count` pipelines.
            unsafe { *pipelines.add(at) = vk::Pipeline::from_raw(withheld) };
        }
        vk::Result::SUCCESS
    }

    /// MoltenVK's warning of a pipeline whose restart is dynamic.
    fn restart_warning() -> metal2vulkan::device::DriverDiagnostic {
        use metal2vulkan::vk::{DynamicState as D, PrimitiveTopology as T};
        metal2vulkan::device::DriverFact::DynamicRestartWarns
            .pipeline_creation_diagnostics(
                &[D::PRIMITIVE_TOPOLOGY, D::PRIMITIVE_RESTART_ENABLE],
                T::TRIANGLE_LIST,
                &[],
            )
            .pop()
            .expect("the warning of a dynamic restart")
    }

    /// How long a slow driver's pipeline making spends in the driver, in nanoseconds: a sleep, so
    /// the call takes at least this however loaded the host is.
    const SPENT: u64 = 2_000_000;

    /// The makings of each kind [`a_making_s_time_in_the_driver_is_counted_by_the_call_it_made`]
    /// asks for.
    const MAKINGS: u64 = 3;

    /// Spend [`SPENT`] nanoseconds, as a driver's compile does.
    fn spend() {
        std::thread::sleep(std::time::Duration::from_nanos(SPENT));
    }

    /// A driver's `vkCreateGraphicsPipelines` that spends [`SPENT`] nanoseconds and makes nothing.
    unsafe extern "system" fn slow_graphics(
        _: vk::Device,
        _: vk::PipelineCache,
        _: u32,
        _: *const vk::GraphicsPipelineCreateInfo<'_>,
        _: *const vk::AllocationCallbacks<'_>,
        _: *mut vk::Pipeline,
    ) -> vk::Result {
        spend();
        vk::Result::SUCCESS
    }

    /// A driver's `vkCreateComputePipelines` that spends [`SPENT`] nanoseconds and makes nothing.
    unsafe extern "system" fn slow_compute(
        _: vk::Device,
        _: vk::PipelineCache,
        _: u32,
        _: *const vk::ComputePipelineCreateInfo<'_>,
        _: *const vk::AllocationCallbacks<'_>,
        _: *mut vk::Pipeline,
    ) -> vk::Result {
        spend();
        vk::Result::SUCCESS
    }

    /// A device of handle `handle` whose driver has both slow makings, routed.
    fn routed_slow(handle: u64) -> ash::Device {
        let graphics: vk::PFN_vkCreateGraphicsPipelines = slow_graphics;
        let compute: vk::PFN_vkCreateComputePipelines = slow_compute;
        // SAFETY: a table of the two slow makings and ash's stubs; no other entry point is called.
        let driver = unsafe {
            ash::Device::load_with(
                |name| {
                    if name == c"vkCreateGraphicsPipelines" {
                        graphics as *const c_void
                    } else if name == c"vkCreateComputePipelines" {
                        compute as *const c_void
                    } else {
                        std::ptr::null()
                    }
                },
                vk::Device::from_raw(handle),
            )
        };
        // SAFETY: its functions are the driver's.
        unsafe { route(driver) }
    }

    /// The time a pipeline making spends inside the driver is counted on whichever thread makes
    /// the call, and by the call it made: makings of one kind leave the other kind's figure
    /// untouched, so a render-heavy run and a dispatch-heavy one are told apart. Each making here
    /// sleeps [`SPENT`] nanoseconds in the driver, so the floor holds however loaded the host is;
    /// a door that timed nothing would read zero for both.
    #[test]
    fn a_making_s_time_in_the_driver_is_counted_by_the_call_it_made() {
        let device = routed_slow(0x6d32_7600_0010);
        let handle = device.handle();
        assert_eq!(
            super::driver_nanos(handle),
            (0, 0),
            "a device just routed has made nothing"
        );
        let graphics_info = vk::GraphicsPipelineCreateInfo::default();
        for _ in 0..MAKINGS {
            // SAFETY: the routed device's making is the slow driver's, which reads no argument.
            let _ = unsafe {
                device.create_graphics_pipelines(vk::PipelineCache::null(), &[graphics_info], None)
            };
        }
        let (graphics, compute) = super::driver_nanos(handle);
        assert!(
            graphics >= MAKINGS * SPENT,
            "{MAKINGS} graphics makings of {SPENT} ns each read {graphics} ns"
        );
        assert_eq!(
            compute, 0,
            "a graphics making is not counted as a compute one"
        );
        let compute_info = vk::ComputePipelineCreateInfo::default();
        for _ in 0..MAKINGS {
            // SAFETY: as above.
            let _ = unsafe {
                device.create_compute_pipelines(vk::PipelineCache::null(), &[compute_info], None)
            };
        }
        let (graphics_after, compute_after) = super::driver_nanos(handle);
        assert!(
            compute_after >= MAKINGS * SPENT,
            "{MAKINGS} compute makings of {SPENT} ns each read {compute_after} ns"
        );
        assert_eq!(
            graphics_after, graphics,
            "a compute making is not counted as a graphics one"
        );
        unregister(handle);
        assert_eq!(
            super::driver_nanos(handle),
            (0, 0),
            "the record went with the device's last hold"
        );
    }

    /// A device of handle `handle` whose driver has only `driver` ([`told`] or [`reports`]),
    /// routed.
    fn routed(handle: u64, driver: vk::PFN_vkCreateGraphicsPipelines) -> ash::Device {
        // SAFETY: a table of `driver` and ash's stubs; no other entry point is called.
        let driver = unsafe {
            ash::Device::load_with(
                |name| {
                    if name == c"vkCreateGraphicsPipelines" {
                        driver as *const c_void
                    } else {
                        std::ptr::null()
                    }
                },
                vk::Device::from_raw(handle),
            )
        };
        // SAFETY: its functions are the driver's.
        unsafe { route(driver) }
    }

    /// `ask`'s answer on a thread that ran `set` first and marked itself the asking thread: a
    /// thread started as the runtime starts its own (`exit_guard::spawn_scoped`), standing for a
    /// program's thread here, since nothing on the way to the driver asks which it is.
    fn asked<R: Send>(set: impl FnOnce() + Send, ask: impl FnOnce() -> R + Send) -> R {
        std::thread::scope(|scope| {
            crate::exit_guard::spawn_scoped(scope, std::thread::Builder::new(), || {
                set();
                ASKING.with(|asking| asking.set(true));
                ask()
            })
            .unwrap()
            .join()
            .unwrap()
        })
    }

    /// The handle of a pipeline the routed device made.
    fn make(device: &ash::Device) -> u64 {
        let info = vk::GraphicsPipelineCreateInfo::default();
        // SAFETY: the routed device's `vkCreateGraphicsPipelines` is a fake driver's behind
        // `compiles.rs`; it reads no argument but the pipelines.
        let made =
            unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None) }
                .unwrap();
        made[0].as_raw()
    }

    /// Sets the calling thread's class to `start`.
    #[cfg(target_os = "macos")]
    fn at(start: libc::qos_class_t) -> impl FnOnce() + Send {
        move || {
            // SAFETY: sets the calling thread's own class; no pointer.
            assert_eq!(unsafe { libc::pthread_set_qos_class_self_np(start, 0) }, 0);
        }
    }

    /// Where the routed device made a pipeline asked for on a thread that ran `set` first: the
    /// class it ran at, whether on the asking thread, and the asking thread's class after.
    fn made_from(device: &ash::Device, set: impl FnOnce() + Send) -> (u32, bool, u32) {
        asked(set, || {
            let raw = make(device);
            (raw as u32, raw >> 32 == 1, class())
        })
    }

    /// A pipeline asked for on a thread below the user-initiated class is made on a thread of
    /// the runtime's at that class, the asking thread's class untouched; at that class or above,
    /// on the asking thread. Elsewhere than macOS, always on the asking thread.
    #[test]
    fn a_pipeline_is_made_at_the_user_initiated_class_and_the_asker_is_left_as_found() {
        let device = routed(0x6d32_7600_0001, told);
        #[cfg(target_os = "macos")]
        {
            use libc::qos_class_t as Q;
            let initiated = Q::QOS_CLASS_USER_INITIATED as u32;
            for start in [
                Q::QOS_CLASS_BACKGROUND,
                Q::QOS_CLASS_UTILITY,
                Q::QOS_CLASS_DEFAULT,
            ] {
                let (ran, asking, after) = made_from(&device, at(start));
                assert_eq!((ran, asking), (initiated, false), "from {start:?}");
                assert_eq!(
                    after, start as u32,
                    "the asking thread's class, from {start:?}"
                );
            }
            for start in [Q::QOS_CLASS_USER_INITIATED, Q::QOS_CLASS_USER_INTERACTIVE] {
                let (ran, asking, after) = made_from(&device, at(start));
                assert_eq!(
                    (ran, asking, after),
                    (start as u32, true, start as u32),
                    "from {start:?}"
                );
            }
        }
        #[cfg(not(target_os = "macos"))]
        assert_eq!(made_from(&device, || {}), (0, true, 0));
        unregister(device.handle());
    }

    /// A device routed twice keeps its record until the runtime let go of both; then its
    /// pipeline makings fail, as for any device never routed.
    #[test]
    fn a_device_s_record_goes_with_its_last_hold() {
        let first = routed(0x6d32_7600_0002, told);
        let second = routed(0x6d32_7600_0002, told);
        unregister(first.handle());
        // Made: `made_from` unwraps the result.
        made_from(&second, || {});
        unregister(second.handle());
        let info = vk::GraphicsPipelineCreateInfo::default();
        // SAFETY: as in `made_from`; the call fails before any driver is reached.
        let made =
            unsafe { second.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None) };
        assert_eq!(
            made.map_err(|(_, error)| error),
            Err(vk::Result::ERROR_UNKNOWN)
        );
    }

    /// A driver diagnostic the asking thread's call is measured to report is withheld where the
    /// driver reports it, on the thread that makes the call for it, and counted as the asking
    /// thread's call's; in a call that expects none it relays.
    #[test]
    fn a_measured_diagnostic_is_withheld_on_whichever_thread_makes_the_call() {
        let device = &routed(0x6d32_7600_0003, reports);
        let warning = restart_warning();
        let expecting = |expected: Vec<metal2vulkan::device::DriverDiagnostic>| {
            move || crate::driver_facts::expecting(expected, || make(device))
        };
        #[cfg(target_os = "macos")]
        {
            use libc::qos_class_t as Q;
            for start in [
                Q::QOS_CLASS_BACKGROUND,
                Q::QOS_CLASS_UTILITY,
                Q::QOS_CLASS_DEFAULT,
                Q::QOS_CLASS_USER_INITIATED,
            ] {
                assert_eq!(
                    asked(at(start), expecting(vec![warning.clone()])),
                    (1, 1),
                    "withheld and counted, from {start:?}"
                );
                assert_eq!(
                    asked(at(start), expecting(Vec::new())),
                    (0, 0),
                    "relayed, from {start:?}"
                );
            }
        }
        assert_eq!(asked(|| {}, expecting(vec![warning.clone()])), (1, 1));
        assert_eq!(asked(|| {}, expecting(Vec::new())), (0, 0));
        unregister(device.handle());
    }

    /// The sink [`reports_to_sink`] reports to.
    static SINK: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    /// A driver's `vkCreateGraphicsPipelines` that raises two messages, "first" then "second",
    /// through the device's messenger callback to the sink at [`SINK`], on the thread that makes
    /// the call, as a driver does.
    unsafe extern "system" fn reports_to_sink(
        _: vk::Device,
        _: vk::PipelineCache,
        count: u32,
        _: *const vk::GraphicsPipelineCreateInfo<'_>,
        _: *const vk::AllocationCallbacks<'_>,
        pipelines: *mut vk::Pipeline,
    ) -> vk::Result {
        use vk::DebugUtilsMessageSeverityFlagsEXT as S;
        use vk::DebugUtilsMessageTypeFlagsEXT as K;
        let sink = SINK.load(std::sync::atomic::Ordering::SeqCst) as *mut c_void;
        for text in [c"first", c"second"] {
            let data = vk::DebugUtilsMessengerCallbackDataEXT::default()
                .message_id_name(c"m2v-test")
                .message(text);
            // SAFETY: the callback data lives for the call, and `sink` is the test's sink, leaked.
            unsafe { super::super::create::deliver(S::WARNING, K::VALIDATION, &data, sink) };
        }
        for at in 0..count as usize {
            // SAFETY: the caller's array of `count` pipelines.
            unsafe { *pipelines.add(at) = vk::Pipeline::from_raw(1) };
        }
        vk::Result::SUCCESS
    }

    /// A lock its holder may take again, as a program's recursive mutex.
    #[derive(Default)]
    struct Reentrant {
        state: std::sync::Mutex<(Option<std::thread::ThreadId>, u32)>,
        freed: std::sync::Condvar,
    }

    impl Reentrant {
        fn take(&self) {
            let me = std::thread::current().id();
            let mut state = self.state.lock().unwrap();
            loop {
                match state.0 {
                    None => {
                        *state = (Some(me), 1);
                        return;
                    }
                    Some(owner) if owner == me => {
                        state.1 += 1;
                        return;
                    }
                    Some(_) => state = self.freed.wait(state).unwrap(),
                }
            }
        }

        fn give(&self) {
            let mut state = self.state.lock().unwrap();
            state.1 -= 1;
            if state.1 == 0 {
                state.0 = None;
                self.freed.notify_all();
            }
        }
    }

    /// A program that holds a lock across a pipeline's creation, whose message sink takes that
    /// lock too, is not deadlocked by the full join: the sink sees the messages the driver raised
    /// in the call on the asking thread, in their order, before the call returns. On macOS the
    /// asking thread is at the utility class, so the call is made on the runtime's thread.
    #[test]
    fn a_sink_taking_a_lock_the_asker_holds_sees_its_messages_on_the_asking_thread() {
        let device = routed(0x6d32_7600_0004, reports_to_sink);
        let lock = std::sync::Arc::new(Reentrant::default());
        type Seen = Vec<(std::thread::ThreadId, String)>;
        let seen: std::sync::Arc<std::sync::Mutex<Seen>> = std::sync::Arc::default();
        let sink: &'static crate::MessageSink = Box::leak(Box::new({
            let (lock, seen) = (lock.clone(), seen.clone());
            crate::MessageSink::new(move |message| {
                lock.take();
                let me = std::thread::current().id();
                seen.lock().unwrap().push((me, message.text.clone()));
                lock.give();
            })
        }));
        SINK.store(
            std::ptr::from_ref(sink) as usize,
            std::sync::atomic::Ordering::SeqCst,
        );
        let (sent, got) = std::sync::mpsc::channel();
        let asking = {
            let (lock, device) = (lock.clone(), device.clone());
            crate::exit_guard::spawn(std::thread::Builder::new(), move || {
                #[cfg(target_os = "macos")]
                at(libc::qos_class_t::QOS_CLASS_UTILITY)();
                lock.take();
                let made = make(&device);
                lock.give();
                sent.send((std::thread::current().id(), made)).unwrap();
            })
            .unwrap()
        };
        let (asker, made) = got
            .recv_timeout(std::time::Duration::from_secs(30))
            .expect("the creation deadlocked: the sink ran off the asking thread");
        asking.join().unwrap();
        assert_eq!(made, 1);
        let seen = seen.lock().unwrap().clone();
        assert_eq!(
            seen,
            [(asker, "first".to_string()), (asker, "second".to_string())]
        );
        unregister(device.handle());
    }
}
