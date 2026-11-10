//! `device/create.rs`: the making of an instance reads the loader's instance extensions once
//! (`ListedExtensions`), in one `vkEnumerateInstanceExtensionProperties` call, whatever it asks of
//! them. Each such call makes the loader scan its driver and layer manifests again and load and
//! ask each driver it finds, so a read per question multiplied that cost by the questions asked,
//! and the usual two-call read (the count, then the list) doubled it. Fake loaders count the calls
//! and refuse the instance, so no driver and no device take part.

use crate::device::create::{instance_of, ListedExtensions, LISTED_ROOM};
use crate::{DeviceOptions, Error};
use ash::vk;
use std::ffi::{c_char, CStr, CString};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// The instance extensions the first fake loader lists: each one the making asks about.
const LISTED: [&CStr; 3] = [
    ash::khr::portability_enumeration::NAME,
    ash::ext::debug_utils::NAME,
    ash::ext::layer_settings::NAME,
];

/// The first fake loader's `vkEnumerateInstanceExtensionProperties` calls, each a scan.
static CALLS: AtomicUsize = AtomicUsize::new(0);

/// The instance extensions the first fake loader's `vkCreateInstance` was asked to enable.
static ENABLED: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// The second fake loader's calls.
static MANY_CALLS: AtomicUsize = AtomicUsize::new(0);

/// How many extensions the second fake loader lists: more than the first call's room, within twice
/// it.
const MANY: usize = LISTED_ROOM * 3 / 2;

/// The name of the second fake loader's extension `at`.
fn many_name(at: usize) -> CString {
    CString::new(format!("VK_FAKE_extension_{at}")).unwrap()
}

/// A `vkEnumerateInstanceExtensionProperties` of `names` (`names.len()` of them, the `at`-th named
/// by `name(at)`): the count where `properties` is null, else as many as the caller's count has
/// room for, `VK_INCOMPLETE` where that is not all.
///
/// # Safety
///
/// `count` is the caller's, and `properties`, where not null, has room for `*count` properties.
unsafe fn answer(
    names: usize,
    name: impl Fn(usize) -> CString,
    count: *mut u32,
    properties: *mut vk::ExtensionProperties,
) -> vk::Result {
    if properties.is_null() {
        // SAFETY: the caller's count, written once.
        unsafe { count.write(names as u32) };
        return vk::Result::SUCCESS;
    }
    // SAFETY: the caller's count, which says how many properties `properties` has room for.
    let room = unsafe { count.read() } as usize;
    let written = room.min(names);
    for at in 0..written {
        let property = vk::ExtensionProperties::default()
            .extension_name(&name(at))
            .expect("an extension name fits");
        // SAFETY: `at` is below the room the caller gave.
        unsafe { properties.add(at).write(property) };
    }
    // SAFETY: the caller's count.
    unsafe { count.write(written as u32) };
    if written < names {
        vk::Result::INCOMPLETE
    } else {
        vk::Result::SUCCESS
    }
}

/// The first fake loader's `vkEnumerateInstanceExtensionProperties`: [`LISTED`], counted in
/// [`CALLS`].
unsafe extern "system" fn enumerate_listed(
    _layer: *const c_char,
    count: *mut u32,
    properties: *mut vk::ExtensionProperties,
) -> vk::Result {
    CALLS.fetch_add(1, Ordering::SeqCst);
    // SAFETY: the caller's pointers, as the loader's entry point takes them.
    unsafe { answer(LISTED.len(), |at| LISTED[at].to_owned(), count, properties) }
}

/// The second fake loader's: [`MANY`] extensions, counted in [`MANY_CALLS`].
unsafe extern "system" fn enumerate_many(
    _layer: *const c_char,
    count: *mut u32,
    properties: *mut vk::ExtensionProperties,
) -> vk::Result {
    MANY_CALLS.fetch_add(1, Ordering::SeqCst);
    // SAFETY: the caller's pointers, as the loader's entry point takes them.
    unsafe { answer(MANY, many_name, count, properties) }
}

/// `vkCreateInstance`: records the extensions asked for in [`ENABLED`], and refuses the instance as
/// a loader with no driver does.
unsafe extern "system" fn create_instance(
    info: *const vk::InstanceCreateInfo<'_>,
    _allocator: *const vk::AllocationCallbacks<'_>,
    _instance: *mut vk::Instance,
) -> vk::Result {
    // SAFETY: the caller's create info, live for the call.
    let info = unsafe { &*info };
    let names = (0..info.enabled_extension_count as usize).map(|at| {
        // SAFETY: `at` is below the count of NUL-terminated names the create info points at.
        let name = unsafe { CStr::from_ptr(*info.pp_enabled_extension_names.add(at)) };
        name.to_string_lossy().into_owned()
    });
    *ENABLED.lock().unwrap() = names.collect();
    vk::Result::ERROR_INCOMPATIBLE_DRIVER
}

/// The entry point `name` names, of the fake loader whose extension list `enumerate` answers.
///
/// # Safety
///
/// `name` is a NUL-terminated string.
unsafe fn proc_addr(
    name: *const c_char,
    enumerate: vk::PFN_vkEnumerateInstanceExtensionProperties,
) -> vk::PFN_vkVoidFunction {
    // SAFETY: the caller's.
    let name = unsafe { CStr::from_ptr(name) }.to_bytes();
    // SAFETY (both arms): the caller casts the pointer back to the named entry point's type,
    // which is the type of the function returned.
    match name {
        b"vkEnumerateInstanceExtensionProperties" => Some(unsafe {
            std::mem::transmute::<
                vk::PFN_vkEnumerateInstanceExtensionProperties,
                unsafe extern "system" fn(),
            >(enumerate)
        }),
        b"vkCreateInstance" => Some(unsafe {
            std::mem::transmute::<vk::PFN_vkCreateInstance, unsafe extern "system" fn()>(
                create_instance,
            )
        }),
        _ => None,
    }
}

/// The first fake loader's `vkGetInstanceProcAddr`.
unsafe extern "system" fn proc_addr_listed(
    _instance: vk::Instance,
    name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    // SAFETY: the loader names an entry point with a NUL-terminated string.
    unsafe { proc_addr(name, enumerate_listed) }
}

/// The second fake loader's `vkGetInstanceProcAddr`.
unsafe extern "system" fn proc_addr_many(
    _instance: vk::Instance,
    name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    // SAFETY: the loader names an entry point with a NUL-terminated string.
    unsafe { proc_addr(name, enumerate_many) }
}

/// The fake loader `get_instance_proc_addr` answers for.
fn entry(get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr) -> ash::Entry {
    let static_fn = ash::StaticFn {
        get_instance_proc_addr,
    };
    // SAFETY: the fake answers the global commands the making calls; ash's stand-in for any other
    // panics if called.
    unsafe { ash::Entry::from_static_fn(static_fn) }
}

#[test]
fn an_instance_reads_the_loaders_extensions_in_one_call() {
    match instance_of(entry(proc_addr_listed), &DeviceOptions::default()) {
        Err(Error::NoDevice) => {}
        Err(other) => panic!("the fake loader's refusal became {other}"),
        Ok(_) => panic!("the fake loader made an instance"),
    }
    assert_eq!(
        CALLS.load(Ordering::SeqCst),
        1,
        "the making called the loader's extension list more than once"
    );
    // Every question was asked, and answered from the one read: each listed extension the
    // making enables where listed was asked for.
    let enabled = ENABLED.lock().unwrap().clone();
    for name in LISTED {
        let name = name.to_string_lossy().into_owned();
        assert!(enabled.contains(&name), "{name} not enabled: {enabled:?}");
    }
}

/// A list longer than the first call's room takes a second call into twice the room, and reads
/// every extension.
#[test]
fn a_list_past_the_first_room_is_read_whole() {
    let listed = ListedExtensions::of(&entry(proc_addr_many)).expect("the list");
    assert_eq!(MANY_CALLS.load(Ordering::SeqCst), 2, "one call per room");
    for at in [0, LISTED_ROOM - 1, LISTED_ROOM, MANY - 1] {
        assert!(listed.lists(&many_name(at)), "extension {at} not read");
    }
    assert!(
        !listed.lists(&many_name(MANY)),
        "an extension past the list"
    );
}
