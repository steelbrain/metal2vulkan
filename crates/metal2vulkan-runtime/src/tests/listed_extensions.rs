//! `device/create.rs`: the making of an instance reads the loader's instance extensions once
//! (`ListedExtensions`), whatever it asks of them. Each `vkEnumerateInstanceExtensionProperties`
//! makes the loader scan its driver and layer manifests again and ask each driver it finds, so a
//! read per question multiplied that cost by the questions asked. A fake loader counts the reads
//! and refuses the instance, so no driver and no device take part.

use crate::device::create::instance_of;
use crate::{DeviceOptions, Error};
use ash::vk;
use std::ffi::{c_char, CStr};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// The instance extensions the fake loader lists: each one the making asks about.
const LISTED: [&CStr; 3] = [
    ash::khr::portability_enumeration::NAME,
    ash::ext::debug_utils::NAME,
    ash::ext::layer_settings::NAME,
];

/// The fake loader's reads of its instance extensions: the calls that ask for their count, the
/// first of each two-call read.
static READS: AtomicUsize = AtomicUsize::new(0);

/// The instance extensions the fake loader's `vkCreateInstance` was asked to enable.
static ENABLED: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// `vkEnumerateInstanceExtensionProperties`: [`LISTED`], counted in [`READS`].
unsafe extern "system" fn enumerate_instance_extension_properties(
    _layer: *const c_char,
    count: *mut u32,
    properties: *mut vk::ExtensionProperties,
) -> vk::Result {
    if properties.is_null() {
        READS.fetch_add(1, Ordering::SeqCst);
        // SAFETY: the caller's count, written once.
        unsafe { count.write(LISTED.len() as u32) };
        return vk::Result::SUCCESS;
    }
    // SAFETY: the caller's count, which says how many properties `properties` has room for.
    let room = unsafe { count.read() } as usize;
    let written = room.min(LISTED.len());
    for (at, name) in LISTED.iter().take(written).enumerate() {
        let property = vk::ExtensionProperties::default()
            .extension_name(name)
            .expect("an extension name fits");
        // SAFETY: `at` is below the room the caller gave.
        unsafe { properties.add(at).write(property) };
    }
    // SAFETY: the caller's count.
    unsafe { count.write(written as u32) };
    if written < LISTED.len() {
        vk::Result::INCOMPLETE
    } else {
        vk::Result::SUCCESS
    }
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

/// The fake loader's `vkGetInstanceProcAddr`: the two global commands above, and nothing else.
unsafe extern "system" fn get_instance_proc_addr(
    _instance: vk::Instance,
    name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    // SAFETY: the caller names an entry point with a NUL-terminated string.
    let name = unsafe { CStr::from_ptr(name) }.to_bytes();
    // SAFETY (both arms): the caller casts the pointer back to the named entry point's type,
    // which is the type of the function returned.
    match name {
        b"vkEnumerateInstanceExtensionProperties" => Some(unsafe {
            std::mem::transmute::<
                vk::PFN_vkEnumerateInstanceExtensionProperties,
                unsafe extern "system" fn(),
            >(enumerate_instance_extension_properties)
        }),
        b"vkCreateInstance" => Some(unsafe {
            std::mem::transmute::<vk::PFN_vkCreateInstance, unsafe extern "system" fn()>(
                create_instance,
            )
        }),
        _ => None,
    }
}

#[test]
fn an_instance_reads_the_loaders_extensions_once() {
    let static_fn = ash::StaticFn {
        get_instance_proc_addr,
    };
    // SAFETY: the fake answers the global commands the making calls; ash's stand-in for any other
    // panics if called.
    let entry = unsafe { ash::Entry::from_static_fn(static_fn) };
    match instance_of(entry, &DeviceOptions::default()) {
        Err(Error::NoDevice) => {}
        Err(other) => panic!("the fake loader's refusal became {other}"),
        Ok(_) => panic!("the fake loader made an instance"),
    }
    assert_eq!(
        READS.load(Ordering::SeqCst),
        1,
        "the making read the loader's instance extensions more than once"
    );
    // Every question was asked, and answered from the one read: each listed extension the
    // making enables where listed was asked for.
    let enabled = ENABLED.lock().unwrap().clone();
    for name in LISTED {
        let name = name.to_string_lossy().into_owned();
        assert!(enabled.contains(&name), "{name} not enabled: {enabled:?}");
    }
}
