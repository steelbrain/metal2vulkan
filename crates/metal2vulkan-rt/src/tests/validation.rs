//! The Khronos validation layer under every instance the device tests open, and its log; but the
//! lock budget's (`crate::tests::locks`), whose count the layer's own locks would be.
//!
//! The job points the loader at the layer's directory (`VK_LAYER_PATH`; `scripts/validation-layer`
//! makes it). [`instance`] enables the layer for its own process through the loader's and the
//! layer's own variables, before the process's first Vulkan call: the layer, its report level
//! (errors and warnings) and its action (a log), with one log file per test process. It checks
//! that every physical device of the instance lists an active validation tool, so a leg whose
//! loader found no layer fails rather than running without it. Once the instance is gone (its
//! guard waits, bounded, on a weak handle of it), the guard reads the process's log, prints each
//! line with the test's name, fails on any message and removes the file. Nothing of this is in
//! the product.
//!
//! The variables are set once per process, before it starts a thread of its own, and the log
//! is one file per process: a test process opens one instance at a time. nextest runs each test
//! in a process of its own; libtest's runner runs them one at a time with `--test-threads=1`.
//!
//! Where the job names a driver library in `M2V_DIRECT_DRIVER` (the Mesa Mac driver on the M3),
//! [`instance`] loads it itself and gives it to the loader alone, through
//! [`Instance::with_drivers`], as a consumer that ships its driver does; elsewhere the loader finds
//! the drivers the job's `VK_DRIVER_FILES` names.
//!
//! Also here: a deliberately invalid call, which its log holds exactly.

use crate::tests::devices_of_the_cfgs_kind;
use crate::Instance;
use std::ffi::CStr;
use std::ops::Deref;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Once, Weak};
use std::thread;
use std::time::{Duration, Instant};

/// The layer every instance of the device tests runs under.
const LAYER: &str = "VK_LAYER_KHRONOS_validation";

/// How long a guard waits for the instance to go once the test lets it go: for the devices and
/// physical devices made from it, which hold it, to go too.
const PATIENCE: Duration = Duration::from_secs(60);

/// The most tools a device's tool query may list: a count past it is no device's, and fails the
/// test by name rather than sizing an allocation.
const MOST_TOOLS: u32 = 64;

/// Whether a guard's instance is alive in this process.
static OPEN: AtomicBool = AtomicBool::new(false);

/// The log of the test process `pid`: the layer writes each error and warning it reports there.
pub(crate) fn log_of(pid: u32) -> PathBuf {
    std::env::temp_dir().join(format!("m2v-validation-{pid}.log"))
}

/// Enables the layer for this process: its variables, set before the process's first Vulkan
/// call. The loader ignores a layer it cannot find; [`instance`] checks that it found one.
fn enable() {
    static ENABLED: Once = Once::new();
    ENABLED.call_once(|| {
        let log = log_of(std::process::id());
        // A log left by an earlier process of the same id is not this one's.
        let _ = std::fs::remove_file(&log);
        std::env::set_var("VK_INSTANCE_LAYERS", LAYER);
        std::env::set_var("VK_KHRONOS_VALIDATION_REPORT_FLAGS", "error,warn");
        std::env::set_var(
            "VK_KHRONOS_VALIDATION_DEBUG_ACTION",
            "VK_DBG_LAYER_ACTION_LOG_MSG",
        );
        std::env::set_var("VK_KHRONOS_VALIDATION_LOG_FILENAME", &log);
    });
}

/// An instance under the layer, for a test, with the extensions `Instance::new` takes. Every
/// device of the cfg's kind it lists names an active validation tool, or the test fails. The
/// guard's check runs when it drops ([`Watch`]).
pub(crate) fn instance(extensions: &[&CStr]) -> Validated {
    let test = test_name();
    enable();
    assert!(
        !OPEN.swap(true, Ordering::AcqRel),
        "{test}: a second instance in the process: its layer's log is one file per process"
    );
    let made = match direct_driver() {
        // SAFETY: the driver's library stays loaded for the process (`direct_driver`).
        Some(driver) => unsafe { Instance::with_drivers(extensions, &[driver]) },
        None => Instance::new(extensions),
    };
    let instance = made.unwrap_or_else(|error| panic!("{test}: an instance at the floor: {error}"));
    let watch = Watch {
        instance: std::sync::Arc::downgrade(&instance.shared),
        test: test.clone(),
        log: log_of(std::process::id()),
        expected: Vec::new(),
    };
    // The guard first: a failed check below drops it, which lets the next test of the process
    // open its own.
    let validated = Validated { instance, watch };
    let devices = devices_of_the_cfgs_kind(&validated);
    assert!(!devices.is_empty(), "{test}: no device of this cfg's kind");
    for (name, physical) in &devices {
        assert!(
            validation_tool(&validated, name, physical.raw()).is_some(),
            "{test}: {name} lists no active validation tool: the loader found no {LAYER} \
             (VK_LAYER_PATH names the layer's directory)"
        );
    }
    drop(devices);
    validated
}

/// The driver `M2V_DIRECT_DRIVER` names, loaded once for the process and never unloaded: its
/// `vk_icdGetInstanceProcAddr` (a driver built for the loader exports it), else its
/// `vkGetInstanceProcAddr`. None where the variable is unset. A library that does not load, or
/// exports neither, fails the test.
fn direct_driver() -> Option<ash::vk::PFN_vkGetInstanceProcAddr> {
    static DRIVER: std::sync::OnceLock<Option<ash::vk::PFN_vkGetInstanceProcAddr>> =
        std::sync::OnceLock::new();
    *DRIVER.get_or_init(|| {
        let path = std::env::var_os("M2V_DIRECT_DRIVER")?;
        Some(load_driver(&PathBuf::from(path)))
    })
}

#[cfg(unix)]
fn load_driver(path: &std::path::Path) -> ash::vk::PFN_vkGetInstanceProcAddr {
    use std::os::unix::ffi::OsStrExt;
    let name = std::ffi::CString::new(path.as_os_str().as_bytes()).expect("a path without a NUL");
    // SAFETY: a path; the library's initialisers are the driver's own.
    let library = unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
    assert!(!library.is_null(), "{} does not load", path.display());
    let symbol = [c"vk_icdGetInstanceProcAddr", c"vkGetInstanceProcAddr"]
        .into_iter()
        // SAFETY: the library is loaded, and is never unloaded.
        .map(|name| unsafe { libc::dlsym(library, name.as_ptr()) })
        .find(|symbol| !symbol.is_null())
        .unwrap_or_else(|| panic!("{} exports no vkGetInstanceProcAddr", path.display()));
    println!("direct driver {}", path.display());
    // SAFETY: the symbol is the driver's vkGetInstanceProcAddr, of that signature.
    unsafe {
        std::mem::transmute::<*mut std::ffi::c_void, ash::vk::PFN_vkGetInstanceProcAddr>(symbol)
    }
}

#[cfg(not(unix))]
fn load_driver(path: &std::path::Path) -> ash::vk::PFN_vkGetInstanceProcAddr {
    panic!(
        "M2V_DIRECT_DRIVER ({}) is loaded on unix hosts alone",
        path.display()
    )
}

/// The active validation tool `physical` (the device `name`) lists, by its name and version.
///
/// The count starts at 0, through the entry point itself: the Windows Intel driver (Vulkan
/// 1.3.215) returns `VK_SUCCESS` and writes no count when asked for the count alone, and the layer
/// adds its own tool to whatever the count below it holds.
pub(crate) fn validation_tool(
    instance: &Instance,
    name: &str,
    physical: ash::vk::PhysicalDevice,
) -> Option<String> {
    let (_, vk_instance) = instance.raw();
    let tools_of = vk_instance.fp_v1_3().get_physical_device_tool_properties;
    let mut count = 0;
    // SAFETY: `physical` is one of `instance`'s, a Vulkan 1.3 instance; a count query.
    unsafe { tools_of(physical, &mut count, std::ptr::null_mut()) }
        .result()
        .unwrap_or_else(|error| panic!("{name}: the tool count: {error}"));
    assert!(
        count <= MOST_TOOLS,
        "{name}: vkGetPhysicalDeviceToolProperties returned {count} tools"
    );
    let mut tools = vec![ash::vk::PhysicalDeviceToolProperties::default(); count as usize];
    // SAFETY: as above; `tools` holds `count` entries.
    unsafe { tools_of(physical, &mut count, tools.as_mut_ptr()) }
        .result()
        .unwrap_or_else(|error| panic!("{name}: the tools: {error}"));
    tools.truncate(count as usize);
    tools
        .iter()
        .find(|tool| {
            tool.purposes
                .contains(ash::vk::ToolPurposeFlags::VALIDATION)
        })
        .map(|tool| {
            let text = |chars: &[std::ffi::c_char]| {
                // SAFETY: the driver's and the layer's strings are NUL-terminated in their arrays.
                unsafe { CStr::from_ptr(chars.as_ptr()) }
                    .to_string_lossy()
                    .into_owned()
            };
            format!("{} {}", text(&tool.name), text(&tool.version))
        })
}

/// The test this thread runs: libtest names its thread after it.
fn test_name() -> String {
    thread::current().name().unwrap_or("a test").to_owned()
}

/// An instance under the layer and its watch: the instance drops first, then the watch checks
/// the log once the instance is gone.
pub(crate) struct Validated {
    instance: Instance,
    watch: Watch,
}

impl Validated {
    /// The guard of a test whose log holds exactly the messages naming `ids` (one each, in
    /// order): a test of the log itself.
    pub(crate) fn expecting(mut self, ids: &[&str]) -> Validated {
        self.watch.expected = ids.iter().map(|id| id.to_string()).collect();
        self
    }

    /// The instance and its watch apart, for a test that drops its instance's handle before
    /// what it made from it.
    pub(crate) fn split(self) -> (Instance, Watch) {
        (self.instance, self.watch)
    }
}

impl Deref for Validated {
    type Target = Instance;

    fn deref(&self) -> &Instance {
        &self.instance
    }
}

/// The check of one instance's log: when it drops, it waits, at most [`PATIENCE`], for the
/// instance to be gone, then reads the log.
pub(crate) struct Watch {
    instance: Weak<crate::instance::Shared>,
    test: String,
    log: PathBuf,
    /// The message ids the log holds, where a test makes invalid calls on purpose.
    expected: Vec<String>,
}

impl Drop for Watch {
    fn drop(&mut self) {
        let deadline = Instant::now() + PATIENCE;
        let mut leaked = false;
        while self.instance.strong_count() > 0 {
            if Instant::now() > deadline {
                leaked = true;
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }
        OPEN.store(false, Ordering::Release);
        let text = std::fs::read_to_string(&self.log).unwrap_or_default();
        let _ = std::fs::remove_file(&self.log);
        let ids = message_ids(&text);
        for line in text.lines().filter(|line| !line.trim().is_empty()) {
            println!("M2V-VALIDATION {}: {line}", self.test);
        }
        if thread::panicking() {
            return;
        }
        assert!(
            !leaked,
            "{}: the instance was still held {PATIENCE:?} after the test let it go",
            self.test
        );
        assert_eq!(
            ids, self.expected,
            "{}: the validation layer's messages (each printed above)",
            self.test
        );
    }
}

/// The ids of a log's messages (each a VUID or the layer's own name for its check): each
/// message begins on a line `Validation <kind>: [ <id> ] ...`. Text before the first counts as
/// one message more, so nothing in a log goes unreported.
fn message_ids(text: &str) -> Vec<String> {
    let mut ids = Vec::new();
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        if line.starts_with("Validation ") {
            let id = (line
                .split_once("[ ")
                .and_then(|(_, rest)| rest.split_once(" ]")))
            .map_or("(no id)", |(id, _)| id);
            ids.push(id.to_string());
        } else if ids.is_empty() {
            ids.push("(not a message of the layer's)".to_string());
        }
    }
    ids
}

/// A log's messages are counted by their first lines, and text that begins none is one more.
#[test]
fn a_log_s_messages_are_counted_by_their_first_lines() {
    let log = "Validation Error: [ VUID-VkBufferCreateInfo-size-00912 ] | MessageID = 0x7c54445e\n\
               vkCreateBuffer(): pCreateInfo->size is zero.\n\
               \n\
               Validation Warning: [ WARNING-something ] | MessageID = 0x1\n\
               Validation Performance Warning: [ BestPractices-x ] text\n";
    assert_eq!(
        message_ids(log),
        [
            "VUID-VkBufferCreateInfo-size-00912",
            "WARNING-something",
            "BestPractices-x"
        ]
    );
    assert!(message_ids("").is_empty());
    assert_eq!(
        message_ids("something else\nValidation Error: [ VUID-x ] y\n"),
        ["(not a message of the layer's)", "VUID-x"]
    );
}

mod on_device {
    use super::{instance, validation_tool};
    use crate::tests::devices_of_the_cfgs_kind;
    use crate::{Device, DeviceOptions};
    use ash::vk;

    /// A sampler whose `maxLod` is below its `minLod` is invalid usage
    /// (`VUID-VkSamplerCreateInfo-maxLod-01973`), and one a driver takes as it takes any clamp:
    /// the layer reports it and passes it down, and the test's log holds exactly that message.
    /// Made once, on the first device of the cfg's kind, through the device's own function table.
    /// Prints the validation tool each device lists.
    #[test]
    fn every_device_logs_a_deliberately_invalid_call_once() {
        let instance = instance(&[]).expecting(&["VUID-VkSamplerCreateInfo-maxLod-01973"]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let tool = validation_tool(&instance, name, physical.raw()).expect("a validation tool");
            println!("{name}: validation tool {tool}");
        }
        let (name, physical) = &devices[0];
        let device = Device::new(physical, DeviceOptions::default())
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let info = vk::SamplerCreateInfo::default().min_lod(1.0).max_lod(0.0);
        // SAFETY: the device is live. The info is invalid on purpose: its LOD range is empty.
        let made = unsafe { device.core().device.create_sampler(&info, None) };
        if let Ok(sampler) = made {
            // SAFETY: the sampler is the device's, unused.
            unsafe { device.core().device.destroy_sampler(sampler, None) };
        }
    }
}
