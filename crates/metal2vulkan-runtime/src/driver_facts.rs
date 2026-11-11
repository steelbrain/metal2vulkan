//! The runtime's informational messages and driver diagnostics. The facts, a driver's measured
//! answers where Vulkan leaves the result undefined
//! ([`DriverFact`](metal2vulkan::device::DriverFact)), are rows of the table type the description's
//! deviations share (`device::driver_facts`), keyed on the driver's identity and, where measured
//! on some alone, its GPU vendor and GPU: `VkPhysicalDeviceDriverProperties`'s `driverID` and
//! `VkPhysicalDeviceProperties`'s `vendorID` and `deviceID`, never its release. A device's facts
//! are in its properties ([`DeviceProperties::driver_facts`](super::DeviceProperties::driver_facts)).
//!
//! A driver diagnostic fact names a message the driver reports inside one call where Vulkan states
//! nothing wrong: inside that call alone the device withholds that message from its sink
//! ([`expecting`], [`withholds`]) and counts it (`WorkCounts::driver_diagnostics_withheld`).

use crate::{DebugMessage, MessageKind, MessageSeverity, MessageSink};
use metal2vulkan::device::DriverDiagnostic;

impl super::Device {
    /// Hand the informational message `text`, id name `id_name`, to the device's sink, else to
    /// stderr: what the runtime says that degrades nothing (the pipeline creation log).
    pub(crate) fn inform(&self, id_name: &str, text: String) {
        inform_through(self.shared().instance.sink(), id_name, text);
    }

    /// [`inform`](Self::inform) the message `text` gives, of kind `told`, unless the device said
    /// one of that kind before: then it is counted, not said (`Device::tell`).
    pub(crate) fn inform_once(
        &self,
        told: crate::degradation::Told,
        id_name: &str,
        text: impl FnOnce() -> String,
    ) {
        self.tell(told, || informational(id_name, text()));
    }
}

/// [`Device::inform`](crate::Device::inform) through `sink`, else on stderr, where the device is
/// ending.
pub(crate) fn inform_through(sink: Option<&MessageSink>, id_name: &str, text: String) {
    match sink {
        Some(sink) => sink.deliver(&informational(id_name, text)),
        None => eprintln!("{text}"),
    }
}

/// The informational message of id name `id_name` and text `text`, as a sink receives it.
fn informational(id_name: &str, text: String) -> DebugMessage {
    DebugMessage {
        severity: MessageSeverity::Info,
        kind: MessageKind {
            general: true,
            ..MessageKind::default()
        },
        id_name: id_name.into(),
        id: 0,
        text,
        degradation: None,
    }
}

/// The driver diagnostics ([`DriverDiagnostic`]) the thread's current driver call is measured to
/// report, and how many of them it reported so far: set only inside [`expecting`].
struct Expected {
    diagnostics: Vec<DriverDiagnostic>,
    withheld: u64,
}

thread_local! {
    static EXPECTED: std::cell::RefCell<Option<Expected>> = const { std::cell::RefCell::new(None) };
}

/// Run `call`, a driver call on this thread that the device's driver facts measure to report
/// `diagnostics` (`DriverFact::pipeline_creation_diagnostics`), withholding exactly those
/// messages from the device's message sink while it runs ([`withholds`]); with how many it
/// withheld. The driver reports from the calling thread inside the call (MoltenVK's
/// `MVKBaseObject::reportMessage`), so a message reported on another thread, or after the call
/// returns, relays. A pipeline making the runtime hands to a thread of its own at the
/// user-initiated class (`device/compiles.rs`) takes the expectation with it ([`expected_here`],
/// [`withheld_for_here`]). With no diagnostics, `call` alone.
pub(crate) fn expecting<T>(
    diagnostics: Vec<DriverDiagnostic>,
    call: impl FnOnce() -> T,
) -> (T, u64) {
    if diagnostics.is_empty() {
        return (call(), 0);
    }
    /// Clears the thread's expectation however `call` ends.
    struct Clear;
    impl Drop for Clear {
        fn drop(&mut self) {
            let _ = EXPECTED.try_with(|expected| expected.borrow_mut().take());
        }
    }
    EXPECTED.with(|expected| {
        *expected.borrow_mut() = Some(Expected {
            diagnostics,
            withheld: 0,
        })
    });
    let clear = Clear;
    let result = call();
    let withheld = EXPECTED.with(|expected| expected.borrow().as_ref().map_or(0, |e| e.withheld));
    drop(clear);
    (result, withheld)
}

/// The diagnostics every pipeline making on a device of `properties` is measured to report,
/// whatever its pipeline's state: those of its color attachments the driver cannot blend
/// (`DriverFact::BlendlessColorAttachmentsWarn`), one for each format the device lists as a color
/// attachment without blending ([`DeviceDescription::blendless_color_formats`]). The driver names
/// the format, so a pipeline that attaches none of them reports none.
///
/// [`DeviceDescription::blendless_color_formats`]: metal2vulkan::device::DeviceDescription::blendless_color_formats
pub(crate) fn making_diagnostics(properties: &crate::DeviceProperties) -> Vec<DriverDiagnostic> {
    let blendless = properties.description.blendless_color_formats();
    (properties.driver_facts.iter())
        .flat_map(|fact| {
            fact.pipeline_creation_diagnostics(
                &[],
                metal2vulkan::vk::PrimitiveTopology::TRIANGLE_LIST,
                &blendless,
            )
        })
        .collect()
}

/// [`expecting`] where this thread's current call expects nothing yet; inside an expectation,
/// `call` alone, which the outer expectation covers.
pub(crate) fn expecting_where_unset<T>(
    diagnostics: Vec<DriverDiagnostic>,
    call: impl FnOnce() -> T,
) -> T {
    let unset = EXPECTED
        .try_with(|expected| expected.try_borrow().is_ok_and(|e| e.is_none()))
        .unwrap_or(false);
    match unset {
        true => expecting(diagnostics, call).0,
        false => call(),
    }
}

/// The diagnostics this thread's current call is measured to report ([`expecting`]); none
/// outside such a call. A thread of the runtime's that makes the driver call for this one, while
/// it waits (`device/compiles.rs`), expects them inside that call alone.
pub(crate) fn expected_here() -> Vec<DriverDiagnostic> {
    EXPECTED
        .try_with(|expected| {
            expected
                .try_borrow()
                .ok()
                .and_then(|expected| expected.as_ref().map(|e| e.diagnostics.clone()))
        })
        .ok()
        .flatten()
        .unwrap_or_default()
}

/// Count `withheld` diagnostics as this thread's current call's ([`expecting`]): those the
/// thread that made the driver call for it withheld ([`expected_here`]).
pub(crate) fn withheld_for_here(withheld: u64) {
    if withheld == 0 {
        return;
    }
    let _ = EXPECTED.try_with(|expected| {
        if let Ok(mut expected) = expected.try_borrow_mut() {
            if let Some(expected) = expected.as_mut() {
                expected.withheld += withheld;
            }
        }
    });
}

/// Whether the driver's `message` is one the thread's current call is measured to report
/// ([`expecting`]): a warning of the validation kind whose name, number and text are a
/// diagnostic's exactly. Such a message is counted, and the caller withholds it.
pub(crate) fn withholds(message: &DebugMessage) -> bool {
    if message.severity != MessageSeverity::Warning
        || !message.kind.validation
        || message.degradation.is_some()
    {
        return false;
    }
    EXPECTED
        .try_with(|expected| {
            let Ok(mut expected) = expected.try_borrow_mut() else {
                return false;
            };
            let Some(expected) = expected.as_mut() else {
                return false;
            };
            let matched = expected.diagnostics.iter().any(|d| {
                d.id_name == message.id_name && d.id == message.id && d.text == message.text
            });
            expected.withheld += u64::from(matched);
            matched
        })
        .unwrap_or(false)
}
