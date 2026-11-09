//! What a device test's claim that the validation layer reported nothing holds, from what the
//! device's instance enabled of the layers it asked for.
//!
//! A device opens although a layer its caller asked for is one the loader cannot provide
//! (`device/create.rs`, `tests/missing_layer.rs`): the creation drops the whole layer list, creates
//! the instance with none, and the device says as it is made which layers it was made without
//! ([`LAYERS_DROPPED`]). That is right for a caller that asked for a device and wrong for a claim:
//! a test that requires the validation layer to have reported nothing passes by reporting nothing
//! at all, on every host the loader can give it no layer.
//!
//! So a test states the claim against the instance's own answer ([`Watched::layer_reported_nothing`]):
//! where the layer was enabled the claim holds over everything the layer checks, and where it was
//! not the test says so in its own output (`LAYER-ABSENT <claim>`) rather than passing on an empty
//! list. The decision is the instance's alone -- never the loader's listing, which a layer it lists
//! and cannot load passes alike, and never the host or the driver's name.
//!
//! One gap of the layer's passes, on the layer releases whose source was read with it: its
//! whole-pool report of a host query reset (`query_reset_gap.rs`), as the validation crate's
//! runtime cases let it pass ([`kept`]).
//!
//! And the loader's own reports of a driver manifest whose library it could not load, or that is
//! no driver, which it then ignores ([`IgnoredDrivers`]): the host's listing of its drivers (a
//! vendor's manifest left behind by a driver no longer installed is common), reported through the
//! messenger chained at `vkCreateInstance` whatever the device does, and nothing it did.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs.

use super::query_reset_gap;
use crate::device::create::LAYERS_DROPPED;
use crate::{DebugMessage, MessageSeverity, MessageSink};
use std::ffi::{CStr, CString};
use std::sync::{Arc, Mutex};

/// The validation layer, which this host may or may not provide.
pub(super) const VALIDATION: &CStr = c"VK_LAYER_KHRONOS_validation";

/// The id name the Vulkan loader gives the messages it reports itself.
const LOADER_MESSAGE: &str = "Loader Message";

/// What the loader's message about a layer it cannot provide names: the check it failed, and what
/// it found (`loader_validate_layers`, Vulkan-Loader `loader/loader.c`).
const VALIDATE_LAYERS: &str = "loader_validate_layers";

/// The rest of that message: the layer is none the loader lists.
const NOT_AVAILABLE: &str = "does not exist in the list of available layers";

/// Whether a message with this id name and text is the loader's own complaint about an instance
/// layer it cannot provide, and nothing wider. The loader reports it through the messenger chained
/// at `vkCreateInstance`, before the call answers `VK_ERROR_LAYER_NOT_PRESENT`.
fn loader_layer_message(id_name: &str, text: &str) -> bool {
    id_name == LOADER_MESSAGE && text.contains(VALIDATE_LAYERS) && text.contains(NOT_AVAILABLE)
}

/// How the loader's message about a driver manifest whose library it could not load begins
/// (`loader_icd_scan`, Vulkan-Loader `loader/loader.c`): the library the manifest names follows,
/// then [`IGNORING`].
const ICD_SCAN_FAILED: &str = "loader_icd_scan: Failed loading library associated with ICD JSON ";

/// How that message ends: the loader goes on without the manifest.
const IGNORING: &str = ". Ignoring this JSON";

/// How the loader's message about a driver library that loaded and gave no `vkCreateInstance`
/// begins (`loader_scanned_icd_add`): the library follows, and the message after it is
/// [`ICD_SCAN_FAILED`]'s about the same library.
const NO_CREATE_INSTANCE: &str = "loader_scanned_icd_add: Could not get 'vkCreateInstance' via \
     'vk_icdGetInstanceProcAddr' for ICD ";

/// How the loader's message about a library a driver manifest names that loaded and exports
/// neither entry point a driver does begins (`loader_scanned_icd_add`): the library follows, then
/// [`NO_ENTRY_POINT_END`]; the loader ignores the manifest with no other message.
const NO_ENTRY_POINT: &str = "loader_scanned_icd_add: Attempt to retrieve either \
     'vkGetInstanceProcAddr' or 'vk_icdGetInstanceProcAddr' from ICD ";

/// How that message ends.
const NO_ENTRY_POINT_END: &str = " failed.";

/// The loader's own reports, in one message list, of driver manifests it ignored: the libraries
/// it said it ignored for failing to load them ([`ICD_SCAN_FAILED`]), which explain its messages
/// about the same library alone.
struct IgnoredDrivers(Vec<String>);

impl IgnoredDrivers {
    /// The libraries the loader said it ignored, among `messages`.
    fn of(messages: &[DebugMessage]) -> IgnoredDrivers {
        let libraries = (messages.iter())
            .filter(|message| message.id_name == LOADER_MESSAGE)
            .filter_map(|message| {
                let library = message.text.strip_prefix(ICD_SCAN_FAILED)?;
                library
                    .strip_suffix(IGNORING)
                    .filter(|library| !library.is_empty())
            })
            .map(str::to_owned)
            .collect();
        IgnoredDrivers(libraries)
    }

    /// Whether a message with this id name and text is the loader's own report of a driver
    /// manifest it ignored, and nothing wider: its message that it ignored the library, the dynamic
    /// linker's error it forwards about that library (`<library>: <why>`), or its message that the
    /// library gave no `vkCreateInstance`; or its message that a library a manifest names is no
    /// driver, which names the library it ignored itself.
    fn explains(&self, id_name: &str, text: &str) -> bool {
        if id_name != LOADER_MESSAGE {
            return false;
        }
        let no_driver = (text.strip_prefix(NO_ENTRY_POINT))
            .and_then(|rest| rest.strip_suffix(NO_ENTRY_POINT_END))
            .is_some_and(|library| !library.is_empty());
        no_driver
            || self.0.iter().any(|library| {
                let named = |prefix: &str, suffix: &str| {
                    (text.strip_prefix(prefix))
                        .and_then(|rest| rest.strip_suffix(suffix))
                        .is_some_and(|named| named == library)
                };
                named(ICD_SCAN_FAILED, IGNORING)
                    || named(NO_CREATE_INSTANCE, "")
                    || (text.strip_prefix(library.as_str()))
                        .is_some_and(|why| why.starts_with(": "))
            })
    }
}

/// What one message list holds.
#[derive(Default)]
struct Seen {
    /// The messages that fail a run, bounded so a flood cannot grow without limit.
    failing: Vec<DebugMessage>,
    /// Whether a device reporting here announced it was made without the layers it asked for.
    dropped: bool,
    /// Whether a device reporting here asked for the validation layer.
    asked: bool,
    /// The release of the validation layer the system's loader offers, read as a device asked for
    /// it (`query_reset_gap::layer_release`): never inside the sink, which runs within the
    /// device's Vulkan calls.
    release: Option<u32>,
}

/// The most messages kept of one list.
const MAX: usize = 64;

/// The messages the devices of one test report, and what their instance enabled of the validation
/// layer they asked for.
#[derive(Clone, Default)]
pub(super) struct Watched(Arc<Mutex<Seen>>);

impl Watched {
    /// The instance layers a device reporting here asks for: the validation layer alone. The
    /// loader may be unable to provide it, which the device itself says
    /// ([`Watched::layer_reported_nothing`] reads that answer, not this list).
    pub(super) fn layers(&self) -> Vec<CString> {
        let release = query_reset_gap::layer_release();
        let mut seen = self.0.lock().unwrap_or_else(|p| p.into_inner());
        seen.asked = true;
        seen.release = release;
        vec![VALIDATION.to_owned()]
    }

    /// A sink reporting here, which calls `also` with each message first: a test's own work on the
    /// messages it watches for (a sentinel the sink owns, a line it prints).
    pub(super) fn sink(&self, also: impl Fn(&DebugMessage) + Send + Sync + 'static) -> MessageSink {
        let seen = Arc::clone(&self.0);
        MessageSink::new(move |message: &DebugMessage| {
            also(message);
            let mut seen = seen.lock().unwrap_or_else(|p| p.into_inner());
            // What the instance enabled of the layers the device asked for: informational, so it
            // fails nothing, and kept rather than dropped with the rest, since a claim of what the
            // validation layer reported holds nothing without it.
            if message.id_name == LAYERS_DROPPED {
                seen.dropped = true;
                return;
            }
            let fails = message.severity == MessageSeverity::Error
                || (message.severity == MessageSeverity::Warning && message.kind.validation);
            if fails && message.degradation.is_none() && seen.failing.len() < MAX {
                if let Some(message) = kept(message, seen.release) {
                    seen.failing.push(message);
                }
            }
        })
    }

    /// Require that the validation layer reported nothing of `claim` on the devices reporting
    /// here, and say what that holds.
    ///
    /// Every message the layer reported fails here, as `assert!(failures.is_empty())` did. What
    /// the claim then stands on is the instance's answer alone: where a device announced it was
    /// made without the layer it asked for, there was nothing to report, so the claim was not
    /// exercised and the test says so; the loader's own complaint about that same layer passes
    /// with it, and so do its reports of driver manifests it ignored ([`IgnoredDrivers`]), each
    /// printed (`LOADER-DRIVER <claim>`); nothing else does.
    ///
    /// # Panics
    ///
    /// Where the layer reported anything that fails a run.
    pub(super) fn layer_reported_nothing(&self, claim: &str) {
        let mut seen = self.0.lock().unwrap_or_else(|p| p.into_inner());
        let dropped = seen.dropped;
        let asked = seen.asked;
        let reported = seen.failing.split_off(0);
        drop(seen);
        let ignored = IgnoredDrivers::of(&reported);
        let failing: Vec<DebugMessage> = (reported.into_iter())
            // The device said it was made without the layer, which is what the loader complained
            // about first and all it complained about.
            .filter(|message| !(dropped && loader_layer_message(&message.id_name, &message.text)))
            .filter(|message| {
                let host = ignored.explains(&message.id_name, &message.text);
                if host {
                    eprintln!(
                        "LOADER-DRIVER {claim}: passed, the loader's own report of a driver \
                         manifest it ignored: {}",
                        message.text.replace('\n', " ")
                    );
                }
                !host
            })
            .collect();
        assert!(
            failing.is_empty(),
            "{claim}: the validation layer reported\n{failing:#?}"
        );
        if dropped || !asked {
            eprintln!(
                "LAYER-ABSENT {claim}: the device's instance enabled no validation layer, so it \
                 reported nothing at all; the claim was not exercised here"
            );
        }
    }
}

/// What a list keeps of `message`, one that fails a run, the validation layer of `release`
/// reporting: nothing where it is the layer's query-reset gap (`query_reset_gap.rs`) on a release
/// whose source was read with it, which passes and says so; the message marked "re-read
/// vkResetQueryPool in layer X.Y.Z" where it is the gap's on another release, or one unknown; else
/// the message as it is.
fn kept(message: &DebugMessage, release: Option<u32>) -> Option<DebugMessage> {
    if !query_reset_gap::shape(message.id, &message.id_name, &message.text) {
        return Some(message.clone());
    }
    if release.is_some_and(query_reset_gap::read) {
        eprintln!(
            "LAYER-GAP: passed, the layer's query-reset gap ({}): {} ({}): {}",
            query_reset_gap::SOURCE,
            message.id_name,
            message.id,
            message.text.replace('\n', " ")
        );
        return None;
    }
    let read: Vec<String> = (query_reset_gap::READ.iter())
        .map(|&version| query_reset_gap::release(version))
        .collect();
    let layer = release.map_or_else(|| "of an unknown release".into(), query_reset_gap::release);
    let mut message = message.clone();
    message.text = format!(
        "re-read vkResetQueryPool in layer {layer}: the layer's query-reset gap passes only on \
         the releases read ({}). {}",
        read.join(", "),
        message.text
    );
    Some(message)
}

#[cfg(test)]
mod tests {
    use super::{kept, loader_layer_message, IgnoredDrivers, Watched, VALIDATION};
    use crate::device::create::LAYERS_DROPPED;
    use crate::{DebugMessage, MessageKind, MessageSeverity};

    /// A message of the layer's or the loader's, as the sink hears it.
    fn message(severity: MessageSeverity, id_name: &str, text: &str) -> DebugMessage {
        DebugMessage {
            severity,
            kind: MessageKind {
                general: severity == MessageSeverity::Error,
                validation: false,
                performance: false,
            },
            id_name: id_name.to_owned(),
            id: 0,
            text: text.to_owned(),
            degradation: None,
        }
    }

    /// The layer's query-reset gap (`query_reset_gap.rs`) passes on the release whose source was
    /// read with it alone: on another, or one unknown, it is kept, marked for a re-read; any other
    /// message is kept as it is. Its own input: no device, no loader, no host.
    #[test]
    fn the_query_reset_gap_passes_on_the_releases_read_alone() {
        let mut gap = message(
            MessageSeverity::Error,
            "VUID-vkResetQueryPool-firstQuery-02741",
            &super::query_reset_gap::seen(),
        );
        gap.id = 150880399;
        let read = ash::vk::make_api_version(0, 1, 4, 363);
        assert!(
            kept(&gap, Some(read)).is_none(),
            "the gap, on the release read"
        );
        for release in [Some(ash::vk::make_api_version(0, 1, 4, 357)), None] {
            let held = kept(&gap, release).expect("the gap, on another release");
            assert!(
                held.text.starts_with("re-read vkResetQueryPool in layer "),
                "{release:?}: {}",
                held.text
            );
            assert!(
                held.text.ends_with(&gap.text),
                "{release:?}: the layer's text kept"
            );
        }
        let other = message(MessageSeverity::Error, "VUID-other", "another report");
        let held = kept(&other, Some(read)).expect("another message");
        assert_eq!(held.text, other.text, "another message as it is");
    }

    /// The loader's own complaint about the layer it cannot provide, and nothing wider.
    #[test]
    fn the_loader_s_message_about_a_layer_it_cannot_provide_is_matched_and_nothing_wider() {
        assert!(loader_layer_message(
            "Loader Message",
            "loader_validate_layers: Layer 0 does not exist in the list of available layers"
        ));
        for (id_name, text) in [
            (
                "Loader Message",
                "terminator_CreateInstance: Failed to CreateInstance in ICD 0",
            ),
            (
                "VUID-vkCreateInstance-ppEnabledLayerNames-01727",
                "loader_validate_layers: Layer 0 does not exist in the list of available layers",
            ),
            ("Loader Message", "loader_validate_layers: Layer 0 is fine"),
        ] {
            assert!(!loader_layer_message(id_name, text), "{id_name}: {text}");
        }
    }

    /// The loader's messages about a driver manifest it ignored, as a host left them: one whose
    /// library is missing (the dynamic linker's error, then the loader's), one whose library loaded
    /// and gave no `vkCreateInstance`, and one whose library is no driver.
    fn ignored_drivers() -> Vec<DebugMessage> {
        [
            "/nonexistent/libstale_vulkan.so: cannot open shared object file: No such file or \
             directory",
            "loader_icd_scan: Failed loading library associated with ICD JSON \
             /nonexistent/libstale_vulkan.so. Ignoring this JSON",
            "loader_scanned_icd_add: Could not get 'vkCreateInstance' via \
             'vk_icdGetInstanceProcAddr' for ICD libGLX_nvidia.so.0",
            "loader_icd_scan: Failed loading library associated with ICD JSON libGLX_nvidia.so.0. \
             Ignoring this JSON",
            "loader_scanned_icd_add: Attempt to retrieve either 'vkGetInstanceProcAddr' or \
             'vk_icdGetInstanceProcAddr' from ICD /usr/lib/libz.so.1 failed.",
        ]
        .map(|text| message(MessageSeverity::Error, "Loader Message", text))
        .to_vec()
    }

    /// The loader's reports of a driver manifest it ignored pass, each about the library it said
    /// it ignored, and nothing wider: the same words of another id name, a linker's error or a
    /// missing `vkCreateInstance` about a library it did not say it ignored, and its other
    /// messages about drivers fail.
    #[test]
    fn the_loader_s_reports_of_a_driver_manifest_it_ignored_are_matched_and_nothing_wider() {
        let reported = ignored_drivers();
        let ignored = IgnoredDrivers::of(&reported);
        for message in &reported {
            assert!(
                ignored.explains(&message.id_name, &message.text),
                "{}",
                message.text
            );
        }
        for (id_name, text) in [
            (
                "VUID-vkCreateInstance-None",
                "loader_icd_scan: Failed loading library associated with ICD JSON \
                 /nonexistent/libstale_vulkan.so. Ignoring this JSON",
            ),
            (
                "Loader Message",
                "/usr/lib/libvulkan_radeon.so: cannot open shared object file: No such file or \
                 directory",
            ),
            (
                "Loader Message",
                "loader_scanned_icd_add: Could not get 'vkCreateInstance' via \
                 'vk_icdGetInstanceProcAddr' for ICD libvulkan_radeon.so",
            ),
            (
                "Loader Message",
                "loader_icd_scan: Failed loading library associated with ICD JSON . Ignoring this \
                 JSON",
            ),
            (
                "Loader Message",
                "terminator_CreateInstance: Failed to CreateInstance in ICD 0",
            ),
            (
                "Loader Message",
                "/nonexistent/libstale_vulkan.so was loaded",
            ),
        ] {
            assert!(!ignored.explains(id_name, text), "{id_name}: {text}");
        }
        // Without the loader's word that it ignored the library, its linker error and its missing
        // `vkCreateInstance` stand alone and fail.
        let alone = IgnoredDrivers::of(&reported[..1]);
        assert!(!alone.explains(&reported[0].id_name, &reported[0].text));
        assert!(!alone.explains(&reported[2].id_name, &reported[2].text));
    }

    /// A claim holds over the loader's reports of driver manifests it ignored, the layer having
    /// reported nothing, and still fails on what the layer reported beside them.
    #[test]
    fn a_claim_holds_over_the_loader_s_reports_of_ignored_drivers_alone() {
        let watched = Watched::default();
        assert_eq!(watched.layers(), [VALIDATION.to_owned()]);
        let sink = watched.sink(|_| {});
        for message in ignored_drivers() {
            sink.deliver(&message);
        }
        watched.layer_reported_nothing("every command was valid");
        for message in ignored_drivers() {
            sink.deliver(&message);
        }
        sink.deliver(&message(
            MessageSeverity::Error,
            "VUID-vkCmdDraw-None-08600",
            "a bound descriptor set was not valid",
        ));
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            watched.layer_reported_nothing("every command was valid")
        }));
        assert!(
            failed.is_err(),
            "the claim held over a message the layer reported"
        );
    }

    /// The message the loader sends about the layer it cannot provide.
    fn loader_complaint() -> DebugMessage {
        message(
            MessageSeverity::Error,
            "Loader Message",
            "loader_validate_layers: Layer 0 does not exist in the list of available layers",
        )
    }

    /// The message a device sends as it is made without the layers it asked for.
    fn made_without() -> DebugMessage {
        message(
            MessageSeverity::Info,
            LAYERS_DROPPED,
            "metal2vulkan: the device was made without the instance layer(s) \
             VK_LAYER_KHRONOS_validation",
        )
    }

    /// A list whose device asked for the validation layer and was made without it states no claim
    /// of the layer's: the loader's own complaint about that layer passes, and the claim prints
    /// that it was not exercised rather than passing on an empty list. Its own input: no device,
    /// no loader, no host.
    #[test]
    fn a_claim_on_a_device_made_without_the_layer_is_not_exercised() {
        let watched = Watched::default();
        assert_eq!(watched.layers(), [VALIDATION.to_owned()]);
        let sink = watched.sink(|_| {});
        sink.deliver(&loader_complaint());
        sink.deliver(&made_without());
        watched.layer_reported_nothing("nothing of the layer's");
    }

    /// The loader's complaint fails where no device says it was made without the layer: a loader
    /// that refused a layer nothing accounts for is not a clean run.
    #[test]
    #[should_panic(expected = "the validation layer reported")]
    fn the_loader_s_complaint_fails_where_no_device_explains_it() {
        let watched = Watched::default();
        let sink = watched.sink(|_| {});
        sink.deliver(&loader_complaint());
        watched.layer_reported_nothing("nothing of the layer's");
    }

    /// A list whose device enabled the layer it asked for and had it report nothing holds the
    /// claim, and anything the layer did report fails it: the check is not vacuous either way.
    #[test]
    fn a_claim_holds_where_the_layer_was_enabled_and_fails_on_what_it_reported() {
        let watched = Watched::default();
        assert_eq!(watched.layers(), [VALIDATION.to_owned()]);
        let sink = watched.sink(|_| {});
        watched.layer_reported_nothing("every command was valid");
        sink.deliver(&message(
            MessageSeverity::Error,
            "VUID-vkCmdDraw-None-08600",
            "a bound descriptor set was not valid",
        ));
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            watched.layer_reported_nothing("every command was valid")
        }));
        assert!(
            failed.is_err(),
            "the claim held over a message the layer reported"
        );
    }

    /// The test's own work on each message runs too, beside the list's.
    #[test]
    fn a_test_s_own_receiver_hears_every_message() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;
        let heard = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&heard);
        let watched = Watched::default();
        let sink = watched.sink(move |_| {
            counted.fetch_add(1, Ordering::Relaxed);
        });
        sink.deliver(&made_without());
        sink.deliver(&loader_complaint());
        assert_eq!(heard.load(Ordering::Relaxed), 2);
        watched.layer_reported_nothing("nothing of the layer's");
    }
}
