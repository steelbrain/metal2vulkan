//! The loader's reports, at `vkCreateInstance`, of each driver manifest whose library it could not
//! load, or that is no driver, and then ignores ([`IgnoredDrivers`]): the host's listing of its
//! drivers (a vendor's manifest left behind by a driver no longer installed is common), whatever
//! the device does, and nothing it did. The executor's sink passes those with a line of their own,
//! each naming the library the loader said it ignored.

/// The id name the Vulkan loader gives the messages it reports itself.
const LOADER_MESSAGE: &str = "Loader Message";

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
pub(crate) struct IgnoredDrivers(Vec<String>);

impl IgnoredDrivers {
    /// The libraries the loader said it ignored, among `messages`, each its id name and text.
    pub(crate) fn of<'a>(messages: impl IntoIterator<Item = (&'a str, &'a str)>) -> IgnoredDrivers {
        let libraries = (messages.into_iter())
            .filter(|&(id_name, _)| id_name == LOADER_MESSAGE)
            .filter_map(|(_, text)| {
                let library = text.strip_prefix(ICD_SCAN_FAILED)?;
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
    pub(crate) fn explains(&self, id_name: &str, text: &str) -> bool {
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

#[cfg(test)]
mod tests {
    use super::IgnoredDrivers;

    /// The loader's messages about a driver manifest it ignored, as a host left them, each its id
    /// name and text: one whose library is missing (the dynamic linker's error, then the
    /// loader's), one whose library loaded and gave no `vkCreateInstance`, and one whose library
    /// is no driver.
    fn ignored_drivers() -> Vec<(&'static str, &'static str)> {
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
        .map(|text| ("Loader Message", text))
        .to_vec()
    }

    /// The loader's reports of a driver manifest it ignored, each about the library it said it
    /// ignored, and nothing wider: the same words of another id name, a linker's error or a missing
    /// `vkCreateInstance` about a library it did not say it ignored, and its other messages.
    #[test]
    fn the_loader_s_reports_of_a_driver_manifest_it_ignored_are_matched_and_nothing_wider() {
        let reported = ignored_drivers();
        let ignored = IgnoredDrivers::of(reported.iter().copied());
        for &(id_name, text) in &reported {
            assert!(ignored.explains(id_name, text), "{text}");
        }
        for (id_name, text) in [
            // The same words from something that is not the loader.
            (
                "VUID-vkCreateInstance-None",
                "loader_icd_scan: Failed loading library associated with ICD JSON \
                 /nonexistent/libstale_vulkan.so. Ignoring this JSON",
            ),
            // A linker's error about a library the loader did not say it ignored.
            (
                "Loader Message",
                "/usr/lib/libvulkan_radeon.so: cannot open shared object file: No such file or \
                 directory",
            ),
            // No `vkCreateInstance` from a library the loader did not say it ignored.
            (
                "Loader Message",
                "loader_scanned_icd_add: Could not get 'vkCreateInstance' via \
                 'vk_icdGetInstanceProcAddr' for ICD libvulkan_radeon.so",
            ),
            // A manifest naming no library.
            (
                "Loader Message",
                "loader_icd_scan: Failed loading library associated with ICD JSON . Ignoring this \
                 JSON",
            ),
            // Another of the loader's messages.
            (
                "Loader Message",
                "terminator_CreateInstance: Failed to CreateInstance in ICD 0",
            ),
            // The library named, in other words.
            (
                "Loader Message",
                "/nonexistent/libstale_vulkan.so was loaded",
            ),
        ] {
            assert!(!ignored.explains(id_name, text), "{id_name}: {text}");
        }
        // Without the loader's word that it ignored the library, its linker error and its missing
        // `vkCreateInstance` stand alone and fail.
        let alone = IgnoredDrivers::of(reported[..1].iter().copied());
        assert!(!alone.explains(reported[0].0, reported[0].1));
        assert!(!alone.explains(reported[2].0, reported[2].1));
    }
}
