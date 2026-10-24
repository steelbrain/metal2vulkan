//! What a device validation makes through the runtime enabled of the instance layers it asked
//! for, and what a claim that needs the validation layer therefore holds.
//!
//! A device opens although a layer its caller asked for is one the loader cannot provide: the
//! runtime drops the whole layer list, creates the instance with none, and says as the device is
//! made which layers it was made without ([`LAYERS_DROPPED`], an informational message, no degrade
//! notice: the layers are diagnostics and nothing of Metal's answers differs). That is the right
//! behaviour for a caller that asked for a device, and the wrong silence for a claim: a test that
//! requires the validation layer to have reported nothing passes by reporting nothing at all.
//!
//! So the sink keeps that message instead of swallowing it ([`LayerFacts::heard`]), beside the
//! layers its device asked for ([`LayerFacts::asked`]), and a layer-dependent claim is stated
//! against the two ([`layer_reported_nothing`]): where the layer was enabled the claim holds over
//! what the layer saw, and where it was dropped the claim says in its own output that it was not
//! exercised here (`LANE-LAYER-ABSENT <test>`, and a count per message list as it goes), so an
//! empty message list is never read as a clean run.
//!
//! The loader says so too, and first: it reports the layer it cannot provide through the messenger
//! chained at `vkCreateInstance`, as an error of the general type, before the call answers
//! `VK_ERROR_LAYER_NOT_PRESENT` ([`loader_layer_message`]). That message is the dropped layer and
//! nothing else wrong, so it is held until the device says whether it was made without the layer
//! ([`LayerFacts::hold`]): once the device says so, the held message passes with a line of its own,
//! and where nothing says so it fails with the rest ([`LayerFacts::release`]). Without that, every
//! claim on such a device fails on the loader's own message instead of running its own work.
//!
//! The decision is the instance's alone: the layers the device asked for, less those it announced
//! it was made without. Never the host, the driver's name or an environment variable.

use super::Messages;
use metal2vulkan_runtime::DebugMessage;
use std::ffi::CString;

/// The id name the Vulkan loader gives the messages it reports itself.
const LOADER_MESSAGE: &str = "Loader Message";

/// What the loader's message about a layer it cannot provide names: the check it failed, and what
/// it found (`loader_validate_layers`, Vulkan-Loader `loader/loader.c`).
const VALIDATE_LAYERS: &str = "loader_validate_layers";

/// The rest of that message: the layer is none the loader lists.
const NOT_AVAILABLE: &str = "does not exist in the list of available layers";

/// Whether a message with this id name and text is the loader's own complaint about an instance
/// layer it cannot provide, and nothing wider.
pub(crate) fn loader_layer_message(id_name: &str, text: &str) -> bool {
    id_name == LOADER_MESSAGE && text.contains(VALIDATE_LAYERS) && text.contains(NOT_AVAILABLE)
}

/// The id name of the informational message a device sends as it is made where the loader could
/// not provide the instance layers its caller asked for; the runtime's `device/create.rs` names it
/// so.
pub const LAYERS_DROPPED: &str = "metal2vulkan-instance-layers-dropped";

/// What the instance of the devices reporting to one message list enabled of the layers they asked
/// for: the list asked for, and whether the loader could not provide it. The runtime drops the
/// whole list on a miss and creates the instance with none, so the enabled set is either all of
/// what was asked for or nothing.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EnabledLayers {
    asked: Vec<CString>,
    dropped: bool,
}

impl EnabledLayers {
    /// The instance layers the device asked for, in the order it asked for them.
    pub fn asked(&self) -> &[CString] {
        &self.asked
    }

    /// The layers the instance enabled: those asked for, or none where the loader could not
    /// provide them.
    pub fn enabled(&self) -> &[CString] {
        match self.dropped {
            true => &[],
            false => &self.asked,
        }
    }

    /// The layers the loader could not provide, which the instance was created without.
    pub fn dropped(&self) -> &[CString] {
        match self.dropped {
            true => &self.asked,
            false => &[],
        }
    }

    /// Whether the instance enabled `layer`.
    pub fn enabled_layer(&self, layer: &std::ffi::CStr) -> bool {
        self.enabled().iter().any(|name| name.as_c_str() == layer)
    }

    /// Whether the instance enabled the validation layer, so that what it reported is evidence.
    pub fn validates(&self) -> bool {
        self.enabled_layer(super::VALIDATION_LAYER)
    }
}

/// What a claim that needs the validation layer holds, stated against what the instance enabled
/// ([`layer_reported_nothing`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LayerClaim {
    /// The device enabled the validation layer, and it reported nothing: the claim holds over
    /// everything the layer checks.
    Held,
    /// The device asked for the validation layer and the loader could not provide it (the layers
    /// it was made without, as it announced them): there was nothing to report, so the claim was
    /// not exercised here. Printed as such, never a silent pass.
    NotExercised { dropped: Vec<String> },
    /// The device asked for no validation layer, so the claim is none of the layer's: a device a
    /// test makes without it, to time it.
    NoLayerAsked,
}

impl LayerClaim {
    /// Whether the claim was exercised: the layer was there to report.
    pub fn exercised(&self) -> bool {
        matches!(self, LayerClaim::Held)
    }
}

/// Require that the validation layer reported nothing of `claim` on the devices reporting to
/// `messages`, and say what that holds.
///
/// Every message the layer reported fails here, as `assert_eq!(messages.take(), Vec::new())` did,
/// and the messages are taken either way. What the claim then stands on is the instance's answer
/// alone ([`EnabledLayers`]): [`LayerClaim::Held`] where the layer was enabled, else a claim that
/// was not exercised, printed on stderr with the test's name.
///
/// # Panics
///
/// Where the layer reported anything that fails a run.
pub fn layer_reported_nothing(messages: &Messages, claim: &str) -> LayerClaim {
    let reported = messages.take();
    let stated = messages.state_layer_claim(claim);
    assert!(
        reported.is_empty(),
        "{claim}: the validation layer reported\n{reported:#?}"
    );
    stated
}

/// What one message list knows of the layers of the devices reporting to it.
#[derive(Clone, Debug, Default)]
pub(super) struct LayerFacts {
    /// The instance layers the device reporting here asked for, in order; empty until a device
    /// records them ([`LayerFacts::asked`]).
    asked: Vec<CString>,
    /// The text of [`LAYERS_DROPPED`], where such a device announced it was made without them.
    dropped: Option<String>,
    /// How many layer-dependent claims were stated here while the layers were dropped.
    unexercised: usize,
    /// Whether the absence was printed already, so one device's many claims print one line and a
    /// count.
    said: bool,
    /// The loader's own messages about a layer it cannot provide, held until the device says
    /// whether it was made without that layer ([`LayerFacts::hold`]).
    held: Vec<DebugMessage>,
    /// How many of them passed, the device having said so.
    passed: usize,
}

/// The most of the loader's own messages one list holds: a flood is nothing it can explain.
const MAX_HELD: usize = 8;

impl LayerFacts {
    /// A device reporting here asks for `layers`: what an earlier device of this list announced
    /// about its own layers is not the new one's.
    pub(super) fn asked(&mut self, layers: &[CString]) {
        self.asked = layers.to_vec();
        self.dropped = None;
        self.said = false;
    }

    /// Take the message `id_name` with text `text` of the test `test`, where it is the one a
    /// device sends as it is made without the layers it asked for; whether it was taken.
    pub(super) fn heard(&mut self, test: &str, id_name: &str, text: &str) -> bool {
        if id_name != LAYERS_DROPPED {
            return false;
        }
        self.dropped = Some(text.to_owned());
        // The loader said the same thing first, about the same layers: the device's announcement
        // explains it, so it passes with a line of its own.
        for message in self.held.split_off(0) {
            self.passed += 1;
            eprintln!(
                "LANE-LAYER-ABSENT {}: passed, the loader's own message about the layer the \
                 device was then made without: {} ({}): {}",
                test,
                message.id_name,
                message.id,
                message.text.replace('\n', " ")
            );
        }
        true
    }

    /// Whether the loader's own complaint about a layer it cannot provide is held here rather
    /// than failed at once: only where this list's device asked for a layer at all (with none
    /// asked for, the loader's complaint is about nothing of ours), and only up to [`MAX_HELD`].
    pub(super) fn holds(&self) -> bool {
        !self.asked.is_empty() && self.held.len() < MAX_HELD
    }

    /// Hold `message`, the loader's own complaint, until the device says whether it was made
    /// without that layer; whether it was held ([`LayerFacts::holds`]).
    pub(super) fn hold(&mut self, message: &DebugMessage) -> bool {
        if !self.holds() {
            return false;
        }
        self.held.push(message.clone());
        true
    }

    /// Whether a device of this list said it was made without the layers it asked for, which is
    /// what explains the loader's complaint.
    pub(super) fn explained(&self) -> bool {
        self.dropped.is_some()
    }

    /// The held messages nothing explained, which fail with the rest: a loader that refused a
    /// layer where no device of this list says it was made without one.
    pub(super) fn release(&mut self) -> Vec<DebugMessage> {
        match self.explained() {
            true => Vec::new(),
            false => self.held.split_off(0),
        }
    }

    /// What the instance of the devices reporting here enabled of the layers they asked for.
    pub(super) fn enabled(&self) -> EnabledLayers {
        EnabledLayers {
            asked: self.asked.clone(),
            dropped: self.dropped.is_some(),
        }
    }

    /// The text of the message a device of this list sent as it was made without the layers it
    /// asked for.
    pub(super) fn dropped_message(&self) -> Option<&str> {
        self.dropped.as_deref()
    }

    /// State `claim` of `test` against the layers the instance enabled, printing the first claim
    /// a layerless device cannot exercise.
    pub(super) fn claim(&mut self, test: &str, claim: &str) -> LayerClaim {
        let enabled = self.enabled();
        if enabled.validates() {
            return LayerClaim::Held;
        }
        let dropped: Vec<String> = enabled
            .dropped()
            .iter()
            .map(|name| name.to_string_lossy().into_owned())
            .collect();
        if dropped.is_empty() {
            // A device that asked for no validation layer: the claim is none of the layer's.
            return LayerClaim::NoLayerAsked;
        }
        self.unexercised += 1;
        if !self.said {
            self.said = true;
            eprintln!(
                "LANE-LAYER-ABSENT {test}: the loader could not provide {}, so the validation \
                 layer reported nothing at all; every claim of what it reported is not exercised \
                 here. First: {claim}",
                dropped.join(", ")
            );
        }
        LayerClaim::NotExercised { dropped }
    }

    /// The lines a message list prints as it goes: the claims no layer exercised, and the
    /// loader's own messages that the device's announcement explained.
    pub(super) fn report(&self, test: &str) {
        if self.unexercised != 0 {
            eprintln!(
                "LANE-LAYER-ABSENT count {test}: {} layer-dependent claim(s) stated with no \
                 validation layer, so not exercised",
                self.unexercised
            );
        }
        if self.passed != 0 {
            eprintln!(
                "LANE-LAYER-ABSENT count {test}: {} message(s) of the loader's about the layer \
                 the device was made without passed",
                self.passed
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{layer_reported_nothing, LayerClaim, LAYERS_DROPPED};
    use crate::candidate::runtime_case::{Messages, VALIDATION_LAYER};
    use std::ffi::CString;

    /// The message a device sends as it is made without the layers it asked for, as the runtime
    /// writes it (`device/create.rs`, `layers_dropped`): its id name is what the sink keys on, and
    /// the text is the device's own.
    fn dropped_message() -> String {
        format!(
            "metal2vulkan: the device on a device was made without the instance layer(s) {} \
             [{LAYERS_DROPPED}]",
            VALIDATION_LAYER.to_string_lossy()
        )
    }

    /// A message list of a device that asked for the validation layer and was made without it,
    /// the loader having none to provide. Its own input: no device, no loader, no host.
    fn layerless() -> Messages {
        let messages = Messages::default();
        messages.device_asked_layers(&[VALIDATION_LAYER.to_owned()]);
        assert!(
            messages.heard_informational(LAYERS_DROPPED, &dropped_message()),
            "the sink keeps the message a device sends as it is made without its layers"
        );
        messages
    }

    /// A claim of what the validation layer reported, stated on a list whose device was made
    /// without it, is not exercised: it names the layer that was dropped, and the message list is
    /// empty, so the claim the lane used to state (`messages.take() == Vec::new()`) passed on the
    /// same input. This test fails if [`layer_reported_nothing`] ever answers [`LayerClaim::Held`]
    /// with no layer present.
    #[test]
    fn a_claim_of_what_the_layer_reported_is_not_exercised_where_the_layer_was_dropped() {
        let messages = layerless();
        // What the instance enabled: nothing of what it asked for.
        let enabled = messages.enabled_layers();
        assert_eq!(enabled.asked(), [VALIDATION_LAYER.to_owned()]);
        assert_eq!(enabled.enabled(), [] as [CString; 0]);
        assert_eq!(enabled.dropped(), [VALIDATION_LAYER.to_owned()]);
        assert!(!enabled.validates());
        assert_eq!(messages.layers_dropped_message(), Some(dropped_message()));
        // The bad input the old claim passed on: nothing was reported, because nothing could be.
        assert_eq!(
            messages.take(),
            Vec::new(),
            "the layerless list reports nothing at all"
        );
        assert_eq!(
            layer_reported_nothing(&messages, "no Vulkan object leaked"),
            LayerClaim::NotExercised {
                dropped: vec![VALIDATION_LAYER.to_string_lossy().into_owned()],
            },
            "a claim with no layer present must say it was not exercised"
        );
        assert!(!layer_reported_nothing(&messages, "and again").exercised());
    }

    /// The same claim on a list whose device enabled the layer it asked for holds: the layer was
    /// there to report, and reported nothing.
    #[test]
    fn a_claim_of_what_the_layer_reported_holds_where_the_layer_was_enabled() {
        let messages = Messages::default();
        messages.device_asked_layers(&[VALIDATION_LAYER.to_owned()]);
        let enabled = messages.enabled_layers();
        assert_eq!(enabled.enabled(), [VALIDATION_LAYER.to_owned()]);
        assert!(enabled.validates());
        assert_eq!(messages.layers_dropped_message(), None);
        let claim = layer_reported_nothing(&messages, "no Vulkan object leaked");
        assert_eq!(claim, LayerClaim::Held);
        assert!(claim.exercised());
    }

    /// A device a test makes with no layer at all, to time it, states no claim of the layer's.
    #[test]
    fn a_device_that_asked_for_no_layer_states_no_claim_of_the_layer_s() {
        let messages = Messages::default();
        messages.device_asked_layers(&[]);
        assert!(!messages.enabled_layers().validates());
        let claim = layer_reported_nothing(&messages, "nothing of the layer's");
        assert_eq!(claim, LayerClaim::NoLayerAsked);
        assert!(!claim.exercised());
    }

    /// A device made again on one list answers for itself: the layers an earlier device of the
    /// list was made without are not the new one's, which enabled every layer it asked for.
    #[test]
    fn a_later_device_on_one_list_answers_for_its_own_layers() {
        let messages = layerless();
        assert!(!messages.enabled_layers().validates());
        messages.device_asked_layers(&[VALIDATION_LAYER.to_owned()]);
        assert!(
            messages.enabled_layers().validates(),
            "the new device announced nothing, so it was made with the layer it asked for"
        );
        assert_eq!(messages.layers_dropped_message(), None);
        assert_eq!(
            layer_reported_nothing(&messages, "no Vulkan object leaked"),
            LayerClaim::Held
        );
    }

    /// The loader's complaint is held only where this list's device asked for a layer, and it is
    /// explained only once a device said it was made without one: so a loader that refuses a
    /// layer no device accounts for still fails the run (`Messages::take` releases it).
    #[test]
    fn the_loader_s_message_is_held_for_a_device_that_asked_and_explained_by_its_answer() {
        use super::LayerFacts;
        let mut facts = LayerFacts::default();
        assert!(
            !facts.holds(),
            "with no layer asked for, the loader's complaint is about nothing of ours"
        );
        facts.asked(&[VALIDATION_LAYER.to_owned()]);
        assert!(
            facts.holds(),
            "a device that asked for a layer accounts for it"
        );
        assert!(
            !facts.explained(),
            "nothing explains it until a device says it was made without the layer"
        );
        assert!(facts.heard("a test", LAYERS_DROPPED, &dropped_message()));
        assert!(facts.explained());
        // A device made again asks afresh: its own answer explains its own loader messages.
        facts.asked(&[VALIDATION_LAYER.to_owned()]);
        assert!(!facts.explained());
        assert!(facts.release().is_empty(), "nothing was held");
    }

    /// The loader's own complaint about the layer it cannot provide, and nothing wider.
    #[test]
    fn the_loader_s_message_about_a_layer_it_cannot_provide_is_matched_and_nothing_wider() {
        assert!(super::loader_layer_message(
            "Loader Message",
            "loader_validate_layers: Layer 0 does not exist in the list of available layers"
        ));
        for (id_name, text) in [
            // Another of the loader's messages.
            (
                "Loader Message",
                "terminator_CreateInstance: Failed to CreateInstance in ICD 0",
            ),
            // The same words from something that is not the loader.
            (
                "VUID-vkCreateInstance-ppEnabledLayerNames-01727",
                "loader_validate_layers: Layer 0 does not exist in the list of available layers",
            ),
            // Half the words.
            ("Loader Message", "loader_validate_layers: Layer 0 is fine"),
        ] {
            assert!(
                !super::loader_layer_message(id_name, text),
                "{id_name}: {text}"
            );
        }
    }

    /// Every other informational message is none of this list's layer facts: the sink keeps the
    /// one message and leaves the rest where they were.
    #[test]
    fn no_other_informational_message_is_taken_for_the_layers() {
        let messages = Messages::default();
        messages.device_asked_layers(&[VALIDATION_LAYER.to_owned()]);
        for id_name in [
            "metal2vulkan-unmeasured-driver",
            "metal2vulkan-driver-pipeline-cache-kept",
            "metal2vulkan-pipeline-creation",
            "",
        ] {
            assert!(
                !messages.heard_informational(id_name, "a message"),
                "{id_name} is not the dropped-layer message"
            );
        }
        assert_eq!(messages.layers_dropped_message(), None);
        assert!(messages.enabled_layers().validates());
    }
}
