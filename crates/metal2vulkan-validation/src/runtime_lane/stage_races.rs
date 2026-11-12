//! The validation layer's hazards between a draw's vertex stage and an earlier draw's fragment
//! stage of one render pass, on a buffer both reach, where the device announced the difference
//! that race is. A lane claim that lets those pass, and nothing wider.
//!
//! Metal runs a pass's vertex functions before its fragments: a probe (an M3 Max, and an M4) read,
//! from a later draw's vertex function, the bytes from before an earlier draw's fragment writes,
//! and an earlier fragment read a later vertex function's write (`pass_start_reads.rs`). Vulkan
//! orders a pass's draws only through the barriers the pass records. Where the runtime keeps no
//! copy of the bytes from before the pass, its device answers the vertex stage's access in its
//! own order, and it announces that:
//!
//! - **`in-place-vertex-reads`**: a vertex stage reads in place bytes a fragment stage of its pass
//!   wrote (a vertex stage of the pass wrote them too, so no copy from before the pass holds what
//!   the reader reads): the layer reports a `READ_AFTER_WRITE` whose later access is the vertex
//!   stage's read.
//! - **`fragments-before-geometry`**: a vertex stage writes bytes a fragment stage of its pass read
//!   or wrote: the layer reports a `WRITE_AFTER_READ` or `WRITE_AFTER_WRITE` whose later access is
//!   the vertex stage's write and whose earlier one the fragment stage's.
//!
//! A test names each buffer whose such accesses it makes
//! ([`layer_reported_nothing_but_announced_stage_races`]). A message passes only when it is
//! reported as the later draw was recorded (both commands draws of one `VkCommandBuffer`), on a
//! buffer the test named, its later access the vertex stage's, of the kind above whose notice the
//! device announced, and, for a write, its earlier access the fragment stage's. It passes with a
//! `LAYER-STAGE-RACE` line. Everything else fails as
//! [`layer_reported_nothing`] fails it: a racing form, an image, a command that is not a draw, a
//! buffer the test did not name, a later access of another stage, and either kind where its notice
//! was not announced.
//!
//! [`layer_reported_nothing`]: super::layer_reported_nothing

use crate::parity::sequences::hazards::{parse, Named};
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
use {
    super::{LayerClaim, Messages},
    ash::vk::Handle as _,
    metal2vulkan_runtime::{Buffer, DebugMessage, Degradation},
};

/// The Vulkan commands the runtime records a Metal draw with, as the layer names them.
const DRAWS: &[&str] = &[
    "vkCmdDraw",
    "vkCmdDrawIndexed",
    "vkCmdDrawIndirect",
    "vkCmdDrawIndexedIndirect",
];

/// How the layer names the later access's stage where a shader's access through a descriptor is
/// the later one.
const VERTEX_ACCESS: &str = " hazard detected. Shader stage VK_SHADER_STAGE_VERTEX_BIT ";

/// How the layer begins naming the earlier access of a write's hazard, and its stage where it is
/// a fragment shader's.
const PRIOR: &str = "does not conflict with a prior ";
const FRAGMENT_STAGE: &str = "at VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT";

/// The notice a race of a hazard of id name `id_name` is announced by, where it is one: a vertex
/// stage's read after a write (`in-place-vertex-reads`), or its write after a read or a write
/// (`fragments-before-geometry`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Race {
    InPlaceRead,
    WriteBeforeFragments,
}

impl Race {
    fn of(id_name: &str) -> Option<Race> {
        match id_name {
            "SYNC-HAZARD-READ-AFTER-WRITE" => Some(Race::InPlaceRead),
            "SYNC-HAZARD-WRITE-AFTER-READ" | "SYNC-HAZARD-WRITE-AFTER-WRITE" => {
                Some(Race::WriteBeforeFragments)
            }
            _ => None,
        }
    }
}

/// Whether the layer's message of id name `id_name` and text `text` passes against the buffers
/// `named` (their raw handles), where `announced` says whether the device announced each race's
/// notice (the module docs).
fn passes(named: &[u64], announced: impl Fn(Race) -> bool, id_name: &str, text: &str) -> bool {
    let Some(race) = Race::of(id_name) else {
        return false;
    };
    let Some(hazard) = parse(id_name, text) else {
        return false;
    };
    let Named::Buffer(buffer) = hazard.memory else {
        return false;
    };
    let recorded = DRAWS
        .iter()
        .any(|draw| text.starts_with(&format!("{draw}(): ")));
    let draws = DRAWS.contains(&hazard.later.as_str()) && DRAWS.contains(&hazard.earlier.as_str());
    // The earlier access, up to the sentence's end: "a prior read (...) at <stage>."
    let prior = (text.split_once(PRIOR)).map(|(_, prior)| prior.split('.').next().unwrap_or(""));
    let fragment_prior = match race {
        Race::InPlaceRead => true,
        Race::WriteBeforeFragments => prior.is_some_and(|prior| prior.contains(FRAGMENT_STAGE)),
    };
    recorded
        && draws
        && named.contains(&buffer)
        && text.contains(VERTEX_ACCESS)
        && fragment_prior
        && announced(race)
}

/// [`layer_reported_nothing`](super::layer_reported_nothing) of `claim`, but for the layer's
/// hazards between a vertex stage and an earlier fragment stage of one pass on one of `buffers`
/// whose notice the device announced, each of which passes with a `LAYER-STAGE-RACE` line (the
/// module docs).
///
/// # Panics
///
/// Where the layer reported anything else.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
pub(super) fn layer_reported_nothing_but_announced_stage_races(
    messages: &Messages,
    claim: &str,
    buffers: &[&Buffer],
) -> LayerClaim {
    let reported = messages.take();
    let named: Vec<u64> = match reported.is_empty() {
        true => Vec::new(),
        // SAFETY: the handles are only compared with the layer's reports, never used.
        false => (buffers.iter())
            .map(|buffer| unsafe { buffer.vk_buffer() }.0.as_raw())
            .collect(),
    };
    let announced = |race| {
        messages.announced(match race {
            Race::InPlaceRead => Degradation::InPlaceVertexReads,
            Race::WriteBeforeFragments => Degradation::FragmentsBeforeGeometry,
        })
    };
    let test = std::thread::current().name().unwrap_or_default().to_owned();
    let rest: Vec<DebugMessage> = reported
        .into_iter()
        .filter(|message| {
            if !passes(&named, announced, &message.id_name, &message.text) {
                return true;
            }
            eprintln!(
                "LAYER-STAGE-RACE {test}: passed, a race between a vertex stage and an earlier \
                 fragment stage of its pass, announced: {} ({}): {}",
                message.id_name,
                message.id,
                message.text.replace('\n', " ")
            );
            false
        })
        .collect();
    let stated = messages.state_layer_claim(claim);
    assert!(
        rest.is_empty(),
        "{claim}: the validation layer reported\n{rest:#?}"
    );
    stated
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The buffer the hazards below name.
    const BUFFER: u64 = 0x4700_0000_0047;

    /// The in-place read as the layer 1.4.366 reported it on lavapipe
    /// (`pass_start_reads::a_vertex_stage_reads_in_place_what_another_stage_of_its_pass_wrote`).
    const IN_PLACE: &str = "vkCmdDraw(): READ_AFTER_WRITE hazard detected. Shader stage \
        VK_SHADER_STAGE_VERTEX_BIT reads VkBuffer 0x470000000047, which was previously written by \
        another vkCmdDraw command.  The buffer is referenced by descriptor binding 0 \
        (VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) from descriptor set 0 (VkDescriptorSet 0x0), \
        VkPipeline 0x7e000000007e. The current synchronization allows \
        VK_ACCESS_2_SHADER_READ_BIT|VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT accesses at \
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, but to prevent this hazard, it must allow \
        VK_ACCESS_2_UNIFORM_READ_BIT accesses at VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT.";

    /// The vertex stage's write after a fragment's read, as the layer 1.4.366 reported it on
    /// lavapipe (`pass_start_reads::a_vertex_stage_s_write_after_a_fragment_s_access_is_announced`).
    const WRITE_AFTER_FRAGMENT: &str = "vkCmdDraw(): WRITE_AFTER_READ hazard detected. Shader \
        stage VK_SHADER_STAGE_VERTEX_BIT writes to VkBuffer 0x470000000047, which was previously \
        read by another vkCmdDraw command.  The buffer is referenced by descriptor binding 1 \
        (VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) from descriptor set 0 (VkDescriptorSet 0x0), \
        VkPipeline 0x460000000046. No sufficient synchronization is present to ensure that a \
        write (VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT) at VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT \
        does not conflict with a prior read (VK_ACCESS_2_SHADER_STORAGE_READ_BIT) at \
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT.";

    const RAW: &str = "SYNC-HAZARD-READ-AFTER-WRITE";
    const WAR: &str = "SYNC-HAZARD-WRITE-AFTER-READ";

    fn every(_: Race) -> bool {
        true
    }

    #[test]
    fn announced_races_on_a_named_buffer_pass() {
        assert!(passes(&[BUFFER], every, RAW, IN_PLACE));
        assert!(passes(&[BUFFER], every, WAR, WRITE_AFTER_FRAGMENT));
    }

    #[test]
    fn a_race_whose_notice_was_not_announced_fails() {
        let only = |announced: Race| move |race: Race| race == announced;
        assert!(!passes(
            &[BUFFER],
            only(Race::WriteBeforeFragments),
            RAW,
            IN_PLACE
        ));
        assert!(!passes(
            &[BUFFER],
            only(Race::InPlaceRead),
            WAR,
            WRITE_AFTER_FRAGMENT
        ));
        assert!(!passes(&[BUFFER], |_| false, RAW, IN_PLACE));
    }

    #[test]
    fn a_buffer_the_test_did_not_name_fails() {
        assert!(!passes(&[], every, RAW, IN_PLACE));
        assert!(!passes(
            &[0x5100_0000_0051],
            every,
            WAR,
            WRITE_AFTER_FRAGMENT
        ));
    }

    #[test]
    fn a_later_access_of_another_stage_fails() {
        let fragment =
            IN_PLACE.replace("VK_SHADER_STAGE_VERTEX_BIT", "VK_SHADER_STAGE_FRAGMENT_BIT");
        assert!(!passes(&[BUFFER], every, RAW, &fragment));
        let later = WRITE_AFTER_FRAGMENT
            .replace("VK_SHADER_STAGE_VERTEX_BIT", "VK_SHADER_STAGE_FRAGMENT_BIT");
        assert!(!passes(&[BUFFER], every, WAR, &later));
        // A vertex stage's write after another vertex stage's read is no fragment's race.
        let vertex_prior = WRITE_AFTER_FRAGMENT.replace(
            "at VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT",
            "at VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT",
        );
        assert!(!passes(&[BUFFER], every, WAR, &vertex_prior));
    }

    #[test]
    fn racing_forms_submissions_copies_and_images_fail() {
        let racing = IN_PLACE.replace("READ_AFTER_WRITE", "READ_RACING_WRITE");
        assert!(!passes(
            &[BUFFER],
            every,
            "SYNC-HAZARD-READ-RACING-WRITE",
            &racing
        ));
        let submitted = format!("vkQueueSubmit(): {}", &IN_PLACE["vkCmdDraw(): ".len()..]);
        assert!(!passes(&[BUFFER], every, RAW, &submitted));
        let copy = IN_PLACE.replace("another vkCmdDraw command", "vkCmdCopyBuffer");
        assert!(!passes(&[BUFFER], every, RAW, &copy));
        let image = IN_PLACE.replace("VkBuffer 0x470000000047", "VkImage 0x470000000047");
        assert!(!passes(&[BUFFER], every, RAW, &image));
    }
}
