//! A command buffer's note of its commands' uses (`DirtyCommit::note`), which takes each run of
//! uses of one key at once, against the same uses noted one by one (`DirtyCommit::note_each`):
//! the same writes kept, each reported to the same logs, the same keys used, the same overflows,
//! and the same bytes in each log of host pages imported in place. Over commands of runs, keys
//! named in two runs, reads beside writes, writes named twice, keys of no log and keys of two, past
//! the ranges a command buffer keeps of one log's pages (64) and past the writes and keys it keeps
//! apart (4096).

use crate::dirty_pages::{hook, DirtyCommit, DirtyShadows, DirtySource, PagesLog};
use crate::resource::ResourceKey;
use crate::tracker::{Access, Use, UseRange};
use crate::{DirtyPages, DirtyStamp};
use ash::vk;
use std::ops::Range;
use std::sync::Arc;

/// The host address of the first logged page.
const BASE: u64 = 0x7000_0000;
/// The bytes of each log's pages.
const SPAN: u64 = 0x10_0000;

/// A log that keeps nothing: the note alone is compared.
struct Silent;

impl DirtyPages for Silent {
    type Watch = ();

    fn watch(&self, _range: Range<u64>) {}

    fn request(&self) -> DirtyStamp {
        DirtyStamp(0)
    }

    fn wait(&self, _stamp: DirtyStamp) -> bool {
        true
    }

    fn changed(&self, _since: DirtyStamp, _out: &mut Vec<Range<u64>>) {}

    fn wrote(&self, _range: Range<u64>) {}
}

fn read() -> Access {
    Access::new(
        vk::PipelineStageFlags2::COMPUTE_SHADER,
        vk::AccessFlags2::SHADER_STORAGE_READ,
    )
}

fn write() -> Access {
    Access::new(
        vk::PipelineStageFlags2::COMPUTE_SHADER,
        vk::AccessFlags2::SHADER_STORAGE_WRITE,
    )
}

/// Both notes of `commands`, in order, into a command buffer each.
fn both(commands: &[Vec<Use<ResourceKey>>], shadows: &DirtyShadows) -> [DirtyCommit; 2] {
    let mut runs = DirtyCommit::default();
    let mut each = DirtyCommit::default();
    for command in commands {
        runs.note(command, shadows);
        each.note_each(command, shadows);
    }
    [runs, each]
}

#[test]
fn runs_of_one_key_note_exactly_as_their_uses_one_by_one() {
    let source = DirtySource::new(hook(Arc::new(Silent)), BASE);
    let logs: Vec<Arc<PagesLog>> = (0..8)
        .map(|at| PagesLog::new((BASE + at * SPAN, SPAN), Some(&source), None).unwrap())
        .collect();
    // Keys 0 to 7 of one log each, 8 and 9 of two, 10 and 11 of none.
    let keys: Vec<ResourceKey> = (0..12).map(|_| ResourceKey::new()).collect();
    let shadows = DirtyShadows::default();
    for (key, log) in keys.iter().zip(&logs) {
        shadows.logged_in_place(*key, log);
    }
    for (key, two) in [(keys[8], [0, 1]), (keys[9], [2, 3])] {
        for at in two {
            shadows.logged_in_place(key, &logs[at]);
        }
    }
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut next = move |below: u64| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed % below
    };
    let mut overflowed_logs = 0;
    for round in 0..300 {
        let commands: Vec<Vec<Use<ResourceKey>>> = (0..1 + next(4))
            .map(|_| {
                let mut command = Vec::new();
                for _ in 0..1 + next(12) {
                    let key = keys[next(12) as usize];
                    let mut at = next(64) * 64;
                    for _ in 0..1 + next(24) {
                        // Mostly past the one before, as a copy's ranges are; at times back over
                        // it, so a write is named twice.
                        at = match next(8) {
                            0 => at.saturating_sub(128),
                            _ => at + 64 * next(4),
                        };
                        let range = UseRange::Bytes(at..at + 64 * (1 + next(3)));
                        command.push(match next(4) {
                            0 => Use::reads(key, true, range, read()),
                            _ => Use::writes(key, true, range, write()),
                        });
                    }
                }
                command
            })
            .collect();
        let [runs, each] = both(&commands, &shadows);
        let noted = runs.noted();
        assert_eq!(noted, each.noted(), "round {round}: {commands:?}");
        overflowed_logs += (noted.host.iter())
            .filter(|(_, ranges)| ranges.first() == Some(&(0..u64::MAX)))
            .count();
    }
    assert!(
        overflowed_logs > 30,
        "{overflowed_logs} logs' pages written whole past the ranges kept of them"
    );
    // Past the writes and the keys a command buffer keeps apart: 4200 keys of no log written once
    // each, between runs of logged keys.
    let many: Vec<Use<ResourceKey>> = (0..4200)
        .map(|at| {
            Use::writes(
                ResourceKey::new(),
                true,
                UseRange::Bytes(at..at + 1),
                write(),
            )
        })
        .collect();
    let logged = |start: u64| -> Vec<Use<ResourceKey>> {
        (0..20)
            .map(|at| {
                let range = UseRange::Bytes(start + at * 64..start + at * 64 + 32);
                Use::writes(keys[(at / 10) as usize], true, range, write())
            })
            .collect()
    };
    let commands = [logged(0), many, logged(4096)];
    let [runs, each] = both(&commands, &shadows);
    let noted = runs.noted();
    assert!(noted.used_all && noted.overflowed, "past the limits");
    assert_eq!(noted, each.noted(), "past the limits");
}
