//! A submission's parts and the Direct3D 11 planes' hand-over, without a device: where a group
//! goes member by member, each part whose segments or uploads use a plane acquires those planes
//! at its head and releases them at its end, the release taking that part's last signals, its
//! progress value among them, so it adds no submit info; a part that uses none takes neither,
//! and a group sent as one part acquires once ahead of everything and releases once after
//! everything.

use crate::submit::{infos, parts, Staged};
use ash::vk::{self, Handle};

/// The progress semaphore's raw handle.
const PROGRESS: u64 = 50;

fn staged(raw: u64, signals: &[u64]) -> Staged {
    Staged {
        waits: Vec::new(),
        command: Some(
            vk::CommandBufferSubmitInfo::default().command_buffer(vk::CommandBuffer::from_raw(raw)),
        ),
        signals: (signals.iter())
            .map(|&s| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(vk::Semaphore::from_raw(s))
                    .value(1)
            })
            .collect(),
    }
}

/// An info as its command buffers and its signals, each signal as its semaphore and value.
type Info = (Vec<u64>, Vec<(u64, u64)>);

/// The parts of `segments`, the first `ahead` of them uploads, at the last value 10, each as its
/// value and its infos; with each hand-over asked for, as its planes and whether it acquires.
/// Each hand-over's command buffer is numbered from 100 in the order asked.
fn shape(
    segments: Vec<Staged>,
    ahead: usize,
    leads: &[usize],
    planes: &[Vec<char>],
) -> (Vec<(u64, Vec<Info>)>, Vec<(Vec<char>, bool)>) {
    let mut asked = Vec::new();
    let made = parts(
        segments,
        ahead,
        leads,
        (vk::Semaphore::from_raw(PROGRESS), 10),
        planes,
        |planes, acquire| {
            asked.push((planes.to_vec(), acquire));
            Some(vk::CommandBuffer::from_raw(99 + asked.len() as u64))
        },
    );
    let shaped = (made.iter())
        .map(|(value, part)| {
            let infos = (infos(part).iter())
                .map(|info| {
                    let commands = (info.commands.iter())
                        .map(|command| command.command_buffer.as_raw())
                        .collect();
                    let signals = (info.signals.iter())
                        .map(|signal| (signal.semaphore.as_raw(), signal.value))
                        .collect();
                    (commands, signals)
                })
                .collect();
            (*value, infos)
        })
        .collect();
    (shaped, asked)
}

/// An upload (90), a segment of a command buffer begun in an earlier group (1), and two command
/// buffers that lead: the first's two segments (2, and 3 signalling event 7), the second's one
/// (4).
fn group() -> Vec<Staged> {
    vec![
        staged(90, &[]),
        staged(1, &[]),
        staged(2, &[]),
        staged(3, &[7]),
        staged(4, &[]),
    ]
}

/// Where the two leads begin among the segments past the upload.
const LEADS: [usize; 2] = [1, 3];

#[test]
fn each_part_of_a_group_sent_apart_hands_its_own_planes_over() {
    // The first lead's part uses plane a; the second's a and b.
    let planes = [vec![], vec!['a'], vec!['a', 'b']];
    let (parts, asked) = shape(group(), 1, &LEADS, &planes);
    assert_eq!(
        parts,
        [
            (8, vec![(vec![90, 1], vec![(PROGRESS, 8)])]),
            (9, vec![(vec![100, 2, 3, 101], vec![(7, 1), (PROGRESS, 9)])]),
            (10, vec![(vec![102, 4, 103], vec![(PROGRESS, 10)])]),
        ],
        "each plane-using part acquires at its head and releases at its end, the release taking \
         the part's signals and its progress value"
    );
    assert_eq!(
        asked,
        [
            (vec!['a'], true),
            (vec!['a'], false),
            (vec!['a', 'b'], true),
            (vec!['a', 'b'], false),
        ]
    );
}

#[test]
fn a_plane_used_only_ahead_of_a_group_s_last_member_is_released_in_its_own_part() {
    // Only the first lead's part uses a plane: the last part, which completes last, hands none
    // over, and the first lead's release is ahead of its own progress value.
    let planes = [vec![], vec!['a'], vec![]];
    let (parts, asked) = shape(group(), 1, &LEADS, &planes);
    assert_eq!(
        parts,
        [
            (8, vec![(vec![90, 1], vec![(PROGRESS, 8)])]),
            (9, vec![(vec![100, 2, 3, 101], vec![(7, 1), (PROGRESS, 9)])]),
            (10, vec![(vec![4], vec![(PROGRESS, 10)])]),
        ]
    );
    assert_eq!(asked, [(vec!['a'], true), (vec!['a'], false)]);
    // The uploads' planes go with the part ahead of the first lead's.
    let planes = [vec!['u'], vec![], vec![]];
    let (parts, _) = shape(group(), 1, &LEADS, &planes);
    assert_eq!(
        parts,
        [
            (8, vec![(vec![100, 90, 1, 101], vec![(PROGRESS, 8)])]),
            (9, vec![(vec![2, 3], vec![(7, 1), (PROGRESS, 9)])]),
            (10, vec![(vec![4], vec![(PROGRESS, 10)])]),
        ]
    );
}

/// The negative control: a group sent as one part hands its planes over once, ahead of its
/// uploads and after its last segment, the release taking the last signals; and parts that use
/// no plane are as they are without the hand-over.
#[test]
fn a_group_sent_as_one_part_hands_its_planes_over_once() {
    let (parts, asked) = shape(group(), 1, &[], &[vec!['a', 'b']]);
    assert_eq!(
        parts,
        [(
            10,
            vec![
                (vec![100, 90, 1, 2, 3], vec![(7, 1)]),
                (vec![4, 101], vec![(PROGRESS, 10)]),
            ]
        )]
    );
    assert_eq!(asked, [(vec!['a', 'b'], true), (vec!['a', 'b'], false)]);
    let (parts, asked) = shape(group(), 1, &LEADS, &[]);
    assert_eq!(
        parts,
        [
            (8, vec![(vec![90, 1], vec![(PROGRESS, 8)])]),
            (9, vec![(vec![2, 3], vec![(7, 1), (PROGRESS, 9)])]),
            (10, vec![(vec![4], vec![(PROGRESS, 10)])]),
        ]
    );
    assert!(asked.is_empty());
}
