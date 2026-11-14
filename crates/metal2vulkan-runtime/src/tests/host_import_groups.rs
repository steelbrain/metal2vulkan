//! How the residency groups the releasable imports it may release (`host_residency.rs`, "A shared
//! file counts once"), without a device: imports that hold one shared file's import go together,
//! joined through every file each holds, and each other alone; a group lets go of each import of
//! its own and each of its files once, and of nothing where another holder keeps one of its files.

use crate::host_residency::{groups, listed, FileHolds, Holds};

/// An import's holds: an import of its own where `own`, and `(file, pieces, all)` of each file.
fn holds(own: bool, files: &[(usize, usize, usize)]) -> Holds {
    Holds {
        own,
        files: (files.iter())
            .map(|&(file, pieces, all)| FileHolds { file, pieces, all })
            .collect(),
    }
}

/// Imports of their own are a group each, letting go of one import each: as before shared files.
#[test]
fn an_import_of_its_own_goes_alone() {
    let own = [holds(true, &[]), holds(true, &[]), holds(true, &[])];
    assert_eq!(
        groups(&own),
        vec![(vec![0], Some(1)), (vec![1], Some(1)), (vec![2], Some(1))]
    );
    assert_eq!(own[0].weight(), 1);
}

/// Four imports over pieces of one file, which they alone hold (one piece each, the second two):
/// one group, letting go of the one import; each counts the file for its weight.
#[test]
fn pieces_of_one_file_go_together_as_one_import() {
    let pieces = [
        holds(false, &[(7, 1, 5)]),
        holds(false, &[(7, 2, 5)]),
        holds(false, &[(7, 1, 5)]),
        holds(false, &[(7, 1, 5)]),
    ];
    assert_eq!(groups(&pieces), vec![(vec![0, 1, 2, 3], Some(1))]);
    assert_eq!(pieces[1].weight(), 1);
    // Fewer of them than hold the file: releasing them lets go of nothing.
    assert_eq!(listed(&pieces[..3]), None);
}

/// A file another holder keeps (a heap, a buffer kept for good, a command buffer's import being
/// made): its pieces' group lets go of nothing; and two counts of one file's holds read apart
/// (a hold taken between the reads) are no whole group either.
#[test]
fn a_file_held_elsewhere_keeps_its_pieces() {
    let held = [holds(false, &[(7, 1, 3)]), holds(false, &[(7, 1, 3)])];
    assert_eq!(groups(&held), vec![(vec![0, 1], None)]);
    let apart = [holds(false, &[(7, 1, 2)]), holds(false, &[(7, 1, 3)])];
    assert_eq!(listed(&apart), None);
}

/// Imports joined through files they share: an import over pieces of files 7 and 9 joins 7's
/// group to 9's, the group letting go of both files and its own import once each; a file of its
/// own beside them is a group apart, in the order of its first member.
#[test]
fn files_join_through_an_import_over_both() {
    let joined = [
        holds(false, &[(7, 1, 2)]),
        holds(true, &[(11, 1, 1)]),
        holds(false, &[(9, 1, 2)]),
        holds(true, &[(7, 1, 2), (9, 1, 2)]),
    ];
    assert_eq!(
        groups(&joined),
        vec![(vec![0, 2, 3], Some(3)), (vec![1], Some(2))]
    );
    assert_eq!(joined[3].weight(), 3);
}
