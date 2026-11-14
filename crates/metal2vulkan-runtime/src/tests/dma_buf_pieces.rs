//! How the runtime takes a dma-buf source's chunks (`dma_buf.rs`'s `Piece`) and keeps one import
//! of each shared file (`DmaBufFiles`), with no device: a chunk is a dma-buf of its own only at
//! offset 0 with the runtime its `Arc`'s only holder; a file is known by its `Arc` alone, never by
//! its descriptor's file; the map hands out holds of a live import, takes a second import of the
//! same file made meanwhile back to its maker, and gives an import back to free only with its last
//! hold, removing its entry then.

use crate::dma_buf::{DmaBufFile, DmaBufFiles, Piece};
use crate::DmaBufChunk;
use std::os::fd::{AsRawFd, OwnedFd};
use std::sync::Arc;

/// A descriptor of `/dev/null`, standing in for a dma-buf: nothing here reads it.
fn descriptor() -> OwnedFd {
    OwnedFd::from(std::fs::File::open("/dev/null").unwrap())
}

/// A chunk at offset 0 whose `Arc` the runtime alone holds is a dma-buf of its own; one whose
/// `Arc` the source still holds, or at an offset, is a piece of a shared file, the same `Arc`.
/// Again, one of its own is a new descriptor, a shared one the same file.
#[test]
fn a_chunk_is_its_own_dma_buf_only_alone_at_offset_0() {
    let length = 1 << 16;
    let own = Piece::of(DmaBufChunk {
        file: Arc::new(descriptor()),
        offset: 0,
        length,
    });
    assert!(own.own(), "{own:?}");
    let shared = Arc::new(descriptor());
    let piece = Piece::of(DmaBufChunk {
        file: Arc::clone(&shared),
        offset: 0,
        length,
    });
    let Piece::Shared {
        file, offset: 0, ..
    } = &piece
    else {
        panic!("a chunk of a file the source holds: {piece:?}");
    };
    assert!(Arc::ptr_eq(file, &shared));
    let at = Piece::of(DmaBufChunk {
        file: Arc::new(descriptor()),
        offset: 1 << 16,
        length,
    });
    assert!(
        matches!(at, Piece::Shared { offset, .. } if offset == 1 << 16),
        "a chunk at an offset: {at:?}"
    );
    let (Piece::Own { fd, .. }, Ok(Piece::Own { fd: again, .. })) = (&own, own.again()) else {
        panic!("a dma-buf of its own again");
    };
    assert_ne!(fd.as_raw_fd(), again.as_raw_fd(), "a new descriptor");
    let Ok(Piece::Shared { file: again, .. }) = piece.again() else {
        panic!("a shared piece again");
    };
    assert!(Arc::ptr_eq(&again, &shared), "the same file, no descriptor");
}

/// The map's holds of one file's import: the entered import is the live one; a second import of
/// the same file made meanwhile goes back to its maker with a hold of the live one; a duplicate
/// descriptor in another `Arc` is another file; letting go of every hold but the last frees
/// nothing, and the last gives the import back, its entry gone, so the file is imported anew.
#[test]
fn a_shared_file_is_one_import_until_its_last_hold_goes() {
    let files = DmaBufFiles::default();
    let file = Arc::new(descriptor());
    let first = files
        .enter(DmaBufFile::stand_in(Arc::clone(&file)))
        .unwrap();
    let joined = files.live(&file).expect("the entered import is live");
    assert!(Arc::ptr_eq(&first, &joined));
    let raced = files.enter(DmaBufFile::stand_in(Arc::clone(&file)));
    let Err((live, _made)) = raced else {
        panic!("a second import of a live file was entered");
    };
    assert!(Arc::ptr_eq(&live, &first));
    let other = Arc::new(file.try_clone().unwrap());
    assert!(
        files.live(&other).is_none(),
        "a duplicate descriptor in another Arc is another file"
    );
    assert!(files.let_go([first, joined].into_iter()).is_empty());
    assert!(files.live(&file).is_some(), "one hold left");
    let last = files.let_go(std::iter::once(live));
    assert_eq!(last.len(), 1, "the last hold gives the import back");
    assert!(files.live(&file).is_none(), "its entry gone");
    let again = files.enter(DmaBufFile::stand_in(Arc::clone(&file)));
    assert!(again.is_ok(), "the file imported anew");
}
