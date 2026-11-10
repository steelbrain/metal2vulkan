//! Windows file identity for scratch-directory ownership checks.
//!
//! A name can be replaced after its lock was opened. Compare the volume and
//! full file identifier of both open handles; path spelling or timestamps do
//! not establish that the lock still belongs to the named file.

use std::ffi::c_void;
use std::fs::File;
use std::io;
use std::mem::MaybeUninit;
use std::os::windows::io::AsRawHandle;
use std::path::Path;

/// The Win32 FILE_ID_INFO layout, including ReFS's full 128-bit file identifier.
#[repr(C)]
#[derive(PartialEq, Eq)]
struct Identity {
    volume: u64,
    file: [u8; 16],
}

#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetFileInformationByHandleEx(
        file: *mut c_void,
        class: i32,
        information: *mut c_void,
        size: u32,
    ) -> i32;
}

/// FileIdInfo is class 18; the C layout is an eight-byte volume and 16-byte ID.
fn identity(file: &File) -> io::Result<Identity> {
    let mut information = MaybeUninit::<Identity>::uninit();
    // SAFETY: the file retains its live handle for the call and information is
    // writable for the complete FILE_ID_INFO structure requested by class 18.
    let result = unsafe {
        GetFileInformationByHandleEx(
            file.as_raw_handle(),
            18,
            information.as_mut_ptr().cast(),
            24,
        )
    };
    if result == 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: a successful query initialized the whole requested structure.
    Ok(unsafe { information.assume_init() })
}

/// Whether the path still names the file whose lock the caller holds.
/// A removed name is a lost ownership race; other query failures are reported.
pub(super) fn same_file(held: &File, path: &Path) -> io::Result<bool> {
    let named = match File::open(path) {
        Ok(named) => named,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error),
    };
    Ok(identity(held)? == identity(&named)?)
}

#[cfg(test)]
mod tests {
    use super::same_file;
    use crate::scratch::Owned;
    use std::fs::{self, File};

    /// Reusing a pathname must not let a lock on the old file claim its replacement.
    #[test]
    fn replacement_at_the_same_name_is_not_the_held_file() {
        let root = Owned::made("windows-file-replacement").expect("owned test directory");
        let path = root.path().join("named");
        let moved = root.path().join("moved");
        let held = File::create(&path).expect("held file");
        assert!(same_file(&held, &path).expect("original identity"));
        fs::rename(&path, &moved).expect("rename while the handle is retained");
        assert!(!same_file(&held, &path).expect("removed name"));
        fs::write(&path, b"replacement").expect("new file at the same name");
        assert!(!same_file(&held, &path).expect("replacement identity"));
        assert!(same_file(&held, &moved).expect("renamed original identity"));
    }

    /// Different hard-link names identify the same held file without comparing paths.
    #[test]
    fn hard_link_has_the_same_identity() {
        let root = Owned::made("windows-file-hard-link").expect("owned test directory");
        let path = root.path().join("named");
        let alias = root.path().join("alias");
        let held = File::create(&path).expect("held file");
        fs::hard_link(&path, &alias).expect("second name for the same file");
        assert!(same_file(&held, &alias).expect("hard-link identity"));
    }
}
