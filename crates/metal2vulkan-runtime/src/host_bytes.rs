//! The consumer's bytes that a call reads in place: [`HostBytes`].

use std::marker::PhantomData;
use std::ptr::NonNull;

/// The bytes a call reads from the host, in place, during the call alone: Metal's `const void *`
/// with its length, as `replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:`
/// takes it ([`Texture::replace_region`](crate::Texture::replace_region)). Nothing differs from
/// Metal: Metal too reads the bytes during the call and keeps no pointer to them.
///
/// The bytes may change under the runtime while the call runs, as memory that another party
/// writes at any time does (a guest's RAM). The runtime reads each byte at most once, by raw
/// copies into its own memory, forms no slice over them, decides nothing from their values (sizes
/// and layout come from the call's other arguments alone) and keeps no pointer past the call's
/// return. A slice gives them safely (`From<&[u8]>`); memory that no slice may cover gives them
/// through [`HostBytes::from_raw`].
#[derive(Clone, Copy, Debug)]
pub struct HostBytes<'a> {
    start: NonNull<u8>,
    len: usize,
    bytes: PhantomData<&'a [u8]>,
}

impl<'a> From<&'a [u8]> for HostBytes<'a> {
    fn from(bytes: &'a [u8]) -> HostBytes<'a> {
        HostBytes {
            start: NonNull::from(bytes).cast(),
            len: bytes.len(),
            bytes: PhantomData,
        }
    }
}

impl<'a> HostBytes<'a> {
    /// The `len` bytes at `start`, read in place by the call that takes them.
    ///
    /// # Safety
    ///
    /// Where `len` is not 0, `start` is not null and the `len` bytes from it are readable memory
    /// of one allocation for as long as the call that takes them runs (`'a`). Others may write
    /// them meanwhile.
    pub unsafe fn from_raw(start: *const u8, len: usize) -> HostBytes<'a> {
        HostBytes {
            start: NonNull::new(start.cast_mut()).unwrap_or(NonNull::dangling()),
            len,
            bytes: PhantomData,
        }
    }

    /// How many bytes there are.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether there are none.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Copies the `len` bytes at offset `from` to `to`, by a raw copy.
    ///
    /// # Safety
    ///
    /// `from + len` is at most [`HostBytes::len`]; `to` is writable for `len` bytes and overlaps
    /// none of these.
    pub(crate) unsafe fn copy_to(&self, from: usize, to: *mut u8, len: usize) {
        // SAFETY: the caller's, and the slice's or `from_raw`'s: the range lies in memory that is
        // readable while the call runs (or is empty, at an aligned, non-null start).
        unsafe {
            self.start
                .as_ptr()
                .cast_const()
                .add(from)
                .copy_to_nonoverlapping(to, len)
        }
    }
}
