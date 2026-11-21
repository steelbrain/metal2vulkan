//! Byte-array memory: regions, bounds and poison checks, and pointer resolution.

use super::value::Root;
use super::{IResult, TrapKind};

/// The kind of memory object a region models (trap reporting).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryKind {
    Function,
    Private,
    Input,
    /// A vertex stage's output variables.
    Output,
    Workgroup,
    Buffer,
    PushConstant,
    /// An address that falls in no buffer.
    Physical,
}

/// One memory object.
#[derive(Clone, Debug)]
pub(super) struct Region {
    pub kind: MemoryKind,
    pub bytes: Vec<u8>,
    /// Per-byte masks of the bits written (0xFF: the byte is defined), for poison-tracked
    /// (`Workgroup`) regions. Bits are tracked individually because a sub-word store is two
    /// word atomics, `and` then `or`, each of which defines only some bits.
    pub init: Option<Vec<u8>>,
    pub writable: bool,
    /// The device address of byte 0 (buffers only).
    pub address: Option<u64>,
}

impl Region {
    pub(super) fn new(kind: MemoryKind, bytes: Vec<u8>, writable: bool) -> Region {
        Region {
            kind,
            bytes,
            init: None,
            writable,
            address: None,
        }
    }

    /// A zero-filled region whose bytes count as unwritten until stored.
    pub(super) fn poisoned(kind: MemoryKind, size: usize) -> Region {
        Region {
            kind,
            bytes: vec![0; size],
            init: Some(vec![0; size]),
            writable: true,
            address: None,
        }
    }

    fn range(&self, off: u64, len: usize) -> Result<std::ops::Range<usize>, TrapKind> {
        let size = self.bytes.len() as u64;
        let oob = TrapKind::OutOfBounds {
            memory: self.kind,
            offset: off,
            len: len as u64,
            size,
        };
        let end = off.checked_add(len as u64).ok_or(oob.clone())?;
        if end > size {
            return Err(oob);
        }
        Ok(off as usize..end as usize)
    }

    pub(super) fn read(&self, off: u64, out: &mut [u8]) -> IResult<()> {
        let r = self.range(off, out.len()).map_err(TrapKind::err)?;
        if let Some(init) = &self.init {
            let written = init
                .get(r.clone())
                .is_some_and(|s| s.iter().all(|b| *b == 0xFF));
            if !written {
                return Err(TrapKind::UninitializedRead { offset: off }.err());
            }
        }
        let src = self.bytes.get(r).ok_or_else(|| {
            TrapKind::OutOfBounds {
                memory: self.kind,
                offset: off,
                len: out.len() as u64,
                size: self.bytes.len() as u64,
            }
            .err()
        })?;
        out.copy_from_slice(src);
        Ok(())
    }

    pub(super) fn write(&mut self, off: u64, data: &[u8]) -> IResult<()> {
        if !self.writable {
            return Err(TrapKind::ReadOnlyWrite { memory: self.kind }.err());
        }
        let r = self.range(off, data.len()).map_err(TrapKind::err)?;
        if let Some(init) = &mut self.init {
            if let Some(s) = init.get_mut(r.clone()) {
                s.fill(0xFF);
            }
        }
        if let Some(dst) = self.bytes.get_mut(r) {
            dst.copy_from_slice(data);
        }
        Ok(())
    }

    /// An atomic read-modify-write of the `len`-byte little-endian integer at `off`: `f` maps
    /// `(old bits, old defined-bit mask)` to `(new bits, new defined-bit mask)`, and the old bits
    /// are returned. Undefined bits read as the region's zero fill and are not a trap here: `f`
    /// decides whether its result depends on them.
    pub(super) fn atomic(
        &mut self,
        off: u64,
        len: usize,
        f: impl FnOnce(u64, u64) -> IResult<(u64, u64)>,
    ) -> IResult<u64> {
        if !self.writable {
            return Err(TrapKind::ReadOnlyWrite { memory: self.kind }.err());
        }
        let r = self.range(off, len).map_err(TrapKind::err)?;
        let mut old = [0u8; 8];
        let mut def = [0xFFu8; 8];
        if let (Some(dst), Some(src)) = (old.get_mut(..len), self.bytes.get(r.clone())) {
            dst.copy_from_slice(src);
        }
        if let Some(init) = &self.init {
            if let (Some(dst), Some(src)) = (def.get_mut(..len), init.get(r.clone())) {
                dst.copy_from_slice(src);
            }
        }
        let keep = if len >= 8 {
            u64::MAX
        } else {
            (1u64 << (8 * len)) - 1
        };
        let old = u64::from_le_bytes(old);
        let (new, defined) = f(old, u64::from_le_bytes(def) & keep)?;
        let (new, defined) = (new.to_le_bytes(), defined.to_le_bytes());
        if let (Some(dst), Some(src)) = (self.bytes.get_mut(r.clone()), new.get(..len)) {
            dst.copy_from_slice(src);
        }
        if let Some(init) = &mut self.init {
            if let (Some(dst), Some(src)) = (init.get_mut(r), defined.get(..len)) {
                dst.copy_from_slice(src);
            }
        }
        Ok(old)
    }
}

/// Memory shared by the invocations of one workgroup (and the whole dispatch).
#[derive(Debug)]
pub(super) struct Shared {
    /// This workgroup's `Workgroup` variables.
    pub wg: Vec<Region>,
    pub bufs: Vec<Region>,
    pub push: Region,
    /// The bound images (`super::image`).
    pub images: Vec<super::Image>,
}

/// The memory one invocation can reach: its own allocations and the shared regions.
pub(super) struct Mem<'a> {
    pub lane: &'a mut Vec<Region>,
    pub sh: &'a mut Shared,
}

impl Mem<'_> {
    /// The region `root` names and the offset of `off` inside it.
    fn region(&mut self, root: Root, off: u64, len: usize) -> IResult<(&mut Region, u64)> {
        let missing = || super::malformed(format!("pointer into a missing region {root:?}"));
        match root {
            Root::Lane(i) => Ok((self.lane.get_mut(i as usize).ok_or_else(missing)?, off)),
            Root::Workgroup(i) => Ok((self.sh.wg.get_mut(i as usize).ok_or_else(missing)?, off)),
            Root::Buffer(i) => Ok((self.sh.bufs.get_mut(i as usize).ok_or_else(missing)?, off)),
            Root::Push => Ok((&mut self.sh.push, off)),
            Root::Image(_) => Err(super::malformed("a byte access to an image")),
            Root::Texel(_) => Err(super::malformed("a byte region of a texel")),
            Root::BufferArray(_) => Err(super::malformed("a byte access to a descriptor array")),
            Root::Physical => {
                let hit = self.sh.bufs.iter_mut().find_map(|b| {
                    let a = b.address?;
                    let rel = off.checked_sub(a)?;
                    (rel < b.bytes.len() as u64).then_some((b, rel))
                });
                hit.ok_or_else(|| {
                    TrapKind::OutOfBounds {
                        memory: MemoryKind::Physical,
                        offset: off,
                        len: len as u64,
                        size: 0,
                    }
                    .err()
                })
            }
        }
    }

    /// The first component of texel `i` of bound image `k` ([`Root::Texel`]), for an access of
    /// `len` bytes, which must be the component's 4.
    fn texel(&mut self, k: u32, i: u64, len: usize) -> IResult<&mut u32> {
        if len != 4 {
            return Err(super::malformed(
                "a texel access that is not one 32-bit component",
            ));
        }
        let image = self.sh.images.get_mut(k as usize);
        let texel = image.and_then(|img| img.texels.get_mut(usize::try_from(i).ok()?));
        let [first, ..] =
            texel.ok_or_else(|| super::malformed(format!("texel {i} of image {k}")))?;
        Ok(first)
    }

    pub(super) fn read(&mut self, root: Root, off: u64, out: &mut [u8]) -> IResult<()> {
        if let Root::Texel(k) = root {
            let t = *self.texel(k, off, out.len())?;
            out.copy_from_slice(&t.to_le_bytes());
            return Ok(());
        }
        let (r, rel) = self.region(root, off, out.len())?;
        r.read(rel, out)
    }

    pub(super) fn write(&mut self, root: Root, off: u64, data: &[u8]) -> IResult<()> {
        if let Root::Texel(k) = root {
            let t = self.texel(k, off, data.len())?;
            let mut bytes = [0u8; 4];
            bytes.copy_from_slice(data);
            *t = u32::from_le_bytes(bytes);
            return Ok(());
        }
        let (r, rel) = self.region(root, off, data.len())?;
        r.write(rel, data)
    }

    /// An atomic read-modify-write of `len` bytes (see [`Region::atomic`]).
    pub(super) fn atomic(
        &mut self,
        root: Root,
        off: u64,
        len: usize,
        f: impl FnOnce(u64, u64) -> IResult<(u64, u64)>,
    ) -> IResult<u64> {
        // A texel's component is always defined.
        if let Root::Texel(k) = root {
            let t = self.texel(k, off, len)?;
            let old = u64::from(*t);
            let (new, _) = f(old, u64::from(u32::MAX))?;
            *t = new as u32;
            return Ok(old);
        }
        let (r, rel) = self.region(root, off, len)?;
        r.atomic(rel, len, f)
    }

    /// The byte size of the memory object behind `root`, and `off`'s offset inside it.
    pub(super) fn extent(&mut self, root: Root, off: u64) -> IResult<(u64, u64)> {
        let (r, rel) = self.region(root, off, 0)?;
        Ok((r.bytes.len() as u64, rel))
    }
}
