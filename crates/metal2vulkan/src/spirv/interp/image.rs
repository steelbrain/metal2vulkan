//! Images: bound images, texel addressing, and the image instructions the new core emits for
//! textures (`OpImageFetch`, `OpImageRead`, `OpImageWrite`, `OpImageQuery*`). A combined image
//! sampler binds its image, and `OpImage` of it is that image (a pixel sampler's exact fetch);
//! there is no sampler state, so sampling itself is not modelled. `OpImageTexelPointer` points to
//! a texel inside the image (`Root::Texel`): an atomic through it reads and writes the texel's
//! first component, as an atomic of an `R32ui` / `R32i` image does.
//!
//! **Sparse residency**. An image may leave texels unmapped ([`Image::unmapped`]), as a sparse
//! image with unbound memory does. `OpImageSparseFetch` and `OpImageSparseRead` return the struct
//! of a residency code and the texel: the code is 0 where the texel read is mapped and 1 where it
//! is not, and `OpImageSparseTexelsResident` is `code == 0`. A sparse or plain access reads an
//! unmapped texel as [`Image::unmapped_texel`]: zero by default, as a device with
//! `residencyNonResidentStrict` reads it, or another value a test chooses to model a device without
//! it, whose value Vulkan leaves undefined. A write of an unmapped texel is discarded. A robust
//! access outside the image reads the robust texel with code 1: Vulkan does not state its
//! residency, and lavapipe reports it not resident (measured), so a module that answers Metal's
//! word there (resident) must not take the device's.
//!
//! `OpImageFetch`'s `ConstOffset` is added to the coordinate's plane components before the texel
//! is addressed, in 32-bit wrapping arithmetic (so a component the offset takes below zero is
//! out of range, as any negative coordinate is); a layer is not offset (SPIR-V's image operands).
//!
//! An image stores each texel as the four 32-bit components of its sampled type, so a format's
//! conversion is outside the model: a write stores exactly the `vec4` the shader writes, and a
//! read returns it. Every component of a coordinate, the level, and the sample index is checked
//! against the image (as an unsigned value, so a negative `int` coordinate is out of range): an
//! access outside the image is a [`TrapKind::ImageOutOfBounds`], never a neighbouring texel,
//! unless the image models the device's robustness ([`Image::robust`]). Such an image takes the
//! permission Vulkan's robustness gives a load outside it: it returns the texel the same
//! invocation last stored to that location, unless the load is volatile: the image's variable is
//! decorated `Volatile`, or the read carries `VolatileTexel`.

use std::collections::BTreeSet;

use crate::pipeline::sym::ResourceSlot;
use crate::spirv::ids::Value;
use crate::spirv::ir::{Dim, ImageOperands, ImageQuery, ImageType, Inst, Type};

use super::memory::Shared;
use super::sched::{Lane, Prog};
use super::value::{Ptr, Root, Val};
use super::{malformed, IResult, TrapKind, Unmodelled};

/// One image bound to a `UniformConstant` image binding (sampled, storage, or texel buffer), or
/// to one element of a descriptor array of images (a storage slot's level views, or a runtime heap
/// array).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    pub slot: ResourceSlot,
    /// The descriptor array element the image is bound to (0 for a plain binding). An array of
    /// `n` images binds `n` images of its slot, elements `0 .. n` in order; a runtime array is as
    /// long as the consecutive elements bound from 0.
    pub element: u32,
    /// Level-0 width, height, and depth (1 for the dimensions the image does not have; a texel
    /// buffer's element count is its width).
    pub extent: [u32; 3],
    /// Array layers (6 per cube of a cube or cube array image).
    pub layers: u32,
    pub levels: u32,
    pub samples: u32,
    /// Texels by level, then layer, sample, z, y, x.
    pub texels: Vec<[u32; 4]>,
    /// The device's robustness for the image (`robustImageAccess2`, or `robustBufferAccess2` for
    /// a texel buffer): a read outside the image returns this texel (zero, with `(0, 0, 1)`
    /// substituted for the view format's missing G, B, and A), and a write outside it is discarded.
    /// A read of a location the invocation wrote outside the image returns that write's texel
    /// instead, as Vulkan permits ("If a value was previously written to the same out of bounds
    /// location in the same shader invocation, that value may be returned instead"), unless the
    /// load is volatile: the image's variable is `Volatile`, or the read carries `VolatileTexel`.
    /// `None` (the default) traps every access outside the image.
    pub robust: Option<[u32; 4]>,
    /// The indices in [`Image::texels`] of the texels a sparse image leaves unmapped: a read of one
    /// returns [`Image::unmapped_texel`], and a sparse access reports it not resident. Empty (the
    /// default) for an image that is resident everywhere.
    pub unmapped: BTreeSet<usize>,
    /// What a read of an unmapped texel returns: zero (the default), as under Vulkan's
    /// `residencyNonResidentStrict`. A device without it returns an undefined value, which a test
    /// models by setting another one.
    pub unmapped_texel: [u32; 4],
}

impl Image {
    /// A zero-filled image.
    pub fn new(
        slot: ResourceSlot,
        extent: [u32; 3],
        layers: u32,
        levels: u32,
        samples: u32,
    ) -> Image {
        let mut img = Image {
            slot,
            element: 0,
            extent,
            layers,
            levels,
            samples,
            texels: Vec::new(),
            robust: None,
            unmapped: BTreeSet::new(),
            unmapped_texel: [0; 4],
        };
        let n: usize = (0..levels).map(|l| img.level_texels(l)).sum();
        img.texels = vec![[0; 4]; n];
        img
    }

    /// This image, bound to element `element` of its slot's descriptor array.
    pub fn at_element(self, element: u32) -> Image {
        Image { element, ..self }
    }

    /// The extent of level `level`: each dimension halved per level, at least 1.
    pub fn level_extent(&self, level: u32) -> [u32; 3] {
        self.extent
            .map(|e| e.checked_shr(level).unwrap_or(0).max(1))
    }

    fn level_texels(&self, level: u32) -> usize {
        let [w, h, d] = self.level_extent(level);
        [w, h, d, self.layers, self.samples]
            .iter()
            .map(|x| *x as usize)
            .product()
    }

    /// The index in [`Image::texels`] of texel `(x, y, z)` of `layer` and `sample` at `level`, if
    /// every component is inside the image.
    pub fn index(&self, level: u32, layer: u32, sample: u32, xyz: [u32; 3]) -> Option<usize> {
        if level >= self.levels || layer >= self.layers || sample >= self.samples {
            return None;
        }
        let [w, h, d] = self.level_extent(level);
        let [x, y, z] = xyz;
        if x >= w || y >= h || z >= d {
            return None;
        }
        let base: usize = (0..level).map(|l| self.level_texels(l)).sum();
        let within = (((layer as usize * self.samples as usize + sample as usize) * d as usize
            + z as usize)
            * h as usize
            + y as usize)
            * w as usize
            + x as usize;
        Some(base + within)
    }
}

/// The bound image of image value `v` and its type.
fn image_of(p: &Prog<'_>, l: &Lane, v: Value) -> IResult<(u32, ImageType)> {
    let k = match l.get(p, v)? {
        Val::P(q) => match q.root {
            Root::Image(k) => k,
            _ => return Err(malformed("an image operand that is no image")),
        },
        _ => return Err(malformed("an image operand that is no image")),
    };
    match p.t.get(p.t.value_ty(v)?)? {
        Type::Image(img) => Ok((k, *img)),
        _ => Err(malformed("an image operand without an image type")),
    }
}

fn bound(sh: &mut Shared, k: u32) -> IResult<&mut Image> {
    sh.images
        .get_mut(k as usize)
        .ok_or_else(|| malformed(format!("image {k} is not bound")))
}

/// The number of plane coordinates of `dim`.
fn plane(dim: Dim) -> usize {
    match dim {
        Dim::D1 | Dim::Buffer => 1,
        Dim::D2 | Dim::SubpassData => 2,
        Dim::D3 | Dim::Cube => 3,
    }
}

/// The texel index of coordinate `c` with `level` and `sample` in image `k` of type `img`: `Err`
/// with the image's robust texel for an access outside an image that models robustness
/// ([`Image::robust`]), and a trap for one outside any other.
fn texel(
    sh: &mut Shared,
    k: u32,
    img: &ImageType,
    c: &[u64],
    level: u64,
    sample: u64,
) -> IResult<Result<usize, [u32; 4]>> {
    if img.dim == Dim::Cube || img.dim == Dim::SubpassData {
        return Err(Unmodelled::Opcode(crate::spv::Op::ImageRead).err());
    }
    let n = plane(img.dim);
    let want = n + usize::from(img.arrayed);
    if c.len() != want {
        return Err(malformed("an image coordinate of the wrong length"));
    }
    let u = |x: u64| u32::try_from(x).ok();
    let mut xyz = Some([0u32; 3]);
    for (i, x) in c.iter().take(n).enumerate() {
        match (xyz.as_mut().and_then(|a| a.get_mut(i)), u(*x)) {
            (Some(slot), Some(x)) => *slot = x,
            _ => xyz = None,
        }
    }
    let layer = if img.arrayed {
        u(c.get(n).copied().unwrap_or(0))
    } else {
        Some(0)
    };
    let image = bound(sh, k)?;
    let at = match (u(level), layer, u(sample), xyz) {
        (Some(level), Some(layer), Some(sample), Some(xyz)) => {
            image.index(level, layer, sample, xyz)
        }
        _ => None,
    };
    match (at, image.robust) {
        (Some(i), _) => Ok(Ok(i)),
        (None, Some(t)) => Ok(Err(t)),
        (None, None) => Err(TrapKind::ImageOutOfBounds { image: k }.err()),
    }
}

/// The integer operands an image access may carry: `Lod` and `Sample` only (a fetch's
/// `ConstOffset` is [`offset_coord`]'s).
fn level_and_sample(p: &Prog<'_>, l: &Lane, ops: &ImageOperands) -> IResult<(u64, u64)> {
    let unsupported =
        ops.bias.is_some() || ops.grad.is_some() || ops.offset.is_some() || ops.min_lod.is_some();
    if unsupported {
        return Err(Unmodelled::Opcode(crate::spv::Op::ImageFetch).err());
    }
    let level = match ops.lod {
        Some(v) => l.get(p, v)?.s()?,
        None => 0,
    };
    let sample = match ops.sample {
        Some(v) => l.get(p, v)?.s()?,
        None => 0,
    };
    Ok((level, sample))
}

/// Add a fetch's `ConstOffset` `o` to the first `n` (plane) components of coordinate `c`, each a
/// 32-bit integer, wrapping.
fn offset_coord(c: &mut [u64], o: &[u64], n: usize) -> IResult<()> {
    if o.len() != n || c.len() < n {
        return Err(malformed("an image offset of the wrong length"));
    }
    for (x, d) in c.iter_mut().zip(o) {
        *x = u64::from((*x as u32).wrapping_add(*d as u32));
    }
    Ok(())
}

/// Execute an image instruction; `None` for an instruction that is not one of them.
pub(super) fn exec(
    p: &Prog<'_>,
    l: &mut Lane,
    sh: &mut Shared,
    inst: &Inst,
) -> IResult<Option<Option<Val>>> {
    Ok(Some(match inst {
        Inst::ImageFetch {
            image,
            coord,
            ops,
            sparse,
        }
        | Inst::ImageRead {
            image,
            coord,
            ops,
            sparse,
            ..
        } => {
            let (k, img) = image_of(p, l, *image)?;
            // A volatile load: the image's `Volatile` decoration, or the read's `VolatileTexel`
            // (its Vulkan memory model form).
            let volatile =
                p.volatile(k) || matches!(inst, Inst::ImageRead { mem, .. } if mem.volatile());
            let mut c = l.get(p, *coord)?.comps()?;
            match (inst, ops.const_offset) {
                (_, None) => {}
                (Inst::ImageFetch { .. }, Some(o)) => {
                    offset_coord(&mut c, &l.get(p, o)?.comps()?, plane(img.dim))?;
                }
                (_, Some(_)) => return Err(Unmodelled::Opcode(crate::spv::Op::ImageRead).err()),
            }
            let (level, sample) = level_and_sample(p, l, ops)?;
            let (t, resident) = match texel(sh, k, &img, &c, level, sample)? {
                Ok(i) => {
                    let image = bound(sh, k)?;
                    if image.unmapped.contains(&i) {
                        (image.unmapped_texel, false)
                    } else {
                        let t = image
                            .texels
                            .get(i)
                            .copied()
                            .ok_or_else(|| malformed("texel index"))?;
                        (t, true)
                    }
                }
                // Robustness's texel, or the invocation's own discarded store to this location
                // where the load is not volatile; either is not resident.
                Err(robust) => match l.oob_stores.get(&(k, c, level, sample)) {
                    Some(stored) if !volatile => (*stored, false),
                    _ => (robust, false),
                },
            };
            let t = Val::C(t.iter().map(|x| Val::S(u64::from(*x))).collect());
            Some(if *sparse {
                Val::C(Box::new([Val::S(u64::from(!resident)), t]))
            } else {
                t
            })
        }
        Inst::ImageSparseTexelsResident(code) => {
            Some(Val::S(u64::from(l.get(p, *code)?.s()? == 0)))
        }
        Inst::ImageWrite {
            image,
            coord,
            texel: value,
            ops,
            ..
        } => {
            let (k, img) = image_of(p, l, *image)?;
            let c = l.get(p, *coord)?.comps()?;
            if ops.const_offset.is_some() {
                return Err(Unmodelled::Opcode(crate::spv::Op::ImageWrite).err());
            }
            let (level, sample) = level_and_sample(p, l, ops)?;
            let comps = l.get(p, *value)?.comps()?;
            let mut t = [0u32; 4];
            for (d, s) in t.iter_mut().zip(&comps) {
                *d = *s as u32;
            }
            // Robustness discards a write outside the image; the invocation remembers it, since
            // a later load of the location may return it. A write of an unmapped texel is
            // discarded.
            match texel(sh, k, &img, &c, level, sample)? {
                Ok(i) => {
                    let image = bound(sh, k)?;
                    if !image.unmapped.contains(&i) {
                        let slot = image
                            .texels
                            .get_mut(i)
                            .ok_or_else(|| malformed("texel index"))?;
                        *slot = t;
                    }
                }
                Err(_) => {
                    l.oob_stores.insert((k, c, level, sample), t);
                }
            }
            None
        }
        // A pointer to a texel inside the image, an atomic's operand. Robustness's atomic outside
        // the image, and one on an unmapped texel, are not modelled.
        Inst::ImageTexelPointer {
            image,
            coord,
            sample,
        } => {
            let unmodelled = || Unmodelled::Opcode(crate::spv::Op::ImageTexelPointer).err();
            let Val::P(Ptr {
                root: Root::Image(k),
                ..
            }) = l.get(p, *image)?
            else {
                return Err(malformed("a texel pointer into no image"));
            };
            let (_, ty) = p.t.pointee(p.t.value_ty(*image)?)?;
            let Type::Image(img) = p.t.get(ty)? else {
                return Err(malformed("a texel pointer into a pointer to no image"));
            };
            let img = *img;
            let c = l.get(p, *coord)?.comps()?;
            let sample = l.get(p, *sample)?.s()?;
            match texel(sh, k, &img, &c, 0, sample)? {
                Ok(i) if !bound(sh, k)?.unmapped.contains(&i) => Some(Val::P(Ptr {
                    root: Root::Texel(k),
                    off: i as u64,
                    ms: None,
                })),
                _ => return Err(unmodelled()),
            }
        }
        Inst::ImageQuery { query, image } => {
            let (k, img) = image_of(p, l, *image)?;
            let level = match query {
                ImageQuery::SizeLod(v) => l.get(p, *v)?.s()?,
                _ => 0,
            };
            let image = bound(sh, k)?;
            let v = match query {
                ImageQuery::SizeLod(_) | ImageQuery::Size => {
                    let level = u32::try_from(level)
                        .ok()
                        .filter(|lv| *lv < image.levels)
                        .ok_or(TrapKind::ImageOutOfBounds { image: k }.err())?;
                    let e = image.level_extent(level);
                    let dims = match img.dim {
                        Dim::Cube => 2,
                        d => plane(d),
                    };
                    let mut out: Vec<u64> = e.iter().take(dims).map(|x| u64::from(*x)).collect();
                    if img.arrayed {
                        let layers = if img.dim == Dim::Cube {
                            image.layers / 6
                        } else {
                            image.layers
                        };
                        out.push(u64::from(layers));
                    }
                    Val::from_comps(out)
                }
                ImageQuery::Levels => Val::S(u64::from(image.levels)),
                ImageQuery::Samples => Val::S(u64::from(image.samples)),
            };
            Some(v)
        }
        _ => return Ok(None),
    }))
}

#[cfg(test)]
mod tests {
    use super::Image;
    use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};

    fn slot() -> ResourceSlot {
        ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::Texture,
            index: 0,
        }
    }

    /// Levels halve each dimension down to 1, and the texels of every level are allocated.
    #[test]
    fn levels_halve_to_one() {
        let img = Image::new(slot(), [8, 3, 1], 2, 4, 1);
        assert_eq!(img.level_extent(0), [8, 3, 1]);
        assert_eq!(img.level_extent(1), [4, 1, 1]);
        assert_eq!(img.level_extent(3), [1, 1, 1]);
        assert_eq!(img.texels.len(), 2 * (24 + 4 + 2 + 1));
    }

    /// Every in-range address is distinct and inside the texels; every out-of-range component is
    /// `None`.
    #[test]
    fn addressing_is_dense_and_checked() {
        let img = Image::new(slot(), [3, 2, 2], 2, 2, 2);
        let mut seen = std::collections::BTreeSet::new();
        for level in 0..2 {
            let [w, h, d] = img.level_extent(level);
            for layer in 0..2 {
                for sample in 0..2 {
                    for z in 0..d {
                        for y in 0..h {
                            for x in 0..w {
                                let i = img
                                    .index(level, layer, sample, [x, y, z])
                                    .expect("in range");
                                assert!(i < img.texels.len());
                                assert!(seen.insert(i));
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(seen.len(), img.texels.len());
        assert_eq!(img.index(2, 0, 0, [0, 0, 0]), None);
        assert_eq!(img.index(0, 2, 0, [0, 0, 0]), None);
        assert_eq!(img.index(0, 0, 2, [0, 0, 0]), None);
        assert_eq!(img.index(0, 0, 0, [3, 0, 0]), None);
        assert_eq!(img.index(1, 0, 0, [0, 1, 0]), None);
        assert_eq!(img.index(1, 0, 0, [0, 0, 1]), None);
    }
}
