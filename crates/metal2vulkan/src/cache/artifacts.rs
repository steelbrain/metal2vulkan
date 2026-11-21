//! `to_bytes`/`from_bytes` for [`CompiledFunction`] and the pipeline artifacts.
//!
//! `to_bytes` frames the `codec` encoding of the whole artifact (`envelope`). `from_bytes` checks
//! the frame, decodes the body (which must use every byte), requires the body's key and profile
//! fingerprint to equal the frame's, and runs the linear sanity scan (`scan`) over every module.
//! It trusts the words by identity and does not re-run the owned verifier.

use crate::device::Device;
use crate::library::{CompiledFunction, FunctionArtifact, FunctionCompileKey};
use crate::pipeline::artifact::{ComputePipeline, RenderPipeline};

use super::codec::{decode, encode, encode_chunked, Codec, Malformed};
use super::digest::Node;
use super::envelope::{open, seal, seal_to, Kind};
use super::scan::{scan_relocatable, scan_words};
use super::{ArtifactError, Hash256, PipelineArtifact, PipelineKey};

/// Write the frame of `value`'s encoding to `out` as [`seal`] of [`encode`] gives it, a chunk at
/// a time, so the encoding is never held whole beside `value`.
fn write_sealed<T: Codec>(
    kind: Kind,
    key: &Hash256,
    profile: &Hash256,
    value: &T,
    out: &mut dyn std::io::Write,
) -> std::io::Result<()> {
    let mut failed = None;
    seal_to(
        kind,
        key,
        profile,
        &|sink| encode_chunked(value, sink),
        &mut |chunk| {
            if failed.is_none() {
                failed = out.write_all(chunk).err();
            }
        },
    );
    failed.map_or(Ok(()), Err)
}

impl From<Malformed> for ArtifactError {
    fn from(_: Malformed) -> ArtifactError {
        ArtifactError::Corrupt
    }
}

/// `Ok` when `ok`, else `Corrupt`.
fn require(ok: bool) -> Result<(), ArtifactError> {
    if ok {
        Ok(())
    } else {
        Err(ArtifactError::Corrupt)
    }
}

impl CompiledFunction {
    /// The versioned, self-checking encoding: the artifact, the fingerprint of the profile it was
    /// compiled for, and its kernel facts.
    pub fn to_bytes(&self) -> Vec<u8> {
        let a = self.artifact();
        seal(Kind::Function, a.key.digest(), &a.profile, &encode(a))
    }

    /// Decode a function compiled for `device`'s profile, or say why the bytes cannot be
    /// trusted; the caller treats every error as a cache miss.
    pub fn from_bytes(bytes: &[u8], device: &Device) -> Result<CompiledFunction, ArtifactError> {
        let frame = open(bytes, Kind::Function, device)?;
        let a: FunctionArtifact = decode(frame.body)?;
        require(a.key.digest() == &frame.key)?;
        require(a.profile == device.profile_fingerprint())?;
        scan_relocatable(&a.module)?;
        Ok(CompiledFunction::new(std::sync::Arc::new(a)))
    }

    /// [`CompiledFunction::from_bytes`] for the bytes a cache stored under `key`: bytes of any
    /// other key are `WrongDevice`, like bytes of another profile.
    pub fn from_bytes_for_key(
        bytes: &[u8],
        key: &FunctionCompileKey,
        device: &Device,
    ) -> Result<CompiledFunction, ArtifactError> {
        let f = CompiledFunction::from_bytes(bytes, device)?;
        if f.key() == key {
            Ok(f)
        } else {
            Err(ArtifactError::WrongDevice)
        }
    }
}

impl ComputePipeline {
    /// The pipeline's content digest: a SHA-256 of what the pipeline says, which names it by its
    /// content where [`ComputePipeline::key`] names it by its inputs and versions.
    ///
    /// **What it covers.** Every field but the cache identity: the stage's module (its SPIR-V
    /// words and everything else [`StageModule`] states), the layout, the runtime dependencies,
    /// the interface, the Metal information, the requirements and the variant bits. It excludes
    /// `key`, which every version bump of the crate moves, and `profile`, the device profile's
    /// fingerprint. So two builds whose pipelines for an input say the same thing give the same
    /// digest, whatever their versions, and a cache of driver objects keyed on it (a built
    /// `VkPipeline`, a pipeline cache's blob) survives a release that changed nothing the
    /// pipeline says.
    ///
    /// **When it changes.** Whenever covered content changes: one SPIR-V word, one binding, one
    /// precision note. A compute and a render pipeline never share one. The one value that holds
    /// it is a field a later release adds to the encoding while it is at its zero value, which by
    /// the encoding's rule means what the pipeline meant before the field existed. Values are
    /// hashed by name (fields and enum variants), never by position, so the digest does not
    /// depend on the encoding's layout.
    ///
    /// **What it is not.** An identity, not an authentication: anyone can compute it, so it
    /// signs nothing, and an artifact from untrusted storage is checked by its decode
    /// (`PipelineArtifact::from_bytes`), not by comparing digests.
    ///
    /// [`StageModule`]: crate::pipeline::StageModule
    pub fn content_digest(&self) -> Hash256 {
        let ComputePipeline {
            key: _,
            profile: _,
            stage,
            layout,
            dependencies,
            interface,
            metal_info,
            requirements,
            variant_bits,
        } = self;
        Node::fields(&mut [
            ("stage", stage.digest()),
            ("layout", layout.digest()),
            ("dependencies", dependencies.digest()),
            ("interface", interface.digest()),
            ("metal_info", metal_info.digest()),
            ("requirements", requirements.digest()),
            ("variant_bits", variant_bits.digest()),
        ])
        .root("compute pipeline")
    }
}

impl RenderPipeline {
    /// The pipeline's content digest: [`ComputePipeline::content_digest`]'s definition over a
    /// render pipeline. It covers every field but `key` and `profile`: each stage's module, the
    /// layout, the vertex input, the fixed-function state, the runtime dependencies, the interface,
    /// the requirements and the variant bits. It is equal across builds whose pipelines for an
    /// input say the same thing, changes whenever covered content changes, and authenticates
    /// nothing, as the compute pipeline's.
    pub fn content_digest(&self) -> Hash256 {
        let RenderPipeline {
            key: _,
            profile: _,
            stages,
            layout,
            vertex_input,
            graphics,
            dependencies,
            interface,
            requirements,
            variant_bits,
        } = self;
        Node::fields(&mut [
            ("stages", stages.digest()),
            ("layout", layout.digest()),
            ("vertex_input", vertex_input.digest()),
            ("graphics", graphics.digest()),
            ("dependencies", dependencies.digest()),
            ("interface", interface.digest()),
            ("requirements", requirements.digest()),
            ("variant_bits", variant_bits.digest()),
        ])
        .root("render pipeline")
    }
}

impl PipelineArtifact for ComputePipeline {
    fn key(&self) -> &PipelineKey {
        &self.key
    }

    fn to_bytes(&self) -> Vec<u8> {
        seal(
            Kind::ComputePipeline,
            self.key.digest(),
            &self.profile,
            &encode(self),
        )
    }

    fn write_to(&self, out: &mut dyn std::io::Write) -> std::io::Result<()> {
        write_sealed(
            Kind::ComputePipeline,
            self.key.digest(),
            &self.profile,
            self,
            out,
        )
    }

    fn from_bytes(bytes: &[u8], device: &Device) -> Result<ComputePipeline, ArtifactError> {
        let frame = open(bytes, Kind::ComputePipeline, device)?;
        let p: ComputePipeline = decode(frame.body)?;
        require(p.key.digest() == &frame.key)?;
        require(p.profile == device.profile_fingerprint())?;
        scan_words(&p.stage.spirv)?;
        Ok(p)
    }
}

impl PipelineArtifact for RenderPipeline {
    fn key(&self) -> &PipelineKey {
        &self.key
    }

    fn to_bytes(&self) -> Vec<u8> {
        seal(
            Kind::RenderPipeline,
            self.key.digest(),
            &self.profile,
            &encode(self),
        )
    }

    fn write_to(&self, out: &mut dyn std::io::Write) -> std::io::Result<()> {
        write_sealed(
            Kind::RenderPipeline,
            self.key.digest(),
            &self.profile,
            self,
            out,
        )
    }

    fn from_bytes(bytes: &[u8], device: &Device) -> Result<RenderPipeline, ArtifactError> {
        let frame = open(bytes, Kind::RenderPipeline, device)?;
        let p: RenderPipeline = decode(frame.body)?;
        require(p.key.digest() == &frame.key)?;
        require(p.profile == device.profile_fingerprint())?;
        for s in &p.stages {
            scan_words(&s.spirv)?;
        }
        Ok(p)
    }
}

/// The frame of `value` under the render descriptor key `key` for `device`'s profile.
fn seal_answer<T: Codec>(kind: Kind, value: &T, key: &PipelineKey, device: &Device) -> Vec<u8> {
    seal(
        kind,
        key.digest(),
        &device.profile_fingerprint(),
        &encode(value),
    )
}

/// The value `bytes` frame under the render descriptor key `key` for `device`'s profile: bytes
/// of another key are `WrongDevice`, like bytes of another profile.
fn open_answer<T: Codec>(
    kind: Kind,
    bytes: &[u8],
    key: &PipelineKey,
    device: &Device,
) -> Result<T, ArtifactError> {
    let frame = open(bytes, kind, device)?;
    if &frame.key != key.digest() {
        return Err(ArtifactError::WrongDevice);
    }
    Ok(decode(frame.body)?)
}

impl crate::reflection::LinkedUseAnswers {
    /// The versioned, self-checking encoding of the answers of the render descriptor of `key`
    /// ([`Device::render_descriptor_key`]) on `device`: the frame of an artifact, its key the
    /// descriptor's and its profile `device`'s.
    pub fn to_bytes(&self, key: &PipelineKey, device: &Device) -> Vec<u8> {
        seal_answer(Kind::LinkedUseAnswers, self, key, device)
    }

    /// Decode the answers [`LinkedUseAnswers::to_bytes`](Self::to_bytes) encoded under `key` for
    /// `device`'s profile, or say why the bytes cannot be trusted (another key or profile:
    /// `WrongDevice`); the caller treats every error as a miss.
    pub fn from_bytes_for_key(
        bytes: &[u8],
        key: &PipelineKey,
        device: &Device,
    ) -> Result<crate::reflection::LinkedUseAnswers, ArtifactError> {
        open_answer(Kind::LinkedUseAnswers, bytes, key, device)
    }
}

impl crate::reflection::RenderPipelineReflection {
    /// The versioned, self-checking encoding of the reflection of the render descriptor of `key`
    /// ([`Device::render_descriptor_key`]) on `device`, as
    /// [`LinkedUseAnswers::to_bytes`](crate::reflection::LinkedUseAnswers::to_bytes) frames its
    /// answers.
    pub fn to_bytes(&self, key: &PipelineKey, device: &Device) -> Vec<u8> {
        seal_answer(Kind::RenderReflection, self, key, device)
    }

    /// Decode the reflection [`RenderPipelineReflection::to_bytes`](Self::to_bytes) encoded under
    /// `key` for `device`'s profile, as
    /// [`LinkedUseAnswers::from_bytes_for_key`](crate::reflection::LinkedUseAnswers::from_bytes_for_key)
    /// decodes its answers.
    pub fn from_bytes_for_key(
        bytes: &[u8],
        key: &PipelineKey,
        device: &Device,
    ) -> Result<crate::reflection::RenderPipelineReflection, ArtifactError> {
        open_answer(Kind::RenderReflection, bytes, key, device)
    }
}
