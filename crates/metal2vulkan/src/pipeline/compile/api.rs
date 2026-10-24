//! The public compile API on [`Device`].
//!
//! - [`Device::compile_function`] compiles one function into the opaque [`CompiledFunction`] the
//!   caller caches under [`Device::function_compile_key`]; the key needs no decode.
//! - [`Device::compile_compute_pipeline`] and [`Device::compile_render_pipeline`] compile each
//!   `StageSource::Function` stage, take each `StageSource::Compiled` stage as it is (link and
//!   relocate only: no decode, no lowering), link the stages, and relocate each module.
//! - [`Device::compute_pipeline_key`] needs no decode. [`Device::render_pipeline_key`] decodes and
//!   specializes a `Function` vertex stage (no lowering): its key includes the vertex-fetch
//!   projection over the attributes it reads, and [`RenderPipelineKeys`] hands those inputs out so
//!   the stage can be compiled on its own. It decodes a `Function` fragment stage only when a color
//!   attachment has a signed-integer format: the fragment's key includes the ones it writes.
//!
//! Every entry point runs under `catch_unwind`: a panic is `Internal(Panic)`. A compute pipeline's
//! kernel compiles under its stage-input descriptor (`LoweringInputs::stage_input`), which a
//! `Compiled` kernel's `[[stage_in]]` fetches must match. A render pipeline whose vertex function
//! is post-tessellation has four stages: link plans the control stage, whose synthesized vertex and
//! control modules come first. A stage's linked functions resolve its visible-function references
//! and are part of its `FunctionCompileKey`. A render pipeline whose `maxVertexAmplificationCount`
//! is above 1 is a multiview pipeline: link checks it, and its requirements add `multiview` and
//! `maxMultiviewViewCount` (and `multiviewTessellationShader` with tessellation stages).
//! [`Device::compile_mesh_pipeline`] compiles a mesh render pipeline: its object and mesh functions
//! as they compile alone, and its fragment function for the mesh function's primitive outputs.

use crate::budget::{Budget, Meter};
use crate::cache::VariantKey;
use crate::device::requirements::{Requirements, VkExtension, VkFeature, VkLimit, VkProperty};
use crate::device::{mechanism, Device, DeviceProfile};
use crate::error::{
    DescriptorError, Error, Feature, InternalError, InternalKind, RuntimeInput, Unsupported,
};
use crate::library::{CompiledFunction, FunctionArtifact, FunctionCompileKey};
use crate::lower::select::LowerInputs;
use crate::lower::simd::workgroup::arena_bytes;
use crate::pipeline::artifact::{
    ComputePipeline, ComputePipelineMetalInfo, PrecisionNote, RenderPipeline, RuntimeDependency,
    StageModule,
};
use crate::pipeline::compile::key::{
    function_key, mesh_descriptor_key, pipeline_key, render_descriptor_key,
};
use crate::pipeline::compile::{
    color_group, compile_stage, compile_stage_from, dynamic_order, guarded, linked_keys,
    primitive_id_carried_on, reads_primitive_id_varying, stage_inputs, FragmentPass, InputsFrom,
    RawIndices, StageEntries, NO_PRIMITIVE_VARYINGS,
};
use crate::pipeline::descriptor::{
    ComputePipelineDescriptor, FunctionDescriptor, MeshPipelineDescriptor,
    RenderPipelineDescriptor, StageSource, TileRenderPipelineDescriptor,
};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::graphics::{GraphicsState, StorageAttachmentView};
use crate::pipeline::link::geometry;
use crate::pipeline::link::mesh::synthesized_stage as mesh_vertex_stage;
use crate::pipeline::link::tessellation::{note_evaluation_stage, synthesized_stages};
use crate::pipeline::link::{link, LinkInputs, Linked, RenderLinkInputs};
use crate::pipeline::lower_inputs::LoweringInputs;
use crate::pipeline::relocate::relocate;
use crate::pipeline::stage_io::{IoRole, StageBuiltin, StageInFetch};
use crate::pipeline::sym::{Stage, VaryingKey};
use crate::spirv::serialize::ENTRY_NAME;
use crate::vk;

/// What one compile cost against its [`Budget`] ([`Device::compile_function_with_report`]): the
/// two bounds the budget sets besides its deadline, as the compile's meter counted them up to its
/// end or its refusal.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CompileReport {
    /// The deterministic work units the compile spent (the budget's work bound; the same input
    /// on the same device spends the same).
    pub work_units: u64,
    /// The peak of the byte ledger the compile's live data was charged to (the budget's memory
    /// bound).
    pub ledger_peak: u64,
}

impl CompileReport {
    /// What `m` counted.
    pub(crate) fn of(m: &Meter) -> CompileReport {
        CompileReport {
            work_units: m.work(),
            ledger_peak: m.ledger_peak(),
        }
    }
}

/// The keys of a render pipeline: the pipeline's, and each stage's `FunctionCompileKey`, so the
/// consumer can look up compiled functions before compiling, with the lowering inputs each
/// `Function` stage compiles under, so a miss compiles it with [`Device::compile_function`] and the
/// pipeline takes it as `StageSource::Compiled`.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RenderPipelineKeys {
    pub pipeline: crate::cache::PipelineKey,
    pub vertex: FunctionCompileKey,
    pub fragment: Option<FunctionCompileKey>,
    /// The inputs a `Function` vertex stage compiles under: the variant's choices for the vertex
    /// stage and the vertex-fetch projection over the attributes it reads, which only a decode
    /// learns. `compile_function` under them produces [`vertex`](Self::vertex). `None` for a
    /// `Compiled` vertex stage.
    pub vertex_inputs: Option<LoweringInputs>,
    /// The inputs a `Function` fragment stage compiles under (`VariantKey::lowering_inputs` of
    /// the fragment stage, the signed-integer color attachments it writes, whether the pipeline is
    /// multisampled, and the state of the attachments it reads where the device orders framebuffer
    /// fetch only through the pixel interlock). `None` for a `Compiled` fragment stage or none.
    pub fragment_inputs: Option<LoweringInputs>,
}

impl Device {
    /// The key a compile of `f` under `inp` produces, without decoding: the function's AIR hash and
    /// entry name, its constant values as supplied, this device's profile fingerprint, the
    /// canonical lowering inputs, and its linked functions.
    pub fn function_compile_key(
        &self,
        f: &FunctionDescriptor,
        inp: &LoweringInputs,
    ) -> FunctionCompileKey {
        let info = f.function.info();
        let dynamic = dynamic_order(&f.function, &f.linked, &f.libraries);
        function_key(
            &info.air_hash,
            info.name,
            &f.constants,
            &self.profile_fingerprint(),
            inp,
            (&linked_keys(&f.linked), &dynamic.key()),
        )
    }

    /// The key of render pipeline descriptor `d` on this device, without decoding: a digest of
    /// every field of `d`, each `Function` stage by its [`Device::function_compile_key`] under the
    /// default lowering inputs (its AIR hash, entry name, constants, linked functions and dynamic
    /// libraries), each `Compiled` stage by its key, with this device's profile fingerprint and
    /// [`SEMANTICS_VERSION`](crate::cache::SEMANTICS_VERSION). Everything a creation asks of `d`
    /// before its compile on this device ([`Device::render_variant_key`] under one variant, the
    /// descriptor's linked-use answers, its reflection) is a function of this key: a consumer
    /// may keep those answers under it. The crate keeps none.
    pub fn render_descriptor_key(&self, d: &RenderPipelineDescriptor) -> crate::cache::PipelineKey {
        let stage = |s: &StageSource| match s {
            StageSource::Function(f) => (
                0,
                *self
                    .function_compile_key(f, &LoweringInputs::default())
                    .digest(),
            ),
            StageSource::Compiled(c) => (1, *c.key().digest()),
        };
        render_descriptor_key(&self.profile_fingerprint(), d, stage)
    }

    /// The key of mesh render pipeline descriptor `d` on this device, without decoding, as
    /// [`Device::render_descriptor_key`] keys a render descriptor: a digest of every field of `d`,
    /// each `Function` stage by its [`Device::function_compile_key`] under the default lowering
    /// inputs, each `Compiled` stage by its key, with this device's profile fingerprint and
    /// [`SEMANTICS_VERSION`](crate::cache::SEMANTICS_VERSION).
    ///
    /// The descriptor's [`reflection`](MeshPipelineDescriptor::reflection) and
    /// [`slot_writes`](MeshPipelineDescriptor::slot_writes) are a function of this key, so a
    /// consumer may keep either under it and ask for it once however many states it makes of the
    /// descriptor. Both read the three stages alone, so the key is wider than they need; it is
    /// every field so that it means this descriptor on this device, as the render form does. The
    /// crate keeps no cache.
    pub fn mesh_descriptor_key(&self, d: &MeshPipelineDescriptor) -> crate::cache::PipelineKey {
        let stage = |s: &StageSource| match s {
            StageSource::Function(f) => (
                0,
                *self
                    .function_compile_key(f, &LoweringInputs::default())
                    .digest(),
            ),
            StageSource::Compiled(c) => (1, *c.key().digest()),
        };
        mesh_descriptor_key(&self.profile_fingerprint(), d, stage)
    }

    /// Compile `f` under `inp` for this device. The result is keyed by
    /// [`Device::function_compile_key`]; the crate keeps no cache.
    pub fn compile_function(
        &self,
        f: &FunctionDescriptor,
        inp: &LoweringInputs,
        b: &Budget,
    ) -> Result<CompiledFunction, Error> {
        self.compile_function_with_report(f, inp, b).0
    }

    /// [`Device::compile_function`], and what the compile cost against `b`
    /// ([`CompileReport`]), whether it compiled or not.
    pub fn compile_function_with_report(
        &self,
        f: &FunctionDescriptor,
        inp: &LoweringInputs,
        b: &Budget,
    ) -> (Result<CompiledFunction, Error>, CompileReport) {
        let mut m = Meter::new(b);
        let compiled = guarded(|| {
            compile_stage(
                &f.function,
                &f.constants,
                (&f.linked, &f.libraries),
                self.profile(),
                InputsFrom::Given(inp),
                &mut m,
            )
        });
        (compiled, CompileReport::of(&m))
    }

    /// [`Device::compile_function`], reading the stage's specialized entry from `entries`: the
    /// one a linked-use question or a key of the same creation made
    /// ([`RenderPipelineDescriptor::fragment_writes_depth_unattached`],
    /// [`Device::render_variant_key`]), else one made there now, so a creation decodes and
    /// specializes each stage once. It compiles what `compile_function` compiles; the
    /// specialization a creation already did is not charged to `b` again.
    pub fn compile_function_from_entries(
        &self,
        f: &FunctionDescriptor,
        inp: &LoweringInputs,
        b: &Budget,
        entries: &StageEntries,
    ) -> Result<CompiledFunction, Error> {
        let mut m = Meter::new(b);
        compile_stage_from(f, self.profile(), InputsFrom::Given(inp), entries, &mut m)
    }

    /// The key of the compute pipeline `d` (no decode).
    pub fn compute_pipeline_key(&self, d: &ComputePipelineDescriptor) -> crate::cache::PipelineKey {
        let stage = match &d.kernel {
            StageSource::Function(f) => self.function_compile_key(f, &kernel_inputs(d)),
            StageSource::Compiled(c) => *c.key(),
        };
        pipeline_key(
            b"compute",
            &self.profile_fingerprint(),
            &[stage],
            &compute_fields(d),
        )
    }

    /// Compile the compute pipeline `d` under budget `b`. A `Function` kernel compiles under
    /// `d`'s stage-input descriptor and `maxCallStackDepth`; a `Compiled` kernel whose recursion's
    /// call stack was sized for another depth (`RuntimeFact::CallStackDepth`) is
    /// `InvalidDescriptor(MaxCallStackDepth)`. A stage-input attribute whose buffer has no layout,
    /// read or not, is `InvalidDescriptor(StageInputLayout)`, where Metal stops at an assertion.
    /// A `Function` kernel's pipeline is held to the device's compiler word bound
    /// ([`Device::fit_compute_pipeline`]).
    pub fn compile_compute_pipeline(
        &self,
        d: &ComputePipelineDescriptor,
        b: &Budget,
    ) -> Result<ComputePipeline, Error> {
        guarded(|| {
            let inputs = kernel_inputs(d);
            let first = self.compute_pipeline(d, &inputs, &mut Meter::new(b))?;
            let bound = self.profile().module_word_bound();
            self.fit_compute_within(d, &inputs, b, first, bound)
        })
    }

    /// The key of the tile render pipeline `d` (no decode).
    pub fn tile_pipeline_key(&self, d: &TileRenderPipelineDescriptor) -> crate::cache::PipelineKey {
        let stage = match &d.tile {
            StageSource::Function(f) => self.function_compile_key(f, &tile_inputs(d)),
            StageSource::Compiled(c) => *c.key(),
        };
        pipeline_key(
            b"tile",
            &self.profile_fingerprint(),
            &[stage],
            &tile_fields(d),
        )
    }

    /// Compile the tile render pipeline `d` under budget `b`: its tile function, compiled under the
    /// descriptor's color attachment formats (`LoweringInputs::tile_attachments`, the color index
    /// of each attachment whose format is set), as a compute pipeline that runs one threadgroup per
    /// tile of the render area and stages the attachments it reaches (`lower::access::tile`). The
    /// consumer dispatches it in place of `dispatchThreadsPerTile:` with
    /// `BindState::imageblock_size` set to the pass's tile size and `BindState::render_area` to its
    /// render area, binding each attachment it reaches (`WriteSource::TileAttachment`). A
    /// `Compiled` tile function must have been compiled under the same formats: one compiled
    /// without tile inputs is `InvalidDescriptor(StageMismatch)`, and one that stages an attachment
    /// the descriptor gives another format (`RuntimeFact::TileAttachmentFormat`) is
    /// `InvalidDescriptor(ColorAttachment)`. A multisampled pass (`raster_sample_count` above 1) is
    /// `Unsupported{TileShading, NotImplemented}`: what its imageblock's samples are is not read.
    /// The tile function's recursion is sized for `d`'s `maxCallStackDepth` as a compute kernel's
    /// is. A `Function` tile function's pipeline is held to the device's compiler word bound
    /// ([`Device::fit_tile_pipeline`]).
    pub fn compile_tile_pipeline(
        &self,
        d: &TileRenderPipelineDescriptor,
        b: &Budget,
    ) -> Result<ComputePipeline, Error> {
        guarded(|| {
            let inputs = tile_inputs(d);
            let first = self.tile_pipeline(d, &inputs, &mut Meter::new(b))?;
            let bound = self.profile().module_word_bound();
            self.fit_tile_within(d, &inputs, b, first, bound)
        })
    }

    /// The keys of the render pipeline `d`, with the lowering inputs of each `Function` stage. A
    /// `Function` vertex stage is decoded and specialized under `b` (not lowered) to learn the
    /// attributes it reads, and a `Function` fragment stage to learn the colors it writes and
    /// reads.
    pub fn render_pipeline_key(
        &self,
        d: &RenderPipelineDescriptor,
        b: &Budget,
    ) -> Result<RenderPipelineKeys, Error> {
        self.render_variant_key(d, &VariantKey::new(), b, &StageEntries::new())
    }

    /// The keys of the render pipeline `d` compiled under variant `v`
    /// ([`Device::compile_render_variant`]). Each `Function` stage's specialized entry is the one
    /// `entries` holds, or one made there now: a creation that gave the same object to its
    /// descriptor's checks specializes each stage once.
    pub fn render_variant_key(
        &self,
        d: &RenderPipelineDescriptor,
        v: &VariantKey,
        b: &Budget,
        entries: &StageEntries,
    ) -> Result<RenderPipelineKeys, Error> {
        guarded(|| {
            crate::pipeline::private_state::check_render(d)?;
            let p = self.profile();
            // The attachments as the private logic operation draws them (`render_pipeline`).
            let (operated, _) = crate::pipeline::private_state::logic_operation(d, p)?;
            let d = operated.as_ref();
            // A color group's variant is the pipeline of its group's attachments
            // (`render_pipeline`).
            let group = color_group::group(d, v, p)?;
            let grouped = group.as_ref().map(|g| color_group::descriptor(d, g));
            let zeroed = with_zeroed_colors(grouped.as_ref().unwrap_or(d), v)?;
            let d = zeroed.as_ref();
            // The vertex function writes the primitive ID a fragment function reads from its
            // varying, on a device without a fragment `PrimitiveId` or behind an emulated
            // post-tessellation vertex function.
            let primitive_id = match primitive_id_carried_on(p) {
                true => reads_primitive_id_varying(
                    d.fragment.as_ref(),
                    d.max_fragment_call_stack_depth,
                    entries,
                    &mut Meter::new(b),
                )?,
                false => false,
            };
            let (vertex, vertex_inputs, written, varyings, uniform) = match &d.vertex {
                StageSource::Function(f) => {
                    let s = stage_inputs(
                        f,
                        p,
                        InputsFrom::Vertex(
                            d.vertex_descriptor.as_ref(),
                            v,
                            d.max_vertex_call_stack_depth,
                            primitive_id,
                            d.tessellation.as_ref(),
                        ),
                        entries,
                        &mut Meter::new(b),
                    )?;
                    let key = self.function_compile_key(f, &s.inputs);
                    (key, Some(s.inputs), s.writes, s.varyings, s.uniform)
                }
                StageSource::Compiled(c) => (
                    *c.key(),
                    None,
                    raw_indices_written(c.artifact()),
                    varyings_written(c.artifact()),
                    uniform_written(c.artifact()),
                ),
            };
            let (fragment, fragment_inputs) = match &d.fragment {
                // Every fragment function is decoded, to learn which colors it writes that the
                // pipeline has no attachment for, as `compile_render_pipeline` and a mesh
                // pipeline's fragment learn them; and so which signed-integer attachments it
                // writes, which attachments it reads on a device that orders framebuffer fetch only
                // through the pixel interlock, and whether it reads the `[[viewport_array_index]]`
                // or `[[render_target_array_index]]` its vertex function writes.
                Some(source @ StageSource::Function(f)) => {
                    // Whether the function applies the private masks itself shows only in its
                    // module, so a pipeline whose masks drop a sample lowers it once.
                    let mut pass: FragmentPass<'_> = d.into();
                    pass.color_group = group.as_ref();
                    if crate::pipeline::private_state::applied_masks(d).is_some() {
                        let inputs = InputsFrom::Fragment(
                            pass,
                            v,
                            written,
                            &NO_PRIMITIVE_VARYINGS,
                            Some(&varyings),
                            &uniform,
                        );
                        let compiled = self.stage(source, inputs, &mut Meter::new(b))?;
                        pass.sample_mask = crate::pipeline::private_state::shader_sample_mask(
                            d,
                            compiled.artifact(),
                        );
                    }
                    let inp = stage_inputs(
                        f,
                        p,
                        InputsFrom::Fragment(
                            pass,
                            v,
                            written,
                            &NO_PRIMITIVE_VARYINGS,
                            Some(&varyings),
                            &uniform,
                        ),
                        entries,
                        &mut Meter::new(b),
                    )?
                    .inputs;
                    (Some(self.function_compile_key(f, &inp)), Some(inp))
                }
                Some(StageSource::Compiled(c)) => (Some(*c.key()), None),
                None => (None, None),
            };
            let stages: Vec<FunctionCompileKey> = std::iter::once(vertex).chain(fragment).collect();
            Ok(RenderPipelineKeys {
                pipeline: pipeline_key(
                    b"render",
                    &self.profile_fingerprint(),
                    &stages,
                    &(render_fields(d) + &variant_fields(v)),
                ),
                vertex,
                fragment,
                vertex_inputs,
                fragment_inputs,
            })
        })
    }

    /// Compile the render pipeline `d` under budget `b`. A vertex attribute whose buffer has no
    /// layout, read by the vertex function or not, is `InvalidDescriptor(VertexLayout)`, where
    /// Metal stops at an assertion.
    pub fn compile_render_pipeline(
        &self,
        d: &RenderPipelineDescriptor,
        b: &Budget,
    ) -> Result<RenderPipeline, Error> {
        self.compile_render_variant(d, &VariantKey::new(), b)
    }

    /// Compile the render pipeline `d` under the variant `v` a render plan asked for
    /// (`BindingPlan::variant`) and budget `b`: each `Function` stage under
    /// [`VariantKey::lowering_inputs`] of its stage (the vertex stage's projection pulls `v`'s
    /// vertex buffers), linked with those layouts pulled. A `Compiled` stage must have been
    /// compiled under the same inputs; link refuses a vertex stage compiled under another
    /// projection. Each stage's recursion is sized for `d`'s `maxVertexCallStackDepth` or
    /// `maxFragmentCallStackDepth`, as a compute kernel's is for its pipeline's; a `Compiled` stage
    /// sized for another depth is `InvalidDescriptor(MaxCallStackDepth)`. A pipeline of `Function`
    /// stages is held to the device's compiler word bound ([`Device::fit_render_variant`]).
    pub fn compile_render_variant(
        &self,
        d: &RenderPipelineDescriptor,
        v: &VariantKey,
        b: &Budget,
    ) -> Result<RenderPipeline, Error> {
        guarded(|| {
            let first = self.render_pipeline(d, v, &mut Meter::new(b))?;
            self.fit_render_within(d, v, b, first, self.profile().module_word_bound())
        })
    }

    /// Compile the mesh render pipeline `d` under budget `b`: its object function (when it has one)
    /// as it compiles alone, its mesh function as it compiles alone but for a payload that holds
    /// at least the object function's (`LoweringInputs::object_payload`), and its fragment
    /// function for the mesh function's primitive outputs (`LoweringInputs::primitive_varyings`),
    /// its raw render target slice and viewport indices and the pipeline's attachments, as a
    /// render pipeline's. The
    /// result is a [`RenderPipeline`] whose stages are the object stage (`TASK_EXT`), the mesh
    /// stage (`MESH_EXT`) and the fragment stage: it has no vertex input and no input assembly (its
    /// `GraphicsState::topology` is unused), and Metal's draws of it are `drawMeshThreadgroups` and
    /// `drawMeshThreads`. On a device without `VK_EXT_mesh_shader` the mesh stage is a `COMPUTE`
    /// module that writes records, followed by a synthesized `VERTEX` stage that draws them as a
    /// list of the mesh's topology (`GraphicsState::mesh_records`). A `Compiled` stage must have
    /// been compiled under the same inputs and Metal's call stack depth of 1.
    pub fn compile_mesh_pipeline(
        &self,
        d: &MeshPipelineDescriptor,
        b: &Budget,
    ) -> Result<RenderPipeline, Error> {
        self.compile_mesh_variant(d, &VariantKey::new(), b)
    }

    /// Compile the mesh render pipeline `d` under the variant `v` a plan of it asked for
    /// (`BindingPlan::variant`) and budget `b`, as [`Device::compile_mesh_pipeline`] does: each
    /// `Function` stage under [`VariantKey::lowering_inputs`] of its stage, the fragment function's
    /// for the mesh function's outputs. A pipeline of `Function` stages is held to the device's
    /// compiler word bound ([`Device::fit_mesh_variant`]).
    pub fn compile_mesh_variant(
        &self,
        d: &MeshPipelineDescriptor,
        v: &VariantKey,
        b: &Budget,
    ) -> Result<RenderPipeline, Error> {
        guarded(|| {
            let first = self.mesh_pipeline(d, v, &mut Meter::new(b))?;
            self.fit_mesh_within(d, v, b, first, self.profile().module_word_bound())
        })
    }

    /// [`Device::compile_mesh_variant`] with the caller's meter, unfitted.
    pub(crate) fn mesh_pipeline(
        &self,
        d: &MeshPipelineDescriptor,
        v: &VariantKey,
        m: &mut Meter,
    ) -> Result<RenderPipeline, Error> {
        // A mesh draw's plan zeroes and drops no color, nor writes one through the interlock it
        // does not fetch, nor shadows one.
        if !v.zeroed_colors.is_empty()
            || !v.dropped_colors.is_empty()
            || !v.interlocked_colors.is_empty()
            || !v.extended_shadows.is_empty()
        {
            return Err(invariant(
                "a mesh variant zeroing, dropping, interlocking or shadowing colors",
            ));
        }
        let p = self.profile();
        let object = match &d.object {
            Some(o) => {
                // An object function on a device that folds its grids runs any object grid in
                // pieces within the task stage's limits (`bind::mesh_arguments`).
                let mut inputs = v.lowering_inputs(Stage::Object);
                inputs.folded_groups = crate::lower::mesh::folds_object_grids(p);
                Some(self.stage(o, InputsFrom::Given(&inputs), m)?)
            }
            None => None,
        };
        // The mesh function reads its payload within the object function's
        // (`LoweringInputs::object_payload`).
        let mut mesh_inputs = v.lowering_inputs(Stage::Mesh);
        mesh_inputs.object_payload = object.as_ref().and_then(|o| o.artifact().interface.payload);
        let mesh = self.stage(&d.mesh, InputsFrom::Given(&mesh_inputs), m)?;
        let written = raw_indices_written(mesh.artifact());
        let primitive = primitive_varyings_written(mesh.artifact());
        let varyings = varyings_written(mesh.artifact());
        let fragment = match &d.fragment {
            Some(f) => {
                let inputs = InputsFrom::Fragment(
                    d.into(),
                    v,
                    written,
                    &primitive,
                    Some(&varyings),
                    &NO_PRIMITIVE_VARYINGS,
                );
                Some(self.stage(f, inputs, m)?)
            }
            None => None,
        };
        let compiled: Vec<&CompiledFunction> = object
            .iter()
            .chain(std::iter::once(&mesh))
            .chain(&fragment)
            .collect();
        for c in &compiled {
            check_call_stack(c.artifact(), 1)?;
        }
        let artifacts: Vec<&FunctionArtifact> = compiled.iter().map(|c| c.artifact()).collect();
        let mut linked = link(&artifacts, &LinkInputs::mesh(d), p)?;
        let mut stages = Vec::with_capacity(artifacts.len());
        for (i, a) in artifacts.iter().enumerate() {
            stages.push(stage_module(a, &linked, i)?);
        }
        note_sample_position(d.into(), &artifacts, &mut stages);
        let keys: Vec<FunctionCompileKey> = artifacts.iter().map(|a| a.key).collect();
        let mut requirements = mesh.artifact().requirements.clone();
        for a in artifacts
            .iter()
            .filter(|a| a.interface.stage != Stage::Mesh)
        {
            requirements = p.union_requirements(&requirements, &a.requirements);
        }
        // An emulated mesh function's records are drawn by a synthesized vertex stage, which
        // follows the mesh stage's compute module, the last compute module (an object
        // function's comes first).
        let records = linked
            .graphics
            .as_ref()
            .and_then(|g| g.mesh_records.as_ref());
        let mut mesh_facts = Vec::new();
        if let Some(records) = records {
            // Drawn from each primitive's last index first: Apple's last provoking vertex on a
            // device that takes flat inputs from the first (`VariantKey::mesh_last_first`).
            let last_first = v.mesh_last_first && records.corners > 1;
            let (module, r) = mesh_vertex_stage(records, last_first, &linked, p, m)?;
            if last_first {
                let fact = RuntimeFact::MeshLastFirst;
                mesh_facts.push(RuntimeDependency::new(fact, mechanism(&fact, p)));
            }
            requirements = p.union_requirements(&requirements, &r);
            let at = stages
                .iter()
                .rposition(|s| s.stage == vk::ShaderStage::COMPUTE)
                .ok_or_else(|| invariant("an emulated mesh pipeline without its compute stage"))?;
            stages.insert(at + 1, module);
        }
        let mut dependencies = dependencies(&artifacts, self);
        dependencies.extend(mesh_facts);
        let depth = (&mut linked, &mut stages, &mut dependencies);
        if let Some(r) = depth16_stage((d.into(), v, d.fragment.is_none()), depth, p, m)? {
            requirements = p.union_requirements(&requirements, &r);
        }
        let (graphics, requirements) = graphics_requirements(requirements, &linked, &artifacts, p)?;
        let profile = self.profile_fingerprint();
        Ok(RenderPipeline {
            key: pipeline_key(
                b"mesh",
                &profile,
                &keys,
                &(mesh_fields(d) + &variant_fields(v)),
            ),
            profile,
            stages,
            layout: linked.layout,
            vertex_input: linked.vertex_input,
            graphics,
            dependencies,
            interface: linked.interface,
            requirements,
            variant_bits: artifacts.iter().fold(0, |bits, a| bits | a.variant_bits),
        })
    }

    /// A stage from its source: compiled now under `inputs`, or taken as compiled (link and
    /// relocate only) when it was compiled for this device's profile.
    /// [`Device::stage`] of a render pipeline's fragment stage `src`, reading a `Function`
    /// stage's specialized entry from `entries` where the pipeline made one there
    /// (`reads_primitive_id_varying`).
    fn fragment_stage(
        &self,
        src: &StageSource,
        inputs: InputsFrom<'_>,
        entries: Option<&StageEntries>,
        m: &mut Meter,
    ) -> Result<CompiledFunction, Error> {
        match (src, entries) {
            (StageSource::Function(fd), Some(e)) => {
                compile_stage_from(fd, self.profile(), inputs, e, m)
            }
            _ => self.stage(src, inputs, m),
        }
    }

    fn stage(
        &self,
        src: &StageSource,
        inputs: InputsFrom<'_>,
        m: &mut Meter,
    ) -> Result<CompiledFunction, Error> {
        match src {
            StageSource::Function(f) => compile_stage(
                &f.function,
                &f.constants,
                (&f.linked, &f.libraries),
                self.profile(),
                inputs,
                m,
            ),
            StageSource::Compiled(c) => {
                if c.artifact().profile != self.profile_fingerprint() {
                    return Err(Error::InvalidDescriptor(
                        DescriptorError::ForeignCompiledFunction,
                    ));
                }
                Ok(c.clone())
            }
        }
    }

    /// [`Device::compile_compute_pipeline`] with the caller's meter (tests count its work),
    /// unfitted, a `Function` kernel compiled under `inputs` (`d`'s own, `kernel_inputs`, or a
    /// fit's).
    pub(crate) fn compute_pipeline(
        &self,
        d: &ComputePipelineDescriptor,
        inputs: &LoweringInputs,
        m: &mut Meter,
    ) -> Result<ComputePipeline, Error> {
        let p = self.profile();
        // Any attribute whose buffer has no layout, read or not, stops Metal.
        if let Some(stage_input) = &d.stage_input {
            stage_input.check_layouts()?;
        }
        let kernel = self.stage(&d.kernel, InputsFrom::Given(inputs), m)?;
        let a = kernel.artifact();
        check_stage_in(a, d)?;
        check_call_stack(a, d.max_call_stack_depth)?;
        let linked = link(&[a], &LinkInputs::compute(), p)?;
        let stage = stage_module(a, &linked, 0)?;
        let profile = self.profile_fingerprint();
        let key = pipeline_key(b"compute", &profile, &[a.key], &compute_fields(d));
        let metal_info = self.metal_info(
            &kernel,
            d.max_total_threads_per_threadgroup,
            d.support_indirect_command_buffers,
        );
        Ok(ComputePipeline {
            key,
            profile,
            stage,
            layout: linked.layout,
            dependencies: dependencies(&[a], self),
            interface: linked.interface,
            metal_info,
            requirements: a.requirements.clone(),
            variant_bits: a.variant_bits,
        })
    }

    /// [`Device::compile_tile_pipeline`] with the caller's meter, unfitted, a `Function` tile
    /// function compiled under `inputs` (`d`'s own, `tile_inputs`, or a fit's).
    pub(crate) fn tile_pipeline(
        &self,
        d: &TileRenderPipelineDescriptor,
        inputs: &LoweringInputs,
        m: &mut Meter,
    ) -> Result<ComputePipeline, Error> {
        if d.raster_sample_count > 1 {
            return Err(Unsupported::not_implemented(Feature::TileShading).into());
        }
        let p = self.profile();
        let kernel = self.stage(&d.tile, InputsFrom::Given(inputs), m)?;
        let a = kernel.artifact();
        if a.interface.stage != Stage::Kernel || !a.facts.contains(&RuntimeFact::TileFunction) {
            return Err(Error::InvalidDescriptor(DescriptorError::StageMismatch {
                expected: Stage::Kernel,
                found: a.interface.stage,
            }));
        }
        check_tile_formats(a, d)?;
        check_tile_threadgroup(a, d)?;
        check_call_stack(a, d.max_call_stack_depth)?;
        let linked = link(&[a], &LinkInputs::compute(), p)?;
        let stage = stage_module(a, &linked, 0)?;
        let profile = self.profile_fingerprint();
        let key = pipeline_key(b"tile", &profile, &[a.key], &tile_fields(d));
        let max_threads = d
            .max_total_threads_per_threadgroup
            .filter(|&n| n > 0)
            .map_or(TILE_MAX_THREADS, |n| n.min(TILE_MAX_THREADS));
        let mut metal_info = self.metal_info(&kernel, Some(max_threads), false);
        metal_info.max_total_threads_per_threadgroup =
            tile_threads_within(metal_info.max_total_threads_per_threadgroup);
        metal_info.metal_max_total_threads_per_threadgroup =
            tile_threads_within(metal_info.metal_max_total_threads_per_threadgroup);
        Ok(ComputePipeline {
            key,
            profile,
            stage,
            layout: linked.layout,
            dependencies: dependencies(&[a], self),
            interface: linked.interface,
            metal_info,
            requirements: a.requirements.clone(),
            variant_bits: a.variant_bits,
        })
    }

    /// [`Device::compile_render_variant`] with the caller's meter, unfitted. A variant of one of
    /// the pipeline's color groups (`VariantKey::color_group`) compiles and links as the pipeline
    /// of its group's attachments (`color_group`), its key's choices of colors naming the group's.
    pub(crate) fn render_pipeline(
        &self,
        d: &RenderPipelineDescriptor,
        v: &VariantKey,
        m: &mut Meter,
    ) -> Result<RenderPipeline, Error> {
        crate::pipeline::private_state::check_render(d)?;
        let p = self.profile();
        // The attachments as the private logic operation draws them: its normalized and integer
        // attachments unblended (unwritten under a no-op), and Vulkan's operation where one is
        // needed.
        let (operated, logic_op) = crate::pipeline::private_state::logic_operation(d, p)?;
        let d = operated.as_ref();
        let group = color_group::group(d, v, p)?;
        let grouped = group.as_ref().map(|g| color_group::descriptor(d, g));
        let zeroed = with_zeroed_colors(grouped.as_ref().unwrap_or(d), v)?;
        let d = zeroed.as_ref();
        let pass = || {
            let mut pass = FragmentPass::from(d);
            pass.color_group = group.as_ref();
            pass
        };
        // On a device without a fragment `PrimitiveId` or without `tessellationShader`, whether
        // the vertex function writes the primitive ID's varying: the fragment entry specialized
        // for that is the one its compile reads.
        let entries = primitive_id_carried_on(p).then(StageEntries::new);
        let primitive_id = match &entries {
            Some(e) => reads_primitive_id_varying(
                d.fragment.as_ref(),
                d.max_fragment_call_stack_depth,
                e,
                m,
            )?,
            None => false,
        };
        let vertex = self.stage(
            &d.vertex,
            InputsFrom::Vertex(
                d.vertex_descriptor.as_ref(),
                v,
                d.max_vertex_call_stack_depth,
                primitive_id,
                d.tessellation.as_ref(),
            ),
            m,
        )?;
        check_call_stack(vertex.artifact(), d.max_vertex_call_stack_depth)?;
        let written = raw_indices_written(vertex.artifact());
        let varyings = varyings_written(vertex.artifact());
        let uniform = uniform_written(vertex.artifact());
        let fragment = match &d.fragment {
            Some(f @ StageSource::Function(_)) => {
                let inputs = InputsFrom::Fragment(
                    pass(),
                    v,
                    written,
                    &NO_PRIMITIVE_VARYINGS,
                    Some(&varyings),
                    &uniform,
                );
                let compiled = self.fragment_stage(f, inputs, entries.as_ref(), m)?;
                // Compiled again applying the private masks itself where they differ between
                // a quad's pixels, or where it could tell Vulkan's mask from Metal's.
                match crate::pipeline::private_state::shader_sample_mask(d, compiled.artifact()) {
                    Some(mask) => {
                        let mut masked = pass();
                        masked.sample_mask = Some(mask);
                        let inputs = InputsFrom::Fragment(
                            masked,
                            v,
                            written,
                            &NO_PRIMITIVE_VARYINGS,
                            Some(&varyings),
                            &uniform,
                        );
                        Some(self.fragment_stage(f, inputs, entries.as_ref(), m)?)
                    }
                    None => Some(compiled),
                }
            }
            Some(f) => {
                let inputs = InputsFrom::Fragment(
                    pass(),
                    v,
                    written,
                    &NO_PRIMITIVE_VARYINGS,
                    Some(&varyings),
                    &uniform,
                );
                Some(self.stage(f, inputs, m)?)
            }
            None => None,
        };
        if let Some(f) = &fragment {
            check_call_stack(f.artifact(), d.max_fragment_call_stack_depth)?;
        }
        let compiled: Vec<&CompiledFunction> = std::iter::once(&vertex).chain(&fragment).collect();
        let artifacts: Vec<&FunctionArtifact> = compiled.iter().map(|c| c.artifact()).collect();
        let mut linked = link(
            &artifacts,
            &LinkInputs::render_variant(d, v, group.clone()),
            p,
        )?;
        let mut stages = Vec::with_capacity(artifacts.len());
        for (i, a) in artifacts.iter().enumerate() {
            stages.push(stage_module(a, &linked, i)?);
        }
        note_sample_position(pass(), &artifacts, &mut stages);
        let keys: Vec<FunctionCompileKey> = artifacts.iter().map(|a| a.key).collect();
        let mut requirements = vertex.artifact().requirements.clone();
        for a in artifacts.iter().skip(1) {
            requirements = p.union_requirements(&requirements, &a.requirements);
        }
        // A tessellation pipeline's synthesized vertex and control stages come first.
        if let Some(c) = &linked.control {
            note_evaluation_stage(c, p, &mut stages);
            let synthesized = synthesized_stages(c, &linked, p, m)?;
            let mut all = Vec::with_capacity(synthesized.len() + stages.len());
            for (module, r) in synthesized {
                requirements = p.union_requirements(&requirements, &r);
                all.push(module);
            }
            all.extend(stages);
            stages = all;
        }
        // A fragment function that reads `[[barycentric_coord]]` as varyings, or a vertex function
        // that selects its slice or viewport through them: the geometry stages of the point, line
        // and triangle classes, before the fragment stage, or last without one.
        if let Some(g) = &linked.geometry {
            let at = stages
                .iter()
                .position(|s| s.stage == vk::ShaderStage::FRAGMENT)
                .unwrap_or(stages.len());
            let mut added = Vec::with_capacity(geometry::CLASSES.len());
            for (module, r) in g.stages(&linked, p, m)? {
                requirements = p.union_requirements(&requirements, &r);
                added.push(module);
            }
            stages.splice(at..at, added);
        }
        let mut dependencies = dependencies(&artifacts, self);
        if let Some(g) = &linked.geometry {
            let fact = RuntimeFact::SynthesizedGeometry {
                weights: !g.forms.is_empty(),
                layer: g.layer,
                viewport: g.viewport,
            };
            dependencies.push(RuntimeDependency {
                fact,
                mechanism: mechanism(&fact, self.profile()),
            });
        }
        let depth = (&mut linked, &mut stages, &mut dependencies);
        if let Some(r) = depth16_stage((pass(), v, d.fragment.is_none()), depth, p, m)? {
            requirements = p.union_requirements(&requirements, &r);
        }
        let (mut graphics, requirements) =
            graphics_requirements(requirements, &linked, &artifacts, p)?;
        graphics.sample_mask = crate::pipeline::private_state::sample_mask(
            d,
            fragment.as_ref().map(|f| f.artifact()),
            &graphics,
        )?;
        crate::pipeline::private_state::alpha_tested(d, fragment.as_ref().map(|f| f.artifact()))?;
        crate::pipeline::private_state::depth_compares_clamped(
            d,
            fragment.as_ref().map(|f| f.artifact()),
        )?;
        crate::pipeline::private_state::logic_operated(
            logic_op,
            fragment.as_ref().map(|f| f.artifact()),
            &graphics,
        )?;
        graphics.logic_op = logic_op;
        let mut requirements = requirements;
        if logic_op.is_some() {
            requirements.features.insert(VkFeature::LogicOp);
        }
        let profile = self.profile_fingerprint();
        Ok(RenderPipeline {
            key: pipeline_key(
                b"render",
                &profile,
                &keys,
                &(render_fields(d) + &variant_fields(v)),
            ),
            profile,
            stages,
            layout: linked.layout,
            vertex_input: linked.vertex_input,
            graphics,
            dependencies,
            interface: linked.interface,
            requirements,
            variant_bits: artifacts.iter().fold(0, |bits, a| bits | a.variant_bits),
        })
    }

    /// Answers to Metal's per-pipeline compute queries, under the descriptor's
    /// `maxTotalThreadsPerThreadgroup` (`max_threads`) and `supportIndirectCommandBuffers`
    /// (`icb`).
    fn metal_info(
        &self,
        kernel: &CompiledFunction,
        max_threads: Option<u32>,
        icb: bool,
    ) -> ComputePipelineMetalInfo {
        let f = self.profile().fields();
        let c = kernel.artifact();
        let mut max = self.metal_properties().max_total_threads_per_threadgroup;
        if c.kernel.flat_local_size {
            // The flattened tiers put the whole local size on X.
            max = max.min(f.limits.max_compute_work_group_size[0]);
            if let Some(s) = c.subgroup.size {
                max = max.min(f.subgroup.max_compute_workgroup_subgroups.saturating_mul(s));
            }
        }
        // Emulated simdgroups take `bytes_per_simdgroup` of threadgroup memory per simdgroup beside
        // the declared arena: at most as many threads as whole simdgroups fit.
        let emulation = c.facts.iter().find_map(|fact| match *fact {
            RuntimeFact::SimdgroupEmulation {
                bytes_per_simdgroup,
                arena,
            } => Some((u64::from(bytes_per_simdgroup.max(1)), arena_bytes(arena, 0))),
            _ => None,
        });
        if let Some((per_simdgroup, fixed)) = emulation {
            let free = u64::from(f.limits.max_compute_shared_memory_size).saturating_sub(fixed);
            let groups = free / per_simdgroup;
            max = max.min(u32::try_from(groups.saturating_mul(32)).unwrap_or(u32::MAX));
        }
        // Metal's own answer: Apple's 1024 under the same caps, without the device's limits.
        let mut metal = METAL_MAX_THREADS;
        for cap in [c.kernel.max_total_threads, max_threads]
            .into_iter()
            .flatten()
            // Metal's 0 is "the device maximum".
            .filter(|&n| n > 0)
        {
            max = max.min(cap);
            metal = metal.min(cap);
        }
        // The emulated scratch of the most threads counts as static threadgroup memory, so any
        // `[[threadgroup(n)]]` lengths that fit beside the reported length fit every threadgroup
        // up to the maximum (the arena grows by exactly their rounded lengths).
        // A predicated walk's vote word is static threadgroup memory beside the arena.
        let vote = if c.facts.contains(&RuntimeFact::PredicationVote) {
            u64::from(crate::lower::simd::divergent::VOTE_BYTES)
        } else {
            0
        };
        let static_bytes = match emulation {
            Some((per_simdgroup, fixed)) => {
                fixed.saturating_add(u64::from(max).div_ceil(32).saturating_mul(per_simdgroup))
            }
            None => c.kernel.static_threadgroup_bytes.saturating_add(vote),
        };
        ComputePipelineMetalInfo {
            max_total_threads_per_threadgroup: max,
            thread_execution_width: self.metal_properties().thread_execution_width,
            static_threadgroup_memory_length: u32::try_from(static_bytes).unwrap_or(u32::MAX),
            support_indirect_command_buffers: icb,
            metal_max_total_threads_per_threadgroup: metal,
        }
    }
}

/// State among a render pipeline's `dependencies` the depth variant `v` compiled it for
/// (`RuntimeFact::Depth16Pass` of `VariantKey::depth16_pass`, `RuntimeFact::Depth16Biased` of
/// `VariantKey::depth16_biased`, `RuntimeFact::Depth16BiasSide` of
/// `VariantKey::depth16_bias_raises`), and give one without a fragment
/// function (`fragmentless`) whose pass `pass` takes Metal's `Depth16Unorm` codes from its fragment
/// stage (`compile::depth16_codes`) the synthesized fragment stage
/// (`lower::stage_io::depth16::fragment_module`), as a fragment function would take them under
/// `LoweringInputs::depth16_codes`: its module after the `stages`, its push range and placement in
/// the `linked` layout and interface, and `RuntimeFact::Depth16Codes` among the `dependencies`.
/// It writes no color, so the pipeline's color state stays a fragment-less pipeline's. The module's
/// requirements, or `None` where the pipeline takes no such stage.
fn depth16_stage(
    (pass, v, fragmentless): (FragmentPass<'_>, &VariantKey, bool),
    (linked, stages, dependencies): (
        &mut Linked,
        &mut Vec<StageModule>,
        &mut Vec<RuntimeDependency>,
    ),
    p: &DeviceProfile,
    m: &mut Meter,
) -> Result<Option<Requirements>, Error> {
    use crate::lower::stage_io::depth16;
    if let Some(depth16) = v.depth16_pass {
        let fact = RuntimeFact::Depth16Pass { depth16 };
        dependencies.push(RuntimeDependency::new(fact, mechanism(&fact, p)));
    }
    if v.depth16_biased {
        let fact = RuntimeFact::Depth16Biased;
        dependencies.push(RuntimeDependency::new(fact, mechanism(&fact, p)));
    }
    if let Some(raises) = v.depth16_bias_raises {
        let fact = RuntimeFact::Depth16BiasSide { raises };
        dependencies.push(RuntimeDependency::new(fact, mechanism(&fact, p)));
    }
    if !fragmentless || !crate::pipeline::compile::depth16_codes(pass, v, p) {
        return Ok(None);
    }
    let module = depth16::fragment_module(v.depth16_bias_raises)?;
    crate::spirv::verify::verify(&module, p, m)?;
    let module = crate::spirv::serialize::serialize_for(module, p)?;
    let words = crate::spirv::serialize::relocate(&module, &*linked)?;
    stages.push(StageModule {
        stage: vk::ShaderStage::FRAGMENT,
        entry_point: ENTRY_NAME.to_owned(),
        spirv: words.into(),
        specialization: crate::pipeline::artifact::SpecializationLayout::default(),
        required_subgroup_size: None,
        stage_flags: vk::PipelineShaderStageCreateFlags::empty(),
        reconvergence: crate::pipeline::artifact::Reconvergence::Guaranteed,
        precision_notes: Vec::new(),
        strategies: crate::pipeline::artifact::StrategyCensus::default(),
        constant_blob: None,
        rounded_colors: Vec::new(),
        specialized_notes: Vec::new(),
        half_bit_colors: Vec::new(),
        own_pixel_textures: Vec::new(),
        snapshot_textures: Vec::new(),
        private_bytes: 0,
    });
    let placement = depth16::fragment_placement();
    if let Some((offset, size)) = placement.push_range {
        linked
            .layout
            .push_constants
            .push(crate::pipeline::layout::PushConstantRange {
                stages: vk::ShaderStageFlags::FRAGMENT,
                offset,
                size,
            });
    }
    linked.interface.push.push(placement);
    let fact = RuntimeFact::Depth16Codes { biased: true };
    dependencies.push(RuntimeDependency::new(fact, mechanism(&fact, p)));
    Ok(Some(module.requirements))
}

/// A render or mesh render pipeline's graphics state, and its stages' `requirements` with what
/// that state adds.
fn graphics_requirements(
    mut requirements: Requirements,
    linked: &Linked,
    artifacts: &[&FunctionArtifact],
    p: &DeviceProfile,
) -> Result<(GraphicsState, Requirements), Error> {
    let graphics = linked
        .graphics
        .clone()
        .ok_or_else(|| invariant("link of a render pipeline without graphics state"))?;
    // Its dynamic states need their features (extended dynamic state).
    requirements
        .features
        .extend(graphics.dynamic_state_features()?);
    // A pipeline that reads through snapshots samples a color attachment of its subpass in the
    // feedback loop layout, created with its flag (`GraphicsState::snapshot_reads`).
    if graphics.snapshot_reads {
        requirements
            .features
            .insert(VkFeature::AttachmentFeedbackLoopLayout);
    }
    // Its color blend state: attachments that blend differently, dual-source blending.
    requirements
        .features
        .extend(graphics.color_blend_features());
    // Its sample shading.
    requirements
        .features
        .extend(graphics.multisample_features());
    // Its sample count against the attachments' sample counts.
    requirements
        .properties
        .extend(graphics.sample_count_properties(&p.fields().enabled.properties)?);
    // The portability subset's features it relies on, named only where the device has them: where
    // the subset withholds one, link refused or pulled what it would allow (and the binder pulls a
    // per-bind stride that does not cover its attributes), and a device without the subset
    // withholds nothing.
    let enabled = &p.fields().enabled;
    let relies = |feature| {
        enabled
            .extensions
            .contains(&VkExtension::KhrPortabilitySubset)
            && !enabled.portability_withholds(feature)
    };
    let constant_alpha = graphics
        .color_blend_attachments()
        .iter()
        .filter_map(|&(_, blend)| blend)
        .any(|b| b.constant_alpha_color());
    if constant_alpha && relies(VkFeature::ConstantAlphaColorBlendFactors) {
        requirements
            .features
            .insert(VkFeature::ConstantAlphaColorBlendFactors);
    }
    if linked.vertex_input.may_access_beyond_stride()
        && relies(VkFeature::VertexAttributeAccessBeyondStride)
    {
        requirements
            .features
            .insert(VkFeature::VertexAttributeAccessBeyondStride);
    }
    // Ordered framebuffer fetch: the subpass and pipeline flags need the feature.
    if graphics.rasterization_order_attachment_access {
        requirements
            .features
            .insert(VkFeature::RasterizationOrderColorAttachmentAccess);
    }
    // The interlocked form's storage views of the attachments; the module's own requirements name
    // the interlock and the fragment stores.
    for a in &graphics.storage_attachments {
        requirements.formats.extend(a.format_usages());
    }
    // The shader-side stencil's shadow; the module's own requirements name the interlock, the
    // fragment stores, and multisampled storage images. A multisampled shadow is copied by image
    // copies (`maintenance8`), and it, or the neutral that stands in for it without a stencil
    // attachment, is a storage image of the pipeline's samples.
    if let Some(s) = &graphics.shadow_stencil {
        requirements.formats.extend(s.format_usages());
        if s.samples > 1 {
            requirements.features.insert(VkFeature::Maintenance8);
        }
    }
    // An in-shader depth test's stand-in; the module's own requirements name the interlock and
    // the fragment stores. A multisampled stand-in has the pipeline's samples; planes are
    // single-sample.
    let mut multisampled_stand_in = false;
    if let Some(t) = &graphics.shader_tests {
        requirements.formats.extend(t.depth.format_usages());
        multisampled_stand_in =
            t.depth.view == crate::pipeline::graphics::StandInView::Image && t.depth.samples > 1;
        // The stencil's stand-in, a multisampled image at more than one sample, copied to and
        // from the stencil aspect by image copies there (`maintenance8`).
        if let Some(s) = &t.stencil {
            requirements.formats.extend(s.format_usages());
            if s.samples > 1 {
                requirements.features.insert(VkFeature::Maintenance8);
                multisampled_stand_in = true;
            }
        }
    }
    let multisampled_shadow = artifacts.iter().any(|a| {
        a.facts.iter().any(|f| {
            matches!(
                f,
                RuntimeFact::ShadowStencil {
                    multisampled: true,
                    ..
                }
            )
        })
    });
    // A per-sample framebuffer fetch's multisampled storage images have the pipeline's
    // samples; planes are single-sample.
    let multisampled_storage = graphics
        .storage_attachments
        .iter()
        .any(|a| a.view == StorageAttachmentView::Attachment)
        && graphics.rasterization_samples.as_raw() > 1;
    if multisampled_shadow || multisampled_storage || multisampled_stand_in {
        requirements
            .properties
            .insert(VkProperty::StorageImageSampleCount(
                graphics.rasterization_samples.as_raw(),
            ));
    }
    // A multiview pipeline needs `multiview` and its view count whether or not a stage reads the
    // view index; the binder draws more amplifications than the device has views one at a time, so
    // the views it needs are at most those.
    let views = graphics
        .max_vertex_amplification_count
        .min(p.fields().limits.max_multiview_view_count.max(1));
    if views > 1 {
        requirements.features.insert(VkFeature::Multiview);
        if linked.control.is_some() {
            requirements
                .features
                .insert(VkFeature::MultiviewTessellationShader);
        }
        let needed = requirements
            .limits
            .entry(VkLimit::MaxMultiviewViewCount)
            .or_insert(0);
        *needed = (*needed).max(u64::from(views));
    }
    Ok((graphics, requirements))
}

/// `d` as variant `v` compiles it: with each color of `v.zeroed_colors`, which `d` has no
/// attachment at, added in its pixel format, every channel written and blending off, so link
/// gives the attachment the zero the fragment function writes to it
/// (`LoweringInputs::zeroed_colors`). `d` itself where `v` zeroes none. A key zeroing a color `d`
/// has, or zeroing any where `d` has no fragment function, is no plan's (an invariant).
fn with_zeroed_colors<'a>(
    d: &'a RenderPipelineDescriptor,
    v: &VariantKey,
) -> Result<std::borrow::Cow<'a, RenderPipelineDescriptor>, Error> {
    if v.zeroed_colors.is_empty() {
        return Ok(std::borrow::Cow::Borrowed(d));
    }
    if d.fragment.is_none() {
        return Err(invariant(
            "a variant zeroing colors of a pipeline without a fragment function",
        ));
    }
    let mut out = d.clone();
    for (&color, &format) in &v.zeroed_colors {
        match out.color_attachments.get_mut(color as usize) {
            Some(slot @ None) => {
                *slot = Some(crate::pipeline::descriptor::ColorAttachmentDescriptor::new(
                    format,
                ));
            }
            _ => {
                return Err(invariant(
                    "a variant zeroing a color the pipeline has, or past its colors",
                ))
            }
        }
    }
    Ok(std::borrow::Cow::Owned(out))
}

fn invariant(detail: &str) -> Error {
    Error::Internal(InternalError::new(
        InternalKind::Invariant,
        detail.to_owned(),
    ))
}

/// The lowering inputs a `Function` kernel of compute pipeline `d` compiles under: its
/// stage-input descriptor and its `maxCallStackDepth` when not Metal's default, 1.
fn kernel_inputs(d: &ComputePipelineDescriptor) -> LoweringInputs {
    LoweringInputs {
        stage_input: d.stage_input.clone(),
        max_call_stack_depth: (d.max_call_stack_depth > 1).then_some(d.max_call_stack_depth),
        ..LowerInputs::new()
    }
}

/// The lowering inputs a `Function` tile function of tile pipeline `d` compiles under: its color
/// attachments' pixel formats, and its `maxCallStackDepth` when not Metal's default, 1.
fn tile_inputs(d: &TileRenderPipelineDescriptor) -> LoweringInputs {
    let formats = d
        .color_attachment_formats
        .iter()
        .enumerate()
        .filter_map(|(i, f)| Some((u32::try_from(i).ok()?, (*f)?)))
        .collect();
    LoweringInputs {
        tile_attachments: Some(formats),
        max_call_stack_depth: (d.max_call_stack_depth > 1).then_some(d.max_call_stack_depth),
        threadgroup_size_matches_tile_size: d.threadgroup_size_matches_tile_size,
        ..LowerInputs::new()
    }
}

/// The tile function of artifact `a` answers the threadgroup `d` gives it: a `Compiled` one whose
/// builtins were compiled under the other `threadgroupSizeMatchesTileSize`
/// (`RuntimeFact::TileThreadgroup`) is `InvalidDescriptor(TileThreadgroupSize)` (a `Function`
/// one is compiled under `d`'s).
fn check_tile_threadgroup(
    a: &FunctionArtifact,
    d: &TileRenderPipelineDescriptor,
) -> Result<(), Error> {
    for fact in &a.facts {
        if let RuntimeFact::TileThreadgroup { tile_sized } = *fact {
            if tile_sized != d.threadgroup_size_matches_tile_size {
                return Err(Error::InvalidDescriptor(
                    DescriptorError::TileThreadgroupSize {
                        compiled: tile_sized,
                    },
                ));
            }
        }
    }
    Ok(())
}

/// Each attachment the tile function of artifact `a` stages has the format it was compiled for in
/// `d`: a `Compiled` tile function compiled under other formats is
/// `InvalidDescriptor(ColorAttachment)` (a `Function` one is compiled under `d`'s).
fn check_tile_formats(a: &FunctionArtifact, d: &TileRenderPipelineDescriptor) -> Result<(), Error> {
    for fact in &a.facts {
        let RuntimeFact::TileAttachmentFormat { color, format } = *fact else {
            continue;
        };
        let given = usize::try_from(color)
            .ok()
            .and_then(|i| d.color_attachment_formats.get(i).copied().flatten());
        if given != Some(format) {
            return Err(Error::InvalidDescriptor(DescriptorError::ColorAttachment {
                index: color,
            }));
        }
    }
    Ok(())
}

/// A tile pipeline's `maxTotalThreadsPerThreadgroup` bound: Metal's answer for every tile function
/// measured (an M4: plain, simdgroup shuffles and reductions, threadgroup memory,
/// an explicit imageblock, many live values), the largest tile threadgroup the binder runs (32x32,
/// `bind::dispatch::TILE_THREADGROUP_SIDE`), which `metal_info` lowers to the device's limits and
/// the descriptor's.
const TILE_MAX_THREADS: u32 = TILE_MAX_SIDE * TILE_MAX_SIDE;
/// Apple's GPUs' threads a threadgroup (a threadgroup of 2048 ran nothing on an M4).
const METAL_MAX_THREADS: u32 = 1024;

/// The widest and tallest tile threadgroup (`bind::dispatch::TILE_THREADGROUP_SIDE`).
const TILE_MAX_SIDE: u32 = crate::bind::dispatch::TILE_THREADGROUP_SIDE;

/// The most threads of a tile threadgroup within `limit`, as Metal answers a tile pipeline's
/// `maxTotalThreadsPerThreadgroup` (an M3 Max and an M4): the largest `w · h <= limit` of even `w`
/// and `h` from 2 to 32 (Metal's validation counts each side rounded up to
/// even), 0 below 4. A descriptor's 1000 answers 960 (32x30), 959 answers 900 (30x30) and 255
/// answers 252 (18x14); 1024 answers itself.
fn tile_threads_within(limit: u32) -> u32 {
    (1..=TILE_MAX_SIDE / 2)
        .map(|half| 2 * half)
        .filter_map(|w| {
            let h = (limit / w).min(TILE_MAX_SIDE) & !1;
            (h >= 2).then_some(w * h)
        })
        .max()
        .unwrap_or(0)
}

/// The tile descriptor fields the pipeline depends on: the attachment formats, the sample count,
/// and a threadgroup limit when the descriptor sets one (its Metal info), so a descriptor without
/// one keys as it did before the field. `threadgroup_size_matches_tile_size` is not one here: the
/// tile function compiles under it (`LoweringInputs::threadgroup_size_matches_tile_size`), so its
/// key is in the function's.
fn tile_fields(d: &TileRenderPipelineDescriptor) -> String {
    let (formats, samples) = (d.color_attachment_formats, d.raster_sample_count);
    let fields = match d.max_total_threads_per_threadgroup {
        None => format!("{:?}", (formats, samples)),
        Some(limit) => format!("{:?}", (formats, samples, limit)),
    };
    // Appended only past Metal's default, so a default descriptor keeps its key.
    match d.max_call_stack_depth {
        0 | 1 => fields,
        k => format!("{fields} max_call_stack_depth {k}"),
    }
}

/// The kernel of artifact `a` fetches each `[[stage_in]]` attribute as `d`'s stage-input
/// descriptor describes it: a `Compiled` kernel compiled under other descriptor entries is
/// `InvalidDescriptor(StageInputAttribute)`, and one reading a stage input without a descriptor
/// `NeedsInput(StageInputDescriptor)` (a `Function` kernel is compiled under `d`'s descriptor, so
/// its fetches are `d`'s).
fn check_stage_in(a: &FunctionArtifact, d: &ComputePipelineDescriptor) -> Result<(), Error> {
    for s in &a.interface.io.stage_in {
        let Some(desc) = d.stage_input.as_ref() else {
            return Err(Error::NeedsInput(RuntimeInput::StageInputDescriptor {
                attribute: s.index,
            }));
        };
        if StageInFetch::of(desc, s.index)? != s.fetch {
            return Err(Error::InvalidDescriptor(
                DescriptorError::StageInputAttribute { attribute: s.index },
            ));
        }
    }
    Ok(())
}

/// The stage of artifact `a` has the call stack its pipeline's `maxCallStackDepth` for it,
/// `given`, sizes: a `Compiled` stage whose stack was sized for another depth is
/// `InvalidDescriptor(MaxCallStackDepth)` (a `Function` stage is compiled under `given`).
fn check_call_stack(a: &FunctionArtifact, given: u32) -> Result<(), Error> {
    for fact in &a.facts {
        if let RuntimeFact::CallStackDepth { depth } = *fact {
            if depth != given.max(1) {
                return Err(Error::InvalidDescriptor(
                    DescriptorError::MaxCallStackDepth {
                        compiled: depth,
                        given,
                    },
                ));
            }
        }
    }
    Ok(())
}

/// The compute descriptor fields the artifact depends on (its Metal info).
fn compute_fields(d: &ComputePipelineDescriptor) -> String {
    let fields = format!(
        "{:?}",
        (
            d.stage_input.as_ref(),
            d.max_total_threads_per_threadgroup,
            d.threadgroup_size_is_multiple_of_execution_width,
            d.support_indirect_command_buffers,
        )
    );
    // Appended only past Metal's default, so a default descriptor keeps its key.
    match d.max_call_stack_depth {
        0 | 1 => fields,
        k => format!("{fields} max_call_stack_depth {k}"),
    }
}

/// The pipeline key's fields of variant `v` past its stages' keys: the depth it was compiled for
/// (`VariantKey::depth16_pass`, `VariantKey::depth16_biased`, `VariantKey::depth16_bias_raises`),
/// which a pipeline without a fragment function keeps in no stage's key, and which every pipeline
/// states (`RuntimeFact::Depth16Pass`, `RuntimeFact::Depth16Biased`,
/// `RuntimeFact::Depth16BiasSide`), and its color group past the first (`VariantKey::color_group`),
/// whose variant of a pipeline without a fragment function has no stage of its own. Empty for
/// every other variant, so a pipeline keeps its key.
fn variant_fields(v: &VariantKey) -> String {
    let mut fields = match v.depth16_pass {
        Some(depth16) => format!(" depth16_pass {depth16}"),
        None => String::new(),
    };
    if v.depth16_biased {
        fields.push_str(" depth16_biased");
    }
    if let Some(raises) = v.depth16_bias_raises {
        fields.push_str(&format!(" depth16_bias_raises {raises}"));
    }
    // The group a pipeline without a fragment function keeps in no stage's key.
    if v.color_group != 0 {
        fields.push_str(&format!(" color_group {}", v.color_group));
    }
    // The synthesized vertex stage of an emulated mesh function's pipeline, in no stage's key.
    if v.mesh_last_first {
        fields.push_str(" mesh_last_first");
    }
    fields
}

/// The render descriptor fields the artifact depends on: all of its fixed-function state (the
/// artifact carries the vertex input and graphics state), not only what changes SPIR-V.
fn render_fields(d: &RenderPipelineDescriptor) -> String {
    let fields = format!(
        "{:?}",
        (
            RenderLinkInputs::from(d),
            d.support_indirect_command_buffers
        )
    );
    // Appended only past Metal's defaults, so a default descriptor keeps its key.
    let fields = match (
        d.max_vertex_call_stack_depth,
        d.max_fragment_call_stack_depth,
    ) {
        (0 | 1, 0 | 1) => fields,
        (v, f) => format!("{fields} max_call_stack_depths {v} {f}"),
    };
    // The private state the artifact depends on, likewise only past Metal's defaults.
    fields + &crate::pipeline::private_state::render_key_fields(d)
}

/// Stage `i` of a linked pipeline as its final module.
fn stage_module(a: &FunctionArtifact, linked: &Linked, i: usize) -> Result<StageModule, Error> {
    let words = relocate(a, linked)?;
    let specialization = linked
        .specialization
        .get(i)
        .filter(|(stage, _)| *stage == a.interface.stage)
        .map(|(_, s)| s.clone())
        .ok_or_else(|| invariant("link produced no specialization layout for a stage"))?;
    let mut stage_flags = vk::PipelineShaderStageCreateFlags::empty();
    if a.subgroup.full_subgroups {
        stage_flags |= vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS;
    }
    // A fragment function that selects an attachment's rules rounds it, and keeps the notes of
    // the rules, as the pipeline's attachment selects them.
    let rules = linked
        .graphics
        .as_ref()
        .map_or(&[][..], |g| &g.color_write_rules[..]);
    // A `half` output of a module without fast math keeps its note where its attachment blends.
    let colors = linked
        .graphics
        .as_ref()
        .map_or(&[][..], |g| &g.color_attachments[..]);
    // Alpha to coverage reads the alpha the function rounds.
    let coverage = linked
        .graphics
        .as_ref()
        .is_some_and(|g| g.alpha_to_coverage);
    let Some((precision_notes, rounded_colors)) =
        crate::pipeline::color_writes::pipeline_writes(a, rules, colors, coverage)
    else {
        return Err(invariant(
            "link gave no rule word to a selected color attachment",
        ));
    };
    Ok(StageModule {
        stage: a.interface.vk_stage(),
        entry_point: ENTRY_NAME.to_owned(),
        spirv: words.into(),
        specialization,
        required_subgroup_size: a.subgroup.size,
        stage_flags,
        reconvergence: a.reconvergence,
        precision_notes,
        strategies: a.strategies.clone(),
        // Each stage binds its own blob at its `Synthetic(ConstantBlob)` slot.
        constant_blob: a.blob.clone(),
        rounded_colors,
        specialized_notes: a.specialized.clone(),
        half_bit_colors: a.interface.io.half_bit_colors.clone(),
        own_pixel_textures: a.interface.own_pixel_textures.clone(),
        snapshot_textures: a.interface.snapshot_textures.clone(),
        private_bytes: a.interface.private_bytes,
    })
}

/// The user varyings `a` writes when it is a vertex function, or a mesh function's per-vertex
/// and per-primitive ones (`LoweringInputs::absent_varyings`).
fn varyings_written(a: &FunctionArtifact) -> std::collections::BTreeSet<VaryingKey> {
    a.interface
        .io
        .outputs
        .iter()
        .filter(|o| match a.interface.stage {
            Stage::Vertex => matches!(o.role, IoRole::Varying { .. }),
            Stage::Mesh => matches!(
                o.role,
                IoRole::Varying { .. } | IoRole::PrimitiveVarying { .. }
            ),
            _ => false,
        })
        .map(|o| o.key.clone())
        .collect()
}

/// The user varyings `a` writes the same at every vertex of a primitive when it is a vertex
/// function (`StageIo::uniform_outputs`), which its fragment stage reads flat where it
/// interpolates them (`LoweringInputs::uniform_varyings`).
fn uniform_written(a: &FunctionArtifact) -> std::collections::BTreeSet<VaryingKey> {
    match a.interface.stage {
        Stage::Vertex => a.interface.io.uniform_outputs.iter().cloned().collect(),
        _ => std::collections::BTreeSet::new(),
    }
}

/// The indices `a` writes when it is a vertex or mesh function, each of which its fragment stage
/// reads from the raw varying `a` writes it to: it declares that varying's role among its outputs
/// (`IoRole::ViewportIndex`, `IoRole::RenderTargetArrayIndex`), in every variant; and the
/// primitive ID (`IoRole::PrimitiveId`).
fn raw_indices_written(a: &FunctionArtifact) -> RawIndices {
    let writes = |role: IoRole| {
        matches!(a.interface.stage, Stage::Vertex | Stage::Mesh)
            && a.interface.io.outputs.iter().any(|o| o.role == role)
    };
    RawIndices {
        viewport: writes(IoRole::ViewportIndex),
        layer: writes(IoRole::RenderTargetArrayIndex),
        primitive_id: writes(IoRole::PrimitiveId),
    }
}

/// The varyings mesh function `a` writes per primitive (`LoweringInputs::primitive_varyings`): its
/// primitive data and its raw render target slice and viewport indices.
fn primitive_varyings_written(a: &FunctionArtifact) -> std::collections::BTreeSet<VaryingKey> {
    a.interface
        .io
        .outputs
        .iter()
        .filter(|o| {
            matches!(
                o.role,
                IoRole::PrimitiveVarying { .. }
                    | IoRole::ViewportIndex
                    | IoRole::RenderTargetArrayIndex
            )
        })
        .map(|o| o.key.clone())
        .collect()
}

/// The mesh descriptor fields the artifact depends on: its fixed-function state, as a render
/// pipeline's ([`render_fields`]).
fn mesh_fields(d: &MeshPipelineDescriptor) -> String {
    format!("{:?}", RenderLinkInputs::from(d))
}

/// A fragment function of a pipeline that rasterizes more than one sample, which reads a value
/// Vulkan evaluates at an implementation-defined position within the fragment, or at the
/// invocation's sample under sample shading, and Metal at the pixel center, notes
/// `PrecisionNote::MultisamplePosition`: `[[position]]`'s z or w, a center varying not read at the
/// pixel (`StageIo::varyings_at_pixel`), a centroid varying, `[[point_coord]]`, or
/// `[[barycentric_coord]]`, at pixel rate or per sample. Link refused a per-sample function's
/// center varying where the device has no pixel form, so a center varying reaches here without it
/// only at pixel rate.
fn note_sample_position(
    d: FragmentPass<'_>,
    stages: &[&FunctionArtifact],
    modules: &mut [StageModule],
) {
    use crate::pipeline::stage_io::{Interpolation, Sampling};
    if d.raster_sample_count <= 1 || !d.rasterization_enabled {
        return;
    }
    for (a, m) in stages.iter().zip(modules) {
        if a.interface.stage != Stage::Fragment {
            continue;
        }
        let io = &a.interface.io;
        let interpolated = |at: Sampling| {
            io.inputs.iter().any(|v| {
                matches!(
                    v.role,
                    IoRole::Varying {
                        interpolation: Some(Interpolation::Smooth { sampling, .. }),
                        ..
                    } if sampling == at
                )
            })
        };
        let reads = (io.builtins.contains(&StageBuiltin::FragCoord) && io.position_depth)
            || (interpolated(Sampling::Center) && !io.varyings_at_pixel)
            || interpolated(Sampling::Centroid)
            || io
                .builtins
                .iter()
                .any(|b| matches!(b, StageBuiltin::PointCoord | StageBuiltin::BaryCoord { .. }));
        if reads
            && !m
                .precision_notes
                .contains(&PrecisionNote::MultisamplePosition)
        {
            m.precision_notes.push(PrecisionNote::MultisamplePosition);
            m.precision_notes.sort_unstable();
        }
        // Unconditional on the stage: no specialization removes it.
        m.specialized_notes
            .retain(|s| s.note != PrecisionNote::MultisamplePosition);
    }
}

/// Every runtime fact of the stages, once, with the mechanism this profile honors it by.
fn dependencies(stages: &[&FunctionArtifact], dev: &Device) -> Vec<RuntimeDependency> {
    let mut out: Vec<RuntimeDependency> = Vec::new();
    for a in stages {
        for fact in &a.facts {
            if !out.iter().any(|d| d.fact == *fact) {
                out.push(RuntimeDependency {
                    fact: *fact,
                    mechanism: mechanism(fact, dev.profile()),
                });
            }
        }
    }
    out
}
