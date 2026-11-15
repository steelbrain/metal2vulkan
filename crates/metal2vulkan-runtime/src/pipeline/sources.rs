//! Where a pipeline state's compiles look before compiling: the device's memo, then the creation's
//! binary archives in order, then the device's shader cache on disk. What a compile makes is kept
//! in the memo and on disk; what an archive or the disk gives, in the memo. A compute pipeline is
//! linked from its function where neither the memo nor an archive holds it, and kept in the memo
//! alone ([`Sources::kernel`]). A creation for a binary archive's add also notes every artifact it
//! used.

use super::memo::Remembered;
use super::render::RenderVariant;
use super::{Kernel, PipelineOptions, Variant};
use crate::binary_archive::{Artifact, BinaryArchive, Used};
use crate::Device;
use metal2vulkan::cache::PipelineKey;
use metal2vulkan::library::{CompiledFunction, FunctionCompileKey};
use metal2vulkan::pipeline::{
    ComputePipeline, FunctionDescriptor, LoweringInputs, RenderPipeline, StageEntries,
};
use metal2vulkan::Budget;
use std::sync::Arc;

/// The translator's census name of a kernel emulated in place of the pin of its device's widest
/// mode (`WorkCounts::widest_pin_kernels_emulated`, and the pipeline creation log's compute lines,
/// `creation_log::compute_kernel`).
pub(super) const WIDEST_PIN_STRATEGY: &str = "simd.emulated.widest-pin";

/// The render pipeline artifacts this process encoded for the directory, which
/// `tests::render_stage_digests`' count reads: a sweep whose stage functions the directory holds
/// must encode none. Carried only where that device test is compiled, since nothing else reads
/// it.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
static ARTIFACTS_ENCODED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// [`ARTIFACTS_ENCODED`].
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) fn render_artifacts_encoded() -> u64 {
    ARTIFACTS_ENCODED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Count a render pipeline artifact encoded for the directory.
fn note_render_artifact_encoded() {
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    ARTIFACTS_ENCODED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Where one creation's (or one later variant's) compiles look, and its budget.
pub(crate) struct Sources<'a> {
    device: &'a Device,
    /// The creation's options: its archives and budget; `None` for a variant a dispatch or
    /// draw needs after creation, which no archive holds and which compiles under the standard
    /// budget.
    options: Option<&'a PipelineOptions>,
    /// What an add's creation used.
    used: Option<&'a Used>,
}

impl<'a> Sources<'a> {
    /// The sources of a creation under `options`, noting what it uses in `used`.
    pub(crate) fn creation(
        device: &'a Device,
        options: &'a PipelineOptions,
        used: Option<&'a Used>,
    ) -> Sources<'a> {
        Sources {
            device,
            options: Some(options),
            used,
        }
    }

    /// The sources of a variant a dispatch or draw needs after creation.
    pub(crate) fn later(device: &'a Device) -> Sources<'a> {
        Sources {
            device,
            options: None,
            used: None,
        }
    }

    pub(crate) fn device(&self) -> &'a Device {
        self.device
    }

    /// The budget of one compile attempt starting now.
    pub(crate) fn budget(&self) -> Budget {
        match self.options {
            Some(options) => options.budget(),
            None => Budget::standard(),
        }
    }

    fn archives(&self) -> &'a [BinaryArchive] {
        self.options
            .map_or(&[], |options| options.binary_archives.as_slice())
    }

    fn note(&self, artifact: Artifact) {
        if let Some(used) = self.used {
            used.borrow_mut().push(artifact);
        }
    }

    /// `function` compiled under `inputs`, whose key is `key`: found, else compiled under
    /// `budget`, reading its specialized entry from `entries` where given (the device's, which a
    /// render creation's checks and key filled, and which a compute creation's kernel reads too).
    /// A refusal is the compile layer's.
    ///
    /// Every ask for a stage's function the runtime makes -- a compute kernel's, a render
    /// variant's vertex and fragment -- comes here, so the two counts this bumps are the whole of
    /// the runtime's lowering: one that translated (`WorkCounts::functions_translated`) against
    /// one answered by the memo, an archive or the disk (`functions_shared`). Counted at the
    /// translation itself and not at the memo's miss, so a function an archive or the shader cache
    /// gives reads as spared, which is what it is.
    pub(crate) fn function(
        &self,
        function: &FunctionDescriptor,
        inputs: &LoweringInputs,
        (key, entries): (&FunctionCompileKey, Option<&StageEntries>),
        budget: &Budget,
    ) -> Result<CompiledFunction, metal2vulkan::error::Error> {
        let compile = self.device.compile_device();
        let shared = self.device.shared();
        let cache = &shared.shader_cache;
        let compiled = match self.found_function(key) {
            Some(compiled) => compiled,
            None => {
                let compiled = match entries {
                    Some(entries) => {
                        compile.compile_function_from_entries(function, inputs, budget, entries)?
                    }
                    None => compile.compile_function(function, inputs, budget)?,
                };
                shared.counts.function_translated();
                if compiled
                    .strategies()
                    .entries
                    .contains_key(WIDEST_PIN_STRATEGY)
                {
                    shared.counts.widest_pin_kernel_emulated();
                }
                let bytes = cache.store_function(compile, &compiled);
                cache.memo.keep_function(&compiled, bytes);
                self.note(Artifact::Function(compiled.clone()));
                compiled
            }
        };
        Ok(compiled)
    }

    /// The function of `key`, found by its key alone: the memo's, an archive's or the disk's, as
    /// [`function`](Self::function) finds it, counted as a function shared and kept in the memo;
    /// `None` where none holds it, translating nothing. A render creation that holds the keys a
    /// variant's record names links what this finds (`ShaderCache::variant_pipeline`).
    pub(crate) fn found_function(&self, key: &FunctionCompileKey) -> Option<CompiledFunction> {
        let compile = self.device.compile_device();
        let shared = self.device.shared();
        let cache = &shared.shader_cache;
        let compiled = match cache.memo.function(key) {
            Some(compiled) => compiled,
            None => {
                let (compiled, bytes) = self
                    .archives()
                    .iter()
                    .find_map(|archive| archive.function(compile, key))
                    .or_else(|| cache.load_function(compile, key))?;
                cache.memo.keep_function(&compiled, bytes);
                compiled
            }
        };
        shared.counts.function_shared();
        self.note(Artifact::Function(compiled.clone()));
        Some(compiled)
    }

    /// The compute pipeline of `descriptor`, a compute or tile descriptor whose kernel or tile
    /// function is compiled: the variant of it the memo knows live, or the pipeline one of the
    /// creation's archives holds, else the pipeline linked now from that function under `budget`
    /// (`compile_compute_pipeline` or `compile_tile_pipeline`, each under its own key). A refusal
    /// is the compile layer's.
    ///
    /// **Linked, never read from the directory.** A compute pipeline is its one function
    /// relocated, with the layout and interface its link derives from that function: a link from
    /// the function in hand translates nothing. So the directory keeps no compute pipeline, and
    /// a creation that found its function links its pipeline rather than reading, checking and
    /// decoding the pipeline's own artifact, which cost more than the link (measured by
    /// `tests::compute_create_bench`'s parts: on lavapipe, a 16,348-word kernel's artifact read
    /// and decoded in about 25 us, the link 6 us), and a creation that translated writes one
    /// artifact, its function's, not two. The link relocates against the function's relocation
    /// slot, so a sweep of states over one function relocates its module once. The memo keeps a
    /// linked pipeline weighed by what it retains ([`ComputePipeline::footprint`]), an archive's
    /// at its encoded length.
    pub(crate) fn kernel(
        &self,
        descriptor: &Kernel,
        budget: &Budget,
    ) -> Result<Remembered<ComputePipeline, Variant>, metal2vulkan::error::Error> {
        let compile = self.device.compile_device();
        let cache = &self.device.shared().shader_cache;
        let key = match descriptor {
            Kernel::Compute(descriptor) => compile.compute_pipeline_key(descriptor),
            Kernel::Tile(descriptor) => compile.tile_pipeline_key(descriptor),
        };
        let found = match cache.memo.compute(&key) {
            Some(found) => found,
            None => {
                let found = self
                    .archives()
                    .iter()
                    .find_map(|archive| archive.compute_pipeline(compile, &key));
                let (pipeline, weight) = match found {
                    Some((pipeline, bytes)) => (Arc::new(pipeline), bytes),
                    None => {
                        let pipeline = Arc::new(match descriptor {
                            Kernel::Compute(descriptor) => {
                                compile.compile_compute_pipeline(descriptor, budget)?
                            }
                            Kernel::Tile(descriptor) => {
                                compile.compile_tile_pipeline(descriptor, budget)?
                            }
                        });
                        let weight = pipeline.footprint();
                        (pipeline, weight)
                    }
                };
                cache.memo.keep_compute(&pipeline, weight);
                Remembered::Compiled(pipeline)
            }
        };
        self.note(Artifact::Compute(match &found {
            Remembered::Live(variant) => Arc::clone(&variant.pipeline),
            Remembered::Compiled(pipeline) => Arc::clone(pipeline),
        }));
        Ok(found)
    }

    /// The render pipeline of `key`, found: the variant of it the memo knows live, or the
    /// pipeline an archive or the disk cache holds.
    pub(crate) fn render(
        &self,
        key: &PipelineKey,
    ) -> Option<Remembered<RenderPipeline, RenderVariant>> {
        let compile = self.device.compile_device();
        let cache = &self.device.shared().shader_cache;
        let found = match cache.memo.render(key) {
            Some(found) => found,
            None => {
                let (pipeline, bytes) = self
                    .archives()
                    .iter()
                    .find_map(|archive| archive.render_pipeline(compile, key))
                    .or_else(|| cache.load_render_pipeline(compile, key))?;
                let pipeline = Arc::new(pipeline);
                cache.memo.keep_render(&pipeline, bytes);
                Remembered::Compiled(pipeline)
            }
        };
        self.note(Artifact::Render(match &found {
            Remembered::Live(variant) => Arc::clone(variant.shared_pipeline()),
            Remembered::Compiled(pipeline) => Arc::clone(pipeline),
        }));
        Some(found)
    }

    /// Keep `pipeline`, linked now from the stage functions of `stage_keys`, in the memo, and on
    /// disk where the directory does not hold those functions.
    ///
    /// **The artifact is stored only where re-linking it would translate.** A later process that
    /// does not find the pipeline looks its stages' functions up by these same keys, which the
    /// creation's variant record names (`ShaderCache::record_variant_pipeline`), and links what
    /// it finds ([`found_function`](Self::found_function), then `compile_render_variant`), so
    /// where the directory holds every one of them the pipeline's own entry spares a link and
    /// nothing more, while writing it costs this creation an encoding of every word of every
    /// stage -- the largest single piece of work a creation of a sweep over one function pair
    /// does, and one whose result is byte-identical to the previous sibling's but for its
    /// fixed-function state. A creation that translated or linked something the directory has no function for
    /// stores the pipeline as every creation did: there the entry is what spares the
    /// translation. So does one that names no stage key at all, which is no evidence about the
    /// directory and must not read as every key being in it.
    ///
    /// The memo keeps it either way, weighed by what it retains rather than by what an encoding
    /// would have been ([`RenderPipeline::footprint`]), so the memo's byte bound holds whichever
    /// way the artifact was kept. Weighing a skipped entry at the zero bytes a skipped encoding
    /// returns would let the memo hold its 4096 entries believing its bound untouched.
    ///
    /// The question about a stage's function names that stage's compile key and the device's
    /// profile, and nothing of the pipeline state, so the device asks the directory once per
    /// function however many states link over it (`ShaderCache::holds_function`,
    /// `shader_cache/held.rs`).
    pub(crate) fn linked_render(
        &self,
        pipeline: RenderPipeline,
        stage_keys: &[FunctionCompileKey],
    ) -> Arc<RenderPipeline> {
        let compile = self.device.compile_device();
        let cache = &self.device.shared().shader_cache;
        let pipeline = Arc::new(pipeline);
        let functions_held = !stage_keys.is_empty()
            && stage_keys
                .iter()
                .all(|key| cache.holds_function(compile, key));
        let encoded = match functions_held {
            true => None,
            false => cache.store_render_pipeline(compile, &pipeline),
        };
        // Weighed at its encoding's length where the directory took it, else at what it retains.
        let weight = match encoded {
            Some(length) => {
                note_render_artifact_encoded();
                length
            }
            None => pipeline.footprint(),
        };
        cache.memo.keep_render(&pipeline, weight);
        self.note(Artifact::Render(Arc::clone(&pipeline)));
        pipeline
    }
}
