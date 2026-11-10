//! What a **tile render pipeline's** reflection says of its arguments, under every
//! `MTLPipelineOption` value.
//!
//! The suite made tile pipelines before this family — [`crate::cases::tile_threadgroups`] makes
//! them and reads the state's own getters — and asked none of them for a reflection. So
//! `newRenderPipelineStateWithTileDescriptor:options:reflection:error:` was called with a nil
//! out-parameter and `MTLPipelineOptionNone` everywhere in the tree, and the whole of
//! `MTLRenderPipelineReflection.tileBindings` was compared against Metal nowhere.
//!
//! That omission cost two things:
//!
//! - a tile function's arguments are the one place a **tile** stage's bindings exist, and nothing
//!   held our answer for them to Metal's;
//! - `MTLPipelineOption`'s eight values were swept for the compute creation
//!   ([`crate::cases::pipeline_option_reflection`]) and for neither render creation, so the claim
//!   that the tile creation reads the same option word the compute one does rested on the
//!   runtime's own documentation.
//!
//! # The sweep: `MTLPipelineOption`, 8 of 8
//!
//! [`crate::pipeline_option_reflection::MASKS`] is the whole of Apple's option word — the
//! three bits its header names, in all eight combinations — and each case of this family runs the
//! tile creation under every one of them. It is the same table
//! [`crate::cases::pipeline_option_reflection`] sweeps for the compute creation, read rather than
//! copied, so the two families cannot drift apart.
//!
//! **Why no value of it is vacuous.** `MTLPipelineOptionNone` answers **no** reflection and every
//! mask holding `1 << 0` or `1 << 1` answers one, so the eight sites of one case disagree with
//! each other at their reflection site: a side that ignored its options and always reflected, or
//! never did, fails at least one site. Measured on an M3 Max under Metal 27.0.1, the tile
//! creation's reflection under `BindingInfo` holds **three** bindings — the implicit imageblock
//! (`MTLBindingTypeImageblock`, index 0, read-only), the threadgroup memory
//! (`MTLBindingTypeThreadgroupMemory`, read-write) and the buffer (`MTLBindingTypeBuffer`,
//! alignment 4, data size 4, `MTLDataTypeFloat` behind a pointer type) — so a site that answered
//! a reflection is comparing a tree and not a bool.
//!
//! The masks holding `1 << 2` are the second claim under test: no case here lists a binary
//! archive, so each such mask must answer exactly what the same mask without that bit answers. A
//! creation refused instead would be a refusal of something plain Metal does, and the site's
//! `refused` leaf says so.
//!
//! # All five stage lists, every site
//!
//! Each site records `vertexBindings`, `fragmentBindings`, `tileBindings`, `objectBindings` and
//! `meshBindings` ([`StageBindings`](crate::function_bindings::StageBindings)), not the tile list
//! alone. Four of the five are empty for a tile pipeline on Metal, and comparing an empty list
//! against an empty list proves nothing by itself — but the five together say **which stage Metal
//! attributes each argument to**, which is a fact with five possible answers and exactly the fact
//! a side that filed a tile function's arguments under `fragmentBindings` would get wrong. No
//! other site in a reflection would see that.
//!
//! # What this family does not reach, and why
//!
//! The sixteen packed pixel `MTLDataType` values (`MTLDataTypeR8Unorm` through
//! `MTLDataTypeRGB9E5Float`) are **not** here, and a tile reflection is not the way to them. They
//! are statable in MSL only as imageblock members, and Metal's imageblock binding — measured on
//! an M3 Max under Metal 27.0.1, from both a tile pipeline's `tileBindings` and a compute
//! pipeline's `bindings` — is an `MTLImageBlockArgument` that conforms to **none** of
//! `MTLBinding`'s public subprotocols (`MTLBufferBinding`, `MTLThreadgroupBinding`,
//! `MTLTextureBinding`, `MTLObjectPayloadBinding`), so no public getter reaches a struct type for
//! it. Its member list sits behind `structType` and `imageBlockMasterStructMembers` on the
//! private `MTLBindingInternal` superclass, which is not a surface a conformance comparison
//! between two implementations can rest on. The sixteen values stay unreached by Metal's public
//! API, not by this harness's choice.
//!
//! # Why every case of this family is safe
//!
//! Each case makes one library and eight tile pipeline states. Nothing is allocated for the
//! device, no render pass is encoded, no command buffer is made and nothing is dispatched, so no
//! case here can fault the device, trip its watchdog or reach a queue. Every options value is one
//! Apple's header names, so no creation asks Metal what an undefined bit means.
//!
//! # Exactness
//!
//! Every field is a number, an enumeration, a name or a bool, and the family declares no
//! tolerance. An enumeration is recorded as its Metal enumerator name beside its ordinal, so a
//! difference reads `MTLBindingTypeImageblock` against `MTLBindingTypeThreadgroupMemory` rather
//! than 17 against 1.

use crate::case::{
    Abort, Case, Failure, Input, MetalSource, Side, TilePipelineSpec, METAL_TILE_THREADS,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::pipeline_option_reflection::MASKS;
use crate::recording::ShaderRef;
use crate::transcript::{Digest, Node, Observed, Sites};
use metal2vulkan::metal::PixelFormat;

/// The site the library's creation sits at.
const LIBRARY: &str = "library";

/// The one colour attachment the tile function's imageblock declares (`float4 c [[color(0)]]`).
const ATTACHMENTS: [PixelFormat; 1] = [PixelFormat::RGBA8Unorm];

/// A case over one tile pipeline's reflection, swept across every `MTLPipelineOption` value.
pub struct TilePipelineReflectionCase {
    name: &'static str,
    /// The fixture holding the tile function.
    fixture: &'static str,
    /// The tile function, by its name in the library.
    entry: &'static str,
    /// Whether the descriptor sets `threadgroupSizeMatchesTileSize`.
    tile_sized: bool,
}

/// Every case of the family.
pub fn all() -> Vec<TilePipelineReflectionCase> {
    vec![
        // The reflection lane's own tile function: an implicit imageblock, threadgroup memory and
        // a buffer, which between them reach all three kinds of binding a tile stage can hold —
        // an imageblock, which conforms to no subprotocol; threadgroup memory, which carries
        // `MTLThreadgroupBinding`'s size and alignment; and a buffer, which carries
        // `MTLBufferBinding`'s pointer type. `threadgroupSizeMatchesTileSize` is set, which is
        // what a tile function holding an implicit imageblock is dispatched under.
        TilePipelineReflectionCase {
            name: "tile-pipeline-reflection-imageblock",
            fixture: "reflection/tile",
            entry: "reflection_tile",
            tile_sized: true,
        },
    ]
}

impl TilePipelineReflectionCase {
    fn loaded(&self) -> Fixture {
        Fixture {
            name: self.fixture,
            entry: self.entry,
        }
    }

    /// The descriptor the case states.
    fn spec(&self) -> TilePipelineSpec<'static> {
        let spec = TilePipelineSpec::new(self.entry, &ATTACHMENTS, METAL_TILE_THREADS);
        match self.tile_sized {
            true => spec.tile_sized(),
            false => spec,
        }
    }
}

impl Case for TilePipelineReflectionCase {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let fixture = self.loaded();
        [RUNTIME_TEXT, METAL_TEXT]
            .into_iter()
            .map(|extension| {
                Ok(ShaderRef {
                    fixture: format!("{}{extension}", self.fixture),
                    entry: self.entry.to_owned(),
                    source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
                })
            })
            .collect()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = self.loaded();
        // Every mask the case sweeps, so a value added to or taken out of the sweep is a change to
        // the case's inputs rather than a recording that silently covers less.
        let mut masks = Vec::new();
        for (bits, name) in MASKS {
            masks.extend(bits.to_le_bytes());
            masks.extend(name.as_bytes());
        }
        Ok(vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("descriptor", &self.spec().words()),
            Input::bytes("options", &masks),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let stop = |at: &str, why: String| Abort {
            at: at.to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        };
        let loaded = self.loaded().loaded().map_err(|why| stop(LIBRARY, why))?;
        let library = crate::case::made(sites, LIBRARY, side.library(&loaded.source()))?;
        let spec = self.spec();
        // Every mask is asked for, whatever an earlier one answered, for the reason
        // [`crate::cases::pipeline_option_reflection`] states: the sweep is the measurement, and
        // stopping at the first value a side cannot take would leave the rest unmeasured.
        let mut stopped = None;
        for (bits, name) in MASKS {
            let at = format!("options-{name}");
            match side.tile_pipeline_bindings(library, &spec, bits) {
                Ok(found) => sites.push(found.node(at)),
                Err(failure) => {
                    sites.push(Node::leaf(at.clone(), Observed::value(failure.recorded())));
                    stopped = stopped.or(Some(Abort { at, failure }));
                }
            }
        }
        match stopped {
            Some(abort) => Err(abort),
            None => Ok(()),
        }
    }
}
