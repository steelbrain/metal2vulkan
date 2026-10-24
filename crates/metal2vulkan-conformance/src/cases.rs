//! The suite's case set.
//!
//! Two seed families proved the loop end to end, chosen to be as unlike each other as the first
//! increment allowed: one compares the bytes a compute pass wrote, the other a typed answer a
//! getter gives.
//!
//! Most of the rest are ports of the runtime lanes whose expectation is already a transcript of a
//! Metal probe's printed lines, which is what made them the first batch. Where Metal's whole path
//! is public, record mode **runs** the probe rather than copying it: the mesh draws, the mesh draws
//! behind a barrier, and the mesh draws of an indirect command buffer's own-state commands. Where
//! it is not, the case carries our own measurement and forbids re-recording: the texture fills,
//! whose `fillTexture:` is Apple SPI, and an indirect command's read-back through private getters.
//! The reason is always the *call*, never the pixel format a case names.
//!
//! [`rate_maps`] is the family where Metal's own API validation layer and plain Metal disagree,
//! and it is what exercises the recording's second oracle ([`crate::validated`]).
//!
//! Two families are ports of lanes that held **no Metal value at all** — a rule, or a list in a
//! module's prose — and whose values this round measured on the oracle for the first time:
//! [`texture_accesses`], where Metal's render-pipeline reflection states each texture slot's
//! access, and [`textures`], where every uncompressed colour format's texture answers what a fresh
//! level 0 reads, what a host round trip gives back, and whether the bytes between the rows stayed
//! as they were.
//!
//! No case here can fault a GPU, hang it, trip its watchdog or end Metal's process.
//!
//! # A coverage gap, named here because no case can see it
//!
//! **A pipeline's declared depth and stencil formats are not tested by anything.** Metal requires
//! `MTLRenderPipelineDescriptor` to declare them before a pipeline may test or write either, while
//! our runtime derives the attachment from the render pass descriptor at encode time and does not
//! need the declaration. The two sides therefore disagreed on those two fields for as long as they
//! existed -- the Metal side set them, our side set neither -- and **no case's verdict moved when
//! that was fixed**: `visibility`'s depth rows and `shader_stencil`'s stencil rows answer the same
//! with the formats forwarded and without, and [`srgb_fetch`]'s depth row answers exactly what its
//! three-draw row answers.
//!
//! So those families cover less than their names suggest: none of them distinguishes a pipeline
//! with depth from one without. The case that would is a **refusal** case -- state a format that
//! disagrees with the pass's attachment, or omit it where the pass has one, and compare our typed
//! error against Metal's, with Metal's own API validation beside it where it also refuses. Nobody
//! has written it. Until somebody does, those two fields are held by construction alone, by
//! `tests::the_runtime_side_forwards_every_field_of_a_pipeline_spec`.

pub mod argument_encoder_crossings;
pub mod argument_encoders;
pub mod astc_textures;
pub mod attribute_formats;
pub mod attribute_strides;
pub mod binary_archives;
pub mod blend_interactions;
pub mod blend_sweeps;
pub mod blit_encoder_commands;
pub mod blit_indirect_commands;
pub mod border_swizzles;
pub mod buffer_blits;
pub mod buffer_depth_samples;
pub mod buffer_target_clears;
pub mod buffer_texture_blits;
pub mod buffer_textures;
pub mod chroma_pairs;
pub mod color_write_masks;
pub mod command_buffer_facts;
pub mod command_queues;
pub mod compressed_blits;
pub mod compressed_decode;
pub mod compressed_moves;
pub mod compressed_views;
pub mod compute_encoder_commands;
pub mod compute_heap_residency;
pub mod compute_pipeline_descriptors;
pub mod compute_render_chains;
pub mod compute_threadgroups;
pub mod denormal_word_fetches;
pub mod depth_resolve_pairs;
pub mod depth_stencil_space;
pub mod depth_texel_reads;
pub mod device_queries;
pub mod discard_hoist;
pub mod draw_state_changes;
pub mod dynamic_libraries;
pub mod etc_textures;
pub mod event_orders;
pub mod extended_range_blends;
pub mod extended_range_planes;
pub mod extended_range_targets;
pub mod extended_range_two_planes;
pub mod extended_swizzles;
pub mod fetch_ordered;
pub mod format_copies;
pub mod format_footprints;
pub mod fragment_imageblock_reflection;
pub mod frame_chains;
pub mod fresh_contents;
pub mod function_bindings;
pub mod function_facts;
pub mod half_output_blends;
pub mod half_read_nans;
pub mod half_storage_writes;
pub mod half_texel_nans;
pub mod half_write_nans;
pub mod handler_waits;
pub mod heap_facts;
pub mod heap_metal_sizes;
pub mod host_detach;
pub mod host_page_writes;
pub mod indirect_commands;
pub mod indirect_private_types;
pub mod int_attributes;
pub mod level_sizes;
pub mod library_facts;
pub mod math_routines;
pub mod mesh_barriers;
pub mod mesh_commands;
pub mod mesh_draws;
pub mod mesh_pipeline_descriptors;
pub mod mesh_pipeline_reflection;
pub mod nearest_positions;
pub mod norm_storage_writes;
pub mod pass_bytes;
pub mod pipeline_descriptors;
pub mod pipeline_facts;
pub mod pipeline_functions;
pub mod pipeline_handlers;
pub mod pipeline_option_reflection;
pub mod pipeline_options;
pub mod pixel_downscales;
pub mod pixel_samplers;
pub mod pvrtc_cpu_order;
pub mod pvrtc_rows;
pub mod rate_map_sweeps;
pub mod rate_maps;
pub mod render_chains;
pub mod render_depth_stencil_state;
pub mod render_draw_forms;
pub mod render_pass_descriptors;
pub mod render_residency;
pub mod render_state;
pub mod resource_facts;
pub mod rgb10a2_srgb;
pub mod rotated_planes;
pub mod rotated_targets;
pub mod sample_fills;
pub mod sample_positions;
pub mod sampler_crossings;
pub mod sampler_filtering;
pub mod scheduled_handlers;
pub mod shader_stencil;
pub mod shared_events;
pub mod shared_exponent_storage;
pub mod simdgroup_lockstep;
pub mod snorm_tile_store;
pub mod sparse_heaps;
pub mod sparse_mappings;
pub mod sparse_tiles;
pub mod srgb_fetch;
pub mod stage_in_regions;
pub mod surface_fills;
pub mod swizzled_storage;
pub mod tessellation_sweeps;
pub mod texel_swizzles;
pub mod texture_accesses;
pub mod texture_buffer_fills;
pub mod texture_fills;
pub mod texture_views;
pub mod textures;
pub mod tile_pipeline_descriptors;
pub mod tile_pipeline_reflection;
pub mod tile_threadgroups;
pub mod two_plane_copies;
pub mod two_plane_heap_blits;
pub mod two_plane_samples;
pub mod vertex_formats;
pub mod visibility;
pub mod visible_tables;
pub mod wide_int_attributes;
pub mod wide_points;
pub mod ycbcr_minify;
pub mod ycbcr_textures;

use crate::case::Case;

/// Every case of the suite, in the order the report lists them.
pub fn all() -> Vec<Box<dyn Case>> {
    let mut cases: Vec<Box<dyn Case>> = Vec::new();
    for case in pass_bytes::all() {
        cases.push(Box::new(case));
    }
    cases.push(Box::new(indirect_private_types::private_type_getters()));
    cases.extend(indirect_commands::boxed());
    for case in math_routines::all() {
        cases.push(Box::new(case));
    }
    cases.push(Box::new(stage_in_regions::stage_in_region_clamps()));
    for case in compressed_decode::all() {
        cases.push(Box::new(case));
    }
    for case in etc_textures::all() {
        cases.push(Box::new(case));
    }
    for case in astc_textures::all() {
        cases.push(Box::new(case));
    }
    for case in argument_encoders::all() {
        cases.push(Box::new(case));
    }
    for case in argument_encoder_crossings::all() {
        cases.push(Box::new(case));
    }
    for case in function_bindings::all() {
        cases.push(Box::new(case));
    }
    for case in function_facts::all() {
        cases.push(Box::new(case));
    }
    for case in library_facts::all() {
        cases.push(Box::new(case));
    }
    for case in pipeline_facts::all() {
        cases.push(Box::new(case));
    }
    for case in pipeline_options::all() {
        cases.push(Box::new(case));
    }
    for case in visible_tables::all() {
        cases.push(Box::new(case));
    }
    for case in pipeline_functions::all() {
        cases.push(Box::new(case));
    }
    for case in pipeline_descriptors::all() {
        cases.push(Box::new(case));
    }
    for case in pipeline_handlers::all() {
        cases.push(Box::new(case));
    }
    for case in pipeline_option_reflection::all() {
        cases.push(Box::new(case));
    }
    for case in binary_archives::all() {
        cases.push(Box::new(case));
    }
    for case in dynamic_libraries::all() {
        cases.push(Box::new(case));
    }
    for case in compute_pipeline_descriptors::all() {
        cases.push(Box::new(case));
    }
    for case in tile_pipeline_descriptors::all() {
        cases.push(Box::new(case));
    }
    for case in mesh_pipeline_descriptors::all() {
        cases.push(Box::new(case));
    }
    for case in tile_pipeline_reflection::all() {
        cases.push(Box::new(case));
    }
    for case in fragment_imageblock_reflection::all() {
        cases.push(Box::new(case));
    }
    for case in mesh_pipeline_reflection::all() {
        cases.push(Box::new(case));
    }
    for case in render_pass_descriptors::all() {
        cases.push(Box::new(case));
    }
    for case in compressed_blits::all() {
        cases.push(Box::new(case));
    }
    for case in compressed_moves::all() {
        cases.push(Box::new(case));
    }
    for case in compressed_views::all() {
        cases.push(Box::new(case));
    }
    for case in pvrtc_rows::all() {
        cases.push(Box::new(case));
    }
    for case in pvrtc_cpu_order::all() {
        cases.push(Box::new(case));
    }
    for case in level_sizes::all() {
        cases.push(Box::new(case));
    }
    for case in border_swizzles::all() {
        cases.push(Box::new(case));
    }
    for case in format_copies::all() {
        cases.push(Box::new(case));
    }
    for case in nearest_positions::all() {
        cases.push(Box::new(case));
    }
    for case in wide_points::all() {
        cases.push(Box::new(case));
    }
    for case in discard_hoist::all() {
        cases.push(Box::new(case));
    }
    for case in int_attributes::all() {
        cases.push(Box::new(case));
    }
    for case in wide_int_attributes::all() {
        cases.push(Box::new(case));
    }
    for case in vertex_formats::all() {
        cases.push(Box::new(case));
    }
    cases.extend(tessellation_sweeps::boxed());
    for case in attribute_formats::all() {
        cases.push(Box::new(case));
    }
    cases.push(Box::new(attribute_formats::ClassCrossings));
    for case in extended_range_planes::all() {
        cases.push(Box::new(case));
    }
    for case in depth_texel_reads::all() {
        cases.push(Box::new(case));
    }
    for case in extended_range_two_planes::all() {
        cases.push(Box::new(case));
    }
    for case in two_plane_samples::all() {
        cases.push(Box::new(case));
    }
    for case in rgb10a2_srgb::all() {
        cases.push(Box::new(case));
    }
    for case in pixel_samplers::all() {
        cases.push(Box::new(case));
    }
    for case in pixel_downscales::all() {
        cases.push(Box::new(case));
    }
    for case in half_write_nans::all() {
        cases.push(Box::new(case));
    }
    for case in half_read_nans::all() {
        cases.push(Box::new(case));
    }
    for case in half_texel_nans::all() {
        cases.push(Box::new(case));
    }
    cases.push(Box::new(denormal_word_fetches::all().remove(0)));
    for case in half_storage_writes::all() {
        cases.push(Box::new(case));
    }
    for case in half_output_blends::all() {
        cases.push(Box::new(case));
    }
    cases.extend(extended_range_targets::all());
    for case in extended_range_blends::all() {
        cases.push(Box::new(case));
    }
    cases.push(Box::new(texture_buffer_fills::streams()));
    cases.push(Box::new(texture_buffer_fills::groups()));
    cases.push(Box::new(texture_buffer_fills::long_fills()));
    cases.push(Box::new(two_plane_heap_blits::heap_blits()));
    cases.push(Box::new(two_plane_copies::row_blits()));
    cases.push(Box::new(two_plane_copies::announced_row_blits()));
    cases.push(Box::new(two_plane_copies::pad_fills()));
    for case in texture_fills::all() {
        cases.push(Box::new(case));
    }
    for case in texture_fills::heap_fill_runs() {
        cases.push(Box::new(case));
    }
    for case in sample_fills::all() {
        cases.push(Box::new(case));
    }
    for case in mesh_draws::all() {
        cases.push(Box::new(case));
    }
    for case in mesh_barriers::all() {
        cases.push(Box::new(case));
    }
    cases.push(Box::new(mesh_commands::own_state_commands()));
    cases.push(Box::new(mesh_commands::unordered_draws()));
    cases.push(Box::new(surface_fills::over_planes_and_buffers()));
    for case in rate_maps::all() {
        cases.push(Box::new(case));
    }
    for case in rate_map_sweeps::all() {
        cases.push(Box::new(case));
    }
    for case in rate_map_sweeps::parameter_data() {
        cases.push(Box::new(case));
    }
    for case in tile_threadgroups::all() {
        cases.push(Box::new(case));
    }
    for case in format_footprints::all() {
        cases.push(Box::new(case));
    }
    for case in heap_metal_sizes::all() {
        cases.push(Box::new(case));
    }
    for case in heap_facts::all() {
        cases.push(Box::new(case));
    }
    for case in resource_facts::all() {
        cases.push(Box::new(case));
    }
    for case in texture_views::all() {
        cases.push(Box::new(case));
    }
    cases.push(Box::new(command_queues::queue_slots()));
    for case in device_queries::all() {
        cases.push(Box::new(case));
    }
    for case in texture_accesses::all() {
        cases.push(Box::new(case));
    }
    for case in snorm_tile_store::all() {
        cases.push(Box::new(case));
    }
    for case in color_write_masks::all() {
        cases.push(Box::new(case));
    }
    for case in blend_sweeps::all() {
        cases.push(Box::new(case));
    }
    for case in blend_interactions::all() {
        cases.push(Box::new(case));
    }
    for case in visibility::all() {
        cases.push(Box::new(case));
    }
    for case in textures::all() {
        cases.push(Box::new(case));
    }
    for case in handler_waits::all() {
        cases.push(Box::new(case));
    }
    cases.push(Box::new(shared_events::rising_value()));
    cases.push(Box::new(shared_events::handle_round_trip()));
    cases.extend(
        event_orders::all()
            .into_iter()
            .map(|case| Box::new(case) as Box<dyn Case>),
    );
    cases.push(Box::new(scheduled_handlers::handler_order()));
    cases.extend(chroma_pairs::all());
    cases.extend(ycbcr_textures::all());
    for case in ycbcr_minify::all() {
        cases.push(Box::new(case));
    }
    cases.extend(compute_threadgroups::all());
    cases.extend(buffer_blits::all());
    cases.extend(blit_encoder_commands::all());
    cases.extend(compute_encoder_commands::all());
    cases.extend(attribute_strides::all());
    cases.extend(blit_indirect_commands::all());
    cases.push(Box::new(compute_heap_residency::heap_residency()));
    cases.extend(buffer_texture_blits::all());
    cases.extend(buffer_textures::all());
    cases.extend(buffer_target_clears::all());
    cases.push(Box::new(host_detach::detach_in_a_completed_handler()));
    cases.extend(host_page_writes::all());
    cases.extend(buffer_depth_samples::all());
    cases.extend(simdgroup_lockstep::all());
    for case in sparse_tiles::all() {
        cases.push(Box::new(case));
    }
    for case in sparse_heaps::all() {
        cases.push(Box::new(case));
    }
    cases.extend(sparse_mappings::all());
    cases.extend(command_buffer_facts::all());
    cases.extend(fresh_contents::all());
    cases.extend(sample_positions::all());
    cases.extend(
        fetch_ordered::all()
            .into_iter()
            .map(|case| Box::new(case) as Box<dyn Case>),
    );
    cases.extend(
        srgb_fetch::all()
            .into_iter()
            .map(|case| Box::new(case) as Box<dyn Case>),
    );
    cases.extend(
        shader_stencil::all()
            .into_iter()
            .map(|case| Box::new(case) as Box<dyn Case>),
    );

    for case in extended_swizzles::all() {
        cases.push(Box::new(case));
    }
    for case in shared_exponent_storage::all() {
        cases.push(Box::new(case));
    }
    cases.extend(norm_storage_writes::NormStorageWrites::all());
    cases.extend(texel_swizzles::all());
    for case in swizzled_storage::all() {
        cases.push(Box::new(case));
    }
    cases.extend(rotated_targets::all());
    cases.extend(rotated_planes::all());
    for case in sampler_filtering::all() {
        cases.push(Box::new(case));
    }
    cases.extend(sampler_crossings::boxed());
    cases.extend(render_chains::boxed());
    cases.extend(compute_render_chains::boxed());
    cases.extend(draw_state_changes::boxed());
    cases.extend(render_state::all());
    cases.extend(render_draw_forms::all());
    cases.extend(render_depth_stencil_state::all());
    cases.extend(depth_stencil_space::all());
    cases.extend(depth_resolve_pairs::all());
    cases.extend(render_residency::all());
    cases.extend(frame_chains::boxed());
    cases
}

/// The case named `name`, where the suite holds one.
pub fn named(name: &str) -> Option<Box<dyn Case>> {
    all().into_iter().find(|case| case.name() == name)
}
