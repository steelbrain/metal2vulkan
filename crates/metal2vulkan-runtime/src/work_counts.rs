//! Counts of work a device's commit path does that the lanes hold to bounds: each counts a piece of
//! work the runtime once repeated at every commit or every change, so a lane fails when it is
//! repeated again; and the textures that took their storage form, which a lane holds to once each;
//! and the buffers made with the device-address usage, which every encoder that reads pointers made
//! resident one by one on MoltenVK; and the completion waits that missed their wake, which a test
//! holds to none. Not part of the API.

use super::buffer::{KeptBecause, ReadApart};
use super::command_buffer::GroupEnd;
use super::degradation::Degradation;
use super::device::lock;
use super::memory::FreeCause;
use super::recorder::CommandsRecorded;
use super::scheduler::HeldFor;
use super::shadow::{Backing, CarryTally, CopyCause, Lost, ShadowCopy, ShadowSource};
use super::Device;
use ash::vk;
use std::panic::Location;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// Which of the two predictions a `VkPipeline` a state's creation made belongs to
/// (`WorkCounts::creation_render_pipelines_made`, `creation_dispatch_pipelines_made`): the render
/// state's predicted key (`pipeline/predicted.rs`) or the compute state's predicted region
/// (`pipeline/dispatch_making.rs`). The two are counted apart because their hit rates differ by
/// orders of magnitude, and the whole count cannot tell them apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Predicted {
    /// A render state's predicted key.
    Render,
    /// A compute state's predicted region.
    Dispatch,
}

/// A device's counters (`DeviceShared::counts`), each only ever added to.
#[derive(Default)]
pub(crate) struct Counters {
    /// Released objects and shadow keeps by kind (`churn_counts.rs`).
    pub(crate) churn: super::churn_counts::ChurnCounters,
    command_pools_made: AtomicU64,
    spare_command_pools_taken: AtomicU64,
    spare_command_pools_reset: AtomicU64,
    live_tables_uploaded: AtomicU64,
    timeline_waits: AtomicU64,
    fence_waits: AtomicU64,
    missed_completion_wakes: AtomicU64,
    completion_arms: AtomicU64,
    completion_arms_after_a_wake: AtomicU64,
    completion_reads_ran_ahead: AtomicU64,
    gate_waits: AtomicU64,
    gate_waits_slept: AtomicU64,
    gate_wakes_found_shut: AtomicU64,
    trail_waits: AtomicU64,
    trail_waits_blocked: AtomicU64,
    completed_where_seen: AtomicU64,
    scheduled_where_submitted: AtomicU64,
    timestamp_slots_taken: AtomicU64,
    timestamp_slots_reset_on_host: AtomicU64,
    calibrations: AtomicU64,
    calibration_readings: AtomicU64,
    calibration_deviation_most: AtomicU64,
    calibration_disagreements: AtomicU64,
    storage_forms_taken: AtomicU64,
    addressable_buffers_made: AtomicU64,
    mipmaps_filtered: AtomicU64,
    mipmap_first_levels_fetched: AtomicU64,
    pipelines_pooled: AtomicU64,
    residual_form_draws: AtomicU64,
    residual_form_makings: AtomicU64,
    stage_entries_specialized: AtomicU64,
    stage_entries_shared: AtomicU64,
    functions_translated: AtomicU64,
    functions_shared: AtomicU64,
    linked_use_questions_asked: AtomicU64,
    descriptor_keys_built: AtomicU64,
    shader_modules_made: AtomicU64,
    shader_modules_shared: AtomicU64,
    pipeline_layouts_made: AtomicU64,
    pipeline_layouts_shared: AtomicU64,
    variant_pipelines_made: AtomicU64,
    flights_followed: AtomicU64,
    predicated_loop_runs_again: AtomicU64,
    variants_taken_back: AtomicU64,
    unlisted_argument_sampler_buffers: AtomicU64,
    draws_dropped_unmade: AtomicU64,
    draws_dropped_still_making: AtomicU64,
    dispatches_dropped_still_making: AtomicU64,
    host_step_sampler_sets_made: AtomicU64,
    creation_pipelines_made: AtomicU64,
    creations_left_making: AtomicU64,
    creation_pipelines_drawn: AtomicU64,
    creation_render_pipelines_made: AtomicU64,
    creation_render_pipelines_drawn: AtomicU64,
    creation_dispatch_pipelines_made: AtomicU64,
    creation_dispatch_pipelines_drawn: AtomicU64,
    sampler_sets_made: AtomicU64,
    sampler_sets_shared: AtomicU64,
    sampler_sets_destroyed: AtomicU64,
    ahead_render_passes_made: AtomicU64,
    render_passes_made: AtomicU64,
    ahead_framebuffers_made: AtomicU64,
    kept_backings_made: AtomicU64,
    pooled_backings_made: AtomicU64,
    storage_placeholders_made: AtomicU64,
    storage_placeholder_attachments: AtomicU64,
    color_resolve_passes_made: AtomicU64,
    color_resolve_passes_kept: AtomicU64,
    ahead_nanos: AtomicU64,
    driver_diagnostics_withheld: AtomicU64,
    stand_ins_made: AtomicU64,
    stand_in_bytes: AtomicU64,
    stale_detached_reaches: AtomicU64,
    detached_storage_reaches: AtomicU64,
    heap_slots_rewritten: AtomicU64,
    heap_slot_writes_deferred: AtomicU64,
    heap_textures_tiled: AtomicU64,
    heap_sample_move_images_made: AtomicU64,
    heap_sample_move_images_lent: AtomicU64,
    requirement_images_made: AtomicU64,
    requirement_limits_asked: AtomicU64,
    arena_pools_made: AtomicU64,
    heap_texture_lossless_layouts: AtomicU64,
    heap_texture_block_compressed_layouts: AtomicU64,
    heap_texture_untiled_layouts: AtomicU64,
    fills_recorded: AtomicU64,
    fills_by_transfers: AtomicU64,
    ycbcr_refreshes: AtomicU64,
    ycbcr_whole_conversions: AtomicU64,
    ycbcr_textures_in_place: AtomicU64,
    ycbcr_images_kept: [AtomicU64; 7],
    ycbcr_planes_bound: AtomicU64,
    ycbcr_planes_images: AtomicU64,
    ycbcr_planes_images_kept: [AtomicU64; 3],
    ycbcr_planes_images_bound: AtomicU64,
    ycbcr_device_conversions: AtomicU64,
    ycbcr_device_samples: AtomicU64,
    attachment_clears_recorded: AtomicU64,
    image_blits_recorded: AtomicU64,
    query_result_copies_recorded: AtomicU64,
    visibility_queries_begun: AtomicU64,
    indirect_draws_recorded: AtomicU64,
    indirect_records_read: AtomicU64,
    gpu_form_commands: AtomicU64,
    merged_draw_commands: AtomicU64,
    gpu_pool_blocks_made: AtomicU64,
    gpu_pool_blocks_released: AtomicU64,
    gpu_pool_leases_peak: AtomicU64,
    internal_submissions: AtomicU64,
    queue_submissions: AtomicU64,
    submit_infos: AtomicU64,
    command_buffers_submitted: AtomicU64,
    batch_submit_nanos: AtomicU64,
    submit_stalls_on_work: AtomicU64,
    submits_behind_idle_stalls: AtomicU64,
    wait_behind_idle_stall_longest_nanos: AtomicU64,
    command_buffers_begun: AtomicU64,
    encoder_command_buffers_begun: AtomicU64,
    command_buffers_begun_by: [AtomicU64; Begun::COUNT],
    render_encoders_ended: AtomicU64,
    render_passes_begun_by: [AtomicU64; PassBegun::COUNT],
    instance_splits_by: [AtomicU64; Split::COUNT],
    encoder_primaries_begun_ahead: AtomicU64,
    primaries_begun_ahead: AtomicU64,
    barrier_buffers_kept: AtomicU64,
    stamped_buffers_kept: AtomicU64,
    stamped_buffers_reused: AtomicU64,
    finales_folded: AtomicU64,
    finales_apart: AtomicU64,
    grouped_submissions: AtomicU64,
    batches_grouped: AtomicU64,
    /// By `GroupEnd`, in its order.
    group_ends: [AtomicU64; 7],
    /// Batches submitted, by how many earlier batches of their `VkQueue` were pending then: none,
    /// one, two or three, four or more.
    batches_by_pending: [AtomicU64; 4],
    groups_kept_pending: AtomicU64,
    groups_sent_for_waits: AtomicU64,
    /// By `HeldFor`, in its order: the looks that found a line's head held for each reason.
    heads_held_for: [AtomicU64; HeldFor::COUNT],
    heads_waiting_at_their_gate: AtomicU64,
    heads_awaiting_check_signals: AtomicU64,
    batches_held_past_a_look: AtomicU64,
    batch_hold_nanos: AtomicU64,
    scheduler_passes_by_waiters: AtomicU64,
    batches_submitted_by_waiters: AtomicU64,
    command_buffer_wakes: AtomicU64,
    command_buffer_wakes_spared: AtomicU64,
    event_wakes: AtomicU64,
    event_wakes_spared: AtomicU64,
    slot_returns: AtomicU64,
    slot_return_nanos: AtomicU64,
    slot_waits: AtomicU64,
    queues_sharing_a_vk_queue: AtomicU64,
    slot_wait_nanos: AtomicU64,
    completed_waits: AtomicU64,
    event_waits_met: AtomicU64,
    shared_event_waits_held: AtomicU64,
    event_signals_behind_merge: AtomicU64,
    bare_starts: AtomicU64,
    prefixed_starts: AtomicU64,
    command_barriers: AtomicU64,
    unrun_instances_ended: AtomicU64,
    untouched_passes_skipped: AtomicU64,
    inline_passes: AtomicU64,
    inline_passes_split: AtomicU64,
    inline_split_instances: AtomicU64,
    inline_memoryless_write_splits: AtomicU64,
    passes_not_inline: [AtomicU64; NotInline::COUNT],
    passes_not_inline_actions_by: [AtomicU64; ActionsRefused::COUNT],
    commits_handed_off: AtomicU64,
    images_started_on_the_host: AtomicU64,
    host_texel_copies: AtomicU64,
    texel_shadows_made: AtomicU64,
    texel_shadow_bytes: AtomicU64,
    texel_shadow_reads: AtomicU64,
    texel_shadows_dropped: AtomicU64,
    staging_submissions: AtomicU64,
    uncached_host_images: AtomicU64,
    cpu_block_decodes: AtomicU64,
    block_decode_passes: AtomicU64,
    chroma_device_twins: AtomicU64,
    chroma_device_decodes: AtomicU64,
    second_twins: AtomicU64,
    second_twin_checks: AtomicU64,
    second_twin_decodes: AtomicU64,
    queue_drains: AtomicU64,
    stand_in_copies_made: AtomicU64,
    stand_in_copies_made_at_draws: AtomicU64,
    draw_shapes_made: AtomicU64,
    draw_shapes_found: AtomicU64,
    argument_samplers_taken: AtomicU64,
    bind_views_worked_out: AtomicU64,
    stage_tables_read: AtomicU64,
    stage_tables_read_in_part: AtomicU64,
    no_copy_lies_read: AtomicU64,
    commands_noted_as_before: AtomicU64,
    draws_settled: AtomicU64,
    two_plane_draws: AtomicU64,
    provoking_last_pipeline_draws: AtomicU64,
    provoking_listed_draws: AtomicU64,
    gathered_draws: AtomicU64,
    bounded_run_draws: AtomicU64,
    bound_notices_made: AtomicU64,
    argument_tables_retained: AtomicU64,
    argument_objects_retained: AtomicU64,
    graphics_pipelines_looked_up: AtomicU64,
    linear_passes_in_place: AtomicU64,
    linear_passes_through_copies: AtomicU64,
    kept_bytes_passes: AtomicU64,
    color_group_passes: AtomicU64,
    color_group_twins: AtomicU64,
    color_group_draws_skipped: AtomicU64,
    draw_states_recorded: AtomicU64,
    draws_pushing_set_zero: AtomicU64,
    set_zero_writes_pushed: AtomicU64,
    set_zero_equal_shape_switches: AtomicU64,
    set_zero_writes_on_equal_shape_switches: AtomicU64,
    set_zero_writes_if_shapes_shared: AtomicU64,
    dispatches_pushing_set_zero: AtomicU64,
    set_zero_sets_reused: AtomicU64,
    set_zero_sets_found: AtomicU64,
    draw_shapes_reused: AtomicU64,
    dispatches_settled_as_before: AtomicU64,
    dispatches_planned_as_before: AtomicU64,
    baked_sampler_dispatches_apart: AtomicU64,
    host_deallocators_delayed: AtomicU64,
    host_import_releases_posted: AtomicU64,
    host_import_releases_pending: AtomicU64,
    host_keep_alives_kept: AtomicU64,
    host_keep_alives_covered: AtomicU64,
    host_mappings_queried: AtomicU64,
    dma_buf_imports: AtomicU64,
    dma_buf_memories_live: AtomicU64,
    dma_buf_image_memories_live: AtomicU64,
    host_imports_released: AtomicU64,
    host_imports_reimported: AtomicU64,
    host_imports_resident: AtomicU64,
    host_imports_lost: AtomicU64,
    host_import_fds_kept: AtomicU64,
    host_imports_kept_for_fds: AtomicU64,
    host_imports_used: AtomicU64,
    host_imports_read_apart: AtomicU64,
    host_imports_read_apart_by: [AtomicU64; ReadApart::COUNT],
    host_imports_kept_by: [AtomicU64; KeptBecause::COUNT],
    host_imports_rebound_unlisted: AtomicU64,
    host_rebind_imports_made: AtomicU64,
    host_rebind_imports_covered: AtomicU64,
    host_remaps_reported: AtomicU64,
    host_remaps_over_imports: AtomicU64,
    host_imports_remapped: AtomicU64,
    host_pointer_imports_made: AtomicU64,
    host_pointer_import_bytes_made: AtomicU64,
    host_imports_refused_by_host: AtomicU64,
    copied_host_buffers_made: AtomicU64,
    copied_host_buffers_refused: AtomicU64,
    copied_host_buffers_degraded: AtomicU64,
    copied_rebind_pages_moved: AtomicU64,
    copied_rebind_pages_made: AtomicU64,
    copied_rebind_images_made: AtomicU64,
    copied_heaps_made: AtomicU64,
    copied_copy_in_bytes: AtomicU64,
    copied_merge_back_bytes: AtomicU64,
    copied_dropped_bytes: AtomicU64,
    host_ram_chunks_imported: AtomicU64,
    host_ram_chunk_bytes: AtomicU64,
    host_ram_chunks_live: AtomicU64,
    chunk_bound_buffers_made: AtomicU64,
    dma_buf_sparse_buffers: AtomicU64,
    dma_buf_images: AtomicU64,
    dma_buf_image_fallbacks: AtomicU64,
    modifier_feature_queries: AtomicU64,
    shadowed_textures: AtomicU64,
    shadows_zeroed: AtomicU64,
    shadows_over_fresh_pages: AtomicU64,
    host_textures_in_place: AtomicU64,
    device_copy_textures: AtomicU64,
    dma_buf_textures_in_place: AtomicU64,
    fresh_page_textures_in_place: AtomicU64,
    pitched_host_images: AtomicU64,
    host_texture_shadows_exact: AtomicU64,
    host_texture_shadows_render_target: AtomicU64,
    host_texture_shadows_shader_write: AtomicU64,
    host_texture_shadows_check_pending: AtomicU64,
    host_texture_shadows_check_failed: AtomicU64,
    host_texture_shadows_unchecked: AtomicU64,
    host_texture_shadows_refused: [AtomicU64; 7],
    in_place_checks_run: AtomicU64,
    in_place_checks_passed: AtomicU64,
    in_place_checks_failed: AtomicU64,
    in_place_checks_unchecked: AtomicU64,
    in_place_checks_recalled: AtomicU64,
    fresh_page_buffers_made: AtomicU64,
    fresh_page_bytes: AtomicU64,
    new_buffer_bytes_cleared: AtomicU64,
    new_buffer_zero_fills: AtomicU64,
    shadow_copies: AtomicU64,
    shadow_texels_copied_in: AtomicU64,
    shadow_texels_first_copied_in: AtomicU64,
    shadow_texels_first_copied_in_images: AtomicU64,
    shadow_texels_first_copied_in_private_buffers: AtomicU64,
    shadow_texels_first_copied_in_mapped_buffers: AtomicU64,
    shadow_texels_first_copied_in_host_pages: AtomicU64,
    shadow_texels_first_copied_in_aliased_host_pages: AtomicU64,
    shadow_texels_first_copied_in_surfaces: AtomicU64,
    shadow_texels_first_copied_in_large_surfaces: AtomicU64,
    shadow_texels_first_copy_in_carried: AtomicU64,
    shadow_texels_first_copy_in_carried_large_surfaces: AtomicU64,
    shadow_texels_first_copied_in_large_surfaces_stood_earlier: AtomicU64,
    shadow_texels_first_copied_in_large_surfaces_unstood: AtomicU64,
    shadow_texels_first_copied_in_again: AtomicU64,
    shadow_texels_copied_in_images: AtomicU64,
    shadow_texels_recopied_beyond_held: AtomicU64,
    shadow_texels_recopied_after_changes: AtomicU64,
    shadow_texels_recopied_after_other_shadows: AtomicU64,
    shadow_texels_recopied_after_two_ways_merges: AtomicU64,
    shadow_texels_recopied_after_writes: AtomicU64,
    shadow_texels_recopied_after_writes_within: AtomicU64,
    shadow_texels_recopied_after_dropped_writes: AtomicU64,
    shadow_texels_bounded_copied_in: AtomicU64,
    shadow_texels_recopied_in_images: AtomicU64,
    shadow_texels_recopied_in_private_buffers: AtomicU64,
    shadow_texels_recopied_in_mapped_buffers: AtomicU64,
    shadow_texels_recopied_in_host_pages: AtomicU64,
    shadow_texels_recopied_in_aliased_host_pages: AtomicU64,
    shadow_texels_recopied_in_surfaces: AtomicU64,
    shadow_texels_recopied_in_large_surfaces: AtomicU64,
    shadow_texels_copied_back: AtomicU64,
    shadow_copies_back_dispatched: AtomicU64,
    shadow_copies_in_dispatched: AtomicU64,
    shadow_keeps_from_staging: AtomicU64,
    shadow_merge_copies_dispatched: AtomicU64,
    shadow_merge_dispatches_planned: AtomicU64,
    shadow_merge_words_examined: AtomicU64,
    shadow_merge_dispatches_in_blocks: AtomicU64,
    shadow_merge_scratches_reused: AtomicU64,
    shadow_merge_scratches_zeroed: AtomicU64,
    row_copies: AtomicU64,
    row_copy_regions: AtomicU64,
    row_copy_regions_most: AtomicU64,
    row_copies_dispatched: AtomicU64,
    shadow_merges_fetched: AtomicU64,
    shadow_merges_kept_in_images: AtomicU64,
    shadow_merge_kept_images_made: AtomicU64,
    shadow_merge_kept_images_followed: AtomicU64,
    shadow_merge_kept_images_live: AtomicU64,
    shadow_attachment_passes: AtomicU64,
    shadow_attachment_textures: AtomicU64,
    shadow_attachment_loads_within: AtomicU64,
    shadow_attachment_loads_whole: AtomicU64,
    shadow_attachment_clears: AtomicU64,
    shadow_attachment_clears_in_part: AtomicU64,
    shadow_attachment_loads_untouched: AtomicU64,
    shadow_attachment_tiles_covered: AtomicU64,
    shadow_attachment_tiles_bounded: AtomicU64,
    shadows_sharing_truth: AtomicU64,
    owned_descriptor_pools: AtomicU64,
    owned_kernels_made: AtomicU64,
    owned_kernel_spirv_bytes: AtomicU64,
    ycbcr_conversion_scratch_buffers: AtomicU64,
    ycbcr_conversion_scratch_bytes: AtomicU64,
    shadow_texels_kept: AtomicU64,
    bounded_shadow_copies: AtomicU64,
    views_through_swizzle: AtomicU64,
    view_shadows: AtomicU64,
    attachment_reads_through_copies: AtomicU64,
    attachment_read_splits: AtomicU64,
    reading_draws_snapshotted: AtomicU64,
    snapshot_inline_splits: AtomicU64,
    feedback_loop_alias_images: AtomicU64,
    depth_attachments_moved: AtomicU64,
    depth_instances_held: AtomicU64,
    held_depths_released: AtomicU64,
    own_pixel_reads: AtomicU64,
    replay_logs_kept: AtomicU64,
    replay_log_bytes: AtomicU64,
    encoders_reencoded: AtomicU64,
    segments_recorded_again: AtomicU64,
    replay_tables_pointed: AtomicU64,
    reencodes_refused: AtomicU64,
    pipeline_libraries_made: AtomicU64,
    fragment_shader_libraries_made: AtomicU64,
    pipelines_linked: AtomicU64,
    dispatch_pipelines_pooled: AtomicU64,
    tile_pipelines_pooled: AtomicU64,
    upload_chunks_in_device_memory: AtomicU64,
    tile_dispatches_recorded: AtomicU64,
    render_log_batches_by_thread: AtomicU64,
    render_log_batches_by_program: AtomicU64,
    render_log_waits: AtomicU64,
    render_log_ends_by_thread: AtomicU64,
    render_log_ends_by_program: AtomicU64,
    render_log_commits_by_thread: AtomicU64,
    render_log_commits_by_program: AtomicU64,
}

/// How a render pass took a shadowed texture as a color attachment
/// (`WorkCounts::shadow_attachment_loads_within` and kin).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ColorTaken {
    /// Loaded, its copies cut to the rectangle bounding the draws' scissors.
    LoadedWithin,
    /// Loaded (or neither loaded nor cleared), its every texel copied in.
    Loaded,
    /// Cleared over the whole texture: no copy in.
    Cleared,
    /// Cleared over part of the texture: its every texel copied in.
    ClearedInPart,
    /// Loaded in a pass that reaches no tile: no copy in or back.
    Untouched,
}

/// What a `VkCommandBuffer` the runtime began was for (`WorkCounts::command_buffers_begun` and
/// its parts, `command_buffers_begun_encoders` and kin).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Begun {
    /// An encoder's own, as it opened (or begun ahead in its spare pool, as it took it).
    Encoder,
    /// A segment's at the commit, an untracked segment's, or an ended encoder's encoded again.
    Segment,
    /// One an encoder went on in after a segment or a pass's end (`EncoderCore::continuation`).
    Continuation,
    /// A render encoder's draws' secondary.
    Secondary,
    /// A batch's work of its own as it is submitted: a sparse binding's mappings, the first work.
    BatchWork,
    /// A batch's barriers (and transitions and timestamps) as it is submitted.
    BatchBarriers,
    /// A barrier command buffer recorded to keep (`kept_barriers.rs`).
    KeptBarriers,
    /// The device's own submissions' (`device/submit.rs`) and staging submissions'
    /// (`device/staging.rs`).
    Internal,
}

impl Begun {
    /// How many there are.
    pub(crate) const COUNT: usize = 8;
}

/// Where the runtime began a render pass instance of a render encoder's pass
/// (`WorkCounts::render_passes_begun_instances` and kin); every other `vkCmdBeginRenderPass` it
/// records (clears, resolves and copies of its own) is among `commands_recorded`'s alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PassBegun {
    /// An instance whose draws a secondary holds, begun where the pass is recorded
    /// (`render/pass_steps.rs`): an encoder's, or a run of an execution's commands.
    Instance,
    /// A pass's first instance, begun at its first draw to record its draws inline
    /// (`render/inline_first.rs`).
    Inline,
    /// An instance after a split, its draws kept and recorded inline (`render/inline_splits.rs`).
    InlineSplit,
    /// An instance of no draws that resolves the pass, after its last
    /// (`render/pass_steps.rs`).
    Resolve,
}

impl PassBegun {
    /// How many there are.
    pub(crate) const COUNT: usize = 4;
}

/// Why a render encoder's pass went on in a new render pass instance after one that recorded,
/// splitting the pass (`render.rs`, **Instances**; `WorkCounts::instance_splits_barrier` and
/// kin): the first that holds, in this order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Split {
    /// A memory barrier of the program's since the instance recorded (`fence.rs`).
    Barrier,
    /// A tile dispatch since the instance recorded (`render/tile_dispatch.rs`), or the instance
    /// of no draws after the pass's last one, for its store actions.
    TileDispatch,
    /// A draw ordered after draws of the instance that reached the tile threadgroup memory
    /// (`render/tile_memory.rs`).
    TileMemory,
    /// A draw that samples a color attachment's texture after a barrier inside the instance
    /// (`texture/attached_reads.rs`).
    AttachedRead,
    /// A draw that reads through snapshots apart from the instance, or binds a color the instance
    /// holds in the feedback loop layout (`render/snapshot.rs`).
    Feedback,
    /// A draw that writes a memoryless color after an instance recorded inline
    /// (`render/inline_first.rs`).
    MemorylessWrite,
    /// A draw whose pipeline is being made after an instance recorded inline
    /// (`render/inline_first.rs`).
    PendingPipeline,
    /// The commands of an indirect command buffer's execution, or a run of them
    /// (`render/executions.rs`).
    Execution,
    /// A visibility result mode set in a pass of several views (`visibility.rs`).
    Visibility,
    /// A draw the instance's shape does not take: the attachments it fetches as storage, the
    /// views it renders, stand-ins or sample planes arriving, its shader-side stencil or depth
    /// test, its colors' places or formats.
    Shape,
}

impl Split {
    /// How many there are.
    pub(crate) const COUNT: usize = 10;
}

/// What kept a render pass's first instance from recording inline (`render/inline_first.rs`): the
/// first check it failed, in this order. Counted (`WorkCounts::passes_not_inline_encoders` and
/// kin), so a run names what sends its passes' draws to secondary command buffers.
#[derive(Clone, Copy, Debug)]
pub(crate) enum NotInline {
    /// The encoder encodes an ended encoder's calls again, or the pass has several color groups.
    Encoders,
    /// The pass's layout: no color attachment, or more than one layer or sample, a depth or
    /// stencil attachment, a visibility buffer, custom sample positions, a rasterization-rate map,
    /// or extended-range shadows that keep its draws.
    Layouts,
    /// A memoryless color attachment not loaded and stored `DontCare`, or resolved.
    Memoryless,
    /// A color attachment's actions, the first ruled out ([`ActionsRefused`]).
    Actions(ActionsRefused),
    /// A color attachment that is a texture over a buffer the pass does not render through an
    /// image over the buffer's bytes.
    BufferTextures,
    /// Another color attachment kind: a stand-in, planes, a clear made apart, an extended-range
    /// format, or a bound linear image.
    Attachments,
    /// The pass's layout allowed it, and its first draw that records reads or writes a color
    /// attachment as storage (a fetch in the pixel interlock leaves it out of the subpass).
    Storage,
    /// The pass's layout allowed it, and a stand-in arrives with its first draw that records (a
    /// fetched attachment's scratch or backing).
    StandIn,
    /// The pass's layout allowed it, and its first draw that records writes a memoryless color
    /// attachment, which no instance recorded inline writes.
    MemorylessWrites,
    /// The pass's layout allowed it, and its first draw that records did not otherwise.
    Draws,
}

impl NotInline {
    /// The kinds.
    const COUNT: usize = 10;
    /// [`Actions`](Self::Actions)' index.
    const ACTIONS: usize = 3;

    /// Its kind's index among the [`COUNT`](Self::COUNT), in the order of the checks.
    fn index(self) -> usize {
        match self {
            NotInline::Encoders => 0,
            NotInline::Layouts => 1,
            NotInline::Memoryless => 2,
            NotInline::Actions(_) => Self::ACTIONS,
            NotInline::BufferTextures => 4,
            NotInline::Attachments => 5,
            NotInline::Storage => 6,
            NotInline::StandIn => 7,
            NotInline::MemorylessWrites => 8,
            NotInline::Draws => 9,
        }
    }
}

/// The color attachment action that kept a render pass's first instance from recording inline
/// ([`NotInline::Actions`]): the first ruled out, color by color in index order, and of one color
/// in this order. Counted (`WorkCounts::passes_not_inline_actions_dont_care_load` and kin).
#[derive(Clone, Copy, Debug)]
pub(crate) enum ActionsRefused {
    /// Loaded `DontCare`.
    DontCareLoad,
    /// Reaching past the render area as another attachment does not: cleared where Metal's clear
    /// of whole tiles (`clear_reach.rs`) reaches another extent than another color's, or loaded,
    /// or a memoryless color smaller, where another's clear reaches past the render area.
    UnequalReach,
    /// Resolved, into a resolve attachment or by a copy after the pass.
    Resolve,
    /// Stored other than by `Store`, or late (`storeActionOptions` set after the encoder).
    Store,
}

impl ActionsRefused {
    /// The kinds: the last's index and one.
    const COUNT: usize = ActionsRefused::Store as usize + 1;
}

/// A command the runtime records that MoltenVK 1.4.1 encodes on Apple's GPUs through a pipeline it
/// compiles per command pool, inside `vkQueueSubmit`, at the pool's first such command
/// (`MVKCommandEncodingPool`): counted, so a run shows which of them it records.
#[derive(Clone, Copy, Debug)]
pub(crate) enum PoolCompiled {
    /// `vkCmdClearAttachments`: a render pipeline per pool and attachment formats.
    AttachmentClear,
    /// `vkCmdBlitImage`: a render pipeline per pool, format and filter where the blit scales.
    ImageBlit,
    /// `vkCmdCopyQueryPoolResults`: a compute pipeline per pool.
    QueryResultCopy,
    /// A visibility result mode set, beginning an occlusion query: MoltenVK accumulates the
    /// queries of each Metal render pass that ran one by a compute pipeline per pool.
    VisibilityQuery,
    /// `vkCmdDrawIndirect` and `vkCmdDrawIndexedIndirect`: a compute conversion per pool where the
    /// pipeline tessellates or the pass is multiview.
    IndirectDraw,
}

impl Counters {
    /// `count` commands of `kind` were recorded (`PoolCompiled`).
    pub(crate) fn pool_compiled(&self, kind: PoolCompiled, count: u64) {
        let counter = match kind {
            PoolCompiled::AttachmentClear => &self.attachment_clears_recorded,
            PoolCompiled::ImageBlit => &self.image_blits_recorded,
            PoolCompiled::QueryResultCopy => &self.query_result_copies_recorded,
            PoolCompiled::VisibilityQuery => &self.visibility_queries_begun,
            PoolCompiled::IndirectDraw => &self.indirect_draws_recorded,
        };
        counter.fetch_add(count, Ordering::Relaxed);
    }

    /// A fill was recorded (`word_fill.rs`): as transfers where `by_transfers`, else as
    /// `vkCmdFillBuffer`.
    pub(crate) fn fill_recorded(&self, by_transfers: bool) {
        let counter = match by_transfers {
            true => &self.fills_by_transfers,
            false => &self.fills_recorded,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// A YCbCr texture's conversion was recorded through its refresh, converting only where its
    /// planes changed (`ycbcr.rs`).
    pub(crate) fn ycbcr_refreshed(&self) {
        self.ycbcr_refreshes.fetch_add(1, Ordering::Relaxed);
    }

    /// A YCbCr texture's conversion was recorded whole (`ycbcr.rs`).
    pub(crate) fn ycbcr_converted_whole(&self) {
        self.ycbcr_whole_conversions.fetch_add(1, Ordering::Relaxed);
    }

    /// A two-plane YCbCr texture was made with its planes read in place
    /// (`texture/ycbcr_image.rs`).
    pub(crate) fn ycbcr_texture_in_place(&self) {
        self.ycbcr_textures_in_place.fetch_add(1, Ordering::Relaxed);
    }

    /// A two-plane YCbCr texture was made keeping its image alone for `kept`
    /// (`texture/ycbcr_image.rs`).
    pub(crate) fn ycbcr_image_kept(&self, kept: crate::texture::KeptImage) {
        let at = crate::texture::KeptImage::ALL
            .iter()
            .position(|k| *k == kept)
            .unwrap_or(0);
        if let Some(counter) = self.ycbcr_images_kept.get(at) {
            counter.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A binding of a two-plane YCbCr texture's planes read in place was written
    /// (`binding.rs`).
    pub(crate) fn ycbcr_planes_bound(&self) {
        self.ycbcr_planes_bound.fetch_add(1, Ordering::Relaxed);
    }

    /// A two-plane YCbCr texture's planes read in place were made an image the device's
    /// conversion samples (`texture/ycbcr_planes.rs`).
    pub(crate) fn ycbcr_planes_image(&self) {
        self.ycbcr_planes_images.fetch_add(1, Ordering::Relaxed);
    }

    /// A two-plane YCbCr texture's planes read in place were made no image the device's
    /// conversion samples, for `kept` (`texture/ycbcr_planes.rs`).
    pub(crate) fn ycbcr_planes_image_kept(&self, kept: crate::texture::KeptTexels) {
        let at = crate::texture::KeptTexels::ALL
            .iter()
            .position(|k| *k == kept)
            .unwrap_or(0);
        if let Some(counter) = self.ycbcr_planes_images_kept.get(at) {
            counter.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A binding of a two-plane YCbCr texture's planes image was written (`binding.rs`).
    pub(crate) fn ycbcr_planes_image_bound(&self) {
        self.ycbcr_planes_images_bound
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A YCbCr texture's conversion, or a render shadow's load, was made to sample its planes'
    /// image through the device's own Y′CbCr conversion (`ycbcr_device.rs`).
    pub(crate) fn ycbcr_device_conversion(&self) {
        self.ycbcr_device_conversions
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A sampling of a YCbCr texture's planes' image through the device's own Y′CbCr conversion
    /// was recorded (`ycbcr_device.rs`).
    pub(crate) fn ycbcr_device_sampled(&self) {
        self.ycbcr_device_samples.fetch_add(1, Ordering::Relaxed);
    }

    /// A `VkCommandPool` was made for a command buffer, none of the device's spares taken.
    pub(crate) fn command_pool_made(&self) {
        self.command_pools_made.fetch_add(1, Ordering::Relaxed);
    }

    /// A command buffer took one of the device's spare command pools (`spares.rs`).
    pub(crate) fn spare_command_pool_taken(&self) {
        self.spare_command_pools_taken
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A command buffer took a spare command pool that one of its command buffers may still be
    /// recording in, and reset it first (`spares.rs`).
    pub(crate) fn spare_command_pool_reset(&self) {
        self.spare_command_pools_reset
            .fetch_add(1, Ordering::Relaxed);
    }

    /// The live buffers' table was uploaded to a buffer made for it, no spare reused.
    pub(crate) fn live_table_uploaded(&self) {
        self.live_tables_uploaded.fetch_add(1, Ordering::Relaxed);
    }

    /// A queue's waiter waited for a timeline's value, not at a fence
    /// (`command_buffer/completion.rs`).
    pub(crate) fn timeline_wait(&self) {
        self.timeline_waits.fetch_add(1, Ordering::Relaxed);
    }

    /// A queue's waiter waited for the fence its queue's front batch signals, instead of its
    /// timeline (`command_buffer/fences.rs`).
    pub(crate) fn fence_wait(&self) {
        self.fence_waits.fetch_add(1, Ordering::Relaxed);
    }

    /// A completion waiter's wait ran out its slice although what it waited for was signaled when
    /// the waiter looked again (`command_buffer/completion.rs`).
    pub(crate) fn missed_completion_wake(&self) {
        self.missed_completion_wakes.fetch_add(1, Ordering::Relaxed);
    }

    /// A queue's waiter armed for its queue's front batch, having slept on its condition variable
    /// first where `woken` (`command_buffer/completion.rs`).
    pub(crate) fn completion_armed(&self, woken: bool) {
        self.completion_arms.fetch_add(1, Ordering::Relaxed);
        if woken {
            (self.completion_arms_after_a_wake).fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A completion waiter's read of its queue's timeline showed a value past the one its wait
    /// returned for, so it waited for that value too (`command_buffer/completion.rs`).
    pub(crate) fn completion_read_ran_ahead(&self) {
        (self.completion_reads_ran_ahead).fetch_add(1, Ordering::Relaxed);
    }

    /// A wait for a command buffer's gate ended, having done `waited`
    /// (`CommandBuffer::wait_until_completed`).
    pub(crate) fn gate_waited(&self, waited: super::command_buffer::gate::Waited) {
        self.gate_waits.fetch_add(1, Ordering::Relaxed);
        if waited.slept {
            self.gate_waits_slept.fetch_add(1, Ordering::Relaxed);
        }
        if waited.woke_shut > 0 {
            (self.gate_wakes_found_shut).fetch_add(waited.woke_shut, Ordering::Relaxed);
        }
    }

    /// A wait for a command buffer's gate went on to wait for the trails below its mark
    /// (`CommandBuffer::wait_until_completed`), `blocked` where one was open when it looked.
    pub(crate) fn trail_waited(&self, blocked: bool) {
        self.trail_waits.fetch_add(1, Ordering::Relaxed);
        if blocked {
            self.trail_waits_blocked.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A command buffer's transition to `Completed` or `Error` ran on the queue waiter that saw
    /// it complete (`CommandBufferShared::finish_here`).
    pub(crate) fn completed_where_seen(&self) {
        self.completed_where_seen.fetch_add(1, Ordering::Relaxed);
    }

    /// A command buffer's transition to `Scheduled` ran on the scheduler's thread as its last
    /// batch went (`CommandBufferShared::schedule`).
    pub(crate) fn scheduled_where_submitted(&self) {
        self.scheduled_where_submitted
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A committed command buffer took a slot of one of the device's timestamp pools, to write
    /// its GPU start and end into (`gpu_time.rs`).
    pub(crate) fn timestamp_slot_taken(&self) {
        self.timestamp_slots_taken.fetch_add(1, Ordering::Relaxed);
    }

    /// A slot of one of the device's timestamp pools was reset on the host as a command buffer
    /// took it (`gpu_time.rs`).
    pub(crate) fn timestamp_slot_reset_on_host(&self) {
        self.timestamp_slots_reset_on_host
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A calibration pair was taken of `readings` readings, its deviation `deviation`
    /// nanoseconds (`gpu_time.rs`, `Timing::pair`).
    pub(crate) fn calibrated(&self, readings: u32, deviation: u64) {
        self.calibrations.fetch_add(1, Ordering::Relaxed);
        self.calibration_readings
            .fetch_add(u64::from(readings), Ordering::Relaxed);
        self.calibration_deviation_most
            .fetch_max(deviation, Ordering::Relaxed);
    }

    /// A command buffer's calibrated GPU times fell outside its life, and it was placed by its
    /// completion (`gpu_time.rs`, `GpuClock::span`).
    pub(crate) fn calibration_disagreed(&self) {
        self.calibration_disagreements
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A texture took its storage form (`texture/storage_form.rs`).
    pub(crate) fn storage_form_taken(&self) {
        self.storage_forms_taken.fetch_add(1, Ordering::Relaxed);
    }

    /// The textures that took their storage form so far, as [`Device::work_counts`] reads it.
    #[cfg(test)]
    pub(crate) fn storage_forms_taken(&self) -> u64 {
        self.storage_forms_taken.load(Ordering::Relaxed)
    }

    /// A `VkBuffer` of `usage` was made: counted where the usage names device addresses.
    pub(crate) fn buffer_made(&self, usage: vk::BufferUsageFlags) {
        if usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
            self.addressable_buffers_made
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A `generate_mipmaps` ran the compute filter rather than the device's blits
    /// (`mipmap_filter.rs`).
    pub(crate) fn mipmaps_filtered(&self) {
        self.mipmaps_filtered.fetch_add(1, Ordering::Relaxed);
    }

    /// A `generate_mipmaps` read its texture's first level in place, through a sampled view,
    /// rather than copying it in (`mipmap_filter.rs`).
    pub(crate) fn mipmap_first_level_fetched(&self) {
        self.mipmap_first_levels_fetched
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A draw handed the making of its `VkPipeline` to the device's pool and went on
    /// (`render/pending.rs`).
    pub(crate) fn pipeline_pooled(&self) {
        self.pipelines_pooled.fetch_add(1, Ordering::Relaxed);
    }

    /// A draw or dispatch whose key's `VkPipeline` was not made took its residual form's
    /// (`pipeline/render/residual.rs`).
    pub(crate) fn residual_form_drawn(&self) {
        self.residual_form_draws.fetch_add(1, Ordering::Relaxed);
    }

    /// A draw or dispatch that began its key's making took its residual form's `VkPipeline`
    /// meanwhile (`pipeline/render/residual.rs`).
    pub(crate) fn residual_form_making(&self) {
        self.residual_form_makings.fetch_add(1, Ordering::Relaxed);
    }

    /// Stage entries the device's pipeline creations specialized, and asks of theirs that found
    /// one specialized already (`pipeline/memo.rs`).
    pub(crate) fn stage_entries(&self, specialized: u64, shared: u64) {
        self.stage_entries_specialized
            .fetch_add(specialized, Ordering::Relaxed);
        self.stage_entries_shared
            .fetch_add(shared, Ordering::Relaxed);
    }

    /// A stage's function the runtime translated to SPIR-V, finding it in neither the device's
    /// memo of compiled functions, nor a creation's binary archives, nor the shader cache on disk
    /// (`pipeline/sources.rs`).
    pub(crate) fn function_translated(&self) {
        self.functions_translated.fetch_add(1, Ordering::Relaxed);
    }

    /// An ask for a stage's function that found it compiled already -- in the device's memo, in a
    /// creation's binary archive or in the shader cache on disk -- so nothing was translated
    /// (`pipeline/sources.rs`).
    pub(crate) fn function_shared(&self) {
        self.functions_shared.fetch_add(1, Ordering::Relaxed);
    }

    /// A render pipeline creation worked out one of its descriptor's linked-use answers rather
    /// than reading it from the shader cache (`pipeline/render.rs`).
    pub(crate) fn linked_use_question_asked(&self) {
        self.linked_use_questions_asked
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A render or mesh pipeline descriptor's key the runtime built
    /// (`Device::render_descriptor_key`, `Device::mesh_descriptor_key`): every build the runtime
    /// makes goes through `pipeline::render::descriptor_key` or `mesh_descriptor_key`.
    pub(crate) fn descriptor_key_built(&self) {
        self.descriptor_keys_built.fetch_add(1, Ordering::Relaxed);
    }

    /// A pipeline state's variant made a `VkShaderModule` (`pipeline/objects.rs`).
    pub(crate) fn shader_module_made(&self) {
        self.shader_modules_made.fetch_add(1, Ordering::Relaxed);
    }

    /// A pipeline state's variant took the device's `VkShaderModule` of its stage's SPIR-V
    /// (`pipeline/objects.rs`).
    pub(crate) fn shader_module_shared(&self) {
        self.shader_modules_shared.fetch_add(1, Ordering::Relaxed);
    }

    /// A pipeline state's variant made a `VkPipelineLayout` and its set layouts
    /// (`pipeline/shared_layouts.rs`).
    pub(crate) fn pipeline_layout_made(&self) {
        self.pipeline_layouts_made.fetch_add(1, Ordering::Relaxed);
    }

    /// A pipeline state's variant took the device's `VkPipelineLayout` of its layout's definition
    /// (`pipeline/shared_layouts.rs`).
    pub(crate) fn pipeline_layout_shared(&self) {
        self.pipeline_layouts_shared.fetch_add(1, Ordering::Relaxed);
    }

    /// A pipeline state's variant made a `VkPipeline` (`pipeline/objects.rs`,
    /// `pipeline/graphics.rs`).
    pub(crate) fn variant_pipeline_made(&self) {
        self.variant_pipelines_made.fetch_add(1, Ordering::Relaxed);
    }

    /// A variant's making waited for its twin's in the air before its own (`pipeline/flight.rs`).
    pub(crate) fn flight_followed(&self) {
        self.flights_followed.fetch_add(1, Ordering::Relaxed);
    }

    /// A predicated loop's check ran its laying again (`command_buffer/rerun.rs`).
    pub(crate) fn predicated_loop_ran_again(&self) {
        self.predicated_loop_runs_again
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A graphics pipeline library of `part` was made (`pipeline/libraries.rs`).
    pub(crate) fn pipeline_library_made(&self, part: crate::pipeline::libraries::Part) {
        self.pipeline_libraries_made.fetch_add(1, Ordering::Relaxed);
        if part == crate::pipeline::libraries::Part::FragmentShader {
            self.fragment_shader_libraries_made
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A dispatch handed the making of a region's `VkPipeline` to the device's pool and went on
    /// (`pipeline/dispatch_making.rs`).
    pub(crate) fn dispatch_pipeline_pooled(&self) {
        self.dispatch_pipelines_pooled
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A render encoder recorded a tile dispatch (`render/tile_dispatch.rs`).
    pub(crate) fn tile_dispatch_recorded(&self) {
        self.tile_dispatches_recorded
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A render encoder's tile dispatch handed the making of a region's `VkPipeline` to the
    /// device's pool and went on (`render/pending_tile.rs`).
    pub(crate) fn tile_pipeline_pooled(&self) {
        self.tile_pipelines_pooled.fetch_add(1, Ordering::Relaxed);
    }

    /// A command buffer's upload arena opened a chunk in device-local memory of a device without
    /// unified memory (`upload.rs`).
    pub(crate) fn upload_chunk_in_device_memory(&self) {
        self.upload_chunks_in_device_memory
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A render variant's `VkPipeline` was linked from libraries (`pipeline/graphics.rs`).
    pub(crate) fn pipeline_linked(&self) {
        self.pipelines_linked.fetch_add(1, Ordering::Relaxed);
    }

    /// A variant made again took back a released one's objects (`pipeline/kept.rs`).
    pub(crate) fn variant_taken_back(&self) {
        self.variants_taken_back.fetch_add(1, Ordering::Relaxed);
    }

    /// A command buffer a command of which left a live sampler state out of its sampler arrays
    /// (`heap_sets::fit_samplers`).
    pub(crate) fn unlisted_argument_sampler_buffer(&self) {
        self.unlisted_argument_sampler_buffers
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A draw whose pipeline the driver did not make after the draw returned ran nothing
    /// (`CommandBuffer::drop_draw`).
    pub(crate) fn draw_dropped_unmade(&self) {
        self.draws_dropped_unmade.fetch_add(1, Ordering::Relaxed);
    }

    /// A draw whose pipeline was still being made at its command buffer's bound ran nothing
    /// (`CommandBuffer::drop_late_draw`).
    pub(crate) fn draw_dropped_still_making(&self) {
        self.draws_dropped_still_making
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A dispatch, or a render pass's tile dispatch, whose pipeline was still being made at its
    /// command buffer's bound ran nothing (`CommandBuffer::drop_late_dispatch`).
    pub(crate) fn dispatch_dropped_still_making(&self) {
        self.dispatches_dropped_still_making
            .fetch_add(1, Ordering::Relaxed);
    }

    /// `count` records of an indirect command buffer's commands were read from its storage and
    /// decoded (`indirect_command_buffer.rs`).
    pub(crate) fn indirect_records_read(&self, count: u64) {
        self.indirect_records_read
            .fetch_add(count, Ordering::Relaxed);
    }

    /// `count` commands of an execution of render commands were recorded in its GPU form
    /// (`render/executions/gpu.rs`): the device reads their records.
    pub(crate) fn gpu_form_commands(&self, count: u64) {
        self.gpu_form_commands.fetch_add(count, Ordering::Relaxed);
    }

    /// `count` commands of an execution of render commands in its GPU form were drawn by one
    /// merged draw (`render/executions/gpu.rs`), not one draw each.
    pub(crate) fn merged_draw_commands(&self, count: u64) {
        self.merged_draw_commands
            .fetch_add(count, Ordering::Relaxed);
    }

    /// A command buffer's lease made its queue's block of a GPU pool: the queue held none
    /// (`gpu_pools.rs`).
    pub(crate) fn gpu_pool_block_made(&self) {
        self.gpu_pool_blocks_made.fetch_add(1, Ordering::Relaxed);
    }

    /// A lease of a queue's block of a GPU pool began, `leased` leases holding it now, this one's
    /// included (`gpu_pools.rs`).
    pub(crate) fn gpu_pool_leased(&self, leased: u64) {
        self.gpu_pool_leases_peak
            .fetch_max(leased, Ordering::Relaxed);
    }

    /// A queue's block of a GPU pool was released before the device's teardown: no lease held it
    /// for too long, or its queue went (`gpu_pools.rs`).
    pub(crate) fn gpu_pool_block_released(&self) {
        self.gpu_pool_blocks_released
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A host step's variant made immutable samplers no live set had (`pipeline/objects.rs`).
    pub(crate) fn host_step_sampler_set_made(&self) {
        self.host_step_sampler_sets_made
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A state's creation made a `VkPipeline` of `predicted` (`pipeline/predicted.rs`,
    /// `pipeline/dispatch_making.rs`): counted in both the whole count and its prediction's.
    pub(crate) fn creation_pipeline_made(&self, predicted: Predicted) {
        self.creation_pipelines_made.fetch_add(1, Ordering::Relaxed);
        match predicted {
            Predicted::Render => &self.creation_render_pipelines_made,
            Predicted::Dispatch => &self.creation_dispatch_pipelines_made,
        }
        .fetch_add(1, Ordering::Relaxed);
    }

    /// A state's creation returned while its predicted key's making, handed to the device's pool,
    /// was still running at the bound (`PipelineWaitPolicy::Bounded`, `pipeline/render.rs`,
    /// `pipeline/dispatch_making.rs`).
    pub(crate) fn creation_left_making(&self) {
        self.creations_left_making.fetch_add(1, Ordering::Relaxed);
    }

    /// A draw or dispatch took a `VkPipeline` a state's creation made for `predicted`, its first
    /// use: counted in both the whole count and its prediction's.
    pub(crate) fn creation_pipeline_drawn(&self, predicted: Predicted) {
        self.creation_pipelines_drawn
            .fetch_add(1, Ordering::Relaxed);
        match predicted {
            Predicted::Render => &self.creation_render_pipelines_drawn,
            Predicted::Dispatch => &self.creation_dispatch_pipelines_drawn,
        }
        .fetch_add(1, Ordering::Relaxed);
    }

    /// A variant made an immutable sampler set: the device's map's, or a host step's or a pool
    /// piece's own (`pipeline/objects.rs`).
    pub(crate) fn sampler_set_made(&self) {
        self.sampler_sets_made.fetch_add(1, Ordering::Relaxed);
    }

    /// A variant took the device's set of its immutable samplers, made before.
    pub(crate) fn sampler_set_shared(&self) {
        self.sampler_sets_shared.fetch_add(1, Ordering::Relaxed);
    }

    /// The device's map destroyed a set of immutable samplers: at a settle, a sweep or the
    /// teardown (`pipeline/objects.rs`).
    pub(crate) fn sampler_set_destroyed(&self) {
        self.sampler_sets_destroyed.fetch_add(1, Ordering::Relaxed);
    }

    /// A commit made ahead, for its render pass's host step, `passes` render passes and
    /// `framebuffers` framebuffers, taking `nanos` nanoseconds (`render/executions/ahead.rs`).
    pub(crate) fn made_ahead(&self, passes: u64, framebuffers: u64, nanos: u64) {
        self.ahead_render_passes_made
            .fetch_add(passes, Ordering::Relaxed);
        self.ahead_framebuffers_made
            .fetch_add(framebuffers, Ordering::Relaxed);
        self.ahead_nanos.fetch_add(nanos, Ordering::Relaxed);
    }

    /// A memoryless attachment's texture made a backing to keep (`attachment_stand_in.rs`).
    pub(crate) fn kept_backing_made(&self) {
        self.kept_backings_made.fetch_add(1, Ordering::Relaxed);
    }

    /// A pass made a backing for a memoryless color attachment, which its queue keeps
    /// (`attachment_stand_in::pooled_backing`).
    pub(crate) fn pooled_backing_made(&self) {
        self.pooled_backings_made.fetch_add(1, Ordering::Relaxed);
    }

    /// The device made a placeholder image for the colors render pass instances leave out of
    /// their subpass for the pixel interlock (`storage_placeholder.rs`).
    pub(crate) fn storage_placeholder_made(&self) {
        self.storage_placeholders_made
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A framebuffer took a placeholder's view for a color its render pass instance leaves out
    /// of its subpass for the pixel interlock (`storage_placeholder.rs`).
    pub(crate) fn storage_placeholder_attachment(&self) {
        self.storage_placeholder_attachments
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A pass made the objects of its exact color resolves (`color_resolve.rs`).
    pub(crate) fn color_resolve_pass_made(&self) {
        self.color_resolve_passes_made
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A pass took the objects of its exact color resolves that an image kept
    /// (`color_resolve::KeptResolves`).
    pub(crate) fn color_resolve_pass_kept(&self) {
        self.color_resolve_passes_kept
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A driver call reported `withheld` diagnostics its driver facts measure it to report, which
    /// the device's sink was not handed (`driver_facts.rs`).
    pub(crate) fn driver_diagnostics_withheld(&self, withheld: u64) {
        self.driver_diagnostics_withheld
            .fetch_add(withheld, Ordering::Relaxed);
    }

    /// An internal submission was made (`device/submit.rs`): one a host call waits for.
    pub(crate) fn internal_submission(&self) {
        self.internal_submissions.fetch_add(1, Ordering::Relaxed);
    }

    /// A `vkQueueSubmit` or `vkQueueBindSparse` call was made with `infos` `VkSubmitInfo`s
    /// (none for a sparse bind) and `command_buffers` `VkCommandBuffer`s among them.
    pub(crate) fn queue_submitted(&self, infos: usize, command_buffers: usize) {
        self.queue_submissions.fetch_add(1, Ordering::Relaxed);
        self.submit_infos.fetch_add(infos as u64, Ordering::Relaxed);
        (self.command_buffers_submitted).fetch_add(command_buffers as u64, Ordering::Relaxed);
    }

    /// A batch's `vkQueueSubmit` (or a group's, or a command buffer's fence's) spent `nanos`
    /// nanoseconds in the driver (`device/relay.rs`).
    pub(crate) fn batch_submitted_in(&self, nanos: u64) {
        self.batch_submit_nanos.fetch_add(nanos, Ordering::Relaxed);
    }

    /// A signalling call of work took the driver's stall (`device/submit_stalls.rs`).
    pub(crate) fn submit_stall_on_work(&self) {
        self.submit_stalls_on_work.fetch_add(1, Ordering::Relaxed);
    }

    /// A submission waited `nanos` nanoseconds for its queue's lock behind a stall taken idle
    /// (`device/submit_stalls.rs`).
    pub(crate) fn submit_behind_idle_stall(&self, nanos: u64) {
        self.submits_behind_idle_stalls
            .fetch_add(1, Ordering::Relaxed);
        (self.wait_behind_idle_stall_longest_nanos).fetch_max(nanos, Ordering::Relaxed);
    }

    /// A render encoder ended (`RenderCommandEncoder::end_encoding`).
    pub(crate) fn render_encoder_ended(&self) {
        self.render_encoders_ended.fetch_add(1, Ordering::Relaxed);
    }

    /// A batch of a render encoder's logged calls was lowered: on the device's lowering thread
    /// (`by_thread`), or on the program's at a drain (`render/deferral.rs`).
    pub(crate) fn render_log_batch_lowered(&self, by_thread: bool) {
        match by_thread {
            true => &self.render_log_batches_by_thread,
            false => &self.render_log_batches_by_program,
        }
        .fetch_add(1, Ordering::Relaxed);
    }

    /// A render encoder's call waited for the device's lowering thread, its encoder's batches
    /// handed and not taken at their cap (`render/deferral.rs`).
    pub(crate) fn render_log_waited(&self) {
        self.render_log_waits.fetch_add(1, Ordering::Relaxed);
    }

    /// A render encoder's end that its call left to the device's lowering thread was lowered:
    /// on that thread (`by_thread`), or on the program's by what followed it in the command
    /// buffer (`render/deferral.rs`).
    pub(crate) fn render_log_end_lowered(&self, by_thread: bool) {
        match by_thread {
            true => &self.render_log_ends_by_thread,
            false => &self.render_log_ends_by_program,
        }
        .fetch_add(1, Ordering::Relaxed);
    }

    /// A commit's own work that waited for a render encoder's end ran: on the device's lowering
    /// thread (`by_thread`), or on the program's (`render/queued_commits.rs`).
    pub(crate) fn render_log_commit_run(&self, by_thread: bool) {
        match by_thread {
            true => &self.render_log_commits_by_thread,
            false => &self.render_log_commits_by_program,
        }
        .fetch_add(1, Ordering::Relaxed);
    }

    /// A render pass instance of a render encoder's pass was begun at `site`.
    pub(crate) fn render_pass_begun(&self, site: PassBegun) {
        self.render_passes_begun_by[site as usize].fetch_add(1, Ordering::Relaxed);
    }

    /// A render encoder's pass split for `cause`: its next draw records into a new instance after
    /// one that recorded.
    pub(crate) fn instance_split(&self, cause: Split) {
        self.instance_splits_by[cause as usize].fetch_add(1, Ordering::Relaxed);
    }

    /// A `VkCommandBuffer` was begun (`vkBeginCommandBuffer`) for `what`.
    pub(crate) fn command_buffer_begun(&self, what: Begun) {
        self.command_buffers_begun.fetch_add(1, Ordering::Relaxed);
        self.command_buffers_begun_by[what as usize].fetch_add(1, Ordering::Relaxed);
    }

    /// An encoder began a `VkCommandBuffer` as it recorded (`vkBeginCommandBuffer`) for `what`:
    /// its own, a segment's, one it went on in, or a render encoder's draws' secondary.
    pub(crate) fn encoder_command_buffer_begun(&self, what: Begun) {
        self.command_buffer_begun(what);
        (self.encoder_command_buffers_begun).fetch_add(1, Ordering::Relaxed);
    }

    /// An encoder took its own `VkCommandBuffer` begun ahead in its spare command pool
    /// (`spares.rs`), counted as begun too ([`encoder_command_buffer_begun`]).
    ///
    /// [`encoder_command_buffer_begun`]: Self::encoder_command_buffer_begun
    pub(crate) fn encoder_primary_begun_ahead(&self) {
        (self.encoder_primaries_begun_ahead).fetch_add(1, Ordering::Relaxed);
    }

    /// A spare command pool's primary was begun ahead, and the pool kept among the spares again
    /// (`spares.rs`).
    pub(crate) fn primary_begun_ahead(&self) {
        self.primaries_begun_ahead.fetch_add(1, Ordering::Relaxed);
    }

    /// A batch's barrier command buffer was recorded to be kept and submitted again
    /// (`command_buffer/kept_barriers.rs`).
    pub(crate) fn barrier_buffer_kept(&self) {
        self.barrier_buffers_kept.fetch_add(1, Ordering::Relaxed);
    }

    /// A batch's barrier command buffer that writes a command buffer's timestamps was recorded
    /// to be kept for its slot and submitted again (`command_buffer/kept_barriers.rs`).
    pub(crate) fn stamped_buffer_kept(&self) {
        self.stamped_buffers_kept.fetch_add(1, Ordering::Relaxed);
    }

    /// A batch took such a kept command buffer again (`command_buffer/kept_barriers.rs`).
    pub(crate) fn stamped_buffer_reused(&self) {
        self.stamped_buffers_reused.fetch_add(1, Ordering::Relaxed);
    }

    /// A batch with work was submitted with its end in its last command buffer
    /// (`command_buffer/submit.rs`).
    pub(crate) fn finale_folded(&self) {
        self.finales_folded.fetch_add(1, Ordering::Relaxed);
    }

    /// A batch with work was submitted with its end in a command buffer of its own.
    pub(crate) fn finale_apart(&self) {
        self.finales_apart.fetch_add(1, Ordering::Relaxed);
    }

    /// One `vkQueueSubmit` submitted `batches` batches, more than one, only the last signalling
    /// (`command_buffer/group.rs`).
    pub(crate) fn batches_grouped(&self, batches: usize) {
        self.grouped_submissions.fetch_add(1, Ordering::Relaxed);
        (self.batches_grouped).fetch_add(batches as u64, Ordering::Relaxed);
    }

    /// A thread about to block for a command buffer's completion made a pass of its own over the
    /// scheduler's order and submitted `batches` batches itself
    /// (`Scheduler::submit_before_waiting`).
    pub(crate) fn waiter_passed(&self, batches: usize) {
        (self.scheduler_passes_by_waiters).fetch_add(1, Ordering::Relaxed);
        (self.batches_submitted_by_waiters).fetch_add(batches as u64, Ordering::Relaxed);
    }

    /// A batch was submitted, or held in a group, while `pending` earlier batches of its
    /// `VkQueue` were submitted or held and not seen complete (`command_buffer/submit.rs`).
    pub(crate) fn batch_submitted_behind(&self, pending: u64) {
        let bucket = match pending {
            0 => 0,
            1 => 1,
            2 | 3 => 2,
            _ => 3,
        };
        self.batches_by_pending[bucket].fetch_add(1, Ordering::Relaxed);
    }

    /// A group of batches held for one kernel submission went, sent by `end`
    /// (`command_buffer/group.rs`).
    pub(crate) fn group_ended(&self, end: GroupEnd) {
        self.group_ends[end as usize].fetch_add(1, Ordering::Relaxed);
    }

    /// A group of batches stayed held past the pass that found no batch ready, for the work its
    /// `VkQueue` still ran (`command_buffer/group.rs`); once per group.
    pub(crate) fn group_kept_pending(&self) {
        self.groups_kept_pending.fetch_add(1, Ordering::Relaxed);
    }

    /// Such a group went at once because a thread waits for one of its command buffers.
    pub(crate) fn group_sent_for_wait(&self) {
        self.groups_sent_for_waits.fetch_add(1, Ordering::Relaxed);
    }

    /// A look of the scheduler's for the next batch to submit found a line's head held for
    /// `reason`, so it submitted no batch of that command buffer (`scheduler/mod.rs`): once per
    /// head the look found held, so a head held through many looks is counted at each.
    pub(crate) fn head_held_for(&self, reason: HeldFor) {
        self.heads_held_for[reason as usize].fetch_add(1, Ordering::Relaxed);
    }

    /// Such a look left a head waiting at its host step's gate for the work submitted before it
    /// to complete (`scheduler/mod.rs`'s `gate_heads`).
    pub(crate) fn head_at_its_gate(&self) {
        self.heads_waiting_at_their_gate
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Such a look left a head's submission checks waiting for a submitted GPU signal to land
    /// (`scheduler/mod.rs`'s `check_heads`).
    pub(crate) fn head_awaiting_check_signals(&self) {
        self.heads_awaiting_check_signals
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A batch a look had found held went (submitted, or held in a group) `nanos` nanoseconds
    /// after the first look that found it held (`scheduler/mod.rs`).
    pub(crate) fn batch_hold_ended(&self, nanos: u64) {
        self.batches_held_past_a_look
            .fetch_add(1, Ordering::Relaxed);
        self.batch_hold_nanos.fetch_add(nanos, Ordering::Relaxed);
    }

    /// A command buffer's change woke the threads sleeping on it where `parked`, else spared the
    /// notify (`command_buffer/mod.rs`).
    pub(crate) fn command_buffer_wake(&self, parked: bool) {
        match parked {
            true => &self.command_buffer_wakes,
            false => &self.command_buffer_wakes_spared,
        }
        .fetch_add(1, Ordering::Relaxed);
    }

    /// An event's rise woke the host waits sleeping on it where `parked`, else spared the notify
    /// (`event.rs`).
    pub(crate) fn event_wake(&self, parked: bool) {
        match parked {
            true => &self.event_wakes,
            false => &self.event_wakes_spared,
        }
        .fetch_add(1, Ordering::Relaxed);
    }

    /// A completed command buffer gave its queue's slot back `nanos` nanoseconds after its last
    /// completed handler returned (`command_buffer/mod.rs`).
    pub(crate) fn slot_returned(&self, nanos: u64) {
        self.slot_returns.fetch_add(1, Ordering::Relaxed);
        self.slot_return_nanos.fetch_add(nanos, Ordering::Relaxed);
    }

    /// A new command buffer waited `nanos` nanoseconds for a slot of its full queue (`queue.rs`).
    pub(crate) fn slot_waited(&self, nanos: u64) {
        self.slot_waits.fetch_add(1, Ordering::Relaxed);
        self.slot_wait_nanos.fetch_add(nanos, Ordering::Relaxed);
    }

    /// A new logical queue took a `VkQueue` an earlier logical queue took too
    /// (`Device::next_queue`).
    pub(crate) fn queue_shared_a_vk_queue(&self) {
        self.queues_sharing_a_vk_queue
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A batch with work was submitted waiting for `waits` timelines at values the host saw them
    /// reach (`command_buffer/submit.rs`).
    pub(crate) fn completed_waited(&self, waits: usize) {
        self.completed_waits
            .fetch_add(waits as u64, Ordering::Relaxed);
    }

    /// A batch went (submitted, or held in a group) with `waits` of its event waits for values
    /// the scheduler found reached as it took the batch, so it waits for none of them
    /// (`scheduler/mod.rs`).
    pub(crate) fn event_waits_met(&self, waits: usize) {
        self.event_waits_met
            .fetch_add(waits as u64, Ordering::Relaxed);
    }

    /// A batch became a line's head holding `waits` of its event waits for values the host may
    /// set that a submitted GPU signal reaches: each one chained on the `VkQueue` before, and
    /// costs a hold and a `vkQueueSubmit` now (`scheduler/mod.rs`). Counted once per batch.
    pub(crate) fn shared_event_waits_held(&self, waits: usize) {
        self.shared_event_waits_held
            .fetch_add(waits as u64, Ordering::Relaxed);
    }

    /// A batch was submitted with `signals` kept event signals while the device kept a copied
    /// host buffer: they are seen at its completion turn alone (`event_timeline.rs`).
    pub(crate) fn event_signals_behind_merge(&self, signals: usize) {
        self.event_signals_behind_merge
            .fetch_add(signals as u64, Ordering::Relaxed);
    }

    /// A batch with work was submitted with no command buffer of barriers before its first
    /// encoder.
    pub(crate) fn bare_start(&self) {
        self.bare_starts.fetch_add(1, Ordering::Relaxed);
    }

    /// A timed command buffer's first batch was submitted with its start written by its first
    /// encoder's own command buffer (`gpu_time.rs`, `command_buffer/submit.rs`).
    pub(crate) fn prefixed_start(&self) {
        self.prefixed_starts.fetch_add(1, Ordering::Relaxed);
    }

    /// A command of an encoder Metal orders (a blit encoder, a serial compute encoder) was
    /// recorded after a barrier for the encoder's earlier commands (`tracker::EncoderUses`).
    pub(crate) fn command_barrier(&self) {
        self.command_barriers.fetch_add(1, Ordering::Relaxed);
    }

    /// A render pass's first instance began inline in a primary command buffer of its encoder's
    /// (`render/inline_first.rs`).
    pub(crate) fn inline_pass(&self) {
        self.inline_passes.fetch_add(1, Ordering::Relaxed);
    }

    /// A render pass's first instance did not record inline, for `why` (`NotInline`).
    pub(crate) fn pass_not_inline(&self, why: NotInline) {
        self.passes_not_inline[why.index()].fetch_add(1, Ordering::Relaxed);
        if let NotInline::Actions(action) = why {
            self.passes_not_inline_actions_by[action as usize].fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A render pass whose first instance recorded inline ended with more than one instance.
    pub(crate) fn inline_pass_split(&self) {
        self.inline_passes_split.fetch_add(1, Ordering::Relaxed);
    }

    /// A render pass instance after a split of such a pass began inline in the same primary
    /// command buffer (`render/inline_splits.rs`).
    pub(crate) fn inline_split_instance(&self) {
        self.inline_split_instances.fetch_add(1, Ordering::Relaxed);
    }

    /// A pass whose first instance recorded inline split before a draw that writes a memoryless
    /// color attachment, which no instance recorded inline writes (`render/inline_first.rs`).
    pub(crate) fn inline_memoryless_write_split(&self) {
        self.inline_memoryless_write_splits
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A render pass instance its pass replaced or gave up before running it had its draws
    /// recording given up (`render.rs`, `RenderCommandEncoder::end_unrun`): ended.
    pub(crate) fn unrun_instance_ended(&self) {
        self.unrun_instances_ended.fetch_add(1, Ordering::Relaxed);
    }

    /// A render pass that changes no texel ended without recording anything
    /// (`render/untouched.rs`).
    pub(crate) fn untouched_pass_skipped(&self) {
        self.untouched_passes_skipped
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A commit left its batches to the scheduler's thread (`scheduler/inbox.rs`).
    pub(crate) fn commit_handed_off(&self) {
        self.commits_handed_off.fetch_add(1, Ordering::Relaxed);
    }

    /// A submission on the staging queue was made (`device/staging.rs`).
    pub(crate) fn staging_submission(&self) {
        self.staging_submissions.fetch_add(1, Ordering::Relaxed);
    }

    /// A texture's host-transfer image was placed in memory the host does not cache
    /// (`memory::host_cached_placement`).
    pub(crate) fn uncached_host_image(&self) {
        self.uncached_host_images.fetch_add(1, Ordering::Relaxed);
    }

    /// A host call's write into a texture with a decoded twin was decoded on the CPU into the
    /// twin (`Image::decode_written`, on the device's decode thread).
    pub(crate) fn cpu_block_decode(&self) {
        self.cpu_block_decodes.fetch_add(1, Ordering::Relaxed);
    }

    /// A GPU decode pass into a decoded twin was made (`Image::decode_pass`).
    pub(crate) fn block_decode_pass(&self) {
        self.block_decode_passes.fetch_add(1, Ordering::Relaxed);
    }

    /// A 4:2:2 texture's decoded twin was made to hold the device's own Y′CbCr conversion of its
    /// pairs (`ycbcr_device::PairsSampling`).
    pub(crate) fn chroma_device_twin(&self) {
        self.chroma_device_twins.fetch_add(1, Ordering::Relaxed);
    }

    /// A GPU decode pass into a 4:2:2 texture's decoded twin through the device's own Y′CbCr
    /// conversion was made (`Image::decode_pass`).
    pub(crate) fn chroma_device_decode(&self) {
        self.chroma_device_decodes.fetch_add(1, Ordering::Relaxed);
    }

    /// An ASTC carrier's second decoded twin was made (`Image::make_second_twin`).
    pub(crate) fn second_twin(&self) {
        self.second_twins.fetch_add(1, Ordering::Relaxed);
    }

    /// A submission checked `n` blit writes into ASTC carriers for second twins' decodes
    /// (`command_buffer/submit.rs`).
    pub(crate) fn second_twin_checks(&self, n: usize) {
        self.second_twin_checks
            .fetch_add(n as u64, Ordering::Relaxed);
    }

    /// A submission recorded `n` decodes into second decoded twins.
    pub(crate) fn second_twin_decodes(&self, n: usize) {
        self.second_twin_decodes
            .fetch_add(n as u64, Ordering::Relaxed);
    }

    /// The device's queues were drained before a failed wait's command buffers let go of what
    /// they used (`Device::drain_queues`).
    pub(crate) fn queue_drain(&self) {
        self.queue_drains.fetch_add(1, Ordering::Relaxed);
    }

    /// A pipeline a depth stand-in's copies run was made (`depth_copy.rs`: the depth split, the
    /// draw back from the stand-in), by a draw's encoding where `at_draw`, else by a render
    /// pipeline state's creation.
    pub(crate) fn stand_in_copy_made(&self, at_draw: bool) {
        self.stand_in_copies_made.fetch_add(1, Ordering::Relaxed);
        if at_draw {
            self.stand_in_copies_made_at_draws
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A draw made the bind shape of its bind state (`LiveBind::shape`).
    pub(crate) fn draw_shape_made(&self) {
        self.draw_shapes_made.fetch_add(1, Ordering::Relaxed);
    }

    /// A draw found the bind shape of its bind state among those its drawing state settled
    /// lately (`LiveBind::shape`).
    pub(crate) fn draw_shape_found(&self) {
        self.draw_shapes_found.fetch_add(1, Ordering::Relaxed);
    }

    /// A draw took the device's live sampler states to list (`sampler::ArgumentSamplers`).
    pub(crate) fn argument_samplers_taken(&self) {
        self.argument_samplers_taken.fetch_add(1, Ordering::Relaxed);
    }

    /// The runtime made a render pass (`render_target::create`).
    pub(crate) fn render_pass_made(&self) {
        self.render_passes_made.fetch_add(1, Ordering::Relaxed);
    }

    /// A render encoder's bind worked out a texture slot's view that its bind memo did not keep
    /// (`binding::BindMemo`).
    pub(crate) fn bind_view_worked_out(&self) {
        self.bind_views_worked_out.fetch_add(1, Ordering::Relaxed);
    }

    /// A draw read a stage's table into its encoder's kept bind state
    /// (`ArgumentTable::refresh_reading`).
    pub(crate) fn stage_table_read(&self) {
        self.stage_tables_read.fetch_add(1, Ordering::Relaxed);
    }

    /// One of those reads brought in only the slots calls changed since the table was read
    /// (`ArgumentTable::refresh_changed`).
    pub(crate) fn stage_table_read_in_part(&self) {
        self.stage_tables_read_in_part
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A look at where a no-copy buffer's bytes lie, which takes its pages' lock
    /// (`buffer::HostCell`).
    pub(crate) fn no_copy_lie_read(&self) {
        self.no_copy_lies_read.fetch_add(1, Ordering::Relaxed);
    }

    /// A draw's uses were noted already: the last command its encoder noted had them
    /// (`noted_apart.rs`).
    pub(crate) fn command_noted_as_before(&self) {
        self.commands_noted_as_before
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A draw settled its variant and plan from its bind shape (`GraphicsState::settle_shape`).
    pub(crate) fn draw_settled(&self) {
        self.draws_settled.fetch_add(1, Ordering::Relaxed);
    }

    /// A draw ran a variant that takes a slot bound to a two-plane extended-range texture
    /// (`VariantKey::extended_range`).
    pub(crate) fn two_plane_draw(&self) {
        self.two_plane_draws.fetch_add(1, Ordering::Relaxed);
    }

    /// A draw under the last provoking vertex ran a `VkPipeline` of the last vertex
    /// (`render/provoking.rs`).
    pub(crate) fn provoking_last_pipeline_draw(&self) {
        self.provoking_last_pipeline_draws
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A draw under the last provoking vertex drew the list of its primitives in Metal's mode-1
    /// order or led by their last vertices (`render/provoking.rs`).
    pub(crate) fn provoking_listed_draw(&self) {
        self.provoking_listed_draws.fetch_add(1, Ordering::Relaxed);
    }

    /// A direct indexed draw drawn as a list joined a run of the index gather pass, which writes
    /// the run's lists by one dispatch (`index_rewrite.rs`).
    pub(crate) fn gathered_draw(&self) {
        self.gathered_draws.fetch_add(1, Ordering::Relaxed);
    }

    /// An indexed indirect draw whose index fetch the index bounds pass bounds joined a run of
    /// it, which plans and fills the run's draws by one dispatch each (`index_bounds.rs`).
    pub(crate) fn bounded_run_draw(&self) {
        self.bounded_run_draws.fetch_add(1, Ordering::Relaxed);
    }

    /// A bind of a variant made the notices it announces at every occurrence
    /// (`VariantObjects::announce_bound`).
    pub(crate) fn bound_notices_made(&self) {
        self.bound_notices_made.fetch_add(1, Ordering::Relaxed);
    }

    /// A command retained what an argument table binds (`ArgumentTable::retain`), taking hold of
    /// `objects` of them, one at least.
    pub(crate) fn argument_table_retained(&self, objects: u64) {
        self.argument_tables_retained
            .fetch_add(1, Ordering::Relaxed);
        self.argument_objects_retained
            .fetch_add(objects, Ordering::Relaxed);
    }

    /// A draw looked its `VkPipeline` up in its variant's memo by key
    /// (`RenderVariant::begin_graphics_pipeline`, `graphics_pipeline_now`, `graphics_pipeline`).
    pub(crate) fn graphics_pipeline_looked_up(&self) {
        self.graphics_pipelines_looked_up
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A pass over a bound linear attachment rendered them all in place, or one through the
    /// image its texture keeps (`through_copies`; `linear_attachment.rs`).
    pub(crate) fn linear_pass(&self, through_copies: bool) {
        let counter = match through_copies {
            false => &self.linear_passes_in_place,
            true => &self.linear_passes_through_copies,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// A pass rendered a texture over a buffer it does not store through the image its texture
    /// keeps (`kept_bytes.rs`).
    pub(crate) fn kept_bytes_pass(&self) {
        self.kept_bytes_passes.fetch_add(1, Ordering::Relaxed);
    }

    /// A render encoder drew one of its pass's later color groups (`render/color_groups.rs`).
    pub(crate) fn color_group_pass(&self) {
        self.color_group_passes.fetch_add(1, Ordering::Relaxed);
    }

    /// A pass of several color groups made the twin of its depth/stencil attachment its later
    /// groups render against (`render/color_groups.rs`).
    pub(crate) fn color_group_twin(&self) {
        self.color_group_twins.fetch_add(1, Ordering::Relaxed);
    }

    /// A later color group recorded nothing for a draw that writes none of its colors, no depth
    /// and no stencil (`render/color_groups.rs`).
    pub(crate) fn color_group_draw_skipped(&self) {
        self.color_group_draws_skipped
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Draws recorded `count` state commands into their secondary command buffers
    /// (`render::Recording::record`).
    pub(crate) fn draw_states_recorded(&self, count: u64) {
        if count > 0 {
            self.draw_states_recorded
                .fetch_add(count, Ordering::Relaxed);
        }
    }

    /// A draw recorded with its set 0 pushed (`pipeline::objects::pushes_set_zero`).
    pub(crate) fn draw_pushing_set_zero(&self) {
        self.draws_pushing_set_zero.fetch_add(1, Ordering::Relaxed);
    }

    /// A dispatch encoded with its set 0 pushed (`pipeline::objects::pushes_set_zero`).
    pub(crate) fn dispatch_pushing_set_zero(&self) {
        self.dispatches_pushing_set_zero
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A command that bound the set 0 its encoder's last command wrote, the same descriptors
    /// into a set of the same layout (`binding::LastSetZero`).
    pub(crate) fn set_zero_set_reused(&self) {
        self.set_zero_sets_reused.fetch_add(1, Ordering::Relaxed);
    }

    /// A command that bound a set 0 an earlier command of its command buffer wrote, the same
    /// descriptors into a set of the same layout (`binding::SetCache`).
    pub(crate) fn set_zero_set_found(&self) {
        self.set_zero_sets_found.fetch_add(1, Ordering::Relaxed);
    }

    /// A draw that took its settings, draw state and `VkPipeline` from its encoder's earlier
    /// draws of the same inputs (`render/draw_shape.rs`).
    pub(crate) fn draw_shape_reused(&self) {
        self.draw_shapes_reused.fetch_add(1, Ordering::Relaxed);
    }

    /// A compute encoder's dispatch that took what its last dispatch settled
    /// (`compute::DispatchMemo`).
    pub(crate) fn dispatch_settled_as_before(&self) {
        self.dispatches_settled_as_before
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A direct dispatch that settled a bind state of the shape its state's last direct dispatch
    /// planned and took what that one planned (`pipeline::Planned`).
    pub(crate) fn dispatch_planned_as_before(&self) {
        self.dispatches_planned_as_before
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A dispatch of a variant that bakes samplers, its pipelines made, recorded in a
    /// `VkCommandBuffer` of its own (`RuntimeDriverDefect::ImmutableSamplersLostAfterCopies`).
    pub(crate) fn baked_sampler_dispatch_apart(&self) {
        self.baked_sampler_dispatches_apart
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Draws' pushes of their set 0 carried `count` writes (`render/bound.rs`).
    pub(crate) fn set_zero_writes_pushed(&self, count: u64) {
        if count > 0 {
            self.set_zero_writes_pushed
                .fetch_add(count, Ordering::Relaxed);
        }
    }

    /// `switches` draws' pushes carried their set 0 whole with another layout of the shape set 0
    /// was last pushed with: `carried` writes, where `if_shared` would do if the layouts were one
    /// (`render/bound.rs`).
    pub(crate) fn set_zero_equal_shape_switches(
        &self,
        switches: u64,
        carried: u64,
        if_shared: u64,
    ) {
        if switches > 0 {
            self.set_zero_equal_shape_switches
                .fetch_add(switches, Ordering::Relaxed);
            self.set_zero_writes_on_equal_shape_switches
                .fetch_add(carried, Ordering::Relaxed);
            self.set_zero_writes_if_shapes_shared
                .fetch_add(if_shared, Ordering::Relaxed);
        }
    }

    /// A no-copy buffer's deallocator waits past its release: given no keep-alive, for the import
    /// of its pages, which something else still holds, or for the batches in flight as that
    /// import's memory is freed (`freed_memory.rs`); or for a live buffers' table that lists its
    /// pages, which a command buffer holds (`buffer.rs`).
    pub(crate) fn host_deallocator_delayed(&self) {
        self.host_deallocators_delayed
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A host import's free, and what waits for it, posted to a handler thread as the import
    /// went (a keep-alive alone waiting for the free), pending until it ran
    /// ([`host_import_release_done`](Self::host_import_release_done)).
    pub(crate) fn host_import_release_posted(&self) {
        self.host_import_releases_posted
            .fetch_add(1, Ordering::Relaxed);
        self.host_import_releases_pending
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A posted host import's free, and what waited for it, ran
    /// ([`host_import_release_posted`](Self::host_import_release_posted)).
    pub(crate) fn host_import_release_done(&self) {
        self.host_import_releases_pending
            .fetch_sub(1, Ordering::Release);
    }

    /// A no-copy buffer's keep-alive is kept past its release by the import of its pages, which
    /// something else still holds and no other buffer lies in (`buffer.rs`).
    pub(crate) fn host_keep_alive_kept(&self) {
        self.host_keep_alives_kept.fetch_add(1, Ordering::Relaxed);
    }

    /// The import that kept a keep-alive was freed, and dropped it.
    pub(crate) fn host_keep_alive_freed(&self) {
        self.host_keep_alives_kept.fetch_sub(1, Ordering::Relaxed);
    }

    /// A no-copy buffer's keep-alive was dropped at its release: another buffer lying in the
    /// import of its pages, or the keep-alive the import keeps, keeps them mapped until it is
    /// freed (`buffer.rs`).
    pub(crate) fn host_keep_alive_covered(&self) {
        self.host_keep_alives_covered
            .fetch_add(1, Ordering::Relaxed);
    }

    /// The kernel was asked about `mappings` of the process's mappings, for their write
    /// permission, before an import of host pages (`host_memory.rs`'s `refuse_unwritable`, which
    /// asks only on Linux).
    #[cfg(target_os = "linux")]
    pub(crate) fn host_mappings_queried(&self, mappings: u64) {
        if mappings > 0 {
            self.host_mappings_queried
                .fetch_add(mappings, Ordering::Relaxed);
        }
    }

    /// `chunks` chunks of host pages a caller's source made were imported as dma-buf memory, and
    /// a buffer bound in them, where the device refused the pages' pointer (`dma_buf.rs`).
    pub(crate) fn dma_buf_imported(&self, chunks: usize) {
        self.dma_buf_imports
            .fetch_add(chunks as u64, Ordering::Relaxed);
    }

    /// A `VkDeviceMemory` was imported from a dma-buf of a caller's source and is held: an
    /// image's dedicated import where `of_image`, else a chunk of a buffer's pages (`dma_buf.rs`).
    pub(crate) fn dma_buf_memory_imported(&self, of_image: bool) {
        self.dma_buf_live(of_image).fetch_add(1, Ordering::Relaxed);
    }

    /// `count` such imports were freed.
    pub(crate) fn dma_buf_memories_freed(&self, of_image: bool, count: usize) {
        (self.dma_buf_live(of_image)).fetch_sub(count as u64, Ordering::Relaxed);
    }

    /// The live gauge of dma-buf imports of images where `of_image`, else of buffers' chunks.
    fn dma_buf_live(&self, of_image: bool) -> &AtomicU64 {
        match of_image {
            true => &self.dma_buf_image_memories_live,
            false => &self.dma_buf_memories_live,
        }
    }

    /// A releasable import was released while idle (`host_residency.rs`).
    pub(crate) fn host_import_released(&self) {
        self.host_imports_released.fetch_add(1, Ordering::Relaxed);
    }

    /// A released import was imported again.
    pub(crate) fn host_import_reimported(&self) {
        self.host_imports_reimported.fetch_add(1, Ordering::Relaxed);
    }

    /// A releasable import holds its memory from now on (`resident`), or no longer.
    pub(crate) fn host_import_resident(&self, resident: bool) {
        match resident {
            true => self.host_imports_resident.fetch_add(1, Ordering::Relaxed),
            false => self.host_imports_resident.fetch_sub(1, Ordering::Relaxed),
        };
    }

    /// A committed command buffer's commands use `count` host imports (`Buffer::distinct_host_imports`).
    pub(crate) fn commit_used_host_imports(&self, count: u64) {
        self.host_imports_used.fetch_add(count, Ordering::Relaxed);
    }

    /// A live host import was first read other than by a blit (`read_apart`), or such an import
    /// went.
    pub(crate) fn host_import_read_apart(&self, read_apart: bool) {
        match read_apart {
            true => self.host_imports_read_apart.fetch_add(1, Ordering::Relaxed),
            false => self.host_imports_read_apart.fetch_sub(1, Ordering::Relaxed),
        };
    }

    /// A live host import was first read apart for `reason` (`read_apart`), or such an import
    /// went.
    pub(crate) fn host_import_read_apart_for(&self, reason: ReadApart, read_apart: bool) {
        let gauge = &self.host_imports_read_apart_by[reason as usize];
        match read_apart {
            true => gauge.fetch_add(1, Ordering::Relaxed),
            false => gauge.fetch_sub(1, Ordering::Relaxed),
        };
    }

    /// A rebind's import that no live buffers' table listed yet was made (`unlisted`), or was
    /// first listed, or went unlisted.
    pub(crate) fn host_import_rebound_unlisted(&self, unlisted: bool) {
        let gauge = &self.host_imports_rebound_unlisted;
        match unlisted {
            true => gauge.fetch_add(1, Ordering::Relaxed),
            false => gauge.fetch_sub(1, Ordering::Relaxed),
        };
    }

    /// A rebind imported its new pages, where bytes buffers over host pages lay over already held
    /// them (`covered`) or not.
    pub(crate) fn host_rebind_import_made(&self, covered: bool) {
        self.host_rebind_imports_made
            .fetch_add(1, Ordering::Relaxed);
        if covered {
            (self.host_rebind_imports_covered).fetch_add(1, Ordering::Relaxed);
        }
    }

    /// The caller reported a remap of host addresses (`Device::host_range_remapped`) whose pages
    /// `imports` live imports have a page among.
    pub(crate) fn host_range_remapped(&self, imports: usize) {
        self.host_remaps_reported.fetch_add(1, Ordering::Relaxed);
        if imports > 0 {
            self.host_remaps_over_imports
                .fetch_add(1, Ordering::Relaxed);
            (self.host_imports_remapped).fetch_add(imports as u64, Ordering::Relaxed);
        }
    }

    /// The device imported `bytes` of host pages by their pointer (`Memory::import_host`).
    pub(crate) fn host_pointer_import_made(&self, bytes: u64) {
        (self.host_pointer_imports_made).fetch_add(1, Ordering::Relaxed);
        (self.host_pointer_import_bytes_made).fetch_add(bytes, Ordering::Relaxed);
    }

    /// The caller's import gate refused a no-copy buffer's import (`HostMemory::with_import_gate`).
    pub(crate) fn host_import_refused_by_host(&self) {
        (self.host_imports_refused_by_host).fetch_add(1, Ordering::Relaxed);
    }

    /// A copied host buffer was made (`copied.rs`), for `cause`.
    pub(crate) fn copied_host_buffer_made(&self, cause: CopiedBecause) {
        (self.copied_host_buffers_made).fetch_add(1, Ordering::Relaxed);
        match cause {
            CopiedBecause::Failed => &self.copied_host_buffers_degraded,
            CopiedBecause::Refused => &self.copied_host_buffers_refused,
            CopiedBecause::Moved => &self.copied_rebind_pages_moved,
        }
        .fetch_add(1, Ordering::Relaxed);
    }

    /// A no-copy heap's own buffer was made a copy of its pages (`copied.rs`).
    pub(crate) fn copied_heap_made(&self) {
        (self.copied_heaps_made).fetch_add(1, Ordering::Relaxed);
    }

    /// A rebind gave a buffer a copy of its new pages (`copied.rs`).
    pub(crate) fn copied_rebind_pages_made(&self) {
        (self.copied_rebind_pages_made).fetch_add(1, Ordering::Relaxed);
    }

    /// The device imported a chunk of `bytes` of a guest RAM it uses (`host_ram.rs`).
    pub(crate) fn host_ram_chunk_imported(&self, bytes: u64) {
        (self.host_ram_chunks_imported).fetch_add(1, Ordering::Relaxed);
        (self.host_ram_chunk_bytes).fetch_add(bytes, Ordering::Relaxed);
        (self.host_ram_chunks_live).fetch_add(1, Ordering::Relaxed);
    }

    /// The device freed a chunk of a guest RAM (`host_ram.rs`).
    pub(crate) fn host_ram_chunk_freed(&self) {
        (self.host_ram_chunks_live).fetch_sub(1, Ordering::Relaxed);
    }

    /// A no-copy buffer (or heap's own buffer) was bound in a chunk of a guest RAM
    /// (`host_ram.rs`).
    pub(crate) fn chunk_bound_buffer_made(&self) {
        (self.chunk_bound_buffers_made).fetch_add(1, Ordering::Relaxed);
    }

    /// A rebind made an image again over a copy of its new pages (`copied.rs`).
    pub(crate) fn copied_rebind_image_made(&self) {
        (self.copied_rebind_images_made).fetch_add(1, Ordering::Relaxed);
    }

    /// A commit copied `bytes` the CPU changed into copied host buffers (`copied.rs`).
    pub(crate) fn copied_copied_in(&self, bytes: u64) {
        (self.copied_copy_in_bytes).fetch_add(bytes, Ordering::Relaxed);
    }

    /// A batch's completion merged `bytes` the GPU changed back into the caller's pages
    /// (`copied.rs`).
    pub(crate) fn copied_merged_back(&self, bytes: u64) {
        (self.copied_merge_back_bytes).fetch_add(bytes, Ordering::Relaxed);
    }

    /// A batch's completion dropped `bytes` the GPU changed in copies of pages the process
    /// cannot write, taking them out of the copies (`copied.rs`).
    pub(crate) fn copied_dropped(&self, bytes: u64) {
        (self.copied_dropped_bytes).fetch_add(bytes, Ordering::Relaxed);
    }

    /// A host import the device never releases while idle, for `because`, was made (`kept`), or
    /// went.
    pub(crate) fn host_import_kept(&self, because: KeptBecause, kept: bool) {
        let gauge = &self.host_imports_kept_by[because as usize];
        match kept {
            true => gauge.fetch_add(1, Ordering::Relaxed),
            false => gauge.fetch_sub(1, Ordering::Relaxed),
        };
    }

    /// A read found a released import that the device could not import again: the device is
    /// lost.
    pub(crate) fn host_import_lost(&self) {
        self.host_imports_lost.fetch_add(1, Ordering::Relaxed);
    }

    /// Keep `count` more descriptors of dma-bufs for imports the device may release, where the
    /// kept ones stay within `limit`: whether they do (counted then).
    pub(crate) fn host_import_fds_keep(&self, count: u64, limit: u64) -> bool {
        let kept = &self.host_import_fds_kept;
        let fits = |held: u64| held.checked_add(count).filter(|after| *after <= limit);
        kept.try_update(Ordering::AcqRel, Ordering::Acquire, fits)
            .is_ok()
    }

    /// `count` such descriptors were closed.
    pub(crate) fn host_import_fds_closed(&self, count: u64) {
        self.host_import_fds_kept
            .fetch_sub(count, Ordering::Relaxed);
    }

    /// A no-copy buffer's own import of dma-bufs is kept imported from now on, as the
    /// descriptors to import it again did not fit (`kept`), or such an import went.
    pub(crate) fn host_import_kept_for_fds(&self, kept: bool) {
        match kept {
            true => self
                .host_imports_kept_for_fds
                .fetch_add(1, Ordering::Relaxed),
            false => self
                .host_imports_kept_for_fds
                .fetch_sub(1, Ordering::Relaxed),
        };
    }

    /// A no-copy buffer's (or heap's, or rebind's) `VkBuffer` was assembled from several chunks'
    /// imports as a sparse buffer (`dma_buf.rs`).
    pub(crate) fn dma_buf_sparse_buffer_made(&self) {
        self.dma_buf_sparse_buffers.fetch_add(1, Ordering::Relaxed);
    }

    /// A texture's image over dma-buf pages was made and bound to a dedicated import of a dma-buf
    /// of its own pages (`dma_buf.rs`).
    pub(crate) fn dma_buf_image_made(&self) {
        self.dma_buf_images.fetch_add(1, Ordering::Relaxed);
    }

    /// A texture placed in a heap over dma-buf pages that such an image would alias is a shadow of
    /// the heap's bytes instead: the device made no such image of it (`texture/heap.rs`).
    pub(crate) fn dma_buf_image_fell_back(&self) {
        self.dma_buf_image_fallbacks.fetch_add(1, Ordering::Relaxed);
    }

    /// The driver was asked a format's `DRM_FORMAT_MOD_LINEAR` tiling features (`dma_buf.rs`'s
    /// `modifier_features`).
    #[cfg(unix)]
    pub(crate) fn modifier_features_asked(&self) {
        self.modifier_feature_queries
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A texture over a buffer's bytes was made a shadow of them (`shadow.rs`), its image zeroed
    /// first where `zeroed` (its copies leave an aspect out).
    pub(crate) fn shadowed_texture_made(&self, zeroed: bool) {
        self.shadowed_textures.fetch_add(1, Ordering::Relaxed);
        if zeroed {
            self.shadows_zeroed.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A texture over a buffer's bytes is a shadow of them because they lie in fresh pages the
    /// buffer imported (`fresh_pages.rs`).
    pub(crate) fn shadow_over_fresh_pages(&self) {
        (self.shadows_over_fresh_pages).fetch_add(1, Ordering::Relaxed);
    }

    /// A texture over host pages was made a device copy of them (`texture.rs`).
    pub(crate) fn device_copy_texture(&self) {
        self.device_copy_textures.fetch_add(1, Ordering::Relaxed);
    }

    /// A texture over host pages of `class` was made an image in place over them (`in_place.rs`),
    /// its rows at Metal's pitch in a `DRM_FORMAT_MOD_LINEAR` image made for host allocations
    /// where `pitched`.
    pub(crate) fn host_texture_in_place(&self, class: crate::in_place::PagesClass, pitched: bool) {
        use crate::in_place::PagesClass as P;
        self.host_textures_in_place.fetch_add(1, Ordering::Relaxed);
        match class {
            P::Pointer => {}
            P::DmaBuf => {
                self.dma_buf_textures_in_place
                    .fetch_add(1, Ordering::Relaxed);
            }
            P::Fresh => {
                (self.fresh_page_textures_in_place).fetch_add(1, Ordering::Relaxed);
            }
        }
        if pitched {
            self.pitched_host_images.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A texture over host pages was made a shadow of them for `cause` (`in_place.rs`).
    pub(crate) fn host_texture_shadowed(&self, cause: &crate::in_place::ShadowCause) {
        use crate::in_place::ShadowCause as C;
        let counter = match cause {
            C::Exact => &self.host_texture_shadows_exact,
            C::RenderTarget => &self.host_texture_shadows_render_target,
            C::ShaderWrite => &self.host_texture_shadows_shader_write,
            C::CheckPending(_) => &self.host_texture_shadows_check_pending,
            C::CheckFailed(_) => &self.host_texture_shadows_check_failed,
            C::Unchecked(_) => &self.host_texture_shadows_unchecked,
            C::Refused(refusal, _) => &self.host_texture_shadows_refused[refusal.index()],
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// The device's check of the first move over host pages answered `verdict` for a class it
    /// ran (`in_place_check.rs`).
    pub(crate) fn in_place_check_ran(&self, verdict: crate::in_place::Verdict) {
        use crate::in_place::Verdict as V;
        self.in_place_checks_run.fetch_add(1, Ordering::Relaxed);
        let counter = match verdict {
            V::Passed => &self.in_place_checks_passed,
            V::Failed => &self.in_place_checks_failed,
            V::Unchecked | V::Pending => &self.in_place_checks_unchecked,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// A class's verdict was read from the record of an earlier check (`in_place.rs`).
    pub(crate) fn in_place_check_recalled(&self) {
        (self.in_place_checks_recalled).fetch_add(1, Ordering::Relaxed);
    }

    /// A buffer was made in `bytes` of fresh pages it imported, which read zero unwritten
    /// (`fresh_pages.rs`).
    pub(crate) fn fresh_page_buffer_made(&self, bytes: u64) {
        (self.fresh_page_buffers_made).fetch_add(1, Ordering::Relaxed);
        (self.fresh_page_bytes).fetch_add(bytes, Ordering::Relaxed);
    }

    /// The host set `bytes` of a new buffer to zero so that bytes nothing wrote read zero: a new
    /// buffer's own, as it was made (`Buffer::clear_first`), and an upload region's past its data,
    /// as the arena hands the region out of a chunk made without a clear (`upload.rs`).
    pub(crate) fn new_buffer_cleared(&self, bytes: u64) {
        (self.new_buffer_bytes_cleared).fetch_add(bytes, Ordering::Relaxed);
    }

    /// A submission's first work zeroed a new buffer's bytes the host cannot write with one
    /// `vkCmdFillBuffer` in place of copies of the device's zeros (`texture/first_move.rs`).
    pub(crate) fn new_buffer_zero_filled(&self) {
        (self.new_buffer_zero_fills).fetch_add(1, Ordering::Relaxed);
    }

    /// An encoder planned `n` copies between shadowed textures' truths and their shadows
    /// (`shadow.rs`), whose copies in and back of buffers' textures' images move `texels_in` and
    /// `texels_back` texels.
    pub(crate) fn shadow_copies_planned(&self, n: usize, texels_in: u64, texels_back: u64) {
        self.shadow_copies.fetch_add(n as u64, Ordering::Relaxed);
        self.shadow_texels_copied_in
            .fetch_add(texels_in, Ordering::Relaxed);
        self.shadow_texels_copied_back
            .fetch_add(texels_back, Ordering::Relaxed);
    }

    /// An encoder planned `copy`, a copy in of a shadowed texture: its texels
    /// (`shadow::ShadowCopy::texels_copied_in`) counted by why it is made
    /// (`shadow::ShadowCopy::cause`, a change of the truth by its reason), by what holds the truth
    /// (`shadow::Backing`), apart where the truth is an IOSurface's plane under a texture of at
    /// least [`LARGE_SURFACE_TEXELS`](crate::shadow::LARGE_SURFACE_TEXELS) texels, and apart where
    /// the shadow is an image's, copied image to image. Nothing for any other step.
    pub(crate) fn shadow_copy_in_planned<M>(&self, copy: &ShadowCopy<M>) {
        let Some(cause) = copy.cause else {
            return;
        };
        let texels = copy.texels_copied_in();
        let add = |counter: &AtomicU64| {
            counter.fetch_add(texels, Ordering::Relaxed);
        };
        let copies = &copy.copies;
        if matches!(copies.source, ShadowSource::Image { .. }) {
            add(&self.shadow_texels_copied_in_images);
        }
        let large = copies.large_surface();
        let first = cause == CopyCause::First;
        match cause {
            CopyCause::First => add(&self.shadow_texels_first_copied_in),
            CopyCause::BeyondHeld => add(&self.shadow_texels_recopied_beyond_held),
            CopyCause::Changed(lost) => {
                add(&self.shadow_texels_recopied_after_changes);
                add(match lost {
                    Lost::OtherShadow => &self.shadow_texels_recopied_after_other_shadows,
                    Lost::TwoWays => &self.shadow_texels_recopied_after_two_ways_merges,
                    Lost::Written => &self.shadow_texels_recopied_after_writes,
                    Lost::Within => &self.shadow_texels_recopied_after_writes_within,
                    Lost::Dropped => &self.shadow_texels_recopied_after_dropped_writes,
                });
            }
            CopyCause::Bounded => add(&self.shadow_texels_bounded_copied_in),
        }
        add(match (first, copies.backing) {
            (true, Backing::Image) => &self.shadow_texels_first_copied_in_images,
            (true, Backing::PrivateBuffer) => &self.shadow_texels_first_copied_in_private_buffers,
            (true, Backing::MappedBuffer) => &self.shadow_texels_first_copied_in_mapped_buffers,
            (true, Backing::HostPages) => &self.shadow_texels_first_copied_in_host_pages,
            (true, Backing::AliasedHostPages) => {
                &self.shadow_texels_first_copied_in_aliased_host_pages
            }
            (true, Backing::Surface) => &self.shadow_texels_first_copied_in_surfaces,
            (false, Backing::Image) => &self.shadow_texels_recopied_in_images,
            (false, Backing::PrivateBuffer) => &self.shadow_texels_recopied_in_private_buffers,
            (false, Backing::MappedBuffer) => &self.shadow_texels_recopied_in_mapped_buffers,
            (false, Backing::HostPages) => &self.shadow_texels_recopied_in_host_pages,
            (false, Backing::AliasedHostPages) => {
                &self.shadow_texels_recopied_in_aliased_host_pages
            }
            (false, Backing::Surface) => &self.shadow_texels_recopied_in_surfaces,
        });
        match (large, first) {
            (true, true) => add(&self.shadow_texels_first_copied_in_large_surfaces),
            (true, false) => add(&self.shadow_texels_recopied_in_large_surfaces),
            (false, _) => {}
        }
    }

    /// An encoder planned `n` copies in of the texels only writes within a rectangle changed
    /// since a shadowed texture's copy in (`shadow.rs`, `shadow::Within`).
    pub(crate) fn bounded_shadow_copies_planned(&self, n: usize) {
        self.bounded_shadow_copies
            .fetch_add(n as u64, Ordering::Relaxed);
    }

    /// An encoder recorded a shadowed texture's copy back as a dispatch
    /// (`shadow_copy_back.rs`).
    pub(crate) fn shadow_copy_back_dispatched(&self) {
        self.shadow_copies_back_dispatched
            .fetch_add(1, Ordering::Relaxed);
    }

    /// An encoder recorded a merged shadowed texture's copy in and keep as one dispatch
    /// (`shadow_copy_in.rs`).
    pub(crate) fn shadow_copy_in_dispatched(&self) {
        self.shadow_copies_in_dispatched
            .fetch_add(1, Ordering::Relaxed);
    }

    /// An encoder recorded a heap's tiled texture's keep as a copy of the staging buffer its copy
    /// in just filled (`ShadowCopies::record_kept_from_staging`).
    pub(crate) fn shadow_keep_from_staging(&self) {
        self.shadow_keeps_from_staging
            .fetch_add(1, Ordering::Relaxed);
    }

    /// An encoder recorded a copy of `regions` rows between two buffers (`row_copies.rs`), as one
    /// dispatch where `dispatched`.
    pub(crate) fn row_copy(&self, regions: usize, dispatched: bool) {
        let regions = regions as u64;
        self.row_copies.fetch_add(1, Ordering::Relaxed);
        self.row_copy_regions.fetch_add(regions, Ordering::Relaxed);
        self.row_copy_regions_most
            .fetch_max(regions, Ordering::Relaxed);
        if dispatched {
            self.row_copies_dispatched.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// An encoder planned `n` keeps and merges of shadowed textures whose copies of the shadow
    /// into the merge's scratch are dispatches (`shadow_merge.rs`, `ShadowMerge::dispatches`).
    pub(crate) fn shadow_merge_copies_dispatched(&self, n: usize) {
        self.shadow_merge_copies_dispatched
            .fetch_add(n as u64, Ordering::Relaxed);
    }

    /// A merge planned `dispatches` dispatches of a merge kernel over `words` words in all
    /// (`shadow_merge.rs`, `linear_merge.rs`).
    pub(crate) fn shadow_merge_planned(&self, dispatches: usize, words: u64) {
        self.shadow_merge_dispatches_planned
            .fetch_add(dispatches as u64, Ordering::Relaxed);
        self.shadow_merge_words_examined
            .fetch_add(words, Ordering::Relaxed);
    }

    /// `dispatches` of a merge planned in whole blocks (`shadow_merge.rs`: a command's that replaces
    /// every texel, or a render pass's that stores the tiles of its area).
    pub(crate) fn shadow_merge_in_blocks(&self, dispatches: usize) {
        self.shadow_merge_dispatches_in_blocks
            .fetch_add(dispatches as u64, Ordering::Relaxed);
    }

    /// A merge of a shadowed texture took its scratch from its shadow's spares (`shadow_merge.rs`,
    /// `ScratchSpares`).
    pub(crate) fn shadow_merge_scratch_reused(&self) {
        self.shadow_merge_scratches_reused
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A merge of a shadowed texture made its scratch zero (`shadow_merge.rs`): one not cut.
    pub(crate) fn shadow_merge_scratch_zeroed(&self) {
        self.shadow_merge_scratches_zeroed
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A merge of a shadowed texture fetches the texels after the commands through its shadow's
    /// unsigned-integer view (`shadow_merge.rs`).
    pub(crate) fn shadow_merge_fetched(&self) {
        self.shadow_merges_fetched.fetch_add(1, Ordering::Relaxed);
    }

    /// A merge of a shadowed texture keeps the texels before the commands in an image, `made`
    /// where its queue kept none it could take, else `followed` where a command buffer not yet
    /// completed holds the one it took (`shadow_merge.rs`, `texture/kept_image.rs`).
    pub(crate) fn shadow_merge_kept_in_image(&self, made: bool, followed: bool) {
        self.shadow_merges_kept_in_images
            .fetch_add(1, Ordering::Relaxed);
        if made {
            self.shadow_merge_kept_images_made
                .fetch_add(1, Ordering::Relaxed);
        }
        if followed {
            self.shadow_merge_kept_images_followed
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A merge's kept image was made (`texture/kept_image.rs`).
    pub(crate) fn shadow_merge_kept_image_made(&self) {
        self.shadow_merge_kept_images_live
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A merge's kept image went.
    pub(crate) fn shadow_merge_kept_image_gone(&self) {
        self.shadow_merge_kept_images_live
            .fetch_sub(1, Ordering::Relaxed);
    }

    /// An ended encoder's pass took `passes` shadowed textures as attachments, `first` of them
    /// taken by no pass of its command buffer before (`render/shadows.rs`).
    pub(crate) fn shadow_attachments(&self, passes: usize, first: usize) {
        self.shadow_attachment_passes
            .fetch_add(passes as u64, Ordering::Relaxed);
        self.shadow_attachment_textures
            .fetch_add(first as u64, Ordering::Relaxed);
    }

    /// A render pass took a shadowed texture as a color attachment so (`render/shadows.rs`).
    pub(crate) fn shadow_color_taken(&self, taken: ColorTaken) {
        let counter = match taken {
            ColorTaken::LoadedWithin => &self.shadow_attachment_loads_within,
            ColorTaken::Loaded => &self.shadow_attachment_loads_whole,
            ColorTaken::Cleared => &self.shadow_attachment_clears,
            ColorTaken::ClearedInPart => &self.shadow_attachment_clears_in_part,
            ColorTaken::Untouched => &self.shadow_attachment_loads_untouched,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// A render pass took a shadowed texture as a color attachment it loads, its copies cut to the
    /// rectangle bounding its draws' scissors: of the `bounded` 32-pixel tiles that rectangle
    /// reaches, the rectangles themselves reach `covered` (`render/shadows.rs`,
    /// `clear_reach::DrawnArea::covered_tiles`).
    pub(crate) fn shadow_attachment_tiles(&self, covered: u64, bounded: u64) {
        self.shadow_attachment_tiles_covered
            .fetch_add(covered, Ordering::Relaxed);
        self.shadow_attachment_tiles_bounded
            .fetch_add(bounded, Ordering::Relaxed);
    }

    /// How an encoder's first copies in of shadowed textures stood against their images' last
    /// standing (`shadow::CarryTally`): the texels the carry took off, and, of large IOSurface
    /// planes, the texels a first copy in moved behind an earlier command buffer's standing or
    /// behind none at all. The three large-surface figures size the carry on a run without a
    /// timing: where the carried one is 0 and the other two hold every texel, each large-surface
    /// copy in the run makes is the first in its own command buffer, or of an image that never
    /// held a standing, and the carry takes nothing off it.
    pub(crate) fn shadow_first_copies_in_tallied(&self, tally: CarryTally) {
        let add = |counter: &AtomicU64, texels: u64| {
            if texels > 0 {
                counter.fetch_add(texels, Ordering::Relaxed);
            }
        };
        add(&self.shadow_texels_first_copy_in_carried, tally.carried);
        add(
            &self.shadow_texels_first_copy_in_carried_large_surfaces,
            tally.carried_large,
        );
        add(
            &self.shadow_texels_first_copied_in_large_surfaces_stood_earlier,
            tally.stood_earlier_large,
        );
        add(
            &self.shadow_texels_first_copied_in_large_surfaces_unstood,
            tally.unstood_large,
        );
    }

    /// An ended encoder's first copies in of shadowed textures moved `texels` texels of textures
    /// an earlier encoder of its command buffer copied in too (`shadow.rs`).
    pub(crate) fn shadow_texels_first_copied_in_again(&self, texels: u64) {
        self.shadow_texels_first_copied_in_again
            .fetch_add(texels, Ordering::Relaxed);
    }

    /// An ended encoder used `n` shadowed textures whose truth one it used before shares
    /// (`shadow.rs`).
    pub(crate) fn shadows_sharing_truth(&self, n: usize) {
        self.shadows_sharing_truth
            .fetch_add(n as u64, Ordering::Relaxed);
    }

    /// A command allocated descriptor sets for its dispatches of the runtime's own kernels
    /// (`owned_kernel.rs`).
    pub(crate) fn owned_descriptor_pool(&self) {
        self.owned_descriptor_pools.fetch_add(1, Ordering::Relaxed);
    }

    /// One of the runtime's own kernels was built, of `spirv` words, by the call that first
    /// needed it (`owned_kernel.rs`).
    pub(crate) fn owned_kernel_made(&self, spirv: usize) {
        self.owned_kernels_made.fetch_add(1, Ordering::Relaxed);
        self.owned_kernel_spirv_bytes
            .fetch_add(4 * spirv as u64, Ordering::Relaxed);
    }

    /// A YCbCr texture's conversion took a scratch buffer of `bytes` (`ycbcr.rs`).
    pub(crate) fn ycbcr_conversion_scratch(&self, bytes: u64) {
        self.ycbcr_conversion_scratch_buffers
            .fetch_add(1, Ordering::Relaxed);
        self.ycbcr_conversion_scratch_bytes
            .fetch_add(bytes, Ordering::Relaxed);
    }

    /// An encoder planned keeps of shadowed textures that copy `texels` texels into their merges'
    /// scratch (`shadow_merge.rs`).
    pub(crate) fn shadow_texels_kept(&self, texels: u64) {
        self.shadow_texels_kept.fetch_add(texels, Ordering::Relaxed);
    }

    /// A texture view was made the image itself to its bindings, through a swizzle over the
    /// image's own format (`texture/view_shadow.rs`, `ViewRecord::read_as`).
    pub(crate) fn view_through_swizzle_made(&self) {
        self.views_through_swizzle.fetch_add(1, Ordering::Relaxed);
    }

    /// A texture view's shadow image was made (`texture/view_shadow.rs`): the view's own, or
    /// the render shadow of a view its bindings present through a swizzle.
    pub(crate) fn view_shadow_made(&self) {
        self.view_shadows.fetch_add(1, Ordering::Relaxed);
    }

    /// A draw that samples, at a slot, a texture that presents a color attachment of its render
    /// pass read the texture's copy, where its shadow is served (`texture/attached_reads.rs`).
    pub(crate) fn attachment_read_through_copy(&self) {
        self.attachment_reads_through_copies
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A draw that samples a color attachment's texture after a memory barrier of its render
    /// pass recorded into a new instance (`texture/attached_reads.rs`).
    pub(crate) fn attachment_read_split(&self) {
        self.attachment_read_splits.fetch_add(1, Ordering::Relaxed);
    }

    /// A draw that samples a color attachment's texture after a memory barrier of its render
    /// pass read it through a snapshot inside its instance (`render/snapshot.rs`).
    pub(crate) fn reading_draw_snapshotted(&self) {
        self.reading_draws_snapshotted
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A draw read through a snapshot recorded into a new instance, as its own was recorded
    /// inline (`render/snapshot.rs`).
    pub(crate) fn snapshot_inline_split(&self) {
        self.snapshot_inline_splits.fetch_add(1, Ordering::Relaxed);
    }

    /// The device made a linear image aliasing a buffer's bytes with feedback loop usage
    /// (`Device::storage_over_buffer`, `texture::feedback_candidate`).
    pub(crate) fn feedback_loop_alias_image(&self) {
        self.feedback_loop_alias_images
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A render pass instance holds its multisampled depth/stencil attachment in the attachment
    /// layout (`render_target::depth_layout`).
    pub(crate) fn depth_attachment_moved(&self) {
        self.depth_attachments_moved.fetch_add(1, Ordering::Relaxed);
    }

    /// A render pass instance renders its multisampled depth/stencil attachment in the layout
    /// its encoder holds it in, with no move around it (`texture/held_depth.rs`).
    pub(crate) fn depth_instance_held(&self) {
        self.depth_instances_held.fetch_add(1, Ordering::Relaxed);
    }

    /// A batch moved `count` held depths out of the attachment layout (`texture/held_depth.rs`).
    pub(crate) fn held_depths_released(&self, count: usize) {
        if count > 0 {
            (self.held_depths_released).fetch_add(count as u64, Ordering::Relaxed);
        }
    }

    /// A draw that samples a color attachment's texture reads it at its fragments' own pixels
    /// alone, and its copy in moves the texels around its scissors alone
    /// (`texture/attached_reads.rs`).
    pub(crate) fn own_pixel_read(&self) {
        self.own_pixel_reads.fetch_add(1, Ordering::Relaxed);
    }

    /// A command buffer kept an ended encoder's log of `bytes` bytes (`replay.rs`).
    pub(crate) fn replay_log_kept(&self, bytes: usize) {
        self.replay_logs_kept.fetch_add(1, Ordering::Relaxed);
        self.replay_log_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
    }

    /// A rebind or a map's reset encoded an ended encoder again (`command_buffer/reencode.rs`).
    pub(crate) fn encoder_reencoded(&self) {
        self.encoders_reencoded.fetch_add(1, Ordering::Relaxed);
    }

    /// A rebind recorded again work its command buffer's commit recorded on the GPU alone
    /// (`command_buffer/reencode.rs`).
    pub(crate) fn segment_recorded_again(&self) {
        self.segments_recorded_again.fetch_add(1, Ordering::Relaxed);
    }

    /// A command buffer's first submission pointed `tables` resident root tables of its work on
    /// the GPU alone at the resident set as it lay then (`command_buffer/first_submission.rs`).
    pub(crate) fn replay_tables_pointed(&self, tables: u64) {
        self.replay_tables_pointed
            .fetch_add(tables, Ordering::Relaxed);
    }

    /// A rebind could not encode an ended encoder again, or record again work recorded on the
    /// GPU alone, which keeps its first recording.
    pub(crate) fn reencode_refused(&self) {
        self.reencodes_refused.fetch_add(1, Ordering::Relaxed);
    }

    /// A new image was moved to `GENERAL` and zeroed on the host (`texture/host_copy.rs`).
    pub(crate) fn image_started_on_the_host(&self) {
        self.images_started_on_the_host
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A texture's CPU copy ran on the host (`texture/host_copy.rs`).
    pub(crate) fn host_texel_copy(&self) {
        self.host_texel_copies.fetch_add(1, Ordering::Relaxed);
    }

    /// A texture's host shadow of `bytes` was made at a CPU read of it
    /// (`texture/host_shadow.rs`).
    pub(crate) fn texel_shadow_made(&self, bytes: u64) {
        self.texel_shadows_made.fetch_add(1, Ordering::Relaxed);
        self.texel_shadow_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// A CPU read was laid out of a texture's host shadow, with no submission
    /// (`texture/host_shadow.rs`).
    pub(crate) fn texel_shadow_read(&self) {
        self.texel_shadow_reads.fetch_add(1, Ordering::Relaxed);
    }

    /// A texture's host shadow was dropped: something other than a CPU write may have written
    /// its texels (`texture/host_shadow.rs`).
    pub(crate) fn texel_shadow_dropped(&self) {
        self.texel_shadows_dropped.fetch_add(1, Ordering::Relaxed);
    }

    /// A stand-in of `bytes` was made for textures over detached bytes (`texture/stand_in.rs`).
    pub(crate) fn stand_in_made(&self, bytes: u64) {
        self.stand_ins_made.fetch_add(1, Ordering::Relaxed);
        self.stand_in_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// A stand-in of `bytes` went.
    pub(crate) fn stand_in_released(&self, bytes: u64) {
        self.stand_in_bytes.fetch_sub(bytes, Ordering::Relaxed);
    }

    /// A texture detached after it was resident was announced reachable
    /// (`Degradation::StaleDetachedReach`).
    pub(crate) fn stale_detached_reach(&self) {
        self.stale_detached_reaches.fetch_add(1, Ordering::Relaxed);
    }

    /// A texture over detached bytes whose stand-in has a sink was announced reachable
    /// (`Degradation::DetachedStorageKeepsWrites`).
    pub(crate) fn detached_storage_reach(&self) {
        self.detached_storage_reaches
            .fetch_add(1, Ordering::Relaxed);
    }

    /// `count` texture slots of the device's heap were written again whatever changed
    /// (`heap_sets::renew_over`, `heap_sets::renew`).
    pub(crate) fn heap_slots_rewritten(&self, count: usize) {
        if count > 0 {
            (self.heap_slots_rewritten).fetch_add(count as u64, Ordering::Relaxed);
        }
    }

    /// A write of texture slots waited for a batch's hold of them to end (`heap_sets::hold`).
    pub(crate) fn heap_slot_writes_deferred(&self) {
        (self.heap_slot_writes_deferred).fetch_add(1, Ordering::Relaxed);
    }

    /// A heap's texture was made tiled as Metal lays it out (`texture/heap_tiled.rs`).
    pub(crate) fn heap_texture_tiled(&self) {
        self.heap_textures_tiled.fetch_add(1, Ordering::Relaxed);
    }

    /// A heap's multisample texture's moves took an image for a copy (`texture/copy_images.rs`):
    /// `lent` one their queue kept, else made one.
    pub(crate) fn heap_sample_move_image(&self, lent: bool) {
        let counter = match lent {
            true => &self.heap_sample_move_images_lent,
            false => &self.heap_sample_move_images_made,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// An image was made to ask its memory requirements alone (`texture/requirements.rs`).
    pub(crate) fn requirement_image_made(&self) {
        self.requirement_images_made.fetch_add(1, Ordering::Relaxed);
    }

    /// The device's limits were asked for an image a heap texture's plan asks the memory
    /// requirements of (`texture/requirements.rs`).
    pub(crate) fn requirement_limits_asked(&self) {
        self.requirement_limits_asked
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A descriptor arena made a pool, no spare having room (`descriptor_arena.rs`).
    pub(crate) fn arena_pool_made(&self) {
        self.arena_pools_made.fetch_add(1, Ordering::Relaxed);
    }

    /// A texture in a heap over host pages was met as laid out otherwise than Metal tiles it, as
    /// `kind` (`heap-texture-lossless-layout`, `heap-texture-block-compressed-layout` or
    /// `heap-texture-untiled-layout`; announced where it is the first of its kind).
    pub(crate) fn heap_layout_degraded(&self, kind: Degradation) {
        let counter = match kind {
            Degradation::HeapTextureLosslessLayout => &self.heap_texture_lossless_layouts,
            Degradation::HeapTextureBlockCompressedLayout => {
                &self.heap_texture_block_compressed_layouts
            }
            _ => &self.heap_texture_untiled_layouts,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// The objects counted of `kind`, for a kind met once per object (`Device::degraded_objects`):
    /// its counter's value; `None` for any other kind.
    pub(crate) fn degraded_objects(&self, kind: Degradation) -> Option<u64> {
        let counter = match kind {
            Degradation::StaleDetachedReach => &self.stale_detached_reaches,
            Degradation::DetachedStorageKeepsWrites => &self.detached_storage_reaches,
            Degradation::HeapTextureLosslessLayout => &self.heap_texture_lossless_layouts,
            Degradation::HeapTextureBlockCompressedLayout => {
                &self.heap_texture_block_compressed_layouts
            }
            Degradation::HeapTextureUntiledLayout => &self.heap_texture_untiled_layouts,
            _ => return None,
        };
        Some(counter.load(Ordering::Relaxed))
    }
}

/// How many places [`WorkCounts::copy_sites`] holds.
pub(crate) const COPY_SITES: usize = 32;

/// The places in the runtime that recorded copy commands through a device's functions
/// (`device/functions.rs`), each with what it recorded: the first [`COPY_SITES`] they met; a
/// place past them has its calls counted in [`CopySites::dropped`] alone, so a reader can tell
/// a complete attribution from a truncated one. With the `vkCmdCopyBuffer` regions those
/// functions split.
#[derive(Default)]
pub(crate) struct CopySites(Mutex<Vec<CopySite>>, AtomicU64, AtomicU64);

impl CopySites {
    /// Count a call of `command` of `regions` regions made at `at`.
    pub(crate) fn count(&self, command: &'static str, at: &Location<'static>, regions: usize) {
        let regions = regions as u64;
        let (file, line) = (at.file(), at.line());
        let mut sites = lock(&self.0);
        let same =
            |site: &CopySite| site.line == line && site.command == command && site.file == file;
        match sites.iter().position(same) {
            Some(at) => {
                let site = &mut sites[at];
                site.calls += 1;
                site.regions += regions;
                site.most = site.most.max(regions);
            }
            None if sites.len() < COPY_SITES => sites.push(CopySite {
                command,
                file,
                line,
                calls: 1,
                regions,
                most: regions,
            }),
            // Past [`COPY_SITES`] places: no slot is left for this one, so the call is counted
            // only as dropped. Without this a reader cannot tell the places' calls summing to
            // less than the total from a total the places account for.
            None => {
                self.2.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// The calls made at places past [`COPY_SITES`], which no slot holds: 0 while the places
    /// read account for every call counted here.
    pub(crate) fn dropped(&self) -> u64 {
        self.2.load(Ordering::Relaxed)
    }

    /// Count `regions` regions split in two (`device/functions.rs`).
    pub(crate) fn split(&self, regions: usize) {
        self.1.fetch_add(regions as u64, Ordering::Relaxed);
    }

    /// The regions split so far.
    pub(crate) fn regions_split(&self) -> u64 {
        self.1.load(Ordering::Relaxed)
    }

    /// The places so far, in the order they were met, then empty entries.
    pub(crate) fn read(&self) -> [CopySite; COPY_SITES] {
        let mut read = [CopySite::default(); COPY_SITES];
        for (into, site) in read.iter_mut().zip(lock(&self.0).iter()) {
            *into = *site;
        }
        read
    }
}

/// A place in the runtime that recorded copy commands, and what it recorded there
/// (`WorkCounts::copy_sites`), for the lanes and benches: no Metal counterpart, not part of the
/// API.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CopySite {
    /// The command: `vkCmdCopyBuffer`, `vkCmdCopyBufferToImage`, `vkCmdCopyImageToBuffer` or
    /// `vkCmdCopyImage`; empty in an entry no place took.
    pub command: &'static str,
    /// The runtime's source file that called it.
    pub file: &'static str,
    /// The line there.
    pub line: u32,
    /// The calls made there.
    pub calls: u64,
    /// Their regions.
    pub regions: u64,
    /// The most regions one of them had.
    pub most: u64,
}

/// Every `VkDeviceMemory` block the runtime's own allocator freed, by what made it go
/// (`memory.rs`'s `FreeCause`), counted where the block went: one `vkFreeMemory` each, whether
/// the call ran there or once the batches in flight completed (`freed_memory.rs`). A driver that
/// unmaps a buffer object's pages as it frees it charges that page-table update to the free, so
/// these are the frees such a driver charges the runtime for, and the cause says which of them a
/// change could remove. A field of `WorkCounts`, for the lanes: no Metal counterpart, not part of
/// the API.
///
/// Every block the allocator frees falls under exactly one cause, so `blocks` is its whole share
/// of `ObjectChurn::vk_memory_freed`. What that total holds beyond `blocks` was freed outside
/// the allocator: a dma-buf import, a GPU pool's block (`gpu_pools.rs`), a neutral, a spare, or
/// the rollback of a block or an import that failed to be made.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryFrees {
    /// Blocks freed as the one resource placed in them went: a placement larger than half a
    /// block, or an image bound while work was queued. One guest or runtime object going away,
    /// each with memory of its own, so nothing but a shorter lifetime removes these.
    pub dedicated: u64,
    /// Pooled blocks freed as their last placement left them, their pool already keeping as many
    /// empty ones as it keeps (`memory.rs`'s `KEPT_EMPTY_BLOCKS`). The allocator's own churn: a
    /// working set that dips below its blocks and comes back frees and makes a block each time.
    pub pooled_emptied: u64,
    /// Kept empty pooled blocks freed after their quiet period without a placement
    /// (`memory.rs`'s `KEPT_EMPTY_FOR`): a working set that shrank for good.
    pub kept_stale: u64,
    /// Kept empty pooled blocks freed so that a block of their memory type could be made, as one
    /// failed to be made: the allocator giving memory back under pressure.
    pub kept_for_room: u64,
    /// Imports of the caller's host memory, a dedicated block each, freed as their placement went
    /// (`Memory::import_host`).
    pub host_import: u64,
    /// Imports of a buffer's fresh pages, a dedicated block each, freed as their placement went
    /// (`fresh_pages.rs`).
    pub own_pages: u64,
    /// Blocks freed as the device was destroyed, its queues idle, whatever they held: once each,
    /// at the end of the process's use of the device. 0 in every snapshot a caller can take,
    /// since reading the counts needs the device these run as it goes; the cause exists so that
    /// no block of the allocator's is freed without naming one, and so `blocks` stays its whole
    /// share of `vk_memory_freed` over the device's whole life.
    pub teardown: u64,
    /// The blocks counted above: the allocator's whole share of `vk_memory_freed`.
    pub blocks: u64,
}

/// What a device's commit path did so far (`Device::work_counts`), for the lanes: not part of the
/// API.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorkCounts {
    /// `VkCommandPool`s made for command buffers, where no spare one was free.
    pub command_pools_made: u64,
    /// Released objects MoltenVK commits its residency set for, by kind (`churn_counts.rs`).
    pub object_churn: super::churn_counts::ObjectChurn,
    /// Merged-back shadows' keeps by who planned them (`churn_counts.rs`).
    pub shadow_keeps: super::churn_counts::ShadowKeeps,
    /// The commands recorded into the device's `VkCommandBuffer`s, by kind, as each was recorded
    /// (`recorder`): one per call, the begins and ends among them, whatever the command buffer
    /// (an encoder's, a batch's submission's, the device's own) and the thread; none on a device
    /// past the slots.
    pub commands_recorded: CommandsRecorded,
    /// The places in the runtime that recorded `vkCmdCopyBuffer`, `vkCmdCopyBufferToImage`,
    /// `vkCmdCopyImageToBuffer` and `vkCmdCopyImage` through the device's functions, as encoders
    /// and the runtime's own work recorded them: the first 32 places met, each with its calls,
    /// their regions and the most regions one had. What names the copy behind a recording's time
    /// on a driver that records each region as an operation of its own (`row_copies.rs`).
    pub copy_sites: [CopySite; 32],
    /// The `vkCmdCopyBuffer` regions over 16 KiB whose size is no multiple of 4 KiB that the
    /// device's functions recorded as two, the bytes up to the last whole multiple of 4 KiB and
    /// the rest (`device/functions.rs`), as a driver may record such a region slowly (ANV).
    pub copy_regions_split: u64,
    /// Spare `VkCommandPool`s command buffers took, made by earlier ones. The pool is not reset
    /// where the command buffer that gave it back completed and ended every command buffer of it
    /// it was handed: each is reset as it is begun again.
    pub spare_command_pools_taken: u64,
    /// The spare `VkCommandPool`s among those taken that were reset first, as one of their
    /// command buffers may still have been recording: the command buffer that gave the pool back
    /// failed, was dropped unsubmitted, or left a command buffer of it unended (a recording it did
    /// not use).
    pub spare_command_pools_reset: u64,
    /// Buffers made to upload a live buffers' table to, where no spare one held it.
    pub live_tables_uploaded: u64,
    /// The queues' waiters' waits for a timeline's value, where the batch one waited for had
    /// no fence.
    pub timeline_waits: u64,
    /// The queues' waiters' waits for the fence their queue's front batch signals, where it has
    /// one (`command_buffer/fences.rs`): a batch that ends a command buffer on a driver that
    /// reads the command buffer's objects after its signal, or one of a queue whose failures are
    /// relayed. Such a wait returns later than the timeline's value by whatever the driver does
    /// between the two: MoltenVK signals the fence from the last of its Metal command buffer's
    /// completed handlers, so the host learns of the completion only once those ran.
    pub fence_waits: u64,
    /// The completion waiters' waits that ran out their 100 ms slice although the fence or the
    /// timeline value they waited for was signaled when the waiter looked again: a wake the wait
    /// missed, which finishes its batch a slice late (MoltenVK 1.4.1's wait for any of several
    /// timelines lost them; `command_buffer/completion.rs`), or a signal in the slice's last
    /// moment.
    pub missed_completion_wakes: u64,
    /// The queue waiters' arms for their queue's front batch: one per batch waited for, and one
    /// more wherever a wait ran out its slice and the waiter armed for the same batch again.
    pub completion_arms: u64,
    /// Of those, the arms the waiter slept on its condition variable before: the batch was watched
    /// while its queue's waiter was parked, so a notify and a thread wake sit between the
    /// submission and the start of the wait for its completion, inside the blocked time of a
    /// thread that committed and then waited. An arm that slept none found the batch already
    /// watched.
    pub completion_arms_after_a_wake: u64,
    /// The queue waiters' reads of their timeline that showed a value past the one their wait
    /// returned for, each costing one more `vkWaitSemaphores` before the value is published
    /// (`command_buffer/completion.rs`): a later batch that completed meanwhile, or the front's
    /// after a wait that ran out.
    pub completion_reads_ran_ahead: u64,
    /// Waits for a command buffer's gate that ended (`CommandBuffer::wait_until_completed` and
    /// its timed form), however they ended.
    pub gate_waits: u64,
    /// Of those, the waits that found the gate shut and slept at least once. The rest found the
    /// command buffer already finished and every runtime thread's hold of it gone.
    pub gate_waits_slept: u64,
    /// Notifies of a gate that woke a waiting thread to a gate still shut, which slept on it
    /// again: work the thread that opens the gate did between its notify and the gate opening.
    /// Held to none by `tests::gate`, which is why a gate notifies only where it opened.
    pub gate_wakes_found_shut: u64,
    /// Waits for the trails below a completed command buffer's mark, which its waiters make once
    /// its gate opened (`trails.rs`): one per wait that saw the gate open.
    pub trail_waits: u64,
    /// Of those, the waits that found a trail still open below the mark and blocked on it: what a
    /// runtime thread held to make what the command buffer's work waited for had not gone yet.
    pub trail_waits_blocked: u64,
    /// Command buffers whose transition to `Completed` or `Error` ran on the queue waiter that
    /// saw their work complete, with no handler thread woken: those with no completed handler and
    /// no earlier transition of their queue that runs
    /// the program's code or runs on a handler thread then, while the runtime held none of the
    /// program's objects a release drops or runs (`command_buffer/completion.rs`).
    pub completed_where_seen: u64,
    /// Command buffers whose transition to `Scheduled` ran on the scheduler's thread as their last
    /// batch went, with no handler thread woken: those with no scheduled handler and no
    /// submission handler, whose queue's context held nothing and ran on no thread then
    /// (`CommandBufferShared::schedule`).
    pub scheduled_where_submitted: u64,
    /// Command buffers that took a slot of the device's timestamp pools: only those that encoded
    /// work, on a device that times them by GPU timestamps.
    pub timestamp_slots_taken: u64,
    /// Of `timestamp_slots_taken`, the slots reset on the host as they were taken
    /// (`vkResetQueryPool`), on a device that enables `hostQueryReset` (every device the runtime
    /// creates): no command buffer the GPU runs resets their queries.
    pub timestamp_slots_reset_on_host: u64,
    /// Calibration pairs the device took to place its command buffers' GPU times on the host's
    /// clock (one a commit at most every 100 ms, on a device with calibrated timestamps).
    pub calibrations: u64,
    /// The `vkGetCalibratedTimestampsKHR` calls those pairs took: each pair is read again while
    /// each reading halves the deviation the one before it stated, and while the smallest so far
    /// states more than 20 us, four readings at most.
    pub calibration_readings: u64,
    /// The largest deviation, in nanoseconds, of the pairs taken: how far apart the GPU and host
    /// values of the pair that placed the times may stand, as its call stated.
    pub calibration_deviation_most: u64,
    /// Command buffers whose calibration pair placed a GPU time outside their life (before their
    /// first batch's submission began, or after their completion was seen) by more than the
    /// pair's stated deviation and the clocks' drift allow: a pair that disagrees with the
    /// device's own timestamps. Each was placed by its completion, as on an uncalibrated device.
    pub calibration_disagreements: u64,
    /// Textures that took their storage form: a fetched render target's image made again with
    /// `STORAGE`, once per texture (`texture/storage_form.rs`). Counted by the time the
    /// committing command buffer is scheduled, and by the time a command buffer made meanwhile
    /// is returned.
    pub storage_forms_taken: u64,
    /// `VkBuffer`s made with the device-address usage: one per memory block a buffer with an
    /// address is placed in (`memory.rs`), and one per buffer that takes its own address (a
    /// no-copy buffer's, a preprocess buffer's). None per buffer placed in the runtime's memory:
    /// MoltenVK makes each such buffer resident in every encoder whose pipeline reads pointers.
    pub addressable_buffers_made: u64,
    /// `generate_mipmaps` calls that ran the compute filter, of a format the device's blits
    /// cannot filter, rather than the blit chain (`mipmap_filter.rs`; the chain's blits count in
    /// [`image_blits_recorded`](Self::image_blits_recorded)): the lanes that make mipmaps hold
    /// themselves to them.
    pub mipmaps_filtered: u64,
    /// `generate_mipmaps` calls that read the first level in place through a sampled view, not
    /// copied in (`mipmap_filter.rs`): the lane that measures the fetch holds itself to them.
    pub mipmap_first_levels_fetched: u64,
    /// Draws that handed the making of their `VkPipeline` to the device's pool and went on,
    /// rather than make it at the draw.
    pub pipelines_pooled: u64,
    /// Draws and dispatches whose key's `VkPipeline` was not made, which were recorded at once
    /// with the made one of its residual form, each of the key's uniform-view slots read at its
    /// residual (which reads the same bytes at every bind offset): none waited for its key's
    /// making.
    pub residual_form_draws: u64,
    /// Of the makings draws and dispatches began, those whose command took its key's residual
    /// form meanwhile: makings on the device's pool that nothing waits for.
    pub residual_form_makings: u64,
    /// The pieces the device's pipeline pool holds as the counts are read, waiting or running:
    /// not a count of events, but the pool's state then. A making a draw that took its key's
    /// residual form began may outlast the draw's command buffer, which does not wait for it: a
    /// lane counting what later draws make waits until this is 0.
    pub pipeline_pool_busy: u64,
    /// Stages the device's pipeline creations decoded and specialized before their compiles: a
    /// stage's function, its constants, its linked functions and its dynamic libraries decide the
    /// entry, so however many states are made of one function pair this counts its stages once.
    /// Every check, key and compile of a creation reads the one entry.
    pub stage_entries_specialized: u64,
    /// Asks for a stage entry that found one specialized already, by this creation or by one
    /// before it on the device: the decodes and specializations the device spared.
    pub stage_entries_shared: u64,
    /// Stages' functions the runtime translated to SPIR-V: the lowering itself, counted where a
    /// pipeline creation or a later variant asks for a stage's function and finds it in neither
    /// the device's memo of compiled functions, nor one of the creation's binary archives, nor
    /// the shader cache on disk. A compute kernel, a vertex function and a fragment function each
    /// ask once per stage, so a run of states over one function pair whose lowering inputs come
    /// out the same translates once however many states it makes.
    ///
    /// It is the count between `stage_entries_specialized` and `shader_modules_made`, and neither
    /// of those sees it: the entry is the AIR decode and specialization *before* lowering, which
    /// is shared across states of one function whatever their lowering inputs, and the module is
    /// made from the SPIR-V *after* it, which is shared whenever two translations come out
    /// byte-identical. A state whose descriptor moves a field the function never reads used to
    /// translate again and make nothing: one entry, one module, and a translation this count is
    /// the only record of.
    pub functions_translated: u64,
    /// Asks for a stage's function that found it compiled already, so nothing was translated: in
    /// the device's memo, in one of the creation's binary archives, or in the shader cache on
    /// disk. With `functions_translated` it adds up to every ask the run made.
    pub functions_shared: u64,
    /// Linked-use answers render pipeline creations worked out rather than reading from the
    /// shader cache: whether the vertex function returns a value where the state does not
    /// rasterize, the fragment function's unattached depth write, the stages using a direct
    /// argument, and the fragment inputs no vertex output writes. Each decodes the stage it asks
    /// about.
    pub linked_use_questions_asked: u64,
    /// Render and mesh pipeline descriptor keys the runtime built
    /// (`Device::render_descriptor_key`, `Device::mesh_descriptor_key`). Building one digests
    /// every field of the descriptor in its derived `Debug` form, so it allocates a dozen
    /// strings and hashes each stage's compile key. A pipeline creation builds one, which its
    /// checks, its compiles, its reflection and its state then read; a later variant of a state
    /// builds one of its own, and a `compiledReflection` read builds none.
    pub descriptor_keys_built: u64,
    /// `VkShaderModule`s pipeline states' variants made: none for a state made again whose
    /// released variant's objects were kept (`pipeline/kept.rs`).
    pub shader_modules_made: u64,
    /// Variants' stages that took the device's `VkShaderModule` of their SPIR-V, made before by
    /// another variant or state.
    pub shader_modules_shared: u64,
    /// `VkPipelineLayout`s, each with its set layouts, pipeline states' variants made: none for a
    /// state made again whose released variant's objects were kept (`pipeline/kept.rs`).
    pub pipeline_layouts_made: u64,
    /// Variants that took the device's `VkPipelineLayout` of their layout's exact definition, made
    /// before by another variant or state: their draws keep what the draws before them bound and
    /// pushed with it (`render/bound.rs`).
    pub pipeline_layouts_shared: u64,
    /// `VkPipeline`s pipeline states' variants made (compute regions and their emulated mesh
    /// stages, indirect-bindable replays, and draws): none for one a released variant kept.
    pub variant_pipelines_made: u64,
    /// Nanoseconds the runtime's `vkCreateGraphicsPipelines` calls spent inside the driver, from
    /// the entry to its return (`device/compiles.rs`, the one door every graphics pipeline the
    /// runtime makes on the device goes through): a state's variants and its creation's predicted
    /// key (`pipeline/graphics.rs`), the device's own blit, resolve, copy and interleave
    /// pipelines, and priming's. A call that makes several pipelines counts once, as the one call
    /// it is.
    ///
    /// The clock is read on whichever thread makes the call, so none is dropped and none counted
    /// twice: a program's thread; one of the pipeline pool's, which runs a creation's first piece
    /// on the calling thread and the rest on its own (`pipeline/pool.rs`); or, where the asking
    /// thread is below the user-initiated class on macOS, a thread of the runtime's that makes the
    /// call while the asking thread waits (`device/compiles.rs`). Outside the window, and so the
    /// runtime's rather than the driver's: the start and join of that thread, the building of the
    /// create infos, and the `VkShaderModule`s, which are a Vulkan call of their own before this
    /// one and are shared between creations (`shader_modules_made`, `shader_modules_shared`) --
    /// on MoltenVK 1.4.2 one module is 100-190 us and is not made per creation.
    ///
    /// A render state's creation that makes no `VkPipeline` adds nothing here, a tile state's
    /// among them (its creation makes none at all). A `VkDevice` the runtime adopted twice counts
    /// each making in both its devices' figures.
    pub render_pipeline_create_nanos: u64,
    /// Nanoseconds the runtime's `vkCreateComputePipelines` calls spent inside the driver, read
    /// exactly as [`render_pipeline_create_nanos`](Self::render_pipeline_create_nanos) is and over
    /// the same one door: a compute state's variants and regions, a tile dispatch's, the device's
    /// own blit, fill, mipmap and copy kernels, and priming's. The two are apart because a
    /// dispatch-heavy run and a draw-heavy run ask the driver for different work, and one figure
    /// over both cannot say which.
    pub compute_pipeline_create_nanos: u64,
    /// Nanoseconds the shader cache's directory reads took on the threads that asked for an
    /// artifact (`shader_cache/store.rs`): the `stat` of each question about an entry
    /// (`holds_function`, which a render creation's linking asks once per stage), and the `open`,
    /// `stat` and `read` of each entry read -- a translated function, a compiled compute or render
    /// pipeline, a render descriptor's linked-use answers, its reflection, its variant record, and
    /// a making's driver pipeline data -- with the refresh of a hit's access time and the removal
    /// of an entry a decoder refused.
    ///
    /// The directory's calls alone are in the window. A lookup the device's memo answered never
    /// reaches the directory; one the write queue answered reads no file
    /// (`ShaderCacheStatistics::queued_hits`); and the encode and decode of an entry's bytes lie
    /// outside it, which matters because the kinds differ -- a compute or tile state stores its
    /// artifact on every miss and a later process decodes it, where a render state stores one only
    /// when linking a variant again would translate, so a warm render creation decodes none
    /// (`pipeline/sources.rs`). `ShaderCacheStatistics`' per-kind hits and misses say which kind a
    /// run's lookups walked. These reads are always on a thread that asked -- a program's or one
    /// of the pipeline pool's -- never the cache thread's.
    pub shader_cache_read_nanos: u64,
    /// Nanoseconds the shader cache's directory writes took on the threads that asked for an
    /// artifact: the `mkdir`, the temporary file, its write and flush, and the `rename` over the
    /// entry -- four journalled filesystem calls -- of each entry such a thread wrote itself,
    /// which is each entry counted in
    /// [`shader_cache_writes_on_thread`](Self::shader_cache_writes_on_thread). A creation whose
    /// entry the device's cache thread took adds nothing here.
    pub shader_cache_write_nanos: u64,
    /// Nanoseconds the device's cache thread spent in the directory (`shader_cache/worker.rs`):
    /// the writes it took from the queue, and the scan at device creation and the removals that
    /// keep the directory under its cap. None of this is on a creation's path -- it is here so the
    /// directory's whole cost is attributable and no part of it is dropped -- and it is host CPU
    /// that runs beside the program's threads.
    pub shader_cache_off_thread_nanos: u64,
    /// Entries a thread that asked for an artifact wrote itself rather than hand to the device's
    /// cache thread, because the queue was at one of its bounds or the entry is larger than the
    /// whole bound: the same count as `ShaderCacheStatistics::writes_on_thread`, here because a
    /// cost record carries `WorkCounts` and not that type. It partitions creations by a mechanism
    /// rather than by their cost: on an M3 Max a creation on a fresh key that writes its own entry
    /// costs about 180 us more than one the cache thread took (a median 223.9 us against 42.6 us),
    /// so this is the largest mechanism left in a creation that reaches 200 us, and the time it
    /// took is [`shader_cache_write_nanos`](Self::shader_cache_write_nanos).
    pub shader_cache_writes_on_thread: u64,
    /// Of those, the makings that waited for a twin's making in the air, of the same functions
    /// and specialization, to land before their own (`pipeline/flight.rs`).
    pub flights_followed: u64,
    /// The runs again of predicated loops' layings that a check submitted, the loop still running
    /// past its copies (`command_buffer/rerun.rs`).
    pub predicated_loop_runs_again: u64,
    /// Variants made again that took back a released variant's objects and `VkPipeline`s.
    pub variants_taken_back: u64,
    /// Command buffers a command of which left a live sampler state out of its argument
    /// samplers as it was encoded: more states lived than its pipeline's set-1 sampler arrays
    /// hold. Each runs, a function reading a state left out reading the neutral sampler
    /// (`Degradation::UnlistedArgumentSampler`), and is counted once whether or not its
    /// functions reach the state left out.
    pub unlisted_argument_sampler_buffers: u64,
    /// Draws whose `VkPipeline` the driver did not make after the draw returned (the runtime makes
    /// a draw's pipeline while the draw goes on): each ran nothing, its command buffer running on
    /// without error (`CommandBuffer::dropped_draw` names the first of a command buffer).
    pub draws_dropped_unmade: u64,
    /// Draws whose `VkPipeline` was still being made once their command buffer's bound passed
    /// (`PipelineWaitPolicy::Bounded`, the default: its commit plus the policy's
    /// `pipeline_commit_wait`): each ran nothing, its command buffer running on without error, and the command buffers after it draw with
    /// the pipeline once it is made (`CommandBuffer::dropped_draw` names the first of a command
    /// buffer, with the error `VK_TIMEOUT`).
    pub draws_dropped_still_making: u64,
    /// Compute dispatches and render passes' tile dispatches whose `VkPipeline` was still being
    /// made once their command buffer's bound passed (`PipelineWaitPolicy::Bounded`, the default:
    /// its commit plus the policy's `pipeline_commit_wait`): each ran nothing, so wrote nothing, its command buffer running on
    /// without error, and the command buffers after it dispatch with the pipeline once it is made
    /// (`CommandBuffer::dropped_draw` names the first of a command buffer, with the error
    /// `VK_TIMEOUT`).
    pub dispatches_dropped_still_making: u64,
    /// Immutable sampler sets a command buffer's host step made, counting samplers on the
    /// scheduler's thread, because the device's map held no set of its variant's samplers: the
    /// open break (a render execution's step drawing a state made after its commit, on the variant
    /// for the encoder's sRGB or YCbCr textures, that bakes samplers no variant of the program's
    /// made).
    pub host_step_sampler_sets_made: u64,
    /// `VkPipeline`s states' creations made: each render state's predicted key
    /// (`pipeline/predicted.rs`), and each compute state's predicted region
    /// (`pipeline/dispatch_making.rs`).
    pub creation_pipelines_made: u64,
    /// Of those, the ones a draw or a dispatch took so far: the rest are makings no command has
    /// used yet, the speculative part of making at creation.
    pub creation_pipelines_drawn: u64,
    /// Of [`creation_pipelines_made`](Self::creation_pipelines_made), the render states' predicted
    /// keys (`pipeline/predicted.rs`).
    pub creation_render_pipelines_made: u64,
    /// Of [`creation_render_pipelines_made`](Self::creation_render_pipelines_made), the ones a
    /// draw took so far.
    pub creation_render_pipelines_drawn: u64,
    /// Of [`creation_pipelines_made`](Self::creation_pipelines_made), the compute states'
    /// predicted regions (`pipeline/dispatch_making.rs`).
    pub creation_dispatch_pipelines_made: u64,
    /// Of [`creation_dispatch_pipelines_made`](Self::creation_dispatch_pipelines_made), the ones a
    /// dispatch took so far.
    pub creation_dispatch_pipelines_drawn: u64,
    /// Render and compute states' creations that returned while the making of their predicted
    /// key's `VkPipeline`, which they handed to the device's pool, was still running the policy's
    /// `pipeline_creation_wait` after they handed it there (`PipelineWaitPolicy::Bounded`, the
    /// default): their draws and dispatches take the making in flight, their command buffer
    /// waiting its own bound for it in turn. Each is counted in
    /// [`creation_pipelines_made`](Self::creation_pipelines_made) too once made.
    pub creations_left_making: u64,
    /// Immutable sampler sets variants made (the device's map's, and a host step's or a pool
    /// piece's own): none for a variant without immutable samplers.
    pub sampler_sets_made: u64,
    /// Variants that took the device's set of their immutable samplers, made before by another
    /// variant or state.
    pub sampler_sets_shared: u64,
    /// Immutable sampler sets destroyed while the device lives: a set of the map's once no
    /// variant shares it (a settle), and a host step's or a pool piece's own once nothing holds
    /// it (a sweep or a settle). A set whose creation failed part way was never made, and
    /// counts in neither.
    pub sampler_sets_destroyed: u64,
    /// Render passes commits took ahead for their render passes' host steps (made, or kept by
    /// the device: `render_passes_made` counts those made): the runs' passes of instances the
    /// encoder gave none, and those of instances over memoryless backings.
    pub ahead_render_passes_made: u64,
    /// Render passes the runtime made (`vkCreateRenderPass2`): a device keeps each it makes, by
    /// its shape, operations and layouts, so a pass an encoder begins as one before it makes
    /// none (`render_target::KeptRenderPasses`, up to its count).
    pub render_passes_made: u64,
    /// Framebuffers commits made ahead so: one per instance over memoryless backings.
    pub ahead_framebuffers_made: u64,
    /// Backings memoryless attachments' textures made and keep, one per texture, subresource
    /// and queue whatever the passes (`attachment_stand_in::kept_backing`): the memory a
    /// memoryless attachment costs here where its passes run as several, which Metal's costs
    /// none.
    pub kept_backings_made: u64,
    /// Backings passes made for memoryless color attachments they render into images of their
    /// own (a framebuffer fetch through the pixel interlock reads one, or the pass splits), which
    /// each queue keeps for its later passes of the attachment's shape, a few of them, whatever the
    /// texture (`attachment_stand_in::pooled_backing`): one per shape a queue renders so (and per
    /// attachment of a pass that renders several of one shape so), not one per pass.
    pub pooled_backings_made: u64,
    /// Placeholder images the device made for the color attachments render pass instances leave
    /// out of their subpass for the pixel interlock, which their framebuffers hold in place of
    /// the images the draws read and write as storage images (`storage_placeholder.rs`): one per
    /// format and sample count, and another only where a larger framebuffer, or one of more
    /// layers, needs it, not one per pass.
    pub storage_placeholders_made: u64,
    /// Framebuffer attachments that took a placeholder's view (`storage_placeholder.rs`): one per
    /// left-out color of each framebuffer of a render pass instance whose draws read and write
    /// colors in the pixel interlock.
    pub storage_placeholder_attachments: u64,
    /// Passes that made the descriptor sets and framebuffers of their exact color resolves
    /// (`color_resolve.rs`).
    pub color_resolve_passes_made: u64,
    /// Passes that took the descriptor sets and framebuffers of their exact color resolves from
    /// the image they resolve into, which a pass of the same views made
    /// (`color_resolve::KeptResolves`).
    pub color_resolve_passes_kept: u64,
    /// Nanoseconds commits spent making ahead for their render passes' host steps, kept
    /// backings included.
    pub ahead_nanos: u64,
    /// Driver diagnostics withheld from the device's message sink: messages a driver diagnostic
    /// fact (`DriverFact::pipeline_creation_diagnostics`) measures the driver to report inside a
    /// call the runtime made, reported there (`driver_facts.rs`). MoltenVK 1.4.1's warning of a
    /// pipeline whose restart is dynamic, once per such pipeline it makes.
    pub driver_diagnostics_withheld: u64,
    /// Bytes of `VkDeviceMemory` the runtime allocated so far for memory of its own (its
    /// blocks, pooled or dedicated, freed since or not): none for host memory it imports. A
    /// detach allocates none for the bytes it detaches (`detach.rs`).
    pub device_memory_allocated: u64,
    /// The runtime's live pooled `VkDeviceMemory` blocks with a placement in them (`memory.rs`):
    /// with the blocks below and every dma-buf import, the allocations ANV on i915 lists in each
    /// of its kernel submissions, beside its own buffer objects.
    pub memory_blocks_live: u64,
    /// Its live pooled blocks with no placement, kept for the next.
    pub empty_memory_blocks_live: u64,
    /// The pooled blocks it made so far (`memory.rs`), each a `vkAllocateMemory`.
    pub pooled_memory_blocks_made: u64,
    /// The pooled blocks it freed so far: emptied while its pool kept as many empty ones as it
    /// keeps, kept empty past their quiet period, or freed for a block of their type that
    /// failed to be made (`memory.rs`'s `KEPT_EMPTY_BLOCKS`).
    pub pooled_memory_blocks_freed: u64,
    /// Its live dedicated blocks: a placement larger than half a block, or one placed apart.
    pub dedicated_memory_blocks_live: u64,
    /// Every block its allocator freed so far, by what made it go (`MemoryFrees`): one
    /// `vkFreeMemory` each, and the allocator's whole share of
    /// `object_churn.vk_memory_freed`.
    pub memory_frees: MemoryFrees,
    /// Its live imports of the caller's host memory, a block each (`Memory::import_host`).
    pub host_imports_live: u64,
    /// Its blocks that went while batches submitted before were in flight, on a device whose
    /// driver fails pending work as memory a device-address buffer lay in is freed
    /// (`RuntimeDriverDefect::FreedAddressMemoryFailsPendingWork`): each block's memory waited
    /// for those batches to complete before it was freed (`freed_memory.rs`). 0 elsewhere.
    pub freed_memory_parked: u64,
    /// Of those, the blocks whose memory waits now.
    pub freed_memory_waiting: u64,
    /// Stand-ins made for textures over detached bytes (`texture/stand_in.rs`): one per distinct
    /// image plan among the textures detached at once.
    pub stand_ins_made: u64,
    /// Bytes of memory the live stand-ins hold now.
    pub stand_in_bytes: u64,
    /// Textures detached after a command buffer made their pages resident that a later
    /// submission could reach unnamed, each counted once (`Degradation::StaleDetachedReach`).
    pub stale_detached_reaches: u64,
    /// Textures of storage usage over detached bytes that a later commit could reach unnamed,
    /// each counted once (`Degradation::DetachedStorageKeepsWrites`).
    pub detached_storage_reaches: u64,
    /// Texture slots of the device's heap (its texture table and heap sets) written again by a
    /// rebind or a detach of host pages, or by a change of storage form: for a rebind or a
    /// detach, one for each live texture whose images or texel views lie in those bytes (in a
    /// no-copy heap or over one of its buffers, for the heap's), and none for any other texture.
    pub heap_slots_rewritten: u64,
    /// Writes of texture slots of the device's heap (a new texture's, or those a rebind, a detach
    /// or a change of storage form writes again) that waited for a batch: asked between its
    /// first work's take and its `vkQueueSubmit`, they were made once its submission was made,
    /// before the next batch took its first work. So a texture's making waits for no batch's
    /// submission. A device whose heap sets take no writes while a command buffer is pending
    /// defers none: there a texture's registration waits for the batch instead.
    pub heap_slot_writes_deferred: u64,
    /// Uses of a buffer over host pages that looked up by address the ranges of host bytes their
    /// bytes meet (`host_aliases.rs`): a use of a range met by no more ranges than the classes of
    /// length that lookup visits reads those its range keeps instead.
    pub host_range_lookups: u64,
    /// Textures made in heaps laid out as Metal tiles them, each a shadow of its elements (a
    /// multisample texture's samples) in the heap's bytes (`texture/heap_tiled.rs`): in a heap over
    /// host pages, a depth or stencil texture in any heap on a driver that lays such images apart
    /// from their memory or keeps them GPU-optimized (MoltenVK), and a multisample texture in any
    /// heap whose memory takes no image of it.
    pub heap_textures_tiled: u64,
    /// The images a heap's multisample texture's moves between its image and Metal's tiles made
    /// for a copy, as an encoder records it: each a wide, scratch or draw image of a band of one
    /// aspect and slice, which the copy's later bands, aspects and slices of its shape take again,
    /// and which the queue keeps for its later copies within its bound
    /// (`texture/copy_images.rs`).
    pub heap_sample_move_images_made: u64,
    /// The images such moves took from those their queue kept, made by an earlier copy of any
    /// texture of the queue.
    pub heap_sample_move_images_lent: u64,
    /// The images heap textures' plans made only to ask their memory requirements, destroyed at
    /// once: none for a plan and tiling asked before on a device that enables `maintenance4`,
    /// which fixes the answer (`texture/requirements.rs`).
    pub requirement_images_made: u64,
    /// The device's limits (`vkGetPhysicalDeviceImageFormatProperties2`) heap textures' plans
    /// asked for the images whose memory requirements they ask: none for a plan and tiling whose
    /// requirements were kept (`texture/requirements.rs`), which the limits took already.
    pub requirement_limits_asked: u64,
    /// The descriptor pools the runtime's descriptor arenas made, where no spare of the device's
    /// had room (`descriptor_arena.rs`): those of commands' set 0 and of the runtime's own
    /// kernels' sets (`owned_kernel.rs`).
    pub arena_pools_made: u64,
    /// Textures made in heaps that Metal keeps losslessly compressed, where the runtime's image
    /// lies in or holds the heap's bytes (`Degradation::HeapTextureLosslessLayout`).
    pub heap_texture_lossless_layouts: u64,
    /// Block-compressed textures made in heaps whose placement is not measured
    /// (`Degradation::HeapTextureBlockCompressedLayout`).
    pub heap_texture_block_compressed_layouts: u64,
    /// Other textures made in heaps laid out otherwise than Metal lays them in the heap's bytes
    /// (`Degradation::HeapTextureUntiledLayout`): over host pages, and over the device's memory
    /// in the driver's layout, in rows or as a multisample image (`texture/heap.rs`).
    pub heap_texture_untiled_layouts: u64,
    /// Fills recorded as `vkCmdFillBuffer` (`word_fill.rs`): every fill but on a driver measured
    /// to compile its fill per command pool.
    pub fills_recorded: u64,
    /// Fills recorded as transfers, `vkCmdUpdateBuffer` and copies of what it wrote, on a driver
    /// measured to compile its fill per command pool (`DriverFact::FillCompiledPerCommandPool`,
    /// MoltenVK 1.4.1), which records no `vkCmdFillBuffer`.
    pub fills_by_transfers: u64,
    /// Two-plane YCbCr textures' conversions recorded through their refresh (`ycbcr.rs`): a
    /// comparison of the planes' words with the copy the last conversion read, and the conversion
    /// dispatched through indirect arguments that run no workgroup where no word changed.
    pub ycbcr_refreshes: u64,
    /// Two-plane YCbCr textures' conversions recorded whole: a copy of the planes and a dispatch
    /// over every texel, where the refresh is not made (planes not whole words, or past one
    /// storage binding) or the planes' `VkBuffer` is not the one it binds.
    pub ycbcr_whole_conversions: u64,
    /// Two-plane YCbCr textures made with their planes read in place: a slot whose function reads
    /// the texture only as the in-place form reads binds a uniform texel buffer over the planes'
    /// bytes, and its draws and dispatches convert no image (`texture/ycbcr_image.rs`).
    pub ycbcr_textures_in_place: u64,
    /// Two-plane YCbCr textures that keep their image alone because the device keeps every one's:
    /// a discrete, virtual or other GPU, whose image reads the planes across the bus once per
    /// change where the planes in place would read them at every pass.
    pub ycbcr_images_kept_device: u64,
    /// Two-plane YCbCr textures that keep their image alone because the form reads no such
    /// layout: a format other than `YCBCR8_420_2P` and `YCBCR10_420_2P`, plane 1, a P3 surface.
    pub ycbcr_images_kept_layout: u64,
    /// Two-plane YCbCr textures that keep their image alone because of their swizzle.
    pub ycbcr_images_kept_swizzle: u64,
    /// Two-plane YCbCr textures that keep their image alone because their planes' first byte is
    /// off the device's `minTexelBufferOffsetAlignment` in their buffer.
    pub ycbcr_images_kept_alignment: u64,
    /// Two-plane YCbCr textures that keep their image alone because their planes hold more
    /// elements than the device's `maxTexelBufferElements`.
    pub ycbcr_images_kept_elements: u64,
    /// Two-plane YCbCr textures that keep their image alone because their planes reach past their
    /// buffer, whose bytes past its end Metal reads as zeros.
    pub ycbcr_images_kept_reach: u64,
    /// Two-plane YCbCr textures that keep their image alone because the device views no uniform
    /// texel buffer of their planes' format in a buffer of the runtime's usage.
    pub ycbcr_images_kept_view: u64,
    /// Bindings of two-plane YCbCr textures' planes read in place written into descriptor sets.
    pub ycbcr_planes_bound: u64,
    /// Two-plane YCbCr textures whose planes read in place are also an image the device's own
    /// sampler Y′CbCr conversion samples, a linear image of the multi-planar 4:2:0 format over
    /// their bytes (`texture/ycbcr_planes.rs`): a slot whose every sample the device's sampler
    /// takes samples it through the device's conversion.
    pub ycbcr_planes_images: u64,
    /// Two-plane YCbCr textures whose planes read in place take no such image because the device
    /// does not convert the format with Metal's reconstruction as a linear image of their size (no
    /// such format features, a driver defect in it, an odd size or one past its limits).
    pub ycbcr_planes_images_kept_conversion: u64,
    /// Two-plane YCbCr textures whose planes read in place take no such image because the planes
    /// lie in memory no such image is made over: fresh pages, a no-copy buffer's pages, the
    /// caller's `VkBuffer`.
    pub ycbcr_planes_images_kept_memory: u64,
    /// Two-plane YCbCr textures whose planes read in place take no such image because the driver's
    /// image lays the planes out otherwise than the surface, or does not fit their buffer there.
    pub ycbcr_planes_images_kept_layout: u64,
    /// Bindings of two-plane YCbCr textures' planes images written into descriptor sets.
    pub ycbcr_planes_images_bound: u64,
    /// Two-plane YCbCr textures' conversions, and render shadows' loads, made to sample the
    /// planes' image through the device's own Y′CbCr conversion where it converts their format
    /// (`ycbcr_device.rs`), in place of Metal's arithmetic.
    pub ycbcr_device_conversions: u64,
    /// Samplings of two-plane YCbCr textures' planes' images through the device's own Y′CbCr
    /// conversion recorded: one per conversion or load of such a texture, run through the
    /// refresh's indirect arguments where it has one.
    pub ycbcr_device_samples: u64,
    // The commands MoltenVK 1.4.1 compiles a pipeline for per command pool on Apple's GPUs
    // (`PoolCompiled`), each counted as the runtime records it: a run's survey of which can
    // stall a commit behind a compile storm, as the fill did.
    /// `vkCmdClearAttachments` recorded (`clear_reach.rs`).
    pub attachment_clears_recorded: u64,
    /// `vkCmdBlitImage` recorded (`generate_mipmaps`, `mipmap.rs`).
    pub image_blits_recorded: u64,
    /// `vkCmdCopyQueryPoolResults` recorded (`visibility.rs`).
    pub query_result_copies_recorded: u64,
    /// Visibility result modes set other than `Disabled`, each beginning an occlusion query.
    pub visibility_queries_begun: u64,
    /// `vkCmdDrawIndirect` and `vkCmdDrawIndexedIndirect` recorded (`render.rs`).
    pub indirect_draws_recorded: u64,
    /// Records of indirect command buffers' commands read from their storage and decoded, one a
    /// command each time an execution recorded at its commit, a host step (its check included,
    /// `IndirectCommandCheck`) or `IndirectCommandBuffer::commands` reads it; a setter's own
    /// read of the command it changes is not counted.
    pub indirect_records_read: u64,
    /// Commands of executions of render commands recorded in their GPU form, one a command each
    /// time (`render/executions/gpu.rs`): the device read their records, the host none.
    pub gpu_form_commands: u64,
    /// Of those, the commands drawn by one merged draw per execution rather than one draw each
    /// (`render/executions/gpu.rs`), one a command each time.
    pub merged_draw_commands: u64,
    /// Blocks of the GPU pools made: a queue's index pool block (for its index bounds passes and
    /// index rewrites) or its mesh records pool block, made by the first command buffer of the
    /// queue that needs one where the queue holds none. A queue's command buffers share its block
    /// of each kind, so a queue makes one while it keeps using them. Each block is a sixty-fourth
    /// of the device's largest device-local heap. Those made less those released are the blocks
    /// the device holds.
    pub gpu_pool_blocks_made: u64,
    /// Blocks of the GPU pools released before the device's teardown: a queue's block no command
    /// buffer held for two seconds, released as another lease of its kind ended, or the block of
    /// a queue that went.
    pub gpu_pool_blocks_released: u64,
    /// The most command buffers that held one queue's block of a GPU pool at once: between each
    /// one's first pass that needs it and its completion.
    pub gpu_pool_leases_peak: u64,
    /// Internal submissions made (`device/submit.rs`), each one a host call waits for: a host
    /// copy through the GPU, a clear of a live buffer, a neutral's making.
    pub internal_submissions: u64,
    /// `vkQueueSubmit` and `vkQueueBindSparse` calls the runtime made on the device's queues once
    /// it was made: every batch's (its new resources' first work rides in the same call, first in
    /// the batch's `VkSubmitInfo`, `texture/first_move.rs`), internal and staging submissions (an
    /// internal submission's first work likewise), sparse binds, and fences submitted with no
    /// work. Each is a
    /// submission of its own to the driver; Mesa's drivers merge the `VkSubmitInfo`s of one call
    /// where no signal lies between a command and a later one, and no further (`vk_queue.c`).
    pub queue_submissions: u64,
    /// The `VkSubmitInfo`s those calls held (a sparse bind's none, a call of none with a fence
    /// none). MoltenVK 1.4.1 commits a Metal command buffer of its own for each, and one for a
    /// call of none with a fence (`MVKQueue::submit`), each a submission of its own on Metal's
    /// queue: on an M3 Max about 65 us of a thread of Metal's (`IOGPUMetalCommandQueue
    /// submitCommandBuffers`, the consumer's guest drag), whether or not it holds work.
    pub submit_infos: u64,
    /// The `VkCommandBuffer`s those calls submitted.
    pub command_buffers_submitted: u64,
    /// Nanoseconds the batches' `vkQueueSubmit` calls spent in the driver, from the call to its
    /// return with the `VkQueue` taken (`device/relay.rs`): each batch's or group's one call, and
    /// a command buffer's fence submitted with no work. The thread that submits waits there for
    /// the driver, so a lane that times the scheduler thread's take-up of later commits leaves it
    /// out (`runtime_lane::queue_order`).
    pub batch_submit_nanos: u64,
    /// Empty submissions the scheduler's thread made while idle to take the driver's stall in
    /// one signalling submission of every 1024 to a `VkQueue` then, on a driver that has it
    /// (NVIDIA's; `device/submit_stalls.rs`): the calls left before the stall, where it was due
    /// within the queue's next burst of work.
    pub idle_submits: u64,
    /// Of those, the ones that took the stall: 1.97 ms or more each on the RTX 5080 measured.
    pub submit_stalls_idle: u64,
    /// Signalling submissions of work the runtime took for the driver's stall: 1 ms or more in
    /// the driver, where the count of the queue's calls puts the stall, or anywhere before that
    /// place is confirmed (`device/submit_stalls.rs`). On a driver with the stall above, the stall
    /// where none was taken idle (before the first was seen, or after a use of the queue the
    /// runtime cannot count), and before the place is confirmed, another long call of the
    /// driver's too (a device's first batches).
    pub submit_stalls_on_work: u64,
    /// Work that came while a stall taken idle ran: a submission that waited 10 us or more for
    /// its queue's lock behind it, and a commit that reached the scheduler meanwhile.
    pub submits_behind_idle_stalls: u64,
    /// The longest of those waits, in nanoseconds: a submission's wait for the lock, and for a
    /// commit, the stall it came during (its wait at most).
    pub wait_behind_idle_stall_longest_nanos: u64,
    /// `VkCommandBuffer`s the runtime began once the device was made: each encoder's, a render
    /// encoder's draws' secondary ones, segments', a batch's barrier and first-work ones, internal
    /// and staging submissions'. A driver may clear a command buffer's state as it begins one
    /// (ANV, Mesa 26.2.4: `anv_cmd_state_init` at each begin after a reset). An encoder's own
    /// begun ahead in its spare command pool counts as the encoder takes it
    /// (`encoder_primaries_begun_ahead`).
    pub command_buffers_begun: u64,
    /// Of `command_buffers_begun`, those encoders began as they recorded: each encoder's own, a
    /// segment's, one an encoder went on in, and a render encoder's draws' secondaries. The others
    /// are begun as batches are submitted (their barriers and ends), on whichever thread submits
    /// them, and by the device's own submissions; a lane that reads its encoding thread's steps
    /// counts these alone there.
    pub encoder_command_buffers_begun: u64,
    /// Of `command_buffers_begun`, the encoders' own (begun as they opened, or begun ahead in
    /// their spare pools).
    pub command_buffers_begun_encoders: u64,
    /// Of `command_buffers_begun`, segments' at the commit, untracked segments', and ended
    /// encoders' encoded again.
    pub command_buffers_begun_segments: u64,
    /// Of `command_buffers_begun`, those encoders went on in after a segment or a pass's end.
    pub command_buffers_begun_continuations: u64,
    /// Of `command_buffers_begun`, render encoders' draws' secondaries: the only secondaries the
    /// runtime begins once the device is made.
    pub command_buffers_begun_secondaries: u64,
    /// Of `command_buffers_begun`, batches' work of their own as they are submitted (a sparse
    /// binding's mappings, the first work).
    pub command_buffers_begun_batch_work: u64,
    /// Of `command_buffers_begun`, batches' barriers, transitions and timestamps as they are
    /// submitted.
    pub command_buffers_begun_batch_barriers: u64,
    /// Of `command_buffers_begun`, barrier command buffers recorded to keep.
    pub command_buffers_begun_kept_barriers: u64,
    /// Of `command_buffers_begun`, the device's own submissions' and staging submissions'.
    pub command_buffers_begun_internal: u64,
    /// Of `encoder_command_buffers_begun`, the encoders' own `VkCommandBuffer`s begun ahead: a
    /// settled spare command pool's primary, begun by the thread that completed the command
    /// buffer that gave it back where no thread waited for that one, which the first encoder of
    /// the command buffer that takes the pool records into, begun no more on its thread.
    pub encoder_primaries_begun_ahead: u64,
    /// The spare command pools' primaries begun ahead (`encoder_primaries_begun_ahead`), taken
    /// by an encoder or not yet: not among `command_buffers_begun` until one is taken. Each is
    /// counted once its pool is among the spares again, so a command buffer made after the count
    /// moved can take it.
    pub primaries_begun_ahead: u64,
    /// Barrier command buffers of batches recorded once, kept, and submitted again by later
    /// batches once the batch that last took one completed (`command_buffer/kept_barriers.rs`):
    /// each is among `command_buffers_begun` once.
    pub barrier_buffers_kept: u64,
    /// Of a batch's barrier command buffers, those that also write a timed command buffer's
    /// timestamps (the start after resetting its slot, or the end), recorded once per list of
    /// barriers and slot of the device's timestamp pools and kept: each is among
    /// `command_buffers_begun` once.
    pub stamped_buffers_kept: u64,
    /// The times a batch took such a kept command buffer again, for a later command buffer
    /// holding the same slot, once the batch that last took it completed and the slot's
    /// timestamps were read: no command buffer begun for it.
    pub stamped_buffers_reused: u64,
    /// Batches with work submitted with their end (the barrier to the host, and in a timed
    /// command buffer's last batch its end timestamp) recorded into their last command buffer as
    /// they ended, the end's own command buffer saved (`command_buffer/submit.rs`).
    pub finales_folded: u64,
    /// Batches with work submitted with their end in a command buffer of its own: one whose
    /// event signal waits for another `VkQueue`'s earlier one, leaves its submission a segment or
    /// sparse update, names a texture over the caller's image that moves between layouts, or
    /// whose last command buffer does not go to the queue as recorded.
    pub finales_apart: u64,
    /// `vkQueueSubmit` calls that submitted more than one batch, of command buffers committed on
    /// one queue and found ready together, only the last signalling its queue's timeline
    /// (`CompletionPolicy::Grouped`, `command_buffer/group.rs`): one kernel submission for them all
    /// on Mesa's drivers.
    pub grouped_submissions: u64,
    /// The batches those calls submitted.
    pub batches_grouped: u64,
    /// Groups of batches held for one kernel submission (`CompletionPolicy::Grouped`) that went
    /// holding the most they may (four), each group counted once by what sent it; with the six
    /// fields after this one, every group. Their spread says why a device's groups hold fewer.
    pub group_ends_full: u64,
    /// Groups that went before a batch that waits for an event value a batch they held signals,
    /// which only the group's end meets, or in the call of a batch that waits for anything else
    /// not seen complete, submitted after them: a batch with a wait leads a group, never joins one.
    pub group_ends_wait: u64,
    /// Groups that went before a batch that may not wait in one (it signals a binary semaphore,
    /// takes a fence, binds a sparse update, or moves bytes that wait for every submission) or
    /// is of another `VkQueue`.
    pub group_ends_apart: u64,
    /// Groups that went in the call of a batch that could join but did first work or names a
    /// texture over the caller's image, submitted last.
    pub group_ends_first_work: u64,
    /// Groups that went when the scheduler found no batch left to submit.
    pub group_ends_drained: u64,
    /// Groups that went when the scheduler found no batch ready, with submission checks or host
    /// steps left for its thread to run.
    pub group_ends_checks: u64,
    /// Groups that went when the scheduler found no batch ready, the rest held for events'
    /// values, host steps' gates, binary semaphores or texture decodes.
    pub group_ends_held: u64,
    /// Batches submitted (each held in a group counted as it is held) while no earlier batch of
    /// their `VkQueue` was pending: every one submitted before them the host had seen complete
    /// (their queue's timeline at their values). With the three fields after this one, every
    /// batch, by how far its queue's work ran behind its submission.
    pub batches_with_none_pending: u64,
    /// Batches submitted while one earlier batch of their `VkQueue` was pending.
    pub batches_with_one_pending: u64,
    /// Batches submitted while two or three earlier batches of their `VkQueue` were pending.
    pub batches_with_two_or_three_pending: u64,
    /// Batches submitted while four or more earlier batches of their `VkQueue` were pending.
    pub batches_with_four_or_more_pending: u64,
    /// Groups of batches held for one kernel submission that stayed held past the pass that found
    /// no batch ready, while two or more earlier submissions of their `VkQueue` were not seen
    /// complete, until one was left or the group filled (`command_buffer/group.rs`); each group
    /// counted once, then by what sent it among the group ends.
    pub groups_kept_pending: u64,
    /// Of those, the groups that went at once because a thread began to wait for one of their
    /// command buffers' completion (`CommandBuffer::wait_until_completed`).
    pub groups_sent_for_waits: u64,
    /// Times the scheduler looked for the next batch to submit and found a logical queue's first
    /// committed batch held for a binary semaphore it waits for whose signal is not submitted
    /// (`encode_wait_for_vk_semaphore`). With the five fields after this one, every reason a look
    /// of the scheduler's did not submit a head, each counted at every look that found it, so a
    /// head held through many looks is counted at each: the spread says what holds a device's
    /// submissions back. A head held for more than one reason is counted under each, so these do
    /// not sum to the looks. `batch_hold_nanos` is the time those holds took.
    pub heads_held_for_binary_semaphores: u64,
    /// Heads found held while a rebind encodes their event group's encoders again, or while their
    /// command buffer's first submission is not resolved (`command_buffer/reencode.rs`,
    /// `command_buffer/first_submission.rs`).
    pub heads_held_for_reencodes: u64,
    /// Heads found held for a CPU decode queued into a decoded twin they may read
    /// (`texture/twin_decode.rs`); every head while an untracked texture's is left.
    pub heads_held_for_decodes: u64,
    /// Heads found held because their host step has not been recorded: it waits for a pipeline it
    /// records, or at its gate (`heads_waiting_at_their_gate`), or it is queued for the
    /// scheduler's thread, which runs one at a time between its looks.
    pub heads_held_for_host_steps: u64,
    /// Heads found held because their command buffer's submission checks have not run
    /// (`command_buffer/checks.rs`): the checks wait for an event value to be reached
    /// (`heads_awaiting_check_signals`), or are queued for the scheduler's thread.
    pub heads_held_for_checks: u64,
    /// Heads found held for an event value (`encode_wait_for_event`) that neither a submitted
    /// signal nor the host reaches.
    pub heads_held_for_event_values: u64,
    /// Heads found held behind a batch of an earlier commit on **another logical queue** that
    /// keeps its place and whose footprint runs in commit order with the head's
    /// (`scheduler/order.rs`): Metal tracks a resource across queues in the order it schedules
    /// the command buffers, so a batch the runtime holds for a reason of its own (its host step,
    /// its submission checks) must not let a later commit's conflicting batch go first. This is
    /// the one hold that crosses logical queues and `VkQueue`s, and a batch whose uses are not
    /// all known yet (work a host step or its submission records, `tracker::ordered`'s `None`)
    /// conflicts with every batch that makes a tracked use or names a fence.
    pub heads_held_for_earlier_commits: u64,
    /// Times a look left a head that needs a host step waiting at its gate: the step reads what
    /// the GPU may write, so it waits until every `VkQueue` completed what was submitted to it
    /// when the batch was otherwise ready (a predicated loop's check waits for its own `VkQueue`
    /// alone). Of `heads_held_for_host_steps`.
    pub heads_waiting_at_their_gate: u64,
    /// Times a look left a head's submission checks waiting for a submitted GPU signal to land:
    /// Metal checks a submission only once the event values it waits for are reached, so a value
    /// only a submitted GPU signal reaches is waited for. Of `heads_held_for_checks`.
    pub heads_awaiting_check_signals: u64,
    /// Batches that a look of the scheduler's had found held and that then went (submitted, or
    /// held in a group): each counted once, at its own submission.
    pub batches_held_past_a_look: u64,
    /// Nanoseconds those batches spent held: from the first look that found each held to its
    /// submission. The reasons above say what the time went on, and `scheduler_passes`,
    /// `scheduler_sleeps` and kin whether the scheduler's thread was busy or asleep meanwhile.
    pub batch_hold_nanos: u64,
    /// Registrations of a new resource's first work (as a texture or buffer is made) that found
    /// a submission holding its place in the submissions' order and waited until that submission
    /// was made: only on a device whose heap sets take no writes while a command buffer that binds
    /// them is pending (no `descriptorBindingUpdateUnusedWhilePending`), where a submission holds
    /// the order from its take of the first work to its `vkQueueSubmit`
    /// (`texture/first_move.rs`). Elsewhere a registration takes
    /// no place, and this stays 0.
    pub registrations_waited: u64,
    /// Submissions (a batch's, or one of the device's own) that found another holding its place in
    /// the new resources' first-work order and waited for it (`texture/first_move.rs`). A
    /// submission holds that order from its take of the first work to its `vkQueueSubmit`, and one
    /// of the device's own (`device/submit.rs`: a host copy into or out of a texture, an explicit
    /// clear) holds it while it waits for its own work to complete on the GPU: a batch's
    /// submission then waits here, on the scheduler's thread, before its own `vkQueueSubmit`. On
    /// every device, not only those whose registrations wait (`registrations_waited`).
    pub first_work_order_waits: u64,
    /// Nanoseconds those submissions waited.
    pub first_work_order_wait_nanos: u64,
    /// Times the scheduler's thread found no work and slept (`scheduler/mod.rs`), each until it
    /// was woken or [`scheduler_sleep_slices`](Self::scheduler_sleep_slices) says it timed out.
    pub scheduler_sleeps: u64,
    /// Times a commit woke the scheduler's thread, which slept: it left its batches to the
    /// thread. Each waking here is a notify of its condition variable, or a signal of its nudge
    /// where it slept on the GPU, made only where it slept.
    pub scheduler_wakes_commit: u64,
    /// Times a completion woke it: what a head waited for from the GPU (its gate's values
    /// completed, an event value its checks wait for landed) was met, seen by a completion turn or
    /// by its own wait on the GPU, or the device was lost.
    pub scheduler_wakes_completion: u64,
    /// Times a host signal woke it: an event's value set by the host, a pipeline a host step waits
    /// for made, a texture decode finished, a command buffer encoded again after a rebind, the
    /// rest of a pass a thread about to block left it, a lost device.
    pub scheduler_wakes_host: u64,
    /// Sleeps that ended at the thread's longest wait with nothing having woken it.
    pub scheduler_sleep_slices: u64,
    /// Passes a thread about to block for a command buffer's completion made over the scheduler's
    /// order itself, instead of waiting for the scheduler's thread to make one
    /// (`CommandBuffer::wait_until_completed`, `scheduler/mod.rs`): one where the batches it
    /// committed still waited in the inbox and no pass was under way. Metal encodes and commits
    /// inline on the program's thread; a commit here submits nothing, as Metal's returns before
    /// its command buffer is scheduled, so a thread that commits and then blocks would otherwise
    /// wait for a wake and a whole pass before its work reached the `VkQueue`.
    pub scheduler_passes_by_waiters: u64,
    /// The batches those passes submitted: those ready with nothing for the scheduler's thread to
    /// do for them (no host step, no submission check, no group) and nothing of the program's to
    /// run (no submission or scheduled handler, no switch of a storage form). A pass stops at the
    /// first batch in commit order that is none of these and leaves the rest to the thread, so
    /// this is below `scheduler_passes_by_waiters` where a command buffer needs it.
    pub batches_submitted_by_waiters: u64,
    /// Passes the scheduler's thread made over its order (`scheduler/mod.rs`): one takes in what
    /// was committed and submits what is ready; one more follows a submission check or a host
    /// step the thread ran without its lock, a waker's nudge during a pass, and a pass that
    /// stopped after a few submissions.
    pub scheduler_passes: u64,
    /// Of `scheduler_sleeps`, those on the GPU: in one wait for the event values every head's
    /// checks wait for to land, or for the scheduler's nudge, which a commit, a host signal or a
    /// stop signals (`scheduler/mod.rs`; not on MoltenVK). A value landed wakes it as a
    /// completion, counted in `scheduler_wakes_completion`.
    pub scheduler_gpu_waits: u64,
    /// Notifies of the queue waiters' condition variable (a batch watched, a turn's finish, a
    /// value reached), each a system call: made only where a waiter slept on it
    /// (`command_buffer/completion.rs`).
    pub wakes_completion_waiters: u64,
    /// Notifies of the handler threads' condition variable (a transition or release posted to a
    /// queue's context): made only where a thread slept on it and a context was ready
    /// (`contexts/mod.rs`).
    pub wakes_handler_threads: u64,
    /// Notifies of a command buffer's condition variable (its submission handlers ran, its
    /// scheduled handlers ran, it completed): made only where a thread slept on it.
    pub wakes_command_buffer_waiters: u64,
    /// Notifies of an event's condition variable (its value rose): made only where a host wait
    /// slept on it (`event.rs`).
    pub wakes_event_waiters: u64,
    /// The notifies those four and the scheduler's wakers spared, as no thread slept on the
    /// condition variable (or, for the handler threads, no context was ready).
    pub wakes_spared: u64,
    /// Committed command buffers that gave their queue's slot back as they completed.
    pub slot_returns: u64,
    /// Nanoseconds from the return of their last completed handler to their slot's return, as
    /// Metal's comes back once the handlers ran, before what the commands used is released (a
    /// no-copy buffer's deallocator runs after it: `queue.rs`).
    pub slot_return_nanos: u64,
    /// Makes of a command buffer that waited for a slot of their full queue
    /// (`CommandQueue::command_buffer`).
    pub slot_waits: u64,
    /// Nanoseconds those makes waited.
    pub slot_wait_nanos: u64,
    /// Command queues made on a `VkQueue` an earlier command queue of the device took too, so
    /// that one's command buffers wait for the other's where Metal's would not: the device shares
    /// its `VkQueue`s round robin, so a program that makes more command queues than the device
    /// took `VkQueue`s shares them (`queue.rs`, `device/create.rs`). Zero on a device with a
    /// `VkQueue` for each.
    pub queues_sharing_a_vk_queue: u64,
    /// The waits batches with work made for timelines at values the host had seen them reach,
    /// which make what completed work wrote visible to them: one per timeline whose completed
    /// value moved since a batch of the same `VkQueue` last waited for it.
    pub completed_waits: u64,
    /// Event waits (`encode_wait_for_event`) for values the scheduler found reached as it took
    /// their batch, submitting it or holding it in a group (`scheduler/mod.rs`): by a signal
    /// seen complete, or the host's. Their batch waits for none of them, so it may join a group
    /// (`CompletionPolicy::Grouped`) as a batch with no wait does; the others' waits make it lead
    /// one or go alone (`group_ends_wait`).
    pub event_waits_met: u64,
    /// Event waits (`encode_wait_for_event`) for a shared event's value (`MTLSharedEvent`, whose
    /// value the host may set) that a GPU signal submitted before the batch reaches, held on the
    /// host as their batch led its line rather than handed to its `VkQueue`: Metal's
    /// `setSignaledValue:` releases such a wait as soon as the value is reached, and a Vulkan wait
    /// a batch was submitted with cannot be released by the host, so the batch waits for the value
    /// to be reached and goes with no Vulkan wait (`event_timeline.rs`). Each one chained on the
    /// `VkQueue` before and costs a hold and a `vkQueueSubmit` now: these are the waits whose cost
    /// moved, apart from the `event_waits_met` that were always free. Counted once per batch,
    /// whatever reaches the value (the host's set, or a completion turn seeing the GPU signal).
    /// A plain event's wait (`MTLEvent`, which no host call signals) is never among them: it
    /// chains on the `VkQueue`, where that is exact.
    pub shared_event_waits_held: u64,
    /// Event signals (`encodeSignalEvent:value:`, kept: above every value known before them)
    /// submitted while the device kept a copied host buffer, and so seen at their batch's
    /// completion turn alone, after the copies' merge put the GPU's writes in the caller's pages
    /// (`copied-host-pages`): no read of the event's value observes them earlier, a host wait
    /// returns at that turn, and a wait for them, a plain event's (`MTLEvent`) too, is held on
    /// the host until it rather than chained on the `VkQueue`. Metal's writes before a signal are
    /// in the pages once its value is seen.
    pub event_signals_behind_merge: u64,
    /// Batches with work submitted with no command buffer of barriers before their first
    /// encoder: nothing planned before it, and no start timestamp to write there (none, or one
    /// its first encoder wrote, `WorkCounts::prefixed_starts`).
    pub bare_starts: u64,
    /// Timed command buffers whose first batch took the start its first encoder's own command
    /// buffer wrote as it opened, with no command buffer of barriers before that encoder for the
    /// timestamp (`gpu_time.rs`): those batches start bare where nothing else is planned.
    pub prefixed_starts: u64,
    /// Barriers recorded before a command of an encoder Metal orders (a blit encoder, a serial
    /// compute encoder) for the encoder's earlier commands (`tracker::EncoderUses`): on MoltenVK
    /// each ends the Metal encoder, and the next waits for the one before (`tracker.rs`,
    /// **Inside an encoder**, orders a blit's untracked uses by key and range).
    pub command_barriers: u64,
    /// Render pass instances a pass replaced or gave up before running them (a draw that needed
    /// another shape before any draw recorded, a reroute's instance left behind, a first instance
    /// whose draws record inline instead), whose draws
    /// recording the runtime gave up: ended, so their command pool goes back to the device's spares unreset
    /// (`WorkCounts::spare_command_pools_reset` counts the others). A pass's first instance that
    /// may record inline begins its draws' secondary only where a command records into it, so
    /// one replaced or recording inline before then began none, ended none, and is not counted.
    pub unrun_instances_ended: u64,
    /// Render passes that changed no texel and ended without recording anything: no draw call,
    /// no clear, resolve or visibility result buffer, no depth/stencil state with depth writes
    /// set, every attachment loaded with `Load` or `DontCare` and stored with `Store` or
    /// `DontCare`, which Metal's pass leaves as it was (`render/untouched.rs`).
    pub untouched_passes_skipped: u64,
    /// Render passes whose first render pass instance recorded its draws inline in a primary
    /// command buffer of its encoder's, begun at its first draw, in place of a secondary
    /// (`render/inline_first.rs`).
    pub inline_passes: u64,
    /// Those of them that ended with more than one instance.
    pub inline_passes_split: u64,
    /// The instances after a split of such passes that recorded their draws inline too, in the
    /// same primary command buffer after the instance before, in place of a secondary
    /// (`render/inline_splits.rs`); the others record in secondaries, after them.
    pub inline_split_instances: u64,
    /// The splits of passes whose first instance recorded inline before a draw that writes a
    /// memoryless color attachment, which no instance recorded inline writes: the draw records
    /// in a secondary, after them (`render/inline_first.rs`).
    pub inline_memoryless_write_splits: u64,
    /// Render encoders ended (`RenderCommandEncoder::end_encoding`): each a Metal pass.
    pub render_encoders_ended: u64,
    /// Batches of render encoders' calls logged under `RenderLoweringPolicy::Worker` that the
    /// device's lowering thread lowered, while the program went on.
    pub render_log_batches_by_thread: u64,
    /// Batches of render encoders' logged calls lowered on the program's thread: at an encoder's
    /// end, a read of these counts, a query of the pass or a call lowered at the call, the calls
    /// the lowering thread had not taken; and all of them where the device runs no such thread.
    pub render_log_batches_by_program: u64,
    /// Render encoders' calls that waited for the device's lowering thread, the program having
    /// handed it as many of its encoder's batches as it holds untaken, or as many encoders' ends
    /// as it holds unlowered (the end waiting).
    pub render_log_waits: u64,
    /// Render encoders' ends that `end_encoding` left to the device's lowering thread, which
    /// lowered them after the encoders' calls, while the program went on.
    pub render_log_ends_by_thread: u64,
    /// Of those ends, the ones the program's thread lowered, the thread having not reached them:
    /// at what followed the end in its command buffer (another encoder, an event, the commit),
    /// at a read of these counts, or at a rebind or detach of host pages.
    pub render_log_ends_by_program: u64,
    /// Commits whose own work waited for a render encoder's end the device's lowering thread had
    /// not lowered, or behind such a commit, run on that thread as it lowered the end, while the
    /// program went on (`CommandBuffer::commit` returned at once).
    pub render_log_commits_by_thread: u64,
    /// Of those commits, the ones run on the program's threads: where one lowered the end (a
    /// wait for the command buffer, a read of these counts, a rebind or detach of host pages),
    /// or committed behind a ready one no thread ran.
    pub render_log_commits_by_program: u64,
    /// Render pass instances of render encoders' passes begun where the pass is recorded, each
    /// running a secondary of its draws (`render/pass_steps.rs`): an encoder's instances not
    /// recorded inline, and the runs of executions' commands. With the three fields after this
    /// one, every render pass instance of a render encoder's pass; the rest of
    /// `commands_recorded.cmd_begin_render_pass` are the runtime's own clears, resolves and
    /// copies.
    pub render_passes_begun_instances: u64,
    /// Passes' first instances begun at their first draw, recording their draws inline
    /// (`render/inline_first.rs`).
    pub render_passes_begun_inline: u64,
    /// Instances after a split recorded inline, their draws kept (`render/inline_splits.rs`).
    pub render_passes_begun_inline_splits: u64,
    /// Instances of no draws that resolve a pass after its last (`render/pass_steps.rs`).
    pub render_passes_begun_resolves: u64,
    /// Splits of render encoders' passes, each a new render pass instance after one that
    /// recorded (`render.rs`, **Instances**), at a memory barrier of the program's (`fence.rs`).
    /// With the nine fields after this one, every split, by its first cause in this order.
    pub instance_splits_barrier: u64,
    /// At a tile dispatch, or for the store actions after the pass's last one.
    pub instance_splits_tile_dispatch: u64,
    /// At a draw ordered after draws that reached the tile threadgroup memory.
    pub instance_splits_tile_memory: u64,
    /// At a draw that samples a color attachment's texture after a barrier inside the instance.
    pub instance_splits_attached_read: u64,
    /// At a draw that reads through snapshots apart, or binds a color held in the feedback loop
    /// layout.
    pub instance_splits_feedback: u64,
    /// At a draw that writes a memoryless color after an instance recorded inline.
    pub instance_splits_memoryless_write: u64,
    /// At a draw whose pipeline is being made after an instance recorded inline.
    pub instance_splits_pending_pipeline: u64,
    /// At an indirect command buffer's execution, or a run of its commands.
    pub instance_splits_execution: u64,
    /// At a visibility result mode set in a pass of several views.
    pub instance_splits_visibility: u64,
    /// At a draw the instance's shape does not take: fetched attachments, views, stand-ins or
    /// sample planes arriving, a shader-side stencil or depth test, its colors' places or formats.
    pub instance_splits_shape: u64,
    /// Render passes whose first instance did not record inline, its draws in a secondary command
    /// buffer, by the first check that ruled it out (`render/inline_first.rs`), counted as the pass
    /// opens or at its first draw that records: here, the encoder encodes an ended encoder's calls
    /// again, or the pass has several color groups.
    pub passes_not_inline_encoders: u64,
    /// Of them, those of a layout of no color attachment, or of more than one layer or sample, a
    /// depth or stencil attachment, a visibility buffer, custom sample positions, a
    /// rasterization-rate map, or extended-range shadows that keep their draws.
    pub passes_not_inline_layouts: u64,
    /// Of them, those with a memoryless color attachment not loaded and stored `DontCare`, or
    /// resolved.
    pub passes_not_inline_memoryless: u64,
    /// Of them, those with a color attachment loaded `DontCare`, reaching past the render area as
    /// another attachment does not, cleared to a NaN the device clears apart, resolved, or stored
    /// other than `Store` or late; each counted once more below, by the first such action, color
    /// by color in index order.
    pub passes_not_inline_actions: u64,
    /// Of those, the ones where that action is a `DontCare` load.
    pub passes_not_inline_actions_dont_care_load: u64,
    /// Of those, the ones where it is a reach past the render area that the pass's attachments do
    /// not share: Metal's clear of whole 32-pixel tiles reaches past the render area on one color
    /// attachment, and another is loaded, cleared to another extent (a smaller texture's), or a
    /// smaller memoryless texture. A pass whose attachments share the reach records its first
    /// instance inline over it.
    pub passes_not_inline_actions_unequal_reach: u64,
    /// Of those, the ones where it is a resolve, into a resolve attachment or by a copy after the
    /// pass.
    pub passes_not_inline_actions_resolve: u64,
    /// Of those, the ones where it is a store other than `Store`, or a late one.
    pub passes_not_inline_actions_store: u64,
    /// Of them, those with a color attachment that is a texture over a buffer they do not render
    /// through an image over the buffer's bytes.
    pub passes_not_inline_buffer_textures: u64,
    /// Of them, those with a color attachment of another kind: a stand-in, planes, a clear made
    /// apart, an extended-range format, or a bound linear image (`linear_attachment.rs`).
    pub passes_not_inline_attachments: u64,
    /// Of them, those whose layout allowed it and whose first draw that records reads or writes a
    /// color attachment as storage: a fetch in the pixel interlock, which leaves it out of the
    /// instance's subpass.
    pub passes_not_inline_storage: u64,
    /// Of them, those whose layout allowed it and with whose first draw that records a stand-in
    /// arrives: a fetched attachment's scratch, or a memoryless one's backing.
    pub passes_not_inline_stand_ins: u64,
    /// Of them, those whose layout allowed it and whose first draw that records writes a
    /// memoryless color attachment, which no instance recorded inline writes.
    pub passes_not_inline_memoryless_writes: u64,
    /// Of them, those whose layout allowed it and whose first draw that records did not
    /// otherwise: a draw whose pipeline is being made, a shadow arriving with it, work the end
    /// records inside the instance pending, a feedback loop, a split.
    pub passes_not_inline_draws: u64,
    /// Commits that left their batches to the device's scheduler thread and returned at once,
    /// which submits them in commit order at its next pass (`scheduler/inbox.rs`): every commit,
    /// as no commit submits anything itself.
    pub commits_handed_off: u64,
    /// New `Shared` and `Managed` textures' images moved to `GENERAL` and zeroed on the host, with
    /// no submission (`texture/host_copy.rs`).
    pub images_started_on_the_host: u64,
    /// `replace_region` and `get_bytes` copies that ran on the host, with no submission
    /// (`texture/host_copy.rs`).
    pub host_texel_copies: u64,
    /// Host shadows of textures' texels made at a CPU read of them (`texture/host_shadow.rs`):
    /// one a texture whose CPU reads the driver keeps off the host and that the program reads
    /// back, made by the submission its first read waits for either way. A texture the program
    /// never reads back has none.
    pub texel_shadows_made: u64,
    /// The bytes those shadows hold on the host, each counted as its shadow was made.
    pub texel_shadow_bytes: u64,
    /// CPU reads laid out of a texture's host shadow, with no submission and no wait
    /// (`texture/host_shadow.rs`).
    pub texel_shadow_reads: u64,
    /// Host shadows dropped because something other than a CPU write may have written the
    /// texture's texels (`texture/host_shadow.rs`): a commit of the program's, its own Vulkan
    /// submission, another `VkImage` presented, or a detach.
    pub texel_shadows_dropped: u64,
    /// Submissions on the device's staging queue (`device/staging.rs`), each one a host call
    /// waits for: a `Shared` or `Managed` texture's first move, `replace_region` or `get_bytes`
    /// on a device that copies none on the host from memory the host caches.
    pub staging_submissions: u64,
    /// New `Shared` and `Managed` textures' images with host-transfer usage placed in memory the
    /// host does not cache, where the device has none the image may take
    /// (`texture/host_copy.rs`): their host copies read at uncached memory's speed (a 1080p
    /// `get_bytes` 108 ms on a Radeon 780M), and none elsewhere.
    pub uncached_host_images: u64,
    /// `replace_region` calls into a texture with a decoded twin (`texture/decoded_twin.rs`)
    /// whose region was decoded on the CPU and copied into the twin: every such call whose blocks
    /// were copied on the host or the staging queue, counted once the device's decode thread
    /// finished it (`texture/twin_decode.rs`).
    pub cpu_block_decodes: u64,
    /// GPU decode passes made into decoded twins (`texture/decoded_twin.rs`, `Image::decode_pass`):
    /// a blit's or a copy's into a carried texture, a heap texture's start, a write through
    /// submissions. None for a twin that takes CPU copies, whose texels only the CPU decodes.
    pub block_decode_passes: u64,
    /// 4:2:2 textures' decoded twins made to hold the device's own Y′CbCr conversion of their
    /// pairs, where it converts their format with Metal's reconstruction (`ycbcr_device.rs`), in
    /// place of Metal's decode: their writes all decode through GPU passes.
    pub chroma_device_twins: u64,
    /// GPU decode passes into such twins made (among `block_decode_passes`), each band's pairs
    /// sampled through the device's conversion.
    pub chroma_device_decodes: u64,
    /// ASTC textures' second decoded twins made (`texture/decoded_twin.rs`): one at a texture's
    /// first view in its other colour space (an `_sRGB` texture's as its LDR form, or the other
    /// way), which reads Metal's decode of the blocks in that form.
    pub second_twins: u64,
    /// Blit commands' writes into an ASTC texture's blocks that a submission checked for a
    /// second twin's decode: each such write, whether or not the texture had one.
    pub second_twin_checks: u64,
    /// Decodes into ASTC textures' second decoded twins that submissions recorded: a new twin's
    /// whole decode, a host write's, and a blit write's where the texture had one as the write was
    /// encoded. None where no texture has a second twin.
    pub second_twin_decodes: u64,
    /// Drains of the device's queues: one each time a wait for command buffers' work failed (a
    /// lost device), before those command buffers let go of what they used, so that the
    /// driver's own completion of the work, which can still read their pools, has run
    /// (`command_buffer/completion.rs`).
    pub queue_drains: u64,
    /// Pipelines a depth stand-in's copies run (the depth split, the draw back), made: once per
    /// device each, by the creation of the first render pipeline state whose draws test the depth
    /// themselves on a device that copies the stand-in that way (`depth_stand_in.rs`).
    pub stand_in_copies_made: u64,
    /// Of them, those a draw's encoding made: none where every such state's creation made them.
    pub stand_in_copies_made_at_draws: u64,
    /// Bind shapes draws made of their bind states: none for a draw whose bound slots keep their
    /// shapes and whose other bind state is its encoder's last draw's.
    pub draw_shapes_made: u64,
    /// Bind shapes draws found among those their drawing states settled lately, compared with
    /// their bind states in place: none for a draw whose bound slots keep their shapes and whose
    /// other bind state's shape is its encoder's last draw's (values no shape reads, such as the
    /// depth/stencil state, the stencil references or the blend color, may differ).
    pub draw_shapes_found: u64,
    /// Draws (and commands of other encoders that settle as draws do) that took the device's
    /// live sampler states to list as their argument samplers: none for a draw whose kept bind
    /// state lists them at the count of their slots' changes that still holds, whose variant has
    /// no set-1 arrays and names no state in a notice (`sampler::ArgumentSamplers`).
    pub argument_samplers_taken: u64,
    /// Texture slot views render encoders' binds worked out: none for a bind of a texture, as a
    /// view, that a bind of the same or an earlier encoder of the device bound, while no image of
    /// the device let its views go (`binding::BindMemo`).
    pub bind_views_worked_out: u64,
    /// Stage tables (vertex, fragment, object, mesh) that draws, and commands of other encoders
    /// that settle as draws do, read into their encoder's kept bind state: four a draw at most,
    /// none for a table no call changed since a draw read it through the same drawing state's
    /// interface, with the same samplers' address-mode maps, while no no-copy buffer was rebound
    /// or detached (`live_bind::ReadFrom`).
    pub stage_tables_read: u64,
    /// Those reads that brought in only the slots calls changed since the table was read, the
    /// table keeping them (`ArgumentTable::refresh_changed`): every other slot is as it was read.
    pub stage_tables_read_in_part: u64,
    /// Looks at where a no-copy buffer's bytes lie, each of which takes its pages' lock
    /// (`buffer::HostCell`): a draw's read of a table binding one takes one for the binding, its
    /// base and device address read together (`argument_table::bound_buffer`), and a descriptor
    /// written for it one for its `VkBuffer` and its space's start (`Buffer::space`).
    pub no_copy_lies_read: u64,
    /// Draws whose uses their encoder did not note again: the last command it noted had the same
    /// uses, with the same uses of host aliases added, into the same part, with the same images
    /// held in their attachment layout, and none of them reached through the resident set alone
    /// (`noted_apart.rs`).
    pub commands_noted_as_before: u64,
    /// Draws that settled their variant and plan from their bind shapes: none for a draw whose
    /// bind shape is one of the latest its pipeline state settled in the same buffer forms.
    pub draws_settled: u64,
    /// Draws that ran a variant taking a texture slot bound to a two-plane extended-range texture
    /// (`RGB10A8_2P_XR10` or its sRGB twin), read or sampled: the variant's function reads the
    /// slot's raw texels and decodes them, and filters each linear sample of the slot itself over
    /// four of them, where a draw of another texture samples through the device's sampler.
    pub two_plane_draws: u64,
    /// Draws under Apple's last provoking vertex (`setProvokingVertexMode:` 1) of a pipeline
    /// state that reads it, which ran a `VkPipeline` whose provoking vertex is the last
    /// (`VK_EXT_provoking_vertex`).
    pub provoking_last_pipeline_draws: u64,
    /// Draws under Apple's last provoking vertex, of a pipeline state that reads it, on a device
    /// without it, which drew the list of their primitives a GPU pass (or the CPU) wrote, in
    /// Metal's mode-1 order or led by each primitive's last vertex.
    pub provoking_listed_draws: u64,
    /// Draws drawn as a list (a restarted strip's, a private primitive type's, or one led by its
    /// primitives' last vertices) whose list a run wrote: the index gather pass's (one dispatch
    /// for a run of direct indexed draws over one index buffer, or of indirect draws after one
    /// plan dispatch) or the index rewrite pass's (one dispatch for a run of direct indexed draws
    /// over one index buffer), where a draw's own rewrite pass takes three dispatches.
    pub gathered_draws: u64,
    /// Indexed indirect draws whose index fetch the index bounds pass bounds (no
    /// `robustBufferAccess2`, or a driver not measured to read index 0 past the bound range) that
    /// joined a run of it: one plan dispatch and one fill dispatch for the run's draws, where a
    /// draw's own pass takes two dispatches between three barriers.
    pub bounded_run_draws: u64,
    /// Notices of a variant's that a bind announces at every occurrence
    /// (`DeviceOptions::announce_each`), made for a bind: once per variant until the
    /// specializations or the formats its notices name change; each bind announces them.
    pub bound_notices_made: u64,
    /// Argument tables a dispatch or draw retained what they bind from, taking hold of an object
    /// at least: none for a table whose every object its encoder holds already, from an earlier
    /// retain of one of its tables at whatever slot (`retained::HeldBy`), a table unchanged since
    /// its last retain among them.
    pub argument_tables_retained: u64,
    /// Objects those retains took hold of: none that the encoder holds already.
    pub argument_objects_retained: u64,
    /// Draws that looked their `VkPipeline` up in their variant's memo by key: none for a draw
    /// whose key is one of those its pipeline state's last draws found made, while that entry
    /// stays.
    pub graphics_pipelines_looked_up: u64,
    /// Passes over color attachments the device renders linear only under
    /// `VK_NV_linear_color_attachment`'s rule (bound ones, `linear_attachment.rs`) that rendered
    /// every one in place.
    pub linear_passes_in_place: u64,
    /// Those that rendered one through the image its texture keeps: copied in and its changed
    /// words merged back, or cleared there and copied back.
    pub linear_passes_through_copies: u64,
    /// Passes that rendered a color attachment over a buffer's bytes, which they do not store
    /// (`DontCare`, `MultisampleResolve`), through the image its texture keeps, so that the buffer
    /// keeps its bytes (`kept_bytes.rs`).
    pub kept_bytes_passes: u64,
    /// Passes a render encoder drew past its first for a pass of more color attachments than the
    /// device draws at once, one per later color group, each of that group's colors over a twin
    /// of the depth and stencil (`render/color_groups.rs`).
    pub color_group_passes: u64,
    /// The twins of a depth/stencil attachment such passes made: one per pass of more groups
    /// with a depth/stencil attachment, loaded by a copy or a clear.
    pub color_group_twins: u64,
    /// Draws a later color group recorded nothing for: they write none of its colors, no depth
    /// and no stencil.
    pub color_group_draws_skipped: u64,
    /// State commands draws recorded into their secondary command buffers: pipeline binds,
    /// descriptor set binds, their push constants, vertex and index buffer binds, and dynamic
    /// states. None for a command that sets what its command buffer already holds from an
    /// earlier draw there.
    pub draw_states_recorded: u64,
    /// Draws recorded with their set 0 pushed (`vkCmdPushDescriptorSet`) rather than allocated
    /// from their command buffer's pools, written and bound: those of a render variant whose set 0
    /// a push descriptor set layout can hold, on a device of Vulkan 1.4 or
    /// `VK_KHR_push_descriptor`. The same descriptors at the same bindings either way.
    pub draws_pushing_set_zero: u64,
    /// The writes those draws' pushes carried: a push carries only the writes that differ from
    /// what set 0 holds where it holds the same bindings, pushed with the same layout
    /// (`render/bound.rs`), and none is recorded where none differs.
    pub set_zero_writes_pushed: u64,
    /// Of those draws, the ones whose push carried set 0 whole although set 0 held the
    /// descriptors of another layout of the same shape: equal set-0 layouts (each binding's
    /// number, type, count, stages, baked sampler and flags) and equal push-constant ranges, which
    /// Vulkan holds compatible for set 0 where the layouts are identically defined. Variants share
    /// a layout only where it is identically defined whole (`pipeline/shared_layouts.rs`), so a
    /// switch between layouts of the same shape that differ in a later set, or bake a host step's
    /// own samplers, carries set 0 whole (`render/bound.rs`). Measured, not acted on.
    pub set_zero_equal_shape_switches: u64,
    /// The writes those pushes carried.
    pub set_zero_writes_on_equal_shape_switches: u64,
    /// The writes those pushes would carry if the layouts were one: those that differ from what
    /// set 0 held, and at least one after a pipeline bind where the driver's pushed descriptors go
    /// stale across one (`RuntimeDriverDefect::PushedDescriptorsStaleAfterPipelineBinds`).
    pub set_zero_writes_if_shapes_shared: u64,
    /// Dispatches encoded with their set 0 pushed (`vkCmdPushDescriptorSet`) at the compute bind
    /// point rather than allocated from their command buffer's pools, written and bound: those of
    /// a kernel's variant whose set 0 a push descriptor set layout can hold, on a device of Vulkan
    /// 1.4 or `VK_KHR_push_descriptor`. Every dispatch pushes all of its set's writes.
    pub dispatches_pushing_set_zero: u64,
    /// Draws and dispatches that bound the set 0 their encoder's last command allocated and wrote,
    /// rather than allocating and writing their own: those that write the same descriptors into a
    /// set of the same layout (`binding::LastSetZero`). Their recording binds no set where the set
    /// is bound already.
    pub set_zero_sets_reused: u64,
    /// Draws and dispatches that bound a set 0 an earlier command of their command buffer, in any
    /// of its encoders, allocated and wrote, rather than allocating and writing their own: those
    /// that write the same descriptors into a set of the same layout as one the command buffer
    /// keeps (`binding::SetCache`, two generations of 128 sets), but not their encoder's last
    /// command's (`set_zero_sets_reused`). A uniform or storage buffer the variant binds through
    /// a dynamic descriptor takes its offset as the set is bound, so commands that differ only in
    /// such offsets write the same set.
    pub set_zero_sets_found: u64,
    /// Render encoders' draws that took their shape, what they derive from the encoder's state
    /// alone, from an earlier draw of their encoder that derived it from the same inputs, rather
    /// than deriving it again: their settings, their draw state (fixed state, viewports, scissors,
    /// depth bias) and their `VkPipeline`. The inputs are the drawing state, variant, plan and
    /// instance's compatible pass by identity, the pass's layout, the encoder's state and vertex
    /// amplification and the instance's shape by version, and the assembly, the vertex strides,
    /// the fixed state and the topology by value (`render/draw_shape.rs`). None of a shape whose
    /// derivation announces a notice the device announces at every occurrence
    /// (`DeviceOptions::announce_each`).
    pub draw_shapes_reused: u64,
    /// Compute encoders' dispatches that took the bind state, variant, plan, commands and made
    /// `VkPipeline`s their encoder's last dispatch settled, rather than settling them again:
    /// those of the same pipeline state, dispatch, table (no call changed it since), threadgroup
    /// lengths, imageblock size and stage-in region, in an encoder whose residency did not change
    /// since (no resource made resident, no usage widened, no heap used), of a state that made no
    /// visible-function table, while no no-copy buffer moved, no sampler state was made or went
    /// and no buffer took a resident indirect command buffer's ID (`compute::DispatchMemo`).
    pub dispatches_settled_as_before: u64,
    /// Direct dispatches that settled their bind state again (another table, another residency,
    /// another encoder) into one of the shape their pipeline state's last direct dispatch that
    /// planned planned, of the same dispatch and buffer forms, and took that one's variant and
    /// plan rather than planning them again, and its regions' made `VkPipeline`s where their keys
    /// are the same (`pipeline::Planned`): a dispatch whose buffers' offsets alone moved, and an
    /// encoder's first dispatch of a state dispatched before, for two. Not counted in
    /// `dispatches_settled_as_before`.
    pub dispatches_planned_as_before: u64,
    /// Dispatches of a kernel's variant whose layout bakes immutable samplers (constexpr
    /// samplers'), their pipelines made, recorded in a `VkCommandBuffer` of their own, as a
    /// dispatch whose pipeline is still being made is (`pending_dispatch.rs`): on a device whose
    /// driver loses baked samplers after a copy in the same `VkCommandBuffer`
    /// (`RuntimeDriverDefect::ImmutableSamplersLostAfterCopies`, every lavapipe release). Zero
    /// elsewhere.
    pub baked_sampler_dispatches_apart: u64,
    /// No-copy buffers and heaps released with a deallocator whose deallocator waited past the
    /// release (`host-deallocator-delayed`), counted at each wait: given no keep-alive, for the
    /// import of their pages, which something else held (a command buffer submitted before a
    /// texture over them went, through the texture's image), until the import is freed, or, on a
    /// device whose driver has `RuntimeDriverDefect::FreedAddressMemoryFailsPendingWork`, for the
    /// batches submitted before the import's memory went, until they completed (with a keep-alive
    /// it runs at the release there); or, given none, for a live buffers' table that listed their
    /// pages, which a command buffer that may reach them unnamed held, until it let go of it
    /// (`live_buffers.rs`; with a keep-alive it runs at the release there too).
    pub host_deallocators_delayed: u64,
    /// Imports of host pages (no-copy buffers' and heaps') whose free, with the keep-alive that
    /// waits for it, went to one of the device's handler threads as their last buffer was
    /// released, the release returning at once: where a keep-alive alone waits for the free. A
    /// deallocator waiting for it runs in the release, after it, as Metal runs it there.
    pub host_import_releases_posted: u64,
    /// Those not yet run: their memory not yet freed (or parked, `freed_memory.rs`), their
    /// keep-alives not yet dropped. A program that waits for a keep-alive's drop after its release
    /// waits until this falls to 0.
    pub host_import_releases_pending: u64,
    /// Keep-alives of no-copy buffers and heaps (`KeepAlive`) held now past their release: each
    /// by the import of its pages, whose last buffer it was while something else (a texture's
    /// image a submitted command buffer may still read) held the import, or that a live buffers'
    /// table a command buffer that may reach them unnamed holds listed as it was released, until
    /// the import is freed. One per import at most: a later keep-alive over the same import is
    /// dropped at its release (`host_keep_alives_covered`).
    pub host_keep_alives_kept: u64,
    /// Keep-alives dropped at their buffer's release, after its deallocator ran, because the
    /// import of its pages stays mapped without them: another no-copy buffer made over the same
    /// bytes lies in it, or it keeps another's keep-alive already.
    pub host_keep_alives_covered: u64,
    /// The process's mappings the kernel was asked about, for their write permission, before a
    /// no-copy buffer's, heap's or rebind's own import of host pages, on Linux (one `PROCMAP_QUERY`
    /// each, or one line of `/proc/self/maps` where the kernel predates the query): one per mapping
    /// the import's pages lie in, a view of scattered pages one per run, at each such import. A
    /// buffer made over the same bytes as a live one, which lies in its import, asks about none.
    /// None off Linux.
    pub host_mappings_queried: u64,
    /// Chunks of host pages imported as dma-buf memory from a caller's source
    /// (`HostMemory::with_dma_buf`), where the device refused the pages' pointer: no-copy
    /// buffers', heaps' and rebinds' alike, one per chunk. None where every pointer was imported.
    pub dma_buf_imports: u64,
    /// Those chunks' imports live now: each a `VkDeviceMemory` the device holds until the last
    /// buffer, texture or command buffer over its pages lets go. RADV lists each, as every
    /// imported memory, in every kernel submission of the device (Mesa 26.2.4's global BO list),
    /// and amdgpu's work per submission grows with them (`runtime_lane::submit_per_import`).
    pub dma_buf_memories_live: u64,
    /// The dedicated dma-buf imports of images over such memory live now (`dma_buf_images`): one
    /// per live image, listed in every kernel submission as those above.
    pub dma_buf_image_memories_live: u64,
    /// No-copy buffers' imports of dma-bufs released while idle, their `VkBuffer` destroyed and
    /// their memory freed, the dma-bufs kept: a runtime-internal residency, which nothing of
    /// Metal's observes (`HostMemory::with_dma_buf` documents it). Their chunks leave
    /// `dma_buf_memories_live`.
    pub host_imports_released: u64,
    /// Released imports imported again from the dma-bufs kept, each at the first blit that read
    /// the buffer after its release (on the encoding thread).
    pub host_imports_reimported: u64,
    /// The imports the device may release that hold their memory now.
    pub host_imports_resident: u64,
    /// Reads other than a blit's that found an import released and the device unable to import
    /// it again: the device was lost then. None but out of memory.
    pub host_imports_lost: u64,
    /// The descriptors of dma-bufs the device keeps open now to import released imports again: at
    /// most a quarter of the process's descriptor limit.
    pub host_import_fds_kept: u64,
    /// The live no-copy buffers' own imports of dma-bufs that the device would release while idle
    /// but keeps imported, as the descriptors to import them again did not fit: past the quarter
    /// of the process's `RLIMIT_NOFILE` soft limit the devices keep, or the host gave no duplicate.
    /// Each stays in every kernel submission's list while it lives. A soft limit raised at startup
    /// (`setrlimit`; the runtime reads it once, at the process's first such import) lets more go;
    /// the runtime never changes the limit.
    pub host_imports_kept_for_fds: u64,
    /// The host imports each committed command buffer's commands use (a binding, a blit's operand,
    /// `use_resource`, a texture over the bytes), each import counted once per command buffer
    /// however many of its buffers lie in it, summed over command buffers: its change over a frame,
    /// against `host_imports_live`, is how many of the imports every kernel submission lists the
    /// frame's commands name. Counted at the commit, on the committing thread.
    pub host_imports_used: u64,
    /// The live host imports some read other than a blit's took (their Metal address or
    /// `VkBuffer` handed out, a binding, a texture over the bytes, a second buffer lying in the
    /// import): those a release while idle could never take, as the device keeps a releasable
    /// import for good after such a read. Of imports of any kind.
    pub host_imports_read_apart: u64,
    /// Of those, the ones whose buffer's Metal address was handed out: `gpu_address` (a heap's
    /// over the import's bytes too), or written for the caller into an argument buffer or an
    /// indirect command buffer. Each of these six counts an import once for its reason, so an
    /// import read apart for several reasons counts in each.
    pub host_imports_read_apart_address: u64,
    /// Of those, the ones whose buffer's `VkBuffer` was handed out (`Buffer::vk_buffer`).
    pub host_imports_read_apart_vk_buffer: u64,
    /// Of those, the ones a command read other than by a blit's copy: a binding (a buffer
    /// argument, a vertex or index buffer, an argument table's entry, `use_resource`), and the
    /// copies of a texture over the bytes that a command reads it through.
    pub host_imports_read_apart_binding: u64,
    /// Of those, the ones a texture over the bytes was made over: an image bound to the import's
    /// memory, or a texel view or a shadow's copies over its `VkBuffer`.
    pub host_imports_read_apart_texture: u64,
    /// Of those, the ones a second no-copy buffer lies in (`host_aliases.rs`), or that a buffer
    /// made over the same host memory shares its range with (`BufferDescriptor::range_of`).
    pub host_imports_read_apart_second_buffer: u64,
    /// Of those, the releasable ones kept imported because the host could start no thread to
    /// release them.
    pub host_imports_read_apart_unreleased: u64,
    /// The caller's reports of host addresses whose pages it replaced
    /// (`Device::host_range_remapped`), whatever lies over them.
    pub host_remaps_reported: u64,
    /// Of those, the reports with a page of a live host import among the addresses, rounded out
    /// to the import's unit: a driver's import by pointer (ANV's userptr on i915) over a page
    /// replaced takes its pages anew, all of them, at its next submission.
    pub host_remaps_over_imports: u64,
    /// The live host imports those reports touched, summed over the reports: each a range of
    /// bytes no-copy buffers or a no-copy heap lie over (two imports only where buffers of two
    /// memory classes lie over the same bytes).
    pub host_imports_remapped: u64,
    /// The host imports the device made by pointer (`VK_EXT_external_memory_host`): no-copy
    /// buffers', heaps', rebinds' and releasable ones' made again. A driver takes an import's
    /// pages at its first submission (ANV's userptr on i915).
    pub host_pointer_imports_made: u64,
    /// Their bytes: whole pages, rounded out to the import's unit.
    pub host_pointer_import_bytes_made: u64,
    /// The no-copy buffers' imports the caller's gate refused (`HostMemory::with_import_gate`):
    /// each made a copied host buffer instead (`copied_host_buffers_refused`).
    pub host_imports_refused_by_host: u64,
    /// The copied host buffers made (`copied-host-pages`): no-copy buffers whose bytes the GPU
    /// uses in a copy of the caller's pages in memory of the device's own, which no import maps.
    /// Each is one of the next three.
    pub copied_host_buffers_made: u64,
    /// Of those, the ones made because the caller's gate refused the import.
    pub copied_host_buffers_refused: u64,
    /// Of those, the ones made because an import the device and the gate admitted failed (out
    /// of memory, another Vulkan error).
    pub copied_host_buffers_degraded: u64,
    /// Of those, the ones a rebind made because the new bytes lie at another offset in their
    /// page than the old, where a texture over the buffer was planned at the old offset
    /// (`Buffer::rebind_bytes_no_copy`): the copy keeps the bytes at the old offset.
    pub copied_rebind_pages_moved: u64,
    /// Of the copies made, those a rebind gave a buffer as its new pages
    /// (`Buffer::rebind_bytes_no_copy`), in `copied_host_buffers_made` and its kin too.
    pub copied_rebind_pages_made: u64,
    /// The images rebinds made again over such a copy (`Buffer::rebind_bytes_no_copy`,
    /// `Heap::rebind_bytes_no_copy`): a heap's textures bound to its bytes, and the images in place
    /// of textures over the buffer, each a linear image of the copy's memory at its texels.
    pub copied_rebind_images_made: u64,
    /// Of the copies made, the no-copy heaps' own buffers (`Device::new_heap_with_bytes_no_copy`),
    /// in `copied_host_buffers_made` and its kin too: the heap's buffers lie in the copy, and its
    /// textures are shadows of its bytes.
    pub copied_heaps_made: u64,
    /// The bytes the CPU changed in copied host buffers' pages that commits copied into the
    /// copies.
    pub copied_copy_in_bytes: u64,
    /// The bytes the GPU changed in copied host buffers' copies that batches' completions merged
    /// back into the caller's pages.
    pub copied_merge_back_bytes: u64,
    /// The bytes the GPU changed in copies of pages the process cannot write (read-only pages)
    /// that batches' completions dropped: taken out of the copies, the pages' bytes in their
    /// place, never written to the pages, as Metal drops the GPU's writes to read-only memory.
    pub copied_dropped_bytes: u64,
    /// The chunks of guest RAM the device imported (`Device::use_host_ram`): 64 MiB of a RAM's
    /// range each (its last fewer), imported the first time a no-copy object over it was made.
    pub host_ram_chunks_imported: u64,
    /// Their bytes.
    pub host_ram_chunk_bytes: u64,
    /// The chunks of guest RAM imported and not yet freed (a gauge): a chunk is freed once the
    /// device's use of its RAM ended and nothing holds it.
    pub host_ram_chunks_live: u64,
    /// The no-copy buffers (and no-copy heaps' own buffers) bound in a chunk of guest RAM, whose
    /// pages their caller resolved to one run inside one chunk (`HostMemory::with_runs`): each
    /// made with no import of its own, its import gate not asked.
    pub chunk_bound_buffers_made: u64,
    /// The live host imports the device never releases while idle as a heap's own import (a
    /// no-copy heap's). With the next three and `host_imports_kept_for_fds`, they are every live
    /// no-copy import the device keeps imported whatever reads it.
    pub host_imports_kept_of_heap: u64,
    /// The live host imports a rebind gave a buffer (`rebind.rs`): never released while idle.
    pub host_imports_kept_rebound: u64,
    /// Of those, the ones no live buffers' table listed yet (`live_buffers.rs`): no command
    /// buffer committed since the rebind used a buffer lying in them, so no command reaches them
    /// unnamed, as Metal reaches a rebound buffer only once a command buffer used it again.
    pub host_imports_rebound_unlisted: u64,
    /// The imports rebinds made of their new pages (`Device::rebind_host_pages`): not those that
    /// lie in the import of a live buffer made over the same bytes (`host_aliases.rs`).
    pub host_rebind_imports_made: u64,
    /// Of those, the ones whose pages, rounded out to the import's unit, lay within the bytes,
    /// rounded out likewise, that a buffer over host pages lay over already (a no-copy buffer
    /// made or rebound there, the rebound buffer itself included, or a no-copy heap's own
    /// buffer): an import alive then held each of their pages.
    pub host_rebind_imports_covered: u64,
    /// The live no-copy buffers' imports of host pages by their pointer, never released while
    /// idle as the device's driver is not measured to read the pages mapped at the pointer at
    /// each submission (`DriverFact::PointerImportsFollowTheirAddress`), or off Unix.
    pub host_imports_kept_pointer: u64,
    /// The live no-copy buffers' imports the device would release while idle, but keeps, as it
    /// could make no stand-in buffer for a read of a released one.
    pub host_imports_kept_no_stand_in: u64,
    /// Of those buffers, heaps and rebinds, the ones whose `VkBuffer` is a sparse buffer
    /// assembled from several chunks.
    pub dma_buf_sparse_buffers: u64,
    /// Images of textures placed in a no-copy heap over such dma-buf memory (or a view of one),
    /// each a `DRM_FORMAT_MOD_LINEAR` image bound to a dedicated import of a dma-buf of its own
    /// pages from the caller's source, aliasing the heap's bytes: one per image made, a rebind's
    /// too. Not counted in `dma_buf_imports`. A texture over a no-copy buffer's pages takes no such
    /// image: it is a shadow (`shadowed-texture`).
    pub dma_buf_images: u64,
    /// Textures placed in such a heap that such an image would alias but the device made none
    /// (the source failed, the format, usage, extent or plane layout is not supported, or the
    /// driver refused the import): each is a shadow of the heap's bytes in memory of its own
    /// (`shadowed-texture`).
    pub dma_buf_image_fallbacks: u64,
    /// Formats whose `DRM_FORMAT_MOD_LINEAR` tiling features the driver was asked, for an image
    /// over dma-bufs: each format once at most, as the answer is kept.
    pub modifier_feature_queries: u64,
    /// Textures over a buffer's bytes (a 2D texture or array over a buffer, a no-copy buffer or an
    /// IOSurface plane, or a view of another layout of one) that no image aliases, each an image
    /// of its own that the encoders copy the bytes into and back (`shadowed-texture`).
    pub shadowed_textures: u64,
    /// Of `shadowed_textures`, those whose image was zeroed as it was made: its copies leave an
    /// aspect out (a depth texture over a surface's plane, whose image's stencil reads zero).
    /// Every other shadow's copy in writes each texel before any command reads it, so its image
    /// starts as its memory lies.
    pub shadows_zeroed: u64,
    /// Of `shadowed_textures`, those over a buffer that lies in fresh pages it imported
    /// (`fresh_page_buffers_made`): an image bound to imported memory starts `UNDEFINED`, and its
    /// first move may write any byte bound to it.
    pub shadows_over_fresh_pages: u64,
    /// Textures over host pages (a no-copy buffer's, a no-copy heap's buffer's, or a buffer in
    /// fresh pages it imported) that only sample or read, made an image aliasing the pages in
    /// place, where the device's check passed for their class (`HostPageTexturePolicy::InPlace`):
    /// no shadow, and no copy in or back.
    pub host_textures_in_place: u64,
    /// Textures over host pages that `host_textures_in_place` would count (or a shadow made
    /// before the check answered or where it failed), made instead a copy in the device's memory
    /// on a discrete GPU, where their memory carries a log of its writes
    /// (`HostMemory::with_dirty_pages`): a shadow whose copies in take only the rows the log
    /// reports changed (`dirty_copies_in_gated` and kin), announced nowhere.
    pub device_copy_textures: u64,
    /// Of `host_textures_in_place`, those over pages imported as dma-bufs: each a
    /// `DRM_FORMAT_MOD_LINEAR` image over a dma-buf of its own pages (also counted in
    /// `dma_buf_images`).
    pub dma_buf_textures_in_place: u64,
    /// Of `host_textures_in_place`, those over a buffer in fresh pages it imported.
    pub fresh_page_textures_in_place: u64,
    /// Of `host_textures_in_place`, those whose rows lie at Metal's pitch in a
    /// `DRM_FORMAT_MOD_LINEAR` image made for host allocations, where the driver's own linear
    /// image lays them out otherwise.
    pub pitched_host_images: u64,
    /// Textures over host pages made shadows because the device's policy asks it
    /// (`HostPageTexturePolicy::Shadow`).
    pub host_texture_shadows_exact: u64,
    /// Textures over host pages made shadows because their usage renders into them
    /// (`RenderTarget`, whatever else it holds; an `Unknown` usage among them).
    pub host_texture_shadows_render_target: u64,
    /// Textures over host pages made shadows because their usage writes them through a shader
    /// (`ShaderWrite` or `ShaderAtomic`) and does not render into them.
    pub host_texture_shadows_shader_write: u64,
    /// Textures over host pages made shadows because the device's check of their class had not
    /// answered yet (a first session's, before the verdict is kept).
    pub host_texture_shadows_check_pending: u64,
    /// Textures over host pages made shadows because the device's check of their class found a
    /// byte changed.
    pub host_texture_shadows_check_failed: u64,
    /// Textures over host pages made shadows because the device's check made no image of their
    /// class.
    pub host_texture_shadows_unchecked: u64,
    /// Textures over host pages made shadows because their rows share bytes (an IOSurface
    /// plane's GPU rows).
    pub host_texture_shadows_refused_rows: u64,
    /// Textures over host pages made shadows because their bytes were detached from their pages.
    pub host_texture_shadows_refused_detached: u64,
    /// Textures over host pages made shadows because the device makes no such linear image of
    /// their plan (format features, usage, limits, an import only a dedicated allocation takes).
    pub host_texture_shadows_refused_image: u64,
    /// Textures over host pages made shadows because no image the device makes lays their rows
    /// (or slices) at the buffer's pitch.
    pub host_texture_shadows_refused_pitch: u64,
    /// Textures over host pages made shadows because the image's alignment, size or memory type
    /// misses their bytes' place.
    pub host_texture_shadows_refused_placement: u64,
    /// Textures over host pages imported as dma-bufs made shadows because the caller's source
    /// made no one dma-buf of their own pages: it failed (a source of `udmabuf`s does where the
    /// pages are more runs of its file than one `udmabuf` takes; its error is in the notice), or
    /// answered with several chunks. Whether the buffer's own import took several chunks does not
    /// decide it: the image asks for its own pages alone.
    pub host_texture_shadows_refused_source: u64,
    /// Textures over host pages imported as dma-bufs made shadows because no
    /// `DRM_FORMAT_MOD_LINEAR` image over one dma-buf of their own pages was made for another
    /// reason than the source or the pitch (counted in `host_texture_shadows_refused_pitch`): the
    /// modifier's features or limits, the image's memory past the pages, or the driver's import.
    pub host_texture_shadows_refused_dma_buf: u64,
    /// Classes of host pages (pointer, dma-buf, fresh) the device's check of the first move ran
    /// for in this process: `in_place_checks_passed`, `in_place_checks_failed` and
    /// `in_place_checks_unchecked` together.
    pub in_place_checks_run: u64,
    /// Of `in_place_checks_run`, those whose every image's first move left every byte alone.
    pub in_place_checks_passed: u64,
    /// Of `in_place_checks_run`, those that found a byte changed (or were made to fail).
    pub in_place_checks_failed: u64,
    /// Of `in_place_checks_run`, those that made no image of the class.
    pub in_place_checks_unchecked: u64,
    /// Classes whose verdict was read from the record of an earlier check of the same driver,
    /// device and runtime: no check ran for them.
    pub in_place_checks_recalled: u64,
    /// Buffers of the runtime's own made in fresh anonymous pages imported by their pointer
    /// (`VK_EXT_external_memory_host`), which read zero with nothing written: host-visible ones
    /// longer than half a pooled block, not of a heap, where the device takes such pages. A
    /// parked one taken again is not counted.
    pub fresh_page_buffers_made: u64,
    /// Their pages' bytes: whole units of the host page and the import alignment.
    pub fresh_page_bytes: u64,
    /// The bytes the host set to zero so that bytes nothing wrote read zero: of new buffers as
    /// they were made, the `Shared` and `Managed` ones' (a `Managed` buffer's host copy too,
    /// without unified memory) but those made in fresh pages; and of an upload region past its
    /// data, since a command buffer's arena chunks are made without a clear and each region zeroes
    /// what its data does not cover (`upload.rs`).
    pub new_buffer_bytes_cleared: u64,
    /// New buffers' bytes the host cannot write (`Private` storage, a `Managed` buffer's device
    /// copy, the device's neutral buffers) that a submission's first work set to zero with one
    /// `vkCmdFillBuffer`, on a device without unified memory whose fills are not transfers, in
    /// place of copies of the device's host-visible zeros, which such a device reads over the bus
    /// (`texture/first_move.rs`); fills apart from `fills_recorded`.
    pub new_buffer_zero_fills: u64,
    /// Copies between a shadowed texture's truth and its shadow that encoders planned
    /// (`shadow.rs`): copies in before a command, copies back after one or as the encoder ends,
    /// and the keeps and merges of a command that writes the bytes two ways or writes through a
    /// texture that merges back (`shadow_merge.rs`). Each is a command of its encoder over the
    /// texture's texels.
    pub shadow_copies: u64,
    /// The texels those copies in move from a 2D texture's buffer's bytes to its image (a texture
    /// over a buffer, a no-copy buffer or an IOSurface plane): every texel, or those inside the
    /// rectangle bounding a pass's draws' scissors over an attachment it loads and takes in no
    /// other way.
    pub shadow_texels_copied_in: u64,
    /// The texels every copy in of an image's shadow moves, image to image (a texture's own
    /// image's, `texture/usage_shadow.rs`, or a view's, `texture/view_shadow.rs`): each region's,
    /// or those inside a rectangle where the copy is cut as a buffer's texture's is. Those are not
    /// among `shadow_texels_copied_in`.
    pub shadow_texels_copied_in_images: u64,
    /// The texels each encoder's first copy in of a texture moves: before the first command of the
    /// encoder that uses it, or the first after work the encoder records apart. Those of a
    /// buffer's texture are among `shadow_texels_copied_in`, those of an image's shadow among
    /// `shadow_texels_copied_in_images`. Every other copy in is a later one, counted by why it is
    /// made (`shadow_texels_recopied_beyond_held`, `shadow_texels_recopied_after_changes`,
    /// `shadow_texels_bounded_copied_in`, which with this sum to both kinds' copies in). The six
    /// counts after it split it by what holds the texture's truth.
    pub shadow_texels_first_copied_in: u64,
    /// Of `shadow_texels_first_copied_in`, those of a truth that is an image: a texture's own
    /// (the shadow a binding beyond its usage or a read of its own pass's attachment takes,
    /// `texture/usage_shadow.rs`), or the one a view presents (`texture/view_shadow.rs`).
    pub shadow_texels_first_copied_in_images: u64,
    /// Those of the bytes of a buffer of private or memoryless storage the device made.
    pub shadow_texels_first_copied_in_private_buffers: u64,
    /// Those of the bytes of a buffer of shared or managed storage the device made, which the
    /// CPU maps.
    pub shadow_texels_first_copied_in_mapped_buffers: u64,
    /// Those of host pages a no-copy buffer imported (or the memory of the caller's `VkBuffer`)
    /// that only the shadow's own image is copied from.
    pub shadow_texels_first_copied_in_host_pages: u64,
    /// Those of host pages that the texture's own image lies in too (an aliasing image, a
    /// pitched host image among them).
    pub shadow_texels_first_copied_in_aliased_host_pages: u64,
    /// Those of an IOSurface's plane (`Buffer::new_texture_with_iosurface_plane`).
    pub shadow_texels_first_copied_in_surfaces: u64,
    /// Of `shadow_texels_first_copied_in_surfaces`, those under a texture of 2^20 texels or more
    /// (a 1920x1080 framebuffer's among them).
    pub shadow_texels_first_copied_in_large_surfaces: u64,
    /// The texels no copy in moved because the encoder's first use of a shadowed texture found
    /// its command buffer already holding that image's truth (`shadow::ShadowCarry`): exactly
    /// what carrying the image across the command buffer's encoders takes off the run. 0 means
    /// every copy in the run makes is the first in its own command buffer, which Metal requires,
    /// and the carry takes nothing off it.
    pub shadow_texels_first_copy_in_carried: u64,
    /// Of those, the ones of a texture over an IOSurface's plane of at least 2^20 texels
    /// (`shadow::ShadowCopies::large_surface`): the first of the three figures that divide such
    /// a plane's first copies in.
    pub shadow_texels_first_copy_in_carried_large_surfaces: u64,
    /// The second: the texels an encoder's first copy in of such a plane moved where the image's
    /// last standing was an earlier command buffer's. Metal shows a later command buffer a CPU
    /// write made since the commit, so it requires these and no rule removes them.
    pub shadow_texels_first_copied_in_large_surfaces_stood_earlier: u64,
    /// The third: the texels an encoder's first copy in of such a plane moved where the image
    /// never held a standing of any command buffer. Unavoidable under any rule.
    pub shadow_texels_first_copied_in_large_surfaces_unstood: u64,
    /// Of `shadow_texels_first_copied_in`, those of a texture an earlier encoder of the same
    /// command buffer copied in too, counted as each encoder ends (not an indirect command
    /// buffer's segment). The carry leaves those copies in out, so this counts only the ones it
    /// could not: a texture an encoder of the command buffer wrote the truth of another way, or
    /// that something let another writer in between.
    pub shadow_texels_first_copied_in_again: u64,
    /// The texels later copies in move (not an encoder's first) before a command that needs
    /// texels outside the rectangle the shadow's last copy in moved (a pass's read at its own
    /// pixels, then a read anywhere), the truth unchanged outside it or not.
    pub shadow_texels_recopied_beyond_held: u64,
    /// The texels later copies in move after the truth changed since the shadow's copy in, other
    /// than by writes within a rectangle that left it stale there alone. The five counts after it
    /// split it by the change that came first.
    pub shadow_texels_recopied_after_changes: u64,
    /// Of `shadow_texels_recopied_after_changes`, those after a copy back or merge of another
    /// shadowed texture over the same truth.
    pub shadow_texels_recopied_after_other_shadows: u64,
    /// Those after the texture's own merge, after a command that wrote its truth two ways.
    pub shadow_texels_recopied_after_two_ways_merges: u64,
    /// Those after a command wrote the truth another way (a pass that clears it, a storage
    /// write, a copy).
    pub shadow_texels_recopied_after_writes: u64,
    /// Those after a render pass instance's writes within a rectangle that left the image stale
    /// nowhere alone: the attachment has no shadow laid out as this one, or this one was
    /// written through or kept.
    pub shadow_texels_recopied_after_writes_within: u64,
    /// Those after a command wrote through the texture where Metal drops the writes.
    pub shadow_texels_recopied_after_dropped_writes: u64,
    /// The texels the copies of `bounded_shadow_copies` move: those inside the rectangle bounding
    /// where writes within rectangles changed the truth since the shadow's copy in.
    pub shadow_texels_bounded_copied_in: u64,
    /// The texels of those later copies in (`shadow_texels_recopied_beyond_held`,
    /// `shadow_texels_recopied_after_changes` and `shadow_texels_bounded_copied_in` together) of
    /// a truth that is an image, split by what holds the truth as the first copies are.
    pub shadow_texels_recopied_in_images: u64,
    /// Those of the bytes of a buffer of private or memoryless storage the device made.
    pub shadow_texels_recopied_in_private_buffers: u64,
    /// Those of the bytes of a buffer of shared or managed storage the device made.
    pub shadow_texels_recopied_in_mapped_buffers: u64,
    /// Those of host pages a no-copy buffer imported (or the memory of the caller's `VkBuffer`).
    pub shadow_texels_recopied_in_host_pages: u64,
    /// Those of host pages that the texture's own image lies in too.
    pub shadow_texels_recopied_in_aliased_host_pages: u64,
    /// Those of an IOSurface's plane.
    pub shadow_texels_recopied_in_surfaces: u64,
    /// Of `shadow_texels_recopied_in_surfaces`, those under a texture of 2^20 texels or more.
    pub shadow_texels_recopied_in_large_surfaces: u64,
    /// The texels those copies back move from a 2D texture's image to its buffer's bytes (a
    /// texture over a buffer, a no-copy buffer or an IOSurface plane): every texel, or those
    /// inside the rectangle bounding a pass's draws' scissors over an attachment it loads.
    pub shadow_texels_copied_back: u64,
    /// Of those copies back, the ones recorded as a compute dispatch that stores the texels'
    /// words, where the device runs one faster than its copy
    /// (`DriverFact::ShadowCopyBackDispatchFaster`): the same bytes either way.
    pub shadow_copies_back_dispatched: u64,
    /// Of the copies in among `shadow_copies`, those recorded with their texture's keep after
    /// them (one of `shadow_merge_copies_dispatched`) as one dispatch that stores the texels in
    /// the shadow and in the merge's scratch, where the device runs them faster so
    /// (`DriverFact::ShadowCopyInDispatchFaster`): the same bytes either way.
    pub shadow_copies_in_dispatched: u64,
    /// Of the keeps of merged shadowed textures, those of a heap's tiled texture recorded right
    /// after its copy in as one copy of the staging buffer that copy filled with the heap's
    /// bytes, in place of a move of the shadow's texels back into Metal's tiles: the same bytes
    /// either way (`shadow.rs`).
    pub shadow_keeps_from_staging: u64,
    /// The copies in of shadows over host pages a consumer logs whose regions the log decided as
    /// their batch was submitted (`dirty_pages.rs`).
    pub dirty_copies_in_gated: u64,
    /// Of those, the ones that copied every region: no hold to count on, a wait the log answered
    /// `false`, or half the rows or more changed.
    pub dirty_copies_in_whole: u64,
    /// Of those, the ones that copied nothing: no page changed.
    pub dirty_copies_in_skipped: u64,
    /// The bytes the gated copies in would have moved copying every region.
    pub dirty_copy_in_bytes_whole: u64,
    /// The bytes the gated copies in moved.
    pub dirty_copy_in_bytes_copied: u64,
    /// The stamps the commits of command buffers with a gated copy in asked of the logs: one per
    /// log and commit.
    pub dirty_stamps_asked: u64,
    /// The waits the submissions made for a log's publication (`DirtyPages::wait`): one per log
    /// serves every stamp asked of it before the wait.
    pub dirty_waits: u64,
    /// Of the waits, those the log answered `false`.
    pub dirty_waits_unanswered: u64,
    /// The nanoseconds the waits took, on the threads that submit command buffers.
    pub dirty_wait_nanos: u64,
    /// Of the keeps and merges among `shadow_copies`, those whose copy of the shadow's texels into
    /// the merge's scratch is the dispatch a copy back takes, where the device runs one faster
    /// than its copy (`DriverFact::ShadowCopyBackDispatchFaster`): the same bytes either way.
    pub shadow_merge_copies_dispatched: u64,
    /// The dispatches of a merge kernel the runtime planned: a shadowed texture's writes merged
    /// into its buffer's bytes (`shadow_merge.rs`), and a linear attachment's pass merged into
    /// its surface's bytes (`linear_merge.rs`), each in windows or bands of at most
    /// `maxStorageBufferRange` bytes and `maxComputeWorkGroupCount` workgroups, one dispatch
    /// each. Planned, not measured on the GPU: a merge through a copy over detached bytes is
    /// planned and never recorded, and a linear attachment's dispatches are recorded again at
    /// each memory barrier inside its pass.
    pub shadow_merge_dispatches_planned: u64,
    /// The words those dispatches examine: each reads the word before the commands and the word
    /// after them and stores into the buffer's bytes only where they differ, so the words stored
    /// are at most these and are not counted (only the GPU knows them).
    pub shadow_merge_words_examined: u64,
    /// Those dispatches that store whole 256-byte blocks (`bind::ShadowMergeStores::Blocks`): a
    /// merge of a command that replaced every texel of a shadow over logged host pages, cut to its
    /// texels' rows, every byte of which the command wrote (`shadow.rs`, **Overwrites over logged
    /// pages**), or of a render pass that loaded and stored its color attachment inside its draws'
    /// scissors, cut to that rectangle's rows, whose tiles Metal's own store writes back
    /// (**Stored attachments**). Of the blocks they examine, each holding a changed word is stored
    /// whole and each other is neither read nor written; which ones changed only the GPU knows.
    pub shadow_merge_dispatches_in_blocks: u64,
    /// The merges of shadowed textures whose scratch was a buffer their shadow kept from an
    /// earlier merge whose command buffers completed (`shadow_merge.rs`): no buffer made.
    pub shadow_merge_scratches_reused: u64,
    /// The merges of shadowed textures whose scratch was made zero, as their kernel reads bytes
    /// no copy writes (those between the texture's rows): a merge not cut to rows of whole words,
    /// through a copy or of rows that start or end inside a word. A cut merge's copies write
    /// every byte its kernel reads, as a heap's tiled texture's merge in place copies the
    /// texture's bytes whole, so neither scratch is zeroed.
    pub shadow_merge_scratches_zeroed: u64,
    /// Copies of rows of bytes between two buffers (`row_copies.rs`): a blit between a buffer and
    /// a texture over a buffer or between two textures over buffers, and a buffer-texture copy's
    /// rows through its staging buffer. Each is one `vkCmdCopyBuffer` of a region per row (rows
    /// contiguous in both buffers one region), or one dispatch.
    pub row_copies: u64,
    /// The regions of those copies.
    pub row_copy_regions: u64,
    /// The most regions of one of them.
    pub row_copy_regions_most: u64,
    /// Of `row_copies`, those recorded as one compute dispatch that stores the same bytes, on a
    /// device that runs it faster (`DriverFact::RowCopiesDispatchFaster`): at least 2 equal rows
    /// of 4-byte words, each the same distance after the one before.
    pub row_copies_dispatched: u64,
    /// The merges of shadowed textures that fetch the texels after the commands from the shadow
    /// through its unsigned-integer view, or its own `BGRA8Unorm` or `RGBA8Unorm` view, comparing
    /// them with the kept texels, in place of a copy of them into the scratch (`shadow_merge.rs`):
    /// a cut merge whose shadow has such a view.
    pub shadow_merges_fetched: u64,
    /// Of `shadow_merges_fetched`, those whose keep copies the shadow's texels into an image at
    /// their own coordinates (`vkCmdCopyImage`), which their kernel fetches beside the shadow's,
    /// in place of a copy or dispatch into the scratch: on a device whose copies in take no keep
    /// (no `DriverFact::ShadowCopyInDispatchFaster`), so they make no scratch.
    pub shadow_merges_kept_in_images: u64,
    /// Of `shadow_merges_kept_in_images`, those that made their image: the others took one their
    /// queue kept from an earlier keep of the same view and extent, whatever its shadow, once the
    /// encoder that took it ended (`texture/kept_image.rs`).
    pub shadow_merge_kept_images_made: u64,
    /// Of the others, those that took an image a command buffer not yet completed still held: in
    /// their queue's order, ahead of that completion, which they did not wait for.
    pub shadow_merge_kept_images_followed: u64,
    /// The kept images live now: those their queues keep, and those the command buffers that
    /// used them hold until they complete.
    pub shadow_merge_kept_images_live: u64,
    /// The shadowed textures render passes took as attachments, each once a pass
    /// (`render/shadows.rs`): among them textures over a buffer's bytes, a no-copy buffer's or an
    /// IOSurface plane's, whose pass copies the texels in before it where it loads them and back
    /// after it where it stores them.
    pub shadow_attachment_passes: u64,
    /// Of `shadow_attachment_passes`, those of a texture no earlier pass of its command buffer
    /// took as an attachment. The difference counts the passes on a texture after its first in a
    /// command buffer: the copies in and back a shadow kept across a command buffer's passes
    /// could leave out, at most.
    pub shadow_attachment_textures: u64,
    /// The shadowed textures render passes took as color attachments and loaded, each once a
    /// pass, whose copies are cut to the rectangle bounding the pass's draws' scissors
    /// (`render/shadows.rs`): the copy in moves the texels inside it alone, as do the copy back
    /// or the keep and merge.
    pub shadow_attachment_loads_within: u64,
    /// Those loaded whose copies move every texel: the pass's draws are not bounded so, or the
    /// attachment is not the whole of a one-level, one-slice 2D texture rendered itself. Those
    /// neither loaded nor cleared are among them.
    pub shadow_attachment_loads_whole: u64,
    /// Those cleared over the whole texture: no copy in.
    pub shadow_attachment_clears: u64,
    /// Those cleared over part of the texture (a render area smaller than it): every texel
    /// copied in.
    pub shadow_attachment_clears_in_part: u64,
    /// Those loaded in a pass that reaches no tile: no copy in or back.
    pub shadow_attachment_loads_untouched: u64,
    /// Of `shadow_attachment_tiles_bounded`, the tiles the pass's draws' scissor rectangles
    /// themselves reach: what a copy cut to those rectangles' tiles would move, where one cut to
    /// their bounding rectangle moves all of them.
    pub shadow_attachment_tiles_covered: u64,
    /// The 32-pixel tiles (from the render area's origin) the rectangle bounding the draws'
    /// scissors reaches, for each count of `shadow_attachment_loads_within`.
    pub shadow_attachment_tiles_bounded: u64,
    /// The shadowed textures an encoder used whose truth one it used before shares (another
    /// texture over the same bytes, or a view's shadow of the same image), counted as each
    /// encoder ends (not an indirect command buffer's segment): each copies its texels in apart.
    pub shadows_sharing_truth: u64,
    /// The commands that allocated descriptor sets for their dispatches of the runtime's own
    /// kernels (`owned_kernel.rs`), one each, from pools of the device's spares (a pool is made
    /// only where no spare has room): none for a kernel whose set 0 is pushed, on a device that
    /// pushes descriptor sets (`vkCmdPushDescriptorSet`): the rows copy's, and a shadowed
    /// texture's copy back's and merges'.
    pub owned_descriptor_pools: u64,
    /// The runtime's own kernels (`owned_kernel.rs`) this device built: each a SPIR-V build, a
    /// `VkShaderModule`, a descriptor set layout, a pipeline layout and a `VkPipeline`, which on
    /// a driver that compiles a pipeline as it is created is a shader compile. A kernel is built
    /// **by the Metal call that first needs it** and then kept for the device, so the count
    /// attributes the build to that call: making a texture of a format the runtime emulates
    /// (`ycbcr.rs`, `two_planes.rs`) where the making is what needs the kernel, encoding a
    /// command where the command is. A later call that takes a kernel already built adds
    /// nothing, so a reading of 0 beside a kernel's dispatch says the build landed in an earlier
    /// call. Metal builds none of these: they are what the emulation costs, and the count is
    /// where that cost fell.
    pub owned_kernels_made: u64,
    /// The SPIR-V those builds compiled, in bytes: what the pipeline creations were handed, which
    /// is the measure a build's cost scales with. It counts no scratch an emulation's work
    /// allocates besides (a YCbCr conversion's staging buffer, whose bytes are in
    /// `device_memory_allocated`).
    pub owned_kernel_spirv_bytes: u64,
    /// The scratch buffers a YCbCr texture's conversion took (`ycbcr.rs`), counted as each was
    /// allocated: one for the conversion that reads the texture's planes, and one more for the
    /// write-back where the texture takes writes. **One of each per texture**, not per device, and
    /// each is `Private` memory the size of the planes' whole byte range: a 3840 x 2160 `x420`
    /// surface's plane is a 24.9 MiB allocation a texture of it, however many such textures a
    /// program makes over the same surface. Metal wraps the plane and samples the format, so it
    /// takes none of this; it is what the emulation costs.
    pub ycbcr_conversion_scratch_buffers: u64,
    /// Those buffers' bytes, summed as each was allocated, so a reading is what a run allocated
    /// rather than what it holds now. It is memory for a texture's planes and is nothing to do
    /// with `owned_kernel_spirv_bytes`, which is the code a kernel's build compiled.
    pub ycbcr_conversion_scratch_bytes: u64,
    /// The texels the keeps among `shadow_copies` copy from a shadow into their merge's scratch:
    /// those inside the rectangle the encoder's commands write in, each row widened to whole
    /// words of the buffer, where the commands bound it and the merge goes in place; every texel
    /// otherwise.
    pub shadow_texels_kept: u64,
    /// Of `shadow_copies`, the copies in of the texels inside the rectangle bounding where the
    /// bytes changed since the image's copy in, where only writes inside rectangles of its texels
    /// changed them: a render pass instance's draws into a color attachment it loads, over a
    /// buffer's bytes, inside their scissors, before the next instance whose draws sample the
    /// attachment's texture (`texture/attached_reads.rs`). Each holds the same texels as a copy
    /// of every one; the rest of the image holds the bytes still.
    pub bounded_shadow_copies: u64,
    /// Texture views made the image itself to their bindings, through a swizzle over the image's
    /// own format: a view in other channels of its image's bits (an `A8Unorm` view of an
    /// `R8Unorm` texture), which neither copies nor announces (`texture/view_shadow.rs`).
    pub views_through_swizzle: u64,
    /// Shadow images of texture views made (`texture/view_shadow.rs`, `shadowed-texture-view`):
    /// a view's own, or the render shadow of one of `views_through_swizzle`, made at the first
    /// pass that renders into it.
    pub view_shadows: u64,
    /// Draws that sample, at a slot, a texture that presents a color attachment of their own
    /// render pass, each reading the texture's copy, taken before its render pass instance, in
    /// the attachment's place where the texture's shadow is served (`texture/attached_reads.rs`;
    /// the copies count in `shadow_copies`).
    pub attachment_reads_through_copies: u64,
    /// Draws that sample a color attachment's texture after a memory barrier of their render
    /// pass, each recorded into a new instance (a barrier inside an instance orders one pixel
    /// alone), whose copy holds what the draws before the barrier wrote
    /// (`texture/attached_reads.rs`).
    pub attachment_read_splits: u64,
    /// Draws that sample a color attachment's texture after a memory barrier of their render
    /// pass, each reading it through a snapshot inside its instance in place of a split: drawn
    /// twice, first copying each fragment's texel into the snapshot, then reading the snapshot
    /// (`render/snapshot.rs`). They count in neither `attachment_reads_through_copies` nor
    /// `attachment_read_splits`.
    pub reading_draws_snapshotted: u64,
    /// Of `reading_draws_snapshotted`, the draws recorded into a new instance because their own
    /// was recorded inline, its render pass instance begun without the feedback loop layout the
    /// snapshot needs (`render/snapshot.rs`, `render/inline_first.rs`).
    pub snapshot_inline_splits: u64,
    /// Linear images aliasing a buffer's bytes (a texture over a buffer, a no-copy buffer or an
    /// IOSurface plane) that the device made with feedback loop usage
    /// (`texture::feedback_candidate`), so a draw may read the texture through a snapshot while
    /// its pass renders it (`render/snapshot.rs`): one per image made. Where the device makes no
    /// such image with the usage, it makes one without, and the texture's readers split.
    pub feedback_loop_alias_images: u64,
    /// Render pass instances that held their multisampled depth/stencil attachment in
    /// `DEPTH_STENCIL_ATTACHMENT_OPTIMAL`, moved there from `GENERAL` just before each began and
    /// back just after it ended, on a device measured to render it faster so
    /// (`render_target::depth_layout`, `DriverFact::MultisampledDepthFasterOutOfGeneral`).
    pub depth_attachments_moved: u64,
    /// Render pass instances that rendered their multisampled depth/stencil attachment in
    /// `DEPTH_STENCIL_ATTACHMENT_OPTIMAL` with no move around them: their encoder holds it there,
    /// and it rests there between the encoders that hold it, on a device measured to render it
    /// faster so (`DriverFact::MultisampledDepthFasterOutOfGeneral`). They count in
    /// `depth_attachments_moved` too.
    pub depth_instances_held: u64,
    /// The moves of such a held depth out of `DEPTH_STENCIL_ATTACHMENT_OPTIMAL`, back to
    /// `GENERAL`, before an encoder that names it otherwise: each costs the device a resolve of
    /// the depth, as each instance's move back did before.
    pub held_depths_released: u64,
    /// Of `attachment_reads_through_copies`, the draws that read the texture at their fragments'
    /// own pixels alone (`StageModule::own_pixel_textures`) over a texture whose copy is cut to a
    /// rectangle: the copy in before their instance moves only the texels inside the rectangle
    /// bounding their scissors, a pixel wider each way (`texture/attached_reads.rs`).
    pub own_pixel_reads: u64,
    /// Ended compute and blit encoders whose logs their command buffers kept, as a rebind before
    /// the first submission may encode them again (`replay.rs`), and the bytes of those logs.
    pub replay_logs_kept: u64,
    /// The bytes of the logs counted in `replay_logs_kept`.
    pub replay_log_bytes: u64,
    /// Ended encoders a rebind, or a reset of a rasterization-rate map a render pass took,
    /// encoded again before their command buffer's first submission
    /// (`command_buffer/reencode.rs`).
    pub encoders_reencoded: u64,
    /// Work a command buffer's commit recorded on the GPU alone (a GPU-written execution's) that
    /// a rebind before its first submission of a buffer it named recorded again
    /// (`command_buffer/reencode.rs`).
    pub segments_recorded_again: u64,
    /// Resident root tables of work recorded on the GPU alone that their command buffer's first
    /// submission pointed at the resident set as it lay then, the live buffers having changed or
    /// moved since the commit (`command_buffer/first_submission.rs`): a rebind or a release
    /// meanwhile that records nothing again.
    pub replay_tables_pointed: u64,
    /// Ended encoders a rebind could not encode again (a call refused the second time, or a
    /// recording unlike the first), and work recorded on the GPU alone it could not record there
    /// again, which keep their first recording and fail `EncodedUseRebound`.
    pub reencodes_refused: u64,
    /// Fences made to end command buffers, where none was idle (`command_buffer/fences.rs`):
    /// none on a driver that reads no command buffer's objects after its value, whose queues'
    /// failures are not relayed.
    pub fences_made: u64,
    /// Graphics pipeline libraries made, one per distinct part of the render pipelines' state,
    /// where the device links its render pipelines from libraries (`pipeline/libraries.rs`):
    /// none elsewhere.
    pub pipeline_libraries_made: u64,
    /// Those of them of a fragment shader: each a compile of a fragment function, the costly part.
    pub fragment_shader_libraries_made: u64,
    /// Render variants' `VkPipeline`s linked from libraries (each counted in
    /// `variant_pipelines_made` too).
    pub pipelines_linked: u64,
    /// Compute encoders' dispatches that handed the making of a region's `VkPipeline` to the
    /// device's pool and went on, rather than make it at the dispatch: each counted once per
    /// region it handed.
    pub dispatch_pipelines_pooled: u64,
    /// Render encoders' tile dispatches that handed the making of a region's `VkPipeline` to the
    /// device's pool and went on, their passes recorded once it is made, rather than make it at
    /// the dispatch: each counted once per region it handed.
    pub tile_pipelines_pooled: u64,
    /// Chunks of command buffers' upload arenas placed in device-local memory on a device without
    /// unified memory, which the encoding thread then writes: none where the device has
    /// host-coherent, host-cached system memory (`upload.rs`).
    pub upload_chunks_in_device_memory: u64,
    /// Tile dispatches render encoders recorded (`dispatchThreadsPerTile:` and its region forms):
    /// each one the compute dispatch of the tile function over the pass's tiles, between the
    /// pass's instances, with its attachments' scratch copies.
    pub tile_dispatches_recorded: u64,
}

impl Device {
    /// What its commit path did so far (`work_counts.rs`): counts the lanes hold to bounds. No
    /// Metal counterpart; not part of the API.
    #[doc(hidden)]
    pub fn work_counts(&self) -> WorkCounts {
        // What the calls of the device's render encoders logged and not yet lowered will count
        // (`render/deferral.rs`).
        self.shared().deferred.lower_all();
        let counters = &self.shared().counts;
        // The driver's time in its pipeline makings (`device/compiles.rs`, by `VkDevice`), and
        // what the shader cache's directory cost (`shader_cache/store.rs`): neither is in
        // `Counters`.
        let (render_create_nanos, compute_create_nanos) = self.shared().pipeline_create_nanos();
        let directory = self.shared().shader_cache.directory_time();
        let churn = counters.churn.snapshot();
        let group_end = |end: GroupEnd| counters.group_ends[end as usize].load(Ordering::Relaxed);
        let first_work_order = self.shared().textures.first_moves.submissions_waited();
        let held_for =
            |reason: HeldFor| counters.heads_held_for[reason as usize].load(Ordering::Relaxed);
        let pending = |bucket: usize| counters.batches_by_pending[bucket].load(Ordering::Relaxed);
        let not_inline =
            |why: NotInline| counters.passes_not_inline[why.index()].load(Ordering::Relaxed);
        let actions = |action: ActionsRefused| {
            counters.passes_not_inline_actions_by[action as usize].load(Ordering::Relaxed)
        };
        let read_apart = |reason: ReadApart| {
            counters.host_imports_read_apart_by[reason as usize].load(Ordering::Relaxed)
        };
        let kept = |because: KeptBecause| {
            counters.host_imports_kept_by[because as usize].load(Ordering::Relaxed)
        };
        let begun =
            |what: Begun| counters.command_buffers_begun_by[what as usize].load(Ordering::Relaxed);
        let pass_begun =
            |at: PassBegun| counters.render_passes_begun_by[at as usize].load(Ordering::Relaxed);
        let split =
            |cause: Split| counters.instance_splits_by[cause as usize].load(Ordering::Relaxed);
        let (sleeps, [by_commit, by_completion, by_host, by_slice], scheduler_spared, gpu_waits) =
            self.shared().scheduler.sleeps_and_wakings();
        let idle_stalls = self.shared().scheduler.idle_stall_counts();
        let (completion_wakes, completion_spared) = self.shared().completion.wakes();
        let (handler_wakes, handler_spared) = self.shared().contexts.wakes();
        let command_buffer_wakes = counters.command_buffer_wakes.load(Ordering::Relaxed);
        let command_buffer_spared = counters.command_buffer_wakes_spared.load(Ordering::Relaxed);
        let event_wakes = counters.event_wakes.load(Ordering::Relaxed);
        let event_spared = counters.event_wakes_spared.load(Ordering::Relaxed);
        let (allocated, census, (parked, waiting), (made, freed), frees) = {
            let memory = lock(&self.shared().memory);
            let turnover = memory.turnover();
            (
                memory.allocated(),
                memory.census(),
                memory.parked(),
                turnover,
                memory.frees(),
            )
        };
        let dirty = self.shared().dirty.tally();
        WorkCounts {
            command_pools_made: counters.command_pools_made.load(Ordering::Relaxed),
            commands_recorded: self.shared().commands_recorded(),
            object_churn: churn.0,
            shadow_keeps: churn.1,
            copy_sites: self.shared().vk().copy_sites(),
            copy_regions_split: self.shared().vk().copy_regions_split(),
            spare_command_pools_taken: counters.spare_command_pools_taken.load(Ordering::Relaxed),
            spare_command_pools_reset: counters.spare_command_pools_reset.load(Ordering::Relaxed),
            live_tables_uploaded: counters.live_tables_uploaded.load(Ordering::Relaxed),
            timeline_waits: counters.timeline_waits.load(Ordering::Relaxed),
            fence_waits: counters.fence_waits.load(Ordering::Relaxed),
            missed_completion_wakes: counters.missed_completion_wakes.load(Ordering::Relaxed),
            completion_arms: counters.completion_arms.load(Ordering::Relaxed),
            completion_arms_after_a_wake: (counters.completion_arms_after_a_wake)
                .load(Ordering::Relaxed),
            completion_reads_ran_ahead: (counters.completion_reads_ran_ahead)
                .load(Ordering::Relaxed),
            gate_waits: counters.gate_waits.load(Ordering::Relaxed),
            gate_waits_slept: counters.gate_waits_slept.load(Ordering::Relaxed),
            gate_wakes_found_shut: counters.gate_wakes_found_shut.load(Ordering::Relaxed),
            trail_waits: counters.trail_waits.load(Ordering::Relaxed),
            trail_waits_blocked: counters.trail_waits_blocked.load(Ordering::Relaxed),
            completed_where_seen: counters.completed_where_seen.load(Ordering::Relaxed),
            scheduled_where_submitted: (counters.scheduled_where_submitted).load(Ordering::Relaxed),
            timestamp_slots_taken: counters.timestamp_slots_taken.load(Ordering::Relaxed),
            timestamp_slots_reset_on_host: (counters.timestamp_slots_reset_on_host)
                .load(Ordering::Relaxed),
            calibrations: counters.calibrations.load(Ordering::Relaxed),
            calibration_readings: counters.calibration_readings.load(Ordering::Relaxed),
            calibration_deviation_most: (counters.calibration_deviation_most)
                .load(Ordering::Relaxed),
            calibration_disagreements: (counters.calibration_disagreements).load(Ordering::Relaxed),
            storage_forms_taken: counters.storage_forms_taken.load(Ordering::Relaxed),
            addressable_buffers_made: counters.addressable_buffers_made.load(Ordering::Relaxed),
            mipmaps_filtered: counters.mipmaps_filtered.load(Ordering::Relaxed),
            mipmap_first_levels_fetched: counters
                .mipmap_first_levels_fetched
                .load(Ordering::Relaxed),
            pipelines_pooled: counters.pipelines_pooled.load(Ordering::Relaxed),
            residual_form_draws: counters.residual_form_draws.load(Ordering::Relaxed),
            residual_form_makings: counters.residual_form_makings.load(Ordering::Relaxed),
            pipeline_pool_busy: self.shared().pipeline_pool.busy() as u64,
            stage_entries_specialized: counters.stage_entries_specialized.load(Ordering::Relaxed),
            stage_entries_shared: counters.stage_entries_shared.load(Ordering::Relaxed),
            functions_translated: counters.functions_translated.load(Ordering::Relaxed),
            functions_shared: counters.functions_shared.load(Ordering::Relaxed),
            linked_use_questions_asked: counters.linked_use_questions_asked.load(Ordering::Relaxed),
            descriptor_keys_built: counters.descriptor_keys_built.load(Ordering::Relaxed),
            shader_modules_made: counters.shader_modules_made.load(Ordering::Relaxed),
            shader_modules_shared: counters.shader_modules_shared.load(Ordering::Relaxed),
            pipeline_layouts_made: counters.pipeline_layouts_made.load(Ordering::Relaxed),
            pipeline_layouts_shared: counters.pipeline_layouts_shared.load(Ordering::Relaxed),
            variant_pipelines_made: counters.variant_pipelines_made.load(Ordering::Relaxed),
            render_pipeline_create_nanos: render_create_nanos,
            compute_pipeline_create_nanos: compute_create_nanos,
            shader_cache_read_nanos: directory.read_nanos,
            shader_cache_write_nanos: directory.write_nanos,
            shader_cache_off_thread_nanos: directory.off_thread_nanos,
            shader_cache_writes_on_thread: directory.writes_on_thread,
            flights_followed: counters.flights_followed.load(Ordering::Relaxed),
            predicated_loop_runs_again: counters.predicated_loop_runs_again.load(Ordering::Relaxed),
            variants_taken_back: counters.variants_taken_back.load(Ordering::Relaxed),
            unlisted_argument_sampler_buffers: counters
                .unlisted_argument_sampler_buffers
                .load(Ordering::Relaxed),
            draws_dropped_unmade: counters.draws_dropped_unmade.load(Ordering::Relaxed),
            draws_dropped_still_making: counters.draws_dropped_still_making.load(Ordering::Relaxed),
            dispatches_dropped_still_making: counters
                .dispatches_dropped_still_making
                .load(Ordering::Relaxed),
            host_step_sampler_sets_made: counters
                .host_step_sampler_sets_made
                .load(Ordering::Relaxed),
            creation_pipelines_made: counters.creation_pipelines_made.load(Ordering::Relaxed),
            creations_left_making: counters.creations_left_making.load(Ordering::Relaxed),
            creation_pipelines_drawn: counters.creation_pipelines_drawn.load(Ordering::Relaxed),
            creation_render_pipelines_made: counters
                .creation_render_pipelines_made
                .load(Ordering::Relaxed),
            creation_render_pipelines_drawn: counters
                .creation_render_pipelines_drawn
                .load(Ordering::Relaxed),
            creation_dispatch_pipelines_made: counters
                .creation_dispatch_pipelines_made
                .load(Ordering::Relaxed),
            creation_dispatch_pipelines_drawn: counters
                .creation_dispatch_pipelines_drawn
                .load(Ordering::Relaxed),
            sampler_sets_made: counters.sampler_sets_made.load(Ordering::Relaxed),
            sampler_sets_shared: counters.sampler_sets_shared.load(Ordering::Relaxed),
            sampler_sets_destroyed: counters.sampler_sets_destroyed.load(Ordering::Relaxed),
            ahead_render_passes_made: counters.ahead_render_passes_made.load(Ordering::Relaxed),
            render_passes_made: counters.render_passes_made.load(Ordering::Relaxed),
            ahead_framebuffers_made: counters.ahead_framebuffers_made.load(Ordering::Relaxed),
            kept_backings_made: counters.kept_backings_made.load(Ordering::Relaxed),
            pooled_backings_made: counters.pooled_backings_made.load(Ordering::Relaxed),
            storage_placeholders_made: counters.storage_placeholders_made.load(Ordering::Relaxed),
            storage_placeholder_attachments: counters
                .storage_placeholder_attachments
                .load(Ordering::Relaxed),
            color_resolve_passes_made: counters.color_resolve_passes_made.load(Ordering::Relaxed),
            color_resolve_passes_kept: counters.color_resolve_passes_kept.load(Ordering::Relaxed),
            ahead_nanos: counters.ahead_nanos.load(Ordering::Relaxed),
            driver_diagnostics_withheld: counters
                .driver_diagnostics_withheld
                .load(Ordering::Relaxed),
            device_memory_allocated: allocated,
            memory_blocks_live: census.pooled,
            empty_memory_blocks_live: census.empty,
            pooled_memory_blocks_made: made,
            pooled_memory_blocks_freed: freed,
            dedicated_memory_blocks_live: census.dedicated,
            memory_frees: MemoryFrees {
                dedicated: frees[FreeCause::Dedicated as usize],
                pooled_emptied: frees[FreeCause::PooledEmptied as usize],
                kept_stale: frees[FreeCause::KeptStale as usize],
                kept_for_room: frees[FreeCause::KeptForRoom as usize],
                host_import: frees[FreeCause::HostImport as usize],
                own_pages: frees[FreeCause::OwnPages as usize],
                teardown: frees[FreeCause::Teardown as usize],
                blocks: frees.iter().sum(),
            },
            host_imports_live: census.imported,
            freed_memory_parked: parked,
            freed_memory_waiting: waiting,
            stand_ins_made: counters.stand_ins_made.load(Ordering::Relaxed),
            stand_in_bytes: counters.stand_in_bytes.load(Ordering::Relaxed),
            stale_detached_reaches: counters.stale_detached_reaches.load(Ordering::Relaxed),
            detached_storage_reaches: counters.detached_storage_reaches.load(Ordering::Relaxed),
            heap_slots_rewritten: counters.heap_slots_rewritten.load(Ordering::Relaxed),
            heap_slot_writes_deferred: counters.heap_slot_writes_deferred.load(Ordering::Relaxed),
            host_range_lookups: self.shared().aliases.range_lookups(),
            heap_textures_tiled: counters.heap_textures_tiled.load(Ordering::Relaxed),
            heap_sample_move_images_made: counters
                .heap_sample_move_images_made
                .load(Ordering::Relaxed),
            heap_sample_move_images_lent: counters
                .heap_sample_move_images_lent
                .load(Ordering::Relaxed),
            requirement_images_made: counters.requirement_images_made.load(Ordering::Relaxed),
            requirement_limits_asked: counters.requirement_limits_asked.load(Ordering::Relaxed),
            arena_pools_made: counters.arena_pools_made.load(Ordering::Relaxed),
            heap_texture_lossless_layouts: counters
                .heap_texture_lossless_layouts
                .load(Ordering::Relaxed),
            heap_texture_block_compressed_layouts: counters
                .heap_texture_block_compressed_layouts
                .load(Ordering::Relaxed),
            heap_texture_untiled_layouts: counters
                .heap_texture_untiled_layouts
                .load(Ordering::Relaxed),
            fills_recorded: counters.fills_recorded.load(Ordering::Relaxed),
            fills_by_transfers: counters.fills_by_transfers.load(Ordering::Relaxed),
            ycbcr_refreshes: counters.ycbcr_refreshes.load(Ordering::Relaxed),
            ycbcr_whole_conversions: counters.ycbcr_whole_conversions.load(Ordering::Relaxed),
            ycbcr_textures_in_place: counters.ycbcr_textures_in_place.load(Ordering::Relaxed),
            ycbcr_images_kept_device: counters.ycbcr_images_kept[0].load(Ordering::Relaxed),
            ycbcr_images_kept_layout: counters.ycbcr_images_kept[1].load(Ordering::Relaxed),
            ycbcr_images_kept_swizzle: counters.ycbcr_images_kept[2].load(Ordering::Relaxed),
            ycbcr_images_kept_alignment: counters.ycbcr_images_kept[3].load(Ordering::Relaxed),
            ycbcr_images_kept_elements: counters.ycbcr_images_kept[4].load(Ordering::Relaxed),
            ycbcr_images_kept_reach: counters.ycbcr_images_kept[5].load(Ordering::Relaxed),
            ycbcr_images_kept_view: counters.ycbcr_images_kept[6].load(Ordering::Relaxed),
            ycbcr_planes_bound: counters.ycbcr_planes_bound.load(Ordering::Relaxed),
            ycbcr_planes_images: counters.ycbcr_planes_images.load(Ordering::Relaxed),
            ycbcr_planes_images_kept_conversion: counters.ycbcr_planes_images_kept[0]
                .load(Ordering::Relaxed),
            ycbcr_planes_images_kept_memory: counters.ycbcr_planes_images_kept[1]
                .load(Ordering::Relaxed),
            ycbcr_planes_images_kept_layout: counters.ycbcr_planes_images_kept[2]
                .load(Ordering::Relaxed),
            ycbcr_planes_images_bound: counters.ycbcr_planes_images_bound.load(Ordering::Relaxed),
            ycbcr_device_conversions: counters.ycbcr_device_conversions.load(Ordering::Relaxed),
            ycbcr_device_samples: counters.ycbcr_device_samples.load(Ordering::Relaxed),
            attachment_clears_recorded: counters.attachment_clears_recorded.load(Ordering::Relaxed),
            image_blits_recorded: counters.image_blits_recorded.load(Ordering::Relaxed),
            query_result_copies_recorded: counters
                .query_result_copies_recorded
                .load(Ordering::Relaxed),
            visibility_queries_begun: counters.visibility_queries_begun.load(Ordering::Relaxed),
            indirect_draws_recorded: counters.indirect_draws_recorded.load(Ordering::Relaxed),
            indirect_records_read: counters.indirect_records_read.load(Ordering::Relaxed),
            gpu_form_commands: counters.gpu_form_commands.load(Ordering::Relaxed),
            merged_draw_commands: counters.merged_draw_commands.load(Ordering::Relaxed),
            gpu_pool_blocks_made: counters.gpu_pool_blocks_made.load(Ordering::Relaxed),
            gpu_pool_blocks_released: counters.gpu_pool_blocks_released.load(Ordering::Relaxed),
            gpu_pool_leases_peak: counters.gpu_pool_leases_peak.load(Ordering::Relaxed),
            internal_submissions: counters.internal_submissions.load(Ordering::Relaxed),
            queue_submissions: counters.queue_submissions.load(Ordering::Relaxed),
            submit_infos: counters.submit_infos.load(Ordering::Relaxed),
            command_buffers_submitted: counters.command_buffers_submitted.load(Ordering::Relaxed),
            batch_submit_nanos: counters.batch_submit_nanos.load(Ordering::Relaxed),
            idle_submits: idle_stalls.empty,
            submit_stalls_idle: idle_stalls.idle,
            submit_stalls_on_work: counters.submit_stalls_on_work.load(Ordering::Relaxed),
            submits_behind_idle_stalls: idle_stalls.behind
                + counters.submits_behind_idle_stalls.load(Ordering::Relaxed),
            wait_behind_idle_stall_longest_nanos: idle_stalls
                .behind_longest
                .max((counters.wait_behind_idle_stall_longest_nanos).load(Ordering::Relaxed)),
            command_buffers_begun: counters.command_buffers_begun.load(Ordering::Relaxed),
            command_buffers_begun_encoders: begun(Begun::Encoder),
            command_buffers_begun_segments: begun(Begun::Segment),
            command_buffers_begun_continuations: begun(Begun::Continuation),
            command_buffers_begun_secondaries: begun(Begun::Secondary),
            command_buffers_begun_batch_work: begun(Begun::BatchWork),
            command_buffers_begun_batch_barriers: begun(Begun::BatchBarriers),
            command_buffers_begun_kept_barriers: begun(Begun::KeptBarriers),
            command_buffers_begun_internal: begun(Begun::Internal),
            encoder_command_buffers_begun: (counters.encoder_command_buffers_begun)
                .load(Ordering::Relaxed),
            encoder_primaries_begun_ahead: (counters.encoder_primaries_begun_ahead)
                .load(Ordering::Relaxed),
            primaries_begun_ahead: counters.primaries_begun_ahead.load(Ordering::Relaxed),
            barrier_buffers_kept: counters.barrier_buffers_kept.load(Ordering::Relaxed),
            stamped_buffers_kept: counters.stamped_buffers_kept.load(Ordering::Relaxed),
            stamped_buffers_reused: counters.stamped_buffers_reused.load(Ordering::Relaxed),
            finales_folded: counters.finales_folded.load(Ordering::Relaxed),
            finales_apart: counters.finales_apart.load(Ordering::Relaxed),
            grouped_submissions: counters.grouped_submissions.load(Ordering::Relaxed),
            batches_grouped: counters.batches_grouped.load(Ordering::Relaxed),
            group_ends_full: group_end(GroupEnd::Full),
            group_ends_wait: group_end(GroupEnd::Wait),
            group_ends_apart: group_end(GroupEnd::Apart),
            group_ends_first_work: group_end(GroupEnd::FirstWork),
            group_ends_drained: group_end(GroupEnd::Drained),
            group_ends_checks: group_end(GroupEnd::Checks),
            group_ends_held: group_end(GroupEnd::Held),
            batches_with_none_pending: pending(0),
            batches_with_one_pending: pending(1),
            batches_with_two_or_three_pending: pending(2),
            batches_with_four_or_more_pending: pending(3),
            groups_kept_pending: counters.groups_kept_pending.load(Ordering::Relaxed),
            groups_sent_for_waits: counters.groups_sent_for_waits.load(Ordering::Relaxed),
            heads_held_for_binary_semaphores: held_for(HeldFor::BinarySemaphore),
            heads_held_for_reencodes: held_for(HeldFor::Reencoding),
            heads_held_for_decodes: held_for(HeldFor::Decode),
            heads_held_for_host_steps: held_for(HeldFor::HostStep),
            heads_held_for_checks: held_for(HeldFor::Check),
            heads_held_for_event_values: held_for(HeldFor::EventValue),
            heads_held_for_earlier_commits: held_for(HeldFor::EarlierCommit),
            heads_waiting_at_their_gate: counters
                .heads_waiting_at_their_gate
                .load(Ordering::Relaxed),
            heads_awaiting_check_signals: counters
                .heads_awaiting_check_signals
                .load(Ordering::Relaxed),
            batches_held_past_a_look: counters.batches_held_past_a_look.load(Ordering::Relaxed),
            batch_hold_nanos: counters.batch_hold_nanos.load(Ordering::Relaxed),
            registrations_waited: self.shared().textures.first_moves.registrations_waited(),
            first_work_order_waits: first_work_order.0,
            first_work_order_wait_nanos: first_work_order.1,
            scheduler_sleeps: sleeps,
            scheduler_wakes_commit: by_commit,
            scheduler_wakes_completion: by_completion,
            scheduler_wakes_host: by_host,
            scheduler_sleep_slices: by_slice,
            scheduler_gpu_waits: gpu_waits,
            scheduler_passes: self.shared().scheduler.passes(),
            scheduler_passes_by_waiters: counters
                .scheduler_passes_by_waiters
                .load(Ordering::Relaxed),
            batches_submitted_by_waiters: counters
                .batches_submitted_by_waiters
                .load(Ordering::Relaxed),
            wakes_completion_waiters: completion_wakes,
            wakes_handler_threads: handler_wakes,
            wakes_command_buffer_waiters: command_buffer_wakes,
            wakes_event_waiters: event_wakes,
            wakes_spared: completion_spared
                + handler_spared
                + command_buffer_spared
                + event_spared
                + scheduler_spared,
            slot_returns: counters.slot_returns.load(Ordering::Relaxed),
            slot_return_nanos: counters.slot_return_nanos.load(Ordering::Relaxed),
            slot_waits: counters.slot_waits.load(Ordering::Relaxed),
            slot_wait_nanos: counters.slot_wait_nanos.load(Ordering::Relaxed),
            queues_sharing_a_vk_queue: counters.queues_sharing_a_vk_queue.load(Ordering::Relaxed),
            completed_waits: counters.completed_waits.load(Ordering::Relaxed),
            event_waits_met: counters.event_waits_met.load(Ordering::Relaxed),
            shared_event_waits_held: counters.shared_event_waits_held.load(Ordering::Relaxed),
            event_signals_behind_merge: counters.event_signals_behind_merge.load(Ordering::Relaxed),
            bare_starts: counters.bare_starts.load(Ordering::Relaxed),
            prefixed_starts: counters.prefixed_starts.load(Ordering::Relaxed),
            command_barriers: counters.command_barriers.load(Ordering::Relaxed),
            unrun_instances_ended: counters.unrun_instances_ended.load(Ordering::Relaxed),
            untouched_passes_skipped: counters.untouched_passes_skipped.load(Ordering::Relaxed),
            inline_passes: counters.inline_passes.load(Ordering::Relaxed),
            inline_passes_split: counters.inline_passes_split.load(Ordering::Relaxed),
            inline_split_instances: counters.inline_split_instances.load(Ordering::Relaxed),
            inline_memoryless_write_splits: (counters.inline_memoryless_write_splits)
                .load(Ordering::Relaxed),
            render_encoders_ended: counters.render_encoders_ended.load(Ordering::Relaxed),
            render_log_batches_by_thread: (counters.render_log_batches_by_thread)
                .load(Ordering::Relaxed),
            render_log_batches_by_program: (counters.render_log_batches_by_program)
                .load(Ordering::Relaxed),
            render_log_waits: counters.render_log_waits.load(Ordering::Relaxed),
            render_log_ends_by_thread: (counters.render_log_ends_by_thread).load(Ordering::Relaxed),
            render_log_ends_by_program: (counters.render_log_ends_by_program)
                .load(Ordering::Relaxed),
            render_log_commits_by_thread: (counters.render_log_commits_by_thread)
                .load(Ordering::Relaxed),
            render_log_commits_by_program: (counters.render_log_commits_by_program)
                .load(Ordering::Relaxed),
            render_passes_begun_instances: pass_begun(PassBegun::Instance),
            render_passes_begun_inline: pass_begun(PassBegun::Inline),
            render_passes_begun_inline_splits: pass_begun(PassBegun::InlineSplit),
            render_passes_begun_resolves: pass_begun(PassBegun::Resolve),
            instance_splits_barrier: split(Split::Barrier),
            instance_splits_tile_dispatch: split(Split::TileDispatch),
            instance_splits_tile_memory: split(Split::TileMemory),
            instance_splits_attached_read: split(Split::AttachedRead),
            instance_splits_feedback: split(Split::Feedback),
            instance_splits_memoryless_write: split(Split::MemorylessWrite),
            instance_splits_pending_pipeline: split(Split::PendingPipeline),
            instance_splits_execution: split(Split::Execution),
            instance_splits_visibility: split(Split::Visibility),
            instance_splits_shape: split(Split::Shape),
            passes_not_inline_encoders: not_inline(NotInline::Encoders),
            passes_not_inline_layouts: not_inline(NotInline::Layouts),
            passes_not_inline_memoryless: not_inline(NotInline::Memoryless),
            passes_not_inline_actions: (counters.passes_not_inline[NotInline::ACTIONS])
                .load(Ordering::Relaxed),
            passes_not_inline_actions_dont_care_load: actions(ActionsRefused::DontCareLoad),
            passes_not_inline_actions_unequal_reach: actions(ActionsRefused::UnequalReach),
            passes_not_inline_actions_resolve: actions(ActionsRefused::Resolve),
            passes_not_inline_actions_store: actions(ActionsRefused::Store),
            passes_not_inline_buffer_textures: not_inline(NotInline::BufferTextures),
            passes_not_inline_attachments: not_inline(NotInline::Attachments),
            passes_not_inline_storage: not_inline(NotInline::Storage),
            passes_not_inline_stand_ins: not_inline(NotInline::StandIn),
            passes_not_inline_memoryless_writes: not_inline(NotInline::MemorylessWrites),
            passes_not_inline_draws: not_inline(NotInline::Draws),
            commits_handed_off: counters.commits_handed_off.load(Ordering::Relaxed),
            images_started_on_the_host: counters.images_started_on_the_host.load(Ordering::Relaxed),
            host_texel_copies: counters.host_texel_copies.load(Ordering::Relaxed),
            texel_shadows_made: counters.texel_shadows_made.load(Ordering::Relaxed),
            texel_shadow_bytes: counters.texel_shadow_bytes.load(Ordering::Relaxed),
            texel_shadow_reads: counters.texel_shadow_reads.load(Ordering::Relaxed),
            texel_shadows_dropped: counters.texel_shadows_dropped.load(Ordering::Relaxed),
            staging_submissions: counters.staging_submissions.load(Ordering::Relaxed),
            uncached_host_images: counters.uncached_host_images.load(Ordering::Relaxed),
            cpu_block_decodes: counters.cpu_block_decodes.load(Ordering::Relaxed),
            block_decode_passes: counters.block_decode_passes.load(Ordering::Relaxed),
            chroma_device_twins: counters.chroma_device_twins.load(Ordering::Relaxed),
            chroma_device_decodes: counters.chroma_device_decodes.load(Ordering::Relaxed),
            second_twins: counters.second_twins.load(Ordering::Relaxed),
            second_twin_checks: counters.second_twin_checks.load(Ordering::Relaxed),
            second_twin_decodes: counters.second_twin_decodes.load(Ordering::Relaxed),
            queue_drains: counters.queue_drains.load(Ordering::Relaxed),
            stand_in_copies_made: counters.stand_in_copies_made.load(Ordering::Relaxed),
            stand_in_copies_made_at_draws: counters
                .stand_in_copies_made_at_draws
                .load(Ordering::Relaxed),
            draw_shapes_made: counters.draw_shapes_made.load(Ordering::Relaxed),
            draw_shapes_found: counters.draw_shapes_found.load(Ordering::Relaxed),
            argument_samplers_taken: counters.argument_samplers_taken.load(Ordering::Relaxed),
            bind_views_worked_out: counters.bind_views_worked_out.load(Ordering::Relaxed),
            stage_tables_read: counters.stage_tables_read.load(Ordering::Relaxed),
            stage_tables_read_in_part: counters.stage_tables_read_in_part.load(Ordering::Relaxed),
            no_copy_lies_read: counters.no_copy_lies_read.load(Ordering::Relaxed),
            commands_noted_as_before: counters.commands_noted_as_before.load(Ordering::Relaxed),
            draws_settled: counters.draws_settled.load(Ordering::Relaxed),
            two_plane_draws: counters.two_plane_draws.load(Ordering::Relaxed),
            provoking_last_pipeline_draws: counters
                .provoking_last_pipeline_draws
                .load(Ordering::Relaxed),
            provoking_listed_draws: counters.provoking_listed_draws.load(Ordering::Relaxed),
            gathered_draws: counters.gathered_draws.load(Ordering::Relaxed),
            bounded_run_draws: counters.bounded_run_draws.load(Ordering::Relaxed),
            bound_notices_made: counters.bound_notices_made.load(Ordering::Relaxed),
            argument_tables_retained: counters.argument_tables_retained.load(Ordering::Relaxed),
            argument_objects_retained: counters.argument_objects_retained.load(Ordering::Relaxed),
            graphics_pipelines_looked_up: counters
                .graphics_pipelines_looked_up
                .load(Ordering::Relaxed),
            linear_passes_in_place: counters.linear_passes_in_place.load(Ordering::Relaxed),
            linear_passes_through_copies: counters
                .linear_passes_through_copies
                .load(Ordering::Relaxed),
            kept_bytes_passes: counters.kept_bytes_passes.load(Ordering::Relaxed),
            color_group_passes: counters.color_group_passes.load(Ordering::Relaxed),
            color_group_twins: counters.color_group_twins.load(Ordering::Relaxed),
            color_group_draws_skipped: counters.color_group_draws_skipped.load(Ordering::Relaxed),
            draw_states_recorded: counters.draw_states_recorded.load(Ordering::Relaxed),
            draws_pushing_set_zero: counters.draws_pushing_set_zero.load(Ordering::Relaxed),
            set_zero_writes_pushed: counters.set_zero_writes_pushed.load(Ordering::Relaxed),
            set_zero_equal_shape_switches: (counters.set_zero_equal_shape_switches)
                .load(Ordering::Relaxed),
            set_zero_writes_on_equal_shape_switches: (counters
                .set_zero_writes_on_equal_shape_switches)
                .load(Ordering::Relaxed),
            set_zero_writes_if_shapes_shared: (counters.set_zero_writes_if_shapes_shared)
                .load(Ordering::Relaxed),
            dispatches_pushing_set_zero: (counters.dispatches_pushing_set_zero)
                .load(Ordering::Relaxed),
            set_zero_sets_reused: counters.set_zero_sets_reused.load(Ordering::Relaxed),
            set_zero_sets_found: counters.set_zero_sets_found.load(Ordering::Relaxed),
            draw_shapes_reused: counters.draw_shapes_reused.load(Ordering::Relaxed),
            dispatches_settled_as_before: (counters.dispatches_settled_as_before)
                .load(Ordering::Relaxed),
            dispatches_planned_as_before: (counters.dispatches_planned_as_before)
                .load(Ordering::Relaxed),
            baked_sampler_dispatches_apart: (counters.baked_sampler_dispatches_apart)
                .load(Ordering::Relaxed),
            host_deallocators_delayed: counters.host_deallocators_delayed.load(Ordering::Relaxed),
            host_import_releases_posted: (counters.host_import_releases_posted)
                .load(Ordering::Relaxed),
            host_import_releases_pending: (counters.host_import_releases_pending)
                .load(Ordering::Acquire),
            host_keep_alives_kept: counters.host_keep_alives_kept.load(Ordering::Relaxed),
            host_keep_alives_covered: counters.host_keep_alives_covered.load(Ordering::Relaxed),
            host_mappings_queried: counters.host_mappings_queried.load(Ordering::Relaxed),
            dma_buf_imports: counters.dma_buf_imports.load(Ordering::Relaxed),
            dma_buf_memories_live: counters.dma_buf_memories_live.load(Ordering::Relaxed),
            dma_buf_image_memories_live: counters
                .dma_buf_image_memories_live
                .load(Ordering::Relaxed),
            host_imports_released: counters.host_imports_released.load(Ordering::Relaxed),
            host_imports_reimported: counters.host_imports_reimported.load(Ordering::Relaxed),
            host_imports_resident: counters.host_imports_resident.load(Ordering::Relaxed),
            host_imports_used: counters.host_imports_used.load(Ordering::Relaxed),
            host_imports_read_apart: counters.host_imports_read_apart.load(Ordering::Relaxed),
            host_imports_read_apart_address: read_apart(ReadApart::Address),
            host_imports_read_apart_vk_buffer: read_apart(ReadApart::VkBuffer),
            host_imports_read_apart_binding: read_apart(ReadApart::Binding),
            host_imports_read_apart_texture: read_apart(ReadApart::Texture),
            host_imports_read_apart_second_buffer: read_apart(ReadApart::SecondBuffer),
            host_imports_read_apart_unreleased: read_apart(ReadApart::Unreleased),
            host_imports_kept_of_heap: kept(KeptBecause::Heap),
            host_imports_kept_rebound: kept(KeptBecause::Rebind),
            host_imports_kept_pointer: kept(KeptBecause::Pointer),
            host_imports_kept_no_stand_in: kept(KeptBecause::NoStandIn),
            host_imports_rebound_unlisted: counters
                .host_imports_rebound_unlisted
                .load(Ordering::Relaxed),
            host_rebind_imports_made: counters.host_rebind_imports_made.load(Ordering::Relaxed),
            host_rebind_imports_covered: counters
                .host_rebind_imports_covered
                .load(Ordering::Relaxed),
            host_remaps_reported: counters.host_remaps_reported.load(Ordering::Relaxed),
            host_remaps_over_imports: counters.host_remaps_over_imports.load(Ordering::Relaxed),
            host_imports_remapped: counters.host_imports_remapped.load(Ordering::Relaxed),
            host_pointer_imports_made: counters.host_pointer_imports_made.load(Ordering::Relaxed),
            host_pointer_import_bytes_made: counters
                .host_pointer_import_bytes_made
                .load(Ordering::Relaxed),
            host_imports_refused_by_host: counters
                .host_imports_refused_by_host
                .load(Ordering::Relaxed),
            copied_host_buffers_made: counters.copied_host_buffers_made.load(Ordering::Relaxed),
            copied_host_buffers_refused: counters
                .copied_host_buffers_refused
                .load(Ordering::Relaxed),
            copied_host_buffers_degraded: counters
                .copied_host_buffers_degraded
                .load(Ordering::Relaxed),
            copied_rebind_pages_moved: counters.copied_rebind_pages_moved.load(Ordering::Relaxed),
            copied_rebind_pages_made: counters.copied_rebind_pages_made.load(Ordering::Relaxed),
            copied_rebind_images_made: counters.copied_rebind_images_made.load(Ordering::Relaxed),
            copied_heaps_made: counters.copied_heaps_made.load(Ordering::Relaxed),
            copied_copy_in_bytes: counters.copied_copy_in_bytes.load(Ordering::Relaxed),
            copied_merge_back_bytes: counters.copied_merge_back_bytes.load(Ordering::Relaxed),
            copied_dropped_bytes: counters.copied_dropped_bytes.load(Ordering::Relaxed),
            host_ram_chunks_imported: counters.host_ram_chunks_imported.load(Ordering::Relaxed),
            host_ram_chunk_bytes: counters.host_ram_chunk_bytes.load(Ordering::Relaxed),
            host_ram_chunks_live: counters.host_ram_chunks_live.load(Ordering::Relaxed),
            chunk_bound_buffers_made: counters.chunk_bound_buffers_made.load(Ordering::Relaxed),
            host_imports_lost: counters.host_imports_lost.load(Ordering::Relaxed),
            host_import_fds_kept: counters.host_import_fds_kept.load(Ordering::Relaxed),
            host_imports_kept_for_fds: counters.host_imports_kept_for_fds.load(Ordering::Relaxed),
            dma_buf_sparse_buffers: counters.dma_buf_sparse_buffers.load(Ordering::Relaxed),
            dma_buf_images: counters.dma_buf_images.load(Ordering::Relaxed),
            dma_buf_image_fallbacks: counters.dma_buf_image_fallbacks.load(Ordering::Relaxed),
            modifier_feature_queries: counters.modifier_feature_queries.load(Ordering::Relaxed),
            shadowed_textures: counters.shadowed_textures.load(Ordering::Relaxed),
            shadows_over_fresh_pages: counters.shadows_over_fresh_pages.load(Ordering::Relaxed),
            host_textures_in_place: counters.host_textures_in_place.load(Ordering::Relaxed),
            device_copy_textures: counters.device_copy_textures.load(Ordering::Relaxed),
            dma_buf_textures_in_place: counters.dma_buf_textures_in_place.load(Ordering::Relaxed),
            fresh_page_textures_in_place: counters
                .fresh_page_textures_in_place
                .load(Ordering::Relaxed),
            pitched_host_images: counters.pitched_host_images.load(Ordering::Relaxed),
            host_texture_shadows_exact: counters.host_texture_shadows_exact.load(Ordering::Relaxed),
            host_texture_shadows_render_target: counters
                .host_texture_shadows_render_target
                .load(Ordering::Relaxed),
            host_texture_shadows_shader_write: counters
                .host_texture_shadows_shader_write
                .load(Ordering::Relaxed),
            host_texture_shadows_check_pending: counters
                .host_texture_shadows_check_pending
                .load(Ordering::Relaxed),
            host_texture_shadows_check_failed: counters
                .host_texture_shadows_check_failed
                .load(Ordering::Relaxed),
            host_texture_shadows_unchecked: counters
                .host_texture_shadows_unchecked
                .load(Ordering::Relaxed),
            host_texture_shadows_refused_rows: counters.host_texture_shadows_refused[0]
                .load(Ordering::Relaxed),
            host_texture_shadows_refused_detached: counters.host_texture_shadows_refused[1]
                .load(Ordering::Relaxed),
            host_texture_shadows_refused_image: counters.host_texture_shadows_refused[2]
                .load(Ordering::Relaxed),
            host_texture_shadows_refused_pitch: counters.host_texture_shadows_refused[3]
                .load(Ordering::Relaxed),
            host_texture_shadows_refused_placement: counters.host_texture_shadows_refused[4]
                .load(Ordering::Relaxed),
            host_texture_shadows_refused_source: counters.host_texture_shadows_refused[5]
                .load(Ordering::Relaxed),
            host_texture_shadows_refused_dma_buf: counters.host_texture_shadows_refused[6]
                .load(Ordering::Relaxed),
            in_place_checks_run: counters.in_place_checks_run.load(Ordering::Relaxed),
            in_place_checks_passed: counters.in_place_checks_passed.load(Ordering::Relaxed),
            in_place_checks_failed: counters.in_place_checks_failed.load(Ordering::Relaxed),
            in_place_checks_unchecked: counters.in_place_checks_unchecked.load(Ordering::Relaxed),
            in_place_checks_recalled: counters.in_place_checks_recalled.load(Ordering::Relaxed),
            fresh_page_buffers_made: counters.fresh_page_buffers_made.load(Ordering::Relaxed),
            fresh_page_bytes: counters.fresh_page_bytes.load(Ordering::Relaxed),
            new_buffer_bytes_cleared: counters.new_buffer_bytes_cleared.load(Ordering::Relaxed),
            new_buffer_zero_fills: counters.new_buffer_zero_fills.load(Ordering::Relaxed),
            views_through_swizzle: counters.views_through_swizzle.load(Ordering::Relaxed),
            view_shadows: counters.view_shadows.load(Ordering::Relaxed),
            attachment_reads_through_copies: counters
                .attachment_reads_through_copies
                .load(Ordering::Relaxed),
            attachment_read_splits: counters.attachment_read_splits.load(Ordering::Relaxed),
            reading_draws_snapshotted: counters.reading_draws_snapshotted.load(Ordering::Relaxed),
            snapshot_inline_splits: counters.snapshot_inline_splits.load(Ordering::Relaxed),
            feedback_loop_alias_images: counters.feedback_loop_alias_images.load(Ordering::Relaxed),
            depth_attachments_moved: counters.depth_attachments_moved.load(Ordering::Relaxed),
            depth_instances_held: counters.depth_instances_held.load(Ordering::Relaxed),
            held_depths_released: counters.held_depths_released.load(Ordering::Relaxed),
            own_pixel_reads: counters.own_pixel_reads.load(Ordering::Relaxed),
            shadows_zeroed: counters.shadows_zeroed.load(Ordering::Relaxed),
            shadow_copies: counters.shadow_copies.load(Ordering::Relaxed),
            shadow_texels_copied_in: counters.shadow_texels_copied_in.load(Ordering::Relaxed),
            shadow_texels_first_copied_in: counters
                .shadow_texels_first_copied_in
                .load(Ordering::Relaxed),
            shadow_texels_first_copied_in_images: counters
                .shadow_texels_first_copied_in_images
                .load(Ordering::Relaxed),
            shadow_texels_first_copied_in_private_buffers: counters
                .shadow_texels_first_copied_in_private_buffers
                .load(Ordering::Relaxed),
            shadow_texels_first_copied_in_mapped_buffers: counters
                .shadow_texels_first_copied_in_mapped_buffers
                .load(Ordering::Relaxed),
            shadow_texels_first_copied_in_host_pages: counters
                .shadow_texels_first_copied_in_host_pages
                .load(Ordering::Relaxed),
            shadow_texels_first_copied_in_aliased_host_pages: counters
                .shadow_texels_first_copied_in_aliased_host_pages
                .load(Ordering::Relaxed),
            shadow_texels_first_copied_in_surfaces: counters
                .shadow_texels_first_copied_in_surfaces
                .load(Ordering::Relaxed),
            shadow_texels_first_copied_in_large_surfaces: counters
                .shadow_texels_first_copied_in_large_surfaces
                .load(Ordering::Relaxed),
            shadow_texels_first_copy_in_carried: counters
                .shadow_texels_first_copy_in_carried
                .load(Ordering::Relaxed),
            shadow_texels_first_copy_in_carried_large_surfaces: counters
                .shadow_texels_first_copy_in_carried_large_surfaces
                .load(Ordering::Relaxed),
            shadow_texels_first_copied_in_large_surfaces_stood_earlier: counters
                .shadow_texels_first_copied_in_large_surfaces_stood_earlier
                .load(Ordering::Relaxed),
            shadow_texels_first_copied_in_large_surfaces_unstood: counters
                .shadow_texels_first_copied_in_large_surfaces_unstood
                .load(Ordering::Relaxed),
            shadow_texels_first_copied_in_again: counters
                .shadow_texels_first_copied_in_again
                .load(Ordering::Relaxed),
            shadow_texels_copied_in_images: counters
                .shadow_texels_copied_in_images
                .load(Ordering::Relaxed),
            shadow_texels_recopied_beyond_held: counters
                .shadow_texels_recopied_beyond_held
                .load(Ordering::Relaxed),
            shadow_texels_recopied_after_changes: counters
                .shadow_texels_recopied_after_changes
                .load(Ordering::Relaxed),
            shadow_texels_recopied_after_other_shadows: counters
                .shadow_texels_recopied_after_other_shadows
                .load(Ordering::Relaxed),
            shadow_texels_recopied_after_two_ways_merges: counters
                .shadow_texels_recopied_after_two_ways_merges
                .load(Ordering::Relaxed),
            shadow_texels_recopied_after_writes: counters
                .shadow_texels_recopied_after_writes
                .load(Ordering::Relaxed),
            shadow_texels_recopied_after_writes_within: counters
                .shadow_texels_recopied_after_writes_within
                .load(Ordering::Relaxed),
            shadow_texels_recopied_after_dropped_writes: counters
                .shadow_texels_recopied_after_dropped_writes
                .load(Ordering::Relaxed),
            shadow_texels_bounded_copied_in: counters
                .shadow_texels_bounded_copied_in
                .load(Ordering::Relaxed),
            shadow_texels_recopied_in_images: counters
                .shadow_texels_recopied_in_images
                .load(Ordering::Relaxed),
            shadow_texels_recopied_in_private_buffers: counters
                .shadow_texels_recopied_in_private_buffers
                .load(Ordering::Relaxed),
            shadow_texels_recopied_in_mapped_buffers: counters
                .shadow_texels_recopied_in_mapped_buffers
                .load(Ordering::Relaxed),
            shadow_texels_recopied_in_host_pages: counters
                .shadow_texels_recopied_in_host_pages
                .load(Ordering::Relaxed),
            shadow_texels_recopied_in_aliased_host_pages: counters
                .shadow_texels_recopied_in_aliased_host_pages
                .load(Ordering::Relaxed),
            shadow_texels_recopied_in_surfaces: counters
                .shadow_texels_recopied_in_surfaces
                .load(Ordering::Relaxed),
            shadow_texels_recopied_in_large_surfaces: counters
                .shadow_texels_recopied_in_large_surfaces
                .load(Ordering::Relaxed),
            shadow_texels_copied_back: counters.shadow_texels_copied_back.load(Ordering::Relaxed),
            shadow_copies_back_dispatched: counters
                .shadow_copies_back_dispatched
                .load(Ordering::Relaxed),
            shadow_copies_in_dispatched: counters
                .shadow_copies_in_dispatched
                .load(Ordering::Relaxed),
            shadow_keeps_from_staging: counters.shadow_keeps_from_staging.load(Ordering::Relaxed),
            dirty_copies_in_gated: dirty.gated.load(Ordering::Relaxed),
            dirty_copies_in_whole: dirty.whole.load(Ordering::Relaxed),
            dirty_copies_in_skipped: dirty.skipped.load(Ordering::Relaxed),
            dirty_copy_in_bytes_whole: dirty.bytes_whole.load(Ordering::Relaxed),
            dirty_copy_in_bytes_copied: dirty.bytes_copied.load(Ordering::Relaxed),
            dirty_stamps_asked: dirty.stamps.load(Ordering::Relaxed),
            dirty_waits: dirty.waits.load(Ordering::Relaxed),
            dirty_waits_unanswered: dirty.waits_unanswered.load(Ordering::Relaxed),
            dirty_wait_nanos: dirty.wait_nanos.load(Ordering::Relaxed),
            shadow_merge_copies_dispatched: counters
                .shadow_merge_copies_dispatched
                .load(Ordering::Relaxed),
            shadow_merge_dispatches_planned: counters
                .shadow_merge_dispatches_planned
                .load(Ordering::Relaxed),
            shadow_merge_words_examined: counters
                .shadow_merge_words_examined
                .load(Ordering::Relaxed),
            shadow_merge_dispatches_in_blocks: counters
                .shadow_merge_dispatches_in_blocks
                .load(Ordering::Relaxed),
            shadow_merge_scratches_reused: counters
                .shadow_merge_scratches_reused
                .load(Ordering::Relaxed),
            shadow_merge_scratches_zeroed: counters
                .shadow_merge_scratches_zeroed
                .load(Ordering::Relaxed),
            row_copies: counters.row_copies.load(Ordering::Relaxed),
            row_copy_regions: counters.row_copy_regions.load(Ordering::Relaxed),
            row_copy_regions_most: counters.row_copy_regions_most.load(Ordering::Relaxed),
            row_copies_dispatched: counters.row_copies_dispatched.load(Ordering::Relaxed),
            shadow_merges_fetched: counters.shadow_merges_fetched.load(Ordering::Relaxed),
            shadow_merges_kept_in_images: counters
                .shadow_merges_kept_in_images
                .load(Ordering::Relaxed),
            shadow_merge_kept_images_made: counters
                .shadow_merge_kept_images_made
                .load(Ordering::Relaxed),
            shadow_merge_kept_images_followed: counters
                .shadow_merge_kept_images_followed
                .load(Ordering::Relaxed),
            shadow_merge_kept_images_live: counters
                .shadow_merge_kept_images_live
                .load(Ordering::Relaxed),
            shadow_attachment_passes: counters.shadow_attachment_passes.load(Ordering::Relaxed),
            shadow_attachment_textures: counters.shadow_attachment_textures.load(Ordering::Relaxed),
            shadow_attachment_loads_within: counters
                .shadow_attachment_loads_within
                .load(Ordering::Relaxed),
            shadow_attachment_loads_whole: counters
                .shadow_attachment_loads_whole
                .load(Ordering::Relaxed),
            shadow_attachment_clears: counters.shadow_attachment_clears.load(Ordering::Relaxed),
            shadow_attachment_clears_in_part: counters
                .shadow_attachment_clears_in_part
                .load(Ordering::Relaxed),
            shadow_attachment_loads_untouched: counters
                .shadow_attachment_loads_untouched
                .load(Ordering::Relaxed),
            shadow_attachment_tiles_covered: counters
                .shadow_attachment_tiles_covered
                .load(Ordering::Relaxed),
            shadow_attachment_tiles_bounded: counters
                .shadow_attachment_tiles_bounded
                .load(Ordering::Relaxed),
            shadows_sharing_truth: counters.shadows_sharing_truth.load(Ordering::Relaxed),
            owned_descriptor_pools: counters.owned_descriptor_pools.load(Ordering::Relaxed),
            owned_kernels_made: counters.owned_kernels_made.load(Ordering::Relaxed),
            owned_kernel_spirv_bytes: counters.owned_kernel_spirv_bytes.load(Ordering::Relaxed),
            ycbcr_conversion_scratch_buffers: counters
                .ycbcr_conversion_scratch_buffers
                .load(Ordering::Relaxed),
            ycbcr_conversion_scratch_bytes: counters
                .ycbcr_conversion_scratch_bytes
                .load(Ordering::Relaxed),
            shadow_texels_kept: counters.shadow_texels_kept.load(Ordering::Relaxed),
            bounded_shadow_copies: counters.bounded_shadow_copies.load(Ordering::Relaxed),
            replay_logs_kept: counters.replay_logs_kept.load(Ordering::Relaxed),
            replay_log_bytes: counters.replay_log_bytes.load(Ordering::Relaxed),
            encoders_reencoded: counters.encoders_reencoded.load(Ordering::Relaxed),
            segments_recorded_again: counters.segments_recorded_again.load(Ordering::Relaxed),
            replay_tables_pointed: counters.replay_tables_pointed.load(Ordering::Relaxed),
            reencodes_refused: counters.reencodes_refused.load(Ordering::Relaxed),
            fences_made: self.shared().fences.made(),
            pipeline_libraries_made: counters.pipeline_libraries_made.load(Ordering::Relaxed),
            fragment_shader_libraries_made: counters
                .fragment_shader_libraries_made
                .load(Ordering::Relaxed),
            pipelines_linked: counters.pipelines_linked.load(Ordering::Relaxed),
            dispatch_pipelines_pooled: counters.dispatch_pipelines_pooled.load(Ordering::Relaxed),
            tile_pipelines_pooled: counters.tile_pipelines_pooled.load(Ordering::Relaxed),
            upload_chunks_in_device_memory: (counters.upload_chunks_in_device_memory)
                .load(Ordering::Relaxed),
            tile_dispatches_recorded: counters.tile_dispatches_recorded.load(Ordering::Relaxed),
        }
    }
}

/// Why a no-copy buffer's bytes lie in a copy of the caller's pages (`copied.rs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CopiedBecause {
    /// The caller's gate refused their import (`HostMemory::with_import_gate`).
    Refused,
    /// The import the device and the gate admitted failed.
    Failed,
    /// A rebind's new bytes lie at another offset in their page than the old, where a texture
    /// over the buffer was planned at the old offset: no import lays them there.
    Moved,
}
