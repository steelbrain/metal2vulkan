//! Counts of work a device's commit path does that the lanes hold to bounds: each counts a piece of
//! work the runtime once repeated at every commit or every change, so a lane fails when it is
//! repeated again; and the textures that took their storage form, which a lane holds to once each;
//! and the buffers made with the device-address usage, which every encoder that reads pointers made
//! resident one by one on MoltenVK; and the completion waits that missed their wake, which a test
//! holds to none. Not part of the API.

use super::buffer::{KeptBecause, ReadApart};
use super::churn_counts::{VkMaking, VkRelease};
use super::command_buffer::{Alone, GroupEnd};
use super::degradation::{Degradation, Told};
use super::device::lock;
use super::encoder_fold::Refusal;
use super::heap_sets::Unkept;
use super::memory::FreeCause;
use super::recorder::CommandsRecorded;
use super::render::{LeadingShare, SplitReason};
use super::scheduler::{HeldFor, LeftAtCommit};
use super::shadow::{Backing, CarryTally, CopyCause, Lost, ShadowCopy, ShadowSource};
use super::shadow_standing::{LetGo, Missed, Unskipped};
use super::texture::faces::Unturned;
use super::Device;
use crate::shards::{Mine, Sharded, Tallies, Tally};
use ash::vk;
use metal2vulkan::pipeline::SlotClass;
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

/// The copied host buffers' counts ([`Counters::copied_tallies`]), for the crate's tests.
#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CopiedTallies {
    pub(crate) copy_in_bytes: u64,
    pub(crate) merge_back_bytes: u64,
    pub(crate) dropped_bytes: u64,
    pub(crate) bases_made: u64,
    pub(crate) bases_refused: u64,
    pub(crate) base_blocks_read: u64,
    pub(crate) base_blocks_read_back: u64,
    pub(crate) compare_threads: u64,
    pub(crate) unnamed_reach_copy_ins: u64,
    pub(crate) unnamed_reach_bytes: u64,
    pub(crate) unnamed_reach_twin_copy_ins: u64,
    pub(crate) unnamed_reach_twin_bytes: u64,
}

/// A device's counters (`DeviceShared::counts`), each only ever added to.
#[derive(Default)]
pub(crate) struct Counters {
    /// Released objects and shadow keeps by kind (`churn_counts.rs`).
    pub(crate) churn: super::churn_counts::ChurnCounters,
    /// The counts commands add as their encoders record them, by encoder and by thread.
    commands: CommandTallies,
    command_pools_made: AtomicU64,
    command_buffers_made: AtomicU64,
    spare_command_pools_taken: AtomicU64,
    spare_command_pools_taken_past_first: AtomicU64,
    spare_command_pools_reset: AtomicU64,
    /// The command buffers the spare command pools freed, by why (`spares.rs`): past a pool's
    /// recent uses, past `HELD` of a pool given back, shed by a spare, and every one of a pool
    /// given back by a command buffer that recorded `RECORDED` commands or more; then the trims
    /// of their pools.
    spare_command_buffers_freed: [AtomicU64; 4],
    spare_command_pool_trim_rounds: AtomicU64,
    spare_command_buffers_trimmed_in_rounds: AtomicU64,
    spare_command_pool_trims: AtomicU64,
    live_tables_uploaded: AtomicU64,
    timeline_waits: AtomicU64,
    fence_waits: AtomicU64,
    missed_completion_wakes: AtomicU64,
    completion_arms: AtomicU64,
    completion_arms_after_a_wake: AtomicU64,
    stall_probes_sent: AtomicU64,
    wedges_declared: AtomicU64,
    watchdog_timeouts: AtomicU64,
    unrun_command_buffers: AtomicU64,
    longest_stall_nanos: AtomicU64,
    completion_reads_ran_ahead: AtomicU64,
    gate_waits: AtomicU64,
    gate_waits_slept: AtomicU64,
    gate_wakes_found_shut: AtomicU64,
    trail_waits: AtomicU64,
    trail_waits_blocked: AtomicU64,
    completed_where_seen: AtomicU64,
    batches_held_for_another_queue: AtomicU64,
    command_buffers_completed_after_another_queue: AtomicU64,
    completion_seen_handlers_run: AtomicU64,
    completion_seen_handlers_where_seen: AtomicU64,
    command_buffers_unsubmitted: AtomicU64,
    sync_files_exported: AtomicU64,
    sync_files_after_host_copies: AtomicU64,
    scheduled_where_submitted: AtomicU64,
    timestamp_slots_taken: AtomicU64,
    timestamp_slots_reset_on_host: AtomicU64,
    calibrations: AtomicU64,
    calibration_readings: AtomicU64,
    calibration_deviation_most: AtomicU64,
    calibration_disagreements: AtomicU64,
    loosely_calibrated_gpu_times: AtomicU64,
    storage_forms_taken: AtomicU64,
    addressable_buffers_made: AtomicU64,
    mipmaps_filtered: AtomicU64,
    mipmap_first_levels_fetched: AtomicU64,
    pipelines_pooled: AtomicU64,
    stage_entries_specialized: AtomicU64,
    stage_entries_shared: AtomicU64,
    functions_translated: AtomicU64,
    widest_pin_kernels_emulated: AtomicU64,
    unit_axes_variants_begun: AtomicU64,
    unit_axes_dispatches: AtomicU64,
    unit_axes_meanwhile: AtomicU64,
    functions_shared: AtomicU64,
    linked_use_questions_asked: AtomicU64,
    descriptor_keys_built: AtomicU64,
    shader_modules_made: AtomicU64,
    shader_modules_shared: AtomicU64,
    pipeline_layouts_made: AtomicU64,
    pipeline_layouts_shared: AtomicU64,
    variant_pipelines_made: AtomicU64,
    flights_followed: AtomicU64,
    makings_held_for_compile_memory: AtomicU64,
    compile_memory_held_nanos: AtomicU64,
    heap_trims: AtomicU64,
    predicated_loop_runs_again: AtomicU64,
    variants_taken_back: AtomicU64,
    unlisted_argument_sampler_buffers: AtomicU64,
    draws_dropped_unmade: AtomicU64,
    host_step_sampler_sets_made: AtomicU64,
    host_steps_recorded: AtomicU64,
    host_step_nanos: AtomicU64,
    host_step_pipelines_made: AtomicU64,
    host_step_chunks_kept: AtomicU64,
    host_step_chunks_recorded_again: AtomicU64,
    host_step_chunk_copies_put_off: AtomicU64,
    host_step_records_compared: AtomicU64,
    host_step_compare_nanos: AtomicU64,
    replayed_heap_draws_unshared: AtomicU64,
    replayed_heap_draws_writing: AtomicU64,
    replayed_heap_draws_noted: AtomicU64,
    replayed_heap_draws_over_sampler_tables: AtomicU64,
    replayed_heap_draws_table_textures: AtomicU64,
    commit_refusals_held: AtomicU64,
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
    pooled_backings_taken: AtomicU64,
    pooled_backings_taken_in_flight: AtomicU64,
    storage_placeholders_made: AtomicU64,
    storage_placeholder_attachments: AtomicU64,
    color_resolve_passes_made: AtomicU64,
    color_resolve_passes_kept: AtomicU64,
    ahead_nanos: AtomicU64,
    driver_diagnostics_withheld: AtomicU64,
    heap_texture_refused_withheld: AtomicU64,
    twin_decode_failed_withheld: AtomicU64,
    late_reservation_withheld: AtomicU64,
    queue_limits_reached_withheld: AtomicU64,
    stand_ins_made: AtomicU64,
    stand_in_bytes: AtomicU64,
    stale_detached_reaches: AtomicU64,
    detached_storage_reaches: AtomicU64,
    heap_slots_rewritten: AtomicU64,
    heap_slot_writes_deferred: AtomicU64,
    heap_arrays_copied: AtomicU64,
    heap_textures_tiled: AtomicU64,
    heap_textures_refit: AtomicU64,
    heap_sample_move_images_made: AtomicU64,
    heap_sample_move_images_lent: AtomicU64,
    requirement_images_made: AtomicU64,
    requirement_limits_asked: AtomicU64,
    arena_pools_made: AtomicU64,
    heap_texture_lossless_layouts: AtomicU64,
    heap_texture_block_compressed_layouts: AtomicU64,
    heap_texture_untiled_layouts: AtomicU64,
    fills_recorded: AtomicU64,
    buffer_copies_merged: AtomicU64,
    ycbcr_refreshes: AtomicU64,
    ycbcr_whole_conversions: AtomicU64,
    ycbcr_textures_in_place: AtomicU64,
    texel_rows_textures: AtomicU64,
    ycbcr_images_kept: [AtomicU64; 7],
    ycbcr_planes_images: AtomicU64,
    fd_blocks_awaited: AtomicU64,
    fd_blocks_refused: AtomicU64,
    ycbcr_planes_images_kept: [AtomicU64; 4],
    ycbcr_device_conversions: AtomicU64,
    ycbcr_device_samples: AtomicU64,
    attachment_clears_recorded: AtomicU64,
    image_blits_recorded: AtomicU64,
    query_result_copies_recorded: AtomicU64,
    visibility_queries_begun: AtomicU64,
    indirect_draws_recorded: AtomicU64,
    indirect_records_read: AtomicU64,
    indirect_record_runs_read: AtomicU64,
    indirect_record_bytes_copied: AtomicU64,
    indirect_record_bytes_gpu_copied: AtomicU64,
    indirect_record_bytes_gpu_reset: AtomicU64,
    indirect_record_reset_gpu_nanos: AtomicU64,
    icb_records_moved_to_host: AtomicU64,
    icb_record_bytes_moved: AtomicU64,
    icb_records_staged_accesses: AtomicU64,
    indirect_write_residencies_unmarked: AtomicU64,
    indirect_chunks_kept: AtomicU64,
    indirect_commands_kept: AtomicU64,
    kept_contexts_made: AtomicU64,
    kept_resident_keys_compared: AtomicU64,
    gpu_form_commands: AtomicU64,
    merged_draw_commands: AtomicU64,
    gpu_form_device_local_bytes: AtomicU64,
    gpu_pool_blocks_made: AtomicU64,
    gpu_pool_blocks_released: AtomicU64,
    gpu_pool_leases_peak: AtomicU64,
    internal_submissions: AtomicU64,
    queue_submissions: AtomicU64,
    submit_infos: AtomicU64,
    command_buffers_submitted: AtomicU64,
    batch_submit_nanos: AtomicU64,
    batch_submit_cpu_nanos: AtomicU64,
    submit_long_calls: AtomicU64,
    submit_long_call_nanos: AtomicU64,
    submit_long_call_cpu_nanos: AtomicU64,
    submit_stalls_on_work: AtomicU64,
    submits_behind_idle_stalls: AtomicU64,
    wait_behind_idle_stall_longest_nanos: AtomicU64,
    command_buffers_begun: AtomicU64,
    encoder_command_buffers_begun: AtomicU64,
    command_buffers_begun_by: [AtomicU64; Begun::COUNT],
    render_encoders_ended: AtomicU64,
    render_passes_begun_by: [AtomicU64; PassBegun::COUNT],
    instance_splits_by: [AtomicU64; Split::COUNT],
    rate_map_pieces: AtomicU64,
    rate_map_index_ranges: AtomicU64,
    rate_map_pieces_timed: AtomicU64,
    rate_map_piece_gpu_nanos: AtomicU64,
    rate_map_piece_gpu_nanos_most: AtomicU64,
    rate_map_probes: AtomicU64,
    rate_map_probe_gpu_nanos: AtomicU64,
    rate_map_probe_wait_nanos: AtomicU64,
    rate_map_probe_parts: AtomicU64,
    rate_map_steps_unfanned: AtomicU64,
    rate_map_steps_nothing_to_probe: AtomicU64,
    rate_map_steps_probing_none: AtomicU64,
    rate_map_gpu_routes: AtomicU64,
    rate_map_gpu_route_wait_nanos: AtomicU64,
    rate_map_flagged_draws: AtomicU64,
    rate_map_flagged_triangles: AtomicU64,
    rate_map_fanned_draws: AtomicU64,
    rate_map_fanned_few_draws: AtomicU64,
    rate_map_parted_draws: AtomicU64,
    rate_map_parted_region_triangles: AtomicU64,
    rate_map_region_runs: AtomicU64,
    rate_map_fan_out_pipelines_ready: AtomicU64,
    rate_map_fan_out_pipelines_waited: AtomicU64,
    rate_map_fan_tables_uploaded: AtomicU64,
    rate_map_region_copy_draws: AtomicU64,
    rate_map_region_copy_recordings: AtomicU64,
    rate_map_run_region_draws: AtomicU64,
    rate_map_runs_every_region: AtomicU64,
    rate_map_straddling_triangles: AtomicU64,
    rate_map_whole_copy_vertex_regions: AtomicU64,
    rate_map_own_copy_vertex_regions: AtomicU64,
    rate_map_run_vertex_regions: AtomicU64,
    rate_map_fanned_vertices: AtomicU64,
    rate_map_fanned_box_vertex_regions: AtomicU64,
    rate_map_fanned_vertices_boxed: [AtomicU64; 3],
    rate_map_boxed_draws: AtomicU64,
    rate_map_gpu_routed_draws: AtomicU64,
    rate_map_gpu_fanned_draws: AtomicU64,
    rate_map_gpu_copied_draws: AtomicU64,
    rate_map_boxed_vertex_regions: AtomicU64,
    encoder_primaries_begun_ahead: AtomicU64,
    encoder_primaries_begun_ahead_missed: AtomicU64,
    primaries_begun_ahead: AtomicU64,
    barrier_buffers_kept: AtomicU64,
    stamped_buffers_kept: AtomicU64,
    stamped_buffers_reused: AtomicU64,
    finales_folded: AtomicU64,
    finales_apart: AtomicU64,
    grouped_submissions: AtomicU64,
    batches_grouped: AtomicU64,
    /// By `GroupEnd`, in its order.
    group_ends: [AtomicU64; 9],
    texture_faces_made: AtomicU64,
    attached_read_swaps: AtomicU64,
    attached_read_swap_reads: AtomicU64,
    attached_read_swap_texels: AtomicU64,
    attached_read_swap_fixes: AtomicU64,
    attached_read_swap_fix_texels: AtomicU64,
    attached_read_swaps_short_area: AtomicU64,
    attached_read_swap_area_texels: AtomicU64,
    /// By `Unturned`, in its order.
    attached_read_unswapped: [AtomicU64; Unturned::COUNT],
    attached_read_unswapped_texels: [AtomicU64; Unturned::COUNT],
    /// Batches submitted, by how many earlier batches of their `VkQueue` were pending then: none,
    /// one, two or three, four or more.
    batches_by_pending: [AtomicU64; 4],
    /// By `Alone`, in its order: the batches on a `VkQueue` whose batches group that went in a
    /// kernel submission of their own.
    batches_alone: [AtomicU64; Alone::COUNT],
    batches_of_unshared_queues: AtomicU64,
    groups_kept_pending: AtomicU64,
    groups_sent_for_waits: AtomicU64,
    scheduler_slice_rescues: AtomicU64,
    /// By `HeldFor`, in its order: the looks that found a line's head held for each reason.
    heads_held_for: [AtomicU64; HeldFor::COUNT],
    heads_waiting_at_their_gate: AtomicU64,
    scheduled_behind_host_steps: AtomicU64,
    heads_let_go_unshared_idle: AtomicU64,
    heads_let_go_turn_idle: AtomicU64,
    heads_held_for_unshared_commits: AtomicU64,
    host_step_gates_narrowed: AtomicU64,
    heads_awaiting_check_signals: AtomicU64,
    batches_held_past_a_look: AtomicU64,
    batch_hold_nanos: AtomicU64,
    scheduler_passes_by_waiters: AtomicU64,
    batches_submitted_by_waiters: AtomicU64,
    scheduler_passes_at_commit: AtomicU64,
    scheduler_passes_at_commit_ended_early: AtomicU64,
    /// By `LeftAtCommit`, in its order: the commits of `commit_and_submit` whose own pass
    /// submitted none of the scheduler's batches, or that made none.
    left_at_commit: [AtomicU64; LeftAtCommit::COUNT],
    batches_submitted_at_commit: AtomicU64,
    batches_checked_at_commit: AtomicU64,
    commit_calls: AtomicU64,
    commit_nanos: AtomicU64,
    commit_lowering_nanos: AtomicU64,
    commit_recording_nanos: AtomicU64,
    commit_pipeline_wait_nanos: AtomicU64,
    commit_lock_wait_nanos: AtomicU64,
    commit_pass_nanos: AtomicU64,
    commit_submit_nanos: AtomicU64,
    footprints_built: AtomicU64,
    footprint_lock_waits: AtomicU64,
    command_buffer_wakes: AtomicU64,
    command_buffer_wakes_spared: AtomicU64,
    event_wakes: AtomicU64,
    event_wakes_spared: AtomicU64,
    slot_returns: AtomicU64,
    slot_return_nanos: AtomicU64,
    slot_waits: AtomicU64,
    queues_sharing_a_vk_queue: AtomicU64,
    command_queues_live_most: AtomicU64,
    unshared_queues_sharing_a_vk_queue: AtomicU64,
    queue_moves: AtomicU64,
    queue_joins: AtomicU64,
    queues_gone_long: AtomicU64,
    long_queue_settles: AtomicU64,
    queue_moves_off_unshared: AtomicU64,
    joins_withheld_from_unshared: AtomicU64,
    picks_with_none_busy: AtomicU64,
    joins_withheld_from_old_heads: AtomicU64,
    /// Command buffers' waits behind other command queues' batches (`queue_wait.rs`): of every
    /// queue's, then of unshared queues' alone.
    behind_other_queues: [BehindOtherQueues; 2],
    /// What unshared queues' waits over 16 ms waited behind (`queue_wait.rs`): the batches ahead
    /// of each, summed and the most, its longest run, summed, and those whose longest run alone
    /// was over 16 ms.
    unshared_long_waits: [AtomicU64; 4],
    queue_switch_checks_run: AtomicU64,
    queue_switch_checks_recalled: AtomicU64,
    submit_checks_run: AtomicU64,
    submit_checks_recalled: AtomicU64,
    submit_check_calls: AtomicU64,
    check_steps_aside: AtomicU64,
    check_aside_nanos: AtomicU64,
    view_check_stand_bys: AtomicU64,
    view_check_stand_by_nanos: AtomicU64,
    check_queue_submissions: AtomicU64,
    queue_sharing_untimed: AtomicU64,
    slot_wait_nanos: AtomicU64,
    completed_waits: AtomicU64,
    event_waits_met: AtomicU64,
    shared_event_waits_held: AtomicU64,
    event_signals_behind_merge: AtomicU64,
    bare_starts: AtomicU64,
    prefixed_starts: AtomicU64,
    command_barriers: AtomicU64,
    command_buffers_split_at_aliases: AtomicU64,
    unrun_instances_ended: AtomicU64,
    untouched_passes_skipped: AtomicU64,
    inline_passes: AtomicU64,
    inline_passes_split: AtomicU64,
    encoders_barriered_at_open: AtomicU64,
    encoders_folded: AtomicU64,
    folds_refused_kind: AtomicU64,
    folds_refused_tail: AtomicU64,
    folds_refused_batch: AtomicU64,
    folds_refused_state: AtomicU64,
    folds_refused_parts: AtomicU64,
    folds_refused_ends: AtomicU64,
    fold_fence_barriers: AtomicU64,
    parked_compute_taken: AtomicU64,
    record_lists_grown: AtomicU64,
    record_storage_kept: AtomicU64,
    command_buffer_states_made: AtomicU64,
    kept_draws_carried: AtomicU64,
    kept_draws_carried_across: AtomicU64,
    kept_draws_carry_dropped: [AtomicU64; CarryDropped::COUNT],
    kept_draws_carried_aside: AtomicU64,
    kept_draws_uncarried_attached: AtomicU64,
    encoders_uncarried: [AtomicU64; Uncarried::COUNT],
    encoders_none_kept_next: [AtomicU64; NoneKeptNext::COUNT],
    encoders_last_draw_empty: AtomicU64,
    inline_split_instances: AtomicU64,
    render_passes_continued: AtomicU64,
    inline_memoryless_write_splits: AtomicU64,
    passes_not_inline: [AtomicU64; NotInline::COUNT],
    passes_not_inline_actions_by: [AtomicU64; ActionsRefused::COUNT],
    commits_handed_off: AtomicU64,
    images_started_on_the_host: AtomicU64,
    images_started_zeroed: AtomicU64,
    private_textures_zero_filled: AtomicU64,
    private_textures_overwritten_unfilled: AtomicU64,
    memoryless_textures_zeroed: AtomicU64,
    host_texel_copies: AtomicU64,
    heap_shadow_cpu_copies: AtomicU64,
    depth_uploads_clamped: AtomicU64,
    texel_shadows_made: AtomicU64,
    texel_shadow_bytes: AtomicU64,
    texel_shadow_reads: AtomicU64,
    texel_shadows_dropped: AtomicU64,
    staging_submissions: AtomicU64,
    copy_queue_batches: AtomicU64,
    staging_copies_queued: AtomicU64,
    staging_moves_queued: AtomicU64,
    staging_flushes: AtomicU64,
    staging_gpu_waits: AtomicU64,
    staging_buffers_grown: AtomicU64,
    staging_buffer_waits: AtomicU64,
    fill_queue_submissions: AtomicU64,
    fill_queue_waits: AtomicU64,
    fill_queue_waits_unused: AtomicU64,
    first_work_waits: AtomicU64,
    first_work_waits_unused: AtomicU64,
    uncached_host_images: AtomicU64,
    cpu_block_decodes: AtomicU64,
    block_decode_passes: AtomicU64,
    shadow_twin_decodes: AtomicU64,
    chroma_device_twins: AtomicU64,
    chroma_device_decodes: AtomicU64,
    second_twins: AtomicU64,
    second_twin_checks: AtomicU64,
    second_twin_decodes: AtomicU64,
    queue_drains: AtomicU64,
    stand_in_copies_made: AtomicU64,
    stand_in_copies_made_at_draws: AtomicU64,
    resident_textures_looked_at: AtomicU64,
    resident_resources_joined: AtomicU64,
    ended_residencies_looked_at: AtomicU64,
    resident_address_indexes_made: AtomicU64,
    resident_kept_keys_listed: AtomicU64,
    root_table_ranges_noted: AtomicU64,
    heap_tables_worked_out: AtomicU64,
    argument_samplers_taken: AtomicU64,
    bound_notices_made: AtomicU64,
    linear_passes_in_place: AtomicU64,
    linear_passes_through_copies: AtomicU64,
    linear_merge_scratches_made: AtomicU64,
    linear_merge_scratches_reused: AtomicU64,
    linear_merges_recorded: AtomicU64,
    linear_merges_taken_over: AtomicU64,
    linear_runs_reloaded: AtomicU64,
    kept_bytes_passes: AtomicU64,
    kept_texels_passes: AtomicU64,
    color_group_passes: AtomicU64,
    color_group_twins: AtomicU64,
    color_group_draws_skipped: AtomicU64,
    host_deallocators_delayed: AtomicU64,
    host_import_releases_posted: AtomicU64,
    host_import_releases_pending: AtomicU64,
    host_keep_alives_kept: AtomicU64,
    host_keep_alives_covered: AtomicU64,
    host_releases_held_inside: AtomicU64,
    host_mappings_queried: AtomicU64,
    dma_buf_imports: AtomicU64,
    dma_buf_buffer_imports_dedicated: AtomicU64,
    dma_buf_memories_live: AtomicU64,
    dma_buf_image_memories_live: AtomicU64,
    dma_buf_pieces_joined: AtomicU64,
    dma_buf_shared_imports_live: AtomicU64,
    dma_buf_shared_imports_kept: AtomicU64,
    dma_buf_import_calls: AtomicU64,
    dma_buf_import_nanos: AtomicU64,
    dma_buf_join_nanos: AtomicU64,
    host_imports_released: AtomicU64,
    host_imports_released_over_most: AtomicU64,
    host_imports_reimported: AtomicU64,
    host_imports_joined_releasable: AtomicU64,
    host_imports_resident: AtomicU64,
    host_imports_held_for_good: AtomicU64,
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
    host_imports_unpinnable: AtomicU64,
    host_imports_refused_by_host: AtomicU64,
    copied_host_buffers_made: AtomicU64,
    copied_host_buffers_refused: AtomicU64,
    copied_host_buffers_device_refused: AtomicU64,
    copied_host_buffers_degraded: AtomicU64,
    copied_rebind_pages_moved: AtomicU64,
    copied_read_only_memory: AtomicU64,
    copied_unwritable_pages: AtomicU64,
    copied_over_live_imports: AtomicU64,
    copied_rebind_pages_made: AtomicU64,
    copied_rebind_images_made: AtomicU64,
    copied_heaps_made: AtomicU64,
    copied_host_buffers_shared: AtomicU64,
    copied_copy_in_bytes: AtomicU64,
    copied_copy_in_bytes_skipped: AtomicU64,
    copied_copy_ins_unused: AtomicU64,
    copied_copy_ins_after_waits: AtomicU64,
    copied_merge_back_bytes: AtomicU64,
    copied_merges_compared: AtomicU64,
    copied_merges_skipped: AtomicU64,
    copied_dropped_bytes: AtomicU64,
    copied_bases_made: AtomicU64,
    copied_bases_refused: AtomicU64,
    copied_base_blocks_read: AtomicU64,
    copied_base_blocks_read_back: AtomicU64,
    copied_compare_threads: AtomicU64,
    copied_copy_ins_compared: AtomicU64,
    copied_copy_in_bytes_compared: AtomicU64,
    copied_unnamed_reach_copy_ins: AtomicU64,
    copied_unnamed_reach_bytes_compared: AtomicU64,
    copied_unnamed_reach_twin_copy_ins: AtomicU64,
    copied_unnamed_reach_twin_bytes_compared: AtomicU64,
    read_twins_made: AtomicU64,
    read_twins_unimported: AtomicU64,
    read_twins_uncontained: AtomicU64,
    read_twin_blit_reads: AtomicU64,
    read_twin_reads_after_writes: AtomicU64,
    read_twin_writes_refused: AtomicU64,
    read_twin_textures_made: AtomicU64,
    read_twin_textures_not_in_place: AtomicU64,
    read_twin_texture_shadows: AtomicU64,
    read_twin_slot_reads: AtomicU64,
    host_ram_copies: AtomicU64,
    host_ram_copy_bytes: AtomicU64,
    host_ram_copy_dispatches: AtomicU64,
    host_ram_copy_regions: AtomicU64,
    host_ram_chunks_imported: AtomicU64,
    host_ram_chunk_import_nanos: AtomicU64,
    host_ram_copy_import_waits: AtomicU64,
    host_ram_copy_import_wait_nanos: AtomicU64,
    host_ram_copies_staged: AtomicU64,
    host_ram_copy_plans_reused: AtomicU64,
    host_ram_copy_tables_kept: AtomicU64,
    host_ram_staged_fills_gated: AtomicU64,
    host_ram_staged_fill_nanos: AtomicU64,
    host_ram_scatters_staged: AtomicU64,
    host_ram_staged_drain_nanos: AtomicU64,
    host_ram_dma_buf_imports: AtomicU64,
    host_ram_run_list_imports: AtomicU64,
    host_ram_run_list_import_nanos: AtomicU64,
    host_ram_pointer_imports_failed: AtomicU64,
    host_ram_dma_buf_imports_failed: AtomicU64,
    host_ram_chunks_imported_ahead: AtomicU64,
    host_ram_chunks_held_back: AtomicU64,
    host_ram_chunks_faulted_ahead: AtomicU64,
    host_ram_chunks_given_back: AtomicU64,
    host_ram_chunks_released: AtomicU64,
    memory_frees_paced: AtomicU64,
    memory_frees_paced_nanos: AtomicU64,
    host_ram_chunk_bytes: AtomicU64,
    host_ram_chunks_live: AtomicU64,
    chunk_bound_buffers_made: AtomicU64,
    run_imports_made: AtomicU64,
    run_imports_live: AtomicU64,
    dma_buf_sparse_buffers: AtomicU64,
    dma_buf_sparse_binds: AtomicU64,
    dma_buf_images: AtomicU64,
    dma_buf_images_at_offset: AtomicU64,
    dma_buf_images_parked: AtomicU64,
    dma_buf_images_taken: AtomicU64,
    dma_buf_images_missed: AtomicU64,
    dma_buf_images_parked_released: AtomicU64,
    dma_buf_images_in_import: AtomicU64,
    exported_dma_buf_images: AtomicU64,
    exported_dma_buf_images_parked: AtomicU64,
    exported_dma_buf_images_taken: AtomicU64,
    exported_dma_buf_images_parked_released: AtomicU64,
    dma_buf_image_fallbacks: AtomicU64,
    modifier_feature_queries: AtomicU64,
    linear_alignments_probed: AtomicU64,
    linear_alignments_recalled: AtomicU64,
    shadowed_textures: AtomicU64,
    shadows_zeroed: AtomicU64,
    shadows_over_fresh_pages: AtomicU64,
    shadows_left_unmade: AtomicU64,
    shadows_made_at_use: AtomicU64,
    host_textures_in_place: AtomicU64,
    host_textures_in_place_written: AtomicU64,
    device_copy_textures: AtomicU64,
    managed_device_copies: AtomicU64,
    managed_device_copies_made: AtomicU64,
    managed_device_copy_bytes: AtomicU64,
    managed_device_copies_in_place: AtomicU64,
    managed_device_copies_zeroed: AtomicU64,
    managed_bytes_zeroed: AtomicU64,
    managed_bytes_copied_in: AtomicU64,
    managed_bytes_copied_out: AtomicU64,
    managed_ranges_modified: AtomicU64,
    managed_bytes_modified: AtomicU64,
    managed_resources_synchronized: AtomicU64,
    managed_bytes_synchronized: AtomicU64,
    managed_synchronizes_unwritten: AtomicU64,
    managed_synchronizes_whole: AtomicU64,
    managed_synchronizes_topped_up: AtomicU64,
    managed_bytes_unwritten_skipped: AtomicU64,
    managed_staging_bytes: AtomicU64,
    managed_writes_undone: AtomicU64,
    managed_bytes_staged_in: AtomicU64,
    managed_bytes_copied_in_from_ram: AtomicU64,
    managed_ranges_inlined: AtomicU64,
    managed_uploads_unbarriered: AtomicU64,
    managed_upload_barriers_narrowed: AtomicU64,
    managed_pages_alone_refused: AtomicU64,
    managed_pages_alone_over_imports: AtomicU64,
    managed_pages_alone_unwritable: AtomicU64,
    managed_pages_in_ram_chunks: AtomicU64,
    dma_buf_textures_in_place: AtomicU64,
    fresh_page_textures_in_place: AtomicU64,
    copied_textures_in_place: AtomicU64,
    pitched_host_images: AtomicU64,
    host_texture_shadows_exact: AtomicU64,
    host_texture_shadows_render_target: AtomicU64,
    host_texture_shadows_shader_write: AtomicU64,
    host_texture_shadows_store_width: AtomicU64,
    host_texture_shadows_check_pending: AtomicU64,
    host_texture_shadows_check_failed: AtomicU64,
    host_texture_shadows_unchecked: AtomicU64,
    host_texture_shadows_refused: [AtomicU64; 7],
    in_place_checks_run: AtomicU64,
    in_place_checks_passed: AtomicU64,
    in_place_checks_failed: AtomicU64,
    in_place_checks_unchecked: AtomicU64,
    in_place_checks_recalled: AtomicU64,
    host_textures_moved_in_place: AtomicU64,
    host_texture_shadows_freed_after_move: AtomicU64,
    fresh_page_buffers_made: AtomicU64,
    fresh_page_bytes: AtomicU64,
    new_buffer_bytes_cleared: AtomicU64,
    new_buffer_zero_fills: AtomicU64,
    shadow_copies: AtomicU64,
    shadow_copies_in_against_standing: AtomicU64,
    shadow_copies_in_skipped_by_standing: AtomicU64,
    shadow_texels_skipped_by_standing: AtomicU64,
    shadow_texels_unskipped: [AtomicU64; 7],
    shadow_texels_unskipped_standing: [AtomicU64; 5],
    shadow_texels_never_stood: [AtomicU64; 5],
    shadow_texels_copied_in: AtomicU64,
    shadow_texels_first_copied_in: AtomicU64,
    shadow_texels_first_copied_in_images: AtomicU64,
    shadow_texels_first_copied_in_private_buffers: AtomicU64,
    shadow_texels_first_copied_in_mapped_buffers: AtomicU64,
    shadow_texels_first_copied_in_host_pages: AtomicU64,
    shadow_texels_first_copied_in_aliased_host_pages: AtomicU64,
    shadow_texels_first_copied_in_surfaces: AtomicU64,
    shadow_texels_first_copied_in_large_surfaces: AtomicU64,
    shadow_texels_first_copied_in_private_surfaces: AtomicU64,
    shadow_texels_first_copied_in_private_images: AtomicU64,
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
    shadow_copies_back_spliced: AtomicU64,
    shadow_merges_on_host: AtomicU64,
    shadow_merge_host_nanos: AtomicU64,
    shadow_merge_bytes_on_host: AtomicU64,
    shadow_host_steps: AtomicU64,
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
    host_step_owned_kernels_made: AtomicU64,
    owned_kernel_spirv_bytes: AtomicU64,
    ycbcr_conversion_scratch_buffers: AtomicU64,
    ycbcr_conversion_scratch_bytes: AtomicU64,
    shadow_texels_kept: AtomicU64,
    bounded_shadow_copies: AtomicU64,
    views_through_swizzle: AtomicU64,
    alpha_textures_held_in_red: AtomicU64,
    view_shadows: AtomicU64,
    attachment_read_splits: AtomicU64,
    attachment_read_splits_spared: AtomicU64,
    attachment_read_clears: AtomicU64,
    fetch_draw_barriers: AtomicU64,
    reading_draws_snapshotted: AtomicU64,
    reading_draws_snapshotted_leading: AtomicU64,
    reading_draws_in_place: AtomicU64,
    snapshot_inline_splits: AtomicU64,
    reading_draws_split: [AtomicU64; 15],
    reading_draws_unreadable: [AtomicU64; 12],
    reading_draws_by_coverage: AtomicU64,
    reading_draws_leading_covered: [AtomicU64; 3],
    coverage_queries: AtomicU64,
    coverage_pools_made: AtomicU64,
    reading_draws_pinned: AtomicU64,
    stand_ins_after_pins: AtomicU64,
    feedback_loop_alias_images: AtomicU64,
    unobserved_padding_aliases: AtomicU64,
    kept_padding_aliases: AtomicU64,
    first_moves_kept: AtomicU64,
    first_move_bytes_kept: AtomicU64,
    pooled_pitched_images: AtomicU64,
    pooled_textures_moved_in_place: AtomicU64,
    pooled_texture_shadows_freed_after_move: AtomicU64,
    feedback_loop_shadow_images: AtomicU64,
    depth_attachments_moved: AtomicU64,
    depth_instances_held: AtomicU64,
    held_depths_released: AtomicU64,
    replay_logs_kept: AtomicU64,
    replay_log_bytes: AtomicU64,
    encoders_reencoded: AtomicU64,
    segments_recorded_again: AtomicU64,
    replay_tables_pointed: AtomicU64,
    reencodes_refused: AtomicU64,
    encoders_raced_rebind: AtomicU64,
    encoders_raced_unreplayable: AtomicU64,
    rebinds_refused_calls_in_flight: AtomicU64,
    rebinds_refused_unsubmitted: AtomicU64,
    encoder_calls_held_by_rebind: AtomicU64,
    pipeline_libraries_made: AtomicU64,
    fragment_shader_libraries_made: AtomicU64,
    pipelines_linked: AtomicU64,
    linked_pipelines_shared: AtomicU64,
    dispatch_pipelines_pooled: AtomicU64,
    region_pipelines_evicted: AtomicU64,
    tile_pipelines_pooled: AtomicU64,
    upload_chunks_in_device_memory: AtomicU64,
    render_log_batches_by_thread: AtomicU64,
    render_log_batches_by_program: AtomicU64,
    render_log_waits: AtomicU64,
    render_log_ends_by_thread: AtomicU64,
    render_log_ends_by_program: AtomicU64,
    render_log_commits_by_thread: AtomicU64,
    render_log_commits_by_program: AtomicU64,
}

/// The counts a command adds as its encoder records it, on the paths a draw, a dispatch, a tile
/// dispatch or a bind takes at every command: its binds and tables read, its shape and plan
/// settled, its `VkPipeline` looked up, its set 0 bound or pushed, its uses walked, its state
/// commands recorded. Kept apart by who adds them (`shards.rs`): the encoder that records the
/// command, while it is open, where the count is made with its core at hand
/// ([`Counters::of`], `EncoderCore::counts`), else the thread that makes it
/// ([`Counters::thread`]); so encoders and threads that count at once write apart, each add a
/// plain load and store, none an atomic another core's add contends for. A count is the sum of
/// every encoder's and thread's, open encoders' included, so it is whole whenever it is read. A
/// rarer event a command meets (a split of its pass, a making, a first use) stays an atomic of
/// the device's own ([`Counters`]): it costs its event, not every command. Those a draw of a
/// steady pass adds come first, in a shard's first 128 bytes.
/// One tally of command buffers' waits behind other command queues' batches
/// (`WorkCounts::command_buffers_behind_other_queues` and kin).
#[derive(Default)]
struct BehindOtherQueues {
    count: AtomicU64,
    over_16ms: AtomicU64,
    over_33ms: AtomicU64,
    longest_nanos: AtomicU64,
}

impl BehindOtherQueues {
    /// Its four counts: the command buffers, those over 16 ms and over 33 ms, the longest.
    fn read(&self) -> [u64; 4] {
        [
            self.count.load(Ordering::Relaxed),
            self.over_16ms.load(Ordering::Relaxed),
            self.over_33ms.load(Ordering::Relaxed),
            self.longest_nanos.load(Ordering::Relaxed),
        ]
    }
}

#[derive(Default)]
struct CommandCounters {
    stage_tables_read: AtomicU64,
    stage_tables_read_in_part: AtomicU64,
    stage_tables_read_whole_by: [AtomicU64; WholeRead::COUNT],
    draws_rebind_reshaped_by: [AtomicU64; ReshapedBy::COUNT],
    commands_noted_as_before: AtomicU64,
    uses_noted_covered: AtomicU64,
    provoking_last_pipeline_draws: AtomicU64,
    draw_states_recorded: AtomicU64,
    draws_pushing_set_zero: AtomicU64,
    set_zero_writes_pushed: AtomicU64,
    draws_as_before: AtomicU64,
    draws_rebound: AtomicU64,
    attachment_reads_through_copies: AtomicU64,
    own_pixel_reads: AtomicU64,
    draws_counted_past_early_masks: AtomicU64,
    draws_moved: AtomicU64,
    draws_moved_inside: AtomicU64,
    resident_uses_made: AtomicU64,
    argument_tables_held_at_bind: AtomicU64,
    bound_offsets_moved: AtomicU64,
    set_zero_sets_reused: AtomicU64,
    set_zero_sets_found: AtomicU64,
    resident_uses_gathered: AtomicU64,
    resident_uses_against_fragments: AtomicU64,
    resident_uses_against_attachments: AtomicU64,
    resident_uses_noted: AtomicU64,
    resident_uses_unordered_writes: AtomicU64,
    resident_uses_indirect_reads: AtomicU64,
    resident_uses_geometry_writes: AtomicU64,
    resident_uses_geometry_checked: AtomicU64,
    resident_uses_again_checked: AtomicU64,
    resident_keys_settled: AtomicU64,
    draw_heap_parts_made: AtomicU64,
    draw_resident_textures_looked_at: AtomicU64,
    no_copy_lies_read: AtomicU64,
    host_holds_looked_up: AtomicU64,
    draws_settled: AtomicU64,
    graphics_pipelines_looked_up: AtomicU64,
    draw_shapes_reused: AtomicU64,
    draw_shapes_carried: AtomicU64,
    draw_shapes_made: AtomicU64,
    draw_shapes_found: AtomicU64,
    draw_shape_compares: AtomicU64,
    dispatch_shapes_made: AtomicU64,
    dispatch_shapes_found: AtomicU64,
    dispatch_used_resources_listed: AtomicU64,
    dispatches_settled_as_before: AtomicU64,
    dispatches_settled_across_encoders: AtomicU64,
    dispatches_instantiated: AtomicU64,
    dispatches_instantiated_in_place: AtomicU64,
    dispatches_instantiated_as_before: AtomicU64,
    dispatches_from_set_aside: AtomicU64,
    set_aside_commands_kept: AtomicU64,
    set_aside_commands_instantiated: AtomicU64,
    dispatches_skipped_inert: AtomicU64,
    dispatches_planned_as_before: AtomicU64,
    dispatches_planned_across_grids: AtomicU64,
    whole_dispatches_on_partial_variants: AtomicU64,
    dispatches_pushing_set_zero: AtomicU64,
    dispatch_set_zero_writes_pushed: AtomicU64,
    dispatches_tracked: AtomicU64,
    dispatch_uses_tracked: AtomicU64,
    baked_sampler_dispatches_apart: AtomicU64,
    argument_tables_retained: AtomicU64,
    argument_objects_retained: AtomicU64,
    bind_views_worked_out: AtomicU64,
    bind_samplers_worked_out: AtomicU64,
    bind_spaces_read: AtomicU64,
    commands_ordered_inside: AtomicU64,
    set_zero_equal_shape_switches: AtomicU64,
    set_zero_writes_on_equal_shape_switches: AtomicU64,
    set_zero_writes_if_shapes_shared: AtomicU64,
    two_plane_draws: AtomicU64,
    provoking_listed_draws: AtomicU64,
    device_wide_line_draws: AtomicU64,
    line_band_draws: AtomicU64,
    nearest_width_line_draws: AtomicU64,
    far_vertex_fetch_pulled_draws: AtomicU64,
    gathered_draws: AtomicU64,
    bounded_run_draws: AtomicU64,
    tile_dispatches_recorded: AtomicU64,
    ycbcr_planes_bound: AtomicU64,
    ycbcr_planes_images_bound: AtomicU64,
    texel_rows_bound: AtomicU64,
    texel_buffer_parts_bound: AtomicU64,
    uniform_texels_bound: AtomicU64,
    uniform_buffers_bound_to_16k: AtomicU64,
    uniform_buffers_bound_past_16k: AtomicU64,
    residual_form_draws: AtomicU64,
    residual_form_makings: AtomicU64,
    draws_unkept: AtomicU64,
    draws_unkept_instance: AtomicU64,
    draws_unkept_instance_later: AtomicU64,
    draws_unkept_instance_steps: AtomicU64,
    draws_unkept_barrier: AtomicU64,
    draws_unkept_shadowed: AtomicU64,
    draws_unkept_tables: AtomicU64,
    draws_unkept_tables_heap: AtomicU64,
    draws_unkept_tables_writing: AtomicU64,
    draws_unkept_index_copied: AtomicU64,
    draws_unkept_writing: AtomicU64,
    draws_unkept_writing_resident: AtomicU64,
    draws_unkept_making: AtomicU64,
    draws_unkept_other: AtomicU64,
    draws_unlike_kept: AtomicU64,
    draws_unlike_kept_pipeline: AtomicU64,
    draws_unlike_kept_call: AtomicU64,
    draws_unlike_kept_counts: AtomicU64,
    draws_unlike_kept_index_buffer: AtomicU64,
    draws_unlike_kept_index_offset: AtomicU64,
    draws_unlike_kept_tessellation: AtomicU64,
    draws_unlike_kept_command_buffer: AtomicU64,
    draws_unlike_kept_under: AtomicU64,
    draws_unlike_kept_arguments_alone: AtomicU64,
    draws_rebind_stopped: AtomicU64,
    draws_restated: AtomicU64,
    draws_restated_scissors: AtomicU64,
    kept_draws_rehomed: AtomicU64,
    kept_draws_rehomed_carried: AtomicU64,
    kept_draws_carried_unopened: AtomicU64,
    draws_rebound_arguments: AtomicU64,
    draws_rebind_stopped_arguments: AtomicU64,
    draws_rebind_stopped_arguments_by: [AtomicU64; ArgumentsStop::COUNT],
    draws_rebound_carried: AtomicU64,
    reading_draws_snapshotted_again: AtomicU64,
    reading_draws_snapshotted_stopped: AtomicU64,
    draws_none_kept: AtomicU64,
    draws_none_kept_uncarried: AtomicU64,
    draws_carry_stopped: AtomicU64,
    draws_carry_stopped_by: [AtomicU64; CarryStop::COUNT],
    kept_draws_let_go_by: [AtomicU64; LoggedCall::COUNT],
    kept_draws_kept_across_by: [AtomicU64; LoggedCall::COUNT],
    attached_read_slots: AtomicU64,
    attached_read_slots_by: [AtomicU64; AttachedLevels::COUNT],
}

/// How the levels a draw's slot reads of a texture a color attachment of its render pass instance
/// renders meet the attached levels (`render/attached_levels.rs`, `WorkCounts::attached_read_slots`
/// and kin): the first that holds, in this order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AttachedLevels {
    /// The texture has one level.
    OneLevel,
    /// A read of the slot may reach any level of its view (`LevelReads::any`), or the module
    /// states no reads of the slot.
    AnyLevel,
    /// A read may reach the lowest attached level or one past it, and one the highest attached
    /// level or one before it (or the instance holds an attached level in the feedback loop
    /// layout), through the sampler the draw binds or at a level it names, or the draw binds the
    /// slot through no write whose sampler it knows; the lowest attached level is the texture's
    /// own first level.
    FirstLevel,
    /// The same, the lowest attached level past the texture's own first.
    Reaching,
    /// Every read reaches only levels before the lowest attached one, or only levels past the
    /// highest, but the texture's own image cannot serve them in place: the texture is no image
    /// of its own, or a sparse one, or one with a decoded twin, or one whose image lacks the
    /// sampled usage.
    Apart,
    /// Every read reaches only levels before the lowest attached one, or only levels past the
    /// highest, but a write the draw's tracked uses or the instance's unordered writes since its
    /// last barrier hold, or an attachment of the instance, meets those levels.
    Written,
    /// Every read reaches only levels before the lowest attached one, and the slot binds the
    /// texture's own image through a view of those levels, in place of its copy.
    Narrowed,
    /// Every read reaches only levels past the highest attached one, and the slot binds the
    /// texture's own image through a view of the levels from its first up to the highest it
    /// reads, in place of its copy.
    Past,
}

impl AttachedLevels {
    /// How many there are.
    pub(crate) const COUNT: usize = 8;
}

/// Why a render encoder's kept draw carried into its command buffer's next render encoder
/// (`render/carried.rs`) was let go before that encoder started from it
/// (`WorkCounts::kept_draws_carry_dropped_*`): the first that held.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CarryDropped {
    /// Another call of the command buffer came first: another encoder's opening, an event, or
    /// any other that settles its last render encoder's end, but the commit.
    Between,
    /// The next render encoder's pass is another's: another shape or compatible render pass, a
    /// color, depth or stencil attachment of another kind, memoryless colors, a visibility
    /// buffer, a rasterization-rate map, color groups, or an encoder that encodes an ended one
    /// again.
    Layout,
    /// What the kept draw read that changes with no call: a no-copy buffer moved, the device's
    /// sampler IDs' slots changed, or the next encoder's residency is of another generation.
    Under,
    /// Its queue held it from a command buffer's commit, and a later commit handed the queue
    /// another: no render encoder took it between.
    Commit,
}

impl CarryDropped {
    /// How many there are.
    pub(crate) const COUNT: usize = 4;
}

/// Why a render encoder that ended carried no kept draw into its command buffer's next render
/// encoder (`render/carried.rs`, `WorkCounts::encoders_uncarried_*`): the first that held, in this
/// order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Uncarried {
    /// It recorded no draw.
    Drawless,
    /// It drew, but kept no draw at its end: its last draw kept nothing, or a call after its
    /// last kept draw let every kept draw go.
    NoneKept,
    /// Its bind state held no settle of its last draw's state.
    Unsettled,
    /// Its current instance split into pieces: draws recorded before an execution of render
    /// commands or a draw of a pipeline being made (`render/executions.rs`, `render/pending.rs`).
    Split,
    /// Its pass is drawn in color groups, over a rasterization-rate map, or at custom sample
    /// positions.
    Pass,
    /// It encoded an ended encoder again, or recorded a draw again into another instance.
    Replay,
    /// Executed render commands, a pipeline being made, or a tile dispatch in its pass.
    Commands,
    /// A shader-side stencil or depth test, or its stand-in, in its pass.
    ShaderTests,
    /// A fragment or tile imageblock in its pass.
    Imageblock,
    /// A visibility result active at its end.
    Visibility,
    /// Its current instance renders a color through a view of another format, draws views, or
    /// holds storage attachments.
    Instance,
    /// Its last kept draw does not rebind (`Kept::rebinds`).
    Unrebound,
    /// Its last kept draw holds a depth/stencil state that fails every fragment.
    FailingState,
}

impl Uncarried {
    /// How many there are.
    pub(crate) const COUNT: usize = 13;
}

/// What a render encoder that drew but ended with no kept draw ([`Uncarried::NoneKept`]) came to
/// for its command buffer's next render encoder, as a kept draw of it would have carried
/// (`render/carried.rs`, `WorkCounts::encoders_none_kept_next_*`): the first that held.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NoneKeptNext {
    /// Its end is not one a kept draw carries from, first because its current instance split into
    /// pieces ([`Uncarried::Split`]).
    Split,
    /// Its end is not one a kept draw carries from, first for another reason of [`Uncarried`]'s.
    Ends,
    /// Another call of the command buffer came before its next render encoder's opening: another
    /// encoder's opening, an event, or any other that settles its end, but the commit.
    Between,
    /// The command buffer committed with no render encoder after it.
    Last,
    /// The next render encoder's pass is another's ([`CarryDropped::Layout`]), or it turned a
    /// color's texture to its other face.
    Layout,
    /// What changes with no call changed by the next render encoder's opening
    /// ([`CarryDropped::Under`]).
    Under,
    /// The next render encoder's pass, and what changes with no call, are as they were: a kept
    /// draw of it would have carried into it.
    Fits,
}

impl NoneKeptNext {
    /// How many there are.
    pub(crate) const COUNT: usize = 7;
}

/// Why a draw read a stage's table into its encoder's kept bind state whole, not in part
/// (`WorkCounts::stage_tables_read_whole_*`): the first of `render.rs`'s `refresh_bind_state`'s
/// checks that failed, in their order (`live_bind::ReadFrom::whole_read`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WholeRead {
    /// The state never read its tables.
    Unread,
    /// A no-copy buffer was rebound or detached since the state's tables were read.
    Moved,
    /// The state's tables were read through another interface.
    Interface,
    /// Through the same interface, with other samplers' address-mode maps.
    Maps,
    /// The slots that tell the view base alignment apart are not as the state holds them.
    Views,
    /// The state was read from an ended encoder's tables, and its encoder started from it as a
    /// spare (`live_bind::SpareLiveBinds`).
    Spare,
    /// The state was read from an ended encoder's tables, and its encoder started from it as
    /// carried in (`render/carried.rs`).
    Carried,
    /// The table does not list the calls since the state read it: more calls than it keeps.
    Unlisted,
}

impl WholeRead {
    /// How many there are.
    pub(crate) const COUNT: usize = 8;
}

/// What first told a kept draw's bind state, rebound, that its shape may have changed, where the
/// rebind stopped for it (`ArgumentsStop::Reshaped`,
/// `WorkCounts::draws_rebind_reshaped_*`): a changed table's slots that tell the view base
/// alignment apart, else the first slot of the changed ones it brought in
/// (`live_bind::Reshape`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReshapedBy {
    /// A changed table's slots that tell the view base alignment apart are not as the state holds
    /// them (`argument_table::AtBindViews`).
    Views,
    /// A buffer slot bound where the state held none.
    BufferBound,
    /// A buffer slot the state held unbound.
    BufferUnbound,
    /// A buffer slot bound to a binding of another shape.
    BufferShape,
    /// A buffer slot whose offset alone moved, to one of another shape.
    BufferOffset,
    /// A texture slot bound where the state held none.
    TextureBound,
    /// A texture slot the state held unbound.
    TextureUnbound,
    /// A texture slot bound to a texture of another shape.
    TextureShape,
    /// A sampler slot bound where the state held none.
    SamplerBound,
    /// A sampler slot the state held unbound.
    SamplerUnbound,
    /// A sampler slot bound to a sampler of another shape.
    SamplerShape,
}

impl ReshapedBy {
    /// How many there are.
    pub(crate) const COUNT: usize = 11;

    /// A slot of `class` bound where the state held none.
    pub(crate) fn bound(class: SlotClass) -> Self {
        match class {
            SlotClass::Buffer => ReshapedBy::BufferBound,
            SlotClass::Texture => ReshapedBy::TextureBound,
            SlotClass::Sampler => ReshapedBy::SamplerBound,
        }
    }

    /// A slot of `class` the state held unbound.
    pub(crate) fn unbound(class: SlotClass) -> Self {
        match class {
            SlotClass::Buffer => ReshapedBy::BufferUnbound,
            SlotClass::Texture => ReshapedBy::TextureUnbound,
            SlotClass::Sampler => ReshapedBy::SamplerUnbound,
        }
    }

    /// A slot of `class` bound to a binding of another shape.
    pub(crate) fn shape(class: SlotClass) -> Self {
        match class {
            SlotClass::Buffer => ReshapedBy::BufferShape,
            SlotClass::Texture => ReshapedBy::TextureShape,
            SlotClass::Sampler => ReshapedBy::SamplerShape,
        }
    }

    /// The class of the slot the change is of; none for the slots that tell the view base
    /// alignment apart, which a stage table holds as a field.
    pub(crate) fn class(self) -> Option<SlotClass> {
        match self {
            ReshapedBy::Views => None,
            ReshapedBy::BufferBound
            | ReshapedBy::BufferUnbound
            | ReshapedBy::BufferShape
            | ReshapedBy::BufferOffset => Some(SlotClass::Buffer),
            ReshapedBy::TextureBound | ReshapedBy::TextureUnbound | ReshapedBy::TextureShape => {
                Some(SlotClass::Texture)
            }
            ReshapedBy::SamplerBound | ReshapedBy::SamplerUnbound | ReshapedBy::SamplerShape => {
                Some(SlotClass::Sampler)
            }
        }
    }
}

/// Where the rebind of a render encoder's first draw of the kept draw its command buffer's last
/// render encoder carried into it stopped (`WorkCounts::draws_carry_stopped_*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CarryStop {
    /// A slot its bind state holds bound that no call of the encoder bound (`render/carried.rs`).
    Stale,
    /// It reads a texture an attachment of the encoder's pass renders through a slot narrowed to
    /// levels apart from the attached ones, which the kept draw did not narrow.
    Attached,
    /// Any other step of a rebind.
    Other,
    /// It reads through snapshots, and would read through them otherwise, or not, at every
    /// step; or it reads through none, and would read the pass's colors through snapshots, in
    /// place or after the split at every step (`render/draw_again/rebound.rs`).
    Snapshot,
}

impl CarryStop {
    /// How many there are.
    pub(crate) const COUNT: usize = 4;
}

/// The kind of a call a render encoder logs (`RenderLowering::log` and kin, `render/replay.rs`)
/// that met its kept draw: which let it go (`WorkCounts::kept_draws_let_go_*`) or kept it
/// (`WorkCounts::kept_draws_kept_across_*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LoggedCall {
    /// `update_fence` or `wait_for_fence`.
    Fence,
    /// `use_resource`, `use_resources`, `use_heap` or `use_heaps`.
    Use,
    /// `memory_barrier_with_scope` or `memory_barrier_with_resources`.
    Barrier,
    /// A color's, the depth's or the stencil's store action or store action options.
    StoreAction,
    /// `set_visibility_result_mode`.
    Visibility,
    /// Any other call that logs itself.
    Other,
}

impl LoggedCall {
    /// How many there are.
    pub(crate) const COUNT: usize = 6;
}

/// What kept a direct draw that recorded from being kept to draw again
/// (`WorkCounts::draws_unkept`): each a group of the conditions a kept draw meets
/// (`render/draw_again.rs`), any number of them at once.
#[derive(Clone, Copy, Default)]
pub(crate) struct UnkeptDraw {
    /// A step of its own beside its Vulkan draw, or a later render pass instance of its pass.
    pub(crate) instance: bool,
    /// Of `instance`: a later instance it records into (a split, a stand-in arriving, a
    /// continuation), a step of its own.
    pub(crate) instance_split: (bool, bool),
    /// A barrier in its instance.
    pub(crate) barrier: bool,
    /// A shadowed texture among its uses, a snapshot or an extended-range shadow.
    pub(crate) shadowed: bool,
    /// A heap, texture or sampler table, a root table, a push hole, or argument samplers unlisted.
    pub(crate) tables: bool,
    /// Its index buffer read through a copy.
    pub(crate) index_copied: bool,
    /// A use that writes, or reads indirect arguments.
    pub(crate) writing: bool,
    /// Of `writing`: such a use among those of what its encoder made resident.
    pub(crate) writing_resident: bool,
    /// Of `tables`: a heap's set.
    pub(crate) tables_heap: bool,
    /// Its `VkPipeline` still being made, or drawn through its key's general form while its own
    /// is made (`pipeline/render/residual.rs`).
    pub(crate) making: bool,
}

/// The clauses that leave out the slots of a draw that no snapshot serves
/// (`SplitReason::UnreadableOwnPixel`; `StageModule::snapshot_refusals`), one bit each in the
/// order of `WorkCounts::reading_draws_split_unreadable_own_pixel_*`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Unreadable(pub(crate) u16);

impl Unreadable {
    /// Each clause's text, in the bits' order: what the log says of it.
    const TEXTS: [&'static str; 12] = [
        "its read decides a branch or a switch",
        "its read reaches a derivative",
        "its read reaches a sample, a gather or a level-of-detail query",
        "its read reaches a subgroup or quad operation",
        "its read is stored outside the invocation",
        "its read reaches an atomic",
        "its read reaches an image write or a texel pointer",
        "its read is an integer divisor, a signed dividend beside a divisor that may be -1, or a \
         dynamic vector index",
        "the function writes memory",
        "the function holds an interlock",
        "the function has a coherent resource or uses the Vulkan memory model",
        "its read reaches a ray query or a call it does not follow, its slot is past 127, or its \
         values do not settle",
    ];

    /// The set with `why` added; a refusal this runtime does not name is `Other`'s.
    pub(crate) fn with(self, why: metal2vulkan::pipeline::SnapshotRefusal) -> Unreadable {
        use metal2vulkan::pipeline::SnapshotRefusal as R;
        let bit = match why {
            R::Branch => 0,
            R::Derivative => 1,
            R::Sample => 2,
            R::GroupOperation => 3,
            R::Store => 4,
            R::Atomic => 5,
            R::ImageWrite => 6,
            R::UndefinedOperand => 7,
            R::WritesMemory => 8,
            R::Interlock => 9,
            R::Coherent => 10,
            _ => 11,
        };
        Unreadable(self.0 | (1 << bit))
    }

    /// What the log says of the set: each clause's text, in order, `; ` between.
    pub(crate) fn text(self) -> String {
        let said = (Self::TEXTS.iter().zip(0..)).filter(|&(_, i)| self.0 & (1 << i) != 0);
        said.map(|(text, _)| *text).collect::<Vec<_>>().join("; ")
    }
}

/// How a direct draw differed from the draw its encoder kept (`WorkCounts::draws_unlike_kept`):
/// each a part of what a draw again compares (`render/draw_again.rs`), any number of them at
/// once.
#[derive(Clone, Copy, Default)]
pub(crate) struct UnlikeKept {
    /// Another pipeline state, or none.
    pub(crate) pipeline: bool,
    /// Another call: one with indices after one without, or the other way.
    pub(crate) call: bool,
    /// Another primitive assembly.
    pub(crate) assembly: bool,
    /// Other vertex or index counts or starts, instance count, base vertex or base instance.
    pub(crate) counts: bool,
    /// Another index buffer or index type, or an index buffer read through a copy.
    pub(crate) index_buffer: bool,
    /// An index buffer read through a copy.
    pub(crate) index_copied: bool,
    /// The same index buffer at another offset.
    pub(crate) index_offset: bool,
    /// Patch or control-point index buffers (a tessellation draw).
    pub(crate) tessellation: bool,
    /// Another command buffer, the `VkCommandBuffer` its instance records into.
    pub(crate) command_buffer: bool,
    /// A no-copy buffer moved, or the residency, the host aliases, parts or held images, or the
    /// sampler IDs changed (`Under`).
    pub(crate) under: bool,
}

/// Where the rebind of a draw of other arguments than its kept draw's stopped
/// (`WorkCounts::draws_rebind_stopped_arguments_*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ArgumentsStop {
    /// Its route: it may make a primitive where the kept draw may not, or the other way.
    Route,
    /// The kept draw keeps nothing to rebind (`Kept::rebinds`).
    Unkept,
    /// A render state set since that the kept draw's derivation does not cover (`restated.rs`).
    Restated,
    /// A carried draw's slot that no call of its encoder bound (`render/carried.rs`).
    Stale,
    /// A changed table read otherwise than the kept draw read it.
    Unread,
    /// A changed table that no longer lists the slots calls changed since.
    Unlisted,
    /// More changed slots than a rebind takes, or a changed slot of another class.
    Slots,
    /// A changed texture or sampler beside a memoryless texture or other samplers' address modes.
    Images,
    /// Changed slots that reshape the bind state.
    Reshaped,
    /// The bind state's kept settle not the kept draw's.
    Unsettled,
    /// Its plan reads its slots some other way than a rebind works out.
    Tables,
    /// A fixed-function vertex buffer's fetch window it moves.
    FetchWindow,
    /// Its pulled attributes' reach.
    Reach,
    /// A check of every step a rebind does not take again.
    Checks,
    /// Slots narrowed to levels apart from an attached one other than the kept draw's.
    Narrowed,
    /// Set-0 writes, uses or push bytes a rebind does not work out.
    Worked,
    /// A changed vertex buffer slot not bound as the kept draw's plan reads it.
    VertexBuffers,
    /// Shadowed textures used where the kept draw used none, or none where it used some.
    Shadows,
    /// A barrier after a geometry write.
    Barrier,
    /// A detached buffer among its uses.
    Detached,
    /// Any other step of a rebind.
    Other,
}

impl ArgumentsStop {
    /// How many there are.
    pub(crate) const COUNT: usize = 21;
}

impl UnlikeKept {
    /// Whether the draw differed by its arguments alone: its call of the same assembly (with
    /// indices or without), its counts and starts, its index buffer, index type or index buffer's
    /// offset, each index buffer read in place, and nothing else a draw again compares
    /// (`WorkCounts::draws_unlike_kept_arguments_alone`).
    pub(crate) fn arguments_alone(&self) -> bool {
        (self.call || self.counts || self.index_buffer || self.index_offset)
            && !(self.pipeline
                || self.assembly
                || self.index_copied
                || self.tessellation
                || self.command_buffer
                || self.under)
    }
}

/// The command counts by who adds them ([`CommandCounters`]): open encoders' tallies, and
/// threads' shards for the counts made with no encoder's core at hand.
#[derive(Default)]
struct CommandTallies {
    by_encoder: Tallies<CommandCounters>,
    by_thread: Sharded<CommandCounters>,
}

impl CommandTallies {
    /// The count `count` picks: every encoder's and thread's additions so far, open encoders'
    /// included.
    fn sum(&self, count: impl Fn(&CommandCounters) -> &AtomicU64 + Copy) -> u64 {
        self.by_encoder.sum(count) + self.by_thread.sum(count)
    }
}

/// A command's counts, added into its encoder's tally or its thread's shard
/// ([`CommandCounters`]).
#[derive(Clone, Copy)]
pub(crate) struct Counting<'a>(Mine<'a, CommandCounters>);

impl Counting<'_> {
    /// A binding of a two-plane YCbCr texture's planes read in place was written
    /// (`binding.rs`).
    pub(crate) fn ycbcr_planes_bound(&self) {
        self.0.add(|counts| &counts.ycbcr_planes_bound, 1);
    }

    /// A binding of a shadowed texture's rows read and written in place was written
    /// (`binding.rs`, `texture/texel_rows.rs`).
    pub(crate) fn texel_rows_bound(&self) {
        self.0.add(|counts| &counts.texel_rows_bound, 1);
    }

    /// A texel buffer view of a part past the first of a texture buffer wider than one view holds
    /// was written (`binding.rs`, `metal2vulkan::bind::texel_buffer_parts`).
    pub(crate) fn texel_buffer_parts_bound(&self) {
        self.0.add(|counts| &counts.texel_buffer_parts_bound, 1);
    }

    /// A buffer slot's uniform view was written in its texel form, a uniform texel buffer view
    /// made for the bind (`binding.rs`, `uniform_texels.rs`).
    pub(crate) fn uniform_texels_bound(&self) {
        self.0.add(|counts| &counts.uniform_texels_bound, 1);
    }

    /// A bound buffer's uniform view was written as a uniform buffer, its slot's window
    /// `window` long, `None` where the translation bounded none (`binding.rs`).
    pub(crate) fn uniform_buffer_bound(&self, window: Option<u64>) {
        match window {
            Some(len) if len <= 16 << 10 => {
                self.0.add(|counts| &counts.uniform_buffers_bound_to_16k, 1)
            }
            _ => self
                .0
                .add(|counts| &counts.uniform_buffers_bound_past_16k, 1),
        }
    }

    /// A binding of a two-plane YCbCr texture's planes image was written (`binding.rs`).
    pub(crate) fn ycbcr_planes_image_bound(&self) {
        self.0.add(|counts| &counts.ycbcr_planes_images_bound, 1);
    }

    /// A draw or dispatch whose key's `VkPipeline` was not made took its general form's
    /// (`pipeline/render/residual.rs`).
    pub(crate) fn residual_form_drawn(&self) {
        self.0.add(|counts| &counts.residual_form_draws, 1);
    }

    /// A draw or dispatch that began its key's making took its general form's `VkPipeline`
    /// meanwhile (`pipeline/render/residual.rs`).
    pub(crate) fn residual_form_making(&self) {
        self.0.add(|counts| &counts.residual_form_makings, 1);
    }

    /// A render encoder recorded a tile dispatch (`render/tile_dispatch.rs`).
    pub(crate) fn tile_dispatch_recorded(&self) {
        self.0.add(|counts| &counts.tile_dispatches_recorded, 1);
    }

    /// A draw made the bind shape of its bind state (`LiveBind::shape`).
    pub(crate) fn draw_shape_made(&self) {
        self.0.add(|counts| &counts.draw_shapes_made, 1);
    }

    /// A draw found the bind shape of its bind state among those its drawing state settled
    /// lately (`LiveBind::shape`).
    pub(crate) fn draw_shape_found(&self) {
        self.0.add(|counts| &counts.draw_shapes_found, 1);
    }

    /// A draw looking for the bind shape of its bind state compared `compared` kept shapes with
    /// it whole, those of its key (`LiveBind::shape`, `GraphicsState::kept_shape`).
    pub(crate) fn draw_shapes_compared(&self, compared: u64) {
        if compared > 0 {
            self.0.add(|counts| &counts.draw_shape_compares, compared);
        }
    }

    /// A compute dispatch made the bind shape of its encoder's kept bind state
    /// (`LiveDispatch::shape`).
    pub(crate) fn dispatch_shape_made(&self) {
        self.0.add(|counts| &counts.dispatch_shapes_made, 1);
    }

    /// A compute dispatch found the bind shape of its encoder's kept bind state among those its
    /// pipeline state planned for lately (`LiveDispatch::shape`).
    pub(crate) fn dispatch_shape_found(&self) {
        self.0.add(|counts| &counts.dispatch_shapes_found, 1);
    }

    /// A compute dispatch listed what its encoder made resident as its kept bind state's used
    /// resources (`LiveDispatch::refresh`).
    pub(crate) fn dispatch_used_resources_listed(&self) {
        self.0
            .add(|counts| &counts.dispatch_used_resources_listed, 1);
    }

    /// An ended encoder's orders took in `count` of its commands (`tracker::EncoderUses`).
    pub(crate) fn commands_ordered_inside(&self, count: u64) {
        self.0.add(|counts| &counts.commands_ordered_inside, count);
    }

    /// A bind worked out a texture slot's view that no bind memo kept (`binding::BindMemo`).
    pub(crate) fn bind_view_worked_out(&self) {
        self.0.add(|counts| &counts.bind_views_worked_out, 1);
    }

    /// A bind worked out a sampler slot's sampler description that no bind memo kept
    /// (`binding::BindMemo`).
    pub(crate) fn bind_sampler_worked_out(&self) {
        self.0.add(|counts| &counts.bind_samplers_worked_out, 1);
    }

    /// A bind's buffer descriptor read where a buffer over host pages lies, which no bind memo
    /// kept (`buffer::KeptSpaces`).
    pub(crate) fn bind_space_read(&self) {
        self.0.add(|counts| &counts.bind_spaces_read, 1);
    }

    /// A draw read a stage's table into its encoder's kept bind state
    /// (`ArgumentTable::refresh_reading`).
    pub(crate) fn stage_table_read(&self) {
        self.0.add(|counts| &counts.stage_tables_read, 1);
    }

    /// One of those reads brought in only the slots calls changed since the table was read
    /// (`ArgumentTable::refresh_changed`).
    pub(crate) fn stage_table_read_in_part(&self) {
        self.0.add(|counts| &counts.stage_tables_read_in_part, 1);
    }

    /// One of those reads read the table whole, for `why`
    /// (`WorkCounts::stage_tables_read_whole_*`).
    pub(crate) fn stage_table_read_whole(&self, why: WholeRead) {
        self.0
            .add(|counts| &counts.stage_tables_read_whole_by[why as usize], 1);
    }

    /// A look at where a no-copy buffer's bytes lie, which takes its pages' lock
    /// (`buffer::HostCell`).
    pub(crate) fn no_copy_lie_read(&self) {
        self.0.add(|counts| &counts.no_copy_lies_read, 1);
    }

    /// A draw's uses were noted already: the last command its encoder noted had them
    /// (`noted_apart.rs`).
    pub(crate) fn command_noted_as_before(&self) {
        self.0.add(|counts| &counts.commands_noted_as_before, 1);
    }

    /// A command's `uses` the uses its encoder noted last covered, so noted already
    /// (`noted_apart.rs`).
    pub(crate) fn uses_noted_covered(&self, uses: u64) {
        self.0.add(|counts| &counts.uses_noted_covered, uses);
    }

    /// A draw settled its variant and plan from its bind shape (`GraphicsState::settle_shape`).
    pub(crate) fn draw_settled(&self) {
        self.0.add(|counts| &counts.draws_settled, 1);
    }

    /// A draw ran a variant that takes a slot bound to a two-plane extended-range texture
    /// (`VariantKey::extended_range`).
    pub(crate) fn two_plane_draw(&self) {
        self.0.add(|counts| &counts.two_plane_draws, 1);
    }

    /// A draw under the last provoking vertex ran a `VkPipeline` of the last vertex
    /// (`render/provoking.rs`).
    pub(crate) fn provoking_last_pipeline_draw(&self) {
        self.0
            .add(|counts| &counts.provoking_last_pipeline_draws, 1);
    }

    /// A draw under the last provoking vertex drew the list of its primitives in Metal's mode-1
    /// order or led by their last vertices (`render/provoking.rs`).
    pub(crate) fn provoking_listed_draw(&self) {
        self.0.add(|counts| &counts.provoking_listed_draws, 1);
    }

    /// A draw of lines at a width other than 1 ran the device's own lines at it
    /// (`render/line_width.rs`).
    pub(crate) fn device_wide_line_draw(&self) {
        self.0.add(|counts| &counts.device_wide_line_draws, 1);
    }

    /// A draw of lines, or of a triangle's edges, at a width other than 1 ran the pipeline's band
    /// variant (`render/line_width.rs`).
    pub(crate) fn line_band_draw(&self) {
        self.0.add(|counts| &counts.line_band_draws, 1);
    }

    /// A draw of lines at a width other than 1 ran the device's own lines at the nearest width
    /// they draw (`render/line_width.rs`).
    pub(crate) fn nearest_width_line_draw(&self) {
        self.0.add(|counts| &counts.nearest_width_line_draws, 1);
    }

    /// A draw pulled a fixed-function layout the device would fetch past 4 GiB wrapped
    /// (`render/vertex_reach.rs`).
    pub(crate) fn far_vertex_fetch_pulled_draw(&self) {
        self.0
            .add(|counts| &counts.far_vertex_fetch_pulled_draws, 1);
    }

    /// A direct indexed draw drawn as a list joined a run of the index gather pass, which writes
    /// the run's lists by one dispatch (`index_rewrite.rs`).
    pub(crate) fn gathered_draw(&self) {
        self.0.add(|counts| &counts.gathered_draws, 1);
    }

    /// An indexed indirect draw whose index fetch the index bounds pass bounds joined a run of
    /// it, which plans and fills the run's draws by one dispatch each (`index_bounds.rs`).
    pub(crate) fn bounded_run_draw(&self) {
        self.0.add(|counts| &counts.bounded_run_draws, 1);
    }

    /// A command's retain of an argument table changed since its last retain into the same
    /// encoder looked at nothing: each object its calls bound since, the encoder held as it was
    /// bound (`argument_table::retain_into`).
    pub(crate) fn argument_table_held_at_bind(&self) {
        self.0.add(|counts| &counts.argument_tables_held_at_bind, 1);
    }

    /// A stage table read in part brought in a buffer slot whose buffer's offset alone moved,
    /// looking at no buffer (`ArgumentTable::refresh_offset`).
    pub(crate) fn bound_offset_moved(&self) {
        self.0.add(|counts| &counts.bound_offsets_moved, 1);
    }

    /// A command retained what an argument table binds (`ArgumentTable::retain`), taking hold of
    /// `objects` of them, one at least.
    pub(crate) fn argument_table_retained(&self, objects: u64) {
        self.0.add(|counts| &counts.argument_tables_retained, 1);
        self.0
            .add(|counts| &counts.argument_objects_retained, objects);
    }

    /// A draw looked its `VkPipeline` up in its variant's memo by key
    /// (`RenderVariant::begin_graphics_pipeline`, `graphics_pipeline_now`, `graphics_pipeline`).
    pub(crate) fn graphics_pipeline_looked_up(&self) {
        self.0.add(|counts| &counts.graphics_pipelines_looked_up, 1);
    }

    /// Draws recorded `count` state commands into their secondary command buffers
    /// (`render::Recording::record`).
    pub(crate) fn draw_states_recorded(&self, count: u64) {
        if count > 0 {
            self.0.add(|counts| &counts.draw_states_recorded, count);
        }
    }

    /// A draw recorded with its set 0 pushed (`pipeline::objects::pushes_set_zero`).
    pub(crate) fn draw_pushing_set_zero(&self) {
        self.0.add(|counts| &counts.draws_pushing_set_zero, 1);
    }

    /// A dispatch encoded with its set 0 pushed (`pipeline::objects::pushes_set_zero`), its
    /// pushes carrying `writes` descriptor writes (`binding::PushedSetZero`), none where none
    /// changed.
    pub(crate) fn dispatch_pushing_set_zero(&self, writes: u64) {
        self.0.add(|counts| &counts.dispatches_pushing_set_zero, 1);
        self.0
            .add(|counts| &counts.dispatch_set_zero_writes_pushed, writes);
    }

    /// A compute dispatch whose `uses` (those it reaches through what its encoder made resident
    /// among them) its encoder's hazard tracking took.
    pub(crate) fn dispatch_tracked(&self, uses: u64) {
        self.0.add(|counts| &counts.dispatches_tracked, 1);
        self.0.add(|counts| &counts.dispatch_uses_tracked, uses);
    }

    /// A command that bound the set 0 its encoder's last command wrote, the same descriptors
    /// into a set of the same layout (`binding::LastSetZero`).
    pub(crate) fn set_zero_set_reused(&self) {
        self.0.add(|counts| &counts.set_zero_sets_reused, 1);
    }

    /// A command that bound a set 0 an earlier command of its command buffer wrote, the same
    /// descriptors into a set of the same layout (`binding::SetCache`).
    pub(crate) fn set_zero_set_found(&self) {
        self.0.add(|counts| &counts.set_zero_sets_found, 1);
    }

    /// A draw that took its settings, draw state and `VkPipeline` from its encoder's earlier
    /// draws of the same inputs (`render/draw_shape.rs`).
    pub(crate) fn draw_shape_reused(&self) {
        self.0.add(|counts| &counts.draw_shapes_reused, 1);
    }

    /// A draw that found no shape in its encoder's memo and took its settings from an earlier
    /// encoder's draw of the same drawing state and values (`render/draw_shape.rs`).
    pub(crate) fn draw_shape_carried(&self) {
        self.0.add(|counts| &counts.draw_shapes_carried, 1);
    }

    /// A draw a render encoder recorded as its last draw again (`render/draw_again.rs`), whose
    /// uses were noted already; `kept` says whether the draw it repeats pushed set 0 and ran a
    /// `VkPipeline` of the last provoking vertex.
    pub(crate) fn draw_drawn_again(&self, kept: (bool, bool)) {
        let counts = self.0;
        counts.add(|counts| &counts.draws_as_before, 1);
        counts.add(|counts| &counts.commands_noted_as_before, 1);
        Self::kept_draw(&counts, kept);
    }

    /// A draw that samples, at a slot, a texture that presents a color attachment of its render
    /// pass read the texture's copy, where its shadow is served (`texture/attached_reads.rs`): one
    /// that takes every step, or a draw again or rebound of one (`render/draw_again.rs`).
    pub(crate) fn attachment_read_through_copy(&self) {
        self.0
            .add(|counts| &counts.attachment_reads_through_copies, 1);
    }

    /// Of [`attachment_read_through_copy`](Self::attachment_read_through_copy), a draw that reads
    /// the texture at its fragments' own pixels alone, whose copy in moves the texels around its
    /// scissors alone (`texture/attached_reads.rs`).
    pub(crate) fn own_pixel_read(&self) {
        self.0.add(|counts| &counts.own_pixel_reads, 1);
    }

    /// A draw under a visibility result whose fragment function writes its sample mask under early
    /// tests, on a device that may count after the mask
    /// (`WorkCounts::draws_counted_past_early_masks`).
    pub(crate) fn draw_counted_past_early_mask(&self) {
        self.0
            .add(|counts| &counts.draws_counted_past_early_masks, 1);
    }

    /// A slot of a draw that takes every step samples a texture a color attachment of its render
    /// pass instance renders, its reads' levels meeting the attached one as `levels` says
    /// (`render/attached_levels.rs`).
    pub(crate) fn attached_read_slot(&self, levels: AttachedLevels) {
        self.0.add(|counts| &counts.attached_read_slots, 1);
        (self.0).add(|counts| &counts.attached_read_slots_by[levels as usize], 1);
    }

    /// `made` uses of what a render encoder made resident that a draw worked out
    /// (`WorkCounts::resident_uses_made`).
    pub(crate) fn resident_uses_made(&self, made: u64) {
        self.0.add(|counts| &counts.resident_uses_made, made);
    }

    /// `moved` keys a draw's working out of what it reaches through what its render encoder made
    /// resident alone moved (`WorkCounts::resident_keys_settled`).
    pub(crate) fn resident_keys_settled(&self, moved: u64) {
        if moved > 0 {
            self.0.add(|counts| &counts.resident_keys_settled, moved);
        }
    }

    /// `walked` uses of what a render encoder made resident that a draw's gathering for its
    /// instance's shadowed textures walked (`WorkCounts::resident_uses_gathered`).
    pub(crate) fn resident_uses_gathered(&self, walked: u64) {
        self.0.add(|counts| &counts.resident_uses_gathered, walked);
    }

    /// `walked` uses of what a render encoder made resident that a draw's checks against what its
    /// render pass's fragment stages wrote and read walked
    /// (`WorkCounts::resident_uses_against_fragments`).
    pub(crate) fn resident_uses_against_fragments(&self, walked: u64) {
        self.0
            .add(|counts| &counts.resident_uses_against_fragments, walked);
    }

    /// `walked` uses of what a render encoder made resident that a draw's check against its
    /// render pass's attachments walked (`WorkCounts::resident_uses_against_attachments`).
    pub(crate) fn resident_uses_against_attachments(&self, walked: u64) {
        self.0
            .add(|counts| &counts.resident_uses_against_attachments, walked);
    }

    /// `walked` uses of what a render encoder made resident that a draw's noting walked
    /// (`WorkCounts::resident_uses_noted`).
    pub(crate) fn resident_uses_noted(&self, walked: u64) {
        self.0.add(|counts| &counts.resident_uses_noted, walked);
    }

    /// `walked` uses of what a render encoder made resident that a draw's adding of its writes
    /// to those no barrier orders yet walked (`WorkCounts::resident_uses_unordered_writes`).
    pub(crate) fn resident_uses_unordered_writes(&self, walked: u64) {
        if walked > 0 {
            self.0
                .add(|counts| &counts.resident_uses_unordered_writes, walked);
        }
    }

    /// `walked` uses of what a render encoder made resident that a draw's adding of its indirect
    /// argument reads walked (`WorkCounts::resident_uses_indirect_reads`).
    pub(crate) fn resident_uses_indirect_reads(&self, walked: u64) {
        if walked > 0 {
            self.0
                .add(|counts| &counts.resident_uses_indirect_reads, walked);
        }
    }

    /// `walked` uses of what a render encoder made resident that a draw's adding of its writes
    /// before its fragments walked (`WorkCounts::resident_uses_geometry_writes`).
    pub(crate) fn resident_uses_geometry_writes(&self, walked: u64) {
        if walked > 0 {
            self.0
                .add(|counts| &counts.resident_uses_geometry_writes, walked);
        }
    }

    /// `walked` uses of what a render encoder made resident that a draw's check against the
    /// writes before earlier draws' fragments walked (`WorkCounts::resident_uses_geometry_checked`).
    pub(crate) fn resident_uses_geometry_checked(&self, walked: u64) {
        if walked > 0 {
            self.0
                .add(|counts| &counts.resident_uses_geometry_checked, walked);
        }
    }

    /// `walked` uses of what a render encoder made resident that a draw's check whether it may be
    /// kept for drawing again walked (`WorkCounts::resident_uses_again_checked`).
    pub(crate) fn resident_uses_again_checked(&self, walked: u64) {
        if walked > 0 {
            self.0
                .add(|counts| &counts.resident_uses_again_checked, walked);
        }
    }

    /// A render draw that worked out its heap part (`WorkCounts::draw_heap_parts_made`).
    pub(crate) fn draw_heap_part_made(&self) {
        self.0.add(|counts| &counts.draw_heap_parts_made, 1);
    }

    /// A render draw or tile dispatch looked at every texture its encoder made resident for those
    /// a carrier holds (`WorkCounts::draw_resident_textures_looked_at`).
    pub(crate) fn draw_resident_textures_looked_at(&self) {
        self.0
            .add(|counts| &counts.draw_resident_textures_looked_at, 1);
    }

    /// A render draw rebound (`WorkCounts::draws_rebound`), which recorded `states` state
    /// commands and pushed `writes` set-0 writes; `kept` as `draw_drawn_again`'s.
    pub(crate) fn draw_rebound(&self, kept: (bool, bool), (states, writes): (u64, u64)) {
        let counts = self.0;
        counts.add(|counts| &counts.draws_rebound, 1);
        if states > 0 {
            counts.add(|counts| &counts.draw_states_recorded, states);
        }
        if writes > 0 {
            counts.add(|counts| &counts.set_zero_writes_pushed, writes);
        }
        Self::kept_draw(&counts, kept);
    }

    /// A render encoder's direct draw that recorded and kept nothing to draw again, for `why`
    /// (`WorkCounts::draws_unkept`).
    pub(crate) fn draw_unkept(&self, why: UnkeptDraw) {
        let counts = self.0;
        counts.add(|counts| &counts.draws_unkept, 1);
        let (later, steps) = why.instance_split;
        let groups: [(bool, fn(&CommandCounters) -> &AtomicU64); 12] = [
            (why.instance, |c| &c.draws_unkept_instance),
            (later, |c| &c.draws_unkept_instance_later),
            (steps, |c| &c.draws_unkept_instance_steps),
            (why.barrier, |c| &c.draws_unkept_barrier),
            (why.shadowed, |c| &c.draws_unkept_shadowed),
            (why.tables, |c| &c.draws_unkept_tables),
            (why.tables_heap, |c| &c.draws_unkept_tables_heap),
            (why.tables && why.writing, |c| {
                &c.draws_unkept_tables_writing
            }),
            (why.index_copied, |c| &c.draws_unkept_index_copied),
            (why.writing, |c| &c.draws_unkept_writing),
            (why.writing_resident, |c| &c.draws_unkept_writing_resident),
            (why.making, |c| &c.draws_unkept_making),
        ];
        let mut any = false;
        for (met, count) in groups {
            if met {
                counts.add(count, 1);
                any = true;
            }
        }
        if !any {
            counts.add(|c| &c.draws_unkept_other, 1);
        }
    }

    /// A render encoder's direct draw after its encoder kept a draw, unlike it as `why` says
    /// (`WorkCounts::draws_unlike_kept`).
    pub(crate) fn draw_unlike_kept(&self, why: UnlikeKept) {
        let counts = self.0;
        counts.add(|counts| &counts.draws_unlike_kept, 1);
        let alone = why.arguments_alone();
        let parts: [(bool, fn(&CommandCounters) -> &AtomicU64); 9] = [
            (why.pipeline, |c| &c.draws_unlike_kept_pipeline),
            (why.call || why.assembly, |c| &c.draws_unlike_kept_call),
            (why.counts, |c| &c.draws_unlike_kept_counts),
            (why.index_buffer, |c| &c.draws_unlike_kept_index_buffer),
            (why.index_offset, |c| &c.draws_unlike_kept_index_offset),
            (why.tessellation, |c| &c.draws_unlike_kept_tessellation),
            (why.command_buffer, |c| &c.draws_unlike_kept_command_buffer),
            (why.under, |c| &c.draws_unlike_kept_under),
            (alone, |c| &c.draws_unlike_kept_arguments_alone),
        ];
        for (met, count) in parts {
            if met {
                counts.add(count, 1);
            }
        }
    }

    /// A render encoder's direct draw of its kept draw again after calls, that took every step
    /// (`WorkCounts::draws_rebind_stopped`).
    pub(crate) fn draw_rebind_stopped(&self) {
        self.0.add(|counts| &counts.draws_rebind_stopped, 1);
    }

    /// A render draw rebound after calls that set its encoder's render state
    /// (`WorkCounts::draws_restated`).
    pub(crate) fn draw_restated(&self) {
        self.0.add(|counts| &counts.draws_restated, 1);
    }

    /// A render encoder's kept draw restated by its scissor rectangles alone
    /// (`WorkCounts::draws_restated_scissors`).
    pub(crate) fn draw_restated_scissors(&self) {
        self.0.add(|counts| &counts.draws_restated_scissors, 1);
    }

    /// `count` kept draws set aside kept into the render pass instance a draw began after a
    /// split (`WorkCounts::kept_draws_rehomed`), `carried` of them carried in and not drawn since
    /// (`WorkCounts::kept_draws_rehomed_carried`).
    pub(crate) fn kept_draws_rehomed(&self, count: u64, carried: u64) {
        if count > 0 {
            self.0.add(|counts| &counts.kept_draws_rehomed, count);
        }
        if carried > 0 {
            self.0
                .add(|counts| &counts.kept_draws_rehomed_carried, carried);
        }
    }

    /// A kept draw carried in and not drawn since let go at its state's next draw, in another
    /// instance than the one its encoder opened with (`WorkCounts::kept_draws_carried_unopened`).
    pub(crate) fn kept_draw_carried_unopened(&self) {
        self.0.add(|counts| &counts.kept_draws_carried_unopened, 1);
    }

    /// A render draw rebound with other arguments than its kept draw's
    /// (`WorkCounts::draws_rebound_arguments`).
    pub(crate) fn draw_rebound_arguments(&self) {
        self.0.add(|counts| &counts.draws_rebound_arguments, 1);
    }

    /// A render draw of other arguments than its kept draw's whose rebind stopped where `why`
    /// says (`WorkCounts::draws_rebind_stopped_arguments`).
    pub(crate) fn draw_rebind_stopped_arguments(&self, why: ArgumentsStop) {
        let counts = self.0;
        counts.add(|counts| &counts.draws_rebind_stopped_arguments, 1);
        counts.add(
            |counts| &counts.draws_rebind_stopped_arguments_by[why as usize],
            1,
        );
    }

    /// A render draw's rebind that stops reshaped stops for `why`
    /// (`WorkCounts::draws_rebind_reshaped_*`).
    pub(crate) fn draw_rebind_reshaped(&self, why: ReshapedBy) {
        self.0
            .add(|counts| &counts.draws_rebind_reshaped_by[why as usize], 1);
    }

    /// A render draw rebound from the kept draw its command buffer's last render encoder carried
    /// into its encoder (`WorkCounts::draws_rebound_carried`).
    pub(crate) fn draw_rebound_carried(&self) {
        self.0.add(|counts| &counts.draws_rebound_carried, 1);
    }

    /// A render draw that read through snapshots as a kept draw, drawn again or rebound
    /// (`WorkCounts::reading_draws_snapshotted_again`).
    pub(crate) fn reading_draw_again(&self) {
        self.0
            .add(|counts| &counts.reading_draws_snapshotted_again, 1);
    }

    /// A render draw of a kept draw that reads through snapshots that took every step, as it
    /// would read through them otherwise, or not (`WorkCounts::reading_draws_snapshotted_stopped`).
    pub(crate) fn draw_reader_stopped(&self) {
        self.0
            .add(|counts| &counts.reading_draws_snapshotted_stopped, 1);
    }

    /// A render encoder's direct draw whose encoder kept no draw of its pipeline state
    /// (`WorkCounts::draws_none_kept`), of an encoder that opened from no carried kept draw where
    /// `uncarried` (`WorkCounts::draws_none_kept_uncarried`).
    pub(crate) fn draw_none_kept(&self, uncarried: bool) {
        self.0.add(|counts| &counts.draws_none_kept, 1);
        if uncarried {
            self.0.add(|counts| &counts.draws_none_kept_uncarried, 1);
        }
    }

    /// A render encoder's kept draw let go by a call of kind `by` it logged
    /// (`WorkCounts::kept_draws_let_go_*`).
    pub(crate) fn kept_draw_let_go(&self, by: LoggedCall) {
        self.0
            .add(|counts| &counts.kept_draws_let_go_by[by as usize], 1);
    }

    /// A render encoder's kept draw kept across a call of kind `by` it logged
    /// (`WorkCounts::kept_draws_kept_across_*`).
    pub(crate) fn kept_draw_kept_across(&self, by: LoggedCall) {
        self.0
            .add(|counts| &counts.kept_draws_kept_across_by[by as usize], 1);
    }

    /// A render draw of the kept draw carried into its encoder whose rebind stopped where `why`
    /// says (`WorkCounts::draws_carry_stopped`).
    pub(crate) fn draw_carry_stopped(&self, why: CarryStop) {
        let counts = self.0;
        counts.add(|counts| &counts.draws_carry_stopped, 1);
        let count: fn(&CommandCounters) -> &AtomicU64 = match why {
            CarryStop::Stale => |c| &c.draws_carry_stopped_by[CarryStop::Stale as usize],
            CarryStop::Attached => |c| &c.draws_carry_stopped_by[CarryStop::Attached as usize],
            CarryStop::Other => |c| &c.draws_carry_stopped_by[CarryStop::Other as usize],
            CarryStop::Snapshot => |c| &c.draws_carry_stopped_by[CarryStop::Snapshot as usize],
        };
        counts.add(count, 1);
    }

    /// A render draw rebound that was moved (`WorkCounts::draws_moved`).
    pub(crate) fn draw_moved(&self) {
        self.0.add(|counts| &counts.draws_moved, 1);
    }

    /// A render draw moved whose moved uses lay inside the last draw's
    /// (`WorkCounts::draws_moved_inside`).
    pub(crate) fn draw_moved_inside(&self) {
        self.0.add(|counts| &counts.draws_moved_inside, 1);
    }

    /// `tables` stage tables read into a kept draw's bind state, each bringing in only the slots
    /// calls changed (`render/draw_again/rebound.rs`), whether the draw is then rebound or takes
    /// every step, which finds them read.
    pub(crate) fn tables_read_in_part(&self, tables: u64) {
        if tables > 0 {
            let counts = self.0;
            counts.add(|counts| &counts.stage_tables_read, tables);
            counts.add(|counts| &counts.stage_tables_read_in_part, tables);
        }
    }

    /// What a draw again or rebound counts as the draw it repeats did: a push of set 0 and a
    /// `VkPipeline` of the last provoking vertex.
    fn kept_draw(counts: &Mine<'_, CommandCounters>, (pushes_set_zero, last_vertex): (bool, bool)) {
        if pushes_set_zero {
            counts.add(|counts| &counts.draws_pushing_set_zero, 1);
        }
        if last_vertex {
            counts.add(|counts| &counts.provoking_last_pipeline_draws, 1);
        }
    }

    /// An encoder's hold of a key its commands named first that looked the key up among the
    /// device's buffers over host pages (`rebind::HostedKeys`).
    pub(crate) fn host_hold_looked_up(&self) {
        self.0.add(|counts| &counts.host_holds_looked_up, 1);
    }

    /// A compute encoder's dispatch that took what its last dispatch settled
    /// (`compute::DispatchMemo`).
    pub(crate) fn dispatch_settled_as_before(&self) {
        self.0.add(|counts| &counts.dispatches_settled_as_before, 1);
    }

    /// A compute encoder's dispatch that took what an earlier compute encoder of its command
    /// buffer settled, its table bringing in the very state that one bound
    /// (`compute::LiveDispatch::settled_across`).
    pub(crate) fn dispatch_settled_across_encoders(&self) {
        self.0
            .add(|counts| &counts.dispatches_settled_across_encoders, 1);
    }

    /// A dispatch whose commands `compute::record` instantiated.
    pub(crate) fn dispatch_instantiated(&self) {
        self.0.add(|counts| &counts.dispatches_instantiated, 1);
    }

    /// A compute encoder's dispatch whose commands `compute::record` instantiated over the
    /// storage of those its kept bind state kept (`compute::Instantiated`).
    pub(crate) fn dispatch_instantiated_in_place(&self) {
        self.0
            .add(|counts| &counts.dispatches_instantiated_in_place, 1);
    }

    /// A compute encoder's dispatch that took the commands its kept bind state kept of what it
    /// planned (`compute::Instantiated`).
    pub(crate) fn dispatch_instantiated_as_before(&self) {
        self.0
            .add(|counts| &counts.dispatches_instantiated_as_before, 1);
    }

    /// A compute encoder's dispatch started from the bind state its pipeline state's last
    /// dispatch left, set aside meanwhile (`LiveDispatch::select`).
    pub(crate) fn dispatch_from_set_aside(&self) {
        self.0.add(|counts| &counts.dispatches_from_set_aside, 1);
    }

    /// A dispatch counted in `dispatch_from_set_aside` took commands kept of its state rather than
    /// instantiating them (`compute::record`).
    pub(crate) fn set_aside_commands_kept(&self) {
        self.0.add(|counts| &counts.set_aside_commands_kept, 1);
    }

    /// A dispatch counted in `dispatch_from_set_aside` instantiated its commands again
    /// (`compute::record`).
    pub(crate) fn set_aside_commands_instantiated(&self) {
        self.0
            .add(|counts| &counts.set_aside_commands_instantiated, 1);
    }

    /// A dispatch of an inert kernel that recorded no Vulkan dispatch (`compute::record`).
    pub(crate) fn dispatch_skipped_inert(&self) {
        self.0.add(|counts| &counts.dispatches_skipped_inert, 1);
    }

    /// A direct dispatch that settled a bind state of the shape its state's last direct dispatch
    /// planned and took what that one planned (`pipeline::Planned`).
    pub(crate) fn dispatch_planned_as_before(&self) {
        self.0.add(|counts| &counts.dispatches_planned_as_before, 1);
    }

    /// A dispatch counted in `dispatch_planned_as_before` of another grid than the one what it
    /// took was planned for, of its class (`pipeline::GridClass`).
    pub(crate) fn dispatch_planned_across_grids(&self) {
        self.0
            .add(|counts| &counts.dispatches_planned_across_grids, 1);
    }

    /// A direct dispatch of whole threadgroups that ran on a large kernel's partial-threadgroups
    /// variant, every thread live (`compute::partial_direct`).
    pub(crate) fn whole_dispatch_on_partial_variant(&self) {
        self.0
            .add(|counts| &counts.whole_dispatches_on_partial_variants, 1);
    }

    /// A dispatch of a variant that bakes samplers, its pipelines made, recorded in a
    /// `VkCommandBuffer` of its own (`RuntimeDriverDefect::ImmutableSamplersLostAfterCopies`).
    pub(crate) fn baked_sampler_dispatch_apart(&self) {
        self.0
            .add(|counts| &counts.baked_sampler_dispatches_apart, 1);
    }

    /// Draws' pushes of their set 0 carried `count` writes (`render/bound.rs`).
    pub(crate) fn set_zero_writes_pushed(&self, count: u64) {
        if count > 0 {
            self.0.add(|counts| &counts.set_zero_writes_pushed, count);
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
            self.0
                .add(|counts| &counts.set_zero_equal_shape_switches, switches);
            self.0.add(
                |counts| &counts.set_zero_writes_on_equal_shape_switches,
                carried,
            );
            self.0
                .add(|counts| &counts.set_zero_writes_if_shapes_shared, if_shared);
        }
    }
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
    /// A draw that reaches storage the pixel interlock orders that an earlier draw of the instance
    /// wrote, on a driver whose such draws race under load
    /// (`DriverFact::InterlockedDrawsRaceUnderLoad`, `render.rs`'s `InterlockedStorage`).
    InterlockedWrites,
    /// A draw that samples a color attachment's texture after a barrier inside the instance, or a
    /// color the pass clears after a barrier before its first draw (`texture/attached_reads.rs`).
    AttachedRead,
    /// A draw that reads through snapshots apart from the instance (those that lead its own among
    /// them), or binds a color the instance holds in the feedback loop layout
    /// (`render/snapshot.rs`).
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
    pub(crate) const COUNT: usize = 11;
}

/// What kept a render pass's first instance from recording inline (`render/inline_first.rs`): the
/// first check it failed, in this order. Counted (`WorkCounts::passes_not_inline_encoders` and
/// kin), so a run names what sends its passes' draws to secondary command buffers.
#[derive(Clone, Copy, Debug)]
pub(crate) enum NotInline {
    /// The encoder encodes an ended encoder's calls again, or the pass has several color groups.
    Encoders,
    /// The pass's layout: no attachment, or more than one layer or sample, a visibility buffer,
    /// custom sample positions, a rasterization-rate map, extended-range shadows that keep its
    /// draws, or a depth/stencil attachment held out of `GENERAL`.
    Layouts,
    /// A memoryless color attachment not loaded and stored `DontCare`, or resolved; or a
    /// memoryless depth or stencil.
    Memoryless,
    /// A color attachment's or depth/stencil aspect's actions, the first ruled out
    /// ([`ActionsRefused`]).
    Actions(ActionsRefused),
    /// A color attachment that is a texture over a buffer the pass does not render through an
    /// image over the buffer's bytes.
    BufferTextures,
    /// Another color attachment kind: a stand-in, planes, a clear made apart, or an
    /// extended-range format; or another depth/stencil attachment kind: separate depth and
    /// stencil textures' scratch, a stand-in, a texture over a buffer, or a NaN depth clear
    /// written apart.
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

/// The color attachment or depth/stencil aspect action that kept a render pass's first instance
/// from recording inline ([`NotInline::Actions`]): the first ruled out, color by color in index
/// order, then the depth and the stencil, and of one in this order. Counted
/// (`WorkCounts::passes_not_inline_actions_dont_care_load` and kin).
#[derive(Clone, Copy, Debug)]
pub(crate) enum ActionsRefused {
    /// Loaded `DontCare`.
    DontCareLoad,
    /// Reaching past the render area as another attachment does not: cleared where Metal's clear
    /// of whole tiles (`clear_reach.rs`) reaches another extent than another attachment's, or
    /// loaded, or a memoryless color smaller, where another's clear reaches past the render area.
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
    /// The counts the commands of the encoder that holds `tally` add (`EncoderCore::counts`).
    #[inline]
    pub(crate) fn of(&self, tally: &Tally) -> Counting<'_> {
        Counting(self.commands.by_encoder.of(tally))
    }

    /// The counts a command adds where no encoder's core is at hand: the calling thread's.
    #[inline]
    pub(crate) fn thread(&self) -> Counting<'_> {
        Counting(self.commands.by_thread.mine())
    }

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

    /// A fill was recorded as `vkCmdFillBuffer` (`word_fill.rs`).
    pub(crate) fn fill_recorded(&self) {
        self.fills_recorded.fetch_add(1, Ordering::Relaxed);
    }

    /// `merged` buffer copies of a blit encoder were recorded as regions of the
    /// `vkCmdCopyBuffer` of the copy before them (`copy_run.rs`).
    pub(crate) fn buffer_copies_merged(&self, merged: u64) {
        self.buffer_copies_merged
            .fetch_add(merged, Ordering::Relaxed);
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

    /// A shadowed texture was made with its rows read and written in place
    /// (`texture/texel_rows.rs`).
    pub(crate) fn texel_rows_texture(&self) {
        self.texel_rows_textures.fetch_add(1, Ordering::Relaxed);
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

    /// A two-plane YCbCr texture's planes read in place were made an image the device's
    /// conversion samples (`texture/ycbcr_planes.rs`).
    pub(crate) fn ycbcr_planes_image(&self) {
        self.ycbcr_planes_images.fetch_add(1, Ordering::Relaxed);
    }

    /// A buffer of the runtime's own lay in a plain block while the device's check of its
    /// exportable blocks was pending (`in_place.rs`'s `Device::fd_blocks_checked`).
    pub(crate) fn fd_blocks_awaited(&self) {
        self.fd_blocks_awaited.fetch_add(1, Ordering::Relaxed);
    }

    /// A buffer of the runtime's own lay in a plain block, the device's check of its exportable
    /// blocks having refused them (`in_place.rs`'s `blocks_take`), or its
    /// `DRM_FORMAT_MOD_LINEAR` tiling, which every image there is (`dma_buf.rs`'s
    /// `Device::modifier_takes_linear`).
    pub(crate) fn fd_blocks_refused(&self) {
        self.fd_blocks_refused.fetch_add(1, Ordering::Relaxed);
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

    /// A YCbCr texture was made whose conversion samples its planes' image through the device's
    /// own Y′CbCr conversion (`ycbcr_device.rs`).
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

    /// `count` `VkCommandBuffer`s were allocated (`DeviceShared::allocate_command_buffers`).
    pub(crate) fn command_buffers_made(&self, count: u32) {
        (self.command_buffers_made).fetch_add(u64::from(count), Ordering::Relaxed);
    }

    /// A command buffer took one of the device's spare command pools (`spares.rs`).
    pub(crate) fn spare_command_pool_taken(&self) {
        self.spare_command_pools_taken
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Of those, a command buffer took one past its first, its pools having handed out every
    /// command buffer of a level they held (`spares.rs`).
    pub(crate) fn spare_command_pool_taken_past_first(&self) {
        self.spare_command_pools_taken_past_first
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A command buffer took a spare command pool that one of its command buffers may still be
    /// recording in, and reset it first (`spares.rs`).
    pub(crate) fn spare_command_pool_reset(&self) {
        self.spare_command_pools_reset
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A pool given back freed `recent` command buffers past its recent uses, `held` past
    /// `HELD` and `recorded` as its command buffer recorded `RECORDED` commands or more, spares
    /// shed `shed`, and their pools are trimmed `trims` times (`spares.rs`).
    pub(crate) fn spare_command_buffers_freed(
        &self,
        [recent, held, shed, recorded]: [usize; 4],
        trims: usize,
    ) {
        for (count, freed) in [recent, held, shed, recorded]
            .into_iter()
            .zip(&self.spare_command_buffers_freed)
        {
            freed.fetch_add(count as u64, Ordering::Relaxed);
        }
        (self.spare_command_pool_trims).fetch_add(trims as u64, Ordering::Relaxed);
    }

    /// A trim round trimmed `pools` spare command pools, destroying `buffers` command buffers
    /// freed from them untrimmed (`spares.rs`).
    pub(crate) fn spare_command_pools_trimmed_in_round(&self, pools: usize, buffers: usize) {
        (self.spare_command_pool_trim_rounds).fetch_add(1, Ordering::Relaxed);
        (self.spare_command_pool_trims).fetch_add(pools as u64, Ordering::Relaxed);
        (self.spare_command_buffers_trimmed_in_rounds).fetch_add(buffers as u64, Ordering::Relaxed);
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

    /// A probe went to another engine after no queue's work advanced for `stalled`, with some
    /// pending (`command_buffer/stall.rs`).
    pub(crate) fn stall_probe_sent(&self, stalled: std::time::Duration) {
        self.stall_probes_sent.fetch_add(1, Ordering::Relaxed);
        self.stall_lasted(stalled);
    }

    /// No queue's work advanced for `stalled`, with some pending (`command_buffer/stall.rs`).
    pub(crate) fn stall_lasted(&self, stalled: std::time::Duration) {
        let nanos = u64::try_from(stalled.as_nanos()).unwrap_or(u64::MAX);
        (self.longest_stall_nanos).fetch_max(nanos, Ordering::Relaxed);
    }

    /// A generation stopped and was parked (`command_buffer/stall.rs`).
    pub(crate) fn wedge_declared(&self) {
        self.wedges_declared.fetch_add(1, Ordering::Relaxed);
    }

    /// A queue's front batch went past the watchdog's bound, its command buffer failed as Metal's
    /// timed-out one, and the generation was parked (`command_buffer/stall.rs`).
    pub(crate) fn watchdog_timed_out(&self) {
        self.watchdog_timeouts.fetch_add(1, Ordering::Relaxed);
    }

    /// A command buffer's queue passed it while its timestamps read unavailable: the driver
    /// dropped its work, and it failed as Metal's victim (`command_buffer/mod.rs`).
    pub(crate) fn command_buffer_unrun(&self) {
        self.unrun_command_buffers.fetch_add(1, Ordering::Relaxed);
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

    /// A completion take left a batch seen complete for a batch of another `VkQueue` it waited
    /// for, not taken yet (`command_buffer/completion.rs`, "After what it waited for").
    pub(crate) fn batch_held_for_another_queue(&self) {
        (self.batches_held_for_another_queue).fetch_add(1, Ordering::Relaxed);
    }

    /// A command buffer's transition to `Completed` was posted behind a gate the transition of
    /// another queue's command buffer it waited for opens as it reaches its completed handlers
    /// (`CommandBufferShared::finish_here`).
    pub(crate) fn command_buffer_completed_after_another_queue(&self) {
        (self.command_buffers_completed_after_another_queue).fetch_add(1, Ordering::Relaxed);
    }

    /// A command buffer's completion-seen handler ran, on the thread that saw it complete where
    /// `where_seen`, else on its queue's handler thread (`CommandBufferShared::complete`).
    pub(crate) fn completion_seen_handler_ran(&self, where_seen: bool) {
        self.completion_seen_handlers_run
            .fetch_add(1, Ordering::Relaxed);
        if where_seen {
            self.completion_seen_handlers_where_seen
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A command buffer that needs no GPU work was taken with no submission
    /// (`command_buffer/submit.rs`).
    pub(crate) fn command_buffer_unsubmitted(&self) {
        self.command_buffers_unsubmitted
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A command buffer's completion sync file was exported from its submitted signal
    /// (`command_buffer/sync_file.rs`).
    #[cfg(unix)]
    pub(crate) fn sync_file_exported(&self) {
        self.sync_files_exported.fetch_add(1, Ordering::Relaxed);
    }

    /// A command buffer's completion sync file was answered signaled once its host copies landed
    /// (`command_buffer/sync_file.rs`, `sync-file-after-host-copies`).
    #[cfg(unix)]
    pub(crate) fn sync_file_after_host_copies(&self) {
        (self.sync_files_after_host_copies).fetch_add(1, Ordering::Relaxed);
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

    /// The first read of a command buffer's GPU times found them placed by a calibration pair
    /// that stated a deviation above 20 us (`gpu_time.rs`, `Placement::read`).
    pub(crate) fn loosely_calibrated(&self) {
        self.loosely_calibrated_gpu_times
            .fetch_add(1, Ordering::Relaxed);
    }

    /// The command buffers counted by [`loosely_calibrated`](Self::loosely_calibrated) so far, as
    /// [`Device::work_counts`] reads them.
    #[cfg(test)]
    pub(crate) fn loosely_calibrated_gpu_times(&self) -> u64 {
        self.loosely_calibrated_gpu_times.load(Ordering::Relaxed)
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
    /// (`render/pending.rs`): one a host step began among them (`host_step_pipelines_made`).
    pub(crate) fn pipeline_pooled(&self) {
        self.pipelines_pooled.fetch_add(1, Ordering::Relaxed);
        if crate::exit_guard::in_host_step() {
            self.host_step_pipelines_made
                .fetch_add(1, Ordering::Relaxed);
        }
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

    /// A kernel the runtime translated took the emulated simdgroup tier in place of the pin of its
    /// device's widest mode (`pipeline/sources.rs`).
    pub(crate) fn widest_pin_kernel_emulated(&self) {
        self.widest_pin_kernels_emulated
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A dispatch began its kernel's unit-axes variant on the device's pool
    /// (`pipeline/unit_axes.rs`).
    pub(crate) fn unit_axes_variant_begun(&self) {
        self.unit_axes_variants_begun
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A dispatch settled on its kernel's unit-axes variant (`pipeline/unit_axes.rs`).
    pub(crate) fn unit_axes_dispatch(&self) {
        self.unit_axes_dispatches.fetch_add(1, Ordering::Relaxed);
    }

    /// A dispatch settled on its kernel's plain variant while its unit-axes variant was being
    /// made (`pipeline/unit_axes.rs`).
    pub(crate) fn unit_axes_meanwhile(&self) {
        self.unit_axes_meanwhile.fetch_add(1, Ordering::Relaxed);
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
        if crate::exit_guard::in_host_step() {
            self.host_step_pipelines_made
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A variant's making waited for its twin's in the air before its own (`pipeline/flight.rs`).
    pub(crate) fn flight_followed(&self) {
        self.flights_followed.fetch_add(1, Ordering::Relaxed);
    }

    /// A making waited `waited` for the words of the process's limit on large makings in flight
    /// (`pipeline/compile_memory.rs`).
    pub(crate) fn making_held_for_compile_memory(&self, waited: std::time::Duration) {
        self.makings_held_for_compile_memory
            .fetch_add(1, Ordering::Relaxed);
        self.compile_memory_held_nanos.fetch_add(
            u64::try_from(waited.as_nanos()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
    }

    /// A large making's return handed glibc's free heap back (`pipeline/compile_memory.rs`).
    pub(crate) fn heap_trimmed(&self) {
        self.heap_trims.fetch_add(1, Ordering::Relaxed);
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

    /// `evicted` region `VkPipeline`s went past a bound: a compute variant's memo's
    /// (`pipeline/mod.rs`, `REGION_PIPELINES`) or the device's across every compute variant
    /// (`pipeline/region_ledger.rs`).
    pub(crate) fn region_pipelines_evicted(&self, evicted: u64) {
        self.region_pipelines_evicted
            .fetch_add(evicted, Ordering::Relaxed);
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

    /// A render variant took the device's linked pipeline of its link, which another variant
    /// linked (`pipeline/libraries.rs`).
    pub(crate) fn linked_pipeline_shared(&self) {
        self.linked_pipelines_shared.fetch_add(1, Ordering::Relaxed);
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

    /// `bytes` of an indirect command buffer's records were copied out of its storage
    /// (`indirect_command_buffer.rs`).
    pub(crate) fn indirect_record_bytes_copied(&self, bytes: u64) {
        (self.indirect_record_bytes_copied).fetch_add(bytes, Ordering::Relaxed);
    }

    /// A blit encoded a copy of `bytes` of an indirect command buffer's records into another's,
    /// which the GPU makes (`blit_indirect.rs`).
    pub(crate) fn indirect_record_bytes_gpu_copied(&self, bytes: u64) {
        (self.indirect_record_bytes_gpu_copied).fetch_add(bytes, Ordering::Relaxed);
    }

    /// A blit encoded a reset of `bytes` of an indirect command buffer's records, which the GPU
    /// writes (`blit_indirect.rs`).
    pub(crate) fn indirect_record_bytes_gpu_reset(&self, bytes: u64) {
        (self.indirect_record_bytes_gpu_reset).fetch_add(bytes, Ordering::Relaxed);
    }

    /// A blit's dispatch that reset an indirect command buffer's commands ran for `nanos` of the
    /// GPU's time (`blit_indirect.rs`, `rate_mapped_pass::PieceTimes`).
    pub(crate) fn indirect_record_reset_time(&self, nanos: u64) {
        (self.indirect_record_reset_gpu_nanos).fetch_add(nanos, Ordering::Relaxed);
    }

    /// An indirect command buffer's records came to lie in host memory for good at a host access,
    /// `bytes` of them copied there from device-local memory (`indirect_command_buffer/home.rs`).
    pub(crate) fn icb_records_moved_to_host(&self, bytes: u64) {
        (self.icb_records_moved_to_host).fetch_add(1, Ordering::Relaxed);
        (self.icb_record_bytes_moved).fetch_add(bytes, Ordering::Relaxed);
    }

    /// A host access of an indirect command buffer's records in device-local memory went through
    /// the staging queue (`indirect_command_buffer/home.rs`).
    pub(crate) fn icb_records_staged_access(&self) {
        (self.icb_records_staged_accesses).fetch_add(1, Ordering::Relaxed);
    }

    /// `count` records of an indirect command buffer's commands were read from its storage and
    /// decoded (`indirect_command_buffer.rs`).
    pub(crate) fn indirect_records_read(&self, count: u64) {
        self.indirect_records_read
            .fetch_add(count, Ordering::Relaxed);
    }

    /// The records of a run of an indirect command buffer's consecutive commands were read at
    /// once (`IndirectCommandBuffer::read_records`).
    pub(crate) fn indirect_record_run_read(&self) {
        self.indirect_record_runs_read
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A commit left `count` indirect command buffers its command buffer made resident for
    /// writing unmarked (`write_routes.rs`).
    pub(crate) fn indirect_write_residencies_unmarked(&self, count: u64) {
        self.indirect_write_residencies_unmarked
            .fetch_add(count, Ordering::Relaxed);
    }

    /// An execution ran a chunk of `commands` of an indirect command buffer's from its kept
    /// recording (`kept_commands.rs`), reading none of their records.
    pub(crate) fn indirect_chunk_kept(&self, commands: u64) {
        self.indirect_chunks_kept.fetch_add(1, Ordering::Relaxed);
        self.indirect_commands_kept
            .fetch_add(commands, Ordering::Relaxed);
    }

    /// A chunk of an indirect command buffer's commands took a context of its own to record a
    /// recording to keep in (`kept_commands::KeptContext`).
    pub(crate) fn kept_context_made(&self) {
        self.kept_contexts_made.fetch_add(1, Ordering::Relaxed);
    }

    /// An execution that may keep its commands compared the resources it made resident with
    /// those a kept chunk was recorded under, whole (`kept_commands::ResidentMet`).
    pub(crate) fn kept_resident_key_compared(&self) {
        self.kept_resident_keys_compared
            .fetch_add(1, Ordering::Relaxed);
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

    /// An execution of render commands in its GPU form placed its scratch of `bytes` in
    /// device-local memory (`render/executions/gpu.rs`, **Scratch**).
    pub(crate) fn gpu_form_device_local_bytes(&self, bytes: u64) {
        (self.gpu_form_device_local_bytes).fetch_add(bytes, Ordering::Relaxed);
    }

    /// A command buffer's lease made its queue's block of a GPU pool: the queue held none, and the
    /// device had no spare (`gpu_pools.rs`).
    pub(crate) fn gpu_pool_block_made(&self) {
        self.gpu_pool_blocks_made.fetch_add(1, Ordering::Relaxed);
    }

    /// A lease of a queue's block of a GPU pool began, `leased` leases holding it now, this one's
    /// included (`gpu_pools.rs`).
    pub(crate) fn gpu_pool_leased(&self, leased: u64) {
        self.gpu_pool_leases_peak
            .fetch_max(leased, Ordering::Relaxed);
    }

    /// A block of a GPU pool was released before the device's teardown: a queue's that no lease
    /// held for too long, or whose queue went, or one made while another lease of its queue made
    /// one, where the device held a spare of its kind already (`gpu_pools.rs`).
    pub(crate) fn gpu_pool_block_released(&self) {
        self.gpu_pool_blocks_released
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A host step's variant made immutable samplers no live set had (`pipeline/objects.rs`).
    pub(crate) fn host_step_sampler_set_made(&self) {
        self.host_step_sampler_sets_made
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A steps' thread recorded a batch's host step, which took `took` (`scheduler/steps.rs`).
    pub(crate) fn host_step_recorded(&self, took: std::time::Duration) {
        self.host_steps_recorded.fetch_add(1, Ordering::Relaxed);
        self.host_step_nanos.fetch_add(
            u64::try_from(took.as_nanos()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
    }

    /// A host step ran a chunk of an indirect command buffer's commands from the recording an
    /// earlier step kept (`render/executions/kept.rs`), its records and inputs as they were.
    pub(crate) fn host_step_chunk_kept(&self) {
        self.host_step_chunks_kept.fetch_add(1, Ordering::Relaxed);
    }

    /// A host step recorded a chunk again whose kept recording a record or an input that differed
    /// ruled out (`render/executions/kept.rs`).
    pub(crate) fn host_step_chunk_recorded_again(&self) {
        self.host_step_chunks_recorded_again
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A host step that records inline read a chunk's records in place, its copy for the next
    /// step's comparison put off (`kept_commands::StepCopy`).
    pub(crate) fn host_step_chunk_copy_put_off(&self) {
        self.host_step_chunk_copies_put_off
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A host step compared `records` records with the bytes a kept chunk was recorded from or
    /// those the last step read, taking `took` (`render/executions/kept.rs`).
    pub(crate) fn host_step_records_compared(&self, records: u64, took: std::time::Duration) {
        self.host_step_records_compared
            .fetch_add(records, Ordering::Relaxed);
        self.host_step_compare_nanos.fetch_add(
            u64::try_from(took.as_nanos()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
    }

    /// An execution's replay recorded a draw, plain in every other respect, that reading the heap
    /// kept from being plain, for `unkept` (`render.rs`, `heap_sets::Unkept`).
    pub(crate) fn replayed_heap_draw_unkept(&self, unkept: Unkept) {
        let counter = match unkept {
            Unkept::Unshared => &self.replayed_heap_draws_unshared,
            Unkept::Writes => &self.replayed_heap_draws_writing,
            Unkept::Noted => &self.replayed_heap_draws_noted,
            Unkept::SamplerTable => &self.replayed_heap_draws_over_sampler_tables,
            Unkept::TableTextures => &self.replayed_heap_draws_table_textures,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// A commit's refused recording of an execution of indirect commands was held for its host
    /// step (`deferred.rs`, `held`).
    pub(crate) fn commit_refusal_held(&self) {
        self.commit_refusals_held.fetch_add(1, Ordering::Relaxed);
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

    /// A pass took a backing its queue kept, `in_flight` where another command buffer that took
    /// it had not completed (`attachment_stand_in::pooled_backing`).
    pub(crate) fn pooled_backing_taken(&self, in_flight: bool) {
        self.pooled_backings_taken.fetch_add(1, Ordering::Relaxed);
        if in_flight {
            self.pooled_backings_taken_in_flight
                .fetch_add(1, Ordering::Relaxed);
        }
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

    /// A message of kind `told` was not said, since the device said one of that kind before
    /// (`Device::tell`).
    pub(crate) fn told_withheld(&self, told: Told) {
        let count = match told {
            Told::HeapTextureRefused => &self.heap_texture_refused_withheld,
            Told::TwinDecodeFailed => &self.twin_decode_failed_withheld,
            Told::LateReservation => &self.late_reservation_withheld,
            Told::QueueLimitsReached => &self.queue_limits_reached_withheld,
        };
        count.fetch_add(1, Ordering::Relaxed);
    }

    /// An internal submission was made (`device/submit.rs`): one a host call waits for.
    pub(crate) fn internal_submission(&self) {
        self.internal_submissions.fetch_add(1, Ordering::Relaxed);
    }

    /// A `vkQueueSubmit` or `vkQueueBindSparse` call was made with `infos` `VkSubmitInfo`s
    /// (none for a sparse bind) and `command_buffers` `VkCommandBuffer`s among them.
    /// Made on a thread whose every call is its check's (`step_aside::Counted::Apart`), it is the
    /// checks' work and counted as theirs alone ([`DeviceCheckWork::queue_submissions`]).
    pub(crate) fn queue_submitted(&self, infos: usize, command_buffers: usize) {
        if crate::step_aside::counted_apart() {
            self.check_queue_submissions.fetch_add(1, Ordering::Relaxed);
            return;
        }
        self.queue_submissions.fetch_add(1, Ordering::Relaxed);
        self.submit_infos.fetch_add(infos as u64, Ordering::Relaxed);
        (self.command_buffers_submitted).fetch_add(command_buffers as u64, Ordering::Relaxed);
    }

    /// The `vkQueueSubmit` and `vkQueueBindSparse` calls made so far, the checks' left out
    /// ([`queue_submitted`](Self::queue_submitted)): the clock a resource's age runs on in a
    /// failed submission's line (`submit_failure.rs`).
    pub(crate) fn submissions_made(&self) -> u64 {
        self.queue_submissions.load(Ordering::Relaxed)
    }

    /// A batch's `vkQueueSubmit` (or a group's, or a command buffer's fence's) spent `nanos`
    /// nanoseconds in the driver (`device/relay.rs`), the calling thread `cpu` of its CPU time
    /// across it where the device read that clock (it notes queue times) and the clock read.
    pub(crate) fn batch_submitted_in(&self, nanos: u64, cpu: Option<u64>) {
        self.batch_submit_nanos.fetch_add(nanos, Ordering::Relaxed);
        let cpu = cpu.unwrap_or(0);
        self.batch_submit_cpu_nanos
            .fetch_add(cpu, Ordering::Relaxed);
        if nanos >= LONG_SUBMIT_NANOS {
            self.submit_long_calls.fetch_add(1, Ordering::Relaxed);
            self.submit_long_call_nanos
                .fetch_add(nanos, Ordering::Relaxed);
            self.submit_long_call_cpu_nanos
                .fetch_add(cpu, Ordering::Relaxed);
        }
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

    /// A batch's work went to its `VkQueue` in `pieces` more pieces than one, each a
    /// `VkSubmitInfo` of its own (`rate_mapped_pass.rs`, **Pieces**).
    pub(crate) fn rate_map_pieces(&self, pieces: u64) {
        (self.rate_map_pieces).fetch_add(pieces, Ordering::Relaxed);
    }

    /// A command's draw in a rate-mapped pass was drawn as `ranges` more ranges of its indices
    /// than one, each a draw of its own (`rate_mapped_pass.rs`, **Pieces**).
    pub(crate) fn rate_map_index_ranges(&self, ranges: u64) {
        (self.rate_map_index_ranges).fetch_add(ranges, Ordering::Relaxed);
    }

    /// A piece of a rate-mapped pass's submission ran for `nanos` of GPU time, read from its
    /// timestamps once its work completed (`rate_mapped_pass.rs`, **Pieces**).
    pub(crate) fn rate_map_piece_time(&self, nanos: u64) {
        (self.rate_map_pieces_timed).fetch_add(1, Ordering::Relaxed);
        (self.rate_map_piece_gpu_nanos).fetch_add(nanos, Ordering::Relaxed);
        (self.rate_map_piece_gpu_nanos_most).fetch_max(nanos, Ordering::Relaxed);
    }

    /// A host step's probe of a rate-mapped pass's draws was submitted and waited for
    /// (`rate_mapped_pass.rs`, **Fan-out**), of which it flagged `flagged` draws, of `triangles`
    /// flagged triangles.
    pub(crate) fn rate_map_probe(&self, flagged: u64, triangles: u64, straddling: u64) {
        (self.rate_map_probes).fetch_add(1, Ordering::Relaxed);
        (self.rate_map_flagged_draws).fetch_add(flagged, Ordering::Relaxed);
        (self.rate_map_flagged_triangles).fetch_add(triangles, Ordering::Relaxed);
        (self.rate_map_straddling_triangles).fetch_add(straddling, Ordering::Relaxed);
    }

    /// A host step that fans its pass's triangles out took the making of a probe's or a
    /// fanned-out draw's pipeline the commit's settle began on the device's pool: made already
    /// (`ready`), or waited for (`rate_mapped_pass.rs`, **Fan-out**).
    pub(crate) fn rate_map_fan_out_pipeline_taken(&self, ready: bool) {
        let counter = match ready {
            true => &self.rate_map_fan_out_pipelines_ready,
            false => &self.rate_map_fan_out_pipelines_waited,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// An encoder's first fan-out draw or probe uploaded its region table to the command buffer
    /// (`render/fan_out.rs`).
    pub(crate) fn rate_map_fan_table_uploaded(&self) {
        (self.rate_map_fan_tables_uploaded).fetch_add(1, Ordering::Relaxed);
    }

    /// A draw of a rate-mapped pass recorded region by region through its region-copy variant, in
    /// `recordings` recordings (`render/fan_out.rs`, **Region-copy variant**).
    pub(crate) fn rate_map_region_copy_draw(&self, recordings: u64) {
        (self.rate_map_region_copy_draws).fetch_add(1, Ordering::Relaxed);
        (self.rate_map_region_copy_recordings).fetch_add(recordings, Ordering::Relaxed);
    }

    /// A recording of a draw through its region-copy variant by `route` recorded
    /// `vertex_regions`, its vertices or indices times its instances times the regions it draws
    /// (`render/fan_out.rs`, **Region-copy variant**).
    pub(crate) fn rate_map_region_copy_vertices(&self, vertex_regions: u64, route: CopyRoute) {
        let counter = match route {
            CopyRoute::Whole { .. } => &self.rate_map_whole_copy_vertex_regions,
            CopyRoute::Run => &self.rate_map_run_vertex_regions,
            CopyRoute::Boxed => &self.rate_map_boxed_vertex_regions,
        };
        counter.fetch_add(vertex_regions, Ordering::Relaxed);
        if route == (CopyRoute::Whole { own: true }) {
            (self.rate_map_own_copy_vertex_regions).fetch_add(vertex_regions, Ordering::Relaxed);
        }
    }

    /// A draw that would have fanned out was drawn through its region-copy variant into the
    /// boxes its triangles reach (`render/fan_out.rs`, **Boxed draws**).
    pub(crate) fn rate_map_boxed_draw(&self) {
        (self.rate_map_boxed_draws).fetch_add(1, Ordering::Relaxed);
    }

    /// `draws` of the encoder's own recorded their probes and both their routes, in an instance
    /// whose steps run the routes' kernel, which decides their counts (`render/gpu_routes.rs`).
    pub(crate) fn rate_map_gpu_routed_draws(&self, draws: u64) {
        (self.rate_map_gpu_routed_draws).fetch_add(draws, Ordering::Relaxed);
    }

    /// The routed draws of an instance whose command buffer completed: `fanned` the GPU decided to
    /// fan out, `copied` to draw by region copies (`render/gpu_routes.rs`).
    pub(crate) fn rate_map_gpu_decided(&self, fanned: u64, copied: u64) {
        (self.rate_map_gpu_fanned_draws).fetch_add(fanned, Ordering::Relaxed);
        (self.rate_map_gpu_copied_draws).fetch_add(copied, Ordering::Relaxed);
    }

    /// A fanned-out draw, or a flagged draw's part fanned out, recorded `vertices`, its vertices
    /// or indices times its instances, into boxes its blocks' bounds reach of `regions` regions
    /// (`rate_mapped_pass.rs`, **Fan-out**).
    pub(crate) fn rate_map_fanned_vertices(&self, vertices: u64, regions: u64) {
        (self.rate_map_fanned_vertices).fetch_add(vertices, Ordering::Relaxed);
        let boxed = vertices.saturating_mul(regions);
        (self.rate_map_fanned_box_vertex_regions).fetch_add(boxed, Ordering::Relaxed);
        for (counter, most) in (self.rate_map_fanned_vertices_boxed.iter()).zip(FANNED_BOXES) {
            if regions <= most {
                counter.fetch_add(vertices, Ordering::Relaxed);
            }
        }
    }

    /// A run of a flagged draw's blocks drawn through its region-copy variant into the `regions`
    /// its bounds reach, `every` region of the plan where they reach them all
    /// (`render/fan_out.rs`, `run_regions`).
    pub(crate) fn rate_map_run_regions(&self, regions: u64, every: bool) {
        (self.rate_map_run_region_draws).fetch_add(regions, Ordering::Relaxed);
        (self.rate_map_runs_every_region).fetch_add(u64::from(every), Ordering::Relaxed);
    }

    /// A host step's probe ran for `nanos` of GPU time, read from its timestamps once it completed
    /// (`rate_mapped_pass.rs`, **Fan-out**).
    pub(crate) fn rate_map_probe_time(&self, nanos: u64) {
        (self.rate_map_probe_gpu_nanos).fetch_add(nanos, Ordering::Relaxed);
    }

    /// A host step waited `took` on the host's clock for its probe, from its last part's
    /// submission until the host saw every part complete (`rate_mapped_pass.rs`, **Fan-out**;
    /// `render/executions.rs`, `DeferredPass::probe_parts`).
    pub(crate) fn rate_map_probe_waited(&self, took: std::time::Duration) {
        let nanos = u64::try_from(took.as_nanos()).unwrap_or(u64::MAX);
        (self.rate_map_probe_wait_nanos).fetch_add(nanos, Ordering::Relaxed);
    }

    /// A host step's probe was submitted in `parts` parts (`render/executions.rs`,
    /// `DeferredPass::probe_parts`).
    pub(crate) fn rate_map_probe_parts(&self, parts: u64) {
        (self.rate_map_probe_parts).fetch_add(parts, Ordering::Relaxed);
    }

    /// A host step of a rate-mapped pass probed no draw, for `why` (`render/executions.rs`,
    /// `DeferredPass::fan_out_routes`).
    pub(crate) fn rate_map_step_unprobed(&self, why: Unprobed) {
        let counter = match why {
            Unprobed::Unfanned => &self.rate_map_steps_unfanned,
            Unprobed::NothingToProbe => &self.rate_map_steps_nothing_to_probe,
            Unprobed::ProbingNone => &self.rate_map_steps_probing_none,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// A host step recorded `routes` executions of a rate-mapped pass in their GPU route
    /// (`render/executions/gpu_route.rs`).
    pub(crate) fn rate_map_gpu_routes(&self, routes: u64) {
        (self.rate_map_gpu_routes).fetch_add(routes, Ordering::Relaxed);
    }

    /// A host step waited `took` for its GPU routes' prework (`render/executions/gpu_route.rs`).
    pub(crate) fn rate_map_gpu_route_waited(&self, took: std::time::Duration) {
        let nanos = u64::try_from(took.as_nanos()).unwrap_or(u64::MAX);
        (self.rate_map_gpu_route_wait_nanos).fetch_add(nanos, Ordering::Relaxed);
    }

    /// A draw the probe flagged was drawn in parts (`render/fan_out.rs`, **Parts**), of which
    /// `by_regions` triangles region by region, in `runs` parts.
    pub(crate) fn rate_map_drawn_in_parts(&self, by_regions: u64, runs: u64) {
        (self.rate_map_parted_draws).fetch_add(1, Ordering::Relaxed);
        (self.rate_map_parted_region_triangles).fetch_add(by_regions, Ordering::Relaxed);
        (self.rate_map_region_runs).fetch_add(runs, Ordering::Relaxed);
    }

    /// A command's draw in a rate-mapped pass was drawn once, fanned out to its regions, through
    /// the variant of few copies where `few` (`rate_mapped_pass.rs`, **Fan-out**).
    pub(crate) fn rate_map_fanned_draw(&self, few: bool) {
        (self.rate_map_fanned_draws).fetch_add(1, Ordering::Relaxed);
        if few {
            (self.rate_map_fanned_few_draws).fetch_add(1, Ordering::Relaxed);
        }
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

    /// An encoder began its own `VkCommandBuffer` where its spare command pool's begin ahead owed
    /// it one: the device's begin-ahead thread had not made it (`begin_ahead.rs`).
    pub(crate) fn encoder_primary_begun_ahead_missed(&self) {
        (self.encoder_primaries_begun_ahead_missed).fetch_add(1, Ordering::Relaxed);
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

    /// A batch's footprint for the scheduler's order across queues was built: none was kept, or
    /// what it is built from changed since (`CommandBufferShared::footprint`).
    pub(crate) fn footprint_built(&self) {
        self.footprints_built.fetch_add(1, Ordering::Relaxed);
    }

    /// An ask of a batch's footprint found its command buffer's state locked and waited for it
    /// (`CommandBufferShared::footprint`).
    pub(crate) fn footprint_lock_waited(&self) {
        self.footprint_lock_waits.fetch_add(1, Ordering::Relaxed);
    }

    /// A thread that committed with `CommandBuffer::commit_and_submit` made a pass of its own
    /// over the scheduler's order and submitted `batches` batches itself, the submission checks
    /// of `checked` of them run by it first (`Scheduler::submit_at_commit`).
    pub(crate) fn committer_passed(&self, batches: usize, checked: usize) {
        (self.scheduler_passes_at_commit).fetch_add(1, Ordering::Relaxed);
        (self.batches_submitted_at_commit).fetch_add(batches as u64, Ordering::Relaxed);
        (self.batches_checked_at_commit).fetch_add(checked as u64, Ordering::Relaxed);
    }

    /// A committing thread's pass ended right after its own command buffer's last batch, which has
    /// submission handlers, with batches left for the scheduler's thread
    /// (`scheduler::AtCommit::pass`).
    pub(crate) fn pass_at_commit_ended_early(&self) {
        (self.scheduler_passes_at_commit_ended_early).fetch_add(1, Ordering::Relaxed);
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

    /// A commit of `commit_and_submit` left its batches to the scheduler's thread, its own pass
    /// submitting none of the scheduler's batches, or making none, for `why`
    /// (`scheduler/mod.rs`).
    pub(crate) fn left_at_commit(&self, why: LeftAtCommit) {
        self.left_at_commit[why as usize].fetch_add(1, Ordering::Relaxed);
    }

    /// A batch on a `VkQueue` whose batches group went in a kernel submission of its own, for
    /// `why` (`command_buffer/group.rs`).
    pub(crate) fn batch_alone(&self, why: Alone) {
        self.batches_alone[why as usize].fetch_add(1, Ordering::Relaxed);
    }

    /// A batch of a command queue made with `Device::new_command_queue_unshared` was submitted,
    /// or held in a group (`command_buffer/submit.rs`).
    pub(crate) fn unshared_batch(&self) {
        self.batches_of_unshared_queues
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A group of batches held for one kernel submission went, sent by `end`
    /// (`command_buffer/group.rs`).
    pub(crate) fn group_ended(&self, end: GroupEnd) {
        self.group_ends[end as usize].fetch_add(1, Ordering::Relaxed);
    }

    /// A pass after the scheduler's sleep ended at its longest wait let a batch go: a wake that
    /// should have come did not (`scheduler/mod.rs`).
    pub(crate) fn scheduler_slice_rescued(&self) {
        self.scheduler_slice_rescues.fetch_add(1, Ordering::Relaxed);
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

    /// A command buffer became `Scheduled` before its last batch went, behind its queue's head
    /// waiting at its host step's gate (`CommandBufferShared::schedule_early`).
    pub(crate) fn scheduled_behind_host_step(&self) {
        self.scheduled_behind_host_steps
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A head's host step gate, as it was taken, left out work submitted before it
    /// (`scheduler/mod.rs`'s `gate_of`).
    pub(crate) fn host_step_gate_narrowed(&self) {
        self.host_step_gates_narrowed
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Such a look left a head's submission checks waiting for a submitted GPU signal to land
    /// (`scheduler/mod.rs`'s `check_heads`).
    pub(crate) fn head_awaiting_check_signals(&self) {
        self.heads_awaiting_check_signals
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A look let a head go under `unshared_hold::IDLE_BUDGET` that `unshared_hold::BUDGET`
    /// would hold back: no unshared queue on its `VkQueue` committed within
    /// `unshared_hold::ACTIVE`.
    pub(crate) fn head_let_go_unshared_idle(&self) {
        self.heads_let_go_unshared_idle
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A look let a head go on its turn under `unshared_hold::IDLE_BUDGET` that
    /// `unshared_hold::BUDGET` would hold back: no other command queue took a batch on its
    /// `VkQueue` within `unshared_hold::ACTIVE` (`queue_turns.rs`).
    pub(crate) fn head_let_go_turn_idle(&self) {
        self.heads_let_go_turn_idle.fetch_add(1, Ordering::Relaxed);
    }

    /// A look held a head whose room had come behind an unshared queue's commit on its `VkQueue`
    /// not yet taken into the scheduler's order, or leading its line on its way
    /// (`unshared_hold.rs`, "Commits not yet taken in", "Commits on their way").
    pub(crate) fn head_held_for_unshared_commits(&self) {
        self.heads_held_for_unshared_commits
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

    /// The program held `live` command queues of the device at once (`queue.rs`).
    pub(crate) fn command_queues_live(&self, live: u64) {
        (self.command_queues_live_most).fetch_max(live, Ordering::Relaxed);
    }

    /// A new unshared command queue took a `VkQueue` another command queue held, the round robin
    /// having none that no command queue holds (`queue_share.rs`).
    pub(crate) fn unshared_queue_shared_a_vk_queue(&self) {
        (self.unshared_queues_sharing_a_vk_queue).fetch_add(1, Ordering::Relaxed);
    }

    /// A command queue moved to another `VkQueue` (`queue_share.rs`): onto one with another
    /// queue's work in flight where `joined`, off one an unshared queue holds where
    /// `off_unshared`.
    pub(crate) fn queue_moved(&self, joined: bool, off_unshared: bool) {
        self.queue_moves.fetch_add(1, Ordering::Relaxed);
        if joined {
            self.queue_joins.fetch_add(1, Ordering::Relaxed);
        }
        if off_unshared {
            (self.queue_moves_off_unshared).fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A command queue went long (`queue_share.rs`).
    pub(crate) fn queue_went_long(&self) {
        self.queues_gone_long.fetch_add(1, Ordering::Relaxed);
    }

    /// A long command queue settled (`queue_share.rs`).
    pub(crate) fn long_queue_settled(&self) {
        self.long_queue_settles.fetch_add(1, Ordering::Relaxed);
    }

    /// A command queue's choice passed over a busy `VkQueue` it would have joined, for the
    /// unshared queue it holds (`queue_share.rs`).
    pub(crate) fn join_withheld_from_unshared(&self) {
        (self.joins_withheld_from_unshared).fetch_add(1, Ordering::Relaxed);
    }

    /// A command queue's choice found no other `VkQueue` with batches in flight
    /// (`queue_share.rs`).
    pub(crate) fn picked_with_none_busy(&self) {
        self.picks_with_none_busy.fetch_add(1, Ordering::Relaxed);
    }

    /// A command queue's choice joined no `VkQueue`, passing over a busy one for its oldest
    /// batch's age (`queue_share.rs`).
    pub(crate) fn join_withheld_from_old_head(&self) {
        (self.joins_withheld_from_old_heads).fetch_add(1, Ordering::Relaxed);
    }

    /// A command buffer's first batch went to its `VkQueue` behind another command queue's batch
    /// not seen complete (`queue_wait.rs`), of an unshared queue where `unshared`.
    pub(crate) fn behind_other_queue(&self, unshared: bool) {
        for tally in &self.behind_other_queues[..1 + usize::from(unshared)] {
            tally.count.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Such a command buffer's wait ended, `waited.nanos` after its submission (`queue_wait.rs`).
    pub(crate) fn waited_behind_other_queue(
        &self,
        waited: crate::queue_wait::Waited,
        unshared: bool,
    ) {
        let nanos = waited.nanos;
        if unshared && nanos > 16_000_000 {
            let [batches, most, longest, one] = &self.unshared_long_waits;
            batches.fetch_add(waited.batches, Ordering::Relaxed);
            most.fetch_max(waited.batches, Ordering::Relaxed);
            longest.fetch_add(waited.longest, Ordering::Relaxed);
            if waited.longest > 16_000_000 {
                one.fetch_add(1, Ordering::Relaxed);
            }
        }
        for tally in &self.behind_other_queues[..1 + usize::from(unshared)] {
            if nanos > 16_000_000 {
                tally.over_16ms.fetch_add(1, Ordering::Relaxed);
            }
            if nanos > 33_000_000 {
                tally.over_33ms.fetch_add(1, Ordering::Relaxed);
            }
            tally.longest_nanos.fetch_max(nanos, Ordering::Relaxed);
        }
    }

    /// The device's check of its GPU's idling between `VkQueue`s ran in this process
    /// (`queue_switch_check.rs`).
    pub(crate) fn queue_switch_check_ran(&self) {
        (self.queue_switch_checks_run).fetch_add(1, Ordering::Relaxed);
    }

    /// That check's verdict was read from the record of an earlier one (`queue_share.rs`).
    pub(crate) fn queue_switch_check_recalled(&self) {
        (self.queue_switch_checks_recalled).fetch_add(1, Ordering::Relaxed);
    }

    /// The device's submission checks ran in this process (`submit_check.rs`).
    pub(crate) fn submit_checks_ran(&self) {
        (self.submit_checks_run).fetch_add(1, Ordering::Relaxed);
    }

    /// Their verdicts were read from the record of an earlier run (`submit_check.rs`).
    pub(crate) fn submit_checks_recalled(&self) {
        (self.submit_checks_recalled).fetch_add(1, Ordering::Relaxed);
    }

    /// The submission checks made `calls` `vkQueueSubmit` calls, theirs or the relay's marks
    /// after their uses of a queue, which `queue_submissions` does not count (`submit_check.rs`,
    /// `device/relay.rs`).
    pub(crate) fn submit_check_called(&self, calls: usize) {
        (self.submit_check_calls).fetch_add(calls as u64, Ordering::Relaxed);
    }

    /// A step of one of the device's checks waited `waited` for the device's allocations and
    /// frees on other threads (`step_aside.rs`).
    pub(crate) fn check_stepped_aside(&self, waited: std::time::Duration) {
        self.check_steps_aside.fetch_add(1, Ordering::Relaxed);
        let nanos = u64::try_from(waited.as_nanos()).unwrap_or(u64::MAX);
        self.check_aside_nanos.fetch_add(nanos, Ordering::Relaxed);
    }

    /// A step of the device's view check stood by for `waited` until the device's work was quiet
    /// (`view_check.rs`).
    pub(crate) fn view_check_stood_by(&self, waited: std::time::Duration) {
        self.view_check_stand_bys.fetch_add(1, Ordering::Relaxed);
        let nanos = u64::try_from(waited.as_nanos()).unwrap_or(u64::MAX);
        (self.view_check_stand_by_nanos).fetch_add(nanos, Ordering::Relaxed);
    }

    /// A device that could share `VkQueue`s does not: its command buffers write no timestamps it
    /// reads (`queue_switch_check::timed`).
    pub(crate) fn queue_sharing_untimed(&self) {
        (self.queue_sharing_untimed).fetch_add(1, Ordering::Relaxed);
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
    /// recorded after a barrier for the encoder's earlier commands (`tracker::EncoderUses`), or a
    /// compute, blit or render encoder's after one for the earlier encoders of its group
    /// (`encoder_fold.rs`).
    pub(crate) fn command_barrier(&self) {
        self.command_barriers.fetch_add(1, Ordering::Relaxed);
    }

    /// An encoder went on in a new `VkCommandBuffer` before a command that uses an image whose
    /// memory a command before it there wrote through a buffer (`alias_split.rs`).
    pub(crate) fn command_buffer_split_at_aliases(&self) {
        self.command_buffers_split_at_aliases
            .fetch_add(1, Ordering::Relaxed);
    }

    /// An encoder past the first `PLANNED_BARRIERS` of its command buffer began its own command
    /// buffer with a barrier from everything before (`encoder.rs`).
    pub(crate) fn encoder_barriered_at_open(&self) {
        self.encoders_barriered_at_open
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A compute, blit or render encoder joined the group of encoders before it, recording into
    /// their `VkCommandBuffer` (`encoder_fold.rs`).
    pub(crate) fn encoder_folded(&self) {
        self.encoders_folded.fetch_add(1, Ordering::Relaxed);
    }

    /// A compute, blit or render encoder that asked for a group of encoders joined none, for
    /// `why` (`encoder_fold.rs`'s **Refusals**).
    pub(crate) fn fold_refused(&self, why: Refusal) {
        let counter = match why {
            Refusal::Kind => &self.folds_refused_kind,
            Refusal::Tail => &self.folds_refused_tail,
            Refusal::Batch => &self.folds_refused_batch,
            Refusal::State => &self.folds_refused_state,
            Refusal::Parts => &self.folds_refused_parts,
            Refusal::Ends => &self.folds_refused_ends,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// A command of a group's encoder recorded a barrier after the writes a member's fence update
    /// covers (`encoder_fold.rs`'s **Fences**).
    pub(crate) fn fold_fence_barrier(&self) {
        self.fold_fence_barriers.fetch_add(1, Ordering::Relaxed);
    }

    /// A compute encoder opened on the core's box and the bind state its command buffer's last
    /// compute encoder parked (`compute/live.rs`'s `Parked`).
    pub(crate) fn parked_compute_taken(&self) {
        self.parked_compute_taken.fetch_add(1, Ordering::Relaxed);
    }

    /// `lists` of a released command buffer's lists of records held more room than it started
    /// with (`command_buffer/spare_states.rs`).
    pub(crate) fn record_lists_grown(&self, lists: u64) {
        self.record_lists_grown.fetch_add(lists, Ordering::Relaxed);
    }

    /// The device kept a released command buffer's state, reset, for the next
    /// (`command_buffer/spare_states.rs`).
    pub(crate) fn record_storage_kept(&self) {
        self.record_storage_kept.fetch_add(1, Ordering::Relaxed);
    }

    /// A command buffer was made on a state built for it, the device keeping none
    /// (`command_buffer/spare_states.rs`).
    pub(crate) fn command_buffer_state_made(&self) {
        self.command_buffer_states_made
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A render pass's first instance began inline in a primary command buffer of its encoder's
    /// (`render/inline_first.rs`).
    pub(crate) fn inline_pass(&self) {
        self.inline_passes.fetch_add(1, Ordering::Relaxed);
    }

    /// A render encoder opened from the kept draw its command buffer's last render encoder
    /// carried, or where `across`, the last render encoder of a command buffer committed on its
    /// queue (`render/carried.rs`, `WorkCounts::kept_draws_carried`,
    /// `WorkCounts::kept_draws_carried_across`).
    pub(crate) fn kept_draw_carried(&self, across: bool) {
        self.kept_draws_carried.fetch_add(1, Ordering::Relaxed);
        if across {
            self.kept_draws_carried_across
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A kept draw carried into its command buffer's next render encoder was let go for `why`
    /// (`WorkCounts::kept_draws_carry_dropped_*`).
    pub(crate) fn kept_draw_carry_dropped(&self, why: CarryDropped) {
        self.kept_draws_carry_dropped[why as usize].fetch_add(1, Ordering::Relaxed);
    }

    /// A render encoder opened from a carried kept draw took `count` kept draws of other pipeline
    /// states carried with it (`render/carried.rs`, `WorkCounts::kept_draws_carried_aside`).
    pub(crate) fn kept_draws_carried_aside(&self, count: u64) {
        if count > 0 {
            self.kept_draws_carried_aside
                .fetch_add(count, Ordering::Relaxed);
        }
    }

    /// A render encoder that ended let go of `count` kept draws that would carry but for their
    /// reads of a texture an attachment of their pass renders (`render/carried.rs`,
    /// `WorkCounts::kept_draws_uncarried_attached`).
    pub(crate) fn kept_draws_uncarried_attached(&self, count: u64) {
        if count > 0 {
            self.kept_draws_uncarried_attached
                .fetch_add(count, Ordering::Relaxed);
        }
    }

    /// A render encoder ended carrying no kept draw, for `why`
    /// (`WorkCounts::encoders_uncarried_*`).
    pub(crate) fn encoder_uncarried(&self, why: Uncarried) {
        self.encoders_uncarried[why as usize].fetch_add(1, Ordering::Relaxed);
    }

    /// A render encoder that drew but ended with no kept draw came to `next` for its command
    /// buffer's next render encoder (`WorkCounts::encoders_none_kept_next_*`).
    pub(crate) fn encoder_none_kept_next(&self, next: NoneKeptNext) {
        self.encoders_none_kept_next[next as usize].fetch_add(1, Ordering::Relaxed);
    }

    /// A render encoder ended whose last draw drew nothing (`WorkCounts::encoders_last_draw_empty`).
    pub(crate) fn encoder_last_draw_empty(&self) {
        self.encoders_last_draw_empty
            .fetch_add(1, Ordering::Relaxed);
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

    /// A render encoder's first instance recorded into the render pass instance the encoder
    /// before it left open (`encoder_fold.rs`, **Continued passes**).
    pub(crate) fn render_pass_continued(&self) {
        self.render_passes_continued.fetch_add(1, Ordering::Relaxed);
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

    /// A program's commit call took `whole` nanoseconds on its thread, `phases` of them in its
    /// phases, each apart from the others (`commit_times.rs`: lowering, recording, pipeline
    /// waits, lock wait, pass, submissions).
    pub(crate) fn commit_timed(&self, whole: u64, phases: &crate::commit_times::Times) {
        let [lowering, recording, pipeline_wait, lock_wait, pass, submit] = *phases;
        self.commit_calls.fetch_add(1, Ordering::Relaxed);
        self.commit_nanos.fetch_add(whole, Ordering::Relaxed);
        (self.commit_lowering_nanos).fetch_add(lowering, Ordering::Relaxed);
        (self.commit_recording_nanos).fetch_add(recording, Ordering::Relaxed);
        (self.commit_pipeline_wait_nanos).fetch_add(pipeline_wait, Ordering::Relaxed);
        (self.commit_lock_wait_nanos).fetch_add(lock_wait, Ordering::Relaxed);
        self.commit_pass_nanos.fetch_add(pass, Ordering::Relaxed);
        self.commit_submit_nanos
            .fetch_add(submit, Ordering::Relaxed);
    }

    /// A commit left its batches to the scheduler's thread (`scheduler/inbox.rs`).
    pub(crate) fn commit_handed_off(&self) {
        self.commits_handed_off.fetch_add(1, Ordering::Relaxed);
    }

    /// A submission on the staging queue was made (`device/staging.rs`).
    pub(crate) fn staging_submission(&self) {
        self.staging_submissions.fetch_add(1, Ordering::Relaxed);
    }

    /// A batch was submitted to the copy queue's `VkQueue`, of a family of transfers alone
    /// (`Device::new_command_queue_for_copies`).
    pub(crate) fn copy_queue_batch(&self) {
        self.copy_queue_batches.fetch_add(1, Ordering::Relaxed);
    }

    /// A host call's copy was queued for the staging queue's next submission
    /// (`texture/staging_copy.rs`).
    pub(crate) fn staging_copy_queued(&self) {
        self.staging_copies_queued.fetch_add(1, Ordering::Relaxed);
    }

    /// New images' first moves were queued for the staging queue's next submission
    /// (`texture/staging_copy.rs`).
    pub(crate) fn staging_moves_queued(&self, moves: u64) {
        self.staging_moves_queued
            .fetch_add(moves, Ordering::Relaxed);
    }

    /// A staging submission carried queued moves or copies (`texture/staging_copy.rs`).
    pub(crate) fn staging_flush(&self) {
        self.staging_flushes.fetch_add(1, Ordering::Relaxed);
    }

    /// A submission of the device's waited on the GPU for a staging submission the host had not
    /// seen complete (`device/staging.rs`).
    pub(crate) fn staging_gpu_wait(&self) {
        self.staging_gpu_waits.fetch_add(1, Ordering::Relaxed);
    }

    /// A submission on the staging queue found its kept command buffers all pending and made
    /// another (`device/staging.rs`).
    pub(crate) fn staging_buffer_grown(&self) {
        self.staging_buffers_grown.fetch_add(1, Ordering::Relaxed);
    }

    /// A submission on the staging queue or the fill queue waited on the host for one of its
    /// command buffers to come free (`device/staging.rs`).
    pub(crate) fn staging_buffer_wait(&self) {
        self.staging_buffer_waits.fetch_add(1, Ordering::Relaxed);
    }

    /// A submission on the fill queue was made (`device/staging.rs`).
    pub(crate) fn fill_queue_submission(&self) {
        self.fill_queue_submissions.fetch_add(1, Ordering::Relaxed);
    }

    /// A batch waited on the GPU for fill-queue submissions, whose fills its commands may reach
    /// where `reached` (`command_buffer/submit.rs`).
    pub(crate) fn fill_queue_waited(&self, reached: bool) {
        self.fill_queue_waits.fetch_add(1, Ordering::Relaxed);
        if !reached {
            self.fill_queue_waits_unused.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A batch waited on the GPU for batches on other `VkQueue`s that did new resources' first
    /// work, whose memory its commands or its own first work may reach where `reached`
    /// (`command_buffer/submit.rs`).
    pub(crate) fn first_work_waited(&self, reached: bool) {
        self.first_work_waits.fetch_add(1, Ordering::Relaxed);
        if !reached {
            self.first_work_waits_unused.fetch_add(1, Ordering::Relaxed);
        }
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

    /// A shadowed carrier's decode into its decoded twin was recorded after a copy in that moved
    /// texels (`shadow::ShadowDecode`).
    pub(crate) fn shadow_twin_decode(&self) {
        self.shadow_twin_decodes.fetch_add(1, Ordering::Relaxed);
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

    /// A compute encoder looked at the textures it made resident for their shadows or for one a
    /// carrier holds (`shadow::ResidentShadows`, `pvrtc::ResidentUnreadable`).
    pub(crate) fn resident_textures_looked_at(&self) {
        self.resident_textures_looked_at
            .fetch_add(1, Ordering::Relaxed);
    }

    /// `count` resources an ended encoder made resident were joined into its command buffer's
    /// residency (`command_buffer::State::joined_residency`).
    pub(crate) fn resident_resources_joined(&self, count: u64) {
        self.resident_resources_joined
            .fetch_add(count, Ordering::Relaxed);
    }

    /// An open of a compute or render encoder looked at `count` ended encoders' residencies for
    /// the shadowed textures it starts with (`command_buffer::State::inherited`).
    pub(crate) fn ended_residencies_looked_at(&self, count: u64) {
        self.ended_residencies_looked_at
            .fetch_add(count, Ordering::Relaxed);
    }

    /// A residency's buffers were indexed by Metal address (`residency::AddressIndex`).
    pub(crate) fn resident_address_index_made(&self) {
        self.resident_address_indexes_made
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A residency listed its `count` resources for the kept key of the executions that take it
    /// (`Residency::kept_key`).
    pub(crate) fn resident_kept_keys_listed(&self, count: u64) {
        self.resident_kept_keys_listed
            .fetch_add(count, Ordering::Relaxed);
    }

    /// A root table's own resident set listed `count` ranges, noted for its command buffer's
    /// resident set (`command_buffer::State::root_table`).
    pub(crate) fn root_table_ranges_noted(&self, count: u64) {
        self.root_table_ranges_noted
            .fetch_add(count, Ordering::Relaxed);
    }

    /// A command's pipeline layout was read for the device's texture and sampler tables it binds
    /// (`heap_sets::HeapTables`, `texture_table`, `sampler_table`).
    pub(crate) fn heap_tables_worked_out(&self) {
        self.heap_tables_worked_out.fetch_add(1, Ordering::Relaxed);
    }

    /// A draw took the device's live sampler states to list (`sampler::ArgumentSamplers`).
    pub(crate) fn argument_samplers_taken(&self) {
        self.argument_samplers_taken.fetch_add(1, Ordering::Relaxed);
    }

    /// The runtime made a render pass (`render_target::create`).
    pub(crate) fn render_pass_made(&self) {
        self.render_passes_made.fetch_add(1, Ordering::Relaxed);
    }

    /// A bind of a variant made the notices it announces at every occurrence
    /// (`VariantObjects::announce_bound`).
    pub(crate) fn bound_notices_made(&self) {
        self.bound_notices_made.fetch_add(1, Ordering::Relaxed);
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

    /// A linear attachment's merge made its scratch (`made`), or took one its command buffer's
    /// earlier merges were done with (`linear_merge.rs`, `merge_scratch.rs`).
    pub(crate) fn linear_merge_scratch(&self, made: bool) {
        let counter = match made {
            true => &self.linear_merge_scratches_made,
            false => &self.linear_merge_scratches_reused,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// A linear attachment's merge was recorded: after its pass, or at its batch's submission
    /// after the last pass of its run (`pass_runs.rs`).
    pub(crate) fn linear_merge_recorded(&self) {
        self.linear_merges_recorded.fetch_add(1, Ordering::Relaxed);
    }

    /// A pass of a run took over the merge of the run's pass before it (`pass_runs.rs`).
    pub(crate) fn linear_merge_taken_over(&self) {
        self.linear_merges_taken_over
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A pass of a linear attachment's run, encoded again alone, copied its run's box in again
    /// and kept it after itself (`pass_runs.rs`'s **Encoded again**).
    pub(crate) fn linear_run_reloaded(&self) {
        self.linear_runs_reloaded.fetch_add(1, Ordering::Relaxed);
    }

    /// A pass rendered a texture over a buffer it does not store through the image its texture
    /// keeps (`kept_bytes.rs`).
    pub(crate) fn kept_bytes_pass(&self) {
        self.kept_bytes_passes.fetch_add(1, Ordering::Relaxed);
    }

    /// A pass rendered a texture's own image it stores `DontCare` through a backing its queue
    /// keeps (`kept_bytes.rs`).
    pub(crate) fn kept_texels_pass(&self) {
        self.kept_texels_passes.fetch_add(1, Ordering::Relaxed);
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

    /// A no-copy buffer was released while a buffer bound inside the import of its pages lived:
    /// the import kept its keep-alive, or, given none, its deallocator waits for the free
    /// (`buffer.rs`'s `Inside`).
    pub(crate) fn host_release_held_inside(&self) {
        self.host_releases_held_inside
            .fetch_add(1, Ordering::Relaxed);
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
    /// asks on Linux and Windows).
    #[cfg(any(target_os = "linux", windows))]
    pub(crate) fn host_mappings_queried(&self, mappings: u64) {
        if mappings > 0 {
            self.host_mappings_queried
                .fetch_add(mappings, Ordering::Relaxed);
        }
    }

    /// `imports` imports of dma-bufs a caller's source made of host pages (its chunks of their
    /// own, and shared files no buffer had imported yet) were made, and a buffer bound in them,
    /// where the device refused the pages' pointer (`dma_buf.rs`).
    #[cfg(unix)]
    pub(crate) fn dma_buf_imported(&self, imports: usize) {
        self.dma_buf_imports
            .fetch_add(imports as u64, Ordering::Relaxed);
    }

    /// A piece of a shared file was bound in the device's import of the file, live already,
    /// its join taking `nanos` nanoseconds (`dma_buf.rs`'s `file_import`).
    #[cfg(unix)]
    pub(crate) fn dma_buf_piece_joined(&self, nanos: u64) {
        self.dma_buf_pieces_joined.fetch_add(1, Ordering::Relaxed);
        self.dma_buf_join_nanos.fetch_add(nanos, Ordering::Relaxed);
    }

    /// The device's import of a dma-buf of host pages, made or failed, took `nanos` nanoseconds
    /// (`dma_buf.rs`'s `import_timed`).
    #[cfg(unix)]
    pub(crate) fn dma_buf_import_took(&self, nanos: u64) {
        self.dma_buf_import_calls.fetch_add(1, Ordering::Relaxed);
        self.dma_buf_import_nanos
            .fetch_add(nanos, Ordering::Relaxed);
    }

    /// A shared file was imported and is held (`dma_buf.rs`'s `file_import`), besides
    /// [`dma_buf_memory_imported`](Self::dma_buf_memory_imported).
    #[cfg(unix)]
    pub(crate) fn dma_buf_shared_import_made(&self) {
        (self.dma_buf_shared_imports_live).fetch_add(1, Ordering::Relaxed);
    }

    /// `count` shared files' imports were freed, besides
    /// [`dma_buf_memories_freed`](Self::dma_buf_memories_freed).
    #[cfg(unix)]
    pub(crate) fn dma_buf_shared_imports_freed(&self, count: usize) {
        (self.dma_buf_shared_imports_live).fetch_sub(count as u64, Ordering::Relaxed);
    }

    /// A generation's map came to hold a shared file's import for the handles that keep the file
    /// where `kept`, else let go of that hold (`dma_buf/kept.rs`).
    #[cfg(unix)]
    pub(crate) fn dma_buf_shared_import_kept(&self, kept: bool) {
        match kept {
            true => self
                .dma_buf_shared_imports_kept
                .fetch_add(1, Ordering::Relaxed),
            false => self
                .dma_buf_shared_imports_kept
                .fetch_sub(1, Ordering::Relaxed),
        };
    }

    /// A dma-buf the driver refused to import dedicated to nothing, in every type tried, was
    /// imported dedicated to the one buffer bound in it (`dma_buf.rs`'s `chunks_buffer`).
    #[cfg(unix)]
    pub(crate) fn dma_buf_buffer_import_dedicated(&self) {
        (self.dma_buf_buffer_imports_dedicated).fetch_add(1, Ordering::Relaxed);
    }

    /// A `VkDeviceMemory` was imported from a dma-buf of a caller's source and is held: an
    /// image's dedicated import where `of_image`, else a chunk of a buffer's pages (`dma_buf.rs`).
    #[cfg(unix)]
    pub(crate) fn dma_buf_memory_imported(&self, of_image: bool) {
        self.dma_buf_live(of_image).fetch_add(1, Ordering::Relaxed);
    }

    /// `count` such imports were freed.
    #[cfg(unix)]
    pub(crate) fn dma_buf_memories_freed(&self, of_image: bool, count: usize) {
        (self.dma_buf_live(of_image)).fetch_sub(count as u64, Ordering::Relaxed);
    }

    /// The live gauge of dma-buf imports of images where `of_image`, else of buffers' chunks.
    #[cfg(unix)]
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

    /// Of those, one released as the releasable imports held more imports on the device's list
    /// than its most (`host_residency::RESIDENT_MOST`).
    pub(crate) fn host_import_released_over_most(&self) {
        (self.host_imports_released_over_most).fetch_add(1, Ordering::Relaxed);
    }

    /// A released import was imported again.
    pub(crate) fn host_import_reimported(&self) {
        self.host_imports_reimported.fetch_add(1, Ordering::Relaxed);
    }

    /// A no-copy buffer came to lie in a releasable import beside another, which it left
    /// releasable (`host_residency.rs`).
    pub(crate) fn host_import_joined_releasable(&self) {
        (self.host_imports_joined_releasable).fetch_add(1, Ordering::Relaxed);
    }

    /// A releasable import holds its memory from now on (`resident`), or no longer (released,
    /// gone, or kept for good).
    pub(crate) fn host_import_resident(&self, resident: bool) {
        match resident {
            true => self.host_imports_resident.fetch_add(1, Ordering::Relaxed),
            false => self.host_imports_resident.fetch_sub(1, Ordering::Relaxed),
        };
    }

    /// An import a read other than a blit's kept for good holds its memory from now on
    /// (`held`), or no longer (gone).
    pub(crate) fn host_import_held_for_good(&self, held: bool) {
        match held {
            true => self
                .host_imports_held_for_good
                .fetch_add(1, Ordering::Relaxed),
            false => self
                .host_imports_held_for_good
                .fetch_sub(1, Ordering::Relaxed),
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

    /// The device imported `bytes` of host pages by their pointer (`memory::import_host`).
    pub(crate) fn host_pointer_import_made(&self, bytes: u64) {
        (self.host_pointer_imports_made).fetch_add(1, Ordering::Relaxed);
        (self.host_pointer_import_bytes_made).fetch_add(bytes, Ordering::Relaxed);
    }

    /// The runtime kept host pages the kernel cannot pin from the driver
    /// (`Device::refuse_unpinnable`).
    pub(crate) fn host_import_unpinnable(&self) {
        (self.host_imports_unpinnable).fetch_add(1, Ordering::Relaxed);
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
            CopiedBecause::DeviceRefused => &self.copied_host_buffers_device_refused,
            CopiedBecause::Moved => &self.copied_rebind_pages_moved,
            CopiedBecause::ReadOnly => &self.copied_read_only_memory,
            CopiedBecause::Unwritable => &self.copied_unwritable_pages,
            CopiedBecause::OverImport | CopiedBecause::OverRamChunk => {
                &self.copied_over_live_imports
            }
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

    /// A no-copy buffer made over, or a rebind to, the bytes a live copied host buffer was made
    /// over lay in its copy, making none of its own (`copied.rs`, `host_aliases.rs`).
    pub(crate) fn copied_host_buffer_shared(&self) {
        (self.copied_host_buffers_shared).fetch_add(1, Ordering::Relaxed);
    }

    /// The device imported a chunk of `bytes` of a guest RAM it uses (`host_ram.rs`).
    pub(crate) fn host_ram_chunk_imported(&self, bytes: u64) {
        (self.host_ram_chunks_imported).fetch_add(1, Ordering::Relaxed);
        (self.host_ram_chunk_bytes).fetch_add(bytes, Ordering::Relaxed);
        (self.host_ram_chunks_live).fetch_add(1, Ordering::Relaxed);
    }

    /// The import of a chunk of a guest RAM took `nanos` nanoseconds, made or failed, by whoever
    /// made it (`host_ram.rs`).
    pub(crate) fn host_ram_chunk_import_took(&self, nanos: u64) {
        (self.host_ram_chunk_import_nanos).fetch_add(nanos, Ordering::Relaxed);
    }

    /// A copy between a guest RAM and a buffer waited `nanos` nanoseconds on its encoding thread
    /// for chunks' imports, made there or under way (`host_ram/copy.rs`).
    pub(crate) fn host_ram_copy_waited(&self, nanos: u64) {
        (self.host_ram_copy_import_waits).fetch_add(1, Ordering::Relaxed);
        (self.host_ram_copy_import_wait_nanos).fetch_add(nanos, Ordering::Relaxed);
    }

    /// A gather from a guest RAM was staged through the CPU (`host_ram/copy.rs`).
    pub(crate) fn host_ram_copy_staged(&self) {
        (self.host_ram_copies_staged).fetch_add(1, Ordering::Relaxed);
    }

    /// A copy between a guest RAM and a buffer was recorded by the plan kept of the same copy
    /// before it (`host_ram/copy.rs`, `KeptPlan`).
    pub(crate) fn host_ram_copy_plan_reused(&self) {
        (self.host_ram_copy_plans_reused).fetch_add(1, Ordering::Relaxed);
    }

    /// A plan kept of a copy between a guest RAM and a buffer put its table in a buffer of its
    /// own at its first reuse (`host_ram/copy.rs`, `KeptPlan`).
    pub(crate) fn host_ram_copy_table_kept(&self) {
        (self.host_ram_copy_tables_kept).fetch_add(1, Ordering::Relaxed);
    }

    /// A staged gather's host step copied its pages into its staging in `nanos` nanoseconds,
    /// after the work submitted before it completed where `gated` (`host_ram/copy.rs`).
    pub(crate) fn host_ram_staged_filled(&self, nanos: u64, gated: bool) {
        (self.host_ram_staged_fill_nanos).fetch_add(nanos, Ordering::Relaxed);
        if gated {
            (self.host_ram_staged_fills_gated).fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A scatter into a guest RAM was staged through the CPU (`host_ram/copy.rs`).
    pub(crate) fn host_ram_scatter_staged(&self) {
        (self.host_ram_scatters_staged).fetch_add(1, Ordering::Relaxed);
    }

    /// A staged scatter's host step copied its staging into the pages in `nanos` nanoseconds
    /// (`host_ram/copy.rs`).
    pub(crate) fn host_ram_staged_drained(&self, nanos: u64) {
        (self.host_ram_staged_drain_nanos).fetch_add(nanos, Ordering::Relaxed);
    }

    /// The device imported a chunk of a guest RAM, or an object's run inside one, as a dma-buf
    /// of the RAM's source (`host_ram.rs`), besides counting it as a chunk or a run.
    pub(crate) fn host_ram_dma_buf_imported(&self) {
        (self.host_ram_dma_buf_imports).fetch_add(1, Ordering::Relaxed);
    }

    /// The device imported a list of a copy's runs of a guest RAM as one dma-buf of the RAM's
    /// source of run lists (`host_ram.rs`), besides counting it live as a run's import.
    pub(crate) fn host_ram_run_list_imported(&self) {
        (self.host_ram_run_list_imports).fetch_add(1, Ordering::Relaxed);
    }

    /// The import of a list of a copy's runs of a guest RAM took `nanos` nanoseconds on the
    /// copy's encoding thread, the source's dma-buf and the device's import together, made or
    /// failed (`host_ram.rs`).
    pub(crate) fn host_ram_run_list_import_took(&self, nanos: u64) {
        (self.host_ram_run_list_import_nanos).fetch_add(nanos, Ordering::Relaxed);
    }

    /// An import of a chunk of a guest RAM, or of an object's run inside one, by its pointer
    /// failed (`host_ram.rs`).
    pub(crate) fn host_ram_pointer_import_failed(&self) {
        (self.host_ram_pointer_imports_failed).fetch_add(1, Ordering::Relaxed);
    }

    /// An import of a chunk of a guest RAM, of an object's run inside one, or of the page a use
    /// asks the driver with, as a dma-buf of the RAM's source failed on a device that imports
    /// dma-bufs (`dma_buf.rs`).
    #[cfg(unix)]
    pub(crate) fn host_ram_dma_buf_import_failed(&self) {
        (self.host_ram_dma_buf_imports_failed).fetch_add(1, Ordering::Relaxed);
    }

    /// The device's thread imported a chunk of a guest RAM ahead of every object over it
    /// (`host_ram.rs`), besides counting it as [`Counters::host_ram_chunk_imported`] does.
    pub(crate) fn host_ram_chunk_imported_ahead(&self) {
        (self.host_ram_chunks_imported_ahead).fetch_add(1, Ordering::Relaxed);
    }

    /// A chunk of a guest RAM was not imported, as a live import of an object's own held some of
    /// its pages (`host_ram.rs`).
    pub(crate) fn host_ram_chunk_held_back(&self) {
        (self.host_ram_chunks_held_back).fetch_add(1, Ordering::Relaxed);
    }

    /// The device's thread faulted a chunk of a guest RAM in before its import (`host_ram.rs`).
    pub(crate) fn host_ram_chunk_faulted_ahead(&self) {
        (self.host_ram_chunks_faulted_ahead).fetch_add(1, Ordering::Relaxed);
    }

    /// A free of one of the device's own threads that free in runs waited `waited` first
    /// (`step_aside::pace`): the residency thread's or a guest RAM's.
    pub(crate) fn memory_free_paced(&self, waited: std::time::Duration) {
        self.memory_frees_paced.fetch_add(1, Ordering::Relaxed);
        let nanos = u64::try_from(waited.as_nanos()).unwrap_or(u64::MAX);
        self.memory_frees_paced_nanos
            .fetch_add(nanos, Ordering::Relaxed);
    }

    /// The device let go of a chunk of a guest RAM its thread imported ahead and no object took,
    /// its allocations near the device's limit (`host_ram.rs`, `allocations.rs`).
    pub(crate) fn host_ram_chunk_given_back(&self) {
        (self.host_ram_chunks_given_back).fetch_add(1, Ordering::Relaxed);
    }

    /// The device's thread released a chunk of a guest RAM no object held for its window, on a
    /// device whose held imports add to the cost of every submission (`host_ram.rs`).
    pub(crate) fn host_ram_chunk_released(&self) {
        (self.host_ram_chunks_released).fetch_add(1, Ordering::Relaxed);
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

    /// An import of runs of a guest RAM was made for a copy (`host_ram.rs`).
    pub(crate) fn run_import_made(&self) {
        (self.run_imports_made).fetch_add(1, Ordering::Relaxed);
        (self.run_imports_live).fetch_add(1, Ordering::Relaxed);
    }

    /// An import of runs of a guest RAM went, its last holder gone.
    pub(crate) fn run_import_freed(&self) {
        (self.run_imports_live).fetch_sub(1, Ordering::Relaxed);
    }

    /// A rebind made an image again over a copy of its new pages (`copied.rs`).
    pub(crate) fn copied_rebind_image_made(&self) {
        (self.copied_rebind_images_made).fetch_add(1, Ordering::Relaxed);
    }

    /// A commit copied `bytes` the CPU changed into copied host buffers (`copied.rs`).
    pub(crate) fn copied_copied_in(&self, bytes: u64) {
        (self.copied_copy_in_bytes).fetch_add(bytes, Ordering::Relaxed);
    }

    /// A commit's copy in left `bytes` of copied host buffers uncompared: no log of their pages
    /// reported them written since the copy last synced (`copied.rs`).
    pub(crate) fn copied_copy_in_skipped(&self, bytes: u64) {
        (self.copied_copy_in_bytes_skipped).fetch_add(bytes, Ordering::Relaxed);
    }

    /// A commit's copy in left `copies` copied host buffers alone: no command of its command
    /// buffer used them (`copied.rs`).
    pub(crate) fn copied_copy_ins_unused(&self, copies: u64) {
        (self.copied_copy_ins_unused).fetch_add(copies, Ordering::Relaxed);
    }

    /// A batch whose event wait its commit's copy in came before copied the copied host buffers
    /// in again as it was submitted (`command_buffer/copies_again.rs`).
    pub(crate) fn copied_copy_ins_after_waits(&self) {
        (self.copied_copy_ins_after_waits).fetch_add(1, Ordering::Relaxed);
    }

    /// A batch's completion merged `bytes` the GPU changed back into the caller's pages
    /// (`copied.rs`).
    pub(crate) fn copied_merged_back(&self, bytes: u64) {
        (self.copied_merge_back_bytes).fetch_add(bytes, Ordering::Relaxed);
    }

    /// A completion turn's merge back compared `compared` copied host buffers and left `skipped`
    /// uncompared, which no batch it finished could have written (`copied.rs`).
    pub(crate) fn copied_merges(&self, compared: u64, skipped: u64) {
        (self.copied_merges_compared).fetch_add(compared, Ordering::Relaxed);
        (self.copied_merges_skipped).fetch_add(skipped, Ordering::Relaxed);
    }

    /// The copied host buffers' counts so far, as [`Device::work_counts`] reads them: the bytes
    /// copied in, merged back and dropped, the bases made, refused and their blocks read, and the
    /// threads comparisons started.
    #[cfg(test)]
    pub(crate) fn copied_tallies(&self) -> CopiedTallies {
        let load = |count: &AtomicU64| count.load(Ordering::Relaxed);
        CopiedTallies {
            copy_in_bytes: load(&self.copied_copy_in_bytes),
            merge_back_bytes: load(&self.copied_merge_back_bytes),
            dropped_bytes: load(&self.copied_dropped_bytes),
            bases_made: load(&self.copied_bases_made),
            bases_refused: load(&self.copied_bases_refused),
            base_blocks_read: load(&self.copied_base_blocks_read),
            base_blocks_read_back: load(&self.copied_base_blocks_read_back),
            compare_threads: load(&self.copied_compare_threads),
            unnamed_reach_copy_ins: load(&self.copied_unnamed_reach_copy_ins),
            unnamed_reach_bytes: load(&self.copied_unnamed_reach_bytes_compared),
            unnamed_reach_twin_copy_ins: load(&self.copied_unnamed_reach_twin_copy_ins),
            unnamed_reach_twin_bytes: load(&self.copied_unnamed_reach_twin_bytes_compared),
        }
    }

    /// Commits' copies in made `made` copied host buffers' bases and found no host memory for
    /// `refused`, and comparisons read `read` blocks of bases, each a block both sides changed
    /// (`copied.rs`).
    pub(crate) fn copied_bases(&self, made: u64, refused: u64, read: u64) {
        (self.copied_bases_made).fetch_add(made, Ordering::Relaxed);
        (self.copied_bases_refused).fetch_add(refused, Ordering::Relaxed);
        (self.copied_base_blocks_read).fetch_add(read, Ordering::Relaxed);
    }

    /// Comparisons of copied host buffers read `blocks` blocks of bases in the device's memory
    /// through a copy into host memory on the GPU, one per comparison that read that many
    /// (`copied.rs`): among the blocks [`Counters::copied_bases`] counts read.
    pub(crate) fn copied_base_blocks_read_back(&self, blocks: u64) {
        (self.copied_base_blocks_read_back).fetch_add(blocks, Ordering::Relaxed);
    }

    /// Comparisons of copied host buffers started `threads` threads beside their calling ones,
    /// each splitting a copy's blocks across them (`copied.rs`).
    pub(crate) fn copied_compare_threads(&self, threads: u64) {
        (self.copied_compare_threads).fetch_add(threads, Ordering::Relaxed);
    }

    /// A commit's copy in compared `copies` copied host buffers, `bytes` of them in all (the
    /// bytes no log skipped), which a command of its command buffer used or may have reached
    /// unnamed (`copied.rs`).
    pub(crate) fn copied_copy_ins_compared(&self, copies: u64, bytes: u64) {
        (self.copied_copy_ins_compared).fetch_add(copies, Ordering::Relaxed);
        (self.copied_copy_in_bytes_compared).fetch_add(bytes, Ordering::Relaxed);
    }

    /// A copy in compared `unnamed` (copies, bytes) that its command buffer's reach of memory no
    /// use names alone took, and of them `twinned` that a read twin reads in place where that
    /// reach only reads (`copied.rs`): each counts the copy in once where it compared any.
    pub(crate) fn copied_unnamed_reach(&self, unnamed: (u64, u64), twinned: (u64, u64)) {
        if unnamed.0 > 0 {
            (self.copied_unnamed_reach_copy_ins).fetch_add(1, Ordering::Relaxed);
            (self.copied_unnamed_reach_bytes_compared).fetch_add(unnamed.1, Ordering::Relaxed);
        }
        if twinned.0 > 0 {
            (self.copied_unnamed_reach_twin_copy_ins).fetch_add(1, Ordering::Relaxed);
            let bytes = &self.copied_unnamed_reach_twin_bytes_compared;
            bytes.fetch_add(twinned.1, Ordering::Relaxed);
        }
    }

    /// A read twin was made for a no-copy buffer whose GPU writes are dropped (`read_twin.rs`).
    pub(crate) fn read_twin_made(&self) {
        (self.read_twins_made).fetch_add(1, Ordering::Relaxed);
    }

    /// No read twin was made for a no-copy buffer whose GPU writes are dropped
    /// (`read_twin.rs`): its pages' import would have been a copy, or failed, or the device's
    /// policy contains no access (`uncontained`).
    pub(crate) fn read_twin_not_made(&self, uncontained: bool) {
        match uncontained {
            true => &self.read_twins_uncontained,
            false => &self.read_twins_unimported,
        }
        .fetch_add(1, Ordering::Relaxed);
    }

    /// A blit read a buffer's bytes through its read twin, in place (`read_twin.rs`).
    pub(crate) fn read_twin_blit_read(&self) {
        (self.read_twin_blit_reads).fetch_add(1, Ordering::Relaxed);
    }

    /// A command that may read a buffer through its read twin read its copy instead, as an
    /// earlier command of its command buffer may have written the copy (`read_twin.rs`).
    pub(crate) fn read_twin_read_after_write(&self) {
        (self.read_twin_reads_after_writes).fetch_add(1, Ordering::Relaxed);
    }

    /// An encoder refused a command whose uses write a read twin (`read_twin.rs`): never, but
    /// for a defect of the runtime's.
    pub(crate) fn read_twin_write_refused(&self) {
        (self.read_twin_writes_refused).fetch_add(1, Ordering::Relaxed);
    }

    /// A texture over a no-copy buffer with a read twin took a read twin of its own, in place
    /// over the twin's bytes (`made`), or its twin's texture would not lie in place there
    /// (`read_twin.rs`).
    pub(crate) fn read_twin_texture(&self, made: bool) {
        match made {
            true => &self.read_twin_textures_made,
            false => &self.read_twin_textures_not_in_place,
        }
        .fetch_add(1, Ordering::Relaxed);
    }

    /// Of the textures whose read twin is not in place, one whose twin is kept as a shadow of
    /// the twin's bytes, the texture being a shadow too (`read_twin.rs`).
    pub(crate) fn read_twin_texture_shadow(&self) {
        (self.read_twin_texture_shadows).fetch_add(1, Ordering::Relaxed);
    }

    /// Whether any buffer took a read twin (`read_twin.rs`), a texture's twin among them: else no
    /// command's slots look for one.
    pub(crate) fn read_twins_any(&self) -> bool {
        self.read_twins_made.load(Ordering::Relaxed) > 0
    }

    /// A command's texture or buffer slot read what it binds through a read twin, in place
    /// (`read_twin.rs`).
    pub(crate) fn read_twin_slot_read(&self) {
        (self.read_twin_slot_reads).fetch_add(1, Ordering::Relaxed);
    }

    /// A batch's completion dropped `bytes` the GPU changed in copies of pages the process
    /// cannot write, taking them out of the copies (`copied.rs`).
    pub(crate) fn copied_dropped(&self, bytes: u64) {
        (self.copied_dropped_bytes).fetch_add(bytes, Ordering::Relaxed);
    }

    /// A blit encoder recorded a copy of `bytes` between a guest RAM and a buffer in
    /// `dispatches` dispatches (`host_ram/copy.rs`).
    pub(crate) fn host_ram_copied(&self, bytes: u64, dispatches: u64) {
        (self.host_ram_copies).fetch_add(1, Ordering::Relaxed);
        (self.host_ram_copy_bytes).fetch_add(bytes, Ordering::Relaxed);
        (self.host_ram_copy_dispatches).fetch_add(dispatches, Ordering::Relaxed);
    }

    /// A blit encoder recorded a copy between a guest RAM and a buffer whose runs are not all
    /// whole pages as `regions` regions of `vkCmdCopyBuffer` (`host_ram/copy.rs`).
    pub(crate) fn host_ram_copy_regions(&self, regions: u64) {
        (self.host_ram_copy_regions).fetch_add(regions, Ordering::Relaxed);
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
    #[cfg(unix)]
    pub(crate) fn host_import_fds_keep(&self, count: u64, limit: u64) -> bool {
        let kept = &self.host_import_fds_kept;
        let fits = |held: u64| held.checked_add(count).filter(|after| *after <= limit);
        kept.try_update(Ordering::AcqRel, Ordering::Acquire, fits)
            .is_ok()
    }

    /// `count` such descriptors were closed.
    #[cfg(unix)]
    pub(crate) fn host_import_fds_closed(&self, count: u64) {
        self.host_import_fds_kept
            .fetch_sub(count, Ordering::Relaxed);
    }

    /// A no-copy buffer's own import of dma-bufs is kept imported from now on, as the
    /// descriptors to import it again did not fit (`kept`), or such an import went.
    #[cfg(unix)]
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
    #[cfg(unix)]
    pub(crate) fn dma_buf_sparse_buffer_made(&self) {
        self.dma_buf_sparse_buffers.fetch_add(1, Ordering::Relaxed);
    }

    /// A sparse buffer over several chunks' imports was bound, `binds` of them, one
    /// `VkSparseMemoryBind` each (`dma_buf.rs`).
    #[cfg(unix)]
    pub(crate) fn dma_buf_sparse_bound(&self, binds: usize) {
        self.dma_buf_sparse_binds
            .fetch_add(binds as u64, Ordering::Relaxed);
    }

    /// A texture's image over dma-buf pages was made and bound to an import of a dma-buf of its
    /// own pages (`dma_buf.rs`).
    #[cfg(unix)]
    pub(crate) fn dma_buf_image_made(&self) {
        self.dma_buf_images.fetch_add(1, Ordering::Relaxed);
    }

    /// Of those, one whose plane lies at offset 0, bound at its texels' offset of an import that
    /// is no dedicated allocation (`dma_buf.rs`'s `offset_bound_image`).
    #[cfg(unix)]
    pub(crate) fn dma_buf_image_bound_at_offset(&self) {
        self.dma_buf_images_at_offset
            .fetch_add(1, Ordering::Relaxed);
    }

    /// An image in place over a no-copy buffer's dma-buf pages was parked under the pages'
    /// import as its texture went (`dma_buf/parked.rs`).
    #[cfg(unix)]
    pub(crate) fn dma_buf_image_parked(&self) {
        self.dma_buf_images_parked.fetch_add(1, Ordering::Relaxed);
    }

    /// A texture in place over a no-copy buffer's dma-buf pages took a parked image of its plan
    /// at its texels (`dma_buf/parked.rs`), asking no source and importing nothing.
    #[cfg(unix)]
    pub(crate) fn dma_buf_image_taken(&self) {
        self.dma_buf_images_taken.fetch_add(1, Ordering::Relaxed);
    }

    /// A texture in place over a no-copy buffer's dma-buf pages found no parked image of its plan
    /// at its texels, and its image was made and imported anew (`dma_buf/parked.rs`).
    #[cfg(unix)]
    pub(crate) fn dma_buf_image_missed(&self) {
        self.dma_buf_images_missed.fetch_add(1, Ordering::Relaxed);
    }

    /// `count` parked images were released untaken (`dma_buf/parked.rs`).
    #[cfg(unix)]
    pub(crate) fn dma_buf_images_parked_released(&self, count: usize) {
        (self.dma_buf_images_parked_released).fetch_add(count as u64, Ordering::Relaxed);
    }

    /// A texture in place over a no-copy buffer's dma-buf pages had its image bound in the
    /// import the buffer lies in, asking no source and importing nothing
    /// (`texture/in_place_image.rs`).
    #[cfg(unix)]
    pub(crate) fn dma_buf_image_in_import(&self) {
        (self.dma_buf_images_in_import).fetch_add(1, Ordering::Relaxed);
    }

    /// The device made an image of a texture over a buffer of its own memory exported as a
    /// dma-buf bound to a dedicated import of that memory's own dma-buf (`dma_buf.rs`'s
    /// `bind_exported_dma_buf_image`).
    #[cfg(unix)]
    pub(crate) fn exported_dma_buf_image_made(&self) {
        (self.exported_dma_buf_images).fetch_add(1, Ordering::Relaxed);
    }

    /// Such an image was parked under its buffer's pages as its texture went
    /// (`dma_buf/parked.rs`).
    #[cfg(unix)]
    pub(crate) fn exported_dma_buf_image_parked(&self) {
        (self.exported_dma_buf_images_parked).fetch_add(1, Ordering::Relaxed);
    }

    /// A texture over a buffer of the device's exported memory took a parked image of its plan at
    /// its texels (`dma_buf/parked.rs`), exporting and importing nothing.
    #[cfg(unix)]
    pub(crate) fn exported_dma_buf_image_taken(&self) {
        (self.exported_dma_buf_images_taken).fetch_add(1, Ordering::Relaxed);
    }

    /// `count` parked images over the device's exported memory were released untaken
    /// (`dma_buf/parked.rs`).
    #[cfg(unix)]
    pub(crate) fn exported_dma_buf_images_parked_released(&self, count: usize) {
        (self.exported_dma_buf_images_parked_released).fetch_add(count as u64, Ordering::Relaxed);
    }

    /// A texture placed in a heap over dma-buf pages that such an image would alias is a shadow of
    /// the heap's bytes instead: the device made no such image of it (`texture/heap.rs`).
    pub(crate) fn dma_buf_image_fell_back(&self) {
        self.dma_buf_image_fallbacks.fetch_add(1, Ordering::Relaxed);
    }

    /// The driver was asked a format's `DRM_FORMAT_MOD_LINEAR` tiling features (`dma_buf.rs`'s
    /// `modifier_features`).
    pub(crate) fn modifier_features_asked(&self) {
        self.modifier_feature_queries
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A pixel format's linear texture alignment was probed of the driver
    /// (`texture/linear_alignment.rs`).
    pub(crate) fn linear_alignment_probed(&self) {
        (self.linear_alignments_probed).fetch_add(1, Ordering::Relaxed);
    }

    /// `formats` pixel formats' linear texture alignments were read from the record of an earlier
    /// probe (`texture/linear_alignment.rs`).
    pub(crate) fn linear_alignments_recalled(&self, formats: u64) {
        (self.linear_alignments_recalled).fetch_add(formats, Ordering::Relaxed);
    }

    /// A texture over a buffer's bytes was made a shadow of them (`shadow.rs`), its image started
    /// as a new texture's where `zeroed` (its copies leave an aspect or levels out).
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

    /// A texture over a buffer's bytes whose rows are in place was made with its shadow left
    /// unmade (`texture/lazy_shadow.rs`).
    pub(crate) fn shadow_left_unmade(&self) {
        (self.shadows_left_unmade).fetch_add(1, Ordering::Relaxed);
    }

    /// A shadow left unmade was made at the first reach of its texture's image
    /// (`texture/lazy_shadow.rs`).
    pub(crate) fn shadow_made_at_use(&self) {
        (self.shadows_made_at_use).fetch_add(1, Ordering::Relaxed);
    }

    /// A texture over host pages was made a device copy of them (`texture.rs`).
    pub(crate) fn device_copy_texture(&self) {
        self.device_copy_textures.fetch_add(1, Ordering::Relaxed);
    }

    /// A two-copy buffer's device copy of `bytes` bytes was made (`made`), or went
    /// (`managed_copy.rs`).
    pub(crate) fn managed_device_copy(&self, bytes: u64, made: bool) {
        match made {
            true => {
                self.managed_device_copies.fetch_add(1, Ordering::Relaxed);
                (self.managed_device_copies_made).fetch_add(1, Ordering::Relaxed);
                (self.managed_device_copy_bytes).fetch_add(bytes, Ordering::Relaxed);
            }
            false => {
                self.managed_device_copies.fetch_sub(1, Ordering::Relaxed);
                (self.managed_device_copy_bytes).fetch_sub(bytes, Ordering::Relaxed);
            }
        }
    }

    /// A `Managed` no-copy buffer that would be two copies lies in its pages: its device copy
    /// found no room (`managed_copy.rs`).
    pub(crate) fn managed_device_copy_in_place(&self) {
        (self.managed_device_copies_in_place).fetch_add(1, Ordering::Relaxed);
    }

    /// A two-copy buffer's device copy was made with `bytes` bytes the caller promised read zero
    /// set to zero by the device rather than copied in (`managed_copy.rs`,
    /// `HostMemory::with_zero_from`).
    pub(crate) fn managed_device_copy_zeroed(&self, bytes: u64) {
        (self.managed_device_copies_zeroed).fetch_add(1, Ordering::Relaxed);
        (self.managed_bytes_zeroed).fetch_add(bytes, Ordering::Relaxed);
    }

    /// `bytes` bytes were copied into a two-copy buffer's device copy (`copied_in`), or out of
    /// it into its pages (`managed_copy.rs`).
    pub(crate) fn managed_copied(&self, bytes: u64, copied_in: bool) {
        match copied_in {
            true => self
                .managed_bytes_copied_in
                .fetch_add(bytes, Ordering::Relaxed),
            false => self
                .managed_bytes_copied_out
                .fetch_add(bytes, Ordering::Relaxed),
        };
    }

    /// The program asked a two-copy buffer (`managed_copy.rs`) to copy a range of `bytes` bytes:
    /// in, by `did_modify_range` (`modified`), or out, by `synchronize_resource` or
    /// `synchronize_texture`.
    pub(crate) fn managed_asked(&self, bytes: u64, modified: bool) {
        let (calls, asked) = match modified {
            true => (&self.managed_ranges_modified, &self.managed_bytes_modified),
            false => (
                &self.managed_resources_synchronized,
                &self.managed_bytes_synchronized,
            ),
        };
        calls.fetch_add(1, Ordering::Relaxed);
        asked.fetch_add(bytes, Ordering::Relaxed);
    }

    /// A two-copy buffer's copy out was decided (`managed_sync.rs`): `skipped` bytes of its range
    /// that no GPU write since its last copy out reached go uncopied, it copies nothing where
    /// `unwritten`, and its range whole where `whole`, the GPU's writes not known.
    pub(crate) fn managed_synchronize_decided(&self, skipped: u64, unwritten: bool, whole: bool) {
        (self.managed_bytes_unwritten_skipped).fetch_add(skipped, Ordering::Relaxed);
        if unwritten {
            (self.managed_synchronizes_unwritten).fetch_add(1, Ordering::Relaxed);
        }
        if whole {
            (self.managed_synchronizes_whole).fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A two-copy buffer's copy out took, at its batch's submission, writes submitted since its
    /// encoding (`managed_sync.rs`).
    pub(crate) fn managed_synchronize_topped_up(&self) {
        (self.managed_synchronizes_topped_up).fetch_add(1, Ordering::Relaxed);
    }

    /// A staging buffer of `bytes` bytes of a two-copy buffer's copies was made (`made`), or
    /// went (`managed_copy.rs`).
    pub(crate) fn managed_staging(&self, bytes: u64, made: bool) {
        match made {
            true => self
                .managed_staging_bytes
                .fetch_add(bytes, Ordering::Relaxed),
            false => self
                .managed_staging_bytes
                .fetch_sub(bytes, Ordering::Relaxed),
        };
    }

    /// A command buffer that used a two-copy buffer whose pages drop the GPU's writes completed,
    /// and its pages were copied in again (`managed_copy.rs`).
    pub(crate) fn managed_write_undone(&self) {
        self.managed_writes_undone.fetch_add(1, Ordering::Relaxed);
    }

    /// The CPU copied `bytes` bytes of a two-copy buffer's host copy into a staging buffer for a
    /// copy in (`managed_copy.rs`'s `stage_in`).
    pub(crate) fn managed_staged_in(&self, bytes: u64) {
        (self.managed_bytes_staged_in).fetch_add(bytes, Ordering::Relaxed);
    }

    /// The GPU's copy of `bytes` bytes of a two-copy buffer's pages into its device copy, from a
    /// guest RAM's chunk imports, was registered (`managed_copy.rs`).
    pub(crate) fn managed_copied_in_from_ram(&self, bytes: u64) {
        (self.managed_bytes_copied_in_from_ram).fetch_add(bytes, Ordering::Relaxed);
    }

    /// A range `did_modify_range` named of a `Managed` buffer with a host copy was written into
    /// its device copy inline, as a submission was recorded (`texture/first_move.rs`).
    pub(crate) fn managed_range_inlined(&self) {
        (self.managed_ranges_inlined).fetch_add(1, Ordering::Relaxed);
    }

    /// A batch's first work, uploads alone into `Managed` buffers' device copies, was recorded
    /// with no barrier of its own from the commands before it: the end of the batch before it on
    /// its `VkQueue` orders them (`texture/first_move.rs`).
    pub(crate) fn managed_upload_unbarriered(&self) {
        (self.managed_uploads_unbarriered).fetch_add(1, Ordering::Relaxed);
    }

    /// A batch's first work, uploads alone into `Managed` buffers' device copies, was made
    /// visible by a barrier to its own commands' uses of those buffers alone
    /// (`texture/first_move.rs`'s `Later::Batch`).
    pub(crate) fn managed_upload_barrier_narrowed(&self) {
        (self.managed_upload_barriers_narrowed).fetch_add(1, Ordering::Relaxed);
    }

    /// A two-copy buffer's host copy was made its pages alone, no import of their own mapping
    /// them, for `why` (`managed_copy.rs`).
    pub(crate) fn managed_pages_alone(&self, why: crate::managed_copy::AloneBecause) {
        use crate::managed_copy::AloneBecause;
        let count = match why {
            AloneBecause::Refused => &self.managed_pages_alone_refused,
            AloneBecause::OverImports => &self.managed_pages_alone_over_imports,
            AloneBecause::Unwritable => &self.managed_pages_alone_unwritable,
        };
        count.fetch_add(1, Ordering::Relaxed);
    }

    /// Such pages alone took the pieces of a guest RAM's chunk imports that hold them all, which
    /// their copies in read (`managed_copy.rs`).
    pub(crate) fn managed_pages_in_ram_chunks(&self) {
        (self.managed_pages_in_ram_chunks).fetch_add(1, Ordering::Relaxed);
    }

    /// A texture over host pages of `class` was made an image in place over them (`in_place.rs`),
    /// its rows at Metal's pitch in a `DRM_FORMAT_MOD_LINEAR` image where `pitched`; over a
    /// copied host buffer's copy of them where `copied` (`copied.rs`), whose class is the copy's
    /// memory's.
    pub(crate) fn host_texture_in_place(
        &self,
        class: crate::in_place::PagesClass,
        (pitched, written, copied): (bool, bool, bool),
    ) {
        use crate::in_place::PagesClass as P;
        self.host_textures_in_place.fetch_add(1, Ordering::Relaxed);
        if written {
            (self.host_textures_in_place_written).fetch_add(1, Ordering::Relaxed);
        }
        if copied {
            (self.copied_textures_in_place).fetch_add(1, Ordering::Relaxed);
            return;
        }
        match class {
            P::Pointer => {}
            P::DmaBuf | P::DmaBufInImport => {
                self.dma_buf_textures_in_place
                    .fetch_add(1, Ordering::Relaxed);
            }
            P::Fresh => {
                (self.fresh_page_textures_in_place).fetch_add(1, Ordering::Relaxed);
            }
            // A texture over the runtime's pooled or exported memory is no texture over host
            // pages: its image at Metal's pitch is counted apart (`pooled_pitched_image`).
            P::Pooled | P::ExportedFd | P::ExportedDmaBuf | P::FdBlocks => {}
        }
        if pitched {
            self.pitched_host_images.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A texture over host pages was made a shadow of them for `cause` (`in_place.rs`), one whose
    /// usage renders into it where `renders`, one a shader writes where `written`.
    pub(crate) fn host_texture_shadowed(
        &self,
        cause: &crate::in_place::ShadowCause,
        (renders, written): (bool, bool),
    ) {
        use crate::in_place::ShadowCause as C;
        let counter = match cause {
            C::Exact => &self.host_texture_shadows_exact,
            C::StoreWidth(_) => &self.host_texture_shadows_store_width,
            C::CheckPending(..) => &self.host_texture_shadows_check_pending,
            C::CheckFailed(..) => &self.host_texture_shadows_check_failed,
            C::Unchecked(..) => &self.host_texture_shadows_unchecked,
            C::Refused(refusal, _) => &self.host_texture_shadows_refused[refusal.index()],
        };
        counter.fetch_add(1, Ordering::Relaxed);
        if renders {
            (self.host_texture_shadows_render_target).fetch_add(1, Ordering::Relaxed);
        }
        if written {
            (self.host_texture_shadows_shader_write).fetch_add(1, Ordering::Relaxed);
        }
    }

    /// The device's check over host pages answered `verdict` for a class and access it ran
    /// (`in_place_check.rs`).
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

    /// A shadow over host pages made before its pages' in-place verdicts moved in place once
    /// they passed (`texture/storage_form.rs`).
    pub(crate) fn host_texture_moved_in_place(&self) {
        (self.host_textures_moved_in_place).fetch_add(1, Ordering::Relaxed);
    }

    /// The shadow image of a texture that moved in place went once the command buffer that took
    /// the move completed (`texture/storage_form.rs`).
    pub(crate) fn host_texture_shadow_freed_after_move(&self) {
        (self.host_texture_shadows_freed_after_move).fetch_add(1, Ordering::Relaxed);
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

    /// A command buffer's first copy in of a shadow that may stand across command buffers was
    /// decided as its batch was submitted (`shadow_standing.rs`): it moved nothing where the
    /// image stood (`skipped`), its `texels` with it.
    pub(crate) fn shadow_copy_in_against_standing(&self, skipped: bool, texels: u64) {
        (self.shadow_copies_in_against_standing).fetch_add(1, Ordering::Relaxed);
        if skipped {
            (self.shadow_copies_in_skipped_by_standing).fetch_add(1, Ordering::Relaxed);
            (self.shadow_texels_skipped_by_standing).fetch_add(texels, Ordering::Relaxed);
        }
    }

    /// A command buffer's first copy in of a shadow whose truth only the runtime's commands write
    /// moved its `texels`, decided `at`; of a shadow that may stand, its standing let go as
    /// `let_go` says, `None` where it stood, and of one that never stood, what last kept a
    /// command buffer's end from leaving it standing, `missed` (`shadow_standing.rs`).
    pub(crate) fn shadow_copy_in_unskipped(
        &self,
        at: Unskipped,
        let_go: Option<LetGo>,
        missed: Missed,
        texels: u64,
    ) {
        (self.shadow_texels_unskipped[at as usize]).fetch_add(texels, Ordering::Relaxed);
        if at == Unskipped::Unstanding {
            return;
        }
        // Where it stood first, then each cause in order.
        let standing = let_go.map_or(0, |let_go| let_go as usize + 1);
        (self.shadow_texels_unskipped_standing[standing]).fetch_add(texels, Ordering::Relaxed);
        if let_go == Some(LetGo::NeverStood) {
            (self.shadow_texels_never_stood[missed as usize]).fetch_add(texels, Ordering::Relaxed);
        }
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
            (true, Backing::Image | Backing::PrivateImage) => {
                &self.shadow_texels_first_copied_in_images
            }
            (true, Backing::PrivateBuffer) => &self.shadow_texels_first_copied_in_private_buffers,
            (true, Backing::MappedBuffer) => &self.shadow_texels_first_copied_in_mapped_buffers,
            (true, Backing::HostPages) => &self.shadow_texels_first_copied_in_host_pages,
            (true, Backing::AliasedHostPages) => {
                &self.shadow_texels_first_copied_in_aliased_host_pages
            }
            (true, Backing::Surface | Backing::PrivateSurface) => {
                &self.shadow_texels_first_copied_in_surfaces
            }
            (false, Backing::Image | Backing::PrivateImage) => {
                &self.shadow_texels_recopied_in_images
            }
            (false, Backing::PrivateBuffer) => &self.shadow_texels_recopied_in_private_buffers,
            (false, Backing::MappedBuffer) => &self.shadow_texels_recopied_in_mapped_buffers,
            (false, Backing::HostPages) => &self.shadow_texels_recopied_in_host_pages,
            (false, Backing::AliasedHostPages) => {
                &self.shadow_texels_recopied_in_aliased_host_pages
            }
            (false, Backing::Surface | Backing::PrivateSurface) => {
                &self.shadow_texels_recopied_in_surfaces
            }
        });
        if first && copies.backing == Backing::PrivateSurface {
            add(&self.shadow_texels_first_copied_in_private_surfaces);
        }
        if first && copies.backing == Backing::PrivateImage {
            add(&self.shadow_texels_first_copied_in_private_images);
        }
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

    /// An encoder recorded a shadowed texture's copy back spliced (`shadow_splice.rs`).
    pub(crate) fn shadow_copy_back_spliced(&self) {
        self.shadow_copies_back_spliced
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A host step stored the bytes `merges` merges on the host changed, comparing `bytes`, in
    /// `nanos` of its thread (`shadow_splice.rs`).
    pub(crate) fn shadow_merged_on_host(&self, merges: u64, nanos: u64, bytes: u64) {
        self.shadow_merges_on_host
            .fetch_add(merges, Ordering::Relaxed);
        self.shadow_merge_host_nanos
            .fetch_add(nanos, Ordering::Relaxed);
        self.shadow_merge_bytes_on_host
            .fetch_add(bytes, Ordering::Relaxed);
    }

    /// A host step ran for an encoder's copies back spliced and merges on the host
    /// (`shadow_splice.rs`).
    pub(crate) fn shadow_host_step(&self) {
        self.shadow_host_steps.fetch_add(1, Ordering::Relaxed);
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

    /// A merge of a shadowed texture took its scratch from its command buffer's pool or its
    /// shadow's spares (`merge_scratch.rs`, `shadow_merge.rs`'s `ScratchSpares`).
    pub(crate) fn shadow_merge_scratch_reused(&self) {
        self.shadow_merge_scratches_reused
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A merge of a shadowed texture's scratch holds zero before each keep (`shadow_merge.rs`):
    /// one not cut.
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
    /// needed it, or ahead on the device's pool (a draw replay's, `owned_kernel.rs`): one a host
    /// step built among them (`host_step_owned_kernels_made`).
    pub(crate) fn owned_kernel_made(&self, spirv: usize) {
        self.owned_kernels_made.fetch_add(1, Ordering::Relaxed);
        if crate::exit_guard::in_host_step() {
            self.host_step_owned_kernels_made
                .fetch_add(1, Ordering::Relaxed);
        }
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

    /// An `A8Unorm` texture over a buffer's bytes was made as an `R8Unorm` texture's storage,
    /// read through a swizzle (`texture/red_held_alpha.rs`).
    pub(crate) fn alpha_texture_held_in_red(&self) {
        self.alpha_textures_held_in_red
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A texture view's shadow image was made (`texture/view_shadow.rs`): the view's own, or
    /// the render shadow of a view its bindings present through a swizzle.
    pub(crate) fn view_shadow_made(&self) {
        self.view_shadows.fetch_add(1, Ordering::Relaxed);
    }

    /// A draw that samples a color attachment's texture after a memory barrier of its render
    /// pass recorded into a new instance (`texture/attached_reads.rs`).
    pub(crate) fn attachment_read_split(&self) {
        self.attachment_read_splits.fetch_add(1, Ordering::Relaxed);
    }

    /// A draw that samples a color attachment's texture after a memory barrier inside its render
    /// pass instance, which drew none of the texels it reads, stayed in the instance
    /// (`texture/attached_reads.rs`).
    pub(crate) fn attachment_read_split_spared(&self) {
        self.attachment_read_splits_spared
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A draw that samples a color its render pass clears, after a memory barrier before the
    /// pass's first draw, read copies cleared to the clear in its instance instead of a split
    /// (`texture/attached_reads.rs`).
    pub(crate) fn attachment_read_cleared(&self) {
        self.attachment_read_clears.fetch_add(1, Ordering::Relaxed);
    }

    /// A draw recorded after a barrier from the earlier draws of its instance because it reads
    /// input attachments ordered between draws alone after one wrote a color, or writes a color
    /// after one read them (`render.rs`'s `InterlockedStorage`, `framebuffer_fetch.rs`).
    pub(crate) fn fetch_draw_barrier(&self) {
        self.fetch_draw_barriers.fetch_add(1, Ordering::Relaxed);
    }

    /// A draw that samples a color attachment's texture after a memory barrier of its render
    /// pass read it through a snapshot inside its instance (`render/snapshot.rs`).
    pub(crate) fn reading_draw_snapshotted(&self) {
        self.reading_draws_snapshotted
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Such a draw took its snapshots in a render pass instance of their own that leads its
    /// instance (`render/snapshot.rs`, leading snapshots).
    pub(crate) fn reading_draw_snapshotted_leading(&self) {
        (self.reading_draws_snapshotted_leading).fetch_add(1, Ordering::Relaxed);
    }

    /// A draw that samples a color attachment's texture at its own pixel, writing nothing of
    /// that color, read it in place inside its instance (`render/snapshot.rs`).
    pub(crate) fn reading_draw_in_place(&self) {
        self.reading_draws_in_place.fetch_add(1, Ordering::Relaxed);
    }

    /// A draw read through a snapshot, or in place, recorded into a new instance, as its own was
    /// recorded inline (`render/snapshot.rs`).
    pub(crate) fn snapshot_inline_split(&self) {
        self.snapshot_inline_splits.fetch_add(1, Ordering::Relaxed);
    }

    /// A draw that reads its pass's texture after a barrier, through a slot its function reads
    /// at its own pixels, took the split, for `reason` (`render/snapshot.rs`).
    pub(crate) fn reading_draw_split(&self, reason: SplitReason) {
        self.reading_draws_split[reason as usize].fetch_add(1, Ordering::Relaxed);
    }

    /// Of the draws split for `SplitReason::UnreadableOwnPixel`, one whose slots no snapshot
    /// serves are left out by the clauses of `clauses` (`Unreadable`): counted once in each.
    pub(crate) fn reading_draw_unreadable(&self, clauses: Unreadable) {
        for (count, i) in self.reading_draws_unreadable.iter().zip(0..) {
            if clauses.0 & (1 << i) != 0 {
                count.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// A reader of its pass's color was weighed by what it covered when a pass last measured it
    /// (`render/coverage.rs`).
    pub(crate) fn reading_draw_by_coverage(&self) {
        self.reading_draws_by_coverage
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Such a reader, whose snapshots would lead its instance, covered `share` of its colors'
    /// pixels (`render/snapshot.rs`, leading snapshots).
    pub(crate) fn reading_draw_leading_covered(&self, share: LeadingShare) {
        self.reading_draws_leading_covered[share as usize].fetch_add(1, Ordering::Relaxed);
    }

    /// A reader of its pass's color recorded its drawing commands inside an occlusion query
    /// that measures what it covers (`render/coverage.rs`).
    pub(crate) fn coverage_query(&self) {
        self.coverage_queries.fetch_add(1, Ordering::Relaxed);
    }

    /// A render pass that measures its readers made an occlusion query pool, the device having
    /// no spare (`render/coverage.rs`).
    pub(crate) fn coverage_pool_made(&self) {
        self.coverage_pools_made.fetch_add(1, Ordering::Relaxed);
    }

    /// A draw read through a snapshot a color whose image a later fetch would render through a
    /// stand-in, its instance pinned to the color's own image (`render/snapshot.rs`).
    pub(crate) fn reading_draw_pinned(&self) {
        self.reading_draws_pinned.fetch_add(1, Ordering::Relaxed);
    }

    /// A pass's stand-ins arrived after its pinned instances, and took the attachments' texels
    /// after the last of them (`render/snapshot.rs`).
    pub(crate) fn stand_ins_after_pin(&self) {
        self.stand_ins_after_pins.fetch_add(1, Ordering::Relaxed);
    }

    /// The device made a linear image aliasing a buffer's bytes with feedback loop usage
    /// (`Device::storage_over_buffer`, `texture::feedback_candidate`).
    pub(crate) fn feedback_loop_alias_image(&self) {
        self.feedback_loop_alias_images
            .fetch_add(1, Ordering::Relaxed);
    }

    /// The device made a linear image aliasing a buffer's bytes that holds the bytes between the
    /// texture's rows or slices, which the buffer declared unobserved
    /// (`BufferDescriptor::padding_unobserved`, `Device::storage_over_buffer`).
    pub(crate) fn unobserved_padding_alias(&self) {
        self.unobserved_padding_aliases
            .fetch_add(1, Ordering::Relaxed);
    }

    /// The device made its linear image, `PREINITIALIZED`, or a `DRM_FORMAT_MOD_LINEAR` one whose
    /// first move keeps them by copies, aliasing a buffer's bytes that holds the bytes between the
    /// texture's rows or slices, which only the device's ordered commands write
    /// (`linear_plan::Between::Kept`, `Device::storage_over_buffer`).
    pub(crate) fn kept_padding_alias(&self) {
        self.kept_padding_aliases.fetch_add(1, Ordering::Relaxed);
    }

    /// A first move kept `bytes` of a buffer by copies around its layout transition
    /// (`texture/first_move.rs`'s `GpuKeep`).
    pub(crate) fn first_move_kept(&self, bytes: u64) {
        self.first_moves_kept.fetch_add(1, Ordering::Relaxed);
        self.first_move_bytes_kept
            .fetch_add(bytes, Ordering::Relaxed);
    }

    /// The device made a `DRM_FORMAT_MOD_LINEAR` image at Metal's pitch aliasing a buffer's bytes
    /// in its pooled memory, where its own linear image lays them out otherwise
    /// (`Device::pooled_pitched_image`).
    pub(crate) fn pooled_pitched_image(&self) {
        self.pooled_pitched_images.fetch_add(1, Ordering::Relaxed);
    }

    /// A shadow over a buffer of the device's pooled memory made before that class's in-place
    /// verdicts moved to its image at Metal's pitch once they passed
    /// (`texture/storage_form.rs`).
    pub(crate) fn pooled_texture_moved_in_place(&self) {
        (self.pooled_textures_moved_in_place).fetch_add(1, Ordering::Relaxed);
    }

    /// The shadow image of a texture over the device's pooled memory that moved in place went
    /// once the command buffer that took the move completed (`texture/storage_form.rs`).
    pub(crate) fn pooled_texture_shadow_freed_after_move(&self) {
        (self.pooled_texture_shadows_freed_after_move).fetch_add(1, Ordering::Relaxed);
    }

    /// The device made the image of a texture over a buffer's bytes that the encoders copy them
    /// into and back (`shadow.rs`) with feedback loop usage (`texture::feedback_candidate`).
    pub(crate) fn feedback_loop_shadow_image(&self) {
        self.feedback_loop_shadow_images
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

    /// An encoder's call raced a rebind of a buffer over host pages it held or pinned
    /// (`rebind.rs`, **Other threads**): its commands run only encoded again.
    pub(crate) fn encoder_raced_rebind(&self) {
        self.encoders_raced_rebind.fetch_add(1, Ordering::Relaxed);
    }

    /// `count` encoders a call of which raced a rebind committed, the runtime unable to encode
    /// them again: each fails `EncodedUseRebound` (`rebind.rs`, **Other threads**).
    pub(crate) fn encoders_raced_unreplayable(&self, count: usize) {
        (self.encoders_raced_unreplayable).fetch_add(count as u64, Ordering::Relaxed);
    }

    /// A rebind was refused (`EncodedUses`) for a gated encoder call still under way past its
    /// bound (`rebind::CallGate`).
    pub(crate) fn rebind_refused_calls_in_flight(&self) {
        (self.rebinds_refused_calls_in_flight).fetch_add(1, Ordering::Relaxed);
    }

    /// A rebind was refused (`EncodedUses`) for a committed command buffer not yet submitted whole
    /// whose encoder no rebind encodes again named the buffer (`Holds::hold_to_submission`).
    pub(crate) fn rebind_refused_unsubmitted(&self) {
        (self.rebinds_refused_unsubmitted).fetch_add(1, Ordering::Relaxed);
    }

    /// A gated encoder call waited, as it began, for a rebind's publish to close
    /// (`rebind::CallGate`).
    pub(crate) fn encoder_call_held_by_rebind(&self) {
        (self.encoder_calls_held_by_rebind).fetch_add(1, Ordering::Relaxed);
    }

    /// A new image was moved to `GENERAL` and zeroed on the host (`texture/host_copy.rs`).
    pub(crate) fn image_started_on_the_host(&self) {
        self.images_started_on_the_host
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Of those, one started in the driver's zeros, in bytes no resource used, moved out of
    /// `VK_IMAGE_LAYOUT_ZERO_INITIALIZED_EXT` with no write of the host's (`texture/host_copy.rs`).
    pub(crate) fn image_started_zeroed(&self) {
        self.images_started_zeroed.fetch_add(1, Ordering::Relaxed);
    }

    /// A new `Private` texture of the program's was made zero-filled, on a device presenting a
    /// Mac that keeps no texture compressed (`Device::metal_compresses_textures`).
    pub(crate) fn private_texture_zero_filled(&self) {
        (self.private_textures_zero_filled).fetch_add(1, Ordering::Relaxed);
    }

    /// A new zero-filled `Private` texture's first move went without its zeros: the first use
    /// among its submission's commands writes every texel before any command reads one
    /// (`texture/first_move.rs`, `FirstMoves::take_holding`).
    pub(crate) fn private_texture_overwritten_unfilled(&self) {
        (self.private_textures_overwritten_unfilled).fetch_add(1, Ordering::Relaxed);
    }

    /// A new memoryless texture in memory that is not lazily allocated was made to be zeroed at
    /// its first move, by render passes (`Device::new_texture_image`).
    pub(crate) fn memoryless_texture_zeroed(&self) {
        (self.memoryless_textures_zeroed).fetch_add(1, Ordering::Relaxed);
    }

    /// A texture's CPU copy ran on the host (`texture/host_copy.rs`).
    pub(crate) fn host_texel_copy(&self) {
        self.host_texel_copies.fetch_add(1, Ordering::Relaxed);
    }

    /// A heap shadow's CPU copy ran in the heap's bytes (`texture/heap.rs`, `HeapShadow::pieces`).
    pub(crate) fn heap_shadow_cpu_copy(&self) {
        self.heap_shadow_cpu_copies.fetch_add(1, Ordering::Relaxed);
    }

    /// An upload's 32-bit float depths were clamped into `[0, 1]` before a copy into a depth
    /// image (`depth_clamp.rs`).
    pub(crate) fn depth_upload_clamped(&self) {
        self.depth_uploads_clamped.fetch_add(1, Ordering::Relaxed);
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

    /// A texture took its two faces at its first attached read (`texture/faces.rs`,
    /// `WorkCounts::texture_faces_made`).
    pub(crate) fn texture_faces_made(&self) {
        self.texture_faces_made.fetch_add(1, Ordering::Relaxed);
    }

    /// A texture turned to its other face at a pass's open (`WorkCounts::attached_read_swaps`).
    pub(crate) fn attached_read_swap(&self) {
        self.attached_read_swaps.fetch_add(1, Ordering::Relaxed);
    }

    /// A turned pass's first instance read the face from before it, `texels` of it, with no copy
    /// in (`WorkCounts::attached_read_swap_reads`).
    pub(crate) fn attached_read_swap_read(&self, texels: u64) {
        self.attached_read_swap_reads
            .fetch_add(1, Ordering::Relaxed);
        (self.attached_read_swap_texels).fetch_add(texels, Ordering::Relaxed);
    }

    /// A batch's submission copied `texels` between a texture's faces for a turn whose pass had
    /// not run (`WorkCounts::attached_read_swap_fixes`).
    pub(crate) fn attached_read_swap_fix(&self, texels: u64) {
        self.attached_read_swap_fixes
            .fetch_add(1, Ordering::Relaxed);
        (self.attached_read_swap_fix_texels).fetch_add(texels, Ordering::Relaxed);
    }

    /// A turn for a pass whose render area is short of its texture copied `texels` outside what
    /// the color's clear rewrites into the face turned to
    /// (`WorkCounts::attached_read_swaps_short_area`).
    pub(crate) fn attached_read_swap_short_area(&self, texels: u64) {
        (self.attached_read_swaps_short_area).fetch_add(1, Ordering::Relaxed);
        (self.attached_read_swap_area_texels).fetch_add(texels, Ordering::Relaxed);
    }

    /// A pass's copy in of `texels` of an attached read's shadow that no turn took off, for
    /// `cause` (`WorkCounts::attached_read_unswapped_load` and kin).
    pub(crate) fn attached_read_unswapped(&self, cause: Unturned, texels: u64) {
        self.attached_read_unswapped[cause as usize].fetch_add(1, Ordering::Relaxed);
        self.attached_read_unswapped_texels[cause as usize].fetch_add(texels, Ordering::Relaxed);
    }

    /// A write of texture slots waited for a batch's hold of them to end (`heap_sets::hold`).
    pub(crate) fn heap_slot_writes_deferred(&self) {
        (self.heap_slot_writes_deferred).fetch_add(1, Ordering::Relaxed);
    }

    /// A new heap set's image array took its elements from a live set's array of its class
    /// (`heap_sets::DeviceHeaps::copy_arrays`).
    pub(crate) fn heap_array_copied(&self) {
        (self.heap_arrays_copied).fetch_add(1, Ordering::Relaxed);
    }

    /// A heap's texture was made tiled as Metal lays it out (`texture/heap_tiled.rs`).
    pub(crate) fn heap_texture_tiled(&self) {
        self.heap_textures_tiled.fetch_add(1, Ordering::Relaxed);
    }

    /// A heap's texture was made in place as an image that fits Metal's answer, where the image
    /// of the route the device first took did not (`texture/heap.rs`).
    pub(crate) fn heap_texture_refit(&self) {
        self.heap_textures_refit.fetch_add(1, Ordering::Relaxed);
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

/// Which draw through its region-copy variant a recording is of
/// (`WorkCounts::rate_map_whole_copy_vertex_regions`): one into every region of the table, the
/// encoder's own where `own`; a run of a flagged draw's blocks; or a boxed draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CopyRoute {
    Whole { own: bool },
    Run,
    Boxed,
}

/// The most regions the boxes of a fanned-out draw hold for each of
/// `WorkCounts::rate_map_fanned_vertices_boxed_16`, `_64` and `_256` to count its vertices.
const FANNED_BOXES: [u64; 3] = [16, 64, 256];

/// From how many nanoseconds a driver's submission call counts as long
/// (`WorkCounts::submit_long_calls`).
const LONG_SUBMIT_NANOS: u64 = 1_000_000;

/// Why a host step of a rate-mapped pass probed no draw (`WorkCounts::rate_map_steps_unfanned`,
/// `rate_map_steps_nothing_to_probe`, `rate_map_steps_probing_none`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Unprobed {
    Unfanned,
    NothingToProbe,
    ProbingNone,
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
/// the call ran there, on the device's free thread (`free_thread.rs`) or once the batches in
/// flight completed (`freed_memory.rs`). A driver that
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
    /// block, an image bound while work was queued, or an exportable buffer's dedicated
    /// allocation, each with memory of its own, past what its heap keeps empty
    /// (`memory.rs`'s `KEPT_EMPTY_BYTES`). A kept one takes the next placement of its size.
    pub dedicated: u64,
    /// Pooled blocks freed as their last placement left them, their pool already keeping as many
    /// empty ones as it keeps and their heap as many bytes (`memory.rs`'s `KEPT_EMPTY_BLOCKS`
    /// and `KEPT_EMPTY_BYTES`). The allocator's own churn: a working set that dips below its
    /// blocks by more than those and comes back frees and makes a block each time.
    pub pooled_emptied: u64,
    /// Kept empty blocks, pooled and dedicated, freed after their quiet period without a
    /// placement (`memory.rs`'s `KEPT_EMPTY_FOR`): a working set that shrank for good.
    pub kept_stale: u64,
    /// Kept empty blocks, pooled and dedicated, freed so that a block of their memory type could
    /// be made, as one failed to be made, and blocks made ahead and not yet added, freed as the
    /// device's allocations came near its `maxMemoryAllocationCount` (`allocations.rs`): the
    /// allocator giving memory back under pressure.
    pub kept_for_room: u64,
    /// Imports of the caller's host memory, a dedicated block each, freed as their placement went
    /// (`memory::import_host`).
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
    /// The blocks counted above: the allocator's whole share of `vk_memory_freed` and of the
    /// device's checks' (`DeviceCheckWork::vk_memory_freed`).
    pub blocks: u64,
}

/// The Vulkan work of the device's submission and queue-switch checks (`submit_check.rs`,
/// `queue_switch_check.rs`), made on their threads, which reach the device's functions directly
/// and make no call but their check's (`step_aside::Counted::Apart`), and counted here alone: the
/// makings and releases of the whole process's such checks, as `ObjectChurn`'s are the whole
/// process's, and the device's queue-switch check's submissions. A check runs once per driver and
/// release with no record of its verdicts, beside whatever the program does then, so none of this
/// is the program's. A field of `WorkCounts`, for the lanes: no Metal counterpart, not part of
/// the API.
///
/// The in-place check's work is the device's, counted as the program's paths count it
/// (`step_aside::Counted::AsTheDevice`): it makes its buffers, textures and command buffers
/// through those paths, and a pass of the scheduler its waits take on its thread submits and
/// releases the program's batches beside its own.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceCheckWork {
    /// `vkAllocateMemory` calls (`ObjectChurn::vk_memory_allocations` leaves them out).
    pub vk_memory_allocations: u64,
    /// `vkFreeMemory` calls (`ObjectChurn::vk_memory_freed` leaves them out).
    pub vk_memory_freed: u64,
    /// `vkCreateBuffer` calls.
    pub vk_buffers_made: u64,
    /// `vkDestroyBuffer` calls.
    pub vk_buffers_destroyed: u64,
    /// `vkCreateImage` calls.
    pub vk_images_made: u64,
    /// `vkDestroyImage` calls.
    pub vk_images_destroyed: u64,
    /// `vkCreateImageView` calls.
    pub vk_image_views_made: u64,
    /// `vkDestroyBufferView` calls.
    pub vk_buffer_views_destroyed: u64,
    /// `vkBindImageMemory` calls.
    pub vk_image_bindings: u64,
    /// `vkBindBufferMemory` calls.
    pub vk_buffer_bindings: u64,
    /// `vkQueueSubmit` calls the device's queue-switch check made, and the relay's marks after its
    /// uses of a queue (`WorkCounts::queue_submissions` leaves them out); the submission checks'
    /// are `WorkCounts::submit_check_calls`.
    pub queue_submissions: u64,
    /// Steps of the device's checks that waited before they went on: until as long as their own
    /// allocations and frees since their last step took had passed since those ended, then for the
    /// device's allocations and frees on other threads (`step_aside.rs`).
    pub steps_aside: u64,
    /// The nanoseconds those steps waited, in all.
    pub aside_nanos: u64,
    /// Steps of the device's view check (a kernel's making, a timed dispatch) that stood by for
    /// the device's work before they went on: until no batch committed waited for its submission,
    /// none submitted was in flight and no use of a `VkQueue` had been under way for a while
    /// (`view_check.rs`).
    pub view_check_stand_bys: u64,
    /// The nanoseconds those steps stood by, in all.
    pub view_check_stand_by_nanos: u64,
}

/// What a device's commit path did so far (`Device::work_counts`), for the lanes: not part of the
/// API.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorkCounts {
    /// The device's `VkDevice`s alive (`device/line.rs`): its first, and after a recovery from
    /// the loss of one the new one, with the lost one until the objects made on it went and, where
    /// `Device::raw` answered its handles, a later `raw` call answered the new one's. Two while a
    /// recovery's lost `VkDevice` is kept so, one once it goes; a second recovery while objects
    /// made before the first live keeps those, a third.
    pub vk_devices_live: u64,
    /// The `VkDevice`s the device made: its first, and one per recovery.
    pub vk_devices_made: u64,
    /// The device's recoveries from the loss of its `VkDevice` that made a new one
    /// (`Degradation::DeviceRecovered`).
    pub device_recoveries: u64,
    /// The probes sent to another engine of the GPU (the staging queue's) after no queue's work
    /// advanced for 10 seconds with some pending, to ask whether the GPU still runs anything
    /// (`command_buffer/stall.rs`); Metal has no such probe.
    pub stall_probes_sent: u64,
    /// The generations that stopped: no work advanced and the probe did not complete in 2
    /// seconds. Each was failed as a loss is (its oldest command buffer in flight as the hang,
    /// `GpuHang`, the rest as victims), recovered from, and its `VkDevice` parked: never destroyed,
    /// nothing of it released, its queues never waited for (`Device::is_parked`).
    pub wedges_declared: u64,
    /// The command buffers failed as Metal's watchdog fails one (`GpuHang`, code 2): its queue's
    /// front batch went 20.2 seconds (the largest of Metal's measured bounds) without its
    /// timeline advancing, whatever another engine answered. Each also stopped its generation,
    /// which was failed, recovered from and parked as one whose probe went unanswered is (which
    /// `wedges_declared` counts, not this), and counts against its queue as Metal counts a
    /// timeout (`command_buffer/stall.rs`).
    pub watchdog_timeouts: u64,
    /// The command buffers whose queue's timeline passed them while their own timestamps read
    /// unavailable: work the driver dropped with no error the device reports (Linux's amdgpu, a
    /// job its timeout ended and those its context queued behind it). Each failed as Metal's
    /// victim of a GPU recovery (`GpuRecoveryVictim`), not counted against its queue, and the
    /// device went on (`command_buffer/mod.rs`).
    pub unrun_command_buffers: u64,
    /// The longest time, in nanoseconds, that no queue's work advanced while some was pending, as
    /// the completion's waiters saw it at their probes and its end.
    pub longest_stall_nanos: u64,
    /// The carried pipeline states' programs on lost generations let go of at those
    /// generations' rest (`pipeline/carry.rs`'s `LostProgram`), the states living on: one per
    /// state and lost generation.
    pub lost_programs_released: u64,
    /// The carried textures' parts on lost generations let go of at those generations' rest
    /// (`texture/carry.rs`'s `LostTexture`), the textures living on: one per texture or view and
    /// lost generation.
    pub lost_textures_released: u64,
    /// `VkCommandPool`s made for command buffers, where no spare one was free.
    pub command_pools_made: u64,
    /// `VkCommandBuffer`s the device allocated (`vkAllocateCommandBuffers`): its command buffers'
    /// own, where no spare command pool held one of their level (`spares.rs`), with those the
    /// spares stock and begin ahead, and those of its own queues' work. On RADV each is a buffer
    /// object made for its command stream, with its pages cleared.
    pub command_buffers_made: u64,
    /// Released objects MoltenVK commits its residency set for, by kind (`churn_counts.rs`).
    pub object_churn: super::churn_counts::ObjectChurn,
    /// Merged-back shadows' keeps by who planned them (`churn_counts.rs`).
    pub shadow_keeps: super::churn_counts::ShadowKeeps,
    /// The commands recorded into the device's `VkCommandBuffer`s, by kind, as each was recorded
    /// (`recorder`): one per call, the begins and ends among them, whatever the command buffer
    /// (an encoder's, a batch's submission's, the device's own) and the thread; none on a device
    /// past the slots (`commands_uncounted`).
    pub commands_recorded: CommandsRecorded,
    /// 1 where the device records its commands uncounted, 0 where `commands_recorded` counts them:
    /// as it was made, every one of the recorder's slots was held (`recorder::SLOTS`, each by a
    /// device of the process until its `vkDestroyDevice`), so its commands go straight to its
    /// driver's entry points and `commands_recorded` reads none of them. A reading of
    /// `commands_recorded` is the device's commands only where this is 0.
    pub commands_uncounted: u64,
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
    /// Of `spare_command_pools_taken`, those a command buffer took past its first, once its pools
    /// had handed out every `VkCommandBuffer` of a level they held: the spare holding the most of
    /// that level, whose command buffers it hands out before it makes any (`spares.rs`).
    pub spare_command_pools_taken_past_first: u64,
    /// The spare `VkCommandPool`s among those taken that were reset first, as one of their
    /// command buffers may still have been recording: the command buffer that gave the pool back
    /// failed, was dropped unsubmitted, or left a command buffer of it unended (a recording it did
    /// not use).
    pub spare_command_pools_reset: u64,
    /// The command buffers a spare `VkCommandPool` given back freed (`vkFreeCommandBuffers`, on
    /// the device's free thread, `spares.rs`) past the most one of its recent uses handed out:
    /// what a later use of the pool that hands out more allocates again.
    pub spare_command_buffers_freed_past_recent_uses: u64,
    /// The command buffers a pool given back freed past the 4096 the spares hold at most
    /// (`spares.rs`'s `HELD`).
    pub spare_command_buffers_freed_past_held: u64,
    /// The command buffers spares freed whole: one no command buffer took for 10 s, and the
    /// oldest ones past the 4096 the spares hold at most (`spares.rs`).
    pub spare_command_buffers_shed: u64,
    /// The command buffers a pool given back freed, every one it held, as the command buffer
    /// that gave it back recorded 16,384 draws and state commands or more (`spares.rs`'s
    /// `RECORDED`): a rate-mapped pass records each draw once per region of its map, and what a
    /// recording took of its driver's device-wide pools stays taken until its command buffer is
    /// begun again, reset or freed.
    pub spare_command_buffers_freed_past_recorded: u64,
    /// The `vkTrimCommandPool` calls after those frees, one after each 32 command buffers freed
    /// at most, which return to the device what its driver kept of them; on a driver whose trim
    /// after them fails the kept barriers, those of the trim rounds instead, one a pool trimmed
    /// (`spares.rs`).
    pub spare_command_pool_trims: u64,
    /// The trim rounds: on a driver whose trim after each free fails the kept barriers, the kept
    /// barrier command buffers marked to be recorded again, then the spare pools holding the most
    /// command buffers freed untrimmed trimmed, once they held 4096 (`spares.rs`).
    pub spare_command_pool_trim_rounds: u64,
    /// The command buffers freed untrimmed whose objects the driver kept for its pools' later
    /// allocations, destroyed by the trim rounds (`spares.rs`).
    pub spare_command_buffers_trimmed_in_rounds: u64,
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
    /// Batches seen complete that a completion take left, with their queue's later batches, until
    /// a batch of another `VkQueue` they waited for through an event (whose GPU signal met their
    /// wait) was taken: each counted once (`command_buffer/completion.rs`, "After what it waited
    /// for"). Metal's completion follows the GPU's, so a command buffer completes after the one it
    /// waited for.
    pub batches_held_for_another_queue: u64,
    /// Command buffers whose transition to `Completed` waited, its place in its queue's context
    /// kept, for the transition of another queue's command buffer whose GPU signal it waited for
    /// to reach its handlers, where nothing that may block (a handler of the program's other than
    /// its completion-seen handlers) came before that transition in its own context
    /// (`CommandBufferShared::finish_here`): Metal's completion follows the GPU's.
    pub command_buffers_completed_after_another_queue: u64,
    /// Completion-seen handlers run (`CommandBuffer::add_completion_seen_handler`), each as its
    /// command buffer's transition to `Completed` or `Error` ran, its status final.
    pub completion_seen_handlers_run: u64,
    /// Of those, the ones run on the thread that saw their command buffer complete (a queue's
    /// waiter), with no handler thread woken: their command buffer had no completed handler, and
    /// its transition ran there (`completed_where_seen`); the others ran on its queue's handler
    /// thread.
    pub completion_seen_handlers_where_seen: u64,
    /// Command buffers taken with no `vkQueueSubmit`: nothing encoded in them recorded work, and
    /// their one batch waits for and signals no event or semaphore and presents no drawable
    /// (`command_buffer/submit.rs`). Each completes in its queue's order, once the work submitted
    /// before it completed.
    pub command_buffers_unsubmitted: u64,
    /// Command buffers' completion sync files exported from their submitted signal, a descriptor
    /// or the driver's -1 (`CommandBuffer::export_completion_sync_file`): the importer waits on its
    /// own GPU. An export answered after the command buffer finished counts nowhere.
    pub sync_files_exported: u64,
    /// Command buffers' completion sync files answered signaled once the export waited on the host
    /// for the command buffer's completion, whose host copies into the caller's pages (a copied
    /// host buffer's merge back, a two-copy buffer's copy out) come after its GPU work
    /// (`sync-file-after-host-copies`).
    pub sync_files_after_host_copies: u64,
    /// Completion turns made by the thread that took such a command buffer, the scheduler's or
    /// one about to block for a command buffer, where its queue had no work in flight before it
    /// and no turn was under way: no queue waiter woken for it
    /// (`command_buffer/completion.rs`).
    pub completion_turns_where_taken: u64,
    /// Command buffers whose transition to `Scheduled` ran on the scheduler's thread as their last
    /// batch went (or before it, `scheduled_behind_host_steps`), with no handler thread woken:
    /// those with no scheduled handler and no submission handler, whose queue's context held
    /// nothing and ran on no thread then (`CommandBufferShared::schedule`).
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
    /// Command buffers whose GPU times, at their first read, were placed by a calibration pair
    /// that stated a deviation above 20 us, the least of its readings (four at most): inside
    /// their life, each time within that deviation of the moment the GPU wrote it. The device
    /// announces `loosely-calibrated-gpu-times` at the first.
    pub loosely_calibrated_gpu_times: u64,
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
    /// [`image_blits_recorded`](Self::image_blits_recorded)), and takes of a tiled texture's copy
    /// whose later levels it makes (`Buffer::new_tiled_texture`): the lanes that make mipmaps hold
    /// themselves to them.
    pub mipmaps_filtered: u64,
    /// `generate_mipmaps` calls that read the first level in place through a sampled view, not
    /// copied in (`mipmap_filter.rs`): the lane that measures the fetch holds itself to them.
    pub mipmap_first_levels_fetched: u64,
    /// Draws that handed the making of their `VkPipeline` to the device's pool and went on,
    /// rather than make it at the draw.
    pub pipelines_pooled: u64,
    /// Draws and dispatches whose key's `VkPipeline` was not made, which were recorded at once
    /// with the made one of its general form, each of the key's view slots read at its residual
    /// (which reads the same bytes at every bind offset) and, for a draw on a device that
    /// samples through Metal's own texture unit, each texture slot's unmeasured `half` class
    /// unnamed (which no instruction reads): none waited for its key's making.
    pub residual_form_draws: u64,
    /// Of the makings draws and dispatches began, those whose command took its key's general
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
    /// Of `functions_translated`, the kernels translated on the emulated simdgroup tier (each
    /// simdgroup 32 consecutive threads, its operations and barriers through workgroup memory and
    /// workgroup barriers) in place of the pin to 32-lane subgroups, on a device whose compiler
    /// chooses a kernel's SIMD width between a `minSubgroupSize` below 32 and a `maxSubgroupSize`
    /// of 32 (ANV): a kernel of more than 256 AIR instructions whose values are estimated to hold
    /// more than 36 32-bit words at once, which the pin would compile in the device's widest mode,
    /// where its compile time grows to seconds and past them (`docs/API.md`, section 3). Metal pins
    /// nothing: an Apple GPU runs every simdgroup 32 wide.
    pub widest_pin_kernels_emulated: u64,
    /// Kernels' unit-axes variants (`VariantKey::unit_axes`) begun on the device's pool, each by
    /// the first dispatch of threadgroups of one thread along axes that relieve its kernel's
    /// predicated walk of emulated simdgroups (`RuntimeFact::UnitAxes`): the variant takes those
    /// threads' positions on such an axis as uniform, which Metal's contract makes them. Metal
    /// compiles no such variant: an Apple GPU runs a simdgroup's lanes in lockstep, so its kernel
    /// walks no divergent block predicated.
    pub unit_axes_variants_begun: u64,
    /// Direct dispatches that settled on a kernel's unit-axes variant, its translation and each
    /// of its regions' `VkPipeline`s made: once per plan settled, so a dispatch that takes what
    /// an earlier one planned (`dispatches_planned_as_before`) counts nothing here.
    pub unit_axes_dispatches: u64,
    /// Direct dispatches that settled on a kernel's plain variant, which gives the same answers,
    /// while its unit-axes variant was being made on the pool: none waited for that making.
    pub unit_axes_meanwhile: u64,
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
    /// stages, indirect-bindable replays, and draws): none for one a released variant kept, nor
    /// for a linked pipeline another variant of the same link linked
    /// ([`linked_pipelines_shared`](Self::linked_pipelines_shared)).
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
    /// Of those, the makings of a large module that waited for the large makings in flight before
    /// their own, so that two never compile at once: each making whose stages hold 65,536 SPIR-V
    /// words or more waits while the others in flight hold more than 4,194,304 words with its own
    /// (`pipeline/compile_memory.rs`). No Metal counterpart.
    pub makings_held_for_compile_memory: u64,
    /// The time those makings waited, in nanoseconds, summed.
    pub compile_memory_held_nanos: u64,
    /// The times glibc's free heap was handed back to the system with `malloc_trim`: by a large
    /// making (65,536 SPIR-V words or more) whose return left glibc's `malloc` more than 256 MiB
    /// of free heap (`pipeline/compile_memory.rs`: glibc keeps what a driver's compile frees in
    /// the compiling thread's arena), and by the device's idle sweep, once the heap held more
    /// than 256 MiB free above the least free it read since the last trim, at two of its looks in
    /// a row, 5 s apart (`idle_sweep.rs`). Always 0 where the process's allocator is not glibc's
    /// (Linux, the GNU environment). No Metal counterpart.
    pub heap_trims: u64,
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
    /// Immutable sampler sets a command buffer's host step made, counting samplers on the
    /// scheduler's thread, because the device's map held no set of its variant's samplers: the
    /// open break (a render execution's step drawing a state made after its commit, on the variant
    /// for the encoder's sRGB or YCbCr textures, that bakes samplers no variant of the program's
    /// made).
    pub host_step_sampler_sets_made: u64,
    /// Batches' host steps the steps' threads recorded (`scheduler/steps.rs`): each batch whose
    /// segments a host step records, once, apart from the scheduler's thread, which submits other
    /// queues' batches meanwhile, and from other queues' steps.
    pub host_steps_recorded: u64,
    /// Nanoseconds the steps' threads spent recording those host steps, summed
    /// ([`host_steps_recorded`](Self::host_steps_recorded)): the records read, the commands
    /// recorded, and what a step made, the `VkPipeline`s no settle made before it among them
    /// ([`host_step_pipelines_made`](Self::host_step_pipelines_made)).
    pub host_step_nanos: u64,
    /// `VkPipeline`s of states' variants a host step began making itself, made on its thread or
    /// handed to the device's pool and waited for: a draw of a state no settle at commit made
    /// ahead (a state a record the GPU wrote names, or one made after the commit), or a
    /// dispatch's region no earlier dispatch made. Their making's time is part of
    /// [`host_step_nanos`](Self::host_step_nanos).
    pub host_step_pipelines_made: u64,
    /// Chunks of indirect command buffers' commands a host step ran from the recording an earlier
    /// step kept, one each time (`render/executions/kept.rs`): their records, as the GPU and the
    /// CPU left them, held the bytes the recording was made from, and its inputs were equal.
    pub host_step_chunks_kept: u64,
    /// Chunks a host step recorded again, one each time, though an earlier step kept a recording
    /// of them: a record or an input differed from those the recording was made from (a step
    /// whose records changed records them inline, and forgets the recording).
    pub host_step_chunks_recorded_again: u64,
    /// Chunks of render commands a host step that records inline read in place, as an execution
    /// that keeps nothing reads them, one a chunk each time: their records changed at two steps in
    /// a row, so it copies them for the next step's comparison at one step in eight only, each
    /// chunk at a step of its own among them, until a comparison finds them as the step before
    /// read them.
    pub host_step_chunk_copies_put_off: u64,
    /// Records a host step compared, byte for byte, with those a kept chunk was recorded from or
    /// those the last step read, one a record each time.
    pub host_step_records_compared: u64,
    /// The time those comparisons took, in nanoseconds: part of
    /// [`host_step_nanos`](Self::host_step_nanos).
    pub host_step_compare_nanos: u64,
    /// Draws an execution's replay recorded, plain in every other respect, that a chunk kept no
    /// recording of for their heap set alone, one each (`render/executions/kept.rs`): the device
    /// lacks `descriptorBindingUpdateUnusedWhilePending`, so each command buffer binds a version
    /// of the set of its own.
    pub replayed_heap_draws_unshared: u64,
    /// Such draws, kept from being plain as they write elements of their own into the heap set
    /// (a function table's), which a later command of another table may have written over.
    pub replayed_heap_draws_writing: u64,
    /// Such draws, kept from being plain as their heap set holds a shadowed texture, a decoded
    /// twin or a raw read, which a draw's binding of it announces.
    pub replayed_heap_draws_noted: u64,
    /// Such draws, kept from being plain as they read the device's sampler tables.
    pub replayed_heap_draws_over_sampler_tables: u64,
    /// Such draws, kept from being plain as their tables bind a texture at a slot their heap part
    /// reads, which no kept recording's key holds (a host step's tables bind none).
    pub replayed_heap_draws_table_textures: u64,
    /// Segments whose recording at `commit` was refused, of an execution of an indirect command
    /// buffer the command buffer made resident for writing, held for a host step, one each
    /// (`deferred.rs`): a route met after the commit (another queue's command buffer the work
    /// waits for) may write the records first, as Metal runs what it wrote. The step records the
    /// work once the work before it completed, and fails the command buffer only where the records
    /// are still refused.
    pub commit_refusals_held: u64,
    /// `VkPipeline`s states' creations made: each render state's predicted key
    /// (`pipeline/predicted.rs`), and each compute state's predicted region
    /// (`pipeline/dispatch_making.rs`), each with its general form (a region's residual form)
    /// beside it where that differs: on a profile whose storage views declare
    /// `SpecKey::RangeAtBindOffset`, the views read at their residuals.
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
    /// Takes of such a backing its queue kept, one per attachment of each pass after the one that
    /// made it: with `pooled_backings_made`, every backing passes were handed.
    pub pooled_backings_taken: u64,
    /// Of those takes, the ones made while another command buffer that took the backing before
    /// had not completed: passes whose uses of one image the hazard tracker orders, as any
    /// texture's (`tracker.rs`), where nothing else orders them.
    pub pooled_backings_taken_in_flight: u64,
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
    /// Placement heaps' textures not made at an offset inside the heap whose message
    /// (`metal2vulkan-heap-texture-refused`) was withheld: each after the device's first, which
    /// it said (`Heap::new_texture_at_offset`). The refusal is answered as before.
    pub heap_texture_refused_withheld: u64,
    /// Decoded twins' failed CPU decodes whose message (`metal2vulkan-twin-decode-failed`) was
    /// withheld: each after the device's first, which it said.
    pub twin_decode_failed_withheld: u64,
    /// Late reservations of resource IDs, ignored, whose message
    /// (`metal2vulkan-late-reservation`) was withheld: each after the device's first, which it
    /// said (`Device::reserve_resource_ids`).
    pub late_reservation_withheld: u64,
    /// Placements and reads of the queue sharing the device's queue limits changed from its queue
    /// plan's whose message (`metal2vulkan-queue-limits-reached`) was withheld: each after the
    /// device's first, which it said (`DeviceOptions::queue_limits`).
    pub queue_limits_reached_withheld: u64,
    /// Bytes of `VkDeviceMemory` the runtime allocated so far for memory of its own (its
    /// blocks, pooled or dedicated, freed since or not): none for host memory it imports. A
    /// detach allocates none for the bytes it detaches (`detach.rs`).
    pub device_memory_allocated: u64,
    /// Placements of a class that requires device-local memory (`Private` resources, a `Managed`
    /// buffer's own bytes without unified memory, `Memoryless` attachments) that lie in system
    /// memory, because no device-local memory type the resource takes had room for them
    /// (`memory.rs`'s `overflow_candidates`): 0 on a device with unified memory, and on one whose
    /// driver pages device-local memory out to make room (`VK_EXT_pageable_device_local_memory`)
    /// unless it still fails an allocation. Each is a placement, the runtime's own included; the
    /// program's resources among them are announced (`Degradation::DeviceMemoryOverflow`).
    pub device_memory_overflows: u64,
    /// The runtime's live pooled `VkDeviceMemory` blocks with a placement in them (`memory.rs`):
    /// with the blocks below and every dma-buf import, the allocations ANV on i915 lists in each
    /// of its kernel submissions, beside its own buffer objects.
    pub memory_blocks_live: u64,
    /// Its live blocks with no placement, pooled or dedicated, kept for the next placements
    /// (`memory.rs`'s `KEPT_EMPTY_BLOCKS` and `KEPT_EMPTY_BYTES`). Once they stayed empty for
    /// 10 s (`KEPT_EMPTY_FOR`), whether or not the program makes anything more
    /// (`idle_sweep.rs`), at most one pooled block of each pool the runtime keeps room in for its
    /// own resources stays (`memory.rs`'s `Memory::keeps_room`).
    pub empty_memory_blocks_live: u64,
    /// The pooled blocks it made so far (`memory.rs`), each a `vkAllocateMemory`.
    pub pooled_memory_blocks_made: u64,
    /// The pooled blocks it freed so far: emptied past what their pool and heap keep, kept empty
    /// past their quiet period (at a release, a creation call or the device's idle sweep,
    /// `idle_sweep.rs`), or freed for a block of their type that failed to be made (`memory.rs`'s
    /// `KEPT_EMPTY_BLOCKS`, `KEPT_EMPTY_BYTES` and `KEPT_EMPTY_FOR`).
    pub pooled_memory_blocks_freed: u64,
    /// Its live dedicated blocks with their placement: a placement larger than half a block, one
    /// placed apart, or an exportable buffer's dedicated allocation (`exported_memory.rs`).
    pub dedicated_memory_blocks_live: u64,
    /// Of `pooled_memory_blocks_made`, the blocks of the pool a color texture's host-transfer
    /// image places in made ready off the program's thread: allocated and every page of their
    /// mapping read once by the host, then added to the pool by the first placement there that
    /// found no room, so that no host copy faults on memory whose kernel work waits behind the
    /// program's queued GPU work (`memory/ready.rs`; on RADV after runtime suspend, a new
    /// block's clear). A block whose making a placement started and waited for, finding none made
    /// or under way, is among them with none of its pages read, as a block made in place. None
    /// on a device that copies no color texture on the host.
    pub host_copy_blocks_readied: u64,
    /// The blocks of that pool a placement made itself, finding no room and no block made ready
    /// while its thread read a block's pages (or its making failed, or no thread started): their
    /// pages untouched, a host copy's first touch there may wait for the memory's kernel work
    /// (`memory/ready.rs`).
    pub host_copy_blocks_made_in_place: u64,
    /// Of `pooled_memory_blocks_made`, the blocks of buffers made ahead off the program's thread,
    /// each right after a burst of the device's submissions ended, and added to their pool where
    /// room was needed: blocks of the pools the runtime's encodings and commits place their own
    /// resources in, and of the pools of exportable buffers (`exported_memory.rs`), made while
    /// such a pool grows (`memory/ready.rs`). A block's `vkAllocateMemory` holds the device's
    /// submissions while it runs on RADV and on NVIDIA's driver, so it is made between them.
    pub buffer_blocks_made_ahead: u64,
    /// The blocks of buffers of those pools a call made itself, where no making ahead served it:
    /// its making failed or its host started no thread, or the placement came before the
    /// device's first creation call (`memory/ready.rs`). Elsewhere a call that finds no block made
    /// ahead of the size it needs waits for one (`blocks_ahead_waited_for`).
    pub buffer_blocks_made_in_place: u64,
    /// The blocks the runtime's own buffers and the program's lie in (`memory.rs`'s
    /// `Holds::FdBuffers`, and `Holds::FdDmaBufs` for those that declare their padding
    /// unobserved) made so far, pooled or dedicated, allocated exportable as an opaque file
    /// descriptor (and a dma-buf, of the latter) and never exported, so that their making holds no
    /// other submission (the module docs of `memory.rs`, "Buffers in exportable blocks"): 0 on a
    /// device that takes no such blocks (`Memory::fd_buffer_types`'s reason, which the memory log
    /// names).
    pub fd_buffer_blocks_made: u64,
    /// The blocks of those buffers made so far as any block of buffers, not exportable: of a
    /// memory type of a device-local heap, which such buffers overflowed into or their class
    /// takes first.
    pub fd_buffer_blocks_plain: u64,
    /// The plain blocks of buffers made so far, pooled or dedicated, allocated with no export
    /// (`memory.rs`'s `Holds::Buffers`): the runtime's own buffers' and the program's on a
    /// device that takes no blocks allocated exportable, before its check of them answered and
    /// once it refused them; those of buffers that declare their padding unobserved
    /// (`BufferDescriptor::padding_unobserved`) where the device takes no blocks exportable as
    /// dma-bufs too for them (the memory log names why); and those of the few buffers the runtime
    /// places apart. Beside `fd_buffer_blocks_made`, the share of the
    /// blocks of buffers whose `vkAllocateMemory` holds the process's submissions on RADV (the
    /// module docs of `memory.rs`, "Buffers in exportable blocks").
    pub plain_buffer_blocks_made: u64,
    /// The blocks the program's host-visible buffers lie in (`memory.rs`'s `Holds::PageBuffers`)
    /// made so far, pooled or dedicated, of pages the runtime mapped and imported by their
    /// pointer, which a recovery imports again on the device's next generation, so that the
    /// buffers there keep their bytes and `contents` pointers (the module docs of `memory.rs`,
    /// "Buffers in the runtime's own pages"): 0 on a device that takes no such blocks (the memory
    /// log names why), and until its submission check found a held import costs its
    /// submissions nothing.
    pub page_buffer_blocks_made: u64,
    /// The blocks of those buffers made so far as any block of buffers: of a memory type the
    /// device imports no such pages as, or where the system or the driver refused them.
    pub page_buffer_blocks_plain: u64,
    /// The buffers of the runtime's own and the program's made so far in a plain block, not one
    /// allocated exportable, on a device that allocates such blocks, while the device's in-place
    /// check of them was pending (`in_place.rs`'s `Device::fd_blocks_checked`): on its first run
    /// on a driver, before the check's record holds its verdicts, until a check a texture over
    /// host pages or `Device::settle_in_place_checks` started answered (a buffer's making starts
    /// none). 0 once they are known. A buffer that declares its padding unobserved is counted too
    /// where the device takes blocks exportable as dma-bufs too for it, and lies in a plain block
    /// elsewhere, not counted.
    pub fd_blocks_awaited: u64,
    /// As `fd_blocks_awaited`, once the device's check of such blocks found a texture over one's
    /// buffer would be a shadow where over a plain block's it is not (`in_place.rs`'s
    /// `blocks_take`), or that such a block does not hold every image a texture over a plain
    /// block's buffer that keeps its padding takes (`texture/fd_blocks_cover.rs`, whose answer
    /// the memory log names): every such buffer from then on.
    pub fd_blocks_refused: u64,
    /// As `buffer_blocks_made_ahead`, of images, those of the pool host copies place in among
    /// them (`host_copy_blocks_readied`).
    pub image_blocks_made_ahead: u64,
    /// As `buffer_blocks_made_in_place`, of images, those of the pool host copies place in among
    /// them (`host_copy_blocks_made_in_place`).
    pub image_blocks_made_in_place: u64,
    /// Of the blocks made ahead, those made while a call that found none of the size it needed
    /// waited, rather than make a block itself (`memory/ready.rs`): the block being made or still
    /// to make, or one the pool's thread made first for the call. In the pool host copies place
    /// in, never while its thread reads a block's pages.
    pub blocks_ahead_waited_for: u64,
    /// Of those, the blocks whose making was still waiting for a burst of the device's
    /// submissions to end as the call came: the call's wait made it allocate at once, so that the
    /// call waited for the allocation alone, never for the program's own submissions to pause
    /// (`memory/ready.rs`).
    pub blocks_ahead_hurried: u64,
    /// Of the blocks made ahead, those whose wait for a burst of the device's submissions to end
    /// reached its bound (50 ms), made all the same: the device submitted with no gap of 1 ms
    /// between its bursts (`device/queue_quiet.rs`).
    pub blocks_ahead_past_quiet_wait: u64,
    /// The blocks made ahead that went unused and were freed: their pool having stopped growing
    /// 10 s before with room for half a block, or given back, no call waiting for them, as the
    /// device's allocations came within a quarter of its `maxMemoryAllocationCount`, so that an
    /// object's allocation at the limit takes theirs (`memory/ready.rs`, `allocations.rs`):
    /// counted in `pooled_memory_blocks_made` and `pooled_memory_blocks_freed`.
    pub blocks_ahead_unused: u64,
    /// Every block its allocator freed so far, by what made it go (`MemoryFrees`): one
    /// `vkFreeMemory` each, and the allocator's whole share of
    /// `object_churn.vk_memory_freed`.
    pub memory_frees: MemoryFrees,
    /// Of `memory_frees`, the blocks whose `vkFreeMemory` was posted to the device's free thread,
    /// off the thread that released them (`free_thread.rs`): every block that went but an import
    /// of the caller's memory, one that waited for the batches in flight, and one that found the
    /// thread's queue full (`free_thread_queue_full`).
    pub memory_frees_posted: u64,
    /// The blocks freed where they went as the free thread's queue was full (`free_thread.rs`).
    pub free_thread_queue_full: u64,
    /// The waits for the frees posted to the free thread: an allocation at
    /// `maxMemoryAllocationCount`, a block whose making would take its heap past the heap's size,
    /// or one that failed for room (`free_thread.rs`).
    pub memory_free_waits: u64,
    /// The frees of the device's own threads that free in runs that first waited as long as the
    /// thread's own frees before them took, less the time since they ended (`step_aside::pace`):
    /// the free thread's (its blocks, and the host imports posted to it,
    /// `host_import_frees_posted`), the residency thread's and a guest RAM thread's, so that the
    /// device's other threads' submissions go between them. A program's own release never waits
    /// so.
    pub memory_frees_paced: u64,
    /// The nanoseconds those frees waited, in all.
    pub memory_frees_paced_nanos: u64,
    /// The `VkDeviceMemory` allocations the runtime refused without calling the driver, the
    /// device holding `maxMemoryAllocationCount` of them live (`allocations.rs`): its blocks, its
    /// imports of host memory and of dma-bufs, counted together.
    pub memory_allocations_refused_at_limit: u64,
    /// Its live imports of the caller's host memory, a block each (`memory::import_host`).
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
    /// rebind or a detach of host pages, by a change of storage form, or by a texture's turn to
    /// its other face (`texture/faces.rs`, one each): for a rebind or a detach, one for each live
    /// texture whose images or texel views lie in those bytes (in a no-copy heap or over one of
    /// its buffers, for the heap's), and none for any other texture.
    pub heap_slots_rewritten: u64,
    /// Writes of texture slots of the device's heap (a new texture's, or those a rebind, a detach
    /// or a change of storage form writes again) that waited for a batch: asked between its
    /// first work's take and its `vkQueueSubmit`, they were made once its submission was made,
    /// before the next batch took its first work. So a texture's making waits for no batch's
    /// submission. A device whose heap sets take no writes while a command buffer is pending
    /// defers none: there a texture's registration waits for the batch instead.
    pub heap_slot_writes_deferred: u64,
    /// Image arrays of new heap sets that took their elements from a live heap set's array of
    /// the same class and descriptor type, copied in runs of the resource IDs textures held, with
    /// what that array noted of them: such a set takes no texture, and walks none of them. An
    /// array of a class no live set holds writes each live texture's element instead.
    pub heap_arrays_copied: u64,
    /// Uses of a buffer over host pages that looked up by address the ranges of host bytes their
    /// bytes meet (`host_aliases.rs`): a use of a range met by no more ranges than the classes of
    /// length that lookup visits reads those its range keeps instead.
    pub host_range_lookups: u64,
    /// Times the host aliases looked uses up under their lock (`host_aliases.rs`): once a buffer
    /// joined another's memory or two ranges of placed bytes crowd, at each recorded draw,
    /// dispatch or blit command one of whose uses of bytes names a buffer its encoder has not
    /// found since a buffer last joined, was placed, moved or went (an encoder opens with the keys
    /// an ended one found: `host_aliases::Quiet`), and at each other reading of a command's uses
    /// (an execution's, a shadowed texture's).
    pub host_alias_lookups: u64,
    /// Textures made in heaps laid out as Metal tiles them, each a shadow of its elements (a
    /// multisample texture's samples) in the heap's bytes (`texture/heap_tiled.rs`): in a heap over
    /// host pages, a depth or stencil texture in any heap on a driver that lays such images apart
    /// from their memory or keeps them GPU-optimized (MoltenVK), a multisample texture in any
    /// heap whose memory takes no image of it, and a texture in any heap whose images take more
    /// bytes or a coarser alignment than Metal's answer where none of them that fits it binds
    /// there (`texture/heap.rs`, `Device::fitting_heap_image`).
    pub heap_textures_tiled: u64,
    /// Textures made in heaps whose image of the route the device first took (a compressed
    /// optimal image, say) takes more bytes or a coarser alignment of the heap than Metal's
    /// answer (under `HeapAnswerPolicy::Device`, an alignment coarser than the answer takes: past
    /// 64 KiB, or past both 4 KiB and its size), bound in the heap's memory in place as an image
    /// of them that fits it instead: its optimal image of host-transfer usage, or its linear image
    /// (`heap-texture-host-transfer`, `heap-texture-linear`; `texture/heap.rs`). Each would else
    /// be a shadow of the heap's bytes (`heap_textures_tiled`).
    pub heap_textures_refit: u64,
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
    /// once (in a heap over host pages, its linear image made for host allocations among them):
    /// none for a plan and image asked before on a device that enables `maintenance4`, which
    /// fixes the answer (`texture/requirements.rs`).
    pub requirement_images_made: u64,
    /// The device's limits (`vkGetPhysicalDeviceImageFormatProperties2`) heap textures' plans
    /// asked for the images whose memory requirements they ask: none for a plan and image whose
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
    /// Fills recorded as `vkCmdFillBuffer` (`word_fill.rs`): every fill the encoders and the
    /// runtime's own work record, but a new buffer's first zeros (`new_buffer_zero_fills`).
    pub fills_recorded: u64,
    /// Buffer copies of a blit encoder recorded as one more region of the `vkCmdCopyBuffer` of the
    /// copy before them (`copy_run.rs`): consecutive `copy_from_buffer_to_buffer` calls between the
    /// same two `VkBuffer`s that the encoder's order needs no barrier between, on any queue.
    pub buffer_copies_merged: u64,
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
    /// Shadowed 2D textures over a buffer's rows made with their rows read and written in place
    /// (`texture/texel_rows.rs`): the device made no image over the rows (its format, features or
    /// limits, the rows' pitch or place, or the dma-buf import refused one), and a slot whose
    /// function only reads, writes or asks the size of such a texture binds a texel buffer over
    /// the rows' bytes, which its commands read and write with no copy into or out of the shadow.
    pub texel_rows_textures: u64,
    /// Bindings of such textures' rows written into descriptor sets.
    pub texel_rows_bound: u64,
    /// Texel buffer views of the parts past the first of texture buffers of more texels than the
    /// device's `maxTexelBufferElements` written into descriptor sets, one per such part
    /// (`metal2vulkan::bind::texel_buffer_parts`): a slot whose function reads, writes, updates
    /// or asks the width of such a texture buffer binds an array of views of its parts, where one
    /// view would be refused (`BindError::TexelBufferTooLarge`).
    pub texel_buffer_parts_bound: u64,
    /// Buffer slots' uniform views written into descriptor sets in their texel form, a uniform
    /// texel buffer view of the descriptor range made for each such write: on a device whose
    /// uniform buffer reads of external memory miss a cache
    /// (`DriverWorkaround::ExternalUniformReadsSkipACache`), a uniform view whose window the
    /// translation bounded to at most 4 KiB reads its words through such a view
    /// (`DescriptorView::UniformTexel`), with the speed of device memory on imported memory too.
    pub uniform_texels_bound: u64,
    /// Bound buffers' uniform views written into descriptor sets as uniform buffers, their slots'
    /// windows at most 16 KiB long: on a device that takes the texel form, where every window of
    /// at most 4 KiB takes it, the windows between 4 and 16 KiB. Counted at each write the
    /// runtime makes (a draw rebound at other slots keeps its write, counting nothing); a slot's
    /// `set*Bytes` data is not counted here.
    pub uniform_buffers_bound_to_16k: u64,
    /// As [`uniform_buffers_bound_to_16k`](Self::uniform_buffers_bound_to_16k), the windows longer
    /// than 16 KiB and those the translation did not bound (`Window::Whole`).
    pub uniform_buffers_bound_past_16k: u64,
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
    /// Two-plane YCbCr textures whose planes read in place take no such image because the planes
    /// lie in a block of the runtime's buffers allocated exportable (`memory.rs`'s "Buffers in
    /// exportable blocks"), whose image would start `UNDEFINED`, which the device's in-place check
    /// covers for single-plane images alone: the texel buffer reads them, announced
    /// (`ycbcr-planes-exportable-block`).
    pub ycbcr_planes_images_kept_exportable_block: u64,
    /// Bindings of two-plane YCbCr textures' planes images written into descriptor sets.
    pub ycbcr_planes_images_bound: u64,
    /// Two-plane YCbCr textures made whose conversions sample the planes' image through the
    /// device's own Y′CbCr conversion where it converts their format (`ycbcr_device.rs`), in
    /// place of Metal's arithmetic: counted as each is made, as its conversion is, though the
    /// conversion's image, kernel and sets are made at its first conversion.
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
    /// `vkCmdDrawIndirect` and `vkCmdDrawIndexedIndirect` recorded (`render.rs`): a draw from an
    /// indirect buffer. A command of an indirect command buffer that a host step or a commit
    /// replays from its record is recorded as a direct draw, counted not here but in
    /// [`indirect_records_read`](Self::indirect_records_read)'s read of it.
    pub indirect_draws_recorded: u64,
    /// Records of indirect command buffers' commands read from their storage and decoded, one a
    /// command each time an execution recorded at its commit, a host step (its check included,
    /// `IndirectCommandCheck`) or `IndirectCommandBuffer::commands` reads it; a setter's own
    /// read of the command it changes is not counted.
    pub indirect_records_read: u64,
    /// Runs of consecutive commands' records of indirect command buffers read at once, one a run
    /// of one or more commands: by `IndirectCommandBuffer::commands`, and by an execution recorded
    /// at its commit, which reads its range a chunk of 256 commands at a time
    /// (`kept_commands::pieces`), or whole where its check, or its command buffer's
    /// executed-commands handlers, see every command before any is recorded.
    pub indirect_record_runs_read: u64,
    /// Bytes of indirect command buffers' records the host copied out of their storage, to decode
    /// them or to keep them for a comparison: a record a host step reads is copied once, into
    /// the copy the next step compares its records with where its execution keeps its chunks
    /// (`render/executions/kept.rs`), and decoded from there.
    pub indirect_record_bytes_copied: u64,
    /// Bytes of indirect command buffers' records blits copied on the GPU
    /// (`copyIndirectCommandBuffer:...`, `blit_indirect.rs`), counted as each copy is encoded:
    /// `bind::indirect_commands::COMMAND_BYTES` a command whatever the buffers' descriptor, read
    /// from the source's storage and written into the destination's (system memory on a discrete
    /// GPU for a `Shared` buffer).
    pub indirect_record_bytes_gpu_copied: u64,
    /// Bytes of indirect command buffers' records blits reset on the GPU
    /// (`resetCommandsInBuffer:withRange:`, `blit_indirect.rs`), counted as each reset is
    /// encoded: `COMMAND_BYTES` a command, written into the buffer's storage.
    pub indirect_record_bytes_gpu_reset: u64,
    /// Nanoseconds of the GPU's time the dispatches of those resets took
    /// (`resetCommandsInBuffer:withRange:`, `blit_indirect.rs`), each between timestamps written
    /// before and after it in its command buffer and read as the command buffer lets go of what it
    /// retained, once it completed: the same timing as a rate-mapped pass's pieces
    /// ([`rate_map_piece_gpu_nanos`](Self::rate_map_piece_gpu_nanos)). Counted only where the
    /// device writes timestamps and its driver's timestamps do not wait for every queue's
    /// (`DriverFact::TimestampsWaitForEveryQueue`); elsewhere none is written and this stays 0.
    pub indirect_record_reset_gpu_nanos: u64,
    /// Indirect command buffers whose records came to lie in host memory for good at a host
    /// access (a setter, a getter, a CPU reset, a commit or host step that reads them) on a device
    /// without unified memory (`indirect_command_buffer/home.rs`): one placed there by its first
    /// use, or one the GPU used first whose records moved there from device-local memory once no
    /// command buffer held them. A `Private` buffer's never move.
    pub icb_records_moved_to_host: u64,
    /// Bytes of indirect command buffers' records those moves copied from device-local memory into
    /// host memory, through the staging queue: the whole storage where the GPU may have written
    /// the records, none where it did not or nothing placed them in device-local memory.
    pub icb_record_bytes_moved: u64,
    /// Host accesses of indirect command buffers' records in device-local memory that went through
    /// the staging queue (`indirect_command_buffer/home.rs`): a `Private` buffer's, and a `Shared`
    /// one's while a command buffer that used its records was not complete or dropped, but an
    /// access that only reads records the GPU never wrote, which reads host memory in place. Each
    /// copies the records it reads into host memory where the GPU may have written them, and the
    /// records it writes back, waiting for the copies (`Degradation::StagedIndirectRecords`).
    pub icb_records_staged_accesses: u64,
    /// Indirect command buffers a committed command buffer made resident for writing
    /// (`use_resource` or `use_resources` with `ResourceUsage::WRITE`, in any of its encoders) and
    /// left unmarked (`IndirectCommandBuffer::gpu_written` still `false`): it set no pipeline state
    /// whose functions may call an indirect-command operation, and executed no indirect commands
    /// whose records may name one, so the GPU cannot write their records and their executions
    /// stay off the host. One per buffer per commit; one the commit marked, or found marked, is
    /// not counted.
    pub indirect_write_residencies_unmarked: u64,
    /// Chunks of indirect command buffers' commands an execution ran from the recording an
    /// earlier one kept, one each time (`kept_commands.rs`): their command buffer runs that
    /// recording again, and their records are not read.
    pub indirect_chunks_kept: u64,
    /// The commands of those chunks, one a command each time.
    pub indirect_commands_kept: u64,
    /// Chunks of indirect command buffers' commands recorded to keep, each in a context of its own
    /// (a command pool, a descriptor arena and uploads, `kept_commands::KeptContext`): one each
    /// time a chunk's second or later execution with no write reaching it since records it apart.
    /// None for a range's first execution, which records as one that keeps nothing.
    pub kept_contexts_made: u64,
    /// Comparisons, whole, of the resources an execution of indirect commands made resident with
    /// those a kept chunk of them was recorded under (`kept_commands::ResidentMet`): one for each
    /// distinct list the execution's kept chunks hold, which the chunks one execution recorded
    /// share; the chunks after compare by pointer.
    pub kept_resident_keys_compared: u64,
    /// Commands of executions of render commands recorded in their GPU form, one a command each
    /// time (`render/executions/gpu.rs`): the device read their records, the host none.
    pub gpu_form_commands: u64,
    /// Of those, the commands drawn by one merged draw per execution rather than one draw each
    /// (`render/executions/gpu.rs`), one a command each time.
    pub merged_draw_commands: u64,
    /// Bytes of the scratches of executions of render commands recorded in their GPU form that
    /// lie in device-local memory, a scratch's bytes each time (`render/executions/gpu.rs`,
    /// **Scratch**): the head the host uploads and a copy carries in, the slot tables and
    /// arguments the preprocess writes, and a merged replay's sections, which the merge passes
    /// and the draws read. A scratch the device placed elsewhere counts nothing.
    pub gpu_form_device_local_bytes: u64,
    /// Blocks of the GPU pools made: a queue's index pool block (for its index bounds passes and
    /// index rewrites) or its mesh records pool block, made by the first command buffer of the
    /// queue that needs one where the queue holds none and the device has no spare of its kind (a
    /// block another queue let go). A queue's command buffers share its block of each kind, so a
    /// queue makes one while it keeps using them, and a device makes no more of a kind than the
    /// most queues that held one at once. Each block is a sixty-fourth of the device's largest
    /// device-local heap. Those made less those released are the blocks the device holds, its
    /// spares included.
    pub gpu_pool_blocks_made: u64,
    /// Blocks of the GPU pools released before the device's teardown: a queue's block no command
    /// buffer held for two seconds, released as another lease of its kind ended, or the block of
    /// a queue that went, where the device held a spare of its kind already (else the block
    /// became the spare, for the next queue that needs one).
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
    /// The submitting thread's CPU time across those calls, in nanoseconds: what the driver
    /// worked on that thread (`CLOCK_THREAD_CPUTIME_ID` on Unix; on Windows `GetThreadTimes`,
    /// which the system charges at its clock's ticks, so a call's share there is sampled, true
    /// over many calls on average). Near `batch_submit_nanos`, the calls worked; well below it,
    /// they waited (a lock, a fence, the thread preempted). Counted only where the device notes
    /// queue times (`METAL2VULKAN_QUEUE_TIMES`, a default-off diagnostic), 0 elsewhere: the
    /// thread's clock is a system call on Linux, not the vDSO's, two reads a call costing about
    /// 0.9 us on a Ryzen 7 7840HS under Linux 7.3 (`tests::submit_stalls`'s
    /// `thread_clock_bench`), which every submission would pay. A call whose thread's clock did
    /// not read adds 0.
    pub batch_submit_cpu_nanos: u64,
    /// Of those calls, the ones that took 1 ms or more, on every driver, by length alone
    /// (`submit_stalls_on_work` counts a driver's stalls instead, by the thread's CPU time where
    /// the driver has them).
    pub submit_long_calls: u64,
    /// Nanoseconds those long calls spent in the driver.
    pub submit_long_call_nanos: u64,
    /// The submitting thread's CPU time across those long calls, as `batch_submit_cpu_nanos`
    /// reads it, and only where it does (the device notes queue times; 0 elsewhere): near
    /// `submit_long_call_nanos`, they worked; well below it, they waited.
    pub submit_long_call_cpu_nanos: u64,
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
    /// Encoders past the first 2,048 of their command buffer, each of whose own command buffers
    /// began with a barrier from everything before on its queue, so its batch begins no command
    /// buffer of barriers before it (the barriers the scheduler plans for the first 2,048 are
    /// recorded so): a batch of any number of encoders begins at most 2,048 command buffers for
    /// their barriers. Such a barrier orders every command of the encoder after every command
    /// before it, where Metal orders only those whose resources meet.
    pub encoders_barriered_at_open: u64,
    /// Compute, blit and render encoders that began no `VkCommandBuffer` of their own: each
    /// recorded into the one of the encoders before it in its command buffer, where nothing
    /// between them needs a boundary (the same batch, a compute encoder's dispatch type that of
    /// the group's compute encoders, no work left for later, nothing failing; a
    /// render encoder's pass recorded at its end, with a log of its calls), after the barrier the
    /// tracker plans between two encoders of a batch, and its batch's submission submits no
    /// command buffer for it, of barriers or of its own. Of `encoder_command_buffers_begun`'s
    /// encoders, those it does not count.
    pub encoders_folded: u64,
    /// Compute, blit and render encoders that asked for a group of encoders and joined none
    /// because the group's encoders were of another kind: a compute encoder of another dispatch
    /// type than the group's compute encoders, or a render encoder that may not join; and an
    /// encoder after one that asks for no group. Each refusal is counted once, as the encoder
    /// opens, by the first reason that holds, in the order of these counts; the command buffer's
    /// first encoder is refused nothing (`encoder_fold.rs`'s **Refusals**).
    pub folds_refused_kind: u64,
    /// Of those refusals, where the group's shared `VkCommandBuffer` was no longer the command
    /// buffer's last.
    pub folds_refused_tail: u64,
    /// Of those refusals, a batch boundary since the group's last encoder: an event's wait or
    /// signal, a host step, a resource state encoder, or a signal pending.
    pub folds_refused_batch: u64,
    /// Of those refusals, where the command buffer used command predication, its protection
    /// options were not 0, or the encoder trail was on.
    pub folds_refused_state: u64,
    /// Of those refusals, where the encoder before left work for later (a dispatch whose
    /// pipelines were being made, an indirect command buffer's execution) or went in more than one
    /// part (a host step inside it).
    pub folds_refused_parts: u64,
    /// Of those refusals, where the encoder before failed or faulted its command buffer, a
    /// command-predication record left its work out, or its batch's submission records after it
    /// or judges by it (a write into an ASTC carrier, a texture written whole before any of its
    /// commands read it).
    pub folds_refused_ends: u64,
    /// Commands of encoders of a group after a member of it updated a fence whose barrier orders
    /// them after the writes the update covers, whether or not their encoder waits for the fence
    /// (a wait covers all of an encoder's commands, and may come after some of them went into the
    /// group's `VkCommandBuffer`): from those writes to what the command may reach of memory no
    /// use names in its stages and to its uses' accesses, once for each wider reach
    /// (`encoder_fold.rs`'s **Fences**).
    pub fold_fence_barriers: u64,
    /// Compute encoders that opened on the box of their core and the bind state (with what its
    /// last dispatch settled) the command buffer's last compute encoder left, rather than on the
    /// device's spares: each compute encoder's end parks them in its command buffer for the next,
    /// and the commit gives them back to the device (`compute/live.rs`'s `Parked`).
    pub parked_compute_taken: u64,
    /// The lists a released command buffer keeps of what its encoders ended with (its ended
    /// encoders' records, their recordings, a group's logs its members did not keep and those held
    /// for its commit, its command predication's lists, and its groups' order's map and lists)
    /// that held more room as it was released than the storage it started on:
    /// counted once each, at the release. A command buffer starts on the state a released one of
    /// the device left (`record_storage_kept`), its lists emptied with their room kept, so one of
    /// no more encoders than that one's grows none.
    pub record_lists_grown: u64,
    /// Released command buffers whose state the device kept, reset by the thread that released
    /// it (what it held gone, its lists of records emptied with their room kept), for a command
    /// buffer it makes next to take whole: at most 64 at once, each of at most 8 MiB of list room
    /// and 16 MiB over all, the latest taken first.
    pub record_storage_kept: u64,
    /// Command buffers made on a state built for them, the device keeping none that a released
    /// one left (`record_storage_kept`). Every other one takes a kept state whole, so its making
    /// neither allocates nor fills one.
    pub command_buffer_states_made: u64,
    /// Of `command_buffers_begun`, the device's own submissions' and staging submissions'.
    pub command_buffers_begun_internal: u64,
    /// Of `encoder_command_buffers_begun`, the encoders' own `VkCommandBuffer`s begun ahead: a
    /// settled spare command pool's primaries, begun on the device's begin-ahead thread once the
    /// command buffer that gave it back completed, as many as encoders of that command buffer
    /// opened their own in it (eight at most), which the encoders of the command buffer that
    /// takes the pool record into, begun no more on its thread.
    pub encoder_primaries_begun_ahead: u64,
    /// Of `encoder_command_buffers_begun`, the encoders' own `VkCommandBuffer`s their spare
    /// command pool's begin ahead owed them and had not begun (`encoder_primaries_begun_ahead`):
    /// the command buffer took the pool before the device's begin-ahead thread made its begins,
    /// or the pool was out of the spares as the thread came to it. Each was begun on the
    /// encoding thread, as one in a pool owed none.
    pub encoder_primaries_begun_ahead_missed: u64,
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
    /// Groups of batches held for one kernel submission (`CompletionPolicy::Grouped`, or command
    /// buffers marked to join the next under `CompletionPolicy::Exact`) that went holding the most
    /// they may (four), each group counted once by what sent it; with the eight fields after this
    /// one, every group. Their spread says why a device's groups hold fewer.
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
    /// Groups that went before a batch that could otherwise join but whose queue's expected GPU
    /// run is unknown (none of its command buffers completed with GPU times yet) or would take
    /// the group's summed expected run past 16 ms (`command_buffer/group.rs`, `GROUP_RUN`): that
    /// batch leads the next group where its own run is known and within the bound, else goes
    /// alone.
    pub group_ends_long: u64,
    /// Groups that went when the scheduler found no batch left to submit.
    pub group_ends_drained: u64,
    /// Groups that went when the scheduler found no batch ready, with submission checks or host
    /// steps left for its thread to run.
    pub group_ends_checks: u64,
    /// Groups that went when the scheduler found no batch ready, the rest held for events'
    /// values, host steps' gates, binary semaphores or texture decodes.
    pub group_ends_held: u64,
    /// Groups that went at once as a batch of a command buffer not marked to join the next went
    /// last in them, under `CompletionPolicy::Exact` (`CommandBuffer::set_completion_joins_next`):
    /// the marked command buffers' batches before it, and it seen complete at its own end.
    pub group_ends_closed: u64,
    /// Batches submitted (each held in a group counted as it is held) while no earlier batch of
    /// their `VkQueue` was pending: every one submitted before them the host had seen complete
    /// (their queue's timeline at their values), as of the completed value the hazard tracker took
    /// for their plan, so a batch counted here is planned after no work in flight. With the three
    /// fields after this one, every batch, by how far its queue's work ran behind its submission.
    pub batches_with_none_pending: u64,
    /// Batches submitted while one earlier batch of their `VkQueue` was pending.
    pub batches_with_one_pending: u64,
    /// Batches submitted while two or three earlier batches of their `VkQueue` were pending.
    pub batches_with_two_or_three_pending: u64,
    /// Batches submitted while four or more earlier batches of their `VkQueue` were pending.
    pub batches_with_four_or_more_pending: u64,
    /// Batches on a `VkQueue` whose batches may wait in a group (`CompletionPolicy::Grouped`, on
    /// a device whose submissions merge; under `CompletionPolicy::Exact` those of command buffers
    /// marked to join the next) that went in a kernel submission of their own, with no
    /// group held, because their command queue is unshared (`Device::new_command_queue_unshared`)
    /// and the `VkQueue` holds other queues' work back for it (`UnsharedIsolationPolicy::Bounded`):
    /// such a batch never waits in a group. With the eighteen fields after this one and
    /// `batches_grouped`, every batch such a `VkQueue` took with GPU work: each counted once, by
    /// the first reason in their order, so the spread says why a device's batches did not group.
    /// A batch counted here is refused before it could be held, whether or not a group was held:
    /// a group held went first.
    pub batches_alone_unshared: u64,
    /// Such batches that waited for a binary semaphore (`encode_wait_for_vk_semaphore`, a sync
    /// file's wait).
    pub batches_alone_binary_waits: u64,
    /// Such batches that signalled a binary semaphore (`encode_signal_vk_semaphore`, a command
    /// buffer's completion sync file).
    pub batches_alone_binary_signals: u64,
    /// Such batches that bound a sparse update.
    pub batches_alone_sparse: u64,
    /// Such batches whose work a segment cut into pieces, each a `VkSubmitInfo` of its own.
    pub batches_alone_pieced: u64,
    /// Such batches that ended their command buffer at a fence.
    pub batches_alone_fences: u64,
    /// Such batches whose command queue's expected GPU run was unknown: none of its command
    /// buffers completed with GPU times yet.
    pub batches_alone_run_unknown: u64,
    /// Such batches whose command queue's expected GPU run alone was past a group's bound.
    pub batches_alone_run_long: u64,
    /// Batches that could otherwise join, went with no group held, and took the device's first
    /// work, staged copies or a fill, or named a texture over the caller's image: such a batch
    /// goes last in a group's call where one is held (`group_ends_first_work`), never in a group.
    pub batches_alone_first_work: u64,
    /// Batches that could otherwise join and that the thread that committed them submitted alone
    /// (`CommandBuffer::commit_and_submit`), as their `VkQueue` ran at most one earlier
    /// submission not seen complete.
    pub batches_alone_at_commit: u64,
    /// Batches held in a group that went holding them alone (no kernel submission saved), when the
    /// scheduler found no batch ready and their `VkQueue` ran at most one earlier submission not
    /// seen complete, so the group was not kept.
    pub batches_held_alone_one_pending: u64,
    /// Batches held alone whose group was not kept because an unshared queue holds their
    /// `VkQueue` and holds other queues' work back there (`UnsharedIsolationPolicy::Bounded`),
    /// where no group is kept.
    pub batches_held_alone_unshared_hold: u64,
    /// Batches held alone whose group went because a line's head waited for the GPU: at its gate,
    /// or for a signal its submission checks wait for.
    pub batches_held_alone_gpu_waits: u64,
    /// Batches held alone whose group went before the scheduler's thread ran another logical
    /// queue's submission handlers.
    pub batches_held_alone_handlers: u64,
    /// Batches held alone whose group went because a thread waited for their command buffer's
    /// completion.
    pub batches_held_alone_waited_for: u64,
    /// Batches held alone in a group kept while their `VkQueue` ran earlier work, which no batch
    /// joined before one submission was left in flight.
    pub batches_held_alone_kept: u64,
    /// Batches held alone whose group went before a batch that may not join it: of another
    /// `VkQueue`, refused, past the group's bound, or waiting for a signal the group holds.
    pub batches_held_alone_sent_by_next: u64,
    /// Batches held alone whose group went because a switch of storage forms rode on it, or the
    /// device was lost.
    pub batches_held_alone_other: u64,
    /// Batches of command buffers marked to join the next (`CommandBuffer::set_completion_joins_next`)
    /// held alone under `CompletionPolicy::Exact`, whose group went as the scheduler found no batch
    /// ready: no group is kept under `Exact`, so no marked batch waits for a later commit.
    pub batches_held_alone_unkept: u64,
    /// Batches of command queues made with `Device::new_command_queue_unshared` submitted, or
    /// held in a group, under either `CompletionPolicy` and on every device: beside the total of
    /// `batches_with_none_pending` and its three siblings, the share of the device's batches its
    /// unshared queues made.
    pub batches_of_unshared_queues: u64,
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
    /// (`encode_wait_for_vk_semaphore`). With the seven fields after this one, every reason a look
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
    /// records, or at its gate (`heads_waiting_at_their_gate`), or it is queued for the steps'
    /// threads or being recorded there (`scheduler/steps.rs`), which record a step a thread.
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
    /// (`scheduler/order.rs`): Metal tracks a resource across queues in the order the GPU takes
    /// the command buffers, so a batch the runtime holds for its submission checks, or for a host
    /// step whose gate was taken, must not let a later commit's conflicting batch go first. A
    /// batch held for its pipelines' making, or for a step that has not taken its gate, gives way:
    /// it holds a head back only where the two share a fence, as an x86 Mac's GPU may take a later
    /// commit of another queue first; one whose step took its gate, whose uses are not known and
    /// whose step reads known records and ranges on the host, gives way too, but holds a head that
    /// writes those as well. A head of a queue made for copies waits behind either as behind one
    /// that keeps its place (`Device::new_command_queue_for_copies` keeps commit order). This is
    /// the one hold that crosses logical queues and `VkQueue`s, and a batch whose uses are not
    /// all known yet (work a host step or its submission records, `tracker::ordered`'s `None`)
    /// conflicts with every batch that makes a tracked use or names a fence, but for a host
    /// step's execution whose uses `commit` bounded (`Segment::bound`: the resources made
    /// resident, the records, and the encoder's bindings its commands inherit), which conflicts
    /// with what shares those.
    pub heads_held_for_earlier_commits: u64,
    /// Heads found held back from a `VkQueue` that a command queue made with
    /// `Device::new_command_queue_unshared` shares with theirs, because 2 ms or more of other
    /// queues' expected GPU work was in flight there (`unshared_hold.rs`,
    /// `UnsharedIsolationPolicy::Bounded`): the unshared queue's next command buffer goes in
    /// behind about one batch of theirs. While no unshared queue there committed a command
    /// buffer within the last 50 ms, 16 ms or more (`heads_let_go_unshared_idle`). With them, the
    /// looks that held one behind an unshared queue's commit there not yet taken into the
    /// scheduler's order, or on its way (`heads_held_for_unshared_commits`).
    pub heads_held_for_unshared_isolation: u64,
    /// Heads a look let go with 2 ms or more of other queues' expected GPU work in flight on
    /// their `VkQueue`, which would hold them for an unshared queue there
    /// (`heads_held_for_unshared_isolation`), because no unshared queue there committed a command
    /// buffer within the last 50 ms and less than 16 ms was in flight (`unshared_hold.rs`'s idle
    /// budget): a compositor gone quiet holds back only work queued 16 ms deep. The unshared
    /// queue's first command buffer after such a stretch goes in behind about that and one
    /// batch, and the 2 ms bound applies from its commit on.
    pub heads_let_go_unshared_idle: u64,
    /// Of `heads_held_for_unshared_isolation`, the looks that held a head whose room on its
    /// `VkQueue` had come behind an unshared queue's commit there posted before that room came
    /// (`unshared_hold.rs`): one not yet taken into the scheduler's order (a scheduler pass takes
    /// its commits in at its start, so one posted during the pass waits for the next), or one
    /// leading its line on its way, waiting only for its host step, its submission checks or a
    /// signal already submitted to land. The pass that takes it in, or that its step, checks or
    /// landing wake the scheduler's thread for, sends the unshared queue's command buffer first
    /// and the held head next. Commits posted after its room came never hold it, nor one that
    /// waits for anything else.
    pub heads_held_for_unshared_commits: u64,
    /// Heads found held back for their turn on a `VkQueue` two or more command queues hold,
    /// because 2 ms or more of their own queue's expected GPU work was in flight there
    /// (`queue_turns.rs`, `QueueTurnPolicy::Fair`): another queue's command buffer committed later
    /// goes in behind about that much of it and one batch, not its whole backlog. While no other
    /// command queue took a batch there within the last 50 ms, 16 ms or more
    /// (`heads_let_go_turn_idle`). An unshared queue's heads are never held so.
    pub heads_held_for_queue_turns: u64,
    /// Heads a look let go on their turn with 2 ms or more of their own queue's expected GPU work
    /// in flight on their `VkQueue`, which would hold them (`heads_held_for_queue_turns`), because
    /// no other command queue took a batch there within the last 50 ms and less than 16 ms was in
    /// flight (`queue_turns.rs`'s idle budget): a queue alone keeps about 16 ms queued.
    pub heads_let_go_turn_idle: u64,
    /// Times a look left a head that needs a host step waiting at its gate: the step reads what
    /// the GPU may write, so it waits until the batches that last wrote what it reads completed
    /// (`host_step_gates_narrowed`), or else until every `VkQueue` completed what was submitted to
    /// it when the batch was otherwise ready (a predicated loop's check waits for its own
    /// `VkQueue` alone). Of `heads_held_for_host_steps`.
    pub heads_waiting_at_their_gate: u64,
    /// Command buffers made `Scheduled` before their last batch was submitted: a look found their
    /// queue's head waiting at its host step's gate (`heads_waiting_at_their_gate`), and every
    /// batch of theirs from it on waiting for nothing else (the step's batch first, then batches
    /// that need no host step, whose event and semaphore waits are met), every earlier command
    /// buffer of the queue `Scheduled`. Metal's are `Scheduled` once their dependencies are
    /// resolved, a fraction of a millisecond after the commit; the runtime's step waits there for
    /// the GPU, as Metal's GPU would. Their submission handlers run then, their kernel end is that
    /// time, and their batches go once the step recorded (`scheduler/mod.rs`, **Scheduled**).
    pub scheduled_behind_host_steps: u64,
    /// Host steps whose gate, as it was taken, left out work submitted before it: each segment
    /// whose step waits for the GPU executes indirect commands whose records and range Metal
    /// tracks, so the step waits for the batches that last wrote those alone, not for every
    /// `VkQueue`'s last submission (the batches after the writer on its queue, and other queues'
    /// work that writes neither). A command buffer's executions behind one writer then cost one
    /// GPU round trip between them, not one each. Whether a gate leaves work out depends on what
    /// was submitted, and seen complete, when it was taken.
    pub host_step_gates_narrowed: u64,
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
    /// thread, or its committing thread's own pass submitted alone a batch that may wait in a
    /// group and then woke the thread, as that batch's hand-off would have
    /// (`CommandBuffer::commit_and_submit`). Each waking here is a notify of its condition variable, or a signal of its nudge
    /// where it slept on the GPU, made only where it slept, and once a sleep: a later waker
    /// before the thread ran makes none. With the completion and host wakings, at most one a
    /// sleep (`scheduler_sleeps`).
    pub scheduler_wakes_commit: u64,
    /// Times a completion woke it: what a head waited for from the GPU (its gate's values
    /// completed, an event value its checks wait for landed) was met, seen by a completion turn or
    /// by its own wait on the GPU, or the device was lost.
    pub scheduler_wakes_completion: u64,
    /// Times a host signal woke it: an event's value set by the host, a pipeline a host step waits
    /// for made, a texture decode finished, a command buffer encoded again after a rebind, the
    /// rest of a pass a thread about to block or a committing thread left it, a lost device.
    pub scheduler_wakes_host: u64,
    /// Sleeps that ended at the thread's longest wait with nothing having woken it. Each, with a
    /// batch held, is followed by a pass.
    pub scheduler_sleep_slices: u64,
    /// Of the passes after those sleeps, the ones that let a batch go (submitted, held in a
    /// group, or failed): what it waited for was met, and the wake that should have come with it
    /// did not (`scheduler/mod.rs`). 0 where every wake comes.
    pub scheduler_slice_rescues: u64,
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
    /// run (no submission or scheduled handler, no switch of a storage form), taken in commit
    /// order, a held head passed over only by another queue's that waits behind no earlier commit,
    /// as the thread's own pass picks them; the rest it leaves to the thread, so this is below
    /// `scheduler_passes_by_waiters` where a command buffer needs it.
    pub batches_submitted_by_waiters: u64,
    /// Passes a thread made over the scheduler's order itself right after its commit
    /// (`CommandBuffer::commit_and_submit`, `scheduler/mod.rs`): one where no pass was under
    /// way. A commit whose pass found one under way leaves its batches to it, uncounted here.
    pub scheduler_passes_at_commit: u64,
    /// Of those passes, the ones that ended right after they submitted their own command
    /// buffer's last batch, which has submission handlers, with batches left in the order: the
    /// committing thread runs those handlers once the lock is free, waiting for no later batch's
    /// submission, and leaves the rest to the scheduler's thread, which it wakes first. Each such
    /// batch then waits for that wake and that thread's pass, not for those handlers, and may be
    /// submitted before or while they run.
    pub scheduler_passes_at_commit_ended_early: u64,
    /// Commits of `CommandBuffer::commit_and_submit` whose committing thread made no pass of its
    /// own, as another pass had taken its commit in first (`scheduler/mod.rs`, `LeftAtCommit`):
    /// its batches go to the scheduler's thread, or to the pass that took them.
    pub scheduler_at_commit_left_taken_in: u64,
    /// Those whose committing thread made no pass, as a pass was under way, which takes the
    /// commit in (the scheduler thread's, while it passes again for the commits that keep coming).
    pub scheduler_at_commit_left_passing: u64,
    /// Passes at commit (`scheduler_passes_at_commit`) that submitted nothing, by why the
    /// committed command buffer's first batch was left: none of its batches was in the order any
    /// more (another thread's pass took them in and on).
    pub scheduler_at_commit_left_gone: u64,
    /// An earlier batch of its own queue led its queue's line.
    pub scheduler_at_commit_left_behind_its_queue: u64,
    /// It waited behind an earlier commit of another queue.
    pub scheduler_at_commit_left_behind_another_queue: u64,
    /// It waits for an event value no submitted signal reached yet.
    pub scheduler_at_commit_left_event_waits: u64,
    /// Its submission checks, which the committing thread did not take.
    pub scheduler_at_commit_left_checks: u64,
    /// It has a host step for the scheduler's thread to record.
    pub scheduler_at_commit_left_host_steps: u64,
    /// It waits for a binary semaphore whose signal was not submitted yet.
    pub scheduler_at_commit_left_binary_waits: u64,
    /// It is held back for an unshared queue (`unshared_hold.rs`).
    pub scheduler_at_commit_left_unshared: u64,
    /// It waits for its turn on a `VkQueue` it shares (`queue_turns.rs`).
    pub scheduler_at_commit_left_turns: u64,
    /// Its `VkQueue`'s batches' `vkQueueSubmit` calls are slow, each of the last 15 100 us or
    /// more, so the scheduler's thread makes them and the committing thread goes on
    /// (`device/submit_calls.rs`).
    pub scheduler_at_commit_left_slow_calls: u64,
    /// Anything else: a rebind encodes it again, a decode it reads is queued, a switch of storage
    /// forms rides on it, a group holds it or was held, or its submission was put back.
    pub scheduler_at_commit_left_other: u64,
    /// The batches those passes submitted: those ready with nothing for the scheduler's thread to
    /// do for them (no host step, no group, no switch of a storage form, no submission check the
    /// committing thread may not run), their submission handlers run by the committing thread,
    /// taken in commit order, a held head passed over only by another queue's that waits behind no
    /// earlier commit, as the thread's own pass picks them; the rest it leaves to the thread.
    pub batches_submitted_at_commit: u64,
    /// The batches whose submission checks those passes ran on the committing thread, each just
    /// before it submitted the batch (`CommandBuffer::set_submission_check`): the first queue's
    /// head in commit order ready but for its checks and behind no earlier commit, every event
    /// value it waits for reached on the device or by the host. A head whose checks wait for a GPU
    /// signal to land, or whose checks the scheduler's thread took first, is left to that thread,
    /// uncounted here, and holds back no later commit of another queue but those behind it.
    pub batches_checked_at_commit: u64,
    /// The program's calls of `CommandBuffer::commit`, `commit_and_submit` and `commit_failed`,
    /// which the commit times below sum over (`commit_times.rs`). A call a submission handler
    /// makes inside another is one of them too, its time counted in both.
    pub commit_calls: u64,
    /// Nanoseconds those calls held the program's thread, each from its start to its return.
    /// The phases below are parts of it, each timed apart from the others, so they sum to at
    /// most this; the rest is the commit's checks and bookkeeping (its hazard and residency
    /// lists, its copies in, its posting to the scheduler).
    pub commit_nanos: u64,
    /// Of `commit_nanos`, lowering on the calling thread the encoders' calls a render encoder
    /// left to the device's lowering thread (`RenderLoweringPolicy::Worker`) and its end, and
    /// letting go of the draw its last render encoder kept for the next: about nothing under
    /// `RenderLoweringPolicy::AtCall`, which lowered each call inside it.
    pub commit_lowering_nanos: u64,
    /// Of `commit_nanos`, recording the work its encoders left to the commit, or settling it for
    /// its host steps (`Deferred::prepare`): an indirect command buffer's execution, a render pass
    /// whose draws' pipelines were being made, an indirect dispatch's variants. Its waits for
    /// pipelines are `commit_pipeline_wait_nanos`, apart from this.
    pub commit_recording_nanos: u64,
    /// Of `commit_nanos`, that recording's waits for the pipelines being made of a render pass
    /// whose executions read the CPU's records as they stand at the commit, up to the command
    /// buffer's bound under `PipelineWaitPolicy::Bounded`.
    pub commit_pipeline_wait_nanos: u64,
    /// Of `commit_nanos`, `commit_and_submit`'s wait to take the scheduler's lock for its pass at
    /// commit, and for each pass again after a submission check it ran. Where a pass is under way
    /// the call makes none and waits for none: that pass takes the commit in.
    pub commit_lock_wait_nanos: u64,
    /// Of `commit_nanos`, `commit_and_submit`'s pass at commit under the scheduler's lock: taking
    /// in what was posted, ordering it and picking what it submits (`Scheduler::submit_at_commit`),
    /// its submissions apart.
    pub commit_pass_nanos: u64,
    /// Of `commit_nanos`, that pass's submissions (`CommandBufferShared::submit_batch`: the
    /// batch's barriers and `vkQueueSubmit`), and the submission checks and submission handlers
    /// the call ran for them, with the commits a handler made inside it.
    pub commit_submit_nanos: u64,
    /// Footprints of batches the scheduler built to order them across queues
    /// (`scheduler/order.rs`), one each time: each batch's is built where a pass first compares
    /// it with another queue's and kept, and built again only where what it is built from
    /// changed since (its command buffer's recordings or batches, or an answer the summary one of
    /// its segments takes turns on, as a host step records the segment), or a segment of it takes
    /// uses its work knows, which are built at each look (`CommandBufferShared::footprint`).
    pub footprints_built: u64,
    /// Asks of a batch's footprint by the scheduler that found its command buffer's state locked
    /// (by a host step's draws, a commit's work) and waited for the lock
    /// (`CommandBufferShared::footprint`): only a build takes it, a kept footprint is read
    /// without it.
    pub footprint_lock_waits: u64,
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
    /// Times a queue's waiter slept on its condition variable (`command_buffer/completion.rs`):
    /// with no batch of its queue pending until it was woken, else until it was woken or a slice
    /// ran out ([`completion_waiter_parks_timed_out`](Self::completion_waiter_parks_timed_out)).
    /// An idle device's waiters sleep once and stay asleep.
    pub completion_waiter_parks: u64,
    /// Of those, the sleeps that ended at their slice, with a batch of the waiter's queue pending.
    pub completion_waiter_parks_timed_out: u64,
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
    /// Command queues made past the round robin's `VkQueue`s, unshared ones among them, each on a
    /// `VkQueue` an earlier command queue of the device took too: the command queues made, live or
    /// gone, less the `VkQueue`s the device lends them (`queue.rs`, `device/create.rs`). Zero on a device
    /// with a `VkQueue` for each. A command queue takes, as it is made, the round robin's
    /// `VkQueue` the fewest live command queues hold (`queue_share.rs`), so its command buffers
    /// wait behind another live queue's, where Metal's would not, only where every `VkQueue` holds
    /// one: a program that keeps more command queues live than the device lends (at most
    /// `command_queues_live_most` of its own). Under `QueueSharingPolicy::Adaptive` queues also
    /// share by moving, after their creation (`queue_joins`).
    pub queues_sharing_a_vk_queue: u64,
    /// The most command queues the program held at once on the device: those its
    /// `new_command_queue` and kin made, unshared and presenting ones included, the runtime's
    /// own left out. A high-water mark: what a program keeps live, against the round robin's
    /// `VkQueue`s (`DeviceOptions::queue_limits`).
    pub command_queues_live_most: u64,
    /// Unshared command queues (`Device::new_command_queue_unshared`) that took, as they were
    /// made, a `VkQueue` another command queue held: the round robin had none that no command
    /// queue holds (a device of one graphics queue, or more queues live than `VkQueue`s), so
    /// their command buffers may wait behind another queue's (`queue_share.rs`). Zero where
    /// each unshared queue took a `VkQueue` of its own.
    pub unshared_queues_sharing_a_vk_queue: u64,
    /// Command queues that moved to another `VkQueue` of their family as they made a command
    /// buffer while they held no other uncompleted one (`QueueSharingPolicy::Adaptive`, on a
    /// device whose GPU idles between `VkQueue`s: `queue_share.rs`). Zero where the device does
    /// not share.
    pub queue_moves: u64,
    /// Of `queue_moves`, those onto a `VkQueue` with another command queue's work in flight.
    pub queue_joins: u64,
    /// Command queues marked long: one of their command buffers ran on the GPU for over 2 ms, by
    /// its own timestamps, so no queue joins their `VkQueue` from then on.
    pub queues_gone_long: u64,
    /// Long command queues that settled, at their next idle point: on an idle `VkQueue` no long or
    /// unshared queue holds (a move, counted in `queue_moves` too), or where they were.
    pub long_queue_settles: u64,
    /// Of `queue_moves`, those off a `VkQueue` an unshared queue
    /// (`Device::new_command_queue_unshared`) holds.
    pub queue_moves_off_unshared: u64,
    /// Choices of a command queue that passed over a busy `VkQueue` whose oldest batch was young,
    /// which it might have joined, because an unshared queue holds it.
    pub joins_withheld_from_unshared: u64,
    /// Choices of a command queue at an idle point that found no other `VkQueue` of its family
    /// with batches in flight: nothing to join. Work committed but not yet submitted is not in
    /// flight, so on a busy host a queue choosing just after another's commits can find so.
    pub picks_with_none_busy: u64,
    /// Choices of a command queue at an idle point that joined no `VkQueue` and passed over
    /// another with batches in flight, which no long or unshared queue holds, because its oldest
    /// batch had run 0.5 ms or longer (or for a time not known): a queue joining it would likely
    /// wait behind a long command buffer. Beside another process's GPU work, or on a host too busy
    /// to see batches complete at once, a busy `VkQueue` can look so.
    pub joins_withheld_from_old_heads: u64,
    /// Command buffers that waited on their `VkQueue` behind another command queue's batch: their
    /// first batch went there (submitted, or held for one submission with later batches) while
    /// the latest batch of another command queue's before it there was not seen complete
    /// (`queue_wait.rs`). That batch went there before this command buffer's first batch did, and
    /// runs first; a command buffer the scheduler held back (an event's value, a binary semaphore)
    /// can be passed so by one committed after it. Each command buffer counts once, at its first
    /// batch; the runtime's own submissions (staging, checks) neither count nor are waited behind.
    /// Counted wherever command queues share a `VkQueue`: RADV's, ANV's and lavapipe's one
    /// graphics queue, unshared queues' included, or past the round robin's `VkQueue`s.
    pub command_buffers_behind_other_queues: u64,
    /// Of those, the ones whose wait lasted over 16 ms: from the host time of their first batch's
    /// submission (or hold) to the host's observation of that other batch's completion. An upper
    /// bound of the wait on the GPU: the host sees a completion after the GPU's end.
    pub command_buffers_behind_other_queues_over_16ms: u64,
    /// Of those, the ones whose wait, measured so, lasted over 33 ms.
    pub command_buffers_behind_other_queues_over_33ms: u64,
    /// The longest of those waits, measured so, in nanoseconds. A wait counts here, and over 16
    /// or 33 ms, once the host saw it end.
    pub wait_behind_other_queues_longest_nanos: u64,
    /// Of `command_buffers_behind_other_queues`, those of command queues made with
    /// `Device::new_command_queue_unshared`.
    pub unshared_command_buffers_behind_other_queues: u64,
    /// Of `command_buffers_behind_other_queues_over_16ms`, those of unshared command queues.
    pub unshared_command_buffers_behind_other_queues_over_16ms: u64,
    /// Of `command_buffers_behind_other_queues_over_33ms`, those of unshared command queues.
    pub unshared_command_buffers_behind_other_queues_over_33ms: u64,
    /// The longest wait of an unshared command queue's command buffer behind another command
    /// queue's batch, in nanoseconds, measured as `wait_behind_other_queues_longest_nanos`.
    pub unshared_wait_behind_other_queues_longest_nanos: u64,
    /// Of the waits counted in `unshared_command_buffers_behind_other_queues_over_16ms`, the
    /// batches in flight ahead of each on its `VkQueue` as its first batch went there (timeline
    /// values submitted past the one the host saw complete), summed: divided by that count, how
    /// many batches such a wait was behind.
    pub unshared_waits_over_16ms_batches_ahead: u64,
    /// The most batches in flight ahead of one of those waits, counted so.
    pub unshared_waits_over_16ms_most_batches_ahead: u64,
    /// The longest run each of those waits waited for, summed, in nanoseconds: a run is the
    /// host's view of how long the oldest batch in flight on the `VkQueue` was so, from the
    /// completion seen before it (or its submission onto a `VkQueue` the host saw idle) to the
    /// completion that ended it. One batch's GPU work and the host's lateness to see it end, or
    /// several seen complete at once.
    pub unshared_waits_over_16ms_longest_runs_nanos: u64,
    /// Of those waits, the ones whose longest run alone was over 16 ms: behind one long batch,
    /// rather than many short ones.
    pub unshared_waits_over_16ms_behind_one_long_run: u64,
    /// Runs in this process of the device's check of its GPU's idling between `VkQueue`s
    /// (`queue_switch_check.rs`), whatever they answered.
    pub queue_switch_checks_run: u64,
    /// Devices whose verdict of that check was read from the record of an earlier one of the same
    /// driver, device and runtime: no check ran for them.
    pub queue_switch_checks_recalled: u64,
    /// Runs in this process of the device's submission checks (`submit_check.rs`: whether
    /// `VkSubmitInfo`s that signal nothing merge, whether a held host-pointer import adds to every
    /// submission's cost), whatever they answered.
    pub submit_checks_run: u64,
    /// Devices whose submission checks' verdicts were read from the record of an earlier run of
    /// the same driver, device, kernel and runtime: no check ran for them.
    pub submit_checks_recalled: u64,
    /// The `vkQueueSubmit` calls the submission checks made, which `queue_submissions` leaves out:
    /// theirs, and on a device that relays its queues' failures the relay's mark after each of
    /// their uses of a queue.
    pub submit_check_calls: u64,
    /// The device's own checks' work, on their threads, which the counts above and
    /// `object_churn` leave out (`DeviceCheckWork`).
    pub device_checks: DeviceCheckWork,
    /// Devices of `QueueSharingPolicy::Adaptive` with two `VkQueue`s of one family to share that
    /// do not share, because their command buffers write no calibrated GPU timestamps the host
    /// reads (`gpu_times_from_host`, a driver whose timestamps wait for every queue's, or none):
    /// a queue is known long only by its command buffers' own GPU time. No check ran for them.
    pub queue_sharing_untimed: u64,
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
    /// compute encoder) for the encoder's earlier commands (`tracker::EncoderUses`), or of a
    /// compute, blit or render encoder for the earlier encoders of the group it joined (a render
    /// encoder's one, from everything of theirs, before its first command; `encoder_fold.rs`): on
    /// MoltenVK each ends the Metal encoder, and the next waits for the one before (`tracker.rs`,
    /// **Inside an encoder**, orders a blit's untracked uses by key and range).
    pub command_barriers: u64,
    /// `VkCommandBuffer`s an encoder ended before a command that uses an image whose memory a
    /// command before it there wrote through a buffer, the command going on in a new one, on a
    /// device whose driver's image uses miss such writes inside one
    /// (`DriverFact::ImageUsesMissBufferWritesInOneCommandBuffer`: NVIDIA's on the RTX 3090);
    /// and render encoders' passes that continued no render pass instance the encoder before them
    /// left open because an image they use meets bytes the group's encoders wrote into its
    /// `VkCommandBuffer`, each beginning its own in a `VkCommandBuffer` of its own
    /// (`encoder_fold.rs`, **Continued passes**). None elsewhere.
    pub command_buffers_split_at_aliases: u64,
    /// Commands the encoders Metal orders (blit encoders, serial compute encoders) took into the
    /// order their later commands compare with, counted as each encoder's first encoding ends
    /// (`tracker::EncoderUses`): every command after the first of an encoder's part, and the first
    /// once a later command of the part orders after it. None for an encoder of one command
    /// (a program that opens an encoder for each dispatch), whose command orders after nothing and
    /// before nothing, nor for an encoder Metal leaves unordered.
    pub commands_ordered_inside: u64,
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
    /// Render encoders that opened from the kept draw their command buffer's last render encoder
    /// carried into them, with that encoder's bind state and settle (`render/carried.rs`), or,
    /// where they hold none, the last render encoder of a command buffer committed on their queue
    /// (`kept_draws_carried_across`): the encoder's first draw of it is rebound
    /// (`draws_rebound_carried`). An encoder ended with a kept draw of its pass's first instance
    /// carries it, with those of other pipeline states it set aside (`kept_draws_carried_aside`),
    /// a draw that reads the texture of a color attachment of its pass through its copy among
    /// them (`texture/attached_reads.rs`), but not one that reads it through a slot narrowed to
    /// levels apart from the attached ones (`kept_draws_uncarried_attached`), nor one whose
    /// depth/stencil state fails every fragment.
    pub kept_draws_carried: u64,
    /// Those of them that opened from the last render encoder of a command buffer committed on
    /// their queue, which the queue held, with those of the command buffers committed just before
    /// it (`queue::CARRIES`), for a render encoder of a later command buffer that held none of its
    /// own, the latest that fits its pass (`render/carried.rs`): the first draw of each of its
    /// states is rebound, and retains into its command buffer what every step retains.
    pub kept_draws_carried_across: u64,
    /// Kept draws carried that were let go before a render encoder started from them because
    /// another call of the command buffer came first: another encoder's opening, an event, or
    /// any other call that settles the last render encoder's end, but the commit, which hands it
    /// to the queue (`kept_draws_carry_dropped_commit`).
    pub kept_draws_carry_dropped_between: u64,
    /// Kept draws carried that were let go because the next render encoder's pass is another's:
    /// another shape or compatible render pass, another kind of attachment, memoryless colors, a
    /// visibility buffer, a rasterization-rate map, color groups, or an encoder that encodes an
    /// ended one again.
    pub kept_draws_carry_dropped_layout: u64,
    /// Kept draws carried that were let go because what they read that changes with no call
    /// changed: a no-copy buffer moved, the device's sampler IDs' slots changed, or the next
    /// encoder's residency is of another generation.
    pub kept_draws_carry_dropped_under: u64,
    /// Kept draws carried that their queue held from a command buffer's commit and let go at a
    /// later command buffer's commit, which handed it another where it held its most
    /// (`queue::CARRIES`), the one held longest: no render encoder took it between.
    pub kept_draws_carry_dropped_commit: u64,
    /// Kept draws of other pipeline states than the last draw's that render encoders opened from
    /// a carried kept draw took with it (`kept_draws_carried`, `render/carried.rs`): each set
    /// aside with its state's bind state in the ended encoder, kept there still, carrying as the
    /// last draw's does and having read what changes with no call as it did; the encoder's first
    /// draw of each state is rebound against its own (`draws_rebound_carried`). An encoder
    /// carries every state its bind state set aside, seven at most. A carried state the encoder
    /// never draws is an opening's work for nothing: these and `kept_draws_carried` less
    /// `draws_rebound_carried` and `draws_carry_stopped`.
    pub kept_draws_carried_aside: u64,
    /// Kept draws a render encoder ended with, its last draw's or those set aside with the states
    /// of other pipeline states, that would carry into its command buffer's next render encoder
    /// but read a texture a color attachment of their pass renders through a sampled slot narrowed
    /// to levels apart from the attached ones (`render/attached_levels.rs`). Let go, so the next
    /// encoder's first draw of each one's state takes every step (`draws_none_kept`). One that
    /// reads it through its copy carries (`kept_draws_carried`).
    pub kept_draws_uncarried_attached: u64,
    /// Render encoders that ended carrying no kept draw into their command buffer's next render
    /// encoder (`render/carried.rs`), by the first reason that held, in this order: they recorded
    /// no draw.
    pub encoders_uncarried_drawless: u64,
    /// They drew but kept no draw at their end: the last draw kept nothing (`draws_unkept`), or
    /// a call after the last kept draw let every kept draw go.
    pub encoders_uncarried_none_kept: u64,
    /// Their bind state held no settle of the last draw's state.
    pub encoders_uncarried_unsettled: u64,
    /// Their current instance split into pieces: draws recorded before an execution of render
    /// commands or a draw of a pipeline being made. A split of their pass into instances carries
    /// (`render/carried.rs`).
    pub encoders_uncarried_split: u64,
    /// Their pass is drawn in color groups, over a rasterization-rate map, or at custom sample
    /// positions.
    pub encoders_uncarried_pass: u64,
    /// They encoded an ended encoder again, or recorded a draw again into another instance.
    pub encoders_uncarried_replay: u64,
    /// Their pass executed render commands, drew through a pipeline still being made, or
    /// dispatched tiles.
    pub encoders_uncarried_commands: u64,
    /// Their pass tests the stencil or depth in the shader, or holds such a test's stand-in.
    pub encoders_uncarried_shader_tests: u64,
    /// Their pass holds a fragment or tile imageblock.
    pub encoders_uncarried_imageblock: u64,
    /// A visibility result was active at their end.
    pub encoders_uncarried_visibility: u64,
    /// Their current instance renders a color through a view of another format, draws views, or
    /// holds storage attachments.
    pub encoders_uncarried_instance: u64,
    /// Their last kept draw does not rebind: a visible-function table, a push hole, a use beyond
    /// its plan's, a set-0 write that holds a keeper, or a recording of more than one Vulkan draw.
    pub encoders_uncarried_unrebound: u64,
    /// Their last kept draw holds a depth/stencil state that fails every fragment.
    pub encoders_uncarried_failing_state: u64,
    /// Render encoders that drew but ended with no kept draw (`encoders_uncarried_none_kept`), by
    /// what a kept draw of theirs would have come to, carried into their command buffer's next
    /// render encoder (`render/carried.rs`), the first that held: their end is not one a kept
    /// draw carries from, first because their current instance split into pieces
    /// (`encoders_uncarried_split`'s reason).
    pub encoders_none_kept_next_split: u64,
    /// Their end is not one a kept draw carries from, first for another of the reasons
    /// `encoders_uncarried_*` count after `encoders_uncarried_none_kept`.
    pub encoders_none_kept_next_ends: u64,
    /// Another call of the command buffer came before its next render encoder's opening: another
    /// encoder's opening, an event, or any other call but the commit.
    pub encoders_none_kept_next_between: u64,
    /// Their command buffer committed with no render encoder after them.
    pub encoders_none_kept_next_last: u64,
    /// The next render encoder's pass is another's (as `kept_draws_carry_dropped_layout` counts),
    /// or it turned a color's texture to its other face.
    pub encoders_none_kept_next_layout: u64,
    /// What changes with no call changed by the next render encoder's opening (as
    /// `kept_draws_carry_dropped_under` counts).
    pub encoders_none_kept_next_under: u64,
    /// The next render encoder's pass, and what changes with no call, were as they were: a kept
    /// draw of theirs would have carried into it.
    pub encoders_none_kept_next_fits: u64,
    /// Render encoders that drew and whose last draw call drew nothing (no vertex, index, patch,
    /// instance or mesh threadgroup): it keeps nothing and lets every kept draw go, as a draw
    /// that keeps nothing does (`render/draw_again.rs`), so they end with no kept draw
    /// (`encoders_uncarried_none_kept`).
    pub encoders_last_draw_empty: u64,
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
    /// Render encoders' first instances recorded into the render pass instance the render
    /// encoder before them left open, with no end, barrier or begin between: one Vulkan render
    /// pass instance for both encoders' draws (`encoder_fold.rs`, **Continued passes**). None
    /// of the begun fields counts them.
    pub render_passes_continued: u64,
    /// Splits of render encoders' passes, each a new render pass instance after one that
    /// recorded, or that runs the pass's load actions alone (`render.rs`, **Instances**), at a
    /// memory barrier of the program's (`fence.rs`).
    /// With the ten fields after this one, every split, by its first cause in this order.
    pub instance_splits_barrier: u64,
    /// At a tile dispatch, or for the store actions after the pass's last one.
    pub instance_splits_tile_dispatch: u64,
    /// At a draw ordered after draws that reached the tile threadgroup memory.
    pub instance_splits_tile_memory: u64,
    /// At a draw that reaches interlocked storage an earlier draw of the instance wrote, on a
    /// driver whose such draws race under load (`DriverFact::InterlockedDrawsRaceUnderLoad`).
    pub instance_splits_interlocked_writes: u64,
    /// At a draw that samples a color attachment's texture after a barrier inside the instance,
    /// or a color the pass clears after a barrier before its first draw.
    pub instance_splits_attached_read: u64,
    /// At a draw that reads through snapshots apart (those that lead its own instance among them,
    /// `reading_draws_snapshotted_leading`), or binds a color held in the feedback loop layout.
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
    /// Pieces past the first that batches' work went to their `VkQueue`s in, each a
    /// `VkSubmitInfo` of its own that signals a semaphore, so a driver that merges only
    /// submissions that signal nothing runs each as a kernel submission of its own: a
    /// rate-mapped pass's executed commands cut where the work recorded since the last cut, its
    /// region draws' vertices and the draws themselves, reached a bound (`rate_mapped_pass.rs`,
    /// **Pieces**).
    pub rate_map_pieces: u64,
    /// Ranges past the first that executed commands' draws in rate-mapped passes were drawn as,
    /// each a range of the draw's indices drawn as a draw of its own, so a piece may begin between
    /// them: a draw that alone would hold more than a piece's work (`rate_mapped_pass.rs`,
    /// **Pieces**).
    pub rate_map_index_ranges: u64,
    /// The pieces of rate-mapped passes that execute commands whose GPU time was read from
    /// timestamps written before and after each (none where the device writes no timestamps or
    /// its driver's timestamps wait for every queue's): every piece of such a pass, its first
    /// included, and a pass recorded in one piece.
    pub rate_map_pieces_timed: u64,
    /// Their GPU time, in nanoseconds.
    pub rate_map_piece_gpu_nanos: u64,
    /// The longest of them, in nanoseconds: a bound on the time of each kernel submission they
    /// ran in, where each ran in one of its own.
    pub rate_map_piece_gpu_nanos_most: u64,
    /// Probes host steps submitted and waited for before rate-mapped passes, each of the draws of
    /// a pass's executed commands, to fan the draws' triangles out to the map's regions
    /// (`rate_mapped_pass.rs`, **Fan-out**; `RateMapRegionPolicy::FanOut`): one a host step,
    /// however many parts it was submitted in
    /// ([`rate_map_probe_parts`](Self::rate_map_probe_parts)).
    pub rate_map_probes: u64,
    /// Their GPU time, in nanoseconds, read from timestamps written around each part (none where
    /// the device writes no timestamps or its driver's timestamps wait for every queue's).
    pub rate_map_probe_gpu_nanos: u64,
    /// The host's wait for them, in nanoseconds on the host's clock, from each one's last part's
    /// submission until the host saw every part complete: what is left then of their GPU time
    /// ([`rate_map_probe_gpu_nanos`](Self::rate_map_probe_gpu_nanos)), after the host recorded
    /// the parts while the GPU ran those before, the submission's way to the device and the
    /// completion's back, and the queue's work ahead of them. Part of
    /// [`host_step_nanos`](Self::host_step_nanos).
    pub rate_map_probe_wait_nanos: u64,
    /// The parts those probes were submitted in, each a `VkSubmitInfo` of its own with a fence and
    /// a render pass instance for each instance whose commands it probes, submitted once the host
    /// recorded it, so the GPU runs it while the host records the next (`render/executions.rs`,
    /// `DeferredPass::probe_parts`: at least 32 commands a part, at most 8 parts a probe).
    pub rate_map_probe_parts: u64,
    /// Render passes over a rasterization-rate map that a host step recorded without fanning
    /// their triangles out, so probing nothing (`render/executions.rs`,
    /// `DeferredPass::fans_out`): the device's policy or features, a memoryless attachment, or a
    /// shadowed texture the pass uses. Their commands are drawn region by region.
    pub rate_map_steps_unfanned: u64,
    /// Such passes a host step fanned out that read no command to probe: each execution over the
    /// map of an empty range, over a shadowed texture, or recorded in its GPU route with no
    /// command that draws ([`rate_map_gpu_routes`](Self::rate_map_gpu_routes)).
    pub rate_map_steps_nothing_to_probe: u64,
    /// Such passes a host step fanned out that read commands to probe and probed no draw: none of
    /// them drew (empty or reset commands, or records that run nothing for a fault), each
    /// execution's commands recorded as several runs or out of its instance's shape, or a part of
    /// the probe failed. With [`rate_map_probes`](Self::rate_map_probes) and these two, every
    /// rate-mapped pass a host step recorded is counted once.
    pub rate_map_steps_probing_none: u64,
    /// Executions of such passes a host step recorded in their GPU route
    /// (`render/executions/gpu_route.rs`): an execution whose records the GPU may write, and
    /// which takes its merged GPU form but for the pass's rate map, has its records decoded on
    /// the GPU and its commands' draw arguments read back, so the host reads none of its records
    /// ([`indirect_records_read`](Self::indirect_records_read)). Its commands are probed and
    /// routed as the records the host reads are.
    pub rate_map_gpu_routes: u64,
    /// The host's wait for those routes' prework (the preprocess and the merge passes), in
    /// nanoseconds on the host's clock, from its submission until the host saw it complete, one
    /// submission a host step for all of the pass's routes. Part of
    /// [`host_step_nanos`](Self::host_step_nanos).
    pub rate_map_gpu_route_wait_nanos: u64,
    /// The draws they probed that they flagged: a triangle crosses `w = 0` or touches more
    /// regions than the fan-out's copies. Each is drawn in parts
    /// ([`rate_map_parted_draws`](Self::rate_map_parted_draws)), or region by region whole where
    /// it cannot be (`render/fan_out.rs`, **Parts**).
    pub rate_map_flagged_draws: u64,
    /// Their triangles the probes flagged, each of each instance once: against
    /// [`rate_map_parted_region_triangles`](Self::rate_map_parted_region_triangles), how many
    /// of the triangles a draw in parts draws region by region, its runs of blocks that hold a
    /// flagged one, are flagged themselves (a flagged draw drawn whole region by region counts
    /// its flagged triangles here, its triangles there not).
    pub rate_map_flagged_triangles: u64,
    /// Executed commands' draws in rate-mapped passes drawn once, fanned out to the regions their
    /// triangles touch.
    pub rate_map_fanned_draws: u64,
    /// Of them, those the probe marked no triangle of too wide for few copies, drawn through the
    /// variant of `bind::fan_out::FEW_COPIES` copies (`RegionFanOut::DrawFew`).
    pub rate_map_fanned_few_draws: u64,
    /// Flagged draws drawn in parts: their runs of blocks that hold a triangle the probe flagged
    /// region by region, the triangles before, between and after them fanned out
    /// (`render/fan_out.rs`, **Parts**).
    pub rate_map_parted_draws: u64,
    /// Their triangles drawn region by region.
    pub rate_map_parted_region_triangles: u64,
    /// Their parts drawn region by region, each a run of blocks, at most
    /// four a draw: each records the plan's region draws.
    pub rate_map_region_runs: u64,
    /// `VkPipeline`s of probes' and fanned-out draws' variants that `commit` began making on the
    /// device's pool, settling what the host step that records the pass compiles, and that a
    /// host step fanning the pass out found made already (`rate_mapped_pass.rs`, **Fan-out**).
    pub rate_map_fan_out_pipelines_ready: u64,
    /// Those a host step fanning the pass out found still being made, and waited for.
    pub rate_map_fan_out_pipelines_waited: u64,
    /// Region tables uploaded to command buffers, each by an encoder's first fan-out draw or probe
    /// that binds it: one for a host step's probe and one for its fanned-out draws, none for
    /// `commit`'s settle, which records nothing (`rate_mapped_pass.rs`, **Fan-out**).
    pub rate_map_fan_tables_uploaded: u64,
    /// Draws in rate-mapped passes drawn region by region through their region-copy variant, on a
    /// device that fans the pass's triangles out: its vertex function puts each vertex in its
    /// region as a fanned-out draw's geometry stage puts a copy's, each region through a viewport
    /// over the render area and its own scissor (`rate_mapped_pass.rs`, **Fan-out**): the
    /// encoder's own draws, executed
    /// commands the probe did not probe or whose host step does not probe, and a flagged draw's
    /// runs or the whole of one drawn so.
    pub rate_map_region_copy_draws: u64,
    /// The recordings they made: one per region, or per as many regions as one multi-draw
    /// command draws where the draw is direct and the device has the multi-draw commands.
    pub rate_map_region_copy_recordings: u64,
    /// The regions the runs of flagged draws' blocks drawn through their region-copy variant
    /// were drawn into, summed over the runs (each recording of a run cut into index ranges
    /// counted apart): the regions each run's blocks' bounds reach, as the probe found them
    /// (`bind::fan_out::PROBE_BOUNDS_AT`).
    pub rate_map_run_region_draws: u64,
    /// Those runs whose bounds reached every region of the plan: their triangles reached every
    /// bucket of segments of each axis (a triangle with some `w` not positive beside a positive
    /// one by its part where `w` is positive), or the probe's blocks ended before the run's.
    pub rate_map_runs_every_region: u64,
    /// The triangles the probes flagged that have some `w` not positive beside a positive one,
    /// among `rate_map_flagged_triangles`.
    pub rate_map_straddling_triangles: u64,
    /// The vertex work of the draws through their region-copy variant into every region of the
    /// table (every one of
    /// [`rate_map_region_copy_draws`](Self::rate_map_region_copy_draws) but a flagged draw's
    /// runs): each recording's vertices or indices times its instances times the regions it
    /// draws, the vertex function's runs it asks for (an indirect draw's counted as none).
    pub rate_map_whole_copy_vertex_regions: u64,
    /// Of it, the encoder's own draws', not an execution's commands'.
    pub rate_map_own_copy_vertex_regions: u64,
    /// The same of the runs of flagged draws' blocks drawn through their region-copy variant into
    /// the regions their bounds reach ([`rate_map_run_region_draws`](Self::rate_map_run_region_draws)).
    pub rate_map_run_vertex_regions: u64,
    /// The vertices or indices times the instances of the draws fanned out
    /// ([`rate_map_fanned_draws`](Self::rate_map_fanned_draws)) and of flagged draws' parts fanned
    /// out, three a triangle of a list.
    pub rate_map_fanned_vertices: u64,
    /// Those times the regions of the boxes their blocks' triangles reach by the probe's bounds
    /// (`bind::fan_out::PROBE_BOUNDS_AT`), as a run's: the vertex work each would ask for drawn
    /// through its region-copy variant into those boxes.
    pub rate_map_fanned_box_vertex_regions: u64,
    /// Of [`rate_map_fanned_vertices`](Self::rate_map_fanned_vertices), those of the draws and
    /// parts whose boxes hold at most 16 regions.
    pub rate_map_fanned_vertices_boxed_16: u64,
    /// Those whose boxes hold at most 64 regions.
    pub rate_map_fanned_vertices_boxed_64: u64,
    /// Those whose boxes hold at most 256 regions.
    pub rate_map_fanned_vertices_boxed_256: u64,
    /// Probed draws the probe flagged nothing of, and flagged draws' parts, that would fan out
    /// and were drawn through their region-copy variant into the boxes their triangles reach by
    /// the probe's bounds instead, where that holds no more work (`render/fan_out.rs`, **Boxed
    /// draws**): among [`rate_map_region_copy_draws`](Self::rate_map_region_copy_draws).
    pub rate_map_boxed_draws: u64,
    /// The encoder's own draws of a fanning pass that recorded their probe and both their routes,
    /// the fan-out variant's and the region-copy variant's, whose counts the GPU decides from the
    /// probe, counted as the encoder ends where their instance's steps run the routes' kernel
    /// (`render/gpu_routes.rs`): among
    /// [`rate_map_region_copy_draws`](Self::rate_map_region_copy_draws).
    pub rate_map_gpu_routed_draws: u64,
    /// Those the GPU fanned out, its probe flagging none of their triangles, read as their command
    /// buffer completed.
    pub rate_map_gpu_fanned_draws: u64,
    /// Those the GPU drew by region copies into every region, its probe flagging one of their
    /// triangles, read as their command buffer completed.
    pub rate_map_gpu_copied_draws: u64,
    /// Their vertices or indices times their instances times the regions each recording draws.
    pub rate_map_boxed_vertex_regions: u64,
    /// Render passes whose first instance did not record inline, its draws in a secondary command
    /// buffer, by the first check that ruled it out (`render/inline_first.rs`), counted as the pass
    /// opens or at its first draw that records: here, the encoder encodes an ended encoder's calls
    /// again, or the pass has several color groups.
    pub passes_not_inline_encoders: u64,
    /// Of them, those of a layout of no attachment, or of more than one layer or sample, a
    /// visibility buffer, custom sample positions, a rasterization-rate map, extended-range
    /// shadows that keep their draws, or a depth/stencil attachment held out of `GENERAL`.
    pub passes_not_inline_layouts: u64,
    /// Of them, those with a memoryless color attachment not loaded and stored `DontCare`, or
    /// resolved, or a memoryless depth or stencil.
    pub passes_not_inline_memoryless: u64,
    /// Of them, those with a color attachment or depth/stencil aspect loaded `DontCare`, reaching
    /// past the render area as another attachment does not, cleared to a NaN the device clears
    /// apart, resolved, or stored other than `Store` or late; each counted once more below, by
    /// the first such action, color by color in index order, then the depth and the stencil.
    pub passes_not_inline_actions: u64,
    /// Of those, the ones where that action is a `DontCare` load.
    pub passes_not_inline_actions_dont_care_load: u64,
    /// Of those, the ones where it is a reach past the render area that the pass's attachments do
    /// not share: Metal's clear of whole 32-pixel tiles reaches past the render area on one
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
    /// Of them, those with a color attachment of another kind: a stand-in (a bound linear image
    /// rendered apart from the pass's open among them, `linear_attachment.rs`), planes, a clear
    /// made apart, or an extended-range format; or a depth/stencil attachment of another kind:
    /// separate depth and stencil textures' scratch, a stand-in, a texture over a buffer, or a NaN
    /// depth clear written apart.
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
    /// Commits a pass took into the scheduler's order (`scheduler/mod.rs`), the scheduler
    /// thread's or a thread's own (`CommandBuffer::wait_until_completed`,
    /// `CommandBuffer::commit_and_submit`), each counted as the pass that took it in ended,
    /// after the submissions and the groups it made, which a later read of the counts sees.
    /// Passes take commits in in the order they were handed off, so once this reaches
    /// `commits_handed_off` read after a commit returned, a pass has taken that commit in and
    /// ended.
    pub commits_taken_in: u64,
    /// New `Shared` and `Managed` textures' images moved to `GENERAL` and zeroed on the host, with
    /// no submission (`texture/host_copy.rs`).
    pub images_started_on_the_host: u64,
    /// Of those, the images that started in the driver's zeros, in bytes of a block allocated
    /// zero-initialized that no resource used, moved to `GENERAL` out of
    /// `VK_IMAGE_LAYOUT_ZERO_INITIALIZED_EXT` with no write of the host's, on a device that
    /// enabled `zeroInitializeDeviceMemory` (`texture/host_copy.rs`). The rest of those started
    /// on the host were zeroed there.
    pub images_started_zeroed: u64,
    /// New `Private` textures the program made on a device that presents a Mac keeping no texture
    /// compressed (`DeviceProperties::lossless_compression` false, `MacClass::IntelOrAmd`), each
    /// filled with zeros at its first use (a clear, or copies of zeros, where the driver's zeros
    /// are not its bytes): zero is what such a Mac's new texture reads where its memory held
    /// nothing, and no texture reads what its memory held, which may be a released texture's
    /// texels of another process of a consumer whose processes share the device. On an Apple GPU
    /// the GPU's own compressed texture would read Apple's fixed pattern, which no such Mac
    /// reads. Zero on a device that presents an Apple silicon Mac's compression, whose new
    /// `Private` textures start as that Mac's.
    pub private_textures_zero_filled: u64,
    /// Of [`private_textures_zero_filled`](Self::private_textures_zero_filled), those whose zeros
    /// were left out: the first use of the texture among the commands of the submission that took
    /// its first move writes every texel before any command reads one, decided from that use's
    /// own description, never a shader's. That use is the submission's first encoder, or comes
    /// after blit encoders alone that name nothing of the texture (any other encoder's commands may
    /// reach a texture through its resource ID, unnamed), and names the texture in no other use
    /// that reads it: a render pass that loads every aspect of the texture `Clear`, stores it, and
    /// renders it itself over its whole extent (one level and one layer), or a blit encoder that
    /// writes its whole extent and reads none of it. A `DontCare` or `Load` first use takes the
    /// zeros.
    pub private_textures_overwritten_unfilled: u64,
    /// New memoryless textures zeroed at their first move, on a device whose memoryless memory
    /// is not lazily allocated: a render pass of each level alone clears every aspect over its
    /// layers, as no transfer writes a transient attachment. Its first use, a pass that loads it
    /// `DontCare` (whose draws test against its depth, or whose resolve reads it), so reads zero,
    /// never what its memory held, which may be a released texture's samples of another process
    /// of a consumer whose processes share the device; later passes read what earlier ones left,
    /// as before. Zero where memoryless memory is lazily allocated (MoltenVK), whose memoryless
    /// images hold nothing past a pass.
    pub memoryless_textures_zeroed: u64,
    /// `replace_region` and `get_bytes` copies that ran on the host, with no submission
    /// (`texture/host_copy.rs`).
    pub host_texel_copies: u64,
    /// `replace_region` and `get_bytes` copies of a heap's texture that is a shadow of the heap's
    /// bytes (a texture Metal tiles there, in a heap over host pages or one whose memory takes no
    /// image of it, and a two-plane texture's first level), made in those bytes on the host, the
    /// texels' truth, with no submission and no staging (`texture/heap.rs`).
    pub heap_shadow_cpu_copies: u64,
    /// Uploads into a 32-bit float depth image whose depths were clamped into `[0, 1]` before
    /// the copy, on a device whose depth images may store a value outside there clamped
    /// (`depth_clamp.rs`): one a `replace_region`'s clamp on the CPU, one a blit copy's
    /// dispatches of `bind::depth_clamp_module` before its copy into an image, and one a
    /// multisampled texture's fill, its samples' depths clamped on the CPU. Zero elsewhere.
    pub depth_uploads_clamped: u64,
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
    /// Submissions on the device's staging queue (`device/staging.rs`): a `get_bytes`, or the
    /// first moves and copies queued for it (`staging_moves_queued`, `staging_copies_queued`), on
    /// a device that copies none on the host from memory the host caches.
    pub staging_submissions: u64,
    /// Batches submitted to the copy queue's `VkQueue`, of a queue family of transfers alone: the
    /// command buffers of the queues made for copies
    /// ([`Device::new_command_queue_for_copies`](crate::Device::new_command_queue_for_copies)).
    pub copy_queue_batches: u64,
    /// `replace_region` copies (and decoded twins' copies) queued for the staging queue's next
    /// submission rather than submitted each alone (`texture/staging_copy.rs`).
    pub staging_copies_queued: u64,
    /// New `Shared` and `Managed` textures' first moves and zeros (and their decoded twins')
    /// queued for the staging queue's next submission rather than submitted and waited for as
    /// the texture is made (`texture/staging_copy.rs`).
    pub staging_moves_queued: u64,
    /// Staging submissions that carried queued moves or copies: one per burst of them, made
    /// before the device's next batch or internal submission, or by a `get_bytes` before its
    /// read, a texture's `VkImage` handed out, the program's use of a `VkQueue`, or past the
    /// bound (`texture/staging_copy.rs`).
    pub staging_flushes: u64,
    /// Batches and internal submissions that waited on the GPU for a staging submission the host
    /// had not seen complete (`Device::staging_wait`, `device/staging.rs`), in place of a wait on
    /// the host: of the batches, at most one per `VkQueue` per staging submission, as every later
    /// batch there follows its wait (`command_buffer/submit.rs`).
    pub staging_gpu_waits: u64,
    /// Command buffers the staging queue made past the 8 it keeps, because every one of its
    /// command buffers was pending as a submission there was recorded (`device/staging.rs`): at
    /// most 128 at once, each freed once the host saw its submission complete. A burst of staging
    /// submissions the queue had not run, as behind the program's kernels on a queue of compute
    /// work.
    pub staging_buffers_grown: u64,
    /// Submissions on the staging queue, or the fill queue, that waited on the host for one of the
    /// queue's command buffers to come free (`device/staging.rs`): on the staging queue, past 128
    /// pending; on the fill queue, past 8, where another submission took the room a batch found.
    /// The wait holds the thread that submits, the scheduler's among them.
    pub staging_buffer_waits: u64,
    /// Submissions on the device's fill queue, a copy engine's (`device/staging.rs`): each a
    /// batch's new buffers' fills (zeros, staged bytes, `Managed` buffers' copies in after
    /// `did_modify_range`), made there before the batch, which waits for it, rather than at the
    /// start of the batch's own work on its `VkQueue`. None on a device without a family of
    /// transfers alone with a second queue (lavapipe, RADV, MoltenVK).
    pub fill_queue_submissions: u64,
    /// Batches that waited on the GPU for submissions on the fill queue (`device/staging.rs`):
    /// for their own fills, or for another's the host had not seen complete; at most one per
    /// `VkQueue` per such submission, as every later batch there follows its wait.
    pub fill_queue_waits: u64,
    /// Of [`fill_queue_waits`](Self::fill_queue_waits), the batches whose commands reach none of
    /// the memory the submissions they waited for filled: neither a resource their encoders'
    /// uses name or make resident owns it, nor may a command of theirs reach memory no use names
    /// (a pointer loaded from memory, work recorded on the GPU alone). Each waited for work it
    /// does not use.
    pub fill_queue_waits_unused: u64,
    /// Batches that waited on the GPU for batches on other `VkQueue`s that did new resources'
    /// first work (`texture/first_move.rs`: images' first moves, staged copies into them, and
    /// fills at a batch's start), until the host saw those complete.
    pub first_work_waits: u64,
    /// Of [`first_work_waits`](Self::first_work_waits), the batches whose commands and own first
    /// work reach none of the memory the work they waited for wrote, as
    /// [`fill_queue_waits_unused`](Self::fill_queue_waits_unused) counts for fills. Each waited
    /// for work it does not use.
    pub first_work_waits_unused: u64,
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
    /// Decodes of a shadowed carrier's whole image into its decoded twin, recorded after a copy
    /// in that moved texels into the shadow (a heap's PVRTC, ETC2 or EAC texture on a device
    /// without the format whose image shadows the heap's bytes, `texture/heap.rs`), one a copy in;
    /// none for a use that copied nothing in.
    pub shadow_twin_decodes: u64,
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
    /// Kept bind shapes draws compared whole with their bind states, looking for theirs among
    /// those their encoders took lately and those their drawing states settled lately: only
    /// those of a draw's key beside its used resources (`BindState::shape_key_beside_used`), and
    /// none for a draw whose shape is its encoder's last draw's.
    pub draw_shape_compares: u64,
    /// Bind shapes compute encoders' dispatches made of their encoder's kept bind state
    /// (`compute/live.rs`): none for a dispatch that took what its encoder's last dispatch settled
    /// (`dispatches_settled_as_before`), nor for one whose bound slots keep their shapes and whose
    /// other bind state's shape is its encoder's last dispatch's that settled (an offset moved).
    pub dispatch_shapes_made: u64,
    /// Bind shapes compute dispatches found among those their pipeline states planned for lately,
    /// compared with their bind states in place: none for a dispatch as `dispatch_shapes_made`
    /// says.
    pub dispatch_shapes_found: u64,
    /// Compute dispatches that listed what their encoder made resident as their bind state's used
    /// resources (`Residency::used_resources`): one at an encoder's first dispatch that settles
    /// after each change of what it made resident, after a no-copy buffer moved, or after an
    /// indirect command buffer made resident lost its ID's slot; none for a dispatch whose
    /// encoder's residency kept its generation since its kept bind state listed them
    /// (`compute/live.rs`), however many resources it made resident.
    pub dispatch_used_resources_listed: u64,
    /// Looks of a compute encoder's dispatches at every texture the encoder made resident: for
    /// their shadows' copies (`shadow::ResidentShadows`), one at the encoder's first dispatch after
    /// each change of what it made resident or after its device made a usage shadow; and for one
    /// whose bytes a carrier holds (`pvrtc::ResidentUnreadable`), one at its first dispatch reading
    /// texture handles after each change of what it made resident. None for a dispatch after which
    /// neither changed, however many textures it made resident.
    pub resident_textures_looked_at: u64,
    /// Resources an ended compute or render encoder made resident, joined into its command
    /// buffer's residency: at the first read of that residency after the encoder ended (the work
    /// its commit or a host step records against every encoder's resources, its resident set, a
    /// texture made in a heap it used since, an encoder encoded again), each of the encoder's
    /// resources once. None for a command buffer whose work never reads it, however many
    /// resources its encoders made resident.
    pub resident_resources_joined: u64,
    /// Ended compute and render encoders' residencies looked at for the shadowed textures
    /// (`shadow.rs`) a later compute or render encoder of their command buffer starts with: each
    /// once for the first compute encoder opened after it ended and once for the first render
    /// encoder, and again only where the command buffer's residency joined the ended encoders'
    /// or an encoder was encoded again since. A command buffer of `n` compute encoders counts at
    /// most `n - 1`, not each earlier encoder at each open.
    pub ended_residencies_looked_at: u64,
    /// Indexes of the buffers and heaps a residency made resident by Metal address, which an
    /// indirect command buffer's commands' addresses resolve in (`Residency::resolve`): one at
    /// the first look after a resource was made resident into a residency holding a buffer or a
    /// heap, however many addresses the commands name.
    pub resident_address_indexes_made: u64,
    /// Resources a residency listed for the kept key of the indirect command buffer executions
    /// that take it, each listing every one (`Residency::kept_key`): one listing for what a
    /// residency holds, shared by its copies, so by every execution of a command buffer's, at its
    /// commit and at its host step, and listed again only where a no-copy buffer moved, an image
    /// let its views go or an indirect command buffer's hold on its ID's slot changed since.
    /// None for an execution whose residency's listing holds, however many resources it holds.
    pub resident_kept_keys_listed: u64,
    /// Ranges the root tables of draws and dispatches listed of their own resident sets
    /// (`metal2vulkan::bind::RootTable::resident_ranges`), noted for their command buffer's
    /// resident set, which the commit points every table at: each table's bound buffers with
    /// device addresses alone, as the set holds the resources made resident
    /// (`BindState::resident_apart`), so a draw's count is its own bindings', however many
    /// resources its encoder made resident.
    pub root_table_ranges_noted: u64,
    /// Reads of a pipeline layout for the device's texture and sampler tables its commands bind
    /// (`heap_sets.rs`): one per variant for compute dispatches, at its first (`HeapTables`); one
    /// per table looked for at each draw and tile dispatch (`heap_sets::texture_table`,
    /// `sampler_table`).
    pub heap_tables_worked_out: u64,
    /// Draws (and commands of other encoders that settle as draws do) and compute encoders'
    /// dispatches that settled (no memo entry held) that took the device's live sampler states to
    /// list as their argument samplers: none for one whose kept bind state lists them at the count
    /// of their slots' changes that still holds, whose variant has no set-1 arrays and names no
    /// state in a notice (`sampler::ArgumentSamplers`). A compute encoder's dispatch whose memo
    /// entry, parked in its command buffer, goes on in the next encoder takes them where that
    /// entry's variant has set-1 arrays (`compute::DispatchMemo::restamp`).
    pub argument_samplers_taken: u64,
    /// Texture slot views binds worked out: none for a render or compute encoder's bind of a
    /// texture, as a view, that a bind of the same or an earlier encoder of its kind of the device
    /// bound, while no image of the device let its views go and no shadow over host pages moved in
    /// place (`binding::BindMemo`). One for each such view any other command binds (an indirect
    /// command buffer's, a host step's, a tile dispatch's), which keeps no memo.
    pub bind_views_worked_out: u64,
    /// Sampler descriptions binds worked out for sampler slots: none for a render or compute
    /// encoder's bind of a sampler state, with the same LOD clamps, address-mode map and use, that
    /// a bind of the same or an earlier encoder of its kind of the device bound, while the state
    /// lives (`binding::BindMemo`). One for each such description any other command binds.
    pub bind_samplers_worked_out: u64,
    /// Reads, for a bind's buffer descriptor, of where a buffer over host pages lies (a no-copy
    /// buffer, or a buffer placed in a no-copy heap): none for a render or compute encoder's bind
    /// of such a buffer that a bind of the same or an earlier encoder of its kind of the device
    /// read, while no no-copy buffer was rebound or detached since (`buffer::KeptSpaces`). One for
    /// each such buffer any other command binds.
    pub bind_spaces_read: u64,
    /// Stage tables (vertex, fragment, object, mesh) that draws, and commands of other encoders
    /// that settle as draws do, read into their encoder's kept bind state: four a draw at most,
    /// none for a table no call changed since a draw read it through the same drawing state's
    /// interface, with the same samplers' address-mode maps, while no no-copy buffer was rebound
    /// or detached (`live_bind::ReadFrom`).
    pub stage_tables_read: u64,
    /// Those reads that brought in only the slots calls changed since the table was read, the
    /// table keeping them (`ArgumentTable::refresh_changed`): every other slot is as it was read.
    pub stage_tables_read_in_part: u64,
    /// Of `stage_tables_read`, those a draw's settle read whole (`ArgumentTable::refresh_reading`,
    /// `render.rs`'s `refresh_bind_state`) because the state it started from never read its
    /// tables: the first of its checks that failed, in their order, names each such read, so these
    /// eight and `stage_tables_read_in_part` add up to `stage_tables_read`.
    pub stage_tables_read_whole_unread: u64,
    /// Those because a no-copy buffer was rebound or detached since the state was read.
    pub stage_tables_read_whole_moved: u64,
    /// Those because the state was read through another interface: another pipeline state's
    /// draws left it (`live_bind::LiveBind::select`).
    pub stage_tables_read_whole_interface: u64,
    /// Those because the state was read through the same interface with other samplers'
    /// address-mode maps.
    pub stage_tables_read_whole_maps: u64,
    /// Those because the slots that tell the view base alignment apart are not as the state
    /// holds them (`argument_table::AtBindViews`).
    pub stage_tables_read_whole_views: u64,
    /// Those because the state was read from an ended encoder's tables, its encoder having started
    /// from it as a spare (`live_bind::SpareLiveBinds`): a new encoder's tables are new, and list
    /// no call since it.
    pub stage_tables_read_whole_spare: u64,
    /// Those because the state was read from an ended encoder's tables, its encoder having taken
    /// it carried in with a kept draw (`render/carried.rs`).
    pub stage_tables_read_whole_carried: u64,
    /// Those because the table does not list the calls since the state read it: more calls, or
    /// a call of more slots, than it keeps (`argument_table::Changes`).
    pub stage_tables_read_whole_unlisted: u64,
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
    /// Uses of draws noted apart that their encoder did not record: each differs from the use the
    /// last command it noted had at its place only by a range that one holds, of the same
    /// resource, tracking and access types, whose bytes are in the same part already
    /// (`noted_apart.rs`), as a draw whose buffer offsets move along a buffer reads bytes its
    /// earlier draw's read held.
    pub uses_noted_covered: u64,
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
    /// Draws of lines at a width other than 1 (Apple's `setLineWidth:`) that ran the device's own
    /// lines at that width (`lineWidth`), a triangle's edges in fill mode `Lines` and mesh draws
    /// among them.
    pub device_wide_line_draws: u64,
    /// Draws of lines at a width other than 1 that ran the pipeline's band variant
    /// (`VariantKey::line_band`): each line two triangles over Metal's parallelogram, and each
    /// triangle in fill mode `Lines` its three edges' parallelograms.
    pub line_band_draws: u64,
    /// Draws of lines at a width other than 1 that neither route above draws, which ran the
    /// device's own lines at the width nearest Metal's they draw (`lineWidthRange`, 1 without
    /// `wideLines`), announced `lines-at-nearest-width`: mesh draws, tessellated patches' edges,
    /// and lines and edges without `tessellationShader` or of a pipeline state whose band the
    /// translator refuses among them.
    pub nearest_width_line_draws: u64,
    /// Draws that pulled a vertex layout their pipeline fetches in fixed function, because they
    /// may fetch it past `2^32` bytes of its bind offset on a device whose fixed-function vertex
    /// input takes that displacement modulo `2^32`
    /// (`metal2vulkan::error::RuntimeDriverDefect::VertexFetchWrapsPast4GiB`): each draw that
    /// took every step and settled so (`BindState::far_vertex_buffers`).
    pub far_vertex_fetch_pulled_draws: u64,
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
    /// A command's retains of an argument table that calls changed since the encoder last
    /// retained it, which looked at none of its objects: each object those calls bound, the
    /// encoder held already as it was bound, from an earlier retain of one of its tables
    /// (`retained::HeldBy`). A retain of a table a call bound another object into since, or of a
    /// table's copy, or of one another encoder retained last, looks at every object it binds.
    pub argument_tables_held_at_bind: u64,
    /// Buffer slots a stage table read in part brought in where every call since its last read
    /// moved the offset (or attribute stride) of the buffer bound there alone: the binding kept,
    /// its offset and stride the table's, and the buffer itself not looked at (no lock of a
    /// no-copy buffer's pages taken).
    pub bound_offsets_moved: u64,
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
    /// Scratch buffers the merges of colors over a buffer's bytes rendered through an image of
    /// the runtime's made (`linear_merge.rs`): a merge makes one only where its command buffer
    /// lends none its earlier passes' merges were done with that holds its box
    /// (`merge_scratch.rs`), so a command buffer of many passes over one attachment makes a few.
    pub linear_merge_scratches_made: u64,
    /// Those merges that took a scratch an earlier pass of their command buffer was done with.
    pub linear_merge_scratches_reused: u64,
    /// The merges of those colors recorded: one after each pass that merges on its own, and one
    /// at its batch's submission for each run of passes over one attachment merged once, in
    /// place of its earlier passes' (`pass_runs.rs`).
    pub linear_merges_recorded: u64,
    /// The merges a later pass of their run took over: none of them is recorded, and the run's
    /// last merge writes what they would have.
    pub linear_merges_taken_over: u64,
    /// Passes of a linear attachment's run that a rebind encoded again alone (their commands
    /// named a buffer over host pages it moved) and a later pass of the run went on over: each
    /// copied the run's box in again from the surface after its own merge, and kept it, for the
    /// run's later passes and its merge (`pass_runs.rs`'s **Encoded again**).
    pub linear_runs_reloaded: u64,
    /// Passes that rendered a color attachment over a buffer's bytes, which they do not store
    /// (`DontCare`, `MultisampleResolve`), through the image its texture keeps, so that the buffer
    /// keeps its bytes (`kept_bytes.rs`).
    pub kept_bytes_passes: u64,
    /// Passes that rendered a color attachment of a texture's own image, which they store
    /// `DontCare`, through a backing their queue keeps, so that the texture keeps its earlier
    /// texels (`kept_bytes.rs`): only under `DontCareColorPolicy::Kept`.
    pub kept_texels_passes: u64,
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
    /// 1.4 or `VK_KHR_push_descriptor`, a dispatch that pushes no write among them.
    pub dispatches_pushing_set_zero: u64,
    /// The descriptor writes those dispatches' pushes carried, one write a descriptor, so over
    /// `dispatches_pushing_set_zero` the descriptors a dispatch pushes. A dispatch pushes only
    /// those of its set 0 (its writes, then its variant's baked samplers' pushes) that differ from
    /// what the encoder's dispatch before it pushed with the same pipeline layout into the same
    /// `VkCommandBuffer`, with nothing else recorded there between, none where none differs; at
    /// least one after a pipeline bind where the driver's pushed descriptors go stale across one
    /// (`RuntimeDriverDefect::PushedDescriptorsStaleAfterPipelineBinds`); else all of them: an
    /// encoder's first dispatch, one of another layout, one after any other command of its
    /// encoder, and one recorded later while its pipelines are made.
    pub dispatch_set_zero_writes_pushed: u64,
    /// Compute dispatches whose uses their encoder's hazard tracking took
    /// (`EncoderCore::command_reaching`, or `command_later` for one that records later): every
    /// dispatch that records, but one of an inert kernel or one naming a detached buffer.
    pub dispatches_tracked: u64,
    /// The uses those dispatches handed to hazard tracking, each resource range a command reads
    /// or writes (those it reaches through what its encoder made resident among them), so over
    /// `dispatches_tracked` the uses a dispatch reaches, which its tracking walks.
    pub dispatch_uses_tracked: u64,
    /// Draws and dispatches that bound the set 0 their encoder's last command allocated and wrote,
    /// rather than allocating and writing their own: those that write the same descriptors into a
    /// set of the same layout (`binding::LastSetZero`). Their recording binds no set where the set
    /// is bound already. Not a draw drawn again (`draws_as_before`), which binds no set.
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
    /// alone, from an earlier draw that derived it from the same inputs, rather than deriving it
    /// again: their settings, their draw state (fixed state, viewports, depth bias; its scissors
    /// are each draw's own, from the encoder's scissor rectangles) and their `VkPipeline`. The
    /// inputs are the drawing state, variant and plan by identity, the pass's layout, the
    /// encoder's state but its scissor rectangles and vertex amplification (by version within an
    /// encoder, by value from an earlier encoder's draw: `draw_shapes_carried`), the instance's
    /// shape by version or value, and the assembly, the pass's sample positions, the vertex
    /// strides, the fixed state and the topology by value (`render/draw_shape.rs`). None of a
    /// shape whose derivation announces a notice the device announces at every occurrence
    /// (`DeviceOptions::announce_each`). Not a draw drawn again (`draws_as_before`), which takes
    /// nothing of the memo.
    pub draw_shapes_reused: u64,
    /// Render encoders' draws that found no shape in their encoder's memo and took their
    /// settings, with the draw state and `VkPipeline` an earlier draw of them found, from an
    /// earlier encoder's draw of the same drawing state that derived them from the same values:
    /// the assembly, the pass's pixel formats, depth and stencil, samples, extent and tile memory,
    /// the encoder's state but its scissor rectangles (each float by its bits) and its vertex
    /// amplification (`render/draw_shape.rs`, the drawing state's last 8 such shapes). None in a
    /// rate-mapped pass, nor of a shape whose derivation announces a notice the device announces
    /// at every occurrence.
    pub draw_shapes_carried: u64,
    /// Render encoders' direct draws recorded as their encoder's last draw again, its Vulkan draw
    /// command alone, rather than taking every step of a draw: a draw of the same call with the
    /// same arguments (its index buffer read in place at the same offset), which no call of the
    /// encoder came between, while no no-copy buffer moved, the encoder's residency kept its
    /// generation, the device's host aliases kept their version and the encoder kept its parts
    /// and held images, after a draw that recorded one Vulkan draw into the instance as it
    /// stood, through a `VkPipeline` made ahead, with no barrier in its instance, no write among
    /// its uses, no copy, snapshot, split, gather, index bounds or rewrite pass, mesh or
    /// tessellation work, set 1, root table, rate map, color group or extended-range color, no
    /// read of a color its pass clears behind its load actions, and no notice the device
    /// announces at every occurrence (`render/draw_again.rs`); its shadowed textures, where it
    /// used any, gathered by its instance already. Each is counted in `commands_noted_as_before`,
    /// `draws_pushing_set_zero`, `provoking_last_pipeline_draws`,
    /// `attachment_reads_through_copies` and `own_pixel_reads` as the draw it repeats was, and in
    /// no count of work it does not take (`draw_shapes_reused`, `set_zero_sets_reused`,
    /// `draw_states_recorded`).
    pub draws_as_before: u64,
    /// Render encoders' direct draws recorded as their encoder's last draw again after calls
    /// that bound other buffers, offsets or bytes at buffer slots, or other textures or sampler
    /// states at texture and sampler slots, rather than taking every step of a draw: a draw that
    /// `draws_as_before` would count but for those calls, where the bind state kept its shape and
    /// its runtime samplers' address modes, the slots are read through set-0 descriptors
    /// (dynamic or pushed), push fields or fixed-function vertex buffers alone, the uses they make
    /// need no barrier or copy, and the draw uses the shadowed textures the kept draw used, each
    /// as it used it (`render/draw_again/rebound.rs`). Each reads its
    /// changed stage tables in part (`stage_tables_read_in_part`), writes again only those slots'
    /// descriptors, dynamic offsets, push fields and vertex buffers, notes its uses as any draw
    /// does, and records the state commands that changed (`draw_states_recorded`) and its draw;
    /// it takes nothing of the shape memo (`draw_shapes_reused`) and settles nothing.
    pub draws_rebound: u64,
    /// Of `draws_rebound`, those moved: each call since the draw they repeat moved a buffer slot's
    /// offset alone, the slot binding the buffer it bound (`render/draw_again/rebound.rs`). A
    /// draw moved looks at no visible-function table, shadow or detached buffer again, its slots
    /// binding the objects that draw bound; keeps that draw's set-0 writes and dynamic offsets
    /// where its descriptor ranges are that draw's; and where its moved slots' uses were patched
    /// in place, checks, notes and gathers those uses alone, every other use being that draw's.
    pub draws_moved: u64,
    /// Of `draws_moved`, those whose patched uses each lay inside the bytes of the last draw's
    /// use at their place, as a buffer offset moving along a buffer reads from it to the buffer's
    /// end (`render/draw_again/rebound.rs`): each noted already, in the pass's reads and among the
    /// instance's gathered uses, so the draw notes, adds and gathers none of them, comparing none
    /// (`uses_noted_covered` counts none of them).
    pub draws_moved_inside: u64,
    /// Of `draws_rebound`, those after calls that set the encoder's render state (its viewports,
    /// scissor rectangles, cull mode, winding, depth/stencil state, stencil references, depth
    /// bias, depth clip mode, fill mode or blend color): each derived again what that state
    /// decides of it, its settings and draw state, and recorded the dynamic states it sets that its
    /// record held otherwise (`render/draw_again/restated.rs`).
    pub draws_restated: u64,
    /// Kept draws restated by their scissor rectangles alone (`render/draw_again/restated.rs`):
    /// the encoder set its scissor rectangles and nothing else of its render state since the kept
    /// draw was kept or last rebound, and the draw's route meets every condition of a restate, so
    /// the draw took the new scissors into its draw state and derived nothing else again (its
    /// route, settings, bind state's values, fragment tests and draw state's static part, which
    /// no scissor rectangle decides). Each whose draw was then recorded rebound is one of
    /// `draws_restated` too; one whose rebind then stopped, or which a build with debug
    /// assertions checked by its every step, took every step.
    pub draws_restated_scissors: u64,
    /// Render encoders' kept draws set aside with the states of other pipeline states, kept into
    /// the render pass instance a draw began after a split, of the same shape, where that draw
    /// kept nothing only for recording into it (`render/draw_again.rs`): each but those that read
    /// the pass's colors other than through snapshots of their own, a reader through snapshots,
    /// or one that read what changes with no call otherwise, which are let go. The next draw of
    /// each is rebound into that instance (`draws_rebound`), in place of taking every step
    /// (`draws_none_kept`).
    pub kept_draws_rehomed: u64,
    /// Of `kept_draws_rehomed`, those carried in from the command buffer's last render encoder,
    /// or past a commit, and not drawn since (`render/carried.rs`): the next draw of each is
    /// rebound as a carried draw is, into the instance after the split, bringing in every slot
    /// its encoder's calls bound since the opening (`draws_rebound_carried`), in place of taking
    /// every step (`draws_none_kept`).
    pub kept_draws_rehomed_carried: u64,
    /// Kept draws carried in and not drawn since, let go at their state's next draw because the
    /// encoder's pass records into another render pass instance than the one the encoder opened
    /// with and no split re-homed them (`render/carried.rs`, `kept_draws_rehomed_carried`): that
    /// draw takes every step (`draws_none_kept`).
    pub kept_draws_carried_unopened: u64,
    /// Of `draws_rebound`, those that differed from the draw they repeat only in their arguments
    /// (`draws_unlike_kept_arguments_alone` lists them: their call of the same primitive type,
    /// with indices or without, their counts and starts, index buffer, index type or index buffer
    /// offset, instance count, base vertex or base instance), which each worked out again where a
    /// step reads them: its route (whether it draws a primitive, a strip restarted at another
    /// index, a de-indexed variant), its Vulkan draw, index overrun, base vertex and instance push
    /// fields and index buffer use (`BindingPlan::redraw_apart_into`), its pulled attributes'
    /// reach and its index buffer's bind, which it records with its draw, its index buffer held
    /// where it is another (`render/draw_again/rebound.rs`).
    pub draws_rebound_arguments: u64,
    /// Of `draws_rebound`, render encoders' first draws of a kept draw their command buffer's
    /// last render encoder carried into them (`kept_draws_carried`, `kept_draws_carried_aside`,
    /// `render/carried.rs`): each derived again what the encoder's render state decides of it,
    /// brought in every slot the encoder's calls bound, made its uses whole, gathered its uses of
    /// shadowed textures into its instance (a copy of its pass's attachment's texture among them),
    /// began its pass's first instance as every step begins it where no draw before it began it,
    /// and recorded every state command it sets into the instance's command buffer.
    pub draws_rebound_carried: u64,
    /// Render encoders' direct draws whose encoder kept no draw of their pipeline state to draw
    /// again or rebind (`render/draw_again.rs`): its first draw of the state where no kept draw
    /// of it was carried in (`render/carried.rs`: the command buffer's first render encoder, a
    /// carry let go, or a kept draw of the state that did not carry, `kept_draws_carry_*`,
    /// `kept_draws_uncarried_attached`), or the first after a call or a draw that let every kept
    /// draw go. Each takes every step. A draw replayed from an execution's commands, or recorded
    /// again into another instance, is not counted.
    pub draws_none_kept: u64,
    /// Of `draws_none_kept`, those of render encoders that opened from no carried kept draw
    /// (`render/carried.rs`): one whose command buffer's last render encoder carried none into it,
    /// or let it go, and whose queue held none that fits its pass. The rest are draws of encoders
    /// that opened from a carry, of a state it did not carry or after a call that let every kept
    /// draw go.
    pub draws_none_kept_uncarried: u64,
    /// Render encoders' first draws of a carried kept draw whose rebind stopped, so that they took
    /// every step: each counted in one of the counts below.
    pub draws_carry_stopped: u64,
    /// Of `draws_carry_stopped`, those whose bind state held a slot bound that no call of their
    /// encoder bound: a new encoder inherits no binding, so the state is stale there.
    pub draws_carry_stopped_stale: u64,
    /// Of `draws_carry_stopped`, those that read a texture an attachment of their pass renders
    /// through a slot narrowed to levels apart from the attached ones
    /// (`render/attached_levels.rs`), which their kept draw, carried, did not narrow.
    pub draws_carry_stopped_attached: u64,
    /// Of `draws_carry_stopped`, those that stopped at another step of a rebind
    /// (`draws_rebind_stopped`'s).
    pub draws_carry_stopped_other: u64,
    /// Of `draws_carry_stopped`, those that read through snapshots and would read through them
    /// otherwise, or not, at every step (`reading_draws_snapshotted_stopped`): another slot, color
    /// or form, or an instance that cannot hold their colors in the feedback loop layout; and
    /// those that read through none and would read their pass's colors through snapshots, in
    /// place or after the split at every step (a texture over a buffer's bytes takes the snapshot
    /// whatever the area).
    pub draws_carry_stopped_snapshot: u64,
    /// Render encoders' kept draws (the last direct draw's) let go by a call the encoder logged,
    /// with every kept draw set aside (`render/draw_again.rs`), so their next draw takes every
    /// step (`draws_none_kept`): by `wait_for_fence` where the encoder used a shadowed texture,
    /// which then takes its copies again (`shadow.rs`).
    pub kept_draws_let_go_fence: u64,
    /// By a `use_resource` or `use_resources` of a memoryless texture.
    pub kept_draws_let_go_use: u64,
    /// By `memory_barrier_with_scope` or `memory_barrier_with_resources`: one that splits the
    /// pass, or, of one over the render targets alone kept across
    /// (`kept_draws_kept_across_barrier`), a kept draw that reads the pass's colors other than
    /// through snapshots of its own.
    pub kept_draws_let_go_barrier: u64,
    /// By a color's, the depth's or the stencil's store action or store action options.
    pub kept_draws_let_go_store_action: u64,
    /// By `set_visibility_result_mode`.
    pub kept_draws_let_go_visibility: u64,
    /// By any other call that logs itself (`RenderLowering::log`): every call but the draws, the
    /// binds, a pipeline state set, the render state sets and those `kept_draws_kept_across_*`
    /// and the counts above name.
    pub kept_draws_let_go_other: u64,
    /// Render encoders' kept draws (the last direct draw's) kept, with every kept draw set aside,
    /// across a call the encoder logged that changes nothing they read or recorded
    /// (`render/draw_again.rs`): `update_fence`, and `wait_for_fence` where the encoder used no
    /// shadowed texture.
    pub kept_draws_kept_across_fence: u64,
    /// Across `use_resource`, `use_resources` (of no memoryless texture), `use_heap` or
    /// `use_heaps`: one that changed the encoder's residency gave it a new generation, which
    /// makes the next draw of each take every step.
    pub kept_draws_kept_across_use: u64,
    /// Across `memory_barrier_with_scope` over the render targets alone that records its barrier
    /// inside the pass or nothing: every kept draw but those that read the pass's colors other
    /// than through snapshots of their own (`render/draw_again.rs`), which the barrier lets go
    /// (`kept_draws_let_go_barrier` where the last direct draw's is one).
    pub kept_draws_kept_across_barrier: u64,
    /// Render encoders' direct draws that recorded a Vulkan draw and kept nothing to draw again
    /// or rebind (`render/draw_again.rs`), so their encoder's next draw takes every step; with,
    /// in the counts below, what kept each from being kept, a draw counted in every group it met
    /// (the conditions `draws_as_before` lists). A draw replayed from an execution's commands is
    /// not counted.
    pub draws_unkept: u64,
    /// Of `draws_unkept`, those that took a step of their own beside their Vulkan draw (a copy,
    /// rewrite, index bounds, stand-in, mesh or tessellation work, a pipeline being made inline)
    /// or recorded into a later render pass instance of their pass. A pass's first draw that
    /// begins its first instance inline (`render/inline_first.rs`) is kept as any.
    pub draws_unkept_instance: u64,
    /// Of `draws_unkept_instance`, those that recorded into a later render pass instance of their
    /// pass: after a split, a stand-in arriving with them, or a continuation; counted in each of
    /// the two it met.
    pub draws_unkept_instance_later: u64,
    /// Of `draws_unkept_instance`, those that took a step of their own beside their Vulkan draw:
    /// a copy, rewrite, index bounds, folded or indirect arguments, mesh or tessellation work, a
    /// stand-in or imageblock made, a pipeline being made, or steps before the instance.
    pub draws_unkept_instance_steps: u64,
    /// Of `draws_unkept`, those that recorded after a barrier in their render pass instance.
    pub draws_unkept_barrier: u64,
    /// Of `draws_unkept`, those that read a color their pass clears behind its load actions
    /// through a copy (`texture/attached_reads.rs`), or that read through a snapshot or an
    /// extended-range color's shadow. A draw that uses other shadowed textures is kept: its
    /// instance gathered them (`render/draw_again.rs`).
    pub draws_unkept_shadowed: u64,
    /// Of `draws_unkept`, those that read a heap's set, a texture or sampler table, a root table
    /// or a push hole, or whose argument samplers left states out.
    pub draws_unkept_tables: u64,
    /// Of `draws_unkept_tables`, those that read a heap's set.
    pub draws_unkept_tables_heap: u64,
    /// Of `draws_unkept_tables`, those also counted in `draws_unkept_writing`.
    pub draws_unkept_tables_writing: u64,
    /// Of `draws_unkept`, those that read their index buffer through a copy.
    pub draws_unkept_index_copied: u64,
    /// Of `draws_unkept`, those with a use that writes memory or reads indirect arguments, or that
    /// bind a zero region of their own at a writable buffer binding nothing is bound to.
    pub draws_unkept_writing: u64,
    /// Of `draws_unkept_writing`, those with such a use among the uses of what their encoder made
    /// resident (`use_resource`, `use_heap`).
    pub draws_unkept_writing_resident: u64,
    /// Of `draws_unkept`, those that recorded against a `VkPipeline` still being made, or through
    /// their key's general form's while their own was made (`residual_form_draws`).
    pub draws_unkept_making: u64,
    /// Of `draws_unkept`, those that met none of the groups above (another condition a kept draw
    /// meets: a split, a color group, an extended-range color, a rate map, a notice announced at
    /// each occurrence, among others).
    pub draws_unkept_other: u64,
    /// Render encoders' direct draws whose encoder kept a draw, of another primitive type,
    /// pipeline state or command buffer, of an index buffer read through a copy, or after a
    /// no-copy buffer moved, the residency, host aliases, parts or held images or the sampler IDs
    /// changed, or of other arguments that draw nothing (`render/draw_again.rs`): each takes every
    /// step. A draw that differs in its arguments alone (its call with indices or without among
    /// them, and its index buffer read in place) and draws something is rebound instead
    /// (`draws_rebound_arguments`).
    pub draws_unlike_kept: u64,
    /// Of `draws_unlike_kept`, those through another pipeline state than the kept draw's (or
    /// none); a draw is counted in each of the parts below it met.
    pub draws_unlike_kept_pipeline: u64,
    /// Of `draws_unlike_kept`, those of another call or primitive assembly.
    pub draws_unlike_kept_call: u64,
    /// Of `draws_unlike_kept`, those of other vertex or index counts or starts, instance count,
    /// base vertex or base instance.
    pub draws_unlike_kept_counts: u64,
    /// Of `draws_unlike_kept`, those of another index buffer or index type, or whose index buffer
    /// is read through a copy.
    pub draws_unlike_kept_index_buffer: u64,
    /// Of `draws_unlike_kept`, those of the same index buffer at another offset.
    pub draws_unlike_kept_index_offset: u64,
    /// Of `draws_unlike_kept`, tessellation draws: of patch or control-point index buffers.
    pub draws_unlike_kept_tessellation: u64,
    /// Of `draws_unlike_kept`, those recorded into another `VkCommandBuffer` than the kept draw.
    pub draws_unlike_kept_command_buffer: u64,
    /// Of `draws_unlike_kept`, those after a no-copy buffer moved, or the residency, the host
    /// aliases, parts or held images, or the sampler IDs changed.
    pub draws_unlike_kept_under: u64,
    /// Of `draws_unlike_kept`, those unlike only by their call of the same primitive type (with
    /// indices or without), their counts and starts, their index buffer, index type or index
    /// buffer's offset, each index buffer read in place: the same pipeline state, assembly,
    /// command buffer and the rest the kept draw read, and no vertex, index or instance to draw.
    /// Every other draw so unlike its kept draw is rebound (`draws_rebound_arguments`) or takes
    /// every step where its rebind stops (`draws_rebind_stopped_arguments`, by where).
    pub draws_unlike_kept_arguments_alone: u64,
    /// Render encoders' direct draws of their kept draw again after calls, that took every step
    /// rather than being rebound (`draws_rebound`): a kept draw that keeps nothing to rebind, or a
    /// rebind that stopped (`render/draw_again/rebound.rs`'s stops). Builds with debug
    /// assertions' checked draws are not counted.
    pub draws_rebind_stopped: u64,
    /// Of `draws_rebind_stopped`, draws that differed from their kept draw only in their
    /// arguments (`draws_rebound_arguments`) and whose rebind stopped, each counted in one of the
    /// `draws_rebind_stopped_arguments_*` below by where, in the order a rebind takes its steps.
    /// Each takes every step. A kept draw is never a de-indexed, mesh or patch draw
    /// (`render/draw_again.rs`); a draw with indices after it that would run de-indexed stops at
    /// its route.
    pub draws_rebind_stopped_arguments: u64,
    /// Of `draws_rebind_stopped_arguments`, those whose route the arguments change: a draw that
    /// may make no complete primitive after one that may make some, or the other way, a strip
    /// over indices restarted at another index than Vulkan's, or a de-indexed variant's draw.
    pub draws_rebind_stopped_arguments_route: u64,
    /// Of `draws_rebind_stopped_arguments`, those whose kept draw keeps nothing to rebind: its
    /// state made a visible-function table, it used more than its plan lists, it left a push
    /// hole, a set-0 write of its holds a keeper, or it recorded other than one Vulkan draw of a
    /// `VkPipeline` made ahead.
    pub draws_rebind_stopped_arguments_unkept: u64,
    /// Of `draws_rebind_stopped_arguments`, those after a call that set the encoder's render state
    /// to what the kept draw's derivation does not cover (`render/draw_again/restated.rs`).
    pub draws_rebind_stopped_arguments_restated: u64,
    /// Of `draws_rebind_stopped_arguments`, a draw carried into its encoder whose bind state holds
    /// a slot no call of the encoder bound (`draws_carry_stopped_stale`).
    pub draws_rebind_stopped_arguments_stale: u64,
    /// Of `draws_rebind_stopped_arguments`, those of a stage table some call changed that the kept
    /// draw's bind state read otherwise: through another interface, after a no-copy buffer
    /// moved, under another stamp, or with other slots that tell the view base alignment apart.
    pub draws_rebind_stopped_arguments_unread: u64,
    /// Of `draws_rebind_stopped_arguments`, those of a stage table that no longer lists the slots
    /// its calls changed since the kept draw read it: one call changed more than it lists, or
    /// later calls pushed them out.
    pub draws_rebind_stopped_arguments_unlisted: u64,
    /// Of `draws_rebind_stopped_arguments`, those after calls that changed more buffer, texture
    /// and sampler slots of the stages, together, than a rebind takes (64), or another kind of
    /// slot.
    pub draws_rebind_stopped_arguments_slots: u64,
    /// Of `draws_rebind_stopped_arguments`, those whose changed texture or sampler meets a
    /// memoryless texture its functions use, or that changes the runtime samplers' address modes.
    pub draws_rebind_stopped_arguments_images: u64,
    /// Of `draws_rebind_stopped_arguments`, those whose changed slots reshape the bind state, which
    /// every step then makes again: a changed table's slots that tell the view base alignment
    /// apart, or changed slots under whose shape the state settles otherwise than the kept draw
    /// (`draws_rebind_reshaped_*` name what, with the draws of the same arguments).
    pub draws_rebind_stopped_arguments_reshaped: u64,
    /// Of `draws_rebind_stopped`, those whose rebind stopped where the changed slots it brought in
    /// may reshape the bind state, a draw of other arguments as a draw of the same (those of other
    /// arguments are `draws_rebind_stopped_arguments_reshaped`), by what told the state so: here,
    /// a changed table's slots that tell the view base alignment apart not as the state holds them
    /// (`argument_table::AtBindViews`). Each of the rest is counted below under the first of its
    /// changed slots that told the state its shape may have changed (`live_bind::Reshape`), in the
    /// order the rebind brings them in, where the state's shape then settles otherwise than the
    /// kept draw.
    pub draws_rebind_reshaped_views: u64,
    /// Those whose first such slot is a buffer slot bound where the state held none.
    pub draws_rebind_reshaped_buffer_bound: u64,
    /// Those whose first such slot is a buffer slot the state held unbound.
    pub draws_rebind_reshaped_buffer_unbound: u64,
    /// Those whose first such slot is a buffer slot bound to a binding of another shape.
    pub draws_rebind_reshaped_buffer_shape: u64,
    /// Those whose first such slot is a buffer slot whose offset alone moved, to one of another
    /// shape (`ArgumentTable::refresh_offset`).
    pub draws_rebind_reshaped_buffer_offset: u64,
    /// Those whose first such slot is a texture slot bound where the state held none.
    pub draws_rebind_reshaped_texture_bound: u64,
    /// Those whose first such slot is a texture slot the state held unbound.
    pub draws_rebind_reshaped_texture_unbound: u64,
    /// Those whose first such slot is a texture slot bound to a texture of another shape.
    pub draws_rebind_reshaped_texture_shape: u64,
    /// Those whose first such slot is a sampler slot bound where the state held none.
    pub draws_rebind_reshaped_sampler_bound: u64,
    /// Those whose first such slot is a sampler slot the state held unbound.
    pub draws_rebind_reshaped_sampler_unbound: u64,
    /// Those whose first such slot is a sampler slot bound to a sampler of another shape.
    pub draws_rebind_reshaped_sampler_shape: u64,
    /// Of `draws_rebind_stopped_arguments`, those whose bind state's kept settle is not the kept
    /// draw's variant and plan.
    pub draws_rebind_stopped_arguments_unsettled: u64,
    /// Of `draws_rebind_stopped_arguments`, those whose plan reads its slots some other way than a
    /// rebind works out (`BindingPlan::slot_rebinds`): a stage's argument or root table, a uniform
    /// view a written slot may alias, or a slot changed since whose texel range, planes or
    /// per-bind stride the commands hold.
    pub draws_rebind_stopped_arguments_tables: u64,
    /// Of `draws_rebind_stopped_arguments`, those that move a fixed-function vertex buffer's fetch
    /// window (`BindingPlan::redraw_apart_into`): another vertex range of a draw without indices,
    /// or another instance range of a per-instance buffer. A base vertex or index offset of an
    /// indexed draw moves none: its vertices' window is the buffer's whole.
    pub draws_rebind_stopped_arguments_fetch_window: u64,
    /// Of `draws_rebind_stopped_arguments`, those whose pulled attribute reaches past 4 GiB of its
    /// buffer (`render/vertex_reach.rs`), which every step refuses.
    pub draws_rebind_stopped_arguments_reach: u64,
    /// Of `draws_rebind_stopped_arguments`, those that meet a check of every step a rebind does
    /// not take again: a visible-function table's, a pass whose fragments wrote bytes an earlier
    /// stage reads, an attachment its pass clears read after a barrier, a color's texture read
    /// through its copy after a barrier inside the draw's instance, or a changed texture beside a
    /// storage color or an extended-range shadow.
    pub draws_rebind_stopped_arguments_checks: u64,
    /// Of `draws_rebind_stopped_arguments`, those that read an attachment of their pass through
    /// slots narrowed to other levels than the kept draw's (`render/attached_levels.rs`).
    pub draws_rebind_stopped_arguments_narrowed: u64,
    /// Of `draws_rebind_stopped_arguments`, those whose set-0 writes, uses or push bytes a rebind
    /// does not work out, or whose narrowed slot its own write meets.
    pub draws_rebind_stopped_arguments_worked: u64,
    /// Of `draws_rebind_stopped_arguments`, those of a changed fixed-function vertex buffer slot
    /// that binds no buffer or bytes, or no offset, where the plan reads it.
    pub draws_rebind_stopped_arguments_vertex_buffers: u64,
    /// Of `draws_rebind_stopped_arguments`, those that use shadowed textures where the kept draw
    /// used none, or none where it used some. One that uses others than the kept draw, or reads
    /// one within other bounds, gathers its own into its instance as every step does and goes on
    /// (`render/draw_again/rebound.rs`).
    pub draws_rebind_stopped_arguments_shadows: u64,
    /// Of `draws_rebind_stopped_arguments`, those whose uses need a barrier after an earlier
    /// draw's geometry writes.
    pub draws_rebind_stopped_arguments_barrier: u64,
    /// Of `draws_rebind_stopped_arguments`, those whose uses meet a detached buffer, a draw of
    /// which records nothing.
    pub draws_rebind_stopped_arguments_detached: u64,
    /// Of `draws_rebind_stopped_arguments`, those that stopped at another step of a rebind: an
    /// error every step reports (`render/draw_again/rebound.rs`'s stops).
    pub draws_rebind_stopped_arguments_other: u64,
    /// The uses of resources a render encoder made resident (`useResource:usage:stages:`,
    /// `useHeap:stages:`) that its draws' binding plans list, as its draws worked them out for
    /// hazard tracking: a draw keeps the last draw's where its plan is the last draw's, or lists
    /// the same uses as the plan they were worked out for, and what the encoder made resident is
    /// unchanged since (no resource made resident, usage or stage
    /// widened, or heap used), and works out none then, a draw rebound or drawn again included;
    /// another works out each its plan lists.
    pub resident_uses_made: u64,
    /// The keys of the buffers among the uses of resources a render encoder made resident
    /// (`resident_uses_made`) that its draws reach through the resident set alone, which no other
    /// use of theirs names, as their draws worked those keys out for hazard tracking: a draw
    /// that works out its uses of what the encoder made resident counts each of them; one that
    /// keeps the last draw's counts only the keys its other uses name and those the last draw's
    /// did, and none where those are the same. A draw's cost here is what it binds, not what its
    /// encoder made resident.
    pub resident_keys_settled: u64,
    /// The uses of resources a render encoder made resident (`resident_uses_made`) that its
    /// draws' gathering walked: compared with the last draw's and gathered into what its render
    /// pass instance's draws use, which the shadowed textures they use are copied around. A draw
    /// that begins with the uses its instance's last draw began with, by identity, walks none.
    pub resident_uses_gathered: u64,
    /// The uses of resources a render encoder made resident (`resident_uses_made`) that its
    /// draws' checks against the bytes their render pass's fragment stages wrote and read walked,
    /// counted whole for each of two: announcing what the device may order apart from Metal
    /// (`Degradation::InPlaceVertexReads`, `Degradation::FragmentsBeforeGeometry`), and noting the
    /// draw's own. A draw that begins with the uses the last draw began with, by identity, notes
    /// none of them again while the pass's notes hold each as it was noted, and checks none again
    /// where none of them had anything to announce while those bytes stand.
    pub resident_uses_against_fragments: u64,
    /// The uses of resources a render encoder made resident (`resident_uses_made`) that its
    /// draws' checks for memory an attachment of their render pass renders walked, while
    /// `Degradation::AttachmentReadInPlace` would be announced. A draw that begins with the uses
    /// found to share none with the pass's attachments, by identity, walks none of them.
    pub resident_uses_against_attachments: u64,
    /// The uses of resources a render encoder made resident (`resident_uses_made`) that its
    /// draws' noting walked: recorded into the encoder's part and touched (each key kept, each
    /// buffer over host pages held) for the hazard tracking of its commands. A draw that begins
    /// with the uses the last draw noted began with, by identity, reaching the same keys through
    /// the resident set alone, in the same part of the encoder while the device's buffers over
    /// host pages stand as they did, walks none of them.
    pub resident_uses_noted: u64,
    /// The uses of resources a render encoder made resident (`resident_uses_made`) that its
    /// draws' adding of their writes to those no memory barrier orders yet walked: what a later
    /// command of the encoder reading them waits for.
    pub resident_uses_unordered_writes: u64,
    /// The uses of resources a render encoder made resident (`resident_uses_made`) that its
    /// draws' adding of what they read as indirect arguments walked: what a later draw of their
    /// render pass instance writing it waits for.
    pub resident_uses_indirect_reads: u64,
    /// The uses of resources a render encoder made resident (`resident_uses_made`) that its
    /// draws' adding of what they write before their fragments walked: what a later draw's
    /// fragments of their render pass instance touching it wait for.
    pub resident_uses_geometry_writes: u64,
    /// The uses of resources a render encoder made resident (`resident_uses_made`) that its
    /// draws' checks against what earlier draws of their render pass instance wrote before their
    /// fragments walked, a rebound draw's check whether it may rebind included. A draw that
    /// begins with the uses the instance's last checked draw began with, by identity, walks only
    /// those of them whose resources a draw wrote before its fragments since.
    pub resident_uses_geometry_checked: u64,
    /// The uses of resources a render encoder made resident (`resident_uses_made`) that its
    /// draws' checks whether they may be kept for drawing again walked: whether they write
    /// nothing, read nothing as indirect arguments, and read an image. A draw walks them once for
    /// each identity they take (as they are made again), not at each draw.
    pub resident_uses_again_checked: u64,
    /// Render draws of a plan that reads handles that worked out its heap part: its heap tables
    /// (`BindingPlan::heap_tables_into`), where its heap writes go
    /// (`BindingPlan::heap_elements`), and the check of each texture a handle names that its
    /// view is one the texture presents. A draw whose bind state's last draw of its pipeline
    /// state worked them out for the same plan, under a residency of the same generation, the same
    /// argument samplers and function tables and the same textures at the plan's heap slots,
    /// works out none: its cost is then what it binds, not what its encoder made resident.
    pub draw_heap_parts_made: u64,
    /// Looks of a render encoder's draws and tile dispatches whose functions read texture handles
    /// at every texture the encoder made resident, for those whose bytes a carrier holds
    /// (`pvrtc::ResidentUnreadable`): one at the encoder's first such command after each change of
    /// what it made resident, none at a command after which it did not change, however many
    /// textures it made resident.
    pub draw_resident_textures_looked_at: u64,
    /// Keys an encoder's commands named first, while the device has a buffer over host pages
    /// (`new_buffer_with_bytes_no_copy`), that were looked up among those buffers under the lock
    /// their command buffers' submissions take: the keys of such buffers, and the rare other key
    /// that shares a bit of their filter with one (`rebind.rs`). A texture's key, or a buffer's of
    /// the runtime's memory, is told apart without the lookup.
    pub host_holds_looked_up: u64,
    /// Compute encoders' dispatches that took the bind state, variant, plan, commands and made
    /// `VkPipeline`s their encoder's last dispatch settled, rather than settling them again:
    /// those of the same pipeline state, dispatch, table (no call changed it since), threadgroup
    /// lengths, imageblock size and stage-in region, in an encoder whose residency did not change
    /// since (no resource made resident, no usage widened, no heap used), of a state that made no
    /// visible-function table, while no no-copy buffer moved, no sampler state was made or went
    /// and no buffer took a resident indirect command buffer's ID (`compute::DispatchMemo`).
    pub dispatches_settled_as_before: u64,
    /// Compute encoders' dispatches that took the bind state, variant, plan, commands and made
    /// `VkPipeline`s an earlier compute encoder of their command buffer settled last, rather than
    /// settling them again: those whose new table, read, brought in the very state that dispatch
    /// bound (no value of any kind changed, the samplers' maps the same), with every other input
    /// of `dispatches_settled_as_before`'s the same. Not counted there.
    pub dispatches_settled_across_encoders: u64,
    /// Dispatches whose commands (`BindingPlan::instantiate_dispatch`'s) `compute::record`
    /// instantiated: of a compute encoder (direct, indirect), of an indirect command buffer's
    /// execution, or of a host step. None for a compute encoder's dispatch counted in
    /// `dispatches_settled_as_before` or `dispatches_instantiated_as_before`.
    pub dispatches_instantiated: u64,
    /// Of `dispatches_instantiated`, compute encoders' direct dispatches whose commands were
    /// written over the storage of those their encoder's kept bind state kept
    /// (`BindingPlan::instantiate_dispatch_into`, `compute::Instantiated`), which nothing else
    /// held any more, rather than made anew: a dispatch of an encoder of its own that binds a
    /// buffer at another offset than its pipeline state's last dispatch, for one. A function of
    /// what the dispatches bind and of their order alone.
    pub dispatches_instantiated_in_place: u64,
    /// Compute encoders' dispatches that took the commands their encoder's kept bind state kept
    /// rather than instantiating them (`compute::Instantiated`): those of the very variant, plan
    /// and dispatch (`pipeline::Planned`) a direct dispatch of the encoder, or of an earlier
    /// compute encoder of the device whose state it took, instantiated last, while no change of
    /// the state since moved a kind of value their plan may read (`BindingPlan::dispatch_reads`):
    /// a dispatch of an encoder of its own that binds two textures of one shape in turn with a
    /// plan that reads no texture's value, for one. Not counted in `dispatches_settled_as_before`.
    pub dispatches_instantiated_as_before: u64,
    /// Compute encoders' dispatches that started from the bind state their pipeline state's last
    /// dispatch left, which their encoder (or an earlier one of the device whose state it took)
    /// set aside while it dispatched through other pipeline states (`compute/live.rs`'s **One per
    /// pipeline state lately dispatched**), rather than from the last dispatch's state: a
    /// dispatch of a pipeline state after dispatches of others, each in an encoder of its own as a
    /// guest's compute workloads encode them, for one. Each is counted again in
    /// `set_aside_commands_kept` or `set_aside_commands_instantiated`, but where it records
    /// nothing.
    pub dispatches_from_set_aside: u64,
    /// Of `dispatches_from_set_aside`, those that took commands their state kept: what their
    /// state's last dispatch settled (`dispatches_settled_as_before`,
    /// `dispatches_settled_across_encoders`) or instantiated (`dispatches_instantiated_as_before`),
    /// no value their commands may read having moved since.
    pub set_aside_commands_kept: u64,
    /// Of `dispatches_from_set_aside`, those that instantiated their commands again
    /// (`dispatches_instantiated`): a value their state's kept commands may read moved since that
    /// state's last dispatch (a buffer bound, or bound at another offset, a resource made
    /// resident), or its shape did.
    pub set_aside_commands_instantiated: u64,
    /// Dispatches of an inert kernel (`ModuleStats::inert`: no loop, and no write but to its
    /// invocations' own and its threadgroups' memory) that recorded no Vulkan dispatch, direct,
    /// indirect and an indirect command buffer's alike (`compute::record`): nothing such a
    /// dispatch does outlasts it, and the runtime samples no counter that would count its
    /// threads. Each is planned as any dispatch (`dispatches_settled_as_before` counts it where
    /// its memo held), and its uses are named, as Metal's command buffer uses what it binds.
    pub dispatches_skipped_inert: u64,
    /// Direct dispatches that settled their bind state again (another table, another residency,
    /// another encoder) into one of the shape their pipeline state's last direct dispatch that
    /// planned planned, of the same dispatch and buffer forms, and took that one's variant and
    /// plan rather than planning them again, and its regions' made `VkPipeline`s where their keys
    /// are the same (`pipeline::Planned`): a dispatch whose buffers' offsets alone moved, and an
    /// encoder's first dispatch of a state dispatched before, for two. Not counted in
    /// `dispatches_settled_as_before`.
    pub dispatches_planned_as_before: u64,
    /// Those of `dispatches_planned_as_before` whose grid was not the one what they took was
    /// planned for, but another of its class: of the same dispatch form and threadgroup size, a
    /// whole number of threadgroups or not alike, one row in threadgroups of one row or not alike,
    /// in the bound buffer form, where that plan runs on a flat or partial-threadgroups variant or
    /// its grid was a whole number of threadgroups, so every region it plans has the call's
    /// threadgroup size (`pipeline::GridClass`), and only after its plan named its variant for
    /// the new grid, the device's limits checked (`BindingPlan::dispatch_variant`): a guest's
    /// fix-up dispatches of a thread for each element, one width after another, plan once.
    pub dispatches_planned_across_grids: u64,
    /// Direct dispatches of whole threadgroups, a `dispatchThreadgroups:` or a `dispatchThreads:`
    /// of a whole number of threadgroups, that ran on their kernel's partial-threadgroups variant
    /// with every thread live: a kernel whose module holds 65,536 SPIR-V words or more and whose
    /// ragged grids run on that variant, so one `VkPipeline` of a threadgroup size runs its whole
    /// and ragged grids (`compute::partial_direct`). Zero for smaller kernels, whose whole grids
    /// take the plain pipeline.
    pub whole_dispatches_on_partial_variants: u64,
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
    /// texture over them went, through the texture's image, or a buffer bound inside the import,
    /// `host_releases_held_inside`), until the import is freed, or, on a
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
    /// Imports of host pages whose free a completed command buffer's release posted to the
    /// device's free thread, which paces it (`free_thread.rs`): the command buffer let go of them
    /// last, their buffers released, while a deallocator or a keep-alive waited for the free,
    /// which runs there after it, possibly after the command buffer's waiters returned.
    pub host_import_frees_posted: u64,
    /// Keep-alives of no-copy buffers and heaps (`KeepAlive`) held now past their release: each
    /// by the import of its pages, whose last buffer it was while something else (a texture's
    /// image a submitted command buffer may still read, or a buffer bound inside the import,
    /// `host_releases_held_inside`) held the import, or that a live buffers' table a command
    /// buffer that may reach them unnamed holds listed as it was released, until the import is
    /// freed. One per import at most: a later keep-alive over the same import is dropped at its
    /// release (`host_keep_alives_covered`).
    pub host_keep_alives_kept: u64,
    /// Keep-alives dropped at their buffer's release, after its deallocator ran, because the
    /// import of its pages stays mapped without them: another no-copy buffer made over the same
    /// bytes lies in it, or it keeps another's keep-alive already.
    pub host_keep_alives_covered: u64,
    /// No-copy buffers released while a buffer bound inside the import of their pages lived
    /// (`host_aliases.rs`, "One page, one import": a buffer over part of their bytes, or over
    /// the same bytes as memory of another class, made while they lived, with no import of its
    /// own, `ObjectChurn::buffers_bound_in_import`), so that the import outlived their release:
    /// their keep-alive kept by the import (`host_keep_alives_kept`), their deallocator run at the
    /// release, as Metal's; or, given no keep-alive, their deallocator waiting for the import's
    /// free (`host_deallocators_delayed`), where Metal's runs at the release. Either goes once
    /// that buffer, and everything else holding the import, went. One live import of a page at a
    /// time: `VK_EXT_external_memory_host` leaves a host allocation imported twice into one
    /// device undefined. Counted at each such release.
    pub host_releases_held_inside: u64,
    /// The process's mappings the kernel was asked about, for their write permission, before a
    /// no-copy buffer's, heap's or rebind's own import of host pages, on Linux (one `PROCMAP_QUERY`
    /// each, or one line of `/proc/self/maps` where the kernel predates the query) and on Windows
    /// (one `VirtualQuery` each region): one per mapping the import's pages lie in, a view of
    /// scattered pages one per run, at each such import. A buffer made over the same bytes as a
    /// live one, which lies in its import, asks about none. None on macOS.
    pub host_mappings_queried: u64,
    /// Imports of dma-bufs of host pages a caller's source made (`HostMemory::with_dma_buf`),
    /// where the device refused the pages' pointer: no-copy buffers', heaps' and rebinds' alike,
    /// one per chunk that is a dma-buf of its own, and one per shared file at its first piece
    /// (`DmaBufChunk`), whose later pieces join it (`dma_buf_pieces_joined`), or as a handle keeps
    /// it (`Device::import_shared_dma_buf`), where the device held no import of it. None where
    /// every pointer was imported.
    pub dma_buf_imports: u64,
    /// Imports of a dma-buf of host pages that back one `VkBuffer` alone for their life (a
    /// no-copy buffer's own import of one chunk, a heap's never; a copy's list of a guest RAM's
    /// runs; the page a use of a RAM asks the driver with) made dedicated to that buffer
    /// (`VkMemoryDedicatedAllocateInfo`), where the driver refused the import dedicated to
    /// nothing, the form every other such import takes, in every memory type tried. NVIDIA's
    /// 615.71.09 refuses a dma-buf of one 4 KiB page so (`ERROR_OUT_OF_DEVICE_MEMORY`) and takes
    /// it dedicated; one of 64 KiB it takes either way. Counted in `dma_buf_imports` too.
    pub dma_buf_buffer_imports_dedicated: u64,
    /// Those imports live now: each a `VkDeviceMemory` the device holds until the last buffer,
    /// texture or command buffer over its pages (a shared file's: over any of its pieces) lets
    /// go. RADV lists each, as every imported memory, in every kernel submission of the device
    /// (Mesa 26.2.4's global BO list), and amdgpu's work per submission grows with them
    /// (`runtime_lane::submit_per_import`).
    pub dma_buf_memories_live: u64,
    /// Pieces of a file a caller's source shares (`DmaBufChunk`) bound in the device's import of
    /// that file, live already as the buffer, heap or rebind was made over them: each a buffer's
    /// chunk that took no import of its own.
    pub dma_buf_pieces_joined: u64,
    /// The device's imports of files a caller's source shares live now, one per file whatever
    /// the pieces bound in it, freed with the last buffer or heap bound in it. Counted in
    /// `dma_buf_memories_live` too.
    pub dma_buf_shared_imports_live: u64,
    /// Of those, the imports kept now by a live `SharedDmaBufImport` of their file
    /// (`Device::import_shared_dma_buf`), which stay imported while no buffer or heap is bound in
    /// them. A handle whose device keeps nothing (it imports no dma-buf for a buffer) adds none,
    /// and after a recovery a handle adds one again once a buffer on the new device imports its
    /// file.
    pub dma_buf_shared_imports_kept: u64,
    /// The device's imports of dma-bufs of host pages tried, made or failed, wherever they were
    /// made: each the driver's `vkGetMemoryFdPropertiesKHR` of the dma-buf and its
    /// `vkAllocateMemory` of the import, in every memory type and form tried. A no-copy buffer's,
    /// heap's or rebind's chunk of its own, a shared file's import (at its first piece, ahead as a
    /// handle keeps it, or again at a piece after the residency released it), a buffer's import
    /// made again after the residency released it, a guest RAM's chunk or list of runs taken as a
    /// dma-buf (whose time is in `host_ram_chunk_import_nanos` or `host_ram_run_list_import_nanos`
    /// too, with the source's), and an image's dma-buf import. A piece that joins a live import
    /// makes none (`dma_buf_pieces_joined`).
    pub dma_buf_import_calls: u64,
    /// The nanoseconds those imports took, on the threads that made them: the encoding thread's
    /// for an import made again at a blit, the making thread's for an object's or a handle's.
    pub dma_buf_import_nanos: u64,
    /// The nanoseconds the joins of pieces to a live import of their shared file took
    /// (`dma_buf_pieces_joined`): each a look in the device's map of imports and a hold of the
    /// import, with no call to the driver.
    pub dma_buf_join_nanos: u64,
    /// The dma-buf imports of images over such memory live now (`dma_buf_images`: each image's
    /// own, a dedicated one, or one bound at its texels' offset, `dma_buf_images_at_offset`), and
    /// the dedicated ones of images over the runtime's memory exported as a dma-buf
    /// (`exported_dma_buf_images`): one per live image, listed in every kernel submission as those
    /// above, freed after the image.
    pub dma_buf_image_memories_live: u64,
    /// No-copy buffers' imports of dma-bufs released while idle, their `VkBuffer` destroyed and
    /// their memory freed, the dma-bufs kept: a runtime-internal residency, which nothing of
    /// Metal's observes (`HostMemory::with_dma_buf` documents it). Their chunks of their own leave
    /// `dma_buf_memories_live`; a shared file's import leaves it once every buffer bound in it
    /// went or was released.
    pub host_imports_released: u64,
    /// Of `host_imports_released`, those released while the releasable imports held more imports
    /// on the device's list than it keeps at most, 64 where a held import adds to every
    /// submission's cost (its import tax check: RADV, ANV on i915), a shared dma-buf file's import
    /// counted once and its pieces released together, the least recently used first, whatever
    /// their idle windows (`host_residency.rs`). 0 on every other device.
    pub host_imports_released_over_most: u64,
    /// Released imports imported again from the dma-bufs kept, each at the first blit that read
    /// the buffer after its release (on the encoding thread).
    pub host_imports_reimported: u64,
    /// No-copy buffers made over the bytes of a releasable import another buffer lies in, which
    /// left it releasable: an import stays so while every read of each buffer in it was a blit's,
    /// or none (`host_residency.rs`, `host_aliases.rs`).
    pub host_imports_joined_releasable: u64,
    /// The releasable imports that hold their memory now: those no read but a blit's took, which
    /// the device may release while idle, and releases past its most where a held import adds to
    /// every submission's cost (`host_imports_released_over_most`, which counts the imports on the
    /// device's list they hold: pieces of one shared dma-buf file hold one). An import a read
    /// other than a blit's took leaves it for `host_imports_held_for_good`.
    pub host_imports_resident: u64,
    /// The imports that were releasable until a read other than a blit's took them (a binding, a
    /// texture over the bytes, the Metal address or `VkBuffer` handed out, a buffer bound inside
    /// the import or sharing its buffer's range, a heap's bytes or a rebound buffer lying in it:
    /// `host_imports_read_apart`), which hold their memory now: kept imported for good, whatever
    /// the device's most, until their buffers go. With `host_imports_resident`,
    /// every import with a residency that holds its memory.
    pub host_imports_held_for_good: u64,
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
    /// `VkBuffer` handed out, a binding, a texture over the bytes, a buffer bound inside the
    /// import or sharing its buffer's range, a heap's bytes or a rebound buffer lying in it):
    /// those a release while idle could never take, as the device keeps a releasable import for
    /// good after such a read. A no-copy buffer made over the import's bytes is none: it reads the
    /// import as its first buffer does (`host_imports_joined_releasable`). Of imports of any
    /// kind.
    pub host_imports_read_apart: u64,
    /// Of those, the ones whose buffer's Metal address was handed out: `gpu_address` (a heap's
    /// over the import's bytes too), or written for the caller into an argument buffer or an
    /// indirect command buffer. Each of these nine counts an import once for its reason, so an
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
    /// Of those, the ones a buffer is bound inside, with a `VkBuffer` of its own: one whose pages
    /// the import holds all of, not over the same bytes (`host_aliases.rs`, "One page, one
    /// import").
    pub host_imports_read_apart_inside: u64,
    /// Of those, the ones whose buffer a buffer made over the same host memory shares its range
    /// with (`BufferDescriptor::range_of`), its Metal address handed to that buffer.
    pub host_imports_read_apart_shared_range: u64,
    /// Of those, the ones a no-copy heap's own bytes lie in, over the bytes a buffer of the
    /// import was made over: the heap's placed buffers' addresses are the heap's.
    pub host_imports_read_apart_heap: u64,
    /// Of those, the ones a rebound buffer lies in (`Buffer::rebind_bytes_no_copy` onto the
    /// bytes a buffer of the import was made over).
    pub host_imports_read_apart_rebound: u64,
    /// Of those, the releasable ones kept imported because the host could start no thread to
    /// release them.
    pub host_imports_read_apart_unreleased: u64,
    /// The caller's reports of host addresses whose pages it replaced
    /// (`Device::host_range_remapped`), whatever lies over them.
    pub host_remaps_reported: u64,
    /// Of those, the reports with a page of a live host import among the addresses, rounded out
    /// to the import's unit: a replacement under a live import, which `HostMemory::new`'s
    /// contract forbids, after which what the import reads is unspecified (docs/FAST-PATHS.md,
    /// known gaps).
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
    /// The imports of host pages by their pointer the runtime refused before the driver saw them,
    /// because the kernel pins no page of a mapping over them (a mapping of a dma-buf), on a
    /// driver whose failed import of such pages breaks its later allocations
    /// (`RuntimeDriverDefect::UnpinnablePagesBreakHostMemory`): no-copy objects', rebinds' and
    /// guest RAM's chunks'. Such an object takes the caller's dma-bufs of the pages, or else a
    /// copy of them (`copied_host_buffers_degraded`). None on every other driver, and off Linux.
    pub host_imports_unpinnable: u64,
    /// The no-copy buffers' imports the caller's gate refused (`HostMemory::with_import_gate`):
    /// each made a copied host buffer instead (`copied_host_buffers_refused`). A read twin's
    /// refused import is counted in `read_twins_unimported` alone.
    pub host_imports_refused_by_host: u64,
    /// The copied host buffers made (`copied-host-pages`): no-copy buffers whose bytes the GPU
    /// uses in a copy of the caller's pages in memory of the device's own, which no import maps.
    /// Each is one of the next six.
    pub copied_host_buffers_made: u64,
    /// Of those, the ones made because the caller's gate refused the import.
    pub copied_host_buffers_refused: u64,
    /// Of those, the ones made because the device refused the pointer's import, or imports no
    /// pointer (it enables no `VK_EXT_external_memory_host`), and no dma-buf of the pages served
    /// it: the caller gave no source of them (`HostMemory::with_dma_buf`), or its source failed,
    /// or the device refused its dma-bufs. RADV refuses shared memory, a `memfd` mapping among it
    /// (`Device::imports_shared_host_pointers`).
    pub copied_host_buffers_device_refused: u64,
    /// Of those, the ones made because an import the device and the gate admitted failed (out
    /// of memory, another Vulkan error), or because the runtime refused the pages for the driver
    /// and no dma-buf of them served (`host_imports_unpinnable`).
    pub copied_host_buffers_degraded: u64,
    /// Of those, the ones a rebind made because the new bytes lie at another offset in their
    /// page than the old, where a texture over the buffer was planned at the old offset
    /// (`Buffer::rebind_bytes_no_copy`): the copy keeps the bytes at the old offset.
    pub copied_rebind_pages_moved: u64,
    /// Of `copied_host_buffers_made`: no-copy buffers, heaps and rebinds of memory whose GPU writes
    /// Metal's private option bit 24 drops, made over a copy that drops them where the system
    /// maps no read-only view of the pages (`copied.rs`, `read_only_view.rs`).
    pub copied_read_only_memory: u64,
    /// Of `copied_host_buffers_made`: no-copy buffers, heaps and rebinds over pages the process
    /// maps read-only, on Linux, made over a copy that drops the GPU's writes, as Metal drops
    /// them: Linux drivers pin a pointer's imported pages for writing, and a dma-buf of the pages
    /// would take the GPU's writes (`buffer_descriptor.rs`, `copied.rs`). Their import is never
    /// tried and their dma-buf source never called; the caller's gate is asked first where the
    /// pointer would be imported, and a copy it refused is counted as refused.
    pub copied_unwritable_pages: u64,
    /// Of `copied_host_buffers_made`: no-copy buffers and heaps whose pages a live import of
    /// another held some of, and not all (a buffer's bytes across the edge of another's or a
    /// heap's), or a heap whose pages one held more of than the heap's own bytes, and those whose
    /// pages an import of a guest RAM's chunk (made or in flight) held some of at the RAM's own
    /// addresses where no chunk bound them (their bytes across a chunk's end, or routed by no run
    /// resolver), made
    /// over a copy so that no page is imported twice (`host_aliases.rs`, "One page, one import";
    /// `host_ram.rs`): `VK_EXT_external_memory_host` leaves a host allocation imported twice into
    /// a device undefined. Their import is never tried, their gate never asked. Pages with a
    /// source of dma-bufs beside live imports of another's that are all of dma-bufs import their
    /// own instead (`ObjectChurn::dma_buf_imports_beside_live`), but on a device whose barriers
    /// do not order blits through two imports of the same pages
    /// (`RuntimeDriverDefect::AliasedImportBlitsUnordered`).
    pub copied_over_live_imports: u64,
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
    /// The no-copy buffers made over, and the rebinds to, the bytes a live copied host buffer was
    /// made over (the same first byte and length) that lay in its copy rather than making one of
    /// their own (`host_aliases.rs`): one `VkBuffer`, one snapshot and one hazard key for the
    /// memory, as an import of it is one, the gate not asked. In none of the counts of copies
    /// made.
    pub copied_host_buffers_shared: u64,
    /// The bytes the CPU changed in copied host buffers' pages that commits copied into the
    /// copies.
    pub copied_copy_in_bytes: u64,
    /// The bytes of copied host buffers that commits' copies in did not compare: their pages'
    /// memory came with a log of the writes to them (`HostMemory::with_dirty_pages`,
    /// `HostRam::with_dirty_pages`), which reported none of them written since the copy last
    /// synced to it.
    pub copied_copy_in_bytes_skipped: u64,
    /// The copied host buffers commits' copies in left alone, one per copy and commit: no
    /// command of the committed command buffer used them (none reached memory its uses do not
    /// name), so neither their bytes nor their pages' logs were asked; the first commit that uses
    /// one takes every write since.
    pub copied_copy_ins_unused: u64,
    /// The copies in again of copied host buffers, one per batch: a batch that waits for an event
    /// value its command buffer's commit found not reached, before the commit's copy in, copies
    /// in again as it is submitted, after the wait, so its commands read the CPU's writes made
    /// before the signal, as Metal's do. A command buffer with no such wait makes none.
    pub copied_copy_ins_after_waits: u64,
    /// The bytes the GPU changed in copied host buffers' copies that batches' completions merged
    /// back into the caller's pages.
    pub copied_merge_back_bytes: u64,
    /// The copied host buffers completion turns' merges back compared, one per copy and turn.
    pub copied_merges_compared: u64,
    /// The copied host buffers completion turns' merges back left uncompared: no batch the turn
    /// finished could have written them (no use of their command buffers' commands wrote one of
    /// their keys, and none may have written memory its uses do not name).
    pub copied_merges_skipped: u64,
    /// The bytes the GPU changed in copies of pages the process cannot write (read-only pages)
    /// that batches' completions dropped: taken out of the copies, the pages' bytes in their
    /// place, never written to the pages, as Metal drops the GPU's writes to read-only memory.
    pub copied_dropped_bytes: u64,
    /// The copied host buffers that took a base: the bytes the pages and the copy last agreed on,
    /// made at the first commit whose command buffer may write the copy (until then the copy
    /// holds them, and a copy the GPU only reads never takes one), a 128-bit tag of each 1 KiB
    /// block in host memory, and the bytes in the device's memory where it has room for them
    /// (device-local memory the host maps, in a heap a resizable BAR makes as large as the
    /// device's memory), else in the host's.
    pub copied_bases_made: u64,
    /// The commits that found no host memory for a copied host buffer's base: each failed with
    /// `Error::OutOfMemory` and ran nothing, the copy left with none.
    pub copied_bases_refused: u64,
    /// The 1 KiB blocks of copied host buffers' bases that comparisons read: blocks the CPU and
    /// the GPU both changed since the pages and the copy last agreed. A base in the device's
    /// memory is read across the bus, uncached, by the host, or by the GPU's copy where one
    /// comparison reads many (`copied_base_blocks_read_back`).
    pub copied_base_blocks_read: u64,
    /// Of `copied_base_blocks_read`, the blocks of bases in the device's memory that comparisons
    /// read through one copy into host memory on the device's staging queue each, where a
    /// comparison read enough of them that the GPU's copy costs less than the host's uncached
    /// reads (the device has a staging queue wherever a base lies in its memory).
    pub copied_base_blocks_read_back: u64,
    /// The threads comparisons of copied host buffers started beside the calling thread: a
    /// commit's copy in or a completion's merge back of a copy with a base of 16 MiB or more
    /// splits its blocks across up to four threads, the calling one among them, each run at
    /// least 8 MiB, and the threads it started end before it returns.
    pub copied_compare_threads: u64,
    /// The copied host buffers commits' copies in compared, one per copy and commit: a command
    /// of the committed command buffer used one, or may have reached it unnamed (the
    /// counterpart of `copied_copy_ins_unused`).
    pub copied_copy_ins_compared: u64,
    /// The bytes of copied host buffers commits' copies in compared with their pages: every
    /// byte of a copy compared, but those its pages' log reported unwritten
    /// (`copied_copy_in_bytes_skipped`).
    pub copied_copy_in_bytes_compared: u64,
    /// The copies in (at a commit, or again after an event wait) whose command buffer reached
    /// memory no use names (a root table's loaded addresses) and compared a copied host buffer
    /// no use of it named: one per copy in, whatever the copies it compared so.
    pub copied_unnamed_reach_copy_ins: u64,
    /// The bytes those copies in compared of the copies no use named
    /// (`copied_unnamed_reach_copy_ins`): a part of `copied_copy_in_bytes_compared`.
    pub copied_unnamed_reach_bytes_compared: u64,
    /// Of `copied_unnamed_reach_copy_ins`, those whose command buffer had every use noted and no
    /// function that may store through an address (`State::writes_unnamed`), and compared a copy
    /// no use named of a no-copy buffer of Metal's private option bit 24 with a read twin
    /// (`read_twin.rs`): those a resident set listing the twins in their buffers' place would
    /// spare that copy.
    pub copied_unnamed_reach_twin_copy_ins: u64,
    /// The bytes those copies in compared of such twinned copies
    /// (`copied_unnamed_reach_twin_copy_ins`): a part of `copied_unnamed_reach_bytes_compared`.
    pub copied_unnamed_reach_twin_bytes_compared: u64,
    /// The read twins made (`read_twin.rs`): of a no-copy buffer of Metal's private option bit
    /// 24 where the system maps no read-only view of its pages, the runtime's own buffer over the
    /// pages in place, which the commands that cannot write the buffer read instead of its copy.
    pub read_twins_made: u64,
    /// The no-copy buffers of bit 24 that took no read twin because their pages' import would
    /// have been a copy (the caller's gate refused it, the process cannot write the pages, a
    /// live import holds some of them) or failed: they are their copy alone.
    pub read_twins_unimported: u64,
    /// The no-copy buffers of bit 24 that took no read twin because the device's policy contains
    /// no access (`ContainmentPolicy::UnsafeOff`): a store may reach past what it names.
    pub read_twins_uncontained: u64,
    /// The blit reads of a buffer's bytes made through its read twin, in place, one per copy
    /// whose source it is.
    pub read_twin_blit_reads: u64,
    /// The reads a read twin could have taken that read the buffer's copy instead, one per
    /// command: an earlier command of the command buffer may have written the copy, and a later
    /// one reads that write back, as Metal's GPU caches let it.
    pub read_twin_reads_after_writes: u64,
    /// The commands an encoder refused because their uses write a read twin: never, but for a
    /// defect of the runtime's, which the refusal keeps from reaching the pages.
    pub read_twin_writes_refused: u64,
    /// The read twins of textures made (`read_twin.rs`): of a program's texture over the bytes of
    /// a no-copy buffer with a read twin, the runtime's texture over the twin's bytes in place,
    /// which the slots that only read the texture bind instead.
    pub read_twin_textures_made: u64,
    /// The textures over a no-copy buffer with a read twin whose twin's texture would not lie in
    /// place over the twin's bytes (a shadow, pending or refused by the device's check of
    /// textures in place).
    pub read_twin_textures_not_in_place: u64,
    /// Of `read_twin_textures_not_in_place`, those whose twin is kept as a shadow of the twin's
    /// bytes, the texture itself being a shadow (of its buffer's copy): the slots that only read
    /// the texture bind it, copied in from the pages in place, where they copied the texture's
    /// own shadow in from the copy, which their commits compared (`read_twin.rs`).
    pub read_twin_texture_shadows: u64,
    /// The slot bindings made through a read twin, in place, one per slot and command (a draw or
    /// a dispatch): a texture slot's through its texture's twin, a buffer slot's through its
    /// buffer's.
    pub read_twin_slot_reads: u64,
    /// The host bytes the live copied host buffers' bases hold (a gauge): their tags, a 64th of
    /// each copy's length, and the bytes of those kept in host memory.
    pub copied_base_host_bytes: u64,
    /// The device bytes the live copied host buffers' bases hold (a gauge): the bytes of those
    /// kept in the device's memory.
    pub copied_base_device_bytes: u64,
    /// The 1 KiB blocks of the live copied host buffers' bases in the device's memory that are
    /// kept in host memory (a gauge, their bytes among `copied_base_host_bytes`): each block a
    /// comparison read there (the CPU and the GPU both changed it), held in host memory for the
    /// base's life, so a block both sides write again is read across the bus once.
    pub copied_base_host_blocks: u64,
    /// The copies between a guest RAM and a buffer blit encoders recorded
    /// (`BlitCommandEncoder::copy_from_host_ram`, `copy_to_host_ram`), each from the device's
    /// imports of the RAM's chunks.
    pub host_ram_copies: u64,
    /// Their bytes.
    pub host_ram_copy_bytes: u64,
    /// Their dispatches: one for each array of the imports a copy's runs touch (the chunks', or a
    /// copy's own of its pages: up to the device's `PagesCopyModule::imports` an array, 30 where
    /// it pushes descriptors and indexes arrays of storage buffers; more where its bytes pass the
    /// device's storage range or its pages of an array its workgroup count), never one per run, of
    /// the copies whose runs all start and end on a word.
    pub host_ram_copy_dispatches: u64,
    /// The regions of `vkCmdCopyBuffer` of the copies whose runs are not all whole pages (a run
    /// that starts or ends inside a page): one for each run's bytes in a chunk. And the regions of
    /// `vkCmdCopyBufferToImage` and `vkCmdCopyImageToBuffer` of the copies between a guest RAM and
    /// a texture (`BlitCommandEncoder::copy_texture_from_host_ram`, `copy_texture_to_host_ram`):
    /// one for each run of rows whose host bytes lie the pitch apart in one chunk, and one for each
    /// piece of a row across a chunk's end or a break in its host bytes.
    pub host_ram_copy_regions: u64,
    /// The chunks of guest RAM the device imported (`Device::use_host_ram`): 64 MiB of a RAM's
    /// range each (its last fewer), each once, by the device's own thread ahead of the objects
    /// over it or by the first object made over it where that thread had not imported it.
    pub host_ram_chunks_imported: u64,
    /// The nanoseconds those imports took, whoever made them (the device's thread, an object, a
    /// copy), failed imports' included: each pins its chunk's pages for the device.
    pub host_ram_chunk_import_nanos: u64,
    /// Copies between a guest RAM and a buffer (`BlitCommandEncoder::copy_from_host_ram` and its
    /// kin) whose encoding waited for a chunk's import: one it made, or one under way on another
    /// thread, where the chunk was not imported and its bytes could not be imported alone.
    pub host_ram_copy_import_waits: u64,
    /// The nanoseconds those copies' encodings waited so.
    pub host_ram_copy_import_wait_nanos: u64,
    /// Copies between a guest RAM and a buffer by dispatches recorded by the plan kept of the
    /// same copy before them (the same direction, runs and buffer), its table and dispatches
    /// taken again, not planned: the buffer's `VkBuffer`, byte and pages, the device's figures
    /// and the imports the copy takes (by serial number) were the plan's.
    pub host_ram_copy_plans_reused: u64,
    /// Plans kept of copies between a guest RAM and a buffer whose table went into a buffer of
    /// its own at their first reuse: each later reuse binds it, uploading nothing.
    pub host_ram_copy_tables_kept: u64,
    /// Gathers from a guest RAM (`BlitCommandEncoder::copy_from_host_ram` and
    /// `copy_runs_from_host_ram`) staged through the CPU: over a chunk the device's thread was
    /// to import, which the copy would have waited for, its runs were copied by the CPU into the
    /// command buffer's memory before its batch was submitted, and from there by the GPU into
    /// the buffer, with no import made or waited for.
    pub host_ram_copies_staged: u64,
    /// Of those, the ones whose CPU copy waited for every batch submitted before it to complete:
    /// a buffer over host pages, or a copy into the RAM, lay over the runs' bytes, so the GPU may
    /// have been writing them.
    pub host_ram_staged_fills_gated: u64,
    /// The nanoseconds the CPU copies of those gathers took, on the device's threads.
    pub host_ram_staged_fill_nanos: u64,
    /// Scatters into a guest RAM (`BlitCommandEncoder::copy_to_host_ram` and
    /// `copy_runs_to_host_ram`) staged through the CPU: over a chunk the device's thread was to
    /// import, which the copy would have imported its ranges of alone or waited for, the GPU
    /// copied its bytes into memory of the command buffer's, which a host step took, and the CPU
    /// copied them on into the pages once that copy completed, before the command buffer did,
    /// with no import made or waited for and no memory made on the encoding thread.
    pub host_ram_scatters_staged: u64,
    /// The nanoseconds the CPU copies of those scatters took, on the device's threads.
    pub host_ram_staged_drain_nanos: u64,
    /// The imports of guest RAM, chunks and objects' runs alike, made as one dma-buf each of the
    /// RAM's source (`HostRam::with_dma_buf`), not by their pointer: where the device's held
    /// pointer imports add to every submission's cost, or it refused the pointer.
    pub host_ram_dma_buf_imports: u64,
    /// The imports of a copy's runs of guest RAM, each list of up to the count the RAM was given
    /// as one dma-buf of its source of run lists (`HostRam::with_dma_buf_of_runs`): a copy by
    /// runs over a chunk not imported, on a device whose held imports add to every submission's
    /// cost and whose chunks of the RAM are its dma-bufs (RADV's). Each is live in
    /// `run_imports_live` while a copy's command buffer holds it or the RAM keeps it for the
    /// copies after it (at most 16 such, released once no copy took one for 2 s): a copy whose
    /// runs a kept list holds imports none for them.
    pub host_ram_run_list_imports: u64,
    /// The nanoseconds those imports took on the copies' encoding threads, failed ones included:
    /// the source's dma-buf of the list's pages (`UDMABUF_CREATE_LIST`, which pins them) and the
    /// device's import of it. Apart from `host_ram_chunk_import_nanos`, whose imports are a
    /// chunk's, and from `host_ram_copy_import_wait_nanos`, whose waits are for a chunk's import:
    /// a list is the copy's own import of its pages, as a range's alone is, and no wait for one.
    pub host_ram_run_list_import_nanos: u64,
    /// The imports of guest RAM, chunks and objects' runs alike, by their pointer that failed:
    /// the driver or the runtime refused the pages (RADV refuses any shared mapping's), or memory
    /// ran out. For a RAM with a source of dma-bufs each was then tried as one dma-buf where the
    /// pointer went first (`host_ram_dma_buf_imports`); without one the object over it takes its
    /// own import.
    pub host_ram_pointer_imports_failed: u64,
    /// The imports of guest RAM, chunks and objects' runs alike, as one dma-buf each of the RAM's
    /// source (`HostRam::with_dma_buf`) that failed on a device that imports dma-bufs: the source
    /// failed or answered other lengths, or the driver refused the dma-buf (NVIDIA's 615.71.09
    /// refuses a udmabuf of a few host pages imported dedicated to nothing, as a chunk's or a
    /// run's import is, as out of device memory). Each was then imported by its
    /// pointer where the dma-buf went first, where the device's held pointer imports add to every
    /// submission's cost, which that import then adds (`host_ram_pointer_imports_failed` where
    /// it failed too). Counted too: the import of a RAM's first page as one that asks the driver
    /// as its use begins, where its chunks would be the source's dma-bufs (`Device::use_host_ram`
    /// on such a device), refused in both forms that import tries
    /// (`dma_buf_buffer_imports_dedicated`), which leaves them to their pointer where the refusal
    /// lasts. A refusal that lasts (the driver refuses the dma-buf itself, or imports it as no
    /// memory type its buffer takes) is remembered for the device's life: no later span or copy's
    /// list asks a source, none is counted here again, and no RAM's chunks are its dma-bufs.
    /// Memory that ran out is that import's alone, and a later one asks again.
    pub host_ram_dma_buf_imports_failed: u64,
    /// Of `host_ram_chunks_imported`, those the device's own thread imported ahead of every
    /// object over them, so that no making waited for their import: the few chunks past the one
    /// an object last took; and every chunk of a RAM imported whole as the device's use of it
    /// began (`Device::use_host_ram`, a RAM every page of which was resident). The imports stop
    /// where the device's allocations would leave less than a quarter of its
    /// `maxMemoryAllocationCount`.
    pub host_ram_chunks_imported_ahead: u64,
    /// The times a chunk of guest RAM was not imported because a live import of a no-copy
    /// buffer's or heap's own pages held some of its pages: one made outside the RAM's routing
    /// (before the device's use of the RAM began, with no run resolver, or over a chunk not
    /// imported where each listed import costs every submission). A page a live import holds is
    /// never imported again while it lives (`VK_EXT_external_memory_host`'s issue 6), so the chunk
    /// waits for every such import over it to go: the objects over it meanwhile take their own
    /// paths, as over memory no RAM holds, and the copies over it go through the CPU. Not counted
    /// where every such import is of dma-bufs, each listed import costs every submission and the
    /// device's barriers order blits through two imports of the same pages: the chunk is then
    /// imported beside them, as one dma-buf (`ObjectChurn::dma_buf_imports_beside_live`).
    /// Each ask of the chunk while it waits counts once: the device's thread's, an object's, a
    /// copy's.
    pub host_ram_chunks_held_back: u64,
    /// The chunks of guest RAM the device's thread faulted in, on Linux, before it imported them
    /// ahead of the objects: none of a chunk the device refuses whatever its pages hold (shared
    /// memory on a device that imports no pointer to it, the RAM given no source of dma-bufs:
    /// RADV), whose pages stay as the program left them. Elsewhere the import faults them in.
    pub host_ram_chunks_faulted_ahead: u64,
    /// Of `host_ram_chunks_imported_ahead`, those freed again before any object took them, as the
    /// device's other allocations came within that quarter of its limit, so that they are never
    /// what an allocation is refused for (`memory_allocations_refused_at_limit`). An object over
    /// one later imports it as it is made.
    pub host_ram_chunks_given_back: u64,
    /// Of `host_ram_chunks_imported`, those freed while the device's use of their RAM lasted,
    /// as no buffer bound in one and no pending command buffer that used one held it for its
    /// idle window, on a device whose held imports add to the cost of every submission (its
    /// submission checks' import tax; HOWTO.md, section 6): there each listed import
    /// costs every submission, so an idle chunk is let go of, and the next object over it imports
    /// it again. The window starts at 2 seconds and doubles at each release of the chunk, up to 64.
    pub host_ram_chunks_released: u64,
    /// Their bytes.
    pub host_ram_chunk_bytes: u64,
    /// The chunks of guest RAM imported and not yet freed (a gauge): a chunk is freed once the
    /// device's use of its RAM ended and nothing holds it.
    pub host_ram_chunks_live: u64,
    /// The no-copy buffers (and no-copy heaps' own buffers, and the read twins of no-copy buffers
    /// of Metal's private option bit 24, `read_twins_made`) bound in a chunk of guest RAM, whose
    /// pages their caller resolved to one run inside one chunk (`HostMemory::with_runs`): each
    /// made with no import of its own, its import gate not asked. Where each listed import costs
    /// every submission (RADV), only over a chunk a copy imported: an object over a chunk not
    /// imported takes its own path, and asks for no chunk.
    pub chunk_bound_buffers_made: u64,
    /// Every import of runs of guest RAM made so far for copies, those `run_imports_live` counts
    /// while they live: a copy's ranges alone, and a copy's lists (`host_ram_run_list_imports`). With `run_imports_live`, the imports of runs freed so
    /// far; with `host_ram_chunks_imported` and `host_ram_chunks_live`, those of chunks; the
    /// rest of `memory_frees.host_import` are imports of objects' own pages.
    pub run_imports_made: u64,
    /// The imports of a copy's own runs of guest RAM (`BlitCommandEncoder::copy_from_host_ram`,
    /// `copy_to_host_ram`: alone, or as a list, `host_ram_run_list_imports`) that are not yet let
    /// go of (a gauge): each goes with the last command buffer that used it, whatever became of
    /// the chunk; a list, once the RAM let it go of too (released while idle, or past the lists it
    /// keeps).
    pub run_imports_live: u64,
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
    /// idle as the device's submission checks did not find a held pointer import to cost every
    /// submission (`Device::releases_idle_pointer_imports`), or had not answered as the buffer
    /// was made, or off Unix.
    pub host_imports_kept_pointer: u64,
    /// The live no-copy buffers' imports the device would release while idle, but keeps, as it
    /// could make no stand-in buffer for a read of a released one.
    pub host_imports_kept_no_stand_in: u64,
    /// Of those buffers, heaps and rebinds, the ones whose `VkBuffer` is a sparse buffer
    /// assembled from several chunks.
    pub dma_buf_sparse_buffers: u64,
    /// The `VkSparseMemoryBind`s those sparse buffers were bound by: one a chunk. RADV makes each
    /// one a GPU virtual-address operation, which updates the GPU's page tables at the binding
    /// and again at the buffer's release (`DmaBufRequest`'s docs say what that costs).
    pub dma_buf_sparse_binds: u64,
    /// Images of textures placed in a no-copy heap over such dma-buf memory (or a view of one),
    /// and of textures in place over a no-copy buffer's such pages (`dma_buf_textures_in_place`),
    /// each a `DRM_FORMAT_MOD_LINEAR` image bound to an import of a dma-buf of its own pages from
    /// the caller's source, aliasing the heap's or buffer's bytes: one per image made, a
    /// rebind's too. Not counted in `dma_buf_imports`.
    pub dma_buf_images: u64,
    /// Of `dma_buf_images`, those whose plane lies at offset 0 of an image bound at its texels'
    /// offset in their first page, in an import of their pages that is no dedicated allocation:
    /// made where the driver lays out an explicit plane at that offset at another one
    /// (`vkGetImageSubresourceLayout`), and takes such a binding (it requires no dedicated
    /// allocation of the image, and its memory's alignment divides the offset). Elsewhere an
    /// image's plane starts at its texels' offset, bound at offset 0 of a dedicated import.
    pub dma_buf_images_at_offset: u64,
    /// Parks of images in place over a no-copy buffer's dma-buf pages (`dma_buf_images`), each
    /// kept as its texture went, with its import, under the import of the pages its buffer lies
    /// in, for the next texture in place over the same pages of the same image plan at the same
    /// texels and pitches: one whose `VkImage` was never handed out, while the device keeps fewer
    /// than 64 such images. An image taken and parked again is counted again. A runtime-internal
    /// reuse, which nothing of Metal's observes (`HostMemory::with_dma_buf` documents it).
    pub dma_buf_images_parked: u64,
    /// Textures in place over a no-copy buffer's dma-buf pages whose image was a parked one
    /// (`dma_buf_images_parked`) taken for them: no source asked and nothing imported, the image
    /// resting in `GENERAL` on the device's queue family where it was moved there, else moved
    /// as a new one is. Not counted in `dma_buf_images`.
    pub dma_buf_images_taken: u64,
    /// Textures in place over a no-copy buffer's dma-buf pages that found no parked image of
    /// their plan at their texels, whose image was made and imported anew (counted in
    /// `dma_buf_images` too): with `dma_buf_images_taken` and `dma_buf_images_in_import`, every
    /// such texture's image.
    pub dma_buf_images_missed: u64,
    /// Parked images released untaken, their `VkImage` destroyed and their import freed: idle
    /// for 2 s (the residency's first idle window, `host_imports_released`), past the most the
    /// device keeps on its list (each parked image one, the least recently parked first, beside
    /// the releasable imports), as `Device::host_range_remapped` names a range of their pages, or
    /// as the import of their pages went.
    pub dma_buf_images_parked_released: u64,
    /// Textures in place over a no-copy buffer's dma-buf pages whose image was bound in the one
    /// import the buffer lies in, at the texels' offset, beside the buffer: a
    /// `DRM_FORMAT_MOD_LINEAR` image whose plane lies at offset 0 with the texels' pitches, made
    /// where the device's check of such images passed (`DmaBufInImport` in its report) and the
    /// import is dedicated to nothing, the image requiring no dedicated allocation and its memory's
    /// alignment dividing the offset. No dma-buf asked of the source and nothing imported: not
    /// counted in `dma_buf_images` or `dma_buf_imports`, and not parked. Never on a driver whose
    /// `DRM_FORMAT_MOD_LINEAR` images for dma-bufs require a dedicated allocation (RADV).
    pub dma_buf_images_in_import: u64,
    /// Images of textures over a buffer of the runtime's memory exported as a dma-buf
    /// (`MemoryExport::DmaBuf`: a shared block of such buffers, or a buffer's dedicated
    /// allocation) where no image binds in that memory itself (RADV takes the dma-buf handle type
    /// for no linear image, and asks a dedicated allocation of a `DRM_FORMAT_MOD_LINEAR` one), each
    /// a `DRM_FORMAT_MOD_LINEAR` image at the texels' offset with Metal's pitches, bound to a
    /// dedicated import of that memory's own dma-buf: the buffer's bytes in place, through a
    /// second `VkDeviceMemory` of the same payload, the import live in
    /// `dma_buf_image_memories_live` until the image goes. One per image made, a shadow's move in
    /// place too; none is a shadow (`shadowed_textures`). A runtime extension: Metal has none.
    pub exported_dma_buf_images: u64,
    /// Such images (`exported_dma_buf_images`) kept as their texture went, with their import,
    /// under their buffer's pages, for the next texture over the same buffer of the same image
    /// plan at the same texels and pitches: one whose `VkImage` was never handed out, within the
    /// bound and windows of `dma_buf_images_parked`, which counts the images over host pages
    /// apart. An image taken and parked again is counted again. A runtime-internal reuse, which
    /// nothing of Metal's observes.
    pub exported_dma_buf_images_parked: u64,
    /// Textures over a buffer of the runtime's exported memory whose image was a parked one
    /// (`exported_dma_buf_images_parked`) taken for them: no `vkGetMemoryFdKHR`, no import and no
    /// `VkImage` made, the image resting in `GENERAL` on the device's queue family where a
    /// submission moved it there, else moved as a new one is. Not counted in
    /// `exported_dma_buf_images`.
    pub exported_dma_buf_images_taken: u64,
    /// Parked images over the runtime's exported memory released untaken, their `VkImage`
    /// destroyed and their import freed: as `dma_buf_images_parked_released` counts those over
    /// host pages (but no remap of host pages names them), and as their buffer went.
    pub exported_dma_buf_images_parked_released: u64,
    /// Textures placed in such a heap that such an image would alias but the device made none
    /// (the source failed, the format, usage, extent or plane layout is not supported, or the
    /// driver refused the import): each is a shadow of the heap's bytes in memory of its own
    /// (`shadowed-texture`).
    pub dma_buf_image_fallbacks: u64,
    /// Formats whose `DRM_FORMAT_MOD_LINEAR` tiling features the driver was asked, for an image
    /// over dma-bufs: each format once at most, as the answer is kept.
    pub modifier_feature_queries: u64,
    /// Pixel formats whose `minimumLinearTextureAlignmentForPixelFormat:` answer the device
    /// probed of its driver, by making the images a texture over host pages takes in place at
    /// each alignment it tries: every format a buffer backs at the first query of any, once for
    /// the device's life, and none where a record of the same driver and runtime held them.
    pub linear_alignments_probed: u64,
    /// Pixel formats whose `minimumLinearTextureAlignmentForPixelFormat:` answer the device read
    /// from the record of an earlier probe of the same driver and runtime, at the first query of
    /// any: it probed nothing.
    pub linear_alignments_recalled: u64,
    /// Textures over a buffer's bytes (a 2D texture or array over a buffer, a no-copy buffer or an
    /// IOSurface plane, or a view of another layout of one) that no image aliases, each an image
    /// of its own that the encoders copy the bytes into and back (`shadowed-texture`).
    pub shadowed_textures: u64,
    /// Of `shadowed_textures`, those whose image started as a new texture of its storage mode
    /// does (zero but in a `Private` one) as it was made: its copies leave an aspect out (a depth
    /// texture over a surface's plane, whose image's stencil reads zero) or is copied in at its
    /// first use alone (a tiled texture's, `Buffer::new_tiled_texture`, whose CPU calls read it
    /// before then). Every other shadow's copy in writes
    /// each texel before any command reads it, so its image starts as its memory lies.
    pub shadows_zeroed: u64,
    /// Of `shadowed_textures`, those over a buffer that lies in fresh pages it imported
    /// (`fresh_page_buffers_made`): an image bound to imported memory starts `UNDEFINED`, and its
    /// first move may write any byte bound to it.
    pub shadows_over_fresh_pages: u64,
    /// Textures over a buffer's bytes whose rows are in place (`TextureFacts::texel_rows`), made
    /// with their shadow left unmade: a slot that only reads, writes or asks the size of one binds
    /// its rows, and its shadow is made at the first reach of its image
    /// (`shadows_made_at_use`, `texture/lazy_shadow.rs`). One only ever bound through its rows
    /// makes none, nor counts in `shadowed_textures`.
    pub shadows_left_unmade: u64,
    /// Of `shadows_left_unmade`, those whose shadow a reach made since: a slot that binds the
    /// image, a residency (`use_resource`), a pass's attachment, a view, a handed-out `VkImage`,
    /// each counted in `shadowed_textures` as it is made.
    pub shadows_made_at_use: u64,
    /// Textures over host pages (a no-copy buffer's, a no-copy heap's buffer's, or a buffer in
    /// fresh pages it imported; or a copied host buffer's copy of them, `copied_textures_in_place`)
    /// that sample, read, render or are written through a shader, made an image aliasing the pages
    /// in place, where the device's check passed for their class and
    /// each of their accesses (`HostPageTexturePolicy::InPlace`): no shadow, and no copy in or
    /// back. One that renders whose image took feedback loop usage is also counted in
    /// `feedback_loop_alias_images`.
    pub host_textures_in_place: u64,
    /// Of `host_textures_in_place`, those whose usage writes them through a shader
    /// (`ShaderWrite` or `ShaderAtomic`): their kernels' writes reach the pages in place.
    pub host_textures_in_place_written: u64,
    /// `Managed` textures over a buffer's host pages (not an IOSurface's) that only sample or
    /// read, which `host_textures_in_place` would count (or a shadow made before the check
    /// answered or where it failed), made instead a copy in the device's memory on a discrete
    /// GPU, where their memory carries a log of its writes (`HostMemory::with_dirty_pages`): the
    /// GPU's copy of the two a `Managed` resource is on a Mac with a discrete GPU, a shadow whose
    /// copies in take only the rows the log reports changed (`dirty_copies_in_gated` and kin),
    /// announced nowhere. A `Shared` texture and an IOSurface's are in place over the pages.
    pub device_copy_textures: u64,
    /// `Managed` no-copy buffers on a device without unified memory that are two copies now, the
    /// caller's pages and the GPU's own in device-local memory (`managed_copy.rs`), as Metal keeps
    /// a managed resource on a Mac with a discrete GPU.
    pub managed_device_copies: u64,
    /// Such buffers made, in all.
    pub managed_device_copies_made: u64,
    /// The bytes of device-local memory their device copies hold now.
    pub managed_device_copy_bytes: u64,
    /// `Managed` no-copy buffers on a device without unified memory that lie in their pages
    /// instead, as `Shared` ones do: no device-local memory had room for the device copy
    /// (announced as `device-memory-overflow`).
    pub managed_device_copies_in_place: u64,
    /// Two-copy buffers' device copies made with bytes their caller promised read zero
    /// (`HostMemory::with_zero_from`), which the device set to zero in its own memory rather than
    /// copy in from the pages.
    pub managed_device_copies_zeroed: u64,
    /// The bytes those device copies took as zeros the device wrote, none of them copied in.
    pub managed_bytes_zeroed: u64,
    /// The bytes copied from their pages into their device copies: as each is made, by each
    /// `did_modify_range`, rebind, and copy in again after a write the pages drop.
    pub managed_bytes_copied_in: u64,
    /// The bytes copied out of their device copies into their pages, by `synchronize_resource`
    /// and `synchronize_texture` (counted as the command's copy is decided for an import of the
    /// pages, at the command buffer's completion for pages no import maps).
    pub managed_bytes_copied_out: u64,
    /// The program's `did_modify_range` calls of a nonempty range on such buffers: each copies
    /// its range in (`managed_bytes_copied_in` counts the bytes copied).
    pub managed_ranges_modified: u64,
    /// The bytes of those ranges.
    pub managed_bytes_modified: u64,
    /// The program's `synchronize_resource` commands of such a buffer, or of a texture over one,
    /// and its `synchronize_texture` commands of such a texture, of a nonempty range: the
    /// buffer's length, or the texture's span in it, of which each copies out the bytes the GPU
    /// wrote since the buffer's last copy out (`managed_sync.rs`; `managed_bytes_copied_out`
    /// counts the bytes copied).
    pub managed_resources_synchronized: u64,
    /// The bytes those commands asked.
    pub managed_bytes_synchronized: u64,
    /// Of those commands, those that copied nothing out: no write of the GPU's since the buffer's
    /// last copy out reached their range (`managed_sync.rs`).
    pub managed_synchronizes_unwritten: u64,
    /// Of those commands, those that copied their range whole as the GPU's writes to the buffer
    /// since its last copy out are not known: an untracked buffer, an exported one or one whose
    /// `VkBuffer` was handed out, the buffer's first synchronize, a command before it in its
    /// command buffer that may write memory no use names or whose recording finds its uses, a
    /// command buffer that may write memory no use names submitted since, or a command inside a
    /// predicated loop's laying.
    pub managed_synchronizes_whole: u64,
    /// Of those commands, those whose batch's submission copied, before their part, writes of
    /// other command buffers submitted between their encoding and that submission: one of a
    /// command buffer committed before theirs and encoded after them (`managed_sync.rs`).
    pub managed_synchronizes_topped_up: u64,
    /// The bytes those commands asked that they did not copy out, as no write of the GPU's since
    /// the buffer's last copy out reached them: `managed_bytes_synchronized` less the bytes copied
    /// out of them.
    pub managed_bytes_unwritten_skipped: u64,
    /// The bytes of the staging buffers of their copies held now: those of copies into or out of
    /// pages no import maps, each held until the submission or command buffer that uses it
    /// completed.
    pub managed_staging_bytes: u64,
    /// Command buffers that used such a buffer whose pages drop the GPU's writes (a read-only
    /// rebind) and completed, its pages then copied in whole again.
    pub managed_writes_undone: u64,
    /// The bytes the CPU copied from `Managed` buffers' host copies into staging buffers for
    /// their copies in, on a device without unified memory: such a buffer's pages no import maps
    /// (but those a guest RAM's chunk imports hold, `managed_bytes_copied_in_from_ram`), and
    /// under `ManagedCopyInPolicy::AtCommit` every range a commit takes, of the runtime's own
    /// `Managed` buffers too. Every other copy in is the GPU's, from the host copy itself.
    pub managed_bytes_staged_in: u64,
    /// The bytes the GPU copied into such buffers' device copies from a guest RAM's chunk imports
    /// that hold their pages, which no import of their own maps (`HostRam`): no CPU copy and no
    /// staging. Of `managed_bytes_copied_in`.
    pub managed_bytes_copied_in_from_ram: u64,
    /// The ranges `did_modify_range` named of `Managed` buffers with a host copy (the runtime's
    /// own, and those over the caller's pages) written into their device copies inline: at most
    /// 16 KiB of whole words each, read from the host copy as the submission that copies them in
    /// is recorded and written by `vkCmdUpdateBuffer`s in its command stream, with no copy from
    /// the host copy (`managed.rs`). Counted as each is recorded.
    pub managed_ranges_inlined: u64,
    /// The batches whose first work was uploads alone into such buffers' device copies (copies
    /// or inline writes) and was recorded with no barrier of its own from the commands before
    /// it, nor one before the batch's GPU start: on a device that orders its batches by their
    /// ends, where the end of the batch before it on its `VkQueue` already orders every command
    /// before it ahead of them and no internal submission of the device's may still be running
    /// (`texture/first_move.rs`). Counted as each is recorded.
    pub managed_uploads_unbarriered: u64,
    /// The batches whose first work was uploads alone into such buffers' device copies and was
    /// made visible by a barrier to the stages and access types the batch's own commands' uses
    /// of those buffers name, not to every later command: on a device that orders its batches by
    /// their ends, whose end orders every later batch, where the batch's uses of each buffer are
    /// known (`texture/first_move.rs`). Counted as each is recorded.
    pub managed_upload_barriers_narrowed: u64,
    /// Host copies of such buffers made (with the buffer, at a rebind, at a recovery) that are
    /// their pages alone, no import of their own mapping them, because the caller's gate refused
    /// their import, the device refused their pointer and no dma-buf of them served, or the import
    /// failed.
    pub managed_pages_alone_refused: u64,
    /// Such host copies made their pages alone because a live import of another object's pages,
    /// or of a guest RAM's chunk, holds some of them (their bytes across an import's edge, or a
    /// chunk's end): no page is imported twice.
    pub managed_pages_alone_over_imports: u64,
    /// Such host copies made their pages alone because the process cannot write them, or Metal's
    /// private option bit 24 drops the GPU's writes and the system maps no read-only view.
    pub managed_pages_alone_unwritable: u64,
    /// Of the host copies counted by `managed_pages_alone_*`, those whose pages a guest RAM's chunk
    /// imports hold all of: their copies in are the GPU's, from those imports
    /// (`managed_bytes_copied_in_from_ram`), never staged by the CPU, but under
    /// `ManagedCopyInPolicy::AtCommit`.
    pub managed_pages_in_ram_chunks: u64,
    /// Of `host_textures_in_place`, those over pages imported as dma-bufs: each a
    /// `DRM_FORMAT_MOD_LINEAR` image over a dma-buf of its own pages (also counted in
    /// `dma_buf_images`, or taken parked, `dma_buf_images_taken`), or bound in the import its
    /// buffer lies in (`dma_buf_images_in_import`).
    pub dma_buf_textures_in_place: u64,
    /// Of `host_textures_in_place`, those over a buffer in fresh pages it imported.
    pub fresh_page_textures_in_place: u64,
    /// Of `host_textures_in_place`, those over a copied host buffer (`copied.rs`): an image over
    /// the copy of the caller's pages the GPU uses in their stead, bound in the copy's memory where
    /// the device's check passed for the caller's pages and, for a copy in fresh pages the runtime
    /// imported, for those (for a `DRM_FORMAT_MOD_LINEAR` image over a copy in memory the device
    /// allocates, for its pooled memory), for the texture's accesses. Counted in neither
    /// `fresh_page_textures_in_place` nor `pitched_host_images`.
    pub copied_textures_in_place: u64,
    /// Of `host_textures_in_place`, those whose rows lie at Metal's pitch in a
    /// `DRM_FORMAT_MOD_LINEAR` image made for host allocations, where the driver's own linear
    /// image lays them out otherwise.
    pub pitched_host_images: u64,
    /// Textures over host pages made shadows because the device's policy asks it
    /// (`HostPageTexturePolicy::Shadow`).
    pub host_texture_shadows_exact: u64,
    /// Of the textures over host pages made shadows, counted by why in the counts beside this
    /// one, those whose usage renders into them (`RenderTarget`, whatever else it holds; an
    /// `Unknown` usage among them): the render targets a pass renders through a shadow, its
    /// copies in and merges back.
    pub host_texture_shadows_render_target: u64,
    /// Of the textures over host pages made shadows, counted by why in the counts beside this
    /// one, those whose usage writes them through a shader (`ShaderWrite` or `ShaderAtomic`,
    /// whatever else it holds; an `Unknown` usage among them).
    pub host_texture_shadows_shader_write: u64,
    /// Textures over host pages made shadows because their usage writes them through a shader
    /// and their texels are of a width the device's check of storage writes stores none of (it
    /// stores those of 1, 2, 4, 8 and 16 bytes, every width a buffer backs).
    pub host_texture_shadows_store_width: u64,
    /// Textures over host pages made shadows because the device's check of their class and an
    /// access of theirs (one that reads, renders or a shader writes) had not answered yet (a
    /// first session's, before the verdicts are kept): each moves in place once they pass
    /// (`host_textures_moved_in_place`).
    pub host_texture_shadows_check_pending: u64,
    /// Textures over host pages made shadows because the device's check of their class and
    /// access found a byte changed.
    pub host_texture_shadows_check_failed: u64,
    /// Textures over host pages made shadows because the device's check made no image of their
    /// class for their access.
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
    /// Classes of host pages (pointer, dma-buf, fresh), each with each access (an image that
    /// reads, one that renders, one a shader writes) and each row ends (texels that share a line
    /// of memory with bytes beside them, and texels that share none), the device's check ran for
    /// in this process: `in_place_checks_passed`, `in_place_checks_failed` and
    /// `in_place_checks_unchecked` together.
    pub in_place_checks_run: u64,
    /// Of `in_place_checks_run`, those whose every image left every byte alone: its first move,
    /// for one that renders its pass's load and store and its feedback loop moves, and for one a
    /// shader writes a kernel's storage writes (each byte but the texels it wrote).
    pub in_place_checks_passed: u64,
    /// Of `in_place_checks_run`, those that found a byte changed (or were made to fail).
    pub in_place_checks_failed: u64,
    /// Of `in_place_checks_run`, those that made no image of the class for the access.
    pub in_place_checks_unchecked: u64,
    /// Classes, accesses and row ends whose verdict was read from the record of an earlier check
    /// of the same driver, device and check: no check ran for them. Measured verdicts (passed or
    /// failed) are recalled, and those unchecked for a cause that lasts for the driver (the device
    /// lacks an extension, a feature, an import, a format or an image of the shape); one the
    /// earlier check could not give for a cause that may pass (pages or memory not made, an import
    /// refused, a move or the stores that failed) is checked again.
    pub in_place_checks_recalled: u64,
    /// Textures over host pages made shadows before their pages' in-place verdicts (counted in
    /// `host_texture_shadows_check_pending`) that moved in place once the verdicts passed, at
    /// the commit of the device's only uncommitted command buffer (`texture/storage_form.rs`):
    /// each is counted in `host_textures_in_place` too from then on.
    pub host_textures_moved_in_place: u64,
    /// Shadow images of textures that moved in place (`host_textures_moved_in_place`) let go of
    /// once the command buffer that took the move completed, its image, views and memory
    /// freed (`texture/storage_form.rs`). A texture with a view made before its move keeps its
    /// shadow until the view and the texture went, and an untracked one until it went, neither
    /// counted here.
    pub host_texture_shadows_freed_after_move: u64,
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
    /// `vkCmdFillBuffer`, on every device, in place of copies of the device's host-visible zeros
    /// (`texture/first_move.rs`); fills apart from `fills_recorded`.
    pub new_buffer_zero_fills: u64,
    /// Copies between a shadowed texture's truth and its shadow that encoders planned
    /// (`shadow.rs`): copies in before a command, copies back after one or as the encoder ends,
    /// and the keeps and merges of a command that writes the bytes two ways or writes through a
    /// texture that merges back (`shadow_merge.rs`). Each is a command of its encoder over the
    /// texture's texels.
    pub shadow_copies: u64,
    /// Of the copies in of `shadow_copies`, command buffers' first ones of a shadow whose truth
    /// only the runtime's commands write (a tracked `Private` buffer's bytes the device made, an
    /// IOSurface plane's among them), decided as their batches were submitted against the
    /// image's standing across command buffers (`shadow_standing.rs`).
    pub shadow_copies_in_against_standing: u64,
    /// Of those, the ones that moved nothing: the image held every texel of the truth, as an
    /// earlier command buffer's end left it, and nothing submitted since wrote the truth or the
    /// image.
    pub shadow_copies_in_skipped_by_standing: u64,
    /// The texels those would have moved; they are among `shadow_texels_copied_in` and its
    /// split, which count every copy in as it is planned.
    pub shadow_texels_skipped_by_standing: u64,
    /// The texels of a command buffer's first copies in of a shadow whose truth only the runtime's
    /// commands write (`shadow_texels_first_copied_in_private_buffers`,
    /// `shadow_texels_first_copied_in_private_surfaces` and
    /// `shadow_texels_first_copied_in_private_images`) that moved them, by where each was
    /// decided (`shadow_standing.rs`): with `shadow_texels_skipped_by_standing`, the seven sum to
    /// those first copies in. This one: of a shadow that never stands (its buffer untracked, a
    /// heap's or over the caller's memory, its `VkBuffer` or its image's `VkImage` handed out, or
    /// its writes dropped).
    pub shadow_texels_unskipped_unstanding: u64,
    /// Recorded between two render pass instances of a pass its encoder holds, which no batch's
    /// submission decides.
    pub shadow_texels_unskipped_between_instances: u64,
    /// After a command of its encoder wrote what it reads.
    pub shadow_texels_unskipped_after_encoder_writes: u64,
    /// Recorded whole as its encoder recorded it: inside a predicated loop's laying, or where no
    /// new `VkCommandBuffer` could be made.
    pub shadow_texels_unskipped_recorded_whole: u64,
    /// An earlier encoder of its command buffer wrote the truth or the image, found as its encoder
    /// recorded it (or, encoded again, at its batch's submission); or, at its batch's submission,
    /// its command buffer may reach memory no use names.
    pub shadow_texels_unskipped_command_buffer: u64,
    /// At its batch's submission, where its standing was let go of.
    pub shadow_texels_unskipped_let_go: u64,
    /// At its batch's submission, where its image stood holding every texel of the truth but
    /// those inside a rectangle that writes within it changed since its copy in (a render pass's
    /// draws over a color attachment it loads, `shadow::Within`): the texels inside it alone.
    pub shadow_texels_unskipped_stale_within: u64,
    /// The same texels but `shadow_texels_unskipped_unstanding`'s, by what the shadow's standing
    /// held as each copy was decided (as its encoder recorded it, by the batches submitted
    /// then, for the first three of where): the five sum to them. This one: it stood, so where
    /// it was decided alone moved them, or it stood stale inside a rectangle alone, whose texels
    /// moved (`shadow_texels_unskipped_stale_within`).
    pub shadow_texels_unskipped_stood: u64,
    /// No command buffer's end left its image standing since it was made.
    pub shadow_texels_unskipped_never_stood: u64,
    /// Of `shadow_texels_unskipped_never_stood`, by what last kept a command buffer's end from
    /// leaving the image standing (`shadow_standing::Missed`; the five sum to them). This one: no
    /// command buffer's end met it, none ending with it among what its encoders carried.
    pub shadow_texels_never_stood_unmet: u64,
    /// An encoder that used it ended with its image not holding the truth: a command wrote the
    /// truth another way after its copy in (a pass rendering the texture's image in place, which
    /// an attached read's shadow copies), another shadow's copy back or merge wrote the bytes, or
    /// nothing copied it in.
    pub shadow_texels_never_stood_not_current: u64,
    /// What the command buffer carried to its later encoders let it go: a later command wrote its
    /// truth another way, another shadow's copy back or merge wrote the bytes, work recorded
    /// apart stood between, or an encoder started after a wait, a signal or a host step.
    pub shadow_texels_never_stood_dropped: u64,
    /// The command buffer's end carried its image holding part of the truth.
    pub shadow_texels_never_stood_partial: u64,
    /// The command buffer's end records no standing: a failure, a rebind, memory no use names,
    /// or a last batch without its last encoder.
    pub shadow_texels_never_stood_end: u64,
    /// A batch that may reach memory no use names was submitted since it stood.
    pub shadow_texels_unskipped_unnamed_since: u64,
    /// A batch submitted since it stood wrote its truth: another shadow's copy back or merge over
    /// the same bytes, a blit's, a pass's or a kernel's write.
    pub shadow_texels_unskipped_truth_written: u64,
    /// A batch submitted since it stood wrote its image.
    pub shadow_texels_unskipped_image_written: u64,
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
    /// Of `shadow_texels_first_copied_in_surfaces`, those of a plane in a buffer of private or
    /// memoryless storage the device made, which the CPU never writes; the rest are a plane's in
    /// host pages or in a buffer the CPU maps.
    pub shadow_texels_first_copied_in_private_surfaces: u64,
    /// Of `shadow_texels_first_copied_in_images`, those of a tracked image of private or
    /// memoryless storage the device made in memory of its own, which only its commands write:
    /// a shadow of one may stand across command buffers (`shadow_standing.rs`).
    pub shadow_texels_first_copied_in_private_images: u64,
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
    /// write made since the commit, so it requires these where the CPU may write the plane;
    /// where only the runtime's commands write it, those the image's standing across command
    /// buffers took off as their batches were submitted are counted here too, and apart in
    /// `shadow_texels_skipped_by_standing`.
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
    /// Of those copies back, the ones into a no-copy buffer's host pages recorded spliced, on a
    /// device whose check found its copies or shader writes of texels writing back a line of
    /// memory they end in whole, or has not answered (`shadow_splice.rs`): the GPU copies the
    /// texels that fill whole lines, and a host step after them copies each row's texels in the
    /// lines it shares with bytes beside them, a byte at a time, as the CPU's stores go, so the
    /// CPU's writes to those bytes stay.
    pub shadow_copies_back_spliced: u64,
    /// The merges of shadowed textures over a no-copy buffer's host pages made on the host, on a
    /// device whose check found its copies or shader writes of texels writing back a line of
    /// memory they end in whole, or has not answered (`shadow_splice.rs`): the GPU copies the
    /// texels the commands left into a `Shared` scratch beside those kept before them, and a host
    /// step after it stores each byte that differs of each row's bytes in the lines that also hold
    /// bytes beside the texels (every byte, for a merge not cut to its texels' rows), so the CPU's
    /// writes to those other bytes stay; the merge's kernel stores the rest. A merge none of whose
    /// rows' lines holds a byte beside its texels goes by its kernel alone, and is not counted.
    pub shadow_merges_on_host: u64,
    /// The nanoseconds of the host steps that made those merges, on the thread that ran them.
    pub shadow_merge_host_nanos: u64,
    /// The bytes those host steps compared.
    pub shadow_merge_bytes_on_host: u64,
    /// The host steps that ran for encoders' copies back spliced and merges on the host
    /// (`shadow_splice.rs`): one per encoder end that left one, each a wait inside its command
    /// buffer for the GPU's work before it, the command buffer's later work going in a batch of
    /// its own. Against `command_buffers_submitted`, the host steps a command buffer pays.
    pub shadow_host_steps: u64,
    /// Of the copies in among `shadow_copies`, those recorded with their texture's keep after
    /// them (one of `shadow_merge_copies_dispatched`) as one dispatch that stores the texels in
    /// the shadow and in the merge's scratch, where the device runs them faster so
    /// (`DriverFact::ShadowCopyInDispatchFaster`): the same bytes either way.
    pub shadow_copies_in_dispatched: u64,
    /// Of the keeps of merged shadowed textures, those of a heap's tiled texture recorded right
    /// after its copy in as one copy of the staging buffer that copy filled with the heap's
    /// bytes, in place of a move of the shadow's texels back into Metal's tiles: the same bytes
    /// either way, but a depth's, whose keep holds the heap's bytes where its image's may differ
    /// (`shadow.rs`).
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
    /// log and commit, and one per log and commit for copied host buffers over logged pages.
    pub dirty_stamps_asked: u64,
    /// The waits the submissions (and, for copied host buffers, the commits) made for a log's
    /// publication (`DirtyPages::wait`): one per log serves every stamp asked of it before the
    /// wait.
    pub dirty_waits: u64,
    /// Of the waits, those the log answered `false`.
    pub dirty_waits_unanswered: u64,
    /// The nanoseconds the waits took, on the threads that submit and commit command buffers.
    pub dirty_wait_nanos: u64,
    /// The holds of shadows over logged pages a commit left untaken, a command buffer committed
    /// before it that waits for an event or a semaphore having written the shadow's truth and not
    /// yet completed: it may run after the commit's work (`dirty_pages.rs`, **A waiting
    /// writer**).
    pub dirty_holds_withheld: u64,
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
    /// The merges of shadowed textures whose scratch was a buffer an earlier encoder of their
    /// command buffer was done with, or one their shadow kept from an earlier merge whose
    /// command buffers completed (`shadow_merge.rs`, `merge_scratch.rs`): no buffer made.
    pub shadow_merge_scratches_reused: u64,
    /// The merges of shadowed textures whose scratch holds zero before each keep, as their kernel
    /// reads bytes no copy writes (those between the texture's rows): a merge not cut to rows of
    /// whole words, through a copy or of rows that start or end inside a word, its scratch filled
    /// with zero (or made so, past the device's largest buffer). A cut merge's copies write
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
    /// attributes the build to that call: encoding a command where the command is, a resource's
    /// own moves included at its first use (`owned_kernel::FirstUse`), and a commit or an
    /// internal submission where a decoded twin's decode staged or queued since the last needs
    /// it (`texture/decoded_twin.rs`: a heap texture's whole decode, a view's second twin, a
    /// host call's staged copy's), before the submission's lock. No making builds one. A later
    /// call that takes a kernel already built adds
    /// nothing, so a reading of 0 beside a kernel's dispatch says the build landed in an earlier
    /// call. Metal builds none of these: they are what the emulation costs, and the count is
    /// where that cost fell. A render execution's draw replay's are begun ahead, on the device's
    /// pool, by the creation of a render state that supports indirect command buffers
    /// (`Device::begin_draw_replay_kernels`), and counted where the pool builds them.
    pub owned_kernels_made: u64,
    /// Of those, the builds a host step made itself (`exit_guard::in_host_step`), each a compute
    /// pipeline's creation inside the step's time
    /// ([`host_step_nanos`](Self::host_step_nanos)), apart from the states' variants'
    /// ([`host_step_pipelines_made`](Self::host_step_pipelines_made)).
    pub host_step_owned_kernels_made: u64,
    /// The SPIR-V those builds compiled, in bytes: what the pipeline creations were handed, which
    /// is the measure a build's cost scales with. It counts no scratch an emulation's work
    /// allocates besides (a YCbCr conversion's staging buffer, whose bytes are in
    /// `device_memory_allocated`).
    pub owned_kernel_spirv_bytes: u64,
    /// The scratch buffers a YCbCr texture's conversion took (`ycbcr.rs`), counted as each was
    /// allocated: one for the conversion that reads the texture's planes, made at its first
    /// conversion (none for a texture whose planes every function reads in place, which never
    /// converts), and one more for the write-back where the texture takes writes. **At most one
    /// of each per texture**, not per device, and
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
    /// `A8Unorm` textures over a buffer's bytes that only read, made as an `R8Unorm` texture's
    /// storage over them (in place where that texture is) and read through
    /// `(Zero, Zero, Zero, Red)`, where the device's linear `A8_UNORM` image lacks what its
    /// optimal one does and its linear `R8_UNORM` image lacks nothing (NVIDIA's linear
    /// `A8_UNORM` samples, filters and copies nothing): `texture/red_held_alpha.rs`.
    pub alpha_textures_held_in_red: u64,
    /// Shadow images of texture views made (`texture/view_shadow.rs`, `shadowed-texture-view`):
    /// a view's own, or the render shadow of one of `views_through_swizzle`, made at the first
    /// pass that renders into it.
    pub view_shadows: u64,
    /// Draws that sample, at a slot, a texture that presents a color attachment of their own
    /// render pass, each reading the texture's copy, taken before its render pass instance, in
    /// the attachment's place where the texture's shadow is served (`texture/attached_reads.rs`;
    /// the copies count in `shadow_copies`), or at a slot narrowed to the levels apart from the
    /// attached ones, the texture's own image (`attached_read_slots_narrowed`,
    /// `attached_read_slots_past`). A draw again or
    /// rebound of one counts as it did (`render/draw_again.rs`).
    pub attachment_reads_through_copies: u64,
    /// Draws that sample a color attachment's texture after a memory barrier of their render
    /// pass, each recorded into a new instance (a barrier inside an instance orders one pixel
    /// alone), whose copy holds what the draws before the barrier wrote; and those that sample a
    /// color the pass clears after a barrier before its first draw, each recorded into an
    /// instance after the pass's first, which runs the clear, whose copy holds it
    /// (`texture/attached_reads.rs`). Those of the second kind that `attachment_read_clears`
    /// counts split nothing.
    pub attachment_read_splits: u64,
    /// Draws that sample a color attachment's texture after a memory barrier inside their render
    /// pass instance, each staying in the instance in place of a split, as nothing the instance
    /// drew meets the texels it reads: the copy taken before the instance holds every one of them
    /// as a split's would (`texture/attached_reads.rs`). Not in `attachment_read_splits`.
    pub attachment_read_splits_spared: u64,
    /// Draws that sample a color their render pass clears after a memory barrier before its
    /// first draw, before any draw of the pass, that read the color's copy in the pass's first
    /// instance, the copy cleared to the clear before the pass, instead of splitting it: one per
    /// draw that brought new such copies, each a whole one-level, one-slice texture the pass
    /// clears in place (`texture/attached_reads.rs`). Not in `attachment_read_splits`, and the
    /// clears are not in `shadow_copies`.
    pub attachment_read_clears: u64,
    /// Barriers recorded before a draw on a device with neither rasterization-order color
    /// attachment access nor `fragmentShaderPixelInterlock`, whose framebuffer fetch reads input
    /// attachments ordered between draws alone: before a draw that reads them after an earlier
    /// draw of its render pass instance, or its load operations, wrote a color, or that writes a
    /// color after an earlier one read them, each the draws' subpass's whole dependency on itself
    /// (`framebuffer_fetch.rs`). 0 on every other device.
    pub fetch_draw_barriers: u64,
    /// Draws that sample a color attachment's texture of their render pass instance at their own
    /// pixel, each reading it through a snapshot inside its instance in place of a split, after
    /// a memory barrier of their render pass, or of the copy taken before the pass, with none:
    /// drawn twice, first copying each fragment's texel into the snapshot, then reading the
    /// snapshot (`render/snapshot.rs`). They count in neither `attachment_reads_through_copies`
    /// nor `attachment_read_splits`. Without a memory barrier the draw reads what the draws
    /// before it in the instance wrote, an x86 Mac's answer, where an M3 Max reads the texels
    /// from before the pass (`docs/API.md`'s accepted differences).
    pub reading_draws_snapshotted: u64,
    /// Of `reading_draws_snapshotted`, the draws after a memory barrier whose snapshot draws ran
    /// in a render pass instance of their own that leads the draw's, as a color they read has no
    /// feedback loop usage (a linear image on NVIDIA's drivers): the draw's instance after a
    /// split of the pass, with no copy (`render/snapshot.rs`, leading snapshots). Each such draw
    /// begins that one render pass instance more, among no `render_passes_begun_*`.
    pub reading_draws_snapshotted_leading: u64,
    /// Draws that sample a color attachment's texture of their render pass instance at their
    /// own pixel alone and write no channel of that color, each reading the attachment in place,
    /// in the feedback loop layout, with a barrier before and after it, with a memory barrier of
    /// their render pass before them or not: no copy, snapshot or split (`render/snapshot.rs`,
    /// in place). They count in neither `attachment_reads_through_copies` nor
    /// `attachment_read_splits`. Without a memory barrier the draw reads what the draws before it
    /// in the instance wrote, an x86 Mac's answer, where an M3 Max reads the texels from before
    /// the pass (`docs/API.md`'s accepted differences).
    pub reading_draws_in_place: u64,
    /// Of `reading_draws_snapshotted`, the draws recorded as a kept draw (`render/draw_again.rs`):
    /// drawn again, their two draws and their barriers alone, or rebound, a kept draw carried
    /// into its encoder's (`render/carried.rs`) among them, each weighing its form again as every
    /// step weighs it.
    pub reading_draws_snapshotted_again: u64,
    /// Draws of a kept draw that reads through snapshots that took every step: the draw would
    /// read through snapshots otherwise, or not (another slot, color, plan or form, or an instance
    /// that cannot hold its colors in the feedback loop layout).
    pub reading_draws_snapshotted_stopped: u64,
    /// Of `reading_draws_snapshotted` and `reading_draws_in_place`, the draws recorded into a new
    /// instance because their own was recorded inline, its render pass instance begun without
    /// the feedback loop layout they need (`render/snapshot.rs`, `render/inline_first.rs`).
    pub snapshot_inline_splits: u64,
    /// Draws that read a texture of their render pass after a memory barrier at their own pixels
    /// and took the split (`attachment_read_splits`), or were spared it where their instance drew
    /// none of what they read (`attachment_read_splits_spared`), by why the snapshot did not
    /// serve them (`render/snapshot.rs`, `SplitReason`; the log names each reason once a device).
    /// This one: the function's read reaches what no snapshot serves
    /// (`StageModule::snapshot_textures`: a branch, a sample, a derivative, a store, or the
    /// function writes memory), and the fields below it say which
    /// (`StageModule::snapshot_refusals`).
    pub reading_draws_split_unreadable_own_pixel: u64,
    /// Of `reading_draws_split_unreadable_own_pixel`, those of a slot whose read decides a branch
    /// or a switch, beside a selection a helper invocation may take either arm of
    /// (`SnapshotRefusal::Branch`). A draw is counted in each of the clauses below that leave out
    /// any of its slots no snapshot serves.
    pub reading_draws_split_unreadable_own_pixel_branch: u64,
    /// Of `reading_draws_split_unreadable_own_pixel`, those of a slot whose read reaches a
    /// derivative (`SnapshotRefusal::Derivative`).
    pub reading_draws_split_unreadable_own_pixel_derivative: u64,
    /// Of `reading_draws_split_unreadable_own_pixel`, those of a slot whose read reaches a sample,
    /// a gather or a level-of-detail query (`SnapshotRefusal::Sample`).
    pub reading_draws_split_unreadable_own_pixel_sample: u64,
    /// Of `reading_draws_split_unreadable_own_pixel`, those of a slot whose read reaches a
    /// subgroup or quad operation (`SnapshotRefusal::GroupOperation`).
    pub reading_draws_split_unreadable_own_pixel_group_operation: u64,
    /// Of `reading_draws_split_unreadable_own_pixel`, those of a slot whose read is stored outside
    /// the invocation's `Function`, `Private` and `Output` variables (`SnapshotRefusal::Store`).
    pub reading_draws_split_unreadable_own_pixel_store: u64,
    /// Of `reading_draws_split_unreadable_own_pixel`, those of a slot whose read reaches an atomic
    /// (`SnapshotRefusal::Atomic`).
    pub reading_draws_split_unreadable_own_pixel_atomic: u64,
    /// Of `reading_draws_split_unreadable_own_pixel`, those of a slot whose read reaches an image
    /// write or a texel pointer (`SnapshotRefusal::ImageWrite`).
    pub reading_draws_split_unreadable_own_pixel_image_write: u64,
    /// Of `reading_draws_split_unreadable_own_pixel`, those of a slot whose read is an integer
    /// divisor, a signed dividend beside a divisor that may be `-1`, or a dynamic vector index
    /// (`SnapshotRefusal::UndefinedOperand`).
    pub reading_draws_split_unreadable_own_pixel_undefined_operand: u64,
    /// Of `reading_draws_split_unreadable_own_pixel`, those of a function that writes memory
    /// (`SnapshotRefusal::WritesMemory`).
    pub reading_draws_split_unreadable_own_pixel_writes_memory: u64,
    /// Of `reading_draws_split_unreadable_own_pixel`, those of a function that holds an interlock
    /// (`SnapshotRefusal::Interlock`).
    pub reading_draws_split_unreadable_own_pixel_interlock: u64,
    /// Of `reading_draws_split_unreadable_own_pixel`, those of a function with a coherent resource
    /// or under the Vulkan memory model (`SnapshotRefusal::Coherent`).
    pub reading_draws_split_unreadable_own_pixel_coherent: u64,
    /// Of `reading_draws_split_unreadable_own_pixel`, those left out for anything else
    /// (`SnapshotRefusal::Other`, and a refusal this runtime does not name): a read that reaches a
    /// ray query or a call the analysis does not follow, a slot past 127, or values that do not
    /// settle.
    pub reading_draws_split_unreadable_own_pixel_other: u64,
    /// The texture is not read where its attachment renders it in place, or its image lacks the
    /// feedback loop usage (`Texture::snapshot_color`).
    pub reading_draws_split_not_in_place: u64,
    /// The device lacks `attachmentFeedbackLoopLayout`, `extendedDynamicState3ColorWriteMask` or
    /// `fragmentShaderPixelInterlock`.
    pub reading_draws_split_device_lacks: u64,
    /// A visibility query is active.
    pub reading_draws_split_visibility_query: u64,
    /// The pass fetches input attachments, or has a rasterization-rate map, samples or layers.
    pub reading_draws_split_pass_shape: u64,
    /// The draw renders the attachment through a view of another format.
    pub reading_draws_split_other_format: u64,
    /// A framebuffer fetch of the attachment would render it through a stand-in, and the draw
    /// cannot pin it.
    pub reading_draws_split_stand_in: u64,
    /// The draw reads two attachments of one image.
    pub reading_draws_split_one_image: u64,
    /// The area rule, weighed at the pixels the reader covered when last measured
    /// (`render/coverage.rs`): the snapshot costs more than the split's copy.
    pub reading_draws_split_coverage: u64,
    /// The area rule, weighed at the draw's scissors: the snapshot costs more than the split's
    /// copy.
    pub reading_draws_split_area: u64,
    /// The compile or bind layer refused the variant that reads through snapshots.
    pub reading_draws_split_variant_refused: u64,
    /// The binder left a slot out of the variant.
    pub reading_draws_split_slot_left_out: u64,
    /// The draw renders views.
    pub reading_draws_split_views: u64,
    /// The draw binds the texture as a storage image.
    pub reading_draws_split_storage_binding: u64,
    /// The draw reads another attachment's texture.
    pub reading_draws_split_other_attachment: u64,
    /// Draws that read a color of their render pass after a memory barrier, which the area rule
    /// weighs between the snapshot and the split, weighed by the pixels they covered in the
    /// latest pass that measured them rather than by their scissors (`render/coverage.rs`):
    /// among `reading_draws_snapshotted` or `attachment_read_splits`, as they took.
    pub reading_draws_by_coverage: u64,
    /// Of `reading_draws_by_coverage`, the draws whose snapshots would lead their instance (as
    /// `reading_draws_snapshotted_leading` counts) and whose pixels covered in the latest pass
    /// that measured them were within the area rule's bound: their snapshot draws cost no more
    /// than the split's copy (`render/snapshot.rs`, leading snapshots), so they led it. With the
    /// next two, where such readers fall: a 4-byte color's reader leads covering up to a fifth
    /// of its pixels.
    pub reading_draws_leading_covered_within: u64,
    /// Of the same draws, those that covered more than the area rule's bound and at most half of
    /// their colors' pixels: they split the pass.
    pub reading_draws_leading_covered_to_half: u64,
    /// Of the same draws, those that covered more than half of their colors' pixels: they split
    /// the pass.
    pub reading_draws_leading_covered_over_half: u64,
    /// Occlusion queries such draws recorded their drawing commands inside, whose counts their
    /// textures keep as the command buffers complete (`render/coverage.rs`): one a draw, on a
    /// device with `occlusionQueryPrecise` that resets queries on the host.
    pub coverage_queries: u64,
    /// Occlusion query pools those queries' passes made, where the device had no spare pool
    /// that an earlier pass gave back as its command buffer completed (`render/coverage.rs`).
    pub coverage_pools_made: u64,
    /// Of `reading_draws_snapshotted`, the draws that read a color whose image lacks the
    /// `STORAGE` a later framebuffer fetch of it in the interlock would need, so that a fetch would
    /// render it through a stand-in for the whole pass: each pins its instance, and those before
    /// it, to the color's own image (`render/snapshot.rs`).
    pub reading_draws_pinned: u64,
    /// Passes whose stand-ins arrived after a pinned instance (`reading_draws_pinned`): the
    /// instances through the last pinned one rendered the attachments themselves, and each
    /// stand-in took its attachment's texels after them, not before the pass
    /// (`render/snapshot.rs`).
    pub stand_ins_after_pins: u64,
    /// Linear images aliasing a buffer's bytes (a texture over a buffer, a no-copy buffer or an
    /// IOSurface plane) that the device made with feedback loop usage
    /// (`texture::feedback_candidate`), so a draw may read the texture through a snapshot while
    /// its pass renders it (`render/snapshot.rs`): one per image made. Where the device makes no
    /// such image with the usage, it makes one without, and the texture's readers split.
    pub feedback_loop_alias_images: u64,
    /// Linear images aliasing a buffer of the device's own memory, of textures whose rows or
    /// slices lie with bytes between them that the buffer declares unobserved
    /// (`BufferDescriptor::padding_unobserved`): each image holds those bytes, where without the
    /// declaration the texture is a shadow (`shadowed_textures`). One per image made. A runtime
    /// extension: Metal has none.
    pub unobserved_padding_aliases: u64,
    /// Linear images aliasing a buffer of the device's own memory, of textures that render
    /// nothing, whose rows or slices lie with bytes between them that only the device's ordered
    /// commands write (a `Private` buffer, a `Managed` buffer's own on a device without unified
    /// memory): each image holds those bytes, which the buffer and the image read alike (Memory
    /// Aliasing), where a buffer other writers reach makes the texture a shadow
    /// (`shadowed_textures`). The device's linear image, made `PREINITIALIZED`, keeps them through
    /// its one layout transition (Vulkan's Image Layout Transitions); a `DRM_FORMAT_MOD_LINEAR`
    /// one at the buffer's pitch, made `UNDEFINED`, through copies around it
    /// (`first_moves_kept`). One per image made.
    pub kept_padding_aliases: u64,
    /// First moves of images aliasing a buffer's bytes that kept those bytes by copies on the
    /// device around the move's layout transition: a `DRM_FORMAT_MOD_LINEAR` image at the
    /// buffer's pitch over rows whose bytes between them only the device's ordered commands write
    /// (`linear_plan::Between::Kept`), made `UNDEFINED`, whose first move may discard what it is
    /// bound to. The move copies that span into a scratch buffer, moves the image to `GENERAL`, and
    /// copies the span back through the buffer, after every earlier access to it on every
    /// `VkQueue` and before every later one. One per move recorded. A runtime extension: Metal
    /// has none.
    pub first_moves_kept: u64,
    /// The bytes those moves kept: each one's span, copied out and back once.
    pub first_move_bytes_kept: u64,
    /// Images aliasing a buffer of the device's own memory that are `DRM_FORMAT_MOD_LINEAR` images
    /// at Metal's row pitch (`VK_EXT_image_drm_format_modifier`), made where the device's own
    /// linear image of the texture lays its rows out otherwise and the device's check of such
    /// images over its pooled memory passed for the texture's accesses (docs/API.md, textures over
    /// host pages); without one the texture is a shadow (`shadowed_textures`). One per image made.
    /// A runtime extension: Metal has none.
    pub pooled_pitched_images: u64,
    /// Textures over a buffer of the device's pooled memory made shadows before the device's
    /// check of that memory had answered (`TexturePlacement::Pending` as they were made, counted
    /// in `shadowed_textures`) that moved to their `DRM_FORMAT_MOD_LINEAR` image at Metal's pitch
    /// once the check passed for their accesses, at the commit of the device's only uncommitted
    /// command buffer, with no copy (`texture/storage_form.rs`, as `host_textures_moved_in_place`
    /// over host pages): each is counted in `pooled_pitched_images` too from then on. Over memory
    /// the runtime exports (`MemoryExport`), likewise once the check of that exported memory
    /// passed, to the linear image or the `DRM_FORMAT_MOD_LINEAR` one made for its handle type
    /// (the latter counted in `pooled_pitched_images`). A runtime extension: Metal has none.
    pub pooled_textures_moved_in_place: u64,
    /// Shadow images of textures that moved in place over the device's pooled memory
    /// (`pooled_textures_moved_in_place`) let go of once the command buffer that took the move
    /// completed, as `host_texture_shadows_freed_after_move` counts those over host pages.
    pub pooled_texture_shadows_freed_after_move: u64,
    /// Images of textures over a buffer's bytes that the encoders copy them into and back (the
    /// texture's shadow, `shadow.rs`: a texture over host pages that renders, among others) that
    /// the device made with feedback loop usage where it cost them nothing
    /// (`texture::feedback_candidate`), so a draw may read the texture through a snapshot while
    /// its pass renders that image (`render/snapshot.rs`): one per image made. Where the usage
    /// costs the image, it is made without, and the texture's readers split.
    pub feedback_loop_shadow_images: u64,
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
    /// Draws under a visibility result (`Counting` or `Boolean`) whose fragment function declares
    /// `[[early_fragment_tests]]` and writes its `SampleMask` output
    /// (`GraphicsState::writes_sample_mask`), on a device that does not state
    /// `earlyFragmentMultisampleCoverageAfterSampleCounting`, each of those that took every step
    /// (a draw drawn again or rebound, `render/draw_again.rs`, repeats its kept draw's
    /// announcement and adds nothing here): each announced `Degradation::VisibilityPastEarlyMask`,
    /// its result possibly short of the samples the mask drops, which Metal counts.
    pub draws_counted_past_early_masks: u64,
    /// The slots of draws that take every step (not a draw again or rebound,
    /// `render/draw_again.rs`) that sample a texture a color attachment of their render pass
    /// instance renders at a level and layer the slot's view presents, and so read its copy
    /// (`texture/attached_reads.rs`): one per such slot of each draw. The eight counts after it
    /// split them by how the levels the slot's reads reach (`StageModule::level_reads`, through
    /// the samplers the draw binds) meet the attached levels, counted from the view's first, and
    /// whether the slot then reads the texture's own levels before the lowest attached one, or
    /// past the highest, in place of its copy (`render/attached_levels.rs`).
    pub attached_read_slots: u64,
    /// Of `attached_read_slots`, those of a texture of one level.
    pub attached_read_slots_one_level: u64,
    /// Those a read of which may reach any level of the view (`LevelReads::any`: a level the
    /// function computes, a level-count query beyond a constant's containment, a level-of-detail
    /// query, a sample through no sampler or with a minimum level of detail).
    pub attached_read_slots_any_level: u64,
    /// Those of a texture of several levels whose own first level an attachment renders, a read
    /// of which may reach an attached level or one before it, and one past it: a sample through
    /// the sampler the draw binds (its bounds, or an explicit level of detail clamped into them),
    /// a fetch, gather or size query at a constant level, or a sample through a sampler the draw
    /// binds apart from the texture (a separate sampler), whose levels are not counted; or whose
    /// instance holds an attached level in the feedback loop layout.
    pub attached_read_slots_first_level: u64,
    /// The same, of a texture whose lowest attached level is past its own first.
    pub attached_read_slots_reaching: u64,
    /// Those every read of which reaches only levels before the lowest attached one, or only
    /// levels past the highest, which keep the copy as the texture's own image cannot serve them:
    /// no image of its own (a texture over a buffer's bytes), a sparse one, one with a decoded
    /// twin, or one whose image lacks the sampled usage.
    pub attached_read_slots_apart: u64,
    /// Those every read of which reaches only levels before the lowest attached one, or only
    /// levels past the highest, which keep the copy as something writes those levels: a write
    /// among the draw's tracked uses, an earlier draw's write since the instance's last barrier,
    /// or an attachment of the instance (a resolve's, a depth or stencil one's) at them.
    pub attached_read_slots_written: u64,
    /// Those every read of which reaches only levels before the lowest attached one, each bound
    /// through a view of the texture's own image of its levels from its first up to the lowest
    /// attached one, in place of the copy: no copy of the texture is taken for the slot, and its
    /// texels are the copy's, which holds the same levels as they were before the instance.
    pub attached_read_slots_narrowed: u64,
    /// Those every read of which reaches only levels past the highest attached one, each bound
    /// through a view of the texture's own image of its levels from its first up to the highest
    /// the slot reads (every level where its sampler's `maxLod` reaches past the last), in place
    /// of the copy: no copy of the texture is taken for the slot, and the levels it reads hold the
    /// copy's texels, as nothing the instance does writes them. The view holds the attached levels
    /// too, unread, in `GENERAL` as the instance holds them.
    pub attached_read_slots_past: u64,
    /// Textures that took two faces, two images that take turns as the texture's own, at their
    /// first attached read (`texture/faces.rs`): a texture of one level, slice and sample the
    /// runtime made `Private` and tracked, in no heap and over no buffer, with no view. The second
    /// face is that read's shadow, as the copy's image would have been.
    pub texture_faces_made: u64,
    /// The turns of such textures to their other face, each at the open of a render pass whose one
    /// color renders the texture, loaded `Clear` or `DontCare` (`render/face_swap.rs`): the pass
    /// renders the face its texture did not present, and its draws' attached reads read the one it
    /// did, which holds the texels from before the pass, as the copy in ahead of the pass's load
    /// actions would.
    pub attached_read_swaps: u64,
    /// The turned passes whose first instance read the face from before the pass with no copy in
    /// (not one whose barrier before its first draw makes that copy in a clear, nor one that reads
    /// nothing).
    pub attached_read_swap_reads: u64,
    /// The texels the copies in of `attached_read_swap_reads` would have moved.
    pub attached_read_swap_texels: u64,
    /// The copies between a texture's faces a batch's submission recorded around it for a turn
    /// whose pass had not run: a batch of a command buffer made after the turn and submitted first
    /// that names the new face, or the first after a turn whose command buffer ran nothing. Each
    /// direction counts.
    pub attached_read_swap_fixes: u64,
    /// The texels the copies of `attached_read_swap_fixes` moved.
    pub attached_read_swap_fix_texels: u64,
    /// The turns among `attached_read_swaps` for a pass whose render area (its extent, from the
    /// origin) is short of the texture's: before the pass's first instance the face turned to takes
    /// the texels outside what the color's clear rewrites (the area, or a `Clear` load's reach past
    /// it) that the faces may not share, from the face presented before the turn
    /// (`render/face_swap.rs`).
    pub attached_read_swaps_short_area: u64,
    /// The texels the copies of `attached_read_swaps_short_area` moved: none where the faces share
    /// every texel outside what the color's clear rewrites, as across command buffers after a turn
    /// of a pass of the same area.
    pub attached_read_swap_area_texels: u64,
    /// The copies in of an attached read's shadow before a pass's first instance that no turn took
    /// off, for its color loaded (`Load`). These nine are exhaustive over the copies in of the
    /// attached reads' shadows of a pass's color textures, each with its texels
    /// (`attached_read_unswapped_texels_*`); `attached_read_unswapped_area` is the sum of its five
    /// parts (`attached_read_unswapped_area_*`).
    pub attached_read_unswapped_load: u64,
    /// The copies in of an attached read's shadow before a pass's first instance that no turn took
    /// off, for the texture of several levels, slices or samples, or not 2D.
    pub attached_read_unswapped_levels: u64,
    /// The copies in of an attached read's shadow before a pass's first instance that no turn took
    /// off, for its color rendering part of the texture, another attachment of the pass rendering
    /// or resolving into it, a resolve of it, its color memoryless or its pass failing its command
    /// buffer: the sum of the five `attached_read_unswapped_area_*`.
    pub attached_read_unswapped_area: u64,
    /// The copies of `attached_read_unswapped_area` for a color that renders another level or
    /// slice of the texture, several layers or samples of it, or through an attachment of another
    /// extent than the texture's.
    pub attached_read_unswapped_area_extent: u64,
    /// The copies of `attached_read_unswapped_area` for a color that resolves.
    pub attached_read_unswapped_area_resolve: u64,
    /// The copies of `attached_read_unswapped_area` for another attachment of the pass that renders
    /// or resolves into the texture: another color, a color's resolve, or the depth or stencil
    /// attachment.
    pub attached_read_unswapped_area_shared: u64,
    /// The copies of `attached_read_unswapped_area` for a memoryless color.
    pub attached_read_unswapped_area_memoryless: u64,
    /// The copies of `attached_read_unswapped_area` for a pass that fails its command buffer.
    pub attached_read_unswapped_area_failing: u64,
    /// The copies in of an attached read's shadow before a pass's first instance that no turn took
    /// off, for the texture no image of its own fit to turn: over a buffer or a surface, in a heap,
    /// shareable, a view or with views, its image handed out, not `Private` and tracked.
    pub attached_read_unswapped_storage: u64,
    /// The copies in of an attached read's shadow before a pass's first instance that no turn took
    /// off, for its turns over for good (`texture/faces.rs`).
    pub attached_read_unswapped_frozen: u64,
    /// The copies in of an attached read's shadow before a pass's first instance that no turn took
    /// off, for its first attached read, which made its faces.
    pub attached_read_unswapped_first: u64,
    /// The copies in of an attached read's shadow before a pass's first instance that no turn took
    /// off, for something else that could name the face presented: another uncommitted command
    /// buffer, a pending batch, a heap set bound or its tickets out, a turn earlier in the command
    /// buffer or one whose pass was not submitted yet.
    pub attached_read_unswapped_busy: u64,
    /// The copies in of an attached read's shadow before a pass's first instance that no turn took
    /// off, for a pass that turns nothing: color groups, a rasterization-rate map, or an encoding
    /// again whose first encoding turned nothing (`texture/faces.rs`, **Encoded again**).
    pub attached_read_unswapped_pass: u64,
    /// The copies in of an attached read's shadow before a later instance of a pass, turned or not:
    /// a read after a barrier, which reads the draws before it.
    pub attached_read_unswapped_split: u64,
    /// The texels the copies of `attached_read_unswapped_load` moved.
    pub attached_read_unswapped_texels_load: u64,
    /// The texels the copies of `attached_read_unswapped_levels` moved.
    pub attached_read_unswapped_texels_levels: u64,
    /// The texels the copies of `attached_read_unswapped_area` moved.
    pub attached_read_unswapped_texels_area: u64,
    /// The texels the copies of `attached_read_unswapped_area_extent` moved.
    pub attached_read_unswapped_texels_area_extent: u64,
    /// The texels the copies of `attached_read_unswapped_area_resolve` moved.
    pub attached_read_unswapped_texels_area_resolve: u64,
    /// The texels the copies of `attached_read_unswapped_area_shared` moved.
    pub attached_read_unswapped_texels_area_shared: u64,
    /// The texels the copies of `attached_read_unswapped_area_memoryless` moved.
    pub attached_read_unswapped_texels_area_memoryless: u64,
    /// The texels the copies of `attached_read_unswapped_area_failing` moved.
    pub attached_read_unswapped_texels_area_failing: u64,
    /// The texels the copies of `attached_read_unswapped_storage` moved.
    pub attached_read_unswapped_texels_storage: u64,
    /// The texels the copies of `attached_read_unswapped_frozen` moved.
    pub attached_read_unswapped_texels_frozen: u64,
    /// The texels the copies of `attached_read_unswapped_first` moved.
    pub attached_read_unswapped_texels_first: u64,
    /// The texels the copies of `attached_read_unswapped_busy` moved.
    pub attached_read_unswapped_texels_busy: u64,
    /// The texels the copies of `attached_read_unswapped_pass` moved.
    pub attached_read_unswapped_texels_pass: u64,
    /// The texels the copies of `attached_read_unswapped_split` moved.
    pub attached_read_unswapped_texels_split: u64,
    /// Ended compute, blit and render encoders whose logs their command buffers kept, as a rebind
    /// before the first submission may encode them again (`replay.rs`), and the bytes of those
    /// logs.
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
    /// Encoders a call of which, on another thread than a rebind of a buffer over host pages
    /// (`rebind_bytes_no_copy`), held or took the buffer while the rebind was making what lies
    /// over its bytes again, or after, having begun before (`rebind.rs`, **Other threads**): what
    /// the call read of the buffer may be the old pages', so the commit, or a rebind's turn,
    /// encodes the encoder again over the new pages, and one that cannot be encoded again fails
    /// `EncodedUseRebound`. Zero where no encoder races a rebind.
    pub encoders_raced_rebind: u64,
    /// Of those, the encoders committed that the runtime could not encode again, each failing
    /// `EncodedUseRebound`: zero, as the calls of every encoder that may be one are gated, so no
    /// rebind's publish overlaps them (`rebind.rs`, **Other threads**). A count past zero is a
    /// defect.
    pub encoders_raced_unreplayable: u64,
    /// Rebinds refused (`RebindProblem::EncodedUses`) because a call of an encoder the runtime
    /// may not encode again (a compute encoder's, or a render encoder's that keeps no log) on
    /// another thread was still under way 2 ms after the rebind asked to go on: the call reads
    /// the buffer wholly before the rebind or wholly after it, so the rebind gives way. Zero
    /// where no such call outlasts the bound (a call waiting for a `VkPipeline` being made, say);
    /// the caller may try again.
    pub rebinds_refused_calls_in_flight: u64,
    /// Rebinds refused (`RebindProblem::EncodedUses`) because a command buffer committed and not
    /// yet submitted whole used the buffer in an encoder the runtime cannot encode again (a
    /// compute encoder its unit of command predication reaches past, among others): a rebind
    /// before that submission would fail it (`EncodedUseRebound`), so it gives way until the
    /// command buffer's last event group is submitted; the caller may try again.
    pub rebinds_refused_unsubmitted: u64,
    /// Calls of such encoders that waited, as they began, for a rebind's publish on another
    /// thread to close (`rebind_bytes_no_copy`): from just before the buffer took its new pages
    /// to when everything over its bytes was made over them.
    pub encoder_calls_held_by_rebind: u64,
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
    /// Render variants' `VkPipeline`s that took the device's linked pipeline of their exact link
    /// (the same four libraries and pipeline layout), linked before by another variant: no driver
    /// call, and counted in neither `pipelines_linked` nor `variant_pipelines_made`
    /// (`pipeline/libraries.rs`). Render states whose whole key is equal but which are variants
    /// of their own share one: compiled pipelines of their own whose stages come out as the same
    /// words, such as descriptors that differ only in the call stack depth of a function that
    /// recurses through no visible function.
    pub linked_pipelines_shared: u64,
    /// Compute encoders' dispatches that handed the making of a region's `VkPipeline` to the
    /// device's pool and went on, rather than make it at the dispatch: each counted once per
    /// region it handed.
    pub dispatch_pipelines_pooled: u64,
    /// Region `VkPipeline`s let go past a bound, the least recently used first: a compute
    /// variant's memo's own (`pipeline/mod.rs`, `REGION_PIPELINES`), or the device's across every
    /// compute variant, released variants' kept ones included (`pipeline/region_ledger.rs`,
    /// `DEVICE_REGION_PIPELINES`): each one a dispatch takes again is made again. None while the
    /// dispatches take no more keys than the bounds.
    pub region_pipelines_evicted: u64,
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
        self.current().work_counts_now()
    }

    /// [`Device::work_counts`] on the generation that serves (`device/line.rs`).
    pub(crate) fn work_counts_now(&self) -> WorkCounts {
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
        let commands = |count: fn(&CommandCounters) -> &AtomicU64| counters.commands.sum(count);
        let attached = |levels: AttachedLevels| {
            (counters.commands).sum(|counts| &counts.attached_read_slots_by[levels as usize])
        };
        let carry_stopped = |why: CarryStop| {
            (counters.commands).sum(|counts| &counts.draws_carry_stopped_by[why as usize])
        };
        let let_go = |by: LoggedCall| {
            (counters.commands).sum(|counts| &counts.kept_draws_let_go_by[by as usize])
        };
        let kept_across = |by: LoggedCall| {
            (counters.commands).sum(|counts| &counts.kept_draws_kept_across_by[by as usize])
        };
        let whole_read = |why: WholeRead| {
            (counters.commands).sum(|counts| &counts.stage_tables_read_whole_by[why as usize])
        };
        let reshaped = |why: ReshapedBy| {
            (counters.commands).sum(|counts| &counts.draws_rebind_reshaped_by[why as usize])
        };
        let arguments_stopped = |why: ArgumentsStop| {
            (counters.commands)
                .sum(|counts| &counts.draws_rebind_stopped_arguments_by[why as usize])
        };
        let carry_dropped = |why: CarryDropped| {
            counters.kept_draws_carry_dropped[why as usize].load(Ordering::Relaxed)
        };
        let uncarried =
            |why: Uncarried| counters.encoders_uncarried[why as usize].load(Ordering::Relaxed);
        let none_kept_next = |next: NoneKeptNext| {
            counters.encoders_none_kept_next[next as usize].load(Ordering::Relaxed)
        };
        let group_end = |end: GroupEnd| counters.group_ends[end as usize].load(Ordering::Relaxed);
        let unswapped = |cause: Unturned| {
            let at = cause as usize;
            (
                counters.attached_read_unswapped[at].load(Ordering::Relaxed),
                counters.attached_read_unswapped_texels[at].load(Ordering::Relaxed),
            )
        };
        // The color's own causes, which `attached_read_unswapped_area` sums.
        let area = [
            Unturned::Extent,
            Unturned::Resolve,
            Unturned::Shared,
            Unturned::Memoryless,
            Unturned::Failing,
        ]
        .map(unswapped)
        .into_iter()
        .fold((0, 0), |(copies, texels), (c, t)| (copies + c, texels + t));
        let first_work_order = self.shared().textures.first_moves.submissions_waited();
        let (copied_base_host_bytes, copied_base_device_bytes) = self.shared().copied.base_bytes();
        let held_for =
            |reason: HeldFor| counters.heads_held_for[reason as usize].load(Ordering::Relaxed);
        let pending = |bucket: usize| counters.batches_by_pending[bucket].load(Ordering::Relaxed);
        let alone = |why: Alone| counters.batches_alone[why as usize].load(Ordering::Relaxed);
        let left_at_commit =
            |why: LeftAtCommit| counters.left_at_commit[why as usize].load(Ordering::Relaxed);
        let [behind, unshared_behind] = (counters.behind_other_queues)
            .each_ref()
            .map(|tally| tally.read());
        let unshared_long_waits = (counters.unshared_long_waits)
            .each_ref()
            .map(|count| count.load(Ordering::Relaxed));
        let fanned_boxed = (counters.rate_map_fanned_vertices_boxed)
            .each_ref()
            .map(|count| count.load(Ordering::Relaxed));
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
        let unskipped =
            |at: Unskipped| counters.shadow_texels_unskipped[at as usize].load(Ordering::Relaxed);
        let standing = |let_go: Option<LetGo>| {
            let at = let_go.map_or(0, |let_go| let_go as usize + 1);
            counters.shadow_texels_unskipped_standing[at].load(Ordering::Relaxed)
        };
        let unreadable_by = |bit: usize| {
            (counters.reading_draws_unreadable.get(bit)).map_or(0, |c| c.load(Ordering::Relaxed))
        };
        let split_by = |reason: SplitReason| {
            counters.reading_draws_split[reason as usize].load(Ordering::Relaxed)
        };
        let leading_covered = |share: LeadingShare| {
            counters.reading_draws_leading_covered[share as usize].load(Ordering::Relaxed)
        };
        let never_stood = |missed: Missed| {
            counters.shadow_texels_never_stood[missed as usize].load(Ordering::Relaxed)
        };
        let (sleeps, [by_commit, by_completion, by_host, by_slice], scheduler_spared, gpu_waits) =
            self.shared().scheduler.sleeps_and_wakings();
        let idle_stalls = self.shared().scheduler.idle_stall_counts();
        let (completion_wakes, completion_spared) = self.shared().completion.wakes();
        let (completion_parks, completion_parks_timed_out) = self.shared().completion.parks();
        let (handler_wakes, handler_spared) = self.shared().contexts.wakes();
        let command_buffer_wakes = counters.command_buffer_wakes.load(Ordering::Relaxed);
        let command_buffer_spared = counters.command_buffer_wakes_spared.load(Ordering::Relaxed);
        let event_wakes = counters.event_wakes.load(Ordering::Relaxed);
        let event_spared = counters.event_wakes_spared.load(Ordering::Relaxed);
        let (
            allocated,
            overflows,
            census,
            (parked, waiting),
            (made, freed),
            frees,
            host_copy,
            (ahead, fd_blocks),
        ) = {
            let memory = lock(&self.shared().memory);
            let turnover = memory.turnover();
            (
                memory.allocated(),
                memory.overflows(),
                memory.census(),
                memory.parked(),
                turnover,
                memory.frees(),
                memory.host_copy_blocks(),
                (memory.ahead_blocks(), memory.fd_blocks()),
            )
        };
        let (frees_posted, frees_queue_full, free_waits) = self.shared().frees.counts();
        let (frees_paced, frees_paced_nanos) = self.shared().frees.paced();
        let dirty = self.shared().dirty.tally();
        let (vk_devices_live, vk_devices_made, device_recoveries) = self.shared().lineage.counts();
        let (lost_programs_released, lost_textures_released) = self.shared().lineage.released();
        WorkCounts {
            vk_devices_live,
            vk_devices_made,
            device_recoveries,
            lost_programs_released,
            lost_textures_released,
            command_pools_made: counters.command_pools_made.load(Ordering::Relaxed),
            command_buffers_made: counters.command_buffers_made.load(Ordering::Relaxed),
            commands_recorded: self.shared().commands_recorded(),
            commands_uncounted: u64::from(self.shared().recording().is_none()),
            object_churn: churn.0,
            shadow_keeps: churn.1,
            copy_sites: self.shared().vk().copy_sites(),
            copy_regions_split: self.shared().vk().copy_regions_split(),
            spare_command_pools_taken: counters.spare_command_pools_taken.load(Ordering::Relaxed),
            spare_command_pools_taken_past_first: counters
                .spare_command_pools_taken_past_first
                .load(Ordering::Relaxed),
            spare_command_pools_reset: counters.spare_command_pools_reset.load(Ordering::Relaxed),
            spare_command_buffers_freed_past_recent_uses: counters.spare_command_buffers_freed[0]
                .load(Ordering::Relaxed),
            spare_command_buffers_freed_past_held: counters.spare_command_buffers_freed[1]
                .load(Ordering::Relaxed),
            spare_command_buffers_shed: counters.spare_command_buffers_freed[2]
                .load(Ordering::Relaxed),
            spare_command_buffers_freed_past_recorded: counters.spare_command_buffers_freed[3]
                .load(Ordering::Relaxed),
            spare_command_pool_trims: counters.spare_command_pool_trims.load(Ordering::Relaxed),
            spare_command_pool_trim_rounds: counters
                .spare_command_pool_trim_rounds
                .load(Ordering::Relaxed),
            spare_command_buffers_trimmed_in_rounds: counters
                .spare_command_buffers_trimmed_in_rounds
                .load(Ordering::Relaxed),
            live_tables_uploaded: counters.live_tables_uploaded.load(Ordering::Relaxed),
            timeline_waits: counters.timeline_waits.load(Ordering::Relaxed),
            fence_waits: counters.fence_waits.load(Ordering::Relaxed),
            missed_completion_wakes: counters.missed_completion_wakes.load(Ordering::Relaxed),
            completion_arms: counters.completion_arms.load(Ordering::Relaxed),
            stall_probes_sent: counters.stall_probes_sent.load(Ordering::Relaxed),
            wedges_declared: counters.wedges_declared.load(Ordering::Relaxed),
            watchdog_timeouts: counters.watchdog_timeouts.load(Ordering::Relaxed),
            unrun_command_buffers: counters.unrun_command_buffers.load(Ordering::Relaxed),
            longest_stall_nanos: counters.longest_stall_nanos.load(Ordering::Relaxed),
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
            batches_held_for_another_queue: (counters.batches_held_for_another_queue)
                .load(Ordering::Relaxed),
            command_buffers_completed_after_another_queue: (counters
                .command_buffers_completed_after_another_queue)
                .load(Ordering::Relaxed),
            completion_seen_handlers_run: (counters.completion_seen_handlers_run)
                .load(Ordering::Relaxed),
            completion_seen_handlers_where_seen: (counters.completion_seen_handlers_where_seen)
                .load(Ordering::Relaxed),
            command_buffers_unsubmitted: (counters.command_buffers_unsubmitted)
                .load(Ordering::Relaxed),
            sync_files_exported: counters.sync_files_exported.load(Ordering::Relaxed),
            sync_files_after_host_copies: (counters.sync_files_after_host_copies)
                .load(Ordering::Relaxed),
            completion_turns_where_taken: self.shared().completion.turns_where_taken(),
            scheduled_where_submitted: (counters.scheduled_where_submitted).load(Ordering::Relaxed),
            timestamp_slots_taken: counters.timestamp_slots_taken.load(Ordering::Relaxed),
            timestamp_slots_reset_on_host: (counters.timestamp_slots_reset_on_host)
                .load(Ordering::Relaxed),
            calibrations: counters.calibrations.load(Ordering::Relaxed),
            calibration_readings: counters.calibration_readings.load(Ordering::Relaxed),
            calibration_deviation_most: (counters.calibration_deviation_most)
                .load(Ordering::Relaxed),
            calibration_disagreements: (counters.calibration_disagreements).load(Ordering::Relaxed),
            loosely_calibrated_gpu_times: (counters.loosely_calibrated_gpu_times)
                .load(Ordering::Relaxed),
            storage_forms_taken: counters.storage_forms_taken.load(Ordering::Relaxed),
            addressable_buffers_made: counters.addressable_buffers_made.load(Ordering::Relaxed),
            mipmaps_filtered: counters.mipmaps_filtered.load(Ordering::Relaxed),
            mipmap_first_levels_fetched: counters
                .mipmap_first_levels_fetched
                .load(Ordering::Relaxed),
            pipelines_pooled: counters.pipelines_pooled.load(Ordering::Relaxed),
            residual_form_draws: commands(|counts| &counts.residual_form_draws),
            residual_form_makings: commands(|counts| &counts.residual_form_makings),
            pipeline_pool_busy: self.shared().pipeline_pool.busy() as u64,
            stage_entries_specialized: counters.stage_entries_specialized.load(Ordering::Relaxed),
            stage_entries_shared: counters.stage_entries_shared.load(Ordering::Relaxed),
            functions_translated: counters.functions_translated.load(Ordering::Relaxed),
            widest_pin_kernels_emulated: (counters.widest_pin_kernels_emulated)
                .load(Ordering::Relaxed),
            unit_axes_variants_begun: (counters.unit_axes_variants_begun).load(Ordering::Relaxed),
            unit_axes_dispatches: (counters.unit_axes_dispatches).load(Ordering::Relaxed),
            unit_axes_meanwhile: (counters.unit_axes_meanwhile).load(Ordering::Relaxed),
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
            makings_held_for_compile_memory: counters
                .makings_held_for_compile_memory
                .load(Ordering::Relaxed),
            compile_memory_held_nanos: counters.compile_memory_held_nanos.load(Ordering::Relaxed),
            heap_trims: counters.heap_trims.load(Ordering::Relaxed),
            predicated_loop_runs_again: counters.predicated_loop_runs_again.load(Ordering::Relaxed),
            variants_taken_back: counters.variants_taken_back.load(Ordering::Relaxed),
            unlisted_argument_sampler_buffers: counters
                .unlisted_argument_sampler_buffers
                .load(Ordering::Relaxed),
            draws_dropped_unmade: counters.draws_dropped_unmade.load(Ordering::Relaxed),
            host_step_sampler_sets_made: counters
                .host_step_sampler_sets_made
                .load(Ordering::Relaxed),
            host_steps_recorded: counters.host_steps_recorded.load(Ordering::Relaxed),
            host_step_nanos: counters.host_step_nanos.load(Ordering::Relaxed),
            host_step_pipelines_made: counters.host_step_pipelines_made.load(Ordering::Relaxed),
            host_step_chunks_kept: counters.host_step_chunks_kept.load(Ordering::Relaxed),
            host_step_chunks_recorded_again: counters
                .host_step_chunks_recorded_again
                .load(Ordering::Relaxed),
            host_step_chunk_copies_put_off: counters
                .host_step_chunk_copies_put_off
                .load(Ordering::Relaxed),
            host_step_records_compared: counters.host_step_records_compared.load(Ordering::Relaxed),
            host_step_compare_nanos: counters.host_step_compare_nanos.load(Ordering::Relaxed),
            replayed_heap_draws_unshared: counters
                .replayed_heap_draws_unshared
                .load(Ordering::Relaxed),
            replayed_heap_draws_writing: counters
                .replayed_heap_draws_writing
                .load(Ordering::Relaxed),
            replayed_heap_draws_noted: counters.replayed_heap_draws_noted.load(Ordering::Relaxed),
            replayed_heap_draws_over_sampler_tables: counters
                .replayed_heap_draws_over_sampler_tables
                .load(Ordering::Relaxed),
            replayed_heap_draws_table_textures: counters
                .replayed_heap_draws_table_textures
                .load(Ordering::Relaxed),
            commit_refusals_held: counters.commit_refusals_held.load(Ordering::Relaxed),
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
            pooled_backings_taken: counters.pooled_backings_taken.load(Ordering::Relaxed),
            pooled_backings_taken_in_flight: counters
                .pooled_backings_taken_in_flight
                .load(Ordering::Relaxed),
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
            heap_texture_refused_withheld: counters
                .heap_texture_refused_withheld
                .load(Ordering::Relaxed),
            twin_decode_failed_withheld: counters
                .twin_decode_failed_withheld
                .load(Ordering::Relaxed),
            late_reservation_withheld: counters.late_reservation_withheld.load(Ordering::Relaxed),
            queue_limits_reached_withheld: counters
                .queue_limits_reached_withheld
                .load(Ordering::Relaxed),
            device_memory_allocated: allocated,
            device_memory_overflows: overflows,
            memory_blocks_live: census.pooled,
            empty_memory_blocks_live: census.empty,
            pooled_memory_blocks_made: made,
            pooled_memory_blocks_freed: freed,
            dedicated_memory_blocks_live: census.dedicated,
            host_copy_blocks_readied: host_copy.readied,
            host_copy_blocks_made_in_place: host_copy.made_in_place,
            buffer_blocks_made_ahead: ahead.buffers_ahead,
            buffer_blocks_made_in_place: ahead.buffers_in_place,
            fd_buffer_blocks_made: fd_blocks.exportable,
            fd_buffer_blocks_plain: fd_blocks.plain,
            plain_buffer_blocks_made: fd_blocks.plain_buffers,
            page_buffer_blocks_made: fd_blocks.pages,
            page_buffer_blocks_plain: fd_blocks.pages_plain,
            image_blocks_made_ahead: ahead.images_ahead,
            image_blocks_made_in_place: ahead.images_in_place,
            blocks_ahead_waited_for: ahead.waited_for,
            blocks_ahead_hurried: ahead.hurried,
            blocks_ahead_past_quiet_wait: ahead.past_quiet_wait,
            blocks_ahead_unused: ahead.unused,
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
            heap_arrays_copied: counters.heap_arrays_copied.load(Ordering::Relaxed),
            host_range_lookups: self.shared().aliases.range_lookups(),
            host_alias_lookups: self.shared().aliases.lookups(),
            heap_textures_tiled: counters.heap_textures_tiled.load(Ordering::Relaxed),
            heap_textures_refit: counters.heap_textures_refit.load(Ordering::Relaxed),
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
            buffer_copies_merged: counters.buffer_copies_merged.load(Ordering::Relaxed),
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
            ycbcr_planes_bound: commands(|counts| &counts.ycbcr_planes_bound),
            texel_rows_textures: counters.texel_rows_textures.load(Ordering::Relaxed),
            texel_rows_bound: commands(|counts| &counts.texel_rows_bound),
            texel_buffer_parts_bound: commands(|counts| &counts.texel_buffer_parts_bound),
            uniform_texels_bound: commands(|counts| &counts.uniform_texels_bound),
            uniform_buffers_bound_to_16k: commands(|counts| &counts.uniform_buffers_bound_to_16k),
            uniform_buffers_bound_past_16k: commands(|counts| {
                &counts.uniform_buffers_bound_past_16k
            }),
            ycbcr_planes_images: counters.ycbcr_planes_images.load(Ordering::Relaxed),
            fd_blocks_awaited: counters.fd_blocks_awaited.load(Ordering::Relaxed),
            fd_blocks_refused: counters.fd_blocks_refused.load(Ordering::Relaxed),
            ycbcr_planes_images_kept_conversion: counters.ycbcr_planes_images_kept[0]
                .load(Ordering::Relaxed),
            ycbcr_planes_images_kept_memory: counters.ycbcr_planes_images_kept[1]
                .load(Ordering::Relaxed),
            ycbcr_planes_images_kept_layout: counters.ycbcr_planes_images_kept[2]
                .load(Ordering::Relaxed),
            ycbcr_planes_images_kept_exportable_block: counters.ycbcr_planes_images_kept[3]
                .load(Ordering::Relaxed),
            ycbcr_planes_images_bound: commands(|counts| &counts.ycbcr_planes_images_bound),
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
            indirect_record_runs_read: counters.indirect_record_runs_read.load(Ordering::Relaxed),
            indirect_record_bytes_gpu_copied: (counters.indirect_record_bytes_gpu_copied)
                .load(Ordering::Relaxed),
            indirect_record_bytes_gpu_reset: (counters.indirect_record_bytes_gpu_reset)
                .load(Ordering::Relaxed),
            indirect_record_reset_gpu_nanos: (counters.indirect_record_reset_gpu_nanos)
                .load(Ordering::Relaxed),
            icb_records_moved_to_host: (counters.icb_records_moved_to_host).load(Ordering::Relaxed),
            icb_record_bytes_moved: counters.icb_record_bytes_moved.load(Ordering::Relaxed),
            icb_records_staged_accesses: (counters.icb_records_staged_accesses)
                .load(Ordering::Relaxed),
            indirect_record_bytes_copied: (counters.indirect_record_bytes_copied)
                .load(Ordering::Relaxed),
            indirect_write_residencies_unmarked: (counters.indirect_write_residencies_unmarked)
                .load(Ordering::Relaxed),
            indirect_chunks_kept: counters.indirect_chunks_kept.load(Ordering::Relaxed),
            indirect_commands_kept: counters.indirect_commands_kept.load(Ordering::Relaxed),
            kept_contexts_made: counters.kept_contexts_made.load(Ordering::Relaxed),
            kept_resident_keys_compared: counters
                .kept_resident_keys_compared
                .load(Ordering::Relaxed),
            gpu_form_commands: counters.gpu_form_commands.load(Ordering::Relaxed),
            merged_draw_commands: counters.merged_draw_commands.load(Ordering::Relaxed),
            gpu_form_device_local_bytes: (counters.gpu_form_device_local_bytes)
                .load(Ordering::Relaxed),
            gpu_pool_blocks_made: counters.gpu_pool_blocks_made.load(Ordering::Relaxed),
            gpu_pool_blocks_released: counters.gpu_pool_blocks_released.load(Ordering::Relaxed),
            gpu_pool_leases_peak: counters.gpu_pool_leases_peak.load(Ordering::Relaxed),
            internal_submissions: counters.internal_submissions.load(Ordering::Relaxed),
            queue_submissions: counters.queue_submissions.load(Ordering::Relaxed),
            submit_infos: counters.submit_infos.load(Ordering::Relaxed),
            command_buffers_submitted: counters.command_buffers_submitted.load(Ordering::Relaxed),
            batch_submit_nanos: counters.batch_submit_nanos.load(Ordering::Relaxed),
            batch_submit_cpu_nanos: counters.batch_submit_cpu_nanos.load(Ordering::Relaxed),
            submit_long_calls: counters.submit_long_calls.load(Ordering::Relaxed),
            submit_long_call_nanos: counters.submit_long_call_nanos.load(Ordering::Relaxed),
            submit_long_call_cpu_nanos: counters.submit_long_call_cpu_nanos.load(Ordering::Relaxed),
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
            encoder_primaries_begun_ahead_missed: (counters.encoder_primaries_begun_ahead_missed)
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
            group_ends_long: group_end(GroupEnd::Long),
            group_ends_drained: group_end(GroupEnd::Drained),
            group_ends_checks: group_end(GroupEnd::Checks),
            group_ends_held: group_end(GroupEnd::Held),
            group_ends_closed: group_end(GroupEnd::Closed),
            batches_with_none_pending: pending(0),
            batches_with_one_pending: pending(1),
            batches_with_two_or_three_pending: pending(2),
            batches_with_four_or_more_pending: pending(3),
            batches_alone_unshared: alone(Alone::Unshared),
            batches_alone_binary_waits: alone(Alone::BinaryWait),
            batches_alone_binary_signals: alone(Alone::BinarySignal),
            batches_alone_sparse: alone(Alone::Sparse),
            batches_alone_pieced: alone(Alone::Pieced),
            batches_alone_fences: alone(Alone::Fence),
            batches_alone_run_unknown: alone(Alone::RunUnknown),
            batches_alone_run_long: alone(Alone::RunLong),
            batches_alone_first_work: alone(Alone::FirstWork),
            batches_alone_at_commit: alone(Alone::AtCommit),
            batches_held_alone_one_pending: alone(Alone::OnePending),
            batches_held_alone_unshared_hold: alone(Alone::UnsharedHold),
            batches_held_alone_gpu_waits: alone(Alone::GpuWait),
            batches_held_alone_handlers: alone(Alone::Handlers),
            batches_held_alone_waited_for: alone(Alone::WaitedFor),
            batches_held_alone_kept: alone(Alone::Kept),
            batches_held_alone_sent_by_next: alone(Alone::SentByNext),
            batches_held_alone_other: alone(Alone::Other),
            batches_held_alone_unkept: alone(Alone::Unkept),
            batches_of_unshared_queues: (counters.batches_of_unshared_queues)
                .load(Ordering::Relaxed),
            groups_kept_pending: counters.groups_kept_pending.load(Ordering::Relaxed),
            groups_sent_for_waits: counters.groups_sent_for_waits.load(Ordering::Relaxed),
            heads_held_for_binary_semaphores: held_for(HeldFor::BinarySemaphore),
            heads_held_for_reencodes: held_for(HeldFor::Reencoding),
            heads_held_for_decodes: held_for(HeldFor::Decode),
            heads_held_for_host_steps: held_for(HeldFor::HostStep),
            heads_held_for_checks: held_for(HeldFor::Check),
            heads_held_for_event_values: held_for(HeldFor::EventValue),
            heads_held_for_earlier_commits: held_for(HeldFor::EarlierCommit),
            heads_held_for_unshared_isolation: held_for(HeldFor::UnsharedIsolation),
            heads_let_go_unshared_idle: counters.heads_let_go_unshared_idle.load(Ordering::Relaxed),
            heads_held_for_unshared_commits: counters
                .heads_held_for_unshared_commits
                .load(Ordering::Relaxed),
            heads_held_for_queue_turns: held_for(HeldFor::QueueTurn),
            heads_let_go_turn_idle: counters.heads_let_go_turn_idle.load(Ordering::Relaxed),
            heads_waiting_at_their_gate: counters
                .heads_waiting_at_their_gate
                .load(Ordering::Relaxed),
            scheduled_behind_host_steps: counters
                .scheduled_behind_host_steps
                .load(Ordering::Relaxed),
            host_step_gates_narrowed: counters.host_step_gates_narrowed.load(Ordering::Relaxed),
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
            scheduler_slice_rescues: counters.scheduler_slice_rescues.load(Ordering::Relaxed),
            scheduler_gpu_waits: gpu_waits,
            scheduler_passes: self.shared().scheduler.passes(),
            scheduler_passes_by_waiters: counters
                .scheduler_passes_by_waiters
                .load(Ordering::Relaxed),
            batches_submitted_by_waiters: counters
                .batches_submitted_by_waiters
                .load(Ordering::Relaxed),
            scheduler_passes_at_commit: counters.scheduler_passes_at_commit.load(Ordering::Relaxed),
            scheduler_passes_at_commit_ended_early: counters
                .scheduler_passes_at_commit_ended_early
                .load(Ordering::Relaxed),
            scheduler_at_commit_left_taken_in: left_at_commit(LeftAtCommit::TakenIn),
            scheduler_at_commit_left_passing: left_at_commit(LeftAtCommit::Passing),
            scheduler_at_commit_left_gone: left_at_commit(LeftAtCommit::Gone),
            scheduler_at_commit_left_behind_its_queue: left_at_commit(LeftAtCommit::BehindItsQueue),
            scheduler_at_commit_left_behind_another_queue: left_at_commit(
                LeftAtCommit::BehindAnotherQueue,
            ),
            scheduler_at_commit_left_event_waits: left_at_commit(LeftAtCommit::EventWait),
            scheduler_at_commit_left_checks: left_at_commit(LeftAtCommit::Checks),
            scheduler_at_commit_left_host_steps: left_at_commit(LeftAtCommit::HostStep),
            scheduler_at_commit_left_binary_waits: left_at_commit(LeftAtCommit::BinaryWait),
            scheduler_at_commit_left_unshared: left_at_commit(LeftAtCommit::Unshared),
            scheduler_at_commit_left_turns: left_at_commit(LeftAtCommit::Turn),
            scheduler_at_commit_left_slow_calls: left_at_commit(LeftAtCommit::SlowCalls),
            scheduler_at_commit_left_other: left_at_commit(LeftAtCommit::Other),
            footprints_built: counters.footprints_built.load(Ordering::Relaxed),
            footprint_lock_waits: counters.footprint_lock_waits.load(Ordering::Relaxed),
            batches_submitted_at_commit: counters
                .batches_submitted_at_commit
                .load(Ordering::Relaxed),
            batches_checked_at_commit: counters.batches_checked_at_commit.load(Ordering::Relaxed),
            commit_calls: counters.commit_calls.load(Ordering::Relaxed),
            commit_nanos: counters.commit_nanos.load(Ordering::Relaxed),
            commit_lowering_nanos: counters.commit_lowering_nanos.load(Ordering::Relaxed),
            commit_recording_nanos: counters.commit_recording_nanos.load(Ordering::Relaxed),
            commit_pipeline_wait_nanos: counters.commit_pipeline_wait_nanos.load(Ordering::Relaxed),
            commit_lock_wait_nanos: counters.commit_lock_wait_nanos.load(Ordering::Relaxed),
            commit_pass_nanos: counters.commit_pass_nanos.load(Ordering::Relaxed),
            commit_submit_nanos: counters.commit_submit_nanos.load(Ordering::Relaxed),
            wakes_completion_waiters: completion_wakes,
            completion_waiter_parks: completion_parks,
            completion_waiter_parks_timed_out: completion_parks_timed_out,
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
            command_queues_live_most: counters.command_queues_live_most.load(Ordering::Relaxed),
            unshared_queues_sharing_a_vk_queue: (counters.unshared_queues_sharing_a_vk_queue)
                .load(Ordering::Relaxed),
            queue_moves: counters.queue_moves.load(Ordering::Relaxed),
            queue_joins: counters.queue_joins.load(Ordering::Relaxed),
            queues_gone_long: counters.queues_gone_long.load(Ordering::Relaxed),
            long_queue_settles: counters.long_queue_settles.load(Ordering::Relaxed),
            queue_moves_off_unshared: counters.queue_moves_off_unshared.load(Ordering::Relaxed),
            joins_withheld_from_unshared: counters
                .joins_withheld_from_unshared
                .load(Ordering::Relaxed),
            picks_with_none_busy: counters.picks_with_none_busy.load(Ordering::Relaxed),
            joins_withheld_from_old_heads: counters
                .joins_withheld_from_old_heads
                .load(Ordering::Relaxed),
            command_buffers_behind_other_queues: behind[0],
            command_buffers_behind_other_queues_over_16ms: behind[1],
            command_buffers_behind_other_queues_over_33ms: behind[2],
            wait_behind_other_queues_longest_nanos: behind[3],
            unshared_command_buffers_behind_other_queues: unshared_behind[0],
            unshared_command_buffers_behind_other_queues_over_16ms: unshared_behind[1],
            unshared_command_buffers_behind_other_queues_over_33ms: unshared_behind[2],
            unshared_wait_behind_other_queues_longest_nanos: unshared_behind[3],
            unshared_waits_over_16ms_batches_ahead: unshared_long_waits[0],
            unshared_waits_over_16ms_most_batches_ahead: unshared_long_waits[1],
            unshared_waits_over_16ms_longest_runs_nanos: unshared_long_waits[2],
            unshared_waits_over_16ms_behind_one_long_run: unshared_long_waits[3],
            queue_switch_checks_run: counters.queue_switch_checks_run.load(Ordering::Relaxed),
            queue_switch_checks_recalled: counters
                .queue_switch_checks_recalled
                .load(Ordering::Relaxed),
            submit_checks_run: counters.submit_checks_run.load(Ordering::Relaxed),
            submit_checks_recalled: counters.submit_checks_recalled.load(Ordering::Relaxed),
            submit_check_calls: counters.submit_check_calls.load(Ordering::Relaxed),
            device_checks: {
                let made = super::churn_counts::checks_made;
                let released = super::churn_counts::checks_released;
                DeviceCheckWork {
                    vk_memory_allocations: made(VkMaking::Memory),
                    vk_memory_freed: released(VkRelease::Memory),
                    vk_buffers_made: made(VkMaking::Buffer),
                    vk_buffers_destroyed: released(VkRelease::Buffer),
                    vk_images_made: made(VkMaking::Image),
                    vk_images_destroyed: released(VkRelease::Image),
                    vk_image_views_made: made(VkMaking::ImageView),
                    vk_buffer_views_destroyed: released(VkRelease::BufferView),
                    vk_image_bindings: made(VkMaking::ImageBinding),
                    vk_buffer_bindings: made(VkMaking::BufferBinding),
                    queue_submissions: counters.check_queue_submissions.load(Ordering::Relaxed),
                    steps_aside: counters.check_steps_aside.load(Ordering::Relaxed),
                    aside_nanos: counters.check_aside_nanos.load(Ordering::Relaxed),
                    view_check_stand_bys: counters.view_check_stand_bys.load(Ordering::Relaxed),
                    view_check_stand_by_nanos: (counters.view_check_stand_by_nanos)
                        .load(Ordering::Relaxed),
                }
            },
            queue_sharing_untimed: counters.queue_sharing_untimed.load(Ordering::Relaxed),
            completed_waits: counters.completed_waits.load(Ordering::Relaxed),
            event_waits_met: counters.event_waits_met.load(Ordering::Relaxed),
            shared_event_waits_held: counters.shared_event_waits_held.load(Ordering::Relaxed),
            event_signals_behind_merge: counters.event_signals_behind_merge.load(Ordering::Relaxed),
            bare_starts: counters.bare_starts.load(Ordering::Relaxed),
            prefixed_starts: counters.prefixed_starts.load(Ordering::Relaxed),
            command_barriers: counters.command_barriers.load(Ordering::Relaxed),
            command_buffers_split_at_aliases: (counters.command_buffers_split_at_aliases)
                .load(Ordering::Relaxed),
            commands_ordered_inside: commands(|counts| &counts.commands_ordered_inside),
            unrun_instances_ended: counters.unrun_instances_ended.load(Ordering::Relaxed),
            untouched_passes_skipped: counters.untouched_passes_skipped.load(Ordering::Relaxed),
            inline_passes: counters.inline_passes.load(Ordering::Relaxed),
            encoders_barriered_at_open: counters.encoders_barriered_at_open.load(Ordering::Relaxed),
            encoders_folded: counters.encoders_folded.load(Ordering::Relaxed),
            folds_refused_kind: counters.folds_refused_kind.load(Ordering::Relaxed),
            folds_refused_tail: counters.folds_refused_tail.load(Ordering::Relaxed),
            folds_refused_batch: counters.folds_refused_batch.load(Ordering::Relaxed),
            folds_refused_state: counters.folds_refused_state.load(Ordering::Relaxed),
            folds_refused_parts: counters.folds_refused_parts.load(Ordering::Relaxed),
            folds_refused_ends: counters.folds_refused_ends.load(Ordering::Relaxed),
            fold_fence_barriers: counters.fold_fence_barriers.load(Ordering::Relaxed),
            parked_compute_taken: counters.parked_compute_taken.load(Ordering::Relaxed),
            record_lists_grown: counters.record_lists_grown.load(Ordering::Relaxed),
            record_storage_kept: counters.record_storage_kept.load(Ordering::Relaxed),
            command_buffer_states_made: counters.command_buffer_states_made.load(Ordering::Relaxed),
            kept_draws_carried: counters.kept_draws_carried.load(Ordering::Relaxed),
            kept_draws_carried_across: counters.kept_draws_carried_across.load(Ordering::Relaxed),
            kept_draws_carry_dropped_between: carry_dropped(CarryDropped::Between),
            kept_draws_carry_dropped_layout: carry_dropped(CarryDropped::Layout),
            kept_draws_carry_dropped_under: carry_dropped(CarryDropped::Under),
            kept_draws_carry_dropped_commit: carry_dropped(CarryDropped::Commit),
            kept_draws_carried_aside: counters.kept_draws_carried_aside.load(Ordering::Relaxed),
            kept_draws_uncarried_attached: (counters.kept_draws_uncarried_attached)
                .load(Ordering::Relaxed),
            encoders_uncarried_drawless: uncarried(Uncarried::Drawless),
            encoders_uncarried_none_kept: uncarried(Uncarried::NoneKept),
            encoders_uncarried_unsettled: uncarried(Uncarried::Unsettled),
            encoders_uncarried_split: uncarried(Uncarried::Split),
            encoders_uncarried_pass: uncarried(Uncarried::Pass),
            encoders_uncarried_replay: uncarried(Uncarried::Replay),
            encoders_uncarried_commands: uncarried(Uncarried::Commands),
            encoders_uncarried_shader_tests: uncarried(Uncarried::ShaderTests),
            encoders_uncarried_imageblock: uncarried(Uncarried::Imageblock),
            encoders_uncarried_visibility: uncarried(Uncarried::Visibility),
            encoders_uncarried_instance: uncarried(Uncarried::Instance),
            encoders_uncarried_unrebound: uncarried(Uncarried::Unrebound),
            encoders_uncarried_failing_state: uncarried(Uncarried::FailingState),
            encoders_none_kept_next_split: none_kept_next(NoneKeptNext::Split),
            encoders_none_kept_next_ends: none_kept_next(NoneKeptNext::Ends),
            encoders_none_kept_next_between: none_kept_next(NoneKeptNext::Between),
            encoders_none_kept_next_last: none_kept_next(NoneKeptNext::Last),
            encoders_none_kept_next_layout: none_kept_next(NoneKeptNext::Layout),
            encoders_none_kept_next_under: none_kept_next(NoneKeptNext::Under),
            encoders_none_kept_next_fits: none_kept_next(NoneKeptNext::Fits),
            encoders_last_draw_empty: counters.encoders_last_draw_empty.load(Ordering::Relaxed),
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
            render_passes_continued: counters.render_passes_continued.load(Ordering::Relaxed),
            instance_splits_barrier: split(Split::Barrier),
            instance_splits_tile_dispatch: split(Split::TileDispatch),
            instance_splits_tile_memory: split(Split::TileMemory),
            instance_splits_interlocked_writes: split(Split::InterlockedWrites),
            instance_splits_attached_read: split(Split::AttachedRead),
            instance_splits_feedback: split(Split::Feedback),
            instance_splits_memoryless_write: split(Split::MemorylessWrite),
            instance_splits_pending_pipeline: split(Split::PendingPipeline),
            instance_splits_execution: split(Split::Execution),
            instance_splits_visibility: split(Split::Visibility),
            instance_splits_shape: split(Split::Shape),
            rate_map_pieces: counters.rate_map_pieces.load(Ordering::Relaxed),
            rate_map_index_ranges: counters.rate_map_index_ranges.load(Ordering::Relaxed),
            rate_map_pieces_timed: counters.rate_map_pieces_timed.load(Ordering::Relaxed),
            rate_map_piece_gpu_nanos: counters.rate_map_piece_gpu_nanos.load(Ordering::Relaxed),
            rate_map_piece_gpu_nanos_most: (counters.rate_map_piece_gpu_nanos_most)
                .load(Ordering::Relaxed),
            rate_map_probes: counters.rate_map_probes.load(Ordering::Relaxed),
            rate_map_probe_gpu_nanos: counters.rate_map_probe_gpu_nanos.load(Ordering::Relaxed),
            rate_map_probe_wait_nanos: (counters.rate_map_probe_wait_nanos).load(Ordering::Relaxed),
            rate_map_probe_parts: counters.rate_map_probe_parts.load(Ordering::Relaxed),
            rate_map_steps_unfanned: (counters.rate_map_steps_unfanned).load(Ordering::Relaxed),
            rate_map_steps_nothing_to_probe: (counters.rate_map_steps_nothing_to_probe)
                .load(Ordering::Relaxed),
            rate_map_steps_probing_none: (counters.rate_map_steps_probing_none)
                .load(Ordering::Relaxed),
            rate_map_gpu_routes: counters.rate_map_gpu_routes.load(Ordering::Relaxed),
            rate_map_gpu_route_wait_nanos: (counters.rate_map_gpu_route_wait_nanos)
                .load(Ordering::Relaxed),
            rate_map_flagged_draws: counters.rate_map_flagged_draws.load(Ordering::Relaxed),
            rate_map_flagged_triangles: (counters.rate_map_flagged_triangles)
                .load(Ordering::Relaxed),
            rate_map_fanned_draws: counters.rate_map_fanned_draws.load(Ordering::Relaxed),
            rate_map_fanned_few_draws: (counters.rate_map_fanned_few_draws).load(Ordering::Relaxed),
            rate_map_parted_draws: (counters.rate_map_parted_draws).load(Ordering::Relaxed),
            rate_map_parted_region_triangles: (counters.rate_map_parted_region_triangles)
                .load(Ordering::Relaxed),
            rate_map_region_runs: counters.rate_map_region_runs.load(Ordering::Relaxed),
            rate_map_fan_out_pipelines_ready: (counters.rate_map_fan_out_pipelines_ready)
                .load(Ordering::Relaxed),
            rate_map_fan_out_pipelines_waited: (counters.rate_map_fan_out_pipelines_waited)
                .load(Ordering::Relaxed),
            rate_map_fan_tables_uploaded: (counters.rate_map_fan_tables_uploaded)
                .load(Ordering::Relaxed),
            rate_map_region_copy_draws: (counters.rate_map_region_copy_draws)
                .load(Ordering::Relaxed),
            rate_map_region_copy_recordings: (counters.rate_map_region_copy_recordings)
                .load(Ordering::Relaxed),
            rate_map_run_region_draws: (counters.rate_map_run_region_draws).load(Ordering::Relaxed),
            rate_map_runs_every_region: (counters.rate_map_runs_every_region)
                .load(Ordering::Relaxed),
            rate_map_straddling_triangles: (counters.rate_map_straddling_triangles)
                .load(Ordering::Relaxed),
            rate_map_whole_copy_vertex_regions: (counters.rate_map_whole_copy_vertex_regions)
                .load(Ordering::Relaxed),
            rate_map_own_copy_vertex_regions: (counters.rate_map_own_copy_vertex_regions)
                .load(Ordering::Relaxed),
            rate_map_run_vertex_regions: (counters.rate_map_run_vertex_regions)
                .load(Ordering::Relaxed),
            rate_map_fanned_vertices: (counters.rate_map_fanned_vertices).load(Ordering::Relaxed),
            rate_map_fanned_box_vertex_regions: (counters.rate_map_fanned_box_vertex_regions)
                .load(Ordering::Relaxed),
            rate_map_fanned_vertices_boxed_16: fanned_boxed[0],
            rate_map_fanned_vertices_boxed_64: fanned_boxed[1],
            rate_map_fanned_vertices_boxed_256: fanned_boxed[2],
            rate_map_boxed_draws: (counters.rate_map_boxed_draws).load(Ordering::Relaxed),
            rate_map_gpu_routed_draws: (counters.rate_map_gpu_routed_draws).load(Ordering::Relaxed),
            rate_map_gpu_fanned_draws: (counters.rate_map_gpu_fanned_draws).load(Ordering::Relaxed),
            rate_map_gpu_copied_draws: (counters.rate_map_gpu_copied_draws).load(Ordering::Relaxed),
            rate_map_boxed_vertex_regions: (counters.rate_map_boxed_vertex_regions)
                .load(Ordering::Relaxed),
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
            commits_taken_in: self.shared().scheduler.commits_taken_in(),
            images_started_on_the_host: counters.images_started_on_the_host.load(Ordering::Relaxed),
            images_started_zeroed: counters.images_started_zeroed.load(Ordering::Relaxed),
            private_textures_zero_filled: (counters.private_textures_zero_filled)
                .load(Ordering::Relaxed),
            private_textures_overwritten_unfilled: (counters.private_textures_overwritten_unfilled)
                .load(Ordering::Relaxed),
            memoryless_textures_zeroed: (counters.memoryless_textures_zeroed)
                .load(Ordering::Relaxed),
            host_texel_copies: counters.host_texel_copies.load(Ordering::Relaxed),
            heap_shadow_cpu_copies: counters.heap_shadow_cpu_copies.load(Ordering::Relaxed),
            depth_uploads_clamped: counters.depth_uploads_clamped.load(Ordering::Relaxed),
            texel_shadows_made: counters.texel_shadows_made.load(Ordering::Relaxed),
            texel_shadow_bytes: counters.texel_shadow_bytes.load(Ordering::Relaxed),
            texel_shadow_reads: counters.texel_shadow_reads.load(Ordering::Relaxed),
            texel_shadows_dropped: counters.texel_shadows_dropped.load(Ordering::Relaxed),
            staging_submissions: counters.staging_submissions.load(Ordering::Relaxed),
            copy_queue_batches: counters.copy_queue_batches.load(Ordering::Relaxed),
            staging_copies_queued: counters.staging_copies_queued.load(Ordering::Relaxed),
            staging_moves_queued: counters.staging_moves_queued.load(Ordering::Relaxed),
            staging_flushes: counters.staging_flushes.load(Ordering::Relaxed),
            staging_gpu_waits: counters.staging_gpu_waits.load(Ordering::Relaxed),
            staging_buffers_grown: counters.staging_buffers_grown.load(Ordering::Relaxed),
            staging_buffer_waits: counters.staging_buffer_waits.load(Ordering::Relaxed),
            fill_queue_submissions: counters.fill_queue_submissions.load(Ordering::Relaxed),
            fill_queue_waits: counters.fill_queue_waits.load(Ordering::Relaxed),
            fill_queue_waits_unused: counters.fill_queue_waits_unused.load(Ordering::Relaxed),
            first_work_waits: counters.first_work_waits.load(Ordering::Relaxed),
            first_work_waits_unused: counters.first_work_waits_unused.load(Ordering::Relaxed),
            uncached_host_images: counters.uncached_host_images.load(Ordering::Relaxed),
            cpu_block_decodes: counters.cpu_block_decodes.load(Ordering::Relaxed),
            block_decode_passes: counters.block_decode_passes.load(Ordering::Relaxed),
            shadow_twin_decodes: counters.shadow_twin_decodes.load(Ordering::Relaxed),
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
            draw_shapes_made: commands(|counts| &counts.draw_shapes_made),
            draw_shapes_found: commands(|counts| &counts.draw_shapes_found),
            draw_shape_compares: commands(|counts| &counts.draw_shape_compares),
            dispatch_shapes_made: commands(|counts| &counts.dispatch_shapes_made),
            dispatch_shapes_found: commands(|counts| &counts.dispatch_shapes_found),
            dispatch_used_resources_listed: commands(|counts| {
                &counts.dispatch_used_resources_listed
            }),
            resident_textures_looked_at: (counters.resident_textures_looked_at)
                .load(Ordering::Relaxed),
            resident_resources_joined: (counters.resident_resources_joined).load(Ordering::Relaxed),
            ended_residencies_looked_at: (counters.ended_residencies_looked_at)
                .load(Ordering::Relaxed),
            root_table_ranges_noted: (counters.root_table_ranges_noted).load(Ordering::Relaxed),
            resident_address_indexes_made: (counters.resident_address_indexes_made)
                .load(Ordering::Relaxed),
            resident_kept_keys_listed: (counters.resident_kept_keys_listed).load(Ordering::Relaxed),
            heap_tables_worked_out: counters.heap_tables_worked_out.load(Ordering::Relaxed),
            argument_samplers_taken: counters.argument_samplers_taken.load(Ordering::Relaxed),
            bind_views_worked_out: commands(|counts| &counts.bind_views_worked_out),
            bind_samplers_worked_out: commands(|counts| &counts.bind_samplers_worked_out),
            bind_spaces_read: commands(|counts| &counts.bind_spaces_read),
            stage_tables_read: commands(|counts| &counts.stage_tables_read),
            stage_tables_read_in_part: commands(|counts| &counts.stage_tables_read_in_part),
            stage_tables_read_whole_unread: whole_read(WholeRead::Unread),
            stage_tables_read_whole_moved: whole_read(WholeRead::Moved),
            stage_tables_read_whole_interface: whole_read(WholeRead::Interface),
            stage_tables_read_whole_maps: whole_read(WholeRead::Maps),
            stage_tables_read_whole_views: whole_read(WholeRead::Views),
            stage_tables_read_whole_spare: whole_read(WholeRead::Spare),
            stage_tables_read_whole_carried: whole_read(WholeRead::Carried),
            stage_tables_read_whole_unlisted: whole_read(WholeRead::Unlisted),
            no_copy_lies_read: commands(|counts| &counts.no_copy_lies_read),
            commands_noted_as_before: commands(|counts| &counts.commands_noted_as_before),
            uses_noted_covered: commands(|counts| &counts.uses_noted_covered),
            draws_settled: commands(|counts| &counts.draws_settled),
            two_plane_draws: commands(|counts| &counts.two_plane_draws),
            provoking_last_pipeline_draws: commands(|counts| &counts.provoking_last_pipeline_draws),
            provoking_listed_draws: commands(|counts| &counts.provoking_listed_draws),
            device_wide_line_draws: commands(|counts| &counts.device_wide_line_draws),
            line_band_draws: commands(|counts| &counts.line_band_draws),
            nearest_width_line_draws: commands(|counts| &counts.nearest_width_line_draws),
            far_vertex_fetch_pulled_draws: commands(|counts| &counts.far_vertex_fetch_pulled_draws),
            gathered_draws: commands(|counts| &counts.gathered_draws),
            bounded_run_draws: commands(|counts| &counts.bounded_run_draws),
            bound_notices_made: counters.bound_notices_made.load(Ordering::Relaxed),
            argument_tables_retained: commands(|counts| &counts.argument_tables_retained),
            argument_objects_retained: commands(|counts| &counts.argument_objects_retained),
            argument_tables_held_at_bind: commands(|counts| &counts.argument_tables_held_at_bind),
            bound_offsets_moved: commands(|counts| &counts.bound_offsets_moved),
            graphics_pipelines_looked_up: commands(|counts| &counts.graphics_pipelines_looked_up),
            linear_passes_in_place: counters.linear_passes_in_place.load(Ordering::Relaxed),
            linear_passes_through_copies: counters
                .linear_passes_through_copies
                .load(Ordering::Relaxed),
            linear_merge_scratches_made: counters
                .linear_merge_scratches_made
                .load(Ordering::Relaxed),
            linear_merge_scratches_reused: counters
                .linear_merge_scratches_reused
                .load(Ordering::Relaxed),
            linear_merges_recorded: counters.linear_merges_recorded.load(Ordering::Relaxed),
            linear_merges_taken_over: counters.linear_merges_taken_over.load(Ordering::Relaxed),
            linear_runs_reloaded: counters.linear_runs_reloaded.load(Ordering::Relaxed),
            kept_bytes_passes: counters.kept_bytes_passes.load(Ordering::Relaxed),
            kept_texels_passes: counters.kept_texels_passes.load(Ordering::Relaxed),
            color_group_passes: counters.color_group_passes.load(Ordering::Relaxed),
            color_group_twins: counters.color_group_twins.load(Ordering::Relaxed),
            color_group_draws_skipped: counters.color_group_draws_skipped.load(Ordering::Relaxed),
            draw_states_recorded: commands(|counts| &counts.draw_states_recorded),
            draws_pushing_set_zero: commands(|counts| &counts.draws_pushing_set_zero),
            set_zero_writes_pushed: commands(|counts| &counts.set_zero_writes_pushed),
            set_zero_equal_shape_switches: commands(|counts| &counts.set_zero_equal_shape_switches),
            set_zero_writes_on_equal_shape_switches: commands(|counts| {
                &counts.set_zero_writes_on_equal_shape_switches
            }),
            set_zero_writes_if_shapes_shared: commands(|counts| {
                &counts.set_zero_writes_if_shapes_shared
            }),
            dispatches_pushing_set_zero: commands(|counts| &counts.dispatches_pushing_set_zero),
            dispatch_set_zero_writes_pushed: commands(|counts| {
                &counts.dispatch_set_zero_writes_pushed
            }),
            dispatches_tracked: commands(|counts| &counts.dispatches_tracked),
            dispatch_uses_tracked: commands(|counts| &counts.dispatch_uses_tracked),
            set_zero_sets_reused: commands(|counts| &counts.set_zero_sets_reused),
            set_zero_sets_found: commands(|counts| &counts.set_zero_sets_found),
            draw_shapes_reused: commands(|counts| &counts.draw_shapes_reused),
            draw_shapes_carried: commands(|counts| &counts.draw_shapes_carried),
            draws_as_before: commands(|counts| &counts.draws_as_before),
            draws_rebound: commands(|counts| &counts.draws_rebound),
            attachment_reads_through_copies: commands(|counts| {
                &counts.attachment_reads_through_copies
            }),
            own_pixel_reads: commands(|counts| &counts.own_pixel_reads),
            draws_counted_past_early_masks: commands(|counts| {
                &counts.draws_counted_past_early_masks
            }),
            attached_read_slots: commands(|counts| &counts.attached_read_slots),
            attached_read_slots_one_level: attached(AttachedLevels::OneLevel),
            attached_read_slots_first_level: attached(AttachedLevels::FirstLevel),
            attached_read_slots_any_level: attached(AttachedLevels::AnyLevel),
            attached_read_slots_reaching: attached(AttachedLevels::Reaching),
            attached_read_slots_apart: attached(AttachedLevels::Apart),
            attached_read_slots_written: attached(AttachedLevels::Written),
            attached_read_slots_narrowed: attached(AttachedLevels::Narrowed),
            attached_read_slots_past: attached(AttachedLevels::Past),
            texture_faces_made: counters.texture_faces_made.load(Ordering::Relaxed),
            attached_read_swaps: counters.attached_read_swaps.load(Ordering::Relaxed),
            attached_read_swap_reads: counters.attached_read_swap_reads.load(Ordering::Relaxed),
            attached_read_swap_texels: counters.attached_read_swap_texels.load(Ordering::Relaxed),
            attached_read_swap_fixes: counters.attached_read_swap_fixes.load(Ordering::Relaxed),
            attached_read_swap_fix_texels: (counters.attached_read_swap_fix_texels)
                .load(Ordering::Relaxed),
            attached_read_swaps_short_area: (counters.attached_read_swaps_short_area)
                .load(Ordering::Relaxed),
            attached_read_swap_area_texels: (counters.attached_read_swap_area_texels)
                .load(Ordering::Relaxed),
            attached_read_unswapped_load: unswapped(Unturned::Load).0,
            attached_read_unswapped_levels: unswapped(Unturned::Levels).0,
            attached_read_unswapped_area: area.0,
            attached_read_unswapped_area_extent: unswapped(Unturned::Extent).0,
            attached_read_unswapped_area_resolve: unswapped(Unturned::Resolve).0,
            attached_read_unswapped_area_shared: unswapped(Unturned::Shared).0,
            attached_read_unswapped_area_memoryless: unswapped(Unturned::Memoryless).0,
            attached_read_unswapped_area_failing: unswapped(Unturned::Failing).0,
            attached_read_unswapped_storage: unswapped(Unturned::Storage).0,
            attached_read_unswapped_frozen: unswapped(Unturned::Frozen).0,
            attached_read_unswapped_first: unswapped(Unturned::First).0,
            attached_read_unswapped_busy: unswapped(Unturned::Busy).0,
            attached_read_unswapped_pass: unswapped(Unturned::Pass).0,
            attached_read_unswapped_split: unswapped(Unturned::Split).0,
            attached_read_unswapped_texels_load: unswapped(Unturned::Load).1,
            attached_read_unswapped_texels_levels: unswapped(Unturned::Levels).1,
            attached_read_unswapped_texels_area: area.1,
            attached_read_unswapped_texels_area_extent: unswapped(Unturned::Extent).1,
            attached_read_unswapped_texels_area_resolve: unswapped(Unturned::Resolve).1,
            attached_read_unswapped_texels_area_shared: unswapped(Unturned::Shared).1,
            attached_read_unswapped_texels_area_memoryless: unswapped(Unturned::Memoryless).1,
            attached_read_unswapped_texels_area_failing: unswapped(Unturned::Failing).1,
            attached_read_unswapped_texels_storage: unswapped(Unturned::Storage).1,
            attached_read_unswapped_texels_frozen: unswapped(Unturned::Frozen).1,
            attached_read_unswapped_texels_first: unswapped(Unturned::First).1,
            attached_read_unswapped_texels_busy: unswapped(Unturned::Busy).1,
            attached_read_unswapped_texels_pass: unswapped(Unturned::Pass).1,
            attached_read_unswapped_texels_split: unswapped(Unturned::Split).1,
            draws_moved: commands(|counts| &counts.draws_moved),
            draws_moved_inside: commands(|counts| &counts.draws_moved_inside),
            draws_restated: commands(|counts| &counts.draws_restated),
            draws_restated_scissors: commands(|counts| &counts.draws_restated_scissors),
            kept_draws_rehomed: commands(|counts| &counts.kept_draws_rehomed),
            kept_draws_rehomed_carried: commands(|counts| &counts.kept_draws_rehomed_carried),
            kept_draws_carried_unopened: commands(|counts| &counts.kept_draws_carried_unopened),
            draws_rebound_arguments: commands(|counts| &counts.draws_rebound_arguments),
            draws_rebound_carried: commands(|counts| &counts.draws_rebound_carried),
            draws_none_kept: commands(|counts| &counts.draws_none_kept),
            draws_none_kept_uncarried: commands(|counts| &counts.draws_none_kept_uncarried),
            draws_carry_stopped: commands(|counts| &counts.draws_carry_stopped),
            draws_carry_stopped_stale: carry_stopped(CarryStop::Stale),
            draws_carry_stopped_attached: carry_stopped(CarryStop::Attached),
            draws_carry_stopped_other: carry_stopped(CarryStop::Other),
            draws_carry_stopped_snapshot: carry_stopped(CarryStop::Snapshot),
            kept_draws_let_go_fence: let_go(LoggedCall::Fence),
            kept_draws_let_go_use: let_go(LoggedCall::Use),
            kept_draws_let_go_barrier: let_go(LoggedCall::Barrier),
            kept_draws_let_go_store_action: let_go(LoggedCall::StoreAction),
            kept_draws_let_go_visibility: let_go(LoggedCall::Visibility),
            kept_draws_let_go_other: let_go(LoggedCall::Other),
            kept_draws_kept_across_fence: kept_across(LoggedCall::Fence),
            kept_draws_kept_across_use: kept_across(LoggedCall::Use),
            kept_draws_kept_across_barrier: kept_across(LoggedCall::Barrier),
            draws_unkept: commands(|counts| &counts.draws_unkept),
            draws_unkept_instance: commands(|counts| &counts.draws_unkept_instance),
            draws_unkept_instance_later: commands(|counts| &counts.draws_unkept_instance_later),
            draws_unkept_instance_steps: commands(|counts| &counts.draws_unkept_instance_steps),
            draws_unkept_barrier: commands(|counts| &counts.draws_unkept_barrier),
            draws_unkept_shadowed: commands(|counts| &counts.draws_unkept_shadowed),
            draws_unkept_tables: commands(|counts| &counts.draws_unkept_tables),
            draws_unkept_tables_heap: commands(|counts| &counts.draws_unkept_tables_heap),
            draws_unkept_tables_writing: commands(|counts| &counts.draws_unkept_tables_writing),
            draws_unkept_index_copied: commands(|counts| &counts.draws_unkept_index_copied),
            draws_unkept_writing: commands(|counts| &counts.draws_unkept_writing),
            draws_unkept_writing_resident: commands(|counts| &counts.draws_unkept_writing_resident),
            draws_unkept_making: commands(|counts| &counts.draws_unkept_making),
            draws_unkept_other: commands(|counts| &counts.draws_unkept_other),
            draws_unlike_kept: commands(|counts| &counts.draws_unlike_kept),
            draws_unlike_kept_pipeline: commands(|counts| &counts.draws_unlike_kept_pipeline),
            draws_unlike_kept_call: commands(|counts| &counts.draws_unlike_kept_call),
            draws_unlike_kept_counts: commands(|counts| &counts.draws_unlike_kept_counts),
            draws_unlike_kept_index_buffer: commands(|counts| {
                &counts.draws_unlike_kept_index_buffer
            }),
            draws_unlike_kept_index_offset: commands(|counts| {
                &counts.draws_unlike_kept_index_offset
            }),
            draws_unlike_kept_tessellation: commands(|counts| {
                &counts.draws_unlike_kept_tessellation
            }),
            draws_unlike_kept_command_buffer: commands(|counts| {
                &counts.draws_unlike_kept_command_buffer
            }),
            draws_unlike_kept_under: commands(|counts| &counts.draws_unlike_kept_under),
            draws_unlike_kept_arguments_alone: commands(|counts| {
                &counts.draws_unlike_kept_arguments_alone
            }),
            draws_rebind_stopped: commands(|counts| &counts.draws_rebind_stopped),
            draws_rebind_stopped_arguments: commands(|counts| {
                &counts.draws_rebind_stopped_arguments
            }),
            draws_rebind_stopped_arguments_route: arguments_stopped(ArgumentsStop::Route),
            draws_rebind_stopped_arguments_unkept: arguments_stopped(ArgumentsStop::Unkept),
            draws_rebind_stopped_arguments_restated: arguments_stopped(ArgumentsStop::Restated),
            draws_rebind_stopped_arguments_stale: arguments_stopped(ArgumentsStop::Stale),
            draws_rebind_stopped_arguments_unread: arguments_stopped(ArgumentsStop::Unread),
            draws_rebind_stopped_arguments_unlisted: arguments_stopped(ArgumentsStop::Unlisted),
            draws_rebind_stopped_arguments_slots: arguments_stopped(ArgumentsStop::Slots),
            draws_rebind_stopped_arguments_images: arguments_stopped(ArgumentsStop::Images),
            draws_rebind_stopped_arguments_reshaped: arguments_stopped(ArgumentsStop::Reshaped),
            draws_rebind_reshaped_views: reshaped(ReshapedBy::Views),
            draws_rebind_reshaped_buffer_bound: reshaped(ReshapedBy::BufferBound),
            draws_rebind_reshaped_buffer_unbound: reshaped(ReshapedBy::BufferUnbound),
            draws_rebind_reshaped_buffer_shape: reshaped(ReshapedBy::BufferShape),
            draws_rebind_reshaped_buffer_offset: reshaped(ReshapedBy::BufferOffset),
            draws_rebind_reshaped_texture_bound: reshaped(ReshapedBy::TextureBound),
            draws_rebind_reshaped_texture_unbound: reshaped(ReshapedBy::TextureUnbound),
            draws_rebind_reshaped_texture_shape: reshaped(ReshapedBy::TextureShape),
            draws_rebind_reshaped_sampler_bound: reshaped(ReshapedBy::SamplerBound),
            draws_rebind_reshaped_sampler_unbound: reshaped(ReshapedBy::SamplerUnbound),
            draws_rebind_reshaped_sampler_shape: reshaped(ReshapedBy::SamplerShape),
            draws_rebind_stopped_arguments_unsettled: arguments_stopped(ArgumentsStop::Unsettled),
            draws_rebind_stopped_arguments_tables: arguments_stopped(ArgumentsStop::Tables),
            draws_rebind_stopped_arguments_fetch_window: arguments_stopped(
                ArgumentsStop::FetchWindow,
            ),
            draws_rebind_stopped_arguments_reach: arguments_stopped(ArgumentsStop::Reach),
            draws_rebind_stopped_arguments_checks: arguments_stopped(ArgumentsStop::Checks),
            draws_rebind_stopped_arguments_narrowed: arguments_stopped(ArgumentsStop::Narrowed),
            draws_rebind_stopped_arguments_worked: arguments_stopped(ArgumentsStop::Worked),
            draws_rebind_stopped_arguments_vertex_buffers: arguments_stopped(
                ArgumentsStop::VertexBuffers,
            ),
            draws_rebind_stopped_arguments_shadows: arguments_stopped(ArgumentsStop::Shadows),
            draws_rebind_stopped_arguments_barrier: arguments_stopped(ArgumentsStop::Barrier),
            draws_rebind_stopped_arguments_detached: arguments_stopped(ArgumentsStop::Detached),
            draws_rebind_stopped_arguments_other: arguments_stopped(ArgumentsStop::Other),
            resident_uses_made: commands(|counts| &counts.resident_uses_made),
            resident_keys_settled: commands(|counts| &counts.resident_keys_settled),
            resident_uses_gathered: commands(|counts| &counts.resident_uses_gathered),
            resident_uses_against_fragments: commands(|counts| {
                &counts.resident_uses_against_fragments
            }),
            resident_uses_against_attachments: commands(|counts| {
                &counts.resident_uses_against_attachments
            }),
            resident_uses_noted: commands(|counts| &counts.resident_uses_noted),
            resident_uses_unordered_writes: commands(|counts| {
                &counts.resident_uses_unordered_writes
            }),
            resident_uses_indirect_reads: commands(|counts| &counts.resident_uses_indirect_reads),
            resident_uses_geometry_writes: commands(|counts| &counts.resident_uses_geometry_writes),
            resident_uses_geometry_checked: commands(|counts| {
                &counts.resident_uses_geometry_checked
            }),
            resident_uses_again_checked: commands(|counts| &counts.resident_uses_again_checked),
            draw_heap_parts_made: commands(|counts| &counts.draw_heap_parts_made),
            draw_resident_textures_looked_at: commands(|counts| {
                &counts.draw_resident_textures_looked_at
            }),
            host_holds_looked_up: commands(|counts| &counts.host_holds_looked_up),
            dispatches_settled_as_before: commands(|counts| &counts.dispatches_settled_as_before),
            dispatches_settled_across_encoders: commands(|counts| {
                &counts.dispatches_settled_across_encoders
            }),
            dispatches_instantiated: commands(|counts| &counts.dispatches_instantiated),
            dispatches_instantiated_in_place: commands(|counts| {
                &counts.dispatches_instantiated_in_place
            }),
            dispatches_instantiated_as_before: commands(|counts| {
                &counts.dispatches_instantiated_as_before
            }),
            dispatches_from_set_aside: commands(|counts| &counts.dispatches_from_set_aside),
            set_aside_commands_kept: commands(|counts| &counts.set_aside_commands_kept),
            set_aside_commands_instantiated: commands(|counts| {
                &counts.set_aside_commands_instantiated
            }),
            dispatches_skipped_inert: commands(|counts| &counts.dispatches_skipped_inert),
            dispatches_planned_as_before: commands(|counts| &counts.dispatches_planned_as_before),
            dispatches_planned_across_grids: commands(|counts| {
                &counts.dispatches_planned_across_grids
            }),
            whole_dispatches_on_partial_variants: commands(|counts| {
                &counts.whole_dispatches_on_partial_variants
            }),
            baked_sampler_dispatches_apart: commands(|counts| {
                &counts.baked_sampler_dispatches_apart
            }),
            host_deallocators_delayed: counters.host_deallocators_delayed.load(Ordering::Relaxed),
            host_import_releases_posted: (counters.host_import_releases_posted)
                .load(Ordering::Relaxed),
            host_import_releases_pending: (counters.host_import_releases_pending)
                .load(Ordering::Acquire),
            host_import_frees_posted: self.shared().frees.imports_posted(),
            host_keep_alives_kept: counters.host_keep_alives_kept.load(Ordering::Relaxed),
            host_keep_alives_covered: counters.host_keep_alives_covered.load(Ordering::Relaxed),
            host_releases_held_inside: (counters.host_releases_held_inside).load(Ordering::Relaxed),
            host_mappings_queried: counters.host_mappings_queried.load(Ordering::Relaxed),
            dma_buf_imports: counters.dma_buf_imports.load(Ordering::Relaxed),
            dma_buf_buffer_imports_dedicated: (counters.dma_buf_buffer_imports_dedicated)
                .load(Ordering::Relaxed),
            dma_buf_memories_live: counters.dma_buf_memories_live.load(Ordering::Relaxed),
            dma_buf_image_memories_live: counters
                .dma_buf_image_memories_live
                .load(Ordering::Relaxed),
            dma_buf_pieces_joined: counters.dma_buf_pieces_joined.load(Ordering::Relaxed),
            dma_buf_shared_imports_live: (counters.dma_buf_shared_imports_live)
                .load(Ordering::Relaxed),
            dma_buf_shared_imports_kept: (counters.dma_buf_shared_imports_kept)
                .load(Ordering::Relaxed),
            dma_buf_import_calls: counters.dma_buf_import_calls.load(Ordering::Relaxed),
            dma_buf_import_nanos: counters.dma_buf_import_nanos.load(Ordering::Relaxed),
            dma_buf_join_nanos: counters.dma_buf_join_nanos.load(Ordering::Relaxed),
            host_imports_released: counters.host_imports_released.load(Ordering::Relaxed),
            host_imports_released_over_most: (counters.host_imports_released_over_most)
                .load(Ordering::Relaxed),
            host_imports_reimported: counters.host_imports_reimported.load(Ordering::Relaxed),
            host_imports_joined_releasable: (counters.host_imports_joined_releasable)
                .load(Ordering::Relaxed),
            host_imports_resident: counters.host_imports_resident.load(Ordering::Relaxed),
            host_imports_held_for_good: (counters.host_imports_held_for_good)
                .load(Ordering::Relaxed),
            host_imports_used: counters.host_imports_used.load(Ordering::Relaxed),
            host_imports_read_apart: counters.host_imports_read_apart.load(Ordering::Relaxed),
            host_imports_read_apart_address: read_apart(ReadApart::Address),
            host_imports_read_apart_vk_buffer: read_apart(ReadApart::VkBuffer),
            host_imports_read_apart_binding: read_apart(ReadApart::Binding),
            host_imports_read_apart_texture: read_apart(ReadApart::Texture),
            host_imports_read_apart_inside: read_apart(ReadApart::Inside),
            host_imports_read_apart_shared_range: read_apart(ReadApart::SharedRange),
            host_imports_read_apart_heap: read_apart(ReadApart::Heap),
            host_imports_read_apart_rebound: read_apart(ReadApart::Rebound),
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
            host_imports_unpinnable: counters.host_imports_unpinnable.load(Ordering::Relaxed),
            host_imports_refused_by_host: counters
                .host_imports_refused_by_host
                .load(Ordering::Relaxed),
            copied_host_buffers_made: counters.copied_host_buffers_made.load(Ordering::Relaxed),
            copied_host_buffers_refused: counters
                .copied_host_buffers_refused
                .load(Ordering::Relaxed),
            copied_host_buffers_device_refused: counters
                .copied_host_buffers_device_refused
                .load(Ordering::Relaxed),
            copied_host_buffers_degraded: counters
                .copied_host_buffers_degraded
                .load(Ordering::Relaxed),
            copied_rebind_pages_moved: counters.copied_rebind_pages_moved.load(Ordering::Relaxed),
            copied_read_only_memory: counters.copied_read_only_memory.load(Ordering::Relaxed),
            copied_unwritable_pages: counters.copied_unwritable_pages.load(Ordering::Relaxed),
            copied_over_live_imports: counters.copied_over_live_imports.load(Ordering::Relaxed),
            copied_rebind_pages_made: counters.copied_rebind_pages_made.load(Ordering::Relaxed),
            copied_rebind_images_made: counters.copied_rebind_images_made.load(Ordering::Relaxed),
            copied_heaps_made: counters.copied_heaps_made.load(Ordering::Relaxed),
            copied_host_buffers_shared: (counters.copied_host_buffers_shared)
                .load(Ordering::Relaxed),
            copied_copy_in_bytes: counters.copied_copy_in_bytes.load(Ordering::Relaxed),
            copied_copy_in_bytes_skipped: (counters.copied_copy_in_bytes_skipped)
                .load(Ordering::Relaxed),
            copied_copy_ins_unused: counters.copied_copy_ins_unused.load(Ordering::Relaxed),
            copied_copy_ins_after_waits: (counters.copied_copy_ins_after_waits)
                .load(Ordering::Relaxed),
            copied_merge_back_bytes: counters.copied_merge_back_bytes.load(Ordering::Relaxed),
            copied_merges_compared: counters.copied_merges_compared.load(Ordering::Relaxed),
            copied_merges_skipped: counters.copied_merges_skipped.load(Ordering::Relaxed),
            copied_dropped_bytes: counters.copied_dropped_bytes.load(Ordering::Relaxed),
            copied_bases_made: counters.copied_bases_made.load(Ordering::Relaxed),
            copied_bases_refused: counters.copied_bases_refused.load(Ordering::Relaxed),
            copied_base_blocks_read: counters.copied_base_blocks_read.load(Ordering::Relaxed),
            copied_base_blocks_read_back: (counters.copied_base_blocks_read_back)
                .load(Ordering::Relaxed),
            copied_compare_threads: counters.copied_compare_threads.load(Ordering::Relaxed),
            copied_copy_ins_compared: counters.copied_copy_ins_compared.load(Ordering::Relaxed),
            copied_copy_in_bytes_compared: (counters.copied_copy_in_bytes_compared)
                .load(Ordering::Relaxed),
            copied_unnamed_reach_copy_ins: (counters.copied_unnamed_reach_copy_ins)
                .load(Ordering::Relaxed),
            copied_unnamed_reach_bytes_compared: (counters.copied_unnamed_reach_bytes_compared)
                .load(Ordering::Relaxed),
            copied_unnamed_reach_twin_copy_ins: (counters.copied_unnamed_reach_twin_copy_ins)
                .load(Ordering::Relaxed),
            copied_unnamed_reach_twin_bytes_compared: (counters
                .copied_unnamed_reach_twin_bytes_compared)
                .load(Ordering::Relaxed),
            read_twins_made: counters.read_twins_made.load(Ordering::Relaxed),
            read_twins_unimported: counters.read_twins_unimported.load(Ordering::Relaxed),
            read_twins_uncontained: counters.read_twins_uncontained.load(Ordering::Relaxed),
            read_twin_blit_reads: counters.read_twin_blit_reads.load(Ordering::Relaxed),
            read_twin_reads_after_writes: (counters.read_twin_reads_after_writes)
                .load(Ordering::Relaxed),
            read_twin_writes_refused: counters.read_twin_writes_refused.load(Ordering::Relaxed),
            read_twin_textures_made: counters.read_twin_textures_made.load(Ordering::Relaxed),
            read_twin_textures_not_in_place: (counters.read_twin_textures_not_in_place)
                .load(Ordering::Relaxed),
            read_twin_texture_shadows: counters.read_twin_texture_shadows.load(Ordering::Relaxed),
            read_twin_slot_reads: counters.read_twin_slot_reads.load(Ordering::Relaxed),
            copied_base_host_bytes,
            copied_base_device_bytes,
            copied_base_host_blocks: self.shared().copied.base_host_blocks(),
            host_ram_copies: counters.host_ram_copies.load(Ordering::Relaxed),
            host_ram_copy_bytes: counters.host_ram_copy_bytes.load(Ordering::Relaxed),
            host_ram_copy_dispatches: counters.host_ram_copy_dispatches.load(Ordering::Relaxed),
            host_ram_copy_regions: counters.host_ram_copy_regions.load(Ordering::Relaxed),
            host_ram_chunks_imported: counters.host_ram_chunks_imported.load(Ordering::Relaxed),
            host_ram_chunk_import_nanos: counters
                .host_ram_chunk_import_nanos
                .load(Ordering::Relaxed),
            host_ram_copy_import_waits: counters.host_ram_copy_import_waits.load(Ordering::Relaxed),
            host_ram_copy_import_wait_nanos: counters
                .host_ram_copy_import_wait_nanos
                .load(Ordering::Relaxed),
            host_ram_copies_staged: counters.host_ram_copies_staged.load(Ordering::Relaxed),
            host_ram_copy_plans_reused: counters.host_ram_copy_plans_reused.load(Ordering::Relaxed),
            host_ram_copy_tables_kept: counters.host_ram_copy_tables_kept.load(Ordering::Relaxed),
            host_ram_staged_fills_gated: counters
                .host_ram_staged_fills_gated
                .load(Ordering::Relaxed),
            host_ram_staged_fill_nanos: counters.host_ram_staged_fill_nanos.load(Ordering::Relaxed),
            host_ram_scatters_staged: counters.host_ram_scatters_staged.load(Ordering::Relaxed),
            host_ram_staged_drain_nanos: counters
                .host_ram_staged_drain_nanos
                .load(Ordering::Relaxed),
            host_ram_dma_buf_imports: counters.host_ram_dma_buf_imports.load(Ordering::Relaxed),
            host_ram_run_list_imports: counters.host_ram_run_list_imports.load(Ordering::Relaxed),
            host_ram_run_list_import_nanos: counters
                .host_ram_run_list_import_nanos
                .load(Ordering::Relaxed),
            host_ram_pointer_imports_failed: counters
                .host_ram_pointer_imports_failed
                .load(Ordering::Relaxed),
            host_ram_dma_buf_imports_failed: counters
                .host_ram_dma_buf_imports_failed
                .load(Ordering::Relaxed),
            host_ram_chunks_imported_ahead: counters
                .host_ram_chunks_imported_ahead
                .load(Ordering::Relaxed),
            host_ram_chunks_held_back: counters.host_ram_chunks_held_back.load(Ordering::Relaxed),
            host_ram_chunks_faulted_ahead: counters
                .host_ram_chunks_faulted_ahead
                .load(Ordering::Relaxed),
            host_ram_chunks_given_back: counters.host_ram_chunks_given_back.load(Ordering::Relaxed),
            host_ram_chunks_released: counters.host_ram_chunks_released.load(Ordering::Relaxed),
            memory_allocations_refused_at_limit: u64::from(
                self.shared().allocations.allocations.refused(),
            ),
            memory_frees_posted: frees_posted,
            free_thread_queue_full: frees_queue_full,
            memory_free_waits: free_waits,
            memory_frees_paced: counters.memory_frees_paced.load(Ordering::Relaxed) + frees_paced,
            memory_frees_paced_nanos: counters.memory_frees_paced_nanos.load(Ordering::Relaxed)
                + frees_paced_nanos,
            host_ram_chunk_bytes: counters.host_ram_chunk_bytes.load(Ordering::Relaxed),
            host_ram_chunks_live: counters.host_ram_chunks_live.load(Ordering::Relaxed),
            chunk_bound_buffers_made: counters.chunk_bound_buffers_made.load(Ordering::Relaxed),
            run_imports_made: counters.run_imports_made.load(Ordering::Relaxed),
            run_imports_live: counters.run_imports_live.load(Ordering::Relaxed),
            host_imports_lost: counters.host_imports_lost.load(Ordering::Relaxed),
            host_import_fds_kept: counters.host_import_fds_kept.load(Ordering::Relaxed),
            host_imports_kept_for_fds: counters.host_imports_kept_for_fds.load(Ordering::Relaxed),
            dma_buf_sparse_buffers: counters.dma_buf_sparse_buffers.load(Ordering::Relaxed),
            dma_buf_sparse_binds: counters.dma_buf_sparse_binds.load(Ordering::Relaxed),
            dma_buf_images: counters.dma_buf_images.load(Ordering::Relaxed),
            dma_buf_images_at_offset: counters.dma_buf_images_at_offset.load(Ordering::Relaxed),
            dma_buf_images_parked: counters.dma_buf_images_parked.load(Ordering::Relaxed),
            dma_buf_images_taken: counters.dma_buf_images_taken.load(Ordering::Relaxed),
            dma_buf_images_missed: counters.dma_buf_images_missed.load(Ordering::Relaxed),
            dma_buf_images_parked_released: (counters.dma_buf_images_parked_released)
                .load(Ordering::Relaxed),
            dma_buf_images_in_import: counters.dma_buf_images_in_import.load(Ordering::Relaxed),
            exported_dma_buf_images: counters.exported_dma_buf_images.load(Ordering::Relaxed),
            exported_dma_buf_images_parked: (counters.exported_dma_buf_images_parked)
                .load(Ordering::Relaxed),
            exported_dma_buf_images_taken: (counters.exported_dma_buf_images_taken)
                .load(Ordering::Relaxed),
            exported_dma_buf_images_parked_released: (counters
                .exported_dma_buf_images_parked_released)
                .load(Ordering::Relaxed),
            dma_buf_image_fallbacks: counters.dma_buf_image_fallbacks.load(Ordering::Relaxed),
            modifier_feature_queries: counters.modifier_feature_queries.load(Ordering::Relaxed),
            linear_alignments_probed: counters.linear_alignments_probed.load(Ordering::Relaxed),
            linear_alignments_recalled: counters.linear_alignments_recalled.load(Ordering::Relaxed),
            shadowed_textures: counters.shadowed_textures.load(Ordering::Relaxed),
            shadows_over_fresh_pages: counters.shadows_over_fresh_pages.load(Ordering::Relaxed),
            shadows_left_unmade: counters.shadows_left_unmade.load(Ordering::Relaxed),
            shadows_made_at_use: counters.shadows_made_at_use.load(Ordering::Relaxed),
            host_textures_in_place: counters.host_textures_in_place.load(Ordering::Relaxed),
            host_textures_in_place_written: counters
                .host_textures_in_place_written
                .load(Ordering::Relaxed),
            device_copy_textures: counters.device_copy_textures.load(Ordering::Relaxed),
            managed_device_copies: counters.managed_device_copies.load(Ordering::Relaxed),
            managed_device_copies_made: counters.managed_device_copies_made.load(Ordering::Relaxed),
            managed_device_copy_bytes: counters.managed_device_copy_bytes.load(Ordering::Relaxed),
            managed_device_copies_in_place: counters
                .managed_device_copies_in_place
                .load(Ordering::Relaxed),
            managed_device_copies_zeroed: counters
                .managed_device_copies_zeroed
                .load(Ordering::Relaxed),
            managed_bytes_zeroed: counters.managed_bytes_zeroed.load(Ordering::Relaxed),
            managed_bytes_copied_in: counters.managed_bytes_copied_in.load(Ordering::Relaxed),
            managed_bytes_copied_out: counters.managed_bytes_copied_out.load(Ordering::Relaxed),
            managed_ranges_modified: counters.managed_ranges_modified.load(Ordering::Relaxed),
            managed_bytes_modified: counters.managed_bytes_modified.load(Ordering::Relaxed),
            managed_resources_synchronized: (counters.managed_resources_synchronized)
                .load(Ordering::Relaxed),
            managed_bytes_synchronized: counters.managed_bytes_synchronized.load(Ordering::Relaxed),
            managed_synchronizes_unwritten: (counters.managed_synchronizes_unwritten)
                .load(Ordering::Relaxed),
            managed_synchronizes_whole: (counters.managed_synchronizes_whole)
                .load(Ordering::Relaxed),
            managed_synchronizes_topped_up: (counters.managed_synchronizes_topped_up)
                .load(Ordering::Relaxed),
            managed_bytes_unwritten_skipped: (counters.managed_bytes_unwritten_skipped)
                .load(Ordering::Relaxed),
            managed_staging_bytes: counters.managed_staging_bytes.load(Ordering::Relaxed),
            managed_writes_undone: counters.managed_writes_undone.load(Ordering::Relaxed),
            managed_bytes_staged_in: counters.managed_bytes_staged_in.load(Ordering::Relaxed),
            managed_bytes_copied_in_from_ram: (counters.managed_bytes_copied_in_from_ram)
                .load(Ordering::Relaxed),
            managed_ranges_inlined: counters.managed_ranges_inlined.load(Ordering::Relaxed),
            managed_uploads_unbarriered: (counters.managed_uploads_unbarriered)
                .load(Ordering::Relaxed),
            managed_upload_barriers_narrowed: (counters.managed_upload_barriers_narrowed)
                .load(Ordering::Relaxed),
            managed_pages_alone_refused: (counters.managed_pages_alone_refused)
                .load(Ordering::Relaxed),
            managed_pages_alone_over_imports: (counters.managed_pages_alone_over_imports)
                .load(Ordering::Relaxed),
            managed_pages_alone_unwritable: (counters.managed_pages_alone_unwritable)
                .load(Ordering::Relaxed),
            managed_pages_in_ram_chunks: (counters.managed_pages_in_ram_chunks)
                .load(Ordering::Relaxed),
            dma_buf_textures_in_place: counters.dma_buf_textures_in_place.load(Ordering::Relaxed),
            fresh_page_textures_in_place: counters
                .fresh_page_textures_in_place
                .load(Ordering::Relaxed),
            copied_textures_in_place: counters.copied_textures_in_place.load(Ordering::Relaxed),
            pitched_host_images: counters.pitched_host_images.load(Ordering::Relaxed),
            host_texture_shadows_exact: counters.host_texture_shadows_exact.load(Ordering::Relaxed),
            host_texture_shadows_render_target: counters
                .host_texture_shadows_render_target
                .load(Ordering::Relaxed),
            host_texture_shadows_shader_write: counters
                .host_texture_shadows_shader_write
                .load(Ordering::Relaxed),
            host_texture_shadows_store_width: counters
                .host_texture_shadows_store_width
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
            host_textures_moved_in_place: (counters.host_textures_moved_in_place)
                .load(Ordering::Relaxed),
            host_texture_shadows_freed_after_move: (counters.host_texture_shadows_freed_after_move)
                .load(Ordering::Relaxed),
            fresh_page_buffers_made: counters.fresh_page_buffers_made.load(Ordering::Relaxed),
            fresh_page_bytes: counters.fresh_page_bytes.load(Ordering::Relaxed),
            new_buffer_bytes_cleared: counters.new_buffer_bytes_cleared.load(Ordering::Relaxed),
            new_buffer_zero_fills: counters.new_buffer_zero_fills.load(Ordering::Relaxed),
            views_through_swizzle: counters.views_through_swizzle.load(Ordering::Relaxed),
            alpha_textures_held_in_red: (counters.alpha_textures_held_in_red)
                .load(Ordering::Relaxed),
            view_shadows: counters.view_shadows.load(Ordering::Relaxed),
            attachment_read_splits: counters.attachment_read_splits.load(Ordering::Relaxed),
            attachment_read_splits_spared: (counters.attachment_read_splits_spared)
                .load(Ordering::Relaxed),
            attachment_read_clears: counters.attachment_read_clears.load(Ordering::Relaxed),
            fetch_draw_barriers: counters.fetch_draw_barriers.load(Ordering::Relaxed),
            reading_draws_snapshotted: counters.reading_draws_snapshotted.load(Ordering::Relaxed),
            reading_draws_snapshotted_leading: (counters.reading_draws_snapshotted_leading)
                .load(Ordering::Relaxed),
            reading_draws_snapshotted_again: commands(|c| &c.reading_draws_snapshotted_again),
            reading_draws_snapshotted_stopped: commands(|c| &c.reading_draws_snapshotted_stopped),
            reading_draws_in_place: counters.reading_draws_in_place.load(Ordering::Relaxed),
            snapshot_inline_splits: counters.snapshot_inline_splits.load(Ordering::Relaxed),
            reading_draws_split_unreadable_own_pixel: split_by(SplitReason::UnreadableOwnPixel),
            reading_draws_split_not_in_place: split_by(SplitReason::NotInPlace),
            reading_draws_split_device_lacks: split_by(SplitReason::DeviceLacks),
            reading_draws_split_visibility_query: split_by(SplitReason::VisibilityQuery),
            reading_draws_split_pass_shape: split_by(SplitReason::PassShape),
            reading_draws_split_other_format: split_by(SplitReason::OtherFormat),
            reading_draws_split_stand_in: split_by(SplitReason::StandIn),
            reading_draws_split_one_image: split_by(SplitReason::OneImage),
            reading_draws_split_coverage: split_by(SplitReason::Coverage),
            reading_draws_split_area: split_by(SplitReason::Area),
            reading_draws_split_variant_refused: split_by(SplitReason::VariantRefused),
            reading_draws_split_slot_left_out: split_by(SplitReason::SlotLeftOut),
            reading_draws_split_views: split_by(SplitReason::Views),
            reading_draws_split_storage_binding: split_by(SplitReason::StorageBinding),
            reading_draws_split_other_attachment: split_by(SplitReason::OtherAttachment),
            reading_draws_split_unreadable_own_pixel_branch: unreadable_by(0),
            reading_draws_split_unreadable_own_pixel_derivative: unreadable_by(1),
            reading_draws_split_unreadable_own_pixel_sample: unreadable_by(2),
            reading_draws_split_unreadable_own_pixel_group_operation: unreadable_by(3),
            reading_draws_split_unreadable_own_pixel_store: unreadable_by(4),
            reading_draws_split_unreadable_own_pixel_atomic: unreadable_by(5),
            reading_draws_split_unreadable_own_pixel_image_write: unreadable_by(6),
            reading_draws_split_unreadable_own_pixel_undefined_operand: unreadable_by(7),
            reading_draws_split_unreadable_own_pixel_writes_memory: unreadable_by(8),
            reading_draws_split_unreadable_own_pixel_interlock: unreadable_by(9),
            reading_draws_split_unreadable_own_pixel_coherent: unreadable_by(10),
            reading_draws_split_unreadable_own_pixel_other: unreadable_by(11),
            reading_draws_by_coverage: counters.reading_draws_by_coverage.load(Ordering::Relaxed),
            reading_draws_leading_covered_within: leading_covered(LeadingShare::Within),
            reading_draws_leading_covered_to_half: leading_covered(LeadingShare::ToHalf),
            reading_draws_leading_covered_over_half: leading_covered(LeadingShare::OverHalf),
            coverage_queries: counters.coverage_queries.load(Ordering::Relaxed),
            coverage_pools_made: counters.coverage_pools_made.load(Ordering::Relaxed),
            reading_draws_pinned: counters.reading_draws_pinned.load(Ordering::Relaxed),
            stand_ins_after_pins: counters.stand_ins_after_pins.load(Ordering::Relaxed),
            feedback_loop_alias_images: counters.feedback_loop_alias_images.load(Ordering::Relaxed),
            unobserved_padding_aliases: counters.unobserved_padding_aliases.load(Ordering::Relaxed),
            kept_padding_aliases: counters.kept_padding_aliases.load(Ordering::Relaxed),
            first_moves_kept: counters.first_moves_kept.load(Ordering::Relaxed),
            first_move_bytes_kept: counters.first_move_bytes_kept.load(Ordering::Relaxed),
            pooled_pitched_images: counters.pooled_pitched_images.load(Ordering::Relaxed),
            pooled_textures_moved_in_place: (counters.pooled_textures_moved_in_place)
                .load(Ordering::Relaxed),
            pooled_texture_shadows_freed_after_move: (counters
                .pooled_texture_shadows_freed_after_move)
                .load(Ordering::Relaxed),
            feedback_loop_shadow_images: counters
                .feedback_loop_shadow_images
                .load(Ordering::Relaxed),
            depth_attachments_moved: counters.depth_attachments_moved.load(Ordering::Relaxed),
            depth_instances_held: counters.depth_instances_held.load(Ordering::Relaxed),
            held_depths_released: counters.held_depths_released.load(Ordering::Relaxed),
            shadows_zeroed: counters.shadows_zeroed.load(Ordering::Relaxed),
            shadow_copies: counters.shadow_copies.load(Ordering::Relaxed),
            shadow_copies_in_against_standing: counters
                .shadow_copies_in_against_standing
                .load(Ordering::Relaxed),
            shadow_copies_in_skipped_by_standing: counters
                .shadow_copies_in_skipped_by_standing
                .load(Ordering::Relaxed),
            shadow_texels_skipped_by_standing: counters
                .shadow_texels_skipped_by_standing
                .load(Ordering::Relaxed),
            shadow_texels_unskipped_unstanding: unskipped(Unskipped::Unstanding),
            shadow_texels_unskipped_between_instances: unskipped(Unskipped::BetweenInstances),
            shadow_texels_unskipped_after_encoder_writes: unskipped(Unskipped::AfterEncoderWrites),
            shadow_texels_unskipped_recorded_whole: unskipped(Unskipped::RecordedWhole),
            shadow_texels_unskipped_command_buffer: unskipped(Unskipped::CommandBuffer),
            shadow_texels_unskipped_let_go: unskipped(Unskipped::LetGo),
            shadow_texels_unskipped_stale_within: unskipped(Unskipped::StaleWithin),
            shadow_texels_unskipped_stood: standing(None),
            shadow_texels_unskipped_never_stood: standing(Some(LetGo::NeverStood)),
            shadow_texels_never_stood_unmet: never_stood(Missed::Unmet),
            shadow_texels_never_stood_not_current: never_stood(Missed::NotCurrent),
            shadow_texels_never_stood_dropped: never_stood(Missed::Dropped),
            shadow_texels_never_stood_partial: never_stood(Missed::Partial),
            shadow_texels_never_stood_end: never_stood(Missed::End),
            shadow_texels_unskipped_unnamed_since: standing(Some(LetGo::UnnamedSince)),
            shadow_texels_unskipped_truth_written: standing(Some(LetGo::TruthWritten)),
            shadow_texels_unskipped_image_written: standing(Some(LetGo::ImageWritten)),
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
            shadow_texels_first_copied_in_private_surfaces: counters
                .shadow_texels_first_copied_in_private_surfaces
                .load(Ordering::Relaxed),
            shadow_texels_first_copied_in_private_images: counters
                .shadow_texels_first_copied_in_private_images
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
            shadow_copies_back_spliced: counters.shadow_copies_back_spliced.load(Ordering::Relaxed),
            shadow_merges_on_host: counters.shadow_merges_on_host.load(Ordering::Relaxed),
            shadow_merge_host_nanos: counters.shadow_merge_host_nanos.load(Ordering::Relaxed),
            shadow_merge_bytes_on_host: counters.shadow_merge_bytes_on_host.load(Ordering::Relaxed),
            shadow_host_steps: counters.shadow_host_steps.load(Ordering::Relaxed),
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
            dirty_holds_withheld: dirty.holds_withheld.load(Ordering::Relaxed),
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
            host_step_owned_kernels_made: counters
                .host_step_owned_kernels_made
                .load(Ordering::Relaxed),
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
            encoders_raced_rebind: counters.encoders_raced_rebind.load(Ordering::Relaxed),
            encoders_raced_unreplayable: (counters.encoders_raced_unreplayable)
                .load(Ordering::Relaxed),
            rebinds_refused_calls_in_flight: (counters.rebinds_refused_calls_in_flight)
                .load(Ordering::Relaxed),
            rebinds_refused_unsubmitted: (counters.rebinds_refused_unsubmitted)
                .load(Ordering::Relaxed),
            encoder_calls_held_by_rebind: (counters.encoder_calls_held_by_rebind)
                .load(Ordering::Relaxed),
            fences_made: self.shared().fences.made(),
            pipeline_libraries_made: counters.pipeline_libraries_made.load(Ordering::Relaxed),
            fragment_shader_libraries_made: counters
                .fragment_shader_libraries_made
                .load(Ordering::Relaxed),
            pipelines_linked: counters.pipelines_linked.load(Ordering::Relaxed),
            linked_pipelines_shared: counters.linked_pipelines_shared.load(Ordering::Relaxed),
            dispatch_pipelines_pooled: counters.dispatch_pipelines_pooled.load(Ordering::Relaxed),
            region_pipelines_evicted: counters.region_pipelines_evicted.load(Ordering::Relaxed),
            tile_pipelines_pooled: counters.tile_pipelines_pooled.load(Ordering::Relaxed),
            upload_chunks_in_device_memory: (counters.upload_chunks_in_device_memory)
                .load(Ordering::Relaxed),
            tile_dispatches_recorded: commands(|counts| &counts.tile_dispatches_recorded),
        }
    }
}

/// Why a no-copy buffer's bytes lie in a copy of the caller's pages (`copied.rs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CopiedBecause {
    /// The caller's gate refused their import (`HostMemory::with_import_gate`).
    Refused,
    /// The device refused their pointer's import, or imports no pointer, and no dma-buf of them
    /// served it: no source of them was given, or it failed, or the device refused its dma-bufs.
    DeviceRefused,
    /// The import the device and the gate admitted failed.
    Failed,
    /// A rebind's new bytes lie at another offset in their page than the old, where a texture
    /// over the buffer was planned at the old offset: no import lays them there.
    Moved,
    /// Metal's private option bit 24 drops the GPU's writes to the memory, and the system maps no
    /// read-only view of the pages (`read_only_view.rs`): the copy drops them.
    ReadOnly,
    /// The process maps some of the pages read-only, whose GPU writes Metal drops and no import
    /// on Linux keeps out of them: the copy drops them.
    Unwritable,
    /// A live import of another buffer or heap holds some of the pages and not all, or more than
    /// a heap's own bytes: a second import would pin them twice where either is of a host
    /// pointer, or be unordered by the device's barriers where both are dma-bufs
    /// (`host_aliases.rs`).
    OverImport,
    /// An import of a guest RAM's chunk, made or in flight, holds some of the pages, at the RAM's
    /// own addresses, and no chunk binds the buffer (its bytes cross a chunk's end, or no run
    /// resolver routed them): a second import would pin them twice (`host_ram.rs`).
    OverRamChunk,
}
