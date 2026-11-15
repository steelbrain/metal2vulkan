//! A device whose options name no shader cache directory, where the platform's default does not
//! exist (no variable it is found by is set), is made all the same, as Metal makes its device
//! whatever its shader cache does: its cache is in memory alone. It says so once, as it is made,
//! as the degrade notice `ShaderCacheEntryNotWritten`, and never again however many entries go
//! unwritten; its pipeline states are made, each function translated once and found in the memo
//! after; nothing reaches a directory, and each entry it would have written is counted as not
//! written.
//!
//! The environment the default is looked up in is the test's own, empty
//! (`CacheSetup::of_in`), so the process's environment, which other tests share, is untouched.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are the owned fixture
//! `validation/fixtures/public/engine/ubo_bind_offsets.ll` (`llvm-as`).

use super::blit_holds::options;
use super::descriptor_key_once::descriptor;
use crate::shader_cache::CacheSetup;
use crate::{Degradation, DegradeNotice, Device, MessageSink, PipelineOptions, ShaderCacheOptions};
use metal2vulkan::metal::BlendFactor;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// The compiled functions every state is of: one vertex and one fragment.
const STAGES: u64 = 2;

#[test]
fn a_device_with_no_default_directory_keeps_its_cache_in_memory_and_says_so_once() {
    let heard: Arc<Mutex<Vec<DegradeNotice>>> = Arc::default();
    let hearing = Arc::clone(&heard);
    let mut options = options(Path::new("unused"));
    options.shader_cache = ShaderCacheOptions::default();
    options.messages = Some(MessageSink::new(move |message| {
        if let Some(notice) = &message.degradation {
            if notice.degradation == Degradation::ShaderCacheEntryNotWritten {
                hearing.lock().unwrap().push(notice.clone());
            }
        }
    }));
    let cache = CacheSetup::of_in(&options.shader_cache, "system_default_with", |_| None)
        .expect("no default directory fails nothing");
    let device = Device::system_default_set_up(&options, &cache).expect("the device is made");
    assert_eq!(device.shader_cache_directory(), None);
    let base = descriptor(&device);
    let factors = [
        BlendFactor::Zero,
        BlendFactor::One,
        BlendFactor::SourceColor,
        BlendFactor::SourceAlpha,
    ];
    let mut kept = Vec::with_capacity(factors.len());
    for factor in factors {
        let mut d = base.clone();
        let color = d.color_attachments[0]
            .as_mut()
            .expect("the fixture's target");
        color.blending_enabled = true;
        color.source_rgb_blend_factor = factor;
        kept.push(
            device
                .new_render_pipeline_state(&d, &PipelineOptions::default())
                .expect("a state made with its cache in memory alone"),
        );
    }
    let statistics = device.shader_cache_statistics();
    let translated = device.work_counts().functions_translated;
    println!(
        "NO DIRECTORY {}: {} states, {translated} functions translated, {statistics:?}",
        device.properties().name,
        kept.len(),
    );
    // Each function is translated once and found in the memo by every later state.
    assert_eq!(translated, STAGES);
    // Nothing reached a directory: no entry given, queued, written or read, no driver record.
    assert_eq!(statistics.writes, 0);
    assert_eq!(statistics.writes_queued, 0);
    assert_eq!(statistics.writes_on_thread, 0);
    assert_eq!(statistics.written_bytes, 0);
    assert_eq!(statistics.hits, 0);
    assert_eq!(statistics.read_failures, 0);
    assert_eq!(statistics.driver_saves, 0);
    // Each entry it would have written is counted as not written: the two functions at least.
    assert!(
        statistics.write_failures >= STAGES,
        "entries counted as not written: {statistics:?}"
    );
    let heard = heard.lock().unwrap();
    assert_eq!(
        heard.len(),
        1,
        "said once, as the device was made: {heard:?}"
    );
    assert_eq!(heard[0].object, "the default shader cache directory");
    assert!(
        heard[0].reason.starts_with("there is none: "),
        "{}",
        heard[0].reason
    );
    drop(kept);
    drop(device);
}
