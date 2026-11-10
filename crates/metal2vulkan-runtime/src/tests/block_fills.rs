//! A pooled block of plain memory has its pages filled through an exportable allocation first
//! (`memory.rs`'s module docs, "Blocks filled exportable first") on a device whose driver
//! identity says its plain allocations hold its submissions
//! (`DriverFact::PlainAllocationsHoldSubmissions`) and that exports pooled memory as an opaque
//! file descriptor (`memory::fd_buffer_types`), each such block counted as filled
//! (`WorkCounts::plain_block_fills_saved` or `plain_block_fills_missed`), and on no other device.
//! The run makes `Private` textures until the pool of images made [`BLOCKS`] blocks and reads the
//! device's counts of fills.
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`).

use super::freed_memory::device;
use crate::{DriverFact, ResourceOptions, Texture, TextureDescriptor, WorkCounts};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::time::{Duration, Instant};

/// The blocks of images the run waits for.
const BLOCKS: u64 = 3;

/// The most textures of 4 MiB the run makes, well past what those blocks hold.
const MOST: usize = 128;

/// The blocks of images made, ahead or in place, from `start` to `counts`.
fn image_blocks(start: &WorkCounts, counts: &WorkCounts) -> u64 {
    (counts.image_blocks_made_ahead + counts.image_blocks_made_in_place)
        - (start.image_blocks_made_ahead + start.image_blocks_made_in_place)
}

#[test]
fn plain_blocks_are_filled_exportable_first_where_their_making_holds_submissions() {
    let scratch = std::env::temp_dir().join(format!("m2v-block-fills-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let device = device(&scratch);
    let holds = device
        .properties()
        .driver_facts
        .contains(&DriverFact::PlainAllocationsHoldSubmissions);
    let types = crate::device::lock(&device.shared().memory).fill_types();
    let start = device.work_counts();
    let mut held: Vec<Texture> = Vec::new();
    let mut counts = start;
    while image_blocks(&start, &counts) < BLOCKS {
        assert!(
            held.len() < MOST,
            "{MOST} textures made {} blocks of images",
            image_blocks(&start, &counts)
        );
        let mut descriptor =
            TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 1024, 1024, false);
        descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
        descriptor.resource_options = ResourceOptions::private();
        held.push(device.new_texture(&descriptor).unwrap());
        let at = Instant::now();
        while !crate::device::lock(&device.shared().memory).readying_ended() {
            assert!(
                at.elapsed() < Duration::from_secs(30),
                "a making never ended"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        counts = device.work_counts();
    }
    let saved = counts.plain_block_fills_saved - start.plain_block_fills_saved;
    let missed = counts.plain_block_fills_missed - start.plain_block_fills_missed;
    let blocks = image_blocks(&start, &counts);
    println!(
        "BLOCK-FILLS {}: plain allocations hold its submissions {holds}, blocks filled first of \
         memory types {types:#x}; of {blocks} blocks of images made, {saved} filled first and \
         made under 1 ms, {missed} filled first and not",
        device.name(),
    );
    match holds {
        true => {
            assert_ne!(
                types, 0,
                "a device whose plain allocations hold its submissions fills no type first"
            );
            assert!(
                saved + missed >= blocks,
                "{blocks} blocks of images, {} filled first",
                saved + missed
            );
        }
        false => {
            assert_eq!(
                types, 0,
                "a device whose plain allocations hold no submission fills types first"
            );
            assert_eq!(
                saved + missed,
                0,
                "a device whose plain allocations hold no submission filled blocks first"
            );
        }
    }
    drop(held);
    drop(device);
    let _ = std::fs::remove_dir_all(&scratch);
}
