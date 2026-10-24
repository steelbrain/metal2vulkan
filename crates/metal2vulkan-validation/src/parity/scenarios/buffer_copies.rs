//! Scenario `buffer_copies`: `copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:` and
//! `fillBuffer:range:value:` at odd offsets and lengths, between buffers and within one, and the
//! calls Metal's API validation stops that the runtime refuses (a range past either buffer, a
//! fill past the buffer). A zero size and an empty fill, which plain Metal runs as nothing and
//! the runtime too, are the lane tests' (`runtime_lane::blits`, DESIGN D406 item 2).

use super::super::api::{Api, Observation};
use super::steps::{pattern, shared_with, submit};
use metal2vulkan_runtime::ResourceOptions;
use std::ops::Range;

/// Every buffer's length.
const LENGTH: u64 = 64;

/// A copy: source offset, destination offset, size.
type Copy = (u64, u64, u64);

/// Copies into another buffer, in order: odd offsets and sizes, some overlapping earlier ones.
const BETWEEN: [Copy; 5] = [(1, 3, 5), (0, 7, 13), (6, 0, 4), (3, 2, 9), (17, 40, 23)];

/// Copies within one buffer, in order: forward and backward overlaps, and a range onto itself.
const WITHIN: [Copy; 4] = [(0, 3, 10), (20, 17, 10), (40, 40, 8), (30, 33, 1)];

/// Fills, in order, after a fill of the whole buffer with `0x11`.
const FILLS: [(Range<u64>, u8); 5] = [
    (1..2, 0x22),
    (3..10, 0x33),
    (13..14, 0x44),
    (15..33, 0x55),
    (61..64, 0x66),
];

pub(super) fn run<A: Api>(api: &A, case: &str, obs: &mut Observation) -> Result<(), String> {
    match case {
        "copies_between_buffers" => copies(api, obs, false),
        "copies_within_one_buffer" => copies(api, obs, true),
        "fills_of_odd_ranges" => fills(api, obs),
        "a_copy_past_the_source" => one_copy(api, obs, (60, 0, 8)),
        "a_copy_past_the_destination" => one_copy(api, obs, (0, 60, 8)),
        "a_fill_past_the_buffer" => one_fill(api, obs, 60..68),
        other => Err(format!("buffer_copies has no case {other}")),
    }
}

/// A `Private` buffer holding `pattern(LENGTH, seed)`, copied in by `blit` from a `Shared` one.
fn private_with<A: Api>(
    api: &A,
    obs: &mut Observation,
    blit: &mut A::Blit,
    seed: u8,
) -> Option<(A::Buffer, A::Buffer)> {
    let source = shared_with(api, obs, "upload", &pattern(LENGTH, seed))?;
    let private = obs.call("private", || api.buffer(LENGTH, ResourceOptions::private()))?;
    obs.call("upload copy", || {
        api.copy_buffer(blit, (&source, 0), (&private, 0), LENGTH)
    })?;
    Some((source, private))
}

/// `pattern(LENGTH, 3)` copied into a `Private` buffer of `pattern(LENGTH, 100)` (or within it,
/// when `within`) by each copy in turn, in one encoder; then the buffer read back.
fn copies<A: Api>(api: &A, obs: &mut Observation, within: bool) -> Result<(), String> {
    let Some(queue) = obs.call("queue", || api.queue()) else {
        return Ok(());
    };
    let Some(cb) = obs.call("command buffer", || api.command_buffer(&queue)) else {
        return Ok(());
    };
    let Some(mut blit) = obs.call("blit", || api.blit(&cb)) else {
        return Ok(());
    };
    let Some((_upload, target)) = private_with(api, obs, &mut blit, 100) else {
        return Ok(());
    };
    let Some(source) = shared_with(api, obs, "source", &pattern(LENGTH, 3)) else {
        return Ok(());
    };
    let copies: &[Copy] = if within { &WITHIN } else { &BETWEEN };
    let from = if within { &target } else { &source };
    for &(at, to, size) in copies {
        let op = format!("copy {at} to {to}, {size} bytes");
        let copied = obs.call(&op, || {
            api.copy_buffer(&mut blit, (from, at), (&target, to), size)
        });
        if copied.is_none() {
            return Ok(());
        }
    }
    read_back(api, obs, cb, blit, &target)
}

/// A `Private` buffer filled with `0x11`, then by each fill in turn, in one encoder; then read
/// back.
fn fills<A: Api>(api: &A, obs: &mut Observation) -> Result<(), String> {
    let Some(queue) = obs.call("queue", || api.queue()) else {
        return Ok(());
    };
    let Some(cb) = obs.call("command buffer", || api.command_buffer(&queue)) else {
        return Ok(());
    };
    let Some(target) = obs.call("private", || api.buffer(LENGTH, ResourceOptions::private()))
    else {
        return Ok(());
    };
    let Some(mut blit) = obs.call("blit", || api.blit(&cb)) else {
        return Ok(());
    };
    let fills = std::iter::once((0..LENGTH, 0x11)).chain(FILLS);
    for (range, value) in fills {
        let op = format!("fill {range:?} with {value:#04x}");
        if obs
            .call(&op, || api.fill(&mut blit, &target, range.clone(), value))
            .is_none()
        {
            return Ok(());
        }
    }
    read_back(api, obs, cb, blit, &target)
}

/// Copy `target` into a `Shared` buffer at the end of `blit`, end it, submit `cb`, and record
/// the bytes.
fn read_back<A: Api>(
    api: &A,
    obs: &mut Observation,
    cb: A::CommandBuffer,
    mut blit: A::Blit,
    target: &A::Buffer,
) -> Result<(), String> {
    let Some(out) = obs.call("read-back buffer", || {
        api.buffer(LENGTH, ResourceOptions::shared())
    }) else {
        return Ok(());
    };
    if obs
        .call("read-back copy", || {
            api.copy_buffer(&mut blit, (target, 0), (&out, 0), LENGTH)
        })
        .is_none()
        || obs.call("end blit", || api.end_blit(blit)).is_none()
        || submit(api, obs, "commit", cb)?.is_none()
    {
        return Ok(());
    }
    obs.bytes("bytes", &api.read(&out));
    Ok(())
}

/// One copy between two `Shared` buffers of `LENGTH` bytes: a call Metal's API validation
/// stops, or not.
fn one_copy<A: Api>(api: &A, obs: &mut Observation, (at, to, size): Copy) -> Result<(), String> {
    let Some(queue) = obs.call("queue", || api.queue()) else {
        return Ok(());
    };
    let Some(cb) = obs.call("command buffer", || api.command_buffer(&queue)) else {
        return Ok(());
    };
    let Some(source) = shared_with(api, obs, "source", &pattern(LENGTH, 3)) else {
        return Ok(());
    };
    let Some(target) = shared_with(api, obs, "target", &pattern(LENGTH, 100)) else {
        return Ok(());
    };
    let Some(mut blit) = obs.call("blit", || api.blit(&cb)) else {
        return Ok(());
    };
    let op = format!("copy {at} to {to}, {size} bytes");
    if obs
        .call(&op, || {
            api.copy_buffer(&mut blit, (&source, at), (&target, to), size)
        })
        .is_none()
        || obs.call("end blit", || api.end_blit(blit)).is_none()
        || submit(api, obs, "commit", cb)?.is_none()
    {
        return Ok(());
    }
    obs.bytes("bytes", &api.read(&target));
    Ok(())
}

/// One fill of a `Shared` buffer of `LENGTH` bytes: a call Metal's API validation stops, or
/// not.
fn one_fill<A: Api>(api: &A, obs: &mut Observation, range: Range<u64>) -> Result<(), String> {
    let Some(queue) = obs.call("queue", || api.queue()) else {
        return Ok(());
    };
    let Some(cb) = obs.call("command buffer", || api.command_buffer(&queue)) else {
        return Ok(());
    };
    let Some(target) = shared_with(api, obs, "target", &pattern(LENGTH, 100)) else {
        return Ok(());
    };
    let Some(mut blit) = obs.call("blit", || api.blit(&cb)) else {
        return Ok(());
    };
    let op = format!("fill {range:?} with 0x5a");
    if obs
        .call(&op, || api.fill(&mut blit, &target, range.clone(), 0x5a))
        .is_none()
        || obs.call("end blit", || api.end_blit(blit)).is_none()
        || submit(api, obs, "commit", cb)?.is_none()
    {
        return Ok(());
    }
    obs.bytes("bytes", &api.read(&target));
    Ok(())
}
