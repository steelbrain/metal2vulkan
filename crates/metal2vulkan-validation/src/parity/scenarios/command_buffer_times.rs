//! Scenario `command_buffer_times`: a command buffer's `GPUStartTime`, `GPUEndTime`,
//! `kernelStartTime` and `kernelEndTime`, in the host's time base, for a
//! command buffer with no work, one fill, and a fill on each side of a shared event's signal and
//! wait. The records are facts, never the times: each of the four is 0 before the commit; once
//! the command buffer completed, the kernel times lie in order between the host times around its
//! commit and its wait, and the GPU times in order no earlier than the kernel start and no later
//! than the host time after the wait, within [`SLACK`]. The times themselves are reported with
//! the case's wall time (`Observation::completed`).

use super::super::api::{Api, Observation, Times};
use metal2vulkan_runtime::ResourceOptions;

/// How far, in seconds, a GPU time may lie outside the host times around it.
const SLACK: f64 = 1e-3;

/// The bytes each fill writes.
const LENGTH: u64 = 1 << 20;

/// What a case's command buffer holds.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Work {
    Nothing,
    OneFill,
    AroundAnEvent,
}

pub(super) fn run<A: Api>(api: &A, case: &str, obs: &mut Observation) -> Result<(), String> {
    let work = match case {
        "no_work" => Work::Nothing,
        "one_fill" => Work::OneFill,
        "fills_around_an_event" => Work::AroundAnEvent,
        other => return Err(format!("command_buffer_times has no case {other}")),
    };
    timed(api, obs, work)
}

/// One fill of the whole of `buffer` with `value`, in an encoder of its own on `cb`, its calls
/// named with `which` after them where a case makes more than one.
fn fill<A: Api>(
    api: &A,
    obs: &mut Observation,
    cb: &A::CommandBuffer,
    (buffer, value): (&A::Buffer, u8),
    which: Option<&str>,
) -> Option<()> {
    let op = |call: &str| which.map_or(call.to_owned(), |which| format!("{call} {which}"));
    let mut blit = obs.call(&op("blit"), || api.blit(cb))?;
    obs.call(&op("fill"), || {
        api.fill(&mut blit, buffer, 0..LENGTH, value)
    })?;
    obs.call(&op("end blit"), || api.end_blit(blit))
}

/// The case's command buffer, its times before the commit and once it completed, as facts.
fn timed<A: Api>(api: &A, obs: &mut Observation, work: Work) -> Result<(), String> {
    let Some(queue) = obs.call("queue", || api.queue()) else {
        return Ok(());
    };
    let Some(buffer) = obs.call("buffer", || api.buffer(LENGTH, ResourceOptions::private())) else {
        return Ok(());
    };
    let Some(cb) = obs.call("command buffer", || api.command_buffer(&queue)) else {
        return Ok(());
    };
    let encoded = match work {
        Work::Nothing => Some(()),
        Work::OneFill => fill(api, obs, &cb, (&buffer, 0x5a), None),
        Work::AroundAnEvent => (|| {
            let event = obs.call("event", || api.event())?;
            fill(api, obs, &cb, (&buffer, 0x11), Some("before the signal"))?;
            obs.call("signal", || api.signal(&cb, &event, 1))?;
            obs.call("wait", || api.wait(&cb, &event, 1))?;
            fill(api, obs, &cb, (&buffer, 0x22), Some("after the wait"))
        })(),
    };
    if encoded.is_none() {
        return Ok(());
    }
    let Some(times) = obs.call("times before the commit", || api.times(&cb)) else {
        return Ok(());
    };
    let zero = [
        times.gpu_start,
        times.gpu_end,
        times.kernel_start,
        times.kernel_end,
    ]
    .map(|time| u8::from(time == 0.0));
    obs.bytes("each time is 0 before the commit", &zero);
    let before = api.host_time();
    if obs.call("commit", || api.commit(&cb)).is_none() {
        return Ok(());
    }
    let completion = api.completion(cb)?;
    let after = api.host_time();
    let times = completion.times;
    obs.completed("completion", completion);
    if let Some(times) = times {
        obs.bytes("the times once completed", &facts(times, before, after));
    }
    Ok(())
}

/// The facts of `times` once the command buffer completed, between the host times `before` its
/// commit and `after` its wait, one byte each (1 where it holds).
fn facts(times: Times, before: f64, after: f64) -> [u8; 8] {
    let Times {
        gpu_start,
        gpu_end,
        kernel_start,
        kernel_end,
    } = times;
    [
        gpu_start > 0.0,
        gpu_end >= gpu_start,
        kernel_start > 0.0,
        kernel_end >= kernel_start,
        kernel_start >= before,
        kernel_end <= after,
        gpu_start >= kernel_start - SLACK,
        gpu_end <= after + SLACK,
    ]
    .map(u8::from)
}
