//! The device's begin-ahead thread (`begin_ahead.rs`): a command buffer that gives its pool back
//! settled posts it, and the thread begins as many of the pool's primaries as encoders of that
//! command buffer opened their own, for the encoders of the command buffer that takes the pool
//! next; the completing thread begins none itself, so a waiter's return waits for no begin.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The function is the owned fixture
//! `validation/fixtures/public/engine/tob_renders.ll` (`llvm-as`).

use super::blit_holds::{Scratch, WAIT};
use super::draw_again::{adding_state, device, rectangles, run_into, Heard};
use crate::begin_ahead::BeginAhead;
use crate::{
    Device, RenderCommandEncoder, ResourceOptions, Texture, TextureDescriptor, WorkCounts,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};

/// The render encoders of each command buffer: fewer than `spares::AHEAD`, with the readback's
/// blit encoder after them.
const ENCODERS: usize = 5;

/// The device's begin-ahead thread held at each posted pool's begins until this drops, a failing
/// case's unwinding too, so that the device's teardown, which joins the thread, never waits for
/// a hold left behind.
pub(super) struct HeldBegins<'a>(&'a BeginAhead);

impl HeldBegins<'_> {
    pub(super) fn new(device: &Device) -> HeldBegins<'_> {
        let begin_ahead = &device.shared().begin_ahead;
        begin_ahead.held().hold();
        HeldBegins(begin_ahead)
    }
}

impl Drop for HeldBegins<'_> {
    fn drop(&mut self) {
        self.0.held().release();
    }
}

/// A new `RGBA8Unorm` render target, 16 texels square.
fn target(device: &Device) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 16, 16, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    device.new_texture(&descriptor).unwrap()
}

/// The primaries begun ahead so far (`WorkCounts::primaries_begun_ahead`).
fn begun(device: &Device) -> u64 {
    device.work_counts().primaries_begun_ahead
}

/// A command buffer of [`ENCODERS`] render encoders, each of which begins its own primary in the
/// new pool it takes, gives its pool back settled: once its wait returned and the device's
/// begin-ahead thread settled, the thread began as many of the pool's primaries ahead, and more
/// for the readback's blit encoder, `spares::AHEAD` at most. The next command buffer of
/// [`ENCODERS`] render encoders takes that pool, the device's only spare, and opens each from a
/// primary begun ahead (`WorkCounts::encoder_primaries_begun_ahead`), none missed, with the
/// pixels of the first.
#[test]
fn each_render_encoder_after_a_settled_command_buffer_opens_from_a_primary_begun_ahead() {
    let scratch = Scratch::new("begin-ahead-encoders");
    let heard = Heard::default();
    let device = device(&scratch.0, &[], &heard);
    let state = adding_state(&device);
    let draw = |encoder: &mut RenderCommandEncoder| rectangles(encoder, &state, |_| {});
    let target = target(&device);
    let queue = device.new_command_queue().unwrap();
    let taken = |counts: &WorkCounts| counts.encoder_primaries_begun_ahead;
    let missed = |counts: &WorkCounts| counts.encoder_primaries_begun_ahead_missed;
    let before = begun(&device);
    let first = run_into(&device, &queue, (&target, ENCODERS), draw, true);
    device.shared().begin_ahead.settle();
    let ahead = begun(&device) - before;
    let spares = device.shared().spares.command_pools_held();
    println!(
        "BEGIN-AHEAD {}: {ahead} primaries begun ahead after a command buffer of {ENCODERS} \
         render encoders; the spares hold {spares:?}",
        device.name()
    );
    assert_eq!(first.counted(taken), 0, "a new pool holds none begun");
    assert_eq!(first.commands["begin_command_buffer"], ENCODERS as u64);
    assert!(
        (ENCODERS as u64..=8).contains(&ahead),
        "{ahead} begun ahead for {ENCODERS} render encoders and a blit encoder"
    );
    assert_eq!(spares.len(), 1, "the first command buffer's pool alone");
    let second = run_into(&device, &queue, (&target, ENCODERS), draw, true);
    assert_eq!(
        second.counted(taken),
        ENCODERS as u64,
        "each render encoder opened from a primary begun ahead: {}",
        second.moved()
    );
    assert_eq!(second.counted(missed), 0, "none missed");
    assert_eq!(second.commands["begin_command_buffer"], ENCODERS as u64);
    assert_eq!(second.words, first.words, "the pixels");
}

/// A command buffer's waiter returns while the device's begin-ahead thread, held, has not begun
/// its pool's primaries: the completing thread only posted the pool, which the thread holds at
/// its begins. The next command buffer takes the pool before the begins, and each of its render
/// encoders, which the pool's begin ahead owed one, counts a miss
/// (`WorkCounts::encoder_primaries_begun_ahead_missed`) and begins its own. Released, the
/// thread begins the pool's primaries once that command buffer gave it back, and the one after
/// opens each render encoder from one.
#[test]
fn a_waiter_returns_before_the_begins_and_a_taker_before_them_counts_its_encoders_missed() {
    let scratch = Scratch::new("begin-ahead-missed");
    let heard = Heard::default();
    let device = device(&scratch.0, &[], &heard);
    let state = adding_state(&device);
    let draw = |encoder: &mut RenderCommandEncoder| rectangles(encoder, &state, |_| {});
    let target = target(&device);
    let queue = device.new_command_queue().unwrap();
    let taken = |counts: &WorkCounts| counts.encoder_primaries_begun_ahead;
    let missed = |counts: &WorkCounts| counts.encoder_primaries_begun_ahead_missed;
    let held = HeldBegins::new(&device);
    let before = begun(&device);
    let first = run_into(&device, &queue, (&target, ENCODERS), draw, true);
    // The waiter returned; the thread holds the posted pool at its begins.
    device.shared().begin_ahead.held().until_waiting(1, WAIT);
    assert_eq!(
        begun(&device),
        before,
        "the waiter returned before any begin"
    );
    let second = run_into(&device, &queue, (&target, ENCODERS), draw, true);
    assert_eq!(
        second.counted(taken),
        0,
        "none begun before the second took the pool"
    );
    assert_eq!(
        second.counted(missed),
        ENCODERS as u64,
        "each render encoder the begin ahead owed one: {}",
        second.moved()
    );
    assert_eq!(second.commands["begin_command_buffer"], ENCODERS as u64);
    drop(held);
    device.shared().begin_ahead.settle();
    assert!(
        begun(&device) - before >= ENCODERS as u64,
        "begun once released"
    );
    let third = run_into(&device, &queue, (&target, ENCODERS), draw, true);
    assert_eq!(third.counted(taken), ENCODERS as u64, "{}", third.moved());
    assert_eq!(third.counted(missed), 0);
    assert_eq!(
        (second.words, third.words),
        (first.words.clone(), first.words)
    );
}
