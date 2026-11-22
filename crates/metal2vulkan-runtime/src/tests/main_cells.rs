//! A render content's main pipeline cell (`crate::render::Main`), without a device: what a state's
//! `finish_pipeline` and a draw before it do with it, a driver's create that fails after the
//! state's answer included.
//!
//! - A finish whose make fails keeps nothing. A draw then makes the pipeline itself, a first-use
//!   wait on an unfinished pipeline; where its make fails too it answers the driver's error and
//!   counts no wait, and the next draw makes it. Once made, a draw reads it and a finish makes
//!   nothing.
//! - A draw that meets a finish's make in flight makes nothing itself: it waits for that make and
//!   takes its pipeline, and a second finish meanwhile makes nothing.

use crate::counters::Counters;
use crate::render::Main;
use crate::Error;
use ash::vk;
use ash::vk::Handle;
use std::sync::mpsc;

/// The driver's error of a create that failed: device memory exhausted, one of the failures
/// Vulkan lets `vkCreateGraphicsPipelines` answer a valid create with.
fn failed() -> Result<vk::Pipeline, Error> {
    Err(Error::Vulkan {
        call: "vkCreateGraphicsPipelines",
        result: vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
    })
}

/// Whether `made` is the driver's error [`failed`] gives.
fn is_failed<T>(made: &Result<T, Error>) -> bool {
    matches!(
        made,
        Err(Error::Vulkan {
            result: vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
            ..
        })
    )
}

#[test]
fn a_failed_finish_keeps_nothing_and_the_first_draw_makes_the_pipeline_again() {
    let counters = Counters::new();
    let main = Main::default();
    let pipeline = vk::Pipeline::from_raw(7);
    let waits = || counters.metrics().first_use_unfinished_waits;
    assert!(is_failed(&main.make_ahead(failed)), "the finish's error");
    assert!(
        is_failed(&main.pipeline(&counters, failed)),
        "a draw whose make fails too answers the driver's error"
    );
    assert_eq!(waits(), 0, "a draw that made nothing counts no wait");
    let made = main.pipeline(&counters, || Ok(pipeline));
    assert_eq!(made.ok(), Some(pipeline), "the next draw makes it");
    assert_eq!(waits(), 1, "a first-use wait on an unfinished pipeline");
    let read = main.pipeline(&counters, || panic!("a made pipeline is not made again"));
    assert_eq!(read.ok(), Some(pipeline), "a draw reads it");
    let finish = main.make_ahead(|| panic!("a made pipeline is not made again"));
    assert_eq!(finish.ok(), Some(None), "a finish makes nothing");
    assert_eq!(waits(), 1, "a draw that read it counts no wait");
}

#[test]
fn a_draw_during_a_finish_waits_for_its_make() {
    let counters = Counters::new();
    let main = Main::default();
    let pipeline = vk::Pipeline::from_raw(9);
    let (started, start) = mpsc::channel();
    let (release, released) = mpsc::channel::<()>();
    let (main, counters) = (&main, &counters);
    std::thread::scope(|scope| {
        let finish = scope.spawn(move || {
            main.make_ahead(move || {
                started.send(()).expect("the test's channel");
                released.recv().expect("the test's channel");
                Ok(pipeline)
            })
        });
        start.recv().expect("the finish's make began");
        let draw = scope.spawn(move || {
            main.pipeline(counters, || {
                panic!("a draw makes nothing while a finish makes it")
            })
        });
        let again = main.make_ahead(|| panic!("a second finish makes nothing meanwhile"));
        assert_eq!(again.ok(), Some(None), "a second finish");
        release.send(()).expect("the finish's make waits");
        let finished = finish.join().expect("the finish");
        assert!(matches!(finished, Ok(Some(_))), "the finish made it");
        let drawn = draw.join().expect("the draw");
        assert_eq!(
            drawn.ok(),
            Some(pipeline),
            "the draw took the finish's pipeline"
        );
    });
    assert!(
        counters.metrics().first_use_unfinished_waits <= 1,
        "a draw that met the make in flight waited once, one after it none"
    );
}
