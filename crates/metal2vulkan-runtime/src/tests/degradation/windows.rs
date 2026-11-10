//! Windows child-process proof of the announcement abort rule.
//!
//! Windows has no fork or POSIX wait status. Run the actual pool operation in a
//! new copy of the test executable and check Rust's native fast-fail status.

const CHILD: &str = "tests::degradation::windows::pool_piece_child";

pub(super) fn assert_pool_piece_child() {
    let output = std::process::Command::new(std::env::current_exe().expect("test executable"))
        .args(["--exact", CHILD, "--ignored", "--nocapture"])
        .output()
        .expect("announcement child");
    if cfg!(debug_assertions) {
        // Rust's Windows abort uses FAST_FAIL_FATAL_APP_EXIT (int 0x29),
        // reported as STATUS_STACK_BUFFER_OVERRUN, not a Rust test panic (101).
        assert_eq!(output.status.code(), Some(0xc000_0409u32.cast_signed()));
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(
                "metal2vulkan-runtime: a degradation was announced inside a pipeline pool piece"
            ),
            "missing breach diagnostic: {:?}",
            output.stderr
        );
    } else {
        assert!(output.status.success(), "announcement child: {output:?}");
    }
}

/// Invoked by the parent above; it must not abort the main test process.
#[test]
#[ignore = "run by the announcement-rule parent in its own process"]
fn pool_piece_child() {
    use crate::degradation::{check_announcement, Degradation, NoticePlace};
    use crate::pipeline::pool::{Joined, Pool};
    #[link(name = "wer")]
    unsafe extern "system" {
        fn WerSetFlags(flags: u32) -> i32;
    }
    // SAFETY: this changes only this child process's reporting UI. A death
    // test must not open a Windows crash dialog or change the host's settings.
    assert_eq!(unsafe { WerSetFlags(0x0020) }, 0);
    let pieces: Vec<Joined<()>> = vec![Box::new(|| {
        assert_eq!(
            check_announcement(Degradation::EncodedUseRebound, false),
            Some(NoticePlace::PipelinePoolPiece)
        );
    })];
    drop(Pool::default().join_all(pieces, true, |_, _| {}));
}
