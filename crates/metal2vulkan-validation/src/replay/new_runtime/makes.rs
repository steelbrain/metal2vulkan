//! Where each pipeline state's making went, from the runtime's own counters read on the play's
//! thread before and after it (`rt::Metrics`): the translator's time (`translation_nanos`), the
//! driver's (`pipeline_creation_nanos`, its `vkCreate*Pipelines` calls), and the rest of the
//! making's wall time, on the clock they count (`rt::host_time`). A making is cold where it lowered a function or made a `VkPipeline`
//! (`translations` or `pipelines_created` rose: the device held no content for the state), held
//! otherwise. The counters sum the calls of every thread, so a making while a call of the
//! pre-pass ran (`prepass::Pool::quiet`) counts that call's work too: such makings are counted
//! apart, and their times are not one making's.

use super::prepass::Pool;
use super::rt;

/// The slowest makings the report lists.
const SLOWEST: usize = 10;

/// A making's times, or the sum of several, in nanoseconds: its wall time, and of it the
/// translator's and the driver's.
#[derive(Clone, Copy, Default)]
struct Split {
    wall: u64,
    translation: u64,
    driver: u64,
}

impl Split {
    fn add(&mut self, other: Split) {
        self.wall += other.wall;
        self.translation += other.translation;
        self.driver += other.driver;
    }

    /// `wall W ms = translation T ms + driver D ms + other O ms`.
    fn text(&self) -> String {
        let ms = |nanos: u64| nanos as f64 / 1e6;
        let other = ms(self.wall) - ms(self.translation) - ms(self.driver);
        format!(
            "wall {:.3} ms = translation {:.3} ms + driver {:.3} ms + other {other:.3} ms",
            ms(self.wall),
            ms(self.translation),
            ms(self.driver),
        )
    }
}

/// The makings of one kind summed: how many, and their times.
#[derive(Default)]
struct Sum {
    states: u64,
    split: Split,
}

/// One of the slowest makings.
struct Slow {
    state: u64,
    kinds: &'static str,
    cold: bool,
    beside: bool,
    split: Split,
}

/// The counters a split reads: `translation_nanos`, `pipeline_creation_nanos`, `translations`
/// and `pipelines_created`.
fn read(device: &rt::Device) -> [u64; 4] {
    let metrics = device.counters().metrics();
    [
        metrics.translation_nanos,
        metrics.pipeline_creation_nanos,
        metrics.translations,
        metrics.pipelines_created,
    ]
}

/// A making's start: the pre-pass's calls begun where none ran then (no pre-pass, none), the
/// counters, and the clock.
pub(super) struct Mark {
    quiet: Option<u64>,
    counters: [u64; 4],
    began: u64,
}

impl Mark {
    /// A making that starts now on `device`, beside `prepass`.
    pub(super) fn new(device: &rt::Device, prepass: Option<&Pool>) -> Mark {
        let quiet = prepass.map_or(Some(0), Pool::quiet);
        let counters = read(device);
        Mark {
            quiet,
            counters,
            began: rt::host_time(),
        }
    }
}

/// The pipeline states' makings, cold and held, the makings beside a pre-pass call, and the
/// slowest ([`SLOWEST`]), slowest first.
#[derive(Default)]
pub(super) struct Makes {
    cold: Sum,
    held: Sum,
    beside: u64,
    slowest: Vec<Slow>,
}

impl Makes {
    /// The making `mark` started, of pipeline state `state`, ended with the state made: counted,
    /// with `kinds` its stages' kinds where it is among the slowest. Its wall time.
    pub(super) fn made(
        &mut self,
        mark: Mark,
        device: &rt::Device,
        prepass: Option<&Pool>,
        state: u64,
        kinds: impl FnOnce() -> &'static str,
    ) -> u64 {
        let wall = rt::host_time().saturating_sub(mark.began);
        let counters = read(device);
        let quiet = (mark.quiet).is_some_and(|begun| prepass.is_none_or(|p| p.quiet_since(begun)));
        let [translation, driver, lowered, pipelines] =
            std::array::from_fn(|at| counters[at].saturating_sub(mark.counters[at]));
        let cold = lowered > 0 || pipelines > 0;
        let split = Split {
            wall,
            translation,
            driver,
        };
        let sum = match cold {
            true => &mut self.cold,
            false => &mut self.held,
        };
        sum.states += 1;
        sum.split.add(split);
        self.beside += u64::from(!quiet);
        let at = (self.slowest.iter())
            .take_while(|slow| slow.split.wall >= wall)
            .count();
        if at < SLOWEST {
            let slow = Slow {
                state,
                kinds: kinds(),
                cold,
                beside: !quiet,
                split,
            };
            self.slowest.insert(at, slow);
            self.slowest.truncate(SLOWEST);
        }
        wall
    }

    /// `pipeline makes N states: C cold, SPLIT; H held, SPLIT; B beside a pre-pass call ...`,
    /// then `pipeline make slowest K #ID KINDS cold|held: SPLIT` for each of the slowest.
    pub(super) fn lines(&self) -> Vec<String> {
        let (cold, held) = (&self.cold, &self.held);
        let mut lines = vec![format!(
            "pipeline makes {} states: {} cold, {}; {} held, {}; {} beside a pre-pass call, whose \
             work their times count",
            cold.states + held.states,
            cold.states,
            cold.split.text(),
            held.states,
            held.split.text(),
            self.beside,
        )];
        for (at, slow) in self.slowest.iter().enumerate() {
            lines.push(format!(
                "pipeline make slowest {} #{} {} {}: {}{}",
                at + 1,
                slow.state,
                slow.kinds,
                if slow.cold { "cold" } else { "held" },
                slow.split.text(),
                if slow.beside {
                    ", beside a pre-pass call"
                } else {
                    ""
                },
            ));
        }
        lines
    }
}
