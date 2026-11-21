//! Where each pipeline state's making went, from the runtime's own counters read on the play's
//! thread before and after it (`rt::Metrics`): the translator's time (`translation_nanos`), the
//! driver's (`pipeline_creation_nanos`, its `vkCreate*Pipelines` calls), and the rest of the
//! making's wall time, on the clock they count (`rt::host_time`); and what it made: functions
//! lowered (`translations`), lowerings loaded from the cache directory (`translations_loaded`) and
//! `VkPipeline`s made (`pipelines_created`). A making that made any of them is of a content new to
//! the device, held otherwise. The counters sum the calls of every thread, so a making while a
//! call of the pre-pass ran (`prepass::Pool::quiet`) counts that call's work too: such makings are
//! counted apart, and their figures are not one making's.

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

/// What a making made, or several: functions lowered, lowerings loaded and `VkPipeline`s.
#[derive(Clone, Copy, Default)]
struct Made {
    lowered: u64,
    loaded: u64,
    pipelines: u64,
}

impl Made {
    fn add(&mut self, other: Made) {
        self.lowered += other.lowered;
        self.loaded += other.loaded;
        self.pipelines += other.pipelines;
    }

    /// Whether it made anything: the content was new to the device.
    fn any(&self) -> bool {
        self.lowered + self.loaded + self.pipelines > 0
    }

    /// `L lowered, D loaded, P VkPipelines`.
    fn text(&self) -> String {
        let Made {
            lowered,
            loaded,
            pipelines,
        } = self;
        format!("{lowered} lowered, {loaded} loaded, {pipelines} VkPipelines")
    }
}

/// The makings of one kind summed: how many, what they made and their times.
#[derive(Default)]
struct Sum {
    states: u64,
    made: Made,
    split: Split,
}

/// One of the slowest makings.
struct Slow {
    state: u64,
    kinds: &'static str,
    beside: bool,
    made: Made,
    split: Split,
}

/// The counters a split reads: `translation_nanos`, `pipeline_creation_nanos`, `translations`,
/// `translations_loaded` and `pipelines_created`.
fn read(device: &rt::Device) -> [u64; 5] {
    let metrics = device.counters().metrics();
    [
        metrics.translation_nanos,
        metrics.pipeline_creation_nanos,
        metrics.translations,
        metrics.translations_loaded,
        metrics.pipelines_created,
    ]
}

/// A making's start: the pre-pass's calls begun where none ran then (no pre-pass, none), the
/// counters, and the clock.
pub(super) struct Mark {
    quiet: Option<u64>,
    counters: [u64; 5],
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

/// The pipeline states' makings, of new contents and held ones, the makings beside a pre-pass
/// call, and the slowest ([`SLOWEST`]), slowest first.
#[derive(Default)]
pub(super) struct Makes {
    new: Sum,
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
        let [translation, driver, lowered, loaded, pipelines] =
            std::array::from_fn(|at| counters[at].saturating_sub(mark.counters[at]));
        let made = Made {
            lowered,
            loaded,
            pipelines,
        };
        let split = Split {
            wall,
            translation,
            driver,
        };
        let sum = match made.any() {
            true => &mut self.new,
            false => &mut self.held,
        };
        sum.states += 1;
        sum.made.add(made);
        sum.split.add(split);
        self.beside += u64::from(!quiet);
        let at = (self.slowest.iter())
            .take_while(|slow| slow.split.wall >= wall)
            .count();
        if at < SLOWEST {
            let slow = Slow {
                state,
                kinds: kinds(),
                beside: !quiet,
                made,
                split,
            };
            self.slowest.insert(at, slow);
            self.slowest.truncate(SLOWEST);
        }
        wall
    }

    /// `pipeline makes N states: C new (MADE), SPLIT; H held, SPLIT; B beside a pre-pass call
    /// ...`, then `pipeline make slowest K #ID KINDS new|held (MADE): SPLIT` for each of the
    /// slowest.
    pub(super) fn lines(&self) -> Vec<String> {
        let (new, held) = (&self.new, &self.held);
        let mut lines = vec![format!(
            "pipeline makes {} states: {} new ({}), {}; {} held, {}; {} beside a pre-pass call, \
             whose work their figures count",
            new.states + held.states,
            new.states,
            new.made.text(),
            new.split.text(),
            held.states,
            held.split.text(),
            self.beside,
        )];
        for (at, slow) in self.slowest.iter().enumerate() {
            lines.push(format!(
                "pipeline make slowest {} #{} {} {} ({}): {}{}",
                at + 1,
                slow.state,
                slow.kinds,
                if slow.made.any() { "new" } else { "held" },
                slow.made.text(),
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
