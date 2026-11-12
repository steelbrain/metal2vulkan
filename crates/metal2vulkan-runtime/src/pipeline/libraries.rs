//! A render variant's `VkPipeline`s linked from graphics pipeline libraries (the coordinator's
//! fast-link stream: "compile each stage once per function into a library, and make a state's first
//! pipeline by a fast link without link-time optimisation, in milliseconds").
//!
//! Where the device enables `VK_EXT_graphics_pipeline_library` and links its libraries fast
//! (`graphicsPipelineLibraryFastLinking`; `device/create.rs`), a render variant's `VkPipeline` of
//! a graphics key is linked, without link-time optimization, from four libraries, one per part
//! of the pipeline's state (`graphics.rs`, [`GraphicsContent`](super::graphics::GraphicsContent)): its vertex input, its
//! pre-rasterization shaders, its fragment shader and its fragment output. Each part's library is
//! the device's, found by the exact content of its part ([`Part`], a digest of everything its
//! create info holds: the layout's description, each stage's words, the constants its stages
//! declare with their values, the part's state, the pass and the dynamic states), so the keys and
//! states that share a part share its library: a fragment function's library is compiled once
//! per distinct fragment specialization, layout, fragment state and pass, and each key links in
//! milliseconds. The pass is in every part's name because the libraries a pipeline links must
//! be made against compatible render passes (`VUID-VkGraphicsPipelineCreateInfo-pLibraries-06628`),
//! and a pass's compatibility includes its attachments' formats.
//!
//! - **Same pipeline.** The libraries are made from the very create infos the whole pipeline's
//!   making takes, split by part; linking them is the driver's, so a linked pipeline is the
//!   whole pipeline the same state describes. Nothing an answer depends on changes.
//! - **The device's, by content** (`content_keyed.rs`). A program's call (a draw, a state's
//!   creation) claims each part's library, inserting it where the device has none; a pool piece
//!   only looks up, and on a miss makes one of its own that no lookup finds. A claimed library is
//!   made by the first caller that needs it made ([`Library::made`]); a caller meanwhile waits for
//!   that making, and one after a failed making makes its own attempt, as a draw's own making does
//!   (`making.rs`).
//! - **One linked pipeline per link.** A linked pipeline is the device's too, found by exactly
//!   what its linking takes ([`Link`]): its four libraries, by their parts' names, and the pipeline
//!   layout it is linked with, which no variant of another layout definition shares
//!   (`shared_layouts.rs`). So render states whose whole key is equal, each a variant of its own
//!   (compiled pipelines of their own whose stages come out as the same words, such as
//!   descriptors that differ only in the call stack depth of a function that recurses through no
//!   visible function), share one `VkPipeline`: the first links it, and every later one takes it
//!   with no driver call (`WorkCounts::linked_pipelines_shared`). Claimed and made as a library
//!   is (below).
//! - **Alive while linked.** A linked pipeline holds its four libraries and its layout ([`Link`]),
//!   each variant's pipeline holds the link ([`Linked`]), and a kept pipeline keeps it
//!   (`kept.rs`): the device's map destroys a linked pipeline only once nothing holds it, and a
//!   library only after every pipeline it linked, at a program's settle or the device's teardown,
//!   so no variant destroys one another variant draws with.
//! - **Cached across processes.** Each library of shaders (pre-rasterization, fragment shader) is
//!   made through the device's shader cache under its part's record (`shader_cache/driver.rs`),
//!   as a whole pipeline is: its part's name but for what reaches no stage's conversion, the
//!   pass's attachment formats and the depth bias (`GraphicsContent::record_parts`). So a part's
//!   libraries made against passes that differ only in an attachment's format, each a library of
//!   its own, share one record and one `VkPipelineCache`, where the driver finds the stages the
//!   first compiled. The vertex input and fragment output libraries hold no shader and keep no
//!   record.

use super::budget;
use super::content_keyed::{program_call, ContentKeyed, Owned};
use super::shared_layouts::SharedLayout;
use crate::device::{lock, DeviceShared};
use crate::error::Error;
use crate::release;
use ash::vk;
use metal2vulkan::cache::Hash256;
use std::sync::{Arc, Condvar, Mutex, PoisonError};

/// The four parts of a graphics pipeline's state, in the order a linked pipeline takes them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Part {
    VertexInput,
    PreRasterization,
    FragmentShader,
    FragmentOutput,
}

impl Part {
    pub(crate) const ALL: [Part; 4] = [
        Part::VertexInput,
        Part::PreRasterization,
        Part::FragmentShader,
        Part::FragmentOutput,
    ];

    /// Its library's flag.
    pub(crate) fn flag(self) -> vk::GraphicsPipelineLibraryFlagsEXT {
        use vk::GraphicsPipelineLibraryFlagsEXT as F;
        match self {
            Part::VertexInput => F::VERTEX_INPUT_INTERFACE,
            Part::PreRasterization => F::PRE_RASTERIZATION_SHADERS,
            Part::FragmentShader => F::FRAGMENT_SHADER,
            Part::FragmentOutput => F::FRAGMENT_OUTPUT_INTERFACE,
        }
    }

    /// Whether its library compiles shaders: the pre-rasterization and fragment shader parts.
    /// Only those are made through the device's shader cache, each under its part's record
    /// (`shader_cache/driver.rs`); the vertex input and fragment output libraries hold no shader,
    /// and are made without a pipeline cache.
    pub(crate) fn holds_shaders(self) -> bool {
        matches!(self, Part::PreRasterization | Part::FragmentShader)
    }

    /// The name its library's making record is begun with (`shader_cache/driver.rs`).
    pub(crate) fn kind(self) -> &'static str {
        match self {
            Part::VertexInput => "vertex input library",
            Part::PreRasterization => "pre-rasterization library",
            Part::FragmentShader => "fragment shader library",
            Part::FragmentOutput => "fragment output library",
        }
    }
}

/// One part's library: the device's, by its part's content, made once by the first caller that
/// needs it ([`Library::made`]).
pub(crate) struct Library {
    state: Mutex<Slot>,
    /// Signalled as a making ends.
    ended: Condvar,
}

/// Where a library's making is.
enum Slot {
    /// Not made: the next caller that needs it makes it.
    Waiting,
    /// A caller is making it.
    Making,
    Made(vk::Pipeline),
}

/// A caller's making of a library: where it ends without a library (an error, or a panic), the
/// library waits for the next caller again, and those waiting are woken.
struct Claim<'a>(&'a Library);

impl Drop for Claim<'_> {
    fn drop(&mut self) {
        let mut state = lock(&self.0.state);
        if matches!(*state, Slot::Making) {
            *state = Slot::Waiting;
        }
        drop(state);
        self.0.ended.notify_all();
    }
}

impl Library {
    fn new() -> Library {
        Library {
            state: Mutex::new(Slot::Waiting),
            ended: Condvar::new(),
        }
    }

    /// Whether it is made.
    pub(crate) fn is_made(&self) -> bool {
        matches!(*lock(&self.state), Slot::Made(_))
    }

    /// The library, made now by `make` where no caller made it or is making it, else the one
    /// another caller made, waited for. A failed making leaves it to the next caller, as one of
    /// a key's draw does (`making.rs`). A caller that waits for another's making holds no turn of
    /// the process's budget meanwhile (`budget.rs`, "no making waits on another while it holds a
    /// turn"): a pool's piece, or a program's thread, gives its thread's turn back for the wait
    /// and takes one again after it at the rank it was given at, before it makes the library or
    /// links its pipeline, so the callers of one fragment shader's library never hold the turns
    /// idle while one of them compiles it.
    pub(crate) fn made(
        &self,
        make: impl FnOnce() -> Result<vk::Pipeline, Error>,
    ) -> Result<vk::Pipeline, Error> {
        let mut state = lock(&self.state);
        loop {
            match *state {
                Slot::Made(raw) => return Ok(raw),
                Slot::Making => {
                    drop(state);
                    budget::without_turn(|| {
                        let mut state = lock(&self.state);
                        while matches!(*state, Slot::Making) {
                            state = self
                                .ended
                                .wait(state)
                                .unwrap_or_else(PoisonError::into_inner);
                        }
                    });
                    state = lock(&self.state);
                }
                Slot::Waiting => break,
            }
        }
        *state = Slot::Making;
        drop(state);
        let claim = Claim(self);
        let raw = make()?;
        *lock(&self.state) = Slot::Made(raw);
        drop(claim);
        Ok(raw)
    }
}

impl Owned<DeviceShared> for Library {
    fn destroy(&self, shared: &DeviceShared) {
        if let Slot::Made(raw) = *lock(&self.state) {
            // SAFETY: the map lets a library go only when nothing else holds it, so no pipeline
            // it linked is left (each holds it, and a command buffer holds the pipelines its
            // commands use until it completes) and none is being linked; or at the device's
            // teardown, when no variant, kept pipeline or command buffer is left.
            unsafe { release::object(shared.vk(), raw) };
        }
    }
}

/// What a linked pipeline's linking takes, exactly: the names of its four parts' libraries, in
/// [`Part::ALL`]'s order, and the pipeline layout it is linked with. The layout goes by its
/// handle, which its [`Link`] holds, so no other layout takes that handle while the link is in
/// the map.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct LinkKey {
    names: [Hash256; 4],
    layout: vk::PipelineLayout,
}

/// A linked pipeline: the device's, by its [`LinkKey`], made once by the first caller that needs
/// it ([`Library::made`], as a library is), holding what it was linked from while it lives.
pub(crate) struct Link {
    pipeline: Library,
    /// Its four libraries, in [`Part::ALL`]'s order.
    parts: [Arc<Library>; 4],
    /// The layout it is linked with: Vulkan may read a pipeline's creation's objects for as long
    /// as the pipeline lives.
    layout: Arc<SharedLayout>,
}

impl Owned<DeviceShared> for Link {
    fn destroy(&self, shared: &DeviceShared) {
        // As a library's: the map lets it go only when no variant's pipeline, kept pipeline or
        // command buffer holds it, or at the device's teardown.
        self.pipeline.destroy(shared);
    }
}

/// The linked pipeline of a variant's key, with the libraries it is linked from, held while the
/// variant's pipeline lives: the device's, which other variants of the same link share.
#[derive(Clone)]
pub(crate) struct Linked(Arc<Link>);

/// The device's libraries, by part and content, and its linked pipelines, by link
/// (`DeviceShared::libraries`).
#[derive(Default)]
pub(crate) struct Libraries {
    parts: [Mutex<ContentKeyed<Hash256, Library>>; 4],
    links: Mutex<ContentKeyed<LinkKey, Link>>,
}

impl Libraries {
    /// The libraries of the four parts `names` names (each its part's digest, in
    /// [`Part::ALL`]'s order), and the pipeline linked from them with `layout`: the device's,
    /// inserted where it holds none in a program's call (`content_keyed::program_call`), else a
    /// caller's own on a miss. Each map sweeps.
    pub(crate) fn claim(
        &self,
        shared: &DeviceShared,
        names: &[Hash256; 4],
        layout: &Arc<SharedLayout>,
    ) -> Linked {
        let program = program_call();
        let parts: [Arc<Library>; 4] = std::array::from_fn(|at| {
            let mut map = lock(&self.parts[at]);
            let (library, _) = match map.get_or_make(names[at], program, || {
                Ok::<Library, std::convert::Infallible>(Library::new())
            }) {
                Ok(found) => found,
                Err(never) => match never {},
            };
            let gone = map.sweep();
            drop(map);
            // Destroyed here, the map's lock released.
            gone.destroy(shared);
            library
        });
        let key = LinkKey {
            names: *names,
            layout: layout.layout(),
        };
        let mut map = lock(&self.links);
        let (link, _) = match map.get_or_make(key, program, || {
            Ok::<Link, std::convert::Infallible>(Link {
                pipeline: Library::new(),
                parts,
                layout: Arc::clone(layout),
            })
        }) {
            Ok(found) => found,
            Err(never) => match never {},
        };
        let gone = map.sweep();
        drop(map);
        gone.destroy(shared);
        Linked(link)
    }

    /// Let go of the linked pipelines nothing holds any more, then of the libraries no pipeline
    /// holds: in a program's call, as the device's modules are (`objects.rs`).
    pub(crate) fn settle(&self, shared: &DeviceShared) {
        let gone = lock(&self.links).settle();
        gone.destroy(shared);
        for part in &self.parts {
            let gone = lock(part).settle();
            gone.destroy(shared);
        }
    }

    /// Destroy every linked pipeline, then every library: the device's teardown, after the kept
    /// pipelines.
    pub(crate) fn drain(&self, shared: &DeviceShared) {
        let gone = lock(&self.links).drain();
        gone.destroy(shared);
        for part in &self.parts {
            let gone = lock(part).drain();
            gone.destroy(shared);
        }
    }
}

impl Linked {
    /// Its libraries, in [`Part::ALL`]'s order.
    pub(crate) fn parts(&self) -> &[Arc<Library>; 4] {
        &self.0.parts
    }

    /// Its linked pipeline, made once by the first caller that needs it ([`Library::made`]).
    pub(crate) fn pipeline(&self) -> &Library {
        &self.0.pipeline
    }

    /// The pipeline layout its pipeline is linked with.
    pub(crate) fn layout(&self) -> vk::PipelineLayout {
        self.0.layout.layout()
    }

    /// Whether the libraries of its shaders, its pre-rasterization and fragment shader parts,
    /// are made: a key whose shaders' libraries are made links on the caller's thread, where its
    /// vertex input and fragment output libraries, which hold no shader, are made in well under a
    /// millisecond if they are not made yet (0.0 ms each on an RTX 5080, the
    /// link 0.0 ms, the pre-rasterization library 3 to 7 ms, the fragment shader library the
    /// whole pipeline's 20 to 38 s).
    pub(crate) fn shaders_made(&self) -> bool {
        Part::ALL
            .into_iter()
            .filter(|part| part.holds_shaders())
            .all(|part| self.0.parts[part as usize].is_made())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::internal;
    use crate::exit_guard;
    use ash::vk::Handle;
    use std::sync::mpsc;
    use std::thread::Builder;
    use std::time::Duration;

    const WAIT: Duration = Duration::from_secs(10);

    fn raw(n: u64) -> vk::Pipeline {
        vk::Pipeline::from_raw(n)
    }

    #[test]
    fn a_library_is_made_once_by_its_first_caller() {
        let library = Library::new();
        assert!(!library.is_made());
        assert_eq!(library.made(|| Ok(raw(7))).unwrap(), raw(7));
        assert!(library.is_made());
        assert_eq!(
            library
                .made(|| panic!("a made library is made again"))
                .unwrap(),
            raw(7)
        );
    }

    #[test]
    fn a_failed_making_leaves_the_library_to_the_next_caller() {
        let library = Library::new();
        assert!(library
            .made(|| Err(internal("the driver refused")))
            .is_err());
        assert!(!library.is_made());
        assert_eq!(library.made(|| Ok(raw(9))).unwrap(), raw(9));
    }

    #[test]
    fn a_making_that_panics_leaves_the_library_to_the_next_caller() {
        let library = Library::new();
        std::thread::scope(|scope| {
            let library = &library;
            let maker = exit_guard::spawn_scoped(scope, Builder::new(), move || {
                library.made(|| panic!("a making panics"))
            })
            .unwrap();
            assert!(maker.join().is_err(), "the making panicked");
        });
        assert!(!library.is_made());
        assert_eq!(library.made(|| Ok(raw(11))).unwrap(), raw(11));
    }

    #[test]
    fn a_caller_waits_for_the_making_another_caller_began() {
        let library = Library::new();
        let (started, has_started) = mpsc::channel();
        let (finish, finishing) = mpsc::channel::<()>();
        let (returned, has_returned) = mpsc::channel();
        std::thread::scope(|scope| {
            let library = &library;
            let maker = exit_guard::spawn_scoped(scope, Builder::new(), move || {
                library.made(|| {
                    started.send(()).unwrap();
                    finishing.recv_timeout(WAIT).unwrap();
                    Ok(raw(13))
                })
            })
            .unwrap();
            has_started.recv_timeout(WAIT).unwrap();
            let waiter = exit_guard::spawn_scoped(scope, Builder::new(), move || {
                let made = library.made(|| panic!("a library being made is made again"));
                returned.send(()).unwrap();
                made
            })
            .unwrap();
            // The second caller is still waiting while the first one's making runs.
            assert!(has_returned
                .recv_timeout(Duration::from_millis(50))
                .is_err());
            finish.send(()).unwrap();
            assert_eq!(maker.join().unwrap().unwrap(), raw(13));
            assert_eq!(waiter.join().unwrap().unwrap(), raw(13));
        });
    }

    /// A caller that waits for the making another caller began holds no turn meanwhile: of a
    /// budget of two turns, the maker's and the waiter's, the waiter's goes back for the wait, a
    /// making asked for meanwhile takes it at once, and the waiter takes one again before it
    /// returns the library.
    #[test]
    fn a_caller_waiting_for_another_s_making_lends_its_turn() {
        use crate::pipeline::budget::{ticket, waited_turn, Budget};
        let turns = Arc::new(Budget::new(2, 2));
        let library = Library::new();
        let (started, has_started) = mpsc::channel();
        let (finish, finishing) = mpsc::channel::<()>();
        let (waiting, is_waiting) = mpsc::channel();
        std::thread::scope(|scope| {
            let (library, turns) = (&library, &turns);
            let maker = exit_guard::spawn_scoped(scope, Builder::new(), move || {
                let _turn = waited_turn(turns, ticket());
                library.made(|| {
                    started.send(()).unwrap();
                    finishing.recv_timeout(WAIT).unwrap();
                    Ok(raw(17))
                })
            })
            .unwrap();
            has_started.recv_timeout(WAIT).unwrap();
            let waiter = exit_guard::spawn_scoped(scope, Builder::new(), move || {
                let _turn = waited_turn(turns, ticket());
                waiting.send(()).unwrap();
                let made = library.made(|| panic!("a library being made is made again"));
                assert!(budget::holds_turn(), "the waiter took a turn again");
                made
            })
            .unwrap();
            is_waiting.recv_timeout(WAIT).unwrap();
            let deadline = std::time::Instant::now() + WAIT;
            while turns.census() != (1, 0) && std::time::Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(5));
            }
            assert_eq!(turns.census(), (1, 0), "the waiter gave its turn back");
            let meanwhile = waited_turn(turns, ticket());
            assert_eq!(
                turns.census(),
                (2, 0),
                "a making asked for meanwhile took it"
            );
            drop(meanwhile);
            finish.send(()).unwrap();
            assert_eq!(maker.join().unwrap().unwrap(), raw(17));
            assert_eq!(waiter.join().unwrap().unwrap(), raw(17));
        });
        assert_eq!(turns.census(), (0, 0));
    }
}
