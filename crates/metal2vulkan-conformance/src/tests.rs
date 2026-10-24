//! Tests of the suite's own machinery, with no device: a stub side stands in for a GPU, so the
//! transcript, the comparison, the justification rule, the verdicts, the recording's staleness and
//! the report are all checked where there is neither Metal nor Vulkan.

use crate::allowance::{
    accepted_differences, accepted_slug, Allowance, Justification, Loosening, Offered, Unjustified,
};
use crate::case::{
    call, made, Abort, Answer, Case, Failure, Handle, Input, MetalSource, ReRecord, ShaderSource,
    Side, Venue,
};
use crate::compare::compare;
use crate::diff::DivergenceKind;
use crate::record::{record, RecordError};
use crate::recording::{
    Answer as RecordedAnswer, Lookup, Outcome, Provenance, Recording, Validated, ORACLE_OS_VERSION,
};
use crate::replay::{replay_against, NOTICES_SITE};
use crate::report::Report;
use crate::transcript::{
    ByteLayout, Digest, Element, Gaps, Negative, Node, Observed, Region, Sites, Transcript, Value,
};
use crate::verdict::Verdict;
use metal2vulkan_runtime::{Degradation, MetalLevel};

/// A side that answers from a script rather than from a device, so the suite's own machinery is
/// testable with no GPU.
struct StubSide {
    /// The bytes `read_buffer` answers.
    output: Vec<u8>,
    /// The status `wait` answers.
    status: &'static str,
    /// The notice codes `announced` hands over, once.
    notices: Vec<String>,
    /// The calls that fail, each by its name, with the failure it gives. A list rather than one,
    /// because a case may record a refusal at a site it carries on past **and** stop at a later
    /// call the venue cannot make, and the two together are a shape the verdicts have to get
    /// right.
    refuse: Vec<(&'static str, Failure)>,
    /// The device name the side's venue answers, so a test can hold a venue-scoped ledger row
    /// against one venue and not another.
    device: &'static str,
    made: u32,
}

impl StubSide {
    fn new(output: Vec<u8>) -> Self {
        Self {
            output,
            status: "Completed",
            notices: Vec::new(),
            refuse: Vec::new(),
            device: "stub",
            made: 0,
        }
    }

    /// The same side on a venue of another name.
    fn on(mut self, device: &'static str) -> Self {
        self.device = device;
        self
    }

    fn announcing(mut self, codes: &[Degradation]) -> Self {
        self.notices = codes.iter().map(|kind| kind.code().to_owned()).collect();
        self
    }

    /// The same side announcing `codes`, as a case's own notice scope handed them over.
    fn announcing_codes(mut self, codes: Vec<String>) -> Self {
        self.notices = codes;
        self
    }

    fn refusing(mut self, call: &'static str, failure: Failure) -> Self {
        self.refuse.push((call, failure));
        self
    }

    /// The failure the side was told to give at `call`, if any.
    fn refusal(&self, call: &str) -> Option<Failure> {
        self.refuse
            .iter()
            .find(|(named, _)| *named == call)
            .map(|(_, failure)| failure.clone())
    }

    fn handle(&mut self) -> Handle {
        self.made += 1;
        Handle(self.made - 1)
    }
}

impl Side for StubSide {
    fn venue(&self) -> Venue {
        venue_named(self.device)
    }

    fn library(&mut self, _source: &ShaderSource<'_>) -> Answer<Handle> {
        match self.refusal("library") {
            Some(failure) => Err(failure),
            None => Ok(self.handle()),
        }
    }

    fn compute_pipeline(&mut self, _library: Handle, _entry: &str) -> Answer<Handle> {
        match self.refusal("pipeline") {
            Some(failure) => Err(failure),
            None => Ok(self.handle()),
        }
    }

    fn buffer(&mut self, _bytes: &[u8]) -> Answer<Handle> {
        Ok(self.handle())
    }

    fn zeroed_buffer(&mut self, _length: usize) -> Answer<Handle> {
        Ok(self.handle())
    }

    fn texture_2d(
        &mut self,
        _format: metal2vulkan::metal::PixelFormat,
        _size: [u32; 2],
        _row_pitch: u32,
        _bytes: &[u8],
    ) -> Answer<Handle> {
        match self.refusal("texture") {
            Some(failure) => Err(failure),
            None => Ok(self.handle()),
        }
    }

    fn read_texture(
        &mut self,
        _texture: Handle,
        size: [u32; 2],
        row_pitch: u32,
    ) -> Answer<Vec<u8>> {
        let mut bytes = self.output.clone();
        bytes.resize(row_pitch as usize * size[1] as usize, 0);
        Ok(bytes)
    }

    fn dispatch(
        &mut self,
        _pipeline: Handle,
        _bindings: &crate::case::Bindings<'_>,
        _threadgroups: [u32; 3],
        _threads: [u32; 3],
    ) -> Answer<Handle> {
        match self.refusal("dispatch") {
            Some(failure) => Err(failure),
            None => Ok(self.handle()),
        }
    }

    fn compute_pipeline_with_stage_input(
        &mut self,
        _library: Handle,
        _entry: &str,
        _stage_input: &crate::case::StageInput,
    ) -> Answer<Handle> {
        match self.refusal("pipeline") {
            Some(failure) => Err(failure),
            None => Ok(self.handle()),
        }
    }

    fn dispatch_threads(
        &mut self,
        _pipeline: Handle,
        _bindings: &crate::case::Bindings<'_>,
        _stage_in: &[crate::case::StageInRegion],
        _grid: [u32; 3],
        _threads: [u32; 3],
    ) -> Answer<Handle> {
        match self.refusal("dispatch") {
            Some(failure) => Err(failure),
            None => Ok(self.handle()),
        }
    }

    fn wait(&mut self, _command_buffer: Handle) -> Answer<Value> {
        Ok(Value::unit_variant(self.status))
    }

    /// A copy-and-fill program, which answers nothing of its own: the refusal a test scripts here
    /// is the interesting half, because a program is where one call stands for a whole list of
    /// steps and a refusal at one of them has to reach the verdict as a refusal.
    fn work(&mut self, _work: &crate::work::Work) -> Answer<crate::work::Done> {
        match self.refusal("work") {
            Some(failure) => Err(failure),
            None => Ok(crate::work::Done {
                reads: Vec::new(),
                statuses: Vec::new(),
            }),
        }
    }

    fn read_buffer(&mut self, _buffer: Handle, _offset: u64, length: usize) -> Answer<Vec<u8>> {
        let mut bytes = self.output.clone();
        bytes.resize(length, 0);
        Ok(bytes)
    }

    fn announced(&mut self) -> Vec<String> {
        std::mem::take(&mut self.notices)
    }
}

/// A case over one buffer of 32-bit words, the smallest thing that exercises the whole loop.
struct WordsCase {
    name: &'static str,
    words: u32,
    source: MetalSource,
    allowances: Vec<Allowance>,
    /// The input fingerprint the case states, so a test can make the inputs move.
    input: Vec<u8>,
    /// Whether the output site holds its digest alone, as a case ported from a probe that kept
    /// one does.
    digest_only: bool,
}

impl WordsCase {
    fn new(name: &'static str, words: u32) -> Self {
        Self {
            name,
            words,
            source: MetalSource::Run,
            allowances: Vec::new(),
            input: vec![0u8; 4],
            digest_only: false,
        }
    }

    fn layout(&self) -> ByteLayout {
        ByteLayout {
            region: Region::Buffer {
                offset: 0,
                length: u64::from(self.words) * 4,
            },
            element: Element::U32,
            row_offset: 0,
            row_elements: 2,
            row_pitch: 8,
            rows: self.words / 2,
            image_pitch: u32::from(u16::try_from(self.words).unwrap()) * 4,
            images: 1,
            gaps: Gaps::Ignored,
        }
    }

    /// What the output site observed: its bytes, or its digest alone where the case is
    /// digest-only.
    fn observed(&self, output: Vec<u8>) -> Observed {
        if self.digest_only {
            return Observed::absent()
                .digested(crate::transcript::Digest::sha256_prefix16(&output));
        }
        Observed::bytes(output, self.layout())
    }

    /// The transcript the case emits when its run answered `output` and its status was `status`.
    fn transcript(&self, output: Vec<u8>, status: &str, notices: &[&str]) -> Transcript {
        let mut sites = Sites::new();
        sites.push(crate::case::inputs_node(&self.inputs().unwrap()));
        sites.value("library", Value::ok(Value::Absent));
        sites.value("pipeline", Value::ok(Value::Absent));
        sites.value("buffer/0", Value::ok(Value::Absent));
        sites.value("dispatch", Value::ok(Value::Absent));
        sites.value("status", Value::ok(Value::unit_variant(status)));
        sites.value("read-back", Value::ok(Value::Absent));
        sites.push(Node::leaf("output", self.observed(output)));
        sites.notices(NOTICES_SITE, notices);
        sites.finish()
    }
}

impl Case for WordsCase {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        self.source.clone()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        Ok(vec![Input::bytes("buffer:0", &self.input)])
    }

    fn allowances(&self) -> Vec<Allowance> {
        self.allowances.clone()
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let source = ShaderSource {
            bitcode: &[],
            air_text: "",
            entry: "words",
        };
        let library = made(sites, "library", side.library(&source))?;
        let pipeline = made(sites, "pipeline", side.compute_pipeline(library, "words"))?;
        let buffer = made(
            sites,
            "buffer/0",
            side.zeroed_buffer(self.words as usize * 4),
        )?;
        let command_buffer = made(
            sites,
            "dispatch",
            side.dispatch(
                pipeline,
                &crate::case::Bindings::buffers(&[buffer]),
                [1, 1, 1],
                [self.words, 1, 1],
            ),
        )?;
        call(sites, "status", side.wait(command_buffer), Clone::clone)?;
        let bytes = call(
            sites,
            "read-back",
            side.read_buffer(buffer, 0, self.words as usize * 4),
            |_| Value::Absent,
        )?;
        sites.push(Node::leaf("output", self.observed(bytes)));
        Ok(())
    }
}

/// A case written for the known-gaps ledger's own tests: one site that a side can be told to
/// refuse **without stopping the case**, and one site whose value can move, so a test can hold a
/// refusal row, a divergence row, or both of one case at once.
///
/// Its own case rather than a real one, on purpose. A test over a real case can only hold the gaps
/// that case happens to have today, so the day the product fixes one the test keeps passing while
/// proving nothing — and the point of the ledger's second direction is that a row cannot outlive
/// its bug. This case owns its inputs and both of its outcomes, so each direction can be broken
/// deliberately and watched to fail.
struct LedgerCase {
    name: &'static str,
}

impl LedgerCase {
    fn new(name: &'static str) -> Self {
        Self { name }
    }

    /// The transcript a side that answered everything, with `output` at the output site, emits:
    /// Metal's answer for this case.
    fn metal(&self, output: u64) -> Transcript {
        let mut sites = Sites::new();
        sites.push(crate::case::inputs_node(&self.inputs().unwrap()));
        sites.value("library", Value::ok(Value::Absent));
        sites.value("texture", Value::ok(Value::Absent));
        sites.value("pipeline", Value::ok(Value::Absent));
        sites.value("buffer/0", Value::ok(Value::Absent));
        sites.value("dispatch", Value::ok(Value::Absent));
        sites.value("status", Value::ok(Value::unit_variant("Completed")));
        sites.value("read-back", Value::ok(Value::Absent));
        sites.value("output", Value::Uint(output));
        sites.notices(NOTICES_SITE, &[] as &[&str]);
        sites.finish()
    }
}

impl Case for LedgerCase {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        Ok(vec![Input::bytes("buffer:0", &[0u8; 4])])
    }

    fn allowances(&self) -> Vec<Allowance> {
        Vec::new()
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let source = ShaderSource {
            bitcode: &[],
            air_text: "",
            entry: "gap",
        };
        let library = made(sites, "library", side.library(&source))?;
        // Recorded and carried past, which is the whole point of the site: a refusal here is one
        // site of the transcript rather than an abort, so one run can hold a refusal and a
        // difference at the same time.
        let _ = call(
            sites,
            "texture",
            side.texture_2d(
                metal2vulkan::metal::PixelFormat::RGBA8Unorm,
                [1, 1],
                4,
                &[0u8; 4],
            ),
            |_| Value::Absent,
        );
        let pipeline = made(sites, "pipeline", side.compute_pipeline(library, "gap"))?;
        let buffer = made(sites, "buffer/0", side.zeroed_buffer(4))?;
        let command_buffer = made(
            sites,
            "dispatch",
            side.dispatch(
                pipeline,
                &crate::case::Bindings::buffers(&[buffer]),
                [1, 1, 1],
                [1, 1, 1],
            ),
        )?;
        call(sites, "status", side.wait(command_buffer), Clone::clone)?;
        let bytes = call(sites, "read-back", side.read_buffer(buffer, 0, 4), |_| {
            Value::Absent
        })?;
        let mut word = [0u8; 4];
        word.copy_from_slice(&bytes[..4]);
        sites.value("output", Value::Uint(u64::from(u32::from_le_bytes(word))));
        Ok(())
    }
}

/// A case whose whole work is one [`crate::work::Work`] program, which is the shape a refusal used
/// to disappear in: the program's many steps are one [`Side::work`] call, so a step's refusal
/// reaches the transcript through that one site.
struct ProgramCase {
    name: &'static str,
}

impl ProgramCase {
    fn new(name: &'static str) -> Self {
        Self { name }
    }

    /// The transcript a side that ran the program emits: Metal's answer for this case.
    fn metal(&self) -> Transcript {
        let mut sites = Sites::new();
        sites.push(crate::case::inputs_node(&self.inputs().unwrap()));
        sites.value("work", Value::ok(Value::Absent));
        sites.notices(NOTICES_SITE, &[] as &[&str]);
        sites.finish()
    }
}

impl Case for ProgramCase {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        Ok(vec![Input::bytes("calls", b"one program of no steps")])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let program = crate::work::Work::new();
        call(sites, "work", side.work(&program), |_| Value::Absent)?;
        Ok(())
    }
}

/// A refusal at one **step of a program** is [`Verdict::Refused`] at the program's site, and the
/// known-gaps ledger holds a row for it in both directions.
///
/// The gap this closes, which was live until this test: each work interpreter named the step a
/// refusal came from by wrapping a fields value **around** the call's `Err`. The comparison reads a
/// refusal out of one shape alone — Metal answered and we answered `Err` — so what it saw instead
/// was "Metal observed a variant, we observed a fields": a structural divergence. The refusal was
/// therefore invisible to the refused count *and* to the ledger, which covers a refusal by its site
/// and its typed answer and had neither. Two real rows could not be written down because of it.
///
/// Both halves are exercised over fabricated ledgers, as `stale_set_diff` is: with the committed
/// ledger holding no row of this shape, a test that only read a passing run could not reach the
/// second direction at all.
#[test]
fn a_refusal_at_a_step_of_a_program_is_refused_and_the_ledger_holds_it() {
    use crate::known_gaps::{against, Expectation, Expects, KnownGap, EVERY_VENUE};
    let case = ProgramCase::new("program-step-refusal");
    let recording = recording_of(&case, case.metal(), 3);
    // The failure a work interpreter gives: the runtime's typed error, said to have happened at
    // step 1 of the program, through the one annotation both interpreters share.
    let refused = Failure::named(
        "Error::NotImplemented",
        "the runtime does not implement MultisampledBufferCopy",
    )
    .at_step(1);
    let mut side = StubSide::new(Vec::new()).refusing("work", refused);
    let replayed = replay_against(&case, &mut side, &recording).unwrap();
    assert_eq!(replayed.verdict.word(), "refused", "{}", replayed.verdict);
    assert!(!replayed.verdict.is_compliant());
    let Verdict::Refused { at, .. } = &replayed.verdict else {
        panic!("{}", replayed.verdict);
    };
    let [printed] = at.as_slice() else {
        panic!("{at:?}");
    };
    let (site, answer) = printed.split_once(": ").expect("a site and an answer");
    let (site, answer) = (site.to_owned(), answer.to_owned());
    assert_eq!(site, "case/work");
    // The answer identifies the gap: the typed error the code path gave, and which step of the
    // program gave it. Neither is readable from a kind mismatch.
    assert!(answer.contains("Error::NotImplemented"), "{answer}");
    assert!(answer.contains("MultisampledBufferCopy"), "{answer}");
    assert!(answer.contains("step: 1"), "{answer}");

    // Direction one: a gap nothing has written down fails the run, and the row to add is printed
    // ready to paste rather than reconstructed by hand.
    let rows = vec![replayed];
    let found = against(&rows, &[]);
    let [unlisted @ Expectation::Unlisted { expects, .. }] = found.as_slice() else {
        panic!("{found:?}");
    };
    assert_eq!(*expects, Expects::Refused);
    let row = unlisted.row().expect("a row to paste");
    let fields: Vec<&str> = row.split('\t').collect();
    assert_eq!(fields.len(), 6, "{row}");
    assert_eq!(fields[0], "program-step-refusal");
    assert_eq!(fields[2], "refused");
    assert_eq!(fields[3], "case/work");

    // Written down exactly: nothing unexpected, and the run may exit 0 with the gap still printed
    // and still counted as a refusal.
    let listed = vec![KnownGap {
        case: "program-step-refusal".to_owned(),
        venue: EVERY_VENUE.to_owned(),
        expects: Expects::Refused,
        site: site.clone(),
        answer: answer.clone(),
        waiting_on: "the stub refuses the program's second step, which is the point of the stub"
            .to_owned(),
    }];
    assert_eq!(against(&rows, &listed), Vec::new());

    // Direction two, the one that keeps the file from rotting into a silencer: the same row
    // against a run that ran the program fails, and says the row must go.
    let mut ran = StubSide::new(Vec::new());
    let passed = replay_against(&case, &mut ran, &recording).unwrap();
    assert_eq!(passed.verdict.word(), "exact", "{}", passed.verdict);
    let found = against(std::slice::from_ref(&passed), &listed);
    let [gone @ Expectation::NoLongerHeld { gap, .. }] = found.as_slice() else {
        panic!("{found:?}");
    };
    assert_eq!(gap.case, "program-step-refusal");
    let printed = gone.to_string();
    assert!(printed.contains("no longer refuses"), "{printed}");
    assert!(printed.contains("drop the row"), "{printed}");

    // And the row's identity is the step as well as the site: the same program refusing at
    // another step is another gap, so the listed row no longer holds and the new one is unlisted.
    let elsewhere = Failure::named(
        "Error::NotImplemented",
        "the runtime does not implement MultisampledBufferCopy",
    )
    .at_step(2);
    let mut other = StubSide::new(Vec::new()).refusing("work", elsewhere);
    let moved = replay_against(&case, &mut other, &recording).unwrap();
    assert_eq!(moved.verdict.word(), "refused");
    let found = against(std::slice::from_ref(&moved), &listed);
    assert_eq!(found.len(), 2, "{found:?}");
    assert!(
        found
            .iter()
            .any(|why| matches!(why, Expectation::Unlisted { .. })),
        "{found:?}"
    );
    assert!(
        found
            .iter()
            .any(|why| matches!(why, Expectation::NoLongerHeld { .. })),
        "{found:?}"
    );
}

/// A case shaped like the families whose answer is one digest per format: it pushes the failing
/// site's node **itself**, and names its own [`Abort`] at a deeper path that says which row of the
/// family failed.
///
/// That is `texture_fills.rs`'s shape, and `chroma_pairs.rs`'s, and `sample_fills.rs`'s: a case
/// whose site is one node but whose rows are many has nothing else to put in the abort. The two
/// paths are therefore different on purpose, which is exactly why the venue's own gap must not be
/// filed under either of them.
struct TwoPathCase {
    name: &'static str,
}

impl TwoPathCase {
    /// The site the case pushes its one node at.
    const SITE: &'static str = "work";
    /// The path the case names in its abort: which row of the family stopped.
    const ROW: &'static str = "work/236/b1";

    fn new(name: &'static str) -> Self {
        Self { name }
    }

    /// Metal's answer: the program ran, and the family's digest stands at its own site.
    fn metal(&self) -> Transcript {
        let mut sites = Sites::new();
        sites.push(crate::case::inputs_node(&self.inputs().unwrap()));
        sites.value(Self::SITE, Value::ok(Value::Absent));
        sites.leaf(
            "format/236",
            Observed::absent().digested(Digest::sha256_prefix16(b"what Metal filled")),
        );
        sites.notices(NOTICES_SITE, &[] as &[&str]);
        sites.finish()
    }
}

impl Case for TwoPathCase {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        Ok(vec![Input::bytes("calls", b"one fill of format 236")])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let program = crate::work::Work::new();
        match side.work(&program) {
            Ok(_) => {
                sites.value(Self::SITE, Value::ok(Value::Absent));
                sites.leaf(
                    "format/236",
                    Observed::absent().digested(Digest::sha256_prefix16(b"what Metal filled")),
                );
                Ok(())
            }
            Err(failure) => {
                sites.push(Node::leaf(Self::SITE, Observed::value(failure.recorded())));
                Err(Abort {
                    at: Self::ROW.to_owned(),
                    failure,
                })
            }
        }
    }
}

/// A **venue's own gap** inside a program is `lacks-feature` even where the case names its abort
/// at a path no site holds, and what the runtime has **not built** still reads `refused` there.
///
/// The gap this closes: [`Failure::lacks`] reached the comparison only through
/// [`Sites::device_gap`](crate::transcript::Sites::device_gap), keyed by a path. Two callers file
/// it as the site is emitted ([`call`] and [`crate::case::probe`]); the third is
/// [`replay_against`], which files it at the path the case's own [`Abort`] names. A case that
/// pushes the node itself and names a deeper row in its abort therefore filed the fact at a site
/// that does not exist, and the refusal read as a gap of **ours** at the site that does.
///
/// Measured on the committed suite, which is what drove this: `texture-fills-236` answers
/// `Unsupported { feature: Format, needs: AnyOf([[Extension(ExtTextureCompressionAstcHdr)]]) }`
/// for `MTLPixelFormatASTC_12x12_HDR`, and the `astc-*-hdr` cases answer the same typed error
/// through [`call`]. One run scored the first `refused` and the second `lacks-feature`: the same
/// device gap, two verdicts, decided by which of two paths a case happened to write. A venue that
/// honestly lacks `VK_EXT_texture_compression_astc_hdr` is not a gap of ours, so the run exited 1
/// for nothing and two ledger rows would have been written for nothing.
///
/// Both directions, because the first alone is vacuous: a reader that answered "the venue lacks
/// it" for every refusal would pass it and launder every unbuilt form in the suite.
#[test]
fn a_venue_gap_inside_a_program_is_the_venues_however_the_case_names_its_abort() {
    use metal2vulkan::device::VkExtension;
    use metal2vulkan::error::{Capability, Feature, Unsupported};
    use metal2vulkan_runtime::Error;
    let case = TwoPathCase::new("two-path-program");
    let recording = recording_of(&case, case.metal(), 3);

    // The venue's own gap: the extension `MTLPixelFormatASTC_12x12_HDR` needs, refused while the
    // program's first resource was made, exactly as a work interpreter hands it over.
    let theirs = Failure::of(&Error::Unsupported(Unsupported::needs(
        Feature::Format,
        Capability::Extension(VkExtension::ExtTextureCompressionAstcHdr),
    )))
    .at_resource(0);
    assert!(
        theirs.lacks.is_some(),
        "the typed refusal names an extension a device could enable"
    );
    let mut side = StubSide::new(Vec::new()).refusing("work", theirs);
    let replayed = replay_against(&case, &mut side, &recording).unwrap();
    assert_eq!(
        replayed.verdict.word(),
        "lacks-feature",
        "{}",
        replayed.verdict
    );
    assert!(replayed.verdict.is_compliant());
    let Verdict::LacksFeature { needs } = &replayed.verdict else {
        panic!("{}", replayed.verdict);
    };
    // The site it names is the one the case pushed, not the row its abort named, and what the
    // venue lacks is readable from it.
    assert!(needs.contains("case/work:"), "{needs}");
    assert!(needs.contains("ExtTextureCompressionAstcHdr"), "{needs}");
    // And a gap of the venue's is not a gap of ours, so the ledger holds nothing against it.
    assert_eq!(
        crate::known_gaps::against(std::slice::from_ref(&replayed), &[]),
        Vec::new()
    );

    // The other direction: the same case, the same two paths, a form the runtime has not built.
    // That is ours to answer for and must still fail.
    let ours = Failure::of(&Error::Unsupported(Unsupported::not_implemented(
        Feature::Format,
    )))
    .at_resource(0);
    assert!(ours.lacks.is_none());
    let mut side = StubSide::new(Vec::new()).refusing("work", ours);
    let refused = replay_against(&case, &mut side, &recording).unwrap();
    assert_eq!(refused.verdict.word(), "refused", "{}", refused.verdict);
    assert!(!refused.verdict.is_compliant());
    let Verdict::Refused { at, .. } = &refused.verdict else {
        panic!("{}", refused.verdict);
    };
    let [printed] = at.as_slice() else {
        panic!("{at:?}");
    };
    assert!(printed.starts_with("case/work: "), "{printed}");
    assert!(printed.contains("resource: 0"), "{printed}");
    // And the run that made the program is still exact, so neither reading comes from the case
    // being unable to pass at all.
    let mut ran = StubSide::new(Vec::new());
    let passed = replay_against(&case, &mut ran, &recording).unwrap();
    assert_eq!(passed.verdict.word(), "exact", "{}", passed.verdict);
}

/// A failure records as the `Err` the comparison reads a refusal from, however it is annotated.
///
/// The invariant [`Failure`] exists to hold, checked rather than described: every way the suite
/// builds a failure, and the annotation on top of each, records a value
/// [`Value::is_error`](crate::transcript::Value::is_error) answers for. Without it a refusal
/// reaches the comparison as two sides observing different kinds of value, which is a divergence
/// and not a refusal, and the refused count and the ledger both lose it.
#[test]
fn every_failure_records_as_an_error_however_it_is_named() {
    let built = [
        (
            "Error::NotImplemented",
            Failure::of(&metal2vulkan_runtime::Error::NotImplemented(
                metal2vulkan_runtime::RuntimeGap::MultisampledBufferCopy,
            )),
        ),
        ("Unavailable", Failure::unavailable("work")),
        (
            "CommandBufferStatus",
            Failure::named("CommandBufferStatus", "the status was not Completed"),
        ),
    ];
    for (variant, failure) in built {
        let plain = failure.recorded();
        assert!(plain.is_error(), "{plain:?}");
        // The annotation names the step and the resource **inside** the error, so the layer the
        // comparison reads stays outermost however many times a failure is named.
        let step = failure.clone().at_step(3).recorded();
        assert!(step.is_error(), "{step:?}");
        let both = failure.at_resource(0).at_step(1).recorded();
        assert!(both.is_error(), "{both:?}");
        // And the typed error is still in what a ledger row would carry, which is what makes the
        // row identify the code path rather than the harness's wrapper.
        let printed = crate::diff::print(&step);
        assert!(printed.contains(variant), "{printed}");
        assert!(printed.contains("step: 3"), "{printed}");
    }
}

/// A case whose two sites are the two shapes a refusal used to vanish in: one whose Metal answer
/// is a **tree of facts**, whose own value is therefore absent, and one whose Metal answer is a
/// bare number.
///
/// Written in the shape the real cases are — a queue's slots, an event's order, a buffer-backed
/// texture's facts and a sparse heap's all record a tree under the site and a typed error at it —
/// so the test holds the pattern those cases use rather than one of its own.
struct FactsCase {
    name: &'static str,
}

impl FactsCase {
    fn new(name: &'static str) -> Self {
        Self { name }
    }

    /// The transcript a side that answered emits: Metal's answer for this case.
    fn metal(&self) -> Transcript {
        let mut sites = Sites::new();
        sites.push(crate::case::inputs_node(&self.inputs().unwrap()));
        sites.push(Self::tree());
        sites.value(COUNT, Value::Uint(7));
        sites.notices(NOTICES_SITE, &[] as &[&str]);
        sites.finish()
    }

    /// The site whose answer is a tree, with each fact at a distinct value so one taking
    /// another's cannot pass by coincidence. One builder, so neither leg can hold a tree the
    /// other does not.
    fn tree() -> Node {
        Node::branch(FACTS, Vec::new())
            .with(Node::leaf("width", Observed::value(Value::Uint(3))))
            .with(Node::leaf("height", Observed::value(Value::Uint(5))))
    }
}

/// The site whose answer is a tree.
const FACTS: &str = "facts";

/// The site whose answer is a bare number.
const COUNT: &str = "count";

impl Case for FactsCase {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        Ok(vec![Input::bytes("facts", b"one tree and one number")])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        match side.texture_2d(
            metal2vulkan::metal::PixelFormat::RGBA8Unorm,
            [1, 1],
            4,
            &[0u8; 4],
        ) {
            Ok(_) => {
                sites.push(Self::tree());
                sites.value(COUNT, Value::Uint(7));
            }
            Err(failure) => {
                // The hand-written shape the real cases use: the site carries our typed error, and
                // the run carries on rather than aborting.
                sites.push(Node::leaf(FACTS, Observed::value(failure.recorded())));
                sites.push(Node::leaf(COUNT, Observed::value(failure.recorded())));
            }
        }
        Ok(())
    }
}

/// A refusal is [`Verdict::Refused`] at a site whose Metal answer is a **tree of facts** and at one
/// whose answer is a bare number, and the ledger holds a row for each in both directions.
///
/// Two more shapes the same class of bug reached, found by auditing every path into a verdict after
/// the `Work` program's was fixed, and both are worse than the one that was reported:
///
/// - A **tree**'s own value is [`Value::Absent`](crate::transcript::Value::Absent), and the
///   comparison skips an absent value, so the typed error was dropped outright and the report held
///   one "only Metal observed this site" divergence per child instead. The refusal appeared nowhere
///   at all.
/// - A **bare number** read as two sides observing different kinds of value: a divergence whose
///   answer names neither the typed error nor anything a reader could act on.
///
/// Neither reached the refused count and neither could be written down, which is the same
/// consequence the program step's refusal had.
#[test]
fn a_refusal_where_metals_answer_is_a_tree_or_a_number_is_refused() {
    use crate::known_gaps::{against, Expectation, Expects, KnownGap, EVERY_VENUE};
    let case = FactsCase::new("facts-refusal");
    let recording = recording_of(&case, case.metal(), 3);
    let mut side = StubSide::new(Vec::new()).refusing(
        "texture",
        Failure::named("Error::Unsupported", "this venue makes no such texture"),
    );
    let replayed = replay_against(&case, &mut side, &recording).unwrap();
    assert_eq!(replayed.verdict.word(), "refused", "{}", replayed.verdict);
    let Verdict::Refused { at, .. } = &replayed.verdict else {
        panic!("{}", replayed.verdict);
    };
    // Both sites, and **only** the two sites: the tree's children are sites our side never
    // reached, so one divergence each would have buried the refusal that is the fact.
    let sites: Vec<&str> = at
        .iter()
        .map(|printed| printed.split_once(": ").expect("a site").0)
        .collect();
    assert_eq!(sites, vec!["case/facts", "case/count"], "{at:?}");
    let printed = replayed.verdict.to_string();
    assert!(!printed.contains("only Metal observed"), "{printed}");
    assert!(printed.contains("Error::Unsupported"), "{printed}");

    // Direction one: two gaps nothing has written down fail the run.
    let rows = vec![replayed];
    let found = against(&rows, &[]);
    assert_eq!(found.len(), 2, "{found:?}");
    let listed: Vec<KnownGap> = found
        .iter()
        .map(|why| {
            let Expectation::Unlisted {
                case,
                expects,
                site,
                answer,
                ..
            } = why
            else {
                panic!("{why:?}");
            };
            assert_eq!(*expects, Expects::Refused);
            KnownGap {
                case: case.clone(),
                venue: EVERY_VENUE.to_owned(),
                expects: *expects,
                site: site.clone(),
                answer: answer.clone(),
                waiting_on: "the stub makes no texture, which is the point of the stub".to_owned(),
            }
        })
        .collect();
    assert_eq!(against(&rows, &listed), Vec::new());

    // Direction two: against a run that made the texture, both rows read as gaps that look fixed.
    let mut made = StubSide::new(Vec::new());
    let passed = replay_against(&case, &mut made, &recording).unwrap();
    assert_eq!(passed.verdict.word(), "exact", "{}", passed.verdict);
    let found = against(std::slice::from_ref(&passed), &listed);
    assert_eq!(found.len(), 2, "{found:?}");
    for why in &found {
        assert!(matches!(why, Expectation::NoLongerHeld { .. }), "{found:?}");
        assert!(why.to_string().contains("drop the row"), "{why}");
    }
}

/// The words `values` are, as little-endian bytes.
fn words(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|word| word.to_le_bytes()).collect()
}

/// A recording of one deterministic answer.
fn recording_of(case: &dyn Case, transcript: Transcript, runs: u32) -> Recording {
    Recording::new(
        case.name(),
        Vec::new(),
        vec![RecordedAnswer {
            key: crate::recording::OracleKey::current(MetalLevel::NEWEST),
            provenance: Provenance::MetalRun,
            device: "the oracle".to_owned(),
            runs,
            outcomes: vec![Outcome { runs, transcript }],
            validated: Validated::Ran,
        }],
    )
}

#[test]
fn an_identical_run_is_exact() {
    let case = WordsCase::new("words", 4);
    let bytes = words(&[1, 2, 3, 4]);
    let recording = recording_of(&case, case.transcript(bytes.clone(), "Completed", &[]), 3);
    let mut side = StubSide::new(bytes);
    let replayed = replay_against(&case, &mut side, &recording).unwrap();
    assert_eq!(replayed.verdict, Verdict::Exact, "{}", replayed.verdict);
}

#[test]
fn one_word_moving_names_its_row_and_index() {
    let case = WordsCase::new("words", 4);
    let recording = recording_of(
        &case,
        case.transcript(words(&[1, 2, 3, 4]), "Completed", &[]),
        3,
    );
    let mut side = StubSide::new(words(&[1, 2, 9, 4]));
    let replayed = replay_against(&case, &mut side, &recording).unwrap();
    let Verdict::Divergent { found, .. } = &replayed.verdict else {
        panic!("not divergent: {}", replayed.verdict);
    };
    assert_eq!(found.len(), 1, "{found:?}");
    assert_eq!(found[0].path, "case/output");
    let DivergenceKind::Element {
        count,
        largest,
        first,
        ..
    } = &found[0].kind
    else {
        panic!("not an element: {:?}", found[0].kind);
    };
    assert_eq!((first.row, first.index, first.at), (1, 0, 8));
    // One element moved, by 6: the summary is the count and how far, which is what a known-gaps
    // row's answer holds and what makes it stable.
    assert_eq!(*count, 1);
    assert_eq!(*largest, Some(6.0));
    let summary = found[0].kind.to_string();
    assert!(summary.contains("1 element differs"), "{summary}");
    assert!(summary.contains("Metal 3"), "{summary}");
    // The walk position is a line of its own, so it is never part of the row's identity.
    assert!(!summary.contains("row 1"), "{summary}");
    let detail = found[0].kind.detail().unwrap();
    assert!(detail.contains("row 1"), "{detail}");
    // The message names the leaf, not two whole regions, and the report prints both lines.
    let printed = replayed.verdict.to_string();
    assert!(printed.contains("row 1"), "{printed}");
    assert!(printed.contains("Metal 3"), "{printed}");
}

#[test]
fn a_notice_we_announce_and_metal_does_not_is_divergent_until_it_is_named() {
    let kind = Degradation::ALL[0];
    let bytes = words(&[1, 2]);
    let plain = WordsCase::new("words", 2);
    let recording = recording_of(&plain, plain.transcript(bytes.clone(), "Completed", &[]), 3);

    // Unnamed, the notice is a difference nobody is accounting for.
    let mut side = StubSide::new(bytes.clone()).announcing(&[kind]);
    let replayed = replay_against(&plain, &mut side, &recording).unwrap();
    assert!(
        matches!(replayed.verdict, Verdict::Divergent { .. }),
        "{}",
        replayed.verdict
    );

    // Named, with the notice announced, it is announced rather than divergent.
    let named = WordsCase {
        allowances: vec![crate::replay::notice_allowance(kind.code())],
        ..WordsCase::new("words", 2)
    };
    let mut side = StubSide::new(bytes).announcing(&[kind]);
    let replayed = replay_against(&named, &mut side, &recording).unwrap();
    assert!(
        matches!(replayed.verdict, Verdict::Announced { .. }),
        "{}",
        replayed.verdict
    );
}

/// The suite's central claim, at the one site it is stated: a difference is a bug unless we
/// announced it. A notice declaration is an allowance at `case/notices/*` loosened `Unchecked`,
/// and the notices site holds the whole announced set as one value, so until the subset check
/// beside the walk existed **one declaration unchecked every other notice**. Measured at
/// `f2fc351534`: this case read `announced` with zero divergences while announcing two codes it
/// names nowhere, one of them unrelated to the one it declares.
#[test]
fn a_declared_notice_does_not_excuse_an_undeclared_one() {
    let declared = Degradation::CarriedDecodedTwin;
    let undeclared = [
        Degradation::DeviceTexelConversion,
        Degradation::NativeDivision,
    ];
    let bytes = words(&[1, 2]);
    let case = WordsCase {
        allowances: vec![crate::replay::notice_allowance(declared.code())],
        ..WordsCase::new("words", 2)
    };
    let recording = recording_of(&case, case.transcript(bytes.clone(), "Completed", &[]), 3);
    let mut side =
        StubSide::new(bytes.clone()).announcing(&[declared, undeclared[0], undeclared[1]]);
    let replayed = replay_against(&case, &mut side, &recording).unwrap();
    let Verdict::Divergent { found, .. } = &replayed.verdict else {
        panic!("not divergent: {}", replayed.verdict);
    };
    // One row per undeclared code, naming the code, and none for the declared one.
    let named: Vec<&str> = found
        .iter()
        .filter_map(|divergence| match &divergence.kind {
            DivergenceKind::UndeclaredNotice { code } => Some(code.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        named,
        vec![undeclared[0].code(), undeclared[1].code()],
        "{found:?}"
    );
    // The row says where it was announced and which notice it was, so a reader needs nothing else.
    let printed = replayed.verdict.to_string();
    assert!(printed.contains("case/notices"), "{printed}");
    assert!(
        printed.contains(undeclared[0].code()) && printed.contains(undeclared[1].code()),
        "{printed}"
    );

    // Declaring all three, the same run is announced: the check fires on what is undeclared and
    // on nothing else.
    let whole = WordsCase {
        allowances: [declared, undeclared[0], undeclared[1]]
            .iter()
            .map(|kind| crate::replay::notice_allowance(kind.code()))
            .collect(),
        ..WordsCase::new("words", 2)
    };
    let mut side = StubSide::new(bytes).announcing(&[declared, undeclared[0], undeclared[1]]);
    let replayed = replay_against(&whole, &mut side, &recording).unwrap();
    assert!(
        matches!(replayed.verdict, Verdict::Announced { .. }),
        "{}",
        replayed.verdict
    );
}

/// A case that names two notices separately must see **both** fire. Before the subset check the
/// walk stopped at the first covering allowance that held, so a case declaring two announced
/// notices read "3 named, 2 fired, 1 never fired" and its second declaration was permanently dead
/// -- which a stream met in live use while porting a buffer-backed texture lane. A declaration now
/// fires on its own code being announced.
#[test]
fn two_notice_declarations_both_fire_when_both_notices_are_announced() {
    let first = Degradation::ALL[0];
    let second = Degradation::ALL[1];
    let bytes = words(&[1, 2]);
    let case = WordsCase {
        allowances: vec![
            crate::replay::notice_allowance(first.code()),
            crate::replay::notice_allowance(second.code()),
        ],
        ..WordsCase::new("words", 2)
    };
    let recording = recording_of(&case, case.transcript(bytes.clone(), "Completed", &[]), 3);
    let mut side = StubSide::new(bytes).announcing(&[first, second]);
    let replayed = replay_against(&case, &mut side, &recording).unwrap();
    assert!(
        matches!(replayed.verdict, Verdict::Announced { .. }),
        "{}",
        replayed.verdict
    );
    assert_eq!(replayed.allowances.named, 2);
    assert_eq!(
        replayed.allowances.unfired,
        Vec::new(),
        "both declarations are the case's own claims and both notices were announced"
    );
    assert_eq!(replayed.allowances.fired(), 2);
}

/// The other half of keeping that counter load-bearing: a **tolerance** justified by a notice fires
/// only where a value needed it, whatever the notices site did. Announcing the code must not mark a
/// value tolerance used, or a tolerance nothing checked would read as checked.
#[test]
fn a_value_tolerance_does_not_fire_just_because_its_notice_was_announced() {
    let kind = Degradation::ALL[0];
    let bytes = words(&[1, 2]);
    let case = WordsCase {
        allowances: vec![
            crate::replay::notice_allowance(kind.code()),
            Allowance::new(
                "case/output",
                Loosening::Unchecked,
                Justification::notice(kind),
            ),
        ],
        ..WordsCase::new("words", 2)
    };
    let recording = recording_of(&case, case.transcript(bytes.clone(), "Completed", &[]), 3);
    // The notice is announced, and every byte matches, so nothing needed the value tolerance.
    let mut side = StubSide::new(bytes).announcing(&[kind]);
    let replayed = replay_against(&case, &mut side, &recording).unwrap();
    let unfired: Vec<&str> = replayed
        .allowances
        .unfired
        .iter()
        .map(|unfired| unfired.path.as_str())
        .collect();
    assert_eq!(unfired, vec!["case/output"], "{unfired:?}");
}

/// And the counter the defect was caught by stays load-bearing: a notice-level allowance whose code
/// no run announces still reads as never fired.
#[test]
fn a_notice_declaration_whose_notice_is_quiet_still_reads_as_never_fired() {
    let fired = Degradation::ALL[0];
    let quiet = Degradation::ALL[1];
    let bytes = words(&[1, 2]);
    let case = WordsCase {
        allowances: vec![
            crate::replay::notice_allowance(fired.code()),
            crate::replay::notice_allowance(quiet.code()),
        ],
        ..WordsCase::new("words", 2)
    };
    let recording = recording_of(&case, case.transcript(bytes.clone(), "Completed", &[]), 3);
    let mut side = StubSide::new(bytes).announcing(&[fired]);
    let replayed = replay_against(&case, &mut side, &recording).unwrap();
    assert_eq!(replayed.allowances.fired(), 1);
    let unfired = &replayed.allowances.unfired;
    assert_eq!(unfired.len(), 1, "{unfired:?}");
    assert!(
        unfired[0].justified_by.contains(quiet.code()),
        "{unfired:?}"
    );
}

/// `announced_notices` is `announced_at` totalled, so the rule is read in one place: the comparison
/// needs each notice's **site** to say where an undeclared one was announced, and the notice census
/// needs the totals. Worth its own test because blinding that walk fails nothing else in the crate's
/// no-device tests but the undeclared-notice case.
#[test]
fn the_notice_walk_names_its_site_and_totals_its_codes() {
    let mut sites = Sites::new();
    sites.notices(NOTICES_SITE, &["b-code", "a-code", "b-code"]);
    let transcript = sites.finish();
    let announced = transcript.announced_at();
    let seen: Vec<(&str, &str, u64)> = announced
        .iter()
        .map(|at| (at.path.as_str(), at.code.as_str(), at.times))
        .collect();
    assert_eq!(
        seen,
        vec![("case/notices", "a-code", 1), ("case/notices", "b-code", 2)],
        "{announced:?}"
    );
    assert_eq!(
        transcript.announced_notices(),
        [("a-code".to_owned(), 1), ("b-code".to_owned(), 2)]
            .into_iter()
            .collect()
    );
}

/// Two shapes the subset check must not fail, because both are honest. A case may name a code its
/// own run never announced -- a notice is announced once per device per kind -- and that is an
/// unfired declaration, not an undeclared notice. And a case that loosens a *value* by a notice's
/// code has named that code, with the same catalogue and announcement checks applied, so it needs
/// no second allowance saying so again.
#[test]
fn a_notice_named_anywhere_in_the_case_is_declared() {
    let at_the_site = Degradation::ALL[0];
    let at_a_value = Degradation::ALL[1];
    let quiet = Degradation::ALL[2];
    let case = WordsCase {
        allowances: vec![
            crate::replay::notice_allowance(at_the_site.code()),
            // Named by a tolerance at a value site, and nowhere else.
            Allowance::new(
                "case/output",
                Loosening::Unchecked,
                Justification::notice(at_a_value),
            ),
            // Named, and not announced by this run: unfired, never a divergence.
            crate::replay::notice_allowance(quiet.code()),
        ],
        ..WordsCase::new("words", 2)
    };
    let recording = recording_of(&case, case.transcript(words(&[1, 2]), "Completed", &[]), 3);
    let mut side = StubSide::new(words(&[1, 7])).announcing(&[at_the_site, at_a_value]);
    let replayed = replay_against(&case, &mut side, &recording).unwrap();
    assert!(
        matches!(replayed.verdict, Verdict::Announced { .. }),
        "{}",
        replayed.verdict
    );
    // And the declaration nothing needed is reported as unfired rather than silently kept.
    let unfired: Vec<&str> = replayed
        .allowances
        .unfired
        .iter()
        .map(|unfired| unfired.path.as_str())
        .collect();
    assert_eq!(
        unfired,
        vec![format!("case/{NOTICES_SITE}/*")],
        "{unfired:?}"
    );
}

#[test]
fn an_allowance_whose_notice_did_not_fire_fails_the_case() {
    let kind = Degradation::ALL[0];
    let case = WordsCase {
        allowances: vec![Allowance::new(
            "case/output",
            Loosening::Unchecked,
            Justification::notice(kind),
        )],
        ..WordsCase::new("words", 2)
    };
    let recording = recording_of(&case, case.transcript(words(&[1, 2]), "Completed", &[]), 3);
    // The bytes differ and the allowance would cover them, but no notice was announced.
    let mut side = StubSide::new(words(&[1, 7]));
    let replayed = replay_against(&case, &mut side, &recording).unwrap();
    let Verdict::Divergent { found, .. } = &replayed.verdict else {
        panic!("not divergent: {}", replayed.verdict);
    };
    let printed = format!("{found:?}");
    assert!(printed.contains("announced no"), "{printed}");
}

/// A notice is announced once per device per kind, so which of a case's notices its own run sees
/// depends on what ran before it. A case that names several therefore has to be justified by any
/// one of them that fired, not by whichever it listed first: the first ported batch failed two
/// cases this way, over digests that matched Metal exactly.
#[test]
fn any_one_of_a_sites_allowances_firing_justifies_the_difference() {
    let fired = Degradation::ALL[1];
    let quiet = Degradation::ALL[0];
    let bytes = words(&[1, 2]);
    let plain = WordsCase::new("words", 2);
    let recording = recording_of(&plain, plain.transcript(bytes.clone(), "Completed", &[]), 3);
    // The quiet kind is listed first, and only the other one was announced.
    let case = WordsCase {
        allowances: vec![
            crate::replay::notice_allowance(quiet.code()),
            crate::replay::notice_allowance(fired.code()),
        ],
        ..WordsCase::new("words", 2)
    };
    let mut side = StubSide::new(bytes).announcing(&[fired]);
    let replayed = replay_against(&case, &mut side, &recording).unwrap();
    assert!(
        matches!(replayed.verdict, Verdict::Announced { .. }),
        "{}",
        replayed.verdict
    );
}

/// A row whose Metal answer is a digest, on a device that announced it cannot match it, has to be
/// loosenable: the comparison checks the digest before anything else, so without consulting the
/// case's allowance there a digest-only site could never be loosened at all.
#[test]
fn a_digest_only_site_compares_under_its_allowance() {
    let kind = Degradation::ALL[0];
    let case = WordsCase {
        allowances: vec![
            Allowance::new(
                "case/output",
                Loosening::Unchecked,
                Justification::notice(kind),
            ),
            crate::replay::notice_allowance(kind.code()),
        ],
        digest_only: true,
        ..WordsCase::new("words", 2)
    };
    let recording = recording_of(&case, case.transcript(words(&[1, 2]), "Completed", &[]), 3);
    // Unnamed notice: the digest moving is a divergence.
    let mut side = StubSide::new(words(&[1, 7]));
    let replayed = replay_against(&case, &mut side, &recording).unwrap();
    assert!(
        matches!(replayed.verdict, Verdict::Divergent { .. }),
        "{}",
        replayed.verdict
    );
    // With the notice announced, the digest goes unchecked, named and counted.
    let mut side = StubSide::new(words(&[1, 7])).announcing(&[kind]);
    let replayed = replay_against(&case, &mut side, &recording).unwrap();
    assert!(
        matches!(replayed.verdict, Verdict::Announced { .. }),
        "{}",
        replayed.verdict
    );
}

#[test]
fn an_allowance_naming_no_catalogue_entry_fails_the_case() {
    for justification in [
        Justification::Degradation("metal2vulkan-degrade-no-such-kind".to_owned()),
        Justification::accepted("no-such-accepted-difference"),
    ] {
        let case = WordsCase {
            allowances: vec![Allowance::new(
                "case/output",
                Loosening::Unchecked,
                justification.clone(),
            )],
            ..WordsCase::new("words", 2)
        };
        let bytes = words(&[1, 2]);
        let recording = recording_of(&case, case.transcript(bytes.clone(), "Completed", &[]), 3);
        // Even an exactly matching run fails: the allowance's justification is gone.
        let mut side = StubSide::new(bytes);
        let replayed = replay_against(&case, &mut side, &recording).unwrap();
        assert!(
            matches!(replayed.verdict, Verdict::Divergent { .. }),
            "{justification:?}: {}",
            replayed.verdict
        );
    }
}

#[test]
fn an_accepted_difference_in_the_document_justifies_a_loosened_field() {
    let slug = accepted_differences()
        .first()
        .expect("docs/API.md section 10 holds entries")
        .clone();
    let case = WordsCase {
        allowances: vec![Allowance::new(
            "case/output",
            Loosening::AnyOf(vec![Value::Uint(7)]),
            Justification::accepted(slug),
        )],
        ..WordsCase::new("words", 2)
    };
    let recording = recording_of(&case, case.transcript(words(&[1, 2]), "Completed", &[]), 3);
    let mut side = StubSide::new(words(&[1, 7]));
    let replayed = replay_against(&case, &mut side, &recording).unwrap();
    assert!(
        matches!(replayed.verdict, Verdict::Announced { .. }),
        "{}",
        replayed.verdict
    );
}

#[test]
fn the_accepted_differences_are_read_from_the_document() {
    let entries = accepted_differences();
    assert!(entries.len() > 20, "{} entries", entries.len());
    assert!(entries.contains(&accepted_slug(
        "A command's read-back answers Metal's private getters"
    )));
    assert!(!entries.contains(&"nothing-of-the-sort".to_owned()));
}

/// An entry whose bold lead-in wraps onto the next line is an entry. Reading the first line alone
/// made those invisible, so an allowance naming one could never be justified. The first of these
/// wraps in the document today; the second, on one line, is the control.
#[test]
fn an_entry_whose_lead_in_wraps_is_read_whole() {
    let entries = accepted_differences();
    for lead_in in [
        "An object grid of more than 2^29 threadgroups fails its command buffer by its count, at \
         once",
        "A mesh grid of 1,048,577 to 2,097,151 threadgroups draws nothing",
    ] {
        let slug = accepted_slug(lead_in);
        assert!(entries.contains(&slug), "{slug} is not an entry");
    }
}

#[test]
fn a_typed_error_where_metal_answered_is_refused_not_divergent() {
    let case = WordsCase::new("words", 2);
    let recording = recording_of(&case, case.transcript(words(&[1, 2]), "Completed", &[]), 3);
    let mut side = StubSide::new(Vec::new()).refusing(
        "dispatch",
        Failure::named("Error::Invalid", "the encoder refused the dispatch"),
    );
    let replayed = replay_against(&case, &mut side, &recording).unwrap();
    let Verdict::Refused {
        at,
        metal_validation,
    } = &replayed.verdict
    else {
        panic!("not refused: {}", replayed.verdict);
    };
    assert_eq!(at.len(), 1, "{at:?}");
    assert!(at[0].starts_with("case/dispatch"), "{at:?}");
    // The recording's validated leg let the case run, so nothing excuses the refusal — and the
    // verdict says which of the three facts that is rather than leaving it to be guessed at.
    assert_eq!(*metal_validation, Validated::Ran.describe());
    let printed = replayed.verdict.to_string();
    assert!(
        printed.contains("plain Metal answered the call"),
        "{printed}"
    );
    assert!(printed.contains("let the case run"), "{printed}");
}

/// A refusal whose recording never took a validated leg says so, rather than reading as a case
/// Metal's validation let through. The two are different facts and the report must tell them
/// apart: one is a bug of ours, the other is a missing measurement.
#[test]
fn a_refusal_with_no_validated_leg_says_the_fact_is_missing() {
    let case = WordsCase::new("words", 2);
    let mut recording = recording_of(&case, case.transcript(words(&[1, 2]), "Completed", &[]), 3);
    recording.answers[0].validated = Validated::NotRun;
    let mut side = StubSide::new(Vec::new()).refusing(
        "dispatch",
        Failure::named("Error::Invalid", "the encoder refused the dispatch"),
    );
    let replayed = replay_against(&case, &mut side, &recording).unwrap();
    let Verdict::Refused {
        metal_validation, ..
    } = &replayed.verdict
    else {
        panic!("not refused: {}", replayed.verdict);
    };
    assert_eq!(*metal_validation, Validated::NotRun.describe());
    assert!(
        metal_validation.contains("never run under it"),
        "{metal_validation}"
    );
    // Never a pass, whatever the validated leg says.
    assert!(!replayed.verdict.is_compliant());
}

#[test]
fn a_venue_that_lacks_the_feature_is_its_own_outcome() {
    let case = WordsCase::new("words", 2);
    let recording = recording_of(&case, case.transcript(words(&[1, 2]), "Completed", &[]), 3);
    let mut lacking = Failure::named("Error::Unsupported", "no such capability");
    lacking.lacks = Some("the device enables no such capability".to_owned());
    let mut side = StubSide::new(Vec::new()).refusing("pipeline", lacking);
    let replayed = replay_against(&case, &mut side, &recording).unwrap();
    assert!(
        matches!(replayed.verdict, Verdict::LacksFeature { .. }),
        "{}",
        replayed.verdict
    );
    // It is neither a pass nor a failure, and it never counts as a divergence.
    assert!(replayed.verdict.is_compliant());
    assert_eq!(replayed.verdict.word(), "lacks-feature");
}

/// A case that records a refusal at a site it **carries on past**, and then stops at a call the
/// venue honestly cannot make.
///
/// This shape had no case in the tree, which is why the hole below went unreached: the only user
/// of [`crate::case::probe`] makes no abortable call after it, and every site that records a
/// failure by hand either runs to the end or aborts at once. So the shape is written here, where
/// it costs no device and cannot be refactored away by a case author.
///
/// `hand_built` chooses how the case says it stopped: through [`made`], which records the venue's
/// gap as it emits the site, or through an `Abort` the case builds itself, which 122 sites in the
/// suite do and which records no such thing. Both have to reach the same verdict.
struct StoppedCase {
    name: &'static str,
    hand_built: bool,
}

impl StoppedCase {
    fn new(name: &'static str, hand_built: bool) -> Self {
        Self { name, hand_built }
    }

    /// The transcript a side that answered everything emits: Metal's answer for this case.
    ///
    /// It holds a site **after** the one that stops our run, so a green verdict here also says
    /// the comparison was told the run stopped: without that, the site Metal alone reached would
    /// be a divergence and no verdict of this case could ever be `lacks-feature`.
    fn metal(&self) -> Transcript {
        let mut sites = Sites::new();
        sites.push(crate::case::inputs_node(&self.inputs().unwrap()));
        sites.value(SWEPT, Value::ok(Value::Uint(11)));
        sites.value("library", Value::ok(Value::Absent));
        sites.value(STOPS, Value::ok(Value::Absent));
        sites.value("after", Value::Uint(13));
        sites.notices(NOTICES_SITE, &[] as &[&str]);
        sites.finish()
    }
}

/// The site the case sweeps over and carries on past.
const SWEPT: &str = "swept";

/// The site the case stops at.
const STOPS: &str = "stops";

impl Case for StoppedCase {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        Ok(vec![Input::bytes("calls", b"one sweep and one stop")])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        // A sweep: one refused entry compares as a refusal at its own site and the run goes on.
        crate::case::probe(
            sites,
            SWEPT,
            side.texture_2d(
                metal2vulkan::metal::PixelFormat::RGBA8Unorm,
                [1, 1],
                4,
                &[0u8; 4],
            ),
            |_| Value::Uint(11),
        );
        let source = ShaderSource {
            bitcode: &[],
            air_text: "",
            entry: "stop",
        };
        let library = made(sites, "library", side.library(&source))?;
        if !self.hand_built {
            made(sites, STOPS, side.compute_pipeline(library, "stop"))?;
        } else {
            match side.compute_pipeline(library, "stop") {
                Ok(_) => sites.value(STOPS, Value::ok(Value::Absent)),
                Err(failure) => {
                    sites.push(Node::leaf(STOPS, Observed::value(failure.recorded())));
                    return Err(Abort {
                        at: STOPS.to_owned(),
                        failure,
                    });
                }
            }
        }
        sites.value("after", Value::Uint(13));
        Ok(())
    }
}

/// A refusal of **ours** at a site the run carried on past outranks the venue's own gap at the
/// site that stopped it, and the ledger can name it.
///
/// The hole this closes, which was the most severe form of a refusal going missing found in this
/// round: `replay` answered `LacksFeature` straight off the abort, **before** comparing anything,
/// so a refusal already recorded at an earlier site vanished behind a verdict that
/// [`Verdict::is_compliant`] answers true for. The run stayed green. `verdict.rs` states the
/// invariant it broke in as many words — "a divergence or a refusal anywhere else in the same case
/// still fails, which is what makes this safe to be compliant" — so this is also the test that
/// catches that claim being false.
///
/// Both ways a case can say it stopped are exercised, because they record different things: the
/// hand-built `Abort` records the typed error and not the fact that the venue is what lacks it,
/// and without `replay` recording that fact at the one place it reads the abort, this case's
/// stopping site would itself read as a refusal of ours.
#[test]
fn a_refusal_before_the_venues_own_gap_outranks_it() {
    use crate::known_gaps::{against, Expectation, Expects, KnownGap, EVERY_VENUE};
    let lacking = || {
        let mut failure = Failure::named("Error::Unsupported", "no pipeline of that shape");
        failure.lacks = Some("the device enables no such capability".to_owned());
        failure
    };
    for hand_built in [false, true] {
        let case = StoppedCase::new("stopped-after-a-refusal", hand_built);
        let recording = recording_of(&case, case.metal(), 3);

        // Only the venue's gap: the case is its own outcome, as it was before, and the allowances
        // it never reached are not reported as a blind spot.
        let mut lacks = StubSide::new(Vec::new()).refusing("pipeline", lacking());
        let only_lacking = replay_against(&case, &mut lacks, &recording).unwrap();
        assert_eq!(
            only_lacking.verdict.word(),
            "lacks-feature",
            "hand_built {hand_built}: {}",
            only_lacking.verdict
        );
        assert!(only_lacking.verdict.is_compliant());
        assert!(
            only_lacking
                .verdict
                .to_string()
                .contains("no such capability"),
            "{}",
            only_lacking.verdict
        );
        assert_eq!(only_lacking.allowances.unfired, Vec::new());

        // And with a refusal of ours at the site the run carried on past, that refusal is the
        // verdict: non-compliant, named at its own site, and the venue's gap still reported.
        let mut both = StubSide::new(Vec::new())
            .refusing(
                "texture",
                Failure::named("Error::Invalid", "the runtime refuses this texture"),
            )
            .refusing("pipeline", lacking());
        let replayed = replay_against(&case, &mut both, &recording).unwrap();
        assert_eq!(
            replayed.verdict.word(),
            "refused",
            "hand_built {hand_built}: {}",
            replayed.verdict
        );
        assert!(!replayed.verdict.is_compliant());
        let Verdict::Refused { at, .. } = &replayed.verdict else {
            panic!("{}", replayed.verdict);
        };
        let [printed] = at.as_slice() else {
            panic!("the swept site alone, not the stopping one: {at:?}");
        };
        let (site, answer) = printed.split_once(": ").expect("a site and an answer");
        let (site, answer) = (site.to_owned(), answer.to_owned());
        assert_eq!(site, "case/swept");
        assert!(answer.contains("Error::Invalid"), "{answer}");

        // The ledger can name it, in both directions, which is what being `Refused` buys.
        let rows = vec![replayed];
        let found = against(&rows, &[]);
        let [Expectation::Unlisted { expects, .. }] = found.as_slice() else {
            panic!("{found:?}");
        };
        assert_eq!(*expects, Expects::Refused);
        let listed = vec![KnownGap {
            case: "stopped-after-a-refusal".to_owned(),
            venue: EVERY_VENUE.to_owned(),
            expects: Expects::Refused,
            site: site.clone(),
            answer: answer.clone(),
            waiting_on: "the stub refuses the swept texture, which is the point of the stub"
                .to_owned(),
        }];
        assert_eq!(against(&rows, &listed), Vec::new());
        let found = against(std::slice::from_ref(&only_lacking), &listed);
        let [gone @ Expectation::NoLongerHeld { .. }] = found.as_slice() else {
            panic!("{found:?}");
        };
        assert!(gone.to_string().contains("drop the row"), "{gone}");
    }
}

#[test]
fn inputs_that_are_not_the_recordings_are_unrecorded_never_a_pass() {
    let recorded = WordsCase::new("words", 2);
    let bytes = words(&[1, 2]);
    let recording = recording_of(
        &recorded,
        recorded.transcript(bytes.clone(), "Completed", &[]),
        3,
    );
    let moved = WordsCase {
        input: vec![9u8; 4],
        ..WordsCase::new("words", 2)
    };
    let mut side = StubSide::new(bytes);
    let replayed = replay_against(&moved, &mut side, &recording).unwrap();
    let Verdict::Unrecorded { why } = &replayed.verdict else {
        panic!("not unrecorded: {}", replayed.verdict);
    };
    assert!(why.contains("buffer:0 differs"), "{why}");
    assert!(!replayed.verdict.is_compliant());
}

#[test]
fn a_recording_of_another_oracle_is_stale_and_fails() {
    let case = WordsCase::new("words", 2);
    let mut recording = recording_of(&case, case.transcript(words(&[1, 2]), "Completed", &[]), 3);
    recording.answers[0].key.os_build = "00A000".to_owned();
    let Err(Lookup::Stale(stale)) = recording.answer(MetalLevel::NEWEST) else {
        panic!("a recording of another build is not stale");
    };
    let printed = stale.to_string();
    assert!(printed.contains("ORACLE_OS_BUILD"), "{printed}");

    // And replay refuses it rather than resolving it to a verdict.
    let mut side = StubSide::new(words(&[1, 2]));
    let error = replay_against(&case, &mut side, &recording).unwrap_err();
    assert!(format!("{error}").contains("stale"), "{error}");
}

#[test]
fn a_level_with_no_answer_is_unrecorded_rather_than_stale() {
    let case = WordsCase::new("words", 2);
    let mut recording = recording_of(&case, case.transcript(words(&[1, 2]), "Completed", &[]), 3);
    recording.answers[0].key.metal_level = "26.7".to_owned();
    assert_eq!(
        recording.answer(MetalLevel::NEWEST),
        Err(Lookup::Unrecorded {
            level: MetalLevel::NEWEST.macos().to_owned()
        })
    );
}

#[test]
fn a_racy_recording_passes_when_the_run_matches_one_of_its_outcomes() {
    let case = WordsCase::new("words", 2);
    let mut recording = recording_of(&case, case.transcript(words(&[1, 2]), "Completed", &[]), 24);
    recording.answers[0].outcomes = vec![
        Outcome {
            runs: 23,
            transcript: case.transcript(words(&[1, 2]), "Completed", &[]),
        },
        Outcome {
            runs: 1,
            transcript: case.transcript(words(&[1, 3]), "Completed", &[]),
        },
    ];
    for seen in [words(&[1, 2]), words(&[1, 3])] {
        let mut side = StubSide::new(seen.clone());
        let replayed = replay_against(&case, &mut side, &recording).unwrap();
        assert_eq!(replayed.verdict, Verdict::Exact, "{seen:?}");
    }
    // A third answer is still a divergence.
    let mut side = StubSide::new(words(&[1, 4]));
    let replayed = replay_against(&case, &mut side, &recording).unwrap();
    assert!(
        matches!(replayed.verdict, Verdict::Divergent { .. }),
        "{}",
        replayed.verdict
    );
}

#[test]
fn record_mode_refuses_a_case_that_must_not_be_re_recorded() {
    let case = WordsCase {
        source: MetalSource::Carried {
            from: "a measured table",
            re_record: ReRecord::Forbidden {
                reason: "its Metal leg faults the GPU",
            },
            measured_on: crate::recording::ORACLE_CHIP,
        },
        ..WordsCase::new("words", 2)
    };
    let mut side = StubSide::new(words(&[1, 2]));
    let error = record(&case, &mut side, 1, MetalLevel::NEWEST, Validated::NotRun).unwrap_err();
    assert_eq!(
        error,
        RecordError::Forbidden {
            case: "words".to_owned(),
            reason: "its Metal leg faults the GPU".to_owned()
        }
    );
    assert!(format!("{error}").contains("must not be re-recorded"));
}

/// Record mode refuses a case whose **Metal** run stopped, rather than writing a recording of part
/// of it -- and still records the refusal a sweep carries past, which is Metal's answer.
///
/// `crate::case::call` aborts the whole case, so the sites after it are never emitted. Writing that
/// down would be the suite blessing a recording nobody took: on replay, every site past the stop
/// reads as one only our side observed, a divergence the case never asked about, and a recorded
/// Metal `Err` our side happens to match compares **exact** -- the case reporting as passing where
/// neither side did the work. Nothing of that is reachable today, because zero of the committed
/// recordings hold any `Err` at all, which is also why it had to be closed before a case reached
/// it rather than after.
///
/// `LedgerCase` carries both shapes, which is why it is the case here: it makes its library with
/// `made`, which stops the run, and its texture with a `call` whose refusal it carries past. So the
/// two directions are one case with one site moved, not two cases that could drift apart.
#[test]
fn record_mode_refuses_a_case_whose_metal_run_stopped() {
    let case = LedgerCase::new("ledger-record-stopped");
    let refusal = Failure::named("Error::Nil", "newLibraryWithData returned nil");
    // Off macOS there is no Metal to record from, and that check comes first: both legs say so,
    // so the test is honest about what it checked rather than appearing to check more.
    #[cfg(not(target_os = "macos"))]
    {
        let mut stops = StubSide::new(words(&[7])).refusing("library", refusal.clone());
        assert_eq!(
            record(&case, &mut stops, 1, MetalLevel::NEWEST, Validated::Ran).unwrap_err(),
            RecordError::NoMetal
        );
    }
    #[cfg(target_os = "macos")]
    {
        // A refusal at the site the case makes with `made`: the run stops, and no recording is
        // written. The site is named, and the refusal in the words a ledger row uses.
        let mut stops = StubSide::new(words(&[7])).refusing("library", refusal.clone());
        let error = record(&case, &mut stops, 1, MetalLevel::NEWEST, Validated::Ran).unwrap_err();
        let RecordError::Refused {
            case: named,
            at,
            refusal: said,
        } = &error
        else {
            panic!("a stopped Metal run must refuse the recording, not write one: {error:?}");
        };
        assert_eq!(named, case.name());
        assert_eq!(at, "library");
        assert!(said.contains("Error::Nil"), "{said}");
        assert!(
            format!("{error}").contains("would hold part of the case"),
            "{error}"
        );

        // The other direction, so this cannot become a blanket ban on recording a refusal: the
        // same case, the same refusal, at the site it carries past. The transcript is the whole
        // case with one query Metal answered nil for, which is Metal's answer, so it records.
        let mut carries = StubSide::new(words(&[7])).refusing("texture", refusal);
        let answer = record(&case, &mut carries, 1, MetalLevel::NEWEST, Validated::Ran)
            .expect("a refusal a sweep carries past is Metal's answer, not a truncated run");
        assert_eq!(answer.provenance, Provenance::MetalRun);
        let printed = format!("{:?}", answer.outcomes[0].transcript);
        assert!(
            printed.contains("Error::Nil"),
            "the recording holds the refusal at its own site: {printed}"
        );
        assert!(
            printed.contains("output"),
            "and every site after it, which is what makes it the case's answer: {printed}"
        );
    }
}

#[test]
fn record_mode_counts_the_outcomes_it_saw_with_their_runs() {
    // Off macOS there is no Metal to record from, and record mode says so rather than writing a
    // recording keyed to an oracle that is not here.
    #[cfg(not(target_os = "macos"))]
    {
        let case = WordsCase::new("words", 2);
        let mut side = StubSide::new(words(&[1, 2]));
        assert_eq!(
            record(&case, &mut side, 2, MetalLevel::NEWEST, Validated::Ran).unwrap_err(),
            RecordError::NoMetal
        );
    }
    #[cfg(target_os = "macos")]
    {
        let case = WordsCase::new("words", 2);
        let mut side = StubSide::new(words(&[1, 2]));
        let answer = record(&case, &mut side, 3, MetalLevel::NEWEST, Validated::Ran).unwrap();
        assert_eq!(answer.runs, 3);
        assert_eq!(
            answer.outcomes.len(),
            1,
            "a deterministic case: one outcome"
        );
        assert_eq!(answer.outcomes[0].runs, 3);
        assert_eq!(answer.provenance, Provenance::MetalRun);
    }
}

#[test]
fn a_set_diff_prints_what_is_missing_and_what_is_extra() {
    let metal = Transcript::new(vec![Node::leaf(
        "kinds",
        Observed::value(Value::set(["a", "b", "c"])),
    )]);
    let ours = Transcript::new(vec![Node::leaf(
        "kinds",
        Observed::value(Value::set(["b", "c", "d"])),
    )]);
    let found = compare(&metal, &ours, &[], Offered::notices(&[]));
    assert_eq!(found.divergences.len(), 1);
    let printed = found.divergences[0].to_string();
    assert!(printed.contains("2 in both"), "{printed}");
    assert!(printed.contains("missing {\"a\"}"), "{printed}");
    assert!(printed.contains("extra {\"d\"}"), "{printed}");
}

/// **A site name held twice under one parent is reported, on either side, rather than compared by
/// luck.**
///
/// `no_recorded_site_shares_its_name_with_a_sibling` holds the committed recordings to this, which
/// is Metal's half of it. Our side's transcript is built fresh at every replay and is written to no
/// recording, so that test cannot see it at all: a case whose *own* answer repeats a name would
/// have the repeat compared against nothing, with no error and no change of verdict, on every
/// venue forever. So the comparison itself refuses, which covers both sides and every run.
///
/// The repeat Metal holds and the repeat we hold are separate assertions, because they fail for
/// different reasons: the first is a recording nobody should have committed, the second is a case
/// measuring less than it claims right now.
#[test]
fn a_site_name_held_twice_under_one_parent_is_reported_on_either_side() {
    let twins = || {
        Transcript::new(vec![Node::branch(
            "pass",
            vec![
                Node::leaf("buffer", Observed::value(Value::Uint(1))),
                Node::leaf("buffer", Observed::value(Value::Uint(2))),
            ],
        )])
    };
    let single = || {
        Transcript::new(vec![Node::branch(
            "pass",
            vec![Node::leaf("buffer", Observed::value(Value::Uint(1)))],
        )])
    };

    // Neither side repeats: the comparison is of one pair and says nothing about names.
    let clean = compare(&single(), &single(), &[], Offered::notices(&[]));
    assert!(
        clean.divergences.is_empty(),
        "a transcript with no repeat is reported: {:?}",
        clean.divergences
    );

    for (who, metal, ours) in [
        (crate::diff::Side::Ours, single(), twins()),
        (crate::diff::Side::Metal, twins(), single()),
    ] {
        let found = compare(&metal, &ours, &[], Offered::notices(&[]));
        let repeats: Vec<_> = found
            .divergences
            .iter()
            .filter(|held| {
                matches!(
                    held.kind,
                    DivergenceKind::DuplicateSite { held_by, held: 2 } if held_by == who
                )
            })
            .collect();
        assert_eq!(
            repeats.len(),
            1,
            "{who} holding two `buffer` sites is reported {} time(s): {:?}",
            repeats.len(),
            found.divergences
        );
        assert_eq!(repeats[0].path, "case/pass/buffer");
        let printed = repeats[0].to_string();
        assert!(printed.contains("2 sites of this name"), "{printed}");
        // The two sides do not fail the same way, and the difference is why the report has to be
        // on both of them. The sites hold 1 and 2, and Metal's side is what the walk iterates:
        //
        // - the repeat is **ours**, so Metal's one site pairs with our first namesake, matches it,
        //   and our second observation is compared with nothing. Without the report above these
        //   transcripts compare clean while a value nobody read sits in the answer. The loss is
        //   total and silent, and the repeat is the whole of what is reported.
        // - the repeat is **Metal's**, so both its namesakes pair with our one site: one matches
        //   and the other reports a difference against an answer that was never its own. Nothing
        //   is lost, but what is reported is about a pair that does not exist.
        let expected = match who {
            crate::diff::Side::Ours => 1,
            crate::diff::Side::Metal => 2,
        };
        assert_eq!(
            found.divergences.len(),
            expected,
            "the repeat on {who} reported {} divergence(s), not {expected}: {:?}",
            found.divergences.len(),
            found.divergences
        );
    }

    // And no allowance may excuse it: a tolerance names a site by its path, and a path that names
    // two sites names nothing the allowance could be about.
    let loosened = compare(
        &single(),
        &twins(),
        &[Allowance::new(
            "pass/buffer",
            Loosening::Unchecked,
            Justification::accepted(
                accepted_differences()
                    .first()
                    .expect("docs/API.md section 10 holds entries")
                    .clone(),
            ),
        )],
        Offered::notices(&[]),
    );
    assert!(
        loosened
            .divergences
            .iter()
            .any(|held| matches!(held.kind, DivergenceKind::DuplicateSite { .. })),
        "an allowance at the path excused the repeat: {:?}",
        loosened.divergences
    );
}

#[test]
fn an_ordered_list_of_lines_names_the_line_that_moved() {
    let lines = |words: &[&str]| {
        Transcript::new(vec![Node::leaf(
            "printed",
            Observed::value(Value::Lines(words.iter().map(|w| w.to_string()).collect())),
        )])
    };
    let found = compare(
        &lines(&["one", "two", "three"]),
        &lines(&["one", "TWO", "three"]),
        &[],
        Offered::notices(&[]),
    );
    assert_eq!(found.divergences.len(), 1);
    // The summary says how many lines moved and, for one, what the two sides hold. The line's
    // index is a walk position, so it sits in the detail and not in the row's identity.
    let summary = found.divergences[0].kind.to_string();
    assert!(summary.contains("1 line differs"), "{summary}");
    assert!(summary.contains("\"TWO\""), "{summary}");
    assert!(!summary.contains("line 1"), "{summary}");
    let detail = found.divergences[0].kind.detail().unwrap();
    assert!(detail.contains("line 1"), "{detail}");

    // More than one line moving is a count, because the row's job is to say how much: the index
    // of the first is in the detail either way.
    let found = compare(
        &lines(&["one", "two", "three"]),
        &lines(&["ONE", "TWO", "three"]),
        &[],
        Offered::notices(&[]),
    );
    let summary = found.divergences[0].kind.to_string();
    assert_eq!(summary, "2 lines differ");
    assert!(
        found.divergences[0]
            .kind
            .detail()
            .unwrap()
            .contains("line 0"),
        "{:?}",
        found.divergences[0].kind
    );
}

#[test]
fn a_negative_fact_differs_from_a_value_rather_than_from_an_absent_field() {
    let metal = Transcript::new(vec![Node::leaf(
        "blocked",
        Observed::value(Value::Nothing(Negative::DidNotBlock)),
    )]);
    let ours = Transcript::new(vec![Node::leaf("blocked", Observed::value(Value::Uint(3)))]);
    let found = compare(&metal, &ours, &[], Offered::notices(&[]));
    assert_eq!(found.divergences.len(), 1);
    let DivergenceKind::Negative {
        metal,
        ours,
        observed,
    } = &found.divergences[0].kind
    else {
        panic!("{:?}", found.divergences[0]);
    };
    assert_eq!(*metal, Some(Negative::DidNotBlock));
    assert_eq!(*ours, None);
    // The value the other side observed is named, not left as "a value": at the notices site the
    // value IS the whole difference, and a known-gaps row's answer is this printed line.
    assert_eq!(observed.as_deref(), Some("3 (0x3)"));
    let printed = found.divergences[0].kind.to_string();
    assert!(printed.contains("we observed 3 (0x3)"), "{printed}");
}

#[test]
fn the_tree_descends_to_the_leaf_that_moved() {
    let tree = |leaf: Value| {
        Transcript::new(vec![Node::branch(
            "level-0",
            vec![Node::branch(
                "level-1",
                vec![Node::leaf("level-2", Observed::value(leaf))],
            )],
        )])
    };
    let found = compare(
        &tree(Value::Uint(1)),
        &tree(Value::Uint(2)),
        &[],
        Offered::notices(&[]),
    );
    assert_eq!(found.divergences.len(), 1);
    assert_eq!(found.divergences[0].path, "case/level-0/level-1/level-2");
}

#[test]
fn the_gaps_between_rows_compare_where_the_case_asks() {
    let layout = |gaps: Gaps| ByteLayout {
        region: Region::Buffer {
            offset: 0,
            length: 8,
        },
        element: Element::U8,
        row_offset: 0,
        row_elements: 2,
        row_pitch: 4,
        rows: 2,
        image_pitch: 8,
        images: 1,
        gaps,
    };
    let bytes = |pad: u8| vec![1, 2, pad, pad, 3, 4, pad, pad];
    for (gaps, divergences) in [(Gaps::Ignored, 0), (Gaps::Compared, 1)] {
        let metal = Transcript::new(vec![Node::leaf(
            "bytes",
            Observed::bytes(bytes(0xaa), layout(gaps)),
        )]);
        let ours = Transcript::new(vec![Node::leaf(
            "bytes",
            Observed::bytes(bytes(0x00), layout(gaps)),
        )]);
        let found = compare(&metal, &ours, &[], Offered::notices(&[]));
        assert_eq!(found.divergences.len(), divergences, "{gaps:?}");
    }
}

/// An allowance in units in the last place reaches a float **inside a byte column**.
///
/// This is the shape every exact-numbers case needs and the one easiest to assume is missing: a
/// case's answer is a buffer of `float`s, not a `Value::Float`, and `Loosening::Ulps` is written
/// against `Value::Float`. The comparison reaches it because it consults the allowance twice --
/// once over the whole `Value::Bytes`, where `Ulps` cannot hold and so falls through, and again
/// per element, where `Element::read` has turned the four bytes into a `Value::Float` of the
/// column's width. So the loosening works, and it works at the right granularity: the element
/// that moved further than the bound is still named.
///
/// The test pins all three facts, because each could regress on its own: one ulp inside a bound of
/// one is announced, two ulps is not, and the bound counts in the column's own width (a `F16`
/// column's ulp is not a `F32` column's).
#[test]
fn an_ulps_allowance_reaches_a_float_inside_a_byte_column() {
    let layout = |element: Element, elements: u32| ByteLayout {
        region: Region::Buffer {
            offset: 0,
            length: u64::from(elements * element.size()),
        },
        element,
        row_offset: 0,
        row_elements: elements,
        row_pitch: elements * element.size(),
        rows: 1,
        image_pitch: elements * element.size(),
        images: 1,
        gaps: Gaps::Ignored,
    };
    let site = |bytes: Vec<u8>, element: Element, elements: u32| {
        Transcript::new(vec![Node::leaf(
            "output",
            Observed::bytes(bytes, layout(element, elements)),
        )])
    };
    let words =
        |words: &[u32]| -> Vec<u8> { words.iter().flat_map(|word| word.to_le_bytes()).collect() };
    let allowance = |ulps: u32| {
        vec![Allowance::new(
            "case/output",
            Loosening::Ulps(ulps),
            Justification::Degradation(Degradation::DeviceTranscendentals.code().to_owned()),
        )]
    };
    let announced = [Degradation::DeviceTranscendentals.code().to_owned()];

    // One ulp of a `F32` column, inside a bound of one: announced, nothing divergent.
    let metal = site(words(&[0x3f80_0000, 0x4000_0000]), Element::F32, 2);
    let ours = site(words(&[0x3f80_0001, 0x4000_0000]), Element::F32, 2);
    let found = compare(&metal, &ours, &allowance(1), Offered::notices(&announced));
    assert!(
        found.divergences.is_empty(),
        "one ulp inside a bound of one: {:?}",
        found.divergences
    );
    assert_eq!(found.announced.len(), 1);
    assert_eq!(found.announced[0].path, "case/output");
    assert_eq!(found.announced[0].count, 1);

    // Two ulps against the same bound: the element that moved too far is still named.
    let far = site(words(&[0x3f80_0002, 0x4000_0000]), Element::F32, 2);
    let found = compare(&metal, &far, &allowance(1), Offered::notices(&announced));
    assert_eq!(found.divergences.len(), 1);
    assert_eq!(found.divergences[0].path, "case/output");
    assert!(
        matches!(
            found.divergences[0].kind,
            DivergenceKind::Element {
                element: Element::F32,
                count: 1,
                ..
            }
        ),
        "{:?}",
        found.divergences[0].kind
    );

    // The bound counts in the column's own width: the same two bytes are one ulp apart as a
    // `F16` column and so are announced, where as a `F32` column they are not comparable at all.
    let halves =
        |pair: [u16; 2]| -> Vec<u8> { pair.iter().flat_map(|half| half.to_le_bytes()).collect() };
    let metal_half = site(halves([0x3c00, 0x4000]), Element::F16, 2);
    let ours_half = site(halves([0x3c01, 0x4000]), Element::F16, 2);
    let found = compare(
        &metal_half,
        &ours_half,
        &allowance(1),
        Offered::notices(&announced),
    );
    assert!(
        found.divergences.is_empty(),
        "one ulp of a half column: {:?}",
        found.divergences
    );
}

#[test]
fn a_digest_only_recording_compares_by_its_digest() {
    let digested = |bytes: &[u8]| {
        Transcript::new(vec![Node::leaf(
            "output",
            Observed::absent().digested(Digest::sha256(bytes)),
        )])
    };
    let ours = |bytes: &[u8]| {
        Transcript::new(vec![Node::leaf(
            "output",
            Observed::value(Value::Bytes(bytes.to_vec())).digested(Digest::sha256(bytes)),
        )])
    };
    assert!(compare(
        &digested(b"same"),
        &ours(b"same"),
        &[],
        Offered::notices(&[])
    )
    .is_exact());
    let found = compare(
        &digested(b"same"),
        &ours(b"other"),
        &[],
        Offered::notices(&[]),
    );
    assert!(matches!(
        found.divergences.first().map(|d| &d.kind),
        Some(DivergenceKind::Digest { .. })
    ));
}

#[test]
fn the_report_counts_the_verdicts_and_lays_out_the_matrix() {
    let case = WordsCase::new("words", 2);
    let bytes = words(&[1, 2]);
    let recording = recording_of(&case, case.transcript(bytes.clone(), "Completed", &[]), 3);
    let mut report = Report::new();
    let mut exact = StubSide::new(bytes);
    report.add_result(replay_against(&case, &mut exact, &recording));
    let mut diverged = StubSide::new(words(&[1, 9]));
    report.add_result(replay_against(&case, &mut diverged, &recording));
    let counts = report.counts();
    assert_eq!(counts.get("exact"), Some(&1));
    assert_eq!(counts.get("divergent"), Some(&1));
    let rendered = report.render();
    assert!(rendered.contains("matrix"), "{rendered}");
    assert!(rendered.contains("words"), "{rendered}");
    assert!(!report.is_compliant());
    assert_eq!(report.failures().len(), 1);
}

#[test]
fn every_seed_case_states_its_inputs_and_its_metal_source() {
    for case in crate::cases::all() {
        let inputs = case
            .inputs()
            .unwrap_or_else(|why| panic!("{}: {why}", case.name()));
        assert!(
            !inputs.is_empty(),
            "{}: a case must prove it fed Metal's inputs",
            case.name()
        );
        // A carried answer says where it came from; a forbidden one says why it must stay carried.
        if let MetalSource::Carried { from, .. } = case.metal_source() {
            assert!(!from.is_empty(), "{}", case.name());
        }
        case.shaders()
            .unwrap_or_else(|why| panic!("{}: {why}", case.name()));
    }
}

/// A carried answer taken on another chip says so in its recording, in a field, and one taken on
/// the oracle says nothing extra.
///
/// Two families here hold an Apple M4's values under a key that names this M3 Max — the machine is
/// gone, so they cannot be re-taken. Before the field existed the fact lived only in the
/// provenance's prose, where no reader and no check could reach it. This test is what keeps the
/// field honest in both directions: a case that starts carrying another chip's values without
/// saying so fails here, and a case that claims another chip while the oracle took its values
/// fails too.
#[test]
fn a_carried_answer_taken_on_another_chip_names_it() {
    use crate::recording::{Provenance, APPLE_M4, ORACLE_CHIP};
    let mut elsewhere = Vec::new();
    for case in crate::cases::all() {
        let claimed = match case.metal_source() {
            MetalSource::Carried { measured_on, .. } => measured_on,
            MetalSource::Run | MetalSource::DidNotReturn { .. } => continue,
        };
        let recorded = crate::recording::read(case.name())
            .unwrap_or_else(|why| panic!("{}: {why}", case.name()))
            .answer(MetalLevel::NEWEST)
            .unwrap_or_else(|why| panic!("{}: {why}", case.name()))
            .provenance
            .clone();
        let Provenance::Carried { measured_on, .. } = &recorded else {
            panic!("{}: a carried case recorded as {recorded:?}", case.name());
        };
        assert_eq!(measured_on, claimed, "{}", case.name());
        if claimed != ORACLE_CHIP {
            elsewhere.push(case.name());
        }
        assert_eq!(
            recorded.measured_elsewhere(),
            (claimed != ORACLE_CHIP).then_some(claimed),
            "{}",
            case.name()
        );
    }
    // These families, and no others. One the list does not name would mean a table nobody noticed
    // came from another machine. Two families have left it, each because its Metal leg became
    // runnable here: `two_plane_heap_blits`, since nothing it does is Apple SPI, and
    // `ycbcr-minify-band-*`, since the Metal side now lays a texture over a real IOSurface
    // (`metal_surface.rs`). Both record on this chip, and the M4's digests the second carried are
    // kept as its cross-chip check.
    //
    // - `texture-fills-*`: the M4's fill digests, which `fillTexture:` being Apple SPI keeps
    //   carried.
    // - `compute-threadgroup-past-metals-limit`: the M4's answer for a threadgroup past Metal's
    //   1024, which stays carried because this round may not hand a GPU an over-limit
    //   threadgroup to ask again.
    // - `sample-positions-fewer-*`: the M4's coverage for fewer custom sample positions than the
    //   pass's samples, which stays carried because one position **past** the pass's samples is
    //   measured to end Metal's process, so a record run may not walk that family.
    const KNOWN: [&str; 3] = [
        "texture-fills-",
        crate::cases::compute_threadgroups::PAST_METALS_LIMIT,
        "sample-positions-fewer-",
    ];
    for family in KNOWN {
        assert!(
            elsewhere.iter().any(|name| name.starts_with(family)),
            "{family} carries the M4's values: {elsewhere:?}"
        );
    }
    for name in &elsewhere {
        assert!(
            KNOWN.iter().any(|family| name.starts_with(family)),
            "{name} carries another chip's values and is not one of the known families"
        );
    }
    // The file says the chip only where it differs: an oracle-taken recording holds no such field.
    let on_oracle = crate::recording::read(crate::cases::surface_fills::SURFACE_FILLS).unwrap();
    assert_eq!(
        on_oracle.answers[0].provenance.measured_elsewhere(),
        None,
        "a recording taken on the oracle names no other chip"
    );
    assert!(!serde_json::to_string(&on_oracle.answers[0].provenance)
        .unwrap()
        .contains(APPLE_M4));
}

#[test]
fn the_carried_table_of_the_private_type_getters_is_metals_measured_one() {
    let case = crate::cases::indirect_private_types::private_type_getters();
    let transcript = case.measured_transcript().unwrap();
    let answers = |raw: u64, form: &str| {
        let node = transcript
            .root
            .child(&format!("type/{raw}"))
            .unwrap_or_else(|| panic!("no type/{raw}"))
            .child(form)
            .unwrap_or_else(|| panic!("no type/{raw}/{form}"));
        let read = |name: &str| node.child(name).unwrap().observed.value.clone();
        (
            read("command_type"),
            read("answered_primitive_type"),
            read("recorded_primitive_type"),
        )
    };
    let some = |value: u64| Value::Variant {
        name: "Some".to_owned(),
        payload: Box::new(Value::Uint(value)),
    };
    // Types 0 to 6 read back as set; 7, 8 and 9 read back as points.
    assert_eq!(answers(6, "direct"), (Value::Uint(1), some(6), some(6)));
    assert_eq!(answers(7, "direct"), (Value::Uint(1), some(0), some(7)));
    assert_eq!(answers(8, "indexed"), (Value::Uint(2), some(0), some(8)));
    // The command type of an indexed draw of type 9 is none, and its record keeps type 9.
    assert_eq!(answers(9, "indexed"), (Value::Uint(0), some(0), some(9)));
    assert_eq!(answers(9, "direct"), (Value::Uint(1), some(0), some(9)));
}

/// The validated oracle's reader attributes a complaint to the call the marker before it named,
/// strips `NSLog`'s host-specific prefix, and drops the layer's own banner.
#[test]
fn the_validated_leg_reads_metals_own_words() {
    let said = format!(
        "2026-10-23 07:33:40.512 conformance[51249:39389017] {}\n\
         {}rate_map\n\
         2026-10-23 07:33:40.513 conformance[51249:39389017] RasterizationRateMap Layers \
         Validation\n\
         layerCount (0) must be >0 and <=2.\n\
         layers must not be nil.\n",
        "Metal API Validation Enabled",
        crate::validated::MARKER,
    );
    let read = crate::validated::read(&said).expect("a complaint");
    let Validated::Stopped { call, message } = &read else {
        panic!("{read:?}");
    };
    assert_eq!(call, "rate_map");
    // The complaint keeps Metal's words and none of this host's: no process id, no timestamp.
    assert_eq!(
        message,
        "RasterizationRateMap Layers Validation\nlayerCount (0) must be >0 and <=2.\n\
         layers must not be nil."
    );
    assert!(!message.contains("51249"), "{message}");
    // A leg the layer complained about nothing in is not a stop.
    assert_eq!(
        crate::validated::read(&format!(
            "2026-10-23 07:33:40.512 conformance[1:2] Metal API Validation Enabled\n{}rate_map\n",
            crate::validated::MARKER
        )),
        None
    );
}

/// A refusal the recording's validated oracle justifies holds, and the same allowance holds for
/// nothing else: not where the validated leg let the call run, not where it was never taken, and
/// not where Apple's check has moved.
#[test]
fn metals_validation_justifies_a_refusal_and_only_a_recorded_one() {
    let says = "screenSize.width (0) must be >0 and <=32768.";
    let justification = Justification::metal_validation(says);
    let stopped = Validated::Stopped {
        call: "rate_map".to_owned(),
        message: format!("RasterizationRateMap Layers Validation\n{says}"),
    };
    assert_eq!(
        justification.unjustified(&Offered::new(&[], &stopped)),
        None
    );
    assert_eq!(
        justification.unjustified(&Offered::new(&[], &Validated::Ran)),
        Some(Unjustified::MetalValidationRan)
    );
    // The one the whole axis turns on: an unexercised validated leg justifies nothing.
    assert_eq!(
        justification.unjustified(&Offered::notices(&[])),
        Some(Unjustified::MetalValidationNotRun)
    );
    let moved = Validated::Stopped {
        call: "rate_map".to_owned(),
        message: "screenSize.width (0) must be >0 and <=16384.".to_owned(),
    };
    assert!(matches!(
        justification.unjustified(&Offered::new(&[], &moved)),
        Some(Unjustified::MetalValidationSaysOtherwise { .. })
    ));
}

/// A refusal stays a refusal however well documented it is.
///
/// Metal's API validation stopping a call is not Metal refusing it: plain Metal answered, so the
/// row is `Refused` and is not compliant, and the justification buys the words in the report and
/// nothing else. Where the recording holds no such fact, the claim itself fails too.
#[test]
fn a_refusal_reads_by_what_metals_validation_did() {
    let says = "screenSize.width (0) must be >0 and <=32768.";
    let metal = Transcript::new(vec![Node::leaf(
        "map",
        Observed::value(Value::ok(Value::Absent)),
    )]);
    let ours = Transcript::new(vec![Node::leaf(
        "map",
        Observed::value(Value::error("Error::Invalid", Value::Absent)),
    )]);
    let allowances = vec![Allowance::new(
        "case/map",
        Loosening::Unchecked,
        Justification::metal_validation(says),
    )];
    let stopped = Validated::Stopped {
        call: "rate_map".to_owned(),
        message: says.to_owned(),
    };
    let found = crate::compare::compare_run(
        &metal,
        &ours,
        &allowances,
        Offered::new(&[], &stopped),
        true,
    );
    // The refusal is not laundered into an announced difference: it stays its own row.
    assert_eq!(found.refusals.len(), 1, "{found:?}");
    assert!(found.announced.is_empty(), "{found:?}");
    assert!(found.bad_allowances.is_empty(), "{found:?}");
    // What the justification bought: Metal's own words beside the row, and nothing else.
    assert_eq!(
        found.refusals[0].documented_by,
        Some(Justification::metal_validation(says).describe())
    );
    let verdict = Verdict::of(&found, &stopped);
    assert_eq!(verdict.word(), "refused");
    assert!(!verdict.is_compliant(), "a refusal is never compliant");
    let printed = verdict.to_string();
    assert!(printed.contains("documented, not excused"), "{printed}");
    assert!(printed.contains(says), "{printed}");

    // Without the validated leg the claim fails too, so the case says it has no explanation
    // rather than offering one nobody took.
    let found =
        crate::compare::compare_run(&metal, &ours, &allowances, Offered::notices(&[]), true);
    assert_eq!(found.bad_allowances.len(), 1, "{found:?}");
    assert_eq!(
        found.bad_allowances[0].why,
        Unjustified::MetalValidationNotRun
    );
    assert_eq!(found.refusals.len(), 1, "{found:?}");
    assert_eq!(found.refusals[0].documented_by, None);
    // A claim that does not hold fails the case on its own, ahead of the refusal it was made for.
    assert!(!Verdict::of(&found, &Validated::NotRun).is_compliant());
}

/// A refusal nothing documents is still exactly one refusal, and the words say plain Metal
/// answered. This is the shape the ported lanes will mostly meet.
#[test]
fn an_undocumented_refusal_is_one_row_and_names_no_excuse() {
    let metal = Transcript::new(vec![Node::leaf(
        "map",
        Observed::value(Value::ok(Value::Absent)),
    )]);
    let ours = Transcript::new(vec![Node::leaf(
        "map",
        Observed::value(Value::error("Error::Invalid", Value::Absent)),
    )]);
    let found =
        crate::compare::compare_run(&metal, &ours, &[], Offered::new(&[], &Validated::Ran), true);
    assert_eq!(found.refusals.len(), 1, "{found:?}");
    assert_eq!(found.refusals[0].documented_by, None);
    let verdict = Verdict::of(&found, &Validated::Ran);
    assert_eq!(verdict.word(), "refused");
    assert!(!verdict.is_compliant());
    let printed = verdict.to_string();
    assert!(
        printed.contains("plain Metal answered the call"),
        "{printed}"
    );
    assert!(!printed.contains("documented"), "{printed}");
}

/// The rate-map cases state the descriptor they feed, name the words Metal's validation used, and
/// keep the refusal's allowance to the call's own site.
#[test]
fn the_rate_map_cases_name_what_justifies_them() {
    let cases = crate::cases::rate_maps::all();
    let names: Vec<&str> = cases.iter().map(|case| case.name()).collect();
    assert_eq!(
        names,
        vec![
            crate::cases::rate_maps::NO_LAYERS,
            crate::cases::rate_maps::ZERO_SCREEN_WIDTH
        ]
    );
    for case in &cases {
        // Record mode runs both on Metal: neither can fault the GPU, because neither reaches it.
        assert_eq!(case.metal_source(), MetalSource::Run);
        assert!(case.metal_source().forbidden().is_none());
        // Every case states its own inputs, and the descriptor is what it feeds.
        let inputs = case.inputs().unwrap();
        assert!(
            inputs.iter().any(|input| input.name == "descriptor"),
            "{inputs:?}"
        );
        // Every allowance names something that exists -- the rate-map degradation, or the notice
        // a device with no unified memory announces for the `Shared` read-back buffer -- over the
        // notices site alone, so every answer of the map itself still compares exactly.
        for allowance in case.allowances() {
            match &allowance.justified_by {
                Justification::Degradation(code) => {
                    assert!(
                        code == Degradation::RasterizationRateMap.code()
                            || code == Degradation::SharedInSystemMemory.code(),
                        "{code}"
                    );
                    assert_eq!(allowance.path, "case/notices/*");
                }
                other => panic!("an unexpected tolerance: {other:?}"),
            }
        }
    }
    // The map of the identity announces no rate-map degradation and declares none; the one that is
    // not the identity announces it when the map is made. Both declare the venue's own
    // system-memory notice. Neither rests on the validated oracle any more: the zero-screen-width
    // refusal that did is closed.
    let codes = |case: &crate::cases::rate_maps::RateMapCase| -> Vec<String> {
        case.allowances()
            .iter()
            .filter_map(|allowance| match &allowance.justified_by {
                Justification::Degradation(code) => Some(code.clone()),
                _ => None,
            })
            .collect()
    };
    assert_eq!(
        codes(&cases[0]),
        vec![Degradation::SharedInSystemMemory.code().to_owned()]
    );
    assert_eq!(
        codes(&cases[1]),
        vec![
            Degradation::SharedInSystemMemory.code().to_owned(),
            Degradation::RasterizationRateMap.code().to_owned()
        ]
    );
    for case in &cases {
        assert!(
            !case.allowances().iter().any(|allowance| matches!(
                allowance.justified_by,
                Justification::MetalValidation { .. }
            )),
            "{}",
            case.name()
        );
    }
}

/// The one forwarding wrapper forwards **every** `Side` call, and this is what keeps it from
/// rotting.
///
/// `Side` gives most of its calls a default that answers `Failure::unavailable`, so a method the
/// wrapper forgets to forward makes `Announcing<MetalSide>` answer "no such call" where
/// `MetalSide` answers perfectly well — and the compiler says nothing, because the default is
/// there to be inherited. The trait grows with every ported batch, so the guard must be mechanical
/// rather than a list somebody remembers to extend.
///
/// Comparing transcripts through a stub does not do it: a stub that does not implement the method
/// either inherits the same default, so both sides agree and the check is vacuous. (Verified by
/// removing a forward and watching such a check pass.) So the guard reads the two sources and
/// compares the method names directly: a call `Side` declares and `Announcing` does not forward
/// fails here, naming it.
#[test]
fn the_announcing_side_forwards_every_call() {
    /// The names of the `fn` items `block` declares at one level of indentation.
    fn methods(source: &str, opening: &str) -> std::collections::BTreeSet<String> {
        let from = source
            .find(opening)
            .unwrap_or_else(|| panic!("no {opening} in the source"))
            + opening.len();
        let rest = &source[from..];
        // The block ends at the first `}` in the first column, which closes it.
        let to = rest.find("\n}\n").unwrap_or(rest.len());
        rest[..to]
            .lines()
            .filter_map(|line| {
                line.strip_prefix("    fn ")
                    .or(line.strip_prefix("    pub fn "))
            })
            .filter_map(|rest| rest.split('(').next())
            .map(|name| name.trim().to_owned())
            .collect()
    }
    let declared = methods(include_str!("case.rs"), "\npub trait Side {");
    // One forwarding impl serves every call-boundary watcher — the announcing one and the timing
    // one — so this guard covers both. Two hand-written wrappers would have been two places to
    // forget a method, and in the cost table a forgotten method reads as a *free* call.
    let forwarded = methods(
        include_str!("watch.rs"),
        "\nimpl<S: Side, W: Watcher> Side for Watching<S, W> {",
    );
    assert!(declared.len() > 10, "the trait was not found: {declared:?}");
    let missing: Vec<&String> = declared.difference(&forwarded).collect();
    assert!(
        missing.is_empty(),
        "watch::Watching does not forward {missing:?}, so it answers Unavailable where the \
         side it wraps answers, and a cost table would read each as a free call. Forward each of \
         them."
    );
    let extra: Vec<&String> = forwarded.difference(&declared).collect();
    assert!(
        extra.is_empty(),
        "watch::Watching forwards {extra:?}, which Side no longer declares"
    );
}

/// Wrapping a side changes nothing a case observes: the announcing wrapper is a marker on standard
/// error and nothing else. This catches a forward wired to the wrong inner call, which the
/// name-set check above cannot see.
#[test]
fn announcing_a_side_changes_no_transcript() {
    let transcript_of = |case: &dyn Case, announcing: bool| {
        let inputs = case.inputs().unwrap_or_default();
        let mut sites = Sites::new();
        sites.push(crate::case::inputs_node(&inputs));
        if announcing {
            let mut side = crate::validated::Announcing::new(StubSide::new(vec![0; 4096]));
            let _ = case.run(&mut side, &mut sites);
        } else {
            let mut side = StubSide::new(vec![0; 4096]);
            let _ = case.run(&mut side, &mut sites);
        }
        sites.finish()
    };
    for case in crate::cases::all() {
        assert_eq!(
            transcript_of(case.as_ref(), false),
            transcript_of(case.as_ref(), true),
            "{}: the announcing side answered otherwise than the side it wraps",
            case.name()
        );
    }
}

/// The committed recordings hold the second Metal oracle, and hold all three of its answers.
///
/// This ties the recordings to the cases: a re-record that lost the validated leg, or an Apple
/// release whose wording moved, fails here with no device and no Metal, instead of surfacing as a
/// puzzling verdict later. It is also the standing proof that the axis has been exercised — the
/// thing the suite could not say before these cases existed.
#[test]
fn the_committed_recordings_carry_all_three_validated_answers() {
    let validated = |case: &str| {
        crate::recording::read(case)
            .unwrap_or_else(|why| panic!("{case}: {why}"))
            .answer(MetalLevel::NEWEST)
            .unwrap_or_else(|why| panic!("{case}: {why}"))
            .validated
            .clone()
    };
    let says = |case: &str, words: &str| match validated(case) {
        Validated::Stopped { call, message } => {
            assert_eq!(call, "rate_map", "{case}");
            assert!(message.contains(words), "{case}: {message}");
        }
        other => panic!("{case}: {other:?}"),
    };
    // Stopped: Metal's own API validation refuses what plain Metal answered, in its own words.
    says(
        crate::cases::rate_maps::NO_LAYERS,
        crate::cases::rate_maps::NO_LAYERS_SAYS,
    );
    says(
        crate::cases::rate_maps::ZERO_SCREEN_WIDTH,
        crate::cases::rate_maps::ZERO_WIDTH_SAYS,
    );
    // Ran: the layer had nothing to say about a compute case, and the plain answer stood.
    for case in crate::cases::pass_bytes::all() {
        assert_eq!(validated(case.name()), Validated::Ran, "{}", case.name());
    }
    // NotRun: a carried answer was never run here at all, and the file says so rather than
    // implying Metal's validation was content with it.
    assert_eq!(
        validated(crate::cases::indirect_private_types::PRIVATE_TYPE_GETTERS),
        Validated::NotRun
    );
}

/// **No committed recording holds an address Metal printed.**
///
/// A validation complaint names the object it is about by its pointer
/// (`<AGXG15XFamilyBuffer: 0x7b81438e00>`), which is this run's heap and moves under ASLR, so a
/// recording that kept one could never be re-recorded byte for byte — and a re-record that changes
/// nothing is how a fingerprint refresh is told from a re-measurement. `crate::validated::read`
/// normalises them to `0xADDR`; this is what keeps "contained to six recordings" from quietly
/// becoming sixty, in this family or any other.
///
/// Every answer of every recording is read, not only the newest chip's, so an answer carried from
/// another chip cannot smuggle one in.
#[test]
fn no_recorded_message_holds_an_address() {
    /// `0x` and this many hexadecimal digits is an address, which is
    /// `crate::validated::ADDRESS_DIGITS`.
    const DIGITS: usize = 6;
    let holds_address = |text: &str| {
        let bytes = text.as_bytes();
        (0..bytes.len()).any(|at| {
            bytes[at] == b'0'
                && at + 1 < bytes.len()
                && bytes[at + 1] | 0x20 == b'x'
                && bytes[at + 2..]
                    .iter()
                    .take_while(|byte| byte.is_ascii_hexdigit())
                    .count()
                    >= DIGITS
        })
    };
    // The scan itself finds one, so a green run means the recordings are clean and not that the
    // check cannot fire.
    assert!(holds_address("<AGXG15XFamilyBuffer: 0x7b81438e00>"));
    assert!(!holds_address("length = 8, mask = 0xffff"));
    let mut held: Vec<String> = Vec::new();
    let mut messages = 0;
    for case in crate::cases::all() {
        let recording = crate::recording::read(case.name())
            .unwrap_or_else(|why| panic!("{}: {why}", case.name()));
        for answer in &recording.answers {
            let Validated::Stopped { call, message } = &answer.validated else {
                continue;
            };
            messages += 1;
            if holds_address(message) {
                held.push(format!("{}: {call}: {message}", case.name()));
            }
        }
    }
    assert!(
        held.is_empty(),
        "{} recording(s) keep an address Metal printed, which no re-record can reproduce:\n{}",
        held.len(),
        held.join("\n")
    );
    // And the walk really read the validated messages there are, so the emptiness above is a
    // result rather than a selection that found nothing to look at.
    assert!(messages > 0, "no recording holds a validation complaint");
}

/// **No site of a committed recording shares its name with a sibling.**
///
/// A path is how the comparison finds the answer to hold a site's answer against, so two children
/// of one node under one name is a pair the walk cannot settle. It does not fail loudly: it pairs
/// a site with whichever namesake it reaches, reports a difference of shape between two things
/// that were never the same observation, and never compares what the case was asking about. That
/// is what it did -- a case of this round named the call that made its buffer and the buffer's
/// bytes after the pass both `buffer`, and its first whole-case replay reported six cases
/// divergent at `case/buffer`, with Metal's bytes against our `Ok` of the make, having compared no
/// byte of either.
///
/// So the invariant is held here rather than left to each case's author, over every answer of
/// every recording: a name is unique among its siblings. It was measured before it was written,
/// across all 706 recordings, and those six sites were the only duplicates in the suite.
#[test]
fn no_recorded_site_shares_its_name_with_a_sibling() {
    fn duplicates(path: &str, node: &crate::transcript::Node, found: &mut Vec<String>) {
        let here = match path.is_empty() {
            true => node.name.to_string(),
            false => format!("{path}/{}", node.name),
        };
        // The same reading of the rule the comparison uses, so the two cannot drift: a recording
        // this accepts is one `compare` will not report a duplicate at, and the other way about.
        for (name, held) in node.repeated_children() {
            found.push(format!("{here}: {held} children named {name}"));
        }
        for child in &node.children {
            duplicates(&here, child, found);
        }
    }
    // The walk finds one where there is one, so a green run means the recordings are clean rather
    // than that the check cannot fire.
    let twins = Node::branch(
        "case",
        vec![
            Node::leaf("buffer", Observed::absent()),
            Node::leaf("buffer", Observed::absent()),
        ],
    );
    let mut control = Vec::new();
    duplicates("", &twins, &mut control);
    assert_eq!(
        control.len(),
        1,
        "the walk does not see a duplicate: {control:?}"
    );
    let mut found = Vec::new();
    let mut sites = 0;
    for case in crate::cases::all() {
        let recording = crate::recording::read(case.name())
            .unwrap_or_else(|why| panic!("{}: {why}", case.name()));
        for answer in &recording.answers {
            for outcome in &answer.outcomes {
                sites += 1;
                let mut here = Vec::new();
                duplicates("", &outcome.transcript.root, &mut here);
                for one in here {
                    found.push(format!("{}: {one}", case.name()));
                }
            }
        }
    }
    assert!(
        found.is_empty(),
        "{} site(s) share a name with a sibling, so the comparison pairs them by luck:\n{}",
        found.len(),
        found.join("\n")
    );
    // And the walk really read the recordings there are.
    assert!(sites > 0, "no recording holds a transcript");
}

/// The committed ledger parses, every row names a gap, and each row's case is a real case.
///
/// A row whose case has been renamed away would otherwise read as "no longer held" forever, or
/// never be held at all, and either way the row would stop meaning anything.
#[test]
fn the_known_gaps_ledger_parses_and_names_real_cases() {
    let ledger = crate::known_gaps::ledger().unwrap();
    assert!(
        !ledger.is_empty(),
        "the ledger is empty; drop the machinery"
    );
    for known in &ledger {
        assert!(
            crate::cases::named(&known.case).is_some(),
            "{}: no such case",
            known.case
        );
        assert!(known.site.starts_with("case/"), "{known:?}");
        assert!(!known.answer.is_empty(), "{known:?}");
        // A row says where it is expected, and says "everywhere" in words rather than by leaving
        // the field blank.
        assert!(!known.venue.is_empty(), "{known:?}");
        // A row is an open bug, so it says where the fix belongs.
        assert!(
            known.waiting_on.len() > 40,
            "{}: the gap is not described: {}",
            known.case,
            known.waiting_on
        );
    }
    // The rows this round left open, and what they are waiting on: `drawMeshThreads` whose
    // threads leave a partial object threadgroup, which Metal runs with the threads left over.
    let partial: Vec<&crate::known_gaps::KnownGap> = ledger
        .iter()
        .filter(|known| {
            known
                .case
                .starts_with("mesh-object-draws-partial-threadgroup-")
        })
        .collect();
    assert_eq!(partial.len(), 2, "{ledger:?}");
    for known in partial {
        assert_eq!(known.expects, crate::known_gaps::Expects::Refused);
        assert!(known.site.ends_with("drawMeshThreads"), "{known:?}");
        assert!(
            known.waiting_on.contains("partial object threadgroup"),
            "{}",
            known.waiting_on
        );
    }
    // The divergence rows this round left open: the fragment quad's placement on one lavapipe
    // build, which MoltenVK answers exactly, so the rows are venue-scoped rather than global.
    //
    // The scope is the build and not the implementation. These rows were measured in the gate
    // container's arm64 lavapipe, and the suite's other lavapipe -- x86-64, `LLVM 23.1.1`,
    // 256-bit vectors -- does not hold them. A venue field of `llvmpipe` is a substring of both
    // device names, so it held the rows on a venue the gap was never measured on, where the
    // retire direction then read all three as fixed. The two legs are told apart by the only thing
    // in a venue's description that differs, which is the device name `llvmpipe` writes itself,
    // so that is what the field carries.
    let quads: Vec<&crate::known_gaps::KnownGap> = ledger
        .iter()
        .filter(|known| known.case == "discard-quad-sum-placement")
        .collect();
    assert_eq!(quads.len(), 3, "{ledger:?}");
    let lavapipe = |device: &str| crate::case::Venue {
        device: device.to_owned(),
        driver: "vendor 0x10005 device 0x0000".to_owned(),
        api: "1.2".to_owned(),
        metal_level: "27.0.1".to_owned(),
    };
    let measured = lavapipe("llvmpipe (LLVM 21.1.8, 128 bits)");
    let other = lavapipe("llvmpipe (LLVM 23.1.1, 256 bits)");
    for known in &quads {
        assert!(known.covers_venue(&measured), "{known:?}");
        assert!(
            !known.covers_venue(&other),
            "the row holds on a lavapipe it was never measured on: {known:?}"
        );
    }
    for known in quads {
        assert_eq!(known.expects, crate::known_gaps::Expects::Divergent);
        assert_eq!(known.venue, "llvmpipe (LLVM 21.1.8, 128 bits)");
        // A divergence row says how much moved -- how many of the site's leaves differ and how
        // far -- and never where the walk first noticed, which is a property of the data and not
        // of the bug. All four texels of the kept quad differ, which the first-mismatch answer
        // this row used to carry hid entirely: it named one element and read as one difference.
        assert_eq!(
            known.answer, "4 elements differ (I32), the largest difference 1",
            "{known:?}"
        );
        // And it is honest about what is not established, which is the whole cause.
        assert!(known.waiting_on.contains("NOT established"), "{known:?}");
    }

    // Every row's site must be one its case's committed recording actually holds. A row for a
    // site that cannot exist is otherwise committable and silently unheld: one stream in this
    // round nearly added a row for `case/texels/argb8` of a case whose `argb8` leg refuses at the
    // pipeline and so produces no texels at all, and it was caught only because a hand count
    // disagreed with a parser's. This makes that near-miss impossible without a device.
    for known in &ledger {
        let recording = crate::recording::read(&known.case)
            .unwrap_or_else(|why| panic!("{}: {why}", known.case));
        let held = recording.answers.iter().any(|answer| {
            answer
                .outcomes
                .iter()
                .any(|outcome| site_exists(&outcome.transcript.root, "case", &known.site))
        });
        assert!(
            held,
            "{}: no site {} in the committed recording. A row's site is the one the report \
             prints, and a site the case never produces can never be held in either direction",
            known.case, known.site
        );
    }

    // The divergence rows this round left open over every venue: Metal's `tailSizeInBytes` on
    // extreme aspect ratios, where our model is short by whole 16384-byte pages. Scoped `*`
    // because the answer is CPU arithmetic over a descriptor, the same model on every device, and
    // it was verified identical on both venues.
    let tails: Vec<&crate::known_gaps::KnownGap> = ledger
        .iter()
        .filter(|known| known.case == crate::cases::sparse_heaps::TAILS_WIDE)
        .collect();
    assert_eq!(tails.len(), 13, "{ledger:?}");
    for known in tails {
        assert_eq!(known.expects, crate::known_gaps::Expects::Divergent);
        assert_eq!(known.venue, crate::known_gaps::EVERY_VENUE, "{known:?}");
        assert!(known.site.ends_with("/tail_size_in_bytes"), "{known:?}");
        // Each row carries its own two numbers, and the shared description carries the one thing
        // the thirteen of them establish together: the difference is whole pages, not a rounding.
        // Each side's first decimal number: the answer prints "Metal 32768 (0x8000), ours 16384
        // (0x4000)", so a naive digit scan would take the hex as the second value.
        let first_number = |text: &str| -> u64 {
            text.split_whitespace()
                .find_map(|word| word.trim_end_matches(',').parse::<u64>().ok())
                .unwrap_or_else(|| panic!("no number in {text:?}"))
        };
        let (metal, ours) = known
            .answer
            .split_once(", ours ")
            .map(|(metal, ours)| (first_number(metal), first_number(ours)))
            .unwrap_or_else(|| panic!("{known:?}"));
        assert!(metal > ours, "{known:?}");
        assert_eq!((metal - ours) % 16384, 0, "{known:?}");
    }

    // The zero-screen-width refusal is closed, so no row may name that case: the ledger holds
    // rows in both directions and a stale row would fail the run it is selected in.
    assert!(
        !ledger
            .iter()
            .any(|known| known.case == crate::cases::rate_maps::ZERO_SCREEN_WIDTH),
        "the zero-screen-width gap is closed; its row must be gone"
    );
}

/// A malformed ledger is refused rather than read loosely: a row must carry all six fields, must
/// name a verdict the ledger can carry, must say which venue it holds on, and must name the gap it
/// waits on.
#[test]
fn a_ledger_row_missing_a_field_is_refused() {
    let parse = crate::known_gaps::parse;
    assert!(parse("# just a comment\n\n").unwrap().is_empty());
    let row =
        "a-case\t*\trefused\tcase/map\tErr(Error::Invalid)\tthe runtime refuses what Metal does";
    assert_eq!(parse(row).unwrap().len(), 1);
    let why = parse("a-case\t*\trefused\tcase/map\tErr(Error::Invalid)\t").unwrap_err();
    assert!(why.contains("names no gap"), "{why}");
    let why = parse("a-case\t*\trefused\tcase/map").unwrap_err();
    assert!(why.contains("six"), "{why}");
    // An unrecorded case and a stale recording are never written down, so no other word parses.
    let why = parse("a-case\t*\tunrecorded\tcase/map\tnothing\tsome gap").unwrap_err();
    assert!(why.contains("refused"), "{why}");
    assert!(why.contains("divergent"), "{why}");
    // "Everywhere" is said in words. An empty field is a row nobody scoped, which would read as a
    // venue pattern matching nothing.
    let why = parse("a-case\t\trefused\tcase/map\tErr(Error::Invalid)\tsome gap").unwrap_err();
    assert!(why.contains("names no venue"), "{why}");
    let row = parse("a-case\tllvmpipe\tdivergent\tcase/out\tMetal 1, ours 2\ta gap of ours")
        .unwrap()
        .remove(0);
    assert_eq!(row.expects, crate::known_gaps::Expects::Divergent);
    assert_eq!(row.venue, "llvmpipe");
}

/// Whether `want` is the path of `node` or of anything under it, with `node` sitting at `at`.
///
/// The path is built the way the comparison's walk builds it, and for the same reason it has to
/// descend into the **value** as well as into the children: a node's name may itself carry
/// slashes, and a site may lie inside a variant (`.../Ok`) or a field (`.../tail_size_in_bytes`),
/// which is where the sparse tails' sites live. A guard that only walked the nodes would have
/// rejected thirteen legitimate rows -- it did, which is how this was found.
fn site_exists(node: &Node, at: &str, want: &str) -> bool {
    if at == want {
        return true;
    }
    if value_holds(&node.observed.value, at, want) {
        return true;
    }
    node.children
        .iter()
        .any(|child| site_exists(child, &format!("{at}/{}", child.name), want))
}

/// [`site_exists`] through a value: a variant adds its name, a field its key, a list element its
/// index, exactly as `compare`'s walk does.
fn value_holds(value: &Value, at: &str, want: &str) -> bool {
    if at == want {
        return true;
    }
    match value {
        Value::Variant { name, payload } => value_holds(payload, &format!("{at}/{name}"), want),
        Value::Fields(fields) => fields
            .iter()
            .any(|(name, held)| value_holds(held, &format!("{at}/{name}"), want)),
        Value::List(values) => values
            .iter()
            .enumerate()
            .any(|(index, held)| value_holds(held, &format!("{at}/{index}"), want)),
        _ => false,
    }
}

/// A venue, for a test that holds a row against one venue and not another.
fn venue_named(device: &str) -> Venue {
    Venue {
        device: device.to_owned(),
        driver: "stub".to_owned(),
        api: "none".to_owned(),
        metal_level: ORACLE_OS_VERSION.to_owned(),
    }
}

/// The ledger is an expectation held in both directions, and it changes no verdict.
#[test]
fn the_ledger_fails_on_a_new_refusal_and_on_one_that_is_gone() {
    use crate::known_gaps::{against, Expectation, Expects, KnownGap, EVERY_VENUE};
    let case = WordsCase::new("words", 2);
    let recording = recording_of(&case, case.transcript(words(&[1, 2]), "Completed", &[]), 3);
    let mut side = StubSide::new(Vec::new()).refusing(
        "dispatch",
        Failure::named("Error::Invalid", "the encoder refused the dispatch"),
    );
    let replayed = replay_against(&case, &mut side, &recording).unwrap();
    // The verdict is untouched by any ledger: a refusal stays a refusal and stays non-compliant.
    assert_eq!(replayed.verdict.word(), "refused");
    assert!(!replayed.verdict.is_compliant());
    let rows = vec![replayed];

    // Nothing written down: the refusal is a new gap and fails the run.
    let found = against(&rows, &[]);
    assert_eq!(found.len(), 1, "{found:?}");
    let Expectation::Unlisted {
        case: named,
        expects,
        site,
        answer,
        ..
    } = &found[0]
    else {
        panic!("{found:?}");
    };
    assert_eq!(named, "words");
    assert_eq!(*expects, Expects::Refused);
    assert_eq!(site, "case/dispatch");

    // Written down exactly: nothing unexpected, so the run may exit 0.
    let listed = vec![KnownGap {
        case: "words".to_owned(),
        venue: EVERY_VENUE.to_owned(),
        expects: Expects::Refused,
        site: site.clone(),
        answer: answer.clone(),
        waiting_on: "the stub refuses the dispatch, which is the point of the stub".to_owned(),
    }];
    assert_eq!(against(&rows, &listed), Vec::new());

    // The row's identity is the verdict it expects as well as the site and the answer: the same
    // site written down as a divergence does not cover a refusal.
    let mut as_divergence = listed.clone();
    as_divergence[0].expects = Expects::Divergent;
    let found = against(&rows, &as_divergence);
    assert_eq!(found.len(), 2, "{found:?}");

    // The direction that keeps the file from rotting: the same ledger against a run that did NOT
    // refuse fails, and says the row must go.
    let mut clean = StubSide::new(words(&[1, 2]));
    let passed = replay_against(&case, &mut clean, &recording).unwrap();
    assert_eq!(passed.verdict.word(), "exact");
    let found = against(std::slice::from_ref(&passed), &listed);
    assert_eq!(found.len(), 1, "{found:?}");
    let Expectation::NoLongerHeld { gap, .. } = &found[0] else {
        panic!("{found:?}");
    };
    assert_eq!(gap.case, "words");
    let printed = found[0].to_string();
    assert!(printed.contains("no longer refuses"), "{printed}");
    assert!(printed.contains("drop the row"), "{printed}");

    // A row is held only against the cases a run selected, so replaying one case does not read
    // every other row as fixed.
    let elsewhere = vec![KnownGap {
        case: "another-case".to_owned(),
        venue: EVERY_VENUE.to_owned(),
        expects: Expects::Refused,
        site: "case/map".to_owned(),
        answer: "Err(Error::Invalid)".to_owned(),
        waiting_on: "a gap of some other case, which this run did not select".to_owned(),
    }];
    assert_eq!(
        against(&rows, &elsewhere).len(),
        1,
        "the unlisted one alone"
    );

    // An unrecorded case is never a ledger row: only a refusal and a divergence are.
    let stale = crate::replay::Replayed {
        case: "words".to_owned(),
        venue: venue_named("stub"),
        verdict: Verdict::Unrecorded {
            why: "no Metal answer at Metal level 99".to_owned(),
        },
        allowances: Default::default(),
        measured_on: None,
        ours: Sites::new().finish(),
    };
    let found = against(std::slice::from_ref(&stale), &[]);
    assert!(
        matches!(found.as_slice(), [Expectation::NotLedgerable { .. }]),
        "{found:?}"
    );

    // Nor does an unrecorded case read its own rows as fixed. It compared nothing, so the run
    // holds no opinion about them: the only thing it says is that the case is not ledgerable.
    // Reading them as fixed is what reported two live rows as retirable on a host whose `llvm-as`
    // was not the recording Mac's.
    let found = against(std::slice::from_ref(&stale), &listed);
    assert!(
        matches!(found.as_slice(), [Expectation::NotLedgerable { .. }]),
        "an unrecorded case retires nothing: {found:?}"
    );
}

/// A row that names a venue is held on that venue, in **both** directions, and nowhere else.
///
/// The gap this closes: a row was global, so a case that plain Metal answers and we refuse on one
/// venue for a structural reason — while the same pipeline translates on another — could not be
/// written down at all. The row was needed by one run and failed the other for existing.
///
/// It is held on a case of this test's own ([`LedgerCase`]) rather than on a real one, because a
/// test over a real case can only hold the gaps that case happens to have today: the day the
/// product fixes one, such a test stops proving anything without saying so.
#[test]
fn a_venue_scoped_row_is_held_on_its_venue_and_nowhere_else() {
    use crate::known_gaps::{against, Expectation, Expects, KnownGap};
    let case = LedgerCase::new("ledger-venue-scope");
    let recording = recording_of(&case, case.metal(7), 3);
    let refusal = Failure::named("Error::Invalid", "this venue's pipeline is refused");

    // One venue refuses; the other runs the same case exactly.
    let mut refusing = StubSide::new(words(&[7]))
        .on("Woven Vulkan")
        .refusing("texture", refusal);
    let refused = replay_against(&case, &mut refusing, &recording).unwrap();
    assert_eq!(refused.verdict.word(), "refused");
    let mut elsewhere = StubSide::new(words(&[7])).on("llvmpipe (LLVM 19.1.7, 128 bits)");
    let exact = replay_against(&case, &mut elsewhere, &recording).unwrap();
    assert_eq!(exact.verdict.word(), "exact");

    let Verdict::Refused { at, .. } = &refused.verdict else {
        panic!("{:?}", refused.verdict);
    };
    let (site, answer) = at[0].split_once(": ").unwrap();
    let row = KnownGap {
        case: case.name().to_owned(),
        venue: "woven".to_owned(),
        expects: Expects::Refused,
        site: site.to_owned(),
        answer: answer.to_owned(),
        waiting_on: "the venue's own structural gap, written down for this test alone".to_owned(),
    };
    let listed = vec![row.clone()];

    // On the venue it names, the row covers the refusal and the run may exit 0.
    assert_eq!(against(std::slice::from_ref(&refused), &listed), Vec::new());
    // On the other venue, the row is not held at all — neither direction — so a run there is clean
    // rather than failing because another venue's gap is written down.
    assert_eq!(against(std::slice::from_ref(&exact), &listed), Vec::new());

    // BOTH directions on the venue it names. The row must not become a way to silence a venue, so
    // a run on `Woven Vulkan` that stops refusing fails and names the gap the row waited on.
    let mut fixed = StubSide::new(words(&[7])).on("Woven Vulkan");
    let now_exact = replay_against(&case, &mut fixed, &recording).unwrap();
    assert_eq!(now_exact.verdict.word(), "exact");
    let found = against(std::slice::from_ref(&now_exact), &listed);
    let [Expectation::NoLongerHeld { gap, venue }] = found.as_slice() else {
        panic!("{found:?}");
    };
    assert_eq!(gap.venue, "woven");
    assert_eq!(venue, "Woven Vulkan");

    // And the other direction on the venue it does not name: the refusal appearing there is a new
    // gap, because this row says nothing about that venue.
    let mut other = StubSide::new(words(&[7]))
        .on("llvmpipe (LLVM 19.1.7, 128 bits)")
        .refusing(
            "texture",
            Failure::named("Error::Invalid", "this venue's pipeline is refused"),
        );
    let refused_elsewhere = replay_against(&case, &mut other, &recording).unwrap();
    let found = against(std::slice::from_ref(&refused_elsewhere), &listed);
    assert!(
        matches!(found.as_slice(), [Expectation::Unlisted { .. }]),
        "{found:?}"
    );

    // A row naming no venue is the STRONGER claim, not the looser one: it is then held on every
    // venue the run selected the case on, so the venue that does not refuse fails it.
    let mut global = listed.clone();
    global[0].venue = crate::known_gaps::EVERY_VENUE.to_owned();
    let rows = vec![refused.clone(), exact.clone()];
    let found = against(&rows, &global);
    let [Expectation::NoLongerHeld { venue, .. }] = found.as_slice() else {
        panic!("{found:?}");
    };
    assert_eq!(venue, "llvmpipe (LLVM 19.1.7, 128 bits)");

    // A venue pattern that matches nothing holds the row nowhere, so its gap reads as unlisted and
    // the run fails: a typo is loud rather than silent.
    let mut typo = listed.clone();
    typo[0].venue = "wovenn".to_owned();
    let found = against(std::slice::from_ref(&refused), &typo);
    assert!(
        matches!(found.as_slice(), [Expectation::Unlisted { .. }]),
        "{found:?}"
    );
}

/// A divergence is a ledger row of its own, held in **both** directions, and it is still printed
/// and still counted as `Divergent`.
///
/// The gap this closes: the ledger covered refusals alone, so a known product bug that shows as a
/// difference rather than as a typed error had nowhere to live and made the whole-set replay exit
/// non-zero with no entry possible.
#[test]
fn a_divergence_row_is_held_in_both_directions() {
    use crate::known_gaps::{against, Expectation, Expects, KnownGap, EVERY_VENUE};
    let case = LedgerCase::new("ledger-divergence");
    let recording = recording_of(&case, case.metal(7), 3);

    let mut moved = StubSide::new(words(&[9]));
    let diverged = replay_against(&case, &mut moved, &recording).unwrap();
    // The ledger changes no verdict: the row is still `Divergent` and still non-compliant, so it
    // is still printed and still counted in the suite's own numbers.
    assert_eq!(diverged.verdict.word(), "divergent");
    assert!(!diverged.verdict.is_compliant());
    let Verdict::Divergent { found, .. } = &diverged.verdict else {
        panic!("{:?}", diverged.verdict);
    };
    let [divergence] = found.as_slice() else {
        panic!("{found:?}");
    };
    // The row carries our answer and Metal's, because the leaf fact that moved names both.
    let answer = divergence.kind.to_string();
    assert!(answer.contains("Metal 7"), "{answer}");
    assert!(answer.contains("ours 9"), "{answer}");

    // Unlisted, a divergence fails the run, and the report says which verdict it was.
    let found = against(std::slice::from_ref(&diverged), &[]);
    let [unlisted @ Expectation::Unlisted { expects, .. }] = found.as_slice() else {
        panic!("{found:?}");
    };
    assert_eq!(*expects, Expects::Divergent);
    assert!(unlisted.to_string().contains("divergent"), "{unlisted}");

    let listed = vec![KnownGap {
        case: case.name().to_owned(),
        venue: EVERY_VENUE.to_owned(),
        expects: Expects::Divergent,
        site: divergence.path.clone(),
        answer: answer.clone(),
        waiting_on: "a difference this test writes down on purpose, to prove the row is held"
            .to_owned(),
    }];
    assert_eq!(
        against(std::slice::from_ref(&diverged), &listed),
        Vec::new()
    );

    // The answer is part of the row's identity, so a difference that moved to another value is a
    // new gap rather than this one.
    let mut again = StubSide::new(words(&[8]));
    let moved_again = replay_against(&case, &mut again, &recording).unwrap();
    let found = against(std::slice::from_ref(&moved_again), &listed);
    assert_eq!(found.len(), 2, "{found:?}");

    // The direction that keeps the file from rotting, for a divergence as for a refusal: a run
    // that no longer diverges fails and names the gap the row was waiting on.
    let mut fixed = StubSide::new(words(&[7]));
    let exact = replay_against(&case, &mut fixed, &recording).unwrap();
    assert_eq!(exact.verdict.word(), "exact");
    let found = against(std::slice::from_ref(&exact), &listed);
    let [gone @ Expectation::NoLongerHeld { gap, .. }] = found.as_slice() else {
        panic!("{found:?}");
    };
    assert!(gap.waiting_on.contains("on purpose"), "{gap:?}");
    let printed = gone.to_string();
    assert!(printed.contains("no longer diverges at"), "{printed}");
    assert!(printed.contains("drop the row"), "{printed}");
}

/// A divergence never launders a refusal of the same case: the verdict carries both, and the
/// ledger holds a refusal row against a case that also diverges.
///
/// Without this the refusal would vanish from the verdict — `Divergent` is the more severe of the
/// two and used to be the only thing reported — so its row would read as "no longer refused" and
/// somebody would drop a row whose gap was still there.
#[test]
fn a_divergence_does_not_launder_a_refusal_of_the_same_case() {
    use crate::known_gaps::{against, Expects, KnownGap, EVERY_VENUE};
    let case = LedgerCase::new("ledger-both");
    let recording = recording_of(&case, case.metal(7), 3);
    let mut side = StubSide::new(words(&[9])).refusing(
        "texture",
        Failure::named("Error::Invalid", "the texture was refused"),
    );
    let both = replay_against(&case, &mut side, &recording).unwrap();
    let Verdict::Divergent { found, refused } = &both.verdict else {
        panic!("{:?}", both.verdict);
    };
    assert_eq!(found.len(), 1, "{found:?}");
    // The refusal is reported, not swallowed by the divergence.
    assert_eq!(refused.len(), 1, "{refused:?}");
    let printed = both.verdict.to_string();
    assert!(printed.contains("and refused case/texture"), "{printed}");

    let (site, answer) = refused[0].split_once(": ").unwrap();
    let rows = vec![
        KnownGap {
            case: case.name().to_owned(),
            venue: EVERY_VENUE.to_owned(),
            expects: Expects::Refused,
            site: site.to_owned(),
            answer: answer.to_owned(),
            waiting_on: "a refusal of this test's own, beside a divergence of the same case"
                .to_owned(),
        },
        KnownGap {
            case: case.name().to_owned(),
            venue: EVERY_VENUE.to_owned(),
            expects: Expects::Divergent,
            site: found[0].path.clone(),
            answer: found[0].kind.to_string(),
            waiting_on: "a divergence of this test's own, beside a refusal of the same case"
                .to_owned(),
        },
    ];
    // Both rows are held, so a case holding both gaps can reach exit 0 — and neither row can go
    // unchecked behind the other.
    assert_eq!(against(std::slice::from_ref(&both), &rows), Vec::new());

    // Drop the refusal from the run and its row fails, although the divergence is still there.
    let mut only_moved = StubSide::new(words(&[9]));
    let diverged = replay_against(&case, &mut only_moved, &recording).unwrap();
    let found = against(std::slice::from_ref(&diverged), &rows);
    let [crate::known_gaps::Expectation::NoLongerHeld { gap, .. }] = found.as_slice() else {
        panic!("{found:?}");
    };
    assert_eq!(gap.expects, Expects::Refused);
}

/// A site the venue honestly lacks is the **venue's** gap at that site, not a refusal of ours, and
/// the rest of the case still compares.
///
/// The gap this closes: the `lacks-feature` verdict came from the case **stopping**, so a case
/// that carried on past a call the device cannot take had that site read as a refusal of ours —
/// the one thing the verdict exists to prevent. It is per site now, and a divergence anywhere else
/// in the same case still fails, which is what makes it safe to be compliant.
#[test]
fn a_site_the_venue_lacks_is_the_venues_gap_and_not_a_refusal() {
    use metal2vulkan::error::{Capability, Feature, Unsupported};
    let case = LedgerCase::new("ledger-device-gap");
    let recording = recording_of(&case, case.metal(7), 3);
    // A typed refusal naming capabilities a device could enable: the device's own gap.
    let lacking = Failure::of(&metal2vulkan_runtime::Error::Unsupported(
        Unsupported::needs_any_of(
            Feature::Format,
            vec![vec![Capability::Feature(
                metal2vulkan::device::VkFeature::ShaderStorageImageMultisample,
            )]],
        ),
    ));
    assert!(lacking.lacks.is_some(), "{lacking:?}");

    let mut side = StubSide::new(words(&[7])).refusing("texture", lacking);
    let replayed = replay_against(&case, &mut side, &recording).unwrap();
    assert_eq!(replayed.verdict.word(), "lacks-feature");
    assert!(replayed.verdict.is_compliant());
    let printed = replayed.verdict.to_string();
    assert!(printed.contains("case/texture"), "{printed}");
    assert!(printed.contains("the venue lacks"), "{printed}");
    // The venue's own gap is never a ledger row, in either direction: it is not a bug of ours.
    assert_eq!(
        crate::known_gaps::against(std::slice::from_ref(&replayed), &[]),
        Vec::new()
    );

    // The rest of the case still compares: a difference elsewhere in the same run still fails.
    let lacking = Failure::of(&metal2vulkan_runtime::Error::Unsupported(
        Unsupported::needs_any_of(
            Feature::Format,
            vec![vec![Capability::Feature(
                metal2vulkan::device::VkFeature::ShaderStorageImageMultisample,
            )]],
        ),
    ));
    let mut moved = StubSide::new(words(&[9])).refusing("texture", lacking);
    let replayed = replay_against(&case, &mut moved, &recording).unwrap();
    assert_eq!(replayed.verdict.word(), "divergent");
    assert!(!replayed.verdict.is_compliant());

    // And a refusal that is ours to answer for is still ours: the same site, a typed error the
    // device could not have enabled away, reads as `refused`.
    let ours = Failure::of(&metal2vulkan_runtime::Error::Unsupported(
        Unsupported::not_implemented(Feature::Format),
    ));
    assert!(ours.lacks.is_none(), "{ours:?}");
    let mut unbuilt = StubSide::new(words(&[7])).refusing("texture", ours);
    let replayed = replay_against(&case, &mut unbuilt, &recording).unwrap();
    assert_eq!(replayed.verdict.word(), "refused");
}

#[test]
fn an_allowance_no_site_needed_is_counted_and_named_rather_than_silently_kept() {
    // The hole this closes: the up-front check over a case's allowances has to exempt a degrade
    // notice the run did not announce, because a notice is announced once per device per kind. So
    // a case could name a tolerance that fires on no venue, naming a notice no venue announces,
    // and every run would read green with nobody told. Two allowances of exactly this shape were
    // written and deleted by hand in this round.
    let mut case = WordsCase::new("words", 2);
    case.allowances = vec![
        // A tolerance nothing needs, for a notice nothing announces: speculation.
        Allowance::new(
            "case/output",
            Loosening::Absolute(8.0),
            Justification::notice(Degradation::DeviceArithmetic),
        ),
        // A tolerance nothing needs, for a notice the run did announce: not speculation, just
        // unused on this row.
        Allowance::new(
            "case/output",
            Loosening::Absolute(8.0),
            Justification::notice(Degradation::ShadowedTexture),
        ),
    ];
    let bytes = words(&[1, 2]);
    // The notice the second allowance names is announced, so it is justified; the run is
    // byte-identical, so neither allowance fires.
    let announced = [Degradation::ShadowedTexture.code()];
    let recording = recording_of(
        &case,
        case.transcript(bytes.clone(), "Completed", &announced),
        3,
    );
    let mut side = StubSide::new(bytes).announcing(&[Degradation::ShadowedTexture]);
    let replayed = replay_against(&case, &mut side, &recording).unwrap();

    // Report-only: the run is still exact, and still compliant.
    assert_eq!(replayed.verdict, Verdict::Exact, "{}", replayed.verdict);
    assert_eq!(replayed.allowances.named, 2);
    assert_eq!(replayed.allowances.fired(), 0);
    assert_eq!(
        replayed.allowances.unfired.len(),
        2,
        "{:?}",
        replayed.allowances.unfired
    );

    let mut report = Report::new();
    report.add(replayed);
    assert!(report.is_compliant());
    assert_eq!(report.allowance_counts(), (2, 2));
    assert_eq!(report.unfired_everywhere().len(), 2);
    // Only the one nothing justifies is called out as a tolerance no run has checked.
    let unchecked = report.unchecked_everywhere();
    assert_eq!(unchecked.len(), 1, "{unchecked:?}");
    assert_eq!(
        unchecked[0].1.justified_by,
        Justification::notice(Degradation::DeviceArithmetic).describe()
    );
    let rendered = report.render();
    assert!(
        rendered.contains("2 named, 0 fired, 2 never fired"),
        "{rendered}"
    );
    assert!(
        rendered.contains(Degradation::DeviceArithmetic.code()),
        "{rendered}"
    );
    assert!(rendered.contains("words                "), "{rendered}");
}

#[test]
fn an_allowance_that_fires_is_not_counted_as_unfired() {
    // The other half of the guard: a tolerance the run needs must not show up in the unfired
    // count, or the count would say "nobody checked this" of the one allowance a run did check.
    let mut case = WordsCase::new("words", 2);
    case.allowances = vec![
        Allowance::new(
            "case/output",
            Loosening::Absolute(8.0),
            Justification::notice(Degradation::DeviceArithmetic),
        ),
        // The notice site differs too, because the device announced what the tolerance names and
        // Metal announces nothing. That allowance fires as well, which is the point: both are
        // claims the run used.
        crate::replay::notice_allowance(Degradation::DeviceArithmetic.code()),
    ];
    let recording = recording_of(&case, case.transcript(words(&[1, 2]), "Completed", &[]), 3);
    // One word differs by 7, inside the allowance, and the notice it names is announced.
    let mut side = StubSide::new(words(&[1, 9])).announcing(&[Degradation::DeviceArithmetic]);
    let replayed = replay_against(&case, &mut side, &recording).unwrap();
    assert!(
        matches!(replayed.verdict, Verdict::Announced { .. }),
        "{}",
        replayed.verdict
    );
    assert_eq!(replayed.allowances.fired(), 2);
    assert!(
        replayed.allowances.unfired.is_empty(),
        "{:?}",
        replayed.allowances.unfired
    );
    let mut report = Report::new();
    report.add(replayed);
    assert!(report.unfired_everywhere().is_empty());
    assert!(report.render().contains("2 named, 2 fired, 0 never fired"));
}

#[test]
fn metal_on_this_chip_answers_the_two_plane_blits_exactly_as_the_probes_printed_them() {
    // Two rows lost their Metal leg to a reason that named the wrong thing: `NO_PUBLIC_CALL` said
    // the two-plane extended-range formats are Apple's private ones, so Metal had no public call
    // for them. The fill is private; the format is not. Both rows now record on Metal, and this
    // holds the committed recordings to the tables the probes printed — so the conversion is
    // checked rather than asserted, and a Metal that starts answering differently shows up here
    // instead of in a quiet re-record.
    let held = |case: &dyn Case, carried: &Transcript| {
        let recorded = crate::recording::read(case.name())
            .unwrap_or_else(|why| panic!("{}: {why}", case.name()));
        let answer = recorded
            .answer(MetalLevel::NEWEST)
            .unwrap_or_else(|why| panic!("{}: {why}", case.name()));
        assert_eq!(
            answer.provenance,
            Provenance::MetalRun,
            "{}: the case records on Metal now",
            case.name()
        );
        let taken = answer
            .deterministic()
            .unwrap_or_else(|| panic!("{}: one outcome", case.name()));
        // Field by field, through the suite's own comparison: a difference names the site.
        let found = compare(carried, taken, &[], Offered::notices(&[]));
        // The first few sites, not the whole comparison: a difference has to read on one screen.
        assert!(
            found.is_exact(),
            "{}: Metal's recorded answer is not the probe's table:\n  {}",
            case.name(),
            found
                .divergences
                .iter()
                .take(4)
                .map(|divergence| divergence.to_string())
                .collect::<Vec<String>>()
                .join("\n  ")
        );
    };

    let heap_blits = crate::cases::two_plane_heap_blits::heap_blits();
    let (runs, carried) = heap_blits.carried_from_the_m4().unwrap();
    assert_eq!(runs, 6, "the probe's runs behind the carried digests");
    held(&heap_blits, &carried);

    let announced = crate::cases::two_plane_copies::announced_row_blits();
    let printed = announced.printed_by_the_probe().unwrap();
    held(&announced, &printed);
    // And the case that still carries, still carries: its reason is the fill, not the format.
    let fills = crate::cases::two_plane_copies::row_blits();
    assert!(
        matches!(
            fills.metal_source(),
            MetalSource::Carried {
                re_record: ReRecord::Forbidden { reason },
                ..
            } if reason.contains("fillTexture:") && !reason.contains("formats are Apple's private")
        ),
        "{:?}",
        fills.metal_source()
    );
}

/// A case whose Metal leg does not come back: the shape there was no verdict for.
///
/// `pass`'s call answers on our side, which is what the row pins; `answers` makes it answer
/// instead, which is the break the verdict has to survive.
struct DiedOnMetal {
    /// Whether our side answers the call rather than refusing it.
    answers: bool,
}

/// The call the probe died in, as the case names its site.
const DIED_AT: &str = "the-call";

/// What the probe saw, in its own words.
const DIED_SAYS: &str = "Metal's process went away inside the call and never came back";

impl DiedOnMetal {
    /// The transcript the case states by hand: the inputs, and the call Metal never returned
    /// from.
    fn stated(&self) -> Transcript {
        let mut sites = Sites::new();
        sites.push(crate::case::inputs_node(&self.inputs().unwrap()));
        sites.value(DIED_AT, Value::Nothing(Negative::NeverReturned));
        sites.notices(NOTICES_SITE, &[] as &[&str]);
        sites.finish()
    }
}

impl Case for DiedOnMetal {
    fn name(&self) -> &'static str {
        "died-on-metal"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::did_not_return("the suite's own tests", DIED_AT, DIED_SAYS)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        Ok(vec![Input::bytes("buffer:0", &[0u8; 4])])
    }

    fn carried_transcript(&self) -> Result<Option<(u32, Transcript)>, String> {
        Ok(Some((1, self.stated())))
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        if self.answers {
            // We came back from the call Metal did not: the divergence the row exists to catch.
            let _ = made(sites, DIED_AT, side.zeroed_buffer(4))?;
            return Ok(());
        }
        sites.push(Node::leaf(
            DIED_AT,
            Observed::value(Failure::named("Unsupported", "this call is not built").recorded()),
        ));
        Ok(())
    }
}

#[test]
fn a_call_metal_did_not_return_from_is_its_own_verdict() {
    // The hole this closes: a row that ends Metal's process has nothing to carry -- the lane holds
    // no line for it -- so it was not recorded-only material, it had no verdict at all. `Exact`
    // would claim a value matched where Metal produced none; `Divergent` would claim a behaviour
    // of Metal's to differ from; `Refused` is for a call plain Metal answered, and counting it
    // there would bury a real gap of ours among rows that are not one.
    let case = DiedOnMetal { answers: false };
    let (runs, stated) = case.carried_transcript().unwrap().unwrap();
    let recording = Recording::new(
        case.name(),
        Vec::new(),
        vec![crate::record::never_returned(
            MetalLevel::NEWEST,
            "the suite's own tests",
            DIED_AT,
            "the probe's device",
            runs,
            stated,
        )],
    );
    let mut side = StubSide::new(Vec::new());
    let replayed = replay_against(&case, &mut side, &recording).unwrap();
    let Verdict::MetalNeverReturned { at, justified } = &replayed.verdict else {
        panic!("{}", replayed.verdict);
    };
    assert_eq!(at.len(), 1, "{at:?}");
    assert!(at[0].starts_with(&format!("case/{DIED_AT}: ")), "{at:?}");
    assert!(justified.is_empty(), "{justified:?}");

    // Counted like the others, and it does not loosen compliance: it is a distinct word in the
    // report, not a quiet pass hidden inside "exact".
    let mut report = Report::new();
    report.add(replayed);
    assert_eq!(report.counts().get("never-returned"), Some(&1));
    assert_eq!(report.counts().get("exact"), None);
    assert!(report.is_compliant());
    let rendered = report.render();
    assert!(rendered.contains("never-returned"), "{rendered}");
    assert!(
        rendered.contains("Metal did not return from the call, and we refused it"),
        "{rendered}"
    );

    // No allowance can reach it: a case that loosens the site whole still gets this verdict and
    // not `announced`, so the row cannot be dressed up as a justified difference.
    let unchecked = vec![Allowance::new(
        format!("case/{DIED_AT}"),
        Loosening::Unchecked,
        Justification::accepted(accepted_differences()[0].clone()),
    )];
    let found = compare(
        &case.stated(),
        &{
            let mut side = StubSide::new(Vec::new());
            replay_against(&case, &mut side, &recording).unwrap().ours
        },
        &unchecked,
        Offered::notices(&[]),
    );
    assert_eq!(found.never_returned.len(), 1, "{found:?}");
    assert!(found.announced.is_empty(), "{found:?}");
}

#[test]
fn answering_where_metal_did_not_return_is_divergent() {
    // The other half, and the reason the verdict is a mechanism and not a label: the one fact the
    // row holds is that *we* refuse where Metal went away. Answering instead has to fail.
    let case = DiedOnMetal { answers: true };
    let (runs, stated) = DiedOnMetal { answers: false }
        .carried_transcript()
        .unwrap()
        .unwrap();
    let recording = Recording::new(
        case.name(),
        Vec::new(),
        vec![crate::record::never_returned(
            MetalLevel::NEWEST,
            "the suite's own tests",
            DIED_AT,
            "the probe's device",
            runs,
            stated,
        )],
    );
    let mut side = StubSide::new(Vec::new());
    let replayed = replay_against(&case, &mut side, &recording).unwrap();
    let Verdict::Divergent { found, .. } = &replayed.verdict else {
        panic!("{}", replayed.verdict);
    };
    assert!(
        found.iter().any(|divergence| matches!(
            divergence.kind,
            DivergenceKind::AnsweredWhereMetalDidNotReturn { .. }
        )),
        "{found:?}"
    );
    assert!(!replayed.verdict.is_compliant());
}

#[test]
fn record_mode_refuses_a_row_metal_does_not_return_from() {
    // It may never be produced by running the shape: making the call is what ends Metal's
    // process. `forbidden` carries the probe's own words, so `record` refuses before anything is
    // encoded, on a host with no Metal as on this one.
    let case = DiedOnMetal { answers: false };
    assert_eq!(case.metal_source().forbidden(), Some(DIED_SAYS));
    let mut side = StubSide::new(Vec::new());
    let refused = record(&case, &mut side, 1, MetalLevel::NEWEST, Validated::NotRun);
    assert_eq!(
        refused,
        Err(RecordError::Forbidden {
            case: case.name().to_owned(),
            reason: DIED_SAYS.to_owned(),
        }),
        "{refused:?}"
    );
    // And the answer that is written instead says so in the file: the provenance names the call,
    // and Metal's API validation was never asked either.
    let answer = crate::record::never_returned(
        MetalLevel::NEWEST,
        "the suite's own tests",
        DIED_AT,
        "the probe's device",
        1,
        case.stated(),
    );
    assert_eq!(answer.provenance.never_returned_from(), Some(DIED_AT));
    assert_eq!(answer.provenance.measured_elsewhere(), None);
    assert_eq!(answer.validated, Validated::NotRun);
}

#[test]
fn the_report_says_how_many_answers_came_from_another_chip() {
    // 53 recordings carried an Apple M4's values and `conformance cases` showed it only through a
    // `Debug` print, so a run could read entirely green while a fifth of its answers came from a
    // machine that is gone. The suite's own rule is that a recording of another oracle is stale;
    // a carried table is the one place that rule is relaxed, so the relaxation has to be counted
    // where a reader of the report sees it.
    use crate::recording::APPLE_M4;
    let carried = |case: &dyn Case, chip: &str| {
        let bytes = words(&[1, 2]);
        let transcript = WordsCase::new(case.name(), 2).transcript(bytes, "Completed", &[]);
        Recording::new(
            case.name(),
            Vec::new(),
            vec![RecordedAnswer {
                key: crate::recording::OracleKey::current(MetalLevel::NEWEST),
                provenance: Provenance::Carried {
                    from: "a probe this tree holds".to_owned(),
                    measured_on: chip.to_owned(),
                },
                device: "the probe's device".to_owned(),
                runs: 1,
                outcomes: vec![Outcome {
                    runs: 1,
                    transcript,
                }],
                validated: Validated::NotRun,
            }],
        )
    };
    let mut elsewhere = WordsCase::new("from-the-m4", 2);
    elsewhere.source = MetalSource::carried("a probe this tree holds").measured_on(APPLE_M4);
    let mut here = WordsCase::new("from-the-oracle", 2);
    here.source = MetalSource::carried("a probe this tree holds");

    let mut report = Report::new();
    let mut side = StubSide::new(words(&[1, 2]));
    let far = replay_against(&elsewhere, &mut side, &carried(&elsewhere, APPLE_M4)).unwrap();
    assert_eq!(far.verdict, Verdict::Exact, "{}", far.verdict);
    assert_eq!(far.measured_on.as_deref(), Some(APPLE_M4));
    report.add(far);
    let mut side = StubSide::new(words(&[1, 2]));
    let near = replay_against(
        &here,
        &mut side,
        &carried(&here, crate::recording::ORACLE_CHIP),
    )
    .unwrap();
    // An answer taken on the oracle names no other chip: the count is of the relaxation, not of
    // every carried row.
    assert_eq!(near.measured_on, None);
    report.add(near);

    assert_eq!(
        report.carried_from_another_chip(),
        [(APPLE_M4, 1)].into_iter().collect()
    );
    let rendered = report.render();
    assert!(rendered.contains("carried from another chip"), "{rendered}");
    assert!(
        rendered.contains(&format!("1 row(s) answered from {APPLE_M4}")),
        "{rendered}"
    );
    // The rows are still compliant: a carried answer from another chip is weaker evidence, not a
    // failure. Counting it is the whole mechanism.
    assert!(report.is_compliant());
}

// The cost half: what our side did to produce a case's transcript. Counts, never Metal's, and
// never a correctness verdict.

/// The cost half reads **every** count the runtime keeps, and fails loudly on a field it cannot.
///
/// `WorkCounts` has 386 fields of its own plus three nested count structs, and a hand-written list
/// of them would be stale the day a runtime change added one, so the reading comes out of the
/// derived `Debug`. The risk that buys is a field of another shape being dropped silently, which
/// would be a hole in the gate nobody could see — so a leaf that is not a count is an error here,
/// and the only fields left out are the ones named with their reason.
#[test]
fn the_cost_half_reads_every_count_the_runtime_keeps() {
    let counts = crate::cost::Counts::of(&metal2vulkan_runtime::WorkCounts::default())
        .expect("the runtime's counts read as counts");
    assert!(
        counts.len() > 350,
        "only {} counts were read, so the reading is not of the whole struct",
        counts.len()
    );
    // One count from each nesting level, so a change that flattened or renamed a level fails here.
    for name in [
        "queue_submissions",
        "variant_pipelines_made",
        "commands_replayed.cmd_draw",
        "object_churn.images_parked",
        "shadow_keeps.render_cut",
    ] {
        assert_eq!(counts.get(name), 0, "{name} was not read");
        assert!(
            counts.iter().any(|(held, _)| held == name),
            "{name} is not among the counts the cost half reads"
        );
    }
    // The site tables are left out, and nothing of them leaks in under another name.
    assert!(
        !counts.iter().any(|(name, _)| name.contains("copy_sites")),
        "a site table was read as a count"
    );
    assert!(crate::cost::excluded("copy_sites").is_some());
    assert!(crate::cost::excluded("row_copy_regions_most").is_some());
    assert!(crate::cost::excluded("batch_submit_nanos").is_some());
    assert!(crate::cost::excluded("queue_submissions").is_none());
}

/// A field of the runtime's counts that is not a count fails the reading by name.
///
/// This is the guard that makes the test above load-bearing: without it, a runtime field of a new
/// shape would simply not be counted, and the gate would quietly stop holding it.
#[test]
fn a_field_that_is_not_a_count_fails_the_reading() {
    let shape = "WorkCounts {\n    queue_submissions: 3,\n    a_new_flag: true,\n}";
    let why = crate::cost::Counts::read(shape).expect_err("a bool is not a count");
    assert!(why.contains("a_new_flag"), "{why}");
    assert!(why.contains("NOT_A_COUNT"), "{why}");
    // And the same shape with the field left out reads fine, so the failure is the field's.
    let read = crate::cost::Counts::read("WorkCounts {\n    queue_submissions: 3,\n}")
        .expect("a count reads");
    assert_eq!(read.get("queue_submissions"), 3);
}

/// Nesting, arrays and the site tables all come out of the reading as they went in.
#[test]
fn the_reading_follows_the_nesting_and_skips_the_site_tables() {
    let shape = "\
WorkCounts {
    queue_submissions: 2,
    commands_replayed: CommandsReplayed {
        cmd_draw: 7,
    },
    copy_sites: [
        CopySite {
            command: \"vkCmdCopyBuffer\",
            file: \"buffer.rs\",
            line: 10,
            calls: 1,
        },
        CopySite {
            command: \"\",
            file: \"\",
            line: 0,
            calls: 0,
        },
    ],
    render_passes_made: 1,
}";
    let read = crate::cost::Counts::read(shape).expect("the shape reads");
    assert_eq!(read.get("queue_submissions"), 2);
    assert_eq!(read.get("commands_replayed.cmd_draw"), 7);
    assert_eq!(read.get("render_passes_made"), 1);
    assert_eq!(read.len(), 3, "{:?}", read);
}

/// The three regions are the ones the cost regime names, and the device API is the residual.
#[test]
fn the_work_is_attributed_to_the_three_regions() {
    use crate::cost::{region_of, Region};
    for name in [
        "variant_pipelines_made",
        "shader_modules_made",
        "pipeline_layouts_shared",
        "graphics_pipelines_looked_up",
    ] {
        assert_eq!(region_of(name), Region::Translation, "{name}");
    }
    for name in [
        "commands_replayed.cmd_draw",
        "queue_submissions",
        "command_buffers_submitted",
        "scheduler_passes",
        "wakes_handler_threads",
        "completed_waits",
    ] {
        assert_eq!(region_of(name), Region::Submission, "{name}");
    }
    // What a call does on the calling thread, including the counts whose names mention a pipeline
    // or a submission but which are draws and binds.
    for name in [
        "draw_shapes_made",
        "object_churn.images_parked",
        "stage_tables_read",
        "fills_recorded",
        "provoking_last_pipeline_draws",
        "instance_splits_pending_pipeline",
    ] {
        assert_eq!(region_of(name), Region::DeviceApi, "{name}");
    }
}

/// A count the measurement never saw move is bounded at zero.
///
/// This is the whole point of the gate: a case that starts doing something it did not do before
/// reads as `Over` rather than as nothing at all.
#[test]
fn a_count_the_measurement_never_saw_is_bounded_at_zero() {
    let record = crate::cost::CostRecord {
        venue: "a venue".to_owned(),
        runs: 2,
        measured: "two runs of the whole selection".to_owned(),
        contrasted: None,
        degraded: Vec::new(),
        bounds: [("queue_submissions".to_owned(), 1)].into_iter().collect(),
        moving: Default::default(),
        composed: Default::default(),
        ceilings: Default::default(),
        unmeasurable: None,
    };
    let judge = |text: &str| {
        crate::cost::judge(
            std::slice::from_ref(&record),
            "a venue",
            true,
            &crate::cost::Counts::read(text).expect("the shape reads"),
            None,
            &[],
        )
    };
    // At the bound: within.
    assert_eq!(
        judge("WorkCounts {\n    queue_submissions: 1,\n}").word(),
        "within"
    );
    // Over the bound: over, naming the count, the region, the bound and what was found.
    let over = judge("WorkCounts {\n    queue_submissions: 2,\n}");
    assert_eq!(over.word(), "over");
    assert!(
        over.to_string().contains(
            "queue_submissions (submission) reached 2 where the \
             measurement's most was 1"
        ),
        "{over}"
    );
    // A count nothing bounded, which the measurement never saw: over, bounded at zero.
    let fresh = judge("WorkCounts {\n    queue_submissions: 1,\n    render_passes_made: 1,\n}");
    assert_eq!(fresh.word(), "over");
    assert!(fresh.to_string().contains("render_passes_made"), "{fresh}");
}

/// A count whose readings moved between runs of the same case on the same venue is reported, not
/// gated: a bound that fails at random is worse than no bound.
#[test]
fn a_count_whose_readings_moved_is_reported_and_not_gated() {
    let bound = |range: Option<u64>| crate::cost::CostRecord {
        venue: "a venue".to_owned(),
        runs: 3,
        measured: "three runs".to_owned(),
        contrasted: None,
        degraded: Vec::new(),
        bounds: [("render_passes_made".to_owned(), 2)].into_iter().collect(),
        moving: range
            .map(|least| {
                [("render_passes_made".to_owned(), least)]
                    .into_iter()
                    .collect()
            })
            .unwrap_or_default(),
        composed: Default::default(),
        ceilings: Default::default(),
        unmeasurable: None,
    };
    let found = crate::cost::Counts::read("WorkCounts {\n    render_passes_made: 9,\n}").unwrap();
    // Deterministic in the measurement: held, and 9 is over 2.
    let gated = bound(None);
    assert_eq!(
        crate::cost::judge(
            std::slice::from_ref(&gated),
            "a venue",
            true,
            &found,
            None,
            &[]
        )
        .word(),
        "over"
    );
    assert_eq!(gated.gated(), 1);
    // Seen to move between 0 and 2: reported, and 9 does not fail the run.
    let moving = bound(Some(0));
    assert_eq!(
        crate::cost::judge(
            std::slice::from_ref(&moving),
            "a venue",
            true,
            &found,
            None,
            &[]
        )
        .word(),
        "within"
    );
    assert_eq!(moving.gated(), 0);
}

/// Every way a run can fail to have a bound is its own `Unmeasured`, with the reason in words.
///
/// Four of them, and each matters: a venue the measurement was not taken on; a selection that is
/// not the whole case set, because a case's counts depend on what ran before it; a run that
/// stopped part way; and — the one the correctness half's own fix is about — a run the **device**
/// degraded otherwise than the measurement did, whose lower counts are the device's gap and not a
/// cost we saved.
#[test]
fn a_run_with_no_bound_that_stands_is_unmeasured_and_says_why() {
    let record = crate::cost::CostRecord {
        venue: "the measured venue".to_owned(),
        runs: 2,
        measured: "two runs".to_owned(),
        contrasted: None,
        degraded: vec!["half-nan-quieted".to_owned()],
        bounds: [("queue_submissions".to_owned(), 1)].into_iter().collect(),
        moving: Default::default(),
        composed: Default::default(),
        ceilings: Default::default(),
        unmeasurable: None,
    };
    let found = crate::cost::Counts::read("WorkCounts {\n    queue_submissions: 1,\n}").unwrap();
    let records = std::slice::from_ref(&record);
    let degraded = [String::from("half-nan-quieted")];
    let why = |verdict: crate::cost::CostVerdict| match verdict {
        crate::cost::CostVerdict::Unmeasured { why } => why,
        other => panic!("{other:?}"),
    };
    // The measurement's own shape: within.
    assert_eq!(
        crate::cost::judge(records, "the measured venue", true, &found, None, &degraded).word(),
        "within"
    );
    // Another venue.
    let other = why(crate::cost::judge(
        records,
        "another venue",
        true,
        &found,
        None,
        &degraded,
    ));
    assert!(other.contains("another venue"), "{other}");
    assert!(other.contains("the measured venue"), "{other}");
    // A selection rather than the whole case set.
    let part = why(crate::cost::judge(
        records,
        "the measured venue",
        false,
        &found,
        None,
        &degraded,
    ));
    assert!(part.contains("whole case set"), "{part}");
    // A run that stopped part way.
    let stopped = why(crate::cost::judge(
        records,
        "the measured venue",
        true,
        &found,
        Some("the venue lacks 2x multisampling"),
        &degraded,
    ));
    assert!(stopped.contains("2x multisampling"), "{stopped}");
    // A run the device degraded otherwise: the gap must not read as a win.
    let elsewhere = why(crate::cost::judge(
        records,
        "the measured venue",
        true,
        &found,
        None,
        &[],
    ));
    assert!(
        elsewhere.contains("half-nan-quieted") && elsewhere.contains("device's gap"),
        "{elsewhere}"
    );
}

/// The cost record rides in the recording, and a recording written before the cost half existed
/// still reads.
#[test]
fn the_cost_record_rides_in_the_recording() {
    let answer = RecordedAnswer {
        key: crate::recording::OracleKey::current(MetalLevel::NEWEST),
        provenance: Provenance::MetalRun,
        device: "a device".to_owned(),
        runs: 1,
        outcomes: vec![Outcome {
            runs: 1,
            transcript: Sites::new().finish(),
        }],
        validated: Validated::NotRun,
    };
    let costs = vec![crate::cost::CostRecord {
        venue: "a venue".to_owned(),
        runs: 2,
        measured: "two runs of the whole selection".to_owned(),
        contrasted: None,
        degraded: Vec::new(),
        bounds: [("queue_submissions".to_owned(), 1)].into_iter().collect(),
        moving: Default::default(),
        composed: Default::default(),
        ceilings: Default::default(),
        unmeasurable: None,
    }];
    let recording = Recording::new("a case", Vec::new(), vec![answer]).with_costs(costs.clone());
    let text = serde_json::to_string_pretty(&recording).expect("it writes");
    let read: Recording = serde_json::from_str(&text).expect("it reads back");
    assert_eq!(read.costs, costs);
    // A recording of the format as it was before the cost half holds no `costs` member at all: it
    // reads, with no cost record, which is `Unmeasured` and never a pass. So the member is left
    // out where it is empty, and the 127 recordings already committed stay byte for byte as they
    // are until a cost measurement is written into them.
    let mut without = recording.clone();
    without.costs = Vec::new();
    let text = serde_json::to_string(&without).expect("it writes");
    assert!(!text.contains("costs"), "an empty cost half is not written");
    let read: Recording = serde_json::from_str(&text).expect("it reads back");
    assert!(read.costs.is_empty());
}

/// A measurement turns into bounds that name it, and a count whose runs disagreed carries its
/// range.
#[test]
fn a_measurement_becomes_bounds_that_name_it() {
    let cost = crate::cost::CaseCost {
        case: "a case".to_owned(),
        counts: crate::cost::Counts::read(
            "WorkCounts {\n    queue_submissions: 2,\n    render_passes_made: 5,\n}",
        )
        .unwrap(),
        least: crate::cost::Counts::read(
            "WorkCounts {\n    queue_submissions: 2,\n    render_passes_made: 1,\n}",
        )
        .unwrap(),
        runs: 3,
        stopped: None,
        degraded: vec!["a-code".to_owned()],
        nanos: 1_000,
    };
    let record = cost.record("a venue", "three runs of the whole selection".to_owned());
    assert_eq!(record.venue, "a venue");
    assert_eq!(record.runs, 3);
    assert_eq!(record.measured, "three runs of the whole selection");
    assert_eq!(record.degraded, vec!["a-code".to_owned()]);
    let submissions = record.bound("queue_submissions").expect("it is bounded");
    assert_eq!((submissions.most, submissions.range), (2, None));
    assert!(submissions.gated());
    let passes = record.bound("render_passes_made").expect("it is bounded");
    assert_eq!((passes.most, passes.range), (5, Some(1)));
    assert!(!passes.gated());
    assert_eq!(record.gated(), 1);
    assert_eq!(cost.moved(), vec![("render_passes_made", 1, 5)]);
}

/// Re-taking a case's Metal answer keeps its cost record.
///
/// The two halves are measured by different legs — Metal's answer on this Mac, our cost on a venue
/// — so a re-record must not empty the cost gate. Without this, every re-recording would silently
/// turn the cost half off for the case it touched.
#[test]
fn re_recording_a_metal_answer_keeps_the_cost_record() {
    let held = crate::recording::read(crate::cases::pass_bytes::all()[0].name())
        .expect("a committed recording");
    let costs = vec![crate::cost::CostRecord {
        venue: "a venue".to_owned(),
        runs: 1,
        measured: "one run".to_owned(),
        contrasted: None,
        degraded: Vec::new(),
        bounds: Default::default(),
        moving: Default::default(),
        composed: Default::default(),
        ceilings: Default::default(),
        unmeasurable: None,
    }];
    let held = held.with_costs(costs.clone());
    // A fresh Metal recording of the same case, with different answers, which is what re-recording
    // produces. `record::write` is not called here: it would write over a committed recording.
    let fresh = Recording::new(held.case.clone(), Vec::new(), Vec::new());
    let kept = held.clone().carrying_the_answers_of(fresh);
    // What Metal measures is replaced...
    assert!(
        kept.answers.is_empty(),
        "the fresh answers are the ones kept"
    );
    assert!(kept.shaders.is_empty());
    // ...and the cost half survives.
    assert_eq!(kept.costs, costs);
    // **And so does every other part of the record, including one this test does not name.** The
    // check runs over the serialised form rather than over named fields, so a half another leg
    // adds later is covered by it before anyone thinks to come back here. That is the property the
    // inverted carry exists for: keeping is the default, and dropping takes an explicit line.
    //
    // This assertion replaces one that searched `record.rs` for a line of its own source. A guard
    // that matches text cannot tell a correct change from a broken one: it failed when this carry
    // was *improved*, and it would have passed had the carry been replaced by something that
    // dropped every half while keeping the string.
    let before = serde_json::to_value(&held).expect("it serialises");
    let after = serde_json::to_value(&kept).expect("it serialises");
    let measured = ["shaders", "answers", "format", "suite_version"];
    let before = before.as_object().expect("a record is an object");
    let after = after.as_object().expect("a record is an object");
    for (key, value) in before {
        if measured.contains(&key.as_str()) {
            continue;
        }
        assert_eq!(after.get(key), Some(value), "{key} was not carried");
    }
    for key in after.keys() {
        assert!(
            before.contains_key(key) || measured.contains(&key.as_str()),
            "{key} appeared from nowhere"
        );
    }
}

/// A count that says which thread arrived first is reported and never gated, whatever a
/// measurement saw — and the count that holds its total still is.
///
/// Four measurements of the whole case set and three replays of it put every moving reading and
/// every overrun in these two families, so the rule is theirs and not a measurement's. The test
/// pins both halves of it: the race is ungated, and the deterministic total beside it is not.
#[test]
fn a_count_that_says_which_thread_arrived_first_is_never_gated() {
    use crate::cost::racy;
    for count in [
        "command_buffers_replayed",
        "command_buffers_replayed_inline",
        "object_churn.images_evicted",
        "object_churn.vk_images_destroyed",
        "object_churn.vk_images_made",
        "scheduler_passes",
        "scheduler_wakes_host",
        "wakes_spared",
        "wakes_handler_threads",
        // Process-dependent rather than thread-dependent, and no more boundable for it: a reading
        // of what is alive now, a release another thread got round to, an object found already
        // made, and waits borrowed from whatever ran before the case.
        "memory_blocks_live",
        "empty_memory_blocks_live",
        "sampler_sets_destroyed",
        "pipeline_layouts_shared",
        "shader_modules_shared",
        "completed_waits",
        // A draw takes none of a shape whose derivation announces a notice the device announces at
        // every occurrence, and this suite sets that over every kind, so what a case reuses
        // depends on the notice state every case before it left behind:
        // `integer-vertex-attributes` reads 0 alone and reached 19 and then 11 in two whole-set
        // replays.
        "draw_shapes_reused",
    ] {
        assert!(racy(count).is_some(), "{count}");
    }
    // Counts the rule does not name, so the measurement decides them case by case. The rule is
    // about what a counter counts; whether a particular case's reading of one depends on the
    // selection is a question only the contrast run answers, and
    // `a_count_the_selection_decides_is_reported_and_not_gated` holds that half.
    for count in [
        "command_buffers_deferred",
        "command_buffers_submitted",
        "queue_submissions",
        "commands_replayed.cmd_draw",
        "render_passes_made",
        "slot_returns",
        // The `_made` partner of a `_shared` or `_reused` count stays gated: finding one already
        // made, or reusing a shape already derived, only lowers it, and the gate is an upper bound.
        "pipeline_layouts_made",
        "shader_modules_made",
        "draw_shapes_made",
    ] {
        assert!(racy(count).is_none(), "{count}");
    }
    // The ones whole-set replays caught, each for a reason the runtime's own doc comment gives
    // before any measurement does: the pool's state at an instant, what a completion thread began
    // ahead for a pool an earlier case gave back, the same thing counted again once that pool is
    // back among the spares, which of our threads ran a transition, and a block freed because its
    // quiet period elapsed.
    for count in [
        "pipeline_pool_busy",
        "encoder_primaries_begun_ahead",
        "primaries_begun_ahead",
        "completed_where_seen",
        "pooled_memory_blocks_freed",
        "finales_folded",
    ] {
        assert!(racy(count).is_some(), "{count}");
    }
    // The total `encoder_primaries_begun_ahead` splits stays gated, so no work escapes by it, as
    // does what a case allocates even though what the pool hands back does not.
    assert!(racy("encoder_command_buffers_begun").is_none());
    assert!(racy("pooled_memory_blocks_made").is_none());
    // `finales_apart` and `scheduled_where_submitted` were left gated by an earlier pass of this
    // round *because nothing had been seen to move them*. That was the wrong test, and these two
    // assertions are the reversal: "nothing has moved it" is a fact about the afternoon the
    // measurement was taken, not about the count, and a rule written from a quiet machine fires on
    // a busy one. Both are reported now, on their definitions.
    for count in ["finales_apart", "scheduled_where_submitted"] {
        assert!(racy(count).is_some(), "{count}");
    }
    // The third class: counts that are not a property of the case at all, read off what the
    // counter *is* in the runtime's source and not off whether a reading moved. Four buckets of
    // one queue-depth histogram — "every batch, by how far its queue's work ran behind its
    // submission" — and the rest of the family that asks what the host had seen complete.
    //
    // `batches_with_none_pending` was the most widely gated count in the suite, at 440 cases, and
    // it is why the whole set reported four cases over bounds twice with only one case in common:
    // its sibling `batches_with_one_pending` was bounded at 0 almost everywhere, so a batch that
    // found one earlier batch still pending moved a reading into a bound of zero, in whichever
    // cases the GPU happened to be behind on that run.
    //
    // `groups_kept_pending` and `groups_sent_for_waits` read 0 in every case of the set, so **no
    // measurement corroborates them**. They are here on their definitions alone, which is the whole
    // point of the class: a measurement-first rule would have left them gated at zero, waiting to
    // fail on someone else's busy afternoon.
    for count in [
        "batches_with_none_pending",
        "batches_with_one_pending",
        "batches_with_two_or_three_pending",
        "batches_with_four_or_more_pending",
        "groups_kept_pending",
        "groups_sent_for_waits",
        "registrations_waited",
    ] {
        assert!(
            crate::cost::not_the_cases_own(count).is_some(),
            "{count} is not the case's own"
        );
        assert!(racy(count).is_some(), "{count} must never be gated");
    }
    // And the class is a list of names, not a guess at a shape: a count whose definition makes it
    // the case's own work stays out of it however much its name resembles one that is in.
    for count in [
        "batches_grouped",
        "queue_submissions",
        "submit_infos",
        "encoder_command_buffers_begun",
        "internal_submissions",
        "owned_descriptor_pools",
    ] {
        assert!(
            crate::cost::not_the_cases_own(count).is_none(),
            "{count} is the case's own work"
        );
    }
    // A record that bounds both at 1, with neither seen to move.
    let record = crate::cost::CostRecord {
        venue: "a venue".to_owned(),
        runs: 8,
        measured: "eight rounds".to_owned(),
        contrasted: None,
        degraded: Vec::new(),
        bounds: [
            ("command_buffers_replayed_inline".to_owned(), 1),
            ("command_buffers_deferred".to_owned(), 1),
        ]
        .into_iter()
        .collect(),
        moving: Default::default(),
        composed: Default::default(),
        ceilings: Default::default(),
        unmeasurable: None,
    };
    assert_eq!(record.gated(), 1, "only the total is gated");
    let judge = |text: &str| {
        crate::cost::judge(
            std::slice::from_ref(&record),
            "a venue",
            true,
            &crate::cost::Counts::read(text).expect("the shape reads"),
            None,
            &[],
        )
    };
    // The race going over its bound is not a failure.
    assert_eq!(
        judge(
            "WorkCounts {\n    command_buffers_deferred: 1,\n    \
             command_buffers_replayed_inline: 9,\n}"
        )
        .word(),
        "within"
    );
    // The total going over it is.
    let over = judge(
        "WorkCounts {\n    command_buffers_deferred: 2,\n    \
         command_buffers_replayed_inline: 1,\n}",
    );
    assert_eq!(over.word(), "over");
    assert!(
        over.to_string().contains("command_buffers_deferred"),
        "{over}"
    );
    // And a racy count the record never bounded at all is still not gated, which is the case that
    // sent this rule out of the measurement and into the code.
    assert_eq!(
        judge(
            "WorkCounts {\n    object_churn: ObjectChurn {\n        images_evicted: 6,\n    },\n}"
        )
        .word(),
        "within"
    );
}

/// Every field of a [`crate::render::ColorState`] reaches the color attachment descriptor the
/// runtime side gives the runtime (`runtime_side::color_descriptor`), checked field by field with
/// no device.
///
/// It is a guard against the bug that wrote it: the write mask was never assigned there, so a case
/// that asked for a mask of no channel got a pipeline that wrote every channel, and the family
/// asking what Metal's tile store leaves behind read as our implementation ignoring a legal Metal
/// state. The Metal side set the mask, so the two sides were asking two different questions and
/// nothing in the suite noticed.
#[test]
fn the_runtime_side_forwards_every_field_of_a_color_state() {
    use crate::render::{Blend, ColorState};
    use metal2vulkan::metal::{BlendFactor, BlendOperation, ColorWriteMask, PixelFormat};

    // The mask, at the three shapes that tell a dropped field from a kept one: none, some, all.
    for mask in [
        ColorWriteMask::NONE,
        ColorWriteMask::RED | ColorWriteMask::ALPHA,
        ColorWriteMask::ALL,
    ] {
        let made = crate::runtime_side::color_descriptor(
            &ColorState::new(PixelFormat::RGBA8Unorm).masked(mask),
        );
        assert_eq!(made.write_mask, mask, "the write mask {mask:?}");
        assert_eq!(made.pixel_format, PixelFormat::RGBA8Unorm);
        assert!(!made.blending_enabled, "no blend was asked for");
    }

    // The blend's six factors and operations, each a different value so that a field taking
    // another's value fails rather than passing by coincidence.
    let blend = Blend {
        source_rgb: BlendFactor::SourceColor,
        destination_rgb: BlendFactor::DestinationAlpha,
        rgb: BlendOperation::ReverseSubtract,
        source_alpha: BlendFactor::OneMinusSourceAlpha,
        destination_alpha: BlendFactor::BlendColor,
        alpha: BlendOperation::Min,
    };
    let made = crate::runtime_side::color_descriptor(
        &ColorState::new(PixelFormat::R16Snorm)
            .masked(ColorWriteMask::GREEN)
            .blended(blend),
    );
    assert_eq!(made.pixel_format, PixelFormat::R16Snorm);
    assert_eq!(made.write_mask, ColorWriteMask::GREEN);
    assert!(made.blending_enabled);
    assert_eq!(made.source_rgb_blend_factor, blend.source_rgb);
    assert_eq!(made.destination_rgb_blend_factor, blend.destination_rgb);
    assert_eq!(made.rgb_blend_operation, blend.rgb);
    assert_eq!(made.source_alpha_blend_factor, blend.source_alpha);
    assert_eq!(made.destination_alpha_blend_factor, blend.destination_alpha);
    assert_eq!(made.alpha_blend_operation, blend.alpha);
}

/// A count that reads one number inside the case set and another with the case alone is the
/// selection's and not the case's: it is reported, its bound is the larger of the two readings, and
/// the gate does not hold it.
///
/// This is the rule that stops a bound going stale on the next landing. A reading a landing can
/// change is a reading that depends on what ran before the case, and measuring the case alone is
/// how the record sees that.
#[test]
fn a_count_the_selection_decides_is_reported_and_not_gated() {
    // What the gate's own shape wrote: one pipeline layout made, four command buffers begun, two
    // submissions.
    let held = crate::cost::CostRecord {
        venue: "a venue".to_owned(),
        runs: 8,
        measured: "eight runs of the whole selection".to_owned(),
        contrasted: None,
        degraded: Vec::new(),
        bounds: [
            ("command_buffers_submitted".to_owned(), 1),
            ("command_buffers_begun".to_owned(), 4),
            ("queue_submissions".to_owned(), 2),
        ]
        .into_iter()
        .collect(),
        moving: Default::default(),
        composed: Default::default(),
        ceilings: Default::default(),
        unmeasurable: None,
    };
    // What the case does alone: two submissions where company needed one, and three command
    // buffers where company began four, because the work it batches with is not there. The two
    // move in opposite directions, and the 393-case measurement found both of them doing that
    // across the set, so no MONOTONE row covers either and neither shape bounds them.
    let apart = crate::cost::CaseCost {
        case: "a case".to_owned(),
        counts: crate::cost::Counts::read(
            "WorkCounts {\n    command_buffers_submitted: 2,\n    command_buffers_begun: 3,\n    \
             queue_submissions: 2,\n}",
        )
        .unwrap(),
        least: crate::cost::Counts::read(
            "WorkCounts {\n    command_buffers_submitted: 2,\n    command_buffers_begun: 3,\n    \
             queue_submissions: 2,\n}",
        )
        .unwrap(),
        runs: 8,
        stopped: None,
        degraded: Vec::new(),
        nanos: 1_000,
    };
    assert_eq!(
        apart.apart_from(&held),
        vec![
            ("command_buffers_begun", 4, 3),
            ("command_buffers_submitted", 1, 2)
        ],
        "both disagreements are reported, with the reading from each shape"
    );
    let record = apart.fold_alone_into(&held, "eight runs of each case alone".to_owned());
    // The gate's own measurement is kept, and the contrast is named beside it.
    assert_eq!(record.measured, held.measured);
    assert_eq!(
        record.contrasted.as_deref(),
        Some("eight runs of each case alone")
    );
    // The bound is the larger of the two readings, in both directions.
    assert_eq!(record.bounds.get("command_buffers_submitted"), Some(&2));
    assert_eq!(record.bounds.get("command_buffers_begun"), Some(&4));
    // And each carries the other shape's reading as the evidence for not gating it.
    assert_eq!(record.composed.get("command_buffers_submitted"), Some(&2));
    assert_eq!(record.composed.get("command_buffers_begun"), Some(&3));
    // The count the two shapes agreed on is the only one the gate holds.
    assert_eq!(record.gated(), 1);
    let gated: Vec<String> = record
        .held()
        .into_iter()
        .filter(|bound| bound.gated())
        .map(|bound| bound.count)
        .collect();
    assert_eq!(gated, vec!["queue_submissions".to_owned()]);
    // A run that does more of a reported count than either shape saw is still `Within`: no bound
    // of this case's own can hold a number the selection decides.
    let verdict = crate::cost::judge(
        std::slice::from_ref(&record),
        "a venue",
        true,
        &crate::cost::Counts::read(
            "WorkCounts {\n    command_buffers_submitted: 9,\n    command_buffers_begun: 9,\n    \
             queue_submissions: 2,\n}",
        )
        .unwrap(),
        None,
        &[],
    );
    assert!(verdict.is_within(), "{verdict}");
    // While the count both shapes agreed on still bites.
    let verdict = crate::cost::judge(
        std::slice::from_ref(&record),
        "a venue",
        true,
        &crate::cost::Counts::read("WorkCounts {\n    queue_submissions: 3,\n}").unwrap(),
        None,
        &[],
    );
    assert!(!verdict.is_within(), "{verdict}");
}

/// A count the cases-alone measurement never moved, which the gate's own shape did, is reported
/// too: a reading of zero is a reading, and the two shapes still disagree.
///
/// Without this the contrast would only ever catch a count that grew alone, and every count the
/// company does the work for would stay gated on a number the selection decides.
#[test]
fn a_count_only_one_shape_moved_is_reported_as_well() {
    let held = crate::cost::CostRecord {
        venue: "a venue".to_owned(),
        runs: 8,
        measured: "eight runs of the whole selection".to_owned(),
        contrasted: None,
        degraded: Vec::new(),
        bounds: [("commands_replayed.cmd_dispatch".to_owned(), 4)]
            .into_iter()
            .collect(),
        moving: Default::default(),
        composed: Default::default(),
        ceilings: Default::default(),
        unmeasurable: None,
    };
    let apart = crate::cost::CaseCost {
        case: "a case".to_owned(),
        // Alone the case moved a count the whole-set run never did, and did not move the one it
        // did: both directions of "only one shape saw it".
        counts: crate::cost::Counts::read("WorkCounts {\n    inline_passes: 1,\n}").unwrap(),
        least: crate::cost::Counts::read("WorkCounts {\n    inline_passes: 1,\n}").unwrap(),
        runs: 8,
        stopped: None,
        degraded: Vec::new(),
        nanos: 1_000,
    };
    assert_eq!(
        apart.apart_from(&held),
        vec![
            ("commands_replayed.cmd_dispatch", 4, 0),
            ("inline_passes", 0, 1)
        ]
    );
    let record = apart.fold_alone_into(&held, "eight runs of each case alone".to_owned());
    assert_eq!(record.gated(), 0, "neither count is the case's own");
    // The count only the lone run moved still gets a bound, or a selection where this case is the
    // first to run would be held to zero.
    assert_eq!(record.bounds.get("inline_passes"), Some(&1));
    assert_eq!(
        record.bounds.get("commands_replayed.cmd_dispatch"),
        Some(&4)
    );
}

/// The **Metal** side states every stage-independent field of a case's own
/// [`RenderPipelineSpec`](crate::render::RenderPipelineSpec).
///
/// The runtime side has a value-level guard for these
/// (`the_runtime_side_forwards_every_field_of_a_pipeline_spec`), because the stage-independent
/// fields are mapped into a value the test can build. The Metal side has no such intermediate:
/// `MTLRenderPipelineDescriptor` is an Objective-C object and making one needs a Metal device, so
/// on a host without Metal there is nothing to hold to its fields.
///
/// So for that one side this reads the source and asks whether each field is mentioned at all. A
/// mention is not a correct use, so this is deliberately the weaker kind of guard — but it is
/// mechanical, it needs no device, and a field the Metal side never names is a field it runs
/// Metal's default for while a case states a value.
///
/// It reads `render_descriptor_from`, which is where the Metal side builds the descriptor:
/// `render_pipeline` makes the state from it, and so does
/// [`Side::pipeline_both_paths`](crate::case::Side::pipeline_both_paths), which hands one
/// descriptor to a `completionHandler:` call and to its synchronous sibling. The guard follows
/// the building rather than the creation, because that is where a field is set or forgotten.
#[test]
fn the_metal_side_mentions_every_pipeline_field() {
    let source = include_str!("metal_side.rs");
    let from = source
        .find("fn render_descriptor_from(")
        .expect("no render_descriptor_from in metal_side.rs");
    let rest = &source[from..];
    // The method ends at the first `}` at one level of indentation, which closes it.
    let to = rest
        .find("\n    }\n")
        .expect("no end of render_descriptor_from");
    let body = &rest[..to];
    for field in [
        "depth_format",
        "stencil_format",
        "samples",
        "rasterization",
        "vertex_layout",
        "colors",
        "label",
        "support_indirect_command_buffers",
        "alpha_to_coverage",
        "alpha_to_one",
        "input_primitive_topology",
    ] {
        assert!(
            body.contains(field),
            "metal_side.rs's render_descriptor_from never mentions `{field}`, so it runs \
             Metal's default where the case states a value"
        );
    }
}

/// Every field of a case's [`RenderPipelineSpec`](crate::render::RenderPipelineSpec) except its two
/// stages reaches the pipeline descriptor the runtime side builds
/// (`runtime_side::pipeline_fields`), checked field by field with no device.
///
/// This is `the_runtime_side_forwards_every_field_of_a_color_state` one level up, and it is here
/// because the omission happened a second time: after the write mask was found never to be
/// assigned, `depth_format` and `stencil_format` turned out never to be assigned either, so our
/// side built every pipeline declaring no depth and no stencil attachment while the Metal side
/// declared the case's.
///
/// A whole `RenderPipelineDescriptor` cannot be built without a device — its `new` takes a
/// `StageSource`, and neither a `Function` nor a `CompiledFunction` can be made without one — so
/// the assertions are on the stage-independent value the real path maps the spec into, which is
/// not a mock.
///
/// Every field is given a value distinct from Metal's default **and** from the other fields', so a
/// field wired to its neighbour's fails here rather than passing by coincidence. The depth and
/// stencil formats especially: they are two different formats, so swapping them fails.
#[test]
fn the_runtime_side_forwards_every_field_of_a_pipeline_spec() {
    use crate::render::{
        ColorState, RenderPipelineSpec, StageFunction, VertexAttribute, VertexBufferLayout,
        VertexLayout,
    };
    use metal2vulkan::metal::{ColorWriteMask, PixelFormat, VertexFormat, VertexStepFunction};

    // Two attachments of different formats under different masks, with an unattached gap between
    // them, so an attachment written to the wrong index fails too.
    let colors = [
        Some(ColorState::new(PixelFormat::RGBA8Unorm).masked(ColorWriteMask::RED)),
        None,
        Some(ColorState::new(PixelFormat::R16Snorm).masked(ColorWriteMask::GREEN)),
    ];
    let attributes = [VertexAttribute {
        index: 3,
        format: VertexFormat::Float2,
        offset: 8,
        buffer: 5,
    }];
    let layouts = [VertexBufferLayout {
        index: 5,
        stride: 16,
        step: VertexStepFunction::PerInstance,
        rate: 2,
    }];
    let mut spec = RenderPipelineSpec::of(
        StageFunction::named("v"),
        Some(StageFunction::named("f")),
        &colors,
    );
    spec.depth_format = Some(PixelFormat::Depth32Float);
    spec.stencil_format = Some(PixelFormat::Stencil8);
    spec.samples = 4;
    spec.rasterization = false;
    spec.vertex_layout = Some(VertexLayout {
        attributes: &attributes,
        layouts: &layouts,
    });
    spec.label = Some("a-pipeline-spec-label");
    spec.support_indirect_command_buffers = true;
    spec.alpha_to_coverage = true;
    spec.alpha_to_one = true;
    // A class that is neither Metal's default nor the ordinal of any other field here, so a class
    // forwarded from a neighbour reads wrong.
    spec.input_primitive_topology = metal2vulkan::metal::PrimitiveTopologyClass::Line;
    // Every tessellation field away from Metal's default, and no two of them equal as ordinals, so
    // a field forwarded from another's reads wrong rather than passing.
    spec.tessellation = Some(
        crate::render::TessellationState::new()
            .partitioned(metal2vulkan::metal::TessellationPartitionMode::FractionalOdd)
            .stepping(metal2vulkan::metal::TessellationFactorStepFunction::PerPatchAndPerInstance)
            .clamped_to(12)
            .indexed_by(metal2vulkan::metal::TessellationControlPointIndexType::UInt16)
            .wound(metal2vulkan::metal::Winding::CounterClockwise),
    );
    assert_ne!(
        spec.depth_format, spec.stencil_format,
        "the two formats must differ, or swapping them would pass"
    );

    let made = crate::runtime_side::pipeline_fields(&spec).expect("three attachments of eight");
    assert_eq!(
        made.depth_format,
        Some(PixelFormat::Depth32Float),
        "depth_format"
    );
    assert_eq!(
        made.stencil_format,
        Some(PixelFormat::Stencil8),
        "stencil_format"
    );
    assert_eq!(made.raster_sample_count, 4, "samples");
    assert!(!made.rasterization_enabled, "rasterization");
    assert_eq!(
        made.label.as_deref(),
        Some("a-pipeline-spec-label"),
        "label"
    );
    assert!(
        made.support_indirect_command_buffers,
        "support_indirect_command_buffers"
    );
    assert!(made.alpha_to_coverage, "alpha_to_coverage");
    assert!(made.alpha_to_one, "alpha_to_one");
    assert_eq!(
        made.input_primitive_topology,
        metal2vulkan::metal::PrimitiveTopologyClass::Line,
        "input_primitive_topology"
    );
    let tessellation = made.tessellation.expect("the tessellation fields");
    assert_eq!(tessellation.max_tessellation_factor, 12, "max_factor");
    assert_eq!(
        tessellation.partition_mode,
        metal2vulkan::metal::TessellationPartitionMode::FractionalOdd,
        "partition"
    );
    assert_eq!(
        tessellation.factor_step_function,
        metal2vulkan::metal::TessellationFactorStepFunction::PerPatchAndPerInstance,
        "factor_step"
    );
    assert_eq!(
        tessellation.output_winding_order,
        metal2vulkan::metal::Winding::CounterClockwise,
        "winding"
    );
    assert_eq!(
        tessellation.control_point_index_type,
        metal2vulkan::metal::TessellationControlPointIndexType::UInt16,
        "control_point_index_type"
    );
    assert_eq!(
        tessellation.factor_format,
        metal2vulkan::metal::TessellationFactorFormat::Half,
        "factor_format"
    );
    assert!(
        !tessellation.factor_scale_enabled,
        "factor_scale_enabled stays where the spec left it"
    );

    let first = made.color_attachments[0].expect("color 0");
    assert_eq!(
        first.pixel_format,
        PixelFormat::RGBA8Unorm,
        "color 0 format"
    );
    assert_eq!(first.write_mask, ColorWriteMask::RED, "color 0 mask");
    assert!(
        made.color_attachments[1].is_none(),
        "color 1 was not attached"
    );
    let third = made.color_attachments[2].expect("color 2");
    assert_eq!(third.pixel_format, PixelFormat::R16Snorm, "color 2 format");
    assert_eq!(third.write_mask, ColorWriteMask::GREEN, "color 2 mask");
    for at in 3..8 {
        assert!(made.color_attachments[at].is_none(), "color {at}");
    }

    let vertex = made.vertex_descriptor.expect("vertex_descriptor");
    let attribute = vertex.attributes.get(&3).expect("attribute 3");
    assert_eq!(attribute.format, VertexFormat::Float2, "attribute format");
    assert_eq!(attribute.offset, 8, "attribute offset");
    assert_eq!(attribute.buffer_index, 5, "attribute buffer");
    let bound = vertex.layouts.get(&5).expect("layout 5");
    assert_eq!(bound.stride, Some(16), "layout stride");
    assert_eq!(
        bound.step_function,
        VertexStepFunction::PerInstance,
        "layout step"
    );
    assert_eq!(bound.step_rate, 2, "layout rate");

    // A spec naming a ninth colour attachment is refused rather than silently dropped.
    let past_the_eight = [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Some(ColorState::new(PixelFormat::RGBA8Unorm)),
    ];
    let past = RenderPipelineSpec::of(StageFunction::named("v"), None, &past_the_eight);
    assert!(
        crate::runtime_side::pipeline_fields(&past).is_err(),
        "a ninth colour attachment"
    );
}

/// A divergence that **is** a notice names the notice each side announced, and a known-gaps row
/// carries it.
///
/// The shape: our side announces a degrade notice where Metal announces nothing, and every other
/// field is bit-exact. The whole difference is the notice, so a row whose answer did not name it
/// would record a difference without saying what it was waiting on. The notices site used to print
/// "Metal recorded AnnouncedNothing, we observed a value", which names nothing a reader can act on.
///
/// It is a divergence and not a tolerance on purpose: where the product's own degrade note says
/// validation derives no explanation from it, loosening the site here would be this suite
/// overruling the product's stated contract.
#[test]
fn a_divergence_that_is_a_notice_names_the_notice_each_side_announced() {
    use crate::known_gaps::{against, Expectation, Expects, KnownGap};
    let case = WordsCase::new("ledger-notice-only", 2);
    // Metal announces nothing; the case names no tolerance.
    let recording = recording_of(&case, case.transcript(words(&[1, 2]), "Completed", &[]), 3);
    let code = Degradation::ShaderColorConversion.code();

    let mut side = StubSide::new(words(&[1, 2]))
        .on("llvmpipe (LLVM 21.1.8, 128 bits)")
        .announcing(&[Degradation::ShaderColorConversion]);
    let replayed = replay_against(&case, &mut side, &recording).unwrap();
    assert_eq!(replayed.verdict.word(), "divergent");
    let Verdict::Divergent { found, .. } = &replayed.verdict else {
        panic!("{:?}", replayed.verdict);
    };
    // The notice is the only difference: the two sites the notices record sit at.
    let set = found
        .iter()
        .find(|divergence| divergence.path.ends_with("/set"))
        .unwrap_or_else(|| panic!("{found:?}"));
    let answer = set.kind.to_string();
    // Metal's side of it, and ours, in the one line a row's answer field holds.
    assert!(answer.contains("AnnouncedNothing"), "{answer}");
    assert!(answer.contains(code), "{answer}");

    let rows: Vec<KnownGap> = found
        .iter()
        .map(|divergence| KnownGap {
            case: case.name().to_owned(),
            venue: "llvmpipe".to_owned(),
            expects: Expects::Divergent,
            site: divergence.path.clone(),
            answer: divergence.kind.to_string(),
            waiting_on: format!(
                "our side announces {code} on this venue where Metal announces nothing, with every \
                 other field bit-exact; the product's degrade note derives no tolerance, so this \
                 stays a difference until the notice goes"
            ),
        })
        .collect();
    assert_eq!(against(std::slice::from_ref(&replayed), &rows), Vec::new());

    // Both directions, as for any other row: the venue that does not announce it is clean, and the
    // day this venue stops announcing it the rows fail and must go.
    let mut elsewhere = StubSide::new(words(&[1, 2])).on("Apple M3 Max");
    let exact = replay_against(&case, &mut elsewhere, &recording).unwrap();
    assert_eq!(exact.verdict.word(), "exact");
    assert_eq!(against(std::slice::from_ref(&exact), &rows), Vec::new());

    let mut fixed = StubSide::new(words(&[1, 2])).on("llvmpipe (LLVM 21.1.8, 128 bits)");
    let gone = replay_against(&case, &mut fixed, &recording).unwrap();
    assert_eq!(gone.verdict.word(), "exact");
    let found = against(std::slice::from_ref(&gone), &rows);
    assert_eq!(found.len(), rows.len(), "{found:?}");
    assert!(
        found
            .iter()
            .all(|why| matches!(why, Expectation::NoLongerHeld { .. })),
        "{found:?}"
    );
}

/// A multisampled sparse texture on a device without `sparseResidencyNSamples` is the **device's**
/// gap, never a refusal of ours and never a ledger row.
///
/// This pins the cross-crate chain that made four unlisted refusals appear on lavapipe: the
/// runtime's `sparse_refusal` names the Vulkan capability the device lacks
/// (`Needs::DeviceCapability`), and the suite reads it as the venue's own gap. It used to say
/// `Needs::NotRepresentable` -- "no Vulkan mechanism exists on any device" -- which the suite could
/// only score as a gap of ours, so a device that simply does not have the feature produced a
/// refusal with no bug behind it to write down.
///
/// Measured on lavapipe through `vulkaninfo` in the Linux container (2026-10-23): `sparseBinding`
/// true, its one queue family carries `QUEUE_SPARSE_BINDING_BIT`, `sparseResidencyImage2D` and
/// `3D` true, and `sparseResidency2Samples`, `4Samples`, `8Samples` and `16Samples` all **false**.
/// So a multisampled sparse texture reaches the sample-count check there, and nothing about it is
/// ours.
#[test]
fn a_multisampled_sparse_texture_without_the_sample_count_is_the_devices_gap() {
    use metal2vulkan::error::{Feature, Needs, SparseNeed, Unsupported};
    for count in [2u32, 4, 8, 16] {
        let needs = Unsupported::device_capability(
            Feature::SparseTextures {
                missing: SparseNeed::SparseResidencySamples(count),
            },
            "x",
        );
        assert!(
            crate::verdict::device_gap_of(&needs).is_some(),
            "a device without sparseResidency{count}Samples is the device's gap"
        );
    }
    // And what the runtime has not built stays ours, by the same reading: the two must not be
    // confused in either direction.
    assert!(crate::verdict::device_gap_of(&Unsupported::not_implemented(
        Feature::SparseTextures {
            missing: SparseNeed::SparseMetadata,
        }
    ))
    .is_none());
    // The vocabulary can now say both, which is the whole point: `NotRepresentable` claims no
    // Vulkan form exists anywhere and reads as our unbuilt work, so it must not be the word for a
    // feature bit a device happens not to have.
    assert!(matches!(
        Unsupported::device_capability(Feature::Format, "sparseResidency4Samples").needs,
        Needs::DeviceCapability("sparseResidency4Samples")
    ));
}

/// A device's gap is the device's gap in **every** variant that carries a typed refusal, and our
/// unbuilt work stays ours in every one of them.
///
/// Which `Error` variant wraps an `Unsupported` says *where the refusal was decided*, never *whose
/// gap it is*: the plain form decides it as the call is made, `Error::Bind` as a dispatch or draw
/// is planned, and `Error::Compile` as a kernel the runtime dispatches for itself is built. Only
/// [`Unsupported::needs`] says whose gap it is, so the same `needs` must read the same way through
/// all three, and the carrier must not change the answer.
///
/// `Error::Compile` was the one [`crate::verdict::device_gap`] did not read, and it is reachable:
/// an extended-range attachment's shadow load is built on demand through `owned_kernel`, whose 43
/// builds wrap the compile layer's refusal in `Error::Compile`, and `extended_shadow_module`
/// refuses `Unsupported::needs(FramebufferFetch, c)` -- a `Needs::AnyOf`, the device's own gap --
/// where the device's profile lacks a capability its module needs.
/// `render/extended_shadow.rs`'s pre-instance load hands that refusal to the caller with `?`, so
/// before this a device without the capability had its gap scored as a refusal of ours.
///
/// Both halves over the same three carriers, because one direction alone is vacuous here: a test
/// that only checked the gaps would pass a `device_gap` that answered `Some` for everything.
#[test]
fn a_device_gap_reads_the_same_through_every_variant_that_carries_a_refusal() {
    use metal2vulkan::device::VkFeature;
    use metal2vulkan::error::{BindError, Capability, Error as CompileError, Feature, Unsupported};
    use metal2vulkan_runtime::Error;
    // One `Unsupported` that is the device's gap, and one that is ours, each put through every
    // carrier unchanged. Built once so the two directions cannot drift apart.
    let theirs = Unsupported::needs(
        Feature::FramebufferFetch,
        Capability::Feature(VkFeature::ShaderInt8),
    );
    let ours = Unsupported::not_implemented(Feature::FramebufferFetch);
    /// One variant that carries a typed refusal: what to call it, and how it wraps one.
    type Carrier = (&'static str, fn(Unsupported) -> Error);
    let carriers: [Carrier; 3] = [
        ("Error::Unsupported", Error::Unsupported),
        ("Error::Bind", |u| Error::Bind(BindError::Unsupported(u))),
        ("Error::Compile", |u| {
            Error::Compile(CompileError::Unsupported(u))
        }),
    ];
    for (carrier, wrap) in carriers {
        let gap = wrap(theirs.clone());
        assert!(
            crate::verdict::device_gap(&gap).is_some(),
            "{carrier} carries the same typed refusal, so a device's gap in it is the device's gap"
        );
        // And through the one call every refusal of the suite passes: `Failure::of` is where
        // `lacks` is set, so a carrier it does not read loses the fact for the whole run.
        assert!(
            Failure::of(&gap).lacks.is_some(),
            "{carrier}: a refusal the venue cannot answer must reach the run as the venue's gap"
        );
        let mine = wrap(ours.clone());
        assert!(
            crate::verdict::device_gap(&mine).is_none(),
            "{carrier}: what we have not built stays ours to answer for"
        );
        assert!(
            Failure::of(&mine).lacks.is_none(),
            "{carrier}: our own unbuilt work must not be excused as the venue's gap"
        );
    }
}

/// Every field of a [`crate::texture::TextureRequest`] reaches **both** sides' texture
/// descriptors, checked field by field with no device, each at a value no other field holds.
///
/// It is written for the swizzle families, where a dropped harness field is the easiest way to
/// measure nothing at all: if one side set the swizzle and the other did not, both legs would be
/// reading the texture's *unswizzled* texels, every swizzle case would compare equal, and the
/// suite would report a sweep of the question neither leg was asked. The four channels therefore
/// take four **different** sources, so a dropped, duplicated or transposed channel cannot pass by
/// coincidence.
///
/// The Metal half runs wherever Metal's frameworks are: making an `MTLTextureDescriptor` asks
/// nothing of a device.
#[test]
fn both_sides_state_every_field_of_a_texture_request() {
    use crate::texture::TextureRequest;
    use metal2vulkan::metal::{PixelFormat, TextureSwizzle, TextureSwizzleChannels, TextureType};

    // Four different sources, so no channel can take another's and still compare equal.
    let swizzle = TextureSwizzleChannels::new(
        TextureSwizzle::Blue,
        TextureSwizzle::Alpha,
        TextureSwizzle::Zero,
        TextureSwizzle::Red,
    );
    let request = TextureRequest::read_2d(PixelFormat::RG16Uint.as_raw(), 7, 5)
        .levels(3)
        .used(metal2vulkan::metal::TextureUsage::SHADER_WRITE)
        .swizzled(swizzle)
        .private();

    let ours = crate::runtime_side::texture_descriptor(&request, PixelFormat::RG16Uint);
    assert_eq!(ours.pixel_format, PixelFormat::RG16Uint);
    assert_eq!(ours.texture_type, TextureType::Type2D);
    assert_eq!([ours.width, ours.height, ours.depth], [7, 5, 1]);
    assert_eq!(ours.mipmap_level_count, 3);
    assert_eq!(ours.array_length, 1);
    assert_eq!(
        ours.usage,
        metal2vulkan::metal::TextureUsage::SHADER_WRITE,
        "the usage the case stated, and nothing added to it"
    );
    assert!(ours.allow_gpu_optimized_contents);
    assert_eq!(ours.swizzle, swizzle, "every channel's own source");

    #[cfg(target_os = "macos")]
    {
        use objc2_metal::{MTLStorageMode, MTLTextureDescriptor, MTLTextureSwizzle};
        let theirs: objc2::rc::Retained<MTLTextureDescriptor> =
            crate::metal_side::texture_descriptor(&request);
        assert_eq!(
            theirs.pixelFormat().0 as u32,
            PixelFormat::RG16Uint.as_raw()
        );
        assert_eq!(theirs.textureType().0 as u32, TextureType::Type2D.as_raw());
        assert_eq!([theirs.width(), theirs.height(), theirs.depth()], [7, 5, 1]);
        assert_eq!(theirs.mipmapLevelCount(), 3);
        assert_eq!(theirs.arrayLength(), 1);
        assert_eq!(
            theirs.usage().0 as u32,
            metal2vulkan::metal::TextureUsage::SHADER_WRITE.as_raw()
        );
        assert_eq!(
            theirs.storageMode(),
            MTLStorageMode::Private,
            "the case asked for private storage"
        );
        assert!(theirs.allowGPUOptimizedContents());
        // The one field the whole family turns on, channel by channel against the case's own.
        let held = theirs.swizzle();
        let raw = |source: TextureSwizzle| MTLTextureSwizzle(source.as_raw() as u8);
        assert_eq!(held.red, raw(swizzle.red), "red");
        assert_eq!(held.green, raw(swizzle.green), "green");
        assert_eq!(held.blue, raw(swizzle.blue), "blue");
        assert_eq!(held.alpha, raw(swizzle.alpha), "alpha");
        // And the four are really four different sources, so the four assertions above are four
        // facts rather than one repeated.
        let sources = [held.red, held.green, held.blue, held.alpha];
        let distinct: std::collections::BTreeSet<u8> =
            sources.iter().map(|source| source.0).collect();
        assert_eq!(distinct.len(), 4, "{sources:?}");
    }
}

/// Every case states the inputs its own recording's answers were taken from, and the cases that
/// no longer do are exactly the ones named here.
///
/// A case whose [`crate::case::Case::inputs`] no longer fingerprint to its recording's is
/// `Unrecorded` at replay. That verdict is this suite's bookkeeping, not Metal's answer, and it
/// names no cause: the case reads exactly as it reads when no one has recorded it at all. What
/// moves a fingerprint is a widening of a shared vocabulary -- a field on a texture request, a
/// step on a work list -- and it moves it for every family that states the widened thing, not
/// only for the family the widening was made for. Found through a replay, a restaled family is
/// one whole-set GPU run plus a name-by-name verdict diff away, and it still looks like a family
/// nobody has recorded.
///
/// This is the same fact with no device and no oracle: it reads the committed recordings and
/// names the case and the input whose fingerprint moved.
///
/// The cases that have moved are held as a **set**, through [`STALE`] and [`stale_set_diff`]: a
/// re-recording one without striking it from here fails too. All three state an unwritten buffer,
/// whose fingerprint was widened from the buffer's length alone to its length and its storage
/// mode after their answers were taken. Their Metal answers did not move; only the key did, so
/// what each needs is a re-record. The lengths behind the three recorded digests, recovered from
/// them, are 64, 4099 and 614400 bytes.
#[test]
fn every_cases_inputs_are_the_ones_its_recording_was_taken_from() {
    /// The cases whose committed answers were taken under an older input fingerprint.
    ///
    /// Empty, and it is meant to stay empty: a case here is a recording that no longer belongs to
    /// its case, which is a bug to fix rather than a row to keep. The three buffer-blit cases that
    /// were listed were fixed by writing a private buffer's storage word only where the storage is
    /// not private, which restored the bytes their recordings were taken from exactly, so nothing
    /// was re-recorded.
    const STALE: [&str; 0] = [];
    let mut moved: std::collections::BTreeMap<&str, Vec<String>> =
        std::collections::BTreeMap::new();
    let mut checked = 0usize;
    let mut unrecorded = Vec::new();
    for case in crate::cases::all() {
        let Ok(recording) = crate::recording::read(case.name()) else {
            // A case nothing has recorded yet is unrecorded for that reason alone.
            unrecorded.push(case.name());
            continue;
        };
        let inputs = case
            .inputs()
            .unwrap_or_else(|why| panic!("{}: {why}", case.name()));
        let fed = crate::case::inputs_node(&inputs);
        for answer in &recording.answers {
            for outcome in &answer.outcomes {
                let Some(recorded) = outcome.transcript.root.child(crate::case::INPUTS_SITE) else {
                    moved
                        .entry(case.name())
                        .or_default()
                        .push("its recording states no input fingerprint".to_owned());
                    continue;
                };
                checked += 1;
                if *recorded == fed {
                    continue;
                }
                let why = moved.entry(case.name()).or_default();
                for input in &recorded.children {
                    match fed.child(&input.name) {
                        Some(ours) if ours.observed == input.observed => {}
                        Some(_) => why.push(format!("{} differs", input.name)),
                        None => why.push(format!("{} is missing", input.name)),
                    }
                }
                for input in &fed.children {
                    if recorded.child(&input.name).is_none() {
                        why.push(format!("{} is extra", input.name));
                    }
                }
            }
        }
    }
    assert!(
        checked > 0,
        "the suite holds no recorded answer to check an input fingerprint against"
    );
    assert_eq!(
        stale_set_diff(&moved, &STALE),
        None,
        "{checked} recorded answers checked (cases with no recording at all, which this says \
         nothing about: {})",
        unrecorded.join(", ")
    );
}

/// Every AIR input a case states is the digest of a **committed fixture text**, never of the
/// bitcode `llvm-as` assembles from one.
///
/// `llvm-as`'s output is a function of the text *and the assembler's release*: the same
/// `fixtures/public/fragment_constant_color.ll` assembles to `1a4cc972040870f8` under Homebrew
/// LLVM 23.1.4 and to `38d752a345c98db2` under 23.1.1. So a fingerprint holding those bytes can
/// only match on the host that assembled them, and nine cases of the archive and dynamic-library
/// families read `unrecorded: the inputs this run fed are not the recording's: air differs` on
/// every other host. The sharper cost is the one that made this a test: a case that never compared
/// holds no verdict, so a `--check` run off the recording host reported two `known-gaps.tsv` rows
/// as looking fixed and would have retired live ledger rows.
///
/// [`crate::recording::ShaderRef`] already states the rule for a recording's own shader digests --
/// the text is what the tree commits and what every host reads alike. This holds the **input**
/// fingerprint to the same rule, and it is the rule rather than a list of case names: an AIR input
/// must be the digest of one of the texts the case's own [`crate::case::Case::shaders`] names.
///
/// It also means no case needs an assembler to state its fingerprint, which is why this test runs
/// where `llvm-as` does not.
#[test]
fn every_air_input_is_a_committed_fixture_texts_digest() {
    let mut checked = 0usize;
    let mut wrong = Vec::new();
    for case in crate::cases::all() {
        let shaders = case
            .shaders()
            .unwrap_or_else(|why| panic!("{}: {why}", case.name()));
        let committed: std::collections::BTreeMap<&str, &str> = shaders
            .iter()
            .map(|shader| (shader.source_sha256.as_str(), shader.fixture.as_str()))
            .collect();
        let inputs = case
            .inputs()
            .unwrap_or_else(|why| panic!("{}: {why}", case.name()));
        for input in inputs {
            // The AIR inputs, however a family names them: `air`, `air-typed`, `air-0`,
            // `air:px_positions`.
            if !input.name.starts_with("air") {
                continue;
            }
            checked += 1;
            if committed.contains_key(input.digest.hex.as_str()) {
                continue;
            }
            wrong.push(format!(
                "{}: {} is {}, which is none of the texts the case names ({})",
                case.name(),
                input.name,
                input.digest.hex,
                committed.values().copied().collect::<Vec<_>>().join(", "),
            ));
        }
    }
    assert!(
        checked > 0,
        "no case states an AIR input, so this says nothing"
    );
    assert!(
        wrong.is_empty(),
        "{checked} AIR inputs checked; these are digests of something the tree does not commit, \
         so they can only match on the host that produced them:\n{}",
        wrong.join("\n")
    );
}

/// Why the cases whose input fingerprints have moved are not the ones named as stale, if they are
/// not: what [`every_cases_inputs_are_the_ones_its_recording_was_taken_from`] fails with, or
/// `None` when the two sets are the same.
///
/// `moved` holds each case that no longer states its recording's inputs, with what moved for it;
/// `named` is the set the test states. Comparing them as sets rather than asserting a floor of
/// zero is what makes both halves hold: a case that moved and is named nowhere fails, and a name
/// whose case states its recording's inputs again fails too.
///
/// It is a helper rather than an assertion inside the test so that
/// [`a_moved_fingerprint_and_a_name_that_outlived_its_case_both_fail`] can exercise both halves
/// over fabricated sets. With `STALE` empty, the second half is unreachable from the test itself
/// however it is worded, because nothing reaches that path until a name exists -- and an
/// unreachable direction is not a direction the suite holds.
fn stale_set_diff(
    moved: &std::collections::BTreeMap<&str, Vec<String>>,
    named: &[&str],
) -> Option<String> {
    let found: std::collections::BTreeSet<&str> = moved.keys().copied().collect();
    let named: std::collections::BTreeSet<&str> = named.iter().copied().collect();
    if found == named {
        return None;
    }
    let mut said = Vec::new();
    let unexpected: Vec<&str> = found.difference(&named).copied().collect();
    if !unexpected.is_empty() {
        said.push(format!(
            "these cases no longer state the inputs their recordings were taken from, so each is \
             Unrecorded at replay for a reason that is this suite's and not Metal's:\n{}",
            unexpected
                .iter()
                .map(|case| format!("{case}: {}", moved[case].join(", ")))
                .collect::<Vec<_>>()
                .join("\n")
        ));
    }
    let spent: Vec<&str> = named.difference(&found).copied().collect();
    if !spent.is_empty() {
        said.push(format!(
            "these cases are named as stale but state the inputs their recordings were taken \
             from, so each name stands behind nothing and must go: {spent:?}"
        ));
    }
    Some(said.join("\n"))
}

/// Both directions of the stale set hold, over fabricated sets: a case whose fingerprint moved
/// and that nothing names fails, and a name whose case states its recording's inputs again fails.
///
/// The second direction is the one that cannot be reached through
/// [`every_cases_inputs_are_the_ones_its_recording_was_taken_from`] while `STALE` is empty, which
/// is exactly why it is checked here instead of being read off a passing run.
#[test]
fn a_moved_fingerprint_and_a_name_that_outlived_its_case_both_fail() {
    let none = std::collections::BTreeMap::new();
    assert_eq!(stale_set_diff(&none, &[]), None);
    let mut moved = std::collections::BTreeMap::new();
    moved.insert(
        "buffer-fills-of-every-size",
        vec!["private-buffer:0 differs".to_owned()],
    );
    let said = stale_set_diff(&moved, &[]).expect("a moved fingerprint nothing names");
    assert!(said.contains("buffer-fills-of-every-size"), "{said}");
    assert!(said.contains("private-buffer:0 differs"), "{said}");
    assert!(said.contains("not Metal's"), "{said}");
    // Naming it is what parks it, and it parks that case alone.
    assert_eq!(
        stale_set_diff(&moved, &["buffer-fills-of-every-size"]),
        None
    );
    let said = stale_set_diff(&moved, &["buffer-blit-any-range"])
        .expect("a moved case parked under another case's name");
    assert!(said.contains("buffer-fills-of-every-size"), "{said}");
    assert!(said.contains("buffer-blit-any-range"), "{said}");
    // And the name has to go when the case comes back, whether by a re-record or, as the three
    // buffer-blit cases did, by the fingerprint keeping the bytes it always had.
    let said = stale_set_diff(&none, &["buffer-fills-of-every-size"])
        .expect("a name that outlived its case");
    assert!(said.contains("buffer-fills-of-every-size"), "{said}");
    assert!(said.contains("stands behind nothing"), "{said}");
}

/// A count whose reading with the case alone is its ceiling keeps its gate there rather than being
/// reported — and loses it again the moment a measurement reads the lone run *lower*, because that
/// is the fact its row rests on.
///
/// This is the one place the contrast gives back what it takes away, so it is the one place a wrong
/// name would quietly un-gate nothing and over-gate everything else. The table's rows are held to
/// the data that justified them: `device_memory_allocated` is monotone over the same 393 cases and
/// is not a row, because its two shapes differ by 268,435,456 and a bound no regression can reach
/// is worse than none.
#[test]
fn a_count_whose_ceiling_is_the_lone_run_is_gated_there() {
    use crate::cost::monotone;
    for count in [
        "addressable_buffers_made",
        "pooled_memory_blocks_made",
        "arena_pools_made",
        "creation_pipelines_made",
        "variant_pipelines_made",
        "shader_modules_made",
        "pipeline_layouts_made",
    ] {
        assert!(monotone(count).is_some(), "{count}");
    }
    // Monotone over the same measurement, and deliberately not a row: a bound of 256 MiB against
    // an in-set reading of 0 cannot fail.
    assert!(monotone("device_memory_allocated").is_none());
    // Counts the 393-case measurement found moving both ways, which no single shape bounds.
    for count in ["command_buffers_begun", "command_buffers_submitted"] {
        assert!(monotone(count).is_none(), "{count}");
    }
    let held = |count: &str, bound: u64| crate::cost::CostRecord {
        venue: "a venue".to_owned(),
        runs: 8,
        measured: "eight runs of the whole selection".to_owned(),
        contrasted: None,
        degraded: Vec::new(),
        bounds: [(count.to_owned(), bound)].into_iter().collect(),
        moving: Default::default(),
        composed: Default::default(),
        ceilings: Default::default(),
        unmeasurable: None,
    };
    let alone = |text: &str| crate::cost::CaseCost {
        case: "a case".to_owned(),
        counts: crate::cost::Counts::read(text).expect("the shape reads"),
        least: crate::cost::Counts::read(text).expect("the shape reads"),
        runs: 8,
        stopped: None,
        degraded: Vec::new(),
        nanos: 1_000,
    };
    // One layout in company, two with nothing before it: gated at the two, not reported.
    let record = alone("WorkCounts {\n    pipeline_layouts_made: 2,\n}").fold_alone_into(
        &held("pipeline_layouts_made", 1),
        "eight runs alone".to_owned(),
    );
    assert!(
        record.composed.is_empty(),
        "its ceiling is not a disagreement"
    );
    assert_eq!(record.bounds.get("pipeline_layouts_made"), Some(&2));
    assert_eq!(record.gated(), 1);
    // And the bound bites at three.
    let over = |record: &crate::cost::CostRecord, text: &str| {
        !crate::cost::judge(
            std::slice::from_ref(record),
            "a venue",
            true,
            &crate::cost::Counts::read(text).expect("the shape reads"),
            None,
            &[],
        )
        .is_within()
    };
    assert!(over(
        &record,
        "WorkCounts {\n    pipeline_layouts_made: 3,\n}"
    ));
    assert!(!over(
        &record,
        "WorkCounts {\n    pipeline_layouts_made: 2,\n}"
    ));
    // The direction is checked, not assumed. A lone run reading *lower* than the in-set one breaks
    // the claim the row rests on, so that count is reported like any other the selection decides.
    let record = alone("WorkCounts {\n    pipeline_layouts_made: 1,\n}").fold_alone_into(
        &held("pipeline_layouts_made", 2),
        "eight runs alone".to_owned(),
    );
    assert_eq!(record.composed.get("pipeline_layouts_made"), Some(&1));
    assert_eq!(record.gated(), 0);
    assert!(!over(
        &record,
        "WorkCounts {\n    pipeline_layouts_made: 9,\n}"
    ));
    // What the command prints and what the record stores are the same judgement, held together
    // here because they are two code paths over the same rule and nothing else compares them: a
    // count `apart_from` prints as a disagreement is exactly one `fold_alone_into` writes to
    // `composed`, for the ceiling counts and the ordinary ones alike.
    for (text, bound) in [
        ("WorkCounts {\n    pipeline_layouts_made: 2,\n}", 1),
        ("WorkCounts {\n    pipeline_layouts_made: 1,\n}", 2),
        ("WorkCounts {\n    command_buffers_begun: 2,\n}", 1),
        ("WorkCounts {\n    command_buffers_begun: 1,\n}", 2),
    ] {
        let count = text
            .trim_start_matches("WorkCounts {\n    ")
            .split(':')
            .next()
            .expect("the shape names one count");
        let record = held(count, bound);
        let measured = alone(text);
        let printed: Vec<&str> = measured
            .apart_from(&record)
            .into_iter()
            .map(|(count, _, _)| count)
            .collect();
        let stored = measured.fold_alone_into(&record, "eight runs alone".to_owned());
        let kept: Vec<&str> = stored.composed.keys().map(String::as_str).collect();
        assert_eq!(printed, kept, "{text} against a bound of {bound}");
    }
    // A count with no row is reported whichever way it differs, so the table is the whole of the
    // exception.
    let record = alone("WorkCounts {\n    device_memory_allocated: 268435456,\n}").fold_alone_into(
        &held("device_memory_allocated", 0),
        "eight runs alone".to_owned(),
    );
    assert_eq!(
        record.composed.get("device_memory_allocated"),
        Some(&268_435_456)
    );
    assert_eq!(record.gated(), 0);
}

/// Re-measuring the gate's own shape over an unchanged tree changes **nothing**.
///
/// This is the check that would have caught the defect it was written for, so it belongs in the
/// suite rather than in someone's memory. A measurement of the whole case set says nothing new
/// about what each case does alone, so every piece of evidence the lone run established has to
/// survive the re-measure — and it survives only because it is written in the record rather than
/// left implicit in a number this path recomputes.
///
/// The sharp case is a ceiling. Those counts are **gated**, and the gate's own shape reads a
/// *lower* number for every one of them by construction: that is what makes the lone run's reading
/// the ceiling. Before `CostRecord::ceilings` existed, a ceiling lived only as a silently-raised
/// entry in `bounds`, this path rebuilt `bounds` from the new measurement, and 389 ceiling bounds
/// vanished across 149 cases with nothing in any output saying so. The gate kept passing. It was
/// simply weaker, on exactly the family the ceiling exception exists to protect.
#[test]
fn a_second_measurement_of_the_same_shape_changes_nothing() {
    // A record that has had its contrast folded in: one ceiling count, one the selection decides,
    // one whose runs disagreed, and one plain gated count.
    let held = crate::cost::CostRecord {
        venue: "a venue".to_owned(),
        runs: 8,
        measured: "eight runs of the whole selection".to_owned(),
        contrasted: Some("eight runs of each case alone".to_owned()),
        degraded: Vec::new(),
        bounds: [
            ("pipeline_layouts_made".to_owned(), 2),
            ("command_buffers_begun".to_owned(), 4),
            ("render_passes_made".to_owned(), 5),
            ("queue_submissions".to_owned(), 1),
        ]
        .into_iter()
        .collect(),
        moving: [("render_passes_made".to_owned(), 1)].into_iter().collect(),
        composed: [("command_buffers_begun".to_owned(), 3)]
            .into_iter()
            .collect(),
        ceilings: [("pipeline_layouts_made".to_owned(), 2)]
            .into_iter()
            .collect(),
        unmeasurable: None,
    };
    // The whole set measured again, reading exactly what it read before. For the ceiling count
    // that is the *lower* number, which is the whole point: 1 in company against 2 alone.
    let again = crate::cost::CaseCost {
        case: "a case".to_owned(),
        counts: crate::cost::Counts::read(
            "WorkCounts {\n    pipeline_layouts_made: 1,\n    command_buffers_begun: 4,\n    \
             render_passes_made: 5,\n    queue_submissions: 1,\n}",
        )
        .unwrap(),
        least: crate::cost::Counts::read(
            "WorkCounts {\n    pipeline_layouts_made: 1,\n    command_buffers_begun: 4,\n    \
             render_passes_made: 1,\n    queue_submissions: 1,\n}",
        )
        .unwrap(),
        runs: 8,
        stopped: None,
        degraded: Vec::new(),
        nanos: 1_000,
    };
    let fresh = again
        .record("a venue", "eight runs of the whole selection".to_owned())
        .carrying_the_contrast_of(&held);
    assert_eq!(
        fresh, held,
        "a re-measure of the same shape moved the record"
    );
    // And again, because an operation that is not idempotent drifts a little on every re-measure
    // rather than all at once.
    let twice = again
        .record("a venue", "eight runs of the whole selection".to_owned())
        .carrying_the_contrast_of(&fresh);
    assert_eq!(twice, held);
    // The ceiling is what makes this test bite: without it carried, the bound would fall to the
    // number this composition happened to share, and the gate would quietly stop holding 2.
    assert_eq!(fresh.bounds.get("pipeline_layouts_made"), Some(&2));
    assert_eq!(fresh.ceilings.get("pipeline_layouts_made"), Some(&2));
    assert!(
        fresh
            .bound("pipeline_layouts_made")
            .expect("it is bounded")
            .gated(),
        "a ceiling count is gated, which is why losing its bound is a silent weakening"
    );
    // A bound never rises just because it was higher last time: that would ratchet upward for ever
    // and hide a genuine reduction, and a gate that can only loosen passes everything in the end.
    let lower = crate::cost::CostRecord {
        bounds: [("queue_submissions".to_owned(), 9)].into_iter().collect(),
        ..held.clone()
    };
    let fresh = again
        .record("a venue", "eight runs of the whole selection".to_owned())
        .carrying_the_contrast_of(&lower);
    assert_eq!(
        fresh.bounds.get("queue_submissions"),
        Some(&1),
        "a plain gated count takes the new measurement's reading, not the old bound"
    );
}

/// A ceiling is written into the record as a fact, not left to be inferred from a raised bound.
#[test]
fn a_ceiling_is_recorded_as_the_evidence_for_its_bound() {
    let held = crate::cost::CostRecord {
        venue: "a venue".to_owned(),
        runs: 8,
        measured: "whole set".to_owned(),
        contrasted: None,
        degraded: Vec::new(),
        bounds: [("queue_submissions".to_owned(), 1)].into_iter().collect(),
        moving: Default::default(),
        composed: Default::default(),
        ceilings: Default::default(),
        unmeasurable: None,
    };
    // A monotone count the lone run moved and the whole-set run never did at all: the case that
    // produced 389 vanishing bounds, because nothing wrote down where the bound came from.
    let apart = crate::cost::CaseCost {
        case: "a case".to_owned(),
        counts: crate::cost::Counts::read(
            "WorkCounts {\n    addressable_buffers_made: 2,\n    queue_submissions: 1,\n}",
        )
        .unwrap(),
        least: crate::cost::Counts::read(
            "WorkCounts {\n    addressable_buffers_made: 2,\n    queue_submissions: 1,\n}",
        )
        .unwrap(),
        runs: 8,
        stopped: None,
        degraded: Vec::new(),
        nanos: 1,
    };
    let folded = apart.fold_alone_into(&held, "alone".to_owned());
    assert_eq!(folded.bounds.get("addressable_buffers_made"), Some(&2));
    assert_eq!(folded.ceilings.get("addressable_buffers_made"), Some(&2));
    assert!(
        folded.composed.is_empty(),
        "a ceiling is not a disagreement"
    );
    assert_eq!(folded.gated(), 2, "both counts are held");
    // A count the table does not name gets no ceiling, whichever way it differs, so the map is
    // exactly the exception and never a second place the rule is decided.
    let apart = crate::cost::CaseCost {
        counts: crate::cost::Counts::read("WorkCounts {\n    inline_passes: 2,\n}").unwrap(),
        least: crate::cost::Counts::read("WorkCounts {\n    inline_passes: 2,\n}").unwrap(),
        ..apart
    };
    let folded = apart.fold_alone_into(&held, "alone".to_owned());
    assert!(folded.ceilings.is_empty());
    assert_eq!(folded.composed.get("inline_passes"), Some(&2));
}

/// Reading a committed recording and writing it back changes **no byte**, for every case in the set.
///
/// The corollary the ceiling defect taught: *the check that an invariant holds must survive the
/// operation twice.* A record is written by one measurement and read by every later one, so any
/// asymmetry between the two — a field the writer omits and the reader defaults differently, a map
/// that serialises in an order its type does not fix, a number that round-trips through a narrower
/// form — drifts a little on each pass and is invisible on any single one. The cost half alone added
/// five fields that serialise only when non-empty, which is exactly the shape that hides such a
/// drift.
///
/// This holds the whole committed set rather than one synthetic record, because the interesting
/// combinations are the ones real measurements produced: a case with ceilings and no `composed`, one
/// with `moving` and no contrast, one with every map empty.
#[test]
fn reading_a_recording_and_writing_it_back_changes_no_byte() {
    let mut checked = 0;
    let mut drifted = Vec::new();
    for case in crate::cases::all() {
        let path = crate::recording::recording_path(case.name());
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let held: crate::recording::Recording = match serde_json::from_str(&text) {
            Ok(held) => held,
            Err(why) => {
                drifted.push(format!("{}: does not parse: {why}", case.name()));
                continue;
            }
        };
        let mut again = serde_json::to_string_pretty(&held).expect("it serialises");
        again.push('\n');
        if again != text {
            drifted.push(format!(
                "{}: {} bytes in, {} bytes out",
                case.name(),
                text.len(),
                again.len()
            ));
        }
        checked += 1;
    }
    assert!(
        drifted.is_empty(),
        "{} of {checked} recording(s) changed on a read-and-write round trip:\n  {}",
        drifted.len(),
        drifted.join("\n  ")
    );
    assert!(checked > 0, "no recording was read, so this proved nothing");
}

/// The diff tool sees the gate's own facts and is blind to the noise around them.
///
/// A re-measure rewrites a whole record and most of it legitimately differs between two runs of the
/// same binary over the same cases: a reported count's bound, a range the union widened, a count
/// the selection decides. So a diff of the files says nothing. The one time it said something — 389
/// ceilings gone across 149 cases — the signal sat under a hundred files of noise and was read as
/// something else first, which is why this comparison exists and why it is of four facts only.
#[test]
fn the_diff_tool_reads_the_gate_and_not_the_noise() {
    use crate::cost::{drift, Drifted};
    let record = |bounds: &[(&str, u64)],
                  moving: &[(&str, u64)],
                  ceilings: &[(&str, u64)]|
     -> crate::cost::CostRecord {
        crate::cost::CostRecord {
            venue: "a venue".to_owned(),
            runs: 8,
            measured: "whole set".to_owned(),
            contrasted: Some("alone".to_owned()),
            degraded: Vec::new(),
            bounds: bounds.iter().map(|(n, v)| ((*n).to_owned(), *v)).collect(),
            moving: moving.iter().map(|(n, v)| ((*n).to_owned(), *v)).collect(),
            composed: Default::default(),
            ceilings: ceilings
                .iter()
                .map(|(n, v)| ((*n).to_owned(), *v))
                .collect(),
            unmeasurable: None,
        }
    };
    let what = |moved: &[crate::cost::Drift]| -> Vec<(String, Drifted)> {
        moved
            .iter()
            .map(|one| (one.count.clone(), one.what))
            .collect()
    };

    // Noise the tool must not report: a reported count's bound moving, and a range widening.
    // `object_churn.buffers_evicted` is reported by the race rule and `render_passes_made` by this
    // record's own `moving` entry.
    let held = record(
        &[
            ("queue_submissions", 2),
            ("object_churn.buffers_evicted", 4),
            ("render_passes_made", 5),
        ],
        &[("render_passes_made", 1)],
        &[],
    );
    let noisy = record(
        &[
            ("queue_submissions", 2),
            ("object_churn.buffers_evicted", 40),
            ("render_passes_made", 9),
        ],
        &[("render_passes_made", 0)],
        &[],
    );
    assert!(
        drift(&held, &noisy).is_empty(),
        "reported counts and widened ranges are noise: {:?}",
        what(&drift(&held, &noisy))
    );

    // A gated bound moving is the gate's own number, in either direction: a re-measure that raises
    // one blesses a regression into the bounds, and one that lowers it is a reduction nobody saw.
    for now in [1, 3] {
        let moved = drift(&held, &record(&[("queue_submissions", now)], &[], &[]));
        assert!(
            moved.iter().any(|one| one.count == "queue_submissions"
                && one.what == Drifted::GatedBound
                && one.was == Some(2)
                && one.now == Some(now)),
            "a gated bound moving to {now} is reported: {:?}",
            what(&moved)
        );
        assert!(moved.iter().any(|one| one.what.is_load_bearing()));
    }

    // A ceiling going missing: the defect this tool exists for. The bound still looks like a bound,
    // and the number under it is gone.
    let held = record(
        &[("pipeline_layouts_made", 2), ("queue_submissions", 2)],
        &[],
        &[("pipeline_layouts_made", 2)],
    );
    let lost = record(
        &[("pipeline_layouts_made", 1), ("queue_submissions", 2)],
        &[],
        &[],
    );
    let moved = drift(&held, &lost);
    assert!(
        moved.iter().any(|one| one.count == "pipeline_layouts_made"
            && one.what == Drifted::Ceiling
            && one.was == Some(2)
            && one.now.is_none()),
        "a vanished ceiling is reported: {:?}",
        what(&moved)
    );
    assert!(moved.iter().any(|one| one.what.is_load_bearing()));

    // Reach moving either way is reported and never fatal: reclassifying a count is deliberate
    // work and whoever does it is reading the output, where a number moving under a gate that
    // still looks the same is what nobody reads.
    let held = record(&[("render_passes_made", 5)], &[], &[]);
    let ungated = record(
        &[("render_passes_made", 5)],
        &[("render_passes_made", 1)],
        &[],
    );
    let moved = drift(&held, &ungated);
    assert_eq!(
        what(&moved),
        vec![("render_passes_made".to_owned(), Drifted::GateLost)]
    );
    assert!(!moved.iter().any(|one| one.what.is_load_bearing()));
    let moved = drift(&held, &ungated);
    assert_eq!(
        crate::cost::reach(&moved),
        crate::cost::Reach {
            given_up: 1,
            taken_on: 0
        },
        "a count the measurement stops gating is counted as reach given up, never taken on"
    );
    let moved = drift(&ungated, &held);
    assert_eq!(
        what(&moved),
        vec![("render_passes_made".to_owned(), Drifted::GateGained)]
    );
    assert!(!moved.iter().any(|one| one.what.is_load_bearing()));
    // The two directions are reported apart because only one of them is dangerous, so the tally
    // must not be able to swap them: the same pair of records compared each way has to put its
    // one count in the opposite field.
    assert_eq!(
        crate::cost::reach(&moved),
        crate::cost::Reach {
            given_up: 0,
            taken_on: 1
        },
        "a count the measurement starts gating is counted as reach taken on, never given up"
    );

    // And a record compared with itself drifts in nothing, so the tool cannot cry wolf on a
    // measurement that read exactly what the record holds.
    assert!(drift(&held, &held).is_empty());
}

/// The counts a whole re-measure moved, which is the shape the first question needs.
///
/// Case by case says what moved; by count it says whether the drift belongs to the path that
/// changed. The two things this can get wrong are counting one case twice for a count that drifted
/// in two ways there, and an order that is not the same on every run.
#[test]
fn the_counts_a_re_measure_moved_are_summed_over_cases_once_each() {
    use crate::cost::{counts_that_moved, Drift, Drifted, Moved, Region};
    let one = |count: &str, what: Drifted| Drift {
        count: count.to_owned(),
        region: Region::Submission,
        what,
        was: Some(2),
        now: Some(3),
    };
    // Three cases. `command_buffers_begun` moved in all three, and in the first it moved twice --
    // its bound and its gate -- which is still one case.
    let per_case = vec![
        vec![
            one("command_buffers_begun", Drifted::GatedBound),
            one("command_buffers_begun", Drifted::GateLost),
            one("host_texel_copies", Drifted::GatedBound),
        ],
        vec![
            one("command_buffers_begun", Drifted::GatedBound),
            one("host_texel_copies", Drifted::GatedBound),
        ],
        vec![one("command_buffers_begun", Drifted::GateGained)],
    ];
    let moved = counts_that_moved(&per_case);
    assert_eq!(
        moved,
        vec![
            Moved {
                count: "command_buffers_begun".to_owned(),
                cases: 3,
                gated: 2,
            },
            Moved {
                count: "host_texel_copies".to_owned(),
                cases: 2,
                gated: 2,
            },
        ],
        "a count is one case's drift however many lines it took there, most cases first"
    );
    // Reach alone is drift a reader must still see, with nothing gated under it.
    let moved = counts_that_moved(&[vec![one("submit_infos", Drifted::GateGained)]]);
    assert_eq!(moved[0].cases, 1);
    assert_eq!(moved[0].gated, 0);
    // Nothing moved, nothing summed: the summary cannot invent a row.
    assert!(counts_that_moved(&[]).is_empty());
    assert!(counts_that_moved(&[vec![], vec![]]).is_empty());
}

/// A side for the creation-handler family: the real case body, with the handler's delivery stated
/// by the test rather than taken from a device.
///
/// It exists for one reason — to **falsify** the family's arrival site. An asynchronous case can
/// pass by never running its handler at all, so the one check that must be shown to fail is
/// "the handler did not arrive", and there is no way to make a real Metal handler not run.
struct HandlerStub {
    /// Whether the creation handler ran. `false` is the suppressed callback.
    arrives: bool,
}

/// The facts the stub's delivered compute state answers, whatever asked.
fn stub_compute_facts() -> crate::pipeline_facts::ComputeStateFacts {
    crate::pipeline_facts::ComputeStateFacts {
        label: None,
        device_is_the_cases: true,
        static_threadgroup_memory_length: 0,
        support_indirect_command_buffers: false,
        imageblock_memory_length: 0,
        required_threads_per_threadgroup: [0, 0, 0],
    }
}

impl Side for HandlerStub {
    fn venue(&self) -> Venue {
        venue_named("the stub")
    }

    fn library(&mut self, _source: &ShaderSource<'_>) -> Answer<Handle> {
        Ok(Handle(0))
    }

    fn compute_pipeline(&mut self, _library: Handle, _entry: &str) -> Answer<Handle> {
        Err(Failure::unavailable("compute_pipeline"))
    }

    fn buffer(&mut self, _bytes: &[u8]) -> Answer<Handle> {
        Err(Failure::unavailable("buffer"))
    }

    fn zeroed_buffer(&mut self, _length: usize) -> Answer<Handle> {
        Err(Failure::unavailable("zeroed_buffer"))
    }

    fn texture_2d(
        &mut self,
        _format: metal2vulkan::metal::PixelFormat,
        _size: [u32; 2],
        _row_pitch: u32,
        _bytes: &[u8],
    ) -> Answer<Handle> {
        Err(Failure::unavailable("texture_2d"))
    }

    fn read_texture(
        &mut self,
        _texture: Handle,
        _size: [u32; 2],
        _row_pitch: u32,
    ) -> Answer<Vec<u8>> {
        Err(Failure::unavailable("read_texture"))
    }

    fn dispatch(
        &mut self,
        _pipeline: Handle,
        _bindings: &crate::case::Bindings<'_>,
        _groups: [u32; 3],
        _threads: [u32; 3],
    ) -> Answer<Handle> {
        Err(Failure::unavailable("dispatch"))
    }

    fn compute_pipeline_with_stage_input(
        &mut self,
        _library: Handle,
        _entry: &str,
        _stage_input: &crate::case::StageInput,
    ) -> Answer<Handle> {
        Err(Failure::unavailable("compute_pipeline_with_stage_input"))
    }

    fn dispatch_threads(
        &mut self,
        _pipeline: Handle,
        _bindings: &crate::case::Bindings<'_>,
        _stage_in: &[crate::case::StageInRegion],
        _grid: [u32; 3],
        _threads: [u32; 3],
    ) -> Answer<Handle> {
        Err(Failure::unavailable("dispatch_threads"))
    }

    fn wait(&mut self, _command_buffer: Handle) -> Answer<Value> {
        Err(Failure::unavailable("wait"))
    }

    fn read_buffer(&mut self, _buffer: Handle, _offset: u64, _length: usize) -> Answer<Vec<u8>> {
        Err(Failure::unavailable("read_buffer"))
    }

    fn announced(&mut self) -> Vec<String> {
        Vec::new()
    }

    fn pipeline_both_paths(
        &mut self,
        _library: Handle,
        _form: &crate::pipeline_handlers::Form<'_>,
    ) -> Answer<crate::pipeline_handlers::BothPaths> {
        use crate::pipeline_handlers::{BothPaths, Delivered, Handled, StateFacts};
        let delivered = Delivered::made(StateFacts::Compute(stub_compute_facts()));
        let handler = match self.arrives {
            true => Handled::answered(delivered.clone()),
            false => Handled::never_arrived(),
        };
        Ok(BothPaths::new(handler, delivered))
    }
}

/// The transcript `case` leaves when it runs on `side`, as a replay builds one.
fn transcript_of(case: &dyn Case, side: &mut dyn Side) -> Transcript {
    let mut sites = Sites::new();
    let _ = case.run(side, &mut sites);
    sites.notices::<&str>(NOTICES_SITE, &[]);
    sites.finish()
}

/// **A creation handler that never ran is a divergence, not a pass.**
///
/// This is the falsification the asynchronous family needs. A case that waits on a handler with a
/// bound can "pass" by timing out every time — the state it compares is simply never read — so the
/// arrival site has to be one the comparison reads, and a timeout has to be visible as a timeout.
///
/// The test runs the **real case body** twice: once against a side whose handler arrives, which
/// stands in for the recorded Metal answer, and once against a side whose handler is suppressed.
/// The comparison must report a divergence, and it must name the arrival site: a diff that said
/// only "the facts are missing" would leave a reader unable to tell a timeout from a creation that
/// failed.
#[test]
fn a_creation_handler_that_never_ran_is_divergent_not_a_pass() {
    let case = crate::cases::named("pipeline-handler-compute-function")
        .expect("the compute-function handler case");
    let metal = transcript_of(case.as_ref(), &mut HandlerStub { arrives: true });
    // The control: the same body, the same side, the handler arriving. Without this the test
    // could pass because the case never compares anything at all.
    let arrived = transcript_of(case.as_ref(), &mut HandlerStub { arrives: true });
    let same = compare(
        &metal,
        &arrived,
        &case.allowances(),
        Offered::new(&[], &Validated::Ran),
    );
    assert!(same.is_exact(), "the arriving run is not exact: {same:?}");

    let timed_out = transcript_of(case.as_ref(), &mut HandlerStub { arrives: false });
    let found = compare(
        &metal,
        &timed_out,
        &case.allowances(),
        Offered::new(&[], &Validated::Ran),
    );
    assert!(
        !found.divergences.is_empty(),
        "a suppressed creation handler compared exact: {found:?}"
    );
    let paths: Vec<&str> = found
        .divergences
        .iter()
        .map(|divergence| divergence.path.as_str())
        .collect();
    assert!(
        paths.iter().any(|path| path.ends_with("/arrival")),
        "no divergence names the arrival site: {paths:?}"
    );
    assert!(
        paths
            .iter()
            .any(|path| path.ends_with("/the-two-paths-agree")),
        "no divergence names the two paths disagreeing: {paths:?}"
    );
}

/// Widening [`crate::case::SamplerSpec`] to all fourteen of `MTLSamplerDescriptor`'s fields did
/// not move one committed recording's input fingerprint.
///
/// This is the test that had to exist before the eight fields were added. A case's recording is
/// the answer to the inputs it was taken from, and the sampler state is one of those inputs, so a
/// `words()` that wrote a field unconditionally would have decayed **every** recording in the
/// suite at once — 111 cases make a sampler — and the suite would have needed re-recording whole.
/// The words a default state and an argument-buffer state write are pinned here as bytes, so the
/// only way to decay them is to edit this test and say so.
#[test]
fn the_default_sampler_words_are_the_words_they_always_were() {
    use crate::case::SamplerSpec;
    // Nearest, nearest, clamp to edge, clamp to edge, transparent black, normalized: the six
    // words `words()` wrote before `mipFilter` and the seven fields beside it existed.
    let six: Vec<u8> = [0u32, 0, 0, 0, 0, 1]
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    assert_eq!(SamplerSpec::new().words(), six);
    let mut seven = six.clone();
    seven.extend(1u32.to_le_bytes());
    assert_eq!(SamplerSpec::new().for_argument_buffer().words(), seven);
}

/// Each of the eight fields added to [`crate::case::SamplerSpec`] fingerprints apart from every
/// other and from `supportArgumentBuffers`.
///
/// `supportArgumentBuffers` writes a bare `1`, so an added field writing its raw ordinal would
/// have made a `mipFilter` of `Nearest` fingerprint as an argument-buffer state: two different
/// sampler states, one fingerprint, and a recording of one standing as the answer for the other.
/// The tag word is what keeps them apart, and this is what holds it.
#[test]
fn the_tagged_sampler_words_tell_every_field_apart() {
    use crate::case::SamplerSpec;
    use metal2vulkan::metal::{
        CompareFunction, SamplerAddressMode, SamplerMipFilter, SamplerReductionMode,
    };
    use std::collections::BTreeSet;

    let mut specs = vec![SamplerSpec::new(), SamplerSpec::new().for_argument_buffer()];
    let mut one = SamplerSpec::new();
    one.mip_filter = SamplerMipFilter::Nearest;
    specs.push(one);
    let mut one = SamplerSpec::new();
    one.r_address_mode = SamplerAddressMode::Repeat;
    specs.push(one);
    let mut one = SamplerSpec::new();
    one.lod_min_clamp = 1.0;
    specs.push(one);
    let mut one = SamplerSpec::new();
    one.lod_max_clamp = 1.0;
    specs.push(one);
    let mut one = SamplerSpec::new();
    one.lod_average = true;
    specs.push(one);
    let mut one = SamplerSpec::new();
    one.max_anisotropy = 2;
    specs.push(one);
    let mut one = SamplerSpec::new();
    one.compare_function = CompareFunction::Less;
    specs.push(one);
    let mut one = SamplerSpec::new();
    one.reduction_mode = SamplerReductionMode::Minimum;
    specs.push(one);

    let words: BTreeSet<Vec<u8>> = specs.iter().map(SamplerSpec::words).collect();
    assert_eq!(
        words.len(),
        specs.len(),
        "two of the ten states fingerprint alike"
    );
    // A clamp fingerprints as its bits, so a signed zero and a NaN are each their own input
    // rather than values a float comparison would fold into the default or into each other.
    let mut zero = SamplerSpec::new();
    zero.lod_min_clamp = -0.0;
    let mut nan = SamplerSpec::new();
    nan.lod_min_clamp = f32::NAN;
    assert_ne!(zero.words(), SamplerSpec::new().words());
    assert_ne!(nan.words(), SamplerSpec::new().words());
    assert_ne!(zero.words(), nan.words());
}

// ---------------------------------------------------------------------------------------------
// A case's verdict is a function of that case alone.
// ---------------------------------------------------------------------------------------------

/// The suite's integrity property, composed from the two halves that decide it: **a case's verdict
/// is the case's.** One device serves a whole selection, so the two halves are the notice
/// attribution ([`crate::device::Notices`]) and what the comparison is allowed to read
/// ([`replay_against`]).
///
/// **Measured, at `cbbdca70a3`.** The comparison took the codes the device had announced *earlier
/// in the run* as justifications a case's own tolerances could cite. The case below — a value
/// tolerance justified by a notice, a run that announced nothing itself, and a word that genuinely
/// differs — read `Divergent` alone and `Announced` inside a selection where **any** earlier case
/// had provoked that code. A real difference was excused by a neighbour's notice, which is the one
/// failure mode that lets a broken implementation pass a whole-set run while every added case
/// still looks green.
///
/// What the test drives is the second case of a two-case selection, with the selection's own
/// notice sink: the first case provokes the code, the second announces nothing of its own, and its
/// verdict must be the one it reads alone.
#[test]
fn a_notice_one_case_provoked_does_not_excuse_the_nexts_difference() {
    let kind = Degradation::DeviceArithmetic;
    let case = WordsCase {
        allowances: vec![Allowance::new(
            "case/output",
            Loosening::Unchecked,
            Justification::notice(kind),
        )],
        ..WordsCase::new("words", 2)
    };
    let recording = recording_of(&case, case.transcript(words(&[1, 2]), "Completed", &[]), 3);

    /// What the case reads with `announced` as its own run's notices: one word differs, and
    /// whether that difference is accounted for turns on whether this run announced the code.
    fn verdict_with(case: &WordsCase, recording: &Recording, announced: Vec<String>) -> Verdict {
        let mut side = StubSide::new(words(&[1, 9])).announcing_codes(announced);
        replay_against(case, &mut side, recording).unwrap().verdict
    }

    // Alone. Nothing this run announced justifies the word, so nobody is accounting for it.
    let alone = verdict_with(&case, &recording, Vec::new());
    assert!(
        matches!(alone, Verdict::Divergent { .. }),
        "a word differs with nothing announced to justify it: {alone}"
    );

    // The same case as the **second** of a selection, over the selection's own sink, after a
    // neighbour provoked exactly the code its tolerance cites.
    let mut notices = crate::device::Notices::default();
    let announce = notices.announcer_for_tests();
    let first = notices.open("the-neighbour");
    announce(kind.code());
    assert_eq!(
        first.take(),
        vec![kind.code()],
        "the neighbour provoked the notice, so the test is not vacuous"
    );
    drop(first);
    let second = notices.open("words");
    let in_set = verdict_with(&case, &recording, second.take());
    drop(second);

    assert_eq!(
        alone, in_set,
        "the same case over the same run read two verdicts, one of them inside a selection"
    );

    // The other leak, which is the same property from the other side: a neighbour whose notice
    // arrives **after** it was compared — off a runtime thread, or as it lets its last resource go
    // — leaves nothing for this case to pick up either.
    let mut notices = crate::device::Notices::default();
    let announce = notices.announcer_for_tests();
    let neighbour = notices.open("the-neighbour");
    drop(neighbour);
    announce(kind.code());
    let second = notices.open("words");
    let after_a_late_notice = verdict_with(&case, &recording, second.take());
    drop(second);
    assert_eq!(
        alone, after_a_late_notice,
        "a notice that arrived between the two cases was charged to the second"
    );
    // Never lost: the run reports it against no case.
    assert_eq!(notices.unattributed().len(), 1);
}

/// Every case carries a cost recording, or is named here with the reason it does not.
///
/// The suite records cost for **every** case by structure and not for a selection, so a recording
/// with no cost record is a case the cost half does not cover. The absence is silent: the gate's
/// reach shrinks with each case authored and nothing turns red. This test is the count, and it
/// fails when a case that could carry a record does not.
///
/// **Both causes are a run's reach, and neither is a case's shape.** `cost --write` measures the
/// selection it is given against the tree it runs in, and nothing in authoring a case produces a
/// record, so a case is covered only where some later whole-set run happened to include it.
///
/// A case the venue honestly cannot measure is a third cause, and **no case is in it**: none of
/// the 398 is a recorded refusal. That used to be unknowable rather than known, because
/// [`crate::run::record_costs`] turned a stopped run into an `Err` whose reason it printed and
/// dropped, leaving a recording byte-identical to one nobody had measured. A refusal is now
/// written down instead ([`crate::cost::CostRecord::unmeasurable`], read as `Unmeasured` before
/// any count by [`a_recorded_refusal_is_unmeasured_and_gates_no_count`]) and tallied by the
/// writing command, so from the next run on the two are distinguishable and countable. Every name
/// below is a case no run has covered, which is why a whole-set measurement empties both lists.
///
/// Held in **both** directions: a case carrying no record and not named here fails, and a name
/// here whose case now carries one fails, so the bill shrinks as runs fill it rather than going
/// stale. [`the_cost_coverage_set_is_held_in_both_directions`] proves both halves refuse.
#[test]
fn every_case_carries_a_cost_recording() {
    /// Cases authored after the last measurement that wrote cost records.
    ///
    /// **Not one recording added after that measurement carries a record**, which is what makes this
    /// one cause and not a list of names: these cases postdate the only run that could have covered
    /// them. A whole-set `cost --write` fills every one.
    const AFTER_THE_LAST_RUN: [&str; 324] = [
        "a2c-off-1-alpha-50",
        "a2c-off-4-alpha-50",
        "a2c-off-4-alpha-50-shadermask-5",
        "a2c-off-4-no-alpha-output",
        "a2c-on-1-alpha-50",
        "a2c-on-4-alpha-00",
        "a2c-on-4-alpha-100",
        "a2c-on-4-alpha-25",
        "a2c-on-4-alpha-50",
        "a2c-on-4-alpha-50-mask-no-alpha",
        "a2c-on-4-alpha-50-shadermask-5",
        "a2c-on-4-alpha-50-shadermask-a",
        "a2c-on-4-alpha-75",
        "a2c-on-4-no-alpha-output",
        "a2one-off-1-blend-source-alpha",
        "a2one-on-1-blend-source-alpha",
        "a2one-on-1-mask-no-alpha",
        "argument-encoder-cross-nested",
        "argument-encoder-cross-offsets",
        "argument-encoder-cross-type-mixtures",
        "argument-encoder-cross-type-sizes",
        "astc-10x10-hdr",
        "astc-10x5-hdr",
        "astc-10x6-hdr",
        "astc-10x8-hdr",
        "astc-12x10-hdr",
        "astc-12x12-hdr",
        "astc-4x4-hdr",
        "astc-5x4-hdr",
        "astc-5x5-hdr",
        "astc-6x5-hdr",
        "astc-6x6-hdr",
        "astc-8x5-hdr",
        "astc-8x6-hdr",
        "astc-8x8-hdr",
        "attribute-format-class-crossings",
        "attribute-format-fetches",
        "attribute-stride-buffer",
        "attribute-stride-buffer-offset",
        "attribute-stride-inline-bytes",
        "attribute-stride-plural-buffers",
        "binary-archive-adders",
        "binary-archive-fail-on-miss",
        "binary-archive-mesh-adder",
        "binary-archive-round-trip",
        "binary-archive-serializing",
        "binary-archive-sources",
        "blend-interaction-format-float32",
        "blend-interaction-format-srgb8",
        "blend-interaction-format-unorm8",
        "blend-interaction-mask-blend",
        "blend-sweep-blend-color",
        "blend-sweep-blend-color-order",
        "blend-sweep-destination-alpha",
        "blend-sweep-destination-rgb",
        "blend-sweep-enabled",
        "blend-sweep-operation-pairs",
        "blend-sweep-operations",
        "blend-sweep-second-source",
        "blend-sweep-source-alpha",
        "blend-sweep-source-rgb",
        "blend-sweep-write-masks",
        "blit-copy-indirect-command-buffer",
        "blit-optimize-indirect-command-buffer",
        "blit-reset-commands-in-buffer",
        "command-buffer-facts",
        "compute-descriptor-defaults",
        "compute-descriptor-reset",
        "compute-descriptor-set",
        "compute-heap-residency",
        "depthstencil-depth-codes",
        "depthstencil-depth-compares",
        "depthstencil-depth-slot",
        "depthstencil-faces",
        "depthstencil-facing-faces",
        "depthstencil-mask-crossings",
        "depthstencil-stencil-compares",
        "depthstencil-stencil-operations",
        "detach-in-a-completed-handler",
        "device-heap-buffer-sizes",
        "device-linear-texture-alignments",
        "device-rate-map-layer-counts",
        "device-sparse-region-conversions",
        "device-vertex-amplification-counts",
        "dynamic-library-from-a-library",
        "dynamic-library-from-a-url",
        "format-footprints-astc-4x2-and-8x4",
        "format-footprints-one-plane-ycbcr",
        "format-footprints-packed-without-textures",
        "format-footprints-padded-wide-channels",
        "format-footprints-plain-stragglers",
        "format-footprints-two-plane-ten-bit",
        "fragment-imageblock-reflection",
        "function-bindings-data-types-array-strides",
        "function-bindings-data-types-arrays",
        "function-bindings-data-types-member-floats",
        "function-bindings-data-types-member-integers",
        "function-bindings-data-types-member-narrow",
        "function-bindings-data-types-pointer-floats",
        "function-bindings-data-types-pointer-integers",
        "function-bindings-data-types-pointer-narrow",
        "function-bindings-data-types-tables",
        "function-bindings-data-types-texture-components",
        "function-bindings-imageblock",
        "function-bindings-visible-function-table",
        "function-facts-mesh",
        "function-facts-object",
        "function-facts-visible",
        "heap-available-sizes",
        "heap-labels",
        "heap-own-getters",
        "heap-placed-buffer-sizes",
        "heap-purgeable-states",
        "indirect-commands-bind-count-limit",
        "indirect-compute-commands-reset-and-ranges",
        "indirect-compute-commands-residency-usage",
        "indirect-render-commands-inherited-state",
        "indirect-render-commands-patch-draws",
        "indirect-render-commands-under-the-pass-state",
        "layered-attachment-levels",
        "layered-attachment-places",
        "layered-depth-other-slice",
        "layered-render-target-array-length",
        "layered-resolve-places",
        "library-facts-descriptor",
        "library-facts-descriptor-compile-to-binary",
        "library-facts-descriptor-constants",
        "library-facts-descriptor-every-option",
        "library-facts-descriptor-fail-on-binary-archive-miss",
        "library-facts-descriptor-options",
        "library-facts-descriptor-pipeline-independent",
        "library-facts-many-functions",
        "library-facts-one-function",
        "library-from-data",
        "mesh-descriptor-defaults",
        "mesh-descriptor-reset",
        "mesh-descriptor-set",
        "mesh-pipeline-reflection-no-object-stage",
        "mesh-pipeline-reflection-object-stage",
        "pass-attachment-actions",
        "pass-attachment-filters",
        "pass-attachment-indices",
        "pass-bound-clear-reach",
        "pass-clear-colors",
        "pass-clear-depths",
        "pass-clear-stencils",
        "pass-descriptor-assigned",
        "pass-descriptor-cleared",
        "pass-descriptor-defaults",
        "pass-descriptor-set",
        "pass-numbers",
        "pass-sample-positions",
        "pass-sequence-aspect-actions",
        "pass-sequence-color-actions",
        "pass-two-attachments-loads",
        "pass-two-attachments-resolves",
        "pipeline-descriptor-defaults",
        "pipeline-descriptor-reset",
        "pipeline-descriptor-set",
        "pipeline-functions-cleared",
        "pipeline-functions-emptied",
        "pipeline-functions-every-stage",
        "pipeline-functions-fragment-only",
        "pipeline-functions-replaced",
        "pipeline-functions-tile-only",
        "pipeline-functions-untouched",
        "pipeline-functions-vertex-only",
        "pipeline-handler-compute-descriptor-options-binding-info",
        "pipeline-handler-compute-descriptor-options-none",
        "pipeline-handler-compute-descriptor-options-refused",
        "pipeline-handler-compute-function",
        "pipeline-handler-compute-function-options-all-three",
        "pipeline-handler-compute-function-options-binding-and-buffer-type",
        "pipeline-handler-compute-function-options-binding-and-fail-on-miss",
        "pipeline-handler-compute-function-options-binding-info",
        "pipeline-handler-compute-function-options-buffer-type-and-fail-on-miss",
        "pipeline-handler-compute-function-options-buffer-type-info",
        "pipeline-handler-compute-function-options-fail-on-archive-miss",
        "pipeline-handler-compute-function-options-none",
        "pipeline-handler-compute-function-options-refused",
        "pipeline-handler-compute-function-refused",
        "pipeline-handler-mesh-options-binding-info",
        "pipeline-handler-mesh-options-none",
        "pipeline-handler-render",
        "pipeline-handler-render-options-binding-info",
        "pipeline-handler-render-options-none",
        "pipeline-handler-render-options-refused",
        "pipeline-handler-render-refused",
        "pipeline-handler-tile-options-binding-info",
        "pipeline-handler-tile-options-none",
        "pipeline-handler-tile-options-refused",
        "pipeline-options-argument-buffer",
        "pipeline-options-compute-archive-miss",
        "pipeline-options-compute-binding-and-buffer-type",
        "pipeline-options-compute-binding-info",
        "pipeline-options-compute-binding-info-and-archive-miss",
        "pipeline-options-compute-binding-info-and-undefined-bit",
        "pipeline-options-compute-buffer-type-and-archive-miss",
        "pipeline-options-compute-buffer-type-info",
        "pipeline-options-compute-every-defined-bit",
        "pipeline-options-compute-none",
        "pipeline-options-compute-undefined-bit-3",
        "pipeline-options-compute-undefined-bit-31",
        "pipeline-options-compute-undefined-bit-40",
        "pipeline-options-render-archive-miss",
        "pipeline-options-render-binding-and-buffer-type",
        "pipeline-options-render-binding-info",
        "pipeline-options-render-binding-info-and-archive-miss",
        "pipeline-options-render-buffer-type-and-archive-miss",
        "pipeline-options-render-buffer-type-info",
        "pipeline-options-render-every-defined-bit",
        "pipeline-options-render-none",
        "pipeline-options-render-undefined-bit-40",
        "pipeline-options-texture-slots",
        "pipeline-options-threadgroup-memory",
        "primitive-type-restart-indices",
        "rate-map-descriptor-forms",
        "rate-map-layer-sweep",
        "rate-map-parameters-identity",
        "rate-map-parameters-partial-tile",
        "rate-map-parameters-two-layers",
        "rate-map-quality-sweep",
        "rate-map-sample-sweep",
        "rate-map-screen-sweep",
        "render-draw-forms-empty",
        "render-draw-forms-lines",
        "render-draw-forms-points",
        "render-draw-forms-triangles",
        "resolve-depth-filters-2",
        "resolve-depth-filters-4",
        "resolve-stencil-depth-resolved-sample",
        "resolve-stencil-sample0",
        "resolve-stencil-under-depth-filters",
        "resource-aliasing",
        "resource-labels",
        "resource-own-getters",
        "resource-purgeable-states",
        "sampler-axis-border-opaque-black",
        "sampler-axis-border-opaque-white",
        "sampler-axis-border-transparent-black",
        "sampler-axis-mode-triples",
        "sampler-border-colours",
        "sampler-compare-always",
        "sampler-compare-equal",
        "sampler-compare-filter-pairs",
        "sampler-compare-greater",
        "sampler-compare-greater-equal",
        "sampler-compare-less",
        "sampler-compare-less-equal",
        "sampler-compare-never",
        "sampler-compare-not-equal",
        "sampler-cubic-filter",
        "sampler-mip-filter-pairs-linear",
        "sampler-mip-filter-pairs-nearest",
        "sampler-mip-filter-pairs-not-mipmapped",
        "sampler-mip-linear",
        "sampler-mip-nearest",
        "sampler-mip-not-mipmapped",
        "sampler-r-address-modes",
        "sampler-reduction-modes",
        "shared-event-handle-round-trip",
        "sparse-mapping-a-level-past-the-textures",
        "sparse-mapping-fences",
        "sparse-mapping-indirect-updates",
        "sparse-mapping-levels-and-the-tail",
        "sparse-mapping-mapped-tiles-hold-their-texels",
        "sparse-mapping-modes-and-region-sizes",
        "sparse-mapping-moves",
        "sparse-mapping-plural-updates",
        "sparse-mapping-slices-of-an-array",
        "tessellation-edge-factors",
        "tessellation-factor-scales",
        "tessellation-factor-steps",
        "tessellation-non-finite-factors",
        "tessellation-partition-degenerates",
        "tessellation-partition-factors",
        "tessellation-partition-insides-quad",
        "tessellation-partition-insides-triangle",
        "tessellation-partition-modes-quad",
        "tessellation-partition-modes-triangle",
        "tessellation-partition-scales",
        "tessellation-partition-steps",
        "tessellation-patch-indices",
        "texture-pairs-heap-texture-getters",
        "texture-pairs-plain-contents",
        "texture-pairs-storage-crossings",
        "texture-pairs-type-levels-slices",
        "texture-pairs-usage-crossings",
        "texture-shared-handles",
        "texture-view-descriptors",
        "texture-view-parents",
        "texture-view-selectors",
        "tile-descriptor-defaults",
        "tile-descriptor-reset",
        "tile-descriptor-set",
        "tile-pipeline-reflection-imageblock",
        "topology-class-plain-function",
        "topology-class-point-size-function",
        "vertex-format-constant-step",
        "vertex-format-float4",
        "vertex-format-half4",
        "vertex-format-instance-step",
        "vertex-format-int4",
        "vertex-format-uint4",
        "visible-tables-compute-a-private-function-has-no-handle",
        "visible-tables-compute-an-unlinked-function-has-no-handle",
        "visible-tables-compute-count-3-two-entries-written",
        "visible-tables-compute-every-entry-nil",
        "visible-tables-compute-one-entry-nil",
        "visible-tables-compute-size-0",
        "visible-tables-compute-size-1",
        "visible-tables-compute-size-2",
        "visible-tables-compute-size-2-swapped",
        "visible-tables-compute-size-3",
        "visible-tables-compute-table-unbound",
        "visible-tables-compute-whole-range",
        "visible-tables-stage-fragment",
        "visible-tables-stage-mesh",
        "visible-tables-stage-object",
        "visible-tables-stage-tile",
        "visible-tables-stage-vertex",
        "visible-tables-stage-vertex-and-fragment",
        "write-mask-unspecialized",
        "write-mask-unspecialized-and-red",
    ];

    /// Cases already committed when that measurement ran, still carrying no record.
    ///
    /// These are plain cases a venue runs -- `compute-inline-bytes`, `blit-whole-texture-copy`,
    /// `function-facts-vertex` -- so the cause is the measurement's tree and not the case. Cases land
    /// from concurrent branches and a run sees only the tree it runs in, so commit order is not
    /// coverage order. A whole-set run fills these too.
    const OLDER_THAN_THE_LAST_RUN: [&str; 74] = [
        "argb8-targets-destination-alpha",
        "argument-encoder-array-element",
        "argument-encoder-mixed-descriptors",
        "argument-encoder-ranges",
        "argument-encoder-state-descriptors",
        "argument-encoder-texture-sampler",
        "blit-fenced-encoders",
        "blit-generated-mipmaps",
        "blit-optimized-contents",
        "blit-synchronized-managed-buffer",
        "blit-synchronized-managed-texture",
        "blit-texture-counts-of-none",
        "blit-texture-slice-and-level-counts",
        "blit-unoptioned-buffer-and-texture-copies",
        "blit-whole-texture-copy",
        "buffer-depth-samples-d16-54-2",
        "buffer-depth-samples-d16-54-4",
        "buffer-target-clears-pitch-16-dont-care",
        "buffer-target-clears-pitch-16-stored",
        "buffer-target-clears-pitch-256-dont-care",
        "buffer-target-clears-pitch-256-stored",
        "buffer-target-clears-pitch-32-dont-care",
        "buffer-target-clears-pitch-32-stored",
        "buffer-texture-blits-overlapping",
        "buffer-texture-blits-to-and-from",
        "compute-indirect-threadgroup-grid",
        "compute-inline-bytes",
        "compute-plural-buffer-binding",
        "compute-plural-texture-and-sampler-binding",
        "compute-resource-barrier",
        "compute-resource-residency",
        "compute-scope-barrier",
        "compute-threadgroup-memory-and-grid",
        "fragment-writes-then-kernel-reads",
        "frame-length-chain",
        "function-bindings-argument-buffer",
        "function-bindings-texture-slots",
        "function-bindings-threadgroup-memory",
        "function-facts-constants",
        "function-facts-fragment",
        "function-facts-gated-attribute",
        "function-facts-kernel",
        "function-facts-optional-constant",
        "function-facts-patch-attributes",
        "function-facts-specialized-by-index",
        "function-facts-specialized-by-name",
        "function-facts-specialized-by-range",
        "function-facts-stage-in-kernel",
        "function-facts-vertex",
        "function-facts-vertex-attributes",
        "host-page-writes-kernel-store-read-only",
        "host-page-writes-kernel-store-writable",
        "host-page-writes-visibility-read-only",
        "host-page-writes-visibility-writable",
        "indirect-command-buffer-descriptor-properties",
        "indirect-compute-commands-executed",
        "indirect-compute-commands-reset",
        "indirect-compute-commands-stage-in-stride",
        "indirect-render-commands-drawn",
        "indirect-render-commands-executed",
        "kernel-writes-then-fragment-samples",
        "mesh-object-draws-partial-threadgroup-5x3-by-4x2",
        "mesh-object-draws-partial-threadgroup-6-by-4",
        "multi-draw-state-changes",
        "pass-writes-then-kernel-reads",
        "pipeline-facts-compute-described",
        "pipeline-facts-compute-imageblock",
        "pipeline-facts-compute-plain",
        "pipeline-facts-render-multisampled",
        "pipeline-facts-render-plain",
        "pipeline-facts-render-two-attachments",
        "render-then-sample-exact",
        "render-then-sample-magnified-linear",
        "render-then-sample-magnified-nearest",
    ];

    let mut missing: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    let mut checked = 0usize;
    for case in crate::cases::all() {
        checked += 1;
        match crate::recording::read(case.name()) {
            Ok(recording) if !recording.costs.is_empty() => {}
            // A case with no recording at all has no cost record either, and for this count the
            // two are the same hole: the cost half does not cover it.
            _ => {
                missing.insert(case.name());
            }
        }
    }
    assert!(
        checked > 0,
        "the suite holds no case, so this count would pass by covering nothing"
    );
    let named: std::collections::BTreeSet<&str> = AFTER_THE_LAST_RUN
        .iter()
        .chain(OLDER_THAN_THE_LAST_RUN.iter())
        .copied()
        .collect();
    assert_eq!(
        named.len(),
        AFTER_THE_LAST_RUN.len() + OLDER_THAN_THE_LAST_RUN.len(),
        "a case is named under both causes, so the two counts do not add up"
    );
    if let Some(said) = cost_coverage_diff(&missing, &named) {
        panic!(
            "{checked} case(s) checked, {} carrying no cost record against {} named:\n{said}",
            missing.len(),
            named.len()
        );
    }
}

/// Both directions of the cost-coverage set: what the recordings hold against what is named.
///
/// Kept apart from [`stale_set_diff`] because the two answer different questions and share only
/// their shape; folding them would make one message describe the other's failure.
fn cost_coverage_diff(
    missing: &std::collections::BTreeSet<&str>,
    named: &std::collections::BTreeSet<&str>,
) -> Option<String> {
    if missing == named {
        return None;
    }
    let mut said = Vec::new();
    let unexpected: Vec<&str> = missing.difference(named).copied().collect();
    if !unexpected.is_empty() {
        said.push(format!(
            "these cases carry no cost record and are not named as uncovered, so the cost half \
             stopped covering them with nothing turning red. Measure them (`conformance cost \
             --write`, naming them) or name them above with the reason: {unexpected:?}"
        ));
    }
    let filled: Vec<&str> = named.difference(missing).copied().collect();
    if !filled.is_empty() {
        said.push(format!(
            "these cases are named as uncovered but now carry a cost record, so each name stands \
             behind nothing and must go: {filled:?}"
        ));
    }
    Some(said.join("\n"))
}

/// Both directions of the cost-coverage set hold, over fabricated sets.
///
/// The check this proves is a set comparison, so it is provable with no device, no venue and no
/// recording. A check nobody has watched refuse is not a check: an earlier synthetic test of mine
/// passed and failed for the same reason and proved nothing, so each direction is driven here.
#[test]
fn the_cost_coverage_set_is_held_in_both_directions() {
    fn set(names: &[&'static str]) -> std::collections::BTreeSet<&'static str> {
        names.iter().copied().collect()
    }
    // Agreement passes, which is the only outcome that may pass.
    assert!(cost_coverage_diff(&set(&["a", "b"]), &set(&["a", "b"])).is_none());
    // A case that carries no record and is not named fails, and the message names it so the
    // reader knows which case lost coverage.
    let said = cost_coverage_diff(&set(&["a", "b"]), &set(&["a"]))
        .expect("an unnamed case with no cost record must fail");
    assert!(said.contains("\"b\""), "{said}");
    assert!(said.contains("not named as uncovered"), "{said}");
    // A name whose case carries a record again fails, so filling the hole shrinks the bill
    // instead of leaving a name standing behind nothing.
    let said = cost_coverage_diff(&set(&["a"]), &set(&["a", "b"]))
        .expect("a name whose case now carries a record must fail");
    assert!(said.contains("must go"), "{said}");
    assert!(said.contains("\"b\""), "{said}");
    // Empty on both sides is the state this work is for: every case covered, nothing named.
    assert!(cost_coverage_diff(&set(&[]), &set(&[])).is_none());
}

/// A recorded refusal reads as `Unmeasured` and gates no count.
///
/// The venue writes down that it cannot measure a case ([`crate::run`]'s refusal writer), and that
/// record holds no bounds. An absent bound otherwise means **zero**, which is the gate's whole
/// point, so a refusal record read naively would hold every count the case moves to nothing and
/// fail it. [`crate::cost::judge`] therefore reads the refusal before it reads a count.
///
/// The second half of this test is the one that matters: the same record with the refusal removed
/// reads `Over`, so the early return is load-bearing rather than decorative.
#[test]
fn a_recorded_refusal_is_unmeasured_and_gates_no_count() {
    let refused = crate::cost::CostRecord {
        venue: "a venue".to_owned(),
        runs: 0,
        measured: "no measurement: the run stopped before the case finished".to_owned(),
        contrasted: None,
        degraded: Vec::new(),
        bounds: Default::default(),
        moving: Default::default(),
        composed: Default::default(),
        ceilings: Default::default(),
        unmeasurable: Some("the venue lacks the feature the case needs".to_owned()),
    };
    let counts = crate::cost::Counts::read("WorkCounts {\n    queue_submissions: 7,\n}")
        .expect("the shape reads");
    let judge = |record: &crate::cost::CostRecord| {
        crate::cost::judge(
            std::slice::from_ref(record),
            "a venue",
            true,
            &counts,
            None,
            &[],
        )
    };
    assert_eq!(judge(&refused).word(), "unmeasured");
    // The reason survives into the verdict, so a reader of the gate learns why rather than that
    // the case is simply uncovered.
    let crate::cost::CostVerdict::Unmeasured { why } = judge(&refused) else {
        panic!("a recorded refusal is unmeasured")
    };
    assert!(why.contains("lacks the feature the case needs"), "{why}");
    // Without the refusal the identical record bounds all seven submissions at zero.
    let bare = crate::cost::CostRecord {
        unmeasurable: None,
        ..refused.clone()
    };
    assert_eq!(
        judge(&bare).word(),
        "over",
        "an empty record bounds every count at zero, which is what the refusal must not do"
    );
}
