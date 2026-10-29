# metal2vulkan-conformance

The Metal conformance suite. It makes one sentence executable:

> We are a Metal implementation. Any difference from Metal is a bug, unless we announced it.

A case is written once, against one vocabulary, and run by either side: native Metal on macOS, and
`metal2vulkan-runtime` on whatever Vulkan device the host has. Each run emits a **transcript**: a
tree of named sites holding what a Metal program can observe. Metal's answer is **recorded** on a
Mac, once per case, and **replayed** anywhere — the replay leg needs no Metal at all.

Comparison is exact by default. A loosened field names either a degrade notice's code or an entry of
`docs/API.md`'s accepted differences, and the comparison fails where that entry is gone or the
notice did not fire. A site may name several, because a notice is announced once per device per
kind: the difference is justified where any of them holds.

**Every degrade notice our side announces must be one the case names**, or the case is divergent.
Metal announces none, so a notice of ours is always a difference, and the only thing that can
account for it is the case naming its code. The codes a case names are a set and the notices its run
announced must be a subset of it: declaring one code never excuses another's. A code the case names
whose notice this run did not announce is the other direction, and is reported as an unfired
allowance rather than failed — a notice is announced once per device per kind, so which of a case's
notices its own run provokes depends on what ran before it in the same process.

**The one set of notices no case names is the device's own arithmetic.** The owner's rule
(AGENTS.md, rule 1) makes the GPU's native arithmetic and value answers the contract — `float` and
`half` maths, NaN bits, denormals, minimum and maximum, rounding, texel, vertex and colour
conversions, sampling and filtering, resolves. Where the device has a native form it is the only form the
runtime builds, and the note that says so fires in nearly every case that computes, converts or
samples. So the suite declares those notes itself, at the notices site, for every case:
`replay::NATIVE_NOTES` lists them, chosen from the notes' own documentation (the translator's
`PrecisionNote` and the runtime's `Degradation`), with the rule for what is in it and what is not.
They are held to a case declaration's rules — the notices site alone, justified only where the code
was announced — and they never count among the allowances a case named, so a native note a run did
not announce is not an allowance that never fired. A note of what the device **lacks**, of an
emulation's own precision, or of a Metal form not measured or not built is not in the set, and a
case that provokes one still names it.

**They declare nothing about a value.** Every value still compares field by field, and comparison
is never widened for the whole suite. A value the device's native answer moves is a divergence at
its own site. Nearly every note of the set says in its own documentation that validation derives no
tolerance from it, and the suite loosens no value by such a note either: that would be the suite
overruling the product's stated contract. So the site is a `known-gaps.tsv` row, scoped to the
venue whose device answers so, that says what differs, whose answer it is, and whether the note's
recorded bound covers it. A tolerance justified by a note is for a note whose documentation gives a
bound validation may hold, and is never wider than that bound.

A site where plain Metal answered a call and we returned a typed error is a **refusal**, and no
allowance can make it anything else: it counts in its own verdict and is never compliant.

That holds **wherever in a case** the refusal happened and **whatever shape Metal's answer** at the
site has. Our side's answer has one shape — a typed error, which a `Failure` can only record as an
`Err` — and Metal's has every other shape there is: a value, a tree of facts whose own value is
absent, a digest with no value beside it, a fact about what did not happen, or the answer of one
`Work` program whose many steps are a single call. Only two of Metal's answers are not a refusal of
ours: its **own** error, where the difference is between the errors, and the one fact that says
Metal never came back from the call, which is its own verdict. A refusal that arrives as a shape
mismatch instead is a bug of the suite's rather than a finding about Metal; it went unseen four
ways in this round, and each way lost both the refused count and the ledger row.

A replay takes the accurate form of each of the runtime's speed-for-accuracy switches
(`DevicePolicy::exact_answers()`: `heap_compression` `Uncompressed`, `completion` `Exact`,
`batch_order` `Exact`, `render_lowering` `AtCall`, `host_page_textures` `Shadow`); no switch touches
arithmetic, which is the device's native answer under either policy. `conformance replay
--default-policy` replays under the runtime's own defaults instead — the fast form of every
speed-for-accuracy choice, which is what a consumer runs — so the suite can say whether a
difference is the policy's or the implementation's.

A refusal the **venue** honestly cannot answer is not a gap of ours and reads `lacks-feature`
instead. That fact rides in the failure's own recorded value, beside the typed error, so it reaches
the comparison wherever in a case the refusal happened: a case whose site is one node but whose rows
are many names the row in its abort and pushes the node at the site, and filing the fact under a
path the case states twice is how one run came to score the same missing extension `refused` in one
family and `lacks-feature` in another.

## Does it answer the same thing twice?

```sh
conformance replay --runs 3                   # three passes, one process, one device
```

Every other reading the suite takes is of one run, so a reading that moves run to run is
indistinguishable from one that moved because the implementation changed — and a one-shot gate gives
the second answer, because it is the only one it has. `--runs N` replays the selection N times in
the one process and flags every verdict, every transcript site and every cost count that is not the
same in all N, naming the case, the reading and what each run read, in order. A time is deliberately
not a reading: it moves on every host under every load.

It is off by default, and it has its own line of the exit status.

## The known-gaps ledger

`known-gaps.tsv` holds the gaps we have written down as open. Each row is an **open bug with a todo
behind it** — not an accepted difference, not permission for anything. It exists so that one case
holding a known gap visible does not make every run red, because a report that is always red is a
report nobody reads, and that would cost us the next real gap.

It changes the **exit status** and nothing else. Every row is still printed as a `Refused` or
`Divergent` row, with our answer and Metal's, with Metal's own words where its API validation
refuses the call too, and still counted in the suite's own numbers.

A row names the verdict it expects:

- **`refused`** — a call plain Metal answered and we returned a typed error for;
- **`divergent`** — a field that differs with nothing we announced to justify it.

Both are bugs of ours by the owner's standing rule, both have to stay visible, and both need the
same discipline, so they share one file and one mechanism. Nothing else is ever written down: an
unrecorded case and a stale recording must fail loudly, and a venue that honestly lacks a feature
the case needs is not a gap of ours at all — that is its own verdict, per site.

A row may also name the **venue** it is expected on, because a gap can be one venue's and not
another's: a pipeline the runtime refuses on MoltenVK for a structural reason can translate on
lavapipe, and a global row would then fail the lavapipe run for existing. The field is held, without
case, against the venue's own description — its device name, driver and API version — so a row can
name one device (`Apple M3 Max`), one implementation (`llvmpipe`) or a family of them, and a venue
that joins later needs no new vocabulary. `*` means every venue, and it is the
**stronger** claim rather than the looser one: the row is then held on every venue the run selected
its case on, so widening a row's scope makes it harder to satisfy. That is what stops a
venue-scoped row from becoming a way to silence a venue. A venue pattern that matches nothing holds
the row nowhere, so its gap reads as unlisted and the run fails — a typo is loud.

It is an expectation, not an allowlist, and `replay` holds it in **both** directions, per venue:

- a refusal or a divergence the ledger does not list fails the run — a new gap appeared;
- a row the ledger lists that the run no longer holds **also** fails the run, saying the gap looks
  fixed and the row must go.

That second direction is the point: closing a gap forces the ledger edit, so a row cannot outlive
the bug it stands for and the file cannot rot into a silencer. It has already earned its keep twice
in this round — a product fix landed and the replay named four stale rows before anybody reasoned
about the fix. A row is held only against the cases a run selected and the venues it ran on, so
replaying one case does not read every other row as fixed, and a run on one venue does not read
another venue's rows as fixed.

A row's identity is its case, the verdict it expects, the site and the answer there, so a gap that
changed into another gap at the same site is a new gap and not the old one.

For a divergence the answer says **how much** moved — how many of the site's leaves differ, and the
largest difference where the values are numbers — and never where the walk first noticed. Two
reasons, and the second is the worse one. A bound read off the first mismatch is not a bound: of
twelve rotated-target divergences measured in this round every one differs by a single code at its
first mismatch and four reach **two** codes well past it, one of them at row 1, byte 905. And a walk
position is a property of the data rather than of the bug, so it would rename the row whenever an
earlier leaf moved, for no change in the gap. The report prints the first differing leaf on the line
below, where it helps a reader and binds nothing. The shape degrades cleanly across every kind of
difference, because a count of differing leaves always exists and only the largest difference is
numbers-only.

`replay` also emits, under one heading, a ready-to-paste row for each unlisted gap, so nobody
reconstructs a site or an answer by hand or with a parser — the step where one stream in this round
nearly committed a row for a site that does not exist.

Where the difference **is** a notice — our side announces a degrade code the case does not declare
and the suite's native set does not hold, Metal announces nothing, every other field bit-exact — the
row's answer names what each side announced, because that is the whole difference and a row that
did not name it would record a gap without saying what it waits on. A case that both diverges and refuses holds its refused sites in the `Divergent` verdict alongside
the differences, so neither row can go unchecked behind the other: a divergence never launders a
refusal.

## Replaying off the host that recorded

A replay needs no Metal, so it runs on any Vulkan device. One of its inputs is the venue's, which is
the point of replaying elsewhere; none of them is the recording host's any more, and the one that
was is worth saying how it was fixed.

**A case's input fingerprint covers the committed fixture's `.ll` and `.air.ll` text, never the
bitcode `llvm-as` assembles from the `.ll`.** That output is a function of the text *and the
assembler's release*: the same `fixtures/public/fragment_constant_color.ll` assembles to
`1a4cc972040870f8` under Homebrew's LLVM 23.1.4 and to `38d752a345c98db2` under 23.1.1. Nine cases
once fingerprinted it — `binary-archive-*` (six), `library-from-data` and the two
`dynamic-library-*` — and read `unrecorded: the inputs this run fed are not the recording's: air
differs` on every host but the Mac that recorded them. A fingerprint exists to say the run fed the
recording's inputs, and the committed text is the input a case states; the assembler has no part in
it, which is why no case needs `llvm-as` to state its fingerprint at all.
`tests::every_air_input_is_a_committed_fixture_texts_digest` holds that over the whole case set, and
`tests::every_cases_inputs_are_the_ones_its_recording_was_taken_from` checks every committed
recording against it with no tool and no device.

**The consequence was sharper than nine skips, and it is the reason this section exists: a run off
the recording host reported a ledger row whose case never compared as one that looks fixed.** Two
rows did exactly that (`binary-archive-mesh-adder` at `case/program`, `binary-archive-sources` at
`case/program/from-a-metallib/refused`). Neither was fixed.

So the rule stands, and is now mechanical as well: **a `known-gaps.tsv` row is never retired on the
strength of a run whose case did not compare** — a run made off the recording host retires nothing
unless the fingerprint covers the difference between the two hosts. An `Unrecorded` case holds no
verdict, so `replay` reads no row of it in either direction: it reports the case as one the ledger
cannot carry, and says nothing about the rows. The row's own verdict is read where the case
compares.

The venue is the other input, and it needs no warning: a notice a case does not declare is a real
difference on that driver, and a row scoped to a venue holds only there.

## A site the venue honestly lacks

A device's honest gap must never read as a refusal of ours. Where a typed error names capabilities
a device could enable, or a driver measured to run the thing wrong, the site is the **venue's** gap
and not ours: it is never a refusal, never a ledger row, and the rest of the case still compares,
so a divergence anywhere else in the same case still fails. The case's verdict is then
`lacks-feature`, which is neither a pass nor a failure and is never counted as coverage.

Two shapes reach that verdict: the case **stopping** at such a call, and a case carrying on past
one, which is recorded per site. The second matters because until it existed a case that did not
have to stop could not say the venue lacked anything, and the site read as a gap of ours.

## Two Metal oracles

Each recording holds **two** Metal facts, and never collapses them.

**Plain Metal is the behavioural oracle**, because that is how a consumer runs the runtime: its
transcript is what comparison uses, and matching it is what the suite asks for.

**The same case under Metal's own API validation layer is a second recorded fact**
(`Validated::{Ran, Stopped, NotRun}`). It is never compared with our behaviour.

**A validated stop is not Metal refusing the call.** Where plain Metal answers and we return a
typed error, that is a gap of ours whatever its validation layer thinks, so the row is `Refused`
and is not compliant. The validated fact may **document** such a refusal
(`Justification::MetalValidation`) — a refusal Metal's own validation also refuses reads
differently from one Metal is happy with, and the first is the better bug report — and it is held
to its evidence: the case names the words Metal's validation used, and the claim holds only where
the recording's message contains them. An unexercised leg documents nothing. So the claim buys an
explanation in the report, never a pass, and it cannot be made at all unless the validated leg
really ran.

`record` takes the validated leg in a **child process** of the command, under `MTL_DEBUG_LAYER=1`
with `MTL_DEBUG_LAYER_ERROR_MODE=nslog`. Two reasons, both measured rather than assumed:

- Metal reads `MTL_DEBUG_LAYER` as it initialises, so one process cannot run both legs.
- Metal's validation reports by assertion, and its **default** error mode ends the process.
  `nslog` is the one mode that reports every check and lets the call run, so the suite's own
  process never has the layer on and no assertion of Apple's can end a record run.

The child announces each call it makes on its standard error, so Metal's own output and those
markers interleave in order and each complaint is attributed to the call that provoked it. Nothing
is matched against Apple's wording, so a check in a release the suite has not seen is still
recorded. `NSLog`'s timestamp and process prefix are stripped, because a recording holds nothing
host-specific.

A Metal-less host reads the validated fact out of the recording like everything else, so the axis
is as portable as the rest of a case.

## The cost half: what our side did

The transcript says what a case **observed**. The cost half says what our implementation **did** to
produce it, over the same case set:

```sh
conformance cost --runs 8                     # measure the whole set in order, write nothing
conformance cost --runs 8 --write             # put those bounds in the cases' recordings
conformance cost --runs 8 --alone --write     # measure each case alone and fold the contrast in
conformance replay                            # hold each case to them, and report
```

Both shapes, as two commands, because a bound is written from both or it is not worth writing.

A case's cost record lives in its own recording file, beside the Metal answers, so a host with no
Metal reads it with the rest of the case.

**Counts are ours, never Metal's.** Metal has no counterpart for any of them and a host with no
Metal can have no answer, so a count never enters the comparison with Metal and never produces a
correctness verdict. The cost half has its own four verdicts — `within`, `over`, `under`,
`unmeasured` — its own section of the report, and its own line of the command's exit status.

**It gates on counts, not times.** A count is deterministic, so it is assertable on any host under
any load, and a regression reads "this case now does two allocations where it did none" rather than
as a percentage nobody can reproduce. The mechanism is the runtime's own `WorkCounts`, read twice
around each case; nothing here counts anything the runtime does not already count, and a cost run
asks it for no more than an ordinary run does.

Work is attributed to three regions, kept apart because moving work between them is not a win:
**translation** (making a module, a layout or a pipeline — the one-time cost, which has its own
ceilings), **the device API** (what a call does on the calling thread), and **submission** (what the
commit path, our own other threads and the GPU do afterwards, so a call that returns at once while
queueing work cannot score as a win). The device API is the residual.

A count the measurement never saw move is bounded at **zero**, so a case that starts doing something
it did not do before is `over` rather than unnoticed.

**A count of work the runtime avoided is gated from below too**, and it is the only count that is.
Every other bound is a ceiling, which cannot see a saving go away: a change that makes a render or
compute state's creation-time prediction miss *lowers* `creation_pipelines_drawn` and raises
`variant_pipelines_made` into the room the ceiling leaves, so both readings pass. Such a count is
named in `cost.rs`'s `SPARED` table with the measurement behind it and gated from **below alone**,
and a reading below it is `under` rather than `over`, because the two findings are opposite. It has
no ceiling: a ceiling on a count of work avoided reads a win as "cost more", and the whole set has
`integer-vertex-attributes` taking all 64 of its predictions where the record's reading was 0.
Nothing is lost by dropping it, because no command can take a prediction that was not made and
`creation_pipelines_made` keeps its own ceiling. A gated count's bound is every run's reading, so
the floor is that reading; a count the measurement saw move, or that reads differently with the
case alone, stays reported in both directions.

Some counts are reported and never gated, because holding them would give a gate that fires at
random on honest work. Each reason is written at its site in `cost.rs` with the runs behind it:

- one whose readings **moved** between the measurement's own rounds, which the record keeps as the
  range it was seen in;
- one that says **which thread arrived first** — when one of our background threads ran, and everything on the asynchronous release path, whose
  counts may land in a later case's window;
- one that is a reading of **what was alive at that instant** rather than work a case did;
- one that says an object was **found already made** rather than made, or that a making was
  **begun ahead** by a thread finishing an earlier case's work, which depends on what every case
  before it in the same process did;
- one that is the **complement of a count already in this list**, such as the command-buffer
  transitions that ran on our own waiter with no handler thread woken: one side of a coin whose
  other side is thread arrival cannot be deterministic while that side is not;
- one that moves when something has sat **past its quiet period**, which is a question about
  elapsed time;
- `completed_waits`, which charges a case for waits belonging to whatever ran before it.

A count in the `SPARED` table is gated from below only while it is gated at all: `racy` wins, so a
reported count has no floor either.

Totals stay gated where one exists, so nothing is lost: `encoder_command_buffers_begun` holds the
primaries a completion thread began ahead, and a `_made` count holds its `_shared` partner — finding
one already made only ever *lowers* it, and the gate is an upper bound.

### The two shapes, and why a bound needs both

A case's counts depend on what ran before it in the same process, because the runtime pools command
pools, memory blocks, descriptor sets, pipelines and their layouts across cases. So the same case
reads differently in the two shapes a measurement can have, and **neither bounds the other**:

- A **pooled making** is largest with the case *alone*: a case with nothing before it makes what a
  case in company finds already made. A bound taken in company goes stale the moment a case lands
  beside this one and makes the shared thing first.
- A count one of **our own background threads** moves is often largest *in company*: the work they
  do ahead and the waits they borrow come from whatever ran before. A bound taken alone would fail
  such a count for ever.

Measured over the whole case set in both shapes, thousands of readings differ, most of them larger
alone and a minority larger in the set. So the gate holds a count only where the two shapes
**agree**, at the larger of the two readings, and a count they disagree on is reported with both
numbers as the evidence.

**With one exception, and it is a list of names rather than a rule.** Where a count's larger reading
is the *no-sharing extreme* — a case with nothing before it makes the object itself where a case in
company finds it already made — that reading is a ceiling no composition can exceed, so the gate
keeps the count there rather than giving it up. `cost.rs`'s `MONOTONE` holds those names, each with
the measurement that earned it: a whole-set measurement in both shapes with **no case where the lone
run read lower**, and a gap small enough that the bound still bites. `device_memory_allocated` passes
the first test and fails the second — it counts bytes, so the two shapes differ by 268,435,456 where
the largest of the object counts differs by 8, and a bound of 256 MiB on a case that allocates
nothing in company is a number no regression can reach. A name earns a row on its own measurement,
never by resembling one already there, and the code holds each row to its direction: a later
measurement that reads the lone run *lower* reports that count like any other. Each shape is
measured by its own command, into the same record, because one process cannot measure both cleanly:
`bc1-rgba` run on its own reads `commands_recorded.cmd_dispatch` 4 in three invocations out of
three, and 0 when the same leg runs after a whole-set leg in the same process.

That is what stops a bound going stale on a landing, because a reading a landing can change is a
reading that depends on what ran before the case — which is exactly what the two shapes disagreeing
means. It is also why the record names both measurements, `measured` and `contrasted`: a record with
no `contrasted` has had no contrast taken, and claims nothing about composition.

The recordings' own history is the check on that claim, and it is the answer to the doubt a reader
arrives with: a count like `command_buffers_begun`, which the contrast finds composition-dependent
in 110 cases, looks as though the 328 cases where the two shapes agreed agreed only by accident of
one selection. Over every re-measure of a case whose body did not change, where the selection size
changed between the two measurements, and after the two shapes were split into their own commands,
**11,376 gated readings across 132 counts survived the set growing 541 to 579 and 579 to 686
without one of them moving.** A count moving in both directions is why it cannot take the
`MONOTONE` shortcut — the lone run is not its ceiling, so there is no single reading to gate at
without the contrast. It is not a reason to stop gating it.

A whole-set record is still evidence for a whole-set run and nothing else, so a run of fewer cases
reads `unmeasured`. To read one case's two shapes by hand:

```sh
conformance cost --runs 8 --show <case>           # a selection of one, on its own fresh device
conformance cost --runs 8 --alone --show <case>   # the same, printed against the record
```

That contrast is how `completed_waits` was caught charging three cases for four or five waits that
were not theirs, after it had held still over seven rounds of an earlier measurement — and how
`encoder_primaries_begun_ahead` was caught doing the opposite, reading 0 alone and 2 in the set.

**A device's gap is not a cost win.** A count that is lower because the device lacks the exact form,
so the runtime announced a notice and did something cheaper, is the device's gap and not work we
saved. A record carries the notices its measurement announced and refuses to compare a run that
announced otherwise. The device's native answers (`replay::NATIVE_NOTES`) count on neither side
(`replay::degrading`): the native form is the only one the runtime builds, so a native note marks
no cheaper work. Four things make a run `unmeasured` rather than a pass, each said in words: a
venue the measurement was not taken on, a selection that is not the whole case set (a case's counts
depend on what ran before it in the same process), a run that stopped part way, and a run the device
degraded otherwise.

**The one bound that is a ratio, and why it had to be one.** Our largest unfixed cost against Metal
is that we make more `VkPipeline`s per Metal pipeline state than we need: measured over all 1,103
cases, 1.519 each in the 639 that create a state, with 624 of the 1,439 made at a state's creation
taken by no command. Nothing held it. `variant_pipelines_made` is gated from above and `creation_pipelines_drawn` from below, and both
pass a change that makes one more pipeline per state while the case grows a state, because the
absolute counts rise for two reasons at once. So `cost::PipelinesPerState` is a bound on the
quotient, kept as its two counts and judged by cross-multiplication: a case that grows a state reads
the same ratio and needs no re-bless, and a tree that makes more pipelines for the same work fails.
The denominator is the suite's own (`states::making_a_state`, the one door every creation on our leg
goes through, held by a source guard), because the states a case asks for are the case's doing and
not the runtime's. It is the only bound taken in one shape alone: a case alone makes more pipelines
per state than the same case in company, and `judge` refuses any selection but the whole set anyway.

**Every record carries its contrast, and that is a gate.** A record with none gates its `MONOTONE`
counts -- `creation_pipelines_made`, `variant_pipelines_made`, `pipeline_layouts_made`,
`shader_modules_made` and the rest -- on whichever composition its one measurement met. Two cases
showed what that costs: `render-state-render-barriers` and `render-state-heap-residency` read `over`
on all four, a `--write` pass wrote the same numbers back, and neither a regression nor case growth
was involved. Their one in-set reading was taken in a run where the variant **took back a released
variant's objects**, which makes no pipeline, layout or module at all (`pipeline/kept.rs`), so
heap-residency was bounded at 0/0/0/0 while binding and drawing a pipeline it had taken back. The
count that decides it, `variants_taken_back`, is one this half refuses to gate. Taking the contrast
raised them to 1/1/1/2 and 2/2/2/4 as recorded ceilings -- the readings the breaches had been
showing -- and all 700 records that hold bounds now carry one.

**Two gates over the committed files, needing no device.** The suite-wide ceiling on predicted
pipelines no command took (1,439 made, 815 taken, 624 by none over 585 rows; a difference is not a
defect, since a creation cannot know whether a command will come, but a widening one is), and the
reach over the 111 of 158 cases whose made ceiling and drawn floor together already say they take
every one. The ceiling rose with the coverage and the rate improved as it did, from 32% of predicted
pipelines taken to 57%: a ceiling that rises with a bigger census is honest, and what it must not do
is rise while the rows stay the same.

**Every recording carries a cost record.** 399 of the 1,103 used to carry none, among them
`vertex-format-constant-step`, `vertex-format-float4` and every `blend-sweep-*` -- exactly the
many-pipeline sweeps with the worst per-case ratios against Metal, so the fix that halved their
pipeline count was held by nothing on the cases that most needed it. They now read 45 made / 45
taken, 45 / 45 and 32 / 32, and `integer-vertex-attributes` reads 64 / 64 / 64 where its blessed
ceiling had been 64 / 0 / 128.

**Every case that creates a Metal pipeline state carries the ratio.** 639 of the 1,103 recordings
hold a `per_state` bound, measured on this M3 Max over 8 runs of the whole set: 2,665 `VkPipeline`s
over 1,755 states, 1.519 each, the worst row `swizzled-storage-rgba8unorm` at 11.25. The other 464
hold none, and for a reason the record states: 453 cases create no pipeline state, so there is no
denominator, and 11 carry a recorded refusal instead. A bound measured over a selection would not
do: `judge` refuses any selection but the whole set, which is the shape the gate runs in, so the
bound is written from a whole-set `cost --runs 8 --write` and nothing smaller.

`cost`'s own per-case wall-clock figure stays **advisory**: it is printed so a reader can see where
the counts are large, and it is neither recorded nor gated. Timing against Metal is the times half,
below.

## The times half: Metal's cost beside ours

The counters say what our side **did**. The times half says what each leg **cost**, over the same
case set, with native Metal's own cost as the thing to beat:

```sh
conformance times --runs 8                  # both legs, print the tables
conformance times --runs 8 --show           # and every per-case row, not only the interesting ones
conformance times --leg metal --runs 8      # one leg in a process of its own
conformance times --runs 8 --write          # keep the figures in the cases' recordings
conformance times --runs 8 --call replace_region   # where one call's cost sits, case by case
conformance timing-coverage                 # which calls are timed, and which are not
```

`--call` is the triage, and it earns its place: the first whole-set run put `replace_region` at
105x over 108 calls, and the three cases that obviously make it came out at 1.2x over 3 when run
alone. The figure was real and almost none of it was there. `--call replace_region` put all of it
in one family — 16 `xr-two-plane-samples` cases, one such call each, Metal 72 us against our
19,132 us (**265x**, 92% of ours spent blocked) — while the other 86 cases that make the call sit
at 1.3x together. A per-call total says *that* a call is expensive and never *where*, and a reading
over one selection is not evidence about another.

Both legs are measured **at one boundary**, `watch::Watching`'s span: around the Metal or Vulkan
call and the side's own handle bookkeeping, outside the transcript, the fixtures and the input
fingerprint. One wrapper, not one span in `MetalSide` and another in `RuntimeSide`, because two
spans placed separately are two spans that drift and a comparison of two legs measured in different
places is not a comparison.

**A difference is the whole chain's, and attributes to no layer.** On this Mac our leg is
translation, then our runtime, then MoltenVK, then Metal. A ratio is the cost of all of that against
native Metal. The suite cannot say what MoltenVK's share is or what ours is, and no figure it prints
licenses a sentence of the form "MoltenVK costs X". Separating the shares needs counters inside each
layer or the same case on a non-Apple driver to difference against.

**Three attributions, kept apart, because moving work between them is not a win.** The calling
thread's own CPU inside the call; what the caller waited for; and CPU the rest of the process burned
over the case. A call that returns at once and hands its work to a thread the caller must then wait
on has saved nothing, so the margin is judged on `wall-in-calls + elsewhere` — everything the
machine spent on the case's calls, however attributed. No split of one total changes a verdict.

The split is the instrument's, not a preference. Measured on this M3 Max: the wall clock reads in
16 ns and answers nanoseconds, the thread CPU clock reads in 95 ns and steps in 125 ns, and the
**process** CPU clock reads in 230 ns and steps in **1000 ns**. A microsecond step is most of a
short call, so the handoff is a per-case figure and the other two are per call. The instrument's own
~0.24 us a call is measured on the host and printed beside the table; it is never subtracted, and it
is paid identically on both legs, so a ratio carries it top and bottom.

**Two tables, and the per-call one needs the other to be honest.** Metal compiles the AIR at
`library` and we translate it at `compute_pipeline`, so the same work sits at a different call on
each leg: measured here, `library` costs Metal about 23 ms and us about 0.1 ms, while
`compute_pipeline` costs Metal 0.09 ms and us 2.1 ms. Neither row alone is a result. The per-case
total is the figure that cannot be moved.

The per-call table is **paired over the cases both legs did exactly**, and that is not optional. Each
leg's own pool is over the cases *that leg* finished without a notice, and the two sets differ: 77 of
this suite's cases announce a notice on our leg and none do on Metal's. The first whole-set run,
before the pairing, compared `render_pipeline` over 132 Metal calls against 85 of ours and printed a
ratio of different work. Each row still prints both legs' call counts, because a case whose legs make
a call a different number of times is comparing different work and only the counts can say so. The
ratio is on the **totals**, so a call is weighted by how often it runs.

**The degraded-set rule, which matters more here than for counts.** A leg that announced a degrade
notice did a different amount of work: the device lacked the exact form and the runtime did something
cheaper and said so. Its cost is `unmeasured`, never a fast row. Without that rule a device's honest
gap reads as a cost win, which is the most flattering lie the suite could tell. A native note is no
such notice and counts for neither leg, as for counts.

**But the rule withholds the verdict and no longer deletes the figures, which it used to.** Every
function that pooled, ranked or printed a figure asked `judge` whether a case was `unmeasured` and
skipped it, so a case with any notice left the per-call table, the `--call` triage and every total,
figures and all. Decomposed offline over the stored recordings: the paired cases carry 2,811 ms of
excess, 18.6% of it pipeline creation; the cases with no notice carry 658 ms of it, 74.5% pipeline
creation; and the cases a notice erased carry **2,154 ms — 3.3x the whole comparable excess — of
which 1.5% is pipeline creation.** So the filter was removing the larger half of the finding and
skewing what was left towards the one cause that survived it. A notice also need not explain what it
hides: the largest cluster in the set (`extended-swizzles-*`, 1,884 ms) sits behind a notice all 15
siblings do while four of them run 3x *faster* than Metal, and `depth16-texel-reads` hides 173 ms of
purely **blocked** cost behind `fast-math-rewrites`, a notice about arithmetic, which cannot make a
caller wait. The withheld rows now print as their own per-call table, their own `--call` block and a
per-case ratio marked with a trailing `?`; nothing gated reads any of them.

**What is still erased, on purpose.** A leg that **stopped** part way, and a case Metal spent nothing
measurable on, have no comparison to withhold rather than a withheld one — 72 `texture-fills-*` and
`sample-fills-*` cases have no Metal leg at all, because Metal fills a texture through private API
alone, and ranked beside the rest they would read as the largest non-pipeline item in the set while
meaning nothing. `times::Comparison` is the three classes apart, `times::judge` reads it rather than
applying the rule a second time, and the two `times::paired*` tables are the two reportable classes.

**The per-case margin is reported, never gated.** `times::MARGIN` is an order of magnitude,
computed and printed as a verdict, and the **per-case** rows decide no exit status. Measured twice
over the whole set on this Mac, eight rounds each: the verdicts were identical both times (221
within, 0 over, 162 unmeasured). What the commit that gates them has to show first: a bound per case
rather than one margin for all, each naming its measurement the way a cost bound does; and the
contrast between a case run alone and the same case inside the set, which for counts turned five
apparent regressions into four counters that were never per-case facts.

**The per-call rows are gated, and the gate withholds more than it judges.** `known-slow-calls.tsv`
holds it in both directions. Three independent whole-set passes have been recorded on this M3 Max,
and `--write` keeps both legs' per-case figures in each case's recording, so a pass can be re-read
without re-running it. One of the three is degraded, and the instrument says so: it cost 0.275 us a
call against 0.236 us in the quietest, and it is not reproducible by the binary that produced it —
re-run over the same fifteen cases, that binary reads `function_facts` at 46.3x and 40.9x against
the 89.5x it recorded. The cause is the CPU the measurement ran on: one binary and one case set
inside one minute read 38.8x at the ordinary scheduling class and **75.5x** under a background one,
with the one-minute load average holding at 6.1 and every load rule judging both.

Nearly all the movement the ledger's band pays for is that one pass. The paired rows common to the
degraded pass and a good one move a median 2.62x, 14.96x at the 90th percentile and 63.9x at worst;
**the two undegraded passes move a median 1.12x, 1.51x at the 90th percentile and 2.00x at worst
over all 100 rows.** So the ratio is not inherently unstable, and for a while the gate did not
refuse the pass that makes it look so: 0.275 against 0.236 us is 1.17x, inside the 1.5x the
calibration check allows. **It refuses it now, on the floor under the measurement rather than on
the instrument's reading of itself.** The suite makes hundreds of calls that do almost nothing, so a
leg that is measuring at all resolves some of them under one instrument cost: both undegraded
passes read a cheapest figure of 41 ns on each leg — one tick of this chip's 24 MHz timebase, 0.17
instrument costs — while the degraded pass has none of Metal's 2,979 figures under even the resolve
line and a floor of 5.15 instrument costs. `constant_data_offset`, a getter that does nothing, cost
41 ns on Metal's leg in both good passes and 3.000 us in that one. Every other quantity the gate
reads ranks the degraded pass as the best of the three — it ran at the lowest load, its two legs'
instrument readings agree the most tightly, its eight rounds are the most repeatable, and it sat
closer to its recorded reference than the good pass beside it, because a pass overwrites that
reference and so inherits its predecessor's degradation. `known-slow-calls.tsv`'s header carries the
rest, including the four live per-call costs that went unwritten because that header had said they
changed sides when they had not.

**And a per-call row now has to carry its per-case figure**, because the two-tables warning above
is not advisory. Metal compiles the AIR at `library` and we translate it at pipeline creation and
at reflection, so five rows that read 20x to 50x per call turned out to sit on cases we finish
25 to 50 times faster than Metal: `function-facts-*` is 0.04x a case and
`library-facts-descriptor-*` 0.02x, against Metal's 21-27 ms at `library` where we spend under a
millisecond. The per-call gate still earns its place — `replace_region` is 265x with 92% of ours
blocked and its own cases are 0.23x, so nothing else could see it — but what separates it from
those five is that its cost is work Metal never does, not work Metal already did earlier in the
case. The ledger's header asks that question of every new row.

So the gate judges a per-call row only when the measurement can say something about it. It withholds
when the **instrument's own cost** disagrees with itself across a leg, across the two legs, with
what this chip's recordings carry, or with the cheapest figure that leg actually produced —
bracketed per leg, because a single reading at a leg's start is from a freshly-spawned thread and a
leg is minutes long. And it withholds a row whose cheaper leg's
typical call costs under four instrument costs: of the 41 such rows, the ratio moved a median 4.23x
between the two passes against 1.87x above the line. A withheld row neither adds a ledger row nor
retires one, which is what protects `replace_region`'s row — a pooled ratio that moved 7.4x to 31.3x
while its localised finding is 265x — from being deleted by a quiet pass.

**What a recording keeps, and why.** Metal's cost on a given macOS build is the one half that cannot
be re-taken: a measurement of ours is a command away on any host, but what Metal charged for a call
on 26A434 is gone the moment the machine updates. `--write` keeps both legs' figures in the case's own
recording for that reason. Nothing compares a stored figure with a live one — a comparison is only
ever between two legs measured in the same run, on one machine, from the identical transcript — so
the stored figures are history, and a case with none is a reported hole. `RECORDING_FORMAT_VERSION`
is untouched: the field is optional, the pattern the counters half established, so no committed
recording moves.

## Timing coverage is a number

A per-call table over an unknown fraction of the API is a table nobody can act on, so `untimed` is
tracked and driven down the way `unrecorded` is. `conformance timing-coverage` answers it in two
parts, and only the first is exact:

- **The suite's own vocabulary.** Every call a case can make is a `Side` method, and a run says
  which ones the selection made. Both the trait and the wrapper are read from their own sources, so
  a call a stream adds cannot make coverage look complete by being left off a list. Today: **51 of
  51 spanned calls are timed** by the whole case set; `venue` and `announced` are forwarded without
  a span on purpose and are listed as such rather than counted as holes.
- **The runtime's public frame surface**, which can only be answered approximately from here. A
  `Side` call reaches the runtime through our own code, which calls further helpers, so knowing
  which public method a timed call reaches needs a call graph this crate has no business building.
  The census counts the public methods of the objects a frame makes its calls on and says which are
  **named anywhere** in the suite's runtime-side sources — an **upper bound** on timed coverage, and
  the report says so in those words. Today: **148 of 540**, so at least 392 frame-surface methods
  cannot be timed by any case. Narrowing the bound to a real figure is a call graph's job.

The frame-surface list is the one `scripts/api-cost-coverage` uses for the other harness's table,
kept the same deliberately: two censuses over two denominators would be two numbers nobody can put
beside each other.

## This crate is not the validation suite

The harvested corpus, the stored observations, the corpus index, the censuses and audits and the
executors all belong to `metal2vulkan-validation`. None of it is this crate's business, and this
crate depends on none of it. What it uses from outside is `metal2vulkan-runtime`, the translator's
public API, `metal2vulkan-devtools` for fixture paths and `llvm_assemble`, and, on macOS,
`objc2-metal`.

The suite's own cases are its **case set** (`src/cases/`). Its recordings are its own, in
`recordings/`, in its own format; they are not observations and they enter no census.

A case ported from a runtime lane restates that lane's measured Metal values and reads the lane's
committed measurement files under `fixtures/public/`. Those are our own measurements over our own
fixtures, which is what the boundary allows; the lanes stay where they are and a ported case never
edits one.

## Layout

| Module | What it is |
|---|---|
| `transcript` | The record shape: a tree of sites, each a value, its shape and an optional digest |
| `allowance` | The justification a loosened field must name, checked against the degrade catalogue and `docs/API.md` |
| `compare`, `diff` | Field-by-field comparison, and the diff that names the leaf that moved |
| `verdict` | Exact, announced, divergent, refused, unrecorded, and per venue "lacks the device feature" |
| `recording` | A case's Metal answer, keyed by chip, macOS build and the release whose Metal it is, with its runs and its outcome set |
| `case` | The case API and the vocabulary both sides answer in |
| `runtime_side` | `metal2vulkan-runtime` as a side |
| `metal_side` | Native Metal as a side (macOS only), which record mode runs |
| `validated` | The second oracle: the same case under Metal's API validation layer, in a child process |
| `record`, `replay` | Taking Metal's answer, and comparing a run against it |
| `report` | Verdict counts and the matrix of case against venue |
| `cost` | The cost half: the counts our side moved, their regions, their bounds and the three cost verdicts |
| `work`, `runtime_work`, `metal_work` | The copy-and-fill vocabulary a case states as data, and the two sides that run it |
| `measured` | The committed Metal measurements a ported case reads, under the public fixtures |
| `argument_encoders` | The argument-encoder vocabulary: what a case asks an `MTLArgumentEncoder` to do, and the layout facts it can observe without a host's own addresses in them |
| `function_bindings` | What a function's reflection says its arguments are: `MTLBinding` and the type tree under it |
| `indirect_commands` | The indirect-command-buffer vocabulary: the descriptor a case states whole, and each command as the list of setter calls the host makes on it |
| `known_gaps` | The ledger of open gaps — a refusal or a divergence, per venue — held in both directions; it sets the exit status, never a verdict |
| `cases` | The case set itself, one module per family |

## The case set

| Family | What it compares | Metal's answer |
|---|---|---|
| `pass_bytes` | The words a compute pass wrote, exhaustively over a small input space | run on Metal |
| `math_routines` | The words `frexp` and `ilogb` write, scalar and per lane, over both widths | run on Metal |
| `stage_in_regions` | How a stage-in region clamps a dispatch's indices | run on Metal |
| `extended_range_planes` | A two-plane extended-range texture joined into one and split back again | run on Metal |
| `rgb10a2_srgb` | What a kernel reads, writes and samples of Apple's padded 10-bit sRGB format, and what a full-screen draw stores into it | run on Metal |
| `texture_buffer_fills` | Which bytes a fill changed over a texture made on a buffer | carried; `fillTexture:` is Apple SPI |
| `texture_fills`, `sample_fills` | Which bytes a texture fill, and a multisample sample fill, changed | carried; the per-format digests are the probes' own truncating formatter's, on an Apple M4 |
| `texture_fills`' heap runs | Every byte of the heap one two-plane fill changed, where the per-format digest elided the run | run on Metal, through the fill's SPI selector |
| `two_plane_heap_blits` | What a blit moved between two-plane textures placed in a heap | run on Metal |
| `two_plane_copies` | What a blit or a fill moved across a two-plane surface's rows | run on Metal where the case fills nothing; carried where it does (`fillTexture:` is Apple SPI) |
| `mesh_draws` | The texels each mesh threadgroup's primitive drew, and the grids Metal draws nothing for | run on Metal |
| `mesh_barriers` | Whether a reader draw sees a writer draw's words across a memory barrier | run on Metal |
| `mesh_commands` | What an indirect command buffer's own-state mesh commands drew, and what two commands with no barrier between them determine | run on Metal |
| `surface_fills` | Which bytes one private texture fill changed over a surface's plane or a buffer | carried; `fillTexture:` is Apple SPI |
| `indirect_private_types` | An indirect render command's read-back for each primitive type | carried; the getters are private |
| `pixel_samplers` | A pixel-coordinate sampler's samples, through a slot, an argument buffer, a normalized control and a constexpr sampler | run on Metal |
| `pixel_downscales` | A linear downscale by eight through a pixel-coordinate sampler, at seven texture sizes | run on Metal |
| `chroma_pairs` | What a copy of a 4:2:2 chroma-pair texture moved, and what a kernel's `read()` of one answers | run on Metal |
| `ycbcr_textures` | What Metal says about every two-plane YCbCr format's descriptors, and its `read()` of a `420v` surface's texels | the descriptors run on Metal; the texels carried, a YCbCr texture needs an IOSurface |
| `ycbcr_minify` | Where a minified 10-bit YCbCr texture's linear footprint puts the switch, along each axis | carried; a YCbCr texture needs an IOSurface |
| `compute_threadgroups` | What the threads of a 32-, 64- and 1024-thread threadgroup wrote through a simdgroup shuffle | run on Metal; the over-limit row carried from an Apple M4 |
| `buffer_blits` | Where a blit encoder's buffer fills and copies landed, overlapping ones included | run on Metal |
| `simdgroup_lockstep` | The threadgroup words two loops only a simdgroup's lockstep orders came to | run on Metal |
| `color_write_masks` | Which channels of a texel a draw changes under a write mask of none, some and all of them | run on Metal |
| `blend_sweeps` | Every settable `MTLBlendFactor` in each of the four slots, every `MTLBlendOperation` against six factor pairs and against each other, all sixteen write masks with the blend on and off, blending enabled against disabled, the blend colour at and outside its range, and the `Source1` factors through a second fragment output | run on Metal |
| `fresh_contents` | What a new buffer or texture reads before anything writes it, and after memory that was written was released | run on Metal |
| `argument_encoders` | What an argument encoder says of a layout, which word of an argument buffer each argument landed in, and what the kernel then sampled through it | run on Metal |
| `function_bindings` | What a kernel's reflection says each argument is: the binding, its subprotocol's fields, and an argument buffer's struct type member by member | run on Metal |
| `indirect_commands` | What an indirect command buffer's CPU-encoded compute and render commands ran: both dispatch kinds, ordinary and indexed draws, the resets, the barriers, an attribute stride, a stage-in region, and the descriptor properties no execution needs | run on Metal |

A carried answer is our own measurement of this chip, stated in the shape the case emits, and its
case forbids re-recording with the reason why. Record mode refuses such a case rather than
pretending it can ask Metal again.

**Apple SPI is not the same thing as absent.** `fillTexture:level:slice:region:bytes:length:` has
no public header, so `objc2-metal` binds nothing for it and the copy-and-fill vocabulary's pattern
fill was carried from `probes/`. `metal_work` reaches it by its selector after
`respondsToSelector:`, so a case built on it can be **run** rather than carried; the heap-run cases
above are the first. Two consequences worth stating plainly, because a stale reason in this table is
how `two_plane_copies` lost its Metal leg in the first place:

- `texture_fills`' per-format digests stay carried for a different reason than they used to. Not
  because the fill cannot be reached, but because each is a digest of twenty per-case answers of
  which only the outer digest was ever written down, and that one was taken on an Apple M4. Making
  those rows `MetalSource::Run` and recording per-case values on this chip is a packet of its own.
- `texture_buffer_fills`, and `two_plane_copies`' filling cases, say "`fillTexture:` is Apple SPI".
  That is still true of the words and no longer the whole reason: nothing has moved them yet. The
  colour fills (`fillTexture:...:color:` and `...:color:pixelFormat:`) have no arm in `metal_work`
  at all, so a case built on one is carried outright.

## `probes/`

Where a carried answer came from a call record mode cannot make — Apple SPI, a private getter —
the Objective-C probe that measured it is committed here, with the command that builds it in its
header comment. A probe deleted after one run is how a measurement evaporates: the
`RG10Unorm_X12_sRGB` row of `surface_fills` was missing for exactly that reason, because the
original probe was thrown away and the lane that kept its number printed only the first 24 bytes
of each 32-byte run. A probe that re-measures rows already carried, as a control, says whether its
new rows can be trusted.
