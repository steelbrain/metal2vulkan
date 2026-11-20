# metal2vulkan-conformance

The Metal conformance suite. It makes one sentence executable:

> We are a Metal implementation. Any difference from Metal is a bug, unless we announced it.

A case is written once, against one vocabulary, and run by either side: native Metal on macOS, and
`metal2vulkan-rt` on whatever Vulkan device the host has. Each run emits a **transcript**: a
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
`PrecisionNote`, under the suite's own `Degradation` codes), with the rule for what is in it and what is not.
They are held to a case declaration's rules — the notices site alone, justified only where the code
was announced — and they never count among the allowances a case named, so a native note a run did
not announce is not an allowance that never fired. A note of what the device **lacks**, of an
emulation's own precision, or of a Metal form not measured or not built is not in the set, and a
case that provokes one still names it.

**They declare nothing about a value.** Every value still compares field by field, and comparison
is never widened for the whole suite. A value the device's native answer moves is a divergence at
its own site. Nearly every note of the set says in its own documentation that validation derives no
tolerance from it, and the suite loosens no value by such a note either: that would be the suite
overruling the product's stated contract. So the site is a `known-gaps-rt.tsv` row, scoped to the
venue whose device answers so, that says what differs, whose answer it is, and whether the note's
recorded bound covers it. A tolerance justified by a note is for a note whose documentation gives a
bound validation may hold, and is never wider than that bound.

A tolerance may name one element of a byte region rather than the whole site: the element's path is
the site's and its byte offset in the region, `case/pass-0/words/24`. The visibility family loosens
each `Boolean` result's word that way, and only by `Loosening::Nonzero` (zero exactly where Metal's
word is zero), the bound `device-boolean-visibility` documents; every other word of the buffer
still compares exactly.

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

A refusal the **venue** honestly cannot answer is not a gap of ours and reads `lacks-feature`
instead. That fact rides in the failure's own recorded value, beside the typed error, so it reaches
the comparison wherever in a case the refusal happened: a case whose site is one node but whose rows
are many names the row in its abort and pushes the node at the site, and filing the fact under a
path the case states twice is how one run came to score the same missing extension `refused` in one
family and `lacks-feature` in another. A getter whose answer is the device's limit reads the same
way where the device refused the work of Metal's size as its gap: a tile state whose descriptor
states no limit answers the device's `maxTotalThreadsPerThreadgroup`, below Metal's 1024 where the
device holds fewer threads of the pipeline, and its tile dispatch of Metal's threadgroup is refused.

## Does it answer the same thing twice?

```sh
conformance replay --runs 3                   # three passes, one process, one device
```

Every other reading the suite takes is of one run, so a reading that moves run to run is
indistinguishable from one that moved because the implementation changed — and a one-shot gate gives
the second answer, because it is the only one it has. `--runs N` replays the selection N times in
the one process and flags every verdict and every transcript site that is not the same in all N, naming the case, the reading and what each run read, in order. A time is deliberately
not a reading: it moves on every host under every load.

It is off by default, and it has its own line of the exit status.

## The known-gaps ledger

`known-gaps-rt.tsv` holds the gaps we have written down as open. Each row is an **open bug with a todo
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
case, against the venue's own description — its device name, for a CPU implementation the host's
widest vector instruction set (`llvmpipe (LLVM 23.1.4, 256 bits) on x86-64 AVX-512`, which its
compiler targets and its answers follow), then its driver and API version — so a row can name one
device (`Apple M3 Max`), one implementation (`llvmpipe`), one host's (`on x86-64 AVX2`) or a family
of them, and a venue that joins later needs no new vocabulary. `*` means every venue, and it is the
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

So the rule stands, and is now mechanical as well: **a `known-gaps-rt.tsv` row is never retired on the
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

## Metal's times

`conformance times` takes what each call of the selection costs on native Metal on this Mac and
prints it. The figures gate nothing.

```sh
conformance times --runs 8                  # Metal's leg, print the table
conformance times --runs 8 --show           # and every per-case row, not only the interesting ones
conformance times --runs 8 --write          # keep the figures in the cases' recordings
```

The leg is measured at `watch::Watching`'s span: around the Metal call and the side's own handle
bookkeeping, outside the transcript, the fixtures and the input fingerprint. The table splits each
call's wall time into the calling thread's own CPU and the time it was blocked. The instrument's own
cost a call is measured on the host and printed above the table; it is never subtracted.

`--write` keeps the figures in the case's own recording, because Metal's cost on a given macOS build
cannot be re-taken once the machine updates. A recording may also hold a `costs` field (work counts
and their bounds) and a leg of ours beside Metal's under `times`. Nothing reads or writes either:
`metal2vulkan-rt` keeps no work counter the suite can read, and `times` measures Metal's leg alone.
`known-slow-calls.tsv`, the per-call ledger that compared the two legs, is likewise read by nothing
this command runs.

## This crate is not the validation suite

The harvested corpus, the stored observations, the corpus index, the censuses and audits and the
executors all belong to `metal2vulkan-validation`. None of it is this crate's business, and this
crate depends on none of it. What it uses from outside is `metal2vulkan-rt`, the translator's
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
| `rt_side` | `metal2vulkan-rt` as a side |
| `metal_side` | Native Metal as a side (macOS only), which record mode runs |
| `validated` | The second oracle: the same case under Metal's API validation layer, in a child process |
| `record`, `replay` | Taking Metal's answer, and comparing a run against it |
| `report` | Verdict counts and the matrix of case against venue |
| `cost` | The cost record a recording may hold: work counts, their regions and their bounds |
| `work`, `rt_side::work`, `metal_work` | The copy-and-fill vocabulary a case states as data, and the two sides that run it |
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
| `nan_integer_conversions` | The words a kernel and a fragment function write for MSL's conversions of NaNs and of other values from `float` and `half` to every integer width, signed and unsigned, compiled with fast math off and on | run on Metal |
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
| `surface_textures` | What a texture of an ordinary format over a `'BGRA'`, `'RGhA'`, `'420v'` or `'x420'` surface's plane reads back, by `getBytes:` and by a blit, after `replaceRegion:` wrote it | run on Metal |
| `texture_view_casts` | A cube viewed as a 2D array (every level and face, and some) and as a 2D texture, a 2D texture as a 2D array, and six slices of a square 2D array as a cube: the view rendered into and the parent sampled, then the parent rendered into and the view sampled, and the view's type, levels and array length | run on Metal |
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
