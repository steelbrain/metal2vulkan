//! What a case observed, as a tree of named sites.
//!
//! A site holds one [`Observed`]: a [`Value`] in the portable vocabulary, the [`Shape`] that says
//! how to read and compare it, and an optional [`Digest`], a projection of the value a recording
//! may carry in its place. One record type serves every fidelity, so a case ported digest-first
//! gains raw values without being rewritten.
//!
//! The tree, rather than a list, is what lets a failure descend: a case whose answer is a digest of
//! digests of per-read-back lines keeps each level as a node, and the diff names the leaf that
//! moved.
//!
//! Nothing here names a Metal object, a host path or a device address, so a recording of a case is
//! the same file whichever host wrote it.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// A site's name inside its parent: stable across the sides, because the comparison pairs the
/// sides' nodes by it.
pub type NodeName = String;

/// A node of a case's transcript: the site's name, what it observed, and the sites under it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Node {
    pub name: NodeName,
    pub observed: Observed,
    /// The sites under this one, in the order the case emitted them. The comparison pairs them by
    /// name, so a side that emits them in another order still compares; a name the other side
    /// does not hold is a divergence of its own.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<Node>,
}

impl Node {
    /// A leaf site: `name` observed `observed`.
    pub fn leaf(name: impl Into<NodeName>, observed: Observed) -> Self {
        Self {
            name: name.into(),
            observed,
            children: Vec::new(),
        }
    }

    /// A site that observed nothing of its own and holds `children`.
    pub fn branch(name: impl Into<NodeName>, children: Vec<Node>) -> Self {
        Self {
            name: name.into(),
            observed: Observed::absent(),
            children,
        }
    }

    /// The node with `child` added under it.
    #[must_use]
    pub fn with(mut self, child: Node) -> Self {
        self.children.push(child);
        self
    }

    /// The child named `name`, if the node holds one.
    pub fn child(&self, name: &str) -> Option<&Node> {
        self.children.iter().find(|child| child.name == name)
    }

    /// The names this node holds more than one child under, each with how many it holds.
    ///
    /// A site's path is how the comparison finds the answer to hold it against, and [`Self::child`]
    /// answers with the **first** namesake, so two children under one name are a pair the walk
    /// cannot settle: it compares a site against whichever namesake it reaches, and the other
    /// site's observation is never compared with anything. Nothing refuses it -- [`Self::with`]
    /// and [`Sites::push`] append whatever they are given -- so the loss is silent, with no error
    /// and no change of verdict. A case of this round named the call that made its buffer and the
    /// buffer's bytes after the pass both `buffer`, and its replay reported a difference at
    /// `case/buffer` between Metal's bytes and our `Ok` of the make, having compared no byte of
    /// either.
    ///
    /// So this is the one reading of the rule, and both the places that hold it call it: the
    /// comparison, which reports a repeat rather than pairing by luck, and
    /// `no_recorded_site_shares_its_name_with_a_sibling`, which holds every committed recording to
    /// it. Children keep their order, so the names come back in the order the repeats appear.
    pub fn repeated_children(&self) -> Vec<(&str, usize)> {
        let mut held: Vec<(&str, usize)> = Vec::new();
        for child in &self.children {
            let name = child.name.as_str();
            match held.iter_mut().find(|(seen, _)| *seen == name) {
                Some((_, count)) => *count += 1,
                None => held.push((name, 1)),
            }
        }
        held.retain(|(_, count)| *count > 1);
        held
    }
}

/// What one site observed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observed {
    /// The value, [`Value::Absent`] where the site observed nothing of its own (a branch, or a
    /// recording that carries the digest alone).
    pub value: Value,
    /// How to read and compare the value.
    #[serde(default, skip_serializing_if = "Shape::is_value")]
    pub shape: Shape,
    /// A projection of `value`, which the comparison checks beside it. A recording taken
    /// digest-first holds the digest with [`Value::Absent`]; the comparison then checks the
    /// digest alone and says so, so the case's verdict never reads as exact over bytes nobody
    /// recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub digest: Option<Digest>,
}

impl Observed {
    /// A site that observed `value`, compared as itself.
    pub fn value(value: Value) -> Self {
        Self {
            value,
            shape: Shape::Value,
            digest: None,
        }
    }

    /// A site that observed nothing of its own.
    pub fn absent() -> Self {
        Self::value(Value::Absent)
    }

    /// A site whose bytes are the region `layout` names.
    pub fn bytes(bytes: Vec<u8>, layout: ByteLayout) -> Self {
        Self {
            value: Value::Bytes(bytes),
            shape: Shape::Bytes(layout),
            digest: None,
        }
    }

    /// The site with `digest` recorded beside its value.
    #[must_use]
    pub fn digested(mut self, digest: Digest) -> Self {
        self.digest = Some(digest);
        self
    }

    /// Whether the value is absent, so only the digest or the children carry the fact.
    pub fn is_absent(&self) -> bool {
        self.value == Value::Absent
    }
}

/// A value a Metal program can observe, in a vocabulary both sides can produce and a host without
/// Metal can read back from a file.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Value {
    /// The site observed nothing of its own.
    Absent,
    /// A fact about what did not happen, from the vocabulary ([`Negative`]). An absent field is
    /// never such a fact: a case that means "this call did not block" says so.
    Nothing(Negative),
    Bool(bool),
    Uint(u64),
    Int(i64),
    /// A float by its bits, so a NaN's payload and a signed zero compare exactly and an allowance
    /// in units in the last place has a width to work in.
    Float {
        bits: u64,
        width: FloatWidth,
    },
    /// Bytes, whose layout its site's [`Shape::Bytes`] names.
    Bytes(#[serde(with = "crate::base64_bytes")] Vec<u8>),
    /// Lines of text in the order they were printed: the cheapest shape to port a lane whose
    /// expectation is already an ordered list of probe lines.
    Lines(Vec<String>),
    /// A set, compared without order: the diff prints what is missing, what is extra and what both
    /// sides hold.
    Set(BTreeSet<String>),
    /// Values in order.
    List(Vec<Value>),
    /// A named variant and its payload: a typed answer or a typed error, whose variant and payload
    /// both compare.
    Variant {
        name: String,
        payload: Box<Value>,
    },
    /// Named fields.
    Fields(BTreeMap<String, Value>),
}

/// The variant name a call that did not answer records under, and the one the comparison reads a
/// refusal from.
const ERR: &str = "Err";

impl Value {
    /// `Ok(payload)`, a call's answer.
    pub fn ok(payload: Value) -> Self {
        Self::Variant {
            name: "Ok".to_owned(),
            payload: Box::new(payload),
        }
    }

    /// `Err(payload)`, a call's typed error: `name` is the variant's path and `payload` its
    /// fields.
    pub fn error(name: impl Into<String>, payload: Value) -> Self {
        Self::err(Self::Variant {
            name: name.into(),
            payload: Box::new(payload),
        })
    }

    /// `Err(error)`, where `error` is already whatever the typed error is.
    ///
    /// The one place the `Err` name is spelled, because the comparison reads a **refusal** out of
    /// exactly this shape — Metal answering a call and us answering `Err`
    /// ([`crate::compare`]) — so a value that means "we did not answer" and is not an `Err` is a
    /// refusal the suite cannot see. [`crate::case::Failure`] keeps the typed error alone and adds
    /// this layer as it records, so nothing a failure carries can displace it.
    pub fn err(error: Value) -> Self {
        Self::Variant {
            name: ERR.to_owned(),
            payload: Box::new(error),
        }
    }

    /// A variant of `name` with no payload.
    pub fn unit_variant(name: impl Into<String>) -> Self {
        Self::Variant {
            name: name.into(),
            payload: Box::new(Value::Absent),
        }
    }

    /// Fields from pairs.
    pub fn fields<K: Into<String>>(pairs: impl IntoIterator<Item = (K, Value)>) -> Self {
        Self::Fields(
            pairs
                .into_iter()
                .map(|(key, value)| (key.into(), value))
                .collect(),
        )
    }

    /// A set from words.
    pub fn set<S: Into<String>>(words: impl IntoIterator<Item = S>) -> Self {
        Self::Set(words.into_iter().map(Into::into).collect())
    }

    /// Whether the value is a call's `Err`.
    pub fn is_error(&self) -> bool {
        matches!(self, Value::Variant { name, .. } if name == ERR)
    }

    /// The word the report prints for the value's kind.
    pub fn kind(&self) -> &'static str {
        match self {
            Value::Absent => "absent",
            Value::Nothing(_) => "nothing",
            Value::Bool(_) => "bool",
            Value::Uint(_) => "uint",
            Value::Int(_) => "int",
            Value::Float { .. } => "float",
            Value::Bytes(_) => "bytes",
            Value::Lines(_) => "lines",
            Value::Set(_) => "set",
            Value::List(_) => "list",
            Value::Variant { .. } => "variant",
            Value::Fields(_) => "fields",
        }
    }
}

/// A fact about what did not happen. A case states one of these rather than leaving a field out,
/// so the comparison can tell "nothing happened" from "nobody looked".
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Negative {
    /// The encoder, pass or command buffer ran none of its work.
    RanNothing,
    /// The draw covered no pixel.
    DrewNothing,
    /// The bytes are the ones that were there before.
    ChangedNothing,
    /// The copy moved no byte.
    CopiedNoBytes,
    /// No degrade notice was announced.
    AnnouncedNothing,
    /// The call never returned.
    NeverReturned,
    /// The call returned without waiting for anything.
    DidNotBlock,
    /// The call returned without waiting for the device's work.
    DidNotWaitOnTheGpu,
}

/// A float's width, for a value's bits and for an allowance in units in the last place.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FloatWidth {
    Half,
    Single,
    Double,
}

/// A projection of a site's value, which a recording may carry in place of the value.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Digest {
    /// How the digest was taken, so a side computes the same projection: `sha256` of the bytes.
    pub kind: DigestKind,
    /// The projection, lowercase hexadecimal.
    pub hex: String,
}

impl Digest {
    /// The SHA-256 of `bytes`.
    pub fn sha256(bytes: &[u8]) -> Self {
        Self {
            kind: DigestKind::Sha256,
            hex: crate::sha256_hex(bytes),
        }
    }

    /// The first sixteen hexadecimal digits of the SHA-256 of `bytes`.
    ///
    /// This is the projection the probes printed and the runtime lanes keep, so a case ported
    /// from such a lane records Metal's own value rather than a digest nobody measured.
    pub fn sha256_prefix16(bytes: &[u8]) -> Self {
        let mut hex = crate::sha256_hex(bytes);
        hex.truncate(16);
        Self {
            kind: DigestKind::Sha256Prefix16,
            hex,
        }
    }

    /// The digest of `value`, where its kind projects it: bytes, and lines joined by newlines, as
    /// the probes digested their printed lines.
    pub fn of(kind: DigestKind, value: &Value) -> Option<Self> {
        let bytes = match value {
            Value::Bytes(bytes) => bytes.clone(),
            Value::Lines(lines) => lines.join("\n").into_bytes(),
            _ => return None,
        };
        Some(match kind {
            DigestKind::Sha256 => Self::sha256(&bytes),
            DigestKind::Sha256Prefix16 => Self::sha256_prefix16(&bytes),
        })
    }
}

/// How a [`Digest`] projects a value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DigestKind {
    Sha256,
    /// The SHA-256's first sixteen hexadecimal digits: what our probes printed, and so the widest
    /// projection a case ported from a lane that kept one can honestly record.
    Sha256Prefix16,
}

/// How to read and compare a site's value.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Shape {
    /// The value is compared as itself, by its own kind.
    #[default]
    Value,
    /// The value's bytes are the region `0` names, read element by element.
    Bytes(ByteLayout),
}

impl Shape {
    /// Whether the shape is the default, so a recording leaves it out.
    pub fn is_value(&self) -> bool {
        matches!(self, Shape::Value)
    }
}

/// Where a site's bytes came from and how they are laid out, so a diff names the element that
/// moved rather than a byte offset, and the bytes between the rows are compared where the case
/// asks.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ByteLayout {
    /// The resource and the part of it the bytes are.
    pub region: Region,
    /// What one element is.
    pub element: Element,
    /// Bytes from a row's start to its first element: a column of an interleaved layout, where a
    /// case compares one of a thread's words apart from the others because only that one is
    /// loosened.
    #[serde(default)]
    pub row_offset: u32,
    /// Elements in a row.
    pub row_elements: u32,
    /// Bytes from one row's start to the next: the row pitch, at least
    /// `row_elements * element.size()`.
    pub row_pitch: u32,
    /// Rows in an image.
    pub rows: u32,
    /// Bytes from one image's start to the next: the image pitch, at least `rows * row_pitch`.
    pub image_pitch: u32,
    /// Images.
    pub images: u32,
    /// Whether the bytes between a row's elements and the next row's start, and between an
    /// image's rows and the next image's start, are compared too.
    pub gaps: Gaps,
}

impl ByteLayout {
    /// A buffer's `length` bytes from `offset`, as `element`s in one row of one image, with no
    /// gaps to compare.
    pub fn buffer(offset: u64, length: u64, element: Element) -> Self {
        let pitch = u32::try_from(length).unwrap_or(u32::MAX);
        Self {
            region: Region::Buffer { offset, length },
            element,
            row_offset: 0,
            row_elements: pitch / element.size(),
            row_pitch: pitch,
            rows: 1,
            image_pitch: pitch,
            images: 1,
            gaps: Gaps::Ignored,
        }
    }

    /// The bytes the layout describes, at least.
    pub fn extent(&self) -> usize {
        let images = self.images.saturating_sub(1) as usize * self.image_pitch as usize;
        let rows = self.rows.saturating_sub(1) as usize * self.row_pitch as usize;
        let row =
            self.row_offset as usize + self.row_elements as usize * self.element.size() as usize;
        images + rows + row
    }

    /// The byte offset of element `index` of row `row` of image `image`, inside the value.
    pub fn at(&self, image: u32, row: u32, index: u32) -> usize {
        image as usize * self.image_pitch as usize
            + row as usize * self.row_pitch as usize
            + self.row_offset as usize
            + index as usize * self.element.size() as usize
    }
}

/// Which resource a site's bytes came from, and which part of it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Region {
    Buffer {
        offset: u64,
        length: u64,
    },
    Texture {
        level: u32,
        slice: u32,
        aspect: Aspect,
        /// The sample a multisample texture's bytes are of.
        sample: u32,
        origin: [u32; 3],
        size: [u32; 3],
    },
}

/// Which plane of a texture a site's bytes are.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Aspect {
    Color,
    Depth,
    Stencil,
    /// A planar format's plane.
    Plane(u32),
}

/// Whether the bytes outside a layout's elements are compared.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Gaps {
    /// The bytes between the rows and the images compare too: a case that asks whether the padding
    /// stayed as it was.
    Compared,
    /// Only the elements compare.
    Ignored,
}

/// What one element of a site's bytes is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Element {
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    F16,
    F32,
}

impl Element {
    /// The bytes one element takes.
    pub fn size(self) -> u32 {
        match self {
            Element::U8 | Element::I8 => 1,
            Element::U16 | Element::I16 | Element::F16 => 2,
            Element::U32 | Element::I32 | Element::F32 => 4,
            Element::U64 => 8,
        }
    }

    /// `bytes` read as one element, where they are long enough.
    pub fn read(self, bytes: &[u8]) -> Option<Value> {
        let size = self.size() as usize;
        let taken = bytes.get(..size)?;
        let mut word = [0u8; 8];
        word[..size].copy_from_slice(taken);
        let raw = u64::from_le_bytes(word);
        let signed = |bits: u32| ((raw << (64 - bits)) as i64) >> (64 - bits);
        Some(match self {
            Element::U8 | Element::U16 | Element::U32 | Element::U64 => Value::Uint(raw),
            Element::I8 => Value::Int(signed(8)),
            Element::I16 => Value::Int(signed(16)),
            Element::I32 => Value::Int(signed(32)),
            Element::F16 => Value::Float {
                bits: raw,
                width: FloatWidth::Half,
            },
            Element::F32 => Value::Float {
                bits: raw,
                width: FloatWidth::Single,
            },
        })
    }
}

/// Everything one run of a case observed, under the root site the case names.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Transcript {
    pub root: Node,
    /// The sites where the **venue honestly lacks** what the call needed, by their paths from the
    /// root, each with what it lacks ([`crate::verdict::device_gap`]). A refused call's site, and
    /// a getter's whose answer is a limit the device holds below Metal's, where the device refused
    /// the work of Metal's size as its gap (a tile state's `maxTotalThreadsPerThreadgroup`,
    /// [`crate::cases::tile_threadgroups`]): either differs from Metal's as the venue's gap.
    ///
    /// Our side's own fact about the device, not an observation of anything Metal did, so it is
    /// kept apart from the tree: it is never compared, never fingerprinted and never written to a
    /// recording ([`Vec::is_empty`] skips it), and Metal's side never holds one, because the fact
    /// is read from a typed runtime error. A recording from before this field existed reads it as
    /// empty.
    ///
    /// Why it exists: a device's honest gap must not read as a refusal of ours, and until now the
    /// only way to say so was to **stop the case** — the whole-case
    /// [`LacksFeature`](crate::verdict::Verdict::LacksFeature) verdict comes from an abort. A case
    /// whose one unsupported site is not fatal could not say it at all, so coverage on a venue
    /// that lacks one feature read as a gap of ours at that site, which is the one thing the
    /// verdict exists to prevent.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub device_gaps: Vec<(String, String)>,
}

impl Transcript {
    /// A transcript whose root holds `children`.
    pub fn new(children: Vec<Node>) -> Self {
        Self {
            root: Node::branch("case", children),
            device_gaps: Vec::new(),
        }
    }

    /// What the venue lacks at `path`, where it lacks anything.
    pub fn device_gap_at(&self, path: &str) -> Option<&str> {
        self.device_gaps
            .iter()
            .find(|(site, _)| site == path)
            .map(|(_, needs)| needs.as_str())
    }

    /// Every degrade notice the transcript records, by code, with how many times it was
    /// announced: the `counts` child of each site [`Sites::notices`] emitted, wherever it sits in
    /// the tree.
    ///
    /// The closed-vocabulary notice census (`runtime_lane::degrade_notices`) stays a lane and
    /// reads the suite's runs through this, rather than becoming a case: its subject is the
    /// vocabulary across every lane, which has no Metal counterpart to compare against.
    pub fn announced_notices(&self) -> BTreeMap<String, u64> {
        let mut found = BTreeMap::new();
        for announced in self.announced_at() {
            *found.entry(announced.code).or_default() += announced.times;
        }
        found
    }

    /// Every degrade notice the transcript records, with the **site** that announced it
    /// ([`AnnouncedAt`]).
    ///
    /// The site is what the comparison needs, because a notice our side announced and no
    /// declaration of the case names is a difference with nothing justifying it, and the row has
    /// to say where it was announced. [`Transcript::announced_notices`] is this, totalled, so
    /// there is one walk of the tree and not two readings of one rule.
    pub fn announced_at(&self) -> Vec<AnnouncedAt> {
        let mut found = Vec::new();
        collect_notices("", &self.root, &mut found);
        found
    }
}

/// A degrade notice one site of a [`Transcript`] announced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnnouncedAt {
    /// The site that announced it, as node names from the root: what a notice declaration's
    /// allowance path covers.
    pub path: String,
    /// The notice's code (`Degradation::code`).
    pub code: String,
    /// How many times that site announced it.
    pub times: u64,
}

/// The name [`Sites::notices`] gives the child that holds a code's count.
const NOTICE_COUNTS: &str = "counts";

/// Add every notice `node` and its children announced to `found`, under `parent`'s path.
fn collect_notices(parent: &str, node: &Node, found: &mut Vec<AnnouncedAt>) {
    let path = if parent.is_empty() {
        node.name.to_string()
    } else {
        format!("{parent}/{}", node.name)
    };
    if let Some(Node {
        observed: Observed {
            value: Value::Fields(counts),
            ..
        },
        ..
    }) = node.child(NOTICE_COUNTS)
    {
        for (code, count) in counts {
            if let Value::Uint(times) = count {
                found.push(AnnouncedAt {
                    path: path.clone(),
                    code: code.clone(),
                    times: *times,
                });
            }
        }
    }
    for child in &node.children {
        collect_notices(&path, child, found);
    }
}

/// A transcript under construction: the sites a run emits, in order.
#[derive(Debug, Default)]
pub struct Sites {
    nodes: Vec<Node>,
    device_gaps: Vec<(String, String)>,
}

impl Sites {
    /// No site yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add `node`.
    pub fn push(&mut self, node: Node) {
        self.nodes.push(node);
    }

    /// Add a leaf site.
    pub fn leaf(&mut self, name: impl Into<NodeName>, observed: Observed) {
        self.push(Node::leaf(name, observed));
    }

    /// Add a site holding `value`.
    pub fn value(&mut self, name: impl Into<NodeName>, value: Value) {
        self.leaf(name, Observed::value(value));
    }

    /// Record that the venue honestly lacks `needs` at the site named `name`, which
    /// [`Transcript::device_gaps`] carries by its path from the root.
    ///
    /// Every failing call goes through [`crate::case::call`], which records this whenever the
    /// typed error names capabilities a device could enable or a measured driver defect, so a
    /// case needs no code of its own and cannot forget to say it.
    /// A case may state it at a site whose answer is a value, not an error: a getter answering a
    /// limit the device holds below Metal's, where the device refused the work of Metal's size as
    /// its gap, which the comparison then reads as that gap ([`Transcript::device_gaps`]).
    /// Recording the same site twice is a no-op, so the one place that reads a case's
    /// [`Abort`](crate::case::Abort) may record the venue's gap at the site that stopped the case
    /// without having to know whether [`crate::case::call`] already did.
    pub fn device_gap(&mut self, name: &str, needs: impl Into<String>) {
        let path = format!("case/{name}");
        if self.device_gaps.iter().any(|(site, _)| *site == path) {
            return;
        }
        self.device_gaps.push((path, needs.into()));
    }

    /// Add the two sites that record what a device announced: the exact set of degrade-notice
    /// codes, and how many times each was announced, so a case can hold Metal's empty set and ours
    /// to one notice per kind. An empty set records [`Negative::AnnouncedNothing`].
    pub fn notices<S: AsRef<str>>(&mut self, name: impl Into<NodeName>, codes: &[S]) {
        let mut counts: BTreeMap<String, Value> = BTreeMap::new();
        for code in codes {
            let entry = counts
                .entry(code.as_ref().to_owned())
                .or_insert(Value::Uint(0));
            if let Value::Uint(count) = entry {
                *count += 1;
            }
        }
        let set = if codes.is_empty() {
            Value::Nothing(Negative::AnnouncedNothing)
        } else {
            Value::set(codes.iter().map(|code| code.as_ref().to_owned()))
        };
        self.push(
            Node::branch(name, Vec::new())
                .with(Node::leaf("set", Observed::value(set)))
                .with(Node::leaf(
                    NOTICE_COUNTS,
                    Observed::value(Value::Fields(counts)),
                )),
        );
    }

    /// The transcript of the sites emitted so far.
    pub fn finish(self) -> Transcript {
        Transcript {
            root: Node::branch("case", self.nodes),
            device_gaps: self.device_gaps,
        }
    }
}
