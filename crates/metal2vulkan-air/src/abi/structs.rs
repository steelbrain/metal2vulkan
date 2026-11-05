//! `air.struct_type_info` layouts.
//!
//! A layout node is a flat list of members; each member is
//! `[air.struct_type_info, !nested] <offset> <size> <array len> <type name> <name> <suffix>*`,
//! where a suffix is `air.indirect_argument, (!arg | <first id>)`, `air.render_target, <n>`, or
//! `air.raster_order_group, <n>`. Layouts are decoded once per node into [`AirAbi::structs`]
//! and shared by reference, so a layout reachable along many paths costs one decode, and layouts
//! whose members decode alike are one entry, however the module spells their nodes and the struct
//! types that hold them (a typed and an opaque spelling of one module decode the same table); a layout
//! that contains itself is `BadValue`, and one nested deeper than [`Limit::LayoutNesting`] is
//! refused. The decoder walks nested layouts with an explicit stack (a layout's members are read
//! first, then the layouts they name are decoded, then the layout is built), so its host stack
//! use is the same at every depth.
//!
//! [`AirAbi::structs`]: super::AirAbi::structs

use std::collections::{BTreeMap, BTreeSet};
use std::hash::{DefaultHasher, Hash, Hasher};

use super::args::{field_holder, key_of, Cx, MemberSpace, Node};
use super::keys::Key;
use super::md::{kept_bytes, kept_diag, Ops, Site, Tok};
use super::{Member, MemberKind, StructInfo, StructRef, UserKey};
use crate::error::{AbiError, DiagName};
use crate::error::{Error, InternalError, InternalKind, Limit};
use crate::ir::{MdId, TypeId};

/// A layout node together with the IR struct type whose fields its members describe (`None`
/// where the module does not state one). Both decide what the layout decodes to, so both key
/// [`Structs::memo`]: a member with no `air.address_space` takes it from its field.
type LayoutKey = (MdId, Option<TypeId>);

/// Ledger charge for one [`Structs::active`] entry (a tree entry, at twice its size).
const ACTIVE_ENTRY: usize = 2 * std::mem::size_of::<LayoutKey>();
/// Ledger charge for one [`Structs::memo`] entry.
const MEMO_ENTRY: usize = 2 * std::mem::size_of::<(LayoutKey, StructRef)>();
/// Ledger charge for one [`Structs::by_content`] entry.
const CONTENT_ENTRY: usize = 2 * std::mem::size_of::<(u64, StructRef)>();
/// Ledger charge for one frame of the decode stack.
const FRAME: usize = std::mem::size_of::<Frame>();
/// Ledger charge for one boxed argument-buffer member node.
const NODE: usize = std::mem::size_of::<Node>();

/// The layout arena under construction.
///
/// A layout names nested layouts (directly, or through an argument-buffer member's binding), so
/// both the decode depth and the height of every finished layout are capped at
/// [`Limit::LayoutNesting`]. The decoder walks nested layouts with an explicit stack of
/// [`Frame`]s, so its host stack use does not grow with nesting; a consumer that walks a decoded
/// layout is bounded by the height cap, which already counts the layouts shared by reference.
#[derive(Default)]
pub(super) struct Structs {
    pub(super) list: Vec<StructInfo>,
    /// The height of each layout of `list`: 1 for a layout with no nested layout.
    heights: Vec<u32>,
    /// Whether each layout of `list` holds an argument-buffer member (`air.indirect_argument`)
    /// at any depth by value.
    indirect: Vec<bool>,
    memo: BTreeMap<LayoutKey, StructRef>,
    /// Each built layout of `list` by the hash of its members: a layout whose members equal an
    /// earlier one's is that layout. Two keys of `memo` can decode alike (one layout node held by
    /// two struct types, or two nodes a spelling does not unique), and the table depends only on
    /// what they decode to.
    by_content: BTreeMap<u64, StructRef>,
    active: BTreeSet<LayoutKey>,
}

/// One layout being decoded: its members as read, and the layout nodes they name, which are
/// all decoded (in order) before the layout is built.
struct Frame {
    key: LayoutKey,
    site: Site,
    members: Vec<Read>,
    /// The layout nodes the members name, in the order the members name them, each with the
    /// struct type whose fields its members describe.
    nested: Vec<(LayoutKey, Site)>,
    /// How many of `nested` are decoded.
    done: usize,
    /// The greatest height of a decoded nested layout.
    height: u32,
}

/// One member as read, before its nested layouts are decoded.
struct Read {
    layout: Option<LayoutKey>,
    /// What the holding struct's field at `offset` says about this member's address space.
    space: MemberSpace,
    /// The holding struct's field at `offset`, the holding type of the layouts this member names.
    field: Option<TypeId>,
    offset: u32,
    size: u32,
    array_len: u32,
    type_name: DiagName,
    name: DiagName,
    key: UserKey,
    kind: ReadKind,
    render_target: Option<u32>,
    raster_order_group: Option<u32>,
}

/// A member's `air.indirect_argument` suffix, as read.
enum ReadKind {
    Plain,
    Indirect(Box<Node>),
    IndirectStruct { first_id: u32 },
}

impl Structs {
    /// The layout of node `id`, whose members are the fields of `holder` where the module states
    /// one, decoding it (and every layout it names) on first use.
    pub(super) fn get(
        cx: &mut Cx<'_, '_>,
        id: MdId,
        site: &Site,
        holder: Option<TypeId>,
    ) -> Result<StructRef, Error> {
        let key = (id, holder);
        if let Some(&r) = cx.structs.memo.get(&key) {
            return Ok(r);
        }
        let mut stack = Vec::new();
        open(cx, &mut stack, key, site)?;
        let mut last = None;
        while let Some(top) = stack.last_mut() {
            if let Some((child, child_site)) = top.nested.get(top.done) {
                if let Some(&r) = cx.structs.memo.get(child) {
                    top.height = top.height.max(cx.structs.height(r));
                    top.done += 1;
                } else {
                    let (child, child_site) = (*child, child_site.clone());
                    open(cx, &mut stack, child, &child_site)?;
                }
                continue;
            }
            let Some(frame) = stack.pop() else { break };
            cx.meter.release(FRAME);
            let (r, height) = finish(cx, frame)?;
            if let Some(parent) = stack.last_mut() {
                parent.height = parent.height.max(height);
                parent.done += 1;
            }
            last = Some(r);
        }
        // `open` pushed a frame, and the last frame popped is the one for `id`.
        last.ok_or_else(|| {
            Error::Internal(InternalError::new(
                InternalKind::Invariant,
                "layout decode finished no frame",
            ))
        })
    }

    /// The height of decoded layout `r`.
    fn height(&self, r: StructRef) -> u32 {
        self.heights.get(r.0 as usize).copied().unwrap_or(0)
    }

    /// Whether decoded layout `r` holds an argument-buffer member (`air.indirect_argument`), at
    /// any depth by value. Metal writes that key only in an argument buffer's layout.
    pub(super) fn holds_indirect(&self, r: StructRef) -> bool {
        self.indirect.get(r.0 as usize).copied().unwrap_or(false)
    }
}

/// Start decoding layout `id` on top of `stack`: refuse a layout that contains itself or one
/// nested past the cap, then read its members.
fn open(
    cx: &mut Cx<'_, '_>,
    stack: &mut Vec<Frame>,
    layout: LayoutKey,
    site: &Site,
) -> Result<(), Error> {
    let key = Some(Key::StructTypeInfo.name());
    // Tree entries are charged at twice their size (node overhead), before insertion.
    cx.meter.charge(ACTIVE_ENTRY)?;
    if !cx.structs.active.insert(layout) {
        return Err(site.err(AbiError::BadValue, key));
    }
    cx.meter.cap(Limit::LayoutNesting, stack.len() as u64 + 1)?;
    let frame = read(cx, layout, site)?;
    // Charged per push and released per pop ([`FRAME`]): the stack's charge is its depth.
    cx.meter.reserve(stack, 1)?;
    stack.push(frame);
    Ok(())
}

/// Build the layout of `frame`, whose nested layouts are all decoded, into the arena.
fn finish(cx: &mut Cx<'_, '_>, frame: Frame) -> Result<(StructRef, u32), Error> {
    let key = Some(Key::StructTypeInfo.name());
    let site = &frame.site;
    let height = frame.height.saturating_add(1);
    cx.meter.cap(Limit::LayoutNesting, u64::from(height))?;
    // The frame's lists were charged per element; the built members replace them.
    cx.meter.release(
        std::mem::size_of::<Read>().saturating_mul(frame.members.len())
            + std::mem::size_of::<(LayoutKey, Site)>().saturating_mul(frame.nested.len()),
    );
    let mut members = Vec::new();
    cx.meter.reserve(&mut members, frame.members.len())?;
    for read in frame.members {
        let decoded = |cx: &Cx<'_, '_>, id: LayoutKey| {
            cx.structs
                .memo
                .get(&id)
                .copied()
                .ok_or_else(|| site.err(AbiError::BadValue, key))
        };
        let layout = match read.layout {
            Some(id) => Some(decoded(cx, id)?),
            None => None,
        };
        let kind = match read.kind {
            ReadKind::Plain => MemberKind::Plain,
            ReadKind::IndirectStruct { first_id } => MemberKind::IndirectStruct { first_id },
            ReadKind::Indirect(node) => {
                // A layout the member's own node names is keyed by the member's field, the type
                // its members are fields of, as `read` pushed it.
                let field = read.field;
                let layouts = node.layouts(|id, _| decoded(cx, (id, field)))?;
                let arg = cx.indirect(&node, layouts, read.space)?;
                cx.meter.release(NODE);
                MemberKind::Indirect(arg)
            }
        };
        members.push(Member {
            offset: read.offset,
            size: read.size,
            array_len: read.array_len,
            layout,
            kind,
            render_target: read.render_target,
            raster_order_group: read.raster_order_group,
            type_name: read.type_name,
            name: read.name,
            key: read.key,
        });
    }
    cx.structs.active.remove(&frame.key);
    cx.meter.release(ACTIVE_ENTRY);
    let info = StructInfo { members };
    let mut hasher = DefaultHasher::new();
    info.hash(&mut hasher);
    let hash = hasher.finish();
    if let Some(&same) = cx.structs.by_content.get(&hash) {
        if cx.structs.list.get(same.0 as usize) == Some(&info) {
            // Its nested layouts are the same layouts, so its height is the same.
            cx.meter
                .release(std::mem::size_of::<Member>().saturating_mul(info.members.len()));
            cx.meter.charge(MEMO_ENTRY)?;
            cx.structs.memo.insert(frame.key, same);
            return Ok((same, cx.structs.height(same)));
        }
    }
    let r = u32::try_from(cx.structs.list.len())
        .map(StructRef)
        .map_err(|_| site.err(AbiError::BadValue, key))?;
    // Nested layouts are built before the layouts that hold them, so their flags are known.
    let indirect = info.members.iter().any(|mem| {
        !matches!(mem.kind, MemberKind::Plain)
            || mem.layout.is_some_and(|l| cx.structs.holds_indirect(l))
    });
    cx.meter.reserve(&mut cx.structs.list, 1)?;
    cx.meter.reserve(&mut cx.structs.heights, 1)?;
    cx.meter.reserve(&mut cx.structs.indirect, 1)?;
    cx.structs.list.push(info);
    cx.structs.heights.push(height);
    cx.structs.indirect.push(indirect);
    cx.meter.charge(MEMO_ENTRY)?;
    cx.structs.memo.insert(frame.key, r);
    // A hash shared by two unequal layouts keeps the first: the later one is its own entry.
    if let std::collections::btree_map::Entry::Vacant(slot) = cx.structs.by_content.entry(hash) {
        cx.meter.charge(CONTENT_ENTRY)?;
        slot.insert(r);
    }
    Ok((r, height))
}

/// Read the members of a layout, noting the layout nodes they name and, for each member, what
/// the holding struct's field at its offset says about its address space.
fn read(cx: &mut Cx<'_, '_>, layout: LayoutKey, site: &Site) -> Result<Frame, Error> {
    let m = cx.m;
    let (id, holder) = layout;
    let mut ops = Ops::node(m, id, site.clone())?;
    let mut members = Vec::new();
    let mut nested = Vec::new();
    while !ops.is_done() {
        cx.meter.tick(1)?;
        let mut nested_node = None;
        if matches!(ops.peek(), Some(Tok::Str(_))) {
            let k = key_of(&mut ops)?;
            if k != Key::StructTypeInfo {
                return Err(ops.site.err(AbiError::UnknownKey, Some(k.name())));
            }
            nested_node = Some(ops.child(Some(k.name()))?);
        }
        let key = Some(Key::StructTypeInfo.name());
        let offset = ops.u32(key)?;
        let size = ops.u32(key)?;
        let array_len = ops.u32(key)?;
        // The member's own field of the holding struct: its address space where the member's node
        // states none, and the holding type of the layout it names.
        let space = MemberSpace::of(m, cx.layouts, holder, offset);
        let field = field_holder(m, cx.layouts, holder, offset);
        let layout = nested_node.map(|child| (child, field));
        if let Some(child) = layout {
            cx.meter.reserve(&mut nested, 1)?;
            nested.push((child, site.clone()));
        }
        let type_name = ops.diag(key, cx.meter)?;
        let raw = ops.string(key)?;
        let name = kept_diag(raw, cx.meter)?;
        let user = UserKey(kept_bytes(raw, cx.meter)?);
        let mut member = Read {
            layout,
            space,
            field,
            offset,
            size,
            array_len,
            kind: ReadKind::Plain,
            render_target: None,
            raster_order_group: None,
            type_name,
            name,
            key: user,
        };
        suffixes(cx, &mut ops, &mut member)?;
        if let ReadKind::Indirect(node) = &member.kind {
            for child in node.nested() {
                cx.meter.reserve(&mut nested, 1)?;
                nested.push(((child, field), node.site().clone()));
            }
        }
        cx.meter.reserve(&mut members, 1)?;
        members.push(member);
    }
    Ok(Frame {
        key: layout,
        site: site.clone(),
        members,
        nested,
        done: 0,
        height: 0,
    })
}

/// Read the member suffixes, stopping at the next member.
fn suffixes(cx: &mut Cx<'_, '_>, ops: &mut Ops<'_>, member: &mut Read) -> Result<(), Error> {
    let mut seen = Vec::new();
    loop {
        let Some(Tok::Str(s)) = ops.peek() else {
            return Ok(());
        };
        if Key::parse(s) == Some(Key::StructTypeInfo) {
            return Ok(());
        }
        let k = key_of(ops)?;
        if seen.contains(&k) {
            return Err(ops.site.err(AbiError::BadValue, Some(k.name())));
        }
        cx.meter.reserve(&mut seen, 1)?;
        seen.push(k);
        let name = Some(k.name());
        match k {
            Key::IndirectArgument => {
                member.kind = match ops.next(name)? {
                    Tok::Node(id) => {
                        let node = cx.indirect_node(id, &ops.site)?;
                        cx.meter.charge(NODE)?;
                        ReadKind::Indirect(Box::new(node))
                    }
                    Tok::Int { bits, value } if bits <= 32 => ReadKind::IndirectStruct {
                        first_id: u32::try_from(value)
                            .map_err(|_| ops.site.err(AbiError::BadValue, name))?,
                    },
                    _ => return Err(ops.site.err(AbiError::BadValue, name)),
                }
            }
            Key::RenderTarget => member.render_target = Some(ops.u32(name)?),
            Key::RasterOrderGroup => member.raster_order_group = Some(ops.u32(name)?),
            _ => return Err(ops.site.err(AbiError::UnknownKey, name)),
        }
    }
}
