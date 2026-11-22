//! Content-digest nodes of encoded types: what moves a digest and what does not.

use std::sync::Arc;

use super::*;
use crate::cache::Hash256;

/// One shape of a value, and the same shape after a field was added at the end, inside a sequence
/// item, and before the others.
#[derive(Clone, Debug, Default, PartialEq)]
struct Before {
    count: u32,
    items: Vec<Item>,
    name: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
struct After {
    added: Option<u32>,
    count: u32,
    items: Vec<ItemAfter>,
    name: String,
    flag: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
struct Item {
    word: u64,
}

#[derive(Clone, Debug, Default, PartialEq)]
struct ItemAfter {
    word: u64,
    extra: Vec<u32>,
}

/// `Before`'s fields declared in another order.
#[derive(Clone, Debug, Default, PartialEq)]
struct Reordered {
    name: String,
    items: Vec<Item>,
    count: u32,
}

codec_struct! {
    Before { count, items, name }
    After { added, count, items, name, flag }
    Item { word }
    ItemAfter { word, extra }
    Reordered { name, items, count }
}

/// An enum, the same variants reordered, and with one inserted.
#[derive(Clone, Debug, PartialEq)]
enum Mode {
    Off,
    Level(u32),
    Window { start: u32, len: u32 },
}

#[derive(Clone, Debug, PartialEq)]
enum ModeReordered {
    Window { start: u32, len: u32 },
    Level(u32),
    Off,
}

#[derive(Clone, Debug, PartialEq)]
enum ModeInserted {
    Off,
    Auto,
    Level(u32),
    Window { start: u32, len: u32 },
}

codec_enum! {
    Mode { Off, Level(f0), Window { start, len } }
    ModeReordered { Window { start, len }, Level(f0), Off }
    ModeInserted { Off, Auto, Level(f0), Window { start, len } }
}

fn before() -> Before {
    Before {
        count: 3,
        items: vec![Item { word: 7 }, Item { word: 0 }],
        name: "main".into(),
    }
}

fn after(before: &Before) -> After {
    After {
        added: None,
        count: before.count,
        items: before
            .items
            .iter()
            .map(|item| ItemAfter {
                word: item.word,
                extra: Vec::new(),
            })
            .collect(),
        name: before.name.clone(),
        flag: false,
    }
}

/// A field added with its zero value, anywhere in the value, moves nothing; any non-zero value of
/// it moves the digest.
#[test]
fn an_added_field_at_zero_keeps_the_digest() {
    let old = before();
    let new = after(&old);
    assert_eq!(old.digest(), new.digest());
    assert!(!old.digest().is_zero());

    let mut added = new.clone();
    added.added = Some(0);
    assert_ne!(added.digest(), new.digest(), "Some(0) is not None");
    let mut flagged = new.clone();
    flagged.flag = true;
    assert_ne!(flagged.digest(), new.digest());
    let mut extra = new.clone();
    extra.items[1].extra.push(0);
    assert_ne!(
        extra.digest(),
        new.digest(),
        "a non-empty sequence of zeros"
    );
}

/// Field declaration order does not enter the digest; field values and names do.
#[test]
fn fields_are_named_not_numbered() {
    let old = before();
    let reordered = Reordered {
        name: old.name.clone(),
        items: old.items.clone(),
        count: old.count,
    };
    assert_eq!(old.digest(), reordered.digest());

    let mut changed = old.clone();
    changed.count = 4;
    assert_ne!(changed.digest(), old.digest());
    let mut changed = old.clone();
    changed.items[1].word = 1;
    assert_ne!(changed.digest(), old.digest());
    let mut changed = old.clone();
    changed.items.pop();
    assert_ne!(changed.digest(), old.digest());

    // Two fields of one type: the names keep a value in one apart from the same value in the
    // other.
    let a = Window::Bounded { start: 5, len: 0 };
    let b = Window::Bounded { start: 0, len: 5 };
    assert_ne!(a.digest(), b.digest());
}

/// A struct whose fields are all zero is zero, so a struct field added at its default is omitted
/// too; a value of all zeros is zero only when its encoding is.
#[test]
fn zero_is_the_all_zero_encoding() {
    assert!(Before::default().digest().is_zero());
    assert!(After::default().digest().is_zero());
    assert!(Item::default().digest().is_zero());
    assert!(0u32.digest().is_zero());
    assert!(false.digest().is_zero());
    assert!(None::<u32>.digest().is_zero());
    assert!(Vec::<Item>::new().digest().is_zero());
    assert!(String::new().digest().is_zero());
    assert!(0f32.digest().is_zero());
    assert!(!(-0f32).digest().is_zero());
    assert!(!vec![0u32].digest().is_zero());
    assert!(!vec![Item::default()].digest().is_zero());
    assert!(!Some(Item::default()).digest().is_zero());
    assert!(!Mode::Off.digest().is_zero(), "an enum value is never zero");
}

/// Variants are named: reordering or inserting variants moves nothing, and switching a value to
/// another variant does, also to one whose fields encode the same bytes.
#[test]
fn variants_are_named_not_numbered() {
    let cases = [
        (Mode::Off, ModeReordered::Off, ModeInserted::Off),
        (
            Mode::Level(9),
            ModeReordered::Level(9),
            ModeInserted::Level(9),
        ),
        (
            Mode::Window { start: 1, len: 2 },
            ModeReordered::Window { start: 1, len: 2 },
            ModeInserted::Window { start: 1, len: 2 },
        ),
    ];
    for (mode, reordered, inserted) in &cases {
        assert_eq!(mode.digest(), reordered.digest(), "{mode:?}");
        assert_eq!(mode.digest(), inserted.digest(), "{mode:?}");
    }
    let digests = [
        Mode::Off.digest(),
        Mode::Level(0).digest(),
        Mode::Level(1).digest(),
        Mode::Window { start: 0, len: 0 }.digest(),
        Mode::Window { start: 1, len: 0 }.digest(),
        ModeInserted::Auto.digest(),
    ];
    for (i, a) in digests.iter().enumerate() {
        for b in &digests[i + 1..] {
            assert_ne!(a, b);
        }
    }
    // The same field bytes under another variant name.
    assert_ne!(BufferForm::Descriptor.digest(), BufferForm::Psb.digest());
    assert_ne!(Stage::Vertex.digest(), Stage::Fragment.digest());
}

/// A sequence or option of leaves is the leaf of its encoding (not one hash per item), streamed in
/// chunks as `encode_chunked` writes it; a sequence of structs is the sequence of their nodes, and
/// a map the sequence of its entries.
#[test]
fn leaves_are_their_whole_encoding() {
    let words: Arc<[u32]> = (0..(3 * CHUNK as u32)).collect();
    let whole = encode(&words);
    assert_eq!(words.digest(), Node::leaf(|sink| sink(&whole)));
    assert_eq!(
        Some(7u64).digest(),
        Node::leaf(|sink| sink(&encode(&Some(7u64))))
    );
    let pair = (3u32, vec![1u8, 2]);
    assert_eq!(pair.digest(), Node::leaf(|sink| sink(&encode(&pair))));

    let items = vec![Item { word: 1 }, Item { word: 2 }];
    assert_eq!(
        items.digest(),
        Node::sequence(items.iter().map(Codec::digest))
    );
    assert_ne!(items.digest(), Node::leaf(|sink| sink(&encode(&items))));

    let map: BTreeMap<u32, Item> = [(1, Item { word: 0 }), (2, Item { word: 5 })].into();
    assert_eq!(
        map.digest(),
        Node::sequence(
            map.iter()
                .map(|(k, v)| Node::tuple([k.digest(), v.digest()]))
        )
    );
    let mut moved = map.clone();
    moved.insert(3, Item::default());
    assert_ne!(moved.digest(), map.digest());
}

/// The digest's definition is pinned: a change to how nodes hash moves every recorded digest, so
/// it is never accidental.
#[test]
fn the_digest_definition_is_pinned() {
    let hex = |digest: Hash256| {
        digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    };
    let value = (
        before(),
        vec![Mode::Off, Mode::Level(2), Mode::Window { start: 1, len: 0 }],
    );
    assert_eq!(
        hex(value.digest().root("pinned")),
        "affc438d6bb30724408d763b8b905d9bcf6f7bda55c7535dcee01eb275b05a7b",
        "the content digest's definition changed"
    );
}

/// A sequence of integers decodes, taken at once, to what reading its items one at a time gives,
/// and a sequence its bytes cut short is `Malformed`.
#[test]
fn integer_sequences_decode_item_for_item() {
    fn one_at_a_time<T: Codec>(bytes: &[u8]) -> Vec<T> {
        let mut r = Reader::new(bytes);
        let n = u32::get(&mut r).expect("count") as usize;
        let items = (0..n).map(|_| T::get(&mut r).expect("item")).collect();
        assert_eq!(r.remaining(), 0);
        items
    }
    fn check<T: Codec + Clone + PartialEq + std::fmt::Debug>(items: Vec<T>) {
        let bytes = encode(&items);
        assert_eq!(decode::<Vec<T>>(&bytes), Ok(items.clone()));
        assert_eq!(one_at_a_time::<T>(&bytes), items);
        let shared: Arc<[T]> = Arc::from(items.clone());
        assert_eq!(decode::<Arc<[T]>>(&encode(&shared)), Ok(shared));
        if !items.is_empty() {
            let short = &bytes[..bytes.len() - 1];
            assert_eq!(decode::<Vec<T>>(short), Err(Malformed));
            assert_eq!(decode::<Arc<[T]>>(short), Err(Malformed));
        }
    }
    check::<u16>(vec![0, 1, 0x8000, u16::MAX]);
    check::<u32>(
        (0..1000)
            .map(|i: u32| i.wrapping_mul(0x9e37_79b9))
            .collect(),
    );
    check::<u32>(Vec::new());
    check::<u64>(vec![u64::MAX, 0x0123_4567_89ab_cdef, 7]);
    check::<i32>(vec![i32::MIN, -1, 0, i32::MAX]);
    check::<i64>(vec![i64::MIN, -2, i64::MAX]);
}
