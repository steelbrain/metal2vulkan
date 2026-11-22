//! The content digest's nodes, and the same rules through `serde` for the types the cache does not
//! encode.

use super::*;

fn node(byte: u8) -> Node {
    Node::leaf(|sink| sink(&[byte]))
}

/// Zero nodes are omitted from structs and keep tuples and sequences apart by position and count;
/// roots of different kinds differ.
#[test]
fn nodes_omit_only_struct_fields() {
    assert!(Node::leaf(|sink| sink(&[0, 0, 0])).is_zero());
    assert!(Node::leaf(|_| {}).is_zero());
    assert!(!node(1).is_zero());
    // A leaf handed in pieces is the leaf of the whole.
    assert_eq!(
        Node::leaf(|sink| {
            sink(&[1, 2]);
            sink(&[]);
            sink(&[3]);
        }),
        Node::leaf(|sink| sink(&[1, 2, 3]))
    );

    assert_eq!(
        Node::fields(&mut [("a", node(1)), ("b", Node::ZERO)]),
        Node::fields(&mut [("a", node(1))])
    );
    assert_eq!(
        Node::fields(&mut [("b", node(2)), ("a", node(1))]),
        Node::fields(&mut [("a", node(1)), ("b", node(2))])
    );
    assert_ne!(
        Node::fields(&mut [("a", node(1))]),
        Node::fields(&mut [("b", node(1))])
    );
    assert!(Node::fields(&mut [("a", Node::ZERO)]).is_zero());
    assert!(Node::fields(&mut []).is_zero());

    assert!(!Node::variant("A", &mut []).is_zero());
    assert_ne!(Node::variant("A", &mut []), Node::variant("B", &mut []));
    assert_eq!(
        Node::variant("A", &mut [("x", Node::ZERO)]),
        Node::variant("A", &mut [])
    );

    assert!(Node::sequence([]).is_zero());
    assert!(!Node::sequence([Node::ZERO]).is_zero());
    assert_ne!(
        Node::sequence([Node::ZERO]),
        Node::sequence([Node::ZERO, Node::ZERO])
    );
    assert_ne!(
        Node::sequence([node(1), Node::ZERO]),
        Node::sequence([Node::ZERO, node(1)])
    );

    assert!(Node::tuple([Node::ZERO, Node::ZERO]).is_zero());
    assert_ne!(
        Node::tuple([node(1), Node::ZERO]),
        Node::tuple([Node::ZERO, node(1)])
    );
    assert!(!Node::some(Node::ZERO).is_zero());
    assert_ne!(Node::some(node(1)), node(1));

    assert_ne!(node(1).root("a"), node(1).root("b"));
    assert_ne!(Node::ZERO.root("a"), node(1).root("a"));
}
