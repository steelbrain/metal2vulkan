"""The call graph of the Metal leg, over sources whose answer is known by construction.

Run by `coverage.py --selftest`, with the option-space axis's own tests.

The reachability graph is what attributes a Metal call to a case module: an item is credited to the
families whose `Side` methods reach its call site, so an edge the walk cannot see makes a call the
suite really does make read as called by nobody. That is how the two
`MTLRasterizationRateLayerDescriptor` items the rate-map cases exercise came to sit in the
document's *attributed to no family* table: `Side::rate_map` builds each layer's descriptor through
`.map(layer_descriptor)`, and every edge the walk read needed a `(` after the name.

So both directions are held here, because the narrowing is the whole fix: a function handed to a
combinator **is** an edge, and a local variable that happens to share a name with a function is
**not**. Every test owns its inputs; nothing here reads the repository.
"""

import tempfile
import unittest
from pathlib import Path

import metal_leg

#: Every tree a test made, removed when the module is done with them.
MADE: list[tempfile.TemporaryDirectory] = []

#: The smallest `objc2-metal` map a leg can be read against: one class, one selector on it.
API = (
    "# type\tkind\tsupertypes\n"
    "MTLThing\tclass\tNSObject\n"
    "# type\tselector\trust\tanswers\n"
    "MTLThing\tmakeIt:\tmakeIt\t\n"
)


def tearDownModule() -> None:
    """Remove the trees the tests built."""
    while MADE:
        MADE.pop().cleanup()


def leg_of(source: str) -> metal_leg.Leg:
    """Read a leg whose one file holds `source`."""
    held = tempfile.TemporaryDirectory()
    MADE.append(held)
    root = Path(held.name)
    (root / "objc2_metal.tsv").write_text(API)
    (root / "leg.rs").write_text(source)
    api = metal_leg.Api.read(root / "objc2_metal.tsv")
    return metal_leg.read_leg(api, root, ["leg.rs"], set())


class HigherOrderEdges(unittest.TestCase):
    """A function of the leg handed to a combinator is an edge of the graph."""

    def test_a_function_passed_to_map_is_an_edge(self):
        """`.map(f)` reaches `f`, with no call of `f` written anywhere.

        This is `Side::rate_map`'s shape, reduced: the only mention of the callee is its name as a
        value. Before the walk read it, the Metal calls inside such a function were credited to no
        case module at all.
        """
        leg = leg_of(
            """
            fn outer(things: &[u32]) -> Vec<u32> {
                things.iter().map(inner).collect()
            }
            fn inner(one: &u32) -> u32 {
                MTLThing::makeIt(*one)
            }
            """
        )
        self.assertIn("inner", leg.calls_function.get("outer", set()))

    def test_every_combinator_the_list_names_is_an_edge(self):
        """Each combinator of `HIGHER_ORDER`, held one at a time.

        A list in a regex is the shape that silently loses an alternative to a typo, and the cost
        of losing one is a hole exactly like the one this closed, so each is checked rather than
        the list being read as correct.
        """
        for combinator in metal_leg.HIGHER_ORDER.split("|"):
            leg = leg_of(
                f"""
                fn outer(thing: &Held) -{{}} u32 {{
                    thing.{combinator}(inner)
                }}
                fn inner(one: &u32) -> u32 {{
                    MTLThing::makeIt(*one)
                }}
                """.replace("-{}", "-")
            )
            self.assertIn(
                "inner",
                leg.calls_function.get("outer", set()),
                f".{combinator}(inner) is an edge",
            )

    def test_a_local_that_shares_a_functions_name_is_not_an_edge(self):
        """`.map(handle)` where `handle` is a local, not the leg's `handle` function.

        The direction that keeps the reading honest. A walk that read any bare identifier sharing
        a name with a function would add 187 such edges over this leg alone, nearly all of them a
        local named `handle`, `texture` or `buffer`, and the graph would say every `Side` method
        reaches every call site.
        """
        leg = leg_of(
            """
            fn outer(thing: &Held) -> Option<u32> {
                let handle = 3u32;
                thing.held().map(handle)
            }
            fn unrelated(one: u32) -> u32 {
                MTLThing::makeIt(one)
            }
            """
        )
        self.assertNotIn("handle", leg.calls_function.get("outer", set()))

    def test_a_type_qualified_path_is_not_an_edge(self):
        """`.map(MTLThing::makeIt)` is a message to a class, not a function of the leg.

        The same exclusion the other loops make: a path whose last qualifier is a type is Metal's
        own, and reading it as one of ours would credit the call to a function that does not exist.
        """
        leg = leg_of(
            """
            fn outer(thing: &Held) -> Option<u32> {
                thing.held().map(MTLThing::makeIt)
            }
            fn makeIt(one: u32) -> u32 {
                one
            }
            """
        )
        self.assertNotIn("makeIt", leg.calls_function.get("outer", set()))
