"""The option-space axis over inputs whose answers are known.

Run by `coverage.py --selftest`, which is what the suite's
`the_option_space_axis_still_measures_what_it_claims` runs.

The axis is a derivation over text, and the way a derivation like it fails is by quietly matching
nothing: a renamed JSON key, a spelling the sources use that the regex does not, and every count
reads zero while the document still looks like a measurement. So each test here builds its own
little tree, where the answer is known by construction, and asserts both directions — that a value
it should find is found, and that a value it must not count is not.

Every test owns its inputs. Nothing here reads the repository.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import option_space

#: Every tree `tree` made, removed by `tearDownModule`. Each test wants a fresh one and none of
#: them wants it afterwards, so one list and one sweep is the whole lifetime.
MADE: list[Path] = []


def tearDownModule() -> None:
    """Remove the trees the tests built."""
    while MADE:
        shutil.rmtree(MADE.pop(), ignore_errors=True)

#: A generated table of one enum and one option set, as `regen-metal-enums` writes them.
GENERATED = """\
// Generated from the macOS 27.0 SDK Metal.framework headers (names and values).

metal_enum! {
    /// `MTLThing`.
    #[non_exhaustive]
    Thing = "MTLThing" {
        Alpha = 0, "MTLThingAlpha";
        Beta = 1, "MTLThingBeta";
        Gamma = 2, "MTLThingGamma";
        Delta = 3, "MTLThingDelta";
        Epsilon = 4, "MTLThingEpsilon";
        Zeta = 5, "MTLThingZeta";
    }
    aliases {
        "MTLThingOldBeta" => Beta;
    }
}

metal_flags! {
    /// `MTLSpread`.
    Spread = "MTLSpread" {
        NONE = 0, "MTLSpreadNone";
        LEFT = 0x1, "MTLSpreadLeft";
        RIGHT = 0x2, "MTLSpreadRight";
    }
    aliases {
    }
    dropped {
    }
}
"""


#: A generated enumerator table, as `regen-metal-enums` writes `TABLE_OUTPUT`: the SDK's values
#: for a type the tree models by hand, with one deprecated spelling. `MTLKept` is declared by no
#: generated Rust here, exactly as `MTLStoreAction` and `MTLPipelineOption` are declared by none.
TABLE = """\
# Generated from the macOS 27.0 SDK Metal.framework headers (names and values).
type\tkind\trole\trust\tenumerator\tvalue
MTLKept\tenum\tvalue\tFirst\tMTLKeptFirst\t0
MTLKept\tenum\tvalue\tSecond\tMTLKeptSecond\t1
MTLKept\tenum\tvalue\tThird\tMTLKeptThird\t2
MTLKept\tenum\talias\tSecond\tMTLKeptOldSecond\t1
"""

#: A hand-written declaration of `MTLKept`, whose variants are spelled unlike the table's derived
#: names, which is `MTLBlitOption`'s shape: the tree models an option set as a closed enum.
BY_HAND_KEPT = """\
/// `MTLKept`, which this tree declares itself.
pub enum Kept {
    First,
    Second,
    Third,
}
"""


def tree(
    cases: dict[str, str],
    shared: str = "",
    recordings: tuple[dict, ...] = (),
    generated: str = GENERATED,
    table: str = TABLE,
    by_hand: str = "",
    unreachable: str = "",
    crossings: str = "",
) -> Path:
    """A repository-shaped tree of exactly these sources and recordings."""
    root = Path(tempfile.mkdtemp(prefix="option-space-selftest-"))
    MADE.append(root)
    enumerators = root / option_space.ENUMERATORS
    enumerators.parent.mkdir(parents=True, exist_ok=True)
    enumerators.write_text(table)
    for name in option_space.GENERATED:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        # The front end's file holds a different type in the tree; here one copy is enough, and
        # `read_types` keeping the first reading is what this relies on.
        path.write_text(generated)
    for name in option_space.BY_HAND:
        (root / name).mkdir(parents=True, exist_ok=True)
    if by_hand:
        (root / option_space.BY_HAND[0] / "kept.rs").write_text(by_hand)
    suite = root / "crates/metal2vulkan-conformance"
    (suite / "src/cases").mkdir(parents=True, exist_ok=True)
    (suite / "recordings").mkdir(parents=True, exist_ok=True)
    for module, text in cases.items():
        (suite / "src/cases" / f"{module}.rs").write_text(text)
    (suite / "src/side.rs").write_text(shared)
    if unreachable:
        declared = root / option_space.UNREACHABLE
        declared.parent.mkdir(parents=True, exist_ok=True)
        declared.write_text(unreachable)
    if crossings:
        crossed = root / option_space.CROSSINGS
        crossed.parent.mkdir(parents=True, exist_ok=True)
        crossed.write_text(crossings)
    for data in recordings:
        (suite / "recordings" / f"{data['case']}.json").write_text(json.dumps(data))
    return root


def recording(case: str, value: str, provenance: str = option_space.RAN, where: str = "ran") -> dict:
    """A recording of `case` whose transcript, or whose validated leg, names `value`."""
    node = {
        "name": "case",
        "observed": {"value": {"fields": {"factor": {"fields": {"name": {"lines": [value]}}}}}},
    }
    answer = {"provenance": provenance, "runs": 1, "outcomes": []}
    if where == "ran":
        answer["outcomes"] = [{"runs": 1, "transcript": {"root": node}}]
    else:
        answer["outcomes"] = [{"runs": 1, "transcript": {"root": {"name": "case"}}}]
        answer["validated"] = {"stopped": {"message": f"Thing Validation\n{value} is not allowed"}}
    return {"format": 1, "case": case, "suite_version": 1, "answers": [answer]}


def runs(ran: set[str]) -> dict[str, str]:
    """One recorded case a module, which is the case-level form of "this module ran Metal".

    The axis takes the cases, because a crossing is declared at one of them; a test about the
    per-value columns states the modules, which is all those columns read.
    """
    return {f"{module}-case": module for module in ran}


def axis(root: Path, ran: set[str]):
    """The axis over `root`, as the one row of `MTLThing`."""
    held = option_space.inventory(root, runs(ran))
    return next(row for row in held["rows"] if row["metal"] == "MTLThing")


class Denominator(unittest.TestCase):
    """The value list comes from the generated tables, and a parse of nothing is a failure."""

    def test_the_values_are_the_generated_table_s(self):
        row = axis(tree({}), set())
        self.assertEqual(len(row["values"]), 6)
        self.assertEqual(row["values"][0], "MTLThingAlpha")
        self.assertEqual(row["compared"], [])
        self.assertEqual(row["missing"], row["values"])

    def test_an_option_set_s_named_combinations_are_values(self):
        held = option_space.inventory(tree({}), {})
        row = next(r for r in held["rows"] if r["metal"] == "MTLSpread")
        self.assertEqual(row["values"], ["MTLSpreadNone", "MTLSpreadLeft", "MTLSpreadRight"])

    def test_a_generated_file_that_parses_to_nothing_fails(self):
        with self.assertRaises(SystemExit):
            option_space.read_types(tree({}, generated="// nothing at all\n"))

    def test_a_type_that_parses_to_no_values_fails(self):
        broken = GENERATED.replace('Alpha = 0, "MTLThingAlpha";', "").replace(
            'Beta = 1, "MTLThingBeta";', ""
        )
        broken = broken.replace('Gamma = 2, "MTLThingGamma";', "")
        broken = broken.replace('Delta = 3, "MTLThingDelta";', "")
        broken = broken.replace('Epsilon = 4, "MTLThingEpsilon";', "")
        broken = broken.replace('Zeta = 5, "MTLThingZeta";', "")
        with self.assertRaises(SystemExit):
            option_space.read_types(tree({}, generated=broken))


class InEffect(unittest.TestCase):
    """A value a recorded case module states is in effect; one it does not is not."""

    def test_a_value_a_case_module_names_is_in_effect(self):
        row = axis(tree({"blends": "fn go() { let x = Thing::Beta; }"}), {"blends"})
        self.assertEqual(row["in_effect"], ["MTLThingBeta"])
        self.assertIn("MTLThingBeta", row["compared"])
        self.assertNotIn("MTLThingBeta", row["missing"])

    def test_an_aliased_spelling_is_in_effect(self):
        # `use Thing as T;` and then `T::Gamma`, which is how the blend sweeps are written. A
        # derivation reading only `Thing::` spellings reports this value as never exercised.
        row = axis(tree({"blends": "use Thing as T;\nfn go() { let x = T::Gamma; }"}), {"blends"})
        self.assertEqual(row["in_effect"], ["MTLThingGamma"])

    def test_the_objc2_spelling_is_in_effect(self):
        # `MTLThing::Delta`, which is how the Metal leg names a fixed value: `objc2` drops the
        # type's prefix, so the enumerator is the two spelled back together.
        row = axis(tree({"blends": "fn go() { d.set(MTLThing::Delta); }"}), {"blends"})
        self.assertEqual(row["in_effect"], ["MTLThingDelta"])

    def test_an_objc2_spelling_of_no_generated_name_is_not_counted(self):
        row = axis(tree({"blends": "fn go() { d.set(MTLThing::Omega); }"}), {"blends"})
        self.assertEqual(row["in_effect"], [])

    def test_a_module_with_no_metal_run_recording_puts_nothing_in_effect(self):
        text = "fn go() { let x = Thing::Beta; }"
        self.assertEqual(axis(tree({"blends": text}), set())["in_effect"], [])
        self.assertEqual(axis(tree({"blends": text}), {"blends"})["in_effect"], ["MTLThingBeta"])

    def test_a_cfg_test_item_is_not_in_effect(self):
        text = (
            "fn go() { let x = Thing::Beta; }\n"
            "#[cfg(test)]\nmod tests {\n"
            "    #[test]\n    fn refused() { assert!(!matches!(f, Thing::Zeta)); }\n"
            "}\n"
        )
        row = axis(tree({"blends": text}), {"blends"})
        self.assertEqual(row["in_effect"], ["MTLThingBeta"])
        self.assertIn("MTLThingZeta", row["missing"])

    def test_a_cfg_test_item_with_no_body_is_blanked_to_its_semicolon(self):
        text = (
            "#[cfg(test)]\nconst WIDTHS: [u8; 4] = [Thing::Zeta as u8; 4];\n"
            "fn go() { let x = Thing::Beta; }\n"
        )
        row = axis(tree({"blends": text}), {"blends"})
        self.assertEqual(row["in_effect"], ["MTLThingBeta"])

    def test_a_sweep_by_ordinal_states_nothing(self):
        # The boundary of this column, kept visible: a case that hands the leg `0..n` reaches every
        # value and names none, so the column reads zero and the document says it is a floor by
        # exactly this. There is no sound textual reading of a computed ordinal, so the test holds
        # the limit rather than a fix for it.
        text = "fn go() { for raw in 0..6 { side.set_thing(raw); } }"
        self.assertEqual(axis(tree({"sweep": text}), {"sweep"})["in_effect"], [])

    def test_a_comment_naming_a_value_is_not_in_effect(self):
        row = axis(tree({"blends": "// Thing::Beta is what Metal defaults to.\nfn go() {}"}), {"blends"})
        self.assertEqual(row["in_effect"], [])


class Answered(unittest.TestCase):
    """A value a recording's compared transcript holds is answered; the other legs are not read."""

    def test_a_value_in_a_run_transcript_is_answered(self):
        root = tree({}, recordings=(recording("one", "MTLThingGamma"),))
        row = axis(root, set())
        self.assertEqual(row["answered"], ["MTLThingGamma"])
        self.assertIn("MTLThingGamma", row["compared"])

    def test_a_carried_answer_is_not_answered(self):
        root = tree({}, recordings=(recording("one", "MTLThingGamma", provenance="carried"),))
        row = axis(root, set())
        self.assertEqual(row["answered"], [])
        self.assertIn("MTLThingGamma", row["missing"])

    def test_the_validated_leg_is_not_answered(self):
        root = tree({}, recordings=(recording("one", "MTLThingGamma", where="validated"),))
        row = axis(root, set())
        self.assertEqual(row["answered"], [])

    def test_base64_bytes_are_not_read_as_names(self):
        # A recorded buffer is base64, and the alphabet is alphanumeric, so a payload **can** spell
        # an enumerator name exactly: `MTLThingZeta` is twelve characters of the base64 alphabet,
        # which is to say nine bytes of some texture. Reading it would credit the suite with a
        # value from the contents of a buffer.
        data = recording("one", "MTLThingGamma")
        data["answers"][0]["outcomes"][0]["transcript"]["root"]["observed"] = {
            "value": {"bytes": "MTLThingZeta"}
        }
        row = axis(tree({}, recordings=(data,)), set())
        self.assertEqual(row["answered"], [])

    def test_a_compound_site_name_is_split_into_its_tokens(self):
        # The tokenising is what this is about, and it still holds. The column is `at_a_site` and
        # not `answered`: a site's name is the case's label, so a value found there is in the
        # compared text without Metal having named it back. See `AtASite`.
        data = recording("one", "MTLThingGamma")
        data["answers"][0]["outcomes"][0]["transcript"]["root"] = {
            "name": "MTLThingZeta/TextureBuffer/64x1x1",
            "observed": {"value": "absent"},
        }
        row = axis(tree({}, recordings=(data,)), set())
        self.assertEqual(row["at_a_site"], ["MTLThingZeta"])
        self.assertEqual(row["answered"], [])
        self.assertIn("MTLThingZeta", row["compared"])


class SharedOnly(unittest.TestCase):
    """A value only the suite's shared sources name is its own column, not a gap."""

    def test_a_value_only_shared_sources_name_is_neither_compared_nor_missing(self):
        row = axis(tree({}, shared="fn go() { d.set(Thing::Delta); }"), set())
        self.assertEqual(row["shared"], ["MTLThingDelta"])
        self.assertNotIn("MTLThingDelta", row["compared"])
        self.assertNotIn("MTLThingDelta", row["missing"])

    def test_a_case_module_naming_it_takes_it_out_of_shared(self):
        root = tree(
            {"blends": "fn go() { let x = Thing::Delta; }"},
            shared="fn go() { d.set(Thing::Delta); }",
        )
        row = axis(root, {"blends"})
        self.assertEqual(row["shared"], [])
        self.assertEqual(row["in_effect"], ["MTLThingDelta"])

    def test_every_value_is_in_exactly_one_column(self):
        root = tree(
            {"blends": "use Thing as T;\nfn go() { let x = T::Beta; }"},
            shared="fn go() { d.set(Thing::Delta); }",
            recordings=(recording("one", "MTLThingGamma"),),
        )
        row = axis(root, {"blends"})
        self.assertEqual(
            len(row["compared"]) + len(row["shared"]) + len(row["missing"]), len(row["values"])
        )
        self.assertEqual(sorted(row["compared"]), ["MTLThingBeta", "MTLThingGamma"])
        self.assertEqual(row["shared"], ["MTLThingDelta"])
        self.assertEqual(row["missing"], ["MTLThingAlpha", "MTLThingEpsilon", "MTLThingZeta"])


class ByHand(unittest.TestCase):
    """A Metal type the tree models by hand is reported apart, with no denominator."""

    SOURCE = """\
/// `MTLThingy`.
pub enum Thingy {
    /// `MTLThingyFirst`.
    First,
    /// `MTLThingySecond`.
    Second,
}

impl Thingy {
    pub fn from_raw(raw: u64) -> Result<Thingy, Error> {
        Ok(match raw {
            0 => Thingy::First,
            1 => Thingy::Second,
            raw => return Err(undefined("MTLThingy", raw)),
        })
    }
}
"""

    def held(self, cases: dict[str, str], ran: set[str]):
        root = tree(cases)
        (root / option_space.BY_HAND[0] / "values.rs").write_text(self.SOURCE)
        found = option_space.inventory(root, runs(ran))
        return next(row for row in found["by_hand"] if row["metal"] == "MTLThingy")

    def test_a_hand_modelled_type_is_found_with_its_own_variants(self):
        row = self.held({}, set())
        self.assertEqual(row["rust"], "Thingy")
        self.assertEqual(row["declared"], ["First", "Second"])
        self.assertEqual(row["stated"], [])

    def test_a_recorded_case_naming_one_states_it(self):
        row = self.held({"blends": "fn go() { let x = Thingy::Second; }"}, {"blends"})
        self.assertEqual(row["stated"], ["Second"])

    def test_the_objc2_spelling_of_a_hand_modelled_type_states_it(self):
        row = self.held({"blends": "fn go() { c.setThing(MTLThingy::First); }"}, {"blends"})
        self.assertEqual(row["stated"], ["First"])

    def test_an_option_set_newtype_is_listed_with_no_variants(self):
        root = tree({})
        (root / option_space.BY_HAND[0] / "values.rs").write_text(
            "/// `MTLThingOptions`.\n#[repr(transparent)]\npub struct ThingOptions(u64);\n"
        )
        found = option_space.inventory(root, {})
        row = next(r for r in found["by_hand"] if r["metal"] == "MTLThingOptions")
        self.assertEqual((row["rust"], row["declared"]), ("ThingOptions", []))

    def test_a_metal_structure_is_not_an_option_space(self):
        # `MTLOrigin` is a structure of three numbers, not an enum; it has no values to count.
        root = tree({})
        (root / option_space.BY_HAND[0] / "values.rs").write_text(
            "/// `MTLWhere`.\npub struct Where {\n    pub x: u64,\n    pub y: u64,\n}\n"
        )
        self.assertEqual(option_space.inventory(root, {})["by_hand"], [])

    def test_a_metal_name_part_way_through_a_doc_is_prose(self):
        # A type that says it models part of a Metal type is describing itself, not declaring that
        # it is that type's value list.
        root = tree({})
        (root / option_space.BY_HAND[0] / "values.rs").write_text(
            "/// What a buffer allows (Metal's\n"
            "/// `MTLThingy`, as far as the format needs it): the kinds it takes.\n"
            "pub enum Shape {\n    First,\n    Second,\n}\n"
        )
        self.assertEqual(option_space.inventory(root, {})["by_hand"], [])

    def test_a_generated_type_is_not_listed_as_hand_modelled(self):
        root = tree({})
        (root / option_space.BY_HAND[0] / "values.rs").write_text(
            self.SOURCE.replace("MTLThingy", "MTLThing")
        )
        found = option_space.inventory(root, {})
        self.assertEqual([row["metal"] for row in found["by_hand"]], [])


def site_named(case: str, value: str) -> dict:
    """A recording of `case` whose transcript holds `value` as a site's **name**, observing nothing.

    The shape 105 values of `MTLPixelFormat` are in across the committed recordings: a case labels
    the site it asks about with the enumerator, and the site's own answer is a number.
    """
    node = {
        "name": "case",
        "children": [{"name": value, "observed": {"value": {"uint": 7}}}],
    }
    return {
        "format": 1,
        "case": case,
        "suite_version": 1,
        "answers": [
            {
                "provenance": option_space.RAN,
                "runs": 1,
                "outcomes": [{"runs": 1, "transcript": {"root": node}}],
            }
        ],
    }


class AtASite(unittest.TestCase):
    """A value in a site's **name** is not Metal naming it back, and was counted as if it were.

    The comparison pairs sites by name, so such a value is in the compared text and the case did
    put it to Metal. But the name is the case's own label, so it is no evidence Metal said the
    value -- and folding it into *answered* made that column read 172 where Metal's own
    observations held 67.
    """

    def test_a_value_only_in_a_site_name_is_at_a_site_and_not_answered(self):
        root = tree({"blends": "fn go() {}\n"}, recordings=(site_named("b-one", "MTLThingBeta"),))
        row = axis(root, {"blends"})
        self.assertEqual(row["at_a_site"], ["MTLThingBeta"])
        self.assertEqual(row["answered"], [])
        # Still compared: the value is in the text the comparison reads.
        self.assertEqual(row["compared"], ["MTLThingBeta"])
        self.assertEqual(row["missing"], [n for n in row["values"] if n != "MTLThingBeta"])

    def test_a_value_a_site_observed_is_answered(self):
        root = tree({"blends": "fn go() {}\n"}, recordings=(recording("b-one", "MTLThingBeta"),))
        row = axis(root, {"blends"})
        self.assertEqual(row["answered"], ["MTLThingBeta"])
        self.assertEqual(row["at_a_site"], [])

    def test_a_value_both_observed_and_named_is_answered_only(self):
        # The two columns never hold one value twice: what Metal observed is answered, whatever
        # else spells it.
        root = tree(
            {"blends": "fn go() {}\n"},
            recordings=(
                recording("b-one", "MTLThingBeta"),
                site_named("b-two", "MTLThingBeta"),
            ),
        )
        row = axis(root, {"blends"})
        self.assertEqual(row["answered"], ["MTLThingBeta"])
        self.assertEqual(row["at_a_site"], [])

    def test_a_carried_answers_site_name_is_not_read(self):
        # The same gate the rest of `answered` applies: a carried answer ran no Metal.
        held = site_named("b-one", "MTLThingBeta")
        held["answers"][0]["provenance"] = "carried"
        root = tree({"blends": "fn go() {}\n"}, recordings=(held,))
        row = axis(root, {"blends"})
        self.assertEqual(row["at_a_site"], [])
        self.assertEqual(row["answered"], [])


class BracedUse(unittest.TestCase):
    """`use Type::{A, B};` names those values, which only `Head::Tail` reading missed entirely.

    `border_swizzles` sweeps all three `MTLSamplerBorderColor` values in nine samplers through a
    braced `use` and then writes the variants bare, naming `SamplerBorderColor::` nowhere. The
    axis read 1 of 3 and the document said so while the sweep was complete.
    """

    def test_a_braced_use_names_every_variant_it_imports(self):
        root = tree({"blends": "use Thing::{Alpha, Gamma};\nlet pair = [Alpha, Gamma];\n"})
        self.assertEqual(axis(root, {"blends"})["in_effect"], ["MTLThingAlpha", "MTLThingGamma"])

    def test_a_renamed_import_names_the_value_imported_not_the_local_name(self):
        root = tree({"blends": "use Thing::{Beta as B};\nlet x = [B, B];\n"})
        self.assertEqual(axis(root, {"blends"})["in_effect"], ["MTLThingBeta"])

    def test_the_objc2_spelling_of_a_braced_use_is_read(self):
        root = tree({"blends": "use MTLThing::{Delta, Epsilon};\nlet x = [Delta, Epsilon];\n"})
        self.assertEqual(axis(root, {"blends"})["in_effect"], ["MTLThingDelta", "MTLThingEpsilon"])

    def test_a_qualified_path_resolves_to_its_last_segment(self):
        root = tree({"blends": "use metal2vulkan::metal::Thing::{Zeta};\nlet x = Zeta;\n"})
        self.assertEqual(axis(root, {"blends"})["in_effect"], ["MTLThingZeta"])

    def test_a_braced_use_of_something_else_names_no_value(self):
        # The direction this could go wrong in: a brace list of names that are not the type's
        # values, and a type name that is not a Metal type at all.
        root = tree({"blends": "use Thing::{Omega};\nuse Other::{Alpha, Beta};\n"})
        self.assertEqual(axis(root, {"blends"})["in_effect"], [])

    def test_a_bare_name_with_no_use_names_nothing(self):
        # The unsound reading this deliberately does not take: a module that writes `Alpha` and
        # imports nothing has not said which type's value it means.
        root = tree({"blends": "let x = [Alpha, Beta, Gamma];\n"})
        self.assertEqual(axis(root, {"blends"})["in_effect"], [])

    def test_a_braced_use_in_a_comment_names_nothing(self):
        root = tree({"blends": "// use Thing::{Alpha, Beta};\nfn go() {}\n"})
        self.assertEqual(axis(root, {"blends"})["in_effect"], [])


class Table(unittest.TestCase):
    """The generated enumerator table is the denominator for a type the tree declares by hand.

    These types have no generated Rust, so before the table the axis measured a hand-written
    variant list against itself: it could not report a value the SDK declares and the tree does
    not, and it reported an option-set newtype -- which declares no variants at all -- as a type
    with no values. Each test here asserts both that the table's values are counted and that
    nothing is counted the table does not hold.
    """

    def row(self, root: Path, ran: set[str]):
        """`MTLKept`'s row of the per-type table."""
        held = option_space.inventory(root, runs(ran))
        return next(row for row in held["rows"] if row["metal"] == "MTLKept")

    def test_the_table_is_the_denominator_and_leaves_the_hand_modelled_list(self):
        root = tree({}, by_hand=BY_HAND_KEPT)
        held = option_space.inventory(root, {})
        row = next(one for one in held["rows"] if one["metal"] == "MTLKept")
        self.assertEqual(row["values"], ["MTLKeptFirst", "MTLKeptSecond", "MTLKeptThird"])
        self.assertEqual(row["compared"], [])
        self.assertEqual(row["missing"], row["values"])
        # And it is no longer in the table of types with no denominator, which is where it sat.
        self.assertNotIn("MTLKept", [one["metal"] for one in held["by_hand"]])

    def test_a_value_the_table_does_not_hold_is_not_counted(self):
        # The one direction a table-backed denominator could go wrong in: a name shaped like an
        # enumerator of the type, which the SDK does not declare.
        root = tree({"kept": "let x = Kept::Fourth;\n"}, by_hand=BY_HAND_KEPT)
        row = self.row(root, {"kept"})
        self.assertEqual(row["compared"], [])
        self.assertEqual(len(row["values"]), 3)

    def test_the_trees_own_variant_name_states_the_sdk_value(self):
        # `MTLBlitOption`'s shape: our variant is `Second`, the table's derived name is `Second`
        # here but the join that matters is the enumerator, so a tree whose spelling differs from
        # the table's still resolves. The objc2 spelling resolves the same way.
        root = tree({"kept": "let x = Kept::Second;\n"}, by_hand=BY_HAND_KEPT)
        self.assertEqual(self.row(root, {"kept"})["in_effect"], ["MTLKeptSecond"])
        root = tree({"kept": "let x = MTLKept::Third;\n"}, by_hand=BY_HAND_KEPT)
        self.assertEqual(self.row(root, {"kept"})["in_effect"], ["MTLKeptThird"])

    def test_a_deprecated_spelling_names_the_value_it_is_an_alias_of(self):
        # `MTLPipelineOption`'s shape: the headers declare `MTLPipelineOptionArgumentInfo` first,
        # so the canonical enumerator of bit 1 is the one nothing in the tree writes, and the
        # spelling the leg does write -- `MTLPipelineOption::BindingInfo` -- is the alias. Without
        # the alias rows that value reads as named nowhere.
        root = tree({"kept": "let x = MTLKept::OldSecond;\n"}, by_hand=BY_HAND_KEPT)
        row = self.row(root, {"kept"})
        self.assertEqual(row["in_effect"], ["MTLKeptSecond"])
        self.assertNotIn("MTLKeptOldSecond", row["values"])
        # And answered reads an alias in a recording's transcript as the value too.
        root = tree(
            {"kept": "fn run() {}\n"},
            recordings=(recording("kept-one", "MTLKeptOldSecond"),),
            by_hand=BY_HAND_KEPT,
        )
        self.assertEqual(self.row(root, {"kept"})["answered"], ["MTLKeptSecond"])

    def test_a_table_that_parsed_to_nothing_is_a_failure(self):
        # The one way a denominator can read as a closed option space: a table whose rows the
        # reader no longer matches lowers every count at once and still looks like a measurement.
        root = tree({}, table="# only a comment\ntype\tkind\trole\trust\tenumerator\tvalue\n")
        with self.assertRaises(SystemExit):
            option_space.inventory(root, {})


class NamedWithNoValue(unittest.TestCase):
    """A type a module is about and states no value of: the raw-ordinal limit as a work list."""

    def test_a_module_naming_the_type_and_no_value_is_reported(self):
        root = tree({"blends": "for bits in 0..6 { side.go(MTLThing(bits)); }\n"})
        self.assertEqual(axis(root, {"blends"})["named_no_value"], ["blends"])

    def test_a_module_that_states_a_value_is_not_reported(self):
        # The same module with one value named: it is no longer a case whose depth is unreadable,
        # whatever else it sweeps by ordinal.
        root = tree({"blends": "let x = MTLThing::Alpha;\nfor bits in 0..6 { go(bits); }\n"})
        row = axis(root, {"blends"})
        self.assertEqual(row["named_no_value"], [])
        self.assertEqual(row["in_effect"], ["MTLThingAlpha"])

    def test_a_module_naming_neither_is_not_reported(self):
        root = tree({"blends": "let x = 3;\n"})
        self.assertEqual(axis(root, {"blends"})["named_no_value"], [])

    def test_the_type_name_is_matched_as_a_whole_identifier(self):
        # `ThingCase` is not `Thing`, and `MTLThingAlpha` is a value's name and not the type's.
        root = tree({"blends": "struct ThingCase;\nfn go(x: MTLThingAlphaish) {}\n"})
        self.assertEqual(axis(root, {"blends"})["named_no_value"], [])

    def test_a_module_with_no_metal_run_is_not_reported(self):
        # The same gate the rest of the axis applies: a module that ran no Metal put nothing to it.
        root = tree({"blends": "for bits in 0..6 { side.go(MTLThing(bits)); }\n"})
        self.assertEqual(axis(root, set())["named_no_value"], [])



class Unreachable(unittest.TestCase):
    """A value no case can put to Metal is counted apart from the gap, and the list is falsifiable.

    The column exists so that a work list is work. What makes it trustworthy is not the list but
    the two refusals: it cannot claim a value some case compared, and it cannot name a value the
    type does not have. Without those, one row would quietly raise the share this axis reports.
    """

    def test_a_declared_value_leaves_the_gap_and_joins_its_own_column(self):
        row = axis(tree({}, unreachable="# type\tvalue\twhy\tfinding\nMTLThing\tMTLThingDelta\tthe process ends on it\tcases/m.rs\n"), set())
        self.assertNotIn("MTLThingDelta", row["missing"])
        self.assertEqual(["MTLThingDelta"], row["settled"])
        self.assertNotIn("MTLThingDelta", row["compared"])

    def test_declaring_a_value_a_case_compared_is_refused(self):
        root = tree(
            {"blends": "fn go() { let x = Thing::Beta; }"},
            unreachable="MTLThing\tMTLThingBeta\tclaimed unreachable\tcases/blends.rs\n",
        )
        with self.assertRaises(SystemExit) as refused:
            option_space.inventory(root, runs({"blends"}))
        self.assertIn("but a case compared it", str(refused.exception))

    def test_declaring_a_value_the_type_does_not_have_is_refused(self):
        root = tree({}, unreachable="MTLThing\tMTLThingNope\trenamed or mistyped\tcases/m.rs\n")
        with self.assertRaises(SystemExit) as refused:
            option_space.inventory(root, {})
        self.assertIn("is not a value of MTLThing", str(refused.exception))

    def test_a_row_without_a_reason_is_refused(self):
        root = tree({}, unreachable="MTLThing\tMTLThingDelta\t\tcases/m.rs\n")
        with self.assertRaises(SystemExit) as refused:
            option_space.inventory(root, {})
        self.assertIn("carries no reason", str(refused.exception))

    def test_a_malformed_row_is_refused(self):
        root = tree({}, unreachable="MTLThing\tMTLThingDelta\n")
        with self.assertRaises(SystemExit) as refused:
            option_space.inventory(root, {})
        self.assertIn("is not `type", str(refused.exception))

    def test_a_tree_declaring_none_settles_nothing(self):
        self.assertEqual([], axis(tree({}), set())["settled"])

class Crossings(unittest.TestCase):
    """A declared crossing of two types, and the five ways a row can be false of the tree.

    This is the pairs axis, and it rests on a declaration, so what it is worth is what it refuses.
    Four of the refusals catch a row that no longer matches the sources -- a renamed cell, a module
    that ran nothing, an axis that is not a Metal type -- and the fifth is the one that bears on the
    claim: a module whose own text states fewer than two values of an axis cannot vary it, so a
    single value would falsify the row and the cross is not stateable there.
    """

    #: Two values of each type, which is what a cross of the two needs.
    PAIR = "fn go() { let a = [Thing::Beta, Thing::Gamma]; let s = Spread::LEFT | Spread::RIGHT; }"

    def declared(self, text: str = "", cell: str = "blends-case", axes: str = "MTLThing+MTLSpread"):
        """The crossings of a tree whose `blends` module states `text` and declares one row."""
        root = tree(
            {"blends": text or self.PAIR},
            crossings=f"# module\tcell\taxes\twhat only the cross decides\n"
            f"blends\t{cell}\t{axes}\tthe pair alone says which\n",
        )
        return option_space.inventory(root, {"blends-case": "blends"})["crossings"]

    def test_a_declared_crossing_is_counted_with_its_pair(self):
        held = self.declared()
        self.assertEqual(len(held), 1)
        self.assertEqual(held[0]["module"], "blends")
        self.assertEqual(held[0]["cell"], "blends-case")
        self.assertEqual(held[0]["axes"], ["MTLThing", "MTLSpread"])
        self.assertEqual(held[0]["pairs"], [("MTLSpread", "MTLThing")])
        self.assertEqual(held[0]["why"], "the pair alone says which")

    def test_a_pair_is_sorted_so_either_order_is_one_pair(self):
        # Two rows crossing the same two types in either order must count one pair, not two.
        held = self.declared(axes="MTLSpread+MTLThing")
        self.assertEqual(held[0]["pairs"], [("MTLSpread", "MTLThing")])

    def test_a_three_axis_row_holds_its_three_pairs(self):
        text = (
            "fn go() { let a = [Thing::Beta, Thing::Gamma]; let s = Spread::LEFT | Spread::RIGHT; "
            "let k = [Kept::First, Kept::Second]; }"
        )
        root = tree(
            {"blends": text},
            by_hand=BY_HAND_KEPT,
            crossings="blends\tblends-case\tMTLThing+MTLSpread+MTLKept\tonly the triple says\n",
        )
        held = option_space.inventory(root, {"blends-case": "blends"})["crossings"]
        self.assertEqual(
            held[0]["pairs"],
            [
                ("MTLKept", "MTLSpread"),
                ("MTLKept", "MTLThing"),
                ("MTLSpread", "MTLThing"),
            ],
        )

    def test_the_cells_may_be_the_familys_own_cases(self):
        root = tree({"blends": self.PAIR}, crossings="blends\tfamily\tMTLThing+MTLSpread\tthe cross is read across its cases\n")
        held = option_space.inventory(
            root, {"blends-one": "blends", "blends-two": "blends"}
        )["crossings"]
        self.assertEqual(held[0]["cell"], option_space.FAMILY)

    def test_a_tree_declaring_none_crosses_nothing(self):
        root = tree({"blends": self.PAIR})
        self.assertEqual(option_space.inventory(root, {"blends-case": "blends"})["crossings"], [])

    def test_a_cell_that_is_not_a_case_of_the_module_is_refused(self):
        # The rot this catches: a case renamed or deleted, its row left behind.
        with self.assertRaises(SystemExit) as refused:
            self.declared(cell="blends-renamed")
        self.assertIn("which is not a case of blends", str(refused.exception))

    def test_a_module_that_ran_no_metal_is_refused(self):
        root = tree(
            {"blends": self.PAIR, "others": self.PAIR},
            crossings="others\tothers-case\tMTLThing+MTLSpread\tthe pair alone says which\n",
        )
        with self.assertRaises(SystemExit) as refused:
            option_space.inventory(root, {"blends-case": "blends"})
        self.assertIn("no case of others has a recording", str(refused.exception))

    def test_a_family_of_one_recorded_case_is_refused(self):
        # A grid whose cells are the family's cases needs at least two of them.
        root = tree({"blends": self.PAIR}, crossings="blends\tfamily\tMTLThing+MTLSpread\tacross its cases\n")
        with self.assertRaises(SystemExit) as refused:
            option_space.inventory(root, {"blends-case": "blends"})
        self.assertIn("a cross across cases needs at least two", str(refused.exception))

    def test_an_axis_with_no_generated_value_list_is_refused(self):
        with self.assertRaises(SystemExit) as refused:
            self.declared(axes="MTLThing+MTLNotAType")
        self.assertIn("MTLNotAType, which has no generated value list", str(refused.exception))

    def test_an_axis_the_module_states_one_value_of_is_refused(self):
        # The refusal that bears on the claim: one value is a sweep's worth of an axis, so nothing
        # in the text tells the cross from a sweep of the other axis at a fixed cell.
        with self.assertRaises(SystemExit) as refused:
            self.declared(text="fn go() { let a = [Thing::Beta, Thing::Gamma]; let s = Spread::LEFT; }")
        self.assertIn("states 1 of its values by name", str(refused.exception))

    def test_an_axis_the_module_states_no_value_of_is_refused(self):
        with self.assertRaises(SystemExit) as refused:
            self.declared(text="fn go() { let a = [Thing::Beta, Thing::Gamma]; }")
        self.assertIn("states 0 of its values by name", str(refused.exception))

    def test_a_value_only_the_shared_sources_name_does_not_vary_an_axis(self):
        # A module that takes its value list from the suite's shared vocabulary states none of them
        # itself, which is how the suite's widest cross product reads here.
        root = tree(
            {"blends": "fn go() { let a = [Thing::Beta, Thing::Gamma]; go(SPREADS); }"},
            shared="const SPREADS: [Spread; 2] = [Spread::LEFT, Spread::RIGHT];",
            crossings="blends\tblends-case\tMTLThing+MTLSpread\tthe pair alone says which\n",
        )
        with self.assertRaises(SystemExit) as refused:
            option_space.inventory(root, {"blends-case": "blends"})
        self.assertIn("states 0 of its values by name", str(refused.exception))

    def test_a_row_of_one_axis_is_refused(self):
        with self.assertRaises(SystemExit) as refused:
            self.declared(axes="MTLThing")
        self.assertIn("which is one axis", str(refused.exception))

    def test_a_row_naming_an_axis_twice_is_refused(self):
        with self.assertRaises(SystemExit) as refused:
            self.declared(axes="MTLThing+MTLThing")
        self.assertIn("names an axis twice", str(refused.exception))

    def test_a_row_saying_nothing_about_the_cross_is_refused(self):
        root = tree({"blends": self.PAIR}, crossings="blends\tblends-case\tMTLThing+MTLSpread\t \n")
        with self.assertRaises(SystemExit) as refused:
            option_space.inventory(root, {"blends-case": "blends"})
        self.assertIn("says nothing about what only the cross decides", str(refused.exception))

    def test_a_malformed_row_is_refused(self):
        root = tree({"blends": self.PAIR}, crossings="blends\tblends-case\tMTLThing+MTLSpread\n")
        with self.assertRaises(SystemExit) as refused:
            option_space.inventory(root, {"blends-case": "blends"})
        self.assertIn("is not `module", str(refused.exception))

    def test_a_row_declared_twice_is_refused(self):
        root = tree(
            {"blends": self.PAIR},
            crossings=(
                "blends\tblends-case\tMTLThing+MTLSpread\tthe pair alone says which\n"
                "blends\tblends-case\tMTLThing+MTLSpread\tthe same claim again\n"
            ),
        )
        with self.assertRaises(SystemExit) as refused:
            option_space.inventory(root, {"blends-case": "blends"})
        self.assertIn("is declared twice", str(refused.exception))


if __name__ == "__main__":
    unittest.main()
