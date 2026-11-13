import unittest

from regen_metal_enums import DECL, parse_body, render, rows, strip_comments, variant_name


class ParseTests(unittest.TestCase):
    def decl(self, text):
        m = DECL.search(strip_comments(text))
        self.assertIsNotNone(m)
        return m.groups()

    def test_annotations_comments_and_implicit_values(self):
        kind, _, name, body = self.decl("""
            typedef NS_ENUM(NSUInteger, MTLThing) {
                /// doc
                MTLThingA = 0,
                MTLThingB API_AVAILABLE(macos(10.11), ios(8.0)) = 4, // trailing
                MTLThingC,
                MTLThing2D NS_SWIFT_NAME(twoD) API_UNAVAILABLE(macos) = 0x10,
            } API_AVAILABLE(macos(10.11));""")
        self.assertEqual((kind, name), ("ENUM", "MTLThing"))
        entries = parse_body(name, body)
        self.assertEqual(entries, [("MTLThingA", 0), ("MTLThingB", 4), ("MTLThingC", 5),
                                   ("MTLThing2D", 16)])
        out, aliases = rows(name, entries, flags=False)
        self.assertEqual([r[0] for r in out], ["A", "B", "C", "Thing2D"])
        self.assertEqual(aliases, [])

    def test_option_expressions_and_aliases(self):
        _, _, name, body = self.decl("""
            typedef NS_OPTIONS(NSUInteger, MTLStages) {
                MTLStageVertex = 1UL << 0,
                MTLStageFragment = (1 << 1),
                MTLStageAll = MTLStageVertex | MTLStageFragment,
                MTLOldStageVertex = MTLStageVertex,
            };""")
        entries = parse_body(name, body)
        out, aliases = rows(name, entries, flags=True)
        self.assertEqual(out, [("VERTEX", "MTLStageVertex", 1),
                               ("FRAGMENT", "MTLStageFragment", 2),
                               ("ALL", "MTLStageAll", 3)])
        self.assertEqual(aliases, [("MTLOldStageVertex", "VERTEX")])

    def test_unknown_forms_fail(self):
        for body in ["MTLXA = sizeof(int)", "MTLXA = MTLYB", "MTLXA = 1 + 1", "MTLXA = -1"]:
            with self.subTest(body=body), self.assertRaises(ValueError):
                rows("MTLX", parse_body("MTLX", body), flags=False)

    def test_private_values_follow_the_headers_with_their_docs(self):
        found = {"MTLTriangleFillMode": (
            "ENUM", "MTLTriangleFillModeFill = 0, MTLTriangleFillModeLines = 1,")}
        text = render(found, "0.0", ["MTLTriangleFillMode"], [])
        lines = [line.strip() for line in text.splitlines()]
        after = lines.index('Lines = 1, "MTLTriangleFillModeLines";') + 1
        points = lines.index('Points = 2, "MTLTriangleFillModePoints";')
        self.assertEqual(lines[after], "///")
        self.assertTrue(all(line.startswith("/// ") for line in lines[after + 1:points]))
        self.assertLess(after + 1, points)
        declared = {"MTLTriangleFillMode": (
            "ENUM", "MTLTriangleFillModeFill = 0, MTLTriangleFillModePoints = 2,")}
        with self.assertRaises(ValueError):
            render(declared, "0.0", ["MTLTriangleFillMode"], [])

    def test_variant_names(self):
        prefixes = ["MTLPixelFormat", "MTLPixelForma"]
        self.assertEqual(variant_name("MTLPixelFormatBGRA8Unorm_sRGB", prefixes),
                         "BGRA8UnormSRGB")
        self.assertEqual(variant_name("MTLPixelFormatASTC_4x4_HDR", prefixes), "ASTC4x4HDR")
        with self.assertRaises(ValueError):
            variant_name("MTLOtherThing", prefixes)


if __name__ == "__main__":
    unittest.main()
