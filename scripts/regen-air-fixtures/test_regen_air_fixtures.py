"""Tests for regen-air-fixtures.sh's LLVM toolchain handling.

Each test runs a copy of the script on its own fixture tree against fake `xcrun`, `llvm-dis`, and
`llvm-as`, so no Metal toolchain or LLVM install is needed and no tool on the host's PATH is
reachable. A fake `llvm-dis` of major M marks its output `; llvm-dis M`; a fake `llvm-as` of
major M rejects text marked by a newer `llvm-dis` (as LLVM 21's rejects LLVM 23's
`splat (float f0x...)`) and text containing `REJECT` (a verifier failure). The fake Metal
toolchain's version and SDK come from FAKE_METAL, FAKE_SDK, and FAKE_SDK_BUILD (default 1.0, 99.0,
99A1), and its modules record them as Apple's do (`!llvm.ident`, the `SDK Version` module flag).
Its modules target the macOS version `-mmacosx-version-min` names, else FAKE_HOST_MACOS (default
99.0), as Apple's compiler takes a host-dependent default.
"""

import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "regen-air-fixtures.sh"
REPO = HERE.parent.parent
PIN = int(re.search(r"^llvm_major=(\d+)$", SCRIPT.read_text(), re.M).group(1))
MACOS_MIN = re.search(r"^default_macos_min=([0-9.]+)$", SCRIPT.read_text(), re.M).group(1)
NEWER = PIN + 2
# What the script and the fake run.
UTILITIES = ["awk", "basename", "bash", "cat", "cmp", "cp", "diff", "dirname", "grep", "head",
             "mkdir", "mktemp", "mv", "rm", "sed", "sort"]
# A host whose Metal toolchain was updated, and the TOOLCHAIN records of both toolchains.
NEW_METAL = {"FAKE_METAL": "2.0", "FAKE_SDK": "100.0", "FAKE_SDK_BUILD": "100A1"}
OLD_RECORD = "metal: Fake metal version 1.0; macosx-sdk: 99.0 (99A1)"
NEW_RECORD = "metal: Fake metal version 2.0; macosx-sdk: 100.0 (100A1)"

# One fake for every tool, reached through symlinks: the invoked name picks the tool, and a
# VERSION file next to an LLVM tool's link gives its version. xcrun's bitcode is the source
# text, and its typed text is that behind a `; typed` line, followed by the module's target triple
# and toolchain metadata; either ends with a `; flags` line naming the language-standard, fast-math, and
# deployment-target flags the compile received, if any.
FAKE = r"""#!/bin/sh
metal="${FAKE_METAL:-1.0}"; sdk="${FAKE_SDK:-99.0}"; build="${FAKE_SDK_BUILD:-99A1}"
target="${FAKE_HOST_MACOS:-99.0}"
case "$(basename "$0")" in
  xcrun)
    [ "$1" = -sdk ] && shift 2
    case "$1" in
      -f) echo "/fake/$2"; exit 0 ;;
      --show-sdk-version) echo "$sdk"; exit 0 ;;
      --show-sdk-build-version) echo "$build"; exit 0 ;;
    esac
    tool="$1"; shift; out=""; typed=0; input=""; flags=""
    while [ $# -gt 0 ]; do
      case "$1" in
        --version) echo "Fake metal version $metal"; exit 0 ;;
        -o) out="$2"; shift ;;
        -S) typed=1 ;;
        -mmacosx-version-min=*) flags="$flags $1"; target="${1#*=}" ;;
        -std=*|-fno-fast-math) flags="$flags $1" ;;
        -*) ;;
        *) input="$1" ;;
      esac
      shift
    done
    case "$tool" in
      metal)
        if [ $typed -eq 1 ]; then { echo "; typed"; cat; } >"$out"; else cat >"$out"; fi
        printf '%s\n' "target triple = \"air64_v1-apple-macosx$target.0\"" \
          '!llvm.ident = !{!90}' "!90 = !{!\"Fake metal version $metal\"}" \
          "!91 = !{i32 2, !\"SDK Version\", [2 x i32] [i32 ${sdk%%.*}, i32 ${sdk#*.}]}" >>"$out"
        if [ -n "$flags" ]; then echo "; flags$flags" >>"$out"; fi ;;
      metal-as) sed 1d "$input" >"$out" ;;
      *) echo "fake xcrun: unknown tool $tool" >&2; exit 1 ;;
    esac ;;
  llvm-dis*|llvm-as*)
    version="$(cat "$(dirname "$0")/VERSION")"; major="${version%%.*}"
    if [ "$1" = --version ]; then
      printf 'LLVM (http://llvm.org/):\n  Fake LLVM version %s\n' "$version"; exit 0
    fi
    [ "$1" = -o ] || exit 2
    case "$(basename "$0")" in
      llvm-dis*) { echo "; llvm-dis $major"; cat; } >"$2" ;;
      *)
        writer="$(sed -n 's/^; llvm-dis \([0-9]*\)$/\1/p' "$3")"
        if [ -n "$writer" ] && [ "$writer" -gt "$major" ]; then
          echo "$3:2:1: error: expected value token" >&2; exit 1
        fi
        if grep -q REJECT "$3"; then echo "$3: error: verifier failure" >&2; exit 1; fi ;;
    esac ;;
  *) echo "fake: unknown tool $0" >&2; exit 1 ;;
esac
"""


class RegenAirFixturesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The toolchains are shared read-only inputs. macOS scans a newly written executable on
        # its first exec, which can take a minute, so there is one fake file, written once.
        cls.tools = Path(tempfile.mkdtemp(prefix="regen-air-fixtures-tools."))
        cls.addClassCleanup(shutil.rmtree, cls.tools)
        fake = cls.tools / "fake"
        fake.write_text(FAKE)
        fake.chmod(0o755)
        # Only the utilities the script needs, so no LLVM tool of the host is reachable.
        cls.bin = cls.tools / "bin"
        cls.bin.mkdir()
        for utility in UTILITIES:
            found = shutil.which(utility)
            if found is None:
                raise unittest.SkipTest(f"{utility} is not on PATH")
            (cls.bin / utility).symlink_to(found)
        (cls.bin / "xcrun").symlink_to(fake)
        cls.newer = cls.llvm("newer", f"{NEWER}.1.2")
        cls.pinned = cls.llvm("pinned", f"{PIN}.1.8")
        cls.patch = cls.llvm("patch", f"{PIN}.1.9")
        cls.debian = cls.llvm("debian", f"{PIN}.1.8", suffix=f"-{PIN}")

    @classmethod
    def llvm(cls, name, version, suffix=""):
        """A directory with llvm-dis<suffix> and llvm-as<suffix> of `version`."""
        directory = cls.tools / name
        directory.mkdir()
        (directory / "VERSION").write_text(version + "\n")
        for tool in ("llvm-dis", "llvm-as"):
            (directory / (tool + suffix)).symlink_to(cls.tools / "fake")
        return directory

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="regen-air-fixtures-test."))
        self.addCleanup(shutil.rmtree, self.root)
        self.script = self.root / "scripts/regen-air-fixtures/regen-air-fixtures.sh"
        self.script.parent.mkdir(parents=True)
        shutil.copy2(SCRIPT, self.script)
        self.fixtures = self.root / "crates/metal2vulkan-validation/fixtures/public/engine"
        self.fixtures.mkdir(parents=True)
        (self.fixtures / "kernel.metal").write_text("kernel void k() {}\n")
        (self.fixtures / "legacy.metal").write_text("// metal-std: macos-metal2.2\nkernel\n")
        (self.root / "tmp").mkdir()

    def run_script(self, path, *args, env=None):
        environment = {"PATH": os.pathsep.join(str(p) for p in [self.bin, *path]),
                       "TMPDIR": str(self.root / "tmp")}
        environment.update(env or {})
        # bash reads the copy rather than executing it (a new executable, see setUpClass); the
        # timeout is only a hang guard.
        return subprocess.run([str(self.bin / "bash"), str(self.script), *args],
                              env=environment, capture_output=True, text=True, timeout=600)

    def committed(self):
        return {p.name: p.read_text() for p in sorted(self.fixtures.iterdir())
                if p.suffix != ".metal"}

    def test_the_pinned_llvm_writes_behind_a_newer_one_on_path(self):
        result = self.run_script([self.newer, self.pinned])
        self.assertEqual(result.returncode, 0, result.stderr)
        files = self.committed()
        self.assertEqual(sorted(files), ["TOOLCHAIN", "kernel.air.ll", "kernel.ll",
                                         "legacy.air.ll", "legacy.ll"])
        self.assertEqual(files["kernel.ll"].splitlines()[1], f"; llvm-dis {PIN}")
        self.assertEqual(files["TOOLCHAIN"].splitlines()[1:], [
            "metal: Fake metal version 1.0",
            "macosx-sdk: 99.0 (99A1)",
            f"llvm: {PIN} (pinned: llvm-dis writes each .ll, and llvm-as must accept it)",
            f"llvm-dis: Fake LLVM version {PIN}.1.8",
            f"llvm-as: Fake LLVM version {PIN}.1.8",
        ])
        # The check only compares: in either PATH order, with Debian's versioned names, and
        # with overrides naming the pinned tools.
        overrides = {"METAL2VULKAN_LLVM_DIS": str(self.pinned / "llvm-dis"),
                     "METAL2VULKAN_LLVM_AS": str(self.pinned / "llvm-as")}
        for path, env in (([self.newer, self.pinned], None), ([self.pinned, self.newer], None),
                          ([self.newer, self.debian], None), ([self.newer], overrides)):
            with self.subTest(path=[p.name for p in path], env=env):
                check = self.run_script(path, "--check", env=env)
                self.assertEqual(check.returncode, 0, check.stderr)
                self.assertEqual(check.stdout, "air fixtures up to date "
                                 "(2 sources: 2 checked, 0 not checked)\n")
        self.assertEqual(self.committed(), files)

    def test_source_directives_reach_both_compiles(self):
        (self.fixtures / "precise.metal").write_text("// metal-fast-math: off\nkernel\n")
        (self.fixtures / "both.metal").write_text(
            "// metal-std: macos-metal2.2\n// metal-fast-math: off\nkernel\n")
        (self.fixtures / "targeted.metal").write_text(
            "// metal-std: metal4.0\n// metal-macos-min: 26.2\nkernel\n")
        result = self.run_script([self.pinned])
        self.assertEqual(result.returncode, 0, result.stderr)
        files = self.committed()
        # A source without a deployment target compiles for the pinned one, whatever the host's
        # default is.
        pinned = f" -mmacosx-version-min={MACOS_MIN}"
        for name, flags in (("kernel", pinned), ("legacy", " -std=macos-metal2.2" + pinned),
                            ("precise", " -fno-fast-math" + pinned),
                            ("both", " -std=macos-metal2.2 -fno-fast-math" + pinned),
                            ("targeted", " -std=metal4.0 -mmacosx-version-min=26.2")):
            for suffix in (".ll", ".air.ll"):
                with self.subTest(fixture=name + suffix):
                    self.assertEqual(files[name + suffix].splitlines()[-1], "; flags" + flags)

    def test_a_rename_reaches_both_companions_and_the_check(self):
        (self.fixtures / "agx.metal").write_text(
            "// air-rename: _Z4stubt llvm.agx3.yield\n"
            "declare void @_Z4stubt(i16)\n"
            "call void @_Z4stubt(i16 0)\n")
        # The fixture commits its .air, so the run refreshes one rather than writing none.
        (self.fixtures / "agx.air").write_text("; stale\n")
        result = self.run_script([self.pinned])
        self.assertEqual(result.returncode, 0, result.stderr)
        files = self.committed()
        for suffix in (".ll", ".air.ll", ".air"):
            with self.subTest(file="agx" + suffix):
                body = [line for line in files["agx" + suffix].splitlines()
                        if not line.startswith(("//", ";"))]
                self.assertIn("declare void @llvm.agx3.yield(i16)", body)
                self.assertIn("call void @llvm.agx3.yield(i16 0)", body)
                self.assertFalse(any("_Z4stubt" in line for line in body), body)
        check = self.run_script([self.pinned], "--check")
        self.assertEqual(check.returncode, 0, check.stderr)
        self.assertEqual(self.committed(), files)

    def test_a_rename_needs_a_declared_function_and_an_intrinsic_name(self):
        for text, message in (
            ("// air-rename: _Z4stubt yield\ndeclare void @_Z4stubt(i16)\n",
             "agx.metal: air-rename needs <declared> <llvm.* or air.* name>, not: _Z4stubt yield"),
            ("// air-rename: _Z4stubt llvm.agx3.yield extra\ndeclare void @_Z4stubt(i16)\n",
             "agx.metal: air-rename needs <declared> <llvm.* or air.* name>"),
            ("// air-rename: _Z4stubt llvm.agx3.yield\ndeclare void @_Z4other(i16)\n",
             "agx.metal: air-rename: the module declares no _Z4stubt"),
            ("// air-rename: _Z4stubt llvm.agx3.yield\ndeclare void @_Z4stubt(i16)\n"
             "define void @_Z4stubt(i16 %0) {\n",
             "agx.metal: air-rename: the module defines _Z4stubt"),
        ):
            with self.subTest(text=text):
                (self.fixtures / "agx.metal").write_text(text)
                result = self.run_script([self.pinned])
                self.assertEqual(result.returncode, 1)
                self.assertIn(message, result.stderr)
                self.assertEqual(self.committed(), {})

    BY_VALUE_SOURCE = (
        "// air-by-value-buffer: p\n"
        '!4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", '
        '!"air.address_space", i32 2, !"air.struct_type_info", !5, !"air.arg_type_size", i32 160, '
        '!"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"Params", !"air.arg_name", '
        '!"p"}\n'
        '!6 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", '
        '!"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", '
        'i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}\n')

    def test_a_by_value_buffer_reaches_both_companions_and_the_check(self):
        (self.fixtures / "byvalue.metal").write_text(self.BY_VALUE_SOURCE)
        result = self.run_script([self.pinned])
        self.assertEqual(result.returncode, 0, result.stderr)
        files = self.committed()
        rewritten = ('!4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, '
                     '!"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, '
                     '!"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"char", '
                     '!"air.arg_name", !"p"}')
        untouched = self.BY_VALUE_SOURCE.splitlines()[2]
        for suffix in (".ll", ".air.ll"):
            with self.subTest(file="byvalue" + suffix):
                body = files["byvalue" + suffix].splitlines()
                self.assertIn(rewritten, body)
                self.assertIn(untouched, body)
                self.assertFalse(any('!"Params"' in line for line in body), body)
        check = self.run_script([self.pinned], "--check")
        self.assertEqual(check.returncode, 0, check.stderr)
        self.assertEqual(self.committed(), files)

    def test_a_by_value_buffer_needs_one_read_only_constant_buffer_of_its_name(self):
        node = self.BY_VALUE_SOURCE.splitlines()[1]
        for text, message in (
            ("// air-by-value-buffer: p q\n" + node + "\n",
             "byvalue.metal: air-by-value-buffer needs one argument name, not: p q"),
            ("// air-by-value-buffer: q\n" + node + "\n",
             "byvalue.metal: air-by-value-buffer: 0 buffer nodes name q, not one"),
            ("// air-by-value-buffer: p\n" + node + "\n" + node + "\n",
             "byvalue.metal: air-by-value-buffer: 2 buffer nodes name p, not one"),
            ("// air-by-value-buffer: out\n" + self.BY_VALUE_SOURCE.splitlines()[2] + "\n",
             "byvalue.metal: air-by-value-buffer: out is no read-only constant buffer"),
        ):
            with self.subTest(text=text):
                (self.fixtures / "byvalue.metal").write_text(text)
                result = self.run_script([self.pinned])
                self.assertEqual(result.returncode, 1)
                self.assertIn(message, result.stderr)
                self.assertEqual(self.committed(), {})

    def test_a_fast_math_value_other_than_off_is_refused(self):
        (self.fixtures / "kernel.metal").write_text("// metal-fast-math: on\nkernel\n")
        for args in ((), ("--check",)):
            with self.subTest(args=args):
                result = self.run_script([self.pinned], *args)
                self.assertEqual(result.returncode, 1)
                self.assertIn("kernel.metal: metal-fast-math must be off, not on", result.stderr)
                self.assertEqual(self.committed(), {})

    def test_a_deployment_target_that_is_not_a_dotted_version_is_refused(self):
        (self.fixtures / "kernel.metal").write_text("// metal-macos-min: 26.x\nkernel\n")
        for args in ((), ("--check",)):
            with self.subTest(args=args):
                result = self.run_script([self.pinned], *args)
                self.assertEqual(result.returncode, 1)
                self.assertIn("kernel.metal: metal-macos-min must be a dotted version, not 26.x",
                              result.stderr)
                self.assertEqual(self.committed(), {})

    def test_check_fails_when_a_module_targets_another_macos_than_its_source(self):
        (self.fixtures / "targeted.metal").write_text("// metal-macos-min: 26.2\nkernel\n")
        self.assertEqual(self.run_script([self.pinned]).returncode, 0)
        self.mixed_record("legacy")
        pristine = self.committed()
        self.assertIn(f'target triple = "air64_v1-apple-macosx{MACOS_MIN}.0"', pristine["kernel.ll"])
        self.assertIn('target triple = "air64_v1-apple-macosx26.2.0"', pristine["targeted.ll"])
        # Each module is held to its source's target, whether or not this host checks it (legacy
        # is recorded under another toolchain), and a module without one fails too.
        edits = (
            ("kernel.ll", f"macosx{MACOS_MIN}.0", "macosx27.0.0",
             f"deployment target mismatch: kernel.ll targets macOS 27.0.0; kernel.metal selects "
             f"{MACOS_MIN} (the pinned default)"),
            ("targeted.air.ll", "macosx26.2.0", "macosx26.0.0",
             "deployment target mismatch: targeted.air.ll targets macOS 26.0.0; targeted.metal "
             "selects 26.2 (its metal-macos-min line)"),
            ("legacy.ll", f'target triple = "air64_v1-apple-macosx{MACOS_MIN}.0"\n', "",
             "deployment target mismatch: legacy.ll targets macOS nothing; legacy.metal selects "
             f"{MACOS_MIN} (the pinned default)"),
        )
        for file, old, new, message in edits:
            with self.subTest(file=file):
                for name, text in pristine.items():
                    (self.fixtures / name).write_text(text)
                self.assertEqual(self.run_script([self.pinned], "--check").returncode, 0)
                self.assertIn(old, pristine[file])
                (self.fixtures / file).write_text(pristine[file].replace(old, new, 1))
                check = self.run_script([self.pinned], "--check")
                self.assertEqual(check.returncode, 1)
                self.assertIn(message, check.stderr.splitlines())
        # A host whose own default is another macOS version still compiles the pinned one.
        for name, text in pristine.items():
            (self.fixtures / name).write_text(text)
        check = self.run_script([self.pinned], "--check", env={"FAKE_HOST_MACOS": "27.0"})
        self.assertEqual(check.returncode, 0, check.stderr)

    def mixed_record(self, name):
        """Records fixture `name` under another Metal toolchain, so `--check` does not compile it."""
        toolchain = self.fixtures / "TOOLCHAIN"
        toolchain.write_text(toolchain.read_text() + "# Fixtures compiled by another Metal toolchain "
                             f"(regen-air-fixtures.sh <name>...).\nfixture {name}: {NEW_RECORD}\n")
        for suffix in (".ll", ".air.ll"):
            path = self.fixtures / (name + suffix)
            path.write_text(path.read_text().replace("Fake metal version 1.0", "Fake metal version 2.0")
                            .replace("[i32 99, i32 0]", "[i32 100, i32 0]"))

    def test_no_pinned_llvm_is_refused_before_writing(self):
        for args in ((), ("--check",)):
            with self.subTest(args=args):
                result = self.run_script([self.newer], *args)
                self.assertEqual(result.returncode, 1)
                self.assertIn(f"no llvm-dis of LLVM {PIN}", result.stderr)
                self.assertIn(f"{self.newer / 'llvm-dis'}: Fake LLVM version {NEWER}.1.2",
                              result.stderr)
                self.assertEqual(self.committed(), {})

    def test_an_override_of_another_major_is_refused(self):
        for variable, tool in (("METAL2VULKAN_LLVM_DIS", "llvm-dis"),
                               ("METAL2VULKAN_LLVM_AS", "llvm-as")):
            with self.subTest(variable=variable):
                result = self.run_script([self.pinned], env={variable: str(self.newer / tool)})
                self.assertEqual(result.returncode, 1)
                self.assertIn(f"{variable}={self.newer / tool} is Fake LLVM version {NEWER}.1.2, "
                              f"but the fixtures pin LLVM {PIN}", result.stderr)
                self.assertEqual(self.committed(), {})

    def test_text_the_pinned_llvm_as_rejects_fails_and_writes_nothing(self):
        self.assertEqual(self.run_script([self.pinned]).returncode, 0)
        before = self.committed()
        (self.fixtures / "kernel.metal").write_text("kernel void k() { REJECT }\n")
        for args in ((), ("--check",)):
            with self.subTest(args=args):
                result = self.run_script([self.pinned], *args)
                self.assertEqual(result.returncode, 1)
                self.assertIn(f"kernel.ll: llvm-as of LLVM {PIN} rejects the text:",
                              result.stderr)
                self.assertIn("verifier failure", result.stderr)
                self.assertEqual(self.committed(), before)

    def test_check_names_stale_and_orphan_companions(self):
        self.assertEqual(self.run_script([self.pinned]).returncode, 0)
        with open(self.fixtures / "kernel.ll", "a") as ll:
            ll.write("; edited\n")
        (self.fixtures / "orphan.ll").write_text("; orphan\n")
        # Another build of the pinned major writes the same text but is recorded exactly.
        result = self.run_script([self.patch], "--check")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stderr.splitlines(),
                         ["stale: kernel.ll", "stale: TOOLCHAIN", "orphan: orphan.ll",
                          "air fixtures not up to date (2 sources: 2 checked, 0 not checked)"])

    def toolchain(self):
        return (self.fixtures / "TOOLCHAIN").read_text().splitlines()

    #: Two fixture names a dictionary collation and byte order disagree about: en_US.UTF-8 weighs
    #: the letters and ignores the `_` and `:`, so it reads `color_groups_fetch` before
    #: `color_groups`, while byte order puts `:` (0x3A) and `.` before `_` (0x5F) and reads them
    #: the other way round. Both the `*.metal` glob and the fixture lines are affected.
    COLLATING = ("color_groups", "color_groups_fetch")

    def collates_differently(self, locale):
        """Whether `locale` really orders COLLATING's fixture lines against byte order.

        A host without the locale generated falls back to byte order, and a test that did not ask
        would then pass without having varied anything.
        """
        lines = "".join(f"fixture {n}: x\n" for n in sorted(self.COLLATING))
        sorted_there = subprocess.run(
            [str(self.bin / "bash"), "-c", "sort"],
            env={"PATH": str(self.bin), "LC_ALL": locale},
            input=lines, capture_output=True, text=True, timeout=60,
        )
        return sorted_there.returncode == 0 and sorted_there.stdout != lines

    def collating_fixtures(self):
        """A tree holding COLLATING's two fixtures, and a TOOLCHAIN of the old Metal toolchain."""
        for name in self.COLLATING:
            (self.fixtures / f"{name}.metal").write_text(f"kernel void {name}() {{}}\n")
        self.assertEqual(self.run_script([self.pinned]).returncode, 0)

    def test_the_fixture_lines_are_in_byte_order_whatever_the_locale(self):
        # A partial regeneration writes a line per named fixture, in an order that followed the
        # host's collation before `LC_COLLATE` was pinned and the lines sorted.
        self.collating_fixtures()
        wanted = None
        for locale in (None, "en_US.UTF-8", "C"):
            variable = {} if locale is None else {"LC_ALL": locale}
            with self.subTest(locale=locale):
                if locale not in (None, "C") and not self.collates_differently(locale):
                    self.skipTest(f"{locale} is not generated on this host")
                result = self.run_script([self.pinned], *self.COLLATING,
                                         env={**NEW_METAL, **variable})
                self.assertEqual(result.returncode, 0, result.stderr)
                lines = [l for l in self.toolchain() if l.startswith("fixture ")]
                self.assertEqual(lines,
                                 [f"fixture {n}: {NEW_RECORD}" for n in sorted(self.COLLATING)])
                if wanted is None:
                    wanted = self.toolchain()
                self.assertEqual(self.toolchain(), wanted)

    def test_check_refuses_fixture_lines_another_collation_ordered(self):
        # The file a host with another collation writes: the same records, the other order.
        self.collating_fixtures()
        self.assertEqual(self.run_script([self.pinned], *self.COLLATING, env=NEW_METAL).returncode, 0)
        committed = self.fixtures / "TOOLCHAIN"
        text = committed.read_text()
        self.assertEqual(self.run_script([self.pinned], "--check", env=NEW_METAL).returncode, 0)
        swapped, lines = [], [l for l in text.splitlines() if l.startswith("fixture ")]
        self.assertEqual(len(lines), 2)
        for line in text.splitlines():
            swapped.append(line if not line.startswith("fixture ") else lines[1 - lines.index(line)])
        committed.write_text("\n".join(swapped) + "\n")
        check = self.run_script([self.pinned], "--check", env=NEW_METAL)
        self.assertEqual(check.returncode, 1, check.stdout)
        self.assertIn("out of order: TOOLCHAIN fixture lines", check.stderr)
        # And rerunning the script puts them back.
        self.assertEqual(self.run_script([self.pinned], *self.COLLATING, env=NEW_METAL).returncode, 0)
        self.assertEqual(committed.read_text(), text)

    def test_the_air_is_written_only_where_the_directory_holds_one(self):
        # <name>.air is an intermediate except for the fixtures that commit one for tests to read
        # as bitcode, which a run keeps in step with their text. A run must add no other: the
        # committed .air files are linked together by `xcrun metallib`, which resolves each shared
        # `air.*` symbol to one definition for the whole link, so one more module can make a
        # symbol that linked before fail over the deployment targets in the set.
        (self.fixtures / "kernel.air").write_text("; stale\n")
        self.assertEqual(self.run_script([self.pinned]).returncode, 0)
        files = self.committed()
        self.assertEqual(sorted(files), ["TOOLCHAIN", "kernel.air", "kernel.air.ll", "kernel.ll",
                                         "legacy.air.ll", "legacy.ll"])
        self.assertNotIn("; stale", files["kernel.air"])
        self.assertEqual(self.run_script([self.pinned], "--check").returncode, 0)
        # A run naming fixtures follows the same rule.
        (self.fixtures / "kernel.air").unlink()
        (self.fixtures / "legacy.air").write_text("; stale\n")
        self.assertEqual(self.run_script([self.pinned], "kernel", "legacy").returncode, 0)
        self.assertEqual(sorted(self.committed()),
                         ["TOOLCHAIN", "kernel.air.ll", "kernel.ll",
                          "legacy.air", "legacy.air.ll", "legacy.ll"])
        self.assertNotIn("; stale", (self.fixtures / "legacy.air").read_text())

    def test_another_directory_regenerates_alone(self):
        other = self.root / "crates/metal2vulkan-validation/fixtures/public/reflection"
        other.mkdir(parents=True)
        (other / "probe.metal").write_text("kernel void p() {}\n")
        result = self.run_script([self.pinned], "--dir", "reflection")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(result.stdout.startswith(
            "regenerated 1 fixtures in crates/metal2vulkan-validation/fixtures/public/reflection"), result.stdout)
        self.assertEqual(sorted(p.name for p in other.iterdir()),
                         ["TOOLCHAIN", "probe.air.ll", "probe.ll", "probe.metal"])
        self.assertEqual(sorted(p.name for p in self.fixtures.iterdir()),
                         ["kernel.metal", "legacy.metal"])
        self.assertEqual(self.run_script([self.pinned], "--dir", "Reflection").returncode, 2)
        self.assertEqual(self.run_script([self.pinned], "--dir", "absent").returncode, 1)

    def mixed(self):
        """Every fixture compiled by Metal 1.0 (SDK 99.0), then `kernel` alone by Metal 2.0
        (SDK 100.0): the committed files of a host that updated its Metal toolchain and compiled
        one new fixture."""
        self.assertEqual(self.run_script([self.pinned]).returncode, 0)
        result = self.run_script([self.pinned], "kernel", env=NEW_METAL)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "regenerated 1 of 2 fixtures in "
                         f"crates/metal2vulkan-validation/fixtures/public/engine ({NEW_RECORD})\n")

    def test_a_partial_regeneration_records_its_fixtures_toolchain(self):
        self.mixed()
        files = self.committed()
        self.assertEqual(self.toolchain()[1:], [
            "metal: Fake metal version 1.0",
            "macosx-sdk: 99.0 (99A1)",
            f"llvm: {PIN} (pinned: llvm-dis writes each .ll, and llvm-as must accept it)",
            f"llvm-dis: Fake LLVM version {PIN}.1.8",
            f"llvm-as: Fake LLVM version {PIN}.1.8",
            "# Fixtures compiled by another Metal toolchain (regen-air-fixtures.sh <name>...).",
            f"fixture kernel: {NEW_RECORD}",
        ])
        self.assertIn('!90 = !{!"Fake metal version 2.0"}', files["kernel.ll"])
        self.assertIn('!90 = !{!"Fake metal version 1.0"}', files["legacy.ll"])
        # Each host checks exactly the fixtures recorded under its toolchain.
        for env, checked, other in ((NEW_METAL, "kernel", f"legacy (compiled by {OLD_RECORD})"),
                                    (None, "legacy", f"kernel (compiled by {NEW_RECORD})")):
            with self.subTest(checked=checked):
                check = self.run_script([self.pinned], "--check", env=env)
                self.assertEqual(check.returncode, 0, check.stderr)
                self.assertEqual(check.stdout.splitlines(), [
                    f"not checked: {other}",
                    "air fixtures up to date (2 sources: 1 checked, 1 not checked)"])
        # A fixture not checked is not compared, but a checked one still is.
        with open(self.fixtures / "legacy.air.ll", "a") as ll:
            ll.write("; edited\n")
        self.assertEqual(self.run_script([self.pinned], "--check", env=NEW_METAL).returncode, 0)
        with open(self.fixtures / "kernel.air.ll", "a") as ll:
            ll.write("; edited\n")
        check = self.run_script([self.pinned], "--check", env=NEW_METAL)
        self.assertEqual(check.returncode, 1)
        self.assertEqual(check.stderr.splitlines(), [
            "stale: kernel.air.ll",
            "air fixtures not up to date (2 sources: 1 checked, 1 not checked)"])
        # Naming a fixture again under the base toolchain drops its record; a plain run
        # regenerates everything under one toolchain.
        self.assertEqual(self.run_script([self.pinned], "kernel").returncode, 0)
        self.assertNotIn(f"fixture kernel: {NEW_RECORD}", self.toolchain())
        self.assertEqual(len(self.toolchain()), 6)
        self.assertEqual(self.run_script([self.pinned], env=NEW_METAL).returncode, 0)
        self.assertEqual(self.toolchain()[1:3],
                         ["metal: Fake metal version 2.0", "macosx-sdk: 100.0 (100A1)"])
        self.assertEqual(len(self.toolchain()), 6)

    def test_check_fails_when_a_module_disagrees_with_its_toolchain_record(self):
        self.mixed()
        pristine = self.committed()
        edits = (
            ("legacy.ll", '!{!"Fake metal version 1.0"}', '!{!"Fake metal version 1.5"}',
             "toolchain mismatch: legacy.ll records Fake metal version 1.5, SDK 99.0; "
             f"TOOLCHAIN records {OLD_RECORD}"),
            ("legacy.air.ll", "[i32 99, i32 0]", "[i32 98, i32 0]",
             "toolchain mismatch: legacy.air.ll records Fake metal version 1.0, SDK 98.0; "
             f"TOOLCHAIN records {OLD_RECORD}"),
            ("kernel.ll", "!llvm.ident = !{!90}\n", "",
             "toolchain mismatch: kernel.ll records no Metal version, SDK 100.0; "
             f"TOOLCHAIN records {NEW_RECORD}"),
            ("TOOLCHAIN", "Fake metal version 2.0;", "Fake metal version 1.0;",
             "toolchain mismatch: kernel.ll records Fake metal version 2.0, SDK 100.0; "
             "TOOLCHAIN records metal: Fake metal version 1.0; macosx-sdk: 100.0 (100A1)"),
        )
        for file, old, new, message in edits:
            with self.subTest(file=file):
                for name, text in pristine.items():
                    (self.fixtures / name).write_text(text)
                self.assertIn(old, pristine[file])
                (self.fixtures / file).write_text(pristine[file].replace(old, new, 1))
                check = self.run_script([self.pinned], "--check", env=NEW_METAL)
                self.assertEqual(check.returncode, 1)
                self.assertIn(message, check.stderr.splitlines())

    def test_a_partial_regeneration_needs_the_recorded_llvm_and_known_fixtures(self):
        result = self.run_script([self.pinned], "kernel")
        self.assertEqual(result.returncode, 1)
        self.assertIn("no TOOLCHAIN in crates/metal2vulkan-validation/fixtures/public/engine", result.stderr)
        self.assertEqual(self.run_script([self.pinned]).returncode, 0)
        before = self.committed()
        result = self.run_script([self.patch], "kernel", env=NEW_METAL)
        self.assertEqual(result.returncode, 1)
        self.assertIn("the host's LLVM tools are not the ones TOOLCHAIN records", result.stderr)
        result = self.run_script([self.pinned], "absent")
        self.assertEqual(result.returncode, 1)
        self.assertIn("no fixture absent", result.stderr)
        self.assertEqual(self.committed(), before)

    def test_the_committed_toolchains_fixture_lines_are_in_byte_order(self):
        # The order write_toolchain writes and --check demands. The two tests above hold the
        # script to it against fake tools; this one holds the committed files to it, on any host
        # and with no Metal toolchain or LLVM install, so a file written by a host whose script
        # followed its own collation is a failure here and not a diff in the next packet.
        directories = sorted((REPO / "crates/metal2vulkan-validation/fixtures/public").glob("*/TOOLCHAIN"))
        self.assertTrue(directories, "the fixture directories record their toolchains")
        for toolchain in directories:
            with self.subTest(toolchain=toolchain.parent.name):
                lines = [line for line in toolchain.read_text().splitlines()
                         if line.startswith("fixture ")]
                self.assertEqual(lines, sorted(lines))

    def test_committed_toolchain_records_the_pin(self):
        lines = (REPO / "crates/metal2vulkan-validation/fixtures/public/engine/TOOLCHAIN").read_text().splitlines()
        fields = dict(line.split(": ", 1) for line in lines if not line.startswith("#"))
        self.assertEqual(fields["llvm"].split()[0], str(PIN))
        for tool in ("llvm-dis", "llvm-as"):
            with self.subTest(tool=tool):
                self.assertRegex(fields[tool], rf"[Vv]ersion {PIN}\.")


if __name__ == "__main__":
    unittest.main()
