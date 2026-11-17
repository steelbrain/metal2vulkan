# metal2vulkan-rt

The Metal API on Vulkan, with one path for every device. It replaces `metal2vulkan-runtime` and
takes its name when it does; until then it is unpublished.

It has no public API yet.

## One path for every device

Every difference between devices comes from a Vulkan query: a feature, a property, a format, a
limit or a memory type. The crate holds:

- no vendor, device or driver identity;
- no environment gate;
- no runtime policy switch;
- no check that runs beside the program's work.

The test `tests::one_path` reads the crate's sources and fails on the first three.
