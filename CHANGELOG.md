# Changelog

## 0.2.0

A full rewrite: nothing of 0.1.0's API or command line remains.

- **Translator** (`metal2vulkan`): Metal AIR (a metallib or AIR bitcode) to Vulkan SPIR-V through
  a Metal-shaped API. It compiles compute, render, tessellation, tile and mesh pipelines for a
  described Vulkan device: SPIR-V per stage, one pipeline layout, fixed-function state, and binding
  plans that turn what a draw or dispatch binds into descriptor writes and push constants. The
  `cli` feature builds a `metal2vulkan` command that compiles one pipeline.
- **Runtime** (`metal2vulkan-runtime`): the Metal API on Vulkan. Devices, command queues and
  buffers, the encoders, buffers, textures, heaps, pipeline states, events and indirect command
  buffers run on a Vulkan device through `ash`.
- **AIR front end** (`metal2vulkan-air`): the metallib container, the bitcode reader, a typed IR,
  AIR's metadata ABI and canonicalization, as a crate of its own.
- **Typed refusals and static precision notes**: input a device cannot run fails with an error that
  names what is missing; where a device's answer can differ from Metal's, the operation carries a
  static precision note.
- **Validation tooling** (unpublished): authored cases run on Metal and on Vulkan and compared,
  plus bounded audits over a private corpus.
- **License**: dual-licensed `GPL-2.0-or-later OR LGPL-3.0-or-later`; 0.1.0 was `LGPL-3.0-or-later`.

## 0.1.0

The initial release: a single-stage AIR to SPIR-V translator and its command-line tool.
