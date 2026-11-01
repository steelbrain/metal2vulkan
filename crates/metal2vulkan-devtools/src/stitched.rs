//! A stitched-library writer for tests: the modules Metal's stitcher
//! (`newLibraryWithStitchedDescriptor:`) writes for graphs of `float` stitchable functions, as
//! LLVM text, and the metallib that holds them. It reproduces the structure of the library the
//! probe `stitched_function_linked.m` (the conformance crate's `probes/`) read back from Metal on
//! an Apple M3 Max under macOS 27.0.1 (the library's own bytes); the text is written here, no
//! Apple-owned bytes.
//!
//! The library is an executable one (file type 0), each stitched function a visible function
//! (`TYPE` 4) with a module of its own ([`library`]). Each module ([`module`]) holds:
//!
//! - the stitched function, `float` of its `float` inputs (`%input<i>.0`), with each node's
//!   stitchable function inlined into its body in node order and the output node's value
//!   returned;
//! - its `air.visible` entry (inputs and output) with an `air.stitching_info` whose stitching
//!   types name internal helper functions (`load_argument`, `store_return_value`, `destroy`,
//!   `copy_from_buffer`, one set for the result and one for the arguments), which
//!   `llvm.compiler.used` keeps;
//! - its `!air.signature`, an `air.stitching_modules_signature`: the graph (the input and
//!   function nodes the output node reaches, in post-order, the output last and marked) and an
//!   `air.module_signature` per stitchable function the graph calls (a digest of the function's
//!   module; here the SHA-256 of its name, which no translation reads).

use std::fmt::Write;

use sha2::{Digest, Sha256};

use crate::metallib::{metallib, wrap, MtlbFunction};

/// The data layout of Apple's AIR modules.
const DATA_LAYOUT: &str = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-\
f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-\
v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32";

/// The stitching helpers of a `float` (`metal::_stitching_traits_impl<float>`), each its name
/// and its definition's signature and body (LLVM text, `{name}` in place of its name).
const HELPERS: [(&str, &str); 4] = [
    (
        "_ZN5metal22_stitching_traits_implIfE13load_argumentEPU9MTLthreadv",
        "float @{name}(ptr noundef %ptr) #0 align 2 {\nentry:\n  \
         %0 = load float, ptr %ptr, align 4\n  ret float %0\n}",
    ),
    (
        "_ZN5metal22_stitching_traits_implIfE18store_return_valueEPU9MTLthreadvf",
        "void @{name}(ptr noundef %ptr, float noundef %arg) #0 align 2 {\nentry:\n  \
         store float %arg, ptr %ptr, align 4\n  ret void\n}",
    ),
    (
        "_ZN5metal22_stitching_traits_implIfE7destroyEPU9MTLthreadv",
        "void @{name}(ptr noundef %ptr) #0 align 2 {\nentry:\n  ret void\n}",
    ),
    (
        "_ZN5metal22_stitching_traits_implIfE16copy_from_bufferEPU9MTLthreadvPU11MTLconstantKv",
        "void @{name}(ptr noundef %ptr, ptr addrspace(2) noundef %buffer_ptr) #0 align 2 {\n\
         entry:\n  %0 = load float, ptr addrspace(2) %buffer_ptr, align 4\n  \
         store float %0, ptr %ptr, align 4\n  ret void\n}",
    ),
];

/// The suffixes of the helper sets: the result's, then the arguments'.
const SETS: [&str; 2] = ["", ".1"];

/// An operand of a [`Node`]: the stitched function's input of this index, or the value of an
/// earlier node of its graph (by index).
#[derive(Clone, Copy, Debug)]
pub enum Operand {
    Input(u32),
    Node(usize),
}

/// A function node: a call of the stitchable function `function`, `float(float, float)`, whose
/// body is the one LLVM instruction `op` (`fadd`, `fmul`, ...) of its two arguments, on `args`.
#[derive(Clone, Copy, Debug)]
pub struct Node<'a> {
    pub function: &'a str,
    pub op: &'a str,
    pub args: [Operand; 2],
}

/// A stitched function (`MTLFunctionStitchingGraph`): its name, its number of `float` inputs,
/// and its function nodes in order, the last its output node.
#[derive(Clone, Copy, Debug)]
pub struct Graph<'a> {
    pub name: &'a str,
    pub inputs: u32,
    pub nodes: &'a [Node<'a>],
}

/// `sum2(a, b) = a + b` of `args`.
const fn sum2(args: [Operand; 2]) -> Node<'static> {
    Node {
        function: "sum2",
        op: "fadd",
        args,
    }
}

/// `product2(a, b) = a * b` of `args`.
const fn product2(args: [Operand; 2]) -> Node<'static> {
    Node {
        function: "product2",
        op: "fmul",
        args,
    }
}

/// The stitched functions of the probe `stitched_function_linked.m`, which the owned fixture
/// `engine/stitched_caller` calls, over its stitchable `sum2(a, b) = a + b` and
/// `product2(a, b) = a * b`: `fused(a, b, c) = product2(sum2(a, b), c)` (two function nodes) and
/// `fused4(a, b, c, d) = sum2(product2(sum2(a, b), c), d)` (three).
pub const PROBE_GRAPHS: [Graph<'static>; 2] = [
    Graph {
        name: "fused",
        inputs: 3,
        nodes: &[
            sum2([Operand::Input(0), Operand::Input(1)]),
            product2([Operand::Node(0), Operand::Input(2)]),
        ],
    },
    Graph {
        name: "fused4",
        inputs: 4,
        nodes: &[
            sum2([Operand::Input(0), Operand::Input(1)]),
            product2([Operand::Node(0), Operand::Input(2)]),
            sum2([Operand::Node(1), Operand::Input(3)]),
        ],
    },
];

/// Metadata nodes numbered in the order they are added.
#[derive(Default)]
struct Md(Vec<String>);

impl Md {
    /// Add `!{body}`; answers its reference (`!<n>`).
    fn add(&mut self, body: String) -> String {
        self.0.push(body);
        format!("!{}", self.0.len() - 1)
    }
}

/// The stitched module of `g` as LLVM text (opaque pointers), in the form Metal's stitcher
/// writes. Panics on a graph without nodes, or with an operand that names a later node or an
/// input it does not have.
pub fn module(g: &Graph<'_>) -> String {
    assert!(!g.nodes.is_empty(), "a stitched graph has an output node");
    let mut out = String::new();
    let o = &mut out;
    writeln!(o, "source_filename = \"{}\"", g.name).unwrap();
    writeln!(o, "target datalayout = \"{DATA_LAYOUT}\"").unwrap();
    writeln!(o, "target triple = \"air64_v29-apple-macosx27.0.0\"\n").unwrap();
    let helper = |i: usize, set: &str| format!("@{}{set}", HELPERS[i].0);
    let used: Vec<String> = SETS
        .iter()
        .flat_map(|set| (0..HELPERS.len()).map(move |i| format!("ptr {}", helper(i, set))))
        .collect();
    writeln!(
        o,
        "@llvm.compiler.used = appending global [{} x ptr] [{}], section \"llvm.metadata\"\n",
        used.len(),
        used.join(", ")
    )
    .unwrap();
    for set in SETS {
        for (name, body) in HELPERS {
            let name = format!("{name}{set}");
            writeln!(o, "define internal {}\n", body.replace("{name}", &name)).unwrap();
        }
    }
    let operand = |at: usize, a: Operand| match a {
        Operand::Input(i) => {
            assert!(i < g.inputs, "{}: input {i} of {}", g.name, g.inputs);
            format!("%input{i}.0")
        }
        Operand::Node(j) => {
            assert!(j < at, "{}: node {at} reads the later node {j}", g.name);
            format!("%node{j}")
        }
    };
    let params: Vec<String> = (0..g.inputs)
        .map(|i| format!("float %input{i}.0"))
        .collect();
    writeln!(
        o,
        "define float @{}({}) local_unnamed_addr #1 {{\nentry:",
        g.name,
        params.join(", ")
    )
    .unwrap();
    for (k, n) in g.nodes.iter().enumerate() {
        let [a, b] = n.args.map(|a| operand(k, a));
        writeln!(o, "  %node{k} = {} fast float {a}, {b}", n.op).unwrap();
    }
    writeln!(o, "  ret float %node{}\n}}\n", g.nodes.len() - 1).unwrap();
    writeln!(
        o,
        "attributes #0 = {{ alwaysinline mustprogress nounwind \"no-builtins\" }}\n\
         attributes #1 = {{ convergent nounwind \"no-builtins\" }}\n"
    )
    .unwrap();

    let mut md = Md::default();
    let flags: Vec<String> = [
        "i32 7, !\"air.max_device_buffers\", i32 31",
        "i32 7, !\"air.max_constant_buffers\", i32 31",
        "i32 7, !\"air.max_threadgroup_buffers\", i32 31",
        "i32 7, !\"air.max_textures\", i32 128",
        "i32 7, !\"air.max_read_write_textures\", i32 8",
        "i32 7, !\"air.max_samplers\", i32 16",
        "i32 1, !\"wchar_size\", i32 4",
        "i32 7, !\"frame-pointer\", i32 2",
    ]
    .iter()
    .map(|f| md.add(f.to_string()))
    .collect();
    let version = md.add("i32 2, i32 9, i32 0".into());
    let ident = md.add("!\"metal2vulkan-devtools stitched module\"".into());
    let language = md.add("!\"Metal\", i32 4, i32 1, i32 0".into());
    let options: Vec<String> = [
        "denorms_disable",
        "fast_math_enable",
        "framebuffer_fetch_enable",
    ]
    .iter()
    .map(|c| md.add(format!("!\"air.compile.{c}\"")))
    .collect();

    // The visible entry and its stitching info.
    let output = md.add("!\"air.visible_output\", !\"air.arg_type_name\", !\"float\"".into());
    let outputs = md.add(output);
    let inputs: Vec<String> = (0..g.inputs)
        .map(|i| {
            md.add(format!(
                "i32 {i}, !\"air.visible_input\", !\"air.arg_type_name\", !\"float\""
            ))
        })
        .collect();
    let inputs = md.add(inputs.join(", "));
    let float = md.add("!\"air.float_type\", i32 4, i32 4, null".into());
    let types: Vec<String> = SETS
        .iter()
        .map(|set| {
            let helpers: Vec<String> = (0..HELPERS.len())
                .map(|i| format!("ptr {}", helper(i, set)))
                .collect();
            md.add(format!(
                "!\"air.stitching_type\", {float}, {}",
                helpers.join(", ")
            ))
        })
        .collect();
    let argument = md.add(format!("!\"air.stitching_argument\", {}", types[1]));
    let mut info = format!("!\"air.stitching_info\", {}", types[0]);
    for _ in 0..g.inputs {
        write!(info, ", {argument}").unwrap();
    }
    let info = md.add(info);
    let visible = md.add(format!("ptr @{}, {outputs}, {inputs}, {info}", g.name));

    // The graph, in post-order from the output node, and the stitchable functions it calls.
    let empty = md.add(String::new());
    let mut list = Vec::new();
    let mut inputs_at = vec![None; g.inputs as usize];
    let mut nodes_at = vec![None; g.nodes.len()];
    let mut functions: Vec<&str> = Vec::new();
    let output_node = g.nodes.len() - 1;
    let mut stack = vec![(output_node, false)];
    while let Some((k, ready)) = stack.pop() {
        if nodes_at[k].is_some() {
            continue;
        }
        let n = &g.nodes[k];
        if !ready {
            stack.push((k, true));
            for a in n.args.iter().rev() {
                if let Operand::Node(j) = *a {
                    stack.push((j, false));
                }
            }
            continue;
        }
        let mut args = Vec::new();
        for a in n.args {
            let at = match a {
                Operand::Input(i) => inputs_at[i as usize]
                    .get_or_insert_with(|| {
                        let r = md.add(format!(
                            "!\"air.stitching_input_node\", i32 {i}, {empty}, {empty}"
                        ));
                        list.push(r.clone());
                        r
                    })
                    .clone(),
                Operand::Node(j) => nodes_at[j].clone().expect("an operand node comes first"),
            };
            args.push(at);
        }
        let args = md.add(args.join(", "));
        let (label, last) = if k == output_node {
            ("node_output".to_string(), "true")
        } else {
            (format!("node_{k}"), "false")
        };
        let r = md.add(format!(
            "!\"air.stitching_function_node\", !\"{label}\", i1 {last}, !\"{}\", {args}, {empty}",
            n.function
        ));
        list.push(r.clone());
        nodes_at[k] = Some(r);
        if !functions.contains(&n.function) {
            functions.push(n.function);
        }
    }
    let list = md.add(list.join(", "));
    let signatures: Vec<String> = functions
        .iter()
        .map(|f| {
            let digest: String = Sha256::digest(f.as_bytes())
                .iter()
                .map(|b| format!("{b:02X}"))
                .collect();
            md.add(format!("!\"air.module_signature\", !\"{digest}\""))
        })
        .collect();
    let signatures = md.add(signatures.join(", "));
    let signature = md.add(format!(
        "!\"air.stitching_modules_signature\", {list}, {signatures}"
    ));

    writeln!(out, "!llvm.module.flags = !{{{}}}", flags.join(", ")).unwrap();
    writeln!(out, "!air.version = !{{{version}}}").unwrap();
    writeln!(out, "!llvm.ident = !{{{ident}}}").unwrap();
    writeln!(out, "!air.language_version = !{{{language}}}").unwrap();
    writeln!(out, "!air.compile_options = !{{{}}}", options.join(", ")).unwrap();
    writeln!(out, "!air.visible = !{{{visible}}}").unwrap();
    writeln!(out, "!air.signature = !{{{signature}}}\n").unwrap();
    for (i, body) in md.0.iter().enumerate() {
        writeln!(out, "!{i} = !{{{body}}}").unwrap();
    }
    out
}

/// The stitched library of `modules` (each stitched function's name and its module's bitcode,
/// as `llvm-as` assembles [`module`]'s text), in the layout Metal writes it: an executable
/// library, each function visible (`TYPE` 4) with its own module.
pub fn library(modules: &[(&str, &[u8])]) -> Vec<u8> {
    let functions: Vec<MtlbFunction<'_>> = modules
        .iter()
        .map(|&(name, bitcode)| MtlbFunction {
            name,
            type_code: 4,
            bitcode: wrap(bitcode),
            constants: Vec::new(),
            public: Vec::new(),
            tags: Vec::new(),
        })
        .collect();
    metallib(&functions)
}
