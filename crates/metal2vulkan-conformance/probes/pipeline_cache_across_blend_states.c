// What `vkCreateGraphicsPipelines` spends on MoltenVK when a family of render states shares one
// vertex+fragment pair and differs only in blend factors, and whether a pipeline cache shared by
// the family cuts it.
//
// THE QUESTION. A render pipeline making is attributed end to end elsewhere; the driver's
// `vkCreateGraphicsPipelines` is the largest single phase of it. A runtime that gives every making
// a `VkPipelineCache` of its own, seeded from a record named by the whole graphics key, gets 32
// records and 32 caches for 32 states over one shader pair. MoltenVK keys a converted stage by its SPIR-V and its conversion
// configuration (`MVKPipelineCache::getShaderLibraryImpl`), not by the pipeline's state, so a
// cache shared by the family SHOULD convert each stage once. This probe measures whether it does,
// and what share of the call that is.
//
// WHY A PROBE AND NOT A TEST. The instrument is MoltenVK's own per-activity performance tracking
// (`vkGetPerformanceStatisticsMVK`, `MVK_CONFIG_PERFORMANCE_TRACKING=1`), which is private API of
// one driver and must not enter a product path or a suite. It is also the only thing on this
// machine that separates SPIR-V-to-MSL conversion from Metal's MSL compile from Metal's pipeline
// compile: `VK_EXT_pipeline_creation_feedback` reports a cache hit per stage and nothing finer.
//
// WHAT IT PRINTS, per arrangement, in one process of its own:
//   - every `vkCreateGraphicsPipelines` call's wall time, and the creation feedback of the whole
//     pipeline and of each stage (`H` = APPLICATION_PIPELINE_CACHE_HIT, `-` = valid and no hit,
//     `?` = not valid);
//   - MoltenVK's activity counts and totals: `spirvToMSL`, `mslCompile`, `mslLoad`,
//     `mslCompress`, `mslDecompress`, `shaderLibraryFromCache`, `functionRetrieval`,
//     `functionSpecialization`, `pipelineCompile`, and the cache's own read/write/size.
//
// COUNTS ARE THE EVIDENCE, NOT THE MILLISECONDS. This machine runs other work; a count of
// conversions is immune to load and a duration is not. Read `spirvToMSL.count` and
// `pipelineCompile.count` first.
//
// METAL'S OWN CACHE IS A CONFOUNDER FOR THE DURATIONS AND NOT FOR THE COUNTS. `MTLCompilerService`
// keeps compiled MSL keyed by its source, across processes, so the second arrangement to compile
// the same MSL text may compile it faster than the first. Every arrangement therefore runs in its
// own process, the runner runs each one twice and reports both, and a duration here is what was
// read, never a verdict.
//
// ARRANGEMENTS (argv[1]):
//   none          32 calls, no pipeline cache at all.
//   per           32 calls, each with an EMPTY `VkPipelineCache` made and destroyed around it.
//                 This is the runtime's shape on a cold boot: one making, one cache, no record.
//   shared        32 calls against ONE empty `VkPipelineCache`. The hypothesis.
//   per-warm      two passes. Pass one is `per`, keeping each making's cache data. Pass two makes
//                 each pipeline again against a cache created FROM that making's own data. The
//                 runtime's shape on a warm boot, within one process.
//   shared-warm   two passes. Pass one is `shared`, keeping the one cache's data at the end. Pass
//                 two makes all 32 against one cache created from it. The proposal's warm shape.
//   batch         ONE call with all 32 create infos, no cache.
//   batch-shared  ONE call with all 32 create infos against one empty cache.
//   ab            `per`, then `shared`, then `per` AGAIN, in one process, printing the three
//                 means and the ratios. For a loaded machine, where an absolute microsecond figure
//                 says nothing: both arms meet whatever load is present, so the RATIO survives
//                 where neither absolute does. The second `per` is the control that makes the
//                 ratio readable at all: `MTLCompilerService` keeps compiled MSL keyed by its
//                 source across the whole process, so some of the second arm's gain is only that
//                 the first arm warmed that cache, and the third arm says how much. Read it as
//                 `shared` beating BOTH `per` arms, or not.
//
// A warm arrangement reports pass two's calls and the activity counts of BOTH passes, so the load
// of the seeded cache (`mslLoad`, and the `mslCompile` that MoltenVK's cache load forces: it
// compiles every shader the data holds at the load) is visible and attributed.
//
// argv[2] and argv[3], optional: paths to a vertex and a fragment `.spv`. Without them the probe
// uses the two modules embedded below. Conversion cost scales with the module, and the embedded
// fragment module is 92 words, so a run that wants the cost of a real fragment function must pass
// one. The modules the findings were taken with were built as:
//
//   # 92-word pair
//   printf '#version 450\nlayout(location=0) in vec4 pos;\nlayout(location=0) out vec4 col;\n
//     void main(){gl_Position=pos;col=pos*0.5+0.5;}\n' > small.vert
//   printf '#version 450\nlayout(location=0) in vec4 col;\nlayout(location=0) out vec4 out0;\n
//     void main(){out0=col;}\n' > small.frag
//   glslangValidator -V -o small.vert.spv small.vert
//   glslangValidator -V -o small.frag.spv small.frag
//
//   # 88235-word fragment module: the same, with 2000 statements of
//   #   a = fma(a, vec4((i%7)+1.0), sin(a.yzwx * ((i%11)+1.5))) * 0.5;
//   # between the input and the output.
//
// WHAT IT READ, on an Apple M3 Max, macOS 27.0.1, MoltenVK 1.4.2 (driverVersion 0x28a2,
// apiVersion 1.2.357), over 32 blend states of one function pair, with other work on the machine.
// The driver lists `VK_EXT_pipeline_creation_feedback` and does NOT list
// `VK_EXT_graphics_pipeline_library`, so the runtime's split-and-link path cannot run here and
// every making is a whole pipeline.
//
// By conversion COUNT, which no load moves, reproduced across runs and in both module sizes:
//
//   arrangement   spirvToMSL   mslCompile   pipelineCompile   stage feedback
//   none                   2            3                32   no hit, ever
//   shared                 2            3                32   a hit on 31 of 32
//   per                   64           65                32   no hit, ever
//   batch                  2            3                32   no hit, ever
//
// So MoltenVK converts a stage ONCE for the whole family when it is handed no pipeline cache at
// all, and once when it is handed one cache for the family; it converts the stage AGAIN FOR EVERY
// MAKING when each making brings a cache of its own. A converted stage is reused out of the
// shader module's own library cache where no pipeline cache is given, and out of the pipeline
// cache where one is, so a cache that lives for the length of one call carries nothing to the
// next making and the arrangement defeats the reuse it was meant to add. `shaderLibraryFromCache`
// counts 62 of the 64 stage lookups in `none` and `shared`, and 0 in `per`, which is the same
// fact from the driver's other side.
//
// `pipelineCompile` is 32 in every arrangement: Metal builds a distinct `MTLRenderPipelineState`
// for each of the 32 blend states whatever any cache holds, and no cache of ours can touch it.
// `whole` feedback is never a hit on this driver, only ever valid, which agrees with what
// `ShaderCacheStatistics::driver_pipeline_hits` already documents.
//
// Batching changes nothing: one call of 32 infos reads the same counts as 32 calls, so the driver
// amortises nothing across an array.
//
// A RECORD OF THE FAMILY IS THE SIZE OF A RECORD OF ONE MAKING. `vkGetPipelineCacheData` after 32
// makings into one cache is 1522 bytes at 92 fragment words -- byte for byte the size of a single
// making's record -- and 109902 bytes at 88235 words. The data holds the converted shader
// libraries and no pipeline state at all, so sharing a record across a family grows it not at
// all, and MoltenVK's compile of every shader a saved cache holds at its load costs such a record
// the compile of its FAMILY'S DISTINCT STAGES, not of its pipelines: a seeded load of the shared record was
// 2 `mslCompile`s and 50 us, where the 32 per-making records were 64 `mslCompile`s and 2639 us.
//
// TIMES READ, which are not a verdict: the machine was loaded (another session's CPU stress loop,
// load average 14 to 33), and these are what was read, back to back in one batch so both arms met
// the same load. At 92 fragment words, 128 us a call with no cache, 127 us shared, 225 us per
// making. At 88235 words, 911 us shared against 9550 us per making, a ratio of 10.5. An
// in-process A/B of the same thing (`ab`) read shared at 0.497 of a second `per` arm at 92 words.
//
// ONE ANOMALY THIS PROBE DOES NOT EXPLAIN, recorded so no later reader mistakes it for an
// arrangement's cost: the FIRST process of a batch to make these 32 pipelines over the
// 88235-word module spends about 8.3 SECONDS a pipeline in `pipelineCompile`, and every later
// process in the same batch spends about 1 ms. It followed the first process and never the
// arrangement -- it landed on `none` once and on `shared` twice -- and it reappeared after a
// minutes-long gap, so it is neither a one-off nor a permanent warm cache. Discard the first
// arrangement of any batch over a module this size.
//
// BOTH OF THOSE LAST TWO PARAGRAPHS HAVE SINCE BEEN CHECKED ON METAL'S OWN API, with no Vulkan in
// the process, by `metal_pipeline_marginal_cost.m`. Read it for the figures; what it changes here:
//
//   - `pipelineCompile` BEING 32 IS THE FLOOR, and now for a stated reason. Metal recompiles the
//     fragment stage for every distinct blend state: 32 distinct blend states over one function
//     pair cost 32 fragment recompiles through `newRenderPipelineStateWithDescriptor:` too, at 3.1
//     to 8.6 SECONDS each over a 2000-statement fragment against 27 us over a trivial one. The cost
//     follows the fragment function, which is what a recompile does. Blending reaches an Apple
//     GPU's fragment epilogue, so it is part of the fragment program. The only cheap answer Metal
//     has for a family is an IDENTICAL descriptor, which it memoises at 5 to 11 us.
//   - THE 8.3-SECOND FIRST-PROCESS ANOMALY IS EXPLAINED AND IS NOT MOLTENVK'S. MTLCompilerService
//     keys its compiled output on the AIR and keeps it ACROSS PROCESSES. Every process of a batch
//     here hands the driver the same SPIR-V, so the same MSL and the same AIR: the first pays real
//     compiles and every later one reads that cache. The sibling probe isolates the mechanism by a
//     one-character change -- its nonce in a live constant against a comment, nothing else
//     differing -- and reads a distinct blend state at 4052 us cold against 27 us warm over a
//     trivial fragment, 150x, where this probe's 8.3 s against 1 ms is 8000x over a module 960
//     times larger. So it is not an anomaly to discard: it is a cache, and a probe that wants a
//     COLD Metal compile must vary a live constant, not a comment.
// BUILD AND RUN:
//   clang -O2 -o /tmp/pcab \
//     -I/opt/homebrew/opt/molten-vk/include -I/opt/homebrew/opt/vulkan-headers/include \
//     crates/metal2vulkan-conformance/probes/pipeline_cache_across_blend_states.c \
//     -L/opt/homebrew/opt/molten-vk/lib -lMoltenVK
//   MVK_CONFIG_PERFORMANCE_TRACKING=1 /tmp/pcab shared vert.spv frag.spv
//
// The 32 states are 32 distinct (srcColorBlendFactor, dstColorBlendFactor) pairs with blending
// enabled, and nothing else differs: the same modules, layout, render pass, dynamic state and
// every other piece of state. Metal must build 32 distinct `MTLRenderPipelineState`s for them
// whatever the cache holds; what a cache can save is the conversion and the MSL compile that
// stand in front of those.

#include <MoltenVK/mvk_private_api.h>
#include <vulkan/vulkan.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define STATES 32
#define STAGES 2

// A trivial vertex module: `gl_Position` and one varying from one `vec4` input
// (`small.vert` above), as glslangValidator emits it.
static const uint32_t VERT_SPV[] = {
    0x07230203, 0x00010000, 0x0008000b, 0x0000001b, 0x00000000, 0x00020011, 0x00000001,
    0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e,
    0x00000000, 0x00000001, 0x0008000f, 0x00000000, 0x00000004, 0x6e69616d, 0x00000000,
    0x0000000d, 0x00000011, 0x00000015, 0x00030003, 0x00000002, 0x000001c2, 0x00040005,
    0x00000004, 0x6e69616d, 0x00000000, 0x00060005, 0x0000000b, 0x505f6c67, 0x65567265,
    0x78657472, 0x00000000, 0x00060006, 0x0000000b, 0x00000000, 0x505f6c67, 0x7469736f,
    0x006e6f69, 0x00070006, 0x0000000b, 0x00000001, 0x505f6c67, 0x746e696f, 0x657a6953,
    0x00000000, 0x00070006, 0x0000000b, 0x00000002, 0x435f6c67, 0x4470696c, 0x61747369,
    0x0065636e, 0x00070006, 0x0000000b, 0x00000003, 0x435f6c67, 0x446c6c75, 0x61747369,
    0x0065636e, 0x00030005, 0x0000000d, 0x00000000, 0x00030005, 0x00000011, 0x00736f70,
    0x00030005, 0x00000015, 0x006c6f63, 0x00030047, 0x0000000b, 0x00000002, 0x00050048,
    0x0000000b, 0x00000000, 0x0000000b, 0x00000000, 0x00050048, 0x0000000b, 0x00000001,
    0x0000000b, 0x00000001, 0x00050048, 0x0000000b, 0x00000002, 0x0000000b, 0x00000003,
    0x00050048, 0x0000000b, 0x00000003, 0x0000000b, 0x00000004, 0x00040047, 0x00000011,
    0x0000001e, 0x00000000, 0x00040047, 0x00000015, 0x0000001e, 0x00000000, 0x00020013,
    0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006, 0x00000020,
    0x00040017, 0x00000007, 0x00000006, 0x00000004, 0x00040015, 0x00000008, 0x00000020,
    0x00000000, 0x0004002b, 0x00000008, 0x00000009, 0x00000001, 0x0004001c, 0x0000000a,
    0x00000006, 0x00000009, 0x0006001e, 0x0000000b, 0x00000007, 0x00000006, 0x0000000a,
    0x0000000a, 0x00040020, 0x0000000c, 0x00000003, 0x0000000b, 0x0004003b, 0x0000000c,
    0x0000000d, 0x00000003, 0x00040015, 0x0000000e, 0x00000020, 0x00000001, 0x0004002b,
    0x0000000e, 0x0000000f, 0x00000000, 0x00040020, 0x00000010, 0x00000001, 0x00000007,
    0x0004003b, 0x00000010, 0x00000011, 0x00000001, 0x00040020, 0x00000013, 0x00000003,
    0x00000007, 0x0004003b, 0x00000013, 0x00000015, 0x00000003, 0x0004002b, 0x00000006,
    0x00000017, 0x3f000000, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003,
    0x000200f8, 0x00000005, 0x0004003d, 0x00000007, 0x00000012, 0x00000011, 0x00050041,
    0x00000013, 0x00000014, 0x0000000d, 0x0000000f, 0x0003003e, 0x00000014, 0x00000012,
    0x0004003d, 0x00000007, 0x00000016, 0x00000011, 0x0005008e, 0x00000007, 0x00000018,
    0x00000016, 0x00000017, 0x00070050, 0x00000007, 0x00000019, 0x00000017, 0x00000017,
    0x00000017, 0x00000017, 0x00050081, 0x00000007, 0x0000001a, 0x00000018, 0x00000019,
    0x0003003e, 0x00000015, 0x0000001a, 0x000100fd, 0x00010038,
};

// A trivial fragment module: one varying out of one varying in (`small.frag` above).
static const uint32_t FRAG_SPV[] = {
    0x07230203, 0x00010000, 0x0008000b, 0x0000000d, 0x00000000, 0x00020011, 0x00000001,
    0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e,
    0x00000000, 0x00000001, 0x0007000f, 0x00000004, 0x00000004, 0x6e69616d, 0x00000000,
    0x00000009, 0x0000000b, 0x00030010, 0x00000004, 0x00000007, 0x00030003, 0x00000002,
    0x000001c2, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000, 0x00040005, 0x00000009,
    0x3074756f, 0x00000000, 0x00030005, 0x0000000b, 0x006c6f63, 0x00040047, 0x00000009,
    0x0000001e, 0x00000000, 0x00040047, 0x0000000b, 0x0000001e, 0x00000000, 0x00020013,
    0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006, 0x00000020,
    0x00040017, 0x00000007, 0x00000006, 0x00000004, 0x00040020, 0x00000008, 0x00000003,
    0x00000007, 0x0004003b, 0x00000008, 0x00000009, 0x00000003, 0x00040020, 0x0000000a,
    0x00000001, 0x00000007, 0x0004003b, 0x0000000a, 0x0000000b, 0x00000001, 0x00050036,
    0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0004003d,
    0x00000007, 0x0000000c, 0x0000000b, 0x0003003e, 0x00000009, 0x0000000c, 0x000100fd,
    0x00010038,
};

#define CHECK(expr)                                                                  \
    do {                                                                             \
        VkResult _r = (expr);                                                         \
        if (_r != VK_SUCCESS) {                                                       \
            fprintf(stderr, "%s:%d: %s = %d\n", __FILE__, __LINE__, #expr, (int)_r); \
            exit(2);                                                                  \
        }                                                                             \
    } while (0)

static double now_us(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e6 + (double)t.tv_nsec / 1e3;
}

/// The words of `path`, or exits. The caller owns the allocation.
static uint32_t *read_spv(const char *path, size_t *words) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(2);
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size <= 0 || size % 4 != 0) {
        fprintf(stderr, "%s is %ld bytes, not whole SPIR-V words\n", path, size);
        exit(2);
    }
    uint32_t *code = malloc((size_t)size);
    if (fread(code, 1, (size_t)size, f) != (size_t)size) {
        fprintf(stderr, "short read of %s\n", path);
        exit(2);
    }
    fclose(f);
    *words = (size_t)size / 4;
    return code;
}

/// The feedback flag of `f` as one character: `H` a hit, `-` valid and no hit, `?` not valid.
static char feedback_char(VkPipelineCreationFeedback f) {
    if (!(f.flags & VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT)) {
        return '?';
    }
    return (f.flags & VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT) ? 'H' : '-';
}

/// Everything the makings share.
struct Fixture {
    VkDevice device;
    VkPipelineLayout layout;
    VkRenderPass pass;
    VkShaderModule vert;
    VkShaderModule frag;
    int feedback;
};

/// The create info of state `at`, with its own feedback chain. Every pointer it holds lives in
/// `state`, which the caller keeps alive across the call.
struct State {
    VkPipelineShaderStageCreateInfo stages[STAGES];
    VkVertexInputBindingDescription binding;
    VkVertexInputAttributeDescription attribute;
    VkPipelineVertexInputStateCreateInfo vertex;
    VkPipelineInputAssemblyStateCreateInfo assembly;
    VkViewport viewport;
    VkRect2D scissor;
    VkPipelineViewportStateCreateInfo viewports;
    VkPipelineRasterizationStateCreateInfo raster;
    VkPipelineMultisampleStateCreateInfo multisample;
    VkPipelineDepthStencilStateCreateInfo depth_stencil;
    VkPipelineColorBlendAttachmentState attachment;
    VkPipelineColorBlendStateCreateInfo blend;
    VkPipelineCreationFeedback whole;
    VkPipelineCreationFeedback stage_feedback[STAGES];
    VkPipelineCreationFeedbackCreateInfo feedback;
    VkGraphicsPipelineCreateInfo info;
};

/// Fills `state` with state `at`'s pipeline: the fixture's modules, layout and pass, and the
/// `at`th of 32 (source factor, destination factor) pairs. Nothing else differs between states.
static void fill_state(struct State *state, const struct Fixture *fixture, unsigned at) {
    memset(state, 0, sizeof(*state));
    state->stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    state->stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    state->stages[0].module = fixture->vert;
    state->stages[0].pName = "main";
    state->stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    state->stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    state->stages[1].module = fixture->frag;
    state->stages[1].pName = "main";

    // The vertex module reads one `vec4` at location 0, so the pipeline declares a binding and
    // an attribute for it: Metal refuses to compile a vertex function whose attribute the
    // descriptor does not declare, and the probe is not about that.
    state->binding.binding = 0;
    state->binding.stride = 16;
    state->binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    state->attribute.location = 0;
    state->attribute.binding = 0;
    state->attribute.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    state->attribute.offset = 0;
    state->vertex.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    state->vertex.vertexBindingDescriptionCount = 1;
    state->vertex.pVertexBindingDescriptions = &state->binding;
    state->vertex.vertexAttributeDescriptionCount = 1;
    state->vertex.pVertexAttributeDescriptions = &state->attribute;
    state->assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    state->assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    state->viewport.width = 64.0f;
    state->viewport.height = 64.0f;
    state->viewport.maxDepth = 1.0f;
    state->scissor.extent.width = 64;
    state->scissor.extent.height = 64;
    state->viewports.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    state->viewports.viewportCount = 1;
    state->viewports.pViewports = &state->viewport;
    state->viewports.scissorCount = 1;
    state->viewports.pScissors = &state->scissor;

    state->raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    state->raster.polygonMode = VK_POLYGON_MODE_FILL;
    state->raster.cullMode = VK_CULL_MODE_NONE;
    state->raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    state->raster.lineWidth = 1.0f;

    state->multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    state->multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    state->depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;

    // The one axis the family varies: eight source factors against four destination factors.
    // Every value is below `VK_BLEND_FACTOR_SRC1_COLOR`, so none asks for dual-source blending.
    state->attachment.blendEnable = VK_TRUE;
    state->attachment.srcColorBlendFactor = (VkBlendFactor)(at % 8);
    state->attachment.dstColorBlendFactor = (VkBlendFactor)(at / 8 + 1);
    state->attachment.colorBlendOp = VK_BLEND_OP_ADD;
    state->attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    state->attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    state->attachment.alphaBlendOp = VK_BLEND_OP_ADD;
    state->attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                       VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    state->blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    state->blend.attachmentCount = 1;
    state->blend.pAttachments = &state->attachment;

    state->feedback.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO;
    state->feedback.pPipelineCreationFeedback = &state->whole;
    state->feedback.pipelineStageCreationFeedbackCount = STAGES;
    state->feedback.pPipelineStageCreationFeedbacks = state->stage_feedback;

    state->info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    state->info.stageCount = STAGES;
    state->info.pStages = state->stages;
    state->info.pVertexInputState = &state->vertex;
    state->info.pInputAssemblyState = &state->assembly;
    state->info.pViewportState = &state->viewports;
    state->info.pRasterizationState = &state->raster;
    state->info.pMultisampleState = &state->multisample;
    state->info.pDepthStencilState = &state->depth_stencil;
    state->info.pColorBlendState = &state->blend;
    state->info.layout = fixture->layout;
    state->info.renderPass = fixture->pass;
    state->info.subpass = 0;
    if (fixture->feedback) {
        state->info.pNext = &state->feedback;
    }
}

/// One making's reading.
struct Reading {
    double us;
    char whole;
    char stages[STAGES];
};

static void print_readings(const char *what, const struct Reading *readings, unsigned n) {
    double total = 0.0;
    printf("%-14s", what);
    for (unsigned i = 0; i < n; i++) {
        total += readings[i].us;
    }
    printf(" calls %u  total %.0f us  mean %.0f us\n", n, total, total / (double)n);
    for (unsigned i = 0; i < n; i++) {
        printf("  [%2u] %8.0f us  whole %c  vert %c  frag %c\n", i, readings[i].us,
               readings[i].whole, readings[i].stages[0], readings[i].stages[1]);
    }
}

static void print_tracker(const char *name, MVKPerformanceTracker t) {
    printf("  %-24s count %6u  total %9.2f ms  mean %7.3f ms  max %7.3f ms\n", name, t.count,
           (double)t.count * t.average, t.average, t.maximum);
}

static void print_statistics(VkDevice device, const char *when) {
    // Called directly, not through `vkGetDeviceProcAddr`: MoltenVK exports the symbol but does
    // not return it from either `get*ProcAddr`, so the probe links the dylib itself. That is also
    // why the probe links `-lMoltenVK` rather than going through the Vulkan loader.
    MVKPerformanceStatistics perf;
    memset(&perf, 0, sizeof(perf));
    size_t size = sizeof(perf);
    VkResult r = vkGetPerformanceStatisticsMVK(device, &perf, &size);
    if (r != VK_SUCCESS && r != VK_INCOMPLETE) {
        printf("%s: vkGetPerformanceStatisticsMVK = %d\n", when, (int)r);
        return;
    }
    printf("%s (MVKPerformanceStatistics, %s):\n", when,
           r == VK_INCOMPLETE ? "struct sizes differ, fields read up to the smaller" : "whole");
    print_tracker("hashShaderCode", perf.shaderCompilation.hashShaderCode);
    print_tracker("spirvToMSL", perf.shaderCompilation.spirvToMSL);
    print_tracker("mslCompile", perf.shaderCompilation.mslCompile);
    print_tracker("mslLoad", perf.shaderCompilation.mslLoad);
    print_tracker("mslCompress", perf.shaderCompilation.mslCompress);
    print_tracker("mslDecompress", perf.shaderCompilation.mslDecompress);
    print_tracker("shaderLibraryFromCache", perf.shaderCompilation.shaderLibraryFromCache);
    print_tracker("functionRetrieval", perf.shaderCompilation.functionRetrieval);
    print_tracker("functionSpecialization", perf.shaderCompilation.functionSpecialization);
    print_tracker("pipelineCompile", perf.shaderCompilation.pipelineCompile);
    print_tracker("sizePipelineCache", perf.pipelineCache.sizePipelineCache);
    print_tracker("writePipelineCache", perf.pipelineCache.writePipelineCache);
    print_tracker("readPipelineCache", perf.pipelineCache.readPipelineCache);
}

/// Makes state `at` against `cache`, destroys the pipeline, and reads the call.
static struct Reading make_one(const struct Fixture *fixture, unsigned at, VkPipelineCache cache) {
    struct State state;
    fill_state(&state, fixture, at);
    VkPipeline pipeline = VK_NULL_HANDLE;
    double started = now_us();
    CHECK(vkCreateGraphicsPipelines(fixture->device, cache, 1, &state.info, NULL, &pipeline));
    double us = now_us() - started;
    vkDestroyPipeline(fixture->device, pipeline, NULL);
    struct Reading reading = {us, feedback_char(state.whole),
                              {feedback_char(state.stage_feedback[0]),
                               feedback_char(state.stage_feedback[1])}};
    return reading;
}

/// Makes every state in ONE call against `cache` and reads the call once per state: the wall time
/// is the whole call's, repeated, and the feedback is each state's own.
static void make_batch(const struct Fixture *fixture, VkPipelineCache cache,
                       struct Reading *readings) {
    struct State states[STATES];
    VkGraphicsPipelineCreateInfo infos[STATES];
    VkPipeline pipelines[STATES];
    for (unsigned at = 0; at < STATES; at++) {
        fill_state(&states[at], fixture, at);
        infos[at] = states[at].info;
    }
    double started = now_us();
    CHECK(vkCreateGraphicsPipelines(fixture->device, cache, STATES, infos, NULL, pipelines));
    double us = now_us() - started;
    printf("one call of %u infos: %.0f us whole, %.0f us a state\n", STATES, us,
           us / (double)STATES);
    for (unsigned at = 0; at < STATES; at++) {
        vkDestroyPipeline(fixture->device, pipelines[at], NULL);
        readings[at].us = us / (double)STATES;
        readings[at].whole = feedback_char(states[at].whole);
        readings[at].stages[0] = feedback_char(states[at].stage_feedback[0]);
        readings[at].stages[1] = feedback_char(states[at].stage_feedback[1]);
    }
}

/// The mean of `n` readings, in microseconds.
static double mean_us(const struct Reading *readings, unsigned n) {
    double total = 0.0;
    for (unsigned i = 0; i < n; i++) {
        total += readings[i].us;
    }
    return total / (double)n;
}

/// An empty cache.
static VkPipelineCache empty_cache(VkDevice device) {
    VkPipelineCacheCreateInfo info = {.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    VkPipelineCache cache = VK_NULL_HANDLE;
    CHECK(vkCreatePipelineCache(device, &info, NULL, &cache));
    return cache;
}

/// A cache of `bytes`, timed: the load is what compiling the data's shaders at load costs.
static VkPipelineCache seeded_cache(VkDevice device, const void *bytes, size_t size, double *us) {
    VkPipelineCacheCreateInfo info = {.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    info.initialDataSize = size;
    info.pInitialData = bytes;
    VkPipelineCache cache = VK_NULL_HANDLE;
    double started = now_us();
    CHECK(vkCreatePipelineCache(device, &info, NULL, &cache));
    *us = now_us() - started;
    return cache;
}

/// The data of `cache`. The caller owns the allocation.
static void *cache_data(VkDevice device, VkPipelineCache cache, size_t *size) {
    CHECK(vkGetPipelineCacheData(device, cache, size, NULL));
    void *bytes = malloc(*size ? *size : 1);
    CHECK(vkGetPipelineCacheData(device, cache, size, bytes));
    return bytes;
}

int main(int argc, char **argv) {
    const char *arrangement = argc > 1 ? argv[1] : "per";
    const uint32_t *vert_code = VERT_SPV;
    size_t vert_words = sizeof(VERT_SPV) / 4;
    const uint32_t *frag_code = FRAG_SPV;
    size_t frag_words = sizeof(FRAG_SPV) / 4;
    if (argc > 3) {
        vert_code = read_spv(argv[2], &vert_words);
        frag_code = read_spv(argv[3], &frag_words);
    }
    printf("arrangement %s, vertex %zu words, fragment %zu words\n", arrangement, vert_words,
           frag_words);

    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo instance_info = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &app;
    VkInstance instance = VK_NULL_HANDLE;
    CHECK(vkCreateInstance(&instance_info, NULL, &instance));

    uint32_t count = 1;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkResult enumerated = vkEnumeratePhysicalDevices(instance, &count, &physical);
    if ((enumerated != VK_SUCCESS && enumerated != VK_INCOMPLETE) || count == 0) {
        fprintf(stderr, "no physical device\n");
        return 2;
    }
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(physical, &properties);
    printf("device %s, driverVersion 0x%x, apiVersion %u.%u.%u\n", properties.deviceName,
           properties.driverVersion, VK_API_VERSION_MAJOR(properties.apiVersion),
           VK_API_VERSION_MINOR(properties.apiVersion),
           VK_API_VERSION_PATCH(properties.apiVersion));

    // The creation-feedback extension, which the readings need; a device without it reports `?`.
    uint32_t extensions = 0;
    CHECK(vkEnumerateDeviceExtensionProperties(physical, NULL, &extensions, NULL));
    VkExtensionProperties *available = malloc(extensions * sizeof(*available));
    CHECK(vkEnumerateDeviceExtensionProperties(physical, NULL, &extensions, available));
    int has_feedback = 0;
    int has_library = 0;
    for (uint32_t i = 0; i < extensions; i++) {
        if (strcmp(available[i].extensionName, VK_EXT_PIPELINE_CREATION_FEEDBACK_EXTENSION_NAME) ==
            0) {
            has_feedback = 1;
        }
        if (strcmp(available[i].extensionName, "VK_EXT_graphics_pipeline_library") == 0) {
            has_library = 1;
        }
    }
    printf("VK_EXT_pipeline_creation_feedback %s, VK_EXT_graphics_pipeline_library %s\n",
           has_feedback ? "yes" : "no", has_library ? "yes" : "no");

    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue.queueCount = 1;
    queue.pQueuePriorities = &priority;
    const char *enabled[1] = {VK_EXT_PIPELINE_CREATION_FEEDBACK_EXTENSION_NAME};
    VkDeviceCreateInfo device_info = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue;
    device_info.enabledExtensionCount = has_feedback ? 1 : 0;
    device_info.ppEnabledExtensionNames = enabled;
    VkDevice device = VK_NULL_HANDLE;
    CHECK(vkCreateDevice(physical, &device_info, NULL, &device));
    free(available);

    VkShaderModuleCreateInfo module_info = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    module_info.codeSize = vert_words * 4;
    module_info.pCode = vert_code;
    VkShaderModule vert = VK_NULL_HANDLE;
    CHECK(vkCreateShaderModule(device, &module_info, NULL, &vert));
    module_info.codeSize = frag_words * 4;
    module_info.pCode = frag_code;
    VkShaderModule frag = VK_NULL_HANDLE;
    CHECK(vkCreateShaderModule(device, &module_info, NULL, &frag));

    VkPipelineLayoutCreateInfo layout_info = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    VkPipelineLayout layout = VK_NULL_HANDLE;
    CHECK(vkCreatePipelineLayout(device, &layout_info, NULL, &layout));

    VkAttachmentDescription attachment = {0};
    attachment.format = VK_FORMAT_B8G8R8A8_UNORM;
    attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkAttachmentReference reference = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass = {0};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &reference;
    VkRenderPassCreateInfo pass_info = {.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    pass_info.attachmentCount = 1;
    pass_info.pAttachments = &attachment;
    pass_info.subpassCount = 1;
    pass_info.pSubpasses = &subpass;
    VkRenderPass pass = VK_NULL_HANDLE;
    CHECK(vkCreateRenderPass(device, &pass_info, NULL, &pass));

    struct Fixture fixture = {device, layout, pass, vert, frag, has_feedback};
    struct Reading readings[STATES];

    if (strcmp(arrangement, "none") == 0) {
        for (unsigned at = 0; at < STATES; at++) {
            readings[at] = make_one(&fixture, at, VK_NULL_HANDLE);
        }
        print_readings("none", readings, STATES);
    } else if (strcmp(arrangement, "per") == 0) {
        for (unsigned at = 0; at < STATES; at++) {
            VkPipelineCache cache = empty_cache(device);
            readings[at] = make_one(&fixture, at, cache);
            vkDestroyPipelineCache(device, cache, NULL);
        }
        print_readings("per", readings, STATES);
    } else if (strcmp(arrangement, "shared") == 0) {
        VkPipelineCache cache = empty_cache(device);
        for (unsigned at = 0; at < STATES; at++) {
            readings[at] = make_one(&fixture, at, cache);
        }
        print_readings("shared", readings, STATES);
        size_t size = 0;
        void *bytes = cache_data(device, cache, &size);
        printf("the shared cache's data after 32 makings: %zu bytes\n", size);
        free(bytes);
        vkDestroyPipelineCache(device, cache, NULL);
    } else if (strcmp(arrangement, "per-warm") == 0) {
        // Pass one: each making against an empty cache of its own, keeping its data.
        void *kept[STATES];
        size_t sizes[STATES];
        for (unsigned at = 0; at < STATES; at++) {
            VkPipelineCache cache = empty_cache(device);
            (void)make_one(&fixture, at, cache);
            kept[at] = cache_data(device, cache, &sizes[at]);
            vkDestroyPipelineCache(device, cache, NULL);
        }
        printf("pass one kept 32 records, %zu bytes the first\n", sizes[0]);
        print_statistics(device, "after pass one");
        // Pass two: each making against a cache made from its own record.
        double load_us = 0.0;
        size_t load_bytes = 0;
        for (unsigned at = 0; at < STATES; at++) {
            double one = 0.0;
            VkPipelineCache cache = seeded_cache(device, kept[at], sizes[at], &one);
            load_us += one;
            load_bytes += sizes[at];
            readings[at] = make_one(&fixture, at, cache);
            vkDestroyPipelineCache(device, cache, NULL);
            free(kept[at]);
        }
        printf("pass two loaded 32 records, %zu bytes in all, %.0f us in vkCreatePipelineCache\n",
               load_bytes, load_us);
        print_readings("per-warm", readings, STATES);
    } else if (strcmp(arrangement, "shared-warm") == 0) {
        // Pass one: all 32 against one empty cache, keeping its data once.
        VkPipelineCache cache = empty_cache(device);
        for (unsigned at = 0; at < STATES; at++) {
            (void)make_one(&fixture, at, cache);
        }
        size_t size = 0;
        void *bytes = cache_data(device, cache, &size);
        vkDestroyPipelineCache(device, cache, NULL);
        printf("pass one kept one record of %zu bytes\n", size);
        print_statistics(device, "after pass one");
        // Pass two: all 32 against one cache made from that one record.
        double load_us = 0.0;
        VkPipelineCache warm = seeded_cache(device, bytes, size, &load_us);
        printf("pass two loaded one record of %zu bytes, %.0f us in vkCreatePipelineCache\n", size,
               load_us);
        for (unsigned at = 0; at < STATES; at++) {
            readings[at] = make_one(&fixture, at, warm);
        }
        print_readings("shared-warm", readings, STATES);
        vkDestroyPipelineCache(device, warm, NULL);
        free(bytes);
    } else if (strcmp(arrangement, "ab") == 0) {
        struct Reading first[STATES];
        struct Reading second[STATES];
        struct Reading third[STATES];
        for (unsigned at = 0; at < STATES; at++) {
            VkPipelineCache cache = empty_cache(device);
            first[at] = make_one(&fixture, at, cache);
            vkDestroyPipelineCache(device, cache, NULL);
        }
        print_readings("per (first)", first, STATES);
        print_statistics(device, "after per (first)");
        VkPipelineCache shared = empty_cache(device);
        for (unsigned at = 0; at < STATES; at++) {
            second[at] = make_one(&fixture, at, shared);
        }
        vkDestroyPipelineCache(device, shared, NULL);
        print_readings("shared", second, STATES);
        print_statistics(device, "after shared");
        for (unsigned at = 0; at < STATES; at++) {
            VkPipelineCache cache = empty_cache(device);
            third[at] = make_one(&fixture, at, cache);
            vkDestroyPipelineCache(device, cache, NULL);
        }
        print_readings("per (again)", third, STATES);
        double a = mean_us(first, STATES);
        double b = mean_us(second, STATES);
        double c = mean_us(third, STATES);
        printf("ratios: shared/per-first %.3f, shared/per-again %.3f, per-again/per-first %.3f\n",
               b / a, b / c, c / a);
        printf("(per-again/per-first near 1 means the process's own warming did not carry; far\n"
               " below 1 means it did, and shared/per-again is then the arm to read.)\n");
    } else if (strcmp(arrangement, "batch") == 0) {
        make_batch(&fixture, VK_NULL_HANDLE, readings);
        print_readings("batch", readings, STATES);
    } else if (strcmp(arrangement, "batch-shared") == 0) {
        VkPipelineCache cache = empty_cache(device);
        make_batch(&fixture, cache, readings);
        print_readings("batch-shared", readings, STATES);
        vkDestroyPipelineCache(device, cache, NULL);
    } else {
        fprintf(stderr,
                "arrangements: none per shared per-warm shared-warm ab batch batch-shared\n");
        return 2;
    }

    print_statistics(device, "at the end");

    vkDestroyRenderPass(device, pass, NULL);
    vkDestroyPipelineLayout(device, layout, NULL);
    vkDestroyShaderModule(device, frag, NULL);
    vkDestroyShaderModule(device, vert, NULL);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
    return 0;
}
