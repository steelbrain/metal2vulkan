# Resource IDs

A Metal texture's or sampler's `gpuResourceID` is a 64-bit value that a shader reads from an
argument buffer or a buffer of its own and then samples, reads or writes through. Under the
translator's identity binding domains the ID is computed on the guest's side, with no round trip
to the runtime, from the index the consumer grants and the object's Metal descriptor:

```text
gpuResourceID = index | word << 32
```

The low 32 bits are the index into the pipeline's heap arrays (set 1, [API.md](API.md) section 5);
the high 32 bits are the word, which carries what a shader reads of the object beyond its
descriptor. Both are pure functions of what the guest already holds, so the same program produces
the same IDs on every run, and a replay reproduces them.

## Index classes

| Index | Object | Arrays it is valid in |
|---|---|---|
| 0 | nil (the ID 0) | none |
| 1 to 4,095 | a storage-capable texture: usage with `MTLTextureUsageShaderWrite`, or `MTLTextureUsageUnknown` | the sampled-image array and the storage-image array |
| 4,096 to 65,535 | any other texture | the sampled-image array |
| 1 to 3,999 | a sampler | the sampler array |

A texture view takes its own index, in the class of its own usage. A texture buffer has no index:
it is a typed refusal under these domains.

## The arrays

Under the identity domains set 1 has one layout for every pipeline: binding 0 is the sampled-image
array of 65,536 elements, binding 1 the storage-image array of 4,096, and bindings 2, 3 and 4 the
sampler arrays of 4,000 elements each, every one partially bound, updated after binding and while
unused elements are pending, and seen by every stage. Element `i` of each image array is the
texture whose ID holds index `i`, so the consumer writes an element once, as it makes the texture,
for every pipeline. Element 0 is a null descriptor (`nullDescriptor`): a nil
texture sizes 0, reads and samples zero, and drops a store, as Metal answers for a nil texture, with
no code in the shader. Each texture class a shader declares (its shape, numeric type and depth
comparison, sampled, or storage without a format) is a variable of its descriptor type's binding.

A shader takes a texture slot's ID from the slot's push field (`PushField::SlotGuestBase`), and an
argument buffer's texture's ID from the member's 8 bytes, with no table between. It indexes the
arrays with the low word, answers `is_null_texture` by whether the index is 0, and,
where it checks a `read` itself (a device without `robustImageAccess2`), takes an out-of-bounds
alpha of 1 from bit 12 of the word.

Element `i` of each sampler array is a device sampler made of the sampler whose ID holds index `i`,
one per use, which the shader picks by the instruction and the texture's numeric type: binding 2
for a plain sample, 3 for a depth comparison, and 4 for a texture of an integer format, every
filter nearest. The consumer makes each from the descriptor (its reduction mode and anisotropy
included); it makes none of a descriptor whose min or mag filter is cubic. A shader takes a
runtime sampler slot's ID from the slot's push field, samples through the element its low word
indexes, and reads the word's fields where a sample depends on them (pixel coordinates, a nearest
pixel sampler). A constexpr sampler and an argument buffer's sampler are not lowered under these
domains yet.

## The texture word

`metal2vulkan::metal::resource_word::TextureWord::of(pixelFormat, swizzle)`:

| Bits | Field |
|---|---|
| 0-11 | the border texels: bit `3c + b` is set where channel `c` (0 red, 1 green, 2 blue, 3 alpha) reads 1 rather than 0 outside the level for border color `b` (`MTLSamplerBorderColor`'s raw value) |
| 12 | the format has no alpha, so an out-of-bounds `read` reads alpha 1 |
| 13-15 | 0 |
| 16-31 | the `MTLPixelFormat` raw value |

The border texels are taken after the swizzle. A color format's channels are its own, a missing
red, green or blue reads 0 and a missing alpha reads 1; `A8Unorm`'s alpha is the color's red; a
format held rotated (`ARGB8Unorm`) reads red from the color's green, green from its blue, blue from
its alpha and alpha from its red; a depth or stencil format reads as red, green and blue, alpha 1.
Every pixel format's raw value is above 0, so a texture's word is never 0. The placeholders
(`Invalid`, `Unspecialized`) and the formats the translator holds no texture of or only copies have
no word.

## The sampler word

`metal2vulkan::metal::sampler::SamplerWord::of(descriptor)`. Each field holds its
`MTLSamplerDescriptor` value's raw Metal value, so the default descriptor's word is 0:

| Bits | Field |
|---|---|
| 0 | `normalizedCoordinates` is false |
| 1 | `minFilter` is `Linear` |
| 2 | `magFilter` is `Linear` |
| 3-4 | `mipFilter` |
| 5-7, 8-10, 11-13 | `sAddressMode`, `tAddressMode`, `rAddressMode` |
| 14-15 | `borderColor` |
| 16-18 | `compareFunction` |
| 19-20 | `reductionMode` |
| 21 | `maxAnisotropy` is above 1 |
| 22-25 | the level a nearest mip filter reads at `lodMinClamp` |
| 26-29 | 15 less the level a nearest mip filter reads at `lodMaxClamp` |
| 30 | `minFilter` or `magFilter` is `Cubic` |
| 31 | 0 |

The level a nearest mip filter reads at a clamp `c`: take `c` below 0 as 0, and below 64 truncate
it onto sixty-fourths of a level; if the result is 15 or more the level is 15; otherwise, with `L`
its whole levels and `q` its fraction, the level is `L + 1` where `q` is above one half, else `L`.
The LOD clamps themselves, `lodAverage` and the anisotropy's degree are not in the word: the
Vulkan sampler applies them.

## Test vectors

[`crates/metal2vulkan/tests/resource_words.tsv`](../crates/metal2vulkan/tests/resource_words.tsv)
holds one descriptor per line with its word, in tab-separated raw Metal values; its header names
the columns. A texture line is a pixel format and its four swizzle sources; a sampler line is every
`MTLSamplerDescriptor` field above, with `lodMinClamp` and `lodMaxClamp` as the float's IEEE-754
bits. Words are hexadecimal.

The translator's tests (`tests/resource_words.rs`) recompute every line's word and hold the file
equal to what the word functions generate, so a changed word changes the file. A guest's own
computation of the words checks itself against the same file.
