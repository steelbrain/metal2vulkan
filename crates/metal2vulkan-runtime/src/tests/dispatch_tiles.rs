//! Kernels that do not observe their threadgroup partition, dispatched by
//! `dispatchThreadgroups:threadsPerThreadgroup:` at threadgroup sizes whose grids the runtime's own
//! workgroups do not divide: under a device cfg, on every device of the cfg's kind. Such a kernel
//! runs in workgroups of the runtime's choosing (`encoder::choose`): an image-only one
//! (`tiles_copy`, [`Form::Square`]) in square workgroups over a grid one deep and its tile over a
//! deeper one, any other (`tiles_mark`, [`Form::Tiled`]) in the tiles its region carries. In
//! Metal every thread of the grid, the threadgroups times their size, runs once at its own
//! position: the copies leave each texel inside the grid copied and every other one as the host
//! wrote it, and the marks hold each element's position and the grid's size, with nothing past
//! the grid written. Each test states the form its kernel took, so the path it covers is the one
//! that ran. `llvm-as` assembles the kernels; without it the tests are skipped, visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::pipeline::Form;
    use crate::tests::command::on_device::{each_device, finished};
    use crate::{
        Buffer, ComputeCommandEncoder, ComputePipelineDescriptor, ComputePipelineState, Device,
        DispatchType, FunctionDescriptor, HostBytes, IndexClass, Library, MemoryClass, PixelFormat,
        Region, StageSource, Texture, TextureDescriptor, TextureType, TextureUsage,
    };
    use metal2vulkan_devtools::{fixtures_dir, llvm_assemble, AssembleError};

    /// The 2D textures' size.
    const SIDES: [u32; 3] = [70, 45, 1];
    /// The 3D textures' size.
    const DEEP: [u32; 3] = [13, 11, 7];
    /// The 2D dispatches, each `(threadgroups, threads per threadgroup)`: grids inside the
    /// texture, past it on both axes, past it on one, and the texture's own size, none a multiple
    /// of 16 on both axes.
    const FLAT: [([u32; 3], [u32; 3]); 6] = [
        ([5, 3, 1], [7, 9, 1]),
        ([11, 6, 1], [7, 9, 1]),
        ([3, 7, 1], [32, 1, 1]),
        ([9, 5, 1], [8, 8, 1]),
        ([1, 45, 1], [70, 1, 1]),
        ([70, 45, 1], [1, 1, 1]),
    ];
    /// The 3D dispatches, as [`FLAT`]'s.
    const DEEPER: [([u32; 3], [u32; 3]); 4] = [
        ([3, 2, 2], [5, 6, 3]),
        ([2, 2, 7], [7, 6, 1]),
        ([1, 1, 7], [13, 11, 1]),
        ([4, 4, 2], [4, 4, 4]),
    ];

    fn library() -> Option<Library> {
        let path = fixtures_dir().join("examples/dispatch_tiles.ll");
        match llvm_assemble(&path) {
            Ok(bitcode) => Some(Library::from_air(&bitcode).expect("the library")),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: dispatch tiles: {why}");
                None
            }
            Err(error) => panic!("{}: {error:?}", path.display()),
        }
    }

    fn pipeline(name: &str, device: &Device, library: &Library, f: &str) -> ComputePipelineState {
        let source = StageSource::Function(FunctionDescriptor::new(
            library.function(f).expect("the function"),
        ));
        (device.new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(source),
            crate::tests::no_constexpr_samplers,
        ))
        .unwrap_or_else(|error| panic!("{name}: {f}: {error}"))
    }

    /// The form `state` runs a direct dispatch in.
    fn form(state: &ComputePipelineState) -> &'static str {
        match state.inner.compiled.form {
            Form::Square => "square",
            Form::Tiled => "tiled",
            Form::Sized { .. } => "sized",
        }
    }

    /// One dispatch of `state` with `bind` binding its resources, in one command buffer.
    fn dispatch(
        name: &str,
        device: &Device,
        state: &ComputePipelineState,
        (groups, size): ([u32; 3], [u32; 3]),
        bind: impl FnOnce(&mut ComputeCommandEncoder),
    ) {
        let queue = device.new_command_queue(1).expect("a queue");
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut encoder =
            (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
        encoder
            .set_compute_pipeline_state(state)
            .expect("the state");
        bind(&mut encoder);
        (encoder.dispatch_threadgroups(groups, size))
            .unwrap_or_else(|error| panic!("{name}: {groups:?} of {size:?}: {error}"));
        encoder.end_encoding();
        command_buffer.commit().expect("a commit");
        finished(name, &command_buffer);
        assert_eq!(command_buffer.error(), None, "{name}");
    }

    /// A Shared RGBA8Unorm texture of `sides` at `index` (3D where it is more than one deep),
    /// holding `texel(x, y, z)`.
    fn texture(
        device: &Device,
        sides: [u32; 3],
        index: u32,
        usage: TextureUsage,
        texel: impl Fn(u32, u32, u32) -> [u8; 4],
    ) -> Texture {
        let kind = match sides[2] {
            1 => TextureType::Type2D,
            _ => TextureType::Type3D,
        };
        let mut d = TextureDescriptor::new(kind, PixelFormat::RGBA8Unorm, sides[0], sides[1]);
        d.depth = sides[2];
        d.usage = usage;
        let texture =
            (device.new_texture(&d, Some(index))).unwrap_or_else(|error| panic!("{d:?}: {error}"));
        let bytes: Vec<u8> = (0..sides[2])
            .flat_map(|z| (0..sides[1]).flat_map(move |y| (0..sides[0]).map(move |x| (x, y, z))))
            .flat_map(|(x, y, z)| texel(x, y, z))
            .collect();
        let row = u64::from(sides[0]) * 4;
        let whole = Region {
            origin: [0; 3],
            size: sides,
        };
        (texture.replace_region(
            whole,
            0,
            0,
            HostBytes::from(&bytes[..]),
            row,
            row * u64::from(sides[1]),
        ))
        .unwrap_or_else(|error| panic!("{d:?}: {error}"));
        texture
    }

    /// The bytes of `texture` of `sides`, read back.
    fn bytes_of(texture: &Texture, sides: [u32; 3]) -> Vec<u8> {
        let row = u64::from(sides[0]) * 4;
        let image = row * u64::from(sides[1]);
        let mut bytes = vec![0; (image * u64::from(sides[2])) as usize];
        let whole = Region {
            origin: [0; 3],
            size: sides,
        };
        (texture.get_bytes(&mut bytes, row, image, whole, 0, 0)).expect("the texels");
        bytes
    }

    /// The grid a dispatch of `groups` threadgroups of `size` launches.
    fn grid((groups, size): ([u32; 3], [u32; 3])) -> [u32; 3] {
        std::array::from_fn(|axis| groups[axis] * size[axis])
    }

    /// Copies a texture of `sides` with `copy` at each of `dispatches`, the source at texture
    /// index `first` and the destination at the next: each texel inside the grid takes the
    /// source's, each other keeps the destination's, which the host writes again before each.
    fn copies(
        name: &str,
        device: &Device,
        copy: &ComputePipelineState,
        (sides, first): ([u32; 3], u32),
        dispatches: &[([u32; 3], [u32; 3])],
    ) {
        let source_texel = |x: u32, y: u32, z: u32| {
            [
                (x * 3 + 1) as u8,
                (y * 5 + 2) as u8,
                (z * 29 + x + y) as u8,
                200,
            ]
        };
        let kept = [0xA5, 0x5A, 0x33, 0x77];
        let source = texture(
            device,
            sides,
            first,
            TextureUsage::SHADER_READ,
            source_texel,
        );
        let destination = texture(
            device,
            sides,
            first + 1,
            TextureUsage::SHADER_WRITE,
            |_, _, _| kept,
        );
        let row = u64::from(sides[0]) * 4;
        let all_kept: Vec<u8> = (0..sides[0] * sides[1] * sides[2])
            .flat_map(|_| kept)
            .collect();
        let whole = Region {
            origin: [0; 3],
            size: sides,
        };
        for &at in dispatches {
            (destination.replace_region(
                whole,
                0,
                0,
                HostBytes::from(&all_kept[..]),
                row,
                row * u64::from(sides[1]),
            ))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            dispatch(name, device, copy, at, |e| {
                e.set_texture(Some(&source), 0).expect("the source");
                e.set_texture(Some(&destination), 1)
                    .expect("the destination");
            });
            let grid = grid(at);
            let now = bytes_of(&destination, sides);
            let mut wrong = 0;
            let mut first_wrong = None;
            for z in 0..sides[2] {
                for y in 0..sides[1] {
                    for x in 0..sides[0] {
                        let inside = x < grid[0] && y < grid[1] && z < grid[2];
                        let want = match inside {
                            true => source_texel(x, y, z),
                            false => kept,
                        };
                        let i = (((z * sides[1] + y) * sides[0] + x) * 4) as usize;
                        if now[i..i + 4] != want {
                            wrong += 1;
                            first_wrong.get_or_insert((
                                [x, y, z],
                                [now[i], now[i + 1], now[i + 2], now[i + 3]],
                                want,
                            ));
                        }
                    }
                }
            }
            if let Some((at_texel, got, want)) = first_wrong {
                panic!(
                    "{name}: {} copy of {sides:?}, {:?} threadgroups of {:?}: {wrong} texels \
                     wrong, the first {at_texel:?}: {got:?}, not {want:?}",
                    form(copy),
                    at.0,
                    at.1
                );
            }
        }
        drop((source, destination));
    }

    /// Marks a grid with `mark` at each of `dispatches`: each element holds its thread's
    /// position and the grid's size, and the 16 elements past the grid keep the host's bytes.
    fn marks(
        name: &str,
        device: &Device,
        mark: &ComputePipelineState,
        dispatches: &[([u32; 3], [u32; 3])],
        deep: bool,
    ) {
        for &at in dispatches {
            let grid = grid(at);
            let elements = u64::from(grid[0] * grid[1] * grid[2]);
            let len = (elements + 16) * 16;
            let block = (device.take_block(MemoryClass::Shared, len.next_multiple_of(65_536)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let marks: Buffer = (device.new_buffer(block.id, 0, len, None))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let contents = marks.contents().expect("Shared");
            // SAFETY: the buffer holds `len` bytes, and no command reads them meanwhile.
            unsafe { contents.as_ptr().write_bytes(0xEE, len as usize) };
            dispatch(name, device, mark, at, |e| {
                e.set_buffer(Some(&marks), 0, 0).expect("the marks");
            });
            // SAFETY: the dispatch that wrote the buffer completed, and its writes reach the host.
            let words = unsafe {
                std::slice::from_raw_parts(contents.as_ptr().cast::<u32>(), (len / 4) as usize)
            };
            let mut wrong = 0;
            let mut first_wrong = None;
            for z in 0..grid[2] {
                for y in 0..grid[1] {
                    for x in 0..grid[0] {
                        let want = match deep {
                            false => [x + 1, y + 1, grid[0], grid[1]],
                            true => [
                                x + 1,
                                y + 1,
                                z + 1,
                                (grid[2] << 16) | (grid[1] << 8) | grid[0],
                            ],
                        };
                        let i = (((z * grid[1] + y) * grid[0] + x) * 4) as usize;
                        if words[i..i + 4] != want {
                            wrong += 1;
                            first_wrong.get_or_insert((
                                [x, y, z],
                                [words[i], words[i + 1], words[i + 2], words[i + 3]],
                                want,
                            ));
                        }
                    }
                }
            }
            let past = (elements * 4) as usize;
            let written_past = words[past..].iter().filter(|&&w| w != 0xEEEE_EEEE).count();
            if first_wrong.is_some() || written_past > 0 {
                panic!(
                    "{name}: {} marks, {:?} threadgroups of {:?}: {wrong} elements wrong, the \
                     first {first_wrong:?}; {written_past} words past the grid written",
                    form(mark),
                    at.0,
                    at.1
                );
            }
            drop(marks);
            (device.release_block(block)).unwrap_or_else(|error| panic!("{name}: {error}"));
        }
    }

    #[test]
    fn partition_oblivious_kernels_run_every_thread_of_a_grid_their_workgroups_do_not_divide() {
        let Some(library) = library() else {
            return;
        };
        each_device("dispatch tiles", |name, _, device| {
            let first = IndexClass::Texture.range().start;
            (device.grant_indices(IndexClass::Texture, first, 4)).expect("a grant");
            let copy = pipeline(name, &device, &library, "tiles_copy");
            let copy_deep = pipeline(name, &device, &library, "tiles_copy_deep");
            let mark = pipeline(name, &device, &library, "tiles_mark");
            let mark_deep = pipeline(name, &device, &library, "tiles_mark_deep");
            assert_eq!(
                [form(&copy), form(&copy_deep), form(&mark), form(&mark_deep)],
                ["square", "square", "tiled", "tiled"],
                "{name}: the forms"
            );
            copies(name, &device, &copy, (SIDES, first), &FLAT);
            copies(name, &device, &copy_deep, (DEEP, first + 2), &DEEPER);
            marks(name, &device, &mark, &FLAT, false);
            marks(name, &device, &mark_deep, &DEEPER, true);
            println!(
                "dispatch tiles: {name}: square copies and tiled marks exact at {} grids one \
                 deep and {} deeper",
                FLAT.len(),
                DEEPER.len()
            );
        });
    }

    /// `ComputePipelineState::dispatch_variant` keys the dispatches of `tiles_mark`, a tiling
    /// kernel, by the `VkPipeline` each runs: every direct dispatch, whatever its grid, its call
    /// and its threadgroup size, runs the kernel's tiles, the pipeline made with the state, so
    /// they share one key and prepared they make nothing; an indirect dispatch runs the whole
    /// module at its own size, without tiles, one key and one pipeline for each size. The state's
    /// creation and the preparations make one pipeline for each key.
    #[test]
    fn every_device_keys_a_tiling_kernel_s_dispatches_by_the_pipeline_they_run() {
        use crate::Dispatch;
        use std::collections::HashSet;

        let Some(library) = library() else {
            return;
        };
        each_device("dispatch tiles variant", |name, _, device| {
            let counters = device.counters();
            let before = counters.metrics();
            let state = pipeline(name, &device, &library, "tiles_mark");
            assert_eq!(form(&state), "tiled", "{name}: the form");
            let made = counters.metrics();
            // Each entry: the dispatch, its size, and the pipeline it runs.
            let table = [
                (Dispatch::Threadgroups([5, 3, 1]), [7, 9, 1], "tiles"),
                (Dispatch::Threadgroups([11, 6, 1]), [8, 8, 1], "tiles"),
                (Dispatch::Threads([70, 45, 1]), [32, 1, 1], "tiles"),
                (Dispatch::Indirect, [7, 9, 1], "whole, 7 by 9"),
                (Dispatch::Indirect, [8, 8, 1], "whole, 8 by 8"),
            ];
            let keys: Vec<_> = (table.iter())
                .map(|&(dispatch, size, pipeline)| {
                    (state.dispatch_variant(dispatch, size, &[]))
                        .map(|(variant, _)| variant)
                        .unwrap_or_else(|| panic!("{name}: {pipeline}: {dispatch:?} by {size:?}"))
                })
                .collect();
            for (a, key_a) in table.iter().zip(&keys) {
                for (b, key_b) in table.iter().zip(&keys) {
                    assert_eq!(key_a == key_b, a.2 == b.2, "{name}: {a:?} and {b:?}");
                }
            }
            for &(dispatch, size, _) in &table {
                state
                    .prepare_dispatch(dispatch, size, &[])
                    .expect("made ahead");
            }
            let after = counters.metrics();
            let pipelines: HashSet<&str> = table.iter().map(|entry| entry.2).collect();
            assert_eq!(
                (
                    after.pipelines_created - before.pipelines_created,
                    after.made_ahead_variants - made.made_ahead_variants,
                ),
                (pipelines.len() as u64, pipelines.len() as u64 - 1),
                "{name}: one pipeline for each key, the tiles' made with the state"
            );
            println!(
                "dispatch tiles variant: {name}: {} dispatches, {} keys",
                table.len(),
                pipelines.len()
            );
        });
    }
}
