//! A null descriptor (`VK_EXT_robustness2`'s `nullDescriptor`, enabled alone) at element 0 of
//! the texture arrays a nil `gpuResourceID` names: under a device cfg, on every device of the cfg's
//! kind, a compute shader reaches element 0 of the sampled-image array through two variables of
//! different classes (a `float` 2D and a `uint` 3D image) and of the storage-image array through
//! two more (`float` and `uint` 2D), as the translator's identity domains alias them. Each size
//! and level query answers 0, each load and each sample (through a real sampler) answers the zero
//! texel, as Metal answers each for a nil texture, and each store is dropped: a load after it
//! still answers zero, and the texture at element 1 keeps its texel. Vulkan has no null sampler
//! (`VkWriteDescriptorSet` lists buffers, image views and buffer views), so the sampler is real.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::devices_of_the_cfgs_kind;
    use crate::Instance;
    use ash::vk;
    use std::ffi::c_char;

    /// The sampled-image array: one element per texture index (`docs/RESOURCE-ID.md`).
    const SAMPLED: u32 = 65_536;
    /// The storage-image array: one element per texture index, as the sampled-image array.
    const STORAGE: u32 = 65_536;
    /// The sampler array.
    const SAMPLERS: u32 = 4_000;
    /// The words the shader stores, each 0 where the device answers as Metal does.
    const ANSWERS: usize = 30;
    /// Words of the output buffer: the answers, two words the shader never writes, and the texel
    /// of element 1 of the storage array, copied out after the dispatch.
    const WORDS: usize = 33;
    /// The texel of element 1 of the storage array, before the dispatch and after it.
    const NEIGHBOUR: u32 = 0x5a5a_5a5a;
    /// What the output buffer holds before the dispatch.
    const UNWRITTEN: u32 = 0xffff_ffff;

    /// A SPIR-V module, built instruction by instruction: its preamble (capabilities, memory
    /// model, entry point and execution mode), its annotations and the rest, each in its section.
    struct Spirv {
        head: Vec<u32>,
        annotations: Vec<u32>,
        body: Vec<u32>,
        bound: u32,
    }

    /// Instruction `opcode` with `operands`, appended to `into`.
    fn encode(into: &mut Vec<u32>, opcode: u32, operands: &[u32]) {
        let count = u32::try_from(operands.len() + 1).expect("an instruction's length");
        into.push(count << 16 | opcode);
        into.extend_from_slice(operands);
    }

    impl Spirv {
        fn new() -> Spirv {
            Spirv {
                head: Vec::new(),
                annotations: Vec::new(),
                body: Vec::new(),
                bound: 1,
            }
        }

        fn id(&mut self) -> u32 {
            self.bound += 1;
            self.bound - 1
        }

        fn head(&mut self, opcode: u32, operands: &[u32]) {
            encode(&mut self.head, opcode, operands);
        }

        fn decorate(&mut self, opcode: u32, operands: &[u32]) {
            encode(&mut self.annotations, opcode, operands);
        }

        fn op(&mut self, opcode: u32, operands: &[u32]) {
            encode(&mut self.body, opcode, operands);
        }

        /// An instruction that defines a result of type `ty`.
        fn value(&mut self, opcode: u32, ty: u32, operands: &[u32]) -> u32 {
            let id = self.id();
            let mut all = vec![ty, id];
            all.extend_from_slice(operands);
            self.op(opcode, &all);
            id
        }

        /// An instruction that defines a type or a typeless result.
        fn ty(&mut self, opcode: u32, operands: &[u32]) -> u32 {
            let id = self.id();
            let mut all = vec![id];
            all.extend_from_slice(operands);
            self.op(opcode, &all);
            id
        }

        /// The module's words: the header (magic, SPIR-V 1.5, generator 0, the bound, schema 0)
        /// and the sections in order.
        fn finish(self) -> Vec<u32> {
            let mut words = vec![0x0723_0203, 0x0001_0500, 0, self.bound, 0];
            words.extend(self.head);
            words.extend(self.annotations);
            words.extend(self.body);
            words
        }
    }

    /// `s` as a SPIR-V literal string: nul-terminated, in little-endian words.
    fn string(s: &str) -> Vec<u32> {
        let mut bytes = s.as_bytes().to_vec();
        bytes.push(0);
        bytes.resize(bytes.len().div_ceil(4) * 4, 0);
        bytes
            .chunks(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    // The opcodes and enumerants the probe names (the SPIR-V 1.5 specification).
    const OP_MEMORY_MODEL: u32 = 14;
    const OP_ENTRY_POINT: u32 = 15;
    const OP_EXECUTION_MODE: u32 = 16;
    const OP_CAPABILITY: u32 = 17;
    const OP_TYPE_VOID: u32 = 19;
    const OP_TYPE_INT: u32 = 21;
    const OP_TYPE_FLOAT: u32 = 22;
    const OP_TYPE_VECTOR: u32 = 23;
    const OP_TYPE_IMAGE: u32 = 25;
    const OP_TYPE_SAMPLER: u32 = 26;
    const OP_TYPE_SAMPLED_IMAGE: u32 = 27;
    const OP_TYPE_RUNTIME_ARRAY: u32 = 29;
    const OP_TYPE_STRUCT: u32 = 30;
    const OP_TYPE_POINTER: u32 = 32;
    const OP_TYPE_FUNCTION: u32 = 33;
    const OP_CONSTANT: u32 = 43;
    const OP_CONSTANT_COMPOSITE: u32 = 44;
    const OP_FUNCTION: u32 = 54;
    const OP_FUNCTION_END: u32 = 56;
    const OP_VARIABLE: u32 = 59;
    const OP_LOAD: u32 = 61;
    const OP_STORE: u32 = 62;
    const OP_ACCESS_CHAIN: u32 = 65;
    const OP_DECORATE: u32 = 71;
    const OP_MEMBER_DECORATE: u32 = 72;
    const OP_COMPOSITE_EXTRACT: u32 = 81;
    const OP_SAMPLED_IMAGE: u32 = 86;
    const OP_IMAGE_SAMPLE_EXPLICIT_LOD: u32 = 88;
    const OP_IMAGE_FETCH: u32 = 95;
    const OP_IMAGE_WRITE: u32 = 99;
    const OP_IMAGE_QUERY_SIZE_LOD: u32 = 103;
    const OP_IMAGE_QUERY_SIZE: u32 = 104;
    const OP_IMAGE_QUERY_LEVELS: u32 = 106;
    const OP_BITCAST: u32 = 124;
    const OP_LABEL: u32 = 248;
    const OP_RETURN: u32 = 253;
    const CAPABILITY_SHADER: u32 = 1;
    const CAPABILITY_IMAGE_QUERY: u32 = 50;
    const CAPABILITY_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT: u32 = 56;
    const CAPABILITY_RUNTIME_DESCRIPTOR_ARRAY: u32 = 5302;
    const DECORATION_BLOCK: u32 = 2;
    const DECORATION_ARRAY_STRIDE: u32 = 6;
    const DECORATION_BINDING: u32 = 33;
    const DECORATION_DESCRIPTOR_SET: u32 = 34;
    const DECORATION_OFFSET: u32 = 35;
    const STORAGE_UNIFORM_CONSTANT: u32 = 0;
    const STORAGE_BUFFER: u32 = 12;
    const DIM_2D: u32 = 1;
    const DIM_3D: u32 = 2;
    const IMAGE_OPERANDS_LOD: u32 = 2;

    /// The probe: it stores, in order, word by word to set 0's buffer, of element 0 of each array:
    ///
    /// - the `float` 2D sampled image's size at level 0 (2 words), its levels, a fetch of texel
    ///   (0, 0) (4) and a sample at (0.5, 0.5) at level 0 through sampler 1 (4);
    /// - the `uint` 3D sampled image's size at level 0 (3) and a fetch of texel (0, 0, 0) (4);
    /// - the `float` and the `uint` 2D storage images' sizes (2 each), then a store of ones and of
    ///   sevens to their texel (0, 0);
    /// - the two sampled images' fetches again (4 each).
    fn probe() -> Vec<u32> {
        let mut s = Spirv::new();
        for c in [
            CAPABILITY_SHADER,
            CAPABILITY_RUNTIME_DESCRIPTOR_ARRAY,
            CAPABILITY_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT,
            CAPABILITY_IMAGE_QUERY,
        ] {
            s.head(OP_CAPABILITY, &[c]);
        }
        // Logical addressing, the GLSL450 memory model.
        s.head(OP_MEMORY_MODEL, &[0, 1]);
        // The ids the entry point names are taken ahead of their definitions.
        let main = s.id();
        let out = s.id();
        let (tf, tu, sf, su, samplers) = (s.id(), s.id(), s.id(), s.id(), s.id());
        let mut entry = vec![5, main];
        entry.extend(string("main"));
        entry.extend([out, tf, tu, sf, su, samplers]);
        s.head(OP_ENTRY_POINT, &entry);
        // LocalSize 1 1 1.
        s.head(OP_EXECUTION_MODE, &[main, 17, 1, 1, 1]);

        let void = s.ty(OP_TYPE_VOID, &[]);
        let func = s.ty(OP_TYPE_FUNCTION, &[void]);
        let f32_ = s.ty(OP_TYPE_FLOAT, &[32]);
        let u32_ = s.ty(OP_TYPE_INT, &[32, 0]);
        let i32_ = s.ty(OP_TYPE_INT, &[32, 1]);
        let v2i = s.ty(OP_TYPE_VECTOR, &[i32_, 2]);
        let v3i = s.ty(OP_TYPE_VECTOR, &[i32_, 3]);
        let v2f = s.ty(OP_TYPE_VECTOR, &[f32_, 2]);
        let v4f = s.ty(OP_TYPE_VECTOR, &[f32_, 4]);
        let v4u = s.ty(OP_TYPE_VECTOR, &[u32_, 4]);
        // Sampled type, dim, depth 0, arrayed 0, multisampled 0, sampled (1) or storage (2),
        // format Unknown.
        let itf = s.ty(OP_TYPE_IMAGE, &[f32_, DIM_2D, 0, 0, 0, 1, 0]);
        let itu = s.ty(OP_TYPE_IMAGE, &[u32_, DIM_3D, 0, 0, 0, 1, 0]);
        let isf = s.ty(OP_TYPE_IMAGE, &[f32_, DIM_2D, 0, 0, 0, 2, 0]);
        let isu = s.ty(OP_TYPE_IMAGE, &[u32_, DIM_2D, 0, 0, 0, 2, 0]);
        let sampler = s.ty(OP_TYPE_SAMPLER, &[]);
        let sampled_f = s.ty(OP_TYPE_SAMPLED_IMAGE, &[itf]);
        let words = s.ty(OP_TYPE_RUNTIME_ARRAY, &[u32_]);
        let block = s.ty(OP_TYPE_STRUCT, &[words]);
        let block_ptr = s.ty(OP_TYPE_POINTER, &[STORAGE_BUFFER, block]);
        let word_ptr = s.ty(OP_TYPE_POINTER, &[STORAGE_BUFFER, u32_]);
        s.decorate(OP_DECORATE, &[words, DECORATION_ARRAY_STRIDE, 4]);
        s.decorate(OP_MEMBER_DECORATE, &[block, 0, DECORATION_OFFSET, 0]);
        s.decorate(OP_DECORATE, &[block, DECORATION_BLOCK]);
        // Each array variable, its element pointer type, and its set and binding.
        let mut arrays = Vec::new();
        for (var, elem, binding) in [
            (tf, itf, 0),
            (tu, itu, 0),
            (sf, isf, 1),
            (su, isu, 1),
            (samplers, sampler, 2),
        ] {
            let array = s.ty(OP_TYPE_RUNTIME_ARRAY, &[elem]);
            let ptr = s.ty(OP_TYPE_POINTER, &[STORAGE_UNIFORM_CONSTANT, array]);
            let elem_ptr = s.ty(OP_TYPE_POINTER, &[STORAGE_UNIFORM_CONSTANT, elem]);
            s.decorate(OP_DECORATE, &[var, DECORATION_DESCRIPTOR_SET, 1]);
            s.decorate(OP_DECORATE, &[var, DECORATION_BINDING, binding]);
            arrays.push((var, ptr, elem, elem_ptr));
        }
        s.decorate(OP_DECORATE, &[out, DECORATION_DESCRIPTOR_SET, 0]);
        s.decorate(OP_DECORATE, &[out, DECORATION_BINDING, 0]);

        let c0 = s.value(OP_CONSTANT, i32_, &[0]);
        let c1 = s.value(OP_CONSTANT, i32_, &[1]);
        let f0 = s.value(OP_CONSTANT, f32_, &[0.0f32.to_bits()]);
        let half = s.value(OP_CONSTANT, f32_, &[0.5f32.to_bits()]);
        let one = s.value(OP_CONSTANT, f32_, &[1.0f32.to_bits()]);
        let seven = s.value(OP_CONSTANT, u32_, &[7]);
        let z2 = s.value(OP_CONSTANT_COMPOSITE, v2i, &[c0, c0]);
        let z3 = s.value(OP_CONSTANT_COMPOSITE, v3i, &[c0, c0, c0]);
        let h2 = s.value(OP_CONSTANT_COMPOSITE, v2f, &[half, half]);
        let ones = s.value(OP_CONSTANT_COMPOSITE, v4f, &[one, one, one, one]);
        let sevens = s.value(OP_CONSTANT_COMPOSITE, v4u, &[seven, seven, seven, seven]);
        let indices: Vec<u32> = (0..ANSWERS as u32)
            .map(|k| s.value(OP_CONSTANT, u32_, &[k]))
            .collect();
        s.op(OP_VARIABLE, &[block_ptr, out, STORAGE_BUFFER]);
        for &(var, ptr, _, _) in &arrays {
            s.op(OP_VARIABLE, &[ptr, var, STORAGE_UNIFORM_CONSTANT]);
        }

        s.op(OP_FUNCTION, &[void, main, 0, func]);
        s.ty(OP_LABEL, &[]);
        // Element `index` of array `a`, loaded.
        let element = |s: &mut Spirv, a: usize, index: u32| {
            let (var, _, elem, elem_ptr) = arrays[a];
            let p = s.value(OP_ACCESS_CHAIN, elem_ptr, &[var, index]);
            s.value(OP_LOAD, elem, &[p])
        };
        // Stores lanes `0 .. n` of `v` (of lane type `lane`) at the next words.
        let mut next = 0usize;
        let mut store = |s: &mut Spirv, v: u32, lane: u32, n: u32| {
            for i in 0..n {
                let x = match n {
                    1 => v,
                    _ => s.value(OP_COMPOSITE_EXTRACT, lane, &[v, i]),
                };
                let w = match lane == u32_ {
                    true => x,
                    false => s.value(OP_BITCAST, u32_, &[x]),
                };
                let p = s.value(OP_ACCESS_CHAIN, word_ptr, &[out, c0, indices[next]]);
                s.op(OP_STORE, &[p, w]);
                next += 1;
            }
        };
        let t = element(&mut s, 0, c0);
        let size = s.value(OP_IMAGE_QUERY_SIZE_LOD, v2i, &[t, c0]);
        store(&mut s, size, i32_, 2);
        let levels = s.value(OP_IMAGE_QUERY_LEVELS, i32_, &[t]);
        store(&mut s, levels, i32_, 1);
        let texel = s.value(OP_IMAGE_FETCH, v4f, &[t, z2, IMAGE_OPERANDS_LOD, c0]);
        store(&mut s, texel, f32_, 4);
        let smp = element(&mut s, 4, c1);
        let si = s.value(OP_SAMPLED_IMAGE, sampled_f, &[t, smp]);
        let sample = s.value(
            OP_IMAGE_SAMPLE_EXPLICIT_LOD,
            v4f,
            &[si, h2, IMAGE_OPERANDS_LOD, f0],
        );
        store(&mut s, sample, f32_, 4);
        let u = element(&mut s, 1, c0);
        let size = s.value(OP_IMAGE_QUERY_SIZE_LOD, v3i, &[u, c0]);
        store(&mut s, size, i32_, 3);
        let texel = s.value(OP_IMAGE_FETCH, v4u, &[u, z3, IMAGE_OPERANDS_LOD, c0]);
        store(&mut s, texel, u32_, 4);
        let wf = element(&mut s, 2, c0);
        let size = s.value(OP_IMAGE_QUERY_SIZE, v2i, &[wf]);
        store(&mut s, size, i32_, 2);
        let wu = element(&mut s, 3, c0);
        let size = s.value(OP_IMAGE_QUERY_SIZE, v2i, &[wu]);
        store(&mut s, size, i32_, 2);
        s.op(OP_IMAGE_WRITE, &[wf, z2, ones]);
        s.op(OP_IMAGE_WRITE, &[wu, z2, sevens]);
        let t = element(&mut s, 0, c0);
        let texel = s.value(OP_IMAGE_FETCH, v4f, &[t, z2, IMAGE_OPERANDS_LOD, c0]);
        store(&mut s, texel, f32_, 4);
        let u = element(&mut s, 1, c0);
        let texel = s.value(OP_IMAGE_FETCH, v4u, &[u, z3, IMAGE_OPERANDS_LOD, c0]);
        store(&mut s, texel, u32_, 4);
        assert_eq!(next, ANSWERS);
        s.op(OP_RETURN, &[]);
        s.op(OP_FUNCTION_END, &[]);
        s.finish()
    }

    /// The first memory type of `bits` with `flags`.
    fn memory_type(
        memory: &vk::PhysicalDeviceMemoryProperties,
        bits: u32,
        flags: vk::MemoryPropertyFlags,
    ) -> u32 {
        (0..memory.memory_type_count)
            .find(|&i| {
                bits & 1 << i != 0
                    && memory.memory_types[i as usize]
                        .property_flags
                        .contains(flags)
            })
            .expect("a memory type")
    }

    /// What the probe reads and stores through the null descriptors, and the texel of element 1
    /// of the storage array after it.
    ///
    /// # Safety
    ///
    /// `physical` is one of `instance`'s, and offers `VK_EXT_robustness2` with `nullDescriptor`.
    unsafe fn run(
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
    ) -> Result<Vec<u32>, vk::Result> {
        // SAFETY: the caller's contract.
        let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let family = families
            .iter()
            .position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .expect("a compute queue family") as u32;
        let priorities = [1.0];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let core =
            vk::PhysicalDeviceFeatures::default().shader_storage_image_write_without_format(true);
        let mut vulkan12 = vk::PhysicalDeviceVulkan12Features::default()
            .runtime_descriptor_array(true)
            .descriptor_binding_partially_bound(true)
            .descriptor_binding_sampled_image_update_after_bind(true)
            .descriptor_binding_storage_image_update_after_bind(true)
            .descriptor_binding_update_unused_while_pending(true);
        // `nullDescriptor` alone: neither `robustBufferAccess2` nor `robustImageAccess2`.
        let mut robustness2 =
            vk::PhysicalDeviceRobustness2FeaturesEXT::default().null_descriptor(true);
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .features(core)
            .push_next(&mut vulkan12)
            .push_next(&mut robustness2);
        let extensions: [*const c_char; 1] = [ash::ext::robustness2::NAME.as_ptr()];
        let info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&extensions)
            .push_next(&mut features);
        // SAFETY: the caller's contract; `info` and what it points to live through the call.
        let device = unsafe { instance.create_device(physical, &info, None) }?;
        // SAFETY: `device` is live and every object below is made from it and destroyed before it.
        let answers = unsafe { dispatch(instance, physical, &device, family) };
        // SAFETY: `dispatch` destroyed everything it made and waited for its work.
        unsafe { device.destroy_device(None) };
        answers
    }

    /// The probe's dispatch on `device`, made from `physical` with [`run`]'s features and queue
    /// `family`.
    ///
    /// # Safety
    ///
    /// As [`run`]'s.
    unsafe fn dispatch(
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        device: &ash::Device,
        family: u32,
    ) -> Result<Vec<u32>, vk::Result> {
        // SAFETY (every call below): `device` is live, and each handle passed was made from it
        // above and is destroyed only at the end, after the fence wait.
        unsafe {
            let memory = instance.get_physical_device_memory_properties(physical);
            let bytes = (WORDS * 4) as u64;
            let buffer = device.create_buffer(
                &vk::BufferCreateInfo::default().size(bytes).usage(
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                ),
                None,
            )?;
            let needs = device.get_buffer_memory_requirements(buffer);
            let host =
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let buffer_memory = device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(needs.size)
                    .memory_type_index(memory_type(&memory, needs.memory_type_bits, host)),
                None,
            )?;
            device.bind_buffer_memory(buffer, buffer_memory, 0)?;
            let mapped = device
                .map_memory(
                    buffer_memory,
                    0,
                    vk::WHOLE_SIZE,
                    vk::MemoryMapFlags::empty(),
                )?
                .cast::<u32>();
            std::slice::from_raw_parts_mut(mapped, WORDS).fill(UNWRITTEN);

            let image = device.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(vk::Format::R32_UINT)
                    .extent(vk::Extent3D {
                        width: 1,
                        height: 1,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(
                        vk::ImageUsageFlags::STORAGE
                            | vk::ImageUsageFlags::TRANSFER_DST
                            | vk::ImageUsageFlags::TRANSFER_SRC,
                    )
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )?;
            let needs = device.get_image_memory_requirements(image);
            let image_memory = device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(needs.size)
                    .memory_type_index(memory_type(
                        &memory,
                        needs.memory_type_bits,
                        vk::MemoryPropertyFlags::empty(),
                    )),
                None,
            )?;
            device.bind_image_memory(image, image_memory, 0)?;
            let range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1);
            let view = device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R32_UINT)
                    .subresource_range(range),
                None,
            )?;
            let sampler = device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )?;

            let set0_bindings = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)];
            let set0 = device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&set0_bindings),
                None,
            )?;
            let array = |binding, ty, count| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(binding)
                    .descriptor_type(ty)
                    .descriptor_count(count)
                    .stage_flags(vk::ShaderStageFlags::ALL)
            };
            let set1_bindings = [
                array(0, vk::DescriptorType::SAMPLED_IMAGE, SAMPLED),
                array(1, vk::DescriptorType::STORAGE_IMAGE, STORAGE),
                array(2, vk::DescriptorType::SAMPLER, SAMPLERS),
            ];
            let flags = [vk::DescriptorBindingFlags::PARTIALLY_BOUND
                | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
                | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING;
                3];
            let mut binding_flags =
                vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&flags);
            let set1 = device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                    .bindings(&set1_bindings)
                    .push_next(&mut binding_flags),
                None,
            )?;
            let sizes = [
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 1,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: SAMPLED,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: STORAGE,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::SAMPLER,
                    descriptor_count: SAMPLERS,
                },
            ];
            let pool = device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
                    .max_sets(2)
                    .pool_sizes(&sizes),
                None,
            )?;
            let layouts = [set0, set1];
            let sets = device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts),
            )?;
            let out = [vk::DescriptorBufferInfo::default()
                .buffer(buffer)
                .range(vk::WHOLE_SIZE)];
            // Element 0 of each texture array is a null descriptor; element 1 of the storage
            // array is the R32Uint texture and of the sampler array the sampler.
            let null_sampled = [vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let null_storage =
                [vk::DescriptorImageInfo::default().image_layout(vk::ImageLayout::GENERAL)];
            let neighbour = [vk::DescriptorImageInfo::default()
                .image_view(view)
                .image_layout(vk::ImageLayout::GENERAL)];
            let samplers = [vk::DescriptorImageInfo::default().sampler(sampler)];
            let write = |set, binding, element, ty| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(binding)
                    .dst_array_element(element)
                    .descriptor_type(ty)
            };
            let writes = [
                write(sets[0], 0, 0, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&out),
                write(sets[1], 0, 0, vk::DescriptorType::SAMPLED_IMAGE).image_info(&null_sampled),
                write(sets[1], 1, 0, vk::DescriptorType::STORAGE_IMAGE).image_info(&null_storage),
                write(sets[1], 1, 1, vk::DescriptorType::STORAGE_IMAGE).image_info(&neighbour),
                write(sets[1], 2, 1, vk::DescriptorType::SAMPLER).image_info(&samplers),
            ];
            device.update_descriptor_sets(&writes, &[]);

            let words = probe();
            let module = device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)?;
            let pipeline_layout = device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts),
                None,
            )?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main");
            let pipeline = device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(stage)
                        .layout(pipeline_layout)],
                    None,
                )
                .map_err(|(_, error)| error)?[0];

            let command_pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(family),
                None,
            )?;
            let cb = device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];
            device.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            let to_general = vk::ImageMemoryBarrier::default()
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(range);
            device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_general],
            );
            device.cmd_clear_color_image(
                cb,
                image,
                vk::ImageLayout::GENERAL,
                &vk::ClearColorValue {
                    uint32: [NEIGHBOUR; 4],
                },
                &[range],
            );
            let barrier = |src, dst| {
                [vk::MemoryBarrier::default()
                    .src_access_mask(src)
                    .dst_access_mask(dst)]
            };
            device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &barrier(
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                ),
                &[],
                &[],
            );
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                &sets,
                &[],
            );
            device.cmd_dispatch(cb, 1, 1, 1);
            device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &barrier(
                    vk::AccessFlags::SHADER_WRITE,
                    vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE,
                ),
                &[],
                &[],
            );
            let copy = vk::BufferImageCopy::default()
                .buffer_offset((WORDS as u64 - 1) * 4)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1),
                )
                .image_extent(vk::Extent3D {
                    width: 1,
                    height: 1,
                    depth: 1,
                });
            device.cmd_copy_image_to_buffer(cb, image, vk::ImageLayout::GENERAL, buffer, &[copy]);
            device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &barrier(vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::HOST_READ),
                &[],
                &[],
            );
            device.end_command_buffer(cb)?;
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let queue = device.get_device_queue(family, 0);
            let cbs = [cb];
            device.queue_submit(
                queue,
                &[vk::SubmitInfo::default().command_buffers(&cbs)],
                fence,
            )?;
            let waited = device.wait_for_fences(&[fence], true, 60_000_000_000);
            let answers = std::slice::from_raw_parts(mapped, WORDS).to_vec();
            device.device_wait_idle()?;
            device.destroy_fence(fence, None);
            device.destroy_command_pool(command_pool, None);
            device.destroy_pipeline(pipeline, None);
            device.destroy_pipeline_layout(pipeline_layout, None);
            device.destroy_shader_module(module, None);
            device.destroy_descriptor_pool(pool, None);
            device.destroy_descriptor_set_layout(set1, None);
            device.destroy_descriptor_set_layout(set0, None);
            device.destroy_sampler(sampler, None);
            device.destroy_image_view(view, None);
            device.destroy_image(image, None);
            device.free_memory(image_memory, None);
            device.unmap_memory(buffer_memory);
            device.destroy_buffer(buffer, None);
            device.free_memory(buffer_memory, None);
            waited.map(|()| answers)
        }
    }

    /// Through null descriptors at element 0 of aliased texture arrays, each query, load and
    /// sample answers zero and each store is dropped, on every device of the cfg's kind.
    #[test]
    fn a_null_texture_descriptor_answers_zero_and_drops_stores() {
        let instance = Instance::new(&[]).expect("an instance");
        let (_, vk_instance) = instance.raw();
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(
            !devices.is_empty(),
            "the loader lists no device of this cfg's kind"
        );
        let mut want = vec![0; ANSWERS];
        want.extend([UNWRITTEN, UNWRITTEN, NEIGHBOUR]);
        for (name, physical) in devices {
            let mut driver = vk::PhysicalDeviceVulkan12Properties::default();
            let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut driver);
            let mut robustness2 = vk::PhysicalDeviceRobustness2FeaturesEXT::default();
            let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut robustness2);
            // SAFETY: `physical` is one of the instance's, a Vulkan 1.3 instance.
            let offered = unsafe {
                vk_instance.get_physical_device_properties2(physical.raw(), &mut properties);
                vk_instance.get_physical_device_features2(physical.raw(), &mut features);
                vk_instance.enumerate_device_extension_properties(physical.raw())
            }
            .expect("the device extensions")
            .iter()
            .any(|e| e.extension_name_as_c_str() == Ok(ash::ext::robustness2::NAME));
            let info = driver
                .driver_info_as_c_str()
                .map(|info| info.to_string_lossy().into_owned())
                .unwrap_or_default();
            assert!(
                offered && robustness2.null_descriptor == vk::TRUE,
                "{name} ({info}): no VK_EXT_robustness2 nullDescriptor"
            );
            // SAFETY: as above, and the device offers `nullDescriptor`.
            let got = unsafe { run(vk_instance, physical.raw()) }
                .unwrap_or_else(|error| panic!("{name} ({info}): {error}"));
            println!("null descriptor: {name} ({info}): {got:#x?}");
            assert_eq!(got, want, "{name} ({info})");
        }
    }
}
