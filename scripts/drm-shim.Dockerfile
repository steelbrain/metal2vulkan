# Pinned image for `scripts/linux-gate --recipe drm-shim`: L4 driver acceptance on real vendor
# compilers without hardware.
#
# - Two stages. `mesa` builds Mesa MESA_VERSION from its release tarball (checked against
#   MESA_SHA256) with the RADV (amd), ANV (intel), and NVK (nouveau) Vulkan drivers and their noop
#   drm-shims into /opt/mesa, and nothing else: no gallium, GL, EGL, GBM, or window system. The
#   final stage is the Linux gate's toolchain (Ubuntu 26.04 by digest, LLVM, SPIR-V tools,
#   lavapipe, the validation layer, Rust, cargo-nextest; not SwiftShader or the profiles layer)
#   plus vulkaninfo and a copy of /opt/mesa, so no Mesa build dependency or build tree is kept.
# - A drm-shim is an LD_PRELOAD library that answers the driver's DRM ioctls as a kernel would for
#   a chosen GPU and never touches hardware: the driver compiles every pipeline for that GPU, and
#   nothing executes. `scripts/linux-gate` selects the shim, the GPU, and the ICD;
#   `scripts/drm-shim-exec` applies them to one process.
# - MESA_VERSION is 26.2.3: the release the RADV and ANV profiles were captured on
#   (amd-rdna3-radv, amd-rdna2-apu-radv, intel-arl-anv; intel-irisxe-anv on 26.2.2), and the first
#   pinned release whose nouveau shim answers Ada and Blackwell (26.1.0's answers chipsets up to
#   Ampere, so NVK took a later chip for Fermi and refused it). Re-read the facts below on a bump.
# - What the pin's sources say:
#   - Shims, installed into the libdir (`install : true`): libamdgpu_noop_drm_shim.so
#     (src/amd/drm-shim), libnouveau_noop_drm_shim.so (src/nouveau/drm-shim), and
#     libintel_noop_drm_shim.so, which src/intel/tools/meson.build builds only when the Intel tools
#     are built: hence `-Dtools=drm-shim,intel`. `--libdir=lib`, because meson's default libdir on
#     Debian-like systems is lib/<multiarch triple>.
#   - ICD manifests: <datadir>/vulkan/icd.d/{radeon,intel,nouveau}_icd.<cpu>.json
#     (vulkan-manifest-per-architecture defaults to true), i.e. /opt/mesa/share/vulkan/icd.d.
#   - GPU selection: AMDGPU_GPU_ID, a name from src/amd/common/amdgpu_devices.c, compared without
#     case (an unknown name aborts); INTEL_STUB_GPU_PLATFORM, a name from the table in
#     src/intel/dev/intel_device_info.c, compared exactly (an unknown name leaves the shim
#     uninitialized); NOUVEAU_CHIPSET, a hexadecimal chipset id (strtol base 16, default f0).
# - Build dependencies are Ubuntu's own (its mesa source package builds these drivers with them):
#   LLVM LLVM_VERSION with clang and the SPIRV-LLVM-Translator for mesa_clc (ANV's and NVK's
#   internal OpenCL shaders), SPIRV-Tools, glslang, cbindgen, and bindgen (0.72.1, the version
#   Mesa's CI builds). RADV compiles with ACO only (-Damd-use-llvm=false). NVK's compiler (NAK) is
#   Rust: rustc MESA_RUST_VERSION (Mesa's CI toolchain for this release) and the crates syn,
#   proc-macro2, quote, unicode-ident, paste, and rustc-hash, Mesa's meson wraps, each fetched by
#   its pinned source hash while the image build has network; setup runs with
#   --wrap-mode=nodownload, so nothing is fetched later or silently skipped.
# - libdrm: Mesa 26.2.3 needs libdrm and libdrm_amdgpu 2.4.133 (meson.build `_drm_amdgpu_ver`), and
#   Ubuntu 26.04 has 2.4.131. Mesa's own libdrm wrap (2.4.133, fetched by its source hash like the
#   crates) is built through Mesa's `allow-fallback-for` option, always (`--force-fallback-for`,
#   so a newer system libdrm changes nothing), as a static library linked into the drivers and
#   shims: they never load the system's libdrm, which lavapipe uses. No subproject is installed
#   but the wrap's amdgpu.ids, the ASIC table libdrm_amdgpu reads at <prefix>/share/libdrm (the
#   path compiled into it) to name the device: without it every process that opens the amdgpu
#   device reports the missing file, and RADV's device name omits the marketing name.
# - Compile steps run with JOBS jobs: `scripts/linux-gate` passes its --cpus share, because
#   BuildKit takes no per-build --cpus/--memory. The final stage copies /opt/mesa before anything
#   else, so it starts only when the Mesa build has finished and the two stages never compile at
#   once. Each compile step prints its peak memory (cgroup v2 memory.peak) last.
# - The image carries the marker /etc/m2v-drm-shim-image; scripts/drm-shim-exec refuses to run
#   anywhere else (a preloaded shim must never sit next to real hardware).
FROM ubuntu:26.04@sha256:da6fc2be547864451aa253836dd926da33623312df4a9a243e35dc877c378a78 AS mesa

ARG MESA_VERSION=26.2.3
ARG MESA_SHA256=1628058a8d2c0615975de5a15ab7bbb9638c50000b5bed9456ff423ea034a81f
ARG MESA_RUST_VERSION=1.96.0
ARG LLVM_VERSION=21

ENV DEBIAN_FRONTEND=noninteractive \
    RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/opt/cargo \
    PATH=/opt/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates curl xz-utils build-essential pkgconf python3 python3-mako \
        python3-yaml python3-packaging meson ninja-build bison flex glslang-tools cbindgen \
        bindgen libdrm-dev libelf-dev libexpat1-dev libzstd-dev zlib1g-dev \
        "llvm-${LLVM_VERSION}-dev" "libclang-${LLVM_VERSION}-dev" \
        "libclang-cpp${LLVM_VERSION}-dev" "libllvmspirvlib-${LLVM_VERSION}-dev" spirv-tools-dev; \
    rm -rf /var/lib/apt/lists/*; \
    meson --version; bindgen --version; cbindgen --version; glslangValidator --version; \
    "llvm-config-${LLVM_VERSION}" --version

RUN set -eux; \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup-init.sh; \
    sh /tmp/rustup-init.sh -y --no-modify-path --profile minimal \
        --default-toolchain "${MESA_RUST_VERSION}" -c rustfmt; \
    rm /tmp/rustup-init.sh; \
    rustc --version

ARG JOBS=1

RUN set -eux; \
    curl --proto '=https' --tlsv1.2 -sSfL \
        "https://archive.mesa3d.org/mesa-${MESA_VERSION}.tar.xz" -o /tmp/mesa.tar.xz; \
    echo "${MESA_SHA256}  /tmp/mesa.tar.xz" | sha256sum -c -; \
    mkdir /tmp/mesa; tar -xJf /tmp/mesa.tar.xz -C /tmp/mesa --strip-components=1; \
    cd /tmp/mesa; \
    meson subprojects download syn-2-rs proc-macro2-1-rs quote-1-rs unicode-ident-1-rs \
        paste-1-rs rustc-hash-2-rs libdrm; \
    printf "[binaries]\nllvm-config = '/usr/bin/llvm-config-%s'\n" "${LLVM_VERSION}" > native.ini; \
    meson setup build --native-file native.ini --prefix=/opt/mesa --libdir=lib \
        --buildtype=release --wrap-mode=nodownload \
        -Dvulkan-drivers=amd,intel,nouveau -Dgallium-drivers= -Dtools=drm-shim,intel \
        -Dplatforms= -Dglx=disabled -Degl=disabled -Dgbm=disabled -Dgles1=disabled \
        -Dgles2=disabled -Dopengl=false -Dglvnd=disabled -Dvideo-codecs= -Dllvm=enabled \
        -Damd-use-llvm=false \
        -Dvalgrind=disabled -Dlibunwind=disabled -Dlmsensors=disabled \
        -Dallow-fallback-for=libdrm --force-fallback-for=libdrm -Dlibdrm:default_library=static; \
    ninja -j "${JOBS}" -C build; \
    meson install -C build --no-rebuild --skip-subprojects; \
    install -D -m 0644 subprojects/libdrm-*/data/amdgpu.ids /opt/mesa/share/libdrm/amdgpu.ids; \
    cd /; rm -rf /tmp/mesa /tmp/mesa.tar.xz /opt/cargo/registry; \
    ls -l /opt/mesa/lib/libamdgpu_noop_drm_shim.so /opt/mesa/lib/libintel_noop_drm_shim.so \
        /opt/mesa/lib/libnouveau_noop_drm_shim.so; \
    ls -l /opt/mesa/share/vulkan/icd.d/ /opt/mesa/share/libdrm/; \
    cat /sys/fs/cgroup/memory.peak || true

FROM ubuntu:26.04@sha256:da6fc2be547864451aa253836dd926da33623312df4a9a243e35dc877c378a78

# First, so nothing below runs (or compiles) while the mesa stage builds.
COPY --from=mesa /opt/mesa /opt/mesa

ARG RUST_VERSION=nightly-2026-11-03
ARG NEXTEST_VERSION=0.9.140

ENV DEBIAN_FRONTEND=noninteractive \
    RUSTUP_HOME=/opt/rustup \
    PATH=/opt/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# The gate's packages, vulkaninfo, and the drivers' runtime libraries (libdrm is linked in). Every
# library the copied drivers and shims link must resolve.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates curl build-essential pkg-config clang \
        llvm spirv-tools mesa-vulkan-drivers libvulkan1 vulkan-validationlayers python3 \
        vulkan-tools libelf1t64 libexpat1 libzstd1 zlib1g; \
    rm -rf /var/lib/apt/lists/*; \
    command -v llvm-dis; command -v llvm-as; command -v spirv-val; command -v spirv-as; \
    command -v vulkaninfo; \
    ls /usr/share/vulkan/icd.d/lvp_icd*.json; \
    ls /usr/share/vulkan/explicit_layer.d/VkLayer_khronos_validation.json; \
    for lib in /opt/mesa/lib/*.so; do \
        if ldd "$lib" | grep 'not found'; then echo "unresolved: $lib" >&2; exit 1; fi; \
    done; \
    for icd in radeon intel nouveau; do ls "/opt/mesa/share/vulkan/icd.d/${icd}_icd".*.json; done; \
    test -s /opt/mesa/share/libdrm/amdgpu.ids

ARG JOBS=1

RUN set -eux; \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup-init.sh; \
    CARGO_HOME=/opt/cargo sh /tmp/rustup-init.sh -y --no-modify-path --profile minimal \
        --default-toolchain "${RUST_VERSION}" -c rustfmt -c clippy; \
    rm /tmp/rustup-init.sh; \
    CARGO_HOME=/opt/cargo cargo install -j "${JOBS}" --locked cargo-nextest \
        --version "${NEXTEST_VERSION}"; \
    rm -rf /opt/cargo/registry /opt/cargo/git; \
    rustc --version; cargo --version; cargo nextest --version; \
    cat /sys/fs/cgroup/memory.peak || true; \
    echo "m2v drm-shim image: no GPU, no /dev/dri; see scripts/drm-shim-exec" \
        > /etc/m2v-drm-shim-image
