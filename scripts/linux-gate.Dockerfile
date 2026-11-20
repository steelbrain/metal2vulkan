# Pinned image for `scripts/linux-gate`: the Linux CI job of .github/workflows/ci.yml, locally, plus
# the profiles layer the validation spec asks for (§4.1.2, §4.1.4).
#
# - Base: Ubuntu 26.04 (the CI runner label `ubuntu-26.04`), pinned by multi-arch index digest.
#   The host builds its native architecture (arm64 on Apple silicon); the GitHub runner is x86_64.
# - Packages: the CI "Install LLVM + SPIR-V tools (Ubuntu)" list (llvm, spirv-tools,
#   mesa-vulkan-drivers, libvulkan1, vulkan-validationlayers), plus what the GitHub runner image
#   already ships and the build needs (a C toolchain for linking and bundled SQLite, clang for the
#   tests that probe it), `vulkan-tools` (vulkaninfo), and what the source build below needs
#   (git, cmake, ninja).
# - Rust: the dated nightly `rust-toolchain.toml` names, with rustfmt + clippy, as in CI. Keep
#   RUST_VERSION equal to that file's channel.
# - cargo-nextest: `taiki-e/install-action` installs the latest in CI; pinned here by
#   NEXTEST_VERSION and built from crates.io with its lockfile.
# - VK_LAYER_KHRONOS_profiles (Khronos Vulkan-Profiles; Ubuntu does not package it): built from
#   source at VULKAN_PROFILES_COMMIT, the `vulkan-sdk-1.4.341.0` tag, the SDK release of Ubuntu's
#   loader and validation layer. Its build fetches Vulkan-Headers, Vulkan-Utility-Libraries (both
#   v1.4.341), jsoncpp (1.9.6), and valijson (v1.0.2) at the tags pinned in that commit's
#   scripts/known_good.json. Only the layer target is built; its manifest goes to
#   /usr/local/share/vulkan/explicit_layer.d, so the loader lists it, and it is active only when
#   enabled (`scripts/linux-gate --profile P`).
# - Compile steps run with JOBS jobs: `scripts/linux-gate` passes its --cpus share, because
#   BuildKit takes no per-build --cpus/--memory.
#
# Changing this file changes the image tag `scripts/linux-gate` computes, so the next run rebuilds.
FROM ubuntu:26.04@sha256:da6fc2be547864451aa253836dd926da33623312df4a9a243e35dc877c378a78

ARG RUST_VERSION=nightly-2026-11-03
ARG NEXTEST_VERSION=0.9.140

ENV DEBIAN_FRONTEND=noninteractive \
    RUSTUP_HOME=/opt/rustup \
    PATH=/opt/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates curl build-essential pkg-config clang \
        llvm spirv-tools mesa-vulkan-drivers libvulkan1 vulkan-validationlayers python3 \
        vulkan-tools git cmake ninja-build; \
    rm -rf /var/lib/apt/lists/*; \
    command -v llvm-dis; command -v llvm-as; command -v spirv-val; command -v spirv-as; command -v python3; \
    command -v vulkaninfo; \
    ls /usr/share/vulkan/icd.d/lvp_icd*.json; \
    ls /usr/share/vulkan/explicit_layer.d/VkLayer_khronos_validation.json

# The CPU share scripts/linux-gate assigns (--cpus); every compile step below runs with this many
# jobs. Declared after the package layer so a different share does not invalidate it.
ARG JOBS=1

RUN set -eux; \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup-init.sh; \
    CARGO_HOME=/opt/cargo sh /tmp/rustup-init.sh -y --no-modify-path --profile minimal \
        --default-toolchain "${RUST_VERSION}" -c rustfmt -c clippy; \
    rm /tmp/rustup-init.sh; \
    CARGO_HOME=/opt/cargo cargo install -j "${JOBS}" --locked cargo-nextest \
        --version "${NEXTEST_VERSION}"; \
    rm -rf /opt/cargo/registry /opt/cargo/git; \
    rustc --version; cargo --version; cargo fmt --version; cargo clippy --version; \
    cargo nextest --version

# The `vulkan-sdk-1.4.341.0` tag.
ARG VULKAN_PROFILES_COMMIT=a400d0b10059b2cc3cd285475593eee79975a5c4

# update_deps.py builds the dependencies with `--parallel <cpu count>`; PYTHON_CPU_COUNT (Python
# 3.13+) makes that count JOBS.
RUN set -eux; \
    mkdir /tmp/vulkan-profiles; cd /tmp/vulkan-profiles; \
    git init -q; \
    git fetch -q --depth 1 https://github.com/KhronosGroup/Vulkan-Profiles.git \
        "${VULKAN_PROFILES_COMMIT}"; \
    git checkout -q FETCH_HEAD; \
    test "$(git rev-parse HEAD)" = "${VULKAN_PROFILES_COMMIT}"; \
    PYTHON_CPU_COUNT="${JOBS}" cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DUPDATE_DEPS=ON -DBUILD_TESTS=OFF; \
    ninja -j "${JOBS}" -C build ProfilesLayer; \
    install -m 0644 build/layer/libVkLayer_khronos_profiles.so /usr/local/lib/; \
    install -d /usr/local/share/vulkan/explicit_layer.d; \
    install -m 0644 build/layer/json/unix_install_profile.json \
        /usr/local/share/vulkan/explicit_layer.d/VkLayer_khronos_profiles.json; \
    ldconfig; \
    cd /; rm -rf /tmp/vulkan-profiles; \
    cat /usr/local/share/vulkan/explicit_layer.d/VkLayer_khronos_profiles.json; \
    cat /sys/fs/cgroup/memory.peak || true
