# ggml-speech: tetherto/qvac-ext-ggml@speech HEAD 11581cd5 (merge of PR #35,
# ggml-vulkan: optimize parakeet/whisper Vulkan on ARM Mali). Two
# ARM/Mali-gated changes: disable KHR_cooperative_matrix (Mali's coopmat matmul
# is ~5x slower than the scalar path for our encoder shapes) and use the medium
# (64x64) matmul tile instead of large (128x128) on ARM (few shader cores leave
# tall-K/small-N GEMMs underutilized). Bumps the speech-stack ggml pin from
# f5727c32; off ARM (desktop NVIDIA/AMD/Intel, Adreno, Apple) byte-identical.

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO tetherto/qvac-ext-ggml
    REF e6ddb8a0ec139871862eda1035765104a95e0baf
    SHA512 a992d37c57d868caf2695c776d1fcfe26e3b62f6ab4c58cb0d2b4ccf2b7d85f93453230c1e2278eaf792196a4b596d8a37b924d58bbefeb3ba8828cfc3eeb0bb
    HEAD_REF speech
)

set(GGML_METAL  OFF)
set(GGML_VULKAN OFF)
set(GGML_CUDA   OFF)
set(GGML_OPENCL OFF)
set(GGML_METAL_FUSE_MV_BIAS OFF)

if("metal" IN_LIST FEATURES)
    set(GGML_METAL ON)
endif()

# Off by default: the chatterbox Q-variant mul_mv + bias/residual fusion
# produces zero tokens on parakeet's EOU q8_0 joint network. Consumers
# whose models stay clear of that pattern can opt in for the speedup.
if("metal-fuse-mv-bias" IN_LIST FEATURES)
    set(GGML_METAL_FUSE_MV_BIAS ON)
endif()

if("vulkan" IN_LIST FEATURES)
    set(GGML_VULKAN ON)
endif()

set(GGML_CUDA_COMPILER_OPTION "")

if("cuda" IN_LIST FEATURES)
    set(GGML_CUDA ON)
    find_program(NVCC_EXECUTABLE nvcc
        PATHS /usr/local/cuda/bin /usr/local/cuda-12.8/bin
        NO_DEFAULT_PATH
    )
    if(NOT NVCC_EXECUTABLE)
        find_program(NVCC_EXECUTABLE nvcc REQUIRED)
    endif()
    set(GGML_CUDA_COMPILER_OPTION "-DCMAKE_CUDA_COMPILER=${NVCC_EXECUTABLE}")
    message(STATUS "CUDA compiler: ${NVCC_EXECUTABLE}")
endif()

if("opencl" IN_LIST FEATURES)
    set(GGML_OPENCL ON)
endif()

if(VCPKG_TARGET_IS_ANDROID AND "vulkan" IN_LIST FEATURES)
    include(${CMAKE_CURRENT_LIST_DIR}/android-vulkan-version.cmake)
    detect_ndk_vulkan_version()
    message(STATUS "NDK Vulkan version: ${vulkan_version}")

    file(DOWNLOAD
        "https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/v${vulkan_version}.tar.gz"
        "${SOURCE_PATH}/vulkan-hpp-${vulkan_version}.tar.gz"
        TLS_VERIFY ON
    )
    file(ARCHIVE_EXTRACT
        INPUT "${SOURCE_PATH}/vulkan-hpp-${vulkan_version}.tar.gz"
        DESTINATION "${SOURCE_PATH}"
        PATTERNS "*.hpp"
    )
    file(COPY "${SOURCE_PATH}/Vulkan-Headers-${vulkan_version}/include/"
         DESTINATION "${SOURCE_PATH}/src/")
endif()

set(PLATFORM_OPTIONS)

if(VCPKG_TARGET_IS_IOS)
    list(APPEND PLATFORM_OPTIONS -DGGML_BLAS=OFF -DGGML_ACCELERATE=OFF)
endif()

# Hybrid Android backend mode: GPU backends as MODULE .so loaded at runtime
# via dlopen, CPU built as per-arch MODULE .so variants (one per ARMv8.0/
# 8.2/8.6/9.0/9.2 feature tier) also loaded at runtime via dlopen. The
# downstream addon installs the resulting libqvac-speech-ggml-cpu-android_armv*
# .so files alongside the .bare binary; the per-variant scoring in
# ggml-cpu's `ggml_backend_cpu_aarch64_score` then picks the highest tier
# the running device supports at first use. Pairs with the speech-branch
# `ggml-backend: android per-arch CPU variant dlopen fallback` patch
# (commit 9562ed04) so the variant lookup also succeeds when the consumer
# APK keeps native .so files compressed (AGP `useLegacyPackaging=false`).
if(VCPKG_TARGET_IS_ANDROID)
    list(APPEND PLATFORM_OPTIONS
        -DGGML_BACKEND_DL=ON
        -DGGML_CPU_ALL_VARIANTS=ON
        -DGGML_CPU_REPACK=ON
        -DGGML_VULKAN_DISABLE_COOPMAT=ON
        -DGGML_VULKAN_DISABLE_COOPMAT2=ON
    )
endif()

# PR #13 (v0.10.2 sync) introduces an unconditional
# `#include <spirv/unified1/spirv.hpp>` in src/ggml-vulkan/ggml-vulkan.cpp,
# but the upstream ggml-vulkan CMakeLists.txt never finds spirv-headers nor
# wires its include dir into the ggml-vulkan target. Apply a small patch
# so it does (and depend on spirv-headers in vcpkg.json's vulkan feature).
# TODO: push the equivalent fix upstream and drop this patch.
if("vulkan" IN_LIST FEATURES)
    vcpkg_apply_patches(
        SOURCE_PATH "${SOURCE_PATH}"
        PATCHES
            "${CMAKE_CURRENT_LIST_DIR}/patches/0001-ggml-vulkan-find-spirv-headers.patch"
    )
endif()

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DBUILD_SHARED_LIBS=OFF
        -DGGML_NATIVE=OFF
        -DGGML_CCACHE=OFF
        -DGGML_OPENMP=OFF
        -DGGML_LLAMAFILE=OFF
        -DGGML_BUILD_TESTS=OFF
        -DGGML_BUILD_EXAMPLES=OFF
        -DGGML_METAL=${GGML_METAL}
        -DGGML_VULKAN=${GGML_VULKAN}
        -DGGML_CUDA=${GGML_CUDA}
        -DGGML_OPENCL=${GGML_OPENCL}
        -DGGML_METAL_FUSE_MV_BIAS=${GGML_METAL_FUSE_MV_BIAS}
        -DGGML_LIB_OUTPUT_PREFIX=qvac-speech-
        ${GGML_CUDA_COMPILER_OPTION}
        ${PLATFORM_OPTIONS}
)

vcpkg_cmake_install()

# Pick up the MODULE backend .so files ggml builds into the buildtree's
# bin/ directory (Android dynamic-backend mode). cmake install() doesn't
# move them by default.
if(VCPKG_TARGET_IS_ANDROID)
    file(GLOB _backend_sos
        "${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-rel/bin/libqvac-speech-ggml-*.so"
    )
    if(_backend_sos)
        file(INSTALL ${_backend_sos} DESTINATION "${CURRENT_PACKAGES_DIR}/lib")
    endif()
endif()

vcpkg_cmake_config_fixup(PACKAGE_NAME ggml CONFIG_PATH lib/cmake/ggml)

if(EXISTS "${CURRENT_PACKAGES_DIR}/share/pkgconfig/ggml.pc")
    file(MAKE_DIRECTORY "${CURRENT_PACKAGES_DIR}/lib/pkgconfig")
    file(RENAME "${CURRENT_PACKAGES_DIR}/share/pkgconfig/ggml.pc"
                "${CURRENT_PACKAGES_DIR}/lib/pkgconfig/ggml.pc")
endif()
if(EXISTS "${CURRENT_PACKAGES_DIR}/debug/share/pkgconfig/ggml.pc")
    file(MAKE_DIRECTORY "${CURRENT_PACKAGES_DIR}/debug/lib/pkgconfig")
    file(RENAME "${CURRENT_PACKAGES_DIR}/debug/share/pkgconfig/ggml.pc"
                "${CURRENT_PACKAGES_DIR}/debug/lib/pkgconfig/ggml.pc")
endif()
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/share/pkgconfig"
                    "${CURRENT_PACKAGES_DIR}/debug/share/pkgconfig")
vcpkg_fixup_pkgconfig()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

set(VCPKG_POLICY_MISMATCHED_NUMBER_OF_BINARIES enabled)

file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/usage" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
