# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ----------------------------------------------------------------------------
# pto_chunk_gdn — pre-compile Bisheng PTO kernels for Qwen3.5/3.6 GDN layer
# ----------------------------------------------------------------------------
#
# This target pre-compiles the PTO C++ kernel sources for a set of common
# model configurations (H, Hg, D, C) known to be used by Qwen3.5/3.6 models.
# The compiled .so files are installed beside the Python module so they are
# found without JIT compilation at inference time.
#
# Kernels are also compiled on first use by
# ``vllm_ascend.ops.pto_chunk_gdn.compile`` (Bisheng JIT), so the cmake
# target is optional — it only provides a pre-warmed cache.
#
# Usage (from repo root):
#   cmake -DBUILD_PTO_CHUNK_GDN=ON -S csrc -B csrc/build
#   cmake --build csrc/build --target pto_chunk_gdn_kernels
#
# Or build everything including PTO:
#   BUILD_PTO_CHUNK_GDN=ON python setup.py build_ext --inplace
# ----------------------------------------------------------------------------

option(BUILD_PTO_CHUNK_GDN "Pre-compile PTO chunk-GDN kernels via Bisheng" OFF)

if(NOT BUILD_PTO_CHUNK_GDN)
    return()
endif()

# ── Locate bisheng ────────────────────────────────────────────────────────────
find_program(BISHENG_EXECUTABLE bisheng
    HINTS ENV ASCEND_HOME_PATH
    PATH_SUFFIXES bin
    DOC "Bisheng C++ compiler for Ascend AI-Core kernels"
)
if(NOT BISHENG_EXECUTABLE)
    message(WARNING "bisheng not found — PTO chunk-GDN kernels will not be pre-compiled."
                    "Set BISHENG_EXECUTABLE or ensure bisheng is in PATH.")
    return()
endif()
message(STATUS "Found bisheng: ${BISHENG_EXECUTABLE}")

# ── Paths ─────────────────────────────────────────────────────────────────────
set(PTO_SRC_DIR    "${CMAKE_CURRENT_SOURCE_DIR}")
set(PTO_INC_DIR    "${PTO_SRC_DIR}/include")
set(PTO_ISA_DIR    "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/pto-isa")
set(DRIVER_INC     "/usr/local/Ascend/driver/kernel/inc")

# Resolve ASCEND_HOME_PATH / ASCEND_TOOLKIT_HOME
if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME "$ENV{ASCEND_HOME_PATH}")
elseif(DEFINED ENV{ASCEND_TOOLKIT_HOME})
    set(ASCEND_HOME "$ENV{ASCEND_TOOLKIT_HOME}")
else()
    message(WARNING "ASCEND_HOME_PATH not set — PTO kernels may not compile correctly.")
    set(ASCEND_HOME "")
endif()

# Install destination inside the Python package
if(NOT DEFINED Python_SITELIB)
    execute_process(
        COMMAND python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
        OUTPUT_VARIABLE Python_SITELIB OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()
set(PTO_COMPILED_INSTALL_DIR
    "${Python_SITELIB}/vllm_ascend/ops/pto_chunk_gdn/kernels/compiled_lib"
)

# ── Common compiler flags ─────────────────────────────────────────────────────
set(PTO_COMMON_FLAGS
    -fPIC -shared -xcce -DMEMORY_BASE -O2 -std=gnu++17
    --cce-aicore-arch=dav-c220
    -mllvm -cce-aicore-stack-size=0x8000
    -mllvm -cce-aicore-function-stack-size=0x8000
    -mllvm -cce-aicore-record-overflow=true
    -mllvm -cce-aicore-dcci-insert-for-scalar=false
    -Wno-macro-redefined -Wno-ignored-attributes
    -I${PTO_INC_DIR}
    -I${PTO_ISA_DIR}/include
    -I${ASCEND_HOME}/include
    -I${ASCEND_HOME}/pkg_inc
    -I${ASCEND_HOME}/pkg_inc/runtime
    -I${ASCEND_HOME}/pkg_inc/profiling
)
if(EXISTS "${DRIVER_INC}")
    list(APPEND PTO_COMMON_FLAGS -I${DRIVER_INC})
endif()

# Chunk-sized kernels use C=128
set(PTO_C 128)

# ── Model configurations to pre-compile ──────────────────────────────────────
# Format: "H;Hg;D"   (H=value heads, Hg=query/key heads, D=head dim)
# Qwen3.5-0.8B / 1.5B / 3B: H=16, Hg=8,  D=128
# Qwen3.5-7B / 14B:          H=16, Hg=16, D=128  (MHA or GQA Hg=8)
# Qwen3.6-27B:               H=32, Hg=16, D=128
# Qwen3.6-35B-A3B-MoE:       H=32, Hg=16, D=128   (shared-expert GDN layers)
# Additional common configs for custom models or future models
set(PTO_MODEL_CONFIGS
    "16;8;128"
    "16;16;128"
    "32;16;128"
    "48;16;128"
    "64;16;128"
    "64;64;128"
)

# Chunk kernels (parameterised by H, Hg, D)
set(PTO_CHUNK_KERNELS
    chunk_cumsum
    scaled_dot_kkt
    wy_fast
    chunk_h
    chunk_o
)

set(_pto_outputs "")
file(MAKE_DIRECTORY "${PTO_COMPILED_INSTALL_DIR}")

foreach(cfg ${PTO_MODEL_CONFIGS})
    string(REPLACE ";" ";" _cfg_list "${cfg}")
    list(GET _cfg_list 0 H)
    list(GET _cfg_list 1 Hg)
    list(GET _cfg_list 2 D)

    set(_def_flags -DGDN_H=${H} -DGDN_HG=${Hg} -DGDN_D=${D} -DGDN_C=${PTO_C})

    foreach(kernel ${PTO_CHUNK_KERNELS})
        set(_src  "${PTO_SRC_DIR}/${kernel}.cpp")
        set(_out  "${PTO_COMPILED_INSTALL_DIR}/${kernel}_H${H}_Hg${Hg}_D${D}_C${PTO_C}.so")
        add_custom_command(
            OUTPUT  "${_out}"
            COMMAND "${BISHENG_EXECUTABLE}"
                    ${PTO_COMMON_FLAGS} ${_def_flags}
                    "${_src}" -o "${_out}"
            DEPENDS "${_src}"
            COMMENT "PTO: compiling ${kernel} H=${H} Hg=${Hg} D=${D} C=${PTO_C}"
            VERBATIM
        )
        list(APPEND _pto_outputs "${_out}")
    endforeach()

    # Megakernel
    set(_mega_src "${PTO_SRC_DIR}/mega_kernel.cpp")
    set(_mega_out "${PTO_COMPILED_INSTALL_DIR}/mega_kernel_H${H}_Hg${Hg}_D${D}_C${PTO_C}.so")
    add_custom_command(
        OUTPUT  "${_mega_out}"
        COMMAND "${BISHENG_EXECUTABLE}"
                ${PTO_COMMON_FLAGS} ${_def_flags}
                "${_mega_src}" -o "${_mega_out}"
        DEPENDS "${_mega_src}"
        COMMENT "PTO: compiling mega_kernel H=${H} Hg=${Hg} D=${D} C=${PTO_C}"
        VERBATIM
    )
    list(APPEND _pto_outputs "${_mega_out}")
endforeach()

# ── tri_inverse (CubeCore kernel, no per-config variants) ─────────────────────
set(_tri_src "${PTO_SRC_DIR}/tri_inverse.cpp")
set(_tri_out "${PTO_COMPILED_INSTALL_DIR}/tri_inverse_jit.so")
add_custom_command(
    OUTPUT  "${_tri_out}"
    COMMAND "${BISHENG_EXECUTABLE}"
            -fPIC -shared -xcce -DMEMORY_BASE -O2 -std=c++17
            -I${PTO_INC_DIR}
            -I${PTO_ISA_DIR}/include
            --cce-soc-version=Ascend910B4
            --cce-soc-core-type=CubeCore
            "${_tri_src}" -o "${_tri_out}"
    DEPENDS "${_tri_src}"
    COMMENT "PTO: compiling tri_inverse (CubeCore)"
    VERBATIM
)
list(APPEND _pto_outputs "${_tri_out}")

# ── Top-level target ──────────────────────────────────────────────────────────
add_custom_target(pto_chunk_gdn_kernels ALL
    DEPENDS ${_pto_outputs}
    COMMENT "Building all PTO chunk-GDN Bisheng kernels"
)
