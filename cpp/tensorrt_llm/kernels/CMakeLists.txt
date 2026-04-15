#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#

add_subdirectory(cutlass_kernels)
add_subdirectory(cuteDslKernels)
if(BUILD_FLASH_MLA)
  add_subdirectory(flashMLA)
endif()

if(BUILD_CONTEXT_FMHA)
  add_subdirectory(contextFusedMultiHeadAttention)
endif()

if(BUILD_DECODER_ATTENTION)
  add_subdirectory(decoderMaskedMultiheadAttention)
endif()
add_subdirectory(selectiveScan)
add_subdirectory(userbuffers)
add_subdirectory(trtllmGenKernels)
add_subdirectory(fusedLayernormKernels)
add_subdirectory(groupRmsNormKernels)
add_subdirectory(llama4MinLatencyKernels)
add_subdirectory(dsv3MinLatencyKernels)
add_subdirectory(causalConv1d)
add_subdirectory(fusedGatedRMSNormQuant)
add_subdirectory(mamba2MTPSSMCache)
add_subdirectory(flashInferGatedDeltaPrefill)
add_subdirectory(flashInferGatedDeltaDecode)

file(GLOB_RECURSE SRC_CPP *.cpp)
file(GLOB_RECURSE SRC_CU *.cu)

# Exclude files in the cutlass_kernels, decoderMaskedMultiheadAttention and
# selectiveScan trtllmGenKernels folder
list(FILTER SRC_CPP EXCLUDE REGEX "cutlass_kernels/.*")
list(FILTER SRC_CU EXCLUDE REGEX "cutlass_kernels/.*")
list(FILTER SRC_CPP EXCLUDE REGEX "flashInferGatedDeltaDecode/.*")
list(FILTER SRC_CU EXCLUDE REGEX "flashInferGatedDeltaDecode/.*")
list(FILTER SRC_CPP EXCLUDE REGEX "flashInferGatedDeltaPrefill/.*")
list(FILTER SRC_CU EXCLUDE REGEX "flashInferGatedDeltaPrefill/.*")
list(FILTER SRC_CPP EXCLUDE REGEX "cuteDslKernels/.*")
list(FILTER SRC_CU EXCLUDE REGEX "cuteDslKernels/.*")
list(FILTER SRC_CPP EXCLUDE REGEX "flashMLA/.*")
list(FILTER SRC_CU EXCLUDE REGEX "flashMLA/.*")
list(FILTER SRC_CPP EXCLUDE REGEX "contextFusedMultiHeadAttention/.*")
list(FILTER SRC_CU EXCLUDE REGEX "contextFusedMultiHeadAttention/.*")
list(FILTER SRC_CPP EXCLUDE REGEX "decoderMaskedMultiheadAttention/.*")
list(FILTER SRC_CU EXCLUDE REGEX "decoderMaskedMultiheadAttention/.*")
list(FILTER SRC_CPP EXCLUDE REGEX "trtllmGenKernels/.*")
list(FILTER SRC_CU EXCLUDE REGEX "trtllmGenKernels/.*")
list(FILTER SRC_CPP EXCLUDE REGEX "selectiveScan/.*")
list(FILTER SRC_CU EXCLUDE REGEX "selectiveScan/.*")
list(FILTER SRC_CPP EXCLUDE REGEX "userbuffers/.*")
list(FILTER SRC_CU EXCLUDE REGEX "userbuffers/.*")
list(FILTER SRC_CU EXCLUDE REGEX "fusedLayernormKernels/.*")
list(FILTER SRC_CU EXCLUDE REGEX "fusedGatedRMSNormQuant/.*")
list(FILTER SRC_CU EXCLUDE REGEX "mamba2MTPSSMCache/.*")

if(NOT ENABLE_MULTI_DEVICE)
  list(FILTER SRC_CU EXCLUDE REGEX "customAllReduceKernels*.*cu$")
endif()

# Lower CUDA optimization level when FAST_BUILD is enabled to speed up
# compilation. CMake by default adds -O3 to CMAKE_CUDA_FLAGS_RELEASE. We
# override it here before creating the target to avoid the "incompatible
# redefinition" warning.
if(FAST_BUILD)
  string(REPLACE "-O3" "-O1" CMAKE_CUDA_FLAGS_RELEASE
                 "${CMAKE_CUDA_FLAGS_RELEASE}")
  string(REPLACE "-O3" "-O1" CMAKE_CUDA_FLAGS_RELWITHDEBINFO
                 "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO}")
  message(
    STATUS "FAST_BUILD enabled for kernels: using -O1 for CUDA compilation")
endif()

add_library(kernels_src STATIC ${SRC_CPP} ${SRC_CU})
set_property(TARGET kernels_src PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET kernels_src PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_include_directories(
  kernels_src
  PUBLIC
    $<TARGET_PROPERTY:${INTERNAL_CUTLASS_KERNELS_TARGET},INTERFACE_INCLUDE_DIRECTORIES>
)
target_link_libraries(kernels_src PUBLIC trtllm_gen_fmha_interface flashinfer_gdn_prefill_src
                                         flashinfer_gdn_decode_src)
add_cuda_architectures(kernels_src 89)
