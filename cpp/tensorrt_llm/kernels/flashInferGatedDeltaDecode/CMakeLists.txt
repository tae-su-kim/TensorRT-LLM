# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

file(GLOB SRC_CU *.cu)

add_library(flashinfer_gdn_decode_src STATIC ${SRC_CU})
set_property(TARGET flashinfer_gdn_decode_src PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET flashinfer_gdn_decode_src PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_property(TARGET flashinfer_gdn_decode_src PROPERTY CXX_STANDARD 20)
set_property(TARGET flashinfer_gdn_decode_src PROPERTY CXX_STANDARD_REQUIRED YES)
set_property(TARGET flashinfer_gdn_decode_src PROPERTY CXX_EXTENSIONS NO)
set_property(TARGET flashinfer_gdn_decode_src PROPERTY CUDA_STANDARD 20)

target_include_directories(
  flashinfer_gdn_decode_src
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PUBLIC
    $<TARGET_PROPERTY:${INTERNAL_CUTLASS_KERNELS_TARGET},INTERFACE_INCLUDE_DIRECTORIES>
)

set_cuda_architectures(flashinfer_gdn_decode_src 90)
