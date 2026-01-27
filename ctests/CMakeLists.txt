cmake_minimum_required(VERSION 3.16)
project(opus_ctests C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(OPUS_ROOT "${CMAKE_CURRENT_LIST_DIR}/../opus-c")

set(OPUS_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(OPUS_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
set(OPUS_BUILD_SHARED_LIBRARY OFF CACHE BOOL "" FORCE)

option(OPUS_CTESTS_FIXED_POINT "Build opus-c as fixed-point for ctests" OFF)
option(OPUS_CTESTS_ENABLE_FLOAT_API "Enable opus-c float API for ctests" ON)
set(OPUS_FIXED_POINT ${OPUS_CTESTS_FIXED_POINT} CACHE BOOL "" FORCE)
set(OPUS_ENABLE_FLOAT_API ${OPUS_CTESTS_ENABLE_FLOAT_API} CACHE BOOL "" FORCE)
if (OPUS_CTESTS_FIXED_POINT)
  add_compile_definitions(FIXED_POINT)
endif()

if (OPUS_CTESTS_FIXED_POINT)
  add_library(celt_fixedpoint_ctest_support STATIC
    "${OPUS_ROOT}/celt/bands.c"
    "${OPUS_ROOT}/celt/celt_lpc.c"
    "${OPUS_ROOT}/celt/mathops.c"
    "${OPUS_ROOT}/celt/pitch.c"
    "${CMAKE_CURRENT_LIST_DIR}/opus_celt_stubs.c"
  )
  target_include_directories(celt_fixedpoint_ctest_support
    PRIVATE
      "${OPUS_ROOT}/celt"
      "${OPUS_ROOT}/include"
      "${OPUS_ROOT}/src"
  )
  target_compile_definitions(celt_fixedpoint_ctest_support PRIVATE OPUS_BUILD)
else()
  add_subdirectory("${OPUS_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/opus-c-build" EXCLUDE_FROM_ALL)
  target_compile_definitions(opus PRIVATE RESYNTH)
endif()

option(OPUS_ANALYSIS_STRICT_FP "Enable strict FP flags for analysis.c only" OFF)
if (OPUS_ANALYSIS_STRICT_FP AND NOT OPUS_CTESTS_FIXED_POINT)
  set(OPUS_ANALYSIS_STRICT_FP_FLAGS
    -ffp-model=strict
    -frounding-math
    -ftrapping-math
    -fno-vectorize
    -fno-slp-vectorize
    -fno-associative-math
    -fno-reciprocal-math
  )
  set_source_files_properties(
    "${OPUS_ROOT}/src/analysis.c"
    PROPERTIES COMPILE_OPTIONS "${OPUS_ANALYSIS_STRICT_FP_FLAGS}"
  )
endif()

add_executable(celt_compute_band_energies_test compute_band_energies_test.c)
if (OPUS_CTESTS_FIXED_POINT)
  add_executable(celt_bands_norm_test bands_norm_test.c)
  add_executable(celt_mathops_test mathops_test.c)
  add_executable(celt_pitch_test pitch_test.c)
endif()
if (NOT OPUS_CTESTS_FIXED_POINT)
  add_executable(celt_mdct_backward_compare mdct_backward_compare.c)
  add_executable(celt_synthesis_compare celt_synthesis_compare.c)
  if (OPUS_CTESTS_ENABLE_FLOAT_API)
    add_executable(analysis_compare analysis_compare.c)
    add_executable(analysis_fft_trace analysis_fft_trace.c)
    add_executable(analysis_fft_stage_trace analysis_fft_stage_trace.c)
  endif()
  add_executable(opus_packet_encode opus_packet_encode.c)
  add_executable(opus_packet_decode opus_packet_decode.c)
endif()

target_include_directories(celt_compute_band_energies_test
  PRIVATE
    "${OPUS_ROOT}/celt"
    "${OPUS_ROOT}/include"
)
if (OPUS_CTESTS_FIXED_POINT)
  target_include_directories(celt_bands_norm_test
    PRIVATE
      "${OPUS_ROOT}/celt"
      "${OPUS_ROOT}/include"
  )
  target_include_directories(celt_mathops_test
    PRIVATE
      "${OPUS_ROOT}/celt"
      "${OPUS_ROOT}/include"
  )
  target_include_directories(celt_pitch_test
    PRIVATE
      "${OPUS_ROOT}/celt"
      "${OPUS_ROOT}/include"
  )
endif()

if (NOT OPUS_CTESTS_FIXED_POINT)
  target_include_directories(celt_mdct_backward_compare
    PRIVATE
      "${OPUS_ROOT}/celt"
      "${OPUS_ROOT}/include"
  )

  target_include_directories(celt_synthesis_compare
    PRIVATE
      "${OPUS_ROOT}/celt"
      "${OPUS_ROOT}/include"
  )

  if (OPUS_CTESTS_ENABLE_FLOAT_API)
    target_include_directories(analysis_compare
      PRIVATE
        "${OPUS_ROOT}/celt"
        "${OPUS_ROOT}/include"
        "${OPUS_ROOT}/src"
    )
    target_include_directories(analysis_fft_trace
      PRIVATE
        "${OPUS_ROOT}/celt"
        "${OPUS_ROOT}/include"
        "${OPUS_ROOT}/src"
    )
    target_include_directories(analysis_fft_stage_trace
      PRIVATE
        "${OPUS_ROOT}/celt"
        "${OPUS_ROOT}/include"
        "${OPUS_ROOT}/src"
    )
  endif()

  target_include_directories(opus_packet_encode
    PRIVATE
      "${OPUS_ROOT}/include"
  )

  target_include_directories(opus_packet_decode
    PRIVATE
      "${OPUS_ROOT}/include"
  )
endif()

if (OPUS_CTESTS_FIXED_POINT)
  target_link_libraries(celt_compute_band_energies_test PRIVATE celt_fixedpoint_ctest_support)
  target_link_libraries(celt_bands_norm_test PRIVATE celt_fixedpoint_ctest_support)
  target_link_libraries(celt_mathops_test PRIVATE celt_fixedpoint_ctest_support)
  target_link_libraries(celt_pitch_test PRIVATE celt_fixedpoint_ctest_support)
else()
  target_link_libraries(celt_compute_band_energies_test PRIVATE opus)
  target_link_libraries(celt_mdct_backward_compare PRIVATE opus)
  target_link_libraries(celt_synthesis_compare PRIVATE opus)
  if (OPUS_CTESTS_ENABLE_FLOAT_API)
    target_link_libraries(analysis_compare PRIVATE opus)
    target_link_libraries(analysis_fft_trace PRIVATE opus)
    target_link_libraries(analysis_fft_stage_trace PRIVATE opus)
  endif()
  target_link_libraries(opus_packet_encode PRIVATE opus)
  target_link_libraries(opus_packet_decode PRIVATE opus)
endif()

target_compile_definitions(celt_compute_band_energies_test PRIVATE OPUS_BUILD)
if (OPUS_CTESTS_FIXED_POINT)
  target_compile_definitions(celt_bands_norm_test PRIVATE OPUS_BUILD)
  target_compile_definitions(celt_mathops_test PRIVATE OPUS_BUILD)
  target_compile_definitions(celt_pitch_test PRIVATE OPUS_BUILD)
endif()
if (NOT OPUS_CTESTS_FIXED_POINT)
  target_compile_definitions(celt_mdct_backward_compare PRIVATE OPUS_BUILD)
  target_compile_definitions(celt_synthesis_compare PRIVATE OPUS_BUILD RESYNTH)
  if (OPUS_CTESTS_ENABLE_FLOAT_API)
    target_compile_definitions(analysis_compare PRIVATE OPUS_BUILD)
    target_compile_definitions(analysis_fft_trace PRIVATE OPUS_BUILD)
    target_compile_definitions(analysis_fft_stage_trace PRIVATE OPUS_BUILD)
  endif()
  target_compile_definitions(opus_packet_encode PRIVATE OPUS_BUILD)
  target_compile_definitions(opus_packet_decode PRIVATE OPUS_BUILD)
endif()

if (OPUS_VAR_ARRAYS)
  target_compile_definitions(celt_compute_band_energies_test PRIVATE VAR_ARRAYS)
  if (OPUS_CTESTS_FIXED_POINT)
    target_compile_definitions(celt_bands_norm_test PRIVATE VAR_ARRAYS)
    target_compile_definitions(celt_mathops_test PRIVATE VAR_ARRAYS)
    target_compile_definitions(celt_pitch_test PRIVATE VAR_ARRAYS)
    target_compile_definitions(celt_fixedpoint_ctest_support PRIVATE VAR_ARRAYS)
  endif()
  if (NOT OPUS_CTESTS_FIXED_POINT)
    target_compile_definitions(celt_mdct_backward_compare PRIVATE VAR_ARRAYS)
    target_compile_definitions(celt_synthesis_compare PRIVATE VAR_ARRAYS)
  endif()
elseif (OPUS_USE_ALLOCA)
  target_compile_definitions(celt_compute_band_energies_test PRIVATE USE_ALLOCA)
  if (OPUS_CTESTS_FIXED_POINT)
    target_compile_definitions(celt_bands_norm_test PRIVATE USE_ALLOCA)
    target_compile_definitions(celt_mathops_test PRIVATE USE_ALLOCA)
    target_compile_definitions(celt_pitch_test PRIVATE USE_ALLOCA)
    target_compile_definitions(celt_fixedpoint_ctest_support PRIVATE USE_ALLOCA)
  endif()
  if (NOT OPUS_CTESTS_FIXED_POINT)
    target_compile_definitions(celt_mdct_backward_compare PRIVATE USE_ALLOCA)
    target_compile_definitions(celt_synthesis_compare PRIVATE USE_ALLOCA)
  endif()
elseif (OPUS_NONTHREADSAFE_PSEUDOSTACK)
  target_compile_definitions(celt_compute_band_energies_test PRIVATE NONTHREADSAFE_PSEUDOSTACK)
  if (OPUS_CTESTS_FIXED_POINT)
    target_compile_definitions(celt_bands_norm_test PRIVATE NONTHREADSAFE_PSEUDOSTACK)
    target_compile_definitions(celt_mathops_test PRIVATE NONTHREADSAFE_PSEUDOSTACK)
    target_compile_definitions(celt_pitch_test PRIVATE NONTHREADSAFE_PSEUDOSTACK)
    target_compile_definitions(celt_fixedpoint_ctest_support PRIVATE NONTHREADSAFE_PSEUDOSTACK)
  endif()
  if (NOT OPUS_CTESTS_FIXED_POINT)
    target_compile_definitions(celt_mdct_backward_compare PRIVATE NONTHREADSAFE_PSEUDOSTACK)
    target_compile_definitions(celt_synthesis_compare PRIVATE NONTHREADSAFE_PSEUDOSTACK)
  endif()
else()
  target_compile_definitions(celt_compute_band_energies_test PRIVATE VAR_ARRAYS)
  if (OPUS_CTESTS_FIXED_POINT)
    target_compile_definitions(celt_bands_norm_test PRIVATE VAR_ARRAYS)
    target_compile_definitions(celt_mathops_test PRIVATE VAR_ARRAYS)
    target_compile_definitions(celt_pitch_test PRIVATE VAR_ARRAYS)
    target_compile_definitions(celt_fixedpoint_ctest_support PRIVATE VAR_ARRAYS)
  endif()
  if (NOT OPUS_CTESTS_FIXED_POINT)
    target_compile_definitions(celt_mdct_backward_compare PRIVATE VAR_ARRAYS)
    target_compile_definitions(celt_synthesis_compare PRIVATE VAR_ARRAYS)
  endif()
endif()

find_library(MATH_LIBRARY m)
if (MATH_LIBRARY)
  target_link_libraries(celt_compute_band_energies_test PRIVATE "${MATH_LIBRARY}")
  if (OPUS_CTESTS_FIXED_POINT)
    target_link_libraries(celt_bands_norm_test PRIVATE "${MATH_LIBRARY}")
    target_link_libraries(celt_mathops_test PRIVATE "${MATH_LIBRARY}")
    target_link_libraries(celt_pitch_test PRIVATE "${MATH_LIBRARY}")
  endif()
  if (NOT OPUS_CTESTS_FIXED_POINT)
    target_link_libraries(celt_mdct_backward_compare PRIVATE "${MATH_LIBRARY}")
    target_link_libraries(celt_synthesis_compare PRIVATE "${MATH_LIBRARY}")
  endif()
endif()

enable_testing()
add_test(NAME celt_compute_band_energies COMMAND celt_compute_band_energies_test)
if (OPUS_CTESTS_FIXED_POINT)
  add_test(NAME celt_bands_norm_test COMMAND celt_bands_norm_test)
  add_test(NAME celt_mathops_test COMMAND celt_mathops_test)
  add_test(NAME celt_pitch_test COMMAND celt_pitch_test)
endif()
if (NOT OPUS_CTESTS_FIXED_POINT)
  add_test(NAME celt_mdct_backward_compare COMMAND celt_mdct_backward_compare)
  add_test(NAME celt_synthesis_compare COMMAND celt_synthesis_compare)
  if (OPUS_CTESTS_ENABLE_FLOAT_API)
    add_test(NAME analysis_compare COMMAND analysis_compare)
    add_test(NAME analysis_fft_trace COMMAND analysis_fft_trace)
    add_test(NAME analysis_fft_stage_trace COMMAND analysis_fft_stage_trace)
  endif()
endif()
