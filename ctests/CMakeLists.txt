cmake_minimum_required(VERSION 3.16)
project(opus_ctests C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(OPUS_ROOT "${CMAKE_CURRENT_LIST_DIR}/../opus-c")

set(OPUS_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(OPUS_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
set(OPUS_BUILD_SHARED_LIBRARY OFF CACHE BOOL "" FORCE)
set(OPUS_FIXED_POINT OFF CACHE BOOL "" FORCE)
set(OPUS_ENABLE_FLOAT_API ON CACHE BOOL "" FORCE)

add_subdirectory("${OPUS_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/opus-c-build" EXCLUDE_FROM_ALL)

target_compile_definitions(opus PRIVATE RESYNTH)

add_executable(celt_compute_band_energies_test compute_band_energies_test.c)
add_executable(celt_mdct_backward_compare mdct_backward_compare.c)
add_executable(celt_synthesis_compare celt_synthesis_compare.c)
add_executable(opus_packet_encode opus_packet_encode.c)
add_executable(opus_packet_decode opus_packet_decode.c)

target_include_directories(celt_compute_band_energies_test
  PRIVATE
    "${OPUS_ROOT}/celt"
    "${OPUS_ROOT}/include"
)

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

target_include_directories(opus_packet_encode
  PRIVATE
    "${OPUS_ROOT}/include"
)

target_include_directories(opus_packet_decode
  PRIVATE
    "${OPUS_ROOT}/include"
)

target_link_libraries(celt_compute_band_energies_test PRIVATE opus)
target_link_libraries(celt_mdct_backward_compare PRIVATE opus)
target_link_libraries(celt_synthesis_compare PRIVATE opus)
target_link_libraries(opus_packet_encode PRIVATE opus)
target_link_libraries(opus_packet_decode PRIVATE opus)

target_compile_definitions(celt_compute_band_energies_test PRIVATE OPUS_BUILD)
target_compile_definitions(celt_mdct_backward_compare PRIVATE OPUS_BUILD)
target_compile_definitions(celt_synthesis_compare PRIVATE OPUS_BUILD RESYNTH)
target_compile_definitions(opus_packet_encode PRIVATE OPUS_BUILD)
target_compile_definitions(opus_packet_decode PRIVATE OPUS_BUILD)

if (OPUS_VAR_ARRAYS)
  target_compile_definitions(celt_compute_band_energies_test PRIVATE VAR_ARRAYS)
  target_compile_definitions(celt_mdct_backward_compare PRIVATE VAR_ARRAYS)
  target_compile_definitions(celt_synthesis_compare PRIVATE VAR_ARRAYS)
elseif (OPUS_USE_ALLOCA)
  target_compile_definitions(celt_compute_band_energies_test PRIVATE USE_ALLOCA)
  target_compile_definitions(celt_mdct_backward_compare PRIVATE USE_ALLOCA)
  target_compile_definitions(celt_synthesis_compare PRIVATE USE_ALLOCA)
elseif (OPUS_NONTHREADSAFE_PSEUDOSTACK)
  target_compile_definitions(celt_compute_band_energies_test PRIVATE NONTHREADSAFE_PSEUDOSTACK)
  target_compile_definitions(celt_mdct_backward_compare PRIVATE NONTHREADSAFE_PSEUDOSTACK)
  target_compile_definitions(celt_synthesis_compare PRIVATE NONTHREADSAFE_PSEUDOSTACK)
else()
  target_compile_definitions(celt_compute_band_energies_test PRIVATE VAR_ARRAYS)
  target_compile_definitions(celt_mdct_backward_compare PRIVATE VAR_ARRAYS)
  target_compile_definitions(celt_synthesis_compare PRIVATE VAR_ARRAYS)
endif()

find_library(MATH_LIBRARY m)
if (MATH_LIBRARY)
  target_link_libraries(celt_compute_band_energies_test PRIVATE "${MATH_LIBRARY}")
  target_link_libraries(celt_mdct_backward_compare PRIVATE "${MATH_LIBRARY}")
  target_link_libraries(celt_synthesis_compare PRIVATE "${MATH_LIBRARY}")
endif()

enable_testing()
add_test(NAME celt_compute_band_energies COMMAND celt_compute_band_energies_test)
add_test(NAME celt_mdct_backward_compare COMMAND celt_mdct_backward_compare)
add_test(NAME celt_synthesis_compare COMMAND celt_synthesis_compare)
