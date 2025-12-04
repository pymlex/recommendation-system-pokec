cmake_minimum_required(VERSION 3.10)
project(kurs LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/config)
include_directories(${CMAKE_SOURCE_DIR}/third_party/lemmagen/include)

option(USE_MATPLOT "Enable Matplot++ plotting (optional)" OFF)

file(GLOB ALL_SRC "${CMAKE_SOURCE_DIR}/src/*.cpp" "${CMAKE_SOURCE_DIR}/third_party/lemmagen/src/*.cpp")

set(CORE_SRCS "")
foreach(f IN LISTS ALL_SRC)
    get_filename_component(fname ${f} NAME)
    if (NOT fname STREQUAL "main.cpp" AND NOT fname STREQUAL "api_cli.cpp")
        list(APPEND CORE_SRCS ${f})
    endif()
endforeach()

add_library(core STATIC ${CORE_SRCS})

if (USE_MATPLOT)
  if (EXISTS "${CMAKE_SOURCE_DIR}/third_party/matplotplusplus/CMakeLists.txt")
    add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/matplotplusplus EXCLUDE_FROM_ALL)
    target_link_libraries(core PUBLIC matplot)
    target_compile_definitions(core PRIVATE USE_MATPLOT=1)
  else()
    message(WARNING "USE_MATPLOT is ON but matplotplusplus subdirectory not found. Continuing without plotting.")
  endif()
endif()

add_executable(kurs ${CMAKE_SOURCE_DIR}/src/main.cpp)
target_link_libraries(kurs PRIVATE core)

add_executable(api_cli ${CMAKE_SOURCE_DIR}/src/api_cli.cpp)
target_link_libraries(api_cli PRIVATE core)

if (MSVC)
  target_compile_options(core PRIVATE /EHsc)
  target_compile_options(kurs PRIVATE /EHsc)
  target_compile_options(api_cli PRIVATE /EHsc)
endif()
