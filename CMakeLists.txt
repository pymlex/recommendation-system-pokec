cmake_minimum_required(VERSION 3.10)
project(kurs LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include_directories(include)
include_directories(config)
include_directories(third_party/lemmagen/include)

# Option to enable Matplot++ (OFF by default to avoid platform-specific build issues)
option(USE_MATPLOT "Enable Matplot++ plotting (optional)" OFF)

file(GLOB SRC_FILES "src/*.cpp" "third_party/lemmagen/src/*.cpp")
add_executable(kurs ${SRC_FILES})

# If user enabled plotting and the library is present, add it.
if (USE_MATPLOT)
  if (EXISTS "${CMAKE_SOURCE_DIR}/third_party/matplotplusplus/CMakeLists.txt")
    add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/matplotplusplus EXCLUDE_FROM_ALL)
    target_link_libraries(kurs PUBLIC matplot)
    # Make preprocessor symbol visible to sources
    target_compile_definitions(kurs PRIVATE USE_MATPLOT=1)
  else()
    message(WARNING "USE_MATPLOT is ON but matplotplusplus subdirectory not found. Continuing without plotting.")
  endif()
endif()

if (MSVC)
  target_compile_options(kurs PRIVATE /EHsc)
endif()