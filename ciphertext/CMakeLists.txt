cmake_minimum_required(VERSION 3.21)
project(
  HELLM
  VERSION 0.0.1
  DESCRIPTION "Homomorphic Encryption Library for Language Models"
  LANGUAGES CXX)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

set(PROJECT_LIB_NAME ${PROJECT_NAME})

# Find HEaaN library (using our mock versions)
include_directories(${PROJECT_SOURCE_DIR}/include
                    ${PROJECT_SOURCE_DIR}/external/include)
link_directories(${PROJECT_SOURCE_DIR}/external/lib)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}
                      ${PROJECT_SOURCE_DIR}/external/lib/cmake)

find_package(HEaaN REQUIRED)
find_package(HEaaN-math REQUIRED)

include(BuildType)
include(CTest)
include(CheckLanguage)
include(CCache)

option(BUILD_TEST "Build tests" OFF)
option(BUILD_EXAMPLE "Build examples" ON)
option(ENABLE_MULTIGPU "Enable multi-GPU support (requires MPI and NCCL)." OFF)

set(SRCS
    src/HEMMer.cpp
    src/LoRA.cpp
    src/MatrixUtils.cpp
    src/Loss.cpp
    src/Exp.cpp
    src/LayerNorm.cpp
    src/ReLU.cpp
    src/Tanh.cpp
    src/Softmax.cpp
    src/TorchTransformerBlock.cpp
    src/TransformerBlock.cpp
    CACHE FILEPATH "Sources" FORCE)

add_library(${PROJECT_LIB_NAME} STATIC ${SRCS})

target_include_directories(
  ${PROJECT_LIB_NAME}
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

add_library(${PROJECT_LIB_NAME}::${PROJECT_LIB_NAME} ALIAS ${PROJECT_LIB_NAME})

add_library(external-libs INTERFACE)

# Try to find PyTorch
execute_process(
  COMMAND python3 -c "import torch;print(torch.utils.cmake_prefix_path, end='')"
  OUTPUT_VARIABLE Torch_DIR
  ERROR_QUIET)
if(Torch_DIR)
  set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${Torch_DIR})
  find_package(Torch QUIET)
  if(Torch_FOUND)
    target_include_directories(external-libs INTERFACE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(external-libs INTERFACE torch)
  endif()
endif()

# Create mock libraries for HEaaN
add_library(heaan_mock INTERFACE)
add_library(heaan_math_mock INTERFACE)

target_link_libraries(external-libs INTERFACE heaan_mock heaan_math_mock)
target_link_libraries(${PROJECT_LIB_NAME} PUBLIC $<BUILD_INTERFACE:external-libs>)

# Define exported target
include(GNUInstallDirs)
install(
  TARGETS ${PROJECT_LIB_NAME}
  EXPORT ${PROJECT_LIB_NAME}Targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES
  DESTINATION include
  PUBLIC_HEADER DESTINATION include)

# Export the target
install(
  EXPORT ${PROJECT_LIB_NAME}Targets
  FILE ${PROJECT_LIB_NAME}Targets.cmake
  NAMESPACE ${PROJECT_LIB_NAME}::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_LIB_NAME})

# Create examples directory
if(BUILD_EXAMPLE)
  add_subdirectory(examples)
endif()
