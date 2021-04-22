cmake_minimum_required(VERSION 3.14)

# Set the compilers based upon the PETSc package
find_package(PkgConfig REQUIRED)
pkg_get_variable(PETSC_C_COMPILER PETSc ccompiler)
set(CMAKE_C_COMPILER ${PETSC_C_COMPILER})
pkg_get_variable(PETSC_CXX_COMPILER PETSc cxxcompiler)
set(CMAKE_CXX_COMPILER ${PETSC_CXX_COMPILER})

# FetContent will allow downloading of other libraries
include(FetchContent)

# ABLATE must be available for the project.  If the ABLATE_PATH is not specified it will download the version specified in FetchContent_Declare
set(ABLATE_PATH "" CACHE PATH "Custom path to local ABLATE project.  If specified this will be used instead of downloading ABLATE.")
message(ABLATE_PATH ${ABLATE_PATH})

if("${ABLATE_PATH}" STREQUAL "")
    message(STATUS "ABLATE_PATH was not specified.  ABLATE is being downloaded and built.")

    # Download and make ablate available to the client
    FetchContent_Declare(
            ablate
            GIT_REPOSITORY https://github.com/UBCHREST/ablate.git
            GIT_TAG v0.2.6
    )
    FetchContent_MakeAvailable(ablate)

else()
    message(STATUS "Configuring with ABLATE_PATH: ${ABLATE_PATH}")
    add_subdirectory(${ABLATE_PATH} ${CMAKE_CURRENT_BINARY_DIR}/ablate-build-directory)
endif()

# Set the project details
project(ablateClientTemplate VERSION 0.0.1)

# Set the c/c++ Standards
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# define the two sample programs and link to ablate
add_executable(ablateLibraryClient ablateLibraryClient.cpp)
target_link_libraries(ablateLibraryClient PUBLIC ablateLibrary)

# define the two sample programs
add_executable(ablateCoreClient ablateCoreClient.c)
target_link_libraries(ablateCoreClient PUBLIC ablateCore)

add_executable(ex11 ex11.c)
target_link_libraries(ex11 PUBLIC ablateCore)

add_executable(euler euler.c)
target_link_libraries(euler PUBLIC ablateCore)

add_executable(eulerFlow eulerFlow.c)
target_link_libraries(eulerFlow PUBLIC ablateCore)

add_executable(ex17 ex17.c)
target_link_libraries(ex17 PUBLIC ablateCore)

add_executable(euler2DMMS euler2DMMS.c)
target_link_libraries(euler2DMMS PUBLIC ablateCore)

add_executable(fvOutput fvOutput.c)
target_link_libraries(fvOutput PUBLIC ablateCore)

add_executable(eulerIsentropicVortex eulerIsentropicVortex.c)
target_link_libraries(eulerIsentropicVortex PUBLIC ablateCore)