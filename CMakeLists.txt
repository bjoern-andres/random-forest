cmake_minimum_required(VERSION 2.6)
project(ml)
include_directories(include)
file(GLOB headers include/andres/* include/andres/ml/*)
enable_testing()

##############################################################################
# Doxygen
##############################################################################
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    message(STATUS "Doxygen found")
else()
    message("doxygen not found")
endif()

##############################################################################
# C++11 support
##############################################################################
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Some functionality will not be available.")
endif()

##############################################################################
# MSVC-specific settings
##############################################################################
if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

##############################################################################
# OpenMP
##############################################################################
find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP found")
else()
    message("OpenMP not found")
endif()

##############################################################################
# targets
##############################################################################
add_executable(test-decision-trees src/unittest/ml/test-decision-trees.cxx ${headers})
add_test(test-decision-trees test-decision-trees)

if(DOXYGEN_FOUND)
    configure_file("${ml_SOURCE_DIR}/doxygen/doxyfile-ml-decision-trees.in" "${ml_BINARY_DIR}/doxyfile-ml-decision-trees" @ONLY IMMEDIATE)
    add_custom_target(doc-ml-decision-trees ALL COMMAND ${DOXYGEN} "${ml_BINARY_DIR}/doxyfile-ml-decision-trees")
endif()
