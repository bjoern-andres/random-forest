project(andres-ml-random-forest)
include_directories(include)
file(GLOB headers_insegt include/andres/* include/andres/ml/*)

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
