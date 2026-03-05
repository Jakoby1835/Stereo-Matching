cmake_minimum_required(VERSION 3.10)

project(ParallelStereoMatching_WiSe2023_2024 LANGUAGES C CXX)

# checks for OpenCL, OpenMP and either MS-MPI or OpenMPI in default install locations
find_package(OpenCL REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

# take care that our cl-kernel file is visible in visual studio under windows as well
set(STEREO_MATCHING_CL_KERNELS ${CMAKE_SOURCE_DIR}/cl_kernels/stereo_matching_kernels.cl)
source_group("cl_kernels" FILES ${STEREO_MATCHING_CL_KERNELS} )


# creates the target for our StereoMatcher-Executable
add_executable(StereoMatcher StereoMatcher.cpp ${STEREO_MATCHING_CL_KERNELS})
add_executable(BMPCompare
    BMPCompare/bmp_compare.cpp
)






###################################
# HANDLE OPENCL CONFIG OF PROJECT #
###################################
# adds OpenCL include directory to StereoMatcher-Project
target_include_directories(StereoMatcher PUBLIC ${OpenCL_INCLUDE_DIRS})
# links against OpenCL-Libraries in StereoMatcher-Project
target_link_libraries(StereoMatcher PUBLIC ${OpenCL_LIBRARIES})

###################################
# HANDLE OpenMP CONFIG OF PROJECT #
###################################
# adds OpenMP include directory to StereoMatcher-Project
target_include_directories(StereoMatcher PUBLIC ${OpenMP_C_INCLUDE_DIRS})
# sets OpenMP linker flags automatically such that it actually is enabled (like -fopenmp under linux or /openmp on windows)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
# links target against OpenMP
target_link_libraries(StereoMatcher PUBLIC ${OpenMP_C_LIBRARIES})

###################################
#   HANDLE MPI CONFIG OF PROJECT  #
###################################
# adds MPI implementation include directory to StereoMatcher-Project
target_include_directories(StereoMatcher PUBLIC ${MPI_C_INCLUDE_DIRS})
# links MPI library
target_link_libraries(StereoMatcher PUBLIC ${MPI_C_LIBRARIES})



#########################################################################
#   CONFIGURATION FOR CREATING AN INSTALL DIRECTORY IN THE PROJECT ROOT #
#########################################################################
# this will copy the built executable and the folder cl_kernels with all files
# ending on .cl into PROJECT_ROOT/install/bin

# install target config

#if user didnt set install dir, override it and write it to the cache -> Type and description necessary,
#overwrite variable, not just write it to cache
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/install" CACHE STRING "Install path prefix, prepended onto install directories." FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cl_kernels
	    DESTINATION bin
          FILES_MATCHING
          PATTERN "*.cl"
	)

install(TARGETS StereoMatcher
	    DESTINATION bin
	)
install(TARGETS BMPCompare
    DESTINATION bin
)
