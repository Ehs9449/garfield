cmake_minimum_required(VERSION 3.5...3.31)
project(KSR_Building)

# Set C++ standard (CGAL 6.x requires C++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type for performance
set(CMAKE_BUILD_TYPE Release)

# Point to local CGAL 6.1.1 source
set(CGAL_DIR "$ENV{HOME}/cgal-6.1.1")

# Find packages
find_package(CGAL REQUIRED HINTS ${CGAL_DIR})
find_package(Eigen3 3.1.0 REQUIRED NO_MODULE)
find_package(Boost REQUIRED)

# Create executable
add_executable(ksr_building ksr_building.cpp)

# Link CGAL
target_link_libraries(ksr_building CGAL::CGAL)

# Add Eigen include directory
if(TARGET Eigen3::Eigen)
    target_link_libraries(ksr_building Eigen3::Eigen)
else()
    target_include_directories(ksr_building PRIVATE ${EIGEN3_INCLUDE_DIR})
endif()
