cmake_minimum_required(VERSION 3.5...3.31)
project(KSR_Building)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
set(CMAKE_BUILD_TYPE Release)

# CGAL 6.1.1
set(CGAL_DIR "$ENV{HOME}/cgal-6.1.1")

# Find packages
find_package(CGAL REQUIRED HINTS ${CGAL_DIR})
find_package(Eigen3 3.1.0 REQUIRED NO_MODULE)
find_package(Boost REQUIRED)

# 1. KSR Building reconstruction
add_executable(ksr_building ksr_building.cpp)
target_link_libraries(ksr_building CGAL::CGAL)
if(TARGET Eigen3::Eigen)
    target_link_libraries(ksr_building Eigen3::Eigen)
else()
    target_include_directories(ksr_building PRIVATE ${EIGEN3_INCLUDE_DIR})
endif()

# 2. RANSAC Detection
add_executable(ransac_detection ransac_detection.cpp)
target_link_libraries(ransac_detection CGAL::CGAL)
if(TARGET Eigen3::Eigen)
    target_link_libraries(ransac_detection Eigen3::Eigen)
else()
    target_include_directories(ransac_detection PRIVATE ${EIGEN3_INCLUDE_DIR})
endif()

# 3. Region Growing Detection
add_executable(region_growing_detection region_growing_detection.cpp)
target_link_libraries(region_growing_detection CGAL::CGAL)
if(TARGET Eigen3::Eigen)
    target_link_libraries(region_growing_detection Eigen3::Eigen)
else()
    target_include_directories(region_growing_detection PRIVATE ${EIGEN3_INCLUDE_DIR})
endif()
