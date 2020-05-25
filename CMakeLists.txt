cmake_minimum_required(VERSION 2.8)

project(texturemapgen)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -O3")
add_definitions(-DPROJECT_DIR="${PROJECT_SOURCE_DIR}")

include_directories(include)

find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)
include_directories(${OPENGL_INCLUDE_DIRS})

find_library(ASSIMP_LIBRARY assimp HINTS ${PROJECT_SOURCE_DIR}/libs)
#target_link_libraries(test PUBLIC ${PROTOBUF_LIBRARY})

# Boost
find_package(Boost 1.40 COMPONENTS program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

set(SOURCES
    src/main.cpp
    src/stb_image.cpp
    src/glad.c)


add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${OPENGL_LIBRARIES} glfw OpenMeshCore ${ASSIMP_LIBRARY} ${Boost_LIBRARIES})
