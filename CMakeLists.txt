cmake_minimum_required(VERSION 2.8)

project(texturemapgen)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -O3")
add_definitions(-DPROJECT_DIR="${PROJECT_SOURCE_DIR}")

include_directories(include)

find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(ZLIB REQUIRED)
include_directories(${OPENGL_INCLUDE_DIRS})

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_library(ASSIMP_LIBRARY assimp HINTS ${PROJECT_SOURCE_DIR}/libs)
#target_link_libraries(test PUBLIC ${PROTOBUF_LIBRARY})

# Boost
find_package(Boost 1.70 COMPONENTS program_options system filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

set(SOURCES
    src/main.cpp
    src/renderer.cpp
    src/cameraloader.cpp
    src/deferred_neural_renderer.cpp
    src/timer.cpp
    src/frame_writer.cpp
    src/keyhandler.cpp
    src/stb_image.cpp
    src/stb_image_write.cpp
    src/glad.c
    src/scene.cpp)


add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(
    ${PROJECT_NAME}
    ${OPENGL_LIBRARIES}
    glfw
    ${ASSIMP_LIBRARY}
    ${Boost_LIBRARIES}
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    ${TORCH_LIBRARIES}
    ${CMAKE_DL_LIBS}
    ZLIB::ZLIB
    )
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)
