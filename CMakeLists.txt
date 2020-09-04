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

# OpenCV
find_package(OpenCV 3 QUIET COMPONENTS core features2d)
if (OpenCV_FOUND)
  message(STATUS "Found OpenCV 3 headers in: ${OpenCV_INCLUDE_DIRS}")
else()
  message(STATUS "Didn't find OpenCV 3, looking for 2")
  # HINTS needed for macos homebrew (install with `brew install opencv@2`)
  find_package( OpenCV 2 REQUIRED COMPONENTS core features2d
      HINTS /usr/local/opt/opencv@2 )
  message(STATUS "Found OpenCV 2 headers in: ${OpenCV_INCLUDE_DIRS}")
endif()

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
    src/scene.cpp
    src/scenemodel.cpp
    src/object.cpp)


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
    ${OpenCV_LIBS}
    )
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)
