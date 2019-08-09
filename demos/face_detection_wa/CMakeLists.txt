cmake_minimum_required(VERSION 2.8)
project( three )

add_executable( three three.cpp )

find_package(InferenceEngine)

include_directories(${InferenceEngine_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} ${InferenceEngine_LIBRARIES} dl)

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

target_link_libraries( three ${OpenCV_LIBS} )


