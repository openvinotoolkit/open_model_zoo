file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

add_demo(NAME mri_reconstruction_demo
         SOURCES ${SOURCES}
         HEADERS ${HEADERS})
