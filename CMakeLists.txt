cmake_minimum_required(VERSION 3.23)

set(CMAKE_CXX_STANDARD 23)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)

get_filename_component(ROOT_FOLDER_NAME ${CMAKE_SOURCE_DIR} NAME) # get root folder name
set(TARGET_NAME main)
project(${TARGET_NAME})

add_executable(${TARGET_NAME})


# get files -----------------------------------
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
  ${CMAKE_SOURCE_DIR}/src/*.cpp
  ${CMAKE_SOURCE_DIR}/third_party/lib/*.cpp
  ${CMAKE_SOURCE_DIR}/third_party/lib/*.c
  ${CMAKE_SOURCE_DIR}/third_party/dropIn/*.cpp
  ${CMAKE_SOURCE_DIR}/third_party/dropIn/*.c
)

file(GLOB_RECURSE HEADER_FILES CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/include/${ROOT_FOLDER_NAME}/*.h
)



# set source and header files -----------------------------------
target_sources(${TARGET_NAME}
  PRIVATE
    ${SRC_FILES}

  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS
      #include/${ROOT_FOLDER_NAME}
      ${CMAKE_SOURCE_DIR}/include/
    FILES
      ${HEADER_FILES}
)





# libraries -----------------------------------

# for .h only library or .h interfaces of libraries
target_include_directories(${TARGET_NAME}
	PRIVATE
    third_party/include # for libraries with seperable .h and .cpp files
    third_party/dropIn # for libraries with inseperable .h and .cpp files
)
target_compile_definitions(${TARGET_NAME}
  PRIVATE
    GLFW_INCLUDE_NONE # stop glfw3.h from including the system GL header (glad provides it)
    GLM_ENABLE_EXPERIMENTAL # allow experimental glm features
    INDEX_TRACKER # enable this to use the morton code solvers
)

# .a .so .c implementations who have a .h in third_party/include
target_link_directories(${TARGET_NAME}
  PRIVATE
    third_party/lib
)

# find .a and .so files
file(GLOB_RECURSE THIRD_PARTY_LIBS CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/third_party/lib/*.a
    ${CMAKE_SOURCE_DIR}/third_party/lib/*.so
)

# thinngs that cant be included in the project folder
find_package(OpenGL REQUIRED)
target_link_libraries(${TARGET_NAME}
  PRIVATE
    ${THIRD_PARTY_LIBS}
    OpenGL::GL
    ${CMAKE_DL_LIBS}
)
