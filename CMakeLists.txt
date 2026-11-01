cmake_minimum_required(VERSION 3.24)

set(CMAKE_CXX_STANDARD 23)
# set(CMAKE_CXX_STANDARD_REQUIRED ON) # I did this for cxx26
# set(CMAKE_CXX_EXTENSIONS OFF) # I did this for cxx26
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native) # build for the GPU in this machine
endif()


get_filename_component(ROOT_FOLDER_NAME ${CMAKE_SOURCE_DIR} NAME) # get root folder name
set(TARGET_NAME main)
project(${TARGET_NAME} LANGUAGES C CXX CUDA)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF) # idk doing this make red line go away when including headers in a .cu file

add_executable(${TARGET_NAME})


# get files -----------------------------------
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
  ${CMAKE_SOURCE_DIR}/src/*.cpp
  ${CMAKE_SOURCE_DIR}/src/*.cu
  ${CMAKE_SOURCE_DIR}/third_party/lib/*.cpp
  ${CMAKE_SOURCE_DIR}/third_party/lib/*.c
  ${CMAKE_SOURCE_DIR}/third_party/lib/*.cu
  ${CMAKE_SOURCE_DIR}/third_party/dropIn/*.cpp
  ${CMAKE_SOURCE_DIR}/third_party/dropIn/*.c
  ${CMAKE_SOURCE_DIR}/third_party/lib/*.cu
)

file(GLOB_RECURSE HEADER_FILES CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/include/${ROOT_FOLDER_NAME}/*.h
    ${CMAKE_SOURCE_DIR}/include/${ROOT_FOLDER_NAME}/*.cuh
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

# things that cant be included in the project folder
find_package(OpenGL REQUIRED)
# find_package(CUDAToolkit REQUIRED)
target_link_libraries(${TARGET_NAME}
  PRIVATE
    ${THIRD_PARTY_LIBS}
    OpenGL::GL
    ${CMAKE_DL_LIBS}
    # CUDA::cudart
    # CUDA::cublas
)

# if flags are needed for only .cu files
# target_compile_options(${TARGET_NAME} PRIVATE
#   $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
# )
