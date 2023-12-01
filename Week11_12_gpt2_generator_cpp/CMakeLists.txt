CMAKE_MINIMUM_REQUIRED(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)
set(Torch_DIR /usr/local/include/libtorch/share/cmake/Torch)

set(PROJECT_RUN gpt2_run)
set(PROJECT_TEST gpt2_test)

enable_testing()

project(${PROJECT_RUN})
project(${PROJECT_TEST})


#add_subdirectory(src)
add_subdirectory(src_test)

