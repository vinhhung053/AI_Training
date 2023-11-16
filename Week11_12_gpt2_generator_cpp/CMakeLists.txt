cmake_minimum_required(VERSION 3.13)
project(Week11_12_gpt2_generator_cpp)

set(CMAKE_PREFIX_PATH "/home/lap13385/Projects/Zalo_AI_Fresher_Training/Week11_12_gpt2_generator_cpp/library/libtorch")
find_package(Torch REQUIRED)
find_package(Crow REQUIRED)

if (Torch_FOUND)
    message(STATUS "Torch library found")
else()
    message(FATAL_ERROR "Torch library not found.")
endif()

if (Crow_FOUND)
    message(STATUS "Crow library found")
else()
    message(FATAL_ERROR "Crow library not found.")
endif()


set(CMAKE_CXX_STANDARD 17)

add_executable(Week11_12_gpt2_generator_cpp main.cpp
        Header.h
        Tokenizer.cpp
        Tokenizer.h
        Generator.cpp
        Generator.h)

target_link_libraries(Week11_12_gpt2_generator_cpp PUBLIC Crow::Crow "${TORCH_LIBRARIES}")
#target_link_libraries(Week11_12_gpt2_generator_cpp )

