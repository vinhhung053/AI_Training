cmake_minimum_required(VERSION 3.13)
project(Week11_12_gpt2_generator_cpp)

find_package(Crow)

set(CMAKE_CXX_STANDARD 17)

add_executable(Week11_12_gpt2_generator_cpp main.cpp
        Header.h)
#        Tokenizer.cpp
#        Tokenizer.h
#        Generator.cpp
#        Generator.h)

target_link_libraries(Week11_12_gpt2_generator_cpp PUBLIC Crow::Crow)
