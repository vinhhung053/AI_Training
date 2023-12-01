
set(CMAKE_PREFIX_PATH "/usr/local/include")

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
#add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
add_executable(${PROJECT_RUN} main.cpp
        header.h
        tokenizer.cpp
        tokenizer.h
        generator.cpp
        generator.h)

target_link_libraries(${PROJECT_RUN} PUBLIC Crow::Crow ${TORCH_LIBRARIES})
