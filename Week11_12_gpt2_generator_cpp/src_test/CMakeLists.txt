include_directories("/usr/local/include")


find_package(Torch REQUIRED)
find_package(GTest REQUIRED)

add_executable(${PROJECT_TEST} test_generator.cpp
        ../src/generator.h
        ../src/generator.cpp
        ../src/tokenizer.h
        ../src/tokenizer.cpp
        ../src/header.h
)

target_link_libraries(${PROJECT_TEST} ${TORCH_LIBRARIES} ${GTEST_BOTH_LIBRARIES} pthread )

add_test(${PROJECT_TEST} ${PROJECT_TEST})
