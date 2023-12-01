#include "../src/generator.h"
#include "gtest/gtest.h"
#include<iostream>

TEST(GeneratorTest, TestGenerator)  {
    Generator generator;
//    std::vector<std::string> inputs = {"my", "when i"};
//    std::cout << generator.get_max_length_input(inputs);
    EXPECT_EQ(generator.add(1,2) , 3);
    EXPECT_EQ(generator.add(2,2) , 3);
}

int main() {
    ::testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}