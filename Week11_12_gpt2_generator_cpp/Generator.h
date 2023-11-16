//
// Created by lap13385 on 16/11/2023.
//

#ifndef WEEK11_12_GPT2_GENERATOR_CPP_GENERATOR_H
#define WEEK11_12_GPT2_GENERATOR_CPP_GENERATOR_H

#include "Header.h"
#include "Tokenizer.h"

class Generator {
public:
    std::unique_ptr<torch::jit::script::Module> model = nullptr;
    std::unique_ptr<Tokenizer> tokenizer = std::make_unique<Tokenizer>();

    Generator();

    void init();
    std::vector<int> convert_string_to_vec_ids(std::string str);

    int get_max_length_input(std::vector<std::string> &input_vec_string);


    void generator_greedy_search(std::vector<std::string> &input_vec_string, crow::json::rvalue use_kv_cache,
                            crow::json::rvalue max_length_output);
};
#endif //WEEK11_12_GPT2_GENERATOR_CPP_GENERATOR_H
