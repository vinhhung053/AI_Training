//
// Created by lap13385 on 16/11/2023.
//

#ifndef WEEK11_12_GPT2_GENERATOR_CPP_TOKENIZER_H
#define WEEK11_12_GPT2_GENERATOR_CPP_TOKENIZER_H

#include "header.h"

class Tokenizer{
public:
    std::unique_ptr<std::map<std::string, int>> token2id = std::make_unique<std::map<std::string, int>>();
    std::unique_ptr<std::map<int, std::string>> id2token = std::make_unique<std::map<int, std::string>>();

    Tokenizer();

    void read_tokenizer() const;

    int encode(const std::string& word);

    std::string decode(int id) const;
};

#endif //WEEK11_12_GPT2_GENERATOR_CPP_TOKENIZER_H
