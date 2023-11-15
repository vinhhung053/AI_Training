//
// Created by lap13385 on 16/11/2023.
//

#ifndef WEEK11_12_GPT2_GENERATOR_CPP_TOKENIZER_H
#define WEEK11_12_GPT2_GENERATOR_CPP_TOKENIZER_H

#include "Header.h"

class Tokenizer{
public:
    std::unique_ptr<std::map<std::string, int>> token2id;
    std::unique_ptr<std::map<int, std::string>> id2token;

    Tokenizer();

    void read_tokenizer();

    int encode(std::string word);

    std::string decode(int id);
};

#endif //WEEK11_12_GPT2_GENERATOR_CPP_TOKENIZER_H
