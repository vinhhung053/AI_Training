//
// Created by lap13385 on 16/11/2023.
//

#include "Tokenizer.h"

Tokenizer::Tokenizer()= default;

void Tokenizer::read_tokenizer() const {
    freopen("/home/lap13385/Projects/Zalo_AI_Fresher_Training/Week11_12_gpt2_generator_cpp/tokenizer/tokenizer.txt", "r", stdin);
    std::string st;
    int x;
    while (std::cin >> st >> x) {
        std::cout << st << x << std::endl;
        token2id->insert({st, x});
        id2token->insert({x, st});
    }
    fclose(stdin);
}
int Tokenizer::encode(const std::string& word) {
    if (token2id->find(word) == token2id->end())
        return 2;
    return (*(this->token2id))[word];
}
std::string Tokenizer::decode(int id) const {
    return (*id2token)[id];
}