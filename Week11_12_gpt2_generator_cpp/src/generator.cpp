//
// Created by lap13385 on 16/11/2023.
//

#include "generator.h"

Generator::Generator() = default;

void Generator::init() {
    tokenizer->read_tokenizer();
    std::string model_path = "/home/lap13385/Projects/Zalo_AI_Fresher_Training/Week11_12_gpt2_generator_cpp/model_pre_train/m1.pt";
    try {
        model = std::make_unique<torch::jit::script::Module>(torch::jit::load(model_path));
    } catch (const std::exception& e) {
        // Print any exceptions that occurred during model loading
        std::cerr << "Error loading model: " << std::endl;
    }
}

std::vector<int> Generator::convert_string_to_vec_ids(std::string str) {
    std::istringstream iss(str);
    std::string word;
    std::vector<int> vec_ids;
    while (iss >> word) {
        vec_ids.push_back(tokenizer->encode(word));
    }
    return vec_ids;
}

int Generator::get_max_length_input(std::vector<std::string> &input_vec_string){
    int max_length_input = 0;
    for (int i = 0; i < input_vec_string.size(); i++) {
        max_length_input = std::max(max_length_input, static_cast<int>((Generator::convert_string_to_vec_ids(input_vec_string[i])).size()));
    }
    return max_length_input;
}




void Generator::generator_greedy_search(std::vector<std::string> &input_vec_string, crow::json::rvalue use_kv_cache, crow::json::rvalue max_length_output) {
    std::vector<int> vec_input_tokens;
    std::cout << input_vec_string[0] << std::endl;
    // Chinh cac string ve cung so luong id;
    int max_length_input = Generator::get_max_length_input(input_vec_string);
    for (const auto & i : input_vec_string) {
        std::vector<int> vec_ids = Generator::convert_string_to_vec_ids(i);
        std::cout << vec_ids << std::endl;
        while(vec_ids.size() < max_length_input) {
            vec_ids.emplace(vec_ids.begin(),2);
        }
        vec_input_tokens.insert(vec_input_tokens.end(), vec_ids.begin(),vec_ids.end());
    }
    torch::Tensor tensor_input = torch::from_blob(vec_input_tokens.data(), {static_cast<long>(input_vec_string.size()), max_length_input}, torch::kInt);
    torch::Tensor kv_cache = torch::empty({1});
    std::cout << tensor_input << std::endl;
    std::cout << kv_cache << std::endl;

    for(int loop = 1; loop <= static_cast<long> (max_length_output); loop++) {
        auto outputs = model->forward({tensor_input, kv_cache}).toTuple();
        torch::Tensor prob = outputs->elements()[0].toTensor();
        torch::Tensor kv_cache_new = outputs->elements()[1].toTensor();
        torch::Tensor max_ids = torch::argmax(prob.slice(2,4), -1)+ 4;
        torch::Tensor max_ids_column_last = max_ids.index({torch::indexing::Slice(), -1});
        max_ids_column_last = max_ids_column_last.reshape({-1,1});
        tensor_input = torch::cat({tensor_input, max_ids_column_last}, /*dim=*/1);
        for (int index = 0; index < input_vec_string.size(); index++)
            input_vec_string[index] += " " + tokenizer->decode(max_ids_column_last[index][0].item<int>());
        if (static_cast<std::string>(use_kv_cache) == "true")
            kv_cache = kv_cache_new;
        if(tensor_input.size(1) >= 30) {
            tensor_input = tensor_input.slice(-1, 16);
            kv_cache = torch::empty({1});
        }
    }

}
int Generator::add(int a,int b)
{
    return a + b;
}


