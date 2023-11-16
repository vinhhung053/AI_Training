
#include "Tokenizer.h"
#include "Header.h"
#include <sstream>
std::vector<std::string> generator_greedy_search(std::vector<std::string> input_vec_string, crow::json::rvalue use_kv_cache, crow::json::rvalue max_length_output) {
    std::unique_ptr<Tokenizer> tokenizer = std::make_unique<Tokenizer>();
    tokenizer->read_tokenizer();
    const std::string model_path = "/home/lap13385/Projects/Zalo_AI_Fresher_Training/Week11_12_gpt2_generator_cpp/model_pre_train/m.pt";
    torch::jit::script::Module model = torch::jit::load(model_path);
    std::vector<int> vec_input_tokens;
    int max_length_input = 4;
    for (int i = 0; i < input_vec_string.size(); i++) {
        std::string string_input = input_vec_string[i];
        std::istringstream iss (string_input);
        std::string word;
        while(iss >> string_input)
        {
            std::cout << tokenizer->encode(string_input) << std::endl;
            vec_input_tokens.push_back(tokenizer->encode(string_input));
        }
    }
    torch::Tensor kv_cache = torch::empty({1});
    std::cout << kv_cache << std::endl;
    torch::Tensor myTensor = torch::from_blob(vec_input_tokens.data(), {input_vec_string.size(), max_length_input}, torch::kInt);
    auto outputs = model.forward({myTensor, kv_cache}).toTuple();
    torch::Tensor out1 = outputs->elements()[0].toTensor();
    torch::Tensor out2 = outputs->elements()[1].toTensor();
    std::cout << out1 << " " << out2 << std::endl;

}

void handleGeneratorRequest(const crow::request& req, crow::response& res, crow::json::rvalue input, crow::json::rvalue use_kv_cache, crow::json::rvalue max_length_output) {
    std::vector<std::string> input_vec_string;
    crow::json::wvalue response_data;
    int max_length_input = 0;
    for(auto data : input) {
        std::string st = static_cast<std::string>(data);
        input_vec_string.push_back(st);
//        max_length_input = std::max(max_length_input, static_cast<int>((generator.encode(data)).size()));
    }
    time_t start_time = time(nullptr);
    if(static_cast<std::string>(use_kv_cache) == "false") {
        std::vector<std::string> output = generator_greedy_search(input_vec_string, use_kv_cache, max_length_output);
    }
    else {
        std::vector<std::string> output2 = generator_greedy_search(input_vec_string, use_kv_cache, max_length_output);
    }

    time_t end_time = time(nullptr);
    response_data["time"] = float(end_time - start_time) /  CLOCKS_PER_SEC;
}

int main() {
    crow::SimpleApp app;

    CROW_ROUTE(app, "/").methods("POST"_method)
            ([](const crow::request& req, crow::response& res){
                crow::json::rvalue data = crow::json::load(req.body);
                auto input = data["input"];
                auto use_kv_cache = data["use_kv_cache"];
                auto max_length_output = data["max_length_output"];

                handleGeneratorRequest(req, res, input, use_kv_cache, max_length_output);

                res.end();
            });

    // Chạy ứng dụng trên cổng 8080
    app.port(8080).multithreaded().run();

    return 0;
}