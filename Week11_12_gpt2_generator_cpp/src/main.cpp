
#include "tokenizer.h"
#include "generator.h"
#include "header.h"



void handleGeneratorRequest(const crow::request& req, crow::response& res, const crow::json::rvalue& input, std::unique_ptr<Generator> &generator, crow::json::rvalue &use_kv_cache, crow::json::rvalue &max_length_output) {
    std::vector<std::string> input_vec_string;
    crow::json::wvalue response_data;

    for(const auto& data : input) {
        std::string st = static_cast<std::string>(data);
        input_vec_string.push_back(st);
    }
    time_t start_time = clock();
    if(static_cast<std::string>(use_kv_cache) == "false") {
        generator->generator_greedy_search(input_vec_string, use_kv_cache, max_length_output);
        response_data["output no use kv cache"] = input_vec_string;
    }
    else {
        generator->generator_greedy_search(input_vec_string, use_kv_cache, max_length_output);
        response_data["output use kv cache"] = input_vec_string;
    }


    time_t end_time = clock();
    response_data["time"] = static_cast<float>(end_time - start_time) / CLOCKS_PER_SEC;
    res = response_data;
}

int main() {
    crow::SimpleApp app;
    int a = 2;
    std::unique_ptr<Generator> generator = std::make_unique<Generator>();
    generator->init();

    CROW_ROUTE(app, "/").methods("POST"_method)
            ([&generator](const crow::request& req, crow::response& res){
                crow::json::rvalue data = crow::json::load(req.body);
                const auto& input = data["input"];
                auto use_kv_cache = data["use_kv_cache"];
                auto max_length_output = data["max_length_output"];
                handleGeneratorRequest(req, res, input, generator, use_kv_cache, max_length_output);
                res.end();
            });

    // Chạy ứng dụng trên cổng 8080
    app.port(8080).multithreaded().run();

    return 0;
}