
#include "Generator.h"
#include "Header.h"




void handleGeneratorRequest(const crow::request& req, crow::response& res, const crow::json::rvalue& input, crow::json::rvalue &use_kv_cache, crow::json::rvalue &max_length_output) {
    std::vector<std::string> input_vec_string;
    std::cout << "1";
    crow::json::wvalue response_data;
    std::cout << "2";

    std::unique_ptr<Generator> generator = std::make_unique<Generator>();
    std::cout << "3";

    generator->init();
    std::cout << "4";

    std::cout << "cc";
    for(auto data : input) {
        std::string st = static_cast<std::string>(data);
        input_vec_string.push_back(st);
    }
    time_t start_time = time(nullptr);
    if(static_cast<std::string>(use_kv_cache) == "false") {
        generator->generator_greedy_search(input_vec_string, use_kv_cache, max_length_output);
    }
    else {
        generator->generator_greedy_search(input_vec_string, use_kv_cache, max_length_output);
    }
//    response_data["output"] = input_vec_string;

    time_t end_time = time(nullptr);
    response_data["time"] = float(end_time - start_time) /  CLOCKS_PER_SEC;
//    res = response_data;
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