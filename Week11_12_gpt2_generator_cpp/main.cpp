#include <iostream>
#include "Header.h"




void handleGeneratorRequest(const crow::request& req, crow::response& res, crow::json::rvalue input, crow::json::rvalue use_kv_cache, crow::json::rvalue max_length_output) {
    std::vector<std::string> vec_input;
    int max_length_input = 0;
    for(auto data : input) {
        std::string st = static_cast<std::string>(data);
        vec_input.push_back(st);

    crow::json::wvalue response_data;
    response_data["output"] = "hung";

    res = response_data;

//        max_length_input = std::max(max_length_input, static_cast<int>((generator.encode(data)).size()));
    }


////    std::cout  << "--------------- generator no use kv cache-----------------------------" << std::endl;
//    res.write("--------------- generator no use kv cache-----------------------------\n");
//    bool use_kv_cache = false;
//    auto time_before_loop_begins = time(nullptr);
//    std::string output = generator.generator_greedy_search(inputs,use_kv_cache, maximum_length_input);
//    res.write(output + '\n');
//    auto time_after_loop_ends = time(nullptr);
//    auto time_diff = time_after_loop_ends - time_before_loop_begins;
////    std::cout << "Time taken to run generator no use kv cache = " << time_diff << " seconds." << std::endl;
//
//    //std::cout  << "--------------- generator use kv cache-----------------------------" << std::endl;
//    res.write("--------------- generator use use kv cache-----------------------------\n");
//    time_before_loop_begins = time(nullptr);
//    use_kv_cache = true;
//    std::string output2 = generator.generator_greedy_search(inputs, use_kv_cache, maximum_length_input);
//    res.write(output2 + '\n');
//    time_after_loop_ends = time(nullptr);
//    time_diff = time_after_loop_ends - time_before_loop_begins;
//    res.end();
////    std::cout << "Time taken to run generator use kv cache = " << time_diff << " seconds." << std::endl;
//
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