#pragma once
#include <string>
#include <vector>

namespace mnist {
struct Sample {
    std::vector<double> pixels;
    int label;
};

std::vector<Sample> load_mnist_train(const std::string &path);
std::vector<std::vector<double>> load_mnist_test(const std::string &path);
}
