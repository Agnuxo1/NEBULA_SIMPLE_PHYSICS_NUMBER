#include "mnist_loader.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

namespace mnist {
static std::vector<double> parse_pixels(std::stringstream &ss) {
    std::vector<double> pixels;
    pixels.reserve(784);
    std::string cell;
    for (int i = 0; i < 784; ++i) {
        if (!std::getline(ss, cell, ',')) {
            cell = "0";
        }
        pixels.push_back(std::stod(cell) / 255.0);
    }
    return pixels;
}

std::vector<Sample> load_mnist_train(const std::string &path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    std::vector<Sample> samples;
    std::string line;
    // skip header
    std::getline(file, line);
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::getline(ss, cell, ',');
        int label = std::stoi(cell);
        auto pixels = parse_pixels(ss);
        samples.push_back({std::move(pixels), label});
    }
    return samples;
}

std::vector<std::vector<double>> load_mnist_test(const std::string &path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    std::vector<std::vector<double>> samples;
    std::string line;
    // skip header
    std::getline(file, line);
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        auto pixels = parse_pixels(ss);
        samples.push_back(std::move(pixels));
    }
    return samples;
}
}
