#include "data/mnist_loader.hpp"
#include "optical/optical_network.hpp"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <sstream>
#include <algorithm>

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: infer <model.bin> <sample_csv_line>\n";
        return 1;
    }
    OpticalNetwork net;
    net.load(argv[1]);
    std::stringstream ss(argv[2]);
    std::vector<double> pixels;
    pixels.reserve(784);
    std::string cell;
    for (int i = 0; i < 784; ++i) {
        if (!std::getline(ss, cell, ',')) cell = "0";
        pixels.push_back(std::stod(cell) / 255.0);
    }
    auto probs = net.predict(pixels);
    int pred = std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
    std::cout << pred << "\n";
    return 0;
}
