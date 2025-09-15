#include "data/mnist_loader.hpp"
#include "optical/optical_network.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "Usage: generate_submission <model.bin> <test.csv> <submission.csv>\n";
        return 1;
    }
    OpticalNetwork net;
    net.load(argv[1]);
    auto test_data = mnist::load_mnist_test(argv[2]);
    std::ofstream out(argv[3]);
    out << "ImageId,Label\n";
    for (size_t i = 0; i < test_data.size(); ++i) {
        auto probs = net.predict(test_data[i]);
        int pred = std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
        out << (i+1) << "," << pred << "\n";
    }
    std::cout << "Wrote " << test_data.size() << " predictions to " << argv[3] << "\n";
    return 0;
}
