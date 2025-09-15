#include "data/mnist_loader.hpp"
#include "optical/optical_network.hpp"
#include <iostream>

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: train <train.csv> <model.bin> [epochs] [lr]\n";
        return 1;
    }
    auto train_data_raw = mnist::load_mnist_train(argv[1]);
    std::vector<std::pair<std::vector<double>, int>> data;
    data.reserve(train_data_raw.size());
    for (auto &s : train_data_raw) {
        data.push_back({std::move(s.pixels), s.label});
    }
    size_t epochs = argc > 3 ? std::stoul(argv[3]) : 1;
    double lr = argc > 4 ? std::stod(argv[4]) : 0.01;
    OpticalNetwork net({784, 64, 10});
    net.train(data, epochs, lr);
    net.save(argv[2]);
    std::cout << "Model saved to " << argv[2] << "\n";
    return 0;
}
