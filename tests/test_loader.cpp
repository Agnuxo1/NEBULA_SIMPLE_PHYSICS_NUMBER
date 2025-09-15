#include "data/mnist_loader.hpp"
#include "optical/optical_network.hpp"
#include <fstream>
#include <cassert>
#include <cstdio>

int main() {
    // create temporary train.csv
    const char* train_path = "train_temp.csv";
    {
        std::ofstream f(train_path);
        f << "label";
        for (int i=0;i<784;++i) f << ",pixel" << i;
        f << "\n";
        f << "1";
        for (int i=0;i<784;++i) f << "," << i%255;
        f << "\n";
        f << "0";
        for (int i=0;i<784;++i) f << "," << (255-i%255);
        f << "\n";
    }
    auto data = mnist::load_mnist_train(train_path);
    assert(data.size()==2);
    OpticalNetwork net({784,64,10});
    std::vector<std::pair<std::vector<double>,int>> d;
    for(auto &s: data) d.push_back({s.pixels,s.label});
    net.train(d,1,0.001);
    std::remove(train_path);
    return 0;
}
