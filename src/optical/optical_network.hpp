#pragma once
#include <vector>
#include <string>
#include <fstream>

class OpticalLayer {
public:
    OpticalLayer(size_t in, size_t out);
    std::vector<double> forward(const std::vector<double> &input);
    std::vector<double> backward(const std::vector<double> &grad, double lr);
    void save(std::ofstream &out) const;
    void load(std::ifstream &in);
private:
    size_t in_features;
    size_t out_features;
    std::vector<double> amplitude; // out * in
    std::vector<double> phase;     // out * in
    std::vector<double> last_input;
    std::vector<double> last_z;
};

class OpticalNetwork {
public:
    OpticalNetwork() = default;
    OpticalNetwork(const std::vector<size_t> &layers);
    std::vector<double> predict(const std::vector<double> &input);
    void train(const std::vector<std::pair<std::vector<double>, int>> &data,
               size_t epochs, double lr);
    void save(const std::string &path) const;
    void load(const std::string &path);
private:
    std::vector<OpticalLayer> layers;
};
