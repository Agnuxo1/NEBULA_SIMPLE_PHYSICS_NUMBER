#include "optical_network.hpp"
#include <cmath>
#include <random>
#include <stdexcept>
#include <algorithm>

static double random_weight() {
    static std::mt19937 gen(42);
    static std::uniform_real_distribution<double> dist(-0.5, 0.5);
    return dist(gen);
}

OpticalLayer::OpticalLayer(size_t in, size_t out)
    : in_features(in), out_features(out),
      amplitude(out * in), phase(out * in) {
    for (size_t i = 0; i < amplitude.size(); ++i) {
        amplitude[i] = std::abs(random_weight());
        phase[i] = random_weight();
    }
}

std::vector<double> OpticalLayer::forward(const std::vector<double> &input) {
    last_input = input;
    std::vector<double> output(out_features, 0.0);
    last_z.assign(out_features, 0.0);
    for (size_t o = 0; o < out_features; ++o) {
        double sum = 0.0;
        for (size_t i = 0; i < in_features; ++i) {
            size_t idx = o * in_features + i;
            double w = amplitude[idx] * std::cos(phase[idx]);
            sum += w * input[i];
        }
        last_z[o] = sum;
        output[o] = std::sin(sum); // optical non-linearity
    }
    return output;
}

std::vector<double> OpticalLayer::backward(const std::vector<double> &grad, double lr) {
    std::vector<double> grad_input(in_features, 0.0);
    for (size_t o = 0; o < out_features; ++o) {
        double dz = grad[o] * std::cos(last_z[o]);
        for (size_t i = 0; i < in_features; ++i) {
            size_t idx = o * in_features + i;
            double w = amplitude[idx] * std::cos(phase[idx]);
            grad_input[i] += w * dz;
            double dW = dz * last_input[i];
            double dA = dW * std::cos(phase[idx]);
            double dP = -dW * amplitude[idx] * std::sin(phase[idx]);
            amplitude[idx] -= lr * dA;
            phase[idx] -= lr * dP;
        }
    }
    return grad_input;
}

void OpticalLayer::save(std::ofstream &out) const {
    out.write(reinterpret_cast<const char*>(&in_features), sizeof(in_features));
    out.write(reinterpret_cast<const char*>(&out_features), sizeof(out_features));
    out.write(reinterpret_cast<const char*>(amplitude.data()), amplitude.size()*sizeof(double));
    out.write(reinterpret_cast<const char*>(phase.data()), phase.size()*sizeof(double));
}

void OpticalLayer::load(std::ifstream &in) {
    in.read(reinterpret_cast<char*>(&in_features), sizeof(in_features));
    in.read(reinterpret_cast<char*>(&out_features), sizeof(out_features));
    amplitude.resize(out_features*in_features);
    phase.resize(out_features*in_features);
    in.read(reinterpret_cast<char*>(amplitude.data()), amplitude.size()*sizeof(double));
    in.read(reinterpret_cast<char*>(phase.data()), phase.size()*sizeof(double));
}

OpticalNetwork::OpticalNetwork(const std::vector<size_t> &layers_sizes) {
    for (size_t i = 1; i < layers_sizes.size(); ++i) {
        layers.emplace_back(layers_sizes[i-1], layers_sizes[i]);
    }
}

std::vector<double> OpticalNetwork::predict(const std::vector<double> &input) {
    std::vector<double> act = input;
    for (size_t i = 0; i < layers.size(); ++i) {
        act = layers[i].forward(act);
    }
    return act;
}

void OpticalNetwork::train(const std::vector<std::pair<std::vector<double>, int>> &data,
                           size_t epochs, double lr) {
    for (size_t e = 0; e < epochs; ++e) {
        for (const auto &sample : data) {
            // forward
            std::vector<double> act = sample.first;
            std::vector<std::vector<double>> activations;
            activations.push_back(act);
            for (auto &layer : layers) {
                act = layer.forward(act);
                activations.push_back(act);
            }
            // softmax
            std::vector<double> probs(act.size());
            double maxv = *std::max_element(act.begin(), act.end());
            double sum = 0.0;
            for (size_t i = 0; i < act.size(); ++i) {
                probs[i] = std::exp(act[i]-maxv);
                sum += probs[i];
            }
            for (double &p : probs) p /= sum;
            // gradient of loss
            std::vector<double> grad(probs.size());
            for (size_t i = 0; i < probs.size(); ++i) {
                grad[i] = probs[i];
            }
            grad[sample.second] -= 1.0;
            // backward
            for (int i = (int)layers.size()-1; i >= 0; --i) {
                grad = layers[i].backward(grad, lr);
            }
        }
    }
}

void OpticalNetwork::save(const std::string &path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot save model");
    size_t L = layers.size();
    out.write(reinterpret_cast<const char*>(&L), sizeof(L));
    for (const auto &layer : layers) {
        layer.save(out);
    }
}

void OpticalNetwork::load(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot load model");
    size_t L; in.read(reinterpret_cast<char*>(&L), sizeof(L));
    layers.clear();
    layers.reserve(L);
    for (size_t i = 0; i < L; ++i) {
        OpticalLayer layer(1,1);
        layer.load(in);
        layers.push_back(layer);
    }
}
