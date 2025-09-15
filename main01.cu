/*
 * NEBULA PHYSICS EMERGENT - RSNA Intracranial Aneurysm Detection Edition

 * Author: Francisco Angulo de Lafuente (Model Made in 2023)

 * Scientific Foundation:
 * - Mach-Zehnder interferometer networks (Shen et al. 2017)
 * - Optical matrix multiplication (Feldmann et al. 2019)
 * - Photonic neural computing (Lin et al. 2018)
 * 
 * Target: RSNA Intracranial Aneurysm Detection Competition
 * - Train on medical CT scan MIP images (4 views: front, back, left, right)
 * - Predict aneurysm presence in intracranial arteries
 * - Output: JSON predictions for 14 arteries + global detection
 * 
 * Architecture: 784→512→256→128→2 optical neurons (binary classification)
 * Learning: Hebbian plasticity + physical clustering
 * Multi-expert: 4 specialized networks for different viewing angles
 */



#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <memory>
#include <chrono>
#include <random>
#include <algorithm>
#include <iomanip>
#include <numeric>
#include <cmath>
#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

// ============================================================================ 
// PHYSICAL CONSTANTS & OPTICAL STRUCTURES
// ============================================================================ 

namespace PhysicalConstants {
    constexpr float SPEED_OF_LIGHT = 299792458.0f;
    constexpr float PLANCK_CONSTANT = 6.62607015e-34f;
    constexpr float BOLTZMANN_CONSTANT = 1.380649e-23f;
    constexpr float PI = 3.14159265359f;
    constexpr float TWO_PI = 2.0f * PI;
}

struct Complex {
    float real, imag;
    
    __host__ __device__ Complex(float r = 0.0f, float i = 0.0f) : real(r), imag(i) {}
    
    __host__ __device__ Complex operator+(const Complex& other) const {
        return Complex(real + other.real, imag + other.imag);
    }
    
    __host__ __device__ Complex operator*(const Complex& other) const {
        return Complex(real * other.real - imag * other.imag,
                      real * other.imag + imag * other.real);
    }
    
    __host__ __device__ Complex operator*(float scalar) const {
        return Complex(real * scalar, imag * scalar);
    }
    
    __host__ __device__ float magnitude() const {
        return sqrtf(real * real + imag * imag);
    }
    
    __host__ __device__ float phase() const {
        return atan2f(imag, real);
    }
};

// Enhanced Mach-Zehnder Interferometer with true interference
struct MachZehnderUnit {
    float phase_upper;      
    float phase_lower;       
    float coupling_ratio;   
    float loss_factor;      
    
    __host__ __device__ Complex transfer(const Complex& input, float wavelength) const {
        float theta = acosf(sqrtf(coupling_ratio));
        
        // True interferometric operation
        Complex upper_path = Complex(cosf(phase_upper), sinf(phase_upper)) * cosf(theta);
        Complex lower_path = Complex(cosf(phase_lower), sinf(phase_lower)) * sinf(theta);
        
        // Interference between paths
        Complex output = (upper_path + lower_path) * input;
        
        // Apply realistic optical loss
        return output * sqrtf(loss_factor);
    }
    
    __host__ __device__ void updatePhases(float delta_upper, float delta_lower) {
        phase_upper = fmodf(phase_upper + delta_upper + PhysicalConstants::TWO_PI, 
                           PhysicalConstants::TWO_PI);
        phase_lower = fmodf(phase_lower + delta_lower + PhysicalConstants::TWO_PI, 
                           PhysicalConstants::TWO_PI);
    }
};

// Improved Optical Neuron with better dynamics
struct OpticalNeuron {
    float3 position;           
    float3 velocity;           
    float mass;               
    
    // Optical properties  
    float wavelength;         
    float coherence_length;   
    Complex field_amplitude;  
    float polarization_angle; 
    
    // Neural properties
    float activation;         
    float threshold;          
    float potential;          
    float temperature;        
    
    // Learning properties
    float plasticity;         
    float memory_trace;       
    int cluster_id;           
    float age;               
    
    // Improved activation with momentum and better scaling
    __host__ __device__ void updateActivation(float input_intensity, float dt) {
        // Proper scaling for optical intensity
        float input_signal = input_intensity * 2.0f;
        
        // Momentum-based dynamics for stability
        float momentum = 0.9f;
        float leak_rate = 0.1f;
        
        // Update potential with momentum
        potential = momentum * potential * expf(-leak_rate * dt) + 
                   (1.0f - momentum) * input_signal;
        
        // Temperature-dependent activation
        float beta = 1.0f / (temperature + 1e-6f);
        
        // Use centered tanh for better gradient flow (range [-1,1])
        activation = tanhf(beta * (potential - threshold));
        
        // Update optical field (use non-negative magnitude proxy)
        field_amplitude = Complex(sqrtf(fmaxf(0.0f, activation)), 0.0f);
        
        // Update memory trace with decay
        memory_trace *= expf(-dt / 0.02f);
        if (activation > 0.5f) {
            memory_trace = fmaxf(memory_trace, activation);
        }
        
        // Age-dependent plasticity
        age += dt;
        plasticity = 0.5f + 0.5f * expf(-age / 1000.0f);
    }
};

// Improved Optical Synapse
struct OpticalSynapse {
    int pre_neuron_id;
    int post_neuron_id;
    float weight;              
    float delay;               
    MachZehnderUnit mz_unit;   
    
    // Learning traces
    float pre_trace;           
    float post_trace;          
    float eligibility_trace;   
    
    __host__ __device__ Complex propagateSignal(const Complex& input_signal, 
                                               float wavelength, float dt) {
        Complex weighted_signal = mz_unit.transfer(input_signal, wavelength);
        return weighted_signal * weight;
    }
};

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// Helper to catch kernel launch and device-side errors early
inline void cudaCheckLastKernel(const char* kernel_name, const char* file, int line) {
    cudaError_t err_async = cudaGetLastError();
    if (err_async != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error after %s at %s:%d -> %s\n",
                kernel_name, file, line, cudaGetErrorString(err_async));
        exit(1);
    }
    err_async = cudaDeviceSynchronize();
    if (err_async != cudaSuccess) {
        fprintf(stderr, "CUDA runtime error during sync after %s at %s:%d -> %s\n",
                kernel_name, file, line, cudaGetErrorString(err_async));
        exit(1);
    }
}
#define CUDA_LAUNCH_CHECK(KERNEL_NAME) cudaCheckLastKernel(KERNEL_NAME, __FILE__, __LINE__)

// ============================================================================ 
// IMAGE UTILS (CPU): robust PGM load and image formalization to 28x28
// ============================================================================ 

static bool load_pgm_as_gray(const std::string& path, std::vector<float>& out, int& w, int& h) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        printf("LOAD_PGM_FAIL: fopen() returned null. Cannot open file. Path: '%s'\n", path.c_str());
        return false;
    }

    char magic[3] = {0};
    if (std::fread(magic, 1, 2, f) != 2) {
        printf("LOAD_PGM_FAIL: fread() for magic number failed. File too small? Path: '%s'\n", path.c_str());
        std::fclose(f);
        return false;
    }

    if (magic[0] != 'P' || (magic[1] != '5' && magic[1] != '2')) {
        printf("LOAD_PGM_FAIL: Incorrect magic number. Expected P2 or P5, got '%c%c'. Path: '%s'\n", magic[0], magic[1], path.c_str());
        std::fclose(f);
        return false;
    }

    auto skip_ws_comments = [&](){
        int c = std::fgetc(f);
        while (c == '#') { while (c != '\n' && c != EOF) c = std::fgetc(f); c = std::fgetc(f); }
        while (c==' '||c=='\t'||c=='\r'||c=='\n') c = std::fgetc(f);
        std::ungetc(c, f);
    };

    skip_ws_comments();
    int maxval = 0;
    if (std::fscanf(f, "%d %d", &w, &h) != 2) {
        printf("LOAD_PGM_FAIL: fscanf() for dimensions (width, height) failed. Path: '%s'\n", path.c_str());
        std::fclose(f);
        return false;
    }

    skip_ws_comments();
    if (std::fscanf(f, "%d", &maxval) != 1) {
        printf("LOAD_PGM_FAIL: fscanf() for maxval failed. Path: '%s'\n", path.c_str());
        std::fclose(f);
        return false;
    }

    if (maxval <= 0) {
        printf("LOAD_PGM_FAIL: maxval is <= 0. Path: '%s'\n", path.c_str());
        std::fclose(f);
        return false;
    }

    std::fgetc(f);
    out.resize((size_t)w * (size_t)h);

    if (magic[1] == '5') {
        if (maxval < 256) {
            std::vector<unsigned char> buf((size_t)w * (size_t)h);
            size_t want = buf.size();
            size_t got  = std::fread(buf.data(), 1, want, f);
            if (got < want) {
                printf("LOAD_PGM_FAIL: fread() for pixel data failed. Expected %zu bytes, got %zu. Path: '%s'\n", want, got, path.c_str());
                std::fclose(f);
                return false;
            }
            for (size_t i = 0; i < want; ++i) out[i] = buf[i] / (float)maxval;
        } else {
            std::vector<unsigned char> buf((size_t)w * (size_t)h * 2);
            size_t want = buf.size();
            size_t got  = std::fread(buf.data(), 1, want, f);
            if (got < want) { 
                printf("LOAD_PGM_FAIL: 16-bit fread() for pixel data failed. Expected %zu bytes, got %zu. Path: '%s'\n", want, got, path.c_str());
                std::fclose(f); 
                return false; 
            }
            for (size_t i = 0, j = 0; i < (size_t)w*(size_t)h; ++i, j+=2) {
                unsigned short v = (unsigned short)buf[j] << 8 | (unsigned short)buf[j+1];
                out[i] = v / (float)maxval;
            }
        }
    } else {
        for (int i = 0; i < w*h; ++i) {
            int val=0; 
            if (std::fscanf(f, "%d", &val) != 1) {
                printf("LOAD_PGM_FAIL: ASCII fscanf() for pixel data failed at pixel %d. Path: '%s'\n", i, path.c_str());
                std::fclose(f); 
                return false; 
            }
            out[i] = val / (float)maxval;
        }
    }

    std::fclose(f);
    return true;
}

static std::string trim_and_normalize_path(std::string p) {
    auto is_space = [](char c){ return c==' ' || c=='\t' || c=='\r' || c=='\n'; };
    while (!p.empty() && is_space(p.front())) p.erase(p.begin());
    while (!p.empty() && is_space(p.back())) p.pop_back();
    if (!p.empty() && (p.front()=='"' || p.front()=='\'')) p.erase(p.begin());
    if (!p.empty() && (p.back()=='"' || p.back()=='\'')) p.pop_back();
#if defined(_WIN32)
    std::replace(p.begin(), p.end(), '/', '\\');
#endif
    return p;
}

static std::vector<float> resize_nn_to_28x28(const std::vector<float>& img, int w, int h) {
    const int TW = 28, TH = 28;
    std::vector<float> out(TW * TH, 0.0f);
    for (int y = 0; y < TH; ++y) {
        int sy = (int)std::round((y + 0.5f) * h / (float)TH - 0.5f);
        if (sy < 0) sy = 0; if (sy >= h) sy = h-1;
        for (int x = 0; x < TW; ++x) {
            int sx = (int)std::round((x + 0.5f) * w / (float)TW - 0.5f);
            if (sx < 0) sx = 0; if (sx >= w) sx = w-1;
            out[y*TW + x] = img[sy * w + sx];
        }
    }
    return out;
}

static std::vector<float> resize_bl_to_28x28(const std::vector<float>& img, int w, int h) {
    const int TW = 28, TH = 28;
    std::vector<float> out(TW*TH, 0.0f);
    for (int y=0; y<TH; ++y) {
        float gy = ((y + 0.5f) * h / (float)TH - 0.5f);
        int y0 = (int)floorf(gy); int y1 = y0 + 1; float wy = gy - y0;
        if (y0 < 0) { y0 = 0; y1 = 0; wy = 0; } if (y1 >= h) { y1 = h-1; y0 = y1; wy = 0; }
        for (int x=0; x<TW; ++x) {
            float gx = ((x + 0.5f) * w / (float)TW - 0.5f);
            int x0 = (int)floorf(gx); int x1 = x0 + 1; float wx = gx - x0;
            if (x0 < 0) { x0=0; x1=0; wx=0; } if (x1 >= w) { x1=w-1; x0=x1; wx=0; }
            float v00 = img[y0*w + x0];
            float v01 = img[y0*w + x1];
            float v10 = img[y1*w + x0];
            float v11 = img[y1*w + x1];
            float v0 = v00*(1-wx) + v01*wx;
            float v1 = v10*(1-wx) + v11*wx;
            out[y*TW + x] = v0*(1-wy) + v1*wy;
        }
    }
    return out;
}

static std::vector<float> formalize_and_resize_to_28x28(const std::vector<float>& img, int w, int h) {
    if (img.empty() || w <= 1 || h <= 1) return std::vector<float>(28*28, 0.0f);
    // Clipping + min-max normalize
    float mn = img[0], mx = img[0];
    for (size_t i=1;i<img.size();++i){ float v=img[i]; if (!std::isfinite(v)) continue; if (v<mn) mn=v; if (v>mx) mx=v; }
    float denom = (mx - mn); if (denom <= 1e-8f) denom = 1.0f;
    std::vector<float> tmp(img.size());
    for (size_t i=0;i<img.size();++i){ float v=img[i]; if (!std::isfinite(v)) v=0.0f; v=(v-mn)/denom; if (v<0) v=0; if (v>1) v=1; tmp[i]=v; }
    return resize_bl_to_28x28(tmp, w, h);
}

// ============================================================================
// DATA LOADER (RSNA Challenge)
// ============================================================================

// Structure to hold info from the RSNA train.csv
struct RSNA_Sample {
    std::string series_uid;
    float label; // 0.0 or 1.0 for binary classification
};

class RSNAAneurysmLoader {
private:
    std::vector<RSNA_Sample> train_samples;
    std::string data_root_dir;

public:
    RSNAAneurysmLoader(const std::string& root_dir = ".") : data_root_dir(root_dir) {}

    bool loadTrainData(const std::string& csv_filepath) {
        std::ifstream file(csv_filepath);
        if (!file.is_open()) {
            std::cerr << "Cannot open train CSV file: " << csv_filepath << std::endl;
            return false;
        }

        std::string line;
        std::getline(file, line); // Skip header

        size_t max_samples = SIZE_MAX;
        if (const char* env_ts = std::getenv("MAX_TRAIN_SAMPLES")) {
            try { max_samples = std::stoul(env_ts); } catch (...) { max_samples = SIZE_MAX; }
        }

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            RSNA_Sample sample;

            // 1. SeriesInstanceUID
            std::getline(ss, sample.series_uid, ',');

            // Skip intermediate columns until we get to the last one
            for (int i = 0; i < 17; ++i) {
                if (!std::getline(ss, cell, ',')) break;
            }
            
            // 18. Aneurysm Present (the label)
            try {
                sample.label = std::stof(cell);
            } catch (...) {
                // Handle cases where conversion might fail
                continue; 
            }

            train_samples.push_back(sample);
            if (train_samples.size() >= max_samples) break;
        }

        file.close();
        std::cout << "Loaded " << train_samples.size() << " training sample records from " << csv_filepath << std::endl;
        return !train_samples.empty();
    }

    // Method to get a pre-processed image and its label for a given index
    bool getTrainSample(size_t index, std::vector<float>& out_image_28x28, float& out_label) {
        if (index >= train_samples.size()) return false;

        const auto& sample = train_samples[index];
        out_label = sample.label;

        // Construct the path to the image file (assuming PGM in mips folder)
        std::string image_path_str = data_root_dir + "/mips/" + sample.series_uid + ".pgm";
        
        // Check if file exists before trying to load
        if (!std::filesystem::exists(image_path_str)) {
             // Fallback to trying .png if .pgm is not found
            image_path_str = data_root_dir + "/mips/" + sample.series_uid + ".png";
            if (!std::filesystem::exists(image_path_str)) {
                // std::cerr << "Warning: Image file not found for UID " << sample.series_uid << std::endl;
                return false; // Skip this sample if image is not found
            }
        }

        std::vector<float> raw_image;
        int w, h;
        // For now, we assume a function that can handle png/jpg might be needed
        // The existing `load_pgm_as_gray` is a good start.
        if (!load_pgm_as_gray(image_path_str, raw_image, w, h)) {
            // std::cerr << "Warning: Failed to load image " << image_path_str << std::endl;
            return false;
        }

        if (raw_image.empty()) return false;

        // Resize the loaded image to the network's input size (28x28)
        out_image_28x28 = resize_nn_to_28x28(raw_image, w, h);
        return true;
    }

    size_t getTrainCount() const { return train_samples.size(); }
    const std::vector<RSNA_Sample>& getTrainSamples() const { return train_samples; }
};

// ============================================================================
// LEGACY DIGITS DATA LOADER (MNIST/Kaggle Digit Recognizer CSV)
// ---------------------------------------------------------------------------
// Minimal implementation to satisfy legacy training path when enabled via
// environment variable LEGACY_DIGITS=1. Not used for RSNA flows.
// ----------------------------------------------------------------------------
class KaggleDigitLoader {
private:
    std::vector<std::vector<float>> train_images; // each size 784 normalized to [0,1]
    std::vector<int> train_labels;                // 0..9
    std::vector<std::vector<float>> test_images;  // each size 784

    static bool parse_csv_row(const std::string& line, std::vector<int>& out_vals) {
        out_vals.clear();
        out_vals.reserve(785);
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            try { out_vals.push_back(std::stoi(cell)); }
            catch (...) { return false; }
        }
        return !out_vals.empty();
    }

public:
    bool loadTrainData(const std::string& csv_path) {
        std::ifstream f(csv_path);
        if (!f.is_open()) {
            std::cerr << "Cannot open train CSV: " << csv_path << std::endl;
            return false;
        }
        train_images.clear();
        train_labels.clear();
        std::string line;
        // Skip header if present
        if (std::getline(f, line)) {
            if (line.find("label") == std::string::npos) {
                // first line is data; process it
                std::vector<int> vals;
                if (parse_csv_row(line, vals)) {
                    if (!vals.empty()) {
                        int label = vals[0];
                        std::vector<float> img(784, 0.0f);
                        for (size_t i = 1; i < vals.size() && i <= 784; ++i) {
                            img[i-1] = std::clamp(vals[i] / 255.0f, 0.0f, 1.0f);
                        }
                        train_labels.push_back(label);
                        train_images.push_back(std::move(img));
                    }
                }
            }
        }
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::vector<int> vals;
            if (!parse_csv_row(line, vals)) continue;
            if (vals.empty()) continue;
            int label = vals[0];
            std::vector<float> img(784, 0.0f);
            for (size_t i = 1; i < vals.size() && i <= 784; ++i) {
                img[i-1] = std::clamp(vals[i] / 255.0f, 0.0f, 1.0f);
            }
            train_labels.push_back(label);
            train_images.push_back(std::move(img));
        }
        std::cout << "Loaded digits train: images=" << train_images.size() << std::endl;
        return !train_images.empty();
    }

    bool loadTestData(const std::string& csv_path) {
        std::ifstream f(csv_path);
        if (!f.is_open()) {
            std::cerr << "Cannot open test CSV: " << csv_path << std::endl;
            return false;
        }
        test_images.clear();
        std::string line;
        // Skip header if present (no label column expected here)
        if (std::getline(f, line)) {
            if (line.find("pixel") == std::string::npos) {
                // first line is data; process it
                std::vector<int> vals;
                if (parse_csv_row(line, vals)) {
                    std::vector<float> img(784, 0.0f);
                    for (size_t i = 0; i < vals.size() && i < 784; ++i) {
                        img[i] = std::clamp(vals[i] / 255.0f, 0.0f, 1.0f);
                    }
                    test_images.push_back(std::move(img));
                }
            }
        }
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::vector<int> vals;
            if (!parse_csv_row(line, vals)) continue;
            std::vector<float> img(784, 0.0f);
            for (size_t i = 0; i < vals.size() && i < 784; ++i) {
                img[i] = std::clamp(vals[i] / 255.0f, 0.0f, 1.0f);
            }
            test_images.push_back(std::move(img));
        }
        std::cout << "Loaded digits test: images=" << test_images.size() << std::endl;
        return !test_images.empty();
    }

    const std::vector<std::vector<float>>& getTrainImages() const { return train_images; }
    const std::vector<int>& getTrainLabels() const { return train_labels; }
    const std::vector<std::vector<float>>& getTestImages() const { return test_images; }
};

// ============================================================================ 
// IMPROVED CUDA KERNELS
// ============================================================================ 

__global__ void initRandomStates(curandState* states, unsigned long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void setInputKernel(OpticalNeuron* all_neurons_batch, const float* batch_inputs, 
                             int input_size, int total_neurons, int batch_size) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sample_idx = global_idx / input_size;
    int neuron_idx = global_idx % input_size;

    if (sample_idx >= batch_size || neuron_idx >= input_size) return;

    float input_value = batch_inputs[global_idx];
    OpticalNeuron& neuron = all_neurons_batch[sample_idx * total_neurons + neuron_idx];
    
    neuron.activation = input_value;
    neuron.potential = input_value * 2.0f; // Better initial scaling
    neuron.field_amplitude = Complex(sqrtf(input_value + 0.01f), 0.0f);
}

// CRITICAL FIX: Forward pass with correct synapse iteration and signal propagation
__global__ void forwardPassKernel(OpticalNeuron* all_neurons_batch, OpticalSynapse* synapses,
                                 int num_neurons, int num_synapses,
                                 int* layer_offsets, int num_layers,
                                 const int* incoming_indices, const int* incoming_offsets,
                                 float dt, curandState* rand_states, int batch_size) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sample_idx = global_idx / num_neurons;
    int neuron_idx = global_idx % num_neurons;

    if (sample_idx >= batch_size) return;

    OpticalNeuron* neurons = all_neurons_batch + sample_idx * num_neurons;
    OpticalNeuron& neuron = neurons[neuron_idx];
    
    // Find layer - CORRECTED logic
    int layer = num_layers - 1;  // Default to last layer
    for (int l = 0; l < num_layers - 1; l++) {
        if (neuron_idx >= layer_offsets[l] && neuron_idx < layer_offsets[l + 1]) {
            layer = l;
            break;
        }
    }
    
    if (layer == 0) return; // Skip input layer
    
    // CRITICAL: Reset neuron potential before accumulating
    neuron.potential = 0.0f;
    Complex total_field(0.0f, 0.0f);
    int connections = 0;
    
    // Iterate only incoming synapses for this neuron using adjacency
    int start = incoming_offsets[neuron_idx];
    int end   = incoming_offsets[neuron_idx + 1];
    for (int idx = start; idx < end; ++idx) {
        int s = incoming_indices[idx];
        const OpticalSynapse& synapse = synapses[s];
        const OpticalNeuron& pre_neuron = neurons[synapse.pre_neuron_id];
        float signal_strength = pre_neuron.activation * synapse.weight;
        neuron.potential += signal_strength;
        Complex weighted_field = pre_neuron.field_amplitude * synapse.weight;
        total_field = total_field + weighted_field;
        connections++;
    }
    
    // Debug print disabled by default to avoid console flooding
    // if (sample_idx == 0 && neuron_idx < 790 && neuron_idx >= 784) {
    //     printf("Neuron %d (layer %d): found %d connections\n", neuron_idx, layer, connections);
    // }
    
    // Only update if we have connections
    if (connections > 0) {
        // Normalize by sqrt of connections to prevent explosion
        float norm_factor = 1.0f / sqrtf((float)connections);
        neuron.potential *= norm_factor;
        
        // Simple activation function
        neuron.activation = tanhf(neuron.potential);
        neuron.field_amplitude = total_field * norm_factor;
    } else {
        // Debug print disabled (no connections case)
        // if (sample_idx == 0 && neuron_idx < 790 && neuron_idx >= 784) {
        //     printf("Neuron %d has NO connections!\n", neuron_idx);
        // }
    }
}

// Layer-sequential forward pass kernel: processes only one layer [layer_start, layer_start+layer_size)
__global__ void forwardPassLayerKernel(OpticalNeuron* all_neurons_batch, const OpticalSynapse* synapses,
                                       int num_neurons,
                                       const int* incoming_indices, const int* incoming_offsets,
                                       int layer_start, int layer_size,
                                       float dt, int batch_size,
                                       int use_max_plus, float li_alpha) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sample_idx = global_idx / layer_size;
    int idx_in_layer = global_idx % layer_size;
    if (sample_idx >= batch_size) return;

    int neuron_idx = layer_start + idx_in_layer;

    OpticalNeuron* neurons = all_neurons_batch + sample_idx * num_neurons;
    OpticalNeuron& neuron = neurons[neuron_idx];

    // Reset and accumulate only from incoming synapses
    float acc = use_max_plus ? -1.0e9f : 0.0f;
    Complex total_field(0.0f, 0.0f);
    int start = incoming_offsets[neuron_idx];
    int end   = incoming_offsets[neuron_idx + 1];
    int connections = 0;
    for (int idx = start; idx < end; ++idx) {
        int s = incoming_indices[idx];
        const OpticalSynapse& syn = synapses[s];
        const OpticalNeuron& pre = neurons[syn.pre_neuron_id];
        if (use_max_plus) {
            float candidate = pre.activation + syn.weight;
            acc = fmaxf(acc, candidate);
        } else {
            float signal_strength = pre.activation * syn.weight;
            acc += signal_strength;
        }
        Complex weighted_field = pre.field_amplitude * syn.weight;
        total_field = total_field + weighted_field;
        connections++;
    }

    if (connections > 0) {
        float norm_factor = rsqrtf((float)connections + 1e-3f); // 1/sqrt
        // Normalize potential to avoid activation saturation with large fan-in
        neuron.potential = acc;
        if (!use_max_plus) {
            neuron.potential *= norm_factor;
        }
        // Simple per-neuron inhibition proxy (no cross-thread reduction): shrink towards zero
        if (li_alpha > 0.0f) {
            neuron.potential -= li_alpha * neuron.potential;
        }
        neuron.activation = tanhf(neuron.potential);
        neuron.field_amplitude = total_field * norm_factor;
    }
}

// Apply K-Winners-Take-All per layer (keep top-k activations, zero the rest) per sample.
__global__ void applyKWTAKernel(OpticalNeuron* all_neurons_batch,
                                int num_neurons, int layer_start, int layer_size,
                                int batch_size, int k_keep) {
    int sample_idx = blockIdx.x;
    if (sample_idx >= batch_size || k_keep <= 0) return;

    if (threadIdx.x == 0) {
        int base = sample_idx * num_neurons + layer_start;
        if (k_keep > layer_size) k_keep = layer_size;

        bool selected_local[1024];
        for (int i = 0; i < layer_size; ++i) selected_local[i] = false;

        for (int sel = 0; sel < k_keep; ++sel) {
            int best_i = -1; float best_v = -1.0e9f;
            for (int i = 0; i < layer_size; ++i) {
                if (selected_local[i]) continue;
                float v = all_neurons_batch[base + i].activation;
                if (v > best_v) { best_v = v; best_i = i; }
            }
            if (best_i >= 0) selected_local[best_i] = true; else break;
        }

        for (int i = 0; i < layer_size; ++i) {
            if (!selected_local[i]) {
                all_neurons_batch[base + i].activation = 0.0f;
                all_neurons_batch[base + i].potential = 0.0f;
                all_neurons_batch[base + i].field_amplitude = Complex(0.0f, 0.0f);
            } else {
                float a = fmaxf(0.0f, all_neurons_batch[base + i].activation);
                all_neurons_batch[base + i].field_amplitude = Complex(sqrtf(a), 0.0f);
            }
        }
    }
}

// Apply in-place Fast Walsh-Hadamard Transform on a layer per sample.
__global__ void applyHadamardKernel(OpticalNeuron* all_neurons_batch,
                                    int num_neurons, int layer_start, int layer_size,
                                    int batch_size) {
    int sample_idx = blockIdx.x;
    if (sample_idx >= batch_size) return;
    if ((layer_size & (layer_size - 1)) != 0) return; // require power of two

    if (threadIdx.x == 0) {
        int base = sample_idx * num_neurons + layer_start;
        for (int len = 1; len < layer_size; len <<= 1) {
            for (int i = 0; i < layer_size; i += (len << 1)) {
                for (int j = 0; j < len; ++j) {
                    int i0 = base + i + j;
                    int i1 = base + i + j + len;
                    float u = all_neurons_batch[i0].activation;
                    float v = all_neurons_batch[i1].activation;
                    all_neurons_batch[i0].activation = u + v;
                    all_neurons_batch[i1].activation = u - v;
                }
            }
        }
        float norm = rsqrtf((float)layer_size);
        for (int i = 0; i < layer_size; ++i) {
            int idx = base + i;
            all_neurons_batch[idx].activation *= norm;
            all_neurons_batch[idx].potential = all_neurons_batch[idx].activation;
            float a = fmaxf(0.0f, all_neurons_batch[idx].activation);
            all_neurons_batch[idx].field_amplitude = Complex(sqrtf(a), 0.0f);
        }
    }
}

// Improved weight update calculation (one thread per synapse, loop over batch)
__global__ void calculateWeightDeltasKernel(const OpticalSynapse* synapses, const OpticalNeuron* all_neurons_batch,
                                            int num_synapses, int num_neurons, float learning_rate, float dt, 
                                            float* delta_weights, int batch_size) {
    int synapse_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (synapse_idx >= num_synapses) return;

    const OpticalSynapse& synapse = synapses[synapse_idx];
    float accum = 0.0f;

    for (int sample_idx = 0; sample_idx < batch_size; ++sample_idx) {
        const OpticalNeuron* neurons = all_neurons_batch + sample_idx * num_neurons;
        float pre_activity = neurons[synapse.pre_neuron_id].activation;
        float post_activity = neurons[synapse.post_neuron_id].activation;
        float pre_memory = neurons[synapse.pre_neuron_id].memory_trace;
        float post_memory = neurons[synapse.post_neuron_id].memory_trace;
        float plasticity_factor = neurons[synapse.post_neuron_id].plasticity;

        // Enhanced Hebbian with STDP-like components
        float hebbian_term = pre_activity * post_activity;
        float stdp_term = pre_memory * post_activity - post_memory * pre_activity;
        accum += plasticity_factor * (hebbian_term + 0.1f * stdp_term);
    }

    // Average over batch and scale
    delta_weights[synapse_idx] = learning_rate * (accum / max(1, batch_size)) * dt;
}

// Weight application with momentum
__global__ void applyWeightUpdatesKernel(OpticalSynapse* synapses, 
                                        float* delta_weights, 
                                        float* momentum_velocities,
                                        int num_synapses,
                                        float momentum) {
    int synapse_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (synapse_idx >= num_synapses) return;

    float avg_delta_w = delta_weights[synapse_idx]; // already averaged per batch
    
    // Per-synapse momentum to smooth updates
    float velocity = momentum * momentum_velocities[synapse_idx] + (1.0f - momentum) * avg_delta_w;
    momentum_velocities[synapse_idx] = velocity;
    
    OpticalSynapse& synapse = synapses[synapse_idx];
    // Symmetric clamp around 0 to avoid positive-only drift and saturation
    synapse.weight = fmaxf(-1.0f, fminf(1.0f, synapse.weight + velocity));
    
    // Update MZ phases to implement new weight
    float target_phase = synapse.weight * PhysicalConstants::PI;
    synapse.mz_unit.updatePhases(target_phase - synapse.mz_unit.phase_upper, 0.0f);
    
    delta_weights[synapse_idx] = 0.0f;
}

// Apply supervised gradient directly to output neurons (teaching signal)
__global__ void applyOutputGradients(OpticalNeuron* all_neurons_batch, 
                                     const float* gradients,
                                     int num_neurons, int output_size, 
                                     int batch_size, int output_offset,
                                     float learning_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sample_idx = idx / output_size;
    int output_idx = idx % output_size;
    if (sample_idx >= batch_size) return;

    OpticalNeuron& neuron = all_neurons_batch[sample_idx * num_neurons + output_offset + output_idx];
    float grad = gradients[idx];
    neuron.potential -= learning_rate * grad;
    neuron.activation = tanhf(neuron.potential);
    float a = fmaxf(0.0f, neuron.activation);
    neuron.field_amplitude = Complex(sqrtf(a), 0.0f);
}

// Physical clustering with improved dynamics
__global__ void clusterNeuronsKernel(OpticalNeuron* all_neurons_batch, int num_neurons,
                                    float gravitational_strength, float dt, int batch_size) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sample_idx = global_idx / num_neurons;
    int neuron_idx = global_idx % num_neurons;

    if (sample_idx >= batch_size) return;

    OpticalNeuron* neurons = all_neurons_batch + sample_idx * num_neurons;
    OpticalNeuron& neuron = neurons[neuron_idx];
    
    float3 force = make_float3(0.0f, 0.0f, 0.0f);
    
    // Only compute for a subset to save computation
    const int MAX_INTERACTIONS = 32;
    int step = max(1, num_neurons / MAX_INTERACTIONS);
    
    for (int j = 0; j < num_neurons; j += step) {
        if (j == neuron_idx) continue;
        
        const OpticalNeuron& other = neurons[j];
        
        float3 diff = make_float3(
            other.position.x - neuron.position.x,
            other.position.y - neuron.position.y,
            other.position.z - neuron.position.z
        );
        
        float dist_sq = diff.x*diff.x + diff.y*diff.y + diff.z*diff.z + 1.0f;
        float dist = sqrtf(dist_sq);
        
        // Adaptive force based on activation similarity
        float activation_similarity = 1.0f - fabsf(neuron.activation - other.activation);
        float strength = gravitational_strength * activation_similarity / dist_sq;
        
        force.x += strength * diff.x / dist;
        force.y += strength * diff.y / dist;
        force.z += strength * diff.z / dist;
    }
    
    // Update with improved damping
    const float damping = 0.95f;
    neuron.velocity.x = damping * neuron.velocity.x + force.x * dt;
    neuron.velocity.y = damping * neuron.velocity.y + force.y * dt;
    neuron.velocity.z = damping * neuron.velocity.z + force.z * dt;
    
    neuron.position.x += neuron.velocity.x * dt;
    neuron.position.y += neuron.velocity.y * dt;
    neuron.position.z += neuron.velocity.z * dt;
}

// Adaptive temperature annealing
__global__ void temperatureAnnealingKernel(OpticalNeuron* all_neurons_batch, int num_neurons,
                                          float cooling_rate, float min_temperature, 
                                          int batch_size, int epoch) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sample_idx = global_idx / num_neurons;
    int neuron_idx = global_idx % num_neurons;

    if (sample_idx >= batch_size) return;

    OpticalNeuron& neuron = all_neurons_batch[sample_idx * num_neurons + neuron_idx];
    
    // Adaptive cooling based on epoch
    float adaptive_rate = cooling_rate * (1.0f + 0.001f * epoch);
    neuron.temperature = fmaxf(min_temperature, neuron.temperature * adaptive_rate);
}

// DEBUG KERNELS for systematic diagnosis
__global__ void debugNeuronValues(OpticalNeuron* neurons, int layer_start, 
                                  int layer_size, const char* layer_name) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float min_act = 1e9f, max_act = -1e9f, sum_act = 0.0f;
        int zero_count = 0;
        
        for (int i = 0; i < min(layer_size, 10); i++) {
            OpticalNeuron& n = neurons[layer_start + i];
            min_act = fminf(min_act, n.activation);
            max_act = fmaxf(max_act, n.activation);
            sum_act += n.activation;
            if (fabsf(n.activation) < 1e-6f) zero_count++;
            
            // Print first 3 neurons in detail
            if (i < 3) {
                printf("%s[%d]: act=%.6f, pot=%.6f, temp=%.2f, field_mag=%.6f\n",
                       layer_name, i, n.activation, n.potential, 
                       n.temperature, n.field_amplitude.magnitude());
            }
        }
        
        printf("%s Summary: min=%.6f, max=%.6f, mean=%.6f, zeros=%d/%d\n",
               layer_name, min_act, max_act, sum_act/layer_size, zero_count, layer_size);
    }
}

__global__ void debugSynapseWeights(OpticalSynapse* synapses, int start_idx, 
                                   int count, const char* layer_name) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float min_w = 1e9f, max_w = -1e9f, sum_w = 0.0f;
        int zero_count = 0;
        
        for (int i = 0; i < min(count, 100); i++) {
            float w = synapses[start_idx + i].weight;
            min_w = fminf(min_w, w);
            max_w = fmaxf(max_w, w);
            sum_w += w;
            if (fabsf(w) < 1e-6f) zero_count++;
        }
        
        printf("%s Synapses: min_weight=%.6f, max_weight=%.6f, mean=%.6f, zeros=%d\n",
               layer_name, min_w, max_w, sum_w/count, zero_count);
    }
}

__global__ void verifyInputLoaded(float* batch_inputs, int batch_size, int input_size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float sum = 0.0f;
        int non_zero = 0;
        for (int i = 0; i < min(batch_size * input_size, 1000); i++) {
            sum += batch_inputs[i];
            if (batch_inputs[i] > 0.01f) non_zero++;
        }
        printf("INPUT BUFFER: sum=%.2f, non_zero=%d/1000\n", sum, non_zero);
    }
}

// Output layer specific updates
__global__ void rewardPunishmentKernel(OpticalNeuron* all_neurons_batch, const int* target_labels, 
                                     int num_neurons, int output_size, int batch_size, 
                                     int output_offset, float reward_strength) {
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_idx >= batch_size) return;

    OpticalNeuron* output_neurons = all_neurons_batch + sample_idx * num_neurons + output_offset;
    int target_label = target_labels[sample_idx];

    for (int i = 0; i < output_size; ++i) {
        if (i == target_label) {
            // Reward correct neuron
            output_neurons[i].potential += reward_strength;
            output_neurons[i].plasticity = fminf(1.0f, output_neurons[i].plasticity + 0.1f);
        } else {
            // Mild punishment for incorrect neurons
            output_neurons[i].potential -= reward_strength * 0.2f;
            output_neurons[i].plasticity = fmaxf(0.1f, output_neurons[i].plasticity * 0.95f);
        }
    }
    // Immediately reflect new potentials into activations so Hebbian updates see the change
    for (int i = 0; i < output_size; ++i) {
        output_neurons[i].activation = tanhf(output_neurons[i].potential);
        float a = fmaxf(0.0f, output_neurons[i].activation);
        output_neurons[i].field_amplitude = Complex(sqrtf(a), 0.0f);
    }
}

// Gather output logits (potentials) into a compact buffer [batch_size * output_size]
__global__ void getOutputActivationsKernel(const OpticalNeuron* all_neurons_batch, float* output_activations,
                                           int num_neurons, int output_size, int batch_size, int output_offset) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sample_idx = global_idx / output_size;
    int output_neuron_idx = global_idx % output_size;
    if (sample_idx >= batch_size) return;
    const OpticalNeuron* neurons = all_neurons_batch + sample_idx * num_neurons;
    // Use potential (logit) for loss/gradient computation to avoid tanh saturation
    output_activations[global_idx] = neurons[output_offset + output_neuron_idx].potential;
}

// Apply a simple Clements-like MZI mesh to the output layer field amplitudes.
// Alternating pair stages; parameters provided per pair.
__global__ void applyOutputMZIMesh(OpticalNeuron* all_neurons_batch,
                                   int num_neurons, int batch_size,
                                   int output_offset, int output_size,
                                   const float* theta, const float* phi,
                                   const float* visibility, const float* loss,
                                   const float* path_diff, const float* coherence_len,
                                   const int* stage_offsets, int num_stages,
                                   curandState* rand_states,
                                   float logit_gain) {
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_idx >= batch_size) return;

    OpticalNeuron* neurons = all_neurons_batch + sample_idx * num_neurons;

    // Local buffer for output fields
    Complex e[16]; // supports up to 16 outputs; OUTPUT_SIZE=10
    for (int i = 0; i < output_size; ++i) e[i] = neurons[output_offset + i].field_amplitude;

    int pair_index = 0;
    for (int s = 0; s < num_stages; ++s) {
        int start = stage_offsets[s];
        int end   = stage_offsets[s+1];
        int parity = s & 1; // 0: (0,1)(2,3)... 1: (1,2)(3,4)...
        for (int p = start; p < end; ++p) {
            int k = 2*(p - start) + parity;
            if (k+1 >= output_size) break;
            Complex a = e[k];
            Complex b = e[k+1];
            float th = theta[p];
            float ph = phi[p];
            float vis0 = visibility[p];
            float pd  = path_diff[p];
            float Lc  = fmaxf(coherence_len[p], 1e-6f);
            float vis = vis0 * expf(-(pd*pd)/(2.0f*Lc*Lc));
            float ls = loss[p];

            // First 50/50 coupler
            float inv_sqrt2 = 0.70710678f;
            Complex u = Complex(a.real*inv_sqrt2 - a.imag*0.0f, a.imag*inv_sqrt2 + a.real*0.0f); // copy a
            u = Complex((a.real + b.real)*inv_sqrt2, (a.imag + b.imag)*inv_sqrt2);
            Complex v = Complex((a.real - b.real)*inv_sqrt2, (a.imag - b.imag)*inv_sqrt2);

            // Phase shift on one arm (v)
            float cth = cosf(th), sth = sinf(th);
            Complex vps = Complex(v.real*cth - v.imag*sth, v.real*sth + v.imag*cth);

            // Second 50/50 coupler
            Complex o1 = Complex((u.real + vps.real)*inv_sqrt2, (u.imag + vps.imag)*inv_sqrt2);
            Complex o2 = Complex((u.real - vps.real)*inv_sqrt2, (u.imag - vps.imag)*inv_sqrt2);

            // Output phase on o2 (simple)
            float cph = cosf(ph), sph = sinf(ph);
            o2 = Complex(o2.real*cph - o2.imag*sph, o2.real*sph + o2.imag*cph);

            // Apply visibility and loss (amplitude)
            o1 = o1 * (vis * ls);
            o2 = o2 * (vis * ls);

            e[k]   = o1;
            e[k+1] = o2;
        }
    }

    // Compute detected magnitudes with small noise
    float mags[16];
    float sum_mag = 0.0f;
    for (int i = 0; i < output_size; ++i) {
        int rng_idx = sample_idx * num_neurons + output_offset + i;
        float noise = 0.01f * curand_normal(&rand_states[rng_idx]);
        float mag = fmaxf(0.0f, e[i].magnitude() + noise);
        mags[i] = mag;
        sum_mag += mag;
    }
    float mean_mag = sum_mag / fmaxf(1.0f, (float)output_size);
    // Compute std for normalization
    float var = 0.0f;
    for (int i = 0; i < output_size; ++i) {
        float d = mags[i] - mean_mag;
        var += d * d;
    }
    float std_mag = sqrtf(var / fmaxf(1.0f, (float)output_size)) + 1e-6f;

    // Write back fields and set potentials as centered, scaled logits
    const float gain = logit_gain; // tuneable gain for logits
    for (int i = 0; i < output_size; ++i) {
        neurons[output_offset + i].field_amplitude = e[i];
        float logit = gain * ((mags[i] - mean_mag) / std_mag);
        neurons[output_offset + i].potential = logit;        // unbounded logits
        neurons[output_offset + i].activation = tanhf(logit); // bounded activation for physics
    }
}

// Compute per-sample total optical energy on output layer (sum |E|^2)
__global__ void computeOutputEnergy(const OpticalNeuron* all_neurons_batch,
                                    int num_neurons, int batch_size,
                                    int output_offset, int output_size,
                                    float* out_energy) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;
    const OpticalNeuron* neurons = all_neurons_batch + b * num_neurons;
    float sumE = 0.0f;
    for (int i = 0; i < output_size; ++i) {
        Complex f = neurons[output_offset + i].field_amplitude;
        sumE += f.real * f.real + f.imag * f.imag;
    }
    out_energy[b] = sumE;
}

// Phase update for output MZI mesh using targets (local, no backprop).
// For each pair (k,k+1) in each stage, push theta to favor the target channel
// based on intensity difference after the mesh.
__global__ void updateMZIPhasesFromTargets(const OpticalNeuron* all_neurons_batch,
                                           int num_neurons, int batch_size,
                                           int output_offset, int output_size,
                                           const int* target_labels,
                                           const int* stage_offsets, int num_stages,
                                           float* theta, float phase_lr) {
    int p = blockIdx.x * blockDim.x + threadIdx.x; // global pair index
    // total number of pairs is stage_offsets[num_stages]
    int total_pairs = stage_offsets[num_stages];
    if (p >= total_pairs) return;

    // Find stage s for pair p
    int s = 0;
    for (int i = 0; i < num_stages; ++i) { if (p >= stage_offsets[i]) s = i; else break; }
    int local = p - stage_offsets[s];
    int parity = s & 1;
    int k = 2 * local + parity;
    if (k + 1 >= output_size) return;

    float grad = 0.0f;
    for (int b = 0; b < batch_size; ++b) {
        const OpticalNeuron* neurons = all_neurons_batch + b * num_neurons;
        Complex f1 = neurons[output_offset + k].field_amplitude;
        Complex f2 = neurons[output_offset + k + 1].field_amplitude;
        float I1 = f1.real * f1.real + f1.imag * f1.imag;
        float I2 = f2.real * f2.real + f2.imag * f2.imag;
        int t = target_labels[b];
        if (t == k)       grad += (I1 - I2);
        else if (t == k+1) grad -= (I1 - I2);
    }
    grad /= max(1, batch_size);
    float new_theta = theta[p] + phase_lr * grad;
    // wrap into [0, 2pi)
    float two_pi = 2.0f * PhysicalConstants::PI;
    new_theta = fmodf(new_theta + two_pi, two_pi);
    theta[p] = new_theta;
}

// ============================================================================ 
// IMPROVED OPTICAL NEURAL NETWORK
// ============================================================================ 

class OpticalNeuralNetwork {
public:
    static constexpr int BATCH_SIZE = 128;

private:
    // Architecture
    static constexpr int INPUT_SIZE = 784;
    static constexpr int HIDDEN1_SIZE = 512;
    static constexpr int HIDDEN2_SIZE = 256;
    static constexpr int HIDDEN3_SIZE = 128;
    static constexpr int OUTPUT_SIZE = 10;
    static constexpr int TOTAL_NEURONS = INPUT_SIZE + HIDDEN1_SIZE + HIDDEN2_SIZE + HIDDEN3_SIZE + OUTPUT_SIZE;
    
    // Device memory
    OpticalNeuron* d_neurons;
    OpticalSynapse* d_synapses;
    int* d_layer_offsets;
    int* d_incoming_indices;
    int* d_incoming_offsets;
    curandState* d_rand_states;
    float* d_delta_weights;
    float* d_momentum_velocities;
    float* d_batch_inputs;
    int* d_target_labels;
    float* d_temp_energy; // BATCH_SIZE-sized buffer for optical energy diagnostics
    float* d_output_activations;
    
    // Hyperparameters (host-configurable)
    float logit_gain;         // scaling for output logits
    float mzi_phase_lr;       // learning rate for MZI phase updates
    
    // Non-conventional feature toggles
    int   use_max_plus_hidden;   // 1: max-plus accumulation on hidden layers
    int   use_hadamard;          // 1: apply Hadamard mixing after hidden layers
    int   use_kwta;              // 1: apply K-Winners-Take-All after hidden layers
    int   kwta_keep_hidden;      // number of winners to keep (if >0)
    float kwta_frac_hidden;      // fraction of winners to keep (if kwta_keep_hidden==0)
    float li_alpha_hidden;       // simple inhibition shrink factor [0..1] for hidden layers
    
    // Output MZI mesh parameters (Clements-style alternating pairs)
    float* d_mzi_theta;       // phase shifts per pair
    float* d_mzi_phi;         // output phase per pair (simple model)
    float* d_mzi_visibility;    // base visibility factor [0,1]
    float* d_mzi_loss;          // amplitude loss factor per pair (0..1)
    float* d_mzi_path_diff;     // path difference per pair (arbitrary units)
    float* d_mzi_coherence_len; // coherence length per pair (same units)
    int*   d_mzi_stage_offsets; // offsets into pair arrays per stage
    int    mzi_num_stages;
    int    mzi_total_pairs;
    
    // Network structure
    std::vector<int> layer_sizes;
    std::vector<int> layer_offsets;
    int num_synapses;
    
    // CUDA resources
    cudaStream_t compute_stream;
    cudaStream_t memory_stream;
    cudaEvent_t memory_event;
    
    // Training parameters
    float learning_rate;
    float base_temperature;
    float momentum;
    int epoch;
    int global_step;  // Track total steps instead of per-batch increments
    
public:
    OpticalNeuralNetwork() : 
        learning_rate(0.20f),           // Auto-tuned default
        base_temperature(4.0f),          // Auto-tuned default
        momentum(0.85f),                 // Auto-tuned default
        epoch(0),
        global_step(0),
        logit_gain(1.5f),
        mzi_phase_lr(0.002f),
        use_max_plus_hidden(0),
        use_hadamard(0),
        use_kwta(0),
        kwta_keep_hidden(0),
        kwta_frac_hidden(0.0f),
        li_alpha_hidden(0.0f) {
        
        // Override hyperparameters via environment variables if provided
        if (const char* s = std::getenv("LEARNING_RATE_INIT")) {
            try { learning_rate = std::stof(s); } catch (...) {}
        }
        if (const char* s = std::getenv("BASE_TEMPERATURE")) {
            try { base_temperature = std::stof(s); } catch (...) {}
        }
        if (const char* s = std::getenv("MOMENTUM")) {
            try { momentum = std::stof(s); } catch (...) {}
        }
        if (const char* s = std::getenv("LOGIT_GAIN")) {
            try { logit_gain = std::stof(s); } catch (...) {}
        }
        if (const char* s = std::getenv("MZI_PHASE_LR")) {
            try { mzi_phase_lr = std::stof(s); } catch (...) {}
        }
        if (const char* s = std::getenv("USE_MAX_PLUS")) {
            try { use_max_plus_hidden = std::stoi(s); } catch (...) {}
        }
        if (const char* s = std::getenv("USE_HADAMARD")) {
            try { use_hadamard = std::stoi(s); } catch (...) {}
        }
        if (const char* s = std::getenv("USE_KWTA")) {
            try { use_kwta = std::stoi(s); } catch (...) {}
        }
        if (const char* s = std::getenv("KWTA_KEEP")) {
            try { kwta_keep_hidden = std::stoi(s); } catch (...) {}
        }
        if (const char* s = std::getenv("KWTA_FRAC")) {
            try { kwta_frac_hidden = std::stof(s); } catch (...) {}
        }
        if (const char* s = std::getenv("LI_ALPHA")) {
            try { li_alpha_hidden = std::stof(s); } catch (...) {}
        }

        // Define architecture
        layer_sizes = {INPUT_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, HIDDEN3_SIZE, OUTPUT_SIZE};
        layer_offsets.resize(layer_sizes.size());
        layer_offsets[0] = 0;
        for (size_t i = 1; i < layer_sizes.size(); i++) {
            layer_offsets[i] = layer_offsets[i-1] + layer_sizes[i-1];
        }
        
        // Calculate synapses
        num_synapses = 0;
        for (size_t i = 0; i < layer_sizes.size() - 1; i++) {
            num_synapses += layer_sizes[i] * layer_sizes[i+1];
        }
        
        std::cout << "Network architecture: ";
        for (size_t i = 0; i < layer_sizes.size(); i++) {
            std::cout << layer_sizes[i];
            if (i < layer_sizes.size() - 1) std::cout << " → ";
        }
        std::cout << "\nTotal synapses: " << num_synapses << std::endl;
        
        // Initialize CUDA
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaStreamCreate(&compute_stream));
        CUDA_CHECK(cudaStreamCreate(&memory_stream));
        CUDA_CHECK(cudaEventCreate(&memory_event));
        
        allocateMemory();
        initializeNetwork();
        initializeRandomStates();

        // Initialize output MZI mesh parameters (alternating pair stages)
        mzi_num_stages = OUTPUT_SIZE - 1;
        std::vector<int> stage_offsets(mzi_num_stages + 1, 0);
        for (int s = 0; s < mzi_num_stages; ++s) {
            int pairs = (OUTPUT_SIZE - (s % 2)) / 2;
            stage_offsets[s + 1] = stage_offsets[s] + pairs;
        }
        mzi_total_pairs = stage_offsets.back();

        std::vector<float> h_theta(mzi_total_pairs), h_phi(mzi_total_pairs), h_vis(mzi_total_pairs), h_loss(mzi_total_pairs);
        std::vector<float> h_path(mzi_total_pairs), h_cohl(mzi_total_pairs);
        std::mt19937 gen2(12345);
        std::uniform_real_distribution<float> u01(0.0f, 1.0f);
        for (int i = 0; i < mzi_total_pairs; ++i) {
            h_theta[i] = 2.0f * PhysicalConstants::PI * u01(gen2);
            h_phi[i]   = 2.0f * PhysicalConstants::PI * u01(gen2);
            h_vis[i]   = 0.95f;
            h_loss[i]  = 0.98f; // amplitude loss factor
            h_path[i]  = 0.02f * (u01(gen2) - 0.5f); // small random path diff
            h_cohl[i]  = 0.10f; // coherence length
        }
        CUDA_CHECK(cudaMalloc(&d_mzi_theta, mzi_total_pairs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_mzi_phi,   mzi_total_pairs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_mzi_visibility, mzi_total_pairs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_mzi_loss,  mzi_total_pairs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_mzi_path_diff,  mzi_total_pairs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_mzi_coherence_len,  mzi_total_pairs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_mzi_stage_offsets, stage_offsets.size() * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_mzi_theta, h_theta.data(), h_theta.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mzi_phi,   h_phi.data(),   h_phi.size()   * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mzi_visibility, h_vis.data(), h_vis.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mzi_loss,  h_loss.data(),  h_loss.size()  * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mzi_path_diff,  h_path.data(),  h_path.size()  * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mzi_coherence_len,  h_cohl.data(),  h_cohl.size()  * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mzi_stage_offsets, stage_offsets.data(), stage_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    }

    // Save only synapse weights to a simple binary file.
    bool saveWeights(const std::string& dir, const std::string& filename = "weights.bin") {
        namespace fs = std::filesystem;
        try { fs::create_directories(fs::path(dir)); } catch (...) {}
        std::vector<OpticalSynapse> h_syn(num_synapses);
        CUDA_CHECK(cudaMemcpy(h_syn.data(), d_synapses, num_synapses * sizeof(OpticalSynapse), cudaMemcpyDeviceToHost));
        std::vector<float> w(num_synapses);
        for (int i = 0; i < num_synapses; ++i) w[i] = h_syn[i].weight;
        std::ofstream f(fs::path(dir) / filename, std::ios::binary);
        if (!f.is_open()) return false;
        int n = num_synapses;
        f.write(reinterpret_cast<const char*>(&n), sizeof(int));
        f.write(reinterpret_cast<const char*>(w.data()), n * sizeof(float));
        return true;
    }

    // Load synapse weights from binary file saved by saveWeights.
    bool loadWeights(const std::string& filepath) {
        std::ifstream f(filepath, std::ios::binary);
        if (!f.is_open()) return false;
        int n_file = 0;
        f.read(reinterpret_cast<char*>(&n_file), sizeof(int));
        if (n_file != num_synapses) {
            std::cerr << "Checkpoint mismatch: expected " << num_synapses << ", got " << n_file << std::endl;
            return false;
        }
        std::vector<float> w(n_file);
        f.read(reinterpret_cast<char*>(w.data()), n_file * sizeof(float));
        if (!f) return false;
        std::vector<OpticalSynapse> h_syn(num_synapses);
        CUDA_CHECK(cudaMemcpy(h_syn.data(), d_synapses, num_synapses * sizeof(OpticalSynapse), cudaMemcpyDeviceToHost));
        for (int i = 0; i < num_synapses; ++i) h_syn[i].weight = w[i];
        CUDA_CHECK(cudaMemcpy(d_synapses, h_syn.data(), num_synapses * sizeof(OpticalSynapse), cudaMemcpyHostToDevice));
        return true;
    }
    
    ~OpticalNeuralNetwork() {
        cudaFree(d_neurons);
        cudaFree(d_synapses);
        cudaFree(d_layer_offsets);
        cudaFree(d_incoming_indices);
        cudaFree(d_incoming_offsets);
        cudaFree(d_rand_states);
        cudaFree(d_delta_weights);
        cudaFree(d_momentum_velocities);
        cudaFree(d_batch_inputs);
        cudaFree(d_target_labels);
        cudaFree(d_temp_energy);
        cudaFree(d_output_activations);
        cudaFree(d_mzi_theta);
        cudaFree(d_mzi_phi);
        cudaFree(d_mzi_visibility);
        cudaFree(d_mzi_loss);
        cudaFree(d_mzi_stage_offsets);
        cudaFree(d_mzi_path_diff);
        cudaFree(d_mzi_coherence_len);
        cudaEventDestroy(memory_event);
        cudaStreamDestroy(compute_stream);
        cudaStreamDestroy(memory_stream);
    }
    
private:
    void allocateMemory() {
        size_t neuron_bytes = BATCH_SIZE * TOTAL_NEURONS * sizeof(OpticalNeuron);
        size_t synapse_bytes = num_synapses * sizeof(OpticalSynapse);
        size_t offset_bytes = layer_offsets.size() * sizeof(int);
        size_t rand_bytes = BATCH_SIZE * TOTAL_NEURONS * sizeof(curandState);
        size_t delta_bytes = num_synapses * sizeof(float);
        size_t input_bytes = BATCH_SIZE * INPUT_SIZE * sizeof(float);
        size_t label_bytes = BATCH_SIZE * sizeof(int);
        size_t energy_bytes = BATCH_SIZE * sizeof(float);
        size_t out_bytes = BATCH_SIZE * OUTPUT_SIZE * sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_neurons, neuron_bytes));
        CUDA_CHECK(cudaMalloc(&d_synapses, synapse_bytes));
        CUDA_CHECK(cudaMalloc(&d_layer_offsets, offset_bytes));
        CUDA_CHECK(cudaMalloc(&d_rand_states, rand_bytes));
        CUDA_CHECK(cudaMalloc(&d_delta_weights, delta_bytes));
        CUDA_CHECK(cudaMalloc(&d_momentum_velocities, delta_bytes));
        CUDA_CHECK(cudaMalloc(&d_batch_inputs, input_bytes));
        CUDA_CHECK(cudaMalloc(&d_target_labels, label_bytes));
        CUDA_CHECK(cudaMalloc(&d_temp_energy, energy_bytes));
        CUDA_CHECK(cudaMalloc(&d_output_activations, out_bytes));
        
        // CRITICAL FIX: Initialize delta weights to zero
        CUDA_CHECK(cudaMemset(d_delta_weights, 0, delta_bytes));
        CUDA_CHECK(cudaMemset(d_momentum_velocities, 0, delta_bytes));
        
        CUDA_CHECK(cudaMemcpy(d_layer_offsets, layer_offsets.data(), 
                              offset_bytes, cudaMemcpyHostToDevice));
        
        std::cout << "Allocated " << (neuron_bytes + synapse_bytes + offset_bytes +
                                      rand_bytes + 2*delta_bytes + input_bytes + label_bytes + energy_bytes + out_bytes) / (1024*1024) 
                  << " MB GPU memory" << std::endl;
    }
    
    void initializeNetwork() {
        std::vector<OpticalNeuron> h_neurons(BATCH_SIZE * TOTAL_NEURONS);
        std::vector<OpticalSynapse> h_synapses(num_synapses);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        
        // Create template neurons
        std::vector<OpticalNeuron> template_neurons(TOTAL_NEURONS);
        
        int neuron_idx = 0;
        for (size_t layer = 0; layer < layer_sizes.size(); layer++) {
            int layer_size = layer_sizes[layer];
            int grid_size = static_cast<int>(ceilf(sqrtf(layer_size)));
            float spacing = 50.0f * (layer + 1);
            
            for (int i = 0; i < layer_size; i++) {
                OpticalNeuron& neuron = template_neurons[neuron_idx++];
                
                int row = i / grid_size;
                int col = i % grid_size;
                
                neuron.position = make_float3(
                    (col - grid_size/2.0f) * spacing,
                    (row - grid_size/2.0f) * spacing,
                    layer * 200.0f
                );
                
                neuron.velocity = make_float3(0.0f, 0.0f, 0.0f);
                neuron.mass = 1.0f;
                
                // Spectral multiplexing
                neuron.wavelength = 400e-9f + layer * 50e-9f;
                neuron.coherence_length = 1e-3f;
                neuron.field_amplitude = Complex(0.0f, 0.0f);
                neuron.polarization_angle = uniform(gen) * PhysicalConstants::TWO_PI;
                
                neuron.activation = 0.0f;
                neuron.threshold = 0.3f;  // Lower threshold
                neuron.potential = 0.0f;
                neuron.temperature = base_temperature;
                
                neuron.plasticity = 1.0f;
                neuron.memory_trace = 0.0f;
                neuron.cluster_id = -1;
                neuron.age = 0.0f;
            }
        }

        // Replicate across batch
        for (int b = 0; b < BATCH_SIZE; ++b) {
            memcpy(h_neurons.data() + (b * TOTAL_NEURONS), 
                   template_neurons.data(), 
                   TOTAL_NEURONS * sizeof(OpticalNeuron));
        }
        
        // Initialize synapses with improved weights
        int synapse_idx = 0;
        for (size_t layer = 0; layer < layer_sizes.size() - 1; layer++) {
            int pre_layer_start = layer_offsets[layer];
            int pre_layer_size = layer_sizes[layer];
            int post_layer_start = layer_offsets[layer + 1];
            int post_layer_size = layer_sizes[layer + 1];
            
            float scale = sqrtf(2.0f / (pre_layer_size + post_layer_size));
            
            for (int pre = 0; pre < pre_layer_size; pre++) {
                for (int post = 0; post < post_layer_size; post++) {
                    OpticalSynapse& synapse = h_synapses[synapse_idx++];
                    
                    synapse.pre_neuron_id = pre_layer_start + pre;
                    synapse.post_neuron_id = post_layer_start + post;
                    
                    // Zero-mean weight initialization to avoid positive-only drift
                    synapse.weight = normal(gen) * scale;
                    synapse.weight = fmaxf(-1.0f, fminf(1.0f, synapse.weight));
                    
                    // Calculate delay
                    float3 pre_pos = template_neurons[synapse.pre_neuron_id].position;
                    float3 post_pos = template_neurons[synapse.post_neuron_id].position;
                    float distance = sqrtf(
                        (pre_pos.x - post_pos.x) * (pre_pos.x - post_pos.x) +
                        (pre_pos.y - post_pos.y) * (pre_pos.y - post_pos.y) +
                        (pre_pos.z - post_pos.z) * (pre_pos.z - post_pos.z)
                    );
                    synapse.delay = distance / PhysicalConstants::SPEED_OF_LIGHT;
                    
                    // Initialize MZ unit
                    synapse.mz_unit.phase_upper = synapse.weight * PhysicalConstants::PI;
                    synapse.mz_unit.phase_lower = 0.0f;
                    synapse.mz_unit.coupling_ratio = 0.5f;
                    synapse.mz_unit.loss_factor = 0.98f;
                    
                    synapse.pre_trace = 0.0f;
                    synapse.post_trace = 0.0f;
                    synapse.eligibility_trace = 0.0f;
                }
            }
        }
        
        // Upload to GPU
        // Build incoming adjacency for forward pass
        {
            std::vector<std::vector<int>> incoming(TOTAL_NEURONS);
            for (int s = 0; s < num_synapses; ++s) {
                int post = h_synapses[s].post_neuron_id;
                if (post >= 0 && post < TOTAL_NEURONS) incoming[post].push_back(s);
            }
            std::vector<int> h_in_offsets(TOTAL_NEURONS + 1, 0);
            for (int n = 0; n < TOTAL_NEURONS; ++n) h_in_offsets[n+1] = h_in_offsets[n] + (int)incoming[n].size();
            std::vector<int> h_in_indices(h_in_offsets.back());
            for (int n = 0; n < TOTAL_NEURONS; ++n) {
                int base = h_in_offsets[n];
                for (size_t j = 0; j < incoming[n].size(); ++j) h_in_indices[base + (int)j] = incoming[n][j];
            }
            CUDA_CHECK(cudaMalloc(&d_incoming_indices, h_in_indices.size() * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_incoming_offsets, (TOTAL_NEURONS + 1) * sizeof(int)));
            if (!h_in_indices.empty()) CUDA_CHECK(cudaMemcpy(d_incoming_indices, h_in_indices.data(), h_in_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_incoming_offsets, h_in_offsets.data(), (TOTAL_NEURONS + 1) * sizeof(int), cudaMemcpyHostToDevice));
        }

        CUDA_CHECK(cudaMemcpy(d_neurons, h_neurons.data(), 
                              BATCH_SIZE * TOTAL_NEURONS * sizeof(OpticalNeuron), 
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_synapses, h_synapses.data(),
                              num_synapses * sizeof(OpticalSynapse), 
                              cudaMemcpyHostToDevice));
        
        std::cout << "Network initialized successfully" << std::endl;
    }
    
    void initializeRandomStates() {
        const int total_threads = BATCH_SIZE * TOTAL_NEURONS;
        dim3 blocks((total_threads + 511) / 512);
        dim3 threads(512);
        
        initRandomStates<<<blocks, threads, 0, compute_stream>>>(
            d_rand_states, time(nullptr), total_threads);
        CUDA_LAUNCH_CHECK("initRandomStates");
    }

    void applyBatchInput() {
        CUDA_CHECK(cudaStreamWaitEvent(compute_stream, memory_event, 0));

        const int total_threads = BATCH_SIZE * INPUT_SIZE;
        const int block_size = 256;
        const int grid_size = (total_threads + block_size - 1) / block_size;

        setInputKernel<<<grid_size, block_size, 0, compute_stream>>>(
            d_neurons, d_batch_inputs, INPUT_SIZE, TOTAL_NEURONS, BATCH_SIZE);
        CUDA_LAUNCH_CHECK("setInputKernel");
    }

public:
    void loadInputBatch(const std::vector<std::vector<float>>& batch_data) {
        if (batch_data.size() > BATCH_SIZE) {
            throw std::runtime_error("Batch size exceeds network capacity");
        }

        std::vector<float> h_batch_inputs(BATCH_SIZE * INPUT_SIZE, 0.0f);
        for(size_t i = 0; i < batch_data.size(); ++i) {
            memcpy(h_batch_inputs.data() + i * INPUT_SIZE, 
                   batch_data[i].data(), 
                   INPUT_SIZE * sizeof(float));
        }

        CUDA_CHECK(cudaMemcpyAsync(d_batch_inputs, h_batch_inputs.data(),
                                  BATCH_SIZE * INPUT_SIZE * sizeof(float), 
                                  cudaMemcpyHostToDevice, memory_stream));
        
        // FIX: Record event for synchronization
        CUDA_CHECK(cudaEventRecord(memory_event, memory_stream));
        applyBatchInput();
    }

    std::vector<std::vector<float>> getOutputBatch() {
        int output_offset = layer_offsets.back();
        std::vector<OpticalNeuron> h_all_neurons(BATCH_SIZE * TOTAL_NEURONS);
        CUDA_CHECK(cudaMemcpy(h_all_neurons.data(), d_neurons, 
                              BATCH_SIZE * TOTAL_NEURONS * sizeof(OpticalNeuron), 
                              cudaMemcpyDeviceToHost));

        std::vector<std::vector<float>> batch_outputs(BATCH_SIZE, 
                                                      std::vector<float>(OUTPUT_SIZE));
        for (int b = 0; b < BATCH_SIZE; ++b) {
            for (int i = 0; i < OUTPUT_SIZE; ++i) {
                batch_outputs[b][i] = h_all_neurons[b * TOTAL_NEURONS + output_offset + i].activation;
            }
        }
        return batch_outputs;
    }

    // Compute accuracy for the most recently processed batch
    float computeBatchAccuracy(const std::vector<int>& targets, int batch_size) {
        // Recompute forward to avoid measuring on mutated outputs (post-teaching signal)
        forward();
        std::vector<std::vector<float>> outs = getOutputBatch();
        int correct = 0;
        int n = std::min((int)targets.size(), batch_size);
        for (int b = 0; b < n; ++b) {
            int best_i = 0; float best_v = outs[b][0];
            for (int i = 1; i < OUTPUT_SIZE; ++i) {
                if (outs[b][i] > best_v) { best_v = outs[b][i]; best_i = i; }
            }
            if (best_i == targets[b]) correct++;
        }
        return n > 0 ? (float)correct / (float)n : 0.0f;
    }
    
    void forward(float dt = 0.01f) {
        // Layer-by-layer propagation to avoid race conditions across layers
        // 1) Hidden1
        {
            const int layer_start = layer_offsets[1];
            const int layer_size  = layer_sizes[1];
            const int total = BATCH_SIZE * layer_size;
            dim3 blocks((total + 511) / 512);
            dim3 threads(512);
            forwardPassLayerKernel<<<blocks, threads, 0, compute_stream>>>(
                d_neurons, d_synapses, TOTAL_NEURONS,
                d_incoming_indices, d_incoming_offsets,
                layer_start, layer_size,
                dt, BATCH_SIZE,
                use_max_plus_hidden, li_alpha_hidden
            );
            CUDA_LAUNCH_CHECK("forwardPassLayerKernel_H1");
            // Optional mixing and sparsification
            if (use_hadamard) {
                applyHadamardKernel<<<BATCH_SIZE, 256, 0, compute_stream>>>(
                    d_neurons, TOTAL_NEURONS, layer_start, layer_size, BATCH_SIZE
                );
                CUDA_LAUNCH_CHECK("applyHadamardKernel_H1");
            }
            if (use_kwta) {
                int k = kwta_keep_hidden > 0 ? kwta_keep_hidden : max(1, (int)(layer_size * kwta_frac_hidden));
                applyKWTAKernel<<<BATCH_SIZE, 256, 0, compute_stream>>>(
                    d_neurons, TOTAL_NEURONS, layer_start, layer_size, BATCH_SIZE, k
                );
                CUDA_LAUNCH_CHECK("applyKWTAKernel_H1");
            }
        }
        // 2) Hidden2
        {
            const int layer_start = layer_offsets[2];
            const int layer_size  = layer_sizes[2];
            const int total = BATCH_SIZE * layer_size;
            dim3 blocks((total + 511) / 512);
            dim3 threads(512);
            forwardPassLayerKernel<<<blocks, threads, 0, compute_stream>>>(
                d_neurons, d_synapses, TOTAL_NEURONS,
                d_incoming_indices, d_incoming_offsets,
                layer_start, layer_size,
                dt, BATCH_SIZE,
                use_max_plus_hidden, li_alpha_hidden
            );
            CUDA_LAUNCH_CHECK("forwardPassLayerKernel_H2");
            if (use_hadamard) {
                applyHadamardKernel<<<BATCH_SIZE, 256, 0, compute_stream>>>(
                    d_neurons, TOTAL_NEURONS, layer_start, layer_size, BATCH_SIZE
                );
                CUDA_LAUNCH_CHECK("applyHadamardKernel_H2");
            }
            if (use_kwta) {
                int k = kwta_keep_hidden > 0 ? kwta_keep_hidden : max(1, (int)(layer_size * kwta_frac_hidden));
                applyKWTAKernel<<<BATCH_SIZE, 256, 0, compute_stream>>>(
                    d_neurons, TOTAL_NEURONS, layer_start, layer_size, BATCH_SIZE, k
                );
                CUDA_LAUNCH_CHECK("applyKWTAKernel_H2");
            }
        }
        // 3) Hidden3
        {
            const int layer_start = layer_offsets[3];
            const int layer_size  = layer_sizes[3];
            const int total = BATCH_SIZE * layer_size;
            dim3 blocks((total + 511) / 512);
            dim3 threads(512);
            forwardPassLayerKernel<<<blocks, threads, 0, compute_stream>>>(
                d_neurons, d_synapses, TOTAL_NEURONS,
                d_incoming_indices, d_incoming_offsets,
                layer_start, layer_size,
                dt, BATCH_SIZE,
                use_max_plus_hidden, li_alpha_hidden
            );
            CUDA_LAUNCH_CHECK("forwardPassLayerKernel_H3");
            if (use_hadamard) {
                applyHadamardKernel<<<BATCH_SIZE, 256, 0, compute_stream>>>(
                    d_neurons, TOTAL_NEURONS, layer_start, layer_size, BATCH_SIZE
                );
                CUDA_LAUNCH_CHECK("applyHadamardKernel_H3");
            }
            if (use_kwta) {
                int k = kwta_keep_hidden > 0 ? kwta_keep_hidden : max(1, (int)(layer_size * kwta_frac_hidden));
                applyKWTAKernel<<<BATCH_SIZE, 256, 0, compute_stream>>>(
                    d_neurons, TOTAL_NEURONS, layer_start, layer_size, BATCH_SIZE, k
                );
                CUDA_LAUNCH_CHECK("applyKWTAKernel_H3");
            }
        }
        // 4) Output layer
        {
            const int layer_start = layer_offsets[4];
            const int layer_size  = layer_sizes[4];
            const int total = BATCH_SIZE * layer_size;
            dim3 blocks((total + 511) / 512);
            dim3 threads(512);
            forwardPassLayerKernel<<<blocks, threads, 0, compute_stream>>>(
                d_neurons, d_synapses, TOTAL_NEURONS,
                d_incoming_indices, d_incoming_offsets,
                layer_start, layer_size,
                dt, BATCH_SIZE,
                0, 0.0f
            );
            CUDA_LAUNCH_CHECK("forwardPassLayerKernel_OUT");
        }

        // Optional: clustering for visualization occasionally
        if (epoch % 100 == 0) {
            const int total_neuron_instances = BATCH_SIZE * TOTAL_NEURONS;
            dim3 neuron_blocks((total_neuron_instances + 511) / 512);
            dim3 neuron_threads(512);
            clusterNeuronsKernel<<<neuron_blocks, neuron_threads, 0, compute_stream>>>(
                d_neurons, TOTAL_NEURONS, 1e-8f, dt, BATCH_SIZE
            );
            CUDA_LAUNCH_CHECK("clusterNeuronsKernel");
        }

        bool dbg_optics = (std::getenv("DEBUG_OPTICS") != nullptr);
        if (dbg_optics) {
            computeOutputEnergy<<<(BATCH_SIZE + 511)/512, 512, 0, compute_stream>>>(
                d_neurons, TOTAL_NEURONS, BATCH_SIZE, layer_offsets.back(), OUTPUT_SIZE, d_temp_energy
            );
            CUDA_LAUNCH_CHECK("computeOutputEnergy_pre");
        }

        // Apply output MZI mesh mixing on field amplitudes of the output layer
        applyOutputMZIMesh<<<(BATCH_SIZE + 511)/512, 512, 0, compute_stream>>>(
            d_neurons, TOTAL_NEURONS, BATCH_SIZE, layer_offsets.back(), OUTPUT_SIZE,
            d_mzi_theta, d_mzi_phi, d_mzi_visibility, d_mzi_loss, d_mzi_path_diff, d_mzi_coherence_len,
            d_mzi_stage_offsets, mzi_num_stages, d_rand_states, logit_gain
        );
        CUDA_LAUNCH_CHECK("applyOutputMZIMesh");

        if (dbg_optics) {
            // compute post-mesh energy and print summary
            computeOutputEnergy<<<(BATCH_SIZE + 511)/512, 512, 0, compute_stream>>>(
                d_neurons, TOTAL_NEURONS, BATCH_SIZE, layer_offsets.back(), OUTPUT_SIZE, d_temp_energy
            );
            CUDA_LAUNCH_CHECK("computeOutputEnergy_post");
            std::vector<float> hE(BATCH_SIZE);
            CUDA_CHECK(cudaMemcpy(hE.data(), d_temp_energy, BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
            float meanE = 0.0f, maxE = -1e9f, minE = 1e9f;
            for (int i = 0; i < BATCH_SIZE; ++i) { meanE += hE[i]; if (hE[i]>maxE) maxE=hE[i]; if (hE[i]<minE) minE=hE[i]; }
            meanE /= std::max(1, BATCH_SIZE);
            std::cout << "[DEBUG_OPTICS] Output energy (post-mesh): min=" << minE << ", max=" << maxE << ", mean=" << meanE << std::endl;
        }
    }
    
    // Debug version with diagnostic outputs
    void forward_debug(float dt = 0.01f) {
        const int total_neuron_instances = BATCH_SIZE * TOTAL_NEURONS;
        dim3 neuron_blocks((total_neuron_instances + 255) / 256);
        dim3 neuron_threads(256);
        
        // Check input layer BEFORE forward pass
        debugNeuronValues<<<1, 1, 0, compute_stream>>>(
            d_neurons, 0, INPUT_SIZE, "INPUT_BEFORE");
            
        // Verify input buffer
        verifyInputLoaded<<<1, 1, 0, compute_stream>>>(
            d_batch_inputs, BATCH_SIZE, INPUT_SIZE);
        
        // Run forward pass using the layer-sequential path for consistency
        forward(dt);
        
        // Check each layer AFTER forward pass
        debugNeuronValues<<<1, 1, 0, compute_stream>>>(
            d_neurons, 0, INPUT_SIZE, "INPUT_AFTER");
        CUDA_LAUNCH_CHECK("debugNeuronValues_INPUT_AFTER");
        debugNeuronValues<<<1, 1, 0, compute_stream>>>(
            d_neurons, INPUT_SIZE, HIDDEN1_SIZE, "HIDDEN1");
        CUDA_LAUNCH_CHECK("debugNeuronValues_HIDDEN1");
        debugNeuronValues<<<1, 1, 0, compute_stream>>>(
            d_neurons, INPUT_SIZE + HIDDEN1_SIZE, HIDDEN2_SIZE, "HIDDEN2");
        CUDA_LAUNCH_CHECK("debugNeuronValues_HIDDEN2");
        debugNeuronValues<<<1, 1, 0, compute_stream>>>(
            d_neurons, INPUT_SIZE + HIDDEN1_SIZE + HIDDEN2_SIZE, HIDDEN3_SIZE, "HIDDEN3");
        CUDA_LAUNCH_CHECK("debugNeuronValues_HIDDEN3");
        debugNeuronValues<<<1, 1, 0, compute_stream>>>(
            d_neurons, layer_offsets.back(), OUTPUT_SIZE, "OUTPUT");
        CUDA_LAUNCH_CHECK("debugNeuronValues_OUTPUT");
        
        // Check a sample of synapse weights
        debugSynapseWeights<<<1, 1, 0, compute_stream>>>(
            d_synapses, 0, 100, "Layer0->1");
        CUDA_LAUNCH_CHECK("debugSynapseWeights");
    }
    
    float trainBatch(const std::vector<std::vector<float>>& inputs, 
                    const std::vector<int>& target_labels, 
                    float dt = 0.01f) {
        loadInputBatch(inputs);
        forward(dt);
        // Read output activations for loss/grad computation
        getOutputActivationsKernel<<<(BATCH_SIZE * OUTPUT_SIZE + 511)/512, 512, 0, compute_stream>>>(
            d_neurons, d_output_activations, TOTAL_NEURONS, OUTPUT_SIZE, BATCH_SIZE, layer_offsets.back()
        );
        CUDA_LAUNCH_CHECK("getOutputActivationsKernel");
        std::vector<float> batch_outputs_flat(BATCH_SIZE * OUTPUT_SIZE);
        CUDA_CHECK(cudaMemcpy(batch_outputs_flat.data(), d_output_activations, batch_outputs_flat.size() * sizeof(float), cudaMemcpyDeviceToHost));

        const int actual_bs = static_cast<int>(inputs.size());

        // Cross-entropy loss + gradient wrt logits (here activations)
        float total_loss = 0.0f;
        std::vector<float> output_gradients(BATCH_SIZE * OUTPUT_SIZE, 0.0f);
        for (int b = 0; b < actual_bs; ++b) {
            float max_val = -1e9f;
            for (int i = 0; i < OUTPUT_SIZE; ++i) max_val = fmaxf(max_val, batch_outputs_flat[b*OUTPUT_SIZE + i]);
            float sum_exp = 0.0f;
            for (int i = 0; i < OUTPUT_SIZE; ++i) sum_exp += expf(batch_outputs_flat[b*OUTPUT_SIZE + i] - max_val);
            // probs and gradients
            for (int i = 0; i < OUTPUT_SIZE; ++i) {
                float p = expf(batch_outputs_flat[b*OUTPUT_SIZE + i] - max_val) / (sum_exp + 1e-8f);
                float g = p - ((i == target_labels[b]) ? 1.0f : 0.0f);
                output_gradients[b*OUTPUT_SIZE + i] = g;
            }
            int t = target_labels[b];
            float pt = expf(batch_outputs_flat[b*OUTPUT_SIZE + t] - max_val) / (sum_exp + 1e-8f);
            total_loss -= logf(pt + 1e-8f);
        }

        // Push supervised gradient into output neurons
        float* d_output_gradients = nullptr;
        CUDA_CHECK(cudaMalloc(&d_output_gradients, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_output_gradients, output_gradients.data(), BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        applyOutputGradients<<<(BATCH_SIZE * OUTPUT_SIZE + 511)/512, 512, 0, compute_stream>>>(
            d_neurons, d_output_gradients, TOTAL_NEURONS, OUTPUT_SIZE, actual_bs, layer_offsets.back(), learning_rate
        );
        CUDA_LAUNCH_CHECK("applyOutputGradients");
        CUDA_CHECK(cudaFree(d_output_gradients));

        // Upload labels for output MZI phase learning
        CUDA_CHECK(cudaMemcpyAsync(d_target_labels, target_labels.data(), 
                                  target_labels.size() * sizeof(int), 
                                  cudaMemcpyHostToDevice, compute_stream));

        // Phase learning at output mesh (local, SPSA-free heuristic)
        {
            float phase_lr = mzi_phase_lr;
            int threads = 128;
            int blocks = (mzi_total_pairs + threads - 1) / threads;
            updateMZIPhasesFromTargets<<<blocks, threads, 0, compute_stream>>>(
                d_neurons, TOTAL_NEURONS, BATCH_SIZE, layer_offsets.back(), OUTPUT_SIZE,
                d_target_labels, d_mzi_stage_offsets, mzi_num_stages, d_mzi_theta, phase_lr
            );
            CUDA_LAUNCH_CHECK("updateMZIPhasesFromTargets");
        }

        // Calculate weight updates (averaged per batch inside the kernel)
        dim3 delta_blocks((num_synapses + 511) / 512);
        dim3 delta_threads(512);
        calculateWeightDeltasKernel<<<delta_blocks, delta_threads, 0, compute_stream>>>(
            d_synapses, d_neurons, num_synapses, TOTAL_NEURONS, 
            learning_rate, dt, d_delta_weights, actual_bs
        );
        CUDA_LAUNCH_CHECK("calculateWeightDeltasKernel");

        // Apply weight updates with momentum (no extra averaging here)
        dim3 apply_blocks((num_synapses + 511) / 512);
        dim3 apply_threads(512);
        applyWeightUpdatesKernel<<<apply_blocks, apply_threads, 0, compute_stream>>>(
            d_synapses, d_delta_weights, d_momentum_velocities, num_synapses, momentum
        );
        CUDA_LAUNCH_CHECK("applyWeightUpdatesKernel");

        // Temperature annealing
        const int total_neuron_instances = BATCH_SIZE * TOTAL_NEURONS;
        dim3 neuron_blocks((total_neuron_instances + 255) / 256);
        dim3 neuron_threads(256);
        temperatureAnnealingKernel<<<neuron_blocks, neuron_threads, 0, compute_stream>>>(
            d_neurons, TOTAL_NEURONS, 0.999f, 0.5f, BATCH_SIZE, epoch
        );
        CUDA_LAUNCH_CHECK("temperatureAnnealingKernel");
        
        // Increment step counter, not epoch
        global_step++;
        
        // Adaptive learning rate based on steps not epochs - slower decay
        if (global_step % 100 == 0 && global_step > 0) {
            learning_rate *= 0.99f;  // Slower decay
            learning_rate = fmaxf(1e-4f, learning_rate);  // Higher minimum
        }
        
        // Remove epoch++ from here - will be done in trainer
        return total_loss / std::max(1, actual_bs);
    }
    
    int predict(const std::vector<float>& input) {
        std::vector<std::vector<float>> batch_input(BATCH_SIZE, 
                                                    std::vector<float>(INPUT_SIZE, 0.0f));
        batch_input[0] = input;

        loadInputBatch(batch_input);
        forward();
        
        std::vector<std::vector<float>> batch_output = getOutputBatch();
        
        int prediction = 0;
        float max_activation = batch_output[0][0];
        for (int i = 1; i < OUTPUT_SIZE; i++) {
            if (batch_output[0][i] > max_activation) {
                max_activation = batch_output[0][i];
                prediction = i;
            }
        }
        
        return prediction;
    }

    // Train on binary target using output neuron 0 as logit (BCE with logits)
    float trainBatchBinary(const std::vector<std::vector<float>>& inputs,
                           const std::vector<float>& targets,
                           float dt = 0.01f) {
        loadInputBatch(inputs);
        forward(dt);

        // Gather logits (potentials)
        getOutputActivationsKernel<<<(BATCH_SIZE * OUTPUT_SIZE + 511)/512, 512, 0, compute_stream>>>(
            d_neurons, d_output_activations, TOTAL_NEURONS, OUTPUT_SIZE, BATCH_SIZE, layer_offsets.back()
        );
        CUDA_LAUNCH_CHECK("getOutputActivationsKernel_Binary");
        std::vector<float> batch_logits(BATCH_SIZE * OUTPUT_SIZE);
        CUDA_CHECK(cudaMemcpy(batch_logits.data(), d_output_activations, batch_logits.size() * sizeof(float), cudaMemcpyDeviceToHost));

        const int actual_bs = static_cast<int>(inputs.size());
        std::vector<float> output_gradients(BATCH_SIZE * OUTPUT_SIZE, 0.0f);
        float total_loss = 0.0f;
        for (int b = 0; b < actual_bs; ++b) {
            float logit = batch_logits[b * OUTPUT_SIZE + 0];
            float y = targets[b];
            float p = 1.0f / (1.0f + expf(-logit));
            total_loss += -(y * logf(p + 1e-8f) + (1.0f - y) * logf(1.0f - p + 1e-8f));
            output_gradients[b * OUTPUT_SIZE + 0] = (p - y);
        }

        float* d_output_gradients = nullptr;
        CUDA_CHECK(cudaMalloc(&d_output_gradients, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_output_gradients, output_gradients.data(), BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        applyOutputGradients<<<(BATCH_SIZE * OUTPUT_SIZE + 511)/512, 512, 0, compute_stream>>>(
            d_neurons, d_output_gradients, TOTAL_NEURONS, OUTPUT_SIZE, actual_bs, layer_offsets.back(), learning_rate
        );
        CUDA_LAUNCH_CHECK("applyOutputGradients_Binary");
        CUDA_CHECK(cudaFree(d_output_gradients));

        dim3 delta_blocks((num_synapses + 511) / 512);
        dim3 delta_threads(512);
        calculateWeightDeltasKernel<<<delta_blocks, delta_threads, 0, compute_stream>>>(
            d_synapses, d_neurons, num_synapses, TOTAL_NEURONS, 
            learning_rate, dt, d_delta_weights, actual_bs
        );
        CUDA_LAUNCH_CHECK("calculateWeightDeltasKernel_Binary");

        dim3 apply_blocks((num_synapses + 511) / 512);
        dim3 apply_threads(512);
        applyWeightUpdatesKernel<<<apply_blocks, apply_threads, 0, compute_stream>>>(
            d_synapses, d_delta_weights, d_momentum_velocities, num_synapses, momentum
        );
        CUDA_LAUNCH_CHECK("applyWeightUpdatesKernel_Binary");

        const int total_neuron_instances = BATCH_SIZE * TOTAL_NEURONS;
        dim3 neuron_blocks((total_neuron_instances + 255) / 256);
        dim3 neuron_threads(256);
        temperatureAnnealingKernel<<<neuron_blocks, neuron_threads, 0, compute_stream>>>(
            d_neurons, TOTAL_NEURONS, 0.999f, 0.5f, BATCH_SIZE, epoch
        );
        CUDA_LAUNCH_CHECK("temperatureAnnealingKernel_Binary");

        global_step++;
        if (global_step % 100 == 0 && global_step > 0) {
            learning_rate *= 0.99f;
            learning_rate = fmaxf(1e-4f, learning_rate);
        }

        return total_loss / std::max(1, actual_bs);
    }

    // Convenience: forward on a single 28x28 image; return output activations for sample 0
    std::vector<float> inferSingle(const std::vector<float>& input28x28) {
        std::vector<std::vector<float>> batch(BATCH_SIZE, std::vector<float>(INPUT_SIZE, 0.0f));
        int copy_n = (int)std::min<size_t>(input28x28.size(), (size_t)INPUT_SIZE);
        std::copy(input28x28.begin(), input28x28.begin() + copy_n, batch[0].begin());
        loadInputBatch(batch);
        forward();
        auto outs = getOutputBatch();
        return outs[0];
    }
    
    void printStatus() {
        std::cout << "Epoch: " << epoch 
                  << " | Step: " << global_step
                  << " | LR: " << std::scientific << std::setprecision(3) << learning_rate 
                  << " | Temp: " << std::fixed << std::setprecision(2) << base_temperature 
                  << "K" << std::endl;
    }
    
    void incrementEpoch() {
        epoch++;
    }
};

// ============================================================================ 
// TRAINER
// ============================================================================ 

class Trainer {
private:
    OpticalNeuralNetwork& network;
    KaggleDigitLoader& loader;
    std::vector<float> loss_history;
    std::string ckpt_dir;
    bool save_best;
    
public:
    Trainer(OpticalNeuralNetwork& net, KaggleDigitLoader& data) 
        : network(net), loader(data), save_best(false) {
        if (const char* s = std::getenv("CKPT_DIR")) {
            ckpt_dir = s;
        }
        if (const char* s = std::getenv("SAVE_BEST_CKPT")) {
            try { save_best = std::stoi(s) != 0; } catch (...) { save_best = true; }
        }
    }

    
    void trainEpoch() {
        const auto& train_images = loader.getTrainImages();
        const auto& train_labels = loader.getTrainLabels();
        
        if (train_images.empty()) {
            throw std::runtime_error("No training data loaded");
        }
        
        int total_samples = train_images.size();
        std::vector<int> indices(total_samples);
        std::iota(indices.begin(), indices.end(), 0);
        
        bool no_shuffle = (std::getenv("NO_SHUFFLE") != nullptr);
        if (!no_shuffle) {
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indices.begin(), indices.end(), g);
        }
        
        float total_loss = 0.0f;
        int batches = 0;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        int max_batches = -1;
        if (const char* env_mb = std::getenv("MAX_BATCHES")) {
            try { max_batches = std::stoi(env_mb); } catch (...) { max_batches = -1; }
        }

        for (int i = 0; i < total_samples; i += OpticalNeuralNetwork::BATCH_SIZE) {
            int batch_size = std::min((int)OpticalNeuralNetwork::BATCH_SIZE, total_samples - i);
            
            std::vector<std::vector<float>> batch_images;
            std::vector<int> batch_labels;
            
            for (int j = 0; j < batch_size; ++j) {
                int idx = indices[i + j];
                batch_images.push_back(train_images[idx]);
                batch_labels.push_back(train_labels[idx]);
            }
            
            // Pad batch if needed
            while(batch_images.size() < OpticalNeuralNetwork::BATCH_SIZE) {
                batch_images.push_back(std::vector<float>(784, 0.0f));
                batch_labels.push_back(0);
            }

            float loss = network.trainBatch(batch_images, batch_labels);
            float acc = network.computeBatchAccuracy(batch_labels, batch_size);
            total_loss += loss;
            batches++;

            if (batches % 10 == 0) {
                auto current_time = std::chrono::high_resolution_clock::now();
                float elapsed = std::chrono::duration<float>(current_time - start_time).count();
                float samples_per_sec = (i + batch_size) / elapsed;
                
                std::cout << "\rProgress: " << i + batch_size << "/" << total_samples 
                          << " | Loss: " << std::fixed << std::setprecision(4) 
                          << (total_loss / batches)
                          << " | Acc: " << std::fixed << std::setprecision(4) << acc
                          << " | Speed: " << std::setprecision(1) << samples_per_sec << " sps" 
                          << std::flush;
            }

            if (max_batches > 0 && batches >= max_batches) break;
        }
        
        std::cout << std::endl;
        float avg_loss = total_loss / batches;
        loss_history.push_back(avg_loss);
        
        // Save best checkpoint if requested
        if (save_best && !loss_history.empty()) {
            bool improved = loss_history.size() == 1 || loss_history.back() <= *std::min_element(loss_history.begin(), loss_history.end()-1);
            if (improved && !ckpt_dir.empty()) {
                std::string best_path = ckpt_dir;
                if (best_path.back() == '/' || best_path.back() == '\\') best_path.pop_back();
                network.saveWeights(best_path, "best.bin");
            }
        }

        // CRITICAL: Only increment epoch HERE, not in trainBatch
        network.incrementEpoch();
        
        // Add simple data flow check for debugging
        if (avg_loss > 2.30f) {
            std::cout << "\nWARNING: Loss = " << avg_loss << " (random chance level)" << std::endl;
            std::cout << "Data may not be flowing through network properly" << std::endl;
            
            // Print simple diagnostics without GPU kernels
            auto outputs = network.getOutputBatch();
            std::cout << "Sample output activations: ";
            for (int i = 0; i < std::min(5, (int)outputs[0].size()); i++) {
                std::cout << std::fixed << std::setprecision(4) << outputs[0][i] << " ";
            }
            std::cout << std::endl;
        }
        
        network.printStatus();
        std::cout << "EpochAvgLoss: " << std::fixed << std::setprecision(6) << avg_loss << std::endl;
    }
    
    void generateSubmission(const std::string& filename) {
        const auto& test_images = loader.getTestImages();
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot create submission file");
        }
        file << "ImageId,Label" << std::endl;

        size_t limit = test_images.size();
        if (const char* env_lim = std::getenv("MAX_TEST_SAMPLES")) {
            try { limit = std::min<size_t>(limit, std::stoul(env_lim)); } catch (...) {}
        }

        std::cout << "Generating predictions for " << limit << " samples (batched)..." << std::endl;
        size_t image_id = 1;
        for (size_t i = 0; i < limit; i += OpticalNeuralNetwork::BATCH_SIZE) {
            size_t chunk = std::min<size_t>(OpticalNeuralNetwork::BATCH_SIZE, limit - i);
            std::vector<std::vector<float>> batch(OpticalNeuralNetwork::BATCH_SIZE, std::vector<float>(784, 0.0f));
            for (size_t b = 0; b < chunk; ++b) batch[b] = test_images[i + b];

            network.loadInputBatch(batch);
            network.forward();
            auto outs = network.getOutputBatch();

            for (size_t b = 0; b < chunk; ++b) {
                int best_i = 0; float best_v = outs[b][0];
                for (int k = 1; k < 10; ++k) if (outs[b][k] > best_v) { best_v = outs[b][k]; best_i = k; }
                file << image_id++ << "," << best_i << std::endl;
            }

            if (image_id % 1000 < OpticalNeuralNetwork::BATCH_SIZE) {
                std::cout << "\rProgress: " << (image_id - 1) << "/" << limit << std::flush;
            }
        }
        std::cout << "\nSubmission saved to: " << filename << std::endl;
    }
    
    bool isImproving() const {
        if (loss_history.size() < 3) return true;
        
        // Check if loss is decreasing
        size_t n = loss_history.size();
        return loss_history[n-1] < loss_history[n-2] * 1.01f;
    }
};

// ============================================================================ 
// MAIN
// ============================================================================ 

int main(int argc, char** argv) {
    std::cout << "==================================================\n";
    std::cout << "  OPTICAL NEURAL NETWORK - RSNA / DIGITS  v2.0   \n";
    std::cout << "==================================================\n";
    std::cout << "Architecture: Interferometric Deep Learning\n";
    std::cout << "Physics: Mach-Zehnder Networks with Coherent Interference\n";
    std::cout << "Learning: Enhanced Hebbian with STDP\n";
    std::cout << "==================================================\n\n";
    
    try {
        // Auto-config: if a RSNA MIPs CSV exists and no explicit mode set, default to RSNA training.
        auto getenv_s = [](const char* k) -> std::string { const char* v = std::getenv(k); return v ? std::string(v) : std::string(); };
        auto ensure_env = [](const char* k, const char* val){ if (!std::getenv(k)) {
#if defined(_WIN32)
            _putenv_s(k, val);
#else
            setenv(k, val, 1);
#endif
        }};
        bool explicit_mode = (std::getenv("RSNA_INFER") != nullptr) || (std::getenv("RSNA_TRAIN_MIPS") != nullptr);
        if (!explicit_mode) {
            namespace fs = std::filesystem;
            // Solo autoconfig con dataset completo si existe en raíz
            if (fs::exists("train_mips_full.csv")) {
                // Enable non-conventional features by default
                ensure_env("USE_MAX_PLUS", "1");
                ensure_env("USE_HADAMARD", "1");
                ensure_env("USE_KWTA", "1");
                ensure_env("KWTA_FRAC", "0.10");
                ensure_env("LI_ALPHA", "0.05");
                ensure_env("DISABLE_EARLY_STOP", "1");
                // Checkpoints
                ensure_env("CKPT_DIR", "ckpt_full");
                // Default to multi-expert training by default
                ensure_env("MULTIEXPERT_4", "1");
                ensure_env("CKPT_DIR_FRONT", "ckpt_front");
                ensure_env("CKPT_DIR_BACK",  "ckpt_back");
                ensure_env("CKPT_DIR_LEFT",  "ckpt_left");
                ensure_env("CKPT_DIR_RIGHT", "ckpt_right");
                // Hook RSNA training mode automatically
                ensure_env("RSNA_TRAIN_MIPS", "1");
                ensure_env("RSNA_MIPS_CSV", "train_mips_full.csv");
                std::cout << "[AUTO] Detectado train_mips_full.csv -> modo RSNA activado" << std::endl;
            }
        }
        // RSNA CLI quick path: if RSNA_INFER is set, run 4-view inference and exit.
        if (std::getenv("RSNA_INFER") != nullptr) {
            auto getenv_s = [](const char* k) -> std::string { const char* v = std::getenv(k); return v ? std::string(v) : std::string(); };
            std::string f_front = getenv_s("RSNA_FRONT");
            std::string f_back  = getenv_s("RSNA_BACK");
            std::string f_left  = getenv_s("RSNA_LEFT");
            std::string f_right = getenv_s("RSNA_RIGHT");
            if (f_front.empty() || f_back.empty() || f_left.empty() || f_right.empty()) {
                std::cerr << "RSNA_INFER set, but one of RSNA_FRONT/RSNA_BACK/RSNA_LEFT/RSNA_RIGHT is missing" << std::endl;
                return 2;
            }

            auto infer_view = [&](const std::string& img_path, const std::string& ckpt) -> float {
                int w=0,h=0; std::vector<float> img;
                if (!load_pgm_as_gray(img_path, img, w, h)) {
                    std::cerr << "Failed to load image: " << img_path << std::endl; return 0.5f;
                }
                auto in28 = formalize_and_resize_to_28x28(img, w, h);
                OpticalNeuralNetwork net;
                if (!ckpt.empty()) net.loadWeights(ckpt);
                auto out = net.inferSingle(in28);
                // Map tanh activations to [0,1] via max
                float best = out.empty() ? 0.0f : out[0];
                for (size_t i = 1; i < out.size(); ++i) if (out[i] > best) best = out[i];
                float p = 0.5f * (best + 1.0f);
                if (p < 0.0f) p = 0.0f; if (p > 1.0f) p = 1.0f;
                return p;
            };

            std::string ck_router = getenv_s("CKPT_ROUTER"); // optional (not used here yet)
            std::string ck_front  = getenv_s("CKPT_FRONT");
            std::string ck_back   = getenv_s("CKPT_BACK");
            std::string ck_left   = getenv_s("CKPT_LEFT");
            std::string ck_right  = getenv_s("CKPT_RIGHT");

            float p_front = infer_view(f_front, ck_front);
            float p_back  = infer_view(f_back,  ck_back);
            float p_left  = infer_view(f_left,  ck_left);
            float p_right = infer_view(f_right, ck_right);

            float present_prob = (p_front + p_back + p_left + p_right) * 0.25f;
            // Build 15 outputs: 14 arteries + global
            std::vector<float> probs(15, present_prob);
            // Print as JSON line
            std::cout << "{\"probs\":[";
            for (int i = 0; i < 15; ++i) {
                if (i) std::cout << ",";
                std::cout << std::fixed << std::setprecision(6) << probs[i];
            }
            std::cout << "]}" << std::endl;
            return 0;
        }

        // RSNA TRAIN from MIPs CSV: expects RSNA_TRAIN_MIPS=1 and RSNA_MIPS_CSV pointing to CSV with columns:
        // SeriesInstanceUID,label,front,back,left,right  where label in {0,1}, paths are PGM files.
        if (std::getenv("RSNA_TRAIN_MIPS") != nullptr) {
            auto getenv_s = [](const char* k) -> std::string { const char* v = std::getenv(k); return v ? std::string(v) : std::string(); };
            std::string csv_path = getenv_s("RSNA_MIPS_CSV");
            if (csv_path.empty()) {
                std::cerr << "RSNA_TRAIN_MIPS=1 requiere RSNA_MIPS_CSV con ruta al CSV" << std::endl;
                return 2;
            }

            struct Sample { std::string id; int label; std::string f,b,l,r; };
            std::vector<Sample> samples;
            {
                std::ifstream f(csv_path);
                if (!f.is_open()) { std::cerr << "No se puede abrir: " << csv_path << std::endl; return 2; }
                std::string line; std::getline(f, line); // header
                while (std::getline(f, line)) {
                    if (line.empty()) continue;
                    printf("DEBUG_LINE: %s\n", line.c_str());
                    std::stringstream ss(line);
                    std::string id,label_s,fp1,fp2,fp3,fp4;
                    std::getline(ss, id, ',');
                    std::getline(ss, label_s, ',');
                    std::getline(ss, fp1, ',');
                    std::getline(ss, fp2, ',');
                    std::getline(ss, fp3, ',');
                    std::getline(ss, fp4, ',');
                    printf("DEBUG_PATHS: front='%s', back='%s', left='%s', right='%s'\n", fp1.c_str(), fp2.c_str(), fp3.c_str(), fp4.c_str());
                    if (id.size()==0) continue;
                    Sample s{ id, 0, fp1, fp2, fp3, fp4 };
                    try { s.label = std::stoi(label_s); } catch(...) { s.label = 0; }
                    samples.push_back(std::move(s));
                }
            }
            if (samples.empty()) { std::cerr << "CSV sin muestras: " << csv_path << std::endl; return 2; }
            std::cout << "[RSNA] Series en CSV: " << samples.size() << ", fichero: " << csv_path << std::endl;

            bool multi = (std::getenv("MULTIEXPERT_4") != nullptr);
            if (!multi) {
                std::cout << "Cargando red (fused MIPs)..." << std::endl;
                OpticalNeuralNetwork net;
                if (const char* lw = std::getenv("LOAD_WEIGHTS")) { net.loadWeights(lw); }
                int max_epochs = 1000; if (const char* s = std::getenv("MAX_EPOCHS")) { try { max_epochs = std::stoi(s); } catch(...){} }
                int max_batches = -1; if (const char* s = std::getenv("MAX_BATCHES")) { try { max_batches = std::stoi(s); } catch(...){} }
                int bs = OpticalNeuralNetwork::BATCH_SIZE;
                size_t idx = 0;
                const size_t total_samples = samples.size();
                for (int ep = 0; ep < max_epochs; ++ep) {
                    // Optional shuffle per epoch
                    if (std::getenv("RSNA_NO_SHUFFLE") == nullptr) {
                        std::shuffle(samples.begin(), samples.end(), std::mt19937(1337 + ep));
                    }
                    std::cout << "\n[RSNA TRAIN] Epoch " << (ep+1) << "/" << max_epochs << std::endl;
                    float epoch_loss = 0.0f; int batches = 0; size_t processed = 0;
                    auto t_epoch0 = std::chrono::high_resolution_clock::now();
                    double io_sec = 0.0, gpu_sec = 0.0;
                    idx = 0;
                    while (idx < samples.size()) {
                        auto t_io0 = std::chrono::high_resolution_clock::now();
                        std::vector<std::vector<float>> inputs;
                        std::vector<float> targets;
                        for (int b = 0; b < bs && idx < samples.size(); ++b, ++idx) {
                            const auto& s = samples[idx];
                            int wF=0,hF=0,wB=0,hB=0,wL=0,hL=0,wR=0,hR=0; std::vector<float> imF, imB, imL, imR;
                            if (!load_pgm_as_gray(s.f, imF, wF, hF)) imF.assign(28*28,0.0f);
                            if (!load_pgm_as_gray(s.b, imB, wB, hB)) imB.assign(28*28,0.0f);
                            if (!load_pgm_as_gray(s.l, imL, wL, hL)) imL.assign(28*28,0.0f);
                            if (!load_pgm_as_gray(s.r, imR, wR, hR)) imR.assign(28*28,0.0f);
                            auto F = resize_nn_to_28x28(imF, wF, hF);
                            auto B = resize_nn_to_28x28(imB, wB, hB);
                            auto L = resize_nn_to_28x28(imL, wL, hL);
                            auto R = resize_nn_to_28x28(imR, wR, hR);
                            std::vector<float> fused(28*28, 0.0f);
                            for (int i = 0; i < 28*28; ++i) fused[i] = std::max(std::max(F[i], B[i]), std::max(L[i], R[i]));
                            inputs.push_back(std::move(fused));
                            targets.push_back((float)(s.label != 0));
                            if (std::getenv("RSNA_DEBUG_SAMPLES") != nullptr && b < 2 && ep == 0) {
                                namespace fs = std::filesystem;
                                std::cout << "  sample id=" << s.id << " front=" << fs::path(s.f).filename().string() << std::endl;
                            }
                        }
                        auto t_io1 = std::chrono::high_resolution_clock::now();
                        if (inputs.empty()) break;
                        processed += inputs.size();
                        double io_dur = std::chrono::duration<double>(t_io1 - t_io0).count();
                        io_sec += io_dur;
                        auto t_gpu0 = std::chrono::high_resolution_clock::now();
                        float loss = net.trainBatchBinary(inputs, targets);
                        auto t_gpu1 = std::chrono::high_resolution_clock::now();
                        gpu_sec += std::chrono::duration<double>(t_gpu1 - t_gpu0).count();
                        epoch_loss += loss; batches++;
                        if (batches % 50 == 0) {
                            std::cout << "  [progress] processed " << processed << "/" << total_samples
                                      << ", avg_loss=" << (epoch_loss / std::max(1, batches))
                                      << ", io=" << std::fixed << std::setprecision(2) << io_sec
                                      << "s, gpu=" << gpu_sec << "s" << std::endl;
                        }
                        if (max_batches > 0 && batches >= max_batches) break;
                    }
                    auto t_epoch1 = std::chrono::high_resolution_clock::now();
                    double epoch_sec = std::chrono::duration<double>(t_epoch1 - t_epoch0).count();
                    std::cout << "[RSNA TRAIN] samples=" << processed << "/" << total_samples
                              << ", batches=" << batches
                              << ", avg loss=" << (epoch_loss / std::max(1, batches))
                              << ", io=" << std::fixed << std::setprecision(2) << io_sec
                              << "s, gpu=" << gpu_sec << "s, epoch=" << epoch_sec << "s" << std::endl;
                    net.incrementEpoch();
                    if (const char* ck = std::getenv("CKPT_DIR")) {
                        std::string dir = ck; if (dir.back()=='/'||dir.back()=='\\') dir.pop_back();
                        net.saveWeights(dir, (std::string("epoch_")+std::to_string(ep+1)+".bin"));
                    }
                }
                return 0;
            } else {
                std::cout << "Cargando 4 expertos (front/back/left/right)..." << std::endl;
                OpticalNeuralNetwork netF, netB, netL, netR;
                std::string lf = getenv_s("LOAD_FRONT"), lb = getenv_s("LOAD_BACK"), ll = getenv_s("LOAD_LEFT"), lr = getenv_s("LOAD_RIGHT");
                if (!lf.empty()) netF.loadWeights(lf); if (!lb.empty()) netB.loadWeights(lb); if (!ll.empty()) netL.loadWeights(ll); if (!lr.empty()) netR.loadWeights(lr);
                std::string ckF = getenv_s("CKPT_DIR_FRONT"); if (ckF.empty()) ckF = "ckpt_front";
                std::string ckB = getenv_s("CKPT_DIR_BACK");  if (ckB.empty()) ckB = "ckpt_back";
                std::string ckL = getenv_s("CKPT_DIR_LEFT");  if (ckL.empty()) ckL = "ckpt_left";
                std::string ckR = getenv_s("CKPT_DIR_RIGHT"); if (ckR.empty()) ckR = "ckpt_right";
                int max_epochs = 1000; if (const char* s = std::getenv("MAX_EPOCHS")) { try { max_epochs = std::stoi(s); } catch(...){} }
                int max_batches = -1; if (const char* s = std::getenv("MAX_BATCHES")) { try { max_batches = std::stoi(s); } catch(...){} }
                int bs = OpticalNeuralNetwork::BATCH_SIZE;
                size_t idx = 0;
                for (int ep = 0; ep < max_epochs; ++ep) {
                    if (std::getenv("RSNA_NO_SHUFFLE") == nullptr) {
                        std::shuffle(samples.begin(), samples.end(), std::mt19937(7331 + ep));
                    }
                    std::cout << "\n[RSNA TRAIN 4X] Epoch " << (ep+1) << "/" << max_epochs << std::endl;
                    float epoch_loss = 0.0f; int batches = 0; size_t processed = 0; const size_t total_samples = samples.size();
                    auto t_epoch0 = std::chrono::high_resolution_clock::now();
                    double io_sec = 0.0, gpu_sec = 0.0;
                    idx = 0;
                    while (idx < samples.size()) {
                        auto t_io0 = std::chrono::high_resolution_clock::now();
                        std::vector<std::vector<float>> inF, inB, inL, inR;
                        std::vector<float> targets;
                        for (int b = 0; b < bs && idx < samples.size(); ++b, ++idx) {
                            const auto& s = samples[idx];
                            int wF=0,hF=0,wB=0,hB=0,wL=0,hL=0,wR=0,hR=0; std::vector<float> imF, imB, imL, imR;
                            bool ok = true;
                            std::string pf = trim_and_normalize_path(s.f);
                            std::string pb = trim_and_normalize_path(s.b);
                            std::string pl = trim_and_normalize_path(s.l);
                            std::string pr = trim_and_normalize_path(s.r);
                            if (!load_pgm_as_gray(pf, imF, wF, hF)) { std::cerr << "[RSNA] Error cargando " << pf << "\n"; ok=false; }
                            if (!load_pgm_as_gray(pb, imB, wB, hB)) { std::cerr << "[RSNA] Error cargando " << pb << "\n"; ok=false; }
                            if (!load_pgm_as_gray(pl, imL, wL, hL)) { std::cerr << "[RSNA] Error cargando " << pl << "\n"; ok=false; }
                            if (!load_pgm_as_gray(pr, imR, wR, hR)) { std::cerr << "[RSNA] Error cargando " << pr << "\n"; ok=false; }
                            if (!ok) { continue; }
                            inF.push_back(formalize_and_resize_to_28x28(imF, wF, hF));
                            inB.push_back(formalize_and_resize_to_28x28(imB, wB, hB));
                            inL.push_back(formalize_and_resize_to_28x28(imL, wL, hL));
                            inR.push_back(formalize_and_resize_to_28x28(imR, wR, hR));
                            targets.push_back((float)(s.label != 0));
                            if (std::getenv("RSNA_DEBUG_SAMPLES") != nullptr && b < 2 && ep == 0) {
                                namespace fs = std::filesystem;
                                std::cout << "  sample id=" << s.id << " front=" << fs::path(s.f).filename().string() << std::endl;
                            }
                        }
                        auto t_io1 = std::chrono::high_resolution_clock::now();
                        if (targets.empty()) break;
                        processed += targets.size();
                        io_sec += std::chrono::duration<double>(t_io1 - t_io0).count();
                        auto t_gpu0 = std::chrono::high_resolution_clock::now();
                        float lf1 = netF.trainBatchBinary(inF, targets);
                        float lb1 = netB.trainBatchBinary(inB, targets);
                        float ll1 = netL.trainBatchBinary(inL, targets);
                        float lr1 = netR.trainBatchBinary(inR, targets);
                        auto t_gpu1 = std::chrono::high_resolution_clock::now();
                        gpu_sec += std::chrono::duration<double>(t_gpu1 - t_gpu0).count();
                        float loss = 0.25f * (lf1 + lb1 + ll1 + lr1);
                        epoch_loss += loss; batches++;
                        if (batches % 50 == 0) {
                            std::cout << "  [progress] processed " << processed << "/" << total_samples
                                      << ", avg_loss=" << (epoch_loss / std::max(1, batches))
                                      << ", io=" << std::fixed << std::setprecision(2) << io_sec
                                      << "s, gpu=" << gpu_sec << "s" << std::endl;
                        }
                        if (max_batches > 0 && batches >= max_batches) break;
                    }
                    auto t_epoch1 = std::chrono::high_resolution_clock::now();
                    double epoch_sec = std::chrono::duration<double>(t_epoch1 - t_epoch0).count();
                    std::cout << "[RSNA TRAIN 4X] samples=" << processed << "/" << total_samples
                              << ", batches=" << batches
                              << ", avg loss=" << (epoch_loss / std::max(1, batches))
                              << ", io=" << std::fixed << std::setprecision(2) << io_sec
                              << "s, gpu=" << gpu_sec << "s, epoch=" << epoch_sec << "s" << std::endl;
                    if (processed == 0) {
                        std::cerr << "[RSNA] Ninguna muestra valida procesada en la epoca. Aborting. Verifique rutas en RSNA_MIPS_CSV." << std::endl;
                        return 2;
                    }
                    netF.incrementEpoch(); netB.incrementEpoch(); netL.incrementEpoch(); netR.incrementEpoch();
                    netF.saveWeights(ckF, (std::string("epoch_") + std::to_string(ep+1) + ".bin"));
                    netB.saveWeights(ckB, (std::string("epoch_") + std::to_string(ep+1) + ".bin"));
                    netL.saveWeights(ckL, (std::string("epoch_") + std::to_string(ep+1) + ".bin"));
                    netR.saveWeights(ckR, (std::string("epoch_") + std::to_string(ep+1) + ".bin"));
                }
                return 0;
            }
        }
        // If we reach here, no RSNA mode has been selected. To avoid accidental fallback
        // to the legacy MNIST/Digits path, require explicit opt-in via LEGACY_DIGITS=1.
        if (std::getenv("LEGACY_DIGITS") == nullptr) {
            std::cerr << "\n==================================================\n";
            std::cerr << "ERROR: Modo RSNA no configurado correctamente\n";
            std::cerr << "==================================================\n";
            std::cerr << "Para entrenar con el dataset RSNA:\n";
            std::cerr << "1. Genere el archivo 'train_mips.csv' con las rutas de las imágenes MIP\n";
            std::cerr << "2. O establezca variables de entorno:\n";
            std::cerr << "   RSNA_TRAIN_MIPS=1\n";
            std::cerr << "   RSNA_MIPS_CSV=ruta/al/archivo.csv\n";
            std::cerr << "\nPara inferencia RSNA:\n";
            std::cerr << "   RSNA_INFER=1\n";
            std::cerr << "   RSNA_FRONT=ruta/imagen_front.pgm\n";
            std::cerr << "   RSNA_BACK=ruta/imagen_back.pgm\n";
            std::cerr << "   RSNA_LEFT=ruta/imagen_left.pgm\n";
            std::cerr << "   RSNA_RIGHT=ruta/imagen_right.pgm\n";
            std::cerr << "\nPara usar el flujo legado de dígitos, establezca LEGACY_DIGITS=1\n";
            std::cerr << "==================================================\n" << std::endl;
            return 2;
        }
        // Check CUDA
        int device_count;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count == 0) {
            throw std::runtime_error("No CUDA devices found");
        }
        
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        std::cout << "GPU: " << prop.name << std::endl;
        std::cout << "Compute: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "Memory: " << (prop.totalGlobalMem >> 30) << " GB\n\n";
        
        // Load data
        std::cout << "Loading dataset...\n";
        KaggleDigitLoader loader;
        
        if (!loader.loadTrainData("train.csv")) {
            throw std::runtime_error("Cannot load train.csv");
        }
        
        if (!loader.loadTestData("test.csv")) {
            throw std::runtime_error("Cannot load test.csv");
        }
        
        // Initialize network
        std::cout << "\nInitializing optical network...\n";
        OpticalNeuralNetwork network;
        if (const char* s = std::getenv("LOAD_WEIGHTS")) {
            std::cout << "Attempting to load checkpoint: " << s << std::endl;
            if (!network.loadWeights(s)) {
                std::cout << "Failed to load weights from '" << s << "'. Continuing with initialized weights.\n";
            } else {
                std::cout << "Loaded weights from checkpoint.\n";
            }
        }
        
        // Initialize trainer
        Trainer trainer(network, loader);
        
        // Training (can be skipped with PREDICT_ONLY)
        int max_epochs = 999;
        if (const char* env_me = std::getenv("MAX_EPOCHS")) {
            try { max_epochs = std::stoi(env_me); } catch (...) {}
        }
        std::cout << "\n==================================================\n";
        std::cout << "TRAINING PHASE\n";
        std::cout << "==================================================\n";
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        if (std::getenv("PREDICT_ONLY") == nullptr) {
            bool disable_es = (std::getenv("DISABLE_EARLY_STOP") != nullptr);
            for (int epoch = 0; epoch < max_epochs; epoch++) {
                std::cout << "\n--- Epoch " << (epoch + 1) << "/" << max_epochs << " ---\n";
                trainer.trainEpoch();
                if (!disable_es && !trainer.isImproving() && epoch > 5) {
                    std::cout << "Early stopping - loss not improving\n";
                    break;
                }
            }
        } else {
            std::cout << "\n[PREDICT_ONLY] Skipping training as requested.\n";
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        float training_time = std::chrono::duration<float>(end_time - start_time).count();
        
        std::cout << "\n==================================================\n";
        std::cout << "Training completed in " << std::fixed << std::setprecision(1) 
                  << training_time / 60.0f << " minutes\n";
        std::cout << "==================================================\n\n";
        
        // Generate submission (can skip via env)
        if (std::getenv("SKIP_SUBMISSION") == nullptr) {
            trainer.generateSubmission("submission.csv");
        } else {
            std::cout << "\nSkipping submission generation (SKIP_SUBMISSION set)." << std::endl;
        }
        
        std::cout << "\n==================================================\n";
        std::cout << "SUCCESS! Submission ready for Kaggle\n";
        std::cout << "Upload submission.csv to compete\n";
        std::cout << "==================================================\n";
        
    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        std::cerr << "\nPlease ensure:\n";
        std::cerr << "1. CUDA-capable GPU is available\n";
        std::cerr << "2. train.csv and test.csv are in current directory\n";
        std::cerr << "3. Files downloaded from Kaggle Digit Recognizer competition\n";
        return 1;
    }
    
    return 0;
}
