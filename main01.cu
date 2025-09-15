/*
 * NEBULA NUMBER EMERGENT - Kaggle Digit Recognizer Edition

 * Author: Francisco Angulo de Lafuente (Model Made in 2023)

 * Scientific Foundation:
 * - Mach-Zehnder interferometer networks (Shen et al. 2017)
 * - Optical matrix multiplication (Feldmann et al. 2019)
 * - Photonic neural computing (Lin et al. 2018)
 * 
 * Target: Kaggle Digit Recognizer Competition
 * - Train on 42,000 labeled samples
 * - Predict 28,000 test samples
 * - Output: submission.csv for Kaggle
 * 
 * Architecture: 784→512→256→128→10 optical neurons
 * Learning: Hebbian plasticity + physical clustering
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
// DATA LOADER
// ============================================================================ 

class KaggleDigitLoader {
private:
    std::vector<std::vector<float>> train_images;
    std::vector<int> train_labels;
    std::vector<std::vector<float>> test_images;
    
public:
    bool loadTrainData(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Cannot open train file: " << filepath << std::endl;
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
            
            std::getline(ss, cell, ',');
            int label = std::stoi(cell);
            train_labels.push_back(label);
            
            std::vector<float> image(784);
            for (int i = 0; i < 784; i++) {
                std::getline(ss, cell, ',');
                image[i] = std::stof(cell) / 255.0f;
            }
            train_images.push_back(image);
            if (train_images.size() >= max_samples) break;
        }
        
        file.close();
        std::cout << "Loaded " << train_images.size() << " training samples" << std::endl;
        return true;
    }
    
    bool loadTestData(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Cannot open test file: " << filepath << std::endl;
            return false;
        }
        
        std::string line;
        std::getline(file, line); // Skip header
        
        size_t max_samples = SIZE_MAX;
        if (const char* env_ts = std::getenv("MAX_TEST_SAMPLES")) {
            try { max_samples = std::stoul(env_ts); } catch (...) { max_samples = SIZE_MAX; }
        }
        
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            
            std::vector<float> image(784);
            for (int i = 0; i < 784; i++) {
                std::getline(ss, cell, ',');
                image[i] = std::stof(cell) / 255.0f;
            }
            test_images.push_back(image);
            if (test_images.size() >= max_samples) break;
        }
        
        file.close();
        std::cout << "Loaded " << test_images.size() << " test samples" << std::endl;
        return true;
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
                                       float dt, int batch_size) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sample_idx = global_idx / layer_size;
    int idx_in_layer = global_idx % layer_size;
    if (sample_idx >= batch_size) return;

    int neuron_idx = layer_start + idx_in_layer;

    OpticalNeuron* neurons = all_neurons_batch + sample_idx * num_neurons;
    OpticalNeuron& neuron = neurons[neuron_idx];

    // Reset and accumulate only from incoming synapses
    neuron.potential = 0.0f;
    Complex total_field(0.0f, 0.0f);
    int start = incoming_offsets[neuron_idx];
    int end   = incoming_offsets[neuron_idx + 1];
    int connections = 0;
    for (int idx = start; idx < end; ++idx) {
        int s = incoming_indices[idx];
        const OpticalSynapse& syn = synapses[s];
        const OpticalNeuron& pre = neurons[syn.pre_neuron_id];
        float signal_strength = pre.activation * syn.weight;
        neuron.potential += signal_strength;
        Complex weighted_field = pre.field_amplitude * syn.weight;
        total_field = total_field + weighted_field;
        connections++;
    }

    if (connections > 0) {
        float norm_factor = rsqrtf((float)connections + 1e-3f); // 1/sqrt
        // Normalize potential to avoid activation saturation with large fan-in
        neuron.potential *= norm_factor;
        neuron.activation = tanhf(neuron.potential);
        neuron.field_amplitude = total_field * norm_factor;
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
    static constexpr int BATCH_SIZE = 32;

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
        mzi_phase_lr(0.002f) {
        
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
        dim3 blocks((total_threads + 255) / 256);
        dim3 threads(256);
        
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
            dim3 blocks((total + 255) / 256);
            dim3 threads(256);
            forwardPassLayerKernel<<<blocks, threads, 0, compute_stream>>>(
                d_neurons, d_synapses, TOTAL_NEURONS,
                d_incoming_indices, d_incoming_offsets,
                layer_start, layer_size,
                dt, BATCH_SIZE
            );
            CUDA_LAUNCH_CHECK("forwardPassLayerKernel_H1");
        }
        // 2) Hidden2
        {
            const int layer_start = layer_offsets[2];
            const int layer_size  = layer_sizes[2];
            const int total = BATCH_SIZE * layer_size;
            dim3 blocks((total + 255) / 256);
            dim3 threads(256);
            forwardPassLayerKernel<<<blocks, threads, 0, compute_stream>>>(
                d_neurons, d_synapses, TOTAL_NEURONS,
                d_incoming_indices, d_incoming_offsets,
                layer_start, layer_size,
                dt, BATCH_SIZE
            );
            CUDA_LAUNCH_CHECK("forwardPassLayerKernel_H2");
        }
        // 3) Hidden3
        {
            const int layer_start = layer_offsets[3];
            const int layer_size  = layer_sizes[3];
            const int total = BATCH_SIZE * layer_size;
            dim3 blocks((total + 255) / 256);
            dim3 threads(256);
            forwardPassLayerKernel<<<blocks, threads, 0, compute_stream>>>(
                d_neurons, d_synapses, TOTAL_NEURONS,
                d_incoming_indices, d_incoming_offsets,
                layer_start, layer_size,
                dt, BATCH_SIZE
            );
            CUDA_LAUNCH_CHECK("forwardPassLayerKernel_H3");
        }
        // 4) Output layer
        {
            const int layer_start = layer_offsets[4];
            const int layer_size  = layer_sizes[4];
            const int total = BATCH_SIZE * layer_size;
            dim3 blocks((total + 255) / 256);
            dim3 threads(256);
            forwardPassLayerKernel<<<blocks, threads, 0, compute_stream>>>(
                d_neurons, d_synapses, TOTAL_NEURONS,
                d_incoming_indices, d_incoming_offsets,
                layer_start, layer_size,
                dt, BATCH_SIZE
            );
            CUDA_LAUNCH_CHECK("forwardPassLayerKernel_OUT");
        }

        // Optional: clustering for visualization occasionally
        if (epoch % 100 == 0) {
            const int total_neuron_instances = BATCH_SIZE * TOTAL_NEURONS;
            dim3 neuron_blocks((total_neuron_instances + 255) / 256);
            dim3 neuron_threads(256);
            clusterNeuronsKernel<<<neuron_blocks, neuron_threads, 0, compute_stream>>>(
                d_neurons, TOTAL_NEURONS, 1e-8f, dt, BATCH_SIZE
            );
            CUDA_LAUNCH_CHECK("clusterNeuronsKernel");
        }

        bool dbg_optics = (std::getenv("DEBUG_OPTICS") != nullptr);
        if (dbg_optics) {
            computeOutputEnergy<<<(BATCH_SIZE + 255)/256, 256, 0, compute_stream>>>(
                d_neurons, TOTAL_NEURONS, BATCH_SIZE, layer_offsets.back(), OUTPUT_SIZE, d_temp_energy
            );
            CUDA_LAUNCH_CHECK("computeOutputEnergy_pre");
        }

        // Apply output MZI mesh mixing on field amplitudes of the output layer
        applyOutputMZIMesh<<<(BATCH_SIZE + 255)/256, 256, 0, compute_stream>>>(
            d_neurons, TOTAL_NEURONS, BATCH_SIZE, layer_offsets.back(), OUTPUT_SIZE,
            d_mzi_theta, d_mzi_phi, d_mzi_visibility, d_mzi_loss, d_mzi_path_diff, d_mzi_coherence_len,
            d_mzi_stage_offsets, mzi_num_stages, d_rand_states, logit_gain
        );
        CUDA_LAUNCH_CHECK("applyOutputMZIMesh");

        if (dbg_optics) {
            // compute post-mesh energy and print summary
            computeOutputEnergy<<<(BATCH_SIZE + 255)/256, 256, 0, compute_stream>>>(
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
        getOutputActivationsKernel<<<(BATCH_SIZE * OUTPUT_SIZE + 255)/256, 256, 0, compute_stream>>>(
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
        applyOutputGradients<<<(BATCH_SIZE * OUTPUT_SIZE + 255)/256, 256, 0, compute_stream>>>(
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
        dim3 delta_blocks((num_synapses + 255) / 256);
        dim3 delta_threads(256);
        calculateWeightDeltasKernel<<<delta_blocks, delta_threads, 0, compute_stream>>>(
            d_synapses, d_neurons, num_synapses, TOTAL_NEURONS, 
            learning_rate, dt, d_delta_weights, actual_bs
        );
        CUDA_LAUNCH_CHECK("calculateWeightDeltasKernel");

        // Apply weight updates with momentum (no extra averaging here)
        dim3 apply_blocks((num_synapses + 255) / 256);
        dim3 apply_threads(256);
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
    
public:
    Trainer(OpticalNeuralNetwork& net, KaggleDigitLoader& data) 
        : network(net), loader(data) {}
    
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
    std::cout << "  OPTICAL NEURAL NETWORK - DIGIT RECOGNIZER v2.0  \n";
    std::cout << "==================================================\n";
    std::cout << "Architecture: Interferometric Deep Learning\n";
    std::cout << "Physics: Mach-Zehnder Networks with Coherent Interference\n";
    std::cout << "Learning: Enhanced Hebbian with STDP\n";
    std::cout << "==================================================\n\n";
    
    try {
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
        
        // Initialize trainer
        Trainer trainer(network, loader);
        
        // Training (can be skipped with PREDICT_ONLY)
        int max_epochs = 99;
        if (const char* env_me = std::getenv("MAX_EPOCHS")) {
            try { max_epochs = std::stoi(env_me); } catch (...) {}
        }
        std::cout << "\n==================================================\n";
        std::cout << "TRAINING PHASE\n";
        std::cout << "==================================================\n";
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        if (std::getenv("PREDICT_ONLY") == nullptr) {
            for (int epoch = 0; epoch < max_epochs; epoch++) {
                std::cout << "\n--- Epoch " << (epoch + 1) << "/" << max_epochs << " ---\n";
                trainer.trainEpoch();
                if (!trainer.isImproving() && epoch > 5) {
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
