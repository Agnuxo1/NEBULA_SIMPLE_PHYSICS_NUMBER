cmake_minimum_required(VERSION 3.10)
project(optical_mnist)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(optical_nn
    src/data/mnist_loader.cpp
    src/optical/optical_network.cpp)

target_include_directories(optical_nn PUBLIC src)

add_executable(train src/train.cpp)
target_link_libraries(train optical_nn)

add_executable(infer src/infer.cpp)
target_link_libraries(infer optical_nn)

add_executable(generate_submission src/tools/generate_submission.cpp)
target_link_libraries(generate_submission optical_nn)

enable_testing()
add_executable(test_loader tests/test_loader.cpp)
target_link_libraries(test_loader optical_nn)
add_test(NAME test_loader COMMAND test_loader)
