// g++ sibm_experiment.cpp -o build/sibm_experiment
#include "sbm.h"

int main() {
    int repeat = 2;
    int n = 3200;
    int k = 2;
    double a = 16;
    double b = 4;
    double alpha = 8;
    double beta = 0.2;
    int num_of_sibm_samples = 2; // inner repeat;
    int m = 3;
    int _N = 100;
    std::cout << task_cpp(repeat, n, k, a, b, alpha, beta, num_of_sibm_samples, m, _N) << '\n';
}