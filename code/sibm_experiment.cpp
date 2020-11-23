// g++ sibm_experiment.cpp -o build/sibm_experiment
#include "sbm.h"

int main() {
    int repeat = 2;
    int n = 600;
    int k = 2;
    double a = 16;
    double b = 4;
    double alpha = 8;
    double beta = 0.4;
    int num_of_sibm_samples = 4; // inner repeat;
    int m = 3; 
    int _N = 40;
    std::cout << task_cpp(repeat, n, k, a, b, alpha, beta, num_of_sibm_samples, m, _N) << '\n';
}