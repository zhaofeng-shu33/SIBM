#include "sbm.h"

int main() {
    int repeat = 400;
    int n = 3000;
    int k = 2;
    double a = 16;
    double b = 4;
    double alpha = 8;
    double beta = 0.4;
    int m = 1000; // inner repeat;
    int _N = 40;
    std::cout << task_cpp(repeat, n, k, a, b, alpha, beta, m, _N) << '\n';
}