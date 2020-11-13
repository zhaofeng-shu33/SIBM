#include "sbm.h"

int main() {
    ListGraph* g = sbm_graph(100, 2, 16, 4);
    SIBM2 sibm(*g, 8, 1);
    sibm.metropolis(100);
    for (int i = 0; i < 100; i++) {
        std::cout << sibm.sigma[i] << ',';
    }
    std::cout << '\n';
    delete g;
}