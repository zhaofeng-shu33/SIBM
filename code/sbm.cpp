#include <cmath>
#include <iostream>
#include <random>

#include <lemon/list_graph.h>

using namespace lemon;

bool same_community(int n, int k, int i, int j) {
    int com = n / k;
    if (i / com == j / com) {
        return true;
    }
    return false;
}
ListGraph* sbm_graph(int n, int k, int a, int b) {
    double p = a * log(n) / n;
    double q = b * log(n) / n;
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0,1.0);    
    ListGraph* g = new ListGraph();
    for (int i = 0; i < n; i++) {
        g->addNode();
    }
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            bool is_same_community = same_community(n, k, i, j);
            double random_number = distribution(generator);
            if (is_same_community) {
                if (random_number < p) {
                    g->addEdge(g->nodeFromId(i), g->nodeFromId(j));
                }
            }
            else {
                if (random_number < q) {
                    g->addEdge(g->nodeFromId(i), g->nodeFromId(j));
                }
            }
        }
    }
    return g;
}
int main() {
    ListGraph* g = sbm_graph(100, 2, 16, 4);
    int cnt = 0;
    for(ListGraph::EdgeIt e(*g); e != INVALID; ++e) {
        int u_id = g->id(g->u(e));
        int v_id = g->id(g->v(e));
        std::cout << u_id << ',' << v_id << std::endl;
    }
    delete g;
}