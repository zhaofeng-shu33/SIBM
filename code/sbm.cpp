#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <lemon/list_graph.h>

using namespace lemon;
class SIBM2 {
    // SIBM with two community
    public: // make all members and functions public for quick prototyping
        ListGraph& G;
        double _beta;
        double _alpha_divide_beta;
        double mixed_param;
        int n;
        int k;
        int m;
        std::vector<int> sigma;
        SIBM2(ListGraph& graph, double alpha, double beta): G(graph) {
            _beta = beta;
            _alpha_divide_beta = alpha / _beta;
            n = countNodes(G);
            // randomly initiate a configuration
            for (int i = 0; i < n; i++) {
                sigma.push_back(1);
            }
            k = 2;
            m = n; // number of +1
            std::default_random_engine generator;
            std::uniform_int_distribution<int> distribution(0, 1);        
            for (int i = 0; i < n; i++) {
                int candidate = distribution(generator);
                if (candidate == 1) {
                    sigma[i] = -1;      
                    m--;
                }
            }
            // node state is +1 or -1
            mixed_param = _alpha_divide_beta * log(n);
            mixed_param /= n;
        }
        double get_dH(int trial_location) {
            double _sum = 0;
            int w_s = sigma[trial_location];
            for(ListGraph::OutArcIt arc(G, G.nodeFromId(trial_location)); arc != INVALID; ++arc) {
                int i = G.id(G.target(arc));
                _sum += sigma[i];
            }
            _sum *= w_s * (1 + mixed_param);
            _sum -= mixed_param * (w_s * (2 * m - n) - 1);
            return _sum;
        }
        void _metropolis_single() {
            // randomly select one position to inspect
            std::default_random_engine generator;
            std::uniform_int_distribution<int> distribution(0, n - 1);        
            int r = distribution(generator);
            double delta_H = get_dH(r);
            if (delta_H < 0) { // lower energy: flip for sure
                sigma[r] *= -1;
                m += sigma[r];
            } else {  // Higher energy: flip sometimes
                double probability = exp(-1.0 * _beta * delta_H);
                std::default_random_engine generator;
                std::uniform_int_distribution<int> distribution(0, 1);        
                if (distribution(generator) < probability) {
                    sigma[r] *= -1;
                    m += sigma[r];
                }
            }
        }
        void metropolis(int N=40) {
            // iterate given rounds
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < n; j++) {
                    this->_metropolis_single();
                }
            }
        }
};

bool same_community(int n, int k, int i, int j) {
    int com = n / k;
    if (i / com == j / com) {
        return true;
    }
    return false;
}
ListGraph* sbm_graph(int n, int k, int a, int b) {
    double p = 1.0 * a * log(n) / n;
    double q = 1.0 * b * log(n) / n;
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);    
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
        cnt++;
    }
    std::cout << cnt << std::endl;
    delete g;
}