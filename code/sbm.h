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
            int r = std::rand() % n;
            double delta_H = get_dH(r);
            if (delta_H < 0) { // lower energy: flip for sure
                sigma[r] *= -1;
                m += sigma[r];
            } else {  // Higher energy: flip sometimes
                double probability = exp(-1.0 * _beta * delta_H);
                std::random_device dev;
                std::default_random_engine generator(dev());
                std::uniform_real_distribution<double> distribution(0, 1);        
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

class SIBMk {
    // SIBM with two community
    public: // make all members and functions public for quick prototyping
        ListGraph& G;
        double _beta;
        double _alpha_divide_beta;
        double mixed_param;
        int n;
        int k;
        std::vector<int> m;
        std::vector<int> sigma;
        SIBMk(ListGraph& graph, double alpha, double beta, int _k=2): G(graph) {
            _beta = beta;
            _alpha_divide_beta = alpha / _beta;
            k = _k;
            n = countNodes(G);
            // randomly initiate a configuration
            for (int i = 0; i < n; i++) {
                sigma.push_back(0);
            }
            m.push_back(n);
            for (int i = 1; i < k; i++) {
                m.push_back(0);
            }
            std::random_device dev;
            std::default_random_engine generator(dev());
            std::uniform_int_distribution<int> distribution(0, k - 1);        
            for (int i = 0; i < n; i++) {
                int candidate = distribution(generator);
                if (candidate > 0) {
                    sigma[i] = candidate;      
                    m[0]--;
                    m[candidate]++;
                }
            }
            // node state is 0, 1, \dots, k-1
            mixed_param = _alpha_divide_beta * log(n);
            mixed_param /= n;
        }
        double get_dH(int trial_location, int w_s) {
            double _sum = 0;
            int sigma_r = sigma[trial_location];
            int w_s_sigma_r = (w_s + sigma_r) % k;
            for(ListGraph::OutArcIt arc(G, G.nodeFromId(trial_location)); arc != INVALID; ++arc) {
                int i = G.id(G.target(arc));
                if (sigma_r == sigma[i]) {
                    _sum += 1;
                } else if (w_s_sigma_r == sigma[i]) {
                    _sum -= 1;
                }
            }
            _sum *= (1 + mixed_param);
            _sum += mixed_param * (m[w_s_sigma_r] - m[sigma_r] + 1);
            return _sum;
        }
        void _metropolis_single() {
            // randomly select one position to inspect
            int r = std::rand() % n;
            int w_s = std::rand() % (k - 1) + 1;
            double delta_H = get_dH(r, w_s);
            if (delta_H < 0) { // lower energy: flip for sure
                m[sigma[r]]--;
                sigma[r] = (w_s + sigma[r]) % k;
                m[sigma[r]]++;
            } else {  // Higher energy: flip sometimes
                double probability = exp(-1.0 * _beta * delta_H);
                std::random_device dev;
                std::default_random_engine generator(dev());
                std::uniform_real_distribution<double> distribution(0, 1);        
                if (distribution(generator) < probability) {
                    m[sigma[r]]--;
                    sigma[r] = (w_s + sigma[r]) % k;
                    m[sigma[r]]++;
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

bool exact_compare(std::vector<int> labels) {
    // return 1 if labels = X or -X
    int result = 0;
    int n2 = labels.size() / 2;
    for (int i = 0; i < n2; i++) {
        result += labels[i];
    }
    if (int(abs(result)) != n2) {
        return false;
    }
    result = std::accumulate(labels.begin(), labels.end(), 0);
    return result == 0;
}

bool exact_compare_k(std::vector<int> labels, int k) {
    int n = labels.size();
    int nk = n / k;
    for (int i = 0; i < k; i++) {
        int candidate = labels[nk * i];
        for(int j = nk * i + 1; j < nk * (i + 1); j++) {
            if (labels[j] != candidate)
                return false;
        }
    }
    return true;
}

ListGraph* sbm_graph(int n, int k, int a, int b) {
    double p = 1.0 * a * log(n) / n;
    double q = 1.0 * b * log(n) / n;
    std::random_device dev;
    std::default_random_engine generator(dev());
    std::uniform_real_distribution<double> distribution(0, 1);
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

double task_cpp(int repeat, int n, int k, double a, double b, double alpha, double beta, int m, int _N) {
    double total_acc = 0;
    if (k == 2) {
        for (int i = 0; i < repeat; i++) {
            ListGraph* G = sbm_graph(n, 2, a, b);
            SIBM2 sibm(*G, alpha, beta);
            sibm.metropolis(_N);
            double acc = 0;
            for (int j = 0; j < m; j++) {
                sibm._metropolis_single();
                double inner_acc = double(exact_compare(sibm.sigma)); // for exact recovery
                acc += inner_acc;
            }
            acc /= m;
            total_acc += acc;
            delete G;
        }
    } else {
        for (int i = 0; i < repeat; i++) {
            ListGraph* G = sbm_graph(n, k, a, b);
            SIBMk sibm(*G, alpha, beta, k);
            sibm.metropolis(_N);
            double acc = 0;
            for (int j = 0; j < m; j++) {
                sibm._metropolis_single();
                double inner_acc = double(exact_compare_k(sibm.sigma, k)); // for exact recovery
                acc += inner_acc;
            }
            acc /= m;
            total_acc += acc;
            delete G;
        }
    }

    total_acc /= repeat;
    return total_acc;
}
