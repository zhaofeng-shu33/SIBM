// g++ test.cpp -lgtest -lgtest_main -lpthread -o build/test
#include <gtest/gtest.h>
#include <lemon/list_graph.h>
#include "sbm.h"
using namespace lemon;


bool vector_equal_array(const std::vector<int>& a_v, int arr[], int m) {
    if (a_v.size() != m)
        return false;
    for (int i = 0; i < m; i++) {
        if (a_v[i] != arr[i])
            return false;
    }
    return true;
}

TEST(SIBM, Rest) {
    ListGraph* g = sbm_graph(300, 2, 16, 4);
    SIBM2 sibm(*g, 8, 1);
    sibm.metropolis(100);
    EXPECT_TRUE(exact_compare(sibm.sigma));
    delete g;
}

TEST(SIBMk, Simple) {
    int k = 3;
    ListGraph* g = sbm_graph(300, k, 20, 4);
    SIBMk sibm(*g, 8, 1, k);
    sibm.metropolis(100);
    EXPECT_TRUE(exact_compare_k(sibm.sigma, k));
    delete g;
}

TEST(Util, majority_voting_k) {
    std::vector<int> voting_result;
    voting_result.resize(9);
    int ground_truth[] = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    int _a1[] = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    int _a2[] = {1, 1, 1, 0, 0, 0, 2, 2, 2};
    int _a3[] = {1, 0, 0, 1, 1, 1, 2, 2, 2};
    std::vector<int> b1(_a1, _a1 + 9);
    std::vector<int> b2(_a2, _a2 + 9);
    std::vector<int> b3(_a3, _a3 + 9);

    std::vector<std::vector<int>*> a;
    a.push_back(&b1);
    majority_voting_k(a, 3, voting_result);
    EXPECT_TRUE(vector_equal_array(voting_result, ground_truth, 9));
    a.push_back(&b2);
    majority_voting_k(a, 3, voting_result);
    EXPECT_TRUE(vector_equal_array(voting_result, ground_truth, 9));

    a.push_back(&b3);
    majority_voting_k(a, 3, voting_result);
    EXPECT_TRUE(vector_equal_array(voting_result, ground_truth, 9));
}