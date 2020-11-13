// g++ test.cpp -lgtest -lgtest_main -lpthread -o build/test
#include <gtest/gtest.h>
#include <lemon/list_graph.h>
#include "sbm.h"
using namespace lemon;
TEST(SIBM, Rest) {
    ListGraph* g = sbm_graph(100, 2, 16, 4);
    SIBM2 sibm(*g, 8, 1);
    sibm.metropolis(100);
    EXPECT_TRUE(exact_compare(sibm.sigma));
    delete g;
}
