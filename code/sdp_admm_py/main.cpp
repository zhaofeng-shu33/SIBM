// #include <iostream>
#include <Eigen/Dense>
#include "sdp_admm.h"
using Eigen::MatrixXd;
 
int main()
{
  MatrixXd m(2,2);
  VectorXd b = Ac(m, 2);
//  std::cout << m << std::endl;
}
