#pragma once
// rewrite sdp_admm using eigen api
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <map>
using namespace Eigen;
typedef std::map<std::string, double> List;
// [[Rcpp::depends(RcppArmadillo)]]

VectorXd Ac(const MatrixXd& X, int n);
MatrixXd Acs(const VectorXd& z, int n);
VectorXd Pinv(const VectorXd& z, int n);
MatrixXd projA(const MatrixXd& X0, int n);

VectorXd Ac_si(const MatrixXd& X, int n);
MatrixXd Acs_si(const VectorXd& z, int n);
VectorXd Pinv_si(const VectorXd& z, int n);
MatrixXd projA_si(const MatrixXd& X0, int n);

MatrixXd projAXB(const MatrixXd& X0, double alpha, int n);
MatrixXd projToSDC(const MatrixXd& M);

struct SDPResult {
    MatrixXd X;
    VectorXd delta;
    int T_term;
    SDPResult(MatrixXd _X, VectorXd _delta, int _T_term) {
        X = _X;
        delta = _delta;
        T_term = _T_term;
    }
    SDPResult() {}
};
SDPResult sdp1_admm(const MatrixXd& As, int K, List opts);
SDPResult sdp_admm_sbm_2(const MatrixXd& As, List opts);
SDPResult sdp_admm_sbm_si_2(const MatrixXd& As, List opts);