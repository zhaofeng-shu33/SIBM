#pragma once
#include <Eigen/Dense>
#include "sdp_admm.h"
using namespace Eigen;


void set_value(MatrixXd & arr, int x, int y, double value) {
    arr(x, y) = value;
}
void get_mat(MatrixXd& arr, SDPResult& fit_results) {
    arr = fit_results.X;
}
void set_list_value(List& list, double rho, int T, double tol, int report_interval) {
    list["rho"] = rho;
    list["T"] = double(T);
    list["tol"] = tol;
    list["report_interval"] = double(report_interval);
}
