#include "sdp_admm.h"
#include <stdio.h>
// [[Rcpp::export]]
SDPResult sdp1_admm(const MatrixXd& As, int K, List opts) {

  double rho = (opts.count("rho") ?  opts["rho"] : .1);
  int    T   = (opts.count("T") ?  int(opts["T"]) : 10000);
  double tol = (opts.count("tol") ?  opts["tol"] : 1e-5);
  int report_interval = (opts.count("report_interval") ?  int(opts["report_interval"]) : 100);
  
  int    n = As.rows();
  VectorXd delta = VectorXd::Zero(T);
  
  MatrixXd As_rescaled = (1. / rho) * As,
            U = MatrixXd::Zero(n, n),
            V = MatrixXd::Zero(n, n),
            X = MatrixXd::Zero(n, n),
            Xold = MatrixXd::Zero(n, n),
            Y = MatrixXd::Zero(n, n),
            Z = MatrixXd::Zero(n, n);
  
  double alpha = (n * 1.) / K;
  

  int t = 0;
  bool CONVERGED = false;
  while (!CONVERGED && t < T) {
    Xold = X;
    X = projAXB( 0.5 * (Z - U + Y - V + As_rescaled), alpha, n);
    Z = (X + U).cwiseMax(MatrixXd::Zero(n, n));
    Y = projToSDC(X + V);
    U = U + X - Z;
    V = V + X - Y;
   
    delta(t) = (X - Xold).norm(); // Frobenius norm
    CONVERGED = delta(t) < tol;
    
    if ((t + 1) % report_interval == 0) {
      printf("%4d | %15e\n", t + 1, delta(t));
    }
    
    t++;
  }
  
  return SDPResult(X, delta, t);

}

// can be used to solve SBM with two community
SDPResult sdp_admm_sbm_2(const MatrixXd& As, List opts) {
  
  double rho = (opts.count("rho") ?  opts["rho"] : .1);
  int    T   = (opts.count("T") ?  int(opts["T"]) : 10000);
  double tol = (opts.count("tol") ?  opts["tol"] : 1e-5);
  int report_interval = (opts.count("report_interval") ?  int(opts["report_interval"]) : 100);
  
  int    n = As.rows();
  VectorXd delta = VectorXd::Zero(T);
  
  MatrixXd As_rescaled = (1. / rho) * As,
            U = MatrixXd::Zero(n, n),
            X = MatrixXd::Zero(n, n),
            Z = MatrixXd::Zero(n, n);
    

  int t = 0;
  bool CONVERGED = false;
  while (t < T) {
    X = Z - U + As_rescaled;
    X.diagonal() = VectorXd::Ones(n);
    Z = projToSDC(X + U);
    delta(t) = (X - Z).norm(); // Frobenius norm
    CONVERGED = delta(t) < tol;
    if (CONVERGED) {
      break;
    }
    U = U + X - Z;
   
    
    if ((t + 1) % report_interval == 0) {
      printf("%4d | %15e\n", t + 1, delta(t));
    }
    
    t++;
  }
  
  return SDPResult(X, delta, t);
}


SDPResult sdp_admm_sbm_si_2(const MatrixXd& As, List opts) {
  
  double rho = (opts.count("rho") ?  opts["rho"] : .1);
  int    T   = (opts.count("T") ?  int(opts["T"]) : 10000);
  double tol = (opts.count("tol") ?  opts["tol"] : 1e-5);
  int report_interval = (opts.count("report_interval") ?  int(opts["report_interval"]) : 100);
  
  int    n = As.rows();
  VectorXd delta = VectorXd::Zero(T);
  
  MatrixXd As_rescaled = (1. / rho) * As,
            U = MatrixXd::Zero(n, n),
            X = MatrixXd::Zero(n, n),
            Z = MatrixXd::Zero(n, n),
            X_new = MatrixXd::Zero(n, n);

  int t = 0;
  bool CONVERGED = false;
  while (t < T) {
    X_new = Z - U + As_rescaled;
    X = projA_si(X_new, n);
    Z = projToSDC(X + U);
    delta(t) = (X - Z).norm(); // Frobenius norm
    CONVERGED = delta(t) < tol;
    if (CONVERGED) {
      break;
    }
    U = U + X - Z;
   
    
    if ((t + 1) % report_interval == 0) {
      printf("%4d | %15e\n", t + 1, delta(t));
    }
    
    t++;
  }
  
  return SDPResult(X, delta, t);
}

MatrixXd projToSDC(const MatrixXd& M) {
  VectorXd eigval;
  MatrixXd eigvec;
  SelfAdjointEigenSolver<MatrixXd> es; // for symmetric matrix
  es.compute(M, ComputeEigenvectors);
  eigval = es.eigenvalues();
  eigvec = es.eigenvectors();
  
  for (int i=0; i < eigval.size(); i++){
    if ( eigval(i) < 0 ){ 
      eigval(i) = 0;
    }
  }
  
  return eigvec * eigval.asDiagonal() * eigvec.transpose();
}


MatrixXd projAXB(const MatrixXd& X0, double alpha, int n) {
//   VectorXd b (2*n);
//   b.ones();
  VectorXd b = VectorXd::Ones(2 * n);
    
  b.head(n) = 2 * (alpha - 1) * VectorXd::Ones(n);
  return X0 - Acs( Pinv( Ac(X0, n) - b, n ), n);
}

MatrixXd projA(const MatrixXd& X0, int n) {
//   VectorXd b (2*n);
//   b.ones();
  VectorXd b = VectorXd::Ones(2 * n);

  b.head(n) = VectorXd::Zero(n);
  return X0 - Acs( Pinv( Ac(X0, n)-b,n ), n);
}

MatrixXd Acs(const VectorXd& z, int n) {
  VectorXd mu = z.head(n);
  VectorXd nu = z.tail(n);
  MatrixXd Z(n,n);
  
  for (int i=0; i < n; i++) {
    Z(i,i) = nu(i);
  }
  
  for (int i=0; i < n; i++) {
    for (int j=i+1; j < n; j++) {
        Z(i,j) = mu(i) + mu(j);
        Z(j,i) = Z(i,j);
    }
  }
  
  return Z;
}


VectorXd Pinv(const VectorXd& z, int n) {
  VectorXd mu = z.head(n);
  VectorXd nu = z.tail(n);
  VectorXd vec_joined(2 * n);
  vec_joined << (1./(2*(n-2)))*(mu - VectorXd::Ones(n) * mu.sum()/(2*n-2)), nu;
  return vec_joined;
}


VectorXd Ac(const MatrixXd& X, int n) {
  VectorXd vec_joined(2 * n);
  vec_joined << 2 * (X * VectorXd::Ones(n) - X.diagonal()), X.diagonal();
  return vec_joined;
}

MatrixXd projA_si(const MatrixXd& X0, int n) {
//   VectorXd b (2*n);
//   b.ones();
  VectorXd b = VectorXd::Ones(2*n);

  b.head(n) = VectorXd::Constant(n, -2);
  b(0) = 0;
  return X0 - Acs_si( Pinv_si( Ac_si(X0, n)-b,n ), n);
}

MatrixXd Acs_si(const VectorXd& z, int n) {
  VectorXd mu = z.head(n);
  VectorXd nu = z.tail(n);
  MatrixXd Z(n,n);
  
  for (int i=1; i < n; i++) {
    for (int j=i+1; j < n; j++) {
        Z(i,j) = mu(i) + mu(j);
        Z(j,i) = Z(i,j);
    }
  }
  for (int i=0; i < n; i++) {
    Z(0, i) = mu(0);
    Z(i, 0) = mu(0);
    Z(i, i) = nu(i);
  }  
  return Z;
}


VectorXd Pinv_si(const VectorXd& z, int n) {
  VectorXd mu = z.segment(1, n);
  VectorXd nu = z.tail(n);
  VectorXd vec_joined(2 * n);
  vec_joined << z(0) / (2 * n - 2), (1./(2 * (n - 3)))*(mu - VectorXd::Ones(n - 1) * mu.sum()/(2*n-4)), nu;
  return vec_joined;
}


VectorXd Ac_si(const MatrixXd& X, int n) {
  VectorXd vec_joined(2 * n);
  vec_joined << 2 * X.block(0, 1, 1, n-1).sum(), 2 * (X.block(1, 1, n-1, n-1) * VectorXd::Ones(n-1) - X.diagonal().segment(1, n-1)), X.diagonal();
  return vec_joined;
}
