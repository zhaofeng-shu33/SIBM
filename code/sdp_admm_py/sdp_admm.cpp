// rewrite sdp_admm using eigen api
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <map>
using namespace Eigen;
typedef std::map<std::string, double> List;
// [[Rcpp::depends(RcppArmadillo)]]

MatrixXd Ac(const MatrixXd& X, int n);
MatrixXd Acs(const VectorXd& z, int n);
VectorXd Pinv(const VectorXd& z, int n);
MatrixXd projAXB(const MatrixXd& X0, double alpha, int n);
MatrixXd projA(const MatrixXd& X0, int n);
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
};

// [[Rcpp::export]]
SDPResult sdp1_admm(MatrixXd As, int K, List opts) {
  
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
   
    delta(t) = (X - Xold).norm(); // Fronebius norm
    CONVERGED = delta(t) < tol;
    
    if ((t + 1) % report_interval == 0) {
      printf("%4d | %15e\n", t + 1, delta(t));
    }
    
    t++;
  }
  
  return SDPResult(X, delta, t);

}
// List sdp1_admm_si(MatrixXd As, List opts) {
  
//   double rho = (opts.containsElementNamed("rho") ?  opts["rho"] : .1);
//   int    T   = (opts.containsElementNamed("T") ?  opts["T"] : 10000);
//   double tol = (opts.containsElementNamed("tol") ?  opts["tol"] : 1e-5);
//   int report_interval = (opts.containsElementNamed("report_interval") ?  opts["report_interval"] : 100);
  
//   int    n = As.n_rows;
//   VectorXd delta = arma::zeros(T);
  
//   MatrixXd As_rescaled = (1./rho)*As, 
//             U = arma::zeros(n,n),
//             X = arma::zeros(n,n),
//             Xold = arma::zeros(n,n),
//             Z = arma::zeros(n,n);
  
  
  

//   int t = 0;
//   bool CONVERGED = false;
//   while (!CONVERGED && t<T) {
//     Xold = X;
//     X = projA( 0.5*(Z-U+As_rescaled), n);
//     Z = projToSDC(X+U);
//     U = U+X-Z;
   
//     delta(t) = norm(X-Xold, "F");
//     CONVERGED = delta(t) < tol;
    
//     if ((t+1) % report_interval == 0) {
//       Rprintf("%4d | %15e\n", t+1, delta(t));  
//     }
    
//     t++;
//   }
  
//   return List::create(
//       _["X"]=X,
//       _["delta"]=delta,
//       _["T_term"]=t
//   );
// }

MatrixXd projToSDC(const MatrixXd& M) {
  int n = M.rows();;

  
  VectorXd eigval;
  MatrixXd eigvec;
  SelfAdjointEigenSolver<MatrixXd> es; // for symmetric matrix
  es.compute(M, true);
  eigval = es.eigenvalues();
  eigvec = es.eigenvectors();
  
  for (int i=0; i < eigval.size(); i++){
    if ( eigval(i) < 0 ){ 
      eigval(i) = 0;
    }
  }
  
  return eigvec * eigval.asDiagonal() * eigvec.transpose();
  // VectorXd x = arma::eig_sym(M);
  // std::cout << x(3);
}


MatrixXd projAXB(const MatrixXd& X0, double alpha, int n) {
//   VectorXd b (2*n);
//   b.ones();
  VectorXd b = VectorXd::Ones(2*n);
    
  b.head(n) = 2*(alpha-1) * VectorXd::Ones(n);
  return X0 - Acs( Pinv( Ac(X0, n)-b,n ), n);
}

MatrixXd projA(const MatrixXd& X0, int n) {
//   VectorXd b (2*n);
//   b.ones();
  VectorXd b = VectorXd::Ones(2*n);

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


MatrixXd Ac(const MatrixXd& X, int n) {
  VectorXd vec_joined(2 * n);
  vec_joined << 2 * (X - X.diagonal()) * VectorXd::Ones(n), X.asDiagonal();
  return vec_joined;
}

