// rewrite sdp_admm using eigen api
#include <Eigen/Dense>
#include <map>
using namespace Eigen;
typedef std::map<std::string, double> List;
// [[Rcpp::depends(RcppArmadillo)]]

MatrixXd Ac(MatrixXd X, int n);
MatrixXd Acs(VectorXd z, int n);
VectorXd Pinv(VectorXd z, int n);
MatrixXd projAXB(MatrixXd X0, double alpha, int n);
MatrixXd projA(MatrixXd X0, int n);
MatrixXd projToSDC(MatrixXd M);

struct SDPResult {
    MatrixXd X;
    VectorXd delta;
    int T_term;
};

// [[Rcpp::export]]
SDPResult sdp1_admm(MatrixXd As, int K, List opts) {
  
  double rho = (opts.count("rho") ?  opts["rho"] : .1);
  int    T   = (opts.count("T") ?  int(opts["T"]) : 10000);
  double tol = (opts.count("tol") ?  opts["tol"] : 1e-5);
  int report_interval = (opts.count("report_interval") ?  int(opts["report_interval"]) : 100);
  
  int    n = As.rows();
  VectorXd delta = VectorXd::Zero(T);
  
  MatrixXd As_rescaled = (1./rho) * As, 
            U = MatrixXd::Zero(n,n),
            V = MatrixXd::Zero(n,n),
            X = MatrixXd::Zero(n,n),
            Xold = MatrixXd::Zero(n,n),
            Y = MatrixXd::Zero(n,n),
            Z = MatrixXd::Zero(n,n);
  
  double alpha = (n*1.)/K;
  

  int t = 0;
  bool CONVERGED = false;
  while (!CONVERGED && t<T) {
    Xold = X;
    X = projAXB( 0.5*(Z-U+Y-V+As_rescaled), alpha, n);
    Z = (X+U).cwiseMax(MatrixXd::Zero(n,n));
    Y = projToSDC(X+V);
    U = U+X-Z;
    V = V+X-Y;
   
    delta(t) = (X - Xold).norm(); // Fronebius norm
    CONVERGED = delta(t) < tol;
    
    if ((t+1) % report_interval == 0) {
      printf("%4d | %15e\n", t+1, delta(t));
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

MatrixXd projToSDC(MatrixXd M) {
  int n = M.n_rows;

  
  VectorXd eigval;
  MatrixXd eigvec;
  
  arma::eig_sym(eigval, eigvec, M);
  
  for (int i=0; i < eigval.n_elem; i++){
    if ( eigval(i) < 0 ){ 
      eigval(i) = 0;
    }
  }
  
  M = eigvec * arma::diagmat(eigval) * eigvec.t();
  // VectorXd x = arma::eig_sym(M);
  // std::cout << x(3);
  return M;
}


MatrixXd projAXB(MatrixXd X0, double alpha, int n) {
//   VectorXd b (2*n);
//   b.ones();
  VectorXd b = arma::ones(2*n);

  b(arma::span(0,n-1)) = 2*(alpha-1)*arma::ones(n);
  return X0 - Acs( Pinv( Ac(X0, n)-b,n ), n);
}

MatrixXd projA(MatrixXd X0, int n) {
//   VectorXd b (2*n);
//   b.ones();
  VectorXd b = arma::ones(2*n);

  b(arma::span(0,n-1)) = 0 * arma::ones(n);
  return X0 - Acs( Pinv( Ac(X0, n)-b,n ), n);
}

MatrixXd Acs(VectorXd z, int n) {
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


VectorXd Pinv(VectorXd z, int n) {
  VectorXd mu = z.head(n);
  VectorXd nu = z.tail(n);
  
  return arma::join_vert( (1./(2*(n-2)))*(mu - arma::ones(n)*arma::sum(mu)/(2*n-2)), nu);
}


MatrixXd Ac( MatrixXd X, int n) {
  return arma::join_vert( 2*(X - X.asDiagonal()) * VectorXf::Ones(n), arma::diagvec(X) );
}

