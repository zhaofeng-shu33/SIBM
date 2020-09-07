# Rscript sbm_semi_definite.R
library("sbmSDP")
get_labels <- function(cluster_matrix) {
    n <- dim(cluster_matrix)[1]
    labels <- array(0, n)
    label_index <- 1
    for(i in 1:n) {
       if(labels[i] != 0)
            next
       labels[i] <- label_index
       for(j in i:n) {
           if(cluster_matrix[i, j] > 0.5)
               labels[j] <- label_index
       }
       label_index <- label_index + 1     
    }
    labels
}
blkmodel <- list(m=20, K=3, p=.9, q=.4)
blkmodel <- within(blkmodel, { 
                   n <- m*K
                   M <- kronecker(matrix(c(p,q,q,q,p,q,q,q,p),nrow=3),matrix(1,m,m))
                   As <- 1*(matrix(runif(n^2),nrow=n) < M)
                   })
# Call sdp1_admm with options:
#  rho  the ADMM parameter, 
#  T    maximum number of iteration
#  tol  tolerance for norm(X_{t+1} - X_t)
#  report_interval  how many iteration between reporting progress
sdp.fit <- with(blkmodel, 
                sdp1_admm(as.matrix(As), K, list(rho=.1, T=10000, tol=1e-5, report_interval=100)))

get_labels(sdp.fit$X)
