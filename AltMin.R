library(pracma)
a <- pracma::rand(n = 8, m = 1)
b <- pracma::rand(n = 200, m = 1)
X <- pracma::rand(n = 2700, m = 8)
Y <- X%*%a%*%t(b)

listU <- list()
listV <- list()
r <- 1
epsilon <- .01
lambda = c(.1, .01, .001, .0001, .00001)
for (val in lambda) {
  k <- 0
  U <- pracma::rand(n = ncol(X), m = r)
  V <- pracma::rand(n = ncol(Y), m = r)
  converge <- FALSE
  while (k == 0 || !converge) {
    k = k + 1
    # V step
    input <- t(Y) %*% X %*% U
    result <- svd(input, nu = r, nv = r)
    V <- result$u %*% t(result$v)
    
    # U step
    input <- Y %*% V
    test <- glmnetPlus::glmnet(X, input, family = "mgaussian", lambda = val)
    U <- as.matrix(test$beta)
    
    listU[[k]] = U
    listV[[k+1]] = V
    
    # check for convergence
    if (k > 2) {
      input <- listU[[k]] %*% t(listV[[k]]) - listU[[k-1]] %*% t(listV[[k-1]])
      delta <- norm(input)
    }
    if (k > 2 && delta < epsilon) {
      converge <- TRUE
    }
  }
}

