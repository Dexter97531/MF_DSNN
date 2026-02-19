# Load required libraries
library(matrixcalc)
library(clusterGeneration)
library(fBasics)
library(scoringRules)
library(glmnet)
library(CVglasso)
library(MASS)  # For mvrnorm

#==========Input Parameters=============
k1 <- 76    # Number of monthly variables
k2 <- 9     # Number of quarterly variables
# train_size <- 216 # Number of training observations
# test_size <- 24 # Number of testing/forecast observations

train_size <- 236 # Number of training observations
test_size <- 24 # Number of testing/forecast observations

nobs <- train_size - 1
horizon <- test_size # Forecast horizon = 24
total_time <- nobs + test_size + 1 # 240 time points
theta_true <- 0.5
iteration <- 3000
burn <- 1000
# iteration <- 300
# burn <- 100
repl <- 1
lag <- 1
spec_rad <- 0.7
edge_density <- 0.04

N <- 1000

# Compute dimensions
p <- (3*k1) + k2  # Total variables: 3*76 + 9 = 237
p1 <- (k1 + k2)   # 76 + 9 = 85

#===========Load Data================
# Read the pre-generated data from CSV
data_full <- as.matrix(read.csv("D:/Code/MF/data/data3_196001_202503_76_9.csv")[,-1]) # Remove index column

# Restrict data to first 240 time points
data_full <- data_full[, 1:total_time]

# Debug: Check dimensions of data_full
cat("Dimensions of data_full:", dim(data_full), "\n")
if (nrow(data_full) != p) stop("data_full does not have p = ", p, " rows")
if (ncol(data_full) != total_time) stop("data_full has ", ncol(data_full), " columns, expected ", total_time)

# Define permutation vectors based on number of quarters
n_quarters <- floor(ncol(data_full) / 3)  # Approximate number of quarters
s1 <- seq(1, n_quarters * 3, by=3)  # First month of each quarter
s2 <- seq(2, n_quarters * 3, by=3)  # Second month
s3 <- seq(3, n_quarters * 3, by=3)  # Third month
vec <- c(s3, s1, s2, (3*k1 + 1):(3*k1 + k2))  # Permutation: hf3, hf1, hf2, lf
vec1 <- c((k1+1):(k1+k2), 1:k1)               # Permutation: quarterly then monthly variables
vec2 <- 1:p                                   # Identity permutation

# Debug: Check lengths of permutation vectors
cat("Length of vec:", length(vec), ", Expected:", p, "\n")
cat("Length of vec1:", length(vec1), ", Expected:", p1, "\n")

# Prepare data
y_full <- data_full
y_mod_full <- y_full[vec1, ]
data <- y_full[, 1:(nobs+1)] # Training data: first 216 time points
data_mod <- as.matrix(y_mod_full)[, 1:(nobs+1)]
data_t <- t(data)
forecast_true <- y_full[, (nobs + 2):(total_time)] # Testing data: time points 217 to 240

# Debug: Check dimensions of derived data
cat("Dimensions of y_full:", dim(y_full), "\n")
cat("Dimensions of y_mod_full:", dim(y_mod_full), "\n")
cat("Dimensions of data:", dim(data), "\n")
cat("Dimensions of data_mod:", dim(data_mod), "\n")
cat("Dimensions of data_t:", dim(data_t), "\n")
cat("Dimensions of forecast_true:", dim(forecast_true), "\n")

# Least squares estimation for initialization
X <- data_t[1:(nobs), ]
Y <- data_t[2:(nobs+1), ] # Adjust to match dimensions
LSE <- lm(Y ~ X - 1)$coef

#======Initial Values===========
A11_initial <- t(LSE)[1:k1, 1:k1]
b1 <- as.vector(A11_initial)
b2 <- numeric(k1 * k2)
b3 <- numeric(k1 * k2)
b4 <- numeric(k2 * k2)

fit_cv <- cv.glmnet(X, Y, alpha=1, family="mgaussian", nfolds=5, type.measure="mse")
lambda_min <- fit_cv$lambda.min
fit <- glmnet(X, Y, lambda=lambda_min, family="mgaussian")
sparse_coef <- do.call(cbind, fit$beta)
Phi_initial <- as.matrix(sparse_coef)
A22_initial <- t(Phi_initial)[(k1+1):(k1+k2), (k1+1):(k1+k2)]
b4 <- as.vector(A22_initial)

# Initialize arrays
W_sim <- array(0, c(p, p, (iteration-burn)))
theta_sim <- array(0, c(1, 1, (iteration-burn)))
sigma_sim <- array(0, c(p, p, (iteration-burn)))
b1_sim <- array(0, c((k1*k1), 1, (iteration-burn)))
b2_sim <- array(0, c((k1*k2), 1, (iteration-burn)))
b3_sim <- array(0, c((k1*k2), 1, (iteration-burn)))
b4_sim <- array(0, c((k2*k2), 1, (iteration-burn)))

w_final <- matrix(0, p, p)
theta_final <- 0
sigma_final <- matrix(0, p, p)
b1_final <- rep(0, (k1*k1))
b2_final <- rep(0, (k1*k2))
b3_final <- rep(0, (k1*k2))
b4_final <- rep(0, (k2*k2))

count <- 0

# Source custom functions (assumed to be available)
source("Example/functions.R")

# Theta and related matrices
cat("Calling theta_matrix...\n")
theta_func <- theta_matrix(theta_true, b1, b2, b3, b4, k1, k2, vec1)
theta_mat <- theta_func[[1]]
u <- theta_func[[2]]
v <- theta_func[[3]]
w11 <- theta_func[[4]]
w12 <- theta_func[[5]]
w21 <- theta_func[[6]]
w22 <- theta_func[[7]]
w <- theta_func[[8]]
w_mod <- theta_func[[9]]

# Debug: Check dimensions of theta_func outputs
cat("Dimensions of theta_mat:", dim(theta_mat), "\n")
cat("Dimensions of w:", dim(w), "\n")
cat("Dimensions of w_mod:", dim(w_mod), "\n")
cat("Length of u:", length(u), "\n")
cat("Length of v:", length(v), "\n")

# S matrices
cat("Calling S_matrix...\n")
S_mat <- S_matrix(data, nobs, k1, k2)
s1 <- S_mat[[1]]
s2 <- S_mat[[2]]
s3 <- S_mat[[3]]
s11 <- S_mat[[4]]
s12 <- S_mat[[5]]
s21 <- S_mat[[6]]
s22 <- S_mat[[7]]
ss11 <- S_mat[[8]]
ss12 <- S_mat[[9]]
ss21 <- S_mat[[10]]
ss22 <- S_mat[[11]]
s <- S_mat[[12]]

# Debug: Check dimensions of S_mat outputs
cat("Dimensions of s11:", dim(s11), "\n")
cat("Dimensions of s12:", dim(s12), "\n")
cat("Dimensions of s21:", dim(s21), "\n")
cat("Dimensions of s22:", dim(s22), "\n")
cat("Dimensions of s:", dim(s), "\n")
cat("Dimensions of ss11:", dim(ss11), "\n")
cat("Dimensions of ss12:", dim(ss12), "\n")
cat("Dimensions of ss21:", dim(ss21), "\n")
cat("Dimensions of ss22:", dim(ss22), "\n")

#============Initial Values of Sigma===================
omega <- list()
sigma_adj <- numeric()
for (i in 1:k1) {
  sigma1 <- 0.3 * diag(3)
  omega[[i]] <- solve(sigma1)
  sigma_adj[i] <- 1 / (t(u) %*% solve(sigma1) %*% u)
}
sigma2 <- rep(1, k2)

# Debug: Check sigma_adj and sigma2
cat("Length of sigma_adj:", length(sigma_adj), ", Expected:", k1, "\n")
cat("Length of sigma2:", length(sigma2), ", Expected:", k2, "\n")

#==============Hyperparameters=================
alpha <- 1
beta <- 1
Q <- 0.3 * diag(3)
df <- 1

cat("Calling sigma_inverse...\n")
sigma_inv_func <- sigma_inverse(omega, sigma2, k1, k2)
sigma_11 <- sigma_inv_func[[1]]
sigma_22 <- sigma_inv_func[[2]]
sigma_inv <- sigma_inv_func[[3]]
Final_Sigma_Inv <- sigma_inv_func[[4]]
sigma_mat <- sigma_inv_func[[5]]

# Debug: Check dimensions of sigma_inv_func outputs
cat("Dimensions of sigma_11:", dim(sigma_11), "\n")
cat("Dimensions of sigma_22:", dim(sigma_22), "\n")
cat("Dimensions of sigma_inv:", dim(sigma_inv), "\n")
cat("Dimensions of Final_Sigma_Inv:", dim(Final_Sigma_Inv), "\n")
cat("Dimensions of sigma_mat:", dim(sigma_mat), "\n")

# Parameter calculations
cat("Calling para_A11...\n")
A11_func <- para_A11(theta_mat, s, sigma_inv, sigma_11, s11, ss11, s12, w12)
gamma_A11 <- A11_func[[1]]
d_A11 <- A11_func[[2]]
q11 <- 0
tao11 <- 0.5

cat("Calling para_A12...\n")
A12_func <- para_A12(k1, k2, u, sigma_inv, sigma_11, s12, ss12, s22, w11)
gamma_A12 <- A12_func[[1]]
d_A12 <- A12_func[[2]]
q12 <- 0.96
tao12 <- 0.5

cat("Calling para_A21...\n")
A21_func <- para_A21(k1, k2, v, sigma_22, s, s11, ss21, s12, w22)
gamma_A21 <- A21_func[[1]]
d_A21 <- A21_func[[2]]
q21 <- 0.96
tao21 <- 0.5

cat("Calling para_A22...\n")
A22_func <- para_A22(v, sigma_22, s22, ss22, s12, w21)
gamma_A22 <- A22_func[[1]]
d_A22 <- A22_func[[2]]
q22 <- 0.7
tao22 <- 0.5

#============Gibbs Sampler============
for (i in 1:iteration) {
  cat("Gibbs iteration:", i, "\n")
  for (j1 in 1:k1) {
    for (j in (((j1-1)*k1) + 1):(((j1-1)*k1) + k1)) {
      a1 <- sum(b1 * gamma_A11[, j]) - (b1[j] * gamma_A11[j, j]) - d_A11[j]
      a2 <- gamma_A11[j, j] + (1 / ((tao11^2) * sigma_adj[j - ((j1-1)*k1)]))
      pr <- q11 / (q11 + (((1 - q11) * exp(a1^2 / (2 * a2))) / (tao11 * sqrt(a2 * sigma_adj[j - ((j1-1)*k1)]))))
      x1 <- -(a1 / a2)
      x2 <- sqrt(1 / a2)
      b1[j] <- draw(x1, x2, pr, j)
    }
  }
  
  for (l1 in 1:k2) {
    for (l in (((l1-1)*k1) + 1):(((l1-1)*k1) + k1)) {
      a1 <- sum(b2 * gamma_A12[, l]) - (b2[l] * gamma_A12[l, l]) - d_A12[l]
      a2 <- gamma_A12[l, l] + (1 / ((tao12^2) * sigma_adj[l - ((l1-1)*k1)]))
      pr <- q12 / (q12 + (((1 - q12) * exp(a1^2 / (2 * a2))) / (tao12 * sqrt(a2 * sigma_adj[l - ((l1-1)*k1)]))))
      x1 <- -(a1 / a2)
      x2 <- sqrt(1 / a2)
      b2[l] <- draw(x1, x2, pr, l)
    }
  }
  
  for (m1 in 1:k1) {
    for (m in (((m1-1)*k2) + 1):(((m1-1)*k2) + k2)) {
      a1 <- sum(b3 * gamma_A21[, m]) - (b3[m] * gamma_A21[m, m]) - d_A21[m]
      a2 <- gamma_A21[m, m] + (1 / ((tao21^2) * sigma2[m - ((m1-1)*k2)]))
      pr <- q21 / (q21 + (((1 - q21) * exp(a1^2 / (2 * a2))) / (tao21 * sqrt(a2 * sigma2[m - ((m1-1)*k2)]))))
      x1 <- -(a1 / a2)
      x2 <- sqrt(1 / a2)
      b3[m] <- draw(x1, x2, pr, m)
    }
  }
  
  for (n1 in 1:k2) {
    for (n in (((n1-1)*k2) + 1):(((n1-1)*k2) + k2)) {
      a1 <- sum(b4 * gamma_A22[, n]) - (b4[n] * gamma_A22[n, n]) - d_A22[n]
      a2 <- gamma_A22[n, n] + (1 / ((tao22^2) * sigma2[n - ((n1-1)*k2)]))
      pr <- q22 / (q22 + (((1 - q22) * exp(a1^2 / (2 * a2))) / (tao22 * sqrt(a2 * sigma2[n - ((n1-1)*k2)]))))
      x1 <- -(a1 / a2)
      x2 <- sqrt(1 / a2)
      b4[n] <- draw(x1, x2, pr, n)
    }
  }
  
  cat("Calling theta_dist_coeff...\n")
  d_theta <- theta_dist_coeff(s1, s2, s3, Final_Sigma_Inv, b1, b2, b3, b4, k1, k2, vec1)
  theta <- draw_theta(N, d_theta)
  cat("Calling theta_matrix in Gibbs...\n")
  theta_func <- theta_matrix(theta, b1, b2, b3, b4, k1, k2, vec1)
  theta_mat <- theta_func[[1]]
  u <- theta_func[[2]]
  v <- theta_func[[3]]
  w11 <- theta_func[[4]]
  w12 <- theta_func[[5]]
  w21 <- theta_func[[6]]
  w22 <- theta_func[[7]]
  w <- theta_func[[8]]
  w_mod <- theta_func[[9]]
  A11_mat <- matrix(b1, nrow=k1, ncol=k1, byrow=F)
  A12_mat <- matrix(b2, nrow=k1, ncol=k2, byrow=F)
  A_upper <- cbind(A11_mat, A12_mat)
  A21_mat <- matrix(b3, nrow=k2, ncol=k1, byrow=F)
  A22_mat <- matrix(b4, nrow=k2, ncol=k2, byrow=F)
  A_lower <- cbind(A21_mat, A22_mat)
  cat("Calling parameter_sigma...\n")
  sigma_func <- parameter_sigma(k1, k2, nobs, u, Q, df, alpha, beta, data, data_mod, w, w_mod, A21_mat, A22_mat, tao11, tao21, tao22, A_upper, A_lower)
  omega <- sigma_func[[1]]
  sigma2 <- sigma_func[[2]]
  sigma_adj <- sigma_func[[3]]
  cat("Calling sigma_inverse in Gibbs...\n")
  sigma_inv_func <- sigma_inverse(omega, sigma2, k1, k2)
  sigma_11 <- sigma_inv_func[[1]]
  sigma_22 <- sigma_inv_func[[2]]
  sigma_inv <- sigma_inv_func[[3]]
  Final_Sigma_Inv <- sigma_inv_func[[4]]
  sigma_mat <- sigma_inv_func[[5]]
  
  cat("Calling para_A11 in Gibbs...\n")
  A11_func <- para_A11(theta_mat, s, sigma_inv, sigma_11, s11, ss11, s12, w12)
  gamma_A11 <- A11_func[[1]]
  d_A11 <- A11_func[[2]]
  
  cat("Calling para_A12 in Gibbs...\n")
  A12_func <- para_A12(k1, k2, u, sigma_inv, sigma_11, s12, ss12, s22, w11)
  gamma_A12 <- A12_func[[1]]
  d_A12 <- A12_func[[2]]
  
  cat("Calling para_A21 in Gibbs...\n")
  A21_func <- para_A21(k1, k2, v, sigma_22, s, s11, ss21, s12, w22)
  gamma_A21 <- A21_func[[1]]
  d_A21 <- A21_func[[2]]
  
  cat("Calling para_A22 in Gibbs...\n")
  A22_func <- para_A22(v, sigma_22, s22, ss22, s12, w21)
  gamma_A22 <- A22_func[[1]]
  d_A22 <- A22_func[[2]]
  
  if (i > burn) {
    count <- count + 1
    W_sim[, , count] <- w
    theta_sim[, , count] <- theta
    sigma_sim[, , count] <- as.matrix(sigma_mat)
    b1_sim[, , count] <- b1
    b2_sim[, , count] <- b2
    b3_sim[, , count] <- b3
    b4_final <- b4_final + b4
  }
}

# Compute final estimates
w_final <- w_final / count
theta_est <- theta_final / count
sigma_final <- sigma_final / count
b1_final <- b1_final / count
b2_final <- b2_final / count
b3_final <- b3_final / count
b4_final <- b4_final / count
w_est <- apply(W_sim, c(1, 2), mean)
sigma_est <- apply(sigma_sim, c(1, 2), mean)

# Construct estimated A matrix
corrected_b1 <- apply(b1_sim, 1, FUN = function(x) mean(x) * (abs(mean(x)) > 1e-5)) # Zero function approximation
corrected_b2 <- apply(b2_sim, 1, FUN = function(x) mean(x) * (abs(mean(x)) > 1e-5))
corrected_b3 <- apply(b3_sim, 1, FUN = function(x) mean(x) * (abs(mean(x)) > 1e-5))
corrected_b4 <- apply(b4_sim, 1, FUN = function(x) mean(x) * (abs(mean(x)) > 1e-5))
A11_est <- matrix(corrected_b1, nrow=k1, ncol=k1, byrow=F)
A12_est <- matrix(corrected_b2, nrow=k1, ncol=k2, byrow=F)
A21_est <- matrix(corrected_b3, nrow=k2, ncol=k1, byrow=F)
A22_est <- matrix(corrected_b4, nrow=k2, ncol=k2, byrow=F)
A_upper <- cbind(A11_est, A12_est)
A_lower <- cbind(A21_est, A22_est)
A_est <- rbind(A_upper, A_lower)

#==================Predictions==========================
pred <- matrix(0, nrow=p, ncol=nobs)
err <- matrix(0, nrow=p, ncol=nobs)
for (t in 2:nobs) {
  pred[, t] <- as.vector(w_est %*% y_full[, t-1])
  err[, t] <- y_full[, t] - pred[, t]
}

# Estimating error-covariance using graphical lasso
lasso <- CVglasso(X = t(err), S = NULL, nlam = 10, lam.min.ratio = 0.01,
                  lam = NULL, diagonal = FALSE, path = FALSE, tol = 1e-04,
                  maxit = 10000, adjmaxit = NULL, K = 5, crit.cv = "loglik",
                  start = "warm", cores = 1)
cov_mat <- lasso$Sigma

# Forecasting
# Initialize forecast array
pred_sim <- array(0, c(p, horizon, (iteration - burn)))

# Forecasting loop using true values
for (i in 1:(iteration - burn)) {
  pred_y <- matrix(0, nrow = p, ncol = horizon)
  for (t in 1:horizon) {
    # Use true data from previous time point (nobs + t)
    pred_y[, t] <- mvrnorm(1, W_sim[, , i] %*% y_full[, nobs + t], cov_mat)
  }
  pred_sim[, , i] <- pred_y
}

# Compute mean forecast
mean_forecast <- apply(pred_sim, c(1, 2), mean)
lf_indices <- (3 * k1 + 1):(3 * k1 + k2)

# Subset low-frequency variables
mean_forecast_lf <- mean_forecast[lf_indices, , drop = FALSE]  # 9 x 24
forecast_true_lf <- forecast_true[lf_indices, , drop = FALSE]  # 9 x 24

# Check for exploding forecasts
cat("Max absolute value in mean_forecast_lf:", max(abs(mean_forecast_lf)), "\n")

# Define error functions
mean_absolute_error_lf <- function(true, pred) {
  mae_t <- colMeans(abs(true - pred))  # Mean absolute error per time point
  mae <- mean(mae_t)                   # Overall MAE
  return(list(mae_t = mae_t, mae = mae))
}

root_mean_squared_error_lf <- function(true, pred) {
  mse_t <- colMeans((true - pred)^2)   # Mean squared error per time point
  rmse_loss <- sqrt(mean(mse_t))       # Overall RMSE
  return(list(mse_t = mse_t, rmse_loss = rmse_loss))
}

# Compute errors
mae_metrics <- mean_absolute_error_lf(forecast_true_lf, mean_forecast_lf)
rmse_metrics <- root_mean_squared_error_lf(forecast_true_lf, mean_forecast_lf)

# Print results
cat("\n=== Low-Frequency Variables Forecast Results ===\n")
cat('forecast_true_lf',forecast_true_lf)
cat("MAE by time point (low-frequency):\n")
print(mae_metrics$mae_t)
cat("Mean MAE (low-frequency):", mae_metrics$mae, "\n")
cat("RMSE by time point (low-frequency):\n")
print(rmse_metrics$mse_t)
cat("Mean RMSE (low-frequency):", rmse_metrics$rmse_loss, "\n")

# # Optional: Check spectral radius of W_sim to diagnose stability
# library(matrixcalc)
# spectral_radii <- numeric(iteration - burn)
# for (i in 1:(iteration - burn)) {
#   spectral_radii[i] <- max(abs(eigen(W_sim[, , i])$values))
# }
# cat("Mean spectral radius of W_sim:", mean(spectral_radii), "\n")
# cat("Max spectral radius of W_sim:", max(spectral_radii), "\n")