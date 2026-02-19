# Load required libraries
library(matrixcalc)
library(clusterGeneration)
library(fBasics)
library(glmnet)
library(psych)
library(MASS)
library(Matrix)
library(mfbvar)
library(scoringRules)

# Source custom functions
source("data_gen.R")
source("functions.R")

#==========Input Parameters=============
k1 <- 76       # Number of monthly variables
k2 <- 9        # Number of quarterly variables
nobs <- 215    # Number of training observations
test_size <- 24 # Number of testing/forecast observations
horizon <- test_size # Forecast horizon = 24
repl <- 50     # Number of replicates
p <- (3 * k1) + k2  # Total variables: 3*76 + 9 = 237
p1 <- k1 + k2       # Monthly + quarterly: 76 + 9 = 85
total_time <- nobs + test_size + 1 # 240 time points

#===========Load Data================
# Read the pre-generated data from CSV
data_full <- as.matrix(read.csv("D:/Code/MF/data/data3_196001_202503_76_9.csv")[,-1]) # Remove index column

# Restrict data to first 240 time points
data_full <- data_full[, 1:total_time]

# Debug: Check dimensions
cat("Dimensions of data_full:", dim(data_full), "\n")
if (nrow(data_full) != p) stop("data_full does not have p = ", p, " rows")
if (ncol(data_full) != total_time) stop("data_full has ", ncol(data_full), " columns, expected ", total_time)

# Define permutation vector
vec1 <- c((k1 + 1):(k1 + k2), 1:k1) # Permutation: quarterly then monthly
cat("Length of vec1:", length(vec1), ", Expected:", p1, "\n")

# Prepare data
y_full <- data_full
y_mod_full <- y_full[vec1, ]
data_mod <- as.matrix(y_mod_full)[, 1:(nobs + 1)] # Training data: first 216 time points
forecast_true_m1 <- y_full[, (nobs + 2):(nobs + 1 + horizon)] # Test data: 217 to 240

# Debug: Check dimensions
cat("Dimensions of y_full:", dim(y_full), "\n")
cat("Dimensions of y_mod_full:", dim(y_mod_full), "\n")
cat("Dimensions of data_mod:", dim(data_mod), "\n")
cat("Dimensions of forecast_true_m1:", dim(forecast_true_m1), "\n")

# Initialize arrays for predictions and evaluation
pred1 <- array(0, dim = c((3 * (k1 + k2) * horizon), 3, repl))  # Minnesota prior
pred2 <- array(0, dim = c((3 * (k1 + k2) * horizon), 3, repl))  # Steady-state prior
pred3 <- array(0, dim = c((3 * (k1 + k2) * horizon), 3, repl))  # Hierarchical SS prior
crps_minn <- array(0, dim = c(k2, 3 * horizon, repl))
crps_ss <- array(0, dim = c(k2, 3 * horizon, repl))
crps_ssng <- array(0, dim = c(k2, 3 * horizon, repl))
logs_minn <- array(0, dim = c(k2, 3 * horizon, repl))
logs_ss <- array(0, dim = c(k2, 3 * horizon, repl))
logs_ssng <- array(0, dim = c(k2, 3 * horizon, repl))
ferr_mfb_low1 <- array(0, dim = c(3 * horizon, 1, repl))
ferr_mfb_low2 <- array(0, dim = c(3 * horizon, 1, repl))
ferr_mfb_low3 <- array(0, dim = c(3 * horizon, 1, repl))

#===========Main Loop over Replicates===========
for (r in 1:repl) {
  cat("Processing replicate:", r, "\n")
  
  # MFBVAR model fitting
  arr <- mfb_rearr(k1, k2, horizon)
  mfb_compare <- mfb(data_mod, k1, k2, horizon, arr, forecast_true_m1)
  
  # Store predictions and evaluation metrics
  pred1[, , r] <- as.matrix(mfb_compare$pred_1)  # Minnesota prior
  crps_minn[, , r] <- mfb_compare$crps_minn
  logs_minn[, , r] <- mfb_compare$logs_minn
  ferr_mfb_low1[, , r] <- mfb_compare$ferr_mfb_low1
  
  pred2[, , r] <- as.matrix(mfb_compare$pred_2)  # Steady-state prior
  crps_ss[, , r] <- mfb_compare$crps_ss
  logs_ss[, , r] <- mfb_compare$logs_ss
  ferr_mfb_low2[, , r] <- mfb_compare$ferr_mfb_low2
  
  pred3[, , r] <- as.matrix(mfb_compare$pred_3)  # Hierarchical SS prior
  crps_ssng[, , r] <- mfb_compare$crps_ssng
  logs_ssng[, , r] <- mfb_compare$logs_ssng
  ferr_mfb_low3[, , r] <- mfb_compare$ferr_mfb_low3
  
  # Save evaluation metrics
  save(crps_minn, file = paste("crps_minn_test-", r, ".dat", sep = ""))
  save(logs_minn, file = paste("logs_minn_test-", r, ".dat", sep = ""))
  save(crps_ss, file = paste("crps_ss_test-", r, ".dat", sep = ""))
  save(logs_ss, file = paste("logs_ss_test-", r, ".dat", sep = ""))
  save(crps_ssng, file = paste("crps_ssng_test-", r, ".dat", sep = ""))
  save(logs_ssng, file = paste("logs_ssng_test-", r, ".dat", sep = ""))
  save(ferr_mfb_low1, file = paste("ferr_low_minn_test-", r, ".dat", sep = ""))
  save(ferr_mfb_low2, file = paste("ferr_low_ss_test-", r, ".dat", sep = ""))
  save(ferr_mfb_low3, file = paste("ferr_low_ssng_test-", r, ".dat", sep = ""))
}

#===========Combine Results Across Replicates===========
val_minn <- val_ss <- val_ssng <- NULL
avg_crps_minn <- avg_logs_minn <- NULL
avg_crps_ss <- avg_logs_ss <- NULL
avg_crps_ssng <- avg_logs_ssng <- NULL

for (r in 1:repl) {
  # Load forecast errors
  load(paste("ferr_low_minn_test-", r, ".dat", sep = ""))
  load(paste("ferr_low_ss_test-", r, ".dat", sep = ""))
  load(paste("ferr_low_ssng_test-", r, ".dat", sep = ""))
  err_minn <- ferr_mfb_low1[, , 1]
  err_ss <- ferr_mfb_low2[, , 1]
  err_ssng <- ferr_mfb_low3[, , 1]
  val_minn <- rbind(val_minn, err_minn)
  val_ss <- rbind(val_ss, err_ss)
  val_ssng <- rbind(val_ssng, err_ssng)
  
  # Load CRPS and LPS
  load(paste("crps_minn_test-", r, ".dat", sep = ""))
  load(paste("logs_minn_test-", r, ".dat", sep = ""))
  load(paste("crps_ss_test-", r, ".dat", sep = ""))
  load(paste("logs_ss_test-", r, ".dat", sep = ""))
  load(paste("crps_ssng_test-", r, ".dat", sep = ""))
  load(paste("logs_ssng_test-", r, ".dat", sep = ""))
  
  # Compute averages per horizon
  minn_crps <- minn_logs <- numeric(3 * horizon)
  for (j in 1:(3 * horizon)) {
    minn_crps[j] <- mean(crps_minn[, j, 1])
    minn_logs[j] <- mean(logs_minn[, j, 1])
  }
  avg_crps_minn <- rbind(avg_crps_minn, minn_crps)
  avg_logs_minn <- rbind(avg_logs_minn, minn_logs)
  
  ss_crps <- ss_logs <- numeric(3 * horizon)
  for (j in 1:(3 * horizon)) {
    ss_crps[j] <- mean(crps_ss[, j, 1])
    ss_logs[j] <- mean(logs_ss[, j, 1])
  }
  avg_crps_ss <- rbind(avg_crps_ss, ss_crps)
  avg_logs_ss <- rbind(avg_logs_ss, ss_logs)
  
  ssng_crps <- ssng_logs <- numeric(3 * horizon)
  for (j in 1:(3 * horizon)) {
    ssng_crps[j] <- mean(crps_ssng[, j, 1])
    ssng_logs[j] <- mean(logs_ssng[, j, 1])
  }
  avg_crps_ssng <- rbind(avg_crps_ssng, ssng_crps)
  avg_logs_ssng <- rbind(avg_logs_ssng, ssng_logs)
}

# Compute final metrics
minn <- apply(val_minn, 2, mean)
ss <- apply(val_ss, 2, mean)
ssng <- apply(val_ssng, 2, mean)
RMSE_minn <- sqrt(minn)
RMSE_ss <- sqrt(ss)
RMSE_ssng <- sqrt(ssng)
final_crps_minn <- apply(avg_crps_minn, 2, mean)
final_logs_minn <- apply(avg_logs_minn, 2, mean)
final_crps_ss <- apply(avg_crps_ss, 2, mean)
final_logs_ss <- apply(avg_logs_ss, 2, mean)
final_crps_ssng <- apply(avg_crps_ssng, 2, mean)
final_logs_ssng <- apply(avg_logs_ssng, 2, mean)

# Combine results
crps <- cbind(final_crps_minn, final_crps_ss, final_crps_ssng)
logs <- cbind(final_logs_minn, final_logs_ss, final_logs_ssng)

# Print results
cat("\n=== MFBVAR Forecast Results ===\n")
cat("RMSE (Minnesota):\n", RMSE_minn, "\n")
cat("RMSE (Steady-state):\n", RMSE_ss, "\n")
cat("RMSE (Hierarchical SS):\n", RMSE_ssng, "\n")
cat("CRPS (Minnesota):\n", final_crps_minn, "\n")
cat("CRPS (Steady-state):\n", final_crps_ss, "\n")
cat("CRPS (Hierarchical SS):\n", final_crps_ssng, "\n")
cat("LPS (Minnesota):\n", final_logs_minn, "\n")
cat("LPS (Steady-state):\n", final_logs_ss, "\n")
cat("LPS (Hierarchical SS):\n", final_logs_ssng, "\n")

# Save final results
# save(RMSE_minn, file = "RMSE_minn_final.dat")
# save(RMSE_ss, file = "RMSE_ss_final.dat")
# save(RMSE_ssng, file = "RMSE_ssng_final.dat")
# save(crps, file = "crps_final.dat")
# save(logs, file = "logs_final.dat")