#rm(list=ls())
#sink("log.txt")
time1 <- proc.time()

# Load the data
data <- read.csv("/home/victor/YearPredictionMSD.txt", header=F)

# Get the first 324600 (70%) examples for the 90 variables
X <- data[1:324600, c(seq(2,91,by=3))]
X_validation <- data[324601:463715, c(seq(2,91,by=3))]

# Get the first 463715 targets
y <- data[1:324600, 1]
y_validation <- data[324601:463715, 1]

m <- length(y)
m_validation <- length(y_validation)

# Starts to normalize the features
mu <- matrix(0, 1, ncol(X))
sigma <- matrix(0, 1, ncol(X))

mu <- mu + apply(X, 2, mean)
sigma <- sigma + apply(X, 2, sd)

X <- X - mu
X <- X / sigma

mu <- apply(X_validation, 2, mean)
sigma <- apply(X_validation, 2, sd)

X_validation <- X_validation - mu
X_validation <- X_validation - sigma

# Grid
#grid <- seq(0.06, 0.07, 0.002)
#grid_length <- length(grid)

alpha = 0.04

# Starts Gradient Descent
num_iters <- 400
theta = matrix(0, 90, 1)
J_history = matrix(0, num_iters, 1)
X_sum <- apply(X, 2, sum)

#for (i in 1:grid_length) {
  
  #alpha <- grid[i]
  
  for (i in 1:num_iters) {
    cat("\nCurrent iteration: ", i)
    
    theta_transp <- t(theta)
    
    M <- theta_transp * X
    M_col_sum <- apply(M, 2, sum)
    S <- (M_col_sum - y) * X
    #S <- S * X
    
    theta <- theta_transp - ((alpha/m) * X_sum)
    
    # Calculate Cost Function J
    #M <- theta_transp * X
    M_row_sum <- apply(M, 1, sum)
    S <- (M_row_sum - y)^2
    #S <- S^2
    
    J_history[i] <- (sum(S) / (2*m))
  } 
  
  jpeg(paste(paste(alpha, num_iters, sep="_"), "jpg", sep="."))
  plot(J_history, type="o", col="blue")
  dev.off()
#}

predictions <- theta * X_validation
predictions

result <- y_validation - predictions
result

result <- abs(result)
mean(result)

time2 <- proc.time()

time2 - time1
