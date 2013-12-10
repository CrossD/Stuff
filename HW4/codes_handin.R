library(RCUDA)
source("utility.R")

# Load modules and set up kernels
m = loadModule("t3.ptx")
k = m$rtruncnorm_kernel
k_setup <- m$setup_kernel


# 1. (a) (b) (c). 
mu <- 2
sigma <- 1
lo <- 0
hi <- 1.5

GPU_truncnorm <- function(n, mu, sigma, lo, hi, n_seed=2^14L) {
    n <- as.integer(n)
    mu_dev <- copyToDevice(mu)
    sigma_dev <- copyToDevice(sigma)
    lo_dev <- copyToDevice(lo)
    hi_dev <- copyToDevice(hi)
    
    grid_info <- compute_grid(n_seed, sqrt_threads_per_block=32L)
    # Set up random number generators
    grid_seed_info <- compute_grid(n_seed, sqrt_threads_per_block=32L)
    rng_states <- cudaMalloc(elType = "curandState", numEls=n_seed, sizeof=48L) 
    .cuda(k_setup, rng_states, n_seed, gridDim = grid_seed_info$grid_dims, blockDim = grid_seed_info$block_dims)
    
    # Set up output memory and calculate
    vals = cudaMalloc(elType="float", numEls=n, sizeof=4L)
    .cuda(k, vals, n, mu_dev, sigma_dev, lo_dev, hi_dev, length(mu), length(sigma), length(lo), length(hi), rng_states, n_seed, gridDim = grid_info$grid_dims, blockDim = grid_info$block_dims)
    samp <- vals[]
    print("CUDA calculation time:")
    print(proc.time() - tic_cuda)
    
    return(samp)
}

CPU_truncnorm <- function(n, mu, sigma, lo, hi) {
    return(mu + qnorm(pnorm((lo - mu) / sigma) + runif(n) * (pnorm((hi - mu) / sigma) - pnorm((lo - mu) / sigma))))
}

theo_mean <- function(mu, sigma, lo, hi) {
# Calculate the theoretical mean of a truncated normal distribution
    a <- (lo - mu) / sigma
    b <- (hi - mu) / sigma
    Z <- pnorm(b) - pnorm(a)
    return(mu + (dnorm(a) - dnorm(b)) / Z * sigma)
}


# (c). 
samp_GPU <- GPU_truncnorm(1e4, mu, sigma, lo, hi)
s_mean_GPU <- mean(samp_GPU)
t_mean <- theo_mean(mu, sigma, lo, hi)
# > t_mean
# [1] 0.9570067
# > s_mean_GPU
# [1] 0.9560445

# (d). 
samp_CPU <- CPU_truncnorm(1e4, mu, sigma, lo, hi)
s_mean_CPU <- mean(samp_CPU)
t_mean <- theo_mean(mu, sigma, lo, hi)
# > s_mean_CPU
# [1] 0.9620902
# > t_mean
# [1] 0.9570067

# (e). 
k_list <- 1:8
cpu_time <- cuda_time <- matrix(NA, length(k_list), 3)
# GPU time:
for (i in k_list) {
    cuda_time[i, ] <- as.vector(system.time(
        samp_GPU <- GPU_truncnorm(10^i, mu, sigma, lo, hi)
    )[1:3])
}
# CPU time:
for (i in k_list) {
    cpu_time[i, ] <- as.vector(system.time(
        samp_CPU <- CPU_truncnorm(10^i, mu, sigma, lo, hi)
    )[1:3])
}

pdf(file="1_e.pdf")
par(cex=1.3)
t1.e <- matrix(c(0.098,  0.000, 
                 0.100,  0.000, 
                 0.100,  0.000, 
                 0.100,  0.001, 
                 0.100,  0.012, 
                 0.150,  0.124, 
                 0.246,  1.220, 
                 0.799, 12.324), 8, 2, byrow=TRUE)
matplot(y=t1.e, x=1:dim(t1.e)[1], main="Truncated Normal Sampling Runtime", xlab=expression(log[10](n)), ylab="Seconds", type="l", pch=c(1,2), lwd=2)
matplot(y=t1.e, x=1:dim(t1.e)[1], type="p", add=T, pch=c(1, 2))
legend(1.2, 10, c("CUDA", "R"), lty=1:2, pch=1:2, col=1:2)
dev.off()

# (f) (g). a = -Inf, b= -10 and a = 10, b = Inf:
pdf("f.pdf", width=12)
par(mfrow=c(1, 2))
s1 <- GPU_truncnorm(1e4, 0, 1, -Inf, -10)
s2 <- GPU_truncnorm(1e4, 0, 1, 10, Inf)
hist(s1, title="GPU, lo=-Inf, hi=-10")
hist(s2, title="GPU, lo=10, hi=Inf")
dev.off()

mean(s1)
theo_mean(0, 1, -Inf, -10)
# > mean(s1)
# [1] -10.09868
# > theo_mean(0, 1, -Inf, -10)
# [1] -10.09809
mean(s2)
-theo_mean(0, 1, -Inf, -10)
# > mean(s2)
# [1] 10.09868
# > -theo_mean(0, 1, -Inf, -10)
# [1] 10.09809

# Also works for a = -5, b = -3:
pdf("double_trunc.pdf")
hist(GPU_truncnorm(1e4, 0, 1, -5, -3))
dev.off()


# 2. (a). 
# source("sim_probit.R")
library(mvtnorm)
library(MCMCpack)


probit_mcmc <- function
                  (y,           # vector of length n 
                   X,           # (n x p) design matrix
                   beta_0,      # (p x 1) prior mean
                   Sigma_0_inv, # (p x p) prior precision 
                   n_iter,      # number of post burnin iterations
                   burnin,      # number of burnin iterations
                   use="CPU",   # "CPU" or "GPU"
                   timing=TRUE, # print timing or not
                   n_seed=as.integer(2^14L))# Number of seeds to setup for CURAND
# Propose beta first and then propose z                  
{
    tic <- proc.time()
    r_calc_time <- rep(0, 3) # For storing R calculation time
    cuda_calc_time <- rep(0, 3) # For storing CUDA calculation time
    
    # Initialize
    n <- as.integer(dim(X)[1])
    p <- as.integer(dim(X)[2])
    samples <- matrix(NA, n_iter, p)
    beta_samp <- rep(0, p)
    
    # Check conditions
    if (!is.matrix(beta_0))
        beta_0 <- as.matrix(beta_0)
    
    # Iteration 0
    tic_calc <- proc.time()
    Sigma_inv_beta_0 <- Sigma_0_inv %*% beta_0
    z <- as.matrix(rep(0, n))
    lo <- ifelse(y==1, 0, -Inf)
    hi <- ifelse(y==1, Inf, 0)
    Sigma_1_inv <- crossprod(X) + Sigma_0_inv
    Sigma_1 <- solve(Sigma_1_inv)
    r_calc_time <- r_calc_time + (proc.time() - tic_calc)[0:3]

    
    # GPU setup if necessary
    if (use == "GPU") {
        grid_info <- compute_grid(n_seed, sqrt_threads_per_block=32L)
        grid_seed_info <- compute_grid(n_seed, sqrt_threads_per_block=32L)
        rng_states <- cudaMalloc(elType = "curandState", numEls=n_seed, sizeof=48L) 
        .cuda(k_setup, rng_states, n_seed, gridDim = grid_seed_info$grid_dims, blockDim = grid_seed_info$block_dims)
        vals = cudaMalloc(elType="float", numEls=n, sizeof=4L)
        sigma_dev <- copyToDevice(1)
        lo_dev <- copyToDevice(lo)
        hi_dev <- copyToDevice(hi)
    }
    
    # Burnin period
    for (i in 1:burnin) {
        tic_calc <- proc.time()
        beta_1 <- Sigma_1 %*% (crossprod(X, z) + Sigma_inv_beta_0)
        beta_samp <- t(rmvnorm(1, beta_1, Sigma_1))
        r_calc_time <- r_calc_time + (proc.time() - tic_calc)[0:3]
        if (use == "CPU") {
            tic_calc <- proc.time()
            mu <- X %*% beta_samp
            z <- as.matrix(CPU_truncnorm(n, mu, 1, lo, hi))
            r_calc_time <- r_calc_time + (proc.time() - tic_calc)[0:3]
        }    
        else {
            tic_calc <- proc.time()
            mu <- X %*% beta_samp
            r_calc_time <- r_calc_time + (proc.time() - tic_calc)[0:3]
            mu_dev <- copyToDevice(mu)
            tic_calc <- proc.time()
            .cuda(k, vals, n, mu_dev, sigma_dev, lo_dev, hi_dev, n, 1L, n, n, rng_states, n_seed, gridDim = grid_info$grid_dims, blockDim = grid_info$block_dims)
            cuda_calc_time <- cuda_calc_time + (proc.time() - tic_calc)[0:3]
            z <- as.matrix(vals[])
        }
    }

    # Sampling period
    for (i in 1:n_iter) {
        tic_calc <- proc.time()
        beta_1 <- Sigma_1 %*% (crossprod(X, z) + Sigma_inv_beta_0)
        beta_samp <- t(rmvnorm(1, beta_1, Sigma_1))
        r_calc_time <- r_calc_time + (proc.time() - tic_calc)[0:3]
        if (use == "CPU") {
            tic_calc <- proc.time()
            mu <- X %*% beta_samp
            z <- as.matrix(CPU_truncnorm(n, mu, 1, lo, hi))
            r_calc_time <- r_calc_time + (proc.time() - tic_calc)[0:3]
        }    
        else {
            tic_calc <- proc.time()
            mu <- X %*% beta_samp
            r_calc_time <- r_calc_time + (proc.time() - tic_calc)[0:3]
            mu_dev <- copyToDevice(mu)
            tic_calc <- proc.time()
            .cuda(k, vals, n, mu_dev, sigma_dev, lo_dev, hi_dev, n, 1L, n, n, rng_states, n_seed, gridDim = grid_info$grid_dims, blockDim = grid_info$block_dims)
            cuda_calc_time <- cuda_calc_time + (proc.time() - tic_calc)[0:3]
            z <- as.matrix(vals[])
        }
        samples[i, ] <- t(beta_samp)
    }
    
    total_time <- proc.time() - tic
    if (timing) 
        print(total_time)
    
    res <- list(total_time=total_time[0:3], 
                r_calc_time=r_calc_time,
                cuda_calc_time=cuda_calc_time, 
                samp=mcmc(samples))
    return(res)
}

# (c). 
dat <- read.table("mini_data.txt", header=TRUE)
para <- read.table("mini_pars.txt", header=TRUE)

X <- as.matrix(dat[, (-1)])
y <- as.integer(dat[, 1])
n <- as.integer(dim(X)[1])
p <- as.integer(dim(X)[2])
n_iter <- 100000
n_burn <- 500

samp_mcmc_cpu <- probit_mcmc(y, X, rep(0, p), diag(0, p), n_iter, n_burn) 
samp_mcmc_gpu <- probit_mcmc(y, X, rep(0, p), diag(0, p), n_iter, n_burn, use="GPU") 
colMeans(samp_mcmc_cpu$samp)
m_glm <- glm(y ~ . - 1, dat, family=binomial(link="probit"))
m_glm$coefficients
unname(unlist(para))

# (d).
i_list <- 1:5
cpu_time_mcmc <- cuda_time_mcmc <- matrix(NA, length(i_list), 3)
cpu_r_calc_time_mcmc <- cuda_r_calc_time_mcmc <- matrix(NA, length(i_list), 3)
cuda_calc_time_mcmc <- matrix(NA, length(i_list), 3)
# GPU time:
for (i in i_list) {
    file_name <- sprintf("data_0%d.txt", i)
    dat <- read.table(file_name, header=TRUE)
    X <- as.matrix(dat[, (-1)])
    y <- as.integer(dat[, 1])
    n <- as.integer(dim(X)[1])
    p <- as.integer(dim(X)[2])
    samp_CPU_mcmc <- probit_mcmc(y, X, rep(0, p), diag(0, p), n_iter, n_burn, use="CPU")
    cpu_time_mcmc[i, ] <- samp_CPU_mcmc$total_time
    cpu_r_calc_time_mcmc[i, ] <- samp_CPU_mcmc$r_calc_time    
    samp_GPU_mcmc <- probit_mcmc(y, X, rep(0, p), diag(0, p), n_iter, n_burn, use="GPU")
    cuda_time_mcmc[i, ] <- samp_GPU_mcmc$total_time
    cuda_calc_time_mcmc[i, ] <- samp_GPU_mcmc$cuda_calc_time
    cuda_r_calc_time_mcmc[i, ] <- samp_GPU_mcmc$r_calc_time
}

load("time_200.RData")
pdf(file="2_d.pdf", width=10, height=5)
par(mfrow=c(1, 2), cex=1.3)
# Left
matplot(y=cbind(cuda_time_mcmc[, 3], cpu_time_mcmc[, 3]), x=3:7, main="Total Run Time", xlab=expression(log[10](n)), ylab="Seconds", type="l", pch=c(1,2), lwd=2)
matplot(y=cbind(cuda_time_mcmc[, 3], cpu_time_mcmc[, 3]), x=3:7, type="p", add=T, pch=c(1, 2))
legend(3.2, 1500, c("CUDA", "R"), lty=1:2, pch=1:2, col=1:2)
# Right
matplot(y=cbind(cuda_calc_time_mcmc[, 3], cuda_r_calc_time_mcmc[, 3]), x=3:7, main="TN Sampling v.s. the Remaining", xlab=expression(log[10](n)), ylab="Seconds", type="l", pch=c(1,2), lwd=2)
matplot(y=cbind(cuda_calc_time_mcmc[, 3], cuda_r_calc_time_mcmc[, 3]), x=3:7, type="p", add=T, pch=c(1, 2))
legend(3.2, 120, c("TN", "Other"), lty=1:2, pch=1:2, col=1:2)
dev.off()