setwd("~/Stuff/HW2/BLB")
# setwd("D:/Documents/My Documents/Dropbox/Fall_2013/Assignments/STA 250/hw2")

mini <- FALSE
if (mini == FALSE)
    n_orig <- 1000000 else
    n_orig <- 10000
    
#============================== Setup for running on Gauss... ==============================#

args <- commandArgs(TRUE)

# cat("Command-line arguments:\n")
# print(args)

####
# sim_start ==> Lowest possible dataset number
###

###################
sim_start <- 1000
###################

if (length(args)==0){
  sim_num <- sim_start + 1
  set.seed(121231)
} else {
  # SLURM can use either 0- or 1-indexing...
  # Lets use 1-indexing here...
  sim_num <- sim_start + as.numeric(args[1])
  sim_seed <- (762*(sim_num-1) + 121231)
}

cat(paste("\nAnalyzing dataset number ",sim_num,"...\n\n",sep=""))

# Find r and s indices:

#============================== Run the simulation study ==============================#

# Load packages:
library(BH)
library(bigmemory.sri)
library(bigmemory)
library(biganalytics)

# I/O specifications:
if (mini){
    datapath <- "tmp_mini_data"
	rootfilename <- "blb_lin_reg_mini"
} else {
    datapath <- "tmp_data_1" # modify here!
	rootfilename <- "blb_lin_reg_data"
}
outpath <- "output/"

# s and r index number
s_ind <- (sim_num - sim_start - 1) %/% 50 + 1
r_ind <- (sim_num - sim_start) %% 50

# Read data and fit model
dat <- read.csv(sprintf("%s/%d.csv", datapath, s_ind)) # modify nrows!
set.seed(881 * sim_num + 1)
n <- dim(dat)[1]
p <- dim(dat)[2] - 1
weight <- rmultinom(1, n_orig, rep(1/n, n)) # Bootstrap
colnames(dat) <- c(paste0("x", 1:p), "y")
model <- lm(y ~ . - 1, data=dat, weights=weight)

# Output file:
outfile = paste0(outpath, "coef_", s_ind, "_", r_ind, ".txt")

# Save estimates to file:
write.table(model$coefficients, file=outfile)
