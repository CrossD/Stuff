# This script is to devide the big file into s small files each with n^0.7 observations
s <- 5
n <- 1000000
n.power <- 0.7

setwd("~/Stuff/HW2/BLB")
mini <- FALSE
# Install packages:
# if (Sys.info()["sysname"] != "Windows")
    # install.packages(c("BH", "bigmemory.sri", "bigmemory", "biganalytics"), repos="http://cran.cnr.Berkeley.edu")

# Load packages:
library(BH)
library(bigmemory.sri)
library(bigmemory)
library(biganalytics)

# I/O specifications:
datapath <- "/home/pdbaines/data/"
outdatapath <- "tmp_data/"

# mini or full?
if (mini){
	rootfilename <- "blb_lin_reg_mini"
    outdatapath <- "tmp_mini_data/"
} else {
	rootfilename <- "blb_lin_reg_data"
    outdatapath <- "tmp_data/"
}

dat <- attach.big.matrix(paste0(datapath, rootfilename, ".desc"))
set.seed(1)
# Make s subsets of data, each of size n^0.7
for (i in 1:s) {
    ind <- sort(sample(n, n^n.power))
    write.csv(dat[ind, ], paste0(outdatapath, "/", i, ".csv"), row.names=FALSE)
}

# Filenames:
# Set up I/O stuff:
# Attach big.matrix :
# Remaining BLB specs:
# Extract the subset:
# Reset simulation seed:
# Bootstrap dataset:
# Fit lm:
# Output file:
# Save estimates to file:

