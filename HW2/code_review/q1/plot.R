tmp <- read.table("blb_lin_reg_data_s5_r50_SE.txt", header=TRUE)
plot(1:dim(tmp)[1], tmp[[1]], xlab="Index", ylab="SE")
