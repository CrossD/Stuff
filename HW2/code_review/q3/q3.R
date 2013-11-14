# Mean vs Variance plot

res <- read.table("full_res.tsv", sep="\t", row.names=1)
colnames(res) <- c("mean", "var")
plot(res$mean, res$var, type="p", xlab="Group Mean", ylab="Group Variance")