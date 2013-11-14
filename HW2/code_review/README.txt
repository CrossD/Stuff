Q1: 
1. run BLB_pre_process.sh, which calls BLB_pre_process.R
2. run BLB_lin_reg_run_R.sh, which calls BLB_lin_reg_job.R
3. run BLB_lin_reg_process.R, which calls BLB_lin_reg_process.R

Q2:
See mapper.py and reducer.py

Q3:
1. Run q3.sql inside hive. 
2. Run "hive -e 'SELECT grp, AVG(val), VAR_SAMP(val) FROM full GROUP BY grp;' > full_res.csv" in bash
3. Run q3.R in R