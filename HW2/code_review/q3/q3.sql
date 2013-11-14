CREATE TABLE IF NOT EXISTS mini(
grp INT,
val FLOAT) 
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t';

LOAD DATA LOCAL INPATH '/home/hadoop/q3/mini.txt' INTO TABLE mini;

SELECT grp, AVG(val), VAR_SAMP(val)
FROM mini
GROUP BY grp;

CREATE TABLE IF NOT EXISTS full(
grp INT,
val FLOAT) 
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t';

LOAD DATA LOCAL INPATH '/home/hadoop/q3/group.txt' INTO TABLE full;

SELECT grp, AVG(val), VAR_SAMP(val)
FROM full
GROUP BY grp;

-- hive -e 'SELECT grp, AVG(val), VAR_SAMP(val) FROM full GROUP BY grp;' > full_res.csv