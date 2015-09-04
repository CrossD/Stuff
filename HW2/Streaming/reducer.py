#!/usr/bin/env python

# from operator import itemgetter
import sys

current_sum = 0.0
last_group = None
current_count = 1
verbose = False

# input comes from STDIN
for line in sys.stdin:
    # Remove trailing '\n'
    line = line.strip()
    # Extract (key,value)
    vs = line.split('\t')
    # print vs[0]
    current_group = vs[0].strip()
    if last_group == current_group:
        current_count += int(vs[1])
    else:
        if last_group != None:
            print last_group, current_count
        last_group = current_group
        current_count = 1

# Last one:
if last_group != None:
    print last_group, current_count