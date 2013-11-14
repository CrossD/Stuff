#!/usr/bin/env python

import sys
import math

# Actually does nothing, just ...

# input comes from STDIN (standard input)
for line in sys.stdin:
    # split the line into (group, value)
    line = line.strip() # Need to remove trailing '\n'
    entries = map(float, line.split('\t'))
    x = entries[0]
    y = entries[1]
    # Put (x, y) into a bin of width 0.1 by 0.1
    xlo = math.floor(x * 10) / 10
    xhi = xlo + 0.1
    ylo = math.floor(y * 10) / 10
    yhi = ylo + 0.1
    print xlo, xhi, ylo, yhi, "\t", 1
