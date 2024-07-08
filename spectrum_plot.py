import matplotlib.pyplot as plt
import csv
import numpy as np
from utils import *
import os
from pathlib import Path


cache_policies_single = ['CLAM', 'PRL']
cache_policies_multi = ['SHEL', 'C-SHEL']
data_sizes = ['medium']
benchmarks_single = ['2mm', '3mm', 'adi', 'atax', 'bicg', 'cholesky', 'correlation', 'covariance', 'deriche', 'doitgen', 'durbin', 'fdtd-2d', 'floyd-warshall', 
              'gemm', 'gemver', 'gesummv', 'gramschmidt', 'heat-3d', 'jacobi-1d', 'jacobi-2d', 'lu', 'ludcmp', 'mvt', 'nussinov', 'seidel-2d',
              'symm', 'syr2k', 'syrk', 'trisolv', 'trmm']

benchmarks_multi = ['2mm','3mm','adi','correlation','covariance','deriche','fdtd-2d','gemver','heat-3d','jacobi-2d','lu','ludcmp','mvt']

for policy in cache_policies_single:
    for size in data_sizes:
        for benchmark in benchmarks_single:
            spec_plot(policy, size, benchmark, 1)

for policy in cache_policies_multi:
    for size in data_sizes:
        for benchmark in benchmarks_multi:
            spec_plot(policy, size, benchmark, 1)
