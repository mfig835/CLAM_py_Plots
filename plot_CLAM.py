import matplotlib.pyplot as plt
import csv
import numpy as np
from utils import *


#fig, ax = plt.subplots()             # Create a figure containing a single Axes.
#ax.plot([1, 2, 3, 4], [1, 4, 2, 3])  # Plot some data on the Axes.
#plt.show()                           # Show the figure.

cache_policies = ['CLAM', 'SHEL', 'C-SHEL', 'PRL']
data_sizes = ['small', 'medium', 'large']

num_policies = len(cache_policies)
num_sizes = len(data_sizes)

single_scope_benchmarks = ['atax', 'bicg', 'cholesky', 'doitgen', 'durbin', 'floyd-warshall', 'gemm', 'gesummv', 
                           'gramschmidt', 'jacobi-1d', 'nussinov', 'seidel-2d', 'symm', 'syr2k', 'syrk', 'trisolv', 'trmm']

single_scope_benchmarks_misses = [[[0 for _ in range(len(single_scope_benchmarks))]for i in range (num_policies)] for i in range (num_sizes)]#[[[0] * 17] * 4 ] * 3                         #empty 4 x 4 x 17 array (4 policies for each bechmark, 3 sizes)
single_scope_benchmarks_hits = [[[0 for _ in range(len(single_scope_benchmarks))]for i in range (num_policies)] for i in range (num_sizes)]
single_scope_benchmarks_miss_ratio = [[[0 for _ in range(len(single_scope_benchmarks))]for i in range (num_policies)] for i in range (num_sizes)]


multi_scope_benchmarks = ['2mm', '3mm', 'adi', 'correlation', 'covariance', 'deriche', 'fdtd-2d', 'gemver', 'heat-3d', 'jacobi-2d', 'lu', 'ludcmp', 'mvt']

multi_scope_benchmarks_misses = [[[0 for _ in range(len(multi_scope_benchmarks))]for i in range (num_policies)] for i in range (num_sizes)]                         #empty 4 x 4 x 17 array (4 policies for each bechmark, 3 sizes)
multi_scope_benchmarks_hits = [[[0 for _ in range(len(multi_scope_benchmarks))]for i in range (num_policies)] for i in range (num_sizes)]
multi_scope_benchmarks_miss_ratio = [[[0 for _ in range(len(multi_scope_benchmarks))]for i in range (num_policies)] for i in range (num_sizes)]


num_single_scopes = len(single_scope_benchmarks)
num_multi_scopes = len(multi_scope_benchmarks)

plru_single_misses = [0] * 17
plru_multi_misses = [0] * 13

plru_single_hits = [0] * 17
plru_multi_hits = [0] * 13

plru_single_miss_ratio = [0] * 17
plru_multi_miss_ratio = [0] * 13

lines = 0
with open('data/results_large_multi_level.txt', newline='') as csvfile: #Open results file 
        data = csv.reader(csvfile, delimiter=',', quotechar='|')            #Use python csv package to store results in data (2d array of strings)
        for row in data:                                                    #iterate through data
            single_flag = 0                                                 #reset these every iteration
            multi_flag = 0


            path_segments = row[0].split('/')                               #split first field of data line into segments (path_segments[2] = POLICY_SIZE, policy_segments[3] = benchmark name)
            policy_size = path_segments[2].split('_')                       #policy_size[0] = POLICY, policy_size[1] = SIZE
            
            if(lines < 30):                                                 #If in the first 30 lines, looking for baseline PLRU data
                
                if path_segments[3] in single_scope_benchmarks:
                    single_ind = find_index_of_corresponding(path_segments[3], single_scope_benchmarks, num_single_scopes)
                    plru_single_misses[single_ind] = row[11]
                    plru_single_hits[single_ind] = row[10]
                    plru_single_miss_ratio[single_ind] = int(row[11])/(int(row[11])+int(row[10]))
                else:
                    multi_ind = find_index_of_corresponding(path_segments[3], multi_scope_benchmarks, num_multi_scopes)
                    plru_multi_misses[multi_ind] = row[11]
                    plru_multi_hits[multi_ind] = row[10]
                    plru_multi_miss_ratio[multi_ind] = int(row[11])/(int(row[11])+int(row[10]))
                lines += 1
            else:
                size_ind = find_index_of_corresponding(policy_size[1], data_sizes, num_sizes)#sizes will be small, medium, large. I'm changing the data format names because they're dumb as shit

                policy_ind = single_ind = find_index_of_corresponding(policy_size[0], cache_policies, num_policies)#Policies are CLAM, SHEL, C-SHEL, PRL

                if path_segments[3] in single_scope_benchmarks:
                    single_flag = 1
                    single_ind = find_index_of_corresponding(path_segments[3], single_scope_benchmarks, num_single_scopes)
                else:
                    multi_flag = 1
                    multi_ind =find_index_of_corresponding(path_segments[3], multi_scope_benchmarks, num_multi_scopes)

                if single_flag == 1:                                            #Compile misses into massive miss holding 3D array. row[11] holds misses for multi-level
                    single_scope_benchmarks_misses[size_ind][policy_ind][single_ind] = row[11]
                    single_scope_benchmarks_hits[size_ind][policy_ind][single_ind] = row[10]
                    single_scope_benchmarks_miss_ratio[size_ind][policy_ind][single_ind] = int(row[11])/(int(row[11])+int(row[10]))
                elif multi_flag == 1:
                    multi_scope_benchmarks_misses[size_ind][policy_ind][multi_ind] = row[11]
                    multi_scope_benchmarks_hits[size_ind][policy_ind][multi_ind] = row[10]
                    multi_scope_benchmarks_miss_ratio[size_ind][policy_ind][multi_ind] = int(row[11])/(int(row[11])+int(row[10]))

#^------------------------------------------------------------------------------------------------------------------------------------GET DATA

i = 0
j = 0
k = 0

single_norm_miss_points = [[[]for i in range (num_policies)] for i in range (num_sizes)]# creates an nested list such that a[num_sizes][num_policies][]
single_hit_points = [[[]for i in range (num_policies)] for i in range (num_sizes)]
single_bench_points = [[[]for i in range (num_policies)] for i in range (num_sizes)]
single_ratio_points = [[[]for i in range (num_policies)] for i in range (num_sizes)]


multi_norm_miss_points = [[[]for i in range (num_policies)] for i in range (num_sizes)]
multi_hit_points = [[[]for i in range (num_policies)] for i in range (num_sizes)]
multi_bench_points = [[[]for i in range (num_policies)] for i in range (num_sizes)]#^These empty arrays will be filled over time in the next set of nested for loops and hold the actual data which will be plotted
multi_ratio_points = [[[]for i in range (num_policies)] for i in range (num_sizes)]


for i in range(num_sizes) :#small, medium, large

    for j in range(num_policies): #CLAM, SHEL, C-SHEL, PRL

        #single scope
        for k in range(num_single_scopes):
            if single_scope_benchmarks_misses[i][j][k] != 0: #If this specific benchmark was in the dataset
                single_norm_miss_points[i][j].append(int(single_scope_benchmarks_misses[i][j][k])/int(plru_single_misses[k])) #generate plotable points
                single_hit_points[i][j].append(single_scope_benchmarks_hits[i][j][k])
                single_ratio_points[i][j].append(single_scope_benchmarks_miss_ratio[i][j][k])
                single_bench_points[i][j].append(single_scope_benchmarks[k])


        #multi scope
        for k in range(num_multi_scopes):
            if multi_scope_benchmarks_misses[i][j][k] != 0: #If this specific benchmark was in the dataset
                multi_norm_miss_points[i][j].append(int(multi_scope_benchmarks_misses[i][j][k])/int(plru_multi_misses[k])) #generate plotable points
                multi_hit_points[i][j].append(multi_scope_benchmarks_hits[i][j][k])
                multi_ratio_points[i][j].append(multi_scope_benchmarks_miss_ratio[i][j][k])
                multi_bench_points[i][j].append(multi_scope_benchmarks[k])

#print(single_norm_miss_points[i][k])
#^------------------------------------------------------------------------------------------------------------------------------------SORT DATA

#Plot Large (0-CLAM, 1-SHEL, 2-C-SHEL, 3-PRL)
fig1, ax1 = plt.subplots()
CLAM_large_misses = ax1.bar(multi_bench_points[2][0], multi_norm_miss_points[2][0])
ax1.axhline(y = 1, color = 'b', linestyle = '-')
ax1.set_xlabel('Names of Benchmarks')
ax1.set_ylabel('Misses Normalized to PLRU')
ax1.set_title('Multi Scope Benchmarks Normalized Miss Count')
ax1.grid(True, linewidth = "0.5", axis = 'y' )
ax1.legend([CLAM_large_misses], ['CLAM'])

plt.show()

#^CLAM Large Normalized Misses

#PARAMS---------------------------------------------------
ylim = 1.2
width = 0.2
x = np.arange(len(multi_bench_points[2][0]))
#---------------------------------------------------------


normed_misses = {
    'CLAM':     multi_norm_miss_points[2][0],
    'SHEL':     multi_norm_miss_points[2][1],
    'C-SHEL':    multi_norm_miss_points[2][2],
    'PRL':      multi_norm_miss_points[2][3],
}


giga = gen_graph(normed_misses, width, ylim, x)


for i, label in enumerate((multi_bench_points[2][0])):
    giga[1].text(x[i], -0.08, plru_multi_misses[i], ha='center', va='center', rotation=45)

giga[1].axhline(y = 1, color = 'b', linestyle = '-')
giga[1].set_xlabel('Names of Benchmarks')
giga[1].xaxis.set_label_coords(.5, -0.11)
giga[1].set_ylabel('Misses Normalized to PLRU')
giga[1].set_xticks(x + width + 0.1, multi_bench_points[2][0])#Add 0.1 to perfectly align ticks
giga[1].set_title('Normalized Miss Counts for Multi-Scoped Benchmarks')
giga[1].grid(True, linewidth = "0.5", axis = 'y' )
giga[1].legend(loc = 'upper right')
giga[1].set_ylim(0, ylim)

plt.show()

#--------------------------------------------------------------------------------------------------------

#PARAMS---------------------------------------------------
ylim = 1.2
width = 0.3
x = np.arange(len(single_bench_points[2][0]))
#---------------------------------------------------------

miss_ratio = {
    'CLAM':     single_norm_miss_points[2][0],
    #'SHEL':     single_norm_miss_points[2][1],
    #'C-SHEL':   single_norm_miss_points[2][2],
    'PRL':      single_norm_miss_points[2][3],
}


giga = gen_graph(miss_ratio, width, ylim, x)


for i, label in enumerate((single_bench_points[2][0])):
    giga[1].text(x[i], -0.08, plru_single_misses[i], ha='center', va='center', rotation=45)

giga[1].axhline(y = 1, color = 'b', linestyle = '-')
giga[1].set_xlabel('Names of Benchmarks')
giga[1].xaxis.set_label_coords(.5, -0.11)
giga[1].set_ylabel('Misses Normalized to PLRU')
giga[1].set_xticks(x + width - 0.15, single_bench_points[2][0])
giga[1].set_title('Normalized Miss Counts for Single-Scoped Benchmarks')
giga[1].grid(True, linewidth = "0.5", axis = 'y' )
giga[1].legend(loc = 'upper right')
giga[1].set_ylim(0, ylim)

plt.show()
#-----------------------------------------------------------------------------------------------------

#-----------------PARAMS
ylim = 1
width = 0.15
x = np.arange(len(multi_bench_points[2][0]))
#-----------------------


miss_ratios = {
    'PLRU':     plru_multi_miss_ratio,
    'CLAM':     multi_ratio_points[2][0],
    'SHEL':     multi_ratio_points[2][1],
    'C-SHEL':   multi_ratio_points[2][2],
    'PRL':      multi_ratio_points[2][3],
}


giga = gen_graph(miss_ratios, width, ylim, x)



#giga[1].axhline(y = 1, color = 'b', linestyle = '-')
giga[1].set_xlabel('Names of Benchmarks')
giga[1].set_ylabel('Miss Ratios [Misses/(Misses + Hits)]')
giga[1].set_xticks(x + width + 0.15, multi_bench_points[2][0])
giga[1].set_title('Miss Ratios for Multi-Scoped Benchmarks')
giga[1].grid(True, linewidth = "0.5", axis = 'y' )
giga[1].legend(loc = 'upper right')
giga[1].set_ylim(0, ylim)

plt.show()
#-----------------------------------------------------------------------------------------------------------------------------------------------


#-----------------PARAMS
ylim = 1
width =0.2
x = np.arange(len(single_bench_points[2][0]))
#-----------------------


miss_ratios = {
    'PLRU':     plru_single_miss_ratio,
    'CLAM':     single_ratio_points[2][0],
    #'SHEL':     multi_ratio_points[2][1],
    #'C-SHEL':   multi_ratio_points[2][2],
    'PRL':      single_ratio_points[2][3],
}


giga = gen_graph(miss_ratios, width, ylim, x)



#giga[1].axhline(y = 1, color = 'b', linestyle = '-')
giga[1].set_xlabel('Names of Benchmarks')
giga[1].set_ylabel('Miss Ratios [Misses/(Misses + Hits)]')
giga[1].set_xticks(x + width, single_bench_points[2][0])
giga[1].set_title('Miss Ratios for Single-Scoped Benchmarks')
giga[1].grid(True, linewidth = "0.5", axis = 'y' )
giga[1].legend(loc = 'upper right')
giga[1].set_ylim(0, ylim)

plt.show()

#^CLAM, SHEL, C-SHEL, PRL, Miss Ratio

#^------------------------------------------------------------------------------------------------------------------------------------PLOT DATA      



