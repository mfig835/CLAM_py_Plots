import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from utils import *
import csv



def find_index_of_corresponding(item, array, array_len): #Function for finding the corresponding index of a match item of an array
    i = 0
    while i < array_len:
        if item == array[i]:
            return(i)
        i += 1


def gen_graph(data_dict, bar_width, ylim, x):

    giga = fig1, ax2 = plt.subplots()
    multiplier = 0


    for attribute, measurement in data_dict.items():
        offset = (bar_width)* multiplier
        rects = ax2.bar(x + offset, measurement, width=bar_width, label=attribute, edgecolor = 'black')
        #ax2.bar_label(rects, label_type='center' ,padding=0)
        #ax2.set_xticklabels(attribute, rotation = 90)

        multiplier += 1
        for bar in rects:
            yval=bar.get_height()
            if(yval > ylim):
                ax2.text(bar.get_x() + bar.get_width()/2, ylim/2, round(yval, 3), rotation = 90, ha='center', va = 'bottom')  

    return(giga)

def spec_plot(policy, size, benchmark, is_multilevel):
    #PREPROCESSING------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if is_multilevel == 1:
        data_path = 'data/track/{policy}_{size}_multi_level/{benchmark}.csv'.format(policy=policy, size=size, benchmark=benchmark)
        plot_path = 'plots/spectra_tracking/{policy}_{size}_multi_level'.format(policy=policy, size=size)
    else:
        data_path = 'data/track/{policy}_{size}/{benchmark}.csv'.format(policy=policy, size=size, benchmark=benchmark)
        plot_path = 'plots/spectra_tracking/{policy}_{size}'.format(policy=policy, size=size)

    #PACK DATA FOR PLOTTING---------------------------------------------------------------------------------------------------------------------------------------------------------------
    with open(data_path, newline='') as csvfile:     #Open spectrum results file 
        data = csv.reader(csvfile, delimiter=',', quotechar='|')                        #Use python csv package to store spectrum values in data (2d array of strings)
        row_count = sum(1 for j in data)                                                #Count number of rows (access samples) in csv file

    spectrum_vals = [[0 for _ in range(512)] for i in range(row_count)]                 #Create a data frame in which spectrum values will be placed (512 x num access samples)                                


    with open(data_path, newline='') as csvfile:     #Open spectrum results file 
        data = csv.reader(csvfile, delimiter=',', quotechar='|')                        #Use python csv package to store spectrum values in data (2d array of strings)
        
        row_ind = 0                                                                     #Holds index for rows
        for row in data:        
            for lease_length in range(3):                                               #iterate through 3 possible lease lenths (Short, medium, long)
                for i in range(16):                                                     #iterate through all 16 32 bit hex entries in each "length section" of the row
                    hex_integer = int((row[lease_length * 16 + i].strip()), 16)         #find integer value of hex string. also strips leading and trailing excess whitespace from string
                    bin_string = bin(hex_integer)                                       #Convert int to binary
                    bin_string = bin_string[2:]                                         #remove '0b' prefix
                    bin_string = bin_string.zfill(32)                                   #make sure string is 32 bits

                    for x in range(32):                                                 #iterate through all 32 bits of word
                        if bin_string[x] == '1':                                        #if this binary digit is high, give it the corresponding length assignment in the data frame
                            spectrum_vals[row_ind][i*32 + x] = lease_length + 1         #if the cache line is assigned a 0 for that sample, it is expired (no lease assigned).
                                                                                        #1 is short, 2 is medium, and 3 is long
            row_ind += 1                                                                #increment row index


    #PLOT DATA----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    #for sample in spectrum_vals:                                                                          #Iterate through all members of populated data structure
    #    for line in sample:
            
    num_accesses = row_count * 256                                                                         #find approx number of accesses
    millions_accesses = num_accesses/1000000                                                               #value in millions

    xaxis = np.arange(0, millions_accesses, millions_accesses/row_count)                                   #find values along x-axis


    plt.style.use('_mpl-gallery-nogrid')                                                                   #Spectrum graph

    c = plt.imshow(np.transpose(spectrum_vals), aspect = 'auto', origin = 'lower', vmin =0, vmax = 3, extent = [xaxis[0], xaxis[-1], 0, 512])      #Plot "spectrum"

    plt.ylabel('Cache Line Number')
    plt.xlabel('Millions of Accesses')
    plt.title('Cache Tenancy Spectrum')
    cbar = plt.colorbar(orientation = 'horizontal', aspect = 50)                                           #find colorbar
    cbar.set_label('Lease Length')
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(['Expired', 'Short', 'Medium', 'Long'])

    plt.subplots_adjust(left = 0.05, right=0.95, bottom=0.05, top = 0.95)                                  #adjust position of graph to make it more visib
    plt.gcf().set_size_inches(7,4.5)                                                                       #Set size and shape of plot to be normal(width, height)
    plt.tight_layout()






    if Path(plot_path).exists() == 0:                                                                       #If path doesn't exist, make it
        os.makedirs(plot_path)
    plt.savefig('{plot_path}/{benchmark}.png'.format(plot_path=plot_path, benchmark=benchmark), dpi = 150)  #higher dpi gives a better resolution

    cbar.remove()
    #plt.show()