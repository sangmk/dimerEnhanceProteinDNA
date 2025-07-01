import sys
import multiprocessing
import pandas as pd
import numpy as np
import argparse

# Create the parser
parser = argparse.ArgumentParser()

# Add an argument for the option
parser.add_argument("-f", "--file", help="input files", required=False, default='NONE')

# Parse the arguments
args = parser.parse_args()
if args.file == 'NONE':
    pdir = input("Enter the directory to parameters and initial conditions: ") + '/'
    numClusters = int(input("Enter the number of targets clustered: "))
    maxT = input("Enter the maximum time to run the simulation (default auto): ")
    if maxT.lower().strip() in ['auto', '']:
        maxT = 'auto'
    else:
        maxT = float(maxT)
    tStart = input("Enter the start time for calculating survival probability (default 0): ")
    if tStart.strip() == '':
        tStart = 0
    else:
        tStart = float(tStart)
    tEnd = input("Enter the end time for calculating survival probability (default inf): ")
    if tEnd.strip() == '':
        tEnd = np.inf
    else:
        tEnd = float(tEnd)
    coresSpecified = input("Enter the number of cores to use: (skip for all available cores)")
else:
    with open(args.file, 'r') as f:
        for line in f:
            if line[0] == '#':
                continue
            linelist = line.split(',')
            pdir = linelist[0] + '/'
            numClusters = int(linelist[1])
            maxT = linelist[2]
            if maxT.lower().strip() in ['auto', '']:
                maxT = 'auto'
            else:
                maxT = float(maxT)
            tStart = linelist[3]
            if tStart.strip() == '':
                tStart = 0
            else:
                tStart = float(tStart)
            tEnd = linelist[4]
            if tEnd.strip() == '':
                tEnd = np.inf
            else:
                tEnd = float(tEnd)
            coresSpecified = linelist[5]


# Check if the user has specified the number of cores to use
useUserDefinedCores = False
if coresSpecified.strip() != '': 
    useUserDefinedCores = True
    coresSpecified = int(coresSpecified) # number of cores to use

print('pdir:', pdir)
N_repeat = 1
# totally 200 proteins in the system
NP0_sys = 200
# time parameters
rMaxT = 1e3
rMinRatio = np.inf
# Read parameters from CSV file
parameters = []
equi_file = pd.read_csv(pdir+'equilibrium.csv')
parm_file = pd.read_csv(pdir+'parameters.csv')
for iloc in range(parm_file.shape[0]):
    for i in range(N_repeat):
        parameters.append((
            equi_file.iloc[iloc], parm_file.iloc[iloc], i, 
            rMaxT, rMinRatio, NP0_sys, pdir, maxT, tStart, tEnd,
        ))
print('Total jobs:', len(parameters))
# Run tasks in parallel
num_processes = multiprocessing.cpu_count()  # Get the number of available CPU cores
if useUserDefinedCores:
    cores_used = coresSpecified
else:
    cores_used = min(len(parameters), num_processes-2)

print('Number of cores used: ',f'{cores_used}/{num_processes}')

if numClusters == 0:
     from Nonly import main_Gillespie
elif numClusters == 1:
     from singleS import main_Gillespie
elif numClusters == 2:
     from doubleS import main_Gillespie

pool = multiprocessing.Pool(processes=cores_used)
pool.map(main_Gillespie, parameters)
pool.close()
pool.join()