import numpy as np
import threading

# tau=f(x)
# x = (KPN, kaP, kbP, CP0, gamma)
def f_DNA(x):
    '''Given x, calculate the mean residence time through two-state equilbrium model
    
    Inputs:
        x (list): (KPN, kaP, kbP, CP0, gamma) Note that, kbN=120.44/KPN s^-1
        
    Outputs;
        mean residence times (float)
    '''
    
    import sys
    sys.path.append('/home/local/WIN/msang2/mankun/Notebooks/[01]dimerEnhanceProteinDNA/')
    # modules for direct calculation
    import analytics as ana
    import analyzeEqui as num
    # stochastic simulation (Gillespie)
    from Simulations.odeSolver.Nonly import rxnNetwork, labels
    from Simulations.odeSolver.main import main as numericSolver
    # for generating parameters
    from GenerateParameters import GenParameters
    ######## finish importing ########
    KPN, kaP, kbP, CP0, gamma = np.array(x)
    if kbP == 0:
        parms = GenParameters(
            ifwrite=False, hasTargets = False,
            KPS=lambda KPN: 1e3*KPN, 
            CP0=[CP0], KPP=[np.inf], kbPP_fixed=kbP,
            gamma=[gamma], KPN=[KPN]
        )
    else:
        parms = GenParameters(
            ifwrite=False, hasTargets = False,
            KPS=lambda KPN: 1e3*KPN, 
            CP0=[CP0], KPP=[kaP/kbP], kbPP_fixed=kbP,
            gamma=[gamma], KPN=[KPN]
        )
    # there should be only one set of parameters
    equi = numericSolver(parm_df=parms, labels=labels, model=rxnNetwork, ifprint=False)
    # return the mean residence time
    return num.calc_resT_modelA(parms.iloc[0], equi.iloc[0])

# sensitivity analysis
def sobolSA(problem, numberSamples, model, num_cores=1):
    
    from SALib.analyze import sobol
    from SALib.sample import sobol as sobol_sampling
    from tqdm import tqdm 
    import multiprocessing
    
    # Generate parameter samples using Saltelli's method (for Sobol analysis)
    print(threading.current_thread().daemon)
    param_values = sobol_sampling.sample(problem, numberSamples)
    print(threading.current_thread().daemon)
    num_cores = min(num_cores, multiprocessing.cpu_count())  # Get the number of available CPU cores
    if num_cores > 1:
        print(threading.current_thread().daemon)
        pool = multiprocessing.Pool(processes=num_cores)
        print(threading.current_thread().daemon)
        Y = np.array(pool.map(model, param_values))
    else:
        # Run the model for each set of parameters
        Y = np.array([model(x) for x in tqdm(param_values, desc="Evaluating model", unit="sample")])
    
    # Perform Sobol sensitivity analysis and return 
    return sobol.analyze(problem, Y)

# Define the problem with bounds for each parameter (support A)
problem = {
    'num_vars': 5,
    'names': ['KPN', 'kaP', 'kbP', 'CP0', 'gamma'],
    'bounds': [
        ([2, 2000]), # KPN
        10.0**np.array([2, 7]), # kaP
        10.0**np.array([-3, 1]), # kbP
        10.0**np.array([-2-6, 2-6]), # CP0
        10.0**np.array([0, 3]), # gamma
    ] 
}

# Perform Sobol sensitivity analysis
print(threading.current_thread().daemon)
Si = sobolSA(problem, 2**12, f_DNA, num_cores=32)

# Print first-order and total-order Sobol indices
print("First-order Sobol indices:", Si['S1'])
print("Total-order Sobol indices:", Si['ST'])