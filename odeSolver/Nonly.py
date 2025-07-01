import numpy as np
# from sympy import nsolve, symbols
from scipy.integrate import solve_ivp
import sys
from datetime import datetime

# perform ODE solver based on the reactions used for Gillespie input
def ODE_function(current_counts, t, parms, reactions):
    propList, rxnMatrix = reactions(current_counts, parms)
    return np.matmul(rxnMatrix.T, propList)

# input parameters:
#    N, P, PP, PN, PPN, PNPN
#    kaP, kbP, kaN, kbN, gamma, V
def reversible_dimer_Nonly(current_counts, kineticParms):
    # get parameters and currernt state
    kaP, kbP, kaN, kbN, gamma, V = kineticParms
    N, P, PP, PN, PPN, PNPN = np.array(current_counts)
    
    C0 = 0.6022
    
    # 3D reaction (1) ---------------------
    # P + P <-> PP
    a_P_P_PP = kaP*P*P/V    # if P == 1: a_P_P_PP = 0 # no dimerization if there is only one P left
    a_PP_P_P = kbP*PP
    # 3D and 1D transitions (3) ---------------------
    # P + N <-> PN
    a_P_N_PN = kaN*P*N/V
    a_PN_P_N = kbN*PN
    # P + PN <-> PPN
    a_P_PN_PPN = 2*kaP*P*PN/V
    a_PPN_P_PN = kbP*PPN
    # PP + N <-> PPN
    a_PP_N_PPN = 2*kaN*PP*N/V
    a_PPN_PP_N = kbN*PPN
    # 1D dimerizations (2) ----------------------
    # PN + PN <-> PNPN (gamma)
    a_PN_PN_PNPN = gamma*kaP*PN*PN/V    # if PN == 1: a_PN_PN_PNPN = 0
    a_PNPN_PN_PN = kbP*PNPN
    # PPN + N <-> PNPN (gamma)
    a_PPN_N_PNPN = gamma*kaN*PPN*N/V
    a_PNPN_PPN_N = 2*kbN*PNPN
    
    
    propensities = np.array(
        [
            a_P_P_PP, a_PP_P_P, 
            a_P_N_PN, a_PN_P_N, 
            a_P_PN_PPN, a_PPN_P_PN,
            a_PP_N_PPN, a_PPN_PP_N, 
            a_PN_PN_PNPN, a_PNPN_PN_PN, 
            a_PPN_N_PNPN, a_PNPN_PPN_N,
        ]
    )
    
    rxnMatrix = np.array(
        [
            # N,  P, PP, PN,PPN,PNPN,
            [ 0, -2,  1,  0,  0,   0,], # P + P -> PP
            [ 0,  2, -1,  0,  0,   0,], # PP -> P + P
            [-1, -1,  0,  1,  0,   0,], # P + N -> PN
            [ 1,  1,  0, -1,  0,   0,], # PN -> P + N
            [ 0, -1,  0, -1,  1,   0,], # P + PN -> PPN
            [ 0,  1,  0,  1, -1,   0,], # PPN -> P + PN
            [-1,  0, -1,  0,  1,   0,], # PP + N -> PPN
            [ 1,  0,  1,  0, -1,   0,], # PPN -> PP + N
            [ 0,  0,  0, -2,  0,   1,], # PN + PN -> PNPN
            [ 0,  0,  0,  2,  0,  -1,], # PNPN -> PN + PN
            [-1,  0,  0,  0, -1,   1,], # PPN + N -> PNPN
            [ 1,  0,  0,  0,  1,  -1,], # PNPN -> PPN + N
        ]
    )
    
    return propensities, rxnMatrix


# compare observed equilibrium constants with true values
def keq_deviation(EQUILIBRIUM, equiParms, tol):

    def zeroDivid(equi, rxn):
        '''To capture zero division error and irreversible reactions'''
        for i, ai in enumerate(rxn):
            if ai < 0 and equi[i]==0:
                return True
        return False
    
    # get parameters
    KPP, KPN, gamma, V = equiParms

    C0 = 0.6022

    # define the reaction matrix
    rxnMatrix = np.array(
        [
            [ 0, -2,  1,  0,  0,   0,], # P + P -> PP
            [-1, -1,  0,  1,  0,   0,], # P + N -> PN
            [ 0, -1,  0, -1,  1,   0,], # P + PN -> PPN
            [-1,  0, -1,  0,  1,   0,], # PP + N -> PPN
            [ 0,  0,  0, -2,  0,   1,], # PN + PN -> PNPN
            [-1,  0,  0,  0, -1,   1,], # PPN + N -> PNPN
        ]
    )
    # calculate true equilibrium constants
    KeqList = np.array(
        [
            KPP, 
            KPN, 
            2*KPP,
            2*KPN, 
            gamma*KPP, 
            gamma*KPN/2,
        ]
    )
    # observed equilibrium constants
    for i, rxn in enumerate(rxnMatrix):
        if KeqList[i] is np.nan or KeqList[i] == 0 or KeqList[i] == np.inf:
            pass
        elif zeroDivid(EQUILIBRIUM, rxn):
            pass
        elif abs(np.prod((EQUILIBRIUM/V)**rxn) / KeqList[i] - 1) > tol:
            return True
    return False

# determine whether fluctuation is too large
def flut_toolarge(solution, tol):
    for i, soli in enumerate(solution):
        if soli[-1] == 0:
            if np.std(soli[-10:]) > tol:
                return True
        elif np.std(soli[-10:])/soli[-1] > tol:
            return True
    return False

# ----------------------------------------
# ----------------------------------------
# The main function

labels = 'ID, N, P, PP, PN, PPN, PNPN'.split(', ')

def rxnNetwork(parm):

    # print('ID:',parm['ID'],flush=True)

    ## the initial concentrations
    CP0:float = parm['CP0'] # protein
    CN0:float = parm['CN0'] # nonspecific sites
    ## equilibrium constants
    KPN:float = parm['KPN'] # P-N
    KPP:float = parm['KPP'] # P-P
    ## dimension factor
    gamma:float = parm['gamma']
    # calculate input for Gillespie 
    ## default system parameters
    NP0_sys = 1e6
    V = NP0_sys / CP0

    # ------------- set up solver --------------
    #    N, P, PP, PN, PPN, PNPN
    if KPP == np.inf:
        ini_count = np.array(
            [
                round(NP0_sys*CN0/CP0), 0, 
                NP0_sys/2, 0, 0, 0,
            ]
        )
    else:
        ini_count = np.array(
            [
                round(NP0_sys*CN0/CP0), NP0_sys, 
                0, 0, 0, 0,
            ]
        )
    # S, N, P
    totalCopy = [round(NP0_sys*CN0/CP0), NP0_sys]
    # define how to calculate the total number of monomers
    # N, P, PP, PN, PPN, PNPN
    totalSum = np.array([
        [
            1, 0, 0, 1, 1, 2 
        ], # N
        [
            0, 1, 2, 1, 2, 2, 
        ] # P
    ])

    ## reaction parameters
    kaN = 10
    kbN = kaN/KPN
    if KPP == 0:
        kaP = 0
        kbP = 0
    else:
        kaP = 10
        kbP = kaP/KPP
    # kaP, kbP, kaN, kbN, gamma, V
    kineticParms = [kaP, kbP, kaN, kbN, gamma, V]
    # KPP, KPN, gamma, V
    equiParms = [KPP, KPN, gamma, V]
    
    # define a function which equals zero at equilibrium
    flow_func = lambda t, conc: ODE_function(conc, 0, kineticParms, reversible_dimer_Nonly)
    # define a function for determining if observed Keq's have deviations
    keq_dev = lambda count, tol: keq_deviation(count, equiParms, tol)
    # --------------------------------------------------
    # --------------------------------------------------

    # run simulation
    ntrys = 0
    # Time interval for integration
    t_span = (0, 1e9)
    y0 = ini_count
    n_limit = 10
    while ntrys < n_limit:
        dynamics = solve_ivp(
            flow_func, t_span, y0, 
            t_eval=np.logspace(0, np.log10(t_span[1]), 1000), 
            method='LSODA'
        )
        solution = dynamics.y
        # determine whether the system is in equilibrium
        # time fluctuation v.s. last value
        # compare observed Keq's v.s. true values
        if keq_dev(solution[:,-1], 1e-3) or flut_toolarge(solution, 1e-3):
            t_span = [0, t_span[-1]*10]
            ntrys += 1
        else:
            # output
            if any((np.sum(solution[:,-1].flatten()*totalSum, axis=1) - totalCopy)/totalCopy > 1e-3):
                print('# Mass conservation fails!',flush=True)
            return [parm['ID']]+[value for value in solution[:,-1]]
        
    print('Failed. ID:',parm['ID'],flush=True)
    # print('finished ID:',parm['ID'],datetime.now(),flush=True)